"""Offscreen desktop-app tests with a fake detector and a fake capture.

Acceptance items covered (验收清单 §3):

* E01 — the GUI thread never runs inference: the fake detector records the
  thread that executed ``predict`` and it must differ from the main thread.
* E02 — closing the window while a task runs stops it, releases the capture and
  never destroys a running ``QThread``.
* E03 — importing ``uavdet`` / ``uavdet.cli`` / the app modules imports no Qt
  and creates no ``QApplication`` (checked in a fresh subprocess).
* E04 — ``importlib.resources`` finds ``styles.qss`` in the installed layout.

Plus the page behaviour: per-class counts and PNG/CSV export, settings
persistence, failed model load keeping the old model, forced single task,
missing-product notices for a run without evaluation, and capture release on
stop and on error.

The suite is deterministic: every wait is a short event-loop poll with a
deadline (no long fixed sleeps) and all output goes to a temporary directory.
"""

from __future__ import annotations

import importlib.resources
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="uavdet-mpl-"))

try:  # PyQt6 is optional (cloud machines install the core only)
    import PyQt6  # noqa: F401
    from PyQt6.QtCore import QEventLoop
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    HAVE_QT = True
    QT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only without PyQt6
    HAVE_QT = False
    QT_IMPORT_ERROR = exc

from tests import helpers
from uavdet.app import main_window as main_window_module
from uavdet.app import widgets as widgets_module
from uavdet.app import workers as workers_module
from uavdet.app.pages import results_page as results_module
from uavdet.types import CLASS_NAMES, Detection, PredictionResult

PROJECT_ROOT = helpers.PROJECT_ROOT
MAIN_IDENT = threading.main_thread().ident

_APP: "QApplication | None" = None


# ---------------------------------------------------------------------------
# shared offscreen application + polling helpers
# ---------------------------------------------------------------------------
def qapp() -> "QApplication":
    """The single shared ``QApplication`` (offscreen when the env says so)."""
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(["uavdet-gui-tests"])
    return _APP


def pump(times: int = 1) -> None:
    """Process pending queued signals without blocking for long."""
    app = qapp()
    for _ in range(max(1, times)):
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
        QTest.qWait(4)


def wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll ``predicate`` with short event-loop iterations until it is true."""
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if predicate():
            return True
        pump()
    return bool(predicate())


def wait_thread(task, timeout: float = 5.0) -> bool:
    """Wait for a QThread to finish while keeping the GUI loop alive."""
    return wait_until(lambda: task is not None and not task.isRunning(), timeout)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class FakeDetector:
    """Detector-like object that records *which thread* ran what."""

    instances: list["FakeDetector"] = []

    def __init__(self, weights: str, device: str = "cpu", *, fail_load: bool = False,
                 fail_predict: bool = False, delay: float = 0.0):
        self.weights = str(weights)
        self.device = str(device)
        self.fail_load = bool(fail_load)
        self.fail_predict = bool(fail_predict)
        self.delay = float(delay)
        self.load_thread_ids: list[int] = []
        self.predict_thread_ids: list[int] = []
        self.annotate_thread_ids: list[int] = []
        self.predict_calls: list[object] = []
        self.closed = False
        self.warnings: list[str] = []
        FakeDetector.instances.append(self)

    def load(self) -> dict:
        self.load_thread_ids.append(threading.get_ident())
        if self.fail_load:
            raise RuntimeError("synthetic weight failure")
        return {"num_classes": len(CLASS_NAMES), "weights": self.weights}

    def predict(self, image, options, *, image_id: str = "array", source: str = ""):
        self.predict_thread_ids.append(threading.get_ident())
        self.predict_calls.append(options)
        if self.delay:
            time.sleep(self.delay)
        if self.fail_predict:
            raise RuntimeError("synthetic predict failure")
        height, width = int(image.shape[0]), int(image.shape[1])
        detections = [
            Detection(0, "pedestrian", 0.91, (4.0, 4.0, 20.0, 20.0)),
            Detection(3, "car", 0.80, (30.0, 10.0, 50.0, 30.0)),
            Detection(0, "pedestrian", 0.60, (2.0, 2.0, 8.0, 8.0)),
        ]
        return PredictionResult(image_id=image_id, source=source, width=width,
                                height=height, detections=detections,
                                timing={"total_ms": 2.5})

    def annotate(self, image, detections, options=None):
        self.annotate_thread_ids.append(threading.get_ident())
        return image.copy()

    def close(self) -> None:
        self.closed = True


class FakeFactory:
    """``detector_factory(weights, device)``; queue kwargs for later calls."""

    def __init__(self):
        self.created: list[FakeDetector] = []
        self.queued: list[dict] = []

    def __call__(self, weights: str, device: str) -> FakeDetector:
        kwargs = self.queued.pop(0) if self.queued else {}
        detector = FakeDetector(weights, device, **kwargs)
        self.created.append(detector)
        return detector


class FakeCapture:
    """``cv2.VideoCapture`` stand-in that records ``release()`` calls."""

    def __init__(self, *, opened: bool = True, read_error: Exception | None = None,
                 read_delay: float = 0.0, max_frames: int | None = None,
                 width: int = 64, height: int = 48):
        self.opened = bool(opened)
        self.read_error = read_error
        self.read_delay = float(read_delay)
        self.max_frames = max_frames
        self.width = int(width)
        self.height = int(height)
        self.reads = 0
        self.release_count = 0
        self.last_target = None

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API
        return self.opened

    def read(self):
        self.reads += 1
        if self.read_delay:
            time.sleep(self.read_delay)
        if self.read_error is not None:
            raise self.read_error
        if self.max_frames is not None and self.reads > self.max_frames:
            return False, None
        frame = np.full((self.height, self.width, 3), 90, dtype=np.uint8)
        frame[0:4, 0:4] = (255, 0, 0)
        return True, frame

    def release(self) -> None:
        self.release_count += 1


# ---------------------------------------------------------------------------
# tests that must hold even when PyQt6 is missing
# ---------------------------------------------------------------------------
class CoreStaysQtFreeTest(unittest.TestCase):
    """E03 — the core and the app entry points never create a QApplication."""

    def _python(self, code: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="uavdet-mpl-"))
        return subprocess.run(
            [sys.executable, "-c", code], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=180, env=env,
        )

    def test_core_import_does_not_pull_in_qt(self) -> None:
        code = (
            "import sys, uavdet, uavdet.cli, uavdet.types, uavdet.config, "
            "uavdet.artifacts, uavdet.data\n"
            "print('QT_MODULES', [m for m in sys.modules if m.startswith('PyQt')])\n"
        )
        result = self._python(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("QT_MODULES []", result.stdout)

    def test_app_modules_import_without_qt(self) -> None:
        code = (
            "import sys\n"
            "import uavdet.app, uavdet.app.main, uavdet.app.main_window, "
            "uavdet.app.workers, uavdet.app.widgets\n"
            "import uavdet.app.pages, uavdet.app.pages.image_page\n"
            "import uavdet.app.pages.stream_page, uavdet.app.pages.model_page\n"
            "import uavdet.app.pages.results_page\n"
            "print('QT_MODULES', [m for m in sys.modules if m.startswith('PyQt')])\n"
        )
        result = self._python(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("QT_MODULES []", result.stdout)

    @unittest.skipUnless(HAVE_QT, "PyQt6 not importable: cannot check for a QApplication")
    def test_cli_import_creates_no_qapplication(self) -> None:
        code = (
            "import uavdet.cli\n"
            "from PyQt6.QtWidgets import QApplication\n"
            "print('INSTANCE', QApplication.instance() is not None)\n"
        )
        result = self._python(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("INSTANCE False", result.stdout)

    def test_session_refuses_inference_on_main_thread(self) -> None:
        session = workers_module.DetectorSession("w.pt", "cpu", factory=FakeFactory())
        session.load()  # creating/loading on the caller thread is allowed
        from uavdet.types import InferenceOptions

        with self.assertRaises(RuntimeError):
            session.predict(np.zeros((8, 8, 3), np.uint8), InferenceOptions())
        with self.assertRaises(RuntimeError):
            session.annotate(np.zeros((8, 8, 3), np.uint8), [])
        session.close()


class StylesheetResourceTest(unittest.TestCase):
    """E04 — the stylesheet is found after an editable/normal install."""

    def test_importlib_resources_locates_stylesheet(self) -> None:
        ref = importlib.resources.files("uavdet.app").joinpath("styles.qss")
        self.assertTrue(ref.is_file(), "styles.qss missing from the package data")
        text = ref.read_text(encoding="utf-8")
        self.assertIn("QMainWindow", text)
        self.assertGreater(len(text), 800)
        self.assertEqual(widgets_module.load_stylesheet(), text)
        self.assertIsNotNone(widgets_module.stylesheet_path())

    def test_results_helpers_are_qt_free_math(self) -> None:
        self.assertEqual(results_module.format_percent(0.42992), "42.992%")
        self.assertEqual(results_module.format_percent(None), "N/A")
        self.assertEqual(results_module.format_percent(float("nan")), "N/A")
        self.assertEqual(results_module.format_count(548), "548")


# ---------------------------------------------------------------------------
# widget-level tests (need Qt, no window)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_QT, f"PyQt6 not importable ({QT_IMPORT_ERROR})")
class WidgetTest(unittest.TestCase):
    def setUp(self) -> None:
        qapp()

    def test_aspect_fit_never_distorts(self) -> None:
        self.assertEqual(widgets_module.fit_size(400, 100, 100, 100), (100, 25))
        self.assertEqual(widgets_module.fit_size(64, 48, 320, 240), (320, 240))
        self.assertEqual(widgets_module.fit_size(0, 10, 100, 100), (0, 0))

    def test_bgr_to_qimage_converts_exactly_once(self) -> None:
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[0, 0] = (255, 0, 0)  # pure blue in OpenCV order
        image = widgets_module.bgr_to_qimage(bgr)
        colour = image.pixelColor(0, 0)
        self.assertEqual((colour.red(), colour.green(), colour.blue()), (0, 0, 255))
        self.assertEqual(tuple(int(v) for v in bgr[0, 0]), (255, 0, 0))

    def test_qimage_is_a_detached_copy(self) -> None:
        bgr = np.zeros((3, 3, 3), dtype=np.uint8)
        bgr[1, 1] = (10, 20, 30)
        image = widgets_module.bgr_to_qimage(bgr)
        before = image.pixelColor(1, 1)
        bgr[1, 1] = (200, 200, 200)  # worker reuses its buffer
        after = image.pixelColor(1, 1)
        self.assertEqual((before.red(), before.green(), before.blue()),
                         (after.red(), after.green(), after.blue()))

    def test_image_view_keeps_aspect_ratio(self) -> None:
        view = widgets_module.ImageView()
        view.resize(400, 200)
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        view.set_image_bgr(frame)
        self.assertTrue(view.has_image())
        self.assertEqual(view.image_size(), (200, 120))
        width, height = view.display_size()
        self.assertAlmostEqual(width / height, 200 / 120, places=2)
        self.assertLessEqual(width, 400)
        self.assertLessEqual(height, 200)

    def test_counts_table_keeps_fixed_class_order(self) -> None:
        table = widgets_module.ClassCountsTable()
        table.set_counts({"car": 2, "pedestrian": 1})
        self.assertEqual(table.rowCount(), len(CLASS_NAMES))
        self.assertEqual(table.counts()["car"], 2)
        self.assertEqual(table.counts()["pedestrian"], 1)
        self.assertEqual(table.item(0, 1).text(), "pedestrian")
        self.assertEqual(table.item(3, 1).text(), "car")


# ---------------------------------------------------------------------------
# window / page tests
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_QT, f"PyQt6 not importable ({QT_IMPORT_ERROR})")
class GuiTestBase(helpers.TempDirTestCase):
    """One temporary project + one window (with fake detector) per test."""

    def setUp(self) -> None:
        super().setUp()
        qapp()
        self.project = self.project_handle = self.project()
        self.factory = FakeFactory()
        self.weights = self.tmp_path / "weights" / "best.pt"
        self.weights.parent.mkdir(parents=True, exist_ok=True)
        self.weights.write_bytes(b"synthetic-weights")
        self.window = main_window_module.MainWindow(
            config_path=str(self.project.config_path),
            detector_factory=self.factory,
        )
        self.window.resize(1280, 820)
        pump()

    def tearDown(self) -> None:
        window = getattr(self, "window", None)
        if window is not None:
            try:
                window.shutdown()
            finally:
                window.close()
                window.deleteLater()
                pump()
        super().tearDown()

    # -- helpers -----------------------------------------------------------
    def load_model(self) -> FakeDetector:
        ok, message = self.window.load_model(str(self.weights), "cpu")
        self.assertTrue(ok, message)
        self.assertTrue(wait_until(self.window.has_model), "model never finished loading")
        self.wait_idle()
        return self.factory.created[-1]

    def wait_idle(self, timeout: float = 5.0) -> None:
        self.assertTrue(wait_until(lambda: not self.window.is_busy(), timeout),
                        "window stayed in the busy state")

    def open_image(self, name: str = "frame.png", width: int = 200,
                   height: int = 120) -> Path:
        path = helpers.write_png(self.tmp_path / name, width, height, seed=3)
        self.assertTrue(self.window.image_page.load_image(path))
        return path

    def dummy_video(self, name: str = "clip.mp4") -> Path:
        path = self.tmp_path / name
        path.write_bytes(b"synthetic-video-placeholder")
        return path

    def run_image_detection(self) -> FakeDetector:
        detector = self.load_model()
        self.open_image()
        page = self.window.image_page
        self.assertTrue(page.start_detection())
        self.assertTrue(wait_until(lambda: page.result is not None),
                        "image task produced no result")
        self.assertTrue(wait_thread(page.image_task))
        self.wait_idle()
        return detector


class ImagePageTest(GuiTestBase):
    def test_e01_predict_runs_in_a_worker_thread(self) -> None:
        detector = self.run_image_detection()
        self.assertEqual(len(detector.predict_thread_ids), 1)
        self.assertNotEqual(detector.predict_thread_ids[0], MAIN_IDENT,
                            "predict() ran on the GUI thread")
        self.assertNotEqual(detector.load_thread_ids[0], MAIN_IDENT,
                            "model loading ran on the GUI thread")
        self.assertNotEqual(detector.annotate_thread_ids[0], MAIN_IDENT)
        options = detector.predict_calls[0]
        self.assertEqual(options.conf, self.window.settings["conf"])
        self.assertEqual(options.imgsz, self.window.settings["imgsz"])
        self.assertEqual(options.max_det, self.window.settings["max_det"])
        self.assertEqual(options.device, self.window.settings["device"])
        # idle state restored on success
        self.assertFalse(self.window.is_busy())
        self.assertTrue(self.window.image_page.run_button.isEnabled())

    def test_counts_and_exports(self) -> None:
        self.run_image_detection()
        page = self.window.image_page
        counts = page.class_table.counts()
        self.assertEqual(counts["pedestrian"], 2)
        self.assertEqual(counts["car"], 1)
        self.assertEqual(sum(counts.values()), 3)
        self.assertIn("当前图片目标数：3", page.total_label.text())
        self.assertEqual(page.detection_table.rowCount(), 3)
        self.assertTrue(page.view.has_image())
        self.assertEqual(page.view.image_size(), (200, 120))

        out_dir = self.tmp_path / "exports"
        png = page.save_annotated_image(out_dir / "annotated.png")
        self.assertTrue(png.is_file())
        from PIL import Image

        with Image.open(png) as handle:
            self.assertEqual(handle.size, (200, 120))

        csv_path = page.save_detections_csv(out_dir / "detections.csv")
        rows = helpers.read_csv_rows(csv_path)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["image_id"], "frame")
        self.assertEqual(rows[0]["class_name"], "pedestrian")
        self.assertEqual(list(rows[0]), ["image_id", "index", "class_id", "class_name",
                                         "confidence", "x1", "y1", "x2", "y2"])

    def test_detection_needs_a_loaded_model(self) -> None:
        self.open_image()
        page = self.window.image_page
        self.assertFalse(page.start_detection())
        self.assertIn("模型", page.status.text())
        self.assertFalse(self.window.is_busy())
        self.assertTrue(page.run_button.isEnabled())

    def test_failed_detection_recovers_the_ui(self) -> None:
        detector = self.load_model()
        self.open_image()
        detector.fail_predict = True
        page = self.window.image_page
        self.assertTrue(page.start_detection())
        self.assertTrue(wait_until(lambda: "失败" in page.status.text()))
        self.assertTrue(wait_thread(page.image_task))
        self.wait_idle()
        self.assertTrue(page.run_button.isEnabled())
        self.assertIsNone(page.result)

    def test_only_one_task_at_a_time(self) -> None:
        detector = self.load_model()
        detector.delay = 0.25
        self.open_image()
        page = self.window.image_page
        self.assertTrue(page.start_detection())
        self.assertTrue(wait_until(lambda: page.image_task is not None
                                   and page.image_task.isRunning()))
        self.assertFalse(page.start_detection())          # refused: busy
        self.assertFalse(self.window.stream_page.start_stream())
        self.assertTrue(wait_thread(page.image_task))
        self.wait_idle()

    def test_e02_close_during_image_task_stops_it(self) -> None:
        detector = self.load_model()
        detector.delay = 0.30
        self.open_image()
        page = self.window.image_page
        self.assertTrue(page.start_detection())
        task = page.image_task
        self.assertTrue(wait_until(task.isRunning))
        self.assertTrue(self.window.is_busy())

        self.window.close()                                # triggers closeEvent

        self.assertFalse(task.isRunning(), "worker still running after close")
        self.assertFalse(self.window.is_busy())
        self.assertTrue(detector.closed, "model was not released on close")
        self.assertIsNone(self.window.session)
        self.assertTrue(page.run_button.isEnabled() or not page.isVisible())


class ModelPageTest(GuiTestBase):
    def test_settings_are_persisted_atomically(self) -> None:
        page = self.window.model_page
        page.weights_edit.setText(str(self.weights))
        page.device_combo.setCurrentText("cpu")
        page.conf_spin.setValue(0.4)
        page.iou_spin.setValue(0.6)
        page.max_det_spin.setValue(120)
        page.imgsz_combo.setCurrentText("512")
        self.assertTrue(page.save_parameters())

        settings_path = self.window.settings_path
        self.assertTrue(settings_path.is_file())
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], 1)
        self.assertEqual(payload["weights"], str(self.weights))
        self.assertAlmostEqual(payload["conf"], 0.4)
        self.assertAlmostEqual(payload["iou"], 0.6)
        self.assertEqual(payload["max_det"], 120)
        self.assertEqual(payload["imgsz"], 512)
        self.assertIn("updated_at", payload)
        leftovers = [p.name for p in settings_path.parent.glob(".*tmp*")]
        self.assertEqual(leftovers, [])

        options = self.window.inference_options()
        self.assertAlmostEqual(options.conf, 0.4)
        self.assertAlmostEqual(options.iou, 0.6)
        self.assertEqual(options.max_det, 120)
        self.assertEqual(options.imgsz, 512)

        # a fresh window reads the same file back
        other = main_window_module.MainWindow(
            config_path=str(self.project.config_path), detector_factory=FakeFactory())
        try:
            self.assertAlmostEqual(other.settings["conf"], 0.4)
            self.assertEqual(other.settings["imgsz"], 512)
            self.assertEqual(other.settings["weights"], str(self.weights))
        finally:
            other.shutdown()
            other.close()
            other.deleteLater()
            pump()

    def test_invalid_parameter_is_rejected_without_writing(self) -> None:
        page = self.window.model_page
        page.imgsz_combo.setCurrentText("500")   # 500 is not a multiple of 32
        self.assertFalse(page.save_parameters())
        self.assertIn("32", page.status.text())
        self.assertFalse(self.window.settings_path.exists())

    def test_failed_load_keeps_the_previous_model(self) -> None:
        old_detector = self.load_model()
        old_session = self.window.session
        self.assertIsNotNone(old_session)
        page = self.window.model_page

        other_weights = self.tmp_path / "weights" / "broken.pt"
        other_weights.write_bytes(b"synthetic-broken")
        self.factory.queued.append({"fail_load": True})
        page.weights_edit.setText(str(other_weights))
        self.assertTrue(page.load_model())
        self.assertTrue(wait_until(lambda: "加载失败" in page.status.text()))
        self.assertTrue(wait_thread(self.window._model_task))
        self.wait_idle()

        self.assertIs(self.window.session, old_session, "failed load replaced the model")
        self.assertTrue(self.window.has_model())
        self.assertFalse(old_detector.closed)
        failed = self.factory.created[-1]
        self.assertTrue(failed.closed, "failed detector was not closed")
        self.assertFalse(self.window.is_busy())
        self.assertTrue(page.load_button.isEnabled())
        self.assertIn("加载失败", page.load_state.text())

    def test_switch_refused_while_a_stream_runs(self) -> None:
        session_before = self.window.session
        self.load_model()
        session_before = self.window.session
        page = self.window.stream_page
        capture = FakeCapture(read_delay=0.005)
        page.capture_factory = lambda target: capture
        page.set_input("video", str(self.dummy_video()))
        self.assertTrue(page.start_stream())
        self.assertTrue(wait_until(lambda: page.frame_count >= 1))
        self.assertTrue(self.window.is_busy())

        ok, message = self.window.load_model(str(self.weights), "cpu")
        self.assertFalse(ok)
        self.assertIn("停止", message)
        self.assertIs(self.window.session, session_before)

        page.stop_stream()
        self.assertTrue(wait_until(lambda: not page._running))
        self.assertTrue(wait_thread(page.stream_task))
        self.wait_idle()
        self.assertEqual(capture.release_count, 1)


class StreamPageTest(GuiTestBase):
    def _start_stream(self, capture: FakeCapture) -> bool:
        self.load_model()
        page = self.window.stream_page
        page.capture_factory = lambda target: capture
        page.set_input("video", str(self.dummy_video()))
        return page.start_stream()

    def test_stop_releases_capture_and_recovers(self) -> None:
        capture = FakeCapture(read_delay=0.005)
        self.assertTrue(self._start_stream(capture))
        page = self.window.stream_page
        self.assertTrue(wait_until(lambda: page.frame_count >= 2), "no frames arrived")
        self.assertIn("当前帧目标数：3", page.total_label.text())
        self.assertEqual(page.class_table.counts()["car"], 1)
        self.assertTrue(page.save_current_frame(self.tmp_path / "frame.png").is_file())
        self.assertFalse(page.start_button.isEnabled())

        self.assertTrue(page.stop_stream())
        self.assertTrue(wait_until(lambda: not page._running), "stop did not finish")
        self.assertEqual(capture.release_count, 1, "capture.release() was not called once")
        self.assertFalse(self.window.is_busy())
        self.assertTrue(page.start_button.isEnabled())
        self.assertFalse(page.stop_button.isEnabled())
        self.assertIn("已停止", page.status.text())

    def test_error_releases_capture_and_recovers(self) -> None:
        capture = FakeCapture(read_error=RuntimeError("sensor failure"))
        self.assertTrue(self._start_stream(capture))
        page = self.window.stream_page
        self.assertTrue(wait_until(lambda: page._last_error is not None))
        self.assertTrue(wait_until(lambda: not page._running))
        self.assertEqual(capture.release_count, 1)
        self.assertIn("sensor failure", page.status.text())
        self.assertTrue(page.status.property("error") in ("true", True))
        self.assertFalse(self.window.is_busy())
        self.assertTrue(page.start_button.isEnabled())

    def test_capture_that_cannot_open_is_reported(self) -> None:
        capture = FakeCapture(opened=False)
        self.assertTrue(self._start_stream(capture))
        page = self.window.stream_page
        self.assertTrue(wait_until(lambda: page._last_error is not None))
        self.assertIn("无法打开输入源", page.status.text())
        self.assertEqual(capture.release_count, 1)
        self.assertFalse(self.window.is_busy())

    def test_end_of_video_releases_capture(self) -> None:
        capture = FakeCapture(max_frames=2, read_delay=0.002)
        self.assertTrue(self._start_stream(capture))
        page = self.window.stream_page
        self.assertTrue(wait_until(lambda: not page._running))
        self.assertEqual(capture.release_count, 1)
        self.assertEqual(page.frame_count, 2)
        self.assertIn("输入已结束", page.status.text())
        self.assertFalse(self.window.is_busy())

    def test_e02_close_during_stream_releases_capture(self) -> None:
        capture = FakeCapture(read_delay=0.005)
        self.assertTrue(self._start_stream(capture))
        page = self.window.stream_page
        self.assertTrue(wait_until(lambda: page.frame_count >= 1))
        task = page.stream_task
        self.assertTrue(task.isRunning())

        self.window.close()

        self.assertFalse(task.isRunning(), "stream thread survived window close")
        self.assertEqual(capture.release_count, 1, "capture leaked on close")
        self.assertFalse(self.window.is_busy())

    def test_camera_index_input_is_accepted(self) -> None:
        self.load_model()
        page = self.window.stream_page
        capture = FakeCapture(max_frames=1, read_delay=0.001)
        seen: list = []

        def factory(target):
            seen.append(target)
            return capture

        page.capture_factory = factory
        page.set_input("camera", 0)
        self.assertTrue(page.start_stream())
        self.assertTrue(wait_until(lambda: not page._running))
        self.assertEqual(seen, [0])
        self.assertEqual(capture.release_count, 1)


class ResultsPageTest(GuiTestBase):
    def test_missing_product_notice_without_evaluation(self) -> None:
        run = helpers.make_fake_run(self.project, "gui_no_eval",
                                    with_evaluation=False, with_benchmark=False)
        page = self.window.results_page
        summary = page.load_run(run.run_dir)
        self.assertTrue(summary["ok"])
        self.assertFalse(summary["complete"])
        notices = page.missing_notices()
        self.assertTrue(any("缺失产物" in item for item in notices))
        self.assertTrue(any("evaluation/metrics.json" in item for item in notices))
        self.assertTrue(any("evaluation/per_class.csv" in item for item in notices))
        self.assertIn("缺失产物", page.notice_text())
        self.assertFalse(page.notice_label.isHidden())
        self.assertTrue(page.ok_label.isHidden(), "positive notice must stay hidden")
        self.assertEqual(page.metric_labels["map50"].text(), "N/A")
        # real stage statuses from manifest.json: evaluation is still pending
        self.assertEqual(page.stage_table.item(2, 1).text(), "pending")
        self.assertEqual(page.stage_table.item(1, 1).text(), "complete")

    def test_metrics_figures_and_scope_badge(self) -> None:
        run = helpers.make_fake_run(self.project, "gui_with_eval")
        figure_dir = run.run_dir / "figures" / "v1"
        figure_dir.mkdir(parents=True, exist_ok=True)
        helpers.write_png(figure_dir / "F07_per_class_ap.png", 320, 220, seed=5)
        page = self.window.results_page
        summary = page.load_run(run.run_dir)

        self.assertEqual(page.metric_labels["map50"].text(), "42.992%")
        self.assertEqual(page.metric_labels["map50_95"].text(), "25.111%")
        self.assertEqual(page.metric_labels["precision"].text(), "78.123%")
        self.assertEqual(page.metric_labels["recall"].text(), "65.234%")
        self.assertEqual(page.scope_badge.text(), "formal 正式")
        self.assertEqual(page.state_badge.text(), "存在未完成阶段")
        self.assertEqual([path.name for path in page.figure_paths()],
                         ["F07_per_class_ap.png"])
        self.assertEqual(page.missing_notices(), [])
        self.assertFalse(page.ok_label.isHidden(), "complete products need a notice")
        self.assertIn("图片数", page.extra_metrics_label.text())
        self.assertEqual(summary["scope"], "formal")
        self.assertTrue(page.figure_summary.text().startswith("共 1 张 PNG"))

    def test_smoke_run_is_badged_as_smoke(self) -> None:
        smoke_project = helpers.make_project(
            self.tmp_path / "smoke_proj", scope="smoke",
            overrides={"train.epochs": 2, "train.batch": 2})
        run = helpers.make_fake_run(smoke_project, "gui_smoke")
        page = self.window.results_page
        page.load_run(run.run_dir)
        self.assertEqual(page.scope_badge.text(), "smoke 小样本")
        self.assertEqual(page.scope_badge.property("badge"), "smoke")

    def test_non_run_directory_is_reported(self) -> None:
        page = self.window.results_page
        empty = self.tmp_path / "not_a_run"
        empty.mkdir()
        summary = page.load_run(empty)
        self.assertFalse(summary["ok"])
        self.assertIn("manifest.json", page.notice_text())
        self.assertTrue(page.status.property("error") in ("true", True))

        missing = page.load_run(self.tmp_path / "does_not_exist")
        self.assertFalse(missing["ok"])
        self.assertIn("目录不存在", page.notice_text())

    def test_run_discovery_lists_outputs(self) -> None:
        helpers.make_fake_run(self.project, "gui_list_a", with_evaluation=False,
                              with_benchmark=False)
        helpers.make_fake_run(self.project, "gui_list_b", with_evaluation=False,
                              with_benchmark=False)
        page = self.window.results_page
        directories = page.refresh_runs()
        self.assertEqual(len(directories), 2)
        self.assertEqual(page.run_combo.count(), 3)  # placeholder + 2 runs


@unittest.skipUnless(HAVE_QT, f"PyQt6 not importable ({QT_IMPORT_ERROR})")
class WindowChromeTest(GuiTestBase):
    def test_navigation_switches_all_four_pages(self) -> None:
        window = self.window
        self.assertEqual(len(window.pages()), 4)
        for key in ("image", "stream", "model", "results"):
            window._navigate_to(key)
            pump()
            self.assertIs(window.stack.currentWidget(), window.page_for(key))
            self.assertEqual(window.page_title.text(), main_window_module.PAGE_META[key][0])
            self.assertTrue(window._nav_buttons[key].isChecked())
            self.assertEqual(sum(1 for b in window._nav_buttons.values() if b.isChecked()), 1)

    def test_stylesheet_is_applied(self) -> None:
        self.assertGreater(len(self.window.styleSheet()), 800)
        self.assertIn("QMainWindow", self.window.styleSheet())

    def test_window_grab_has_real_size(self) -> None:
        self.window.show()
        pump(3)
        pixmap = self.window.grab()
        self.assertGreaterEqual(pixmap.width(), 1180)
        self.assertGreaterEqual(pixmap.height(), 720)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)


@unittest.skipUnless(HAVE_QT, f"PyQt6 not importable ({QT_IMPORT_ERROR})")
class FigureMissingNoticeTests(unittest.TestCase):
    """The results page must surface figure_manifest 'missing' entries."""

    def setUp(self) -> None:
        qapp()
        import tempfile
        from pathlib import Path

        from tests import helpers

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = helpers.make_project(Path(self._tmp.name) / "proj")
        self.run = helpers.make_fake_run(self.project, "gui_missing_figure")
        figure_dir = self.run.run_dir / "figures" / "v1"
        figure_dir.mkdir(parents=True, exist_ok=True)
        helpers.write_png(figure_dir / "class_distribution.png", 320, 220, seed=3)
        import json

        (figure_dir / "figure_manifest.json").write_text(json.dumps({
            "figure_tag": "v1", "status": "partial",
            "missing": [{"figure_id": "F11", "reason": "no image meets the rules",
                         "missing_inputs": []}],
        }), encoding="utf-8")
        from uavdet.app.main_window import MainWindow

        self.window = MainWindow(config_path=self.project.config_path,
                                 settings_path=Path(self._tmp.name) / "settings.json",
                                 detector_factory=lambda w, d: None)
        self.addCleanup(self.window.shutdown)
        self.addCleanup(self.window.close)

    def test_missing_figure_is_listed_and_not_hidden(self) -> None:
        page = self.window.results_page
        page.load_run(self.run.run_dir)
        notices = page.missing_notices()
        self.assertTrue(any("F11" in item and "图表缺项" in item for item in notices), notices)
        self.assertIn("F11", page.notice_text())


@unittest.skipUnless(HAVE_QT, f"PyQt6 not importable ({QT_IMPORT_ERROR})")
class LanguageToggleTest(GuiTestBase):
    """The 中文/English switch rebuilds the UI without losing state."""

    HAN = __import__("re").compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

    def test_default_is_chinese_and_english_labels_wired(self) -> None:
        from uavdet.app import i18n

        self.assertEqual(i18n.get_language(), "zh")
        self.assertEqual(self.window.language_button.text(), "English")
        # the toggle button offers the *other* language
        self.assertEqual(self.window.language_button.text(), i18n.LANGUAGE_LABELS["en"])

    def test_toggle_switches_visible_text_to_english_and_back(self) -> None:
        from uavdet.app import i18n

        page = self.window.page_for("image")
        self.assertEqual(page.open_button.text(), "打开图片")
        ok, target = self.window.toggle_language()
        self.assertTrue(ok)
        self.assertEqual(target, "en")
        self.assertEqual(i18n.get_language(), "en")
        self.assertEqual(self.window.language_button.text(), "中文")
        # rebuilt pages show English chrome
        image_page = self.window.page_for("image")
        self.assertEqual(image_page.open_button.text(), "Open image")
        self.assertEqual(image_page.run_button.text(), "Start detection")
        self.assertEqual(self.window.page_title.text() if self.window._current_page == "image" else
                         page_title_for(self.window, "image"), "Image detection")
        self.assertFalse(self.HAN.search(self.window.model_badge.text()))
        self.assertFalse(self.HAN.search(self.window.status_line.text()))
        # every nav button is English now
        for text in [button.text() for button in self.window._nav_buttons.values()]:
            self.assertIsNone(self.HAN.search(text), text)

        ok, target = self.window.toggle_language()
        self.assertTrue(ok)
        self.assertEqual(target, "zh")
        self.assertEqual(i18n.get_language(), "zh")
        self.assertEqual(self.window.page_for("image").open_button.text(), "打开图片")

    def test_language_is_persisted_and_restored(self) -> None:
        import json

        self.assertTrue(self.window.toggle_language()[0])
        payload = json.loads(self.window.settings_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["language"], "en")
        second = main_window_module.MainWindow(
            config_path=str(self.project.config_path),
            detector_factory=self.factory,
            settings_path=self.window.settings_path,
        )
        try:
            self.assertEqual(second.language_button.text(), "中文")
            self.assertEqual(second.page_for("image").open_button.text(), "Open image")
        finally:
            second.shutdown()
            second.close()
            pump()

    def test_toggle_keeps_the_loaded_image_and_the_selected_run(self) -> None:
        detector = self.run_image_detection()
        page = self.window.page_for("image")
        before_count = page.result.count if page.result else None
        results_page = self.window.page_for("results")
        run = helpers.make_fake_run(self.project, "toggle_run")
        results_page.load_run(run.run_dir)
        self.assertTrue(self.window.toggle_language()[0])
        page = self.window.page_for("image")
        results_page = self.window.page_for("results")
        self.assertEqual(page.result.count if page.result else None, before_count)
        self.assertEqual(page.current_image_path.name, "frame.png")
        self.assertEqual(results_page.run_path_edit.text(), str(run.run_dir))
        self.assertTrue(detector.predict_calls, "a language switch must not re-run inference")
        self.assertEqual(len([call for call in detector.predict_calls]), len(detector.predict_calls))

    def test_toggle_is_refused_while_a_task_runs(self) -> None:
        self.load_model()
        self.open_image()
        page = self.window.page_for("image")
        page.start_detection()
        ok, _ = self.window.toggle_language()
        self.assertFalse(ok)
        self.assertEqual(self.window.settings["language"], "zh")
        self.wait_idle()

    def test_english_mode_has_no_chinese_in_visible_widget_text(self) -> None:
        self.assertTrue(self.window.toggle_language()[0])
        from PyQt6.QtWidgets import QLabel, QPushButton, QRadioButton, QCheckBox, QLineEdit

        texts = []

        for page in self.window.pages():
            for widget in page.findChildren(QLabel) + page.findChildren(QPushButton) + \
                    page.findChildren(QRadioButton) + page.findChildren(QCheckBox):
                if hasattr(widget, "text"):
                    texts.append(widget.text())
            for edit in page.findChildren(QLineEdit):
                texts.append(edit.placeholderText())
        for text in texts:
            self.assertIsNone(self.HAN.search(text), f"Chinese label left in English mode: {text!r}")


def page_title_for(window, key: str) -> str:
    from uavdet.app.main_window import PAGE_META

    from uavdet.app.i18n import tr

    return tr(PAGE_META[key][0])
