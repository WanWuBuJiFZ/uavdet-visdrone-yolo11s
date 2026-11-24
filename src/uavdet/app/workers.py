"""Background workers for the desktop app.

Threading model (``docs/INTERFACE_CONTRACT.md`` §3.7, planning 00 §3 模块F):

* the model lives inside a :class:`DetectorSession`; ``predict`` is only ever
  called from a worker thread and the session refuses calls from the main
  thread, so no YOLO object is used from two threads by construction;
* one task at a time — the main window refuses to start a second one;
* :class:`StreamTask` stops through a :class:`threading.Event`, which works
  even while the worker is inside a blocking ``capture.read()`` (a queued Qt
  signal would stay unprocessed there);
* the capture is released in ``finally`` on normal end, error *and* stop.

PyQt6 is imported inside :func:`_build` so importing this module (and therefore
the whole app package) never needs Qt.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from ..types import InferenceOptions, PredictionResult, UAVDetError
from .widgets import LazyExports
from .i18n import tr

__all__ = [
    "DetectorSession",
    "default_detector_factory",
    "default_capture_factory",
    "describe_error",
    "ModelLoadTask",
    "ImageTask",
    "StreamTask",
]


# ---------------------------------------------------------------------------
# default factories (heavy imports stay inside)
# ---------------------------------------------------------------------------
def default_detector_factory(weights: str, device: str) -> Any:
    """Build the real :class:`uavdet.inference.Detector` (lazy import)."""
    from ..inference import Detector

    return Detector(weights, device)


def warm_up_runtime() -> list[str]:
    """Import torch/Ultralytics on the *calling* (main) thread.

    On macOS + PyQt6 the first ``import torch`` inside a ``QThread`` can abort the
    process (SIGBUS in the BLAS/OpenMP initialisation). The GUI therefore imports
    the inference stack once on the main thread before any worker starts; the
    heavy import costs a few seconds and keeps worker threads crash-free.
    Returns human-readable warnings for the status bar (never raises).
    """
    warnings: list[str] = []
    try:
        import torch  # noqa: F401  (intentional main-thread import)
    except Exception as exc:  # pragma: no cover - depends on the install
        warnings.append(tr('torch 不可用：{0}').format(describe_error(exc)))
        return warnings
    try:
        import ultralytics  # noqa: F401  (intentional main-thread import)
    except Exception as exc:  # pragma: no cover - depends on the install
        warnings.append(tr('ultralytics 不可用：{0}').format(describe_error(exc)))
    _ = (torch, ultralytics)
    return warnings


def default_capture_factory(target: str | int) -> Any:
    """``cv2.VideoCapture(target)`` — video path or camera index."""
    import cv2

    return cv2.VideoCapture(target)


def describe_error(exc: BaseException) -> str:
    """Short, user facing error text that keeps the exception type."""
    message = str(exc).strip()
    kind = type(exc).__name__
    return f"{kind}: {message}" if message else kind


def _close_quietly(target: Any) -> str | None:
    """Call ``target.close()`` swallowing (and reporting) failures."""
    closer = getattr(target, "close", None)
    if not callable(closer):
        return None
    try:
        closer()
    except Exception as exc:  # noqa: BLE001 - closing must never mask a result
        return describe_error(exc)
    return None


# ---------------------------------------------------------------------------
# the model owner
# ---------------------------------------------------------------------------
class DetectorSession:
    """Owns exactly one detector and performs every call on it.

    ``load`` builds a *new* detector first and only replaces the current one
    after it loaded successfully — a failed switch keeps the previous valid
    model usable (acceptance rule C04). ``predict``/``annotate`` raise when
    called from the main thread, which makes "no inference on the GUI thread"
    an enforced invariant instead of a convention.
    """

    def __init__(self, weights: str = "", device: str = "cpu", *,
                 factory: Callable[[str, str], Any] | None = None):
        self.weights = str(weights or "")
        self.device = str(device or "cpu")
        self._factory = factory or default_detector_factory
        self._detector: Any = None
        self.loaded = False
        self.info: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.thread_id: int | None = None
        self.load_thread_id: int | None = None

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> dict[str, Any]:
        """Create and load a new detector; only then swap it in."""
        if not self.weights:
            raise UAVDetError(tr('未指定权重文件，无法加载模型'))
        try:
            detector = self._factory(self.weights, self.device)
        except Exception as exc:
            raise UAVDetError(tr('无法创建检测器 {0}: {1}').format(self.weights, describe_error(exc))) from exc
        if detector is None:
            raise UAVDetError(tr('检测器工厂返回了 None'))

        try:
            loader = getattr(detector, "load", None)
            info = dict(loader() or {}) if callable(loader) else {}
        except Exception as exc:
            _close_quietly(detector)
            raise UAVDetError(tr('加载权重失败 {0}: {1}').format(self.weights, describe_error(exc))) from exc

        previous, self._detector = self._detector, detector
        self.loaded = True
        self.info = info
        self.thread_id = threading.get_ident()
        self.load_thread_id = self.thread_id
        self.warnings = [str(w) for w in (getattr(detector, "warnings", None) or [])]
        if previous is not None and previous is not detector:
            warning = _close_quietly(previous)
            if warning:
                self.warnings.append(tr('关闭旧模型时出错：{0}').format(warning))
        return info

    def close(self) -> list[str]:
        """Release the detector; returns warnings (never raises)."""
        detector, self._detector = self._detector, None
        self.loaded = False
        errors: list[str] = []
        if detector is not None:
            warning = _close_quietly(detector)
            if warning:
                errors.append(warning)
        return errors

    # -- guards ------------------------------------------------------------
    def _require_detector(self) -> Any:
        if self._detector is None or not self.loaded:
            raise UAVDetError(tr('尚未加载检测模型，请先在【模型与参数】页加载 10 类权重'))
        return self._detector

    @staticmethod
    def _require_worker_thread() -> None:
        if threading.current_thread() is threading.main_thread():
            raise RuntimeError(
                tr('推理必须在工作线程中执行；主线程不允许调用 Detector.predict')
            )

    # -- inference ---------------------------------------------------------
    def predict(self, image_bgr: Any, options: InferenceOptions, *,
                image_id: str = "array", source: str = "") -> PredictionResult:
        """Run one prediction inside the calling (worker) thread."""
        detector = self._require_detector()
        self._require_worker_thread()
        if not isinstance(options, InferenceOptions):
            raise UAVDetError(tr('options 必须是 InferenceOptions，收到 {0}').format(type(options).__name__))
        result = detector.predict(image_bgr, options, image_id=image_id, source=source)
        self.thread_id = threading.get_ident()
        return result

    def annotate(self, image_bgr: Any, detections: Any) -> Any:
        """Draw the detections; a drawing failure must not lose the results."""
        detector = self._require_detector()
        self._require_worker_thread()
        annotate = getattr(detector, "annotate", None)
        if not callable(annotate):
            return image_bgr
        try:
            annotated = annotate(image_bgr, detections)
        except Exception as exc:  # noqa: BLE001 - keep the detection table usable
            self.warnings.append(tr('标注绘制失败，显示原图：{0}').format(describe_error(exc)))
            return image_bgr.copy()
        return image_bgr if annotated is None else annotated

    # -- info --------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "device": self.device,
            "loaded": bool(self.loaded),
            "warnings": list(self.warnings),
            "info": dict(self.info),
            "thread_id": self.thread_id,
        }


# ---------------------------------------------------------------------------
# QThread workers (built lazily)
# ---------------------------------------------------------------------------
_lazy = LazyExports(__name__, lambda: _build()).register(
    "ModelLoadTask", "ImageTask", "StreamTask"
)


def __getattr__(name: str) -> Any:
    return _lazy.resolve(name)


def _build() -> dict[str, Any]:
    """Import Qt and define the worker classes (called once, lazily)."""
    from PyQt6.QtCore import QThread, pyqtSignal

    class ModelLoadTask(QThread):
        """Loads weights inside the worker thread (never on the GUI thread)."""

        loaded = pyqtSignal(object, dict)   # DetectorSession, info dict
        failed = pyqtSignal(str)

        def __init__(self, session: DetectorSession, parent: Any = None):
            super().__init__(parent)
            self.session = session

        def run(self) -> None:  # noqa: D401 - Qt entry point
            try:
                info = self.session.load()
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                self.failed.emit(describe_error(exc))
                return
            self.loaded.emit(self.session, dict(info or {}))

    class ImageTask(QThread):
        """One image prediction, executed on the worker thread."""

        succeeded = pyqtSignal(object, object)   # PredictionResult, annotated BGR
        failed = pyqtSignal(str)

        def __init__(self, session: DetectorSession, image_bgr: Any,
                     options: InferenceOptions, *, image_id: str = "image",
                     source: str = "", annotate: bool = True, parent: Any = None):
            super().__init__(parent)
            self.session = session
            self.image = image_bgr
            self.options = options
            self.image_id = image_id
            self.source = source
            self.annotate_output = bool(annotate)

        def run(self) -> None:  # noqa: D401 - Qt entry point
            try:
                result = self.session.predict(
                    self.image, self.options,
                    image_id=self.image_id, source=self.source,
                )
                annotated = (self.session.annotate(self.image, result.detections)
                             if self.annotate_output else None)
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                self.failed.emit(describe_error(exc))
                return
            self.succeeded.emit(result, annotated)

    class StreamTask(QThread):
        """Video file / camera capture with detection on every frame."""

        frame_ready = pyqtSignal(object, object)   # annotated BGR, PredictionResult
        failed = pyqtSignal(str)

        def __init__(self, session: DetectorSession, source: str | int,
                     options: InferenceOptions, *,
                     capture_factory: Callable[[str | int], Any] | None = None,
                     annotate: bool = True, parent: Any = None):
            super().__init__(parent)
            self.session = session
            self.source = source
            self.options = options
            self.capture_factory = capture_factory or default_capture_factory
            self.annotate_output = bool(annotate)
            #: set from another thread, checked before every read/predict
            self._stop = threading.Event()
            self.frames_processed = 0
            self.released = False
            self.last_error: str | None = None

        # -- control -------------------------------------------------------
        def request_stop(self) -> None:
            """Ask the loop to end; safe from any thread, no event loop needed."""
            self._stop.set()

        @property
        def stop_requested(self) -> bool:
            return self._stop.is_set()

        def source_label(self) -> str:
            if isinstance(self.source, int):
                return f"camera{self.source}"
            return str(self.source)

        # -- Qt entry point -------------------------------------------------
        def run(self) -> None:
            capture: Any = None
            error: str | None = None
            try:
                capture = self.capture_factory(self.source)
                if capture is None:
                    raise UAVDetError(tr('无法创建视频/摄像头采集对象'))
                is_opened = getattr(capture, "isOpened", None)
                if callable(is_opened) and not bool(is_opened()):
                    raise UAVDetError(tr('无法打开输入源：{0}').format(self.source))
                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    if self._stop.is_set():
                        break
                    index = self.frames_processed + 1
                    result = self.session.predict(
                        frame, self.options,
                        image_id=f"{self.source_label()}#{index}",
                        source=self.source_label(),
                    )
                    annotated = (self.session.annotate(frame, result.detections)
                                 if self.annotate_output else frame)
                    self.frames_processed = index
                    self.frame_ready.emit(annotated, result)
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                error = describe_error(exc)
            finally:
                # ALWAYS release: normal end, error and user stop.
                if capture is not None:
                    releaser = getattr(capture, "release", None)
                    try:
                        if callable(releaser):
                            releaser()
                            self.released = True
                    except Exception as exc:  # noqa: BLE001
                        extra = tr('释放采集资源失败：{0}').format(describe_error(exc))
                        error = f"{error}; {extra}" if error else extra
            self.last_error = error
            if error:
                self.failed.emit(error)

    return {"ModelLoadTask": ModelLoadTask, "ImageTask": ImageTask, "StreamTask": StreamTask}
