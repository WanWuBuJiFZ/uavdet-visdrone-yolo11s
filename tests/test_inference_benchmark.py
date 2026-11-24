"""Detection engine and benchmark boundaries (planning checklist C01-C06).

All models are fakes from :mod:`tests.helpers`; no real weights are loaded and
nothing outside the temporary test directory is written.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tests import helpers
from uavdet import benchmark, evaluation, inference, training
from uavdet.artifacts import begin_stage, complete_stage, open_run
from uavdet.data import audit_dataset, prepare_dataset_view, save_audit, view_split_entries
from uavdet.inference import Detector, save_predictions
from uavdet.types import (
    CLASS_NAMES,
    InferenceOptions,
    PredictionResult,
    RunContext,
    UAVDetError,
    WeightError,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def make_weights(tmp_path: Path, name: str = "best.pt") -> Path:
    path = Path(tmp_path) / "weights" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic-10-class-checkpoint")
    return path


def frame(width: int = 64, height: int = 48) -> np.ndarray:
    """A deterministic BGR uint8 frame with distinguishable channels."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)          # B, G, R
    image[0, 0] = (10, 200, 30)         # a known "green-ish" BGR pixel
    image[height - 1, width - 1] = (240, 10, 10)  # a known "blue-ish" BGR pixel
    return image


class SequenceModel:
    """``predict`` returns the queued results in order (last item repeats)."""

    def __init__(self, results: Sequence[Any], names: Mapping[int, str] | None = None) -> None:
        self.names = dict(names or {index: name for index, name in enumerate(CLASS_NAMES)})
        self.queue = list(results)
        self.calls: list[dict[str, Any]] = []

    def predict(self, source: Any = None, **kwargs: Any) -> list[Any]:
        self.calls.append(dict(kwargs, source=source))
        if not self.queue:
            return [helpers.FakeResult(helpers.FakeBoxes([], [], []))]
        if len(self.queue) == 1:
            return [self.queue[0]]
        return [self.queue.pop(0)]


def one_box_result(boxes: Sequence[Sequence[float]] = ((1, 2, 10, 12),),
                   confidences: Sequence[float] = (0.9,),
                   classes: Sequence[int] = (0,)) -> helpers.FakeResult:
    return helpers.FakeResult(
        helpers.FakeBoxes(boxes, confidences, classes),
        orig_shape=(48, 64),
        speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5},
    )


def empty_result() -> helpers.FakeResult:
    return helpers.FakeResult(helpers.FakeBoxes([], [], []), orig_shape=(48, 64))


class EngineTestCase(helpers.TempDirTestCase):
    def detector(self, model: Any, device: str = "cpu", **kwargs: Any) -> Detector:
        self.weights = make_weights(self.tmp_path)
        return Detector(self.weights, device=device,
                       model_factory=lambda weights: model, **kwargs)


# ---------------------------------------------------------------------------
# C01 - explicit engine parameters
# ---------------------------------------------------------------------------
class EngineParameterTests(EngineTestCase):
    def test_c01_predict_passes_every_configured_parameter(self) -> None:
        model = SequenceModel([one_box_result()])
        detector = self.detector(model)
        options = InferenceOptions(imgsz=512, conf=0.4, iou=0.6, max_det=7, device="cpu")
        detector.predict(frame(), options, image_id="img_0001", source="/tmp/img_0001.jpg")
        self.assertEqual(len(model.calls), 1)
        call = model.calls[0]
        self.assertEqual(call["imgsz"], 512)
        self.assertEqual(call["conf"], 0.4)
        self.assertEqual(call["iou"], 0.6)
        self.assertEqual(call["max_det"], 7)
        self.assertEqual(call["device"], "cpu")
        self.assertFalse(call["verbose"])
        # Precision is always explicit: the historical ``half`` keyword or the
        # current ``quantize`` equivalent, never an implicit default.
        self.assertTrue("half" in call or "quantize" in call, f"no precision arg: {sorted(call)}")
        if "half" in call:
            self.assertFalse(call["half"])
        if "quantize" in call:
            self.assertIsNone(call["quantize"])

    def test_c01_second_predict_does_not_reload_the_model(self) -> None:
        model = SequenceModel([one_box_result()])
        loads: list[Any] = []

        def factory(weights: Any) -> Any:
            loads.append(weights)
            return model

        detector = Detector(make_weights(self.tmp_path), "cpu", model_factory=factory)
        detector.load()
        detector.predict(frame(), InferenceOptions())
        detector.predict(frame(), InferenceOptions())
        self.assertEqual(len(loads), 1)

    def test_cpu_fallback_is_recorded_when_the_gpu_is_missing(self) -> None:
        model = SequenceModel([one_box_result()])
        detector = self.detector(model, device="0")
        stats = detector.load()
        self.assertEqual(stats["device"]["requested"], "0")
        self.assertEqual(stats["device"]["effective"], "cpu")
        self.assertTrue(detector.warnings, "a CPU fallback must never be silent")
        self.assertTrue(any("CPU" in text or "cpu" in text for text in detector.warnings))
        detector.predict(frame(), InferenceOptions())
        self.assertEqual(model.calls[-1]["device"], "cpu")


# ---------------------------------------------------------------------------
# C02 - BGR is preserved, exactly one conversion point
# ---------------------------------------------------------------------------
class ColourTests(EngineTestCase):
    def test_c02_bgr_frame_reaches_the_model_unchanged(self) -> None:
        model = SequenceModel([one_box_result()])
        detector = self.detector(model)
        image = frame()
        detector.predict(image, InferenceOptions(), image_id="colour")
        sent = model.calls[-1]["source"]
        self.assertIs(sent, image, "predict() must hand the same BGR array to Ultralytics")
        np.testing.assert_array_equal(sent, image)
        self.assertEqual(int(sent[0, 0, 1]), 200, "the green channel must stay in place")

    def test_c02_annotate_keeps_colours_and_the_original_frame(self) -> None:
        model = SequenceModel([one_box_result()])
        detector = self.detector(model)
        image = frame()
        result = detector.predict(image, InferenceOptions())
        annotated = detector.annotate(image, result.detections)
        self.assertIsNot(annotated, image)
        self.assertEqual(annotated.dtype, np.uint8)
        self.assertEqual(annotated.shape, image.shape)
        # The source frame is untouched ...
        np.testing.assert_array_equal(image, frame())
        # ... and untouched pixels keep their BGR order (no flip, no grey).
        np.testing.assert_array_equal(annotated[0, 0], image[0, 0])
        np.testing.assert_array_equal(annotated[47, 63], image[47, 63])
        self.assertLessEqual(abs(int(annotated[10, 20, 0]) - int(image[10, 20, 0])), 255)

    def test_c02_annotate_marks_the_box_area(self) -> None:
        model = SequenceModel([one_box_result()])
        detector = self.detector(model)
        image = frame()
        result = detector.predict(image, InferenceOptions())
        annotated = detector.annotate(image, result.detections)
        changed = np.any(annotated != image, axis=2)
        self.assertTrue(changed.any(), "the annotated copy must show the detection")
        self.assertFalse(changed[47, 63], "the bottom-right corner is outside the box")


# ---------------------------------------------------------------------------
# annotation geometry: a label must never be clipped by the image edge
# ---------------------------------------------------------------------------
class LabelGeometryTests(helpers.TempDirTestCase):
    def test_label_rect_stays_inside_the_image_for_corner_boxes(self) -> None:
        width, height = 2000, 1500
        cases = [
            (0.0, 0.0, 40.0),          # top-left corner (common in VisDrone)
            (1990.0, 0.0, 1500.0),     # top-right corner
            (0.0, 1480.0, 1500.0),     # bottom-left corner
            (1980.0, 1470.0, 1500.0),  # bottom-right corner
            (900.0, 700.0, 760.0),     # middle
        ]
        for x1, y1, y2 in cases:
            x, y, w, h = inference.label_rect(x1, y1, y2, 120, 14, 4, width, height)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, width)
            self.assertLessEqual(y + h, height)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)

    def test_label_prefers_above_then_below_then_inside(self) -> None:
        # room above -> above the box
        _, y, _, h = inference.label_rect(100, 500, 520, 120, 14, 4, 2000, 1500)
        self.assertLessEqual(y + h, 500)
        # no room above -> below the box
        _, y, _, h = inference.label_rect(100, 2, 30, 120, 14, 4, 2000, 1500)
        self.assertGreaterEqual(y, 30)
        # neither -> clamped inside the image
        _, y, _, h = inference.label_rect(100, 2, 1499, 120, 14, 4, 2000, 1500)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(y + h, 1500)

    def test_rect_is_clamped_in_a_tiny_image(self) -> None:
        x, y, w, h = inference.label_rect(0, 0, 12, 200, 14, 4, 16, 12)
        self.assertEqual((x, y), (0, 0))
        self.assertLessEqual(w, 16)
        self.assertLessEqual(h, 12)

    def test_font_follows_the_frame_and_shrinks_for_long_labels(self) -> None:
        # A full-resolution export gets a legible label ...
        self.assertAlmostEqual(inference.label_font_scale("car 0.70", 2000, 1500), 1.0)
        # ... a mid-size frame a proportionate one ...
        self.assertAlmostEqual(inference.label_font_scale("car 0.70", 640, 480),
                               480 / 900.0, places=6)
        # ... and a long label in a small image is shrunk, never below the floor.
        self.assertLess(inference.label_font_scale("awning-tricycle 0.90", 64, 48), 0.4)
        self.assertGreaterEqual(inference.label_font_scale("awning-tricycle 0.90", 64, 48), 0.25)

    def test_annotate_labels_a_box_at_the_image_edge_without_raising(self) -> None:
        model = SequenceModel([one_box_result((0, 0, 8, 8), (0.7,), (3,))])
        detector = Detector(make_weights(self.tmp_path), "cpu",
                            model_factory=lambda weights: model)
        image = frame(2000, 1500)
        result = detector.predict(image, InferenceOptions(conf=0.5))
        annotated = detector.annotate(image, result.detections)
        self.assertEqual(annotated.shape, image.shape)
        np.testing.assert_array_equal(image, frame(2000, 1500))
        self.assertFalse(np.array_equal(annotated, image))
        # The label block is drawn at the left edge but the text origin stays inside.
        self.assertTrue((annotated[0:20, 0:160] != image[0:20, 0:160]).any())


# ---------------------------------------------------------------------------
# C03 - max_det and empty results
# ---------------------------------------------------------------------------
class MaxDetTests(EngineTestCase):
    def test_c03_max_det_caps_the_output(self) -> None:
        boxes = [[i, i, i + 5, i + 5] for i in range(10)]
        confidences = [0.9 - 0.05 * i for i in range(10)]
        classes = [0] * 10
        model = SequenceModel([one_box_result(boxes, confidences, classes)])
        detector = self.detector(model)
        result = detector.predict(frame(), InferenceOptions(conf=0.1, max_det=3))
        self.assertEqual(result.count, 3)
        for detection, expected in zip(result.detections, confidences[:3]):
            self.assertAlmostEqual(detection.confidence, expected, places=5)

    def test_c03_empty_result_is_a_valid_result(self) -> None:
        model = SequenceModel([empty_result()])
        detector = self.detector(model)
        result = detector.predict(frame(), InferenceOptions(), image_id="empty")
        self.assertEqual(result.count, 0)
        self.assertEqual(result.detections, [])
        self.assertEqual(result.image_id, "empty")
        payload = result.to_dict()
        self.assertEqual(payload["num_detections"], 0)
        self.assertEqual(payload["detections"], [])
        annotated = detector.annotate(frame(), result.detections)
        np.testing.assert_array_equal(annotated, frame())


# ---------------------------------------------------------------------------
# input contract
# ---------------------------------------------------------------------------
class InputContractTests(EngineTestCase):
    def test_float_and_greyscale_frames_are_rejected(self) -> None:
        detector = self.detector(SequenceModel([empty_result()]))
        with self.assertRaises(UAVDetError):
            detector.predict(frame().astype(np.float32), InferenceOptions())
        with self.assertRaises(UAVDetError):
            detector.predict(frame()[:, :, 0], InferenceOptions())
        with self.assertRaises(UAVDetError):
            detector.predict([[1, 2, 3]], InferenceOptions())  # type: ignore[arg-type]
        with self.assertRaises(UAVDetError):
            detector.predict(np.zeros((0, 5, 3), dtype=np.uint8), InferenceOptions())


# ---------------------------------------------------------------------------
# C04 - load failures and model switching hygiene
# ---------------------------------------------------------------------------
class LoadFailureTests(EngineTestCase):
    def test_c04_load_failure_raises_and_leaves_nothing_loaded(self) -> None:
        def factory(weights: Any) -> Any:
            raise RuntimeError("cannot load weights")

        detector = Detector(make_weights(self.tmp_path), "cpu", model_factory=factory)
        with self.assertRaises(WeightError) as raised:
            detector.load()
        self.assertIn("cannot load weights", str(raised.exception))
        self.assertIsNone(detector.model)
        self.assertFalse(detector.loaded)
        # A second attempt reports the same clean error (no half-initialised state).
        with self.assertRaises(WeightError):
            detector.predict(frame(), InferenceOptions())

    def test_c04_missing_weights_file_is_a_weight_error(self) -> None:
        detector = Detector(self.tmp_path / "nope.pt", "cpu",
                            model_factory=lambda weights: SequenceModel([]))
        with self.assertRaises(WeightError):
            detector.load()
        self.assertFalse(detector.loaded)

    def test_c04_wrong_class_count_is_rejected(self) -> None:
        coco = SequenceModel([empty_result()],
                             names={index: f"coco_{index}" for index in range(80)})
        detector = Detector(make_weights(self.tmp_path), "cpu",
                            model_factory=lambda weights: coco)
        with self.assertRaises(WeightError) as raised:
            detector.load()
        self.assertIn("80 class", str(raised.exception))
        self.assertFalse(detector.loaded)
        self.assertIsNone(detector.model)

    def test_failed_switch_keeps_the_previous_model_usable(self) -> None:
        good = SequenceModel([one_box_result()])
        calls: list[str] = []

        def flaky(weights: Any) -> Any:
            calls.append(str(weights))
            if len(calls) == 1:
                raise RuntimeError("bad checkpoint")
            return good

        detector = Detector(make_weights(self.tmp_path), "cpu", model_factory=flaky)
        with self.assertRaises(WeightError):
            detector.load()
        self.assertFalse(detector.loaded)
        self.assertIsNone(detector.model)
        # A second attempt (the operator switched to a valid file) works and the
        # detector is fully usable afterwards.
        self.assertEqual(detector.load()["num_classes"], 10)
        self.assertEqual(detector.predict(frame(), InferenceOptions()).count, 1)

        # An already operating detector is never invalidated by a later factory:
        # load() is idempotent, so a failed switch cannot leave it half-loaded.
        def broken(weights: Any) -> Any:
            raise RuntimeError("bad checkpoint")

        detector.model_factory = broken
        self.assertTrue(detector.loaded)
        self.assertEqual(detector.predict(frame(), InferenceOptions()).count, 1)


# ---------------------------------------------------------------------------
# C05 - lifecycle
# ---------------------------------------------------------------------------
class LifecycleTests(EngineTestCase):
    def test_c05_close_releases_the_model_and_is_idempotent(self) -> None:
        model = SequenceModel([one_box_result()])
        detector = self.detector(model)
        detector.load()
        self.assertTrue(detector.loaded)
        detector.close()
        self.assertFalse(detector.loaded)
        self.assertIsNone(detector.model)
        detector.close()  # idempotent
        self.assertFalse(detector.loaded)

    def test_c05_context_manager_closes_on_exit(self) -> None:
        model = SequenceModel([one_box_result()])
        detector = self.detector(model)
        with detector as active:
            self.assertIs(active, detector)
            self.assertEqual(active.predict(frame(), InferenceOptions()).count, 1)
        self.assertFalse(detector.loaded)
        self.assertIsNone(detector.model)

    def test_c05_context_manager_closes_after_an_error(self) -> None:
        model = SequenceModel([one_box_result()])
        detector = self.detector(model)
        with self.assertRaises(UAVDetError):
            with detector:
                detector.predict(frame().astype(np.float32), InferenceOptions())
        self.assertFalse(detector.loaded)


# ---------------------------------------------------------------------------
# save_predictions
# ---------------------------------------------------------------------------
class SavePredictionsTests(EngineTestCase):
    def test_zero_detection_images_are_still_saved(self) -> None:
        images_dir = self.tmp_path / "images"
        first = helpers.write_png(images_dir / "img_a.jpg", 64, 48)
        second = helpers.write_png(images_dir / "img_b.jpg", 64, 48, colour=(9, 9, 9))
        model = SequenceModel([one_box_result(), empty_result()])
        detector = self.detector(model)
        options = InferenceOptions(imgsz=640, conf=0.3, iou=0.7, max_det=11)
        out_dir = self.tmp_path / "predictions"
        results = save_predictions(detector, [first, second], out_dir, options)

        self.assertEqual([result.image_id for result in results], ["img_a", "img_b"])
        for name in ("detections.csv", "predictions.json", "run_meta.json"):
            self.assertTrue((out_dir / name).is_file(), f"{name} missing")
        self.assertTrue((out_dir / "annotated" / "img_a.jpg").is_file())
        self.assertTrue((out_dir / "annotated" / "img_b.jpg").is_file())

        rows = helpers.read_csv_rows(out_dir / "detections.csv")
        self.assertEqual(len(rows), 2, "one detection row plus one zero-detection row")
        self.assertEqual(rows[0]["image_id"], "img_a")
        self.assertEqual(rows[0]["num_detections"], "1")
        self.assertEqual(rows[1]["image_id"], "img_b")
        self.assertEqual(rows[1]["num_detections"], "0")
        self.assertEqual(rows[1]["class_name"], "")

        payload = json.loads((out_dir / "predictions.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["num_images"], 2)
        self.assertEqual(payload["num_detections"], 1)
        self.assertEqual([item["image_id"] for item in payload["results"]], ["img_a", "img_b"])
        self.assertEqual(payload["results"][1]["detections"], [])
        self.assertEqual(payload["options"]["conf"], 0.3)
        self.assertEqual(payload["options"]["max_det"], 11)

        meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["num_images"], 2)
        self.assertEqual(meta["options"]["imgsz"], 640)
        self.assertEqual(meta["device"]["requested"], "cpu")
        self.assertEqual(meta["device"]["effective"], "cpu")
        self.assertIsInstance(meta["device"]["warnings"], list)
        self.assertTrue(meta["weights_sha256"])

    def test_cpu_fallback_is_recorded_in_run_meta(self) -> None:
        image = helpers.write_png(self.tmp_path / "images" / "img_c.jpg", 64, 48)
        model = SequenceModel([empty_result()])
        detector = self.detector(model, device="0")
        out_dir = self.tmp_path / "predictions_gpu"
        save_predictions(detector, [image], out_dir, InferenceOptions())
        meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["device"]["requested"], "0")
        self.assertEqual(meta["device"]["effective"], "cpu")
        self.assertTrue(meta["device"]["warnings"], "the CPU fallback must be recorded")
        self.assertTrue(meta["warnings"])

    def test_unreadable_image_raises(self) -> None:
        broken = self.tmp_path / "images" / "broken.jpg"
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_bytes(b"not an image")
        detector = self.detector(SequenceModel([empty_result()]))
        with self.assertRaises(UAVDetError):
            save_predictions(detector, [broken], self.tmp_path / "out", InferenceOptions())


# ---------------------------------------------------------------------------
# C06 - benchmark timing boundaries
# ---------------------------------------------------------------------------
class _FakeDetector:
    """A detector whose cost is fully controlled by the test."""

    def __init__(self, weights: Path, device: str, *, predict_s: float = 0.004,
                 load_s: float = 0.03) -> None:
        self.weights = Path(weights)
        self.device = device
        self.predict_s = predict_s
        self.load_s = load_s
        self.events: list[str] = []
        self.images: list[np.ndarray] = []
        self.options: list[Any] = []

    def load(self) -> dict[str, Any]:
        self.events.append("load")
        time.sleep(self.load_s)
        return {"weights": str(self.weights)}

    def predict(self, image: Any, options: Any = None, *, image_id: str = "array",
                source: str = "") -> PredictionResult:
        self.events.append("predict")
        self.images.append(image)
        self.options.append(options)
        time.sleep(self.predict_s)
        return PredictionResult(image_id=image_id, source=source,
                                width=int(image.shape[1]), height=int(image.shape[0]))

    def close(self) -> None:
        self.events.append("close")


class BenchmarkTests(helpers.TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.project = helpers.make_project(self.tmp_path / "proj", n_train=3, n_val=6)
        init = Path(self.project.root) / "weights" / "yolo11s.pt"
        init.parent.mkdir(parents=True, exist_ok=True)
        init.write_bytes(b"synthetic-pretrained")
        config = self.project.config({
            "train.epochs": 2,
            "train.close_mosaic": 0,
            "benchmark.num_images": 3,
            "benchmark.warmup": 2,
            "benchmark.repeats": 2,
            "benchmark.seed": 0,
        })
        self.run: RunContext = open_run(config, "bench_run")
        begin_stage(self.run, "audit")
        audit = audit_dataset(config)
        save_audit(audit, self.run)
        complete_stage(self.run, "audit", status=audit["status"],
                       manifest_digest=audit["manifest_digest"])
        self.data_yaml = prepare_dataset_view(audit, self.run, config)
        self.config = config
        self.weights = make_weights(self.tmp_path)

    def _factory(self, **kwargs: Any):
        created: list[_FakeDetector] = []

        def factory(weights: Any, device: str) -> _FakeDetector:
            detector = _FakeDetector(weights, device, **kwargs)
            created.append(detector)
            return detector

        return factory, created

    def test_c06_timing_covers_only_predict(self) -> None:
        factory, created = self._factory()
        summary = benchmark.run_benchmark(
            self.config, self.run, self.weights,
            data_yaml=self.data_yaml, detector_factory=factory,
        )
        self.assertEqual(len(created), 1, "the detector must be built exactly once")
        detector = created[0]
        # Load happens first and outside the timer; every timed call is a predict.
        self.assertEqual(detector.events[0], "load")
        self.assertEqual(detector.events.count("predict"), 2 + 2 * 3)
        self.assertEqual(detector.events[-1], "close")
        # predict() only ever sees decoded arrays: decoding is outside the timer.
        for image in detector.images:
            self.assertIsInstance(image, np.ndarray)
        rows = helpers.read_csv_rows(self.run.benchmark_dir / "timings.csv")
        self.assertEqual(len(rows), 6)
        for row in rows:
            ms = float(row["ms"])
            self.assertGreaterEqual(ms, 3.0, "the timed window must contain the predict call")
            self.assertLess(ms, 25.0, "model loading must not be inside the timed window")
        self.assertEqual(summary["batch"], 1)
        self.assertFalse(summary["half"])
        self.assertIn("excludes weight loading", summary["timing_scope"])
        self.assertGreaterEqual(summary["p95_ms"], summary["p50_ms"])
        self.assertAlmostEqual(summary["fps"], 1000.0 / summary["mean_ms"], places=6)

    def test_c06_artefacts_match_the_returned_summary(self) -> None:
        factory, _ = self._factory()
        summary = benchmark.run_benchmark(
            self.config, self.run, self.weights,
            data_yaml=self.data_yaml, detector_factory=factory,
        )
        directory = self.run.benchmark_dir
        for name in ("image_list.json", "timings.csv", "summary.json"):
            self.assertTrue((directory / name).is_file(), f"{name} missing")
        written = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        for key in ("run_id", "weights_sha256", "num_images", "warmup", "repeats",
                    "mean_ms", "p50_ms", "p95_ms", "fps", "imgsz", "conf", "iou", "max_det",
                    "device", "effective_device", "timing_scope", "status"):
            self.assertIn(key, written, f"summary.json is missing {key}")
        self.assertEqual(written["num_images"], summary["num_images"])
        self.assertEqual(written["mean_ms"], summary["mean_ms"])
        self.assertEqual(written["status"], "ok")
        self.assertEqual(written["weights_sha256"], helpers.sha_tree(self.weights.parent)["best.pt"])
        self.assertEqual(written["device"], "cpu")
        self.assertEqual(written["effective_device"], "cpu")
        self.assertEqual(len(written["per_repeat"]), 2)
        self.assertIn("device", written["hardware"])
        image_list = json.loads((directory / "image_list.json").read_text(encoding="utf-8"))
        self.assertEqual(len(image_list["images"]), 3)
        self.assertEqual(image_list["seed"], 0)

    def test_c06_image_selection_is_deterministic_and_from_the_view(self) -> None:
        factory, _ = self._factory()
        first = benchmark.run_benchmark(self.config, self.run, self.weights,
                                        data_yaml=self.data_yaml, detector_factory=factory)
        second = benchmark.run_benchmark(self.config, self.run, self.weights,
                                         data_yaml=self.data_yaml, detector_factory=factory)
        image_list = json.loads(
            (self.run.benchmark_dir / "image_list.json").read_text(encoding="utf-8"))
        ids = [item["image_id"] for item in image_list["images"]]
        view_ids = sorted(entry["image_id"] for entry in
                          view_split_entries(self.data_yaml, "val"))
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), 3)
        self.assertTrue(set(ids) <= set(view_ids))
        self.assertEqual(first["mean_ms"] > 0, second["mean_ms"] > 0)
        again = benchmark.run_benchmark(self.config, self.run, self.weights,
                                        data_yaml=self.data_yaml, detector_factory=factory)
        repeat_ids = [item["image_id"] for item in json.loads(
            (self.run.benchmark_dir / "image_list.json").read_text(encoding="utf-8"))["images"]]
        self.assertEqual(ids, repeat_ids, "the same seed must select the same images")
        self.assertTrue(again["mean_ms"] > 0)

    def test_short_of_the_requested_image_count_is_reported(self) -> None:
        short_config = self.project.config({
            "train.epochs": 2,
            "train.close_mosaic": 0,
            "benchmark.num_images": 10,   # the view only has 6 val images
            "benchmark.warmup": 1,
            "benchmark.repeats": 1,
            "benchmark.seed": 0,
        })
        factory, _ = self._factory()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            summary = benchmark.run_benchmark(
                short_config, self.run, self.weights,
                data_yaml=self.data_yaml, detector_factory=factory,
            )
        self.assertEqual(summary["requested_images"], 10)
        self.assertEqual(summary["available_images"], 6)
        self.assertEqual(summary["num_images"], 6)
        self.assertTrue(summary["warnings"], "a smaller protocol must be reported")
        self.assertTrue(any("only 6 image" in text for text in summary["warnings"]))
        written = json.loads(
            (self.run.benchmark_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(written["requested_images"], 10)
        self.assertTrue(written["warnings"])
        # Visible in the CLI output without changing cli.py.
        self.assertIn("benchmark.num_images=10", buffer.getvalue())
        events = self.run.manifest.get("events") or []
        self.assertTrue(any(event.get("kind") == "benchmark_shortfall" for event in events))

    def test_full_protocol_has_no_shortfall_warning(self) -> None:
        factory, _ = self._factory()
        summary = benchmark.run_benchmark(
            self.config, self.run, self.weights,
            data_yaml=self.data_yaml, detector_factory=factory,
        )
        self.assertEqual(summary["requested_images"], 3)
        self.assertEqual(summary["available_images"], 6)
        self.assertEqual([text for text in summary["warnings"] if "num_images=" in text], [])

    def test_benchmark_rejects_missing_weights(self) -> None:
        factory, _ = self._factory()
        with self.assertRaises(WeightError):
            benchmark.run_benchmark(self.config, self.run, self.tmp_path / "nope.pt",
                                    data_yaml=self.data_yaml, detector_factory=factory)


# ---------------------------------------------------------------------------
# wiring hooks required by the integration test
# ---------------------------------------------------------------------------
class FactoryHookTests(helpers.TempDirTestCase):
    def test_default_factories_are_exposed_and_do_not_load_eagerly(self) -> None:
        self.assertTrue(callable(training.default_yolo_factory))
        self.assertTrue(callable(evaluation.default_adapter))
        self.assertTrue(callable(benchmark.default_detector_factory))
        self.assertTrue(callable(inference.default_model_factory))

        detector = benchmark.default_detector_factory(make_weights(self.tmp_path), "cpu")
        self.assertIsInstance(detector, Detector)
        self.assertFalse(detector.loaded, "constructing a Detector must not load weights")

    def test_default_adapter_is_weighted_to_the_native_validator(self) -> None:
        import inspect

        signature = inspect.signature(evaluation.default_adapter)
        self.assertEqual(list(signature.parameters)[:4],
                         ["config", "weights", "data_yaml", "split"])
        self.assertIn("record", signature.parameters)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
