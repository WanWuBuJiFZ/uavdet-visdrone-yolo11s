"""Training and evaluation behaviour (planning checklist B01-B08).

Every fixture here is synthetic: tiny generated PNGs, a temporary run directory
and fake Ultralytics objects from :mod:`tests.helpers`. The numbers are test
logic, never model-performance evidence.
"""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from tests import helpers
from uavdet import evaluation, training
from uavdet import ultralytics_adapter as adapter
from uavdet.artifacts import (
    begin_stage,
    complete_stage,
    fail_stage,
    open_run,
    read_json,
)
from uavdet.data import audit_dataset, prepare_dataset_view, save_audit, view_split_entries
from uavdet.devices import resolve_device
from uavdet.types import (
    CLASS_NAMES,
    NUM_CLASSES,
    AdapterError,
    EvaluationError,
    RunError,
    WeightError,
)


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------
def _tensor(values: Any, shape: tuple[int, ...]):
    """torch tensor like the native validator passes, numpy when torch is absent."""
    array = np.asarray(values, dtype=np.float32).reshape(shape)
    try:
        import torch  # noqa: PLC0415 - test-only, optional

        return torch.as_tensor(array)
    except Exception:  # noqa: BLE001 - numpy fallback keeps the test runnable
        return array


def make_init_weights(project: helpers.TempProject, name: str = "yolo11s.pt") -> Path:
    """The COCO start file the config points at (content is irrelevant here)."""
    path = Path(project.root) / "weights" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic-pretrained-checkpoint")
    return path


def prepare_run(
    project: helpers.TempProject,
    run_id: str = "train_run",
    *,
    config_extra: Mapping[str, Any] | None = None,
    epochs: int = 3,
) -> tuple[dict[str, Any], Any, Path]:
    """Audit + dataset view, i.e. exactly what the CLI does before ``train``."""
    # A short test run must also shorten close_mosaic (the config cross-validates it).
    overrides = {"train.epochs": int(epochs),
                 "train.close_mosaic": 0 if int(epochs) < 10 else 10}
    overrides.update(dict(config_extra or {}))
    config = project.config(overrides)
    run = open_run(config, run_id)
    begin_stage(run, "audit")
    audit = audit_dataset(config)
    save_audit(audit, run)
    complete_stage(
        run,
        "audit",
        status=audit["status"],
        manifest_digest=audit["manifest_digest"],
        num_images=audit["totals"]["num_images"],
    )
    data_yaml = prepare_dataset_view(audit, run, config)
    return config, run, Path(data_yaml)


class _ShortYOLO(helpers.RecordingYOLO):
    """Records the call but only produces ``epochs`` rows (an unfinished run)."""

    def __init__(self, weights: Any, produced_epochs: int = 2, **kwargs: Any) -> None:
        super().__init__(weights, **kwargs)
        self.produced_epochs = int(produced_epochs)

    def train(self, **kwargs: Any) -> Any:
        patched = dict(kwargs)
        patched["epochs"] = self.produced_epochs
        return super().train(**patched)


class _BadCsvYOLO(helpers.RecordingYOLO):
    """Writes a results.csv whose epoch column repeats / moves backwards."""

    def train(self, **kwargs: Any) -> Any:
        result = super().train(**kwargs)
        self.save_dir.joinpath("results.csv").write_text(
            "epoch,time\n1,10\n2,20\n2,30\n3,40\n", encoding="utf-8"
        )
        return result


# ---------------------------------------------------------------------------
# B03 - start weights
# ---------------------------------------------------------------------------
class InitWeightsTests(helpers.TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.project = helpers.make_project(self.tmp_path / "proj")
        self.config = self.project.config()

    def test_missing_file_is_a_weight_error(self) -> None:
        missing = Path(self.config["train"]["init_weights"])
        self.assertFalse(missing.exists())
        with self.assertRaises(WeightError) as raised:
            training.check_init_weights(self.config)
        self.assertIn("not found", str(raised.exception))
        self.assertIn("random initialisation", str(raised.exception))

    def test_empty_file_is_a_weight_error(self) -> None:
        path = Path(self.config["train"]["init_weights"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        with self.assertRaises(WeightError):
            training.check_init_weights(self.config)

    def test_coco_named_file_is_accepted_and_recorded(self) -> None:
        path = make_init_weights(self.project)
        result = training.check_init_weights(self.config)
        self.assertEqual(result["kind"], "pretrained-coco")
        self.assertEqual(Path(result["path"]).resolve(), path.resolve())
        self.assertEqual(result["sha256"], helpers.sha_tree(path.parent)[path.name])
        # Metadata of a fake file cannot be verified: that is a warning, not a guess.
        self.assertTrue(result["warnings"])
        self.assertIsNone(result["num_classes"])

    def test_best_pt_without_flag_is_rejected(self) -> None:
        best = Path(self.project.root) / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"synthetic-old-best")
        config = self.project.config({"train.init_weights": str(best)})
        with self.assertRaises(WeightError) as raised:
            training.check_init_weights(config)
        self.assertIn("best.pt", str(raised.exception))
        self.assertIn("allow_trained_init", str(raised.exception))

    def test_best_pt_with_config_flag_is_allowed_and_warned(self) -> None:
        best = Path(self.project.root) / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"synthetic-old-best")
        config = self.project.config(
            {"train.init_weights": str(best), "train.allow_trained_init": True}
        )
        result = training.check_init_weights(config)
        self.assertEqual(result["kind"], "trained")
        self.assertTrue(any("allow_trained" in text for text in result["warnings"]))

    def test_explicit_allow_trained_argument_wins(self) -> None:
        best = Path(self.project.root) / "weights" / "last.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"synthetic-old-last")
        config = self.project.config({"train.init_weights": str(best)})
        self.assertEqual(training.check_init_weights(config, allow_trained=True)["kind"], "trained")

    def test_checkpoint_with_visdrone_class_names_is_rejected(self) -> None:
        try:
            import torch
        except Exception:  # pragma: no cover - torch is a core dependency
            self.skipTest("torch is not importable")
        path = Path(self.project.root) / "weights" / "renamed_coco.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        model = SimpleNamespace(
            names={index: name for index, name in enumerate(CLASS_NAMES)},
            yaml={"nc": NUM_CLASSES},
        )
        torch.save({"model": model, "epoch": 299}, str(path))
        config = self.project.config({"train.init_weights": str(path)})
        with self.assertRaises(WeightError) as raised:
            training.check_init_weights(config)
        self.assertIn("VisDrone class names", str(raised.exception))

    def test_ten_class_checkpoint_is_allowed_with_flag(self) -> None:
        try:
            import torch
        except Exception:  # pragma: no cover - torch is a core dependency
            self.skipTest("torch is not importable")
        path = Path(self.project.root) / "weights" / "renamed_coco.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        model = SimpleNamespace(names={index: name for index, name in enumerate(CLASS_NAMES)})
        torch.save({"model": model}, str(path))
        config = self.project.config(
            {"train.init_weights": str(path), "train.allow_trained_init": True}
        )
        result = training.check_init_weights(config)
        self.assertEqual(result["kind"], "trained")
        self.assertEqual(result["num_classes"], NUM_CLASSES)


# ---------------------------------------------------------------------------
# B01 / B04 / B08 - train_model
# ---------------------------------------------------------------------------
class TrainModelTests(helpers.TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.project = helpers.make_project(self.tmp_path / "proj", n_train=4, n_val=3)
        make_init_weights(self.project)
        self.config, self.run, self.data_yaml = prepare_run(self.project, epochs=3)

    def _factory(self, cls: type = helpers.RecordingYOLO, **kwargs: Any):
        created: list[helpers.RecordingYOLO] = []

        def factory(weights: Any) -> helpers.RecordingYOLO:
            model = cls(weights, **kwargs)
            created.append(model)
            return model

        return factory, created

    def test_b01_every_configured_parameter_is_passed(self) -> None:
        factory, created = self._factory()
        result = training.train_model(self.config, self.run, yolo_factory=factory)
        self.assertEqual(len(created), 1)
        kwargs = created[0].train_kwargs
        train = self.config["train"]

        self.assertEqual(Path(created[0].weights), Path(train["init_weights"]))
        expected = {
            "epochs": int(train["epochs"]),
            "imgsz": int(train["imgsz"]),
            "batch": int(train["batch"]),
            "workers": int(train["workers"]),
            "device": str(train["device"]),
            "optimizer": str(train["optimizer"]),
            "lr0": float(train["lr0"]),
            "lrf": float(train["lrf"]),
            "momentum": float(train["momentum"]),
            "weight_decay": float(train["weight_decay"]),
            "warmup_epochs": float(train["warmup_epochs"]),
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "cos_lr": bool(train["cos_lr"]),
            "seed": int(train["seed"]),
            "deterministic": bool(train["deterministic"]),
            "amp": bool(train["amp"]),
            "mosaic": float(train["mosaic"]),
            "close_mosaic": int(train["close_mosaic"]),
            "hsv_h": float(train["hsv_h"]),
            "hsv_s": float(train["hsv_s"]),
            "hsv_v": float(train["hsv_v"]),
            "translate": float(train["translate"]),
            "scale": float(train["scale"]),
            "fliplr": float(train["fliplr"]),
            "flipud": float(train["flipud"]),
            "mixup": float(train["mixup"]),
            "copy_paste": float(train["copy_paste"]),
            "patience": int(train["patience"]),
            "save_period": int(train["save_period"]),
            "plots": bool(train["plots"]),
            "project": str(self.run.run_dir),
            "name": "train",
            "exist_ok": True,
            "data": str(self.data_yaml),
            "val": True,
            "iou": float(self.config["eval"]["iou"]),
            "max_det": int(self.config["eval"]["max_det"]),
            "verbose": True,
        }
        for key, value in expected.items():
            self.assertIn(key, kwargs, f"train() did not receive '{key}'")
            self.assertEqual(kwargs[key], value, f"train('{key}') got {kwargs[key]!r}")
        self.assertEqual(kwargs["multi_scale"], float(bool(train["multi_scale"])))
        for forbidden in ("erasing", "auto_augment", "degrees", "shear", "perspective"):
            self.assertNotIn(forbidden, kwargs)

    def test_b01_artefacts_are_copied_into_the_run(self) -> None:
        factory, _ = self._factory()
        result = training.train_model(self.config, self.run, yolo_factory=factory)
        for name in ("args.yaml", "results.csv", "weights/best.pt", "weights/last.pt", "train.log"):
            self.assertTrue((self.run.train_dir / name).is_file(), f"{name} missing after training")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["save_dir"], "train")
        self.assertEqual(result["results_csv"], "train/results.csv")
        self.assertEqual(result["args_yaml"], "train/args.yaml")
        self.assertEqual(result["best_weights"], "train/weights/best.pt")
        self.assertEqual(result["last_weights"], "train/weights/last.pt")
        self.assertEqual(result["epochs_done"], result["epochs_target"])
        self.assertEqual(result["epochs_target"], 3)
        self.assertFalse(result["resume"])
        self.assertIsNone(result["resumed_from"])
        self.assertEqual(result["device"]["effective"], "cpu")
        self.assertEqual(result["weights_sha256"]["best"],
                         helpers.sha_tree(self.run.train_dir)["weights/best.pt"])

    def test_b01_actual_trainer_save_dir_is_used(self) -> None:
        """Wherever the trainer really wrote, the products end up in run.train_dir."""
        elsewhere = self.tmp_path / "somewhere_else"

        class _ElsewhereYOLO(helpers.RecordingYOLO):
            def train(self, **kwargs: Any) -> Any:
                patched = dict(kwargs)
                patched["project"] = str(elsewhere)
                return super().train(**patched)

        factory, created = self._factory(_ElsewhereYOLO)
        result = training.train_model(self.config, self.run, yolo_factory=factory)
        self.assertEqual(Path(result["save_dir"]).resolve(), (elsewhere / "train").resolve())
        self.assertTrue((elsewhere / "train" / "results.csv").is_file())
        for name in ("args.yaml", "results.csv", "train.log", "weights/best.pt", "weights/last.pt"):
            self.assertTrue((self.run.train_dir / name).is_file(), f"{name} was not copied")
        self.assertEqual(training.read_results_epochs(self.run.train_dir / "results.csv"), [1, 2, 3])

    def test_b04_resume_outside_the_run_is_rejected(self) -> None:
        outside = Path(self.project.root) / "weights" / "last.pt"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_bytes(b"synthetic")
        factory, _ = self._factory()
        with self.assertRaises(RunError) as raised:
            training.train_model(self.config, self.run, resume=outside, yolo_factory=factory)
        self.assertIn("not inside this run", str(raised.exception))

    def test_b04_resume_missing_checkpoint_is_rejected(self) -> None:
        factory, _ = self._factory()
        with self.assertRaises(RunError):
            training.train_model(
                self.config, self.run,
                resume=self.run.train_dir / "weights" / "last.pt",
                yolo_factory=factory,
            )

    def test_b04_resume_uses_the_same_run_checkpoint_without_adding_epochs(self) -> None:
        factory, _ = self._factory()
        training.train_model(self.config, self.run, yolo_factory=factory)
        checkpoint = self.run.train_dir / "weights" / "last.pt"
        self.assertTrue(checkpoint.is_file())

        resume_factory, created = self._factory()
        result = training.train_model(
            self.config, self.run, resume=checkpoint, yolo_factory=resume_factory
        )
        self.assertEqual(Path(created[0].weights), checkpoint.resolve())
        self.assertIs(created[0].train_kwargs["resume"], True)
        # The target never becomes "already finished epochs + configured epochs".
        self.assertEqual(created[0].train_kwargs["epochs"], 3)
        self.assertEqual(result["epochs_target"], 3)
        self.assertTrue(result["resume"])
        self.assertEqual(result["resumed_from"], "train/weights/last.pt")
        epochs = training.read_results_epochs(self.run.train_dir / "results.csv")
        self.assertEqual(epochs, sorted(set(epochs)))
        self.assertEqual(len(epochs), 3)

    def test_b04_non_monotonic_results_csv_is_rejected(self) -> None:
        factory, _ = self._factory(_BadCsvYOLO)
        with self.assertRaises(RunError) as raised:
            training.train_model(self.config, self.run, yolo_factory=factory)
        self.assertIn("repeats the epoch", str(raised.exception))

    def test_b04_unfinished_training_raises_but_keeps_artefacts(self) -> None:
        factory, _ = self._factory(_ShortYOLO, produced_epochs=2)
        with self.assertRaises(RunError) as raised:
            training.train_model(self.config, self.run, yolo_factory=factory)
        message = str(raised.exception)
        self.assertIn("NOT complete", message)
        self.assertIn("2 of the configured 3", message)
        self.assertTrue((self.run.train_dir / "results.csv").is_file())
        self.assertTrue((self.run.train_dir / "weights" / "best.pt").is_file())
        self.assertEqual(training.read_results_epochs(self.run.train_dir / "results.csv"), [1, 2])

    def test_b08_training_failure_raises_and_never_claims_complete(self) -> None:
        factory, _ = self._factory(fail_train=True)
        begin_stage(self.run, "train")
        with self.assertRaises(RuntimeError):
            try:
                training.train_model(self.config, self.run, yolo_factory=factory)
            except RuntimeError as exc:
                fail_stage(self.run, "train", str(exc))
                raise
        self.assertNotEqual(self.run.stage_status("train"), "complete")
        self.assertEqual(self.run.stage_status("train"), "failed")
        self.assertFalse((self.run.train_dir / "weights" / "best.pt").exists())

    def test_load_failure_becomes_a_weight_error(self) -> None:
        factory, _ = self._factory(fail_load=True)
        with self.assertRaises(WeightError):
            training.train_model(self.config, self.run, yolo_factory=factory)


# ---------------------------------------------------------------------------
# B05 / B06 / B07 - exporter unit behaviour
# ---------------------------------------------------------------------------
class AdapterExtractionTests(unittest.TestCase):
    def test_b05_per_class_maps_by_class_id(self) -> None:
        valid = [0, 1, 2, 3, 4, 5, 6, 8]
        metrics = helpers.FakeMetrics()
        box = metrics.box
        positions = np.arange(len(valid), dtype=float)
        box.ap_class_index = np.asarray(valid, dtype=int)
        box.p = 0.50 + 0.01 * positions
        box.r = 0.40 + 0.01 * positions
        box.ap50 = 0.30 + 0.01 * positions
        box.ap = 0.20 + 0.01 * positions
        box.all_ap = np.tile((0.20 + 0.01 * positions).reshape(-1, 1), (1, 10))
        metrics.nt_per_class = np.array([10 if index in valid else 0 for index in range(NUM_CLASSES)])

        rows = adapter.extract_per_class(metrics)
        by_id = {row["class_id"]: row for row in rows}
        self.assertEqual([row["class_id"] for row in rows], list(range(NUM_CLASSES)))
        # position 7 in the native arrays belongs to class 8, not to class 7.
        self.assertAlmostEqual(by_id[8]["precision"], 0.57, places=9)
        self.assertAlmostEqual(by_id[8]["ap50_95"], 0.27, places=9)
        self.assertAlmostEqual(by_id[6]["precision"], 0.56, places=9)
        self.assertIsNone(by_id[7]["precision"])
        self.assertIsNone(by_id[7]["recall"])
        self.assertIsNone(by_id[9]["precision"])
        self.assertEqual(by_id[7]["num_gt"], 0)
        self.assertEqual(by_id[7]["class_name"], CLASS_NAMES[7])

    def test_b06_no_gt_class_is_none_not_zero(self) -> None:
        metrics = helpers.FakeMetrics(valid=[0, 1, 2])
        rows = adapter.extract_per_class(metrics)
        for index in range(NUM_CLASSES):
            row = rows[index]
            if index < 3:
                self.assertIsNotNone(row["precision"], f"class {index} should have a value")
            else:
                for key in ("precision", "recall", "ap50", "ap50_95"):
                    self.assertIsNone(row[key], f"class {index}.{key} must be N/A, not 0")

    def test_b05_curves_keep_class_order_and_mean_row(self) -> None:
        metrics = helpers.FakeMetrics()
        exported = adapter.extract_curves(metrics)
        self.assertIsNotNone(exported)
        arrays, meta = exported
        for key in ("recall", "conf", "pr_precision", "f1", "p_conf", "r_conf", "valid_classes"):
            self.assertIn(key, arrays)
        points = arrays["recall"].size
        self.assertEqual(points, 50)
        self.assertEqual(arrays["conf"].shape, (points,))
        for key in ("pr_precision", "f1", "p_conf", "r_conf"):
            self.assertEqual(arrays[key].shape, (NUM_CLASSES + 1, points), key)
        self.assertEqual(arrays["valid_classes"].shape, (NUM_CLASSES,))

        # Class rows follow the native per-class curve of the same class id.
        recall_axis = arrays["recall"]
        for class_id in (0, 4, 9):
            expected = recall_axis * (0.9 - 0.01 * class_id)
            np.testing.assert_allclose(arrays["pr_precision"][class_id], expected, rtol=0, atol=1e-12)
        # ... and the last row is the mean of the class rows.
        np.testing.assert_allclose(
            arrays["pr_precision"][NUM_CLASSES],
            arrays["pr_precision"][:NUM_CLASSES].mean(axis=0),
            atol=1e-12,
        )
        self.assertEqual(meta["mean_row_index"], NUM_CLASSES)
        self.assertEqual(meta["class_order"], list(CLASS_NAMES))
        self.assertEqual(meta["valid_classes"], list(range(NUM_CLASSES)))
        self.assertIn("no extra smoothing", meta["smoothing"])
        self.assertTrue(isinstance(meta["source"]["version"], str) and meta["source"]["version"])

    def test_b05_missing_curves_return_none_instead_of_zeros(self) -> None:
        bare = SimpleNamespace(names={index: name for index, name in enumerate(CLASS_NAMES)},
                               box=SimpleNamespace())
        self.assertIsNone(adapter.extract_curves(bare))

    def test_b05_confusion_is_exported_true_by_predicted(self) -> None:
        metrics = helpers.FakeMetrics()
        exported = adapter.extract_confusion(metrics, conf=0.001, iou=0.5, version="8.4.test")
        self.assertIsNotNone(exported)
        matrix, meta = exported
        native = helpers.FakeConfusionMatrix().matrix
        self.assertEqual(matrix.dtype, np.int64)
        self.assertEqual(matrix.shape, (NUM_CLASSES + 1, NUM_CLASSES + 1))
        # The native array is [predicted, true]; the contract wants [true, predicted].
        np.testing.assert_array_equal(matrix, native.T.astype(np.int64))
        self.assertEqual(matrix[1, 0], native[0, 1])
        self.assertEqual(meta["true_axis"], "row")
        self.assertEqual(meta["pred_axis"], "column")
        self.assertEqual(meta["background_index"], NUM_CLASSES)
        self.assertEqual(meta["class_order"], list(CLASS_NAMES))
        self.assertEqual(meta["conf"], 0.001)
        self.assertEqual(meta["iou"], 0.5)
        self.assertIn("raw integer counts", meta["normalization"])
        self.assertEqual(meta["source_version"], "8.4.test")

    def test_b05_missing_confusion_returns_none(self) -> None:
        bare = SimpleNamespace(names={index: name for index, name in enumerate(CLASS_NAMES)},
                               box=SimpleNamespace())
        self.assertIsNone(adapter.extract_confusion(bare))

    def test_b05_missing_per_class_arrays_raise_adapter_error(self) -> None:
        bare = SimpleNamespace(names={index: name for index, name in enumerate(CLASS_NAMES)},
                               box=SimpleNamespace())
        with self.assertRaises(Exception) as raised:
            adapter.extract_per_class(bare)
        self.assertIn("per_class.csv", str(raised.exception))

    def test_to_original_boxes_inverts_the_letterbox(self) -> None:
        # 64x48 original, letterboxed to 96x64 with gain (1.5, 1.3333...).
        ratio_pad = ((64 / 48, 96 / 64), (0, 0))
        boxes = adapter.to_original_boxes([[9, 8, 27, 24]], (64, 96), (48, 64), ratio_pad)
        np.testing.assert_allclose(boxes[0], [6.0, 6.0, 18.0, 18.0], atol=1e-6)
        clipped = adapter.to_original_boxes([[-5, -5, 200, 200]], (64, 96), (48, 64), ratio_pad)
        np.testing.assert_allclose(clipped[0], [0.0, 0.0, 64.0, 48.0], atol=1e-6)

    def test_b07_capture_batch_records_ids_and_original_pixel_coords(self) -> None:
        ratio_pad = ((64 / 48, 96 / 64), (0, 0))

        class _StubValidator:
            names = {index: name for index, name in enumerate(CLASS_NAMES)}

            def _prepare_batch(self, index: int, batch: Mapping[str, Any]) -> dict[str, Any]:
                return {
                    "cls": batch["cls"][index],
                    "bboxes": batch["bboxes"][index],
                    "ori_shape": (48, 64),
                    "imgsz": (64, 96),
                    "ratio_pad": ratio_pad,
                    "im_file": batch["im_file"][index],
                }

        batch = {
            "cls": [_tensor([0], (1,)), _tensor([], (0,))],
            "bboxes": [_tensor([[9, 8, 27, 24]], (1, 4)), _tensor([], (0, 4))],
            "im_file": ["/view/images/val/val_0000.jpg", "/view/images/val/val_0001.jpg"],
        }
        preds = [
            {"bboxes": _tensor([[9, 8, 27, 24], [30, 24, 39, 32]], (2, 4)),
             "conf": _tensor([0.9, 0.4], (2,)),
             "cls": _tensor([0, 1], (2,))},
            {"bboxes": _tensor([], (0, 4)), "conf": _tensor([], (0,)), "cls": _tensor([], (0,))},
        ]
        capture = {"predictions": [], "ground_truth": [], "warnings": [], "seen_ids": set(),
                   "view_root": Path("/view")}
        adapter.capture_batch(_StubValidator(), capture, preds, batch)

        self.assertEqual([row["image_id"] for row in capture["ground_truth"]],
                         ["val_0000", "val_0001"])
        self.assertEqual([row["image_id"] for row in capture["predictions"]],
                         ["val_0000", "val_0001"])
        first_gt = capture["ground_truth"][0]
        self.assertEqual(first_gt["source_rel"], "images/val/val_0000.jpg")
        self.assertEqual((first_gt["width"], first_gt["height"]), (64, 48))
        self.assertEqual(first_gt["num_objects"], 1)
        np.testing.assert_allclose(first_gt["objects"][0]["xyxy"], [6.0, 6.0, 18.0, 18.0], atol=1e-5)
        self.assertEqual(first_gt["objects"][0]["class_name"], CLASS_NAMES[0])
        # The detected count is the post-NMS prediction count, not the GT count.
        self.assertEqual(capture["predictions"][0]["num_detections"], 2)
        np.testing.assert_allclose(
            capture["predictions"][0]["detections"][1]["xyxy"],
            [20.0, 18.0, 26.0, 24.0], atol=1e-5,
        )
        # Zero-object / zero-detection images still produce a line each.
        self.assertEqual(capture["ground_truth"][1]["num_objects"], 0)
        self.assertEqual(capture["predictions"][1]["num_detections"], 0)
        self.assertEqual(capture["predictions"][1]["detections"], [])
        self.assertEqual(capture["warnings"], [])


# ---------------------------------------------------------------------------
# B02 - what the native validation call actually receives
# ---------------------------------------------------------------------------
class ValidationCallTests(helpers.TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.project = helpers.make_project(self.tmp_path / "proj", n_train=3, n_val=3)
        make_init_weights(self.project)
        self.config, self.run, self.data_yaml = prepare_run(self.project, epochs=2)
        self.weights = self.run.train_dir / "weights" / "best.pt"
        self.weights.parent.mkdir(parents=True, exist_ok=True)
        self.weights.write_bytes(b"synthetic-best")

    def _factory(self, **kwargs: Any):
        created: list[helpers.RecordingYOLO] = []

        def factory(weights: Any) -> helpers.RecordingYOLO:
            model = helpers.RecordingYOLO(weights, **kwargs)
            created.append(model)
            return model

        return factory, created

    def test_b02_val_arguments_are_passed_explicitly(self) -> None:
        factory, created = self._factory()
        outcome = adapter.run_native_validation(
            self.config, self.weights, self.data_yaml, "val",
            record=False, yolo_factory=factory,
        )
        self.assertEqual(len(created), 1)
        self.assertEqual(Path(created[0].weights), self.weights.resolve())
        kwargs = created[0].val_kwargs
        evaluation_cfg = self.config["eval"]
        self.assertEqual(kwargs["data"], str(self.data_yaml))
        self.assertEqual(kwargs["split"], "val")
        self.assertEqual(kwargs["imgsz"], int(evaluation_cfg["imgsz"]))
        self.assertEqual(kwargs["conf"], float(evaluation_cfg["conf"]))
        self.assertEqual(kwargs["iou"], float(evaluation_cfg["iou"]))
        self.assertEqual(kwargs["max_det"], int(evaluation_cfg["max_det"]))
        self.assertEqual(kwargs["batch"], int(evaluation_cfg["batch"]))
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["name"], "val_best")
        self.assertEqual(Path(kwargs["project"]),
                         Path(self.config["project"]["workspace_root"]) / "val_runs")
        self.assertTrue(kwargs["exist_ok"])
        self.assertFalse(kwargs["verbose"])
        self.assertIsNotNone(outcome.metrics)
        self.assertEqual(outcome.image_ids, [])
        self.assertTrue(any("record=False" in text for text in outcome.warnings))
        self.assertGreater(outcome.speed["inference"], 0)
        self.assertEqual(outcome.evaluator_name, "ultralytics.DetectionValidator")

    def test_b02_plots_are_forced_on_when_the_confusion_matrix_is_needed(self) -> None:
        config = self.project.config({
            "train.epochs": 2, "train.close_mosaic": 0, "eval.plots": False,
        })
        factory, created = self._factory()
        with self.assertRaises(AdapterError):
            adapter.run_native_validation(
                config, self.weights, self.data_yaml, "val",
                record=True, yolo_factory=factory,
            )
        self.assertTrue(created[0].val_kwargs["plots"],
                        "the native confusion matrix is only built with plots=True")

    def test_record_true_without_the_validator_hook_raises(self) -> None:
        factory, _ = self._factory()
        # RecordingYOLO.val() never instantiates the validator: no same-pass evidence.
        with self.assertRaises(AdapterError) as raised:
            adapter.run_native_validation(
                self.config, self.weights, self.data_yaml, "val",
                record=True, yolo_factory=factory,
            )
        self.assertIn("per-image predictions", str(raised.exception))

    def test_missing_weights_are_rejected_before_any_validation(self) -> None:
        factory, created = self._factory()
        with self.assertRaises(WeightError):
            adapter.run_native_validation(
                self.config, self.tmp_path / "nope.pt", self.data_yaml, "val",
                record=False, yolo_factory=factory,
            )
        self.assertEqual(created, [])


# ---------------------------------------------------------------------------
# validation output must stay inside the run directory
# ---------------------------------------------------------------------------
class ValidationOutputLocationTests(helpers.TempDirTestCase):
    """``val()`` output must land in ``outputs/<run>/evaluation``, not in runs/.

    The legacy project passed project/name to ``validate()`` and the output path
    was not effective; this test pins the argument *and* what the native
    validator would create from it.
    """

    def setUp(self) -> None:
        super().setUp()
        self.project = helpers.make_project(self.tmp_path / "proj", n_train=3, n_val=3)
        make_init_weights(self.project)
        self.config, self.run, self.data_yaml = prepare_run(self.project, epochs=2)
        self.weights = self.run.train_dir / "weights" / "best.pt"
        self.weights.parent.mkdir(parents=True, exist_ok=True)
        self.weights.write_bytes(b"synthetic-best")

    def test_native_validation_writes_only_inside_the_run(self) -> None:
        created: list[Path] = []

        class _DirCreatingYOLO(helpers.RecordingYOLO):
            """Mimics the native validator: it writes into project/name."""

            def val(self, **kwargs: Any) -> Any:
                target = Path(kwargs["project"]) / str(kwargs["name"])
                target.mkdir(parents=True, exist_ok=True)
                (target / "confusion_matrix.png").write_bytes(b"native-plot")
                created.append(target)
                return super().val(**kwargs)

        factory_holder: list[Any] = []

        def factory(weights: Any) -> Any:
            model = _DirCreatingYOLO(weights)
            factory_holder.append(model)
            return model

        adapter.run_native_validation(
            self.config, self.weights, self.data_yaml, "val",
            record=False, yolo_factory=factory, save_dir=self.run.evaluation_dir,
        )
        self.assertEqual(len(created), 1)
        target = created[0].resolve()
        evaluation_dir = self.run.evaluation_dir.resolve()
        self.assertTrue(
            evaluation_dir in target.parents,
            f"native validation wrote outside the run: {target}",
        )
        self.assertEqual(target.name, adapter.NATIVE_VAL_DIRNAME)
        workspace_val = Path(self.config["project"]["workspace_root"]) / "val_runs"
        self.assertFalse(workspace_val.exists(),
                         "the validation must not create a directory outside the run")

    def test_validation_arguments_point_at_the_run_evaluation_dir(self) -> None:
        arguments = adapter.validation_arguments(
            self.config, self.weights, self.data_yaml, "val", "cpu",
            record=True, save_dir=self.run.evaluation_dir,
        )
        self.assertEqual(Path(arguments["project"]), self.run.evaluation_dir)
        self.assertEqual(arguments["name"], adapter.NATIVE_VAL_DIRNAME)
        self.assertTrue(arguments["exist_ok"])

    def test_evaluate_model_hands_the_run_dir_to_the_default_adapter(self) -> None:
        seen: dict[str, Any] = {}
        original = evaluation.default_adapter

        def fake_default_adapter(config: Any, weights: Any, data_yaml: Any,
                                 split: str = "val", *, record: bool = True,
                                 save_dir: Any = None) -> Any:
            seen.update({"split": split, "record": record, "save_dir": save_dir})
            return fake_outcome(data_yaml, self.run)

        evaluation.default_adapter = fake_default_adapter
        try:
            result = evaluation.evaluate_model(
                self.config, self.run, self.weights, split="val", data_yaml=self.data_yaml,
            )
        finally:
            evaluation.default_adapter = original
        self.assertTrue(seen["record"])
        self.assertEqual(Path(seen["save_dir"]), self.run.evaluation_dir)
        self.assertEqual(result["status"], "ok")

    def test_replacement_adapter_without_the_save_dir_argument_still_works(self) -> None:
        """A monkeypatched default_adapter with the documented signature must work."""
        original = evaluation.default_adapter
        calls: list[str] = []

        def five_argument_adapter(config: Any, weights: Any, data_yaml: Any,
                                  split: str = "val", *, record: bool = True) -> Any:
            calls.append(split)
            return fake_outcome(data_yaml, self.run)

        evaluation.default_adapter = five_argument_adapter
        try:
            result = evaluation.evaluate_model(
                self.config, self.run, self.weights, split="val", data_yaml=self.data_yaml,
            )
        finally:
            evaluation.default_adapter = original
        self.assertEqual(calls, ["val"])
        self.assertEqual(result["status"], "ok")


# ---------------------------------------------------------------------------
# B01/B02 cross-check against the INSTALLED Ultralytics cfg
# ---------------------------------------------------------------------------
class InstalledVersionArgumentsTests(helpers.TempDirTestCase):
    """The keyword names must be real arguments of the installed version.

    ``ultralytics.cfg.get_cfg`` runs ``check_dict_alignment`` + ``check_cfg``, so
    this fails on a typo, a removed argument (e.g. ``half``) or a wrong type --
    exactly the failure mode a tutorial-based implementation would hide.
    """

    def setUp(self) -> None:
        super().setUp()
        try:
            import ultralytics  # noqa: F401,PLC0415 - optional dependency
            from ultralytics.cfg import get_cfg  # noqa: F401,PLC0415
        except Exception as exc:  # noqa: BLE001 - skip on a machine without Ultralytics
            self.skipTest(f"ultralytics is not importable: {exc}")
        self.project = helpers.make_project(self.tmp_path / "proj", n_train=3, n_val=3)
        make_init_weights(self.project)
        self.config, self.run, self.data_yaml = prepare_run(self.project, epochs=3)
        self.weights = self.run.train_dir / "weights" / "best.pt"
        self.weights.parent.mkdir(parents=True, exist_ok=True)
        self.weights.write_bytes(b"synthetic-best")

    def test_training_arguments_are_real_ultralytics_arguments(self) -> None:
        from ultralytics.cfg import get_cfg

        choice = resolve_device(self.config["train"]["device"])
        arguments = training.training_arguments(self.config, self.run, self.data_yaml, choice)
        namespace = get_cfg(overrides=arguments)
        self.assertEqual(namespace.epochs, arguments["epochs"])
        self.assertEqual(namespace.imgsz, arguments["imgsz"])
        self.assertEqual(namespace.batch, arguments["batch"])
        self.assertEqual(namespace.name, "train")
        self.assertTrue(namespace.exist_ok)
        self.assertEqual(float(namespace.multi_scale), float(arguments["multi_scale"]))
        self.assertEqual(namespace.iou, arguments["iou"])
        self.assertEqual(namespace.max_det, arguments["max_det"])
        # patience=0 must disable early stopping in the installed version.
        from ultralytics.utils.torch_utils import EarlyStopping

        self.assertEqual(EarlyStopping(patience=int(namespace.patience)).patience, float("inf"))

    def test_validation_arguments_are_real_ultralytics_arguments(self) -> None:
        from ultralytics.cfg import get_cfg

        arguments = adapter.validation_arguments(
            self.config, self.weights, self.data_yaml, "val", "cpu", record=True
        )
        namespace = get_cfg(overrides=arguments)
        self.assertEqual(namespace.split, "val")
        self.assertEqual(namespace.data, str(self.data_yaml))
        self.assertEqual(namespace.conf, float(self.config["eval"]["conf"]))
        self.assertEqual(namespace.max_det, int(self.config["eval"]["max_det"]))
        self.assertTrue(namespace.plots)

    def test_recording_validator_is_a_detection_validator_subclass(self) -> None:
        from ultralytics.models.yolo.detect import DetectionValidator

        capture = {"predictions": [], "ground_truth": [], "warnings": [], "seen_ids": set()}
        recorder = adapter.make_recording_validator(capture)
        self.assertTrue(issubclass(recorder, DetectionValidator))
        self.assertIsNot(recorder, DetectionValidator)


# ---------------------------------------------------------------------------
# fixed-threshold matching
# ---------------------------------------------------------------------------
def _row(image_id: str, detections: Sequence[Mapping[str, Any]], width: int = 64,
         height: int = 48) -> dict[str, Any]:
    return {
        "image_id": image_id, "source": f"/x/{image_id}.jpg", "source_rel": f"images/val/{image_id}.jpg",
        "width": width, "height": height, "num_detections": len(detections),
        "detections": [dict(item) for item in detections], "timing": {},
    }


def _gt_row(image_id: str, objects: Sequence[Mapping[str, Any]], width: int = 64,
            height: int = 48) -> dict[str, Any]:
    return {
        "image_id": image_id, "source": f"/x/{image_id}.jpg", "source_rel": f"images/val/{image_id}.jpg",
        "width": width, "height": height, "num_objects": len(objects),
        "objects": [dict(item) for item in objects],
    }


def _det(class_id: int, confidence: float, xyxy: Sequence[float]) -> dict[str, Any]:
    return {"class_id": class_id, "class_name": CLASS_NAMES[class_id],
            "confidence": confidence, "xyxy": [float(v) for v in xyxy]}


def _obj(class_id: int, xyxy: Sequence[float]) -> dict[str, Any]:
    return {"class_id": class_id, "class_name": CLASS_NAMES[class_id],
            "xyxy": [float(v) for v in xyxy]}


class MatchTests(unittest.TestCase):
    def test_greedy_one_to_one_class_aware(self) -> None:
        predictions = [
            _row("a", [
                _det(0, 0.90, [0, 0, 10, 10]),      # matches GT 0
                _det(0, 0.80, [0, 0, 10, 10]),      # duplicate box: no GT left -> FP
                _det(3, 0.70, [20, 20, 30, 30]),    # right box, wrong class -> FP
                _det(1, 0.20, [40, 0, 50, 10]),     # below conf -> ignored
            ]),
            _row("b", []),
        ]
        ground_truth = [
            _gt_row("a", [_obj(0, [0, 0, 10, 10]), _obj(1, [40, 0, 50, 10])]),
            _gt_row("b", []),
        ]
        match = evaluation.match_detections(predictions, ground_truth, conf=0.25, iou=0.5)
        image_a = match["images"][0]
        self.assertEqual(image_a["image_id"], "a")
        self.assertEqual((image_a["tp"], image_a["fp"], image_a["fn"]), (1, 2, 1))
        self.assertEqual(image_a["num_candidates"], 3)
        self.assertEqual(image_a["false_negative_indices"], [1])
        self.assertEqual(match["totals"]["tp"], 1)
        self.assertEqual(match["totals"]["fp"], 2)
        self.assertEqual(match["totals"]["fn"], 1)
        self.assertAlmostEqual(match["totals"]["precision"], 1 / 3)
        self.assertAlmostEqual(match["totals"]["recall"], 0.5)
        self.assertIn("not", match["definition"].lower())

    def test_mismatched_image_sets_are_rejected(self) -> None:
        with self.assertRaises(EvaluationError):
            evaluation.match_detections([_row("a", [])], [_gt_row("b", [])])

    def test_select_cases_is_deterministic_and_capped(self) -> None:
        predictions = [
            _row("img_c", [_det(0, 0.9, [0, 0, 10, 10]), _det(0, 0.8, [0, 0, 10, 10])]),
            _row("img_a", [_det(0, 0.95, [0, 0, 10, 10])]),
            _row("img_b", []),
        ]
        ground_truth = [
            _gt_row("img_c", [_obj(0, [0, 0, 10, 10])]),
            _gt_row("img_a", [_obj(0, [0, 0, 10, 10])]),
            _gt_row("img_b", [_obj(2, [5, 5, 15, 15])]),
        ]
        match = evaluation.match_detections(predictions, ground_truth, conf=0.25, iou=0.5)
        cases = evaluation.select_cases(match, max_cases=1)
        self.assertEqual([item["image_id"] for item in cases["successful"]], ["img_a"])
        self.assertEqual([item["image_id"] for item in cases["failure"]], ["img_b"])
        self.assertEqual(cases["successful"][0]["num_gt"], 1)
        self.assertIn("false_negative", cases["failure"][0]["failure_kinds"])
        self.assertEqual(cases["conf"], 0.25)
        self.assertEqual(cases["iou"], 0.5)
        again = evaluation.select_cases(match, max_cases=1)
        for key in ("successful", "failure"):
            for left, right in zip(cases[key], again[key]):
                left.pop("created_at", None)
                self.assertEqual(left, right)


# ---------------------------------------------------------------------------
# evaluate_model (B05 / B06 / B07 + the full artefact set)
# ---------------------------------------------------------------------------
def fake_outcome(
    data_yaml: Path,
    run: Any,
    *,
    metrics: Any = None,
    valid_classes: Sequence[int] | None = None,
) -> adapter.ValidationOutcome:
    """A ValidationOutcome-shaped object built from the real dataset view."""
    # The image ids/sizes come from the real dataset view, like the native pass.
    entries = view_split_entries(data_yaml, "val")
    predictions = []
    ground_truth = []
    for index, entry in enumerate(entries):
        width, height = int(entry["width"]), int(entry["height"])
        if index == 0:
            ground_truth.append(_gt_row(entry["image_id"], [_obj(0, [6, 6, 18, 18]),
                                                           _obj(3, [28, 20, 46, 36])],
                                        width=width, height=height))
            predictions.append(_row(entry["image_id"], [_det(0, 0.93, [6, 6, 18, 18]),
                                                        _det(3, 0.81, [29, 21, 47, 37])],
                                    width=width, height=height))
        elif index == 1:
            ground_truth.append(_gt_row(entry["image_id"], [], width=width, height=height))
            predictions.append(_row(entry["image_id"], [], width=width, height=height))
        else:
            ground_truth.append(_gt_row(entry["image_id"], [], width=width, height=height))
            predictions.append(_row(entry["image_id"], [_det(5, 0.66, [4, 4, 20, 20])],
                                    width=width, height=height))
    return adapter.ValidationOutcome(
        metrics=metrics if metrics is not None else helpers.FakeMetrics(),
        predictions=predictions,
        ground_truth=ground_truth,
        image_ids=[row["image_id"] for row in ground_truth],
        speed={"preprocess": 1.2, "inference": 8.5, "postprocess": 0.7, "loss": 0.0},
        evaluator_name="ultralytics.DetectionValidator",
        evaluator_version="8.4.test",
        warnings=[],
        details={"confusion_conf": 0.001, "confusion_iou": 0.5, "iou_nms": 0.7},
    )


class _Adapter:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def __call__(self, config: Any, weights: Any, data_yaml: Any, split: str = "val", *,
                 record: bool = True) -> Any:
        self.calls.append({"config": config, "weights": str(weights),
                           "data_yaml": str(data_yaml), "split": split, "record": record})
        return self.outcome


class EvaluateModelTests(helpers.TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.project = helpers.make_project(self.tmp_path / "proj", n_train=4, n_val=3)
        make_init_weights(self.project)
        self.config, self.run, self.data_yaml = prepare_run(self.project, epochs=2)
        (self.run.train_dir / "weights").mkdir(parents=True, exist_ok=True)
        self.weights = self.run.train_dir / "weights" / "best.pt"
        self.weights.write_bytes(b"synthetic-best")

    def _evaluate(self, outcome: Any = None) -> tuple[dict[str, Any], _Adapter]:
        fake = _Adapter(outcome if outcome is not None else fake_outcome(self.data_yaml, self.run))
        result = evaluation.evaluate_model(
            self.config, self.run, self.weights, split="val",
            data_yaml=self.data_yaml, adapter=fake,
        )
        return result, fake

    def test_b05_b07_all_contract_files_are_written(self) -> None:
        result, fake = self._evaluate()
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["split"], "val")
        self.assertTrue(fake.calls[0]["record"])
        directory = self.run.evaluation_dir
        for name in (*evaluation.EVALUATION_FILES, "curves.npz", "curves_meta.json",
                     "confusion_matrix.npz", "confusion_meta.json"):
            self.assertTrue((directory / name).is_file(), f"{name} missing")

        metrics = read_json(directory / "metrics.json")
        self.assertEqual(metrics["run_id"], self.run.run_id)
        self.assertEqual(metrics["scope"], "formal")
        self.assertEqual(metrics["split"], "val")
        self.assertEqual(metrics["num_images"], 3)
        self.assertEqual(metrics["num_instances"], 2)
        self.assertEqual(metrics["status"], "ok")
        self.assertAlmostEqual(metrics["map50"], 0.6)
        self.assertAlmostEqual(metrics["map50_95"], 0.4)
        self.assertAlmostEqual(metrics["precision"], 0.8)
        self.assertAlmostEqual(metrics["recall"], 0.7)
        self.assertEqual(metrics["weights_sha256"], helpers.sha_tree(self.weights.parent)["best.pt"])
        self.assertEqual(metrics["evaluator_version"], "8.4.test")
        self.assertEqual(metrics["evaluator_name"], "ultralytics.DetectionValidator")
        self.assertEqual(metrics["cases"]["conf"], 0.25)
        self.assertEqual(metrics["cases"]["iou"], 0.5)
        self.assertEqual(metrics["cases"]["tp"], 2)
        self.assertEqual(metrics["cases"]["fp"], 1)
        self.assertEqual(metrics["cases"]["fn"], 0)
        self.assertGreater(metrics["speed_ms"]["total"], 0)
        self.assertIn("not", metrics["pr_definition"].lower())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["num_images"], 3)
        self.assertEqual(result["weights_sha256"], metrics["weights_sha256"])

    def test_b07_jsonl_ids_are_stable_and_coordinates_are_original_pixels(self) -> None:
        self._evaluate()
        directory = self.run.evaluation_dir
        predictions = helpers.assert_jsonl_parses(self, directory / "predictions.jsonl")
        ground_truth = helpers.assert_jsonl_parses(self, directory / "ground_truth.jsonl")
        entries = view_split_entries(self.data_yaml, "val")
        expected_ids = [str(entry["image_id"]) for entry in entries]
        self.assertEqual([row["image_id"] for row in predictions], expected_ids)
        self.assertEqual([row["image_id"] for row in ground_truth], expected_ids)
        # One line per image, including the zero-detection one.
        self.assertEqual(len(predictions), len(entries))
        zero = [row for row in predictions if row["num_detections"] == 0]
        self.assertEqual(len(zero), 1)
        self.assertEqual(zero[0]["detections"], [])
        first = predictions[0]
        self.assertEqual((first["width"], first["height"]), (64, 48))
        self.assertEqual([det["xyxy"] for det in first["detections"]],
                         [[6.0, 6.0, 18.0, 18.0], [29.0, 21.0, 47.0, 37.0]])
        for row in predictions:
            for det in row["detections"]:
                x1, y1, x2, y2 = det["xyxy"]
                self.assertTrue(0 <= x1 <= x2 <= row["width"])
                self.assertTrue(0 <= y1 <= y2 <= row["height"])
                self.assertEqual(det["class_name"], CLASS_NAMES[det["class_id"]])
        self.assertEqual(ground_truth[0]["num_objects"], 2)
        self.assertEqual(ground_truth[0]["objects"][1]["xyxy"], [28.0, 20.0, 46.0, 36.0])

    def test_b06_per_class_uses_na_for_classes_without_gt(self) -> None:
        self._evaluate()
        rows = helpers.read_csv_rows(self.run.evaluation_dir / "per_class.csv")
        self.assertEqual(len(rows), NUM_CLASSES)
        by_id = {int(row["class_id"]): row for row in rows}
        self.assertEqual(by_id[0]["num_gt"], "1")
        self.assertEqual(by_id[3]["num_gt"], "1")
        self.assertNotEqual(by_id[0]["precision"], "N/A")
        for class_id in (2, 5, 7, 9):
            self.assertEqual(by_id[class_id]["num_gt"], "0")
            for column in ("precision", "recall", "ap50", "ap50_95"):
                self.assertEqual(by_id[class_id][column], "N/A",
                                 f"class {class_id}.{column} must be N/A, not 0")
        self.assertEqual([row["class_name"] for row in rows], list(CLASS_NAMES))

    def test_b05_curves_and_confusion_are_written_from_the_same_metrics(self) -> None:
        self._evaluate()
        directory = self.run.evaluation_dir
        with np.load(directory / "curves.npz") as data:
            for key in ("recall", "conf", "pr_precision", "f1", "p_conf", "r_conf", "valid_classes"):
                self.assertIn(key, data.files)
            self.assertEqual(data["pr_precision"].shape[0], NUM_CLASSES + 1)
            self.assertEqual(data["pr_precision"].dtype, np.float64)
        curves_meta = read_json(directory / "curves_meta.json")
        self.assertEqual(curves_meta["mean_row_index"], NUM_CLASSES)
        self.assertEqual(curves_meta["class_order"], list(CLASS_NAMES))
        self.assertEqual(curves_meta["source"]["version"], "8.4.test")
        with np.load(directory / "confusion_matrix.npz") as data:
            matrix = data["matrix"]
        self.assertEqual(matrix.dtype, np.int64)
        self.assertEqual(matrix.shape, (NUM_CLASSES + 1, NUM_CLASSES + 1))
        np.testing.assert_array_equal(matrix, helpers.FakeConfusionMatrix().matrix.T)
        confusion_meta = read_json(directory / "confusion_meta.json")
        self.assertEqual(confusion_meta["true_axis"], "row")
        self.assertEqual(confusion_meta["pred_axis"], "column")
        self.assertEqual(confusion_meta["conf"], 0.001)
        self.assertEqual(confusion_meta["iou"], 0.5)
        self.assertEqual(confusion_meta["background_index"], NUM_CLASSES)

    def test_cases_json_separates_the_fixed_threshold_analysis(self) -> None:
        self._evaluate()
        cases = read_json(self.run.evaluation_dir / "cases.json")
        self.assertEqual(cases["conf"], 0.25)
        self.assertEqual(cases["iou"], 0.5)
        self.assertEqual([item["image_id"] for item in cases["successful"]],
                         [view_split_entries(self.data_yaml, "val")[0]["image_id"]])
        self.assertEqual(len(cases["failure"]), 1)
        self.assertIn("false_positive", cases["failure"][0]["failure_kinds"])
        self.assertIn("separate from the native mAP", cases["selection_rules"]["threshold_note"])
        log = (self.run.evaluation_dir / "evaluation.log").read_text(encoding="utf-8")
        self.assertIn("NOT mAP", log)

    def test_rerun_replaces_the_jsonl_instead_of_appending(self) -> None:
        self._evaluate()
        self._evaluate()
        lines = (self.run.evaluation_dir / "predictions.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)

    def test_image_set_mismatch_is_rejected(self) -> None:
        outcome = fake_outcome(self.data_yaml, self.run)
        outcome.predictions = outcome.predictions[:2]
        outcome.ground_truth = outcome.ground_truth[:2]
        outcome.image_ids = outcome.image_ids[:2]
        with self.assertRaises(EvaluationError) as raised:
            self._evaluate(outcome)
        self.assertIn("lists 3", str(raised.exception))
        # A failed evaluation must not leave a success claim behind.
        metrics = read_json(self.run.evaluation_dir / "metrics.json")
        self.assertEqual(metrics["status"], "failed")

    def test_eight_class_metrics_are_rejected(self) -> None:
        metrics = helpers.FakeMetrics(num_classes=8)
        metrics.names = {index: CLASS_NAMES[index] for index in range(8)}
        outcome = fake_outcome(self.data_yaml, self.run, metrics=metrics)
        with self.assertRaises(WeightError) as raised:
            self._evaluate(outcome)
        self.assertIn("do not describe 10 classes", str(raised.exception))

    def test_wrong_class_order_is_rejected(self) -> None:
        metrics = helpers.FakeMetrics()
        metrics.names = {index: name for index, name in enumerate(reversed(CLASS_NAMES))}
        outcome = fake_outcome(self.data_yaml, self.run, metrics=metrics)
        with self.assertRaises(WeightError) as raised:
            self._evaluate(outcome)
        self.assertIn("class order", str(raised.exception))

    def test_adapter_failure_is_wrapped_and_logged(self) -> None:
        class _Broken:
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("native validator exploded")

        with self.assertRaises(EvaluationError) as raised:
            evaluation.evaluate_model(
                self.config, self.run, self.weights, split="val",
                data_yaml=self.data_yaml, adapter=_Broken(),
            )
        self.assertIn("native validator exploded", str(raised.exception))
        log = (self.run.evaluation_dir / "evaluation.log").read_text(encoding="utf-8")
        self.assertIn("FAILED", log)

    def test_missing_weights_are_rejected(self) -> None:
        with self.assertRaises(WeightError):
            evaluation.evaluate_model(
                self.config, self.run, self.run.train_dir / "weights" / "nope.pt",
                data_yaml=self.data_yaml, adapter=_Adapter(fake_outcome(self.data_yaml, self.run)),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class CaseFallbackTests(unittest.TestCase):
    """The relaxed success tier must be recorded, never silent."""

    def _match(self, images):
        return {"conf": 0.25, "iou": 0.5, "images": images}

    def test_strict_cases_win_and_are_labelled(self):
        from uavdet import evaluation

        match = self._match([
            {"image_id": "perfect", "num_gt": 2, "tp": 2, "fp": 0, "fn": 0,
             "max_conf": 0.9, "num_candidates": 2},
            {"image_id": "dense", "num_gt": 10, "tp": 9, "fp": 0, "fn": 1,
             "max_conf": 0.9, "num_candidates": 9},
        ])
        cases = evaluation.select_cases(match, max_cases=2)
        self.assertEqual(cases["successful_criterion"], "strict_no_fn_no_fp")
        self.assertEqual([c["image_id"] for c in cases["successful"]], ["perfect"])
        self.assertEqual(cases["successful"][0]["criterion"], "strict_no_fn_no_fp")

    def test_relaxed_tier_requires_high_precision_and_is_labelled(self):
        from uavdet import evaluation

        match = self._match([
            {"image_id": "good", "num_gt": 10, "tp": 6, "fp": 1, "fn": 4,
             "max_conf": 0.8, "num_candidates": 7},
            {"image_id": "sloppy", "num_gt": 10, "tp": 5, "fp": 5, "fn": 5,
             "max_conf": 0.7, "num_candidates": 10},
        ])
        cases = evaluation.select_cases(match, max_cases=2)
        self.assertEqual(cases["successful_criterion"], "relaxed_high_precision_partial_recall")
        self.assertEqual([c["image_id"] for c in cases["successful"]], ["good"])
        self.assertIn("relaxed success", cases["successful"][0]["reason"])
        self.assertIn("successful_relaxed", cases["selection_rules"])

    def test_no_candidate_stays_empty(self):
        from uavdet import evaluation

        match = self._match([
            {"image_id": "bad", "num_gt": 10, "tp": 1, "fp": 4, "fn": 9,
             "max_conf": 0.4, "num_candidates": 5},
        ])
        cases = evaluation.select_cases(match, max_cases=2)
        self.assertEqual(cases["successful"], [])
        self.assertEqual(cases["successful_criterion"], "strict_no_fn_no_fp")


class CompletenessExpectationTests(unittest.TestCase):
    """A smoke run must not be judged against the formal 548-image expectation."""

    def _config(self, scope):
        return {
            "scope": scope,
            "data": {
                "val_split": "val", "train_split": "train",
                "expected_val_images": 548, "expected_train_images": 6471,
                "smoke_val_images": 10, "smoke_train_images": 20,
            },
        }

    def test_scope_aware_expected_counts(self):
        from uavdet.evaluation import _expected_images_for_split

        formal = self._config("formal")
        smoke = self._config("smoke")
        self.assertEqual(_expected_images_for_split(formal, "val"), 548)
        self.assertEqual(_expected_images_for_split(formal, "train"), 6471)
        self.assertEqual(_expected_images_for_split(smoke, "val"), 10)
        self.assertEqual(_expected_images_for_split(smoke, "train"), 20)
        self.assertIsNone(_expected_images_for_split(formal, "test"))


class MaxDetWarningTests(unittest.TestCase):
    """Truncation at max_det must be surfaced, never silently ignored."""

    def test_hit_ids_are_detected(self):
        from uavdet.evaluation import _max_det_hit_ids

        predictions = [
            {"image_id": "a", "num_detections": 300},
            {"image_id": "b", "num_detections": 299},
            {"image_id": "c", "num_detections": 500},
            {"image_id": "d", "num_detections": 0},
            {"image_id": "e"},
        ]
        self.assertEqual(_max_det_hit_ids(predictions, 300), ["a", "c"])
        self.assertEqual(_max_det_hit_ids(predictions, 1000), [])
        self.assertEqual(_max_det_hit_ids(predictions, 0), [])


class ValidationArgumentRegressionTests(unittest.TestCase):
    """B02: the validator must receive the configured values, not defaults."""

    def test_non_default_max_det_conf_and_iou_reach_the_validator(self):
        from uavdet import ultralytics_adapter as adapter
        from uavdet.config import apply_overrides, load_config

        base = load_config("configs/project.yaml")
        config = apply_overrides(base, {"eval.max_det": 777, "eval.conf": 0.02,
                                        "eval.iou": 0.6, "eval.batch": 3, "eval.imgsz": 512})
        args = adapter.validation_arguments(config, __import__("pathlib").Path("w.pt"),
                                            __import__("pathlib").Path("d.yaml"), "val", "cpu")
        self.assertEqual(args["max_det"], 777)
        self.assertAlmostEqual(args["conf"], 0.02)
        self.assertAlmostEqual(args["iou"], 0.6)
        self.assertEqual(args["batch"], 3)
        self.assertEqual(args["imgsz"], 512)

    def test_split_and_output_location_reach_the_validator(self):
        from pathlib import Path

        from uavdet import ultralytics_adapter as adapter
        from uavdet.config import load_config

        config = load_config("configs/project.yaml")
        args = adapter.validation_arguments(config, Path("w.pt"), Path("d.yaml"), "train", "cpu",
                                            record=True, save_dir=Path("/tmp/run/evaluation"))
        self.assertEqual(args["split"], "train")
        self.assertEqual(args["device"], "cpu")
        self.assertEqual(Path(str(args["project"])), Path("/tmp/run/evaluation"))
        self.assertTrue(args["plots"], "confusion matrix needs plots enabled when recording")


class CaseSelectionArtifactTests(unittest.TestCase):
    """Plan 02 §2: case_selection.json, per-case features, explicit image ids."""

    def _images(self):
        return [
            {"image_id": "a", "num_gt": 3, "num_detections": 3, "num_candidates": 3,
             "classes": ["car", "pedestrian"], "gt_median_rel_area": 0.0012,
             "tp": 3, "fp": 0, "fn": 0, "max_conf": 0.9},
            {"image_id": "b", "num_gt": 5, "num_detections": 4, "num_candidates": 4,
             "classes": ["truck"], "gt_median_rel_area": 0.004,
             "tp": 3, "fp": 1, "fn": 2, "max_conf": 0.7},
        ]

    def test_features_and_explicit_ids_are_recorded(self):
        from uavdet import evaluation

        match = {"conf": 0.25, "iou": 0.5, "images": self._images()}
        cases = evaluation.select_cases(match, max_cases=3, explicit_ids=["b", "missing"])
        self.assertEqual(cases["explicit_case_images"], ["b"])
        self.assertEqual(cases["ignored_case_images"], ["missing"])
        self.assertEqual(cases["successful"][0]["image_id"], "b")
        self.assertEqual(cases["successful"][0]["criterion"], "user_selected")
        self.assertEqual(cases["successful"][0]["classes"], ["truck"])
        self.assertAlmostEqual(cases["successful"][0]["gt_median_rel_area"], 0.004)
        self.assertIn("explicitly requested", cases["successful"][0]["reason"].lower())
        automatic = [item for item in cases["successful"] if item["criterion"] != "user_selected"]
        self.assertTrue(automatic and automatic[0]["image_id"] == "a")
        self.assertEqual(automatic[0]["classes"], ["car", "pedestrian"])
        self.assertTrue(cases["failure"][0]["classes"])


class ZeroAreaAccountingTests(unittest.TestCase):
    def test_zero_area_detections_are_counted_not_hidden(self):
        from uavdet.evaluation import _zero_area_count

        predictions = [
            {"detections": [{"xyxy": [0, 0, 10, 10]}, {"xyxy": [5, 5, 5, 9]}]},
            {"detections": [{"xyxy": [3, 3, 9, 3]}]},
            {"detections": []},
        ]
        self.assertEqual(_zero_area_count(predictions), 2)
