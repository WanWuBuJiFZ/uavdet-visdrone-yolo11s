"""Shared fixtures for the uavdet test suite.

Everything here builds *synthetic* material in a temporary directory: tiny
generated PNGs, hand-written labels, fake YOLO objects and a fully populated
"fake run" with the same on-disk schema the real pipeline writes. The synthetic
numbers are test logic only — they are never model performance evidence and are
never written outside the test temp directory.

The helpers are deliberately dependency-light at import time (PIL, numpy, yaml)
so ``python -m unittest discover -s tests`` works even without torch installed.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from uavdet import types as uav_types
from uavdet.artifacts import (
    begin_stage,
    complete_stage,
    open_run,
    write_json_atomic,
    write_text_atomic,
)
from uavdet.config import DEFAULTS, load_config
from uavdet.data import audit_dataset, save_audit
from uavdet.types import CLASS_NAMES, NUM_CLASSES, RunContext

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# tiny image / label writers
# ---------------------------------------------------------------------------
def write_png(path: str | Path, width: int = 64, height: int = 48,
              colour: tuple[int, int, int] = (120, 130, 140),
              seed: int | None = None) -> Path:
    """Write a deterministic small PNG (optionally with noise for uniqueness)."""
    from PIL import Image

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if seed is None:
        image = Image.new("RGB", (width, height), colour)
    else:
        rng = random.Random(seed)
        pixels = bytes(
            value
            for _ in range(width * height)
            for value in (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        )
        image = Image.frombytes("RGB", (width, height), pixels)
    image.save(target)
    return target


def write_label(path: str | Path,
                rows: Iterable[Sequence[float]] = ()) -> Path:
    """Write a YOLO label file; ``rows`` are ``(cls, cx, cy, w, h)``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        values = list(row)
        if len(values) == 5:
            cls, cx, cy, w, h = values
            lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        else:  # raw line (used to create malformed labels)
            lines.append(" ".join(str(v) for v in values))
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target


def default_rows(count: int = 2, class_id: int = 0, seed: int = 0) -> list[tuple[int, float, float, float, float]]:
    """Deterministic, valid label rows."""
    rng = random.Random(seed)
    rows: list[tuple[int, float, float, float, float]] = []
    for index in range(count):
        cx = rng.uniform(0.15, 0.85)
        cy = rng.uniform(0.15, 0.85)
        w = rng.uniform(0.02, 0.15)
        h = rng.uniform(0.02, 0.15)
        cx = min(max(cx, w / 2 + 0.01), 1 - w / 2 - 0.01)
        cy = min(max(cy, h / 2 + 0.01), 1 - h / 2 - 0.01)
        rows.append(((class_id + index) % NUM_CLASSES, cx, cy, w, h))
    return rows


# ---------------------------------------------------------------------------
# synthetic dataset + project
# ---------------------------------------------------------------------------
def make_dataset(root: str | Path, n_train: int = 4, n_val: int = 2, *,
                 width: int = 64, height: int = 48,
                 train_rows: int = 2, val_rows: int = 2,
                 anomalies: Sequence[str] = (),
                 duplicate_across_splits: bool = False,
                 seed: int = 0) -> dict[str, Any]:
    """Create a miniature VisDrone-shaped dataset below ``root``.

    ``anomalies`` may contain:
      ``"corrupt_image"``   — a truncated/invalid JPEG in the val split
      ``"bad_box"``         — a label line with a negative width
      ``"out_of_range_cls"``— a label line with class id 14
      ``"orphan_label"``    — a label without a matching image
      ``"missing_label"``   — an image without a label file
    """
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    data_yaml = {
        "path": ".",
        "train": "VisDrone2019-DET-train/images",
        "val": "VisDrone2019-DET-val/images",
        "nc": NUM_CLASSES,
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (base / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    created: dict[str, list[str]] = {"train": [], "val": []}
    for split, count, rows in (("train", n_train, train_rows), ("val", n_val, val_rows)):
        images_dir = base / f"VisDrone2019-DET-{split}" / "images"
        labels_dir = base / f"VisDrone2019-DET-{split}" / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = f"{split}_{index:04d}"
            split_offset = 0 if split == "train" else 500
            write_png(images_dir / f"{stem}.jpg", width, height,
                      colour=(40 + index * 10, 60, 90 + split_offset // 10),
                      seed=seed * 1000 + split_offset + index)
            write_label(labels_dir / f"{stem}.txt",
                        default_rows(rows, class_id=(index % NUM_CLASSES),
                                     seed=split_offset + index))
            created[split].append(stem)

    if duplicate_across_splits:
        # Byte-identical image in both splits with the same label content.
        source = base / "VisDrone2019-DET-train" / "images" / f"{created['train'][0]}.jpg"
        target = base / "VisDrone2019-DET-val" / "images" / f"{created['val'][0]}.jpg"
        shutil.copy2(source, target)

    if "corrupt_image" in anomalies:
        bad = base / "VisDrone2019-DET-val" / "images" / "val_broken.jpg"
        bad.write_bytes(b"not a jpeg")
        write_label(base / "VisDrone2019-DET-val" / "labels" / "val_broken.txt", default_rows(1))

    if "bad_box" in anomalies:
        write_label(base / "VisDrone2019-DET-val" / "labels" / f"{created['val'][-1]}.txt",
                    [(0, 0.5, 0.5, -0.2, 0.3)])

    if "out_of_range_cls" in anomalies:
        write_label(base / "VisDrone2019-DET-train" / "labels" / f"{created['train'][-1]}.txt",
                    [(14, 0.5, 0.5, 0.2, 0.2)])

    if "orphan_label" in anomalies:
        write_label(base / "VisDrone2019-DET-val" / "labels" / "orphan.txt", default_rows(1))

    if "missing_label" in anomalies:
        (base / "VisDrone2019-DET-train" / "labels" / f"{created['train'][-1]}.txt").unlink()

    return {
        "root": base,
        "data_yaml": base / "data.yaml",
        "images": {
            split: base / f"VisDrone2019-DET-{split}" / "images" for split in ("train", "val")
        },
        "labels": {
            split: base / f"VisDrone2019-DET-{split}" / "labels" for split in ("train", "val")
        },
        "counts": {"train": n_train, "val": n_val},
        "stems": created,
        "width": width,
        "height": height,
    }


def write_config(path: str | Path, *, project_root: str | Path = ".",
                 data_root: str | Path = "data", scope: str = "formal",
                 expected_train: int | None = None, expected_val: int | None = None,
                 overrides: Mapping[str, Any] | None = None) -> Path:
    """Write a complete project YAML (all defaults) with the given paths."""
    payload = {key: dict(value) for key, value in DEFAULTS.items()
               if isinstance(value, dict)}
    payload["scope"] = scope
    payload["project"]["root"] = str(project_root)
    payload["data"]["root"] = str(data_root)
    if expected_train is not None:
        payload["data"]["expected_train_images"] = int(expected_train)
    if expected_val is not None:
        payload["data"]["expected_val_images"] = int(expected_val)
    for dotted, value in (overrides or {}).items():
        section, key = dotted.split(".", 1)
        payload[section][key] = value
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target


class TempProject:
    """A temp directory holding ``configs/project.yaml`` + ``data/``."""

    def __init__(self, root: Path, config_path: Path, dataset: dict[str, Any]):
        self.root = root
        self.config_path = config_path
        self.dataset = dataset
        self.data_root = root / "data"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def workspace_dir(self) -> Path:
        return self.root / "workspace"

    def config(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        from uavdet.config import apply_overrides

        return apply_overrides(load_config(self.config_path), dict(extra or {}))


def make_project(tmp_path: str | Path, *, n_train: int = 4, n_val: int = 2,
                 anomalies: Sequence[str] = (),
                 duplicate_across_splits: bool = False,
                 scope: str = "formal",
                 overrides: Mapping[str, Any] | None = None) -> TempProject:
    """Create ``<tmp>/configs/project.yaml`` + ``<tmp>/data`` and return a handle."""
    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    dataset_root = root / "data"
    dataset = make_dataset(dataset_root, n_train=n_train, n_val=n_val,
                           anomalies=anomalies,
                           duplicate_across_splits=duplicate_across_splits)
    effective: dict[str, Any] = {}
    if scope == "smoke":
        # smoke scope must stay a short run per config validation
        effective.update({"train.epochs": 2, "train.batch": 2, "train.workers": 0,
                          "train.close_mosaic": 0})
    effective.update(dict(overrides or {}))
    config_path = write_config(
        root / "configs" / "project.yaml",
        project_root="..",
        data_root="data",
        scope=scope,
        expected_train=n_train,
        expected_val=n_val,
        overrides=effective,
    )
    return TempProject(root, config_path, dataset)


# ---------------------------------------------------------------------------
# fake Ultralytics objects
# ---------------------------------------------------------------------------
class FakeBoxes:
    """Mimics ``ultralytics.engine.results.Boxes`` enough for the engine."""

    def __init__(self, xyxy: Sequence[Sequence[float]], conf: Sequence[float],
                 cls: Sequence[int]):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class FakeResult:
    """Mimics one ``ultralytics.engine.results.Results`` item."""

    def __init__(self, boxes: FakeBoxes | None, orig_shape: tuple[int, int] = (48, 64),
                 path: str = "image.jpg", names: Mapping[int, str] | None = None,
                 speed: Mapping[str, float] | None = None):
        self.boxes = boxes
        self.orig_shape = orig_shape
        self.path = path
        self.names = dict(names or {index: name for index, name in enumerate(CLASS_NAMES)})
        self.speed = dict(speed or {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5})

    def plot(self):  # pragma: no cover - only used if a caller still calls plot()
        raise AssertionError("uavdet must annotate with its own engine, not result.plot()")


class FakePredictModel:
    """Callable/`predict` object standing in for ``YOLO`` in the inference engine."""

    def __init__(self, results: Sequence[FakeResult] | None = None,
                 names: Mapping[int, str] | None = None):
        self.names = dict(names or {index: name for index, name in enumerate(CLASS_NAMES)})
        self.results = list(results or [])
        self.calls: list[dict[str, Any]] = []
        self.predict_calls = self.calls

    def predict(self, source=None, **kwargs):
        self.calls.append(dict(kwargs, source=source))
        if self.results:
            return list(self.results)
        return [FakeResult(FakeBoxes([], [], []))]

    __call__ = predict


class RecordingYOLO:
    """Stands in for ``ultralytics.YOLO`` in ``training.train_model``.

    ``train(**kwargs)`` records the arguments, creates a realistic
    ``save_dir`` (args.yaml, results.csv, weights/best.pt, weights/last.pt) and
    returns a small result object; ``val``/``predict`` delegate to fakes.
    """

    instances: list["RecordingYOLO"] = []

    def __init__(self, weights: str | Path = "yolo11s.pt", *, epochs: int | None = None,
                 fail_train: bool = False, fail_load: bool = False,
                 metrics: Any = None):
        if fail_load:
            raise RuntimeError("cannot load weights")
        self.weights = str(weights)
        self.epochs = epochs
        self.fail_train = fail_train
        self.metrics = metrics
        self.train_kwargs: dict[str, Any] = {}
        self.val_kwargs: dict[str, Any] = {}
        self.predict_kwargs: list[dict[str, Any]] = []
        self.save_dir: Path | None = None
        self.names = {index: name for index, name in enumerate(CLASS_NAMES)}
        self.trainer = SimpleNamespace(save_dir=None)
        RecordingYOLO.instances.append(self)

    # -- training ---------------------------------------------------------
    def train(self, **kwargs):
        self.train_kwargs = dict(kwargs)
        if self.fail_train:
            raise RuntimeError("synthetic training failure")
        project = Path(kwargs.get("project") or "runs")
        name = str(kwargs.get("name") or "train")
        save_dir = project / name if name else project
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "weights").mkdir(exist_ok=True)
        epochs = int(kwargs.get("epochs") or 3)
        write_results_csv(save_dir / "results.csv", epochs=epochs)
        (save_dir / "args.yaml").write_text(
            yaml.safe_dump({k: str(v) for k, v in kwargs.items()}, sort_keys=False),
            encoding="utf-8")
        (save_dir / "train.log").write_text("epoch 1/1 synthetic\n", encoding="utf-8")
        for weight_name in ("best.pt", "last.pt"):
            (save_dir / "weights" / weight_name).write_bytes(b"synthetic-weights-" + weight_name.encode())
        self.save_dir = save_dir
        self.trainer = SimpleNamespace(save_dir=str(save_dir))
        return SimpleNamespace(save_dir=str(save_dir), epochs=epochs)

    # -- validation -------------------------------------------------------
    def val(self, **kwargs):
        self.val_kwargs = dict(kwargs)
        return self.metrics if self.metrics is not None else FakeMetrics()

    def predict(self, **kwargs):
        self.predict_kwargs.append(dict(kwargs))
        return [FakeResult(FakeBoxes([[1, 2, 10, 12]], [0.9], [0]))]


class FakeMetricBox:
    """The ``metrics.box`` object of Ultralytics ``DetMetrics``."""

    def __init__(self, num_classes: int = NUM_CLASSES, valid: Sequence[int] | None = None):
        self.p = np.array([0.80] * num_classes, dtype=float)
        self.r = np.array([0.70] * num_classes, dtype=float)
        self.ap50 = np.array([0.60] * num_classes, dtype=float)
        self.ap = np.array([0.40] * num_classes, dtype=float)
        if valid is not None:
            for index in range(num_classes):
                if index not in set(valid):
                    self.p[index] = np.nan
                    self.r[index] = np.nan
                    self.ap50[index] = np.nan
                    self.ap[index] = np.nan
        self.all_ap = np.tile(np.array([0.4] * 10, dtype=float), (num_classes, 1))
        self.mp = float(np.nanmean(self.p))
        self.mr = float(np.nanmean(self.r))
        self.map50 = float(np.nanmean(self.ap50))
        self.map = float(np.nanmean(self.ap))
        self.f1 = np.array([0.75] * num_classes, dtype=float)
        self.maps = np.array([self.map50, self.map, 0.3, 0.2])


class FakeConfusionMatrix:
    """Mimics ``ultralytics.utils.metrics.ConfusionMatrix``."""

    def __init__(self, nc: int = NUM_CLASSES):
        self.nc = nc
        self.matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)
        for index in range(nc):
            self.matrix[index, index] = 10 + index
        self.matrix[nc, nc] = 4          # true background, predicted background
        self.matrix[0, 1] = 3            # pedestrian predicted as people
        self.matrix[nc, 0] = 2           # background predicted as pedestrian
        self.matrix[2, nc] = 1           # bicycle missed


class FakeCurve:
    """One native PR / confidence curve object."""

    def __init__(self, x: Any, y: Any, xlabel: str = "x", ylabel: str = "y"):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.xlabel = xlabel
        self.ylabel = ylabel


class FakeCurves(list):
    """List of ``FakeCurve`` with class names attached (``.names``)."""

    def __init__(self, curves: Iterable[FakeCurve], names: Mapping[int, str]):
        super().__init__(curves)
        self.names = dict(names)


class FakeMetrics:
    """Stand-in for ``ultralytics.utils.metrics.DetMetrics``."""

    def __init__(self, num_classes: int = NUM_CLASSES, valid: Sequence[int] | None = None):
        self.names = {index: name for index, name in enumerate(CLASS_NAMES)}
        self.box = FakeMetricBox(num_classes, valid)
        self.speed = {"preprocess": 1.2, "inference": 8.5, "postprocess": 0.7, "loss": 0.0}
        self.confusion_matrix = FakeConfusionMatrix(num_classes)
        self.nt_per_class = np.array([100 - 5 * index for index in range(num_classes)], dtype=int)
        self.nt_per_image = np.array([5, 4, 3], dtype=int)
        # Native curves: PR + P/R/F1 versus confidence.
        recall = np.linspace(0, 1, 50)
        conf = np.linspace(0, 1, 50)
        curves = []
        for class_id in range(num_classes):
            curves.append(FakeCurve(recall, recall * (0.9 - 0.01 * class_id), "Recall", "Precision"))
        for class_id in range(num_classes):
            curves.append(FakeCurve(conf, np.clip(1.0 - conf * (0.1 + 0.01 * class_id), 0, 1),
                                    "Confidence", "Precision"))
        for class_id in range(num_classes):
            curves.append(FakeCurve(conf, np.clip(0.95 - conf * 0.2, 0, 1), "Confidence", "Recall"))
        for class_id in range(num_classes):
            f1 = 2 * (0.9 - 0.01 * class_id) / (1 + (0.9 - 0.01 * class_id))
            curves.append(FakeCurve(conf, np.interp(conf, [0, 0.5 + 0.01 * class_id, 1],
                                                    [0.0, f1, 0.0]), "Confidence", "F1"))
        self.curves = FakeCurves(curves, self.names)
        self.curves_results: list[Any] = []
        self.results_dict = {"metrics/precision(B)": self.box.mp,
                             "metrics/recall(B)": self.box.mr,
                             "metrics/mAP50(B)": self.box.map50,
                             "metrics/mAP50-95(B)": self.box.map}
        self.save_dir = ""


class FakeValidator:
    """Stands in for a DetectionValidator instance."""

    def __init__(self, metrics: Any = None):
        self.metrics = metrics if metrics is not None else FakeMetrics()

    def __call__(self, *args, **kwargs):
        return self.metrics


# ---------------------------------------------------------------------------
# fake run with the real on-disk schema
# ---------------------------------------------------------------------------
RESULTS_COLUMNS = (
    "epoch", "time",
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "lr/pg0", "lr/pg1", "lr/pg2",
)


def write_results_csv(path: str | Path, epochs: int = 12, start_epoch: int = 1,
                      seed: int = 0) -> Path:
    """Write a synthetic Ultralytics ``results.csv`` with monotone improvement."""
    rng = random.Random(seed)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    elapsed = 0.0
    for step in range(epochs):
        epoch = start_epoch + step
        progress = (step + 1) / epochs
        elapsed += 45.0 + rng.uniform(-2, 2)
        rows.append([
            epoch, round(elapsed, 3),
            round(2.0 - 1.0 * progress + rng.uniform(-0.02, 0.02), 5),
            round(1.5 - 0.9 * progress + rng.uniform(-0.02, 0.02), 5),
            round(1.3 - 0.5 * progress + rng.uniform(-0.02, 0.02), 5),
            round(0.45 + 0.35 * progress, 5),
            round(0.35 + 0.30 * progress, 5),
            round(0.20 + 0.23 * progress, 5),
            round(0.10 + 0.15 * progress, 5),
            round(1.9 - 0.8 * progress, 5),
            round(1.4 - 0.7 * progress, 5),
            round(1.2 - 0.4 * progress, 5),
            0.01 * (1 - progress) + 0.0001, 0.01 * (1 - progress), 0.01 * (1 - progress),
        ])
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch"] + [c for c in RESULTS_COLUMNS[1:]])
        writer.writerows(rows)
    return target


def make_per_class_csv(path: str | Path, valid: Sequence[int] | None = None) -> Path:
    """Per-class metrics CSV with real N/A handling for classes without GT."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    valid_set = set(range(NUM_CLASSES) if valid is None else valid)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "num_gt", "precision", "recall", "ap50", "ap50_95"])
        for index, name in enumerate(CLASS_NAMES):
            if index in valid_set:
                writer.writerow([index, name, 120 - 8 * index, f"{0.8 - 0.01 * index:.4f}",
                                 f"{0.7 - 0.02 * index:.4f}", f"{0.6 - 0.03 * index:.4f}",
                                 f"{0.4 - 0.02 * index:.4f}"])
            else:
                writer.writerow([index, name, 0, "N/A", "N/A", "N/A", "N/A"])
    return target


def make_curves_npz(path: str | Path, num_classes: int = NUM_CLASSES,
                    points: int = 40) -> Path:
    """Native-curve arrays: PR, P/R/F1 vs confidence, plus a valid-class mask."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    recall = np.linspace(0.0, 1.0, points)
    conf = np.linspace(0.0, 1.0, points)
    pr = np.zeros((num_classes + 1, points), dtype=float)
    f1 = np.zeros((num_classes + 1, points), dtype=float)
    p_conf = np.zeros((num_classes + 1, points), dtype=float)
    r_conf = np.zeros((num_classes + 1, points), dtype=float)
    for index in range(num_classes):
        pr[index] = np.clip(0.9 - 0.01 * index - 0.3 * recall ** 2, 0, 1)
        f1[index] = np.clip(0.2 + 0.9 * np.sin(np.pi * conf) * (1 - 0.03 * index), 0, 1)
        p_conf[index] = np.clip(0.95 - 0.4 * conf * (1 + 0.02 * index), 0, 1)
        r_conf[index] = np.clip(0.95 - 0.6 * conf * (1 + 0.02 * index), 0, 1)
    pr[num_classes] = pr[:num_classes].mean(axis=0)
    f1[num_classes] = f1[:num_classes].mean(axis=0)
    p_conf[num_classes] = p_conf[:num_classes].mean(axis=0)
    r_conf[num_classes] = r_conf[:num_classes].mean(axis=0)
    valid = np.ones(num_classes, dtype=bool)
    np.savez_compressed(target, recall=recall, conf=conf, pr_precision=pr, f1=f1,
                        p_conf=p_conf, r_conf=r_conf, valid_classes=valid)
    return target


def make_metrics_json(path: str | Path, *, scope: str = "formal", split: str = "val",
                      num_images: int = 2, num_instances: int = 4,
                      weights_sha256: str = "0" * 64,
                      dataset_manifest_sha256: str = "1" * 64,
                      data_view_root: str | None = None) -> dict[str, Any]:
    payload = {
        "run_id": Path(path).parent.parent.name,
        "scope": scope,
        "split": split,
        "num_images": num_images,
        "num_instances": num_instances,
        "weights_sha256": weights_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "evaluator_name": "ultralytics.DetectionValidator",
        "evaluator_version": "8.3.test",
        "imgsz": 640,
        "conf": 0.001,
        "iou_nms": 0.7,
        "max_det": 300,
        "precision_mode": "FP32",
        "map50": 0.42992,
        "map50_95": 0.25111,
        "precision": 0.78123,
        "recall": 0.65234,
        "pr_definition": "native Ultralytics detection validator: per-class precision/recall "
                         "curves integrated at IoU=0.70, confidence swept over the full range",
        "status": "ok",
        "warnings": [],
        "image_ids_digest": "2" * 64,
        "data_view_root": data_view_root,
        "cases": {"conf": 0.25, "iou": 0.5, "tp": 3, "fp": 1, "fn": 1},
        "speed_ms": {"preprocess": 1.2, "inference": 8.5, "postprocess": 0.7, "total": 10.4},
    }
    write_json_atomic(path, payload)
    return payload


def make_predictions_jsonl(path: str | Path, image_paths: Sequence[Path],
                           gt_rows: Sequence[Sequence[tuple[int, tuple[float, float, float, float]]]] | None = None,
                           ) -> tuple[Path, Path]:
    """Write matching predictions.jsonl + ground_truth.jsonl for the given images."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    gt_target = target.with_name("ground_truth.jsonl")
    pred_lines: list[dict[str, Any]] = []
    gt_lines: list[dict[str, Any]] = []
    from PIL import Image

    for index, image in enumerate(image_paths):
        image = Path(image)
        with Image.open(image) as handle:
            width, height = handle.size
        boxes = gt_rows[index] if gt_rows is not None and index < len(gt_rows) else []
        gt_objects = [
            {"class_id": int(cls), "class_name": CLASS_NAMES[int(cls)],
             "xyxy": [float(v) for v in xyxy]}
            for cls, xyxy in boxes
        ]
        detections = []
        for cls, xyxy in boxes:
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            # A near-perfect prediction plus a small shift keeps IoU above 0.5.
            detections.append({
                "class_id": int(cls), "class_name": CLASS_NAMES[int(cls)],
                "confidence": 0.9 - 0.01 * len(detections),
                "xyxy": [x1 + 1.0, y1 + 1.0, x2 + 1.0, y2 + 1.0],
            })
        pred_lines.append({
            "image_id": image.stem, "source": str(image), "source_rel": f"images/val/{image.name}",
            "width": width, "height": height, "num_detections": len(detections),
            "detections": detections,
            "timing": {"total_ms": 10.0 + index},
        })
        gt_lines.append({
            "image_id": image.stem, "source": str(image), "source_rel": f"images/val/{image.name}",
            "width": width, "height": height, "num_objects": len(gt_objects),
            "objects": gt_objects,
        })
    write_text_atomic(target, "\n".join(json.dumps(row, ensure_ascii=False) for row in pred_lines) + "\n")
    write_text_atomic(gt_target, "\n".join(json.dumps(row, ensure_ascii=False) for row in gt_lines) + "\n")
    return target, gt_target


def make_cases_json(path: str | Path, image_ids: Sequence[str], failure_ids: Sequence[str] = ()) -> dict[str, Any]:
    payload = {
        "created_at": "2026-01-01T00:00:00+0800",
        "conf": 0.25,
        "iou": 0.5,
        "selection_rules": {
            "successful": "highest matched-TP count at conf>=0.25, then densest GT",
            "failure": "has false negatives or false positives at conf>=0.25",
        },
        "successful": [
            {"image_id": image_id, "reason": "all GT matched at conf>=0.25", "num_gt": 3,
             "tp": 3, "fp": 0, "fn": 0, "max_conf": 0.93}
            for image_id in image_ids
        ],
        "failure": [
            {"image_id": image_id, "reason": "missed small objects", "num_gt": 4,
             "tp": 2, "fp": 1, "fn": 2, "failure_kinds": ["false_negative", "false_positive"]}
            for image_id in (failure_ids or image_ids[:1])
        ],
    }
    write_json_atomic(path, payload)
    return payload


def make_benchmark(run: RunContext, *, num_images: int = 5, repeats: int = 3) -> dict[str, Any]:
    benchmark_dir = run.benchmark_dir
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = random.Random(0)
    for repeat in range(repeats):
        for index in range(num_images):
            rows.append([repeat, index, f"img_{index:04d}", round(9.0 + rng.uniform(-0.8, 0.8), 4)])
    write_text_atomic(
        benchmark_dir / "timings.csv",
        "repeat,index,image_id,ms\n" + "\n".join(",".join(str(v) for v in row) for row in rows) + "\n",
    )
    values = np.array([row[3] for row in rows])
    summary = {
        "run_id": run.run_id,
        "weights_sha256": "0" * 64,
        "device": "cpu",
        "effective_device": "cpu",
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,
        "max_det": 300,
        "num_images": num_images,
        "warmup": 5,
        "repeats": repeats,
        "batch": 1,
        "half": False,
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "fps": float(1000.0 / values.mean()),
        "timing_scope": "decoded original image -> preprocess -> model -> postprocess -> structured result",
        "hardware": {"platform": "test", "device": "cpu"},
    }
    write_json_atomic(benchmark_dir / "summary.json", summary)
    write_json_atomic(benchmark_dir / "image_list.json", {"images": [f"img_{i:04d}" for i in range(num_images)]})
    return summary


def make_fake_run(project: TempProject, run_id: str = "test_run", *,
                  scope: str | None = None,
                  with_train: bool = True, with_evaluation: bool = True,
                  with_benchmark: bool = True,
                  valid_classes: Sequence[int] | None = None,
                  epochs: int = 12,
                  config_extra: Mapping[str, Any] | None = None) -> RunContext:
    """Create a run directory with real audit stats and schema-correct products."""
    config = project.config(config_extra)
    run = open_run(config, run_id)

    # -- audit --------------------------------------------------------------
    begin_stage(run, "audit")
    audit = audit_dataset(config)
    save_audit(audit, run)
    complete_stage(run, "audit", status=audit["status"],
                   manifest_digest=audit["manifest_digest"],
                   num_images=audit["totals"]["num_images"])

    # -- train --------------------------------------------------------------
    if with_train:
        begin_stage(run, "train")
        run.train_dir.mkdir(parents=True, exist_ok=True)
        write_results_csv(run.train_dir / "results.csv", epochs=epochs)
        (run.train_dir / "args.yaml").write_text(
            yaml.safe_dump({"epochs": epochs, "imgsz": 640, "batch": 4, "optimizer": "SGD"}),
            encoding="utf-8")
        (run.train_dir / "train.log").write_text("synthetic training log\n", encoding="utf-8")
        (run.train_dir / "weights").mkdir(exist_ok=True)
        for name in ("best.pt", "last.pt"):
            (run.train_dir / "weights" / name).write_bytes(b"synthetic-" + name.encode())
        complete_stage(run, "train", best_weights="train/weights/best.pt",
                       epochs_done=epochs)

    # -- evaluation ---------------------------------------------------------
    if with_evaluation:
        begin_stage(run, "evaluation")
        eval_dir = run.evaluation_dir
        eval_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(project.dataset["images"]["val"].glob("*.jpg"))
        for index, image in enumerate(images):
            width = project.dataset["width"]
            height = project.dataset["height"]
            write_png(image, width, height, colour=(30 + index * 20, 80, 120),
                      seed=100 + index)
        gt_rows = [
            [(0, (6.0, 6.0, 18.0, 18.0)), (3, (28.0, 20.0, 46.0, 36.0))],
            [(3, (4.0, 4.0, 20.0, 20.0))],
        ]
        make_predictions_jsonl(eval_dir / "predictions.jsonl", images, gt_rows)
        make_metrics_json(eval_dir / "metrics.json", scope=scope or config["scope"],
                          num_images=len(images), num_instances=3,
                          dataset_manifest_sha256=str(audit["manifest_digest"]))
        make_per_class_csv(eval_dir / "per_class.csv", valid=valid_classes)
        make_curves_npz(eval_dir / "curves.npz")
        write_json_atomic(eval_dir / "curves_meta.json", {
            "source": {"evaluator": "ultralytics.DetectionValidator", "version": "8.3.test"},
            "class_order": list(CLASS_NAMES),
            "mean_row_index": NUM_CLASSES,
            "smoothing": "native curves, no extra smoothing applied",
            "arrays": {
                "recall": {"axes": ["point"], "units": "fraction 0-1"},
                "conf": {"axes": ["point"], "units": "fraction 0-1"},
                "pr_precision": {"axes": ["class+mean", "recall"], "units": "fraction 0-1"},
                "f1": {"axes": ["class+mean", "conf"], "units": "fraction 0-1"},
                "p_conf": {"axes": ["class+mean", "conf"], "units": "fraction 0-1"},
                "r_conf": {"axes": ["class+mean", "conf"], "units": "fraction 0-1"},
                "valid_classes": {"axes": ["class"], "units": "bool"},
            },
            "valid_classes": [index for index in range(NUM_CLASSES)
                              if valid_classes is None or index in set(valid_classes)],
        })
        base = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=np.int64)
        for index in range(NUM_CLASSES):
            base[index, index] = 10 + index
        base[NUM_CLASSES, NUM_CLASSES] = 4
        base[0, 1] = 3
        base[NUM_CLASSES, 0] = 2
        base[2, NUM_CLASSES] = 1
        np.savez_compressed(eval_dir / "confusion_matrix.npz", matrix=base)
        write_json_atomic(eval_dir / "confusion_meta.json", {
            "true_axis": "row", "pred_axis": "column", "background_index": NUM_CLASSES,
            "class_order": list(CLASS_NAMES), "conf": 0.001, "iou": 0.7,
            "normalization": "raw integer counts; GT-normalised only at plot time",
            "source_version": "8.3.test",
        })
        make_cases_json(eval_dir / "cases.json",
                        [image.stem for image in images[:2]],
                        [images[0].stem] if images else [])
        complete_stage(run, "evaluation", map50=0.42992, map50_95=0.25111,
                       num_images=len(images))

    # -- benchmark ----------------------------------------------------------
    if with_benchmark:
        begin_stage(run, "benchmark")
        make_benchmark(run)
        complete_stage(run, "benchmark", mean_ms=9.4)

    return run


# ---------------------------------------------------------------------------
# misc assertions used by several test modules
# ---------------------------------------------------------------------------
class TempDirTestCase(unittest.TestCase):
    """unittest case with a temporary project directory per test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def project(self, name: str = "proj", **kwargs: Any) -> TempProject:
        return make_project(self.tmp_path / name, **kwargs)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha_tree(root: str | Path) -> dict[str, str]:
    """sha256 of every file below ``root`` keyed by relative path."""
    from uavdet.artifacts import sha256_file

    base = Path(root)
    return {
        path.relative_to(base).as_posix(): sha256_file(path)
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def assert_jsonl_parses(testcase: unittest.TestCase, path: str | Path) -> list[dict[str, Any]]:
    testcase.assertTrue(Path(path).exists(), f"{path} missing")
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - failure path
                testcase.fail(f"{path}:{line_no} is not valid JSON: {exc}")
    return rows


def matrix_iou(a: Sequence[float], b: Sequence[float]) -> float:
    return uav_types.bbox_iou(a, b)


__all__ = [
    "PROJECT_ROOT",
    "TempProject",
    "TempDirTestCase",
    "make_dataset",
    "write_config",
    "make_project",
    "write_png",
    "write_label",
    "default_rows",
    "FakeBoxes",
    "FakeResult",
    "FakePredictModel",
    "RecordingYOLO",
    "FakeMetrics",
    "FakeMetricBox",
    "FakeCurve",
    "FakeCurves",
    "FakeConfusionMatrix",
    "FakeValidator",
    "write_results_csv",
    "make_fake_run",
    "make_per_class_csv",
    "make_curves_npz",
    "make_metrics_json",
    "make_predictions_jsonl",
    "make_cases_json",
    "make_benchmark",
    "read_csv_rows",
    "sha_tree",
    "assert_jsonl_parses",
    "matrix_iou",
]
