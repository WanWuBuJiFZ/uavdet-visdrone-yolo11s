"""Version-tolerant bridge to the installed Ultralytics detection validator.

Why this module exists
----------------------
The evaluation must be reproducible evidence: the mAP numbers, the per-image
predictions of the original image, the ground truth, the native curves and the
native confusion matrix all have to come from **one** validator pass. Running a
second, differently configured ``predict`` loop and presenting it as the metric
evidence is exactly the failure mode the plan forbids.

The installed Ultralytics (8.4.215) exposes a public hook for this::

    model.val(validator=<DetectionValidator subclass>, **args)

``YOLO.val`` builds the validator with ``validator(args=args, ...)`` and calls it,
so a subclass can record inside ``DetectionValidator.update_metrics`` where both
the post-NMS predictions and the batch ground truth are available for the very
same image. Attributes actually used from the installed version (read from
``.venv/lib/python3.11/site-packages/ultralytics``):

* ``models/yolo/detect/val.py`` — ``DetectionValidator.__init__``,
  ``_prepare_batch`` (``cls``, ``bboxes`` in letterbox pixels, ``ori_shape``,
  ``imgsz``, ``ratio_pad``, ``im_file``), ``update_metrics``, ``scale_preds``,
  ``init_metrics`` (``self.names``, ``self.args``), ``self.speed``.
* ``engine/model.py`` — ``Model.val(validator=None, **kwargs)``: kwargs are
  merged into the cfg namespace, so ``data``/``split``/``imgsz``/``conf``/``iou``/
  ``max_det``/``batch``/``device``/``project``/``name``/``exist_ok``/``plots``
  are all real arguments.
* ``utils/metrics.py`` — ``DetMetrics`` (``box``, ``names``, ``speed``,
  ``nt_per_class``, ``confusion_matrix``), ``Metric`` (``p``, ``r``, ``f1``,
  ``all_ap``, ``ap50``, ``ap``, ``ap_class_index``, ``px``, ``prec_values``,
  ``p_curve``, ``r_curve``, ``f1_curve``), ``ConfusionMatrix.matrix``.

Two hard rules:

1. ``ap_class_index`` maps metric rows to **class ids**. Rows of ``Metric.p`` /
   ``r`` / ``all_ap`` / the curve arrays are only produced for classes that have
   ground truth, so indexing them by class id would silently shift every number
   to the wrong class. :func:`extract_per_class` and :func:`extract_curves`
   always go through that mapping (or, when the installed object has no mapping,
   treat the arrays as class-id indexed, which is what older shapes are).
2. Nothing is invented. A field that the installed version does not expose is
   ``None`` plus a warning, or :class:`~uavdet.types.AdapterError` with the
   version in the message — never a zero, a blank array or a guessed curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .devices import resolve_device
from .types import (
    CLASS_NAMES,
    NUM_CLASSES,
    AdapterError,
    UAVDetError,
    WeightError,
)

__all__ = [
    "NATIVE_VAL_DIRNAME",
    "ValidationOutcome",
    "validation_arguments",
    "capture_batch",
    "make_recording_validator",
    "run_native_validation",
    "extract_metrics",
    "extract_per_class",
    "extract_curves",
    "extract_confusion",
]

#: The native confusion matrix matching IoU. ``DetectionValidator.update_metrics``
#: calls ``confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)`` and
#: leaves ``iou_thres`` at its 0.5 default, i.e. the matrix is *not* built with
#: ``args.iou`` (which is the NMS IoU). Recorded as a fact, not a guess.
NATIVE_CONFUSION_IOU = 0.5

#: Name of the native validation sub-directory created inside a run's evaluation
#: directory (``<run>/evaluation/ultralytics/``): the validator's own plots,
#: ``predictions.json`` and label caches must never leak outside the run.
NATIVE_VAL_DIRNAME = "ultralytics"

#: PR definition text stored in metrics.json (kept in one place so the log, the
#: JSON and the report agree).
PR_DEFINITION = (
    "native Ultralytics detection validator: per-class precision/recall are the "
    "values at the confidence that maximises the class mean F1 on the native P/R "
    "curve (then averaged over classes that have ground truth); mAP50/mAP50-95 "
    "integrate the native per-class PR curves at IoU 0.50:0.95. This is NOT a "
    "fixed-confidence micro average - the fixed-threshold TP/FP/FN analysis in "
    "cases.json is a separate, explicitly defined computation."
)


@dataclass
class ValidationOutcome:
    """Everything one native validation pass produced.

    ``predictions`` / ``ground_truth`` are ready-to-write JSONL rows in original
    image pixels (see ``docs/INTERFACE_CONTRACT.md`` §1).
    """

    metrics: Any
    predictions: list[dict[str, Any]] = field(default_factory=list)
    ground_truth: list[dict[str, Any]] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    speed: dict[str, float] = field(default_factory=dict)
    evaluator_name: str = "ultralytics.DetectionValidator"
    evaluator_version: str = ""
    warnings: list[str] = field(default_factory=list)
    #: Extra facts for the artefact metadata (thresholds actually used, ...).
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _version() -> str:
    try:
        import ultralytics  # noqa: PLC0415 - optional dependency

        return str(getattr(ultralytics, "__version__", "unknown"))
    except Exception:  # noqa: BLE001 - reporting only
        return "unknown"


def _as_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor / list / ndarray to a numpy array without copying twice."""
    if value is None:
        return np.zeros((0,), dtype=np.float64)
    if isinstance(value, np.ndarray):
        return value
    to_numpy = getattr(value, "detach", None)
    if to_numpy is not None:  # torch tensor
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _finite_or_none(value: Any) -> float | None:
    """float(value) when it is a finite number, else ``None`` (never NaN)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _class_names(metrics: Any, default: Sequence[str] = CLASS_NAMES) -> list[str]:
    """Native class names in class-id order, falling back to the fixed order."""
    raw = getattr(metrics, "names", None)
    if isinstance(raw, Mapping) and raw:
        try:
            ordered = sorted((int(key), str(value)) for key, value in raw.items())
        except (TypeError, ValueError):
            ordered = []
        if ordered and [index for index, _ in ordered] == list(range(len(ordered))):
            return [name for _, name in ordered]
    elif isinstance(raw, (list, tuple)) and raw:
        return [str(name) for name in raw]
    return list(default)


def _metric_branch(metrics: Any) -> Any:
    """The task metric object (``DetMetrics.box`` for detection)."""
    for attribute in ("box", "seg", "pose", "obb"):
        branch = getattr(metrics, attribute, None)
        if branch is not None:
            return branch
    return None


def _array(branch: Any, name: str) -> np.ndarray | None:
    """A native numpy array attribute, or ``None`` when the field does not exist."""
    if branch is None:
        return None
    value = getattr(branch, name, None)
    if value is None or callable(value):
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return array if array.size else None


def _class_positions(branch: Any, count: int) -> dict[int, int] | None:
    """Map class id -> row position of the native per-class arrays.

    ``None`` means "the arrays are already indexed by class id" (the shape the
    installed version uses when every class has ground truth, and the shape of
    older/fake metric objects).
    """
    raw = getattr(branch, "ap_class_index", None) if branch is not None else None
    if raw is None:
        return None
    try:
        positions = [int(value) for value in raw]
    except (TypeError, ValueError):
        return None
    if not positions:
        return None
    if positions == list(range(count)):
        return None
    return {class_id: position for position, class_id in enumerate(positions)}


def _gt_counts(metrics: Any, count: int) -> list[int | None] | None:
    """Per-class ground-truth instance counts from the native metrics object."""
    raw = getattr(metrics, "nt_per_class", None)
    if raw is None:
        return None
    try:
        values = [int(value) for value in np.asarray(raw).reshape(-1)]
    except (TypeError, ValueError):
        return None
    if len(values) < count:
        return None
    return values[:count]


# ---------------------------------------------------------------------------
# original-pixel conversion (the inverse of the letterbox transform)
# ---------------------------------------------------------------------------
def to_original_boxes(
    boxes: Any,
    imgsz: Sequence[int],
    ori_shape: Sequence[int],
    ratio_pad: Any = None,
) -> np.ndarray:
    """Convert letterbox-space ``xyxy`` boxes to original-image pixels.

    Mirrors ``ultralytics.utils.ops.scale_boxes`` exactly (same
    ``ratio_pad = ((gain_y, gain_x), (pad_x, pad_y))`` convention, subtract the
    padding, divide by the gain, clip to the image). ``data.label_row_to_xyxy``
    is the forward direction of the same transform, so labels and boxes agree.

    ``imgsz``/``ori_shape`` are ``(height, width)``.
    """
    array = _as_numpy(boxes).astype(np.float64, copy=True).reshape(-1, 4)
    if array.size == 0:
        return array.reshape(0, 4)
    height, width = float(ori_shape[0]), float(ori_shape[1])
    if ratio_pad is None:
        gain = min(float(imgsz[0]) / height, float(imgsz[1]) / width)
        new_height = max(round(height * gain), 1)
        new_width = max(round(width * gain), 1)
        gain_y, gain_x = new_height / height, new_width / width
        pad_x = round((float(imgsz[1]) - new_width) / 2 - 0.1)
        pad_y = round((float(imgsz[0]) - new_height) / 2 - 0.1)
    else:
        gain_y, gain_x = (float(value) for value in ratio_pad[0])
        pad_x, pad_y = (float(value) for value in ratio_pad[1])
    if gain_x <= 0 or gain_y <= 0:
        raise AdapterError(
            f"the validator reported a non-positive letterbox gain "
            f"({gain_x!r}, {gain_y!r}); boxes cannot be mapped back to original pixels"
        )
    array[:, [0, 2]] = (array[:, [0, 2]] - pad_x) / gain_x
    array[:, [1, 3]] = (array[:, [1, 3]] - pad_y) / gain_y
    array[:, [0, 2]] = np.clip(array[:, [0, 2]], 0.0, width)
    array[:, [1, 3]] = np.clip(array[:, [1, 3]], 0.0, height)
    return array


# ---------------------------------------------------------------------------
# recording validator
# ---------------------------------------------------------------------------
def _source_rel(image_path: str, view_root: Path | None) -> str:
    path = Path(image_path)
    if view_root is not None:
        try:
            return path.resolve().relative_to(view_root.resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return path.name


def capture_batch(validator: Any, capture: dict[str, Any], preds: Any, batch: Any) -> None:
    """Record one validation batch (predictions **and** ground truth).

    Called from the subclass' ``update_metrics`` before the native update, so the
    recorded rows belong to the same pass that produces the metrics. Only the
    validator's own API is used (``_prepare_batch``, ``names``, ``args``) plus
    plain numpy, which keeps the function testable with a stub validator.
    """
    names = _class_names(validator)
    view_root = capture.get("view_root")
    recorded_ids: set[str] = capture.setdefault("seen_ids", set())
    for sample_index, prediction in enumerate(preds):
        prepared = validator._prepare_batch(sample_index, batch)  # noqa: SLF001 - documented hook
        ori_shape = tuple(int(value) for value in prepared["ori_shape"])
        height, width = ori_shape[0], ori_shape[1]
        imgsz = tuple(int(value) for value in prepared["imgsz"])
        ratio_pad = prepared.get("ratio_pad")
        im_file = str(prepared["im_file"])
        image_id = Path(im_file).stem
        if image_id in recorded_ids:
            capture["warnings"].append(
                f"duplicate image id {image_id!r} in the validation pass; ids would be ambiguous"
            )
        recorded_ids.add(image_id)
        rel = _source_rel(im_file, view_root)

        gt_boxes = to_original_boxes(prepared["bboxes"], imgsz, ori_shape, ratio_pad)
        gt_classes = _as_numpy(prepared["cls"]).reshape(-1)
        objects = [
            {
                "class_id": int(class_id),
                "class_name": names[int(class_id)] if 0 <= int(class_id) < len(names) else f"class_{int(class_id)}",
                "xyxy": [float(value) for value in box],
            }
            for box, class_id in zip(gt_boxes, gt_classes)
        ]

        pred_boxes = to_original_boxes(prediction.get("bboxes"), imgsz, ori_shape, ratio_pad)
        pred_conf = _as_numpy(prediction.get("conf")).reshape(-1)
        pred_cls = _as_numpy(prediction.get("cls")).reshape(-1)
        detections = [
            {
                "class_id": int(class_id),
                "class_name": names[int(class_id)] if 0 <= int(class_id) < len(names) else f"class_{int(class_id)}",
                "confidence": float(confidence),
                "xyxy": [float(value) for value in box],
            }
            for box, confidence, class_id in zip(pred_boxes, pred_conf, pred_cls)
        ]

        capture["predictions"].append(
            {
                "image_id": image_id,
                "source": im_file,
                "source_rel": rel,
                "width": width,
                "height": height,
                "num_detections": len(detections),
                "detections": detections,
                # The native validator only reports per-image *averaged* speed; no
                # per-image timing is fabricated here (aggregate is in metrics.json).
                "timing": {},
            }
        )
        capture["ground_truth"].append(
            {
                "image_id": image_id,
                "source": im_file,
                "source_rel": rel,
                "width": width,
                "height": height,
                "num_objects": len(objects),
                "objects": objects,
            }
        )


def make_recording_validator(capture: dict[str, Any]) -> type:
    """Build a ``DetectionValidator`` subclass that records into ``capture``."""
    try:
        from ultralytics.models.yolo.detect import DetectionValidator  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - version/dependency problem
        raise AdapterError(
            f"cannot import ultralytics.models.yolo.detect.DetectionValidator "
            f"(ultralytics {_version()}): {type(exc).__name__}: {exc}"
        ) from exc

    class _RecordingDetectionValidator(DetectionValidator):
        """DetectionValidator that exports the same-pass predictions and GT."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._uavdet_capture = capture
            capture.setdefault("validators", []).append(self)

        def update_metrics(self, preds: Any, batch: Any) -> None:
            try:
                capture_batch(self, self._uavdet_capture, preds, batch)
            except Exception as exc:  # noqa: BLE001 - metrics stay the primary product
                capture["warnings"].append(
                    f"per-image export failed for one batch: {type(exc).__name__}: {exc}"
                )
            return super().update_metrics(preds, batch)

    _RecordingDetectionValidator.__name__ = "UAVDetRecordingDetectionValidator"
    _RecordingDetectionValidator.__qualname__ = "UAVDetRecordingDetectionValidator"
    return _RecordingDetectionValidator


# ---------------------------------------------------------------------------
# validation run
# ---------------------------------------------------------------------------
def validation_arguments(
    config: Mapping[str, Any],
    weights: Path,
    data_yaml: Path,
    split: str,
    device: str,
    *,
    record: bool = False,
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Every validation parameter, spelled out (B02).

    ``plots`` is forced on when ``record`` is true: the installed validator only
    fills ``self.confusion_matrix`` inside ``if self.args.plots:`` (see
    ``DetectionValidator.update_metrics``), so with ``plots=False`` the native
    matrix stays a zero array and exporting it would fabricate F08's input.

    ``save_dir`` is the directory the native validator may write into. Within a
    run, :func:`uavdet.evaluation.evaluate_model` passes
    ``<run>/evaluation`` so every native product lands in
    ``<run>/evaluation/ultralytics/`` -- the legacy bug was a ``val()`` output
    path that was not effective and ended up in an unrelated ``runs/`` tree.
    """
    evaluation = dict(config["eval"])
    if save_dir is not None:
        project = Path(save_dir)
        name = NATIVE_VAL_DIRNAME
    else:
        project = Path(str(config["project"]["workspace_root"])) / "val_runs"
        name = f"{split}_{Path(weights).stem}"
    plots = bool(evaluation["plots"])
    return {
        "data": str(data_yaml),
        "split": str(split),
        "imgsz": int(evaluation["imgsz"]),
        "conf": float(evaluation["conf"]),
        "iou": float(evaluation["iou"]),
        "max_det": int(evaluation["max_det"]),
        "batch": int(evaluation["batch"]),
        "device": str(device),
        # Never write native validation output into the cwd or a "runs/" tree:
        # the run directory keeps its own evaluation/ products.
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "plots": plots or bool(record),
        "verbose": False,
    }


def run_native_validation(
    config: Mapping[str, Any],
    weights: Path,
    data_yaml: Path,
    split: str = "val",
    *,
    record: bool = True,
    yolo_factory: Callable[[Path], Any] | None = None,
    save_dir: str | Path | None = None,
) -> ValidationOutcome:
    """Run one native validation pass and export predictions/GT from it.

    ``save_dir`` is the directory the native validator writes its own products
    into; :mod:`uavdet.evaluation` always passes the run's ``evaluation``
    directory so nothing is created outside ``outputs/<run>/``.

    Raises :class:`~uavdet.types.WeightError` for a missing weight file,
    :class:`~uavdet.types.AdapterError` when the installed version cannot provide
    the per-image evidence that ``record=True`` promises.
    """
    weights = Path(weights)
    if not weights.is_file():
        raise WeightError(f"weights not found: {weights}")
    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        raise UAVDetError(f"data.yaml not found: {data_yaml}")

    version = _version()
    warnings: list[str] = []
    choice = resolve_device(dict(config["eval"]).get("device", "cpu"))
    warnings.extend(choice.warnings)

    if yolo_factory is None:
        from .training import default_yolo_factory as yolo_factory  # noqa: PLC0415 - single source

    capture: dict[str, Any] = {
        "predictions": [],
        "ground_truth": [],
        "warnings": [],
        "validators": [],
        "seen_ids": set(),
        "view_root": data_yaml.parent,
    }
    arguments = validation_arguments(
        config, weights, data_yaml, split, choice.effective,
        record=record, save_dir=save_dir,
    )
    if record and not bool(dict(config["eval"])["plots"]):
        warnings.append(
            "eval.plots was false, but the installed validator only builds its confusion "
            "matrix when plots is on; native plotting was enabled for this pass so F08 has "
            "real counts instead of zeros"
        )
    model = yolo_factory(weights)

    metrics: Any = None
    if record:
        validator_class = make_recording_validator(capture)
        try:
            metrics = model.val(validator=validator_class, **arguments)
        except TypeError as exc:
            if "validator" not in str(exc):
                raise
            warnings.append(
                f"ultralytics {version} rejected the 'validator' hook ({exc}); the per-image "
                f"predictions/GT of this pass cannot be exported"
            )
            capture["predictions"].clear()
            capture["ground_truth"].clear()
            metrics = model.val(**arguments)
        if not capture["validators"]:
            raise AdapterError(
                f"ultralytics {version} did not instantiate the recording DetectionValidator, so "
                f"the per-image predictions and ground truth of this validation pass cannot be "
                f"exported. Refusing to present a second, differently configured predict loop as "
                f"evaluation evidence."
            )
        warnings.extend(capture["warnings"])
        if not capture["ground_truth"]:
            raise AdapterError(
                f"the recording validator of ultralytics {version} saw no image; the evaluation "
                f"would be empty and is not reported as a result"
            )
    else:
        metrics = model.val(**arguments)
        warnings.append(
            "record=False: only the aggregate metrics were collected; no per-image "
            "predictions/ground truth are available from this pass"
        )

    validator = capture["validators"][0] if capture["validators"] else None
    speed: dict[str, float] = {}
    for candidate in (getattr(validator, "speed", None), getattr(metrics, "speed", None)):
        if isinstance(candidate, Mapping) and candidate:
            speed = {str(key): float(value) for key, value in candidate.items()}
            break
    if not speed:
        warnings.append(
            f"ultralytics {version} exposed no per-image speed record for this pass; "
            f"metrics.json carries no speed_ms block"
        )

    predictions = capture["predictions"]
    ground_truth = capture["ground_truth"]
    image_ids = [str(row["image_id"]) for row in ground_truth]
    if predictions and [str(row["image_id"]) for row in predictions] != image_ids:
        raise AdapterError(
            "the recorded predictions and ground truth cover different image ids; the export "
            "would not be evidence of this pass"
        )

    details: dict[str, Any] = {
        "conf": float(arguments["conf"]),
        "iou_nms": float(arguments["iou"]),
        "imgsz": int(arguments["imgsz"]),
        "max_det": int(arguments["max_det"]),
        "split": str(split),
        "project": arguments["project"],
        "name": arguments["name"],
        "device": {"requested": choice.requested, "effective": choice.effective,
                   "warnings": list(choice.warnings)},
    }
    if validator is not None:
        validator_args = getattr(validator, "args", None)
        details["confusion_conf"] = _finite_or_none(getattr(validator_args, "conf", None))
        details["native_iou_nms"] = _finite_or_none(getattr(validator_args, "iou", None))
        details["confusion_iou"] = NATIVE_CONFUSION_IOU
        details["save_dir"] = str(getattr(validator, "save_dir", ""))
        details["images_seen"] = int(getattr(validator, "seen", 0) or 0)
        # The validator records the precision it actually ran with (None = FP32).
        details["quantize"] = getattr(validator_args, "quantize", None)

    return ValidationOutcome(
        metrics=metrics,
        predictions=predictions,
        ground_truth=ground_truth,
        image_ids=image_ids,
        speed=speed,
        evaluator_name="ultralytics.DetectionValidator",
        evaluator_version=version,
        warnings=warnings,
        details=details,
    )


# ---------------------------------------------------------------------------
# metric extraction
# ---------------------------------------------------------------------------
def extract_metrics(
    metrics: Any,
    *,
    num_images: int = 0,
    num_instances: int = 0,
    run_id: str = "",
    scope: str = "",
    split: str = "val",
    weights_sha256: str = "",
    dataset_manifest_sha256: str | None = None,
    image_ids_digest: str = "",
    data_view_root: str | None = None,
    imgsz: int = 640,
    conf: float = 0.001,
    iou_nms: float = 0.70,
    max_det: int = 300,
    precision_mode: str = "FP32",
    cases: Mapping[str, Any] | None = None,
    speed: Mapping[str, float] | None = None,
    warnings: Sequence[str] = (),
    evaluator_name: str = "ultralytics.DetectionValidator",
    evaluator_version: str | None = None,
) -> dict[str, Any]:
    """Build the ``metrics.json`` payload from a native metrics object."""
    version = evaluator_version if evaluator_version is not None else _version()
    problems = list(warnings)
    branch = _metric_branch(metrics)
    results_dict = getattr(metrics, "results_dict", None)
    results_dict = results_dict if isinstance(results_dict, Mapping) else {}

    def scalar(attribute: str, fallback_key: str) -> float | None:
        value = _finite_or_none(getattr(branch, attribute, None) if branch is not None else None)
        if value is None and fallback_key:
            value = _finite_or_none(results_dict.get(fallback_key))
        return value

    map50 = scalar("map50", "metrics/mAP50(B)")
    map50_95 = scalar("map", "metrics/mAP50-95(B)")
    precision = scalar("mp", "metrics/precision(B)")
    recall = scalar("mr", "metrics/recall(B)")
    missing = [name for name, value in
               (("map50", map50), ("map50_95", map50_95),
                ("precision", precision), ("recall", recall)) if value is None]
    status = "ok"
    if missing:
        status = "partial"
        problems.append(
            f"ultralytics {version} did not expose {', '.join(missing)}; they are null in "
            f"metrics.json instead of an invented number"
        )

    speed_ms: dict[str, float] = {}
    if speed:
        for key in ("preprocess", "inference", "postprocess", "loss"):
            value = _finite_or_none(speed.get(key))
            if value is not None:
                speed_ms[key] = value
        total = [speed_ms[key] for key in ("preprocess", "inference", "postprocess")
                 if key in speed_ms]
        if total:
            speed_ms["total"] = float(sum(total))

    return {
        "run_id": str(run_id),
        "scope": str(scope),
        "split": str(split),
        "num_images": int(num_images),
        "num_instances": int(num_instances),
        "weights_sha256": str(weights_sha256),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "evaluator_name": str(evaluator_name),
        "evaluator_version": str(version),
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou_nms": float(iou_nms),
        "max_det": int(max_det),
        "precision_mode": str(precision_mode),
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "pr_definition": PR_DEFINITION,
        "status": status,
        "warnings": problems,
        "image_ids_digest": str(image_ids_digest),
        "data_view_root": data_view_root,
        "cases": dict(cases) if cases else None,
        "speed_ms": speed_ms,
    }


def extract_per_class(
    metrics: Any,
    num_gt: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Per-class P/R/AP rows in the fixed class order.

    ``None`` marks "not computable" (no ground truth, or a field the installed
    version does not expose); it is never rendered as ``0``.
    """
    names = _class_names(metrics)
    count = len(names)
    branch = _metric_branch(metrics)
    array_p = _array(branch, "p")
    array_r = _array(branch, "r")
    array_ap50 = _array(branch, "ap50")
    array_ap = _array(branch, "ap")
    if array_p is None and array_r is None and array_ap is None and array_ap50 is None:
        raise AdapterError(
            f"ultralytics {_version()} exposed no per-class arrays on the metrics object "
            f"(looked for box.p/r/ap50/ap); per_class.csv cannot be produced without "
            f"fabricating it"
        )

    positions = _class_positions(branch, count)
    derived_counts = _gt_counts(metrics, count)
    if num_gt is not None:
        gt_counts: list[int | None] = [int(value) for value in num_gt][:count]
    elif derived_counts is not None:
        gt_counts = list(derived_counts)
    else:
        gt_counts = [None] * count

    rows: list[dict[str, Any]] = []
    for class_id in range(count):
        gt = gt_counts[class_id] if class_id < len(gt_counts) else None
        row: dict[str, Any] = {
            "class_id": class_id,
            "class_name": names[class_id],
            "num_gt": gt,
            "precision": None,
            "recall": None,
            "ap50": None,
            "ap50_95": None,
        }
        if positions is None:
            index: int | None = class_id
            has_gt = (gt is None) or int(gt) > 0
        else:
            index = positions.get(class_id)
            has_gt = index is not None and ((gt is None) or int(gt) > 0)
        if has_gt and index is not None:
            row["precision"] = _finite_or_none(array_p[index]) if array_p is not None and index < array_p.size else None
            row["recall"] = _finite_or_none(array_r[index]) if array_r is not None and index < array_r.size else None
            row["ap50"] = _finite_or_none(array_ap50[index]) if array_ap50 is not None and index < array_ap50.size else None
            row["ap50_95"] = _finite_or_none(array_ap[index]) if array_ap is not None and index < array_ap.size else None
        rows.append(row)
    return rows


def _curve_rows(
    values: Any,
    order: Sequence[int],
    count: int,
) -> np.ndarray | None:
    """Place native per-class curve rows at their class-id row, NaN elsewhere."""
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        return None
    points = array.shape[1]
    placed = np.full((count + 1, points), np.nan, dtype=np.float64)
    for position, class_id in enumerate(order):
        if position < array.shape[0] and 0 <= class_id < count:
            placed[class_id] = array[position]
    return placed


def _legacy_curves(metrics: Any) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Group an older-style ``metrics.curves`` list into named curve families."""
    curves = getattr(metrics, "curves", None)
    if not curves:
        return None
    try:
        items = list(curves)
    except TypeError:
        return None
    if not items:
        return None
    grouped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for item in items:
        x = getattr(item, "x", None)
        y = getattr(item, "y", None)
        if x is None or y is None:
            return None
        xlabel = str(getattr(item, "xlabel", "") or "")
        ylabel = str(getattr(item, "ylabel", "") or "")
        key = _curve_family(xlabel, ylabel)
        if key is None:
            continue
        x_array = np.asarray(x, dtype=np.float64).reshape(-1)
        y_array = np.asarray(y, dtype=np.float64).reshape(-1)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = (x_array, y_array.reshape(1, -1))
        else:
            if existing[0].shape != x_array.shape:
                return None
            grouped[key] = (existing[0], np.vstack([existing[1], y_array.reshape(1, -1)]))
    return grouped or None


def extract_curves(
    metrics: Any,
    *,
    version: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    """Export the native PR / F1 / P / R curves, or ``None`` when not exposed."""
    names = _class_names(metrics)
    count = len(names)
    branch = _metric_branch(metrics)
    positions = _class_positions(branch, count)
    order: list[int]
    if positions is None:
        order = list(range(count))
    else:
        order = [class_id for class_id, _ in sorted(positions.items(), key=lambda item: item[1])]

    arrays: dict[str, np.ndarray] = {}
    recall_axis: np.ndarray | None = None
    conf_axis: np.ndarray | None = None
    source_field = ""

    px = _array(branch, "px")
    prec_values = _array(branch, "prec_values")
    f1_curve = _array(branch, "f1_curve")
    p_curve = _array(branch, "p_curve")
    r_curve = _array(branch, "r_curve")
    if px is not None and prec_values is not None:
        source_field = "metrics.box.px / prec_values / f1_curve / p_curve / r_curve"
        recall_axis = np.asarray(px, dtype=np.float64).reshape(-1)
        conf_axis = recall_axis
        for key, values in (("pr_precision", prec_values), ("f1", f1_curve),
                            ("p_conf", p_curve), ("r_conf", r_curve)):
            placed = _curve_rows(values, order, count)
            if placed is None:
                continue
            arrays[key] = placed
    else:
        results = getattr(branch, "curves_results", None) if branch is not None else None
        labelled = _curves_from_results(results)
        if labelled is None:
            labelled = _legacy_curves(metrics)
            if labelled is not None:
                source_field = "metrics.curves (per-class Curve objects)"
        else:
            source_field = "metrics.box.curves_results"
        if labelled is None:
            return None
        for label, (x_array, y_array) in labelled.items():
            if label == "pr":
                recall_axis = x_array
                arrays["pr_precision"] = _curve_rows(y_array, order, count)
            elif label == "p_conf":
                conf_axis = x_array
                arrays["p_conf"] = _curve_rows(y_array, order, count)
            elif label == "r_conf":
                conf_axis = conf_axis if conf_axis is not None else x_array
                arrays["r_conf"] = _curve_rows(y_array, order, count)
            elif label == "f1":
                conf_axis = conf_axis if conf_axis is not None else x_array
                arrays["f1"] = _curve_rows(y_array, order, count)

    required = ("pr_precision", "f1", "p_conf", "r_conf")
    if any(key not in arrays or arrays[key] is None for key in required):
        return None
    if recall_axis is None or conf_axis is None:
        return None
    point_counts = {arrays[key].shape[1] for key in required}
    if (len(point_counts) != 1
            or int(recall_axis.size) not in point_counts
            or int(conf_axis.size) not in point_counts):
        return None
    point_count = point_counts.pop()

    gt_counts = _gt_counts(metrics, count)
    valid = np.zeros(count, dtype=bool)
    for class_id in range(count):
        if positions is not None:
            has_gt = class_id in positions
        else:
            has_gt = True
        if gt_counts is not None and gt_counts[class_id] is not None:
            has_gt = has_gt and int(gt_counts[class_id]) > 0
        valid[class_id] = bool(has_gt)

    for key in required:
        rows = arrays[key]
        with_mean = np.full((count + 1, point_count), np.nan, dtype=np.float64)
        with_mean[:count] = rows[:count]
        if valid.any():
            with_mean[count] = np.nanmean(rows[:count][valid], axis=0)
        arrays[key] = with_mean

    arrays["recall"] = np.asarray(recall_axis, dtype=np.float64).reshape(-1)
    arrays["conf"] = np.asarray(conf_axis, dtype=np.float64).reshape(-1)
    arrays["valid_classes"] = valid

    meta: dict[str, Any] = {
        "source": {
            "evaluator": "ultralytics.DetMetrics",
            "version": version if version is not None else _version(),
            "field": source_field,
        },
        "class_order": list(names),
        "mean_row_index": count,
        "smoothing": "native curves, no extra smoothing applied",
        "arrays": {
            "recall": {"axes": ["point"], "units": "fraction 0-1",
                       "definition": "native PR-curve x axis (recall)"},
            "conf": {"axes": ["point"], "units": "fraction 0-1",
                     "definition": "native confidence sweep used by the F1/P/R curves"},
            "pr_precision": {"axes": ["class+mean", "recall"], "units": "fraction 0-1"},
            "f1": {"axes": ["class+mean", "conf"], "units": "fraction 0-1"},
            "p_conf": {"axes": ["class+mean", "conf"], "units": "fraction 0-1"},
            "r_conf": {"axes": ["class+mean", "conf"], "units": "fraction 0-1"},
            "valid_classes": {"axes": ["class"], "units": "bool",
                              "definition": "class has ground truth, so its AP is computable"},
        },
        "valid_classes": [int(class_id) for class_id in np.flatnonzero(valid)],
        "row_mapping": (
            "row i is class id i from class_order; the last row is the mean over the "
            "rows listed in valid_classes"
        ),
        "invalid_rows": (
            "rows of classes that are not in valid_classes are NaN (no ground truth in this "
            "evaluation); renderers must skip them instead of drawing a zero curve"
        ),
        "mean_row": "arithmetic mean over the valid class rows of the same array",
    }
    return arrays, meta


def _curves_from_results(results: Any) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Group ``Metric.curves_results`` (``[x, y, xlabel, ylabel]``) into families."""
    if not results:
        return None
    try:
        entries = list(results)
    except TypeError:
        return None
    if not entries:
        return None
    grouped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 4:
            return None
        x, y, xlabel, ylabel = entry[0], entry[1], str(entry[2]), str(entry[3])
        if x is None or y is None:
            return None
        key = _curve_family(xlabel, ylabel)
        if key is None:
            continue
        x_array = np.asarray(x, dtype=np.float64).reshape(-1)
        y_array = np.asarray(y, dtype=np.float64)
        if y_array.ndim == 1:
            y_array = y_array.reshape(1, -1)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = (x_array, y_array)
        else:
            if existing[0].shape != x_array.shape:
                return None
            grouped[key] = (existing[0], np.vstack([existing[1], y_array]))
    return grouped or None


def _curve_family(xlabel: str, ylabel: str) -> str | None:
    x = xlabel.strip().lower()
    y = ylabel.strip().lower()
    if "recall" in x and y.startswith("precision"):
        return "pr"
    if "conf" in x and y.startswith("precision"):
        return "p_conf"
    if "conf" in x and y.startswith("recall"):
        return "r_conf"
    if "conf" in x and y == "f1":
        return "f1"
    return None


def extract_confusion(
    metrics: Any,
    *,
    conf: float | None = None,
    iou: float | None = None,
    version: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Export the native confusion matrix as ``int64 [true, predicted]``.

    The installed Ultralytics indexes its matrix ``[predicted, true]`` (see
    ``ConfusionMatrix.plot``: x label "True", y label "Predicted"); the frozen
    contract wants ``row = ground truth, column = prediction``, so the array is
    transposed on export and the reason is recorded in the metadata.
    """
    confusion = getattr(metrics, "confusion_matrix", None)
    matrix = getattr(confusion, "matrix", None)
    if matrix is None:
        return None
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.size == 0:
        return None
    if not np.all(np.isfinite(array)):
        return None
    native = np.rint(array).astype(np.int64)
    exported = np.ascontiguousarray(native.T)
    size = exported.shape[0]
    names = _class_names(metrics)
    if len(names) == size:
        class_order = list(names)
    else:
        class_order = list(CLASS_NAMES[: size - 1]) if size - 1 <= NUM_CLASSES else [
            f"class_{index}" for index in range(size - 1)
        ]
    meta: dict[str, Any] = {
        "true_axis": "row",
        "pred_axis": "column",
        "background_index": size - 1,
        "class_order": class_order,
        "conf": None if conf is None else float(conf),
        "iou": None if iou is None else float(iou),
        "normalization": "raw integer counts; GT-normalised only at plot time",
        "source_version": version if version is not None else _version(),
        "source": "ultralytics.utils.metrics.ConfusionMatrix",
        "source_orientation": (
            "the native matrix is indexed [predicted, true]; it is transposed on export so that "
            "row = ground truth and column = prediction"
        ),
        "matching": (
            "native ConfusionMatrix.process_batch: greedy one-to-one, class-agnostic matching "
            "at IoU > iou and confidence > conf"
        ),
        "notes": (
            "the native validator matches the confusion matrix at its own threshold "
            "(conf = eval conf, IoU = 0.5 by the installed implementation), which is not the "
            "GUI/case-analysis threshold of 0.25"
        ),
    }
    return exported, meta
