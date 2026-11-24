"""Evaluation of a trained 10-class checkpoint and export of all evidence.

``evaluate_model`` is the only writer of ``<run>/evaluation`` and it produces
every file listed in ``docs/INTERFACE_CONTRACT.md`` §1:

``metrics.json``, ``per_class.csv``, ``curves.npz`` + ``curves_meta.json``,
``confusion_matrix.npz`` + ``confusion_meta.json``, ``predictions.jsonl``,
``ground_truth.jsonl``, ``cases.json`` and ``evaluation.log``.

Two definitions that must never be mixed up (and are therefore spelled out in
the artefacts themselves):

* the **native mAP** numbers (mAP50, mAP50-95, native P/R) come straight from
  the one Ultralytics validation pass;
* the **fixed-threshold case analysis** (:func:`match_detections`,
  ``conf >= 0.25``, class-aware IoU ``>= 0.5``, greedy one-to-one) is a separate
  computation used to pick the success/failure examples. It is not mAP and never
  replaces it.

Data honesty: a class without ground truth is ``N/A`` (never ``0``), and a curve
or matrix field the installed Ultralytics does not expose is simply not written;
``metrics.json.status`` then becomes ``"partial"`` with a warning instead of
inventing an array.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import ultralytics_adapter
from .artifacts import (
    now_iso,
    read_json,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_text_atomic,
)
from .training import resolve_data_yaml
from .types import (
    CLASS_NAMES,
    NUM_CLASSES,
    EvaluationError,
    RunContext,
    UAVDetError,
    WeightError,
    bbox_iou,
)

__all__ = [
    "evaluate_model",
    "default_adapter",
    "match_detections",
    "select_cases",
]

#: Files this module owns (used by the failure path and by the tests).
EVALUATION_FILES: tuple[str, ...] = (
    "metrics.json",
    "per_class.csv",
    "predictions.jsonl",
    "ground_truth.jsonl",
    "cases.json",
    "evaluation.log",
    "case_selection.json",
)

MATCH_DEFINITION = (
    "fixed-threshold case analysis (separate from the native mAP): detections with "
    "confidence >= conf are visited in descending confidence order and greedily claim the "
    "best unclaimed ground-truth box of the SAME class with IoU >= iou (one-to-one). "
    "tp/fp/fn below are counts of this matching only; they are not precision/recall of the "
    "native PR curve and not a substitute for mAP."
)


# ---------------------------------------------------------------------------
# adapter wiring
# ---------------------------------------------------------------------------
def default_adapter(
    config: Mapping[str, Any],
    weights: Path,
    data_yaml: Path,
    split: str = "val",
    *,
    record: bool = True,
    save_dir: str | Path | None = None,
) -> ultralytics_adapter.ValidationOutcome:
    """The real adapter (Ultralytics imported lazily inside it).

    ``save_dir`` is where the native validator may write its own products; the
    evaluation stage always passes the run's ``evaluation`` directory so all
    validation output stays inside ``outputs/<run>/``.
    """
    return ultralytics_adapter.run_native_validation(
        config, Path(weights), Path(data_yaml), split, record=record, save_dir=save_dir
    )


def _call_default_adapter(
    config: Mapping[str, Any],
    weights: Path,
    data_yaml: Path,
    split: str,
    run: RunContext,
) -> Any:
    """Call :func:`default_adapter`, passing the run directory when it accepts it.

    A monkeypatched/replacement adapter with the documented five-argument
    signature keeps working: ``save_dir`` is only passed when the callable really
    declares it.
    """
    import inspect  # noqa: PLC0415 - trivial, kept local to the call site

    accepts_save_dir = False
    try:
        accepts_save_dir = "save_dir" in inspect.signature(default_adapter).parameters
    except (TypeError, ValueError):
        accepts_save_dir = False
    if accepts_save_dir:
        return default_adapter(
            config, weights, data_yaml, split, record=True, save_dir=run.evaluation_dir
        )
    return default_adapter(config, weights, data_yaml, split, record=True)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _native_class_names(metrics: Any) -> list[str]:
    raw = getattr(metrics, "names", None)
    if isinstance(raw, Mapping) and raw:
        try:
            ordered = sorted((int(key), str(value)) for key, value in raw.items())
        except (TypeError, ValueError):
            return []
        if [index for index, _ in ordered] == list(range(len(ordered))):
            return [name for _, name in ordered]
        return []
    if isinstance(raw, (list, tuple)) and raw:
        return [str(name) for name in raw]
    return []


def _native_class_count(metrics: Any) -> int | None:
    raw = getattr(metrics, "nt_per_class", None)
    if raw is not None:
        try:
            size = int(np.asarray(raw).reshape(-1).size)
        except (TypeError, ValueError):
            size = 0
        if size:
            return size
    branch = getattr(metrics, "box", None)
    for attribute in ("p", "r", "all_ap"):
        value = getattr(branch, attribute, None) if branch is not None else None
        if value is None:
            continue
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if array.size:
            return int(array.shape[0])
    return None


def _write_log(lines: Sequence[str], run: RunContext) -> Path:
    """Append lines to ``evaluation/evaluation.log`` (never truncates history)."""
    target = run.evaluation_dir / "evaluation.log"
    run.evaluation_dir.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{now_iso()} {line}\n")
    return target


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """One JSON object per line, NaN/inf rejected (they would not parse)."""
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )
    return write_text_atomic(path, payload)


def _require_finite(values: Sequence[float], where: str) -> None:
    for value in values:
        if not np.isfinite(float(value)):
            raise EvaluationError(f"{where} contains a non-finite coordinate: {value!r}")


def _view_split_ids(data_yaml: Path, split: str) -> list[str] | None:
    """Image ids of the dataset view for ``split`` (``None`` when unavailable)."""
    from .data import load_view_manifest  # noqa: PLC0415 - avoid an import cycle

    try:
        manifest = load_view_manifest(data_yaml)
    except Exception:  # noqa: BLE001 - a hand-written data.yaml has no view manifest
        return None
    entries = ((manifest.get("splits") or {}).get(split) or {}).get("entries") or []
    if not entries:
        # The view simply has no record for this split (e.g. an external test set).
        return None
    return [str(entry["image_id"]) for entry in entries]


def _view_root(data_yaml: Path) -> Path:
    return Path(data_yaml).resolve().parent


def _dataset_manifest_sha256(run: RunContext, data_yaml: Path) -> str | None:
    from .data import load_view_manifest  # noqa: PLC0415 - avoid an import cycle

    try:
        manifest = load_view_manifest(data_yaml)
    except Exception:  # noqa: BLE001 - fall through to the audit copy
        manifest = {}
    digest = manifest.get("source_manifest_digest")
    if digest:
        return str(digest)
    audit_path = run.audit_dir / "data_manifest.json"
    if audit_path.exists():
        try:
            return str(read_json(audit_path).get("manifest_digest") or "")
        except Exception:  # noqa: BLE001 - an unreadable audit is not fatal here
            return None
    return None


# ---------------------------------------------------------------------------
# fixed-threshold matching (separate from native mAP)
# ---------------------------------------------------------------------------
def match_detections(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    conf: float = 0.25,
    iou: float = 0.5,
) -> dict[str, Any]:
    """Greedy one-to-one, class-aware matching at a fixed confidence/IoU.

    Returns per-image ``tp/fp/fn`` (with the matched indices) and totals. This is
    the case-analysis definition, independent of the native mAP numbers.
    """
    conf = float(conf)
    iou = float(iou)
    if not (0.0 <= conf <= 1.0):
        raise EvaluationError(f"case conf must be within [0, 1], got {conf}")
    if not (0.0 < iou <= 1.0):
        raise EvaluationError(f"case IoU must be within (0, 1], got {iou}")

    gt_by_id = {str(row["image_id"]): row for row in ground_truth}
    pred_by_id = {str(row["image_id"]): row for row in predictions}
    if set(gt_by_id) != set(pred_by_id):
        only_gt = sorted(set(gt_by_id) - set(pred_by_id))[:5]
        only_pred = sorted(set(pred_by_id) - set(gt_by_id))[:5]
        raise EvaluationError(
            "predictions and ground truth do not cover the same images "
            f"(only in GT: {only_gt}, only in predictions: {only_pred})"
        )
    if len(gt_by_id) != len(ground_truth) or len(pred_by_id) != len(predictions):
        raise EvaluationError("duplicate image_id in predictions/ground truth")

    totals = {"tp": 0, "fp": 0, "fn": 0, "num_gt": 0, "num_detections": 0}
    images: list[dict[str, Any]] = []
    for image_id in sorted(gt_by_id):
        gt_row = gt_by_id[image_id]
        pred_row = pred_by_id[image_id]
        gts = list(gt_row.get("objects") or [])
        dets = list(pred_row.get("detections") or [])

        candidates = [
            {"index": index, "class_id": int(det["class_id"]),
             "confidence": float(det["confidence"]), "xyxy": [float(v) for v in det["xyxy"]]}
            for index, det in enumerate(dets)
            if float(det["confidence"]) >= conf
        ]
        # Descending confidence, original index as the stable tie-break.
        candidates.sort(key=lambda item: (-item["confidence"], item["index"]))
        claimed = [False] * len(gts)
        matches: list[dict[str, Any]] = []
        false_positive_indices: list[int] = []
        for candidate in candidates:
            best_value = -1.0
            best_index: int | None = None
            for gt_index, gt in enumerate(gts):
                if claimed[gt_index]:
                    continue
                if int(gt["class_id"]) != candidate["class_id"]:
                    continue
                value = bbox_iou(candidate["xyxy"], [float(v) for v in gt["xyxy"]])
                if value >= iou and value > best_value:
                    best_value = value
                    best_index = gt_index
            if best_index is None:
                false_positive_indices.append(candidate["index"])
                continue
            claimed[best_index] = True
            matches.append(
                {
                    "det_index": candidate["index"],
                    "gt_index": best_index,
                    "class_id": candidate["class_id"],
                    "confidence": candidate["confidence"],
                    "iou": float(best_value),
                }
            )
        false_negative_indices = [index for index, flag in enumerate(claimed) if not flag]
        true_positives = len(matches)
        false_positives = len(false_positive_indices)
        false_negatives = len(false_negative_indices)
        num_gt = len(gts)
        confidences = [float(det["confidence"]) for det in dets]
        totals["tp"] += true_positives
        totals["fp"] += false_positives
        totals["fn"] += false_negatives
        totals["num_gt"] += num_gt
        totals["num_detections"] += len(dets)
        # Case-selection features (class mix and relative GT box size).
        class_names = sorted({str(obj.get("class_name") or obj.get("class_id"))
                              for obj in gts})
        width = float(gt_row.get("width") or 0) or 0.0
        height = float(gt_row.get("height") or 0) or 0.0
        areas: list[float] = []
        if width > 0 and height > 0:
            for obj in gts:
                x1, y1, x2, y2 = (float(v) for v in obj.get("xyxy", [0, 0, 0, 0]))
                areas.append(max(0.0, (x2 - x1) / width) * max(0.0, (y2 - y1) / height))
        areas.sort()
        median_area = areas[len(areas) // 2] if areas else None
        images.append(
            {
                "image_id": image_id,
                "num_gt": num_gt,
                "num_detections": len(dets),
                "num_candidates": len(candidates),
                "classes": class_names,
                "gt_median_rel_area": median_area,
                "tp": true_positives,
                "fp": false_positives,
                "fn": false_negatives,
                "max_conf": max(confidences) if confidences else None,
                "matches": matches,
                "false_positive_indices": false_positive_indices,
                "false_negative_indices": false_negative_indices,
            }
        )

    precision = (totals["tp"] / (totals["tp"] + totals["fp"])
                 if (totals["tp"] + totals["fp"]) else None)
    recall = (totals["tp"] / (totals["tp"] + totals["fn"])
              if (totals["tp"] + totals["fn"]) else None)
    totals["precision"] = precision
    totals["recall"] = recall
    return {
        "conf": conf,
        "iou": iou,
        "definition": MATCH_DEFINITION,
        "totals": totals,
        "images": images,
    }


def _case_features(image: Mapping[str, Any]) -> dict[str, Any]:
    """Cheap, recorded features of one case (class mix, relative GT size)."""
    classes = sorted({str(name) for name in image.get("classes") or []})
    area = image.get("gt_median_rel_area")
    return {
        "classes": classes,
        "gt_median_rel_area": None if area is None else float(area),
        "num_gt": int(image.get("num_gt") or 0),
        "num_candidates": int(image.get("num_candidates") or 0),
    }


def select_cases(match: Mapping[str, Any], *, max_cases: int = 6,
                 explicit_ids: Sequence[str] = ()) -> dict[str, Any]:
    """Deterministically pick the success/failure examples for F11/F12.

    ``explicit_ids`` (config ``eval.case_images``) are always included first and
    labelled with ``criterion="user_selected"`` so a hand-picked example can never
    be mistaken for an automatically ranked one.
    """
    limit = max(1, int(max_cases))
    conf = float(match.get("conf", 0.25))
    iou = float(match.get("iou", 0.5))
    images = list(match.get("images") or [])
    by_id = {str(image["image_id"]): image for image in images}
    explicit = [by_id[str(image_id)] for image_id in explicit_ids if str(image_id) in by_id]
    ignored_explicit = [str(image_id) for image_id in explicit_ids if str(image_id) not in by_id]

    successful = [
        image for image in images
        if int(image["num_gt"]) > 0 and int(image["fn"]) == 0 and int(image["fp"]) == 0
    ]
    criterion = "strict_no_fn_no_fp"
    if not successful:
        # Documented fallback for dense scenes (VisDrone images hold 40-900 objects,
        # so "zero FN and zero FP" is rarely met even by a good model). Cases must
        # still be honest: high precision and a substantial recall share, and every
        # record carries the criterion so the figure can say so.
        successful = [
            image for image in images
            if int(image["tp"]) >= 3
            and int(image["tp"]) + int(image["fp"]) > 0
            and int(image["tp"]) / (int(image["tp"]) + int(image["fp"])) >= 0.8
            and int(image["tp"]) / max(1, int(image["num_gt"])) >= 0.4
        ]
        if successful:
            criterion = "relaxed_high_precision_partial_recall"
    successful.sort(
        key=lambda image: (
            -int(image["tp"]),
            -int(image["num_gt"]),
            -(float(image["max_conf"]) if image.get("max_conf") is not None else 0.0),
            str(image["image_id"]),
        )
    )
    failure = [
        image for image in images if int(image["fn"]) > 0 or int(image["fp"]) > 0
    ]
    failure.sort(
        key=lambda image: (
            -(int(image["fn"]) + int(image["fp"])),
            -int(image["fn"]),
            -int(image["num_gt"]),
            str(image["image_id"]),
        )
    )

    success_records = [
        {
            "image_id": str(image["image_id"]),
            "criterion": "user_selected",
            "reason": "explicitly requested through eval.case_images (not an automatic ranking)",
            "num_gt": int(image["num_gt"]),
            "tp": int(image["tp"]),
            "fp": int(image["fp"]),
            "fn": int(image["fn"]),
            "max_conf": image.get("max_conf"),
            **_case_features(image),
        }
        for image in explicit[:limit]
    ] + [
        {
            "image_id": str(image["image_id"]),
            "criterion": criterion,
            "reason": (
                f"all {int(image['num_gt'])} ground-truth object(s) matched at "
                f"conf>={conf:g} and IoU>={iou:g}, no false positive"
                if criterion == "strict_no_fn_no_fp"
                else (
                    f"relaxed success: {int(image['tp'])}/{int(image['num_gt'])} ground-truth "
                    f"object(s) matched at conf>={conf:g} and IoU>={iou:g} with "
                    f"{int(image['fp'])} false positive(s); the strict no-FN/no-FP rule had "
                    f"no candidate in this validation set"
                )
            ),
            "num_gt": int(image["num_gt"]),
            "tp": int(image["tp"]),
            "fp": int(image["fp"]),
            "fn": int(image["fn"]),
            "max_conf": image.get("max_conf"),
            **_case_features(image),
        }
        for image in successful[:limit - len(explicit)]
    ]
    failure_records = []
    for image in failure[:limit]:
        kinds: list[str] = []
        if int(image["fn"]) > 0:
            kinds.append("false_negative")
        if int(image["fp"]) > 0:
            kinds.append("false_positive")
        if int(image["num_gt"]) > 0 and int(image["num_candidates"]) == 0:
            kinds.append("no_detection_above_conf")
        parts = []
        if int(image["fn"]) > 0:
            parts.append(f"missed {int(image['fn'])} ground-truth object(s)")
        if int(image["fp"]) > 0:
            parts.append(f"{int(image['fp'])} false positive(s)")
        failure_records.append(
            {
                "image_id": str(image["image_id"]),
                "reason": "; ".join(parts) if parts else "matched nothing",
                "num_gt": int(image["num_gt"]),
                "tp": int(image["tp"]),
                "fp": int(image["fp"]),
                "fn": int(image["fn"]),
                "failure_kinds": kinds,
                **_case_features(image),
            }
        )
    return {
        "created_at": now_iso(),
        "conf": conf,
        "iou": iou,
        "successful_criterion": criterion,
        "explicit_case_images": [str(image["image_id"]) for image in explicit],
        "ignored_case_images": ignored_explicit,
        "selection_rules": {
            "successful": (
                "no false negative and no false positive at the fixed threshold; ranked by "
                "matched objects, then ground-truth count, then highest confidence, then image_id"
            ),
            "successful_relaxed": (
                "only when no strict candidate exists: tp>=3, precision>=0.8 and tp/num_gt>=0.4, "
                "ranked like the strict rule; every record carries its criterion"
            ),
            "failure": (
                "at least one false negative or false positive at the fixed threshold; ranked by "
                "error count, then false negatives, then ground-truth count, then image_id"
            ),
            "threshold_note": MATCH_DEFINITION,
        },
        "successful": success_records,
        "failure": failure_records,
    }


# ---------------------------------------------------------------------------
# the evaluation stage
# ---------------------------------------------------------------------------
def evaluate_model(
    config: Mapping[str, Any],
    run: RunContext,
    weights: Path,
    split: str = "val",
    *,
    data_yaml: Path | None = None,
    adapter: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Validate ``weights`` on ``split`` and write every evaluation artefact."""
    run.evaluation_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(weights)
    if not weights.is_file():
        raise WeightError(
            f"weights not found: {weights}; the evaluation must run on the real best.pt "
            f"of this run (or an explicitly passed 10-class checkpoint)"
        )
    data_yaml_path = resolve_data_yaml(config, run, data_yaml)
    weights_sha256 = sha256_file(weights)
    evaluation = dict(config["eval"])
    log_lines: list[str] = [
        f"run={run.run_id} scope={config.get('scope', run.scope)} split={split}",
        f"weights={weights} sha256={weights_sha256}",
        f"data_yaml={data_yaml_path}",
        f"protocol: imgsz={evaluation['imgsz']} conf={evaluation['conf']} "
        f"iou_nms={evaluation['iou']} max_det={evaluation['max_det']} batch={evaluation['batch']}",
    ]
    try:
        return _evaluate(
            config=config,
            run=run,
            weights=weights,
            weights_sha256=weights_sha256,
            data_yaml=data_yaml_path,
            split=split,
            adapter=adapter,
            log_lines=log_lines,
        )
    except BaseException as exc:  # noqa: BLE001 - log, then re-raise
        _write_log(log_lines + [f"FAILED {type(exc).__name__}: {exc}"], run)
        _write_failure_metrics(run, config, split, weights_sha256, exc)
        if not isinstance(exc, Exception) or isinstance(exc, UAVDetError):
            raise
        raise EvaluationError(f"evaluation failed: {type(exc).__name__}: {exc}") from exc


def _write_failure_metrics(
    run: RunContext,
    config: Mapping[str, Any],
    split: str,
    weights_sha256: str,
    exc: BaseException,
) -> None:
    """A failed evaluation must not leave a half-written success claim behind."""
    payload = {
        "run_id": run.run_id,
        "scope": str(config.get("scope", run.scope)),
        "split": str(split),
        "num_images": 0,
        "num_instances": 0,
        "weights_sha256": weights_sha256,
        "status": "failed",
        "warnings": [f"{type(exc).__name__}: {exc}"],
        "map50": None,
        "map50_95": None,
        "precision": None,
        "recall": None,
    }
    try:
        write_json_atomic(run.evaluation_dir / "metrics.json", payload)
    except Exception:  # noqa: BLE001 - the original error is what matters
        return


def _evaluate(
    *,
    config: Mapping[str, Any],
    run: RunContext,
    weights: Path,
    weights_sha256: str,
    data_yaml: Path,
    split: str,
    adapter: Callable[..., Any] | None,
    log_lines: list[str],
) -> dict[str, Any]:
    evaluation = dict(config["eval"])
    if adapter is None:
        outcome = _call_default_adapter(config, weights, data_yaml, split, run)
    else:
        # Injected adapters keep the frozen signature (the integration tests use it).
        outcome = adapter(config, weights, data_yaml, split, record=True)

    warnings: list[str] = list(getattr(outcome, "warnings", []) or [])
    metrics_object = getattr(outcome, "metrics", None)
    if metrics_object is None:
        raise EvaluationError("the adapter returned no native metrics object")

    names = _native_class_names(metrics_object)
    class_count = len(names) if names else _native_class_count(metrics_object)
    if class_count != NUM_CLASSES:
        raise WeightError(
            f"the evaluated weights do not describe {NUM_CLASSES} classes "
            f"(found {class_count if class_count is not None else 'an unverifiable count'}"
            f"{', names: ' + str(names) if names else ''}); this project evaluates the fixed "
            f"VisDrone class order {list(CLASS_NAMES)} only"
        )
    if names and tuple(names) != CLASS_NAMES:
        raise WeightError(
            f"the evaluated weights use the class order {names}, but this project requires "
            f"{list(CLASS_NAMES)}; per-class evidence would be attributed to the wrong class"
        )
    if not names:
        warnings.append(
            "the native metrics object exposed no class names; only the class count could be "
            "verified against the fixed VisDrone order"
        )

    predictions = [dict(row) for row in getattr(outcome, "predictions", []) or []]
    ground_truth = [dict(row) for row in getattr(outcome, "ground_truth", []) or []]
    image_ids = [str(value) for value in getattr(outcome, "image_ids", []) or []]
    if not ground_truth:
        raise EvaluationError("the adapter exported no ground truth; nothing was evaluated")
    if not predictions:
        raise EvaluationError(
            "the adapter exported no predictions, so the per-image evidence of this pass is "
            "missing; refusing to write an evaluation that cannot be traced"
        )
    if image_ids != [str(row["image_id"]) for row in ground_truth]:
        raise EvaluationError(
            "the adapter's image_ids do not match its ground-truth rows; the evaluated id set "
            "is inconsistent"
        )
    if len(set(image_ids)) != len(image_ids):
        raise EvaluationError("the adapter exported duplicate image ids")

    expected_ids = _view_split_ids(data_yaml, split)
    if expected_ids is not None:
        expected = sorted(expected_ids)
        evaluated = sorted(image_ids)
        if expected != evaluated:
            missing = sorted(set(expected) - set(evaluated))[:5]
            extra = sorted(set(evaluated) - set(expected))[:5]
            raise EvaluationError(
                f"the validation pass evaluated {len(evaluated)} image(s) but the dataset view "
                f"of split '{split}' lists {len(expected)} (missing: {missing}, unexpected: "
                f"{extra}). An incomplete validation is not reported as a result."
            )
    elif run.scope == "formal":
        warnings.append(
            "the dataset view manifest was not readable, so the evaluated image ids could only "
            "be checked for internal consistency"
        )

    _validate_rows(predictions, ground_truth)

    num_images = len(image_ids)
    num_instances = sum(int(row["num_objects"]) for row in ground_truth)
    gt_counts = [0] * NUM_CLASSES
    for row in ground_truth:
        for obj in row["objects"]:
            class_id = int(obj["class_id"])
            if not 0 <= class_id < NUM_CLASSES:
                raise EvaluationError(f"ground truth contains class id {class_id} outside 0-9")
            gt_counts[class_id] += 1
    for row in predictions:
        for det in row["detections"]:
            class_id = int(det["class_id"])
            if not 0 <= class_id < NUM_CLASSES:
                raise WeightError(
                    f"a prediction carries class id {class_id} outside the 10 VisDrone classes; "
                    f"the checkpoint is not a 10-class VisDrone model"
                )

    image_ids_digest = sha256_text("\n".join(sorted(image_ids)))
    details = getattr(outcome, "details", {}) or {}

    # ---- per class ---------------------------------------------------------
    per_class = ultralytics_adapter.extract_per_class(metrics_object, gt_counts)
    _write_per_class_csv(run.evaluation_dir / "per_class.csv", per_class)
    for row in per_class:
        if row["num_gt"] == 0:
            warnings.append(
                f"class {row['class_id']} ({row['class_name']}) has no ground truth in this "
                f"split: its metrics are N/A, not 0"
            )

    # ---- native curves / confusion matrix ----------------------------------
    status = "ok"
    curves = ultralytics_adapter.extract_curves(
        metrics_object, version=getattr(outcome, "evaluator_version", None)
    )
    if curves is None:
        status = "partial"
        warnings.append(
            f"ultralytics {getattr(outcome, 'evaluator_version', '?')} exposed no native curve "
            f"arrays; curves.npz/curves_meta.json were not written instead of being invented"
        )
    confusion = ultralytics_adapter.extract_confusion(
        metrics_object,
        conf=details.get("confusion_conf", evaluation["conf"]),
        iou=details.get("confusion_iou"),
        version=getattr(outcome, "evaluator_version", None),
    )
    if confusion is None:
        status = "partial"
        warnings.append(
            f"ultralytics {getattr(outcome, 'evaluator_version', '?')} exposed no native "
            f"confusion matrix; confusion_matrix.npz/confusion_meta.json were not written"
        )
    elif num_instances > 0 and int(confusion[0].sum()) == 0:
        status = "partial"
        warnings.append(
            "the native confusion matrix is all zeros even though ground truth was evaluated; "
            "confusion_matrix.npz/confusion_meta.json were not written instead of publishing a "
            "matrix that was never filled"
        )
        confusion = None

    # ---- fixed-threshold case analysis -------------------------------------
    match = match_detections(
        predictions,
        ground_truth,
        conf=float(evaluation["case_conf"]),
        iou=float(evaluation["case_iou"]),
    )
    cases = select_cases(
        match,
        max_cases=int(evaluation["max_cases"]),
        explicit_ids=[str(value) for value in (evaluation.get("case_images") or [])],
    )
    write_json_atomic(run.evaluation_dir / "cases.json", cases)
    # Plan 02 §2 names this artefact `case_selection.json`; keep both names so
    # either consumer finds it (same payload, one source of truth).
    write_json_atomic(run.evaluation_dir / "case_selection.json", cases)
    totals = match["totals"]
    log_lines.append(
        f"case analysis (NOT mAP): conf>={match['conf']:g} iou>={match['iou']:g} "
        f"tp={totals['tp']} fp={totals['fp']} fn={totals['fn']} "
        f"precision={totals['precision']} recall={totals['recall']}"
    )

    zero_area = _zero_area_count(predictions)
    if zero_area:
        warnings.append(
            f"{zero_area} exported detection(s) have zero width or height after the "
            f"evaluator clipped them to the image; they cannot match any ground-truth box "
            f"and are kept only because the export mirrors the validator output"
        )
    max_det_hits = _max_det_hit_ids(predictions, int(evaluation["max_det"]))
    if max_det_hits:
        warnings.append(
            f"{len(max_det_hits)}/{len(predictions)} evaluated image(s) reached "
            f"max_det={int(evaluation['max_det'])}; their detections are truncated at that "
            f"limit, which can cap recall on dense scenes. Raise eval.max_det or record why "
            f"this level is acceptable for the formal run."
        )
        log_lines.append(
            f"max_det warning: {len(max_det_hits)} image(s) hit max_det={int(evaluation['max_det'])}"
        )

    # ---- metrics -----------------------------------------------------------
    metrics = ultralytics_adapter.extract_metrics(
        metrics_object,
        num_images=num_images,
        num_instances=num_instances,
        run_id=run.run_id,
        scope=str(config.get("scope", run.scope)),
        split=str(split),
        weights_sha256=weights_sha256,
        dataset_manifest_sha256=_dataset_manifest_sha256(run, data_yaml),
        image_ids_digest=image_ids_digest,
        data_view_root=str(_view_root(data_yaml)),
        imgsz=int(evaluation["imgsz"]),
        conf=float(evaluation["conf"]),
        iou_nms=float(evaluation["iou"]),
        max_det=int(evaluation["max_det"]),
        precision_mode=_precision_mode(details),
        cases={
            "conf": match["conf"],
            "iou": match["iou"],
            "tp": int(totals["tp"]),
            "fp": int(totals["fp"]),
            "fn": int(totals["fn"]),
        },
        speed=getattr(outcome, "speed", None) or {},
        warnings=[],
        evaluator_name=str(getattr(outcome, "evaluator_name", "ultralytics.DetectionValidator")),
        evaluator_version=getattr(outcome, "evaluator_version", None),
    )
    warnings.extend(metrics.get("warnings") or [])
    if metrics["status"] == "partial":
        status = "partial"
    # Completeness against the configured split size (plan 03 B07 / §7: a formal
    # validation must cover exactly the shipped 548 images, not a subset).
    expected_images = _expected_images_for_split(config, split)
    metrics["expected_images"] = expected_images
    metrics["complete"] = bool(expected_images is None or num_images == expected_images)
    if expected_images is None:
        warnings.append(
            f"no configured expected image count for split '{split}'; completeness cannot be "
            f"checked against the audit expectations"
        )
    elif num_images != expected_images:
        warnings.append(
            f"evaluated {num_images} image(s) but the configured expected count for split "
            f"'{split}' is {expected_images}; this validation is incomplete"
        )
        status = "partial"
    metrics["weights_path"] = str(weights)
    metrics["zero_area_detections"] = zero_area
    metrics["max_det_hits"] = len(max_det_hits)
    metrics["max_det_hit_images"] = max_det_hits[:50]
    metrics["warnings"] = warnings
    metrics["status"] = status

    # ---- files -------------------------------------------------------------
    _write_jsonl(run.evaluation_dir / "predictions.jsonl", predictions)
    _write_jsonl(run.evaluation_dir / "ground_truth.jsonl", ground_truth)
    if curves is not None:
        arrays, curves_meta = curves
        np.savez_compressed(run.evaluation_dir / "curves.npz", **arrays)
        write_json_atomic(run.evaluation_dir / "curves_meta.json", curves_meta)
    if confusion is not None:
        matrix, confusion_meta = confusion
        np.savez_compressed(run.evaluation_dir / "confusion_matrix.npz", matrix=matrix)
        write_json_atomic(run.evaluation_dir / "confusion_meta.json", confusion_meta)
    write_json_atomic(run.evaluation_dir / "metrics.json", metrics)

    log_lines.extend(
        [
            f"images={num_images} instances={num_instances}",
            f"native mAP50={metrics['map50']} mAP50-95={metrics['map50_95']} "
            f"P={metrics['precision']} R={metrics['recall']}",
            f"status={status} warnings={len(warnings)}",
        ]
    )
    _write_log(log_lines, run)

    return {
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "num_images": num_images,
        "num_instances": num_instances,
        "expected_images": metrics.get("expected_images"),
        "complete": metrics.get("complete", True),
        "max_det_hits": metrics.get("max_det_hits", 0),
        "status": status,
        "warnings": warnings,
        "weights_sha256": weights_sha256,
        "split": str(split),
        "cases": {"conf": match["conf"], "iou": match["iou"],
                  "tp": int(totals["tp"]), "fp": int(totals["fp"]), "fn": int(totals["fn"])},
    }


def _max_det_hit_ids(predictions: Sequence[Mapping[str, Any]], max_det: int) -> list[str]:
    """Image ids whose detection count reached ``max_det`` (possible truncation).

    The config comment promises this warning: with ``conf=0.001`` on dense
    VisDrone scenes the raw candidate count can exceed ``max_det`` for every
    image, which silently caps recall. The count is surfaced instead of assumed
    to be irrelevant.
    """
    limit = int(max_det)
    if limit <= 0:
        return []
    return [
        str(row.get("image_id", "?"))
        for row in predictions
        if int(row.get("num_detections") or 0) >= limit
    ]


def _zero_area_count(predictions: Sequence[Mapping[str, Any]]) -> int:
    """Detections whose clipped box has no area (cannot match any GT)."""
    count = 0
    for row in predictions:
        for det in row.get("detections") or []:
            x1, y1, x2, y2 = (float(value) for value in det.get("xyxy", [0, 0, 0, 0]))
            if x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
                count += 1
    return count


def _expected_images_for_split(config: Mapping[str, Any], split: str) -> int | None:
    """Configured image count for ``split`` (``None`` for an unmodelled split).

    A ``scope=smoke`` run is *supposed* to validate only the smoke subset, so its
    expectation is the smoke count, not the full formal 548 — otherwise a
    correct smoke run would be flagged incomplete.
    """
    data = dict(config.get("data") or {})
    smoke = str(config.get("scope", "formal")) == "smoke"
    if str(split) == str(data.get("val_split", "val")):
        return int(data["smoke_val_images"] if smoke else data["expected_val_images"])
    if str(split) == str(data.get("train_split", "train")):
        return int(data["smoke_train_images"] if smoke else data["expected_train_images"])
    return None


def _precision_mode(details: Mapping[str, Any]) -> str:
    """What the validator actually ran with (``None``/32 = FP32, 16 = FP16)."""
    quantize = details.get("quantize") if isinstance(details, Mapping) else None
    if quantize == 16:
        return "FP16"
    if quantize == 8:
        return "INT8"
    return "FP32"


def _validate_rows(predictions: Sequence[Mapping[str, Any]],
                   ground_truth: Sequence[Mapping[str, Any]]) -> None:
    """Reject rows that would not be usable evidence (no silent repair)."""
    for row in predictions:
        for key in ("image_id", "source", "width", "height", "detections"):
            if key not in row:
                raise EvaluationError(f"prediction row is missing '{key}': {sorted(row)}")
        width, height = int(row["width"]), int(row["height"])
        if width <= 0 or height <= 0:
            raise EvaluationError(f"prediction row {row['image_id']} has an invalid image size")
        if int(row.get("num_detections", len(row["detections"]))) != len(row["detections"]):
            raise EvaluationError(
                f"prediction row {row['image_id']} has an inconsistent num_detections"
            )
        for det in row["detections"]:
            _require_finite(
                [float(v) for v in det["xyxy"]] + [float(det["confidence"])],
                f"prediction {row['image_id']}",
            )
    for row in ground_truth:
        for key in ("image_id", "source", "width", "height", "objects"):
            if key not in row:
                raise EvaluationError(f"ground-truth row is missing '{key}': {sorted(row)}")
        if int(row.get("num_objects", len(row["objects"]))) != len(row["objects"]):
            raise EvaluationError(
                f"ground-truth row {row['image_id']} has an inconsistent num_objects"
            )
        for obj in row["objects"]:
            _require_finite([float(v) for v in obj["xyxy"]], f"ground truth {row['image_id']}")


def _write_per_class_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """``N/A`` for a class whose metric cannot be computed (never ``0``)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["class_id", "class_name", "num_gt", "precision", "recall", "ap50", "ap50_95"])
    for row in rows:
        values: list[Any] = [int(row["class_id"]), str(row["class_name"])]
        num_gt = row.get("num_gt")
        values.append("N/A" if num_gt is None else int(num_gt))
        for key in ("precision", "recall", "ap50", "ap50_95"):
            value = row.get(key)
            values.append("N/A" if value is None else f"{float(value):.6f}")
        writer.writerow(values)
    return write_text_atomic(path, buffer.getvalue())
