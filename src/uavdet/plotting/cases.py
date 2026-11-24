"""Qualitative detection-case figures (F11/F12).

Both figures read the recorded evaluation products only — never a new inference:

* ``evaluation/predictions.jsonl`` / ``ground_truth.jsonl`` for the boxes,
* ``evaluation/cases.json`` for which images were selected and why,
* ``evaluation/metrics.json`` for ``data_view_root`` (image fallback path).

The image is resolved from the recorded absolute ``source`` when it still
exists, otherwise from ``<data_view_root>/<source_rel>``.

One row per case, four panels:

``(a) crop context`` — the full frame with the crop rectangle,
``(b) original``     — the same crop, no annotations,
``(c) ground truth`` — boxes on the crop,
``(d) prediction``   — detections at the displayed confidence threshold on the crop.

Every panel of a case uses the identical crop rectangle. The crop is a
deterministic "densest window": at most 30 % of each frame dimension, centred
on the location that contains the most GT/prediction boxes, so a VisDrone-scale
object stays a handful of pixels wide at the final figure width. Boxes are drawn
with class colours only — no per-box text, which would turn a 40-90 object scene
into an unreadable mass; the class-to-colour key is a small figure legend.

Photographs stay raster; only boxes, the crop rectangle and text are vector
overlays. If no case satisfies the selection rules, a ``missing`` record
explains it instead of inventing a story, and reasons written by the selector
are marked as *possible* because they are not verified here.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..artifacts import read_json, read_jsonl
from ..types import CLASS_NAMES, RunContext
from .dataset import finish_figure, figure_settings, missing, prepare_render
from .style import CLASS_COLORS, REPORT_COLORS

__all__ = ["render"]

_MUTED = REPORT_COLORS["muted"]
_TEXT = REPORT_COLORS["text"]

#: figure id -> (output basename, title, manuscript section); mirrors
#: ``uavdet.report.FIGURE_META`` without importing it (see plotting.dataset).
FIGURE_META: dict[str, tuple[str, str, str]] = {
    "F11": ("successful_cases", "Successful detections at conf=0.25",
            "Qualitative results"),
    "F12": ("failure_cases", "Missed, false and crowded detections at conf=0.25",
            "Qualitative results"),
}

#: at most this many cases per figure (matches the selector's default cap)
_MAX_CASES = 6
_DEFAULT_CONF = 0.25
#: crop = a window of this fraction of each frame dimension (the densest one)
_CROP_SHARE = 0.30
_MIN_CROP_SHARE = 0.12
_BOX_LINEWIDTH = 0.7
_LEGEND_COLUMNS = 6


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def _case_inputs(run: RunContext) -> dict[str, Any]:
    evaluation = run.evaluation_dir
    predictions_path = evaluation / "predictions.jsonl"
    ground_truth_path = evaluation / "ground_truth.jsonl"
    cases_path = evaluation / "cases.json"
    metrics_path = evaluation / "metrics.json"

    present = [p for p in (predictions_path, ground_truth_path, cases_path) if p.is_file()]
    payload: dict[str, Any] = {
        "paths": {
            "predictions": predictions_path,
            "ground_truth": ground_truth_path,
            "cases": cases_path,
            "metrics": metrics_path,
        },
        "missing": [p for p in (predictions_path, ground_truth_path, cases_path)
                    if not p.is_file()],
        "predictions": {},
        "ground_truth": {},
        "cases": {},
        "metrics": {},
    }
    if not present:
        return payload
    if cases_path.is_file():
        payload["cases"] = read_json(cases_path)
    for row in read_jsonl(predictions_path):
        image_id = str(row.get("image_id") or "")
        if image_id:
            payload["predictions"][image_id] = row
    for row in read_jsonl(ground_truth_path):
        image_id = str(row.get("image_id") or "")
        if image_id:
            payload["ground_truth"][image_id] = row
    if metrics_path.is_file():
        payload["metrics"] = read_json(metrics_path)
    return payload


def _data_view_root(ctx: Mapping[str, Any]) -> Path | None:
    metrics = ctx.get("metrics") or {}
    root = metrics.get("data_view_root")
    if root and Path(str(root)).is_dir():
        return Path(str(root))
    return None


def _resolve_image(entry: Mapping[str, Any], record: Mapping[str, Any] | None,
                   data_view_root: Path | None) -> Path | None:
    """``source`` first (if it still exists), then ``data_view_root/source_rel``."""
    for candidate in ((record or {}).get("source"), entry.get("source")):
        if candidate and Path(str(candidate)).is_file():
            return Path(str(candidate))
    source_rel = (record or {}).get("source_rel") or entry.get("source_rel")
    if source_rel and data_view_root is not None:
        candidate = data_view_root / str(source_rel)
        if candidate.is_file():
            return candidate
    return None


def _display_threshold(ctx: Mapping[str, Any], run: RunContext) -> float:
    metrics_cases = (ctx.get("metrics") or {}).get("cases")
    for value in ((ctx.get("cases") or {}).get("conf"),
                  metrics_cases.get("conf") if isinstance(metrics_cases, Mapping) else None,
                  ((run.config or {}).get("eval") or {}).get("cases_conf")):
        if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
            return float(value)
    return _DEFAULT_CONF


# ---------------------------------------------------------------------------
# geometry / drawing
# ---------------------------------------------------------------------------
def _boxes(objects: Sequence[Mapping[str, Any]]) -> list[tuple[int, str, float, list[float]]]:
    drawn: list[tuple[int, str, float, list[float]]] = []
    for obj in objects or ():
        xyxy = obj.get("xyxy")
        if not xyxy or len(xyxy) != 4:
            continue
        try:
            class_id = int(obj.get("class_id", -1))
            coords = [float(value) for value in xyxy]
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(coords)) or coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        name = str(obj.get("class_name") or
                   (CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"class {class_id}"))
        confidence = float(obj.get("confidence", 0.0) or 0.0)
        drawn.append((class_id, name, confidence, coords))
    return drawn


def _colour(class_id: int) -> str:
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def _focus_window(boxes: Sequence[Sequence[float]], width: int,
                  height: int) -> tuple[float, float, float, float]:
    """Densest crop window: fixed share of the frame, most boxes inside.

    Deterministic: candidates are the box centres plus the frame centre; the
    window with the most box centres inside wins, ties go to the window closest
    to the frame centre and then to the smallest top-left coordinate.
    """
    frame = (0.0, 0.0, float(width), float(height))
    if not boxes:
        return frame
    share = min(max(_CROP_SHARE, _MIN_CROP_SHARE), 1.0)
    window_w = max(float(width) * share, 8.0)
    window_h = max(float(height) * share, 8.0)
    window_w = min(window_w, float(width))
    window_h = min(window_h, float(height))
    centres = np.asarray(
        [[(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0] for box in boxes], dtype=float)
    candidates = np.vstack([centres, [[width / 2.0, height / 2.0]]])
    best: tuple[float, float, float, float] | None = None
    best_key: tuple[float, ...] | None = None
    for centre_x, centre_y in candidates:
        x1 = min(max(centre_x - window_w / 2.0, 0.0), max(float(width) - window_w, 0.0))
        y1 = min(max(centre_y - window_h / 2.0, 0.0), max(float(height) - window_h, 0.0))
        x2, y2 = x1 + window_w, y1 + window_h
        inside = ((centres[:, 0] >= x1) & (centres[:, 0] <= x2)
                  & (centres[:, 1] >= y1) & (centres[:, 1] <= y2))
        key = (
            -float(inside.sum()),
            math.hypot((x1 + x2) / 2.0 - width / 2.0, (y1 + y2) / 2.0 - height / 2.0),
            x1,
            y1,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = (x1, y1, x2, y2)
    return best or frame


def _draw_boxes(ax, boxes: Sequence[tuple[int, str, float, list[float]]]) -> None:
    """Class-coloured boxes, no text: a 40-90 object scene must stay readable."""
    from matplotlib.patches import Rectangle

    for class_id, _name, _confidence, (x1, y1, x2, y2) in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor=_colour(class_id), linewidth=_BOX_LINEWIDTH,
                               alpha=0.95, zorder=3))


def _panel(ax, image: np.ndarray, *, crop: tuple[float, float, float, float],
           boxes: Sequence[tuple[int, str, float, list[float]]] = (),
           title: str | None = None,
           context_box: tuple[float, float, float, float] | None = None) -> None:
    from matplotlib.patches import Rectangle

    height, width = image.shape[:2]
    ax.imshow(image, extent=(0.0, float(width), float(height), 0.0),
              interpolation="antialiased")
    x1, y1, x2, y2 = crop
    ax.set_xlim(x1, x2)
    ax.set_ylim(y2, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        spine.set_linewidth(0.6)
        spine.set_color("#333333")
    ax.set_facecolor(REPORT_COLORS["bg"])
    if context_box is not None:
        cx1, cy1, cx2, cy2 = context_box
        ax.add_patch(Rectangle((cx1, cy1), cx2 - cx1, cy2 - cy1, fill=False,
                               edgecolor="white", linewidth=2.0, zorder=4))
        ax.add_patch(Rectangle((cx1, cy1), cx2 - cx1, cy2 - cy1, fill=False,
                               edgecolor=_TEXT, linewidth=0.8, linestyle="--", zorder=5))
    _draw_boxes(ax, boxes)
    if title:
        ax.set_title(title, loc="left", fontsize=9, color=_TEXT, pad=3)


def _case_note(entry: Mapping[str, Any]) -> str:
    counts = []
    for key in ("tp", "fp", "fn", "num_gt"):
        if isinstance(entry.get(key), (int, float)):
            counts.append(f"{key.upper()}={int(entry[key])}")
    reason = str(entry.get("reason") or "").strip()
    parts = []
    if counts:
        parts.append(", ".join(counts))
    if reason:
        parts.append(f"possible reason: {reason}")
    kinds = entry.get("failure_kinds") or []
    if kinds:
        parts.append("kinds: " + ", ".join(str(kind) for kind in kinds))
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# shared renderer
# ---------------------------------------------------------------------------
def _render_cases(run: RunContext, out_dir: Path, figure_id: str, *,
                  cases_key: str) -> dict[str, Any]:
    meta = FIGURE_META[figure_id]
    ctx = _case_inputs(run)
    paths = ctx["paths"]
    if ctx["missing"]:
        return missing(run, out_dir, figure_id, meta,
                       reason="the evaluation products needed for the case panels are "
                              "missing (predictions, ground truth or cases.json)",
                       required=ctx["missing"])
    cases_path = paths["cases"]
    cases_payload = ctx["cases"]
    entries = [entry for entry in (cases_payload.get(cases_key) or [])
               if isinstance(entry, Mapping) and entry.get("image_id")]
    if not entries:
        # cases.json exists -- the file is not missing, it simply has no qualifying
        # case, so this is not reported as a missing input path.
        return missing(run, out_dir, figure_id, meta,
                       reason=(f"no image meets the '{cases_key}' selection rules of "
                               f"{cases_path.name} ({run.rel(cases_path)}); nothing is "
                               f"fabricated to fill the panel"))

    data_view_root = _data_view_root(ctx)
    threshold = _display_threshold(ctx, run)
    warnings: list[str] = []
    if len(entries) > _MAX_CASES:
        warnings.append(f"{len(entries)} cases available; the first {_MAX_CASES} are shown "
                        f"(the selector's deterministic order is preserved)")
        entries = entries[:_MAX_CASES]

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    prepared: list[dict[str, Any]] = []
    for entry in entries:
        image_id = str(entry["image_id"])
        prediction = ctx["predictions"].get(image_id)
        ground_truth = ctx["ground_truth"].get(image_id)
        if prediction is None or ground_truth is None:
            warnings.append(f"case {image_id} has no recorded predictions/ground truth; skipped")
            continue
        image_path = _resolve_image(entry, prediction, data_view_root)
        if image_path is None:
            warnings.append(f"case {image_id}: the recorded image file could not be resolved")
            continue
        try:
            from PIL import Image

            with Image.open(image_path) as handle:
                image = np.asarray(handle.convert("RGB"))
        except (OSError, ValueError) as exc:
            warnings.append(f"case {image_id}: image could not be decoded ({exc})")
            continue
        height, width = image.shape[:2]
        recorded = (prediction.get("width"), prediction.get("height"))
        if recorded[0] and recorded[1] and (int(recorded[0]), int(recorded[1])) != (width, height):
            warnings.append(
                f"case {image_id}: image file is {width}x{height} but the recorded boxes "
                f"assume {recorded[0]}x{recorded[1]}; box positions may be offset"
            )
        gt_boxes = _boxes(ground_truth.get("objects") or [])
        det_boxes = [box for box in _boxes(prediction.get("detections") or [])
                     if box[2] >= threshold]
        prepared.append({
            "image_id": image_id,
            "entry": entry,
            "image": image,
            "image_path": image_path,
            "gt_boxes": gt_boxes,
            "det_boxes": det_boxes,
            "crop": _focus_window([box[3] for box in gt_boxes + det_boxes], width, height),
            "width": width,
            "height": height,
            "note": _case_note(entry),
        })
    if not prepared:
        return missing(run, out_dir, figure_id, meta,
                       reason="none of the selected cases could be rendered (unresolvable or "
                              "undecodable image files); no panel is invented")

    settings = figure_settings(run)
    width_in = settings["width"]
    column_titles = ("Crop context", "Original (crop)", "Ground truth", "Prediction")
    ncols = len(column_titles)

    # Manual layout: every panel keeps the frame's aspect ratio (the crop uses
    # the same share of both dimensions), so the raster is never stretched.
    margin_left, margin_right, gap = 0.05, 0.03, 0.07
    panel_width = (width_in - margin_left - margin_right - gap * (ncols - 1)) / ncols
    note_width_chars = max(40, int((width_in - margin_left - margin_right) * 17))
    footer_text = (
        f"Class colours identify classes; per-box text is omitted so crowded scenes stay "
        f"readable. Only detections with conf >= {threshold:g} are drawn"
        + (f" (matching IoU {float(cases_payload['iou']):g})."
           if isinstance(cases_payload.get("iou"), (int, float)) else ".")
        + " Each case uses one identical crop rectangle for panels (b)-(d); reason text "
          "is the selector's explanation, marked 'possible' because it is not verified here."
    )
    footer_text = textwrap.fill(footer_text, width=max(50, int(width_in * 17)))
    footer_lines = footer_text.count("\n") + 1

    present_classes = sorted({box[0] for case in prepared
                              for box in case["gt_boxes"] + case["det_boxes"]})
    legend_rows = max(1, math.ceil(len(present_classes) / _LEGEND_COLUMNS)) if present_classes else 0

    rows: list[dict[str, Any]] = []
    for case in prepared:
        aspect = case["height"] / float(case["width"])
        row_height = panel_width * aspect
        caption = textwrap.fill(
            f"case {len(rows) + 1}: {case['image_id']}"
            + (f" — {case['note']}" if case["note"] else ""),
            width=note_width_chars,
        )
        # reserve exactly the height the wrapped caption needs (never overlaps)
        note_height = 0.145 * (caption.count("\n") + 1) + 0.10
        rows.append({"case": case, "height": row_height, "note": caption,
                     "note_height": note_height})

    header_height = 0.26
    legend_height = 0.16 * legend_rows + 0.05 if legend_rows else 0.0
    footer_height = 0.145 * footer_lines + 0.10
    total_height = (header_height + footer_height + legend_height
                    + sum(row["height"] + row["note_height"] for row in rows))
    fig = plt.figure(figsize=(width_in, total_height))

    y_cursor = header_height
    for index, row in enumerate(rows):
        case = row["case"]
        crop = case["crop"]
        row_bottom = y_cursor + row["height"]
        for column in range(ncols):
            left = margin_left + column * (panel_width + gap)
            box = [left / width_in,
                   1.0 - (row_bottom / total_height),
                   panel_width / width_in,
                   row["height"] / total_height]
            ax = fig.add_axes(box)
            title = f"({chr(97 + column)}) {column_titles[column]}" if index == 0 else None
            if column == 0:
                _panel(ax, case["image"], crop=(0.0, 0.0, float(case["width"]),
                                                float(case["height"])),
                       title=title, context_box=crop)
            elif column == 1:
                _panel(ax, case["image"], crop=crop, title=title)
            elif column == 2:
                _panel(ax, case["image"], crop=crop, title=title, boxes=case["gt_boxes"])
            else:
                _panel(ax, case["image"], crop=crop, title=title, boxes=case["det_boxes"])
        fig.text(margin_left / width_in,
                 1.0 - (row_bottom + 0.04) / total_height,
                 row["note"], fontsize=8, color=_MUTED, va="top", ha="left")
        y_cursor = row_bottom + row["note_height"]

    if present_classes:
        handles = [Patch(facecolor=_colour(class_id), edgecolor="none",
                         label=CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES)
                         else f"class {class_id}")
                   for class_id in present_classes]
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, footer_height / total_height),
                   ncol=min(_LEGEND_COLUMNS, max(1, len(handles))), fontsize=8,
                   frameon=False, handlelength=1.1, handleheight=1.1,
                   labelspacing=0.3, columnspacing=1.4)
    fig.text(0.5, 0.006, footer_text, fontsize=8, color=_MUTED, ha="center", va="bottom")

    inputs = [paths["cases"], paths["predictions"], paths["ground_truth"]]
    inputs += [case["image_path"] for case in prepared]
    if paths["metrics"].is_file():
        inputs.append(paths["metrics"])
    if cases_key == "failure":
        notes = (
            f"{len(prepared)} cases selected by the evaluation's failure rules at conf="
            f"{threshold:g}: a full-frame crop context plus identical cropped views of the "
            f"original image, the ground truth and the displayed predictions. Reasons are "
            f"the selector's possible explanations, not verified causes."
        )
    else:
        notes = (
            f"{len(prepared)} cases selected by the evaluation's success rules at conf="
            f"{threshold:g}: a full-frame crop context plus identical cropped views of the "
            f"original image, the ground truth and the displayed predictions. Detection "
            f"quality is judged from the panels, not from the number of boxes."
        )
    return finish_figure(
        fig, run, out_dir, figure_id, meta,
        input_paths=inputs,
        source_data={
            "arrays": {"case_index": np.arange(len(prepared), dtype=np.int64)},
            "columns": {
                "case_index": list(range(1, len(prepared) + 1)),
                "image_id": [case["image_id"] for case in prepared],
                "image_path": [case["image_path"].name for case in prepared],
                "width": [case["width"] for case in prepared],
                "height": [case["height"] for case in prepared],
                "crop_x1": [round(case["crop"][0], 3) for case in prepared],
                "crop_y1": [round(case["crop"][1], 3) for case in prepared],
                "crop_x2": [round(case["crop"][2], 3) for case in prepared],
                "crop_y2": [round(case["crop"][3], 3) for case in prepared],
                "num_gt": [len(case["gt_boxes"]) for case in prepared],
                "num_displayed_detections": [len(case["det_boxes"]) for case in prepared],
                "reason": [str(case["entry"].get("reason") or "") for case in prepared],
            },
            "meta": {
                "units": {"xyxy": "original-image pixels (x1, y1, x2, y2)",
                          "confidence": "fraction 0-1"},
                "display_threshold": threshold,
                "matching_iou": cases_payload.get("iou"),
                "crop_rule": (f"densest window of {_CROP_SHARE:.0%} of each frame dimension "
                              f"(candidates: GT and displayed detection centres plus the frame "
                              f"centre), clipped to the frame; identical for panels (b)-(d)"),
                "labels": "class-colour boxes only; no per-box text so crowded scenes stay "
                          "readable",
                "class_colors": list(CLASS_COLORS),
                "classes_in_figure": present_classes,
                "reason_status": "selector explanation, marked 'possible'; not verified here",
                "raster_note": "photo panels are embedded raster images; only boxes, the crop "
                               "rectangle and text are vector",
                "selection_rules": cases_payload.get("selection_rules"),
            },
        },
        variables=["image_id", "crop_rect", "gt_boxes", "detections", "confidence"],
        units={"xyxy": "original-image pixels", "confidence": "fraction 0-1"},
        filters=[f"detections drawn only at conf >= {threshold:g}",
                 f"cases come from evaluation/cases.json '{cases_key}' list"],
        transforms=[f"crop = densest window of {_CROP_SHARE:.0%} of each frame dimension, "
                    f"identical for every panel of a case",
                    "class-coloured boxes without per-box labels"],
        normalization=["none (original pixel coordinates)"],
        sample_count={"cases_rendered": len(prepared), "cases_listed": len(entries)},
        notes=notes,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render(run: RunContext, out_dir: str | Path) -> list[dict]:
    """Render F11 and F12 into ``out_dir`` and return one record per figure."""
    _, target = prepare_render(run, out_dir)
    return [
        _render_cases(run, target, "F11", cases_key="successful"),
        _render_cases(run, target, "F12", cases_key="failure"),
    ]
