"""Evaluation report figures (F07-F10).

Inputs are the saved evaluation products of *this* run:

* ``evaluation/per_class.csv`` — per-class AP (F07).
* ``evaluation/confusion_matrix.npz`` + ``confusion_meta.json`` — F08.
* ``evaluation/curves.npz`` + ``curves_meta.json`` — PR and F1-confidence (F09/F10).

Nothing is inferred: AP values are never reconstructed from the confusion matrix,
and PR/F1 curves are only ever the stored native arrays. Classes without ground
truth are drawn as an explicit ``N/A`` marker, never as a zero bar. Percent
conversion happens once, at draw time, with
:func:`uavdet.plotting.style.to_percent`; ``source_data`` keeps the 0-1 values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..artifacts import read_json
from ..types import CLASS_NAMES, NUM_CLASSES, RunContext
from .dataset import finish_figure, figure_settings, missing, prepare_render
from .style import (
    CLASS_COLORS,
    LINE_STYLES,
    METRIC_COLORS,
    REPORT_COLORS,
    add_panel_label,
    style_axes,
    to_percent,
)

__all__ = ["render"]

_MUTED = REPORT_COLORS["muted"]
_TEXT = REPORT_COLORS["text"]

#: figure id -> (output basename, title, manuscript section); mirrors
#: ``uavdet.report.FIGURE_META`` without importing it (see plotting.dataset).
FIGURE_META: dict[str, tuple[str, str, str]] = {
    "F07": ("per_class_ap", "Per-class AP@0.5 and AP@0.5:0.95", "Results"),
    "F08": ("confusion_matrix", "Confusion matrix (counts and GT-normalised)", "Results"),
    "F09": ("pr_curve", "Precision-recall curves", "Results"),
    "F10": ("f1_confidence", "F1 versus confidence", "Results"),
}

_PER_CLASS_FILE = "per_class.csv"
_CURVES_FILE = "curves.npz"
_CURVES_META_FILE = "curves_meta.json"
_CONFUSION_FILE = "confusion_matrix.npz"
_CONFUSION_META_FILE = "confusion_meta.json"

_NA_TOKENS = {"", "n/a", "na", "nan", "none", "null", "-"}


# ---------------------------------------------------------------------------
# small parsing helpers
# ---------------------------------------------------------------------------
def _optional_float(text: Any) -> float | None:
    """Parse a CSV cell that may legitimately be ``N/A`` (returns ``None``)."""
    if text is None:
        return None
    value = str(text).strip()
    if value.lower() in _NA_TOKENS:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if np.isfinite(parsed) else None


def _numerical_column(frame: Any, column: str) -> np.ndarray:
    return np.asarray([_optional_float(v) for v in frame[column].tolist()], dtype=float)


# ---------------------------------------------------------------------------
# F07 — per-class AP
# ---------------------------------------------------------------------------
def _per_class_ap(run: RunContext, out_dir: Path) -> dict[str, Any]:
    figure_id = "F07"
    path = run.evaluation_dir / _PER_CLASS_FILE
    if not path.is_file():
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason="no evaluation/per_class.csv, so per-class AP is not available "
                              "(AP is never inferred from other products)",
                       required=[path])
    import pandas as pd

    try:
        frame = pd.read_csv(path, keep_default_na=False, dtype=str)
        frame.columns = [str(column).strip() for column in frame.columns]
    except (OSError, ValueError) as exc:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"evaluation/per_class.csv is unreadable: {exc}", required=[path])
    needed = ["class_id", "class_name", "num_gt", "ap50", "ap50_95"]
    absent = [column for column in needed if column not in frame.columns]
    if absent:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"per_class.csv lacks the column(s): {', '.join(absent)}",
                       required=[path])

    rows: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []
    for _index, row in frame.iterrows():
        raw_id = _optional_float(row["class_id"])
        if raw_id is None or int(raw_id) != raw_id:
            warnings.append(f"row with an unparsable class_id skipped: {dict(row)}")
            continue
        class_id = int(raw_id)
        if not 0 <= class_id < NUM_CLASSES:
            warnings.append(f"class_id {class_id} outside the fixed 0-9 order skipped")
            continue
        ap50 = _optional_float(row["ap50"])
        ap50_95 = _optional_float(row["ap50_95"])
        num_gt = _optional_float(row["num_gt"])
        name = str(row["class_name"]).strip() or CLASS_NAMES[class_id]
        valid = ap50 is not None and ap50_95 is not None
        if num_gt is not None and int(num_gt) == 0 and valid:
            # a class with no ground truth has no computable AP: never plot a bar
            warnings.append(
                f"class {class_id} ({name}) has num_gt=0 but numeric AP values; shown as N/A")
            valid = False
        rows[class_id] = {
            "class_name": name,
            "num_gt": int(num_gt) if num_gt is not None else 0,
            "ap50": ap50,
            "ap50_95": ap50_95,
            "valid": valid,
        }
    if not rows:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason="per_class.csv has no usable class rows", required=[path])
    for class_id in range(NUM_CLASSES):
        if class_id not in rows:
            rows[class_id] = {"class_name": CLASS_NAMES[class_id], "num_gt": 0,
                              "ap50": None, "ap50_95": None, "valid": False}
            warnings.append(f"class {class_id} is absent from per_class.csv; shown as N/A")

    class_ids = np.arange(NUM_CLASSES, dtype=np.int64)
    names = [rows[class_id]["class_name"] for class_id in class_ids]
    valid = np.asarray([bool(rows[class_id]["valid"]) for class_id in class_ids], dtype=bool)
    num_gt = np.asarray([int(rows[class_id]["num_gt"]) for class_id in class_ids], dtype=np.int64)
    ap50 = np.asarray([rows[class_id]["ap50"] if valid[i] else 0.0
                       for i, class_id in enumerate(class_ids)], dtype=float)
    ap50_95 = np.asarray([rows[class_id]["ap50_95"] if valid[i] else 0.0
                          for i, class_id in enumerate(class_ids)], dtype=float)

    settings = figure_settings(run)
    import matplotlib.pyplot as plt

    width = settings["width"]
    fig, ax = plt.subplots(figsize=(width, width * 0.60), layout="constrained")
    positions = np.arange(NUM_CLASSES, dtype=float)
    height = 0.38
    for index, class_id in enumerate(class_ids):
        if not valid[index]:
            ax.text(1.0, positions[index], "N/A", fontsize=8, color=_MUTED,
                    va="center", ha="left")
            continue
        ax.barh(positions[index] + height / 2, to_percent(ap50[index]), height=height,
                color=METRIC_COLORS["map50"])
        ax.barh(positions[index] - height / 2, to_percent(ap50_95[index]), height=height,
                color=METRIC_COLORS["map50_95"])
        if ap50[index] == 0.0 and ap50_95[index] == 0.0:
            # a real 0 % AP is data, not a missing value: mark it instead of hiding it
            ax.text(1.0, positions[index], "0", fontsize=8, color=_MUTED,
                    va="center", ha="left")
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{name}" for name in names])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 100.0)
    style_axes(ax, xlabel="Average precision (%)",
               title="Per-class AP@0.5 and AP@0.5:0.95")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=METRIC_COLORS["map50"], label="AP@0.5"),
        plt.Rectangle((0, 0), 1, 1, color=METRIC_COLORS["map50_95"], label="AP@0.5:0.95"),
        plt.Line2D([], [], linestyle="none", marker="$N/A$", color=_MUTED,
                   markersize=8, label="no ground truth (N/A)"),
        plt.Line2D([], [], linestyle="none", marker="$0$", color=_MUTED,
                   markersize=8, label="AP is 0 %"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right",
              handlelength=1.8, labelspacing=0.4)
    fig.supxlabel(
        "Classes without ground truth are marked N/A and are not drawn as zero bars; "
        "AP values are stored 0-1 and shown as percent.",
        fontsize=8, color=_MUTED,
    )

    na_classes = [names[i] for i in range(NUM_CLASSES) if not valid[i]]
    notes = (
        f"Per-class AP@0.5 and AP@0.5:0.95 for the {int(valid.sum())} classes that have "
        f"ground truth in this validation set; classes whose AP is exactly 0 % are marked 0"
        + (f"; {', '.join(na_classes)} has/have no GT and is marked N/A." if na_classes else ".")
    )
    return finish_figure(
        fig, run, out_dir, figure_id, FIGURE_META[figure_id],
        input_paths=[path],
        source_data={
            "arrays": {
                "class_id": class_ids,
                "num_gt": num_gt,
                "ap50_0to1": ap50,
                "ap50_95_0to1": ap50_95,
                "valid_mask": valid,
                "ap50_percent": to_percent(ap50),
                "ap50_95_percent": to_percent(ap50_95),
            },
            "columns": {
                "class_id": class_ids.tolist(),
                "class_name": names,
                "num_gt": num_gt.tolist(),
                "ap50_0to1": [("N/A" if not valid[i] else float(ap50[i]))
                              for i in range(NUM_CLASSES)],
                "ap50_95_0to1": [("N/A" if not valid[i] else float(ap50_95[i]))
                                 for i in range(NUM_CLASSES)],
                "valid": [bool(value) for value in valid],
            },
            "meta": {
                "units": {"ap": "fraction 0-1 (displayed percent)", "num_gt": "count"},
                "na_handling": "classes without GT are masked in valid_mask and written as "
                               "'N/A' in the CSV; the numeric arrays are 0-filled placeholders "
                               "and must always be read together with valid_mask",
                "na_classes": na_classes,
                "class_order": names,
                "display_transform": "percent = stored_fraction * 100, applied once",
            },
        },
        variables=["class_id", "class_name", "num_gt", "ap50", "ap50_95"],
        units={"ap50": "fraction 0-1 (percent on the axis)", "ap50_95": "fraction 0-1",
               "num_gt": "count"},
        filters=["class_id ordered 0-9 (fixed VisDrone DET order)",
                 "classes without GT kept as N/A rows, not dropped and not zeroed"],
        transforms=["percent = fraction * 100 (once, display only)"],
        normalization=["none (AP values as stored by the evaluator)"],
        sample_count={
            "classes": NUM_CLASSES,
            "classes_with_gt": int(valid.sum()),
            "classes_without_gt": int(NUM_CLASSES - valid.sum()),
        },
        notes=notes,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# F08 — confusion matrix
# ---------------------------------------------------------------------------
def _confusion_labels(meta: Mapping[str, Any], size: int) -> tuple[list[str], int | None]:
    order = [str(name) for name in (meta.get("class_order") or CLASS_NAMES)]
    background = meta.get("background_index")
    background_index = int(background) if isinstance(background, (int, float)) else None
    labels: list[str] = []
    for index in range(size):
        if background_index is not None and index == background_index:
            labels.append("background")
        elif index < len(order):
            labels.append(order[index])
        else:  # defensive: never invent a class name
            labels.append(f"class {index}")
    return labels, background_index


def _confusion_matrix(run: RunContext, out_dir: Path) -> dict[str, Any]:
    figure_id = "F08"
    matrix_path = run.evaluation_dir / _CONFUSION_FILE
    meta_path = run.evaluation_dir / _CONFUSION_META_FILE
    absent = [path for path in (matrix_path, meta_path) if not path.is_file()]
    if absent:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason="the evaluator did not save the raw confusion matrix, so no "
                              "confusion figure can be drawn",
                       required=absent)
    try:
        with np.load(matrix_path) as loaded:
            matrix = np.asarray(loaded["matrix"])
        meta = read_json(meta_path)
    except (OSError, KeyError, ValueError) as exc:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"the saved confusion matrix is unusable: {exc}",
                       required=[matrix_path, meta_path])
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"confusion matrix has shape {matrix.shape}, expected a square "
                              f"(num_classes+1) matrix",
                       required=[matrix_path, meta_path])
    if matrix.shape[0] != NUM_CLASSES + 1:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"confusion matrix is {matrix.shape[0]}x{matrix.shape[1]}, "
                              f"expected {NUM_CLASSES + 1}x{NUM_CLASSES + 1}",
                       required=[matrix_path, meta_path])
    if str(meta.get("true_axis", "row")) != "row" or str(meta.get("pred_axis", "column")) != "column":
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason="confusion_meta.json declares an axis convention other than "
                              "row=ground truth / column=prediction, which this renderer "
                              "does not guess about",
                       required=[matrix_path, meta_path])

    counts = matrix.astype(np.int64)
    labels, background_index = _confusion_labels(meta, counts.shape[0])
    row_sums = counts.sum(axis=1)
    valid_rows = row_sums > 0
    normalised = np.zeros(counts.shape, dtype=float)
    # row-wise: every ground-truth row is divided by its own total
    normalised[valid_rows] = counts[valid_rows] / row_sums[valid_rows][:, None]
    masked = np.ma.masked_where(~np.broadcast_to(valid_rows[:, None], counts.shape), normalised)

    warnings: list[str] = []
    if not bool(valid_rows.all()):
        empty_rows = [labels[i] for i in range(len(labels)) if not valid_rows[i]]
        warnings.append(f"row(s) with no ground truth are N/A in the normalised panel: "
                        f"{', '.join(empty_rows)}")

    settings = figure_settings(run)
    import matplotlib.pyplot as plt

    width = settings["width"]
    fig, axes = plt.subplots(2, 1, figsize=(width, width * 1.05), layout="constrained")
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="#E6E6E6")

    raw_ax, norm_ax = axes
    image_raw = raw_ax.imshow(counts, cmap=cmap, aspect="auto")
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            value = int(counts[row, column])
            if value == 0:
                continue
            dark = value > counts.max() * 0.55
            raw_ax.text(column, row, f"{value}", ha="center", va="center", fontsize=8,
                        color="white" if dark else _TEXT)
    style_axes(raw_ax, xlabel="Predicted class", ylabel="Ground-truth class",
               title="Raw counts")
    raw_ax.set_xticks(np.arange(counts.shape[1]))
    raw_ax.set_yticks(np.arange(counts.shape[0]))
    raw_ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    raw_ax.set_yticklabels(labels)
    add_panel_label(raw_ax, "a", x=-0.22, y=1.02)
    fig.colorbar(image_raw, ax=raw_ax, fraction=0.035, pad=0.015, shrink=0.9,
                 label="matched pairs (count)")

    image_norm = norm_ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    for row in range(counts.shape[0]):
        if not valid_rows[row]:
            continue
        for column in range(counts.shape[1]):
            share = normalised[row, column]
            if share <= 0.0:
                continue
            norm_ax.text(column, row, f"{share * 100.0:.1f}", ha="center", va="center",
                         fontsize=8, color="white" if share > 0.55 else _TEXT)
    style_axes(norm_ax, xlabel="Predicted class", ylabel="Ground-truth class",
               title="GT-normalised row shares (%)")
    norm_ax.set_xticks(np.arange(counts.shape[1]))
    norm_ax.set_yticks(np.arange(counts.shape[0]))
    norm_ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    norm_ax.set_yticklabels(labels)
    add_panel_label(norm_ax, "b", x=-0.22, y=1.02)
    from matplotlib.ticker import PercentFormatter

    fig.colorbar(image_norm, ax=norm_ax, fraction=0.035, pad=0.015, shrink=0.9,
                 label="share of ground-truth row (%)",
                 format=PercentFormatter(xmax=1.0, decimals=0))

    conf = meta.get("conf")
    iou = meta.get("iou")
    fig.supxlabel(
        f"Confusion matrix from confusion_meta.json: conf={conf}, IoU={iou}. Rows are ground "
        f"truth, columns are predictions; 'background' is the evaluator's extra row/column, "
        f"not an 11th trained class.",
        fontsize=8, color=_MUTED,
    )

    notes = (
        f"Confusion matrix at conf={conf} and IoU={iou}: panel (a) shows raw matched-pair "
        f"counts, panel (b) divides each ground-truth row by its own total. The 'background' "
        f"row and column are the evaluator's background handling, not an extra trained class."
    )
    return finish_figure(
        fig, run, out_dir, figure_id, FIGURE_META[figure_id],
        input_paths=[matrix_path, meta_path],
        source_data={
            "arrays": {
                "counts": counts,
                "row_sums": row_sums,
                "normalised": normalised,
                "row_valid_mask": valid_rows,
            },
            "columns": {
                "label": labels,
                "row_index": list(range(counts.shape[0])),
            },
            "meta": {
                "true_axis": "row",
                "pred_axis": "column",
                "background_index": background_index,
                "labels": labels,
                "conf": conf,
                "iou": iou,
                "normalization": "panel (a): raw integer counts; panel (b): cell / row sum "
                                 "(ground-truth-normalised), rows without GT are N/A",
                "units": {"counts": "matched pairs (count)",
                          "normalised": "share of ground-truth row (0-1)"},
                "source_version": meta.get("source_version"),
            },
        },
        variables=["counts", "row_sums", "normalised", "background_index"],
        units={"counts": "matched pairs (count)", "normalised": "share of GT row (0-1)"},
        filters=["axes as declared by confusion_meta.json (true=row, pred=column)"],
        transforms=["row share = cell / ground-truth row sum; rows with no GT are masked"],
        normalization=["panel (a) raw counts; panel (b) GT-normalised (0-1)"],
        sample_count={
            "classes": NUM_CLASSES,
            "rows": int(counts.shape[0]),
            "total_pairs": int(counts.sum()),
            "rows_with_gt": int(valid_rows.sum()),
            "rows_without_gt": int((~valid_rows).sum()),
        },
        notes=notes,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# curves.npz
# ---------------------------------------------------------------------------
def _load_curves(run: RunContext) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str]]:
    arrays_path = run.evaluation_dir / _CURVES_FILE
    meta_path = run.evaluation_dir / _CURVES_META_FILE
    absent = [path for path in (arrays_path, meta_path) if not path.is_file()]
    if absent:
        raise FileNotFoundError(", ".join(str(path) for path in absent))
    with np.load(arrays_path) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    meta = read_json(meta_path)
    return arrays, meta, [arrays_path, meta_path]


def _valid_mask(arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> np.ndarray:
    mask = arrays.get("valid_classes")
    if mask is None:
        raw = meta.get("valid_classes")
        mask = np.asarray(raw if raw is not None else [True] * NUM_CLASSES, dtype=bool)
    return np.asarray(mask, dtype=bool).reshape(-1)


def _mean_row(meta: Mapping[str, Any], rows: int) -> int:
    index = meta.get("mean_row_index", NUM_CLASSES)
    index = int(index) if isinstance(index, (int, float)) else NUM_CLASSES
    if not 0 <= index < rows:
        raise ValueError(f"mean_row_index {index} is outside the array with {rows} rows")
    return index


def _class_name(meta: Mapping[str, Any], class_id: int) -> str:
    order = [str(name) for name in (meta.get("class_order") or CLASS_NAMES)]
    return order[class_id] if 0 <= class_id < len(order) else CLASS_NAMES[class_id]



# ---------------------------------------------------------------------------
# curve panel layout (legend outside the axes, never over the curves)
# ---------------------------------------------------------------------------
def _curve_axes(width_in: float, legend_entries: int) -> tuple[Any, Any, float]:
    """Figure + axes with room below for the x label and an out-of-axes legend.

    Returns ``(fig, ax, legend_anchor)`` where ``legend_anchor`` is the figure
    fraction the legend's *top* edge hangs from (above the caption, below the
    x label), so a ten-class legend never covers a curve or an axis label.
    """
    import matplotlib.pyplot as plt

    legend_rows = max(1, (legend_entries + 3) // 4)
    legend_height = 0.155 * legend_rows + 0.05
    caption_height = 0.36
    xlabel_height = 0.42
    title_height = 0.30
    axes_height = width_in * 0.70
    total = title_height + axes_height + xlabel_height + legend_height + caption_height
    fig = plt.figure(figsize=(width_in, total))
    ax = fig.add_axes([0.085, (legend_height + xlabel_height + caption_height) / total,
                       (width_in - 0.16) / width_in, axes_height / total])
    return fig, ax, (caption_height + legend_height) / total


def _outside_legend(fig, ax, anchor: float, *, columns: int = 4) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, anchor),
               ncol=columns, fontsize=8, frameon=False, handlelength=1.8,
               labelspacing=0.25, columnspacing=1.4)


def _figure_caption(fig, text: str, width_in: float) -> None:
    import textwrap

    fig.text(0.5, 0.006, textwrap.fill(text, width=max(50, int(width_in * 17))),
             fontsize=8, color=_MUTED, ha="center", va="bottom")


# ---------------------------------------------------------------------------
# F09 — precision-recall curves
# ---------------------------------------------------------------------------
def _pr_curve(run: RunContext, out_dir: Path) -> dict[str, Any]:
    figure_id = "F09"
    try:
        arrays, meta, inputs = _load_curves(run)
    except FileNotFoundError:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason="the evaluator saved no native PR curves for this run, so the "
                              "curve cannot be drawn (it is never interpolated from summary "
                              "P/R numbers)",
                       required=[run.evaluation_dir / _CURVES_FILE,
                                 run.evaluation_dir / _CURVES_META_FILE])
    if "recall" not in arrays or "pr_precision" not in arrays:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason="curves.npz does not contain the 'recall' / 'pr_precision' arrays",
                       required=inputs)
    recall = np.asarray(arrays["recall"], dtype=float).reshape(-1)
    pr = np.asarray(arrays["pr_precision"], dtype=float)
    if pr.ndim != 2 or pr.shape[1] != recall.size:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"pr_precision has shape {pr.shape}, which does not match the "
                              f"{recall.size}-point recall axis",
                       required=inputs)
    try:
        mean_index = _mean_row(meta, pr.shape[0])
    except ValueError as exc:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"curves_meta.json is inconsistent: {exc}", required=inputs)

    mask = _valid_mask(arrays, meta)
    class_rows = min(mask.size, NUM_CLASSES, pr.shape[0] - 1)
    settings = figure_settings(run)

    width = settings["width"]
    visible = [index for index in range(class_rows) if bool(mask[index])]
    fig, ax, legend_anchor = _curve_axes(width, len(visible) + 1)
    drawn: list[int] = []
    for class_id in range(class_rows):
        if not bool(mask[class_id]):
            continue
        ax.plot(recall, pr[class_id], color=CLASS_COLORS[class_id % len(CLASS_COLORS)],
                linestyle=LINE_STYLES[class_id % len(LINE_STYLES)], linewidth=0.9,
                alpha=0.9, label=_class_name(meta, class_id))
        drawn.append(class_id)
    ax.plot(recall, pr[mean_index], color=_TEXT, linewidth=2.4, linestyle="-",
            label=f"mean of {len(drawn)} classes")
    style_axes(ax, xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    _outside_legend(fig, ax, legend_anchor)
    _figure_caption(
        fig,
        "Thin lines are the per-class native curves, the thick line is the stored mean row "
        f"(row {mean_index} of curves.npz); no smoothing or interpolation was applied.",
        width,
    )

    na_classes = [_class_name(meta, i) for i in range(class_rows) if not bool(mask[i])]
    notes = (
        f"Precision-recall curves from the evaluator's stored arrays: {len(drawn)} per-class "
        f"curves plus the mean row. The mean is the evaluator's own macro mean over classes "
        f"with ground truth."
        + (f" Classes without GT ({', '.join(na_classes)}) are omitted." if na_classes else "")
    )
    return finish_figure(
        fig, run, out_dir, figure_id, FIGURE_META[figure_id],
        input_paths=inputs,
        source_data={
            "arrays": {
                "recall": recall,
                "per_class_precision": pr[:class_rows],
                "mean_precision": pr[mean_index],
                "valid_mask": mask[:class_rows],
            },
            "columns": {
                "recall": recall.tolist(),
                "mean_precision": pr[mean_index].tolist(),
            },
            "meta": {
                "units": {"recall": "fraction 0-1", "precision": "fraction 0-1"},
                "mean_row_index": mean_index,
                "mean_definition": "stored mean row from curves.npz (evaluator macro mean)",
                "class_order": [_class_name(meta, index) for index in range(class_rows)],
                "drawn_classes": drawn,
                "classes_without_gt": na_classes,
                "smoothing": meta.get("smoothing"),
                "source": meta.get("source"),
                "transform": "none (native curves)",
            },
        },
        variables=["recall", "precision", "valid_classes"],
        units={"recall": "fraction 0-1", "precision": "fraction 0-1"},
        filters=["only classes flagged in valid_classes (they have GT) are drawn"],
        transforms=["none"],
        normalization=["none (native per-class and mean curves)"],
        sample_count={
            "classes_with_gt": int(mask[:class_rows].sum()),
            "classes_without_gt": int(class_rows - mask[:class_rows].sum()),
            "curve_points": int(recall.size),
        },
        notes=notes,
        warnings=[],
    )


# ---------------------------------------------------------------------------
# F10 — F1 versus confidence
# ---------------------------------------------------------------------------
def _f1_confidence(run: RunContext, out_dir: Path) -> dict[str, Any]:
    figure_id = "F10"
    try:
        arrays, meta, inputs = _load_curves(run)
    except FileNotFoundError:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason="the evaluator saved no native confidence curves for this run, so "
                              "F1 versus confidence cannot be drawn",
                       required=[run.evaluation_dir / _CURVES_FILE,
                                 run.evaluation_dir / _CURVES_META_FILE])
    if "conf" not in arrays or "f1" not in arrays:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason="curves.npz does not contain the 'conf' / 'f1' arrays",
                       required=inputs)
    conf = np.asarray(arrays["conf"], dtype=float).reshape(-1)
    f1 = np.asarray(arrays["f1"], dtype=float)
    if f1.ndim != 2 or f1.shape[1] != conf.size:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"f1 has shape {f1.shape}, which does not match the "
                              f"{conf.size}-point confidence axis",
                       required=inputs)
    try:
        mean_index = _mean_row(meta, f1.shape[0])
    except ValueError as exc:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"curves_meta.json is inconsistent: {exc}", required=inputs)

    mask = _valid_mask(arrays, meta)
    class_rows = min(mask.size, NUM_CLASSES, f1.shape[0] - 1)
    settings = figure_settings(run)

    width = settings["width"]
    visible = [index for index in range(class_rows) if bool(mask[index])]
    fig, ax, legend_anchor = _curve_axes(width, len(visible) + 1)
    drawn: list[int] = []
    for class_id in range(class_rows):
        if not bool(mask[class_id]):
            continue
        ax.plot(conf, f1[class_id], color=CLASS_COLORS[class_id % len(CLASS_COLORS)],
                linestyle=LINE_STYLES[class_id % len(LINE_STYLES)], linewidth=0.9,
                alpha=0.9, label=_class_name(meta, class_id))
        drawn.append(class_id)
    mean_curve = f1[mean_index]
    ax.plot(conf, mean_curve, color=_TEXT, linewidth=2.4, linestyle="-",
            label=f"mean of {len(drawn)} classes")
    peak_index = int(np.argmax(mean_curve)) if mean_curve.size else 0
    if mean_curve.size:
        peak_conf = float(conf[peak_index])
        peak_f1 = float(mean_curve[peak_index])
        ax.plot([peak_conf], [peak_f1], marker="o", markersize=4.5, color=_TEXT,
                linestyle="none", zorder=5)
        ax.annotate(
            f"mean peak F1={peak_f1:.3f} at conf={peak_conf:.2f}\n(this validation set only)",
            xy=(peak_conf, peak_f1), xytext=(0.02, 0.90), textcoords="axes fraction",
            fontsize=8, color=_TEXT, va="top", ha="left",
            arrowprops={"arrowstyle": "-", "linewidth": 0.8, "color": _TEXT,
                        "shrinkA": 2, "shrinkB": 3},
        )
    style_axes(ax, xlabel="Confidence threshold", ylabel="F1",
               title="F1 versus confidence")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    _outside_legend(fig, ax, legend_anchor)
    _figure_caption(
        fig,
        "Curves are the evaluator's stored F1-versus-confidence arrays; the marked peak "
        "describes this validation set only and is not a universal operating threshold.",
        width,
    )

    notes = (
        f"F1 against the confidence threshold for {len(drawn)} classes and their stored mean "
        f"row, computed on this run's validation set. The marked maximum is the highest F1 "
        f"observed on this data only and is not a threshold that generalises to other data."
    )
    return finish_figure(
        fig, run, out_dir, figure_id, FIGURE_META[figure_id],
        input_paths=inputs,
        source_data={
            "arrays": {
                "conf": conf,
                "per_class_f1": f1[:class_rows],
                "mean_f1": mean_curve,
                "valid_mask": mask[:class_rows],
                "mean_peak_index": np.array([peak_index], dtype=np.int64),
            },
            "columns": {
                "conf": conf.tolist(),
                "mean_f1": mean_curve.tolist(),
            },
            "meta": {
                "units": {"conf": "confidence threshold (0-1)", "f1": "F1 score (0-1)"},
                "mean_row_index": mean_index,
                "mean_definition": "stored mean row from curves.npz (evaluator macro mean)",
                "class_order": [_class_name(meta, index) for index in range(class_rows)],
                "drawn_classes": drawn,
                "mean_peak": {"conf": float(conf[peak_index]) if mean_curve.size else None,
                              "f1": float(mean_curve[peak_index]) if mean_curve.size else None,
                              "scope": "this validation set only, not a universal threshold"},
                "smoothing": meta.get("smoothing"),
                "source": meta.get("source"),
                "transform": "none (native curves)",
            },
        },
        variables=["conf", "f1", "valid_classes"],
        units={"conf": "confidence threshold (0-1)", "f1": "F1 score (0-1)"},
        filters=["only classes flagged in valid_classes (they have GT) are drawn"],
        transforms=["none"],
        normalization=["none (native per-class and mean curves)"],
        sample_count={
            "classes_with_gt": int(mask[:class_rows].sum()),
            "curve_points": int(conf.size),
        },
        notes=notes,
        warnings=[],
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render(run: RunContext, out_dir: str | Path) -> list[dict]:
    """Render F07-F10 into ``out_dir`` and return one record per figure."""
    _, target = prepare_render(run, out_dir)
    return [
        _per_class_ap(run, target),
        _confusion_matrix(run, target),
        _pr_curve(run, target),
        _f1_confidence(run, target),
    ]
