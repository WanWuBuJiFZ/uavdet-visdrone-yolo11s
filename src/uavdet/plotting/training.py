"""Training-process report figures (F04-F06).

All three read this run's own ``train/results.csv`` (the Ultralytics epoch log
with the optional leading index column stripped). Nothing here loads a model:
the numbers are exactly the values the training loop recorded.

* F04 ``loss_curves``            — box / cls / dfl panels, train solid, val dashed.
* F05 ``map_curves``             — mAP@0.5 and mAP@0.5:0.95, displayed in percent.
* F06 ``precision_recall_epochs``— Precision and Recall in percent, explicitly
  marked as the native training-log definitions.

The three losses are never summed or compared on one shared axis: each panel has
its own y scale. ``results.csv`` keeps 0-1 metrics; ``to_percent`` is applied
exactly once, at draw time, and ``source_data`` keeps the stored 0-1 values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..types import RunContext
from .dataset import finish_figure, figure_settings, missing, prepare_render
from .style import (
    LOSS_COLORS,
    METRIC_COLORS,
    REPORT_COLORS,
    add_panel_label,
    style_axes,
    to_percent,
)

__all__ = ["render"]

_MUTED = REPORT_COLORS["muted"]

#: figure id -> (output basename, title, manuscript section); mirrors
#: ``uavdet.report.FIGURE_META`` without importing it (see plotting.dataset).
FIGURE_META: dict[str, tuple[str, str, str]] = {
    "F04": ("loss_curves", "Training and validation losses", "Training analysis"),
    "F05": ("map_curves", "mAP during training", "Training analysis"),
    "F06": ("precision_recall_epochs", "Precision and recall during training",
            "Training analysis"),
}

_RESULTS_FILE = "results.csv"

_LOSS_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("box", "Box loss", "train/box_loss"),
    ("cls", "Cls loss", "train/cls_loss"),
    ("dfl", "DFL loss", "train/dfl_loss"),
)

_METRIC_COLUMNS: dict[str, str] = {
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}


# ---------------------------------------------------------------------------
# results.csv
# ---------------------------------------------------------------------------
def _load_results(path: Path) -> tuple[Any, list[str]]:
    """Read ``results.csv``, normalise the header and drop empty metric rows."""
    import pandas as pd

    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.loc[:, [column for column in frame.columns
                          if not column.lower().startswith("unnamed")]]
    warnings: list[str] = []
    if "epoch" not in frame.columns:
        raise ValueError("results.csv has no 'epoch' column")
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    numeric = [column for column in frame.columns if column != "epoch"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    incomplete = frame[numeric].isna().any(axis=1)
    if bool(incomplete.any()):
        dropped = frame.loc[incomplete, "epoch"].tolist()
        warnings.append(
            f"{len(dropped)} epoch row(s) with missing or non-numeric values were not "
            f"plotted (epochs {dropped[:5]}{'…' if len(dropped) > 5 else ''})"
        )
        frame = frame.loc[~incomplete]
    frame = frame.dropna(subset=["epoch"]).sort_values("epoch", kind="mergesort")
    if frame.empty:
        raise ValueError("results.csv has no complete epoch rows to plot")
    return frame.reset_index(drop=True), warnings


def _columns_present(frame: Any, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _missing_columns(run: RunContext, out_dir: Path, figure_id: str,
                     path: Path, columns: Sequence[str]) -> dict[str, Any]:
    return missing(
        run, out_dir, figure_id, FIGURE_META[figure_id],
        reason=(f"{_RESULTS_FILE} does not contain the required column(s): "
                f"{', '.join(columns)}"),
        required=[path],
    )


# ---------------------------------------------------------------------------
# F04 — loss curves
# ---------------------------------------------------------------------------
def _loss_curves(run: RunContext, out_dir: Path) -> dict[str, Any]:
    figure_id = "F04"
    path = run.train_dir / _RESULTS_FILE
    if not path.is_file():
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason="this run has no train/results.csv, so no loss curve can be "
                              "drawn (a figure is never synthesised from memory)",
                       required=[path])
    try:
        frame, warnings = _load_results(path)
    except (ValueError, OSError) as exc:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"train/results.csv is unusable: {exc}", required=[path])

    required = [train for _key, _label, train in _LOSS_COLUMNS]
    required += [f"val/{key}_loss" for key, _label, _train in _LOSS_COLUMNS]
    absent = [column for column in required if column not in frame.columns]
    if absent:
        return _missing_columns(run, out_dir, figure_id, path, absent)

    settings = figure_settings(run)
    import matplotlib.pyplot as plt

    width = settings["width"]
    fig, axes = plt.subplots(1, 3, figsize=(width, width * 0.36), layout="constrained")
    epochs = frame["epoch"].to_numpy(dtype=float)
    for ax, tag, (key, label, train_column) in zip(axes, "abc", _LOSS_COLUMNS):
        colour = LOSS_COLORS[key]
        ax.plot(epochs, frame[train_column].to_numpy(dtype=float), color=colour,
                linestyle="-", linewidth=1.8, label="train")
        ax.plot(epochs, frame[f"val/{key}_loss"].to_numpy(dtype=float), color=colour,
                linestyle="--", linewidth=1.8, label="val")
        style_axes(ax, xlabel="Epoch", ylabel="Loss", title=label)
        ax.legend(fontsize=8, frameon=False, loc="upper right", handlelength=2.6)
        add_panel_label(ax, tag, x=-0.20, y=1.04)
    fig.supxlabel(
        "Each panel keeps its own y scale: the three losses are different quantities and "
        "are neither summed nor compared on a shared axis.",
        fontsize=8, color=_MUTED,
    )

    arrays: dict[str, Any] = {"epoch": epochs}
    for key, _label, train_column in _LOSS_COLUMNS:
        arrays[f"train_{key}_loss"] = frame[train_column].to_numpy(dtype=float)
        arrays[f"val_{key}_loss"] = frame[f"val/{key}_loss"].to_numpy(dtype=float)
    columns = {"epoch": epochs.tolist()}
    for key, _label, train_column in _LOSS_COLUMNS:
        columns[f"train_{key}_loss"] = frame[train_column].tolist()
        columns[f"val_{key}_loss"] = frame[f"val/{key}_loss"].tolist()

    notes = (
        "Training (solid) and validation (dashed) box, classification and DFL losses "
        "recorded once per epoch in this run's results.csv. Box, cls and DFL are "
        "different loss terms, so they are shown in separate panels with separate scales."
    )
    return finish_figure(
        fig, run, out_dir, figure_id, FIGURE_META[figure_id],
        input_paths=[path],
        source_data={
            "arrays": arrays,
            "columns": columns,
            "meta": {
                "units": {"epoch": "epoch index (1-based as logged)", "loss": "native loss value"},
                "reference_lines": {"train": "solid", "val": "dashed"},
                "loss_colors": {key: LOSS_COLORS[key] for key, _l, _c in _LOSS_COLUMNS},
                "transform": "none (raw logged loss values)",
            },
        },
        variables=["epoch", "train_box_loss", "val_box_loss", "train_cls_loss",
                   "val_cls_loss", "train_dfl_loss", "val_dfl_loss"],
        units={"epoch": "epoch index", "loss": "native loss value"},
        filters=["rows with non-numeric values dropped (reported in warnings)"],
        transforms=["none"],
        normalization=["none (raw logged losses)"],
        sample_count={"epochs": int(epochs.size),
                      "train_series": int(len(_LOSS_COLUMNS)),
                      "val_series": int(len(_LOSS_COLUMNS))},
        notes=notes,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# F05 — mAP curves
# ---------------------------------------------------------------------------
def _map_curves(run: RunContext, out_dir: Path) -> dict[str, Any]:
    figure_id = "F05"
    path = run.train_dir / _RESULTS_FILE
    if not path.is_file():
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason="this run has no train/results.csv, so no mAP curve can be drawn",
                       required=[path])
    try:
        frame, warnings = _load_results(path)
    except (ValueError, OSError) as exc:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"train/results.csv is unusable: {exc}", required=[path])

    series = (("map50", _METRIC_COLUMNS["map50"], "mAP@0.5"),
              ("map50_95", _METRIC_COLUMNS["map50_95"], "mAP@0.5:0.95"))
    absent = [column for _key, column, _label in series if column not in frame.columns]
    if absent:
        return _missing_columns(run, out_dir, figure_id, path, absent)

    settings = figure_settings(run)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    width = settings["width"]
    fig, ax = plt.subplots(figsize=(width, width * 0.55), layout="constrained")
    epochs = frame["epoch"].to_numpy(dtype=float)
    arrays: dict[str, Any] = {"epoch": epochs}
    columns: dict[str, Any] = {"epoch": epochs.tolist()}
    for key, column, label in series:
        stored = frame[column].to_numpy(dtype=float)
        arrays[f"{key}_0to1"] = stored
        arrays[f"{key}_percent"] = to_percent(stored)
        columns[f"{key}_0to1"] = stored.tolist()
        columns[f"{key}_percent"] = to_percent(stored).tolist()
        ax.plot(epochs, to_percent(stored), color=METRIC_COLORS[key], linewidth=1.8,
                linestyle="-", label=label)
    style_axes(ax, xlabel="Epoch", ylabel="mAP (%)", title="mAP during training")
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.legend(fontsize=8, frameon=False, loc="lower right", handlelength=2.6)
    fig.supxlabel(
        "Stored mAP values are fractions (0-1) and are multiplied by 100 once for "
        "display; the source data keeps both the stored and the displayed values.",
        fontsize=8, color=_MUTED,
    )

    notes = (
        "Per-epoch mAP@0.5 and mAP@0.5:0.95 from this run's own training log, shown in "
        "percent. The curve describes the validation measurements taken during that "
        "training run and nothing beyond it."
    )
    return finish_figure(
        fig, run, out_dir, figure_id, FIGURE_META[figure_id],
        input_paths=[path],
        source_data={
            "arrays": arrays,
            "columns": columns,
            "meta": {
                "units": {"epoch": "epoch index", "map50_0to1": "fraction 0-1",
                          "map50_percent": "percent (0-100)",
                          "map50_95_0to1": "fraction 0-1",
                          "map50_95_percent": "percent (0-100)"},
                "display_transform": "percent = stored_fraction * 100, applied once at draw time",
                "metric_colors": {"map50": METRIC_COLORS["map50"],
                                  "map50_95": METRIC_COLORS["map50_95"]},
            },
        },
        variables=["epoch", "map50", "map50_95"],
        units={"epoch": "epoch index", "map": "percent (0-100) for display; stored 0-1"},
        filters=["rows with non-numeric values dropped (reported in warnings)"],
        transforms=["percent = fraction * 100 (once, display only)"],
        normalization=["none"],
        sample_count={"epochs": int(epochs.size)},
        notes=notes,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# F06 — precision / recall per epoch
# ---------------------------------------------------------------------------
def _precision_recall_epochs(run: RunContext, out_dir: Path) -> dict[str, Any]:
    figure_id = "F06"
    path = run.train_dir / _RESULTS_FILE
    if not path.is_file():
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason="this run has no train/results.csv, so no precision/recall "
                              "curve can be drawn",
                       required=[path])
    try:
        frame, warnings = _load_results(path)
    except (ValueError, OSError) as exc:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"train/results.csv is unusable: {exc}", required=[path])

    series = (("precision", _METRIC_COLUMNS["precision"], "Precision"),
              ("recall", _METRIC_COLUMNS["recall"], "Recall"))
    absent = [column for _key, column, _label in series if column not in frame.columns]
    if absent:
        return _missing_columns(run, out_dir, figure_id, path, absent)

    settings = figure_settings(run)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    width = settings["width"]
    fig, ax = plt.subplots(figsize=(width, width * 0.55), layout="constrained")
    epochs = frame["epoch"].to_numpy(dtype=float)
    arrays: dict[str, Any] = {"epoch": epochs}
    columns: dict[str, Any] = {"epoch": epochs.tolist()}
    for key, column, label in series:
        stored = frame[column].to_numpy(dtype=float)
        arrays[f"{key}_0to1"] = stored
        arrays[f"{key}_percent"] = to_percent(stored)
        columns[f"{key}_0to1"] = stored.tolist()
        columns[f"{key}_percent"] = to_percent(stored).tolist()
        ax.plot(epochs, to_percent(stored), color=METRIC_COLORS[key], linewidth=1.8,
                linestyle="-", label=label)
    style_axes(ax, xlabel="Epoch", ylabel="(%)", title="Precision and recall during training")
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.legend(fontsize=8, frameon=False, loc="lower right", handlelength=2.6)
    fig.supxlabel(
        "Native Ultralytics training-log definitions (metrics/precision(B) and "
        "metrics/recall(B) at the validator's operating point); percent = stored "
        "fraction * 100, applied once.",
        fontsize=8, color=_MUTED,
    )

    notes = (
        "Per-epoch Precision and Recall exactly as recorded in this run's training log, "
        "in percent. These are the native training-time validator definitions, not a "
        "fixed-confidence re-matching of the saved predictions."
    )
    return finish_figure(
        fig, run, out_dir, figure_id, FIGURE_META[figure_id],
        input_paths=[path],
        source_data={
            "arrays": arrays,
            "columns": columns,
            "meta": {
                "units": {"epoch": "epoch index", "precision_0to1": "fraction 0-1",
                          "precision_percent": "percent (0-100)",
                          "recall_0to1": "fraction 0-1",
                          "recall_percent": "percent (0-100)"},
                "display_transform": "percent = stored_fraction * 100, applied once at draw time",
                "definition": "native training-log metrics/precision(B) and metrics/recall(B)",
                "metric_colors": {"precision": METRIC_COLORS["precision"],
                                  "recall": METRIC_COLORS["recall"]},
            },
        },
        variables=["epoch", "precision", "recall"],
        units={"epoch": "epoch index", "precision": "percent (0-100)",
               "recall": "percent (0-100)"},
        filters=["rows with non-numeric values dropped (reported in warnings)"],
        transforms=["percent = fraction * 100 (once, display only)"],
        normalization=["none"],
        sample_count={"epochs": int(epochs.size)},
        notes=notes,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render(run: RunContext, out_dir: str | Path) -> list[dict]:
    """Render F04-F06 into ``out_dir`` and return one record per figure."""
    _, target = prepare_render(run, out_dir)
    return [
        _loss_curves(run, target),
        _map_curves(run, target),
        _precision_recall_epochs(run, target),
    ]
