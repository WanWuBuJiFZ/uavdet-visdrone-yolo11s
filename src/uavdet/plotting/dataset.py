"""Dataset-characteristics report figures (F01-F03).

Inputs are the saved audit products only (``audit/dataset_stats.json``); no
image is re-read, no dataset is re-scanned and no model is loaded.

* F01 ``class_distribution``       — ground-truth instances per class, train/val.
* F02 ``object_size_distribution`` — relative object area ``w_norm * h_norm``
  using the audit's log-spaced bins and quantiles. This is **not** the COCO
  ``AP_small`` area definition; the figure says so on its face.
* F03 ``objects_per_image``        — objects per image histogram + ECDF, shared
  bins, empty-label images included, sample counts in the legend.

The module-level helpers below (``figure_settings``, ``finish_figure``, …) are
shared by :mod:`uavdet.plotting.training`, :mod:`uavdet.plotting.evaluation`
and :mod:`uavdet.plotting.cases`; all four modules belong to the same plotting
deliverable, so the kit lives here instead of duplicating it four times.

Every renderer returns one record per figure: ``FigureRecord(...).to_dict()``
only after the files were really written, ``missing_record(...)`` otherwise.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..artifacts import read_json, sha256_file
from ..types import CLASS_NAMES, NUM_CLASSES, RunContext
from .style import (
    REPORT_COLORS,
    SPLIT_COLORS,
    STYLE_VERSION,
    FigureRecord,
    add_panel_label,
    apply_style,
    ensure_dir,
    missing_record,
    save_figure,
    software_versions,
    style_axes,
    write_source_data,
)

__all__ = ["render"]

_MUTED = REPORT_COLORS["muted"]
_TEXT = REPORT_COLORS["text"]

#: figure id -> (output basename, title, manuscript section).
#:
#: Mirrors ``uavdet.report.FIGURE_META`` (which is the authority for the file
#: names). It is repeated here instead of imported because ``report`` imports
#: every plotting module, so importing it back from a renderer would be a
#: circular import; ``tests/test_figures.py`` asserts the two tables agree.
FIGURE_META: dict[str, tuple[str, str, str]] = {
    "F01": ("class_distribution", "Ground-truth instances per class and split", "Dataset"),
    "F02": ("object_size_distribution", "Relative object area distribution", "Dataset"),
    "F03": ("objects_per_image", "Objects per image (train vs val)", "Dataset"),
}


# ---------------------------------------------------------------------------
# shared figure kit (used by training/evaluation/cases too)
# ---------------------------------------------------------------------------
def figure_settings(run: RunContext) -> dict[str, Any]:
    """Export settings from ``run.config['figures']`` (DPI, formats, width, font)."""
    figures = dict((run.config or {}).get("figures") or {})
    allowed = []
    for fmt in figures.get("formats") or ("png", "pdf", "svg"):
        name = str(fmt).strip().lower().lstrip(".")
        if name in ("png", "pdf", "svg") and name not in allowed:
            allowed.append(name)
    if not allowed:
        allowed = ["png"]
    width = float(figures.get("figure_width", 7.0))
    if not math.isfinite(width) or width <= 0:
        width = 7.0
    dpi = int(figures.get("dpi", 300))
    if dpi <= 0:
        dpi = 300
    return {
        "dpi": dpi,
        "formats": tuple(allowed),
        "width": width,
        "font_family": str(figures.get("font_family", "auto")),
    }


def figure_tag(out_dir: str | Path) -> str:
    """The ``figures/<tag>`` directory name, recorded in every manifest entry."""
    return Path(out_dir).resolve().name or "v1"


def prepare_render(run: RunContext, out_dir: str | Path) -> tuple[dict[str, Any], Path]:
    """Apply the report style once and make sure the output directory exists."""
    settings = figure_settings(run)
    apply_style(settings["font_family"])
    target = ensure_dir(Path(out_dir))
    return settings, target


def input_paths_and_hashes(run: RunContext,
                           paths: Iterable[str | Path]) -> tuple[list[str], dict[str, str]]:
    """Portable input paths plus SHA256 of the files that really exist.

    Files inside the run are recorded run-relative; files outside it (e.g. the
    validation images used by the case panels) are recorded as
    ``external/<file name>`` so the manifest never embeds a machine-specific,
    possibly non-ASCII absolute path. Identity is preserved by the SHA256.
    """
    rel_paths: list[str] = []
    hashes: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        digest = sha256_file(path) if path.is_file() else None
        rel = run.rel(path)
        if Path(rel).is_absolute():
            rel = f"external/{path.name}"
            if rel in hashes and hashes[rel] != digest:
                rel = f"external/{(digest or '')[:8]}_{path.name}"
        if rel not in rel_paths:
            rel_paths.append(rel)
        if digest is not None:
            hashes[rel] = digest
    return rel_paths, hashes


def missing(run: RunContext, out_dir: str | Path, figure_id: str,
            meta: tuple[str, str, str], *,
            reason: str, required: Sequence[str | Path] = ()) -> dict[str, Any]:
    """Missing record for one figure id (never a blank placeholder image)."""
    return missing_record(
        figure_id,
        meta[1],
        meta[2],
        run_id=run.run_id,
        scope=str(run.scope),
        figure_tag=figure_tag(out_dir),
        reason=reason,
        required=[str(p) for p in required],
    )


def finish_figure(
    fig,
    run: RunContext,
    out_dir: str | Path,
    figure_id: str,
    meta: tuple[str, str, str],
    *,
    input_paths: Sequence[str | Path],
    source_data: Mapping[str, Any] | None,
    variables: Sequence[str],
    units: Mapping[str, str],
    sample_count: Mapping[str, int],
    notes: str,
    filters: Sequence[str] = (),
    transforms: Sequence[str] = (),
    normalization: Sequence[str] = (),
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Export one finished figure and return its manifest record.

    ``save_figure`` raises instead of leaving a blank file behind, so a record
    with ``data_status="ok"`` means the PNG/PDF/SVG really exist.
    """
    basename = meta[0]
    try:
        output_paths, output_sizes = save_figure(
            fig, out_dir, basename, formats=figure_settings(run)["formats"],
            dpi=figure_settings(run)["dpi"],
        )
    except Exception:
        import matplotlib.pyplot as plt

        plt.close(fig)
        raise
    source_data_paths: list[str] = []
    if source_data:
        written = write_source_data(out_dir, basename, **dict(source_data))
        source_data_paths = list(written.values())
    rel_inputs, hashes = input_paths_and_hashes(run, input_paths)
    portable_outputs = [run.rel(path) for path in output_paths]
    portable_sizes = {run.rel(path): size for path, size in output_sizes.items()}
    record = FigureRecord(
        figure_id=figure_id,
        title=meta[1],
        manuscript_section=meta[2],
        run_id=run.run_id,
        scope=str(run.scope),
        figure_tag=figure_tag(out_dir),
        input_paths=rel_inputs,
        input_sha256=hashes,
        variables=list(variables),
        units=dict(units),
        class_order=list(CLASS_NAMES),
        filters=list(filters),
        transforms=list(transforms),
        normalization=list(normalization),
        sample_count={str(k): int(v) for k, v in sample_count.items()},
        data_status="ok",
        output_paths=portable_outputs,
        output_sizes=portable_sizes,
        source_data_paths=[run.rel(path) for path in source_data_paths],
        software_versions=software_versions(),
        style_version=STYLE_VERSION,
        warnings=list(warnings),
        notes=notes,
    )
    return record.to_dict()


def legend_with_counts(ax, **kwargs) -> None:
    """Compact legend placed outside the data area (never over the curves)."""
    options = dict(fontsize=8, frameon=False, handlelength=2.6, labelspacing=0.4)
    options.update(kwargs)
    ax.legend(**options)


def split_label(split: str, num_images: int, num_instances: int | None = None) -> str:
    text = f"{split} ({num_images} images"
    if num_instances is not None:
        text += f", {num_instances} instances"
    return text + ")"


# ---------------------------------------------------------------------------
# audit input
# ---------------------------------------------------------------------------
def _load_stats(run: RunContext) -> tuple[dict[str, Any], list[str]]:
    """Read ``audit/dataset_stats.json``; returns ``(stats, warnings)``."""
    path = run.audit_dir / "dataset_stats.json"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    stats = read_json(path)
    warnings: list[str] = []
    if str(stats.get("status", "")) != "ok":
        warnings.append(
            f"audit status is '{stats.get('status')}'; counts come from a dataset with "
            f"recorded issues"
        )
    return stats, warnings


def _require_split(stats: Mapping[str, Any], split: str, keys: Sequence[str]) -> dict[str, Any]:
    block = (stats.get("splits") or {}).get(split)
    if not isinstance(block, Mapping):
        raise KeyError(f"dataset_stats.json has no splits.{split}")
    missing_keys = [key for key in keys if key not in block]
    if missing_keys:
        raise KeyError(f"splits.{split} lacks {', '.join(missing_keys)}")
    return dict(block)


def _class_names(stats: Mapping[str, Any]) -> list[str]:
    names = [str(name) for name in (stats.get("class_names") or CLASS_NAMES)]
    if len(names) != NUM_CLASSES:  # defensive: never mislabel the axis
        raise ValueError(f"dataset_stats.json lists {len(names)} class names, expected {NUM_CLASSES}")
    return names


# ---------------------------------------------------------------------------
# F01 — class distribution
# ---------------------------------------------------------------------------
def _class_distribution(run: RunContext, out_dir: Path) -> dict[str, Any]:
    figure_id = "F01"
    stats_path = run.audit_dir / "dataset_stats.json"
    try:
        stats, warnings = _load_stats(run)
        train = _require_split(stats, "train", ("per_class_list", "num_images", "num_instances"))
        val = _require_split(stats, "val", ("per_class_list", "num_images", "num_instances"))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"dataset audit statistics are not usable: {exc}",
                       required=[stats_path])

    names = _class_names(stats)
    train_counts = np.asarray(train["per_class_list"], dtype=np.int64)
    val_counts = np.asarray(val["per_class_list"], dtype=np.int64)
    if train_counts.size != NUM_CLASSES or val_counts.size != NUM_CLASSES:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason="per-class counts do not cover the 10 fixed classes",
                       required=[stats_path])

    settings = figure_settings(run)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    width = settings["width"]
    fig, ax = plt.subplots(figsize=(width, width * 0.52), layout="constrained")
    positions = np.arange(NUM_CLASSES, dtype=float)
    bar_width = 0.38
    ax.bar(positions - bar_width / 2, train_counts, width=bar_width,
           color=SPLIT_COLORS["train"], label=split_label(
               "train", int(train["num_images"]), int(train["num_instances"])))
    ax.bar(positions + bar_width / 2, val_counts, width=bar_width,
           color=SPLIT_COLORS["val"], label=split_label(
               "val", int(val["num_images"]), int(val["num_instances"])))
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=30, ha="right", rotation_mode="anchor")
    style_axes(ax, xlabel="Class (fixed VisDrone DET order 0-9)",
               ylabel="Ground-truth instances", title="Ground-truth instances per class")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.margins(x=0.01, y=0.08)
    legend_with_counts(ax, loc="upper right")
    fig.supxlabel(
        "Raw ground-truth instance counts recorded by the dataset audit; "
        "counts and shares are not mixed on one axis.",
        fontsize=8, color=_MUTED,
    )

    notes = (
        f"Ground-truth instances of each of the {NUM_CLASSES} fixed classes in the train "
        f"and val splits, as recorded by the dataset audit. Counts are raw integer "
        f"instances, so classes with few annotations stay visibly small."
    )
    return finish_figure(
        fig, run, out_dir, figure_id, FIGURE_META[figure_id],
        input_paths=[stats_path],
        source_data={
            "arrays": {
                "class_id": np.arange(NUM_CLASSES, dtype=np.int64),
                "train_gt_instances": train_counts,
                "val_gt_instances": val_counts,
            },
            "columns": {
                "class_id": list(range(NUM_CLASSES)),
                "class_name": names,
                "train_gt_instances": train_counts.tolist(),
                "val_gt_instances": val_counts.tolist(),
            },
            "meta": {
                "units": {"gt_instances": "count"},
                "transform": "none",
                "class_order": names,
                "source_fields": ["splits.train.per_class_list", "splits.val.per_class_list"],
            },
        },
        variables=["class_id", "class_name", "train_gt_instances", "val_gt_instances"],
        units={"gt_instances": "count"},
        filters=[f"split in {{train, val}} as recorded by the audit (scope={stats.get('scope')})"],
        transforms=[],
        normalization=["none (raw ground-truth instance counts)"],
        sample_count={
            "train_images": int(train["num_images"]),
            "train_instances": int(train["num_instances"]),
            "val_images": int(val["num_images"]),
            "val_instances": int(val["num_instances"]),
            "classes": NUM_CLASSES,
        },
        notes=notes,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# F02 — relative object area distribution
# ---------------------------------------------------------------------------
def _histogram_block(block: Mapping[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
    histogram = block.get("area_histogram") or {}
    edges = np.asarray(histogram.get("edges") or [], dtype=float)
    counts = np.asarray(histogram.get("counts") or [], dtype=np.int64)
    if edges.size < 2 or counts.size != edges.size - 1:
        raise ValueError(f"splits.{split}.area_histogram is malformed")
    if not np.all(np.isfinite(edges)) or np.any(edges <= 0):
        raise ValueError(f"splits.{split}.area_histogram edges are not positive finite")
    return edges, counts


def _object_size_distribution(run: RunContext, out_dir: Path) -> dict[str, Any]:
    figure_id = "F02"
    stats_path = run.audit_dir / "dataset_stats.json"
    try:
        stats, warnings = _load_stats(run)
        train = _require_split(stats, "train", ("area_histogram", "area_quantiles", "num_images"))
        val = _require_split(stats, "val", ("area_histogram", "area_quantiles", "num_images"))
        train_edges, train_counts = _histogram_block(train, "train")
        val_edges, val_counts = _histogram_block(val, "val")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"the audit's relative-area statistics are not usable: {exc}",
                       required=[stats_path])

    if train_edges.shape != val_edges.shape or not np.allclose(train_edges, val_edges,
                                                              rtol=1e-9, atol=0.0):
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason="train and val area histograms do not share the same bin edges, "
                              "so the shared-bin comparison would be invalid",
                       required=[stats_path])

    settings = figure_settings(run)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    width = settings["width"]
    fig, axes = plt.subplots(2, 1, figsize=(width, width * 0.86), sharex=True,
                             layout="constrained")
    bar_widths = np.diff(train_edges)
    panels = (("a", "train", train, train_counts, SPLIT_COLORS["train"]),
              ("b", "val", val, val_counts, SPLIT_COLORS["val"]))
    for ax, (label, split, block, counts, colour) in zip(axes, panels):
        ax.bar(train_edges[:-1], counts, width=bar_widths, align="edge",
               color=colour, linewidth=0, alpha=0.9)
        quantiles = block.get("area_quantiles") or {}
        for key, name, style in (("p50", "median (p50)", "-"), ("p95", "p95", "--")):
            value = quantiles.get(key)
            if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                continue
            ax.axvline(float(value), color=_TEXT, linestyle=style, linewidth=0.9,
                       alpha=0.8, label=f"{name} = {float(value):.3g}")
        ax.set_xscale("log")
        style_axes(ax, ylabel="GT instances per bin")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.set_title(
            f"({label}) {split} — {int(block.get('num_instances', int(counts.sum())))} instances, "
            f"{int(block['num_images'])} images",
            loc="left", pad=8,
        )
        legend_with_counts(ax, loc="upper left")
    axes[-1].set_xlabel("Relative area  $w_{norm} \\times h_{norm}$  (log scale, fraction of image)")
    fig.supxlabel(
        "Bins are the audit's stored log10-spaced relative-area bins; shared by train and "
        "val. Relative area is not the COCO $AP_{small}$ definition.",
        fontsize=8, color=_MUTED,
    )

    train_q = {key: float((train.get("area_quantiles") or {}).get(key, 0.0))
               for key in ("min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max", "mean")}
    val_q = {key: float((val.get("area_quantiles") or {}).get(key, 0.0))
             for key in ("min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max", "mean")}
    notes = (
        "Distribution of ground-truth relative object area "
        "($w_{norm} \\times h_{norm}$) in the train and val splits, drawn with the audit's "
        "stored log-spaced bins. The definition is relative area as recorded by the audit, "
        "not the COCO $AP_{small}$ area range."
    )
    return finish_figure(
        fig, run, out_dir, figure_id, FIGURE_META[figure_id],
        input_paths=[stats_path],
        source_data={
            "arrays": {
                "bin_edges": train_edges,
                "train_counts": train_counts,
                "val_counts": val_counts,
                "train_area_p50": np.array([train_q["p50"]], dtype=float),
                "train_area_p95": np.array([train_q["p95"]], dtype=float),
                "val_area_p50": np.array([val_q["p50"]], dtype=float),
                "val_area_p95": np.array([val_q["p95"]], dtype=float),
            },
            "columns": {
                "bin_left": train_edges[:-1].tolist(),
                "bin_right": train_edges[1:].tolist(),
                "train_gt_instances": train_counts.tolist(),
                "val_gt_instances": val_counts.tolist(),
            },
            "meta": {
                "units": {"relative_area": "fraction of image area (w_norm*h_norm)",
                          "gt_instances_per_bin": "count"},
                "binning": "log10-spaced bins as stored by the audit",
                "shared_bins": True,
                "definition": "relative area = (w_px/width)*(h_px/height); not COCO AP_small",
                "quantiles_train": train_q,
                "quantiles_val": val_q,
            },
        },
        variables=["bin_left", "bin_right", "relative_area", "gt_instances_per_bin",
                   "area_p50", "area_p95"],
        units={"relative_area": "fraction of image area (w_norm*h_norm)",
               "gt_instances_per_bin": "count"},
        filters=["relative area > 0 (zero-area boxes are audit errors, not plotted)"],
        transforms=["log10-spaced x axis; x = relative area = (w/width)*(h/height)"],
        normalization=["none (raw GT instance counts per bin)"],
        sample_count={
            "train_images": int(train["num_images"]),
            "train_instances": int(train.get("num_instances", int(train_counts.sum()))),
            "val_images": int(val["num_images"]),
            "val_instances": int(val.get("num_instances", int(val_counts.sum()))),
            "bins": int(train_counts.size),
        },
        notes=notes,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# F03 — objects per image
# ---------------------------------------------------------------------------
def _per_image_counts(block: Mapping[str, Any], split: str) -> tuple[np.ndarray, int]:
    raw = list(block.get("per_image_objects") or [])
    if not raw:
        raise ValueError(f"splits.{split}.per_image_objects is empty")
    valid = [int(value) for value in raw if isinstance(value, (int, float))]
    dropped = len(raw) - len(valid)
    if not valid:
        raise ValueError(f"splits.{split}.per_image_objects has no numeric entries")
    return np.asarray(valid, dtype=np.int64), dropped


def _objects_per_image(run: RunContext, out_dir: Path) -> dict[str, Any]:
    figure_id = "F03"
    stats_path = run.audit_dir / "dataset_stats.json"
    try:
        stats, warnings = _load_stats(run)
        train = _require_split(stats, "train",
                               ("per_image_objects", "per_image_ids", "num_images"))
        val = _require_split(stats, "val",
                             ("per_image_objects", "per_image_ids", "num_images"))
        train_counts, train_dropped = _per_image_counts(train, "train")
        val_counts, val_dropped = _per_image_counts(val, "val")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return missing(run, out_dir, figure_id, FIGURE_META[figure_id],
                       reason=f"per-image object counts are not usable: {exc}",
                       required=[stats_path])

    if train_dropped:
        warnings.append(f"{train_dropped} train image(s) without a decodable object count excluded")
    if val_dropped:
        warnings.append(f"{val_dropped} val image(s) without a decodable object count excluded")

    settings = figure_settings(run)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    width = settings["width"]
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.44), layout="constrained")
    highest = int(max(train_counts.max(), val_counts.max(), 1))
    bin_count = int(min(highest, 25))
    edges = np.linspace(0, highest, bin_count + 1)
    hist_ax, ecdf_ax = axes[0], axes[1]
    handles = []
    for label, values, colour in (("train", train_counts, SPLIT_COLORS["train"]),
                                  ("val", val_counts, SPLIT_COLORS["val"])):
        hist_ax.hist(values, bins=edges, histtype="step", linewidth=1.6, color=colour,
                     label=f"{label} (n={int(values.size)} images)")
    style_axes(hist_ax, xlabel="Ground-truth objects per image", ylabel="Images",
               title="Objects per image")
    hist_ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    add_panel_label(hist_ax, "a")

    for label, values, colour in (("train", train_counts, SPLIT_COLORS["train"]),
                                  ("val", val_counts, SPLIT_COLORS["val"])):
        ordered = np.sort(values)
        share = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
        ecdf_ax.step(ordered, share, where="post", linewidth=1.6, color=colour,
                     label=f"{label} (n={int(values.size)} images)")
    style_axes(ecdf_ax, xlabel="Ground-truth objects per image",
               ylabel="Cumulative fraction of images", title="ECDF")
    ecdf_ax.set_ylim(0.0, 1.0)
    add_panel_label(ecdf_ax, "b")
    for ax in axes:
        ax.set_xlim(-0.5, float(highest) + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    handles, labels = hist_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=8,
               frameon=False, handlelength=2.6, columnspacing=1.6)
    fig.supxlabel(
        "Both panels use the same definition and the same bins; the denominator is the "
        "number of images and images with empty labels count as 0 objects.",
        fontsize=8, color=_MUTED,
    )

    notes = (
        "Distribution of ground-truth objects per image for train and val, including "
        "images whose label file is empty or missing (counted as 0). Sample sizes are the "
        "number of images per split, shown in the legend."
    )
    return finish_figure(
        fig, run, out_dir, figure_id, FIGURE_META[figure_id],
        input_paths=[stats_path],
        source_data={
            "arrays": {
                "train_objects_per_image": train_counts,
                "val_objects_per_image": val_counts,
                "bin_edges": edges,
            },
            "columns": {
                "split": ["train"] * train_counts.size + ["val"] * val_counts.size,
                "objects_per_image": np.concatenate([train_counts, val_counts]).tolist(),
            },
            "meta": {
                "units": {"objects_per_image": "count",
                          "ecdf": "fraction of images (0-1)"},
                "shared_bins": {"count": int(bin_count), "edges": edges.tolist()},
                "definition": "ground-truth objects per image; label-empty images count as 0",
                "empty_label_images": {
                    "train": int(train.get("empty_label_images", 0)),
                    "val": int(val.get("empty_label_images", 0)),
                },
            },
        },
        variables=["objects_per_image", "images", "cumulative_fraction_of_images"],
        units={"objects_per_image": "count", "images": "count",
               "cumulative_fraction_of_images": "fraction of images (0-1)"},
        filters=["all audited images, including empty / missing label files (count = 0)"],
        transforms=["ECDF = share of images with at most n objects",
                    "shared histogram bins over 0..max objects per image"],
        normalization=["none (counts and cumulative fractions, no resampling)"],
        sample_count={
            "train_images": int(train_counts.size),
            "val_images": int(val_counts.size),
            "train_instances": int(train_counts.sum()),
            "val_instances": int(val_counts.sum()),
            "train_empty_label_images": int(np.count_nonzero(train_counts == 0)),
            "val_empty_label_images": int(np.count_nonzero(val_counts == 0)),
        },
        notes=notes,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render(run: RunContext, out_dir: str | Path) -> list[dict]:
    """Render F01-F03 into ``out_dir`` and return one record per figure."""
    _, target = prepare_render(run, out_dir)
    return [
        _class_distribution(run, target),
        _object_size_distribution(run, target),
        _objects_per_image(run, target),
    ]
