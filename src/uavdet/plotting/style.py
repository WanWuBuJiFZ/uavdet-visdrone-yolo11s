"""Shared report-figure style, palette and export helpers (no Qt, no YOLO).

Style provenance: the rules come from the approved course-project figures
(the legacy project's chart-style implementation), adapted into
this package as ``plotting/assets/sci-report.mplstyle`` so a cloud machine never
needs the original skill directory.

Colour semantics (must stay identical across every figure):

===============  ==============
metric           colour
===============  ==============
mAP@0.5          ``#0072B2`` blue
mAP@0.5:0.95     ``#009E73`` green
Precision        ``#E69F00`` orange
Recall           ``#D55E00`` vermilion
box loss         blue
cls loss         green
dfl loss         orange
train            solid line
val              dashed line
===============  ==============
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .. import __version__
from ..types import CLASS_NAMES

__all__ = [
    "STYLE_VERSION",
    "REPORT_COLORS",
    "METRIC_COLORS",
    "LOSS_COLORS",
    "SPLIT_COLORS",
    "CLASS_COLORS",
    "FigureRecord",
    "apply_style",
    "available_fonts",
    "choose_font_family",
    "style_axes",
    "add_panel_label",
    "save_figure",
    "figure_output_paths",
    "write_source_data",
    "missing_record",
    "to_percent",
    "software_versions",
    "load_style_path",
    "ensure_dir",
]

STYLE_VERSION = "sci-report-1.0"

REPORT_COLORS: dict[str, str] = {
    "primary": "#0072B2",     # blue
    "secondary": "#009E73",   # green
    "success": "#E69F00",     # orange (Precision)
    "danger": "#D55E00",      # vermilion (Recall)
    "info": "#56B4E9",        # light blue
    "bg": "#FFFFFF",
    "grid": "#D8DDE3",
    "text": "#252525",
    "muted": "#6B7280",
}

METRIC_COLORS: dict[str, str] = {
    "map50": REPORT_COLORS["primary"],
    "map50_95": REPORT_COLORS["secondary"],
    "precision": REPORT_COLORS["success"],
    "recall": REPORT_COLORS["danger"],
    "f1": "#CC79A7",
}

LOSS_COLORS: dict[str, str] = {
    "box": REPORT_COLORS["primary"],
    "cls": REPORT_COLORS["secondary"],
    "dfl": REPORT_COLORS["success"],
}

SPLIT_COLORS: dict[str, str] = {
    "train": REPORT_COLORS["primary"],
    "val": REPORT_COLORS["danger"],
}

#: Ten-class palette (Okabe-Ito extended); used with distinct line styles.
CLASS_COLORS: tuple[str, ...] = (
    "#0072B2", "#009E73", "#E69F00", "#D55E00", "#56B4E9",
    "#CC79A7", "#F0E442", "#8C6D31", "#4B4B4B", "#7E57C2",
)
LINE_STYLES: tuple[str, ...] = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)))
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "<")

_FONT_CANDIDATES: tuple[str, ...] = (
    "Arial", "Helvetica", "DejaVu Sans", "Noto Sans CJK SC", "Source Han Sans SC",
    "Hiragino Sans GB", "Arial Unicode MS", "Microsoft YaHei", "PingFang SC",
)


def load_style_path() -> Path:
    """Locate the bundled ``sci-report.mplstyle`` (works after pip install)."""
    try:  # package resource (preferred)
        from importlib import resources

        resource = resources.files("uavdet.plotting").joinpath("assets/sci-report.mplstyle")
        with resources.as_file(resource) as path:
            if Path(path).exists():
                return Path(path)
    except (ModuleNotFoundError, FileNotFoundError, AttributeError):
        pass
    local = Path(__file__).resolve().parent / "assets" / "sci-report.mplstyle"
    if local.exists():
        return local
    raise FileNotFoundError(
        "bundled figure style assets/sci-report.mplstyle is missing from the install"
    )


def available_fonts() -> set[str]:
    from matplotlib import font_manager

    return {font.name for font in font_manager.fontManager.ttflist}


def choose_font_family(requested: str = "auto") -> str:
    """Pick the first installed sans-serif font, DejaVu Sans as the floor."""
    installed = available_fonts()
    if requested and requested != "auto":
        if requested in installed:
            return requested
        return "DejaVu Sans"
    for name in _FONT_CANDIDATES:
        if name in installed:
            return name
    return "DejaVu Sans"


def _ensure_mpl_config_dir() -> None:
    """Point matplotlib at a writable cache dir (locked-down home directories)."""
    import os
    import tempfile

    if os.environ.get("MPLCONFIGDIR"):
        return
    candidate = Path(tempfile.gettempdir()) / "uavdet-mplconfig"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover - unusual sandbox
        return
    os.environ["MPLCONFIGDIR"] = str(candidate)


def apply_style(font_family: str = "auto", *, chinese: bool = False) -> dict[str, Any]:
    """Activate the bundled SCI report style; returns the effective rcParams."""
    _ensure_mpl_config_dir()
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg", force=True)
    style_path = load_style_path()
    plt.style.use(str(style_path))
    family = choose_font_family(font_family)
    if chinese and family in ("DejaVu Sans", ""):
        # Chinese output requires an installed CJK font; caller verifies the glyphs.
        for candidate in ("Noto Sans CJK SC", "Source Han Sans SC", "Hiragino Sans GB",
                          "Arial Unicode MS", "Microsoft YaHei", "PingFang SC"):
            if candidate in available_fonts():
                family = candidate
                break
    plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = REPORT_COLORS["bg"]
    plt.rcParams["savefig.facecolor"] = REPORT_COLORS["bg"]
    return {
        "style_path": str(style_path),
        "style_version": STYLE_VERSION,
        "font_family": family,
    }


def style_axes(ax, *, xlabel: str | None = None, ylabel: str | None = None,
               title: str | None = None, title_pad: float = 8.0,
               grid: bool = False) -> None:
    """Apply the report look to one axes (thin spines, no top/right frame)."""
    ax.set_facecolor(REPORT_COLORS["bg"])
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=REPORT_COLORS["text"])
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=REPORT_COLORS["text"])
    if title:
        ax.set_title(title, color=REPORT_COLORS["text"], loc="left", pad=title_pad,
                     fontweight="semibold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=REPORT_COLORS["text"], labelsize=8, length=3, width=0.7, pad=4)
    if grid:
        ax.grid(True, color=REPORT_COLORS["grid"], linewidth=0.6, alpha=0.8)
    else:
        ax.grid(False)
    ax.margins(x=0.02, y=0.08)


def add_panel_label(ax, label: str, *, x: float = -0.12, y: float = 1.06) -> None:
    """Place an ``(a)``/``(b)`` panel tag at a consistent position and size."""
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=10,
            color=REPORT_COLORS["text"], fontweight="semibold",
            ha="left", va="bottom")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def figure_output_paths(out_dir: str | Path, basename: str,
                        formats: Sequence[str] = ("png", "pdf", "svg")) -> dict[str, Path]:
    base = Path(out_dir)
    return {fmt: base / f"{basename}.{fmt}" for fmt in formats}


def save_figure(fig, out_dir: str | Path, basename: str,
                formats: Sequence[str] = ("png", "pdf", "svg"),
                dpi: int = 300) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Save PNG/PDF/SVG and return ``(paths, sizes)`` including PNG pixel size.

    A blank/failed render must raise instead of leaving a white PNG behind: the
    caller only records a figure after this function returns.
    """
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    paths = figure_output_paths(out_dir, basename, formats)
    sizes: dict[str, dict[str, Any]] = {}
    for fmt, path in paths.items():
        fig.savefig(path, dpi=dpi, facecolor=REPORT_COLORS["bg"], bbox_inches="tight",
                    pad_inches=0.04)
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"figure export produced an empty file: {path}")
        sizes[str(path)] = {"format": fmt, "bytes": path.stat().st_size}
    plt.close(fig)
    png = paths.get("png")
    if png and png.exists():
        from PIL import Image

        with Image.open(png) as image:
            sizes[str(png)]["width_px"], sizes[str(png)]["height_px"] = image.size
            sizes[str(png)]["dpi"] = float(image.info.get("dpi", (dpi, dpi))[0])
    return [str(p) for p in paths.values()], sizes


def to_percent(values: Any) -> np.ndarray:
    """Convert a 0-1 metric to percent **once** (stored source data stays 0-1)."""
    return np.asarray(values, dtype=float) * 100.0


def write_source_data(out_dir: str | Path, name: str, *,
                      arrays: Mapping[str, Any] | None = None,
                      columns: Mapping[str, Sequence[Any]] | None = None,
                      meta: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Persist the exact numbers a figure was drawn from.

    * ``source_data/<name>_data.npz`` — numeric arrays (curves, matrices, bars).
    * ``source_data/<name>_values.csv`` — the same numbers in human-readable form.
    * ``source_data/<name>_meta.json`` — units, transforms, class order, counts.

    This is what makes the figures traceable back to the run artefacts.
    """
    target = ensure_dir(Path(out_dir) / "source_data")
    written: dict[str, str] = {}
    if arrays:
        npz_path = target / f"{name}_data.npz"
        np.savez_compressed(npz_path, **{key: np.asarray(value) for key, value in arrays.items()})
        written["npz"] = str(npz_path)
    if columns:
        import csv

        csv_path = target / f"{name}_values.csv"
        headers = list(columns.keys())
        lengths = {len(list(values)) for values in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"source columns for {name} have different lengths: {sorted(lengths)}")
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for row in zip(*(list(values) for values in columns.values())):
                writer.writerow([_csv_cell(cell) for cell in row])
        written["csv"] = str(csv_path)
    if meta is not None:
        meta_path = target / f"{name}_meta.json"
        meta_path.write_text(
            json.dumps(dict(meta), indent=2, ensure_ascii=False, allow_nan=False, default=str),
            encoding="utf-8",
        )
        written["meta"] = str(meta_path)
    return written


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.ndarray, list, tuple)):
        return " ".join(_csv_cell(v) for v in value)
    return value


# ---------------------------------------------------------------------------
# figure records
# ---------------------------------------------------------------------------
@dataclass
class FigureRecord:
    """One entry of ``figure_manifest.json`` (traceable figure metadata)."""

    figure_id: str
    title: str
    manuscript_section: str
    run_id: str
    scope: str
    figure_tag: str
    input_paths: list[str] = field(default_factory=list)
    input_sha256: dict[str, str] = field(default_factory=dict)
    variables: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    class_order: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    filters: list[str] = field(default_factory=list)
    transforms: list[str] = field(default_factory=list)
    normalization: list[str] = field(default_factory=list)
    sample_count: dict[str, int] = field(default_factory=dict)
    data_status: str = "ok"
    output_paths: list[str] = field(default_factory=list)
    output_sizes: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_data_paths: list[str] = field(default_factory=list)
    software_versions: dict[str, str] = field(default_factory=dict)
    style_version: str = STYLE_VERSION
    warnings: list[str] = field(default_factory=list)
    scientific_review: str = "pending"
    visual_review: str = "pending"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "title": self.title,
            "manuscript_section": self.manuscript_section,
            "run_id": self.run_id,
            "scope": self.scope,
            "figure_tag": self.figure_tag,
            "input_paths": self.input_paths,
            "input_sha256": self.input_sha256,
            "variables": self.variables,
            "units": self.units,
            "class_order": self.class_order,
            "filters": self.filters,
            "transforms": self.transforms,
            "normalization": self.normalization,
            "sample_count": self.sample_count,
            "data_status": self.data_status,
            "output_paths": self.output_paths,
            "output_sizes": self.output_sizes,
            "source_data_paths": self.source_data_paths,
            "software_versions": self.software_versions,
            "style_version": self.style_version,
            "warnings": self.warnings,
            "scientific_review": self.scientific_review,
            "visual_review": self.visual_review,
            "notes": self.notes,
        }


def missing_record(figure_id: str, title: str, manuscript_section: str, *,
                   run_id: str, scope: str, figure_tag: str,
                   reason: str, required: Iterable[str] = ()) -> dict[str, Any]:
    """Record for a figure that cannot be built yet (never a blank placeholder)."""
    record = FigureRecord(
        figure_id=figure_id,
        title=title,
        manuscript_section=manuscript_section,
        run_id=run_id,
        scope=scope,
        figure_tag=figure_tag,
        data_status="missing",
        warnings=[f"missing input: {p}" for p in required],
        notes=reason,
    )
    payload = record.to_dict()
    payload["missing_inputs"] = list(required)
    payload["reason"] = reason
    return payload


def software_versions() -> dict[str, str]:
    """Versions recorded in every figure manifest entry."""
    versions = {
        "uavdet": __version__,
        "python": platform.python_version(),
        "style": STYLE_VERSION,
    }
    for name in ("matplotlib", "numpy", "pandas", "PIL"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = "unavailable"
    return versions
