"""Report figure renderers.

Each module exposes::

    render(run, out_dir) -> list[dict]

where every dict is either a :class:`~uavdet.plotting.style.FigureRecord`
payload (``data_status="ok"``) for a figure that was really written, or a
``data_status="missing"`` record explaining which input is absent. A renderer
never fabricates values and never writes a blank PNG as a success.

Modules:

* :mod:`uavdet.plotting.dataset`    — F01 class distribution, F02 object size, F03 objects/image
* :mod:`uavdet.plotting.training`   — F04 loss curves, F05 mAP curves, F06 P/R per epoch
* :mod:`uavdet.plotting.evaluation` — F07 per-class AP, F08 confusion matrix, F09 PR, F10 F1-conf
* :mod:`uavdet.plotting.cases`      — F11 successful cases, F12 failure cases

:mod:`uavdet.report` calls the renderers and writes the combined manifest.
"""

from __future__ import annotations

from .style import (  # noqa: F401  (re-exported for renderers and tests)
    CLASS_COLORS,
    LOSS_COLORS,
    METRIC_COLORS,
    REPORT_COLORS,
    SPLIT_COLORS,
    STYLE_VERSION,
    FigureRecord,
    add_panel_label,
    apply_style,
    choose_font_family,
    ensure_dir,
    figure_output_paths,
    load_style_path,
    missing_record,
    save_figure,
    software_versions,
    style_axes,
    to_percent,
    write_source_data,
)

__all__ = [
    "STYLE_VERSION",
    "FigureRecord",
    "apply_style",
    "save_figure",
    "style_axes",
    "missing_record",
    "write_source_data",
    "software_versions",
    "to_percent",
    "add_panel_label",
    "ensure_dir",
]
