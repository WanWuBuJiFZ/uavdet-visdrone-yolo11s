"""Build the report figure set from saved run artefacts only.

``build_figures`` never loads a model, never touches CUDA/Qt and never runs
inference: it reads the audit JSON, ``train/results.csv``, ``evaluation/*`` and
the recorded predictions, calls the figure renderers and writes:

* ``figures/<tag>/*.png``/``*.pdf``/``*.svg``
* ``figures/<tag>/source_data/`` — the exact numbers plotted
* ``figures/<tag>/figure_manifest.json`` — full provenance per figure
* ``figures/<tag>/figure_index.md`` — human index with neutral caption templates

Missing inputs are reported (``data_status="missing"``), never replaced by a
blank figure, and make the run non-``ok``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import (
    file_sha_records,
    now_iso,
    read_json,
    write_json_atomic,
    write_text_atomic,
)
from .plotting import STYLE_VERSION, FigureRecord, missing_record, software_versions
from .plotting import cases as cases_plots
from .plotting import dataset as dataset_plots
from .plotting import evaluation as evaluation_plots
from .plotting import training as training_plots
from .types import RunContext, RunError

__all__ = ["build_figures", "FIGURE_META", "figure_manifest", "read_figure_manifest"]

#: figure id -> (output basename, title, manuscript section)
FIGURE_META: dict[str, tuple[str, str, str]] = {
    "F01": ("class_distribution", "Ground-truth instances per class and split", "Dataset"),
    "F02": ("object_size_distribution", "Relative object area distribution", "Dataset"),
    "F03": ("objects_per_image", "Objects per image (train vs val)", "Dataset"),
    "F04": ("loss_curves", "Training and validation losses", "Training analysis"),
    "F05": ("map_curves", "mAP during training", "Training analysis"),
    "F06": ("precision_recall_epochs", "Precision and recall during training", "Training analysis"),
    "F07": ("per_class_ap", "Per-class AP@0.5 and AP@0.5:0.95", "Results"),
    "F08": ("confusion_matrix", "Confusion matrix (counts and GT-normalised)", "Results"),
    "F09": ("pr_curve", "Precision-recall curves", "Results"),
    "F10": ("f1_confidence", "F1 versus confidence", "Results"),
    "F11": ("successful_cases", "Successful detections at conf=0.25", "Qualitative results"),
    "F12": ("failure_cases", "Missed, false and crowded detections at conf=0.25", "Qualitative results"),
}

#: module -> figure ids it renders
MODULE_FIGURES: dict[str, tuple[str, ...]] = {
    "dataset": ("F01", "F02", "F03"),
    "training": ("F04", "F05", "F06"),
    "evaluation": ("F07", "F08", "F09", "F10"),
    "cases": ("F11", "F12"),
}


@dataclass
class _Module:
    name: str
    render: Callable[[RunContext, Path], list[dict[str, Any]]]


_MODULES: tuple[_Module, ...] = (
    _Module("dataset", dataset_plots.render),
    _Module("training", training_plots.render),
    _Module("evaluation", evaluation_plots.render),
    _Module("cases", cases_plots.render),
)


def figure_manifest(run: RunContext, figure_tag: str = "v1") -> dict[str, Any]:
    path = run.figures_dir(figure_tag) / "figure_manifest.json"
    if not path.exists():
        raise RunError(f"no figure manifest for tag '{figure_tag}' in {run.run_dir}")
    return read_json(path)


def read_figure_manifest(run: RunContext, figure_tag: str = "v1") -> dict[str, Any]:
    return figure_manifest(run, figure_tag)


def _not_applicable_record(figure_id: str, *, run_id: str, scope: str,
                           figure_tag: str, reason: str) -> dict[str, Any]:
    """Record for a figure group this run cannot support (no stage, no data)."""
    payload = FigureRecord(
        figure_id=figure_id,
        title=FIGURE_META[figure_id][1],
        manuscript_section=FIGURE_META[figure_id][2],
        run_id=run_id,
        scope=scope,
        figure_tag=figure_tag,
        data_status="not_applicable",
        notes=reason,
    ).to_dict()
    payload["reason"] = reason
    return payload


def _check_scope_consistency(run: RunContext) -> None:
    """Refuse to build a report when artefacts come from a different scope.

    A smoke run's metrics can never silently become the formal report, and a
    stale ``formal`` metrics.json cannot be attached to a smoke run.
    """
    mismatches: list[str] = []
    stats_path = run.audit_dir / "dataset_stats.json"
    if stats_path.exists():
        audit_scope = read_json(stats_path).get("scope")
        if audit_scope and audit_scope != run.scope:
            mismatches.append(f"audit scope={audit_scope!r}")
    metrics_path = run.evaluation_dir / "metrics.json"
    if metrics_path.exists():
        metrics_scope = read_json(metrics_path).get("scope")
        if metrics_scope and metrics_scope != run.scope:
            mismatches.append(f"evaluation scope={metrics_scope!r}")
    if mismatches:
        raise RunError(
            f"run '{run.run_id}' is scope={run.scope!r} but its saved artefacts disagree "
            f"({', '.join(mismatches)}). Refusing to mix smoke/historical results into a "
            f"report; rebuild the missing stage inside this run."
        )


def _applicable_modules(run: RunContext) -> list[str]:
    """Which figure groups this run can support, based on completed stages."""
    modules = ["dataset"]
    if run.stage_status("train") == "complete" and (run.train_dir / "results.csv").exists():
        modules.append("training")
    if run.stage_status("evaluation") == "complete" and (
        run.evaluation_dir / "metrics.json"
    ).exists():
        modules.extend(["evaluation", "cases"])
    return modules


def build_figures(run: RunContext, figure_tag: str = "v1") -> dict[str, Any]:
    """Render every applicable figure group; return the aggregated status."""
    figure_tag = str(figure_tag)
    if not figure_tag or "/" in figure_tag or "\\" in figure_tag or ".." in figure_tag:
        raise RunError(f"invalid figure tag {figure_tag!r}")
    out_dir = run.figures_dir(figure_tag)
    if (out_dir / "figure_manifest.json").exists():
        raise RunError(
            f"figures/{figure_tag} already has a figure_manifest.json; refusing to "
            f"overwrite it. Use --figure-tag with a new tag (a new style/font change "
            f"also requires a new tag)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_scope_consistency(run)

    applicable = _applicable_modules(run)
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    for module in _MODULES:
        if module.name not in applicable:
            continue
        try:
            produced = module.render(run, out_dir) or []
        except Exception as exc:  # one broken figure must not hide the others
            errors.append(f"{module.name}: {type(exc).__name__}: {exc}")
            produced = [
                missing_record(
                    figure_id,
                    FIGURE_META[figure_id][1],
                    FIGURE_META[figure_id][2],
                    run_id=run.run_id,
                    scope=run.scope,
                    figure_tag=figure_tag,
                    reason=f"renderer failed: {type(exc).__name__}: {exc}",
                )
                for figure_id in MODULE_FIGURES[module.name]
            ]
        records.extend(produced)

    # Figures of groups this run cannot support at all are reported explicitly as
    # not-applicable (an honest "this run has no such stage", not a broken input).
    for module in _MODULES:
        if module.name in applicable:
            continue
        for figure_id in MODULE_FIGURES[module.name]:
            records.append(
                _not_applicable_record(
                    figure_id,
                    run_id=run.run_id,
                    scope=run.scope,
                    figure_tag=figure_tag,
                    reason=(
                        f"this run has no completed "
                        f"{'training' if module.name == 'training' else 'evaluation'} stage"
                    ),
                )
            )

    produced = [rec for rec in records if rec.get("data_status") == "ok"]
    missing = [rec for rec in records if rec.get("data_status") == "missing"]
    not_applicable = [rec for rec in records if rec.get("data_status") == "not_applicable"]
    unseen = sorted(set(FIGURE_META) - {rec["figure_id"] for rec in records})
    for figure_id in unseen:  # defensive: a renderer skipped an id silently
        missing.append(
            missing_record(
                figure_id,
                FIGURE_META[figure_id][1],
                FIGURE_META[figure_id][2],
                run_id=run.run_id,
                scope=run.scope,
                figure_tag=figure_tag,
                reason="renderer returned no record for this figure id",
            )
        )

    if produced and missing:
        status = "partial"
    elif produced:
        status = "ok"
    else:
        status = "failed"

    manifest = {
        "schema": 1,
        "run_id": run.run_id,
        "scope": run.scope,
        "figure_tag": figure_tag,
        "created_at": now_iso(),
        "style_version": STYLE_VERSION,
        "software_versions": software_versions(),
        "applicable_modules": applicable,
        "figures": records,
        "produced": [rec["figure_id"] for rec in produced],
        "missing": [
            {"figure_id": rec["figure_id"], "reason": rec.get("reason", ""),
             "missing_inputs": rec.get("missing_inputs", [])}
            for rec in missing
        ],
        "not_applicable": [
            {"figure_id": rec["figure_id"], "reason": rec.get("reason", "")}
            for rec in not_applicable
        ],
        "errors": errors,
        "status": status,
    }
    write_json_atomic(out_dir / "figure_manifest.json", manifest)
    write_text_atomic(out_dir / "figure_index.md", _render_index(manifest))
    return {
        "status": status,
        "figure_tag": figure_tag,
        "figure_dir": str(out_dir),
        "produced": [rec["figure_id"] for rec in produced],
        "missing": manifest["missing"],
        "errors": errors,
        "figures": records,
    }


def _display_path(path: Any) -> str:
    """Portable, ASCII-safe path for the human-facing index.

    The manifest keeps the full (possibly absolute) provenance path; the index
    only shows run-relative paths or bare file names so a shared report never
    leaks a local user directory (which may contain non-ASCII characters).
    """
    text = str(path)
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate.name
    return text


def _render_index(manifest: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Report figure index — run `{manifest['run_id']}` (scope: {manifest['scope']})")
    lines.append("")
    lines.append(f"Figure tag: `{manifest['figure_tag']}` · style `{manifest['style_version']}` · "
                 f"status: **{manifest['status']}**")
    lines.append("")
    if manifest.get("produced"):
        lines.append("## Produced figures")
        lines.append("")
        by_id = {rec["figure_id"]: rec for rec in manifest["figures"]}
        for figure_id in manifest["produced"]:
            record = by_id[figure_id]
            lines.append(f"### {figure_id} — {record['title']}")
            lines.append("")
            lines.append(f"- Suggested section: {record['manuscript_section']}")
            evidence = [_display_path(item) for item in (record.get("input_paths") or [])]
            lines.append(f"- Evidence: {', '.join(evidence or ['(see manifest)'])}")
            counts = record.get("sample_count") or {}
            if counts:
                counts_text = ", ".join(f"{key}={value}" for key, value in counts.items())
                lines.append(f"- Sample count: {counts_text}")
            if record.get("warnings"):
                lines.append(f"- Warnings: {'; '.join(record['warnings'])}")
            lines.append(f"- Files: {', '.join(Path(p).name for p in record.get('output_paths') or [])}")
            lines.append("")
            lines.append(f"> {record.get('notes') or '(no caption template)'}")
            lines.append("")
    if manifest.get("missing"):
        lines.append("## Missing figures (inputs absent — nothing fabricated)")
        lines.append("")
        for item in manifest["missing"]:
            lines.append(f"- **{item['figure_id']}**: {item['reason']}")
            for path in item.get("missing_inputs") or []:
                lines.append(f"  - missing input: `{path}`")
        lines.append("")
    if manifest.get("not_applicable"):
        lines.append("## Figures not applicable to this run")
        lines.append("")
        for item in manifest["not_applicable"]:
            lines.append(f"- **{item['figure_id']}**: {item['reason']}")
        lines.append("")
    if manifest.get("errors"):
        lines.append("## Renderer errors")
        lines.append("")
        for error in manifest["errors"]:
            lines.append(f"- {error}")
        lines.append("")
    lines.append("## Data honesty notes")
    lines.append("")
    lines.append("- Metrics are stored 0-1 in the run artefacts and multiplied by 100 exactly "
                 "once for display.")
    lines.append("- `source_data/` holds the plotted numbers (npz/csv) and the transform "
                 "metadata (json).")
    lines.append("- Photo panels (F11/F12) are raster content; only text/curves are editable "
                 "vector elements.")
    lines.append("")
    return "\n".join(lines)


def figure_input_hashes(paths: list[str | Path]) -> dict[str, str]:
    """SHA256 for existing inputs, used in each record's provenance block."""
    return file_sha_records(paths)
