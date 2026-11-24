"""实验结果页：选择 run 目录，展示真实阶段状态、指标、图表与缺失产物。

Everything shown here is read from the run directory as written by the CLI:

* stage statuses come from ``manifest.json`` (``pending/running/complete/…``),
* metric numbers come from ``evaluation/metrics.json`` and stay 0-1 in the file
  (``×100`` happens exactly once, in :func:`format_percent`),
* figures are the PNG files under ``figures/<tag>/`` that really exist,
* every expected product that is missing gets an explicit notice — the page
  never shows a placeholder that could be mistaken for a real result.

A ``smoke`` run is badged as smoke (orange) so it can never be read as a formal
experiment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping
from typing import Any

from ...types import STAGE_NAMES
from ..widgets import LazyExports
from ..i18n import tr

__all__ = [
    "ResultsPage",
    "PRODUCT_CHECKS",
    "SCOPE_LABELS",
    "format_percent",
    "format_count",
    "read_run_summary",
    "product_rows",
]

#: ``(stage, run-relative path, human label)`` — the products the GUI checks.
PRODUCT_CHECKS: tuple[tuple[str, str, str], ...] = (
    ("audit", "audit/dataset_stats.json", "数据集统计"),
    ("audit", "audit/data_manifest.json", "数据清单"),
    ("train", "train/results.csv", "逐轮训练 CSV"),
    ("train", "train/weights/best.pt", "best.pt 权重"),
    ("evaluation", "evaluation/metrics.json", "验证指标"),
    ("evaluation", "evaluation/per_class.csv", "逐类指标"),
    ("benchmark", "benchmark/summary.json", "测速摘要"),
)

SCOPE_LABELS = {"formal": "formal 正式", "smoke": "smoke 小样本"}

STAGE_LABELS = {
    "audit": "数据审计",
    "train": "训练",
    "evaluation": "验证评价",
    "benchmark": "测速",
    "figures": "图表",
}


# ---------------------------------------------------------------------------
# pure helpers (importable / testable without Qt)
# ---------------------------------------------------------------------------
def format_percent(value: Any, digits: int = 3) -> str:
    """``0.42992 -> '42.992%'``; ``None``/NaN/unparsable -> ``'N/A'``."""
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number * 100:.{digits}f}%"


def _format_float(value: Any, digits: int = 2) -> str:
    """Compact number for ms/FPS readouts; ``None``/NaN -> ``'N/A'``."""
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def format_count(value: Any) -> str:
    """Integer-ish display for counts; ``None`` -> ``'N/A'``."""
    if value is None:
        return "N/A"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


def _read_json_safe(path: Path) -> tuple[Any, str | None]:
    """Read a JSON file; a *missing* file is not an error (see ``missing``)."""
    if not path.is_file():
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), None
    except (json.JSONDecodeError, OSError) as exc:
        return None, tr('{0} 无法解析：{1}').format(path.name, exc)


def figure_files(run_dir: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return ``(figure_tags, [{tag, name, path}])`` for the saved PNGs."""
    root = Path(run_dir) / "figures"
    if not root.is_dir():
        return [], []
    tags = sorted((child.name for child in root.iterdir() if child.is_dir()))
    records: list[dict[str, str]] = []
    for tag in tags:
        for png in sorted((root / tag).glob("*.png")):
            records.append({"tag": tag, "name": png.name, "path": str(png)})
    return tags, records


def product_rows(run_dir: str | Path, manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """One row per expected product: exists / stage status."""
    base = Path(run_dir)
    stages = dict((manifest or {}).get("stages") or {})
    rows: list[dict[str, Any]] = []
    for stage, rel, label in PRODUCT_CHECKS:
        label = tr(label)
        status = str((stages.get(stage) or {}).get("status") or "pending")
        exists = (base / rel).is_file()
        rows.append({"stage": stage, "rel": rel, "label": label,
                     "status": status, "exists": exists})
    return rows


def read_run_summary(run_dir: str | Path) -> dict[str, Any]:
    """Read one run directory into a plain dict for the page (never raises)."""
    from ...artifacts import run_summary

    base = Path(run_dir)
    summary: dict[str, Any] = {
        "run_dir": base,
        "run_id": base.name,
        "ok": False,
        "scope": "unknown",
        "created_at": None,
        "updated_at": None,
        "stages": [],
        "metrics": {},
        "benchmark": {},
        "figure_tags": [],
        "figures": [],
        "missing": [],
        "notes": [],
        "complete": False,
    }
    if not base.is_dir():
        summary["notes"].append(tr('目录不存在：{0}').format(base))
        return summary

    manifest, error = _read_json_safe(base / "manifest.json")
    if not isinstance(manifest, dict):
        summary["notes"].append(
            tr('缺少 manifest.json：这不是本项目生成的实验目录，无法确认阶段状态')
            + (f"（{error}）" if error else "")
        )
        return summary

    summary["ok"] = True
    summary["manifest"] = manifest
    summary["run_id"] = str(manifest.get("run_id") or base.name)
    summary["scope"] = str(manifest.get("scope") or "unknown")
    summary["created_at"] = manifest.get("created_at")
    summary["updated_at"] = manifest.get("updated_at")

    stage_rows: list[dict[str, Any]] = []
    for stage in STAGE_NAMES:
        block = dict((manifest.get("stages") or {}).get(stage) or {})
        stage_rows.append({
            "name": stage,
            "label": tr(STAGE_LABELS.get(stage, stage)),
            "status": str(block.get("status") or "pending"),
            "attempts": int(block.get("attempts") or 0),
            "finished_at": block.get("finished_at"),
            "error": block.get("error"),
            "summary": dict(block.get("summary") or {}),
        })
    summary["stages"] = stage_rows
    statuses = {row["name"]: row["status"] for row in stage_rows}
    summary["complete"] = all(statuses.get(name) == "complete" for name in STAGE_NAMES)

    metrics, metrics_error = _read_json_safe(base / "evaluation" / "metrics.json")
    if isinstance(metrics, dict):
        summary["metrics"] = metrics
    elif metrics_error:
        summary["notes"].append(f"evaluation/{metrics_error}")

    benchmark, benchmark_error = _read_json_safe(base / "benchmark" / "summary.json")
    if isinstance(benchmark, dict):
        summary["benchmark"] = benchmark
    elif benchmark_error:
        summary["notes"].append(f"benchmark/{benchmark_error}")

    tags, figures = figure_files(base)
    summary["figure_tags"] = tags
    summary["figures"] = figures

    missing: list[str] = []
    for row in product_rows(base, manifest):
        if row["exists"]:
            continue
        missing.append(
            tr('缺失产物（{0} 阶段当前状态 {1}）：{2} — {3}').format(row['stage'], row['status'], row['rel'], row['label'])
        )
    if not figures:
        figure_status = statuses.get("figures", "pending")
        missing.append(
            tr('缺失产物（figures 阶段当前状态 {0}）：figures/<tag>/*.png — 尚未生成图表').format(figure_status)
        )
    # A figure that exists but was honestly reported missing (e.g. F11 had no valid
    # case) must show up here too, otherwise the page says "0 条缺失提示" while the
    # figure manifest says otherwise.
    for tag in tags:
        manifest_block, _error = _read_json_safe(base / "figures" / tag / "figure_manifest.json")
        if not isinstance(manifest_block, dict):
            continue
        for item in manifest_block.get("missing") or []:
            missing.append(
                tr('图表缺项（{0}，status={1}）：{2} — {3}').format(tag, manifest_block.get('status'), item.get('figure_id'), item.get('reason') or 'no reason recorded')
            )
    summary["missing"] = missing

    try:  # the CLI's own aggregation, used only for the completion badge
        aggregate = run_summary(_open_context(base, manifest))
    except Exception as exc:  # noqa: BLE001 - badge is cosmetic, never fatal
        summary["notes"].append(tr('无法汇总 run（{0}: {1}）').format(type(exc).__name__, exc))
    else:
        summary["is_formal_complete"] = bool(aggregate.get("is_formal_complete"))
        summary["aggregate"] = aggregate

    return summary


def _open_context(run_dir: Path, manifest: dict[str, Any]) -> Any:
    """Minimal :class:`RunContext` for :func:`uavdet.artifacts.run_summary`."""
    from ...types import RunContext

    return RunContext(
        run_id=str(manifest.get("run_id") or run_dir.name),
        run_dir=run_dir,
        config={},
        scope=str(manifest.get("scope") or "formal"),
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------
_lazy = LazyExports(__name__, lambda: _build()).register("ResultsPage")


def __getattr__(name: str) -> Any:
    return _lazy.resolve(name)


def _build() -> dict[str, Any]:
    """Import Qt and define :class:`ResultsPage` (called once, lazily)."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QComboBox,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    from .. import widgets

    class ResultsPage(QWidget):
        """Read-only view of one experiment directory."""

        def __init__(self, main_window: Any, parent: QWidget | None = None):
            super().__init__(parent)
            self.main_window = main_window
            self.summary: dict[str, Any] = {}
            self._figure_paths: list[Path] = []
            self.setObjectName("page")
            self._build_ui()
            self.refresh_runs()

        # -- construction ------------------------------------------------------
        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(16, 12, 16, 12)
            root.setSpacing(10)

            toolbar = QWidget()
            toolbar.setObjectName("card")
            bar = QHBoxLayout(toolbar)
            bar.setContentsMargins(12, 10, 12, 10)
            bar.setSpacing(8)

            self.run_combo = QComboBox()
            self.run_combo.setMinimumWidth(220)
            self.run_combo.currentIndexChanged.connect(self._on_combo_changed)
            bar.addWidget(QLabel(tr('实验')))
            bar.addWidget(self.run_combo)

            self.run_path_edit = QLineEdit()
            self.run_path_edit.setPlaceholderText(tr('outputs/<run_id> 目录…'))
            self.run_path_edit.returnPressed.connect(self.load_run)
            bar.addWidget(self.run_path_edit, 1)

            self.browse_button = QPushButton(tr('浏览…'))
            self.browse_button.setProperty("kind", "ghost")
            self.browse_button.clicked.connect(self._choose_dir)
            bar.addWidget(self.browse_button)

            self.refresh_button = QPushButton(tr('刷新'))
            self.refresh_button.setProperty("kind", "primary")
            self.refresh_button.clicked.connect(self._refresh_clicked)
            bar.addWidget(self.refresh_button)

            self.load_button = QPushButton(tr('载入'))
            self.load_button.setProperty("kind", "primary")
            self.load_button.clicked.connect(self.load_run)
            bar.addWidget(self.load_button)
            root.addWidget(toolbar)

            content = QWidget()
            content_layout = QHBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(10)

            left = QWidget()
            left.setObjectName("card")
            left_layout = QVBoxLayout(left)
            left_layout.setContentsMargins(12, 10, 12, 10)
            left_layout.setSpacing(8)

            header_row = QHBoxLayout()
            header_row.setSpacing(8)
            self.run_id_label = QLabel(tr('未选择实验'))
            self.run_id_label.setObjectName("metric_value")
            header_row.addWidget(self.run_id_label)
            self.scope_badge = QLabel("scope ?")
            self.scope_badge.setProperty("badge", "unknown")
            header_row.addWidget(self.scope_badge)
            self.state_badge = QLabel(tr('状态未知'))
            self.state_badge.setProperty("badge", "unknown")
            header_row.addWidget(self.state_badge)
            header_row.addStretch()
            left_layout.addLayout(header_row)

            self.timestamp_label = QLabel("")
            self.timestamp_label.setObjectName("hint_line")
            left_layout.addWidget(self.timestamp_label)

            stages_title = QLabel(tr('阶段状态（来自 manifest.json）'))
            stages_title.setObjectName("card_title")
            left_layout.addWidget(stages_title)
            self.stage_table = QTableWidget(len(STAGE_NAMES), 5)
            self.stage_table.setHorizontalHeaderLabels(
                [tr('阶段'), tr('状态'), tr('尝试'), tr('结束时间'), tr('错误')])
            self.stage_table.verticalHeader().setVisible(False)
            self.stage_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.stage_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
            self.stage_table.setAlternatingRowColors(True)
            self.stage_table.setShowGrid(False)
            for column in range(5):
                item = self.stage_table.horizontalHeaderItem(column)
                if item is not None:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            stage_header = self.stage_table.horizontalHeader()
            stage_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            stage_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            stage_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.stage_table.setMinimumHeight(170)
            left_layout.addWidget(self.stage_table)

            metrics_title = QLabel(tr('验证指标（metrics.json 原值 ×100，仅显示层换算）'))
            metrics_title.setObjectName("card_title")
            left_layout.addWidget(metrics_title)

            metrics_card = QWidget()
            metrics_card.setObjectName("card")
            grid = QGridLayout(metrics_card)
            grid.setContentsMargins(8, 4, 8, 4)
            grid.setHorizontalSpacing(18)
            grid.setVerticalSpacing(2)
            self.metric_labels: dict[str, QLabel] = {}
            entries = (
                ("map50", "mAP50"), ("map50_95", "mAP50-95"),
                ("precision", "Precision"), ("recall", "Recall"),
            )
            for index, (key, label) in enumerate(entries):
                value_label = QLabel("N/A")
                value_label.setObjectName("metric_value")
                caption = QLabel(label)
                caption.setObjectName("metric_label")
                grid.addWidget(value_label, 0, index * 2)
                grid.addWidget(caption, 1, index * 2)
                self.metric_labels[key] = value_label
            left_layout.addWidget(metrics_card)

            self.extra_metrics_label = QLabel("")
            self.extra_metrics_label.setObjectName("hint_line")
            self.extra_metrics_label.setWordWrap(True)
            left_layout.addWidget(self.extra_metrics_label)

            self.notice_label = QLabel("")
            self.notice_label.setObjectName("warning_line")
            self.notice_label.setWordWrap(True)
            self.notice_label.setVisible(False)
            left_layout.addWidget(self.notice_label)

            self.ok_label = QLabel(tr('产物齐全：本 run 的审计 / 训练 / 评价 / 测速产物都存在'))
            self.ok_label.setObjectName("ok_line")
            self.ok_label.setWordWrap(True)
            self.ok_label.setVisible(False)
            left_layout.addWidget(self.ok_label)
            left_layout.addStretch()
            content_layout.addWidget(left, 3)

            right = QWidget()
            right.setObjectName("card")
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(12, 10, 12, 10)
            right_layout.setSpacing(8)
            figures_title = QLabel(tr('已保存图表（figures/<tag>/*.png）'))
            figures_title.setObjectName("card_title")
            right_layout.addWidget(figures_title)

            self.figure_scroll = QScrollArea()
            self.figure_scroll.setWidgetResizable(True)
            self.figure_container = QWidget()
            self.figure_layout = QVBoxLayout(self.figure_container)
            self.figure_layout.setContentsMargins(4, 4, 4, 4)
            self.figure_layout.setSpacing(8)
            self.figure_layout.addStretch()
            self.figure_scroll.setWidget(self.figure_container)
            right_layout.addWidget(self.figure_scroll, 1)
            self.figure_summary = QLabel(tr('尚未载入实验'))
            self.figure_summary.setObjectName("hint_line")
            self.figure_summary.setWordWrap(True)
            right_layout.addWidget(self.figure_summary)
            content_layout.addWidget(right, 2)

            root.addWidget(content, 1)

            self.status = QLabel(tr('选择 outputs/<run_id> 目录后点击【载入】'))
            self.status.setObjectName("status_line")
            self.status.setWordWrap(True)
            root.addWidget(self.status)

        # -- helpers -----------------------------------------------------------
        def _set_status(self, text: str, error: bool = False) -> None:
            self.status.setText(str(text))
            self.status.setProperty("error", "true" if error else "false")
            style = self.status.style()
            style.unpolish(self.status)
            style.polish(self.status)
            self.main_window.set_status(str(text), error=error)

        def _set_badge(self, label: QLabel, text: str, kind: str) -> None:
            label.setText(text)
            label.setProperty("badge", kind)
            style = label.style()
            style.unpolish(label)
            style.polish(label)

        def output_root(self) -> Path:
            config = getattr(self.main_window, "config", {}) or {}
            root = (config.get("project") or {}).get("output_root")
            return Path(root) if root else Path.cwd() / "outputs"

        def refresh_runs(self) -> list[Path]:
            """Rescan ``outputs/*`` and fill the run combo (no run is loaded)."""
            from ...artifacts import list_runs

            self.run_combo.blockSignals(True)
            self.run_combo.clear()
            self.run_combo.addItem(tr('（选择实验目录）'), "")
            directories: list[Path] = []
            try:
                for record in list_runs(self.output_root()):
                    run_dir = Path(str(record.get("run_dir")))
                    directories.append(run_dir)
                    stages = record.get("stages") or {}
                    done = sum(1 for value in stages.values() if value == "complete")
                    scope = record.get("scope") or "?"
                    self.run_combo.addItem(
                        tr('{0} · {1} · {2}/{3} 阶段完成').format(record.get('run_id'), scope, done, len(stages) or len(STAGE_NAMES)),
                        str(run_dir),
                    )
            except OSError as exc:
                self._set_status(tr('无法读取输出目录 {0}: {1}').format(self.output_root(), exc), error=True)
            self.run_combo.blockSignals(False)
            if not directories:
                self._set_status(tr('{0} 下还没有实验目录（先运行 CLI 训练/验证）').format(self.output_root()))
            return directories

        def _refresh_clicked(self) -> None:
            directories = self.refresh_runs()
            self._set_status(tr('已刷新：发现 {0} 个实验目录').format(len(directories)))

        def _on_combo_changed(self, index: int) -> None:
            data = self.run_combo.itemData(index)
            if data:
                self.run_path_edit.setText(str(data))
                self.load_run(str(data))

        def _choose_dir(self) -> None:
            start = self.run_path_edit.text().strip() or str(self.output_root())
            path = QFileDialog.getExistingDirectory(self, tr('选择实验目录'), start)
            if path:
                self.run_path_edit.setText(path)
                self.load_run(path)

        # -- loading -----------------------------------------------------------
        def capture_state(self) -> dict[str, Any]:
            return {"run_dir": self.run_path_edit.text(), "loaded": bool(self.summary)}

        def restore_state(self, state: Mapping[str, Any]) -> None:
            run_dir = str(state.get("run_dir") or "").strip()
            if not run_dir:
                return
            self.run_path_edit.setText(run_dir)
            if state.get("loaded"):
                self.load_run(run_dir)

        def load_run(self, run_dir: str | Path | None = None) -> dict[str, Any]:
            """Read ``run_dir`` (default: the text field) and update every widget."""
            target = run_dir if run_dir is not None else self.run_path_edit.text().strip()
            if not target:
                self._set_status(tr('请先选择或填写实验目录'), error=True)
                return {}
            base = Path(str(target)).expanduser()
            self.run_path_edit.setText(str(base))
            summary = read_run_summary(base)
            self.summary = summary
            self._apply_summary(summary)
            return summary

        def _apply_summary(self, summary: dict[str, Any]) -> None:
            if not summary.get("ok"):
                self.run_id_label.setText(tr('无法读取实验'))
                self._set_badge(self.scope_badge, "scope ?", "unknown")
                self._set_badge(self.state_badge, tr('状态未知'), "unknown")
                self.timestamp_label.setText("")
                self.stage_table.clearContents()
                for label in self.metric_labels.values():
                    label.setText("N/A")
                self.extra_metrics_label.setText("")
                self._clear_figures()
                notices = list(summary.get("notes") or []) + list(summary.get("missing") or [])
                self._show_notices(notices)
                self.ok_label.setVisible(False)
                self.figure_summary.setText(tr('未载入图表'))
                message = notices[0] if notices else tr('无法读取实验目录')
                self._set_status(message, error=True)
                return

            scope = summary.get("scope", "unknown")
            self.run_id_label.setText(str(summary.get("run_id")))
            self._set_badge(self.scope_badge, tr(SCOPE_LABELS.get(scope, f"scope {scope}")),
                            scope if scope in SCOPE_LABELS else "unknown")
            if summary.get("complete"):
                self._set_badge(self.state_badge, tr('五阶段完成'), "complete")
            else:
                self._set_badge(self.state_badge, tr('存在未完成阶段'), "incomplete")
            self.timestamp_label.setText(
                tr('创建：{0}    更新：{1}').format(summary.get('created_at') or 'N/A', summary.get('updated_at') or 'N/A')
            )

            rows = list(summary.get("stages") or [])
            self.stage_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                values = (row.get("label", row.get("name")), row.get("status"),
                          str(row.get("attempts", 0)), row.get("finished_at") or "—",
                          row.get("error") or "")
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                    status = str(row.get("status"))
                    if column == 1:
                        if status == "complete":
                            item.setForeground(Qt.GlobalColor.green)
                        elif status in ("failed", "interrupted"):
                            item.setForeground(Qt.GlobalColor.red)
                        elif status == "skipped":
                            item.setForeground(Qt.GlobalColor.yellow)
                    self.stage_table.setItem(row_index, column, item)

            metrics = dict(summary.get("metrics") or {})
            for key in ("map50", "map50_95", "precision", "recall"):
                self.metric_labels[key].setText(format_percent(metrics.get(key)))
            extra_bits = []
            if metrics:
                extra_bits.append(tr('评价划分：{0}').format(metrics.get('split') or 'N/A'))
                extra_bits.append(tr('图片数：{0}').format(format_count(metrics.get('num_images'))))
                extra_bits.append(tr('实例数：{0}').format(format_count(metrics.get('num_instances'))))
                extra_bits.append(tr('评价器：{0} {1}').format(metrics.get('evaluator_name') or 'N/A', metrics.get('evaluator_version') or '').strip())
                extra_bits.append(tr('状态：{0}').format(metrics.get('status') or 'N/A'))
                extra_bits.append(f"imgsz {metrics.get('imgsz', 'N/A')} · "
                                  f"conf {metrics.get('conf', 'N/A')} · "
                                  f"iou {metrics.get('iou_nms', 'N/A')} · "
                                  f"max_det {metrics.get('max_det', 'N/A')}")
                warnings = metrics.get("warnings") or []
                if warnings:
                    extra_bits.append("warnings：" + "；".join(str(w) for w in warnings))
            else:
                extra_bits.append(tr('该 run 没有 evaluation/metrics.json，指标显示 N/A（不是 0）'))
            benchmark = dict(summary.get("benchmark") or {})
            if benchmark:
                extra_bits.append(
                    tr('测速：mean {0} ms · p95 {1} ms · FPS {2} · device {3}').format(_format_float(benchmark.get('mean_ms')), _format_float(benchmark.get('p95_ms')), _format_float(benchmark.get('fps')), benchmark.get('effective_device', 'N/A'))
                )
            self.extra_metrics_label.setText("    ".join(extra_bits))

            notices = list(summary.get("notes") or []) + list(summary.get("missing") or [])
            self._show_notices(notices)
            self._populate_figures(summary)
            tags = summary.get("figure_tags") or []
            figures = summary.get("figures") or []
            if figures:
                self.figure_summary.setText(
                    tr('共 {0} 张 PNG，标签：{1}').format(len(figures), ', '.join(tags) or tr('无')))
                self._set_status(tr('已载入 {0}：{1} 张图表，{2} 条缺失提示').format(summary.get('run_id'), len(figures), len(notices)))
            else:
                self.figure_summary.setText(tr('没有已保存的 PNG 图表'))
                self._set_status(tr('已载入 {0}：图表尚未生成').format(summary.get('run_id'))
                                 + (tr('（见下方缺失提示）') if notices else ""))

        def _show_notices(self, notices: list[str]) -> None:
            text = "\n".join(f"• {item}" for item in notices)
            self.notice_label.setText(text)
            self.notice_label.setVisible(bool(notices))
            self.ok_label.setVisible(not notices)

        # -- figures -----------------------------------------------------------
        def _clear_figures(self) -> None:
            self._figure_paths = []
            while self.figure_layout.count() > 1:
                item = self.figure_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        def _populate_figures(self, summary: dict[str, Any]) -> None:
            self._clear_figures()
            for record in summary.get("figures") or []:
                path = Path(str(record.get("path")))
                pixmap = widgets.pixmap_from_file(path)
                if pixmap is None:
                    continue
                card = QWidget()
                card.setObjectName("figure_card")
                layout = QVBoxLayout(card)
                layout.setContentsMargins(6, 6, 6, 4)
                layout.setSpacing(4)
                image_label = QLabel()
                width = max(220, self.figure_scroll.width() - 40)
                scaled = pixmap.scaledToWidth(min(width, 520),
                                              Qt.TransformationMode.SmoothTransformation)
                image_label.setPixmap(scaled)
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(image_label)
                caption = QLabel(f"{record.get('tag')} / {record.get('name')}")
                caption.setObjectName("figure_caption")
                layout.addWidget(caption)
                self.figure_layout.insertWidget(self.figure_layout.count() - 1, card)
                self._figure_paths.append(path)

        # -- test / automation surface -----------------------------------------
        def missing_notices(self) -> list[str]:
            """All explicit "missing product" notices of the loaded run."""
            summary = self.summary or {}
            return list(summary.get("notes") or []) + list(summary.get("missing") or [])

        def figure_paths(self) -> list[Path]:
            return list(self._figure_paths)

        def notice_text(self) -> str:
            return self.notice_label.text()

        # -- lifecycle ----------------------------------------------------------
        def on_busy_changed(self, busy: bool) -> None:
            for widget in (self.refresh_button, self.load_button, self.browse_button):
                widget.setEnabled(True)

        def on_shutdown(self) -> None:
            """Read-only page: nothing to release."""
            return None

    return {"ResultsPage": ResultsPage}
