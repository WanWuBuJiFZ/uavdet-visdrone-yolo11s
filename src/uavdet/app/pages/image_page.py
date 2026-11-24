"""图片检测页：选择/预览单张图片，调用工作线程检测，统计并导出结果。

The page never touches the detector itself: it asks the main window for the
loaded :class:`~uavdet.app.workers.DetectorSession`, hands the BGR array to an
:class:`~uavdet.app.workers.ImageTask` and only updates widgets from the
queued signals (so the GUI thread stays responsive).

Aspect-fit preview comes from :class:`uavdet.app.widgets.ImageView`; the image
is shown un-annotated before detection and annotated afterwards. Counts are
labelled "当前图片目标数" — never "unique tracked objects".
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping
from typing import Any

from ...types import UAVDetError
from ..widgets import LazyExports
from ..i18n import tr

__all__ = ["ImagePage"]

_lazy = LazyExports(__name__, lambda: _build()).register("ImagePage")


def __getattr__(name: str) -> Any:
    return _lazy.resolve(name)


def _build() -> dict[str, Any]:
    """Import Qt and define :class:`ImagePage` (called once, lazily)."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    from .. import widgets
    from ..workers import ImageTask

    class ImagePage(QWidget):
        """One image in, detections table + per-class counts + exports out."""

        def __init__(self, main_window: Any, parent: QWidget | None = None):
            super().__init__(parent)
            self.main_window = main_window
            self.current_image: Any = None          # BGR uint8 ndarray
            self.current_image_path: Path | None = None
            self.result: Any = None                 # PredictionResult
            self.annotated: Any = None              # annotated BGR ndarray
            self.image_task: Any = None             # last ImageTask (kept alive)
            self._last_dir = Path.cwd()
            self.setObjectName("page")
            self._build_ui()
            self.on_busy_changed(False)

        # -- construction ------------------------------------------------------
        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(16, 12, 16, 12)
            root.setSpacing(10)

            toolbar = QWidget()
            toolbar.setObjectName("card")
            bar = QHBoxLayout(toolbar)
            bar.setContentsMargins(12, 10, 12, 10)
            bar.setSpacing(9)

            self.open_button = self._button(tr('打开图片'), "primary", self._choose_image)
            self.run_button = self._button(tr('开始检测'), "primary", self.start_detection)
            self.export_png_button = self._button(tr('导出标注图'), "ghost", self._export_png)
            self.export_csv_button = self._button(tr('导出 CSV'), "ghost", self._export_csv)
            self.clear_button = self._button(tr('清除'), "ghost", self.clear)
            for button in (self.open_button, self.run_button, self.export_png_button,
                           self.export_csv_button, self.clear_button):
                bar.addWidget(button)
            bar.addStretch()
            self.image_info = QLabel(tr('未选择图片'))
            self.image_info.setObjectName("hint_line")
            bar.addWidget(self.image_info)
            root.addWidget(toolbar)

            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.setHandleWidth(6)

            left = QWidget()
            left_layout = QVBoxLayout(left)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(8)
            self.view = widgets.ImageView(placeholder=tr('尚未选择图片'))
            left_layout.addWidget(self.view, 1)
            splitter.addWidget(left)

            right = QWidget()
            right.setObjectName("card")
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(12, 10, 12, 10)
            right_layout.setSpacing(8)

            self.total_label = QLabel(tr('当前图片目标数：0'))
            self.total_label.setObjectName("total_label")
            self.total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_layout.addWidget(self.total_label)

            counts_title = QLabel(tr('分类计数（当前图片）'))
            counts_title.setObjectName("card_title")
            right_layout.addWidget(counts_title)
            self.class_table = widgets.ClassCountsTable()
            # ten fixed rows: reserve their height so no row hides behind a scrollbar
            self.class_table.setMinimumHeight(300)
            self.class_table.setSizePolicy(QSizePolicy.Policy.Expanding,
                                           QSizePolicy.Policy.Preferred)
            right_layout.addWidget(self.class_table, 0)

            detections_title = QLabel(tr('检测结果明细（按置信度降序）'))
            detections_title.setObjectName("card_title")
            right_layout.addWidget(detections_title)
            self.detection_table = widgets.DetectionTable()
            right_layout.addWidget(self.detection_table, 1)
            splitter.addWidget(right)

            splitter.setSizes([620, 520])
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)
            root.addWidget(splitter, 1)

            self.status = QLabel(tr('请点击【打开图片】选择一张图片'))
            self.status.setObjectName("status_line")
            self.status.setWordWrap(True)
            root.addWidget(self.status)

        @staticmethod
        def _button(text: str, kind: str, slot: Any) -> QPushButton:
            button = QPushButton(text)
            button.setProperty("kind", kind)
            button.clicked.connect(slot)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            return button

        # -- state helpers -----------------------------------------------------
        def _set_status(self, text: str, error: bool = False) -> None:
            self.status.setText(str(text))
            self.status.setProperty("error", "true" if error else "false")
            style = self.status.style()
            style.unpolish(self.status)
            style.polish(self.status)
            self.main_window.set_status(str(text), error=error)

        def _set_image_info(self) -> None:
            if self.current_image is None:
                self.image_info.setText(tr('未选择图片'))
                return
            height, width = self.current_image.shape[:2]
            name = self.current_image_path.name if self.current_image_path else tr('内存图像')
            self.image_info.setText(f"{name} · {width}×{height}")

        def _reset_results(self) -> None:
            self.result = None
            self.annotated = None
            self.total_label.setText(tr('当前图片目标数：0'))
            self.class_table.set_counts({})
            self.detection_table.set_detections([])
            self._update_export_state()

        def _update_export_state(self) -> None:
            busy = self.main_window.is_busy()
            has_result = self.result is not None
            self.export_png_button.setEnabled(has_result and not busy)
            self.export_csv_button.setEnabled(has_result and not busy)
            self.clear_button.setEnabled(not busy)
            self.run_button.setEnabled(bool(self.current_image is not None) and not busy)
            self.open_button.setEnabled(not busy)

        # -- loading -----------------------------------------------------------
        def _choose_image(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, tr('选择要检测的图片'), str(self._last_dir),
                tr('图片 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;所有文件 (*)'),
            )
            if path:
                self.load_image(path)

        def load_image(self, path: str | Path) -> bool:
            """Load and preview ``path``; returns ``False`` on a decode error."""
            target = Path(path)
            try:
                image = widgets.read_image_bgr(target)
            except UAVDetError as exc:
                self._set_status(str(exc), error=True)
                return False
            self.current_image = image
            self.current_image_path = target
            self._last_dir = target.parent if target.parent.exists() else self._last_dir
            self._reset_results()
            self.view.set_image_bgr(image)
            self._set_image_info()
            height, width = image.shape[:2]
            self._set_status(tr('已载入 {0}（{1}×{2}），点击【开始检测】运行模型').format(target.name, width, height))
            self._update_export_state()
            return True

        def capture_state(self) -> dict[str, Any]:
            """Remember what a language switch must keep (image + last result)."""
            return {"path": str(self.current_image_path) if self.current_image_path else None,
                    "result": self.result, "annotated": self.annotated}

        def restore_state(self, state: Mapping[str, Any]) -> None:
            path = state.get("path")
            if path and Path(path).is_file():
                self.load_image(path)
                if state.get("result") is not None:
                    self._on_result(state.get("result"), state.get("annotated"))

        def clear(self) -> None:
            if self.main_window.is_busy():
                self._set_status(tr('当前有任务在运行，无法清除'), error=True)
                return
            self.current_image = None
            self.current_image_path = None
            self._reset_results()
            self.view.clear()
            self._set_image_info()
            self._set_status(tr('已清除，请重新选择图片'))
            self._update_export_state()

        # -- detection ---------------------------------------------------------
        def start_detection(self) -> bool:
            """Start one image task; returns ``False`` when it cannot start."""
            if self.main_window.is_busy():
                self._set_status(tr('当前已有任务在运行，请等待结束或先停止'), error=True)
                return False
            if self.current_image is None:
                self._set_status(tr('请先打开一张图片'), error=True)
                return False
            session = self.main_window.session
            if session is None or not session.loaded:
                self._set_status(tr('尚未加载模型：请先在【模型与参数】页加载 10 类权重'), error=True)
                return False

            options = self.main_window.inference_options()
            stem = self.current_image_path.stem if self.current_image_path else "image"
            task = ImageTask(
                session, self.current_image, options,
                image_id=stem,
                source=str(self.current_image_path or tr('内存图像')),
            )
            if not self.main_window.acquire_task(task, tr('检测中：{0}').format(stem)):
                return False
            task.succeeded.connect(self._on_result)
            task.failed.connect(self._on_failure)
            task.finished.connect(self._on_finished)
            self.image_task = task
            self._update_export_state()
            self._set_status(tr('检测中：{0} …').format(stem))
            task.start()
            return True

        def _on_result(self, result: Any, annotated: Any) -> None:
            self.result = result
            self.annotated = annotated
            counts = result.class_counts()
            self.class_table.set_counts(counts)
            self.detection_table.set_detections(result.sorted_detections())
            self.total_label.setText(tr('当前图片目标数：{0}').format(int(result.count)))
            if annotated is not None:
                self.view.set_image_bgr(annotated)
            timing = dict(getattr(result, "timing", None) or {})
            extra = ""
            if "total_ms" in timing:
                try:
                    extra = tr(' · 耗时 {0} ms').format(format(float(timing['total_ms']), tr('f" · 耗时 {float(timing[\'total_ms\']):.1f} ms')))
                except (TypeError, ValueError):
                    extra = ""
            self._set_status(tr('检测完成：{0} 个目标{1}').format(int(result.count), extra))
            self._update_export_state()

        def _on_failure(self, message: str) -> None:
            self._set_status(tr('检测失败：{0}').format(message), error=True)
            self._update_export_state()

        def _on_finished(self) -> None:
            self._update_export_state()

        # -- exports -----------------------------------------------------------
        def _default_export_dir(self) -> Path:
            root = getattr(self.main_window, "app_data_root", None) or Path.cwd()
            return Path(root) / "exports"

        def _export_png(self) -> None:
            if self.annotated is None:
                self._set_status(tr('没有可导出的检测结果，请先执行检测'), error=True)
                return
            stem = self.current_image_path.stem if self.current_image_path else "image"
            default = self._default_export_dir() / f"{stem}_detected.png"
            path, _ = QFileDialog.getSaveFileName(
                self, tr('导出标注图'), str(default), tr('PNG 图片 (*.png);;JPEG 图片 (*.jpg)'))
            if path:
                try:
                    target = self.save_annotated_image(path)
                except UAVDetError as exc:
                    self._set_status(str(exc), error=True)
                    return
                self._set_status(tr('已导出标注图：{0}').format(target))

        def _export_csv(self) -> None:
            if self.result is None:
                self._set_status(tr('没有可导出的检测结果，请先执行检测'), error=True)
                return
            stem = self.current_image_path.stem if self.current_image_path else "image"
            default = self._default_export_dir() / f"{stem}_detections.csv"
            path, _ = QFileDialog.getSaveFileName(
                self, tr('导出检测结果 CSV'), str(default), tr('CSV (*.csv);;所有文件 (*)'))
            if path:
                target = self.save_detections_csv(path)
                self._set_status(tr('已导出 CSV：{0}').format(target))

        def save_annotated_image(self, path: str | Path) -> Path:
            """Write the annotated image (PNG/JPEG by suffix)."""
            if self.annotated is None:
                raise UAVDetError(tr('没有可导出的标注图，请先执行检测'))
            return widgets.write_image_bgr(path, self.annotated)

        def save_detections_csv(self, path: str | Path) -> Path:
            """Write one row per detection (original-image pixel boxes)."""
            if self.result is None:
                raise UAVDetError(tr('没有可导出的检测结果，请先执行检测'))
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            rows = self.result.sorted_detections()
            with open(target, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image_id", "index", "class_id", "class_name",
                                 "confidence", "x1", "y1", "x2", "y2"])
                for index, detection in enumerate(rows, start=1):
                    x1, y1, x2, y2 = detection.xyxy
                    writer.writerow([
                        self.result.image_id, index, detection.class_id,
                        detection.class_name, f"{detection.confidence:.6f}",
                        f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                    ])
            return target

        # -- lifecycle ---------------------------------------------------------
        def on_busy_changed(self, busy: bool) -> None:
            self._update_export_state()

        def on_shutdown(self) -> None:
            task = self.image_task
            if task is not None and task.isRunning():
                stop = getattr(task, "request_stop", None)
                if callable(stop):
                    stop()
                task.wait(4000)

    return {"ImagePage": ImagePage}
