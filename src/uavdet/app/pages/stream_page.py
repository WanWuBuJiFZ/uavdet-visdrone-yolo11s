"""视频与摄像头页：选择输入源，开始/停止检测，显示当前帧与当前帧统计。

Stop is driven by a :class:`threading.Event` inside
:class:`~uavdet.app.workers.StreamTask`, so it works even while the worker is
blocked in ``capture.read()``. The capture is released in a ``finally`` block on
normal end, on error and on stop; this page only reacts to signals.

Statistics are labelled "当前帧目标数" on purpose: they count the objects in the
frame on screen, not unique objects tracked over the whole video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from typing import Any

from ...types import UAVDetError
from ..widgets import LazyExports
from ..i18n import tr

__all__ = ["StreamPage"]

_lazy = LazyExports(__name__, lambda: _build()).register("StreamPage")


def __getattr__(name: str) -> Any:
    return _lazy.resolve(name)


def _build() -> dict[str, Any]:
    """Import Qt and define :class:`StreamPage` (called once, lazily)."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QRadioButton,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    from .. import widgets
    from ..workers import StreamTask

    class StreamPage(QWidget):
        """Video file or camera index in, annotated frames and counts out."""

        def __init__(self, main_window: Any, parent: QWidget | None = None):
            super().__init__(parent)
            self.main_window = main_window
            self.current_frame: Any = None        # annotated BGR ndarray
            self.last_result: Any = None          # PredictionResult of that frame
            self.frame_count = 0
            self.stream_task: Any = None
            #: injectable ``capture_factory(target) -> capture`` (tests use a fake)
            self.capture_factory: Any = None
            self._running = False
            self._last_error: str | None = None
            self._last_dir = Path.cwd()
            self.setObjectName("page")
            self._build_ui()
            self._sync_buttons()

        # -- construction ------------------------------------------------------
        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(16, 12, 16, 12)
            root.setSpacing(10)

            toolbar = QWidget()
            toolbar.setObjectName("card")
            outer = QVBoxLayout(toolbar)
            outer.setContentsMargins(12, 10, 12, 10)
            outer.setSpacing(8)

            mode_row = QHBoxLayout()
            mode_row.setSpacing(10)
            self.video_radio = QRadioButton(tr('视频文件'))
            self.video_radio.setChecked(True)
            self.camera_radio = QRadioButton(tr('摄像头'))
            mode_row.addWidget(self.video_radio)
            mode_row.addWidget(self.camera_radio)
            mode_row.addSpacing(8)

            self.path_edit = QLineEdit()
            self.path_edit.setPlaceholderText(tr('选择 mp4 / avi / mov 视频文件…'))
            self.path_edit.returnPressed.connect(self.start_stream)
            mode_row.addWidget(self.path_edit, 1)
            self.browse_button = QPushButton(tr('浏览…'))
            self.browse_button.setProperty("kind", "ghost")
            self.browse_button.clicked.connect(self._choose_video)
            mode_row.addWidget(self.browse_button)

            self.index_label = QLabel(tr('摄像头序号'))
            self.index_label.setObjectName("field_label")
            mode_row.addWidget(self.index_label)
            self.index_spin = QSpinBox()
            self.index_spin.setRange(0, 16)
            self.index_spin.setValue(0)
            mode_row.addWidget(self.index_spin)
            outer.addLayout(mode_row)

            action_row = QHBoxLayout()
            action_row.setSpacing(9)
            self.start_button = QPushButton(tr('开始'))
            self.start_button.setProperty("kind", "primary")
            self.start_button.clicked.connect(self.start_stream)
            self.stop_button = QPushButton(tr('停止'))
            self.stop_button.setProperty("kind", "danger")
            self.stop_button.clicked.connect(self.stop_stream)
            self.save_button = QPushButton(tr('保存当前帧'))
            self.save_button.setProperty("kind", "ghost")
            self.save_button.clicked.connect(self._save_frame_dialog)
            for button in (self.start_button, self.stop_button, self.save_button):
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                action_row.addWidget(button)
            action_row.addStretch()
            self.input_label = QLabel(tr('输入：未选择'))
            self.input_label.setObjectName("hint_line")
            action_row.addWidget(self.input_label)
            outer.addLayout(action_row)
            root.addWidget(toolbar)

            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.setHandleWidth(6)

            self.view = widgets.ImageView(placeholder=tr('尚未开始视频 / 摄像头检测'))
            splitter.addWidget(self.view)

            right = QWidget()
            right.setObjectName("card")
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(12, 10, 12, 10)
            right_layout.setSpacing(8)

            self.total_label = QLabel(tr('当前帧目标数：0'))
            self.total_label.setObjectName("total_label")
            self.total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_layout.addWidget(self.total_label)

            self.frame_label = QLabel(tr('已接收帧：0'))
            self.frame_label.setObjectName("hint_line")
            self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_layout.addWidget(self.frame_label)

            counts_title = QLabel(tr('分类计数（当前帧）'))
            counts_title.setObjectName("card_title")
            right_layout.addWidget(counts_title)
            self.class_table = widgets.ClassCountsTable()
            self.class_table.setSizePolicy(QSizePolicy.Policy.Expanding,
                                           QSizePolicy.Policy.Expanding)
            right_layout.addWidget(self.class_table, 1)
            splitter.addWidget(right)

            splitter.setSizes([640, 420])
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)
            root.addWidget(splitter, 1)

            self.status = QLabel(tr('选择视频文件或摄像头后点击【开始】'))
            self.status.setObjectName("status_line")
            self.status.setWordWrap(True)
            root.addWidget(self.status)

            self.video_radio.toggled.connect(self._sync_mode)
            self._sync_mode()

        # -- small helpers -----------------------------------------------------
        def _set_status(self, text: str, error: bool = False) -> None:
            self.status.setText(str(text))
            self.status.setProperty("error", "true" if error else "false")
            style = self.status.style()
            style.unpolish(self.status)
            style.polish(self.status)
            self.main_window.set_status(str(text), error=error)

        def capture_state(self) -> dict[str, Any]:
            return {"video": bool(self.video_radio.isChecked()),
                    "path": self.path_edit.text(),
                    "camera_index": int(self.index_spin.value())}

        def restore_state(self, state: Mapping[str, Any]) -> None:
            if state.get("path"):
                self.path_edit.setText(str(state["path"]))
            self.index_spin.setValue(int(state.get("camera_index") or 0))
            if state.get("video", True):
                self.video_radio.setChecked(True)
            else:
                self.camera_radio.setChecked(True)
            self._sync_mode()

        def _sync_mode(self) -> None:
            video = self.video_radio.isChecked()
            self.path_edit.setVisible(video)
            self.browse_button.setVisible(video)
            self.index_label.setVisible(not video)
            self.index_spin.setVisible(not video)
            self._sync_buttons()

        def _sync_buttons(self) -> None:
            busy = self.main_window.is_busy()
            running = self._running
            self.start_button.setEnabled(not busy and not running)
            self.stop_button.setEnabled(running)
            self.save_button.setEnabled(self.current_frame is not None)
            for widget in (self.path_edit, self.browse_button, self.video_radio,
                           self.camera_radio):
                widget.setEnabled(not running)
            self.index_spin.setEnabled(not running and self.camera_radio.isChecked())
            if running:
                self.input_label.setText(tr('输入：{0}').format(self._source_text()))
            elif self.video_radio.isChecked():
                text = self.path_edit.text().strip()
                self.input_label.setText(tr('输入：{0}').format(text) if text else tr('输入：未选择视频文件'))
            else:
                self.input_label.setText(tr('输入：摄像头 {0}').format(self.index_spin.value()))

        def _source_text(self) -> str:
            if self.camera_radio.isChecked():
                return tr('摄像头 {0}').format(self.index_spin.value())
            text = self.path_edit.text().strip()
            return text or tr('未选择视频文件')

        # -- input -------------------------------------------------------------
        def _choose_video(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, tr('选择视频文件'), str(self._last_dir),
                tr('视频 (*.mp4 *.avi *.mov *.mkv *.m4v);;所有文件 (*)'),
            )
            if path:
                self.path_edit.setText(path)
                self.video_radio.setChecked(True)
                self._last_dir = Path(path).parent
                self._sync_buttons()

        def set_input(self, kind: str, value: str | int = "") -> None:
            """Select the input without a dialog (used by tests and screenshots)."""
            if str(kind).lower().startswith("cam"):
                self.camera_radio.setChecked(True)
                try:
                    self.index_spin.setValue(int(value or 0))
                except (TypeError, ValueError):
                    self.index_spin.setValue(0)
            else:
                self.video_radio.setChecked(True)
                self.path_edit.setText(str(value))
                if value:
                    candidate = Path(str(value))
                    if candidate.parent.exists():
                        self._last_dir = candidate.parent
            self._sync_buttons()

        def _resolve_source(self) -> tuple[Any, str | None]:
            """Return ``(source, error)`` for the current input selection."""
            if self.camera_radio.isChecked():
                return int(self.index_spin.value()), None
            text = self.path_edit.text().strip()
            if not text:
                return None, tr('请先选择视频文件，或切换到摄像头输入')
            path = Path(text).expanduser()
            if not path.is_file():
                return None, tr('视频文件不存在：{0}').format(path)
            return str(path), None

        # -- stream lifecycle ---------------------------------------------------
        def start_stream(self) -> bool:
            """Start the stream task; returns ``False`` when it cannot start."""
            if self.main_window.is_busy():
                self._set_status(tr('当前已有任务在运行，请等待结束或先停止'), error=True)
                return False
            session = self.main_window.session
            if session is None or not session.loaded:
                self._set_status(tr('尚未加载模型：请先在【模型与参数】页加载 10 类权重'), error=True)
                return False
            source, error = self._resolve_source()
            if error is not None:
                self._set_status(error, error=True)
                return False

            options = self.main_window.inference_options()
            task = StreamTask(session, source, options,
                              capture_factory=self.capture_factory)
            if not self.main_window.acquire_task(task, tr('视频检测中：{0}').format(self._source_text())):
                return False
            task.frame_ready.connect(self._on_frame)
            task.failed.connect(self._on_failure)
            task.finished.connect(self._on_finished)
            self.stream_task = task
            self._running = True
            self._last_error = None
            self.frame_count = 0
            self.frame_label.setText(tr('已接收帧：0'))
            self._sync_buttons()
            self._set_status(tr('检测中：{0} … 点击【停止】结束').format(self._source_text()))
            task.start()
            return True

        def stop_stream(self) -> bool:
            """Ask the worker to stop; the capture is released in its ``finally``."""
            task = self.stream_task
            if task is None or not self._running:
                return False
            task.request_stop()
            self.stop_button.setEnabled(False)
            self._set_status(tr('正在停止…'))
            return True

        def _on_frame(self, annotated: Any, result: Any) -> None:
            self.current_frame = annotated
            self.last_result = result
            self.frame_count += 1
            if annotated is not None:
                self.view.set_image_bgr(annotated)
            counts = result.class_counts() if result is not None else {}
            self.class_table.set_counts(counts)
            total = int(result.count) if result is not None else 0
            self.total_label.setText(tr('当前帧目标数：{0}').format(total))
            self.frame_label.setText(tr('已接收帧：{0}').format(self.frame_count))
            self._sync_buttons()

        def _on_failure(self, message: str) -> None:
            self._last_error = str(message)
            self._set_status(tr('视频/摄像头检测失败：{0}').format(message), error=True)

        def _on_finished(self) -> None:
            self._running = False
            self._sync_buttons()
            if self._last_error is not None:
                return
            task = self.stream_task
            processed = int(getattr(task, "frames_processed", 0) or 0)
            if task is not None and getattr(task, "stop_requested", False):
                self._set_status(tr('已停止，共处理 {0} 帧').format(processed))
            else:
                self._set_status(tr('输入已结束，共处理 {0} 帧').format(processed))

        # -- frame export -------------------------------------------------------
        def _default_export_dir(self) -> Path:
            root = getattr(self.main_window, "app_data_root", None) or Path.cwd()
            return Path(root) / "captures"

        def _save_frame_dialog(self) -> None:
            if self.current_frame is None:
                self._set_status(tr('还没有可保存的帧，请先开始检测'), error=True)
                return
            default = self._default_export_dir() / f"frame_{self.frame_count:04d}.png"
            path, _ = QFileDialog.getSaveFileName(
                self, tr('保存当前帧'), str(default), tr('PNG 图片 (*.png);;JPEG 图片 (*.jpg)'))
            if not path:
                return
            try:
                target = self.save_current_frame(path)
            except UAVDetError as exc:
                self._set_status(str(exc), error=True)
                return
            self._set_status(tr('已保存当前帧：{0}').format(target))

        def save_current_frame(self, path: str | Path) -> Path:
            """Write the frame on screen (with annotation boxes) to ``path``."""
            if self.current_frame is None:
                raise UAVDetError(tr('还没有可保存的帧，请先开始检测'))
            return widgets.write_image_bgr(path, self.current_frame)

        # -- lifecycle ----------------------------------------------------------
        def on_busy_changed(self, busy: bool) -> None:
            self._sync_buttons()

        def on_shutdown(self) -> None:
            task = self.stream_task
            if task is None:
                return
            if task.isRunning():
                task.request_stop()
                task.wait(4000)

    return {"StreamPage": StreamPage}
