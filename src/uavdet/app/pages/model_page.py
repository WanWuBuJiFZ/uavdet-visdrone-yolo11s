"""模型与参数页：选择 10 类权重、设置推理参数、加载模型并持久化设置。

Model loading happens in a worker thread owned by
:class:`~uavdet.app.main_window.MainWindow`; this page only edits controls and
reports status. Switching is refused while a task is running (stop first), and a
failed load keeps the previous valid model (acceptance rule C04).

Settings are written to ``<app_data_root>/settings.json`` with
:func:`uavdet.artifacts.write_json_atomic`; the window owns that file so every
page shares exactly one source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...types import CLASS_NAMES, NUM_CLASSES, UAVDetError
from ..widgets import LazyExports
from ..i18n import tr

__all__ = ["ModelPage"]

#: Values offered in the ``imgsz`` combo (all multiples of the YOLO stride 32).
IMGSZ_CHOICES = ("320", "416", "480", "512", "640", "800", "960", "1280")
DEVICE_CHOICES = ("cpu", "mps", "0", "cuda:0")

_lazy = LazyExports(__name__, lambda: _build()).register("ModelPage")


def __getattr__(name: str) -> Any:
    return _lazy.resolve(name)


def _build() -> dict[str, Any]:
    """Import Qt and define :class:`ModelPage` (called once, lazily)."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    class ModelPage(QWidget):
        """Weights + inference parameters + load status."""

        def __init__(self, main_window: Any, parent: QWidget | None = None):
            super().__init__(parent)
            self.main_window = main_window
            self._last_dir = Path.cwd()
            self.setObjectName("page")
            self._build_ui()
            self.load_settings_into_controls()
            main_window.model_load_finished.connect(self._on_load_finished)
            main_window.model_status_changed.connect(self._on_model_status)
            self._on_busy_changed(False)

        # -- construction ------------------------------------------------------
        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(16, 12, 16, 12)
            root.setSpacing(10)

            weights_card = QWidget()
            weights_card.setObjectName("card")
            weights_layout = QVBoxLayout(weights_card)
            weights_layout.setContentsMargins(12, 10, 12, 10)
            weights_layout.setSpacing(7)

            title = QLabel(tr('检测权重（必须是本项目训练的 10 类模型）'))
            title.setObjectName("card_title")
            weights_layout.addWidget(title)

            row = QHBoxLayout()
            row.setSpacing(8)
            self.weights_edit = QLineEdit()
            self.weights_edit.setPlaceholderText(tr('例如 outputs/formal_001/train/weights/best.pt'))
            self.weights_edit.textChanged.connect(self._refresh_weights_hint)
            row.addWidget(self.weights_edit, 1)
            self.browse_button = QPushButton(tr('浏览…'))
            self.browse_button.setProperty("kind", "ghost")
            self.browse_button.clicked.connect(self._choose_weights)
            row.addWidget(self.browse_button)
            self.recent_button = QPushButton(tr('最近实验 best.pt'))
            self.recent_button.setProperty("kind", "ghost")
            self.recent_button.clicked.connect(self.use_recent_weights)
            row.addWidget(self.recent_button)
            weights_layout.addLayout(row)

            self.weights_hint = QLabel("")
            self.weights_hint.setObjectName("hint_line")
            weights_layout.addWidget(self.weights_hint)
            root.addWidget(weights_card)

            params_card = QGroupBox(tr('推理参数（与 CLI predict / benchmark 使用同一套取值）'))
            form = QFormLayout(params_card)
            form.setContentsMargins(14, 14, 14, 12)
            form.setHorizontalSpacing(14)
            form.setVerticalSpacing(9)

            self.device_combo = QComboBox()
            self.device_combo.setEditable(True)
            self.device_combo.addItems(list(DEVICE_CHOICES))
            form.addRow(tr('设备 (cpu / mps / 0)'), self.device_combo)

            self.conf_spin = QDoubleSpinBox()
            self.conf_spin.setRange(0.01, 1.0)
            self.conf_spin.setSingleStep(0.05)
            self.conf_spin.setDecimals(2)
            self.conf_spin.setValue(0.25)
            form.addRow(tr('置信度阈值 conf'), self.conf_spin)

            self.iou_spin = QDoubleSpinBox()
            self.iou_spin.setRange(0.05, 1.0)
            self.iou_spin.setSingleStep(0.05)
            self.iou_spin.setDecimals(2)
            self.iou_spin.setValue(0.70)
            form.addRow("NMS IoU", self.iou_spin)

            self.max_det_spin = QSpinBox()
            self.max_det_spin.setRange(1, 10000)
            self.max_det_spin.setValue(300)
            form.addRow("max_det", self.max_det_spin)

            self.imgsz_combo = QComboBox()
            self.imgsz_combo.setEditable(True)
            self.imgsz_combo.addItems(list(IMGSZ_CHOICES))
            self.imgsz_combo.setCurrentText("640")
            form.addRow(tr('推理尺寸 imgsz'), self.imgsz_combo)

            root.addWidget(params_card)

            actions = QHBoxLayout()
            actions.setSpacing(9)
            self.load_button = QPushButton(tr('加载模型'))
            self.load_button.setProperty("kind", "primary")
            self.load_button.clicked.connect(self.load_model)
            self.save_button = QPushButton(tr('保存参数'))
            self.save_button.setProperty("kind", "primary")
            self.save_button.clicked.connect(self.save_parameters)
            self.reset_button = QPushButton(tr('恢复默认'))
            self.reset_button.setProperty("kind", "ghost")
            self.reset_button.clicked.connect(self.reset_defaults)
            for button in (self.load_button, self.save_button, self.reset_button):
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                actions.addWidget(button)
            actions.addStretch()
            root.addLayout(actions)

            status_card = QWidget()
            status_card.setObjectName("card")
            status_layout = QVBoxLayout(status_card)
            status_layout.setContentsMargins(12, 10, 12, 10)
            status_layout.setSpacing(6)
            status_title = QLabel(tr('状态'))
            status_title.setObjectName("card_title")
            status_layout.addWidget(status_title)

            self.load_state = QLabel(tr('未加载'))
            self.load_state.setObjectName("metric_value")
            status_layout.addWidget(self.load_state)

            self.class_info = QLabel(tr('期望类别：{0} 类（{1}）').format(NUM_CLASSES, '、'.join(CLASS_NAMES)))
            self.class_info.setObjectName("hint_line")
            self.class_info.setWordWrap(True)
            status_layout.addWidget(self.class_info)

            self.warning_label = QLabel("")
            self.warning_label.setObjectName("warning_line")
            self.warning_label.setWordWrap(True)
            self.warning_label.setVisible(False)
            status_layout.addWidget(self.warning_label)

            root.addWidget(status_card)

            self.status = QLabel(tr('调整参数后可点击【保存参数】，再点击【加载模型】'))
            self.status.setObjectName("status_line")
            self.status.setWordWrap(True)
            root.addWidget(self.status)
            root.addStretch()

        # -- helpers -----------------------------------------------------------
        def _set_status(self, text: str, error: bool = False) -> None:
            self.status.setText(str(text))
            self.status.setProperty("error", "true" if error else "false")
            style = self.status.style()
            style.unpolish(self.status)
            style.polish(self.status)
            self.main_window.set_status(str(text), error=error)

        def _settings(self) -> dict[str, Any]:
            return dict(getattr(self.main_window, "settings", {}) or {})

        def load_settings_into_controls(self) -> None:
            """Push the window's persisted settings into the controls."""
            settings = self._settings()
            self.weights_edit.setText(str(settings.get("weights") or ""))
            self.device_combo.setCurrentText(str(settings.get("device") or "cpu"))
            try:
                self.conf_spin.setValue(float(settings.get("conf", 0.25)))
                self.iou_spin.setValue(float(settings.get("iou", 0.70)))
                self.max_det_spin.setValue(int(settings.get("max_det", 300)))
                self.imgsz_combo.setCurrentText(str(int(settings.get("imgsz", 640))))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                self.reset_defaults()
            self._refresh_weights_hint()

        def _refresh_weights_hint(self) -> None:
            text = self.weights_edit.text().strip()
            if not text:
                self.weights_hint.setText(tr('尚未选择权重文件'))
                return
            path = Path(text).expanduser()
            if path.is_file():
                self.weights_hint.setText(tr('✔ 文件存在：{0}').format(path))
            else:
                self.weights_hint.setText(tr('⚠ 文件不存在（加载时会报错）：{0}').format(path))

        def _choose_weights(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, tr('选择训练权重'), str(self._last_dir or Path.cwd()),
                tr('PyTorch 权重 (*.pt *.pth);;所有文件 (*)'),
            )
            if path:
                self.weights_edit.setText(path)
                self._last_dir = Path(path).parent

        def use_recent_weights(self) -> str | None:
            """Fill the field with the newest run's ``train/weights/best.pt``."""
            try:
                found = self.main_window.recent_weights()
            except UAVDetError as exc:
                self._set_status(str(exc), error=True)
                return None
            if not found:
                self._set_status(tr('没有找到任何实验的 train/weights/best.pt，请先完成训练'), error=True)
                return None
            self.weights_edit.setText(str(found))
            self._last_dir = Path(found).parent
            self._set_status(tr('已填入最近一次训练的权重：{0}').format(found))
            return str(found)

        # -- read / validate ---------------------------------------------------
        def read_controls(self) -> dict[str, Any]:
            """Read and validate the six persisted fields.

            Raises :class:`ValueError` with a user-facing message.
            """
            weights = self.weights_edit.text().strip()
            device = self.device_combo.currentText().strip() or "cpu"
            try:
                imgsz = int(float(self.imgsz_combo.currentText().strip()))
            except (TypeError, ValueError) as exc:
                raise ValueError(tr('imgsz 不是整数：{0}').format(repr(self.imgsz_combo.currentText()))) from exc
            if imgsz <= 0 or imgsz % 32 != 0:
                raise ValueError(tr('imgsz 必须是 32 的正整数倍，收到 {0}').format(imgsz))
            return {
                "weights": weights,
                "device": device,
                "imgsz": imgsz,
                "conf": round(float(self.conf_spin.value()), 4),
                "iou": round(float(self.iou_spin.value()), 4),
                "max_det": int(self.max_det_spin.value()),
            }

        # -- actions ------------------------------------------------------------
        def save_parameters(self) -> bool:
            """Validate and persist the current controls to ``settings.json``."""
            try:
                values = self.read_controls()
            except ValueError as exc:
                self._set_status(str(exc), error=True)
                return False
            ok, message = self.main_window.update_settings(values)
            self._set_status(message, error=not ok)
            return ok

        def load_model(self) -> bool:
            """Ask the window to load the configured weights in a worker thread."""
            try:
                values = self.read_controls()
            except ValueError as exc:
                self._set_status(str(exc), error=True)
                return False
            if not values["weights"]:
                self._set_status(tr('请先填写或选择训练权重路径'), error=True)
                return False
            ok, message = self.main_window.load_model(values["weights"], values["device"])
            self._set_status(message, error=not ok)
            if ok:
                self.load_button.setEnabled(False)
            return ok

        def reset_defaults(self) -> None:
            """Restore the code defaults (does not clear the weights path)."""
            self.conf_spin.setValue(0.25)
            self.iou_spin.setValue(0.70)
            self.max_det_spin.setValue(300)
            self.imgsz_combo.setCurrentText("640")
            self.device_combo.setCurrentText("cpu")
            self._set_status(tr('已恢复默认参数（点击【保存参数】后生效）'))

        # -- window signals -----------------------------------------------------
        def _on_load_finished(self, ok: bool, message: str) -> None:
            self.load_button.setEnabled(not self.main_window.is_busy())
            if ok:
                self.load_state.setText(tr('已加载'))
                self._set_status(message)
            else:
                self.load_state.setText(tr('加载失败（保留上一个有效模型）')
                                        if self.main_window.has_model() else tr('未加载'))
                self._set_status(message, error=True)

        def _on_model_status(self, text: str) -> None:
            if self.main_window.has_model():
                self.load_state.setText(tr('已加载'))
            self.warning_label.setVisible(False)
            warnings = list(getattr(self.main_window.session, "warnings", []) or [])
            if warnings:
                self.warning_label.setText(tr('提示：') + "；".join(warnings))
                self.warning_label.setVisible(True)

        def _on_busy_changed(self, busy: bool) -> None:
            self.load_button.setEnabled(not busy and bool(self.weights_edit.text().strip()))
            self.browse_button.setEnabled(not busy)
            self.recent_button.setEnabled(not busy)

        def on_busy_changed(self, busy: bool) -> None:
            self._on_busy_changed(busy)

        def on_shutdown(self) -> None:
            """Nothing to release: the window owns the session."""
            return None

    return {"ModelPage": ModelPage}
