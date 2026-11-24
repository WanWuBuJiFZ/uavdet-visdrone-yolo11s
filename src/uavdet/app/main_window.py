"""主窗口:四页导航、任务状态、模型切换与关闭协调。

Responsibilities (``docs/INTERFACE_CONTRACT.md`` §3.7):

* own the only :class:`~uavdet.app.workers.DetectorSession` (the model lives in
  the worker threads; the GUI thread never calls ``predict``),
* allow exactly one background task at a time; every path — success, failure and
  user stop — restores the idle state,
* refuse a model switch while a task is running instead of racing it; a failed
  load keeps the previous valid model,
* persist the inference settings to ``<app_data_root>/settings.json`` through
  :func:`uavdet.artifacts.write_json_atomic`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from ..artifacts import now_iso, read_json, write_json_atomic
from ..config import DEFAULTS, ConfigError, default_config_path, load_config
from ..types import InferenceOptions
from .widgets import LazyExports, load_stylesheet
from .workers import DetectorSession
from . import i18n
from .i18n import tr

__all__ = [
    "MainWindow",
    "NAV_ITEMS",
    "PAGE_META",
    "SETTINGS_FILENAME",
    "SETTINGS_KEYS",
    "SETTINGS_SCHEMA",
    "default_settings",
    "load_settings",
    "save_settings",
    "resolve_config",
]

SETTINGS_FILENAME = "settings.json"
SETTINGS_SCHEMA = 1
SETTINGS_KEYS: tuple[str, ...] = ("weights", "device", "imgsz", "conf", "iou", "max_det",
                                  "language")

NAV_ITEMS: tuple[tuple[str, str, str], ...] = (
    ("🖼️", "图片检测", "image"),
    ("🎬", "视频与摄像头", "stream"),
    ("⚙️", "模型与参数", "model"),
    ("📊", "实验结果", "results"),
)

PAGE_META: dict[str, tuple[str, str]] = {
    "image": ("图片检测", "选择单张图片，加载 10 类模型检测并导出标注图 / CSV"),
    "stream": ("视频与摄像头", "视频文件或摄像头逐帧检测；停止与关闭都会释放采集资源"),
    "model": ("模型与参数", "选择训练权重，设置 conf / iou / max_det / imgsz / device 并保存"),
    "results": ("实验结果", "读取 run 目录：阶段状态、验证指标、已保存图表与缺失产物提示"),
}

#: Threads that outlived a hard close are kept referenced so Python never frees
#: a still-running ``QThread`` (that aborts the process).
_LINGERING_THREADS: list[Any] = []


# ---------------------------------------------------------------------------
# settings file (JSON, atomic)
# ---------------------------------------------------------------------------
def default_settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Code defaults, seeded from the ``predict`` section of the config."""
    predict = dict((config or {}).get("predict") or {})
    return {
        "schema": SETTINGS_SCHEMA,
        "weights": "",
        "device": str(predict.get("device", "cpu")),
        "imgsz": int(predict.get("imgsz", 640)),
        "conf": float(predict.get("conf", 0.25)),
        "iou": float(predict.get("iou", 0.70)),
        "max_det": int(predict.get("max_det", 300)),
        "language": i18n.DEFAULT_LANGUAGE,
    }


def _coerce_settings(raw: Any, defaults: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate a settings mapping; invalid entries fall back with a warning."""
    settings = dict(defaults)
    warnings: list[str] = []
    if not isinstance(raw, Mapping):
        return settings, [tr('settings.json 不是 JSON 对象，已使用默认参数')]
    for key in ("weights", "device"):
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            settings[key] = value.strip()
        else:
            warnings.append(tr('settings.json 的 {0} 不是字符串，已忽略').format(key))
    for key in ("imgsz", "max_det"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            warnings.append(tr('settings.json 的 {0} 不是整数，已忽略').format(key))
            continue
        if key == "imgsz" and (parsed <= 0 or parsed % 32 != 0):
            warnings.append(tr('settings.json 的 imgsz={0} 不是 32 的倍数，已忽略').format(parsed))
            continue
        if key == "max_det" and parsed < 1:
            warnings.append(tr('settings.json 的 max_det={0} 小于 1，已忽略').format(parsed))
            continue
        settings[key] = parsed
    for key in ("conf", "iou"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            settings[key] = float(value)
        except (TypeError, ValueError):
            warnings.append(tr('settings.json 的 {0} 不是数字，已忽略').format(key))
    language = raw.get("language")
    if language is not None:
        if isinstance(language, str) and i18n.is_supported(language.strip()):
            settings["language"] = language.strip()
        else:
            warnings.append(tr('settings.json 的 {0} 不是受支持的语言，已忽略').format("language"))
    try:
        InferenceOptions(imgsz=settings["imgsz"], conf=settings["conf"],
                         iou=settings["iou"], max_det=settings["max_det"],
                         device=str(settings["device"] or "cpu"))
    except ValueError as exc:
        warnings.append(tr('settings.json 参数非法（{0}），已使用默认参数').format(exc))
        settings = dict(defaults)
    return settings, warnings


def load_settings(path: str | Path, config: Mapping[str, Any] | None = None
                  ) -> tuple[dict[str, Any], list[str]]:
    """Read ``settings.json``; returns ``(settings, warnings)`` and never raises."""
    defaults = default_settings(config)
    target = Path(path)
    if not target.is_file():
        return defaults, []
    try:
        raw = read_json(target)
    except (ValueError, OSError) as exc:
        return defaults, [tr('无法读取设置 {0}（{1}）；已使用默认参数').format(target, exc)]
    return _coerce_settings(raw, defaults)


def save_settings(path: str | Path, settings: Mapping[str, Any]) -> Path:
    """Atomically persist the (already validated) settings mapping."""
    payload = {key: settings[key] for key in SETTINGS_KEYS if key in settings}
    payload["schema"] = SETTINGS_SCHEMA
    payload["updated_at"] = now_iso()
    return write_json_atomic(Path(path), payload)


def resolve_config(config_path: str | Path | None) -> tuple[dict[str, Any], list[str]]:
    """Load the project config; fall back to code defaults with a warning."""
    warnings: list[str] = []
    path: Path | None = Path(config_path) if config_path else None
    if path is None:
        try:
            path = default_config_path()
        except ConfigError:
            path = None
    if path is not None:
        try:
            return load_config(path), warnings
        except ConfigError as exc:
            warnings.append(tr('配置文件无法加载（{0}）；界面使用代码默认值').format(exc))

    config = copy.deepcopy(DEFAULTS)
    root = Path.cwd()
    config["project"]["root"] = str(root)
    for key in ("output_root", "workspace_root", "app_data_root"):
        config["project"][key] = str(root / config["project"][key])
    config["data"]["root"] = str(root / config["data"]["root"])
    config["train"]["init_weights"] = str(root / config["train"]["init_weights"])
    config["meta"] = {"config_path": None, "config_chain": [], "loaded_at": now_iso(),
                      "schema": 1}
    if not warnings:
        warnings.append(tr('未找到 configs/project.yaml，界面使用代码默认值（相对当前目录）'))
    return config, warnings


# ---------------------------------------------------------------------------
# window (built lazily: importing this module must not import Qt)
# ---------------------------------------------------------------------------
_lazy = LazyExports(__name__, lambda: _build()).register("MainWindow")


def __getattr__(name: str) -> Any:
    return _lazy.resolve(name)


def _build() -> dict[str, Any]:
    """Import Qt and define :class:`MainWindow` (called once, lazily)."""
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )

    from . import workers

    class MainWindow(QMainWindow):
        """Sidebar + page stack + status bar; the single task coordinator."""

        model_status_changed = pyqtSignal(str)
        model_load_finished = pyqtSignal(bool, str)
        settings_changed = pyqtSignal(object)
        task_state_changed = pyqtSignal(bool, str)

        def __init__(self, config_path: str | Path | None = None,
                     detector_factory: Any = None, settings: Mapping[str, Any] | None = None,
                     settings_path: str | Path | None = None, parent: Any = None):
            super().__init__(parent)
            self.config, self.config_warnings = resolve_config(config_path)
            self.detector_factory = detector_factory or workers.default_detector_factory
            self.app_data_root = Path(self.config["project"]["app_data_root"])
            self.settings_path = (Path(settings_path) if settings_path
                                  else self.app_data_root / SETTINGS_FILENAME)

            if settings is None:
                self.settings, self.settings_warnings = load_settings(
                    self.settings_path, self.config)
            else:
                self.settings, self.settings_warnings = _coerce_settings(
                    dict(settings), default_settings(self.config))

            self.session: DetectorSession | None = None
            self._pending_session: DetectorSession | None = None
            self._active_task: Any = None
            self._model_task: Any = None
            self._busy = False
            self._pages_built = False
            self._shutdown_done = False
            self._nav_buttons: dict[str, QPushButton] = {}
            self._current_page = "image"

            i18n.set_language(str(self.settings.get("language") or i18n.DEFAULT_LANGUAGE))
            self.setWindowTitle(tr('无人机航拍小目标检测系统 · uavdet'))
            self.setMinimumSize(1180, 720)
            self.resize(1400, 880)
            self._apply_style()
            self._build_ui()
            self._navigate_to("image")
            self._update_model_badge()
            self._report_startup_warnings()

        # -- construction ------------------------------------------------------
        def _apply_style(self) -> None:
            stylesheet = load_stylesheet()
            if stylesheet:
                self.setStyleSheet(stylesheet)

        def _build_ui(self) -> None:
            self.setWindowTitle(tr('无人机航拍小目标检测系统 · uavdet'))
            central = QWidget()
            central.setObjectName("root")
            self.setCentralWidget(central)
            root = QHBoxLayout(central)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)
            root.addWidget(self._build_sidebar())

            content = QWidget()
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(0)
            content_layout.addWidget(self._build_header())

            self.stack = QStackedWidget()
            content_layout.addWidget(self.stack, 1)
            root.addWidget(content, 1)

            # status bar widgets exist before pages: pages may report at build time
            status = self.statusBar()
            self.status_line = QLabel(tr('就绪'))
            self.status_line.setObjectName("status_line")
            self.task_state = QLabel(tr('● 就绪'))
            self.task_state.setObjectName("task_state")
            status.addWidget(self.status_line, 1)
            status.addPermanentWidget(self.task_state)

            from .pages.image_page import ImagePage
            from .pages.model_page import ModelPage
            from .pages.results_page import ResultsPage
            from .pages.stream_page import StreamPage

            self.image_page = ImagePage(self)
            self.stream_page = StreamPage(self)
            self.model_page = ModelPage(self)
            self.results_page = ResultsPage(self)
            self._pages_built = True
            for page in self.pages():
                self.stack.addWidget(page)
            self.set_busy(False, tr('就绪'))

        def _build_sidebar(self) -> QWidget:
            sidebar = QWidget()
            sidebar.setObjectName("sidebar")
            sidebar.setFixedWidth(208)
            layout = QVBoxLayout(sidebar)
            layout.setContentsMargins(0, 16, 0, 10)
            layout.setSpacing(0)

            icon = QLabel("🛸")
            icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon.setStyleSheet("font-size:34px;background:transparent;")
            layout.addWidget(icon)

            title = QLabel(tr('无人机目标检测'))
            title.setObjectName("sidebar_logo_title")
            layout.addWidget(title)
            subtitle = QLabel("UAV Small Object Detection")
            subtitle.setObjectName("sidebar_logo_sub")
            layout.addWidget(subtitle)

            divider = QFrame()
            divider.setObjectName("sidebar_divider")
            divider.setFrameShape(QFrame.Shape.HLine)
            layout.addWidget(divider)

            section = QLabel(tr('功 能 导 航'))
            section.setObjectName("sidebar_section")
            layout.addWidget(section)

            for icon_text, label, key in NAV_ITEMS:
                button = QPushButton(f"{icon_text}  {tr(label)}".replace("&", "&&"))
                button.setObjectName("nav_btn")
                button.setCheckable(True)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                button.clicked.connect(lambda checked=False, k=key: self._navigate_to(k))
                layout.addWidget(button)
                self._nav_buttons[key] = button

            layout.addStretch()

            footer = QLabel(tr('VisDrone2019-DET · 10 类\n键盘/鼠标操作 · 离屏可测'))
            footer.setObjectName("sidebar_footer")
            layout.addWidget(footer)
            return sidebar

        def _build_header(self) -> QWidget:
            header = QWidget()
            header.setObjectName("page_header")
            layout = QHBoxLayout(header)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)

            text_column = QVBoxLayout()
            text_column.setContentsMargins(0, 0, 0, 0)
            text_column.setSpacing(0)
            self.page_title = QLabel(tr('图片检测'))
            self.page_title.setObjectName("page_title")
            self.page_subtitle = QLabel("")
            self.page_subtitle.setObjectName("page_subtitle")
            text_column.addWidget(self.page_title)
            text_column.addWidget(self.page_subtitle)
            layout.addLayout(text_column)
            layout.addStretch()

            self.language_button = QPushButton(i18n.LANGUAGE_LABELS[i18n.next_language()])
            self.language_button.setObjectName("language_button")
            self.language_button.setToolTip("Switch UI language / 切换界面语言")
            self.language_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.language_button.clicked.connect(self.toggle_language)
            layout.addWidget(self.language_button)

            self.model_badge = QLabel(tr('模型：未加载'))
            self.model_badge.setObjectName("model_badge")
            layout.addWidget(self.model_badge)
            return header

        # -- language switching ------------------------------------------------
        def toggle_language(self) -> tuple[bool, str]:
            """Switch the whole UI between Chinese and English.

            The pages are rebuilt with the other language instead of carrying a
            second copy of every label; the loaded model, the current image and
            the selected run are carried over so nothing is lost by switching.
            Refused while a task runs (no thread is ever destroyed mid-run).
            """
            if not self._pages_built:
                return False, ""
            if self.is_busy():
                message = tr('当前有任务在运行：请先停止（或等待结束）后再切换语言')
                self.set_status(message, error=True)
                return False, message
            target = i18n.next_language(str(self.settings.get("language")
                                            or i18n.get_language()))
            state = self._capture_page_state()
            i18n.set_language(target)
            self.settings["language"] = target
            self._rebuild_ui()
            self._restore_page_state(state)
            ok, message = self.update_settings({"language": target})
            if not ok:
                self.set_status(message, error=True)
            else:
                self.set_status(tr('就绪'))
            return True, target

        def _capture_page_state(self) -> dict[str, Any]:
            state: dict[str, Any] = {"page": self._current_page}
            for key, page in (("image", self.image_page), ("stream", self.stream_page),
                              ("model", self.model_page), ("results", self.results_page)):
                capture = getattr(page, "capture_state", None)
                if callable(capture):
                    try:
                        state[key] = capture()
                    except Exception:  # noqa: BLE001 - a rebuild must never fail on state
                        state[key] = {}
            return state

        def _restore_page_state(self, state: Mapping[str, Any]) -> None:
            for key, page in (("image", self.image_page), ("stream", self.stream_page),
                              ("model", self.model_page), ("results", self.results_page)):
                restore = getattr(page, "restore_state", None)
                if callable(restore):
                    try:
                        restore(dict(state.get(key) or {}))
                    except Exception as exc:  # noqa: BLE001 - never break the toggle
                        self.set_status(tr('无法恢复页面状态: {0}').format(exc), error=True)
            page_key = str(state.get("page") or "image")
            if page_key in ("image", "stream", "model", "results"):
                self._navigate_to(page_key)
            self._update_model_badge()

        def _rebuild_ui(self) -> None:
            for page in self.pages():
                shutdown = getattr(page, "on_shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception:  # noqa: BLE001
                        pass
            self._pages_built = False
            status = self.statusBar()
            for widget in (getattr(self, "status_line", None),
                           getattr(self, "task_state", None)):
                if widget is not None:
                    try:
                        status.removeWidget(widget)
                    except Exception:  # noqa: BLE001 - older Qt builds
                        pass
            old = self.centralWidget()
            self.setCentralWidget(None)
            self._build_ui()
            if old is not None:
                old.deleteLater()
            self._navigate_to(str(self._current_page or "image"))
            self._update_model_badge()

        # -- navigation --------------------------------------------------------
        def pages(self) -> tuple[Any, ...]:
            if not self._pages_built:
                return ()
            return (self.image_page, self.stream_page, self.model_page, self.results_page)

        def page_for(self, key: str) -> Any:
            mapping = {
                "image": "image_page", "stream": "stream_page",
                "model": "model_page", "results": "results_page",
            }
            return getattr(self, mapping[key], None)

        def _navigate_to(self, key: str) -> None:
            page = self.page_for(key)
            if page is None or key not in PAGE_META:
                return
            self._current_page = key
            for name, button in self._nav_buttons.items():
                button.setChecked(name == key)
            self.stack.setCurrentWidget(page)
            title, subtitle = PAGE_META[key]
            title, subtitle = tr(title), tr(subtitle)
            self.page_title.setText(title)
            self.page_subtitle.setText(subtitle)

        @property
        def current_page(self) -> str:
            return self._current_page

        # -- shared state ------------------------------------------------------
        def inference_options(self) -> InferenceOptions:
            settings = self.settings
            try:
                return InferenceOptions(
                    imgsz=int(settings["imgsz"]),
                    conf=float(settings["conf"]),
                    iou=float(settings["iou"]),
                    max_det=int(settings["max_det"]),
                    device=str(settings.get("device") or "cpu"),
                )
            except (KeyError, TypeError, ValueError):
                return InferenceOptions(device=str(settings.get("device") or "cpu"))

        def has_model(self) -> bool:
            return self.session is not None and bool(self.session.loaded)

        def is_busy(self) -> bool:
            return bool(self._busy)

        def set_status(self, message: str, error: bool = False) -> None:
            line = getattr(self, "status_line", None)
            if line is None:
                return
            line.setText(str(message))
            line.setProperty("error", "true" if error else "false")
            style = line.style()
            style.unpolish(line)
            style.polish(line)

        def set_busy(self, busy: bool, message: str = "") -> None:
            self._busy = bool(busy)
            task_state = getattr(self, "task_state", None)
            if task_state is not None:
                task_state.setText(tr('● 任务运行中') if busy else tr('● 就绪'))
                task_state.setProperty("busy", "true" if busy else "false")
                style = task_state.style()
                style.unpolish(task_state)
                style.polish(task_state)
            for page in self.pages():
                hook = getattr(page, "on_busy_changed", None)
                if callable(hook):
                    hook(self._busy)
            self.task_state_changed.emit(self._busy, str(message))

        # -- task coordination -------------------------------------------------
        def acquire_task(self, task: Any, message: str = "") -> bool:
            """Register ``task`` as the single running job (one at a time)."""
            if self._active_task is not None:
                self.set_status(tr('已有任务正在运行，请等待结束或先停止'), error=True)
                return False
            self._active_task = task
            task.finished.connect(self._on_task_finished)
            self.set_busy(True, message)
            return True

        def _on_task_finished(self) -> None:
            task = self.sender()
            if task is self._active_task:
                self._active_task = None
            self.set_busy(False, tr('就绪'))

        # -- model switching ---------------------------------------------------
        def load_model(self, weights: str, device: str) -> tuple[bool, str]:
            """Load ``weights`` in a worker thread; the old model stays on error."""
            if self._busy:
                return False, tr('当前有任务在运行：请先停止（或等待结束）后再切换模型')
            text = str(weights or "").strip()
            if not text:
                return False, tr('请先填写或选择 10 类训练权重文件')
            # Import the inference stack on this (main) thread: on macOS a first
            # ``import torch`` inside a QThread aborts the process.
            for warning in workers.warm_up_runtime():
                self.set_status(warning, error=True)
            session = DetectorSession(weights=text, device=str(device or "cpu"),
                                      factory=self.detector_factory)
            task = workers.ModelLoadTask(session)
            if not self.acquire_task(task, tr('加载模型中：{0}').format(Path(text).name)):
                return False, tr('当前已有任务在运行，无法加载模型')
            self._pending_session = session
            self._model_task = task
            task.loaded.connect(self._on_model_loaded)
            task.failed.connect(self._on_model_failed)
            task.start()
            return True, tr('正在加载模型：{0}（device={1}）').format(text, device)

        def _on_model_loaded(self, session: Any, info: dict[str, Any]) -> None:
            previous = self.session
            self.session = session
            self._pending_session = None
            self.model_load_finished.emit(
                True, self._model_summary(session, info, prefix=tr('已加载模型')))
            self.model_status_changed.emit(self._model_summary(session, info))
            self._update_model_badge()
            ok, message = self.update_settings(
                {"weights": session.weights, "device": session.device})
            if not ok:
                self.set_status(message, error=True)
            else:
                self.set_status(self._model_summary(session, info))
            if previous is not None and previous is not session:
                previous.close()

        def _on_model_failed(self, message: str) -> None:
            session, self._pending_session = self._pending_session, None
            if session is not None:
                session.close()
            text = tr('模型加载失败：{0}').format(message)
            if self.has_model():
                text += tr('（继续使用上一个有效模型）')
            self.model_status_changed.emit(text)
            self.model_load_finished.emit(False, text)
            self._update_model_badge()
            self.set_status(text, error=True)

        @staticmethod
        def _model_summary(session: Any, info: Mapping[str, Any] | None = None,
                           prefix: str = "") -> str:
            info = dict(info or {})
            parts = [f"{prefix}：" if prefix else "",
                     f"{Path(str(session.weights)).name}",
                     f"device={session.device}"]
            num_classes = info.get("num_classes")
            if num_classes is not None:
                parts.append(tr('{0} 类').format(num_classes))
            warnings = list(getattr(session, "warnings", None) or [])
            text = " · ".join(part for part in parts if part)
            if warnings:
                text += "（" + "；".join(warnings) + "）"
            return text

        def _update_model_badge(self) -> None:
            badge = getattr(self, "model_badge", None)
            if badge is None:
                return
            if self.has_model():
                badge.setText(tr('模型：{0} · {1}').format(Path(str(self.session.weights)).name, self.session.device))
                badge.setProperty("loaded", "true")
            else:
                badge.setText(tr('模型：未加载'))
                badge.setProperty("loaded", "false")
            style = badge.style()
            style.unpolish(badge)
            style.polish(badge)

        # -- settings ----------------------------------------------------------
        def update_settings(self, values: Mapping[str, Any]) -> tuple[bool, str]:
            """Validate + persist the inference settings; returns ``(ok, message)``."""
            merged = dict(self.settings)
            for key, value in dict(values).items():
                if key not in SETTINGS_KEYS:
                    continue
                merged[key] = value
            try:
                InferenceOptions(
                    imgsz=int(merged["imgsz"]), conf=float(merged["conf"]),
                    iou=float(merged["iou"]), max_det=int(merged["max_det"]),
                    device=str(merged.get("device") or "cpu"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                return False, tr('参数非法：{0}').format(exc)
            try:
                save_settings(self.settings_path, merged)
            except OSError as exc:
                return False, tr('无法写入设置文件 {0}: {1}').format(self.settings_path, exc)
            self.settings = merged
            self.settings_changed.emit(dict(merged))
            return True, tr('已保存设置：{0}').format(self.settings_path)

        def recent_weights(self) -> str | None:
            """Newest run that already has ``train/weights/best.pt``."""
            from ..artifacts import list_runs

            root = Path(self.config["project"]["output_root"])
            for record in list_runs(root):
                run_dir = Path(str(record.get("run_dir")))
                for name in ("best.pt", "last.pt"):
                    candidate = run_dir / "train" / "weights" / name
                    if candidate.is_file():
                        return str(candidate)
            return None

        def _report_startup_warnings(self) -> None:
            warnings = list(self.config_warnings) + list(self.settings_warnings)
            if warnings:
                self.set_status(tr('启动提示：') + "；".join(warnings), error=False)
            else:
                self.set_status(tr('就绪：请先在【模型与参数】页加载 10 类权重'))

        # -- shutdown ----------------------------------------------------------
        def shutdown(self) -> None:
            """Stop and wait for the running worker, then release the model."""
            if self._shutdown_done:
                return
            self._shutdown_done = True
            task = self._active_task
            if task is not None and task.isRunning():
                stopper = getattr(task, "request_stop", None)
                if callable(stopper):
                    stopper()
                finished = task.wait(5000)
                if not finished:
                    if callable(stopper):
                        stopper()
                    finished = task.wait(5000)
                if not finished and task.isRunning():
                    # never destroy a running QThread: keep it referenced
                    _LINGERING_THREADS.append(task)
            self._active_task = None
            self.set_busy(False, tr('已关闭'))
            for page in self.pages():
                hook = getattr(page, "on_shutdown", None)
                if callable(hook):
                    hook()
            for session in (self._pending_session, self.session):
                if session is not None:
                    session.close()
            self.session = None
            self._pending_session = None
            self._model_task = None

        def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            self.shutdown()
            event.accept()

    return {"MainWindow": MainWindow}
