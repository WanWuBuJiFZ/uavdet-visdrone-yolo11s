"""Shared widgets, BGR→QImage conversion and the app stylesheet loader.

Design rules (see ``docs/INTERFACE_CONTRACT.md`` §3.7):

* importing this module must **not** import PyQt6 — the command line
  (``python -m uavdet --help``) has to work on a machine without Qt. Every
  Qt class is therefore built inside :func:`_build` and re-exported lazily via
  the module level ``__getattr__`` hook (PEP 562). ``from uavdet.app.widgets
  import ImageView`` still works; it just needs Qt at that moment.
* the BGR→RGB conversion happens **exactly once**, here. Pages receive
  ``QImage``/``QPixmap`` values from these helpers and must never convert an
  array themselves (double conversion is the classic colour bug).
* ``ImageView`` keeps the aspect ratio: the painting code scales with
  ``KeepAspectRatio`` and centres the result, so a frame is never distorted.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from . import i18n
from .i18n import tr

__all__ = [
    "LazyExports",
    "PACKAGE_NAME",
    "STYLESHEET_NAME",
    "fit_size",
    "stylesheet_ref",
    "stylesheet_path",
    "load_stylesheet",
    "read_image_bgr",
    "write_image_bgr",
    "bgr_to_qimage",
    "bgr_to_pixmap",
    "pixmap_from_file",
    "ImageView",
    "ClassCountsTable",
    "DetectionTable",
]

#: Package that owns ``styles.qss`` (looked up with :mod:`importlib.resources`).
PACKAGE_NAME = "uavdet.app"
STYLESHEET_NAME = "styles.qss"


# ---------------------------------------------------------------------------
# lazy Qt export plumbing (shared by every module of the desktop app)
# ---------------------------------------------------------------------------
class LazyExports:
    """Defer building Qt classes until one of them is actually read.

    A module uses it like this::

        _lazy = LazyExports(__name__, _build).register("ImageView")

        def __getattr__(name):
            return _lazy.resolve(name)

    ``_build()`` imports Qt, defines the classes and returns ``{name: object}``.
    The result is cached in the module namespace, so ``_build`` runs once and
    later attribute lookups are plain globals.
    """

    def __init__(self, module_name: str, builder: Callable[[], Mapping[str, Any]]):
        self.module_name = module_name
        self.builder = builder
        self.names: tuple[str, ...] = ()

    def register(self, *names: str) -> "LazyExports":
        self.names = tuple(names)
        return self

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def resolve(self, name: str) -> Any:
        """Build (once) and return ``name``; ``AttributeError`` when unknown."""
        import sys

        if name not in self.names:
            raise AttributeError(f"module {self.module_name!r} has no attribute {name!r}")
        namespace = vars(sys.modules[self.module_name])
        if name not in namespace:
            namespace.update(dict(self.builder()))
        return namespace[name]


# ---------------------------------------------------------------------------
# pure geometry helpers (tested without Qt)
# ---------------------------------------------------------------------------
def fit_size(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int]:
    """Aspect-fit ``(src_w, src_h)`` inside ``(dst_w, dst_h)``; never distorts."""
    src_w, src_h = int(src_w), int(src_h)
    dst_w, dst_h = int(dst_w), int(dst_h)
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return (0, 0)
    scale = min(dst_w / float(src_w), dst_h / float(src_h))
    return (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale))))


# ---------------------------------------------------------------------------
# stylesheet resource (E04: must resolve after a normal package install)
# ---------------------------------------------------------------------------
def stylesheet_ref() -> Any:
    """``importlib.resources`` traversable for ``styles.qss`` (or ``None``)."""
    try:
        return importlib.resources.files(PACKAGE_NAME).joinpath(STYLESHEET_NAME)
    except (ModuleNotFoundError, AttributeError, TypeError):  # pragma: no cover
        return None


def stylesheet_path() -> Path | None:
    """Filesystem path of ``styles.qss`` when the resource lives on disk."""
    ref = stylesheet_ref()
    candidates: list[Path] = []
    if ref is not None:
        try:
            if ref.is_file():
                candidates.append(Path(str(ref)))
        except (OSError, TypeError):  # pragma: no cover - exotic loaders
            pass
    candidates.append(Path(__file__).resolve().with_name(STYLESHEET_NAME))
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:  # pragma: no cover
            continue
    return None


def load_stylesheet() -> str:
    """Read the dark stylesheet; returns ``""`` (never raises) when missing."""
    ref = stylesheet_ref()
    try:
        if ref is not None and ref.is_file():
            return ref.read_text(encoding="utf-8")
    except (OSError, TypeError, UnicodeDecodeError):  # pragma: no cover
        pass
    path = stylesheet_path()
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover
        return ""


# ---------------------------------------------------------------------------
# image I/O (OpenCV is imported inside the functions)
# ---------------------------------------------------------------------------
def read_image_bgr(path: str | Path) -> Any:
    """Decode an image file to a BGR ``uint8`` array.

    ``imdecode`` on a byte buffer is used on purpose so that non-ASCII paths
    (``参考图片/示例.png``) work on every platform.
    """
    import cv2
    import numpy as np

    from ..types import UAVDetError

    target = Path(path)
    try:
        data = np.fromfile(str(target), dtype=np.uint8)
    except OSError as exc:
        raise UAVDetError(tr('无法读取图片 {0}: {1}').format(target, exc)) from exc
    if data.size == 0:
        raise UAVDetError(tr('图片为空或不是文件: {0}').format(target))
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or getattr(image, "size", 0) == 0:
        raise UAVDetError(tr('无法解码图片（格式不受支持或文件损坏）: {0}').format(target))
    return image


def write_image_bgr(path: str | Path, image_bgr: Any) -> Path:
    """Write a BGR array as PNG/JPEG (chosen by the target suffix)."""
    import cv2
    import numpy as np

    from ..types import UAVDetError

    target = Path(path)
    suffix = target.suffix.lower() or ".png"
    array = np.asarray(image_bgr)
    if array.ndim != 3 or array.shape[2] != 3:
        raise UAVDetError(tr('只能保存 BGR 三通道图像，收到 shape={0}').format(array.shape))
    ok, buffer = cv2.imencode(suffix, array)
    if not ok:
        raise UAVDetError(tr('无法编码图像为 {0}: {1}').format(suffix, target))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(buffer.tobytes())
    return target


def bgr_to_qimage(image_bgr: Any) -> Any:
    """Convert a BGR ``uint8`` array to a detached ``QImage`` (one conversion)."""
    import cv2
    import numpy as np

    from ..types import UAVDetError

    if image_bgr is None:
        raise UAVDetError(tr('没有可显示的图像'))
    array = np.asarray(image_bgr)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim != 3 or array.shape[2] != 3:
        raise UAVDetError(tr('图像必须是 BGR 三通道，收到 shape={0}').format(array.shape))
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)  # the single BGR→RGB step
    rgb = np.ascontiguousarray(rgb)

    from PyQt6.QtGui import QImage

    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    # ``copy()`` owns the pixels: the numpy buffer may be reused by the worker.
    return image.copy()


def bgr_to_pixmap(image_bgr: Any) -> Any:
    """Convert a BGR array to a ``QPixmap`` (main thread only)."""
    from PyQt6.QtGui import QPixmap

    return QPixmap.fromImage(bgr_to_qimage(image_bgr))


def pixmap_from_file(path: str | Path) -> Any:
    """Load an image file as ``QPixmap``; returns ``None`` when unreadable."""
    from PyQt6.QtGui import QPixmap

    pixmap = QPixmap(str(path))
    return None if pixmap.isNull() else pixmap


# ---------------------------------------------------------------------------
# widgets (built on first access)
# ---------------------------------------------------------------------------
_lazy = LazyExports(__name__, lambda: _build()).register(
    "ImageView", "ClassCountsTable", "DetectionTable"
)


def __getattr__(name: str) -> Any:
    return _lazy.resolve(name)


def _build() -> dict[str, Any]:
    """Import Qt and define the widget classes (called once, lazily)."""
    from PyQt6.QtCore import QSize, Qt
    from PyQt6.QtGui import QColor, QPainter, QPixmap
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QHeaderView,
        QSizePolicy,
        QTableWidget,
        QTableWidgetItem,
        QWidget,
    )

    from ..types import CLASS_NAMES, CLASS_NAMES_CN

    class ImageView(QWidget):
        """Aspect-fit image canvas with a placeholder text.

        The pixmap is scaled at paint time to the widget's current size with
        ``KeepAspectRatio``; nothing is pre-scaled into a distorted copy, and
        ``display_size()`` exposes the same computation for tests.
        """

        def __init__(self, parent: QWidget | None = None,
                     placeholder: str = tr('尚未选择图片'), minimum: tuple[int, int] = (260, 200)):
            super().__init__(parent)
            self._pixmap: QPixmap | None = None
            self._placeholder = str(placeholder)
            self.setMinimumSize(QSize(int(minimum[0]), int(minimum[1])))
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # -- API ------------------------------------------------------------
        def set_placeholder(self, text: str) -> None:
            self._placeholder = str(text)
            self.update()

        def set_image_bgr(self, image_bgr: Any) -> None:
            self.set_pixmap(bgr_to_pixmap(image_bgr))

        def set_pixmap(self, pixmap: QPixmap | None) -> None:
            self._pixmap = pixmap
            self.update()

        def set_image_file(self, path: str | Path) -> bool:
            pixmap = pixmap_from_file(path)
            self.set_pixmap(pixmap)
            return pixmap is not None

        def clear(self) -> None:
            self._pixmap = None
            self.update()

        def has_image(self) -> bool:
            return self._pixmap is not None and not self._pixmap.isNull()

        def image_size(self) -> tuple[int, int]:
            if not self.has_image():
                return (0, 0)
            return (int(self._pixmap.width()), int(self._pixmap.height()))

        def display_size(self) -> tuple[int, int]:
            """Size the image currently occupies inside the widget."""
            width, height = self.image_size()
            return fit_size(width, height, self.width(), self.height())

        # -- painting --------------------------------------------------------
        def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#0b1220"))
            if not self.has_image():
                painter.setPen(QColor("#64748b"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                                 self._placeholder)
                return
            width, height = self.display_size()
            scaled = self._pixmap.scaled(
                width, height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
            return QSize(480, 360)

    class _BaseTable(QTableWidget):
        """Read-only table with the shared dark styling."""

        def __init__(self, headers: Sequence[str], parent: QWidget | None = None):
            super().__init__(0, len(headers), parent)
            self.setHorizontalHeaderLabels(list(headers))
            self.verticalHeader().setVisible(False)
            self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.setAlternatingRowColors(True)
            self.setShowGrid(False)
            self.verticalHeader().setDefaultSectionSize(24)
            header = self.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            header.setHighlightSections(False)
            self.setMinimumHeight(140)

        def _item(self, text: Any, align_right: bool = False) -> QTableWidgetItem:
            item = QTableWidgetItem(str(text))
            if align_right:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            else:
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            return item

        def _set_row_count(self, count: int) -> None:
            self.setRowCount(int(count))

    class ClassCountsTable(_BaseTable):
        """Per-class counts for the current image / current frame (0-9 order)."""

        def __init__(self, parent: QWidget | None = None):
            english_only = i18n.get_language() == "en"
            headers = ([tr('类别'), tr('数量')] if english_only
                       else [tr('类别'), tr('英文名'), tr('数量')])
            super().__init__(headers, parent)
            self.set_counts({})

        def set_counts(self, counts: Mapping[str, int] | None) -> None:
            counts = dict(counts or {})
            english_only = i18n.get_language() == "en"
            self.setColumnCount(2 if english_only else 3)
            self._set_row_count(len(CLASS_NAMES))
            for row, (name, name_cn) in enumerate(zip(CLASS_NAMES, CLASS_NAMES_CN)):
                value = int(counts.get(name, 0) or 0)
                self.setItem(row, 0, self._item(name if english_only else name_cn))
                if not english_only:
                    self.setItem(row, 1, self._item(name))
                count_item = self._item(value, align_right=True)
                if value:
                    count_item.setForeground(QColor("#93c5fd"))
                self.setItem(row, 2 if not english_only else 1, count_item)

        def counts(self) -> dict[str, int]:
            values: dict[str, int] = {}
            count_column = 1 if i18n.get_language() == "en" else 2
            for row, name in enumerate(CLASS_NAMES):
                item = self.item(row, count_column)
                try:
                    values[name] = int(item.text()) if item is not None else 0
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    values[name] = 0
            return values

    class DetectionTable(_BaseTable):
        """Detection list: index, class, confidence and original-image xyxy."""

        def __init__(self, parent: QWidget | None = None):
            super().__init__(["#", tr('类别'), tr('置信度'), "x1", "y1", "x2", "y2"], parent)
            header = self.horizontalHeader()
            for column in (0, 1, 2):
                header.setSectionResizeMode(
                    column, QHeaderView.ResizeMode.ResizeToContents)

        def set_detections(self, detections: Iterable[Any] | None) -> None:
            rows = list(detections or [])
            self._set_row_count(len(rows))
            for row, detection in enumerate(rows):
                class_id = int(getattr(detection, "class_id", -1))
                name_cn = CLASS_NAMES_CN[class_id] if 0 <= class_id < len(CLASS_NAMES_CN) else tr('未知')
                name = str(getattr(detection, "class_name", ""))
                display = name if i18n.get_language() == "en" else f"{name_cn} / {name}"
                self.setItem(row, 0, self._item(row + 1, align_right=True))
                self.setItem(row, 1, self._item(display))
                self.setItem(row, 2, self._item(f"{float(detection.confidence):.3f}",
                                               align_right=True))
                x1, y1, x2, y2 = (float(v) for v in detection.xyxy)
                for column, value in enumerate((x1, y1, x2, y2), start=3):
                    self.setItem(row, column, self._item(f"{value:.1f}", align_right=True))

        def row_count(self) -> int:
            return int(self.rowCount())

    return {
        "ImageView": ImageView,
        "ClassCountsTable": ClassCountsTable,
        "DetectionTable": DetectionTable,
    }
