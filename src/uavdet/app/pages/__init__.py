"""功能页（图片检测 / 视频与摄像头 / 模型与参数 / 实验结果）。

The page classes are re-exported lazily so that ``import uavdet.app.pages`` does
not need Qt; ``from uavdet.app.pages import ImagePage`` builds the class on first
use, exactly like the rest of the app package.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ImagePage", "StreamPage", "ModelPage", "ResultsPage"]

_PAGE_MODULES = {
    "ImagePage": "image_page",
    "StreamPage": "stream_page",
    "ModelPage": "model_page",
    "ResultsPage": "results_page",
}


def __getattr__(name: str) -> Any:
    module_name = _PAGE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
