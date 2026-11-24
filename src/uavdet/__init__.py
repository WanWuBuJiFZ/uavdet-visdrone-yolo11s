"""uavdet — VisDrone2019-DET UAV small-object detection project.

The package is Qt-free at import time: importing :mod:`uavdet` only pulls the
Python standard library. Heavy dependencies (torch, Ultralytics, OpenCV, PyQt6)
are imported by the modules that actually need them.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__all__ = ["__version__", "CLASS_NAMES", "NUM_CLASSES"]


def __getattr__(name: str) -> Any:
    """Expose the fixed class order without importing the whole type module."""
    if name in {"CLASS_NAMES", "NUM_CLASSES"}:
        from . import types

        return getattr(types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
