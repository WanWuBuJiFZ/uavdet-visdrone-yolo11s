"""Desktop application package (PyQt6).

``import uavdet.app`` stays harmless: PyQt6 is imported only when :func:`main`
runs or when a widget/worker class is first requested, so the command line and
the cloud tooling keep working on a machine without Qt.

Entry points::

    python -m uavdet.app                 # same as ``uavdet-app``
    uavdet.app.main:main                 # console script target
"""

from __future__ import annotations

from typing import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the desktop app (lazy wrapper around :mod:`uavdet.app.main`)."""
    from .main import main as _main

    return _main(argv)
