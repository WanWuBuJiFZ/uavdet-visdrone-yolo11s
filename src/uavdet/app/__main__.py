"""``python -m uavdet.app`` → :func:`uavdet.app.main.main`."""

from __future__ import annotations

import sys


def run() -> int:
    """Execute the desktop app and return its exit code."""
    from uavdet.app.main import main

    return main()


if __name__ == "__main__":
    sys.exit(run())
