"""``python -m uavdet`` entry point (kept tiny so ``--help`` never imports torch)."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
