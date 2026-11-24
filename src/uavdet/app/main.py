"""Desktop entry point: ``python -m uavdet.app`` / ``uavdet-app``.

:func:`main` creates the single :class:`QApplication` (offscreen when
``QT_QPA_PLATFORM`` says so), shows :class:`~uavdet.app.main_window.MainWindow`
and runs the Qt event loop. Importing this module never imports Qt.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence
from .i18n import tr

__all__ = ["main", "build_parser"]

PROG = "uavdet-app"

_MISSING_QT = (
    "PyQt6 未安装，无法启动桌面界面。\n"
    "请安装 GUI 依赖：python -m pip install -e '.[gui]'\n"
    "（命令行检测/训练/绘图不需要 Qt，可继续使用 python -m uavdet）"
)


def build_parser() -> argparse.ArgumentParser:
    """Arguments for the desktop entry point."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=tr('VisDrone 无人机小目标检测桌面界面（PyQt6）'),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            tr('examples:\n  python -m uavdet.app\n  python -m uavdet.app --config configs/project.yaml\n  QT_QPA_PLATFORM=offscreen python -m uavdet.app   # 离屏/无显示环境\n')
        ),
    )
    parser.add_argument("--config", metavar="PATH", default=None,
                        help=tr('项目 YAML 配置（默认自动查找 configs/project.yaml）'))
    parser.add_argument("--settings", metavar="PATH", default=None,
                        help=tr('覆盖本地 settings.json 路径（默认 <app_data_root>/settings.json）'))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Create the QApplication, show the main window and enter the event loop."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        options = build_parser().parse_args(arguments)
    except SystemExit as exc:  # --help / --version
        return int(exc.code or 0)

    try:
        from PyQt6.QtWidgets import QApplication
    except Exception:  # noqa: BLE001 - Qt missing is a normal cloud situation
        print(tr(_MISSING_QT), file=sys.stderr)
        return 2

    from .main_window import MainWindow

    application = QApplication.instance()
    owns_application = application is None
    if owns_application:
        application = QApplication([sys.argv[0] if sys.argv else PROG])
        application.setApplicationName(PROG)
        application.setApplicationDisplayName(tr('无人机航拍小目标检测系统'))

    try:
        window = MainWindow(config_path=options.config, settings_path=options.settings)
    except Exception as exc:  # noqa: BLE001 - report instead of a bare traceback
        print(tr('无法创建主窗口：{0}: {1}').format(type(exc).__name__, exc), file=sys.stderr)
        return 1

    window.show()
    if not owns_application:
        # Someone else (a test, a notebook) owns the loop; just hand back the window.
        return 0
    return int(application.exec())


if __name__ == "__main__":  # pragma: no cover - convenience only
    sys.exit(main())
