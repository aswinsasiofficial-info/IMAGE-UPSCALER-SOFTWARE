"""Application entry point — launches the PySide6 desktop GUI.

Run with:  python -m app.main
"""

from __future__ import annotations

import sys


def main() -> int:
    # torch/torchvision must be imported before QApplication on some Windows
    # CUDA setups to avoid a DLL-initialisation race.
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment problem
        print(f"PyTorch is not installed correctly: {exc}", file=sys.stderr)
        print("Run:  .venv\\Scripts\\python -m pip install -r requirements.txt", file=sys.stderr)
        return 2

    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Batch Image Upscaler")
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
