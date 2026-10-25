"""Application entry point — launches the desktop app.

Run with:  python -m app.main      (or BatchUpscaler.exe when packaged)

``--self-test REPORT.json`` runs a headless check instead of the window —
bundled font and icons, the window and its settings form, the Metadata
provider libraries, PyTorch/CUDA, and the real Real-ESRGAN model at every
upscale factor (sizes, aspect ratio, JPEG + batch names) — and writes the
results to REPORT.json (exit code 0 = all passed). It verifies a packaged
build the same way as a source checkout.
"""

from __future__ import annotations

import sys


def _import_torch() -> int:
    # torch must be imported before QApplication on some Windows CUDA setups
    # to avoid a DLL-initialisation race.
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment problem
        print(f"PyTorch is not installed correctly: {exc}", file=sys.stderr)
        print("Run:  .venv\\Scripts\\python -m pip install -r requirements.txt", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if "--self-test" in argv:
        index = argv.index("--self-test")
        report = argv[index + 1] if len(argv) > index + 1 else "self-test.json"
        from app.selftest import run
        return run(report)

    if _import_torch():
        return 2
    from PySide6.QtWidgets import QApplication

    from app.ui import theme
    from app.ui.main_window import MainWindow

    app = QApplication(argv)
    # No display name: Qt would append " - Batch Upscaler" to every window title.
    app.setApplicationName("Batch Upscaler")
    theme.apply(app)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
