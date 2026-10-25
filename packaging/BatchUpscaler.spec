# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the one-folder desktop app: dist\BatchUpscaler\BatchUpscaler.exe
# Build with build.ps1 (it also ships the model weights and self-tests the result).
from pathlib import Path

ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[(str(ROOT / "app" / "assets"), "app/assets")],   # Inter Tight, icons, app icon
    hiddenimports=[
        "app.ui.main_window",
        "app.selftest",
        "keyring.backends.Windows",
    ],
    excludes=[
        # Desktop-only: no embedded browser, QML or multimedia.
        "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
        "PySide6.QtWebChannel", "PySide6.QtWebSockets", "PySide6.QtQuick", "PySide6.QtQml",
        "PySide6.QtQuickWidgets", "PySide6.QtMultimedia", "PySide6.Qt3DCore",
        "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtPdf",
        # Not used by the app.
        "torchvision", "tkinter", "matplotlib", "IPython", "pytest", "tensorboard",
        "torch.utils.tensorboard",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BatchUpscaler",
    icon=str(ROOT / "app" / "assets" / "app.ico"),
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="BatchUpscaler", upx=False)
