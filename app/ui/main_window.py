"""PySide6 main window — hosts the feature screens behind top navigation.

Two independent features, two screens:
  * Upscaler  — app/ui/upscaler_screen.py (core in app/upscaler)
  * Metadata  — app/ui/metadata_page.py   (engine in app/engine/metadata_*)

They share no pipeline: upscaling saves files to disk, and the Metadata screen
is then pointed at whatever the user wants to describe — it never starts by
itself after an upscale.

Both screens use the shared design (app/design.py, applied by app/ui/theme.py)
with the bundled Inter Tight font, and follow the system's light/dark setting.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QWidget,
)

from app import paths
from app.ui import theme
from app.ui.icons import icon, refresh_icons
from app.ui.metadata_page import MetadataPage
from app.ui.upscaler_screen import UpscalerScreen


def cuda_info() -> dict:
    """Best-effort CUDA probe (torch import kept local so startup stays light)."""
    info = {"available": False, "name": "", "vram_gb": 0.0, "torch": ""}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["available"] = True
            info["name"] = props.name
            info["vram_gb"] = round(props.total_memory / (1024**3), 1)
    except Exception as exc:  # pragma: no cover - diagnostic only
        info["error"] = str(exc)
    return info


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        theme.ensure_applied()
        self.setWindowTitle("Batch Upscaler & Metadata Generator")
        app_icon = paths.asset("app.ico")
        if app_icon.exists():
            self.setWindowIcon(QIcon(str(app_icon)))
        self.resize(1280, 820)
        self.setMinimumSize(720, 560)
        self._info = cuda_info()

        # Feature screens (each one self-contained with its own worker thread).
        self.upscale_page = UpscalerScreen(self._info)
        self.metadata_page = MetadataPage()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.upscale_page)
        self.stack.addWidget(self.metadata_page)
        self.stack.currentChanged.connect(self._on_page_changed)
        self.setCentralWidget(self.stack)

        self._build_menu()
        self._build_statusbar()
        try:   # follow the system light/dark switch while running (Qt 6.5+)
            QGuiApplication.styleHints().colorSchemeChanged.connect(self._on_scheme_changed)
        except AttributeError:
            pass

    # -- construction ---------------------------------------------------------
    def _build_menu(self) -> None:
        bar = QToolBar("Main")
        bar.setObjectName("navBar")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setIconSize(QSize(18, 18))
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        bar.setAccessibleName("Screens")
        self.addToolBar(bar)

        title = QLabel("Batch Upscaler")
        title.setObjectName("appTitle")
        bar.addWidget(title)

        # Top navigation between the two feature screens.
        self.nav_group = QActionGroup(self)
        self.nav_group.setExclusive(True)
        self.act_upscale = QAction(icon("scaling"), "Upscaler", self, checkable=True)
        self.act_upscale.setChecked(True)
        self.act_upscale.setToolTip("Upscaler — enlarge images 1×–5× with Real-ESRGAN (Ctrl+1)")
        self.act_upscale.setShortcut("Ctrl+1")
        self.act_upscale.triggered.connect(lambda: self.stack.setCurrentWidget(self.upscale_page))
        self.act_metadata = QAction(icon("tags"), "Metadata", self, checkable=True)
        self.act_metadata.setToolTip("Metadata — titles, descriptions and keywords for images "
                                     "(Ctrl+2)")
        self.act_metadata.setShortcut("Ctrl+2")
        self.act_metadata.triggered.connect(lambda: self.stack.setCurrentWidget(self.metadata_page))
        for a in (self.act_upscale, self.act_metadata):
            self.nav_group.addAction(a)
            bar.addAction(a)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        self.act_about = QAction(icon("info"), "About", self)
        self.act_about.setToolTip("About this app")
        self.act_about.triggered.connect(self.show_about)
        bar.addAction(self.act_about)
        about_button = bar.widgetForAction(self.act_about)
        about_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        about_button.setAccessibleName("About")

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        sb.setSizeGripEnabled(False)
        self.setStatusBar(sb)
        if self._info["available"]:
            txt = (f"GPU: {self._info['name']} ({self._info['vram_gb']} GB VRAM)"
                   f"  ·  CUDA enabled  ·  torch {self._info['torch']}")
        else:
            txt = f"⚠ CUDA unavailable — upscaler will run on CPU  ·  torch {self._info['torch']}"
        self.status_text = QLabel(txt)
        sb.addWidget(self.status_text, 1)

    # -- behaviour ------------------------------------------------------------
    def _on_page_changed(self, index: int) -> None:
        # Keep the nav check state in sync however the page was switched.
        (self.act_upscale if index == 0 else self.act_metadata).setChecked(True)

    def _on_scheme_changed(self, *_args) -> None:
        theme.apply(QApplication.instance())
        refresh_icons(self)
        self.act_upscale.setIcon(icon("scaling"))
        self.act_metadata.setIcon(icon("tags"))
        self.act_about.setIcon(icon("info"))
        self.update()

    def show_about(self) -> None:
        QMessageBox.about(
            self, "About",
            "<b>Batch Upscaler &amp; Metadata Generator</b><br>"
            "Upscaler: Real-ESRGAN x4plus on PyTorch (CUDA when available) — 1×–5×, every "
            "image keeps its shape and whole picture. 4× is the model's native scale; other "
            "factors are finished with a Lanczos resize.<br>"
            "Metadata: Adobe-ready titles, descriptions and keywords via Gemini, Groq or "
            "Ollama (local server or hosted cloud).<br><br>"
            "The two features are independent: upscale and save first, then open Metadata "
            "and choose the saved images or folder.<br><br>"
            "Font: Inter Tight (SIL Open Font License). Icons: Lucide (ISC).")

    # -- shutdown -------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        self.upscale_page.shutdown()
        self.metadata_page.shutdown()
        event.accept()
