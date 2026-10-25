"""PySide6 main window — hosts the feature screens behind top navigation.

Two independent features, two pages:
  * Upscaler  — the Django-based upscaler (app/upscaler), shown through
                app/ui/upscaler_web_page.py
  * Metadata  — the Adobe-ready metadata generator (app/ui/metadata_page.py)

They share no pipeline: upscaling saves files to disk, and the Metadata page
is then pointed at whatever the user wants to describe.

Both pages use the shared design (app/design.py, applied by app/ui/theme.py)
and follow the system's light/dark setting live.
"""

from __future__ import annotations

from PySide6.QtGui import QAction, QActionGroup, QGuiApplication
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

from app.ui import theme
from app.ui.upscaler_web_page import UpscalerWebPage
from app.ui.metadata_page import MetadataPage


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
        self.resize(1280, 820)
        self.setMinimumSize(720, 560)
        self._info = cuda_info()

        # Feature pages (each one self-contained with its own worker thread).
        self.upscale_page = UpscalerWebPage(self._info)
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
        self.addToolBar(bar)

        title = QLabel("Batch Upscaler")
        title.setObjectName("appTitle")
        bar.addWidget(title)

        # Top navigation between the two feature screens.
        self.nav_group = QActionGroup(self)
        self.nav_group.setExclusive(True)
        self.act_upscale = QAction("Upscaler", self, checkable=True)
        self.act_upscale.setChecked(True)
        self.act_upscale.setToolTip("Enlarge images with Real-ESRGAN, keeping their shape")
        self.act_upscale.triggered.connect(lambda: self.stack.setCurrentWidget(self.upscale_page))
        self.act_metadata = QAction("Metadata", self, checkable=True)
        self.act_metadata.setToolTip("Generate titles, descriptions and keywords for images")
        self.act_metadata.triggered.connect(lambda: self.stack.setCurrentWidget(self.metadata_page))
        for a in (self.act_upscale, self.act_metadata):
            self.nav_group.addAction(a)
            bar.addAction(a)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        act_about = QAction("About", self)
        act_about.triggered.connect(self.show_about)
        bar.addAction(act_about)

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
        scheme = theme.apply(QApplication.instance())
        self.upscale_page.set_scheme(scheme)
        self.metadata_page.update()

    def show_about(self) -> None:
        QMessageBox.about(
            self, "About",
            "<b>Batch Image Upscaler + Metadata Generator</b><br>"
            "Upscaler: Real-ESRGAN x4plus · PyTorch + CUDA · Django · local AI upscaling "
            "that keeps every image's shape and whole picture.<br>"
            "Metadata: Adobe-ready titles/descriptions/keywords via Gemini, Groq "
            "or Ollama (local server / hosted cloud).<br><br>"
            "The two features are independent: upscale and save first, then open "
            "Metadata and select the saved images or folder.")

    # -- shutdown -------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        self.upscale_page.shutdown()
        self.metadata_page.shutdown()
        event.accept()
