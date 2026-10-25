"""The Upscaler screen — hosts the Django-based Upscaler inside the shell.

The Upscaler itself is a Django app (app/upscaler) served by a private local
server in this process (app/web/server.py). This widget shows it in an
embedded Chromium view (Qt WebEngine) under the same top navigation as the
Metadata screen, and lends it what a web page can't do alone: native file
and folder dialogs that return real paths, Windows message boxes, drag & drop
from Explorer, and the app's light/dark scheme (the page renders the same
design tokens as the Qt screens, app/design.py). If Qt WebEngine is missing,
it offers to open the same page in the default browser.

Nothing starts until the screen is first shown, so building the main window
stays cheap (and headless tests that never show it start no server).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QObject, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app import design
from app.ui import theme
from app.upscaler.files import IMAGE_SUFFIXES
from app.upscaler.workspace import get_workspace
from app.web import server


class _GuiThreadBridge(QObject):
    """Runs a callable on the GUI thread and waits for its result.

    Django serves requests on its own threads, but Qt dialogs must be opened
    from the GUI thread; a blocking queued signal hands the call across.
    """

    _call = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._call.connect(self._run, Qt.BlockingQueuedConnection)

    def invoke(self, fn: Callable, *args):
        if threading.current_thread() is threading.main_thread():
            return fn(*args)
        box = {"fn": fn, "args": args}
        self._call.emit(box)
        if "error" in box:
            raise box["error"]
        return box.get("result")

    @Slot(object)
    def _run(self, box: dict) -> None:
        try:
            box["result"] = box["fn"](*box["args"])
        except Exception as exc:  # re-raised on the calling thread
            box["error"] = exc


class _DesktopHost:
    """What the Django views may ask of the desktop (see app/upscaler/host.py)."""

    def __init__(self, page: QWidget, bridge: _GuiThreadBridge) -> None:
        self._page = page
        self._bridge = bridge
        # Kept as a plain attribute (set on the GUI thread) so Django's request
        # threads never touch Qt objects to read it.
        self.scheme = theme.current_scheme()

    def theme(self) -> dict[str, str]:
        return {"scheme": self.scheme}

    def pick_images(self) -> list[str]:
        def ask() -> list[str]:
            exts = " ".join(f"*{e}" for e in sorted(IMAGE_SUFFIXES))
            files, _ = QFileDialog.getOpenFileNames(
                self._page, "Select images", str(Path.home()),
                f"Images ({exts});;All files (*)")
            return files
        return self._bridge.invoke(ask)

    def pick_folder(self) -> Optional[str]:
        return self._bridge.invoke(lambda: QFileDialog.getExistingDirectory(
            self._page, "Select a folder", str(Path.home())) or None)

    def pick_output_folder(self, current: str) -> Optional[str]:
        start = current if current and Path(current).is_dir() else str(Path.home())
        return self._bridge.invoke(lambda: QFileDialog.getExistingDirectory(
            self._page, "Choose output folder", start) or None)

    def show_message(self, kind: str, title: str, text: str) -> None:
        box = QMessageBox.warning if kind == "warning" else QMessageBox.information
        self._bridge.invoke(lambda: box(self._page, title, text))


class UpscalerWebPage(QWidget):
    def __init__(self, info: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._info = info                 # CUDA probe the main window already made
        self._bridge = _GuiThreadBridge()
        self._host = _DesktopHost(self, self._bridge)
        self._server: Optional[server.LocalServer] = None
        self.view = None                  # QWebEngineView once started
        self.setAcceptDrops(True)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QLabel("Starting the Upscaler…")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._placeholder)

    # -- startup ----------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._server is None:
            self.start()

    def start(self) -> str:
        """Start the local Django server (once) and show its page; returns the URL."""
        if self._server is None:
            self._host.scheme = theme.current_scheme()
            self._server = server.start(host=self._host)
            self._show_page(self._server.launch_url)
        return self._server.launch_url

    @property
    def url(self) -> Optional[str]:
        return self._server.url if self._server else None

    def set_scheme(self, scheme: str) -> None:
        """Follow a light/dark switch of the desktop without reloading the page."""
        self._host.scheme = design.normalize_scheme(scheme)
        if self.view is not None:
            self.view.page().setBackgroundColor(QColor(design.colors(scheme)["bg"]))
            self.view.page().runJavaScript(
                f"document.documentElement.dataset.theme = {json.dumps(self._host.scheme)};")

    def _show_page(self, url: str) -> None:
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
        except ImportError as exc:
            self._show_browser_fallback(url, exc)
            return
        view = QWebEngineView(self)
        # No white flash before the page paints in dark mode.
        view.page().setBackgroundColor(QColor(design.colors(self._host.scheme)["bg"]))
        # Explorer drops are taken by this widget (it knows the real paths).
        view.setAcceptDrops(False)
        view.setContextMenuPolicy(Qt.NoContextMenu)
        view.installEventFilter(self)
        view.loadFinished.connect(lambda _ok: self._filter_view_children())
        view.load(QUrl(url))
        self._replace_placeholder(view)
        self.view = view

    def _show_browser_fallback(self, url: str, exc: Exception) -> None:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.addStretch(1)
        msg = QLabel(
            "The embedded browser (Qt WebEngine) isn't available, so the Upscaler "
            "can't be shown inside this window.\n\nIt still runs locally — open it in "
            f"your web browser instead. File dialogs will open from this app.\n\n({exc})")
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignCenter)
        btn = QPushButton("Open Upscaler in Browser")
        btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))
        lay.addWidget(msg)
        lay.addWidget(btn, 0, Qt.AlignCenter)
        lay.addStretch(1)
        self._replace_placeholder(box)

    def _replace_placeholder(self, widget: QWidget) -> None:
        self._layout.removeWidget(self._placeholder)
        self._placeholder.deleteLater()
        self._layout.addWidget(widget)
        self._placeholder = widget

    def _filter_view_children(self) -> None:
        # The view renders through child widgets; make sure their drops reach us too.
        if self.view is not None:
            for child in self.view.findChildren(QWidget):
                child.installEventFilter(self)

    # -- drag & drop from Explorer ------------------------------------------------
    def add_paths(self, paths: list[str]) -> None:
        if not paths:
            return
        if self.view is not None:
            # Through the page, so its log/messages report the result as for Add Images.
            self.view.page().runJavaScript(
                f"window.upscalerApp && window.upscalerApp.addPaths({json.dumps(paths)});")
        else:
            get_workspace().add(paths)

    @staticmethod
    def _dropped_paths(event) -> list[str]:
        return [u.toLocalFile() for u in event.mimeData().urls() if u.toLocalFile()]

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if kind in (QEvent.DragEnter, QEvent.DragMove) and event.mimeData().hasUrls():
            event.acceptProposedAction()
            return True
        if kind == QEvent.Drop and event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.add_paths(self._dropped_paths(event))
            return True
        return super().eventFilter(obj, event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self.add_paths(self._dropped_paths(event))

    # -- shutdown -------------------------------------------------------------------
    def shutdown(self) -> None:
        get_workspace().shutdown(timeout=3.0)
        if self._server is not None:
            server.stop()
            self._server = None
