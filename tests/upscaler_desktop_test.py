"""Desktop integration of the rebuilt Upscaler (offscreen Qt, no model).

Run with:  python -m tests.upscaler_desktop_test

Builds the real main window and checks: both feature pages are present and
independent (Upscaler + Metadata); the Upscaler starts its local Django
server only when shown; native dialogs requested by the page from a server
thread open on the GUI thread and feed the queue; Explorer drops reach the
queue; and, when Qt WebEngine can render offscreen, the embedded page loads
and drives Django end to end — clicking Start in the real page, being
blocked on a shape conflict until Pad is picked, and writing proportional
files. Screenshots of the page are saved under <temp>/upscaler_desktop_test.
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")

try:
    from PySide6 import QtWebEngineWidgets  # noqa: F401 — must precede QApplication
except ImportError:
    pass

from PIL import Image
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from tests.upscaler_test import StubEngine, make_marked  # noqa: E402 (sets up Django)

from app.upscaler.workspace import get_workspace, reset_workspace  # noqa: E402
from app.web import server  # noqa: E402


def pump(app: QApplication, until, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if until():
            return True
        time.sleep(0.01)
    return False


def snap(app: QApplication, win, path: Path) -> None:
    """Screenshot after giving WebEngine's compositor time to deliver the frame."""
    pump(app, lambda: False, 1.0)
    win.grab().save(str(path))


def run_js(app: QApplication, view, script: str, timeout: float = 10.0):
    box = {}
    view.page().runJavaScript(script, 0, lambda value: box.setdefault("v", value))
    pump(app, lambda: "v" in box, timeout)
    return box.get("v")


class Http:
    """Talks to the embedded server like the page does (launch cookie + CSRF)."""

    def __init__(self, launch_url: str, base: str) -> None:
        self.base = base
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        with self.opener.open(launch_url) as resp:
            self.page = resp.read().decode()

    def csrf(self) -> str:
        return next(c.value for c in self.jar if c.name == "csrftoken")

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "X-CSRFToken": self.csrf(),
                                              "Origin": self.base.rstrip("/")})
        with self.opener.open(req) as resp:
            return json.loads(resp.read())


def post_from_thread(app: QApplication, client: Http, path: str, body: dict) -> dict:
    """POST on a worker thread while the GUI thread keeps processing events."""
    box: dict = {}
    t = threading.Thread(target=lambda: box.setdefault("r", client.post(path, body)), daemon=True)
    t.start()
    assert pump(app, lambda: not t.is_alive(), 20), f"{path} never returned (GUI bridge stuck?)"
    return box["r"]


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    from app.ui.main_window import MainWindow
    from app.ui.metadata_page import MetadataPage
    from app.ui.upscaler_web_page import UpscalerWebPage

    reset_workspace(engine_factory=StubEngine)
    shots = Path(tempfile.gettempdir()) / "upscaler_desktop_test"
    shots.mkdir(exist_ok=True)
    calls: dict[str, list] = {"dialog_threads": [], "info": [], "warning": []}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        single = tmp / "single"
        single.mkdir()
        make_marked(single / "portrait.png", 64, 96)
        folder = tmp / "album"
        folder.mkdir()
        make_marked(folder / "landscape.png", 96, 64)
        dropped = tmp / "dropped.png"
        make_marked(dropped, 48, 48)

        win = MainWindow()
        assert win.stack.count() == 2
        assert isinstance(win.upscale_page, UpscalerWebPage)
        assert isinstance(win.metadata_page, MetadataPage)
        assert server._active is None, "server must not start before the screen is shown"
        print("  main window: Upscaler (Django) + Metadata pages, server not started yet")

        win.resize(1180, 760)
        win.show()
        assert pump(app, lambda: server._active is not None, 5)
        page: UpscalerWebPage = win.upscale_page
        client = Http(server._active.launch_url, server._active.url)
        assert "Start Upscaling" in client.page and "--window:#" in client.page
        print(f"  local Django server at {server._active.url} (launch token + CSRF), "
              "Qt palette passed to the page")

        # Native dialogs, requested from a Django request thread.
        main_thread = threading.main_thread()

        def fake_open(*_a, **_k):
            calls["dialog_threads"].append(threading.current_thread() is main_thread)
            return [str(single / "portrait.png")], ""

        def fake_dir(*_a, **_k):
            calls["dialog_threads"].append(threading.current_thread() is main_thread)
            return str(folder)

        QFileDialog.getOpenFileNames = staticmethod(fake_open)
        QFileDialog.getExistingDirectory = staticmethod(fake_dir)
        QMessageBox.information = staticmethod(lambda _p, t, x, *a: calls["info"].append((t, x)))
        QMessageBox.warning = staticmethod(lambda _p, t, x, *a: calls["warning"].append((t, x)))

        r = post_from_thread(app, client, "api/dialog/images/", {})
        assert [q["name"] for q in r["queue"]] == ["portrait.png"]
        r = post_from_thread(app, client, "api/dialog/folder/", {})
        assert [q["name"] for q in r["queue"]] == ["portrait.png", "landscape.png"]
        assert r["added"]["folders"][0]["path"] == str(folder)
        post_from_thread(app, client, "api/dialog/message/",
                         {"kind": "info", "title": "Hello", "text": "from Django"})
        assert calls["info"][-1] == ("Hello", "from Django")
        assert calls["dialog_threads"] and all(calls["dialog_threads"]), calls["dialog_threads"]
        print("  native file/folder dialogs + message box ran on the GUI thread for Django")

        # The embedded page.
        view = page.view
        loaded = view is not None and pump(app, lambda: run_js(
            app, view, "document.readyState === 'complete' && !!window.upscalerApp", 2) is True, 30)
        if not loaded:
            print("  (Qt WebEngine could not render offscreen here — page checks skipped)")
            page.view = None
            mime = _mime(dropped)      # the event doesn't own it: keep it alive
            page.dropEvent(QDropEvent(QPointF(5, 5), Qt.CopyAction, mime,
                                      Qt.LeftButton, Qt.NoModifier))
            assert any(i.path == dropped for i in get_workspace().items)
            print("  Explorer drop -> queue OK (direct path)")
        else:
            assert run_js(app, view, "document.getElementById('btn-start').textContent") == "Start Upscaling"
            outs_js = ("[...document.querySelectorAll('#queue-body td.out')]"
                       ".map(td => td.textContent).join('|')")
            # Rows appear first; the planned sizes follow from the (debounced) plan request.
            assert pump(app, lambda: run_js(app, view, outs_js) == "256 × 384|384 × 256", 10), \
                run_js(app, view, outs_js)
            outs = run_js(app, view, outs_js)
            print(f"  page loaded in Qt WebEngine; previewed outputs: {outs}")

            # Explorer drop on the page -> JS -> Django.
            mime = _mime(dropped)      # the event doesn't own it: keep it alive
            page.dropEvent(QDropEvent(QPointF(5, 5), Qt.CopyAction, mime,
                                      Qt.LeftButton, Qt.NoModifier))
            assert pump(app, lambda: any(i.path == dropped for i in get_workspace().items), 10)
            assert pump(app, lambda: run_js(app, view,
                        "document.querySelectorAll('#queue-body tr').length") == 3, 10)
            print("  Explorer drop -> page -> Django queue OK")

            out_dir = tmp / "out"
            run_js(app, view, f"document.getElementById('output-dir').value = {json.dumps(str(out_dir))};")
            snap(app, win, shots / "upscaler_scale.png")

            # Exact frame of a different shape: Start is refused until Pad/Crop is picked.
            # (Each snippet in its own block: page scripts share one global scope.)
            run_js(app, view, """{
                document.getElementById('mode-exact').click();
                const w = document.getElementById('exact-width'), h = document.getElementById('exact-height');
                w.value = 400; h.value = 400;
                w.dispatchEvent(new Event('input')); h.dispatchEvent(new Event('input'));
            }""")
            assert pump(app, lambda: run_js(app, view,
                        "!document.getElementById('conflict-box').hidden") is True, 10)
            text = run_js(app, view, "document.getElementById('conflict-text').textContent")
            assert text.startswith("2 of 3 image(s) have a different shape from 400 × 400"), text
            snap(app, win, shots / "upscaler_conflict.png")
            calls["warning"].clear()
            run_js(app, view, "document.getElementById('btn-start').click();")
            assert pump(app, lambda: bool(calls["warning"]), 10)
            assert "Choose Pad" in calls["warning"][-1][1], calls["warning"]
            assert get_workspace().job is None, "no batch may start without a choice"
            print("  exact 400x400: conflict box shown, Start refused until Pad/Crop is chosen")

            run_js(app, view, "document.querySelector('input[name=fit][value=pad]').click();")
            pump(app, lambda: False, 0.5)
            calls["info"].clear()
            run_js(app, view, "document.getElementById('btn-start').click();")
            assert pump(app, lambda: bool(calls["info"]), 30), "no completion message"
            assert calls["info"][-1] == ("Done", "Upscaled 3 image(s)."), calls["info"]
            with Image.open(out_dir / "portrait_400x400.png") as im:
                assert im.size == (400, 400)
                px = im.convert("RGB").getpixel((5, 200))
                assert px == (255, 255, 255), px      # white border beside the portrait
            with Image.open(out_dir / "dropped_400x400.png") as im:
                assert im.size == (400, 400)          # square matched the frame: no border
            with Image.open(folder / "upscaled" / "landscape_400x400.png") as im:
                assert im.size == (400, 400)
            snap(app, win, shots / "upscaler_done.png")

            # A new frame size needs a new decision: the old Pad choice is cleared.
            run_js(app, view, """{
                const w = document.getElementById('exact-width');
                w.value = 401; w.dispatchEvent(new Event('input'));
            }""")
            assert pump(app, lambda: run_js(app, view,
                        "!document.querySelector('input[name=fit]:checked')") is True, 10), \
                run_js(app, view, "JSON.stringify({w: document.getElementById('exact-width').value,"
                                  " fit: document.querySelector('input[name=fit]:checked')?.value,"
                                  " log: document.getElementById('log').textContent.slice(-400)})")
            print("  changing the frame afterwards cleared the Pad choice (asked again)")
            print("  Pad chosen -> batch ran from the page, completion message via Qt, "
                  "files written with borders only where the shape differed")
            print(f"  screenshots: {shots}")

        # Metadata page still independent and working alongside.
        win.act_metadata.trigger()
        assert win.stack.currentWidget() is win.metadata_page
        win.act_upscale.trigger()
        assert win.stack.currentWidget() is page

        win.close()
        assert server._active is None, "closing the window stops the server"
        print("  closing the window stopped the batch thread and the server")

    reset_workspace()
    print("UPSCALER DESKTOP TEST PASSED")
    return 0


def _mime(path: Path) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    return mime


if __name__ == "__main__":
    raise SystemExit(main())
