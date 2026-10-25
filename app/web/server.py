"""Embedded Django server for the desktop app.

Serves the Upscaler on 127.0.0.1 at a free port picked by the OS, from
daemon threads inside this process — no separate service to install, and
nothing reachable from other machines. Each launch gets a fresh secret key
and launch token (see app/web/middleware.py).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

SETTINGS_MODULE = "app.web.settings"


def configure() -> None:
    """Point Django at the app's settings and initialise it (idempotent)."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", SETTINGS_MODULE)
    import django

    django.setup()


@dataclass
class LocalServer:
    httpd: object
    thread: threading.Thread
    url: str            # http://127.0.0.1:<port>/
    launch_url: str     # url + this launch's access token

    def stop(self) -> None:
        from app.web.middleware import clear_launch_token

        self.httpd.shutdown()
        self.httpd.server_close()
        clear_launch_token()


_active: Optional[LocalServer] = None


def start(host=None) -> LocalServer:
    """Start (once) and return the local server; *host* lends it native dialogs."""
    global _active
    if _active is not None:
        return _active
    configure()

    from django.contrib.staticfiles.handlers import StaticFilesHandler
    from django.core.handlers.wsgi import WSGIHandler
    from django.core.servers.basehttp import ThreadedWSGIServer, WSGIRequestHandler

    from app.upscaler import host as host_registry
    from app.web.middleware import new_launch_token

    host_registry.register(host)
    httpd = ThreadedWSGIServer(("127.0.0.1", 0), WSGIRequestHandler,
                               ipv6=False, allow_reuse_address=False)
    # StaticFilesHandler serves the page's CSS/JS without DEBUG, as runserver does.
    httpd.set_app(StaticFilesHandler(WSGIHandler()))
    thread = threading.Thread(target=httpd.serve_forever, name="upscaler-django", daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    _active = LocalServer(httpd, thread, url, f"{url}?launch={new_launch_token()}")
    return _active


def stop() -> None:
    global _active
    if _active is not None:
        _active.stop()
        _active = None
