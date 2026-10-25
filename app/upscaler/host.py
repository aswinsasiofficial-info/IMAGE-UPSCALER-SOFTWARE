"""Hooks the desktop shell lends to the Django Upscaler.

A web page can't open a native file dialog that returns real paths, show a
Windows message box, or know the app's Qt palette. When the Upscaler runs
inside the desktop app, ``app/ui/upscaler_web_page.py`` registers a host that
can. When it runs standalone (``python manage.py runserver``) there is no
host, and the page falls back to typed paths and in-page messages.
"""

from __future__ import annotations

from typing import Optional, Protocol


class DesktopHost(Protocol):
    def pick_images(self) -> list[str]: ...
    def pick_folder(self) -> Optional[str]: ...
    def pick_output_folder(self, current: str) -> Optional[str]: ...
    def show_message(self, kind: str, title: str, text: str) -> None: ...
    def theme(self) -> dict[str, str]: ...


_host: Optional[DesktopHost] = None


def register(host: Optional[DesktopHost]) -> None:
    global _host
    _host = host


def current() -> Optional[DesktopHost]:
    return _host
