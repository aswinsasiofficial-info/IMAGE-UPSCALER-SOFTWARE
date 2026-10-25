"""Icons (Lucide, ISC license — app/assets/icons) and icon buttons.

SVGs are recoloured for the current theme and rendered at several sizes, so
they stay sharp at any display scaling. Every icon-only button gets a tooltip
(what it does, plus its shortcut) and an accessible name and description for
screen readers — :func:`icon_button` makes that impossible to forget.
"""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QIcon, QImage, QKeySequence, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QToolButton, QWidget

from app import design, paths
from app.ui import theme

_SIZES = (16, 20, 24, 32, 40, 48, 64)


@lru_cache(maxsize=None)
def _svg(name: str) -> str:
    return paths.asset("icons", f"{name}.svg").read_text(encoding="utf-8")


def _pixmap(name: str, color: str, size: int) -> QPixmap:
    renderer = QSvgRenderer(QByteArray(_svg(name).replace("currentColor", color).encode("utf-8")))
    image = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter)
    painter.end()
    return QPixmap.fromImage(image)


@lru_cache(maxsize=None)
def _icon(name: str, color: str, disabled: str) -> QIcon:
    icon = QIcon()
    for size in _SIZES:
        icon.addPixmap(_pixmap(name, color, size), QIcon.Normal)
        icon.addPixmap(_pixmap(name, disabled, size), QIcon.Disabled)
    return icon


def icon(name: str, role: str = "text", scheme: str | None = None) -> QIcon:
    """Theme-coloured icon. ``role``: "text", "muted", "accent", "on-primary"."""
    c = design.colors(scheme or theme.current_scheme())
    color = {"text": c["text"], "muted": c["text-muted"], "accent": c["accent"],
             "on-primary": c["primary-text"], "danger": c["danger"]}[role]
    return _icon(name, color, c["text-faint"])


def icon_button(name: str, tooltip: str, *, accessible_name: str | None = None,
                shortcut: str | None = None, text: str | None = None,
                variant: str | None = None, checkable: bool = False,
                parent: QWidget | None = None) -> QToolButton:
    """A themed tool button; icon-only unless ``text`` is given.

    ``tooltip`` explains the action; the accessible name defaults to its first
    sentence, and the shortcut (if any) is appended to the tooltip.
    """
    btn = QToolButton(parent)
    btn.setProperty("button", True)
    btn.setProperty("iconName", name)
    if variant:
        btn.setProperty("variant", variant)
    btn.setIcon(icon(name, "on-primary" if variant == "primary" else "text"))
    btn.setIconSize(QSize(18, 18))
    btn.setCheckable(checkable)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFocusPolicy(Qt.StrongFocus)
    if text:
        btn.setText(text)
        btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    else:
        btn.setProperty("iconOnly", True)
        btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
    name_text = accessible_name or tooltip.split(" — ")[0].rstrip(".")
    btn.setAccessibleName(name_text)
    btn.setAccessibleDescription(tooltip)
    if shortcut:
        btn.setShortcut(QKeySequence(shortcut))
        native = QKeySequence(shortcut).toString(QKeySequence.NativeText)
        tooltip = f"{tooltip} ({native})"
    btn.setToolTip(tooltip)
    return btn


def refresh_icons(root: QWidget) -> None:
    """Re-colour every icon button under *root* after a light/dark switch."""
    for btn in root.findChildren(QToolButton):
        name = btn.property("iconName")
        if name:
            role = "on-primary" if btn.property("variant") == "primary" else "text"
            btn.setIcon(icon(name, role))
