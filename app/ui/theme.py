"""Qt side of the shared design (app/design.py): palette + stylesheet.

The desktop follows the system's light/dark setting (Qt 6.5+), and re-themes
live when it changes; ``UPSCALER_THEME=light|dark`` forces one. Widgets opt
into roles with dynamic properties instead of per-widget styles:

  * ``QFrame[card="true"]``            white (or dark) rounded panel
  * ``QPushButton[variant="primary"]`` main action; ``"ghost"`` quiet action
  * ``QLabel[role="title"|"section"|"muted"|"hint"]``  type styles
  * ``QFrame[banner="success"|"warning"|"danger"|"info"]`` inline messages
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

from app import design

_ICON_DIR = Path(tempfile.gettempdir()) / "batch-upscaler-ui"


def system_scheme() -> str:
    forced = os.environ.get("UPSCALER_THEME", "").lower()
    if forced in (design.LIGHT, design.DARK):
        return forced
    try:
        dark = QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:      # Qt < 6.5: no scheme API
        dark = False
    return design.DARK if dark else design.LIGHT


def current_scheme() -> str:
    app = QApplication.instance()
    value = app.property("uiScheme") if app is not None else None
    return design.normalize_scheme(value) if value else system_scheme()


def _arrow(name: str, color: str, points: str, size: int = 10, stroke: float = 1.6) -> str:
    """Small SVG line icon (arrows, check mark) — QSS needs a file URL."""
    _ICON_DIR.mkdir(parents=True, exist_ok=True)
    path = _ICON_DIR / f"{name}-{color.lstrip('#')}.svg"
    if not path.exists():
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 {size} {size}"><polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round"/></svg>',
            encoding="utf-8")
    return path.as_posix()


def palette(scheme: str) -> QPalette:
    c = design.colors(scheme)
    pal = QPalette()
    roles = {
        QPalette.Window: c["bg"], QPalette.WindowText: c["text"],
        QPalette.Base: c["surface"], QPalette.AlternateBase: c["surface-alt"],
        QPalette.Text: c["text"], QPalette.Button: c["surface"],
        QPalette.ButtonText: c["text"], QPalette.Highlight: c["primary"],
        QPalette.HighlightedText: c["primary-text"], QPalette.ToolTipBase: c["surface"],
        QPalette.ToolTipText: c["text"], QPalette.PlaceholderText: c["text-faint"],
        QPalette.Link: c["accent"], QPalette.Light: c["surface"],
        QPalette.Midlight: c["border"], QPalette.Mid: c["border-strong"],
        QPalette.Dark: c["border-strong"], QPalette.Shadow: "#000000",
    }
    for role, value in roles.items():
        pal.setColor(role, QColor(value))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(c["text-faint"]))
    return pal


def stylesheet(scheme: str) -> str:
    c = design.colors(scheme)
    down = _arrow("down", c["text-muted"], "2,3.5 5,6.5 8,3.5")
    up = _arrow("up", c["text-muted"], "2,6.5 5,3.5 8,6.5")
    check = _arrow("check", c["primary-text"], "3.5,8.5 6.5,11.5 12.5,5", size=16, stroke=2)
    r, rs = design.RADIUS, design.RADIUS_SM
    return f"""
QWidget {{ color: {c['text']}; font-family: "{design.FONT_FAMILY}"; font-size: {design.FONT_SIZE_PT}pt; }}
QMainWindow, QStackedWidget, QWidget#page, QScrollArea, QScrollArea > QWidget > QWidget#scrollBody {{ background: {c['bg']}; }}
QToolTip {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 6px 8px; border-radius: {rs}px; }}

QFrame[card="true"] {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {r}px; }}
QFrame[card="true"] QLabel, QFrame[card="true"] QCheckBox, QFrame[card="true"] QRadioButton {{ background: transparent; }}
QFrame[divider="true"] {{ background: {c['border']}; border: none; max-height: 1px; min-height: 1px; }}
QLabel[role="title"] {{ font-size: 14pt; font-weight: 600; }}
QLabel[role="section"] {{ font-size: 11pt; font-weight: 600; }}
QLabel[role="muted"] {{ color: {c['text-muted']}; }}
QLabel[role="hint"] {{ color: {c['text-muted']}; font-size: 9pt; }}
QLabel[role="counter"] {{ color: {c['text-faint']}; font-size: 9pt; }}
QLabel[role="counter"][over="true"] {{ color: {c['danger']}; font-weight: 600; }}
QLabel[role="error"] {{ color: {c['danger']}; }}

QPushButton, QToolButton[button="true"] {{
    background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border-strong']};
    border-radius: {rs}px; padding: 6px 14px; min-height: 18px; }}
QPushButton:hover, QToolButton[button="true"]:hover {{ background: {c['hover']}; }}
QPushButton:pressed, QToolButton[button="true"]:pressed {{ background: {c['border']}; }}
QPushButton:disabled, QToolButton[button="true"]:disabled {{ color: {c['text-faint']}; background: {c['surface-alt']}; border-color: {c['border']}; }}
QPushButton:focus, QToolButton[button="true"]:focus {{ border-color: {c['accent']}; }}
QPushButton[variant="primary"], QToolButton[variant="primary"] {{
    background: {c['primary']}; color: {c['primary-text']}; border: 1px solid {c['primary']}; font-weight: 600; }}
QPushButton[variant="primary"]:hover, QToolButton[variant="primary"]:hover {{ background: {c['primary-hover']}; border-color: {c['primary-hover']}; }}
QPushButton[variant="primary"]:disabled, QToolButton[variant="primary"]:disabled {{ background: {c['primary-soft']}; color: {c['text-faint']}; border-color: {c['primary-soft']}; }}
QPushButton[variant="ghost"] {{ background: transparent; border-color: transparent; color: {c['text-muted']}; padding: 4px 10px; }}
QPushButton[variant="ghost"]:hover {{ background: {c['hover']}; color: {c['text']}; }}
QPushButton[variant="danger"] {{ color: {c['danger']}; }}
QPushButton[segment="true"] {{ border-radius: {rs}px; padding: 4px 10px; color: {c['text-muted']}; background: transparent; border-color: transparent; }}
QPushButton[segment="true"]:hover {{ background: {c['hover']}; color: {c['text']}; }}
QPushButton[segment="true"]:checked {{ background: {c['primary-soft']}; color: {c['accent']}; font-weight: 600; }}
QToolButton[variant="primary"]::menu-button {{ border: none; border-left: 1px solid {c['primary-hover']}; width: 24px; }}
QToolButton::menu-arrow {{ image: url({_arrow("down", c['primary-text'], "2,3.5 5,6.5 8,3.5")}); width: 10px; height: 10px; }}
QToolButton#exportButton {{ padding-left: 18px; padding-right: 34px; }}

QCheckBox, QRadioButton {{ spacing: 8px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; border: 1px solid {c['border-strong']}; background: {c['surface']}; }}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {c['accent']}; }}
QCheckBox::indicator:checked {{ background: {c['primary']}; border-color: {c['primary']}; image: url({check}); }}
QRadioButton::indicator:checked {{ background: {c['primary-text']}; border: 5px solid {c['primary']}; }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ background: {c['surface-alt']}; border-color: {c['border']}; }}
QCheckBox::indicator:checked:disabled {{ background: {c['primary-soft']}; border-color: {c['primary-soft']}; }}
QScrollArea#cardScroll, QScrollArea#cardScroll > QWidget, QWidget#cardBody {{ background: transparent; }}

QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {{
    background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border-strong']};
    border-radius: {rs}px; padding: 5px 8px; selection-background-color: {c['primary']};
    selection-color: {c['primary-text']}; }}
QPlainTextEdit {{ padding: 4px 6px; }}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus {{ border-color: {c['accent']}; }}
QLineEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{
    background: {c['surface-alt']}; color: {c['text-faint']}; border-color: {c['border']}; }}
QPlainTextEdit[readOnly="true"] {{ background: {c['surface-alt']}; }}
QComboBox {{ padding-right: 24px; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url({down}); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border']};
    selection-background-color: {c['selection']}; selection-color: {c['text']}; outline: none; padding: 4px; }}
QSpinBox {{ padding-right: 20px; }}
QSpinBox::up-button, QSpinBox::down-button {{ border: none; width: 18px; background: transparent; }}
QSpinBox::up-arrow {{ image: url({up}); width: 9px; height: 9px; }}
QSpinBox::down-arrow {{ image: url({down}); width: 9px; height: 9px; }}

QProgressBar {{ background: {c['surface-alt']}; border: 1px solid {c['border']}; border-radius: {rs}px;
    min-height: 10px; max-height: 10px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {c['primary']}; border-radius: {rs - 1}px; }}

QListWidget, QListView {{ background: {c['surface']}; border: none; outline: none; }}
QListWidget::item:selected, QListView::item:selected {{ background: transparent; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {c['text-muted']}; padding: 8px 14px; border: none;
    border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:selected {{ color: {c['text']}; border-bottom-color: {c['primary']}; }}
QTabBar::tab:hover {{ color: {c['text']}; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 10px; }}
QSplitter::handle:vertical {{ height: 10px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{ background: {c['border-strong']}; border-radius: 3px; }}
QScrollBar::handle:vertical {{ min-height: 32px; }}
QScrollBar::handle:horizontal {{ min-width: 32px; }}
QScrollBar::handle:hover {{ background: {c['text-faint']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QToolBar#navBar {{ background: {c['surface']}; border: none; border-bottom: 1px solid {c['border']}; padding: 6px 12px; spacing: 4px; }}
QToolBar#navBar QToolButton {{ border: 1px solid transparent; border-radius: {rs}px; padding: 6px 14px;
    color: {c['text-muted']}; font-weight: 600; }}
QToolBar#navBar QToolButton:hover {{ background: {c['hover']}; color: {c['text']}; }}
QToolBar#navBar QToolButton:checked {{ background: {c['primary-soft']}; color: {c['accent']}; }}
QLabel#appTitle {{ font-size: 11.5pt; font-weight: 700; padding-right: 14px; }}
QStatusBar {{ background: {c['surface']}; border-top: 1px solid {c['border']}; color: {c['text-muted']}; }}
QStatusBar QLabel {{ color: {c['text-muted']}; }}
QMenu {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {rs}px; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {c['selection']}; color: {c['text']}; }}
QMessageBox {{ background: {c['bg']}; }}

QFrame[banner="info"] {{ background: {c['primary-soft']}; border: 1px solid {c['selection']}; border-radius: {rs}px; }}
QFrame[banner="success"] {{ background: {c['success-soft']}; border: 1px solid {c['success']}; border-radius: {rs}px; }}
QFrame[banner="warning"] {{ background: {c['warning-soft']}; border: 1px solid {c['warning-border']}; border-radius: {rs}px; }}
QFrame[banner="danger"] {{ background: {c['danger-soft']}; border: 1px solid {c['danger']}; border-radius: {rs}px; }}
QFrame[banner] QLabel {{ background: transparent; }}
"""


def apply(app: QApplication | None = None, scheme: str | None = None) -> str:
    """Theme the whole application; returns the scheme used."""
    app = app or QApplication.instance()
    scheme = design.normalize_scheme(scheme or system_scheme())
    app.setStyle("Fusion")
    app.setPalette(palette(scheme))
    font = QFont(design.FONT_FAMILY)
    font.setPointSizeF(design.FONT_SIZE_PT)
    app.setFont(font)
    app.setStyleSheet(stylesheet(scheme))
    app.setProperty("uiScheme", scheme)
    return scheme


def ensure_applied() -> str:
    """Apply the theme once (windows built outside app.main, e.g. in tests)."""
    app = QApplication.instance()
    if app.property("uiScheme"):
        return design.normalize_scheme(app.property("uiScheme"))
    return apply(app)


def repolish(widget) -> None:
    """Re-evaluate the stylesheet after changing a dynamic property."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()
