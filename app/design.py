"""Design tokens for the desktop app: one palette (light and dark), type
scale and set of radii shared by every screen. ``app/ui/theme.py`` turns
these into the Qt palette + stylesheet. No Qt imports here.
"""

from __future__ import annotations

LIGHT, DARK = "light", "dark"

FONT_FAMILY = "Inter Tight"    # bundled: app/assets/fonts (SIL Open Font License)
FONT_FILE = "InterTight-Variable.ttf"
FONT_SIZE_PT = 10              # body text; small text is 9pt, headings 11–14pt
RADIUS = 8                     # cards, panels
RADIUS_SM = 6                  # buttons, inputs

COLORS: dict[str, dict[str, str]] = {
    LIGHT: {
        "bg": "#f3f4f6",            # window background
        "surface": "#ffffff",       # cards, inputs
        "surface-alt": "#f8f9fb",   # subtle fills (headers, disabled)
        "hover": "#eef1f5",
        "border": "#e0e4e9",
        "border-strong": "#c7cdd5",
        "text": "#1b1f24",
        "text-muted": "#5b6470",
        "text-faint": "#8b939d",
        "primary": "#2563eb",       # primary buttons, progress
        "primary-hover": "#1d4ed8",
        "primary-text": "#ffffff",
        "primary-soft": "#e7eefd",  # selected nav item, soft fills
        "accent": "#1d4ed8",        # accent text/focus on surfaces
        "selection": "#dbe6fd",
        "success": "#1a7f37",
        "success-soft": "#e3f4e8",
        "warning": "#8a5d00",
        "warning-soft": "#fff5d6",
        "warning-border": "#e3c15a",
        "danger": "#c9252f",
        "danger-soft": "#fdeceb",
        "picture-a": "#8db9e3",     # placeholder "photo" in shape diagrams
        "picture-b": "#3f7cbd",
    },
    DARK: {
        "bg": "#14161a",
        "surface": "#1d2026",
        "surface-alt": "#23272e",
        "hover": "#2a2f37",
        "border": "#31363f",
        "border-strong": "#454c58",
        "text": "#e7e9ec",
        "text-muted": "#a3abb5",
        "text-faint": "#747c87",
        "primary": "#2f6fe4",
        "primary-hover": "#3d7cf0",
        "primary-text": "#ffffff",
        "primary-soft": "#1e2b45",
        "accent": "#7aa7ff",
        "selection": "#24365a",
        "success": "#4ac262",
        "success-soft": "#1a2e21",
        "warning": "#e0b04a",
        "warning-soft": "#342b15",
        "warning-border": "#7d6526",
        "danger": "#f36d6d",
        "danger-soft": "#3a1f23",
        "picture-a": "#4f86b8",
        "picture-b": "#2b5d8f",
    },
}


def normalize_scheme(value: str | None) -> str:
    return DARK if (value or "").lower() == DARK else LIGHT


def colors(scheme: str) -> dict[str, str]:
    return COLORS[normalize_scheme(scheme)]
