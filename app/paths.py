"""Where the app finds its bundled files and keeps its data.

The same code runs from a source checkout (``python -m app.main``) and from
the packaged desktop build (PyInstaller one-folder app, ``BatchUpscaler.exe``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
# Folder holding BatchUpscaler.exe, or the project root when run from source.
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parents[1]
# Read-only bundled files (fonts, icons). PyInstaller unpacks them under _MEIPASS.
ASSETS_DIR = (Path(getattr(sys, "_MEIPASS", APP_DIR)) / "app" / "assets") if FROZEN \
    else Path(__file__).resolve().parent / "assets"

APP_NAME = "BatchUpscaler"
WEIGHTS_FILE = "RealESRGAN_x4plus.pth"


def asset(*parts: str) -> Path:
    return ASSETS_DIR.joinpath(*parts)


def user_data_dir() -> Path:
    """Per-user writable folder (%LOCALAPPDATA%\\BatchUpscaler)."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir() -> Path:
    """Where the Real-ESRGAN weights are read from (and downloaded to).

    ``UPSCALER_MODEL_DIR`` overrides; otherwise ``models`` next to the app
    (the project's models folder from source, or weights shipped with the
    packaged app); a packaged app without weights downloads them into the
    user's data folder, since Program Files may not be writable.
    """
    override = os.environ.get("UPSCALER_MODEL_DIR")
    if override:
        return Path(override)
    beside = APP_DIR / "models"
    if not FROZEN or (beside / WEIGHTS_FILE).is_file():
        return beside
    return user_data_dir() / "models"
