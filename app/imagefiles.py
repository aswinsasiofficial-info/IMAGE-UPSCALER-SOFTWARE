"""Finding supported images — shared by the Upscaler and Metadata screens.

Folders are scanned recursively, skipping the ``upscaled`` folder the
Upscaler writes inside a folder it processed, so its results are never
picked up again as inputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
OUTPUT_DIR_NAME = "upscaled"


def path_key(path: Path) -> str:
    """Identity of a file on disk (case-insensitive on Windows)."""
    try:
        path = Path(path).resolve()
    except OSError:
        path = Path(path).absolute()
    return os.path.normcase(str(path))


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()


def _inside_output_dir(child: Path, root: Path) -> bool:
    """True for files under ``root/upscaled`` (results of an earlier run)."""
    try:
        rel = child.relative_to(root)
    except ValueError:
        return False
    return len(rel.parts) > 1 and rel.parts[0].lower() == OUTPUT_DIR_NAME


def discover(paths: Iterable[Path]) -> list[Path]:
    """Expand files and folders into a de-duplicated, order-preserving image list."""
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found.extend(c for c in sorted(p.rglob("*"))
                         if is_image(c) and not _inside_output_dir(c, p))
        elif is_image(p):
            found.append(p)
    seen: set[str] = set()
    unique: list[Path] = []
    for f in found:
        key = path_key(f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def file_dialog_filter() -> str:
    exts = " ".join(f"*{e}" for e in sorted(IMAGE_SUFFIXES))
    return f"Images ({exts});;All files (*)"
