"""Finding input images and deciding where each result is written.

Keeps the behaviour Upscaler users already rely on:
  * folders are scanned recursively, skipping the ``upscaled`` folder that
    earlier runs created inside them;
  * images that came from an added folder save to
    ``<folder>/upscaled/<same relative subfolder>/``;
  * other images save beside the original or into the chosen output folder;
  * nothing is ever overwritten — clashes get ``(1)``, ``(2)``, … suffixes;
  * optional batch rename: one random 4-5 letter prefix + 1, 2, 3 … per folder.
"""

from __future__ import annotations

import os
import random
import string
from pathlib import Path
from typing import Iterable

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
OUTPUT_DIR_NAME = "upscaled"
DEFAULT_OUTPUT_DIR = Path.home() / "Upscaled"
INVALID_NAME_CHARS = frozenset('<>:"/\\|?*')


def path_key(path: Path) -> str:
    """Identity of a file on disk (case-insensitive on Windows)."""
    try:
        path = path.resolve()
    except OSError:
        path = path.absolute()
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


def find_root(src: Path, roots: Iterable[Path]) -> tuple[Path, Path] | None:
    """(root, subfolder relative to it) for the deepest added folder holding *src*."""
    try:
        resolved = src.resolve()
    except OSError:
        return None
    best: tuple[Path, Path] | None = None
    for root in roots:
        try:
            rel = resolved.relative_to(root.resolve())
        except (ValueError, OSError):
            continue
        if best is None or len(root.resolve().parts) > len(best[0].parts):
            best = (root.resolve(), rel.parent)
    return best


def output_dir_for(src: Path, roots: Iterable[Path], save_beside: bool, output_dir: Path) -> Path:
    """Folder mode first, then beside-the-original, then the chosen output folder."""
    match = find_root(src, roots)
    if match is not None:
        root, rel_parent = match
        return root / OUTPUT_DIR_NAME / rel_parent
    return src.parent if save_beside else output_dir


def extension(output_format: str) -> str:
    return ".jpg" if output_format == "jpg" else ".png"


def output_filename(src: Path, suffix: str, output_format: str) -> str:
    return f"{src.stem}{suffix}{extension(output_format)}"


def unique_path(path: Path) -> Path:
    """*path*, or ``stem(1).ext``, ``stem(2).ext`` … if it already exists."""
    if not path.exists():
        return path
    n = 1
    while True:
        candidate = path.with_name(f"{path.stem}({n}){path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


class SequentialNamer:
    """``<PREFIX><n>.<ext>`` names for batch rename, numbered per output folder."""

    def __init__(self, prefix: str | None = None) -> None:
        self.prefix = prefix or "".join(
            random.choice(string.ascii_uppercase) for _ in range(random.randint(4, 5)))
        self._counters: dict[str, int] = {}

    def next_path(self, folder: Path, ext: str) -> Path:
        key = path_key(folder)
        n = self._counters.get(key, 0) + 1
        while (folder / f"{self.prefix}{n}{ext}").exists():
            n += 1
        self._counters[key] = n
        return folder / f"{self.prefix}{n}{ext}"

    def release(self, folder: Path) -> None:
        """Give back the last number in *folder* after a failed image (no gaps)."""
        key = path_key(folder)
        if self._counters.get(key, 0) > 0:
            self._counters[key] -= 1
