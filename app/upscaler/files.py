"""Deciding where each Upscaler result is written, and under which name.

  * images that came from an added folder save to
    ``<folder>/upscaled/<same relative subfolder>/``;
  * other images save beside the original or into the chosen output folder;
  * batch rename is the default: one random 4-5 letter prefix shared by the
    batch + 1, 2, 3 … per output folder (e.g. KQATZ1.jpg, KQATZ2.jpg);
  * nothing is ever overwritten — taken numbers are skipped, and kept
    original names get ``(1)``, ``(2)``, … when they clash.

Finding the input images is shared with the Metadata screen (app/imagefiles.py).
"""

from __future__ import annotations

import random
import string
from pathlib import Path
from typing import Iterable

from app.imagefiles import IMAGE_SUFFIXES, OUTPUT_DIR_NAME, discover, path_key  # noqa: F401

DEFAULT_OUTPUT_DIR = Path.home() / "Upscaled"


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


def output_filename(src: Path, output_format: str) -> str:
    """The original name with the output format's extension (batch rename off)."""
    return f"{src.stem}{extension(output_format)}"


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
