"""Thumbnails and large previews for the Upscaler page.

Made on demand when the page first shows an image (the page lazy-loads them,
so a queue of hundreds only pays for what is on screen): JPEGs decode at a
reduced size via Pillow's draft mode, EXIF rotation is applied, and the
result is a small WebP (keeps transparency) cached in memory by path, size
and modification time.
"""

from __future__ import annotations

import io
import threading
from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageOps

from .imaging import _to_8bit

THUMB_EDGE = 160
PREVIEW_EDGE = 960
_LIMITS = {THUMB_EDGE: 1500, PREVIEW_EDGE: 40}     # cached entries per size

_cache: dict[int, OrderedDict] = {edge: OrderedDict() for edge in _LIMITS}
_lock = threading.Lock()


def _render(path: Path, edge: int) -> bytes:
    with Image.open(path) as im:
        if im.format == "JPEG":
            im.draft("RGB", (edge, edge))        # decode at 1/2, 1/4 or 1/8 size
        im = _to_8bit(ImageOps.exif_transpose(im))     # 16-bit greys shouldn't clip white
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA" if "A" in im.getbands() or "transparency" in im.info else "RGB")
        im.thumbnail((edge, edge), Image.Resampling.BILINEAR, reducing_gap=2.0)
        out = io.BytesIO()
        im.save(out, "WEBP", quality=82 if edge == THUMB_EDGE else 86, method=2)
        return out.getvalue()


def image_bytes(path: Path, edge: int) -> bytes:
    """WebP bytes of *path* scaled to fit an ``edge`` × ``edge`` box."""
    path = Path(path)
    key = (str(path), path.stat().st_mtime_ns)
    cache = _cache[edge]
    with _lock:
        data = cache.get(key)
        if data is not None:
            cache.move_to_end(key)
            return data
    data = _render(path, edge)
    with _lock:
        cache[key] = data
        while len(cache) > _LIMITS[edge]:
            cache.popitem(last=False)
    return data
