"""Pixel handling for the Upscaler: probing, loading, composing and saving.

Nothing here distorts or crops an image: the network's 4× result is resized
to the planned size with one factor for both axes, and saved whole.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from .geometry import Plan

EXIF_ORIENTATION = 0x0112
_ROTATED_ORIENTATIONS = {5, 6, 7, 8}   # EXIF values that swap width and height
_ALPHA_MODES = {"RGBA", "LA", "PA", "RGBa", "La"}
_HIGH_BIT_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}


@dataclass(frozen=True)
class ImageInfo:
    width: int          # as displayed, i.e. after EXIF rotation
    height: int
    has_alpha: bool


@dataclass
class SourceImage:
    rgb: np.ndarray                 # (H, W, 3) uint8, upright
    alpha: Optional[Image.Image]    # "L" mask or None
    icc_profile: Optional[bytes]

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


def _has_alpha(im: Image.Image) -> bool:
    return im.mode in _ALPHA_MODES or (im.mode == "P" and "transparency" in im.info)


def probe(path: Path) -> ImageInfo:
    """Upright size without decoding pixels (EXIF rotation honoured)."""
    with Image.open(path) as im:
        width, height = im.size
        orientation = 1
        # PNG's getexif() decodes the whole image when there's no eXIf chunk
        # before the pixel data, so only ask when the header already has one.
        if im.format != "PNG" or "exif" in im.info:
            try:
                orientation = im.getexif().get(EXIF_ORIENTATION, 1)
            except Exception:  # malformed EXIF must not hide the image
                pass
        if orientation in _ROTATED_ORIENTATIONS:
            width, height = height, width
        return ImageInfo(width, height, _has_alpha(im))


def _to_8bit(im: Image.Image) -> Image.Image:
    """16-bit / float greyscale -> 8-bit, instead of letting PIL clip to white."""
    if im.mode not in _HIGH_BIT_MODES:
        return im
    arr = np.asarray(im, dtype=np.float64)
    if im.mode == "F":
        arr = arr * 255.0 if arr.max(initial=0.0) <= 1.0 else arr
    elif arr.max(initial=0.0) > 255:
        arr = arr / 257.0
    return Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8))   # -> "L"


def _rgb_profile(icc: Optional[bytes]) -> Optional[bytes]:
    """Keep an embedded ICC profile only if it describes RGB data."""
    return icc if icc and len(icc) >= 20 and icc[16:20] == b"RGB " else None


def load(path: Path) -> SourceImage:
    with Image.open(path) as im:
        im.load()
        icc = im.info.get("icc_profile")
        im = _to_8bit(ImageOps.exif_transpose(im))
        alpha = None
        if _has_alpha(im):
            rgba = im.convert("RGBA")
            alpha = rgba.getchannel("A")
            rgb = rgba.convert("RGB")
        else:
            if im.mode not in ("RGB", "L", "P"):
                icc = None       # CMYK/LAB profiles don't apply after conversion
            rgb = im.convert("RGB")
        return SourceImage(np.array(rgb, dtype=np.uint8), alpha, _rgb_profile(icc))


def compose(ai_rgb: np.ndarray, alpha: Optional[Image.Image], plan: Plan) -> Image.Image:
    """Turn the network's 4× output into the planned final image (whole picture)."""
    img = Image.fromarray(ai_rgb)       # (H, W, 3) uint8 -> "RGB"
    if alpha is not None:
        img.putalpha(alpha.resize(img.size, Image.Resampling.BICUBIC))
    if img.size != plan.output:
        # 1-3× reduce and 5× enlarge the 4× result with the same factor on both
        # axes; Pillow resizes RGBA premultiplied, so edges stay clean.
        img = img.resize(plan.output, Image.Resampling.LANCZOS)
    return img


def save(img: Image.Image, path: Path, output_format: str, jpg_quality: int = 95,
         icc_profile: Optional[bytes] = None) -> None:
    """Write atomically (temp file + rename) so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    extra = {"icc_profile": icc_profile} if icc_profile else {}
    try:
        if output_format == "jpg":
            if img.mode in ("RGBA", "LA"):
                # JPEG has no transparency; flatten onto white as the app always has.
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img.convert("RGBA"))
            img.convert("RGB").save(tmp, "JPEG", quality=int(jpg_quality), optimize=True, **extra)
        else:
            img.save(tmp, "PNG", **extra)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def saved_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size
