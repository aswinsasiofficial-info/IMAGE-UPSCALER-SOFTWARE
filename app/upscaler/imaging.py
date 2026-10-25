"""Pixel handling for the Upscaler: probing, loading, composing and saving.

Nothing here distorts an image. Resizes use the plan's single scale factor
for both axes, padding only adds borders around the whole picture, and
cropping only happens when the plan records that the user chose it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from .geometry import CROP, PAD, Plan

EXIF_ORIENTATION = 0x0112
_ROTATED_ORIENTATIONS = {5, 6, 7, 8}   # EXIF values that swap width and height
PAD_COLORS = {
    "white": (255, 255, 255, 255),
    "black": (0, 0, 0, 255),
    "transparent": (0, 0, 0, 0),
}
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


def compose(ai_rgb: np.ndarray, alpha: Optional[Image.Image], plan: Plan,
            pad_color: str = "white") -> Image.Image:
    """Turn the 4x network output into the planned final image."""
    if not plan.resolved:
        raise ValueError("this image needs an explicit Pad or Crop choice")
    img = Image.fromarray(ai_rgb)       # (H, W, 3) uint8 -> "RGB"
    if alpha is not None:
        img.putalpha(alpha.resize(img.size, Image.Resampling.BICUBIC))
    if img.size != plan.content:
        # Same factor on both axes (from the plan); Pillow resizes RGBA premultiplied.
        img = img.resize(plan.content, Image.Resampling.LANCZOS)
    if plan.content == plan.output:
        return img

    ow, oh = plan.output
    if plan.fit == CROP:
        left, top = -plan.offset[0], -plan.offset[1]
        return img.crop((left, top, left + ow, top + oh))
    if plan.fit == PAD:
        color = PAD_COLORS[pad_color]
        mode = "RGBA" if img.mode == "RGBA" or color[3] < 255 else "RGB"
        canvas = Image.new(mode, plan.output, color if mode == "RGBA" else color[:3])
        canvas.paste(img.convert(mode), plan.offset)
        return canvas
    raise ValueError("plan has a different output size but no fit")


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
