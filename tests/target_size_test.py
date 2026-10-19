"""Focused tests for the exact-target-size (crop-to-fill) behaviour.

Covers everything the feature must guarantee, without running the (slow)
neural network:
  * every output is exactly the configured target dimensions;
  * the resize covers the target without stretching (aspect ratio kept);
  * the crop is centered (edge content is cut, middle content survives);
  * small 4x results are enlarged to fill the target;
  * transparent PNG output remains transparent (and JPEG still flattens).

Usage:
    .venv\\Scripts\\python -m tests.target_size_test
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.engine.realesrgan_upscaler import RealESRGANUpscaler, UpscaleSettings

fill = RealESRGANUpscaler.compute_fill_dimensions


def _check_cover_and_aspect(width: int, height: int, tw: int, th: int) -> None:
    """Shared invariants: fully covers the target, aspect ratio preserved."""
    fill_w, fill_h = fill(width, height, tw, th)
    assert fill_w >= tw, f"width {fill_w} does not cover target {tw}"
    assert fill_h >= th, f"height {fill_h} does not cover target {th}"
    # Uniform scale => aspect ratio drifts at most by rounding (< 1 px).
    assert abs(fill_w * height - fill_h * width) <= max(width, height), (
        f"aspect ratio drifted: {width}x{height} -> {fill_w}x{fill_h}")
    # The scale is the minimal covering one: max(tw/w, th/h), within rounding.
    expected_scale = max(tw / width, th / height)
    assert abs(fill_w / width - expected_scale) < 2 / width, "scale is not cover-minimal"


def test_fill_dimensions() -> int:
    failures = 0

    # 1. Wide source: height binds, both side margins get cropped.
    if fill(20000, 10000, 8000, 5000) == (10000, 5000):
        print("  OK  20000x10000 covers 8000x5000 as 10000x5000 (1000 px cropped/axis)")
    else:
        print(f"  FAIL 20000x10000 -> {fill(20000, 10000, 8000, 5000)} (expected 10000x5000)")
        failures += 1

    # 2. Tall source: width binds.
    if fill(10000, 20000, 8000, 5000) == (8000, 16000):
        print("  OK  10000x20000 covers 8000x5000 as 8000x16000")
    else:
        print(f"  FAIL 10000x20000 -> {fill(10000, 20000, 8000, 5000)} (expected 8000x16000)")
        failures += 1

    # 3. Small source: must be ENLARGED to cover the target.
    if fill(1200, 400, 8000, 5000) == (15000, 5000):
        print("  OK  1200x400 enlarges to 15000x5000 (fills, then crops)")
    else:
        print(f"  FAIL 1200x400 -> {fill(1200, 400, 8000, 5000)} (expected 15000x5000)")
        failures += 1

    # 4. Same aspect ratio as the target: exact cover, zero crop.
    if fill(12000, 7500, 8000, 5000) == (8000, 5000):
        print("  OK  16:10 source maps exactly onto 8000x5000 (no crop)")
    else:
        print(f"  FAIL 12000x7500 -> {fill(12000, 7500, 8000, 5000)} (expected 8000x5000)")
        failures += 1

    # 5. Assorted shapes must all cover the target with the ratio intact.
    for w, h in [(8001, 4999), (640, 480), (33, 9001), (5000, 5000), (1, 1)]:
        try:
            _check_cover_and_aspect(w, h, 8000, 5000)
            print(f"  OK  {w}x{h} covers 8000x5000, aspect kept")
        except AssertionError as exc:
            print(f"  FAIL {w}x{h}: {exc}")
            failures += 1

    return failures


def _make_rgba_array(w: int, h: int) -> np.ndarray:
    """RGBA uint8 array with a clear transparent region to prove alpha survives."""
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200  # reddish
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[: h // 2] = 255   # top half opaque
    alpha[h // 2:] = 0      # bottom half fully transparent
    return np.dstack([rgb, alpha])


def _upscaler(tw: int, th: int, fmt: str = "png") -> RealESRGANUpscaler:
    return RealESRGANUpscaler(UpscaleSettings(output_format=fmt, target_width=tw, target_height=th))


def test_exact_output_size() -> int:
    failures = 0
    tw, th = 400, 250
    up = _upscaler(tw, th)

    # Downscale case, upscale case, matching and mismatching ratios — ALL must
    # land on exactly the target size (this is the core promise).
    for w, h in [(1600, 800), (300, 200), (400, 250), (80, 900), (1024, 127)]:
        arr = _make_rgba_array(w, h)
        out, _ = up._fill_to_target(arr)
        got = (out.shape[1], out.shape[0])
        if got == (tw, th):
            print(f"  OK  {w}x{h} -> exactly {tw}x{th}")
        else:
            print(f"  FAIL {w}x{h} -> {got} (expected {tw}x{th})")
            failures += 1

    # Small 4x results are enlarged as needed (no smaller-than-target output).
    out, cropped = up._fill_to_target(_make_rgba_array(100, 60))
    if (out.shape[1], out.shape[0]) == (tw, th) and cropped:
        print(f"  OK  tiny 100x60 enlarged + cropped to {tw}x{th}")
    else:
        print(f"  FAIL tiny source enlarged -> {(out.shape[1], out.shape[0])}, cropped={cropped}")
        failures += 1

    # Already exact: untouched, no crop reported.
    same = _make_rgba_array(tw, th)
    out, cropped = up._fill_to_target(same)
    if np.array_equal(out, same) and not cropped:
        print(f"  OK  {tw}x{th} input passes through untouched (no crop)")
    else:
        print("  FAIL exact-size input was altered or wrongly flagged as cropped")
        failures += 1

    # Disabled (0 on either axis) keeps the 4x result as-is.
    up_off = _upscaler(0, 0)
    kept, cropped = up_off._fill_to_target(_make_rgba_array(64, 64))
    if kept.shape[:2] == (64, 64) and not cropped:
        print("  OK  target 0x0 disables crop-to-fill (64x64 untouched)")
    else:
        print(f"  FAIL disabled target altered image to {kept.shape[:2]}")
        failures += 1

    return failures


def test_center_crop() -> int:
    """Prove the crop is *centered*: outer content is cut, middle survives."""
    failures = 0
    # 100x50 source into a 40x40 target: scale = max(0.4, 0.8) = 0.8 ->
    # 80x40 cover -> crop x from 20 to 60 of the cover (== source x 25..75).
    tw, th = 40, 40
    up = _upscaler(tw, th)

    src = np.zeros((50, 100, 3), dtype=np.uint8)
    # White central column band (source x 20..80), black margins outside it.
    src[:, 20:80] = 255
    # Distinct 4-row red bands at the very top and bottom (full vertical extent).
    src[:4, :] = (255, 0, 0)
    src[-4:, :] = (255, 0, 0)

    out, cropped = up._fill_to_target(src)
    h, w = out.shape[:2]
    if (w, h) != (tw, th):
        print(f"  FAIL crop output size {w}x{h} != {tw}x{tw}")
        return 1
    if not cropped:
        print("  FAIL crop-to-fill did not report cropping for a 2:1 source")
        failures += 1

    def is_white(px: np.ndarray) -> bool:
        return bool(px[0] > 200 and px[1] > 200 and px[2] > 200)

    def is_red(px: np.ndarray) -> bool:
        return bool(px[0] > 180 and px[1] < 80 and px[2] < 80)

    # Left/right edge columns sit at the middle of the white band: the black
    # source margins must have been cropped away.
    mid_row = h // 2
    if is_white(out[mid_row, 0]) and is_white(out[mid_row, -1]):
        print("  OK  horizontal crop centered (black source edges cropped out)")
    else:
        print("  FAIL black edges survived the crop")
        failures += 1
    # Full vertical extent preserved: top/bottom rows still show the red bands.
    if is_red(out[0, mid_row]) and is_red(out[-1, mid_row]):
        print("  OK  vertical extent kept (top/bottom bands present after crop)")
    else:
        print("  FAIL top/bottom content lost — crop was not centered vertically")
        failures += 1

    # Symmetry: crop margins must be equal on both sides: sample two pixels
    # mirrored around the horizontal center of the cover and compare values.
    diffs = np.abs(out.astype(int) - np.flip(out, axis=1).astype(int))
    if int(diffs.max()) <= 2:
        print("  OK  output is left/right symmetric around the center crop")
    else:
        print(f"  FAIL crop not centered (max asymmetric delta {int(diffs.max())})")
        failures += 1

    return failures


def test_save_paths() -> int:
    failures = 0
    tw, th = 320, 200
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)

        # PNG: exact target size AND transparency after crop-to-fill.
        up = _upscaler(tw, th, "png")
        out, _ = up._fill_to_target(_make_rgba_array(1280, 640))
        png_path = out_dir / "filled.png"
        up._save(out, png_path)
        with Image.open(png_path) as im:
            mode, size = im.mode, im.size
            arr = np.asarray(im.convert("RGBA"))
        if size == (tw, th):
            print(f"  OK  saved PNG is exactly {tw}x{th}")
        else:
            print(f"  FAIL saved PNG size={size} (expected {(tw, th)})")
            failures += 1
        if mode == "RGBA" and arr[:, :, 3].min() == 0 and arr[:, :, 3].max() == 255:
            print("  OK  PNG transparency preserved (RGBA, alpha 0 and 255 present)")
        else:
            print(f"  FAIL PNG lost transparency (mode={mode})")
            failures += 1

        # JPEG: same exact size, transparency flattened (existing behaviour).
        up_jpg = _upscaler(tw, th, "jpg")
        out, _ = up_jpg._fill_to_target(_make_rgba_array(640, 1200))
        jpg_path = out_dir / "filled.jpg"
        up_jpg._save(out, jpg_path)
        with Image.open(jpg_path) as im:
            mode, size = im.mode, im.size
        if size == (tw, th) and mode == "RGB":
            print(f"  OK  saved JPEG flattened to RGB at exactly {tw}x{th}")
        else:
            print(f"  FAIL saved JPEG size={size} mode={mode}")
            failures += 1

    return failures


def main() -> int:
    print("compute_fill_dimensions (cover + aspect math):")
    f1 = test_fill_dimensions()
    print("exact target dimensions:")
    f2 = test_exact_output_size()
    print("centered crop:")
    f3 = test_center_crop()
    print("save paths (PNG transparency / JPEG flatten):")
    f4 = test_save_paths()
    total = f1 + f2 + f3 + f4
    if total == 0:
        print("\nTARGET SIZE TEST PASSED")
        return 0
    print(f"\nTARGET SIZE TEST FAILED: {total} failure(s).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
