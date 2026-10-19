"""Focused tests for the configurable max-output-size (final resize) behaviour.

Covers the three things the feature must guarantee, without running the (slow)
neural network:
  * aspect-ratio preservation and fit-inside-the-bounds math;
  * enforcement of the configured maximum dimensions on a real saved image;
  * transparent PNG output surviving the final resize.

Usage:
    .venv\\Scripts\\python -m tests.max_size_test
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.engine.realesrgan_upscaler import RealESRGANUpscaler, UpscaleSettings

fit = RealESRGANUpscaler.compute_fit_dimensions


def _check_aspect_and_bounds(width: int, height: int, max_w: int, max_h: int) -> None:
    """Shared invariants: fits inside bounds, aspect ratio preserved, no upscale."""
    new_w, new_h = fit(width, height, max_w, max_h)
    assert new_w <= max_w + 1, f"width {new_w} exceeds max {max_w}"
    assert new_h <= max_h + 1, f"height {new_h} exceeds max {max_h}"
    assert new_w <= width and new_h <= height, "must never enlarge"
    # Aspect ratio preserved within rounding tolerance (<= 1 px).
    assert abs(new_w * height - new_h * width) <= max(width, height), (
        f"aspect ratio drifted: {width}x{height} -> {new_w}x{new_h}")


def test_fit_dimensions() -> int:
    failures = 0

    # 1. The reference example: a 20000x10000 result saves as 8000x4000.
    if fit(20000, 10000, 8000, 5000) == (8000, 4000):
        print("  OK  20000x10000 -> 8000x4000")
    else:
        print(f"  FAIL 20000x10000 -> {fit(20000, 10000, 8000, 5000)} (expected 8000x4000)")
        failures += 1

    # 2. Tall image where height is the binding constraint.
    got = fit(10000, 10000, 8000, 5000)
    if got == (5000, 5000):
        print("  OK  10000x10000 -> 5000x5000 (square fits height bound)")
    else:
        print(f"  FAIL 10000x10000 -> {got} (expected 5000x5000)")
        failures += 1

    # 3. Extreme wide panorama: width binds, height shrinks proportionally.
    got = fit(32000, 4000, 8000, 5000)
    if got == (8000, 1000):
        print("  OK  32000x4000 -> 8000x1000 (wide fits width bound)")
    else:
        print(f"  FAIL 32000x4000 -> {got} (expected 8000x1000)")
        failures += 1

    # 4. Assorted aspect ratios must all stay inside bounds with ratio intact.
    for w, h in [(16000, 9000), (6000, 12000), (40000, 3000), (9000, 8000)]:
        try:
            _check_aspect_and_bounds(w, h, 8000, 5000)
            print(f"  OK  {w}x{h} fits within 8000x5000, aspect kept")
        except AssertionError as exc:
            print(f"  FAIL {w}x{h}: {exc}")
            failures += 1

    # 5. Already-small images are never enlarged.
    if fit(1920, 1080, 8000, 5000) == (1920, 1080):
        print("  OK  1920x1080 unchanged (within bounds)")
    else:
        print(f"  FAIL 1920x1080 -> {fit(1920, 1080, 8000, 5000)} (should be unchanged)")
        failures += 1

    # 6. A 0 on either axis disables that cap; both 0 disables the resize.
    if fit(20000, 10000, 0, 0) == (20000, 10000):
        print("  OK  max 0x0 disables resize")
    else:
        print(f"  FAIL 0x0 -> {fit(20000, 10000, 0, 0)} (expected no change)")
        failures += 1
    got = fit(20000, 10000, 0, 5000)  # only height capped -> 10000x5000
    if got == (10000, 5000):
        print("  OK  width 0 = no width cap (20000x10000 -> 10000x5000)")
    else:
        print(f"  FAIL height-only cap -> {got} (expected 10000x5000)")
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


def test_resize_and_transparency() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        settings = UpscaleSettings(output_format="png", max_width=800, max_height=500)
        up = RealESRGANUpscaler(settings)  # no model needed for the resize/save path

        # 1600x800 (2:1) result must downscale to fit 800x500 -> 800x400.
        big = _make_rgba_array(1600, 800)
        small = up._fit_to_max(big)
        h, w = small.shape[:2]
        if (w, h) == (800, 400):
            print("  OK  RGBA 1600x800 fit -> 800x400")
        else:
            print(f"  FAIL RGBA fit -> {w}x{h} (expected 800x400)")
            failures += 1

        # Save through the real PNG path and confirm transparency survived.
        out_path = out_dir / "fitted.png"
        up._save(small, out_path)
        with Image.open(out_path) as im:
            mode, size = im.mode, im.size
            arr = np.asarray(im.convert("RGBA"))
        if mode == "RGBA":
            print("  OK  saved PNG keeps RGBA mode (transparency-capable)")
        else:
            print(f"  FAIL saved PNG mode={mode} (expected RGBA)")
            failures += 1
        if size == (800, 400):
            print("  OK  saved PNG dimensions within bounds (800x400)")
        else:
            print(f"  FAIL saved PNG size={size} (expected 800x400)")
            failures += 1
        if arr[:, :, 3].min() == 0 and arr[:, :, 3].max() == 255:
            print("  OK  alpha channel preserved (0 and 255 both present)")
        else:
            print("  FAIL alpha channel lost during resize")
            failures += 1

        # 1x max bounds (0 = disabled) leaves a small image untouched.
        tiny = _make_rgba_array(64, 64)
        settings_off = UpscaleSettings(output_format="png", max_width=0, max_height=0)
        up_off = RealESRGANUpscaler(settings_off)
        kept = up_off._fit_to_max(tiny)
        if kept.shape[:2] == (64, 64):
            print("  OK  resize disabled (0x0) leaves 64x64 untouched")
        else:
            print(f"  FAIL disabled resize altered image to {kept.shape[:2]}")
            failures += 1

    return failures


def main() -> int:
    print("compute_fit_dimensions:")
    f1 = test_fit_dimensions()
    print("resize + transparency:")
    f2 = test_resize_and_transparency()
    total = f1 + f2
    if total == 0:
        print("\nMAX SIZE TEST PASSED")
        return 0
    print(f"\nMAX SIZE TEST FAILED: {total} failure(s).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
