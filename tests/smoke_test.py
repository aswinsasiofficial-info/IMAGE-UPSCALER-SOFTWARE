"""Headless smoke test for the upscaler engine (no GUI required).

Generates a few synthetic images (RGB + RGBA), runs a real batch through
Real-ESRGAN x4plus and asserts the outputs have the expected 4x dimensions.

Usage:
    .venv\\Scripts\\python -m tests.smoke_test            # auto device
    .venv\\Scripts\\python -m tests.smoke_test --cpu       # force CPU
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.engine.realesrgan_upscaler import RealESRGANUpscaler, UpscaleSettings


def _make_images(folder: Path) -> list[Path]:
    """Create small test images with hard edges so upscaling is measurable."""
    paths: list[Path] = []
    rng = np.random.default_rng(0)
    for i, size in enumerate([(48, 64), (64, 64), (40, 56)]):
        h, w = size
        base = rng.integers(0, 255, (h, w), dtype=np.uint8)
        # overlay a crisp diagonal line for detail
        yy, xx = np.ogrid[:h, :w]
        base[xx == yy] = 255
        rgb = np.dstack([base, np.roll(base, 40, axis=1), np.roll(base, 80, axis=0)])
        img = Image.fromarray(rgb.astype(np.uint8), "RGB")
        p = folder / f"test_{i}.png"
        img.save(p)
        paths.append(p)
    # one RGBA image to exercise the alpha path
    rgba = Image.fromarray(
        np.dstack([rng.integers(0, 255, (32, 32, 3), dtype=np.uint8),
                   rng.integers(0, 255, (32, 32), dtype=np.uint8)]), "RGBA")
    pa = folder / "test_alpha.png"
    rgba.save(pa)
    paths.append(pa)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA exists")
    parser.add_argument("--tile", type=int, default=64, help="Tile size (default small for test)")
    parser.add_argument("--scale", type=float, default=4.0, help="Output scale (network is 4x; others resampled)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        files = _make_images(tmp_path)
        out_dir = tmp_path / "out"
        settings = UpscaleSettings(
            device="cpu" if args.cpu else "auto",
            fp16=False if args.cpu else True,
            tile=args.tile,
            tile_pad=8,
            output_scale=args.scale,
            output_format="png",
        )
        up = RealESRGANUpscaler(settings)
        up.load(log=print)
        print(f"Device in use: {up.device}, fp16={up.use_fp16}")

        failures = 0
        for src in files:
            dst = out_dir / src.name
            up.upscale_image(src, dst, log=print)
            with Image.open(dst) as im:
                out_size, out_mode = im.size, im.mode
            with Image.open(src) as in_im:
                expected = (round(in_im.width * args.scale), round(in_im.height * args.scale))
                in_mode = in_im.mode
            ok = out_size == expected and out_mode == in_mode
            print(f"  -> {dst.name}: {out_size} mode={out_mode} "
                  f"[{'OK' if ok else 'MISMATCH, expected ' + str(expected)}]")
            failures += 0 if ok else 1

        if failures:
            print(f"\nSMOKE TEST FAILED: {failures} mismatched output(s).")
            return 1
        print(f"\nSMOKE TEST PASSED: all outputs are {args.scale:g}x with correct mode.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
