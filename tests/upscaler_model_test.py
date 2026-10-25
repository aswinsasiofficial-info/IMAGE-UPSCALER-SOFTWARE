"""Real Real-ESRGAN runs through the Upscaler core (needs torch + weights).

Run with:  python -m tests.upscaler_model_test             # small images, CUDA if available
           python -m tests.upscaler_model_test --cpu
           python -m tests.upscaler_model_test --example   # also 1024x1536 -> 4096x6144

Downloads RealESRGAN_x4plus.pth into models/ on first use (as the app does).
Portrait, landscape, EXIF-rotated and transparent images go through the
actual network at every factor 1×–5× (with small tiles, so tiling is
exercised). Each output must have exactly factor × the source size on both
sides, and hold the whole picture: shrunk back down it lines up with the
source at zero offset — a crop, zoom or shift would break that.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.upscaler.engine import probe_device, weights_ready
from app.upscaler.geometry import FACTORS
from app.upscaler.workspace import JobOptions, Workspace
from tests.upscaler_test import MARKERS, make_marked


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def _visible(img: Image.Image) -> Image.Image:
    """What a viewer sees: transparent areas over white (their hidden colour is irrelevant
    and is dropped by premultiplied resizing)."""
    if img.mode != "RGBA":
        return img.convert("RGB")
    return Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img).convert("RGB")


def check_whole_picture(upright: Image.Image, out: Image.Image, label: str) -> float:
    """Shrink the AI output back to the source size; it must line up at offset (0, 0)."""
    src = np.asarray(_visible(upright), np.float64)
    back = np.asarray(_visible(out.resize(upright.size, Image.Resampling.BOX)), np.float64)
    h, w = src.shape[:2]
    m = 3
    core = src[m:h - m, m:w - m]
    scores = {(dx, dy): _psnr(back[m + dy:h - m + dy, m + dx:w - m + dx], core)
              for dy in range(-2, 3) for dx in range(-2, 3)}
    best = max(scores, key=scores.get)
    psnr = _psnr(back, src)
    assert best == (0, 0), f"{label}: content is shifted/zoomed (best match at offset {best})"
    assert psnr >= 24.0, f"{label}: shrunk output doesn't match the source (PSNR {psnr:.1f} dB)"
    arr = np.asarray(out.convert("RGB")).astype(int)
    oh, ow = arr.shape[:2]
    ix, iy = max(2, ow // 30), max(2, oh // 30)
    corners = {"tl": arr[iy, ix], "tr": arr[iy, ow - 1 - ix],
               "bl": arr[oh - 1 - iy, ix], "br": arr[oh - 1 - iy, ow - 1 - ix]}
    for name, color in corners.items():
        assert np.abs(color - MARKERS[name]).max() <= 40, f"{label}: {name} corner is {color}"
    return psnr


def run(ws: Workspace, options: JobOptions, timeout: float = 1800) -> dict:
    job = ws.start(options)
    job.join(timeout)
    snap = job.snapshot()
    assert snap["state"] == "done" and snap["failed"] == 0, snap
    return snap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    parser.add_argument("--tile", type=int, default=64, help="tile size (small = tiling exercised)")
    parser.add_argument("--example", action="store_true",
                        help="also run the 1024x1536 -> 4096x6144 example through the model")
    args = parser.parse_args()

    device = probe_device()
    use = "cpu" if args.cpu or not device["cuda"] else "cuda"
    print(f"torch {device['torch']} | running on {use.upper()}"
          + (f" ({device['name']})" if use == "cuda" else "")
          + f" | weights {'cached' if weights_ready() else 'will be downloaded'}")
    base = dict(device=use, fp16=use == "cuda", tile=args.tile, output_format="png",
                rename_batch=False)

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "photos"
        src.mkdir()
        images = {
            "portrait": make_marked(src / "portrait.png", 96, 144),
            "landscape": make_marked(src / "landscape.png", 144, 96),
            "phone": make_marked(src / "phone.jpg", 96, 144, orientation=6),
            "transparent": make_marked(src / "transparent.png", 64, 96, alpha=True),
        }
        for factor in FACTORS:
            ws = Workspace()                      # the real Real-ESRGAN engine
            ws.add([str(p) for p in sorted(src.iterdir())])
            out = Path(tmp) / f"x{factor}"
            started = time.monotonic()
            run(ws, JobOptions(factor=factor, output_dir=out, **base))
            elapsed = time.monotonic() - started
            line = []
            for name, upright in images.items():
                with Image.open(out / f"{name}.png") as im:
                    assert im.size == (upright.width * factor, upright.height * factor), (name, factor, im.size)
                    assert im.width * upright.height == im.height * upright.width
                    psnr = check_whole_picture(upright, im, f"{name} {factor}x")
                    if name == "transparent":
                        assert im.mode == "RGBA"
                        a = np.asarray(im.getchannel("A"))
                        assert a[im.height // 2, im.width // 2] < 40 and a[3, 3] == 255
                    line.append(f"{name} {upright.width}x{upright.height}->{im.width}x{im.height} "
                                f"({psnr:.1f} dB)")
            print(f"  {factor}x in {elapsed:.1f}s: " + "; ".join(line))

        # JPEG + batch rename (the app defaults) through the real model.
        ws = Workspace()
        ws.add([str(src / "portrait.png"), str(src / "landscape.png")])
        out = Path(tmp) / "defaults"
        run(ws, JobOptions(output_dir=out, device=use, fp16=use == "cuda", tile=args.tile))
        names = sorted(p.name for p in out.iterdir())
        assert len(names) == 2 and all(n.endswith(".jpg") for n in names), names
        assert len({n[:-5] for n in names}) == 1, names        # one shared prefix + 1, 2
        print(f"  defaults (4×, JPEG, batch rename): {names}")

        if args.example:
            ex = Path(tmp) / "example"
            ex.mkdir()
            upright = make_marked(ex / "portrait_1024x1536.png", 1024, 1536)
            ws = Workspace()
            ws.add([str(ex / "portrait_1024x1536.png")])
            started = time.monotonic()
            run(ws, JobOptions(factor=4, output_dir=ex / "out", **{**base, "tile": 256}), timeout=7200)
            with Image.open(ex / "out" / "portrait_1024x1536.png") as im:
                assert im.size == (4096, 6144)
                psnr = check_whole_picture(upright, im, "1024x1536 example")
            print(f"  1024x1536 at 4x -> 4096x6144 in {time.monotonic() - started:.0f}s, "
                  f"aligned, PSNR {psnr:.1f} dB")

    print("UPSCALER MODEL TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
