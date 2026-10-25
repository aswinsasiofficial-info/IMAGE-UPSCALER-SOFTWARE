"""Real Real-ESRGAN runs through the rebuilt Upscaler (needs torch + weights).

Run with:  python -m tests.upscaler_model_test             # small images, CUDA if available
           python -m tests.upscaler_model_test --cpu
           python -m tests.upscaler_model_test --example   # also 1024x1536 -> 4096x6144

Downloads RealESRGAN_x4plus.pth into models/ on first use (as the app does).
Portrait, landscape, EXIF-rotated and transparent images go through the
actual network via the Django API (with small tiles, so tiling is exercised).
Each output must have exactly the planned size and its source's aspect
ratio, and must hold the whole picture: shrunk back down it lines up with
the source at zero offset — a crop, zoom or shift would break that.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import time
from pathlib import Path

import numpy as np
from django.test import Client
from PIL import Image

from app.upscaler.engine import probe_device, weights_ready
from app.upscaler.files import OUTPUT_DIR_NAME
from app.upscaler.geometry import CROP, EXACT, PAD, SizeOptions, plan_size
from app.upscaler.workspace import reset_workspace
from tests.upscaler_test import MARKERS, Api, _settings, assert_same_shape, make_marked


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def check_whole_picture(upright: Image.Image, out: Image.Image, label: str) -> float:
    """Shrink the AI output back to the source size; it must line up at offset (0, 0)."""
    src = np.asarray(upright.convert("RGB"), np.float64)
    back = np.asarray(out.convert("RGB").resize(upright.size, Image.Resampling.BOX), np.float64)
    h, w = src.shape[:2]
    m = 3
    core = src[m:h - m, m:w - m]
    scores = {}
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            scores[(dx, dy)] = _psnr(back[m + dy:h - m + dy, m + dx:w - m + dx], core)
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

    ws = reset_workspace()            # the real Real-ESRGAN engine
    api = Api(Client(enforce_csrf_checks=True))
    api.get("/")
    base = dict(device=use, fp16=use == "cuda", tile=args.tile)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "photos"
        root.mkdir()
        images = {
            "portrait": make_marked(root / "portrait.png", 96, 144),
            "landscape": make_marked(root / "landscape.png", 144, 96),
            "phone": make_marked(root / "phone.jpg", 96, 144, orientation=6),
            "transparent": make_marked(root / "transparent.png", 64, 96, alpha=True),
        }
        out_dir = root / OUTPUT_DIR_NAME
        api.post("/api/queue/add/", {"paths": [str(root)]})

        # 1) Default 4x: every image keeps its shape and its whole picture.
        started = time.monotonic()
        r = api.post("/api/jobs/", {"settings": _settings(**base)})
        assert r.status_code == 200, r.json()
        job = api.wait_job(timeout=1800)
        assert job["state"] == "done" and job["ok"] == 4, job
        print(f"  4x batch of 4 finished in {time.monotonic() - started:.1f}s")
        for name, upright in images.items():
            ext = "png"
            with Image.open(out_dir / f"{name}_x4.{ext}") as im:
                assert_same_shape(upright.size, im.size, 4)
                psnr = check_whole_picture(upright, im, name)
                extra = ""
                if name == "transparent":
                    assert im.mode == "RGBA"
                    alpha = np.asarray(im.getchannel("A"))
                    assert alpha[im.height // 2, im.width // 2] < 40 and alpha[8, 8] == 255
                    extra = ", alpha kept"
                print(f"  {name:11s} {upright.width}x{upright.height} "
                      f"({upright.width / upright.height:.4f}) -> {im.width}x{im.height} "
                      f"({im.width / im.height:.4f}), aligned at (0,0), PSNR {psnr:.1f} dB{extra}")

        # 2) 2x: the model's 4x result is reduced with one factor — still proportional.
        ws.remove([i.id for i in ws.items if i.path.stem in ("phone", "transparent")])
        r = api.post("/api/jobs/", {"settings": _settings(scale_factor="2", suffix="_x2", **base)})
        assert r.status_code == 200, r.json()
        assert api.wait_job(timeout=1800)["ok"] == 2
        for name in ("portrait", "landscape"):
            with Image.open(out_dir / f"{name}_x2.png") as im:
                assert_same_shape(images[name].size, im.size, 2)
                psnr = check_whole_picture(images[name], im, f"{name} 2x")
                print(f"  {name:11s} 2x -> {im.width}x{im.height}, aligned, PSNR {psnr:.1f} dB")

        # 3) Exact frame of the other orientation: needs a choice, then pad / crop.
        frame = dict(size_mode=EXACT, exact_width=576, exact_height=384)
        r = api.post("/api/jobs/", {"settings": _settings(**frame, **base)})
        assert r.status_code == 400 and "fit" in r.json()["errors"]
        r = api.post("/api/jobs/", {"settings": _settings(**frame, fit=PAD, suffix="_pad", **base)})
        assert r.status_code == 200 and api.wait_job(timeout=1800)["ok"] == 2
        with Image.open(out_dir / "portrait_pad.png") as im:
            plan = plan_size(96, 144, SizeOptions(EXACT, width=576, height=384, fit=PAD))
            (x, y), (cw, ch) = plan.offset, plan.content
            assert im.size == (576, 384) and (cw, ch) == (256, 384) and (x, y) == (160, 0)
            arr = np.asarray(im.convert("RGB"))
            assert arr[:, :x].min() == 255 and arr[:, x + cw:].min() == 255   # white borders
            psnr = check_whole_picture(images["portrait"], im.crop((x, y, x + cw, y + ch)), "pad")
            print(f"  portrait into 576x384, pad: whole 256x384 picture + 160 px borders each "
                  f"side, aligned, PSNR {psnr:.1f} dB")
        r = api.post("/api/jobs/", {"settings": _settings(**frame, fit=CROP, suffix="_crop", **base)})
        assert r.status_code == 200 and api.wait_job(timeout=1800)["ok"] == 2
        with Image.open(out_dir / "landscape_crop.png") as im:
            assert im.size == (576, 384)        # landscape 3:2 already matches: no trim
            psnr = check_whole_picture(images["landscape"], im, "landscape exact")
        with Image.open(out_dir / "portrait_crop.png") as im:
            plan = plan_size(96, 144, SizeOptions(EXACT, width=576, height=384, fit=CROP))
            assert im.size == (576, 384) and plan.cropped == (0, 240, 0, 240)
            print(f"  portrait into 576x384, crop: filled {plan.content[0]}x{plan.content[1]}, "
                  f"trimmed 240 px top and bottom as planned; landscape matched the frame as-is")

        # 4) The README example through the real network.
        if args.example:
            ex = Path(tmp) / "example"
            ex.mkdir()
            upright = make_marked(ex / "portrait_1024x1536.png", 1024, 1536)
            ws.clear()
            api.post("/api/queue/add/", {"paths": [str(ex / "portrait_1024x1536.png")]})
            plan = api.post("/api/plan/", {"settings": _settings(**base)}).json()
            assert plan["items"][0]["output"] == [4096, 6144], plan["items"][0]
            started = time.monotonic()
            r = api.post("/api/jobs/", {"settings": _settings(output_dir=str(ex / "out"),
                                                              **{**base, "tile": 256})})
            assert r.status_code == 200, r.json()
            job = api.wait_job(timeout=7200)
            assert job["ok"] == 1, job
            with Image.open(ex / "out" / "portrait_1024x1536_x4.png") as im:
                assert im.size == (4096, 6144)
                psnr = check_whole_picture(upright, im, "1024x1536 example")
            print(f"  1024x1536 -> previewed 4096x6144 -> saved 4096x6144 in "
                  f"{time.monotonic() - started:.0f}s, aligned, PSNR {psnr:.1f} dB")

    reset_workspace()
    print("UPSCALER MODEL TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
