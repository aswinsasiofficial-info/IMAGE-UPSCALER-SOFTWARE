"""Tests for the rebuilt (Django) Upscaler — no model, no GPU, no network.

Run with:  python -m tests.upscaler_test

The Real-ESRGAN network is replaced by an exact nearest-neighbour 4x stand-in
so every pixel is predictable. Covers the size planning (proportional scale,
exact sizes with explicit pad/crop, no stretching), the settings form, the
JSON API through Django's test client (launch token + CSRF included), real
files written by a full batch for portrait and landscape images — checking
dimensions, aspect ratio and that the whole picture survives — plus folder
mode, naming, alpha/ICC handling, cancel, and independence from the old
upscaler code.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.web.settings")

import django

django.setup()

import numpy as np
from django.test import Client
from django.test.utils import setup_test_environment
from PIL import Image, ImageCms

from app.upscaler import imaging
from app.upscaler.engine import Cancelled
from app.upscaler.files import OUTPUT_DIR_NAME, SequentialNamer, discover, unique_path
from app.upscaler.forms import UpscaleForm
from app.upscaler.geometry import (CROP, EXACT, PAD, SCALE, SizeOptions, plan_size,
                                   round_half_up)
from app.upscaler.workspace import reset_workspace
from app.web import middleware

setup_test_environment()

ROOT = Path(__file__).resolve().parents[1]
MARKERS = {"tl": (255, 0, 0), "tr": (0, 200, 0), "bl": (0, 0, 255), "br": (255, 220, 0)}


# -- fixtures -------------------------------------------------------------------
class StubEngine:
    """Stands in for Real-ESRGAN: exactly 4x, nearest neighbour (pixel-exact)."""

    loads = 0
    delay = 0.0

    def __init__(self, options) -> None:
        self.options = options

    def load(self, log, progress=None) -> None:
        StubEngine.loads += 1
        log("Model loaded and ready.")

    def upscale(self, rgb, cancelled=lambda: False, log=None):
        deadline = time.monotonic() + StubEngine.delay
        while time.monotonic() < deadline:
            if cancelled():
                raise Cancelled()
            time.sleep(0.01)
        if cancelled():
            raise Cancelled()
        h, w = rgb.shape[:2]
        return np.asarray(Image.fromarray(rgb).resize((w * 4, h * 4), Image.Resampling.NEAREST))

    def close(self) -> None:
        pass


def make_marked(path: Path, w: int, h: int, *, alpha: bool = False, orientation: int = 0,
                icc: bytes | None = None) -> Image.Image:
    """Gradient picture with a solid colour block in each corner.

    Returns the *upright* image (what a viewer shows) for later comparison.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.dstack([(xx * 255 // max(1, w - 1)), (yy * 255 // max(1, h - 1)),
                     np.full_like(xx, 128)]).astype(np.uint8)
    bw, bh = max(2, w // 6), max(2, h // 6)
    arr[:bh, :bw] = MARKERS["tl"]
    arr[:bh, -bw:] = MARKERS["tr"]
    arr[-bh:, :bw] = MARKERS["bl"]
    arr[-bh:, -bw:] = MARKERS["br"]
    upright = Image.fromarray(arr)
    if alpha:
        a = np.full((h, w), 255, np.uint8)
        a[h // 2 - 2:h // 2 + 2, :] = 0        # a transparent stripe across the middle
        upright = upright.copy()
        upright.putalpha(Image.fromarray(a))
    stored = upright
    kwargs = {}
    if orientation == 6:
        # Store the pixels rotated the other way; EXIF 6 tells viewers to rotate back.
        stored = upright.transpose(Image.Transpose.ROTATE_90)
        exif = Image.Exif()
        exif[imaging.EXIF_ORIENTATION] = 6
        kwargs["exif"] = exif.tobytes()
    if icc:
        kwargs["icc_profile"] = icc
    if path.suffix.lower() == ".jpg":
        stored.convert("RGB").save(path, "JPEG", quality=98, **kwargs)
    else:
        stored.save(path, **kwargs)
    return upright


def assert_same_shape(src: tuple[int, int], out: tuple[int, int], factor: float) -> None:
    sw, sh = src
    ow, oh = out
    assert (ow, oh) == (round_half_up(sw * factor), round_half_up(sh * factor)), (src, out, factor)
    # Same aspect ratio up to sub-pixel rounding.
    assert abs(ow / oh - sw / sh) <= (sw / sh) * (1.0 / min(oh, ow)) + 1e-9, (src, out)


def assert_full_content(upright: Image.Image, out: Image.Image, tol: float = 2.0) -> None:
    """Shrinking the output back must reproduce the source: nothing cropped or shifted."""
    back = np.asarray(out.convert("RGB").resize(upright.size, Image.Resampling.BOX), np.float64)
    diff = np.abs(back - np.asarray(upright.convert("RGB"), np.float64)).mean()
    assert diff <= tol, f"output does not contain the whole picture (mean diff {diff:.2f})"
    # Corner blocks must sit in the matching output corners.
    arr = np.asarray(out.convert("RGB"))
    h, w = arr.shape[:2]
    corners = {"tl": arr[2, 2], "tr": arr[2, w - 3], "bl": arr[h - 3, 2], "br": arr[h - 3, w - 3]}
    for name, color in corners.items():
        assert np.abs(color.astype(int) - MARKERS[name]).max() <= 8, (name, color)


# -- geometry ----------------------------------------------------------------------
def test_geometry() -> None:
    four = SizeOptions(SCALE, factor=4)
    p = plan_size(1024, 1536, four)
    assert p.output == (4096, 6144) and p.content == p.output and p.offset == (0, 0)
    assert plan_size(1536, 1024, four).output == (6144, 4096)
    assert plan_size(1000, 1000, four).output == (4000, 4000)
    for w, h in [(1024, 1536), (1536, 1024), (1001, 667), (333, 1999), (7, 3)]:
        for f in (1, 1.5, 2, 2.5, 3, 4, 8):
            assert_same_shape((w, h), plan_size(w, h, SizeOptions(SCALE, factor=f)).output, f)

    # Exact frame with the image's own shape: plain uniform resize, no choice needed.
    p = plan_size(1024, 1536, SizeOptions(EXACT, width=2000, height=3000))
    assert p.output == p.content == (2000, 3000) and not p.conflict and p.resolved

    # Different shape: unresolved until the user picks; nothing is ever stretched.
    unchosen = plan_size(1024, 1536, SizeOptions(EXACT, width=8000, height=5000))
    assert unchosen.conflict and not unchosen.resolved and unchosen.output == (8000, 5000)
    pad = plan_size(1024, 1536, SizeOptions(EXACT, width=8000, height=5000, fit=PAD))
    assert pad.content == (3333, 5000) and pad.padding == (2333, 0, 2334, 0)
    crop = plan_size(1024, 1536, SizeOptions(EXACT, width=8000, height=5000, fit=CROP))
    assert crop.content == (8000, 12000) and crop.cropped == (0, 3500, 0, 3500)
    land = plan_size(1536, 1024, SizeOptions(EXACT, width=8000, height=5000, fit=PAD))
    assert land.content == (7500, 5000) and land.padding == (250, 0, 250, 0)
    land = plan_size(1536, 1024, SizeOptions(EXACT, width=8000, height=5000, fit=CROP))
    assert land.content == (8000, 5333) and land.cropped == (0, 166, 0, 167)

    # A frame typed from the image's own proportions always counts as the same shape
    # (regression: 1920x1080 -> 5005x2815 used to be flagged by 1 px).
    for w, h in [(1920, 1080), (1080, 1920), (1366, 768), (1001, 667), (3, 7)]:
        for fw in range(1, 20001, 7):
            fh = max(1, round_half_up(fw * h / w))
            pl = plan_size(w, h, SizeOptions(EXACT, width=fw, height=fh))
            assert not pl.conflict and pl.content == (fw, fh), (w, h, fw, fh, pl)
        for f in (1, 1.5, 2.5, 4, 8):      # any scale-mode result, typed as an exact size
            out = plan_size(w, h, SizeOptions(SCALE, factor=f)).output
            assert not plan_size(w, h, SizeOptions(EXACT, width=out[0], height=out[1])).conflict
    assert plan_size(1920, 1080, SizeOptions(EXACT, width=5005, height=2817)).conflict

    # Every exact plan resizes the whole picture by one factor (shape kept).
    for w, h in [(1024, 1536), (1536, 1024), (640, 480), (999, 1001)]:
        for fw, fh in [(8000, 5000), (5000, 8000), (1920, 1080), (1000, 1000)]:
            for fit in (PAD, CROP):
                pl = plan_size(w, h, SizeOptions(EXACT, width=fw, height=fh, fit=fit))
                cw, ch = pl.content
                # One factor s for both axes, each side rounded to the nearest pixel.
                assert abs(cw / w - ch / h) <= 0.5 / w + 0.5 / h + 1e-12, (w, h, fw, fh, fit, pl)
                assert pl.output == (fw, fh)
                if fit == PAD:
                    assert cw <= fw and ch <= fh and min(pl.padding) >= 0
                else:
                    assert cw >= fw and ch >= fh and min(pl.cropped) >= 0
    print("  geometry OK")


# -- form ----------------------------------------------------------------------------
def _settings(**kw) -> dict:
    base = dict(size_mode=SCALE, scale_factor="4", exact_width="", exact_height="", fit="",
                pad_color="white", output_dir="", save_beside=False, device="cpu", fp16=False,
                tile=64, output_format="png", jpg_quality=95, suffix="_x4", rename_batch=False)
    base.update(kw)
    return base


def test_form() -> None:
    imgs = [(1024, 1536), (1536, 1024)]
    assert UpscaleForm(_settings(), images=imgs).is_valid()

    exact = _settings(size_mode=EXACT, exact_width=8000, exact_height=5000)
    form = UpscaleForm(exact, images=imgs)
    assert not form.is_valid() and "fit" in form.errors, form.errors
    assert "2 of 2" in form.errors["fit"][0]
    assert UpscaleForm({**exact, "fit": PAD}, images=imgs).is_valid()
    assert UpscaleForm({**exact, "fit": CROP}, images=imgs).is_valid()
    # Matching frame: no choice required.
    assert UpscaleForm(_settings(size_mode=EXACT, exact_width=4096, exact_height=6144),
                       images=[(1024, 1536)]).is_valid()
    # Unknown fit values (e.g. "stretch") are rejected.
    assert "fit" in UpscaleForm({**exact, "fit": "stretch"}, images=imgs).errors

    f = UpscaleForm({**exact, "fit": PAD, "pad_color": "transparent", "output_format": "jpg"},
                    images=imgs)
    assert not f.is_valid() and "pad_color" in f.errors
    # A leftover pad choice doesn't matter once back in scale mode.
    assert UpscaleForm(_settings(fit=PAD, pad_color="transparent", output_format="jpg"),
                       images=imgs).is_valid()
    f = UpscaleForm(_settings(scale_factor="8", output_format="jpg"), images=[(9000, 100)])
    assert not f.is_valid() and "__all__" in f.errors        # 72000 px > JPEG limit
    assert f.size_options() is not None                      # ...but the preview still works
    f = UpscaleForm(_settings(size_mode=EXACT, exact_width=65501, exact_height=100),
                    images=[(65501, 100)])
    assert "exact_width" in f.errors                          # libjpeg stops at 65500
    # Values left in boxes the page greys out never block a batch.
    assert UpscaleForm({**exact, "fit": PAD, "scale_factor": "20"}, images=imgs).is_valid()
    assert UpscaleForm(_settings(exact_width="70000", fit="bogus"), images=imgs).is_valid()
    assert UpscaleForm(_settings(jpg_quality=""), images=imgs).is_valid()       # png output
    assert UpscaleForm(_settings(rename_batch=True, suffix="a:b"), images=imgs).is_valid()
    assert UpscaleForm(_settings(save_beside=True, output_dir=__file__), images=imgs).is_valid()
    assert "jpg_quality" in UpscaleForm(_settings(output_format="jpg", jpg_quality=""),
                                        images=imgs).errors
    assert "suffix" in UpscaleForm(_settings(suffix="a/b"), images=imgs).errors
    assert "scale_factor" in UpscaleForm(_settings(scale_factor="9"), images=imgs).errors
    assert "exact_width" in UpscaleForm(_settings(size_mode=EXACT), images=imgs).errors
    print("  form OK")


# -- files / naming --------------------------------------------------------------------
def test_files(tmp: Path) -> None:
    root = tmp / "files"
    (root / "sub").mkdir(parents=True)
    (root / OUTPUT_DIR_NAME / "sub").mkdir(parents=True)
    for p in (root / "a.png", root / "sub" / "b.jpg", root / OUTPUT_DIR_NAME / "old_x4.png",
              root / OUTPUT_DIR_NAME / "sub" / "b_x4.png"):
        Image.new("RGB", (4, 4)).save(p)
    (root / "notes.txt").write_text("x")
    found = discover([root, root / "a.png"])
    assert [p.name for p in found] == ["a.png", "b.jpg"], found   # upscaled/ skipped, deduped

    taken = tmp / "taken.png"
    Image.new("RGB", (2, 2)).save(taken)
    assert unique_path(taken).name == "taken(1).png"
    namer = SequentialNamer("ABCD")
    assert namer.next_path(tmp, ".png").name == "ABCD1.png"
    assert namer.next_path(tmp, ".png").name == "ABCD2.png"
    namer.release(tmp)
    assert namer.next_path(tmp, ".png").name == "ABCD2.png"
    assert re.fullmatch(r"[A-Z]{4,5}", SequentialNamer().prefix)
    print("  files OK")


# -- imaging ------------------------------------------------------------------------------
def test_imaging(tmp: Path) -> None:
    # EXIF-rotated phone portrait: stored landscape, shown (and processed) as portrait.
    p = tmp / "phone.jpg"
    upright = make_marked(p, 60, 90, orientation=6)
    with Image.open(p) as raw:
        assert raw.size == (90, 60)
    assert (imaging.probe(p).width, imaging.probe(p).height) == (60, 90)
    src = imaging.load(p)
    assert src.size == (60, 90)
    assert np.abs(src.rgb.astype(int) - np.asarray(upright, int)).mean() < 4

    # Probing a PNG must not decode its pixels (regression: getexif() loaded them).
    from PIL import PngImagePlugin
    big = tmp / "big.png"
    Image.new("RGB", (3000, 2000), (10, 20, 30)).save(big)
    loads = []
    real_load = PngImagePlugin.PngImageFile.load
    PngImagePlugin.PngImageFile.load = lambda self: (loads.append(1), real_load(self))[1]
    try:
        assert (imaging.probe(big).width, imaging.probe(big).height) == (3000, 2000)
    finally:
        PngImagePlugin.PngImageFile.load = real_load
    assert not loads, "probe decoded the PNG"

    # 16-bit greyscale must not clip to white.
    p16 = tmp / "grey16.png"
    Image.fromarray(np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 16).save(p16)
    g = imaging.load(p16).rgb
    assert 0 < g.mean() < 255 and g.min() < 20 and g.max() > 200, (g.min(), g.mean(), g.max())

    # An unresolved plan can never be turned into pixels.
    plan = plan_size(10, 20, SizeOptions(EXACT, width=40, height=20))
    try:
        imaging.compose(np.zeros((80, 40, 3), np.uint8), None, plan)
        raise AssertionError("compose accepted an unresolved plan")
    except ValueError:
        pass
    print("  imaging OK")


# -- API + full batches ----------------------------------------------------------------------
class Api:
    def __init__(self, client: Client) -> None:
        self.c = client

    def _csrf(self) -> str:
        return self.c.cookies["csrftoken"].value

    def get(self, path: str):
        return self.c.get(path)

    def post(self, path: str, body: dict | None = None):
        return self.c.post(path, data=json.dumps(body or {}), content_type="application/json",
                           HTTP_X_CSRFTOKEN=self._csrf())

    def wait_job(self, timeout: float = 120) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get("/api/jobs/current/").json()["job"]
            if job and not job["running"]:
                return job
            time.sleep(0.05)
        raise AssertionError("batch did not finish in time")


def test_security() -> None:
    token = middleware.new_launch_token()
    try:
        c = Client(enforce_csrf_checks=True)
        assert c.get("/").status_code == 403                       # no token
        assert c.get("/?launch=wrong").status_code == 403
        assert c.get("/?launch=é").status_code == 403             # non-ASCII: refused, not a 500
        r = c.get(f"/?launch={token}")
        assert r.status_code == 302 and middleware.COOKIE_NAME in r.cookies
        assert r.cookies[middleware.COOKIE_NAME]["samesite"] == "Strict"
        r = c.get("/")
        assert r.status_code == 200 and b"Start Upscaling" in r.content
        # POST without the CSRF header is refused even with the launch cookie.
        r = c.post("/api/queue/clear/", data="{}", content_type="application/json")
        assert r.status_code == 403
        assert Api(c).post("/api/queue/clear/").status_code == 200
        # A foreign Host header (DNS rebinding) is refused (Django logs it; keep that quiet here).
        security_log = logging.getLogger("django.security.DisallowedHost")
        security_log.disabled = True
        try:
            assert c.get("/", HTTP_HOST="evil.example").status_code == 400
        finally:
            security_log.disabled = False
    finally:
        middleware.clear_launch_token()
    print("  launch token / CSRF / host checks OK")


def test_page() -> None:
    c = Client(enforce_csrf_checks=True)
    r = c.get("/")
    assert r.status_code == 200
    html = r.content.decode()
    for text in ("Images to process", "Add Images…", "Add Folder…", "Remove Selected",
                 "Save beside each original", "Half precision (fp16)", "Tile size (px):",
                 "Scale by", "Exact size", "Pad to fit", "Crop to fill", "Filename suffix:",
                 "Rename batch", "Start Upscaling", "Cancel"):
        assert text in html, text
    cfg = json.loads(re.search(r'id="upscaler-config"[^>]*>(.*?)</script>', html, re.S).group(1))
    assert cfg["defaults"]["size_mode"] == "scale" and cfg["defaults"]["scale_factor"] == "4"
    assert cfg["defaults"]["fit"] == ""                 # no silent default for pad/crop
    assert "8000" not in html and "5000" not in html    # the fixed-shape target is gone
    for asset in ("/static/upscaler/upscaler.css", "/static/upscaler/upscaler.js"):
        assert asset in html
    print("  page OK")


def test_batches(tmp: Path) -> None:
    ws = reset_workspace(engine_factory=StubEngine)
    StubEngine.loads = 0
    api = Api(Client(enforce_csrf_checks=True))
    api.get("/")                                          # sets the CSRF cookie
    src = tmp / "in"
    src.mkdir()
    out = tmp / "out"
    portrait = make_marked(src / "portrait.png", 1024, 1536)
    landscape = make_marked(src / "landscape.png", 1536, 1024)
    phone = make_marked(src / "phone.jpg", 300, 450, orientation=6)

    r = api.post("/api/queue/add/", {"paths": [str(src / "portrait.png"), str(src / "landscape.png"),
                                               str(src / "phone.jpg")]})
    queue = r.json()["queue"]
    sizes = {q["name"]: (q["width"], q["height"], q["orientation"]) for q in queue}
    assert sizes == {"portrait.png": (1024, 1536, "portrait"),
                     "landscape.png": (1536, 1024, "landscape"),
                     "phone.jpg": (300, 450, "portrait")}, sizes

    # The preview shows every expected output size before anything runs.
    plan = api.post("/api/plan/", {"settings": _settings(output_dir=str(out))}).json()
    assert plan["valid"] and plan["conflicts"] == 0
    outputs = {row["name"]: tuple(row["output"]) for row in plan["items"]}
    assert outputs == {"portrait.png": (4096, 6144), "landscape.png": (6144, 4096),
                       "phone.jpg": (1200, 1800)}, outputs

    # Full batch at 4x: real files, exact sizes, same shape, whole picture.
    r = api.post("/api/jobs/", {"settings": _settings(output_dir=str(out))})
    assert r.status_code == 200, r.json()
    job = api.wait_job()
    assert job["state"] == "done" and job["ok"] == 3 and job["failed"] == 0, job
    assert StubEngine.loads == 1, "model must load once per batch"
    for name, upright in (("portrait", portrait), ("landscape", landscape), ("phone", phone)):
        path = out / f"{name}_x4.png"
        with Image.open(path) as im:
            assert_same_shape(upright.size, im.size, 4)
            assert_full_content(upright, im, tol=2.0 if name != "phone" else 6.0)
    print("  4x batch: portrait 1024x1536 -> 4096x6144, landscape 1536x1024 -> 6144x4096, "
          "EXIF portrait 300x450 -> 1200x1800, whole picture kept")

    # 2.5x goes through the Lanczos resize after the 4x model: still proportional.
    small = tmp / "small"
    small.mkdir()
    p_small = make_marked(small / "p.png", 120, 180)
    l_small = make_marked(small / "l.png", 180, 120)
    api.post("/api/queue/clear/")
    api.post("/api/queue/add/", {"paths": [str(small)]})
    r = api.post("/api/jobs/", {"settings": _settings(scale_factor="2.5", suffix="_x2.5")})
    assert r.status_code == 200, r.json()
    job = api.wait_job()
    assert job["ok"] == 2, job
    for name, upright in (("p", p_small), ("l", l_small)):
        with Image.open(small / OUTPUT_DIR_NAME / f"{name}_x2.5.png") as im:
            assert_same_shape(upright.size, im.size, 2.5)
            assert_full_content(upright, im, tol=4.0)

    # Exact frame of a different shape: refused until Pad or Crop is chosen.
    exact = _settings(size_mode=EXACT, exact_width=800, exact_height=500, output_dir=str(out),
                      suffix="_frame")
    plan = api.post("/api/plan/", {"settings": exact}).json()
    assert not plan["valid"] and plan["conflicts"] == 2 and "fit" in plan["errors"]
    assert all(row["kind"] == "conflict" and row["output"] == [800, 500] for row in plan["items"])
    assert all(set(row["alternatives"]) == {PAD, CROP} for row in plan["items"])
    r = api.post("/api/jobs/", {"settings": exact})
    assert r.status_code == 400 and "fit" in r.json()["errors"]
    assert not list(small.rglob("*_frame*")), "nothing may be written without a choice"

    # Pad: whole picture kept, centred, borders in the chosen colour.
    r = api.post("/api/jobs/", {"settings": {**exact, "fit": PAD, "pad_color": "black"}})
    assert r.status_code == 200
    assert api.wait_job()["ok"] == 2
    with Image.open(small / OUTPUT_DIR_NAME / "p_frame.png") as im:
        assert im.size == (800, 500)
        pl = plan_size(120, 180, SizeOptions(EXACT, width=800, height=500, fit=PAD))
        arr = np.asarray(im.convert("RGB"))
        x, y = pl.offset
        cw, ch = pl.content
        assert (cw, ch) == (333, 500) and arr[:, :x].max() == 0 and arr[:, x + cw:].max() == 0
        assert_full_content(p_small, im.crop((x, y, x + cw, y + ch)), tol=6.0)

    # Crop: fills the frame; the trimmed part is exactly the planned centre band.
    r = api.post("/api/jobs/", {"settings": {**exact, "fit": CROP, "suffix": "_crop"}})
    assert r.status_code == 200
    assert api.wait_job()["ok"] == 2
    with Image.open(small / OUTPUT_DIR_NAME / "l_crop.png") as im:
        assert im.size == (800, 500)
        pl = plan_size(180, 120, SizeOptions(EXACT, width=800, height=500, fit=CROP))
        expected = l_small.resize(pl.content, Image.Resampling.LANCZOS)
        left, top = -pl.offset[0], -pl.offset[1]
        expected = expected.crop((left, top, left + 800, top + 500))
        diff = np.abs(np.asarray(im, float) - np.asarray(expected, float)).mean()
        assert diff < 6, diff
    print("  exact size: blocked without a choice; pad keeps the whole picture; crop trims "
          "only the planned band")

    # One oversized image flags itself without blanking the others' preview, and the
    # memory warning counts the 4x intermediate, not just the output.
    huge = tmp / "huge"
    huge.mkdir()
    Image.new("RGB", (17000, 10), (5, 5, 5)).save(huge / "strip.png")      # 68000 px at 4x
    Image.new("RGB", (4000, 3000), (5, 5, 5)).save(huge / "photo.png")     # 192 MP at 4x
    api.post("/api/queue/clear/")
    api.post("/api/queue/add/", {"paths": [str(huge)]})
    plan = api.post("/api/plan/", {"settings": _settings()}).json()
    rows = {row["name"]: row for row in plan["items"]}
    assert not plan["valid"] and "__all__" in plan["errors"], plan["errors"]
    assert rows["strip.png"]["warning"].startswith("Too large")
    assert rows["photo.png"]["output"] == [16000, 12000]
    plan = api.post("/api/plan/", {"settings": _settings(scale_factor="1")}).json()
    rows = {row["name"]: row for row in plan["items"]}
    assert plan["valid"] and rows["photo.png"]["output"] == [4000, 3000]
    assert "192 MP while processing" in rows["photo.png"]["warning"], rows["photo.png"]
    r = api.post("/api/jobs/", {"settings": _settings(tile="")})
    assert r.status_code == 400 and r.json()["error"].startswith("Tile size:"), r.json()
    ws.clear()


def test_folder_mode_alpha_icc_rename(tmp: Path) -> None:
    reset_workspace(engine_factory=StubEngine)
    api = Api(Client(enforce_csrf_checks=True))
    api.get("/")
    root = tmp / "photos"
    (root / "trip").mkdir(parents=True)
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    make_marked(root / "trip" / "a.png", 40, 60, alpha=True)
    make_marked(root / "b.jpg", 60, 40, icc=srgb)
    (root / OUTPUT_DIR_NAME).mkdir()
    Image.new("RGB", (8, 8)).save(root / OUTPUT_DIR_NAME / "ignored.png")

    added = api.post("/api/queue/add/", {"paths": [str(root)]}).json()["added"]
    assert added["added"] == 2 and added["folders"][0]["count"] == 2
    empty = tmp / "empty"
    empty.mkdir()
    added = api.post("/api/queue/add/", {"paths": [str(empty)]}).json()["added"]
    assert added["added"] == 0 and added["empty_folders"] == [str(empty)]

    assert api.post("/api/jobs/", {"settings": _settings()}).status_code == 200
    assert api.wait_job()["ok"] == 2
    a_out = root / OUTPUT_DIR_NAME / "trip" / "a_x4.png"
    b_out = root / OUTPUT_DIR_NAME / "b_x4.png"
    with Image.open(a_out) as im:
        assert im.mode == "RGBA" and im.size == (160, 240)
        alpha = np.asarray(im.getchannel("A"))
        assert alpha[120, 80] < 30 and alpha[10, 10] == 255      # stripe stays transparent
    with Image.open(b_out) as im:
        assert im.size == (240, 160) and im.info.get("icc_profile") == srgb

    # Second run: no overwrite, (1) suffix.
    assert api.post("/api/jobs/", {"settings": _settings()}).status_code == 200
    assert api.wait_job()["ok"] == 2
    assert (root / OUTPUT_DIR_NAME / "b_x4(1).png").exists()

    # Batch rename + JPEG (alpha flattened onto white).
    assert api.post("/api/jobs/", {"settings": _settings(rename_batch=True,
                                                         output_format="jpg")}).status_code == 200
    job = api.wait_job()
    assert job["ok"] == 2
    names = sorted(p.name for p in (root / OUTPUT_DIR_NAME).rglob("*.jpg"))
    assert len(names) == 2 and all(re.fullmatch(r"[A-Z]{4,5}1\.jpg", n) for n in names), names
    assert not list(root.rglob("*.part")), "temporary files must not be left behind"
    print("  folder mode, alpha, ICC profile, no-overwrite and batch rename OK")


def test_cancel(tmp: Path) -> None:
    ws = reset_workspace(engine_factory=StubEngine)
    api = Api(Client(enforce_csrf_checks=True))
    api.get("/")
    folder = tmp / "cancel"
    folder.mkdir()
    for i in range(4):
        make_marked(folder / f"c{i}.png", 30, 20)
    api.post("/api/queue/add/", {"paths": [str(folder)]})
    StubEngine.delay = 0.5
    try:
        assert api.post("/api/jobs/", {"settings": _settings()}).status_code == 200
        assert api.post("/api/jobs/", {"settings": _settings()}).status_code == 409   # busy
        time.sleep(0.2)
        api.post("/api/jobs/current/cancel/")
        job = api.wait_job()
    finally:
        StubEngine.delay = 0.0
    assert job["state"] == "cancelled" and job["ok"] == 0, job
    assert not list(folder.rglob(f"{OUTPUT_DIR_NAME}/*")), "a cancelled image must not be saved"
    # Cancel arriving just after a batch ended must not leave it "cancelling" forever.
    finished = ws.job
    assert finished.state == "cancelled"
    finished.state = "done"
    finished.cancel()
    assert finished.state == "done" and not finished.running
    assert api.post("/api/jobs/", {"settings": _settings()}).status_code == 200   # not "busy"
    api.wait_job()

    # Shutdown while running stops the batch promptly.
    StubEngine.delay = 5.0
    try:
        api.post("/api/jobs/", {"settings": _settings()})
        started = time.monotonic()
        ws.shutdown(timeout=3.0)
        assert not ws.job.running and time.monotonic() - started < 3.0
    finally:
        StubEngine.delay = 0.0
    print("  cancel / busy / shutdown OK")


def test_independent_of_old_upscaler() -> None:
    old = re.compile(r"app\.engine|upscale_page|realesrgan_upscaler|rrdbnet|UpscaleSettings|BatchWorker")
    for folder in (ROOT / "app" / "upscaler", ROOT / "app" / "web"):
        for path in folder.rglob("*"):
            if path.suffix in (".py", ".html", ".js", ".css"):
                assert not old.search(path.read_text(encoding="utf-8")), f"{path} uses the old upscaler"
    page = (ROOT / "app" / "ui" / "upscaler_web_page.py").read_text(encoding="utf-8")
    assert not old.search(page)
    # The Metadata screen keeps its own workflow and is not wired to upscaling.
    meta = (ROOT / "app" / "ui" / "metadata_page.py").read_text(encoding="utf-8")
    assert "app.upscaler" not in meta and "upscaler_web_page" not in meta
    print("  independent of the old upscaler; Metadata untouched by it")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        try:
            test_geometry()
            test_form()
            test_files(t)
            test_imaging(t)
            test_security()
            test_page()
            test_batches(t)
            test_folder_mode_alpha_icc_rename(t)
            test_cancel(t)
            test_independent_of_old_upscaler()
        finally:
            reset_workspace()
    print("UPSCALER TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
