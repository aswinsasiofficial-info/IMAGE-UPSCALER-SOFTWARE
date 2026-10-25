"""Tests for the Upscaler core — no model, no GPU, no network, no window.

Run with:  python -m tests.upscaler_test

The Real-ESRGAN network is replaced by an exact nearest-neighbour 4x stand-in
so every pixel is predictable. Covers the 1×–5× size planning (proportional,
1× keeps the size, 1024×1536 at 4× → 4096×6144) and the honest description
of how each factor is produced; the Django settings form; full batches that
write real files for portrait and landscape images at every factor —
checking size, aspect ratio and that the whole picture survives — with the
JPEG and batch-rename defaults, no overwriting, folder mode, transparency,
colour profiles and cancel; and that no website/web server is left.
"""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from app.upscaler import imaging
from app.upscaler.engine import Cancelled
from app.upscaler.files import OUTPUT_DIR_NAME, SequentialNamer, discover, unique_path
from app.upscaler.forms import UpscaleForm
from app.upscaler.geometry import (DETAIL_NOTE, FACTORS, MODEL_SCALE, describe, how_produced,
                                   plan_size)
from app.upscaler.workspace import JobOptions, Workspace

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


def assert_full_content(upright: Image.Image, out: Image.Image, tol: float = 2.0) -> None:
    """Shrinking the output back must reproduce the source: nothing cropped or shifted."""
    back = np.asarray(out.convert("RGB").resize(upright.size, Image.Resampling.BOX), np.float64)
    diff = np.abs(back - np.asarray(upright.convert("RGB"), np.float64)).mean()
    assert diff <= tol, f"output does not contain the whole picture (mean diff {diff:.2f})"
    arr = np.asarray(out.convert("RGB"))
    h, w = arr.shape[:2]
    inset_x, inset_y = max(2, w // 40), max(2, h // 40)
    corners = {"tl": arr[inset_y, inset_x], "tr": arr[inset_y, w - 1 - inset_x],
               "bl": arr[h - 1 - inset_y, inset_x], "br": arr[h - 1 - inset_y, w - 1 - inset_x]}
    for name, color in corners.items():
        assert np.abs(color.astype(int) - MARKERS[name]).max() <= 24, (name, color)


def run_batch(ws: Workspace, options: JobOptions, timeout: float = 120) -> dict:
    job = ws.start(options)
    job.join(timeout)
    assert not job.running, "batch did not finish in time"
    return job.snapshot()


# -- planning -----------------------------------------------------------------------
def test_geometry() -> None:
    assert FACTORS == (1, 2, 3, 4, 5) and MODEL_SCALE == 4
    assert plan_size(1024, 1536, 4).output == (4096, 6144)
    assert plan_size(1536, 1024, 4).output == (6144, 4096)
    assert plan_size(1024, 1536, 1).output == (1024, 1536)          # 1× keeps the size
    for w, h in [(1024, 1536), (1536, 1024), (1001, 667), (7, 3), (4000, 4000)]:
        for f in FACTORS:
            ow, oh = plan_size(w, h, f).output
            assert (ow, oh) == (w * f, h * f) and ow * h == oh * w, (w, h, f)  # exact ratio
    for bad in (0, 6, 2.5, 8):
        try:
            plan_size(100, 100, bad)
            raise AssertionError(f"factor {bad} accepted")
        except ValueError:
            pass
    assert "whole picture kept" in describe(plan_size(10, 20, 3))
    assert describe(plan_size(10, 20, 1)).startswith("Same size")

    # Honest about the model: only 4× is native; others are explained.
    assert how_produced(4).startswith("Native")
    assert "reduced back to the original size" in how_produced(1)
    for f in (2, 3):
        assert "native 4×" in how_produced(f) and "reduced" in how_produced(f)
    assert "interpolation" in how_produced(5) and "not new detail" in how_produced(5)
    assert "Native" not in how_produced(5)
    assert "more real detail" in DETAIL_NOTE
    print("  1×–5× planning + honest factor descriptions OK")


def test_form() -> None:
    base = dict(factor=4, output_dir="", save_beside=False, device="cpu", fp16=False, tile=64,
                output_format="jpg", jpg_quality=95, rename_batch=True)
    imgs = [(1024, 1536), (1536, 1024)]
    form = UpscaleForm(base, images=imgs)
    assert form.is_valid(), form.errors
    opts = form.job_options()
    assert (opts.factor, opts.output_format, opts.rename_batch) == (4, "jpg", True)
    for f in FACTORS:
        assert UpscaleForm({**base, "factor": f}, images=imgs).is_valid()
    for bad in (0, 6, "2.5", "x"):
        assert "factor" in UpscaleForm({**base, "factor": bad}, images=imgs).errors, bad
    form = UpscaleForm({**base, "factor": 5}, images=[(14000, 100)])        # 70000 px > limit
    assert not form.is_valid() and "factor" in form.errors and form.first_error().startswith("Upscale factor:")
    assert "jpg_quality" in UpscaleForm({**base, "jpg_quality": ""}, images=imgs).errors
    assert UpscaleForm({**base, "output_format": "png", "jpg_quality": ""}, images=imgs).is_valid()
    assert "output_dir" in UpscaleForm({**base, "output_dir": __file__}, images=imgs).errors
    assert UpscaleForm({**base, "output_dir": __file__, "save_beside": True}, images=imgs).is_valid()
    assert {"suffix", "exact_width", "fit"}.isdisjoint(UpscaleForm.base_fields)   # removed options
    assert UpscaleForm.base_fields["output_format"].initial == "jpg"
    assert UpscaleForm.base_fields["rename_batch"].initial is True
    print("  Django settings form OK")


def test_files(tmp: Path) -> None:
    root = tmp / "files"
    (root / "sub").mkdir(parents=True)
    (root / OUTPUT_DIR_NAME / "sub").mkdir(parents=True)
    for p in (root / "a.png", root / "sub" / "b.jpg", root / OUTPUT_DIR_NAME / "old.png",
              root / OUTPUT_DIR_NAME / "sub" / "b.png"):
        Image.new("RGB", (4, 4)).save(p)
    (root / "notes.txt").write_text("x")
    found = discover([root, root / "a.png"])
    assert [p.name for p in found] == ["a.png", "b.jpg"], found   # upscaled/ skipped, deduped

    taken = tmp / "taken.png"
    Image.new("RGB", (2, 2)).save(taken)
    assert unique_path(taken).name == "taken(1).png"
    namer = SequentialNamer("ABCD")
    assert namer.next_path(tmp, ".jpg").name == "ABCD1.jpg"
    assert namer.next_path(tmp, ".jpg").name == "ABCD2.jpg"
    namer.release(tmp)
    assert namer.next_path(tmp, ".jpg").name == "ABCD2.jpg"
    Image.new("RGB", (2, 2)).save(tmp / "WXYZ1.jpg")
    assert SequentialNamer("WXYZ").next_path(tmp, ".jpg").name == "WXYZ2.jpg"   # skips taken
    assert re.fullmatch(r"[A-Z]{4,5}", SequentialNamer().prefix)
    assert JobOptions().output_format == "jpg" and JobOptions().rename_batch is True
    print("  discovery, batch naming, no-overwrite OK")


def test_imaging(tmp: Path) -> None:
    p = tmp / "phone.jpg"
    upright = make_marked(p, 60, 90, orientation=6)
    with Image.open(p) as raw:
        assert raw.size == (90, 60)
    info = imaging.probe(p)
    assert (info.width, info.height) == (60, 90)
    src = imaging.load(p)
    assert src.size == (60, 90)
    assert np.abs(src.rgb.astype(int) - np.asarray(upright, int)).mean() < 4
    p16 = tmp / "grey16.png"
    Image.fromarray(np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 16).save(p16)
    g = imaging.load(p16).rgb
    assert 0 < g.mean() < 255 and g.min() < 20 and g.max() > 200
    print("  imaging (EXIF, 16-bit) OK")


def test_batches(tmp: Path) -> None:
    StubEngine.loads = 0
    src = tmp / "in"
    src.mkdir()
    portrait = make_marked(src / "portrait.png", 1024, 1536)
    landscape = make_marked(src / "landscape.png", 1536, 1024)
    phone = make_marked(src / "phone.jpg", 300, 450, orientation=6)

    ws = Workspace(StubEngine)
    result = ws.add([str(src / "portrait.png"), str(src / "landscape.png"), str(src / "phone.jpg")])
    assert len(result.added) == 3
    sizes = {i.path.name: (i.width, i.height) for i in ws.snapshot_items()}
    assert sizes == {"portrait.png": (1024, 1536), "landscape.png": (1536, 1024),
                     "phone.jpg": (300, 450)}, sizes

    # Defaults: 4×, JPEG, batch rename.
    out = tmp / "out4"
    snap = run_batch(ws, JobOptions(output_dir=out))
    assert snap["state"] == "done" and snap["ok"] == 3 and StubEngine.loads == 1, snap
    files = sorted(out.iterdir())
    assert len(files) == 3 and all(f.suffix == ".jpg" for f in files)
    prefixes = {re.fullmatch(r"([A-Z]{4,5})([123])\.jpg", f.name).group(1) for f in files}
    assert len(prefixes) == 1, files                                   # one shared prefix
    by_source = {}
    for item_id, res in snap["results"].items():
        name = next(i.path.name for i in ws.snapshot_items() if i.id == item_id)
        by_source[name] = Path(res["output"])
    for name, upright in (("portrait.png", portrait), ("landscape.png", landscape),
                          ("phone.jpg", phone)):
        with Image.open(by_source[name]) as im:
            assert im.format == "JPEG"
            assert im.size == (upright.width * 4, upright.height * 4), (name, im.size)
            assert_full_content(upright, im, tol=4.0)
    print("  4× JPEG batch: 1024×1536 → 4096×6144, 1536×1024 → 6144×4096, EXIF portrait "
          "300×450 → 1200×1800; one rename prefix; whole picture kept")

    # Every factor on small images, PNG so pixels can be compared exactly.
    small = tmp / "small"
    small.mkdir()
    p_small = make_marked(small / "p.png", 120, 180)
    l_small = make_marked(small / "l.png", 180, 120)
    for factor in FACTORS:
        ws = Workspace(StubEngine)
        ws.add([str(small / "p.png"), str(small / "l.png")])
        out = tmp / f"x{factor}"
        snap = run_batch(ws, JobOptions(factor=factor, output_dir=out, output_format="png",
                                        rename_batch=False))
        assert snap["ok"] == 2, snap
        for name, upright in (("p", p_small), ("l", l_small)):
            with Image.open(out / f"{name}.png") as im:     # rename off: original names
                assert im.size == (upright.width * factor, upright.height * factor), (factor, im.size)
                assert im.width * upright.height == im.height * upright.width
                assert_full_content(upright, im, tol=4.0 if factor in (1, 4) else 6.0)
    print("  1×, 2×, 3×, 4×, 5×: exact proportional sizes, whole picture kept")

    # Never overwrites: same batch again into the same folder.
    ws = Workspace(StubEngine)
    ws.add([str(small / "p.png")])
    run_batch(ws, JobOptions(factor=2, output_dir=tmp / "x2", output_format="png", rename_batch=False))
    assert (tmp / "x2" / "p(1).png").exists() and (tmp / "x2" / "p.png").exists()
    before = sorted(p.name for p in (tmp / "x4").iterdir())
    ws = Workspace(StubEngine)
    ws.add([str(small / "p.png"), str(small / "l.png")])
    run_batch(ws, JobOptions(factor=4, output_dir=tmp / "x4"))      # rename into a used folder
    after = sorted(p.name for p in (tmp / "x4").iterdir())
    assert set(before) <= set(after) and len(after) == len(before) + 2
    print("  existing files never overwritten (rename and original names)")


def test_folder_mode_alpha_icc(tmp: Path) -> None:
    root = tmp / "photos"
    (root / "trip").mkdir(parents=True)
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    make_marked(root / "trip" / "a.png", 40, 60, alpha=True)
    make_marked(root / "b.jpg", 60, 40, icc=srgb)
    (root / OUTPUT_DIR_NAME).mkdir()
    Image.new("RGB", (8, 8)).save(root / OUTPUT_DIR_NAME / "ignored.png")

    ws = Workspace(StubEngine)
    result = ws.add([str(root)])
    assert len(result.added) == 2 and result.folders[0][1] == 2
    empty = tmp / "empty"
    empty.mkdir()
    assert Workspace(StubEngine).add([str(empty)]).empty_folders == [empty]

    snap = run_batch(ws, JobOptions(factor=2, output_format="png"))
    assert snap["ok"] == 2
    trip_out = list((root / OUTPUT_DIR_NAME / "trip").glob("*.png"))
    top_out = [p for p in (root / OUTPUT_DIR_NAME).glob("*.png") if p.name != "ignored.png"]
    assert len(trip_out) == 1 and len(top_out) == 1
    assert re.fullmatch(r"[A-Z]{4,5}1\.png", trip_out[0].name)      # numbering per folder
    with Image.open(trip_out[0]) as im:
        assert im.mode == "RGBA" and im.size == (80, 120)
        alpha = np.asarray(im.getchannel("A"))
        assert alpha[60, 40] < 30 and alpha[5, 5] == 255            # stripe stays transparent
    with Image.open(top_out[0]) as im:
        assert im.size == (120, 80) and im.info.get("icc_profile") == srgb

    # JPEG (the default) flattens transparency onto white.
    snap = run_batch(ws, JobOptions(factor=1))
    jpgs = list((root / OUTPUT_DIR_NAME / "trip").glob("*.jpg"))
    assert snap["ok"] == 2 and len(jpgs) == 1
    with Image.open(jpgs[0]) as im:
        assert im.mode == "RGB" and im.size == (40, 60)
    assert not list(root.rglob("*.part")), "temporary files must not be left behind"
    print("  folder mode, transparency, colour profile OK")


def test_cancel(tmp: Path) -> None:
    folder = tmp / "cancel"
    folder.mkdir()
    for i in range(4):
        make_marked(folder / f"c{i}.png", 30, 20)
    ws = Workspace(StubEngine)
    ws.add([str(folder)])
    StubEngine.delay = 0.5
    try:
        job = ws.start(JobOptions())
        try:
            ws.start(JobOptions())
            raise AssertionError("a second batch must be refused while one runs")
        except RuntimeError:
            pass
        time.sleep(0.2)
        ws.cancel()
        job.join(10)
    finally:
        StubEngine.delay = 0.0
    snap = job.snapshot()
    assert snap["state"] == "cancelled" and snap["ok"] == 0, snap
    assert not list(folder.rglob(f"{OUTPUT_DIR_NAME}/*")), "a cancelled image must not be saved"
    # A cancel arriving after the end changes nothing.
    job.cancel()
    assert job.snapshot()["state"] == "cancelled"
    StubEngine.delay = 5.0
    try:
        ws.start(JobOptions())
        started = time.monotonic()
        ws.shutdown(timeout=3.0)
        assert not ws.job.running and time.monotonic() - started < 3.0
    finally:
        StubEngine.delay = 0.0
    print("  cancel / busy / shutdown OK")


def test_desktop_only() -> None:
    """No website: no web server, pages, URL routes or browser mode remain."""
    assert not (ROOT / "app" / "web").exists() and not (ROOT / "manage.py").exists()
    assert not list((ROOT / "app").rglob("*.html")) and not list((ROOT / "app").rglob("*.js"))
    banned = re.compile(r"QtWebEngine|runserver|django\.urls|django\.http|WSGI|http\.server")
    for path in (ROOT / "app").rglob("*.py"):
        assert not banned.search(path.read_text(encoding="utf-8")), f"web code left in {path}"
    # Metadata and Upscaler stay separate: neither screen starts the other.
    meta = (ROOT / "app" / "ui" / "metadata_page.py").read_text(encoding="utf-8")
    ups = (ROOT / "app" / "ui" / "upscaler_screen.py").read_text(encoding="utf-8")
    assert "app.upscaler" not in meta and "metadata" not in ups.lower().replace(
        "metadata is a separate screen", "")
    print("  desktop-only: no web layer; Upscaler and Metadata independent")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        test_geometry()
        test_form()
        test_files(t)
        test_imaging(t)
        test_batches(t)
        test_folder_mode_alpha_icc(t)
        test_cancel(t)
        test_desktop_only()
    print("UPSCALER TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
