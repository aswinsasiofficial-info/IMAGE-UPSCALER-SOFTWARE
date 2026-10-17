"""Real-ESRGAN x4plus inference engine (torch + CUDA, VRAM-friendly tiling).

Responsibilities:
  * Locate / download the official ``RealESRGAN_x4plus.pth`` weights and cache
    them under ``<project>/models``.
  * Build the RRDBNet, move it to the chosen device (CUDA when available) and
    optionally run in half precision.
  * Upscale a single image file end-to-end: EXIF-aware load, alpha (RGBA)
    handling, tile-based inference so a 6GB RTX 3050 never runs out of memory,
    and save as PNG/JPEG.

The model is loaded once via :meth:`RealESRGANUpscaler.load` and reused for the
whole batch — that reuse is what makes batch processing of 20-30 images fast.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image, ImageOps

from .rrdbnet import RRDBNet

# --- Model registry -----------------------------------------------------------
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_FILENAME = "RealESRGAN_x4plus.pth"
# Official checkpoint is ~67 MB; anything far smaller is a truncated/failed file.
MIN_VALID_BYTES = 40 * 1024 * 1024

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

ImageLog = Callable[[str], None]


class WeightDownloadError(RuntimeError):
    """Raised when the pretrained weights cannot be obtained."""


@dataclass
class UpscaleSettings:
    """Tunable knobs surfaced in the GUI."""

    device: str = "cuda"          # "cuda" | "cpu" | "auto"
    fp16: bool = True             # half precision (big speed/VRAM win on RTX 30xx)
    tile: int = 256               # input tile size in px; 0 = no tiling (whole image)
    tile_pad: int = 16            # overlap between tiles (hides seams)
    output_scale: float = 4.0     # final size = src * output_scale (network is 4x; resampled to target)
    output_format: str = "png"    # "png" | "jpg"
    jpg_quality: int = 95
    filename_suffix: str = "_4x"
    keep_original_name: bool = True
    output_in_source_dir: bool = False  # True: write next to each source; False: use chosen folder

    @staticmethod
    def recommended_tile_for_vram(total_vram_gb: float) -> int:
        """Pick a safe tile size for the available VRAM (RTX 3050 = 6GB)."""
        if total_vram_gb >= 12:
            return 512
        if total_vram_gb >= 8:
            return 384
        if total_vram_gb >= 6:
            return 256
        if total_vram_gb >= 4:
            return 192
        return 128


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def ensure_weights(
    model_dir: Path = DEFAULT_MODEL_DIR,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log: Optional[ImageLog] = None,
) -> Path:
    """Return the path to the weights, downloading them on first use.

    ``progress_cb(downloaded_bytes, total_bytes)`` is called during download.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / MODEL_FILENAME

    if target.exists() and target.stat().st_size >= MIN_VALID_BYTES:
        if log:
            log(f"Using cached weights: {target.name} ({_human_bytes(target.stat().st_size)})")
        return target

    if target.exists() and target.stat().st_size < MIN_VALID_BYTES:
        if log:
            log("Removing truncated weight file and re-downloading...")
        target.unlink(missing_ok=True)

    if log:
        log(f"Downloading Real-ESRGAN x4plus weights from {MODEL_URL}")

    tmp = target.with_suffix(".part")
    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Upscaler/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            done = 0
            chunk = 256 * 1024
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if progress_cb:
                    progress_cb(done, total)
    except Exception as exc:  # network / IO errors
        tmp.unlink(missing_ok=True)
        raise WeightDownloadError(f"Failed to download weights: {exc}") from exc

    if not tmp.exists() or tmp.stat().st_size < MIN_VALID_BYTES:
        tmp.unlink(missing_ok=True)
        raise WeightDownloadError("Downloaded weight file looks incomplete.")

    tmp.replace(target)
    if log:
        log(f"Weights saved to {target} ({_human_bytes(target.stat().st_size)})")
    return target


class RealESRGANUpscaler:
    def __init__(self, settings: UpscaleSettings, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        self.settings = settings
        self.model_dir = Path(model_dir)
        self.model: Optional[RRDBNet] = None
        self.device: Optional[torch.device] = None
        self.use_fp16 = False
        self._loaded = False

    # -- setup ----------------------------------------------------------------
    @staticmethod
    def resolve_device(preference: str) -> torch.device:
        if preference == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        # "cuda" explicitly requested but unavailable, or "auto": fall back.
        return torch.device("cpu")

    def load(self, log: Optional[ImageLog] = None, progress_cb: Optional[Callable[[int, int], None]] = None) -> None:
        weights = ensure_weights(self.model_dir, progress_cb=progress_cb, log=log)
        self.device = self.resolve_device(self.settings.device)
        self.use_fp16 = self.settings.fp16 and self.device.type == "cuda"

        if log:
            if self.device.type == "cpu" and self.settings.device != "cpu":
                log("CUDA not available - falling back to CPU (much slower).")
            log(f"Device: {self.device.type.upper()}"
                + (f" ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else "")
                + f" | Half precision: {self.use_fp16}")

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        state = torch.load(weights, map_location="cpu")
        # Official checkpoint stores both 'params' and 'params_ema'; prefer EMA.
        if isinstance(state, dict) and ("params_ema" in state or "params" in state):
            state = state.get("params_ema", state.get("params"))
        # Strip any DataParallel 'module.' prefix defensively.
        state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        model.load_state_dict(state, strict=True)

        model.eval().to(self.device)
        if self.use_fp16:
            model.half()
        self.model = model
        self._loaded = True
        if log:
            log("Model loaded and ready.")

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.float16 if self.use_fp16 else torch.float32

    # -- image IO -------------------------------------------------------------
    @staticmethod
    def _read_image(path: Path) -> tuple[np.ndarray, Optional[Image.Image]]:
        """Return (RGB float array HxWx3 in [0,1], optional alpha PIL image)."""
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)
        has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
        alpha: Optional[Image.Image] = None
        if has_alpha:
            im = im.convert("RGBA")
            alpha = im.getchannel("A")
            rgb = im.convert("RGB")
        else:
            rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        return arr, alpha

    @staticmethod
    def _merge_alpha(rgb_uint8: np.ndarray, alpha: Optional[Image.Image]) -> np.ndarray:
        if alpha is None:
            return rgb_uint8
        h, w = rgb_uint8.shape[:2]
        alpha_resized = alpha.resize((w, h), Image.Resampling.BICUBIC)
        a = np.asarray(alpha_resized, dtype=np.uint8)
        rgba = np.dstack([rgb_uint8, a])
        return rgba

    @staticmethod
    def _resize_rgb(rgb_uint8: np.ndarray, w: int, h: int) -> np.ndarray:
        """Resample the native 4x result to the requested output size.

        Uses Lanczos for both up- and down-scaling; when the target already
        matches the 4x dimensions this is a no-op (fast path, no softening).
        """
        if (rgb_uint8.shape[1], rgb_uint8.shape[0]) == (w, h):
            return rgb_uint8
        img = Image.fromarray(rgb_uint8).resize((w, h), Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=np.uint8)

    def _save(self, array_hwc: np.ndarray, out_path: Path) -> None:
        img = Image.fromarray(array_hwc)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.settings.output_format == "jpg":
            # JPEG cannot store alpha; flatten onto white.
            if img.mode == "RGBA":
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img).convert("RGB")
            img.save(out_path, "JPEG", quality=int(self.settings.jpg_quality), optimize=True)
        else:
            img.save(out_path, "PNG", optimize=False)

    # -- inference ------------------------------------------------------------
    @torch.inference_mode()
    def _infer_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Run the network on a (1,3,H,W) float32 tensor -> (1,3,H*4,W*4) float32."""
        assert self.model is not None
        scale = 4
        tile = self.settings.tile
        h, w = t.shape[-2:]
        if tile <= 0 or (h <= tile and w <= tile):
            inp = t.to(self.device)
            if self.use_fp16:
                inp = inp.half()
            return self.model(inp).float().cpu()
        return self._tile_inference(t, scale, tile)

    def _tile_inference(self, t: torch.Tensor, scale: int, tile_size: int) -> torch.Tensor:
        """Split the input into overlapping tiles, upscale each, reassemble.

        Keeps peak VRAM bounded regardless of the source resolution, which is
        essential on a 6GB RTX 3050. Returns a CPU float32 tensor in [0,1].
        """
        assert self.model is not None
        pad = self.settings.tile_pad
        batch, channel, height, width = t.shape
        out_h, out_w = height * scale, width * scale
        output = torch.empty((batch, channel, out_h, out_w), dtype=torch.float32)

        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        # Pre-convert the whole image to the working dtype/device once.
        base = t.to(self.device)
        if self.use_fp16:
            base = base.half()

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Output tile bounds.
                ox_l, ox_r = tx * tile_size * scale, min((tx + 1) * tile_size, width) * scale
                oy_t, oy_b = ty * tile_size * scale, min((ty + 1) * tile_size, height) * scale
                # Input tile bounds, expanded by `pad` and clamped to the image.
                ix_l = max(tx * tile_size - pad, 0)
                ix_r = min((tx + 1) * tile_size + pad, width)
                iy_t = max(ty * tile_size - pad, 0)
                iy_b = min((ty + 1) * tile_size + pad, height)

                tile_in = base[:, :, iy_t:iy_b, ix_l:ix_r]
                tile_out = self.model(tile_in).float()

                # Only paste the non-overlapping core region.
                paste_left = ox_l - ix_l * scale
                paste_top = oy_t - iy_t * scale
                cropped = tile_out[:, :, paste_top:paste_top + (oy_b - oy_t),
                                  paste_left:paste_left + (ox_r - ox_l)]
                output[:, :, oy_t:oy_b, ox_l:ox_r] = cropped
                del tile_out, cropped, tile_in
        return output

    def upscale_image(self, in_path: Path, out_path: Path, log: Optional[ImageLog] = None) -> Path:
        """Upscale one image file. Assumes :meth:`load` already ran."""
        if not self._loaded:
            self.load(log=log)
        assert self.model is not None

        arr, alpha = self._read_image(in_path)
        src_h, src_w = arr.shape[:2]
        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).contiguous()

        out = self._infer_tensor(tensor)          # (1,3,src*4) float32 [0,1] — native 4x
        out = torch.clamp(out, 0.0, 1.0)
        rgb4x = (out.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

        # Resample the 4x neural result to the requested output scale.
        target_w = max(1, round(src_w * self.settings.output_scale))
        target_h = max(1, round(src_h * self.settings.output_scale))
        rgb = self._resize_rgb(rgb4x, target_w, target_h)
        rgb = self._merge_alpha(rgb, alpha)
        self._save(rgb, out_path)

        if log:
            log(f"[OK] {in_path.name}: {src_w}x{src_h} -> {target_w}x{target_h} "
                f"({self.settings.output_scale:g}x)")
        return out_path

    def cleanup(self) -> None:
        if self.model is not None and self.device is not None and self.device.type == "cuda":
            del self.model
            self.model = None
            self._loaded = False
            torch.cuda.empty_cache()
