"""Real-ESRGAN x4plus inference for the Upscaler.

The network always enlarges by exactly 4x on both axes; every other output
size is reached afterwards by :mod:`app.upscaler.imaging` from the geometry
plan. Large images run in overlapping tiles so a 6 GB GPU never runs out of
memory, results go straight into an 8-bit canvas to keep RAM use low, and if
CUDA still runs out of memory the tile size is halved and the image retried.

torch is imported lazily so the Django pages load without it.
"""

from __future__ import annotations

import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from app import paths

from .geometry import MODEL_SCALE

WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHTS_FILE = paths.WEIGHTS_FILE
WEIGHTS_MIN_BYTES = 40 * 1024 * 1024    # the official file is ~64 MB; less means truncated
# models/ beside the app (or the user's data folder for a packaged app without weights).
MODEL_DIR = paths.model_dir()
TILE_PAD = 16       # context added around each tile so seams don't show
MIN_TILE = 64

Log = Callable[[str], None]
Progress = Callable[[int, int], None]


def _quiet(_message: str) -> None:
    pass


class Cancelled(Exception):
    """Raised between tiles when the user cancels the batch."""


class WeightsError(RuntimeError):
    """The pretrained weights could not be obtained."""


def weights_path(model_dir: Path = MODEL_DIR) -> Path:
    return Path(model_dir) / WEIGHTS_FILE


def weights_ready(model_dir: Path = MODEL_DIR) -> bool:
    path = weights_path(model_dir)
    return path.is_file() and path.stat().st_size >= WEIGHTS_MIN_BYTES


def fetch_weights(model_dir: Path = MODEL_DIR, progress: Optional[Progress] = None,
                  log: Log = _quiet) -> Path:
    """Return the weights file, downloading it once if needed."""
    target = weights_path(model_dir)
    if weights_ready(model_dir):
        log(f"Using cached weights: {target.name}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        log("Removing an incomplete weights file and downloading again…")
        target.unlink()

    log(f"Downloading Real-ESRGAN x4plus weights from {WEIGHTS_URL}")
    part = target.with_suffix(".part")
    try:
        request = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "BatchUpscaler/2"})
        with urllib.request.urlopen(request, timeout=60) as resp, open(part, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(256 * 1024):
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    except OSError as exc:      # URLError, timeouts and disk errors are all OSErrors
        part.unlink(missing_ok=True)
        raise WeightsError(f"Could not download the model weights: {exc}") from exc
    if part.stat().st_size < WEIGHTS_MIN_BYTES:
        part.unlink(missing_ok=True)
        raise WeightsError("The downloaded weights file is incomplete — try again.")
    part.replace(target)
    log(f"Weights saved to {target}")
    return target


def recommended_tile(vram_gb: float) -> int:
    """Safe tile size for the available VRAM (RTX 3050 6 GB -> 256)."""
    for min_gb, tile in ((12, 512), (8, 384), (6, 256), (4, 192)):
        if vram_gb >= min_gb:
            return tile
    return 128


@lru_cache(maxsize=1)
def probe_device() -> dict:
    """Best-effort CUDA probe, done once per process."""
    info = {"cuda": False, "name": "", "vram_gb": 0.0, "torch": "", "error": ""}
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment problem
        info["error"] = f"PyTorch is not available: {exc}"
        return info
    info["torch"] = torch.__version__
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(cuda=True, name=props.name,
                        vram_gb=round(props.total_memory / 1024 ** 3, 1))
    except Exception as exc:  # pragma: no cover - driver problem
        info["error"] = str(exc)
    return info


class RealESRGAN:
    """Loads the model once and upscales any number of images with it."""

    def __init__(self, device: str = "auto", fp16: bool = True, tile: int = 256,
                 model_dir: Path = MODEL_DIR) -> None:
        self.device_preference = device       # "auto" | "cuda" | "cpu"
        self.fp16_preference = fp16
        self.tile = max(0, int(tile))         # 0 = whole image in one pass
        self.model_dir = Path(model_dir)
        self._torch = None
        self._model = None
        self._device = None
        self._half = False

    def load(self, log: Log = _quiet, progress: Optional[Progress] = None) -> None:
        weights = fetch_weights(self.model_dir, progress=progress, log=log)
        import torch

        from .network import RRDBNetX4

        self._torch = torch
        use_cuda = self.device_preference != "cpu" and torch.cuda.is_available()
        if self.device_preference != "cpu" and not use_cuda:
            log("CUDA not available - falling back to CPU (much slower).")
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self._half = self.fp16_preference and use_cuda

        state = torch.load(weights, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and ("params_ema" in state or "params" in state):
            state = state.get("params_ema", state.get("params"))   # EMA weights are the inference ones
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model = RRDBNetX4()
        model.load_state_dict(state, strict=True)
        model.eval().to(self._device)
        if self._half:
            model.half()
        self._model = model
        name = f" ({torch.cuda.get_device_name(0)})" if use_cuda else ""
        log(f"Device: {self._device.type.upper()}{name} | Half precision: {self._half}")
        log("Model loaded and ready.")

    def upscale(self, rgb: np.ndarray, cancelled: Callable[[], bool] = lambda: False,
                log: Log = _quiet) -> np.ndarray:
        """(H, W, 3) uint8 RGB -> (4H, 4W, 3) uint8 RGB."""
        if self._model is None:
            raise RuntimeError("load() must be called first")
        torch = self._torch
        h, w = rgb.shape[:2]
        while True:
            try:
                return self._run(rgb, self.tile, cancelled)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                current = self.tile or max(h, w)
                if current <= MIN_TILE:
                    raise MemoryError("Out of GPU memory even with the smallest tile size.")
                # Keep the smaller tile for the rest of the batch.
                self.tile = max(MIN_TILE, current // 2)
                log(f"Out of GPU memory — retrying with tile size {self.tile}.")

    def _run(self, rgb: np.ndarray, tile: int, cancelled: Callable[[], bool]) -> np.ndarray:
        torch = self._torch
        s = MODEL_SCALE
        h, w = rgb.shape[:2]
        out = np.empty((h * s, w * s, 3), dtype=np.uint8)
        with torch.inference_mode():
            x = torch.from_numpy(rgb).to(self._device).permute(2, 0, 1).unsqueeze(0)
            x = (x.half() if self._half else x.float()).div_(255.0)
            if tile <= 0 or (h <= tile and w <= tile):
                if cancelled():
                    raise Cancelled()
                out[:] = self._to_uint8(self._model(x))
                return out
            for y0 in range(0, h, tile):
                for x0 in range(0, w, tile):
                    if cancelled():
                        raise Cancelled()
                    y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                    py0, px0 = max(y0 - TILE_PAD, 0), max(x0 - TILE_PAD, 0)
                    py1, px1 = min(y1 + TILE_PAD, h), min(x1 + TILE_PAD, w)
                    result = self._model(x[:, :, py0:py1, px0:px1])
                    # Keep only the tile's own area; the padding just gave the convs context.
                    core = result[:, :, (y0 - py0) * s:(y1 - py0) * s, (x0 - px0) * s:(x1 - px0) * s]
                    out[y0 * s:y1 * s, x0 * s:x1 * s] = self._to_uint8(core)
                    del result, core
        return out

    def _to_uint8(self, t) -> np.ndarray:
        t = t.float().clamp_(0.0, 1.0).mul_(255.0).round_().to(self._torch.uint8)
        return t[0].permute(1, 2, 0).cpu().numpy()

    def close(self) -> None:
        """Release the model (and VRAM — local Ollama on the Metadata page may need it)."""
        self._model = None
        if self._torch is not None and self._device is not None and self._device.type == "cuda":
            self._torch.cuda.empty_cache()
