"""Background batch-processing worker for the GUI.

Runs the whole batch on a ``QThread`` so the UI stays responsive, emitting
signals for per-image progress, logs and completion. The heavy model is loaded
once at the start of the run and reused for every image (the key to efficient
batching), and cancellation is checked between images.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .realesrgan_upscaler import RealESRGANUpscaler, UpscaleSettings

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(paths: list[Path]) -> list[Path]:
    """Expand a mix of files/folders into a de-duplicated, sorted image list."""
    found: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                    found.append(child)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            found.append(p)
    # De-dup while preserving order, skip files that are already upscaled output.
    seen: set[str] = set()
    result: list[Path] = []
    for f in found:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            result.append(f)
    return result


def resolve_output_path(src: Path, out_dir: Path, settings: UpscaleSettings) -> Path:
    ext = ".jpg" if settings.output_format == "jpg" else ".png"
    suffix = settings.filename_suffix if settings.keep_original_name else ""
    return out_dir / f"{src.stem}{suffix}{ext}"


class BatchWorker(QThread):
    log_message = Signal(str)                       # human-readable log line
    progress = Signal(int, int)                     # (images_done, total)
    image_started = Signal(str)                     # filename being processed
    image_done = Signal(str, str)                   # (filename, output path)
    image_failed = Signal(str, str)                 # (filename, error message)
    download_progress = Signal(int, int)            # (bytes, total) for first-run weights
    finished_all = Signal(int, int)                 # (succeeded, failed)
    cancelled = Signal()

    def __init__(self, files: list[Path], out_dir: Path, settings: UpscaleSettings) -> None:
        super().__init__()
        self.files = files
        self.out_dir = Path(out_dir)
        self.settings = settings
        self._cancel = False
        self._upscaler: RealESRGANUpscaler | None = None

    def request_cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102 (QThread entry point)
        total = len(self.files)
        succeeded = failed = 0
        try:
            if not self.settings.output_in_source_dir:
                self.out_dir.mkdir(parents=True, exist_ok=True)
            self._upscaler = RealESRGANUpscaler(self.settings)
            self.log_message.emit("Loading model…")
            self._upscaler.load(
                log=self.log_message.emit,
                progress_cb=lambda d, t: self.download_progress.emit(d, t),
            )
        except Exception as exc:
            self.log_message.emit(f"✗ Setup failed: {exc}")
            self.finished_all.emit(0, total)
            return

        for idx, src in enumerate(self.files):
            if self._cancel:
                self.log_message.emit("Cancelled by user.")
                self.cancelled.emit()
                break
            self.image_started.emit(src.name)
            try:
                base_dir = src.parent if self.settings.output_in_source_dir else self.out_dir
                out_path = resolve_output_path(src, base_dir, self.settings)
                out_path = self._avoid_collision(out_path)
                self._upscaler.upscale_image(src, out_path, log=self.log_message.emit)
                succeeded += 1
                self.image_done.emit(src.name, str(out_path))
            except Exception as exc:
                failed += 1
                self.image_failed.emit(src.name, str(exc))
                self.log_message.emit(f"✗ {src.name}: {exc}")
            finally:
                self.progress.emit(idx + 1, total)

        if self._upscaler is not None:
            self._upscaler.cleanup()
        if not self._cancel:
            self.log_message.emit(f"Batch complete — {succeeded} ok, {failed} failed.")
        self.finished_all.emit(succeeded, failed)

    def _avoid_collision(self, out_path: Path) -> None:
        """Never overwrite an existing input/output file silently."""
        if out_path.exists():
            stem, suffix = out_path.stem, out_path.suffix
            n = 1
            while out_path.exists():
                out_path = out_path.parent / f"{stem}({n}){suffix}"
                n += 1
        return out_path
