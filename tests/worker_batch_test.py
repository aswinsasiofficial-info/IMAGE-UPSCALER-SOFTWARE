"""End-to-end test of the BatchWorker (the code path behind the Start button).

Builds a synthetic batch, runs the worker synchronously and asserts every image
is upscaled 4x and that the model is loaded only once for the whole batch.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from app.engine.realesrgan_upscaler import UpscaleSettings
from app.engine.worker import BatchWorker, collect_images, resolve_output_path


def _make_batch(folder: Path, n: int) -> None:
    rng = np.random.default_rng(1)
    for i in range(n):
        h, w = 40 + i, 32 + i
        arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(folder / f"img_{i}.png")


def main() -> int:
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "in"
        src.mkdir()
        out = root / "out"
        n = 6
        _make_batch(src, n)

        files = collect_images([src])
        assert len(files) == n, f"expected {n} images, found {len(files)}"

        settings = UpscaleSettings(device="auto", fp16=True, tile=128, tile_pad=8)
        worker = BatchWorker(files, out, settings)

        logs: list[str] = []
        done: list[str] = []
        failed: list[tuple[str, str]] = []
        progress: list[tuple[int, int]] = []
        worker.log_message.connect(logs.append)
        worker.image_done.connect(lambda name, path: done.append(path))
        worker.image_failed.connect(lambda name, err: failed.append((name, err)))
        worker.progress.connect(lambda d, t: progress.append((d, t)))

        result: dict[str, tuple[int, int]] = {}
        worker.finished_all.connect(lambda ok, bad: result.update(ok=ok, bad=bad))

        worker.run()  # run synchronously (no separate thread needed for the test)

        # Every output must exist and be exactly 4x its source.
        ok = 0
        for f in files:
            expected = resolve_output_path(f, out, settings)
            if expected.exists():
                with Image.open(expected) as im:
                    with Image.open(f) as src_im:
                        if im.size == (src_im.width * 4, src_im.height * 4):
                            ok += 1

        model_loads = sum(1 for line in logs if line == "Model loaded and ready.")
        print(f"processed={len(done)} failed={len(failed)} correct_outputs={ok}/{n}")
        print(f"model_load_count={model_loads} (must be 1) progress_events={len(progress)}")
        for name, err in failed:
            print(f"  FAIL {name}: {err}")

        if ok == n and len(failed) == 0 and model_loads == 1 and len(progress) == n:
            print("WORKER BATCH TEST PASSED")
            return 0
        print("WORKER BATCH TEST FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
