"""Folder-mode path logic test (no model / no GPU required).

Verifies the "Add Folder" workflow: recursive scanning, exclusion of the
``upscaled`` output subfolder, relative-structure-preserving output
directories, fallback to the classic output modes for non-folder images,
existing naming/collision behaviour, and a graceful empty-folder scan.
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
from app.engine.worker import (
    UPSCALED_DIR_NAME,
    BatchWorker,
    collect_images,
    resolve_output_path,
)


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def main() -> int:
    app = QApplication.instance() or QApplication([])  # noqa: F841 (Qt needs an instance)
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"  {'PASS' if cond else 'FAIL'} {name}")
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "photos"
        _png(root / "top.png")
        _png(root / "trip" / "image.png")
        _png(root / "trip" / "day2" / "image.png")   # same name, different subfolder
        (root / "trip" / "notes.txt").write_text("not an image")  # unsupported extension
        # Simulate a previous folder-mode run whose outputs must be ignored.
        _png(root / UPSCALED_DIR_NAME / "top_x4.png")
        _png(root / UPSCALED_DIR_NAME / "trip" / "image_x4.png")
        outside = Path(tmp) / "loose.png"
        _png(outside)

        # -- scanning --------------------------------------------------------
        found = collect_images([root])
        check("scan finds 3 originals recursively", len(found) == 3)
        check("scan skips the 'upscaled' output folder",
              not any(UPSCALED_DIR_NAME in f.parts for f in found))
        check("scan ignores unsupported extensions",
              all(f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
                  for f in found))

        # -- empty folder ----------------------------------------------------
        empty = Path(tmp) / "empty"
        empty.mkdir()
        check("empty folder yields no images", collect_images([empty]) == [])
        only_outputs = Path(tmp) / "done"
        _png(only_outputs / UPSCALED_DIR_NAME / "x_x4.png")
        check("folder with only 'upscaled' outputs scans as empty",
              collect_images([only_outputs]) == [])

        # -- output path mapping ---------------------------------------------
        settings = UpscaleSettings(device="auto", output_format="jpg", filename_suffix="_x4")
        out_dir = Path(tmp) / "out"
        worker = BatchWorker(found, out_dir, settings, folder_roots=[root])

        base_trip = worker.output_base_dir(root / "trip" / "image.png")
        check("subfolder image maps to <folder>/upscaled/<subfolder>",
              base_trip == root / UPSCALED_DIR_NAME / "trip")
        base_day2 = worker.output_base_dir(root / "trip" / "day2" / "image.png")
        check("deeper subfolder structure preserved",
              base_day2 == root / UPSCALED_DIR_NAME / "trip" / "day2")
        check("same-named files from different subfolders do not collide",
              resolve_output_path(root / "trip" / "image.png", base_trip, settings)
              != resolve_output_path(root / "trip" / "day2" / "image.png", base_day2, settings))
        base_top = worker.output_base_dir(root / "top.png")
        check("root-level image maps to <folder>/upscaled",
              base_top == root / UPSCALED_DIR_NAME)

        out = resolve_output_path(root / "trip" / "image.png", base_trip, settings)
        check("naming/format settings honoured (stem + suffix + ext)",
              out.name == "image_x4.jpg" and out.parent == base_trip)

        # -- non-folder images keep the classic behaviour ---------------------
        worker_none = BatchWorker([outside], out_dir, settings)
        check("loose image still uses the global output folder",
              worker_none.output_base_dir(outside) == out_dir)
        beside = UpscaleSettings(device="auto", output_in_source_dir=True)
        worker_beside = BatchWorker([outside], out_dir, beside)
        check("'save beside original' still honoured for non-folder images",
              worker_beside.output_base_dir(outside) == outside.parent)
        check("folder mode takes precedence over global output folder",
              worker.output_base_dir(root / "top.png") != out_dir)

        # -- collision behaviour unchanged ------------------------------------
        clash = root / UPSCALED_DIR_NAME / "top_x4.jpg"
        clash.parent.mkdir(parents=True, exist_ok=True)
        _png(clash)
        moved = BatchWorker([], out_dir, settings)._avoid_collision(clash)
        check("existing outputs are never silently overwritten",
              moved is not None and moved.name == "top_x4(1).jpg")

    print("FOLDER MODE TEST PASSED" if not failures
          else f"FOLDER MODE TEST FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
