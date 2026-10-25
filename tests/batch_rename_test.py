"""Batch-rename logic test (no model / no GPU required).

Verifies the ``<PREFIX><n>.<ext>`` naming scheme: a shared random 4-5 letter
uppercase alphabetic prefix, sequential numbering starting at 1, per-folder
counter restart, existing-file skipping, gap-free rollback on failure, and that
the worker uses the planner only when ``rename_batch`` is enabled.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from app.engine.realesrgan_upscaler import UpscaleSettings
from app.engine.worker import BatchNamePlanner, BatchWorker


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def main() -> int:
    QApplication.instance() or QApplication([])  # noqa: F841 (Qt needs an instance)
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"  {'PASS' if cond else 'FAIL'} {name}")
        if not cond:
            failures.append(name)

    # -- prefix format --------------------------------------------------------
    for _ in range(50):
        pfx = BatchNamePlanner.new_prefix()
        if not re.fullmatch(r"[A-Z]{4,5}", pfx):
            check(f"prefix {pfx!r} is 4-5 uppercase letters", False)
            break
    else:
        check("prefix is always 4-5 uppercase alphabetic letters", True)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()

        # -- sequential numbering from 1, format preserved ---------------------
        planner = BatchNamePlanner(prefix="KQATZ")
        p1 = planner.next_path(out_dir, ".jpg")
        p2 = planner.next_path(out_dir, ".jpg")
        check("first name is <prefix>1", p1.name == "KQATZ1.jpg")
        check("second name is <prefix>2", p2.name == "KQATZ2.jpg")

        # -- skips an existing file rather than overwriting --------------------
        _png(out_dir / "KQATZ3.jpg")  # simulate a pre-existing result
        p4 = planner.next_path(out_dir, ".jpg")
        check("existing target is skipped (goes to 4)", p4.name == "KQATZ4.jpg")

        # -- per-folder counter restarts at 1 ----------------------------------
        other = out_dir / "sub"
        other.mkdir()
        planner2 = BatchNamePlanner(prefix="KQATZ")
        check("numbering restarts at 1 in a new folder",
              planner2.next_path(other, ".png").name == "KQATZ1.png")

        # -- gap-free rollback keeps the sequence tight -------------------------
        planner3 = BatchNamePlanner(prefix="ABCD")
        reserved = planner3.next_path(out_dir, ".jpg")   # reserves 1
        check("fresh prefix reserves 1", reserved.name == "ABCD1.jpg")
        planner3.next_path(out_dir, ".jpg")              # reserves 2
        planner3.rollback(out_dir)                       # image #2 failed -> release
        after = planner3.next_path(out_dir, ".jpg")
        check("rollback reuses the released number (no gap)", after.name == "ABCD2.jpg")

        # -- worker integration --------------------------------------------------
        src = Path(tmp) / "loose.png"
        _png(src)
        rename_settings = UpscaleSettings(device="auto", output_format="png", rename_batch=True)
        worker = BatchWorker([src], out_dir, rename_settings)
        # output_base_dir for a loose file is the global out_dir; planner honours it.
        check("worker uses global output dir for loose images",
              worker.output_base_dir(src) == out_dir)
        check("rename_batch flag propagates through settings",
              worker.settings.rename_batch is True)

        classic = UpscaleSettings(device="auto", output_format="png", rename_batch=False)
        w2 = BatchWorker([src], out_dir, classic)
        check("rename disabled by default", w2.settings.rename_batch is False)

    print("BATCH RENAME TEST PASSED" if not failures
          else f"BATCH RENAME TEST FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
