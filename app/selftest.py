"""Headless self-test (``BatchUpscaler.exe --self-test REPORT.json``).

Checks what a packaged build most often gets wrong: the bundled font and
icons load, the main window builds and its default settings pass the Django
form, the Metadata provider libraries are present, PyTorch (and CUDA, if
present) works, the model weights load, and every upscale factor produces the
exact proportional size as a JPEG with batch-rename names. Results go to a
JSON report; exit code 0 means all passed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path


def run(report_path: str) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    report: dict = {"ok": False, "checks": {}, "python": sys.version.split()[0],
                    "frozen": bool(getattr(sys, "frozen", False))}
    checks = report["checks"]
    try:
        import torch

        checks["torch"] = {"version": torch.__version__, "cuda": torch.cuda.is_available(),
                           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}

        from PySide6.QtGui import QFontInfo
        from PySide6.QtWidgets import QApplication

        from app import paths
        from app.ui import icons, theme

        app = QApplication.instance() or QApplication([])
        theme.apply(app, "light")
        family = QFontInfo(app.font()).family()
        checks["font"] = {"family": family, "ok": family == "Inter Tight"}
        names = sorted(p.stem for p in paths.asset("icons").glob("*.svg"))
        rendered = [n for n in names if not icons.icon(n).pixmap(24, 24).isNull()]
        checks["icons"] = {"count": len(names), "rendered": len(rendered),
                           "ok": bool(names) and len(rendered) == len(names)}

        # The real window (both screens) and the Django settings form behind Upscale.
        from PySide6.QtWidgets import QToolButton

        from app.ui.main_window import MainWindow
        from app.upscaler.forms import UpscaleForm

        window = MainWindow()
        screen = window.upscale_page
        form = UpscaleForm(screen.settings(), images=[(1024, 1536)])
        valid = form.is_valid()
        opts = form.job_options() if valid else None
        buttons = [b for b in window.findChildren(QToolButton) if b.property("iconOnly")]
        labelled = [b for b in buttons
                    if not b.icon().isNull() and b.toolTip() and b.accessibleName()]
        checks["window"] = {
            "icon_buttons": len(buttons), "labelled": len(labelled),
            "defaults": {"factor": screen.factor(), "format": screen.output_format(),
                         "rename_batch": screen.rename.isChecked()},
            "form": "valid" if valid else form.first_error(),
            "ok": (valid and opts.factor == 4 and opts.output_format == "jpg"
                   and opts.rename_batch and bool(buttons) and len(labelled) == len(buttons)),
        }
        window.close()

        # Metadata provider libraries are imported lazily — make sure they were bundled.
        try:
            import keyring
            import requests  # noqa: F401
            from google import genai  # noqa: F401

            backend = type(keyring.get_keyring())
            checks["metadata"] = {"keyring": f"{backend.__module__}.{backend.__name__}",
                                  "ok": not backend.__module__.endswith(".fail")}
        except Exception as exc:
            checks["metadata"] = {"error": repr(exc), "ok": False}

        from PIL import Image

        from app.upscaler.engine import MODEL_DIR, weights_ready
        from app.upscaler.workspace import JobOptions, Workspace

        checks["weights"] = {"dir": str(MODEL_DIR), "ready_before": weights_ready()}
        factors: dict = {}
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in"
            src.mkdir()
            sizes = {"portrait.png": (64, 96), "landscape.png": (96, 64)}
            for name, size in sizes.items():
                Image.new("RGB", size, (90, 140, 200)).save(src / name)
            for factor in (1, 2, 3, 4, 5):
                ws = Workspace()
                ws.add([str(src / n) for n in sizes])
                out = Path(tmp) / f"x{factor}"
                started = time.monotonic()
                job = ws.start(JobOptions(factor=factor, output_dir=out))
                job.join(600)
                snap = job.snapshot()
                saved = {}
                for result in snap["results"].values():
                    if result.get("status") == "done":
                        with Image.open(result["output"]) as im:
                            saved[Path(result["output"]).name] = (im.size, im.format)
                expected = sorted((w * factor, h * factor) for w, h in sizes.values())
                ok = (snap["ok"] == 2 and sorted(s for s, _ in saved.values()) == expected
                      and all(fmt == "JPEG" for _, fmt in saved.values())
                      and all(re.fullmatch(r"[A-Z]{4,5}[12]\.jpg", n) for n in saved))
                factors[f"{factor}x"] = {"ok": ok, "outputs": {n: list(s) for n, (s, _) in saved.items()},
                                         "seconds": round(time.monotonic() - started, 1),
                                         "errors": [r.get("error") for r in snap["results"].values()
                                                    if r.get("status") == "failed"]}
        checks["factors"] = factors
        report["ok"] = (checks["font"]["ok"] and checks["icons"]["ok"] and checks["window"]["ok"]
                        and checks["metadata"]["ok"] and all(f["ok"] for f in factors.values()))
    except Exception as exc:  # report, don't crash, so the JSON always appears
        report["error"] = f"{exc}\n{traceback.format_exc()}"
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if sys.stdout is not None:          # the windowed .exe has no console
        print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1
