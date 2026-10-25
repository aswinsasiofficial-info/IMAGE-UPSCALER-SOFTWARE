"""The desktop Upscaler screen inside the real main window (offscreen, no model).

Run with:  python -m tests.upscaler_ui_test

Checks the defaults (4×, JPEG, batch rename) and that the filename-suffix and
exact-size options are gone; Inter Tight everywhere; every icon-only button on
both screens has an icon, a tooltip and an accessible name; thumbnails and the
large preview; the expected output size for each factor 1×–5× before
processing, with the explanation of how the factor is produced; a batch run
from the Upscale button; and the too-large validation message.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtGui import QFontInfo
from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QMessageBox, QToolButton

from tests.upscaler_test import StubEngine, make_marked


def pump(seconds: float, until=lambda: False) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if until():
            return True
        time.sleep(0.01)
    return until()


def check_icon_buttons(root, where: str) -> int:
    buttons = [b for b in root.findChildren(QToolButton) if b.property("iconOnly")]
    for b in buttons:
        assert not b.icon().isNull(), f"{where}: icon button without icon"
        assert b.toolTip().strip(), f"{where}: {b.accessibleName()!r} has no tooltip"
        assert b.accessibleName().strip(), f"{where}: tooltip {b.toolTip()!r} has no accessible name"
        assert not b.text(), f"{where}: icon-only button shows text {b.text()!r}"
    return len(buttons)


def main() -> int:
    app = QApplication.instance() or QApplication([])
    from app.ui import theme
    from app.upscaler.workspace import reset_workspace
    theme.apply(app)
    reset_workspace(engine_factory=StubEngine)
    from app.ui.main_window import MainWindow

    calls = {"info": [], "warning": []}
    QMessageBox.information = staticmethod(lambda _p, t, x, *a: calls["info"].append((t, x)))
    QMessageBox.warning = staticmethod(lambda _p, t, x, *a: calls["warning"].append((t, x)))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "in"
        src.mkdir()
        make_marked(src / "portrait.png", 1024, 1536)
        make_marked(src / "landscape.png", 1536, 1024)

        win = MainWindow()
        win.resize(1300, 860)
        win.show()
        screen = win.upscale_page
        pump(0.3)

        # Defaults and removed options.
        assert screen.factor() == 4 and screen.output_format() == "jpg" and screen.rename.isChecked()
        assert [b.text() for b in screen.factor_buttons.values()] == ["1×", "2×", "3×", "4×", "5×"]
        texts = " ".join(w.text() for w in screen.findChildren(QLabel)) + " ".join(
            w.text() for w in screen.findChildren(QAbstractButton))
        for gone in ("Filename suffix", "Exact size", "Pad to fit", "Crop to fill"):
            assert gone not in texts, gone
        print("  defaults 4× / JPEG / batch rename; suffix and exact size removed")

        # Inter Tight everywhere.
        for widget in (win, screen, screen.start_btn, screen.preview_title, win.metadata_page.title_edit):
            assert QFontInfo(widget.font()).family() == "Inter Tight", widget
        print("  Inter Tight font in use")

        # Icon-only buttons on both screens: icon + tooltip + accessible name.
        n_up = check_icon_buttons(screen, "Upscaler")
        n_meta = check_icon_buttons(win.metadata_page, "Metadata")
        assert n_up >= 6 and n_meta >= 9, (n_up, n_meta)
        assert screen.start_btn.text() == "Upscale" and "Ctrl+Return" in screen.start_btn.toolTip()
        assert screen.btn_add_images.accessibleName() == "Add images"
        print(f"  {n_up} + {n_meta} icon buttons, all with tooltips and accessible names")

        # Thumbnails, preview and expected sizes for every factor.
        screen.add_paths([str(src / "portrait.png"), str(src / "landscape.png")])
        assert screen.list.count() == 2 and screen.list.currentRow() == 0
        assert pump(8, lambda: all(screen.loader.cached(str(src / n), 128) is not None
                                   for n in ("portrait.png", "landscape.png"))), "thumbnails"
        expected = {1: "1024 × 1536 → 1024 × 1536", 2: "1024 × 1536 → 2048 × 3072",
                    3: "1024 × 1536 → 3072 × 4608", 4: "1024 × 1536 → 4096 × 6144",
                    5: "1024 × 1536 → 5120 × 7680"}
        for f, text in expected.items():
            screen.factor_buttons[f].click()
            assert screen.factor() == f
            assert text in screen.preview_dims.text(), (f, screen.preview_dims.text())
            assert "still 2:3" in screen.preview_dims.text()
        screen.factor_buttons[4].click()
        assert screen.factor_how.text().startswith("Native")
        screen.factor_buttons[5].click()
        assert "interpolation" in screen.factor_how.text()
        assert "more real detail" in screen.detail_note.text()
        assert "4096 × 6144" not in screen.size_summary.text() and "5120 × 7680" in screen.size_summary.text()
        assert pump(5, lambda: screen.preview._pix is not None)
        print("  thumbnails + preview; 1×–5× sizes shown before processing; factor explained")

        # Run a batch from the button (2× keeps it quick).
        out = tmp / "out"
        screen.output_dir.setText(str(out))
        screen.factor_buttons[2].click()
        screen.start_btn.click()
        assert screen._running and not screen.start_btn.isEnabled() and screen.cancel_btn.isEnabled()
        assert pump(60, lambda: not screen._running), "batch did not finish"
        assert calls["info"] and calls["info"][-1][0] == "Done", calls
        files = sorted(out.iterdir())
        assert len(files) == 2 and all(re.fullmatch(r"[A-Z]{4,5}[12]\.jpg", f.name) for f in files), files
        sizes = sorted(Image.open(f).size for f in files)
        assert sizes == [(2048, 3072), (3072, 2048)], sizes
        assert all(r["status"] == "done" for r in screen.results.values())
        print(f"  batch from the Upscale button: {[f.name for f in files]} at 2×, JPEG, renamed")

        # Too large at 5× → explained, nothing starts.
        Image.new("RGB", (14000, 60), (5, 5, 5)).save(src / "strip.png")
        screen.add_paths([str(src / "strip.png")])
        screen.factor_buttons[5].click()
        calls["warning"].clear()
        screen.start_btn.click()
        assert calls["warning"] and calls["warning"][-1][1].startswith("Upscale factor:"), calls
        assert not screen._running and screen.factor_error.isVisible()
        print("  too-large output at 5× is explained and blocked")

        win.close()
        pump(0.3)
    reset_workspace()
    print("UPSCALER UI TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
