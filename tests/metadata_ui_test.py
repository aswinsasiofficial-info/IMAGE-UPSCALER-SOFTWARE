"""Tests for the redesigned Metadata screen (offscreen Qt, no network).

Run with:  python -m tests.metadata_ui_test

Covers the Export button (replaces Review & Export): saves the current
metadata including edits, asks for a location only the first time, Export
As… asks again; folder batches of 25 images that retry temporary failures
for the failed images only and save the complete CSV into the folder only
when every image succeeded (never after a permanent failure or a cancel);
the 25-image folder limit and its chooser; lazy thumbnails and the large
preview; status filters/counts; and the responsive layouts.
"""

from __future__ import annotations

import csv
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QTabWidget

import app.engine.metadata_worker as mw
from app.engine import credential_store
from app.engine.metadata_settings import ImageMetadata, ProviderError
from tests.metadata_test import _release_qt

KEY = Qt.UserRole + 1          # item role holding the page's path key


def _read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.reader(fh))


def _drain(page) -> None:
    loop = QEventLoop()
    page.worker.finished_all.connect(loop.quit)
    QTimer.singleShot(20000, loop.quit)
    loop.exec()
    assert page.worker.wait(5000), "worker did not finish"
    QApplication.processEvents()


def _pump(seconds: float, until=lambda: False) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if until():
            return True
        time.sleep(0.01)
    return until()


class SaveDialog:
    """Stands in for QFileDialog.getSaveFileName and records each call."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, *_a, **_k):
        self.calls += 1
        return (self.answers.pop(0) if self.answers else "", "")


def test_export_button(tmp: Path) -> None:
    from app.ui.metadata_page import DONE, MetadataPage

    folder = tmp / "export"
    folder.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        Image.new("RGB", (40, 30), (9, 9, 9)).save(folder / name)
    page = MetadataPage()
    page._add_paths([folder / "a.png", folder / "b.png", folder / "c.png"])
    assert not hasattr(page, "review_btn") and page.export_btn.text() == "Export"
    assert not page.export_btn.isEnabled()                    # nothing generated yet

    keys = list(page.items)
    page._set_status(keys[0], DONE, meta={"title": "Alpha", "description": "d1",
                                          "keywords": ["x", "y"], "category": 7})
    page._set_status(keys[1], DONE, meta={"title": "Beta", "description": "d2", "keywords": ["z"]})
    assert page.export_btn.isEnabled()

    # The editor shows the suggested category; Releases starts blank.
    page.grid.setCurrentItem(page._grid_item(keys[0]))
    assert page.category_combo.currentData() == 7 and page.releases_edit.text() == ""
    assert page.category_combo.count() == 22                  # "Not set" + 21 categories

    # A user edit in the editor is part of the export.
    page.grid.setCurrentItem(page._grid_item(keys[1]))
    assert page.category_combo.currentData() is None
    page.title_edit.setText("Beta, edited by hand")
    page.kw_edit.setText("one, two,\nthree")
    page.category_combo.setCurrentIndex(page.category_combo.findData(11))
    page.releases_edit.setText("model-release.jpg")

    dest = tmp / "chosen" / "my metadata.csv"
    dialog = SaveDialog(str(dest))
    original = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(dialog)
    try:
        page.export_btn.click()
        assert dialog.calls == 1, "first Export must ask where to save"
        rows = _read_csv(dest)
        assert rows[0] == ["Filename", "Title", "Keywords", "Category", "Releases"]
        assert rows[1:] == [["a.png", "Alpha", "x, y", "7", ""],
                            ["b.png", "Beta, edited by hand", "one, two, three", "11",
                             "model-release.jpg"]], rows
        assert "1 image(s) without metadata were left out" in page.banner.text.text()
        assert page.csv_path == dest and dest.name in page.export_location.text()

        # Later Exports save to the same file without asking.
        page.title_edit.setText("Beta v2")
        page.export_btn.click()
        assert dialog.calls == 1
        assert _read_csv(dest)[2][1] == "Beta v2"

        # More than 49 keywords: the first 49 are exported and the banner says so.
        page.kw_edit.setText(", ".join(f"k{i}" for i in range(55)))
        page.export_btn.click()
        assert _read_csv(dest)[2][2].split(", ") == [f"k{i}" for i in range(49)]
        assert "only the first 49 were exported" in page.banner.text.text()

        # Regenerating keeps the releases the user typed (they're never generated).
        page._set_status(keys[1], DONE, meta={"title": "Beta new", "keywords": ["n"],
                                              "category": 13})
        assert page.items[keys[1]].meta.releases == "model-release.jpg"
        assert page.items[keys[1]].meta.category == 13

        # Export As… asks again; cancelling it changes nothing.
        page.act_export_as.trigger()
        assert dialog.calls == 2 and page.csv_path == dest
        other = tmp / "elsewhere"
        dialog.answers.append(str(other))                    # no extension given
        page.act_export_as.trigger()
        assert page.csv_path == other.with_suffix(".csv") and page.csv_path.exists()
    finally:
        QFileDialog.getSaveFileName = original

    # Clearing the list forgets the location.
    page.clear_all()
    assert page.csv_path is None and "asks where" in page.export_location.text()
    page.shutdown()
    print("  Export: saves edits, asks once, Export As asks again OK")


def _folder(tmp: Path, name: str, count: int) -> Path:
    root = tmp / name
    root.mkdir()
    for i in range(count):
        Image.new("RGB", (32, 24), (i * 9 % 255, 60, 90)).save(root / f"img{i:02d}.jpg")
    return root


def test_folder_batch_of_25(tmp: Path) -> None:
    from app.ui.metadata_page import DONE, FAILED, MetadataPage

    orig_gen, orig_sleep = mw.generate_metadata, mw.time.sleep
    mw.time.sleep = lambda s: None                            # no real backoff
    credential_store._session_keys.clear()
    try:
        # 1) Temporary failures on 6 of 25 images heal automatically; only those re-run.
        root = _folder(tmp, "batch25", 25)
        flaky = {f"img{i:02d}.jpg" for i in (2, 5, 11, 17, 20, 24)}
        calls: list[str] = []

        def generate(path, settings, api_key, log=None):
            name = Path(path).name
            calls.append(name)
            if name in flaky and calls.count(name) == 1:
                raise ProviderError("rate limit reached (HTTP 429)", code=429)
            return ImageMetadata(f"Title {name}", "desc", ["k1", "k2"])

        mw.generate_metadata = generate
        page = MetadataPage()
        page._add_paths([root])
        assert len(page.items) == 25
        credential_store.set_session_key(page.provider_combo.currentText(), "test-key")
        page.generate_all()
        _drain(page)
        assert all(it.status == DONE for it in page.items.values())
        for name in {f"img{i:02d}.jpg" for i in range(25)}:
            assert calls.count(name) == (2 if name in flaky else 1), (name, calls.count(name))
        csv_path = root / "metadata.csv"
        rows = _read_csv(csv_path)
        assert len(rows) == 26 and sorted(r[0] for r in rows[1:]) == sorted(f"img{i:02d}.jpg" for i in range(25))
        assert page.csv_path == csv_path, "Export should now update the folder's CSV"
        assert page.banner.property("banner") == "success" and "Complete CSV saved" in page.banner.text.text()

        # Edits after the automatic save go into the same CSV with Export (no prompt).
        page.grid.setCurrentRow(3)
        page.title_edit.setText("Edited after the auto-save")
        dialog = SaveDialog()
        original = QFileDialog.getSaveFileName
        QFileDialog.getSaveFileName = staticmethod(dialog)
        try:
            page.export_btn.click()
        finally:
            QFileDialog.getSaveFileName = original
        assert dialog.calls == 0
        assert "Edited after the auto-save" in [r[1] for r in _read_csv(csv_path)]
        assert sorted(p.name for p in root.glob("*.csv")) == ["metadata.csv"]
        page.shutdown()

        # 2) A permanent failure: no CSV until every image succeeds; Retry Failed
        #    runs only that image, then the complete CSV appears.
        root2 = _folder(tmp, "batch25_fail", 25)
        calls.clear()

        def generate_bad(path, settings, api_key, log=None):
            name = Path(path).name
            calls.append(name)
            if name == "img07.jpg":
                raise ProviderError("unsupported image (HTTP 400)", code=400)
            return ImageMetadata(f"Title {name}", "desc", ["k"])

        mw.generate_metadata = generate_bad
        page = MetadataPage()
        page._add_paths([root2])
        page.generate_all()
        _drain(page)
        failed = [it.path.name for it in page.items.values() if it.status == FAILED]
        assert failed == ["img07.jpg"] and calls.count("img07.jpg") == 1     # permanent: not retried
        assert not list(root2.glob("*.csv")), "a partial folder CSV must never be written"
        assert page.banner.property("banner") == "danger"
        assert page.filter_buttons["failed"].text() == "Failed 1"

        calls.clear()
        mw.generate_metadata = generate
        page.retry_failed()
        _drain(page)
        assert calls == ["img07.jpg"], calls
        assert len(_read_csv(root2 / "metadata.csv")) == 26
        page.shutdown()

        # 3) Cancelled batch: no CSV.
        root3 = _folder(tmp, "batch25_cancel", 25)

        def slow(path, settings, api_key, log=None):
            orig_sleep(0.03)                     # the real sleep (backoff is stubbed out)
            return ImageMetadata("t", "d", ["k"])

        mw.generate_metadata = slow
        page = MetadataPage()
        page._add_paths([root3])
        page.generate_all()
        _pump(0.2)
        page.cancel()
        _drain(page)
        assert any(it.status != DONE for it in page.items.values())
        assert not list(root3.glob("*.csv")), "a cancelled batch must not export"
        assert page.banner.property("banner") == "warning" and "cancelled" in page.banner.text.text().lower()
        page.shutdown()
    finally:
        mw.generate_metadata, mw.time.sleep = orig_gen, orig_sleep
        credential_store._session_keys.clear()
    print("  25-image folder batches: retry only failed, complete-only CSV, cancel = no CSV OK")


def test_thumbnails_and_preview(tmp: Path) -> None:
    from app.ui.metadata_page import PREVIEW_EDGE, THUMB_EDGE, MetadataPage

    many = tmp / "many"
    many.mkdir()
    for i in range(300):
        Image.new("RGB", (60, 40), (i % 255, 100, 150)).save(many / f"p{i:03d}.png")
    # A phone photo stored landscape with EXIF "rotate 90": shown portrait.
    rotated = Image.new("RGB", (90, 60), (200, 50, 50))
    exif = Image.Exif()
    exif[0x0112] = 6
    rotated.save(many / "000_phone.jpg", exif=exif.tobytes())

    page = MetadataPage()
    started = time.monotonic()
    page._add_paths(sorted(many.iterdir()))          # picked files: no folder-batch limit
    elapsed = time.monotonic() - started
    assert len(page.items) == 301 and elapsed < 3.0, elapsed
    assert not page.loader._cache, "adding images must not decode thumbnails up front"

    page.resize(1400, 900)
    page.show()
    assert _pump(8, lambda: page.loader.cached(str(many / "000_phone.jpg"), THUMB_EDGE) is not None)
    _pump(1.0)
    loaded = sum(1 for k in page.loader._cache if k.startswith(f"{THUMB_EDGE}|"))
    assert 0 < loaded < 40, f"only visible tiles should load thumbnails (loaded {loaded})"
    thumb = page.loader.cached(str(many / "000_phone.jpg"), THUMB_EDGE)
    assert thumb.height() > thumb.width(), "EXIF rotation must be applied to thumbnails"

    # Selecting an image shows its large preview.
    page.grid.setCurrentRow(5)
    target = str(page.items[page.grid.currentItem().data(KEY)].path)
    assert _pump(5, lambda: page.loader.cached(target, PREVIEW_EDGE) is not None)
    _pump(0.2)
    assert page.preview._pix is not None and not page.preview._pix.isNull()
    # The preview-sized decode is shown (not the list tile).
    assert page.preview._pix.cacheKey() == page.loader.cached(target, PREVIEW_EDGE).cacheKey()
    assert "p004.png" in page.editor_file.text()

    # Scrolling to the end loads those thumbnails too (on demand).
    page.grid.scrollToBottom()
    last = str(page.items[page.grid.item(page.grid.count() - 1).data(KEY)].path)
    assert _pump(5, lambda: page.loader.cached(last, THUMB_EDGE) is not None)
    page.hide()
    page.shutdown()
    print("  thumbnails: lazy, EXIF-aware, 301 images added fast; large preview OK")


def test_folder_batch_limit(tmp: Path) -> None:
    from app.ui.folder_picker import FolderPickDialog
    from app.ui.metadata_page import FOLDER_BATCH_LIMIT, MetadataPage

    assert FOLDER_BATCH_LIMIT == 25
    big = _folder(tmp, "big30", 30)
    page = MetadataPage()
    asked: list[tuple] = []

    def choose(folder, images, limit, already):
        asked.append((folder, len(images), limit, already))
        return images[5:5 + limit]                       # the user picks 25 of them

    page._choose_folder_images = choose
    page._add_paths([big])
    assert asked == [(big, 30, 25, 0)], asked
    names = sorted(it.path.name for it in page.items.values())
    assert len(names) == 25 and names[0] == "img05.jpg" and names[-1] == "img29.jpg"
    assert page._folder_roots == [big]
    assert any("30 supported images" in line for line in page.log.toPlainText().splitlines())

    # The folder batch is full now: adding it again explains and adds nothing.
    shown: list[tuple] = []
    original = QMessageBox.information
    QMessageBox.information = staticmethod(lambda _p, t, x, *a: shown.append((t, x)))
    try:
        page._add_paths([big])
    finally:
        QMessageBox.information = original
    assert len(page.items) == 25 and shown and shown[0][0] == "Folder batch is full", shown

    # Room left after removing some: the chooser offers only that many.
    page.grid.setCurrentRow(0)
    for i in range(3):
        page.grid.item(i).setSelected(True)
    page.remove_selected()
    asked.clear()
    page._add_paths([big])
    assert asked and asked[0][1:] == (8, 3, 22), asked      # 8 not yet added, room for 3
    assert len(page.items) == 25

    # A cancelled chooser adds nothing.
    page2 = MetadataPage()
    page2._choose_folder_images = lambda *a: None
    page2._add_paths([big])
    assert not page2.items and not page2._folder_roots

    # Exactly 25: no chooser at all.
    page3 = MetadataPage()
    page3._choose_folder_images = lambda *a: (_ for _ in ()).throw(AssertionError("asked"))
    page3._add_paths([_folder(tmp, "exact25", 25)])
    assert len(page3.items) == 25

    # The chooser itself: first 25 ticked, can't accept more than the limit.
    images = sorted(big.glob("*.jpg"))
    dialog = FolderPickDialog(big, images, 25, page.loader)
    assert dialog.grid.count() == 30 and len(dialog.selected_paths()) == 25
    assert dialog.counter.text() == "25 of 25 selected" and dialog.ok.isEnabled()
    assert dialog.ok.text() == "Add 25 Images"
    dialog.toggle(dialog.grid.item(27))                      # tick a 26th
    assert not dialog.ok.isEnabled() and "untick 1" in dialog.counter.text()
    dialog.toggle(dialog.grid.item(0))                       # untick one → back to 25
    assert dialog.ok.isEnabled()
    picked = dialog.selected_paths()
    assert len(picked) == 25 and images[0] not in picked and images[27] in picked
    dialog.select_first()
    assert dialog.selected_paths() == images[:25]
    dialog._set_all(0)
    assert not dialog.ok.isEnabled() and dialog.counter.text() == "0 of 25 selected"
    assert dialog.grid.accessibleName() and dialog.windowTitle() == "Choose up to 25 images"
    dialog.deleteLater()
    for p in (page, page2, page3):
        p.shutdown()
    print("  25-image folder limit: chooser, full batch, remaining room, cancel OK")


def test_filters_counts_and_layout(tmp: Path) -> None:
    from app.ui.metadata_page import DONE, FAILED, GENERATING, MetadataPage

    root = _folder(tmp, "filters", 6)
    page = MetadataPage()
    page._add_paths([root])
    keys = list(page.items)
    page._set_status(keys[0], DONE, meta={"title": "T", "description": "", "keywords": ["a"]})
    page._set_status(keys[1], DONE, meta={"title": "U", "description": "", "keywords": ["b"]})
    page._set_status(keys[2], FAILED, error="bad key (HTTP 401)")
    page._set_status(keys[3], GENERATING)
    assert page.count_label.text() == "6 images · 2 done · 1 failed · 3 pending", page.count_label.text()
    assert [page.filter_buttons[n].text() for n in ("all", "pending", "done", "failed")] == \
        ["All 6", "Pending 3", "Done 2", "Failed 1"]
    page.filter_buttons["failed"].click()
    visible = [page.grid.item(i).text() for i in range(page.grid.count()) if not page.grid.isRowHidden(i)]
    assert visible == ["img02.jpg"], visible
    page.filter_buttons["all"].click()
    assert not any(page.grid.isRowHidden(i) for i in range(page.grid.count()))

    # The failed image explains itself in the editor.
    page.grid.setCurrentRow(2)
    assert "bad key (HTTP 401)" in page.editor_status.text()
    assert page.retry_failed_btn.isEnabled()

    # Responsive layout: three columns, then tabs, then stacked.
    page.show()
    page.resize(1400, 900)
    _pump(0.2)
    assert page.splitter.count() == 3 and page.splitter.widget(1) is page.editor_panel
    page.resize(1000, 800)
    _pump(0.2)
    assert isinstance(page.splitter.widget(1), QTabWidget) and page.detail_tabs.count() == 2
    assert [page.detail_tabs.tabText(i) for i in range(2)] == ["Metadata", "Provider && options"]
    # The window can't get this narrow (min 720 px), so drive the breakpoint directly.
    page._apply_layout(680)
    assert page.splitter.orientation() == Qt.Vertical and page.splitter.count() == 2
    page.resize(1400, 900)
    _pump(0.2)
    assert page.splitter.count() == 3 and page.editor_panel.isVisible() and page.settings_panel.isVisible()
    page.hide()
    page.shutdown()
    print("  counts, filters, failed details and responsive layouts OK")


def main() -> int:
    app = QApplication.instance() or QApplication([])
    from app.ui import theme
    theme.apply(app)
    info = QMessageBox.information
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            test_export_button(t)
            test_folder_batch_of_25(t)
            test_thumbnails_and_preview(t)
            test_folder_batch_limit(t)
            test_filters_counts_and_layout(t)
    except Exception:
        import traceback
        traceback.print_exc()
        print("METADATA UI TEST FAILED")
        _release_qt(app)
        return 1
    finally:
        QMessageBox.information = info
    _release_qt(app)
    print("METADATA UI TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
