"""Metadata Generator screen — fully independent from the upscaler.

The user upscales and saves images first, then opens this screen and picks
the saved images or folder. Every image gets a preview tile, a per-image
status and editable title/description/keywords before the final review-and-
export step. Providers: Gemini or Groq, with the API key handled through
app.engine.credential_store (session → env var → Windows Credential Manager).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QIcon, QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.engine import credential_store
from app.engine.metadata_settings import (
    MetadataSettings, ImageMetadata, SUPPORTED_PROVIDERS, DEFAULT_MODELS,
    GEMINI_MODELS, GROQ_MODELS, LANGUAGES, ADOBE_KEYWORDS_MAX, ADOBE_TITLE_MAX,
)
from app.engine.metadata_export import write_csv, embed_metadata, EMBEDDABLE_SUFFIXES
from app.engine.metadata_worker import MetadataWorker
from app.engine.worker import IMAGE_EXTENSIONS, collect_images

from app.ui.metadata_review import ReviewDialog

_THUMB = 120  # grid tile edge in px

# Per-item generation states.
PENDING, GENERATING, DONE, FAILED = "pending", "generating", "done", "failed"
_STATUS_BG = {
    PENDING: QColor("#f2f2f2"),
    GENERATING: QColor("#fff3b0"),
    DONE: QColor("#c8f0c8"),
    FAILED: QColor("#f6c0c0"),
}


@dataclass
class MetaItem:
    """One queued image plus its generated/edited metadata and status."""
    path: Path
    status: str = PENDING
    error: str = ""
    meta: ImageMetadata = field(default_factory=ImageMetadata)


class MetadataPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.items: dict[str, MetaItem] = {}   # key: resolved path, lowercased
        self.worker: MetadataWorker | None = None
        self._loading_editor = False
        # Folder selections are remembered so a finished folder batch can
        # auto-export its CSV back into the folder the user picked; batches
        # started from individual files keep the manual export workflow.
        self._folder_roots: list[Path] = []
        self._run_keys: list[str] = []         # items in the current/last run
        self._run_roots: list[Path] = []       # folder roots that run touches
        self._last_csv_dir: Path | None = None  # chosen via Review & Export
        self._cancel_requested = False
        self.setAcceptDrops(True)

        self._build_body()
        self._on_provider_changed(self.provider_combo.currentText())

    # -- construction ---------------------------------------------------------
    def _build_body(self) -> None:
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(self._build_grid_panel(), stretch=3)
        top.addWidget(self._build_side_panel(), stretch=2)
        root.addLayout(top)

        root.addWidget(self._build_editor_panel())
        root.addWidget(self._build_control_bar())

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("Metadata log will appear here.")
        self.log.setMaximumHeight(96)
        root.addWidget(self.log)

    def _build_grid_panel(self) -> QWidget:
        box = QGroupBox("Selected images  (drag & drop images or folders here)")
        lay = QVBoxLayout(box)
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setIconSize(QSize(_THUMB, _THUMB))
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setMovement(QListWidget.Static)
        self.grid.setSelectionMode(QListWidget.ExtendedSelection)
        self.grid.setSpacing(8)
        self.grid.currentItemChanged.connect(self._on_item_selected)
        lay.addWidget(self.grid)

        row = QHBoxLayout()
        for text, fn in (
            ("Add Images…", self.add_images),
            ("Add Folder…", self.add_folder),
            ("Remove Selected", self.remove_selected),
            ("Clear", self.clear_all),
        ):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        return box

    def _build_side_panel(self) -> QWidget:
        box = QGroupBox("Provider & options")
        form = QVBoxLayout(box)

        # Provider
        prov_row = QHBoxLayout()
        prov_row.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(SUPPORTED_PROVIDERS))
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        prov_row.addWidget(self.provider_combo, 1)
        form.addLayout(prov_row)

        # Model (editable so retired/new models need no code change)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        model_row.addWidget(self.model_combo, 1)
        form.addLayout(model_row)

        # API key
        form.addWidget(QLabel("API key:"))
        key_row = QHBoxLayout()
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setClearButtonEnabled(True)
        self.btn_show_key = QPushButton("Show")
        self.btn_show_key.setCheckable(True)
        self.btn_show_key.toggled.connect(
            lambda v: self.key_edit.setEchoMode(
                QLineEdit.Normal if v else QLineEdit.Password))
        self.btn_show_key.toggled.connect(
            lambda v: self.btn_show_key.setText("Hide" if v else "Show"))
        key_row.addWidget(self.key_edit, 1)
        key_row.addWidget(self.btn_show_key)
        form.addLayout(key_row)

        remember_row = QHBoxLayout()
        self.chk_remember = QCheckBox("Remember key in Windows Credential Manager")
        self.btn_forget = QPushButton("Forget saved key")
        self.btn_forget.clicked.connect(self.forget_key)
        remember_row.addWidget(self.chk_remember, 1)
        remember_row.addWidget(self.btn_forget)
        form.addLayout(remember_row)
        self.key_hint = QLabel("")
        self.key_hint.setWordWrap(True)
        form.addWidget(self.key_hint)

        # Language + keyword count
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(LANGUAGES)
        self.language_combo.setEditable(True)
        lang_row.addWidget(self.language_combo, 1)
        lang_row.addWidget(QLabel("Keywords:"))
        self.keywords_spin = QSpinBox()
        self.keywords_spin.setRange(1, ADOBE_KEYWORDS_MAX)
        self.keywords_spin.setValue(45)
        self.keywords_spin.setToolTip(
            f"Maximum keywords kept per image (Adobe Stock allows {ADOBE_KEYWORDS_MAX + 1}).")
        lang_row.addWidget(self.keywords_spin)
        form.addLayout(lang_row)

        self.chk_description = QCheckBox("Generate a description too")
        self.chk_description.setChecked(True)
        form.addWidget(self.chk_description)

        form.addStretch(1)
        return box

    def _build_editor_panel(self) -> QWidget:
        box = QGroupBox("Metadata of the selected image  (edit freely — changes are kept)")
        lay = QVBoxLayout(box)
        self.editor_file = QLabel("No image selected.")
        self.editor_file.setTextFormat(Qt.RichText)
        lay.addWidget(self.editor_file)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Title:"))
        self.title_edit = QLineEdit()
        self.title_edit.setMaxLength(ADOBE_TITLE_MAX)
        title_row.addWidget(self.title_edit, 1)
        self.title_count = QLabel("")
        title_row.addWidget(self.title_count)
        lay.addLayout(title_row)

        lay.addWidget(QLabel("Description:"))
        self.desc_edit = QPlainTextEdit()
        self.desc_edit.setMaximumHeight(56)
        lay.addWidget(self.desc_edit)

        lay.addWidget(QLabel("Keywords (comma separated):"))
        self.kw_edit = QLineEdit()
        lay.addWidget(self.kw_edit)

        self.editor_status = QLabel("")
        self.editor_status.setWordWrap(True)
        lay.addWidget(self.editor_status)

        self.title_edit.textChanged.connect(self._commit_editor)
        self.desc_edit.textChanged.connect(self._commit_editor)
        self.kw_edit.textChanged.connect(self._commit_editor)
        return box

    def _build_control_bar(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        self.generate_btn = QPushButton("Generate Metadata")
        self.generate_btn.setDefault(True)
        self.generate_btn.clicked.connect(self.generate_all)
        self.retry_failed_btn = QPushButton("Retry Failed")
        self.retry_failed_btn.clicked.connect(self.retry_failed)
        self.regen_selected_btn = QPushButton("Regenerate Selected")
        self.regen_selected_btn.clicked.connect(self.regenerate_selected)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        self.review_btn = QPushButton("Review & Export…")
        self.review_btn.clicked.connect(self.review_and_export)
        self.status_label = QLabel("Idle")
        self.progress = QProgressBar()
        self.progress.setValue(0)
        lay.addWidget(self.generate_btn)
        lay.addWidget(self.retry_failed_btn)
        lay.addWidget(self.regen_selected_btn)
        lay.addWidget(self.cancel_btn)
        lay.addWidget(self.review_btn)
        lay.addWidget(self.status_label, 1)
        lay.addWidget(self.progress, 2)
        return bar

    # -- provider / key handling -------------------------------------------------
    def _on_provider_changed(self, provider: str) -> None:
        models = GEMINI_MODELS if provider == "Gemini" else GROQ_MODELS
        self.model_combo.clear()
        self.model_combo.addItems(models)
        self.model_combo.setCurrentText(DEFAULT_MODELS[provider])
        env_var = "GEMINI_API_KEY" if provider == "Gemini" else "GROQ_API_KEY"
        self.key_edit.setPlaceholderText(f"or set {env_var}, or a saved key")
        self._update_key_hint()

    def _update_key_hint(self) -> None:
        provider = self.provider_combo.currentText()
        if self.key_edit.text().strip():
            self.key_hint.setText("✓ Key typed in — used for this run only unless "
                                  '"Remember" is ticked at generation time.')
        elif credential_store.get_key(provider):
            self.key_hint.setText(
                f"✓ A key for {provider} was found (typed earlier, "
                f"{provider.upper()}_API_KEY env var, or Windows Credential Manager).")
        else:
            self.key_hint.setText(
                f"✗ No {provider} key configured. Enter one above, set the "
                f"{provider.upper()}_API_KEY environment variable, or save it "
                "via Windows Credential Manager.")

    def _resolve_api_key(self) -> str:
        """Pick the key for the current provider and remember it if asked."""
        provider = self.provider_combo.currentText()
        typed = self.key_edit.text().strip()
        if typed:
            credential_store.set_session_key(provider, typed)
            if self.chk_remember.isChecked():
                ok, msg = credential_store.save_key(provider, typed)
                self.log.appendPlainText(("✓ " if ok else "✗ ") + msg)
            return typed
        return credential_store.get_key(provider)

    def forget_key(self) -> None:
        provider = self.provider_combo.currentText()
        ok, msg = credential_store.delete_key(provider)
        self.key_edit.clear()
        self.log.appendPlainText(("✓ " if ok else "✗ ") + msg)
        self._update_key_hint()

    # -- queue management -------------------------------------------------------
    def _key(self, path: Path) -> str:
        try:
            return str(Path(path).resolve()).lower()
        except OSError:
            return str(path).lower()

    @staticmethod
    def _norm(path: Path) -> str:
        """Resolved, case-folded path string — Windows-safe comparison key."""
        try:
            return str(Path(path).resolve()).lower().rstrip("\\/")
        except OSError:
            return str(path).lower().rstrip("\\/")

    def _is_under(self, path: Path, root: Path) -> bool:
        p, r = self._norm(path), self._norm(root)
        return p == r or p.startswith(r + os.sep)

    def _roots_for(self, files: list[Path]) -> list[Path]:
        """Folder roots touched by a batch — each file maps to its deepest
        matching root, so nested folder selections never export twice."""
        roots: list[Path] = []
        for f in files:
            best: Path | None = None
            for r in self._folder_roots:
                if self._is_under(f, r) and (best is None
                        or len(self._norm(r)) > len(self._norm(best))):
                    best = r
            if best is not None and best not in roots:
                roots.append(best)
        return roots

    @staticmethod
    def _unique_csv_path(folder: Path) -> Path:
        """metadata.csv in ``folder``, or metadata_2.csv, … — never overwrites."""
        candidate = folder / "metadata.csv"
        n = 2
        while candidate.exists():
            candidate = folder / f"metadata_{n}.csv"
            n += 1
        return candidate

    def _add_paths(self, paths: list[Path]) -> None:
        # Remember folder selections (Add Folder… or dropped folders) for the
        # auto CSV export at the end of a batch; plain files register nothing.
        for p in paths:
            try:
                if p.is_dir() and self._norm(p) not in (self._norm(r)
                                                        for r in self._folder_roots):
                    self._folder_roots.append(Path(p))
            except OSError:
                continue
        found = collect_images(paths)
        added = 0
        for p in found:
            key = self._key(p)
            if key in self.items:
                continue
            self.items[key] = MetaItem(path=Path(p))
            self._add_grid_item(Path(p))
            added += 1
        self.status_label.setText(f"{len(self.items)} image(s) in list")
        if added == 0:
            self.log.appendPlainText("No new supported images found in that selection.")
        else:
            self.log.appendPlainText(f"Added {added} image(s) for metadata generation.")

    def _add_grid_item(self, path: Path) -> QListWidgetItem:
        item = QListWidgetItem(self._thumbnail(path), path.name)
        item.setToolTip(str(path))
        item.setData(Qt.UserRole, str(path))
        item.setData(Qt.UserRole + 1, self._key(path))
        item.setBackground(_STATUS_BG[PENDING])
        self.grid.addItem(item)
        return item

    @staticmethod
    def _thumbnail(path: Path) -> QIcon:
        reader = QImageReader(str(path))
        size = reader.size()
        if size.isValid() and max(size.width(), size.height()) > _THUMB:
            ratio = _THUMB / max(size.width(), size.height())
            reader.setScaledSize(QSize(round(size.width() * ratio),
                                       round(size.height() * ratio)))
        image = reader.read()
        if image.isNull():
            return QIcon()
        return QIcon(QPixmap.fromImage(image))

    def add_images(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select images", str(Path.home()), f"Images ({exts});;All files (*)")
        if files:
            self._add_paths([Path(f) for f in files])

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select a folder", str(Path.home()))
        if folder:
            self._add_paths([Path(folder)])

    def remove_selected(self) -> None:
        for item in self.grid.selectedItems():
            self.items.pop(item.data(Qt.UserRole + 1), None)
            self.grid.takeItem(self.grid.row(item))
        self.status_label.setText(f"{len(self.items)} image(s) in list")

    def clear_all(self) -> None:
        if self.worker and self.worker.isRunning():
            self.cancel()
        self.grid.clear()
        self.items.clear()
        self._folder_roots.clear()
        self._show_item(None)
        self.status_label.setText("0 image(s) in list")

    # -- editor ------------------------------------------------------------------
    def _on_item_selected(self, current: QListWidgetItem | None,
                          previous: QListWidgetItem | None) -> None:
        self._commit_editor()
        self._show_item(current)

    def _grid_item(self, key: str) -> QListWidgetItem | None:
        for i in range(self.grid.count()):
            it = self.grid.item(i)
            if it.data(Qt.UserRole + 1) == key:
                return it
        return None

    def _show_item(self, grid_item: QListWidgetItem | None) -> None:
        self._loading_editor = True
        if grid_item is None:
            self.editor_file.setText("No image selected.")
            self.title_edit.setText("")
            self.desc_edit.setPlainText("")
            self.kw_edit.setText("")
            self.editor_status.setText("")
        else:
            meta_item = self.items.get(grid_item.data(Qt.UserRole + 1))
            if meta_item is not None:
                self.editor_file.setText(
                    f"<b>{meta_item.path.name}</b> — {meta_item.path.parent}")
                self.title_edit.setText(meta_item.meta.title)
                self.desc_edit.setPlainText(meta_item.meta.description)
                self.kw_edit.setText(", ".join(meta_item.meta.keywords))
                self._update_status_label(meta_item)
        self._loading_editor = False
        self._update_title_count()

    def _commit_editor(self) -> None:
        """Write the visible editor fields back into the selected item."""
        if self._loading_editor:
            return
        grid_item = self.grid.currentItem()
        if grid_item is None:
            return
        meta_item = self.items.get(grid_item.data(Qt.UserRole + 1))
        if meta_item is None or meta_item.status == GENERATING:
            return
        meta_item.meta.title = self.title_edit.text().strip()
        meta_item.meta.description = self.desc_edit.toPlainText().strip()
        meta_item.meta.keywords = [
            k.strip() for k in self.kw_edit.text().split(",") if k.strip()]
        self._update_title_count()

    def _update_title_count(self) -> None:
        self.title_count.setText(f"{len(self.title_edit.text())}/{ADOBE_TITLE_MAX}")

    def _update_status_label(self, meta_item: MetaItem) -> None:
        if meta_item.status == DONE:
            self.editor_status.setText(
                f"Status: generated ✓ ({len(meta_item.meta.keywords)} keywords)")
        elif meta_item.status == FAILED:
            self.editor_status.setText(
                f"Status: FAILED — {meta_item.error} → use “Retry Failed” or "
                "“Regenerate Selected”.")
        elif meta_item.status == GENERATING:
            self.editor_status.setText("Status: generating…")
        else:
            self.editor_status.setText("Status: not generated yet")

    # -- status painting on grid -------------------------------------------------
    def _set_status(self, key: str, status: str, error: str = "",
                    meta: dict | None = None) -> None:
        meta_item = self.items.get(key)
        if meta_item is None:
            return
        meta_item.status = status
        meta_item.error = error
        if meta is not None:
            meta_item.meta = ImageMetadata.from_dict(meta)
        grid_item = self._grid_item(key)
        if grid_item is not None:
            grid_item.setBackground(_STATUS_BG[status])
            tip = str(meta_item.path)
            if status == FAILED:
                tip += f"\n✗ {error}"
            elif status == DONE:
                tip += f"\n{meta_item.meta.title}\n{', '.join(meta_item.meta.keywords)}"
            grid_item.setToolTip(tip)
        if self.grid.currentItem() is grid_item:
            self._show_item(grid_item)

    # -- generation --------------------------------------------------------------
    def _build_settings(self) -> MetadataSettings:
        return MetadataSettings(
            provider=self.provider_combo.currentText(),
            model=self.model_combo.currentText().strip() or DEFAULT_MODELS[
                self.provider_combo.currentText()],
            language=self.language_combo.currentText().strip() or "English",
            keyword_count=self.keywords_spin.value(),
            generate_description=self.chk_description.isChecked(),
        )

    def _start_generation(self, keys: list[str], what: str) -> None:
        if self.worker and self.worker.isRunning():
            # Double-start protection: the buttons are disabled while a batch
            # runs, but keyboard/programmatic clicks must never queue a second
            # worker on the same images.
            self.log.appendPlainText("A metadata batch is already running — "
                                     "wait for it or press Cancel.")
            return
        if not keys:
            QMessageBox.information(self, "Nothing to do", f"No {what} images in the list.")
            return
        api_key = self._resolve_api_key()
        if not api_key:
            QMessageBox.warning(
                self, "Missing API key",
                f"No {self.provider_combo.currentText()} API key is configured.\n\n"
                "Enter one in the key field, or set the "
                f"{self.provider_combo.currentText().upper()}_API_KEY environment variable.")
            self._update_key_hint()
            return
        files = [self.items[k].path for k in keys]
        # State for the auto CSV export in _on_finished: which items this run
        # covers and which remembered folder roots those files live under.
        self._run_keys = list(keys)
        self._run_roots = self._roots_for(files)
        self._cancel_requested = False
        for k in keys:
            self._set_status(k, GENERATING)
        self._set_running(True)
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self.log.appendPlainText(f"{what.capitalize()}: {len(files)} image(s)…")

        self.worker = MetadataWorker(files, self._build_settings(), api_key)
        self.worker.log_message.connect(self.log.appendPlainText)
        self.worker.progress.connect(self._on_progress)
        self.worker.item_started.connect(
            lambda p, i, n: self.status_label.setText(
                f"Processing image {i} of {n} — {Path(p).name}…"))
        self.worker.item_done.connect(
            lambda p, m: self._set_status(self._key(Path(p)), DONE, meta=m))
        self.worker.item_failed.connect(
            lambda p, e: self._set_status(self._key(Path(p)), FAILED, error=e))
        self.worker.retry_pass.connect(self._on_retry_pass)
        self.worker.finished_all.connect(self._on_finished)
        self.worker.start()

    def generate_all(self) -> None:
        keys = [k for k, it in self.items.items() if it.status in (PENDING, FAILED)]
        self._start_generation(keys, "generating pending items")

    def retry_failed(self) -> None:
        keys = [k for k, it in self.items.items() if it.status == FAILED]
        self._start_generation(keys, "retrying failed items")

    def regenerate_selected(self) -> None:
        keys = [it.data(Qt.UserRole + 1) for it in self.grid.selectedItems()
                if it.data(Qt.UserRole + 1) in self.items]
        if not keys:
            QMessageBox.information(
                self, "Nothing selected",
                "Select one or more images in the grid first — “Regenerate "
                "Selected” always re-runs the chosen images, even if they "
                "already have metadata.")
            return
        self._start_generation(keys, "regenerating selected items")

    def cancel(self) -> None:
        if self.worker and self.worker.isRunning():
            self._cancel_requested = True
            self.worker.request_cancel()
            self.status_label.setText("Cancelling…")
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setValue(done)
        self.status_label.setText(f"{done}/{total} processed")

    def _on_retry_pass(self, pass_no: int, total_passes: int, count: int,
                       waited: float) -> None:
        """The worker kept every success and is re-running ONLY the images
        that failed with temporary errors — rescale the bar to this pass."""
        self.progress.setRange(0, count)
        self.progress.setValue(0)
        self.status_label.setText(
            f"Auto-retry pass {pass_no} of {total_passes} "
            f"(after {waited:.0f}s pause) — {count} image(s)…")
        self.log.appendPlainText(
            f"↻ Automatic retry pass {pass_no}/{total_passes}: {count} "
            f"image(s) with temporary failures, retried only after "
            f"{waited:.0f}s — successful metadata is kept.")

    def _on_finished(self, ok: int, failed: int) -> None:
        self._set_running(False)
        self.status_label.setText(f"Metadata: {ok} ok, {failed} failed")
        if failed:
            names = [it.path.name for it in self.items.values()
                     if it.status == FAILED]
            shown = ", ".join(names[:10]) + (" …" if len(names) > 10 else "")
            self.log.appendPlainText(f"✗ Failed image(s): {shown}")
            self.log.appendPlainText(
                "Temporary failures were already retried automatically — the "
                "items above failed permanently or are still unresolved. Fix "
                "the issue (key/model/image) and use “Retry Failed”; "
                "completed images are not regenerated.")
        if failed and self._run_roots and not self._cancel_requested:
            # Folder batches are all-or-nothing: successful rows stay visible
            # and retryable, but no partial CSV is written.
            self.log.appendPlainText(
                "CSV auto-export skipped — every image in the folder batch "
                "must succeed first. Automatic retries already ran; if "
                "failures remain, fix them and use “Retry Failed” — once all "
                "images are ✓ the full CSV is written automatically.")
        if ok and not self._cancel_requested:
            self._auto_export_csv()
        elif self._cancel_requested and self._run_roots:
            self.log.appendPlainText("Batch cancelled — no CSV was exported.")
        self._update_key_hint()

    def _auto_export_csv(self) -> None:
        """Write the completed metadata back to the folder the user selected.

        Folder batches export into a touched root only when EVERY image under
        that root succeeded — a root with any failed or still-pending image is
        skipped so the CSV never ships incomplete. Batches started from
        individual files keep the manual Review & Export flow and only
        auto-export once an export folder was chosen there.
        """
        if self._run_roots:
            for root in self._run_roots:
                under_root = [it for it in self.items.values()
                              if self._is_under(it.path, root)]
                if not under_root:
                    continue
                if any(it.status != DONE for it in under_root):
                    self.log.appendPlainText(
                        f"CSV not exported to {root} — not every image in "
                        "this folder batch has metadata yet.")
                    continue
                rows = [(it.path, it.meta) for it in under_root]
                self._write_auto_csv(root, rows)
        elif self._last_csv_dir is not None:
            rows = [(self.items[k].path, self.items[k].meta)
                    for k in self._run_keys if k in self.items
                    and self.items[k].status == DONE]
            if rows:
                self._write_auto_csv(self._last_csv_dir, rows)

    def _write_auto_csv(self, folder: Path, rows: list) -> None:
        dest = self._unique_csv_path(folder)
        try:
            write_csv(rows, dest)
            self.log.appendPlainText(
                f"✓ CSV auto-exported to {dest} ({len(rows)} row(s)).")
        except Exception as exc:
            self.log.appendPlainText(f"✗ CSV auto-export to {dest} failed: {exc}")

    def _set_running(self, running: bool) -> None:
        self.generate_btn.setEnabled(not running)
        self.retry_failed_btn.setEnabled(not running)
        self.regen_selected_btn.setEnabled(not running)
        self.review_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        for w in (self.provider_combo, self.model_combo, self.key_edit,
                  self.chk_remember, self.btn_forget, self.language_combo,
                  self.keywords_spin, self.chk_description):
            w.setEnabled(not running)

    # -- review & export ----------------------------------------------------------
    def review_and_export(self) -> None:
        self._commit_editor()
        if not self.items:
            QMessageBox.information(self, "Nothing to review", "Add some images first.")
            return
        rows = [(it.path, it.status, it.meta.as_dict()) for it in self.items.values()]
        dialog = ReviewDialog(rows, self)
        if dialog.exec() != ReviewDialog.Accepted:
            return
        self._do_export(dialog.export_options())

    def _do_export(self, opts) -> None:
        done_items = [it for it in self.items.values() if it.status == DONE]
        if not done_items:
            QMessageBox.information(
                self, "Nothing generated yet",
                "Generate metadata first — only completed rows can be exported.")
            return
        messages: list[str] = []
        if opts.csv_path is not None:
            try:
                rows = [(it.path, it.meta) for it in done_items]
                write_csv(rows, opts.csv_path)
                messages.append(f"CSV written: {opts.csv_path} ({len(rows)} row(s))")
                # Remembered so later file-only batches can auto-export here.
                self._last_csv_dir = opts.csv_path.parent
            except Exception as exc:
                messages.append(f"✗ CSV export failed: {exc}")

        if opts.embed:
            embeddable = [it for it in done_items
                          if it.path.suffix.lower() in EMBEDDABLE_SUFFIXES]
            skipped = len(done_items) - len(embeddable)
            ok = bad = 0
            for it in embeddable:
                try:
                    dest = None
                    if opts.embed_into_copies_dir is not None:
                        dest = opts.embed_into_copies_dir / it.path.name
                    embed_metadata(it.path, it.meta, dest)
                    ok += 1
                except Exception as exc:
                    bad += 1
                    self.log.appendPlainText(f"✗ {it.path.name}: {exc}")
            note = f"Embedded metadata in {ok} file(s)"
            if bad:
                note += f", {bad} failed"
            if skipped:
                note += f" ({skipped} non-PNG/JPEG skipped)"
            target = ("in place" if opts.embed_into_copies_dir is None
                      else f"as copies in {opts.embed_into_copies_dir}")
            messages.append(f"{note} {target}.")

        text = "\n".join(messages) if messages else "No export option was selected."
        self.log.appendPlainText(text)
        QMessageBox.information(self, "Export finished", text)

    # -- drag & drop ----------------------------------------------------------
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.toLocalFile()]
        if paths:
            self._add_paths(paths)

    # -- shutdown -------------------------------------------------------------
    def shutdown(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.request_cancel()
            self.worker.wait(3000)
