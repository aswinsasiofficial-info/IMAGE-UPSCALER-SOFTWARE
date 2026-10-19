"""PySide6 main window for the batch image upscaler."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent
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
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.engine.realesrgan_upscaler import UpscaleSettings, DEFAULT_MODEL_DIR, MODEL_FILENAME
from app.engine.worker import BatchWorker, IMAGE_EXTENSIONS, UPSCALED_DIR_NAME, collect_images


def cuda_info() -> dict:
    """Best-effort CUDA probe (torch import kept local so startup stays light)."""
    info = {"available": False, "name": "", "vram_gb": 0.0, "torch": ""}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["available"] = True
            info["name"] = props.name
            info["vram_gb"] = round(props.total_memory / (1024**3), 1)
    except Exception as exc:  # pragma: no cover - diagnostic only
        info["error"] = str(exc)
    return info


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Batch Image Upscaler — Real-ESRGAN x4plus")
        self.resize(940, 620)
        self.setAcceptDrops(True)
        self.worker: BatchWorker | None = None
        self._files: list[Path] = []
        self._folder_roots: list[Path] = []  # folders added via Add Folder / drag-drop
        self._info = cuda_info()

        self._build_menu()
        self._build_body()
        self._build_statusbar()
        self._refresh_model_hint()

    # -- construction ---------------------------------------------------------
    def _build_menu(self) -> None:
        bar = QToolBar("Main")
        bar.setMovable(False)
        self.addToolBar(bar)

        act_imgs = QAction("Add Images", self)
        act_imgs.triggered.connect(self.add_images)
        act_folder = QAction("Add Folder", self)
        act_folder.triggered.connect(self.add_folder)
        act_clear = QAction("Clear List", self)
        act_clear.triggered.connect(self.clear_queue)
        act_about = QAction("About", self)
        act_about.triggered.connect(self.show_about)
        for a in (act_imgs, act_folder, act_clear):
            bar.addAction(a)
        bar.addSeparator()
        bar.addAction(act_about)

    def _build_body(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        splitter_row = QHBoxLayout()
        splitter_row.addWidget(self._build_queue_panel(), stretch=3)
        splitter_row.addWidget(self._build_settings_panel(), stretch=2)
        root.addLayout(splitter_row)

        root.addWidget(self._build_control_bar())

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("Processing log will appear here.")
        root.addWidget(self.log, stretch=1)

        self.setCentralWidget(central)

    def _build_queue_panel(self) -> QWidget:
        box = QGroupBox("Images to process  (drag & drop here)")
        lay = QVBoxLayout(box)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        lay.addWidget(self.list)

        row = QHBoxLayout()
        for text, fn in (
            ("Add Images…", self.add_images),
            ("Add Folder…", self.add_folder),
            ("Remove Selected", self.remove_selected),
            ("Clear", self.clear_queue),
        ):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        return box

    def _build_settings_panel(self) -> QWidget:
        box = QGroupBox("Settings")
        form = QVBoxLayout(box)

        # Output location
        form.addWidget(QLabel("Output folder:"))
        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText(str(Path.home() / "Upscaled"))
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self.browse_output)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(btn_out)
        form.addLayout(out_row)
        self.chk_same_dir = QCheckBox("Save beside each original")
        self.chk_same_dir.toggled.connect(lambda v: self.out_edit.setEnabled(not v))
        form.addWidget(self.chk_same_dir)
        folder_hint = QLabel(
            f"Folders added with \"Add Folder…\" always save to "
            f"<folder>/{UPSCALED_DIR_NAME}/, keeping each image's subfolder path.")
        folder_hint.setWordWrap(True)
        form.addWidget(folder_hint)

        # Device
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CUDA (GPU)", "CPU"])
        if self._info["available"]:
            self.device_combo.setCurrentIndex(1)
        dev_row.addWidget(self.device_combo, 1)
        form.addLayout(dev_row)

        # fp16
        self.chk_fp16 = QCheckBox("Half precision (fp16) — faster, less VRAM")
        self.chk_fp16.setChecked(self._info["available"])
        form.addWidget(self.chk_fp16)

        # Tile size
        tile_row = QHBoxLayout()
        tile_row.addWidget(QLabel("Tile size (px):"))
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(0, 1024)
        self.tile_spin.setSingleStep(64)
        recommended = (
            UpscaleSettings.recommended_tile_for_vram(self._info["vram_gb"])
            if self._info["available"] else 256
        )
        self.tile_spin.setValue(recommended)
        self.tile_spin.setToolTip("0 = process whole image at once (needs more VRAM).")
        tile_row.addWidget(self.tile_spin, 1)
        tile_row.addWidget(QLabel("0 = auto"))
        form.addLayout(tile_row)

        # Exact target size (final crop-to-fill dimensions)
        form.addWidget(QLabel("Exact target size (px):"))
        target_row = QHBoxLayout()
        self.targetw_spin = QSpinBox()
        self.targetw_spin.setRange(0, 100000)
        self.targetw_spin.setSingleStep(500)
        self.targetw_spin.setValue(8000)
        self.targetw_spin.setToolTip("Exact saved image width; 0 = disable crop-to-fill.")
        self.targeth_spin = QSpinBox()
        self.targeth_spin.setRange(0, 100000)
        self.targeth_spin.setSingleStep(500)
        self.targeth_spin.setValue(5000)
        self.targeth_spin.setToolTip("Exact saved image height; 0 = disable crop-to-fill.")
        target_row.addWidget(self.targetw_spin, 1)
        target_row.addWidget(QLabel("×"))
        target_row.addWidget(self.targeth_spin, 1)
        form.addLayout(target_row)
        target_hint = QLabel(
            "Every output is exactly this size: the 4× result is resized to cover "
            "it (aspect kept, never stretched) and center-cropped. 0 disables.")
        target_hint.setWordWrap(True)
        form.addWidget(target_hint)

        # Format
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["png", "jpg"])
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        fmt_row.addWidget(self.format_combo, 1)
        fmt_row.addWidget(QLabel("JPG quality:"))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(50, 100)
        self.quality_spin.setValue(95)
        fmt_row.addWidget(self.quality_spin)
        form.addLayout(fmt_row)

        # Filename suffix
        suf_row = QHBoxLayout()
        suf_row.addWidget(QLabel("Filename suffix:"))
        self.suffix_edit = QLineEdit()
        self.suffix_edit.setText("_x4")
        suf_row.addWidget(self.suffix_edit, 1)
        form.addLayout(suf_row)
        self._on_format_changed(self.format_combo.currentText())

        form.addStretch(1)
        return box

    def _build_control_bar(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.status_label = QLabel("Idle")
        self.start_btn = QPushButton("Start Upscaling")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        lay.addWidget(self.start_btn)
        lay.addWidget(self.cancel_btn)
        lay.addWidget(self.status_label, 1)
        lay.addWidget(self.progress, 2)
        return bar

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        if self._info["available"]:
            txt = (f"GPU: {self._info['name']} ({self._info['vram_gb']} GB VRAM)"
                   f" • CUDA enabled • torch {self._info['torch']}")
        else:
            txt = f"⚠ CUDA unavailable — will run on CPU • torch {self._info['torch']}"
        sb.showMessage(txt)

    # -- helpers --------------------------------------------------------------
    def _on_format_changed(self, fmt: str) -> None:
        self.quality_spin.setEnabled(fmt == "jpg")

    def _refresh_model_hint(self) -> None:
        weights = DEFAULT_MODEL_DIR / MODEL_FILENAME
        if weights.exists():
            self.log.appendPlainText(f"Model ready: {weights.name}")
        else:
            self.log.appendPlainText("Model weights will download automatically on first run.")

    def _add_paths(self, paths: list[Path], scanned: list[Path] | None = None) -> None:
        # Remember folder selections — their images use the <folder>/upscaled/ output.
        known_roots = {str(r).lower() for r in self._folder_roots}
        for p in paths:
            if p.is_dir() and str(p).lower() not in known_roots:
                self._folder_roots.append(p)
                known_roots.add(str(p).lower())
        new = collect_images(paths) if scanned is None else scanned
        existing = {str(f).lower() for f in self._files}
        added = 0
        for p in new:
            if str(p).lower() not in existing:
                self._files.append(p)
                item = QListWidgetItem(p.name)
                item.setToolTip(str(p))
                item.setData(Qt.UserRole, str(p))
                self.list.addItem(item)
                existing.add(str(p).lower())
                added += 1
        self.status_label.setText(f"{len(self._files)} image(s) queued")
        if added == 0:
            self.log.appendPlainText("No new supported images found in that selection.")

    # -- actions --------------------------------------------------------------
    def add_images(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select images", str(Path.home()), f"Images ({exts});;All files (*)")
        if files:
            self._add_paths([Path(f) for f in files])

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select a folder", str(Path.home()))
        if not folder:
            return
        root = Path(folder)
        found = collect_images([root])
        if not found:
            QMessageBox.information(
                self, "No images found",
                f"No supported images were found in:\n{root}\n\n"
                f"Nothing was added to the queue. (A previously generated "
                f"'{UPSCALED_DIR_NAME}' output folder is skipped on purpose.)")
            return
        self._add_paths([root], scanned=found)
        self.log.appendPlainText(
            f"Folder mode: {len(found)} image(s) from {root} will be saved under "
            f"{root / UPSCALED_DIR_NAME} with subfolders preserved.")

    def _prune_folder_roots(self) -> None:
        """Drop remembered folder roots that no queued image lives under."""
        keep: list[Path] = []
        for root in self._folder_roots:
            for f in self._files:
                try:
                    Path(f).resolve().relative_to(root.resolve())
                    keep.append(root)
                    break
                except (ValueError, OSError):
                    continue
        self._folder_roots = keep

    def browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", str(Path.home()))
        if folder:
            self.out_edit.setText(folder)

    def remove_selected(self) -> None:
        for item in self.list.selectedItems():
            self._files.remove(Path(item.data(Qt.UserRole)))
            self.list.takeItem(self.list.row(item))
        self._prune_folder_roots()
        self.status_label.setText(f"{len(self._files)} image(s) queued")

    def clear_queue(self) -> None:
        self.list.clear()
        self._files.clear()
        self._folder_roots.clear()
        self.status_label.setText("0 image(s) queued")

    def show_about(self) -> None:
        QMessageBox.about(
            self, "About",
            "<b>Batch Image Upscaler</b><br>"
            "Real-ESRGAN x4plus · PyTorch + CUDA<br><br>"
            "Local 4x super-resolution for batches of images. "
            "Tile-based inference keeps large images within GPU VRAM.")

    # -- run / cancel ---------------------------------------------------------
    def _build_settings(self) -> UpscaleSettings:
        dev_choice = self.device_combo.currentText()
        device = {"Auto": "auto", "CUDA (GPU)": "cuda", "CPU": "cpu"}[dev_choice]
        out_dir = self.out_edit.text().strip() or str(Path.home() / "Upscaled")
        return UpscaleSettings(
            device=device,
            fp16=self.chk_fp16.isChecked(),
            tile=self.tile_spin.value(),
            tile_pad=16,
            output_format=self.format_combo.currentText(),
            jpg_quality=self.quality_spin.value(),
            filename_suffix=self.suffix_edit.text(),
            keep_original_name=True,
            output_in_source_dir=self.chk_same_dir.isChecked(),
            target_width=self.targetw_spin.value(),
            target_height=self.targeth_spin.value(),
        )

    def start(self) -> None:
        if not self._files:
            QMessageBox.information(self, "Nothing to do", "Add some images first.")
            return
        settings = self._build_settings()
        out_dir = Path(self.out_edit.text().strip() or str(Path.home() / "Upscaled"))

        self._set_running(True)
        self.progress.setRange(0, len(self._files))
        self.progress.setValue(0)
        self.log.appendPlainText(f"Starting batch of {len(self._files)} image(s)…")

        self.worker = BatchWorker(list(self._files), out_dir, settings,
                                  folder_roots=list(self._folder_roots))
        self.worker.log_message.connect(self.log.appendPlainText)
        self.worker.progress.connect(self._on_progress)
        self.worker.image_started.connect(lambda n: self.status_label.setText(f"Processing {n}…"))
        self.worker.download_progress.connect(self._on_download)
        self.worker.finished_all.connect(self._on_finished)
        self.worker.start()

    def cancel(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.request_cancel()
            self.status_label.setText("Cancelling…")
            self.cancel_btn.setEnabled(False)

    # -- signal handlers ------------------------------------------------------
    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.status_label.setText(f"{done}/{total} done")

    def _on_download(self, done: int, total: int) -> None:
        if total > 0:
            self.status_label.setText(f"Downloading model {done * 100 // total}%")

    def _on_finished(self, ok: int, failed: int) -> None:
        self._set_running(False)
        self.status_label.setText(f"Finished — {ok} ok, {failed} failed")
        if failed == 0:
            QMessageBox.information(self, "Done", f"Upscaled {ok} image(s).")
        else:
            QMessageBox.warning(self, "Finished with errors",
                                f"{ok} succeeded, {failed} failed. See log.")

    def _set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        for w in (self.device_combo, self.chk_fp16, self.tile_spin, self.format_combo,
                  self.quality_spin, self.chk_same_dir, self.out_edit, self.suffix_edit,
                  self.targetw_spin, self.targeth_spin):
            w.setEnabled(not running)
        if running:
            self.chk_same_dir.setEnabled(False)
        else:
            self.chk_same_dir.setEnabled(True)
            self.out_edit.setEnabled(not self.chk_same_dir.isChecked())

    # -- drag & drop ----------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.toLocalFile()]
        if paths:
            self._add_paths(paths)

    # -- shutdown -------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker and self.worker.isRunning():
            self.worker.request_cancel()
            self.worker.wait(3000)
        event.accept()
