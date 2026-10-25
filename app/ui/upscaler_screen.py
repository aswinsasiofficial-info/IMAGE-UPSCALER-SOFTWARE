"""The Upscaler screen (native Qt).

Queue images or folders, pick an upscale factor (1×–5×, applied to both
sides), see each image's thumbnail and exact output size before anything
runs, and upscale with Real-ESRGAN on a background thread. Settings are
validated with a Django form (app/upscaler/forms.py) before a batch starts;
the batch itself lives in app/upscaler/workspace.py and is polled for
progress. Metadata is a separate screen and never starts from here.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from app import design
from app.imagefiles import file_dialog_filter
from app.ui import theme
from app.ui.icons import icon, icon_button
from app.ui.thumbnails import ImageLoader
from app.ui.widgets import Banner, Preview, card, label, paint_thumbnail, paint_tile_background
from app.upscaler.engine import WEIGHTS_FILE, probe_device, recommended_tile, weights_ready
from app.upscaler.files import DEFAULT_OUTPUT_DIR, OUTPUT_DIR_NAME
from app.upscaler.forms import UpscaleForm
from app.upscaler.geometry import (DEFAULT_FACTOR, DETAIL_NOTE, FACTORS, LARGE_OUTPUT_MP, MAX_SIDE,
                                   describe, how_produced, orientation, plan_size, ratio_label)
from app.upscaler.workspace import AddResult, Busy, QueueItem, get_workspace

THUMB_EDGE = 128
PREVIEW_EDGE = 900
POLL_MS = 250
_ID_ROLE = Qt.UserRole + 1
_STATUS = {"processing": ("Processing…", "accent"), "done": ("Saved", "success"),
           "failed": ("Failed", "danger"), "cancelled": ("Cancelled", "text-muted")}


def dims(wh: tuple[int, int]) -> str:
    return f"{wh[0]} × {wh[1]}"


class _QueueDelegate(QStyledItemDelegate):
    """Row: thumbnail | name + original size | → output size + status."""

    ROW = 72
    THUMB = 56

    def __init__(self, screen: "UpscalerScreen") -> None:
        super().__init__(screen)
        self.screen = screen

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        return QSize(260, self.ROW)

    def paint(self, painter: QPainter, option, index) -> None:
        item = self.screen.queue_item(index.data(_ID_ROLE))
        if item is None:
            return
        c = design.colors(theme.current_scheme())
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        rect = option.rect.adjusted(4, 2, -4, -2)
        paint_tile_background(painter, rect, option, c)
        box = QRect(rect.left() + 8, rect.top() + (rect.height() - self.THUMB) // 2,
                    self.THUMB, self.THUMB)
        paint_thumbnail(painter, box, self.screen.thumbnail(item.path), c)

        base = QFont(option.font)
        bold = QFont(base)
        bold.setWeight(QFont.Weight.DemiBold)
        fm, fm_bold = QFontMetrics(base), QFontMetrics(bold)
        line = fm.lineSpacing()
        top = rect.top() + (rect.height() - 2 * line) // 2
        x = box.right() + 12
        right_w = max(130, fm_bold.horizontalAdvance("→ 00000 × 00000") + 8)
        left_w = rect.right() - x - right_w - 16

        painter.setFont(bold)
        painter.setPen(QColor(c["text"]))
        painter.drawText(QRect(x, top, left_w, line), Qt.AlignLeft | Qt.AlignVCenter,
                         fm_bold.elidedText(item.path.name, Qt.ElideMiddle, left_w))
        painter.setFont(base)
        painter.setPen(QColor(c["text-muted"]))
        src = (f"{dims((item.width, item.height))} · {ratio_label(item.width, item.height)} "
               f"{orientation(item.width, item.height)}")
        painter.drawText(QRect(x, top + line, left_w, line), Qt.AlignLeft | Qt.AlignVCenter,
                         fm.elidedText(src, Qt.ElideRight, left_w))

        plan = plan_size(item.width, item.height, self.screen.factor())
        rx = rect.right() - right_w - 8
        painter.setFont(bold)
        painter.setPen(QColor(c["text"]))
        painter.drawText(QRect(rx, top, right_w, line), Qt.AlignRight | Qt.AlignVCenter,
                         f"→ {dims(plan.output)}")
        painter.setFont(base)
        result = self.screen.results.get(item.id)
        warning = self.screen.warning_for(plan)
        if result and result.get("status") in _STATUS:
            text, color = _STATUS[result["status"]]
        elif warning:
            text, color = warning.split(" — ")[0], "danger"
        else:
            text, color = f"{plan.factor}× · {ratio_label(*plan.output)}", "text-faint"
        painter.setPen(QColor(c[color]))
        painter.drawText(QRect(rx, top + line, right_w, line), Qt.AlignRight | Qt.AlignVCenter,
                         fm.elidedText(text, Qt.ElideRight, right_w))
        painter.restore()


class UpscalerScreen(QWidget):
    def __init__(self, info: dict | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("page")
        self._info = info or {}
        self._items: dict[str, QueueItem] = {}
        self._rows: dict[str, QListWidgetItem] = {}
        self.results: dict[str, dict] = {}
        self._log_next = 0
        self._running = False
        self.loader = ImageLoader(self)
        self.loader.ready.connect(self._on_image_ready)
        self._poll = QTimer(self)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self._poll_job)
        self.setAcceptDrops(True)
        self._build()
        self._apply_defaults()
        self.log(f"Model ready: {WEIGHTS_FILE}" if weights_ready()
                 else "Model weights will download automatically on first run.")
        self.refresh_queue()

    # -- construction ---------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)
        self.banner = Banner()
        root.addWidget(self.banner)

        left = QSplitter(Qt.Vertical)
        left.setChildrenCollapsible(False)
        left.addWidget(self._build_queue_panel())
        left.addWidget(self._build_preview_panel())
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 2)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self._build_settings_panel())
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([820, 360])
        root.addWidget(self.splitter, 1)
        root.addWidget(self._build_action_bar())
        root.addWidget(self._build_log_panel())

    def _build_queue_panel(self) -> QWidget:
        box, lay = card()
        box.setMinimumSize(360, 220)
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(label("Images", "section"))
        self.count_label = label("No images yet", "muted")
        titles.addWidget(self.count_label)
        head.addLayout(titles)
        head.addStretch(1)
        self.btn_add_images = icon_button("image-plus", "Add images — choose image files to upscale",
                                          shortcut="Ctrl+O")
        self.btn_add_images.clicked.connect(self.add_images)
        self.btn_add_folder = icon_button(
            "folder-plus", f"Add a folder — its results are saved in <folder>/{OUTPUT_DIR_NAME}",
            accessible_name="Add folder", shortcut="Ctrl+Shift+O")
        self.btn_add_folder.clicked.connect(self.add_folder)
        self.btn_remove = icon_button("trash-2", "Remove the selected images from the queue",
                                      accessible_name="Remove selected", shortcut="Del")
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear = icon_button("list-x", "Clear the queue — removes every image",
                                     accessible_name="Clear queue")
        self.btn_clear.clicked.connect(self.clear_queue)
        for b in (self.btn_add_images, self.btn_add_folder, self.btn_remove, self.btn_clear):
            head.addWidget(b)
        lay.addLayout(head)

        self.list = QListWidget()
        self.list.setObjectName("queueList")
        self.list.setAccessibleName("Images to upscale")
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setUniformItemSizes(True)
        self.list.setMouseTracking(True)
        self.list.setItemDelegate(_QueueDelegate(self))
        self.list.currentItemChanged.connect(lambda *_: self._show_preview())
        self.list.itemSelectionChanged.connect(self._update_controls)
        delete = QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self.list)
        delete.setContext(Qt.ShortcutContext.WidgetShortcut)
        delete.activated.connect(self.remove_selected)
        self.empty_hint = label("Add images or a folder — or drag & drop them here.\nEach image "
                                "shows its thumbnail and the exact size it will be saved at.",
                                "muted", wrap=True)
        self.empty_hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.empty_hint, 1)
        return box

    def _build_preview_panel(self) -> QWidget:
        box, lay = card()
        box.setMinimumHeight(170)
        row = QHBoxLayout()
        row.setSpacing(14)
        self.preview = Preview("Select an image to see it here.")
        row.addWidget(self.preview, 3)
        info = QVBoxLayout()
        info.setSpacing(4)
        self.preview_title = label("No image selected", "section", wrap=True)
        self.preview_dims = label("", wrap=True)
        self.preview_dims.setTextFormat(Qt.RichText)
        self.preview_note = label("", "muted", wrap=True)
        info.addStretch(1)
        info.addWidget(self.preview_title)
        info.addWidget(self.preview_dims)
        info.addWidget(self.preview_note)
        info.addStretch(1)
        row.addLayout(info, 2)
        lay.addLayout(row)
        return box

    def _build_settings_panel(self) -> QWidget:
        body = QWidget()
        body.setObjectName("scrollBody")
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 4, 0)
        col.setSpacing(10)

        # Upscale factor.
        box, lay = card()
        lay.addWidget(label("Upscale factor", "section"))
        seg = QHBoxLayout()
        seg.setSpacing(4)
        self.factor_group = QButtonGroup(self)
        self.factor_buttons: dict[int, QPushButton] = {}
        for f in FACTORS:
            btn = QPushButton(f"{f}×")
            btn.setProperty("segment", True)
            btn.setProperty("factor", f)
            btn.setCheckable(True)
            btn.setMinimumWidth(44)
            btn.setAccessibleName(f"Upscale {f} times" if f > 1 else "Keep the original size (1×)")
            btn.setToolTip(how_produced(f))
            btn.toggled.connect(lambda checked, f=f: checked and self._on_factor_changed(f))
            self.factor_group.addButton(btn)
            self.factor_buttons[f] = btn
            seg.addWidget(btn)
        seg.addStretch(1)
        lay.addLayout(seg)
        self.factor_how = label("", "hint", wrap=True)
        lay.addWidget(self.factor_how)
        note = QHBoxLayout()
        note.setSpacing(6)
        info_icon = QLabel()
        info_icon.setPixmap(icon("info", "muted").pixmap(16, 16))
        info_icon.setAlignment(Qt.AlignTop)
        note.addWidget(info_icon, 0, Qt.AlignTop)
        self.detail_note = label(DETAIL_NOTE, "hint", wrap=True)
        note.addWidget(self.detail_note, 1)
        lay.addLayout(note)
        self.size_summary = label("", wrap=True)
        lay.addWidget(self.size_summary)
        self.factor_error = label("", "error", wrap=True)
        self.factor_error.hide()
        lay.addWidget(self.factor_error)
        col.addWidget(box)

        # Save to.
        box, lay = card()
        lay.addWidget(label("Save to", "section"))
        out_head = label("Output folder", "muted")
        lay.addWidget(out_head)
        out_row = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText(str(DEFAULT_OUTPUT_DIR))
        self.output_dir.setAccessibleName("Output folder")
        out_head.setBuddy(self.output_dir)
        self.btn_browse = icon_button("folder-open", "Choose the output folder",
                                      accessible_name="Browse for output folder")
        self.btn_browse.clicked.connect(self.browse_output)
        out_row.addWidget(self.output_dir, 1)
        out_row.addWidget(self.btn_browse)
        lay.addLayout(out_row)
        self.save_beside = QCheckBox("Save beside each original")
        self.save_beside.toggled.connect(lambda _v: self._update_controls())
        lay.addWidget(self.save_beside)
        lay.addWidget(label(f"Images from an added folder always save to "
                            f"<folder>/{OUTPUT_DIR_NAME}/, keeping their subfolders.", "hint", wrap=True))
        fmt_head = label("Format", "muted")
        lay.addWidget(fmt_head)
        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(4)
        self.format_group = QButtonGroup(self)
        self.format_buttons: dict[str, QPushButton] = {}
        for value, text, tip in (("jpg", "JPEG", "JPEG — smaller files (default)"),
                                 ("png", "PNG", "PNG — lossless, keeps transparency")):
            btn = QPushButton(text)
            btn.setProperty("segment", True)
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setAccessibleName(f"Save as {text}")
            btn.toggled.connect(lambda _c: self._update_controls())
            self.format_group.addButton(btn)
            self.format_buttons[value] = btn
            fmt_row.addWidget(btn)
        fmt_row.addSpacing(12)
        q_head = label("Quality", "muted")
        fmt_row.addWidget(q_head)
        self.quality = QSpinBox()
        self.quality.setRange(50, 100)
        self.quality.setAccessibleName("JPEG quality")
        q_head.setBuddy(self.quality)
        fmt_row.addWidget(self.quality)
        fmt_row.addStretch(1)
        lay.addLayout(fmt_row)
        self.rename = QCheckBox("Rename batch (e.g. KQATZ1, KQATZ2…)")
        self.rename.setToolTip("One random 4–5 letter prefix for the whole batch, numbered from 1 "
                               "in each output folder. Untick to keep the original file names.")
        lay.addWidget(self.rename)
        lay.addWidget(label("Existing files are never overwritten — taken names are skipped "
                            "or get (1), (2)…", "hint", wrap=True))
        col.addWidget(box)

        # Processing.
        box, lay = card()
        lay.addWidget(label("Processing", "section"))
        dev_head = label("Device", "muted")
        lay.addWidget(dev_head)
        self.device = QComboBox()
        self.device.addItem("Auto", "auto")
        self.device.addItem("CUDA (GPU)", "cuda")
        self.device.addItem("CPU", "cpu")
        self.device.setAccessibleName("Device")
        dev_head.setBuddy(self.device)
        lay.addWidget(self.device)
        tile_head = label("Tile size (px)", "muted")
        lay.addWidget(tile_head)
        self.tile = QSpinBox()
        self.tile.setRange(0, 1024)
        self.tile.setSingleStep(64)
        self.tile.setAccessibleName("Tile size in pixels")
        tile_head.setBuddy(self.tile)
        lay.addWidget(self.tile)
        lay.addWidget(label("Tiles keep big images within GPU memory · 0 = no tiling.", "hint",
                            wrap=True))
        self.fp16 = QCheckBox("Half precision (fp16) — faster, less VRAM")
        lay.addWidget(self.fp16)
        col.addWidget(box)
        col.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(300)
        self.settings_panel = scroll
        return scroll

    def _build_action_bar(self) -> QWidget:
        box, lay = card(padding=10, spacing=8)
        row = QHBoxLayout()
        block = QVBoxLayout()
        block.setSpacing(6)
        line = QHBoxLayout()
        self.status_label = label("Idle")
        self.progress_text = label("", "muted")
        line.addWidget(self.status_label, 1)
        line.addWidget(self.progress_text)
        block.addLayout(line)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setAccessibleName("Batch progress")
        block.addWidget(self.progress)
        row.addLayout(block, 1)
        row.addSpacing(8)
        self.cancel_btn = icon_button("circle-stop", "Cancel the batch — stops after the current "
                                      "tile; finished images are kept", accessible_name="Cancel",
                                      shortcut="Esc")
        self.cancel_btn.clicked.connect(self.cancel)
        self.start_btn = icon_button("play", "Upscale every image in the queue",
                                     accessible_name="Start upscaling", text="Upscale",
                                     variant="primary", shortcut="Ctrl+Return")
        self.start_btn.clicked.connect(self.start)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.start_btn)
        lay.addLayout(row)
        return box

    def _build_log_panel(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.log_toggle = icon_button("chevron-right", "Show or hide the activity log",
                                      accessible_name="Activity log", text="Activity log",
                                      variant="ghost", checkable=True)
        self.log_toggle.toggled.connect(self._toggle_log)
        lay.addWidget(self.log_toggle, 0, Qt.AlignLeft)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setMaximumHeight(130)
        self.log_view.setAccessibleName("Activity log")
        self.log_view.hide()
        lay.addWidget(self.log_view)
        return box

    def _toggle_log(self, shown: bool) -> None:
        self.log_view.setVisible(shown)
        name = "chevron-down" if shown else "chevron-right"
        self.log_toggle.setProperty("iconName", name)
        self.log_toggle.setIcon(icon(name))

    def _apply_defaults(self) -> None:
        device = probe_device()
        self.factor_buttons[DEFAULT_FACTOR].setChecked(True)
        self.format_buttons["jpg"].setChecked(True)          # JPEG by default
        self.quality.setValue(95)
        self.rename.setChecked(True)                         # batch rename by default
        self.device.setCurrentIndex(self.device.findData("cuda" if device["cuda"] else "auto"))
        self.fp16.setChecked(device["cuda"])
        self.tile.setValue(recommended_tile(device["vram_gb"]) if device["cuda"] else 256)
        self._on_factor_changed(DEFAULT_FACTOR)

    # -- small helpers ---------------------------------------------------------------
    @property
    def workspace(self):
        return get_workspace()

    def log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def factor(self) -> int:
        btn = self.factor_group.checkedButton()
        return int(btn.property("factor")) if btn is not None else DEFAULT_FACTOR

    def output_format(self) -> str:
        return "png" if self.format_buttons["png"].isChecked() else "jpg"

    def queue_item(self, item_id: str) -> QueueItem | None:
        return self._items.get(item_id)

    def thumbnail(self, path: Path) -> QPixmap | None:
        return self.loader.request(str(path), THUMB_EDGE)

    @staticmethod
    def warning_for(plan) -> str:
        if max(plan.output) > MAX_SIDE:
            return f"Too large — the limit is {MAX_SIDE:,} px per side; choose a smaller factor"
        if plan.peak_megapixels > LARGE_OUTPUT_MP:
            return (f"Very large — about {plan.peak_megapixels:,.0f} MP in memory while "
                    "processing; needs several GB of RAM")
        return ""

    def _current_item(self) -> QueueItem | None:
        row = self.list.currentItem()
        return self._items.get(row.data(_ID_ROLE)) if row is not None else None

    def _selected_ids(self) -> list[str]:
        return [row.data(_ID_ROLE) for row in self.list.selectedItems()]

    # -- queue -----------------------------------------------------------------------
    def refresh_queue(self) -> None:
        items = self.workspace.snapshot_items()
        ids = [i.id for i in items]
        self._items = {i.id: i for i in items}
        for item_id in [k for k in self._rows if k not in self._items]:
            row = self._rows.pop(item_id)
            self.list.takeItem(self.list.row(row))
        for index, item in enumerate(items):
            row = self._rows.get(item.id)
            if row is None:
                row = QListWidgetItem(item.path.name)
                row.setData(_ID_ROLE, item.id)
                row.setToolTip(str(item.path))
                self.list.insertItem(index, row)
                self._rows[item.id] = row
        if self.list.currentItem() is None and ids:
            self.list.setCurrentRow(0)
        self.list.viewport().update()
        n = len(items)
        self.count_label.setText(f"{n} image{'s' if n != 1 else ''}" if n else "No images yet")
        self.list.setVisible(bool(n))
        self.empty_hint.setVisible(not n)
        if not self._running:
            self.status_label.setText(f"{n} image(s) queued" if n else "Idle")
        self._update_summary()
        self._show_preview()
        self._update_controls()

    def _report_added(self, result: AddResult) -> None:
        for folder, count in result.folders:
            self.log(f"Folder mode: {count} image(s) from {folder} will be saved under "
                     f"{folder / OUTPUT_DIR_NAME} with subfolders preserved.")
        for path, error in result.unreadable:
            self.log(f"✗ Can't read {path}: {error}")
        for path in result.missing:
            self.log(f"✗ Not found: {path}")
        if result.duplicates:
            self.log(f"{result.duplicates} image(s) were already in the queue.")
        if result.empty_folders:
            QMessageBox.information(
                self, "No images found",
                "No supported images were found in:\n"
                + "\n".join(str(p) for p in result.empty_folders)
                + f"\n\nNothing was added. (A previously generated '{OUTPUT_DIR_NAME}' folder is "
                  "skipped on purpose.)")
        elif not result.added:
            self.log("No new supported images found in that selection.")
        self.refresh_queue()

    def add_paths(self, paths: list[str | Path]) -> AddResult | None:
        if self._running or not paths:
            return None
        result = self.workspace.add(paths)
        self._report_added(result)
        return result

    def add_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select images", str(Path.home()),
                                                file_dialog_filter())
        if files:
            self.add_paths(files)

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select a folder", str(Path.home()))
        if folder:
            self.add_paths([folder])

    def remove_selected(self) -> None:
        ids = self._selected_ids()
        if ids and not self._running:
            self.workspace.remove(ids)
            self.refresh_queue()

    def clear_queue(self) -> None:
        if not self._running:
            self.workspace.clear()
            self.results.clear()
            self.refresh_queue()

    def browse_output(self) -> None:
        start = self.output_dir.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder",
                                                  start if Path(start).is_dir() else str(Path.home()))
        if folder:
            self.output_dir.setText(folder)

    # -- size preview ------------------------------------------------------------------
    def _on_factor_changed(self, factor: int) -> None:
        self.factor_how.setText(how_produced(factor))
        self.factor_error.hide()
        self._update_summary()
        self._show_preview()
        self.list.viewport().update()

    def _update_summary(self) -> None:
        f = self.factor()
        items = list(self._items.values())
        if not items:
            self.size_summary.setText(f"Width and height are both multiplied by {f}.")
            return
        first = items[0]
        example = plan_size(first.width, first.height, f)
        too_big = sum(bool(self.warning_for(plan_size(i.width, i.height, f)).startswith("Too"))
                      for i in items)
        text = (f"Width and height are both multiplied by {f} — e.g. {dims(example.source)} → "
                f"{dims(example.output)}. Every image keeps its shape and whole picture.")
        if too_big:
            text += f" ⚠ {too_big} image(s) would be too large at {f}×."
        self.size_summary.setText(text)

    def _show_preview(self) -> None:
        item = self._current_item()
        if item is None:
            self.preview.show_pixmap(None, "Select an image to see it here." if self._items
                                     else "Add images to get started.")
            self.preview_title.setText("No image selected")
            self.preview_dims.setText("")
            self.preview_note.setText("")
            return
        plan = plan_size(item.width, item.height, self.factor())
        c = design.colors(theme.current_scheme())
        self.preview_title.setText(item.path.name)
        self.preview_title.setToolTip(str(item.path))
        self.preview_dims.setText(
            f"<b>{dims(plan.source)} → {dims(plan.output)}</b>"
            f"<br><span style='color:{c['text-muted']}'>{ratio_label(*plan.source)} "
            f"{orientation(*plan.source)} · still {ratio_label(*plan.output)}</span>")
        warning = self.warning_for(plan)
        self.preview_note.setText(f"{describe(plan)}." + (f"\n⚠ {warning}." if warning else ""))
        pix = self.loader.request(str(item.path), PREVIEW_EDGE) or self.loader.cached(
            str(item.path), THUMB_EDGE)
        self.preview.show_pixmap(pix, "Loading preview…")
        self.preview.setAccessibleDescription(f"Preview of {item.path.name}")

    def _on_image_ready(self, cache_key: str, pix: QPixmap) -> None:
        edge, path = cache_key.split("|", 1)
        if int(edge) == THUMB_EDGE:
            self.list.viewport().update()
        item = self._current_item()
        if item is not None and str(item.path) == path:
            if int(edge) == PREVIEW_EDGE or self.loader.cached(path, PREVIEW_EDGE) is None:
                self.preview.show_pixmap(pix)

    # -- running ---------------------------------------------------------------------------
    def settings(self) -> dict:
        return {
            "factor": self.factor(),
            "output_dir": self.output_dir.text(),
            "save_beside": self.save_beside.isChecked(),
            "device": self.device.currentData(),
            "fp16": self.fp16.isChecked(),
            "tile": self.tile.value(),
            "output_format": self.output_format(),
            "jpg_quality": self.quality.value(),
            "rename_batch": self.rename.isChecked(),
        }

    def _update_controls(self) -> None:
        r = self._running
        has_items = bool(self._items)
        for w in (self.btn_add_images, self.btn_add_folder):
            w.setEnabled(not r)
        self.btn_remove.setEnabled(not r and bool(self.list.selectedItems()))
        self.btn_clear.setEnabled(not r and has_items)
        for w in (*self.factor_buttons.values(), *self.format_buttons.values(), self.save_beside,
                  self.rename, self.device, self.tile, self.fp16):
            w.setEnabled(not r)
        self.output_dir.setEnabled(not r and not self.save_beside.isChecked())
        self.btn_browse.setEnabled(not r and not self.save_beside.isChecked())
        self.quality.setEnabled(not r and self.output_format() == "jpg")
        self.start_btn.setEnabled(not r and has_items)
        self.cancel_btn.setEnabled(r)

    def start(self) -> None:
        if self._running:
            return
        items = self.workspace.snapshot_items()
        if not items:
            QMessageBox.information(self, "Nothing to do", "Add some images first.")
            return
        form = UpscaleForm(self.settings(), images=[(i.width, i.height) for i in items])
        if not form.is_valid():
            message = form.first_error()
            self.factor_error.setText(message)
            self.factor_error.setVisible("factor" in form.errors)
            QMessageBox.warning(self, "Check the settings", message)
            return
        try:
            job = self.workspace.start(form.job_options())
        except Busy as exc:
            QMessageBox.warning(self, "Already running", str(exc))
            return
        except ValueError as exc:
            QMessageBox.information(self, "Nothing to do", str(exc))
            return
        self.results = {}
        self._log_next = 0
        self._running = True
        self.banner.hide()
        self.progress.setRange(0, len(items))
        self.progress.setValue(0)
        self.log(f"Starting batch of {len(items)} image(s) at {form.cleaned_data['factor']}×…")
        self._job = job
        self._update_controls()
        self._poll.start()
        self._poll_job()

    def cancel(self) -> None:
        if self._running:
            self.workspace.cancel()
            self.status_label.setText("Cancelling…")
            self.cancel_btn.setEnabled(False)

    def _poll_job(self) -> None:
        job = self.workspace.job
        if job is None:
            return
        snap = job.snapshot(self._log_next)
        for line in snap["log"]:
            self.log(line)
        self._log_next = snap["log_next"]
        if snap["results"] != self.results:
            self.results = snap["results"]
            self.list.viewport().update()
        self.progress.setRange(0, max(1, snap["total"]))
        self.progress.setValue(snap["done"])
        pct = round(100 * snap["done"] / snap["total"]) if snap["total"] else 0
        self.progress_text.setText(f"{snap['done']} / {snap['total']} · {pct}%")
        if snap["running"]:
            self.status_label.setText(self._status_text(snap))
            return
        self._poll.stop()
        self._finish(snap)

    @staticmethod
    def _status_text(snap: dict) -> str:
        if snap["state"] == "cancelling":
            return "Cancelling…"
        if snap["phase"] == "downloading" and snap["download"]:
            done, total = snap["download"]
            return f"Downloading model {done * 100 // total}%" if total else "Downloading model…"
        if snap["phase"] in ("starting", "loading"):
            return "Loading model…"
        return f"Processing {snap['current']}…"

    def _finish(self, snap: dict) -> None:
        self._running = False
        self._update_controls()
        cancelled = snap["state"] == "cancelled"
        self.status_label.setText(
            f"{'Cancelled' if cancelled else 'Finished'} — {snap['ok']} ok, {snap['failed']} failed")
        if snap["state"] == "failed":
            self.banner.show_message("danger", "The model could not be prepared — see the activity log.")
            self.log_toggle.setChecked(True)
            QMessageBox.warning(self, "Couldn't start",
                                "The model could not be prepared. See the activity log for details.")
        elif snap["failed"]:
            self.log_toggle.setChecked(True)
            self.banner.show_message("danger", f"{snap['ok']} image(s) upscaled, {snap['failed']} "
                                     "failed — see the activity log.")
            QMessageBox.warning(self, "Finished with errors",
                                f"{snap['ok']} succeeded, {snap['failed']} failed. See the activity log.")
        else:
            text = (f"Upscaled {snap['ok']} image(s) before cancelling." if cancelled
                    else f"Upscaled {snap['ok']} image(s).")
            self.banner.show_message("warning" if cancelled else "success", text)
            QMessageBox.information(self, "Cancelled" if cancelled else "Done", text)

    # -- drag & drop / shutdown -------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.toLocalFile()]
        if paths:
            self.add_paths(paths)

    def shutdown(self) -> None:
        self._poll.stop()
        self.workspace.shutdown(timeout=3.0)
        self.loader.shutdown()
