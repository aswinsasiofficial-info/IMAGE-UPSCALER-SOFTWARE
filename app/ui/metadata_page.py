"""Metadata Generator screen — fully independent from the upscaler.

The user picks images or a folder here (often ones the Upscaler saved), and
every image gets a list tile with its thumbnail, filename, status and title;
the selected one shows a large preview next to its editable title,
description and keywords. Providers: Gemini, Groq, or Ollama (either the
user's own local server or Ollama's hosted cloud API). Hosted keys go
through app.engine.credential_store (session → env var → Windows Credential
Manager); local Ollama needs no key at all and never uploads anything.

A folder batch holds at most 25 images; a bigger folder opens a chooser.
Temporary failures are retried automatically for the failed images only, and
the folder's complete CSV is saved by itself once every image in the batch
has metadata — never a partial one. Export writes the current metadata
(including edits) to a CSV, asking where only the first time.

Layout adapts to the window: three columns (images | editor | settings) when
wide, images + tabs when narrower.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt
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
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app import design
from app.engine import credential_store
from app.engine.metadata_settings import (
    MetadataSettings, ImageMetadata, SUPPORTED_PROVIDERS, DEFAULT_MODELS,
    MODEL_CATALOGUES, API_KEY_ENV_VARS, LANGUAGES, ADOBE_KEYWORDS_MAX, ADOBE_CATEGORIES,
    ADOBE_TITLE_MAX, ADOBE_DESCRIPTION_MAX, KEYWORD_COUNT_DEFAULT, OLLAMA_KEYWORD_COUNT,
    OLLAMA_LOCAL, OLLAMA_CLOUD, OLLAMA_LOCAL_URL, OLLAMA_CLOUD_URL,
    is_ollama, provider_requires_key, sanitize_keywords,
)
from app.engine.metadata_export import write_csv, embed_metadata, EMBEDDABLE_SUFFIXES
from app.engine.metadata_worker import MetadataWorker, OllamaLookupWorker
from app.imagefiles import discover, file_dialog_filter
from app.ui import theme
from app.ui.folder_picker import FolderPickDialog
from app.ui.icons import icon, icon_button
from app.ui.thumbnails import ImageLoader
from app.ui.widgets import (Banner, ElidedLabel, Preview, card, label, paint_thumbnail,
                            paint_tile_background)

THUMB_EDGE = 128        # decoded thumbnail size (crisp at 200 % scaling)
PREVIEW_EDGE = 900      # decoded size of the large preview
WIDE_MIN = 1180         # three columns at or above this page width
STACKED_MAX = 700       # list above details below this width (narrower than the window allows)
FOLDER_BATCH_LIMIT = 25

# Per-item generation states.
PENDING, GENERATING, DONE, FAILED = "pending", "generating", "done", "failed"
_STATUS_TEXT = {PENDING: "Not generated", GENERATING: "Generating…",
                DONE: "Done", FAILED: "Failed"}
_STATUS_COLOR = {PENDING: "text-faint", GENERATING: "accent", DONE: "success", FAILED: "danger"}
_KEY_ROLE = Qt.UserRole + 1


@dataclass
class MetaItem:
    """One queued image plus its generated/edited metadata and status."""
    path: Path
    status: str = PENDING
    error: str = ""
    meta: ImageMetadata = field(default_factory=ImageMetadata)


class _TextArea(QPlainTextEdit):
    """Multi-line field with QLineEdit-style ``text()``/``setText()``."""

    def __init__(self, min_lines: int = 2, max_lines: int = 5) -> None:
        super().__init__()
        line = QFontMetrics(self.font()).lineSpacing()
        self.setMinimumHeight(line * min_lines + 16)
        self.setMaximumHeight(line * max_lines + 16)
        self.setTabChangesFocus(True)

    def text(self) -> str:
        return self.toPlainText()

    def setText(self, text: str) -> None:  # noqa: N802 (Qt naming)
        self.setPlainText(text)


class _ImageDelegate(QStyledItemDelegate):
    """List tile: thumbnail, filename, status and title (or error)."""

    ROW = 76
    THUMB = 60

    def __init__(self, page: "MetadataPage") -> None:
        super().__init__(page)
        self.page = page

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        return QSize(240, self.ROW)       # rows stretch to the list's width anyway

    def paint(self, painter: QPainter, option, index) -> None:
        item = self.page.items.get(index.data(_KEY_ROLE))
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
        paint_thumbnail(painter, box, self.page.thumbnail(item.path), c)

        x = box.right() + 12
        width = rect.right() - x - 8
        base = QFont(option.font)
        bold = QFont(base)
        bold.setWeight(QFont.Weight.DemiBold)
        fm, fm_bold = QFontMetrics(base), QFontMetrics(bold)
        line = fm.lineSpacing()
        top = rect.top() + (rect.height() - 3 * line) // 2

        painter.setFont(bold)
        painter.setPen(QColor(c["text"]))
        painter.drawText(QRect(x, top, width, line), Qt.AlignLeft | Qt.AlignVCenter,
                         fm_bold.elidedText(item.path.name, Qt.ElideMiddle, width))

        painter.setFont(base)
        status_color = QColor(c[_STATUS_COLOR[item.status]])
        dot = QRect(x, top + line + (line - 8) // 2, 8, 8)
        painter.setPen(Qt.NoPen)
        painter.setBrush(status_color)
        painter.drawEllipse(dot)
        text = _STATUS_TEXT[item.status]
        if item.status == DONE:
            text += f" · {len(item.meta.keywords)} keywords"
        painter.setPen(status_color)
        painter.drawText(QRect(x + 14, top + line, width - 14, line),
                         Qt.AlignLeft | Qt.AlignVCenter, fm.elidedText(text, Qt.ElideRight, width - 14))

        if item.status == FAILED:
            detail, color = item.error or "Unknown error", c["danger"]
        elif item.meta.title:
            detail, color = item.meta.title, c["text-muted"]
        else:
            detail, color = ("Waiting for the provider…" if item.status == GENERATING
                             else "No metadata yet"), c["text-faint"]
        painter.setPen(QColor(color))
        painter.drawText(QRect(x, top + 2 * line, width, line), Qt.AlignLeft | Qt.AlignVCenter,
                         fm.elidedText(detail, Qt.ElideRight, width))
        painter.restore()


class MetadataPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("page")
        self.items: dict[str, MetaItem] = {}   # key: resolved path, lowercased
        self._grid_items: dict[str, QListWidgetItem] = {}
        self.worker: MetadataWorker | None = None
        self.lookup: OllamaLookupWorker | None = None   # Ollama check thread
        self._lookup_provider = ""     # which provider the running check belongs to
        self._loading_editor = False
        # Keyword count remembered while the user visits the Ollama modes, whose
        # target is a full Adobe set (exactly 49) instead of "up to 45".
        self._keyword_pref = KEYWORD_COUNT_DEFAULT
        # Folder selections are remembered so a finished folder batch can save
        # its complete CSV back into the folder the user picked.
        self._folder_roots: list[Path] = []
        self._run_keys: list[str] = []         # items in the current/last run
        self._run_roots: list[Path] = []       # folder roots that run touches
        self.csv_path: Path | None = None      # where Export saves (asked once)
        self._cancel_requested = False
        self._running = False                  # a batch owns the controls
        self._wide: bool | None = None
        self._stacked: bool | None = None
        self.loader = ImageLoader(self)
        self.loader.ready.connect(self._on_image_ready)
        self.setAcceptDrops(True)

        self._build_body()
        self._on_provider_changed(self.provider_combo.currentText())
        self._refresh_counts()

    # -- construction ---------------------------------------------------------
    def _build_body(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        self.banner = Banner()
        root.addWidget(self.banner)

        self.list_panel = self._build_list_panel()
        self.editor_panel = self._build_editor_panel()
        self.settings_panel = self._build_settings_panel()
        self.detail_tabs = QTabWidget()
        self.detail_tabs.setDocumentMode(True)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.list_panel)
        root.addWidget(self.splitter, 1)

        root.addWidget(self._build_action_bar())
        root.addWidget(self._build_log_panel())
        self._apply_layout(1280)

    def _build_list_panel(self) -> QWidget:
        box, lay = card()
        box.setMinimumSize(320, 210)
        head = QHBoxLayout()
        head.addWidget(label("Images", "section"))
        head.addStretch(1)
        self.btn_add_images = icon_button(
            "image-plus", "Add images — choose image files to describe", shortcut="Ctrl+O")
        self.btn_add_images.clicked.connect(self.add_images)
        self.btn_add_folder = icon_button(
            "folder-plus", f"Add a folder — up to {FOLDER_BATCH_LIMIT} images; its CSV is saved "
            "there automatically when all succeed", accessible_name="Add folder",
            shortcut="Ctrl+Shift+O")
        self.btn_add_folder.clicked.connect(self.add_folder)
        head.addWidget(self.btn_add_images)
        head.addWidget(self.btn_add_folder)
        lay.addLayout(head)

        self.count_label = label("", "muted")
        lay.addWidget(self.count_label)

        filters = QHBoxLayout()
        filters.setSpacing(2)
        self.filter_group = QButtonGroup(self)
        self.filter_buttons: dict[str, QPushButton] = {}
        for name in ("all", PENDING, DONE, FAILED):
            btn = QPushButton()
            btn.setProperty("segment", True)
            btn.setCheckable(True)
            btn.clicked.connect(self._apply_filter)
            self.filter_group.addButton(btn)
            self.filter_buttons[name] = btn
            filters.addWidget(btn)
        self.filter_buttons["all"].setChecked(True)
        filters.addStretch(1)
        lay.addLayout(filters)

        self.grid = QListWidget()
        self.grid.setObjectName("imageList")
        self.grid.setAccessibleName("Images to describe")
        self.grid.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.grid.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.grid.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.grid.setUniformItemSizes(True)
        self.grid.setMouseTracking(True)
        self.grid.setItemDelegate(_ImageDelegate(self))
        self.grid.currentItemChanged.connect(self._on_item_selected)
        self.grid.itemSelectionChanged.connect(self._update_actions)
        delete = QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self.grid)
        delete.setContext(Qt.ShortcutContext.WidgetShortcut)
        delete.activated.connect(self.remove_selected)
        self.empty_hint = label(f"Add images or a folder (up to {FOLDER_BATCH_LIMIT} images) — "
                                "or drag & drop them here.\nEach image gets its own preview "
                                "and metadata.", "muted", wrap=True)
        self.empty_hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.grid, 1)
        lay.addWidget(self.empty_hint, 1)

        foot = QHBoxLayout()
        foot.addStretch(1)
        self.btn_remove = icon_button("trash-2", "Remove the selected images from the list",
                                      accessible_name="Remove selected", shortcut="Del")
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear = icon_button("list-x", "Clear the list — removes every image",
                                     accessible_name="Clear list")
        self.btn_clear.clicked.connect(self.clear_all)
        foot.addWidget(self.btn_remove)
        foot.addWidget(self.btn_clear)
        lay.addLayout(foot)
        return box

    def _build_editor_panel(self) -> QWidget:
        # The fields scroll when the window is short instead of being squeezed
        # on top of each other.
        box, outer = card(padding=4, spacing=0)
        box.setMinimumWidth(340)
        body = QWidget()
        body.setObjectName("cardBody")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        scroll = QScrollArea()
        scroll.setObjectName("cardScroll")
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        self.preview = Preview()
        lay.addWidget(self.preview, 1)

        self.editor_file = QLabel("No image selected.")
        self.editor_file.setTextFormat(Qt.RichText)
        self.editor_file.setWordWrap(True)
        lay.addWidget(self.editor_file)
        self.editor_status = label("", wrap=True)
        self.editor_status.setTextFormat(Qt.RichText)
        lay.addWidget(self.editor_status)

        def field_head(text: str, counter: QLabel | None, buddy: QWidget) -> QHBoxLayout:
            row = QHBoxLayout()
            row.setContentsMargins(0, 6, 0, 0)
            head = label(text, "section")
            head.setBuddy(buddy)
            row.addWidget(head)
            row.addStretch(1)
            if counter is not None:
                row.addWidget(counter)
            return row

        self.title_edit = QLineEdit()
        self.title_edit.setMaxLength(ADOBE_TITLE_MAX)
        self.title_edit.setPlaceholderText("A short, descriptive title")
        self.title_edit.setAccessibleName("Title")
        self.title_count = label("", "counter")
        lay.addLayout(field_head("Title", self.title_count, self.title_edit))
        lay.addWidget(self.title_edit)

        self.desc_edit = _TextArea(2, 3)
        self.desc_edit.setPlaceholderText("One or two sentences about the picture")
        self.desc_edit.setAccessibleName("Description")
        self.desc_count = label("", "counter")
        lay.addLayout(field_head("Description", self.desc_count, self.desc_edit))
        lay.addWidget(self.desc_edit)

        self.kw_edit = _TextArea(2, 4)
        self.kw_edit.setPlaceholderText("Comma-separated, most important first")
        self.kw_edit.setAccessibleName("Keywords")
        self.kw_count = label("", "counter")
        lay.addLayout(field_head("Keywords", self.kw_count, self.kw_edit))
        lay.addWidget(self.kw_edit)

        self.category_combo = QComboBox()
        self.category_combo.addItem("Not set", None)
        for number, name in ADOBE_CATEGORIES.items():
            self.category_combo.addItem(f"{number} — {name}", number)
        self.category_combo.setAccessibleName("Adobe Stock category")
        self.category_combo.setToolTip("The category number written to the CSV — suggested by "
                                       "the AI; change it if it doesn't fit")
        lay.addLayout(field_head("Category", None, self.category_combo))
        lay.addWidget(self.category_combo)

        self.releases_edit = QLineEdit()
        self.releases_edit.setPlaceholderText("Release file names, if you have releases")
        self.releases_edit.setAccessibleName("Releases")
        self.releases_edit.setToolTip(
            "Exact file names of the model or property releases uploaded to Adobe Stock, "
            "comma-separated. Leave blank when there is no release.")
        lay.addLayout(field_head("Releases", None, self.releases_edit))
        lay.addWidget(self.releases_edit)
        lay.addWidget(label("Edits are kept per image and included when you export.", "hint"))

        self.title_edit.textChanged.connect(self._commit_editor)
        self.desc_edit.textChanged.connect(self._commit_editor)
        self.kw_edit.textChanged.connect(self._commit_editor)
        self.category_combo.currentIndexChanged.connect(self._commit_editor)
        self.releases_edit.textChanged.connect(self._commit_editor)
        self._set_editor_enabled(False)
        return box

    def _build_settings_panel(self) -> QWidget:
        body = QWidget()
        body.setObjectName("scrollBody")
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 4, 0)
        col.setSpacing(10)

        def field(lay: QVBoxLayout, text: str, widget: QWidget) -> None:
            head = label(text, "muted")
            head.setBuddy(widget)
            widget.setAccessibleName(text)
            lay.addWidget(head)
            lay.addWidget(widget)

        # Provider card.
        provider_card, lay = card()
        lay.addWidget(label("Provider", "section"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(SUPPORTED_PROVIDERS))
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        field(lay, "Service", self.provider_combo)
        # Model (editable so retired/new models need no code change)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        field(lay, "Model", self.model_combo)

        # Ollama-only controls: server address (local), the connection/model
        # availability check, and a plain statement of where the image goes.
        self.ollama_box = QWidget()
        ol = QVBoxLayout(self.ollama_box)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(6)
        check_row = QHBoxLayout()
        self.ollama_url_widget = QWidget()
        url_lay = QVBoxLayout(self.ollama_url_widget)
        url_lay.setContentsMargins(0, 0, 0, 0)
        url_lay.setSpacing(4)
        self.ollama_url_edit = QLineEdit(OLLAMA_LOCAL_URL)
        self.ollama_url_edit.setToolTip(
            "Address of the Ollama API on this computer (or another machine "
            f"running Ollama). Default: {OLLAMA_LOCAL_URL}")
        field(url_lay, "Ollama server", self.ollama_url_edit)
        check_row.addWidget(self.ollama_url_widget, 1)
        self.btn_ollama_check = icon_button(
            "plug-zap", "Check connection — test the Ollama server and list the models that "
            "can read images", accessible_name="Check connection and vision models")
        self.btn_ollama_check.clicked.connect(self.check_ollama)
        check_row.addWidget(self.btn_ollama_check, 0, Qt.AlignBottom)
        ol.addLayout(check_row)
        self.ollama_notice = label("", "hint", wrap=True)
        ol.addWidget(self.ollama_notice)
        lay.addWidget(self.ollama_box)

        # API key (hosted providers only — local Ollama never needs one).
        self.key_box = QWidget()
        key_lay = QVBoxLayout(self.key_box)
        key_lay.setContentsMargins(0, 0, 0, 0)
        key_lay.setSpacing(4)
        key_head = label("API key", "muted")
        key_lay.addWidget(key_head)
        key_row = QHBoxLayout()
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setClearButtonEnabled(True)
        self.key_edit.setAccessibleName("API key")
        key_head.setBuddy(self.key_edit)
        self.key_edit.textChanged.connect(lambda _t: self._update_key_hint())
        self.btn_show_key = icon_button("eye", "Show the key while typing",
                                        accessible_name="Show key", checkable=True)
        self.btn_show_key.toggled.connect(self._toggle_key_visible)
        self.btn_forget = icon_button("trash-2", "Forget the saved key for this provider "
                                      "(Windows Credential Manager)",
                                      accessible_name="Forget saved key", variant="ghost")
        self.btn_forget.clicked.connect(self.forget_key)
        key_row.addWidget(self.key_edit, 1)
        key_row.addWidget(self.btn_show_key)
        key_row.addWidget(self.btn_forget)
        key_lay.addLayout(key_row)
        self.chk_remember = QCheckBox("Remember key in Windows Credential Manager")
        key_lay.addWidget(self.chk_remember)
        self.key_hint = label("", "hint", wrap=True)
        key_lay.addWidget(self.key_hint)
        lay.addWidget(self.key_box)
        col.addWidget(provider_card)

        # Output card.
        output_card, lay = card()
        lay.addWidget(label("Output", "section"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(LANGUAGES)
        self.language_combo.setEditable(True)
        field(lay, "Language", self.language_combo)
        self.keywords_spin = QSpinBox()
        self.keywords_spin.setRange(1, ADOBE_KEYWORDS_MAX)
        self.keywords_spin.setValue(KEYWORD_COUNT_DEFAULT)
        self.keywords_spin.setToolTip(
            f"Keywords kept per image (Adobe Stock allows {ADOBE_KEYWORDS_MAX} "
            "at most). Ollama modes ask for exactly this many.")
        self.keywords_spin.valueChanged.connect(self._on_keyword_count_changed)
        field(lay, "Keywords per image", self.keywords_spin)
        self.chk_description = QCheckBox("Generate a description too")
        self.chk_description.setChecked(True)
        self.chk_description.setToolTip("The description is written into images with Embed "
                                        "Metadata. The Adobe Stock CSV has no description column.")
        lay.addWidget(self.chk_description)
        col.addWidget(output_card)
        col.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(290)
        return scroll

    def _toggle_key_visible(self, shown: bool) -> None:
        self.key_edit.setEchoMode(QLineEdit.Normal if shown else QLineEdit.Password)
        name = "eye-off" if shown else "eye"
        self.btn_show_key.setProperty("iconName", name)
        self.btn_show_key.setIcon(icon(name))
        tip = "Hide the key" if shown else "Show the key while typing"
        self.btn_show_key.setToolTip(tip)
        self.btn_show_key.setAccessibleName("Hide key" if shown else "Show key")

    def _build_action_bar(self) -> QWidget:
        box, lay = card(padding=10, spacing=8)
        progress_row = QHBoxLayout()
        self.status_label = label("Idle", "muted")
        self.status_label.setMinimumWidth(120)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setAccessibleName("Overall progress")
        progress_row.addWidget(self.status_label, 2)
        progress_row.addWidget(self.progress, 3)
        lay.addLayout(progress_row)

        row = QHBoxLayout()
        self.generate_btn = icon_button(
            "sparkles", "Generate metadata for every image that doesn't have it yet",
            accessible_name="Generate metadata", text="Generate", variant="primary",
            shortcut="Ctrl+Return")
        self.generate_btn.clicked.connect(self.generate_all)
        self.retry_failed_btn = icon_button(
            "rotate-ccw", "Retry failed — run only the failed images again; finished ones "
            "are kept", accessible_name="Retry failed images")
        self.retry_failed_btn.clicked.connect(self.retry_failed)
        self.regen_selected_btn = icon_button(
            "refresh-cw", "Regenerate selected — replace the metadata of the selected images",
            accessible_name="Regenerate selected images")
        self.regen_selected_btn.clicked.connect(self.regenerate_selected)
        self.cancel_btn = icon_button("circle-stop", "Cancel the running batch — finished "
                                      "images keep their metadata", accessible_name="Cancel",
                                      shortcut="Esc")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        for b in (self.generate_btn, self.retry_failed_btn, self.regen_selected_btn, self.cancel_btn):
            row.addWidget(b)
        row.addStretch(1)

        self.export_location = ElidedLabel()
        self.export_location.setProperty("role", "hint")
        self.export_location.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.export_location, 1)
        self.export_btn = icon_button(
            "download", "Export — save the current metadata, including your edits, as a CSV",
            accessible_name="Export CSV", text="Export", shortcut="Ctrl+E")
        self.export_btn.setObjectName("exportButton")
        self.export_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self.export_btn.clicked.connect(lambda: self.export_csv())
        menu = QMenu(self.export_btn)
        self.act_export_as = menu.addAction(icon("save"), "Export As…")
        self.act_export_as.triggered.connect(lambda: self.export_csv(choose=True))
        self.act_embed = menu.addAction(icon("tag"), "Embed Metadata into Images…")
        self.act_embed.triggered.connect(self.embed_into_images)
        self.export_btn.setMenu(menu)
        row.addWidget(self.export_btn)
        lay.addLayout(row)
        self._update_export_location()
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
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("Metadata log will appear here.")
        self.log.setAccessibleName("Activity log")
        self.log.setMaximumHeight(130)
        self.log.hide()
        lay.addWidget(self.log)
        return box

    def _toggle_log(self, shown: bool) -> None:
        self.log.setVisible(shown)
        name = "chevron-down" if shown else "chevron-right"
        self.log_toggle.setProperty("iconName", name)
        self.log_toggle.setIcon(icon(name))

    # -- responsive layout ----------------------------------------------------------
    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_layout(event.size().width())

    def _apply_layout(self, width: int) -> None:
        wide, stacked = width >= WIDE_MIN, width < STACKED_MAX
        if (wide, stacked) == (self._wide, self._stacked):
            return
        self._wide, self._stacked = wide, stacked
        current_tab = self.detail_tabs.currentIndex()
        while self.detail_tabs.count():
            self.detail_tabs.removeTab(0)
        if wide:
            self.detail_tabs.setParent(None)
            self.splitter.setOrientation(Qt.Horizontal)
            self.splitter.insertWidget(1, self.editor_panel)
            self.splitter.insertWidget(2, self.settings_panel)
            # Tab pages are hidden explicitly; a splitter won't show them by itself.
            self.editor_panel.show()
            self.settings_panel.show()
            self.splitter.setSizes([360, 560, 330])
            self.splitter.setStretchFactor(0, 2)
            self.splitter.setStretchFactor(1, 3)
            self.splitter.setStretchFactor(2, 0)
        else:
            self.detail_tabs.addTab(self.editor_panel, "Metadata")
            self.detail_tabs.addTab(self.settings_panel, "Provider && options")
            self.detail_tabs.setCurrentIndex(max(0, current_tab))
            self.splitter.insertWidget(1, self.detail_tabs)
            self.splitter.setOrientation(Qt.Vertical if stacked else Qt.Horizontal)
            self.splitter.setSizes([320, 680] if not stacked else [300, 500])
            self.splitter.setStretchFactor(0, 1)
            self.splitter.setStretchFactor(1, 2)

    # -- provider / key handling -------------------------------------------------
    def _on_provider_changed(self, provider: str) -> None:
        self.model_combo.clear()
        self.model_combo.addItems(list(MODEL_CATALOGUES.get(provider, [])))
        self.model_combo.setCurrentText(DEFAULT_MODELS.get(provider, ""))
        ollama = is_ollama(provider)
        cloud = provider == OLLAMA_CLOUD
        self.ollama_box.setVisible(ollama)
        self.ollama_url_widget.setVisible(ollama and not cloud)
        self.key_box.setVisible(not ollama or cloud)
        if ollama:
            # Ollama aims at a complete Adobe keyword set; the hosted providers
            # go back to whatever the user had chosen before.
            self.keywords_spin.setValue(OLLAMA_KEYWORD_COUNT)
            self.ollama_notice.setText(
                "☁ Ollama Cloud is a hosted service: every image is uploaded to "
                f"{OLLAMA_CLOUD_URL} for analysis, and which models work depends "
                "on your account and plan. Keys are stored in the Windows "
                "Credential Manager (or the OLLAMA_API_KEY environment variable)."
                if cloud else
                "⛁ Local Ollama: images are only sent to the Ollama server on "
                "your computer and never leave it. No API key is needed — use "
                "the check button to see your installed image-capable models.")
        else:
            self.keywords_spin.setValue(min(self._keyword_pref,
                                            ADOBE_KEYWORDS_MAX))
        env_var = API_KEY_ENV_VARS.get(provider, "")
        self.key_edit.setPlaceholderText(f"or set {env_var}, or a saved key")
        self._update_key_hint()

    def _on_keyword_count_changed(self, value: int) -> None:
        """Remember the user's target for the hosted providers. The Ollama modes
        always aim for a full Adobe set, so their value is not carried over."""
        if not is_ollama(self.provider_combo.currentText()):
            self._keyword_pref = value

    def _update_key_hint(self) -> None:
        provider = self.provider_combo.currentText()
        if not provider_requires_key(provider):
            self.key_hint.setText("✓ Local Ollama needs no key — nothing is "
                                  "stored or uploaded.")
        elif self.key_edit.text().strip():
            self.key_hint.setText("✓ Key typed in — used for this run only unless "
                                  '"Remember" is ticked at generation time.')
        elif credential_store.get_key(provider):
            self.key_hint.setText(
                f"✓ A key for {provider} was found (typed earlier, "
                f"{API_KEY_ENV_VARS.get(provider, '')} env var, or Windows "
                "Credential Manager).")
        else:
            self.key_hint.setText(
                f"✗ No {provider} key configured. Enter one above, set the "
                f"{API_KEY_ENV_VARS.get(provider, '')} environment variable, or "
                "save it via Windows Credential Manager.")

    def _resolve_api_key(self) -> str:
        """Pick the key for the current provider and remember it if asked."""
        provider = self.provider_combo.currentText()
        if not provider_requires_key(provider):
            return ""          # local Ollama: never store, never require a key
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

    # -- Ollama connection / model availability ----------------------------------
    def check_ollama(self) -> None:
        """Ask the selected Ollama mode what it is and which models take images."""
        provider = self.provider_combo.currentText()
        if not is_ollama(provider):
            return
        if self.lookup is not None and self.lookup.isRunning():
            self.log.appendPlainText("An Ollama check is already running.")
            return
        cloud = provider == OLLAMA_CLOUD
        base = OLLAMA_CLOUD_URL if cloud else self.ollama_url_edit.text()
        key = self._resolve_api_key() if cloud else ""
        self.btn_ollama_check.setEnabled(False)
        self.status_label.setText(f"Checking {provider}…")
        self.log.appendPlainText(
            f"Checking {provider} at {base} for image-capable models…")
        self.lookup = OllamaLookupWorker(base, key, cloud)
        self._lookup_provider = provider
        self.lookup.log_message.connect(self.log.appendPlainText)
        self.lookup.result.connect(self._on_lookup_result)
        self.lookup.start()

    def _on_lookup_result(self, models: list, message: str) -> None:
        provider = self.provider_combo.currentText()
        self.btn_ollama_check.setEnabled(not self.worker_running())
        self.status_label.setText(f"{len(self.items)} image(s) in list")
        self.log.appendPlainText(message)
        if provider != self._lookup_provider:
            self.log.appendPlainText(
                f"Result was for {self._lookup_provider}; you already switched "
                "providers, so the model list was left alone.")
            return
        if not models:
            self.ollama_notice.setText(f"✗ {message}")
            return
        self.model_combo.clear()
        self.model_combo.addItems(models)
        self.model_combo.setCurrentText(models[0])
        found = f"✓ {len(models)} image-capable model(s) ready to use."
        if provider == OLLAMA_CLOUD:
            found += (" These are the models this account can reach now; cloud "
                      "requests upload the image to " + OLLAMA_CLOUD_URL + ".")
        else:
            found += (" Only models installed on this machine are listed — "
                      "images stay local.")
        self.ollama_notice.setText(found)

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

    def _choose_folder_images(self, folder: Path, images: list[Path], limit: int,
                              already: int) -> list[Path] | None:
        """Let the user pick up to *limit* of a folder's images (None = cancelled)."""
        dialog = FolderPickDialog(folder, images, limit, self.loader, already, self)
        if dialog.exec() != FolderPickDialog.Accepted:
            return None
        return dialog.selected_paths()[:limit]

    def _folder_files(self, folder: Path) -> list[Path] | None:
        """The images of *folder* to add, keeping its batch within the 25-image limit."""
        found = [p for p in discover([folder]) if self._key(p) not in self.items]
        already = sum(1 for it in self.items.values() if self._is_under(it.path, folder))
        room = FOLDER_BATCH_LIMIT - already
        if len(found) <= room:
            return found
        if room <= 0:
            QMessageBox.information(
                self, "Folder batch is full",
                f"“{folder.name}” already has {already} images in the list — a folder batch "
                f"holds up to {FOLDER_BATCH_LIMIT}. Remove some to add others.")
            return None
        self.log.appendPlainText(
            f"“{folder}” has {len(found) + already} supported images — a folder batch holds "
            f"up to {FOLDER_BATCH_LIMIT}; choose which to add.")
        return self._choose_folder_images(folder, found, room, already)

    def _add_paths(self, paths: list[Path]) -> None:
        files: list[Path] = []
        for p in map(Path, paths):
            try:
                is_dir = p.is_dir()
            except OSError:
                continue
            if not is_dir:
                files.append(p)
                continue
            chosen = self._folder_files(p)
            if not chosen:
                continue
            # Remember the folder for the automatic CSV once its batch completes.
            if self._norm(p) not in (self._norm(r) for r in self._folder_roots):
                self._folder_roots.append(p)
            files.extend(chosen)
        found = discover(files)
        added = 0
        self.grid.setUpdatesEnabled(False)
        for p in found:
            key = self._key(p)
            if key in self.items:
                continue
            self.items[key] = MetaItem(path=Path(p))
            self._add_grid_item(Path(p))
            added += 1
        self.grid.setUpdatesEnabled(True)
        self.status_label.setText(f"{len(self.items)} image(s) in list")
        if added == 0:
            self.log.appendPlainText("No new supported images found in that selection.")
        else:
            self.log.appendPlainText(f"Added {added} image(s) for metadata generation.")
        if self.grid.currentItem() is None and self.grid.count():
            self.grid.setCurrentRow(0)
        self._refresh_counts()

    def _add_grid_item(self, path: Path) -> QListWidgetItem:
        item = QListWidgetItem(path.name)      # tile painted by _ImageDelegate
        item.setToolTip(str(path))
        item.setData(Qt.UserRole, str(path))
        item.setData(_KEY_ROLE, self._key(path))
        self.grid.addItem(item)
        self._grid_items[self._key(path)] = item
        return item

    def thumbnail(self, path: Path) -> QPixmap | None:
        """Cached thumbnail, or None while it loads (requested on first paint)."""
        return self.loader.request(str(path), THUMB_EDGE)

    def _on_image_ready(self, cache_key: str, pix: QPixmap) -> None:
        edge, path = cache_key.split("|", 1)
        if int(edge) == THUMB_EDGE:
            self.grid.viewport().update()
        current = self._current_meta_item()
        if current is not None and str(current.path) == path:
            if int(edge) == PREVIEW_EDGE or self.loader.cached(path, PREVIEW_EDGE) is None:
                self.preview.show_pixmap(pix)

    def add_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select images", str(Path.home()), file_dialog_filter())
        if files:
            self._add_paths([Path(f) for f in files])

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select a folder", str(Path.home()))
        if folder:
            self._add_paths([Path(folder)])

    def remove_selected(self) -> None:
        for item in self.grid.selectedItems():
            key = item.data(_KEY_ROLE)
            self.items.pop(key, None)
            self._grid_items.pop(key, None)
            self.grid.takeItem(self.grid.row(item))
        self.status_label.setText(f"{len(self.items)} image(s) in list")
        if self.grid.currentItem() is None:
            self._show_item(None)
        self._refresh_counts()

    def clear_all(self) -> None:
        if self.worker and self.worker.isRunning():
            self.cancel()
        self.grid.clear()
        self.items.clear()
        self._grid_items.clear()
        self._folder_roots.clear()
        self.csv_path = None
        self._update_export_location()
        self._show_item(None)
        self.banner.hide()
        self.status_label.setText("0 image(s) in list")
        self._refresh_counts()

    # -- counts, filters, action state ---------------------------------------------
    def _counts(self) -> dict[str, int]:
        counts = {PENDING: 0, DONE: 0, FAILED: 0}
        for it in self.items.values():
            counts[PENDING if it.status == GENERATING else it.status] += 1
        return counts

    def _refresh_counts(self) -> None:
        counts, total = self._counts(), len(self.items)
        parts = [f"{total} image{'s' if total != 1 else ''}"]
        if total:
            parts += [f"{counts[DONE]} done", f"{counts[FAILED]} failed", f"{counts[PENDING]} pending"]
        self.count_label.setText(" · ".join(parts))
        names = {"all": "All", PENDING: "Pending", DONE: "Done", FAILED: "Failed"}
        for name, btn in self.filter_buttons.items():
            count = total if name == "all" else counts[name]
            btn.setText(f"{names[name]} {count}")
            btn.setAccessibleName(f"Show {names[name].lower()} images ({count})")
        has_items = total > 0
        self.grid.setVisible(has_items)
        self.empty_hint.setVisible(not has_items)
        self._apply_filter()
        self._update_actions()

    def _apply_filter(self) -> None:
        wanted = next((name for name, btn in self.filter_buttons.items() if btn.isChecked()), "all")
        for i in range(self.grid.count()):
            it = self.grid.item(i)
            meta_item = self.items.get(it.data(_KEY_ROLE))
            status = PENDING if meta_item is None or meta_item.status == GENERATING else meta_item.status
            self.grid.setRowHidden(i, wanted != "all" and status != wanted)

    def _update_actions(self) -> None:
        running = self._running
        counts = self._counts()
        self.generate_btn.setEnabled(not running and bool(self.items))
        self.retry_failed_btn.setEnabled(not running and counts[FAILED] > 0)
        self.regen_selected_btn.setEnabled(not running and bool(self.grid.selectedItems()))
        self.export_btn.setEnabled(not running and counts[DONE] > 0)
        self.cancel_btn.setEnabled(running and not self._cancel_requested)
        self.btn_remove.setEnabled(bool(self.grid.selectedItems()))
        self.btn_clear.setEnabled(bool(self.items))

    def _update_export_location(self) -> None:
        if self.csv_path is None:
            self.export_location.setText("Export asks where to save the CSV")
            self.export_location.setToolTip("")
        else:
            self.export_location.setText(f"Saves to {self.csv_path.name}")
            self.export_location.setToolTip(str(self.csv_path))

    # -- editor ------------------------------------------------------------------
    def _on_item_selected(self, current: QListWidgetItem | None,
                          previous: QListWidgetItem | None) -> None:
        # Commit against the item being LEFT: by the time this signal fires the
        # grid's current item is already the new one, so a plain
        # _commit_editor() would paste the previous image's text over the newly
        # selected image's generated metadata.
        if previous is not None:
            try:
                self._commit_editor(previous)
            except RuntimeError:      # item already destroyed (grid cleared)
                pass
        self._show_item(current)

    def _grid_item(self, key: str) -> QListWidgetItem | None:
        return self._grid_items.get(key)

    def _current_meta_item(self) -> MetaItem | None:
        current = self.grid.currentItem()
        return self.items.get(current.data(_KEY_ROLE)) if current is not None else None

    def _set_editor_enabled(self, enabled: bool) -> None:
        for w in (self.title_edit, self.desc_edit, self.kw_edit, self.category_combo,
                  self.releases_edit):
            w.setEnabled(enabled)

    def _show_item(self, grid_item: QListWidgetItem | None) -> None:
        self._loading_editor = True
        meta_item = self.items.get(grid_item.data(_KEY_ROLE)) if grid_item is not None else None
        if meta_item is None:
            self.editor_file.setText("No image selected.")
            self.title_edit.setText("")
            self.desc_edit.setPlainText("")
            self.kw_edit.setText("")
            self.category_combo.setCurrentIndex(0)
            self.releases_edit.setText("")
            self.editor_status.setText("")
            self.preview.show_pixmap(None, "Select an image to see it here." if self.items
                                     else "Add images to get started.")
            self._set_editor_enabled(False)
        else:
            path = meta_item.path
            c = design.colors(theme.current_scheme())
            # Break opportunities after separators, so a long folder path wraps
            # instead of forcing the editor wider than its panel.
            folder = html.escape(str(path.parent)).replace("\\", "\\&#8203;").replace("/", "/&#8203;")
            self.editor_file.setText(
                f"<b>{html.escape(path.name)}</b><br><span style='color:{c['text-muted']}'>{folder}</span>")
            self.editor_file.setToolTip(str(path))
            self.title_edit.setText(meta_item.meta.title)
            self.desc_edit.setPlainText(meta_item.meta.description)
            self.kw_edit.setText(", ".join(meta_item.meta.keywords))
            self.category_combo.setCurrentIndex(
                max(0, self.category_combo.findData(meta_item.meta.category)))
            self.releases_edit.setText(meta_item.meta.releases)
            self._update_status_label(meta_item)
            pix = self.loader.request(str(path), PREVIEW_EDGE) or self.loader.cached(str(path), THUMB_EDGE)
            self.preview.show_pixmap(pix, "Loading preview…")
            self.preview.setAccessibleDescription(f"Preview of {path.name}")
            self._set_editor_enabled(meta_item.status != GENERATING)
        self._loading_editor = False
        self._update_counters()

    def _commit_editor(self, target=None) -> None:
        """Write the visible editor fields back into the selected item.

        ``target`` lets the selection-change handler name the item that is
        being left. The ``textChanged`` slots hand us the new text instead, so
        anything that is not a grid item means "use the current selection".
        """
        if self._loading_editor:
            return
        grid_item = (target if isinstance(target, QListWidgetItem)
                     else self.grid.currentItem())
        if grid_item is None:
            return
        meta_item = self.items.get(grid_item.data(_KEY_ROLE))
        if meta_item is None or meta_item.status == GENERATING:
            return
        meta_item.meta.title = self.title_edit.text().strip()
        meta_item.meta.description = self.desc_edit.toPlainText().strip()
        meta_item.meta.keywords = [
            k.strip() for k in self.kw_edit.text().replace("\n", ",").split(",") if k.strip()]
        meta_item.meta.category = self.category_combo.currentData()
        meta_item.meta.releases = self.releases_edit.text().strip()
        self._update_counters()
        self.grid.update(self.grid.indexFromItem(grid_item))

    def _update_counters(self) -> None:
        def show(lbl: QLabel, text: str, over: bool) -> None:
            lbl.setText(text)
            if lbl.property("over") != over:
                lbl.setProperty("over", over)
                theme.repolish(lbl)

        title = len(self.title_edit.text())
        show(self.title_count, f"{title}/{ADOBE_TITLE_MAX}", title > ADOBE_TITLE_MAX)
        desc = len(self.desc_edit.toPlainText())
        show(self.desc_count, f"{desc}/{ADOBE_DESCRIPTION_MAX}", desc > ADOBE_DESCRIPTION_MAX)
        keywords = len([k for k in self.kw_edit.text().replace("\n", ",").split(",") if k.strip()])
        show(self.kw_count, f"{keywords}/{ADOBE_KEYWORDS_MAX}", keywords > ADOBE_KEYWORDS_MAX)

    # Kept for callers of the old name.
    _update_title_count = _update_counters

    def _update_status_label(self, meta_item: MetaItem) -> None:
        c = design.colors(theme.current_scheme())
        color = c[_STATUS_COLOR[meta_item.status]]
        if meta_item.status == DONE:
            text = f"● Generated · {len(meta_item.meta.keywords)} keywords"
        elif meta_item.status == FAILED:
            text = (f"● Failed — {html.escape(meta_item.error)}<br><span style='color:{c['text-muted']}'>"
                    "Fix the cause, then use Retry failed or Regenerate selected.</span>")
        elif meta_item.status == GENERATING:
            text = "● Generating…"
        else:
            text = "● Not generated yet"
        self.editor_status.setText(f"<span style='color:{color}'>{text}</span>")

    # -- status painting on grid -------------------------------------------------
    def _set_status(self, key: str, status: str, error: str = "",
                    meta: dict | None = None) -> None:
        meta_item = self.items.get(key)
        if meta_item is None:
            return
        meta_item.status = status
        meta_item.error = error
        if meta is not None:
            releases = meta_item.meta.releases     # typed by the user, never generated
            meta_item.meta = ImageMetadata.from_dict(meta)
            meta_item.meta.releases = meta_item.meta.releases or releases
        grid_item = self._grid_item(key)
        if grid_item is not None:
            tip = str(meta_item.path)
            if status == FAILED:
                tip += f"\n✗ {error}"
            elif status == DONE:
                tip += f"\n{meta_item.meta.title}\n{', '.join(meta_item.meta.keywords)}"
            grid_item.setToolTip(tip)
            self.grid.update(self.grid.indexFromItem(grid_item))
        if grid_item is not None and self.grid.currentItem() is grid_item:
            self._show_item(grid_item)
        self._refresh_counts()

    # -- generation --------------------------------------------------------------
    def worker_running(self) -> bool:
        """True while a metadata batch owns the screen's controls."""
        return bool(self.worker and self.worker.isRunning())

    def _build_settings(self) -> MetadataSettings:
        provider = self.provider_combo.currentText()
        return MetadataSettings(
            provider=provider,
            model=self.model_combo.currentText().strip() or DEFAULT_MODELS.get(
                provider, ""),
            language=self.language_combo.currentText().strip() or "English",
            keyword_count=self.keywords_spin.value(),
            generate_description=self.chk_description.isChecked(),
            ollama_url=self.ollama_url_edit.text().strip() or OLLAMA_LOCAL_URL,
            # Ollama fills the Adobe keyword slots instead of leaving one spare.
            exact_keyword_count=is_ollama(provider),
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
        provider = self.provider_combo.currentText()
        api_key = self._resolve_api_key()
        if provider_requires_key(provider) and not api_key:
            QMessageBox.warning(
                self, "Missing API key",
                f"No {provider} API key is configured.\n\n"
                "Enter one in the key field, or set the "
                f"{API_KEY_ENV_VARS.get(provider, '')} environment variable.")
            self._update_key_hint()
            return
        if is_ollama(provider) and not self.model_combo.currentText().strip():
            QMessageBox.warning(
                self, "No Ollama model selected",
                f"{provider} needs a model that can analyse images.\n\n"
                "Use the check button to list the image-capable ones, or type "
                "a model name you know works.")
            return
        self._commit_editor()
        files = [self.items[k].path for k in keys]
        # State for the automatic CSV in _on_finished: which items this run
        # covers and which remembered folder roots those files live under.
        self._run_keys = list(keys)
        self._run_roots = self._roots_for(files)
        self._cancel_requested = False
        self.banner.hide()
        self._set_running(True)
        for k in keys:
            self._set_status(k, GENERATING)
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self.log.appendPlainText(f"{what.capitalize()}: {len(files)} image(s)…")

        self.worker = MetadataWorker(files, self._build_settings(), api_key)
        self.worker.log_message.connect(self.log.appendPlainText)
        self.worker.progress.connect(self._on_progress)
        self.worker.item_started.connect(
            lambda p, i, n: self.status_label.setText(
                f"Image {i} of {n} — {Path(p).name}"))
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
        keys = [it.data(_KEY_ROLE) for it in self.grid.selectedItems()
                if it.data(_KEY_ROLE) in self.items]
        if not keys:
            QMessageBox.information(
                self, "Nothing selected",
                "Select one or more images in the list first — Regenerate selected "
                "always re-runs the chosen images, even if they already have metadata.")
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
        self.status_label.setText(f"{done} of {total} processed")

    def _on_retry_pass(self, pass_no: int, total_passes: int, count: int,
                       waited: float) -> None:
        """The worker kept every success and is re-running ONLY the images
        that failed with temporary errors — rescale the bar to this pass."""
        self.progress.setRange(0, count)
        self.progress.setValue(0)
        self.status_label.setText(
            f"Automatic retry {pass_no - 1} of {total_passes - 1} — {count} image(s)")
        self.log.appendPlainText(
            f"↻ Automatic retry pass {pass_no}/{total_passes}: {count} "
            f"image(s) with temporary failures, retried only after "
            f"{waited:.0f}s — successful metadata is kept.")

    def _on_finished(self, ok: int, failed: int) -> None:
        self._set_running(False)
        self.status_label.setText(f"Finished — {ok} ok, {failed} failed")
        folder_batch = bool(self._run_roots)
        if self._cancel_requested:
            self.log.appendPlainText("Batch cancelled — no CSV was exported."
                                     if folder_batch else "Batch cancelled.")
            self.banner.show_message(
                "warning", f"Batch cancelled after {ok} image(s). "
                + ("No CSV was saved automatically — it is only written once every image "
                   "in the folder has metadata." if folder_batch else
                   "Finished metadata is kept; click Generate to continue."))
            self._update_key_hint()
            return
        if failed:
            names = [it.path.name for it in self.items.values() if it.status == FAILED]
            shown = ", ".join(names[:10]) + (" …" if len(names) > 10 else "")
            self.log.appendPlainText(f"✗ Failed image(s): {shown}")
            self.log.appendPlainText(
                "Temporary failures were already retried automatically — the "
                "items above failed permanently or are still unresolved. Fix "
                "the issue (key/model/image) and use “Retry failed”; "
                "completed images are not regenerated.")
            text = (f"{failed} image(s) failed after automatic retries. Select one (or use the "
                    "Failed filter) to see why, fix it, then use Retry failed — finished "
                    "images are kept.")
            if folder_batch:
                self.log.appendPlainText(
                    "CSV auto-export skipped — every image in the folder batch "
                    "must succeed first. Automatic retries already ran; if "
                    "failures remain, fix them and use “Retry failed” — once all "
                    "images are ✓ the full CSV is written automatically.")
                text += " The folder's CSV is saved automatically once every image succeeds."
            self.banner.show_message("danger", text)
        saved = self._auto_export_csv() if ok and folder_batch else []
        if not failed:
            if saved:
                self.banner.show_message(
                    "success", f"All images have metadata. Complete CSV saved to {saved[-1]}. "
                    "Edit anything you like, then use Export to update it.")
            elif not folder_batch:
                self.banner.show_message(
                    "success", f"Metadata generated for {ok} image(s). Review or edit it, "
                    "then use Export to save a CSV.")
        self._update_key_hint()

    def _auto_export_csv(self) -> list[Path]:
        """Save the completed metadata into the folder(s) the user selected.

        A folder gets its CSV only when EVERY image under it has metadata — a
        folder with any failed or still-pending image is skipped, so the CSV
        never ships incomplete. Batches of individually picked files are
        saved with Export instead.
        """
        written: list[Path] = []
        for root in self._run_roots:
            under_root = [it for it in self.items.values() if self._is_under(it.path, root)]
            if not under_root:
                continue
            if any(it.status != DONE for it in under_root):
                self.log.appendPlainText(
                    f"CSV not exported to {root} — not every image in "
                    "this folder batch has metadata yet.")
                continue
            # Re-use the CSV this folder already has from us; never touch other files.
            dest = (self.csv_path if self.csv_path is not None
                    and self._norm(self.csv_path.parent) == self._norm(root)
                    else self._unique_csv_path(root))
            rows = [(it.path, it.meta) for it in under_root]
            try:
                write_csv(rows, dest)
            except Exception as exc:
                self.log.appendPlainText(f"✗ CSV auto-export to {dest} failed: {exc}")
                self.banner.show_message("danger", f"Could not save the CSV to {dest}: {exc}")
                continue
            self.log.appendPlainText(f"✓ CSV auto-exported to {dest} ({len(rows)} row(s)).")
            written.append(dest)
        if len(written) == 1:
            self.csv_path = written[0]           # Export updates this file from now on
            self._update_export_location()
        return written

    def _set_running(self, running: bool) -> None:
        self._running = running
        for w in (self.provider_combo, self.model_combo, self.key_edit,
                  self.chk_remember, self.btn_forget, self.language_combo,
                  self.keywords_spin, self.chk_description,
                  self.ollama_url_edit, self.btn_ollama_check):
            w.setEnabled(not running)
        self._update_actions()

    # -- export ---------------------------------------------------------------------
    def _default_export_dir(self) -> Path:
        if self._folder_roots:
            return self._folder_roots[0]
        first = next(iter(self.items.values()), None)
        return first.path.parent if first is not None else Path.home()

    def export_csv(self, choose: bool = False) -> Path | None:
        """Save every image's current metadata (edits included) as a CSV.

        Asks for a location the first time (or with ``choose``), then keeps
        saving to it. Images without metadata yet are left out and reported.
        """
        self._commit_editor()
        done = [it for it in self.items.values() if it.status == DONE]
        if not done:
            QMessageBox.information(
                self, "Nothing to export yet",
                "Generate metadata first — the CSV contains images that have metadata.")
            return None
        dest = self.csv_path
        if choose or dest is None:
            start = dest if dest is not None else self._default_export_dir() / "metadata.csv"
            path, _ = QFileDialog.getSaveFileName(
                self, "Export metadata as CSV", str(start), "CSV files (*.csv);;All files (*)")
            if not path:
                return None
            dest = Path(path)
            if not dest.suffix:
                dest = dest.with_suffix(".csv")
        try:
            write_csv([(it.path, it.meta) for it in done], dest)
        except Exception as exc:
            self.log.appendPlainText(f"✗ CSV export to {dest} failed: {exc}")
            QMessageBox.warning(self, "Export failed", f"Could not save the CSV:\n{dest}\n\n{exc}")
            return None
        self.csv_path = dest
        self._update_export_location()
        missing = len(self.items) - len(done)
        untitled = sum(1 for it in done if not it.meta.title.strip())
        trimmed = sum(1 for it in done if len(sanitize_keywords(
            it.meta.keywords, len(it.meta.keywords))) > ADOBE_KEYWORDS_MAX)
        text = f"Saved {len(done)} row(s) to {dest}."
        if missing:
            text += f" {missing} image(s) without metadata were left out."
        if untitled:
            text += f" ⚠ {untitled} row(s) have no title."
        if trimmed:
            text += (f" ⚠ {trimmed} row(s) had more than {ADOBE_KEYWORDS_MAX} keywords — only "
                     f"the first {ADOBE_KEYWORDS_MAX} were exported.")
        self.log.appendPlainText(f"✓ {text}")
        self.banner.show_message("warning" if missing or untitled or trimmed else "success", text)
        return dest

    def embed_into_images(self) -> None:
        """Write each finished image's metadata into the file (copies or in place)."""
        self._commit_editor()
        done = [it for it in self.items.values() if it.status == DONE]
        embeddable = [it for it in done if it.path.suffix.lower() in EMBEDDABLE_SUFFIXES]
        if not embeddable:
            QMessageBox.information(
                self, "Nothing to embed",
                "Embedding needs PNG or JPEG images that already have metadata.")
            return
        box = QMessageBox(self)
        box.setWindowTitle("Embed metadata into images")
        box.setText(f"Write title, description and keywords into {len(embeddable)} PNG/JPEG file(s)?")
        box.setInformativeText("Copies keep your originals untouched. Overwriting rewrites the "
                               "selected image files in place.")
        copies = box.addButton("Save Copies…", QMessageBox.AcceptRole)
        overwrite = box.addButton("Overwrite Originals", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is copies:
            folder = QFileDialog.getExistingDirectory(
                self, "Folder for the copies with metadata", str(self._default_export_dir()))
            if not folder:
                return
            target_dir: Path | None = Path(folder)
        elif box.clickedButton() is overwrite:
            target_dir = None
        else:
            return
        ok = bad = 0
        for it in embeddable:
            try:
                embed_metadata(it.path, it.meta, target_dir / it.path.name if target_dir else None)
                ok += 1
            except Exception as exc:
                bad += 1
                self.log.appendPlainText(f"✗ {it.path.name}: {exc}")
        where = "in place" if target_dir is None else f"as copies in {target_dir}"
        text = f"Embedded metadata in {ok} file(s) {where}."
        if bad:
            text += f" {bad} failed — see the activity log."
        if len(done) > len(embeddable):
            text += f" {len(done) - len(embeddable)} non-PNG/JPEG image(s) were skipped."
        self.log.appendPlainText(text)
        self.banner.show_message("warning" if bad else "success", text)

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
        if self.lookup and self.lookup.isRunning():
            # Destroying a running QThread is never safe, so ask the check to
            # stop and wait it out — every HTTP call has its own timeout, so
            # this can only last as long as one in-flight request.
            self.lookup.request_stop()
            if not self.lookup.wait(5000):
                self.lookup.wait()
        self.loader.shutdown()
