"""Choose which images of a large folder go into a Metadata folder batch.

A folder batch holds at most 25 images. When a folder has more, this dialog
shows them all as thumbnail tiles with tick boxes (the first ones ticked) and
lets the user pick up to the limit; it won't accept more.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStyledItemDelegate,
    QVBoxLayout,
)

from app import design
from app.ui import theme
from app.ui.thumbnails import ImageLoader
from app.ui.widgets import label, paint_thumbnail, paint_tile_background

THUMB = 128          # decoded size (shared cache with the Metadata list)


class _TileDelegate(QStyledItemDelegate):
    """Fixed tile: whole thumbnail, tick box in the corner, filename below."""

    SIZE = QSize(136, 152)
    BOX = 108

    def __init__(self, dialog: "FolderPickDialog") -> None:
        super().__init__(dialog)
        self.dialog = dialog

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        return self.SIZE

    def paint(self, painter: QPainter, option, index) -> None:
        c = design.colors(theme.current_scheme())
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        rect = option.rect.adjusted(3, 3, -3, -3)
        paint_tile_background(painter, rect, option, c)
        state = index.data(Qt.CheckStateRole)    # PySide6 hands back a plain int here
        checked = state in (Qt.Checked, Qt.Checked.value)
        if checked:
            outline = QPainterPath()
            outline.addRoundedRect(rect.adjusted(1, 1, -1, -1), design.RADIUS_SM, design.RADIUS_SM)
            painter.setPen(QColor(c["accent"]))
            painter.drawPath(outline)
        box = QRect(rect.center().x() - self.BOX // 2, rect.top() + 6, self.BOX, self.BOX)
        path = index.data(Qt.UserRole)
        paint_thumbnail(painter, box, self.dialog.loader.request(path, THUMB), c)

        tick = QRect(box.left() + 5, box.top() + 5, 20, 20)
        mark = QPainterPath()
        mark.addRoundedRect(tick, 4, 4)
        painter.fillPath(mark, QColor(c["primary"] if checked else c["surface"]))
        painter.setPen(QColor(c["primary"] if checked else c["border-strong"]))
        painter.drawPath(mark)
        if checked:
            check = QPainterPath()
            check.moveTo(tick.left() + 5, tick.top() + 10.5)
            check.lineTo(tick.left() + 8.5, tick.top() + 14)
            check.lineTo(tick.left() + 15, tick.top() + 6.5)
            pen = QPen(QColor(c["primary-text"]), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(check)

        fm = QFontMetrics(option.font)
        text_rect = QRect(rect.left() + 4, box.bottom() + 6, rect.width() - 8, fm.height())
        painter.setPen(QColor(c["text"] if checked else c["text-muted"]))
        painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignVCenter,
                         fm.elidedText(Path(path).name, Qt.ElideMiddle, text_rect.width()))
        painter.restore()


class FolderPickDialog(QDialog):
    def __init__(self, folder: Path, images: list[Path], limit: int, loader: ImageLoader,
                 already: int = 0, parent=None) -> None:
        super().__init__(parent)
        self.limit = limit
        self.loader = loader
        self.setWindowTitle(f"Choose up to {limit} images")
        self.setAccessibleName(self.windowTitle())
        self.resize(860, 620)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        intro = (f"“{folder.name}” has {len(images) + already} supported images. A folder batch "
                 f"holds up to 25 images")
        if already:
            intro += f", and {already} from this folder are already in the list"
        intro += (f" — tick the {limit} you want (the first {limit} are ticked). The folder's "
                  "CSV is saved automatically once all of them have metadata.")
        lay.addWidget(label(intro, wrap=True))

        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setGridSize(_TileDelegate.SIZE)
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setMovement(QListWidget.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.setMouseTracking(True)
        self.grid.setSelectionMode(QAbstractItemView.SingleSelection)
        self.grid.setItemDelegate(_TileDelegate(self))
        self.grid.setAccessibleName("Images in the folder")
        self.grid.setAccessibleDescription(f"Click or press Space to tick up to {limit} images")
        self._items: dict[str, QListWidgetItem] = {}
        for i, path in enumerate(images):
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            item.setData(Qt.UserRole, str(path))
            # Not "user checkable": the whole tile toggles (one click, or Space),
            # so a click on the box can't flip it twice.
            item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i < limit else Qt.Unchecked)
            self.grid.addItem(item)
            self._items[str(path)] = item
        loader.ready.connect(self._on_thumb)
        self.grid.itemChanged.connect(self._update)
        self.grid.itemClicked.connect(self.toggle)
        space = QShortcut(QKeySequence(Qt.Key_Space), self.grid)
        space.setContext(Qt.WidgetShortcut)
        space.activated.connect(lambda: self.toggle(self.grid.currentItem()))
        lay.addWidget(self.grid, 1)

        bottom = QHBoxLayout()
        self.counter = label("", "muted")
        bottom.addWidget(self.counter, 1)
        first = QPushButton(f"Select First {limit}")
        first.setToolTip(f"Tick the first {limit} images, untick the rest")
        first.clicked.connect(self.select_first)
        none = QPushButton("Clear")
        none.setToolTip("Untick every image")
        none.clicked.connect(lambda: self._set_all(0))
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.ok = QPushButton()
        self.ok.setProperty("variant", "primary")
        self.ok.setDefault(True)
        self.ok.clicked.connect(self.accept)
        for b in (first, none, cancel, self.ok):
            bottom.addWidget(b)
        lay.addLayout(bottom)
        self._update()

    def _on_thumb(self, cache_key: str, _pix: QPixmap) -> None:
        if cache_key.split("|", 1)[1] in self._items:
            self.grid.viewport().update()

    def toggle(self, item: QListWidgetItem | None) -> None:
        """Tick or untick a tile (click anywhere on it, or Space)."""
        if item is not None:
            item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)

    def _set_all(self, count: int) -> None:
        self.grid.blockSignals(True)
        for i in range(self.grid.count()):
            self.grid.item(i).setCheckState(Qt.Checked if i < count else Qt.Unchecked)
        self.grid.blockSignals(False)
        self.grid.viewport().update()
        self._update()

    def select_first(self) -> None:
        self._set_all(self.limit)

    def selected_paths(self) -> list[Path]:
        return [Path(self.grid.item(i).data(Qt.UserRole)) for i in range(self.grid.count())
                if self.grid.item(i).checkState() == Qt.Checked]

    def _update(self, *_args) -> None:
        n = len(self.selected_paths())
        over = n > self.limit
        self.counter.setText(f"{n} of {self.limit} selected"
                             + (f" — untick {n - self.limit} to continue" if over else ""))
        self.counter.setProperty("role", "error" if over else "muted")
        theme.repolish(self.counter)
        self.ok.setText(f"Add {n} Image{'s' if n != 1 else ''}")
        self.ok.setEnabled(0 < n <= self.limit)
