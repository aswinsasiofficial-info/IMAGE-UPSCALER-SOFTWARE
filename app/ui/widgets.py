"""Building blocks shared by the Upscaler and Metadata screens."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from app import design
from app.ui import theme
from app.ui.icons import icon_button


def card(*, padding: int = 12, spacing: int = 8) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setProperty("card", True)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(padding, padding, padding, padding)
    lay.setSpacing(spacing)
    return frame, lay


def label(text: str = "", role: str | None = None, wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    if role:
        lbl.setProperty("role", role)
    lbl.setWordWrap(wrap)
    return lbl


class ElidedLabel(QLabel):
    """One-line label that shortens long text with … instead of being clipped."""

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, _event) -> None:  # noqa: N802
        if self.width() < 90:          # a few letters of a path help nobody
            return
        p = QPainter(self)
        p.setPen(self.palette().color(self.foregroundRole()))
        text = self.fontMetrics().elidedText(self.text(), Qt.ElideMiddle, self.width())
        p.drawText(self.rect(), int(self.alignment()), text)


class Preview(QWidget):
    """Large preview of the selected image, scaled to fit (shape kept)."""

    def __init__(self, message: str = "Select an image to see it here.") -> None:
        super().__init__()
        self._pix: QPixmap | None = None
        self._message = message
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAccessibleName("Preview of the selected image")

    def sizeHint(self) -> QSize:  # noqa: N802
        # Modest preferred height: fields stay in view first; spare height goes here.
        return QSize(480, 170)

    def show_pixmap(self, pix: QPixmap | None, message: str = "") -> None:
        self._pix = pix
        if message:
            self._message = message
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        c = design.colors(theme.current_scheme())
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(0, 0, -1, -1), design.RADIUS_SM, design.RADIUS_SM)
        p.fillPath(path, QColor(c["surface-alt"]))
        if self._pix is not None and not self._pix.isNull():
            area = self.rect().adjusted(8, 8, -8, -8)
            size = self._pix.size().scaled(area.size(), Qt.KeepAspectRatio)
            target = QRect(0, 0, size.width(), size.height())
            target.moveCenter(area.center())
            p.drawPixmap(target, self._pix)
        else:
            p.setPen(QColor(c["text-muted"]))
            p.drawText(self.rect().adjusted(12, 12, -12, -12), Qt.AlignCenter | Qt.TextWordWrap,
                       self._message)


class Banner(QFrame):
    """Inline message strip for outcomes (CSV saved, batch failed, …)."""

    def __init__(self) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 6, 6)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        close = icon_button("x", "Dismiss this message", variant="ghost")
        close.clicked.connect(self.hide)
        lay.addWidget(self.text, 1)
        lay.addWidget(close, 0, Qt.AlignTop)
        self.setAccessibleName("Message")
        self.hide()

    def show_message(self, kind: str, text: str) -> None:
        self.setProperty("banner", kind)
        theme.repolish(self)
        self.text.setText(text)
        self.setAccessibleDescription(text)
        self.show()


# -- painting helpers for thumbnail list tiles -----------------------------------
def paint_tile_background(painter: QPainter, rect: QRect, option, c: dict) -> None:
    selected = bool(option.state & QStyle.State_Selected)
    hovered = bool(option.state & QStyle.State_MouseOver)
    if selected or hovered:
        bg = QPainterPath()
        bg.addRoundedRect(rect, design.RADIUS_SM, design.RADIUS_SM)
        painter.fillPath(bg, QColor(c["selection"] if selected else c["hover"]))


def paint_thumbnail(painter: QPainter, box: QRect, pix: QPixmap | None, c: dict) -> None:
    """Thumbnail letterboxed into *box*, so the whole picture shows."""
    frame = QPainterPath()
    frame.addRoundedRect(box, 4, 4)
    painter.fillPath(frame, QColor(c["surface-alt"]))
    if pix is not None:
        size = pix.size().scaled(box.size(), Qt.KeepAspectRatio)
        target = QRect(0, 0, size.width(), size.height())
        target.moveCenter(box.center())
        painter.setClipPath(frame)
        painter.drawPixmap(target, pix)
        painter.setClipping(False)
    painter.setPen(QColor(c["border"]))
    painter.drawPath(frame)
