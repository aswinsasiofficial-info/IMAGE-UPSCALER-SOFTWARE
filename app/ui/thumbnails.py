"""Background thumbnail/preview loading for the desktop screens.

Images decode off the GUI thread with ``QImageReader``'s scaled decoding
(JPEGs decode straight at the small size), honour EXIF orientation, and are
requested only when a list tile is first painted — so a list of hundreds of
images opens instantly and only the visible ones cost anything. Results are
cached per (path, size).
"""

from __future__ import annotations

from collections import OrderedDict

from PySide6.QtCore import QObject, QRunnable, QSize, QThreadPool, Signal
from PySide6.QtGui import QImage, QImageReader, QPixmap


class _Job(QRunnable):
    def __init__(self, loader: "ImageLoader", key: str, path: str, edge: int) -> None:
        super().__init__()
        self.loader, self.key, self.path, self.edge = loader, key, path, edge

    def run(self) -> None:  # worker thread
        reader = QImageReader(self.path)
        reader.setAutoTransform(True)            # EXIF rotation, like the viewers
        size = reader.size()
        if size.isValid() and max(size.width(), size.height()) > self.edge:
            ratio = self.edge / max(size.width(), size.height())
            reader.setScaledSize(QSize(max(1, round(size.width() * ratio)),
                                       max(1, round(size.height() * ratio))))
        image = reader.read()
        self.loader.finished.emit(self.key, image, reader.errorString() if image.isNull() else "")


class ImageLoader(QObject):
    """Loads scaled images on a small thread pool; ``ready(key, pixmap)`` on the GUI thread."""

    finished = Signal(str, QImage, str)      # internal: worker -> GUI thread
    ready = Signal(str, QPixmap)
    failed = Signal(str, str)

    def __init__(self, parent: QObject | None = None, threads: int = 2, cache: int = 2000) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(threads)
        self._cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._cache_size = cache
        self._pending: set[str] = set()
        self._closed = False
        self.finished.connect(self._on_finished)

    @staticmethod
    def cache_key(path: str, edge: int) -> str:
        return f"{edge}|{path}"

    def cached(self, path: str, edge: int) -> QPixmap | None:
        key = self.cache_key(path, edge)
        pix = self._cache.get(key)
        if pix is not None:
            self._cache.move_to_end(key)
        return pix

    def request(self, path: str, edge: int) -> QPixmap | None:
        """Cached pixmap now, or None and a ``ready`` signal later."""
        pix = self.cached(path, edge)
        if pix is not None or self._closed:
            return pix
        key = self.cache_key(path, edge)
        if key not in self._pending:
            self._pending.add(key)
            self._pool.start(_Job(self, key, path, edge))
        return None

    def forget(self, path: str) -> None:
        for key in [k for k in self._cache if k.split("|", 1)[1] == path]:
            del self._cache[key]

    def _on_finished(self, key: str, image: QImage, error: str) -> None:
        self._pending.discard(key)
        if self._closed:
            return
        if image.isNull():
            self.failed.emit(key, error)
            return
        pix = QPixmap.fromImage(image)            # QPixmap only on the GUI thread
        self._cache[key] = pix
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        self.ready.emit(key, pix)

    def shutdown(self, timeout_ms: int = 3000) -> None:
        self._closed = True
        self._pool.clear()
        self._pool.waitForDone(timeout_ms)
