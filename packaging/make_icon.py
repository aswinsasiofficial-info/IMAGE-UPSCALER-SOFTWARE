"""Render app/assets/app.svg into the multi-size Windows icon app/assets/app.ico.

Run with:  python packaging/make_icon.py   (only needed after editing app.svg)
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ASSETS = Path(__file__).resolve().parents[1] / "app" / "assets"
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def render(svg: bytes, size: int) -> Image.Image:
    image = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    QSvgRenderer(QByteArray(svg)).render(painter)
    painter.end()
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, "PNG")
    return Image.open(io.BytesIO(bytes(buffer.data()))).convert("RGBA")


def main() -> int:
    QGuiApplication(sys.argv)
    svg = (ASSETS / "app.svg").read_bytes()
    images = [render(svg, s) for s in SIZES]
    images[-1].save(ASSETS / "app.ico", sizes=[(s, s) for s in SIZES],
                    append_images=images[:-1])
    images[-1].save(ASSETS / "app.png")
    print(f"wrote {ASSETS / 'app.ico'} and app.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
