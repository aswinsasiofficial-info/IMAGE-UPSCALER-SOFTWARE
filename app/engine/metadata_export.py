"""Export paths for the Metadata Generator: contributor CSV and in-file embedding.

Everything works on the reviewable rows built by the UI — nothing here is
connected to the upscaler pipeline.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from .metadata_settings import ImageMetadata

# Adobe Stock contributor CSV template (Description added for generality —
# stock sites ignore unknown columns they don't use).
CSV_HEADERS = ["Filename", "Title", "Description", "Keywords"]

# File formats we can safely rewrite with embedded metadata.
EMBEDDABLE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def write_csv(rows: list[tuple[Path, ImageMetadata]], dest: Path) -> Path:
    """Write all reviewed rows to one contributor-friendly CSV (UTF-8 with BOM).

    ``rows`` is a list of (image_path, metadata) pairs; keyword lists are
    joined with commas as stock sites expect.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADERS)
        for path, meta in rows:
            writer.writerow([
                path.name,
                meta.title,
                meta.description,
                ", ".join(meta.keywords),
            ])
    return dest


def _metadata_payload(meta: ImageMetadata) -> str:
    return json.dumps(
        {"title": meta.title, "description": meta.description,
         "keywords": meta.keywords},
        ensure_ascii=False)


def embed_metadata(src: Path, meta: ImageMetadata, dest: Path | None = None) -> Path:
    """Write title/description/keywords into the image file itself.

    - PNG: standard tEXt chunks (Title / Description / Keywords).
    - JPEG: a COM comment segment holding a JSON payload (quality preserved).

    ``dest`` defaults to overwriting ``src`` in place (written to a temp file
    first, then swapped, so a crash can never leave a half-written image).
    Raises ``ValueError`` for formats we cannot safely rewrite.
    """
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    src = Path(src)
    dest = Path(dest) if dest is not None else src
    suffix = src.suffix.lower()
    if suffix not in EMBEDDABLE_SUFFIXES:
        raise ValueError(
            f"Embedding supports PNG and JPEG only — skipped '{src.name}' ({suffix or 'no extension'}).")

    with Image.open(src) as im:
        im.load()
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite the source file before the new one is complete.
        fd, tmp_name = tempfile.mkstemp(suffix=suffix, dir=str(dest.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            if suffix == ".png":
                info = PngInfo()
                info.add_text("Title", meta.title)
                info.add_text("Description", meta.description)
                info.add_text("Keywords", ", ".join(meta.keywords))
                im.save(tmp, "PNG", pnginfo=info)
            else:
                # quality="keep" reuses the original quantisation tables, so
                # the picture itself is not degraded by the metadata rewrite.
                im.save(tmp, "JPEG", quality="keep", comment=_metadata_payload(meta))
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    return dest


def read_embedded(path: Path) -> dict:
    """Best-effort round-trip reader used by tests and the review dialog."""
    from PIL import Image

    with Image.open(path) as im:
        suffix = Path(path).suffix.lower()
        if suffix == ".png":
            info = im.info or {}
            return {
                "title": info.get("Title", ""),
                "description": info.get("Description", ""),
                "keywords": [k.strip() for k in info.get("Keywords", "").split(",") if k.strip()],
            }
        if suffix in (".jpg", ".jpeg"):
            raw = (im.info or {}).get("comment", b"")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except Exception:
                return {"title": "", "description": "", "keywords": []}
    return {}
