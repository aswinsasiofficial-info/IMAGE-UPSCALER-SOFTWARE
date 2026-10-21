"""Final review step before exporting generated metadata.

Shows every row (filename, title, description, keywords, status) in a
read-only table with validation warnings, and collects the user's explicit
export choices: contributor CSV and/or embedding into image files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.engine.metadata_export import EMBEDDABLE_SUFFIXES


@dataclass
class ExportOptions:
    csv_path: Path | None = None
    embed: bool = False
    embed_into_copies_dir: Path | None = None   # None = in place (overwrite)
    embeddable_count: int = 0


class ReviewDialog(QDialog):
    """Read-only overview of all generated rows + explicit export controls."""

    def __init__(self, rows: list[tuple[Path, str, dict]], parent: QWidget | None = None) -> None:
        """``rows``: list of (image path, status label, metadata dict)."""
        super().__init__(parent)
        self.setWindowTitle("Review metadata before export")
        self.resize(980, 560)
        self._rows = rows
        self._result = ExportOptions()
        self._build(rows)

    def _build(self, rows: list[tuple[Path, str, dict]]) -> None:
        lay = QVBoxLayout(self)

        # -- summary + warnings ------------------------------------------------
        done = [r for r in rows if r[1] == "done"]
        missing_title = sum(1 for r in done if not r[2].get("title", "").strip())
        embeddable = sum(1 for r in done
                         if Path(r[0]).suffix.lower() in EMBEDDABLE_SUFFIXES)
        parts = [f"{len(done)} generated / {len(rows)} queued"]
        if missing_title:
            parts.append(f"⚠ {missing_title} row(s) have no title")
        if len(done) - embeddable > 0:
            parts.append(f"{len(done) - embeddable} row(s) are not PNG/JPEG (cannot embed)")
        summary = QLabel(" · ".join(parts))
        summary.setWordWrap(True)
        lay.addWidget(summary)

        # -- read-only review table -------------------------------------------
        table = QTableWidget(len(rows), 5)
        table.setHorizontalHeaderLabels(
            ["Filename", "Status", "Title", "Description", "Keywords"])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        for i, (path, status, meta) in enumerate(rows):
            values = [path.name, status, meta.get("title", ""),
                      meta.get("description", ""),
                      ", ".join(meta.get("keywords", []))]
            for col, val in enumerate(values):
                item = QTableWidgetItem(str(val))
                if col == 1:
                    if status == "failed":
                        item.setForeground(QBrush(QColor(Qt.red)))
                    elif status != "done":
                        item.setForeground(QBrush(QColor(Qt.gray)))
                table.setItem(i, col, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(table, stretch=1)

        # -- export options ------------------------------------------------------
        box = QGroupBox("Export")
        box_lay = QVBoxLayout(box)

        self.chk_csv = QCheckBox("Contributor CSV (Filename, Title, Description, Keywords)")
        self.chk_csv.setChecked(True)
        csv_row = QHBoxLayout()
        self.csv_edit = QLineEdit()
        self.csv_edit.setPlaceholderText("metadata.csv — choose where to save it")
        self.csv_edit.setEnabled(False)
        btn_csv = QPushButton("Browse…")
        btn_csv.setEnabled(False)
        btn_csv.clicked.connect(self._browse_csv)
        self.chk_csv.toggled.connect(lambda v: (self.csv_edit.setEnabled(v),
                                                btn_csv.setEnabled(v)))
        csv_row.addWidget(self.chk_csv, 1)
        csv_row.addWidget(self.csv_edit, 2)
        csv_row.addWidget(btn_csv)
        box_lay.addLayout(csv_row)

        self.chk_embed = QCheckBox(
            "Embed metadata into image files (PNG tEXt chunks / JPEG comment)")
        embed_row = QHBoxLayout()
        self.radio_copy = QRadioButton("Into copies in folder:")
        self.radio_copy.setChecked(True)
        self.embed_dir_edit = QLineEdit()
        self.embed_dir_edit.setPlaceholderText("choose a destination folder")
        self.embed_dir_edit.setEnabled(False)
        btn_dir = QPushButton("Browse…")
        btn_dir.setEnabled(False)
        btn_dir.clicked.connect(self._browse_embed_dir)
        self.radio_inplace = QRadioButton(
            "In place (rewrites the selected image files)")
        self.chk_embed.toggled.connect(lambda v: (
            self.radio_copy.setEnabled(v), self.radio_inplace.setEnabled(v),
            self.embed_dir_edit.setEnabled(v and self.radio_copy.isChecked()),
            btn_dir.setEnabled(v and self.radio_copy.isChecked())))
        self.radio_copy.toggled.connect(lambda v: (
            self.embed_dir_edit.setEnabled(v and self.chk_embed.isChecked()),
            btn_dir.setEnabled(v and self.chk_embed.isChecked())))
        embed_row.addWidget(self.chk_embed, 2)
        embed_row.addWidget(self.radio_copy)
        embed_row.addWidget(self.embed_dir_edit, 2)
        embed_row.addWidget(btn_dir)
        embed_row.addWidget(self.radio_inplace)
        box_lay.addLayout(embed_row)

        embed_note = QLabel(
            f"Embedding supports {', '.join(sorted(s.upper().lstrip('.') for s in EMBEDDABLE_SUFFIXES))} only. "
            "The picture data is left untouched as much as the format allows.")
        embed_note.setWordWrap(True)
        box_lay.addWidget(embed_note)
        lay.addWidget(box)

        self.embeddable_count = embeddable

        # -- accept / reject -----------------------------------------------------
        btns = QHBoxLayout()
        btns.addStretch(1)
        btn_cancel = QPushButton("Back to Editing")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Export")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self._accept)
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_ok)
        lay.addLayout(btns)

    # -- callbacks ---------------------------------------------------------------
    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save contributor CSV", str(Path.home() / "metadata.csv"),
            "CSV files (*.csv);;All files (*)")
        if path:
            self.csv_edit.setText(path)

    def _browse_embed_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Folder for embedded copies", str(Path.home()))
        if folder:
            self.embed_dir_edit.setText(folder)
            self.radio_copy.setChecked(True)

    def _accept(self) -> None:
        opts = ExportOptions()
        if self.chk_csv.isChecked():
            csv_path = self.csv_edit.text().strip()
            if not csv_path:
                # Default to metadata.csv beside the first queued image.
                first = self._rows[0][0].parent if self._rows else Path.home()
                csv_path = str(first / "metadata.csv")
            opts.csv_path = Path(csv_path)
        if self.chk_embed.isChecked():
            opts.embed = True
            if self.radio_copy.isChecked():
                folder = self.embed_dir_edit.text().strip()
                if folder:
                    opts.embed_into_copies_dir = Path(folder)
                else:
                    opts.embed_into_copies_dir = Path(
                        (self._rows[0][0].parent if self._rows else Path.home()) / "embedded")
        self._result = opts
        self.accept()

    def export_options(self) -> ExportOptions:
        return self._result
