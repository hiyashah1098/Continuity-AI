"""Extract readable text from supported upload formats."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError

from .models import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}


class DocumentParseError(ValueError):
    """Raised when an uploaded file cannot provide usable text."""


def parse_uploaded_file(filename: str, data: bytes) -> Document:
    """Parse one supported upload and preserve its filename for citations."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError(
            f"Unsupported file type for '{filename}'. Use PDF, DOCX, TXT, or CSV."
        )
    try:
        if suffix == ".pdf":
            text = _parse_pdf(data)
        elif suffix == ".docx":
            text = _parse_docx(data)
        elif suffix == ".csv":
            text = _parse_csv(data)
        else:
            text = data.decode("utf-8-sig")
    except (
        UnicodeDecodeError,
        OSError,
        ValueError,
        RuntimeError,
        PdfReadError,
        PdfStreamError,
    ) as exc:
        raise DocumentParseError(f"Could not read '{filename}': {exc}") from exc

    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        raise DocumentParseError(f"'{filename}' is empty or contains no readable text.")
    return Document(filename=filename, text=cleaned)


def _parse_pdf(data: bytes) -> str:
    if not data.lstrip().startswith(b"%PDF-"):
        raise ValueError("PDF data does not contain a valid PDF header.")
    pages = PdfReader(io.BytesIO(data)).pages
    return "\n".join(page.extract_text() or "" for page in pages)


def _parse_docx(data: bytes) -> str:
    document = DocxDocument(io.BytesIO(data))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        paragraphs.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
    return "\n".join(paragraphs)


def _parse_csv(data: bytes) -> str:
    rows = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
