"""Document loader factory.

MIME type (veya fallback: dosya uzantısı) bakarak uygun DocumentLoader'ı döndürür.
"""
from __future__ import annotations

from pathlib import Path

from src.core.interfaces import DocumentLoader

from .doc_loader import DocLoader
from .docx_loader import DocxLoader
from .pdf_loader import PdfLoader
from .txt_loader import TxtLoader


class UnsupportedFileType(ValueError):
    """Yüklenen dosya formatı desteklenmiyor."""


_MIME_TO_LOADER: dict[str, type[DocumentLoader]] = {
    "text/plain": TxtLoader,
    "application/pdf": PdfLoader,
    "application/msword": DocLoader,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxLoader,
}

_EXT_TO_LOADER: dict[str, type[DocumentLoader]] = {
    ".txt": TxtLoader,
    ".pdf": PdfLoader,
    ".doc": DocLoader,
    ".docx": DocxLoader,
}


def get_loader(
    mime_type: str | None,
    filename: str | None = None,
) -> DocumentLoader:
    """MIME type veya dosya uzantısına göre loader döndür."""
    if mime_type and mime_type in _MIME_TO_LOADER:
        return _MIME_TO_LOADER[mime_type]()

    if filename:
        ext = Path(filename).suffix.lower()
        if ext in _EXT_TO_LOADER:
            return _EXT_TO_LOADER[ext]()

    raise UnsupportedFileType(
        f"Desteklenmeyen dosya formatı: mime={mime_type!r}, filename={filename!r}. "
        "Destek: .txt, .pdf, .doc, .docx"
    )


__all__ = ["get_loader", "UnsupportedFileType"]
