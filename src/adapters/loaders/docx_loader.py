"""DOCX loader — docx2txt (basit, BytesIO destekli)."""
from __future__ import annotations

import io
import uuid

import docx2txt

from src.core.interfaces import Document


class DocxLoader:
    """`.docx` dosyalarını metne dönüştürür."""

    DOCX_MIME = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    def load(self, filename: str, content: bytes) -> Document:
        try:
            text = docx2txt.process(io.BytesIO(content))
        except Exception as e:
            raise ValueError(f"DOCX parse failed for {filename}: {e}") from e

        return Document(
            id=str(uuid.uuid4()),
            name=filename,
            content=text or "",
            mime_type=self.DOCX_MIME,
            metadata={"size_bytes": len(content)},
        )
