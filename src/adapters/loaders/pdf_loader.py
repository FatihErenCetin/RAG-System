"""PDF loader — pypdf ile doğrudan."""
from __future__ import annotations

import io
import uuid

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.core.interfaces import Document


class PdfLoader:
    """`.pdf` dosyalarını metne dönüştürür (pypdf ile)."""

    def load(self, filename: str, content: bytes) -> Document:
        try:
            reader = PdfReader(io.BytesIO(content))
        except PdfReadError as e:
            raise ValueError(f"PDF parse failed for {filename}: {e}") from e
        except Exception as e:  # pypdf bazen generic Exception
            raise ValueError(f"PDF parse failed for {filename}: {e}") from e

        pages_text: list[str] = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                pages_text.append("")

        full_text = "\n\n".join(p for p in pages_text if p.strip())

        return Document(
            id=str(uuid.uuid4()),
            name=filename,
            content=full_text,
            mime_type="application/pdf",
            metadata={
                "size_bytes": len(content),
                "page_count": len(reader.pages),
            },
        )
