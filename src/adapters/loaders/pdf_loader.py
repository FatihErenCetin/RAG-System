"""PDF loader — pypdf ile doğrudan."""
from __future__ import annotations

import io
import uuid

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.core.interfaces import Document


def _fix_spaced_chars(text: str) -> str:
    """Bazı PDF'lerde her karakter ayrı ayrı boşlukla yazılır: 'P R O J E' → 'PROJE'.

    Satır başına bakar: token'ların %60'ından fazlası tek-harf alfabetik ise
    ve satırda en az 6 token varsa, ardışık tek-harf token'larını birleştirir.
    Noktalama işaretleri ayrı tutulur.
    """
    return "\n".join(_fix_line(line) for line in text.split("\n"))


def _fix_line(line: str) -> str:
    tokens = [t for t in line.split(" ") if t]
    if not tokens:
        return line

    single_alpha = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    if len(tokens) < 6 or single_alpha / len(tokens) < 0.6:
        return line

    # Ardışık tek-harf alfabetik token'ları birleştir; diğerlerini koru
    parts: list[str] = []
    run: list[str] = []
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            run.append(tok)
        else:
            if run:
                parts.append("".join(run))
                run = []
            parts.append(tok)
    if run:
        parts.append("".join(run))

    return " ".join(parts)


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
        full_text = _fix_spaced_chars(full_text)

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
