"""TXT loader — UTF-8 öncelikli, cp1254 fallback."""
from __future__ import annotations

import uuid

from src.core.interfaces import Document


class TxtLoader:
    """`.txt` dosyalarını metne dönüştürür."""

    def load(self, filename: str, content: bytes) -> Document:
        text = self._decode(content)
        return Document(
            id=str(uuid.uuid4()),
            name=filename,
            content=text,
            mime_type="text/plain",
            metadata={"size_bytes": len(content)},
        )

    @staticmethod
    def _decode(content: bytes) -> str:
        for encoding in ("utf-8", "cp1254", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="replace")
