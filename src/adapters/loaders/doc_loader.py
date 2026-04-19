"""DOC (eski Word binary) loader.

textract paketi arka planda `antiword` sistem binary'sini kullanır.
Bulunmazsa DocSupportUnavailable hatası verir ve kullanıcıya kurulum ipucu sunar.
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from src.core.interfaces import Document


class DocSupportUnavailable(RuntimeError):
    """`.doc` parse edilemedi — antiword/textract eksik."""


def _extract_with_textract(content: bytes) -> str:
    """textract ile geçici dosyaya yazıp parse et.

    ImportError veya OSError durumunda caller yakalar.
    """
    import textract  # noqa: F401 — optional dep, may be commented out in requirements

    with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        raw = textract.process(tmp_path)
        return raw.decode("utf-8", errors="replace")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class DocLoader:
    """`.doc` dosyaları için parser; sistem antiword kuruluysa çalışır."""

    DOC_MIME = "application/msword"

    def load(self, filename: str, content: bytes) -> Document:
        try:
            text = _extract_with_textract(content)
        except ImportError as e:
            raise DocSupportUnavailable(
                "'.doc' desteği için `textract` paketi gerekli. "
                "`pip install textract` komutunu çalıştırın veya dosyayı `.docx` formatına çevirin."
            ) from e
        except (OSError, Exception) as e:  # textract bazen generic hata atıyor
            raise DocSupportUnavailable(
                f"'.doc' parse edilemedi ({e}). "
                "Sistem `antiword` kurulu olmalı: `brew install antiword` (macOS) "
                "veya `apt install antiword` (Linux). "
                "Alternatif olarak dosyayı `.docx` formatına çevirin."
            ) from e

        return Document(
            id=str(uuid.uuid4()),
            name=filename,
            content=text,
            mime_type=self.DOC_MIME,
            metadata={"size_bytes": len(content)},
        )
