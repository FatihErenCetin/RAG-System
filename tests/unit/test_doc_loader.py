"""DOC loader testleri."""
import pytest

from src.adapters.loaders.doc_loader import DocLoader, DocSupportUnavailable


def test_doc_loader_returns_clear_error_without_antiword(monkeypatch):
    """textract/antiword yoksa DocSupportUnavailable atılır."""
    loader = DocLoader()

    # textract'i yok gibi davran
    def _fail(*args, **kwargs):
        raise OSError("antiword binary not found")

    monkeypatch.setattr("src.adapters.loaders.doc_loader._extract_with_textract", _fail)

    with pytest.raises(DocSupportUnavailable) as exc_info:
        loader.load("eski.doc", b"\xd0\xcf\x11\xe0some old doc binary")

    # Mesaj kullanıcıya kurulum ipucu vermeli
    assert "antiword" in str(exc_info.value) or ".docx" in str(exc_info.value)


def test_doc_loader_returns_document_when_antiword_works(monkeypatch):
    """textract başarılı olursa Document döndür."""
    loader = DocLoader()

    def _ok(content: bytes) -> str:
        return "parsed content from .doc"

    monkeypatch.setattr("src.adapters.loaders.doc_loader._extract_with_textract", _ok)

    doc = loader.load("eski.doc", b"whatever")
    assert doc.content == "parsed content from .doc"
    assert doc.mime_type == "application/msword"
