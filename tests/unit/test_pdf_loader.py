"""PDF loader testleri."""
from pathlib import Path

from src.adapters.loaders.pdf_loader import PdfLoader


def test_pdf_loader_extracts_text(fixtures_dir: Path):
    loader = PdfLoader()
    path = fixtures_dir / "sample.pdf"
    content_bytes = path.read_bytes()

    doc = loader.load("sample.pdf", content_bytes)

    assert doc.name == "sample.pdf"
    assert doc.mime_type == "application/pdf"
    # İçerikten en az bir kelime olmalı
    assert len(doc.content.strip()) > 0


def test_pdf_loader_concatenates_pages(fixtures_dir: Path):
    loader = PdfLoader()
    content_bytes = (fixtures_dir / "sample.pdf").read_bytes()
    doc = loader.load("sample.pdf", content_bytes)

    # İki farklı sayfanın içeriği birleşmiş olmalı
    assert "Birinci" in doc.content or "sayfa" in doc.content.lower() or "page" in doc.content.lower()
    assert "English" in doc.content or "Second" in doc.content
    assert doc.metadata.get("page_count", 0) >= 2


def test_pdf_loader_corrupt_bytes_raises():
    import pytest

    loader = PdfLoader()
    with pytest.raises(ValueError, match="parse"):
        loader.load("broken.pdf", b"not a real pdf")
