"""DOCX loader testleri."""
from pathlib import Path

import pytest

from src.adapters.loaders.docx_loader import DocxLoader


def test_docx_loader_extracts_text(fixtures_dir: Path):
    loader = DocxLoader()
    content_bytes = (fixtures_dir / "sample.docx").read_bytes()
    doc = loader.load("sample.docx", content_bytes)

    assert doc.name == "sample.docx"
    assert doc.mime_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert "Test" in doc.content or "Dokumani" in doc.content
    assert "Birinci" in doc.content
    assert "English" in doc.content


def test_docx_loader_corrupt_bytes_raises():
    loader = DocxLoader()
    with pytest.raises(ValueError, match="parse"):
        loader.load("bad.docx", b"not a real docx")
