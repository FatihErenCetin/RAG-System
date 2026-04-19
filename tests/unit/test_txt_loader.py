"""TXT loader testleri."""
from pathlib import Path

import pytest

from src.adapters.loaders.txt_loader import TxtLoader


def test_txt_loader_reads_utf8(fixtures_dir: Path):
    loader = TxtLoader()
    path = fixtures_dir / "sample.txt"
    content_bytes = path.read_bytes()

    doc = loader.load("sample.txt", content_bytes)

    assert doc.name == "sample.txt"
    assert "ğüşiöçİĞÜŞÖÇ" in doc.content
    assert "quick brown fox" in doc.content
    assert doc.mime_type == "text/plain"
    assert doc.id  # non-empty UUID
    assert doc.metadata["size_bytes"] == len(content_bytes)


def test_txt_loader_empty_file():
    loader = TxtLoader()
    doc = loader.load("empty.txt", b"")
    assert doc.content == ""
    assert doc.metadata["size_bytes"] == 0


def test_txt_loader_handles_cp1254_fallback():
    """Latin-5 encoded Türkçe content (UTF-8 decode fail'e uğrarsa cp1254 dene)."""
    loader = TxtLoader()
    # "özel" cp1254'te
    content = "özel".encode("cp1254")
    doc = loader.load("tr.txt", content)
    assert "özel" in doc.content
