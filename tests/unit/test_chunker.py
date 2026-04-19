"""Recursive character chunker testleri."""
import pytest

from src.adapters.chunkers.recursive import RecursiveChunker
from src.core.interfaces import Document


def _make_doc(text: str) -> Document:
    return Document(
        id="doc-1",
        name="test.txt",
        content=text,
        mime_type="text/plain",
        metadata={},
    )


def test_chunker_splits_long_text():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    long_text = "a" * 200  # 200 karakter
    chunks = chunker.chunk(_make_doc(long_text))
    assert len(chunks) >= 3, "Long text should produce multiple chunks"
    for c in chunks:
        assert len(c.content) <= 50


def test_chunker_preserves_order_and_indices():
    chunker = RecursiveChunker(chunk_size=30, chunk_overlap=5)
    text = "paragraf 1.\n\nparagraf 2.\n\nparagraf 3."
    chunks = chunker.chunk(_make_doc(text))
    for i, c in enumerate(chunks):
        assert c.index == i
        assert c.document_id == "doc-1"
        assert c.document_name == "test.txt"


def test_chunker_empty_document():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(_make_doc(""))
    assert chunks == []


def test_chunker_single_short_document():
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.chunk(_make_doc("kısa metin"))
    assert len(chunks) == 1
    assert chunks[0].content == "kısa metin"


def test_chunker_adds_char_metadata():
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
    text = "abcdefghij" * 5  # 50 karakter
    chunks = chunker.chunk(_make_doc(text))
    for c in chunks:
        assert "char_start" in c.metadata
        assert "char_end" in c.metadata
        assert c.metadata["char_end"] > c.metadata["char_start"]
