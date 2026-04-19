"""Gemini embedding adapter testleri (mocked)."""
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.embeddings.gemini import GeminiEmbedding


@pytest.fixture
def mock_genai():
    with patch("src.adapters.embeddings.gemini.genai") as m:
        yield m


def test_gemini_embedding_batches_texts(mock_genai):
    mock_genai.embed_content.return_value = {
        "embedding": [0.1] * 768,
    }

    embedder = GeminiEmbedding(api_key="fake-key", model="text-embedding-004")
    vectors = embedder.embed(["metin 1", "metin 2"])

    assert len(vectors) == 2
    assert all(len(v) == 768 for v in vectors)
    assert mock_genai.embed_content.call_count == 2


def test_gemini_embedding_dimension_property():
    embedder = GeminiEmbedding(api_key="fake-key", model="text-embedding-004")
    # text-embedding-004 → 768
    assert embedder.dimension == 768


def test_gemini_embedding_empty_input(mock_genai):
    embedder = GeminiEmbedding(api_key="fake-key", model="text-embedding-004")
    assert embedder.embed([]) == []
    mock_genai.embed_content.assert_not_called()
