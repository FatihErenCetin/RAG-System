"""Pytest shared fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.fakes import (
    FakeChunker,
    FakeDocumentLoader,
    FakeEmbeddingProvider,
    FakeLLMProvider,
    FakeVectorStore,
)


@pytest.fixture
def fake_embedder():
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_store():
    return FakeVectorStore()


@pytest.fixture
def fake_llm():
    return FakeLLMProvider()


@pytest.fixture
def fake_chunker():
    return FakeChunker(chunk_size=50)


@pytest.fixture
def fake_loader():
    return FakeDocumentLoader()


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
