"""RAG Pipeline orchestrator.

Ingestion: loader → chunker → embedder → store.
Retrieval+Generation: embedder → store.search → llm.
"""
from __future__ import annotations

from typing import Callable

from src.core.interfaces import (
    Chunker,
    Document,
    DocumentLoader,
    EmbeddingProvider,
    LLMProvider,
    VectorStore,
)


LoaderFactory = Callable[[str | None, str | None], DocumentLoader]
"""MIME type ve/veya filename'den DocumentLoader döndüren fonksiyon."""


class RAGPipeline:
    """Tüm RAG akışını koordine eden orchestrator."""

    def __init__(
        self,
        loader_factory: LoaderFactory,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LLMProvider,
    ):
        self._loader_factory = loader_factory
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._llm = llm

    # ---- ingestion ----

    def ingest(
        self,
        filename: str,
        content: bytes,
        mime_type: str | None = None,
    ) -> Document:
        """Dosyayı parse et, chunk'la, embed'le, store'a yaz."""
        loader = self._loader_factory(mime_type, filename)
        document = loader.load(filename, content)

        chunks = self._chunker.chunk(document)
        if not chunks:
            return document  # boş doküman; no-op

        embeddings = self._embedder.embed([c.content for c in chunks])
        self._store.add(chunks, embeddings)

        return document
