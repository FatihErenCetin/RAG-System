"""RAG Pipeline orchestrator.

Ingestion: loader → chunker → embedder → store.
Retrieval+Generation: embedder → store.search → llm.
"""
from __future__ import annotations

from typing import Callable

from src.core.interfaces import (
    Answer,
    Chunker,
    Document,
    DocumentLoader,
    EmbeddingProvider,
    LLMProvider,
    RetrievedChunk,
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

    # ---- retrieval + generation ----

    def answer(
        self,
        question: str,
        document_ids: list[str] | None = None,
        top_k: int = 4,
    ) -> Answer:
        """Soru → retrieve → LLM → Answer."""
        query_vec = self._embedder.embed([question])[0]
        retrieved = self._store.search(
            query_embedding=query_vec,
            top_k=top_k,
            document_ids=document_ids,
        )
        generated_text = self._llm.generate(question, context=retrieved)
        return Answer(
            text=generated_text,
            sources=retrieved,
            model=self._llm.model_name,
        )

    def summarize(self, document_id: str) -> Answer:
        """Bir dokümanın tüm chunk'larını LLM'e vererek özet üret."""
        from src.core.prompts import build_summarization_prompt

        chunks = self._store.get_document_chunks(document_id)
        if not chunks:
            raise ValueError(f"Document not found or has no chunks: {document_id}")

        retrieved = [RetrievedChunk(chunk=c, score=1.0) for c in chunks]
        document_name = chunks[0].document_name
        full_content = "\n\n".join(c.content for c in chunks)

        prompt = build_summarization_prompt(document_name=document_name, content=full_content)

        generated = self._llm.generate(prompt, context=retrieved)
        return Answer(
            text=generated,
            sources=retrieved,
            model=self._llm.model_name,
        )
