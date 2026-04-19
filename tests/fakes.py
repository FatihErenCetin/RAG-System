"""Fake (test double) provider implementations.

Bu fake'ler test sırasında gerçek Gemini/Chroma çağırmadan RAGPipeline'ı
test etmemize olanak sağlar. Deterministiktirler; aynı girdi → aynı çıktı.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from src.core.interfaces import (
    Chunk,
    Document,
    EmbeddingProvider,
    LLMProvider,
    RetrievedChunk,
    VectorStore,
)


def _text_to_vector(text: str, dim: int = 8) -> list[float]:
    """Deterministic hash-based vector (test amaçlı)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # İlk dim bytei float [-1, 1]'e map et
    return [(b - 128) / 128.0 for b in h[:dim]]


class FakeEmbeddingProvider:
    """Dimension=8 deterministic vektör üretir."""

    def __init__(self, dim: int = 8):
        self._dim = dim
        self.call_log: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_log.append(list(texts))
        return [_text_to_vector(t, self._dim) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim


@dataclass
class _StoredItem:
    chunk: Chunk
    embedding: list[float]


class FakeVectorStore:
    """In-memory liste; `search` cosine similarity ile yaklaşık sıralar."""

    def __init__(self):
        self._items: list[_StoredItem] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        assert len(chunks) == len(embeddings)
        for c, e in zip(chunks, embeddings):
            self._items.append(_StoredItem(chunk=c, embedding=e))

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        import math

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        candidates = self._items
        if document_ids is not None:
            candidates = [i for i in self._items if i.chunk.document_id in document_ids]

        scored = [
            RetrievedChunk(chunk=item.chunk, score=cosine(query_embedding, item.embedding))
            for item in candidates
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def delete_document(self, document_id: str) -> None:
        self._items = [i for i in self._items if i.chunk.document_id != document_id]

    def list_documents(self) -> list[dict]:
        by_doc: dict[str, dict] = {}
        for item in self._items:
            doc_id = item.chunk.document_id
            if doc_id not in by_doc:
                by_doc[doc_id] = {
                    "id": doc_id,
                    "name": item.chunk.document_name,
                    "chunk_count": 0,
                }
            by_doc[doc_id]["chunk_count"] += 1
        return list(by_doc.values())


class FakeLLMProvider:
    """Fixed-response LLM. `answer_text` ile cevap şekillendirilebilir."""

    def __init__(self, answer_text: str = "fake answer", model: str = "fake-llm-v1"):
        self.answer_text = answer_text
        self._model = model
        self.call_log: list[tuple[str, list[RetrievedChunk]]] = []

    def generate(self, prompt: str, context: list[RetrievedChunk]) -> str:
        self.call_log.append((prompt, list(context)))
        return self.answer_text

    @property
    def model_name(self) -> str:
        return self._model


@dataclass
class FakeChunker:
    """Tüm içeriği tek chunk olarak döndürür veya chunk_size'e böler."""
    chunk_size: int = 1000

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []
        chunks = []
        for i, start in enumerate(range(0, len(text), self.chunk_size)):
            part = text[start : start + self.chunk_size]
            chunks.append(
                Chunk(
                    id=f"{document.id}-chunk-{i}",
                    document_id=document.id,
                    document_name=document.name,
                    content=part,
                    index=i,
                    metadata={"char_start": start, "char_end": start + len(part)},
                )
            )
        return chunks


@dataclass
class FakeDocumentLoader:
    """bytes'ı utf-8 decode ederek Document döndürür."""

    def load(self, filename: str, content: bytes) -> Document:
        return Document(
            id="fake-doc-id",
            name=filename,
            content=content.decode("utf-8"),
            mime_type="text/plain",
            metadata={"size_bytes": len(content)},
        )
