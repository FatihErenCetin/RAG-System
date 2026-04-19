"""Domain interfaces (ports) + data types.

Bu modül sistemin anayasasıdır: tüm adapter'lar ve pipeline bu Protocol'lere
uymak zorundadır. LangChain, Gemini, Chroma gibi dış kütüphane tipleri buraya
ASLA sızmamalıdır.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Document:
    """Parse edilmiş tam doküman."""
    id: str
    name: str
    content: str
    mime_type: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Embedding'lenmeye hazır, dokümanın bir parçası."""
    id: str
    document_id: str
    document_name: str
    content: str
    index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """Vector search sonucu: chunk + benzerlik skoru [0, 1]."""
    chunk: Chunk
    score: float


@dataclass
class Answer:
    """LLM'nin ürettiği cevap ve kaynak chunk'lar."""
    text: str
    sources: list[RetrievedChunk]
    model: str


@runtime_checkable
class DocumentLoader(Protocol):
    """Raw bytes → Document (parse)."""

    def load(self, filename: str, content: bytes) -> Document:
        """Dosyayı parse et ve Document döndür."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Document → list[Chunk]."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Dokümanı chunk'lara böl."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """list[str] → list[vector]."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Metinleri vektörlere çevir."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vektör boyutu."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Chunk + embedding saklama ve benzerlik araması."""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Chunk'ları embedding'leriyle birlikte kaydet."""
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Sorgu vektörüne en yakın top_k chunk'ı döndür."""
        ...

    def delete_document(self, document_id: str) -> None:
        """Bir dokümana ait tüm chunk'ları sil."""
        ...

    def list_documents(self) -> list[dict]:
        """Kayıtlı dokümanların bilgilerini döndür (id, name, chunk_count, vs)."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Prompt + context → generated text."""

    def generate(self, prompt: str, context: list[RetrievedChunk]) -> str:
        """LLM'e soruyu ve context'i ver, cevabı döndür."""
        ...

    @property
    def model_name(self) -> str:
        """Kullanılan model adı (response metadata için)."""
        ...
