"""Demo mode providers — gerçek API çağrısı yapmadan full UI demo için.

Bu providers `USE_MOCK_PROVIDERS=true` ayarı aktifken `backend.dependencies`
tarafından kullanılır. Hiçbir dış servis çağırmazlar; deterministic çalışırlar.

LLM "cevap" olarak retrieval sonuçlarını template ile özetler — böylece
kullanıcı gerçek RAG akışının (chunking → embedding → retrieval → LLM)
çalıştığını görebilir, ama Gemini API key gerekmez.
"""
from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass

from src.core.interfaces import Chunk, RetrievedChunk


def _text_to_vector(text: str, dim: int = 32) -> list[float]:
    """SHA-256 tabanlı deterministic vektör."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # 32 byte — bunu dim uzunluğunda float listesine çevir
    # Eğer dim > 32 ise tekrarla
    byts = (h * ((dim // len(h)) + 1))[:dim]
    return [(b - 128) / 128.0 for b in byts]


class DemoEmbedding:
    """Deterministic SHA-256 tabanlı embedding. API çağrısı yapmaz."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_text_to_vector(t, self._dim) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim


@dataclass
class _StoredItem:
    chunk: Chunk
    embedding: list[float]


class DemoVectorStore:
    """In-memory vector store. Chroma yerine kullanılır demo için.

    NOT: Bu store persist ETMEZ — backend yeniden başlatılırsa tüm dokümanlar
    kaybolur. Bu kasıtlı; demo modu temiz bir sayfa sunar.
    """

    def __init__(self):
        self._items: list[_StoredItem] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks ve embeddings eşit uzunlukta olmalı")
        for c, e in zip(chunks, embeddings):
            self._items.append(_StoredItem(chunk=c, embedding=e))

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        candidates = self._items
        if document_ids is not None:
            candidates = [i for i in self._items if i.chunk.document_id in document_ids]

        scored = [
            RetrievedChunk(
                chunk=item.chunk,
                score=self._cosine(query_embedding, item.embedding),
            )
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

    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        items = [i for i in self._items if i.chunk.document_id == document_id]
        return sorted([i.chunk for i in items], key=lambda c: c.index)


class DemoLLM:
    """Template-tabanlı LLM. Retrieved chunk'ları özetleyerek cevap üretir.

    Gerçek bir LLM cevabı değildir, ama retrieval'in çalıştığını gösterir:
    bulunan chunk'ların önizlemelerini döndürür. Özet promptu verilirse
    (summarize flow) farklı bir template'e düşer.
    """

    MODEL_NAME = "demo-mock-llm"

    def generate(self, prompt: str, context: list[RetrievedChunk]) -> str:
        is_summary = "özet" in prompt.lower() or "summary" in prompt.lower() or "özetini" in prompt.lower()

        if not context:
            return (
                "🟡 **Demo modu** — Sorunuz için ilgili doküman bulunamadı. "
                "Önce bir doküman yükleyin, sonra tekrar deneyin.\n\n"
                "_Bu bir gerçek LLM cevabı değildir. Gerçek cevaplar için "
                "`.env` dosyasına `GEMINI_API_KEY` ekleyin ve "
                "`USE_MOCK_PROVIDERS=false` yapın._"
            )

        if is_summary:
            previews = "\n".join(
                f"- **Bölüm {r.chunk.index + 1}:** {r.chunk.content[:120]}..."
                for r in context[:8]
            )
            return (
                f"🟡 **Demo modu — Doküman Özeti**\n\n"
                f"Dokümanın {len(context)} bölümü bulundu. İçerik parçaları:\n\n"
                f"{previews}\n\n"
                f"_Bu bir gerçek özet değildir. Gerçek özet için "
                f"`GEMINI_API_KEY` yapılandırın._"
            )

        previews = "\n".join(
            f"- **{r.chunk.document_name}** (chunk {r.chunk.index}, "
            f"benzerlik: {r.score:.3f}): _{r.chunk.content[:150]}..._"
            for r in context
        )
        return (
            f"🟡 **Demo modu cevabı**\n\n"
            f"Sorunuza en yakın {len(context)} içerik parçası bulundu:\n\n"
            f"{previews}\n\n"
            f"_Gerçek bir LLM cevabı için `.env` dosyasına "
            f"`GEMINI_API_KEY` ekleyin ve `USE_MOCK_PROVIDERS=false` yapın._"
        )

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME
