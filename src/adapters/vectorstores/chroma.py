"""Chroma PersistentClient adapter."""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.interfaces import Chunk, RetrievedChunk


class ChromaVectorStore:
    """Chroma persistent store; disk-backed."""

    def __init__(self, path: str, collection_name: str = "documents"):
        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks ve embeddings eşit uzunlukta olmalı")

        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "document_name": c.document_name,
                    "chunk_index": c.index,
                    "char_start": c.metadata.get("char_start", 0),
                    "char_end": c.metadata.get("char_end", 0),
                }
                for c in chunks
            ],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        where = None
        if document_ids:
            where = {"document_id": {"$in": document_ids}}

        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        ids = raw.get("ids", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]

        results: list[RetrievedChunk] = []
        for _id, dist, content, meta in zip(ids, distances, documents, metadatas):
            score = max(0.0, 1.0 - dist)  # cosine distance → similarity
            chunk = Chunk(
                id=_id,
                document_id=meta["document_id"],
                document_name=meta["document_name"],
                content=content,
                index=int(meta.get("chunk_index", 0)),
                metadata={
                    "char_start": int(meta.get("char_start", 0)),
                    "char_end": int(meta.get("char_end", 0)),
                },
            )
            results.append(RetrievedChunk(chunk=chunk, score=score))
        return results

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def list_documents(self) -> list[dict]:
        raw = self._collection.get()  # tüm kayıtları çek
        metadatas = raw.get("metadatas", []) or []

        by_doc: dict[str, dict] = {}
        for meta in metadatas:
            doc_id = meta["document_id"]
            if doc_id not in by_doc:
                by_doc[doc_id] = {
                    "id": doc_id,
                    "name": meta["document_name"],
                    "chunk_count": 0,
                }
            by_doc[doc_id]["chunk_count"] += 1
        return list(by_doc.values())

    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        raw = self._collection.get(where={"document_id": document_id})

        ids = raw.get("ids", []) or []
        documents = raw.get("documents", []) or []
        metadatas = raw.get("metadatas", []) or []

        chunks: list[Chunk] = []
        for _id, content, meta in zip(ids, documents, metadatas):
            chunks.append(
                Chunk(
                    id=_id,
                    document_id=meta["document_id"],
                    document_name=meta["document_name"],
                    content=content,
                    index=int(meta.get("chunk_index", 0)),
                    metadata={
                        "char_start": int(meta.get("char_start", 0)),
                        "char_end": int(meta.get("char_end", 0)),
                    },
                )
            )
        chunks.sort(key=lambda c: c.index)
        return chunks
