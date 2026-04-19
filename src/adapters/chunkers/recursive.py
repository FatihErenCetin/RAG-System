"""Recursive character splitter — LangChain wrapper.

Faz 2'de bu wrapper kaldırılıp saf Python'la yeniden yazılacak.
Şimdilik battle-tested LangChain splitter'ını kullanıyoruz.
"""
from __future__ import annotations

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.interfaces import Chunk, Document


class RecursiveChunker:
    """`RecursiveCharacterTextSplitter` wrapper'ı; `Chunker` Protocol'ünü uygular."""

    _DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or self._DEFAULT_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
            add_start_index=True,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.content:
            return []

        lc_docs = self._splitter.create_documents(
            texts=[document.content],
            metadatas=[{"document_id": document.id}],
        )

        chunks: list[Chunk] = []
        for i, lc_doc in enumerate(lc_docs):
            start = lc_doc.metadata.get("start_index", 0)
            end = start + len(lc_doc.page_content)
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document.id,
                    document_name=document.name,
                    content=lc_doc.page_content,
                    index=i,
                    metadata={"char_start": start, "char_end": end},
                )
            )
        return chunks
