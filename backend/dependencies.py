"""Dependency injection: RAGPipeline'ı FastAPI endpoint'lerine sun."""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from backend.config import Settings, get_settings
from src.adapters.chunkers.recursive import RecursiveChunker
from src.adapters.embeddings.gemini import GeminiEmbedding
from src.adapters.llm.gemini import GeminiLLM
from src.adapters.loaders import get_loader as loader_factory_fn
from src.adapters.vectorstores.chroma import ChromaVectorStore
from src.rag.pipeline import RAGPipeline


@lru_cache(maxsize=1)
def _build_pipeline(settings_hash: str) -> RAGPipeline:
    """Settings'in string representasyonuyla cache'le (tek instance yeterli)."""
    settings = get_settings()
    chunker = RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedder = GeminiEmbedding(
        api_key=settings.gemini_api_key,
        model=settings.gemini_embedding_model,
    )
    store = ChromaVectorStore(path=settings.chroma_path)
    llm = GeminiLLM(
        api_key=settings.gemini_api_key,
        model=settings.gemini_llm_model,
    )
    return RAGPipeline(
        loader_factory=loader_factory_fn,
        chunker=chunker,
        embedder=embedder,
        store=store,
        llm=llm,
    )


def get_pipeline(settings: Settings = Depends(get_settings)) -> RAGPipeline:
    """Endpoint'lere RAGPipeline inject et."""
    return _build_pipeline(settings.chroma_path + ":" + settings.gemini_llm_model)
