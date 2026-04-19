"""Query endpoint'leri: /query, /summarize."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.dependencies import get_pipeline
from backend.schemas import (
    QueryRequest,
    QueryResponse,
    SourceCitation,
)
from src.rag.pipeline import RAGPipeline

router = APIRouter(prefix="", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    answer = pipeline.answer(
        question=req.question,
        document_ids=req.document_ids,
        top_k=req.top_k,
    )
    sources = [
        SourceCitation(
            document_id=s.chunk.document_id,
            document_name=s.chunk.document_name,
            chunk_index=s.chunk.index,
            chunk_preview=s.chunk.content[:200],
            score=round(s.score, 4),
        )
        for s in answer.sources
    ]
    return QueryResponse(
        answer=answer.text,
        sources=sources,
        model=answer.model,
        retrieval_count=len(answer.sources),
    )
