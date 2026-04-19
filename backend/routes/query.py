"""Query endpoint'leri: /query, /summarize."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import get_pipeline
from backend.schemas import (
    QueryRequest,
    QueryResponse,
    SourceCitation,
    SummarizeRequest,
    SummarizeResponse,
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


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(
    req: SummarizeRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> SummarizeResponse:
    try:
        answer = pipeline.summarize(document_id=req.document_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e

    doc_name = answer.sources[0].chunk.document_name if answer.sources else "(unknown)"
    return SummarizeResponse(
        summary=answer.text,
        document_id=req.document_id,
        document_name=doc_name,
        model=answer.model,
    )
