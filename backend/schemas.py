"""Pydantic request/response modelleri."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---- Health ----

class HealthResponse(BaseModel):
    status: str
    version: str


# ---- Documents ----

class DocumentInfo(BaseModel):
    id: str
    name: str
    chunk_count: int


class UploadResult(BaseModel):
    document: DocumentInfo


class UploadResponse(BaseModel):
    results: list[UploadResult]
    total_chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    deleted_document_id: str


# ---- Query ----

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    document_ids: list[str] | None = None
    top_k: int = Field(4, ge=1, le=20)


class SourceCitation(BaseModel):
    document_id: str
    document_name: str
    chunk_index: int
    chunk_preview: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    model: str
    retrieval_count: int


# ---- Summarize ----

class SummarizeRequest(BaseModel):
    document_id: str


class SummarizeResponse(BaseModel):
    summary: str
    document_id: str
    document_name: str
    model: str


# ---- Error ----

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
