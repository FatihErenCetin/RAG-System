"""Document endpoint'leri: upload, list, delete."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.dependencies import get_pipeline
from backend.schemas import (
    DocumentInfo,
    UploadResponse,
    UploadResult,
)
from src.adapters.loaders import UnsupportedFileType
from src.rag.pipeline import RAGPipeline

router = APIRouter(prefix="", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: list[UploadFile] = File(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> UploadResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="En az bir dosya yüklemelisiniz.",
        )

    results: list[UploadResult] = []
    total_chunks = 0

    for upload in files:
        content = await upload.read()
        try:
            doc = pipeline.ingest(
                filename=upload.filename,
                content=content,
                mime_type=upload.content_type,
            )
        except UnsupportedFileType as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{upload.filename}: {e}",
            ) from e

        # chunk sayısını store'dan al (Task 19'da list_documents pass-through eklenecek)
        docs = pipeline._store.list_documents()  # noqa: SLF001
        chunk_count = next(
            (d["chunk_count"] for d in docs if d["id"] == doc.id), 0
        )
        total_chunks += chunk_count

        results.append(
            UploadResult(
                document=DocumentInfo(
                    id=doc.id, name=doc.name, chunk_count=chunk_count,
                )
            )
        )

    return UploadResponse(results=results, total_chunks=total_chunks)
