"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.routes import documents, health, query as query_routes


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Chat With Your Documents",
        version="0.1.0-mvp",
        description="RAG API — upload docs, ask questions, get summaries",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def limit_upload_size(request: Request, call_next):
        if request.url.path == "/upload" and request.method == "POST":
            cl = request.headers.get("content-length")
            if cl and int(cl) > settings.max_upload_mb * 1024 * 1024:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "PayloadTooLarge",
                        "detail": f"Toplam upload {settings.max_upload_mb}MB'ı aşamaz.",
                    },
                )
        return await call_next(request)

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query_routes.router)
    return app


app = create_app()
