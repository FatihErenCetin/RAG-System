"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.routes import health


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

    app.include_router(health.router)
    return app


app = create_app()
