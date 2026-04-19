"""Summarize endpoint testi."""
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.dependencies import get_pipeline
from src.adapters.loaders import get_loader as real_loader_factory
from src.rag.pipeline import RAGPipeline
from tests.fakes import (
    FakeChunker, FakeEmbeddingProvider,
    FakeLLMProvider, FakeVectorStore,
)


def _pipeline(answer: str = "özet"):
    return RAGPipeline(
        loader_factory=real_loader_factory,
        chunker=FakeChunker(chunk_size=50),
        embedder=FakeEmbeddingProvider(),
        store=FakeVectorStore(),
        llm=FakeLLMProvider(answer_text=answer),
    )


def _client(pipeline):
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)


def test_summarize_returns_summary():
    pipeline = _pipeline(answer="Bu bir test özetidir.")
    client = _client(pipeline)
    up = client.post(
        "/upload",
        files=[("files", ("rapor.txt", b"Uzun bir dokuman icerigi x" * 20, "text/plain"))],
    )
    doc_id = up.json()["results"][0]["document"]["id"]

    resp = client.post("/summarize", json={"document_id": doc_id})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["summary"] == "Bu bir test özetidir."
    assert body["document_id"] == doc_id
    assert body["document_name"] == "rapor.txt"


def test_summarize_nonexistent_document_returns_404():
    pipeline = _pipeline()
    client = _client(pipeline)
    resp = client.post("/summarize", json={"document_id": "nope"})
    assert resp.status_code == 404
