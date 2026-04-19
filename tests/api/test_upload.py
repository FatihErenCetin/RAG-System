"""Upload endpoint testi — dependency override ile."""
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.dependencies import get_pipeline
from src.adapters.loaders import get_loader as real_loader_factory
from src.rag.pipeline import RAGPipeline
from tests.fakes import (
    FakeChunker,
    FakeEmbeddingProvider,
    FakeLLMProvider,
    FakeVectorStore,
)


def _fake_pipeline():
    return RAGPipeline(
        loader_factory=real_loader_factory,  # USE REAL FACTORY
        chunker=FakeChunker(chunk_size=50),
        embedder=FakeEmbeddingProvider(),
        store=FakeVectorStore(),
        llm=FakeLLMProvider(),
    )


def _client_with_fake_pipeline():
    app = create_app()
    pipeline = _fake_pipeline()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app), pipeline


def test_upload_txt_file_single():
    client, _ = _client_with_fake_pipeline()
    files = [("files", ("sample.txt", b"Hello world from test.", "text/plain"))]
    response = client.post("/upload", files=files)

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["document"]["name"] == "sample.txt"
    assert body["total_chunks"] >= 1


def test_upload_multiple_files():
    client, _ = _client_with_fake_pipeline()
    files = [
        ("files", ("a.txt", b"Content A xxxxxxx", "text/plain")),
        ("files", ("b.txt", b"Content B yyyyyyy", "text/plain")),
    ]
    response = client.post("/upload", files=files)

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    names = [r["document"]["name"] for r in body["results"]]
    assert "a.txt" in names and "b.txt" in names


def test_upload_unsupported_type_returns_400():
    client, _ = _client_with_fake_pipeline()
    files = [("files", ("image.png", b"\x89PNG\r\n...", "image/png"))]
    response = client.post("/upload", files=files)

    assert response.status_code == 400
    assert "image/png" in response.json()["detail"] or "desteklen" in response.json()["detail"]


def test_upload_no_files_returns_422():
    client, _ = _client_with_fake_pipeline()
    response = client.post("/upload")
    assert response.status_code == 422


def test_upload_size_limit(monkeypatch):
    """MAX_UPLOAD_MB aşıldığında 413 dön."""
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    from backend.dependencies import _build_pipeline
    _build_pipeline.cache_clear()

    client, _ = _client_with_fake_pipeline()
    big_content = b"x" * (2 * 1024 * 1024)  # 2MB
    response = client.post("/upload", files=[("files", ("big.txt", big_content, "text/plain"))])
    assert response.status_code == 413
