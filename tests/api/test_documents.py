"""Documents list/delete endpoint testleri."""
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.dependencies import get_pipeline
from src.adapters.loaders import get_loader as real_loader_factory
from src.rag.pipeline import RAGPipeline
from tests.fakes import (
    FakeChunker, FakeEmbeddingProvider,
    FakeLLMProvider, FakeVectorStore,
)


def _fake_pipeline():
    return RAGPipeline(
        loader_factory=real_loader_factory,
        chunker=FakeChunker(chunk_size=50),
        embedder=FakeEmbeddingProvider(),
        store=FakeVectorStore(),
        llm=FakeLLMProvider(),
    )


def _client_with_pipeline(pipeline):
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)


def test_list_documents_empty():
    pipeline = _fake_pipeline()
    client = _client_with_pipeline(pipeline)
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == {"documents": []}


def test_list_documents_after_upload():
    pipeline = _fake_pipeline()
    client = _client_with_pipeline(pipeline)
    client.post("/upload", files=[("files", ("a.txt", b"content abc", "text/plain"))])
    response = client.get("/documents")
    assert response.status_code == 200
    docs = response.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["name"] == "a.txt"


def test_delete_document():
    pipeline = _fake_pipeline()
    client = _client_with_pipeline(pipeline)
    upload_resp = client.post(
        "/upload", files=[("files", ("a.txt", b"content abc", "text/plain"))]
    )
    doc_id = upload_resp.json()["results"][0]["document"]["id"]

    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted_document_id"] == doc_id

    list_resp = client.get("/documents")
    assert list_resp.json()["documents"] == []


def test_delete_nonexistent_document_returns_200():
    """Idempotent delete: olmayan id için de 200 dön (hata atma)."""
    pipeline = _fake_pipeline()
    client = _client_with_pipeline(pipeline)
    response = client.delete("/documents/nonexistent-id")
    assert response.status_code == 200
