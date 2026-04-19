"""Query endpoint testi."""
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.dependencies import get_pipeline
from src.adapters.loaders import get_loader as real_loader_factory
from src.rag.pipeline import RAGPipeline
from tests.fakes import (
    FakeChunker, FakeEmbeddingProvider,
    FakeLLMProvider, FakeVectorStore,
)


def _fake_pipeline(answer: str = "canned answer"):
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


def test_query_returns_answer_and_sources():
    pipeline = _fake_pipeline(answer="Test cevabı.")
    client = _client(pipeline)
    client.post("/upload", files=[("files", ("doc1.txt", b"python bir dildir", "text/plain"))])

    response = client.post("/query", json={"question": "Python nedir?", "top_k": 3})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "Test cevabı."
    assert body["model"] == "fake-llm-v1"
    assert body["retrieval_count"] >= 1
    assert len(body["sources"]) >= 1
    src = body["sources"][0]
    for field in ["document_id", "document_name", "chunk_index", "chunk_preview", "score"]:
        assert field in src


def test_query_validates_min_length():
    pipeline = _fake_pipeline()
    client = _client(pipeline)
    response = client.post("/query", json={"question": "a"})  # < 3
    assert response.status_code == 422


def test_query_respects_document_ids():
    pipeline = _fake_pipeline(answer="filtered")
    client = _client(pipeline)
    r = client.post(
        "/upload",
        files=[
            ("files", ("A.txt", b"alpha content xxxxxxxxxxx", "text/plain")),
            ("files", ("B.txt", b"bravo content yyyyyyyyyyyy", "text/plain")),
        ],
    )
    doc_a_id = r.json()["results"][0]["document"]["id"]

    response = client.post(
        "/query",
        json={"question": "What about alpha?", "document_ids": [doc_a_id], "top_k": 5},
    )
    assert response.status_code == 200
    for src in response.json()["sources"]:
        assert src["document_id"] == doc_a_id
