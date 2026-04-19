"""RAGPipeline summarize testleri."""
import pytest

from src.core.interfaces import Chunk
from src.rag.pipeline import RAGPipeline


def test_summarize_uses_all_chunks_of_document(
    fake_chunker, fake_embedder, fake_store, fake_llm, fake_loader
):
    chunks = [
        Chunk(id="c1", document_id="D1", document_name="plan.txt",
              content="İlk bölüm: giriş ve amaç.", index=0, metadata={}),
        Chunk(id="c2", document_id="D1", document_name="plan.txt",
              content="İkinci bölüm: metodoloji.", index=1, metadata={}),
        Chunk(id="c3", document_id="D1", document_name="plan.txt",
              content="Üçüncü bölüm: sonuçlar.", index=2, metadata={}),
        Chunk(id="x1", document_id="OTHER", document_name="other.txt",
              content="Alakasız içerik.", index=0, metadata={}),
    ]
    embeddings = fake_embedder.embed([c.content for c in chunks])
    fake_store.add(chunks, embeddings)

    fake_llm.answer_text = "Bu bir özet cevabıdır."

    pipeline = RAGPipeline(
        loader_factory=lambda m, f=None: fake_loader,
        chunker=fake_chunker,
        embedder=fake_embedder,
        store=fake_store,
        llm=fake_llm,
    )

    answer = pipeline.summarize(document_id="D1")

    assert answer.text == "Bu bir özet cevabıdır."
    assert answer.model == "fake-llm-v1"
    _, context = fake_llm.call_log[-1]
    assert len(context) == 3
    assert all(rc.chunk.document_id == "D1" for rc in context)


def test_summarize_missing_document_raises(
    fake_chunker, fake_embedder, fake_store, fake_llm, fake_loader
):
    pipeline = RAGPipeline(
        loader_factory=lambda m, f=None: fake_loader,
        chunker=fake_chunker,
        embedder=fake_embedder,
        store=fake_store,
        llm=fake_llm,
    )
    with pytest.raises(ValueError, match="not found"):
        pipeline.summarize(document_id="NONEXISTENT")
