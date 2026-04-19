"""RAGPipeline ingestion testleri — fake providers'la."""
from src.rag.pipeline import RAGPipeline


def test_ingest_calls_loader_chunker_embedder_store(
    fake_chunker, fake_embedder, fake_store, fake_llm, fake_loader
):
    def _loader_factory(mime_type, filename=None):
        return fake_loader

    pipeline = RAGPipeline(
        loader_factory=_loader_factory,
        chunker=fake_chunker,
        embedder=fake_embedder,
        store=fake_store,
        llm=fake_llm,
    )

    doc = pipeline.ingest(
        filename="test.txt",
        content=b"Lorem ipsum dolor sit amet, consectetur adipiscing elit." * 3,
        mime_type="text/plain",
    )

    # Document doğru parse edildi mi?
    assert doc.name == "test.txt"
    assert doc.content.startswith("Lorem")

    # Embedder çağrıldı mı?
    assert len(fake_embedder.call_log) == 1
    embedded_texts = fake_embedder.call_log[0]
    assert len(embedded_texts) >= 1  # chunker en az 1 chunk üretti

    # Store'a yazıldı mı?
    stored = fake_store.list_documents()
    assert len(stored) == 1
    assert stored[0]["name"] == "test.txt"


def test_ingest_empty_content_noop(fake_chunker, fake_embedder, fake_store, fake_llm, fake_loader):
    def _loader_factory(mime_type, filename=None):
        return fake_loader

    pipeline = RAGPipeline(
        loader_factory=_loader_factory,
        chunker=fake_chunker,
        embedder=fake_embedder,
        store=fake_store,
        llm=fake_llm,
    )

    doc = pipeline.ingest(filename="empty.txt", content=b"", mime_type="text/plain")
    assert doc.content == ""
    # Boş içerik → 0 chunk → embedder çağrılmaz
    assert fake_embedder.call_log == []
