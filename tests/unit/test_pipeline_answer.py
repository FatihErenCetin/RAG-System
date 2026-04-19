"""RAGPipeline answer (Q&A) testleri."""
from src.core.interfaces import Chunk
from src.rag.pipeline import RAGPipeline


def _seed_store(store, chunks_with_embeddings):
    """Helper: store'a ön-embedded chunk'lar ekle."""
    chunks = [c for c, _ in chunks_with_embeddings]
    embs = [e for _, e in chunks_with_embeddings]
    store.add(chunks, embs)


def test_answer_returns_Answer_with_sources(
    fake_chunker, fake_embedder, fake_store, fake_llm, fake_loader
):
    _seed_store(
        fake_store,
        [
            (
                Chunk(
                    id="c1", document_id="d1", document_name="doc1.txt",
                    content="Python en popüler programlama dillerinden biridir.",
                    index=0, metadata={},
                ),
                fake_embedder.embed(["Python en popüler programlama dillerinden biridir."])[0],
            ),
            (
                Chunk(
                    id="c2", document_id="d2", document_name="doc2.txt",
                    content="Tamamen alakasız içerik foo bar baz.",
                    index=0, metadata={},
                ),
                fake_embedder.embed(["Tamamen alakasız içerik foo bar baz."])[0],
            ),
        ],
    )

    pipeline = RAGPipeline(
        loader_factory=lambda m, f=None: fake_loader,
        chunker=fake_chunker,
        embedder=fake_embedder,
        store=fake_store,
        llm=fake_llm,
    )

    answer = pipeline.answer(question="Python nedir?", top_k=2)

    assert answer.text == "fake answer"
    assert answer.model == "fake-llm-v1"
    assert len(answer.sources) <= 2
    assert len(fake_llm.call_log) == 1


def test_answer_respects_document_ids_filter(
    fake_chunker, fake_embedder, fake_store, fake_llm, fake_loader
):
    _seed_store(
        fake_store,
        [
            (
                Chunk(id="a1", document_id="A", document_name="a.txt",
                      content="Apple text", index=0, metadata={}),
                fake_embedder.embed(["Apple text"])[0],
            ),
            (
                Chunk(id="b1", document_id="B", document_name="b.txt",
                      content="Banana text", index=0, metadata={}),
                fake_embedder.embed(["Banana text"])[0],
            ),
        ],
    )

    pipeline = RAGPipeline(
        loader_factory=lambda m, f=None: fake_loader,
        chunker=fake_chunker,
        embedder=fake_embedder,
        store=fake_store,
        llm=fake_llm,
    )

    answer = pipeline.answer(question="xyz?", document_ids=["A"], top_k=5)

    for src in answer.sources:
        assert src.chunk.document_id == "A"
