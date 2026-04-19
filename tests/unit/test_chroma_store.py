"""Chroma store testleri (PersistentClient, tmp_path)."""
from src.adapters.vectorstores.chroma import ChromaVectorStore
from src.core.interfaces import Chunk


def _make_chunk(i: int, doc_id: str = "d1", doc_name: str = "d1.txt") -> Chunk:
    return Chunk(
        id=f"{doc_id}-c{i}",
        document_id=doc_id,
        document_name=doc_name,
        content=f"content {i}",
        index=i,
        metadata={"char_start": 0, "char_end": 0},
    )


def test_add_and_search(tmp_path):
    store = ChromaVectorStore(path=str(tmp_path / "db"), collection_name="test")

    chunks = [_make_chunk(0), _make_chunk(1), _make_chunk(2)]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

    store.add(chunks, embeddings)

    results = store.search(query_embedding=[1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk.id == "d1-c0"


def test_filter_by_document_ids(tmp_path):
    store = ChromaVectorStore(path=str(tmp_path / "db"), collection_name="test")

    store.add(
        [_make_chunk(0, "A"), _make_chunk(0, "B")],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    results = store.search(query_embedding=[1.0, 0.0], top_k=5, document_ids=["B"])
    assert len(results) == 1
    assert results[0].chunk.document_id == "B"


def test_delete_document(tmp_path):
    store = ChromaVectorStore(path=str(tmp_path / "db"), collection_name="test")

    store.add(
        [_make_chunk(0, "A"), _make_chunk(1, "A"), _make_chunk(0, "B")],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )

    store.delete_document("A")

    results = store.search(query_embedding=[1.0, 0.0], top_k=5)
    assert all(r.chunk.document_id != "A" for r in results)
    assert len(results) == 1


def test_list_documents(tmp_path):
    store = ChromaVectorStore(path=str(tmp_path / "db"), collection_name="test")

    store.add(
        [_make_chunk(0, "A", "a.pdf"), _make_chunk(1, "A", "a.pdf"), _make_chunk(0, "B", "b.txt")],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )

    docs = store.list_documents()
    by_id = {d["id"]: d for d in docs}
    assert by_id["A"]["chunk_count"] == 2
    assert by_id["A"]["name"] == "a.pdf"
    assert by_id["B"]["chunk_count"] == 1
