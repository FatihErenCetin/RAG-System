"""Core interface ve dataclass'ların instantiation testi."""
from src.core.interfaces import Document, Chunk, RetrievedChunk, Answer


def test_document_creation():
    doc = Document(
        id="abc-123",
        name="test.pdf",
        content="hello world",
        mime_type="application/pdf",
        metadata={"size_bytes": 100},
    )
    assert doc.id == "abc-123"
    assert doc.content == "hello world"


def test_chunk_creation():
    chunk = Chunk(
        id="chunk-1",
        document_id="abc-123",
        document_name="test.pdf",
        content="partial text",
        index=0,
        metadata={"char_start": 0, "char_end": 12},
    )
    assert chunk.index == 0
    assert chunk.document_name == "test.pdf"


def test_retrieved_chunk_carries_score():
    chunk = Chunk(
        id="c1", document_id="d1", document_name="x.txt",
        content="t", index=0, metadata={},
    )
    retrieved = RetrievedChunk(chunk=chunk, score=0.87)
    assert retrieved.score == 0.87
    assert retrieved.chunk.id == "c1"


def test_answer_default_structure():
    answer = Answer(text="42", sources=[], model="gemini-2.5-flash")
    assert answer.text == "42"
    assert answer.sources == []
    assert answer.model == "gemini-2.5-flash"
