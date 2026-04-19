# Chat With Your Documents — Faz 1 (MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bir RAG (Retrieval-Augmented Generation) uygulaması için MVP teslim et: kullanıcı TXT/PDF/DOCX/DOC dokümanlarını yükler, sistem chunk + embed + Chroma'ya kaydeder, Gemini LLM ile soru-cevap / özet üretir; FastAPI backend + Streamlit frontend.

**Architecture:** Hexagonal (Ports & Adapters): domain (`src/core/`) Protocol'leri tanımlar, adapter'lar (`src/adapters/`) somut sağlayıcıları uygular, RAG pipeline (`src/rag/`) orchestrator'dır, FastAPI (`backend/`) sadece Protocol'lere bağımlıdır. Streamlit frontend REST üzerinden backend ile konuşur.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, Chroma (PersistentClient), Google Gemini (`gemini-2.5-flash` + `text-embedding-004`), LangChain text splitter + document loaders (sadece loader + chunker için), Streamlit, pytest.

**Referans spec:** `docs/superpowers/specs/2026-04-19-chat-with-your-docs-design.md`

---

## Dosya Yapısı Haritası

Her task'ın hangi dosyaları oluşturduğu / değiştirdiği özet:

```
RAG Systems/
├── backend/
│   ├── __init__.py                   [Task 1]
│   ├── main.py                       [Task 17 + extended: Tasks 18-22]
│   ├── config.py                     [Task 2]
│   ├── dependencies.py               [Task 17]
│   ├── schemas.py                    [Task 17 + extended]
│   └── routes/
│       ├── __init__.py               [Task 17]
│       ├── health.py                 [Task 17]
│       ├── documents.py              [Task 18, 19]
│       └── query.py                  [Task 20, 21]
├── src/
│   ├── __init__.py                   [Task 1]
│   ├── core/
│   │   ├── __init__.py               [Task 3]
│   │   ├── interfaces.py             [Task 3]
│   │   └── prompts.py                [Task 15, 16]
│   ├── rag/
│   │   ├── __init__.py               [Task 14]
│   │   └── pipeline.py               [Task 14, 15, 16]
│   └── adapters/
│       ├── __init__.py               [Task 5]
│       ├── chunkers/
│       │   ├── __init__.py           [Task 5]
│       │   └── recursive.py          [Task 5]
│       ├── loaders/
│       │   ├── __init__.py           [Task 10]
│       │   ├── txt_loader.py         [Task 6]
│       │   ├── pdf_loader.py         [Task 7]
│       │   ├── docx_loader.py        [Task 8]
│       │   └── doc_loader.py         [Task 9]
│       ├── embeddings/
│       │   ├── __init__.py           [Task 11]
│       │   └── gemini.py             [Task 11]
│       ├── vectorstores/
│       │   ├── __init__.py           [Task 12]
│       │   └── chroma.py             [Task 12]
│       └── llm/
│           ├── __init__.py           [Task 13]
│           └── gemini.py             [Task 13]
├── frontend/
│   ├── api_client.py                 [Task 23]
│   ├── app.py                        [Task 27]
│   └── components/
│       ├── __init__.py               [Task 24]
│       ├── upload.py                 [Task 24]
│       ├── documents.py              [Task 25]
│       └── chat.py                   [Task 26]
├── tests/
│   ├── __init__.py                   [Task 1]
│   ├── fakes.py                      [Task 4]
│   ├── conftest.py                   [Task 4]
│   ├── fixtures/
│   │   ├── sample.txt                [Task 6]
│   │   ├── sample.pdf                [Task 7]
│   │   └── sample.docx               [Task 8]
│   ├── unit/
│   │   ├── test_chunker.py           [Task 5]
│   │   ├── test_txt_loader.py        [Task 6]
│   │   ├── test_pdf_loader.py        [Task 7]
│   │   ├── test_docx_loader.py       [Task 8]
│   │   ├── test_loader_factory.py    [Task 10]
│   │   ├── test_pipeline_ingest.py   [Task 14]
│   │   ├── test_pipeline_answer.py   [Task 15]
│   │   └── test_pipeline_summarize.py [Task 16]
│   └── api/
│       ├── test_health.py            [Task 17]
│       ├── test_upload.py            [Task 18]
│       ├── test_documents.py         [Task 19]
│       ├── test_query.py             [Task 20]
│       └── test_summarize.py         [Task 21]
├── data/                             [Task 1 — .gitignore'a eklenir]
├── .env                              [Task 1 — .gitignore'a eklenir]
├── .env.example                      [Task 1]
├── .gitignore                        [Task 1]
├── pyproject.toml                    [Task 1]
├── requirements.txt                  [Task 1]
└── README.md                         [Task 28]
```

---

## Task 1: Proje Yapısı ve Bağımlılıklar

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `backend/__init__.py`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Git repo başlat ve temel klasörleri oluştur**

```bash
cd "/Users/fatiherencetin/Desktop/RAG Systems"
git init
git config --local user.name "<seninle zaten ayarlı olan>"  # skip if already set
mkdir -p backend/routes src/core src/rag src/adapters/chunkers src/adapters/loaders \
         src/adapters/embeddings src/adapters/vectorstores src/adapters/llm \
         frontend/components tests/unit tests/api tests/fixtures data
```

- [ ] **Step 2: `__init__.py` dosyalarını oluştur**

Aşağıdaki her yol için boş `__init__.py` yaz:
```
backend/__init__.py
backend/routes/__init__.py
src/__init__.py
src/core/__init__.py
src/rag/__init__.py
src/adapters/__init__.py
src/adapters/chunkers/__init__.py
src/adapters/loaders/__init__.py
src/adapters/embeddings/__init__.py
src/adapters/vectorstores/__init__.py
src/adapters/llm/__init__.py
frontend/components/__init__.py
tests/__init__.py
tests/unit/__init__.py
tests/api/__init__.py
```

Hepsi tek satır: `"""Module init."""`

- [ ] **Step 3: `pyproject.toml` yaz**

```toml
[project]
name = "chat-with-your-docs"
version = "0.1.0-mvp"
description = "RAG application — chat with your own documents"
requires-python = ">=3.11"
readme = "README.md"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]
```

- [ ] **Step 4: `requirements.txt` yaz**

```
# Web
fastapi>=0.110
uvicorn[standard]>=0.27
python-multipart>=0.0.9
pydantic>=2.6
pydantic-settings>=2.2

# RAG core
google-generativeai>=0.8
chromadb>=0.5
tenacity>=8.2

# Document parsing
langchain-text-splitters>=0.2
langchain-community>=0.2
pypdf>=4.0
python-docx>=1.1
docx2txt>=0.8
textract>=1.6  # .doc desteği; sistem antiword gerektirir

# Frontend
streamlit>=1.35
requests>=2.31

# Dev
pytest>=8.0
pytest-asyncio>=0.23
httpx>=0.27
ruff>=0.3
```

- [ ] **Step 5: `.env.example` yaz**

```bash
# LLM
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
GEMINI_LLM_MODEL=gemini-2.5-flash

# Embedding
EMBEDDING_PROVIDER=gemini
GEMINI_EMBEDDING_MODEL=text-embedding-004

# Vector store
VECTOR_STORE=chroma
CHROMA_PATH=./data/chroma_db

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# API
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:8501
MAX_UPLOAD_MB=50

# Retrieval
DEFAULT_TOP_K=4

# Frontend
BACKEND_URL=http://localhost:8000
```

- [ ] **Step 6: `.gitignore` yaz**

```
# Python
__pycache__/
*.py[cod]
*$py.class
.Python
*.egg-info/
.pytest_cache/
.ruff_cache/

# Venv
venv/
.venv/
env/

# IDE
.vscode/
.idea/
*.swp
.DS_Store

# Env & data
.env
data/
chroma_db/

# Test artifacts
.coverage
htmlcov/
```

- [ ] **Step 7: Virtual environment oluştur ve bağımlılıkları kur**

```bash
cd "/Users/fatiherencetin/Desktop/RAG Systems"
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Expected: Tüm paketler başarıyla kurulur. `textract` kurulumda `antiword`, `swig` vb. sistem bağımlılıkları sorun çıkarırsa, şu an için `textract` satırını geçici olarak comment-out et; Task 9'da tekrar ele alacağız.

- [ ] **Step 8: İlk commit**

```bash
git add pyproject.toml requirements.txt .env.example .gitignore \
        backend/__init__.py backend/routes/__init__.py \
        src/__init__.py src/core/__init__.py src/rag/__init__.py \
        src/adapters/__init__.py src/adapters/chunkers/__init__.py \
        src/adapters/loaders/__init__.py src/adapters/embeddings/__init__.py \
        src/adapters/vectorstores/__init__.py src/adapters/llm/__init__.py \
        frontend/components/__init__.py \
        tests/__init__.py tests/unit/__init__.py tests/api/__init__.py
git commit -m "chore: scaffold project structure and dependencies"
```

---

## Task 2: Configuration — `backend/config.py`

**Files:**
- Create: `backend/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Failing testi yaz**

`tests/unit/test_config.py`:

```python
"""Config yükleme testleri."""
import os
from backend.config import Settings


def test_settings_loads_required_fields(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.setenv("CHROMA_PATH", "./test_data")
    settings = Settings()
    assert settings.gemini_api_key == "test-key-123"
    assert settings.chroma_path == "./test_data"
    assert settings.llm_provider == "gemini"
    assert settings.chunk_size == 1000


def test_settings_has_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    settings = Settings()
    assert settings.api_port == 8000
    assert settings.default_top_k == 4
    assert settings.max_upload_mb == 50
    assert settings.gemini_llm_model == "gemini-2.5-flash"


def test_cors_origins_parsed_as_list(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000")
    settings = Settings()
    assert settings.cors_origins == ["http://localhost:8501", "http://localhost:3000"]
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
source .venv/bin/activate
pytest tests/unit/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'backend.config'`

- [ ] **Step 3: `backend/config.py` yaz**

```python
"""Uygulama ayarları — .env'den yüklenir."""
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Tüm ortam değişkenleri tek bir tip güvenli yapıda."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    llm_provider: str = "gemini"
    gemini_api_key: str = Field(..., description="Google AI Studio'dan alınan API key")
    gemini_llm_model: str = "gemini-2.5-flash"

    # Embedding
    embedding_provider: str = "gemini"
    gemini_embedding_model: str = "text-embedding-004"

    # Vector store
    vector_store: str = "chroma"
    chroma_path: str = "./data/chroma_db"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])
    max_upload_mb: int = 50

    # Retrieval
    default_top_k: int = 4

    # Frontend
    backend_url: str = "http://localhost:8000"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, v):
        """Virgülle ayrılmış string'i listeye çevir."""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


def get_settings() -> Settings:
    """Singleton settings factory."""
    return Settings()
```

- [ ] **Step 4: Testi tekrar çalıştır**

```bash
pytest tests/unit/test_config.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/config.py tests/unit/test_config.py
git commit -m "feat(config): pydantic-settings based typed configuration"
```

---

## Task 3: Core Interfaces — Protocols + Dataclasses

**Files:**
- Create: `src/core/interfaces.py`
- Test: `tests/unit/test_interfaces.py` (basit instantiation testi)

- [ ] **Step 1: Failing testi yaz**

`tests/unit/test_interfaces.py`:

```python
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
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_interfaces.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.interfaces'`

- [ ] **Step 3: `src/core/interfaces.py` yaz**

```python
"""Domain interfaces (ports) + data types.

Bu modül sistemin anayasasıdır: tüm adapter'lar ve pipeline bu Protocol'lere
uymak zorundadır. LangChain, Gemini, Chroma gibi dış kütüphane tipleri buraya
ASLA sızmamalıdır.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Document:
    """Parse edilmiş tam doküman."""
    id: str
    name: str
    content: str
    mime_type: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Embedding'lenmeye hazır, dokümanın bir parçası."""
    id: str
    document_id: str
    document_name: str
    content: str
    index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """Vector search sonucu: chunk + benzerlik skoru [0, 1]."""
    chunk: Chunk
    score: float


@dataclass
class Answer:
    """LLM'nin ürettiği cevap ve kaynak chunk'lar."""
    text: str
    sources: list[RetrievedChunk]
    model: str


@runtime_checkable
class DocumentLoader(Protocol):
    """Raw bytes → Document (parse)."""

    def load(self, filename: str, content: bytes) -> Document:
        """Dosyayı parse et ve Document döndür."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Document → list[Chunk]."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Dokümanı chunk'lara böl."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """list[str] → list[vector]."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Metinleri vektörlere çevir."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vektör boyutu."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Chunk + embedding saklama ve benzerlik araması."""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Chunk'ları embedding'leriyle birlikte kaydet."""
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Sorgu vektörüne en yakın top_k chunk'ı döndür."""
        ...

    def delete_document(self, document_id: str) -> None:
        """Bir dokümana ait tüm chunk'ları sil."""
        ...

    def list_documents(self) -> list[dict]:
        """Kayıtlı dokümanların bilgilerini döndür (id, name, chunk_count, vs)."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Prompt + context → generated text."""

    def generate(self, prompt: str, context: list[RetrievedChunk]) -> str:
        """LLM'e soruyu ve context'i ver, cevabı döndür."""
        ...

    @property
    def model_name(self) -> str:
        """Kullanılan model adı (response metadata için)."""
        ...
```

- [ ] **Step 4: Testi tekrar çalıştır**

```bash
pytest tests/unit/test_interfaces.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/core/interfaces.py tests/unit/test_interfaces.py
git commit -m "feat(core): define Protocol interfaces and dataclass types"
```

---

## Task 4: Test Fakes — Pipeline'ı TDD ile Yazabilmek İçin

**Files:**
- Create: `tests/fakes.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: `tests/fakes.py` yaz**

Fake provider'lar deterministik ve hızlı testler için kritiktir. Gerçek API çağırmadan pipeline'ı test edebilmemizi sağlarlar.

```python
"""Fake (test double) provider implementations.

Bu fake'ler test sırasında gerçek Gemini/Chroma çağırmadan RAGPipeline'ı
test etmemize olanak sağlar. Deterministiktirler; aynı girdi → aynı çıktı.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from src.core.interfaces import (
    Chunk,
    Document,
    EmbeddingProvider,
    LLMProvider,
    RetrievedChunk,
    VectorStore,
)


def _text_to_vector(text: str, dim: int = 8) -> list[float]:
    """Deterministic hash-based vector (test amaçlı)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # İlk dim bytei float [-1, 1]'e map et
    return [(b - 128) / 128.0 for b in h[:dim]]


class FakeEmbeddingProvider:
    """Dimension=8 deterministic vektör üretir."""

    def __init__(self, dim: int = 8):
        self._dim = dim
        self.call_log: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_log.append(list(texts))
        return [_text_to_vector(t, self._dim) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim


@dataclass
class _StoredItem:
    chunk: Chunk
    embedding: list[float]


class FakeVectorStore:
    """In-memory liste; `search` cosine similarity ile yaklaşık sıralar."""

    def __init__(self):
        self._items: list[_StoredItem] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        assert len(chunks) == len(embeddings)
        for c, e in zip(chunks, embeddings):
            self._items.append(_StoredItem(chunk=c, embedding=e))

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        import math

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        candidates = self._items
        if document_ids is not None:
            candidates = [i for i in self._items if i.chunk.document_id in document_ids]

        scored = [
            RetrievedChunk(chunk=item.chunk, score=cosine(query_embedding, item.embedding))
            for item in candidates
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def delete_document(self, document_id: str) -> None:
        self._items = [i for i in self._items if i.chunk.document_id != document_id]

    def list_documents(self) -> list[dict]:
        by_doc: dict[str, dict] = {}
        for item in self._items:
            doc_id = item.chunk.document_id
            if doc_id not in by_doc:
                by_doc[doc_id] = {
                    "id": doc_id,
                    "name": item.chunk.document_name,
                    "chunk_count": 0,
                }
            by_doc[doc_id]["chunk_count"] += 1
        return list(by_doc.values())


class FakeLLMProvider:
    """Fixed-response LLM. `answer_text` ile cevap şekillendirilebilir."""

    def __init__(self, answer_text: str = "fake answer", model: str = "fake-llm-v1"):
        self.answer_text = answer_text
        self._model = model
        self.call_log: list[tuple[str, list[RetrievedChunk]]] = []

    def generate(self, prompt: str, context: list[RetrievedChunk]) -> str:
        self.call_log.append((prompt, list(context)))
        return self.answer_text

    @property
    def model_name(self) -> str:
        return self._model


@dataclass
class FakeChunker:
    """Tüm içeriği tek chunk olarak döndürür veya chunk_size'e böler."""
    chunk_size: int = 1000

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []
        chunks = []
        for i, start in enumerate(range(0, len(text), self.chunk_size)):
            part = text[start : start + self.chunk_size]
            chunks.append(
                Chunk(
                    id=f"{document.id}-chunk-{i}",
                    document_id=document.id,
                    document_name=document.name,
                    content=part,
                    index=i,
                    metadata={"char_start": start, "char_end": start + len(part)},
                )
            )
        return chunks


@dataclass
class FakeDocumentLoader:
    """bytes'ı utf-8 decode ederek Document döndürür."""

    def load(self, filename: str, content: bytes) -> Document:
        return Document(
            id="fake-doc-id",
            name=filename,
            content=content.decode("utf-8"),
            mime_type="text/plain",
            metadata={"size_bytes": len(content)},
        )
```

- [ ] **Step 2: `tests/conftest.py` yaz**

```python
"""Pytest shared fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.fakes import (
    FakeChunker,
    FakeDocumentLoader,
    FakeEmbeddingProvider,
    FakeLLMProvider,
    FakeVectorStore,
)


@pytest.fixture
def fake_embedder():
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_store():
    return FakeVectorStore()


@pytest.fixture
def fake_llm():
    return FakeLLMProvider()


@pytest.fixture
def fake_chunker():
    return FakeChunker(chunk_size=50)


@pytest.fixture
def fake_loader():
    return FakeDocumentLoader()


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
```

- [ ] **Step 3: Fake'leri import testiyle doğrula**

```bash
python -c "from tests.fakes import FakeLLMProvider; print(FakeLLMProvider().model_name)"
```

Expected: `fake-llm-v1`

Protocol uyum kontrolü:
```bash
python -c "
from src.core.interfaces import LLMProvider, EmbeddingProvider, VectorStore
from tests.fakes import FakeLLMProvider, FakeEmbeddingProvider, FakeVectorStore
assert isinstance(FakeLLMProvider(), LLMProvider)
assert isinstance(FakeEmbeddingProvider(), EmbeddingProvider)
assert isinstance(FakeVectorStore(), VectorStore)
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add tests/fakes.py tests/conftest.py
git commit -m "test: fake provider implementations for TDD"
```

---

## Task 5: Chunker Adapter — `RecursiveCharacterTextSplitter` Wrapper

**Files:**
- Create: `src/adapters/chunkers/recursive.py`
- Test: `tests/unit/test_chunker.py`

- [ ] **Step 1: Failing testi yaz**

`tests/unit/test_chunker.py`:

```python
"""Recursive character chunker testleri."""
import pytest

from src.adapters.chunkers.recursive import RecursiveChunker
from src.core.interfaces import Document


def _make_doc(text: str) -> Document:
    return Document(
        id="doc-1",
        name="test.txt",
        content=text,
        mime_type="text/plain",
        metadata={},
    )


def test_chunker_splits_long_text():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    long_text = "a" * 200  # 200 karakter
    chunks = chunker.chunk(_make_doc(long_text))
    assert len(chunks) >= 3, "Long text should produce multiple chunks"
    for c in chunks:
        assert len(c.content) <= 50


def test_chunker_preserves_order_and_indices():
    chunker = RecursiveChunker(chunk_size=30, chunk_overlap=5)
    text = "paragraf 1.\n\nparagraf 2.\n\nparagraf 3."
    chunks = chunker.chunk(_make_doc(text))
    for i, c in enumerate(chunks):
        assert c.index == i
        assert c.document_id == "doc-1"
        assert c.document_name == "test.txt"


def test_chunker_empty_document():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(_make_doc(""))
    assert chunks == []


def test_chunker_single_short_document():
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.chunk(_make_doc("kısa metin"))
    assert len(chunks) == 1
    assert chunks[0].content == "kısa metin"


def test_chunker_adds_char_metadata():
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
    text = "abcdefghij" * 5  # 50 karakter
    chunks = chunker.chunk(_make_doc(text))
    for c in chunks:
        assert "char_start" in c.metadata
        assert "char_end" in c.metadata
        assert c.metadata["char_end"] > c.metadata["char_start"]
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_chunker.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.adapters.chunkers.recursive'`

- [ ] **Step 3: `src/adapters/chunkers/recursive.py` yaz**

```python
"""Recursive character splitter — LangChain wrapper.

Faz 2'de bu wrapper kaldırılıp saf Python'la yeniden yazılacak.
Şimdilik battle-tested LangChain splitter'ını kullanıyoruz.
"""
from __future__ import annotations

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.interfaces import Chunk, Chunker, Document


class RecursiveChunker:
    """`RecursiveCharacterTextSplitter` wrapper'ı; `Chunker` Protocol'ünü uygular."""

    _DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or self._DEFAULT_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
            add_start_index=True,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.content:
            return []

        lc_docs = self._splitter.create_documents(
            texts=[document.content],
            metadatas=[{"document_id": document.id}],
        )

        chunks: list[Chunk] = []
        for i, lc_doc in enumerate(lc_docs):
            start = lc_doc.metadata.get("start_index", 0)
            end = start + len(lc_doc.page_content)
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document.id,
                    document_name=document.name,
                    content=lc_doc.page_content,
                    index=i,
                    metadata={"char_start": start, "char_end": end},
                )
            )
        return chunks
```

`Chunker` Protocol'ünün `chunk()` metodunu uyguluyor; `runtime_checkable` ile `isinstance(x, Chunker)` çalışır.

- [ ] **Step 4: Testi tekrar çalıştır**

```bash
pytest tests/unit/test_chunker.py -v
```

Expected: 5 passed

- [ ] **Step 5: Protocol uyumunu doğrula**

```bash
python -c "
from src.core.interfaces import Chunker
from src.adapters.chunkers.recursive import RecursiveChunker
assert isinstance(RecursiveChunker(), Chunker)
print('OK')
"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/adapters/chunkers/recursive.py tests/unit/test_chunker.py
git commit -m "feat(chunker): recursive character splitter adapter"
```

---

## Task 6: TXT Document Loader

**Files:**
- Create: `src/adapters/loaders/txt_loader.py`
- Create: `tests/fixtures/sample.txt`
- Test: `tests/unit/test_txt_loader.py`

- [ ] **Step 1: Test fixture dosyasını oluştur**

`tests/fixtures/sample.txt`:

```
Bu bir test dosyasıdır.

İkinci paragraf Türkçe karakterler içerir: ğüşiöçİĞÜŞÖÇ.

Üçüncü paragraf İngilizce: The quick brown fox.
```

- [ ] **Step 2: Failing testi yaz**

`tests/unit/test_txt_loader.py`:

```python
"""TXT loader testleri."""
from pathlib import Path

import pytest

from src.adapters.loaders.txt_loader import TxtLoader


def test_txt_loader_reads_utf8(fixtures_dir: Path):
    loader = TxtLoader()
    path = fixtures_dir / "sample.txt"
    content_bytes = path.read_bytes()

    doc = loader.load("sample.txt", content_bytes)

    assert doc.name == "sample.txt"
    assert "ğüşiöçİĞÜŞÖÇ" in doc.content
    assert "quick brown fox" in doc.content
    assert doc.mime_type == "text/plain"
    assert doc.id  # non-empty UUID
    assert doc.metadata["size_bytes"] == len(content_bytes)


def test_txt_loader_empty_file():
    loader = TxtLoader()
    doc = loader.load("empty.txt", b"")
    assert doc.content == ""
    assert doc.metadata["size_bytes"] == 0


def test_txt_loader_handles_cp1254_fallback():
    """Latin-5 encoded Türkçe content (UTF-8 decode fail'e uğrarsa cp1254 dene)."""
    loader = TxtLoader()
    # "özel" cp1254'te
    content = "özel".encode("cp1254")
    doc = loader.load("tr.txt", content)
    assert "özel" in doc.content
```

- [ ] **Step 3: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_txt_loader.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: `src/adapters/loaders/txt_loader.py` yaz**

```python
"""TXT loader — UTF-8 öncelikli, cp1254 fallback."""
from __future__ import annotations

import uuid

from src.core.interfaces import Document, DocumentLoader


class TxtLoader:
    """`.txt` dosyalarını metne dönüştürür."""

    def load(self, filename: str, content: bytes) -> Document:
        text = self._decode(content)
        return Document(
            id=str(uuid.uuid4()),
            name=filename,
            content=text,
            mime_type="text/plain",
            metadata={"size_bytes": len(content)},
        )

    @staticmethod
    def _decode(content: bytes) -> str:
        for encoding in ("utf-8", "cp1254", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="replace")
```

- [ ] **Step 5: Testleri çalıştır**

```bash
pytest tests/unit/test_txt_loader.py -v
```

Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/adapters/loaders/txt_loader.py tests/unit/test_txt_loader.py tests/fixtures/sample.txt
git commit -m "feat(loaders): TXT loader with encoding fallback"
```

---

## Task 7: PDF Document Loader

**Files:**
- Create: `src/adapters/loaders/pdf_loader.py`
- Create: `tests/fixtures/sample.pdf`
- Test: `tests/unit/test_pdf_loader.py`

- [ ] **Step 1: Test fixture (sample PDF) oluştur**

Python script ile basit bir PDF üretelim (pypdf kullanarak veya reportlab ile).

```bash
source .venv/bin/activate
pip install reportlab
python -c "
from reportlab.pdfgen import canvas
c = canvas.Canvas('tests/fixtures/sample.pdf')
c.drawString(100, 750, 'Birinci sayfa: Türkçe test içeriği.')
c.showPage()
c.drawString(100, 750, 'Second page: English test content.')
c.showPage()
c.save()
print('sample.pdf created')
"
pip uninstall -y reportlab  # artık gerekmez
```

- [ ] **Step 2: Failing testi yaz**

`tests/unit/test_pdf_loader.py`:

```python
"""PDF loader testleri."""
from pathlib import Path

from src.adapters.loaders.pdf_loader import PdfLoader


def test_pdf_loader_extracts_text(fixtures_dir: Path):
    loader = PdfLoader()
    path = fixtures_dir / "sample.pdf"
    content_bytes = path.read_bytes()

    doc = loader.load("sample.pdf", content_bytes)

    assert doc.name == "sample.pdf"
    assert doc.mime_type == "application/pdf"
    # İçerikten en az bir kelime olmalı
    assert len(doc.content.strip()) > 0


def test_pdf_loader_concatenates_pages(fixtures_dir: Path):
    loader = PdfLoader()
    content_bytes = (fixtures_dir / "sample.pdf").read_bytes()
    doc = loader.load("sample.pdf", content_bytes)

    # İki farklı sayfanın içeriği birleşmiş olmalı
    assert "Birinci" in doc.content or "Türkçe" in doc.content
    assert "English" in doc.content or "Second" in doc.content
    assert doc.metadata.get("page_count", 0) >= 2


def test_pdf_loader_corrupt_bytes_raises():
    import pytest

    loader = PdfLoader()
    with pytest.raises(ValueError, match="parse"):
        loader.load("broken.pdf", b"not a real pdf")
```

- [ ] **Step 3: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_pdf_loader.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: `src/adapters/loaders/pdf_loader.py` yaz**

```python
"""PDF loader — pypdf ile doğrudan."""
from __future__ import annotations

import io
import uuid

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.core.interfaces import Document


class PdfLoader:
    """`.pdf` dosyalarını metne dönüştürür (pypdf ile)."""

    def load(self, filename: str, content: bytes) -> Document:
        try:
            reader = PdfReader(io.BytesIO(content))
        except PdfReadError as e:
            raise ValueError(f"PDF parse failed for {filename}: {e}") from e
        except Exception as e:  # pypdf bazen generic Exception
            raise ValueError(f"PDF parse failed for {filename}: {e}") from e

        pages_text: list[str] = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                pages_text.append("")

        full_text = "\n\n".join(p for p in pages_text if p.strip())

        return Document(
            id=str(uuid.uuid4()),
            name=filename,
            content=full_text,
            mime_type="application/pdf",
            metadata={
                "size_bytes": len(content),
                "page_count": len(reader.pages),
            },
        )
```

**Not:** `pypdf` bozuk PDF'lerde bazen `PdfReadError` değil başka hatalar da atabiliyor; o yüzden genel `Exception`'ı `ValueError`'a wrap'liyoruz.

- [ ] **Step 5: Testleri çalıştır**

```bash
pytest tests/unit/test_pdf_loader.py -v
```

Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/adapters/loaders/pdf_loader.py tests/unit/test_pdf_loader.py tests/fixtures/sample.pdf
git commit -m "feat(loaders): PDF loader using pypdf"
```

---

## Task 8: DOCX Document Loader

**Files:**
- Create: `src/adapters/loaders/docx_loader.py`
- Create: `tests/fixtures/sample.docx`
- Test: `tests/unit/test_docx_loader.py`

- [ ] **Step 1: Test fixture (sample DOCX) oluştur**

```bash
source .venv/bin/activate
python -c "
from docx import Document as DocxDocument
doc = DocxDocument()
doc.add_heading('Test Dökümanı', level=1)
doc.add_paragraph('Birinci paragraf Türkçe içerir: ğüşçö.')
doc.add_paragraph('Second paragraph in English.')
doc.save('tests/fixtures/sample.docx')
print('sample.docx created')
"
```

- [ ] **Step 2: Failing testi yaz**

`tests/unit/test_docx_loader.py`:

```python
"""DOCX loader testleri."""
from pathlib import Path

import pytest

from src.adapters.loaders.docx_loader import DocxLoader


def test_docx_loader_extracts_text(fixtures_dir: Path):
    loader = DocxLoader()
    content_bytes = (fixtures_dir / "sample.docx").read_bytes()
    doc = loader.load("sample.docx", content_bytes)

    assert doc.name == "sample.docx"
    assert doc.mime_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert "Test Dökümanı" in doc.content or "Dökümanı" in doc.content
    assert "Birinci" in doc.content
    assert "English" in doc.content


def test_docx_loader_corrupt_bytes_raises():
    loader = DocxLoader()
    with pytest.raises(ValueError, match="parse"):
        loader.load("bad.docx", b"not a real docx")
```

- [ ] **Step 3: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_docx_loader.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: `src/adapters/loaders/docx_loader.py` yaz**

```python
"""DOCX loader — python-docx (docx2txt fallback)."""
from __future__ import annotations

import io
import uuid

import docx2txt

from src.core.interfaces import Document


class DocxLoader:
    """`.docx` dosyalarını metne dönüştürür."""

    DOCX_MIME = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    def load(self, filename: str, content: bytes) -> Document:
        try:
            # docx2txt BytesIO üzerinde çalışır; python-docx'e göre daha basit
            text = docx2txt.process(io.BytesIO(content))
        except Exception as e:
            raise ValueError(f"DOCX parse failed for {filename}: {e}") from e

        return Document(
            id=str(uuid.uuid4()),
            name=filename,
            content=text or "",
            mime_type=self.DOCX_MIME,
            metadata={"size_bytes": len(content)},
        )
```

- [ ] **Step 5: Testleri çalıştır**

```bash
pytest tests/unit/test_docx_loader.py -v
```

Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/adapters/loaders/docx_loader.py tests/unit/test_docx_loader.py tests/fixtures/sample.docx
git commit -m "feat(loaders): DOCX loader using docx2txt"
```

---

## Task 9: DOC Document Loader (Graceful Fallback)

**Files:**
- Create: `src/adapters/loaders/doc_loader.py`
- Test: `tests/unit/test_doc_loader.py`

- [ ] **Step 1: Failing testi yaz**

`tests/unit/test_doc_loader.py`:

```python
"""DOC loader testleri."""
import pytest

from src.adapters.loaders.doc_loader import DocLoader, DocSupportUnavailable


def test_doc_loader_returns_clear_error_without_antiword(monkeypatch):
    """textract/antiword yoksa DocSupportUnavailable atılır."""
    loader = DocLoader()

    # textract'i yok gibi davran
    def _fail(*args, **kwargs):
        raise OSError("antiword binary not found")

    monkeypatch.setattr("src.adapters.loaders.doc_loader._extract_with_textract", _fail)

    with pytest.raises(DocSupportUnavailable) as exc_info:
        loader.load("eski.doc", b"\xd0\xcf\x11\xe0some old doc binary")

    # Mesaj kullanıcıya kurulum ipucu vermeli
    assert "antiword" in str(exc_info.value) or ".docx" in str(exc_info.value)


def test_doc_loader_returns_document_when_antiword_works(monkeypatch):
    """textract başarılı olursa Document döndür."""
    loader = DocLoader()

    def _ok(content: bytes) -> str:
        return "parsed content from .doc"

    monkeypatch.setattr("src.adapters.loaders.doc_loader._extract_with_textract", _ok)

    doc = loader.load("eski.doc", b"whatever")
    assert doc.content == "parsed content from .doc"
    assert doc.mime_type == "application/msword"
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_doc_loader.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: `src/adapters/loaders/doc_loader.py` yaz**

```python
"""DOC (eski Word binary) loader.

textract paketi arka planda `antiword` sistem binary'sini kullanır.
Bulunmazsa DocSupportUnavailable hatası verir ve kullanıcıya kurulum ipucu sunar.
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from src.core.interfaces import Document


class DocSupportUnavailable(RuntimeError):
    """`.doc` parse edilemedi — antiword/textract eksik."""


def _extract_with_textract(content: bytes) -> str:
    """textract ile geçici dosyaya yazıp parse et.

    ImportError veya OSError durumunda caller yakalar.
    """
    import textract

    with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        raw = textract.process(tmp_path)
        return raw.decode("utf-8", errors="replace")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class DocLoader:
    """`.doc` dosyaları için parser; sistem antiword kuruluysa çalışır."""

    DOC_MIME = "application/msword"

    def load(self, filename: str, content: bytes) -> Document:
        try:
            text = _extract_with_textract(content)
        except ImportError as e:
            raise DocSupportUnavailable(
                "'.doc' desteği için `textract` paketi gerekli. "
                "`pip install textract` komutunu çalıştırın veya dosyayı `.docx` formatına çevirin."
            ) from e
        except (OSError, Exception) as e:  # textract bazen generic hata atıyor
            raise DocSupportUnavailable(
                f"'.doc' parse edilemedi ({e}). "
                "Sistem `antiword` kurulu olmalı: `brew install antiword` (macOS) "
                "veya `apt install antiword` (Linux). "
                "Alternatif olarak dosyayı `.docx` formatına çevirin."
            ) from e

        return Document(
            id=str(uuid.uuid4()),
            name=filename,
            content=text,
            mime_type=self.DOC_MIME,
            metadata={"size_bytes": len(content)},
        )
```

- [ ] **Step 4: Testleri çalıştır**

```bash
pytest tests/unit/test_doc_loader.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/adapters/loaders/doc_loader.py tests/unit/test_doc_loader.py
git commit -m "feat(loaders): DOC loader with graceful antiword fallback"
```

---

## Task 10: Loader Factory — `get_loader(mime_type)`

**Files:**
- Create: `src/adapters/loaders/__init__.py`
- Test: `tests/unit/test_loader_factory.py`

- [ ] **Step 1: Failing testi yaz**

`tests/unit/test_loader_factory.py`:

```python
"""Loader factory testleri."""
import pytest

from src.adapters.loaders import get_loader, UnsupportedFileType
from src.adapters.loaders.docx_loader import DocxLoader
from src.adapters.loaders.pdf_loader import PdfLoader
from src.adapters.loaders.txt_loader import TxtLoader
from src.adapters.loaders.doc_loader import DocLoader


def test_factory_returns_txt_loader_for_plain_text():
    assert isinstance(get_loader("text/plain"), TxtLoader)


def test_factory_returns_pdf_loader_for_pdf():
    assert isinstance(get_loader("application/pdf"), PdfLoader)


def test_factory_returns_docx_loader_for_docx():
    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert isinstance(get_loader(mime), DocxLoader)


def test_factory_returns_doc_loader_for_doc():
    assert isinstance(get_loader("application/msword"), DocLoader)


def test_factory_by_extension_fallback():
    # MIME bilinmiyorsa filename ile eşleştir
    assert isinstance(get_loader(None, filename="report.pdf"), PdfLoader)
    assert isinstance(get_loader(None, filename="notes.txt"), TxtLoader)


def test_factory_unsupported_raises():
    with pytest.raises(UnsupportedFileType):
        get_loader("image/png", filename="screenshot.png")
    with pytest.raises(UnsupportedFileType):
        get_loader(None, filename="archive.zip")
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_loader_factory.py -v
```

Expected: FAIL — `ImportError: cannot import name 'get_loader'`

- [ ] **Step 3: `src/adapters/loaders/__init__.py` yaz**

```python
"""Document loader factory.

MIME type (veya fallback: dosya uzantısı) bakarak uygun DocumentLoader'ı döndürür.
"""
from __future__ import annotations

from pathlib import Path

from src.core.interfaces import DocumentLoader

from .doc_loader import DocLoader
from .docx_loader import DocxLoader
from .pdf_loader import PdfLoader
from .txt_loader import TxtLoader


class UnsupportedFileType(ValueError):
    """Yüklenen dosya formatı desteklenmiyor."""


_MIME_TO_LOADER: dict[str, type[DocumentLoader]] = {
    "text/plain": TxtLoader,
    "application/pdf": PdfLoader,
    "application/msword": DocLoader,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxLoader,
}

_EXT_TO_LOADER: dict[str, type[DocumentLoader]] = {
    ".txt": TxtLoader,
    ".pdf": PdfLoader,
    ".doc": DocLoader,
    ".docx": DocxLoader,
}


def get_loader(
    mime_type: str | None,
    filename: str | None = None,
) -> DocumentLoader:
    """MIME type veya dosya uzantısına göre loader döndür."""
    if mime_type and mime_type in _MIME_TO_LOADER:
        return _MIME_TO_LOADER[mime_type]()

    if filename:
        ext = Path(filename).suffix.lower()
        if ext in _EXT_TO_LOADER:
            return _EXT_TO_LOADER[ext]()

    raise UnsupportedFileType(
        f"Desteklenmeyen dosya formatı: mime={mime_type!r}, filename={filename!r}. "
        "Destek: .txt, .pdf, .doc, .docx"
    )


__all__ = ["get_loader", "UnsupportedFileType"]
```

- [ ] **Step 4: Testleri çalıştır**

```bash
pytest tests/unit/test_loader_factory.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/adapters/loaders/__init__.py tests/unit/test_loader_factory.py
git commit -m "feat(loaders): factory to resolve loader by MIME or extension"
```

---

## Task 11: Gemini Embedding Adapter

**Files:**
- Create: `src/adapters/embeddings/__init__.py`
- Create: `src/adapters/embeddings/gemini.py`
- Test: `tests/unit/test_gemini_embedding.py`

> **Not:** Gerçek Gemini API çağrıları integration testtir; Faz 1'de unit test için mock'lanır. Gerçek çağrı Task 28'in manuel E2E testinde yapılır.

- [ ] **Step 1: Failing testi yaz (mocked)**

`tests/unit/test_gemini_embedding.py`:

```python
"""Gemini embedding adapter testleri (mocked)."""
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.embeddings.gemini import GeminiEmbedding


@pytest.fixture
def mock_genai():
    with patch("src.adapters.embeddings.gemini.genai") as m:
        yield m


def test_gemini_embedding_batches_texts(mock_genai):
    mock_genai.embed_content.return_value = {
        "embedding": [0.1] * 768,
    }

    embedder = GeminiEmbedding(api_key="fake-key", model="text-embedding-004")
    vectors = embedder.embed(["metin 1", "metin 2"])

    assert len(vectors) == 2
    assert all(len(v) == 768 for v in vectors)
    assert mock_genai.embed_content.call_count == 2


def test_gemini_embedding_dimension_property():
    embedder = GeminiEmbedding(api_key="fake-key", model="text-embedding-004")
    # text-embedding-004 → 768
    assert embedder.dimension == 768


def test_gemini_embedding_empty_input(mock_genai):
    embedder = GeminiEmbedding(api_key="fake-key", model="text-embedding-004")
    assert embedder.embed([]) == []
    mock_genai.embed_content.assert_not_called()
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_gemini_embedding.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: `src/adapters/embeddings/gemini.py` yaz**

```python
"""Gemini embedding adapter (google-generativeai doğrudan)."""
from __future__ import annotations

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.interfaces import EmbeddingProvider


# text-embedding-004 → 768 boyutlu vektör döner
_MODEL_DIMENSIONS = {
    "text-embedding-004": 768,
    "models/text-embedding-004": 768,
}


class GeminiEmbedding:
    """Google Gemini embedding API wrapper."""

    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        genai.configure(api_key=api_key)
        self._model = model if model.startswith("models/") else f"models/{model}"
        self._short_name = self._model.removeprefix("models/")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _embed_one(self, text: str) -> list[float]:
        result = genai.embed_content(
            model=self._model,
            content=text,
            task_type="RETRIEVAL_DOCUMENT",
        )
        return list(result["embedding"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Gemini'nin batch endpoint'i de var ama sürüm uyumsuzluğu riski;
        # tek tek çağırmak basit ve güvenli (tenacity retry ile).
        return [self._embed_one(t) for t in texts]

    @property
    def dimension(self) -> int:
        return _MODEL_DIMENSIONS.get(self._short_name, 768)
```

- [ ] **Step 4: `src/adapters/embeddings/__init__.py` yaz**

```python
"""Embedding providers."""
from .gemini import GeminiEmbedding

__all__ = ["GeminiEmbedding"]
```

- [ ] **Step 5: Testleri çalıştır**

```bash
pytest tests/unit/test_gemini_embedding.py -v
```

Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/adapters/embeddings/__init__.py src/adapters/embeddings/gemini.py \
        tests/unit/test_gemini_embedding.py
git commit -m "feat(embeddings): Gemini embedding adapter with retry"
```

---

## Task 12: Chroma Vector Store Adapter

**Files:**
- Create: `src/adapters/vectorstores/__init__.py`
- Create: `src/adapters/vectorstores/chroma.py`
- Test: `tests/unit/test_chroma_store.py`

- [ ] **Step 1: Failing testi yaz**

Chroma in-memory mode (PersistentClient'ın `Client()` varyantı) testler için ideal.

`tests/unit/test_chroma_store.py`:

```python
"""Chroma store testleri (in-memory mode)."""
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
    assert results[0].chunk.id == "d1-c0"  # en yakın


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
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_chroma_store.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: `src/adapters/vectorstores/chroma.py` yaz**

```python
"""Chroma PersistentClient adapter."""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.interfaces import Chunk, RetrievedChunk


class ChromaVectorStore:
    """Chroma persistent store; disk-backed."""

    def __init__(self, path: str, collection_name: str = "documents"):
        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks ve embeddings eşit uzunlukta olmalı")

        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "document_name": c.document_name,
                    "chunk_index": c.index,
                    "char_start": c.metadata.get("char_start", 0),
                    "char_end": c.metadata.get("char_end", 0),
                }
                for c in chunks
            ],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        where = None
        if document_ids:
            where = {"document_id": {"$in": document_ids}}

        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        ids = raw.get("ids", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]

        results: list[RetrievedChunk] = []
        for _id, dist, content, meta in zip(ids, distances, documents, metadatas):
            score = max(0.0, 1.0 - dist)  # cosine distance → similarity
            chunk = Chunk(
                id=_id,
                document_id=meta["document_id"],
                document_name=meta["document_name"],
                content=content,
                index=int(meta.get("chunk_index", 0)),
                metadata={
                    "char_start": int(meta.get("char_start", 0)),
                    "char_end": int(meta.get("char_end", 0)),
                },
            )
            results.append(RetrievedChunk(chunk=chunk, score=score))
        return results

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def list_documents(self) -> list[dict]:
        raw = self._collection.get()  # tüm kayıtları çek
        metadatas = raw.get("metadatas", []) or []

        by_doc: dict[str, dict] = {}
        for meta in metadatas:
            doc_id = meta["document_id"]
            if doc_id not in by_doc:
                by_doc[doc_id] = {
                    "id": doc_id,
                    "name": meta["document_name"],
                    "chunk_count": 0,
                }
            by_doc[doc_id]["chunk_count"] += 1
        return list(by_doc.values())
```

- [ ] **Step 4: `src/adapters/vectorstores/__init__.py` yaz**

```python
"""Vector store providers."""
from .chroma import ChromaVectorStore

__all__ = ["ChromaVectorStore"]
```

- [ ] **Step 5: Testleri çalıştır**

```bash
pytest tests/unit/test_chroma_store.py -v
```

Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/adapters/vectorstores/__init__.py src/adapters/vectorstores/chroma.py \
        tests/unit/test_chroma_store.py
git commit -m "feat(vectorstore): Chroma persistent store adapter"
```

---

## Task 13: Gemini LLM Adapter

**Files:**
- Create: `src/adapters/llm/__init__.py`
- Create: `src/adapters/llm/gemini.py`
- Test: `tests/unit/test_gemini_llm.py`

- [ ] **Step 1: Failing testi yaz**

`tests/unit/test_gemini_llm.py`:

```python
"""Gemini LLM adapter testleri (mocked)."""
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.llm.gemini import GeminiLLM
from src.core.interfaces import Chunk, RetrievedChunk


def _retrieved(content: str, name: str = "d.txt") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id="c1", document_id="d1", document_name=name,
            content=content, index=0, metadata={},
        ),
        score=0.9,
    )


@pytest.fixture
def mock_genai():
    with patch("src.adapters.llm.gemini.genai") as m:
        yield m


def test_llm_generates_answer(mock_genai):
    fake_response = MagicMock()
    fake_response.text = "fake answer text"
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = fake_response
    mock_genai.GenerativeModel.return_value = mock_model_instance

    llm = GeminiLLM(api_key="k", model="gemini-2.5-flash")
    answer = llm.generate("soru?", context=[_retrieved("context 1"), _retrieved("context 2")])

    assert answer == "fake answer text"
    mock_model_instance.generate_content.assert_called_once()
    # prompt parametresinde context ve soru bulunmalı
    prompt_arg = mock_model_instance.generate_content.call_args.args[0]
    assert "context 1" in prompt_arg
    assert "soru?" in prompt_arg


def test_llm_model_name_property():
    llm = GeminiLLM(api_key="k", model="gemini-2.5-flash")
    assert llm.model_name == "gemini-2.5-flash"


def test_llm_handles_empty_context(mock_genai):
    fake_response = MagicMock()
    fake_response.text = "no context answer"
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = fake_response
    mock_genai.GenerativeModel.return_value = mock_model_instance

    llm = GeminiLLM(api_key="k", model="gemini-2.5-flash")
    answer = llm.generate("soru?", context=[])
    assert answer == "no context answer"
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_gemini_llm.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: `src/core/prompts.py` yaz**

```python
"""LLM prompt template'leri — tek merkez."""

QA_SYSTEM_PROMPT = """Sen yalnızca sana verilen dokümanlardan bilgi kullanarak cevap veren bir asistansın.
Eğer cevap dokümanlarda yoksa, kesinlikle uydurma; "Bu soru için yeterli bilgi bulunamadı" de.
Cevabını soru hangi dildeyse o dilde ver (Türkçe soru → Türkçe cevap, English question → English answer)."""


def build_qa_prompt(question: str, context_blocks: list[str]) -> str:
    """Q&A için tam prompt oluştur."""
    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "(Doküman sağlanmadı)"
    return f"""{QA_SYSTEM_PROMPT}

Dokümanlar:
{context_text}

Soru: {question}

Cevap:"""


def build_summarization_prompt(document_name: str, content: str) -> str:
    """Doküman özeti için prompt oluştur."""
    return f"""Aşağıdaki doküman içeriğinin kapsamlı ama öz bir özetini çıkar.
Ana temaları, önemli bulgularını ve sonuçları içersin.
Eğer doküman Türkçe ise özeti Türkçe, İngilizce ise İngilizce yaz.

Doküman adı: {document_name}
İçerik:
{content}

Özet:"""
```

- [ ] **Step 4: `src/adapters/llm/gemini.py` yaz**

```python
"""Gemini LLM adapter."""
from __future__ import annotations

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.interfaces import RetrievedChunk
from src.core.prompts import build_qa_prompt


class GeminiLLM:
    """Google Gemini generative text model wrapper."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self._model_name = model
        self._model = genai.GenerativeModel(model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, prompt: str, context: list[RetrievedChunk]) -> str:
        context_blocks = [
            f"[{r.chunk.document_name} — chunk {r.chunk.index}]\n{r.chunk.content}"
            for r in context
        ]
        full_prompt = build_qa_prompt(question=prompt, context_blocks=context_blocks)
        response = self._model.generate_content(full_prompt)
        return response.text or ""

    @property
    def model_name(self) -> str:
        return self._model_name
```

- [ ] **Step 5: `src/adapters/llm/__init__.py` yaz**

```python
"""LLM providers."""
from .gemini import GeminiLLM

__all__ = ["GeminiLLM"]
```

- [ ] **Step 6: Testleri çalıştır**

```bash
pytest tests/unit/test_gemini_llm.py -v
```

Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add src/core/prompts.py src/adapters/llm/__init__.py src/adapters/llm/gemini.py \
        tests/unit/test_gemini_llm.py
git commit -m "feat(llm): Gemini LLM adapter with prompt templates"
```

---

## Task 14: RAG Pipeline — Ingestion

**Files:**
- Create: `src/rag/__init__.py`
- Create: `src/rag/pipeline.py`
- Test: `tests/unit/test_pipeline_ingest.py`

- [ ] **Step 1: Failing testi yaz**

`tests/unit/test_pipeline_ingest.py`:

```python
"""RAGPipeline ingestion testleri — fake providers'la."""
from src.adapters.loaders import get_loader
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
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_pipeline_ingest.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.rag.pipeline'`

- [ ] **Step 3: `src/rag/pipeline.py` yaz (sadece `__init__` + `ingest`)**

```python
"""RAG Pipeline orchestrator.

Ingestion: loader → chunker → embedder → store.
Retrieval+Generation: embedder → store.search → llm.
"""
from __future__ import annotations

from typing import Callable

from src.core.interfaces import (
    Answer,
    Chunker,
    Document,
    DocumentLoader,
    EmbeddingProvider,
    LLMProvider,
    RetrievedChunk,
    VectorStore,
)


LoaderFactory = Callable[[str | None, str | None], DocumentLoader]
"""MIME type ve/veya filename'den DocumentLoader döndüren fonksiyon."""


class RAGPipeline:
    """Tüm RAG akışını koordine eden orchestrator."""

    def __init__(
        self,
        loader_factory: LoaderFactory,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LLMProvider,
    ):
        self._loader_factory = loader_factory
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._llm = llm

    # ---- ingestion ----

    def ingest(
        self,
        filename: str,
        content: bytes,
        mime_type: str | None = None,
    ) -> Document:
        """Dosyayı parse et, chunk'la, embed'le, store'a yaz."""
        loader = self._loader_factory(mime_type, filename)
        document = loader.load(filename, content)

        chunks = self._chunker.chunk(document)
        if not chunks:
            return document  # boş doküman; no-op

        embeddings = self._embedder.embed([c.content for c in chunks])
        self._store.add(chunks, embeddings)

        return document
```

- [ ] **Step 4: `src/rag/__init__.py` yaz**

```python
"""RAG orchestration layer."""
from .pipeline import RAGPipeline

__all__ = ["RAGPipeline"]
```

- [ ] **Step 5: `get_loader` signature uyumluluğu**

`src/adapters/loaders/__init__.py` `get_loader(mime_type, filename=None)` olduğu için `LoaderFactory` signature'ı uyumlu. Pipeline testinde `_loader_factory(mime_type, filename=None)` kullanıyoruz. Bu yüzden `LoaderFactory` tipi `Callable[[str | None, str | None], DocumentLoader]` olmalı (yukarıda öyle). Başka değişiklik gerekmez.

- [ ] **Step 6: Testleri çalıştır**

```bash
pytest tests/unit/test_pipeline_ingest.py -v
```

Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
git add src/rag/__init__.py src/rag/pipeline.py tests/unit/test_pipeline_ingest.py
git commit -m "feat(rag): pipeline ingestion (loader → chunker → embedder → store)"
```

---

## Task 15: RAG Pipeline — Answer (Q&A)

**Files:**
- Modify: `src/rag/pipeline.py` (add `answer()` + internal retrieval helper)
- Test: `tests/unit/test_pipeline_answer.py`

- [ ] **Step 1: Failing testi yaz**

`tests/unit/test_pipeline_answer.py`:

```python
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
    # Store'u elle doldur
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

    # Sadece A'dan kaynak gelmeli
    for src in answer.sources:
        assert src.chunk.document_id == "A"
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_pipeline_answer.py -v
```

Expected: FAIL — `AttributeError: 'RAGPipeline' object has no attribute 'answer'`

- [ ] **Step 3: `src/rag/pipeline.py`'a `answer()` ekle**

Dosyanın sonuna (RAGPipeline class'ının içine) ekle:

```python
    # ---- retrieval + generation ----

    def answer(
        self,
        question: str,
        document_ids: list[str] | None = None,
        top_k: int = 4,
    ) -> Answer:
        """Soru → retrieve → LLM → Answer."""
        query_vec = self._embedder.embed([question])[0]
        retrieved = self._store.search(
            query_embedding=query_vec,
            top_k=top_k,
            document_ids=document_ids,
        )
        generated_text = self._llm.generate(question, context=retrieved)
        return Answer(
            text=generated_text,
            sources=retrieved,
            model=self._llm.model_name,
        )
```

- [ ] **Step 4: Testleri çalıştır**

```bash
pytest tests/unit/test_pipeline_answer.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/rag/pipeline.py tests/unit/test_pipeline_answer.py
git commit -m "feat(rag): pipeline answer (retrieve → LLM → Answer)"
```

---

## Task 16: RAG Pipeline — Summarize

**Files:**
- Modify: `src/rag/pipeline.py` (add `summarize()`)
- Modify: `src/core/prompts.py` (summarization prompt zaten var — Task 13'te eklenmişti)
- Test: `tests/unit/test_pipeline_summarize.py`

- [ ] **Step 1: Failing testi yaz**

`tests/unit/test_pipeline_summarize.py`:

```python
"""RAGPipeline summarize testleri."""
from src.core.interfaces import Chunk
from src.rag.pipeline import RAGPipeline


def test_summarize_uses_all_chunks_of_document(
    fake_chunker, fake_embedder, fake_store, fake_llm, fake_loader
):
    # Elle chunk'lar ekle
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
    # LLM'e D1'in tüm 3 chunk'ı gönderilmeli
    _, context = fake_llm.call_log[-1]
    assert len(context) == 3
    assert all(rc.chunk.document_id == "D1" for rc in context)


def test_summarize_missing_document_raises(
    fake_chunker, fake_embedder, fake_store, fake_llm, fake_loader
):
    import pytest
    pipeline = RAGPipeline(
        loader_factory=lambda m, f=None: fake_loader,
        chunker=fake_chunker,
        embedder=fake_embedder,
        store=fake_store,
        llm=fake_llm,
    )
    with pytest.raises(ValueError, match="not found"):
        pipeline.summarize(document_id="NONEXISTENT")
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_pipeline_summarize.py -v
```

Expected: FAIL — `AttributeError: 'RAGPipeline' object has no attribute 'summarize'`

- [ ] **Step 3: `VectorStore` Protocol'e yeni bir metod ekle**

Özet için "bir dokümana ait tüm chunk'ları sıra ile getir" gerekiyor. Bunu `search` ile yapmak mümkün değil (çünkü relevance'a göre sıralıyor). Protocol'e ekleyelim.

`src/core/interfaces.py`'te `VectorStore` Protocol'üne ekle:

```python
    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        """Bir dokümana ait tüm chunk'ları index sırasında döndür."""
        ...
```

- [ ] **Step 4: `tests/fakes.py`'te `FakeVectorStore`'a ekle**

```python
    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        items = [i for i in self._items if i.chunk.document_id == document_id]
        return sorted([i.chunk for i in items], key=lambda c: c.index)
```

- [ ] **Step 5: `src/adapters/vectorstores/chroma.py`'te ekle**

`ChromaVectorStore` class'ının içine ekle:

```python
    def get_document_chunks(self, document_id: str):
        from src.core.interfaces import Chunk
        raw = self._collection.get(where={"document_id": document_id})

        ids = raw.get("ids", []) or []
        documents = raw.get("documents", []) or []
        metadatas = raw.get("metadatas", []) or []

        chunks: list[Chunk] = []
        for _id, content, meta in zip(ids, documents, metadatas):
            chunks.append(
                Chunk(
                    id=_id,
                    document_id=meta["document_id"],
                    document_name=meta["document_name"],
                    content=content,
                    index=int(meta.get("chunk_index", 0)),
                    metadata={
                        "char_start": int(meta.get("char_start", 0)),
                        "char_end": int(meta.get("char_end", 0)),
                    },
                )
            )
        chunks.sort(key=lambda c: c.index)
        return chunks
```

- [ ] **Step 6: `src/rag/pipeline.py`'a `summarize()` ekle**

Ayrıca Chroma/ FakeStore test'lerinin bozulmadığını doğrulamalıyız.

`src/rag/pipeline.py`'a:

```python
    def summarize(self, document_id: str) -> Answer:
        """Bir dokümanın tüm chunk'larını LLM'e vererek özet üret."""
        chunks = self._store.get_document_chunks(document_id)
        if not chunks:
            raise ValueError(f"Document not found or has no chunks: {document_id}")

        # tüm chunk'ları RetrievedChunk'a wrap'le (score=1.0, kaynak olarak kullanmak için)
        retrieved = [RetrievedChunk(chunk=c, score=1.0) for c in chunks]

        # LLM'e "özet çıkar" promptu ile ver
        document_name = chunks[0].document_name
        full_content = "\n\n".join(c.content for c in chunks)

        from src.core.prompts import build_summarization_prompt
        prompt = build_summarization_prompt(document_name=document_name, content=full_content)

        generated = self._llm.generate(prompt, context=retrieved)
        return Answer(
            text=generated,
            sources=retrieved,
            model=self._llm.model_name,
        )
```

- [ ] **Step 7: Testleri çalıştır**

```bash
pytest tests/unit/test_pipeline_summarize.py tests/unit/test_chroma_store.py -v
```

Expected: Hepsi pass

- [ ] **Step 8: Tüm pipeline testlerini smoke test et**

```bash
pytest tests/unit/ -v
```

Expected: Hepsi pass

- [ ] **Step 9: Commit**

```bash
git add src/core/interfaces.py src/rag/pipeline.py \
        src/adapters/vectorstores/chroma.py tests/fakes.py \
        tests/unit/test_pipeline_summarize.py
git commit -m "feat(rag): pipeline summarize with per-document chunk retrieval"
```

---

## Task 17: FastAPI Foundation + Dependencies + Health

**Files:**
- Create: `backend/main.py`
- Create: `backend/dependencies.py`
- Create: `backend/schemas.py`
- Create: `backend/routes/__init__.py`
- Create: `backend/routes/health.py`
- Test: `tests/api/test_health.py`

- [ ] **Step 1: Failing testi yaz**

`tests/api/test_health.py`:

```python
"""Health endpoint testi."""
from fastapi.testclient import TestClient

from backend.main import create_app


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
pytest tests/api/test_health.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: `backend/schemas.py` yaz**

```python
"""Pydantic request/response modelleri."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---- Health ----

class HealthResponse(BaseModel):
    status: str
    version: str


# ---- Documents ----

class DocumentInfo(BaseModel):
    id: str
    name: str
    chunk_count: int


class UploadResult(BaseModel):
    document: DocumentInfo


class UploadResponse(BaseModel):
    results: list[UploadResult]
    total_chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    deleted_document_id: str


# ---- Query ----

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    document_ids: list[str] | None = None
    top_k: int = Field(4, ge=1, le=20)


class SourceCitation(BaseModel):
    document_id: str
    document_name: str
    chunk_index: int
    chunk_preview: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    model: str
    retrieval_count: int


# ---- Summarize ----

class SummarizeRequest(BaseModel):
    document_id: str


class SummarizeResponse(BaseModel):
    summary: str
    document_id: str
    document_name: str
    model: str


# ---- Error ----

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
```

- [ ] **Step 4: `backend/dependencies.py` yaz**

```python
"""Dependency injection: RAGPipeline'ı FastAPI endpoint'lerine sun."""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from backend.config import Settings, get_settings
from src.adapters.chunkers.recursive import RecursiveChunker
from src.adapters.embeddings.gemini import GeminiEmbedding
from src.adapters.llm.gemini import GeminiLLM
from src.adapters.loaders import get_loader as loader_factory_fn
from src.adapters.vectorstores.chroma import ChromaVectorStore
from src.rag.pipeline import RAGPipeline


@lru_cache(maxsize=1)
def _build_pipeline(settings_hash: str) -> RAGPipeline:
    """Settings'in string representasyonuyla cache'le (tek instance yeterli)."""
    settings = get_settings()
    chunker = RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedder = GeminiEmbedding(
        api_key=settings.gemini_api_key,
        model=settings.gemini_embedding_model,
    )
    store = ChromaVectorStore(path=settings.chroma_path)
    llm = GeminiLLM(
        api_key=settings.gemini_api_key,
        model=settings.gemini_llm_model,
    )
    return RAGPipeline(
        loader_factory=loader_factory_fn,
        chunker=chunker,
        embedder=embedder,
        store=store,
        llm=llm,
    )


def get_pipeline(settings: Settings = Depends(get_settings)) -> RAGPipeline:
    """Endpoint'lere RAGPipeline inject et."""
    # settings_hash'i lru_cache key'i olarak kullan
    return _build_pipeline(settings.chroma_path + ":" + settings.gemini_llm_model)
```

- [ ] **Step 5: `backend/routes/health.py` yaz**

```python
"""Health check endpoint."""
from fastapi import APIRouter

from backend.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0-mvp")
```

- [ ] **Step 6: `backend/routes/__init__.py` yaz**

```python
"""Routers."""
from . import health  # diğer route'lar sonraki task'larda eklenecek

__all__ = ["health"]
```

- [ ] **Step 7: `backend/main.py` yaz**

```python
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
```

- [ ] **Step 8: Testleri çalıştır**

```bash
# GEMINI_API_KEY yoksa test başlarken config hatası alırız; dummy değer set et
GEMINI_API_KEY=test-key pytest tests/api/test_health.py -v
```

Expected: 1 passed

- [ ] **Step 9: Manuel smoke test — uvicorn ile ayağa kaldır**

```bash
# Ayrı terminalde:
GEMINI_API_KEY=test-key uvicorn backend.main:app --reload
# Sonra curl ile:
curl http://localhost:8000/health
```

Expected: `{"status":"ok","version":"0.1.0-mvp"}`

- [ ] **Step 10: Commit**

```bash
git add backend/main.py backend/dependencies.py backend/schemas.py \
        backend/routes/__init__.py backend/routes/health.py \
        tests/api/test_health.py
git commit -m "feat(backend): FastAPI app with health endpoint and DI"
```

---

## Task 18: `/upload` Endpoint

**Files:**
- Create: `backend/routes/documents.py`
- Modify: `backend/routes/__init__.py`
- Modify: `backend/main.py`
- Test: `tests/api/test_upload.py`

- [ ] **Step 1: Failing testi yaz**

`tests/api/test_upload.py`:

```python
"""Upload endpoint testi — dependency override ile."""
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.dependencies import get_pipeline
from src.rag.pipeline import RAGPipeline
from tests.fakes import (
    FakeChunker, FakeDocumentLoader, FakeEmbeddingProvider,
    FakeLLMProvider, FakeVectorStore,
)


def _fake_pipeline():
    return RAGPipeline(
        loader_factory=lambda m, f=None: FakeDocumentLoader(),
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
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
GEMINI_API_KEY=test-key pytest tests/api/test_upload.py -v
```

Expected: FAIL — 404 veya endpoint not found

- [ ] **Step 3: `backend/routes/documents.py` yaz**

```python
"""Document endpoint'leri: upload, list, delete."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.dependencies import get_pipeline
from backend.schemas import (
    DocumentInfo,
    UploadResponse,
    UploadResult,
)
from src.adapters.loaders import UnsupportedFileType
from src.rag.pipeline import RAGPipeline

router = APIRouter(prefix="", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: list[UploadFile] = File(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> UploadResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="En az bir dosya yüklemelisiniz.",
        )

    results: list[UploadResult] = []
    total_chunks = 0

    for upload in files:
        content = await upload.read()
        try:
            doc = pipeline.ingest(
                filename=upload.filename,
                content=content,
                mime_type=upload.content_type,
            )
        except UnsupportedFileType as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{upload.filename}: {e}",
            ) from e

        # chunk sayısını store'dan al
        docs = pipeline._store.list_documents()  # pragma: no cover -  pipeline.store is private; see Task 19 for cleaner API
        chunk_count = next(
            (d["chunk_count"] for d in docs if d["id"] == doc.id), 0
        )
        total_chunks += chunk_count

        results.append(
            UploadResult(
                document=DocumentInfo(
                    id=doc.id, name=doc.name, chunk_count=chunk_count,
                )
            )
        )

    return UploadResponse(results=results, total_chunks=total_chunks)
```

**Not:** `pipeline._store` private member erişimi hoş değil. Task 19'da `RAGPipeline`'a `list_documents()` gibi pass-through metodları ekleyeceğiz. Şimdilik bu geçici.

- [ ] **Step 4: `backend/routes/__init__.py`'yi güncelle**

```python
"""Routers."""
from . import documents, health

__all__ = ["documents", "health"]
```

- [ ] **Step 5: `backend/main.py`'ye router'ı ekle**

```python
# include_router(health.router) altına:
from backend.routes import documents
app.include_router(documents.router)
```

- [ ] **Step 6: Testleri çalıştır**

```bash
GEMINI_API_KEY=test-key pytest tests/api/test_upload.py -v
```

Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add backend/routes/documents.py backend/routes/__init__.py backend/main.py \
        tests/api/test_upload.py
git commit -m "feat(api): POST /upload endpoint with multi-file support"
```

---

## Task 19: `/documents` ve `/documents/{id}` Endpoint'leri + Pipeline Pass-Through

**Files:**
- Modify: `src/rag/pipeline.py` (add `list_documents()`, `delete_document()`)
- Modify: `backend/routes/documents.py` (add GET /documents, DELETE /documents/{id})
- Modify: `backend/routes/documents.py` (upload'daki private access'i temizle)
- Test: `tests/api/test_documents.py`

- [ ] **Step 1: Pipeline'a pass-through metodları ekle**

`src/rag/pipeline.py`'a:

```python
    # ---- document management ----

    def list_documents(self) -> list[dict]:
        """Kayıtlı dokümanları özet bilgileriyle döndür."""
        return self._store.list_documents()

    def delete_document(self, document_id: str) -> None:
        """Dokümanı ve tüm chunk'larını sil."""
        self._store.delete_document(document_id)
```

- [ ] **Step 2: `upload_files`'daki private access'i temizle**

`backend/routes/documents.py`'te `pipeline._store.list_documents()`'ı `pipeline.list_documents()` ile değiştir.

- [ ] **Step 3: Failing testi yaz**

`tests/api/test_documents.py`:

```python
"""Documents list/delete endpoint testleri."""
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.dependencies import get_pipeline
from src.rag.pipeline import RAGPipeline
from tests.fakes import (
    FakeChunker, FakeDocumentLoader, FakeEmbeddingProvider,
    FakeLLMProvider, FakeVectorStore,
)


def _fake_pipeline():
    return RAGPipeline(
        loader_factory=lambda m, f=None: FakeDocumentLoader(),
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
    # upload
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


def test_delete_nonexistent_document_204():
    """Idempotent delete: olmayan id için de 200 dön (hata atma)."""
    pipeline = _fake_pipeline()
    client = _client_with_pipeline(pipeline)
    response = client.delete("/documents/nonexistent-id")
    # Chroma silme idempotent; 200 kabul edilebilir
    assert response.status_code == 200
```

- [ ] **Step 4: `backend/routes/documents.py`'e endpoint'leri ekle**

```python
from backend.schemas import (
    DeleteResponse,
    DocumentListResponse,
)


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> DocumentListResponse:
    raw = pipeline.list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in raw]
    )


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(
    document_id: str,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> DeleteResponse:
    pipeline.delete_document(document_id)
    return DeleteResponse(deleted_document_id=document_id)
```

- [ ] **Step 5: Testleri çalıştır**

```bash
GEMINI_API_KEY=test-key pytest tests/api/test_documents.py tests/api/test_upload.py -v
```

Expected: Tümü pass

- [ ] **Step 6: Commit**

```bash
git add src/rag/pipeline.py backend/routes/documents.py tests/api/test_documents.py
git commit -m "feat(api): GET /documents and DELETE /documents/{id} endpoints"
```

---

## Task 20: `/query` Endpoint

**Files:**
- Create: `backend/routes/query.py`
- Modify: `backend/routes/__init__.py`, `backend/main.py`
- Test: `tests/api/test_query.py`

- [ ] **Step 1: Failing testi yaz**

`tests/api/test_query.py`:

```python
"""Query endpoint testi."""
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.dependencies import get_pipeline
from src.rag.pipeline import RAGPipeline
from tests.fakes import (
    FakeChunker, FakeDocumentLoader, FakeEmbeddingProvider,
    FakeLLMProvider, FakeVectorStore,
)


def _fake_pipeline(answer: str = "canned answer"):
    return RAGPipeline(
        loader_factory=lambda m, f=None: FakeDocumentLoader(),
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
    # önce içerik yükle
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
    # İki doküman yükle
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
    # Sadece A'dan kaynak olmalı
    for src in response.json()["sources"]:
        assert src["document_id"] == doc_a_id
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
GEMINI_API_KEY=test-key pytest tests/api/test_query.py -v
```

Expected: FAIL — endpoint yok

- [ ] **Step 3: `backend/routes/query.py` yaz**

```python
"""Query endpoint'leri: /query, /summarize."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.dependencies import get_pipeline
from backend.schemas import (
    QueryRequest,
    QueryResponse,
    SourceCitation,
)
from src.rag.pipeline import RAGPipeline

router = APIRouter(prefix="", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    answer = pipeline.answer(
        question=req.question,
        document_ids=req.document_ids,
        top_k=req.top_k,
    )
    sources = [
        SourceCitation(
            document_id=s.chunk.document_id,
            document_name=s.chunk.document_name,
            chunk_index=s.chunk.index,
            chunk_preview=s.chunk.content[:200],
            score=round(s.score, 4),
        )
        for s in answer.sources
    ]
    return QueryResponse(
        answer=answer.text,
        sources=sources,
        model=answer.model,
        retrieval_count=len(answer.sources),
    )
```

- [ ] **Step 4: Router'ı bağla**

`backend/routes/__init__.py`:
```python
from . import documents, health, query
__all__ = ["documents", "health", "query"]
```

`backend/main.py`'de `app.include_router(documents.router)` altına ekle:
```python
from backend.routes import query as query_routes
app.include_router(query_routes.router)
```

- [ ] **Step 5: Testleri çalıştır**

```bash
GEMINI_API_KEY=test-key pytest tests/api/test_query.py -v
```

Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add backend/routes/query.py backend/routes/__init__.py backend/main.py tests/api/test_query.py
git commit -m "feat(api): POST /query endpoint with source citations"
```

---

## Task 21: `/summarize` Endpoint

**Files:**
- Modify: `backend/routes/query.py` (add /summarize)
- Test: `tests/api/test_summarize.py`

- [ ] **Step 1: Failing testi yaz**

`tests/api/test_summarize.py`:

```python
"""Summarize endpoint testi."""
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.dependencies import get_pipeline
from src.rag.pipeline import RAGPipeline
from tests.fakes import (
    FakeChunker, FakeDocumentLoader, FakeEmbeddingProvider,
    FakeLLMProvider, FakeVectorStore,
)


def _pipeline(answer: str = "özet"):
    return RAGPipeline(
        loader_factory=lambda m, f=None: FakeDocumentLoader(),
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
        files=[("files", ("rapor.txt", b"Uzun bir doküman içeriği x" * 20, "text/plain"))],
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
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
GEMINI_API_KEY=test-key pytest tests/api/test_summarize.py -v
```

Expected: FAIL

- [ ] **Step 3: `/summarize`'ı `query.py`'e ekle**

```python
# backend/routes/query.py'e aşağıdakileri ekle:
from fastapi import HTTPException, status

from backend.schemas import SummarizeRequest, SummarizeResponse


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(
    req: SummarizeRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> SummarizeResponse:
    try:
        answer = pipeline.summarize(document_id=req.document_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e

    # document_name: sources'un ilk chunk'ından al
    doc_name = answer.sources[0].chunk.document_name if answer.sources else "(unknown)"
    return SummarizeResponse(
        summary=answer.text,
        document_id=req.document_id,
        document_name=doc_name,
        model=answer.model,
    )
```

- [ ] **Step 4: Testleri çalıştır**

```bash
GEMINI_API_KEY=test-key pytest tests/api/test_summarize.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add backend/routes/query.py tests/api/test_summarize.py
git commit -m "feat(api): POST /summarize endpoint"
```

---

## Task 22: Upload Size Limit + Global Exception Handler

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Failing testi yaz**

`tests/api/test_upload.py`'e ekle:

```python
def test_upload_size_limit(monkeypatch):
    """MAX_UPLOAD_MB aşıldığında 413 dön."""
    # config'i 1MB'a düşür
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    # pipeline cache'i temizle
    from backend.dependencies import _build_pipeline
    _build_pipeline.cache_clear()

    client, _ = _client_with_fake_pipeline()
    big_content = b"x" * (2 * 1024 * 1024)  # 2MB
    response = client.post("/upload", files=[("files", ("big.txt", big_content, "text/plain"))])
    assert response.status_code == 413
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
GEMINI_API_KEY=test-key pytest tests/api/test_upload.py::test_upload_size_limit -v
```

Expected: FAIL — request aslında geçiyor, 413 gelmiyor

- [ ] **Step 3: `backend/main.py`'e middleware ekle**

```python
# create_app() içine CORS'tan sonra:
from fastapi import Request
from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def limit_upload_size(request: Request, call_next):
        if request.url.path == "/upload" and request.method == "POST":
            cl = request.headers.get("content-length")
            if cl and int(cl) > settings.max_upload_mb * 1024 * 1024:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "PayloadTooLarge",
                        "detail": f"Toplam upload {settings.max_upload_mb}MB'ı aşamaz.",
                    },
                )
        return await call_next(request)
```

- [ ] **Step 4: Testleri çalıştır**

```bash
GEMINI_API_KEY=test-key pytest tests/api/ -v
```

Expected: Tümü pass

- [ ] **Step 5: Commit**

```bash
git add backend/main.py tests/api/test_upload.py
git commit -m "feat(api): global upload size limit middleware"
```

---

## Task 23: Streamlit API Client

**Files:**
- Create: `frontend/api_client.py`
- Test: `tests/unit/test_api_client.py`

- [ ] **Step 1: Failing testi yaz**

`tests/unit/test_api_client.py`:

```python
"""Frontend API client testleri (requests-mock ile)."""
from unittest.mock import patch, MagicMock

import pytest

from frontend.api_client import APIClient


@pytest.fixture
def client():
    return APIClient(base_url="http://localhost:8000")


def test_health_check(client):
    with patch("frontend.api_client.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": "ok", "version": "0.1.0-mvp"},
        )
        result = client.health()
        assert result["status"] == "ok"


def test_list_documents(client):
    with patch("frontend.api_client.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"documents": [{"id": "a", "name": "a.txt", "chunk_count": 3}]},
        )
        docs = client.list_documents()
        assert len(docs) == 1
        assert docs[0]["name"] == "a.txt"


def test_query(client):
    with patch("frontend.api_client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "answer": "test",
                "sources": [],
                "model": "gemini-2.5-flash",
                "retrieval_count": 0,
            },
        )
        result = client.query(question="test?", document_ids=None, top_k=4)
        assert result["answer"] == "test"
```

- [ ] **Step 2: Testi çalıştır ve fail'i gör**

```bash
pytest tests/unit/test_api_client.py -v
```

Expected: FAIL

- [ ] **Step 3: `frontend/api_client.py` yaz**

```python
"""Streamlit frontend için FastAPI backend HTTP client."""
from __future__ import annotations

from typing import IO

import requests


class APIClientError(RuntimeError):
    """API çağrısı başarısız."""


class APIClient:
    """FastAPI endpoint'lerini çağıran basit HTTP wrapper."""

    def __init__(self, base_url: str, timeout_seconds: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds

    # ---- system ----

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        return self._handle(resp)

    # ---- documents ----

    def upload_files(self, files: list[tuple[str, IO, str]]) -> dict:
        """files = [(filename, file_obj, mime_type), ...]."""
        multipart = [("files", (name, fobj, mime)) for name, fobj, mime in files]
        resp = requests.post(
            f"{self.base_url}/upload",
            files=multipart,
            timeout=self.timeout,
        )
        return self._handle(resp)

    def list_documents(self) -> list[dict]:
        resp = requests.get(f"{self.base_url}/documents", timeout=self.timeout)
        return self._handle(resp).get("documents", [])

    def delete_document(self, document_id: str) -> dict:
        resp = requests.delete(
            f"{self.base_url}/documents/{document_id}", timeout=self.timeout,
        )
        return self._handle(resp)

    # ---- query ----

    def query(
        self,
        question: str,
        document_ids: list[str] | None,
        top_k: int = 4,
    ) -> dict:
        payload = {"question": question, "top_k": top_k}
        if document_ids:
            payload["document_ids"] = document_ids
        resp = requests.post(
            f"{self.base_url}/query", json=payload, timeout=self.timeout,
        )
        return self._handle(resp)

    def summarize(self, document_id: str) -> dict:
        resp = requests.post(
            f"{self.base_url}/summarize",
            json={"document_id": document_id},
            timeout=self.timeout,
        )
        return self._handle(resp)

    # ---- internal ----

    @staticmethod
    def _handle(resp: requests.Response) -> dict:
        if not resp.ok:
            raise APIClientError(
                f"{resp.status_code} {resp.reason}: {resp.text[:500]}"
            )
        return resp.json()
```

- [ ] **Step 4: Testleri çalıştır**

```bash
pytest tests/unit/test_api_client.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add frontend/api_client.py tests/unit/test_api_client.py
git commit -m "feat(frontend): APIClient wrapping backend REST calls"
```

---

## Task 24: Streamlit Upload Component

**Files:**
- Create: `frontend/components/upload.py`
- Modify: `frontend/components/__init__.py`

- [ ] **Step 1: `frontend/components/upload.py` yaz**

```python
"""Dosya yükleme Streamlit component'i."""
from __future__ import annotations

import mimetypes
from io import BytesIO

import streamlit as st

from frontend.api_client import APIClient, APIClientError


def render_upload(client: APIClient) -> None:
    """Dosya yükleme UI bölümü."""
    st.subheader("📥 Doküman Yükle")
    uploaded_files = st.file_uploader(
        "Bir veya birden fazla doküman yükleyin",
        type=["txt", "pdf", "doc", "docx"],
        accept_multiple_files=True,
        key="uploader",
    )

    if uploaded_files and st.button("Yükle", type="primary", key="upload-btn"):
        files_payload = []
        for f in uploaded_files:
            mime, _ = mimetypes.guess_type(f.name)
            mime = mime or "application/octet-stream"
            files_payload.append((f.name, BytesIO(f.getvalue()), mime))

        with st.spinner("Dokümanlar işleniyor..."):
            try:
                result = client.upload_files(files_payload)
                st.success(
                    f"✅ {len(result['results'])} doküman yüklendi, "
                    f"{result['total_chunks']} chunk oluşturuldu."
                )
            except APIClientError as e:
                st.error(f"❌ Yükleme başarısız: {e}")
```

- [ ] **Step 2: `frontend/components/__init__.py`'yi güncelle**

```python
"""Streamlit UI components."""
from .upload import render_upload

__all__ = ["render_upload"]
```

- [ ] **Step 3: Manuel smoke test (import)**

```bash
python -c "from frontend.components.upload import render_upload; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add frontend/components/upload.py frontend/components/__init__.py
git commit -m "feat(frontend): file upload component"
```

---

## Task 25: Streamlit Documents Management Component

**Files:**
- Create: `frontend/components/documents.py`
- Modify: `frontend/components/__init__.py`

- [ ] **Step 1: `frontend/components/documents.py` yaz**

```python
"""Doküman listesi / silme / özet UI component'i."""
from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClient, APIClientError


def render_documents(client: APIClient) -> list[dict]:
    """Doküman listesi, silme ve özet butonları. Döndürür: güncel liste."""
    st.subheader("📚 Yüklü Dokümanlar")

    try:
        docs = client.list_documents()
    except APIClientError as e:
        st.error(f"Doküman listesi alınamadı: {e}")
        return []

    if not docs:
        st.info("Henüz doküman yüklenmedi.")
        return []

    for doc in docs:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.markdown(f"**{doc['name']}**  \n_{doc['chunk_count']} chunk_")
            if c2.button("Özet", key=f"sum-{doc['id']}"):
                with st.spinner("Özet üretiliyor..."):
                    try:
                        s = client.summarize(doc["id"])
                        st.info(f"**Özet ({doc['name']}):**\n\n{s['summary']}")
                    except APIClientError as e:
                        st.error(f"Özet başarısız: {e}")
            if c3.button("Sil", key=f"del-{doc['id']}"):
                try:
                    client.delete_document(doc["id"])
                    st.success(f"{doc['name']} silindi.")
                    st.rerun()
                except APIClientError as e:
                    st.error(f"Silme başarısız: {e}")

    return docs
```

- [ ] **Step 2: `frontend/components/__init__.py`'yi güncelle**

```python
"""Streamlit UI components."""
from .documents import render_documents
from .upload import render_upload

__all__ = ["render_documents", "render_upload"]
```

- [ ] **Step 3: Import smoke test**

```bash
python -c "from frontend.components.documents import render_documents; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add frontend/components/documents.py frontend/components/__init__.py
git commit -m "feat(frontend): documents list/delete/summarize component"
```

---

## Task 26: Streamlit Chat Component

**Files:**
- Create: `frontend/components/chat.py`
- Modify: `frontend/components/__init__.py`

- [ ] **Step 1: `frontend/components/chat.py` yaz**

```python
"""Chat UI component'i."""
from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClient, APIClientError


def render_chat(client: APIClient, docs: list[dict]) -> None:
    """Chat arayüzü: soru-cevap + kaynak gösterimi."""
    st.subheader("💬 Sohbet")

    # Doküman seçimi (opsiyonel filtre)
    doc_options = {d["id"]: d["name"] for d in docs}
    selected_ids: list[str] | None = None
    if doc_options:
        selected_names = st.multiselect(
            "Sadece seçili dokümanlarda ara (boşsa tümünde ara)",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
        )
        selected_ids = selected_names or None

    # Session history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Geçmişi göster
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Kaynaklar"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**{s['document_name']}** (chunk {s['chunk_index']}, "
                            f"skor: {s['score']:.3f})\n\n> {s['chunk_preview']}"
                        )

    # Input
    if prompt := st.chat_input("Dokümanlara soru sorun..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Düşünüyor..."):
                try:
                    result = client.query(
                        question=prompt,
                        document_ids=selected_ids,
                        top_k=4,
                    )
                except APIClientError as e:
                    st.error(f"Soru başarısız: {e}")
                    return

            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Kaynaklar"):
                    for s in result["sources"]:
                        st.markdown(
                            f"**{s['document_name']}** (chunk {s['chunk_index']}, "
                            f"skor: {s['score']:.3f})\n\n> {s['chunk_preview']}"
                        )

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )
```

- [ ] **Step 2: `frontend/components/__init__.py`'yi güncelle**

```python
"""Streamlit UI components."""
from .chat import render_chat
from .documents import render_documents
from .upload import render_upload

__all__ = ["render_chat", "render_documents", "render_upload"]
```

- [ ] **Step 3: Import smoke test**

```bash
python -c "from frontend.components.chat import render_chat; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add frontend/components/chat.py frontend/components/__init__.py
git commit -m "feat(frontend): chat UI with sources expansion"
```

---

## Task 27: Streamlit Main App Integration

**Files:**
- Create: `frontend/app.py`

- [ ] **Step 1: `frontend/app.py` yaz**

```python
"""Streamlit entry point: 'Chat With Your Documents'."""
from __future__ import annotations

import os

import streamlit as st

from frontend.api_client import APIClient, APIClientError
from frontend.components import render_chat, render_documents, render_upload


def main() -> None:
    st.set_page_config(page_title="Chat With Your Docs", page_icon="💬", layout="wide")
    st.title("💬 Chat With Your Documents")
    st.caption("Dokümanlarınızı yükleyin, onlara soru sorun.")

    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    client = APIClient(base_url=backend_url)

    # Health check
    try:
        client.health()
    except APIClientError:
        st.error(
            f"⚠️ Backend'e ulaşılamıyor: {backend_url}\n\n"
            "`uvicorn backend.main:app --reload` ile backend'i başlattığınızdan emin olun."
        )
        st.stop()

    # Layout: Sol kolonda upload + docs, sağda chat
    left, right = st.columns([1, 2], gap="large")
    with left:
        render_upload(client)
        st.divider()
        docs = render_documents(client)
    with right:
        render_chat(client, docs)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Import smoke test**

```bash
python -c "from frontend.app import main; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add frontend/app.py
git commit -m "feat(frontend): Streamlit main app integration"
```

---

## Task 28: End-to-End Manual Test + README

**Files:**
- Create: `README.md`

> **Ön gereksinim:** Google AI Studio'dan bir Gemini API key al. https://aistudio.google.com → API keys → Create API key. `.env` dosyasına `GEMINI_API_KEY=<sizin_key>` yaz.

- [ ] **Step 1: Gerçek API key ile backend'i başlat**

```bash
cd "/Users/fatiherencetin/Desktop/RAG Systems"
source .venv/bin/activate
# .env dosyası yoksa .env.example'dan kopyala ve GEMINI_API_KEY'i doldur
cp -n .env.example .env
# .env'yi düzenle: GEMINI_API_KEY=<senin_gerçek_key'in>

uvicorn backend.main:app --reload
```

Expected: `Uvicorn running on http://0.0.0.0:8000`

- [ ] **Step 2: Health check**

Ayrı terminalde:
```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok","version":"0.1.0-mvp"}`

- [ ] **Step 3: `/docs` sayfasını tarayıcıda aç**

`http://localhost:8000/docs` → Swagger UI'da tüm endpoint'ler görünmeli.

- [ ] **Step 4: Streamlit'i başlat (yeni terminal)**

```bash
cd "/Users/fatiherencetin/Desktop/RAG Systems"
source .venv/bin/activate
streamlit run frontend/app.py
```

Tarayıcı otomatik açılır: `http://localhost:8501`

- [ ] **Step 5: E2E smoke test — UI'den yap**

1. Sample PDF ve TXT yükle (`tests/fixtures/sample.pdf`, `tests/fixtures/sample.txt`)
2. "Yükle" butonuna bas → success mesajı görmeli
3. Doküman listesinde iki doküman görmeli
4. Chat'ten "Dokümanlarda ne anlatılıyor?" diye sor → cevap + kaynaklar gelmeli
5. Bir dokümanın "Özet" butonuna bas → özet gelmeli
6. Bir dokümanı "Sil" butonuyla sil → listeden kalkmalı, disk'te de silinmeli

Her adımda hata alırsan:
- Backend terminalinde traceback'e bak
- `.env` dosyasındaki `GEMINI_API_KEY`'i kontrol et
- `.venv` aktif olduğundan emin ol

- [ ] **Step 6: `README.md` yaz**

```markdown
# Chat With Your Documents

Dokümanlarınızı yükleyin ve AI ile sohbet edin. TXT, PDF, DOC, DOCX desteklenir.

Bu repo **RAG (Retrieval-Augmented Generation)** mantığını uygulayan bir challenge projesidir.

## Özellikler (v0.1 MVP)

- 📥 Çoklu doküman yükleme (TXT, PDF, DOC, DOCX)
- 🔪 Otomatik chunking + embedding (Gemini `text-embedding-004`)
- 🧠 Chroma vector DB ile lokal persist
- 💬 Gemini LLM (`gemini-2.5-flash`) ile soru-cevap
- 📝 Doküman özeti üretme
- 🎯 Kaynak gösterme (cevabın hangi chunk'tan geldiği)
- 🎛️ Seçili dokümanlar üzerinde sorgulama

## Mimari

**Hexagonal (Ports & Adapters):**
- `src/core/` — domain interfaces (Protocol)
- `src/adapters/` — somut sağlayıcılar (Gemini, Chroma, LangChain loaders)
- `src/rag/pipeline.py` — orchestrator
- `backend/` — FastAPI REST API
- `frontend/` — Streamlit UI

## Kurulum

### 1. Gereksinimler

- Python 3.11+
- Google Gemini API key ([Google AI Studio](https://aistudio.google.com))
- Opsiyonel: `.doc` dosyaları için `antiword` (`brew install antiword` / `apt install antiword`)

### 2. Kurulum

\`\`\`bash
git clone <repo-url>
cd "RAG Systems"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\`\`\`

### 3. Konfigürasyon

\`\`\`bash
cp .env.example .env
# .env dosyasını düzenle:
# GEMINI_API_KEY=<your_key>
\`\`\`

## Çalıştırma

İki terminal gerekir:

**Terminal 1 — Backend (FastAPI):**
\`\`\`bash
source .venv/bin/activate
uvicorn backend.main:app --reload
# http://localhost:8000/docs → Swagger UI
\`\`\`

**Terminal 2 — Frontend (Streamlit):**
\`\`\`bash
source .venv/bin/activate
streamlit run frontend/app.py
# http://localhost:8501
\`\`\`

## Test

\`\`\`bash
GEMINI_API_KEY=test-key pytest -v
\`\`\`

## API Endpoint'leri

| Method | Path | Açıklama |
|---|---|---|
| POST | /upload | Dosya yükleme (multipart) |
| POST | /query | Soru sor |
| POST | /summarize | Doküman özetini al |
| GET | /documents | Yüklü dokümanları listele |
| DELETE | /documents/{id} | Dokümanı sil |
| GET | /health | Health check |

## Yol Haritası

- **v0.1 (MVP, bu sürüm):** LangChain + saf SDK karma; Gemini only
- **v0.2 (planned):** Saf SDK (LangChain'siz), kapsamlı testler, temiz README
- **v0.3 (planned):** Docker, auth, streaming, HTML/JS frontend, deployment, opsiyonel HuggingFace/Groq

## Lisans

MIT
```

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs: README with setup and usage instructions"
```

- [ ] **Step 8: Tüm testleri son kez çalıştır**

```bash
source .venv/bin/activate
GEMINI_API_KEY=test-key pytest -v
```

Expected: Tüm testler pass. Eğer .doc testleri fail olursa (sistem antiword'suz): bu beklenen davranış (test monkeypatch kullanıyor, ama `textract` import'u fail edebiliyorsa test file başına `pytest.importorskip('textract')` ekle).

---

## Task 29: Final Git Tag — v0.1-mvp-langchain

**Files:** (hiçbiri)

- [ ] **Step 1: `git status` ile temiz tree doğrula**

```bash
git status
```

Expected: `nothing to commit, working tree clean`

- [ ] **Step 2: Tüm testlerin geçtiğini tekrar doğrula**

```bash
GEMINI_API_KEY=test-key pytest -v
```

Expected: Hepsi pass.

- [ ] **Step 3: Faz 1 Kabul Kriterlerini (spec §14) elle kontrol et**

Spec `docs/superpowers/specs/2026-04-19-chat-with-your-docs-design.md` §14'teki 12 maddenin hepsini işaretle:

1. ✅ `.env` konfigüre edilebilir → ✓ Task 2
2. ✅ `uvicorn backend.main:app` başlar → ✓ Task 28 Step 1
3. ✅ `streamlit run frontend/app.py` başlar → ✓ Task 28 Step 4
4. ✅ TXT/PDF/DOCX yüklenir → ✓ Task 28 Step 5
5. ✅ `.doc` için hata/destek → ✓ Task 9
6. ✅ Doküman list/delete → ✓ Task 19
7. ✅ Soru-cevap + kaynak → ✓ Task 20
8. ✅ Çoklu doküman seçerek soru → ✓ Task 26
9. ✅ Özet → ✓ Task 21
10. ✅ `/docs` API dökümantasyonu → ✓ Task 28 Step 3
11. ✅ En az 3 unit test → ✓ Çok daha fazlası mevcut (chunker, pipeline, adapters, api)
12. ✅ Git tag → şimdi ataceğız

- [ ] **Step 4: Git tag at**

```bash
git tag -a v0.1-mvp-langchain -m "Phase 1 MVP: LangChain hybrid, Gemini providers, Chroma, Streamlit UI"
```

- [ ] **Step 5: Tag'in doğru olduğunu verify et**

```bash
git tag -l -n1
# Expected: v0.1-mvp-langchain   Phase 1 MVP: ...
git log --oneline -n 5
```

Expected: Son commit(ler) görünmeli, tag'in üzerinde olduğu commit ile son commit aynı olmalı.

- [ ] **Step 6: (Opsiyonel) GitHub'a push**

Eğer GitHub remote henüz ayarlanmamışsa, ayarladıktan sonra:
```bash
git remote add origin git@github.com:<user>/<repo>.git  # ilk kez
git push -u origin main
git push origin v0.1-mvp-langchain
```

---

## Özet ve Sonraki Adımlar

**Faz 1 (MVP) tamamlandı:** Çalışan RAG uygulaması, 29 task, ~45+ commit, ~15+ unit test, 7 adapter, 1 pipeline, 5 API endpoint, 3 frontend component.

**Faz 2 için yeni plan yazılacak konular:**
- LangChain'siz refactor (loader'ları `pypdf`/`python-docx` saf SDK'ya çevir; splitter'ı kendin yaz)
- Test coverage'ı >%70'e çıkar
- CI (GitHub Actions) ekle
- README'yi production kalitesine getir (mimari diyagram, GIF demo, badge'ler)
- `v0.2-pure-sdk` tag'i

**Faz 3 için:**
- Docker/compose
- JWT auth + user management
- Streaming SSE
- HTML/JS frontend
- Deployment

Her yeni faz için ayrı brainstorming → spec → plan döngüsü çalıştırılacak.

---

**Bu plan onaylandığında:** `subagent-driven-development` (önerilen) veya `executing-plans` skill'i ile adım adım implementasyona geçilecek.
