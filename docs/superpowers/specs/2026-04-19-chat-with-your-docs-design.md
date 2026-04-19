# Chat With Your Documents — Tasarım Dokümanı (Spec)

**Tarih:** 2026-04-19
**Durum:** Taslak — Kullanıcı onayı bekliyor
**Kapsam:** Faz 1 (MVP) detaylı; Faz 2 (GitHub polish) ve Faz 3 (Production) yüksek seviye yol haritası

---

## 1. Projenin Amacı

Kullanıcının yüklediği dokümanlar (TXT, PDF, DOC, DOCX) üzerinden doğal dilde soru-cevap yapabilen, özet üretebilen ve birden fazla dokümanı birleştirerek cevaplayabilen bir **Retrieval-Augmented Generation (RAG)** uygulaması geliştirmek.

### Faz yapısı

| Faz | Hedef | Teslim |
|---|---|---|
| **Faz 1 — MVP** | Challenge teslimi, öğrenme odaklı çalışır prototip | Yerel çalışan uygulama, temel işlevler |
| **Faz 2 — GitHub polish** | Açık kaynak kalitesinde repo: saf SDK, README, testler | Public GitHub repo, `v0.2-pure-sdk` tag |
| **Faz 3 — Production** | Docker, auth, streaming, deployment | Canlı URL, HTML/JS frontend |

Bu doküman **Faz 1** implementasyonunu detaylı tanımlar. Faz 2 ve 3, mimari kararları etkilediği ölçüde anılır.

---

## 2. Hedefler ve Hedef Olmayanlar

### Faz 1 Hedefleri

- TXT, PDF, DOC, DOCX dosyalarının yüklenebilmesi ve metne dönüştürülmesi
- Birden fazla dokümanın aynı anda sisteme eklenebilmesi
- Chunking + embedding + Chroma'ya kayıt işlem hattının çalışması
- Kullanıcının yüklediği dokümanlar üzerinde soru sorabilmesi
- Doküman özetinin üretilebilmesi
- Birden fazla doküman üzerinden birleşik cevap üretilebilmesi
- **Kaynak gösterme** (hangi dokümandan, hangi chunk'tan geldiği) — Chroma metadata ile ücretsiz gelir
- FastAPI tabanlı REST API + Streamlit UI'ın çalışır halde olması
- Gemini sağlayıcısının devrede olması, ama başka sağlayıcı eklemek için soyut katmanın hazır olması

### Faz 1 Hedef Olmayanları (explicitly out of scope)

- Streaming response (SSE) — Faz 3
- Kullanıcı yönetimi, authentication — Faz 3
- Docker / docker-compose — Faz 3
- Deployment (Railway/Render/Fly.io) — Faz 3
- Gelişmiş UI (React, HTMX) — Faz 3
- OCR'lı taranmış PDF desteği — Faz 1'de scope dışı (parse edilemeyen dosyalar için net hata mesajı)
- Çoklu kullanıcı / workspace izolasyonu — Faz 3
- Kapsamlı pytest test suite — Faz 2 (Faz 1'de yalnızca kritik unit testler)

---

## 3. Mimari Genel Bakış

```
┌──────────────────────────────────────────────────────────────┐
│  Streamlit UI (frontend/app.py)                              │
│  - Dosya yükleme                                              │
│  - Chat arayüzü                                               │
│  - Doküman listesi + silme                                    │
│  HTTP Client (requests)                                       │
└──────────────────────────┬───────────────────────────────────┘
                           │ REST
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI Backend (backend/main.py)                            │
│  Endpoints: /upload, /query, /summarize, /documents,          │
│             /documents/{id} (DELETE), /health                 │
│  Pydantic Schemas (backend/schemas.py)                        │
└──────────────────────────┬───────────────────────────────────┘
                           │ depends on
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  RAG Pipeline (src/rag/pipeline.py)                           │
│  Orchestrator: ingestion + retrieval + generation             │
└──────────┬────────────────┬────────────────┬────────────────┘
           │                │                │
           ▼                ▼                ▼
    ┌──────────┐      ┌──────────┐     ┌──────────┐
    │ Chunker  │      │ Retriever│     │Generator │
    └────┬─────┘      └────┬─────┘     └────┬─────┘
         │                 │                │
         │ depends on the following Protocols (src/core/interfaces.py):
         │   - DocumentLoader  (TXT/PDF/DOCX parse)
         │   - Chunker         (text → chunks)
         │   - EmbeddingProvider
         │   - VectorStore
         │   - LLMProvider
         ▼
  Adapter Layer (src/adapters/)
  ┌────────────────────┬──────────────────┬──────────────────┐
  │ LangChain Loaders  │ Gemini SDK       │ Chroma Client    │
  │ (PDF/DOCX/TXT)     │ (LLM+Embedding)  │ (persist local)  │
  └────────────────────┴──────────────────┴──────────────────┘
```

### Mimari prensipler

1. **Ports & Adapters (Hexagonal):** Domain (`src/core/`) interface'leri tanımlar; dış dünyaya bağımlılık sadece `src/adapters/` içinde yaşar.
2. **Dependency Inversion:** API katmanı Protocol'lere bağımlıdır, somut sağlayıcılara değil.
3. **Configuration-driven:** Sağlayıcı seçimi `.env` üzerinden yapılır (`LLM_PROVIDER=gemini`), kodda hardcoded değildir.
4. **Stateless REST:** Backend state tutmaz; tüm persist Chroma'dadır. Bu, Faz 3'te yatay ölçeklendirmeyi mümkün kılar.

---

## 4. Core Interfaces (src/core/interfaces.py)

Faz 1'de yazılacak ve tüm sistemin bağımlı olacağı sözleşmeler:

```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class Document:
    """Ham yüklenmiş doküman (parse edilmiş tam metin)."""
    id: str                    # UUID
    name: str                  # "rapor_2026.pdf"
    content: str               # tam metin
    mime_type: str             # "application/pdf"
    metadata: dict             # {"upload_date": ..., "size_bytes": ...}

@dataclass
class Chunk:
    """Doküman'ın bir parçası, embedding'lenmeye hazır."""
    id: str                    # UUID
    document_id: str           # bağlı olduğu Document
    document_name: str         # citation için
    content: str               # chunk metni
    index: int                 # chunk sırası (0-based)
    metadata: dict             # {"char_start": ..., "char_end": ...}

@dataclass
class RetrievedChunk:
    """Retrieval sonucunda dönen chunk + skoru."""
    chunk: Chunk
    score: float               # cosine similarity [0, 1]

@dataclass
class Answer:
    """LLM'nin ürettiği cevap + kaynaklar."""
    text: str
    sources: list[RetrievedChunk]
    model: str                 # "gemini-2.5-flash"

class DocumentLoader(Protocol):
    """Raw bytes → Document (parse işlemi)."""
    def load(self, filename: str, content: bytes) -> Document: ...

class Chunker(Protocol):
    """Document → list[Chunk]."""
    def chunk(self, document: Document) -> list[Chunk]: ...

class EmbeddingProvider(Protocol):
    """list[str] → list[vector]."""
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: ...

class VectorStore(Protocol):
    """Chunk + embedding saklama, benzerlik araması."""
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]: ...
    def delete_document(self, document_id: str) -> None: ...
    def list_documents(self) -> list[dict]: ...

class LLMProvider(Protocol):
    """Prompt + context → generated text."""
    def generate(self, prompt: str, context: list[RetrievedChunk]) -> str: ...
    @property
    def model_name(self) -> str: ...
```

---

## 5. Adapter Katmanı (Faz 1 implementasyonları)

### 5.1 Document Loaders — `src/adapters/loaders/`

LangChain'in hazır loader'larından yararlanılacak (karma yaklaşımın LangChain parçası):

- **`txt_loader.py`** — Python stdlib yeterli (`decode('utf-8')`), LangChain gereksiz
- **`pdf_loader.py`** — `langchain_community.document_loaders.PyPDFLoader` veya doğrudan `pypdf`
- **`docx_loader.py`** — `langchain_community.document_loaders.Docx2txtLoader` veya doğrudan `python-docx`
- **`doc_loader.py`** — `.doc` (eski Word binary formatı) için iki aşamalı yaklaşım:
  1. **Tercih edilen:** `textract` Python paketi (arka planda sistem `antiword` binary'sini kullanır). README'de platform başına kurulum talimatı olur (`brew install antiword` / `apt install antiword`).
  2. **Fallback:** `textract` veya `antiword` bulunmazsa, kullanıcıya net mesaj döner: _"`.doc` desteği için antiword gerekli. Alternatif olarak dosyayı `.docx`'e çevirip tekrar yükleyin."_
  Bu yaklaşım challenge şartnamesini (`.doc` desteği zorunlu) karşılar, aynı zamanda bağımlılık gereksinimi yetersiz sistemlerde kullanıcıyı yalnız bırakmaz.

**Tasarım kararı:** Her loader `DocumentLoader` Protocol'ünü uygular; LangChain'in `Document` tipi loader'ın içinde kalır, **dışarıya kendi `Document` tipimizi döndürürüz.** Faz 2'de LangChain'i kaldırmak bu dosyaları yeniden yazmak anlamına gelir; API, pipeline, schemalar etkilenmez.

Factory: `get_loader(mime_type: str) -> DocumentLoader`

### 5.2 Chunker — `src/adapters/chunkers/recursive.py`

LangChain'in `RecursiveCharacterTextSplitter` wrapper'ı:
- `chunk_size=1000` karakter (Türkçe için yeterli, ~200-250 token)
- `chunk_overlap=200` (bağlamın korunması için)
- Separator'lar: `["\n\n", "\n", ". ", "! ", "? ", " ", ""]`

Output'u kendi `Chunk` tipimize dönüştürür.

### 5.3 Embedding Provider — `src/adapters/embeddings/gemini.py`

Google Gemini SDK'sı doğrudan kullanılır (LangChain yok):
- Model: `text-embedding-004` (768 boyut, multilingual)
- Batch size: 100 (API limiti)
- Rate limit handling: `tenacity` ile exponential backoff

### 5.4 Vector Store — `src/adapters/vectorstores/chroma.py`

`chromadb.PersistentClient(path=settings.chroma_path)`:
- Tek collection: `documents`
- Metadata fields: `document_id`, `document_name`, `chunk_index`
- `where` filter ile `document_ids` parametresi filtrelenir

### 5.5 LLM Provider — `src/adapters/llm/gemini.py`

`google-generativeai` SDK doğrudan kullanılır:
- Model: `gemini-2.5-flash` (varsayılan), config ile `gemini-2.5-pro` açılabilir
- System prompt'lar `src/core/prompts.py`'de sabitlenir
- Non-streaming (Faz 1); streaming Faz 3'te eklenecek

---

## 6. RAG Pipeline — `src/rag/pipeline.py`

Tek orchestrator class:

```python
class RAGPipeline:
    def __init__(
        self,
        loader_factory: Callable[[str], DocumentLoader],
        chunker: Chunker,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LLMProvider,
    ): ...

    def ingest(self, filename: str, content: bytes) -> Document:
        """Yükleme → parse → chunk → embed → store."""

    def answer(
        self,
        question: str,
        document_ids: list[str] | None = None,
        top_k: int = 4,
    ) -> Answer:
        """Soru → retrieve → LLM → Answer."""

    def summarize(
        self,
        document_id: str,
    ) -> Answer:
        """Tek dokümanın özetini çıkar (tüm chunk'ları context olarak verir)."""
```

### Özet mantığı
Dokümanın tüm chunk'ları `document_id` filtresi ile çekilir, LLM'e "özet çıkar" promptu ile verilir. Çok büyük dokümanlarda (>50 chunk) **map-reduce** pattern'ini Faz 1'de opsiyonel bırakıyoruz; önce basit versiyonu yazar, test ederken limit sorunu çıkarsa map-reduce'e geçeriz.

---

## 7. API Kontratı (FastAPI Endpoints)

Tüm endpoint'ler `backend/routes/`'da tanımlanır. Pydantic şemaları `backend/schemas.py`.

| Method | Path | Açıklama |
|---|---|---|
| `POST` | `/upload` | Multipart dosya yükleme (tek veya çoklu) |
| `POST` | `/query` | Soru sor (tüm veya seçili dokümanlarda) |
| `POST` | `/summarize` | Belirli dokümanın özetini al |
| `GET` | `/documents` | Yüklenmiş dokümanların listesi |
| `DELETE` | `/documents/{doc_id}` | Dokümanı ve chunk'larını sil |
| `GET` | `/health` | Health check |

### Örnek şemalar

```python
class UploadResponse(BaseModel):
    documents: list[DocumentInfo]
    chunk_count: int

class DocumentInfo(BaseModel):
    id: str
    name: str
    mime_type: str
    chunk_count: int
    uploaded_at: datetime

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    document_ids: list[str] | None = None
    top_k: int = Field(4, ge=1, le=20)

class SourceCitation(BaseModel):
    document_id: str
    document_name: str
    chunk_index: int
    chunk_preview: str     # ilk 200 karakter
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    model: str
    retrieval_count: int
```

### CORS ve güvenlik

- CORS origins `.env`'den okunur (`CORS_ORIGINS=http://localhost:8501`)
- Max upload size `.env`'den (`MAX_UPLOAD_MB=50`)
- Faz 1'de auth yok; tüm istekler public (yalnızca `localhost`)

---

## 8. Data Flow

### 8.1 Ingestion (dosya yüklendiğinde)

```
User → Streamlit (file uploader)
     → POST /upload (multipart)
     → FastAPI route
       → PipelineFactory.get_loader(mime_type) → parse
       → Chunker.chunk(document)
       → EmbeddingProvider.embed([chunk.content for chunk in chunks])
       → VectorStore.add(chunks, embeddings)
     ← UploadResponse
```

### 8.2 Query (kullanıcı soru sorduğunda)

```
User → Streamlit (chat input)
     → POST /query {question, document_ids?}
     → FastAPI route
       → EmbeddingProvider.embed([question])
       → VectorStore.search(query_vec, top_k=4, document_ids)
       → LLMProvider.generate(prompt, retrieved_chunks)
     ← QueryResponse {answer, sources}
```

---

## 9. Proje Klasör Yapısı

```
RAG Systems/
├── backend/                    # FastAPI uygulaması
│   ├── __init__.py
│   ├── main.py                 # app = FastAPI(...)
│   ├── config.py               # Settings (pydantic-settings)
│   ├── dependencies.py         # Dependency injection (get_pipeline)
│   ├── schemas.py              # Pydantic request/response modelleri
│   └── routes/
│       ├── __init__.py
│       ├── documents.py        # /upload, /documents, DELETE
│       ├── query.py            # /query, /summarize
│       └── health.py           # /health
│
├── src/                        # Domain ve adapter katmanları
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── interfaces.py       # Protocol'ler + dataclass'lar
│   │   └── prompts.py          # LLM system prompts
│   ├── rag/
│   │   ├── __init__.py
│   │   └── pipeline.py         # RAGPipeline orchestrator
│   └── adapters/
│       ├── __init__.py
│       ├── loaders/
│       │   ├── __init__.py     # get_loader(mime_type) factory
│       │   ├── txt_loader.py
│       │   ├── pdf_loader.py
│       │   ├── docx_loader.py
│       │   └── doc_loader.py
│       ├── chunkers/
│       │   ├── __init__.py
│       │   └── recursive.py    # LangChain wrapper
│       ├── embeddings/
│       │   ├── __init__.py
│       │   └── gemini.py
│       ├── vectorstores/
│       │   ├── __init__.py
│       │   └── chroma.py
│       └── llm/
│           ├── __init__.py
│           └── gemini.py
│
├── frontend/                   # Streamlit uygulaması
│   ├── app.py                  # ana entry point
│   ├── api_client.py           # FastAPI'ye HTTP istekleri
│   └── components/             # Streamlit component'leri
│       ├── __init__.py
│       ├── upload.py
│       ├── chat.py
│       └── documents.py
│
├── tests/                      # Faz 1'de minimum, Faz 2'de genişler
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_pipeline.py        # in-memory fake provider'larla
│   └── fixtures/
│       ├── sample.pdf
│       └── sample.docx
│
├── data/                       # Yerel Chroma persist (gitignore'da)
│   └── chroma_db/
│
├── docs/
│   └── superpowers/
│       └── specs/              # Bu dosya burada
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── pyproject.toml              # Python 3.11+, ruff, mypy config
```

---

## 10. Konfigürasyon — `.env`

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

`backend/config.py`'de `pydantic-settings` ile tip-güvenli okuma yapılır.

---

## 11. Dependencies (Faz 1)

`requirements.txt`:

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

# Document parsing (LangChain yardımcıları)
langchain-text-splitters>=0.2
langchain-community>=0.2
pypdf>=4.0
python-docx>=1.1
docx2txt>=0.8
textract>=1.6  # .doc desteği; sistem antiword gerektirir

# Frontend
streamlit>=1.35
requests>=2.31

# Dev (optional Phase 1, mandatory Phase 2)
pytest>=8.0
ruff>=0.3
```

Toplam ~15-20 paket. Faz 2'de LangChain kaldırıldığında `langchain-*` paketleri gider, saf `pypdf` + `python-docx` kalır.

---

## 12. Hata Yönetimi

### Yükleme hataları
- Desteklenmeyen dosya tipi → `400 Bad Request` + kullanıcıya net mesaj
- Çok büyük dosya → `413 Payload Too Large` + `MAX_UPLOAD_MB` değerini söyler
- Parse edilemeyen dosya (bozuk PDF vs) → `422 Unprocessable Entity` + dosya adı + parser hatası

### API / LLM hataları
- Gemini rate limit → `tenacity` ile 3 retry + exponential backoff, sonunda başarısızsa `503`
- Geçersiz API key → startup'ta health check, kullanıcıya `.env` hatası göster
- Embedding boyut uyumsuzluğu → startup'ta Chroma collection ile karşılaştır, boyut farklıysa hata ver

### Frontend (Streamlit) hataları
- Backend'e ulaşılamazsa → `st.error` ile kullanıcıya gösterilir
- API'den 4xx/5xx → hata mesajı chat'te görünür, işlem durur

---

## 13. Temel Prompt Template'leri (`src/core/prompts.py`)

### QA prompt

```
Sen yalnızca sana verilen dokümanlardan bilgi kullanarak cevap veren bir asistansın.
Eğer cevap dokümanlarda yoksa, kesinlikle uydurma; "Bu soru için yeterli bilgi bulunamadı" de.

Dokümanlar:
{context}

Soru: {question}

Cevabı doğrudan, net ve Türkçe (soru Türkçe ise) veya İngilizce (soru İngilizce ise) ver.
```

### Summarization prompt

```
Aşağıdaki doküman içeriğinin kapsamlı ama öz bir özetini çıkar.
Ana temaları, önemli bulgularını ve sonuçları içersin.

Doküman: {document_name}
İçerik:
{content}

Özet:
```

---

## 14. Faz 1 Kabul Kriterleri (Definition of Done)

Faz 1 aşağıdaki kriterler karşılandığında bitmiş sayılır:

1. ✅ `.env` konfigüre edilebilir, `GEMINI_API_KEY` ile sistem çalışır hale gelir
2. ✅ `uvicorn backend.main:app` ile backend ayağa kalkar
3. ✅ `streamlit run frontend/app.py` ile frontend ayağa kalkar
4. ✅ TXT, PDF, DOCX dosyaları yüklenebilir ve Chroma'ya kaydedilir
5. ✅ `.doc` dosyası parse edilebilir (sistem `antiword` kuruluysa) veya kullanıcıya kurulum talimatı gösteren net hata döner
6. ✅ Yüklenmiş dokümanlar listelenebilir ve silinebilir
7. ✅ Chat arayüzünden soru sorulur, cevap + en az 1 kaynak gösterilir
8. ✅ Çoklu doküman seçerek soru sorulabilir
9. ✅ Bir dokümanın özeti alınabilir
10. ✅ API dökümantasyonu `/docs` yolunda açılır ve endpoint'leri gösterir
11. ✅ En az 3 unit test geçer (chunker, pipeline ingestion, pipeline answer)
12. ✅ `git tag v0.1-mvp-langchain` atılarak Faz 1 dondurulur

---

## 15. Faz 2 Yol Haritası (özet)

Faz 1 tamamlandıktan sonra açılacak `phase-2/pure-sdk-refactor` branch'inde yapılacaklar:

- `langchain-*` paketlerini kaldır; `RecursiveCharacterTextSplitter` yerine kendi char-based splitter'ı yaz
- PDF ve DOCX loader'larını `pypdf` / `python-docx` doğrudan çağıran saf versiyonlara çevir
- `pytest` suite'i genişlet (>%70 coverage)
- README'yi production standardına getir (kurulum, kullanım, API örnekleri, mimari diyagram)
- `docker-compose.yml` taslağı (opsiyonel, Faz 3 için başlangıç)
- `v0.2-pure-sdk` tag'i ile dondur

## 16. Faz 3 Yol Haritası (özet)

- Kullanıcı yönetimi: JWT auth, `users` tablosu (SQLite yeterli), çok kullanıcılı Chroma collection'lar
- Streaming response (SSE) — `/query` endpoint'i `text/event-stream` döner
- Ayrı HTML/JS frontend (veya React) — Streamlit yerine
- `Dockerfile` + `docker-compose.yml` — backend + frontend + Chroma (persistent volume)
- Deployment: Railway veya Fly.io — canlı URL
- Opsiyonel: HuggingFace lokal embedding (`multilingual-e5-large`) provider adapter'ı
- Opsiyonel: Groq LLM provider adapter'ı

---

## 17. Riskler ve Açık Sorular

### Riskler

1. **Gemini API ücretsiz tier limiti** — Çok doküman yüklemek rate limit'e takılabilir. Mitigasyon: `tenacity` retry, kullanıcıya "lütfen bekleyin" mesajı.
2. **Büyük PDF'lerde bellek** — 100 sayfalı PDF tüm metin belleğe alınır. Faz 1'de kabul ediyoruz; Faz 2'de streaming parse düşünülebilir.
3. **Chroma veri kaybı** — `./data/chroma_db` silinirse tüm index gider. Faz 1'de kabul; Faz 3'te backup stratejisi.
4. **"LangChain'de kalma" tuzağı** — Faz 1 çalışır halde iken Faz 2'ye geçmeme riski. Mitigasyon: git tag ile Faz 1'i dondurmak, Faz 2'yi hemen yeni branch'te başlatmak.

### Açık Sorular (kullanıcıyla netleştirilmesi gereken)

- **Özet token limiti:** Özet için dokümanın tamamı prompt'a veriliyor. Gemini 2.5 Flash'ın 1M token context'i var; pratikte sorun olması zor ama 500+ sayfalı PDF'ler map-reduce gerektirebilir. Faz 1'de basit versiyonla başlayıp, limit hatası gelirse düzelteceğiz.
- **Çoklu doküman filtreleme UX'i:** Streamlit'te multi-select dropdown mu, checkbox listesi mi? Implementation sırasında basit multi-select ile başlayıp kullanıcı geri bildirimine göre iyileştirilecek.

---

## 18. Test Stratejisi (Faz 1 minimum)

**Faz 1'de yazılacak unit testler:**

1. `test_chunker.py` — Chunker'ın `chunk_size` ve `overlap`'a uyduğunu doğrula
2. `test_pipeline_ingest.py` — Fake provider'larla `ingest()` tam akışını test et (LLM/embedding/store çağrılmış mı?)
3. `test_pipeline_answer.py` — Fake provider'larla `answer()` çağrısını test et (retrieval doğru parametrelerle çağrılıyor mu?)

Fake provider'lar için `tests/fakes.py` dosyası: deterministik embedding'ler döndürür (ör: metnin SHA256'sını float'a map et).

Faz 2'de `pytest` coverage >%70 ve integration testler eklenir.

---

## 19. Tasarım Kararları Özeti

| Konu | Karar | Sebep |
|---|---|---|
| LLM sağlayıcısı (Faz 1) | Google Gemini (`gemini-2.5-flash`) | Ücretsiz tier, Türkçe destek, embedding ile aynı sağlayıcı |
| Embedding sağlayıcısı (Faz 1) | `text-embedding-004` | Multilingual, 768 boyut, Gemini ile aynı API key |
| Vector DB | Chroma (PersistentClient) | Tek paket, auto-persist, metadata filtreleme built-in |
| Dil stratejisi | Tek multilingual embedding | Cross-lingual retrieval, UX'te seçim yok, tek collection |
| Framework | Karma: LangChain (loaders+splitter) + saf SDK (LLM+embedding) | Tekerleği yeniden icat etme, ama RAG core'unu kendin yaz |
| Backend | FastAPI | "API tabanlı" şartı, Pydantic validation, OpenAPI otomatik |
| Frontend (Faz 1) | Streamlit | Hızlı MVP, chat widget'ları built-in |
| Frontend (Faz 3) | HTML/JS | Streaming (SSE) için, production için hafif |
| Mimari pattern | Ports & Adapters (Hexagonal) | Faz 2'de saf SDK'ya geçişi ağrısız yapmak |
| Chunk parametreleri | size=1000, overlap=200 | Türkçe + İngilizce karışık için dengeli default |
| Retrieval top-k | 4 | RAG literatüründe yaygın default |
| Kaynak gösterme | Faz 1'de var | Chroma metadata ile "ücretsiz", değerlendirme kriteri |

---

## 20. Sonraki Adımlar

Bu spec onaylandıktan sonra:

1. `writing-plans` skill'i ile **Faz 1 implementasyon planı** yazılacak
2. Plan, bu spec'e referans verecek ve adım adım task'lara bölünecek
3. Her task'a test/verify kriteri eklenecek
4. Plan onaylandıktan sonra implementasyona geçilecek

---

**Status:** Taslak. Kullanıcı geri bildirimi bekleniyor.
