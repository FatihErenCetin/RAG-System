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
- 🚦 Cross-lingual retrieval (Türkçe ↔ İngilizce)

## Mimari

**Hexagonal (Ports & Adapters):**

```
RAG Systems/
├── backend/        FastAPI REST API
├── src/
│   ├── core/       Domain interfaces (Protocol) + data types
│   ├── rag/        RAGPipeline orchestrator
│   └── adapters/   Concrete providers (Gemini, Chroma, LangChain loaders)
├── frontend/       Streamlit UI
├── tests/          Unit + API tests with test doubles (fakes)
└── docs/           Specs and implementation plans
```

- **src/core/interfaces.py:** Protocol definitions. All adapters depend on these.
- **src/adapters/:** Concrete implementations. Swap providers by replacing adapters.
- **src/rag/pipeline.py:** Orchestrates ingest/answer/summarize.
- **backend/:** FastAPI app with DI; imports only from `src.core` types.
- **frontend/:** Streamlit UI; talks to backend via `APIClient`.

## Kurulum

### Gereksinimler

- Python 3.11+
- [Google Gemini API key](https://aistudio.google.com) — "Get API Key"
- Opsiyonel: `.doc` desteği için `antiword` (örn. `brew install antiword` / `apt install antiword`) + `pip install textract`

### Kurulum adımları

```bash
git clone https://github.com/FatihErenCetin/RAG-System.git
cd "RAG-System"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Konfigürasyon
cp .env.example .env
# .env dosyasını düzenleyin: GEMINI_API_KEY=<sizin_key>
```

## Çalıştırma

Backend ve frontend ayrı terminallerde çalışır.

**Terminal 1 — Backend (FastAPI):**
```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
```

- API: http://localhost:8000
- Swagger UI (interaktif dokümantasyon): http://localhost:8000/docs

**Terminal 2 — Frontend (Streamlit):**
```bash
source .venv/bin/activate
streamlit run frontend/app.py
```

- UI: http://localhost:8501

## Test

```bash
GEMINI_API_KEY=test-key pytest -v
```

Not: API key gerçek olmak zorunda değil — unit testler ve API testleri `FakeLLMProvider`/`FakeEmbeddingProvider` kullanır.

## API Endpoint'leri

| Method | Path | Açıklama |
|---|---|---|
| POST | `/upload` | Multipart dosya yükleme (tek veya çoklu) |
| POST | `/query` | Soru sor (opsiyonel `document_ids` ile filtrele) |
| POST | `/summarize` | Belirli dokümanın özetini al |
| GET | `/documents` | Yüklenmiş dokümanları listele |
| DELETE | `/documents/{id}` | Dokümanı ve chunk'larını sil |
| GET | `/health` | Sistem sağlığı |

Detaylı şemalar için: `http://localhost:8000/docs`

## Konfigürasyon (.env)

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `GEMINI_API_KEY` | (zorunlu) | Google AI Studio API key |
| `GEMINI_LLM_MODEL` | `gemini-2.5-flash` | LLM modeli |
| `GEMINI_EMBEDDING_MODEL` | `text-embedding-004` | Embedding modeli (768 boyut, multilingual) |
| `CHROMA_PATH` | `./data/chroma_db` | Chroma persist dizini |
| `CHUNK_SIZE` | `1000` | Chunk karakter uzunluğu |
| `CHUNK_OVERLAP` | `200` | Chunk'lar arası overlap |
| `MAX_UPLOAD_MB` | `50` | Max toplam upload boyutu |
| `DEFAULT_TOP_K` | `4` | Retrieval'de varsayılan kaç chunk |
| `CORS_ORIGINS` | `http://localhost:8501` | Virgülle ayrılmış origin listesi |

## Mimari Prensipleri

1. **Ports & Adapters (Hexagonal):** Domain'in (RAG logic) dış dünya bağımlılığı yok. Tüm dış kütüphaneler (Gemini, Chroma, LangChain) sadece `src/adapters/` altında.
2. **Protocol-based DI:** `LLMProvider`, `EmbeddingProvider`, `VectorStore`, `Chunker`, `DocumentLoader` — hepsi `runtime_checkable` Python Protocol'leri.
3. **Stateless REST:** Backend state tutmaz; tüm veri Chroma'da persist edilir.
4. **Test doubles for speed:** Unit testler gerçek API çağrısı yapmaz; `tests/fakes.py` deterministic hash-based embedding + in-memory store kullanır.

## Yol Haritası

- **v0.1 (MVP — bu sürüm):** LangChain (loaders + splitter) + saf SDK (Gemini LLM + embedding + Chroma) karma yapı
- **v0.2 (planned):** Saf SDK refactor (LangChain kaldırıldı), kapsamlı test suite, CI (GitHub Actions), mimari diyagram
- **v0.3 (planned):** Docker + docker-compose, JWT authentication, streaming response (SSE), HTML/JS frontend (Streamlit yerine), Railway/Fly.io deployment, opsiyonel HuggingFace + Groq provider'ları

## Lisans

MIT

## Katkıda Bulunma

Issues ve pull request'ler açıktır. Öncelikle spec (`docs/superpowers/specs/`) ve implementation plan (`docs/superpowers/plans/`) okunmalı.
