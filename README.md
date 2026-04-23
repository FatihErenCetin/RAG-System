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
- 🔢 **Top-K Slider** — Chat arayüzünden kaç kaynak getirileceği (1–10) ayarlanabilir
- 🔤 **Akıllı PDF Karakter Düzeltici** — Bazı PDF'lerde `P R O J E` şeklinde boşlukla yazılan karakterleri otomatik olarak `PROJE` haline getirir; embedding kalitesini artırır

## Yeni Özellikler

### 🔢 Top-K Retrieval Slider

**Dosya:** `frontend/components/chat.py`

Chat arayüzüne 1–10 arasında ayarlanabilir bir slider eklendi. "Top-K", soruyu yanıtlamak için dokümanlardan kaç metin parçasının (chunk) getirileceğini belirler.

| Değer | Ne zaman kullanmalı? |
|---|---|
| **1–2** | Tek ve net bir bilgi arıyorsanız — *"Şirketin kuruluş yılı nedir?"* Hızlı ve odaklı. |
| **3–5** *(varsayılan)* | Çoğu soru için ideal denge. Birden fazla bölümden bilgi derlenmesi gereken sorularda iyi çalışır. |
| **6–10** | Geniş kapsamlı sorular için — *"Bu rapordaki tüm riskler neler?"* Daha fazla bağlam, daha kapsamlı cevap. |

Slider değeri her sorguda canlı olarak backend'e iletilir; aynı soru farklı Top-K değerleriyle test edilerek retrieval kalitesi gözlemlenebilir.

---

### 🔤 Akıllı PDF Karakter Düzeltici

**Dosya:** `src/adapters/loaders/pdf_loader.py`

Bazı PDF'ler (özellikle belirli fontlarla veya tarama sonrası OCR ile oluşturulmuş dosyalar) pypdf tarafından ayrı ayrı boşluklu harfler olarak çıkarılır:

```
P R O J E L E R   V E   Y A Z I L I M
```

Bu durum embedding kalitesini ciddi ölçüde düşürür; model tek tek harfleri anlamsız token olarak işler.

**Düzeltici mantığı (`_fix_spaced_chars`):**
1. Metni satır satır tarar.
2. Bir satırdaki token'ların **%60'ından fazlası** tek-harfli alfabetik ise satır "bozuk" olarak işaretlenir (en az 6 token şartı ile kısa satırlarda false positive önlenir).
3. Ardışık tek-harf token'ları birleştirilerek okunabilir kelimeler elde edilir.
4. Noktalama işaretleri (`–`, `.`, `,` vb.) ayrı kalır; birleştirilmez.
5. Türkçe karakterler (`İ`, `Ş`, `Ğ`, `Ü`, `Ö`, `Ç`, `ı`, `ş`, `ğ`, `ü`, `ö`, `ç`) Python'un `str.isalpha()` ile doğal olarak desteklenir.

**Sonuç:**
```
PROJELER VE YAZILIM GELİŞTİRMEDENEYİMİ
```
Retrieval skorları ve cevap kalitesi belirgin biçimde artar.

---

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

## 🚀 Hızlı Demo (API Key'siz)

Gemini API key'iniz olmadan uygulamayı denemek ister misiniz? Demo modu fake providers kullanır — gerçek bir LLM çağırmaz ama tüm UI, yükleme, retrieval ve özet akışını görebilirsiniz.

```bash
# Kurulumdan sonra:
echo "USE_MOCK_PROVIDERS=true" > .env

# Terminal 1:
source .venv/bin/activate && uvicorn backend.main:app --reload

# Terminal 2:
source .venv/bin/activate && streamlit run frontend/app.py
```

Demo modunda:
- Dokümanlar in-memory tutulur (backend restart'ta silinir)
- Embedding SHA-256 tabanlı deterministic vektör üretir
- LLM "cevabı" retrieval sonuçlarını template ile özetler
- Upload/query/summarize/delete'in hepsi çalışır

Gerçek LLM cevapları için `.env`'de `USE_MOCK_PROVIDERS=false` ve `GEMINI_API_KEY=<sizin_key>` ayarlayın.

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

- **v0.1 (MVP — bu sürüm):** LangChain (loaders + splitter) + saf SDK (Gemini LLM + embedding + Chroma) karma yapı; Top-K slider, akıllı PDF karakter düzeltici
- **v0.2 (planned):** Saf SDK refactor (LangChain kaldırıldı), kapsamlı test suite, CI (GitHub Actions), mimari diyagram
- **v0.3 (planned):** Docker + docker-compose, JWT authentication, streaming response (SSE), HTML/JS frontend (Streamlit yerine), Railway/Fly.io deployment, opsiyonel HuggingFace + Groq provider'ları

## Lisans

MIT

## Katkıda Bulunma

Issues ve pull request'ler açıktır. Öncelikle spec (`docs/superpowers/specs/`) ve implementation plan (`docs/superpowers/plans/`) okunmalı.
