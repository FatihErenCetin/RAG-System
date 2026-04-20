# Streamlit UI Geliştirme Kılavuzu

> **Hedef kitle:** Streamlit tecrübesi olan, bu repo'yu ilk kez görecek bir geliştirici.
> **Amaç:** Mevcut backend değişmeden, Streamlit frontend'ini iyileştirmek / genişletmek.
> **Kısıt:** Backend API sözleşmesi (endpoint'ler + Pydantic şemaları) sabit; değiştirilmemeli.

---

## 1. Genel Bakış

Bu proje bir **RAG (Retrieval-Augmented Generation)** uygulamasıdır:

1. Kullanıcı doküman yükler (TXT, PDF, DOC, DOCX)
2. Backend chunking → embedding → vector store'a kayıt yapar
3. Kullanıcı soru sorar, backend ilgili chunk'ları bulur ve LLM'e verir
4. LLM cevabı + kaynak chunk'lar kullanıcıya gösterilir
5. Özet çıkarma ve doküman silme de desteklenir

**Mimari:**
```
┌──────────────────┐       HTTP/REST        ┌─────────────────────┐
│  Streamlit UI    │  ───────────────────▶  │  FastAPI Backend    │
│  (sen geliştir)  │  ◀───────────────────  │  (dokunma)          │
└──────────────────┘                         └─────────────────────┘
    port 8501                                      port 8000
```

Streamlit frontend **HTTP istemcidir** — state tutmaz, sadece backend'i çağırır ve cevabı render eder.

---

## 2. Kurulum ve Çalıştırma

### İlk kurulum

```bash
git clone https://github.com/FatihErenCetin/RAG-System.git
cd RAG-System
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Geliştirme için en hızlı başlangıç — Demo Mode

Gemini API key almadan UI'ı test etmek için:

```bash
echo "USE_MOCK_PROVIDERS=true" > .env
```

Demo modunda:
- Hiçbir external API çağrısı yok
- Embedding deterministic (SHA-256 tabanlı)
- LLM "cevap" olarak retrieved chunk'ları template ile formatlar
- Upload / query / summarize / delete akışlarının tümü çalışır

### İki terminalde çalıştır

**Terminal 1 — Backend (dokunma):**
```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
# http://localhost:8000/docs → Swagger UI (tüm endpoint'ler burada)
```

**Terminal 2 — Streamlit (geliştirdiğin):**
```bash
source .venv/bin/activate
streamlit run frontend/app.py
# http://localhost:8501
```

### Status kontrolü
```bash
./check-status.sh
# ✅ Backend  :8000  UP
# ✅ Streamlit :8501  UP
```

---

## 3. Mevcut Frontend Yapısı

```
frontend/
├── app.py                       # Ana entry point — sayfa layout ve orchestration
├── api_client.py                # HTTP wrapper — backend'e tüm istekler buradan
└── components/
    ├── __init__.py
    ├── upload.py                # Dosya yükleme formu
    ├── documents.py             # Doküman listesi + silme + özet butonları
    └── chat.py                  # Chat UI + kaynak göster expander'ı
```

### Altın Kural: Direkt `requests` Kullanma

`frontend/api_client.py` — **tüm** HTTP çağrıları buradan yapılmalı. Backend URL'i, timeout, hata yönetimi tek merkezde.

**Yanlış:**
```python
import requests
resp = requests.post("http://localhost:8000/query", ...)  # ❌
```

**Doğru:**
```python
from frontend.api_client import APIClient, APIClientError
client = APIClient(base_url=os.getenv("BACKEND_URL", "http://localhost:8000"))
try:
    result = client.query(question=..., document_ids=..., top_k=4)  # ✅
except APIClientError as e:
    st.error(str(e))
```

Yeni bir endpoint eklenirse (olmayacak — backend freeze), önce `api_client.py`'a method eklenir, sonra component kullanır.

---

## 4. Backend API Sözleşmesi (Sabit — Değişmez)

Interaktif dokümantasyon: **http://localhost:8000/docs** (Swagger UI).

Aşağıdaki endpoint'lerin hepsi `APIClient` içinde wrap edilmiş. Raw şemalar referans içindir.

### `GET /health`
Sistem ayakta mı?
```json
// Response 200
{"status": "ok", "version": "0.1.0-mvp"}
```
**API client:** `client.health()` → dict

---

### `POST /upload`
Bir veya birden fazla dosya yükle. Multipart form-data.

**Request:**
- Form field: `files` (list[UploadFile]) — tekrar edebilir

**Response 200:**
```json
{
  "results": [
    {
      "document": {
        "id": "uuid-string",
        "name": "rapor.pdf",
        "chunk_count": 12
      }
    }
  ],
  "total_chunks": 12
}
```

**Hata durumları:**
- `400` — desteklenmeyen format (image, zip, vs)
- `413` — toplam upload `MAX_UPLOAD_MB`'ı (default 50) aşıyor
- `422` — parse hatası (bozuk PDF, vs)

**Desteklenen formatlar:** `.txt`, `.pdf`, `.doc` (antiword gerekir), `.docx`

**API client:** `client.upload_files([(filename, file_obj, mime), ...])` → dict

---

### `GET /documents`
Yüklenmiş dokümanların listesi.

**Response 200:**
```json
{
  "documents": [
    {"id": "uuid-1", "name": "rapor.pdf", "chunk_count": 12},
    {"id": "uuid-2", "name": "notes.txt", "chunk_count": 3}
  ]
}
```

**API client:** `client.list_documents()` → list[dict] (direkt documents array'i döner, wrapper değil)

---

### `DELETE /documents/{document_id}`
Dokümanı ve tüm chunk'larını sil (idempotent — olmayan id için de 200).

**Response 200:**
```json
{"deleted_document_id": "uuid-1"}
```

**API client:** `client.delete_document(document_id)` → dict

---

### `POST /query`
Soru sor, cevap + kaynakları al.

**Request:**
```json
{
  "question": "Python nedir?",         // 3-1000 karakter
  "document_ids": ["uuid-1"],           // opsiyonel — sadece bu dokümanlarda ara
  "top_k": 4                            // 1-20 arası, default 4
}
```

**Response 200:**
```json
{
  "answer": "Python genel amaçlı, yüksek seviyeli bir programlama dilidir...",
  "sources": [
    {
      "document_id": "uuid-1",
      "document_name": "rapor.pdf",
      "chunk_index": 2,
      "chunk_preview": "İlk 200 karakter...",
      "score": 0.8745                   // cosine similarity
    }
  ],
  "model": "gemini-2.5-flash",
  "retrieval_count": 3
}
```

**Hata:**
- `422` — validation (question <3 veya >1000 karakter, top_k out of range)

**API client:** `client.query(question, document_ids=None, top_k=4)` → dict

---

### `POST /summarize`
Belirli bir dokümanın özetini çıkar.

**Request:**
```json
{"document_id": "uuid-1"}
```

**Response 200:**
```json
{
  "summary": "Bu doküman X konusunu ele almaktadır...",
  "document_id": "uuid-1",
  "document_name": "rapor.pdf",
  "model": "gemini-2.5-flash"
}
```

**Hata:**
- `404` — doküman bulunamadı

**API client:** `client.summarize(document_id)` → dict

---

## 5. Mevcut UI Durumu

### Neler Var
- Sol kolon:
  - Dosya yükleme (multi-file, type filter, spinner)
  - Doküman listesi (her doküman için Özet + Sil butonu)
- Sağ kolon:
  - Doküman filtresi (multiselect — "sadece şu dokümanlarda ara")
  - Chat input
  - Geçmiş mesajlar
  - Kaynak expander'ı (her cevabın altında)

### Neler Eksik / İyileştirilmeli

**Must-have (beklenenler):**
1. Upload progress bar (şu an sadece spinner, kaç dosyadan kaçı bitti belli değil)
2. Drag-and-drop zone görsel olarak belirgin değil
3. Doküman silme — confirmation dialog yok (yanlışlıkla silme riski)
4. Chat geçmişi persist etmiyor — sayfa yenilenirse kaybolur (session_state var ama reload'da gider)
5. Hata mesajları ham API response döndürür (`500 Internal Server Error: Internal Server Error`) — kullanıcı dostu değil
6. Doküman isimleri uzun olursa layout bozulur (truncate/tooltip gerekli)
7. Özet butonu tıklanınca özet tüm sayfayı kaplar — modal / sidebar tercih edilmeli
8. Boş state'ler (hiç doküman yok, hiç mesaj yok) daha zengin olabilir

**Nice-to-have (bonus):**
9. Doküman kartlarında preview (ilk chunk'ın ilk 150 karakteri)
10. Chat input'a örnek soru önerileri ("Bu dokümanda X hakkında ne var?")
11. Kaynak chunk önizlemesi highlight (sorguyla eşleşen kısmı renklendir)
12. Chat export (markdown dosyasına kaydet)
13. Dark mode toggle
14. Responsive tasarım (mobile preview)
15. Model seçimi UI'dan değiştirilebilir olsun (Gemini Flash vs Pro)
16. Sidebar stat'lar: toplam doküman, toplam chunk, son upload tarihi
17. Recent queries quick access
18. Keyboard shortcuts (Enter ile gönder, Esc ile iptal)

**Advanced (Faz 2 / 3):**
19. Streaming response (SSE) desteği — cevap kelime kelime yazılsın
20. Kullanıcı auth + kendi dokümanları
21. Doküman tag'leri ve kategoriler
22. Chat thread'leri (multiple conversations)

---

## 6. Tasarım Kılavuzu

### Streamlit Best Practices (bu projede uygulanmış / uygulanmalı)

**a) State yönetimi**
- `st.session_state` kullan, global değişken koyma
- `messages` için: `st.session_state.setdefault("messages", [])` pattern'i
- Yeni özellik için: `st.session_state[f"key_{unique_id}"]` (çakışma riski)

**b) Layout**
- Ana layout: `st.columns([1, 2], gap="large")` — sol:sağ = 1:2
- Sidebar alternatif: `st.sidebar` içinde settings/stats
- Container ile gruplama: `with st.container(border=True):`

**c) Butonlar**
- Primary action: `st.button("...", type="primary")`
- Secondary: default stil
- Destructive (Sil): emoji + confirm dialog (Streamlit'te `st.dialog` 1.35+)

**d) Feedback**
- Loading: `st.spinner("...")` context manager
- Success/error/info: `st.success`, `st.error`, `st.info`, `st.warning`
- Toast (1.30+): `st.toast("Yüklendi ✅", icon="✅")` — ephemeral

**e) Form validation**
- Chat input minimum uzunluk: backend 3 karakter ister, UI'da da kontrol et
- `st.chat_input` zaten boş mesajı göndermez

**f) Performance**
- `@st.cache_data` — doküman listesi gibi değişmeyen veriler için
- `@st.cache_resource` — APIClient instance için (tek client yeter)
- Rerun trigger'larına dikkat — her widget state değişikliği rerun'a sebep olur

**g) Yeni Streamlit özellikleri (1.30+)**
- `st.chat_input` / `st.chat_message` — chat UI primitive
- `st.write_stream` — streaming text animation
- `st.dialog` — modal
- `st.popover` — dropdown içerik
- `st.status` — multi-step progress

### Renk / Tipografi

Proje için özel tema yok. `.streamlit/config.toml` oluşturup özelleştirebilirsin:

```toml
[theme]
primaryColor = "#6366F1"          # indigo
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F3F4F6"
textColor = "#111827"
font = "sans serif"
```

Dark mode için `[theme.dark]` aynı şekilde.

### Emoji Kullanımı

Mevcut UI'da emoji'ler var (📥, 💬, 📚). İsterse çıkartılabilir — ama kullanıcı yönlendirmesi için bazılarını tutmak faydalı. Fazla kullanma, anlamı olmalı.

---

## 7. Kod Mimarisi Kuralları

### Component Yapısı

Her yeni UI bölümü için `frontend/components/<isim>.py` oluştur:

```python
"""<Bölüm açıklaması>."""
from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClient, APIClientError


def render_<bölüm>(client: APIClient, <diğer args>) -> <return type>:
    """Tek bir görev yapan render fonksiyonu."""
    st.subheader("...")
    # ... UI kodu ...
```

Sonra `frontend/components/__init__.py`'a export et:
```python
from .<isim> import render_<bölüm>
__all__ = [..., "render_<bölüm>"]
```

### Python Path Gotcha

`frontend/app.py` üstünde şu kod VAR ve **silinmemeli**:

```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
```

`streamlit run frontend/app.py` komutu project root'u sys.path'e eklemez — bu kod onu manuel ekler. Bu olmadan `from frontend.api_client import ...` çalışmaz.

### Error Handling

Her backend çağrısı bir `try/except APIClientError` içinde olmalı:

```python
try:
    result = client.query(question=prompt, ...)
except APIClientError as e:
    st.error(f"Soru gönderilemedi: {e}")
    return
```

`APIClientError` mesajı ham (`500 Internal Server Error: ...`) döner. UI seviyesinde kullanıcı dostu mesaja çevirebilirsin:

```python
def friendly_error(e: APIClientError) -> str:
    msg = str(e)
    if "413" in msg:
        return "Dosya çok büyük. 50 MB'tan küçük dosya yükleyin."
    if "400" in msg:
        return "Bu dosya formatı desteklenmiyor (TXT, PDF, DOCX kabul edilir)."
    return f"Beklenmedik bir hata oldu: {msg}"
```

---

## 8. Çalışma Akışı

### Geliştirirken

Streamlit **hot reload** özelliği var. Dosya kaydettikçe otomatik yenilenir. Sağ üstte "Rerun" mesajı çıkar, "Always rerun"'a basarsan otomatik.

### Test

Şu an frontend için unit test yok (Streamlit component'leri unit-testable değil). Manuel test checklist:

**Upload:**
- [ ] Tek TXT dosya yükle → success mesajı ve doküman listesinde görünüyor
- [ ] Çoklu dosya (3 tane farklı format) yükle → her birinin kaydı doğru
- [ ] Desteklenmeyen dosya (PNG) yükle → 400 hatası gösterilir
- [ ] Çok büyük dosya (>50MB) yükle → 413 hatası gösterilir
- [ ] Boş dosya yükle → parse olur ama chunk_count=0

**Chat:**
- [ ] Doküman yokken soru sor → "belge yok" uyarısı
- [ ] Doküman varken soru sor → cevap + en az 1 kaynak
- [ ] Kaynak expander'ı aç → chunk preview, dokümant adı, skor görünüyor
- [ ] Aynı konuşmada birkaç mesaj → history korunuyor
- [ ] Sayfa yenile → history kayboluyor (bilinen bug — düzelt)
- [ ] Multiselect ile tek doküman seç → sadece o dokümandan kaynaklar geliyor

**Documents panel:**
- [ ] Doküman sil → listeden kayboluyor ve chat'te o dokümanın kaynakları artık çıkmıyor
- [ ] Özet al → LLM'in özet cevabı gösteriliyor

**Edge cases:**
- [ ] Backend kapalıyken Streamlit aç → "Backend'e ulaşılamıyor" uyarısı
- [ ] Backend ayakta ama Gemini API fail → 500 döner (sadece prod mode'da, demo mode'da fail etmez)

### E2E demo senaryosu

1. Demo mode aç (`USE_MOCK_PROVIDERS=true`)
2. 2-3 farklı format (TXT, PDF, DOCX) yükle
3. Her birine soru sor
4. Multiselect ile sadece birine sor
5. Bir tanesinin özetini al
6. Bir tanesini sil, tekrar soru sor (silinen dokümandan kaynak gelmediğini gör)

---

## 9. Proje İçi Kaynaklar

- `docs/superpowers/specs/2026-04-19-chat-with-your-docs-design.md` — tüm proje tasarımı
- `docs/superpowers/plans/2026-04-19-phase1-mvp-implementation.md` — nasıl inşa edildiği (29 task)
- `README.md` — kullanıcı dokümanı
- `backend/schemas.py` — tüm Pydantic request/response modelleri
- `frontend/api_client.py` — HTTP client API'si

---

## 10. Dokunulmaması Gerekenler

- `backend/` — API sözleşmesi sabit
- `src/core/interfaces.py` — Protocol tanımları
- `src/rag/pipeline.py` — orchestrator
- `src/adapters/` — sağlayıcılar
- `tests/` — test altyapısı (yeni frontend testi eklemek hariç)

Backend'de hata bulursan / gerekli bir endpoint eksikse, bir issue açıp proje sahibine söyle — direkt düzenleme.

## 11. Ne Yapılırsa İyi Olur (Öncelik Sırasına Göre)

1. **Error mapping katmanı** — API'den gelen ham hataları kullanıcı dostu mesaja çevirme (ufak bir `frontend/errors.py`)
2. **Sil-onay dialog'u** — `st.dialog` ile
3. **Dosya upload progress bar** — her dosyayı tek tek yükle, `st.progress` göster
4. **`.streamlit/config.toml`** — tema, favicon, page title
5. **Chat history persist** — localStorage-like davranış için `streamlit-extras` paketi veya session file
6. **Responsive layout** — mobile'da single column, desktop'ta 2 column
7. **Doküman kart görünümü** — `st.container(border=True)` + preview + actions
8. **Boş state improvement** — "henüz doküman yok → sol panelden yükle ok'u" gibi yönlendirme
9. **Keyboard shortcuts** — `streamlit-shortcuts` veya `key_press` component
10. **i18n desteği** — TR/EN dil switcher (şimdilik her yer Türkçe)

---

## 12. Özet

- **Ne bozulmamalı:** Backend, `api_client.py` public API'si, `frontend/app.py`'ın sys.path kurulumu, component yapısı
- **Ne serbestçe değişebilir:** Her component'in içi, layout kararları, renk/tipografi, yeni componentler eklemek
- **Teslim formatı:** Yeni branch'te çalış, PR aç. Commit mesajları conventional (`feat(frontend): ...`, `fix(frontend): ...`).

Başarılar 🚀
