import streamlit as st
import time

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Kendi Dokümanların ile Sohbet Et",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- DURUM YÖNETİMİ (SESSION STATE) ---
if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_list" not in st.session_state:
    st.session_state.uploaded_file_list = []
if "sidebar_uploader_key" not in st.session_state:
    st.session_state.sidebar_uploader_key = 1 
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# --- POP-UP (DIALOG) FONKSİYONU ---
@st.dialog("⚠️ Hata: Mükerrer Dosya")
def show_duplicate_warning():
    st.write("Bu dökümanı daha önce eklediniz. Lütfen, farklı bir döküman eklemeyi deneyiniz.")
    if st.button("Tamam", use_container_width=True):
        st.rerun()

# --- SADECE KARANLIK TEMA İÇİN CUSTOM CSS ---
def local_css():
    st.markdown("""
    <style>
        /* 1. Genel Arkaplanlar (Karanlık Tema) */
        .stApp { background-color: #0e1117 !important; }
        section[data-testid="stSidebar"] > div { background-color: #262730 !important; }
        
        /* Tüm Metinler Beyaz */
        .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3 {
            color: #ffffff !important;
        }

        /* 2. Chat Input Alanı */
        [data-testid="stChatInput"] {
            background-color: #1e2127 !important;
            border-color: #4B4B4B !important;
        }
        [data-testid="stChatInput"] textarea {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        /* 3. Dosya Yükleyici (Uploader) Tasarımı */
        [data-testid="stFileUploader"] {
            background-color: #1e2127 !important;
            border: 2px dashed #4B4B4B !important;
            border-radius: 15px !important;
            padding: 20px !important;
        }
        
        /* İngilizce varsayılan metinleri gizle */
        [data-testid="stFileUploadDropzone"] > div > div > span { display: none !important; }
        [data-testid="stFileUploadDropzone"] > div > div > small { display: none !important; }
        
        /* Türkçe Metinler ve Boyut Uyarıları */
        [data-testid="stFileUploadDropzone"] > div > div::before {
            content: "Dosya seçin veya dosyayı buraya sürükleyip bırakın";
            display: block;
            margin-bottom: 15px;
            font-size: 16px;
            font-weight: 500;
        }
        [data-testid="stFileUploadDropzone"] > div > div::after {
            content: "Desteklenen boyut: Dosya başına 200MB \\A Desteklenen Dosya Tipleri: PDF, DOCX, TXT";
            white-space: pre-wrap;
            display: block;
            margin-top: 15px;
            font-size: 13px;
            opacity: 0.7;
        }
        
        /* 'Browse files' butonunu Türkçeleştirme */
        [data-testid="stFileUploadDropzone"] button {
            font-size: 0 !important; 
            background-color: #262730 !important;
            border: 1px solid #ffffff !important;
        }
        [data-testid="stFileUploadDropzone"] button::after {
            content: 'Dosya yükle'; 
            font-size: 14px !important;
            visibility: visible;
        }

        /* 4. Genel Buton Tasarımları */
        .stButton > button {
            background-color: #1e2127 !important;
            border: 1px solid #4B4B4B !important;
            border-radius: 15px !important;
            transition: 0.3s;
        }
        .stButton > button:hover {
            border-color: #FF4B2B !important;
            color: #FF4B2B !important;
        }

        /* 5. Chat Mesajları */
        [data-testid="stChatMessage"] {
            border-radius: 20px;
        }
        [data-testid="stChatMessage"][data-testid="stChatMessageUser"] {
            background-color: #262730 !important;
        }
        
        /* Sağ Üst Butonları Hizalama */
        [data-testid="column"] {
            min-width: 0 !important; 
            padding: 0 5px !important; 
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- HEADER (ÜST BAR) ---
col_empty, col_reg, col_login = st.columns([10, 1.2, 1.2])

with col_reg:
    st.button("Kayıt Ol", use_container_width=True)
with col_login:
    st.button("Giriş Yap", use_container_width=True)

# --- ANA EKRAN (İLK DOSYA YÜKLEME AŞAMASI) ---
if not st.session_state.files_uploaded:
    st.markdown("<br><br>", unsafe_allow_html=True)
    center_col1, center_col2, center_col3 = st.columns([1, 2.5, 1])
    
    with center_col2:
        st.markdown("<h1 style='text-align: center;'>🤖 Kendi Dokümanların ile Sohbet Et</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.1em; opacity: 0.8;'>Yapay zeka asistanınız dokümanlarınızı analiz etmek için hazır</p><br>", unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Başlamak için Dökümanlarınızı yükleyin", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Dökümanları Analiz Et ve Sohbeti Başlat", use_container_width=True):
                for f in uploaded_files:
                    st.session_state.uploaded_file_list.append({
                        "name": f.name,
                        "type": f.name.split('.')[-1].lower()
                    })
                
                with st.spinner("Dökümanlar analiz ediliyor, RAG sistemi hazırlanıyor..."):
                    time.sleep(1.5) 
                
                st.session_state.files_uploaded = True
                st.rerun()

# --- CHAT EKRANI VE YAN PANEL ---
else:
    # SOL PANEL: DİNAMİK DOSYA YÖNETİMİ VE AYARLAR
    with st.sidebar:
        
        # --- 1. DOSYA YÖNETİMİ VE EKLENENLER ---
        st.title("📂 Analiz Geçmişi")
        st.info("Aşağıdaki dökümanlar analiz edildi:")
        
        st.markdown("### Eklenen Dökümanlar")
        for file_info in st.session_state.uploaded_file_list:
            ext = file_info["type"]
            if ext == "pdf": icon = "📕"
            elif ext in ["docx", "doc"]: icon = "📘"
            elif ext == "txt": icon = "📝"
            else: icon = "📄"
            
            st.write(f"{icon} {file_info['name']}")
        
        st.markdown("---")
        
        # --- 2. YENİ DÖKÜMAN EKLE ---
        st.subheader("Yeni Döküman Ekle")
        new_files = st.file_uploader(
            "Ek doküman ekle", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True,
            key=f"sidebar_uploader_{st.session_state.sidebar_uploader_key}"
        )
        
        if new_files:
            if st.button("Dökümanları Sisteme Ekle", use_container_width=True):
                has_duplicate = False
                for f in new_files:
                    if any(d['name'] == f.name for d in st.session_state.uploaded_file_list):
                        has_duplicate = True
                        break 
                
                if has_duplicate:
                    show_duplicate_warning()
                else:
                    for f in new_files:
                        st.session_state.uploaded_file_list.append({
                            "name": f.name,
                            "type": f.name.split('.')[-1].lower()
                        })
                    st.success("Yeni dökümanlar başarıyla eklendi!")
                    time.sleep(1)
                    st.session_state.sidebar_uploader_key += 1
                    st.rerun() 
        
        st.markdown("---")
                    
        # --- 3. AYARLAR VE API ANAHTARI ---
        st.title("⚙️ Ayarlar")
        
        api_key_input = st.text_input("API Anahtarı (API Key)", type="password", placeholder="sk-...", value=st.session_state.api_key)
        if st.button("🔑 Anahtarı Kaydet", use_container_width=True):
            if api_key_input:
                st.session_state.api_key = api_key_input
                st.success("API Anahtarı başarıyla kaydedildi!")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Lütfen geçerli bir API anahtarı girin.")
        
        st.markdown("---")
        
        # --- 4. MODEL TERCİHLERİ ---
        st.markdown("### 🧠 Model Tercihleri")
        selected_model = st.selectbox(
            "Kullanılacak Model", 
            ["GPT-3.5-Turbo", "GPT-4o", "Gemini 1.5 Pro", "Claude 3.5 Sonnet"]
        )
        
        temperature = st.slider(
            "Yaratıcılık (Temperature)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.1,
            help="0 daha tutarlı ve net yanıtlar verirken, 1'e yaklaştıkça model daha yaratıcı yanıtlar üretir."
        )
        
        st.markdown("---")
        
        # --- 5. PARÇALAMA (CHUNK) AYARLARI ---
        st.markdown("### ✂️ Parçalama (Chunk) Ayarları")
        chunk_mode = st.radio(
            "Parça Boyutu Modu", 
            ["Otomatik", "Manuel Belirle"], 
            horizontal=True,
            help="Dokümanların yapay zeka için ne büyüklükte metin parçalarına bölüneceğini belirler."
        )
        
        if chunk_mode == "Manuel Belirle":
            chunk_size = st.slider(
                "Parça Boyutu (Chunk Size)", 
                min_value=100, 
                max_value=2000, 
                value=1000, 
                step=100
            )
            chunk_info = str(chunk_size)
        else:
            chunk_size = "auto"
            chunk_info = "Otomatik"

        st.markdown("---")
        
        # --- 6. SOHBETİ TEMİZLE BUTONU ---
        if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- ANA ALAN: SOHBET EKRANI ---
    st.title("Sohbet")
    
    if not st.session_state.messages:
        st.caption("Mevcut dokümanlarınız üzerinden sohbete başlamak için aşağıya bir soru yazın.")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Dokümanların hakkında bir soru sor..."):
        
        # API Key kontrolü
        if not st.session_state.api_key:
            st.error("Lütfen önce sol menüden API Anahtarınızı girip kaydedin.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Asistanın yanıt metninde seçilen dinamik Chunk Boyutunu gösteriyoruz
            response_text = f"Bu bir örnek AI yanıtıdır. Seçilen Model: **{selected_model}**, Sıcaklık: **{temperature}**, Chunk Boyutu: **{chunk_info}**.\nYüklediğin dökümanlara göre Langchain entegrasyonu buraya gelecek."
            
            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})