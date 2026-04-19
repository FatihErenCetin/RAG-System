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
