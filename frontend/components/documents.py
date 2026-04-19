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
