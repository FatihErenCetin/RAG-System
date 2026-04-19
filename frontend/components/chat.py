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
