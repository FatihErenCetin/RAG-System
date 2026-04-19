"""Streamlit entry point: 'Chat With Your Documents'."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# `streamlit run frontend/app.py` proje kökünü sys.path'e eklemez;
# import'lar çalışsın diye elle ekliyoruz.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from frontend.api_client import APIClient, APIClientError  # noqa: E402
from frontend.components import render_chat, render_documents, render_upload  # noqa: E402


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
