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
