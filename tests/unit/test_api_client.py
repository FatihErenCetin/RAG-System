"""Frontend API client testleri (requests-mock ile)."""
from unittest.mock import patch, MagicMock

import pytest

from frontend.api_client import APIClient


@pytest.fixture
def client():
    return APIClient(base_url="http://localhost:8000")


def test_health_check(client):
    with patch("frontend.api_client.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            ok=True,
            json=lambda: {"status": "ok", "version": "0.1.0-mvp"},
        )
        result = client.health()
        assert result["status"] == "ok"


def test_list_documents(client):
    with patch("frontend.api_client.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            ok=True,
            json=lambda: {"documents": [{"id": "a", "name": "a.txt", "chunk_count": 3}]},
        )
        docs = client.list_documents()
        assert len(docs) == 1
        assert docs[0]["name"] == "a.txt"


def test_query(client):
    with patch("frontend.api_client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            ok=True,
            json=lambda: {
                "answer": "test",
                "sources": [],
                "model": "gemini-2.5-flash",
                "retrieval_count": 0,
            },
        )
        result = client.query(question="test?", document_ids=None, top_k=4)
        assert result["answer"] == "test"
