"""Gemini embedding adapter (google-generativeai doğrudan)."""
from __future__ import annotations

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.interfaces import EmbeddingProvider


# text-embedding-004 → 768 boyutlu vektör döner
_MODEL_DIMENSIONS = {
    "text-embedding-004": 768,
    "models/text-embedding-004": 768,
}


class GeminiEmbedding:
    """Google Gemini embedding API wrapper."""

    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        genai.configure(api_key=api_key)
        self._model = model if model.startswith("models/") else f"models/{model}"
        self._short_name = self._model.removeprefix("models/")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _embed_one(self, text: str) -> list[float]:
        result = genai.embed_content(
            model=self._model,
            content=text,
            task_type="RETRIEVAL_DOCUMENT",
        )
        return list(result["embedding"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._embed_one(t) for t in texts]

    @property
    def dimension(self) -> int:
        return _MODEL_DIMENSIONS.get(self._short_name, 768)
