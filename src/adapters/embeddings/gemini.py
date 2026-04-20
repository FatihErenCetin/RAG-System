"""Gemini embedding adapter (google-generativeai doğrudan)."""
from __future__ import annotations

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential


# Model varsayılan boyutları (output_dimensionality override etmiyorsa).
# gemini-embedding-001 Matryoshka destekler: 768/1536/3072 arasında seçilebilir.
_MODEL_DEFAULT_DIMENSIONS = {
    "gemini-embedding-001": 3072,
    "text-embedding-004": 768,  # legacy; v1beta'da artık çağrılamıyor
}


class GeminiEmbedding:
    """Google Gemini embedding API wrapper.

    `output_dimensionality` parametresi `gemini-embedding-001` için Matryoshka
    representation desteklediğinden, varsayılan olarak 768'e düşürüyoruz
    (Chroma'da saklama maliyeti × 4 daha ucuz, kalite farkı minimal).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        output_dimensionality: int | None = 768,
    ):
        genai.configure(api_key=api_key)
        self._model = model if model.startswith("models/") else f"models/{model}"
        self._short_name = self._model.removeprefix("models/")
        self._output_dim = output_dimensionality

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _embed_one(self, text: str) -> list[float]:
        kwargs = {
            "model": self._model,
            "content": text,
            "task_type": "RETRIEVAL_DOCUMENT",
        }
        if self._output_dim is not None:
            kwargs["output_dimensionality"] = self._output_dim
        result = genai.embed_content(**kwargs)
        return list(result["embedding"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._embed_one(t) for t in texts]

    @property
    def dimension(self) -> int:
        if self._output_dim is not None:
            return self._output_dim
        return _MODEL_DEFAULT_DIMENSIONS.get(self._short_name, 768)
