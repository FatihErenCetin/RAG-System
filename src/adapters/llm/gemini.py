"""Gemini LLM adapter."""
from __future__ import annotations

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.interfaces import RetrievedChunk
from src.core.prompts import build_qa_prompt


class GeminiLLM:
    """Google Gemini generative text model wrapper."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self._model_name = model
        self._model = genai.GenerativeModel(model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, prompt: str, context: list[RetrievedChunk]) -> str:
        context_blocks = [
            f"[{r.chunk.document_name} — chunk {r.chunk.index}]\n{r.chunk.content}"
            for r in context
        ]
        full_prompt = build_qa_prompt(question=prompt, context_blocks=context_blocks)
        response = self._model.generate_content(full_prompt)
        return response.text or ""

    @property
    def model_name(self) -> str:
        return self._model_name
