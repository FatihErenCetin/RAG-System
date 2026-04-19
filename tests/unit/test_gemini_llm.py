"""Gemini LLM adapter testleri (mocked)."""
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.llm.gemini import GeminiLLM
from src.core.interfaces import Chunk, RetrievedChunk


def _retrieved(content: str, name: str = "d.txt") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id="c1", document_id="d1", document_name=name,
            content=content, index=0, metadata={},
        ),
        score=0.9,
    )


@pytest.fixture
def mock_genai():
    with patch("src.adapters.llm.gemini.genai") as m:
        yield m


def test_llm_generates_answer(mock_genai):
    fake_response = MagicMock()
    fake_response.text = "fake answer text"
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = fake_response
    mock_genai.GenerativeModel.return_value = mock_model_instance

    llm = GeminiLLM(api_key="k", model="gemini-2.5-flash")
    answer = llm.generate("soru?", context=[_retrieved("context 1"), _retrieved("context 2")])

    assert answer == "fake answer text"
    mock_model_instance.generate_content.assert_called_once()
    prompt_arg = mock_model_instance.generate_content.call_args.args[0]
    assert "context 1" in prompt_arg
    assert "soru?" in prompt_arg


def test_llm_model_name_property():
    with patch("src.adapters.llm.gemini.genai"):
        llm = GeminiLLM(api_key="k", model="gemini-2.5-flash")
        assert llm.model_name == "gemini-2.5-flash"


def test_llm_handles_empty_context(mock_genai):
    fake_response = MagicMock()
    fake_response.text = "no context answer"
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = fake_response
    mock_genai.GenerativeModel.return_value = mock_model_instance

    llm = GeminiLLM(api_key="k", model="gemini-2.5-flash")
    answer = llm.generate("soru?", context=[])
    assert answer == "no context answer"
