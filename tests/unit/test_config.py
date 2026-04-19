"""Config yükleme testleri."""
import os
from backend.config import Settings


def test_settings_loads_required_fields(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.setenv("CHROMA_PATH", "./test_data")
    settings = Settings()
    assert settings.gemini_api_key == "test-key-123"
    assert settings.chroma_path == "./test_data"
    assert settings.llm_provider == "gemini"
    assert settings.chunk_size == 1000


def test_settings_has_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    settings = Settings()
    assert settings.api_port == 8000
    assert settings.default_top_k == 4
    assert settings.max_upload_mb == 50
    assert settings.gemini_llm_model == "gemini-2.5-flash"


def test_cors_origins_parsed_as_list(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000")
    settings = Settings()
    assert settings.cors_origins == ["http://localhost:8501", "http://localhost:3000"]


def test_gemini_api_key_optional_when_mock_mode(monkeypatch):
    """Demo modu: API key olmadan config geçerli."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("USE_MOCK_PROVIDERS", "true")
    settings = Settings()
    assert settings.use_mock_providers is True
    assert settings.gemini_api_key == ""


def test_use_mock_providers_defaults_false(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    settings = Settings()
    assert settings.use_mock_providers is False
