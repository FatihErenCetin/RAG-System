"""Uygulama ayarları — .env'den yüklenir."""
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Tüm ortam değişkenleri tek bir tip güvenli yapıda."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Demo mode — API key gerektirmeden UI'ı test etmek için
    use_mock_providers: bool = False

    # LLM
    llm_provider: str = "gemini"
    gemini_api_key: str = Field(
        default="",
        description="Google AI Studio'dan alınan API key. use_mock_providers=true ise boş bırakılabilir.",
    )
    gemini_llm_model: str = "gemini-2.5-flash"

    # Embedding
    embedding_provider: str = "gemini"
    gemini_embedding_model: str = "gemini-embedding-001"

    # Vector store
    vector_store: str = "chroma"
    chroma_path: str = "./data/chroma_db"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:8501"]
    )
    max_upload_mb: int = 50

    # Retrieval
    default_top_k: int = 4

    # Frontend
    backend_url: str = "http://localhost:8000"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, v):
        """Virgülle ayrılmış string'i listeye çevir."""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


def get_settings() -> Settings:
    """Singleton settings factory."""
    return Settings()
