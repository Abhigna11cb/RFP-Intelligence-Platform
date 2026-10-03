"""
Central configuration — all secrets/settings from environment variables or .env file.
Never hardcode credentials in code.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ── PostgreSQL ──────────────────────────────────────────────
    db_host: str = Field(default="localhost",     alias="DB_HOST")
    db_port: int = Field(default=5432,            alias="DB_PORT")
    db_name: str = Field(default="rfp_platform",  alias="DB_NAME")
    db_user: str = Field(default="postgres",      alias="DB_USER")
    db_password: str = Field(default="password",  alias="DB_PASSWORD")

    # ── Embedding model ─────────────────────────────────────────
    embedding_provider: str = Field(default="openai",                    alias="EMBEDDING_PROVIDER")
    embedding_model_name: str = Field(default="text-embedding-3-large",  alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1536,                             alias="EMBEDDING_DIM")
    embedding_dimensions: int = Field(default=1536,                      alias="EMBEDDING_DIMENSIONS")

    # ── LLM ────────────────────────────────────────────────────
    llm_provider: str = Field(default="openai",                          alias="LLM_PROVIDER")
    openai_api_key: str = Field(default="",                              alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini",                    alias="OPENAI_MODEL")
    anthropic_api_key: str = Field(default="",                           alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-haiku-20241022",   alias="ANTHROPIC_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434",       alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2",                        alias="OLLAMA_MODEL")

    # ── Chunking ────────────────────────────────────────────────
    chunk_min_chars: int = Field(default=1000,    alias="CHUNK_MIN_CHARS")
    chunk_max_chars: int = Field(default=3000,    alias="CHUNK_MAX_CHARS")
    chunk_overlap_chars: int = Field(default=200, alias="CHUNK_OVERLAP")

    # ── Retrieval ───────────────────────────────────────────────
    default_top_k: int = Field(default=10,        alias="DEFAULT_TOP_K")
    similarity_threshold: float = Field(default=0.8, alias="SIMILARITY_THRESHOLD")
    rrf_k: int = Field(default=60,                alias="RRF_K")

    # ── OCR ─────────────────────────────────────────────────────
    ocr_enabled: bool = Field(default=False,      alias="OCR_ENABLED")
    ocr_language: str = Field(default="eng",      alias="OCR_LANGUAGE")

    # ── API ─────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0",     alias="API_HOST")
    api_port: int = Field(default=8000,           alias="API_PORT")

    @property
    def db_dsn(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def db_config_dict(self) -> dict:
        """psycopg2-compatible connection dict."""
        return {
            "host": self.db_host,
            "port": self.db_port,
            "database": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — call this everywhere instead of instantiating Settings()."""
    return Settings()
