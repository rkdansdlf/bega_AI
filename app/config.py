"""서비스 환경설정을 관리하는 Pydantic 설정 모듈."""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    app_name: str = "KBO AI Service"
    debug: bool = False
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # Database (Supabase/Postgres)
    supabase_db_url: str = Field(..., env="SUPABASE_DB_URL")

    # LLM / Embedding provider 설정
    llm_provider: str = Field("gemini", env="LLM_PROVIDER")

    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-1.5-flash", env="GEMINI_MODEL")
    gemini_embed_model: str = Field("text-embedding-004", env="GEMINI_EMBED_MODEL")

    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_embed_model: str = Field("text-embedding-3-small", env="OPENAI_EMBED_MODEL")

    openrouter_api_key: Optional[str] = Field(None, env="OPENROUTER_API_KEY")
    openrouter_model: str = Field("openai/gpt-4o-mini", env="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(
        "https://openrouter.ai/api/v1", env="OPENROUTER_BASE_URL"
    )
    openrouter_referer: Optional[str] = Field(None, env="OPENROUTER_REFERER")
    openrouter_app_title: Optional[str] = Field(None, env="OPENROUTER_APP_TITLE")
    openrouter_embed_model: Optional[str] = Field(None, env="OPENROUTER_EMBED_MODEL")

    embed_provider: str = Field("gemini", env="EMBED_PROVIDER")
    embed_model: str = Field("", env="EMBED_MODEL")

    # Retrieval
    default_search_limit: int = Field(6, env="DEFAULT_SEARCH_LIMIT")

    # SSE / Chat configuration
    max_output_tokens: int = Field(1024, env="MAX_OUTPUT_TOKENS")

    @field_validator("embed_provider")
    def _validate_embed_provider(cls, value: str) -> str:
        allowed = {"gemini", "local", "hf", "openai", "openrouter"}
        if value not in allowed:
            raise ValueError(
                f"Unsupported EMBED_PROVIDER '{value}'. Choose from {sorted(allowed)}"
            )
        return value

    @field_validator("llm_provider")
    def _validate_llm_provider(cls, value: str) -> str:
        allowed = {"gemini", "openrouter"}
        if value not in allowed:
            raise ValueError(f"Unsupported LLM_PROVIDER '{value}'. Choose from {sorted(allowed)}")
        return value

    @property
    def cors_allowed_origins(self) -> List[str]:
        return self.cors_origins

    @property
    def database_url(self) -> str:
        return self.supabase_db_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
