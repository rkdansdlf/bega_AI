"""서비스의 환경 설정을 관리하는 Pydantic 설정 모듈입니다.

이 모듈은 `pydantic-settings`를 사용하여 .env 파일 또는 환경 변수에서
애플리케이션 설정을 로드하고 유효성을 검사하는 `Settings` 클래스를 정의합니다.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션의 모든 설정을 담는 Pydantic 모델 클래스입니다.

    환경 변수나 .env 파일로부터 설정을 로드하며, 각 설정에 대한 타입 힌트와
    기본값, 유효성 검사를 제공합니다.
    """
    # Pydantic 모델 설정: .env 파일 사용, 추가 필드 무시 등
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # --- 기본 애플리케이션 설정 ---
    app_name: str = "KBO AI Service"
    debug: bool = False
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # --- 데이터베이스 설정 (OCI/Postgres) ---
    oci_db_url: str = Field(..., env="OCI_DB_URL")
    supabase_db_url: Optional[str] = Field(None, env="SUPABASE_DB_URL")

    # --- LLM / 임베딩 프로바이더 설정 ---
    # LLM(거대 언어 모델) 및 임베딩 생성을 위해 사용할 서비스를 지정합니다.
    llm_provider: str = Field("gemini", env="LLM_PROVIDER")
    embed_provider: str = Field("gemini", env="EMBED_PROVIDER")
    embed_model: str = Field("", env="EMBED_MODEL") # 특정 모델을 지정할 때 사용

    # --- Google Gemini 설정 ---
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.5-flash", env="GEMINI_MODEL")
    gemini_embed_model: str = Field("", env="GEMINI_EMBED_MODEL")
    gemini_base_url: str = Field(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        env="GEMINI_BASE_URL",
    )

    # --- OpenAI 설정 ---
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_embed_model: str = Field("text-embedding-3-small", env="OPENAI_EMBED_MODEL")

    # --- OpenRouter 설정 ---
    openrouter_api_key: Optional[str] = Field(None, env="OPENROUTER_API_KEY")
    openrouter_model: str = Field("openai/gpt-oss-120b", env="OPENROUTER_MODEL")
    # Pydantic Settings tries to parse List[str] as JSON. read as str to avoid error.
    openrouter_fallback_models_raw: str = Field("", env="OPENROUTER_FALLBACK_MODELS")
    
    @property
    def openrouter_fallback_models(self) -> List[str]:
        if not self.openrouter_fallback_models_raw:
            return []
        return [m.strip() for m in self.openrouter_fallback_models_raw.split(",") if m.strip()]
    openrouter_base_url: str = Field(
        "https://openrouter.ai/api/v1", env="OPENROUTER_BASE_URL"
    )
    openrouter_referer: Optional[str] = Field(None, env="OPENROUTER_REFERER")
    openrouter_app_title: Optional[str] = Field(None, env="OPENROUTER_APP_TITLE")
    openrouter_embed_model: Optional[str] = Field(None, env="OPENROUTER_EMBED_MODEL")

    # --- Function Calling / Chatbot 설정 ---
    chatbot_model_name: Optional[str] = Field(None, env="CHATBOT_MODEL_NAME")

    # --- 검색(Retrieval) 관련 설정 ---
    default_search_limit: int = Field(3, env="DEFAULT_SEARCH_LIMIT")

    # --- SSE / 채팅 관련 설정 ---
    max_output_tokens: int = Field(1024, env="MAX_OUTPUT_TOKENS")

    @field_validator("embed_provider")
    def _validate_embed_provider(cls, value: str) -> str:
        """`embed_provider` 필드의 값이 지원되는 프로바이더 중 하나인지 검증합니다."""
        allowed = {"gemini", "local", "hf", "openai", "openrouter"}
        if value not in allowed:
            raise ValueError(
                f"지원되지 않는 EMBED_PROVIDER '{value}'입니다. 다음 중에서 선택하세요: {sorted(allowed)}"
            )
        return value

    @field_validator("llm_provider")
    def _validate_llm_provider(cls, value: str) -> str:
        """`llm_provider` 필드의 값이 지원되는 프로바이더 중 하나인지 검증합니다."""
        allowed = {"gemini", "openrouter"}
        if value not in allowed:
            raise ValueError(f"지원되지 않는 LLM_PROVIDER '{value}'입니다. 다음 중에서 선택하세요: {sorted(allowed)}")
        return value

    @property
    def cors_allowed_origins(self) -> List[str]:
        """CORS 정책에 따라 허용된 출처 목록을 반환합니다."""
        return self.cors_origins

    @property
    def database_url(self) -> str:
        """데이터베이스 연결 URL을 반환합니다."""
        return self.oci_db_url

    @property
    def function_calling_model(self) -> str:
        """Function Calling 전용 모델명이 지정되면 사용하고, 그렇지 않으면 기본 LLM 모델을 사용합니다."""
        if self.chatbot_model_name:
            return self.chatbot_model_name
        if self.llm_provider == "gemini":
            return self.gemini_model
        return self.openrouter_model

    @property
    def function_calling_base_url(self) -> str:
        """Function Calling 클라이언트가 사용할 기본 Base URL."""
        return (
            self.gemini_base_url
            if self.llm_provider == "gemini"
            else self.openrouter_base_url
        )

    @property
    def function_calling_api_key(self) -> Optional[str]:
        """Function Calling에 사용할 API 키."""
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        if self.llm_provider == "openrouter":
            return self.openrouter_api_key
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """설정 객체를 반환합니다.

    `lru_cache`를 사용하여 설정 객체를 한 번만 생성하고 캐시하여,
    애플리케이션 전체에서 동일한 설정 인스턴스를 사용하도록 보장합니다.
    """
    return Settings()
