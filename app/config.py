"""서비스의 환경 설정을 관리하는 Pydantic 설정 모듈입니다.

이 모듈은 `pydantic-settings`를 사용하여 .env 파일 또는 환경 변수에서
애플리케이션 설정을 로드하고 유효성을 검사하는 `Settings` 클래스를 정의합니다.
"""

import logging
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, PrivateAttr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


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
    # CORS 설정(자격 증명 쿠키 사용 시 '* '는 허용되지 않음)
    # 로컬 개발 기준 기본값은 Vite/개발 서버에서 사용되는 대표 도메인으로 제한
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5176",
            "http://127.0.0.1:5176",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    # --- 데이터베이스 설정 ---
    # 운영 기본 경로
    postgres_db_url: Optional[str] = Field(None, validation_alias="POSTGRES_DB_URL")
    # 하위 호환 경로 (deprecated)
    legacy_source_db_url: Optional[str] = Field(
        None, validation_alias="SUPABASE_DB_URL"
    )
    _legacy_source_db_warned: bool = PrivateAttr(default=False)

    # --- LLM / 임베딩 프로바이더 설정 ---
    # LLM(거대 언어 모델) 및 임베딩 생성을 위해 사용할 서비스를 지정합니다.
    llm_provider: str = Field("gemini", validation_alias="LLM_PROVIDER")
    embed_provider: str = Field("gemini", validation_alias="EMBED_PROVIDER")
    embed_model: str = Field(
        "", validation_alias="EMBED_MODEL"
    )  # 특정 모델을 지정할 때 사용

    # --- Google Gemini 설정 ---
    gemini_api_key: Optional[str] = Field(None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.0-flash", validation_alias="GEMINI_MODEL")
    gemini_embed_model: str = Field("", validation_alias="GEMINI_EMBED_MODEL")
    gemini_base_url: str = Field(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        validation_alias="GEMINI_BASE_URL",
    )

    # --- OpenAI 설정 ---
    openai_api_key: Optional[str] = Field(None, validation_alias="OPENAI_API_KEY")
    openai_embed_model: str = Field(
        "text-embedding-3-small", validation_alias="OPENAI_EMBED_MODEL"
    )

    # --- OpenRouter 설정 ---
    openrouter_api_key: Optional[str] = Field(
        None, validation_alias="OPENROUTER_API_KEY"
    )
    openrouter_model: str = Field(
        "upstage/solar-pro-3:free", validation_alias="OPENROUTER_MODEL"
    )
    # Pydantic Settings tries to parse List[str] as JSON. read as str to avoid error.
    # Default: openrouter/free - intelligent router that auto-selects available free models
    openrouter_fallback_models_raw: str = Field(
        "openrouter/free", validation_alias="OPENROUTER_FALLBACK_MODELS"
    )

    @property
    def openrouter_fallback_models(self) -> List[str]:
        if not self.openrouter_fallback_models_raw:
            return []
        return [
            m.strip()
            for m in self.openrouter_fallback_models_raw.split(",")
            if m.strip()
        ]

    openrouter_base_url: str = Field(
        "https://openrouter.ai/api/v1", validation_alias="OPENROUTER_BASE_URL"
    )
    openrouter_referer: Optional[str] = Field(
        None, validation_alias="OPENROUTER_REFERER"
    )
    openrouter_app_title: Optional[str] = Field(
        None, validation_alias="OPENROUTER_APP_TITLE"
    )
    openrouter_embed_model: Optional[str] = Field(
        None, validation_alias="OPENROUTER_EMBED_MODEL"
    )
    vision_model: str = Field(
        "google/gemini-2.0-flash-001", validation_alias="VISION_MODEL"
    )

    # --- Coach LLM 설정 ---
    coach_llm_provider: str = Field("openrouter", validation_alias="COACH_LLM_PROVIDER")
    coach_max_output_tokens: int = Field(
        2000, validation_alias="COACH_MAX_OUTPUT_TOKENS"
    )  # 2000 tokens recommended per COACH_PROMPT_V2
    coach_openrouter_model: str = Field(
        "upstage/solar-pro-3:free", validation_alias="COACH_OPENROUTER_MODEL"
    )
    coach_openrouter_fallback_models_raw: str = Field(
        "openrouter/free",
        validation_alias="COACH_OPENROUTER_FALLBACK_MODELS",
    )
    coach_brief_max_output_tokens: int = Field(
        8000, validation_alias="COACH_BRIEF_MAX_OUTPUT_TOKENS"
    )

    @property
    def coach_openrouter_fallback_models(self) -> List[str]:
        if not self.coach_openrouter_fallback_models_raw:
            return []
        return [
            m.strip()
            for m in self.coach_openrouter_fallback_models_raw.split(",")
            if m.strip()
        ]

    coach_llm_read_timeout: float = Field(
        60.0, validation_alias="COACH_LLM_READ_TIMEOUT"
    )

    # --- Function Calling / Chatbot 설정 ---
    chatbot_model_name: Optional[str] = Field(
        None, validation_alias="CHATBOT_MODEL_NAME"
    )
    chat_cache_admin_enabled: bool = Field(
        False, validation_alias="CHAT_CACHE_ADMIN_ENABLED"
    )
    chat_cache_admin_token: Optional[str] = Field(
        None, validation_alias="CHAT_CACHE_ADMIN_TOKEN"
    )

    # --- Moderation 설정 ---
    moderation_high_risk_keywords_raw: str = Field(
        "죽어,죽인다,살인,테러,시발,씨발,병신,개새끼",
        validation_alias="MODERATION_HIGH_RISK_KEYWORDS",
    )
    moderation_spam_keywords_raw: str = Field(
        "광고,홍보,문의,오픈채팅,텔레그램,카카오톡,디엠,수익",
        validation_alias="MODERATION_SPAM_KEYWORDS",
    )
    moderation_spam_url_threshold: int = Field(
        3, validation_alias="MODERATION_SPAM_URL_THRESHOLD"
    )
    moderation_repeated_char_threshold: int = Field(
        8, validation_alias="MODERATION_REPEATED_CHAR_THRESHOLD"
    )
    moderation_spam_medium_score: int = Field(
        2, validation_alias="MODERATION_SPAM_MEDIUM_SCORE"
    )
    moderation_spam_block_score: int = Field(
        3, validation_alias="MODERATION_SPAM_BLOCK_SCORE"
    )

    # --- 검색(Retrieval) 관련 설정 ---
    default_search_limit: int = Field(3, validation_alias="DEFAULT_SEARCH_LIMIT")

    # --- SSE / 채팅 관련 설정 ---
    # Coach 분석 등 상세 응답에 충분한 토큰 수 필요 (기본값 4096)
    max_output_tokens: int = Field(4096, validation_alias="MAX_OUTPUT_TOKENS")

    # --- Monitoring ---
    sentry_dsn: Optional[str] = Field(None, validation_alias="SENTRY_DSN")

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
            raise ValueError(
                f"지원되지 않는 LLM_PROVIDER '{value}'입니다. 다음 중에서 선택하세요: {sorted(allowed)}"
            )
        return value

    @field_validator("coach_llm_provider")
    def _validate_coach_llm_provider(cls, value: str) -> str:
        """`coach_llm_provider` 필드의 값이 지원되는 프로바이더 중 하나인지 검증합니다.

        Note: Coach feature only supports OpenRouter (via openrouter_model setting).
        """
        allowed = {"openrouter"}
        if value not in allowed:
            raise ValueError(
                f"지원되지 않는 COACH_LLM_PROVIDER '{value}'입니다. Coach는 OpenRouter만 지원합니다."
            )
        return value

    @field_validator(
        "moderation_spam_url_threshold",
        "moderation_repeated_char_threshold",
        "moderation_spam_medium_score",
        "moderation_spam_block_score",
    )
    def _validate_positive_threshold(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Moderation threshold 값은 1 이상이어야 합니다.")
        return value

    @property
    def cors_allowed_origins(self) -> List[str]:
        """CORS 정책에 따라 허용된 출처 목록을 반환합니다."""
        if "*" in self.cors_origins:
            return [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:5176",
                "http://127.0.0.1:5176",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ]
        return self.cors_origins

    @property
    def database_url(self) -> str:
        """AI 서비스가 사용할 PostgreSQL 연결 URL을 반환합니다."""
        return self.source_db_url

    @property
    def source_db_url(self) -> str:
        """배치/마이그레이션 스크립트용 Source DB URL을 반환합니다.

        우선순위:
        1) POSTGRES_DB_URL
        2) SUPABASE_DB_URL (deprecated fallback)
        """
        if self.postgres_db_url:
            return self.postgres_db_url

        if self.legacy_source_db_url:
            if not self._legacy_source_db_warned:
                logger.warning(
                    "SUPABASE_DB_URL is deprecated and will be removed. Use POSTGRES_DB_URL instead."
                )
                self._legacy_source_db_warned = True
            return self.legacy_source_db_url

        raise RuntimeError("POSTGRES_DB_URL is not configured.")

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

    @property
    def moderation_high_risk_keywords(self) -> List[str]:
        return self._split_csv(self.moderation_high_risk_keywords_raw)

    @property
    def moderation_spam_keywords(self) -> List[str]:
        return self._split_csv(self.moderation_spam_keywords_raw)

    def _split_csv(self, raw_value: str) -> List[str]:
        if not raw_value:
            return []
        return [item.strip().lower() for item in raw_value.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """설정 객체를 반환합니다.

    `lru_cache`를 사용하여 설정 객체를 한 번만 생성하고 캐시하여,
    애플리케이션 전체에서 동일한 설정 인스턴스를 사용하도록 보장합니다.
    """
    return Settings()
