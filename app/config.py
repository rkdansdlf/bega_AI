"""서비스의 환경 설정을 관리하는 Pydantic 설정 모듈입니다.

이 모듈은 `pydantic-settings`를 사용하여 .env 파일 또는 환경 변수에서
애플리케이션 설정을 로드하고 유효성을 검사하는 `Settings` 클래스를 정의합니다.
"""

import logging
import json
from functools import lru_cache
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import Field, PrivateAttr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.chat_model_usage import ModelPricingCatalog

logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5176",
    "http://127.0.0.1:5176",
    "http://localhost:5177",
    "http://127.0.0.1:5177",
]
LOCAL_DEV_AI_INTERNAL_TOKEN = "local-dev-ai-internal-token"
LOCAL_DEV_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", "host.docker.internal"}
DEFAULT_VISION_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_VISION_FALLBACK_MODELS = "mistralai/mistral-small-3.2-24b-instruct"


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
        hide_input_in_errors=True,
        populate_by_name=True,
    )

    # --- 기본 애플리케이션 설정 ---
    app_name: str = "KBO AI Service"
    debug: bool = False
    # 배포 환경 명시 신호. prod 배포에서 APP_ENV=production 으로 설정하면
    # 로컬 전용 내부 토큰 폴백이 비활성화되고 기동 시 토큰 검증 가드가 작동한다.
    # 미설정(빈 값)이면 CORS origin 휴리스틱으로 로컬 여부를 추론한다(기존 동작 유지).
    app_env: str = Field("", validation_alias="APP_ENV")
    # CORS 설정(자격 증명 쿠키 사용 시 '* '는 허용되지 않음)
    # 로컬 개발 기준 기본값은 Vite/개발 서버에서 사용되는 대표 도메인으로 제한
    cors_origins_raw: str = Field(
        default=",".join(DEFAULT_CORS_ORIGINS),
        validation_alias="CORS_ORIGINS",
    )
    ai_docs_enabled: Optional[bool] = Field(None, validation_alias="AI_DOCS_ENABLED")
    ai_metrics_enabled: Optional[bool] = Field(
        None, validation_alias="AI_METRICS_ENABLED"
    )

    # --- 데이터베이스 설정 ---
    # 운영 Source DB 경로
    oci_db_url: Optional[str] = Field(None, validation_alias="OCI_DB_URL")
    # RAG/로컬 fallback 경로
    postgres_db_url: Optional[str] = Field(None, validation_alias="POSTGRES_DB_URL")
    # 하위 호환 경로 (deprecated)
    legacy_source_db_url: Optional[str] = Field(
        None, validation_alias="SUPABASE_DB_URL"
    )
    # `auto` keeps local/dev startup compatibility. `managed` requires the
    # migration role to provision the schema before the AI process starts.
    ai_db_schema_mode: str = Field("auto", validation_alias="AI_DB_SCHEMA_MODE")
    _legacy_source_db_warned: bool = PrivateAttr(default=False)

    def model_post_init(self, __context) -> None:
        if (
            self.llm_provider == "openrouter"
            and self.embed_provider == "openai"
            and self.openrouter_api_key
        ):
            logger.warning(
                "LLM_PROVIDER=openrouter but EMBED_PROVIDER=openai. "
                "Embeddings will call OpenAI directly, not OpenRouter. "
                "If you intend to use OpenRouter embeddings, set EMBED_PROVIDER=openrouter "
                "and OPENROUTER_EMBED_MODEL explicitly."
            )

    # --- LLM / 임베딩 프로바이더 설정 ---
    # LLM(거대 언어 모델) 및 임베딩 생성을 위해 사용할 서비스를 지정합니다.
    llm_provider: str = Field("gemini", validation_alias="LLM_PROVIDER")
    embed_provider: str = Field("openrouter", validation_alias="EMBED_PROVIDER")
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
        "openrouter/free", validation_alias="OPENROUTER_MODEL"
    )
    # Pydantic Settings tries to parse List[str] as JSON. read as str to avoid error.
    # Default: no fallback by default; use OPENROUTER_FALLBACK_MODELS when explicitly set.
    openrouter_fallback_models_raw: str = Field(
        "", validation_alias="OPENROUTER_FALLBACK_MODELS"
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
        "openai/text-embedding-3-small", validation_alias="OPENROUTER_EMBED_MODEL"
    )
    vision_model: str = Field(DEFAULT_VISION_MODEL, validation_alias="VISION_MODEL")
    vision_fallback_models_raw: str = Field(
        DEFAULT_VISION_FALLBACK_MODELS,
        validation_alias="VISION_FALLBACK_MODELS",
    )

    # --- Coach LLM 설정 ---

    # 내부 AI 호출 인증 토큰(헤더 또는 Bearer 토큰 지원)
    # - 운영 기본 정책은 BFF(/api/ai/*) 경유이며, 외부 direct AI 공개는 금지합니다.
    # - direct AI 호출을 허용하는 내부 환경에서는 AI_INTERNAL_TOKEN이 원칙적으로 필수입니다.
    # - 운영/CI 기준에서 AI_INTERNAL_TOKEN 미설정 상태의 direct AI 호출은
    #   503(Service Unavailable, AI_INTERNAL_AUTH_MISCONFIGURED)로 차단됩니다.
    # - 로컬 환경 전용 폴백이 활성화되면 미설정 상태에서도 내부 토큰을 기본값으로 채워
    #   토큰 누락 요청은 401(Invalid internal API token)으로 응답할 수 있습니다.
    ai_internal_token: Optional[str] = Field(
        None,
        validation_alias="AI_INTERNAL_TOKEN",
    )

    coach_llm_provider: str = Field("openrouter", validation_alias="COACH_LLM_PROVIDER")
    coach_max_output_tokens: int = Field(
        2500, validation_alias="COACH_MAX_OUTPUT_TOKENS"
    )  # 2500 tokens: detailed_markdown 900자 + coach_note 200자 허용에 맞춰 상향
    coach_openrouter_model: str = Field(
        "openrouter/free", validation_alias="COACH_OPENROUTER_MODEL"
    )
    coach_openrouter_fallback_models_raw: str = Field(
        "openrouter/free",
        validation_alias="COACH_OPENROUTER_FALLBACK_MODELS",
    )
    coach_brief_max_output_tokens: int = Field(
        8000, validation_alias="COACH_BRIEF_MAX_OUTPUT_TOKENS"
    )

    # --- Coach FAILED_LOCKED 자동 복구 루프 ---
    # 기본 비활성: 운영자가 명시적으로 켤 때만 백그라운드 자동 삭제가 동작한다.
    coach_failed_recovery_enabled: bool = Field(
        False, validation_alias="COACH_FAILED_RECOVERY_ENABLED"
    )
    coach_failed_recovery_interval_seconds: int = Field(
        1800, validation_alias="COACH_FAILED_RECOVERY_INTERVAL_SECONDS"
    )
    # 재시도 가능 오류로 잠긴 row를 풀기 전 최소 경과 시간(쿨다운).
    coach_failed_recovery_cooldown_seconds: int = Field(
        3600, validation_alias="COACH_FAILED_RECOVERY_COOLDOWN_SECONDS"
    )
    # 한 사이클당 처리 상한(무한 재생성/LLM 비용 폭주 방지).
    coach_failed_recovery_max_rows_per_cycle: int = Field(
        50, validation_alias="COACH_FAILED_RECOVERY_MAX_ROWS_PER_CYCLE"
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
    chat_planner_model_name: Optional[str] = Field(
        None, validation_alias="CHAT_PLANNER_MODEL_NAME"
    )
    chat_answer_model_name: Optional[str] = Field(
        None, validation_alias="CHAT_ANSWER_MODEL_NAME"
    )
    chat_model_pricing_json: Optional[str] = Field(
        None, validation_alias="CHAT_MODEL_PRICING_JSON"
    )
    chat_cache_admin_enabled: bool = Field(
        False, validation_alias="CHAT_CACHE_ADMIN_ENABLED"
    )
    chat_cache_admin_token: Optional[str] = Field(
        None, validation_alias="CHAT_CACHE_ADMIN_TOKEN"
    )
    chat_semantic_cache_enabled: bool = Field(
        False, validation_alias="CHAT_SEMANTIC_CACHE_ENABLED"
    )
    chat_semantic_cache_shadow_enabled: bool = Field(
        False, validation_alias="CHAT_SEMANTIC_CACHE_SHADOW_ENABLED"
    )
    chat_semantic_cache_min_similarity: float = Field(
        0.93, validation_alias="CHAT_SEMANTIC_CACHE_MIN_SIMILARITY"
    )
    chat_semantic_cache_candidate_limit: int = Field(
        3, validation_alias="CHAT_SEMANTIC_CACHE_CANDIDATE_LIMIT"
    )
    chat_semantic_cache_vector_index_enabled: bool = Field(
        False, validation_alias="CHAT_SEMANTIC_CACHE_VECTOR_INDEX_ENABLED"
    )
    chat_semantic_cache_hnsw_ef_search: int = Field(
        0, validation_alias="CHAT_SEMANTIC_CACHE_HNSW_EF_SEARCH"
    )
    chat_cost_input_usd_per_1m_tokens: float = Field(
        0.0, validation_alias="CHAT_COST_INPUT_USD_PER_1M_TOKENS"
    )
    chat_cost_output_usd_per_1m_tokens: float = Field(
        0.0, validation_alias="CHAT_COST_OUTPUT_USD_PER_1M_TOKENS"
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
    # 200,000+ chunk 환경에서 top-k=3은 recall이 낮아 오답률을 높인다.
    # 운영(.env)은 24로 오버라이드되어 있고, 본 기본값은 dev/test 시 합리적인 동작을 보장.
    default_search_limit: int = Field(10, validation_alias="DEFAULT_SEARCH_LIMIT")
    default_kbo_season_year: Optional[int] = Field(
        None, validation_alias="DEFAULT_KBO_SEASON_YEAR"
    )
    retrieval_single_query_for_strict_entity: bool = Field(
        True, validation_alias="RETRIEVAL_SINGLE_QUERY_FOR_STRICT_ENTITY"
    )
    retrieval_multi_query_rule_variation_max: int = Field(
        3, validation_alias="RETRIEVAL_MULTI_QUERY_RULE_VARIATION_MAX"
    )
    retrieval_multi_query_limit_per_query: int = Field(
        8, validation_alias="RETRIEVAL_MULTI_QUERY_LIMIT_PER_QUERY"
    )
    retrieval_fallback_limit_relaxed: int = Field(
        20, validation_alias="RETRIEVAL_FALLBACK_LIMIT_RELAXED"
    )
    retrieval_fallback_limit_minimal: int = Field(
        25, validation_alias="RETRIEVAL_FALLBACK_LIMIT_MINIMAL"
    )
    # 벡터 인덱스 타입 선택:
    #   "hnsw"     - HNSW 세션 GUC(hnsw.ef_search)만 설정. 운영 HNSW 인덱스가 있을 때 사용.
    #   "ivfflat"  - IVFFlat 세션 GUC(ivfflat.probes)만 설정. 레거시 인덱스 유지 시 사용.
    #   "auto"     - 기동 시 pg_indexes에서 HNSW 존재 여부를 감지하여 자동 선택(기본값).
    # 운영 전환 흐름: 256-prefix migration → create_vector_index.py 실행(HNSW 생성)
    # → AI_VECTOR_INDEX=hnsw 배포(halfvec 사용 시 AI_VECTOR_QUANTIZATION=halfvec) → ivfflat 제거.
    ai_vector_index: str = Field("auto", validation_alias="AI_VECTOR_INDEX")
    ai_vector_quantization: str = Field(
        "none", validation_alias="AI_VECTOR_QUANTIZATION"
    )
    # IVFFlat 인덱스(`idx_rag_chunks_embedding`, lists=644) 기준 probes=512는 79% 버킷 스캔을
    # 의미하여 사실상 시퀀셜 스캔에 가깝다. 권장 운영치는 24~64 범위. dev/test 기본을 보수적으로
    # 32로 설정하고, 정확도 회귀가 발견되면 RETRIEVAL_IVFFLAT_PROBES 환경변수로 조정한다.
    retrieval_ivfflat_probes: int = Field(
        32, validation_alias="RETRIEVAL_IVFFLAT_PROBES"
    )
    retrieval_hnsw_ef_search: int = Field(
        100, validation_alias="RETRIEVAL_HNSW_EF_SEARCH"
    )
    retrieval_statement_timeout_ms: int = Field(
        8000, validation_alias="RETRIEVAL_STATEMENT_TIMEOUT_MS"
    )
    rag_chunk_target_chars: int = Field(650, validation_alias="RAG_CHUNK_TARGET_CHARS")
    rag_chunk_max_chars: int = Field(900, validation_alias="RAG_CHUNK_MAX_CHARS")
    rag_chunk_min_chars: int = Field(180, validation_alias="RAG_CHUNK_MIN_CHARS")
    rag_chunk_overlap_chars: int = Field(80, validation_alias="RAG_CHUNK_OVERLAP_CHARS")
    rag_storage_dedup_enabled: bool = Field(
        True, validation_alias="RAG_STORAGE_DEDUP_ENABLED"
    )
    rag_quality_min_chars: int = Field(50, validation_alias="RAG_QUALITY_MIN_CHARS")
    rag_embedding_version: int = Field(2, validation_alias="RAG_EMBEDDING_VERSION")
    rag_chunking_version: int = Field(1, validation_alias="RAG_CHUNKING_VERSION")
    rag_retrieval_active_filter_enabled: bool = Field(
        True, validation_alias="RAG_RETRIEVAL_ACTIVE_FILTER_ENABLED"
    )
    rag_retrieval_event_logging_enabled: bool = Field(
        True, validation_alias="RAG_RETRIEVAL_EVENT_LOGGING_ENABLED"
    )
    rag_rerank_enabled: bool = Field(False, validation_alias="RAG_RERANK_ENABLED")
    rag_rerank_candidate_limit: int = Field(
        20, validation_alias="RAG_RERANK_CANDIDATE_LIMIT"
    )
    rag_context_limit: int = Field(10, validation_alias="RAG_CONTEXT_LIMIT")

    # --- SSE / 채팅 관련 설정 ---
    # Coach 분석 등 상세 응답에 충분한 토큰 수 필요 (기본값 4096)
    max_output_tokens: int = Field(4096, validation_alias="MAX_OUTPUT_TOKENS")
    chat_completion_timeout_seconds: float = Field(
        0.0, validation_alias="CHAT_COMPLETION_TIMEOUT_SECONDS"
    )
    chat_sse_ping_seconds: int = Field(15, validation_alias="CHAT_SSE_PING_SECONDS")
    chat_cached_stream_chunk_size: int = Field(
        200, validation_alias="CHAT_CACHED_STREAM_CHUNK_SIZE"
    )
    chat_queue_enabled: bool = Field(True, validation_alias="CHAT_QUEUE_ENABLED")
    chat_queue_rpm_limit: int = Field(18, validation_alias="CHAT_QUEUE_RPM_LIMIT")
    chat_queue_window_seconds: int = Field(
        60, validation_alias="CHAT_QUEUE_WINDOW_SECONDS"
    )
    chat_queue_max_size: int = Field(60, validation_alias="CHAT_QUEUE_MAX_SIZE")
    chat_queue_max_wait_seconds: int = Field(
        180, validation_alias="CHAT_QUEUE_MAX_WAIT_SECONDS"
    )
    chat_queue_status_interval_seconds: int = Field(
        1, validation_alias="CHAT_QUEUE_STATUS_INTERVAL_SECONDS"
    )
    chat_dynamic_token_enabled: bool = Field(
        True, validation_alias="CHAT_DYNAMIC_TOKEN_ENABLED"
    )
    chat_analysis_max_tokens: int = Field(
        350, validation_alias="CHAT_ANALYSIS_MAX_TOKENS"
    )
    chat_answer_max_tokens_short: int = Field(
        1400, validation_alias="CHAT_ANSWER_MAX_TOKENS_SHORT"
    )
    chat_answer_max_tokens_long: int = Field(
        2600, validation_alias="CHAT_ANSWER_MAX_TOKENS_LONG"
    )
    chat_answer_max_tokens_team: int = Field(
        900, validation_alias="CHAT_ANSWER_MAX_TOKENS_TEAM"
    )
    chat_tool_result_max_chars: int = Field(
        2200, validation_alias="CHAT_TOOL_RESULT_MAX_CHARS"
    )
    chat_tool_result_max_items: int = Field(
        8, validation_alias="CHAT_TOOL_RESULT_MAX_ITEMS"
    )
    chat_first_token_watchdog_seconds: float = Field(
        20.0, validation_alias="CHAT_FIRST_TOKEN_WATCHDOG_SECONDS"
    )
    chat_first_token_retry_max_attempts: int = Field(
        1, validation_alias="CHAT_FIRST_TOKEN_RETRY_MAX_ATTEMPTS"
    )
    chat_stream_first_token_watchdog_seconds: float = Field(
        10.0, validation_alias="CHAT_STREAM_FIRST_TOKEN_WATCHDOG_SECONDS"
    )
    chat_stream_first_token_retry_max_attempts: int = Field(
        1, validation_alias="CHAT_STREAM_FIRST_TOKEN_RETRY_MAX_ATTEMPTS"
    )
    chat_tool_parallel_enabled: bool = Field(
        True, validation_alias="CHAT_TOOL_PARALLEL_ENABLED"
    )
    chat_tool_parallel_split_batch_enabled: bool = Field(
        True, validation_alias="CHAT_TOOL_PARALLEL_SPLIT_BATCH_ENABLED"
    )
    chat_tool_parallel_serial_tools_raw: str = Field(
        "", validation_alias="CHAT_TOOL_PARALLEL_SERIAL_TOOLS"
    )
    chat_tool_parallel_max_concurrency: int = Field(
        2, validation_alias="CHAT_TOOL_PARALLEL_MAX_CONCURRENCY"
    )
    chat_openrouter_empty_chunk_retries: int = Field(
        2, validation_alias="CHAT_OPENROUTER_EMPTY_CHUNK_RETRIES"
    )
    chat_openrouter_empty_chunk_backoff_ms: int = Field(
        400, validation_alias="CHAT_OPENROUTER_EMPTY_CHUNK_BACKOFF_MS"
    )
    chat_perf_metrics_enabled: bool = Field(
        True, validation_alias="CHAT_PERF_METRICS_ENABLED"
    )
    chat_fast_path_enabled: bool = Field(
        True, validation_alias="CHAT_FAST_PATH_ENABLED"
    )
    chat_fast_path_scope: str = Field("team", validation_alias="CHAT_FAST_PATH_SCOPE")
    chat_fast_path_min_messages: int = Field(
        0, validation_alias="CHAT_FAST_PATH_MIN_MESSAGES"
    )
    chat_fast_path_tool_cap: int = Field(2, validation_alias="CHAT_FAST_PATH_TOOL_CAP")
    chat_fast_path_fallback_on_empty: bool = Field(
        True, validation_alias="CHAT_FAST_PATH_FALLBACK_ON_EMPTY"
    )
    operator_data_fast_path_enabled: bool = Field(
        False, validation_alias="OPERATOR_DATA_FAST_PATH_ENABLED"
    )
    chat_planner_cache_ttl_seconds: int = Field(
        300, validation_alias="CHAT_PLANNER_CACHE_TTL_SECONDS"
    )
    chat_planner_cache_history_ttl_seconds: int = Field(
        60, validation_alias="CHAT_PLANNER_CACHE_HISTORY_TTL_SECONDS"
    )
    chat_planner_cache_max_entries: int = Field(
        512, validation_alias="CHAT_PLANNER_CACHE_MAX_ENTRIES"
    )
    chat_team_answer_cap_base: int = Field(
        520, validation_alias="CHAT_TEAM_ANSWER_CAP_BASE"
    )
    chat_team_answer_cap_heavy: int = Field(
        650, validation_alias="CHAT_TEAM_ANSWER_CAP_HEAVY"
    )
    chat_team_answer_cap_brief: int = Field(
        460, validation_alias="CHAT_TEAM_ANSWER_CAP_BRIEF"
    )
    chat_team_answer_cap_high_complexity: int = Field(
        600, validation_alias="CHAT_TEAM_ANSWER_CAP_HIGH_COMPLEXITY"
    )
    chat_team_answer_cap_stream: int = Field(
        420, validation_alias="CHAT_TEAM_ANSWER_CAP_STREAM"
    )
    chat_team_answer_cap_fast_path_stream: int = Field(
        360, validation_alias="CHAT_TEAM_ANSWER_CAP_FAST_PATH_STREAM"
    )
    chat_team_answer_cap_fast_path_completion: int = Field(
        460, validation_alias="CHAT_TEAM_ANSWER_CAP_FAST_PATH_COMPLETION"
    )
    chat_completion_answer_cap_team: int = Field(
        880, validation_alias="CHAT_COMPLETION_ANSWER_CAP_TEAM"
    )
    chat_completion_answer_cap_general: int = Field(
        1200, validation_alias="CHAT_COMPLETION_ANSWER_CAP_GENERAL"
    )

    # --- 임베딩 설정 ---
    embed_batch_size: int = Field(32, validation_alias="EMBED_BATCH_SIZE")
    embed_dim: int = Field(256, validation_alias="EMBED_DIM")
    hf_embed_model: str = Field(
        "intfloat/multilingual-e5-large", validation_alias="HF_EMBED_MODEL"
    )
    hf_embed_batch: int = Field(16, validation_alias="HF_BATCH")
    gemini_embed_max_tokens: int = Field(3072, validation_alias="GEMINI_MAX_TOKENS")
    gemini_embed_rpm: int = Field(60, validation_alias="GEMINI_RPM")

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

    @field_validator("ai_db_schema_mode")
    def _validate_ai_db_schema_mode(cls, value: str) -> str:
        allowed = {"auto", "managed"}
        if value not in allowed:
            raise ValueError(
                f"AI_DB_SCHEMA_MODE must be one of {sorted(allowed)}"
            )
        return value

    @field_validator("chat_openrouter_empty_chunk_retries")
    def _validate_chat_openrouter_empty_chunk_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("CHAT_OPENROUTER_EMPTY_CHUNK_RETRIES must be >= 0")
        return value

    @field_validator("chat_openrouter_empty_chunk_backoff_ms")
    def _validate_chat_openrouter_empty_chunk_backoff_ms(cls, value: int) -> int:
        if value < 50:
            raise ValueError("CHAT_OPENROUTER_EMPTY_CHUNK_BACKOFF_MS must be >= 50")
        return value

    @field_validator("chat_completion_timeout_seconds")
    def _validate_chat_completion_timeout_seconds(cls, value: float) -> float:
        if value < 0:
            raise ValueError("CHAT_COMPLETION_TIMEOUT_SECONDS must be >= 0")
        return value

    @field_validator("chat_first_token_watchdog_seconds")
    def _validate_chat_first_token_watchdog_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("CHAT_FIRST_TOKEN_WATCHDOG_SECONDS must be > 0")
        return value

    @field_validator("chat_stream_first_token_watchdog_seconds")
    def _validate_chat_stream_first_token_watchdog_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("CHAT_STREAM_FIRST_TOKEN_WATCHDOG_SECONDS must be > 0")
        return value

    @field_validator("chat_first_token_retry_max_attempts")
    def _validate_chat_first_token_retry_max_attempts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("CHAT_FIRST_TOKEN_RETRY_MAX_ATTEMPTS must be >= 0")
        return value

    @field_validator("chat_stream_first_token_retry_max_attempts")
    def _validate_chat_stream_first_token_retry_max_attempts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("CHAT_STREAM_FIRST_TOKEN_RETRY_MAX_ATTEMPTS must be >= 0")
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

    @field_validator(
        "chat_analysis_max_tokens",
        "chat_answer_max_tokens_short",
        "chat_answer_max_tokens_long",
        "chat_tool_result_max_chars",
        "chat_tool_result_max_items",
        "chat_semantic_cache_candidate_limit",
        "chat_tool_parallel_max_concurrency",
        "chat_fast_path_tool_cap",
        "chat_planner_cache_ttl_seconds",
        "chat_planner_cache_max_entries",
        "default_search_limit",
        "retrieval_multi_query_rule_variation_max",
        "retrieval_multi_query_limit_per_query",
        "retrieval_fallback_limit_relaxed",
        "retrieval_fallback_limit_minimal",
        "retrieval_ivfflat_probes",
        "retrieval_hnsw_ef_search",
        "retrieval_statement_timeout_ms",
        "rag_chunk_target_chars",
        "rag_chunk_max_chars",
        "rag_chunk_min_chars",
        "rag_chunk_overlap_chars",
        "rag_quality_min_chars",
        "rag_embedding_version",
        "rag_chunking_version",
        "rag_rerank_candidate_limit",
        "rag_context_limit",
        "chat_sse_ping_seconds",
        "chat_cached_stream_chunk_size",
        "chat_team_answer_cap_base",
        "chat_team_answer_cap_heavy",
        "chat_team_answer_cap_brief",
        "chat_team_answer_cap_high_complexity",
        "chat_team_answer_cap_stream",
        "chat_team_answer_cap_fast_path_stream",
        "chat_team_answer_cap_fast_path_completion",
        "chat_completion_answer_cap_team",
        "chat_completion_answer_cap_general",
        "embed_batch_size",
        "embed_dim",
        "hf_embed_batch",
        "gemini_embed_max_tokens",
        "gemini_embed_rpm",
    )
    def _validate_chat_positive_threshold(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Chat optimization threshold 값은 1 이상이어야 합니다.")
        return value

    @field_validator("chat_semantic_cache_min_similarity")
    def _validate_chat_semantic_cache_min_similarity(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("CHAT_SEMANTIC_CACHE_MIN_SIMILARITY must be > 0 and <= 1")
        return value

    @field_validator("chat_semantic_cache_candidate_limit")
    def _validate_chat_semantic_cache_candidate_limit(cls, value: int) -> int:
        if value > 10:
            raise ValueError("CHAT_SEMANTIC_CACHE_CANDIDATE_LIMIT must be <= 10")
        return value

    @field_validator("chat_semantic_cache_hnsw_ef_search")
    def _validate_chat_semantic_cache_hnsw_ef_search(cls, value: int) -> int:
        if value < 0:
            raise ValueError("CHAT_SEMANTIC_CACHE_HNSW_EF_SEARCH must be >= 0")
        if value > 1000:
            raise ValueError("CHAT_SEMANTIC_CACHE_HNSW_EF_SEARCH must be <= 1000")
        return value

    @field_validator(
        "chat_cost_input_usd_per_1m_tokens",
        "chat_cost_output_usd_per_1m_tokens",
    )
    def _validate_chat_cost_rates(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Chat cost rate values must be >= 0")
        return value

    @field_validator("chat_model_pricing_json")
    def _validate_chat_model_pricing_json(
        cls, value: Optional[str]
    ) -> Optional[str]:
        try:
            ModelPricingCatalog.from_json(value)
        except ValueError as exc:
            raise ValueError(f"CHAT_MODEL_PRICING_JSON is invalid: {exc}") from exc
        return value

    @field_validator("rag_chunk_overlap_chars")
    def _validate_chunk_overlap_threshold(cls, value: int) -> int:
        if value < 0:
            raise ValueError("RAG_CHUNK_OVERLAP_CHARS must be >= 0")
        return value

    @field_validator("chat_fast_path_min_messages")
    def _validate_chat_fast_path_min_messages(cls, value: int) -> int:
        if value < 0:
            raise ValueError("CHAT_FAST_PATH_MIN_MESSAGES must be >= 0")
        return value

    @field_validator("chat_fast_path_scope")
    def _validate_chat_fast_path_scope(cls, value: str) -> str:
        allowed = {"team"}
        if value not in allowed:
            raise ValueError(
                f"지원되지 않는 CHAT_FAST_PATH_SCOPE '{value}'입니다. 다음 중에서 선택하세요: {sorted(allowed)}"
            )
        return value

    @property
    def chat_tool_parallel_serial_tools(self) -> List[str]:
        return self._parse_string_list(self.chat_tool_parallel_serial_tools_raw)

    @property
    def vision_fallback_models(self) -> List[str]:
        return self._parse_string_list(self.vision_fallback_models_raw)

    @property
    def cors_allowed_origins(self) -> List[str]:
        """CORS 정책에 따라 허용된 출처 목록을 반환합니다."""
        origins = self.cors_origins
        if "*" in origins:
            return DEFAULT_CORS_ORIGINS
        return origins

    @property
    def cors_origins(self) -> List[str]:
        """CORS_ORIGINS 값을 JSON 배열/콤마 구분 문자열 모두 허용해 파싱합니다."""
        parsed = self._parse_string_list(self.cors_origins_raw)
        return parsed or DEFAULT_CORS_ORIGINS

    @property
    def is_local_dev_environment(self) -> bool:
        origins = self.cors_origins
        if not origins:
            return False
        return all(self._is_local_origin(origin) for origin in origins)

    @property
    def is_production_environment(self) -> bool:
        """배포가 운영(production)인지 판별한다.

        APP_ENV이 명시되면 그 값을 신뢰하고, 미설정이면 CORS origin이 모두
        로컬인지로 추론한다(기존 휴리스틱).
        """
        env = (self.app_env or "").strip().lower()
        if env in {"prod", "production"}:
            return True
        if env in {"local", "dev", "development", "test", "testing", "ci"}:
            return False
        return not self.is_local_dev_environment

    @property
    def api_docs_enabled(self) -> bool:
        if self.ai_docs_enabled is not None:
            return self.ai_docs_enabled
        return not self.is_production_environment

    @property
    def metrics_enabled(self) -> bool:
        if self.ai_metrics_enabled is not None:
            return self.ai_metrics_enabled
        return not self.is_production_environment

    @property
    def resolved_ai_internal_token(self) -> Optional[str]:
        configured = (self.ai_internal_token or "").strip()
        if configured:
            return configured
        # 로컬 개발 편의 폴백 — 운영 환경에서는 절대 사용하지 않는다.
        if self.is_local_dev_environment and not self.is_production_environment:
            return LOCAL_DEV_AI_INTERNAL_TOKEN
        return None

    def validate_internal_token_security(self) -> None:
        """운영 배포(APP_ENV=production)에서 내부 토큰 설정이 안전한지 기동 시 검증한다.

        - 토큰 미설정: 공개 폴백으로 빠질 위험을 막기 위해 기동을 거부한다.
        - 토큰이 공개된 로컬 기본값과 동일: 명백한 오설정이므로 기동을 거부한다.

        로컬/개발/테스트 및 미설정 환경에서는 편의 폴백을 유지하기 위해 통과시킨다.
        """
        env = (self.app_env or "").strip().lower()
        if env not in {"prod", "production"}:
            return
        configured = (self.ai_internal_token or "").strip()
        if not configured:
            raise RuntimeError(
                "AI_INTERNAL_TOKEN must be configured when APP_ENV=production; "
                "the local-dev fallback token is disabled in production."
            )
        if configured == LOCAL_DEV_AI_INTERNAL_TOKEN:
            raise RuntimeError(
                "AI_INTERNAL_TOKEN must not be set to the local-dev default value "
                "when APP_ENV=production."
            )

    @staticmethod
    def _parse_string_list(raw_value: str) -> List[str]:
        raw = (raw_value or "").strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    normalized = [
                        str(item).strip() for item in parsed if str(item).strip()
                    ]
                    if normalized:
                        return normalized
            except Exception:
                logger.warning(
                    "Failed to parse list-like setting as JSON list; fallback to CSV parsing."
                )
        return [item.strip() for item in raw.split(",") if item.strip()]

    @staticmethod
    def _is_local_origin(origin: str) -> bool:
        try:
            hostname = (urlparse(origin).hostname or "").strip().lower()
        except ValueError:
            return False
        return hostname in LOCAL_DEV_HOSTS

    @property
    def database_url(self) -> str:
        """AI 서비스가 사용할 PostgreSQL 연결 URL을 반환합니다."""
        return self.source_db_url

    @property
    def source_db_url(self) -> str:
        """배치/마이그레이션 스크립트용 Source DB URL을 반환합니다.

        우선순위:
        1) OCI_DB_URL
        2) POSTGRES_DB_URL
        3) SUPABASE_DB_URL (deprecated fallback)
        """
        if self.oci_db_url:
            return self.oci_db_url

        if self.postgres_db_url:
            return self.postgres_db_url

        if self.legacy_source_db_url:
            if not self._legacy_source_db_warned:
                logger.warning(
                    "SUPABASE_DB_URL is deprecated and will be removed. Use POSTGRES_DB_URL instead."
                )
                self._legacy_source_db_warned = True
            return self.legacy_source_db_url

        raise RuntimeError("OCI_DB_URL or POSTGRES_DB_URL is not configured.")

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
