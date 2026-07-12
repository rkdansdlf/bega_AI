import pytest

from app.config import LOCAL_DEV_AI_INTERNAL_TOKEN, Settings


GEMMA_VISION_MODEL = "google/gemma-4-31b-it:free"
MISTRAL_VISION_FALLBACK_MODEL = "mistralai/mistral-small-3.2-24b-instruct"


def test_settings_default_vision_model_uses_gemma_vision(monkeypatch):
    monkeypatch.delenv("VISION_MODEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.vision_model == GEMMA_VISION_MODEL


def test_settings_default_vision_fallback_models_uses_mistral_vision(monkeypatch):
    monkeypatch.delenv("VISION_FALLBACK_MODELS", raising=False)

    settings = Settings(_env_file=None)

    assert settings.vision_fallback_models == [MISTRAL_VISION_FALLBACK_MODEL]


def test_chat_model_pricing_json_is_validated(monkeypatch):
    monkeypatch.setenv(
        "CHAT_MODEL_PRICING_JSON",
        '{"openrouter":{"vendor/planner":{"input_usd_per_1m_tokens":"1.00","output_usd_per_1m_tokens":"2.00"}}}',
    )

    settings = Settings(_env_file=None)

    assert settings.chat_model_pricing_json is not None

    monkeypatch.setenv("CHAT_MODEL_PRICING_JSON", "not-json")

    with pytest.raises(ValueError, match="CHAT_MODEL_PRICING_JSON"):
        Settings(_env_file=None)


def test_resolved_ai_internal_token_prefers_explicit_value(monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "explicit-token")
    monkeypatch.setenv("CORS_ORIGINS", "https://www.begabaseball.xyz")

    settings = Settings()

    assert settings.resolved_ai_internal_token == "explicit-token"


def test_resolved_ai_internal_token_uses_local_dev_fallback(monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5176,http://127.0.0.1:5176")

    settings = Settings()

    assert settings.is_local_dev_environment is True
    assert settings.resolved_ai_internal_token == LOCAL_DEV_AI_INTERNAL_TOKEN


def test_resolved_ai_internal_token_requires_explicit_value_for_non_local_origins(
    monkeypatch,
):
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "")
    monkeypatch.setenv("CORS_ORIGINS", "https://www.begabaseball.xyz")

    settings = Settings()

    assert settings.is_local_dev_environment is False
    assert settings.resolved_ai_internal_token is None


def test_resolved_token_skips_local_fallback_when_app_env_production(monkeypatch):
    # 운영 표식이 있으면 로컬 origin이어도 공개 폴백 토큰을 반환하지 않는다.
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5176,http://127.0.0.1:5176")

    settings = Settings()

    assert settings.is_production_environment is True
    assert settings.resolved_ai_internal_token is None


def test_validate_internal_token_security_raises_when_prod_token_missing(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "")
    monkeypatch.setenv("CORS_ORIGINS", "https://www.begabaseball.xyz")

    settings = Settings()

    with pytest.raises(RuntimeError, match="AI_INTERNAL_TOKEN must be configured"):
        settings.validate_internal_token_security()


def test_validate_internal_token_security_raises_when_prod_uses_local_default(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", LOCAL_DEV_AI_INTERNAL_TOKEN)
    monkeypatch.setenv("CORS_ORIGINS", "https://www.begabaseball.xyz")

    settings = Settings()

    with pytest.raises(RuntimeError, match="must not be set to the local-dev default"):
        settings.validate_internal_token_security()


def test_validate_internal_token_security_passes_with_explicit_prod_token(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "a-strong-prod-token")
    monkeypatch.setenv("CORS_ORIGINS", "https://www.begabaseball.xyz")

    settings = Settings()

    # 예외 없이 통과해야 한다.
    settings.validate_internal_token_security()
    assert settings.resolved_ai_internal_token == "a-strong-prod-token"


def test_security_surface_defaults_disable_docs_and_metrics_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "a-strong-prod-token")
    monkeypatch.delenv("AI_DOCS_ENABLED", raising=False)
    monkeypatch.delenv("AI_METRICS_ENABLED", raising=False)

    settings = Settings()

    assert settings.api_docs_enabled is False
    assert settings.metrics_enabled is False


def test_security_surface_allows_explicit_metrics_enablement(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "a-strong-prod-token")
    monkeypatch.setenv("AI_DOCS_ENABLED", "false")
    monkeypatch.setenv("AI_METRICS_ENABLED", "true")

    settings = Settings()

    assert settings.api_docs_enabled is False
    assert settings.metrics_enabled is True


def test_validate_internal_token_security_noop_when_app_env_unset(monkeypatch):
    # APP_ENV 미설정이면 기존 동작 유지: 가드는 통과하고 로컬 폴백도 살아있다.
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5176,http://127.0.0.1:5176")

    settings = Settings()

    settings.validate_internal_token_security()
    assert settings.resolved_ai_internal_token == LOCAL_DEV_AI_INTERNAL_TOKEN


def test_settings_warns_when_openrouter_llm_and_openai_embeddings_are_mixed(
    monkeypatch, caplog
):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("EMBED_PROVIDER", "openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    caplog.clear()

    Settings()

    assert "Embeddings will call OpenAI directly, not OpenRouter." in caplog.text


def test_settings_default_embeddings_use_openrouter(monkeypatch):
    monkeypatch.delenv("EMBED_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_EMBED_MODEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.embed_provider == "openrouter"
    assert settings.openrouter_embed_model == "openai/text-embedding-3-small"
    assert settings.embed_dim == 256


def test_cors_origins_accepts_json_array(monkeypatch):
    monkeypatch.setenv(
        "CORS_ORIGINS",
        '["https://www.begabaseball.xyz","https://api.begabaseball.xyz"]',
    )

    settings = Settings()

    assert settings.cors_origins == [
        "https://www.begabaseball.xyz",
        "https://api.begabaseball.xyz",
    ]


def test_cors_origins_accepts_csv(monkeypatch):
    monkeypatch.setenv(
        "CORS_ORIGINS",
        "https://www.begabaseball.xyz,https://api.begabaseball.xyz",
    )

    settings = Settings()

    assert settings.cors_origins == [
        "https://www.begabaseball.xyz",
        "https://api.begabaseball.xyz",
    ]


def test_chat_planner_cache_ttl_defaults_to_five_minutes(monkeypatch):
    monkeypatch.delenv("CHAT_PLANNER_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.delenv("CHAT_PLANNER_CACHE_HISTORY_TTL_SECONDS", raising=False)

    settings = Settings()

    assert settings.chat_planner_cache_ttl_seconds == 300
    assert settings.chat_planner_cache_history_ttl_seconds == 60


def test_chat_semantic_cache_shadow_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("CHAT_SEMANTIC_CACHE_SHADOW_ENABLED", raising=False)

    settings = Settings()

    assert settings.chat_semantic_cache_shadow_enabled is False


def test_chat_semantic_cache_shadow_can_be_enabled(monkeypatch):
    monkeypatch.setenv("CHAT_SEMANTIC_CACHE_SHADOW_ENABLED", "true")

    settings = Settings()

    assert settings.chat_semantic_cache_shadow_enabled is True


def test_chat_semantic_cache_vector_index_defaults_to_safe_off(monkeypatch):
    monkeypatch.delenv("CHAT_SEMANTIC_CACHE_VECTOR_INDEX_ENABLED", raising=False)
    monkeypatch.delenv("CHAT_SEMANTIC_CACHE_HNSW_EF_SEARCH", raising=False)

    settings = Settings()

    assert settings.chat_semantic_cache_vector_index_enabled is False
    assert settings.chat_semantic_cache_hnsw_ef_search == 0


def test_chat_semantic_cache_vector_index_options_can_be_enabled(monkeypatch):
    monkeypatch.setenv("CHAT_SEMANTIC_CACHE_VECTOR_INDEX_ENABLED", "true")
    monkeypatch.setenv("CHAT_SEMANTIC_CACHE_HNSW_EF_SEARCH", "64")

    settings = Settings()

    assert settings.chat_semantic_cache_vector_index_enabled is True
    assert settings.chat_semantic_cache_hnsw_ef_search == 64


def test_chat_semantic_cache_similarity_must_be_probability(monkeypatch):
    monkeypatch.setenv("CHAT_SEMANTIC_CACHE_MIN_SIMILARITY", "1.01")

    with pytest.raises(ValueError, match="CHAT_SEMANTIC_CACHE_MIN_SIMILARITY"):
        Settings()


def test_chat_semantic_cache_candidate_limit_is_bounded(monkeypatch):
    monkeypatch.setenv("CHAT_SEMANTIC_CACHE_CANDIDATE_LIMIT", "11")

    with pytest.raises(ValueError, match="CHAT_SEMANTIC_CACHE_CANDIDATE_LIMIT"):
        Settings()


def test_chat_semantic_cache_hnsw_search_must_be_non_negative(monkeypatch):
    monkeypatch.setenv("CHAT_SEMANTIC_CACHE_HNSW_EF_SEARCH", "-1")

    with pytest.raises(ValueError, match="CHAT_SEMANTIC_CACHE_HNSW_EF_SEARCH"):
        Settings()


def test_chat_cost_rates_must_be_non_negative(monkeypatch):
    monkeypatch.setenv("CHAT_COST_INPUT_USD_PER_1M_TOKENS", "-0.1")

    with pytest.raises(ValueError, match="Chat cost rate values"):
        Settings()


def test_chat_model_routing_settings_can_be_split(monkeypatch):
    monkeypatch.setenv("CHAT_PLANNER_MODEL_NAME", "openrouter/cheap-planner")
    monkeypatch.setenv("CHAT_ANSWER_MODEL_NAME", "openrouter/quality-answer")

    settings = Settings()

    assert settings.chat_planner_model_name == "openrouter/cheap-planner"
    assert settings.chat_answer_model_name == "openrouter/quality-answer"


def test_source_db_url_prefers_oci_over_postgres(monkeypatch):
    monkeypatch.setenv("OCI_DB_URL", "postgresql://oci/source")
    monkeypatch.setenv("POSTGRES_DB_URL", "postgresql://postgres/fallback")

    settings = Settings()

    assert settings.source_db_url == "postgresql://oci/source"
    assert settings.database_url == "postgresql://oci/source"


def test_source_db_url_falls_back_to_postgres(monkeypatch):
    monkeypatch.setenv("OCI_DB_URL", "")
    monkeypatch.setenv("POSTGRES_DB_URL", "postgresql://postgres/fallback")

    settings = Settings()

    assert settings.source_db_url == "postgresql://postgres/fallback"
