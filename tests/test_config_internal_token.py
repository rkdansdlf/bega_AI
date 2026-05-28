from app.config import LOCAL_DEV_AI_INTERNAL_TOKEN, Settings


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


def test_chat_planner_cache_ttl_defaults_to_60_seconds(monkeypatch):
    monkeypatch.delenv("CHAT_PLANNER_CACHE_TTL_SECONDS", raising=False)

    settings = Settings()

    assert settings.chat_planner_cache_ttl_seconds == 60


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
