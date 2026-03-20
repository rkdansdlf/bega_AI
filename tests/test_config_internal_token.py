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
