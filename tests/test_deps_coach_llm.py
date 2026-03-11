import httpx

from app.config import Settings
from app.deps import (
    clamp_coach_openrouter_max_tokens,
    is_retryable_coach_openrouter_error,
    resolve_coach_openrouter_models,
)


def test_coach_openrouter_default_model_uses_openrouter_free(monkeypatch):
    monkeypatch.delenv("COACH_OPENROUTER_MODEL", raising=False)

    settings = Settings()

    assert settings.coach_openrouter_model == "openrouter/free"


def test_resolve_coach_openrouter_models_skips_retired_slug():
    models = resolve_coach_openrouter_models(
        "upstage/solar-pro-3:free",
        ["openrouter/free", "openrouter/free"],
    )

    assert models == ["openrouter/free"]


def test_is_retryable_coach_openrouter_error_distinguishes_404():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response_404 = httpx.Response(404, request=request)
    response_503 = httpx.Response(503, request=request)

    non_retryable = httpx.HTTPStatusError(
        "not found",
        request=request,
        response=response_404,
    )
    retryable = httpx.HTTPStatusError(
        "server unavailable",
        request=request,
        response=response_503,
    )

    assert is_retryable_coach_openrouter_error(non_retryable) is False
    assert is_retryable_coach_openrouter_error(retryable) is True
    assert (
        is_retryable_coach_openrouter_error(
            httpx.RemoteProtocolError("Server disconnected")
        )
        is True
    )


def test_clamp_coach_openrouter_max_tokens_limits_large_requests():
    assert clamp_coach_openrouter_max_tokens(12000) == 4000
    assert clamp_coach_openrouter_max_tokens(128) == 256
