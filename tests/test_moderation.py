from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import ANY, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import moderation


@pytest.fixture
def client() -> TestClient:
    test_app = FastAPI()
    test_app.include_router(moderation.router)
    test_app.dependency_overrides[moderation.require_ai_internal_token] = lambda: None
    with TestClient(test_app) as test_client:
        yield test_client


@pytest.fixture
def secured_client() -> TestClient:
    test_app = FastAPI()
    test_app.include_router(moderation.router)
    with TestClient(test_app) as test_client:
        yield test_client


def _settings(**overrides):
    base = {
        "llm_provider": "openrouter",
        "openrouter_api_key": None,
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "openrouter_model": "openrouter/free",
        "openrouter_referer": "https://kbo-platform.test",
        "openrouter_app_title": "KBO Platform Test",
        "gemini_api_key": None,
        "gemini_model": "gemini-2.0-flash",
        "moderation_high_risk_keywords": ["죽어", "병신"],
        "moderation_spam_keywords": ["광고", "홍보", "오픈채팅"],
        "moderation_spam_url_threshold": 3,
        "moderation_repeated_char_threshold": 8,
        "moderation_spam_medium_score": 2,
        "moderation_spam_block_score": 3,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_moderation_requires_internal_token(
    secured_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )

    response = secured_client.post(
        "/moderation/safety-check",
        json={"content": "오늘 경기 정말 재밌었어요!"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid internal API token"


def test_moderation_accepts_internal_token(
    secured_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("app.routers.moderation.get_settings", lambda: _settings())

    response = secured_client.post(
        "/moderation/safety-check",
        json={"content": "오늘 경기 정말 재밌었어요!"},
        headers={"X-Internal-Api-Key": "expected-token"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "ALLOW"
    assert body["decisionSource"] == "FALLBACK"
    assert body["riskLevel"] == "LOW"


def test_moderation_no_api_key_high_risk_blocks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.routers.moderation.get_settings", lambda: _settings())

    response = client.post("/moderation/safety-check", json={"content": "너 진짜 죽어"})

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "BLOCK"
    assert body["decisionSource"] == "FALLBACK"
    assert body["riskLevel"] == "HIGH"


def test_moderation_no_api_key_low_risk_allows(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.routers.moderation.get_settings", lambda: _settings())

    response = client.post(
        "/moderation/safety-check", json={"content": "오늘 경기 정말 재밌었어요!"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "ALLOW"
    assert body["decisionSource"] == "FALLBACK"
    assert body["riskLevel"] == "LOW"


def test_moderation_openrouter_provider_uses_model_without_gemini_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(openrouter_api_key="openrouter-test-key")
    monkeypatch.setattr("app.routers.moderation.get_settings", lambda: settings)

    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"category":"SAFE","reason":"정상 콘텐츠",'
                                '"action":"ALLOW","riskLevel":"LOW"}'
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        async def post(self, url: str, **kwargs: object) -> FakeResponse:
            captured["url"] = url
            captured.update(kwargs)
            return FakeResponse()

    monkeypatch.setattr(
        moderation,
        "get_shared_httpx_client",
        lambda *args, **kwargs: FakeClient(),
        raising=False,
    )

    response = client.post(
        "/moderation/safety-check",
        json={"content": "오늘 경기 정말 재밌었어요!"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "category": "SAFE",
        "reason": "정상 콘텐츠",
        "action": "ALLOW",
        "decisionSource": "MODEL",
        "riskLevel": "LOW",
    }
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["json"] == {
        "model": "openrouter/free",
        "messages": [
            {
                "role": "user",
                "content": ANY,
            }
        ],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }


def test_moderation_model_runtime_error_uses_fallback_rule(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.routers.moderation.get_settings",
        lambda: _settings(llm_provider="gemini", gemini_api_key="test-key"),
    )
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.models.generate_content.side_effect = RuntimeError("model unavailable")
    monkeypatch.setattr(
        "app.routers.moderation.genai.Client",
        lambda **_: mock_client,
    )

    response = client.post(
        "/moderation/safety-check", json={"content": "오늘 선발 라인업 공유해줘"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "ALLOW"
    assert body["decisionSource"] == "FALLBACK"
    assert body["riskLevel"] == "LOW"


def test_moderation_model_parse_error_high_risk_uses_fallback_block(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.routers.moderation.get_settings",
        lambda: _settings(llm_provider="gemini", gemini_api_key="test-key"),
    )
    mock_response = MagicMock()
    mock_response.text = "not-a-json-response"
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.models.generate_content.return_value = mock_response
    monkeypatch.setattr(
        "app.routers.moderation.genai.Client",
        lambda **_: mock_client,
    )

    response = client.post(
        "/moderation/safety-check",
        json={"content": "광고 링크 확인 https://a.com https://b.com https://c.com"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "BLOCK"
    assert body["decisionSource"] == "FALLBACK"
    assert body["riskLevel"] == "HIGH"
