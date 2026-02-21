from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import moderation


@pytest.fixture
def client() -> TestClient:
    test_app = FastAPI()
    test_app.include_router(moderation.router)
    with TestClient(test_app) as test_client:
        yield test_client


def _settings(**overrides):
    base = {
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


def test_moderation_model_runtime_error_uses_fallback_rule(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.routers.moderation.get_settings",
        lambda: _settings(gemini_api_key="test-key"),
    )
    monkeypatch.setattr(
        "app.routers.moderation.genai.configure",
        lambda **_: None,
    )

    mock_model = MagicMock()
    mock_model.generate_content.side_effect = RuntimeError("model unavailable")
    monkeypatch.setattr(
        "app.routers.moderation.genai.GenerativeModel",
        lambda *_: mock_model,
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
        lambda: _settings(gemini_api_key="test-key"),
    )
    monkeypatch.setattr(
        "app.routers.moderation.genai.configure",
        lambda **_: None,
    )

    mock_response = MagicMock()
    mock_response.text = "not-a-json-response"
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    monkeypatch.setattr(
        "app.routers.moderation.genai.GenerativeModel",
        lambda *_: mock_model,
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
