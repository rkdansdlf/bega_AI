"""Route-level compatibility tests for chat stream version negotiation."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from unittest.mock import AsyncMock

from app.routers import chat_stream


class _UnusedAgent:
    pass


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    app = FastAPI()
    app.include_router(chat_stream.router)
    app.dependency_overrides[chat_stream.get_agent] = lambda: _UnusedAgent()
    app.dependency_overrides[chat_stream.require_ai_internal_token] = lambda: None
    monkeypatch.setattr(
        chat_stream,
        "get_settings",
        lambda: SimpleNamespace(chat_sse_ping_seconds=15),
    )
    monkeypatch.setattr(
        chat_stream,
        "_build_static_chat_result",
        AsyncMock(
            return_value={
                "answer": "정적 응답",
                "tool_calls": [],
                "tool_results": [],
                "data_sources": [],
                "verified": True,
                "visualizations": [],
                "intent": "static_test",
                "planner_mode": "fast_path",
            }
        ),
    )
    with TestClient(app) as test_client:
        yield test_client


def test_missing_version_header_preserves_v1_wire(client: TestClient) -> None:
    response = client.post("/ai/chat/stream", json={"question": "테스트"})

    assert response.status_code == 200
    assert response.headers["X-AI-Event-Version"] == "1"
    assert "event: status" in response.text
    assert "event: message" in response.text
    assert "event: meta" in response.text
    assert "data: [DONE]" in response.text
    assert '"version":2' not in response.text


def test_explicit_v2_header_returns_typed_envelopes(client: TestClient) -> None:
    response = client.post(
        "/ai/chat/stream",
        json={"question": "테스트"},
        headers={"X-AI-Event-Version": "2"},
    )

    assert response.status_code == 200
    assert response.headers["X-AI-Event-Version"] == "2"
    assert "event: chat.status" in response.text
    assert "event: chat.message.delta" in response.text
    assert "event: chat.meta" in response.text
    assert "event: stream.done" in response.text
    assert '"version":2' in response.text
    assert "data: [DONE]" not in response.text


def test_unsupported_version_fails_before_stream(client: TestClient) -> None:
    response = client.post(
        "/ai/chat/stream",
        json={"question": "테스트"},
        headers={"X-AI-Event-Version": "3"},
    )

    assert response.status_code == 406
    assert response.json()["detail"] == {
        "code": "AI_EVENT_VERSION_UNSUPPORTED",
        "supported_versions": ["1", "2"],
    }
