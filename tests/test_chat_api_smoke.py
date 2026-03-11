from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.routers import chat_stream


class _FakeAgent:
    async def process_query(self, question: str, context: dict[str, Any] | None = None):
        return {
            "answer": "모의 응답: " + str(question),
            "tool_calls": [],
            "tool_results": [],
            "data_sources": ["mock"],
            "verified": True,
            "visualizations": [],
            "intent": "test",
            "error": None,
        }


@pytest.fixture
def client(monkeypatch):
    test_app = FastAPI()

    @test_app.get("/health")
    def health():
        return {"status": "ok"}

    test_app.include_router(chat_stream.router)

    test_app.dependency_overrides[chat_stream.get_agent] = lambda: _FakeAgent()
    test_app.dependency_overrides[chat_stream.rate_limit_chat_dependency] = lambda: None

    monkeypatch.setattr(
        "app.deps.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr("app.deps.record_security_event", lambda *args, **kwargs: None)

    with TestClient(test_app) as client_:
        yield client_


@pytest.fixture
def ai_internal_headers():
    return {"X-Internal-Api-Key": "expected-token"}


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ai_chat_completion_requires_internal_token(client: TestClient):
    response = client.post(
        "/ai/chat/completion",
        json={"question": "현재 경기 관련 모의 질의"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid internal API token"


def test_ai_chat_completion_with_internal_token_returns_success(
    client: TestClient,
    ai_internal_headers: dict[str, str],
):
    response = client.post(
        "/ai/chat/completion",
        json={"question": "현재 경기 관련 모의 질의"},
        headers=ai_internal_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "test"
    assert isinstance(body["answer"], str)
    assert "모의 응답" in body["answer"]
