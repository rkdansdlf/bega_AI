from __future__ import annotations

import base64
import json
import math
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from unittest.mock import AsyncMock, MagicMock

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
            "planner_mode": "default_llm_planner",
            "planner_cache_hit": True,
            "tool_execution_mode": "parallel",
            "perf": {
                "planner_cache_hit": True,
                "tool_execution_mode": "parallel",
                "tool_count": 2,
            },
            "error": None,
        }


class _FakePipeline:
    """Minimal RAGPipeline stub for smoke tests.

    chat_completion() now calls ``pipeline.run()`` (via Depends(get_rag_pipeline))
    instead of ``agent.process_query()``, so we stub the new dependency here.
    """

    async def run(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        history: list | None = None,
    ):
        return {
            "answer": "모의 응답: " + str(question),
            "tool_calls": [],
            "tool_results": [],
            "data_sources": ["mock"],
            "verified": True,
            "visualizations": [],
            "intent": "test",
            "planner_mode": "default_llm_planner",
            "planner_cache_hit": True,
            "tool_execution_mode": "parallel",
            "perf": {
                "planner_cache_hit": True,
                "tool_execution_mode": "parallel",
                "tool_count": 2,
            },
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
    test_app.dependency_overrides[chat_stream.get_rag_pipeline] = lambda: _FakePipeline()
    test_app.dependency_overrides[chat_stream.rate_limit_chat_dependency] = lambda: None

    monkeypatch.setattr(
        "app.deps.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr("app.deps.record_security_event", lambda *args, **kwargs: None)
    mock_pool = MagicMock()
    mock_conn_ctx = MagicMock()
    mock_conn_ctx.__enter__ = MagicMock(return_value=MagicMock())
    mock_conn_ctx.__exit__ = MagicMock(return_value=False)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)
    monkeypatch.setattr(
        "app.routers.chat_stream.get_connection_pool", lambda: mock_pool
    )
    monkeypatch.setattr(
        "app.routers.chat_stream.get_cached_response",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "app.routers.chat_stream.save_to_cache",
        AsyncMock(return_value=None),
    )

    with TestClient(test_app) as client_:
        yield client_


@pytest.fixture
def ai_internal_headers():
    return {"X-Internal-Api-Key": "expected-token"}


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_safe_serialize_replaces_non_finite_floats() -> None:
    payload = chat_stream._safe_serialize(
        {"citations": [{"similarity": math.nan}, {"similarity": math.inf}]}
    )

    assert payload == {"citations": [{"similarity": None}, {"similarity": None}]}


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
    assert body["planner_cache_hit"] is True
    assert body["tool_execution_mode"] == "parallel"
    assert body["perf"]["planner_cache_hit"] is True


def test_ai_chat_completion_cache_hit_sets_cache_planner_metadata(
    client: TestClient,
    ai_internal_headers: dict[str, str],
    monkeypatch,
):
    monkeypatch.setattr(
        "app.routers.chat_stream.get_cached_response",
        AsyncMock(
            return_value={
                "response_text": "캐시 응답",
                "intent": "team_analysis",
                "hit_count": 3,
            }
        ),
    )

    response = client.post(
        "/ai/chat/completion",
        json={"question": "LG 팀 흐름 정리해줘"},
        headers=ai_internal_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["cached"] is True
    assert body["planner_mode"] == "cache"
    assert body["planner_cache_hit"] is False
    assert body["tool_execution_mode"] == "none"
    assert body["perf"]["planner_mode"] == "cache"


@pytest.mark.asyncio
async def test_chat_live_event_generator_emits_pipeline_answer_chunks():
    async def stream():
        yield {"type": "metadata", "data": {"intent": "freeform", "verified": True}}
        yield {"type": "answer_chunk", "content": "HELLO"}

    events = [
        event
        async for event in chat_stream._chat_live_event_generator(
            request=None,
            question="test",
            filters=None,
            style="markdown",
            cache_key=None,
            stream=stream(),
        )
    ]

    message_events = [event for event in events if event["event"] == "message"]
    assert len(message_events) == 1
    assert json.loads(message_events[0]["data"]) == {"delta": "HELLO"}


@pytest.mark.asyncio
async def test_chat_stream_get_passes_validated_question_and_history(monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_stream_response(request, question, **kwargs):
        captured["request"] = request
        captured["question"] = question
        captured.update(kwargs)
        return {"ok": True}

    history = [{"role": "user", "content": "이전 질문"}]
    history_payload = base64.b64encode(
        json.dumps(history, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    request = SimpleNamespace(query_params={"history": history_payload})
    monkeypatch.setattr(chat_stream, "_stream_response", fake_stream_response)

    result = await chat_stream.chat_stream_get(
        q="  LG 흐름 알려줘  ",
        style="markdown",
        _=None,
        __=None,
        pipeline=object(),
        request=request,
    )

    assert result == {"ok": True}
    assert captured["question"] == "LG 흐름 알려줘"
    assert captured["history"] == history
    assert captured["filters"] is None
    assert captured["cache_key"] is None
