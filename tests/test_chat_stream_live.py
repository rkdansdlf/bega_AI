from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import chat_stream


def _collect_sse_events(response) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for raw_line in response.iter_lines():
        line = (
            raw_line
            if isinstance(raw_line, str)
            else raw_line.decode("utf-8", errors="replace")
        )
        if line.startswith("event:"):
            current["type"] = line[6:].strip()
        elif line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            current["data"] = json.loads(data_str)
        elif not line and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


@pytest.fixture
def stream_app(monkeypatch):
    test_app = FastAPI()
    test_app.include_router(chat_stream.router)
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
    monkeypatch.setattr(
        "app.routers.chat_stream._request_is_disconnected",
        AsyncMock(return_value=False),
    )

    return test_app


def test_stream_route_emits_status_before_delayed_first_token(stream_app) -> None:
    class _DelayedStreamAgent:
        async def process_query_stream(
            self, question: str, context: dict[str, Any] | None = None
        ):
            await asyncio.sleep(0.01)
            yield {"type": "answer_chunk", "content": "첫 청크"}
            yield {
                "type": "metadata",
                "data": {
                    "tool_calls": [],
                    "tool_results": [],
                    "data_sources": [],
                    "verified": True,
                    "visualizations": [],
                    "intent": "test",
                    "planner_mode": "test",
                    "grounding_mode": "test",
                    "source_tier": "test",
                    "answer_sources": [],
                    "as_of_date": "2026-03-19",
                    "fallback_reason": None,
                    "perf": {"model": "test"},
                },
            }

    stream_app.dependency_overrides[chat_stream.get_agent] = (
        lambda: _DelayedStreamAgent()
    )

    with TestClient(stream_app, raise_server_exceptions=False) as client:
        with client.stream(
            "POST",
            "/ai/chat/stream",
            json={"question": "테스트 질문"},
            headers={"X-Internal-Api-Key": "expected-token"},
        ) as response:
            events = _collect_sse_events(response)

    assert response.status_code == 200
    assert events[0]["type"] == "status"


@pytest.mark.asyncio
async def test_live_event_generator_emits_fallback_meta_on_partial_stream_error() -> (
    None
):
    async def _broken_stream():
        yield {"type": "answer_chunk", "content": "부분 응답"}
        raise RuntimeError("stream exploded")

    events: list[dict[str, Any]] = []
    async for event in chat_stream._chat_live_event_generator(
        request=None,
        question="스트림 오류 테스트",
        filters=None,
        style="markdown",
        cache_key=None,
        stream=_broken_stream(),
    ):
        parsed = {"type": event["event"]}
        if event["data"] != "[DONE]":
            parsed["data"] = json.loads(event["data"])
        events.append(parsed)

    meta_events = [event for event in events if event["type"] == "meta"]

    assert meta_events
    assert meta_events[-1]["data"]["fallback_answer_used"] is True
    assert meta_events[-1]["data"]["error"] == "temporary_generation_issue"
    assert meta_events[-1]["data"]["verified"] is False


@pytest.mark.asyncio
async def test_live_event_generator_emits_cancelled_meta_and_done_on_abort() -> None:
    async def _cancelled_stream():
        yield {"type": "answer_chunk", "content": "부분 응답"}
        raise asyncio.CancelledError()

    events: list[dict[str, Any]] = []
    async for event in chat_stream._chat_live_event_generator(
        request=None,
        question="스트림 취소 테스트",
        filters=None,
        style="markdown",
        cache_key=None,
        stream=_cancelled_stream(),
    ):
        parsed = {"type": event["event"]}
        if event["data"] != "[DONE]":
            parsed["data"] = json.loads(event["data"])
        events.append(parsed)

    assert [event["type"] for event in events] == ["status", "message", "meta", "done"]
    meta_events = [event for event in events if event["type"] == "meta"]
    assert meta_events
    meta_data = meta_events[-1]["data"]
    assert meta_data["finish_reason"] == "cancelled"
    assert meta_data["cancelled"] is True
