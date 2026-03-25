from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.routers import chat_stream


class _FakeRequest:
    def __init__(self, disconnected_sequence: list[bool] | None = None):
        self._sequence = list(disconnected_sequence or [])
        self._last = False

    async def is_disconnected(self) -> bool:
        if self._sequence:
            self._last = self._sequence.pop(0)
        return self._last


def _build_result(answer: object) -> dict[str, object]:
    return {
        "answer": answer,
        "tool_calls": [],
        "tool_results": [],
        "data_sources": [],
        "verified": True,
        "visualizations": [],
        "intent": "test",
        "error": None,
        "perf": {
            "total_ms": 10.0,
            "analysis_ms": 1.0,
            "tool_ms": 2.0,
            "answer_ms": 3.0,
            "first_token_ms": 999.0,
        },
    }


async def _collect_events(**kwargs) -> list[dict[str, str]]:
    return [event async for event in chat_stream._chat_event_generator(**kwargs)]


@pytest.mark.asyncio
async def test_chat_event_generator_stops_on_answer_cancellation_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancelled_answer() -> AsyncGenerator[str, None]:
        yield "첫 청크"
        raise asyncio.CancelledError()

    save_to_cache = AsyncMock()
    monkeypatch.setattr(chat_stream, "save_to_cache", save_to_cache)
    monkeypatch.setattr(chat_stream, "get_connection_pool", MagicMock())

    events = await _collect_events(
        request=_FakeRequest(),
        question="취소 테스트",
        filters=None,
        style="markdown",
        result=_build_result(cancelled_answer()),
        error_payload=None,
        cache_key="cancelled-key",
    )

    assert [event["event"] for event in events] == ["status", "message", "meta", "done"]
    assert json.loads(events[1]["data"]) == {"delta": "첫 청크"}
    meta_payload = json.loads(events[2]["data"])
    assert meta_payload["finish_reason"] == "cancelled"
    assert meta_payload["cancelled"] is True
    assert isinstance(meta_payload["perf"]["first_token_ms"], (int, float))
    assert meta_payload["perf"]["first_token_ms"] != 999.0
    save_to_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_event_generator_skips_terminal_events_when_client_disconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def one_chunk_answer() -> AsyncGenerator[str, None]:
        yield "부분 응답"

    save_to_cache = AsyncMock()
    monkeypatch.setattr(chat_stream, "save_to_cache", save_to_cache)
    monkeypatch.setattr(chat_stream, "get_connection_pool", MagicMock())

    events = await _collect_events(
        request=_FakeRequest([False, False, True]),
        question="disconnect 테스트",
        filters=None,
        style="markdown",
        result=_build_result(one_chunk_answer()),
        error_payload=None,
        cache_key="disconnect-key",
    )

    assert [event["event"] for event in events] == ["status", "message"]
    assert json.loads(events[1]["data"]) == {"delta": "부분 응답"}
    save_to_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_event_generator_emits_additive_meta_on_normal_completion() -> None:
    events = await _collect_events(
        request=_FakeRequest(),
        question="정상 완료",
        filters=None,
        style="markdown",
        result=_build_result("완료 응답"),
        error_payload=None,
        cache_key=None,
    )

    assert [event["event"] for event in events] == ["status", "message", "meta", "done"]
    meta_payload = json.loads(events[2]["data"])
    assert meta_payload["finish_reason"] == "completed"
    assert meta_payload["cancelled"] is False
    assert isinstance(meta_payload["perf"]["first_token_ms"], (int, float))
    assert meta_payload["perf"]["first_token_ms"] != 999.0
