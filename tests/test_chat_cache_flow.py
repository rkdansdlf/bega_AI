from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core.chat_cache_key import build_chat_cache_key
from app.deps import get_agent
from app.main import app


class _DummyExecuteResult:
    def __init__(
        self,
        row: Any = None,
        rows: list[tuple[Any, ...]] | None = None,
        rowcount: int = 1,
    ) -> None:
        self._row = row
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows


class _DummyConnection:
    def __init__(self, cache_rows: dict[str, dict[str, Any]]):
        self._cache_rows = cache_rows

    def execute(self, query: str, params: tuple[Any, ...] | None = None):
        normalized = " ".join(str(query).split()).lower()
        now_ref = datetime.now(timezone.utc)

        if "create table if not exists coach_analysis_cache" in normalized:
            return _DummyExecuteResult(rowcount=0)

        if "create table if not exists chat_response_cache" in normalized:
            return _DummyExecuteResult(rowcount=0)

        if (
            "select response_text, intent, model_name, hit_count, expires_at" in normalized
            and "from chat_response_cache" in normalized
        ):
            cache_key = str((params or ("",))[0])
            row = self._cache_rows.get(cache_key)
            if not row or row["expires_at"] <= now_ref:
                return _DummyExecuteResult(None, rowcount=0)
            return _DummyExecuteResult(
                (
                    row["response_text"],
                    row["intent"],
                    row["model_name"],
                    row["hit_count"],
                    row["expires_at"],
                ),
                rowcount=1,
            )

        if "insert into chat_response_cache" in normalized:
            (
                cache_key,
                question_text,
                filters_json,
                intent,
                response_text,
                model_name,
                expires_at,
            ) = params or ("", "", None, None, "", None, now_ref)
            self._cache_rows[str(cache_key)] = {
                "question_text": question_text,
                "filters_json": filters_json,
                "intent": intent,
                "response_text": response_text,
                "model_name": model_name,
                "hit_count": 0,
                "expires_at": expires_at,
                "created_at": now_ref,
            }
            return _DummyExecuteResult(rowcount=1)

        if "update chat_response_cache set hit_count = hit_count + 1" in normalized:
            cache_key = str((params or ("",))[0])
            row = self._cache_rows.get(cache_key)
            if row:
                row["hit_count"] += 1
                return _DummyExecuteResult(rowcount=1)
            return _DummyExecuteResult(rowcount=0)

        if (
            "select intent," in normalized
            and "round(avg(hit_count)::numeric, 2) as avg_hits" in normalized
            and "from chat_response_cache" in normalized
        ):
            grouped: dict[str | None, list[int]] = {}
            for row in self._cache_rows.values():
                if row["expires_at"] <= now_ref:
                    continue
                grouped.setdefault(row.get("intent"), []).append(row.get("hit_count", 0))

            rows: list[tuple[Any, ...]] = []
            for intent, hits in sorted(
                grouped.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            ):
                avg_hits = round(sum(hits) / len(hits), 2) if hits else 0.0
                rows.append((intent, len(hits), avg_hits))
            return _DummyExecuteResult(rows=rows, rowcount=len(rows))

        if "delete from chat_response_cache where intent = %s" in normalized:
            intent = (params or ("",))[0]
            keys_to_delete = [
                key for key, row in self._cache_rows.items() if row.get("intent") == intent
            ]
            for key in keys_to_delete:
                del self._cache_rows[key]
            return _DummyExecuteResult(rowcount=len(keys_to_delete))

        if "delete from chat_response_cache where cache_key = %s" in normalized:
            cache_key = str((params or ("",))[0])
            existed = cache_key in self._cache_rows
            self._cache_rows.pop(cache_key, None)
            return _DummyExecuteResult(rowcount=1 if existed else 0)

        return _DummyExecuteResult(rowcount=0)


class _DummyConnectionContext:
    def __init__(self, conn: _DummyConnection):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyPool:
    def __init__(self):
        self.cache_rows: dict[str, dict[str, Any]] = {}
        self._conn = _DummyConnection(self.cache_rows)

    def connection(self):
        return _DummyConnectionContext(self._conn)


class _FakeAgent:
    def __init__(self, intent: str = "stats_lookup"):
        self.intent = intent
        self.call_count = 0

    async def process_query(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.call_count += 1
        return {
            "answer": f"mock-answer-{self.call_count}:{question}",
            "tool_calls": [],
            "tool_results": [],
            "data_sources": ["mock"],
            "verified": True,
            "visualizations": [],
            "intent": self.intent,
            "error": None,
        }


def _clear_settings_cache() -> None:
    try:
        get_settings.cache_clear()
    except AttributeError:
        pass


@contextmanager
def _client_with_dummy_pool(
    monkeypatch: pytest.MonkeyPatch,
    agent: _FakeAgent,
) -> Iterator[tuple[TestClient, _DummyPool]]:
    class _NeverExitEvent:
        async def wait(self):
            await asyncio.sleep(3600)

    dummy_pool = _DummyPool()
    app.dependency_overrides[get_agent] = lambda: agent
    monkeypatch.setattr("app.deps.get_connection_pool", lambda: dummy_pool)
    monkeypatch.setattr("app.deps.close_connection_pool", lambda: None)
    monkeypatch.setattr("app.routers.chat_stream.get_connection_pool", lambda: dummy_pool)
    monkeypatch.setattr(
        "sse_starlette.sse.AppStatus.should_exit_event",
        _NeverExitEvent(),
        raising=False,
    )
    monkeypatch.setattr(
        "sse_starlette.sse.AppStatus.should_exit",
        False,
        raising=False,
    )

    with TestClient(app) as test_client:
        yield test_client, dummy_pool

    app.dependency_overrides.clear()


def _assert_cached_flag(payload_text: str, expected: bool) -> None:
    expected_text = f'"cached": {"true" if expected else "false"}'
    compact_text = f'"cached":{"true" if expected else "false"}'
    assert expected_text in payload_text or compact_text in payload_text


def test_same_question_second_request_hits_cache(monkeypatch: pytest.MonkeyPatch):
    _clear_settings_cache()
    agent = _FakeAgent(intent="stats_lookup")

    with _client_with_dummy_pool(monkeypatch, agent) as (client, pool):
        payload = {"question": "LG 트윈스 시즌 요약해줘"}

        first_response = client.post("/ai/chat/stream", json=payload)
        assert first_response.status_code == 200
        _assert_cached_flag(first_response.text, expected=False)
        assert len(pool.cache_rows) == 1

        second_response = client.post("/ai/chat/stream", json=payload)
        assert second_response.status_code == 200
        _assert_cached_flag(second_response.text, expected=True)
        assert agent.call_count == 1


def test_temporal_keyword_question_bypasses_cache(monkeypatch: pytest.MonkeyPatch):
    _clear_settings_cache()
    agent = _FakeAgent(intent="stats_lookup")

    with _client_with_dummy_pool(monkeypatch, agent) as (client, pool):
        payload = {"question": "오늘 경기 결과 알려줘"}

        first_response = client.post("/ai/chat/stream", json=payload)
        second_response = client.post("/ai/chat/stream", json=payload)

        assert first_response.status_code == 200
        assert second_response.status_code == 200
        _assert_cached_flag(first_response.text, expected=False)
        _assert_cached_flag(second_response.text, expected=False)
        assert len(pool.cache_rows) == 0
        assert agent.call_count == 2


def test_expired_cache_key_is_refreshed_by_upsert(monkeypatch: pytest.MonkeyPatch):
    _clear_settings_cache()
    agent = _FakeAgent(intent="stats_lookup")

    with _client_with_dummy_pool(monkeypatch, agent) as (client, pool):
        question = "삼성 불펜 분석 알려줘"
        payload = {"question": question}

        first_response = client.post("/ai/chat/stream", json=payload)
        assert first_response.status_code == 200
        _assert_cached_flag(first_response.text, expected=False)

        cache_key, _ = build_chat_cache_key(
            question=question,
            filters=None,
            schema_version="v1",
        )
        assert cache_key in pool.cache_rows
        pool.cache_rows[cache_key]["expires_at"] = datetime.now(timezone.utc) - timedelta(
            seconds=1
        )

        second_response = client.post("/ai/chat/stream", json=payload)
        assert second_response.status_code == 200
        _assert_cached_flag(second_response.text, expected=False)
        assert "mock-answer-2:" in second_response.text

        third_response = client.post("/ai/chat/stream", json=payload)
        assert third_response.status_code == 200
        _assert_cached_flag(third_response.text, expected=True)
        assert "mock-answer-2:" in third_response.text
        assert agent.call_count == 2


def test_intent_ttl_is_applied_when_saving(monkeypatch: pytest.MonkeyPatch):
    _clear_settings_cache()
    agent = _FakeAgent(intent="recent_form")

    with _client_with_dummy_pool(monkeypatch, agent) as (client, pool):
        question = "KT 전력 요약해줘"
        payload = {"question": question}

        response = client.post("/ai/chat/stream", json=payload)
        assert response.status_code == 200
        _assert_cached_flag(response.text, expected=False)

        cache_key, _ = build_chat_cache_key(
            question=question,
            filters=None,
            schema_version="v1",
        )
        saved = pool.cache_rows[cache_key]
        assert saved["intent"] == "recent_form"

        now_ref = datetime.now(timezone.utc)
        ttl_seconds = (saved["expires_at"] - now_ref).total_seconds()
        assert 10_500 <= ttl_seconds <= 11_100


def test_cache_admin_api_disabled_by_default_returns_404(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("CHAT_CACHE_ADMIN_ENABLED", raising=False)
    monkeypatch.delenv("CHAT_CACHE_ADMIN_TOKEN", raising=False)
    _clear_settings_cache()

    agent = _FakeAgent()
    with _client_with_dummy_pool(monkeypatch, agent) as (client, _):
        stats_response = client.get("/ai/chat/cache/stats")
        assert stats_response.status_code == 404

        flush_response = client.delete("/ai/chat/cache", params={"intent": "stats_lookup"})
        assert flush_response.status_code == 404

        invalidate_response = client.delete("/ai/chat/cache/sample-key")
        assert invalidate_response.status_code == 404


def test_cache_admin_api_enabled_requires_valid_token(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CHAT_CACHE_ADMIN_ENABLED", "true")
    monkeypatch.setenv("CHAT_CACHE_ADMIN_TOKEN", "secret-token")
    _clear_settings_cache()

    agent = _FakeAgent()
    with _client_with_dummy_pool(monkeypatch, agent) as (client, _):
        no_token_stats = client.get("/ai/chat/cache/stats")
        assert no_token_stats.status_code == 401

        no_token_flush = client.delete(
            "/ai/chat/cache",
            params={"intent": "stats_lookup"},
        )
        assert no_token_flush.status_code == 401

        no_token_invalidate = client.delete("/ai/chat/cache/sample-key")
        assert no_token_invalidate.status_code == 401

        wrong_token_stats = client.get(
            "/ai/chat/cache/stats",
            headers={"X-Cache-Admin-Token": "wrong-token"},
        )
        assert wrong_token_stats.status_code == 401

        wrong_token_flush = client.delete(
            "/ai/chat/cache",
            params={"intent": "stats_lookup"},
            headers={"X-Cache-Admin-Token": "wrong-token"},
        )
        assert wrong_token_flush.status_code == 401

        wrong_token_invalidate = client.delete(
            "/ai/chat/cache/sample-key",
            headers={"X-Cache-Admin-Token": "wrong-token"},
        )
        assert wrong_token_invalidate.status_code == 401

        ok_stats = client.get(
            "/ai/chat/cache/stats",
            headers={"X-Cache-Admin-Token": "secret-token"},
        )
        assert ok_stats.status_code == 200
        assert ok_stats.json() == {"stats": []}

        ok_flush = client.delete(
            "/ai/chat/cache",
            params={"intent": "stats_lookup"},
            headers={"X-Cache-Admin-Token": "secret-token"},
        )
        assert ok_flush.status_code == 200
        assert ok_flush.json() == {"deleted": 0, "intent": "stats_lookup"}

        ok_invalidate = client.delete(
            "/ai/chat/cache/sample-key",
            headers={"X-Cache-Admin-Token": "secret-token"},
        )
        assert ok_invalidate.status_code == 200
        assert ok_invalidate.json() == {"deleted": 0, "cache_key": "sample-key"}

    _clear_settings_cache()


def test_cache_admin_api_enabled_without_token_returns_503(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CHAT_CACHE_ADMIN_ENABLED", "true")
    monkeypatch.delenv("CHAT_CACHE_ADMIN_TOKEN", raising=False)
    _clear_settings_cache()

    agent = _FakeAgent()
    with _client_with_dummy_pool(monkeypatch, agent) as (client, _):
        stats_response = client.get("/ai/chat/cache/stats")
        assert stats_response.status_code == 503

        flush_response = client.delete("/ai/chat/cache", params={"intent": "stats_lookup"})
        assert flush_response.status_code == 503

        invalidate_response = client.delete("/ai/chat/cache/sample-key")
        assert invalidate_response.status_code == 503

    _clear_settings_cache()
