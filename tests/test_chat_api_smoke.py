from __future__ import annotations

import asyncio
import base64
import json
import math
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.routers import chat_stream


class _FakeUsage:
    def to_dict(self) -> dict[str, Any]:
        return {
            "role": "planner",
            "provider": "openrouter",
            "model": "vendor/planner",
            "outcome": "success",
            "pricing_source": "model_catalog",
            "input_chars": 12,
            "output_chars": 4,
            "input_tokens": 4,
            "output_tokens": 2,
            "input_cost_usd": "0.000004000000",
            "output_cost_usd": "0.000004000000",
            "total_cost_usd": "0.000008000000",
        }


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
            "model_usage": [_FakeUsage()],
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


class _AsyncOnlyConnectionContext:
    def __init__(self) -> None:
        self.connection = object()
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        return False


class _AsyncOnlyPool:
    def __init__(self, context: _AsyncOnlyConnectionContext) -> None:
        self.context = context

    def connection(self) -> _AsyncOnlyConnectionContext:
        return self.context


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
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )
    mock_pool = MagicMock()
    mock_conn_ctx = MagicMock()
    mock_conn_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
    mock_conn_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)
    monkeypatch.setattr(
        "app.routers.chat_stream.get_connection_pool", lambda: mock_pool
    )
    monkeypatch.setattr(
        "app.routers.chat_stream.get_cached_response",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "app.routers.chat_stream._get_semantic_cache_hit",
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
    assert body["model_usage"] == [_FakeUsage().to_dict()]
    assert body["model_usage_complete"] is True


def test_ai_chat_completion_serializes_usage_after_answer_generator_consumes(
    client: TestClient,
    ai_internal_headers: dict[str, str],
    monkeypatch,
):
    usage: list[_FakeUsage] = []

    async def fake_process_query(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ):
        del self, question, context

        async def answer_stream():
            yield "first"
            usage.append(_FakeUsage())
            yield " second"

        return {
            "answer": answer_stream(),
            "tool_calls": [],
            "tool_results": [],
            "model_usage": usage,
            "error": None,
        }

    monkeypatch.setattr(_FakeAgent, "process_query", fake_process_query)

    response = client.post(
        "/ai/chat/completion",
        json={"question": "현재 경기 관련 모의 질의"},
        headers=ai_internal_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "first second"
    assert body["model_usage"] == [_FakeUsage().to_dict()]
    assert body["model_usage_complete"] is True


def test_ai_chat_completion_marks_terminal_answer_failure_incomplete_and_skips_cache(
    client: TestClient,
    ai_internal_headers: dict[str, str],
    monkeypatch,
):
    async def fake_process_query(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ):
        del self, question, context

        async def answer_stream():
            yield "partial"
            raise RuntimeError("terminal answer failure")

        return {
            "answer": answer_stream(),
            "tool_calls": [],
            "tool_results": [],
            "model_usage": [_FakeUsage()],
            "verified": True,
            "error": None,
        }

    save_caches = AsyncMock(return_value=None)
    token_estimate = MagicMock()
    monkeypatch.setattr(_FakeAgent, "process_query", fake_process_query)
    monkeypatch.setattr(chat_stream, "_save_chat_response_caches", save_caches)
    monkeypatch.setattr(chat_stream, "record_chat_token_estimate", token_estimate)

    response = client.post(
        "/ai/chat/completion",
        json={"question": "현재 경기 관련 모의 질의"},
        headers=ai_internal_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "partial"
    assert body["error"] == "temporary_generation_issue"
    assert body["verified"] is False
    assert body["model_usage_complete"] is False
    save_caches.assert_not_awaited()
    token_estimate.assert_not_called()


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
    assert body["model_usage"] == []
    assert body["model_usage_complete"] is True


def test_ai_chat_completion_cache_bypass_ignores_exact_cache_hit(
    client: TestClient,
    ai_internal_headers: dict[str, str],
    monkeypatch,
):
    cache_lookup = AsyncMock(
        return_value={
            "response_text": "캐시 응답",
            "intent": "team_analysis",
            "hit_count": 3,
        }
    )
    monkeypatch.setattr("app.routers.chat_stream.get_cached_response", cache_lookup)

    response = client.post(
        "/ai/chat/completion",
        json={"question": "LG 팀 흐름 정리해줘", "cache_bypass": True},
        headers=ai_internal_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["cached"] is False
    assert "모의 응답" in body["answer"]
    cache_lookup.assert_not_awaited()


def test_ai_chat_completion_semantic_cache_hit_sets_metadata(
    client: TestClient,
    ai_internal_headers: dict[str, str],
    monkeypatch,
):
    monkeypatch.setattr(
        "app.routers.chat_stream.get_settings",
        lambda: SimpleNamespace(
            operator_data_fast_path_enabled=False,
            chat_completion_timeout_seconds=0,
            chat_semantic_cache_enabled=True,
            chat_semantic_cache_shadow_enabled=False,
        ),
    )
    monkeypatch.setattr(
        "app.routers.chat_stream.get_cached_response",
        AsyncMock(return_value=None),
    )
    semantic_hit = AsyncMock(
        return_value={
            "cache_key": "semantic-cache-key",
            "question_text": "LG 팀 분위기 정리해줘",
            "filters_json": None,
            "response_text": "의미 캐시 응답",
            "intent": "team_analysis",
            "source_tier": "rag",
            "hit_count": 2,
            "similarity": 0.948,
        }
    )
    monkeypatch.setattr("app.routers.chat_stream._get_semantic_cache_hit", semantic_hit)

    response = client.post(
        "/ai/chat/completion",
        json={"question": "LG 팀 분위기 정리해줘"},
        headers=ai_internal_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["cached"] is True
    assert body["semantic_cached"] is True
    assert body["planner_mode"] == "semantic_cache"
    assert body["grounding_mode"] == "semantic_cache"
    assert body["source_tier"] == "semantic_cache"
    assert body["cache_similarity"] == pytest.approx(0.948, rel=1e-6)
    assert body["model_usage"] == []
    assert body["model_usage_complete"] is True
    semantic_hit.assert_awaited_once()


def test_static_faq_result_has_complete_evidence_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_stream,
        "get_settings",
        lambda: SimpleNamespace(operator_data_fast_path_enabled=False),
    )
    monkeypatch.setattr(
        chat_stream,
        "_build_static_kbo_faq_result",
        lambda question: {"answer": question, "planner_mode": "predefined"},
    )

    result = asyncio.run(chat_stream._build_static_chat_result("FAQ"))

    assert result is not None
    assert result["model_usage"] == []
    assert result["model_usage_complete"] is True
    assert result["fallback_triggered"] is False
    assert result["fallback_answer_used"] is False


def test_static_operator_result_has_complete_evidence_metadata(monkeypatch) -> None:
    class ConnectionContext:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *args: object) -> None:
            return None

    class Pool:
        def connection(self) -> ConnectionContext:
            return ConnectionContext()

    monkeypatch.setattr(
        chat_stream,
        "get_settings",
        lambda: SimpleNamespace(operator_data_fast_path_enabled=True),
    )
    monkeypatch.setattr(chat_stream, "get_connection_pool", lambda: Pool())
    monkeypatch.setattr(
        chat_stream,
        "try_build_operator_fast_path_result",
        AsyncMock(
            return_value={"answer": "operator answer", "planner_mode": "fast_path"}
        ),
    )

    result = asyncio.run(chat_stream._build_static_chat_result("operator question"))

    assert result is not None
    assert result["model_usage"] == []
    assert result["model_usage_complete"] is True
    assert result["fallback_triggered"] is False
    assert result["fallback_answer_used"] is False


def test_semantic_cache_shadow_mode_does_not_serve_cached_completion(
    client: TestClient,
    ai_internal_headers: dict[str, str],
    monkeypatch,
):
    monkeypatch.setattr(
        "app.routers.chat_stream.get_settings",
        lambda: SimpleNamespace(
            operator_data_fast_path_enabled=False,
            chat_completion_timeout_seconds=0,
            chat_semantic_cache_enabled=True,
            chat_semantic_cache_shadow_enabled=True,
        ),
    )
    monkeypatch.setattr(
        "app.routers.chat_stream.get_cached_response",
        AsyncMock(return_value=None),
    )
    semantic_hit = AsyncMock(
        return_value={
            "cache_key": "semantic-cache-key",
            "question_text": "LG 팀 분위기 정리해줘",
            "filters_json": None,
            "response_text": "shadow에서만 관측할 캐시 응답",
            "intent": "team_analysis",
            "source_tier": "rag",
            "hit_count": 2,
            "similarity": 0.951,
        }
    )
    save_caches = AsyncMock(return_value=None)
    monkeypatch.setattr("app.routers.chat_stream._get_semantic_cache_hit", semantic_hit)
    monkeypatch.setattr(
        "app.routers.chat_stream._save_chat_response_caches", save_caches
    )

    response = client.post(
        "/ai/chat/completion",
        json={"question": "LG 팀 분위기 정리해줘"},
        headers=ai_internal_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["cached"] is False
    assert body.get("semantic_cached") is not True
    assert "모의 응답" in body["answer"]
    semantic_hit.assert_awaited_once()
    save_caches.assert_awaited_once()


def test_semantic_cache_shadow_mode_enables_lookup_but_disables_serving() -> None:
    settings = SimpleNamespace(
        chat_semantic_cache_enabled=False,
        chat_semantic_cache_shadow_enabled=True,
    )

    assert chat_stream._is_semantic_cache_lookup_enabled(settings) is True
    assert chat_stream._is_semantic_cache_serving_enabled(settings) is False


def test_semantic_cache_rollout_zero_disables_lookup_and_serving() -> None:
    settings = SimpleNamespace(
        chat_semantic_cache_enabled=True,
        chat_semantic_cache_shadow_enabled=False,
        chat_semantic_cache_rollout_percent=0,
        chat_semantic_cache_kill_switch=False,
    )

    assert chat_stream._is_semantic_cache_lookup_enabled(settings) is False
    assert chat_stream._is_semantic_cache_serving_enabled(settings) is False


def test_semantic_cache_kill_switch_overrides_full_rollout() -> None:
    settings = SimpleNamespace(
        chat_semantic_cache_enabled=True,
        chat_semantic_cache_shadow_enabled=False,
        chat_semantic_cache_rollout_percent=100,
        chat_semantic_cache_kill_switch=True,
    )

    assert chat_stream._is_semantic_cache_lookup_enabled(settings) is False
    assert chat_stream._is_semantic_cache_serving_enabled(settings) is False


def test_semantic_cache_rollout_selection_is_deterministic_and_bounded() -> None:
    cache_key = "stable-cache-key"

    assert chat_stream._semantic_cache_rollout_selected(cache_key, 0) is False
    assert chat_stream._semantic_cache_rollout_selected(cache_key, 100) is True
    assert chat_stream._semantic_cache_rollout_selected(
        cache_key, 5
    ) is chat_stream._semantic_cache_rollout_selected(cache_key, 5)


def test_semantic_cache_quality_gate_rejects_team_mismatch() -> None:
    ok, reason = chat_stream._semantic_cache_quality_gate(
        question="LG 팀 분위기 정리해줘",
        filters=None,
        cached={
            "question_text": "KIA 팀 분위기 정리해줘",
            "filters_json": None,
            "source_tier": "rag",
        },
    )

    assert ok is False
    assert reason == "teams_mismatch"


def test_semantic_cache_quality_gate_rejects_player_mismatch() -> None:
    ok, reason = chat_stream._semantic_cache_quality_gate(
        question="김도영의 2026년 기록 알려줘",
        filters=None,
        cached={
            "question_text": "문보경의 2026년 기록 알려줘",
            "filters_json": None,
            "source_tier": "db",
        },
    )

    assert ok is False
    assert reason == "players_mismatch"


def test_semantic_cache_quality_gate_rejects_question_type_mismatch() -> None:
    ok, reason = chat_stream._semantic_cache_quality_gate(
        question="FA 보상선수 규정 핵심이 뭐야?",
        filters=None,
        cached={
            "question_text": "플래툰 전략이 뭐야?",
            "filters_json": None,
            "source_tier": "rag",
        },
    )

    assert ok is False
    assert reason == "question_type_mismatch"


def test_semantic_cache_quality_gate_accepts_matching_dimensions() -> None:
    ok, reason = chat_stream._semantic_cache_quality_gate(
        question="LG 2025 정규시즌 흐름 정리해줘",
        filters={"team_id": "LG", "season_year": 2025, "league_type": "REGULAR"},
        cached={
            "question_text": "LG 트윈스 2025 정규시즌 분위기",
            "filters_json": {
                "team_id": "LG",
                "season_year": 2025,
                "league_type": "REGULAR",
            },
            "source_tier": "rag",
        },
    )

    assert ok is True
    assert reason is None


def test_semantic_cache_quality_gate_rejects_source_tier_mismatch() -> None:
    ok, reason = chat_stream._semantic_cache_quality_gate(
        question="LG 팀 분위기 정리해줘",
        filters={"source_tier": "operator_data"},
        cached={
            "question_text": "LG 팀 분위기 정리해줘",
            "filters_json": None,
            "source_tier": "rag",
        },
    )

    assert ok is False
    assert reason == "source_tier_mismatch"


def test_semantic_cache_quality_gate_rejects_route_cache_hit_on_mismatch(
    client: TestClient,
    ai_internal_headers: dict[str, str],
    monkeypatch,
):
    monkeypatch.setattr(
        "app.routers.chat_stream.get_settings",
        lambda: SimpleNamespace(
            operator_data_fast_path_enabled=False,
            chat_completion_timeout_seconds=0,
            chat_semantic_cache_enabled=True,
            chat_semantic_cache_shadow_enabled=False,
        ),
    )
    monkeypatch.setattr(
        "app.routers.chat_stream.get_cached_response",
        AsyncMock(return_value=None),
    )
    semantic_hit = AsyncMock(
        return_value={
            "cache_key": "semantic-cache-key",
            "question_text": "KIA 팀 분위기 정리해줘",
            "filters_json": None,
            "response_text": "잘못 재사용되면 안 되는 의미 캐시 응답",
            "intent": "team_analysis",
            "source_tier": "rag",
            "hit_count": 2,
            "similarity": 0.961,
        }
    )
    save_caches = AsyncMock(return_value=None)
    monkeypatch.setattr("app.routers.chat_stream._get_semantic_cache_hit", semantic_hit)
    monkeypatch.setattr(
        "app.routers.chat_stream._save_chat_response_caches", save_caches
    )

    response = client.post(
        "/ai/chat/completion",
        json={"question": "LG 팀 분위기 정리해줘"},
        headers=ai_internal_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["cached"] is False
    assert "모의 응답" in body["answer"]
    semantic_hit.assert_awaited_once()
    save_caches.assert_awaited_once()


def test_chat_endpoint_admission_dependencies_are_explicit() -> None:
    dependencies_by_route: dict[tuple[str, str], list[Any]] = {}
    for route in chat_stream.router.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods:
            dependencies_by_route[(method, route.path)] = [
                dependency.call for dependency in route.dependant.dependencies
            ]

    completion_deps = dependencies_by_route[("POST", "/ai/chat/completion")]
    post_stream_deps = dependencies_by_route[("POST", "/ai/chat/stream")]
    get_stream_deps = dependencies_by_route[("GET", "/ai/chat/stream")]

    assert chat_stream.rate_limit_chat_dependency in completion_deps
    assert chat_stream.rate_limit_chat_dependency in get_stream_deps
    assert chat_stream.rate_limit_chat_dependency not in post_stream_deps


def test_chat_cache_admin_stats_uses_async_connection_context(
    client: TestClient,
    monkeypatch,
) -> None:
    context = _AsyncOnlyConnectionContext()
    get_stats = AsyncMock(return_value={"total_entries": 3})

    monkeypatch.setattr(
        chat_stream,
        "get_settings",
        lambda: SimpleNamespace(
            chat_cache_admin_enabled=True,
            chat_cache_admin_token="admin-token",
        ),
    )
    monkeypatch.setattr(
        chat_stream, "get_connection_pool", lambda: _AsyncOnlyPool(context)
    )
    monkeypatch.setattr(chat_stream, "get_stats", get_stats)

    response = client.get(
        "/ai/chat/cache/stats",
        headers={"X-Cache-Admin-Token": "admin-token"},
    )

    assert response.status_code == 200
    assert response.json() == {"stats": {"total_entries": 3}}
    assert context.entered is True
    get_stats.assert_awaited_once_with(context.connection)


def test_chat_cache_admin_delete_key_uses_async_connection_context(
    client: TestClient,
    monkeypatch,
) -> None:
    context = _AsyncOnlyConnectionContext()
    delete_by_key = AsyncMock(return_value=1)

    monkeypatch.setattr(
        chat_stream,
        "get_settings",
        lambda: SimpleNamespace(
            chat_cache_admin_enabled=True,
            chat_cache_admin_token="admin-token",
        ),
    )
    monkeypatch.setattr(
        chat_stream, "get_connection_pool", lambda: _AsyncOnlyPool(context)
    )
    monkeypatch.setattr(chat_stream, "delete_by_key", delete_by_key)

    response = client.delete(
        "/ai/chat/cache/cache-key-1",
        headers={"X-Cache-Admin-Token": "admin-token"},
    )

    assert response.status_code == 200
    assert response.json() == {"deleted": 1, "cache_key": "cache-key-1"}
    assert context.entered is True
    delete_by_key.assert_awaited_once_with(context.connection, "cache-key-1")


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
async def test_chat_live_event_generator_finalizes_usage_after_answer_exhaustion():
    usage: list[_FakeUsage] = []

    async def stream():
        yield {
            "type": "metadata",
            "data": {"intent": "freeform", "verified": True, "model_usage": usage},
        }
        yield {"type": "answer_chunk", "content": "HELLO"}
        usage.append(_FakeUsage())

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

    meta = json.loads(next(event["data"] for event in events if event["event"] == "meta"))
    assert meta["model_usage"] == [_FakeUsage().to_dict()]
    assert meta["model_usage_complete"] is True


@pytest.mark.asyncio
async def test_chat_live_event_generator_marks_usage_incomplete_on_early_close():
    disconnects = iter([False, False, True, False])

    class _Request:
        async def is_disconnected(self) -> bool:
            return next(disconnects, False)

    async def stream():
        yield {"type": "answer_chunk", "content": "first"}
        yield {"type": "answer_chunk", "content": " second"}

    events = [
        event
        async for event in chat_stream._chat_live_event_generator(
            request=_Request(),
            question="test",
            filters=None,
            style="markdown",
            cache_key=None,
            stream=stream(),
        )
    ]

    meta = json.loads(next(event["data"] for event in events if event["event"] == "meta"))
    assert meta["model_usage_complete"] is False
    assert meta["finish_reason"] == "cancelled"


@pytest.mark.asyncio
async def test_chat_live_event_generator_marks_terminal_answer_failure_incomplete(
    monkeypatch,
):
    async def stream():
        yield {
            "type": "metadata",
            "data": {"intent": "freeform", "verified": True, "model_usage": [_FakeUsage()]},
        }
        yield {"type": "answer_chunk", "content": "partial"}
        raise RuntimeError("terminal answer failure")

    save_caches = AsyncMock(return_value=None)
    token_estimate = MagicMock()
    monkeypatch.setattr(chat_stream, "_save_chat_response_caches", save_caches)
    monkeypatch.setattr(chat_stream, "record_chat_token_estimate", token_estimate)

    events = [
        event
        async for event in chat_stream._chat_live_event_generator(
            request=None,
            question="test",
            filters=None,
            style="markdown",
            cache_key="cache-key",
            stream=stream(),
        )
    ]

    message_events = [event for event in events if event["event"] == "message"]
    error_events = [event for event in events if event["event"] == "error"]
    meta = json.loads(next(event["data"] for event in events if event["event"] == "meta"))
    assert len(message_events) == 2
    assert json.loads(message_events[0]["data"]) == {"delta": "partial"}
    assert json.loads(message_events[1]["data"])["delta"].startswith("지금 답변이")
    assert len(error_events) == 1
    assert meta["model_usage_complete"] is False
    assert meta["finish_reason"] == "error"
    assert meta["error"] == "temporary_generation_issue"
    save_caches.assert_not_awaited()
    token_estimate.assert_not_called()


@pytest.mark.asyncio
async def test_chat_event_generator_exposes_usage_after_normal_completion():
    async def answer_stream():
        yield "first"
        yield " second"

    events = [
        event
        async for event in chat_stream._chat_event_generator(
            request=None,
            question="test",
            filters=None,
            style="markdown",
            result={"answer": answer_stream(), "model_usage": [_FakeUsage()]},
            error_payload=None,
            cache_key=None,
        )
    ]

    meta = json.loads(next(event["data"] for event in events if event["event"] == "meta"))
    assert meta["model_usage"] == [_FakeUsage().to_dict()]
    assert meta["model_usage_complete"] is True


@pytest.mark.asyncio
async def test_chat_event_generator_marks_usage_incomplete_on_early_close():
    disconnects = iter([False, False, True, False])

    class _Request:
        async def is_disconnected(self) -> bool:
            return next(disconnects, False)

    async def answer_stream():
        yield "first"
        yield " second"

    events = [
        event
        async for event in chat_stream._chat_event_generator(
            request=_Request(),
            question="test",
            filters=None,
            style="markdown",
            result={"answer": answer_stream(), "model_usage": [_FakeUsage()]},
            error_payload=None,
            cache_key=None,
        )
    ]

    meta = json.loads(next(event["data"] for event in events if event["event"] == "meta"))
    assert meta["model_usage_complete"] is False


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
