"""RAG 파이프라인 통합 테스트.

실제 RAGPipeline을 사용하되 DB 레이어(similarity_search)와 LLM(_generate)만 모킹하여
HTTP 엔드포인트 레벨에서 에러 처리와 응답 스키마를 검증합니다.
"""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions import DBRetrievalError
from app.core.rag import RAGPipeline
from app.routers import chat_stream

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

_FAKE_EMBEDDING = [0.1] * 768
_AI_TOKEN = "test-internal-token"
_AI_HEADERS = {"X-Internal-Api-Key": _AI_TOKEN}
_CHAT_PAYLOAD = {"question": "2024년 홈런왕은 누구야?"}


def _collect_sse_events(response) -> list[dict]:
    """TestClient stream 응답에서 SSE 이벤트 목록을 파싱합니다."""
    events: list[dict] = []
    current: dict = {}
    for raw_line in response.iter_lines():
        line = (
            raw_line
            if isinstance(raw_line, str)
            else raw_line.decode("utf-8", errors="replace")
        )
        if line.startswith("event:"):
            if current:
                events.append(current)
                current = {}
            current["type"] = line[6:].strip()
        elif line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                if current:
                    events.append(current)
                break
            try:
                current["data"] = json.loads(data_str)
            except json.JSONDecodeError:
                current["data"] = data_str
        elif not line and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


def _find_meta_event(events: list[dict]) -> dict | None:
    """이벤트 목록에서 meta 이벤트를 찾아 반환합니다."""
    for ev in events:
        if ev.get("type") == "meta":
            return ev.get("data", {})
    return None


class _IntegrationFakeAgent:
    """실제 RAGPipeline을 내부적으로 사용하되 외부 의존성만 교체한 통합 에이전트."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    async def process_query(self, question: str, context: dict[str, Any] | None = None):
        result = await self.pipeline.run(question, intent="stats_lookup")
        answer = result.get("answer", "")
        # answer가 비동기 제너레이터인 경우 문자열로 소비
        if hasattr(answer, "__aiter__"):
            chunks = []
            async for chunk in answer:
                chunks.append(str(chunk))
            answer = "".join(chunks)
        return {
            "answer": answer,
            "strategy": result.get("strategy"),
            "citations": result.get("citations", []),
            "tool_calls": [],
            "tool_results": [],
            "data_sources": [],
            "verified": False,
            "visualizations": [],
            "intent": result.get("intent", "stats_lookup"),
            "error": None,
        }


def _make_pipeline() -> RAGPipeline:
    """테스트용 RAGPipeline 인스턴스를 생성합니다 (실제 DB 연결 없음)."""
    from app.config import get_settings

    settings = get_settings()
    mock_conn = MagicMock(spec=psycopg.Connection)
    return RAGPipeline(settings=settings, connection=mock_conn)


@pytest.fixture
def integration_client(monkeypatch):
    """통합 테스트용 FastAPI TestClient 픽스처."""
    test_app = FastAPI()
    test_app.include_router(chat_stream.router)

    pipeline = _make_pipeline()
    agent = _IntegrationFakeAgent(pipeline)

    test_app.dependency_overrides[chat_stream.get_agent] = lambda: agent
    test_app.dependency_overrides[chat_stream.rate_limit_chat_dependency] = lambda: None

    monkeypatch.setattr(
        "app.deps.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token=_AI_TOKEN),
    )
    monkeypatch.setattr("app.deps.record_security_event", lambda *args, **kwargs: None)

    # completion 엔드포인트가 캐시 확인을 위해 get_connection_pool()을 호출하므로
    # 실제 DB 연결 없이 캐시 미스를 반환하도록 패치
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
        AsyncMock(return_value=None),  # 항상 캐시 미스
    )
    monkeypatch.setattr(
        "app.routers.chat_stream.save_to_cache",
        AsyncMock(return_value=None),  # 캐시 저장 no-op
    )
    monkeypatch.setattr(
        "app.routers.chat_stream._request_is_disconnected",
        AsyncMock(
            return_value=False
        ),  # disconnect 변수를 고정해 SSE 메타 검증을 안정화
    )

    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c, agent, pipeline


# ---------------------------------------------------------------------------
# Group 1: DB OperationalError → HTTP 200 보장
# ---------------------------------------------------------------------------


def test_db_down_returns_http_200(integration_client):
    """DB OperationalError가 발생해도 HTTP 500이 아닌 200이어야 한다."""
    client, agent, pipeline = integration_client

    with patch(
        "app.core.rag.async_embed_query",
        new_callable=AsyncMock,
        return_value=_FAKE_EMBEDDING,
    ):
        with patch(
            "app.core.rag.similarity_search",
            side_effect=DBRetrievalError(
                "mock db error", cause=psycopg.OperationalError("conn failed")
            ),
        ):
            with patch.object(
                pipeline,
                "_generate",
                new_callable=AsyncMock,
                return_value="⚠️ 현재 KBO 통계 DB에 일시적으로 접속할 수 없어 정확한 수치를 확인하지 못했습니다.",
            ):
                response = client.post(
                    "/ai/chat/completion",
                    json=_CHAT_PAYLOAD,
                    headers=_AI_HEADERS,
                )

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Group 2: DB-down → SSE meta.strategy = "llm_knowledge_db_unavailable"
# ---------------------------------------------------------------------------


def test_db_down_strategy_in_sse_meta(integration_client):
    """DB-down 경로일 때 SSE meta 이벤트에 strategy가 포함되어야 한다."""
    client, agent, pipeline = integration_client

    async def collect_meta_events() -> list[dict[str, Any]]:
        result = await agent.process_query(_CHAT_PAYLOAD["question"])
        events: list[dict[str, Any]] = []
        async for event in chat_stream._chat_event_generator(
            request=None,
            question=_CHAT_PAYLOAD["question"],
            filters=None,
            style="markdown",
            result=result,
            error_payload=None,
            cache_key=None,
        ):
            payload = event["data"]
            events.append(
                {
                    "type": event["event"],
                    "data": (json.loads(payload) if payload != "[DONE]" else payload),
                }
            )
        return events

    with patch(
        "app.core.rag.async_embed_query",
        new_callable=AsyncMock,
        return_value=_FAKE_EMBEDDING,
    ):
        with patch(
            "app.core.rag.similarity_search",
            side_effect=DBRetrievalError(
                "mock db error", cause=psycopg.OperationalError("conn failed")
            ),
        ):
            with patch.object(
                pipeline,
                "_generate",
                new_callable=AsyncMock,
                return_value="⚠️ DB 접속 불가 답변",
            ):
                events = asyncio.run(collect_meta_events())

    meta = _find_meta_event(events)
    assert meta is not None, "meta 이벤트가 없습니다"
    assert (
        meta.get("strategy") == "llm_knowledge_db_unavailable"
    ), f"strategy가 'llm_knowledge_db_unavailable'이어야 하지만 실제: {meta.get('strategy')}"


# ---------------------------------------------------------------------------
# Group 3: DB-down → citations 빈 배열
# ---------------------------------------------------------------------------


def test_db_down_citations_empty(integration_client):
    """DB-down 경로일 때 data_sources(citations)는 빈 배열이어야 한다."""
    client, agent, pipeline = integration_client

    with patch(
        "app.core.rag.async_embed_query",
        new_callable=AsyncMock,
        return_value=_FAKE_EMBEDDING,
    ):
        with patch(
            "app.core.rag.similarity_search",
            side_effect=DBRetrievalError(
                "mock db error", cause=psycopg.OperationalError("conn failed")
            ),
        ):
            with patch.object(
                pipeline, "_generate", new_callable=AsyncMock, return_value="⚠️ 답변"
            ):
                response = client.post(
                    "/ai/chat/completion",
                    json=_CHAT_PAYLOAD,
                    headers=_AI_HEADERS,
                )

    assert response.status_code == 200
    body = response.json()
    # DB-down 경로는 data_sources(citations)가 없어야 함
    data_sources = body.get("data_sources", [])
    assert (
        data_sources == []
    ), f"DB-down 경로에서 data_sources는 []이어야 하지만 실제: {data_sources}"


# ---------------------------------------------------------------------------
# Group 4: DB-down → answer에 면책 문구 포함
# ---------------------------------------------------------------------------


def test_db_down_answer_contains_disclaimer(integration_client):
    """DB-down 경로일 때 LLM에 전달된 컨텍스트가 면책 문구를 지시해야 한다.

    _generate에 전달된 messages에 면책 지시가 포함되어 있는지 확인합니다.
    """
    client, agent, pipeline = integration_client

    captured = {}

    async def mock_generate(messages):
        captured["messages"] = messages
        return "⚠️ 현재 KBO 통계 DB에 일시적으로 접속할 수 없어 정확한 수치를 확인하지 못했습니다. 일반 지식 기반 답변입니다."

    with patch(
        "app.core.rag.async_embed_query",
        new_callable=AsyncMock,
        return_value=_FAKE_EMBEDDING,
    ):
        with patch(
            "app.core.rag.similarity_search",
            side_effect=DBRetrievalError(
                "mock db error", cause=psycopg.OperationalError("conn failed")
            ),
        ):
            with patch.object(pipeline, "_generate", side_effect=mock_generate):
                response = client.post(
                    "/ai/chat/completion",
                    json=_CHAT_PAYLOAD,
                    headers=_AI_HEADERS,
                )

    assert response.status_code == 200

    # LLM에 전달된 컨텍스트에 면책 지시가 포함되어야 함
    assert captured.get("messages"), "LLM에 전달된 messages가 없습니다"
    all_content = " ".join(
        m.get("content", "")
        for m in captured["messages"]
        if isinstance(m.get("content"), str)
    )
    assert (
        "접속" in all_content or "데이터베이스" in all_content or "DB" in all_content
    ), "LLM 컨텍스트에 DB 장애 안내가 없습니다"
    assert (
        "단정" in all_content or "수치" in all_content or "정확한" in all_content
    ), "LLM 컨텍스트에 수치 단정 금지 지시가 없습니다"


# ---------------------------------------------------------------------------
# Group 5: Zero-hit → answer에 가이드 텍스트 포함
# ---------------------------------------------------------------------------


def test_zero_hit_answer_contains_guidance(integration_client):
    """모든 검색이 [] 반환 시 LLM 컨텍스트에 가이드 텍스트가 포함되어야 한다."""
    client, agent, pipeline = integration_client

    captured = {}

    async def mock_generate(messages):
        captured["messages"] = messages
        return "해당 데이터를 찾을 수 없습니다. 다른 연도나 선수명을 시도해보세요."

    with patch(
        "app.core.rag.async_embed_query",
        new_callable=AsyncMock,
        return_value=_FAKE_EMBEDDING,
    ):
        with patch("app.core.rag.similarity_search", return_value=[]):
            with patch.object(pipeline, "_generate", side_effect=mock_generate):
                response = client.post(
                    "/ai/chat/completion",
                    json={"question": "김철수아무개99 2024년 타율은?"},
                    headers=_AI_HEADERS,
                )

    assert response.status_code == 200

    # LLM에 전달된 컨텍스트에 가이드 텍스트가 있어야 함
    all_content = " ".join(
        m.get("content", "")
        for m in captured.get("messages", [])
        if isinstance(m.get("content"), str)
    )
    has_guidance = (
        "가능한 원인" in all_content
        or "대안" in all_content
        or "검색 결과 없음" in all_content
        or "가이드" in all_content
    )
    assert (
        has_guidance
    ), f"Zero-hit LLM 컨텍스트에 가이드 텍스트가 없습니다. 컨텍스트 미리보기: {all_content[:300]}"


# ---------------------------------------------------------------------------
# Group 6: Multi-query 일부 실패 → HTTP 200 + 답변 반환
# ---------------------------------------------------------------------------


def test_partial_multi_query_failure_returns_200(integration_client):
    """similarity_search 호출 중 일부만 실패해도 HTTP 200으로 답변이 반환되어야 한다."""
    client, agent, pipeline = integration_client

    call_count = {"n": 0}

    def alternating_search(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] % 2 == 1:
            # 홀수 번째 호출: DB 에러
            raise DBRetrievalError(
                "partial failure", cause=psycopg.OperationalError("conn failed")
            )
        # 짝수 번째 호출: 성공 (빈 결과)
        return []

    with patch(
        "app.core.rag.async_embed_query",
        new_callable=AsyncMock,
        return_value=_FAKE_EMBEDDING,
    ):
        with patch("app.core.rag.similarity_search", side_effect=alternating_search):
            with patch.object(
                pipeline,
                "_generate",
                new_callable=AsyncMock,
                return_value="부분적 데이터로 답변드립니다.",
            ):
                response = client.post(
                    "/ai/chat/completion",
                    json=_CHAT_PAYLOAD,
                    headers=_AI_HEADERS,
                )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.get("answer"), str)
    assert len(body["answer"]) > 0
