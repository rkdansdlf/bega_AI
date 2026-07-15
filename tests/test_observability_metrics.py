"""``app.observability.metrics`` 회귀 테스트.

- 메트릭 객체가 정상적으로 라벨/관측을 받음
- /metrics 엔드포인트가 텍스트 응답을 반환
- llm_retry 데코레이터가 retry 시 카운터 증가
- 임베딩 캐시 백엔드 hit/miss가 카운터에 반영
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

pytest.importorskip("prometheus_client")

from prometheus_client import REGISTRY


def _read_metric_value(name: str, labels: dict) -> float:
    """REGISTRY에서 특정 라벨 조합의 현재 값을 읽음."""
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == name and sample.labels == labels:
                return sample.value
    return 0.0


# ---------------------------------------------------------------------------
# 메트릭 모듈 기본 동작
# ---------------------------------------------------------------------------


def test_metric_objects_accept_labels_and_observations() -> None:
    from app.observability.metrics import (
        AI_CHAT_COST_ESTIMATE_USD_TOTAL,
        AI_CHAT_TOKEN_ESTIMATE_TOTAL,
        AI_DB_POOL_SIZE,
        AI_EMBEDDING_CACHE_TOTAL,
        AI_LLM_CALL_DURATION_SECONDS,
        AI_LLM_RETRY_ATTEMPTS_TOTAL,
        AI_MODEL_USAGE_COST_ESTIMATE_USD_TOTAL,
        AI_MODEL_USAGE_OUTCOME_TOTAL,
        AI_MODEL_USAGE_TOKEN_ESTIMATE_TOTAL,
        AI_RAG_STAGE_DURATION_SECONDS,
        AI_SEMANTIC_RESPONSE_CACHE_SHADOW_TOTAL,
        AI_SEMANTIC_RESPONSE_CACHE_TOTAL,
    )

    AI_LLM_RETRY_ATTEMPTS_TOTAL.labels(provider="test", error_class="429").inc()
    AI_EMBEDDING_CACHE_TOTAL.labels(backend="memory", result="hit").inc()
    AI_SEMANTIC_RESPONSE_CACHE_TOTAL.labels(operation="lookup", result="hit").inc()
    AI_SEMANTIC_RESPONSE_CACHE_SHADOW_TOTAL.labels(
        route="completion", result="hit"
    ).inc()
    AI_CHAT_TOKEN_ESTIMATE_TOTAL.labels(
        route="completion", token_type="input", cache_state="generated"
    ).inc(10)
    AI_CHAT_COST_ESTIMATE_USD_TOTAL.labels(
        route="completion", provider="test", model="unit-test"
    ).inc(0.001)
    AI_MODEL_USAGE_TOKEN_ESTIMATE_TOTAL.labels(
        role="planner",
        provider="openrouter",
        model="vendor/planner",
        token_type="input",
        outcome="success",
    ).inc(10)
    AI_MODEL_USAGE_COST_ESTIMATE_USD_TOTAL.labels(
        role="planner", provider="openrouter", model="vendor/planner"
    ).inc(0.001)
    AI_MODEL_USAGE_OUTCOME_TOTAL.labels(
        role="planner",
        provider="openrouter",
        model="vendor/planner",
        result="priced",
    ).inc()
    AI_RAG_STAGE_DURATION_SECONDS.labels(stage="embed").observe(0.05)
    AI_LLM_CALL_DURATION_SECONDS.labels(provider="test", route="rag").observe(1.2)
    AI_DB_POOL_SIZE.labels(state="available").set(7)

    assert _read_metric_value(
        "ai_llm_retry_attempts_total", {"provider": "test", "error_class": "429"}
    ) >= 1.0


def test_ingest_metrics_use_only_bounded_labels() -> None:
    from app.observability.metrics import (
        AI_INGEST_ACTIVE_RUNS,
        AI_INGEST_LEASE_RECOVERIES_TOTAL,
        AI_INGEST_QUEUED_RUNS,
        AI_INGEST_RUN_COMPLETIONS_TOTAL,
        AI_INGEST_RUN_DURATION_SECONDS,
        AI_INGEST_SUBMISSIONS_TOTAL,
        AI_INGEST_TABLE_DURATION_SECONDS,
        AI_INGEST_TABLE_SOURCE_ROWS_TOTAL,
        AI_INGEST_TABLE_WRITTEN_CHUNKS_TOTAL,
        AI_INGEST_WATERMARK_LAG_SECONDS,
    )

    assert AI_INGEST_SUBMISSIONS_TOTAL._labelnames == (
        "trigger_source",
        "result",
    )
    assert AI_INGEST_RUN_COMPLETIONS_TOTAL._labelnames == (
        "status",
        "trigger_source",
    )
    assert AI_INGEST_RUN_DURATION_SECONDS._labelnames == (
        "status",
        "trigger_source",
    )
    assert AI_INGEST_ACTIVE_RUNS._labelnames == ("trigger_source",)
    assert AI_INGEST_QUEUED_RUNS._labelnames == ("trigger_source",)
    assert AI_INGEST_LEASE_RECOVERIES_TOTAL._labelnames == ("result",)
    assert AI_INGEST_TABLE_DURATION_SECONDS._labelnames == ("source_table",)
    assert AI_INGEST_TABLE_SOURCE_ROWS_TOTAL._labelnames == ("source_table",)
    assert AI_INGEST_TABLE_WRITTEN_CHUNKS_TOTAL._labelnames == ("source_table",)
    assert AI_INGEST_WATERMARK_LAG_SECONDS._labelnames == ("source_table",)


def test_rag_total_decorator_observes_total_stage() -> None:
    """_observe_rag_total 데코레이터가 코루틴 실행 후 total 스테이지를 관측한다."""
    from app.core.rag import _observe_rag_total

    before = _read_metric_value(
        "ai_rag_stage_duration_seconds_count", {"stage": "total"}
    )

    @_observe_rag_total
    async def _fake_run():
        return "ok"

    result = asyncio.run(_fake_run())
    assert result == "ok"

    after = _read_metric_value(
        "ai_rag_stage_duration_seconds_count", {"stage": "total"}
    )
    assert after == before + 1.0


def test_rag_total_stream_decorator_observes_total_stage() -> None:
    """_observe_rag_total_stream 데코레이터가 제너레이터 소진 후 total 스테이지를 관측한다."""
    from app.core.rag import _observe_rag_total_stream

    before = _read_metric_value(
        "ai_rag_stage_duration_seconds_count", {"stage": "total"}
    )

    @_observe_rag_total_stream
    async def _fake_stream():
        yield {"type": "answer_chunk", "content": "a"}
        yield {"type": "answer_chunk", "content": "b"}

    async def _drain():
        return [item async for item in _fake_stream()]

    items = asyncio.run(_drain())
    assert len(items) == 2

    after = _read_metric_value(
        "ai_rag_stage_duration_seconds_count", {"stage": "total"}
    )
    assert after == before + 1.0


def test_metrics_asgi_app_returns_prometheus_format() -> None:
    from app.observability.metrics import metrics_asgi_app

    app = metrics_asgi_app()
    transport = httpx.ASGITransport(app=app)

    async def _fetch() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get("/")

    resp = asyncio.run(_fetch())
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")
    body = resp.text
    # 우리가 정의한 메트릭 이름 중 하나라도 포함되어야 함
    assert any(
        name in body
        for name in (
            "ai_llm_retry_attempts_total",
            "ai_embedding_cache_total",
            "ai_rag_stage_duration_seconds",
        )
    )


# ---------------------------------------------------------------------------
# llm_retry 데코레이터: retry 시 카운터 증가
# ---------------------------------------------------------------------------


def test_llm_retry_increments_metric_on_retryable_error(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "test_provider")
    # 최소 wait로 빠른 테스트
    from app.core import retry_utils
    from tenacity import (
        retry as tenacity_retry,
        retry_if_exception,
        stop_after_attempt,
        wait_fixed,
    )

    fast_retry = tenacity_retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(0),
        retry=retry_if_exception(retry_utils.is_retryable_llm_error),
        before_sleep=retry_utils._before_sleep_combined,
        reraise=True,
    )

    call_count = {"n": 0}

    @fast_retry
    def flaky() -> str:
        call_count["n"] += 1
        # httpx.HTTPStatusError(429) 시뮬레이션
        request = httpx.Request("GET", "http://test")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limit", request=request, response=response)

    before = _read_metric_value(
        "ai_llm_retry_attempts_total",
        {"provider": "test_provider", "error_class": "429"},
    )
    with pytest.raises(httpx.HTTPStatusError):
        flaky()
    after = _read_metric_value(
        "ai_llm_retry_attempts_total",
        {"provider": "test_provider", "error_class": "429"},
    )
    # stop_after_attempt(3) → before_sleep는 2번 호출
    assert after - before >= 2.0
    assert call_count["n"] == 3


def test_classify_llm_error_handles_common_cases() -> None:
    from app.core.retry_utils import _classify_llm_error

    request = httpx.Request("GET", "http://test")
    assert (
        _classify_llm_error(
            httpx.HTTPStatusError(
                "x", request=request, response=httpx.Response(429, request=request)
            )
        )
        == "429"
    )
    assert (
        _classify_llm_error(
            httpx.HTTPStatusError(
                "x", request=request, response=httpx.Response(503, request=request)
            )
        )
        == "5xx"
    )
    assert _classify_llm_error(httpx.TimeoutException("t")) == "timeout"
    assert _classify_llm_error(httpx.ConnectError("c")) == "network"


# ---------------------------------------------------------------------------
# 임베딩 캐시 hit/miss 메트릭
# ---------------------------------------------------------------------------


def test_in_memory_backend_records_hit_and_miss() -> None:
    from app.core.embedding_cache import InMemoryLRUBackend

    backend = InMemoryLRUBackend(max_size=4)
    miss_before = _read_metric_value(
        "ai_embedding_cache_total", {"backend": "memory", "result": "miss"}
    )
    hit_before = _read_metric_value(
        "ai_embedding_cache_total", {"backend": "memory", "result": "hit"}
    )
    asyncio.run(backend.get("missing-key"))
    asyncio.run(backend.set("k1", [0.1, 0.2]))
    asyncio.run(backend.get("k1"))

    miss_after = _read_metric_value(
        "ai_embedding_cache_total", {"backend": "memory", "result": "miss"}
    )
    hit_after = _read_metric_value(
        "ai_embedding_cache_total", {"backend": "memory", "result": "hit"}
    )
    assert miss_after - miss_before >= 1.0
    assert hit_after - hit_before >= 1.0
