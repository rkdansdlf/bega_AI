"""KBO AI 서비스 Prometheus 메트릭 카탈로그.

모든 메트릭은 ``ai_`` 접두사로 시작한다. 라벨 카디널리티는 의도적으로 낮게
유지(provider/result/stage/cache_state 등 닫힌 집합)하여 메트릭 폭발을
방지한다.

메트릭이 처음 import될 때 기본 ``REGISTRY``에 등록된다. 테스트에서는
``prometheus_client.REGISTRY``를 직접 사용하거나 본 모듈에 정의된 메트릭
객체를 직접 검사한다.

prometheus_client가 미설치된 환경(테스트 격리, 빌드 단계 등)에서도 import
실패하지 않도록 ``no-op`` fallback을 제공한다.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        make_asgi_app,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover — runtime fallback
    _PROMETHEUS_AVAILABLE = False
    logger.warning("[Observability] prometheus_client not installed — metrics disabled")

    class _NoOpMetric:
        def labels(self, **_kwargs: Any) -> "_NoOpMetric":
            return self

        def inc(self, _amount: float = 1.0) -> None:
            return None

        def observe(self, _value: float) -> None:
            return None

        def set(self, _value: float) -> None:
            return None

        def time(self) -> "_NoOpMetric":
            return self

        def __enter__(self) -> "_NoOpMetric":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    def _make_noop(*_args: Any, **_kwargs: Any) -> _NoOpMetric:
        return _NoOpMetric()

    Counter = Histogram = Gauge = _make_noop  # type: ignore[assignment]
    CollectorRegistry = None  # type: ignore[assignment]

    def make_asgi_app(*_args: Any, **_kwargs: Any) -> Any:
        async def _empty_app(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 503, "headers": []})
            await send(
                {
                    "type": "http.response.body",
                    "body": b"prometheus_client not installed",
                }
            )

        return _empty_app


# ---------------------------------------------------------------------------
# Counters — 발생 횟수
# ---------------------------------------------------------------------------

AI_LLM_RETRY_ATTEMPTS_TOTAL = Counter(
    "ai_llm_retry_attempts_total",
    "Number of LLM retry attempts (after initial failure).",
    ["provider", "error_class"],
)

AI_EMBEDDING_CACHE_TOTAL = Counter(
    "ai_embedding_cache_total",
    "Embedding query cache lookups.",
    ["backend", "result"],  # backend: memory|redis ; result: hit|miss
)

AI_RESPONSE_CACHE_TOTAL = Counter(
    "ai_response_cache_total",
    "Chat response cache lookups and writes.",
    ["operation", "result"],  # operation: lookup|store ; result: hit|miss|skip|ok|error
)

AI_COACH_REQUEST_TOTAL = Counter(
    "ai_coach_request_total",
    "Coach request outcomes by cache state and request mode.",
    ["cache_state", "mode"],
)

AI_COACH_PAYLOAD_COMPRESSION_TOTAL = Counter(
    "ai_coach_payload_compression_total",
    "Coach payload compression invocations (B-PR effect tracking).",
    ["enabled"],  # on|off
)

AI_RETRIEVAL_FALLBACK_LEVEL_TOTAL = Counter(
    "ai_retrieval_fallback_level_total",
    "Number of times each fallback level was reached in similarity_search_with_fallback.",
    ["level"],  # level_1|level_2|level_3|level_4|exhausted
)

AI_RESPONSE_CACHE_BY_INTENT = Counter(
    "ai_response_cache_by_intent_total",
    "Chat response cache hits and misses segmented by query intent.",
    ["intent", "result"],  # intent: stats_lookup|player_profile|… ; result: hit|miss
)

# ---------------------------------------------------------------------------
# Histograms — 분포 (latency, size)
# ---------------------------------------------------------------------------

# RAG 파이프라인 단계별 latency (초). 버킷은 일반적인 RAG 분포에 맞춤.
_RAG_STAGE_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
AI_RAG_STAGE_DURATION_SECONDS = Histogram(
    "ai_rag_stage_duration_seconds",
    "Duration of individual RAG pipeline stages.",
    ["stage"],  # entity_extract|embed|search|format|llm|total
    buckets=_RAG_STAGE_BUCKETS,
)

# LLM 호출 latency. 외부 호출이라 더 큰 버킷.
_LLM_CALL_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
AI_LLM_CALL_DURATION_SECONDS = Histogram(
    "ai_llm_call_duration_seconds",
    "End-to-end LLM call duration including retries.",
    ["provider", "route"],  # route: rag|coach|hyde|query_transform
    buckets=_LLM_CALL_BUCKETS,
)

# Coach 동적 프롬프트 사이즈 (B 작업 효과 추적). 문자 수 단위.
_COACH_PROMPT_BUCKETS = (
    500,
    1000,
    2000,
    3000,
    5000,
    7500,
    10000,
    15000,
    20000,
    30000,
)
AI_COACH_DYNAMIC_PROMPT_CHARS = Histogram(
    "ai_coach_dynamic_prompt_chars",
    "Coach dynamic prompt size in characters (compressed payload effect).",
    ["compress"],  # on|off
    buckets=_COACH_PROMPT_BUCKETS,
)

# ---------------------------------------------------------------------------
# Gauges — 현재 상태
# ---------------------------------------------------------------------------

AI_DB_POOL_SIZE = Gauge(
    "ai_db_pool_size",
    "Current PostgreSQL connection pool state.",
    ["state"],  # max|min|available|requests_waiting
)


# ---------------------------------------------------------------------------
# ASGI 통합
# ---------------------------------------------------------------------------


def metrics_asgi_app() -> Any:
    """``/metrics`` 엔드포인트로 마운트할 ASGI 앱을 반환한다.

    ``app.main``의 lifespan 초기 단계에서 ``app.mount("/metrics", metrics_asgi_app())``
    형태로 마운트한다.
    """
    return make_asgi_app()


__all__ = [
    "AI_COACH_DYNAMIC_PROMPT_CHARS",
    "AI_COACH_PAYLOAD_COMPRESSION_TOTAL",
    "AI_COACH_REQUEST_TOTAL",
    "AI_DB_POOL_SIZE",
    "AI_EMBEDDING_CACHE_TOTAL",
    "AI_LLM_CALL_DURATION_SECONDS",
    "AI_LLM_RETRY_ATTEMPTS_TOTAL",
    "AI_RAG_STAGE_DURATION_SECONDS",
    "AI_RESPONSE_CACHE_BY_INTENT",
    "AI_RESPONSE_CACHE_TOTAL",
    "AI_RETRIEVAL_FALLBACK_LEVEL_TOTAL",
    "metrics_asgi_app",
]
