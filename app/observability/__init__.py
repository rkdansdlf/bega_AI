"""계측 인프라.

Prometheus 메트릭 정의 및 헬퍼 함수를 제공한다. ``app.observability.metrics``
모듈은 모듈 import 시점에 메트릭을 한 번만 등록하므로 호출자에서 import만
하면 된다.

호출 패턴:
    from app.observability.metrics import LLM_RETRY_ATTEMPTS_TOTAL
    LLM_RETRY_ATTEMPTS_TOTAL.labels(provider="gemini", error_class="429").inc()
"""

from app.observability.metrics import (
    AI_COACH_DYNAMIC_PROMPT_CHARS,
    AI_COACH_PAYLOAD_COMPRESSION_TOTAL,
    AI_COACH_REQUEST_TOTAL,
    AI_DB_POOL_SIZE,
    AI_EMBEDDING_CACHE_TOTAL,
    AI_LLM_CALL_DURATION_SECONDS,
    AI_LLM_RETRY_ATTEMPTS_TOTAL,
    AI_RAG_STAGE_DURATION_SECONDS,
    AI_RESPONSE_CACHE_TOTAL,
    metrics_asgi_app,
)

__all__ = [
    "AI_COACH_DYNAMIC_PROMPT_CHARS",
    "AI_COACH_PAYLOAD_COMPRESSION_TOTAL",
    "AI_COACH_REQUEST_TOTAL",
    "AI_DB_POOL_SIZE",
    "AI_EMBEDDING_CACHE_TOTAL",
    "AI_LLM_CALL_DURATION_SECONDS",
    "AI_LLM_RETRY_ATTEMPTS_TOTAL",
    "AI_RAG_STAGE_DURATION_SECONDS",
    "AI_RESPONSE_CACHE_TOTAL",
    "metrics_asgi_app",
]
