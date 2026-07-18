import httpx
import logging
import os
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.observability.metrics import AI_LLM_RETRY_ATTEMPTS_TOTAL

logger = logging.getLogger(__name__)


def is_retryable_llm_error(exc: Exception) -> bool:
    """Return True for transient LLM failures that are worth retrying."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500

    return isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            httpx.TransportError,
            httpx.ReadError,
        ),
    )


def _classify_llm_error(exc: BaseException) -> str:
    """Coarse error class label for low-cardinality Prometheus metric."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return "429"
        if 500 <= status < 600:
            return f"{status // 100}xx"
        return str(status)
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, (httpx.NetworkError, httpx.TransportError)):
        return "network"
    if isinstance(exc, httpx.RemoteProtocolError):
        return "protocol"
    return type(exc).__name__


def _provider_label() -> str:
    return (os.getenv("LLM_PROVIDER") or "unknown").strip().lower()


def _record_retry(retry_state: Any) -> None:
    """tenacity ``before_sleep`` 콜백 — retry 횟수 메트릭 증가."""
    outcome = getattr(retry_state, "outcome", None)
    if outcome is None or not outcome.failed:
        return
    try:
        exc = outcome.exception()
    except Exception:  # noqa: BLE001
        exc = None
    error_class = _classify_llm_error(exc) if exc is not None else "unknown"
    try:
        AI_LLM_RETRY_ATTEMPTS_TOTAL.labels(
            provider=_provider_label(), error_class=error_class
        ).inc()
    except Exception:  # noqa: BLE001
        # 메트릭 실패가 retry 흐름을 막지 않도록
        pass


def _before_sleep_combined(retry_state: Any) -> None:
    _record_retry(retry_state)
    before_sleep_log(logger, logging.WARNING)(retry_state)


llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(is_retryable_llm_error),
    before_sleep=_before_sleep_combined,
    reraise=True,
)
