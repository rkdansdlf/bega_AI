from __future__ import annotations

import httpx

from app.core.retry_utils import is_retryable_llm_error


def _make_http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/llm")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_retryable_llm_error_includes_429_and_5xx() -> None:
    assert is_retryable_llm_error(_make_http_status_error(429)) is True
    assert is_retryable_llm_error(_make_http_status_error(500)) is True
    assert is_retryable_llm_error(_make_http_status_error(503)) is True


def test_retryable_llm_error_excludes_4xx_and_non_transient_errors() -> None:
    assert is_retryable_llm_error(_make_http_status_error(400)) is False
    assert is_retryable_llm_error(ValueError("not retryable")) is False


def test_retryable_llm_error_includes_network_and_timeout_errors() -> None:
    assert is_retryable_llm_error(httpx.TimeoutException("timeout")) is True
    assert is_retryable_llm_error(httpx.NetworkError("network")) is True
