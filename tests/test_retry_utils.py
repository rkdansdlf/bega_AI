"""retry_utils.py 단위 테스트.

is_retryable_llm_error() 와 _classify_llm_error() 순수 로직을 검증한다.
httpx 예외를 직접 생성해 외부 의존성 없이 분류 로직을 검증한다.
"""
from __future__ import annotations

import httpx
import pytest

from app.core.retry_utils import _classify_llm_error, is_retryable_llm_error


def _make_http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/llm")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


# ── TestIsRetryableLLMError ───────────────────────────────────────────────────

class TestIsRetryableLLMError:
    def test_status_429_is_retryable(self):
        assert is_retryable_llm_error(_make_http_status_error(429)) is True

    def test_status_500_is_retryable(self):
        assert is_retryable_llm_error(_make_http_status_error(500)) is True

    def test_status_502_is_retryable(self):
        assert is_retryable_llm_error(_make_http_status_error(502)) is True

    def test_status_503_is_retryable(self):
        assert is_retryable_llm_error(_make_http_status_error(503)) is True

    def test_status_504_is_retryable(self):
        assert is_retryable_llm_error(_make_http_status_error(504)) is True

    def test_status_400_not_retryable(self):
        assert is_retryable_llm_error(_make_http_status_error(400)) is False

    def test_status_401_not_retryable(self):
        assert is_retryable_llm_error(_make_http_status_error(401)) is False

    def test_status_404_not_retryable(self):
        assert is_retryable_llm_error(_make_http_status_error(404)) is False

    def test_timeout_exception_is_retryable(self):
        assert is_retryable_llm_error(httpx.TimeoutException("timeout")) is True

    def test_network_error_is_retryable(self):
        assert is_retryable_llm_error(httpx.NetworkError("network error")) is True

    def test_remote_protocol_error_is_retryable(self):
        assert is_retryable_llm_error(httpx.RemoteProtocolError("protocol")) is True

    def test_read_error_is_retryable(self):
        assert is_retryable_llm_error(httpx.ReadError("read error")) is True

    def test_value_error_not_retryable(self):
        assert is_retryable_llm_error(ValueError("not retryable")) is False

    def test_runtime_error_not_retryable(self):
        assert is_retryable_llm_error(RuntimeError("crash")) is False

    def test_key_error_not_retryable(self):
        assert is_retryable_llm_error(KeyError("missing")) is False


# ── TestClassifyLLMError ──────────────────────────────────────────────────────

class TestClassifyLLMError:
    def test_status_429_classified_as_429(self):
        assert _classify_llm_error(_make_http_status_error(429)) == "429"

    def test_status_500_classified_as_5xx(self):
        assert _classify_llm_error(_make_http_status_error(500)) == "5xx"

    def test_status_503_classified_as_5xx(self):
        assert _classify_llm_error(_make_http_status_error(503)) == "5xx"

    def test_status_400_classified_as_status_string(self):
        assert _classify_llm_error(_make_http_status_error(400)) == "400"

    def test_status_404_classified_as_status_string(self):
        assert _classify_llm_error(_make_http_status_error(404)) == "404"

    def test_timeout_classified_as_timeout(self):
        assert _classify_llm_error(httpx.TimeoutException("timeout")) == "timeout"

    def test_network_error_classified_as_network(self):
        assert _classify_llm_error(httpx.NetworkError("network")) == "network"

    def test_read_error_classified_as_network(self):
        # ReadError is a NetworkError subclass
        assert _classify_llm_error(httpx.ReadError("read")) == "network"

    def test_value_error_classified_as_type_name(self):
        assert _classify_llm_error(ValueError("oops")) == "ValueError"

    def test_runtime_error_classified_as_type_name(self):
        assert _classify_llm_error(RuntimeError("crash")) == "RuntimeError"
