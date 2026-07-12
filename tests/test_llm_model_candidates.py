from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app.agents import runtime_factory
from app.agents.runtime_factory import (
    build_baseball_llm_generator,
    resolve_openrouter_model_candidates,
)


def test_openrouter_model_candidates_include_gpt_oss_fallback() -> None:
    assert resolve_openrouter_model_candidates(
        "openrouter/free",
        ["openai/gpt-oss-120b"],
    ) == ["openrouter/free", "openai/gpt-oss-120b"]


def test_openrouter_model_candidates_skip_blocked_auto_and_dedupe() -> None:
    assert resolve_openrouter_model_candidates(
        "openrouter/auto",
        ["openai/gpt-oss-120b", "openai/gpt-oss-120b"],
    ) == ["openai/gpt-oss-120b"]


@pytest.mark.asyncio
async def test_openrouter_generator_uses_model_override_without_fallback(
    monkeypatch,
) -> None:
    attempted_models: list[str] = []
    observations: list[dict[str, object]] = []

    def observe(**payload: object) -> None:
        observations.append(dict(payload))

    class _FakeResponse:
        status_code = 200

        async def aread(self) -> bytes:
            return b""

        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": "override response"}}]}
            )
            yield "data: [DONE]"

    class _FakeStreamContext:
        async def __aenter__(self) -> _FakeResponse:
            return _FakeResponse()

        async def __aexit__(self, exc_type, exc, traceback) -> bool:
            return False

    class _FakeClient:
        def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStreamContext:
            attempted_models.append(kwargs["json"]["model"])
            return _FakeStreamContext()

    monkeypatch.setattr(
        runtime_factory,
        "get_shared_httpx_client",
        lambda *args, **kwargs: _FakeClient(),
    )
    generator = build_baseball_llm_generator(
        SimpleNamespace(
            llm_provider="openrouter",
            openrouter_api_key="test-key",
            openrouter_base_url="https://openrouter.example/api/v1",
            openrouter_referer="",
            openrouter_app_title="",
            openrouter_model="primary/free",
            openrouter_fallback_models=["openai/gpt-oss-120b"],
            max_output_tokens=512,
            chat_openrouter_empty_chunk_retries=0,
            chat_openrouter_empty_chunk_backoff_ms=50,
        )
    )

    chunks = [
        chunk
        async for chunk in generator(
            [{"role": "user", "content": "hello"}],
            model_override="openrouter/cheap-planner",
            usage_observer=observe,
        )
    ]

    assert attempted_models == ["openrouter/cheap-planner"]
    assert chunks == ["override response"]
    assert observations == [
        {
            "provider": "openrouter",
            "model": "openrouter/cheap-planner",
            "messages": [{"role": "user", "content": "hello"}],
            "output_text": "override response",
            "outcome": "success",
        }
    ]


@pytest.mark.asyncio
async def test_openrouter_empty_choices_retry_reports_each_attempt(monkeypatch) -> None:
    attempted_models: list[str] = []
    observations: list[dict[str, object]] = []

    class _FakeResponse:
        status_code = 200

        async def aread(self) -> bytes:
            return b""

        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            if len(attempted_models) == 1:
                yield "data: " + json.dumps({"choices": []})
            else:
                yield "data: " + json.dumps(
                    {"choices": [{"delta": {"content": "retried response"}}]}
                )
            yield "data: [DONE]"

    class _FakeStreamContext:
        async def __aenter__(self) -> _FakeResponse:
            return _FakeResponse()

        async def __aexit__(self, exc_type, exc, traceback) -> bool:
            return False

    class _FakeClient:
        def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStreamContext:
            attempted_models.append(kwargs["json"]["model"])
            return _FakeStreamContext()

    monkeypatch.setattr(
        runtime_factory,
        "get_shared_httpx_client",
        lambda *args, **kwargs: _FakeClient(),
    )
    generator = build_baseball_llm_generator(
        SimpleNamespace(
            llm_provider="openrouter",
            openrouter_api_key="test-key",
            openrouter_base_url="https://openrouter.example/api/v1",
            openrouter_referer="",
            openrouter_app_title="",
            openrouter_model="primary/free",
            openrouter_fallback_models=[],
            max_output_tokens=512,
            chat_openrouter_empty_chunk_retries=1,
            chat_openrouter_empty_chunk_backoff_ms=50,
        )
    )

    def observe(**payload: object) -> None:
        observations.append(dict(payload))

    chunks = [
        chunk
        async for chunk in generator(
            [{"role": "user", "content": "hello"}],
            usage_observer=observe,
        )
    ]

    assert attempted_models == ["primary/free", "primary/free"]
    assert chunks == ["retried response"]
    assert observations == [
        {
            "provider": "openrouter",
            "model": "primary/free",
            "messages": [{"role": "user", "content": "hello"}],
            "output_text": "",
            "outcome": "failed",
        },
        {
            "provider": "openrouter",
            "model": "primary/free",
            "messages": [{"role": "user", "content": "hello"}],
            "output_text": "retried response",
            "outcome": "success",
        },
    ]


@pytest.mark.asyncio
async def test_gemini_generator_reports_generation_attempt(monkeypatch) -> None:
    import google.generativeai as genai
    from google.generativeai import types as genai_types

    observations: list[dict[str, object]] = []

    class _FakeChunk:
        def __init__(self, text: str) -> None:
            self.text = text

    class _FakeModel:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            self.model_name = model_name

        async def generate_content_async(self, *args: Any, **kwargs: Any):
            async def _response():
                yield _FakeChunk("gemini ")
                yield _FakeChunk("response")

            return _response()

    monkeypatch.setattr(runtime_factory, "_ensure_gemini_configured", lambda settings: None)
    monkeypatch.setattr(genai, "GenerativeModel", _FakeModel)
    monkeypatch.setattr(genai_types, "GenerationConfig", lambda **kwargs: kwargs)
    generator = build_baseball_llm_generator(
        SimpleNamespace(
            llm_provider="gemini",
            gemini_api_key="test-key",
            gemini_model="gemini-2.0-flash",
            max_output_tokens=512,
        )
    )

    chunks = [
        chunk
        async for chunk in generator(
            [{"role": "user", "content": "hello"}],
            usage_observer=lambda **payload: observations.append(dict(payload)),
        )
    ]

    assert chunks == ["gemini ", "response"]
    assert observations == [
        {
            "provider": "gemini",
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "hello"}],
            "output_text": "gemini response",
            "outcome": "success",
        }
    ]


@pytest.mark.asyncio
async def test_openrouter_429_falls_back_to_gpt_oss(monkeypatch) -> None:
    attempted_models: list[str] = []
    fallback_reasons: list[dict[str, str]] = []
    observations: list[dict[str, object]] = []

    class _FakeFallbackCounter:
        def labels(self, **labels: str) -> "_FakeFallbackCounter":
            fallback_reasons.append(labels)
            return self

        def inc(self) -> None:
            return None

    class _FakeResponse:
        def __init__(
            self,
            *,
            status_code: int,
            body: bytes = b"",
            lines: list[str] | None = None,
        ) -> None:
            self.status_code = status_code
            self._body = body
            self._lines = lines or []
            self.request = httpx.Request(
                "POST", "https://openrouter.example/chat/completions"
            )

        async def aread(self) -> bytes:
            return self._body

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                response = httpx.Response(
                    self.status_code,
                    content=self._body,
                    request=self.request,
                )
                raise httpx.HTTPStatusError(
                    f"HTTP {self.status_code}",
                    request=self.request,
                    response=response,
                )

        async def aiter_lines(self):
            for line in self._lines:
                yield line

    class _FakeStreamContext:
        def __init__(self, response: _FakeResponse) -> None:
            self._response = response

        async def __aenter__(self) -> _FakeResponse:
            return self._response

        async def __aexit__(self, exc_type, exc, traceback) -> bool:
            return False

    class _FakeClient:
        def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStreamContext:
            payload = kwargs["json"]
            attempted_models.append(payload["model"])

            if payload["model"] == "primary/free":
                return _FakeStreamContext(
                    _FakeResponse(
                        status_code=429,
                        body=b'{"error":{"message":"daily limit exceeded"}}',
                    )
                )

            return _FakeStreamContext(
                _FakeResponse(
                    status_code=200,
                    lines=[
                        "data: "
                        + json.dumps(
                            {"choices": [{"delta": {"content": "fallback response"}}]}
                        ),
                        "data: [DONE]",
                    ],
                )
            )

    monkeypatch.setattr(
        runtime_factory,
        "get_shared_httpx_client",
        lambda *args, **kwargs: _FakeClient(),
    )
    monkeypatch.setattr(
        runtime_factory,
        "AI_LLM_FALLBACK_TOTAL",
        _FakeFallbackCounter(),
    )
    generator = build_baseball_llm_generator(
        SimpleNamespace(
            llm_provider="openrouter",
            openrouter_api_key="test-key",
            openrouter_base_url="https://openrouter.example/api/v1",
            openrouter_referer="",
            openrouter_app_title="",
            openrouter_model="primary/free",
            openrouter_fallback_models=["openai/gpt-oss-120b"],
            max_output_tokens=512,
            chat_openrouter_empty_chunk_retries=0,
            chat_openrouter_empty_chunk_backoff_ms=50,
        )
    )

    chunks = [
        chunk
        async for chunk in generator(
            [{"role": "user", "content": "hello"}],
            usage_observer=lambda **payload: observations.append(dict(payload)),
        )
    ]

    assert attempted_models == ["primary/free", "openai/gpt-oss-120b"]
    assert fallback_reasons == [{"provider": "openrouter", "reason": "http_status_429"}]
    assert chunks == ["fallback response"]
    assert observations == [
        {
            "provider": "openrouter",
            "model": "primary/free",
            "messages": [{"role": "user", "content": "hello"}],
            "output_text": "",
            "outcome": "failed",
        },
        {
            "provider": "openrouter",
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": "hello"}],
            "output_text": "fallback response",
            "outcome": "success",
        },
    ]


@pytest.mark.asyncio
async def test_openrouter_429_fallback_failure_raises_last_model_error(
    monkeypatch,
) -> None:
    attempted_models: list[str] = []
    fallback_reasons: list[dict[str, str]] = []

    class _FakeFallbackCounter:
        def labels(self, **labels: str) -> "_FakeFallbackCounter":
            fallback_reasons.append(labels)
            return self

        def inc(self) -> None:
            return None

    class _FakeResponse:
        def __init__(self, *, status_code: int, body: bytes) -> None:
            self.status_code = status_code
            self._body = body
            self.request = httpx.Request(
                "POST", "https://openrouter.example/chat/completions"
            )

        async def aread(self) -> bytes:
            return self._body

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                response = httpx.Response(
                    self.status_code,
                    content=self._body,
                    request=self.request,
                )
                raise httpx.HTTPStatusError(
                    f"HTTP {self.status_code}",
                    request=self.request,
                    response=response,
                )

        async def aiter_lines(self):
            for line in ():
                yield line

    class _FakeStreamContext:
        def __init__(self, response: _FakeResponse) -> None:
            self._response = response

        async def __aenter__(self) -> _FakeResponse:
            return self._response

        async def __aexit__(self, exc_type, exc, traceback) -> bool:
            return False

    class _FakeClient:
        def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStreamContext:
            payload = kwargs["json"]
            attempted_models.append(payload["model"])

            if payload["model"] == "primary/free":
                return _FakeStreamContext(
                    _FakeResponse(
                        status_code=429,
                        body=b'{"error":{"message":"daily limit exceeded"}}',
                    )
                )

            return _FakeStreamContext(
                _FakeResponse(
                    status_code=401,
                    body=b'{"error":{"message":"fallback unauthorized"}}',
                )
            )

    monkeypatch.setattr(
        runtime_factory,
        "get_shared_httpx_client",
        lambda *args, **kwargs: _FakeClient(),
    )
    monkeypatch.setattr(
        runtime_factory,
        "AI_LLM_FALLBACK_TOTAL",
        _FakeFallbackCounter(),
    )
    generator = build_baseball_llm_generator(
        SimpleNamespace(
            llm_provider="openrouter",
            openrouter_api_key="test-key",
            openrouter_base_url="https://openrouter.example/api/v1",
            openrouter_referer="",
            openrouter_app_title="",
            openrouter_model="primary/free",
            openrouter_fallback_models=["openai/gpt-oss-120b"],
            max_output_tokens=512,
            chat_openrouter_empty_chunk_retries=0,
            chat_openrouter_empty_chunk_backoff_ms=50,
        )
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        _ = [chunk async for chunk in generator([{"role": "user", "content": "hello"}])]

    assert attempted_models == ["primary/free", "openai/gpt-oss-120b"]
    assert fallback_reasons == [{"provider": "openrouter", "reason": "http_status_429"}]
    assert exc_info.value.response.status_code == 401
