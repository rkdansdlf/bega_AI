from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.core.embeddings import EmbeddingError, _embed_openai


class _FakeResponse:
    def __init__(self, status_code: int, text: str, json_data=None) -> None:
        self.status_code = status_code
        self.text = text
        self._json_data = json_data

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class _FakeAsyncClient:
    def __init__(self, responses, counters) -> None:
        self._responses = responses
        self._counters = counters

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        del args, kwargs
        self._counters["post"] += 1
        return self._responses.pop(0)


def _make_settings() -> Settings:
    return Settings(
        embed_provider="openai",
        openai_api_key="test-key",
        openai_embed_model="text-embedding-3-small",
    )


def test_openai_embed_does_not_retry_insufficient_quota(monkeypatch) -> None:
    counters = {"post": 0, "sleep": 0}
    responses = [
        _FakeResponse(
            429,
            '{"error":{"message":"You exceeded your current quota","type":"insufficient_quota"}}',
        )
    ]

    monkeypatch.setattr(
        "app.core.embeddings.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(responses, counters),
    )

    async def _fake_sleep(_seconds: float) -> None:
        counters["sleep"] += 1

    monkeypatch.setattr("app.core.embeddings.asyncio.sleep", _fake_sleep)

    with pytest.raises(EmbeddingError, match="insufficient_quota"):
        asyncio.run(_embed_openai(["hello"], _make_settings(), max_concurrency=1))

    assert counters["post"] == 1
    assert counters["sleep"] == 0


def test_openai_embed_retries_transient_429(monkeypatch) -> None:
    counters = {"post": 0, "sleep": 0}
    responses = [
        _FakeResponse(
            429,
            '{"error":{"message":"Rate limit reached","type":"rate_limit_exceeded"}}',
        ),
        _FakeResponse(
            200,
            "",
            {"data": [{"embedding": [0.1, 0.2, 0.3]}]},
        ),
    ]

    monkeypatch.setattr(
        "app.core.embeddings.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(responses, counters),
    )

    async def _fake_sleep(_seconds: float) -> None:
        counters["sleep"] += 1

    monkeypatch.setattr("app.core.embeddings.asyncio.sleep", _fake_sleep)

    result = asyncio.run(_embed_openai(["hello"], _make_settings(), max_concurrency=1))

    assert result == [[0.1, 0.2, 0.3]]
    assert counters["post"] == 2
    assert counters["sleep"] == 1
