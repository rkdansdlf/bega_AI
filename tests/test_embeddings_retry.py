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
        self._counters["post"] += 1
        self._counters.setdefault("payloads", []).append(kwargs.get("json"))
        return self._responses.pop(0)


def _make_settings(**overrides) -> Settings:
    values = {
        "EMBED_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_EMBED_MODEL": "text-embedding-3-small",
    }
    values.update(overrides)
    return Settings(
        **values,
    )


def test_openai_embed_does_not_retry_insufficient_quota(monkeypatch) -> None:
    counters = {"post": 0, "sleep": 0}
    responses = [
        _FakeResponse(
            429,
            '{"error":{"message":"You exceeded your current quota","type":"insufficient_quota"}}',
        )
    ]

    fake_client = _FakeAsyncClient(responses, counters)
    monkeypatch.setattr(
        "app.core.embeddings.get_shared_httpx_client",
        lambda *args, **kwargs: fake_client,
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

    fake_client = _FakeAsyncClient(responses, counters)
    monkeypatch.setattr(
        "app.core.embeddings.get_shared_httpx_client",
        lambda *args, **kwargs: fake_client,
    )

    async def _fake_sleep(_seconds: float) -> None:
        counters["sleep"] += 1

    monkeypatch.setattr("app.core.embeddings.asyncio.sleep", _fake_sleep)

    result = asyncio.run(
        _embed_openai(["hello"], _make_settings(EMBED_DIM=3), max_concurrency=1)
    )

    assert result == [[0.1, 0.2, 0.3]]
    assert counters["post"] == 2
    assert counters["sleep"] == 1


def test_openai_embed_payload_includes_dimensions(monkeypatch) -> None:
    counters = {"post": 0}
    vector = [0.1] * 256
    responses = [_FakeResponse(200, "", {"data": [{"embedding": vector}]})]
    fake_client = _FakeAsyncClient(responses, counters)
    monkeypatch.setattr(
        "app.core.embeddings.get_shared_httpx_client",
        lambda *args, **kwargs: fake_client,
    )

    result = asyncio.run(
        _embed_openai(["hello"], _make_settings(EMBED_DIM=256), max_concurrency=1)
    )

    assert len(result[0]) == 256
    assert counters["payloads"][0]["dimensions"] == 256


def test_openai_embed_short_response_dimension_raises(monkeypatch) -> None:
    counters = {"post": 0}
    responses = [_FakeResponse(200, "", {"data": [{"embedding": [0.1] * 255}]})]
    fake_client = _FakeAsyncClient(responses, counters)
    monkeypatch.setattr(
        "app.core.embeddings.get_shared_httpx_client",
        lambda *args, **kwargs: fake_client,
    )

    with pytest.raises(EmbeddingError, match="기대값=256"):
        asyncio.run(
            _embed_openai(["hello"], _make_settings(EMBED_DIM=256), max_concurrency=1)
        )
