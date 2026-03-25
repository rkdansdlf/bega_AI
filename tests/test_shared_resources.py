from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import httpx

from app.config import get_settings
from app.core.http_clients import close_shared_httpx_clients, get_shared_httpx_client
from app.core.rag import RAGPipeline
from app.core.shared_resources import get_shared_latest_baseball_tool


def test_shared_httpx_client_reuses_instance_and_resets_on_close() -> None:
    client1 = get_shared_httpx_client(
        "openrouter",
        timeout=120.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    client2 = get_shared_httpx_client(
        "openrouter",
        timeout=60.0,
        limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
    )

    assert client1 is client2

    asyncio.run(close_shared_httpx_clients())

    client3 = get_shared_httpx_client(
        "openrouter",
        timeout=120.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    assert client3 is not client1

    asyncio.run(close_shared_httpx_clients())


def test_rag_pipeline_uses_shared_latest_baseball_tool(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeAgent:
        def __init__(
            self,
            connection,
            llm_generator,
            settings=None,
            latest_baseball_tool=None,
            **kwargs,
        ):
            captured["latest_baseball_tool"] = latest_baseball_tool

    monkeypatch.setattr("app.core.rag.BaseballStatisticsAgent", _FakeAgent)

    pipeline = RAGPipeline(
        settings=get_settings(),
        connection=MagicMock(),
    )

    assert captured["latest_baseball_tool"] is get_shared_latest_baseball_tool()
    assert pipeline.baseball_agent is not None
