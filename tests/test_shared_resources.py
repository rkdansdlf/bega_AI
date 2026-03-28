from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from app.agents import baseball_agent as agent_module
from app.agents.baseball_agent import BaseballAgentRuntime
from app.config import get_settings
from app.core.http_clients import close_shared_httpx_clients, get_shared_httpx_client
from app.core.rag import RAGPipeline


class _FakeTool:
    def __init__(self, connection, settings=None):
        self.connection = connection
        self.settings = settings


class _FakeDbTool(_FakeTool):
    def get_team_basic_info(self, team_name):
        return {
            "team_name": team_name,
            "found": True,
            "error": None,
            "connection_label": getattr(self.connection, "label", None),
        }


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


def test_rag_pipeline_uses_shared_runtime_provider_when_runtime_is_unset(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeRuntime:
        def __init__(self):
            self.shared_agent = object()

        @contextmanager
        def request_context(self, connection):
            captured["request_context_connection"] = connection
            yield

    fake_runtime = _FakeRuntime()

    def _fake_initialize(settings):
        captured["settings"] = settings
        return fake_runtime

    monkeypatch.setattr(
        "app.core.rag.initialize_shared_baseball_agent_runtime",
        _fake_initialize,
    )

    pipeline = RAGPipeline(
        settings=get_settings(),
        connection=MagicMock(),
    )

    assert captured["settings"] is get_settings()
    assert pipeline.agent_runtime is fake_runtime
    assert pipeline.baseball_agent is fake_runtime.shared_agent


def test_rag_try_agent_first_runs_inside_runtime_request_context() -> None:
    captured: dict[str, object] = {}

    class _FakeAgent:
        async def process_query(self, query, context):
            captured["query"] = query
            captured["context"] = context
            return {
                "answer": "ok",
                "verified": True,
                "tool_results": [],
                "tool_calls": [],
                "data_sources": [],
            }

    class _FakeRuntime:
        def __init__(self):
            self.shared_agent = _FakeAgent()

        @contextmanager
        def request_context(self, connection):
            captured["request_context_connection"] = connection
            yield

    pipeline = RAGPipeline(
        settings=get_settings(),
        connection=MagicMock(),
        agent_runtime=_FakeRuntime(),
    )

    result = asyncio.run(
        pipeline._try_agent_first(
            "테스트 질의",
            intent="stats_lookup",
            filters={"team": "LG"},
            history=[{"role": "user", "content": "hi"}],
        )
    )

    assert captured["request_context_connection"] is pipeline.connection
    assert captured["query"] == "테스트 질의"
    assert captured["context"]["intent"] == "stats_lookup"
    assert result["strategy"] == "verified_agent"


@pytest.mark.asyncio
async def test_rag_try_agent_first_isolated_across_concurrent_pipelines(
    monkeypatch,
) -> None:
    monkeypatch.setattr(agent_module, "DatabaseQueryTool", _FakeDbTool)
    monkeypatch.setattr(agent_module, "RegulationQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "GameQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "DocumentQueryTool", _FakeTool)

    runtime = BaseballAgentRuntime(
        llm_generator=lambda *args, **kwargs: None,
        settings=get_settings(),
    )
    ready = asyncio.Event()
    entered: list[str] = []

    async def _fake_process_query(query, context):
        label = runtime.shared_agent.connection.label
        entered.append(label)
        if len(entered) == 2:
            ready.set()
        await ready.wait()
        return {
            "answer": label,
            "verified": True,
            "tool_results": [],
            "tool_calls": [],
            "data_sources": [label],
        }

    monkeypatch.setattr(runtime.shared_agent, "process_query", _fake_process_query)

    pipeline_one = RAGPipeline(
        settings=get_settings(),
        connection=SimpleNamespace(label="one"),
        agent_runtime=runtime,
    )
    pipeline_two = RAGPipeline(
        settings=get_settings(),
        connection=SimpleNamespace(label="two"),
        agent_runtime=runtime,
    )

    first, second = await asyncio.gather(
        pipeline_one._try_agent_first("첫 번째 질의"),
        pipeline_two._try_agent_first("두 번째 질의"),
    )

    assert first["strategy"] == "verified_agent"
    assert second["strategy"] == "verified_agent"
    assert {first["answer"], second["answer"]} == {"one", "two"}
    assert {tuple(first["data_sources"]), tuple(second["data_sources"])} == {
        ("one",),
        ("two",),
    }
