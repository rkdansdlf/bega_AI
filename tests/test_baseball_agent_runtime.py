from __future__ import annotations

import asyncio
from contextvars import copy_context
from types import SimpleNamespace

import pytest

from app.agents import baseball_agent as agent_module
from app.agents.baseball_agent import BaseballAgentRuntime, BaseballStatisticsAgent
from app.agents.tool_caller import ToolCall


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


async def _fake_llm_generator(messages, max_tokens=None):
    if False:
        yield messages, max_tokens


def test_runtime_reuses_shared_agent_and_rebinds_request_context(monkeypatch):
    monkeypatch.setattr(agent_module, "DatabaseQueryTool", _FakeDbTool)
    monkeypatch.setattr(agent_module, "RegulationQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "GameQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "DocumentQueryTool", _FakeTool)

    runtime = BaseballAgentRuntime(
        llm_generator=_fake_llm_generator,
        settings=SimpleNamespace(),
    )
    conn_one = SimpleNamespace(label="one")
    conn_two = SimpleNamespace(label="two")
    shared_agent = runtime.shared_agent

    with runtime.request_context(conn_one):
        assert shared_agent.runtime_id == runtime.runtime_id
        assert shared_agent.connection is conn_one
        db_query_one = shared_agent.db_query_tool
        document_query_one = shared_agent.document_query_tool

    with runtime.request_context(conn_two):
        assert runtime.shared_agent is shared_agent
        assert shared_agent.connection is conn_two
        db_query_two = shared_agent.db_query_tool
        document_query_two = shared_agent.document_query_tool

    assert shared_agent.llm_generator is _fake_llm_generator
    assert shared_agent.tool_definitions is runtime.tool_definitions
    assert shared_agent.tool_description_text is runtime.tool_description_text
    assert shared_agent.tool_caller.tools is runtime.tool_caller_factory.tools
    assert (
        shared_agent.tool_caller.get_tool_descriptions()
        is runtime.tool_description_text
    )
    assert (
        shared_agent.tool_caller._resolve_tool_function("get_player_stats").__self__
        is shared_agent
    )
    assert shared_agent.chat_intent_router._router is runtime.chat_intent_router
    assert (
        shared_agent.chat_renderer_registry._registry is runtime.chat_renderer_registry
    )
    assert db_query_one is not db_query_two
    assert document_query_one is not document_query_two
    assert db_query_one.connection is conn_one
    assert db_query_two.connection is conn_two
    assert document_query_one.connection is conn_one
    assert document_query_two.connection is conn_two


def test_runtime_request_context_resets_after_exit(monkeypatch):
    monkeypatch.setattr(agent_module, "DatabaseQueryTool", _FakeDbTool)
    monkeypatch.setattr(agent_module, "RegulationQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "GameQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "DocumentQueryTool", _FakeTool)

    runtime = BaseballAgentRuntime(
        llm_generator=_fake_llm_generator,
        settings=SimpleNamespace(),
    )

    with runtime.request_context(SimpleNamespace(label="one")):
        assert runtime.shared_agent.connection.label == "one"

    with pytest.raises(AttributeError):
        _ = runtime.shared_agent.connection


def test_runtime_request_context_cleanup_skips_cross_context_reset(monkeypatch):
    monkeypatch.setattr(agent_module, "DatabaseQueryTool", _FakeDbTool)
    monkeypatch.setattr(agent_module, "RegulationQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "GameQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "DocumentQueryTool", _FakeTool)

    runtime = BaseballAgentRuntime(
        llm_generator=_fake_llm_generator,
        settings=SimpleNamespace(),
    )

    handle = copy_context().run(
        runtime.enter_request_context,
        SimpleNamespace(label="cross-context"),
    )

    copy_context().run(runtime.exit_request_context, handle)

    with runtime.request_context(SimpleNamespace(label="next")):
        assert runtime.shared_agent.connection.label == "next"


@pytest.mark.asyncio
async def test_runtime_request_context_isolated_across_concurrent_tasks(monkeypatch):
    monkeypatch.setattr(agent_module, "DatabaseQueryTool", _FakeDbTool)
    monkeypatch.setattr(agent_module, "RegulationQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "GameQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "DocumentQueryTool", _FakeTool)

    runtime = BaseballAgentRuntime(
        llm_generator=_fake_llm_generator,
        settings=SimpleNamespace(),
    )
    ready = asyncio.Event()
    entered: list[str] = []

    async def _worker(label: str) -> tuple[str, str, int, int]:
        with runtime.request_context(SimpleNamespace(label=label)):
            before = runtime.shared_agent.connection.label
            db_tool = runtime.shared_agent.db_query_tool
            document_tool = runtime.shared_agent.document_query_tool
            entered.append(label)
            if len(entered) == 2:
                ready.set()
            await ready.wait()
            await asyncio.sleep(0)
            after = runtime.shared_agent.connection.label
            return before, after, id(db_tool), id(document_tool)

    first, second = await asyncio.gather(_worker("one"), _worker("two"))

    assert first[0] == first[1] == "one"
    assert second[0] == second[1] == "two"
    assert first[2] != second[2]
    assert first[3] != second[3]


def test_tool_caller_async_preserves_request_context(monkeypatch):
    monkeypatch.setattr(agent_module, "DatabaseQueryTool", _FakeDbTool)
    monkeypatch.setattr(agent_module, "RegulationQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "GameQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "DocumentQueryTool", _FakeTool)

    runtime = BaseballAgentRuntime(
        llm_generator=_fake_llm_generator,
        settings=SimpleNamespace(),
    )

    async def _run():
        with runtime.request_context(SimpleNamespace(label="async")):
            return (
                await runtime.shared_agent.tool_caller.execute_multiple_tools_parallel(
                    [
                        ToolCall("get_team_basic_info", {"team_name": "LG"}),
                        ToolCall("get_team_basic_info", {"team_name": "KT"}),
                    ],
                    max_concurrency=2,
                )
            )

    results = asyncio.run(_run())

    assert [result.success for result in results] == [True, True]
    assert [result.data["team_name"] for result in results] == ["LG", "KT"]


@pytest.mark.asyncio
async def test_tool_caller_async_isolated_across_concurrent_tasks(monkeypatch):
    monkeypatch.setattr(agent_module, "DatabaseQueryTool", _FakeDbTool)
    monkeypatch.setattr(agent_module, "RegulationQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "GameQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "DocumentQueryTool", _FakeTool)

    runtime = BaseballAgentRuntime(
        llm_generator=_fake_llm_generator,
        settings=SimpleNamespace(),
    )
    ready = asyncio.Event()
    entered: list[str] = []

    async def _worker(label: str) -> tuple[str, str]:
        with runtime.request_context(SimpleNamespace(label=label)):
            entered.append(label)
            if len(entered) == 2:
                ready.set()
            await ready.wait()
            results = (
                await runtime.shared_agent.tool_caller.execute_multiple_tools_parallel(
                    [ToolCall("get_team_basic_info", {"team_name": label})],
                    max_concurrency=1,
                )
            )
            return label, results[0].data["connection_label"]

    first, second = await asyncio.gather(_worker("one"), _worker("two"))

    assert first == ("one", "one")
    assert second == ("two", "two")


@pytest.mark.asyncio
async def test_shared_agent_stream_context_isolated_across_concurrent_tasks(
    monkeypatch,
):
    monkeypatch.setattr(agent_module, "DatabaseQueryTool", _FakeDbTool)
    monkeypatch.setattr(agent_module, "RegulationQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "GameQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "DocumentQueryTool", _FakeTool)

    runtime = BaseballAgentRuntime(
        llm_generator=_fake_llm_generator,
        settings=SimpleNamespace(),
    )
    ready = asyncio.Event()
    entered: list[str] = []

    async def _fake_stream(query: str, context=None):
        label = runtime.shared_agent.connection.label
        entered.append(label)
        if len(entered) == 2:
            ready.set()
        await ready.wait()
        yield {"type": "answer_chunk", "content": label}
        await asyncio.sleep(0)
        yield {
            "type": "metadata",
            "data": {"connection_label": runtime.shared_agent.connection.label},
        }

    monkeypatch.setattr(runtime.shared_agent, "process_query_stream", _fake_stream)

    async def _consume(label: str) -> list[dict[str, object]]:
        with runtime.request_context(SimpleNamespace(label=label)):
            return [
                event
                async for event in runtime.shared_agent.process_query_stream(
                    f"question-{label}",
                    {},
                )
            ]

    first, second = await asyncio.gather(_consume("one"), _consume("two"))

    assert first == [
        {"type": "answer_chunk", "content": "one"},
        {"type": "metadata", "data": {"connection_label": "one"}},
    ]
    assert second == [
        {"type": "answer_chunk", "content": "two"},
        {"type": "metadata", "data": {"connection_label": "two"}},
    ]


def test_runtime_id_guard_restores_outer_runtime_after_nested_contexts(monkeypatch):
    monkeypatch.setattr(agent_module, "DatabaseQueryTool", _FakeDbTool)
    monkeypatch.setattr(agent_module, "RegulationQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "GameQueryTool", _FakeTool)
    monkeypatch.setattr(agent_module, "DocumentQueryTool", _FakeTool)

    runtime_one = BaseballAgentRuntime(
        llm_generator=_fake_llm_generator,
        settings=SimpleNamespace(),
    )
    runtime_two = BaseballAgentRuntime(
        llm_generator=_fake_llm_generator,
        settings=SimpleNamespace(),
    )
    conn_one = SimpleNamespace(label="one")
    conn_two = SimpleNamespace(label="two")

    with runtime_one.request_context(conn_one):
        assert runtime_one.shared_agent.connection is conn_one
        with runtime_two.request_context(conn_two):
            assert runtime_two.shared_agent.connection is conn_two
            with pytest.raises(AttributeError):
                _ = runtime_one.shared_agent.connection
        assert runtime_one.shared_agent.connection is conn_one


def test_direct_agent_construction_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="runtime-managed"):
        BaseballStatisticsAgent(
            connection=SimpleNamespace(),
            llm_generator=_fake_llm_generator,
            settings=SimpleNamespace(),
        )
