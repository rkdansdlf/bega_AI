from __future__ import annotations

import asyncio

from app.agents.tool_caller import ToolCall, ToolCaller, ToolDefinition


class _FakeHandlers:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def _tool_alpha(self, value: str) -> dict[str, str]:
        self.calls.append(("alpha", value))
        return {"value": value}

    def _tool_beta(self, year: int) -> dict[str, int]:
        self.calls.append(("beta", year))
        return {"year": year}


def test_tool_caller_from_definitions_resolves_bound_handlers() -> None:
    handlers = _FakeHandlers()
    definitions = (
        ToolDefinition(
            tool_name="alpha",
            description="alpha description",
            parameters_schema={"value": "value"},
            handler_attr="_tool_alpha",
        ),
        ToolDefinition(
            tool_name="beta",
            description="beta description",
            parameters_schema={"year": "year"},
            handler_attr="_tool_beta",
        ),
    )

    caller = ToolCaller.from_definitions(
        definitions,
        lambda handler_attr: getattr(handlers, handler_attr),
    )

    result = caller.execute_tool(ToolCall("alpha", {"value": "hello"}))

    assert result.success is True
    assert result.data == {"value": "hello"}
    assert handlers.calls == [("alpha", "hello")]
    assert caller.get_tool_descriptions() == ToolCaller.describe_definitions(
        definitions
    )


def test_tool_caller_from_definitions_keeps_parallel_execution_behavior() -> None:
    handlers = _FakeHandlers()
    definitions = (
        ToolDefinition(
            tool_name="alpha",
            description="alpha description",
            parameters_schema={"value": "value"},
            handler_attr="_tool_alpha",
        ),
        ToolDefinition(
            tool_name="beta",
            description="beta description",
            parameters_schema={"year": "year"},
            handler_attr="_tool_beta",
        ),
    )
    caller = ToolCaller.from_definitions(
        definitions,
        lambda handler_attr: getattr(handlers, handler_attr),
    )

    async def _run() -> list[object]:
        return await caller.execute_multiple_tools_parallel(
            [
                ToolCall("alpha", {"value": "x"}),
                ToolCall("beta", {"year": "2025년"}),
            ]
        )

    results = asyncio.run(_run())

    assert [result.success for result in results] == [True, True]
    assert handlers.calls == [("alpha", "x"), ("beta", 2025)]
