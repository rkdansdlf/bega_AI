"""Regression tests for coach auto-brief background task scheduling."""

import ast
import asyncio
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

from app.routers import coach


def test_background_generation_schedules_heartbeat_without_blocking_tools(
    monkeypatch,
):
    async def run_scenario() -> None:
        heartbeat_started = asyncio.Event()
        heartbeat_cancelled = asyncio.Event()
        tools_started = asyncio.Event()
        keep_running = asyncio.Event()

        async def fake_heartbeat_cache_lease(**_kwargs) -> None:
            heartbeat_started.set()
            try:
                await keep_running.wait()
            finally:
                heartbeat_cancelled.set()

        async def fake_execute_coach_tools_parallel(*_args, **_kwargs):
            tools_started.set()
            await keep_running.wait()
            return {}

        monkeypatch.setattr(
            coach,
            "_heartbeat_cache_lease",
            fake_heartbeat_cache_lease,
        )
        monkeypatch.setattr(
            coach,
            "_execute_coach_tools_parallel",
            fake_execute_coach_tools_parallel,
        )
        monkeypatch.setattr(
            coach,
            "_should_include_auto_brief_clutch",
            lambda _evidence: False,
        )

        task = asyncio.create_task(
            coach._generate_auto_brief_cache_background(
                pool=object(),
                cache_key="auto-brief:test",
                lease_owner="test-owner",
                home_team_canonical="HH",
                away_team_canonical="LT",
                home_name="한화",
                away_name="롯데",
                year=2026,
                resolved_focus=[],
                game_evidence=SimpleNamespace(
                    game_date="2026-07-18",
                    game_id=1,
                    season_id=2026,
                    stage_label="REGULAR",
                ),
                evidence_assessment=SimpleNamespace(),
                analysis_type="auto_brief",
                coach_model_name="test-model",
                cache_attempt_count=1,
            )
        )
        try:
            await asyncio.wait_for(heartbeat_started.wait(), timeout=0.5)
            await asyncio.wait_for(tools_started.wait(), timeout=0.5)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            assert heartbeat_cancelled.is_set()

    asyncio.run(run_scenario())


def test_task_schedulers_do_not_await_coroutines_before_enqueueing():
    tree = ast.parse(Path(coach.__file__).read_text(encoding="utf-8"))
    awaited_task_calls = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_task"
            and isinstance(node.args[0], ast.Await)
        ):
            continue

        awaited_value = node.args[0].value
        awaited_name = "<unknown>"
        if isinstance(awaited_value, ast.Call):
            if isinstance(awaited_value.func, ast.Name):
                awaited_name = awaited_value.func.id
            elif isinstance(awaited_value.func, ast.Attribute):
                awaited_name = awaited_value.func.attr
        awaited_task_calls.append((node.lineno, awaited_name))

    assert awaited_task_calls == [], (
        "coroutines must be passed directly to asyncio.create_task; "
        f"found awaited calls {awaited_task_calls}"
    )
