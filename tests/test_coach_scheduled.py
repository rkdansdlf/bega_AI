"""
Coach 기능 테스트 모듈.

Fast Path, 캐싱, 응답 검증기 등을 테스트합니다.
"""

import ast
import asyncio
from contextlib import suppress
import json
import logging
import os
import re
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tests.coach_test_support import (
    _build_game_evidence,
    _collect_sse_text,
    _extract_sse_meta_events,
    _install_coach_endpoint_cache_hit,
)


class TestAttachScheduledWinProbability:
    def _tool_results(self, home=(7, 3, 0), away=(5, 5, 0)):
        def block(record):
            wins, losses, draws = record
            return {
                "recent": {
                    "summary": {
                        "wins": wins,
                        "losses": losses,
                        "draws": draws,
                        "run_diff": wins - losses,
                    }
                }
            }

        return {"home": block(home), "away": block(away)}

    def test_uses_payload_value_when_valid(self):
        from app.routers.coach import _attach_scheduled_win_probability

        meta = {}
        _attach_scheduled_win_probability(
            meta,
            game_status_bucket="SCHEDULED",
            tool_results=self._tool_results(),
            response_payload={"win_probability_home": 0.6123},
        )
        assert meta["win_probability_home"] == 0.612

    def test_computes_when_payload_missing_and_data_sufficient(self):
        from app.routers.coach import _attach_scheduled_win_probability

        meta = {}
        _attach_scheduled_win_probability(
            meta,
            game_status_bucket="SCHEDULED",
            tool_results=self._tool_results(),
            response_payload=None,
        )
        assert "win_probability_home" in meta
        assert 0.30 <= meta["win_probability_home"] <= 0.75

    def test_no_key_when_data_insufficient(self):
        from app.routers.coach import _attach_scheduled_win_probability

        meta = {}
        _attach_scheduled_win_probability(
            meta,
            game_status_bucket="SCHEDULED",
            tool_results={"home": {}, "away": {}},
            response_payload={"win_probability_home": 1.5},  # 범위 밖 → 무시
        )
        assert "win_probability_home" not in meta

    def test_no_key_when_not_scheduled(self):
        from app.routers.coach import _attach_scheduled_win_probability

        meta = {}
        _attach_scheduled_win_probability(
            meta,
            game_status_bucket="COMPLETED",
            tool_results=self._tool_results(),
            response_payload={"win_probability_home": 0.55},
        )
        assert "win_probability_home" not in meta


def test_auto_brief_background_generation_does_not_wait_for_heartbeat(
    monkeypatch,
) -> None:
    from app.routers import coach as coach_router

    reached_generation = asyncio.Event()

    async def heartbeat(**kwargs):
        await asyncio.Event().wait()

    async def execute_tools(*args, **kwargs):
        reached_generation.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(coach_router, "_heartbeat_cache_lease", heartbeat)
    monkeypatch.setattr(coach_router, "_execute_coach_tools_parallel", execute_tools)
    monkeypatch.setattr(
        coach_router,
        "_should_include_auto_brief_clutch",
        lambda evidence: False,
    )

    async def exercise() -> None:
        task = asyncio.create_task(
            coach_router._generate_auto_brief_cache_background(
                pool=object(),
                cache_key="background-scheduling",
                lease_owner="owner",
                home_team_canonical="SS",
                away_team_canonical="HH",
                home_name="home",
                away_name="away",
                year=2026,
                resolved_focus=["recent_form"],
                game_evidence=SimpleNamespace(
                    game_date="2026-07-18",
                    game_id="20260718SSHH0",
                    season_id=202600,
                    stage_label="REGULAR",
                    game_status_bucket="SCHEDULED",
                ),
                evidence_assessment=object(),
                analysis_type="game_preview",
                coach_model_name="model",
                cache_attempt_count=1,
            )
        )
        try:
            await asyncio.wait_for(reached_generation.wait(), timeout=0.1)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    asyncio.run(exercise())


def test_auto_brief_background_task_is_retained_until_completion() -> None:
    from app.routers import coach as coach_router

    async def exercise() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def worker() -> None:
            started.set()
            await release.wait()

        task = coach_router._schedule_auto_brief_background_task(worker())
        assert task in coach_router._AUTO_BRIEF_BACKGROUND_TASKS

        await asyncio.wait_for(started.wait(), timeout=0.1)
        release.set()
        await task
        await asyncio.sleep(0)

        assert task not in coach_router._AUTO_BRIEF_BACKGROUND_TASKS

    asyncio.run(exercise())


def test_auto_brief_background_task_rejects_capacity_overflow(monkeypatch) -> None:
    from app.routers import coach as coach_router

    monkeypatch.setattr(coach_router, "COACH_AUTO_BRIEF_BACKGROUND_MAX_TASKS", 1)

    async def exercise() -> None:
        blocker = asyncio.Event()

        async def worker() -> None:
            await blocker.wait()

        active_task = coach_router._schedule_auto_brief_background_task(worker())
        try:
            with pytest.raises(
                TimeoutError,
                match="auto_brief_background_capacity_exceeded",
            ):
                coach_router._schedule_auto_brief_background_task(worker())
        finally:
            active_task.cancel()
            with suppress(asyncio.CancelledError):
                await active_task

    asyncio.run(exercise())


@pytest.mark.asyncio
async def test_auto_brief_capacity_overflow_returns_terminal_retryable_sse(
    monkeypatch,
) -> None:
    from app.routers import coach as coach_router

    evidence = _build_game_evidence(
        game_row_found=True,
        season_year=2026,
        season_id=202600,
        game_id="20260718SSHH0",
        game_date="2026-07-18",
        game_status="SCHEDULED",
        game_status_bucket="SCHEDULED",
        league_type_code=0,
    )
    agent = _install_coach_endpoint_cache_hit(monkeypatch, evidence)
    stored_failure = {}

    async def claim_cache_generation(**kwargs):
        return "MISS_GENERATE", None, None, None, 1

    async def store_failure(**kwargs):
        stored_failure.update(kwargs)
        return {"outcome": "updated"}

    def reject_background_task(coroutine):
        coroutine.close()
        raise TimeoutError("auto_brief_background_capacity_exceeded")

    monkeypatch.setattr(
        coach_router,
        "_claim_cache_generation",
        claim_cache_generation,
    )
    monkeypatch.setattr(coach_router, "_store_failed_cache", store_failure)
    monkeypatch.setattr(
        coach_router,
        "_schedule_auto_brief_background_task",
        reject_background_task,
    )

    response = await coach_router.analyze_team(
        coach_router.AnalyzeRequest(
            home_team_id="SS",
            away_team_id="HH",
            game_id="20260718SSHH0",
            request_mode=coach_router.COACH_REQUEST_MODE_AUTO,
            analysis_type=coach_router.COACH_ANALYSIS_TYPE_PREVIEW,
            league_context={
                "season": 202600,
                "season_year": 2026,
                "game_date": "2026-07-18",
                "league_type": "REGULAR",
                "league_type_code": 0,
            },
        ),
        agent,
        None,
        None,
    )

    sse_text = await _collect_sse_text(response)
    meta_events = _extract_sse_meta_events(sse_text)

    assert "event: error" in sse_text
    assert "data: [DONE]" in sse_text
    assert meta_events[-1]["cache_state"] == "FAILED_RETRY_WAIT"
    assert stored_failure["error_code"] == coach_router.COACH_LLM_TIMEOUT_ERROR_CODE


def test_auto_brief_background_tasks_are_cancelled_on_shutdown() -> None:
    from app.routers import coach as coach_router

    async def exercise() -> None:
        async def worker() -> None:
            await asyncio.Event().wait()

        task = coach_router._schedule_auto_brief_background_task(worker())
        assert task in coach_router._AUTO_BRIEF_BACKGROUND_TASKS

        await coach_router.cancel_auto_brief_background_tasks()

        assert task.cancelled()
        assert task not in coach_router._AUTO_BRIEF_BACKGROUND_TASKS

    asyncio.run(exercise())


def test_auto_brief_background_generation_times_out_and_persists_failure(
    monkeypatch,
) -> None:
    from app.routers import coach as coach_router

    stored_failure = {}

    async def heartbeat(**kwargs):
        await asyncio.Event().wait()

    async def execute_tools(*args, **kwargs):
        await asyncio.Event().wait()

    async def store_failure(**kwargs):
        stored_failure.update(kwargs)
        return {"outcome": "updated"}

    monkeypatch.setattr(coach_router, "_heartbeat_cache_lease", heartbeat)
    monkeypatch.setattr(coach_router, "_execute_coach_tools_parallel", execute_tools)
    monkeypatch.setattr(coach_router, "_store_failed_cache", store_failure)
    monkeypatch.setattr(
        coach_router,
        "_should_include_auto_brief_clutch",
        lambda evidence: False,
    )
    monkeypatch.setattr(
        coach_router,
        "COACH_AUTO_BRIEF_BACKGROUND_TIMEOUT_SECONDS",
        0.01,
    )

    asyncio.run(
        asyncio.wait_for(
            coach_router._generate_auto_brief_cache_background(
                pool=object(),
                cache_key="background-timeout",
                lease_owner="owner",
                home_team_canonical="SS",
                away_team_canonical="HH",
                home_name="home",
                away_name="away",
                year=2026,
                resolved_focus=["recent_form"],
                game_evidence=SimpleNamespace(
                    game_date="2026-07-18",
                    game_id="20260718SSHH0",
                    season_id=202600,
                    stage_label="REGULAR",
                    game_status_bucket="SCHEDULED",
                ),
                evidence_assessment=object(),
                analysis_type="game_preview",
                coach_model_name="model",
                cache_attempt_count=1,
            ),
            timeout=1.0,
        )
    )

    assert stored_failure["error_code"] == coach_router.COACH_LLM_TIMEOUT_ERROR_CODE
    assert stored_failure["error_message"] == "auto_brief_background_timeout"


def test_coach_create_task_calls_do_not_await_their_coroutine_argument() -> None:
    source_path = Path(__file__).parents[1] / "app" / "routers" / "coach.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    offender_lines = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "create_task"
            and isinstance(function.value, ast.Name)
            and function.value.id == "asyncio"
        ):
            continue
        if isinstance(node.args[0], ast.Await):
            offender_lines.append(node.lineno)

    assert offender_lines == []


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
