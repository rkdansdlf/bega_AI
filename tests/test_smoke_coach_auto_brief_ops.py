from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from app.routers.coach_auto_brief_ops import (
    CoachAutoBriefHealthInputs,
    build_auto_brief_ops_gate,
    CoachAutoBriefOpsHealthResponse,
    CoachAutoBriefOpsLatestReport,
    CoachAutoBriefOpsSummary,
)
from scripts import smoke_coach_auto_brief_ops as smoke
from scripts.batch_coach_matchup_cache import MatchupTarget


def _build_target(*, cache_key: str, game_date: str) -> MatchupTarget:
    return MatchupTarget(
        cache_key=cache_key,
        game_id=f"{cache_key}-game",
        season_id=20260,
        season_year=2026,
        game_date=game_date,
        game_type="REGULAR",
        home_team_id="LG",
        away_team_id="KT",
        league_type_code=0,
        stage_label="REGULAR",
        series_game_no=None,
        game_status_bucket="SCHEDULED",
        starter_signature="starter:none",
        lineup_signature="lineup:none",
        request_focus=["recent_form"],
        request_mode="auto_brief",
        question_override=None,
    )


def _build_snapshot(
    *,
    selected_target_count: int = 5,
    unresolved_count: int = 0,
    failed_locked_count: int = 0,
    pending_wait_count: int = 0,
    insufficient_count: int = 0,
    include_latest_report: bool = True,
) -> CoachAutoBriefOpsHealthResponse:
    latest_report = None
    if include_latest_report:
        latest_report = CoachAutoBriefOpsLatestReport(
            path="reports/coach_auto_brief_latest.json",
            run_started_at="2026-04-08T08:55:00Z",
            run_finished_at="2026-04-08T09:00:00Z",
            date_window="2026-04-08",
            unresolved_count=unresolved_count,
            completed_count=max(0, selected_target_count - unresolved_count),
            cache_state_breakdown={"FAILED_LOCKED": failed_locked_count},
            data_quality_breakdown={"insufficient": insufficient_count},
        )

    summary = CoachAutoBriefOpsSummary(
        loaded_target_count=selected_target_count,
        selected_target_count=selected_target_count,
        generated_success_count=max(0, selected_target_count - unresolved_count),
        cache_hit_count=0,
        in_progress_count=pending_wait_count,
        failed_count=failed_locked_count,
        unresolved_count=unresolved_count,
        completed_count=max(0, selected_target_count - unresolved_count),
        cache_state_breakdown={
            "COMPLETED": max(0, selected_target_count - unresolved_count),
            "FAILED_LOCKED": failed_locked_count,
            "PENDING_WAIT": pending_wait_count,
        },
        data_quality_breakdown={
            "grounded": max(0, selected_target_count - insufficient_count),
            "insufficient": insufficient_count,
        },
    )

    return CoachAutoBriefOpsHealthResponse(
        window="today",
        date_window="2026-04-08",
        generated_at_utc=datetime(2026, 4, 8, 0, 0, tzinfo=timezone.utc),
        runbook_path="task/operations/coach-auto-brief-prewarm-runbook.md",
        recommended_command="./.venv/bin/python scripts/batch_coach_auto_brief.py",
        summary=summary,
        gate=build_auto_brief_ops_gate(
            summary=summary,
            latest_report=latest_report,
        ),
        unresolved_targets=[],
        latest_report=latest_report,
    )


def test_select_warm_smoke_target_picks_first_completed_result() -> None:
    targets = [
        _build_target(cache_key="locked", game_date="2026-04-08"),
        _build_target(cache_key="completed", game_date="2026-04-08"),
    ]
    results = [
        {"status": "failed", "meta": {"cache_state": "FAILED_LOCKED"}},
        {"status": "skipped", "meta": {"cache_state": "COMPLETED", "cached": True}},
    ]

    selected = smoke._select_warm_smoke_target(targets, results)

    assert selected is not None
    assert selected.cache_key == "completed"


def test_build_gate_report_fails_on_threshold_and_smoke_miss() -> None:
    snapshot = _build_snapshot(
        selected_target_count=4,
        unresolved_count=2,
        failed_locked_count=1,
        insufficient_count=2,
    )

    report = smoke.build_gate_report(
        snapshot=snapshot,
        thresholds=smoke.GateThresholds(
            max_unresolved=0,
            max_failed_locked=0,
            max_pending_wait=None,
            max_insufficient_ratio=0.25,
            min_selected_targets=0,
            fail_on_missing_report=False,
        ),
        smoke_result=smoke.WarmPathSmokeResult(
            attempted=True,
            ok=False,
            reason="warm_path_miss",
            game_id="completed-game",
            cache_key="completed",
            status="generated",
            cache_state="COMPLETED",
            cached=False,
            elapsed_ms=820,
        ),
    )

    assert report["verdict"] == "FAIL"
    assert any("unresolved_count 2 > 0" in item for item in report["checks"]["failed"])
    assert any(
        "failed_locked_count 1 > 0" in item for item in report["checks"]["failed"]
    )
    assert any(
        "insufficient_ratio 0.500 > 0.250" in item
        for item in report["checks"]["failed"]
    )
    assert any("warm_path_smoke failed" in item for item in report["checks"]["failed"])


def test_build_gate_report_fails_when_report_missing_for_non_empty_window() -> None:
    snapshot = _build_snapshot(
        selected_target_count=3,
        unresolved_count=0,
        include_latest_report=False,
    )

    report = smoke.build_gate_report(
        snapshot=snapshot,
        thresholds=smoke.GateThresholds(
            max_unresolved=0,
            max_failed_locked=0,
            max_pending_wait=None,
            max_insufficient_ratio=None,
            min_selected_targets=1,
            fail_on_missing_report=True,
        ),
        smoke_result=smoke.WarmPathSmokeResult(
            attempted=False,
            ok=True,
            reason="skipped_by_flag",
        ),
    )

    assert report["verdict"] == "FAIL"
    assert (
        "latest auto_brief report missing for selected window"
        in report["checks"]["failed"]
    )


def test_build_gate_report_warns_for_off_day_without_failing() -> None:
    snapshot = _build_snapshot(
        selected_target_count=0,
        unresolved_count=0,
        include_latest_report=False,
    )

    report = smoke.build_gate_report(
        snapshot=snapshot,
        thresholds=smoke.GateThresholds(
            max_unresolved=0,
            max_failed_locked=0,
            max_pending_wait=0,
            max_insufficient_ratio=0.0,
            min_selected_targets=0,
            fail_on_missing_report=True,
        ),
        smoke_result=smoke.WarmPathSmokeResult(
            attempted=False,
            ok=True,
            reason="skipped_by_flag",
        ),
    )

    assert report["verdict"] == "WARN"
    assert report["checks"]["failed"] == []
    assert any(
        "selected target window is empty" in item
        for item in report["checks"]["warnings"]
    )


def test_async_main_writes_output_for_passing_gate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    target = _build_target(cache_key="completed", game_date="2026-04-08")
    inputs = CoachAutoBriefHealthInputs(
        years=[2026],
        target_pool=[target],
        selected_targets=[target],
        results=[
            {"status": "skipped", "meta": {"cache_state": "COMPLETED", "cached": True}}
        ],
    )
    snapshot = _build_snapshot(selected_target_count=1, unresolved_count=0)
    output_path = tmp_path / "auto_brief_ops_gate.json"

    monkeypatch.setattr(
        smoke,
        "_resolve_requested_window",
        lambda **kwargs: (date(2026, 4, 8), date(2026, 4, 8)),
    )
    monkeypatch.setattr(
        smoke, "collect_auto_brief_health_inputs", lambda **kwargs: inputs
    )
    monkeypatch.setattr(
        smoke,
        "_build_auto_brief_health_snapshot_from_inputs",
        lambda **kwargs: snapshot,
    )

    args = SimpleNamespace(
        window="today",
        start_date=None,
        end_date=None,
        sample_size=5,
        max_unresolved=0,
        max_failed_locked=0,
        max_pending_wait=None,
        max_insufficient_ratio=None,
        min_selected_targets=0,
        fail_on_missing_report=False,
        base_url="http://127.0.0.1:8001",
        internal_api_key="secret",
        smoke_timeout=15.0,
        skip_warm_smoke=True,
        output=str(output_path),
    )

    exit_code = asyncio.run(smoke.async_main(args))

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["warm_path_smoke"]["reason"] == "skipped_by_flag"
    assert payload["health"]["summary"]["selected_target_count"] == 1


def test_async_main_returns_2_when_health_collection_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        smoke,
        "_resolve_requested_window",
        lambda **kwargs: (date(2026, 4, 8), date(2026, 4, 8)),
    )
    monkeypatch.setattr(
        smoke,
        "collect_auto_brief_health_inputs",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("db unavailable")),
    )

    args = SimpleNamespace(
        window="today",
        start_date=None,
        end_date=None,
        sample_size=5,
        max_unresolved=0,
        max_failed_locked=0,
        max_pending_wait=None,
        max_insufficient_ratio=None,
        min_selected_targets=0,
        fail_on_missing_report=False,
        base_url="http://127.0.0.1:8001",
        internal_api_key="secret",
        smoke_timeout=15.0,
        skip_warm_smoke=True,
        output=None,
    )

    exit_code = asyncio.run(smoke.async_main(args))

    assert exit_code == 2
