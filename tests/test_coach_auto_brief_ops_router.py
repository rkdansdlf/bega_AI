from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import coach_auto_brief_ops
from scripts.batch_coach_matchup_cache import MatchupTarget


def _build_client() -> TestClient:
    test_app = FastAPI()
    test_app.include_router(coach_auto_brief_ops.router)
    return TestClient(test_app)


def _build_target(
    *, cache_key: str, game_date: str, stage_label: str = "REGULAR"
) -> MatchupTarget:
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
        stage_label=stage_label,
        series_game_no=None,
        game_status_bucket="SCHEDULED",
        starter_signature="starter:none",
        lineup_signature="lineup:none",
        request_focus=["recent_form"],
        request_mode="auto_brief",
        question_override=None,
    )


def test_get_auto_brief_health_requires_internal_token(monkeypatch) -> None:
    settings = type("Settings", (), {"resolved_ai_internal_token": "secret-token"})()
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: settings,
    )
    client = _build_client()

    response = client.get("/ai/coach/auto-brief/ops/health")

    assert response.status_code == 401


def test_load_latest_auto_brief_report_reads_newest_matching_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    older = tmp_path / "coach_auto_brief_old.json"
    latest = tmp_path / "coach_auto_brief_latest.json"
    ignored = tmp_path / "manual_detail.json"
    ignored_verify = tmp_path / "auto_brief_verify_final.json"

    older.write_text(
        json.dumps(
            {
                "summary": {
                    "date_window": "2026-04-07",
                    "unresolved_count": 2,
                    "completed_count": 3,
                    "cache_state_breakdown": {"FAILED_LOCKED": 1},
                    "data_quality_breakdown": {"insufficient": 1},
                },
                "options": {"mode": "auto_brief"},
                "run_finished_at": "2026-04-07T10:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    ignored.write_text(
        json.dumps({"summary": {}, "options": {"mode": "manual_detail"}}),
        encoding="utf-8",
    )
    ignored_verify.write_text(
        json.dumps(
            {
                "summary": {
                    "unresolved_count": 0,
                    "completed_count": 0,
                },
                "run_finished_at": "2026-04-08T09:30:00Z",
            }
        ),
        encoding="utf-8",
    )
    latest.write_text(
        json.dumps(
            {
                "summary": {
                    "date_window": "2026-04-08:2026-04-09",
                    "unresolved_count": 1,
                    "completed_count": 5,
                    "cache_state_breakdown": {"PENDING_WAIT": 1},
                    "data_quality_breakdown": {"partial": 1},
                },
                "options": {"mode": "auto_brief"},
                "run_started_at": "2026-04-08T08:55:00Z",
                "run_finished_at": "2026-04-08T09:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(coach_auto_brief_ops, "REPORTS_ROOT", tmp_path)

    report = coach_auto_brief_ops._load_latest_auto_brief_report()

    assert report is not None
    assert report.path.endswith("coach_auto_brief_latest.json")
    assert report.unresolved_count == 1
    assert report.cache_state_breakdown["PENDING_WAIT"] == 1


def test_build_auto_brief_health_snapshot_aggregates_results(monkeypatch) -> None:
    targets = [
        _build_target(cache_key="completed", game_date="2026-04-08"),
        _build_target(cache_key="locked", game_date="2026-04-08"),
    ]
    results = [
        {
            "cache_key": "completed",
            "game_id": "completed-game",
            "home_team_id": "LG",
            "away_team_id": "KT",
            "status": "skipped",
            "reason": "cache_hit",
            "meta": {
                "cache_state": "COMPLETED",
                "cached": True,
                "data_quality": "grounded",
                "generation_mode": "deterministic_auto",
            },
        },
        {
            "cache_key": "locked",
            "game_id": "locked-game",
            "home_team_id": "LG",
            "away_team_id": "KT",
            "status": "failed",
            "reason": "failed_locked",
            "meta": {
                "cache_state": "FAILED_LOCKED",
                "data_quality": "insufficient",
                "failure_class": "locked_terminal",
            },
        },
    ]

    monkeypatch.setattr(
        coach_auto_brief_ops, "_load_target_pool_for_years", lambda years: targets
    )
    monkeypatch.setattr(
        coach_auto_brief_ops,
        "collect_cache_verification_results",
        lambda selected: results,
    )
    monkeypatch.setattr(
        coach_auto_brief_ops, "_load_latest_auto_brief_report", lambda: None
    )

    snapshot = coach_auto_brief_ops.build_auto_brief_health_snapshot(
        window="today",
        start=date(2026, 4, 8),
        end=date(2026, 4, 8),
        sample_size=5,
    )

    assert snapshot.summary.selected_target_count == 2
    assert snapshot.summary.completed_count == 1
    assert snapshot.summary.unresolved_count == 1
    assert snapshot.summary.cache_state_breakdown["FAILED_LOCKED"] == 1
    assert snapshot.gate.verdict == "FAIL"
    assert snapshot.gate.failed_locked_count == 1
    assert (
        "latest auto_brief report missing for selected window"
        in snapshot.gate.checks.failed
    )
    assert snapshot.unresolved_targets[0].cache_state == "FAILED_LOCKED"
    assert "batch_coach_auto_brief.py" in snapshot.recommended_command


def test_build_auto_brief_ops_gate_warns_for_empty_selected_window() -> None:
    summary = coach_auto_brief_ops.CoachAutoBriefOpsSummary(
        loaded_target_count=4,
        selected_target_count=0,
        generated_success_count=0,
        cache_hit_count=0,
        in_progress_count=0,
        failed_count=0,
        unresolved_count=0,
        completed_count=0,
        cache_state_breakdown={},
        data_quality_breakdown={},
    )

    gate = coach_auto_brief_ops.build_auto_brief_ops_gate(
        summary=summary,
        latest_report=None,
    )

    assert gate.verdict == "WARN"
    assert gate.checks.failed == []
    assert "selected target window is empty" in gate.checks.warnings[0]
