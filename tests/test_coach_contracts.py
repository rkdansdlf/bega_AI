"""
Coach 기능 테스트 모듈.

Fast Path, 캐싱, 응답 검증기 등을 테스트합니다.
"""

import pytest
import json
import asyncio
import logging
import os
import re
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from pydantic import ValidationError

from tests.coach_test_support import (
    _build_game_evidence,
    _collect_sse_text,
    _extract_sse_meta_events,
    _install_coach_endpoint_cache_hit,
)

def test_assess_game_evidence_requires_summary_for_review_only():
    from app.routers import coach as coach_router

    evidence = _build_game_evidence(summary_items=[])

    review = coach_router.assess_game_evidence(
        evidence,
        analysis_type=coach_router.COACH_ANALYSIS_TYPE_REVIEW,
    )
    preview = coach_router.assess_game_evidence(
        evidence,
        analysis_type=coach_router.COACH_ANALYSIS_TYPE_PREVIEW,
    )

    assert "missing_summary" in review.root_causes
    assert "missing_summary" not in preview.root_causes
    assert preview.expected_data_quality == "grounded"


def test_preview_partial_data_does_not_require_manual_baseball_data():
    from app.routers import coach as coach_router

    evidence = _build_game_evidence(
        home_pitcher=None,
        away_pitcher=None,
        lineup_announced=False,
        home_lineup=[],
        away_lineup=[],
        summary_items=[],
    )
    preview_assessment = coach_router.assess_game_evidence(
        evidence,
        analysis_type=coach_router.COACH_ANALYSIS_TYPE_PREVIEW,
    )
    review_assessment = coach_router.assess_game_evidence(
        evidence,
        analysis_type=coach_router.COACH_ANALYSIS_TYPE_REVIEW,
    )
    payload = SimpleNamespace(
        game_id=evidence.game_id,
        league_context={"game_date": evidence.game_date},
    )

    preview_request = asyncio.run(coach_router._build_manual_data_request(
        object(),
        payload,
        evidence,
        preview_assessment,
        analysis_type=coach_router.COACH_ANALYSIS_TYPE_PREVIEW,
    ))
    review_request = asyncio.run(coach_router._build_manual_data_request(
        object(),
        payload,
        evidence,
        review_assessment,
        analysis_type=coach_router.COACH_ANALYSIS_TYPE_REVIEW,
    ))

    assert preview_request is None
    assert review_request is not None
    assert review_request["code"] == coach_router.MANUAL_BASEBALL_DATA_REQUIRED_CODE


def test_meta_defaults_include_analysis_type():
    from app.routers import coach as coach_router

    meta = coach_router._build_meta_payload_defaults(
        generation_mode="deterministic_preview",
        data_quality="partial",
        used_evidence=["game"],
        analysis_type=coach_router.COACH_ANALYSIS_TYPE_PREVIEW,
        game_status_bucket="SCHEDULED",
    )

    assert meta["analysis_type"] == "game_preview"
    assert meta["generation_mode"] == "deterministic_preview"


def test_llm_skip_reason_for_cache_hit():
    from app.routers import coach as coach_router

    assert (
        coach_router._resolve_llm_skip_reason(
            request_mode=coach_router.COACH_REQUEST_MODE_MANUAL,
            generation_mode="llm_manual",
            cache_state="HIT",
            cached=True,
            in_progress=False,
            manual_data_required=False,
        )
        == "cache_hit"
    )


def test_llm_skip_reason_for_auto_brief_pending():
    from app.routers import coach as coach_router

    assert (
        coach_router._resolve_llm_skip_reason(
            request_mode=coach_router.COACH_REQUEST_MODE_AUTO,
            generation_mode="evidence_fallback",
            cache_state="PENDING_WAIT",
            cached=False,
            in_progress=True,
            manual_data_required=False,
        )
        == "pending_wait"
    )


def test_llm_skip_reason_absent_for_successful_manual_llm():
    from app.routers import coach as coach_router

    assert (
        coach_router._resolve_llm_skip_reason(
            request_mode=coach_router.COACH_REQUEST_MODE_MANUAL,
            generation_mode="llm_manual",
            cache_state="COMPLETED",
            cached=False,
            in_progress=False,
            manual_data_required=False,
        )
        is None
    )


def test_llm_skip_metric_increments_from_stream_meta():
    from app.observability.metrics import AI_COACH_LLM_SKIP_TOTAL
    from app.routers import coach as coach_router

    labels = {
        "reason": "pending_wait",
        "request_mode": "auto_brief",
        "analysis_type": "game_preview",
    }
    metric = AI_COACH_LLM_SKIP_TOTAL.labels(**labels)
    before = metric._value.get()

    coach_router._log_coach_stream_meta(
        {
            "request_mode": "auto_brief",
            "analysis_type": "game_preview",
            "cache_state": "PENDING_WAIT",
            "validation_status": "success",
            "generation_mode": "evidence_fallback",
            "llm_skip_reason": "pending_wait",
            "cached": False,
            "in_progress": True,
            "data_quality": "sufficient",
            "supported_fact_count": 0,
        },
        game_id="20260405LGKT0",
    )

    assert metric._value.get() == before + 1


def test_coach_llm_smoke_script_dry_run_does_not_print_secret():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "coach_llm_smoke.py"

    assert script.exists()

    result = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "mode=dry_run" in result.stdout
    assert "openrouter_api_key=" in result.stdout
    assert "sk-" not in result.stdout
    assert "Bearer" not in result.stdout


def test_coach_llm_smoke_script_prints_resolved_models_for_dry_run(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "coach_llm_smoke.py"
    env = os.environ.copy()
    env.update(
        {
            "COACH_OPENROUTER_MODEL": "openrouter/auto",
            "COACH_OPENROUTER_FALLBACK_MODELS": "",
            "OPENROUTER_API_KEY": "sk-test-secret",
        }
    )

    result = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "coach_openrouter_models=openrouter/free" in result.stdout
    assert "openrouter/auto" not in result.stdout
    assert "sk-test-secret" not in result.stdout


def test_coach_llm_smoke_response_validator_requires_expected_json():
    from scripts import coach_llm_smoke

    ok, error = coach_llm_smoke._validate_smoke_response(
        '{"ok": true, "source": "coach_smoke"}'
    )
    assert ok is True
    assert error is None

    ok, error = coach_llm_smoke._validate_smoke_response("not json")
    assert ok is False
    assert error == "invalid_json"

    ok, error = coach_llm_smoke._validate_smoke_response(
        '{"ok": true, "source": "wrong"}'
    )
    assert ok is False
    assert error == "unexpected_response_contract"


@pytest.mark.asyncio
async def test_endpoint_stream_meta_preview_cache_hit_does_not_require_manual_data(monkeypatch):
    from app.routers import coach

    evidence = _build_game_evidence(
        game_row_found=True,
        season_year=2026,
        season_id=266,
        game_id="20260405LGKT0",
        game_date="2026-04-05",
        game_status="SCHEDULED",
        game_status_bucket="SCHEDULED",
        league_type_code=0,
        home_score=None,
        away_score=None,
        summary_items=[],
    )
    agent = _install_coach_endpoint_cache_hit(monkeypatch, evidence)

    response = await coach.analyze_team(
        coach.AnalyzeRequest(
            home_team_id="LG",
            away_team_id="KT",
            game_id="20260405LGKT0",
            request_mode=coach.COACH_REQUEST_MODE_MANUAL,
            analysis_type=coach.COACH_ANALYSIS_TYPE_PREVIEW,
            league_context={
                "season": 266,
                "season_year": 2026,
                "game_date": "2026-04-05",
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

    assert "MANUAL_BASEBALL_DATA_REQUIRED" not in sse_text
    assert meta_events
    assert meta_events[-1]["analysis_type"] == "game_preview"
    assert meta_events[-1]["llm_skip_reason"] == "cache_hit"
    assert meta_events[-1]["cached"] is True


@pytest.mark.asyncio
async def test_endpoint_stream_meta_completed_review_cache_hit(monkeypatch):
    from app.routers import coach

    evidence = _build_game_evidence(
        game_row_found=True,
        season_year=2026,
        season_id=266,
        game_id="20260405LGKT0",
        game_date="2026-04-05",
        game_status="COMPLETED",
        game_status_bucket="COMPLETED",
        league_type_code=0,
        home_score=3,
        away_score=2,
        winning_team_code="LG",
        winning_team_name="LG 트윈스",
        summary_items=["9회말 끝내기 안타"],
    )
    agent = _install_coach_endpoint_cache_hit(monkeypatch, evidence)

    response = await coach.analyze_team(
        coach.AnalyzeRequest(
            home_team_id="LG",
            away_team_id="KT",
            game_id="20260405LGKT0",
            request_mode=coach.COACH_REQUEST_MODE_MANUAL,
            analysis_type=coach.COACH_ANALYSIS_TYPE_REVIEW,
            league_context={
                "season": 266,
                "season_year": 2026,
                "game_date": "2026-04-05",
                "league_type": "REGULAR",
                "league_type_code": 0,
            },
        ),
        agent,
        None,
        None,
    )

    meta_events = _extract_sse_meta_events(await _collect_sse_text(response))

    assert meta_events
    assert meta_events[-1]["analysis_type"] == "game_review"
    assert meta_events[-1]["llm_skip_reason"] == "cache_hit"
    assert meta_events[-1]["structured_response"]["analysis_type"] == "game_review"


@pytest.mark.asyncio
async def test_endpoint_stream_v2_wraps_cached_coach_events(monkeypatch):
    from app.routers import coach

    evidence = _build_game_evidence(
        game_row_found=True,
        season_year=2026,
        season_id=266,
        game_id="20260405LGKT0",
        game_date="2026-04-05",
        game_status="COMPLETED",
        game_status_bucket="COMPLETED",
        league_type_code=0,
        home_score=3,
        away_score=2,
        winning_team_code="LG",
        winning_team_name="LG 트윈스",
        summary_items=["9회말 끝내기 안타"],
    )
    agent = _install_coach_endpoint_cache_hit(monkeypatch, evidence)

    response = await coach.analyze_team(
        coach.AnalyzeRequest(
            home_team_id="LG",
            away_team_id="KT",
            game_id="20260405LGKT0",
            request_mode=coach.COACH_REQUEST_MODE_MANUAL,
            analysis_type=coach.COACH_ANALYSIS_TYPE_REVIEW,
            league_context={
                "season": 266,
                "season_year": 2026,
                "game_date": "2026-04-05",
                "league_type": "REGULAR",
                "league_type_code": 0,
            },
        ),
        agent,
        None,
        None,
        event_version_header="2",
    )

    sse_text = await _collect_sse_text(response)
    assert response.headers["X-AI-Event-Version"] == "2"
    assert "event: coach.status" in sse_text
    assert "event: coach.message.delta" in sse_text
    assert "event: coach.meta" in sse_text
    assert "event: stream.done" in sse_text
    assert '"version":2' in sse_text
    assert "data: [DONE]" not in sse_text
    assert '"analysis_type":"game_review"' in sse_text
    v2_payloads = [
        json.loads(line.split(":", 1)[1].strip())
        for line in sse_text.splitlines()
        if line.startswith("data: {")
    ]
    coach_meta = next(
        event for event in v2_payloads if event.get("type") == "coach.meta"
    )
    assert "analysisType" not in coach_meta["data"]
    assert "analysisType" not in coach_meta["data"]["structured_response"]


@pytest.mark.asyncio
async def test_endpoint_stream_missing_version_header_preserves_v1(monkeypatch):
    from app.routers import coach

    evidence = _build_game_evidence(
        game_row_found=True,
        season_year=2026,
        season_id=266,
        game_id="20260405LGKT0",
        game_date="2026-04-05",
        game_status="SCHEDULED",
        game_status_bucket="SCHEDULED",
        league_type_code=0,
        home_score=None,
        away_score=None,
        summary_items=[],
    )
    agent = _install_coach_endpoint_cache_hit(monkeypatch, evidence)

    response = await coach.analyze_team(
        coach.AnalyzeRequest(
            home_team_id="LG",
            away_team_id="KT",
            game_id="20260405LGKT0",
            request_mode=coach.COACH_REQUEST_MODE_MANUAL,
            analysis_type=coach.COACH_ANALYSIS_TYPE_PREVIEW,
            league_context={"season_year": 2026, "game_date": "2026-04-05"},
        ),
        agent,
        None,
        None,
    )

    sse_text = await _collect_sse_text(response)
    assert response.headers["X-AI-Event-Version"] == "1"
    assert "event: meta" in sse_text
    assert "data: [DONE]" in sse_text
    assert '"version":2' not in sse_text


@pytest.mark.asyncio
async def test_endpoint_stream_rejects_unsupported_version_before_work() -> None:
    from app.routers import coach
    from app.streaming.http_errors import AiStreamHttpException

    with pytest.raises(AiStreamHttpException) as raised:
        await coach.analyze_team(
            coach.AnalyzeRequest(home_team_id="LG"),
            object(),
            None,
            None,
            event_version_header="3",
        )

    assert raised.value.status_code == 406
    assert raised.value.error.model_dump(mode="json") == {
        "code": "AI_EVENT_VERSION_UNSUPPORTED",
        "message": "지원하지 않는 AI 이벤트 버전입니다.",
        "detail": None,
        "retryable": False,
        "retry_after_seconds": None,
        "supported_versions": ["1", "2"],
    }
    assert not isinstance(raised.value.error.detail, dict)


# ============================================================
# Coach Validator Tests
# ============================================================
