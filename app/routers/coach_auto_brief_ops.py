"""Prediction auto_brief 운영 상태 요약 라우터."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..internal_auth import require_ai_internal_token
from scripts.batch_coach_auto_brief import (
    AUTO_BRIEF_MODE,
    _collect_report_breakdowns,
    _extract_brief_headline,
    _extract_data_quality,
    _filter_targets_by_date_window,
    _format_date_window,
    _resolve_cache_state_label,
)
from scripts.batch_coach_matchup_cache import (
    AUTO_BRIEF_FOCUS,
    MatchupTarget,
    collect_cache_verification_results,
    load_targets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _detect_workspace_root(start: Path) -> Path:
    current = start.resolve()
    while True:
        if (current / "docker-compose.yml").exists() or (
            current / ".env.prod"
        ).exists():
            return current
        if (current / "scripts").exists() and (current / "tests").exists():
            return current
        if current.parent == current:
            return start.resolve()
        current = current.parent


WORKSPACE_ROOT = _detect_workspace_root(PROJECT_ROOT)
REPORTS_ROOT = WORKSPACE_ROOT / "reports"
RUNBOOK_PATH = "task/operations/coach-auto-brief-prewarm-runbook.md"
CoachAutoBriefOpsWindow = Literal["today", "tomorrow", "custom"]
UNRESOLVED_STATE_PRIORITY = {
    "FAILED_LOCKED": 0,
    "PENDING_WAIT": 1,
    "PENDING": 2,
    "FAILED": 3,
    "MISSING": 4,
    "UNAVAILABLE": 5,
    "UNKNOWN": 6,
}
DEFAULT_GATE_MAX_UNRESOLVED = 2
DEFAULT_GATE_MAX_FAILED_LOCKED = 0
DEFAULT_GATE_MAX_PENDING_WAIT = 2
DEFAULT_GATE_MAX_INSUFFICIENT_RATIO = 0.4
DEFAULT_GATE_MIN_SELECTED_TARGETS = 0
DEFAULT_GATE_FAIL_ON_MISSING_REPORT = True

router = APIRouter(prefix="/ai/coach/auto-brief/ops", tags=["coach-auto-brief-ops"])


@dataclass(frozen=True)
class CoachAutoBriefHealthInputs:
    years: list[int]
    target_pool: list[MatchupTarget]
    selected_targets: list[MatchupTarget]
    results: list[dict]


class CoachAutoBriefOpsSummary(BaseModel):
    loaded_target_count: int
    selected_target_count: int
    generated_success_count: int
    cache_hit_count: int
    in_progress_count: int
    failed_count: int
    unresolved_count: int
    completed_count: int
    cache_state_breakdown: dict[str, int] = Field(default_factory=dict)
    data_quality_breakdown: dict[str, int] = Field(default_factory=dict)


@dataclass(frozen=True)
class CoachAutoBriefOpsGateThresholdsInput:
    max_unresolved: int = DEFAULT_GATE_MAX_UNRESOLVED
    max_failed_locked: int = DEFAULT_GATE_MAX_FAILED_LOCKED
    max_pending_wait: int | None = DEFAULT_GATE_MAX_PENDING_WAIT
    max_insufficient_ratio: float | None = DEFAULT_GATE_MAX_INSUFFICIENT_RATIO
    min_selected_targets: int = DEFAULT_GATE_MIN_SELECTED_TARGETS
    fail_on_missing_report: bool = DEFAULT_GATE_FAIL_ON_MISSING_REPORT


class CoachAutoBriefOpsGateThresholds(BaseModel):
    max_unresolved: int
    max_failed_locked: int
    max_pending_wait: int | None = None
    max_insufficient_ratio: float | None = None
    min_selected_targets: int
    fail_on_missing_report: bool


class CoachAutoBriefOpsGateChecks(BaseModel):
    failed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CoachAutoBriefOpsGate(BaseModel):
    verdict: Literal["PASS", "WARN", "FAIL"]
    thresholds: CoachAutoBriefOpsGateThresholds
    failed_locked_count: int
    pending_wait_count: int
    insufficient_count: int
    insufficient_ratio: float
    checks: CoachAutoBriefOpsGateChecks = Field(
        default_factory=CoachAutoBriefOpsGateChecks
    )


class CoachAutoBriefOpsLatestReport(BaseModel):
    path: str
    run_started_at: str | None = None
    run_finished_at: str | None = None
    date_window: str | None = None
    unresolved_count: int = 0
    completed_count: int = 0
    cache_state_breakdown: dict[str, int] = Field(default_factory=dict)
    data_quality_breakdown: dict[str, int] = Field(default_factory=dict)


class CoachAutoBriefOpsTargetSample(BaseModel):
    game_id: str
    game_date: str
    away_team_id: str
    home_team_id: str
    stage_label: str
    game_status_bucket: str
    cache_key: str
    cache_state: str
    data_quality: str
    headline: str | None = None
    reason: str | None = None


class CoachAutoBriefOpsHealthResponse(BaseModel):
    window: CoachAutoBriefOpsWindow
    date_window: str
    generated_at_utc: datetime
    runbook_path: str
    recommended_command: str
    summary: CoachAutoBriefOpsSummary
    gate: CoachAutoBriefOpsGate
    unresolved_targets: list[CoachAutoBriefOpsTargetSample] = Field(
        default_factory=list
    )
    latest_report: CoachAutoBriefOpsLatestReport | None = None


def _resolve_requested_window(
    *,
    window: CoachAutoBriefOpsWindow,
    start_date: date | None,
    end_date: date | None,
) -> tuple[date, date]:
    today = date.today()
    if window == "today":
        return today, today
    if window == "tomorrow":
        tomorrow = today + timedelta(days=1)
        return tomorrow, tomorrow
    if start_date is None:
        raise ValueError("custom window requires start_date")

    resolved_end = end_date or start_date
    if resolved_end < start_date:
        raise ValueError("end_date must be later than or equal to start_date")
    return start_date, resolved_end


def _load_target_pool_for_years(years: list[int]) -> list[MatchupTarget]:
    deduped: list[MatchupTarget] = []
    seen_cache_keys: set[str] = set()

    for league_type in ("REGULAR", "POST"):
        league_targets = load_targets(
            years=years,
            league_type=league_type,
            request_focus=list(AUTO_BRIEF_FOCUS),
            request_mode=AUTO_BRIEF_MODE,
            question_override=None,
            offset=0,
            limit=None,
            order="asc",
            status_bucket_filter="ANY",
        )
        for target in league_targets:
            if target.cache_key in seen_cache_keys:
                continue
            seen_cache_keys.add(target.cache_key)
            deduped.append(target)

    return sorted(
        deduped, key=lambda item: (item.game_date, item.game_id, item.cache_key)
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _relative_workspace_path(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def _iter_auto_brief_report_candidates() -> list[Path]:
    if not REPORTS_ROOT.exists():
        return []

    candidates = sorted(
        REPORTS_ROOT.rglob("*auto_brief*.json"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    return [candidate for candidate in candidates if candidate.is_file()]


def _load_latest_auto_brief_report() -> CoachAutoBriefOpsLatestReport | None:
    for path in _iter_auto_brief_report_candidates():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        options = payload.get("options") or {}
        if not isinstance(options, dict):
            continue
        if str(options.get("mode") or "").strip() != AUTO_BRIEF_MODE:
            continue

        summary = payload.get("summary") or {}
        if not isinstance(summary, dict):
            continue
        if not str(summary.get("date_window") or "").strip():
            continue

        return CoachAutoBriefOpsLatestReport(
            path=_relative_workspace_path(path),
            run_started_at=str(payload.get("run_started_at") or "") or None,
            run_finished_at=str(payload.get("run_finished_at") or "") or None,
            date_window=str(summary.get("date_window") or "") or None,
            unresolved_count=int(summary.get("unresolved_count") or 0),
            completed_count=int(summary.get("completed_count") or 0),
            cache_state_breakdown=dict(summary.get("cache_state_breakdown") or {}),
            data_quality_breakdown=dict(summary.get("data_quality_breakdown") or {}),
        )

    return None


def _build_recommended_command(start: date, end: date, years: list[int]) -> str:
    report_suffix = (
        start.isoformat() if start == end else f"{start.isoformat()}_{end.isoformat()}"
    )
    date_window = (
        start.isoformat() if start == end else f"{start.isoformat()}:{end.isoformat()}"
    )
    years_arg = ",".join(str(year) for year in years)
    return (
        "./.venv/bin/python scripts/batch_coach_auto_brief.py "
        f"--years {years_arg} "
        f"--date-window {date_window} "
        "--eligible-only "
        "--prioritize-unresolved "
        f"--quality-report reports/coach_auto_brief_prewarm_{report_suffix}.json"
    )


def _normalize_gate_thresholds(
    thresholds: CoachAutoBriefOpsGateThresholdsInput | None = None,
) -> CoachAutoBriefOpsGateThresholdsInput:
    resolved = thresholds or CoachAutoBriefOpsGateThresholdsInput()
    return CoachAutoBriefOpsGateThresholdsInput(
        max_unresolved=max(0, int(resolved.max_unresolved)),
        max_failed_locked=max(0, int(resolved.max_failed_locked)),
        max_pending_wait=(
            None
            if resolved.max_pending_wait is None
            else max(0, int(resolved.max_pending_wait))
        ),
        max_insufficient_ratio=(
            None
            if resolved.max_insufficient_ratio is None
            else max(0.0, float(resolved.max_insufficient_ratio))
        ),
        min_selected_targets=max(0, int(resolved.min_selected_targets)),
        fail_on_missing_report=bool(resolved.fail_on_missing_report),
    )


def build_auto_brief_ops_gate(
    *,
    summary: CoachAutoBriefOpsSummary,
    latest_report: CoachAutoBriefOpsLatestReport | None,
    thresholds: CoachAutoBriefOpsGateThresholdsInput | None = None,
) -> CoachAutoBriefOpsGate:
    resolved_thresholds = _normalize_gate_thresholds(thresholds)
    failed_locked_count = int(summary.cache_state_breakdown.get("FAILED_LOCKED") or 0)
    pending_wait_count = int(summary.cache_state_breakdown.get("PENDING_WAIT") or 0)
    insufficient_count = int(summary.data_quality_breakdown.get("insufficient") or 0)
    insufficient_ratio = _safe_ratio(insufficient_count, summary.selected_target_count)

    failed_checks: list[str] = []
    warnings: list[str] = []

    if summary.unresolved_count > resolved_thresholds.max_unresolved:
        failed_checks.append(
            f"unresolved_count {summary.unresolved_count} > {resolved_thresholds.max_unresolved}"
        )

    if summary.selected_target_count < resolved_thresholds.min_selected_targets:
        failed_checks.append(
            "selected_target_count "
            f"{summary.selected_target_count} < {resolved_thresholds.min_selected_targets}"
        )

    if failed_locked_count > resolved_thresholds.max_failed_locked:
        failed_checks.append(
            f"failed_locked_count {failed_locked_count} > {resolved_thresholds.max_failed_locked}"
        )

    if (
        resolved_thresholds.max_pending_wait is not None
        and pending_wait_count > resolved_thresholds.max_pending_wait
    ):
        failed_checks.append(
            f"pending_wait_count {pending_wait_count} > {resolved_thresholds.max_pending_wait}"
        )

    if (
        resolved_thresholds.max_insufficient_ratio is not None
        and insufficient_ratio > resolved_thresholds.max_insufficient_ratio
    ):
        failed_checks.append(
            "insufficient_ratio "
            f"{insufficient_ratio:.3f} > {resolved_thresholds.max_insufficient_ratio:.3f}"
        )

    if summary.selected_target_count == 0:
        warnings.append("selected target window is empty (off-day or data not loaded)")
    elif latest_report is None:
        if resolved_thresholds.fail_on_missing_report:
            failed_checks.append("latest auto_brief report missing for selected window")
        else:
            warnings.append("latest auto_brief report not found")

    verdict: Literal["PASS", "WARN", "FAIL"]
    if failed_checks:
        verdict = "FAIL"
    elif warnings:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return CoachAutoBriefOpsGate(
        verdict=verdict,
        thresholds=CoachAutoBriefOpsGateThresholds(**asdict(resolved_thresholds)),
        failed_locked_count=failed_locked_count,
        pending_wait_count=pending_wait_count,
        insufficient_count=insufficient_count,
        insufficient_ratio=insufficient_ratio,
        checks=CoachAutoBriefOpsGateChecks(
            failed=failed_checks,
            warnings=warnings,
        ),
    )


def _build_unresolved_targets(
    *,
    targets: list[MatchupTarget],
    results: list[dict],
    sample_size: int,
) -> list[CoachAutoBriefOpsTargetSample]:
    ranked_samples: list[tuple[int, str, str, CoachAutoBriefOpsTargetSample]] = []

    for target, result in zip(targets, results):
        cache_state = _resolve_cache_state_label(result)
        if cache_state == "COMPLETED":
            continue

        ranked_samples.append(
            (
                UNRESOLVED_STATE_PRIORITY.get(cache_state, 99),
                target.game_date,
                target.game_id,
                CoachAutoBriefOpsTargetSample(
                    game_id=target.game_id,
                    game_date=target.game_date,
                    away_team_id=target.away_team_id,
                    home_team_id=target.home_team_id,
                    stage_label=target.stage_label,
                    game_status_bucket=target.game_status_bucket,
                    cache_key=target.cache_key,
                    cache_state=cache_state,
                    data_quality=_extract_data_quality(result),
                    headline=_extract_brief_headline(result),
                    reason=str(result.get("reason") or "") or None,
                ),
            )
        )

    ranked_samples.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in ranked_samples[:sample_size]]


def _build_summary(
    *,
    target_pool: list[MatchupTarget],
    selected_targets: list[MatchupTarget],
    results: list[dict],
) -> CoachAutoBriefOpsSummary:
    breakdowns = _collect_report_breakdowns(results)
    generated_success_count = sum(
        1 for item in results if item.get("status") == "generated"
    )
    cache_hit_count = sum(1 for item in results if item.get("status") == "skipped")
    in_progress_count = sum(
        1 for item in results if item.get("status") == "in_progress"
    )
    failed_count = sum(1 for item in results if item.get("status") == "failed")

    return CoachAutoBriefOpsSummary(
        loaded_target_count=len(target_pool),
        selected_target_count=len(selected_targets),
        generated_success_count=generated_success_count,
        cache_hit_count=cache_hit_count,
        in_progress_count=in_progress_count,
        failed_count=failed_count,
        unresolved_count=int(breakdowns.get("unresolved_count") or 0),
        completed_count=int(breakdowns.get("completed_count") or 0),
        cache_state_breakdown=dict(breakdowns.get("cache_state_breakdown") or {}),
        data_quality_breakdown=dict(breakdowns.get("data_quality_breakdown") or {}),
    )


def collect_auto_brief_health_inputs(
    *,
    start: date,
    end: date,
) -> CoachAutoBriefHealthInputs:
    years = list(range(start.year, end.year + 1))
    target_pool = _load_target_pool_for_years(years)
    selected_targets = _filter_targets_by_date_window(target_pool, (start, end))
    results = collect_cache_verification_results(selected_targets)
    return CoachAutoBriefHealthInputs(
        years=years,
        target_pool=target_pool,
        selected_targets=selected_targets,
        results=results,
    )


def _build_auto_brief_health_snapshot_from_inputs(
    *,
    window: CoachAutoBriefOpsWindow,
    start: date,
    end: date,
    sample_size: int,
    inputs: CoachAutoBriefHealthInputs,
) -> CoachAutoBriefOpsHealthResponse:
    summary = _build_summary(
        target_pool=inputs.target_pool,
        selected_targets=inputs.selected_targets,
        results=inputs.results,
    )
    latest_report = _load_latest_auto_brief_report()
    return CoachAutoBriefOpsHealthResponse(
        window=window,
        date_window=_format_date_window((start, end)) or start.isoformat(),
        generated_at_utc=datetime.now(timezone.utc),
        runbook_path=RUNBOOK_PATH,
        recommended_command=_build_recommended_command(start, end, inputs.years),
        summary=summary,
        gate=build_auto_brief_ops_gate(
            summary=summary,
            latest_report=latest_report,
        ),
        unresolved_targets=_build_unresolved_targets(
            targets=inputs.selected_targets,
            results=inputs.results,
            sample_size=sample_size,
        ),
        latest_report=latest_report,
    )


def build_auto_brief_health_snapshot(
    *,
    window: CoachAutoBriefOpsWindow,
    start: date,
    end: date,
    sample_size: int,
) -> CoachAutoBriefOpsHealthResponse:
    inputs = collect_auto_brief_health_inputs(start=start, end=end)
    return _build_auto_brief_health_snapshot_from_inputs(
        window=window,
        start=start,
        end=end,
        sample_size=sample_size,
        inputs=inputs,
    )


@router.get("/health", response_model=CoachAutoBriefOpsHealthResponse)
async def get_auto_brief_health(
    window: CoachAutoBriefOpsWindow = Query(default="today"),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    sample_size: int = Query(default=5, ge=1, le=20),
    _: None = Depends(require_ai_internal_token),
) -> CoachAutoBriefOpsHealthResponse:
    try:
        start, end = _resolve_requested_window(
            window=window,
            start_date=start_date,
            end_date=end_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        return build_auto_brief_health_snapshot(
            window=window,
            start=start,
            end=end,
            sample_size=sample_size,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"auto brief ops health unavailable: {exc}",
        ) from exc
