from __future__ import annotations

from scripts.batch_coach_auto_brief import (
    _collect_report_breakdowns,
    _filter_targets_by_date_window,
    _filter_targets_by_eligible_cache_state,
    _parse_date_window,
    _prioritize_unresolved_targets,
)
from scripts.batch_coach_matchup_cache import MatchupTarget


def _build_target(
    *,
    cache_key: str,
    game_date: str,
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
        stage_label="REGULAR",
        series_game_no=None,
        game_status_bucket="SCHEDULED",
        starter_signature="starter:none",
        lineup_signature="lineup:none",
        request_focus=["recent_form"],
        request_mode="auto_brief",
        question_override=None,
    )


def test_parse_date_window_supports_single_day_and_range() -> None:
    single_day = _parse_date_window("2026-04-08")
    assert single_day is not None
    assert single_day[0].isoformat() == "2026-04-08"
    assert single_day[1].isoformat() == "2026-04-08"

    day_range = _parse_date_window("2026-04-08:2026-04-09")
    assert day_range is not None
    assert day_range[0].isoformat() == "2026-04-08"
    assert day_range[1].isoformat() == "2026-04-09"


def test_filter_targets_by_date_window_uses_inclusive_bounds() -> None:
    targets = [
      _build_target(cache_key="a", game_date="2026-04-07"),
      _build_target(cache_key="b", game_date="2026-04-08"),
      _build_target(cache_key="c", game_date="2026-04-09"),
    ]

    filtered = _filter_targets_by_date_window(
        targets,
        _parse_date_window("2026-04-08:2026-04-09"),
    )

    assert [target.cache_key for target in filtered] == ["b", "c"]


def test_eligible_only_and_prioritize_unresolved_prefer_non_completed_targets() -> None:
    targets = [
        _build_target(cache_key="completed", game_date="2026-04-08"),
        _build_target(cache_key="missing", game_date="2026-04-08"),
        _build_target(cache_key="pending", game_date="2026-04-08"),
    ]
    state_by_cache_key = {
        "completed": "COMPLETED",
        "missing": "MISSING",
        "pending": "PENDING",
    }

    eligible = _filter_targets_by_eligible_cache_state(
        targets,
        state_by_cache_key=state_by_cache_key,
    )
    prioritized = _prioritize_unresolved_targets(
        targets,
        state_by_cache_key=state_by_cache_key,
    )

    assert [target.cache_key for target in eligible] == ["missing", "pending"]
    assert [target.cache_key for target in prioritized] == [
        "missing",
        "pending",
        "completed",
    ]


def test_collect_report_breakdowns_counts_cache_states_and_data_quality() -> None:
    breakdowns = _collect_report_breakdowns(
        [
            {
                "status": "generated",
                "meta": {"cache_state": "COMPLETED", "data_quality": "grounded"},
            },
            {
                "status": "in_progress",
                "meta": {"cache_state": "PENDING_WAIT", "data_quality": "partial"},
            },
            {
                "status": "failed",
                "reason": "failed_locked",
                "meta": {"cache_state": "FAILED_LOCKED", "data_quality": "insufficient"},
            },
        ]
    )

    assert breakdowns["completed_count"] == 1
    assert breakdowns["unresolved_count"] == 2
    assert breakdowns["cache_state_breakdown"]["COMPLETED"] == 1
    assert breakdowns["cache_state_breakdown"]["PENDING_WAIT"] == 1
    assert breakdowns["cache_state_breakdown"]["FAILED_LOCKED"] == 1
    assert breakdowns["data_quality_breakdown"]["grounded"] == 1
    assert breakdowns["data_quality_breakdown"]["partial"] == 1
    assert breakdowns["data_quality_breakdown"]["insufficient"] == 1
