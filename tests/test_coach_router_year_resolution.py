from datetime import date, datetime
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.routers.coach import AnalyzeRequest, _is_valid_analysis_year, _resolve_target_year


def _make_pool(fetchone_result):
    conn = MagicMock()
    execute_result = MagicMock()
    execute_result.fetchone.return_value = fetchone_result
    conn.execute.return_value = execute_result

    context = MagicMock()
    context.__enter__.return_value = conn
    context.__exit__.return_value = False

    pool = MagicMock()
    pool.connection.return_value = context
    return pool, conn


def test_resolve_target_year_prefers_season_year():
    pool, _ = _make_pool((2022,))
    payload = AnalyzeRequest(
        home_team_id="SSG",
        league_context={"season_year": 2024, "season": 20255},
    )

    year, source = _resolve_target_year(payload, pool)

    assert year == 2024
    assert source == "league_context.season_year"
    assert pool.connection.call_count == 0


def test_resolve_target_year_from_season_id_lookup():
    pool, conn = _make_pool((2022,))
    payload = AnalyzeRequest(
        home_team_id="SSG",
        league_context={"season": 20225},
    )

    year, source = _resolve_target_year(payload, pool)

    assert year == 2022
    assert source == "league_context.season->kbo_seasons"
    assert pool.connection.call_count == 1
    conn.execute.assert_called_once()


def test_resolve_target_year_falls_back_to_game_date_context():
    pool, _ = _make_pool(None)
    payload = AnalyzeRequest(
        home_team_id="SSG",
        league_context={"season": "invalid", "game_date": "2023-08-11"},
    )

    year, source = _resolve_target_year(payload, pool)

    assert year == 2023
    assert source == "game_date"


def test_resolve_target_year_falls_back_to_game_id_lookup():
    pool, _ = _make_pool((date(2020, 4, 1),))
    payload = AnalyzeRequest(
        home_team_id="SSG",
        league_context={"season": "invalid"},
        game_id="20200401OBLG0",
    )

    year, source = _resolve_target_year(payload, pool)

    assert year == 2020
    assert source == "game_date"


def test_resolve_target_year_rejects_invalid_season_year():
    pool, _ = _make_pool(None)
    payload = AnalyzeRequest(
        home_team_id="SSG",
        league_context={"season_year": 266},
    )

    with pytest.raises(HTTPException) as exc:
        _resolve_target_year(payload, pool)

    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid_season_year_for_analysis"


def test_resolve_target_year_raises_when_unresolvable():
    pool, _ = _make_pool(None)
    payload = AnalyzeRequest(home_team_id="SSG", league_context={})

    with pytest.raises(HTTPException) as exc:
        _resolve_target_year(payload, pool)

    assert exc.value.status_code == 400
    assert exc.value.detail == "unable_to_resolve_analysis_year"


def test_analysis_year_range_guard():
    assert _is_valid_analysis_year(1982)
    assert _is_valid_analysis_year(datetime.now().year + 1)
    assert not _is_valid_analysis_year(1981)
