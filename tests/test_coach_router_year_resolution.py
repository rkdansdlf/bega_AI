import asyncio
from datetime import date, datetime
import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

try:
    import sse_starlette.sse  # noqa: F401
except ModuleNotFoundError:
    sse_starlette_module = types.ModuleType("sse_starlette")
    sse_module = types.ModuleType("sse_starlette.sse")

    class _DummyEventSourceResponse:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    sse_module.EventSourceResponse = _DummyEventSourceResponse
    sse_starlette_module.sse = sse_module
    sys.modules["sse_starlette"] = sse_starlette_module
    sys.modules["sse_starlette.sse"] = sse_module

from app.routers.coach import (
    COACH_INTERNAL_ERROR_CODE,
    AnalyzeRequest,
    _is_valid_analysis_year,
    _resolve_target_year,
    analyze_team,
)
from app.config import get_settings


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


def test_analyze_team_preserves_http_exception_status(monkeypatch):
    payload = AnalyzeRequest(
        home_team_id="SSG",
        league_context={"season_year": 266},
    )
    agent = MagicMock()
    agent._convert_team_id_to_name.return_value = "SSG"

    dummy_pool = MagicMock()
    monkeypatch.setattr("app.routers.coach.get_connection_pool", lambda: dummy_pool)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(analyze_team(payload, agent=agent, _=None))

    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid_season_year_for_analysis"


def test_analyze_team_masks_internal_exception_with_fixed_error_code(monkeypatch):
    payload = AnalyzeRequest(home_team_id="SSG", league_context={"season_year": 2024})
    agent = MagicMock()
    agent._convert_team_id_to_name.return_value = "SSG"

    monkeypatch.setattr(
        "app.routers.coach.get_connection_pool",
        lambda: (_ for _ in ()).throw(RuntimeError("db internal details")),
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(analyze_team(payload, agent=agent, _=None))

    assert exc.value.status_code == 500
    assert exc.value.detail == COACH_INTERNAL_ERROR_CODE


def test_analyze_team_configures_sse_ping(monkeypatch):
    payload = AnalyzeRequest(
        home_team_id="SSG",
        away_team_id="LG",
        request_mode="manual_detail",
        focus=["recent_form"],
        league_context={"season_year": 2024, "league_type": "KBO"},
    )
    agent = MagicMock()
    agent._convert_team_id_to_name.side_effect = lambda team_id: team_id

    monkeypatch.setattr("app.routers.coach.get_connection_pool", lambda: MagicMock())
    monkeypatch.setattr(
        "app.routers.coach._collect_game_evidence",
        lambda *args, **kwargs: types.SimpleNamespace(
            home_team_code="SSG",
            away_team_code="LG",
            home_team_name="SSG",
            away_team_name="LG",
            game_id="20240401SSLG0",
            league_type_code="REG",
            stage_label="REGULAR",
            game_status_bucket="SCHEDULED",
            season_id=20245,
            game_date="2024-04-01",
            home_pitcher=None,
            away_pitcher=None,
            home_lineup=[],
            away_lineup=[],
        ),
    )
    monkeypatch.setattr(
        "app.routers.coach._build_effective_league_context",
        lambda league_context, game_evidence: dict(league_context or {}),
    )
    monkeypatch.setattr(
        "app.routers.coach.build_coach_cache_identity",
        lambda **kwargs: (
            "cache-key",
            {
                "focus_signature": "focus-signature",
                "question_signature": "question-signature",
            },
            "starter-signature",
            "lineup-signature",
        ),
    )
    monkeypatch.setattr(
        "app.routers.coach.assess_game_evidence",
        lambda game_evidence: types.SimpleNamespace(root_causes=[]),
    )

    response = asyncio.run(analyze_team(payload, agent=agent, _=None))
    expected_ping = max(1, int(get_settings().chat_sse_ping_seconds))

    if hasattr(response, "ping_interval"):
        assert response.ping_interval == expected_ping
    else:
        assert response.kwargs["ping"] == expected_ping
