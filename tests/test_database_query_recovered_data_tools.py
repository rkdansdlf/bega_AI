from datetime import date
import asyncio
from unittest.mock import AsyncMock, Mock

from app.tools.database_query import DatabaseQueryTool


def _run(coro):
    return asyncio.run(coro)


def _async_cursor():
    cur = Mock()
    cur.execute = AsyncMock()
    cur.fetchone = AsyncMock(return_value=None)
    cur.fetchall = AsyncMock(return_value=[])
    cur.close = AsyncMock()
    return cur


def _tool_with_connection(connection: Mock) -> DatabaseQueryTool:
    tool = DatabaseQueryTool.__new__(DatabaseQueryTool)
    tool.connection = connection
    tool._record_team_query_result = lambda *_args, **_kwargs: None
    return tool


def test_get_team_tough_matchups_sorts_lowest_win_rate_first() -> None:
    tool = _tool_with_connection(Mock())
    tool.get_team_matchup_stats = AsyncMock(return_value={
        "team_name": "LG 트윈스",
        "year": 2026,
        "matchups": {
            "KT 위즈": {"games": 6, "wins": 1, "losses": 5, "draws": 0},
            "두산 베어스": {"games": 6, "wins": 3, "losses": 3, "draws": 0},
            "KIA 타이거즈": {"games": 5, "wins": 0, "losses": 5, "draws": 0},
        },
        "found": True,
        "error": None,
    })

    result = _run(tool.get_team_tough_matchups("LG", 2026, limit=2))

    assert result["found"] is True
    assert [row["opponent"] for row in result["tough_matchups"]] == [
        "KIA 타이거즈",
        "KT 위즈",
    ]
    assert result["tough_matchups"][0]["win_rate"] == 0.0


def test_current_team_rank_prefers_latest_standings_daily() -> None:
    connection = Mock()
    cursor = _async_cursor()
    connection.cursor.return_value = cursor
    cursor.fetchone.return_value = {
        "team_code": "LG",
        "games_played": 51,
        "wins": 31,
        "losses": 20,
        "draws": 0,
        "win_pct": 0.6078431372549019,
        "games_behind": 0.0,
        "standings_date": date(date.today().year, 5, 28),
        "season_rank": 2,
    }
    tool = _tool_with_connection(connection)
    tool.get_team_variants = lambda *_args, **_kwargs: ["LG"]
    tool.get_team_name = lambda *_args, **_kwargs: "LG 트윈스"

    result = _run(tool._get_current_team_rank_from_standings_daily(
        "LG 트윈스", date.today().year
    ))

    assert result is not None
    assert result["found"] is True
    assert result["source"] == "team_standings_daily"
    assert result["rank"] == 2
    assert result["wins"] == 31
    query, params = cursor.execute.call_args[0]
    assert "team_standings_daily" in query
    assert params[2] == ["LG"]


def test_home_run_leaderboard_uses_counting_stat_without_pa_floor_and_canonical_id() -> None:
    connection = Mock()
    cursor = _async_cursor()
    connection.cursor.return_value = cursor
    cursor.fetchall.return_value = [
        {
            "player_id": 51868,
            "player_name": "고명준",
            "team_code": "SSG",
            "stat_value": 6,
            "plate_appearances": 88,
            "avg": 0.270,
            "obp": 0.320,
            "slg": 0.510,
            "ops": 0.830,
            "home_runs": 6,
            "rbi": 18,
            "total_qualified_players": 120,
        }
    ]
    tool = _tool_with_connection(connection)
    tool.get_team_name = lambda *_args, **_kwargs: "SSG 랜더스"

    result = _run(tool.get_team_leaderboard("home_runs", 2026, "batting", limit=10))

    assert result["found"] is True
    assert result["leaderboard"][0]["player_id"] == 51868
    assert result["leaderboard"][0]["stat_value"] == 6
    query, params = cursor.execute.call_args[0]
    assert "canonical_player_id" in query
    assert "WHEN 59359 THEN 56632" in query
    assert "psb.league = 'REGULAR'" in query
    assert params == [2026, 0, 10]


def test_team_metric_leaderboard_uses_team_aggregate_table() -> None:
    connection = Mock()
    cursor = _async_cursor()
    connection.cursor.return_value = cursor
    cursor.fetchall.return_value = [
        {"team_code": "LG", "stat_value": 42, "games": 30},
        {"team_code": "KT", "stat_value": 38, "games": 30},
    ]
    tool = _tool_with_connection(connection)
    tool.get_team_name = lambda team_code, *_args, **_kwargs: {
        "LG": "LG 트윈스",
        "KT": "KT 위즈",
    }[team_code]

    result = _run(tool.get_team_metric_leaderboard("home_runs", 2026, limit=10))

    assert result["found"] is True
    assert result["metric_label"] == "홈런"
    assert result["team_metric_leaderboard"][0]["team_name"] == "LG 트윈스"
    query, params = cursor.execute.call_args[0]
    assert "FROM team_season_batting" in query
    assert "home_runs AS stat_value" in query
    assert params == (2026, 10)


def test_team_metric_fielding_leaderboard_does_not_require_games_column() -> None:
    connection = Mock()
    cursor = _async_cursor()
    connection.cursor.return_value = cursor
    cursor.fetchall.return_value = [{"team_code": "LG", "stat_value": 12, "games": None}]
    tool = _tool_with_connection(connection)
    tool.get_team_name = lambda *_args, **_kwargs: "LG 트윈스"

    result = _run(tool.get_team_metric_leaderboard("errors", 2026, limit=10))

    assert result["found"] is True
    query, _params = cursor.execute.call_args[0]
    assert "FROM team_season_fielding" in query
    assert "NULL::integer AS games" in query
    assert "ORDER BY stat_value ASC" in query


def test_team_form_table_builds_recent_home_away_and_streak_payload() -> None:
    connection = Mock()
    cursor = _async_cursor()
    connection.cursor.return_value = cursor
    cursor.fetchall.return_value = [
        {
            "team": "LG",
            "opponent": "KT",
            "side": "home",
            "winning_team": "LG",
            "game_id": "20260430KTLG0",
            "game_date": date(2026, 4, 30),
            "team_score": 5,
            "opponent_score": 3,
            "result": "win",
        },
        {
            "team": "LG",
            "opponent": "KIA",
            "side": "away",
            "winning_team": "LG",
            "game_id": "20260429LGHT0",
            "game_date": date(2026, 4, 29),
            "team_score": 4,
            "opponent_score": 1,
            "result": "win",
        },
        {
            "team": "KT",
            "opponent": "LG",
            "side": "away",
            "winning_team": "LG",
            "game_id": "20260430KTLG0",
            "game_date": date(2026, 4, 30),
            "team_score": 3,
            "opponent_score": 5,
            "result": "loss",
        },
    ]
    tool = _tool_with_connection(connection)
    tool.get_team_name = lambda team_code, *_args, **_kwargs: {
        "LG": "LG 트윈스",
        "KT": "KT 위즈",
        "KIA": "KIA 타이거즈",
    }.get(team_code, team_code)

    result = _run(tool.get_team_form_table(2026, form_type="recent", recent_limit=10))

    assert result["found"] is True
    assert result["form_rows"][0]["team_name"] == "LG 트윈스"
    assert result["form_rows"][0]["wins"] == 2
    assert result["form_rows"][0]["streak_type"] == "win"
    assert result["form_rows"][0]["streak_count"] == 2
    query, params = cursor.execute.call_args[0]
    assert "UNION ALL" in query
    assert "kbo_seasons" in query
    assert params == [2026, 0, 2026, 0]


def test_get_team_fielding_error_games_builds_game_payload() -> None:
    connection = Mock()
    cursor = _async_cursor()
    connection.cursor.return_value = cursor
    cursor.fetchall.return_value = [
        {
            "game_id": "20260401KTLG0",
            "game_date": date(2026, 4, 1),
            "home_team": "LG",
            "away_team": "KT",
            "home_team_name": "LG 트윈스",
            "away_team_name": "KT 위즈",
            "home_score": 3,
            "away_score": 5,
            "winning_team": "KT",
            "summary_type": "실책",
            "player_name": "문보경",
            "detail_text": "6회말 실책 이후 결승점 허용",
        }
    ]
    tool = _tool_with_connection(connection)
    tool._is_regular_analysis_team = lambda *_args, **_kwargs: True
    tool.get_team_code = lambda *_args, **_kwargs: "LG"
    tool.get_team_variants = lambda *_args, **_kwargs: ["LG"]
    tool.get_team_name = lambda *_args, **_kwargs: "LG 트윈스"

    result = _run(tool.get_team_fielding_error_games("LG", 2026, limit=5))

    assert result["found"] is True
    assert result["team_name"] == "LG 트윈스"
    assert result["fielding_error_games"][0] == {
        "game_id": "20260401KTLG0",
        "game_date": "2026-04-01",
        "opponent": "KT 위즈",
        "score": "3:5",
        "result": "Loss",
        "summary_type": "실책",
        "player_name": "문보경",
        "detail_text": "6회말 실책 이후 결승점 허용",
    }
    query, params = cursor.execute.call_args[0]
    assert "game_summary" in query
    assert params[0] == ["LG"]


def test_get_player_position_average_comparison_returns_deltas() -> None:
    connection = Mock()
    cursor = _async_cursor()
    connection.cursor.return_value = cursor
    cursor.fetchone.side_effect = [
        {
            "player_id": "50666",
            "player_name": "김도영",
            "team_name": "KIA 타이거즈",
            "team_code": "KIA",
            "season_year": 2026,
            "plate_appearances": 210,
            "at_bats": 180,
            "hits": 58,
            "home_runs": 12,
            "rbi": 37,
            "avg": 0.322,
            "obp": 0.397,
            "slg": 0.561,
            "ops": 0.958,
        },
        {"position_id": "3B", "games": 50, "innings": 420},
        {
            "sample_size": 9,
            "plate_appearances": 185.0,
            "hits": 45.0,
            "home_runs": 6.0,
            "rbi": 25.0,
            "avg": 0.281,
            "obp": 0.348,
            "slg": 0.431,
            "ops": 0.779,
        },
    ]
    tool = _tool_with_connection(connection)

    result = _run(tool.get_player_position_average_comparison("김도영", 2026))

    assert result["found"] is True
    assert result["position_name"] == "3루수"
    assert result["sample_size"] == 9
    assert result["target"]["player_name"] == "김도영"
    assert result["position_average"]["ops"] == 0.779
    assert result["deltas"]["ops"] == 0.179
    assert result["deltas"]["home_runs"] == 6.0
