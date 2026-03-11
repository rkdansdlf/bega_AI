import pytest
from unittest.mock import MagicMock
from app.tools.database_query import DatabaseQueryTool
from app.tools.game_query import GameQueryTool
from app.tools.team_code_resolver import TeamCodeResolver


class TestTeamMappingRobustness:
    @pytest.fixture
    def mock_db_connection(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        # Mock fetchall to return empty list for mapping load
        mock_cursor.fetchall.return_value = []
        return mock_conn

    @pytest.fixture
    def db_tool(self, mock_db_connection):
        return DatabaseQueryTool(mock_db_connection)

    @pytest.fixture
    def game_tool(self, mock_db_connection):
        return GameQueryTool(mock_db_connection)

    def test_db_tool_variants_canonical_input(self, db_tool):
        # Case: Canonical Input -> Should return list including Legacy
        variants = db_tool.get_team_variants("SSG")
        assert "SSG" in variants
        assert "SK" in variants
        assert len(variants) >= 2

        variants = db_tool.get_team_variants("KIA")
        assert "KIA" in variants
        assert "HT" in variants

    def test_db_tool_variants_legacy_input(self, db_tool):
        # Case: Legacy Input -> Should resolve to Canonical then return list
        # Note: NAME_TO_STATS_CODE maps Legacy -> Canonical now
        variants = db_tool.get_team_variants("SK")
        assert "SSG" in variants
        assert "SK" in variants

        variants = db_tool.get_team_variants("HT")
        assert "KIA" in variants
        assert "HT" in variants

    def test_db_tool_variants_korean_alias(self, db_tool):
        # Case: Korean Alias -> Should resolve to Canonical then return list
        variants = db_tool.get_team_variants("삼성")
        assert "SS" in variants

        variants = db_tool.get_team_variants("롯데")
        assert "LT" in variants
        assert "LOT" in variants

    def test_game_tool_variants_consistency(self, game_tool):
        # Verify GameQueryTool shares same logic
        variants = game_tool.get_team_variants("KH")
        assert "KH" in variants
        assert "WO" in variants
        assert "NX" in variants

    def test_game_tool_game_query_params(self, game_tool):
        # Verify that providing a team name results in ANY(%s) param logic
        # We can't easily check the SQL result without a real DB,
        # but we can check internal method behavior if we mocked the cursor execute.

        mock_cursor = game_tool.connection.cursor.return_value

        # Test get_games_by_date
        game_tool.get_games_by_date("2024-05-01", "SSG")

        # Get the arguments passed to execute
        call_args = mock_cursor.execute.call_args
        query, params = call_args[0]

        # Verify query structure
        assert "g.home_team = ANY(%s)" in query or "g.home_team = ANY(" in query

        # Verify SSG variants are passed
        assert "SSG" in params[1]  # [date, variants, variants]
        assert "SK" in params[1]

    def test_game_tool_box_score_uses_inning_column(self, game_tool):
        mock_cursor = game_tool.connection.cursor.return_value
        mock_cursor.execute.reset_mock()
        mock_cursor.fetchall.side_effect = [
            [
                {
                    "game_id": "20250501SSLG0",
                    "game_date": "2025-05-01",
                    "home_team": "LG",
                    "away_team": "SS",
                    "home_score": 5,
                    "away_score": 3,
                    "game_status": "FINAL",
                    "stadium": "잠실",
                    "winning_team": "LG",
                    "home_pitcher": "A",
                    "away_pitcher": "B",
                }
            ],
            [
                {"inning": 1, "team_side": "home", "runs": 1},
                {"inning": 1, "team_side": "away", "runs": 0},
                {"inning": 2, "team_side": "home", "runs": 0},
                {"inning": 2, "team_side": "away", "runs": 2},
            ],
            [
                {"team_code": "LG", "total_hits": 8, "total_rbi": 5},
                {"team_code": "SS", "total_hits": 6, "total_rbi": 3},
            ],
        ]

        result = game_tool.get_game_box_score(date="2025-05-01")

        inning_query = mock_cursor.execute.call_args_list[1][0][0]
        assert "SELECT inning, team_side, runs" in inning_query
        assert "inning_number" not in inning_query
        assert result["games"][0]["box_score"]["home_1"] == 1
        assert result["games"][0]["box_score"]["away_2"] == 2

    def test_db_tool_leaderboard_query_params(self, monkeypatch, mock_db_connection):
        monkeypatch.setenv("TEAM_CODE_READ_MODE", "canonical_only")
        monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_START", "2021")
        monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_END", "2025")
        monkeypatch.setenv("TEAM_CODE_OUTSIDE_WINDOW_MODE", "dual")
        db_tool = DatabaseQueryTool(mock_db_connection)
        mock_cursor = db_tool.connection.cursor.return_value

        # Test get_team_leaderboard
        db_tool.get_team_leaderboard("ops", 2024, "batting", team_filter="KIA")

        call_args = mock_cursor.execute.call_args
        query, params = call_args[0]

        # Verify ANY clause is used regardless of the selected SQL alias.
        assert "team_code = ANY(%s)" in query

        # Verify params contain variants
        # Params order: [year, year, [variants], min_pa, limit]
        found_variants = False
        for param in params:
            if isinstance(param, list) and param == ["KIA"]:
                found_variants = True
                break
        assert found_variants

    def test_db_tool_team_summary_uses_any(self, monkeypatch, mock_db_connection):
        monkeypatch.setenv("TEAM_CODE_READ_MODE", "canonical_only")
        monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_START", "2021")
        monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_END", "2025")
        monkeypatch.setenv("TEAM_CODE_OUTSIDE_WINDOW_MODE", "dual")
        db_tool = DatabaseQueryTool(mock_db_connection)
        mock_cursor = db_tool.connection.cursor.return_value
        mock_cursor.execute.reset_mock()
        mock_cursor.fetchall.return_value = []

        db_tool.get_team_summary("SSG", 2024)

        calls = mock_cursor.execute.call_args_list
        assert any("psb.team_code = ANY(%s)" in call[0][0] for call in calls)
        assert any("psp.team_code = ANY(%s)" in call[0][0] for call in calls)

        found_variants = False
        for call in calls:
            params = call[0][1]
            if not isinstance(params, tuple):
                continue
            for param in params:
                if isinstance(param, list) and param == ["SSG"]:
                    found_variants = True
                    break
            if found_variants:
                break
        assert found_variants

    def test_db_tool_recent_form_filters_with_season_year(
        self, monkeypatch, mock_db_connection
    ):
        monkeypatch.setenv("TEAM_CODE_READ_MODE", "canonical_only")
        monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_START", "2021")
        monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_END", "2025")
        monkeypatch.setenv("TEAM_CODE_OUTSIDE_WINDOW_MODE", "dual")
        db_tool = DatabaseQueryTool(mock_db_connection)

        mock_cursor = db_tool.connection.cursor.return_value
        mock_cursor.execute.reset_mock()
        mock_cursor.fetchall.return_value = []

        db_tool.get_team_recent_form("SSG", 2019, limit=10)

        query, params = mock_cursor.execute.call_args[0]
        assert "JOIN kbo_seasons ks ON g.season_id = ks.season_id" in query
        assert "ks.season_year = %s" in query
        assert "CAST(g.season_id AS TEXT)" not in query
        assert params[4] == 2019
        assert "SSG" in params[2]
        assert "SK" in params[2]

    def test_game_tool_last_game_date_uses_any(self, monkeypatch, mock_db_connection):
        monkeypatch.setenv("TEAM_CODE_READ_MODE", "canonical_only")
        monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_START", "2021")
        monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_END", "2025")
        monkeypatch.setenv("TEAM_CODE_OUTSIDE_WINDOW_MODE", "dual")
        game_tool = GameQueryTool(mock_db_connection)

        mock_cursor = game_tool.connection.cursor.return_value
        mock_cursor.execute.reset_mock()
        mock_cursor.fetchone.return_value = {"last_game_date": None}

        game_tool.get_team_last_game_date("SSG", 2024, "regular_season")

        query, params = mock_cursor.execute.call_args[0]
        assert "g.home_team = ANY(%s)" in query
        assert "g.away_team = ANY(%s)" in query
        assert "SSG" in params[2]
        assert "SK" not in params[2]

    def test_game_tool_last_game_date_outside_window_falls_back_to_dual(
        self, monkeypatch, mock_db_connection
    ):
        monkeypatch.setenv("TEAM_CODE_READ_MODE", "canonical_only")
        monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_START", "2021")
        monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_END", "2025")
        monkeypatch.setenv("TEAM_CODE_OUTSIDE_WINDOW_MODE", "dual")
        game_tool = GameQueryTool(mock_db_connection)

        mock_cursor = game_tool.connection.cursor.return_value
        mock_cursor.execute.reset_mock()
        mock_cursor.fetchone.return_value = {"last_game_date": None}

        game_tool.get_team_last_game_date("SSG", 2019, "regular_season")
        _, params = mock_cursor.execute.call_args[0]
        assert "SSG" in params[2]
        assert "SK" in params[2]

    def test_db_tool_regular_analysis_rejects_special_team(self, db_tool):
        result = db_tool.get_team_summary("EA", 2024)
        assert result["found"] is False
        assert result["error"] == "unsupported_team_for_regular_analysis"
        assert result["reason"] == "unsupported_team_for_regular_analysis"


def test_team_code_resolver_canonical_only(monkeypatch):
    monkeypatch.setenv("TEAM_CODE_READ_MODE", "canonical_only")
    monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_START", "2021")
    monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_END", "2025")
    monkeypatch.setenv("TEAM_CODE_OUTSIDE_WINDOW_MODE", "dual")
    resolver = TeamCodeResolver()
    assert resolver.variants("SSG", 2024) == ["SSG"]
    assert resolver.variants("KH", 2024) == ["KH"]
    outside = resolver.variants("SSG", 2019)
    assert "SSG" in outside
    assert "SK" in outside


def test_team_code_resolver_english_aliases():
    resolver = TeamCodeResolver()
    assert resolver.resolve_canonical("KT Wiz") == "KT"
    assert resolver.resolve_canonical("LG Twins") == "LG"
    assert resolver.resolve_canonical("Hanwha Eagles") == "HH"


def test_team_code_resolver_display_name_aliases():
    resolver = TeamCodeResolver()
    assert resolver.display_name("kt wiz") == "KT 위즈"
    assert resolver.display_name("LG Twins") == "LG 트윈스"
    assert resolver.display_name("Hanwha Eagles") == "한화 이글스"


def test_team_code_resolver_sync_preserves_korean_display_name():
    resolver = TeamCodeResolver()

    resolver.sync_from_team_rows(
        [
            {
                "franchise_id": 1,
                "team_id": "KT",
                "team_name": "kt wiz",
                "current_code": "KT",
            }
        ]
    )

    assert resolver.code_to_name["KT"] == "KT 위즈"
    assert resolver.display_name("kt wiz") == "KT 위즈"
