import pytest
from datetime import datetime
from app.config import get_settings

try:
    import psycopg
    from psycopg import sql
except ModuleNotFoundError:
    psycopg = None
    sql = None

# We can import build_select_query directly from the script
from scripts.ingest_from_kbo import TABLE_PROFILES, build_select_query


@pytest.fixture
def sample_since():
    return datetime(2025, 4, 1, 12, 0)


def test_build_select_query_with_alias(sample_since):
    profile = {
        "select_sql": "SELECT bs.* FROM player_season_batting bs",
        "season_filter_column": "bs.season",
        "since_filter_column": "bs.updated_at",
    }
    query, params = build_select_query(
        table="player_season_batting",
        profile=profile,
        pk_columns=["id"],
        limit=None,
        season_year=2025,
        since=sample_since,
    )

    assert "WHERE bs.season = %s AND bs.updated_at >= %s" in query
    assert params == (2025, sample_since)


def test_build_select_query_without_alias(sample_since):
    profile = {
        "select_sql": "SELECT * FROM team_name_mapping",
        "season_filter_column": None,
        "since_filter_column": "updated_at",
    }
    query, params = build_select_query(
        table="team_name_mapping",
        profile=profile,
        pk_columns=["full_name"],
        limit=None,
        season_year=None,
        since=sample_since,
    )

    assert "WHERE updated_at >= %s" in query
    assert params == (sample_since,)


def test_build_select_query_with_existing_where(sample_since):
    profile = {
        "select_sql": "SELECT tb.* FROM table_b tb WHERE tb.status = 'ACTIVE'",
        "season_filter_column": "tb.season",
        "since_filter_column": "tb.updated_at",
    }
    query, params = build_select_query(
        table="table_b",
        profile=profile,
        pk_columns=["id"],
        limit=10,
        season_year=2024,
        since=sample_since,
    )

    # Original string parsing logic injects AND when WHERE already exists
    assert "WHERE tb.status = 'ACTIVE'" in query
    assert "AND tb.season = %s AND tb.updated_at >= %s" in query
    assert params == (2024, sample_since, 10)


def test_build_select_query_uses_last_order_by_for_custom_sql(sample_since):
    profile = {
        "select_sql": """
            SELECT *
            FROM (
                SELECT
                    game_id,
                    json_agg(
                        json_build_object('inning', inning)
                        ORDER BY inning
                    ) AS inning_lines_json
                FROM game_inning_scores
                GROUP BY game_id
            ) game_flow_rows
            ORDER BY game_id DESC
        """,
        "season_filter_column": "season_year",
        "since_filter_column": "latest_updated_at",
    }

    query, params = build_select_query(
        table="game_flow_summary",
        profile=profile,
        pk_columns=["game_id"],
        limit=5,
        season_year=2025,
        since=sample_since,
    )

    assert "ORDER BY inning" in query
    assert "WHERE season_year = %s AND latest_updated_at >= %s" in query
    assert query.rstrip().endswith("ORDER BY game_id DESC LIMIT %s")
    assert params == (2025, sample_since, 5)


def test_build_select_query_appends_outer_where_after_nested_where(sample_since):
    profile = {
        "select_sql": """
            SELECT *
            FROM (
                SELECT
                    game_id,
                    season_year,
                    latest_updated_at
                FROM game
                WHERE home_score IS NOT NULL
            ) game_flow_rows
            ORDER BY game_id DESC
        """,
        "season_filter_column": "season_year",
        "since_filter_column": "latest_updated_at",
    }

    query, params = build_select_query(
        table="game_flow_summary",
        profile=profile,
        pk_columns=["game_id"],
        limit=None,
        season_year=2025,
        since=sample_since,
    )

    assert "WHERE home_score IS NOT NULL" in query
    assert (
        ") game_flow_rows\nWHERE season_year = %s AND latest_updated_at >= %s" in query
    )
    assert params == (2025, sample_since)


def test_awards_profile_matches_current_schema() -> None:
    profile = TABLE_PROFILES["awards"]

    assert profile["season_filter_column"] == "a.award_year"
    assert profile["since_filter_column"] == "a.updated_at"
    assert "a.award_year AS season_year" in profile["select_sql"]
    assert "a.team_name" in profile["select_sql"]


def test_game_flow_summary_profile_matches_current_schema() -> None:
    profile = TABLE_PROFILES["game_flow_summary"]

    assert profile["since_filter_column"] == "latest_updated_at"
    assert "MAX(ip.updated_at)" in profile["select_sql"]
    assert "g.updated_at" not in profile["select_sql"]


def test_player_basic_profile_disables_incremental_since_filter(sample_since) -> None:
    profile = TABLE_PROFILES["player_basic"]

    assert profile["since_filter_column"] is None

    query, params = build_select_query(
        table="player_basic",
        profile=profile,
        pk_columns=["player_id"],
        limit=None,
        season_year=None,
        since=sample_since,
    )

    assert "updated_at" not in query
    assert params == ()


def test_player_movements_profile_matches_current_schema(sample_since) -> None:
    profile = TABLE_PROFILES["player_movements"]

    assert profile["title_fields"][0] == ["movement_date"]
    assert profile["pk_hint"] == ["id", "movement_date", "player_name"]
    assert profile["season_filter_column"] == "EXTRACT(YEAR FROM pm.movement_date)"
    assert "EXTRACT(YEAR FROM pm.movement_date) AS season_year" in profile["select_sql"]
    assert "ORDER BY pm.movement_date DESC" in profile["select_sql"]

    query, params = build_select_query(
        table="player_movements",
        profile=profile,
        pk_columns=["id"],
        limit=None,
        season_year=2025,
        since=sample_since,
    )

    assert (
        "WHERE EXTRACT(YEAR FROM pm.movement_date) = %s AND pm.updated_at >= %s"
        in query
    )
    assert params == (2025, sample_since)


def test_game_profile_uses_game_metadata_updated_at(sample_since) -> None:
    profile = TABLE_PROFILES["game"]

    assert profile["since_filter_column"] == "gm.updated_at"
    assert "LEFT JOIN game_metadata gm" in profile["select_sql"]
    assert "gm.updated_at AS game_updated_at" in profile["select_sql"]

    query, params = build_select_query(
        table="game",
        profile=profile,
        pk_columns=["game_id"],
        limit=None,
        season_year=2025,
        since=sample_since,
    )

    assert "WHERE ks.season_year = %s AND gm.updated_at >= %s" in query
    assert params == (2025, sample_since)
