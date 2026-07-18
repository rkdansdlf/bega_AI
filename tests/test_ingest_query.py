import pytest
from datetime import datetime
from app.config import get_settings
from app.core.ingest_checkpoints import (
    CheckpointCursor,
    CheckpointOrder,
    CheckpointOrderField,
    IngestCheckpointCursorUnavailableError,
)

try:
    import psycopg
    from psycopg import sql
except ModuleNotFoundError:
    psycopg = None
    sql = None

# We can import build_select_query directly from the script
from scripts.ingest_from_kbo import (
    REQUIRED_SOURCE_COLUMNS,
    TABLE_PROFILES,
    build_select_query,
)
import scripts.ingest_from_kbo as ingest_script


@pytest.fixture
def sample_since():
    return datetime(2025, 4, 1, 12, 0)


def test_ingest_from_kbo_does_not_force_pytest_env_flag() -> None:
    source = __import__("pathlib").Path("scripts/ingest_from_kbo.py").read_text(
        encoding="utf-8"
    )

    assert "PYTEST_CURRENT_TEST" not in source


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


def test_build_select_query_applies_date_to_exclusive_for_game_summary() -> None:
    profile = TABLE_PROFILES["game_summary"]

    query, params = build_select_query(
        table="game_summary",
        profile=profile,
        pk_columns=["id"],
        limit=None,
        season_year=2026,
        since=None,
        date_to_exclusive=datetime(2026, 5, 1).date(),
    )

    assert "ks.season_year = %s" in query
    assert "g.game_date < %s" in query
    assert "ORDER BY g.game_date DESC" in query
    assert params == (2026, datetime(2026, 5, 1).date())


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


def test_build_select_query_default_branch_respects_since_filter_column(sample_since):
    profile = {
        "season_filter_column": None,
        "since_filter_column": "last_changed_at",
    }
    query, params = build_select_query(
        table="plain_table",
        profile=profile,
        pk_columns=["id"],
        limit=None,
        season_year=None,
        since=sample_since,
    )

    rendered = str(query)
    assert "last_changed_at" in rendered
    assert "updated_at" not in rendered
    assert params == (sample_since,)


def test_build_select_query_default_branch_skips_since_when_disabled(sample_since):
    profile = {
        "season_filter_column": None,
        "since_filter_column": None,
    }
    query, params = build_select_query(
        table="plain_table",
        profile=profile,
        pk_columns=["id"],
        limit=None,
        season_year=None,
        since=sample_since,
    )

    rendered = str(query)
    assert "updated_at" not in rendered
    assert params == ()


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


def test_every_custom_database_profile_declares_checkpoint_order():
    missing = [
        table
        for table, profile in TABLE_PROFILES.items()
        if "source_file" not in profile
        and "select_sql" in profile
        and not profile.get("checkpoint_order")
    ]
    assert missing == []


def test_custom_checkpoint_query_wraps_output_aliases(sample_since):
    profile = TABLE_PROFILES["game"]
    order = ingest_script.resolve_checkpoint_order(None, "game", profile)
    cursor = CheckpointCursor((41,))

    query, params = build_select_query(
        table="game",
        profile=profile,
        pk_columns=["id"],
        limit=None,
        season_year=2026,
        since=sample_since,
        checkpoint_order=order,
        resume_cursor=cursor,
    )

    assert "WITH checkpoint_source AS" in query
    assert 'ROW("id") > ROW(%s)' in query
    assert query.rstrip().endswith('ORDER BY "id" ASC')
    assert params == (2026, sample_since, 41)


def test_generic_checkpoint_query_uses_composite_primary_key():
    order = CheckpointOrder(
        "plain_table",
        (
            CheckpointOrderField("season", "integer"),
            CheckpointOrderField("entity_id", "text"),
        ),
    )
    query, params = build_select_query(
        table="plain_table",
        profile={"season_filter_column": None, "since_filter_column": None},
        pk_columns=["season", "entity_id"],
        limit=None,
        season_year=None,
        since=None,
        checkpoint_order=order,
        resume_cursor=CheckpointCursor((2025, "P100")),
    )

    rendered = str(query)
    assert "ROW" in rendered and "season" in rendered and "entity_id" in rendered
    assert params == (2025, "P100")


def test_checkpoint_query_rejects_unsafe_order_field():
    order = CheckpointOrder(
        "plain_table",
        (CheckpointOrderField('id") > ROW(0); --', "integer"),),
    )

    with pytest.raises(IngestCheckpointCursorUnavailableError):
        build_select_query(
            table="plain_table",
            profile={"season_filter_column": None, "since_filter_column": None},
            pk_columns=["id"],
            limit=None,
            season_year=None,
            since=None,
            checkpoint_order=order,
            resume_cursor=CheckpointCursor((1,)),
        )


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


def test_team_profiles_profile_disables_incremental_since_filter(sample_since) -> None:
    profile = TABLE_PROFILES["team_profiles"]

    assert profile["since_filter_column"] is None

    query, params = build_select_query(
        table="team_profiles",
        profile=profile,
        pk_columns=["team_id", "id"],
        limit=None,
        season_year=None,
        since=sample_since,
    )

    assert "tp.updated_at" not in query
    assert "updated_at >=" not in query
    assert "ORDER BY tp.team_id" in query
    assert params == ()


def test_player_movements_profile_matches_current_schema(sample_since) -> None:
    profile = TABLE_PROFILES["player_movements"]

    assert profile["title_fields"][0] == ["date"]
    assert profile["pk_hint"] == ["id", "date", "player_name"]
    assert profile["season_filter_column"] == "EXTRACT(YEAR FROM pm.date)"
    assert "EXTRACT(YEAR FROM pm.date) AS season_year" in profile["select_sql"]
    assert "ORDER BY pm.date DESC" in profile["select_sql"]

    query, params = build_select_query(
        table="player_movements",
        profile=profile,
        pk_columns=["id"],
        limit=None,
        season_year=2025,
        since=sample_since,
    )

    assert (
        "WHERE EXTRACT(YEAR FROM pm.date) = %s AND pm.updated_at >= %s"
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


def test_core_schedule_profiles_declare_required_source_fields() -> None:
    assert REQUIRED_SOURCE_COLUMNS["game"] == frozenset({"game_id", "game_date"})
    assert "game_id" in REQUIRED_SOURCE_COLUMNS["game_metadata"]
    assert {"game_id", "game_date"}.issubset(
        REQUIRED_SOURCE_COLUMNS["game_summary"]
    )
