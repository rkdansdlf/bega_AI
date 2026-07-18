import pytest
from datetime import datetime
from app.config import get_settings
from app.core.ingest_checkpoints import (
    CheckpointCursor,
    CheckpointOrder,
    CheckpointOrderField,
    IngestCheckpointCursorUnavailableError,
    IngestCheckpointIncompatibleError,
    decode_cursor,
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
    build_content,
    build_select_query,
)
import scripts.ingest_from_kbo as ingest_script


@pytest.fixture
def sample_since():
    return datetime(2025, 4, 1, 12, 0)


@pytest.fixture
def sample_cutoff():
    return datetime(2025, 4, 1, 13, 0)


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


def test_game_summary_legacy_query_and_content_keep_team_name_projection():
    profile = TABLE_PROFILES["game_summary"]
    legacy_sql = profile["select_sql"]

    assert "t.team_name" in legacy_sql
    assert (
        "LEFT JOIN teams t ON (t.team_id = g.home_team OR t.team_id = g.away_team)"
        in legacy_sql
    )

    query, params = build_select_query(
        table="game_summary",
        profile=profile,
        pk_columns=["id"],
        limit=None,
        season_year=None,
        since=None,
    )

    assert "t.team_name" in query
    assert (
        "LEFT JOIN teams t ON (t.team_id = g.home_team OR t.team_id = g.away_team)"
        in query
    )
    assert "LEFT JOIN LATERAL" not in query
    assert params == ()
    content = build_content(
        {"id": 7, "game_id": "G7", "team_name": "LG"},
        "game_summary",
        "7",
        profile,
    )
    assert "team_name: LG" in content


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


def test_custom_checkpoint_registry_is_exact_and_static_profiles_are_isolated():
    assert ingest_script.CUSTOM_CHECKPOINT_ORDERS == {
        "player_season_batting": (("id", "integer"),),
        "player_season_pitching": (("id", "integer"),),
        "game": (("id", "integer"),),
        "game_flow_summary": (("game_id", "text"),),
        "game_batting_stats": (("id", "integer"),),
        "game_pitching_stats": (("id", "integer"),),
        "game_inning_scores": (("id", "integer"),),
        "game_lineups": (("id", "integer"),),
        "game_metadata": (("game_id", "text"),),
        "team_history": (("id", "integer"),),
        "awards": (("id", "integer"),),
        "player_movements": (("id", "integer"),),
        "team_franchises": (("id", "integer"),),
        "player_basic": (("player_id", "text"),),
        "team_name_mapping": (("full_name", "text"),),
        "team_profiles": (("id", "integer"),),
        "team_season_batting": (("id", "integer"),),
        "team_season_pitching": (("id", "integer"),),
        "stat_rankings": (("id", "integer"),),
        "game_summary": (("id", "integer"),),
    }
    static_profiles = [
        profile for profile in TABLE_PROFILES.values() if "source_file" in profile
    ]
    assert static_profiles
    assert all("checkpoint_order" not in profile for profile in static_profiles)
    assert all("checkpoint_query_version" not in profile for profile in static_profiles)


def test_full_cutoff_eligibility_is_exact_for_every_trusted_database_profile():
    expected = {
        "player_season_batting": "bs.updated_at",
        "player_season_pitching": "ps.updated_at",
        "game": "gm.updated_at",
        "game_flow_summary": "latest_updated_at",
        "game_batting_stats": "gbs.updated_at",
        "game_pitching_stats": "gps.updated_at",
        "game_inning_scores": "gis.updated_at",
        "game_lineups": "gl.updated_at",
        "game_metadata": "gm.updated_at",
        "kbo_seasons": None,
        "stadiums": None,
        "teams": None,
        "team_history": "th.updated_at",
        "team_name_mapping": None,
        "awards": "a.updated_at",
        "player_movements": "pm.updated_at",
        "team_franchises": "tf.updated_at",
        "player_basic": None,
        "team_profiles": None,
        "team_season_batting": "tsb.updated_at",
        "team_season_pitching": "tsp.updated_at",
        "stat_rankings": "sr.updated_at",
        "game_summary": "gs.updated_at",
    }

    actual = {
        table: ingest_script.resolve_update_filter_column(profile, since=None)
        for table, profile in TABLE_PROFILES.items()
        if "source_file" not in profile
    }

    assert actual == expected


def test_incremental_update_filter_is_exact_for_every_trusted_database_profile(
    sample_since,
):
    expected = {
        "player_season_batting": "bs.updated_at",
        "player_season_pitching": "ps.updated_at",
        "game": "gm.updated_at",
        "game_flow_summary": "latest_updated_at",
        "game_batting_stats": "gbs.updated_at",
        "game_pitching_stats": "gps.updated_at",
        "game_inning_scores": "gis.updated_at",
        "game_lineups": "gl.updated_at",
        "game_metadata": "gm.updated_at",
        "kbo_seasons": "updated_at",
        "stadiums": "updated_at",
        "teams": "updated_at",
        "team_history": "th.updated_at",
        "team_name_mapping": "updated_at",
        "awards": "a.updated_at",
        "player_movements": "pm.updated_at",
        "team_franchises": "tf.updated_at",
        "player_basic": None,
        "team_profiles": None,
        "team_season_batting": "tsb.updated_at",
        "team_season_pitching": "tsp.updated_at",
        "stat_rankings": "sr.updated_at",
        "game_summary": "gs.updated_at",
    }

    actual = {
        table: ingest_script.resolve_update_filter_column(profile, since=sample_since)
        for table, profile in TABLE_PROFILES.items()
        if "source_file" not in profile
    }

    assert actual == expected


def test_kbo_seasons_full_checkpoint_does_not_require_or_emit_default_cutoff(
    sample_cutoff,
):
    profile = TABLE_PROFILES["kbo_seasons"]
    order = CheckpointOrder(
        "kbo_seasons",
        (
            CheckpointOrderField("season_year", "integer"),
            CheckpointOrderField("season_id", "text"),
        ),
    )

    query, params = build_select_query(
        table="kbo_seasons",
        profile=profile,
        pk_columns=["season_year", "season_id"],
        limit=None,
        season_year=None,
        since=None,
        source_updated_before=sample_cutoff,
        checkpoint_order=order,
        resume_cursor=None,
    )

    assert "updated_at" not in query.as_string()
    assert params == ()


def test_kbo_seasons_incremental_checkpoint_keeps_default_bounded_update_filter(
    sample_since,
    sample_cutoff,
):
    profile = TABLE_PROFILES["kbo_seasons"]
    order = CheckpointOrder(
        "kbo_seasons",
        (
            CheckpointOrderField("season_year", "integer"),
            CheckpointOrderField("season_id", "text"),
        ),
    )

    query, params = build_select_query(
        table="kbo_seasons",
        profile=profile,
        pk_columns=["season_year", "season_id"],
        limit=None,
        season_year=None,
        since=sample_since,
        source_updated_before=sample_cutoff,
        checkpoint_order=order,
        resume_cursor=None,
    )

    assert (
        'WHERE "updated_at" >= %s AND "updated_at" <= %s'
        in query.as_string()
    )
    assert params == (sample_since, sample_cutoff)


def test_game_full_checkpoint_includes_null_updates_with_cutoff(sample_cutoff):
    profile = TABLE_PROFILES["game"]
    order = ingest_script.resolve_checkpoint_order(None, "game", profile)

    query, params = build_select_query(
        table="game",
        profile=profile,
        pk_columns=["id"],
        limit=None,
        season_year=None,
        since=None,
        source_updated_before=sample_cutoff,
        checkpoint_order=order,
        resume_cursor=CheckpointCursor((41,)),
    )

    assert "(gm.updated_at IS NULL OR gm.updated_at <= %s)" in query
    assert "gm.updated_at >= %s" not in query
    assert query.index("gm.updated_at <= %s") < query.index('ROW("id") > ROW(%s)')
    assert params == (sample_cutoff, 41)


def test_generic_full_checkpoint_includes_null_updates_with_cutoff(sample_cutoff):
    order = CheckpointOrder(
        "plain_table",
        (CheckpointOrderField("id", "integer"),),
    )

    query, params = build_select_query(
        table="plain_table",
        profile={"season_filter_column": None, "since_filter_column": "updated_at"},
        pk_columns=["id"],
        limit=None,
        season_year=None,
        since=None,
        source_updated_before=sample_cutoff,
        checkpoint_order=order,
        resume_cursor=None,
    )

    assert query.as_string() == (
        'SELECT * FROM "plain_table" WHERE '
        '("updated_at" IS NULL OR "updated_at" <= %s) ORDER BY "id" ASC'
    )
    assert params == (sample_cutoff,)


def test_custom_checkpoint_query_wraps_output_aliases(sample_since, sample_cutoff):
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
        source_updated_before=sample_cutoff,
        checkpoint_order=order,
        resume_cursor=cursor,
    )

    assert "WITH checkpoint_source AS" in query
    assert 'ROW("id") > ROW(%s)' in query
    assert query.rstrip().endswith('ORDER BY "id" ASC')
    assert "gm.updated_at <= %s" in query
    assert "gm.updated_at IS NULL" not in query
    assert params == (2026, sample_since, sample_cutoff, 41)


def test_custom_checkpoint_query_freezes_update_window_before_resume(
    sample_since,
    sample_cutoff,
):
    order = CheckpointOrder(
        "sample",
        (CheckpointOrderField("id", "integer"),),
    )
    query, params = build_select_query(
        table="sample",
        profile={
            "select_sql": "SELECT s.* FROM sample s ORDER BY s.id DESC",
            "season_filter_column": None,
            "since_filter_column": "s.updated_at",
        },
        pk_columns=["id"],
        limit=None,
        season_year=None,
        since=sample_since,
        source_updated_before=sample_cutoff,
        checkpoint_order=order,
        resume_cursor=CheckpointCursor((41,)),
    )

    assert "s.updated_at >= %s AND s.updated_at <= %s" in query
    assert query.index("s.updated_at <= %s") < query.index('ROW("id") > ROW(%s)')
    assert query.rstrip().endswith('ORDER BY "id" ASC')
    assert params == (sample_since, sample_cutoff, 41)


def test_team_profiles_checkpoint_uses_unique_id_for_same_team_profiles():
    profile = TABLE_PROFILES["team_profiles"]
    order = ingest_script.resolve_checkpoint_order(None, "team_profiles", profile)
    rows = [
        {"id": 41, "team_id": "TEST_TEAM", "profile": "history"},
        {"id": 42, "team_id": "TEST_TEAM", "profile": "identity"},
    ]

    yielded = list(
        ingest_script.iter_checkpoint_rows(rows, order=order, previous=None)
    )
    query, params = build_select_query(
        table="team_profiles",
        profile=profile,
        pk_columns=["id"],
        limit=None,
        season_year=None,
        since=None,
        checkpoint_order=order,
        resume_cursor=CheckpointCursor((41,)),
    )

    assert order.fields == (CheckpointOrderField("id", "integer"),)
    assert [cursor for _row, cursor in yielded] == [
        CheckpointCursor((41,)),
        CheckpointCursor((42,)),
    ]
    assert 'ROW("id") > ROW(%s)' in query
    assert query.rstrip().endswith('ORDER BY "id" ASC')
    assert params == (41,)


def test_datetime_naive_checkpoint_query_rebinds_naive_parameter():
    order = CheckpointOrder(
        "event",
        (CheckpointOrderField("occurred_at", "datetime_naive"),),
    )
    payload = {
        "values": [
            {
                "field": "occurred_at",
                "type": "datetime_naive",
                "value": "2026-07-18T04:00:00",
            }
        ]
    }
    resume = decode_cursor(order, payload)

    _query, params = build_select_query(
        table="event",
        profile={"season_filter_column": None, "since_filter_column": None},
        pk_columns=["occurred_at"],
        limit=None,
        season_year=None,
        since=None,
        checkpoint_order=order,
        resume_cursor=resume,
    )

    assert params == (datetime(2026, 7, 18, 4, 0),)
    assert params[0].tzinfo is None


def test_game_summary_checkpoint_source_has_one_logical_row_per_id(sample_since):
    profile = TABLE_PROFILES["game_summary"]
    source_sql = profile["checkpoint_select_sql"]

    assert "FROM game_summary gs" in source_sql
    assert "checkpoint_team.team_name" in source_sql
    assert "LEFT JOIN LATERAL" in source_sql
    assert "CASE WHEN t.team_id = g.home_team THEN 0 ELSE 1 END" in source_sql
    assert "t.team_id ASC" in source_sql
    assert "LIMIT 1" in source_sql

    order = ingest_script.resolve_checkpoint_order(None, "game_summary", profile)
    query, params = build_select_query(
        table="game_summary",
        profile=profile,
        pk_columns=["id"],
        limit=10,
        season_year=2026,
        since=sample_since,
        source_updated_before=datetime(2026, 7, 18, 13, 0),
        date_to_exclusive=datetime(2026, 8, 1).date(),
        checkpoint_order=order,
        resume_cursor=CheckpointCursor((91,)),
    )

    assert query.rstrip().endswith('ORDER BY "id" ASC LIMIT %s')
    assert "checkpoint_team.team_name" in query
    assert "LEFT JOIN LATERAL" in query
    assert "LIMIT 1" in query
    assert (
        "LEFT JOIN teams t ON (t.team_id = g.home_team OR t.team_id = g.away_team)"
        not in query
    )
    assert query.count('ROW("id") > ROW(%s)') == 1
    assert params == (
        2026,
        sample_since,
        datetime(2026, 7, 18, 13, 0),
        datetime(2026, 8, 1).date(),
        91,
        10,
    )


def test_game_flow_checkpoint_preserves_nested_order_and_strips_final_order():
    profile = TABLE_PROFILES["game_flow_summary"]
    order = ingest_script.resolve_checkpoint_order(
        None, "game_flow_summary", profile
    )

    query, params = build_select_query(
        table="game_flow_summary",
        profile=profile,
        pk_columns=["game_id"],
        limit=None,
        season_year=None,
        since=None,
        source_updated_before=datetime(2026, 7, 18, 13, 0),
        checkpoint_order=order,
        resume_cursor=None,
    )

    assert query.count("ORDER BY ip.inning") == 1
    assert "ORDER BY game_date DESC, game_id" not in query
    assert query.rstrip().endswith('ORDER BY "game_id" ASC')
    assert params == (datetime(2026, 7, 18, 13, 0),)


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


def test_generic_checkpoint_query_has_exact_order_limit_and_parameter_order(
    sample_since,
    sample_cutoff,
):
    order = CheckpointOrder(
        "plain_table",
        (
            CheckpointOrderField("season", "integer"),
            CheckpointOrderField("entity_id", "text"),
        ),
    )
    date_to_exclusive = datetime(2026, 8, 1).date()

    query, params = build_select_query(
        table="plain_table",
        profile={
            "season_filter_column": "season",
            "since_filter_column": "updated_at",
            "date_to_exclusive_filter_column": "game_date",
        },
        pk_columns=["legacy_id"],
        limit=25,
        season_year=2026,
        since=sample_since,
        source_updated_before=sample_cutoff,
        date_to_exclusive=date_to_exclusive,
        checkpoint_order=order,
        resume_cursor=CheckpointCursor((2025, "P100")),
    )

    assert query.as_string() == (
        'SELECT * FROM "plain_table" WHERE "season" = %s AND "updated_at" >= %s '
        'AND "updated_at" <= %s AND "game_date" < %s '
        'AND ROW("season", "entity_id") > ROW(%s, %s) '
        'ORDER BY "season" ASC, "entity_id" ASC LIMIT %s'
    )
    assert '"updated_at" IS NULL' not in query.as_string()
    assert params == (
        2026,
        sample_since,
        sample_cutoff,
        date_to_exclusive,
        2025,
        "P100",
        25,
    )


def test_generic_checkpoint_query_requires_cutoff_for_update_filtered_profile():
    order = CheckpointOrder(
        "plain_table",
        (CheckpointOrderField("id", "integer"),),
    )

    with pytest.raises(IngestCheckpointIncompatibleError):
        build_select_query(
            table="plain_table",
            profile={"season_filter_column": None, "since_filter_column": "updated_at"},
            pk_columns=["id"],
            limit=None,
            season_year=None,
            since=None,
            checkpoint_order=order,
            resume_cursor=None,
        )


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


@pytest.mark.parametrize(
    "sql_text",
    [
        "SELECT * FROM sample ORDER BY id -- ORDER BY ignored",
        "SELECT * FROM sample ORDER BY id /* ORDER BY ignored */",
        "SELECT $$ ORDER BY ignored $$ AS payload ORDER BY id",
        "SELECT $body$ ORDER BY ignored $body$ AS payload ORDER BY id",
        r"SELECT E'it\'s ORDER BY ignored' AS payload ORDER BY id",
        "SELECT 'it''s ORDER BY ignored' AS payload ORDER BY id",
        'SELECT "odd"" ORDER BY ignored" FROM sample ORDER BY id',
    ],
)
def test_top_level_order_scanner_ignores_comments_and_quoted_content(sql_text):
    expected = sql_text.rfind("ORDER BY id")

    assert ingest_script._find_top_level_keyword_positions(sql_text, "ORDER BY") == [
        expected
    ]


def test_dollar_quote_scanner_requires_identifier_boundary():
    sql_text = "SELECT foo$tag$ FROM sample WHERE active = TRUE ORDER BY id DESC"

    assert ingest_script._find_top_level_keyword_positions(sql_text, "WHERE") == [
        sql_text.index("WHERE")
    ]
    assert ingest_script._find_top_level_keyword_positions(sql_text, "ORDER BY") == [
        sql_text.index("ORDER BY")
    ]

    order = CheckpointOrder(
        "sample",
        (CheckpointOrderField("id", "integer"),),
    )
    query, params = build_select_query(
        table="sample",
        profile={
            "select_sql": sql_text,
            "season_filter_column": "season",
            "since_filter_column": None,
        },
        pk_columns=["id"],
        limit=None,
        season_year=2026,
        since=None,
        checkpoint_order=order,
        resume_cursor=None,
    )

    assert "WHERE active = TRUE\n   AND season = %s" in query
    assert "ORDER BY id DESC" not in query
    assert query.rstrip().endswith('ORDER BY "id" ASC')
    assert params == (2026,)


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
    assert profile["watermark_fields"] == ("game_updated_at",)

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


def test_game_watermark_uses_the_same_timestamp_as_its_incremental_filter():
    profile = TABLE_PROFILES["game"]
    game_updated_at = datetime(2026, 7, 18, 4, 0)

    value = ingest_script._row_updated_at(
        {
            "updated_at": datetime(2026, 7, 18, 5, 0),
            "game_updated_at": game_updated_at,
        },
        profile,
    )

    assert value == game_updated_at.replace(tzinfo=ingest_script.timezone.utc)


def test_core_schedule_profiles_declare_required_source_fields() -> None:
    assert REQUIRED_SOURCE_COLUMNS["game"] == frozenset({"game_id", "game_date"})
    assert "game_id" in REQUIRED_SOURCE_COLUMNS["game_metadata"]
    assert {"game_id", "game_date"}.issubset(
        REQUIRED_SOURCE_COLUMNS["game_summary"]
    )
