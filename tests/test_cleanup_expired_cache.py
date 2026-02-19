import pytest

from scripts.cleanup_expired_cache import (
    build_where_clause,
    cleanup_expired_cache,
    parse_teams_csv,
    parse_years_csv,
)
from app.tools.team_code_resolver import TeamCodeResolver


def test_parse_years_csv():
    assert parse_years_csv("2025,2024,2025") == [2024, 2025]


def test_parse_teams_csv_resolves_canonical():
    resolver = TeamCodeResolver()
    teams = parse_teams_csv("SSG,sk,두산,OB", resolver)
    assert teams == ["SSG", "DB"]


def test_build_where_clause_includes_filters():
    where, params = build_where_clause(7, [2025], ["SSG", "DB"])
    assert "updated_at < now() - make_interval(days => %s)" in where
    assert "year = ANY(%s)" in where
    assert "UPPER(team_id) = ANY(%s)" in where
    assert params == [7, [2025], ["SSG", "DB"]]


def test_cleanup_expired_cache_blocks_global_without_allow():
    with pytest.raises(ValueError):
        cleanup_expired_cache(
            retention_days=7,
            dry_run=True,
            years=None,
            teams=None,
            allow_global=False,
        )
