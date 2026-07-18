"""Populate team-level fielding and baserunning aggregates from internal tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg

from app.config import get_settings

DOUBLE_PRECISION_COLUMNS = {
    "team_season_fielding": ("fielding_pct", "range_factor_per_game"),
    "team_season_baserunning": ("sb_success_rate",),
}

NULLABLE_SOURCE_GAP_COLUMNS = {
    "team_season_fielding": ("triple_plays",),
    "team_season_baserunning": ("extra_bases_taken", "bunt_hits"),
}


TEAM_FIELDING_UPSERT_SQL = """
WITH fielding AS (
    SELECT
        psf.year AS season,
        psf.team_id AS team_code,
        SUM(COALESCE(psf.errors, 0))::integer AS errors,
        SUM(COALESCE(psf.double_plays, 0))::integer AS double_plays,
        SUM(COALESCE(psf.putouts, 0))::integer AS putouts,
        SUM(COALESCE(psf.assists, 0))::integer AS assists,
        ROUND(SUM(COALESCE(psf.innings, 0)))::integer AS def_innings
    FROM player_season_fielding psf
    WHERE psf.team_id IS NOT NULL
      AND (%s IS NULL OR psf.year = %s)
    GROUP BY psf.year, psf.team_id
),
team_games AS (
    SELECT
        tsb.season,
        tsb.team_id AS team_code,
        MAX(tsb.games) AS team_games
    FROM team_season_batting tsb
    WHERE tsb.league = 'REGULAR'
      AND (%s IS NULL OR tsb.season = %s)
    GROUP BY tsb.season, tsb.team_id
),
prepared AS (
    SELECT
        f.season,
        f.team_code,
        f.errors,
        f.double_plays,
        NULL::integer AS triple_plays,
        (f.putouts + f.assists + f.errors)::integer AS total_chances,
        f.putouts,
        f.assists,
        f.def_innings,
        CASE
            WHEN (f.putouts + f.assists + f.errors) > 0
                THEN (f.putouts + f.assists)::double precision
                     / (f.putouts + f.assists + f.errors)
            ELSE NULL
        END AS fielding_pct,
        CASE
            WHEN tg.team_games > 0
                THEN (f.putouts + f.assists)::double precision / tg.team_games
            ELSE NULL
        END AS range_factor_per_game
    FROM fielding f
    LEFT JOIN team_games tg
      ON tg.season = f.season
     AND tg.team_code = f.team_code
)
INSERT INTO team_season_fielding (
    season, team_code, errors, double_plays, triple_plays, total_chances,
    putouts, assists, def_innings, fielding_pct, range_factor_per_game,
    created_at, updated_at
)
SELECT
    season, team_code, errors, double_plays, triple_plays, total_chances,
    putouts, assists, def_innings, fielding_pct, range_factor_per_game,
    NOW(), NOW()
FROM prepared
ON CONFLICT (season, team_code) DO UPDATE SET
    errors = EXCLUDED.errors,
    double_plays = EXCLUDED.double_plays,
    triple_plays = EXCLUDED.triple_plays,
    total_chances = EXCLUDED.total_chances,
    putouts = EXCLUDED.putouts,
    assists = EXCLUDED.assists,
    def_innings = EXCLUDED.def_innings,
    fielding_pct = EXCLUDED.fielding_pct,
    range_factor_per_game = EXCLUDED.range_factor_per_game,
    updated_at = NOW();
"""


TEAM_BASERUNNING_UPSERT_SQL = """
WITH player_br AS (
    SELECT
        psbr.year AS season,
        psbr.team_id AS team_code,
        SUM(COALESCE(psbr.stolen_bases, 0))::integer AS stolen_bases,
        SUM(COALESCE(psbr.caught_stealing, 0))::integer AS caught_stealing,
        SUM(COALESCE(psbr.out_on_base, 0))::integer AS out_on_base
    FROM player_season_baserunning psbr
    WHERE psbr.team_id IS NOT NULL
      AND (%s IS NULL OR psbr.year = %s)
    GROUP BY psbr.year, psbr.team_id
),
game_br AS (
    SELECT
        EXTRACT(YEAR FROM g.game_date)::integer AS season,
        COALESCE(gbs.canonical_team_code, gbs.team_code) AS team_code,
        SUM(COALESCE(gbs.sacrifice_hits, 0))::integer AS sacrifice_hits,
        SUM(COALESCE(gbs.sacrifice_flies, 0))::integer AS sacrifice_flies
    FROM game_batting_stats gbs
    JOIN game g ON g.game_id = gbs.game_id
    WHERE g.game_date IS NOT NULL
      AND COALESCE(gbs.canonical_team_code, gbs.team_code) IS NOT NULL
      AND (%s IS NULL OR EXTRACT(YEAR FROM g.game_date)::integer = %s)
    GROUP BY EXTRACT(YEAR FROM g.game_date)::integer, COALESCE(gbs.canonical_team_code, gbs.team_code)
),
prepared AS (
    SELECT
        COALESCE(p.season, g.season) AS season,
        COALESCE(p.team_code, g.team_code) AS team_code,
        COALESCE(p.stolen_bases, 0) AS stolen_bases,
        COALESCE(p.caught_stealing, 0) AS caught_stealing,
        CASE
            WHEN (COALESCE(p.stolen_bases, 0) + COALESCE(p.caught_stealing, 0)) > 0
                THEN COALESCE(p.stolen_bases, 0)::double precision
                     / (COALESCE(p.stolen_bases, 0) + COALESCE(p.caught_stealing, 0))
            ELSE 0.0
        END AS sb_success_rate,
        NULL::integer AS extra_bases_taken,
        COALESCE(p.out_on_base, 0) AS out_on_base,
        COALESCE(g.sacrifice_hits, 0) AS sacrifice_hits,
        COALESCE(g.sacrifice_flies, 0) AS sacrifice_flies,
        NULL::integer AS bunt_hits
    FROM player_br p
    FULL OUTER JOIN game_br g
      ON g.season = p.season
     AND g.team_code = p.team_code
)
INSERT INTO team_season_baserunning (
    season, team_code, stolen_bases, caught_stealing, sb_success_rate,
    extra_bases_taken, out_on_base, sacrifice_hits, sacrifice_flies, bunt_hits,
    created_at, updated_at
)
SELECT
    season, team_code, stolen_bases, caught_stealing, sb_success_rate,
    extra_bases_taken, out_on_base, sacrifice_hits, sacrifice_flies, bunt_hits,
    NOW(), NOW()
FROM prepared
WHERE season IS NOT NULL
  AND team_code IS NOT NULL
ON CONFLICT (season, team_code) DO UPDATE SET
    stolen_bases = EXCLUDED.stolen_bases,
    caught_stealing = EXCLUDED.caught_stealing,
    sb_success_rate = EXCLUDED.sb_success_rate,
    extra_bases_taken = EXCLUDED.extra_bases_taken,
    out_on_base = EXCLUDED.out_on_base,
    sacrifice_hits = EXCLUDED.sacrifice_hits,
    sacrifice_flies = EXCLUDED.sacrifice_flies,
    bunt_hits = EXCLUDED.bunt_hits,
    updated_at = NOW();
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate team_season_fielding and team_season_baserunning."
    )
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _column_type(conn: Any, table_name: str, column_name: str) -> str | None:
    row = conn.execute(
        """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
          AND column_name = %s
        """,
        (table_name, column_name),
    ).fetchone()
    if not row:
        return None
    return row[0] if not isinstance(row, dict) else row.get("data_type")


def _column_is_nullable(conn: Any, table_name: str, column_name: str) -> bool | None:
    row = conn.execute(
        """
        SELECT is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
          AND column_name = %s
        """,
        (table_name, column_name),
    ).fetchone()
    if not row:
        return None
    value = row[0] if not isinstance(row, dict) else row.get("is_nullable")
    return str(value).upper() == "YES"


def build_double_precision_migration_sql(
    table_name: str, column_name: str, data_type: str | None
) -> str | None:
    if data_type is None or data_type == "double precision":
        return None
    return (
        f"ALTER TABLE {table_name} "
        f"ALTER COLUMN {column_name} TYPE double precision "
        f"USING {column_name}::double precision"
    )


def build_drop_not_null_migration_sql(
    table_name: str, column_name: str, is_nullable: bool | None
) -> str | None:
    if is_nullable is None or is_nullable:
        return None
    return f"ALTER TABLE {table_name} ALTER COLUMN {column_name} DROP NOT NULL"


def ensure_double_precision_columns(conn: Any, *, dry_run: bool) -> list[str]:
    statements: list[str] = []
    for table_name, columns in DOUBLE_PRECISION_COLUMNS.items():
        for column_name in columns:
            statement = build_double_precision_migration_sql(
                table_name,
                column_name,
                _column_type(conn, table_name, column_name),
            )
            if statement is None:
                continue
            statements.append(statement)
            if not dry_run:
                conn.execute(statement)
    return statements


def ensure_nullable_source_gap_columns(conn: Any, *, dry_run: bool) -> list[str]:
    statements: list[str] = []
    for table_name, columns in NULLABLE_SOURCE_GAP_COLUMNS.items():
        for column_name in columns:
            statement = build_drop_not_null_migration_sql(
                table_name,
                column_name,
                _column_is_nullable(conn, table_name, column_name),
            )
            if statement is None:
                continue
            statements.append(statement)
            if not dry_run:
                conn.execute(statement)
    return statements


def populate(conn: Any, *, season: int | None, dry_run: bool) -> dict[str, Any]:
    migrations = ensure_double_precision_columns(conn, dry_run=dry_run)
    nullable_migrations = ensure_nullable_source_gap_columns(conn, dry_run=dry_run)
    params = (season, season, season, season)
    if dry_run:
        return {
            "dry_run": True,
            "season": season,
            "migrations": migrations,
            "nullable_migrations": nullable_migrations,
            "fielding_sql_ready": True,
            "baserunning_sql_ready": True,
        }

    fielding_result = conn.execute(TEAM_FIELDING_UPSERT_SQL, params)
    baserunning_result = conn.execute(TEAM_BASERUNNING_UPSERT_SQL, params)
    return {
        "dry_run": False,
        "season": season,
        "migrations": migrations,
        "nullable_migrations": nullable_migrations,
        "fielding_rows": fielding_result.rowcount,
        "baserunning_rows": baserunning_result.rowcount,
    }


def main() -> None:
    args = parse_args()
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("SET statement_timeout TO 0")
        result = populate(conn, season=args.season, dry_run=args.dry_run)
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
    print(result)


if __name__ == "__main__":
    main()
