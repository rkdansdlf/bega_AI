#!/usr/bin/env python3
"""Repair known trusted player identity duplicates in internal DB tables."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTEST_CURRENT_TEST", "1")

import psycopg

from app.config import get_settings

CANONICAL_PLAYER_ID = 56632
DUPLICATE_PLAYER_ID = 59359


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonicalize known duplicate player IDs in trusted DB tables."
    )
    parser.add_argument("--apply", action="store_true", help="Commit changes.")
    return parser.parse_args()


def _execute(conn: psycopg.Connection, label: str, sql: str, params: tuple = ()) -> int:
    cur = conn.execute(sql, params)
    count = cur.rowcount if cur.rowcount is not None else 0
    print(f"{label}: {count}")
    return count


def main() -> int:
    args = parse_args()
    settings = get_settings()

    with psycopg.connect(settings.database_url) as conn:
        _execute(
            conn,
            "player_season_batting delete duplicate rows with canonical present",
            """
            DELETE FROM player_season_batting d
            USING player_season_batting c
            WHERE d.player_id = %s
              AND c.player_id = %s
              AND c.season = d.season
              AND COALESCE(c.league, '') = COALESCE(d.league, '')
              AND COALESCE(c.level, '') = COALESCE(d.level, '')
            """,
            (DUPLICATE_PLAYER_ID, CANONICAL_PLAYER_ID),
        )
        _execute(
            conn,
            "player_season_pitching delete duplicate rows with canonical present",
            """
            DELETE FROM player_season_pitching d
            USING player_season_pitching c
            WHERE d.player_id = %s
              AND c.player_id = %s
              AND c.season = d.season
              AND COALESCE(c.league, '') = COALESCE(d.league, '')
              AND COALESCE(c.level, '') = COALESCE(d.level, '')
            """,
            (DUPLICATE_PLAYER_ID, CANONICAL_PLAYER_ID),
        )

        for table, column in (
            ("player_season_batting", "player_id"),
            ("player_season_pitching", "player_id"),
            ("game_lineups", "player_id"),
            ("game_batting_stats", "player_id"),
            ("game_pitching_stats", "player_id"),
            ("game_summary", "player_id"),
            ("team_daily_roster", "player_id"),
            ("game_events", "batter_id"),
            ("game_events", "pitcher_id"),
        ):
            _execute(
                conn,
                f"{table}.{column} canonicalize",
                f"UPDATE {table} SET {column} = %s WHERE {column} = %s",
                (CANONICAL_PLAYER_ID, DUPLICATE_PLAYER_ID),
            )

        _execute(
            conn,
            "stat_rankings delete duplicate rows with canonical present",
            """
            DELETE FROM stat_rankings d
            USING stat_rankings c
            WHERE d.entity_id = %s::text
              AND c.entity_id = %s::text
              AND c.season = d.season
              AND c.metric = d.metric
              AND c.entity_type = d.entity_type
            """,
            (DUPLICATE_PLAYER_ID, CANONICAL_PLAYER_ID),
        )
        _execute(
            conn,
            "stat_rankings canonicalize entity_id",
            "UPDATE stat_rankings SET entity_id = %s::text WHERE entity_id = %s::text",
            (CANONICAL_PLAYER_ID, DUPLICATE_PLAYER_ID),
        )

        _execute(
            conn,
            "rag_chunks canonicalize player_id",
            "UPDATE rag_chunks SET player_id = %s::text WHERE player_id = %s::text",
            (CANONICAL_PLAYER_ID, DUPLICATE_PLAYER_ID),
        )
        _execute(
            conn,
            "rag_chunks canonicalize meta entity_id",
            """
            UPDATE rag_chunks
            SET meta = jsonb_set(meta, '{entity_id}', to_jsonb(%s::text), false)
            WHERE meta->>'entity_id' = %s::text
            """,
            (CANONICAL_PLAYER_ID, DUPLICATE_PLAYER_ID),
        )
        _execute(
            conn,
            "rag_chunks canonicalize metadata entity_id",
            """
            UPDATE rag_chunks
            SET metadata = jsonb_set(metadata, '{entity_id}', to_jsonb(%s::text), false)
            WHERE metadata->>'entity_id' = %s::text
            """,
            (CANONICAL_PLAYER_ID, DUPLICATE_PLAYER_ID),
        )

        print("player_basic duplicate row retained: referenced by players FK")

        if args.apply:
            conn.commit()
            print("committed")
        else:
            conn.rollback()
            print("dry-run rollback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
