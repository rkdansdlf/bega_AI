"""
P0-3: stat_rankings 테이블 채우기
player_season_batting / player_season_pitching 에서 시즌별 부문별 TOP-10 산출

Usage:
    cd bega_AI
    source .venv/bin/activate
    python scripts/populate_stat_rankings.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTEST_CURRENT_TEST", "1")

from app.config import get_settings

import psycopg
from psycopg.rows import dict_row


# ── 메트릭 정의 ──────────────────────────────────────────────

BATTING_METRICS = [
    {
        "metric": "batting_avg",
        "column": "psb.avg",
        "sort": "DESC",
        "min_filter": "psb.plate_appearances >= 300",
        "label": "타율",
    },
    {
        "metric": "home_runs",
        "column": "psb.home_runs",
        "sort": "DESC",
        "min_filter": "psb.home_runs > 0",
        "label": "홈런",
    },
    {
        "metric": "rbi",
        "column": "psb.rbi",
        "sort": "DESC",
        "min_filter": "psb.rbi > 0",
        "label": "타점",
    },
    {
        "metric": "stolen_bases",
        "column": "psb.stolen_bases",
        "sort": "DESC",
        "min_filter": "psb.stolen_bases > 0",
        "label": "도루",
    },
    {
        "metric": "hits",
        "column": "psb.hits",
        "sort": "DESC",
        "min_filter": "psb.hits > 0",
        "label": "안타",
    },
    {
        "metric": "ops",
        "column": "psb.ops",
        "sort": "DESC",
        "min_filter": "psb.plate_appearances >= 300",
        "label": "OPS",
    },
    {
        "metric": "runs",
        "column": "psb.runs",
        "sort": "DESC",
        "min_filter": "psb.runs > 0",
        "label": "득점",
    },
]

PITCHING_METRICS = [
    {
        "metric": "era",
        "column": "psp.era",
        "sort": "ASC",
        "min_filter": "psp.innings_pitched >= 100",
        "label": "방어율",
    },
    {
        "metric": "wins",
        "column": "psp.wins",
        "sort": "DESC",
        "min_filter": "psp.wins > 0",
        "label": "승리",
    },
    {
        "metric": "saves",
        "column": "psp.saves",
        "sort": "DESC",
        "min_filter": "psp.saves > 0",
        "label": "세이브",
    },
    {
        "metric": "pitching_strikeouts",
        "column": "psp.strikeouts",
        "sort": "DESC",
        "min_filter": "psp.strikeouts > 0",
        "label": "탈삼진",
    },
    {
        "metric": "holds",
        "column": "psp.holds",
        "sort": "DESC",
        "min_filter": "psp.holds > 0",
        "label": "홀드",
    },
]

BATTING_QUERY_TEMPLATE = """
WITH ranked AS (
    SELECT
        psb.season,
        '{metric}' AS metric,
        psb.player_id::text AS entity_id,
        COALESCE(pb.name, 'Unknown') AS entity_label,
        'player' AS entity_type,
        COALESCE(psb.canonical_team_code, psb.team_code) AS team_id,
        ({column})::double precision AS value,
        RANK() OVER (PARTITION BY psb.season ORDER BY {column} {sort}) AS rnk,
        COUNT(*) OVER (PARTITION BY psb.season, {column}) AS tie_count
    FROM player_season_batting psb
    LEFT JOIN player_basic pb ON pb.player_id = psb.player_id
    WHERE psb.league = 'REGULAR'
      AND {column} IS NOT NULL
      AND {min_filter}
)
INSERT INTO stat_rankings (season, metric, entity_id, entity_label, entity_type, team_id, value, rank, is_tie, source, extra, created_at, updated_at)
SELECT
    season, metric, entity_id, entity_label, entity_type, team_id, value, rnk,
    (tie_count > 1) AS is_tie,
    'player_season_batting' AS source,
    NULL AS extra,
    NOW(), NOW()
FROM ranked
WHERE rnk <= 10
ON CONFLICT (season, metric, entity_id, entity_type) DO UPDATE SET
    entity_label = EXCLUDED.entity_label,
    team_id = EXCLUDED.team_id,
    value = EXCLUDED.value,
    rank = EXCLUDED.rank,
    is_tie = EXCLUDED.is_tie,
    source = EXCLUDED.source,
    updated_at = NOW();
"""

PITCHING_QUERY_TEMPLATE = """
WITH ranked AS (
    SELECT
        psp.season,
        '{metric}' AS metric,
        psp.player_id::text AS entity_id,
        COALESCE(pb.name, 'Unknown') AS entity_label,
        'player' AS entity_type,
        COALESCE(psp.canonical_team_code, psp.team_code) AS team_id,
        ({column})::double precision AS value,
        RANK() OVER (PARTITION BY psp.season ORDER BY {column} {sort}) AS rnk,
        COUNT(*) OVER (PARTITION BY psp.season, {column}) AS tie_count
    FROM player_season_pitching psp
    LEFT JOIN player_basic pb ON pb.player_id = psp.player_id
    WHERE psp.league = 'REGULAR'
      AND {column} IS NOT NULL
      AND {min_filter}
)
INSERT INTO stat_rankings (season, metric, entity_id, entity_label, entity_type, team_id, value, rank, is_tie, source, extra, created_at, updated_at)
SELECT
    season, metric, entity_id, entity_label, entity_type, team_id, value, rnk,
    (tie_count > 1) AS is_tie,
    'player_season_pitching' AS source,
    NULL AS extra,
    NOW(), NOW()
FROM ranked
WHERE rnk <= 10
ON CONFLICT (season, metric, entity_id, entity_type) DO UPDATE SET
    entity_label = EXCLUDED.entity_label,
    team_id = EXCLUDED.team_id,
    value = EXCLUDED.value,
    rank = EXCLUDED.rank,
    is_tie = EXCLUDED.is_tie,
    source = EXCLUDED.source,
    updated_at = NOW();
"""


def main():
    settings = get_settings()
    conn = psycopg.connect(settings.database_url)
    conn.autocommit = True

    total = 0

    # Batting metrics
    for m in BATTING_METRICS:
        sql = BATTING_QUERY_TEMPLATE.format(**m)
        cur = conn.execute(sql)
        count = cur.rowcount or 0
        total += count
        print(f"  [batting] {m['metric']:20s} ({m['label']}): {count} rows")

    # Pitching metrics
    for m in PITCHING_METRICS:
        sql = PITCHING_QUERY_TEMPLATE.format(**m)
        cur = conn.execute(sql)
        count = cur.rowcount or 0
        total += count
        print(f"  [pitching] {m['metric']:20s} ({m['label']}): {count} rows")

    # Summary
    cur = conn.execute("SELECT COUNT(*) FROM stat_rankings")
    db_total = cur.fetchone()[0]
    print(f"\n  Total stat_rankings rows in DB: {db_total}")

    cur = conn.execute(
        "SELECT MIN(season), MAX(season), COUNT(DISTINCT season), COUNT(DISTINCT metric) FROM stat_rankings"
    )
    r = cur.fetchone()
    print(f"  Seasons: {r[0]} ~ {r[1]} ({r[2]} seasons, {r[3]} metrics)")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
