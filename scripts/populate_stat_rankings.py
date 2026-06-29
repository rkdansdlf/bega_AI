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
import argparse
from datetime import date
from math import ceil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTEST_CURRENT_TEST", "1")

from app.config import get_settings

import psycopg
from psycopg.rows import dict_row

# ── 메트릭 정의 ──────────────────────────────────────────────

IN_PROGRESS_BATTING_RATE_METRICS = frozenset({"batting_avg", "ops"})
IN_PROGRESS_PITCHING_RATE_METRICS = frozenset({"era", "whip"})

BATTING_METRICS = [
    {
        "metric": "batting_avg",
        "aliases": ("avg",),
        "column": "psb.avg",
        "sort": "DESC",
        "min_filter": "psb.plate_appearances >= 300",
        "label": "타율",
    },
    {
        "metric": "home_runs",
        "aliases": ("batting_home_runs",),
        "column": "psb.home_runs",
        "sort": "DESC",
        "min_filter": "psb.home_runs > 0",
        "label": "홈런",
    },
    {
        "metric": "rbi",
        "aliases": ("batting_rbi",),
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
        "aliases": ("batting_hits",),
        "column": "psb.hits",
        "sort": "DESC",
        "min_filter": "psb.hits > 0",
        "label": "안타",
    },
    {
        "metric": "ops",
        "aliases": ("batting_ops",),
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
        "aliases": ("pitching_era",),
        "column": "psp.era",
        "sort": "ASC",
        "min_filter": "psp.innings_pitched >= 100",
        "label": "방어율",
    },
    {
        "metric": "whip",
        "aliases": ("pitching_whip",),
        "column": "psp.whip",
        "sort": "ASC",
        "min_filter": "psp.innings_pitched >= 100",
        "label": "WHIP",
    },
    {
        "metric": "wins",
        "aliases": ("pitching_wins",),
        "column": "psp.wins",
        "sort": "DESC",
        "min_filter": "psp.wins > 0",
        "label": "승리",
    },
    {
        "metric": "saves",
        "aliases": ("pitching_saves",),
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
        "aliases": ("pitching_holds",),
        "column": "psp.holds",
        "sort": "DESC",
        "min_filter": "psp.holds > 0",
        "label": "홀드",
    },
]

BATTING_QUERY_TEMPLATE = """
WITH normalized AS (
    SELECT
        psb.*,
        CASE psb.player_id WHEN 59359 THEN 56632 ELSE psb.player_id END AS canonical_player_id,
        CASE WHEN psb.player_id = 59359 THEN 1 ELSE 0 END AS duplicate_priority
    FROM player_season_batting psb
    WHERE psb.league = 'REGULAR'
      {season_filter}
),
latest AS (
    SELECT DISTINCT ON (psb.season, psb.canonical_player_id, COALESCE(psb.canonical_team_code, psb.team_code))
        psb.*
    FROM normalized psb
    WHERE {column} IS NOT NULL
      AND {min_filter}
    ORDER BY
        psb.season,
        psb.canonical_player_id,
        COALESCE(psb.canonical_team_code, psb.team_code),
        psb.duplicate_priority,
        psb.plate_appearances DESC NULLS LAST,
        psb.updated_at DESC NULLS LAST,
        psb.id DESC
),
ranked AS (
    SELECT
        psb.season,
        '{metric}' AS metric,
        psb.canonical_player_id::text AS entity_id,
        COALESCE(pb.name, 'Unknown') AS entity_label,
        'PLAYER' AS entity_type,
        COALESCE(psb.canonical_team_code, psb.team_code) AS team_id,
        ({column})::double precision AS value,
        RANK() OVER (PARTITION BY psb.season ORDER BY {column} {sort}) AS rnk,
        COUNT(*) OVER (PARTITION BY psb.season, {column}) AS tie_count
    FROM latest psb
    JOIN player_basic pb ON pb.player_id = psb.canonical_player_id
    WHERE pb.name IS NOT NULL
      AND pb.name <> ''
      AND LOWER(pb.name) <> 'unknown'
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
WITH normalized AS (
    SELECT
        psp.*,
        CASE psp.player_id WHEN 59359 THEN 56632 ELSE psp.player_id END AS canonical_player_id,
        CASE WHEN psp.player_id = 59359 THEN 1 ELSE 0 END AS duplicate_priority
    FROM player_season_pitching psp
    WHERE psp.league = 'REGULAR'
      {season_filter}
),
latest AS (
    SELECT DISTINCT ON (psp.season, psp.canonical_player_id, COALESCE(psp.canonical_team_code, psp.team_code))
        psp.*
    FROM normalized psp
    WHERE {column} IS NOT NULL
      AND {min_filter}
    ORDER BY
        psp.season,
        psp.canonical_player_id,
        COALESCE(psp.canonical_team_code, psp.team_code),
        psp.duplicate_priority,
        psp.innings_pitched DESC NULLS LAST,
        psp.updated_at DESC NULLS LAST,
        psp.id DESC
),
ranked AS (
    SELECT
        psp.season,
        '{metric}' AS metric,
        psp.canonical_player_id::text AS entity_id,
        COALESCE(pb.name, 'Unknown') AS entity_label,
        'PLAYER' AS entity_type,
        COALESCE(psp.canonical_team_code, psp.team_code) AS team_id,
        ({column})::double precision AS value,
        RANK() OVER (PARTITION BY psp.season ORDER BY {column} {sort}) AS rnk,
        COUNT(*) OVER (PARTITION BY psp.season, {column}) AS tie_count
    FROM latest psp
    JOIN player_basic pb ON pb.player_id = psp.canonical_player_id
    WHERE pb.name IS NOT NULL
      AND pb.name <> ''
      AND LOWER(pb.name) <> 'unknown'
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Populate stat_rankings from trusted internal season stat tables."
    )
    parser.add_argument("--season", type=int, default=None)
    return parser.parse_args()


def _metric_names(metrics):
    names = []
    for metric in metrics:
        names.append(metric["metric"])
        names.extend(metric.get("aliases", ()))
    return names


def _iter_metric_variants(metric):
    yield metric["metric"]
    yield from metric.get("aliases", ())


def is_in_progress_season(season: int | None, *, today: date | None = None) -> bool:
    if season is None:
        return False
    return int(season) == (today or date.today()).year


def in_progress_batting_rate_pa_threshold(max_team_games: int | None) -> int | None:
    if not max_team_games:
        return None
    return int(ceil(int(max_team_games) * 3.1))


def in_progress_pitching_rate_ip_threshold(max_team_games: int | None) -> int | None:
    if not max_team_games:
        return None
    return int(max_team_games)


def apply_in_progress_metric_filter(
    metric: dict,
    *,
    season: int | None,
    max_team_games: int | None,
    today: date | None = None,
) -> dict:
    resolved = dict(metric)
    if not is_in_progress_season(season, today=today):
        return resolved

    metric_name = str(metric.get("metric") or "")
    if metric_name in IN_PROGRESS_BATTING_RATE_METRICS:
        threshold = in_progress_batting_rate_pa_threshold(max_team_games)
        if threshold is not None:
            resolved["min_filter"] = f"psb.plate_appearances >= {threshold}"
    elif metric_name in IN_PROGRESS_PITCHING_RATE_METRICS:
        threshold = in_progress_pitching_rate_ip_threshold(max_team_games)
        if threshold is not None:
            resolved["min_filter"] = f"psp.innings_pitched >= {threshold}"
    return resolved


def resolve_max_team_games(conn, season: int | None) -> int | None:
    if season is None:
        return None
    row = conn.execute(
        """
        SELECT COALESCE(
            (SELECT MAX(games) FROM team_season_batting WHERE season = %s AND league = 'REGULAR'),
            (SELECT MAX(games) FROM team_season_pitching WHERE season = %s AND league = 'REGULAR')
        ) AS max_games
        """,
        (season, season),
    ).fetchone()
    if not row:
        return None
    value = row[0] if not isinstance(row, dict) else row.get("max_games")
    return int(value) if value is not None else None


def main():
    args = parse_args()
    settings = get_settings()
    conn = psycopg.connect(settings.database_url)
    conn.autocommit = True

    total = 0
    season_filter = f"AND psb.season = {int(args.season)}" if args.season else ""
    pitching_season_filter = (
        f"AND psp.season = {int(args.season)}" if args.season else ""
    )
    managed_metrics = _metric_names(BATTING_METRICS) + _metric_names(PITCHING_METRICS)
    max_team_games = resolve_max_team_games(conn, args.season)
    if args.season and is_in_progress_season(args.season):
        print(
            f"  In-progress eligibility: season={args.season}, max_team_games={max_team_games}"
        )
    if args.season:
        conn.execute(
            "DELETE FROM stat_rankings WHERE season = %s AND metric = ANY(%s)",
            (args.season, managed_metrics),
        )

    # Batting metrics
    for m in BATTING_METRICS:
        metric_config = apply_in_progress_metric_filter(
            m, season=args.season, max_team_games=max_team_games
        )
        for metric_name in _iter_metric_variants(metric_config):
            sql = BATTING_QUERY_TEMPLATE.format(
                **{
                    **metric_config,
                    "metric": metric_name,
                    "season_filter": season_filter,
                }
            )
            cur = conn.execute(sql)
            count = cur.rowcount or 0
            total += count
            print(
                f"  [batting] {metric_name:20s} ({metric_config['label']}): {count} rows"
            )

    # Pitching metrics
    for m in PITCHING_METRICS:
        metric_config = apply_in_progress_metric_filter(
            m, season=args.season, max_team_games=max_team_games
        )
        for metric_name in _iter_metric_variants(metric_config):
            sql = PITCHING_QUERY_TEMPLATE.format(
                **{
                    **metric_config,
                    "metric": metric_name,
                    "season_filter": pitching_season_filter,
                }
            )
            cur = conn.execute(sql)
            count = cur.rowcount or 0
            total += count
            print(
                f"  [pitching] {metric_name:20s} ({metric_config['label']}): {count} rows"
            )

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
