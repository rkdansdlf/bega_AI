from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg
from dotenv import load_dotenv

POSTSEASON_CODES = (2, 3, 4, 5)
POSTSEASON_LABELS = {
    2: "WC",
    3: "SEMI_PO",
    4: "PO",
    5: "KS",
}


@dataclass(frozen=True)
class SeasonStage:
    season_year: int
    league_type_code: int
    season_id: int
    start_date: date


@dataclass(frozen=True)
class GameRow:
    game_id: str
    game_date: date
    season_id: Optional[int]
    raw_code: Optional[int]
    home_team: Optional[str]
    away_team: Optional[str]


@dataclass(frozen=True)
class MismatchRow:
    game_id: str
    game_date: date
    home_team: Optional[str]
    away_team: Optional[str]
    current_season_id: Optional[int]
    current_code: Optional[int]
    inferred_code: int
    target_season_id: int


def mismatch_exit_code(
    mismatches: Sequence[MismatchRow],
    *,
    apply: bool,
    fail_on_mismatch: bool,
) -> int:
    if not apply and fail_on_mismatch and mismatches:
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit or repair postseason game.season_id values from kbo_seasons start dates."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Season years to inspect, for example: --years 2025 2026",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates. Default is dry-run.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with status 2 when dry-run finds postseason season_id mismatches.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum sample mismatches to print per year.",
    )
    return parser.parse_args()


def load_environment() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env.prod")
    load_dotenv(root / ".env")


def normalize_database_url(raw_url: str) -> str:
    if raw_url.startswith("jdbc:"):
        return raw_url[len("jdbc:") :]
    return raw_url


def resolve_database_url() -> str:
    for key in ("POSTGRES_DB_URL", "OCI_DB_URL", "DATABASE_URL"):
        value = os.getenv(key)
        if value:
            return normalize_database_url(value)
    db_url = os.getenv("DB_URL")
    if db_url:
        return normalize_database_url(db_url)
    raise RuntimeError("PostgreSQL connection URL not found in environment.")


def load_season_stages(
    cur, season_years: Sequence[int]
) -> Dict[int, Dict[int, SeasonStage]]:
    cur.execute(
        """
        SELECT season_year, league_type_code, season_id, start_date
        FROM kbo_seasons
        WHERE season_year = ANY(%s)
          AND league_type_code = ANY(%s)
          AND start_date IS NOT NULL
        ORDER BY season_year, league_type_code, season_id
        """,
        (list(season_years), list(POSTSEASON_CODES)),
    )
    stages: Dict[int, Dict[int, SeasonStage]] = {}
    for season_year, league_type_code, season_id, start_date in cur.fetchall():
        year_stages = stages.setdefault(int(season_year), {})
        year_stages[int(league_type_code)] = SeasonStage(
            season_year=int(season_year),
            league_type_code=int(league_type_code),
            season_id=int(season_id),
            start_date=start_date,
        )
    return stages


def load_games(cur, season_years: Sequence[int]) -> List[GameRow]:
    cur.execute(
        """
        SELECT
            g.game_id,
            g.game_date,
            g.season_id,
            ks.league_type_code AS raw_code,
            g.home_team,
            g.away_team
        FROM game g
        LEFT JOIN kbo_seasons ks ON ks.season_id = g.season_id
        WHERE EXTRACT(YEAR FROM g.game_date)::int = ANY(%s)
          AND g.is_dummy IS NOT TRUE
          AND g.game_id NOT LIKE 'MOCK%%'
        ORDER BY g.game_date, g.game_id
        """,
        (list(season_years),),
    )
    return [
        GameRow(
            game_id=row[0],
            game_date=row[1],
            season_id=row[2],
            raw_code=row[3],
            home_team=row[4],
            away_team=row[5],
        )
        for row in cur.fetchall()
    ]


def infer_postseason_code(
    game_date: date, stages: Dict[int, SeasonStage]
) -> Optional[int]:
    for league_type_code in (5, 4, 3, 2):
        stage = stages.get(league_type_code)
        if stage is not None and game_date >= stage.start_date:
            return league_type_code
    return None


def collect_mismatches(
    games: Iterable[GameRow],
    stages_by_year: Dict[int, Dict[int, SeasonStage]],
) -> List[MismatchRow]:
    mismatches: List[MismatchRow] = []
    for game in games:
        stages = stages_by_year.get(game.game_date.year)
        if not stages:
            continue
        inferred_code = infer_postseason_code(game.game_date, stages)
        if inferred_code is None:
            continue
        target_stage = stages.get(inferred_code)
        if target_stage is None:
            continue
        if game.season_id == target_stage.season_id:
            continue
        mismatches.append(
            MismatchRow(
                game_id=game.game_id,
                game_date=game.game_date,
                home_team=game.home_team,
                away_team=game.away_team,
                current_season_id=game.season_id,
                current_code=game.raw_code,
                inferred_code=inferred_code,
                target_season_id=target_stage.season_id,
            )
        )
    return mismatches


def print_report(mismatches: Sequence[MismatchRow], limit: int) -> None:
    if not mismatches:
        print("No postseason season_id mismatches found.")
        return

    by_year: Dict[int, List[MismatchRow]] = {}
    for row in mismatches:
        by_year.setdefault(row.game_date.year, []).append(row)

    for season_year in sorted(by_year):
        rows = by_year[season_year]
        print(f"[year={season_year}] mismatch_count={len(rows)}")
        summary: Dict[Tuple[Optional[int], int], int] = {}
        for row in rows:
            key = (row.current_code, row.inferred_code)
            summary[key] = summary.get(key, 0) + 1
        for (current_code, inferred_code), count in sorted(summary.items()):
            current_label = POSTSEASON_LABELS.get(current_code, str(current_code))
            inferred_label = POSTSEASON_LABELS[inferred_code]
            print(f"  raw={current_label} -> inferred={inferred_label}: {count}")

        for row in rows[:limit]:
            current_label = POSTSEASON_LABELS.get(
                row.current_code, str(row.current_code)
            )
            inferred_label = POSTSEASON_LABELS[row.inferred_code]
            print(
                "  "
                f"{row.game_id} {row.game_date} "
                f"{row.away_team}@{row.home_team} "
                f"season_id {row.current_season_id} ({current_label}) -> "
                f"{row.target_season_id} ({inferred_label})"
            )
        if len(rows) > limit:
            print(f"  ... {len(rows) - limit} more")


def apply_updates(cur, mismatches: Sequence[MismatchRow]) -> int:
    if not mismatches:
        return 0
    cur.executemany(
        """
        UPDATE game
        SET season_id = %s
        WHERE game_id = %s
        """,
        [(row.target_season_id, row.game_id) for row in mismatches],
    )
    return len(mismatches)


def main() -> None:
    args = parse_args()
    load_environment()
    database_url = resolve_database_url()
    mismatches: List[MismatchRow] = []

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            stages_by_year = load_season_stages(cur, args.years)
            games = load_games(cur, args.years)
            mismatches = collect_mismatches(games, stages_by_year)
            print_report(mismatches, args.limit)
            if args.apply:
                updated = apply_updates(cur, mismatches)
                conn.commit()
                print(f"Applied {updated} game.season_id update(s).")
            else:
                conn.rollback()
                print("Dry-run only. No updates were applied.")
    raise SystemExit(
        mismatch_exit_code(
            mismatches,
            apply=args.apply,
            fail_on_mismatch=args.fail_on_mismatch,
        )
    )


if __name__ == "__main__":
    main()
