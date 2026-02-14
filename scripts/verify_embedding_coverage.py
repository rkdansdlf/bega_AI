#!/usr/bin/env python3
"""
Verify embedding coverage in rag_chunks after ingestion.

The script compares expected source_row_id values built from source tables against
actual rows in rag_chunks, and reports missing rows per table/year.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import psycopg
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from scripts.ingest_from_kbo import (
    CANONICAL_SOURCE_ROW_KEYS,
    TABLE_PROFILES,
    build_canonical_source_row_id,
    build_select_query,
    build_source_row_id,
    get_primary_key_columns,
)

SEASONAL_TABLES: List[str] = [
    "kbo_seasons",
    "team_history",
    "awards",
    "player_season_batting",
    "player_season_pitching",
    "game",
    "game_metadata",
    "game_inning_scores",
    "game_lineups",
    "game_batting_stats",
    "game_pitching_stats",
    "game_summary",
]

STATIC_TABLES: List[str] = [
    "teams",
    "team_franchises",
    "stadiums",
    "player_basic",
    "player_movements",
]

# 과거 source_row_id 규칙(legacy)에서 canonical 규칙으로 매핑할 때 사용하는 키.
LEGACY_SOURCE_ROW_KEYS: Dict[str, Sequence[str]] = {
    "team_history": ("team_code", "season"),
    "game": ("game_id",),
    "game_inning_scores": ("game_id", "team_side", "inning"),
    "game_lineups": ("game_id", "team_side", "batting_order", "appearance_seq"),
    "game_batting_stats": ("game_id", "team_side", "appearance_seq"),
    "game_pitching_stats": ("game_id", "team_side", "appearance_seq"),
    "game_summary": ("game_id", "summary_type", "detail_text"),
}

@dataclass(frozen=True)
class CoverageTarget:
    table: str
    year: int
    source_table: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify rag_chunks embedding coverage.")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--mode",
        choices=["all", "seasonal", "static"],
        default="all",
        help="Coverage scope to verify.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--csv-output",
        default="",
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Max missing source_row_id samples per target.",
    )
    parser.add_argument(
        "--source-db-url",
        default="",
        help="Source DB URL override. Defaults to settings.database_url.",
    )
    parser.add_argument(
        "--dest-db-url",
        default="",
        help="Destination DB URL override. Defaults to settings.database_url.",
    )
    return parser


def resolve_db_urls(args: argparse.Namespace) -> tuple[str, str]:
    settings = get_settings()
    source_db_url = args.source_db_url.strip() or settings.database_url
    dest_db_url = args.dest_db_url.strip() or settings.database_url
    return source_db_url, dest_db_url


def build_targets(mode: str, start_year: int, end_year: int) -> List[CoverageTarget]:
    targets: List[CoverageTarget] = []
    if mode in {"all", "seasonal"}:
        for year in range(start_year, end_year + 1):
            for table in SEASONAL_TABLES:
                profile = TABLE_PROFILES.get(table, {})
                source_table = str(profile.get("source_table", table))
                targets.append(
                    CoverageTarget(table=table, year=year, source_table=source_table)
                )
    if mode in {"all", "static"}:
        for table in STATIC_TABLES:
            profile = TABLE_PROFILES.get(table, {})
            source_table = str(profile.get("source_table", table))
            targets.append(CoverageTarget(table=table, year=0, source_table=source_table))
    return targets


def _recreate_temp_tables(dest_cur) -> None:
    dest_cur.execute("DROP TABLE IF EXISTS expected_ids")
    dest_cur.execute("DROP TABLE IF EXISTS actual_ids")
    dest_cur.execute("CREATE TEMP TABLE expected_ids (id text PRIMARY KEY) ON COMMIT DROP")
    dest_cur.execute("CREATE TEMP TABLE actual_ids (id text PRIMARY KEY) ON COMMIT DROP")


def _build_canonical_from_mapping(mapping: Dict[str, Any], table: str) -> Optional[str]:
    keys = CANONICAL_SOURCE_ROW_KEYS.get(table)
    if not keys:
        return None
    canonical_row: Dict[str, Any] = {}
    for key in keys:
        canonical_row[key] = mapping.get(key)
    return build_canonical_source_row_id(canonical_row, table)


def _build_legacy_from_mapping(mapping: Dict[str, Any], table: str) -> Optional[str]:
    keys = LEGACY_SOURCE_ROW_KEYS.get(table)
    if not keys:
        return None
    parts: List[str] = []
    for key in keys:
        value = mapping.get(key)
        if value in (None, ""):
            return None
        normalized = value.strip() if isinstance(value, str) else str(value)
        if table == "stadiums" and key == "stadium_id":
            maybe_stadium = build_canonical_source_row_id({"stadium_id": normalized}, table)
            if maybe_stadium is None:
                return None
            parts.append(maybe_stadium)
            continue
        parts.append(f"{key}={normalized}")
    if len(parts) == 1:
        return parts[0]
    return "|".join(parts)


def build_expected_source_row_id(
    *,
    row: Dict[str, Any],
    target: CoverageTarget,
    pk_columns: Sequence[str],
    pk_hint: Sequence[str],
) -> str:
    canonical_row_id = build_canonical_source_row_id(row, target.table)
    if canonical_row_id:
        return canonical_row_id

    return build_source_row_id(
        row=row,
        table=target.table,
        pk_columns=pk_columns,
        pk_hint=pk_hint,
    )


def _load_expected_ids(
    source_conn,
    dest_cur,
    target: CoverageTarget,
) -> tuple[int, Dict[str, str]]:
    profile = TABLE_PROFILES.get(target.table, {})
    pk_columns = get_primary_key_columns(source_conn, target.table)
    pk_hint: Sequence[str] = profile.get("pk_hint", [])
    season_year = target.year if target.year != 0 else None
    query, params = build_select_query(
        target.table,
        profile,
        pk_columns,
        limit=None,
        season_year=season_year,
        since=None,
    )

    insert_sql = "INSERT INTO expected_ids (id) VALUES (%s) ON CONFLICT DO NOTHING"
    legacy_aliases: Dict[str, str] = {}
    with source_conn.cursor(row_factory=dict_row) as src_cur:
        src_cur.execute(query, params)
        while True:
            rows = src_cur.fetchmany(1000)
            if not rows:
                break
            payload = []
            for row in rows:
                source_row_id = build_expected_source_row_id(
                    row=row,
                    target=target,
                    pk_columns=pk_columns,
                    pk_hint=pk_hint,
                )
                payload.append((source_row_id,))
                legacy_row_id = _build_legacy_from_mapping(row, target.table)
                if (
                    legacy_row_id
                    and legacy_row_id != source_row_id
                    and legacy_row_id not in legacy_aliases
                ):
                    legacy_aliases[legacy_row_id] = source_row_id
            dest_cur.executemany(insert_sql, payload)

    dest_cur.execute("SELECT count(*) FROM expected_ids")
    return int(dest_cur.fetchone()[0]), legacy_aliases


def _parse_row_id_pairs(row_id: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for token in row_id.split("|"):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        pairs[key] = value
    return pairs


def normalize_actual_source_row_id(
    raw_source_row_id: str,
    table: str,
    meta: Optional[Dict[str, Any]] = None,
    legacy_aliases: Optional[Dict[str, str]] = None,
) -> str:
    base = raw_source_row_id.split("#part", 1)[0]
    pairs = _parse_row_id_pairs(base)

    merged: Dict[str, Any] = {}
    if isinstance(meta, dict):
        merged.update(meta)
    merged.update(pairs)

    legacy_row_id = _build_legacy_from_mapping(merged, table)
    if legacy_aliases and legacy_row_id:
        mapped = legacy_aliases.get(legacy_row_id)
        if mapped:
            return mapped

    normalized = _build_canonical_from_mapping(merged, table)
    if normalized:
        return normalized

    if legacy_row_id:
        return legacy_row_id

    if legacy_aliases:
        mapped = legacy_aliases.get(base)
        if mapped:
            return mapped

    return base


def _load_actual_ids(
    dest_cur,
    target: CoverageTarget,
    legacy_aliases: Dict[str, str],
) -> int:
    read_sql = """
        SELECT source_row_id, meta
        FROM rag_chunks
        WHERE source_table = %s
    """
    params: tuple[Any, ...]
    if target.year == 0:
        params = (target.source_table,)
    else:
        read_sql += " AND season_year = %s"
        params = (target.source_table, target.year)

    insert_sql = "INSERT INTO actual_ids (id) VALUES (%s) ON CONFLICT DO NOTHING"
    with dest_cur.connection.cursor() as read_cur:
        read_cur.execute(read_sql, params)
        while True:
            rows = read_cur.fetchmany(5000)
            if not rows:
                break
            payload = []
            for row in rows:
                raw_source_row_id = row[0]
                normalized_id = normalize_actual_source_row_id(
                    row[0],
                    target.table,
                    row[1],
                    legacy_aliases,
                )
                candidate_ids = {normalized_id}

                base = raw_source_row_id.split("#part", 1)[0]
                pairs = _parse_row_id_pairs(base)
                canonical_keys = CANONICAL_SOURCE_ROW_KEYS.get(target.table)
                if canonical_keys and len(pairs) == len(canonical_keys):
                    has_all_canonical_keys = all(key in pairs for key in canonical_keys)
                    if has_all_canonical_keys:
                        canonical_from_base = _build_canonical_from_mapping(
                            pairs, target.table
                        )
                        if canonical_from_base:
                            candidate_ids.add(canonical_from_base)

                for candidate_id in candidate_ids:
                    payload.append((candidate_id,))
            dest_cur.executemany(insert_sql, payload)

    dest_cur.execute("SELECT count(*) FROM actual_ids")
    return int(dest_cur.fetchone()[0])


def verify_target(
    source_conn,
    dest_conn,
    target: CoverageTarget,
    sample_limit: int,
) -> Dict[str, Any]:
    with dest_conn.cursor() as dest_cur:
        _recreate_temp_tables(dest_cur)
        expected_count, legacy_aliases = _load_expected_ids(source_conn, dest_cur, target)
        _load_actual_ids(dest_cur, target, legacy_aliases)

        dest_cur.execute(
            """
            SELECT count(*)
            FROM expected_ids e
            JOIN actual_ids a ON a.id = e.id
            """
        )
        present_count = int(dest_cur.fetchone()[0])
        missing_count = expected_count - present_count

        dest_cur.execute(
            """
            SELECT e.id
            FROM expected_ids e
            LEFT JOIN actual_ids a ON a.id = e.id
            WHERE a.id IS NULL
            ORDER BY e.id
            LIMIT %s
            """,
            (sample_limit,),
        )
        missing_samples = [row[0] for row in dest_cur.fetchall()]

    dest_conn.commit()
    return {
        "table": target.table,
        "year": target.year,
        "expected_count": expected_count,
        "present_count": present_count,
        "missing_count": missing_count,
        "missing_samples": missing_samples,
    }


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("table | year | expected_count | present_count | missing_count | status")
    for row in rows:
        status = "OK" if row["missing_count"] == 0 else "MISSING"
        print(
            f"{row['table']} | {row['year']} | {row['expected_count']} | "
            f"{row['present_count']} | {row['missing_count']} | {status}"
        )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "table",
                "year",
                "expected_count",
                "present_count",
                "missing_count",
                "missing_samples",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "table": row["table"],
                    "year": row["year"],
                    "expected_count": row["expected_count"],
                    "present_count": row["present_count"],
                    "missing_count": row["missing_count"],
                    "missing_samples": "|".join(row["missing_samples"]),
                }
            )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.start_year > args.end_year:
        print("start-year must be less than or equal to end-year", file=sys.stderr)
        return 1
    if args.sample_limit < 0:
        print("sample-limit must be >= 0", file=sys.stderr)
        return 1

    source_db_url, dest_db_url = resolve_db_urls(args)
    targets = build_targets(args.mode, args.start_year, args.end_year)

    rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "mode": args.mode,
            "start_year": args.start_year,
            "end_year": args.end_year,
            "sample_limit": args.sample_limit,
        },
        "rows": rows,
    }

    exit_code = 0
    try:
        with psycopg.connect(source_db_url, autocommit=True) as source_conn:
            with psycopg.connect(dest_db_url) as dest_conn:
                for idx, target in enumerate(targets, start=1):
                    print(
                        f"[{idx}/{len(targets)}] verifying table={target.table} year={target.year}"
                    )
                    result = verify_target(
                        source_conn=source_conn,
                        dest_conn=dest_conn,
                        target=target,
                        sample_limit=args.sample_limit,
                    )
                    rows.append(result)
    except Exception as exc:
        report["fatal_error"] = str(exc)
        exit_code = 1

    missing_rows = [row for row in rows if row["missing_count"] > 0]
    total_expected = sum(row["expected_count"] for row in rows)
    total_present = sum(row["present_count"] for row in rows)
    total_missing = sum(row["missing_count"] for row in rows)
    summary = {
        "targets": len(rows),
        "missing_targets": len(missing_rows),
        "total_expected_count": total_expected,
        "total_present_count": total_present,
        "total_missing_count": total_missing,
    }
    report["summary"] = summary

    if rows:
        print_summary(rows)
    print(json.dumps({"summary": summary}, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"Wrote report: {output_path}")

    if args.csv_output:
        csv_path = Path(args.csv_output).expanduser().resolve()
        write_csv(csv_path, rows)
        print(f"Wrote CSV report: {csv_path}")

    if "fatal_error" in report:
        return 1
    if missing_rows:
        return 1
    if exit_code != 0:
        return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
