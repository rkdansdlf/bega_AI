#!/usr/bin/env python3
"""
Re-embed only missing rows reported by verify_embedding_coverage.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set
import sys

os.environ.setdefault("PYTEST_CURRENT_TEST", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psycopg
    from psycopg.rows import dict_row
except ModuleNotFoundError as exc:
    psycopg = None
    dict_row = None
    _PSYCOPG_IMPORT_ERROR = exc

from app.config import get_settings
from app.core.chunking import smart_chunks
from scripts.ingest_from_kbo import (
    ChunkPayload,
    TABLE_PROFILES,
    UPSERT_SQL,
    build_static_profile_chunk_payloads,
    build_static_source_row_prefix,
    build_content,
    build_select_query,
    build_title,
    coerce_int,
    first_value,
    flush_chunks,
    get_primary_key_columns,
)
from scripts.sync_rag_chunks import _load_settings_from_env_file

if TYPE_CHECKING:
    from scripts.verify_embedding_coverage import CoverageTarget


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required to run reembed_missing_rows.py. "
            "Install dependencies (e.g. pip install psycopg[binary]) and retry. "
            f"Detail: {_PSYCOPG_IMPORT_ERROR}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-embed only missing coverage rows.")
    parser.add_argument(
        "--report-path",
        default="logs/embedding_coverage.json",
        help="Coverage JSON report path.",
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--read-batch-size", type=int, default=500)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--commit-interval", type=int, default=500)
    parser.add_argument(
        "--source-env-file",
        default="",
        help="Env file used to resolve source DB config when --source-db-url is omitted.",
    )
    parser.add_argument(
        "--dest-env-file",
        default="",
        help="Env file used to resolve destination DB config when settings.database_url should be overridden.",
    )
    parser.add_argument(
        "--source-db-url",
        default="",
        help="Source PostgreSQL 연결 URL override (기본값: settings/source env).",
    )
    parser.add_argument(
        "--supabase-url",
        default="",
        help="[Deprecated] --source-db-url 사용 권장",
    )
    return parser


def load_missing_targets(
    report_path: Path,
    start_year: int,
    end_year: int,
) -> List[CoverageTarget]:
    from scripts.verify_embedding_coverage import CoverageTarget

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    targets: List[CoverageTarget] = []
    for row in payload.get("rows", []):
        missing_count = int(row.get("missing_count", 0))
        if missing_count <= 0:
            continue
        year = int(row["year"])
        if year != 0 and (year < start_year or year > end_year):
            continue
        table = str(row["table"])
        profile = TABLE_PROFILES.get(table, {})
        source_table = str(profile.get("source_table", table))
        targets.append(
            CoverageTarget(table=table, year=year, source_table=source_table)
        )
    targets.sort(key=lambda t: (t.year, t.table))
    return targets


def collect_missing_ids(
    source_conn: psycopg.Connection,
    dest_conn: psycopg.Connection,
    target: CoverageTarget,
) -> Set[str]:
    from scripts.verify_embedding_coverage import (
        _load_actual_ids,
        _load_expected_ids,
        _recreate_temp_tables,
    )

    with dest_conn.cursor() as dest_cur:
        _recreate_temp_tables(dest_cur)
        _, legacy_aliases = _load_expected_ids(source_conn, dest_cur, target)
        _load_actual_ids(dest_cur, target, legacy_aliases)
        dest_cur.execute("""
            SELECT e.id
            FROM expected_ids e
            LEFT JOIN actual_ids a ON a.id = e.id
            WHERE a.id IS NULL
            """)
        missing_ids = {row[0] for row in dest_cur.fetchall()}
    dest_conn.commit()
    return missing_ids


def _append_chunks(
    *,
    buffer: List[ChunkPayload],
    settings: Any,
    source_table: str,
    source_row_id: str,
    title: str,
    content: str,
    season_year: int,
    season_id: Optional[int],
    league_type_code: int,
    team_id: Optional[str],
    player_id: Optional[str],
    meta: Dict[str, Any],
) -> None:
    chunks = smart_chunks(content, settings=settings)
    if not chunks:
        return
    if len(chunks) == 1:
        buffer.append(
            ChunkPayload(
                table=source_table,
                source_row_id=source_row_id,
                title=title,
                content=chunks[0],
                season_year=season_year,
                season_id=season_id,
                league_type_code=league_type_code,
                team_id=team_id,
                player_id=player_id,
                meta=meta,
            )
        )
        return

    for idx, chunk in enumerate(chunks, start=1):
        buffer.append(
            ChunkPayload(
                table=source_table,
                source_row_id=f"{source_row_id}#part{idx}",
                title=f"{title} (분할 {idx})",
                content=chunk,
                season_year=season_year,
                season_id=season_id,
                league_type_code=league_type_code,
                team_id=team_id,
                player_id=player_id,
                meta=meta,
            )
        )


def _prepare_dest_cursor(write_cur: Any) -> None:
    write_cur.execute("SET search_path TO public, extensions, security;")
    write_cur.execute("SET statement_timeout TO 0;")


def _copy_existing_chunk_rows(
    *,
    source_conn: psycopg.Connection,
    dest_conn: psycopg.Connection,
    target: CoverageTarget,
    missing_ids: Set[str],
) -> Set[str]:
    if not missing_ids:
        return set()

    with source_conn.cursor(row_factory=dict_row) as source_cur:
        source_cur.execute(
            """
            SELECT
                meta,
                season_year,
                season_id,
                league_type_code,
                team_id,
                player_id,
                source_table,
                source_row_id,
                title,
                content,
                embedding::text AS embedding_text
            FROM rag_chunks
            WHERE source_table = %s
              AND source_row_id = ANY(%s)
            """,
            (target.source_table, list(missing_ids)),
        )
        rows = [dict(row) for row in source_cur.fetchall()]

    if not rows:
        return set()

    payload: List[tuple[Any, ...]] = []
    copied_ids: Set[str] = set()
    for row in rows:
        copied_ids.add(str(row["source_row_id"]))
        meta = row.get("meta")
        payload.append(
            (
                (
                    json.dumps(meta, ensure_ascii=False, default=str)
                    if meta is not None
                    else None
                ),
                row.get("season_year"),
                row.get("season_id"),
                row.get("league_type_code"),
                row.get("team_id"),
                row.get("player_id"),
                row.get("source_table"),
                row.get("source_row_id"),
                row.get("title"),
                row.get("content"),
                row.get("embedding_text"),
            )
        )

    with dest_conn.cursor() as write_cur:
        _prepare_dest_cursor(write_cur)
        write_cur.executemany(UPSERT_SQL, payload)
    dest_conn.commit()
    return copied_ids


def build_static_target_payloads(
    target: CoverageTarget,
    *,
    settings: Any,
) -> List[ChunkPayload]:
    profile = TABLE_PROFILES.get(target.table, {})
    if not profile.get("source_file"):
        raise ValueError(
            f"Coverage target '{target.table}' is not a static source_file profile"
        )
    return build_static_profile_chunk_payloads(
        target.table,
        profile,
        settings=settings,
    )


def reembed_target_missing_rows(
    *,
    source_conn: psycopg.Connection,
    dest_conn: psycopg.Connection,
    target: CoverageTarget,
    missing_ids: Set[str],
    embed_batch_size: int,
    read_batch_size: int,
    max_concurrency: int,
    commit_interval: int,
) -> Dict[str, int]:
    from scripts.verify_embedding_coverage import build_expected_source_row_id

    table = target.table
    profile = TABLE_PROFILES.get(table, {})
    settings = get_settings()
    copied_ids: Set[str] = set()
    if not profile.get("source_file"):
        copied_ids = _copy_existing_chunk_rows(
            source_conn=source_conn,
            dest_conn=dest_conn,
            target=target,
            missing_ids=missing_ids,
        )
    remaining_missing_ids = set(missing_ids) - copied_ids
    if not remaining_missing_ids:
        return {
            "missing_ids": len(missing_ids),
            "matched_rows": len(copied_ids),
            "flushed_chunks": len(copied_ids),
        }

    if profile.get("source_file"):
        payloads = build_static_target_payloads(target, settings=settings)
        static_prefix = build_static_source_row_prefix(table, profile)
        with dest_conn.cursor() as write_cur:
            _prepare_dest_cursor(write_cur)
            write_cur.execute(
                """
                DELETE FROM rag_chunks
                WHERE source_table = %s
                  AND (source_row_id = %s OR source_row_id LIKE %s)
                """,
                (
                    target.source_table,
                    static_prefix,
                    f"{static_prefix}#part%",
                ),
            )
            buffer = list(payloads)
            flushed_chunks = flush_chunks(
                write_cur,
                settings,
                buffer,
                max_concurrency=max_concurrency,
                commit_interval=commit_interval,
                stats={"since_commit": 0},
                skip_embedding=False,
            )
        dest_conn.commit()
        return {
            "missing_ids": len(missing_ids),
            "matched_rows": 1 if payloads else 0,
            "flushed_chunks": flushed_chunks,
        }

    pk_columns = get_primary_key_columns(source_conn, table)
    pk_hint: Sequence[str] = profile.get("pk_hint", [])
    season_year_filter = target.year if target.year != 0 else None
    query, params = build_select_query(
        table,
        profile,
        pk_columns,
        limit=None,
        season_year=season_year_filter,
        since=None,
    )
    source_table = target.source_table
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    buffer: List[ChunkPayload] = []
    stats: Dict[str, Any] = {"since_commit": 0}
    matched_rows = 0
    flushed_chunks = 0
    seen_source_row_ids: Set[str] = set()

    with (
        source_conn.cursor(row_factory=dict_row) as read_cur,
        dest_conn.cursor() as write_cur,
    ):
        _prepare_dest_cursor(write_cur)
        read_cur.execute(query, params)

        while True:
            rows = read_cur.fetchmany(read_batch_size)
            if not rows:
                break
            for raw_row in rows:
                row = dict(raw_row)
                source_row_id = build_expected_source_row_id(
                    row=row,
                    target=target,
                    pk_columns=pk_columns,
                    pk_hint=pk_hint,
                )
                if source_row_id not in remaining_missing_ids:
                    continue
                if source_row_id in seen_source_row_ids:
                    continue
                seen_source_row_ids.add(source_row_id)

                matched_rows += 1
                title = build_title(row, table, source_row_id, profile)
                renderer = profile.get("renderer")
                if renderer:
                    enriched_row = dict(row)
                    enriched_row["source_table"] = table
                    enriched_row["source_row_id"] = source_row_id
                    content = renderer(
                        enriched_row,
                        league_avg=None,
                        percentiles=None,
                        today_str=today_str,
                    )
                else:
                    content = build_content(row, table, source_row_id, profile)

                season_year_value = coerce_int(
                    first_value(row, ["season_year", "season", "year"])
                )
                if season_year_value is None:
                    season_year_value = 0
                season_id = coerce_int(
                    first_value(row, ["season_id", "season_lookup_id"])
                )
                league_type_code = coerce_int(
                    first_value(row, ["league_type_code", "league_type", "league"])
                )
                if league_type_code is None:
                    league_type_code = 0

                team_id_raw = first_value(
                    row,
                    ["team_id", "home_team_id", "away_team_id", "team", "team_code"],
                )
                player_id_raw = first_value(row, ["player_id"])

                _append_chunks(
                    buffer=buffer,
                    settings=settings,
                    source_table=source_table,
                    source_row_id=source_row_id,
                    title=title,
                    content=content,
                    season_year=season_year_value,
                    season_id=season_id,
                    league_type_code=league_type_code,
                    team_id=str(team_id_raw) if team_id_raw is not None else None,
                    player_id=str(player_id_raw) if player_id_raw is not None else None,
                    meta=row,
                )

                if len(buffer) >= embed_batch_size:
                    flushed = flush_chunks(
                        write_cur,
                        settings,
                        buffer,
                        max_concurrency=max_concurrency,
                        commit_interval=commit_interval,
                        stats=stats,
                        skip_embedding=False,
                    )
                    flushed_chunks += flushed

        flushed = flush_chunks(
            write_cur,
            settings,
            buffer,
            max_concurrency=max_concurrency,
            commit_interval=commit_interval,
            stats=stats,
            skip_embedding=False,
        )
        flushed_chunks += flushed
        dest_conn.commit()

    return {
        "missing_ids": len(missing_ids),
        "matched_rows": matched_rows + len(copied_ids),
        "flushed_chunks": flushed_chunks + len(copied_ids),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _require_psycopg()
    report_path = Path(args.report_path).expanduser().resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Coverage report not found: {report_path}")

    targets = load_missing_targets(report_path, args.start_year, args.end_year)
    print(f"targets_to_reembed={len(targets)}")
    if not targets:
        return 0

    settings = get_settings()
    source_db_url = args.source_db_url.strip()
    if not source_db_url and args.supabase_url.strip():
        print("[WARN] --supabase-url is deprecated. Use --source-db-url instead.")
        source_db_url = args.supabase_url.strip()
    if not source_db_url:
        if args.source_env_file.strip():
            source_db_url = _load_settings_from_env_file(
                args.source_env_file
            ).database_url
        else:
            source_db_url = settings.source_db_url

    dest_db_url = settings.database_url
    if args.dest_env_file.strip():
        dest_db_url = _load_settings_from_env_file(args.dest_env_file).database_url

    with psycopg.connect(source_db_url, autocommit=True) as source_conn:
        with psycopg.connect(dest_db_url) as dest_conn:
            with dest_conn.cursor() as cur:
                _prepare_dest_cursor(cur)

            for idx, target in enumerate(targets, start=1):
                print(
                    f"[{idx}/{len(targets)}] collecting missing ids table={target.table} year={target.year}",
                    flush=True,
                )
                missing_ids = collect_missing_ids(source_conn, dest_conn, target)
                print(
                    f"[{idx}/{len(targets)}] reembed table={target.table} year={target.year} missing_ids={len(missing_ids)}",
                    flush=True,
                )
                if not missing_ids:
                    continue

                result = reembed_target_missing_rows(
                    source_conn=source_conn,
                    dest_conn=dest_conn,
                    target=target,
                    missing_ids=missing_ids,
                    embed_batch_size=args.embed_batch_size,
                    read_batch_size=args.read_batch_size,
                    max_concurrency=args.max_concurrency,
                    commit_interval=args.commit_interval,
                )
                print(
                    f"[{idx}/{len(targets)}] done table={target.table} year={target.year} "
                    f"matched_rows={result['matched_rows']} flushed_chunks={result['flushed_chunks']}",
                    flush=True,
                )

    print("REEMBED_MISSING_ROWS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
