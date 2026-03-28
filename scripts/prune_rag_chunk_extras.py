#!/usr/bin/env python3
"""Report or delete stale extra rag_chunks rows detected by coverage checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psycopg
except ModuleNotFoundError as exc:
    psycopg = None
    _PSYCOPG_IMPORT_ERROR = exc

from scripts import verify_embedding_coverage as verify
from scripts.sync_rag_chunks import _load_settings_from_env_file

DEFAULT_OUTPUT = "logs/prune_rag_chunk_extras_report.json"
DEFAULT_FETCH_BATCH_SIZE = 5000
DEFAULT_DELETE_BATCH_SIZE = 1000


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required to run prune_rag_chunk_extras.py. "
            "Install dependencies (e.g. pip install psycopg[binary]) and retry. "
            f"Detail: {_PSYCOPG_IMPORT_ERROR}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report or delete stale rag_chunks rows whose normalized source_row_id "
            "does not map to the current source coverage set."
        )
    )
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument(
        "--mode",
        choices=["seasonal", "static", "all"],
        default="seasonal",
        help="Coverage scope. Defaults to seasonal because source-file profiles share source_table names.",
    )
    parser.add_argument(
        "--table",
        nargs="*",
        default=[],
        help="Optional coverage target.table filters.",
    )
    parser.add_argument(
        "--source-env-file",
        default=".env",
        help="Env file used to resolve source DB config when --source-db-url is omitted.",
    )
    parser.add_argument(
        "--dest-env-file",
        default="",
        help="Env file used to resolve destination DB config when --dest-db-url is omitted.",
    )
    parser.add_argument(
        "--source-db-url",
        default="",
        help="Explicit source DB URL override.",
    )
    parser.add_argument(
        "--dest-db-url",
        default="",
        help="Explicit destination DB URL override.",
    )
    parser.add_argument(
        "--allow-same-db",
        action="store_true",
        help="Allow source and destination DB URLs to match.",
    )
    parser.add_argument(
        "--fetch-batch-size",
        type=int,
        default=DEFAULT_FETCH_BATCH_SIZE,
        help="Destination rag_chunks rows fetched per batch while scanning for extras.",
    )
    parser.add_argument(
        "--delete-batch-size",
        type=int,
        default=DEFAULT_DELETE_BATCH_SIZE,
        help="Rows deleted per batch when --execute is set.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="How many extra source_row_id samples to retain per target.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete the detected extra rag_chunks rows. Without this flag the script is read-only.",
    )
    return parser


def resolve_db_urls(args: argparse.Namespace) -> tuple[str, str]:
    source_db_url = args.source_db_url.strip()
    dest_db_url = args.dest_db_url.strip()

    if not source_db_url:
        source_db_url = _load_settings_from_env_file(args.source_env_file).database_url
    if not dest_db_url:
        dest_env_file = args.dest_env_file.strip() or args.source_env_file
        dest_db_url = _load_settings_from_env_file(dest_env_file).database_url

    if not args.allow_same_db and source_db_url == dest_db_url:
        raise RuntimeError(
            "Source and destination DB URLs are identical. "
            "Pass --allow-same-db only if that is intentional."
        )
    return source_db_url, dest_db_url


def filter_targets(
    *,
    mode: str,
    start_year: int,
    end_year: int,
    tables: Sequence[str],
) -> List[verify.CoverageTarget]:
    table_filters = {value.strip() for value in tables if value.strip()}
    targets = verify.build_targets(mode, start_year, end_year)
    if not table_filters:
        return targets
    return [target for target in targets if target.table in table_filters]


def _iter_expected_ids(dest_cur: Any, batch_size: int) -> Iterable[str]:
    dest_cur.execute("SELECT id FROM expected_ids ORDER BY id")
    while True:
        rows = dest_cur.fetchmany(batch_size)
        if not rows:
            return
        for row in rows:
            yield row[0]


def build_candidate_ids(
    *,
    raw_source_row_id: str,
    table: str,
    meta: Optional[Dict[str, Any]],
    legacy_aliases: Optional[Dict[str, str]],
) -> set[str]:
    normalized_id = verify.normalize_actual_source_row_id(
        raw_source_row_id,
        table,
        meta,
        legacy_aliases,
    )
    candidate_ids = {normalized_id}

    base = raw_source_row_id.split("#part", 1)[0]
    pairs = verify._parse_row_id_pairs(base)
    canonical_keys = verify.CANONICAL_SOURCE_ROW_KEYS.get(table)
    if canonical_keys and len(pairs) == len(canonical_keys):
        has_all_canonical_keys = all(key in pairs for key in canonical_keys)
        if has_all_canonical_keys:
            canonical_from_base = verify._build_canonical_from_mapping(pairs, table)
            if canonical_from_base:
                candidate_ids.add(canonical_from_base)

    return candidate_ids


def collect_extra_source_row_ids(
    *,
    source_conn: Any,
    dest_conn: Any,
    target: verify.CoverageTarget,
    fetch_batch_size: int,
) -> Dict[str, Any]:
    with dest_conn.cursor() as dest_cur:
        verify._recreate_temp_tables(dest_cur)
        expected_count, legacy_aliases = verify._load_expected_ids(
            source_conn,
            dest_cur,
            target,
        )
        expected_ids = set(_iter_expected_ids(dest_cur, fetch_batch_size))

    read_sql, params = verify.build_actual_rows_query(target)

    actual_count = 0
    extra_source_row_ids: List[str] = []
    with dest_conn.cursor() as read_cur:
        read_cur.execute(read_sql, params)
        while True:
            rows = read_cur.fetchmany(fetch_batch_size)
            if not rows:
                break
            for raw_source_row_id, meta in rows:
                actual_count += 1
                candidate_ids = build_candidate_ids(
                    raw_source_row_id=raw_source_row_id,
                    table=target.table,
                    meta=meta,
                    legacy_aliases=legacy_aliases,
                )
                if candidate_ids.isdisjoint(expected_ids):
                    extra_source_row_ids.append(raw_source_row_id)

    return {
        "table": target.table,
        "source_table": target.source_table,
        "year": target.year,
        "expected_count": expected_count,
        "actual_count": actual_count,
        "extra_source_row_ids": extra_source_row_ids,
    }


def delete_extra_source_row_ids(
    *,
    dest_conn: Any,
    target: verify.CoverageTarget,
    source_row_ids: Sequence[str],
    delete_batch_size: int,
) -> int:
    if not source_row_ids:
        return 0

    deleted = 0
    with dest_conn.cursor() as dest_cur:
        dest_cur.execute("SET statement_timeout TO 0")
        for start in range(0, len(source_row_ids), delete_batch_size):
            batch = list(source_row_ids[start : start + delete_batch_size])
            if target.year == 0:
                dest_cur.execute(
                    """
                    DELETE FROM rag_chunks
                    WHERE source_table = %s
                      AND source_row_id = ANY(%s)
                    """,
                    (target.source_table, batch),
                )
            else:
                dest_cur.execute(
                    """
                    DELETE FROM rag_chunks
                    WHERE source_table = %s
                      AND season_year = %s
                      AND source_row_id = ANY(%s)
                    """,
                    (target.source_table, target.year, batch),
                )
            dest_conn.commit()
            deleted += len(batch)
    return deleted


def prune_extra_rag_chunks(
    *,
    source_db_url: str,
    dest_db_url: str,
    targets: Sequence[verify.CoverageTarget],
    fetch_batch_size: int,
    delete_batch_size: int,
    sample_limit: int,
    execute: bool,
) -> Dict[str, Any]:
    _require_psycopg()

    target_reports: List[Dict[str, Any]] = []
    total_expected_count = 0
    total_actual_count = 0
    total_extra_count = 0
    total_deleted_count = 0

    with psycopg.connect(source_db_url) as source_conn, psycopg.connect(
        dest_db_url
    ) as dest_conn:
        with dest_conn.cursor() as init_cur:
            init_cur.execute("SET search_path TO public, extensions, security")
            init_cur.execute("SET statement_timeout TO 0")

        for index, target in enumerate(targets, start=1):
            print(
                f"[{index}/{len(targets)}] scanning extras for table={target.table} year={target.year}",
                flush=True,
            )
            result = collect_extra_source_row_ids(
                source_conn=source_conn,
                dest_conn=dest_conn,
                target=target,
                fetch_batch_size=fetch_batch_size,
            )
            extra_source_row_ids = result["extra_source_row_ids"]
            deleted_count = 0
            if execute and extra_source_row_ids:
                deleted_count = delete_extra_source_row_ids(
                    dest_conn=dest_conn,
                    target=target,
                    source_row_ids=extra_source_row_ids,
                    delete_batch_size=delete_batch_size,
                )

            total_expected_count += int(result["expected_count"])
            total_actual_count += int(result["actual_count"])
            total_extra_count += len(extra_source_row_ids)
            total_deleted_count += deleted_count
            target_reports.append(
                {
                    "table": target.table,
                    "source_table": target.source_table,
                    "year": target.year,
                    "expected_count": int(result["expected_count"]),
                    "actual_count": int(result["actual_count"]),
                    "extra_count": len(extra_source_row_ids),
                    "deleted_count": deleted_count,
                    "sample_extra_source_row_ids": extra_source_row_ids[:sample_limit],
                }
            )

    return {
        "targets": len(target_reports),
        "execute": execute,
        "total_expected_count": total_expected_count,
        "total_actual_count": total_actual_count,
        "total_extra_count": total_extra_count,
        "total_deleted_count": total_deleted_count,
        "rows": target_reports,
    }


def write_report(path_str: str, report: Dict[str, Any]) -> Optional[Path]:
    path_str = path_str.strip()
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source_db_url, dest_db_url = resolve_db_urls(args)
    targets = filter_targets(
        mode=args.mode,
        start_year=args.start_year,
        end_year=args.end_year,
        tables=args.table,
    )
    if not targets:
        raise RuntimeError("No coverage targets matched the requested filters.")

    report = prune_extra_rag_chunks(
        source_db_url=source_db_url,
        dest_db_url=dest_db_url,
        targets=targets,
        fetch_batch_size=args.fetch_batch_size,
        delete_batch_size=args.delete_batch_size,
        sample_limit=args.sample_limit,
        execute=args.execute,
    )
    output_path = write_report(args.output, report)
    print(json.dumps(report["rows"], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "summary": {
                    "targets": report["targets"],
                    "execute": report["execute"],
                    "total_expected_count": report["total_expected_count"],
                    "total_actual_count": report["total_actual_count"],
                    "total_extra_count": report["total_extra_count"],
                    "total_deleted_count": report["total_deleted_count"],
                },
                "output": str(output_path) if output_path else "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
