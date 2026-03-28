#!/usr/bin/env python3
"""Report or delete alias rag_chunks rows when a canonical raw row is present."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from scripts.prune_rag_chunk_extras import (
    _iter_expected_ids,
    build_candidate_ids,
    filter_targets,
    resolve_db_urls,
)

DEFAULT_OUTPUT = "logs/dedupe_rag_chunk_aliases_report.json"
DEFAULT_FETCH_BATCH_SIZE = 5000
DEFAULT_DELETE_BATCH_SIZE = 1000


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required to run dedupe_rag_chunk_aliases.py. "
            "Install dependencies (e.g. pip install psycopg[binary]) and retry. "
            f"Detail: {_PSYCOPG_IMPORT_ERROR}"
        )


def _safe_close_connection(conn: Any) -> None:
    close = getattr(conn, "close", None)
    if close is None:
        return
    error_cls = getattr(psycopg, "OperationalError", Exception)
    try:
        close()
    except error_cls:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report or delete alias rag_chunks rows that normalize to the same "
            "canonical source_row_id as an existing canonical raw row."
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
        help="Destination rag_chunks rows fetched per batch while scanning for alias rows.",
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
        help="How many alias samples to retain per target.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete only alias rows whose canonical raw row is already present.",
    )
    return parser


def base_source_row_id(source_row_id: str) -> str:
    if not source_row_id:
        return ""
    return source_row_id.split("#part", 1)[0]


def analyze_alias_groups(
    *,
    rows: Iterable[tuple[str, Optional[Dict[str, Any]]]],
    table: str,
    expected_ids: set[str],
    legacy_aliases: Optional[Dict[str, str]],
    sample_limit: int,
) -> Dict[str, Any]:
    grouped_raw_ids: Dict[str, set[str]] = defaultdict(set)
    actual_row_count = 0

    for raw_source_row_id, meta in rows:
        actual_row_count += 1
        matched_expected_ids = sorted(
            candidate_id
            for candidate_id in build_candidate_ids(
                raw_source_row_id=raw_source_row_id,
                table=table,
                meta=meta,
                legacy_aliases=legacy_aliases,
            )
            if candidate_id in expected_ids
        )
        if not matched_expected_ids:
            continue
        grouped_raw_ids[matched_expected_ids[0]].add(raw_source_row_id)

    alias_groups: List[Dict[str, Any]] = []
    deletable_alias_source_row_ids: List[str] = []
    blocked_alias_source_row_ids: List[str] = []
    deletable_groups: List[Dict[str, Any]] = []
    blocked_groups: List[Dict[str, Any]] = []

    for canonical_id, raw_ids in grouped_raw_ids.items():
        canonical_raw_ids = sorted(
            raw_id for raw_id in raw_ids if base_source_row_id(raw_id) == canonical_id
        )
        alias_raw_ids = sorted(
            raw_id for raw_id in raw_ids if base_source_row_id(raw_id) != canonical_id
        )
        if not alias_raw_ids:
            continue

        can_delete = bool(canonical_raw_ids)
        if can_delete:
            deletable_alias_source_row_ids.extend(alias_raw_ids)
        else:
            blocked_alias_source_row_ids.extend(alias_raw_ids)

        group = {
            "canonical_id": canonical_id,
            "canonical_raw_ids": canonical_raw_ids,
            "alias_raw_ids": alias_raw_ids,
            "alias_raw_id_count": len(alias_raw_ids),
            "can_delete": can_delete,
        }
        alias_groups.append(
            {
                **group,
                "canonical_raw_ids": canonical_raw_ids[:sample_limit],
                "alias_raw_ids": alias_raw_ids[:sample_limit],
            }
        )
        if can_delete:
            deletable_groups.append(group)
        else:
            blocked_groups.append(group)

    alias_groups.sort(
        key=lambda item: (
            not item["can_delete"],
            -item["alias_raw_id_count"],
            item["canonical_id"],
        )
    )

    return {
        "actual_row_count": actual_row_count,
        "alias_group_count": len(alias_groups),
        "deletable_alias_row_count": len(deletable_alias_source_row_ids),
        "blocked_alias_row_count": len(blocked_alias_source_row_ids),
        "deletable_alias_source_row_ids": deletable_alias_source_row_ids,
        "blocked_alias_source_row_ids": blocked_alias_source_row_ids,
        "deletable_groups": deletable_groups,
        "blocked_groups": blocked_groups,
        "sample_alias_groups": alias_groups[:sample_limit],
    }


def collect_alias_candidates(
    *,
    source_conn: Any,
    dest_conn: Any,
    target: verify.CoverageTarget,
    fetch_batch_size: int,
    sample_limit: int,
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

    with dest_conn.cursor() as read_cur:
        read_cur.execute(read_sql, params)

        def _iter_rows() -> Iterable[tuple[str, Optional[Dict[str, Any]]]]:
            while True:
                rows = read_cur.fetchmany(fetch_batch_size)
                if not rows:
                    return
                for row in rows:
                    yield row[0], row[1]

        analysis = analyze_alias_groups(
            rows=_iter_rows(),
            table=target.table,
            expected_ids=expected_ids,
            legacy_aliases=legacy_aliases,
            sample_limit=sample_limit,
        )

    return {
        "table": target.table,
        "source_table": target.source_table,
        "year": target.year,
        "expected_count": expected_count,
        **analysis,
    }


def delete_alias_source_row_ids(
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


def dedupe_rag_chunk_aliases(
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

    rows: List[Dict[str, Any]] = []
    total_expected_count = 0
    total_actual_row_count = 0
    total_alias_group_count = 0
    total_deletable_alias_row_count = 0
    total_blocked_alias_row_count = 0
    total_deleted_count = 0

    for index, target in enumerate(targets, start=1):
        print(
            f"[{index}/{len(targets)}] scanning alias rows for table={target.table} year={target.year}",
            flush=True,
        )
        source_conn = psycopg.connect(source_db_url, autocommit=True)
        dest_conn = psycopg.connect(dest_db_url)
        try:
            with dest_conn.cursor() as init_cur:
                init_cur.execute("SET search_path TO public, extensions, security")
                init_cur.execute("SET statement_timeout TO 0")

            result = collect_alias_candidates(
                source_conn=source_conn,
                dest_conn=dest_conn,
                target=target,
                fetch_batch_size=fetch_batch_size,
                sample_limit=sample_limit,
            )
            deleted_count = 0
            if execute and result["deletable_alias_source_row_ids"]:
                deleted_count = delete_alias_source_row_ids(
                    dest_conn=dest_conn,
                    target=target,
                    source_row_ids=result["deletable_alias_source_row_ids"],
                    delete_batch_size=delete_batch_size,
                )
            else:
                dest_conn.rollback()
        finally:
            _safe_close_connection(source_conn)
            _safe_close_connection(dest_conn)

        total_expected_count += int(result["expected_count"])
        total_actual_row_count += int(result["actual_row_count"])
        total_alias_group_count += int(result["alias_group_count"])
        total_deletable_alias_row_count += int(result["deletable_alias_row_count"])
        total_blocked_alias_row_count += int(result["blocked_alias_row_count"])
        total_deleted_count += deleted_count
        rows.append(
            {
                "table": result["table"],
                "source_table": result["source_table"],
                "year": result["year"],
                "expected_count": int(result["expected_count"]),
                "actual_row_count": int(result["actual_row_count"]),
                "alias_group_count": int(result["alias_group_count"]),
                "deletable_alias_row_count": int(result["deletable_alias_row_count"]),
                "blocked_alias_row_count": int(result["blocked_alias_row_count"]),
                "deleted_count": deleted_count,
                "sample_alias_groups": result["sample_alias_groups"],
            }
        )

    return {
        "targets": len(rows),
        "execute": execute,
        "total_expected_count": total_expected_count,
        "total_actual_row_count": total_actual_row_count,
        "total_alias_group_count": total_alias_group_count,
        "total_deletable_alias_row_count": total_deletable_alias_row_count,
        "total_blocked_alias_row_count": total_blocked_alias_row_count,
        "total_deleted_count": total_deleted_count,
        "rows": rows,
    }


def write_report(path_str: str, report: Dict[str, Any]) -> Optional[Path]:
    value = path_str.strip()
    if not value:
        return None
    path = Path(value)
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

    report = dedupe_rag_chunk_aliases(
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
                    "total_actual_row_count": report["total_actual_row_count"],
                    "total_alias_group_count": report["total_alias_group_count"],
                    "total_deletable_alias_row_count": report[
                        "total_deletable_alias_row_count"
                    ],
                    "total_blocked_alias_row_count": report[
                        "total_blocked_alias_row_count"
                    ],
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
