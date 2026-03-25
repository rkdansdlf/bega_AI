#!/usr/bin/env python3
"""Promote blocked alias rag_chunks rows to their canonical source_row_id in-place."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
from scripts import verify_embedding_coverage as verify
from scripts.dedupe_rag_chunk_aliases import collect_alias_candidates
from scripts.ingest_from_kbo import (
    TABLE_PROFILES,
    build_content,
    build_select_query,
    build_title,
    coerce_int,
    first_value,
    resolve_primary_key_columns,
)
from scripts.prune_rag_chunk_extras import filter_targets, resolve_db_urls

DEFAULT_OUTPUT = "logs/canonicalize_blocked_rag_chunk_aliases_report.json"
DEFAULT_READ_BATCH_SIZE = 500
DEFAULT_COMMIT_INTERVAL = 500
DEFAULT_EXISTING_LOOKUP_BATCH_SIZE = 1000


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required to run canonicalize_blocked_rag_chunk_aliases.py. "
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
            "Promote blocked alias rag_chunks rows to canonical source_row_id when "
            "the alias set maps 1:1 to a single rebuilt chunk."
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
        default=5000,
        help="Destination rag_chunks rows fetched per batch while scanning blocked groups.",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=DEFAULT_READ_BATCH_SIZE,
        help="Source rows fetched per batch while rebuilding canonical rows.",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=DEFAULT_COMMIT_INTERVAL,
        help="Commit destination updates every N canonicalized rows.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="How many blocked group samples to retain per target.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply updates. Without this flag the script is read-only.",
    )
    return parser


def _plain_numeric_id_value(source_row_id: str) -> Optional[int]:
    base = source_row_id.split("#part", 1)[0]
    if not base.startswith("id="):
        return None
    suffix = base[3:]
    if not suffix.isdigit():
        return None
    return int(suffix)


def choose_primary_alias_raw_id(
    *,
    canonical_id: str,
    alias_raw_ids: Sequence[str],
) -> Optional[str]:
    candidates = [value for value in alias_raw_ids if value]
    if not candidates:
        return None

    def _priority(raw_source_row_id: str) -> tuple[Any, ...]:
        base = raw_source_row_id.split("#part", 1)[0]
        numeric_id = _plain_numeric_id_value(raw_source_row_id)
        return (
            0 if "#part" not in raw_source_row_id else 1,
            0 if canonical_id in base else 1,
            0 if numeric_id is None else 1,
            numeric_id if numeric_id is not None else 10**18,
            len(base),
            base,
            raw_source_row_id,
        )

    return min(candidates, key=_priority)


def plan_blocked_group_action(
    *,
    canonical_id: str,
    alias_raw_ids: Sequence[str],
    sample_limit: int,
) -> Dict[str, Any]:
    primary_alias_raw_id = choose_primary_alias_raw_id(
        canonical_id=canonical_id,
        alias_raw_ids=alias_raw_ids,
    )
    if primary_alias_raw_id is None:
        return {
            "can_canonicalize": False,
            "reason": "no_alias_rows",
            "alias_raw_ids": list(alias_raw_ids)[:sample_limit],
        }

    if "#part" in primary_alias_raw_id:
        return {
            "can_canonicalize": False,
            "reason": "multipart_alias_row",
            "alias_raw_ids": list(alias_raw_ids)[:sample_limit],
        }

    secondary_alias_raw_ids = [
        raw_source_row_id
        for raw_source_row_id in alias_raw_ids
        if raw_source_row_id != primary_alias_raw_id
    ]
    return {
        "can_canonicalize": True,
        "primary_alias_raw_id": primary_alias_raw_id,
        "secondary_alias_raw_ids": secondary_alias_raw_ids,
    }


def plan_destination_write(
    *,
    canonical_id: str,
    primary_alias_raw_id: str,
    secondary_alias_raw_ids: Sequence[str],
    existing_canonical_ids: Set[str],
) -> Dict[str, Any]:
    if canonical_id in existing_canonical_ids:
        return {
            "mode": "merge_into_existing",
            "delete_alias_raw_ids": [primary_alias_raw_id, *secondary_alias_raw_ids],
        }
    return {
        "mode": "promote_alias",
        "delete_alias_raw_ids": list(secondary_alias_raw_ids),
    }


def _build_single_chunk_content(
    *,
    settings: Any,
    table: str,
    source_row_id: str,
    row: Dict[str, Any],
    profile: Dict[str, Any],
) -> tuple[str, str, Dict[str, Any]] | None:
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
            today_str="",
        )
    else:
        content = build_content(row, table, source_row_id, profile)

    chunks = smart_chunks(content, settings=settings)
    if len(chunks) != 1:
        return None
    return title, chunks[0], row


def _load_existing_canonical_ids(
    *,
    dest_conn: Any,
    source_table: str,
    canonical_ids: Sequence[str],
    batch_size: int = DEFAULT_EXISTING_LOOKUP_BATCH_SIZE,
) -> Set[str]:
    if not canonical_ids:
        return set()

    existing: Set[str] = set()
    with dest_conn.cursor() as read_cur:
        read_cur.execute("SET statement_timeout TO 0")
        for start in range(0, len(canonical_ids), batch_size):
            batch = list(canonical_ids[start : start + batch_size])
            read_cur.execute(
                """
                SELECT source_row_id
                FROM rag_chunks
                WHERE source_table = %s
                  AND source_row_id = ANY(%s)
                """,
                (source_table, batch),
            )
            existing.update(row[0] for row in read_cur.fetchall())
    return existing


def canonicalize_target(
    *,
    source_conn: Any,
    dest_conn: Any,
    target: verify.CoverageTarget,
    fetch_batch_size: int,
    read_batch_size: int,
    commit_interval: int,
    execute: bool,
    sample_limit: int,
) -> Dict[str, Any]:
    alias_analysis = collect_alias_candidates(
        source_conn=source_conn,
        dest_conn=dest_conn,
        target=target,
        fetch_batch_size=fetch_batch_size,
        sample_limit=sample_limit,
    )
    blocked_groups = list(alias_analysis.get("blocked_groups") or [])
    if not blocked_groups:
        return {
            "table": target.table,
            "source_table": target.source_table,
            "year": target.year,
            "blocked_group_count": 0,
            "eligible_group_count": 0,
            "canonicalized_count": 0,
            "merged_into_existing_count": 0,
            "secondary_alias_row_count": 0,
            "deleted_alias_row_count": 0,
            "skipped_group_count": 0,
            "sample_skipped_groups": [],
        }

    eligible: Dict[str, Dict[str, Any]] = {}
    skipped_groups: List[Dict[str, Any]] = []
    secondary_alias_row_count = 0
    for group in blocked_groups:
        alias_raw_ids = list(group["alias_raw_ids"])
        action = plan_blocked_group_action(
            canonical_id=group["canonical_id"],
            alias_raw_ids=alias_raw_ids,
            sample_limit=sample_limit,
        )
        if not action["can_canonicalize"]:
            skipped_groups.append(
                {
                    "canonical_id": group["canonical_id"],
                    "alias_raw_ids": action["alias_raw_ids"],
                    "reason": action["reason"],
                }
            )
            continue
        eligible[group["canonical_id"]] = action
        secondary_alias_row_count += len(action["secondary_alias_raw_ids"])

    if not eligible:
        return {
            "table": target.table,
            "source_table": target.source_table,
            "year": target.year,
            "blocked_group_count": len(blocked_groups),
            "eligible_group_count": 0,
            "canonicalized_count": 0,
            "merged_into_existing_count": 0,
            "secondary_alias_row_count": 0,
            "deleted_alias_row_count": 0,
            "skipped_group_count": len(skipped_groups),
            "sample_skipped_groups": skipped_groups[:sample_limit],
        }

    settings = get_settings()
    profile = TABLE_PROFILES.get(target.table, {})
    pk_columns = resolve_primary_key_columns(source_conn, target.table, profile)
    pk_hint: Sequence[str] = profile.get("pk_hint", [])
    query, params = build_select_query(
        target.table,
        profile,
        pk_columns,
        limit=None,
        season_year=target.year if target.year != 0 else None,
        since=None,
    )
    from scripts.verify_embedding_coverage import build_expected_source_row_id

    pending = dict(eligible)
    existing_canonical_ids = _load_existing_canonical_ids(
        dest_conn=dest_conn,
        source_table=target.source_table,
        canonical_ids=list(eligible.keys()),
    )
    canonicalized_count = 0
    merged_into_existing_count = 0
    deleted_alias_row_count = 0
    updates_since_commit = 0

    with (
        source_conn.cursor(row_factory=dict_row) as read_cur,
        dest_conn.cursor() as write_cur,
    ):
        write_cur.execute("SET statement_timeout TO 0")
        read_cur.execute(query, params)

        while True:
            rows = read_cur.fetchmany(read_batch_size)
            if not rows:
                break
            for raw_row in rows:
                row = dict(raw_row)
                canonical_id = build_expected_source_row_id(
                    row=row,
                    target=target,
                    pk_columns=pk_columns,
                    pk_hint=pk_hint,
                )
                action = pending.get(canonical_id)
                if not action:
                    continue
                primary_alias_raw_id = action["primary_alias_raw_id"]
                secondary_alias_raw_ids = list(action["secondary_alias_raw_ids"])
                write_plan = plan_destination_write(
                    canonical_id=canonical_id,
                    primary_alias_raw_id=primary_alias_raw_id,
                    secondary_alias_raw_ids=secondary_alias_raw_ids,
                    existing_canonical_ids=existing_canonical_ids,
                )

                built = _build_single_chunk_content(
                    settings=settings,
                    table=target.table,
                    source_row_id=canonical_id,
                    row=row,
                    profile=profile,
                )
                if built is None:
                    skipped_groups.append(
                        {
                            "canonical_id": canonical_id,
                            "alias_raw_ids": [primary_alias_raw_id, *secondary_alias_raw_ids][
                                :sample_limit
                            ],
                            "reason": "rebuilt_content_is_multipart",
                        }
                    )
                    pending.pop(canonical_id, None)
                    continue

                title, content, meta = built
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

                if execute:
                    if write_plan["mode"] == "merge_into_existing":
                        if target.year == 0:
                            write_cur.execute(
                                """
                                UPDATE rag_chunks
                                SET title = %s,
                                    content = %s,
                                    meta = %s::jsonb,
                                    season_year = %s,
                                    season_id = %s,
                                    league_type_code = %s,
                                    team_id = %s,
                                    player_id = %s,
                                    updated_at = now()
                                WHERE source_table = %s
                                  AND source_row_id = %s
                                """,
                                (
                                    title,
                                    content,
                                    json.dumps(meta, ensure_ascii=False, default=str),
                                    season_year_value,
                                    season_id,
                                    league_type_code,
                                    str(team_id_raw) if team_id_raw is not None else None,
                                    str(player_id_raw) if player_id_raw is not None else None,
                                    target.source_table,
                                    canonical_id,
                                ),
                            )
                        else:
                            write_cur.execute(
                                """
                                UPDATE rag_chunks
                                SET title = %s,
                                    content = %s,
                                    meta = %s::jsonb,
                                    season_year = %s,
                                    season_id = %s,
                                    league_type_code = %s,
                                    team_id = %s,
                                    player_id = %s,
                                    updated_at = now()
                                WHERE source_table = %s
                                  AND source_row_id = %s
                                """,
                                (
                                    title,
                                    content,
                                    json.dumps(meta, ensure_ascii=False, default=str),
                                    season_year_value,
                                    season_id,
                                    league_type_code,
                                    str(team_id_raw) if team_id_raw is not None else None,
                                    str(player_id_raw) if player_id_raw is not None else None,
                                    target.source_table,
                                    canonical_id,
                                ),
                            )
                    else:
                        write_cur.execute(
                            """
                            UPDATE rag_chunks
                            SET source_row_id = %s,
                                title = %s,
                                content = %s,
                                meta = %s::jsonb,
                                season_year = %s,
                                season_id = %s,
                                league_type_code = %s,
                                team_id = %s,
                                player_id = %s,
                                updated_at = now()
                            WHERE source_table = %s
                              AND source_row_id = %s
                            """,
                            (
                                canonical_id,
                                title,
                                content,
                                json.dumps(meta, ensure_ascii=False, default=str),
                                season_year_value,
                                season_id,
                                league_type_code,
                                str(team_id_raw) if team_id_raw is not None else None,
                                str(player_id_raw) if player_id_raw is not None else None,
                                target.source_table,
                                primary_alias_raw_id,
                            ),
                        )
                        existing_canonical_ids.add(canonical_id)
                    delete_alias_raw_ids = list(write_plan["delete_alias_raw_ids"])
                    if delete_alias_raw_ids:
                        if target.year == 0:
                            write_cur.execute(
                                """
                                DELETE FROM rag_chunks
                                WHERE source_table = %s
                                  AND source_row_id = ANY(%s)
                                """,
                                (target.source_table, delete_alias_raw_ids),
                            )
                        else:
                            write_cur.execute(
                                """
                                DELETE FROM rag_chunks
                                WHERE source_table = %s
                                  AND season_year = %s
                                  AND source_row_id = ANY(%s)
                                """,
                                (
                                    target.source_table,
                                    target.year,
                                    delete_alias_raw_ids,
                                ),
                            )
                        deleted_alias_row_count += len(delete_alias_raw_ids)
                    updates_since_commit += 1
                    if updates_since_commit >= commit_interval:
                        dest_conn.commit()
                        updates_since_commit = 0

                canonicalized_count += 1
                if write_plan["mode"] == "merge_into_existing":
                    merged_into_existing_count += 1
                pending.pop(canonical_id, None)

        if execute and updates_since_commit > 0:
            dest_conn.commit()

    for canonical_id, action in pending.items():
        skipped_groups.append(
            {
                "canonical_id": canonical_id,
                "alias_raw_ids": [
                    action["primary_alias_raw_id"],
                    *action["secondary_alias_raw_ids"],
                ][:sample_limit],
                "reason": "canonical_source_row_not_found",
            }
        )

    return {
        "table": target.table,
        "source_table": target.source_table,
        "year": target.year,
        "blocked_group_count": len(blocked_groups),
        "eligible_group_count": len(eligible),
        "canonicalized_count": canonicalized_count,
        "merged_into_existing_count": merged_into_existing_count,
        "secondary_alias_row_count": secondary_alias_row_count,
        "deleted_alias_row_count": deleted_alias_row_count,
        "skipped_group_count": len(skipped_groups),
        "sample_skipped_groups": skipped_groups[:sample_limit],
    }


def canonicalize_blocked_aliases(
    *,
    source_db_url: str,
    dest_db_url: str,
    targets: Sequence[verify.CoverageTarget],
    fetch_batch_size: int,
    read_batch_size: int,
    commit_interval: int,
    execute: bool,
    sample_limit: int,
) -> Dict[str, Any]:
    _require_psycopg()

    rows: List[Dict[str, Any]] = []
    total_blocked_group_count = 0
    total_eligible_group_count = 0
    total_canonicalized_count = 0
    total_merged_into_existing_count = 0
    total_secondary_alias_row_count = 0
    total_deleted_alias_row_count = 0
    total_skipped_group_count = 0

    for index, target in enumerate(targets, start=1):
        print(
            f"[{index}/{len(targets)}] canonicalizing blocked aliases for table={target.table} year={target.year}",
            flush=True,
        )
        source_conn = psycopg.connect(source_db_url, autocommit=True)
        dest_conn = psycopg.connect(dest_db_url)
        try:
            with dest_conn.cursor() as init_cur:
                init_cur.execute("SET search_path TO public, extensions, security")
                init_cur.execute("SET statement_timeout TO 0")

            row = canonicalize_target(
                source_conn=source_conn,
                dest_conn=dest_conn,
                target=target,
                fetch_batch_size=fetch_batch_size,
                read_batch_size=read_batch_size,
                commit_interval=commit_interval,
                execute=execute,
                sample_limit=sample_limit,
            )
            if execute:
                dest_conn.commit()
            else:
                dest_conn.rollback()
        finally:
            _safe_close_connection(source_conn)
            _safe_close_connection(dest_conn)
        rows.append(row)
        total_blocked_group_count += int(row["blocked_group_count"])
        total_eligible_group_count += int(row["eligible_group_count"])
        total_canonicalized_count += int(row["canonicalized_count"])
        total_merged_into_existing_count += int(row["merged_into_existing_count"])
        total_secondary_alias_row_count += int(row["secondary_alias_row_count"])
        total_deleted_alias_row_count += int(row["deleted_alias_row_count"])
        total_skipped_group_count += int(row["skipped_group_count"])

    return {
        "targets": len(rows),
        "execute": execute,
        "total_blocked_group_count": total_blocked_group_count,
        "total_eligible_group_count": total_eligible_group_count,
        "total_canonicalized_count": total_canonicalized_count,
        "total_merged_into_existing_count": total_merged_into_existing_count,
        "total_secondary_alias_row_count": total_secondary_alias_row_count,
        "total_deleted_alias_row_count": total_deleted_alias_row_count,
        "total_skipped_group_count": total_skipped_group_count,
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
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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

    report = canonicalize_blocked_aliases(
        source_db_url=source_db_url,
        dest_db_url=dest_db_url,
        targets=targets,
        fetch_batch_size=args.fetch_batch_size,
        read_batch_size=args.read_batch_size,
        commit_interval=args.commit_interval,
        execute=args.execute,
        sample_limit=args.sample_limit,
    )
    output_path = write_report(args.output, report)
    print(json.dumps(report["rows"], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "summary": {
                    "targets": report["targets"],
                    "execute": report["execute"],
                    "total_blocked_group_count": report["total_blocked_group_count"],
                    "total_eligible_group_count": report["total_eligible_group_count"],
                    "total_canonicalized_count": report["total_canonicalized_count"],
                    "total_merged_into_existing_count": report[
                        "total_merged_into_existing_count"
                    ],
                    "total_secondary_alias_row_count": report[
                        "total_secondary_alias_row_count"
                    ],
                    "total_deleted_alias_row_count": report[
                        "total_deleted_alias_row_count"
                    ],
                    "total_skipped_group_count": report["total_skipped_group_count"],
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
