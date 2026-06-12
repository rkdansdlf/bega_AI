#!/usr/bin/env python3
"""Clean up remaining 256-d embedding migration warnings without re-embedding.

This script never modifies the stored embedding vector values. It only fills
metadata that is provably stale after the vector column is already vector(256),
and it soft-deactivates active rows that still have NULL embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import psycopg
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("BEGA_SKIP_APP_INIT", "1")

from app.config import get_settings
from app.core.rag_storage import (
    resolve_embedding_dim,
    resolve_embedding_model,
    resolve_embedding_version,
)

EXPECTED_EMBEDDING_MODEL = "openrouter:openai/text-embedding-3-small"
EXPECTED_EMBEDDING_DIM = 256
EXPECTED_EMBEDDING_VERSION = 2
EXPECTED_VECTOR_INDEX = "hnsw"
EXPECTED_VECTOR_QUANTIZATION = "halfvec"
PGVECTOR_SEARCH_PATH = "public, extensions"

METADATA_FIX_WHERE = """
    embedding IS NOT NULL
    AND embedding_version = %s
    AND (
        embedding_model IS NULL
        OR btrim(embedding_model) = ''
        OR embedding_model = %s
    )
    AND (
        embedding_dim IS NULL
        OR embedding_dim <> %s
        OR embedding_model IS NULL
        OR btrim(embedding_model) = ''
    )
"""

METADATA_CONFLICT_WHERE = """
    embedding IS NOT NULL
    AND (
        embedding_version IS NULL
        OR embedding_version <> %s
        OR (
            embedding_model IS NOT NULL
            AND btrim(embedding_model) <> ''
            AND embedding_model <> %s
        )
    )
"""

ACTIVE_NULL_WHERE = """
    embedding IS NULL
    AND is_active = true
"""


def metadata_fix_params() -> tuple[Any, ...]:
    return (
        EXPECTED_EMBEDDING_VERSION,
        EXPECTED_EMBEDDING_MODEL,
        EXPECTED_EMBEDDING_DIM,
    )


def metadata_conflict_params() -> tuple[Any, ...]:
    return (
        EXPECTED_EMBEDDING_VERSION,
        EXPECTED_EMBEDDING_MODEL,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean stale rag_chunks embedding metadata and deactivate active NULL "
            "embedding rows after the 256-d migration."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect planned cleanup without writing. This is the default.",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Persist cleanup changes.",
    )
    parser.add_argument(
        "--report-output",
        default="reports/cleanup_embedding_256_warnings.json",
        help="JSON report output path.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Maximum sample rows per issue class. Use 0 to skip samples.",
    )
    parser.add_argument(
        "--statement-timeout-ms",
        type=int,
        default=120_000,
        help="PostgreSQL statement timeout for cleanup queries.",
    )
    parser.add_argument(
        "--lock-timeout-ms",
        type=int,
        default=5_000,
        help="PostgreSQL lock timeout for cleanup writes.",
    )
    parser.add_argument(
        "--expect-metadata-fixable",
        type=int,
        default=None,
        help=(
            "Expected dry-run metadata_fixable count. On --apply, aborts before "
            "writing if the live count differs."
        ),
    )
    parser.add_argument(
        "--expect-active-null-embeddings",
        type=int,
        default=None,
        help=(
            "Expected dry-run active_null_embeddings count. On --apply, aborts "
            "before writing if the live count differs."
        ),
    )
    parser.add_argument(
        "--expect-metadata-conflicts",
        type=int,
        default=None,
        help=(
            "Expected dry-run metadata_conflicts count. On --apply, aborts "
            "before writing if the live count differs."
        ),
    )
    return parser.parse_args()


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _fetch_rows(
    conn: psycopg.Connection,
    query: str,
    params: Sequence[Any] = (),
) -> List[Dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def _fetch_scalar(
    conn: psycopg.Connection,
    query: str,
    params: Sequence[Any] = (),
) -> int:
    with conn.cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
        return int(row[0] or 0) if row else 0


def _fetch_embedding_column_type(conn: psycopg.Connection) -> str:
    rows = _fetch_rows(
        conn,
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS embedding_type
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'rag_chunks' AND a.attname = 'embedding'
        """,
    )
    if not rows:
        return ""
    return str(rows[0].get("embedding_type") or "")


def _configure_session(
    conn: psycopg.Connection,
    *,
    statement_timeout_ms: int,
    lock_timeout_ms: int,
    read_only: bool = False,
) -> None:
    with conn.cursor() as cur:
        if read_only:
            cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(f"SET search_path TO {PGVECTOR_SEARCH_PATH}")
        cur.execute(
            "SELECT set_config('statement_timeout', %s, false)",
            (str(statement_timeout_ms),),
        )
        cur.execute(
            "SELECT set_config('lock_timeout', %s, false)",
            (str(lock_timeout_ms),),
        )


def _runtime_snapshot(settings: Any) -> Dict[str, Any]:
    return {
        "embed_provider": getattr(settings, "embed_provider", None),
        "openrouter_embed_model": getattr(settings, "openrouter_embed_model", None),
        "embed_dim": getattr(settings, "embed_dim", None),
        "rag_embedding_version": getattr(settings, "rag_embedding_version", None),
        "ai_vector_index": getattr(settings, "ai_vector_index", None),
        "ai_vector_quantization": getattr(settings, "ai_vector_quantization", None),
        "resolved_embedding_model": resolve_embedding_model(settings),
        "resolved_embedding_dim": resolve_embedding_dim(settings),
        "resolved_embedding_version": resolve_embedding_version(settings),
    }


def _validate_runtime(settings: Any) -> List[str]:
    errors: List[str] = []
    snapshot = _runtime_snapshot(settings)
    if snapshot["resolved_embedding_model"] != EXPECTED_EMBEDDING_MODEL:
        errors.append(
            "resolved embedding model is not "
            f"{EXPECTED_EMBEDDING_MODEL}: {snapshot['resolved_embedding_model']}"
        )
    if snapshot["resolved_embedding_dim"] != EXPECTED_EMBEDDING_DIM:
        errors.append(
            "resolved embedding dim is not "
            f"{EXPECTED_EMBEDDING_DIM}: {snapshot['resolved_embedding_dim']}"
        )
    if snapshot["resolved_embedding_version"] != EXPECTED_EMBEDDING_VERSION:
        errors.append(
            "resolved embedding version is not "
            f"{EXPECTED_EMBEDDING_VERSION}: {snapshot['resolved_embedding_version']}"
        )
    vector_index = str(snapshot["ai_vector_index"] or "").lower().strip()
    if vector_index != EXPECTED_VECTOR_INDEX:
        errors.append(
            "runtime AI_VECTOR_INDEX is not "
            f"{EXPECTED_VECTOR_INDEX}: {snapshot['ai_vector_index']}"
        )
    vector_quantization = str(snapshot["ai_vector_quantization"] or "").lower().strip()
    if vector_quantization != EXPECTED_VECTOR_QUANTIZATION:
        errors.append(
            "runtime AI_VECTOR_QUANTIZATION is not "
            f"{EXPECTED_VECTOR_QUANTIZATION}: {snapshot['ai_vector_quantization']}"
        )
    return errors


def _validate_db_preconditions(embedding_column_type: str) -> List[str]:
    if not embedding_column_type.endswith("vector(256)"):
        return [
            "rag_chunks.embedding must be vector(256) before metadata cleanup: "
            f"{embedding_column_type or 'missing'}"
        ]
    return []


def _count_metadata_fixable(conn: psycopg.Connection) -> int:
    return _fetch_scalar(
        conn,
        f"SELECT count(*) FROM rag_chunks WHERE {METADATA_FIX_WHERE}",
        metadata_fix_params(),
    )


def _count_metadata_conflicts(conn: psycopg.Connection) -> int:
    return _fetch_scalar(
        conn,
        f"SELECT count(*) FROM rag_chunks WHERE {METADATA_CONFLICT_WHERE}",
        metadata_conflict_params(),
    )


def _count_active_null_embeddings(conn: psycopg.Connection) -> int:
    return _fetch_scalar(
        conn,
        f"SELECT count(*) FROM rag_chunks WHERE {ACTIVE_NULL_WHERE}",
    )


def _collect_counts(conn: psycopg.Connection) -> Dict[str, int]:
    return {
        "metadata_fixable": _count_metadata_fixable(conn),
        "metadata_conflicts": _count_metadata_conflicts(conn),
        "active_null_embeddings": _count_active_null_embeddings(conn),
    }


def _count_expectation_mismatches(
    counts: Dict[str, int],
    expectations: Dict[str, int | None],
) -> List[Dict[str, int | str]]:
    mismatches: List[Dict[str, int | str]] = []
    for key, expected in expectations.items():
        if expected is None:
            continue
        actual = int(counts.get(key, 0) or 0)
        if actual != expected:
            mismatches.append(
                {
                    "count": key,
                    "expected": int(expected),
                    "actual": actual,
                }
            )
    return mismatches


def _sample_rows(
    conn: psycopg.Connection,
    where_sql: str,
    params: Sequence[Any],
    *,
    sample_limit: int,
) -> List[Dict[str, Any]]:
    return _fetch_rows(
        conn,
        f"""
        SELECT
            id, source_table, source_row_id, season_year, team_id, player_id,
            embedding_model, embedding_dim, embedding_version, is_active, updated_at
        FROM rag_chunks
        WHERE {where_sql}
        ORDER BY id
        LIMIT %s
        """,
        tuple(params) + (sample_limit,),
    )


def _metadata_conflict_samples(
    conn: psycopg.Connection,
    *,
    sample_limit: int,
) -> List[Dict[str, Any]]:
    return _sample_rows(
        conn,
        METADATA_CONFLICT_WHERE,
        metadata_conflict_params(),
        sample_limit=sample_limit,
    )


def _metadata_fixable_groups(conn: psycopg.Connection) -> List[Dict[str, Any]]:
    return _fetch_rows(
        conn,
        f"""
        SELECT
            source_table,
            season_year,
            embedding_model,
            embedding_dim,
            embedding_version,
            count(*) AS count
        FROM rag_chunks
        WHERE {METADATA_FIX_WHERE}
        GROUP BY source_table, season_year, embedding_model, embedding_dim, embedding_version
        ORDER BY count DESC, source_table, season_year NULLS LAST
        """,
        metadata_fix_params(),
    )


def _apply_cleanup(conn: psycopg.Connection) -> Dict[str, int]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE rag_chunks
            SET embedding_model = %s,
                embedding_dim = %s,
                embedding_version = %s,
                updated_at = now()
            WHERE {METADATA_FIX_WHERE}
            """,
            (
                EXPECTED_EMBEDDING_MODEL,
                EXPECTED_EMBEDDING_DIM,
                EXPECTED_EMBEDDING_VERSION,
                *metadata_fix_params(),
            ),
        )
        metadata_updated = cur.rowcount
        cur.execute(
            f"""
            UPDATE rag_chunks
            SET is_active = false,
                valid_to = COALESCE(valid_to, now()),
                updated_at = now()
            WHERE {ACTIVE_NULL_WHERE}
            """,
        )
        null_deactivated = cur.rowcount
    return {
        "metadata_updated": int(metadata_updated),
        "active_null_deactivated": int(null_deactivated),
    }


def run(
    *,
    apply: bool,
    report_output: str,
    sample_limit: int,
    statement_timeout_ms: int,
    lock_timeout_ms: int,
    expect_metadata_fixable: int | None = None,
    expect_active_null_embeddings: int | None = None,
    expect_metadata_conflicts: int | None = None,
) -> Dict[str, Any]:
    settings = get_settings()
    runtime_errors = _validate_runtime(settings)
    count_expectations = {
        "metadata_fixable": expect_metadata_fixable,
        "metadata_conflicts": expect_metadata_conflicts,
        "active_null_embeddings": expect_active_null_embeddings,
    }
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": not apply,
        "actions": {
            "metadata_fix": (
                "Fill embedding_model, embedding_dim, and embedding_version only for "
                "rows whose existing metadata is blank/stale but compatible with "
                "the 256-d v2 rollout."
            ),
            "active_null_embeddings": (
                "Soft-deactivate active rows with NULL embeddings; do not delete rows "
                "and do not synthesize vectors."
            ),
        },
        "runtime": _runtime_snapshot(settings),
        "expected": {
            "embedding_model": EXPECTED_EMBEDDING_MODEL,
            "embedding_dim": EXPECTED_EMBEDDING_DIM,
            "embedding_version": EXPECTED_EMBEDDING_VERSION,
            "ai_vector_index": EXPECTED_VECTOR_INDEX,
            "ai_vector_quantization": EXPECTED_VECTOR_QUANTIZATION,
        },
        "db": {},
        "runtime_errors": runtime_errors,
        "db_errors": [],
        "before_counts": {},
        "after_counts": {},
        "counts": {},
        "count_expectations": count_expectations,
        "count_mismatches": [],
        "samples": {},
        "applied": {
            "metadata_updated": 0,
            "active_null_deactivated": 0,
        },
        "status": "fail" if runtime_errors else "pending",
    }

    with psycopg.connect(settings.database_url, connect_timeout=30) as conn:
        _configure_session(
            conn,
            statement_timeout_ms=statement_timeout_ms,
            lock_timeout_ms=lock_timeout_ms,
            read_only=not apply,
        )
        embedding_column_type = _fetch_embedding_column_type(conn)
        db_errors = _validate_db_preconditions(embedding_column_type)
        report["db"] = {"embedding_column_type": embedding_column_type}
        report["db_errors"] = db_errors
        before_counts = _collect_counts(conn)
        report["before_counts"] = before_counts
        report["counts"] = before_counts
        if sample_limit > 0:
            report["samples"] = {
                "metadata_fixable_groups": _metadata_fixable_groups(conn),
                "metadata_conflicts": _metadata_conflict_samples(
                    conn,
                    sample_limit=sample_limit,
                )
                if before_counts["metadata_conflicts"]
                else [],
                "active_null_embeddings": _sample_rows(
                    conn,
                    ACTIVE_NULL_WHERE,
                    (),
                    sample_limit=sample_limit,
                ),
            }

        count_mismatches = (
            _count_expectation_mismatches(before_counts, count_expectations)
            if apply
            else []
        )
        report["count_mismatches"] = count_mismatches

        if runtime_errors or db_errors or before_counts["metadata_conflicts"]:
            report["after_counts"] = before_counts
            conn.rollback()
            report["status"] = "fail"
        elif count_mismatches:
            report["after_counts"] = before_counts
            conn.rollback()
            report["status"] = "count_mismatch"
        elif apply:
            report["applied"] = _apply_cleanup(conn)
            after_counts = _collect_counts(conn)
            report["after_counts"] = after_counts
            report["counts"] = after_counts
            conn.commit()
            if (
                after_counts["metadata_fixable"]
                or after_counts["metadata_conflicts"]
                or after_counts["active_null_embeddings"]
            ):
                report["status"] = "applied_with_remaining"
            else:
                report["status"] = "applied"
        else:
            report["after_counts"] = before_counts
            conn.rollback()
            report["status"] = "dry_run"

    _write_json(PROJECT_ROOT / report_output, report)
    return report


def main() -> int:
    args = parse_args()
    report = run(
        apply=args.apply,
        report_output=args.report_output,
        sample_limit=args.sample_limit,
        statement_timeout_ms=args.statement_timeout_ms,
        lock_timeout_ms=args.lock_timeout_ms,
        expect_metadata_fixable=args.expect_metadata_fixable,
        expect_active_null_embeddings=args.expect_active_null_embeddings,
        expect_metadata_conflicts=args.expect_metadata_conflicts,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return (
        0
        if report["status"] not in {"fail", "count_mismatch", "applied_with_remaining"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
