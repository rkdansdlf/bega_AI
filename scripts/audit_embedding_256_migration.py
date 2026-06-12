#!/usr/bin/env python3
"""Read-only audit for the 256-dimensional RAG embedding migration."""

from __future__ import annotations

import argparse
import contextlib
import io
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

from app.config import Settings, get_settings
from app.core.rag_storage import resolve_embedding_model
from app.core.retrieval import _embedding_distance_sql
from scripts.cleanup_embedding_256_warnings import (
    EXPECTED_EMBEDDING_MODEL,
    EXPECTED_VECTOR_INDEX,
    EXPECTED_VECTOR_QUANTIZATION,
    METADATA_CONFLICT_WHERE,
    METADATA_FIX_WHERE,
    metadata_conflict_params,
    metadata_fix_params,
)
from scripts.create_vector_index import run as run_create_vector_index

EXPECTED_EMBED_DIM = 256
EXPECTED_EMBEDDING_VERSION = 2
HALFVEC_INDEX_NAME = "idx_rag_chunks_embedding_halfvec_hnsw"
VECTOR_HNSW_INDEX_NAME = "idx_rag_chunks_embedding_hnsw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit remaining issues after migrating rag_chunks to 256-d embeddings."
    )
    parser.add_argument(
        "--summary-output",
        default="reports/256_migration_audit_summary.json",
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--samples-output",
        default="reports/256_migration_audit_samples.json",
        help="Samples JSON output path.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Max sample rows per anomaly class.",
    )
    parser.add_argument(
        "--statement-timeout-ms",
        type=int,
        default=120_000,
        help="PostgreSQL statement timeout for audit queries.",
    )
    parser.add_argument(
        "--skip-index-dry-run",
        action="store_true",
        help="Skip scripts/create_vector_index.py --dry-run capture.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
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


def _fetch_one(
    conn: psycopg.Connection,
    query: str,
    params: Sequence[Any] = (),
) -> Dict[str, Any]:
    rows = _fetch_rows(conn, query, params)
    return rows[0] if rows else {}


def _configure_audit_session(
    conn: psycopg.Connection,
    *,
    statement_timeout_ms: int,
) -> None:
    with conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(
            "SELECT set_config('statement_timeout', %s, false)",
            (str(statement_timeout_ms),),
        )


def _fetch_rows_sample(
    conn: psycopg.Connection,
    query: str,
    params: Sequence[Any],
    *,
    statement_timeout_ms: int,
) -> Dict[str, Any]:
    try:
        return {"status": "ok", "rows": _fetch_rows(conn, query, params)}
    except psycopg.Error as exc:
        conn.rollback()
        _configure_audit_session(
            conn,
            statement_timeout_ms=statement_timeout_ms,
        )
        return {
            "status": "error",
            "error_type": exc.__class__.__name__,
            "error": str(exc).strip(),
            "rows": [],
        }


def _skipped_sample(reason: str) -> Dict[str, Any]:
    return {"status": "skipped", "reason": reason, "rows": []}


def _settings_snapshot(settings: Settings) -> Dict[str, Any]:
    return {
        "embed_provider": settings.embed_provider,
        "embed_model": settings.embed_model,
        "openrouter_embed_model": settings.openrouter_embed_model,
        "openai_embed_model": settings.openai_embed_model,
        "embed_dim": settings.embed_dim,
        "rag_embedding_version": settings.rag_embedding_version,
        "resolved_embedding_model": resolve_embedding_model(settings),
        "ai_vector_index": settings.ai_vector_index,
        "ai_vector_quantization": settings.ai_vector_quantization,
    }


def _settings_defaults() -> Dict[str, Any]:
    fields = Settings.model_fields
    names = [
        "embed_provider",
        "embed_model",
        "openrouter_embed_model",
        "openai_embed_model",
        "embed_dim",
        "rag_embedding_version",
        "ai_vector_index",
        "ai_vector_quantization",
    ]
    defaults: Dict[str, Any] = {}
    for name in names:
        field = fields.get(name)
        defaults[name] = getattr(field, "default", None) if field else None
    return defaults


def _fetch_db_audit(conn: psycopg.Connection, *, sample_limit: int) -> Dict[str, Any]:
    summary = _fetch_one(
        conn,
        """
        SELECT
            count(*) AS total,
            count(*) FILTER (WHERE is_active) AS active,
            count(*) FILTER (WHERE embedding IS NULL) AS null_embeddings,
            count(*) FILTER (WHERE is_active AND embedding IS NULL) AS active_null_embeddings,
            count(*) FILTER (WHERE embedding IS NOT NULL) AS embedded,
            count(*) FILTER (
                WHERE embedding IS NOT NULL AND coalesce(embedding_dim, -1) <> %s
            ) AS metadata_dim_not_256,
            count(*) FILTER (
                WHERE is_active AND embedding IS NOT NULL AND coalesce(embedding_dim, -1) <> %s
            ) AS active_metadata_dim_not_256,
            count(*) FILTER (
                WHERE embedding IS NOT NULL AND coalesce(embedding_version, -1) <> %s
            ) AS version_not_2,
            count(*) FILTER (
                WHERE embedding IS NOT NULL AND embedding_model IS NULL
            ) AS embedded_missing_model
        FROM rag_chunks
        """,
        (EXPECTED_EMBED_DIM, EXPECTED_EMBED_DIM, EXPECTED_EMBEDDING_VERSION),
    )
    summary.update(
        _fetch_one(
            conn,
            f"""
            SELECT
                count(*) AS metadata_fixable,
                count(*) FILTER (WHERE is_active) AS active_metadata_fixable
            FROM rag_chunks
            WHERE {METADATA_FIX_WHERE}
            """,
            metadata_fix_params(),
        )
    )
    summary.update(
        _fetch_one(
            conn,
            f"""
            SELECT
                count(*) AS metadata_conflicts,
                count(*) FILTER (WHERE is_active) AS active_metadata_conflicts
            FROM rag_chunks
            WHERE {METADATA_CONFLICT_WHERE}
            """,
            metadata_conflict_params(),
        )
    )

    column_type = _fetch_one(
        conn,
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS embedding_type
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'rag_chunks' AND a.attname = 'embedding'
        """,
    )
    pgvector = _fetch_one(
        conn,
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
    )
    indexes = _fetch_rows(
        conn,
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'rag_chunks' AND indexname LIKE '%%embedding%%'
        ORDER BY indexname
        """,
    )
    return {
        "summary": summary,
        "column_type": column_type,
        "pgvector": pgvector,
        "indexes": indexes,
    }


def _fetch_samples(
    conn: psycopg.Connection,
    *,
    sample_limit: int,
    statement_timeout_ms: int,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    if sample_limit <= 0:
        return {}
    stale_where = "embedding IS NOT NULL AND coalesce(embedding_dim, -1) <> %s"
    base_columns = """
        id, source_table, source_row_id, season_year, team_id, player_id,
        embedding_model, embedding_dim, embedding_version, is_active, updated_at
    """
    has_stale_dim = int(summary.get("metadata_dim_not_256") or 0) > 0
    has_metadata_fixable = int(summary.get("metadata_fixable") or 0) > 0
    has_metadata_conflicts = int(summary.get("metadata_conflicts") or 0) > 0
    has_active_null = int(summary.get("active_null_embeddings") or 0) > 0
    has_version_mismatch = int(summary.get("version_not_2") or 0) > 0
    has_missing_model = int(summary.get("embedded_missing_model") or 0) > 0
    return {
        "metadata_dim_mismatch_groups": (
            _fetch_rows_sample(
                conn,
                f"""
            SELECT
                source_table,
                season_year,
                embedding_model,
                embedding_dim,
                embedding_version,
                count(*) AS count,
                min(updated_at) AS min_updated_at,
                max(updated_at) AS max_updated_at
            FROM rag_chunks
            WHERE {stale_where}
            GROUP BY source_table, season_year, embedding_model, embedding_dim, embedding_version
            ORDER BY count DESC, source_table, season_year NULLS LAST
            LIMIT %s
            """,
                (EXPECTED_EMBED_DIM, sample_limit),
                statement_timeout_ms=statement_timeout_ms,
            )
            if has_stale_dim
            else _skipped_sample("metadata_dim_not_256 is 0")
        ),
        "metadata_dim_mismatch_samples": (
            _fetch_rows_sample(
                conn,
                f"""
            SELECT {base_columns}
            FROM rag_chunks
            WHERE {stale_where}
            LIMIT %s
            """,
                (EXPECTED_EMBED_DIM, sample_limit),
                statement_timeout_ms=statement_timeout_ms,
            )
            if has_stale_dim
            else _skipped_sample("metadata_dim_not_256 is 0")
        ),
        "metadata_fixable_groups": (
            _fetch_rows_sample(
                conn,
                f"""
            SELECT
                source_table,
                season_year,
                embedding_model,
                embedding_dim,
                embedding_version,
                count(*) AS count,
                min(updated_at) AS min_updated_at,
                max(updated_at) AS max_updated_at
            FROM rag_chunks
            WHERE {METADATA_FIX_WHERE}
            GROUP BY source_table, season_year, embedding_model, embedding_dim, embedding_version
            ORDER BY count DESC, source_table, season_year NULLS LAST
            LIMIT %s
            """,
                (*metadata_fix_params(), sample_limit),
                statement_timeout_ms=statement_timeout_ms,
            )
            if has_metadata_fixable
            else _skipped_sample("metadata_fixable is 0")
        ),
        "metadata_fixable_samples": (
            _fetch_rows_sample(
                conn,
                f"""
            SELECT {base_columns}
            FROM rag_chunks
            WHERE {METADATA_FIX_WHERE}
            LIMIT %s
            """,
                (*metadata_fix_params(), sample_limit),
                statement_timeout_ms=statement_timeout_ms,
            )
            if has_metadata_fixable
            else _skipped_sample("metadata_fixable is 0")
        ),
        "metadata_conflict_samples": (
            _fetch_rows_sample(
                conn,
                f"""
            SELECT {base_columns}
            FROM rag_chunks
            WHERE {METADATA_CONFLICT_WHERE}
            LIMIT %s
            """,
                (*metadata_conflict_params(), sample_limit),
                statement_timeout_ms=statement_timeout_ms,
            )
            if has_metadata_conflicts
            else _skipped_sample("metadata_conflicts is 0")
        ),
        "null_embedding_samples": (
            _fetch_rows_sample(
                conn,
                f"""
            SELECT {base_columns}
            FROM rag_chunks
            WHERE is_active AND embedding IS NULL
            LIMIT %s
            """,
                (sample_limit,),
                statement_timeout_ms=statement_timeout_ms,
            )
            if has_active_null
            else _skipped_sample("active_null_embeddings is 0")
        ),
        "version_mismatch_samples": (
            _fetch_rows_sample(
                conn,
                f"""
            SELECT {base_columns}
            FROM rag_chunks
            WHERE embedding IS NOT NULL
              AND coalesce(embedding_version, -1) <> %s
            LIMIT %s
            """,
                (EXPECTED_EMBEDDING_VERSION, sample_limit),
                statement_timeout_ms=statement_timeout_ms,
            )
            if has_version_mismatch
            else _skipped_sample("version_not_2 is 0")
        ),
        "missing_model_samples": (
            _fetch_rows_sample(
                conn,
                f"""
            SELECT {base_columns}
            FROM rag_chunks
            WHERE embedding IS NOT NULL
              AND embedding_model IS NULL
            LIMIT %s
            """,
                (sample_limit,),
                statement_timeout_ms=statement_timeout_ms,
            )
            if has_missing_model
            else _skipped_sample("embedded_missing_model is 0")
        ),
    }


def _capture_index_dry_run(*, drop_vector_hnsw: bool = False) -> Dict[str, Any]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = run_create_vector_index(
            dry_run=True,
            drop_ivfflat=False,
            drop_vector_hnsw=drop_vector_hnsw,
            m=16,
            ef_construction=64,
            ef_search=100,
        )
    return {
        "exit_code": exit_code,
        "output": stdout.getvalue().splitlines(),
    }


def _has_duplicate_hnsw_indexes(db_audit: Dict[str, Any]) -> bool:
    index_names = {row.get("indexname") for row in db_audit["indexes"]}
    return {HALFVEC_INDEX_NAME, VECTOR_HNSW_INDEX_NAME}.issubset(index_names)


def _build_findings(
    *,
    settings: Settings,
    defaults: Dict[str, Any],
    db_audit: Dict[str, Any],
    distance_sql: str,
    index_dry_run: Dict[str, Any] | None,
    index_cleanup_dry_run: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    runtime = _settings_snapshot(settings)
    summary = db_audit["summary"]
    embedding_type = db_audit["column_type"].get("embedding_type")
    index_names = {row.get("indexname") for row in db_audit["indexes"]}

    def add(severity: str, code: str, message: str, value: Any = None) -> None:
        item: Dict[str, Any] = {
            "severity": severity,
            "code": code,
            "message": message,
        }
        if value is not None:
            item["value"] = value
        findings.append(item)

    if runtime["embed_dim"] != EXPECTED_EMBED_DIM:
        add("fail", "runtime_embed_dim", "Runtime EMBED_DIM is not 256.", runtime)
    if runtime["rag_embedding_version"] != EXPECTED_EMBEDDING_VERSION:
        add(
            "fail",
            "runtime_embedding_version",
            "Runtime RAG_EMBEDDING_VERSION is not 2.",
            runtime,
        )
    if runtime["resolved_embedding_model"] != EXPECTED_EMBEDDING_MODEL:
        add(
            "fail",
            "runtime_embedding_model",
            (
                "Runtime resolved embedding model does not match the "
                "post-rollout metadata target."
            ),
            {
                "runtime": runtime["resolved_embedding_model"],
                "expected": EXPECTED_EMBEDDING_MODEL,
            },
        )
    vector_index = str(runtime["ai_vector_index"] or "").lower().strip()
    if vector_index != EXPECTED_VECTOR_INDEX:
        add(
            "fail",
            "runtime_vector_index",
            "Runtime AI_VECTOR_INDEX is not hnsw.",
            {
                "runtime": runtime["ai_vector_index"],
                "expected": EXPECTED_VECTOR_INDEX,
            },
        )
    vector_quantization = str(runtime["ai_vector_quantization"] or "").lower().strip()
    if vector_quantization != EXPECTED_VECTOR_QUANTIZATION:
        add(
            "fail",
            "runtime_vector_quantization",
            "Runtime AI_VECTOR_QUANTIZATION is not halfvec.",
            {
                "runtime": runtime["ai_vector_quantization"],
                "expected": EXPECTED_VECTOR_QUANTIZATION,
            },
        )
    if not str(embedding_type or "").endswith("vector(256)"):
        add(
            "fail",
            "embedding_column_type",
            "rag_chunks.embedding is not vector(256).",
            embedding_type,
        )

    if int(summary.get("metadata_conflicts") or 0) > 0:
        add(
            "fail",
            "metadata_conflicts",
            (
                "Rows have embedding metadata that is not safe to auto-correct; "
                "inspect samples before cleanup."
            ),
            summary.get("metadata_conflicts"),
        )
    if int(summary.get("metadata_fixable") or 0) > 0:
        add(
            "warning",
            "metadata_fixable",
            (
                "Rows have stale/missing 256-d v2 metadata that "
                "cleanup_embedding_256_warnings.py can fill without re-embedding."
            ),
            summary.get("metadata_fixable"),
        )
    if int(summary.get("active_metadata_dim_not_256") or 0) > 0:
        add(
            "warning",
            "metadata_dim_mismatch",
            "Active rows with embeddings still have embedding_dim metadata not equal to 256.",
            summary.get("active_metadata_dim_not_256"),
        )
    if int(summary.get("active_null_embeddings") or 0) > 0:
        add(
            "warning",
            "null_embeddings",
            "Active rag_chunks rows contain NULL embedding.",
            summary.get("active_null_embeddings"),
        )
    if int(summary.get("version_not_2") or 0) > 0:
        add(
            "fail",
            "embedding_version_mismatch",
            "Rows with embeddings have embedding_version metadata not equal to 2.",
            summary.get("version_not_2"),
        )
    if int(summary.get("embedded_missing_model") or 0) > 0:
        add(
            "warning",
            "embedded_missing_model",
            (
                "Rows with embeddings have NULL embedding_model; if "
                "metadata_fixable is nonzero, use the cleanup script."
            ),
            summary.get("embedded_missing_model"),
        )

    if vector_quantization == EXPECTED_VECTOR_QUANTIZATION:
        if HALFVEC_INDEX_NAME not in index_names:
            add(
                "fail",
                "missing_halfvec_hnsw",
                "AI_VECTOR_QUANTIZATION=halfvec but halfvec HNSW index is missing.",
            )
        if "halfvec(256)" not in distance_sql:
            add(
                "fail",
                "retrieval_sql_not_halfvec",
                "Retrieval distance SQL does not use halfvec(256).",
                distance_sql,
            )
    if _has_duplicate_hnsw_indexes(db_audit):
        add(
            "warning",
            "duplicate_hnsw_indexes",
            (
                "Both halfvec and vector HNSW embedding indexes exist; dry-run "
                "and then drop the vector HNSW index after confirming halfvec retrieval."
            ),
        )

    drift = {
        key: {"runtime": runtime.get(key), "default": defaults.get(key)}
        for key in runtime
        if key in defaults and runtime.get(key) != defaults.get(key)
    }
    if drift:
        add(
            "info",
            "env_overrides_defaults",
            "Runtime environment overrides code defaults for some settings.",
            drift,
        )

    if index_dry_run and index_dry_run.get("exit_code") not in {0, None}:
        add(
            "fail",
            "index_dry_run_failed",
            "create_vector_index.py --dry-run returned non-zero.",
            index_dry_run.get("exit_code"),
        )
    if index_cleanup_dry_run and index_cleanup_dry_run.get("exit_code") not in {
        0,
        None,
    }:
        add(
            "fail",
            "index_cleanup_dry_run_failed",
            "create_vector_index.py --dry-run --drop-vector-hnsw returned non-zero.",
            index_cleanup_dry_run.get("exit_code"),
        )
    return findings


def _overall_status(findings: Sequence[Dict[str, Any]]) -> str:
    if any(item.get("severity") == "fail" for item in findings):
        return "fail"
    if any(item.get("severity") == "warning" for item in findings):
        return "warning"
    return "pass"


def main() -> int:
    args = parse_args()
    settings = get_settings()
    defaults = _settings_defaults()
    distance_sql = _embedding_distance_sql(settings)

    with psycopg.connect(settings.database_url, connect_timeout=30) as conn:
        _configure_audit_session(
            conn,
            statement_timeout_ms=args.statement_timeout_ms,
        )
        db_audit = _fetch_db_audit(conn, sample_limit=args.sample_limit)
        samples = _fetch_samples(
            conn,
            sample_limit=args.sample_limit,
            statement_timeout_ms=args.statement_timeout_ms,
            summary=db_audit["summary"],
        )
        conn.rollback()

    index_dry_run = None
    index_cleanup_dry_run = None
    if not args.skip_index_dry_run:
        index_dry_run = _capture_index_dry_run()
        if _has_duplicate_hnsw_indexes(db_audit):
            index_cleanup_dry_run = _capture_index_dry_run(drop_vector_hnsw=True)

    findings = _build_findings(
        settings=settings,
        defaults=defaults,
        db_audit=db_audit,
        distance_sql=distance_sql,
        index_dry_run=index_dry_run,
        index_cleanup_dry_run=index_cleanup_dry_run,
    )
    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit": "256_migration",
        "status": _overall_status(findings),
        "expected": {
            "embedding_model": EXPECTED_EMBEDDING_MODEL,
            "embed_dim": EXPECTED_EMBED_DIM,
            "rag_embedding_version": EXPECTED_EMBEDDING_VERSION,
            "ai_vector_index": EXPECTED_VECTOR_INDEX,
            "ai_vector_quantization": EXPECTED_VECTOR_QUANTIZATION,
        },
        "runtime": _settings_snapshot(settings),
        "code_defaults": defaults,
        "retrieval_distance_sql": distance_sql,
        "db": db_audit,
        "index_dry_run": index_dry_run,
        "index_cleanup_dry_run": index_cleanup_dry_run,
        "findings": findings,
    }
    samples_payload = {
        "generated_at": summary_payload["generated_at"],
        "audit": "256_migration",
        "sample_limit": args.sample_limit,
        "samples": samples,
    }

    _write_json(PROJECT_ROOT / args.summary_output, summary_payload)
    _write_json(PROJECT_ROOT / args.samples_output, samples_payload)
    print(f"summary saved: {args.summary_output}")
    print(f"samples saved: {args.samples_output}")
    print(f"status: {summary_payload['status']}")
    return 0 if summary_payload["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
