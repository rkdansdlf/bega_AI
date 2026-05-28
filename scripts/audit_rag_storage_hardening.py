#!/usr/bin/env python3
"""Read-only audit for rag_chunks storage hardening readiness."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from app.core.rag_storage import (
    is_search_worthy_content,
    json_dumps,
    scan_sensitive_content,
)

PGVECTOR_SEARCH_PATH = "public, extensions, security"


@dataclass(frozen=True)
class AuditConfig:
    source_tables: Sequence[str]
    season_years: Sequence[int]
    limit: Optional[int]
    scan_limit: Optional[int]
    batch_size: int
    sample_size: int
    active_only: bool
    quality_min_chars: int


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required to run audit_rag_storage_hardening.py. "
            f"Detail: {_PSYCOPG_IMPORT_ERROR}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit rag_chunks metadata, sensitive content, and manual lineage readiness."
    )
    parser.add_argument("--source-table", nargs="*", default=[])
    parser.add_argument(
        "--season-year",
        nargs="*",
        type=int,
        default=[],
        help="Limit audit to specific rag_chunks.season_year values.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=None,
        help=(
            "Limit Python content scanning only. SQL summary counts still cover the full filtered scope. "
            "When omitted, --limit is used for backward compatibility."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--active-only", action="store_true")
    parser.add_argument("--quality-min-chars", type=int, default=50)
    parser.add_argument("--output", default="")
    return parser


def _where_clause(
    config: AuditConfig,
    *,
    alias: str = "",
) -> Tuple[str, List[Any]]:
    prefix = f"{alias}." if alias else ""
    clauses: List[str] = []
    params: List[Any] = []
    if config.source_tables:
        clauses.append(f"{prefix}source_table = ANY(%s)")
        params.append(list(config.source_tables))
    if config.season_years:
        clauses.append(f"{prefix}season_year = ANY(%s)")
        params.append(list(config.season_years))
    if config.active_only:
        clauses.append(f"COALESCE({prefix}is_active, true) = true")
    if not clauses:
        return "", params
    return "WHERE " + " AND ".join(clauses), params


def _fetch_one(conn: Any, query: str, params: Sequence[Any]) -> Dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, tuple(params))
        row = cur.fetchone()
        return dict(row) if row else {}


def _fetch_rows(conn: Any, query: str, params: Sequence[Any]) -> List[Dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, tuple(params))
        return [dict(row) for row in cur.fetchall()]


def fetch_summary(conn: Any, config: AuditConfig) -> Dict[str, Any]:
    where, params = _where_clause(config)
    query = f"""
        SELECT
            count(*) AS total_chunks,
            count(*) FILTER (WHERE COALESCE(is_active, true) = true) AS active_chunks,
            count(*) FILTER (WHERE COALESCE(is_active, true) = false) AS inactive_chunks,
            count(*) FILTER (WHERE embedding IS NULL) AS missing_embedding,
            count(*) FILTER (WHERE topic_key IS NULL OR btrim(topic_key) = '') AS missing_topic_key,
            count(*) FILTER (WHERE source_type IS NULL OR btrim(source_type) = '') AS missing_source_type,
            count(*) FILTER (WHERE source_uri IS NULL OR btrim(source_uri) = '') AS missing_source_uri,
            count(*) FILTER (WHERE quality_score IS NULL) AS missing_quality_score,
            count(*) FILTER (WHERE content_hash IS NULL OR btrim(content_hash) = '') AS missing_content_hash,
            count(*) FILTER (WHERE chunk_hash IS NULL OR btrim(chunk_hash) = '') AS missing_chunk_hash,
            count(*) FILTER (WHERE embedding_model IS NULL OR btrim(embedding_model) = '') AS missing_embedding_model,
            count(*) FILTER (WHERE embedding_dim IS NULL) AS missing_embedding_dim,
            count(*) FILTER (WHERE embedding_version IS NULL) AS missing_embedding_version,
            count(*) FILTER (WHERE chunking_version IS NULL) AS missing_chunking_version,
            count(*) FILTER (WHERE metadata IS NULL OR metadata = '{{}}'::jsonb) AS empty_metadata,
            count(*) FILTER (
                WHERE meta IS NOT NULL
                  AND meta <> '{{}}'::jsonb
                  AND (metadata IS NULL OR metadata = '{{}}'::jsonb)
            ) AS metadata_backfill_needed,
            count(*) FILTER (
                WHERE COALESCE(is_active, true) = true
                  AND valid_to IS NOT NULL
                  AND valid_to <= now()
            ) AS active_past_valid_to,
            count(*) FILTER (
                WHERE COALESCE(is_active, true) = true
                  AND expires_at IS NOT NULL
                  AND expires_at <= now()
            ) AS active_expired,
            count(*) FILTER (
                WHERE COALESCE(is_active, true) = true
                  AND valid_from IS NOT NULL
                  AND valid_from > now()
            ) AS active_future_valid
        FROM rag_chunks
        {where}
    """
    return _fetch_one(conn, query, params)


def fetch_source_type_breakdown(
    conn: Any,
    config: AuditConfig,
) -> List[Dict[str, Any]]:
    where, params = _where_clause(config)
    query = f"""
        SELECT
            COALESCE(NULLIF(source_type, ''), '<missing>') AS source_type,
            count(*) AS chunk_count,
            count(*) FILTER (WHERE COALESCE(is_active, true) = true) AS active_chunks,
            round(avg(quality_score)::numeric, 3) AS avg_quality_score
        FROM rag_chunks
        {where}
        GROUP BY COALESCE(NULLIF(source_type, ''), '<missing>')
        ORDER BY chunk_count DESC, source_type
        LIMIT %s
    """
    return _fetch_rows(conn, query, params + [config.sample_size])


def fetch_missing_field_samples(
    conn: Any,
    config: AuditConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    fields = {
        "topic_key": "topic_key IS NULL OR btrim(topic_key) = ''",
        "source_type": "source_type IS NULL OR btrim(source_type) = ''",
        "source_uri": "source_uri IS NULL OR btrim(source_uri) = ''",
        "quality_score": "quality_score IS NULL",
        "content_hash": "content_hash IS NULL OR btrim(content_hash) = ''",
        "chunk_hash": "chunk_hash IS NULL OR btrim(chunk_hash) = ''",
    }
    base_where, base_params = _where_clause(config)
    samples: Dict[str, List[Dict[str, Any]]] = {}
    for field, condition in fields.items():
        if base_where:
            where = f"{base_where} AND ({condition})"
        else:
            where = f"WHERE {condition}"
        query = f"""
            SELECT id, source_table, source_row_id, title
            FROM rag_chunks
            {where}
            ORDER BY id
            LIMIT %s
        """
        samples[field] = _fetch_rows(conn, query, base_params + [config.sample_size])
    return samples


def scan_rows_for_sensitive_content(
    rows: Sequence[Dict[str, Any]],
    *,
    sample_size: int,
    quality_min_chars: int,
) -> Dict[str, Any]:
    finding_counts: Counter[str] = Counter()
    sensitive_samples: List[Dict[str, Any]] = []
    low_value_samples: List[Dict[str, Any]] = []
    scanned = 0

    for row in rows:
        scanned += 1
        content = str(row.get("content") or "")
        meta = row.get("meta") or {}
        metadata = row.get("metadata") or {}
        findings = sorted(
            set(
                scan_sensitive_content(content)
                + scan_sensitive_content(meta)
                + scan_sensitive_content(metadata)
            )
        )
        for finding in findings:
            finding_counts[finding] += 1
        if findings and len(sensitive_samples) < sample_size:
            sensitive_samples.append(
                {
                    "id": row.get("id"),
                    "source_table": row.get("source_table"),
                    "source_row_id": row.get("source_row_id"),
                    "finding_types": findings,
                }
            )

        if not findings and not is_search_worthy_content(
            content,
            min_chars=quality_min_chars,
        ):
            if len(low_value_samples) < sample_size:
                low_value_samples.append(
                    {
                        "id": row.get("id"),
                        "source_table": row.get("source_table"),
                        "source_row_id": row.get("source_row_id"),
                        "content_preview": content[:160],
                    }
                )

    return {
        "scanned_rows": scanned,
        "finding_counts": dict(sorted(finding_counts.items())),
        "sensitive_samples": sensitive_samples,
        "low_value_samples": low_value_samples,
    }


def fetch_rows_for_python_scan(
    conn: Any,
    config: AuditConfig,
) -> List[Dict[str, Any]]:
    where, params = _where_clause(config)
    query = f"""
        SELECT id, source_table, source_row_id, content, meta, metadata
        FROM rag_chunks
        {where}
        ORDER BY id
    """
    scan_limit = config.scan_limit if config.scan_limit is not None else config.limit
    if scan_limit is not None:
        query += " LIMIT %s"
        params.append(scan_limit)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, tuple(params))
        rows: List[Dict[str, Any]] = []
        while True:
            batch = cur.fetchmany(max(1, config.batch_size))
            if not batch:
                break
            rows.extend(dict(row) for row in batch)
        return rows


def fetch_manual_lineup_issues(
    conn: Any,
    config: AuditConfig,
) -> Dict[str, Any]:
    manual_where, params = _where_clause(config, alias="r")
    issue_condition = """
        r.source_table = 'game_lineups'
        AND (
            r.player_id ~ '[가-힣[:space:]]'
            OR (
                COALESCE(r.metadata->>'source_type', r.meta->>'source_type') = 'manual_lineup'
                AND (
                    r.source_type IS DISTINCT FROM 'manual_lineup'
                    OR r.source_uri IS NULL
                    OR r.source_uri = ''
                    OR r.quality_score IS NULL
                    OR r.quality_score > 0.70
                )
            )
            OR (
                r.source_type = 'manual_lineup'
                AND (r.quality_score IS NULL OR r.quality_score > 0.70)
            )
        )
    """
    if manual_where:
        where = f"{manual_where} AND {issue_condition}"
    else:
        where = f"WHERE {issue_condition}"
    count_query = f"SELECT count(*) AS issue_count FROM rag_chunks r {where}"
    sample_query = f"""
        SELECT
            r.id,
            r.source_row_id,
            r.player_id,
            r.source_type,
            r.source_uri,
            r.quality_score,
            COALESCE(r.metadata->>'source_type', r.meta->>'source_type') AS notes_source_type,
            COALESCE(r.metadata->>'confidence', r.meta->>'confidence') AS confidence
        FROM rag_chunks r
        {where}
        ORDER BY r.id
        LIMIT %s
    """
    count_row = _fetch_one(conn, count_query, params)
    return {
        "issue_count": int(count_row.get("issue_count") or 0),
        "samples": _fetch_rows(conn, sample_query, params + [config.sample_size]),
    }


def build_recommendations(report: Dict[str, Any]) -> List[str]:
    summary = report.get("summary") or {}
    sensitive_scan = report.get("sensitive_scan") or {}
    manual_lineup = report.get("manual_lineup_issues") or {}
    recommendations: List[str] = []

    missing_fields = [
        "missing_topic_key",
        "missing_source_type",
        "missing_source_uri",
        "missing_quality_score",
        "missing_content_hash",
        "missing_chunk_hash",
        "missing_embedding_model",
        "missing_embedding_dim",
    ]
    if any(int(summary.get(field) or 0) > 0 for field in missing_fields):
        recommendations.append(
            "Run backfill_rag_chunk_metadata.py in dry-run first, then apply in staging before production."
        )
    if sensitive_scan.get("finding_counts"):
        recommendations.append(
            "Review sensitive_samples and soft deactivate or re-ingest affected chunks after approval."
        )
    if int(manual_lineup.get("issue_count") or 0) > 0:
        recommendations.append(
            "Review manual_lineup_issues; re-ingest verified operator rows so source_type/source_uri/quality_score are preserved."
        )
    if (
        int(summary.get("active_past_valid_to") or 0) > 0
        or int(summary.get("active_expired") or 0) > 0
    ):
        recommendations.append(
            "Investigate active rows past valid_to/expires_at; retrieval filters hide them, but storage should be cleaned up."
        )
    if int(summary.get("active_future_valid") or 0) > 0:
        recommendations.append(
            "Check active future-valid rows; retrieval filters hide them until valid_from, but source metadata should be intentional."
        )
    if not recommendations:
        recommendations.append(
            "No immediate storage hardening issue found in the audited scope."
        )
    return recommendations


def run_audit(config: AuditConfig) -> Dict[str, Any]:
    _require_psycopg()
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {PGVECTOR_SEARCH_PATH}")
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filters": {
                "source_tables": list(config.source_tables),
                "season_years": list(config.season_years),
                "limit": config.limit,
                "scan_limit": config.scan_limit,
                "active_only": config.active_only,
            },
            "summary": fetch_summary(conn, config),
            "source_type_breakdown": fetch_source_type_breakdown(conn, config),
            "missing_field_samples": fetch_missing_field_samples(conn, config),
            "sensitive_scan": scan_rows_for_sensitive_content(
                fetch_rows_for_python_scan(conn, config),
                sample_size=config.sample_size,
                quality_min_chars=config.quality_min_chars,
            ),
            "manual_lineup_issues": fetch_manual_lineup_issues(conn, config),
        }
    report["recommendations"] = build_recommendations(report)
    return report


def main() -> int:
    args = build_parser().parse_args()
    config = AuditConfig(
        source_tables=args.source_table,
        season_years=args.season_year,
        limit=args.limit,
        scan_limit=args.scan_limit,
        batch_size=max(1, args.batch_size),
        sample_size=max(1, args.sample_size),
        active_only=bool(args.active_only),
        quality_min_chars=max(1, args.quality_min_chars),
    )
    report = run_audit(config)
    output = json_dumps(report)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
