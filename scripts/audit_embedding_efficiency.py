#!/usr/bin/env python3
"""Audit rag_chunks for duplicate and low-value embeddings."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings

_PART_SUFFIX_RE = re.compile(r"#part\d+$")
_WHITESPACE_RE = re.compile(r"\s+")
_SOURCE_LINE_RE = re.compile(r"(?m)^(?:\s*)출처:.*$")


@dataclass(frozen=True)
class AuditFilters:
    season_year: Optional[int] = None
    source_table: Optional[str] = None


def base_source_row_id(source_row_id: str) -> str:
    """Strip the chunk suffix used for multi-part source rows."""
    if not source_row_id:
        return ""
    return _PART_SUFFIX_RE.sub("", source_row_id)


def normalize_content(text: str) -> str:
    """Normalize content for duplicate detection."""
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def strip_source_line(text: str) -> str:
    """Remove trailing '출처:' lines from content before length checks."""
    if not text:
        return ""
    stripped = _SOURCE_LINE_RE.sub("", text)
    return stripped.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit rag_chunks for duplicate or low-value embeddings."
    )
    parser.add_argument(
        "--season-year",
        type=int,
        default=None,
        help="Optional season_year filter.",
    )
    parser.add_argument(
        "--source-table",
        default="",
        help="Optional source_table filter.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Max samples per section.",
    )
    parser.add_argument(
        "--short-body-threshold",
        type=int,
        default=120,
        help="Flag chunks whose body minus source line is this short or shorter.",
    )
    parser.add_argument(
        "--tail-threshold",
        type=int,
        default=140,
        help="Flag final multi-part chunks whose content length is this short or shorter.",
    )
    parser.add_argument(
        "--many-parts-threshold",
        type=int,
        default=5,
        help="Flag documents split into at least this many chunks.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def _build_where_clause(
    filters: AuditFilters,
    *,
    alias: str = "",
) -> Tuple[str, List[Any]]:
    prefix = f"{alias}." if alias else ""
    clauses: List[str] = []
    params: List[Any] = []
    if filters.season_year is not None:
        clauses.append(f"{prefix}season_year = %s")
        params.append(filters.season_year)
    if filters.source_table:
        clauses.append(f"{prefix}source_table = %s")
        params.append(filters.source_table)
    if not clauses:
        return "", params
    return "WHERE " + " AND ".join(clauses), params


def _trim_json_samples(rows: Sequence[Dict[str, Any]], keys: Sequence[str], top: int) -> None:
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, list):
                row[key] = value[:top]


def _fetch_rows(
    conn: psycopg.Connection,
    query: str,
    params: Sequence[Any],
) -> List[Dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def _fetch_one(
    conn: psycopg.Connection,
    query: str,
    params: Sequence[Any],
) -> Dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else {}


def fetch_summary(conn: psycopg.Connection, filters: AuditFilters) -> Dict[str, Any]:
    where_clause, params = _build_where_clause(filters)
    query = f"""
        SELECT
            count(*) AS total_chunks,
            count(*) FILTER (WHERE embedding IS NULL) AS null_embedding_chunks,
            count(*) FILTER (WHERE source_row_id ~ '#part[0-9]+$') AS multipart_chunks,
            count(
                DISTINCT source_table || '|' || regexp_replace(source_row_id, '#part[0-9]+$', '')
            ) AS distinct_documents,
            round(avg(char_length(content))::numeric, 1) AS avg_content_chars
        FROM rag_chunks
        {where_clause}
    """
    summary = _fetch_one(conn, query, params)
    total_chunks = int(summary.get("total_chunks") or 0)
    distinct_documents = int(summary.get("distinct_documents") or 0)
    if summary.get("avg_content_chars") is not None:
        summary["avg_content_chars"] = float(summary["avg_content_chars"])
    avg_chunks_per_document = (
        round(total_chunks / distinct_documents, 3) if distinct_documents else 0.0
    )
    summary["avg_chunks_per_document"] = avg_chunks_per_document
    return summary


def fetch_top_tables(
    conn: psycopg.Connection,
    filters: AuditFilters,
    *,
    top: int,
) -> List[Dict[str, Any]]:
    where_clause, params = _build_where_clause(filters)
    query = f"""
        SELECT
            source_table,
            count(*) AS chunk_count,
            count(DISTINCT regexp_replace(source_row_id, '#part[0-9]+$', '')) AS document_count,
            round(avg(char_length(content))::numeric, 1) AS avg_content_chars
        FROM rag_chunks
        {where_clause}
        GROUP BY source_table
        ORDER BY chunk_count DESC, source_table
        LIMIT %s
    """
    return _fetch_rows(conn, query, params + [top])


def fetch_duplicate_groups(
    conn: psycopg.Connection,
    filters: AuditFilters,
    *,
    top: int,
) -> Dict[str, Any]:
    where_clause, params = _build_where_clause(filters)
    count_query = f"""
        WITH normalized AS (
            SELECT
                source_table,
                regexp_replace(source_row_id, '#part[0-9]+$', '') AS base_source_row_id,
                md5(lower(trim(regexp_replace(content, E'\\s+', ' ', 'g')))) AS content_hash
            FROM rag_chunks
            {where_clause}
        )
        SELECT count(*) AS duplicate_group_count
        FROM (
            SELECT source_table, content_hash
            FROM normalized
            GROUP BY source_table, content_hash
            HAVING count(*) > 1
        ) groups
    """
    sample_query = f"""
        WITH normalized AS (
            SELECT
                source_table,
                regexp_replace(source_row_id, '#part[0-9]+$', '') AS base_source_row_id,
                source_row_id,
                left(regexp_replace(content, E'\\s+', ' ', 'g'), 220) AS preview,
                md5(lower(trim(regexp_replace(content, E'\\s+', ' ', 'g')))) AS content_hash
            FROM rag_chunks
            {where_clause}
        )
        SELECT
            source_table,
            content_hash,
            count(*) AS chunk_count,
            count(DISTINCT base_source_row_id) AS distinct_documents,
            min(preview) AS preview,
            json_agg(source_row_id ORDER BY source_row_id) AS sample_source_row_ids
        FROM normalized
        GROUP BY source_table, content_hash
        HAVING count(*) > 1
        ORDER BY chunk_count DESC, distinct_documents DESC, source_table, content_hash
        LIMIT %s
    """
    count_row = _fetch_one(conn, count_query, params)
    samples = _fetch_rows(conn, sample_query, params + [top])
    _trim_json_samples(samples, ["sample_source_row_ids"], top)
    return {
        "group_count": int(count_row.get("duplicate_group_count") or 0),
        "samples": samples,
    }


def fetch_cross_document_duplicates(
    conn: psycopg.Connection,
    filters: AuditFilters,
    *,
    top: int,
) -> Dict[str, Any]:
    where_clause, params = _build_where_clause(filters)
    count_query = f"""
        WITH normalized AS (
            SELECT
                source_table,
                regexp_replace(source_row_id, '#part[0-9]+$', '') AS base_source_row_id,
                md5(lower(trim(regexp_replace(content, E'\\s+', ' ', 'g')))) AS content_hash
            FROM rag_chunks
            {where_clause}
        )
        SELECT count(*) AS cross_document_duplicate_group_count
        FROM (
            SELECT content_hash
            FROM normalized
            GROUP BY content_hash
            HAVING count(DISTINCT source_table || '|' || base_source_row_id) > 1
        ) groups
    """
    sample_query = f"""
        WITH normalized AS (
            SELECT
                source_table,
                regexp_replace(source_row_id, '#part[0-9]+$', '') AS base_source_row_id,
                source_row_id,
                left(regexp_replace(content, E'\\s+', ' ', 'g'), 220) AS preview,
                md5(lower(trim(regexp_replace(content, E'\\s+', ' ', 'g')))) AS content_hash
            FROM rag_chunks
            {where_clause}
        )
        SELECT
            content_hash,
            count(*) AS chunk_count,
            count(DISTINCT source_table || '|' || base_source_row_id) AS distinct_documents,
            count(DISTINCT source_table) AS distinct_source_tables,
            min(preview) AS preview,
            json_agg(
                json_build_object(
                    'source_table', source_table,
                    'source_row_id', source_row_id
                )
                ORDER BY source_table, source_row_id
            ) AS samples
        FROM normalized
        GROUP BY content_hash
        HAVING count(DISTINCT source_table || '|' || base_source_row_id) > 1
        ORDER BY distinct_documents DESC, chunk_count DESC, content_hash
        LIMIT %s
    """
    count_row = _fetch_one(conn, count_query, params)
    samples = _fetch_rows(conn, sample_query, params + [top])
    _trim_json_samples(samples, ["samples"], top)
    return {
        "group_count": int(count_row.get("cross_document_duplicate_group_count") or 0),
        "samples": samples,
    }


def fetch_short_body_suspects(
    conn: psycopg.Connection,
    filters: AuditFilters,
    *,
    threshold: int,
    top: int,
) -> Dict[str, Any]:
    where_clause, params = _build_where_clause(filters)
    count_query = f"""
        WITH stripped AS (
            SELECT
                btrim(regexp_replace(content, E'(^|\\n)출처:.*$', '', 'g')) AS body
            FROM rag_chunks
            {where_clause}
        )
        SELECT count(*) AS short_body_chunk_count
        FROM stripped
        WHERE char_length(body) <= %s
    """
    sample_query = f"""
        WITH stripped AS (
            SELECT
                source_table,
                source_row_id,
                char_length(content) AS content_chars,
                btrim(regexp_replace(content, E'(^|\\n)출처:.*$', '', 'g')) AS body,
                left(regexp_replace(content, E'\\s+', ' ', 'g'), 220) AS preview
            FROM rag_chunks
            {where_clause}
        )
        SELECT
            source_table,
            source_row_id,
            content_chars,
            char_length(body) AS body_chars,
            preview
        FROM stripped
        WHERE char_length(body) <= %s
        ORDER BY body_chars ASC, content_chars ASC, source_table, source_row_id
        LIMIT %s
    """
    count_row = _fetch_one(conn, count_query, params + [threshold])
    samples = _fetch_rows(conn, sample_query, params + [threshold, top])
    return {
        "count": int(count_row.get("short_body_chunk_count") or 0),
        "samples": samples,
    }


def fetch_small_tail_suspects(
    conn: psycopg.Connection,
    filters: AuditFilters,
    *,
    threshold: int,
    top: int,
) -> Dict[str, Any]:
    where_clause, params = _build_where_clause(filters)
    count_query = f"""
        WITH parts AS (
            SELECT
                source_table,
                regexp_replace(source_row_id, '#part[0-9]+$', '') AS base_source_row_id,
                source_row_id,
                COALESCE(
                    ((regexp_match(source_row_id, '#part([0-9]+)$'))[1])::int,
                    1
                ) AS part_index,
                char_length(content) AS content_chars
            FROM rag_chunks
            {where_clause}
        ),
        ranked AS (
            SELECT
                *,
                max(part_index) OVER (
                    PARTITION BY source_table, base_source_row_id
                ) AS max_part_index
            FROM parts
        )
        SELECT count(*) AS small_tail_chunk_count
        FROM ranked
        WHERE max_part_index > 1
          AND part_index = max_part_index
          AND content_chars <= %s
    """
    sample_query = f"""
        WITH parts AS (
            SELECT
                source_table,
                regexp_replace(source_row_id, '#part[0-9]+$', '') AS base_source_row_id,
                source_row_id,
                COALESCE(
                    ((regexp_match(source_row_id, '#part([0-9]+)$'))[1])::int,
                    1
                ) AS part_index,
                char_length(content) AS content_chars,
                left(regexp_replace(content, E'\\s+', ' ', 'g'), 220) AS preview
            FROM rag_chunks
            {where_clause}
        ),
        ranked AS (
            SELECT
                *,
                max(part_index) OVER (
                    PARTITION BY source_table, base_source_row_id
                ) AS max_part_index
            FROM parts
        )
        SELECT
            source_table,
            base_source_row_id,
            source_row_id,
            max_part_index,
            content_chars,
            preview
        FROM ranked
        WHERE max_part_index > 1
          AND part_index = max_part_index
          AND content_chars <= %s
        ORDER BY content_chars ASC, source_table, source_row_id
        LIMIT %s
    """
    count_row = _fetch_one(conn, count_query, params + [threshold])
    samples = _fetch_rows(conn, sample_query, params + [threshold, top])
    return {
        "count": int(count_row.get("small_tail_chunk_count") or 0),
        "samples": samples,
    }


def fetch_many_parts_documents(
    conn: psycopg.Connection,
    filters: AuditFilters,
    *,
    threshold: int,
    top: int,
) -> Dict[str, Any]:
    where_clause, params = _build_where_clause(filters)
    count_query = f"""
        WITH grouped AS (
            SELECT
                source_table,
                regexp_replace(source_row_id, '#part[0-9]+$', '') AS base_source_row_id,
                count(*) AS chunk_count
            FROM rag_chunks
            {where_clause}
            GROUP BY source_table, regexp_replace(source_row_id, '#part[0-9]+$', '')
        )
        SELECT count(*) AS many_parts_document_count
        FROM grouped
        WHERE chunk_count >= %s
    """
    sample_query = f"""
        WITH grouped AS (
            SELECT
                source_table,
                regexp_replace(source_row_id, '#part[0-9]+$', '') AS base_source_row_id,
                count(*) AS chunk_count
            FROM rag_chunks
            {where_clause}
            GROUP BY source_table, regexp_replace(source_row_id, '#part[0-9]+$', '')
        )
        SELECT
            source_table,
            base_source_row_id,
            chunk_count
        FROM grouped
        WHERE chunk_count >= %s
        ORDER BY chunk_count DESC, source_table, base_source_row_id
        LIMIT %s
    """
    count_row = _fetch_one(conn, count_query, params + [threshold])
    samples = _fetch_rows(conn, sample_query, params + [threshold, top])
    return {
        "count": int(count_row.get("many_parts_document_count") or 0),
        "samples": samples,
    }


def fetch_metadata_heavy_tables(
    conn: psycopg.Connection,
    filters: AuditFilters,
    *,
    top: int,
) -> List[Dict[str, Any]]:
    where_clause, params = _build_where_clause(filters)
    query = f"""
        SELECT
            source_table,
            count(*) AS chunk_count,
            sum(CASE WHEN content ~ E'(^|\\n)id: ' THEN 1 ELSE 0 END) AS with_id,
            sum(CASE WHEN content ~ E'(^|\\n)created_at: ' THEN 1 ELSE 0 END) AS with_created_at,
            sum(CASE WHEN content ~ E'(^|\\n)updated_at: ' THEN 1 ELSE 0 END) AS with_updated_at,
            sum(CASE WHEN content ~ E'(^|\\n)season_lookup_id: ' THEN 1 ELSE 0 END) AS with_season_lookup_id
        FROM rag_chunks
        {where_clause}
        GROUP BY source_table
        HAVING
            sum(CASE WHEN content ~ E'(^|\\n)id: ' THEN 1 ELSE 0 END) +
            sum(CASE WHEN content ~ E'(^|\\n)created_at: ' THEN 1 ELSE 0 END) +
            sum(CASE WHEN content ~ E'(^|\\n)updated_at: ' THEN 1 ELSE 0 END) +
            sum(CASE WHEN content ~ E'(^|\\n)season_lookup_id: ' THEN 1 ELSE 0 END) > 0
        ORDER BY
            (
                sum(CASE WHEN content ~ E'(^|\\n)id: ' THEN 1 ELSE 0 END) +
                sum(CASE WHEN content ~ E'(^|\\n)created_at: ' THEN 1 ELSE 0 END) +
                sum(CASE WHEN content ~ E'(^|\\n)updated_at: ' THEN 1 ELSE 0 END) +
                sum(CASE WHEN content ~ E'(^|\\n)season_lookup_id: ' THEN 1 ELSE 0 END)
            ) DESC,
            source_table
        LIMIT %s
    """
    return _fetch_rows(conn, query, params + [top])


def build_report(
    conn: psycopg.Connection,
    filters: AuditFilters,
    *,
    top: int,
    short_body_threshold: int,
    tail_threshold: int,
    many_parts_threshold: int,
) -> Dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "season_year": filters.season_year,
            "source_table": filters.source_table or None,
        },
        "thresholds": {
            "top": top,
            "short_body_threshold": short_body_threshold,
            "tail_threshold": tail_threshold,
            "many_parts_threshold": many_parts_threshold,
        },
        "summary": fetch_summary(conn, filters),
        "top_tables": fetch_top_tables(conn, filters, top=top),
        "duplicate_content_groups": fetch_duplicate_groups(conn, filters, top=top),
        "cross_document_duplicate_groups": fetch_cross_document_duplicates(
            conn, filters, top=top
        ),
        "short_body_suspects": fetch_short_body_suspects(
            conn,
            filters,
            threshold=short_body_threshold,
            top=top,
        ),
        "small_tail_suspects": fetch_small_tail_suspects(
            conn,
            filters,
            threshold=tail_threshold,
            top=top,
        ),
        "many_parts_documents": fetch_many_parts_documents(
            conn,
            filters,
            threshold=many_parts_threshold,
            top=top,
        ),
        "metadata_heavy_tables": fetch_metadata_heavy_tables(conn, filters, top=top),
    }


def print_summary(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    duplicate_groups = report["duplicate_content_groups"]["group_count"]
    cross_document_groups = report["cross_document_duplicate_groups"]["group_count"]
    short_body_count = report["short_body_suspects"]["count"]
    small_tail_count = report["small_tail_suspects"]["count"]
    many_parts_count = report["many_parts_documents"]["count"]

    print(
        "summary "
        f"chunks={summary.get('total_chunks', 0)} "
        f"documents={summary.get('distinct_documents', 0)} "
        f"avg_chunks_per_document={summary.get('avg_chunks_per_document', 0)} "
        f"null_embeddings={summary.get('null_embedding_chunks', 0)}"
    )
    print(
        "suspects "
        f"duplicate_groups={duplicate_groups} "
        f"cross_document_groups={cross_document_groups} "
        f"short_body_chunks={short_body_count} "
        f"small_tail_chunks={small_tail_count} "
        f"many_parts_documents={many_parts_count}"
    )


def main() -> int:
    args = parse_args()
    settings = get_settings()
    filters = AuditFilters(
        season_year=args.season_year,
        source_table=args.source_table.strip() or None,
    )

    report: Dict[str, Any]
    try:
        with psycopg.connect(settings.database_url) as conn:
            report = build_report(
                conn,
                filters,
                top=args.top,
                short_body_threshold=args.short_body_threshold,
                tail_threshold=args.tail_threshold,
                many_parts_threshold=args.many_parts_threshold,
            )
    except Exception as exc:
        error_report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "filters": {
                "season_year": filters.season_year,
                "source_table": filters.source_table,
            },
            "fatal_error": str(exc),
        }
        print(json.dumps(error_report, ensure_ascii=False))
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(error_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return 1

    print_summary(report)
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "duplicate_content_groups": report["duplicate_content_groups"][
                    "group_count"
                ],
                "cross_document_duplicate_groups": report[
                    "cross_document_duplicate_groups"
                ]["group_count"],
                "short_body_suspects": report["short_body_suspects"]["count"],
                "small_tail_suspects": report["small_tail_suspects"]["count"],
                "many_parts_documents": report["many_parts_documents"]["count"],
            },
            ensure_ascii=False,
            default=str,
        )
    )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
