#!/usr/bin/env python3
"""Backfill rag_chunks metadata/hash fields without re-embedding."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

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
from app.core.rag_storage import build_chunk_storage_fields, json_dumps

PGVECTOR_SEARCH_PATH = "public, extensions, security"
GENERIC_QUALITY_SCORE = 0.50
QUALITY_RECOMPUTE_SOURCE_TYPES = {
    "canonical_knowledge",
    "kbo_db_table",
    "official_rulebook",
    "markdown_doc",
    "document",
    "manual_lineup",
    "chat_memory",
}


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required to run backfill_rag_chunk_metadata.py. "
            f"Detail: {_PSYCOPG_IMPORT_ERROR}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill rag_chunks metadata, content_hash, chunk_hash, and embedding model fields."
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source-table", nargs="*", default=[])
    parser.add_argument(
        "--season-year",
        nargs="*",
        type=int,
        default=[],
        help="Limit backfill to specific rag_chunks.season_year values.",
    )
    parser.add_argument(
        "--update-mode",
        choices=("bulk", "row"),
        default="bulk",
        help="Use COPY + UPDATE FROM bulk updates, or row-wise executemany updates.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect planned updates without writing. This is the default mode.",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Persist backfilled metadata/hash fields. Without this flag, no writes occur.",
    )
    return parser


def _build_select(
    *,
    source_tables: Sequence[str],
    season_years: Sequence[int],
    limit: Optional[int],
) -> tuple[str, tuple[Any, ...]]:
    params: List[Any] = []
    query = """
        SELECT
            id,
            source_table,
            source_row_id,
            content,
            meta,
            metadata,
            source_type,
            source_uri,
            topic_key,
            embedding_model,
            embedding_dim,
            embedding_version,
            chunking_version,
            quality_score
        FROM rag_chunks
    """
    where_parts = [
        """(
            metadata IS NULL OR metadata = '{}'::jsonb
            OR source_type IS NULL OR btrim(source_type) = ''
            OR source_uri IS NULL OR btrim(source_uri) = ''
            OR topic_key IS NULL OR btrim(topic_key) = ''
            OR content_hash IS NULL OR btrim(content_hash) = ''
            OR chunk_hash IS NULL OR btrim(chunk_hash) = ''
            OR embedding_model IS NULL OR btrim(embedding_model) = ''
            OR embedding_dim IS NULL
            OR embedding_version IS NULL
            OR chunking_version IS NULL
            OR quality_score IS NULL
            OR (
                quality_score <= 0.50
                AND source_type = ANY(%s)
            )
        )"""
    ]
    params.append(sorted(QUALITY_RECOMPUTE_SOURCE_TYPES))
    if source_tables:
        where_parts.append("source_table = ANY(%s)")
        params.append(list(source_tables))
    if season_years:
        where_parts.append("season_year = ANY(%s)")
        params.append(list(season_years))
    query += " WHERE " + " AND ".join(where_parts)
    query += " ORDER BY id"
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)
    return query, tuple(params)


def _batched_fetch(cur: Any, batch_size: int) -> List[Dict[str, Any]]:
    rows = cur.fetchmany(batch_size)
    return [dict(row) for row in rows]


def _quality_score_for_backfill(row: Dict[str, Any]) -> Any:
    score = row.get("quality_score")
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return None
    source_type = row.get("source_type")
    if numeric_score <= GENERIC_QUALITY_SCORE and (
        not source_type or source_type in QUALITY_RECOMPUTE_SOURCE_TYPES
    ):
        return None
    return score


def _write_payload_bulk(conn: Any, payload: Sequence[tuple[Any, ...]]) -> None:
    with conn.cursor() as write_cur:
        write_cur.execute(f"SET search_path TO {PGVECTOR_SEARCH_PATH}")
        write_cur.execute(
            """
            CREATE TEMP TABLE rag_backfill_payload (
                metadata jsonb,
                source_type text,
                source_uri text,
                topic_key text,
                content_hash text,
                chunk_hash text,
                embedding_model text,
                embedding_dim integer,
                embedding_version integer,
                chunking_version integer,
                quality_score numeric,
                id bigint
            ) ON COMMIT DROP
            """
        )
        with write_cur.copy(
            """
            COPY rag_backfill_payload (
                metadata,
                source_type,
                source_uri,
                topic_key,
                content_hash,
                chunk_hash,
                embedding_model,
                embedding_dim,
                embedding_version,
                chunking_version,
                quality_score,
                id
            ) FROM STDIN
            """
        ) as copy:
            for row in payload:
                copy.write_row(row)
        write_cur.execute(
            """
            UPDATE rag_chunks AS target
            SET metadata = payload.metadata,
                source_type = payload.source_type,
                source_uri = payload.source_uri,
                topic_key = payload.topic_key,
                content_hash = payload.content_hash,
                chunk_hash = payload.chunk_hash,
                embedding_model = payload.embedding_model,
                embedding_dim = payload.embedding_dim,
                embedding_version = payload.embedding_version,
                chunking_version = payload.chunking_version,
                quality_score = payload.quality_score,
                is_active = COALESCE(target.is_active, true),
                updated_at = now()
            FROM rag_backfill_payload AS payload
            WHERE target.id = payload.id
            """
        )


def _write_payload_rowwise(conn: Any, payload: Sequence[tuple[Any, ...]]) -> None:
    with conn.cursor() as write_cur:
        write_cur.execute(f"SET search_path TO {PGVECTOR_SEARCH_PATH}")
        write_cur.executemany(
            """
            UPDATE rag_chunks
            SET metadata = %s::jsonb,
                source_type = %s,
                source_uri = %s,
                topic_key = %s,
                content_hash = %s,
                chunk_hash = %s,
                embedding_model = %s,
                embedding_dim = %s,
                embedding_version = %s,
                chunking_version = %s,
                quality_score = %s,
                is_active = COALESCE(is_active, true),
                updated_at = now()
            WHERE id = %s
            """,
            list(payload),
        )


def run(
    *,
    batch_size: int,
    limit: Optional[int],
    source_tables: Sequence[str],
    season_years: Sequence[int],
    update_mode: str,
    dry_run: bool,
) -> Dict[str, Any]:
    _require_psycopg()
    settings = get_settings()
    query, params = _build_select(
        source_tables=source_tables,
        season_years=season_years,
        limit=limit,
    )
    updated = 0
    skipped = 0

    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor(row_factory=dict_row) as read_cur:
            read_cur.execute(f"SET search_path TO {PGVECTOR_SEARCH_PATH}")
            read_cur.execute(query, params)
            while True:
                rows = _batched_fetch(read_cur, max(1, batch_size))
                if not rows:
                    break
                payload: List[tuple[Any, ...]] = []
                for row in rows:
                    meta = row.get("metadata") or row.get("meta") or {}
                    try:
                        storage = build_chunk_storage_fields(
                            settings=settings,
                            source_table=str(row["source_table"]),
                            source_row_id=str(row["source_row_id"]),
                            content=str(row["content"] or ""),
                            meta=meta,
                            source_type=row.get("source_type"),
                            source_uri=row.get("source_uri"),
                            topic_key=row.get("topic_key"),
                            embedding_model=row.get("embedding_model"),
                            embedding_dim=row.get("embedding_dim"),
                            embedding_version=row.get("embedding_version"),
                            chunking_version=row.get("chunking_version"),
                            quality_score=_quality_score_for_backfill(row),
                        )
                    except ValueError:
                        skipped += 1
                        continue
                    payload.append(
                        (
                            json_dumps(storage["metadata"]),
                            storage["source_type"],
                            storage["source_uri"],
                            storage["topic_key"],
                            storage["content_hash"],
                            storage["chunk_hash"],
                            storage["embedding_model"],
                            storage["embedding_dim"],
                            storage["embedding_version"],
                            storage["chunking_version"],
                            storage["quality_score"],
                            row["id"],
                        )
                    )
                if dry_run:
                    updated += len(payload)
                    continue
                if update_mode == "row":
                    _write_payload_rowwise(conn, payload)
                else:
                    _write_payload_bulk(conn, payload)
                updated += len(payload)
                conn.commit()

    return {"updated": updated, "skipped": skipped, "dry_run": dry_run}


def main() -> int:
    args = build_parser().parse_args()
    result = run(
        batch_size=args.batch_size,
        limit=args.limit,
        source_tables=args.source_table,
        season_years=args.season_year,
        update_mode=args.update_mode,
        dry_run=not args.apply,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
