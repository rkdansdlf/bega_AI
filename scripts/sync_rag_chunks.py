#!/usr/bin/env python3
"""Synchronize rag_chunks rows between two PostgreSQL databases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

from app.config import Settings

_BASE_COLUMNS = (
    "meta",
    "metadata",
    "source_type",
    "source_uri",
    "topic_key",
    "content_hash",
    "chunk_hash",
    "embedding_model",
    "embedding_dim",
    "embedding_version",
    "chunking_version",
    "quality_score",
    "is_active",
    "valid_from",
    "valid_to",
    "expires_at",
    "season_year",
    "season_id",
    "league_type_code",
    "team_id",
    "player_id",
    "source_table",
    "source_row_id",
    "title",
    "content",
)
# Lifecycle column written by KBO_playwright tombstone tools (DELETED etc.).
# Appended last so the positional payload of the base columns is unchanged.
INDEX_STATUS_COLUMN = "index_status"
INDEX_STATUS_COLUMN_EXISTS_SQL = """
SELECT EXISTS(
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = ANY(current_schemas(false))
      AND table_name = 'rag_chunks'
      AND column_name = 'index_status'
)
"""
PGVECTOR_SEARCH_PATH = "public, extensions, security"
STAGE_TABLE_NAME = "rag_chunk_sync_stage"


def build_select_columns(*, include_index_status: bool = False) -> str:
    columns = list(_BASE_COLUMNS)
    columns.append("embedding::text AS embedding_text")
    if include_index_status:
        columns.append(INDEX_STATUS_COLUMN)
    body = ",\n".join(f"    {column}" for column in columns)
    return f"\nSELECT\n{body}\nFROM rag_chunks\n"


def stage_columns(*, include_index_status: bool = False) -> Tuple[str, ...]:
    columns = (
        "meta_json",
        "metadata_json",
        *_BASE_COLUMNS[2:],
        "embedding_text",
    )
    if include_index_status:
        columns += (INDEX_STATUS_COLUMN,)
    return columns


def _insert_column_list(include_index_status: bool) -> str:
    columns = [*_BASE_COLUMNS, "embedding"]
    if include_index_status:
        columns.append(INDEX_STATUS_COLUMN)
    return ",\n".join(f"    {column}" for column in columns)


def _conflict_update_sql(include_index_status: bool) -> str:
    # is_active is copied as-is (an explicit false stays false; only NULL
    # becomes true). index_status is copied verbatim so a source tombstone
    # (DELETED) is never reset to ACTIVE in the destination.
    index_status_update = (
        "\n    index_status = EXCLUDED.index_status," if include_index_status else ""
    )
    return f"""ON CONFLICT (source_table, source_row_id)
DO UPDATE SET
    meta = EXCLUDED.meta,
    metadata = EXCLUDED.metadata,
    source_type = EXCLUDED.source_type,
    source_uri = EXCLUDED.source_uri,
    topic_key = EXCLUDED.topic_key,
    content_hash = EXCLUDED.content_hash,
    chunk_hash = EXCLUDED.chunk_hash,
    embedding_model = EXCLUDED.embedding_model,
    embedding_dim = EXCLUDED.embedding_dim,
    embedding_version = EXCLUDED.embedding_version,
    chunking_version = EXCLUDED.chunking_version,
    quality_score = EXCLUDED.quality_score,
    is_active = EXCLUDED.is_active,{index_status_update}
    valid_from = EXCLUDED.valid_from,
    valid_to = EXCLUDED.valid_to,
    expires_at = EXCLUDED.expires_at,
    content = EXCLUDED.content,
    embedding = COALESCE(EXCLUDED.embedding, rag_chunks.embedding),
    season_year = COALESCE(EXCLUDED.season_year, rag_chunks.season_year),
    season_id = COALESCE(EXCLUDED.season_id, rag_chunks.season_id),
    league_type_code = COALESCE(EXCLUDED.league_type_code, rag_chunks.league_type_code),
    team_id = COALESCE(EXCLUDED.team_id, rag_chunks.team_id),
    player_id = COALESCE(EXCLUDED.player_id, rag_chunks.player_id),
    title = EXCLUDED.title,
    updated_at = now()
"""


def build_merge_stage_sql(*, include_index_status: bool = False) -> str:
    # NULL index_status means ACTIVE (same rule as retrieval); writing the
    # literal keeps NOT NULL destination columns (KBO_playwright schema) happy.
    index_status_select = (
        ",\n    COALESCE(index_status, 'ACTIVE')" if include_index_status else ""
    )
    return f"""
INSERT INTO rag_chunks (
{_insert_column_list(include_index_status)}
)
SELECT
    CASE WHEN meta_json IS NULL THEN NULL ELSE meta_json::jsonb END,
    CASE WHEN metadata_json IS NULL THEN '{{}}'::jsonb ELSE metadata_json::jsonb END,
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
    COALESCE(is_active, true),
    valid_from,
    valid_to,
    expires_at,
    season_year,
    season_id,
    league_type_code,
    team_id,
    player_id,
    source_table,
    source_row_id,
    title,
    content,
    embedding_text::vector{index_status_select}
FROM {STAGE_TABLE_NAME}
{_conflict_update_sql(include_index_status)}"""


def build_upsert_sql(*, include_index_status: bool = False) -> str:
    index_status_value = ", COALESCE(%s, 'ACTIVE')" if include_index_status else ""
    return f"""
INSERT INTO rag_chunks (
{_insert_column_list(include_index_status)}
) VALUES (
    %s::jsonb, COALESCE(%s::jsonb, '{{}}'::jsonb), %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, COALESCE(%s, true), %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s::vector{index_status_value}
)
{_conflict_update_sql(include_index_status)}"""


SELECT_COLUMNS = build_select_columns()
STAGE_COLUMNS = stage_columns()
MERGE_STAGE_SQL = build_merge_stage_sql()
UPSERT_SQL = build_upsert_sql()


def _has_index_status_column(cur: Any) -> bool:
    cur.execute(INDEX_STATUS_COLUMN_EXISTS_SQL, ())
    row = cur.fetchone()
    if row is None:
        return False
    if isinstance(row, dict):
        return bool(next(iter(row.values()), False))
    return bool(row[0])


def resolve_index_status_sync(
    *, source_has_column: bool, dest_has_column: bool
) -> bool:
    """Decide whether index_status is copied; refuse to silently drop tombstones."""
    if source_has_column and not dest_has_column:
        raise RuntimeError(
            "Source rag_chunks has index_status but the destination does not; "
            "syncing would drop tombstones (DELETED chunks would become searchable). "
            "Apply app/db/migrations/010_rag_chunks_index_status.sql to the "
            "destination first."
        )
    if not source_has_column:
        print(
            "source rag_chunks has no index_status column; destination index_status "
            "is left unchanged",
            file=sys.stderr,
            flush=True,
        )
    return source_has_column


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required to run sync_rag_chunks.py. "
            "Install dependencies (e.g. pip install psycopg[binary]) and retry. "
            f"Detail: {_PSYCOPG_IMPORT_ERROR}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch upsert rag_chunks from a source DB into a destination DB."
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
        "--source-table",
        nargs="*",
        default=[],
        help="Optional source_table filters.",
    )
    parser.add_argument(
        "--season-year",
        type=int,
        default=None,
        help="Optional season_year filter for seasonal rag_chunks syncs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Rows fetched from source per batch.",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=10000,
        help="Commit destination every N upserted rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke runs.",
    )
    parser.add_argument(
        "--known-total-rows",
        type=int,
        default=None,
        help=(
            "Optional known source row count. If provided, the sync can continue "
            "even when the upfront source count query fails."
        ),
    )
    parser.add_argument(
        "--allow-same-db",
        action="store_true",
        help="Allow source and destination DB URLs to match.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect source row counts without writing to destination.",
    )
    parser.add_argument(
        "--truncate-dest",
        action="store_true",
        help="Truncate destination rag_chunks before syncing.",
    )
    return parser


def _load_settings_from_env_file(env_file: str) -> Settings:
    env_path = Path(env_file).expanduser()
    if not env_path.is_absolute():
        env_path = (PROJECT_ROOT / env_path).resolve()
    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")

    values: Dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized = value.strip()
        if (
            len(normalized) >= 2
            and normalized[0] == normalized[-1]
            and normalized[0] in {"'", '"'}
        ):
            normalized = normalized[1:-1]
        values[key.strip()] = normalized

    return Settings(_env_file=None, **values)


def resolve_db_urls(args: argparse.Namespace) -> Tuple[str, str]:
    source_db_url = args.source_db_url.strip()
    dest_db_url = args.dest_db_url.strip()

    if not source_db_url:
        source_settings = _load_settings_from_env_file(args.source_env_file)
        source_db_url = source_settings.database_url

    if not dest_db_url:
        dest_env_file = args.dest_env_file.strip() or args.source_env_file
        dest_settings = _load_settings_from_env_file(dest_env_file)
        dest_db_url = dest_settings.database_url

    if not args.allow_same_db and source_db_url == dest_db_url:
        raise RuntimeError(
            "Source and destination DB URLs are identical. "
            "Pass --allow-same-db only if that is intentional."
        )

    return source_db_url, dest_db_url


def build_select_query(
    *,
    source_tables: Sequence[str],
    season_year: Optional[int],
    limit: Optional[int],
    include_index_status: bool = False,
) -> tuple[str, tuple[Any, ...]]:
    query = build_select_columns(include_index_status=include_index_status).strip()
    params: List[Any] = []

    if source_tables:
        query += "\nWHERE source_table = ANY(%s)"
        params.append(list(source_tables))
    if season_year is not None:
        query += "\nAND season_year = %s" if params else "\nWHERE season_year = %s"
        params.append(season_year)

    query += "\nORDER BY id"

    if limit is not None:
        query += "\nLIMIT %s"
        params.append(limit)

    return query, tuple(params)


def _count_source_rows(
    source_cur: Any,
    *,
    source_tables: Sequence[str],
    season_year: Optional[int],
    limit: Optional[int],
) -> int:
    query = "SELECT count(*) FROM rag_chunks"
    params: List[Any] = []
    if source_tables:
        query += " WHERE source_table = ANY(%s)"
        params.append(list(source_tables))
    if season_year is not None:
        query += " AND season_year = %s" if params else " WHERE season_year = %s"
        params.append(season_year)
    if limit is not None:
        return min(limit, _execute_scalar(source_cur, query, tuple(params)))
    return _execute_scalar(source_cur, query, tuple(params))


def _execute_scalar(cur: Any, query: str, params: tuple[Any, ...]) -> int:
    cur.execute(query, params)
    row = cur.fetchone()
    if row is None:
        return 0
    if isinstance(row, dict):
        return int(next(iter(row.values())))
    return int(row[0])


def _build_upsert_rows(
    rows: Sequence[Dict[str, Any]], *, include_index_status: bool = False
) -> List[tuple[Any, ...]]:
    payload: List[tuple[Any, ...]] = []
    for row in rows:
        meta = row.get("meta")
        metadata = row.get("metadata")
        values: tuple[Any, ...] = (
            (
                json.dumps(meta, ensure_ascii=False, default=str)
                if meta is not None
                else None
            ),
            (
                json.dumps(metadata, ensure_ascii=False, default=str)
                if metadata is not None
                else None
            ),
            row.get("source_type"),
            row.get("source_uri"),
            row.get("topic_key"),
            row.get("content_hash"),
            row.get("chunk_hash"),
            row.get("embedding_model"),
            row.get("embedding_dim"),
            row.get("embedding_version"),
            row.get("chunking_version"),
            row.get("quality_score"),
            row.get("is_active"),
            row.get("valid_from"),
            row.get("valid_to"),
            row.get("expires_at"),
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
        if include_index_status:
            values += (row.get(INDEX_STATUS_COLUMN),)
        payload.append(values)
    return payload


def _batched_fetch(
    source_cur: Any,
    *,
    batch_size: int,
) -> Iterable[List[Dict[str, Any]]]:
    while True:
        rows = source_cur.fetchmany(batch_size)
        if not rows:
            break
        yield [dict(row) for row in rows]


def _prepare_stage_table(dest_cur: Any) -> None:
    dest_cur.execute(f"""
        CREATE TEMP TABLE IF NOT EXISTS {STAGE_TABLE_NAME} (
            meta_json text,
            metadata_json text,
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
            is_active boolean,
            valid_from timestamptz,
            valid_to timestamptz,
            expires_at timestamptz,
            season_year integer,
            season_id integer,
            league_type_code integer,
            team_id text,
            player_id text,
            source_table text NOT NULL,
            source_row_id text NOT NULL,
            title text,
            content text,
            embedding_text text,
            index_status text
        ) ON COMMIT PRESERVE ROWS
        """)
    dest_cur.execute(f"TRUNCATE TABLE {STAGE_TABLE_NAME}")


def _stage_batch_with_copy(
    dest_cur: Any,
    payload: Sequence[tuple[Any, ...]],
    *,
    include_index_status: bool = False,
) -> None:
    column_list = ", ".join(stage_columns(include_index_status=include_index_status))
    with dest_cur.copy(f"COPY {STAGE_TABLE_NAME} ({column_list}) FROM STDIN") as copy:
        for row in payload:
            copy.write_row(row)


def _merge_staged_rows(dest_cur: Any, *, include_index_status: bool = False) -> int:
    staged_count = _execute_scalar(
        dest_cur,
        f"SELECT count(*) FROM {STAGE_TABLE_NAME}",
        (),
    )
    if staged_count <= 0:
        return 0
    dest_cur.execute(build_merge_stage_sql(include_index_status=include_index_status))
    dest_cur.execute(f"TRUNCATE TABLE {STAGE_TABLE_NAME}")
    return staged_count


def sync_rag_chunks(
    *,
    source_db_url: str,
    dest_db_url: str,
    source_tables: Sequence[str],
    season_year: Optional[int],
    batch_size: int,
    commit_interval: int,
    limit: Optional[int],
    known_total_rows: Optional[int],
    dry_run: bool,
    truncate_dest: bool,
) -> Dict[str, Any]:
    _require_psycopg()
    synced_rows = 0
    committed_rows = 0
    pending_stage_rows = 0

    with (
        psycopg.connect(source_db_url) as source_conn,
        psycopg.connect(dest_db_url) as dest_conn,
        source_conn.cursor() as count_cur,
    ):
        with dest_conn.cursor() as init_dest_cur:
            init_dest_cur.execute(f"SET search_path TO {PGVECTOR_SEARCH_PATH}")
            init_dest_cur.execute("SET statement_timeout TO 0;")
            dest_has_index_status = _has_index_status_column(init_dest_cur)
            with source_conn.cursor() as init_source_cur:
                init_source_cur.execute("SET statement_timeout TO 0;")
                source_has_index_status = _has_index_status_column(init_source_cur)
            # Checked before any destination write (including --truncate-dest).
            include_index_status = resolve_index_status_sync(
                source_has_column=source_has_index_status,
                dest_has_column=dest_has_index_status,
            )
            if truncate_dest and not dry_run:
                init_dest_cur.execute("TRUNCATE TABLE rag_chunks")
                dest_conn.commit()

        try:
            total_rows = _count_source_rows(
                count_cur,
                source_tables=source_tables,
                season_year=season_year,
                limit=limit,
            )
        except psycopg.OperationalError:
            if known_total_rows is None:
                raise
            print(
                "source row count query failed; falling back to --known-total-rows="
                f"{known_total_rows}",
                file=sys.stderr,
                flush=True,
            )
            total_rows = known_total_rows

        if dry_run:
            return {
                "total_rows": total_rows,
                "synced_rows": 0,
                "committed_rows": 0,
                "dry_run": True,
            }

        query, params = build_select_query(
            source_tables=source_tables,
            season_year=season_year,
            limit=limit,
            include_index_status=include_index_status,
        )
        upsert_sql = build_upsert_sql(include_index_status=include_index_status)

        with (
            source_conn.cursor(
                name="rag_chunk_sync_cursor", row_factory=dict_row
            ) as source_cur,
            dest_conn.cursor() as dest_cur,
        ):
            source_cur.execute(query, params)
            use_copy_pipeline = hasattr(dest_cur, "copy")
            if use_copy_pipeline:
                _prepare_stage_table(dest_cur)

            for batch_index, rows in enumerate(
                _batched_fetch(source_cur, batch_size=max(1, batch_size)),
                start=1,
            ):
                payload = _build_upsert_rows(
                    rows, include_index_status=include_index_status
                )
                if payload and use_copy_pipeline:
                    _stage_batch_with_copy(
                        dest_cur, payload, include_index_status=include_index_status
                    )
                    pending_stage_rows += len(payload)
                elif payload:
                    dest_cur.executemany(upsert_sql, payload)
                    synced_rows += len(payload)

                print(
                    "synced_batches="
                    f"{batch_index} synced_rows={synced_rows + pending_stage_rows}/{total_rows}",
                    flush=True,
                )

                if (
                    use_copy_pipeline
                    and commit_interval > 0
                    and pending_stage_rows >= commit_interval
                ):
                    synced_rows += _merge_staged_rows(
                        dest_cur, include_index_status=include_index_status
                    )
                    pending_stage_rows = 0
                    dest_conn.commit()
                    committed_rows = synced_rows
                elif (
                    not use_copy_pipeline
                    and commit_interval > 0
                    and synced_rows - committed_rows >= commit_interval
                ):
                    dest_conn.commit()
                    committed_rows = synced_rows

            if use_copy_pipeline and pending_stage_rows > 0:
                synced_rows += _merge_staged_rows(
                    dest_cur, include_index_status=include_index_status
                )
                pending_stage_rows = 0
            dest_conn.commit()
            committed_rows = synced_rows

    return {
        "total_rows": total_rows,
        "synced_rows": synced_rows,
        "committed_rows": committed_rows,
        "dry_run": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    source_db_url, dest_db_url = resolve_db_urls(args)
    result = sync_rag_chunks(
        source_db_url=source_db_url,
        dest_db_url=dest_db_url,
        source_tables=args.source_table,
        season_year=args.season_year,
        batch_size=max(1, args.batch_size),
        commit_interval=max(1, args.commit_interval),
        limit=args.limit,
        known_total_rows=args.known_total_rows,
        dry_run=args.dry_run,
        truncate_dest=args.truncate_dest,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
