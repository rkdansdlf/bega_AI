"""AI runtime PostgreSQL schema precondition checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


REQUIRED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "coach_analysis_cache": (
        "cache_key",
        "team_id",
        "year",
        "prompt_version",
        "model_name",
        "status",
        "response_json",
        "error_message",
        "error_code",
        "attempt_count",
        "lease_owner",
        "lease_expires_at",
        "last_heartbeat_at",
        "created_at",
        "updated_at",
    ),
    "chat_response_cache": (
        "cache_key",
        "question_text",
        "filters_json",
        "intent",
        "response_text",
        "model_name",
        "hit_count",
        "created_at",
        "expires_at",
    ),
    "chat_semantic_response_cache": (
        "cache_key",
        "question_text",
        "question_embedding",
        "filters_hash",
        "filters_json",
        "intent",
        "source_tier",
        "response_text",
        "model_name",
        "embedding_signature",
        "hit_count",
        "created_at",
        "expires_at",
    ),
    "chat_semantic_cache_shadow_observation": (
        "id",
        "request_cache_key",
        "candidate_cache_key",
        "route",
        "question_text",
        "filters_json",
        "cached_answer",
        "fresh_answer",
        "similarity",
        "observed_at",
        "completed_at",
    ),
    "ai_ingest_runs": (
        "run_id",
        "request_key",
        "trigger_source",
        "status",
        "request_payload",
        "requested_at",
        "started_at",
        "heartbeat_at",
        "finished_at",
        "lease_owner",
        "lease_expires_at",
        "recovery_attempts",
        "error_code",
        "error_message",
        "table_summary",
        "created_at",
        "updated_at",
    ),
    "ai_ingest_watermarks": (
        "source_table",
        "scope_key",
        "last_successful_updated_at",
        "last_run_id",
        "updated_at",
    ),
    "ai_ingest_checkpoints": (
        "run_id",
        "source_table",
        "scope_key",
        "cursor_version",
        "cursor_signature",
        "cursor_payload",
        "committed_batches",
        "source_rows",
        "written_chunks",
        "reused_embeddings",
        "embedded_chunks",
        "max_updated_at",
        "source_updated_before",
        "completed",
        "completed_at",
        "created_at",
        "updated_at",
    ),
    # rag_chunks is owned by the backend/data migration path, but is a hard
    # runtime dependency of managed AI retrieval and must be validated before
    # accepting traffic.
    "rag_chunks": (
        "id",
        "season_year",
        "team_id",
        "source_table",
        "source_row_id",
        "title",
        "content",
        "content_tsv",
        "embedding",
        "meta",
        "metadata",
        "source_type",
        "source_uri",
        "topic_key",
        "content_hash",
        "quality_score",
        "is_active",
        "valid_from",
        "valid_to",
        "expires_at",
        "updated_at",
    ),
}

REQUIRED_INDEXES: tuple[str, ...] = (
    "idx_coach_cache_created_at",
    "idx_coach_cache_team_year",
    "idx_chat_cache_expires_at",
    "idx_chat_cache_created_at",
    "idx_chat_semantic_cache_lookup",
    "idx_chat_semantic_cache_created_at",
    "idx_chat_semantic_cache_expires_at",
    "idx_chat_semantic_shadow_observed_at",
    "idx_chat_semantic_shadow_request_key",
    "ux_ai_ingest_runs_active_request",
    "idx_ai_ingest_runs_status_requested",
    "idx_ai_ingest_checkpoints_updated_at",
)

OPTIONAL_VECTOR_INDEX = "idx_chat_semantic_cache_embedding_hnsw"


class SchemaContractError(RuntimeError):
    """Raised when the AI runtime schema is not ready for managed startup."""


def _missing_names(expected: Iterable[str], actual: Iterable[str]) -> list[str]:
    actual_set = set(actual)
    return [name for name in expected if name not in actual_set]


async def validate_schema_contract(
    conn: Any,
    *,
    require_vector_index: bool = False,
) -> None:
    """Validate tables, columns, and indexes required by the AI runtime.

    This function performs read-only catalog queries. It deliberately does not
    create or alter objects; schema ownership belongs to the migration role.
    """

    table_names = tuple(REQUIRED_COLUMNS)
    table_rows = await (
        await conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = ANY(%s)
            """,
            (list(table_names),),
        )
    ).fetchall()
    actual_tables = {row[0] for row in table_rows}
    missing_tables = _missing_names(table_names, actual_tables)
    if missing_tables:
        raise SchemaContractError(
            "AI DB schema contract missing tables: " + ", ".join(missing_tables)
        )

    column_rows = await (
        await conn.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = ANY(%s)
            """,
            (list(table_names),),
        )
    ).fetchall()
    actual_columns = {(row[0], row[1]) for row in column_rows}
    missing_columns = [
        f"{table_name}.{column_name}"
        for table_name, column_names in REQUIRED_COLUMNS.items()
        for column_name in column_names
        if (table_name, column_name) not in actual_columns
    ]
    if missing_columns:
        raise SchemaContractError(
            "AI DB schema contract missing columns: " + ", ".join(missing_columns)
        )

    index_rows = await (
        await conn.execute(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND indexname = ANY(%s)
            """,
            (list(REQUIRED_INDEXES) + [OPTIONAL_VECTOR_INDEX],),
        )
    ).fetchall()
    actual_indexes = {row[0] for row in index_rows}
    missing_indexes = _missing_names(REQUIRED_INDEXES, actual_indexes)
    if require_vector_index and OPTIONAL_VECTOR_INDEX not in actual_indexes:
        missing_indexes.append(OPTIONAL_VECTOR_INDEX)
    if missing_indexes:
        raise SchemaContractError(
            "AI DB schema contract missing indexes: " + ", ".join(missing_indexes)
        )
