from __future__ import annotations

import asyncio

import pytest

from app.db.schema_contract import (
    REQUIRED_COLUMNS,
    REQUIRED_INDEXES,
    SchemaContractError,
    validate_schema_contract,
)


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self):
        return self._rows


class _Connection:
    def __init__(self, *, tables, columns, indexes):
        self.tables = tables
        self.columns = columns
        self.indexes = indexes

    async def execute(self, query, params):
        if "information_schema.tables" in query:
            return _Cursor(self.tables)
        if "information_schema.columns" in query:
            return _Cursor(self.columns)
        if "pg_indexes" in query:
            return _Cursor(self.indexes)
        raise AssertionError(f"unexpected schema query: {query}")


def _complete_connection() -> _Connection:
    columns = [
        (table_name, column_name)
        for table_name, column_names in REQUIRED_COLUMNS.items()
        for column_name in column_names
    ]
    return _Connection(
        tables=[(table_name,) for table_name in REQUIRED_COLUMNS],
        columns=columns,
        indexes=[(index_name,) for index_name in REQUIRED_INDEXES],
    )


def test_validate_schema_contract_accepts_complete_contract():
    asyncio.run(validate_schema_contract(_complete_connection()))


def test_complete_contract_includes_rag_chunks_runtime_columns():
    assert "rag_chunks" in REQUIRED_COLUMNS
    assert {
        "embedding",
        "content_tsv",
        "source_table",
        "metadata",
        "is_active",
    }.issubset(REQUIRED_COLUMNS["rag_chunks"])


def test_complete_contract_includes_ingest_orchestration_tables():
    assert set(REQUIRED_COLUMNS["ai_ingest_runs"]) == {
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
    }
    assert set(REQUIRED_COLUMNS["ai_ingest_watermarks"]) == {
        "source_table",
        "scope_key",
        "last_successful_updated_at",
        "last_run_id",
        "updated_at",
    }
    assert {
        "ux_ai_ingest_runs_active_request",
        "idx_ai_ingest_runs_status_requested",
    }.issubset(REQUIRED_INDEXES)


def test_complete_contract_includes_ingest_checkpoints():
    assert set(REQUIRED_COLUMNS["ai_ingest_checkpoints"]) == {
        "run_id", "source_table", "scope_key", "cursor_version",
        "cursor_signature", "cursor_payload", "committed_batches",
        "source_rows", "written_chunks", "reused_embeddings",
        "embedded_chunks", "max_updated_at", "completed", "completed_at",
        "created_at", "updated_at",
    }
    assert "idx_ai_ingest_checkpoints_updated_at" in REQUIRED_INDEXES


def test_validate_schema_contract_reports_missing_columns_and_indexes():
    connection = _complete_connection()
    connection.columns = [
        row for row in connection.columns if row != ("chat_response_cache", "expires_at")
    ]
    connection.indexes = [
        row for row in connection.indexes if row != ("idx_chat_cache_expires_at",)
    ]

    with pytest.raises(SchemaContractError, match="chat_response_cache.expires_at"):
        asyncio.run(validate_schema_contract(connection))


def test_validate_schema_contract_requires_optional_vector_index_only_when_enabled():
    connection = _complete_connection()

    with pytest.raises(SchemaContractError, match="idx_chat_semantic_cache_embedding_hnsw"):
        asyncio.run(
            validate_schema_contract(
                connection,
                require_vector_index=True,
            )
        )
