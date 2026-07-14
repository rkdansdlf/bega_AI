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
