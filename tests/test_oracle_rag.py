from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.oracle_rag import (
    OracleRagConnection,
    _bind_filter_clauses,
    _fuse_oracle_results,
    _oracle_connect_args,
    _search_tokens,
    oracle_exact_document_search,
    oracle_similarity_search,
    oracle_rag_readiness,
    upsert_oracle_rag_chunks,
)


class _Cursor:
    def __init__(self, rows: list[tuple[object, ...]], names: list[str]) -> None:
        self.rows = rows
        self.description = [SimpleNamespace(name=name) for name in names]
        self.executed: list[tuple[str, object]] = []

    async def execute(self, sql: str, params=None) -> None:  # noqa: ANN001
        self.executed.append((sql, params))

    async def executemany(self, sql: str, params) -> None:  # noqa: ANN001
        self.executed.append((sql, params))

    async def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows

    async def fetchone(self) -> tuple[object, ...] | None:
        return self.rows[0] if self.rows else None

    async def close(self) -> None:
        return None


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self.cursor_instance = cursor
        self.commits = 0

    async def cursor(self):  # noqa: ANN201
        return self.cursor_instance

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        return None


def test_oracle_url_is_translated_without_logging_credentials() -> None:
    args = _oracle_connect_args(
        "oracle+oracledb://rag_user:p%40ss@adb_high?service_name=ignored"
    )

    assert args["user"] == "rag_user"
    assert args["password"] == "p@ss"
    assert args["dsn"] == "adb_high/ignored"


def test_oracle_filter_builder_parameterizes_source_and_metadata_filters() -> None:
    clauses, params = _bind_filter_clauses(
        {
            "season_year": 2025,
            "source_table_in": ["player_basic", "awards"],
            "_exclude_source_tables": "game_inning_scores",
            "meta.league": "정규시즌",
        }
    )

    assert "season_year = :filter_season_year" in clauses
    assert "source_table IN (:included_source_0, :included_source_1)" in clauses
    assert "source_table <> :excluded_source_0" in clauses
    metadata_clause = next(
        clause for clause in clauses if clause.startswith("JSON_VALUE")
    )
    metadata_bind = metadata_clause.rsplit(":", maxsplit=1)[-1]
    assert metadata_clause.startswith("JSON_VALUE(meta, '$.league') = :")
    assert params["filter_season_year"] == 2025
    assert params[metadata_bind] == "정규시즌"


@pytest.mark.asyncio
async def test_oracle_similarity_search_renders_native_vector_results() -> None:
    cursor = _Cursor(
        [
            (
                7,
                "선수",
                "내용",
                "player_basic",
                "p:7",
                '{"document_type":"profile"}',
                "hash",
                "rag-v1",
                "ACTIVE",
                None,
                None,
                0.125,
            )
        ],
        [
            "ID",
            "TITLE",
            "CONTENT",
            "SOURCE_TABLE",
            "SOURCE_ROW_ID",
            "META",
            "CONTENT_HASH",
            "INDEX_VERSION",
            "INDEX_STATUS",
            "INDEXED_AT",
            "UPDATED_AT",
            "DISTANCE",
        ],
    )
    connection = OracleRagConnection(_Connection(cursor))

    rows = await oracle_similarity_search(
        connection,
        [0.1, 0.2, 0.3],
        limit=3,
        filters={"source_table": "player_basic"},
        keyword=None,
    )

    assert rows[0]["source_table"] == "player_basic"
    assert rows[0]["meta"] == {"document_type": "profile"}
    assert rows[0]["similarity"] == pytest.approx(0.875)
    assert "VECTOR_DISTANCE" in cursor.executed[0][0]


def test_oracle_sparse_tokens_and_rrf_fusion_preserve_shared_identity() -> None:
    assert _search_tokens("선수는 경기에서") == ["선수", "경기"]

    dense = [
        {
            "id": 1,
            "source_table": "player_basic",
            "source_row_id": "1",
            "similarity": 0.9,
        },
        {
            "id": 2,
            "source_table": "player_basic",
            "source_row_id": "2",
            "similarity": 0.8,
        },
    ]
    sparse = [
        {
            "id": 2,
            "source_table": "player_basic",
            "source_row_id": "2",
            "similarity": 0.0,
        },
        {"id": 3, "source_table": "awards", "source_row_id": "3", "similarity": 0.0},
    ]

    rows = _fuse_oracle_results(dense, sparse, limit=3, intent="stats_lookup")

    assert rows[0]["id"] == 2
    assert rows[0]["vector_rank"] == 2
    assert rows[0]["keyword_rank"] == 1
    assert rows[2]["id"] == 3


@pytest.mark.asyncio
async def test_oracle_readiness_uses_native_dimension_function() -> None:
    cursor = _Cursor(
        [(10, 10, 0, 10, 1)],
        [
            "TOTAL_ROWS",
            "VECTOR_ROWS",
            "MISSING_ROWS",
            "MATCHING_DIM_ROWS",
            "VALID_INDEX_ROWS",
        ],
    )
    connection = OracleRagConnection(_Connection(cursor))

    result = await oracle_rag_readiness(connection)

    assert result["ready"] is True
    assert "VECTOR_DIMENSION_COUNT" in cursor.executed[0][0]


@pytest.mark.asyncio
async def test_oracle_upsert_merges_vectors_and_commits() -> None:
    cursor = _Cursor([(1,)], [])
    raw = _Connection(cursor)
    connection = OracleRagConnection(raw)
    records = [
        (
            1,
            "p:1",
            "선수",
            "검색 가능한 내용",
            {"content_hash": "hash", "metadata": {"source_type": "profile"}},
        )
    ]

    count = await upsert_oracle_rag_chunks(
        connection,
        source_table="player_basic",
        records=records,
        embeddings=[[0.1, 0.2, 0.3]],
        season_year=2025,
        player_id="1",
    )

    assert count == 1
    assert raw.commits == 1
    assert "MERGE INTO rag_chunks" in cursor.executed[0][0]
    assert cursor.executed[0][1]["source_table"] == "player_basic"
    assert "INSERT INTO rag_chunk_terms" in cursor.executed[-1][0]


@pytest.mark.asyncio
async def test_oracle_exact_document_search_parses_meta_json() -> None:
    cursor = _Cursor(
        [
            (
                9,
                "KBO 규정",
                "경기 운영 조항",
                "kbo_regulations",
                "reg-9",
                '{"document_type":"rule","regulation_code":"제42조"}',
                1.0,
                1.0,
            )
        ],
        [
            "ID",
            "TITLE",
            "CONTENT",
            "SOURCE_TABLE",
            "SOURCE_ROW_ID",
            "META",
            "SIMILARITY",
            "COMBINED_SCORE",
        ],
    )
    connection = OracleRagConnection(_Connection(cursor))

    rows = await oracle_exact_document_search(
        connection,
        ["규정"],
        limit=5,
        source_tables=("kbo_regulations", "markdown_docs"),
    )

    assert rows[0]["meta"] == {
        "document_type": "rule",
        "regulation_code": "제42조",
    }
    assert rows[0]["metadata"] == rows[0]["meta"]
    assert "DBMS_LOB.INSTR" in cursor.executed[0][0]
