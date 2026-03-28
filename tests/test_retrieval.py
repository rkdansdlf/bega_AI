from __future__ import annotations

from typing import Any

from app.core.retrieval import similarity_search


class _DummyCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, list[Any] | None]] = []

    def execute(self, query: str, params: list[Any] | tuple[Any, ...] | None = None):
        if params is None:
            self.executed.append((str(query), None))
        else:
            self.executed.append((str(query), list(params)))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        if not self.executed:
            return None
        query, _ = self.executed[-1]
        if "information_schema.tables" in query and "rag_chunks" in query:
            return (True,)
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyConnection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.last_cursor: _DummyCursor | None = None

    def cursor(self, row_factory=None):  # noqa: ANN001
        self.last_cursor = _DummyCursor(self.rows)
        return self.last_cursor


def test_similarity_search_hybrid_rrf_uses_union_and_stable_param_order() -> None:
    rows = [
        {
            "id": 1,
            "title": "sample",
            "content": "sample content",
            "source_table": "markdown_docs",
            "source_row_id": "m:1",
            "meta": {},
            "similarity": 0.8,
            "keyword_rank_val": 0.2,
            "combined_score": 0.03,
        }
    ]
    conn = _DummyConnection(rows)
    embedding = [0.1, 0.2, 0.3]
    keyword = "KIA MVP"
    filters = {"season_year": 2025, "meta.league": "정규시즌"}

    result = similarity_search(
        conn,
        embedding,
        limit=5,
        filters=filters,
        keyword=keyword,
    )

    assert result == rows
    assert conn.last_cursor is not None
    assert len(conn.last_cursor.executed) == 3

    timeout_sql, _ = conn.last_cursor.executed[1]
    assert "SET LOCAL statement_timeout = 8000;" in timeout_sql

    sql, params = conn.last_cursor.executed[2]
    expected_vector = "[0.10000000,0.20000000,0.30000000]"
    assert "keyword_query AS" in sql
    assert "candidates AS" in sql
    assert "UNION" in sql
    assert "source_table <> %s" in sql
    assert "ORDER BY combined_score DESC, similarity DESC" in sql
    assert params == [
        keyword,
        expected_vector,
        "game_inning_scores",
        2025,
        "league",
        "정규시즌",
        5,
        "game_inning_scores",
        2025,
        "league",
        "정규시즌",
        5,
        expected_vector,
        5,
    ]


def test_similarity_search_without_keyword_keeps_vector_path() -> None:
    rows = [
        {
            "id": 7,
            "title": "vector-only",
            "content": "vector content",
            "source_table": "player_season_batting",
            "source_row_id": "b:7",
            "meta": {},
            "similarity": 0.9,
        }
    ]
    conn = _DummyConnection(rows)
    embedding = [0.4, 0.5, 0.6]

    result = similarity_search(
        conn,
        embedding,
        limit=3,
        filters={"season_year": 2025},
        keyword=None,
    )

    assert result == rows
    assert conn.last_cursor is not None
    assert len(conn.last_cursor.executed) == 3

    timeout_sql, _ = conn.last_cursor.executed[1]
    assert "SET LOCAL statement_timeout = 8000;" in timeout_sql

    sql, params = conn.last_cursor.executed[2]
    expected_vector = "[0.40000000,0.50000000,0.60000000]"
    assert "keyword_query AS" not in sql
    assert "candidates AS" not in sql
    assert "source_table <> %s" in sql
    assert "ORDER BY embedding <=> %s::vector ASC" in sql
    assert params == [expected_vector, "game_inning_scores", 2025, expected_vector, 3]


def test_similarity_search_internal_opt_in_allows_game_inning_scores() -> None:
    rows = [
        {
            "id": 9,
            "title": "inning-box",
            "content": "inning box content",
            "source_table": "game_inning_scores",
            "source_row_id": "id=9",
            "meta": {},
            "similarity": 0.7,
        }
    ]
    conn = _DummyConnection(rows)
    embedding = [0.7, 0.8, 0.9]

    result = similarity_search(
        conn,
        embedding,
        limit=2,
        filters={"season_year": 2025, "_include_game_inning_scores": True},
        keyword=None,
    )

    assert result == rows
    assert conn.last_cursor is not None

    timeout_sql, _ = conn.last_cursor.executed[1]
    assert "SET LOCAL statement_timeout = 8000;" in timeout_sql

    sql, params = conn.last_cursor.executed[2]
    expected_vector = "[0.70000000,0.80000000,0.90000000]"
    assert "source_table <> %s" not in sql
    assert params == [expected_vector, 2025, expected_vector, 2]


def test_similarity_search_internal_exclude_source_tables_appends_filters() -> None:
    rows = [
        {
            "id": 11,
            "title": "flow-only",
            "content": "flow content",
            "source_table": "game",
            "source_row_id": "game_id=20250501LGHH0",
            "meta": {},
            "similarity": 0.6,
        }
    ]
    conn = _DummyConnection(rows)
    embedding = [0.9, 0.1, 0.2]

    result = similarity_search(
        conn,
        embedding,
        limit=2,
        filters={
            "season_year": 2025,
            "_exclude_source_tables": ["game_flow_summary"],
        },
        keyword=None,
    )

    assert result == rows
    assert conn.last_cursor is not None

    timeout_sql, _ = conn.last_cursor.executed[1]
    assert "SET LOCAL statement_timeout = 8000;" in timeout_sql

    sql, params = conn.last_cursor.executed[2]
    expected_vector = "[0.90000000,0.10000000,0.20000000]"
    assert sql.count("source_table <> %s") == 2
    assert params == [
        expected_vector,
        "game_inning_scores",
        "game_flow_summary",
        2025,
        expected_vector,
        2,
    ]
