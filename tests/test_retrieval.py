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
    assert len(conn.last_cursor.executed) == 2

    sql, params = conn.last_cursor.executed[1]
    expected_vector = "[0.10000000,0.20000000,0.30000000]"
    assert "keyword_query AS" in sql
    assert "candidates AS" in sql
    assert "UNION" in sql
    assert "ORDER BY combined_score DESC, similarity DESC" in sql
    assert params == [
        keyword,
        expected_vector,
        2025,
        "league",
        "정규시즌",
        5,
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
    assert len(conn.last_cursor.executed) == 2

    sql, params = conn.last_cursor.executed[1]
    expected_vector = "[0.40000000,0.50000000,0.60000000]"
    assert "keyword_query AS" not in sql
    assert "candidates AS" not in sql
    assert "ORDER BY embedding <=> %s::vector ASC" in sql
    assert params == [expected_vector, 2025, expected_vector, 3]
