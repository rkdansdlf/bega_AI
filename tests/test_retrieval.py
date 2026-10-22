from __future__ import annotations

from typing import Any

import pytest

from app.core.retrieval import (
    _RRF_K_BY_INTENT,
    _resolve_rrf_k,
    record_retrieval_event,
    similarity_search,
)


class _DummyCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, list[Any] | None]] = []

    async def execute(
        self, query: str, params: list[Any] | tuple[Any, ...] | None = None
    ):
        if params is None:
            self.executed.append((str(query), None))
        else:
            self.executed.append((str(query), list(params)))

    async def fetchall(self):
        return self.rows

    async def fetchone(self):
        if not self.executed:
            return None
        query, _ = self.executed[-1]
        if "information_schema.tables" in query and "rag_chunks" in query:
            return (True,)
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _assert_rows_preserved(result, rows) -> None:
    """Source columns are untouched; the shared retrieval contract keys are added."""
    assert len(result) == len(rows)
    for got, want in zip(result, rows):
        assert {key: got[key] for key in want} == want
        assert got["retrieval_backend"] == "postgres"
        for key in ("season_year", "team_id", "player_id", "index_generation"):
            assert key in got


class _DummyConnection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.last_cursor: _DummyCursor | None = None

    def cursor(self, row_factory=None):  # noqa: ANN001
        self.last_cursor = _DummyCursor(self.rows)
        return self.last_cursor


class _RaisingEventCursor(_DummyCursor):
    async def execute(
        self, query: str, params: list[Any] | tuple[Any, ...] | None = None
    ):
        await super().execute(query, params)
        if "INSERT INTO rag_retrieval_events" in query:
            raise RuntimeError("event table unavailable")


class _RaisingEventConnection(_DummyConnection):
    def cursor(self, row_factory=None):  # noqa: ANN001
        self.last_cursor = _RaisingEventCursor(self.rows)
        return self.last_cursor


@pytest.mark.asyncio
async def test_similarity_search_hybrid_rrf_uses_union_and_stable_param_order() -> None:
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

    result = await similarity_search(
        conn,
        embedding,
        limit=5,
        filters=filters,
        keyword=keyword,
    )

    _assert_rows_preserved(result, rows)
    assert conn.last_cursor is not None
    # _ensure_pgvector_session uses its own cursor (SET search_path + SET ivfflat.probes).
    # The main-query cursor only sees: SET LOCAL statement_timeout + the SELECT.
    assert len(conn.last_cursor.executed) == 2

    timeout_sql, _ = conn.last_cursor.executed[0]
    assert "SET LOCAL statement_timeout = 8000;" in timeout_sql

    sql, params = conn.last_cursor.executed[1]
    expected_vector = "[0.10000000,0.20000000,0.30000000]"
    assert "keyword_query AS" in sql
    assert "candidates AS" in sql
    assert "UNION" in sql
    assert "source_table <> %s" in sql
    assert "COALESCE(is_active, true) = true" in sql
    assert "(expires_at IS NULL OR expires_at > now())" in sql
    assert "(valid_from IS NULL OR valid_from <= now())" in sql
    assert "(valid_to IS NULL OR valid_to > now())" in sql
    assert "metadata->>%s = %s" in sql
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


@pytest.mark.asyncio
async def test_similarity_search_without_keyword_keeps_vector_path() -> None:
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

    result = await similarity_search(
        conn,
        embedding,
        limit=3,
        filters={"season_year": 2025},
        keyword=None,
    )

    _assert_rows_preserved(result, rows)
    assert conn.last_cursor is not None
    assert len(conn.last_cursor.executed) == 2

    timeout_sql, _ = conn.last_cursor.executed[0]
    assert "SET LOCAL statement_timeout = 8000;" in timeout_sql

    sql, params = conn.last_cursor.executed[1]
    expected_vector = "[0.40000000,0.50000000,0.60000000]"
    assert "keyword_query AS" not in sql
    assert "candidates AS" not in sql
    assert "source_table <> %s" in sql
    assert "COALESCE(is_active, true) = true" in sql
    assert "(expires_at IS NULL OR expires_at > now())" in sql
    assert "(valid_from IS NULL OR valid_from <= now())" in sql
    assert "(valid_to IS NULL OR valid_to > now())" in sql
    assert "ORDER BY embedding::halfvec(1536) <=> %s::halfvec(1536) ASC" in sql
    assert params == [expected_vector, "game_inning_scores", 2025, expected_vector, 3]


@pytest.mark.asyncio
async def test_similarity_search_internal_opt_in_allows_game_inning_scores() -> None:
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

    result = await similarity_search(
        conn,
        embedding,
        limit=2,
        filters={"season_year": 2025, "_include_game_inning_scores": True},
        keyword=None,
    )

    _assert_rows_preserved(result, rows)
    assert conn.last_cursor is not None

    timeout_sql, _ = conn.last_cursor.executed[0]
    assert "SET LOCAL statement_timeout = 8000;" in timeout_sql

    sql, params = conn.last_cursor.executed[1]
    expected_vector = "[0.70000000,0.80000000,0.90000000]"
    assert "source_table <> %s" not in sql
    assert params == [expected_vector, 2025, expected_vector, 2]


@pytest.mark.asyncio
async def test_similarity_search_internal_exclude_source_tables_appends_filters() -> (
    None
):
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

    result = await similarity_search(
        conn,
        embedding,
        limit=2,
        filters={
            "season_year": 2025,
            "_exclude_source_tables": ["game_flow_summary"],
        },
        keyword=None,
    )

    _assert_rows_preserved(result, rows)
    assert conn.last_cursor is not None

    timeout_sql, _ = conn.last_cursor.executed[0]
    assert "SET LOCAL statement_timeout = 8000;" in timeout_sql

    sql, params = conn.last_cursor.executed[1]
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


@pytest.mark.asyncio
async def test_similarity_search_internal_source_table_filter_is_parameterized() -> (
    None
):
    rows = [
        {
            "id": 12,
            "source_table": "kbo_regulations",
            "source_row_id": "r:12",
            "similarity": 0.8,
        }
    ]
    conn = _DummyConnection(rows)

    result = await similarity_search(
        conn,
        [0.2, 0.3, 0.4],
        limit=2,
        filters={"source_table_in": ["kbo_regulations", "markdown_docs"]},
    )

    _assert_rows_preserved(result, rows)
    assert conn.last_cursor is not None
    sql, params = conn.last_cursor.executed[1]
    assert "source_table_in = %s" not in sql
    assert "source_table = ANY(%s)" in sql
    assert params == [
        "[0.20000000,0.30000000,0.40000000]",
        "game_inning_scores",
        ["kbo_regulations", "markdown_docs"],
        "[0.20000000,0.30000000,0.40000000]",
        2,
    ]


@pytest.mark.asyncio
async def test_record_retrieval_event_inserts_best_effort_payload() -> None:
    conn = _DummyConnection([])

    await record_retrieval_event(
        conn,
        user_query="김도영 홈런",
        intent="stat",
        rewritten_queries=["김도영 home_runs"],
        metadata_filter={"season_year": 2025},
        retrieved_chunk_ids=[1, 2],
        selected_chunk_ids=[1],
        scores=[{"id": 1, "similarity": 0.9}],
        latency_ms=123,
        success=True,
        error_type=None,
    )

    assert conn.last_cursor is not None
    insert_sql, params = conn.last_cursor.executed[-1]
    assert "INSERT INTO rag_retrieval_events" in insert_sql
    assert params[0] == "김도영 홈런"
    assert params[1] == "stat"
    assert '"season_year": 2025' in params[3]


@pytest.mark.asyncio
async def test_record_retrieval_event_swallows_event_insert_failure() -> None:
    conn = _RaisingEventConnection([])

    await record_retrieval_event(
        conn,
        user_query="DB 장애",
        intent="stat",
        rewritten_queries=[],
        metadata_filter={},
        retrieved_chunk_ids=[],
        selected_chunk_ids=[],
        scores=[],
        latency_ms=1,
        success=False,
        error_type="db_unavailable",
    )

    assert conn.last_cursor is not None
    assert any(
        "INSERT INTO rag_retrieval_events" in query
        for query, _params in conn.last_cursor.executed
    )


# ── RRF k intent-aware 테스트 ─────────────────────────────────────────────────


def test_resolve_rrf_k_player_profile_returns_30() -> None:
    assert _resolve_rrf_k("player_profile") == 30


def test_resolve_rrf_k_stats_lookup_returns_40() -> None:
    assert _resolve_rrf_k("stats_lookup") == 40


def test_resolve_rrf_k_comparison_returns_50() -> None:
    assert _resolve_rrf_k("comparison") == 50


def test_resolve_rrf_k_unknown_intent_returns_60() -> None:
    assert _resolve_rrf_k("unknown") == 60


def test_resolve_rrf_k_empty_string_returns_60() -> None:
    assert _resolve_rrf_k("") == 60


def test_rrf_k_by_intent_dict_covers_expected_intents() -> None:
    """_RRF_K_BY_INTENT에 player_profile, stats_lookup, comparison 세 키가 있어야 한다."""
    assert set(_RRF_K_BY_INTENT.keys()) == {
        "player_profile",
        "stats_lookup",
        "comparison",
    }


# ---------------------------------------------------------------------------
# index_status tombstone filter (KBO_playwright tombstones: index_status=DELETED)
# ---------------------------------------------------------------------------


class _IndexStatusCursor(_DummyCursor):
    """Reports rag_chunks and rag_chunks.index_status as present."""

    async def fetchone(self):
        if self.executed:
            query, _ = self.executed[-1]
            if "information_schema.columns" in query and "index_status" in query:
                return (True,)
        return await super().fetchone()


class _IndexStatusConnection(_DummyConnection):
    def cursor(self, row_factory=None):  # noqa: ANN001
        self.last_cursor = _IndexStatusCursor(self.rows)
        return self.last_cursor


@pytest.fixture
def reset_index_status_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core import retrieval

    monkeypatch.setattr(retrieval, "_rag_chunks_index_status_exists", None)
    monkeypatch.setattr(retrieval, "_rag_chunks_index_status_checked_at", 0.0)
    monkeypatch.setattr(retrieval, "_rag_chunks_index_status_check_failed_until", 0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("keyword", [None, "오승환 300세이브"])
async def test_similarity_search_excludes_tombstoned_chunks_when_column_exists(
    reset_index_status_cache: None, keyword: str | None
) -> None:
    conn = _IndexStatusConnection([])

    await similarity_search(conn, [0.1, 0.2, 0.3], limit=3, keyword=keyword)

    sql, _ = conn.last_cursor.executed[1]
    predicate = (
        "COALESCE(index_status, 'ACTIVE') NOT IN ('DELETED', 'TOMBSTONED', 'PURGED')"
    )
    # Hybrid search applies the WHERE clause to both the vector and keyword CTEs.
    assert sql.count(predicate) == (2 if keyword else 1)


@pytest.mark.asyncio
async def test_similarity_search_excludes_tombstones_even_without_lifecycle_filter(
    reset_index_status_cache: None,
) -> None:
    from app.config import get_settings

    conn = _IndexStatusConnection([])
    settings = get_settings().model_copy(
        update={"rag_retrieval_active_filter_enabled": False}
    )

    await similarity_search(conn, [0.1, 0.2, 0.3], limit=3, settings=settings)

    sql, _ = conn.last_cursor.executed[1]
    assert "COALESCE(is_active, true) = true" not in sql
    assert "COALESCE(index_status, 'ACTIVE') NOT IN" in sql


@pytest.mark.asyncio
async def test_similarity_search_skips_index_status_filter_when_column_missing(
    reset_index_status_cache: None,
) -> None:
    # _DummyCursor answers the information_schema.columns probe with no row,
    # i.e. a database where migration 010 has not been applied yet.
    conn = _DummyConnection([])

    await similarity_search(conn, [0.1, 0.2, 0.3], limit=3)

    sql, _ = conn.last_cursor.executed[1]
    assert "index_status" not in sql
    assert "COALESCE(is_active, true) = true" in sql


@pytest.mark.asyncio
async def test_index_status_column_detection_caches_positive_and_rechecks_negative(
    reset_index_status_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core import retrieval

    missing = _DummyConnection([])
    assert await retrieval.rag_chunks_has_index_status(missing) is False
    # Negative result is cached for a while (no probe on every search) ...
    probes_before = missing.last_cursor
    assert await retrieval.rag_chunks_has_index_status(missing) is False
    assert missing.last_cursor is probes_before
    # ... and re-checked after the window, so migration 010 needs no restart.
    monkeypatch.setattr(retrieval, "_rag_chunks_index_status_checked_at", -1e9)
    present = _IndexStatusConnection([])
    assert await retrieval.rag_chunks_has_index_status(present) is True
    assert await retrieval.rag_chunks_has_index_status(_DummyConnection([])) is True


@pytest.mark.asyncio
async def test_index_status_column_detection_failure_fails_closed(
    reset_index_status_cache: None,
) -> None:
    from app.core import retrieval

    class _Broken:
        def cursor(self, *a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("boom")

    # Assume the column exists (keep the tombstone filter); nothing is cached.
    assert await retrieval.rag_chunks_has_index_status(_Broken()) is True
    assert retrieval._rag_chunks_index_status_exists is None


@pytest.mark.parametrize(
    ("status", "visible"),
    [
        ("DELETED", False),
        ("TOMBSTONED", False),
        ("PURGED", False),
        ("ACTIVE", True),
        ("INDEXED", True),
        (None, True),
    ],
)
def test_index_status_visible_predicate_semantics(
    status: str | None, visible: bool
) -> None:
    """Evaluate the generated predicate against a real SQL engine.

    COALESCE / NOT IN are standard SQL, so SQLite gives the same NULL handling
    as PostgreSQL: NULL is treated as ACTIVE, retired states are filtered.
    """
    import sqlite3

    from app.core.retrieval import index_status_visible_sql

    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE rag_chunks (id integer, index_status text)")
    db.execute("INSERT INTO rag_chunks VALUES (1, ?)", (status,))
    rows = db.execute(
        f"SELECT id FROM rag_chunks WHERE {index_status_visible_sql()}"
    ).fetchall()
    assert (rows == [(1,)]) is visible


class _UndefinedIndexStatusCursor(_IndexStatusCursor):
    """Main query fails once as if index_status was dropped after caching."""

    def __init__(self, rows, conn) -> None:  # noqa: ANN001
        super().__init__(rows)
        self._conn = conn

    async def execute(self, query, params=None):  # noqa: ANN001
        await super().execute(query, params)
        if "index_status" in query and "information_schema" not in query:
            self._conn.failed_queries += 1
            from psycopg.errors import UndefinedColumn

            raise UndefinedColumn('column "index_status" does not exist')


class _DroppedColumnConnection(_IndexStatusConnection):
    def __init__(self, rows, transaction_status) -> None:  # noqa: ANN001
        super().__init__(rows)
        self.failed_queries = 0
        self.info = type("_Info", (), {"transaction_status": transaction_status})()

    def cursor(self, row_factory=None):  # noqa: ANN001
        self.last_cursor = _UndefinedIndexStatusCursor(self.rows, self)
        return self.last_cursor


@pytest.mark.asyncio
async def test_similarity_search_recovers_when_index_status_column_is_dropped(
    reset_index_status_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg

    from app.core import retrieval

    # Column was present when cached; then dropped (e.g. rollback of 010).
    monkeypatch.setattr(retrieval, "_rag_chunks_index_status_exists", True)
    conn = _DroppedColumnConnection([], psycopg.pq.TransactionStatus.IDLE)

    await similarity_search(conn, [0.1, 0.2, 0.3], limit=3, keyword="오승환")

    assert conn.failed_queries == 1
    retried_sql, retried_params = conn.last_cursor.executed[-1]
    failed_sql, failed_params = conn.last_cursor.executed[-2]
    assert "index_status" in failed_sql
    assert "index_status" not in retried_sql
    assert retried_params == failed_params
    # Cache invalidated: the next search does not add the filter (no failure).
    assert retrieval._rag_chunks_index_status_exists is False
    assert await retrieval.rag_chunks_has_index_status(conn) is False


@pytest.mark.asyncio
async def test_similarity_search_does_not_retry_inside_failed_transaction(
    reset_index_status_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg
    from psycopg.errors import UndefinedColumn

    from app.core import retrieval

    monkeypatch.setattr(retrieval, "_rag_chunks_index_status_exists", True)
    conn = _DroppedColumnConnection([], psycopg.pq.TransactionStatus.INERROR)

    with pytest.raises(UndefinedColumn):
        await similarity_search(conn, [0.1, 0.2, 0.3], limit=3)

    assert conn.failed_queries == 1
    # Still invalidated, so the caller's next request succeeds without the filter.
    assert retrieval._rag_chunks_index_status_exists is False


@pytest.mark.asyncio
async def test_unrelated_undefined_column_is_not_swallowed(
    reset_index_status_cache: None,
) -> None:
    from psycopg.errors import UndefinedColumn

    from app.core import retrieval

    class _Cursor:
        async def execute(self, query, params=None):  # noqa: ANN001
            raise UndefinedColumn('column "quality_score" does not exist')

    with pytest.raises(UndefinedColumn):
        await retrieval.execute_with_index_status_fallback(
            object(), _Cursor(), "SELECT 1", [], filtered=True
        )


@pytest.mark.asyncio
async def test_index_status_check_failure_is_not_cached_and_warning_is_rate_limited(
    reset_index_status_cache: None, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    import logging

    from app.core import retrieval

    calls = {"n": 0}

    class _Broken:
        def cursor(self, *a, **k):  # noqa: ANN002, ANN003
            calls["n"] += 1
            raise RuntimeError("boom")

    caplog.set_level(logging.WARNING, logger=retrieval.logger.name)
    for _ in range(5):
        assert await retrieval.rag_chunks_has_index_status(_Broken()) is True
    # One probe and one warning per short backoff window, no cached answer.
    assert calls["n"] == 1
    assert caplog.text.count("index_status column check failed") == 1
    assert retrieval._rag_chunks_index_status_exists is None

    # After the backoff the check runs again; a successful check is cached.
    monkeypatch.setattr(retrieval, "_rag_chunks_index_status_check_failed_until", 0.0)
    assert await retrieval.rag_chunks_has_index_status(_DummyConnection([])) is False
    assert retrieval._rag_chunks_index_status_exists is False


@pytest.mark.asyncio
async def test_failed_check_on_aborted_connection_keeps_tombstones_hidden(
    reset_index_status_cache: None,
) -> None:
    """QA repro: a check on an aborted-transaction connection must not disable the
    filter for the next (healthy) search."""
    from psycopg.errors import InFailedSqlTransaction

    from app.core import retrieval

    class _AbortedCursor(_DummyCursor):
        async def execute(self, query, params=None):  # noqa: ANN001
            raise InFailedSqlTransaction("current transaction is aborted")

    class _AbortedConnection(_DummyConnection):
        def cursor(self, row_factory=None):  # noqa: ANN001
            return _AbortedCursor([])

    assert await retrieval.rag_chunks_has_index_status(_AbortedConnection([])) is True

    healthy = _IndexStatusConnection([])
    await similarity_search(healthy, [0.1, 0.2, 0.3], limit=3)
    sql, _ = healthy.last_cursor.executed[1]
    assert (
        "COALESCE(index_status, 'ACTIVE') NOT IN ('DELETED', 'TOMBSTONED', 'PURGED')"
        in sql
    )


@pytest.mark.asyncio
async def test_failed_check_with_missing_column_falls_back_without_filter(
    reset_index_status_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assuming True is safe: if the column really is missing, the query's
    UndefinedColumn resets the cache and the search is retried unfiltered."""
    import psycopg

    from app.core import retrieval

    monkeypatch.setattr(
        retrieval, "_rag_chunks_index_status_check_failed_until", float("inf")
    )
    conn = _DroppedColumnConnection([], psycopg.pq.TransactionStatus.IDLE)

    await similarity_search(conn, [0.1, 0.2, 0.3], limit=3)

    assert conn.failed_queries == 1
    assert "index_status" not in conn.last_cursor.executed[-1][0]
    assert retrieval._rag_chunks_index_status_exists is False
    assert retrieval._rag_chunks_index_status_check_failed_until == 0.0
