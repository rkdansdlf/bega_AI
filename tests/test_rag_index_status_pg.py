"""index_status tombstones against a real PostgreSQL + pgvector.

KBO_playwright retires chunks with ``index_status = 'DELETED'`` (and, where the
columns exist, ``is_active = false`` / ``valid_to = now()``). These tests check
that PostgreSQL retrieval never returns such chunks and that
``scripts/sync_rag_chunks.py`` carries the tombstone to the destination.

Skipped unless TEST_PGVECTOR_URL points at a *disposable* database that already
has ``app/db/schema.sql`` applied (migration 010 is applied by the fixture). The
sync test also needs TEST_PGVECTOR_SYNC_DEST_URL, a second disposable database
with the same schema::

    TEST_PGVECTOR_URL=postgresql://postgres:test@localhost:55432/ragsrc \\
    TEST_PGVECTOR_SYNC_DEST_URL=postgresql://postgres:test@localhost:55432/ragdst \\
        pytest tests/test_rag_index_status_pg.py

rag_chunks is TRUNCATEd on every run — never point this at real data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import pytest_asyncio

psycopg = pytest.importorskip("psycopg")

from app.config import get_settings  # noqa: E402
from app.core import retrieval  # noqa: E402
from scripts import sync_rag_chunks as sync_script  # noqa: E402

URL = os.getenv("TEST_PGVECTOR_URL")
DEST_URL = os.getenv("TEST_PGVECTOR_SYNC_DEST_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_PGVECTOR_URL not set")

MIGRATION_010 = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "db"
    / "migrations"
    / "010_rag_chunks_index_status.sql"
).read_text(encoding="utf-8")
DIM = 1536
SEARCH_PATH = "SET search_path TO public, extensions"


def vec_literal(hot: int = 0) -> str:
    values = [0.0] * DIM
    values[hot] = 1.0
    return "[" + ",".join(f"{x:.8f}" for x in values) + "]"


# (source_row_id, index_status, is_active)
CHUNKS = [
    ("row-active", "ACTIVE", True),
    ("row-null", None, None),
    ("row-deleted-active-flag", "DELETED", True),
    ("row-deleted", "DELETED", False),
]


def _prepare(url: str) -> None:
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(SEARCH_PATH)
        conn.execute(MIGRATION_010)
        conn.execute("TRUNCATE rag_chunks RESTART IDENTITY CASCADE")


def _insert(conn, row_id: str, status, is_active, *, table: str = "milestone"):
    conn.execute(
        """
        INSERT INTO rag_chunks
            (source_table, source_row_id, title, content, content_hash,
             season_year, embedding, index_status, is_active)
        VALUES (%s, %s, %s, %s, %s, 2026, %s::vector, %s, %s)
        """,
        (
            table,
            row_id,
            f"오승환 300세이브 {row_id}",
            f"오승환 300세이브 milestone {row_id}",
            f"hash-{row_id}",
            vec_literal(),
            status,
            is_active,
        ),
    )


@pytest.fixture(autouse=True)
def _reset_index_status_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrieval, "_rag_chunks_index_status_exists", None)
    monkeypatch.setattr(retrieval, "_rag_chunks_index_status_checked_at", 0.0)
    monkeypatch.setattr(retrieval, "_rag_chunks_index_status_check_failed_until", 0.0)


@pytest.fixture
def seeded_source() -> None:
    _prepare(URL)
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute(SEARCH_PATH)
        for row_id, status, is_active in CHUNKS:
            _insert(conn, row_id, status, is_active)


@pytest_asyncio.fixture
async def aconn(seeded_source):
    async with await psycopg.AsyncConnection.connect(URL) as conn:
        yield conn
        await conn.rollback()


def _settings():
    return get_settings().model_copy(
        update={"rag_embedding_store": "inline", "rag_generation_gate_enabled": False}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("keyword", [None, "오승환"])
async def test_similarity_search_hides_deleted_chunks(aconn, keyword) -> None:
    rows = await retrieval.similarity_search(
        aconn,
        [1.0] + [0.0] * (DIM - 1),
        limit=10,
        keyword=keyword,
        settings=_settings(),
    )

    assert sorted(row["source_row_id"] for row in rows) == ["row-active", "row-null"]


@pytest.mark.asyncio
async def test_similarity_search_works_before_migration_010(aconn) -> None:
    # Inside the test transaction (rolled back by the fixture): a DB without the
    # column must not fail with UndefinedColumn; tombstones are just not filtered.
    await aconn.execute("ALTER TABLE rag_chunks DROP COLUMN index_status")

    rows = await retrieval.similarity_search(
        aconn, [1.0] + [0.0] * (DIM - 1), limit=10, settings=_settings()
    )

    assert sorted(row["source_row_id"] for row in rows) == [
        "row-active",
        "row-deleted-active-flag",
        "row-null",
    ]


@pytest.mark.skipif(not DEST_URL, reason="TEST_PGVECTOR_SYNC_DEST_URL not set")
def test_sync_propagates_index_status_and_is_active_false(seeded_source) -> None:
    _prepare(DEST_URL)
    with psycopg.connect(DEST_URL, autocommit=True) as conn:
        conn.execute(SEARCH_PATH)
        # Stale destination: the chunk is still ACTIVE there.
        _insert(conn, "row-deleted", "ACTIVE", True)
        _insert(conn, "row-deleted-active-flag", "ACTIVE", True)

    result = sync_script.sync_rag_chunks(
        source_db_url=URL,
        dest_db_url=DEST_URL,
        source_tables=[],
        season_year=None,
        batch_size=2,
        commit_interval=2,
        limit=None,
        known_total_rows=None,
        dry_run=False,
        truncate_dest=False,
    )

    assert result["synced_rows"] == len(CHUNKS)
    with psycopg.connect(DEST_URL) as conn:
        got = dict(
            (row[0], (row[1], row[2]))
            for row in conn.execute(
                "SELECT source_row_id, index_status, is_active FROM rag_chunks"
            ).fetchall()
        )
    assert got == {
        "row-active": ("ACTIVE", True),
        "row-null": ("ACTIVE", True),
        "row-deleted-active-flag": ("DELETED", True),
        "row-deleted": ("DELETED", False),
    }


@pytest.mark.asyncio
async def test_similarity_search_survives_column_drop_after_caching(
    seeded_source,
) -> None:
    async with await psycopg.AsyncConnection.connect(URL, autocommit=True) as conn:
        first = await retrieval.similarity_search(
            conn, [1.0] + [0.0] * (DIM - 1), limit=10, settings=_settings()
        )
        assert retrieval._rag_chunks_index_status_exists is True
        assert len(first) == 2
        try:
            await conn.execute("ALTER TABLE rag_chunks DROP COLUMN index_status")
            # Positive cache is stale: the query fails once, is retried without
            # the filter, and the cache is invalidated (no restart needed).
            rows = await retrieval.similarity_search(
                conn, [1.0] + [0.0] * (DIM - 1), limit=10, settings=_settings()
            )
            assert len(rows) == 3
            assert retrieval._rag_chunks_index_status_exists is False
        finally:
            await conn.execute(MIGRATION_010)


def _drop_source_index_status() -> None:
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("ALTER TABLE rag_chunks DROP COLUMN index_status")


@pytest.fixture
def tombstoned_dest_and_legacy_source(seeded_source):
    """Destination cleaned by KBO_playwright #57; source bega DB without the column."""
    _prepare(DEST_URL)
    with psycopg.connect(DEST_URL, autocommit=True) as conn:
        conn.execute(SEARCH_PATH)
        _insert(conn, "row-deleted", "DELETED", False)
        _insert(conn, "row-deleted-active-flag", "DELETED", True)
        _insert(conn, "row-dest-only", "DELETED", False)
    _drop_source_index_status()
    try:
        yield
    finally:
        _prepare(URL)  # re-applies 010 on the source


def _dest_statuses() -> dict:
    with psycopg.connect(DEST_URL) as conn:
        return dict(
            (row[0], row[1])
            for row in conn.execute(
                "SELECT source_row_id, index_status FROM rag_chunks"
            ).fetchall()
        )


def _sync(**overrides):
    kwargs = dict(
        source_db_url=URL,
        dest_db_url=DEST_URL,
        source_tables=[],
        season_year=None,
        batch_size=2,
        commit_interval=2,
        limit=None,
        known_total_rows=None,
        dry_run=False,
        truncate_dest=False,
    )
    kwargs.update(overrides)
    return sync_script.sync_rag_chunks(**kwargs)


@pytest.mark.skipif(not DEST_URL, reason="TEST_PGVECTOR_SYNC_DEST_URL not set")
def test_sync_from_legacy_source_refuses_truncate(
    tombstoned_dest_and_legacy_source,
) -> None:
    before = _dest_statuses()

    with pytest.raises(RuntimeError, match="--truncate-dest refused"):
        _sync(truncate_dest=True)

    assert _dest_statuses() == before


@pytest.mark.skipif(not DEST_URL, reason="TEST_PGVECTOR_SYNC_DEST_URL not set")
def test_sync_from_legacy_source_keeps_destination_tombstones(
    tombstoned_dest_and_legacy_source,
) -> None:
    result = _sync()

    assert result["synced_rows"] == len(CHUNKS)
    assert _dest_statuses() == {
        # Existing destination rows keep their tombstone ...
        "row-deleted": "DELETED",
        "row-deleted-active-flag": "DELETED",
        "row-dest-only": "DELETED",
        # ... and rows new to the destination get its default.
        "row-active": "ACTIVE",
        "row-null": "ACTIVE",
    }


@pytest.mark.asyncio
async def test_check_on_aborted_transaction_does_not_expose_tombstones(
    seeded_source,
) -> None:
    """QA repro: column check on an aborted transaction, then a normal search."""
    async with await psycopg.AsyncConnection.connect(URL) as aborted:
        with pytest.raises(psycopg.errors.DivisionByZero):
            await aborted.execute("SELECT 1/0")
        assert await retrieval.rag_chunks_has_index_status(aborted) is True
        await aborted.rollback()

    async with await psycopg.AsyncConnection.connect(URL, autocommit=True) as conn:
        rows = await retrieval.similarity_search(
            conn, [1.0] + [0.0] * (DIM - 1), limit=10, settings=_settings()
        )
    assert sorted(row["source_row_id"] for row in rows) == ["row-active", "row-null"]


@pytest.mark.asyncio
async def test_failed_check_on_db_without_column_still_searches(seeded_source) -> None:
    async with await psycopg.AsyncConnection.connect(URL, autocommit=True) as conn:
        try:
            await conn.execute("ALTER TABLE rag_chunks DROP COLUMN index_status")
            async with await psycopg.AsyncConnection.connect(URL) as aborted:
                with pytest.raises(psycopg.errors.DivisionByZero):
                    await aborted.execute("SELECT 1/0")
                assert await retrieval.rag_chunks_has_index_status(aborted) is True
            rows = await retrieval.similarity_search(
                conn, [1.0] + [0.0] * (DIM - 1), limit=10, settings=_settings()
            )
            assert len(rows) == 3
            assert retrieval._rag_chunks_index_status_exists is False
        finally:
            await conn.execute(MIGRATION_010)


def test_migration_010_rerun_takes_no_lock_on_rag_chunks(seeded_source) -> None:
    """Re-running 010 (every deploy) must not queue behind/ahead of readers."""
    with (
        psycopg.connect(URL) as reader,
        psycopg.connect(URL, autocommit=True) as migrator,
    ):
        # Long reader transaction holding ACCESS SHARE on rag_chunks.
        reader.execute("SELECT count(*) FROM rag_chunks").fetchone()
        # Would wait (and fail with LockNotAvailable) if it needed ACCESS EXCLUSIVE.
        migrator.execute("SET lock_timeout = '200ms'")
        migrator.execute(MIGRATION_010)
        held = migrator.execute("""
            SELECT count(*) FROM pg_locks
            WHERE relation = 'rag_chunks'::regclass AND mode = 'AccessExclusiveLock'
            """).fetchone()[0]
        assert held == 0
        reader.rollback()


@pytest_asyncio.fixture
async def alt_schema_conn(seeded_source):
    """Non-autocommit connection whose search_path resolves rag_chunks to kbo_alt.

    Everything (schema, table, search_path) lives in one transaction that is
    rolled back, so public.rag_chunks is left untouched.
    """
    async with await psycopg.AsyncConnection.connect(URL) as conn:
        await conn.execute("CREATE SCHEMA kbo_alt")
        await conn.execute(
            "CREATE TABLE kbo_alt.rag_chunks (LIKE public.rag_chunks INCLUDING DEFAULTS)"
        )
        await conn.execute("SET LOCAL search_path TO kbo_alt, public, extensions")
        try:
            yield conn
        finally:
            await conn.rollback()


@pytest.mark.asyncio
async def test_column_check_follows_search_path_when_only_public_lacks_column(
    alt_schema_conn,
) -> None:
    """QA #31 item 5: the probe resolves rag_chunks via search_path, not 'public'."""
    conn = alt_schema_conn
    await conn.execute("ALTER TABLE public.rag_chunks DROP COLUMN index_status")

    assert await retrieval.rag_chunks_has_index_status(conn) is True


@pytest.mark.asyncio
async def test_similarity_search_probes_under_its_own_search_path(
    alt_schema_conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Session search_path resolves a column-less kbo_alt.rag_chunks, but
    similarity_search pins its own search_path (public first) before probing, so
    the filter still applies to the public table it actually queries."""
    conn = alt_schema_conn
    await conn.execute("ALTER TABLE kbo_alt.rag_chunks DROP COLUMN index_status")
    # Warm table-exists cache: _rag_chunks_exists then skips its own SET search_path.
    monkeypatch.setattr(retrieval, "_rag_chunks_table_exists", True)

    rows = await retrieval.similarity_search(
        conn, [1.0] + [0.0] * (DIM - 1), limit=10, settings=_settings()
    )

    assert retrieval._rag_chunks_index_status_exists is True
    assert sorted(row["source_row_id"] for row in rows) == ["row-active", "row-null"]


@pytest.mark.asyncio
async def test_column_check_follows_search_path_when_resolved_table_lacks_column(
    alt_schema_conn,
) -> None:
    """public.rag_chunks has index_status, but the resolved kbo_alt table does not."""
    conn = alt_schema_conn
    await conn.execute("ALTER TABLE kbo_alt.rag_chunks DROP COLUMN index_status")

    assert await retrieval.rag_chunks_has_index_status(conn) is False


@pytest.mark.asyncio
async def test_column_check_is_false_when_rag_chunks_is_not_on_search_path(
    seeded_source,
) -> None:
    async with await psycopg.AsyncConnection.connect(URL) as conn:
        try:
            await conn.execute("SET LOCAL search_path TO pg_catalog")
            assert await retrieval.rag_chunks_has_index_status(conn) is False
        finally:
            await conn.rollback()
