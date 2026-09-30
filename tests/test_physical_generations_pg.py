"""Physical blue/green against a real PostgreSQL + pgvector.

Skipped unless TEST_PGVECTOR_URL points at a *disposable* database that already
has ``app/db/schema.sql`` and migrations 007 + 009 applied, e.g.::

    docker run -d --name pgvec-test -e POSTGRES_PASSWORD=test -e POSTGRES_DB=ragtest \
        -p 55432:5432 pgvector/pgvector:pg17
    TEST_PGVECTOR_URL=postgresql://postgres:test@localhost:55432/ragtest \
        pytest tests/test_physical_generations_pg.py

The tables are TRUNCATEd on every run — never point this at real data.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

psycopg = pytest.importorskip("psycopg")

from app.config import get_settings  # noqa: E402
from app.core import embedding_generations as eg  # noqa: E402
from app.core import retrieval  # noqa: E402
from app.core.retrieval_contract import RetrievalContractViolation  # noqa: E402

URL = os.getenv("TEST_PGVECTOR_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not URL, reason="TEST_PGVECTOR_URL not set"),
]

N = 12
DIM_A, DIM_B = 1536, 256  # inline column is vector(1536); B is a different model


def vec(dim: int, hot: int) -> list[float]:
    v = [0.0] * dim
    v[hot % dim] = 1.0
    return v


def lit(values: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in values) + "]"


class RecordingCursor(psycopg.AsyncCursor):
    executed: list = []

    async def execute(self, query, params=None, **kw):
        RecordingCursor.executed.append((query, params))
        return await super().execute(query, params, **kw)


@pytest_asyncio.fixture
async def conn():
    async with await psycopg.AsyncConnection.connect(
        URL, autocommit=True, cursor_factory=RecordingCursor
    ) as c:
        await c.execute(
            "TRUNCATE rag_chunk_embeddings, rag_embedding_generation_events, "
            "rag_embedding_generations, rag_chunks RESTART IDENTITY CASCADE"
        )
        for name in ("idx_rag_emb_g1", "idx_rag_emb_g2"):
            await c.execute(f"DROP INDEX IF EXISTS {name}")
        eg.reset_active_cache()
        RecordingCursor.executed = []
        yield c
        eg.reset_active_cache()


async def insert_chunk(conn, i, *, inline=True, model="m1", version=1, dim=DIM_A):
    """Chunk i; inline embedding is one-hot at position i (model m1, 1536-d)."""
    emb = lit(vec(dim, i)) if inline else None
    cur = await conn.execute(
        """
        INSERT INTO rag_chunks
            (source_table, source_row_id, title, content, content_hash, team_id,
             season_year, embedding, embedding_model, embedding_dim,
             embedding_version)
        VALUES ('team_summary', %s, %s, %s, %s, 'LG', 2025, %s::vector, %s, %s, %s)
        RETURNING id
        """,
        (
            f"row-{i}",
            f"title {i}",
            f"chunk number {i} about baseball",
            f"hash-{i}",
            emb,
            model if inline else None,
            dim if inline else None,
            version if inline else None,
        ),
    )
    return (await cur.fetchone())[0]


def settings(**over):
    return get_settings().model_copy(
        update={"rag_embedding_store": "generations", **over}
    )


async def search(conn, hot, dim, **kw):
    return await retrieval.similarity_search(
        conn,
        vec(dim, hot),
        limit=3,
        filters=kw.pop("filters", {"team_id": "LG"}),
        settings=kw.pop("settings", settings()),
        **kw,
    )


async def seed_g1(conn):
    """Chunks 0-5 pre-exist (backfilled); 6-11 arrive after registration
    (mirrored by the trigger) — both routes must end up in g1."""
    for i in range(6):
        await insert_chunk(conn, i)
    await eg.register_generation(
        conn,
        generation_id="g1",
        embedding_model="m1",
        embedding_dim=DIM_A,
        embedding_version=1,
        mirror_inline=True,
    )
    backfill = await eg.backfill_from_inline(conn, "g1", batch_size=4)
    for i in range(6, N):
        await insert_chunk(conn, i)
    return backfill


async def test_backfill_and_trigger_populate_the_generation_then_it_serves(conn):
    backfill = await seed_g1(conn)
    assert backfill["copied"] == 6 and backfill["batches"] == 2  # keyset batches
    cur = await conn.execute(
        "SELECT count(*) FROM rag_chunk_embeddings WHERE generation_id='g1'"
    )
    assert (await cur.fetchone())[0] == N  # 6 backfilled + 6 mirrored by trigger

    audit = await eg.audit_generation_coverage(
        conn, await eg.get_generation(conn, "g1", extended=True), store="generations"
    )
    assert audit["coverage"] == 1.0 and audit["stale"] == 0

    # Activation needs a built ANN index in the physical store.
    with pytest.raises(eg.GenerationError, match="ANN index"):
        await eg.activate_generation(conn, "g1", store="generations")
    built = await eg.build_generation_index(conn, "g1")
    assert built["ready"] and built["index"] == "idx_rag_emb_g1"
    await eg.activate_generation(conn, "g1", store="generations")

    rows = await search(conn, 3, DIM_A)
    assert rows[0]["source_row_id"] == "row-3"
    assert rows[0]["retrieval_backend"] == "postgres"
    assert rows[0]["index_generation"] == "g1"
    assert rows[0]["similarity"] == pytest.approx(1.0, abs=1e-3)


async def test_new_generation_coexists_and_rollback_restores_the_old_one(conn):
    await seed_g1(conn)
    await eg.build_generation_index(conn, "g1")
    await eg.activate_generation(conn, "g1", store="generations")

    # g2: a different model AND dimension, embeddings reversed so its nearest
    # neighbours differ from g1's — proof that results come from g2's rows.
    await eg.register_generation(
        conn,
        generation_id="g2",
        embedding_model="m2",
        embedding_dim=DIM_B,
        embedding_version=1,
    )
    cur = await conn.execute("SELECT id, content_hash FROM rag_chunks ORDER BY id")
    chunks = await cur.fetchall()
    partial = [(cid, vec(DIM_B, N - 1 - i), h) for i, (cid, h) in enumerate(chunks)][:6]
    assert await eg.write_embeddings(conn, "g2", partial) == 6

    # Half-built generation cannot be activated (coverage), nor without an index.
    with pytest.raises(eg.GenerationError, match="ANN index"):
        await eg.activate_generation(conn, "g2", store="generations")
    await eg.build_generation_index(conn, "g2")
    with pytest.raises(eg.GenerationError, match="coverage"):
        await eg.activate_generation(conn, "g2", store="generations")

    rest = [(cid, vec(DIM_B, N - 1 - i), h) for i, (cid, h) in enumerate(chunks)][6:]
    await eg.write_embeddings(conn, "g2", rest)
    # Wrong-dimension vectors can never enter a generation.
    with pytest.raises(eg.GenerationError, match="dims"):
        await eg.write_embeddings(conn, "g2", [(chunks[0][0], vec(DIM_A, 0), None)])

    result = await eg.activate_generation(conn, "g2", store="generations")
    assert result["previous_id"] == "g1"
    top_g2 = await search(conn, 3, DIM_B)
    assert top_g2[0]["source_row_id"] == f"row-{N - 1 - 3}"
    assert top_g2[0]["index_generation"] == "g2"

    # g1 was never touched while g2 was built and served.
    cur = await conn.execute(
        "SELECT count(*) FROM rag_chunk_embeddings WHERE generation_id='g1'"
    )
    assert (await cur.fetchone())[0] == N

    rolled = await eg.rollback_generation(conn, store="generations")
    assert rolled["generation_id"] == "g1"
    top_g1 = await search(conn, 3, DIM_A)
    assert top_g1[0]["source_row_id"] == "row-3"
    assert top_g1[0]["index_generation"] == "g1"
    with pytest.raises(eg.GenerationError, match="already"):
        await eg.rollback_generation(conn, store="generations")

    # Housekeeping: the failed generation can be dropped; the ACTIVE one cannot.
    with pytest.raises(eg.GenerationError, match="ACTIVE"):
        await eg.drop_generation_data(conn, "g1")
    dropped = await eg.drop_generation_data(conn, "g2")
    assert dropped["deleted_rows"] == N
    cur = await conn.execute("SELECT to_regclass('idx_rag_emb_g2')")
    assert (await cur.fetchone())[0] is None


async def test_rollback_target_cannot_be_dropped(conn):
    await seed_g1(conn)
    await eg.build_generation_index(conn, "g1")
    await eg.activate_generation(conn, "g1", store="generations")
    await eg.register_generation(
        conn,
        generation_id="g2",
        embedding_model="m2",
        embedding_dim=DIM_B,
        embedding_version=1,
    )
    cur = await conn.execute("SELECT id, content_hash FROM rag_chunks ORDER BY id")
    rows = [(cid, vec(DIM_B, i), h) for i, (cid, h) in enumerate(await cur.fetchall())]
    await eg.write_embeddings(conn, "g2", rows)
    await eg.build_generation_index(conn, "g2")
    await eg.activate_generation(conn, "g2", store="generations")
    with pytest.raises(eg.GenerationError, match="rollback target"):
        await eg.drop_generation_data(conn, "g1")


async def test_stale_embeddings_are_never_served_and_trigger_heals_them(conn):
    await seed_g1(conn)
    await eg.build_generation_index(conn, "g1")
    await eg.activate_generation(conn, "g1", store="generations")

    # Content changed but not re-embedded: the old vector describes old text.
    await conn.execute(
        "UPDATE rag_chunks SET content_hash = 'changed', "
        "content = 'rewritten' WHERE source_row_id = 'row-3'"
    )
    rows = await search(conn, 3, DIM_A)
    assert "row-3" not in [r["source_row_id"] for r in rows]
    audit = await eg.audit_generation_coverage(
        conn, await eg.get_generation(conn, "g1", extended=True), store="generations"
    )
    assert audit["stale"] == 1 and audit["coverage"] < 1.0

    # Re-embedding through the ordinary inline writer is mirrored automatically.
    await conn.execute(
        "UPDATE rag_chunks SET embedding = %s::vector WHERE source_row_id = 'row-3'",
        (lit(vec(DIM_A, 3)),),
    )
    rows = await search(conn, 3, DIM_A)
    assert rows[0]["source_row_id"] == "row-3"


async def test_hybrid_keyword_path_runs_and_respects_the_generation(conn):
    await seed_g1(conn)
    await eg.build_generation_index(conn, "g1")
    await eg.activate_generation(conn, "g1", store="generations")
    rows = await retrieval.similarity_search(
        conn,
        vec(DIM_A, 5),
        limit=3,
        filters={"team_id": "LG"},
        keyword="5",  # tsquery on the chunk number: matches only row-5
        settings=settings(),
    )
    assert rows and all(r["index_generation"] == "g1" for r in rows)
    assert rows[0]["source_row_id"] == "row-5"
    # A chunk with no fresh embedding in the active generation is invisible to
    # the keyword arm too (it could not be scored).
    await conn.execute(
        "DELETE FROM rag_chunk_embeddings WHERE generation_id='g1' AND chunk_id = "
        "(SELECT id FROM rag_chunks WHERE source_row_id='row-7')"
    )
    rows = await retrieval.similarity_search(
        conn,
        vec(DIM_A, 7),
        limit=20,
        filters={},
        keyword="7",
        settings=settings(),
    )
    assert "row-7" not in [r["source_row_id"] for r in rows]


async def test_query_uses_the_generations_partial_hnsw_index(conn):
    await seed_g1(conn)
    await eg.build_generation_index(conn, "g1")
    await eg.activate_generation(conn, "g1", store="generations")
    RecordingCursor.executed = []
    await search(conn, 3, DIM_A)
    sql, params = next(
        (q, p) for q, p in RecordingCursor.executed if "rag_chunk_embeddings" in str(q)
    )
    # Tiny tables make the planner's own choice meaningless (and stats-dependent).
    # Disabling explicit Sort leaves only an *ordered* ANN scan able to satisfy
    # ORDER BY .. LIMIT, so this asserts the join query CAN use the generation's
    # partial HNSW index (expression + predicate match), deterministically.
    await conn.execute("SET enable_seqscan = off")
    await conn.execute("SET enable_sort = off")
    cur = await conn.execute("EXPLAIN " + sql, params)
    plan = "\n".join(row[0] for row in await cur.fetchall())
    assert "idx_rag_emb_g1" in plan, plan


async def test_generations_mode_fails_closed(conn):
    await seed_g1(conn)  # registered but NOT active
    with pytest.raises(RetrievalContractViolation, match="no generation is ACTIVE"):
        await search(conn, 1, DIM_A)


async def test_inline_mode_is_unchanged(conn):
    for i in range(N):
        await insert_chunk(conn, i)
    rows = await retrieval.similarity_search(
        conn,
        vec(DIM_A, 4),
        limit=3,
        filters={"team_id": "LG"},
        settings=get_settings().model_copy(update={"rag_embedding_store": "inline"}),
    )
    assert rows[0]["source_row_id"] == "row-4"
    assert rows[0]["retrieval_backend"] == "postgres"
    assert rows[0]["index_generation"] is None


async def test_generation_ids_are_validated_tokens(conn):
    for bad in ("G1", "g 1", "g;drop", "-x", "a" * 41, ""):
        with pytest.raises(eg.GenerationError):
            await eg.register_generation(
                conn,
                generation_id=bad,
                embedding_model="m",
                embedding_dim=8,
                embedding_version=1,
            )
    with pytest.raises(eg.GenerationError, match="embedding_dim"):
        await eg.register_generation(
            conn,
            generation_id="big",
            embedding_model="m",
            embedding_dim=4001,
            embedding_version=1,
        )


async def test_real_ingest_upsert_keeps_the_active_generation_correct(conn):
    """Drive the project's own writer SQL (RAG_CHUNKS_UPSERT_SQL), not raw UPDATEs."""
    from app.core.rag_storage import RAG_CHUNKS_UPSERT_SQL, build_upsert_tuple

    await seed_g1(conn)
    await eg.build_generation_index(conn, "g1")
    await eg.activate_generation(conn, "g1", store="generations")

    def upsert(row_id, content, content_hash, embedding):
        fields = {
            "metadata": {},
            "source_type": "static",
            "source_uri": None,
            "topic_key": None,
            "content_hash": content_hash,
            "chunk_hash": content_hash,
            "embedding_model": "m1",
            "embedding_dim": DIM_A,
            "embedding_version": 1,
            "chunking_version": 1,
            "quality_score": 0.9,
        }
        return build_upsert_tuple(
            meta={},
            storage_fields=fields,
            season_year=2025,
            season_id=None,
            league_type_code=None,
            team_id="LG",
            player_id=None,
            source_table="team_summary",
            source_row_id=row_id,
            title="t",
            content=content,
            embedding_text=embedding,
        )

    # 1) Content changes and is re-embedded in the same write: mirrored, fresh.
    await conn.execute(
        RAG_CHUNKS_UPSERT_SQL,
        upsert("row-3", "new text about baseball", "hash-3b", lit(vec(DIM_A, 3))),
    )
    assert (await search(conn, 3, DIM_A))[0]["source_row_id"] == "row-3"

    # 2) Content changes but the write carries no embedding: the inline vector is
    #    cleared by the writer and the generation row is stale -> not served.
    await conn.execute(
        RAG_CHUNKS_UPSERT_SQL,
        upsert("row-3", "different again", "hash-3c", None),
    )
    rows = await search(conn, 3, DIM_A)
    assert "row-3" not in [r["source_row_id"] for r in rows]
    audit = await eg.audit_generation_coverage(
        conn, await eg.get_generation(conn, "g1", extended=True), store="generations"
    )
    assert audit["coverage"] < 1.0

    # 3) A later embed pass (same content hash) restores it.
    await conn.execute(
        RAG_CHUNKS_UPSERT_SQL,
        upsert("row-3", "different again", "hash-3c", lit(vec(DIM_A, 3))),
    )
    assert (await search(conn, 3, DIM_A))[0]["source_row_id"] == "row-3"


class _Pool:
    """Minimal pool over the test connection (readiness only needs .connection)."""

    def __init__(self, conn):
        self._conn = conn

    def connection(self, timeout=None):
        conn = self._conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


async def test_readiness_reflects_the_live_state_of_the_active_generation(conn):
    from app.core import rag_readiness

    cfg = settings()
    down = await rag_readiness._probe_postgres(_Pool(conn), cfg)
    assert down["rag_index"]["code"] == "RAG_GENERATION_NOT_ACTIVE"

    await seed_g1(conn)
    await eg.build_generation_index(conn, "g1")
    await eg.activate_generation(conn, "g1", store="generations")
    up = await rag_readiness._probe_postgres(_Pool(conn), cfg)
    assert up["rag_index"]["status"] == "UP" and up["rag_vector"]["status"] == "UP"
    assert up["rag_vector"]["generation"] == "g1"

    # Registry still says index_ready, but the index is gone: readiness must see it.
    await conn.execute("DROP INDEX idx_rag_emb_g1")
    broken = await rag_readiness._probe_postgres(_Pool(conn), cfg)
    assert broken["rag_index"]["code"] == "RAG_VECTOR_INDEX_NOT_READY"
