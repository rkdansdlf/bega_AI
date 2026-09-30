"""Embedding generation registry: audited activation, rollback, and the
mechanisms that keep old/new embeddings from mixing.

Two storage modes (``RAG_EMBEDDING_STORE``):

* ``inline`` – one embedding per chunk in ``rag_chunks.embedding``; a signature
  filter (``generation_filter_sql``) keeps generations apart, but old rows are
  overwritten in place, so rollback is only possible while they still exist.
* ``generations`` – one row per (generation, chunk) in ``rag_chunk_embeddings``
  (migration 009) with a partial HNSW index per generation. Old and new
  generations physically coexist, so activation is an atomic pointer flip and
  rollback restores rows that were never touched.

Transactions: every mutation runs in ``conn.transaction()`` so the ACTIVE
pointer flips atomically (the partial unique index guarantees a single ACTIVE
row even under concurrent operators). ``CREATE INDEX CONCURRENTLY`` cannot run
inside a transaction, so index builds are separate calls on an autocommit
connection.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MIN_COVERAGE = 0.995
_ACTIVE_CACHE_TTL_S = 30.0
_active_cache: Dict[str, Any] = {"at": 0.0, "value": None}
_active_cache_ext: Dict[str, Any] = {"at": 0.0, "value": None}

STORE_INLINE = "inline"
STORE_GENERATIONS = "generations"
# generation_id is interpolated into DDL / index predicates (a literal is what
# lets the planner match a partial index), so it must be a strict token.
GENERATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
# halfvec (used for the ANN expression index) supports up to 4000 dimensions.
MAX_INDEXED_DIM = 4000


class GenerationError(RuntimeError):
    pass


def validate_generation_id(generation_id: str) -> str:
    if not isinstance(generation_id, str) or not GENERATION_ID_RE.match(generation_id):
        raise GenerationError(
            f"invalid generation_id {generation_id!r}: use [a-z0-9_-], 1-40 chars, "
            "starting with a letter or digit"
        )
    return generation_id


def index_name_for(generation_id: str) -> str:
    return "idx_rag_emb_" + validate_generation_id(generation_id).replace("-", "_")


@dataclass(frozen=True)
class Generation:
    generation_id: str
    embedding_model: str
    embedding_dim: int
    embedding_version: int
    status: str = "BUILDING"
    # Populated only by the extended select (migration 009).
    mirror_inline: bool = False
    index_ready: bool = False

    @property
    def signature(self) -> Tuple[str, int, int]:
        return (self.embedding_model, self.embedding_dim, self.embedding_version)


def _row_to_generation(row: Any) -> Optional[Generation]:
    if row is None:
        return None
    extra = {}
    if len(row) > 5:
        extra = {"mirror_inline": bool(row[5]), "index_ready": bool(row[6])}
    return Generation(
        str(row[0]), str(row[1]), int(row[2]), int(row[3]), str(row[4]), **extra
    )


_SELECT = (
    "SELECT generation_id, embedding_model, embedding_dim, embedding_version, status "
    "FROM rag_embedding_generations"
)
# Requires migration 009; used only by the physical (generations) store.
_SELECT_EXT = (
    "SELECT generation_id, embedding_model, embedding_dim, embedding_version, "
    "status, mirror_inline, index_ready FROM rag_embedding_generations"
)


async def get_generation(
    conn: Any, generation_id: str, *, extended: bool = False
) -> Optional[Generation]:
    select = _SELECT_EXT if extended else _SELECT
    cur = await conn.execute(f"{select} WHERE generation_id = %s", (generation_id,))
    return _row_to_generation(await cur.fetchone())


async def get_active_generation(
    conn: Any, *, use_cache: bool = True, extended: bool = False
) -> Optional[Generation]:
    cache = _active_cache_ext if extended else _active_cache
    now = time.monotonic()
    if use_cache and now - cache["at"] < _ACTIVE_CACHE_TTL_S:
        return cache["value"]
    select = _SELECT_EXT if extended else _SELECT
    cur = await conn.execute(f"{select} WHERE status = 'ACTIVE'")
    value = _row_to_generation(await cur.fetchone())
    cache.update(at=now, value=value)
    return value


def reset_active_cache() -> None:
    for cache in (_active_cache, _active_cache_ext):
        cache.update(at=0.0, value=None)


async def register_generation(
    conn: Any,
    *,
    generation_id: str,
    embedding_model: str,
    embedding_dim: int,
    embedding_version: int,
    note: Optional[str] = None,
    mirror_inline: bool = False,
) -> None:
    """Register a BUILDING generation.

    ``mirror_inline`` (physical store, needs migration 009) makes the DB trigger
    copy every inline ``rag_chunks.embedding`` with a matching signature into
    this generation, so existing writers keep it populated without changes.
    """
    validate_generation_id(generation_id)
    if not 1 <= int(embedding_dim) <= MAX_INDEXED_DIM:
        raise GenerationError(
            f"embedding_dim must be 1..{MAX_INDEXED_DIM} (halfvec ANN index limit)"
        )
    if mirror_inline:
        await conn.execute(
            """
            INSERT INTO rag_embedding_generations
                (generation_id, embedding_model, embedding_dim, embedding_version,
                 note, mirror_inline)
            VALUES (%s, %s, %s, %s, %s, true)
            """,
            (
                generation_id,
                embedding_model,
                int(embedding_dim),
                int(embedding_version),
                note,
            ),
        )
        return
    await conn.execute(
        """
        INSERT INTO rag_embedding_generations
            (generation_id, embedding_model, embedding_dim, embedding_version, note)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            generation_id,
            embedding_model,
            int(embedding_dim),
            int(embedding_version),
            note,
        ),
    )


async def audit_generation_coverage(
    conn: Any, generation: Generation, *, store: str = STORE_INLINE
) -> Dict[str, Any]:
    """Share of active chunks that are servable under ``generation``.

    ``inline``: chunks whose inline embedding carries the generation's signature.
    ``generations``: chunks with a *fresh* row in ``rag_chunk_embeddings`` (its
    ``content_hash`` matches the chunk's) of the right dimension. Stale rows are
    reported separately and never count as coverage.
    """
    if store == STORE_GENERATIONS:
        cur = await conn.execute(
            """
            SELECT
                count(*) AS total,
                count(e.chunk_id) FILTER (
                    WHERE e.content_hash IS NOT DISTINCT FROM r.content_hash
                      AND vector_dims(e.embedding) = %s
                ) AS matching,
                count(*) FILTER (WHERE e.chunk_id IS NULL) AS missing_embedding,
                count(e.chunk_id) FILTER (
                    WHERE e.content_hash IS DISTINCT FROM r.content_hash
                ) AS stale,
                count(e.chunk_id) FILTER (
                    WHERE vector_dims(e.embedding) <> %s
                ) AS wrong_dim
            FROM rag_chunks r
            LEFT JOIN rag_chunk_embeddings e
                   ON e.chunk_id = r.id AND e.generation_id = %s
            WHERE COALESCE(r.is_active, true) = true
            """,
            (
                generation.embedding_dim,
                generation.embedding_dim,
                generation.generation_id,
            ),
        )
        total, matching, missing, stale, wrong_dim = (
            int(v) for v in await cur.fetchone()
        )
        return {
            "generation_id": generation.generation_id,
            "store": STORE_GENERATIONS,
            "total": total,
            "matching": matching,
            "missing_embedding": missing,
            "stale": stale,
            "wrong_dim": wrong_dim,
            "coverage": (matching / total) if total else 0.0,
        }
    cur = await conn.execute(
        """
        SELECT
            count(*) AS total,
            count(*) FILTER (
                WHERE embedding IS NOT NULL
                  AND embedding_model = %s
                  AND embedding_dim = %s
                  AND embedding_version = %s
            ) AS matching,
            count(*) FILTER (WHERE embedding IS NULL) AS missing_embedding
        FROM rag_chunks
        WHERE COALESCE(is_active, true) = true
        """,
        generation.signature,
    )
    total, matching, missing = await cur.fetchone()
    total, matching, missing = int(total), int(matching), int(missing)
    return {
        "generation_id": generation.generation_id,
        "store": STORE_INLINE,
        "total": total,
        "matching": matching,
        "missing_embedding": missing,
        "coverage": (matching / total) if total else 0.0,
    }


async def _log_event(
    conn: Any,
    generation_id: str,
    action: str,
    previous_id: Optional[str],
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO rag_embedding_generation_events
            (generation_id, action, previous_id, detail)
        VALUES (%s, %s, %s, %s::jsonb)
        """,
        (
            generation_id,
            action,
            previous_id,
            json.dumps(detail, ensure_ascii=False) if detail else None,
        ),
    )


async def activate_generation(
    conn: Any,
    generation_id: str,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    force: bool = False,
    store: str = STORE_INLINE,
) -> Dict[str, Any]:
    """Atomically make ``generation_id`` ACTIVE after a coverage audit.

    The previous ACTIVE generation becomes READY (not RETIRED) so
    :func:`rollback_generation` can restore it. In the physical store the target
    must also have its ANN index built (``force`` overrides, at query-speed risk).
    """
    extended = store == STORE_GENERATIONS
    async with conn.transaction():
        target = await get_generation(conn, generation_id, extended=extended)
        if target is None:
            raise GenerationError(f"unknown generation: {generation_id}")
        if target.status == "ACTIVE":
            return {"generation_id": generation_id, "changed": False}
        if extended and not target.index_ready and not force:
            raise GenerationError(
                f"{generation_id} has no ready ANN index; run build-index first"
            )
        audit = await audit_generation_coverage(conn, target, store=store)
        if audit["coverage"] < min_coverage and not force:
            raise GenerationError(
                f"coverage {audit['coverage']:.4f} < {min_coverage:.4f} "
                f"for {generation_id} ({audit['matching']}/{audit['total']})"
            )
        previous = await get_active_generation(conn, use_cache=False, extended=extended)
        if previous is not None:
            await conn.execute(
                "UPDATE rag_embedding_generations SET status = 'READY' "
                "WHERE generation_id = %s",
                (previous.generation_id,),
            )
        await conn.execute(
            "UPDATE rag_embedding_generations "
            "SET status = 'ACTIVE', activated_at = now() WHERE generation_id = %s",
            (generation_id,),
        )
        await _log_event(
            conn,
            generation_id,
            "ACTIVATE",
            previous.generation_id if previous else None,
            {**audit, "forced": bool(force)},
        )
    reset_active_cache()
    logger.info("[Generations] activated %s (previous=%s)", generation_id, previous)
    return {
        "generation_id": generation_id,
        "previous_id": previous.generation_id if previous else None,
        "changed": True,
        "audit": audit,
    }


async def rollback_generation(
    conn: Any, *, store: str = STORE_INLINE
) -> Dict[str, Any]:
    """Reactivate the generation that was ACTIVE before the last activation."""
    extended = store == STORE_GENERATIONS
    async with conn.transaction():
        cur = await conn.execute("""
            SELECT previous_id FROM rag_embedding_generation_events
            WHERE action = 'ACTIVATE' AND previous_id IS NOT NULL
            ORDER BY id DESC LIMIT 1
            """)
        row = await cur.fetchone()
        if row is None:
            raise GenerationError("no previous generation to roll back to")
        previous_id = str(row[0])
        current = await get_active_generation(conn, use_cache=False, extended=extended)
        if current is not None and current.generation_id == previous_id:
            raise GenerationError(f"{previous_id} is already the active generation")
        target = await get_generation(conn, previous_id, extended=extended)
        if target is None:
            raise GenerationError(f"previous generation missing: {previous_id}")
        audit = await audit_generation_coverage(conn, target, store=store)
        if audit["matching"] == 0:
            raise GenerationError(
                f"cannot roll back: no chunks carry {previous_id}'s embeddings "
                + (
                    "(its rows were dropped)"
                    if extended
                    else "(old embeddings were overwritten in place)"
                )
            )
        if extended and not target.index_ready:
            raise GenerationError(
                f"cannot roll back: {previous_id} has no ready ANN index"
            )
        if current is not None:
            await conn.execute(
                "UPDATE rag_embedding_generations "
                "SET status = 'RETIRED', retired_at = now() WHERE generation_id = %s",
                (current.generation_id,),
            )
        await conn.execute(
            "UPDATE rag_embedding_generations "
            "SET status = 'ACTIVE', activated_at = now() WHERE generation_id = %s",
            (previous_id,),
        )
        await _log_event(
            conn,
            previous_id,
            "ROLLBACK",
            current.generation_id if current else None,
            audit,
        )
    reset_active_cache()
    return {"generation_id": previous_id, "audit": audit}


def generation_filter_sql(generation: Optional[Generation]) -> Tuple[str, list]:
    """WHERE fragment restricting retrieval to the active generation."""
    if generation is None:
        return "", []
    return (
        "embedding_model = %s AND embedding_dim = %s AND embedding_version = %s",
        list(generation.signature),
    )


# ─── physical store operations (migration 009) ───────────────────────────────
def _vector_literal(values: Any) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


async def _require_generation(conn: Any, generation_id: str) -> Generation:
    validate_generation_id(generation_id)
    generation = await get_generation(conn, generation_id, extended=True)
    if generation is None:
        raise GenerationError(f"unknown generation: {generation_id}")
    return generation


async def write_embeddings(
    conn: Any,
    generation_id: str,
    rows: Any,
) -> int:
    """Upsert ``(chunk_id, embedding, content_hash)`` rows into a generation.

    This is the writer contract for a *new* embedding model (the crawler, or a
    re-embed job). Dimension is validated against the generation so a wrong-model
    vector can never enter it. Returns the number of rows written.
    """
    generation = await _require_generation(conn, generation_id)
    written = 0
    for chunk_id, embedding, content_hash in rows:
        if len(embedding) != generation.embedding_dim:
            raise GenerationError(
                f"chunk {chunk_id}: embedding has {len(embedding)} dims, "
                f"{generation_id} expects {generation.embedding_dim}"
            )
        await conn.execute(
            """
            INSERT INTO rag_chunk_embeddings
                (generation_id, chunk_id, embedding, content_hash)
            VALUES (%s, %s, %s::vector, %s)
            ON CONFLICT (generation_id, chunk_id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    content_hash = EXCLUDED.content_hash,
                    created_at = now()
            """,
            (generation_id, int(chunk_id), _vector_literal(embedding), content_hash),
        )
        written += 1
    return written


async def backfill_from_inline(
    conn: Any, generation_id: str, *, batch_size: int = 5000
) -> Dict[str, int]:
    """Copy inline embeddings whose signature matches the generation.

    Adopting the physical store needs no re-embedding: the current model's
    vectors already sit in ``rag_chunks.embedding``. Keyset-batched (autocommit
    per batch) so it neither holds long locks nor needs one giant transaction.
    Rows already present with a matching hash are left alone; changed content is
    refreshed.
    """
    generation = await _require_generation(conn, generation_id)
    last_id = 0
    copied = batches = 0
    while True:
        cur = await conn.execute(
            """
            WITH batch AS (
                SELECT id, embedding, content_hash
                FROM rag_chunks
                WHERE embedding IS NOT NULL
                  AND embedding_model = %s
                  AND embedding_dim = %s
                  AND embedding_version = %s
                  AND id > %s
                ORDER BY id
                LIMIT %s
            ), ins AS (
                INSERT INTO rag_chunk_embeddings
                    (generation_id, chunk_id, embedding, content_hash)
                SELECT %s, id, embedding, content_hash FROM batch
                ON CONFLICT (generation_id, chunk_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        content_hash = EXCLUDED.content_hash,
                        created_at = now()
                    WHERE rag_chunk_embeddings.content_hash
                          IS DISTINCT FROM EXCLUDED.content_hash
                RETURNING 1
            )
            SELECT (SELECT max(id) FROM batch), (SELECT count(*) FROM ins)
            """,
            (*generation.signature, last_id, int(batch_size), generation_id),
        )
        max_id, inserted = await cur.fetchone()
        if max_id is None:
            break
        last_id = int(max_id)
        copied += int(inserted)
        batches += 1
    return {"copied": copied, "batches": batches}


async def _index_state(conn: Any, name: str) -> Optional[Tuple[bool, bool]]:
    cur = await conn.execute(
        """
        SELECT i.indisvalid, i.indisready
        FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
        WHERE c.relname = %s
        """,
        (name,),
    )
    row = await cur.fetchone()
    return None if row is None else (bool(row[0]), bool(row[1]))


async def build_generation_index(conn: Any, generation_id: str) -> Dict[str, Any]:
    """Build the generation's partial HNSW index (halfvec expression).

    Must run on an autocommit connection (``CREATE INDEX CONCURRENTLY``). A
    leftover INVALID index from a failed earlier build is dropped and rebuilt.
    The expression and predicate here are exactly what retrieval queries use, so
    the planner can match the partial index.
    """
    generation = await _require_generation(conn, generation_id)
    name = index_name_for(generation_id)
    dim = int(generation.embedding_dim)
    state = await _index_state(conn, name)
    if state is not None and state != (True, True):
        await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
        state = None
    if state is None:
        await conn.execute(
            f"CREATE INDEX CONCURRENTLY {name} ON rag_chunk_embeddings "
            f"USING hnsw ((embedding::halfvec({dim})) halfvec_cosine_ops) "
            f"WITH (m = 16, ef_construction = 64) "
            f"WHERE generation_id = '{generation_id}'"
        )
        state = await _index_state(conn, name)
    ready = state == (True, True)
    await conn.execute(
        "UPDATE rag_embedding_generations SET index_name = %s, index_ready = %s "
        "WHERE generation_id = %s",
        (name, ready, generation_id),
    )
    if not ready:
        raise GenerationError(f"index {name} is not valid after build")
    reset_active_cache()
    return {"generation_id": generation_id, "index": name, "ready": True}


async def drop_generation_data(conn: Any, generation_id: str) -> Dict[str, Any]:
    """Delete a generation's embeddings and index to reclaim space.

    Refused for the ACTIVE generation and for the current rollback target (the
    generation the last activation replaced), so a rollback is never destroyed
    by housekeeping.
    """
    generation = await _require_generation(conn, generation_id)
    if generation.status == "ACTIVE":
        raise GenerationError(f"{generation_id} is ACTIVE; cannot drop")
    cur = await conn.execute("""
        SELECT previous_id FROM rag_embedding_generation_events
        WHERE action = 'ACTIVATE' AND previous_id IS NOT NULL
        ORDER BY id DESC LIMIT 1
        """)
    row = await cur.fetchone()
    if row is not None and str(row[0]) == generation_id:
        raise GenerationError(
            f"{generation_id} is the current rollback target; activate another "
            "generation or roll back first"
        )
    name = index_name_for(generation_id)
    await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
    cur = await conn.execute(
        "DELETE FROM rag_chunk_embeddings WHERE generation_id = %s", (generation_id,)
    )
    deleted = int(getattr(cur, "rowcount", 0) or 0)
    await conn.execute(
        "UPDATE rag_embedding_generations "
        "SET status = 'RETIRED', retired_at = now(), index_ready = false "
        "WHERE generation_id = %s",
        (generation_id,),
    )
    await _log_event(conn, generation_id, "DROP", None, {"deleted_rows": deleted})
    reset_active_cache()
    return {"generation_id": generation_id, "deleted_rows": deleted}
