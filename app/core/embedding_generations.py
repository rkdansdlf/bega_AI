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
_ACTIVE_CACHE_TTL_S = 5.0  # default; RAG_GENERATION_CACHE_TTL_S overrides
DEFAULT_DROP_GRACE_S = 60.0
# "at" is None until a lookup has happened. Never encode "never" as 0.0:
# time.monotonic() counts from boot, so on a host up for less than the TTL a
# zero timestamp looks fresh and would serve a cached None.
_active_cache: Dict[str, Any] = {"at": None, "value": None}
_active_cache_ext: Dict[str, Any] = {"at": None, "value": None}

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
    conn: Any,
    *,
    use_cache: bool = True,
    extended: bool = False,
    ttl: Optional[float] = None,
) -> Optional[Generation]:
    """ACTIVE generation, cached per process for ``ttl`` seconds (0 = never)."""
    cache = _active_cache_ext if extended else _active_cache
    now = time.monotonic()
    max_age = _ACTIVE_CACHE_TTL_S if ttl is None else max(0.0, float(ttl))
    cached_at = cache["at"]
    if (
        use_cache
        and max_age > 0
        and cached_at is not None
        and now - cached_at < max_age
    ):
        return cache["value"]
    select = _SELECT_EXT if extended else _SELECT
    cur = await conn.execute(f"{select} WHERE status = 'ACTIVE'")
    value = _row_to_generation(await cur.fetchone())
    cache.update(at=now, value=value)
    return value


def reset_active_cache() -> None:
    for cache in (_active_cache, _active_cache_ext):
        cache.update(at=None, value=None)


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


def _propagation(cache_ttl_s: Optional[float]) -> Dict[str, Any]:
    """How long until *other* processes stop reading the previous generation.

    The activating process invalidates its own cache immediately; every other
    process keeps its cached lookup for at most ``RAG_GENERATION_CACHE_TTL_S``.
    Requests already in flight finish on the generation they started with.
    """
    ttl = _ACTIVE_CACHE_TTL_S if cache_ttl_s is None else max(0.0, float(cache_ttl_s))
    return {"max_propagation_seconds": ttl}


async def activate_generation(
    conn: Any,
    generation_id: str,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    force: bool = False,
    store: str = STORE_INLINE,
    cache_ttl_s: Optional[float] = None,
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
        **_propagation(cache_ttl_s),
    }


async def rollback_generation(
    conn: Any, *, store: str = STORE_INLINE, cache_ttl_s: Optional[float] = None
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
    return {
        "generation_id": previous_id,
        "audit": audit,
        **_propagation(cache_ttl_s),
    }


def generation_filter_sql(generation: Optional[Generation]) -> Tuple[str, list]:
    """WHERE fragment restricting retrieval to the active generation."""
    if generation is None:
        return "", []
    return (
        "embedding_model = %s AND embedding_dim = %s AND embedding_version = %s",
        list(generation.signature),
    )


# ─── physical store operations (migration 009) ───────────────────────────────
