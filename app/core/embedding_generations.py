"""Embedding generation registry: audited activation, rollback, and the
retrieval-side signature gate that keeps old/new embeddings from mixing.

Transactions: every mutation runs in ``conn.transaction()`` so the ACTIVE
pointer flips atomically (the partial unique index guarantees a single ACTIVE
row even under concurrent operators).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MIN_COVERAGE = 0.995
_ACTIVE_CACHE_TTL_S = 30.0
_active_cache: Dict[str, Any] = {"at": 0.0, "value": None}


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Generation:
    generation_id: str
    embedding_model: str
    embedding_dim: int
    embedding_version: int
    status: str = "BUILDING"

    @property
    def signature(self) -> Tuple[str, int, int]:
        return (self.embedding_model, self.embedding_dim, self.embedding_version)


def _row_to_generation(row: Any) -> Optional[Generation]:
    if row is None:
        return None
    return Generation(str(row[0]), str(row[1]), int(row[2]), int(row[3]), str(row[4]))


_SELECT = (
    "SELECT generation_id, embedding_model, embedding_dim, embedding_version, status "
    "FROM rag_embedding_generations"
)


async def get_generation(conn: Any, generation_id: str) -> Optional[Generation]:
    cur = await conn.execute(f"{_SELECT} WHERE generation_id = %s", (generation_id,))
    return _row_to_generation(await cur.fetchone())


async def get_active_generation(
    conn: Any, *, use_cache: bool = True
) -> Optional[Generation]:
    now = time.monotonic()
    if use_cache and now - _active_cache["at"] < _ACTIVE_CACHE_TTL_S:
        return _active_cache["value"]
    cur = await conn.execute(f"{_SELECT} WHERE status = 'ACTIVE'")
    value = _row_to_generation(await cur.fetchone())
    _active_cache.update(at=now, value=value)
    return value


def reset_active_cache() -> None:
    _active_cache.update(at=0.0, value=None)


async def register_generation(
    conn: Any,
    *,
    generation_id: str,
    embedding_model: str,
    embedding_dim: int,
    embedding_version: int,
    note: Optional[str] = None,
) -> None:
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
    conn: Any, generation: Generation
) -> Dict[str, Any]:
    """Share of active chunks embedded with ``generation``'s signature."""
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
) -> Dict[str, Any]:
    """Atomically make ``generation_id`` ACTIVE after a coverage audit.

    The previous ACTIVE generation becomes READY (not RETIRED) so
    :func:`rollback_generation` can restore it.
    """
    async with conn.transaction():
        target = await get_generation(conn, generation_id)
        if target is None:
            raise GenerationError(f"unknown generation: {generation_id}")
        if target.status == "ACTIVE":
            return {"generation_id": generation_id, "changed": False}
        audit = await audit_generation_coverage(conn, target)
        if audit["coverage"] < min_coverage and not force:
            raise GenerationError(
                f"coverage {audit['coverage']:.4f} < {min_coverage:.4f} "
                f"for {generation_id} ({audit['matching']}/{audit['total']})"
            )
        previous = await get_active_generation(conn, use_cache=False)
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


async def rollback_generation(conn: Any) -> Dict[str, Any]:
    """Reactivate the generation that was ACTIVE before the last activation."""
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
        current = await get_active_generation(conn, use_cache=False)
        target = await get_generation(conn, previous_id)
        if target is None:
            raise GenerationError(f"previous generation missing: {previous_id}")
        audit = await audit_generation_coverage(conn, target)
        if audit["matching"] == 0:
            raise GenerationError(
                f"cannot roll back: no chunks carry {previous_id}'s signature "
                "(old embeddings were overwritten in place)"
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
