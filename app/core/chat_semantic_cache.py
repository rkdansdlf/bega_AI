"""Semantic response cache for chat answers.

This cache is a best-effort companion to the exact ``chat_response_cache``.
Lookup failures are treated as misses so semantic caching never blocks answer
generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Sequence

from app.config import Settings
from app.observability.metrics import (
    AI_SEMANTIC_RESPONSE_CACHE_SHADOW_TOTAL,
    AI_SEMANTIC_RESPONSE_CACHE_TOTAL,
)

from .chat_cache_key import get_ttl_seconds
from .embeddings import _embed_signature
from .retrieval import _vector_literal

logger = logging.getLogger(__name__)

FILTERS_HASH_SCHEMA_VERSION = "chat_semantic_filters_v1"


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_semantic_response_cache (
    cache_key           VARCHAR(64)  PRIMARY KEY,
    question_text       TEXT         NOT NULL,
    question_embedding  VECTOR(256)  NOT NULL,
    filters_hash        VARCHAR(64)  NOT NULL,
    filters_json        JSONB,
    intent              VARCHAR(50),
    source_tier         VARCHAR(50),
    response_text       TEXT         NOT NULL,
    model_name          VARCHAR(100),
    embedding_signature VARCHAR(180) NOT NULL,
    hit_count           INTEGER      NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at          TIMESTAMPTZ  NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_semantic_cache_lookup
    ON chat_semantic_response_cache(filters_hash, embedding_signature, expires_at);
CREATE INDEX IF NOT EXISTS idx_chat_semantic_cache_created_at
    ON chat_semantic_response_cache(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_semantic_cache_expires_at
    ON chat_semantic_response_cache(expires_at);
ALTER TABLE chat_semantic_response_cache
    ADD COLUMN IF NOT EXISTS source_tier VARCHAR(50);
"""

CREATE_VECTOR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chat_semantic_cache_embedding_hnsw
    ON chat_semantic_response_cache
    USING hnsw (question_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""


def _record_semantic_cache(operation: str, result: str) -> None:
    try:
        AI_SEMANTIC_RESPONSE_CACHE_TOTAL.labels(
            operation=operation, result=result
        ).inc()
    except Exception:  # noqa: BLE001
        pass


def record_semantic_shadow_decision(route: str, result: str) -> None:
    try:
        AI_SEMANTIC_RESPONSE_CACHE_SHADOW_TOTAL.labels(route=route, result=result).inc()
    except Exception:  # noqa: BLE001
        pass


def _normalize_filters(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not filters:
        return {}
    return {
        str(key): value
        for key, value in sorted(filters.items())
        if value is not None and value != "" and value != []
    }


def _build_filters_hash(filters: Optional[Dict[str, Any]]) -> str:
    payload = {
        "schema": FILTERS_HASH_SCHEMA_VERSION,
        "filters": _normalize_filters(filters),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _filters_json_payload(filters: Optional[Dict[str, Any]]) -> Optional[str]:
    normalized = _normalize_filters(filters)
    if not normalized:
        return None
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)


def _coerce_embedding(embedding: Sequence[float]) -> list[float]:
    return [float(value) for value in embedding]


def _embedding_signature(settings: Settings) -> str:
    return _embed_signature(settings)


async def _get_semantic_sync(
    conn,
    *,
    question_embedding: Sequence[float],
    filters_json: Optional[Dict[str, Any]],
    settings: Settings,
    threshold: float,
    limit: int,
) -> Optional[Dict[str, Any]]:
    embedding = _coerce_embedding(question_embedding)
    if not embedding:
        return None

    vector_str = _vector_literal(embedding)
    filters_hash = _build_filters_hash(filters_json)
    embedding_signature = _embedding_signature(settings)
    effective_limit = max(1, min(int(limit), 10))
    effective_threshold = max(0.0, min(float(threshold), 1.0))
    hnsw_ef_search = max(
        0, int(getattr(settings, "chat_semantic_cache_hnsw_ef_search", 0) or 0)
    )
    if hnsw_ef_search:
        await conn.execute(f"SET hnsw.ef_search = {hnsw_ef_search}")

    cur = await conn.execute(
        """
        WITH candidates AS (
            SELECT cache_key,
                   question_text,
                   filters_json,
                   response_text,
                   intent,
                   model_name,
                   source_tier,
                   hit_count,
                   expires_at,
                   created_at,
                   (1 - (question_embedding <=> %s::vector)) AS similarity
            FROM chat_semantic_response_cache
            WHERE expires_at > now()
              AND filters_hash = %s
              AND embedding_signature = %s
        )
        SELECT cache_key, question_text, filters_json, response_text, intent,
               model_name, source_tier, hit_count, expires_at, similarity
        FROM candidates
        WHERE similarity >= %s
        ORDER BY similarity DESC, created_at DESC
        LIMIT %s
        """,
        (
            vector_str,
            filters_hash,
            embedding_signature,
            effective_threshold,
            effective_limit,
        ),
    )
    row = await cur.fetchone()
    if row is None:
        return None

    (
        cache_key,
        question_text,
        filters_json_row,
        response_text,
        intent,
        model_name,
        source_tier,
        hit_count,
        expires_at,
        similarity,
    ) = row
    return {
        "cache_key": cache_key,
        "question_text": question_text,
        "filters_json": filters_json_row,
        "response_text": response_text,
        "intent": intent,
        "model_name": model_name,
        "source_tier": source_tier,
        "hit_count": hit_count,
        "expires_at": expires_at,
        "similarity": float(similarity or 0.0),
    }


async def _save_semantic_sync(
    conn,
    *,
    cache_key: str,
    question_text: str,
    question_embedding: Sequence[float],
    filters_json: Optional[Dict[str, Any]],
    intent: Optional[str],
    source_tier: Optional[str],
    response_text: str,
    model_name: Optional[str],
    settings: Settings,
) -> None:
    embedding = _coerce_embedding(question_embedding)
    if not embedding:
        return

    ttl_secs = get_ttl_seconds(intent)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_secs)
    vector_str = _vector_literal(embedding)
    filters_hash = _build_filters_hash(filters_json)
    filters_serialized = _filters_json_payload(filters_json)
    embedding_signature = _embedding_signature(settings)

    await conn.execute(
        """
        INSERT INTO chat_semantic_response_cache
            (cache_key, question_text, question_embedding, filters_hash,
             filters_json, intent, source_tier, response_text, model_name,
             embedding_signature, expires_at)
        VALUES
            (%s, %s, %s::vector, %s, %s::jsonb, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (cache_key) DO UPDATE
        SET
            question_text = EXCLUDED.question_text,
            question_embedding = EXCLUDED.question_embedding,
            filters_hash = EXCLUDED.filters_hash,
            filters_json = EXCLUDED.filters_json,
            intent = EXCLUDED.intent,
            source_tier = EXCLUDED.source_tier,
            response_text = EXCLUDED.response_text,
            model_name = EXCLUDED.model_name,
            embedding_signature = EXCLUDED.embedding_signature,
            expires_at = EXCLUDED.expires_at,
            hit_count = 0,
            created_at = now()
        """,
        (
            cache_key,
            question_text,
            vector_str,
            filters_hash,
            filters_serialized,
            intent,
            source_tier,
            response_text,
            model_name,
            embedding_signature,
            expires_at,
        ),
    )


async def _update_semantic_hit_count_sync(conn, cache_key: str) -> None:
    await conn.execute(
        """
        UPDATE chat_semantic_response_cache
        SET hit_count = hit_count + 1
        WHERE cache_key = %s
        """,
        (cache_key,),
    )


async def _cleanup_semantic_sync(conn) -> int:
    result = await conn.execute(
        "DELETE FROM chat_semantic_response_cache WHERE expires_at <= now()"
    )
    deleted: int = getattr(result, "rowcount", 0) or 0
    if deleted:
        logger.info("[ChatSemanticCache] Cleaned up %d expired entries", deleted)
    return deleted


async def _delete_semantic_by_key_sync(conn, cache_key: str) -> int:
    result = await conn.execute(
        "DELETE FROM chat_semantic_response_cache WHERE cache_key = %s",
        (cache_key,),
    )
    return getattr(result, "rowcount", 0) or 0


async def get_semantic_cached_response(
    conn,
    *,
    question_embedding: Sequence[float],
    filters_json: Optional[Dict[str, Any]],
    settings: Settings,
    threshold: float,
    limit: int,
) -> Optional[Dict[str, Any]]:
    try:
        result = await _get_semantic_sync(
            conn,
            question_embedding=question_embedding,
            filters_json=filters_json,
            settings=settings,
            threshold=threshold,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ChatSemanticCache] lookup failed: %s", exc)
        _record_semantic_cache("lookup", "error")
        return None
    _record_semantic_cache("lookup", "hit" if result is not None else "miss")
    return result


async def save_semantic_cache(
    conn,
    *,
    cache_key: str,
    question_text: str,
    question_embedding: Sequence[float],
    filters_json: Optional[Dict[str, Any]],
    intent: Optional[str],
    source_tier: Optional[str],
    response_text: str,
    model_name: Optional[str],
    settings: Settings,
) -> None:
    try:
        await _save_semantic_sync(
            conn,
            cache_key=cache_key,
            question_text=question_text,
            question_embedding=question_embedding,
            filters_json=filters_json,
            intent=intent,
            source_tier=source_tier,
            response_text=response_text,
            model_name=model_name,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[ChatSemanticCache] save failed for key %.8s: %s", cache_key, exc
        )
        _record_semantic_cache("store", "error")
        return
    _record_semantic_cache("store", "ok")


async def update_semantic_hit_count(conn, cache_key: str) -> None:
    try:
        await _update_semantic_hit_count_sync(conn, cache_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ChatSemanticCache] hit_count update failed: %s", exc)


async def cleanup_expired(conn) -> int:
    return await _cleanup_semantic_sync(conn)


async def delete_semantic_by_key(conn, cache_key: str) -> int:
    try:
        return await _delete_semantic_by_key_sync(conn, cache_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[ChatSemanticCache] delete failed for key %.8s: %s", cache_key, exc
        )
        return 0
