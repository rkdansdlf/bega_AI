"""쿼리 임베딩 캐시 백엔드.

기본은 in-memory LRU. ``EMBEDDING_CACHE_BACKEND=redis`` 환경변수로 Redis 백엔드
활성화. Redis 백엔드는 다중 워커 환경에서 캐시 공유를 가능하게 한다.

Redis 패키지가 설치되지 않았거나 연결이 실패하면 자동으로 in-memory로 폴백한다.
캐시 실패가 요청 자체를 실패시키지 않도록 모든 호출은 try/except로 보호된다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import OrderedDict
from typing import List, Optional, Protocol

logger = logging.getLogger(__name__)


class EmbeddingCacheBackend(Protocol):
    async def get(self, key: str) -> Optional[List[float]]: ...

    async def set(self, key: str, embedding: List[float]) -> None: ...


class InMemoryLRUBackend:
    """프로세스 로컬 OrderedDict LRU 캐시."""

    def __init__(self, max_size: int) -> None:
        self._max_size = max(0, int(max_size))
        self._store: "OrderedDict[str, List[float]]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[List[float]]:
        if self._max_size <= 0:
            return None
        async with self._lock:
            value = self._store.get(key)
            if value is None:
                return None
            self._store.move_to_end(key)
            return value

    async def set(self, key: str, embedding: List[float]) -> None:
        if self._max_size <= 0:
            return
        async with self._lock:
            self._store[key] = embedding
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)


class RedisEmbeddingBackend:
    """Redis 기반 임베딩 캐시. ``redis.asyncio.Redis`` 클라이언트 사용."""

    def __init__(self, client, ttl_seconds: int, key_prefix: str = "embed") -> None:
        self._client = client
        self._ttl = max(1, int(ttl_seconds))
        self._prefix = key_prefix

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    async def get(self, key: str) -> Optional[List[float]]:
        try:
            raw = await self._client.get(self._full_key(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[EmbedCache] Redis GET failed key=%s err=%s", key, exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError) as exc:
            logger.warning("[EmbedCache] Redis decode failed key=%s err=%s", key, exc)
            return None

    async def set(self, key: str, embedding: List[float]) -> None:
        try:
            payload = json.dumps(embedding)
            await self._client.set(self._full_key(key), payload, ex=self._ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[EmbedCache] Redis SET failed key=%s err=%s", key, exc)


_backend_instance: Optional[EmbeddingCacheBackend] = None
_backend_lock = asyncio.Lock()


def _build_redis_backend() -> Optional[EmbeddingCacheBackend]:
    url = os.getenv("EMBEDDING_CACHE_REDIS_URL") or os.getenv("REDIS_URL")
    if not url:
        logger.warning(
            "[EmbedCache] EMBEDDING_CACHE_BACKEND=redis but no REDIS_URL/EMBEDDING_CACHE_REDIS_URL "
            "configured — falling back to in-memory backend"
        )
        return None
    try:
        from redis import asyncio as redis_asyncio  # type: ignore
    except ImportError:
        logger.warning(
            "[EmbedCache] redis package not installed — falling back to in-memory backend"
        )
        return None
    try:
        client = redis_asyncio.from_url(url, decode_responses=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[EmbedCache] Redis client init failed: %s — using in-memory", exc)
        return None
    ttl = int(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", "86400"))
    logger.info("[EmbedCache] Redis backend ready ttl=%ds prefix=embed", ttl)
    return RedisEmbeddingBackend(client, ttl_seconds=ttl)


async def get_backend() -> EmbeddingCacheBackend:
    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance
    async with _backend_lock:
        if _backend_instance is not None:
            return _backend_instance
        backend_name = (os.getenv("EMBEDDING_CACHE_BACKEND") or "memory").strip().lower()
        backend: Optional[EmbeddingCacheBackend] = None
        if backend_name == "redis":
            backend = _build_redis_backend()
        if backend is None:
            max_size = int(os.getenv("EMBED_QUERY_CACHE_MAX", "2048"))
            backend = InMemoryLRUBackend(max_size=max_size)
            logger.info("[EmbedCache] In-memory backend ready max=%d", max_size)
        _backend_instance = backend
        return backend


def reset_backend_for_tests() -> None:
    """테스트 격리용. 백엔드 싱글톤 초기화."""
    global _backend_instance
    _backend_instance = None
