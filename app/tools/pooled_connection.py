from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import Awaitable, Callable, AsyncIterator, TypeVar

import psycopg

T = TypeVar("T")


@asynccontextmanager
async def connection_scope(
    connection: psycopg.AsyncConnection | None,
    *,
    force_fresh: bool = False,
    domain: str = "rag",
) -> AsyncIterator[psycopg.AsyncConnection]:
    """풀에서 커넥션을 빌린다.

    `domain="baseball"` 은 야구 테이블을 읽는 쿼리에만 쓴다. 기본값이 "rag" 이므로
    태깅을 빠뜨린 호출부는 기존과 같은 풀을 쓴다 — AI_BASEBALL_DB_URL 을 설정하기
    전까지는 두 풀이 같은 DB 를 가리키므로 어느 쪽이든 동작이 같다.
    """
    conn = connection
    if (
        not force_fresh
        and conn is not None
        and not bool(getattr(conn, "closed", False))
    ):
        yield conn
        return

    from ..deps import get_baseball_connection_pool, get_connection_pool

    pool = (
        get_baseball_connection_pool() if domain == "baseball" else get_connection_pool()
    )
    async with pool.connection() as pooled_conn:
        yield pooled_conn


async def run_with_fresh_connection_retry(
    *,
    connection: psycopg.AsyncConnection | None,
    operation: Callable[[psycopg.AsyncConnection], Awaitable[T]],
    logger: logging.Logger,
    retry_warning_message: str,
) -> T:
    try:
        async with connection_scope(connection) as conn:
            return await operation(conn)
    except Exception as exc:
        if "connection is closed" not in str(exc).lower():
            raise
        logger.warning(retry_warning_message, exc)
        async with connection_scope(connection, force_fresh=True) as conn:
            return await operation(conn)
