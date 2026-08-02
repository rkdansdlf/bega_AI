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
    domain: str = "cache",
) -> AsyncIterator[psycopg.AsyncConnection]:
    """풀에서 커넥션을 빌린다.

    도메인은 셋이다.
      - "cache"    기본값. AI 자체 테이블(chat/coach 캐시, 인제스트 상태).
                   요청마다 쓰기가 발생하므로 가장 가까운 DB 에 둔다.
      - "rag"      rag_chunks 조회. 별도 호스트로 뺄 수 있다.
      - "baseball" 야구 테이블 조회.

    기본값이 "cache" 인 이유는 태깅을 빠뜨린 호출부가 원격 DB 로 새지 않게
    하기 위해서다. 세 URL 이 모두 같은 값으로 폴백되는 동안에는 어느 쪽이든
    동작이 같으므로, 분리 전 배포는 무해하다.
    """
    conn = connection
    if (
        not force_fresh
        and conn is not None
        and not bool(getattr(conn, "closed", False))
    ):
        yield conn
        return

    from ..deps import (
        get_baseball_connection_pool,
        get_connection_pool,
        get_rag_connection_pool,
    )

    if domain == "baseball":
        pool = get_baseball_connection_pool()
    elif domain == "rag":
        pool = get_rag_connection_pool()
    else:
        pool = get_connection_pool()
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
