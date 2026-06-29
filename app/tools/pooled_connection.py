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
) -> AsyncIterator[psycopg.AsyncConnection]:
    conn = connection
    if (
        not force_fresh
        and conn is not None
        and not bool(getattr(conn, "closed", False))
    ):
        yield conn
        return

    from ..deps import get_connection_pool

    async with get_connection_pool().connection() as pooled_conn:
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
