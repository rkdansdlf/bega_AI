from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Callable, Iterator, TypeVar

import psycopg

T = TypeVar("T")


@contextmanager
def connection_scope(
    connection: psycopg.Connection | None,
    *,
    force_fresh: bool = False,
) -> Iterator[psycopg.Connection]:
    conn = connection
    if (
        not force_fresh
        and conn is not None
        and not bool(getattr(conn, "closed", False))
    ):
        yield conn
        return

    from ..deps import get_connection_pool

    with get_connection_pool().connection() as pooled_conn:
        yield pooled_conn


def run_with_fresh_connection_retry(
    *,
    connection: psycopg.Connection | None,
    operation: Callable[[psycopg.Connection], T],
    logger: logging.Logger,
    retry_warning_message: str,
) -> T:
    try:
        with connection_scope(connection) as conn:
            return operation(conn)
    except Exception as exc:
        if "connection is closed" not in str(exc).lower():
            raise
        logger.warning(retry_warning_message, exc)
        with connection_scope(connection, force_fresh=True) as conn:
            return operation(conn)
