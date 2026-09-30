"""Oracle-backed baseball read boundary.

The crawler (KBO_playwright) owns the baseball tables in both PostgreSQL and
Oracle ADB — same columns, same shapes, different SQL dialect. So unlike RAG
(where the Oracle path uses a genuinely different storage model — native
VECTOR vs. pgvector — see ``oracle_rag.py``), baseball reads only need their
*SQL text* translated: array-membership binds, ``LIMIT``, and date truncation
differ between psycopg and oracledb, but the tables and business logic do
not. That is why this module holds a small dialect-aware query builder
instead of a parallel set of query functions — duplicating ~700 lines of
``GameQueryTool`` per dialect would drift the two copies apart.
"""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Sequence

from .oracle_rag import (
    OracleRagConnection,
    _oracle_connect_args,
    is_oracle_rag_connection,
)

try:
    import oracledb
except ModuleNotFoundError:  # pragma: no cover
    oracledb = None  # type: ignore[assignment]

# Re-exported so callers only need one import for "is this connection Oracle".
# The check itself (``connection.backend == "oracle"``) isn't RAG-specific;
# ``oracle_rag.py`` just happens to be where the marker was defined first.
is_oracle_baseball_connection = is_oracle_rag_connection


class OracleBaseballPool:
    """Lazy async connection pool for the Oracle baseball datasource.

    Mirrors ``OracleRagPool``'s lifecycle shape exactly (same ``open``/
    ``connection``/``close``/``get_stats`` contract) so ``deps.py`` can treat
    both the RAG and baseball Oracle pools identically. Kept as a separate
    class rather than a shared one so the two domains can size and fail
    independently — a baseball ADB outage should not touch RAG capacity.
    """

    backend = "oracle"

    def __init__(self, conninfo: str, *, max_size: int = 6) -> None:
        self.conninfo = conninfo
        self.max_size = max(1, int(max_size))
        self._pool: Any = None
        self._open_lock = asyncio.Lock()

    async def open(self, *, wait: bool = True, timeout: float | None = None) -> None:
        del wait
        if self._pool is not None:
            return
        if oracledb is None:
            raise RuntimeError(
                "oracledb is required when AI_BASEBALL_DB_URL uses Oracle"
            )
        async with self._open_lock:
            if self._pool is not None:
                return
            coroutine = oracledb.create_pool_async(
                **_oracle_connect_args(self.conninfo),
                min=1,
                max=self.max_size,
                increment=1,
            )
            self._pool = (
                await asyncio.wait_for(coroutine, timeout=timeout)
                if timeout is not None
                else await coroutine
            )

    @asynccontextmanager
    async def connection(
        self, timeout: float | None = None
    ) -> AsyncIterator[OracleRagConnection]:
        if self._pool is None:
            await self.open(timeout=timeout)
        raw_connection = await self._pool.acquire()
        try:
            yield OracleRagConnection(raw_connection)
        finally:
            result = self._pool.release(raw_connection)
            if inspect.isawaitable(result):
                await result

    async def close(self) -> None:
        pool = self._pool
        self._pool = None
        if pool is not None:
            result = pool.close()
            if inspect.isawaitable(result):
                await result

    def get_stats(self) -> dict[str, Any]:
        pool = self._pool
        if pool is None:
            return {"pool_open": False}
        get_stats = getattr(pool, "get_stats", None)
        return get_stats() if callable(get_stats) else {"pool_open": True}


class QueryDialect:
    """Accumulates one query's placeholders/params in the active SQL dialect.

    psycopg binds positional ``%s`` against a list; oracledb binds named
    ``:name`` against a dict. ``= ANY(%s)`` (a single array bind) has no
    Oracle equivalent, so on Oracle it expands to ``IN (:v1, :v2, ...)`` —
    one bind per value.

    One instance per logical query (not per ``GameQueryTool`` call — a
    method that runs several queries builds a fresh dialect for each), so
    params never leak between unrelated ``cursor.execute`` calls.
    """

    def __init__(self, is_oracle: bool) -> None:
        self.is_oracle = is_oracle
        self.params: list[Any] = []
        self.named_params: dict[str, Any] = {}
        self._counter = 0

    def _name(self, hint: str) -> str:
        self._counter += 1
        return f"{hint}{self._counter}"

    def bind(self, value: Any, hint: str = "p") -> str:
        """Register one scalar value; return its placeholder token."""
        if self.is_oracle:
            name = self._name(hint)
            self.named_params[name] = value
            return f":{name}"
        self.params.append(value)
        return "%s"

    def in_any(self, column: str, values: Sequence[Any], hint: str = "v") -> str:
        """Return a WHERE fragment matching ``column`` against ``values``."""
        values = list(values)
        if not values:
            # Same as PostgreSQL's `= ANY('{}')`: matches nothing.
            return "1 = 0"
        if self.is_oracle:
            tokens = [self.bind(value, hint) for value in values]
            return f"{column} IN ({', '.join(tokens)})"
        return f"{column} = ANY({self.bind(values, hint)})"

    def bind_date(self, value: Any, hint: str = "d") -> str:
        """Register a 'YYYY-MM-DD' string for comparison against a DATE
        column; return the token to embed in the SQL fragment.

        Oracle has no implicit string->DATE conversion for bind variables —
        a bare bind against a DATE column raises ORA-01861 ('literal does
        not match format string') unless the session's NLS_DATE_FORMAT
        happens to match. This only surfaced against a live ADB; the
        fake-cursor unit tests never bind against a real DATE column so they
        can't catch it. PostgreSQL accepts the ISO string directly, so this
        is a no-op wrapper there — psycopg does the cast.
        """
        token = self.bind(value, hint)
        if self.is_oracle:
            return f"TO_DATE({token}, 'YYYY-MM-DD')"
        return token

    def date_eq(self, column: str, value: Any, hint: str = "d") -> str:
        """Return a WHERE fragment matching the date part of ``column``."""
        if self.is_oracle:
            return f"TRUNC({column}) = {self.bind_date(value, hint)}"
        return f"DATE({column}) = {self.bind(value, hint)}"

    def limit(self, n: Any, hint: str = "lim") -> str:
        """Return a trailing row-limit clause for ``n`` rows."""
        if self.is_oracle:
            return f"FETCH FIRST {self.bind(n, hint)} ROWS ONLY"
        return f"LIMIT {self.bind(n, hint)}"

    def execute_args(self) -> Any:
        """Return the argument shape ``cursor.execute()`` expects."""
        return self.named_params if self.is_oracle else self.params
