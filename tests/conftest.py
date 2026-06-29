import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from app.config import get_settings

# Pillow/PIL is an optional runtime dependency (vision endpoints only).
# Provide a lightweight stub so tests can import app.routers.vision
# in environments where Pillow is not installed.
if "PIL" not in sys.modules:
    _pil_stub = ModuleType("PIL")
    _pil_stub.Image = MagicMock()
    sys.modules["PIL"] = _pil_stub
    sys.modules["PIL.Image"] = MagicMock()


# ---------------------------------------------------------------------------
# Settings/env isolation
#
# Some modules mutate global ``os.environ`` as an *import-time* side effect.
# In particular ``scripts.batch_coach_matchup_cache`` runs
# ``_preload_workspace_env()`` at import, which ``setdefault()``s every key
# from the workspace ``.env.prod`` (including ``AI_VECTOR_QUANTIZATION=halfvec``)
# into ``os.environ``. That import happens during *collection* of tests such as
# ``tests/test_batch_auto_brief_helpers.py``, so by the time other modules
# import ``app.routers.vision`` (which calls ``get_settings()`` at import) the
# cached ``Settings`` singleton is poisoned with prod values.
#
# This breaks tests that assume default settings — e.g.
# ``test_retrieval.py::test_similarity_search_without_keyword_keeps_vector_path``
# expects the plain ``embedding <=> %s::vector`` path but gets the
# ``halfvec(256)`` path. The failure only reproduces in the *full* suite (where
# the polluting module is collected), not in isolation.
#
# We snapshot a clean baseline for these keys before any test module is
# collected, then restore it and reset the cached settings around every test.
_ENV_BASELINE_KEYS = ("AI_VECTOR_QUANTIZATION",)
_env_baseline: dict[str, str | None] = {}


def _clear_settings_cache() -> None:
    try:
        get_settings.cache_clear()
    except AttributeError:
        # Keep tests resilient across versions of get_settings helper.
        pass


def pytest_configure(config):
    os.environ.setdefault("POSTGRES_DB_URL", "postgresql://localhost:5432/postgres")

    # pytest_configure runs before test modules are collected/imported, so the
    # environment here is still free of import-time pollution.
    for key in _ENV_BASELINE_KEYS:
        _env_baseline[key] = os.environ.get(key)

    _clear_settings_cache()


@pytest.fixture(autouse=True)
def _reset_settings_state():
    """Restore the clean env baseline and reset cached settings per test.

    Guards against import-time ``os.environ`` pollution (see module docstring)
    leaking into tests that read the cached ``get_settings()`` singleton.
    """
    for key, value in _env_baseline.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _clear_settings_cache()
    yield
    _clear_settings_cache()


# ---------------------------------------------------------------------------
# Shared async DB fakes (§4 psycopg3 async hot-path migration)
#
# Phases 1–2 convert RAG vector search and chat_cache to native async psycopg3.
# These fakes mirror the existing sync ``_DummyConnection``/``_DummyCursor``
# patterns but support the async protocol (``async with`` cursors, awaitable
# execute/fetch). Use ``make_async_db(rows)`` to build a connection/pool pair.
# ---------------------------------------------------------------------------


class FakeAsyncCursor:
    """Async cursor stub. Captures executed SQL and returns preset rows."""

    def __init__(self, rows=None):
        self._rows = list(rows or [])
        self.executed: list[tuple] = []

    async def execute(self, query, params=None):
        self.executed.append((query, params))
        return self

    async def fetchall(self):
        return list(self._rows)

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeAsyncConnection:
    """Async connection stub yielding ``FakeAsyncCursor``."""

    def __init__(self, rows=None):
        self._rows = list(rows or [])
        self.closed = False
        self.last_cursor: FakeAsyncCursor | None = None

    def cursor(self, *args, **kwargs):
        self.last_cursor = FakeAsyncCursor(self._rows)
        return self.last_cursor

    async def execute(self, query, params=None):
        # psycopg3 shorthand: ``(await conn.execute(...)).fetchone()``
        cur = FakeAsyncCursor(self._rows)
        await cur.execute(query, params)
        self.last_cursor = cur
        return cur

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeAsyncConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakeAsyncPool:
    """Async connection-pool stub. ``async with pool.connection() as conn``."""

    def __init__(self, conn):
        self._conn = conn

    def connection(self):
        return _FakeAsyncConnCtx(self._conn)

    async def open(self, *args, **kwargs):
        return None

    async def close(self, *args, **kwargs):
        return None

    def get_stats(self):
        return {}


@pytest.fixture
def make_async_db():
    """Factory: ``conn, pool = make_async_db(rows)`` for async DB-path tests."""

    def _factory(rows=None):
        conn = FakeAsyncConnection(rows)
        return conn, FakeAsyncPool(conn)

    return _factory
