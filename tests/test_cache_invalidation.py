import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core import cache_invalidation as ci
from tests.test_ingest_worker import (
    EXECUTION_RESULT,
    RUN_ID,
    SETTINGS,
    REQUEST,
    _Store,
)
from app.core.ingest_worker import IngestWorker


class _Cur:
    def __init__(self, rowcount):
        self.rowcount = rowcount


class _Conn:
    def __init__(self):
        self.calls = []

    async def execute(self, sql, params):
        self.calls.append((sql, params))
        return _Cur(2)


class _Pool:
    def __init__(self, conn):
        self._conn = conn

    def connection(self):
        conn = self._conn

        class _Ctx:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


def test_invalidates_both_tables_with_scoped_params():
    conn = _Conn()
    deleted = asyncio.run(
        ci.invalidate_chat_caches(conn, season_years=[2026], team_ids=["LG"])
    )
    assert deleted == {
        "chat_response_cache": 2,
        "chat_semantic_response_cache": 2,
    }
    tables = [sql.split("DELETE FROM")[1].split()[0] for sql, _ in conn.calls]
    assert tables == ["chat_response_cache", "chat_semantic_response_cache"]
    params = conn.calls[0][1]
    assert params["seasons"] == ["2026"]
    assert params["teams"] == ["LG"]
    assert set(params["stable"]) == {"general_conversation", "knowledge_explanation"}
    assert params["everything"] is False


def test_unknown_table_is_rejected():
    with pytest.raises(ValueError):
        ci.build_invalidation_sql("users; DROP TABLE x")


def test_run_without_season_invalidates_everything_non_stable():
    conn = _Conn()
    request = SimpleNamespace(season_year=None, tables=("game",))
    asyncio.run(ci.invalidate_for_ingest_run(request, pool=_Pool(conn)))
    assert conn.calls[0][1]["everything"] is True


def test_invalidation_failure_never_raises():
    class _Broken:
        def connection(self):
            raise RuntimeError("db down")

    request = SimpleNamespace(season_year=2026, tables=("game",))
    assert asyncio.run(ci.invalidate_for_ingest_run(request, pool=_Broken())) == {}


def test_worker_invalidates_after_successful_run(monkeypatch):
    store = _Store()
    invalidator = AsyncMock()
    worker = IngestWorker(
        store=store,
        settings=SETTINGS,
        owner="worker-1",
        cache_invalidator=invalidator,
    )
    monkeypatch.setattr(worker, "_execute", AsyncMock(return_value=EXECUTION_RESULT))
    assert asyncio.run(worker.run_once()) is True
    invalidator.assert_awaited_once_with(REQUEST)


def test_worker_does_not_invalidate_failed_run(monkeypatch):
    store = _Store()
    invalidator = AsyncMock()
    worker = IngestWorker(
        store=store,
        settings=SETTINGS,
        owner="worker-1",
        cache_invalidator=invalidator,
    )
    monkeypatch.setattr(worker, "_execute", AsyncMock(side_effect=RuntimeError("x")))
    asyncio.run(worker.run_once())
    invalidator.assert_not_awaited()
    assert RUN_ID  # imported fixture sanity
