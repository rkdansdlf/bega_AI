from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import psycopg
from psycopg_pool import PoolTimeout
import pytest
from fastapi import HTTPException

from app import deps


class _PoolTimeoutPool:
    def get_stats(self):
        return {"requests_waiting": 3, "pool_available": 0}

    def connection(self):
        class _Ctx:
            async def __aenter__(self_inner):
                raise PoolTimeout("timed out waiting for pool")

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


class _OperationalErrorPool:
    def get_stats(self):
        return {"requests_waiting": 2, "pool_available": 1}

    def connection(self):
        class _Ctx:
            async def __aenter__(self_inner):
                raise psycopg.OperationalError("could not connect")

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


def test_snapshot_connection_pool_stats_includes_configured_limits() -> None:
    snapshot = deps._snapshot_connection_pool_stats(_PoolTimeoutPool())

    assert snapshot["min_size"] == deps.DB_POOL_MIN_SIZE
    assert snapshot["max_size"] == deps.DB_POOL_MAX_SIZE
    assert snapshot["pool_available"] is True
    assert snapshot["stats"]["requests_waiting"] == 3


def test_get_connection_pool_checks_connection_before_borrow(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeConnectionPool:
        check_connection = object()

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(deps, "_connection_pool", None)
    monkeypatch.setattr(
        deps,
        "get_settings",
        lambda: type("_Settings", (), {"database_url": "postgresql://example/db"})(),
    )
    monkeypatch.setattr(deps, "AsyncConnectionPool", _FakeConnectionPool)

    pool = deps.get_connection_pool()

    assert isinstance(pool, _FakeConnectionPool)
    assert captured["check"] is _FakeConnectionPool.check_connection
    monkeypatch.setattr(deps, "_connection_pool", None)


def test_general_and_ingest_pools_are_distinct_and_bounded(monkeypatch) -> None:
    captured = []

    class _FakeConnectionPool:
        check_connection = object()

        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(deps, "_connection_pool", None)
    monkeypatch.setattr(deps, "_ingest_connection_pool", None, raising=False)
    monkeypatch.setattr(
        deps,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(deps, "AsyncConnectionPool", _FakeConnectionPool)

    general_pool = deps.get_connection_pool()
    ingest_pool = deps.get_ingest_connection_pool()

    assert general_pool is not ingest_pool
    assert captured[0]["min_size"] == 1
    assert captured[0]["max_size"] == 30
    assert captured[1]["min_size"] == 1
    assert captured[1]["max_size"] == 2


def test_required_pool_startup_closes_both_pools_on_failure(monkeypatch) -> None:
    class _LifecyclePool:
        def __init__(self, failure=None):
            self.failure = failure
            self.opened = False
            self.closed = False

        async def open(self, *, wait, timeout):
            assert wait is True
            assert timeout == 10.0
            if self.failure is not None:
                raise self.failure
            self.opened = True

        async def close(self):
            self.closed = True

        def get_stats(self):
            return {"pool_available": 0}

    general_pool = _LifecyclePool()
    ingest_pool = _LifecyclePool(RuntimeError("ingest pool unavailable"))
    monkeypatch.setattr(deps, "_connection_pool", general_pool)
    monkeypatch.setattr(deps, "_ingest_connection_pool", ingest_pool, raising=False)
    monkeypatch.setattr(deps, "_prepare_schema", AsyncMock())

    with pytest.raises(RuntimeError, match="ingest pool unavailable"):
        asyncio.run(
            deps._prepare_required_database_pools(
                SimpleNamespace(ai_db_schema_mode="managed")
            )
        )

    assert general_pool.opened is True
    assert general_pool.closed is True
    assert ingest_pool.closed is True
    assert deps._connection_pool is None
    assert deps._ingest_connection_pool is None


def test_close_ingest_pool_is_idempotent(monkeypatch) -> None:
    class _ClosablePool:
        def __init__(self):
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    pool = _ClosablePool()
    monkeypatch.setattr(deps, "_ingest_connection_pool", pool, raising=False)

    asyncio.run(deps.close_ingest_connection_pool())
    asyncio.run(deps.close_ingest_connection_pool())

    assert pool.close_calls == 1
    assert deps._ingest_connection_pool is None


def test_ingest_heartbeat_does_not_acquire_busy_general_pool(monkeypatch) -> None:
    lease_expires_at = datetime(2026, 7, 18, 6, 0, tzinfo=UTC)
    run_id = UUID("44444444-4444-4444-8444-444444444444")

    class _BusyGeneralPool:
        def __init__(self):
            self.connection_calls = 0

        def connection(self):
            self.connection_calls += 1
            raise AssertionError("general pool must not serve ingest heartbeat")

    class _Cursor:
        async def fetchone(self):
            return {"lease_expires_at": lease_expires_at}

    class _Connection:
        async def execute(self, statement, params):
            assert "lease_expires_at > now()" in statement
            assert params[1] == run_id
            return _Cursor()

    class _ConnectionContext:
        async def __aenter__(self):
            return _Connection()

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class _AvailableIngestPool:
        def __init__(self):
            self.connection_calls = 0

        def connection(self):
            self.connection_calls += 1
            return _ConnectionContext()

    general_pool = _BusyGeneralPool()
    ingest_pool = _AvailableIngestPool()
    settings = SimpleNamespace(
        ingest_worker_lease_seconds=120,
        ingest_worker_max_recovery_attempts=1,
    )
    monkeypatch.setattr(deps, "get_settings", lambda: settings)
    monkeypatch.setattr(deps, "get_connection_pool", lambda: general_pool)
    monkeypatch.setattr(deps, "get_ingest_connection_pool", lambda: ingest_pool)

    store = deps.get_ingest_run_store()
    renewed_until = asyncio.run(store.heartbeat(run_id, "worker-1"))

    assert renewed_until == lease_expires_at
    assert general_pool.connection_calls == 0
    assert ingest_pool.connection_calls == 1


def test_get_db_connection_logs_pool_stats_on_pool_timeout(monkeypatch, caplog) -> None:
    monkeypatch.setattr(deps, "get_connection_pool", lambda: _PoolTimeoutPool())
    caplog.set_level(logging.ERROR)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(deps.get_db_connection().__anext__())

    assert excinfo.value.status_code == 503
    assert "Pool timeout while acquiring connection" in caplog.text
    assert "pool_stats=" in caplog.text
    assert '"requests_waiting": 3' in caplog.text


def test_get_db_connection_logs_pool_stats_on_operational_error(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(deps, "get_connection_pool", lambda: _OperationalErrorPool())
    caplog.set_level(logging.ERROR)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(deps.get_db_connection().__anext__())

    assert excinfo.value.status_code == 503
    assert "Operational error while acquiring connection" in caplog.text
    assert "pool_stats=" in caplog.text
    assert '"requests_waiting": 2' in caplog.text
