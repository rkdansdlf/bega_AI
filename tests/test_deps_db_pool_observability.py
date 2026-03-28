from __future__ import annotations

import logging

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
            def __enter__(self_inner):
                raise PoolTimeout("timed out waiting for pool")

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


class _OperationalErrorPool:
    def get_stats(self):
        return {"requests_waiting": 2, "pool_available": 1}

    def connection(self):
        class _Ctx:
            def __enter__(self_inner):
                raise psycopg.OperationalError("could not connect")

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


def test_snapshot_connection_pool_stats_includes_configured_limits() -> None:
    snapshot = deps._snapshot_connection_pool_stats(_PoolTimeoutPool())

    assert snapshot["min_size"] == deps.DB_POOL_MIN_SIZE
    assert snapshot["max_size"] == deps.DB_POOL_MAX_SIZE
    assert snapshot["pool_available"] is True
    assert snapshot["stats"]["requests_waiting"] == 3


def test_get_db_connection_logs_pool_stats_on_pool_timeout(monkeypatch, caplog) -> None:
    monkeypatch.setattr(deps, "get_connection_pool", lambda: _PoolTimeoutPool())
    caplog.set_level(logging.ERROR)

    with pytest.raises(HTTPException) as excinfo:
        next(deps.get_db_connection())

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
        next(deps.get_db_connection())

    assert excinfo.value.status_code == 503
    assert "Operational error while acquiring connection" in caplog.text
    assert "pool_stats=" in caplog.text
    assert '"requests_waiting": 2' in caplog.text
