import asyncio
from types import SimpleNamespace

import pytest

from app import deps


class _Pool:
    def get_stats(self):
        return {"pool_max": 7, "pool_min": 2, "pool_available": 3}


@pytest.mark.asyncio
async def test_pool_metrics_loop_publishes_instead_of_swallowing_a_name_error(
    monkeypatch,
):
    published = []

    class _Gauge:
        def labels(self, pool, state):
            class _Child:
                def set(_self, value):
                    published.append((pool, state, value))

            return _Child()

    import app.observability.metrics as metrics

    monkeypatch.setattr(metrics, "AI_DB_POOL_SIZE", _Gauge())
    monkeypatch.setattr(
        deps,
        "get_settings",
        lambda: SimpleNamespace(
            db_pool_max_size=1,
            ingest_db_pool_max_size=1,
            baseball_db_pool_max_size=1,
            rag_db_pool_max_size=1,
        ),
    )
    for name in (
        "get_connection_pool",
        "get_ingest_connection_pool",
        "get_baseball_connection_pool",
        "get_rag_connection_pool",
    ):
        monkeypatch.setattr(deps, name, lambda: _Pool())

    calls = {"n": 0}

    async def fake_sleep(_):
        calls["n"] += 1
        if calls["n"] > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(deps.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await deps._db_pool_metrics_loop(interval_seconds=1)

    assert ("general", "max", 7.0) in published
    assert {p for p, _, _ in published} == {"general", "ingest", "baseball", "rag"}
