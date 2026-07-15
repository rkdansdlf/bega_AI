from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app import deps


class _ConnectionContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class _Pool:
    def connection(self):
        return _ConnectionContext()


def _settings(mode: str):
    return SimpleNamespace(
        ai_db_schema_mode=mode,
        chat_semantic_cache_vector_index_enabled=True,
    )


def test_managed_schema_preparation_validates_without_startup_ddl():
    validate = AsyncMock()
    ensure_startup_schema = AsyncMock()

    with (
        patch.object(deps, "validate_schema_contract", validate),
        patch.object(deps, "_ensure_startup_schema", ensure_startup_schema),
    ):
        asyncio.run(deps._prepare_schema(_Pool(), _settings("managed")))

    validate.assert_awaited_once()
    assert validate.await_args.kwargs == {"require_vector_index": True}
    ensure_startup_schema.assert_not_awaited()


def test_auto_schema_preparation_keeps_compatibility_startup_ddl():
    validate = AsyncMock()
    ensure_startup_schema = AsyncMock()

    with (
        patch.object(deps, "validate_schema_contract", validate),
        patch.object(deps, "_ensure_startup_schema", ensure_startup_schema),
    ):
        asyncio.run(deps._prepare_schema(_Pool(), _settings("auto")))

    ensure_startup_schema.assert_awaited_once()
    validate.assert_not_awaited()


def test_ingest_run_store_uses_worker_lease_and_recovery_settings():
    settings = SimpleNamespace(
        ingest_worker_lease_seconds=240,
        ingest_worker_max_recovery_attempts=2,
    )
    pool = object()

    with (
        patch.object(deps, "get_settings", return_value=settings),
        patch.object(deps, "get_connection_pool", return_value=pool),
    ):
        store = deps.get_ingest_run_store()

    assert store.pool is pool
    assert store.lease_seconds == 240
    assert store.max_recovery_attempts == 2
