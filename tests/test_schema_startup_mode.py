from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app import deps
from app.core import embedding_cache


class _ConnectionContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class _Pool:
    def connection(self):
        return _ConnectionContext()


class _RecordingConnection:
    def __init__(self):
        self.executed = []

    async def execute(self, statement):
        self.executed.append(statement)


class _RecordingConnectionContext:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class _RecordingPool:
    def __init__(self):
        self.connection_instance = _RecordingConnection()

    def connection(self):
        return _RecordingConnectionContext(self.connection_instance)


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
        patch.object(deps, "get_ingest_connection_pool", return_value=pool),
    ):
        store = deps.get_ingest_run_store()

    assert store.pool is pool
    assert store.lease_seconds == 240
    assert store.max_recovery_attempts == 2


def _lifespan_settings(*, ingest_worker_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        coach_failed_recovery_enabled=False,
        ingest_worker_enabled=ingest_worker_enabled,
    )


def _patch_lifespan_resources(
    monkeypatch,
    *,
    settings,
    cleanup_loop=None,
    metrics_loop=None,
):
    async def wait_forever():
        await asyncio.Event().wait()

    backend_get = AsyncMock(return_value=object())
    close_http = AsyncMock()
    close_ingest = AsyncMock()
    close_general = AsyncMock()
    monkeypatch.setattr(deps, "get_settings", lambda: settings)
    monkeypatch.setattr(deps, "load_clf", lambda: None)
    monkeypatch.setattr(
        deps,
        "_prepare_required_database_pools",
        AsyncMock(return_value=(object(), object())),
    )
    monkeypatch.setattr(deps, "_initialize_shared_baseball_agent_runtime", lambda: None)
    monkeypatch.setattr(
        deps,
        "_chat_cache_cleanup_loop",
        cleanup_loop or wait_forever,
    )
    monkeypatch.setattr(
        deps,
        "_db_pool_metrics_loop",
        metrics_loop or wait_forever,
    )
    monkeypatch.setattr(deps, "close_shared_httpx_clients", close_http)
    monkeypatch.setattr(deps, "close_ingest_connection_pool", close_ingest)
    monkeypatch.setattr(deps, "close_connection_pool", close_general)
    monkeypatch.setattr(deps, "reset_shared_baseball_agent_runtime", lambda: None)
    monkeypatch.setattr(embedding_cache, "get_backend", backend_get)
    return SimpleNamespace(
        backend_get=backend_get,
        close_http=close_http,
        close_ingest=close_ingest,
        close_general=close_general,
    )


def test_lifespan_cleans_post_open_initial_recovery_failure(monkeypatch):
    class InitialRecoveryError(RuntimeError):
        pass

    task_refs = []

    async def tracked_loop():
        task_refs.append(asyncio.current_task())
        await asyncio.Event().wait()

    resources = _patch_lifespan_resources(
        monkeypatch,
        settings=_lifespan_settings(ingest_worker_enabled=True),
        cleanup_loop=tracked_loop,
        metrics_loop=tracked_loop,
    )

    class _FailingWorker:
        def __init__(self, **kwargs):
            del kwargs

        async def recover_expired_once(self):
            await asyncio.sleep(0)
            raise InitialRecoveryError("initial recovery failed")

    monkeypatch.setattr(deps, "get_ingest_run_store", lambda: object())
    monkeypatch.setattr(deps, "IngestWorker", _FailingWorker)

    async def scenario():
        with pytest.raises(InitialRecoveryError, match="initial recovery failed"):
            async with deps.lifespan(None):
                raise AssertionError("startup must not reach yield")
        assert task_refs
        assert all(task is not None and task.done() for task in task_refs)

    asyncio.run(scenario())

    resources.close_ingest.assert_awaited_once()
    resources.close_general.assert_awaited_once()
    resources.backend_get.assert_not_awaited()


def test_lifespan_awaits_all_tasks_and_closers_after_task_cleanup_failure(
    monkeypatch,
):
    class TaskCleanupError(RuntimeError):
        pass

    second_task_stopped = False

    async def failing_cleanup_loop():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise TaskCleanupError("cleanup task failed")

    async def tracked_metrics_loop():
        nonlocal second_task_stopped
        try:
            await asyncio.Event().wait()
        finally:
            second_task_stopped = True

    resources = _patch_lifespan_resources(
        monkeypatch,
        settings=_lifespan_settings(ingest_worker_enabled=False),
        cleanup_loop=failing_cleanup_loop,
        metrics_loop=tracked_metrics_loop,
    )

    async def scenario():
        with pytest.raises(TaskCleanupError, match="cleanup task failed"):
            async with deps.lifespan(None):
                await asyncio.sleep(0)

    asyncio.run(scenario())

    assert second_task_stopped is True
    resources.close_http.assert_awaited_once()
    resources.close_ingest.assert_awaited_once()
    resources.close_general.assert_awaited_once()


def test_lifespan_attempts_general_pool_close_after_ingest_close_failure(
    monkeypatch,
):
    class IngestCloseError(RuntimeError):
        pass

    resources = _patch_lifespan_resources(
        monkeypatch,
        settings=_lifespan_settings(ingest_worker_enabled=False),
    )
    resources.close_ingest.side_effect = IngestCloseError("ingest close failed")

    async def scenario():
        with pytest.raises(IngestCloseError, match="ingest close failed"):
            async with deps.lifespan(None):
                await asyncio.sleep(0)

    asyncio.run(scenario())

    resources.close_ingest.assert_awaited_once()
    resources.close_general.assert_awaited_once()
    assert resources.backend_get.await_count == 1


def test_lifespan_preserves_request_error_while_logging_cleanup_class_only(
    monkeypatch,
    caplog,
):
    class RequestError(RuntimeError):
        pass

    class CleanupError(RuntimeError):
        pass

    resources = _patch_lifespan_resources(
        monkeypatch,
        settings=_lifespan_settings(ingest_worker_enabled=False),
    )
    secret = "postgresql://user:secret@internal/db"
    resources.close_ingest.side_effect = CleanupError(secret)

    async def scenario():
        request_error = RequestError("request failed")
        with pytest.raises(RequestError) as raised:
            async with deps.lifespan(None):
                await asyncio.sleep(0)
                raise request_error
        assert raised.value is request_error

    asyncio.run(scenario())

    resources.close_general.assert_awaited_once()
    assert "CleanupError" in caplog.text
    assert secret not in caplog.text


def test_auto_schema_compatibility_ensures_ingest_orchestration_tables():
    pool = _RecordingPool()

    asyncio.run(deps._ensure_ingest_orchestration_schema(pool))

    executed = pool.connection_instance.executed
    assert len(executed) == 2
    assert "CREATE TABLE IF NOT EXISTS ai_ingest_runs" in executed[0]
    assert "CREATE TABLE IF NOT EXISTS ai_ingest_checkpoints" in executed[1]
