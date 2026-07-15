from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from app.core.ingest_runs import (
    IngestRunRecord,
    IngestRunRequest,
    IngestRunStatus,
    IngestTableResult,
    build_request_key,
    build_watermark_scope_key,
)
from app.core.ingest_worker import IngestWorker
from scripts.ingest_from_kbo import (
    IngestExecutionResult,
    ManualBaseballDataRequiredError,
)


RUN_ID = UUID("33333333-3333-4333-8333-333333333333")
NOW = datetime(2026, 7, 15, 4, 30, tzinfo=UTC)
REQUEST = IngestRunRequest(
    tables=("game",),
    season_year=2026,
    trigger_source="BACKEND_SCHEDULED",
)
RUN = IngestRunRecord(
    run_id=RUN_ID,
    request_key=build_request_key(REQUEST),
    request=REQUEST,
    status=IngestRunStatus.RUNNING,
    requested_at=NOW,
    started_at=NOW,
    lease_owner="worker-1",
)
TABLE_RESULT = IngestTableResult("game", 3, 4, 1, 2, NOW)
EXECUTION_RESULT = IngestExecutionResult(tables={"game": TABLE_RESULT})
SETTINGS = SimpleNamespace(
    source_db_url="postgresql://internal-source",
    embed_batch_size=32,
    ingest_worker_poll_seconds=0.01,
    ingest_worker_lease_seconds=120,
)


def _read_metric_value(name: str, labels: dict[str, str]) -> float:
    prometheus_client = pytest.importorskip("prometheus_client")
    for metric in prometheus_client.REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == name and sample.labels == labels:
                return sample.value
    return 0.0


class _Store:
    def __init__(self, claimed=RUN):
        self.claimed = claimed
        self.successful_run_id = None
        self.successful_results = None
        self.successful_watermarks = None
        self.manual_contract = None
        self.failed = None
        self.heartbeat_calls = 0
        self.watermark_requests = []
        self.recovery_calls = 0
        self.active_counts = {}
        self.latest_watermarks = {}

    async def claim_next(self, owner):
        assert owner == "worker-1"
        claimed, self.claimed = self.claimed, None
        return claimed

    async def heartbeat(self, run_id, owner):
        self.heartbeat_calls += 1
        return run_id == RUN_ID and owner == "worker-1"

    async def finish_success(self, run_id, owner, results, watermarks, scope_key):
        assert owner == "worker-1"
        self.successful_run_id = run_id
        self.successful_results = results
        self.successful_watermarks = watermarks
        self.successful_scope_key = scope_key
        self.latest_watermarks.update(watermarks)

    async def finish_manual_data_required(self, run_id, owner, contract):
        assert run_id == RUN_ID
        assert owner == "worker-1"
        self.manual_contract = contract

    async def finish_failed(self, run_id, owner, **error):
        assert run_id == RUN_ID
        assert owner == "worker-1"
        self.failed = error

    async def get_watermark(self, source_table, scope_key):
        assert source_table == "game"
        self.watermark_requests.append((source_table, scope_key))
        return NOW

    async def recover_expired(self):
        self.recovery_calls += 1
        return (0, 0)

    async def count_active_by_status(self):
        return self.active_counts

    async def get_latest_watermarks_by_table(self):
        return self.latest_watermarks


def test_run_once_finishes_success_and_advances_watermarks(monkeypatch):
    store = _Store()
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    monkeypatch.setattr(worker, "_execute", AsyncMock(return_value=EXECUTION_RESULT))

    assert asyncio.run(worker.run_once()) is True
    assert store.successful_run_id == RUN_ID
    assert store.successful_results == {"game": TABLE_RESULT}
    assert store.successful_watermarks == {"game": NOW}
    assert store.successful_scope_key == build_watermark_scope_key(REQUEST)


def test_run_once_reconciles_persisted_watermark_lag_after_success(monkeypatch):
    store = _Store()
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    monkeypatch.setattr(worker, "_execute", AsyncMock(return_value=EXECUTION_RESULT))

    assert asyncio.run(worker.run_once()) is True

    assert _read_metric_value(
        "ai_ingest_watermark_lag_seconds", {"source_table": "game"}
    ) > 0.0


def test_manual_data_error_becomes_manual_terminal_status(monkeypatch):
    contract = {
        "code": "MANUAL_BASEBALL_DATA_REQUIRED",
        "entity": "game",
        "missing_fields": ["game_date"],
        "import_source": "operator_manual_data",
    }
    store = _Store()
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    monkeypatch.setattr(
        worker,
        "_execute",
        AsyncMock(side_effect=ManualBaseballDataRequiredError(contract)),
    )

    assert asyncio.run(worker.run_once()) is True
    assert store.manual_contract["code"] == "MANUAL_BASEBALL_DATA_REQUIRED"
    assert store.successful_run_id is None


def test_unexpected_error_uses_sanitized_terminal_failure(monkeypatch):
    store = _Store()
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    monkeypatch.setattr(
        worker,
        "_execute",
        AsyncMock(side_effect=RuntimeError("postgresql://user:secret@internal/db")),
    )

    assert asyncio.run(worker.run_once()) is True
    assert store.failed == {
        "error_code": "INGEST_EXECUTION_FAILED",
        "error_message": "RuntimeError",
    }


def test_run_once_returns_false_when_queue_is_empty():
    worker = IngestWorker(store=_Store(claimed=None), settings=SETTINGS, owner="worker-1")

    assert asyncio.run(worker.run_once()) is False


def test_execute_uses_incremental_watermark_for_each_table():
    calls = []

    def fake_ingest(**kwargs):
        calls.append(kwargs)
        return EXECUTION_RESULT

    worker = IngestWorker(
        store=_Store(),
        settings=SETTINGS,
        owner="worker-1",
        ingest_function=fake_ingest,
    )
    duration_count_before = _read_metric_value(
        "ai_ingest_table_duration_seconds_count", {"source_table": "game"}
    )
    rows_before = _read_metric_value(
        "ai_ingest_table_source_rows_total", {"source_table": "game"}
    )
    chunks_before = _read_metric_value(
        "ai_ingest_table_written_chunks_total", {"source_table": "game"}
    )

    result = asyncio.run(worker._execute(RUN))

    assert result.tables == {"game": TABLE_RESULT}
    assert calls[0]["tables"] == ["game"]
    assert calls[0]["since"] == NOW
    assert calls[0]["lease_run_id"] == RUN_ID
    assert calls[0]["lease_owner"] == "worker-1"
    assert worker.store.watermark_requests == [
        ("game", build_watermark_scope_key(REQUEST))
    ]
    assert _read_metric_value(
        "ai_ingest_table_duration_seconds_count", {"source_table": "game"}
    ) == duration_count_before + 1
    assert _read_metric_value(
        "ai_ingest_table_source_rows_total", {"source_table": "game"}
    ) - rows_before == TABLE_RESULT.source_rows
    assert _read_metric_value(
        "ai_ingest_table_written_chunks_total", {"source_table": "game"}
    ) - chunks_before == TABLE_RESULT.written_chunks


def test_execute_records_completed_table_counts_before_later_table_failure():
    request = IngestRunRequest(
        tables=("game", "game_metadata"),
        season_year=2026,
        mode="FULL",
        trigger_source="BACKEND_SCHEDULED",
    )
    run = IngestRunRecord(
        run_id=RUN_ID,
        request_key=build_request_key(request),
        request=request,
        status=IngestRunStatus.RUNNING,
        requested_at=NOW,
        started_at=NOW,
        lease_owner="worker-1",
    )
    calls = []

    def fake_ingest(**kwargs):
        calls.append(kwargs["tables"][0])
        if len(calls) == 1:
            return EXECUTION_RESULT
        raise RuntimeError("second table failed")

    worker = IngestWorker(
        store=_Store(),
        settings=SETTINGS,
        owner="worker-1",
        ingest_function=fake_ingest,
    )
    rows_before = _read_metric_value(
        "ai_ingest_table_source_rows_total", {"source_table": "game"}
    )
    chunks_before = _read_metric_value(
        "ai_ingest_table_written_chunks_total", {"source_table": "game"}
    )

    with pytest.raises(RuntimeError, match="second table failed"):
        asyncio.run(worker._execute(run))

    assert calls == ["game", "game_metadata"]
    assert _read_metric_value(
        "ai_ingest_table_source_rows_total", {"source_table": "game"}
    ) - rows_before == TABLE_RESULT.source_rows
    assert _read_metric_value(
        "ai_ingest_table_written_chunks_total", {"source_table": "game"}
    ) - chunks_before == TABLE_RESULT.written_chunks


def test_periodic_recovery_checks_again_after_startup():
    store = _Store(claimed=None)
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    worker.recovery_seconds = 0.01
    stop_event = asyncio.Event()

    async def run_until_two_recoveries():
        task = asyncio.create_task(worker.run_recovery_forever(stop_event))
        while store.recovery_calls < 2:
            await asyncio.sleep(0.005)
        stop_event.set()
        await task

    asyncio.run(run_until_two_recoveries())

    assert store.recovery_calls >= 2


def test_recovery_records_requeued_and_failed_lease_counts():
    stop_event = asyncio.Event()

    class _RecoveryStore(_Store):
        async def recover_expired(self):
            stop_event.set()
            return (2, 1)

    worker = IngestWorker(
        store=_RecoveryStore(claimed=None),
        settings=SETTINGS,
        owner="worker-1",
    )
    requeued_before = _read_metric_value(
        "ai_ingest_lease_recoveries_total", {"result": "requeued"}
    )
    failed_before = _read_metric_value(
        "ai_ingest_lease_recoveries_total", {"result": "failed"}
    )

    asyncio.run(worker.run_recovery_forever(stop_event))

    assert _read_metric_value(
        "ai_ingest_lease_recoveries_total", {"result": "requeued"}
    ) - requeued_before == 2
    assert _read_metric_value(
        "ai_ingest_lease_recoveries_total", {"result": "failed"}
    ) - failed_before == 1


def test_recover_expired_once_records_startup_recovery_metrics():
    class _StartupRecoveryStore(_Store):
        async def recover_expired(self):
            return (1, 2)

    worker = IngestWorker(
        store=_StartupRecoveryStore(claimed=None),
        settings=SETTINGS,
        owner="worker-1",
    )
    requeued_before = _read_metric_value(
        "ai_ingest_lease_recoveries_total", {"result": "requeued"}
    )
    failed_before = _read_metric_value(
        "ai_ingest_lease_recoveries_total", {"result": "failed"}
    )

    assert asyncio.run(worker.recover_expired_once()) == (1, 2)

    assert _read_metric_value(
        "ai_ingest_lease_recoveries_total", {"result": "requeued"}
    ) - requeued_before == 1
    assert _read_metric_value(
        "ai_ingest_lease_recoveries_total", {"result": "failed"}
    ) - failed_before == 2


def test_run_once_reconciles_queued_and_active_gauges_from_store():
    store = _Store(claimed=None)
    store.active_counts = {
        ("QUEUED", "BACKEND_SCHEDULED"): 3,
        ("RUNNING", "MANUAL_API"): 1,
    }
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")

    assert asyncio.run(worker.run_once()) is False

    assert _read_metric_value(
        "ai_ingest_queued_runs", {"trigger_source": "BACKEND_SCHEDULED"}
    ) == 3
    assert _read_metric_value(
        "ai_ingest_active_runs", {"trigger_source": "MANUAL_API"}
    ) == 1


def test_run_once_reconciles_watermark_lag_from_persisted_scopes():
    store = _Store(claimed=None)
    store.latest_watermarks = {"game": NOW}
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")

    assert asyncio.run(worker.run_once()) is False

    assert _read_metric_value(
        "ai_ingest_watermark_lag_seconds", {"source_table": "game"}
    ) > 0.0


def test_lease_loss_prevents_terminal_success_and_further_tables(monkeypatch):
    store = _Store()
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")

    async def lose_lease(run, lease_lost):
        del run
        lease_lost.set()

    async def complete_after_lease_loss(run, lease_lost):
        del run
        await asyncio.sleep(0)
        assert lease_lost.is_set()
        return EXECUTION_RESULT

    monkeypatch.setattr(worker, "_heartbeat_loop", lose_lease)
    monkeypatch.setattr(worker, "_execute", complete_after_lease_loss)

    assert asyncio.run(worker.run_once()) is True
    assert store.successful_run_id is None
    assert store.failed is None
