from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
import psycopg
from psycopg_pool import PoolTimeout

from app.core.ingest_runs import (
    IngestLeaseLostError,
    IngestRunRecord,
    IngestRunRequest,
    IngestRunStatus,
    IngestTableResult,
    build_request_key,
    build_watermark_scope_key,
)
from app.core.ingest_worker import IngestWorker
from app.core.ingest_checkpoints import (
    IngestCheckpointCursorUnavailableError,
    IngestCheckpointIncompatibleError,
)
from scripts import ingest_from_kbo as ingest_module
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
        if run_id == RUN_ID and owner == "worker-1":
            return NOW
        return None

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


class _ScriptedHeartbeatStore(_Store):
    def __init__(self, responses):
        super().__init__(claimed=None)
        self.responses = list(responses)

    async def heartbeat(self, run_id, owner):
        assert run_id == RUN_ID
        assert owner == "worker-1"
        self.heartbeat_calls += 1
        response = self.responses.pop(0) if self.responses else NOW
        if isinstance(response, Exception):
            raise response
        return response


async def _wait_for_heartbeat_calls(store, expected, timeout=0.5):
    async def reached_expected_calls():
        while store.heartbeat_calls < expected:
            await asyncio.sleep(0.001)

    await asyncio.wait_for(reached_expected_calls(), timeout=timeout)


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
    assert calls[0]["checkpoint_scope_key"] == build_watermark_scope_key(REQUEST)
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


def test_checkpoint_result_keeps_cumulative_and_attempt_counts_separate():
    result = IngestTableResult(
        "game",
        10,
        20,
        3,
        7,
        NOW,
        checkpoint_resumed=True,
        checkpoint_committed_batches=4,
        checkpoint_completed=True,
        attempt_source_rows=2,
        attempt_written_chunks=1,
    )

    assert result.source_rows == 20
    assert result.written_chunks == 10
    assert result.attempt_source_rows == 2
    assert result.attempt_written_chunks == 1


def test_table_metrics_use_current_attempt_deltas_for_resumed_checkpoint():
    worker = IngestWorker(store=_Store(), settings=SETTINGS, owner="worker-1")
    result = IngestTableResult(
        "game",
        10,
        20,
        3,
        7,
        NOW,
        checkpoint_resumed=True,
        checkpoint_committed_batches=4,
        checkpoint_completed=True,
        attempt_source_rows=2,
        attempt_written_chunks=1,
    )
    rows_before = _read_metric_value(
        "ai_ingest_table_source_rows_total", {"source_table": "game"}
    )
    chunks_before = _read_metric_value(
        "ai_ingest_table_written_chunks_total", {"source_table": "game"}
    )

    worker._record_table_result_metrics(
        IngestExecutionResult(tables={"game": result})
    )

    assert _read_metric_value(
        "ai_ingest_table_source_rows_total", {"source_table": "game"}
    ) - rows_before == 2
    assert _read_metric_value(
        "ai_ingest_table_written_chunks_total", {"source_table": "game"}
    ) - chunks_before == 1


def test_table_metrics_do_not_repeat_completed_checkpoint_cumulative_counts():
    worker = IngestWorker(store=_Store(), settings=SETTINGS, owner="worker-1")
    result = IngestTableResult(
        "game",
        10,
        20,
        3,
        7,
        NOW,
        checkpoint_resumed=True,
        checkpoint_committed_batches=4,
        checkpoint_completed=True,
        attempt_source_rows=0,
        attempt_written_chunks=0,
    )
    rows_before = _read_metric_value(
        "ai_ingest_table_source_rows_total", {"source_table": "game"}
    )
    chunks_before = _read_metric_value(
        "ai_ingest_table_written_chunks_total", {"source_table": "game"}
    )

    worker._record_table_result_metrics(
        IngestExecutionResult(tables={"game": result})
    )

    assert _read_metric_value(
        "ai_ingest_table_source_rows_total", {"source_table": "game"}
    ) == rows_before
    assert _read_metric_value(
        "ai_ingest_table_written_chunks_total", {"source_table": "game"}
    ) == chunks_before


@pytest.mark.parametrize(
    "result",
    ["created", "advanced", "completed", "resumed"],
)
def test_checkpoint_lifecycle_event_increments_exactly_once(result):
    before = _read_metric_value(
        "ai_ingest_checkpoint_events_total",
        {"source_table": "game", "result": result},
    )

    ingest_module._record_checkpoint_event("game", result)

    assert _read_metric_value(
        "ai_ingest_checkpoint_events_total",
        {"source_table": "game", "result": result},
    ) - before == 1


@pytest.mark.parametrize(
    ("error", "result"),
    [
        (IngestCheckpointIncompatibleError("stored state changed"), "incompatible"),
        (IngestCheckpointCursorUnavailableError("no stable cursor"), "rejected"),
    ],
)
def test_checkpoint_rejection_event_classifies_exception_without_error_labels(
    error,
    result,
):
    before = _read_metric_value(
        "ai_ingest_checkpoint_events_total",
        {"source_table": "game", "result": result},
    )

    ingest_module._record_checkpoint_rejection("game", error)

    assert _read_metric_value(
        "ai_ingest_checkpoint_events_total",
        {"source_table": "game", "result": result},
    ) - before == 1


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

    async def lose_lease(run, lease_lost, last_confirmed_monotonic):
        del run
        assert last_confirmed_monotonic > 0
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


@pytest.mark.parametrize(
    "transient_error",
    [
        psycopg.OperationalError("temporary db failure"),
        psycopg.InterfaceError("temporary db interface failure"),
        PoolTimeout("temporary coordination pool pressure"),
    ],
    ids=["operational", "interface", "pool-timeout"],
)
def test_heartbeat_retries_recognized_transient_error_without_losing_lease(
    transient_error,
):
    store = _ScriptedHeartbeatStore([transient_error, NOW])
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    worker.lease_seconds = 0.1
    worker.heartbeat_interval_seconds = 0.001
    worker.heartbeat_safety_margin_seconds = 0.01
    worker.heartbeat_retry_initial_seconds = 0.001
    lease_lost = asyncio.Event()
    retry_before = _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "retry"}
    )
    success_before = _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "success"}
    )

    async def scenario():
        task = asyncio.create_task(worker._heartbeat_loop(RUN, lease_lost))
        await _wait_for_heartbeat_calls(store, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert lease_lost.is_set() is False
    assert store.heartbeat_calls >= 2
    assert _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "retry"}
    ) - retry_before == 1
    assert _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "success"}
    ) - success_before >= 1


def test_heartbeat_exhaustion_marks_lease_lost():
    store = _ScriptedHeartbeatStore([PoolTimeout("db unavailable")] * 20)
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    worker.lease_seconds = 0.08
    worker.heartbeat_interval_seconds = 0.001
    worker.heartbeat_safety_margin_seconds = 0.02
    worker.heartbeat_retry_initial_seconds = 0.01
    lease_lost = asyncio.Event()
    exhausted_before = _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "exhausted"}
    )

    async def scenario():
        await asyncio.wait_for(worker._heartbeat_loop(RUN, lease_lost), timeout=0.3)

    asyncio.run(scenario())

    assert lease_lost.is_set() is True
    assert store.heartbeat_calls >= 2
    assert _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "exhausted"}
    ) - exhausted_before == 1


def test_hung_heartbeat_call_times_out_at_safe_deadline_exactly_once():
    call_cancelled = False

    class _HungHeartbeatStore(_Store):
        async def heartbeat(self, run_id, owner):
            nonlocal call_cancelled
            assert run_id == RUN_ID
            assert owner == "worker-1"
            self.heartbeat_calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                call_cancelled = True

    store = _HungHeartbeatStore(claimed=None)
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    worker.lease_seconds = 0.08
    worker.heartbeat_interval_seconds = 0.001
    worker.heartbeat_safety_margin_seconds = 0.02
    lease_lost = asyncio.Event()
    exhausted_before = _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "exhausted"}
    )

    asyncio.run(
        asyncio.wait_for(worker._heartbeat_loop(RUN, lease_lost), timeout=0.2)
    )

    assert lease_lost.is_set() is True
    assert store.heartbeat_calls == 1
    assert call_cancelled is True
    assert _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "exhausted"}
    ) - exhausted_before == 1


def test_delayed_success_keeps_request_start_as_confirmation_point():
    second_call_cancelled = False

    class _DelayedThenHungStore(_Store):
        async def heartbeat(self, run_id, owner):
            nonlocal second_call_cancelled
            assert run_id == RUN_ID
            assert owner == "worker-1"
            self.heartbeat_calls += 1
            if self.heartbeat_calls == 1:
                await asyncio.sleep(0.12)
                return NOW
            try:
                await asyncio.Event().wait()
            finally:
                second_call_cancelled = True

    store = _DelayedThenHungStore(claimed=None)
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    worker.lease_seconds = 0.2
    worker.heartbeat_interval_seconds = 0.001
    worker.heartbeat_safety_margin_seconds = 0.02
    lease_lost = asyncio.Event()
    exhausted_before = _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "exhausted"}
    )

    asyncio.run(
        asyncio.wait_for(worker._heartbeat_loop(RUN, lease_lost), timeout=0.24)
    )

    assert lease_lost.is_set() is True
    assert store.heartbeat_calls == 2
    assert second_call_cancelled is True
    assert _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "exhausted"}
    ) - exhausted_before == 1


def test_non_transient_heartbeat_error_stops_without_retry_or_secret_log(
    caplog,
):
    secret = "postgresql://user:secret@internal/db"
    store = _ScriptedHeartbeatStore([ValueError(secret), NOW])
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    worker.lease_seconds = 0.5
    worker.heartbeat_interval_seconds = 0.001
    lease_lost = asyncio.Event()
    before = {
        result: _read_metric_value(
            "ai_ingest_heartbeats_total", {"result": result}
        )
        for result in ("success", "retry", "rejected", "exhausted")
    }
    caplog.set_level(logging.ERROR)

    asyncio.run(
        asyncio.wait_for(worker._heartbeat_loop(RUN, lease_lost), timeout=0.1)
    )

    assert lease_lost.is_set() is True
    assert store.heartbeat_calls == 1
    assert "ValueError" in caplog.text
    assert str(RUN_ID) in caplog.text
    assert secret not in caplog.text
    for result, previous in before.items():
        assert _read_metric_value(
            "ai_ingest_heartbeats_total", {"result": result}
        ) == previous


def test_run_once_records_initial_confirmation_before_claim_request(monkeypatch):
    class _DelayedClaimStore(_Store):
        def __init__(self):
            super().__init__()
            self.claim_started = 0.0
            self.claim_returned = 0.0

        async def claim_next(self, owner):
            self.claim_started = asyncio.get_running_loop().time()
            await asyncio.sleep(0.03)
            claimed = await super().claim_next(owner)
            self.claim_returned = asyncio.get_running_loop().time()
            return claimed

    store = _DelayedClaimStore()
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    confirmation = {}

    async def capture_confirmation(run, lease_lost, last_confirmed_monotonic):
        del run
        confirmation["value"] = last_confirmed_monotonic
        lease_lost.set()

    async def finish_after_heartbeat_started(run, lease_lost):
        del run
        while "value" not in confirmation:
            await asyncio.sleep(0)
        assert lease_lost.is_set()
        return EXECUTION_RESULT

    monkeypatch.setattr(worker, "_heartbeat_loop", capture_confirmation)
    monkeypatch.setattr(worker, "_execute", finish_after_heartbeat_started)

    assert asyncio.run(worker.run_once()) is True

    assert confirmation["value"] <= store.claim_started
    assert confirmation["value"] < store.claim_returned - 0.01


def test_heartbeat_rejection_marks_lease_lost_without_retry():
    store = _ScriptedHeartbeatStore([None])
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    worker.heartbeat_interval_seconds = 0.001
    lease_lost = asyncio.Event()
    rejected_before = _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "rejected"}
    )

    asyncio.run(worker._heartbeat_loop(RUN, lease_lost))

    assert lease_lost.is_set() is True
    assert store.heartbeat_calls == 1
    assert _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "rejected"}
    ) - rejected_before == 1


def test_heartbeat_cancellation_does_not_mark_lease_lost():
    entered = asyncio.Event()
    release = asyncio.Event()

    class _BlockingStore(_Store):
        async def heartbeat(self, run_id, owner):
            entered.set()
            await release.wait()
            return NOW

    store = _BlockingStore(claimed=None)
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    worker.heartbeat_interval_seconds = 0.001
    lease_lost = asyncio.Event()
    exhausted_before = _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "exhausted"}
    )

    async def scenario():
        task = asyncio.create_task(worker._heartbeat_loop(RUN, lease_lost))
        await asyncio.wait_for(entered.wait(), timeout=0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert lease_lost.is_set() is False
    assert _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "exhausted"}
    ) == exhausted_before


def test_long_sync_ingest_keeps_heartbeating_past_lease_interval():
    store = _Store()
    ingest_started = threading.Event()
    release_ingest = threading.Event()
    required_heartbeat_calls = 32

    def blocked_ingest(**kwargs):
        assert kwargs["lease_run_id"] == RUN_ID
        ingest_started.set()
        if not release_ingest.wait(timeout=5.0):
            raise AssertionError("test did not release blocked ingest")
        return EXECUTION_RESULT

    worker = IngestWorker(
        store=store,
        settings=SETTINGS,
        owner="worker-1",
        ingest_function=blocked_ingest,
    )
    worker.lease_seconds = 0.3
    worker.heartbeat_interval_seconds = 0.01
    worker.heartbeat_safety_margin_seconds = 0.05

    async def scenario():
        run_task = asyncio.create_task(worker.run_once())
        try:
            assert await asyncio.to_thread(ingest_started.wait, 5.0) is True
            await _wait_for_heartbeat_calls(
                store,
                required_heartbeat_calls,
                timeout=5.0,
            )
        finally:
            release_ingest.set()
        return await asyncio.wait_for(run_task, timeout=5.0)

    assert asyncio.run(scenario()) is True
    assert store.successful_run_id == RUN_ID
    assert store.heartbeat_calls >= required_heartbeat_calls


def test_expired_lease_during_success_finish_is_left_for_recovery(monkeypatch):
    class _FinishRaceStore(_Store):
        async def finish_success(self, *args, **kwargs):
            raise IngestLeaseLostError

    store = _FinishRaceStore()
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    monkeypatch.setattr(worker, "_execute", AsyncMock(return_value=EXECUTION_RESULT))

    assert asyncio.run(worker.run_once()) is True
    assert store.successful_run_id is None
    assert store.failed is None


def test_expired_lease_during_failure_finish_is_left_for_recovery(monkeypatch):
    class _FinishRaceStore(_Store):
        async def finish_failed(self, *args, **kwargs):
            raise IngestLeaseLostError

    store = _FinishRaceStore()
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    monkeypatch.setattr(
        worker,
        "_execute",
        AsyncMock(side_effect=RuntimeError("execution failed")),
    )

    assert asyncio.run(worker.run_once()) is True
    assert store.failed is None
    assert store.successful_run_id is None
    assert store.manual_contract is None


def test_expired_lease_during_manual_finish_is_left_for_recovery(monkeypatch):
    contract = {
        "code": "MANUAL_BASEBALL_DATA_REQUIRED",
        "entity": "game",
        "missing_fields": ["game_date"],
        "import_source": "operator_manual_data",
    }

    class _FinishRaceStore(_Store):
        async def finish_manual_data_required(self, *args, **kwargs):
            raise IngestLeaseLostError

    store = _FinishRaceStore()
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    monkeypatch.setattr(
        worker,
        "_execute",
        AsyncMock(side_effect=ManualBaseballDataRequiredError(contract)),
    )

    assert asyncio.run(worker.run_once()) is True
    assert store.manual_contract is None
    assert store.successful_run_id is None
    assert store.failed is None
