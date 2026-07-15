from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

from app.core.ingest_runs import (
    IngestRunRecord,
    IngestRunRequest,
    IngestRunStatus,
    IngestTableResult,
    build_request_key,
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


class _Store:
    def __init__(self, claimed=RUN):
        self.claimed = claimed
        self.successful_run_id = None
        self.successful_results = None
        self.successful_watermarks = None
        self.manual_contract = None
        self.failed = None
        self.heartbeat_calls = 0

    async def claim_next(self, owner):
        assert owner == "worker-1"
        claimed, self.claimed = self.claimed, None
        return claimed

    async def heartbeat(self, run_id, owner):
        self.heartbeat_calls += 1
        return run_id == RUN_ID and owner == "worker-1"

    async def finish_success(self, run_id, owner, results, watermarks):
        assert owner == "worker-1"
        self.successful_run_id = run_id
        self.successful_results = results
        self.successful_watermarks = watermarks

    async def finish_manual_data_required(self, run_id, owner, contract):
        assert run_id == RUN_ID
        assert owner == "worker-1"
        self.manual_contract = contract

    async def finish_failed(self, run_id, owner, **error):
        assert run_id == RUN_ID
        assert owner == "worker-1"
        self.failed = error

    async def get_watermark(self, source_table):
        assert source_table == "game"
        return NOW


def test_run_once_finishes_success_and_advances_watermarks(monkeypatch):
    store = _Store()
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    monkeypatch.setattr(worker, "_execute", AsyncMock(return_value=EXECUTION_RESULT))

    assert asyncio.run(worker.run_once()) is True
    assert store.successful_run_id == RUN_ID
    assert store.successful_results == {"game": TABLE_RESULT}
    assert store.successful_watermarks == {"game": NOW}


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

    result = asyncio.run(worker._execute(RUN))

    assert result.tables == {"game": TABLE_RESULT}
    assert calls[0]["tables"] == ["game"]
    assert calls[0]["since"] == NOW
