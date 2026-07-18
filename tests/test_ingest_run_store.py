from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID

from app.core.ingest_run_store import IngestRunStore
from app.core.ingest_runs import (
    IngestRunRequest,
    IngestRunStatus,
    IngestTableResult,
    build_request_key,
    build_watermark_scope_key,
)


RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
FAILED_RUN_ID = UUID("22222222-2222-4222-8222-222222222222")
REQUESTED_AT = datetime(2026, 7, 15, 4, 30, tzinfo=UTC)
WATERMARK = datetime(2026, 7, 15, 4, 0, tzinfo=UTC)
REQUEST = IngestRunRequest(
    tables=("game",),
    season_year=2026,
    trigger_source="BACKEND_SCHEDULED",
)
WATERMARK_SCOPE = build_watermark_scope_key(REQUEST)


def _run_row(*, status: str = "QUEUED", run_id: UUID = RUN_ID):
    return {
        "run_id": run_id,
        "request_key": build_request_key(REQUEST),
        "trigger_source": REQUEST.trigger_source,
        "status": status,
        "request_payload": REQUEST.to_payload(),
        "requested_at": REQUESTED_AT,
        "started_at": REQUESTED_AT if status == "RUNNING" else None,
        "heartbeat_at": REQUESTED_AT if status == "RUNNING" else None,
        "finished_at": None,
        "lease_owner": "worker-1" if status == "RUNNING" else None,
        "lease_expires_at": None,
        "recovery_attempts": 0,
        "error_code": None,
        "error_message": None,
        "table_summary": {},
    }


class _AsyncContext:
    def __init__(self, value=None, on_enter=None):
        self.value = value
        self.on_enter = on_enter

    async def __aenter__(self):
        if self.on_enter is not None:
            self.on_enter()
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _Cursor:
    def __init__(self, response):
        self.response = response

    async def fetchone(self):
        if isinstance(self.response, list):
            return self.response[0] if self.response else None
        return self.response

    async def fetchall(self):
        if self.response is None:
            return []
        if isinstance(self.response, list):
            return self.response
        return [self.response]


class _Connection:
    def __init__(self, responses):
        self.responses = list(responses)
        self.executed = []
        self.transaction_entries = 0

    def transaction(self):
        return _AsyncContext(on_enter=self._record_transaction)

    def _record_transaction(self):
        self.transaction_entries += 1

    async def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))
        response = self.responses.pop(0) if self.responses else None
        return _Cursor(response)


class _Pool:
    def __init__(self, responses):
        self.connection_instance = _Connection(responses)

    def connection(self):
        return _AsyncContext(self.connection_instance)


def test_create_or_get_active_returns_existing_run_on_unique_conflict():
    pool = _Pool([None, _run_row()])
    store = IngestRunStore(pool)

    record, deduplicated = asyncio.run(store.create_or_get_active(REQUEST))

    assert record.run_id == RUN_ID
    assert record.status is IngestRunStatus.QUEUED
    assert deduplicated is True
    assert "ON CONFLICT DO NOTHING" in pool.connection_instance.executed[0][0]


def test_claim_next_uses_skip_locked_and_assigns_lease_owner():
    pool = _Pool([(RUN_ID,), _run_row(status="RUNNING")])
    store = IngestRunStore(pool, lease_seconds=120)

    record = asyncio.run(store.claim_next("worker-1"))

    assert record is not None
    assert record.status is IngestRunStatus.RUNNING
    sql = " ".join(item[0] for item in pool.connection_instance.executed)
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "lease_owner = %s" in sql
    assert pool.connection_instance.transaction_entries == 1


def test_heartbeat_requires_unexpired_owner_and_returns_new_expiry():
    pool = _Pool([{"lease_expires_at": WATERMARK}])
    store = IngestRunStore(pool, lease_seconds=120)

    lease_expires_at = asyncio.run(store.heartbeat(RUN_ID, "worker-1"))

    sql = pool.connection_instance.executed[0][0]
    assert "status = 'RUNNING'" in sql
    assert "lease_owner = %s" in sql
    assert "lease_expires_at > now()" in sql
    assert "RETURNING lease_expires_at" in sql
    assert lease_expires_at == WATERMARK


def test_finish_success_advances_only_committed_table_watermarks():
    pool = _Pool([(RUN_ID,), None])
    store = IngestRunStore(pool)
    result = IngestTableResult("game", 3, 4, 1, 2, WATERMARK)

    asyncio.run(
        store.finish_success(
            RUN_ID,
            "worker-1",
            {"game": result},
            {"game": WATERMARK},
            WATERMARK_SCOPE,
        )
    )

    sql = " ".join(item[0] for item in pool.connection_instance.executed)
    assert "status = 'SUCCEEDED'" in sql
    assert "INSERT INTO ai_ingest_watermarks" in sql
    assert "ON CONFLICT (source_table, scope_key)" in sql
    assert "GREATEST" in sql
    assert "lease_owner = %s" in sql
    assert "lease_expires_at > now()" in sql
    assert pool.connection_instance.transaction_entries == 1


def test_finish_failed_never_advances_watermarks_and_sanitizes_error():
    pool = _Pool([(RUN_ID,)])
    store = IngestRunStore(pool)

    asyncio.run(
        store.finish_failed(
            RUN_ID,
            "worker-1",
            error_code="DB_FAILURE",
            error_message="x" * 1200,
        )
    )

    sql, params = pool.connection_instance.executed[0]
    assert "status = %s" in sql
    assert params[0] == "FAILED"
    assert "ai_ingest_watermarks" not in sql
    assert len(params[2]) == 1000
    assert "lease_expires_at > now()" in sql


def test_manual_data_terminal_requires_unexpired_owned_lease():
    pool = _Pool([(RUN_ID,)])
    store = IngestRunStore(pool)

    asyncio.run(
        store.finish_manual_data_required(
            RUN_ID,
            "worker-1",
            {
                "code": "MANUAL_BASEBALL_DATA_REQUIRED",
                "entity": "game",
                "missing_fields": ["game_date"],
            },
        )
    )

    sql = pool.connection_instance.executed[0][0]
    assert "status = 'RUNNING'" in sql
    assert "lease_owner = %s" in sql
    assert "lease_expires_at > now()" in sql


def test_recover_expired_increments_before_requeue_and_fails_exhausted_runs():
    pool = _Pool([[(RUN_ID,)], [(FAILED_RUN_ID,)]])
    store = IngestRunStore(pool, max_recovery_attempts=1)

    recovered, failed = asyncio.run(store.recover_expired())

    assert recovered == 1
    assert failed == 1
    sql = " ".join(item[0] for item in pool.connection_instance.executed)
    assert "recovery_attempts = recovery_attempts + 1" in sql
    assert "status = 'QUEUED'" in sql
    assert "status = 'FAILED'" in sql


def test_count_active_by_status_groups_persisted_queue_and_running_rows():
    pool = _Pool(
        [[
            {
                "status": "QUEUED",
                "trigger_source": "BACKEND_SCHEDULED",
                "run_count": 2,
            },
            {
                "status": "RUNNING",
                "trigger_source": "MANUAL_API",
                "run_count": 1,
            },
        ]]
    )
    store = IngestRunStore(pool)

    counts = asyncio.run(store.count_active_by_status())

    assert counts == {
        ("QUEUED", "BACKEND_SCHEDULED"): 2,
        ("RUNNING", "MANUAL_API"): 1,
    }
    sql = pool.connection_instance.executed[0][0]
    assert "status IN ('QUEUED', 'RUNNING')" in sql
    assert "GROUP BY status, trigger_source" in sql


def test_get_latest_watermarks_by_table_aggregates_all_scopes():
    pool = _Pool(
        [[
            {
                "source_table": "game",
                "last_successful_updated_at": WATERMARK,
            }
        ]]
    )
    store = IngestRunStore(pool)

    watermarks = asyncio.run(store.get_latest_watermarks_by_table())

    assert watermarks == {"game": WATERMARK}
    sql = pool.connection_instance.executed[0][0]
    assert "max(last_successful_updated_at)" in sql.lower()
    assert "GROUP BY source_table" in sql


def test_get_watermark_returns_last_successful_timestamp():
    pool = _Pool([(WATERMARK,)])
    store = IngestRunStore(pool)

    assert asyncio.run(store.get_watermark("game", WATERMARK_SCOPE)) == WATERMARK


def test_advance_watermark_is_monotonic_and_scope_partitioned():
    pool = _Pool([None])
    store = IngestRunStore(pool)

    asyncio.run(store.advance_watermark("game", WATERMARK_SCOPE, WATERMARK, RUN_ID))

    sql, params = pool.connection_instance.executed[0]
    assert "ON CONFLICT (source_table, scope_key)" in sql
    assert "GREATEST" in sql
    assert params == ("game", WATERMARK_SCOPE, WATERMARK, RUN_ID)
