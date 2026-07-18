# AI Ingestion Heartbeat and Lease Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep durable AI ingestion leases alive through transient PostgreSQL coordination failures while preventing expired workers from renewing, writing terminal state, or competing with recovery.

**Architecture:** Move every `IngestRunStore` operation to a dedicated 1–2 connection asynchronous PostgreSQL pool. Renew heartbeats with lease-budgeted exponential retry, enforce `lease_expires_at > now()` on every owner mutation, and preserve the existing per-batch write fence and recovery loop.

**Tech Stack:** Python 3.11+, FastAPI lifespan, `asyncio`, psycopg 3, `psycopg_pool.AsyncConnectionPool`, Prometheus client, pytest.

## Global Constraints

- Baseball facts may come only from the internal database, trusted internal sync paths, static project documents, or operator-provided manual data.
- Never add crawling, scraping, public baseball API, web-search repair, or synthesized baseball facts.
- Missing or inconsistent baseball data must keep the `MANUAL_BASEBALL_DATA_REQUIRED` contract.
- This plan makes no database schema migration and adds no persistent batch checkpoint.
- This plan makes no external embedding request and writes no shared production database.
- Preserve the synchronous renderer, chunker, embedding, `rag_chunks` write path, and current recovery-attempt semantics.
- Use test-first red-green-refactor cycles and keep each task in its own commit.

## File Structure

- Modify `app/core/ingest_run_store.py`: make PostgreSQL authoritative for unexpired ownership and return renewed lease expiry from heartbeat.
- Modify `tests/test_ingest_run_store.py`: prove heartbeat and all terminal mutations reject expired leases.
- Modify `app/core/ingest_worker.py`: add bounded heartbeat retry, cancellation-safe loss handling, and terminal-race handling.
- Modify `tests/test_ingest_worker.py`: prove transient recovery, exhaustion, rejection, cancellation, long-running heartbeat, and terminal fencing.
- Modify `app/observability/metrics.py`: define the bounded heartbeat result counter.
- Modify `tests/test_observability_metrics.py`: lock the heartbeat metric to the single `result` label.
- Modify `app/deps.py`: create, open, inject, and close the dedicated ingestion coordination pool.
- Modify `tests/test_deps_db_pool_observability.py`: prove pool separation, size limits, startup cleanup, and idempotent close.
- Modify `tests/test_schema_startup_mode.py`: prove `IngestRunStore` receives the coordination pool.
- Modify `docs/data-sync-orchestration-runbook.md`: document retry, strict expiry, dedicated pool, heartbeat metrics, and rollback behavior.

---

### Task 1: Enforce unexpired ownership in the run store

**Files:**
- Modify: `tests/test_ingest_run_store.py:25-190`
- Modify: `app/core/ingest_run_store.py:214-376`

**Interfaces:**
- Consumes: `IngestRunStore.pool`, `IngestRunStore.lease_seconds`, `IngestLeaseLostError` through `_require_owned_run`.
- Produces: `IngestRunStore.heartbeat(run_id: UUID, owner: str) -> datetime | None`; strict unexpired lease predicates for heartbeat and terminal writes.

- [ ] **Step 1: Write failing store fence tests**

Add the following heartbeat test after `test_claim_next_uses_skip_locked_and_assigns_lease_owner`, then extend the existing success and failure assertions as shown:

```python
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
```

Add these assertions to `test_finish_success_advances_only_committed_table_watermarks`:

```python
    assert "lease_owner = %s" in sql
    assert "lease_expires_at > now()" in sql
```

Add this assertion to `test_finish_failed_never_advances_watermarks_and_sanitizes_error`:

```python
    assert "lease_expires_at > now()" in sql
```

Add a manual-data terminal test to prove the shared terminal helper is also fenced:

```python
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
```

- [ ] **Step 2: Run the store tests and confirm red**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_run_store.py -q
```

Expected: FAIL because heartbeat returns `True` instead of `WATERMARK`, uses `RETURNING run_id`, and the expiry predicates are absent.

- [ ] **Step 3: Implement strict heartbeat and terminal fences**

Replace `IngestRunStore.heartbeat` with:

```python
    async def heartbeat(self, run_id: UUID, owner: str) -> datetime | None:
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    UPDATE ai_ingest_runs
                    SET heartbeat_at = now(),
                        lease_expires_at = now() + make_interval(secs => %s),
                        updated_at = now()
                    WHERE run_id = %s
                      AND status = 'RUNNING'
                      AND lease_owner = %s
                      AND lease_expires_at > now()
                    RETURNING lease_expires_at
                    """,
                    (self.lease_seconds, run_id, owner),
                )
            ).fetchone()
        if row is None:
            return None
        if isinstance(row, Mapping):
            return row["lease_expires_at"]
        return row[0]
```

Add the following predicate immediately after `AND lease_owner = %s` in both the success update and `_finish_terminal` update:

```sql
AND lease_expires_at > now()
```

Do not change `recover_expired`; PostgreSQL expiry remains its source of truth.

- [ ] **Step 4: Run store tests and confirm green**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_run_store.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the store fence**

```bash
git add app/core/ingest_run_store.py tests/test_ingest_run_store.py
git commit -m "fix: fence expired ingest leases"
```

---

### Task 2: Retry transient heartbeats within the lease budget

**Files:**
- Modify: `tests/test_observability_metrics.py:84-121`
- Modify: `app/observability/metrics.py:208-235`
- Modify: `tests/test_ingest_worker.py:1-410`
- Modify: `app/core/ingest_worker.py:20-145,253-274`

**Interfaces:**
- Consumes: `IngestRunStore.heartbeat(run_id, owner) -> datetime | None`, `IngestLeaseLostError`, and the existing synchronous per-write lease guard.
- Produces: `AI_INGEST_HEARTBEATS_TOTAL{result}`, `IngestWorker.heartbeat_interval_seconds`, `heartbeat_safety_margin_seconds`, and `heartbeat_retry_initial_seconds`.

- [ ] **Step 1: Write the failing bounded-metric test**

Import `AI_INGEST_HEARTBEATS_TOTAL` inside `test_ingest_metrics_use_only_bounded_labels` and add:

```python
    assert AI_INGEST_HEARTBEATS_TOTAL._labelnames == ("result",)
```

- [ ] **Step 2: Run the metric test and confirm red**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_observability_metrics.py::test_ingest_metrics_use_only_bounded_labels -q
```

Expected: FAIL with an import error because `AI_INGEST_HEARTBEATS_TOTAL` is not defined.

- [ ] **Step 3: Add the heartbeat metric**

Add immediately after `AI_INGEST_LEASE_RECOVERIES_TOTAL`:

```python
AI_INGEST_HEARTBEATS_TOTAL = Counter(
    "ai_ingest_heartbeats_total",
    "Durable ingestion heartbeat outcomes.",
    ["result"],
)
```

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_observability_metrics.py::test_ingest_metrics_use_only_bounded_labels -q
```

Expected: PASS.

- [ ] **Step 4: Write failing worker retry tests**

Add `import time` and import `IngestLeaseLostError` in `tests/test_ingest_worker.py`. Change `_Store.heartbeat` to return `NOW` for valid ownership and `None` otherwise:

```python
    async def heartbeat(self, run_id, owner):
        self.heartbeat_calls += 1
        if run_id == RUN_ID and owner == "worker-1":
            return NOW
        return None
```

Add this scripted store and wait helper below `_Store`:

```python
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
```

Add the following tests:

```python
def test_heartbeat_retries_transient_error_without_losing_lease():
    store = _ScriptedHeartbeatStore([RuntimeError("temporary db failure"), NOW])
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
    store = _ScriptedHeartbeatStore([RuntimeError("db unavailable")] * 20)
    worker = IngestWorker(store=store, settings=SETTINGS, owner="worker-1")
    worker.lease_seconds = 0.02
    worker.heartbeat_interval_seconds = 0.001
    worker.heartbeat_safety_margin_seconds = 0.01
    worker.heartbeat_retry_initial_seconds = 0.001
    lease_lost = asyncio.Event()
    exhausted_before = _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "exhausted"}
    )

    async def scenario():
        await asyncio.wait_for(worker._heartbeat_loop(RUN, lease_lost), timeout=0.2)

    asyncio.run(scenario())

    assert lease_lost.is_set() is True
    assert store.heartbeat_calls >= 2
    assert _read_metric_value(
        "ai_ingest_heartbeats_total", {"result": "exhausted"}
    ) - exhausted_before == 1


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

    def slow_ingest(**kwargs):
        assert kwargs["lease_run_id"] == RUN_ID
        time.sleep(0.08)
        return EXECUTION_RESULT

    worker = IngestWorker(
        store=store,
        settings=SETTINGS,
        owner="worker-1",
        ingest_function=slow_ingest,
    )
    worker.lease_seconds = 0.06
    worker.heartbeat_interval_seconds = 0.01
    worker.heartbeat_safety_margin_seconds = 0.01

    assert asyncio.run(worker.run_once()) is True
    assert store.successful_run_id == RUN_ID
    assert store.heartbeat_calls >= 2
```

Add a terminal-race test:

```python
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
```

- [ ] **Step 5: Run worker tests and confirm red**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_worker.py -q
```

Expected: FAIL because heartbeat exceptions still set lease loss immediately, timing attributes do not exist, and a terminal lease race escapes `run_once`.

- [ ] **Step 6: Implement bounded retry and terminal-race handling**

Import `AI_INGEST_HEARTBEATS_TOTAL` beside the existing ingest metrics. Add these attributes in `IngestWorker.__init__` after `lease_seconds`:

```python
        self.heartbeat_interval_seconds = max(1.0, self.lease_seconds / 3)
        self.heartbeat_safety_margin_seconds = min(
            5.0,
            max(0.25, self.lease_seconds / 6),
        )
        self.heartbeat_retry_initial_seconds = min(
            1.0,
            self.heartbeat_interval_seconds,
        )
```

Replace `_heartbeat_loop` with:

```python
    async def _heartbeat_loop(
        self,
        run: IngestRunRecord,
        lease_lost: asyncio.Event,
    ) -> None:
        loop = asyncio.get_running_loop()
        last_confirmed = loop.time()
        interval = max(0.001, float(self.heartbeat_interval_seconds))
        retry_initial = max(0.001, float(self.heartbeat_retry_initial_seconds))
        safety_margin = max(
            0.0,
            min(float(self.heartbeat_safety_margin_seconds), self.lease_seconds / 2),
        )

        while True:
            await asyncio.sleep(interval)
            retry_delay = retry_initial
            attempt = 0
            while True:
                attempt += 1
                try:
                    lease_expires_at = await self.store.heartbeat(
                        run.run_id,
                        self.owner,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    remaining = (
                        last_confirmed
                        + float(self.lease_seconds)
                        - safety_margin
                        - loop.time()
                    )
                    if remaining <= 0:
                        AI_INGEST_HEARTBEATS_TOTAL.labels(result="exhausted").inc()
                        logger.error(
                            "Ingestion heartbeat retry budget exhausted "
                            "run_id=%s attempts=%d error_type=%s",
                            run.run_id,
                            attempt,
                            type(exc).__name__,
                        )
                        lease_lost.set()
                        return
                    AI_INGEST_HEARTBEATS_TOTAL.labels(result="retry").inc()
                    logger.warning(
                        "Ingestion heartbeat retry scheduled "
                        "run_id=%s attempt=%d remaining_seconds=%.3f error_type=%s",
                        run.run_id,
                        attempt,
                        remaining,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(min(retry_delay, remaining))
                    retry_delay = min(
                        retry_delay * 2,
                        max(retry_initial, interval),
                    )
                    continue

                if lease_expires_at is None:
                    AI_INGEST_HEARTBEATS_TOTAL.labels(result="rejected").inc()
                    logger.error(
                        "Ingestion run lease rejected run_id=%s",
                        run.run_id,
                    )
                    lease_lost.set()
                    return

                AI_INGEST_HEARTBEATS_TOTAL.labels(result="success").inc()
                last_confirmed = loop.time()
                break
```

In `run_once`, wrap each terminal store call so `IngestLeaseLostError` is treated as ownership handoff. Use this pattern for manual-data and success transitions:

```python
            try:
                await self.store.finish_manual_data_required(
                    run.run_id,
                    self.owner,
                    exc.contract,
                )
            except IngestLeaseLostError:
                logger.error(
                    "Ingestion manual terminal skipped after lease loss run_id=%s",
                    run.run_id,
                )
            else:
                terminal_status = IngestRunStatus.MANUAL_BASEBALL_DATA_REQUIRED
```

```python
            try:
                await self.store.finish_success(
                    run.run_id,
                    self.owner,
                    result.tables,
                    result.watermarks,
                    build_watermark_scope_key(run.request),
                )
            except IngestLeaseLostError:
                logger.error(
                    "Ingestion success terminal skipped after lease loss run_id=%s",
                    run.run_id,
                )
            else:
                terminal_status = IngestRunStatus.SUCCEEDED
```

For generic execution failures, skip `finish_failed` when `lease_lost` is already set and otherwise catch a terminal race:

```python
            if lease_lost.is_set():
                logger.error(
                    "Ingestion failure terminal skipped after lease loss run_id=%s",
                    run.run_id,
                )
            else:
                try:
                    await self.store.finish_failed(
                        run.run_id,
                        self.owner,
                        error_code="INGEST_EXECUTION_FAILED",
                        error_message=error_type,
                    )
                except IngestLeaseLostError:
                    logger.error(
                        "Ingestion failure terminal rejected after lease loss run_id=%s",
                        run.run_id,
                    )
                else:
                    terminal_status = IngestRunStatus.FAILED
```

- [ ] **Step 7: Run focused worker and metric tests**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_worker.py tests/test_observability_metrics.py -q
```

Expected: all tests PASS, including the long synchronous fake ingest without external HTTP calls.

- [ ] **Step 8: Commit heartbeat resilience**

```bash
git add app/core/ingest_worker.py app/observability/metrics.py tests/test_ingest_worker.py tests/test_observability_metrics.py
git commit -m "fix: retry transient ingest heartbeats"
```

---

### Task 3: Isolate ingestion coordination in a dedicated pool

**Files:**
- Modify: `tests/test_deps_db_pool_observability.py:1-110`
- Modify: `tests/test_schema_startup_mode.py:85-105`
- Modify: `app/deps.py:35-50,213-289,528-641`

**Interfaces:**
- Consumes: `get_settings().database_url`, `_prepare_schema(pool, settings)`, and `IngestRunStore(pool, lease_seconds, max_recovery_attempts)`.
- Produces: `get_ingest_connection_pool() -> AsyncConnectionPool`, `close_ingest_connection_pool() -> None`, and `_prepare_required_database_pools(settings) -> tuple[AsyncConnectionPool, AsyncConnectionPool]`.

- [ ] **Step 1: Write failing pool-isolation and lifecycle tests**

Add `from datetime import UTC, datetime`, `from types import SimpleNamespace`, `from unittest.mock import AsyncMock`, and `from uuid import UUID` to `tests/test_deps_db_pool_observability.py`, then add:

```python
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
```

Change `test_ingest_run_store_uses_worker_lease_and_recovery_settings` in `tests/test_schema_startup_mode.py` to patch and assert the dedicated pool:

```python
    with (
        patch.object(deps, "get_settings", return_value=settings),
        patch.object(deps, "get_ingest_connection_pool", return_value=pool),
    ):
        store = deps.get_ingest_run_store()

    assert store.pool is pool
```

Add `import inspect` to `tests/test_schema_startup_mode.py` and add this shutdown-order contract test:

```python
def test_lifespan_stops_ingest_tasks_before_closing_coordination_pool():
    source = inspect.getsource(deps.lifespan)

    assert source.index("ingest_worker_task.cancel()") < source.index(
        "await close_ingest_connection_pool()"
    )
    assert source.index("ingest_recovery_task.cancel()") < source.index(
        "await close_ingest_connection_pool()"
    )
```

- [ ] **Step 2: Run dependency tests and confirm red**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_deps_db_pool_observability.py tests/test_schema_startup_mode.py -q
```

Expected: FAIL because the ingestion pool functions, global, constants, and startup helper do not exist and the store still receives the general pool.

- [ ] **Step 3: Implement the pool factory and dedicated singleton**

Add the dedicated global and limits beside the existing pool constants:

```python
_connection_pool: Optional[AsyncConnectionPool] = None
_ingest_connection_pool: Optional[AsyncConnectionPool] = None
DB_POOL_MIN_SIZE = 1
DB_POOL_MAX_SIZE = 30
INGEST_DB_POOL_MIN_SIZE = 1
INGEST_DB_POOL_MAX_SIZE = 2
```

Extract the common constructor and use it from both getters:

```python
def _create_async_connection_pool(
    *,
    min_size: int,
    max_size: int,
) -> AsyncConnectionPool:
    settings = get_settings()
    return AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=min_size,
        max_size=max_size,
        check=AsyncConnectionPool.check_connection,
        open=False,
        kwargs={
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
            "autocommit": True,
            "target_session_attrs": "read-write",
        },
    )


def get_connection_pool() -> AsyncConnectionPool:
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = _create_async_connection_pool(
            min_size=DB_POOL_MIN_SIZE,
            max_size=DB_POOL_MAX_SIZE,
        )
        logger.info(
            "[DB] General connection pool created (open pending) pool_stats=%s",
            _format_connection_pool_stats(_connection_pool),
        )
    return _connection_pool


def get_ingest_connection_pool() -> AsyncConnectionPool:
    global _ingest_connection_pool
    if _ingest_connection_pool is None:
        _ingest_connection_pool = _create_async_connection_pool(
            min_size=INGEST_DB_POOL_MIN_SIZE,
            max_size=INGEST_DB_POOL_MAX_SIZE,
        )
        logger.info(
            "[DB] Ingest coordination pool created (open pending) pool_stats=%s",
            _format_connection_pool_stats(
                _ingest_connection_pool,
                min_size=INGEST_DB_POOL_MIN_SIZE,
                max_size=INGEST_DB_POOL_MAX_SIZE,
            ),
        )
    return _ingest_connection_pool
```

Extend the pool stats helpers so the dedicated pool reports its true limits without exposing connection information:

```python
def _snapshot_connection_pool_stats(
    pool_instance: Optional[Any] = None,
    *,
    min_size: int = DB_POOL_MIN_SIZE,
    max_size: int = DB_POOL_MAX_SIZE,
) -> dict[str, Any]:
    pool = pool_instance or _connection_pool
    snapshot: dict[str, Any] = {
        "min_size": min_size,
        "max_size": max_size,
        "pool_available": pool is not None,
    }
```

```python
def _format_connection_pool_stats(
    pool_instance: Optional[Any] = None,
    *,
    min_size: int = DB_POOL_MIN_SIZE,
    max_size: int = DB_POOL_MAX_SIZE,
) -> str:
    return json.dumps(
        _snapshot_connection_pool_stats(
            pool_instance,
            min_size=min_size,
            max_size=max_size,
        ),
        ensure_ascii=False,
        sort_keys=True,
    )
```

Add the dedicated close function:

```python
async def close_ingest_connection_pool() -> None:
    global _ingest_connection_pool
    if _ingest_connection_pool is not None:
        await _ingest_connection_pool.close()
        _ingest_connection_pool = None
```

- [ ] **Step 4: Implement fail-fast startup cleanup and store injection**

Add this helper above `get_ingest_run_store`:

```python
async def _prepare_required_database_pools(
    settings: Any,
) -> tuple[AsyncConnectionPool, AsyncConnectionPool]:
    general_pool = get_connection_pool()
    ingest_pool = get_ingest_connection_pool()
    try:
        await general_pool.open(wait=True, timeout=10.0)
        await ingest_pool.open(wait=True, timeout=10.0)
        await _prepare_schema(general_pool, settings)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[Lifespan] required DB pool preparation failed "
            "error_type=%s general_pool_stats=%s ingest_pool_stats=%s",
            type(exc).__name__,
            _format_connection_pool_stats(general_pool),
            _format_connection_pool_stats(
                ingest_pool,
                min_size=INGEST_DB_POOL_MIN_SIZE,
                max_size=INGEST_DB_POOL_MAX_SIZE,
            ),
        )
        await close_ingest_connection_pool()
        await close_connection_pool()
        raise
    logger.info("[Lifespan] required database pools opened")
    return general_pool, ingest_pool
```

Change the store getter to:

```python
def get_ingest_run_store() -> IngestRunStore:
    """Build an ingestion store over its dedicated coordination pool."""

    settings = get_settings()
    return IngestRunStore(
        get_ingest_connection_pool(),
        lease_seconds=settings.ingest_worker_lease_seconds,
        max_recovery_attempts=settings.ingest_worker_max_recovery_attempts,
    )
```

At the start of `lifespan`, replace the current general-pool open and separate `_prepare_schema` call with:

```python
    settings = get_settings()
    load_clf()
    pool, _ = await _prepare_required_database_pools(settings)
    _initialize_shared_baseball_agent_runtime()
```

At shutdown, after all worker and recovery tasks have been cancelled and awaited, close the coordination pool before the general pool:

```python
    reset_shared_baseball_agent_runtime()
    await close_ingest_connection_pool()
    await close_connection_pool()
```

- [ ] **Step 5: Run dependency tests and confirm green**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_deps_db_pool_observability.py tests/test_schema_startup_mode.py tests/test_db_schema_contract.py -q
```

Expected: all tests PASS; no real PostgreSQL connection is opened.

- [ ] **Step 6: Commit coordination pool isolation**

```bash
git add app/deps.py tests/test_deps_db_pool_observability.py tests/test_schema_startup_mode.py
git commit -m "fix: isolate ingest coordination pool"
```

---

### Task 4: Document operations and run the release verification gates

**Files:**
- Modify: `docs/data-sync-orchestration-runbook.md:24-31,94-96,109-132,134-140`

**Interfaces:**
- Consumes: the strict expiry invariant, four heartbeat metric results, 1–2 connection pool limits, and existing recovery behavior implemented in Tasks 1–3.
- Produces: operator guidance for diagnosing retry, rejection, exhaustion, recovery, and rollback.

- [ ] **Step 1: Update the runbook with the exact runtime contract**

Add this paragraph after the worker environment block:

```markdown
AI ingest coordination uses a dedicated PostgreSQL pool with one minimum and two maximum connections per AI process. The pool is not configurable independently: its bounded size is part of the heartbeat isolation contract. Failure to open this required pool fails AI startup instead of starting a worker without durable coordination.
```

Replace the lease paragraph under `## 리스 만료와 재시작 복구` with:

```markdown
AI 서비스는 시작 시와 실행 중 주기적으로 만료된 `RUNNING` lease를 확인합니다. heartbeat는 정상 상태에서 lease 시간의 1/3 간격으로 실행됩니다. 일시적인 PostgreSQL 연결·풀 오류는 마지막으로 확인된 lease의 안전 여유 5초 전까지만 지수 백오프로 재시도합니다. 한 번의 오류만으로 작업을 포기하지 않지만 안전 시간이 끝나면 이전 worker는 lease 상실 상태가 됩니다.

heartbeat와 성공·실패·`MANUAL_BASEBALL_DATA_REQUIRED` 종료는 모두 DB에서 `RUNNING`, owner 일치, `lease_expires_at > now()`를 만족해야 합니다. 만료된 owner는 recovery가 실행되기 전이라도 lease를 되살리거나 terminal 상태를 기록할 수 없습니다. 동기 ingest 경로는 각 쓰기 배치 직전에 같은 DB owner·만료 조건을 확인하고 run row를 배치 commit까지 잠급니다.

`AI_INGEST_WORKER_MAX_RECOVERY_ATTEMPTS` 미만이면 만료 실행을 `QUEUED`로 되돌리고, 한도에 도달하면 `FAILED` 및 `INGEST_LEASE_EXPIRED`로 종결합니다. watermark는 시즌과 명시적 `since` 범위별로 분리되고 `SUCCEEDED` 트랜잭션에서만 단조 증가합니다. 이번 안정화에는 배치 checkpoint가 포함되지 않으므로 recovery 실행은 이미 커밋된 청크를 content hash 기반으로 재확인할 수 있습니다.
```

Add the heartbeat metric to the Prometheus list:

```markdown
- `ai_ingest_heartbeats_total{result}`: `success`, `retry`, `rejected`, `exhausted` heartbeat 결과
```

Add this rollback note before the numbered rollback steps:

```markdown
전용 coordination pool 변경만 비활성화하는 환경 플래그는 없습니다. 코드 롤백 시 이전 버전의 단일 공용 풀이 복원되지만 `ai_ingest_runs`, watermark, `rag_chunks` 데이터는 그대로 호환됩니다.
```

- [ ] **Step 2: Run the focused resilience suite**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_ingest_runs.py \
  tests/test_ingest_results.py \
  tests/test_ingest_run_store.py \
  tests/test_ingest_worker.py \
  tests/test_schema_startup_mode.py \
  tests/test_deps_db_pool_observability.py \
  tests/test_observability_metrics.py \
  tests/test_db_schema_contract.py \
  -q
```

Expected: all selected tests PASS.

- [ ] **Step 3: Run the complete AI test suite**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/ -q
```

Expected: exit code 0 with no failed tests. Tests that require unavailable optional services may be skipped only when their existing skip condition explains the dependency.

- [ ] **Step 4: Run compile and baseball-data policy gates**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m compileall -q app scripts
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python ../../../scripts/validate_baseball_data_policy.py
git diff --check HEAD~3..HEAD
git diff --check
git status --short
```

Expected:

- `compileall` exits 0.
- Policy validator reports success and finds no external baseball collection path.
- Both committed-range and working-tree `git diff --check` commands exit 0.
- `git status --short` lists only the runbook change before the final documentation commit.

- [ ] **Step 5: Commit the runbook and verification handoff**

```bash
git add docs/data-sync-orchestration-runbook.md
git commit -m "docs: explain ingest lease resilience"
```

- [ ] **Step 6: Record the final clean-state evidence**

Run:

```bash
git log -5 --oneline
git status --short --branch
```

Expected: the design and implementation-plan documentation commits plus four implementation commits are visible and the isolated worktree is clean.
