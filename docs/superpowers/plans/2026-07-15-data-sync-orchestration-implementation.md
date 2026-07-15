# KBO Internal Data Synchronization Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one durable, observable internal KBO-to-RAG synchronization workflow whose terminal result drives backend cache refresh and whose missing-data failures preserve the `MANUAL_BASEBALL_DATA_REQUIRED` contract through the frontend.

**Architecture:** Backend JobRunr is the only recurring scheduler. It submits an explicit request to a PostgreSQL-backed AI ingestion queue and schedules one-shot status checks instead of blocking a worker. The AI lifespan owns a lease-based worker, persists run history and table watermarks, and executes the existing trusted internal-DB ingestion code in a thread. Backend controllers stop synthesizing baseball fallbacks, and frontend query code preserves distinct empty, manual-data, synchronization-pending, and generic failure states.

**Tech Stack:** Python 3.14, FastAPI, psycopg/psycopg-pool, PostgreSQL/pgvector, pytest, Java 21, Spring Boot, JobRunr, Redis cache abstraction, Gradle/JUnit 5/Mockito, React 19, TypeScript, TanStack Query, Node test runner, Cypress.

## Global Constraints

- Never add crawling, scraping, web-search repair, browser baseball collection, or a public baseball API request.
- Baseball facts may only come from the internal database, trusted internal sync paths, static project documents, or operator-provided manual data.
- Preserve the root `scripts/sync_kbo_data.py` trusted sync path.
- Missing or inconsistent required baseball data must surface `MANUAL_BASEBALL_DATA_REQUIRED` with entity/range/fields/import source.
- Do not add a new external task queue or broker dependency.
- Keep `/ai/ingest/run` and `/ai/ingest/runs/{run_id}` protected by the existing internal token.
- Do not expose raw rows, prompts, embeddings, database URLs, tokens, or operator-provided values in status responses, metrics, or logs.
- Preserve existing unrelated backend `mate` worktree changes.
- Every production behavior change follows RED → GREEN → REFACTOR and is committed independently.

---

### Task 1: AI ingestion schema and schema contract

**Files:**
- Create: `bega_AI/app/db/migrations/003_ai_ingest_orchestration.sql`
- Modify: `bega_AI/scripts/migrate_ai_runtime_schema.sh`
- Modify: `bega_AI/app/db/schema_contract.py`
- Modify: `bega_AI/tests/test_ai_schema_migrations.py`
- Modify: `bega_AI/tests/test_db_schema_contract.py`

**Interfaces:**
- Consumes: existing `validate_schema_contract(conn, require_vector_index=...)`.
- Produces: PostgreSQL tables `ai_ingest_runs`, `ai_ingest_watermarks`, and indexes `ux_ai_ingest_runs_active_request`, `idx_ai_ingest_runs_status_requested`.

- [ ] **Step 1: Write failing migration assertions**

Add assertions to `tests/test_ai_schema_migrations.py` that the migration contains the exact run states and active-request uniqueness rule:

```python
def test_ingest_orchestration_migration_defines_durable_run_and_watermark_tables():
    sql = (MIGRATIONS_DIR / "003_ai_ingest_orchestration.sql").read_text()
    assert "CREATE TABLE IF NOT EXISTS ai_ingest_runs" in sql
    assert "CREATE TABLE IF NOT EXISTS ai_ingest_watermarks" in sql
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in sql
    assert "WHERE status IN ('QUEUED', 'RUNNING')" in sql
```

Add `ai_ingest_runs` and `ai_ingest_watermarks` expected columns to `tests/test_db_schema_contract.py`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd bega_AI
./.venv/bin/python -m pytest tests/test_ai_schema_migrations.py tests/test_db_schema_contract.py -q
```

Expected: FAIL because migration `003_ai_ingest_orchestration.sql` and schema contract entries do not exist.

- [ ] **Step 3: Add the migration**

Create the migration with this schema:

```sql
CREATE TABLE IF NOT EXISTS ai_ingest_runs (
    run_id uuid PRIMARY KEY,
    request_key varchar(64) NOT NULL,
    trigger_source varchar(32) NOT NULL,
    status varchar(48) NOT NULL,
    request_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    requested_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    heartbeat_at timestamptz,
    finished_at timestamptz,
    lease_owner varchar(128),
    lease_expires_at timestamptz,
    recovery_attempts integer NOT NULL DEFAULT 0,
    error_code varchar(96),
    error_message varchar(1000),
    table_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ck_ai_ingest_runs_status CHECK (
        status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'MANUAL_BASEBALL_DATA_REQUIRED')
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_ai_ingest_runs_active_request
    ON ai_ingest_runs (request_key)
    WHERE status IN ('QUEUED', 'RUNNING');

CREATE INDEX IF NOT EXISTS idx_ai_ingest_runs_status_requested
    ON ai_ingest_runs (status, requested_at);

CREATE TABLE IF NOT EXISTS ai_ingest_watermarks (
    source_table varchar(128) PRIMARY KEY,
    last_successful_updated_at timestamptz,
    last_run_id uuid REFERENCES ai_ingest_runs(run_id),
    updated_at timestamptz NOT NULL DEFAULT now()
);
```

Update `migrate_ai_runtime_schema.sh` so migration 003 always runs after migration 001. Add both tables and their exact columns to `REQUIRED_COLUMNS` in `schema_contract.py`.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C bega_AI add app/db/migrations/003_ai_ingest_orchestration.sql app/db/schema_contract.py scripts/migrate_ai_runtime_schema.sh tests/test_ai_schema_migrations.py tests/test_db_schema_contract.py
git -C bega_AI commit -m "feat: add durable AI ingest run schema"
```

### Task 2: AI ingestion run domain and deterministic request identity

**Files:**
- Create: `bega_AI/app/core/ingest_runs.py`
- Create: `bega_AI/tests/test_ingest_runs.py`

**Interfaces:**
- Consumes: normalized table names and optional season/range values.
- Produces: `IngestRunMode`, `IngestRunStatus`, `IngestRunRequest`, `IngestRunRecord`, `IngestTableResult`, `build_request_key(request) -> str`, and legal transition validation.

- [ ] **Step 1: Write failing domain tests**

Create tests for normalized identity and state transitions:

```python
def test_request_key_is_stable_for_equivalent_table_order():
    left = IngestRunRequest(tables=("game", "teams"), season_year=2026)
    right = IngestRunRequest(tables=("teams", "game"), season_year=2026)
    assert build_request_key(left) == build_request_key(right)

def test_terminal_run_cannot_transition_back_to_running():
    with pytest.raises(ValueError, match="illegal ingest run transition"):
        ensure_transition(IngestRunStatus.SUCCEEDED, IngestRunStatus.RUNNING)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd bega_AI
./.venv/bin/python -m pytest tests/test_ingest_runs.py -q
```

Expected: import failure for `app.core.ingest_runs`.

- [ ] **Step 3: Implement immutable domain types**

Use string enums and frozen dataclasses. `IngestRunRequest.normalized()` must sort/deduplicate tables, reject `rag_chunks`, require at least one table, uppercase `mode`/`trigger_source`, and preserve an ISO timestamp when supplied. Hash canonical JSON using SHA-256:

```python
def build_request_key(request: IngestRunRequest) -> str:
    payload = json.dumps(
        request.normalized().to_payload(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

Allow transitions `QUEUED -> RUNNING`, `RUNNING -> SUCCEEDED|FAILED|MANUAL_BASEBALL_DATA_REQUIRED`, and expired-lease `RUNNING -> QUEUED`. All terminal statuses reject further transitions.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C bega_AI add app/core/ingest_runs.py tests/test_ingest_runs.py
git -C bega_AI commit -m "feat: define AI ingest run state machine"
```

### Task 3: AI PostgreSQL run store, deduplication, leases, and watermarks

**Files:**
- Create: `bega_AI/app/core/ingest_run_store.py`
- Create: `bega_AI/tests/test_ingest_run_store.py`

**Interfaces:**
- Consumes: `AsyncConnectionPool` and Task 2 domain types.
- Produces: `IngestRunStore.create_or_get_active`, `get`, `claim_next`, `heartbeat`, `finish_success`, `finish_failed`, `finish_manual_data_required`, `recover_expired`, `get_watermark`, and `advance_watermark`.

- [ ] **Step 1: Write failing SQL-behavior tests**

Use the existing async fake-connection style and assert:

```python
def test_create_or_get_active_returns_existing_run_on_unique_conflict():
    store = IngestRunStore(FakePool(existing_run=EXISTING))
    record, deduplicated = asyncio.run(store.create_or_get_active(REQUEST))
    assert record.run_id == EXISTING.run_id
    assert deduplicated is True

def test_finish_success_advances_only_committed_table_watermarks():
    store = IngestRunStore(FakePool())
    asyncio.run(store.finish_success(RUN_ID, {"game": RESULT}, {"game": WATERMARK}))
    assert "INSERT INTO ai_ingest_watermarks" in store.pool.executed_sql
```

Also test that `finish_failed` never writes `ai_ingest_watermarks`, and an expired lease increments `recovery_attempts` before requeue.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd bega_AI
./.venv/bin/python -m pytest tests/test_ingest_run_store.py -q
```

Expected: import failure for `ingest_run_store`.

- [ ] **Step 3: Implement transactional store methods**

`claim_next` must use one transaction with:

```sql
SELECT run_id
FROM ai_ingest_runs
WHERE status = 'QUEUED'
ORDER BY requested_at
FOR UPDATE SKIP LOCKED
LIMIT 1
```

Then update the selected row to `RUNNING` with `lease_owner`, `heartbeat_at`, and `lease_expires_at`. `finish_success` must update run status and watermarks in one transaction. Sanitize error text to 1000 characters and never store `repr(connection)` or request secrets.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C bega_AI add app/core/ingest_run_store.py tests/test_ingest_run_store.py
git -C bega_AI commit -m "feat: persist AI ingest runs and watermarks"
```

### Task 4: Structured ingestion results and manual-data failures

**Files:**
- Modify: `bega_AI/scripts/ingest_from_kbo.py`
- Modify: `bega_AI/tests/test_ingest_from_kbo_helpers.py`
- Modify: `bega_AI/tests/test_ingest_query.py`
- Create: `bega_AI/tests/test_ingest_results.py`

**Interfaces:**
- Consumes: existing table profiles, chunking, embedding reuse, and UPSERT logic.
- Produces: `ingest_table(...) -> IngestTableResult`, `ingest(...) -> IngestExecutionResult`, and `ManualBaseballDataRequiredError` with a sanitized contract payload.

- [ ] **Step 1: Write failing structured-result tests**

```python
def test_ingest_returns_per_table_counts(monkeypatch):
    monkeypatch.setattr(module, "ingest_table", lambda *args, **kwargs: IngestTableResult("game", 3, 4, 0, 0, None))
    result = module.ingest(tables=["game"], source_db_url="postgresql://internal", **OPTIONS)
    assert result.tables["game"].written_chunks == 3

def test_missing_required_source_column_raises_manual_contract():
    with pytest.raises(ManualBaseballDataRequiredError) as raised:
        validate_required_source_columns("game", {"game_id"}, {"game_id", "game_date"})
    assert raised.value.contract["code"] == "MANUAL_BASEBALL_DATA_REQUIRED"
    assert raised.value.contract["missing_fields"] == ["game_date"]
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd bega_AI
./.venv/bin/python -m pytest tests/test_ingest_results.py tests/test_ingest_query.py tests/test_ingest_from_kbo_helpers.py -q
```

Expected: missing result classes and validation function.

- [ ] **Step 3: Refactor without changing source behavior**

Add dataclasses for table/run counts. Return counts now printed by `ingest_table`; aggregate them in `ingest`. Before executing a profile query, compare configured required output fields with cursor metadata or `information_schema.columns`. Only structural absence raises `ManualBaseballDataRequiredError`; zero rows remain a valid result. Preserve CLI output by printing from the returned result.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command and existing ingestion helper suite. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C bega_AI add scripts/ingest_from_kbo.py tests/test_ingest_results.py tests/test_ingest_query.py tests/test_ingest_from_kbo_helpers.py
git -C bega_AI commit -m "refactor: return structured RAG ingest results"
```

### Task 5: AI lease worker, lifespan recovery, and worker configuration

**Files:**
- Create: `bega_AI/app/core/ingest_worker.py`
- Modify: `bega_AI/app/config.py`
- Modify: `bega_AI/app/deps.py`
- Create: `bega_AI/tests/test_ingest_worker.py`
- Modify: `bega_AI/tests/test_schema_startup_mode.py`

**Interfaces:**
- Consumes: `IngestRunStore`, Task 4 `ingest`, settings database URL.
- Produces: `IngestWorker.run_once() -> bool`, `run_forever(stop_event)`, and settings `ingest_worker_enabled`, `ingest_worker_poll_seconds`, `ingest_worker_lease_seconds`, `ingest_worker_max_recovery_attempts`.

- [ ] **Step 1: Write failing worker tests**

```python
def test_run_once_finishes_success_and_advances_watermarks(monkeypatch):
    worker = IngestWorker(store=FakeStore(claimed=RUN), settings=SETTINGS, owner="worker-1")
    monkeypatch.setattr(worker, "_execute", AsyncMock(return_value=EXECUTION_RESULT))
    assert asyncio.run(worker.run_once()) is True
    assert worker.store.successful_run_id == RUN.run_id

def test_manual_data_error_becomes_manual_terminal_status(monkeypatch):
    worker = IngestWorker(store=FakeStore(claimed=RUN), settings=SETTINGS, owner="worker-1")
    monkeypatch.setattr(worker, "_execute", AsyncMock(side_effect=ManualBaseballDataRequiredError(CONTRACT)))
    asyncio.run(worker.run_once())
    assert worker.store.manual_contract["code"] == "MANUAL_BASEBALL_DATA_REQUIRED"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd bega_AI
./.venv/bin/python -m pytest tests/test_ingest_worker.py tests/test_schema_startup_mode.py -q
```

Expected: missing worker module and settings.

- [ ] **Step 3: Implement worker and lifecycle task**

Use `asyncio.to_thread(ingest, ...)` for the synchronous DB/embedding pipeline. Heartbeat at less than half the lease interval. In `deps.lifespan`, recover expired runs before starting one worker task when enabled; cancel and await it during shutdown alongside existing cleanup tasks. Default the worker to enabled outside tests and allow `AI_INGEST_WORKER_ENABLED=false`.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: PASS without leaked asyncio tasks.

- [ ] **Step 5: Commit**

```bash
git -C bega_AI add app/core/ingest_worker.py app/config.py app/deps.py tests/test_ingest_worker.py tests/test_schema_startup_mode.py
git -C bega_AI commit -m "feat: run durable AI ingestion worker"
```

### Task 6: AI run submission/status API and low-cardinality metrics

**Files:**
- Modify: `bega_AI/app/routers/ingest.py`
- Modify: `bega_AI/app/deps.py`
- Modify: `bega_AI/app/observability/metrics.py`
- Modify: `bega_AI/tests/test_ingest_router.py`
- Modify: `bega_AI/tests/test_observability_metrics.py`

**Interfaces:**
- Consumes: Task 2 request types and Task 3 store.
- Produces: HTTP 202 submission response `{run_id,status,deduplicated}` and protected GET `/ai/ingest/runs/{run_id}`.

- [ ] **Step 1: Replace the old background-task test with failing durable API tests**

```python
def test_run_ingestion_job_persists_queue_request_without_background_task(monkeypatch):
    store = FakeRunStore(created=QUEUED_RUN)
    response = asyncio.run(ingest.run_ingestion_job(PAYLOAD, store, None, None))
    assert response.status_code == 202
    assert json.loads(response.body) == {"run_id": str(RUN_ID), "status": "QUEUED", "deduplicated": False}

def test_get_ingestion_run_returns_404_for_unknown_run():
    with pytest.raises(HTTPException) as raised:
        asyncio.run(ingest.get_ingestion_run(uuid4(), FakeRunStore(created=None), None, None))
    assert raised.value.status_code == 404
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd bega_AI
./.venv/bin/python -m pytest tests/test_ingest_router.py tests/test_observability_metrics.py -q
```

Expected: old `BackgroundTasks` response and no GET handler.

- [ ] **Step 3: Implement API and metrics**

Remove `BackgroundTasks` from `/run`. Normalize payload, call `create_or_get_active`, return `JSONResponse(status_code=202, ...)`, and add a GET route that returns only sanitized status fields. Add counters/gauges/histograms with bounded labels `status`, `trigger_source`, and configured `source_table`; never label by `run_id`, season, error text, or user data.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command and `tests/test_ingest_router.py`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C bega_AI add app/routers/ingest.py app/deps.py app/observability/metrics.py tests/test_ingest_router.py tests/test_observability_metrics.py
git -C bega_AI commit -m "feat: expose durable AI ingest run contract"
```

### Task 7: Backend typed AI ingestion client contract

**Files:**
- Create: `bega_backend/BEGA_PROJECT/src/main/java/com/example/ai/ingest/AiIngestRunRequest.java`
- Create: `bega_backend/BEGA_PROJECT/src/main/java/com/example/ai/ingest/AiIngestRunSubmission.java`
- Create: `bega_backend/BEGA_PROJECT/src/main/java/com/example/ai/ingest/AiIngestRunStatus.java`
- Create: `bega_backend/BEGA_PROJECT/src/main/java/com/example/ai/ingest/AiIngestRunStatusResponse.java`
- Create: `bega_backend/BEGA_PROJECT/src/main/java/com/example/ai/ingest/RagIngestionPort.java`
- Modify: `bega_backend/BEGA_PROJECT/src/main/java/com/example/ai/adapter/AiRagIngestionAdapter.java`
- Delete: `bega_backend/BEGA_PROJECT/src/main/java/com/example/cheerboard/service/port/RagIngestionPort.java`
- Modify: `bega_backend/BEGA_PROJECT/src/test/java/com/example/ai/adapter/AiRagIngestionAdapterTest.java`

**Interfaces:**
- Consumes: AI JSON from Task 6.
- Produces: `submit(AiIngestRunRequest) -> AiIngestRunSubmission` and `getStatus(UUID) -> AiIngestRunStatusResponse`.

- [ ] **Step 1: Write failing adapter contract tests**

Assert the POST body contains explicit `tables`, `season_year`, `mode=INCREMENTAL`, and `trigger_source=BACKEND_SCHEDULED`; assert GET uses `/ai/ingest/runs/{uuid}` with the internal token and deserializes `MANUAL_BASEBALL_DATA_REQUIRED`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd bega_backend/BEGA_PROJECT
./gradlew test --tests "*AiRagIngestionAdapterTest*"
```

Expected: compile failure because the typed contract does not exist.

- [ ] **Step 3: Implement records, enum, port, and adapter**

Define records with Jackson snake-case annotations where necessary. The adapter must reject missing URL/token before making a call, use `X-Internal-Api-Key`, and preserve non-2xx failures. Do not log response payloads or tokens.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C bega_backend/BEGA_PROJECT add src/main/java/com/example/ai src/main/java/com/example/cheerboard/service/port/RagIngestionPort.java src/test/java/com/example/ai/adapter/AiRagIngestionAdapterTest.java
git -C bega_backend/BEGA_PROJECT commit -m "feat: add typed AI ingestion run client"
```

### Task 8: Backend single scheduler, one-shot monitor, and cache invalidation

**Files:**
- Create: `bega_backend/BEGA_PROJECT/src/main/java/com/example/ai/config/AiIngestProperties.java`
- Create: `bega_backend/BEGA_PROJECT/src/main/java/com/example/ai/service/AiIngestOrchestrationService.java`
- Create: `bega_backend/BEGA_PROJECT/src/main/java/com/example/ai/service/BaseballReadCacheInvalidator.java`
- Create: `bega_backend/BEGA_PROJECT/src/main/java/com/example/ai/service/AiIngestRunFailedException.java`
- Create: `bega_backend/BEGA_PROJECT/src/main/java/com/example/ai/service/AiIngestManualDataRequiredException.java`
- Create: `bega_backend/BEGA_PROJECT/src/main/java/com/example/ai/scheduler/AiIngestScheduler.java`
- Delete: `bega_backend/BEGA_PROJECT/src/main/java/com/example/cheerboard/scheduler/AiIngestScheduler.java`
- Delete: `bega_backend/BEGA_PROJECT/src/main/java/com/example/cheerboard/service/AiIntegrationService.java`
- Modify: `bega_backend/BEGA_PROJECT/src/main/resources/application.yml`
- Create: `bega_backend/BEGA_PROJECT/src/test/java/com/example/ai/scheduler/AiIngestSchedulerTest.java`
- Create: `bega_backend/BEGA_PROJECT/src/test/java/com/example/ai/service/AiIngestOrchestrationServiceTest.java`
- Create: `bega_backend/BEGA_PROJECT/src/test/java/com/example/ai/service/BaseballReadCacheInvalidatorTest.java`
- Delete: `bega_backend/BEGA_PROJECT/src/test/java/com/example/cheerboard/service/AiIntegrationServiceTest.java`

**Interfaces:**
- Consumes: Task 7 port.
- Produces: recurring submit job and one-shot `monitor(UUID runId, Instant deadline)` jobs.

- [ ] **Step 1: Write failing orchestration tests**

Tests must prove:

```java
verify(jobScheduler).scheduleRecurrently(
    eq("ai-rag-ingestion"),
    eq("30 4 * * *"),
    any(JobLambda.class));
```

`QUEUED`/`RUNNING` schedules exactly one future monitor using `Instant.now(clock).plus(checkInterval)`. `SUCCEEDED` clears `GAME_SCHEDULE`, `TEAM_RANKINGS`, `HOME_BOOTSTRAP`, `HOME_WIDGETS`, and `HOME_RANKING_SNAPSHOT`. `FAILED` throws `AiIngestRunFailedException`. Manual-data status throws an operator-visible exception containing only the contract code and sanitized message.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd bega_backend/BEGA_PROJECT
./gradlew test --tests "*AiIngestSchedulerTest*" --tests "*AiIngestOrchestrationServiceTest*" --tests "*BaseballReadCacheInvalidatorTest*"
```

Expected: missing classes.

- [ ] **Step 3: Implement one scheduler and non-blocking monitor**

Use `@ConfigurationProperties(prefix="app.ai-ingest")` with defaults: enabled flag `false`, cron `30 4 * * *`, explicit daily table list, check interval 30 seconds, monitoring duration 2 hours. `submitScheduled()` gets a stable run ID and schedules the first monitor. `monitor()` performs one GET and either schedules one next invocation or terminates. It must not call `sleep` or submit a replacement run on status failure. `AiIngestRunFailedException` carries a sanitized terminal error code; `AiIngestManualDataRequiredException` carries only `MANUAL_BASEBALL_DATA_REQUIRED` plus the operator-safe message.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

Stage only AI-package, scheduler, application config, and associated tests; do not stage existing `mate` changes.

```bash
git -C bega_backend/BEGA_PROJECT add src/main/java/com/example/ai src/main/java/com/example/cheerboard/scheduler/AiIngestScheduler.java src/main/java/com/example/cheerboard/service/AiIntegrationService.java src/main/resources/application.yml src/test/java/com/example/ai src/test/java/com/example/cheerboard/service/AiIntegrationServiceTest.java
git -C bega_backend/BEGA_PROJECT commit -m "feat: orchestrate durable AI ingestion runs"
```

### Task 9: Backend baseball error contract without synthesized fallbacks

**Files:**
- Modify: `bega_backend/BEGA_PROJECT/src/main/java/com/example/homepage/HomePageController.java`
- Modify: `bega_backend/BEGA_PROJECT/src/test/java/com/example/homepage/HomePageControllerTest.java`

**Interfaces:**
- Consumes: existing `ManualBaseballDataRequiredException` global handler.
- Produces: legitimate empty arrays only when services return empty; service exceptions remain errors; manual-data exceptions remain HTTP 409 contracts.

- [ ] **Step 1: Change tests first**

Replace fallback expectations so database failures are HTTP 500, add a ranking snapshot manual-data test, and keep the legitimate no-game empty-array test:

```java
mockMvc.perform(get("/api/kbo/schedule").param("date", "2026-03-13"))
    .andExpect(status().isInternalServerError());

mockMvc.perform(get("/api/kbo/rankings/snapshot").param("date", "2026-04-05"))
    .andExpect(status().isConflict())
    .andExpect(jsonPath("$.code").value("MANUAL_BASEBALL_DATA_REQUIRED"));
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd bega_backend/BEGA_PROJECT
./gradlew test --tests "*HomePageControllerTest*"
```

Expected: current handlers return HTTP 200 fallback values.

- [ ] **Step 3: Remove baseball-fact fallback synthesis**

Return service results directly for schedule, rankings, ranking snapshot, league start dates, and navigation. Delete `buildLeagueStartDatesFallback`, `buildRankingSnapshotFallback`, and date-derived season guessing. Let `ManualBaseballDataRequiredException` reach the global handler and let generic infrastructure exceptions become HTTP 500.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C bega_backend/BEGA_PROJECT add src/main/java/com/example/homepage/HomePageController.java src/test/java/com/example/homepage/HomePageControllerTest.java
git -C bega_backend/BEGA_PROJECT commit -m "fix: preserve baseball data failure contracts"
```

### Task 10: Frontend schedule failure and synchronization-state rendering

**Files:**
- Modify: `bega_frontend/src/api/home.ts`
- Modify: `bega_frontend/src/api/home.test.ts`
- Modify: `bega_frontend/src/utils/errorUtils.ts`
- Modify: `bega_frontend/src/components/CheerLivePanel.tsx`
- Modify: `bega_frontend/src/components/CheerLivePanel.test.tsx`

**Interfaces:**
- Consumes: backend `MANUAL_BASEBALL_DATA_REQUIRED`, optional `BASEBALL_DATA_SYNC_PENDING`, and generic non-2xx responses.
- Produces: React Query error state; legitimate `[]` remains empty state.

- [ ] **Step 1: Write failing API and rendering tests**

Add a `fetchGamesData` test whose mocked 503 response rejects with `PublicApiError` instead of resolving `[]`. Add component-helper coverage proving manual-data, sync-pending, and generic copy are different:

```typescript
await assert.rejects(
  () => fetchGamesData(new Date('2026-03-13T12:00:00')),
  (error: unknown) => error instanceof PublicApiError && error.status === 503,
);
```

Expected visible messages:

- manual: existing operator-data message and code token
- pending: `야구 데이터 동기화가 진행 중입니다.`
- generic: `라이브 경기 정보를 불러오지 못했습니다.`

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd bega_frontend
node --test --import tsx src/api/home.test.ts src/components/CheerLivePanel.test.tsx
```

Expected: `fetchGamesData` resolves an empty array for the 503 response and pending copy is absent.

- [ ] **Step 3: Preserve request errors and render typed states**

Delete the broad `try/catch` in `fetchGamesData`. Add `BASEBALL_DATA_SYNC_PENDING_CODE` and a predicate beside existing manual-data helpers. In `CheerLivePanel`, select copy from parsed response code while keeping retry behavior. Do not infer baseball facts from errors.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C bega_frontend add src/api/home.ts src/api/home.test.ts src/utils/errorUtils.ts src/components/CheerLivePanel.tsx src/components/CheerLivePanel.test.tsx
git -C bega_frontend commit -m "fix: distinguish schedule data failures in UI"
```

### Task 11: Recovery command, rollout documentation, and observability assertions

**Files:**
- Modify: `bega_AI/scripts/daily_ingest_kbo.sh`
- Create: `bega_AI/docs/data-sync-orchestration-runbook.md`
- Modify: `bega_AI/README.md`
- Create: `bega_AI/tests/test_daily_ingest_script.py`
- Modify: `bega_backend/BEGA_PROJECT/src/main/resources/META-INF/additional-spring-configuration-metadata.json`
- Modify: `bega_backend/BEGA_PROJECT/src/test/java/com/example/config/ApplicationPropertyCompatibilityTest.java`

**Interfaces:**
- Consumes: all earlier contracts.
- Produces: one documented production scheduler and one explicit manual recovery command.

- [ ] **Step 1: Write failing policy/config tests**

Create `tests/test_daily_ingest_script.py` with assertions that `daily_ingest_kbo.sh` contains `MANUAL RECOVERY ONLY` and warns against installing a second cron. Add backend property-compatibility assertions for enabled, cron, status interval, monitoring duration, and explicit table list.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd bega_backend/BEGA_PROJECT
./gradlew test --tests "*ApplicationPropertyCompatibilityTest*"
cd ../../../bega_AI
./.venv/bin/python -m pytest tests/test_daily_ingest_script.py tests/test_observability_metrics.py -q
```

Expected: configuration metadata and recovery-only documentation are absent.

- [ ] **Step 3: Write exact operations guidance**

Document enable/disable flags, migration order, one-scheduler invariant, queue/run states, manual recovery command, lease recovery, status lookup, cache invalidation, rollback, metrics, and the `MANUAL_BASEBALL_DATA_REQUIRED` operator handoff fields. Remove the production crontab recommendation from `daily_ingest_kbo.sh`; keep the command executable by an operator.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 commands. Expected: PASS.

- [ ] **Step 5: Commit per repository**

```bash
git -C bega_AI add scripts/daily_ingest_kbo.sh docs/data-sync-orchestration-runbook.md README.md tests/test_daily_ingest_script.py
git -C bega_AI commit -m "docs: define AI ingestion recovery operations"
git -C bega_backend/BEGA_PROJECT add src/main/resources/META-INF/additional-spring-configuration-metadata.json src/test/java/com/example/config/ApplicationPropertyCompatibilityTest.java
git -C bega_backend/BEGA_PROJECT commit -m "docs: register AI ingestion scheduler properties"
```

### Task 12: Cross-service review and release verification

**Files:**
- Inspect all files changed by Tasks 1-11.
- Do not modify unrelated `mate` files unless a reviewer identifies a direct compilation conflict.

**Interfaces:**
- Consumes: completed implementations and tests.
- Produces: evidence for every acceptance criterion in the approved design.

- [ ] **Step 1: Run focused AI verification**

```bash
cd bega_AI
./.venv/bin/python -m pytest tests/test_ingest_runs.py tests/test_ingest_run_store.py tests/test_ingest_worker.py tests/test_ingest_router.py tests/test_ingest_results.py tests/test_ai_schema_migrations.py tests/test_db_schema_contract.py -q
```

Expected: all selected tests pass with zero failures.

- [ ] **Step 2: Run AI service suite**

```bash
cd bega_AI
./.venv/bin/python -m pytest tests/ -q
```

Expected: zero failures; environment-dependent tests may skip with recorded reasons.

- [ ] **Step 3: Run backend targeted and safety gates**

```bash
cd bega_backend/BEGA_PROJECT
./gradlew test --tests "*AiRagIngestionAdapterTest*" --tests "*AiIngestSchedulerTest*" --tests "*AiIngestOrchestrationServiceTest*" --tests "*BaseballReadCacheInvalidatorTest*" --tests "*HomePageControllerTest*"
./gradlew migrationSafetyCheck
```

Expected: BUILD SUCCESSFUL for both commands.

- [ ] **Step 4: Run frontend tests and build**

```bash
cd bega_frontend
node --test --import tsx src/api/home.test.ts src/components/CheerLivePanel.test.tsx src/utils/manualBaseballDataContract.test.ts
npm run build
npm run cy:run -- --spec "cypress/e2e/home-scheduled-tab.cy.ts"
```

Expected: tests pass and Vite build exits 0.

- [ ] **Step 5: Run the full backend test suite**

```bash
cd bega_backend/BEGA_PROJECT
./gradlew test
```

Expected: BUILD SUCCESSFUL with zero failed tests.

- [ ] **Step 6: Run baseball data policy gate**

```bash
cd /Users/mac/project/KBO_platform
python3 scripts/validate_baseball_data_policy.py
```

Expected: policy validation passes with no external baseball collection findings.

- [ ] **Step 7: Dispatch repository-required reviews**

Use `code-reviewer` for the cross-service implementation and `security-reviewer` for internal-token endpoints, queue SQL, error payloads, logs, and configuration. Apply only findings that remain within the approved behavior; any new endpoint exposure, token semantics, or data contract expansion requires renewed user confirmation.

- [ ] **Step 8: Audit acceptance criteria against current files and fresh command output**

Confirm one recurring schedule, durable terminal run IDs, active-run uniqueness, restart recovery, watermark success-only advancement, terminal-status backend behavior, cache eviction, manual-data contract propagation, frontend failure distinction, and absence of external baseball data code. Record any unverified runtime-only item as residual risk instead of claiming production readiness.
