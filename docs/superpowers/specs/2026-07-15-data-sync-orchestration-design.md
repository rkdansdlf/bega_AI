# KBO Internal Data Synchronization Orchestration Design

Date: 2026-07-15
Status: Approved by user
Scope: `bega_AI`, `bega_backend/BEGA_PROJECT`, `bega_frontend`, and the trusted root sync handoff

## Context

The current system has three distinct data paths:

1. A trusted internal sync path populates PostgreSQL baseball tables. The existing root `scripts/sync_kbo_data.py` path must be preserved. The linked service repositories do not own external baseball collection.
2. The backend reads the PostgreSQL baseball tables for schedules, rankings, and home data.
3. The AI service transforms internal database rows into `rag_chunks` for RAG retrieval.

AI ingestion currently has two independently schedulable triggers:

- Backend JobRunr submits a full ingest every day at 04:30.
- `bega_AI/scripts/daily_ingest_kbo.sh` documents a daily incremental cron at 09:00 KST.

The backend considers the HTTP submission successful when FastAPI accepts an in-process background task. It cannot observe whether ingestion later succeeds. The AI service does not persist an ingestion run or an incremental success watermark. The frontend also converts some schedule failures into an empty list, which makes an unavailable or incomplete data source look like a legitimate no-games result.

## Goals

- Keep all baseball facts limited to the internal database, trusted internal sync, static project documents, or operator-provided manual data.
- Establish exactly one scheduled ingestion owner.
- Persist each AI ingestion request and its terminal result.
- Prevent overlapping equivalent runs and safely recover queued work after AI process restarts.
- Advance table watermarks only after successful ingestion.
- Surface `MANUAL_BASEBALL_DATA_REQUIRED` instead of synthesizing or silently hiding missing baseball data.
- Invalidate backend read caches after a successful scheduled ingestion.
- Distinguish empty data, unavailable data, manual-data-required, and synchronization-in-progress states in the frontend.
- Expose low-cardinality operational metrics without logging credentials or row payloads.

## Non-goals

- No crawling, scraping, public baseball API, browser collection, or web-search repair.
- No automatic synthesis of schedules, rankings, teams, players, or league dates.
- No replacement of the trusted root `scripts/sync_kbo_data.py` path.
- No new external task-queue dependency.
- No public ingestion, run-status, or cache-administration endpoint.
- No production deployment or environment-secret mutation as part of implementation.

## Approaches Considered

### A. Backend JobRunr orchestration with an AI database-backed run queue (recommended)

JobRunr remains the single daily scheduler. It submits a durable run to the AI service and schedules short, non-blocking follow-up status checks. The AI service persists runs and uses a single lease-based worker to claim queued work.

Advantages:

- Reuses the existing JobRunr dashboard and retry infrastructure.
- Avoids a long-running HTTP request and avoids occupying a JobRunr worker while ingestion runs.
- Survives an AI process restart because queued state is stored in PostgreSQL.
- Requires no new broker or worker framework.

Trade-offs:

- Adds a small database-backed queue and a worker lifecycle to the AI service.
- Requires an explicit cross-service status contract.

### B. AI-owned cron only

Disable the backend scheduler and keep the shell cron as the production scheduler.

Advantages:

- The AI service fully owns ingestion timing.
- Fewer cross-service API calls.

Trade-offs:

- Loses JobRunr retry and dashboard visibility.
- Shell cron completion is not visible to the backend cache layer.
- Process ownership and deployment-specific crontab setup remain implicit.

### C. External workflow orchestrator

Move trusted sync, AI ingestion, cache refresh, and monitoring into an external workflow system.

Advantages:

- Cleanest long-term separation and strongest workflow observability.

Trade-offs:

- Introduces new infrastructure and operational ownership.
- Exceeds the current requested scope.

Approach A is selected. The documented AI cron becomes a manual recovery command and must not be installed as a second recurring production schedule.

## Architecture

### 1. Trusted source handoff

The trusted root sync remains responsible for populating internal PostgreSQL baseball tables. It must not fetch or repair baseball facts from the public web.

The scheduled AI run treats the source database as authoritative. Before processing a table, it verifies that the table and configured required columns exist. A structurally missing table or required field produces a terminal manual-data result containing:

- contract code: `MANUAL_BASEBALL_DATA_REQUIRED`
- table or entity
- requested season or date range
- missing columns or required values
- expected source: trusted internal sync or operator-provided import

An empty result is not automatically an error because off-season and no-game dates are valid. Row-presence rules are therefore domain-specific and must not be inferred from a generic row-count check.

### 2. AI ingestion persistence

Add AI-owned migrations for two PostgreSQL tables.

`ai_ingest_runs`:

- `run_id` UUID primary key
- `request_key` deterministic hash of normalized tables, season, range, and mode
- `trigger_source` (`BACKEND_SCHEDULED`, `MANUAL_API`, `CLI_RECOVERY`)
- `status` (`QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `MANUAL_BASEBALL_DATA_REQUIRED`)
- normalized request payload as JSONB
- requested, started, heartbeat, and finished timestamps
- lease owner and lease expiration
- error code and sanitized error message
- per-table summary as JSONB
- created and updated timestamps

A partial unique index on `request_key` for `QUEUED` and `RUNNING` runs prevents equivalent overlap.

`ai_ingest_watermarks`:

- composite primary key: `source_table` and a deterministic season/explicit-range scope key
- `last_successful_updated_at`
- `last_run_id`
- updated timestamp

Watermarks advance only after the corresponding table transaction commits successfully. Failed and manual-data runs leave the previous watermark unchanged.

### 3. AI run API

Both endpoints remain protected by the existing internal-token dependency.

`POST /ai/ingest/run`

Request:

```json
{
  "tables": ["game", "game_metadata", "game_summary"],
  "season_year": 2026,
  "mode": "INCREMENTAL",
  "trigger_source": "BACKEND_SCHEDULED"
}
```

Response: HTTP 202

```json
{
  "run_id": "uuid",
  "status": "QUEUED",
  "deduplicated": false
}
```

When an equivalent active run exists, the endpoint returns the existing `run_id` with `deduplicated=true`.

`GET /ai/ingest/runs/{run_id}`

Returns the current status, timestamps, sanitized per-table counts, and error contract. It never returns database URLs, tokens, raw rows, or embedded content.

### 4. AI worker and recovery

One worker task starts with the FastAPI application lifecycle. It:

1. Claims one `QUEUED` run using a transaction and `FOR UPDATE SKIP LOCKED`.
2. Sets `RUNNING`, owner, heartbeat, and lease expiry.
3. Resolves each table's successful watermark for incremental mode.
4. Runs the existing table renderer, chunker, sensitive-content scan, embedding reuse, and UPSERT path.
5. Stores per-table counts and advances successful table watermarks.
6. Marks the run terminal.

On startup and during a periodic recovery loop, an expired `RUNNING` lease is returned to `QUEUED` once. A run that exceeds the configured recovery-attempt limit becomes `FAILED`. Only a lease owner may heartbeat or finish a run. The synchronous ingest path rechecks ownership and holds a row lock across each write batch so a recovered run cannot overlap writes with its prior owner.

The existing ingestion functions are refactored to return structured results instead of relying only on printed output. Their current CLI remains supported.

### 5. Single backend scheduler

Move scheduling responsibility from the cheerboard package to the backend AI package. Add configuration for:

- scheduler enabled flag
- daily cron
- explicit scheduled table list
- status-check interval
- maximum monitoring duration

The recurring JobRunr job submits one incremental run. It then schedules a one-off monitor job. A monitor invocation performs one status request:

- `QUEUED` or `RUNNING`: schedule the next monitor invocation and return.
- `SUCCEEDED`: invalidate affected backend caches and finish.
- `FAILED`: throw a domain-specific exception so JobRunr records failure.
- `MANUAL_BASEBALL_DATA_REQUIRED`: record the terminal contract without automatic repair and raise an operator-visible domain exception.
- monitoring deadline exceeded: fail without submitting a duplicate run.

The monitor is non-blocking between checks. It does not poll in a sleep loop.

`daily_ingest_kbo.sh` remains available for explicit operator recovery but its documentation states that it must not be installed alongside the JobRunr recurring schedule.

### 6. Cache behavior

After `SUCCEEDED`, evict these backend caches through an internal service call:

- `gameSchedule`
- `teamRankings`
- `homeBootstrap`
- `homeWidgets`
- `homeRankingSnapshot`

No new cache administration endpoint is exposed. Existing TTLs remain a fallback if orchestration is unavailable.

### 7. Manual-data error contract

Backend controller behavior becomes consistent:

- `ManualBaseballDataRequiredException` is always rethrown for the global business-exception handler.
- Ranking snapshot, league-date, and schedule paths must not replace manual-data errors with empty or synthesized baseball data.
- Generic infrastructure failures may use an explicitly non-baseball fallback only when it cannot be mistaken for a baseball fact.

The frontend preserves a typed manual-data response and presents five distinct states:

- loaded with rows
- loaded and legitimately empty
- synchronization pending
- `MANUAL_BASEBALL_DATA_REQUIRED`
- generic request failure

`fetchGamesData` must no longer convert every exception to `[]`. React Query owns request error state. Existing manual-data parsing utilities are reused.

### 8. Observability

AI metrics:

- run counts by terminal status and trigger source
- active and queued run gauges
- run duration histogram
- table duration and row/chunk counters using bounded table labels
- watermark lag by configured table
- lease recovery count

Backend metrics and logs:

- submission result and deduplication
- monitor terminal status
- elapsed orchestration duration
- cache invalidation result
- manual-data-required count

Metrics and logs exclude payload content, connection URLs, credentials, and operator-provided row values.

## Data Flow

1. Trusted internal sync updates PostgreSQL baseball tables.
2. JobRunr submits an explicit incremental AI ingestion request.
3. AI stores or deduplicates the queued request.
4. The AI worker claims and executes the run using per-table watermarks.
5. AI commits `rag_chunks`, watermarks, and the terminal run result.
6. JobRunr monitor observes the terminal result.
7. Backend evicts affected read caches after success.
8. Frontend refetches through its normal React Query lifecycle and distinguishes empty, pending, manual-data, and failure states.

## Failure Handling

- Source schema or required baseball field missing: terminal `MANUAL_BASEBALL_DATA_REQUIRED`; no generated substitute and no watermark advance.
- Embedding or database error: terminal `FAILED`; preserve previous watermarks; JobRunr records failure.
- Duplicate submission: return the existing active run.
- AI restart: reclaim expired lease within the recovery limit.
- Backend restart: recurring scheduling remains persistent in JobRunr; monitor jobs use the persisted AI `run_id`.
- Status endpoint unavailable: JobRunr monitor fails/retries without submitting a new ingestion run.
- Frontend request failure: render an error state, not an empty schedule.

## Compatibility and Rollout

1. Add AI migration, repository, state machine, tests, and status API while preserving the current CLI.
2. Add backend status-client and monitor behavior behind a disabled-by-default orchestration flag.
3. Update the manual-data controller behavior and frontend error states.
4. Enable the new backend schedule in one environment while the old AI cron remains disabled.
5. Observe at least one successful scheduled run and one intentional failure test before removing the old fire-and-forget path.
6. Document `daily_ingest_kbo.sh` as manual recovery only.

Rollback disables the new scheduler flag. It does not delete run history, watermarks, or `rag_chunks`.

## Test Strategy

All behavior changes use test-first red-green-refactor cycles.

AI focused tests:

- run creation and normalized request-key deduplication
- legal and illegal state transitions
- lease claim, heartbeat, expiry recovery, and recovery limit
- watermark advance on success only
- structured table result generation
- manual-data result for missing required structure
- POST and GET internal-token contracts
- CLI compatibility

Backend focused tests:

- one recurring scheduler registration with configuration
- explicit incremental payload submission
- one-shot monitor rescheduling for non-terminal status
- success cache eviction
- failed and manual-data terminal behavior
- ranking and league-date manual exception propagation

Frontend focused tests:

- schedule API errors remain errors instead of empty arrays
- legitimate empty arrays remain the empty state
- manual-data response parsing and rendering
- synchronization-pending rendering
- existing query keys and cache timings remain compatible unless explicitly changed

Verification gates:

- targeted AI pytest files, then the relevant AI test suite
- targeted backend Gradle test classes, migration safety check, then backend tests proportional to change
- frontend unit tests/build and targeted Cypress coverage
- `python3 scripts/validate_baseball_data_policy.py`
- cross-service code review and security review for the internal-token/status endpoint changes

## Acceptance Criteria

- Only one recurring production ingestion schedule is enabled.
- Every scheduled submission has a stable `run_id` and terminal persisted status.
- Equivalent queued/running requests cannot overlap.
- A process restart cannot silently lose a queued run.
- A failed table does not advance its watermark.
- Backend success is based on the AI terminal status, not HTTP acceptance.
- Successful runs evict the documented caches.
- Missing or inconsistent required baseball data surfaces `MANUAL_BASEBALL_DATA_REQUIRED` with exact requirements.
- Frontend request failures are not rendered as a legitimate empty schedule.
- No external baseball collection or automatic repair path is introduced.
- Required focused tests and cross-service verification gates pass.
