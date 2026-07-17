# AI Ingestion Heartbeat and Lease Resilience Design

Date: 2026-07-18
Status: Approved by user
Scope: `bega_AI` durable ingestion coordination only

## Context

The durable ingestion worker claims an `ai_ingest_runs` row, executes the synchronous ingestion pipeline in a worker thread, and renews its lease from an asynchronous heartbeat task. The current worker treats the first heartbeat exception as immediate lease loss. All ingestion coordination also shares the application's general asynchronous PostgreSQL pool with chat, coach, retrieval, API status, and background work.

The observed large-table `game_summary` run progressed through committed batches, then lost its lease after a worker restart. Runtime evidence included PostgreSQL `OperationalError` and `PoolTimeout` failures, with no successful heartbeat after the recovered claim. Embedding work already runs through `asyncio.to_thread`, and the synchronous ingestion path checks a database lease fence before writes. The failure is therefore more consistent with coordination connection loss or pool pressure than with the event loop being directly blocked by the embedding call.

The current heartbeat and terminal-update SQL also checks status and owner but does not require `lease_expires_at > now()`. An expired owner may renew or finish a run before the recovery loop processes it. That lease resurrection weakens the intended fencing contract.

## Goals

- Isolate ingestion coordination from general application database pool pressure.
- Tolerate transient heartbeat connection failures without immediately abandoning a healthy run.
- Prevent an expired lease from being renewed or used for any terminal transition.
- Preserve the existing recovery handoff when an outage lasts beyond the lease budget.
- Keep the existing synchronous renderer, chunker, embedding, and `rag_chunks` write path unchanged.
- Add low-cardinality heartbeat metrics and sanitized logs that make lease behavior verifiable.
- Verify the behavior without external embedding calls or writes to a shared production database.

## Non-goals

- No persistent per-batch checkpoint or resume cursor in this change. That is a separate second-stage design.
- No database schema migration.
- No change to the trusted internal baseball source contract.
- No crawling, scraping, public baseball API, web search, or automatic baseball-data repair.
- No backend scheduler-monitor deduplication change. That remains a later task.
- No production canary or credential rotation in this change.

## Approaches Considered

### A. Add bounded heartbeat retry on the shared pool

Keep all coordination on the existing pool and stop treating the first exception as lease loss.

Advantages:

- Smallest implementation.
- Handles brief database network failures.

Trade-offs:

- Heartbeat retries still contend with chat, coach, retrieval, and monitoring traffic.
- A saturated shared pool can consume the complete lease retry budget.

### B. Use a dedicated coordination pool with bounded retry and strict expiry fences (selected)

Route all `IngestRunStore` operations through a small asynchronous PostgreSQL pool dedicated to ingestion coordination. Retry transient heartbeat exceptions only within the current lease's safety budget. Require an unexpired lease for renewal and terminal transitions.

Advantages:

- Protects heartbeats and recovery from general application pool contention.
- Preserves automatic recovery while preventing expired-owner resurrection.
- Keeps the data-processing path and database schema stable.

Trade-offs:

- Adds one or two PostgreSQL connections per AI service process.
- Adds another pool to application lifecycle management.

### C. Increase the lease duration only

Increase the default beyond 120 seconds without changing pool isolation or retry behavior.

Advantages:

- Configuration-only operational change.

Trade-offs:

- Delays rather than fixes failures caused by pool contention or connection loss.
- Extends the time before a genuinely dead worker is recovered.

Approach B is selected.

## Architecture

### Dedicated ingestion coordination pool

Add a lazily created asynchronous PostgreSQL pool in the AI dependency lifecycle using the same internal PostgreSQL connection string as the main pool. It has a minimum size of one and a maximum size of two. The pool uses the same connection health check, TCP keepalive, autocommit, and read-write target settings as the main pool.

The dedicated pool serves all `IngestRunStore` operations:

- submit or deduplicate a run
- get run status
- claim queued work
- heartbeat
- finish success, failure, or manual-data-required
- recover expired leases
- read and advance ingestion watermarks
- reconcile ingestion gauges

The existing general pool remains responsible for chat, coach, retrieval, and other application database work. The synchronous ingestion pipeline continues to manage its existing source and target connections and is not moved into the coordination pool.

The application opens the coordination pool during startup and fails startup if the pool cannot become ready. On shutdown, worker tasks are cancelled before the coordination pool is closed. The lifecycle is idempotent so partial startup failures do not leak a pool.

### Lease ownership invariant

Only a run that satisfies every condition below may renew or transition terminally:

- `status = 'RUNNING'`
- `lease_owner` equals the current worker owner
- `lease_expires_at > now()`

The existing per-write lease guard remains in place. The stricter store queries extend the same invariant to heartbeat and terminal state changes.

No expired owner is allowed to resurrect its lease, mark the run successful, mark it failed, or record a manual-data terminal result. Once expired, the persisted run remains for the recovery loop to requeue or fail according to the configured recovery-attempt limit.

## Heartbeat and Lease Flow

### Normal operation

1. `claim_next` returns a running record with a database lease expiry.
2. The worker records a local monotonic confirmation time when the claim is received.
3. The heartbeat loop waits the normal interval of `lease_seconds / 3`.
4. A successful heartbeat atomically extends the database expiry and returns the new expiry.
5. The worker resets its local confirmation time and returns to the normal interval.

The monotonic clock is used for local retry budgeting so wall-clock adjustments do not extend the retry window. PostgreSQL remains authoritative for the actual lease validity.

### Transient heartbeat failure

The first heartbeat exception does not set `lease_lost`. The worker logs only the run identifier and exception class, records a `retry` metric, and retries with bounded exponential backoff beginning at one second.

Retries stop before the last confirmed lease can expire. The local safety deadline is:

`last_confirmed_monotonic + lease_seconds - safety_margin`

The safety margin is at most five seconds and scales down for short test leases. Retry delay never exceeds the remaining safe budget. A successful retry records `success`, refreshes the confirmation time, and restores the normal heartbeat cadence.

No additional operator-configurable retry knobs are introduced. Retry timing is derived from the existing lease duration so invalid combinations cannot be configured.

### Confirmed lease loss

Lease loss is confirmed in either case:

- The heartbeat update returns no row because status, owner, or expiry no longer matches.
- Transient exceptions continue until the local safety deadline.

The first case records `rejected`; the second records `exhausted`. Both set the shared `lease_lost` event exactly once.

The worker does not attempt to force-cancel an in-flight synchronous embedding HTTP request. The event is checked between tables, and the existing database lease guard checks persisted ownership immediately before write batches. Therefore a late provider response cannot authorize a write after lease expiry.

### Recovery handoff

If the database outage clears before the safety deadline, the same worker renews and continues. If it does not, the previous worker stops producing durable writes and terminal state. After PostgreSQL is available, the existing recovery loop processes the expired run:

- requeue it when recovery attempts remain
- mark it `INGEST_LEASE_EXPIRED` when the recovery limit is exhausted

A newly claiming worker receives a new owner value. The expired worker cannot renew or finish because all ownership mutations include the strict expiry predicate.

## Error Handling

- Coordination pool startup failure: fail application startup with a sanitized error class and pool statistics that contain no connection string.
- Single or brief runtime connection failure: retry inside the current lease budget.
- Heartbeat rejection: stop the current worker's durable progress immediately and leave persisted state to the current owner or recovery loop.
- Retry-budget exhaustion: set lease loss, block later writes and terminal updates, and wait for recovery.
- Terminal-update race with recovery: translate a zero-row update into `IngestLeaseLostError`; do not replace the recovered run with a generic execution failure.
- Shutdown during retry: propagate cancellation immediately and do not convert it into lease exhaustion.
- Missing or inconsistent baseball data: retain the existing `MANUAL_BASEBALL_DATA_REQUIRED` contract. Do not synthesize or fetch replacement facts.

## Observability

Add `ai_ingest_heartbeats_total` with one bounded `result` label:

- `success`: a normal or retried renewal succeeds
- `retry`: a transient exception schedules another attempt
- `rejected`: PostgreSQL rejects renewal because the lease is no longer valid
- `exhausted`: the safe retry budget ends without renewal

Logs include only `run_id`, error class, attempt number, and remaining safe time. They exclude database URLs, credentials, request payloads, source rows, and embedding content.

The existing recovery, active-run, queued-run, completion, table, duration, and watermark metrics remain unchanged.

## Test Strategy

All implementation follows red-green-refactor cycles.

### Worker unit tests

- One heartbeat exception does not immediately set `lease_lost`.
- Multiple transient exceptions followed by success restore normal cadence.
- Exceptions lasting through the safe deadline set `lease_lost` and record `exhausted`.
- A heartbeat rejection sets `lease_lost` immediately and records `rejected`.
- Cancellation during a retry propagates without recording lease exhaustion.
- A synchronous fake ingest lasting longer than one lease interval continues to receive successful heartbeats.
- A worker whose lease is lost cannot record terminal success or process another table.
- Heartbeat metrics use only the four documented result labels.

### Store unit tests

- Heartbeat SQL requires status, owner, and `lease_expires_at > now()`.
- Success, failure, and manual-data terminal SQL require an unexpired owned lease.
- A zero-row terminal update raises `IngestLeaseLostError`.
- Recovery keeps the existing requeue and exhaustion behavior.

### Dependency lifecycle tests

- The ingestion store receives the dedicated pool, not the general pool.
- Startup opens both required pools and closes a partially opened coordination pool on failure.
- Shutdown cancels worker tasks before closing the coordination pool.
- General pool contention in a deterministic fake does not consume the coordination pool's heartbeat connection.

### Regression and policy verification

- Run the focused ingestion worker, store, API, lifecycle, observability, and schema-contract tests.
- Run the complete AI ingestion-related test selection.
- Run the repository baseball-data policy validator.
- Run static checks required by the AI service workflow.

The integration scenario uses a fake embedding implementation and controlled test database or fake pools. It performs no external embedding request and does not write to the shared production database.

## Completion Criteria

- A fake large ingestion that runs beyond one lease interval retains its lease through successful heartbeats.
- A transient database failure shorter than the retry budget does not cause recovery or duplicate ownership.
- A database failure lasting beyond the lease budget prevents the old worker from writing chunks or recording success.
- After expiry, recovery is the only path that can requeue or fail the run.
- Shared application pool contention does not prevent coordination heartbeat progress.
- New heartbeat metrics and sanitized logs are covered by tests.
- Existing ingestion tests and the baseball-data policy validator pass.
- No external baseball data collection, external embedding canary, production database write, or checkpoint schema change occurs.

## Follow-up

After this resilience change is implemented and verified, persistent per-batch checkpointing will be designed as a separate second stage. That design must define cursor identity, transaction boundaries, restart semantics, stale-row cleanup compatibility, and migration rollback before implementation.
