# Final Review Fix Report

## Status

- Branch: `codex/data-sync-orchestration`
- Review base: `4a7ac0d` (`docs: explain ingest lease resilience`)
- Result: all critical, important, and minor findings in the final-review brief are addressed.
- Scope constraints: no network calls, live/shared database access, schema mutation, environment-file edits, external embeddings, or external baseball-data collection were used.

## Finding Resolution

### PostgreSQL lease safety

- `scripts/ingest_from_kbo.py` now locks the target run row before validating ownership and lease state, and evaluates validity with PostgreSQL `clock_timestamp()`.
- `app/core/ingest_run_store.py` now uses actual PostgreSQL time for claims, heartbeats, terminal transitions, and stale-run recovery.
- Heartbeat and terminal mutations lock the run first with `FOR UPDATE`, then perform the state mutation. Recovery selects stale candidates with `FOR UPDATE SKIP LOCKED` and actual-time predicates.
- Deterministic SQL-contract tests cover guard query ordering, row locks, actual-time comparisons, and stale-run recovery.

### Heartbeat safety budget and retry classification

- `app/core/ingest_worker.py` captures a conservative monotonic confirmation immediately before claim and immediately before each heartbeat request.
- Every heartbeat call is bounded by the remaining lease-safe budget with `asyncio.timeout`; a hung call marks the lease lost and increments the exhausted counter exactly once.
- A delayed successful response advances confirmation only to request start, never response completion.
- Retries are limited to `psycopg.OperationalError`, `psycopg.InterfaceError`, and `psycopg_pool.PoolTimeout`.
- Non-transient failures stop the heartbeat loop immediately without retry or exhausted-metric inflation. Logs contain only run identifiers and exception class names.
- Tests cover each transient class, hung calls, delayed success, non-transient failure, metric cardinality, and initial claim confirmation.

### Exception-safe FastAPI lifespan

- `app/deps.py` now wraps the complete lifespan in exception-safe acquisition and cleanup logic.
- All background tasks are canceled and awaited independently. HTTP, cache, runtime, and both database-pool closers are attempted independently.
- The original startup or request exception is preserved when cleanup also fails. During normal shutdown, cleanup completes fully before the first cleanup error is raised.
- Cleanup uses only a backend captured during startup; it never creates an embedding backend solely to close it.
- Pool close helpers clear their global references before awaiting close, making failed cleanup idempotent.
- Tests cover post-open startup failure, partial task cleanup, independent pool closing, request-exception preservation, and startup-error preservation.

### Metrics and documentation

- `AI_INGEST_HEARTBEATS_TOTAL` is explicitly exported with bounded result labels.
- The runbook describes the safety margin as up to five seconds, actual database time, bounded heartbeat requests, transient classification, and cleanup semantics.
- The approved design and implementation plan contain authoritative amendments for the final-review behavior.

## TDD Evidence

### RED

Command:

```text
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_results.py tests/test_ingest_run_store.py tests/test_ingest_worker.py tests/test_schema_startup_mode.py tests/test_deps_db_pool_observability.py tests/test_observability_metrics.py -q
```

Result: exit 1 — `19 failed, 48 passed in 5.13s`.

The failures demonstrated the missing lock-first/actual-time SQL, heartbeat timeout and confirmation rules, transient-only retry policy, exception-safe cleanup behavior, and metric export.

### GREEN

The same focused command after implementation: exit 0 — `67 passed in 1.91s`.

Prescribed broader focused command:

```text
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_runs.py tests/test_ingest_results.py tests/test_ingest_run_store.py tests/test_ingest_worker.py tests/test_schema_startup_mode.py tests/test_deps_db_pool_observability.py tests/test_observability_metrics.py tests/test_db_schema_contract.py -q
```

Result: exit 0 — `82 passed in 1.11s`.

A timing-sensitive retry-count assertion failed once under full-suite load. Its test budget was widened without changing production behavior; the targeted regression then passed: `1 passed in 0.49s`.

## Full Verification

- Full AI suite: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/ -q`
  - Final result: exit 0 — `1700 passed, 5 skipped, 8 warnings in 60.86s`.
  - The five skips require existing operator-provided local reports or migration fixtures. The eight warnings are existing dependency/deprecation warnings.
- Bytecode compilation: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m compileall -q app scripts` — exit 0.
- Baseball-data policy: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python ../../../scripts/validate_baseball_data_policy.py` — exit 0, `External baseball data policy OK`.
- Patch whitespace: `git diff --check` and `git diff --check 4a7ac0d` — exit 0.

## Files Changed

- Runtime: `app/core/ingest_run_store.py`, `app/core/ingest_worker.py`, `app/deps.py`, `app/observability/metrics.py`, `scripts/ingest_from_kbo.py`
- Tests: `tests/test_ingest_results.py`, `tests/test_ingest_run_store.py`, `tests/test_ingest_worker.py`, `tests/test_schema_startup_mode.py`, `tests/test_deps_db_pool_observability.py`, `tests/test_observability_metrics.py`
- Documentation: `docs/data-sync-orchestration-runbook.md`, the approved design, the implementation plan, and this report

## Remaining Concerns

- No live PostgreSQL integration test was run by design; SQL ordering and predicates are covered by deterministic unit/contract tests.
- The full suite retains five environment-dependent skips and eight pre-existing warnings; neither is introduced by this patch.
- No external baseball-data fallback or repair path was added. Missing baseball data continues to require the established `MANUAL_BASEBALL_DATA_REQUIRED` operator flow.

## Important Re-review Follow-up: Terminal Lease-loss Exception

### Resolution

- Commit: `38b8533 fix: surface ingest lease loss races`
- Files: `app/core/ingest_run_store.py`, `tests/test_ingest_run_store.py`
- `IngestRunStore._require_owned_run` now raises the existing domain-level `IngestLeaseLostError` with the constant sanitized message `ingest run lease is not owned`.
- Zero-row success, failure, and `MANUAL_BASEBALL_DATA_REQUIRED` terminal updates now enter the recovery-handoff branches already present in `IngestWorker.run_once` instead of escaping as plain `RuntimeError`.

### Strict TDD Evidence

RED command:

```text
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_run_store.py -q
```

RED result: exit 1 — `3 failed, 11 passed in 2.79s`. The three new zero-row terminal tests each failed because `_require_owned_run` raised plain `RuntimeError` rather than `IngestLeaseLostError`.

GREEN command (same command after the minimal implementation): exit 0 — `14 passed in 0.35s`.

Covering store/worker command:

```text
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_run_store.py tests/test_ingest_worker.py -q
```

The first covering run had one unrelated scheduler-sensitive failure in `test_long_sync_ingest_keeps_heartbeating_past_lease_interval` (`1 failed, 38 passed in 3.86s`). Its isolated rerun passed (`1 passed in 3.38s`), and the fresh complete covering rerun exited 0 with `39 passed in 1.93s`.

Broader resilience command:

```text
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_runs.py tests/test_ingest_results.py tests/test_ingest_run_store.py tests/test_ingest_worker.py tests/test_schema_startup_mode.py tests/test_deps_db_pool_observability.py tests/test_observability_metrics.py tests/test_db_schema_contract.py -q
```

Result: exit 0 — `85 passed in 1.11s`. `git diff --check` and the staged diff check both exited 0.

### Self-review

- Confirmed all three terminal mutations exercise a real zero-row store response and assert the domain exception plus sanitized message.
- Confirmed `run_once` catches `IngestLeaseLostError` independently for success, failure, and manual-data terminal races; existing worker terminal-race coverage passes.
- Confirmed the production change is limited to one import and one exception substitution, with no data, schema, environment, network, or external-service changes.

## Final Test Re-review Follow-up

### Resolution

- Commit: `c1fd354 test: cover ingest terminal recovery handoff`
- File: `tests/test_ingest_worker.py`
- Added independent worker tests proving `finish_failed` and `finish_manual_data_required` lease-loss races return from `run_once` without a terminal record or escaped exception.
- Replaced the fixed 80 ms sleep in the long synchronous-ingest heartbeat test with `threading.Event` coordination. The fake ingest remains blocked until 32 heartbeat calls complete, which exceeds the configured 0.3-second lease at a 0.01-second heartbeat interval, and is then explicitly released.

### Mutation RED / GREEN Evidence

After adding the recovery-handoff tests, the two corresponding production catches were temporarily changed to re-raise solely to prove the tests detect the regression. This mutation was restored before GREEN and is absent from the committed diff.

RED command:

```text
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_worker.py::test_expired_lease_during_failure_finish_is_left_for_recovery tests/test_ingest_worker.py::test_expired_lease_during_manual_finish_is_left_for_recovery -q
```

RED result: exit 1 — `2 failed in 0.62s`; both failures were escaped `IngestLeaseLostError` instances from the deliberately disabled handoff catches.

GREEN targeted command:

```text
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_worker.py::test_expired_lease_during_failure_finish_is_left_for_recovery tests/test_ingest_worker.py::test_expired_lease_during_manual_finish_is_left_for_recovery tests/test_ingest_worker.py::test_long_sync_ingest_keeps_heartbeating_past_lease_interval -q
```

Result: exit 0 — `3 passed in 0.53s`.

Worker-file command:

```text
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_worker.py -q
```

Result: exit 0 — `27 passed in 1.02s`.

Broader resilience command:

```text
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_runs.py tests/test_ingest_results.py tests/test_ingest_run_store.py tests/test_ingest_worker.py tests/test_schema_startup_mode.py tests/test_deps_db_pool_observability.py tests/test_observability_metrics.py tests/test_db_schema_contract.py -q
```

Result: exit 0 — `87 passed in 1.63s`. Working-tree and staged `git diff --check` both exited 0.

### Self-review

- Confirmed the final implementation commit changes tests only; `app/core/ingest_worker.py` has no diff.
- Confirmed the failure and manual branches each assert no failed, success, or manual terminal record remains after handoff.
- Confirmed heartbeat stabilization waits on observable call progress and uses five-second timeouts only as deadlock guards, not as pass criteria.
- Confirmed no unrelated file, external service, shared database, schema, or environment was touched.
