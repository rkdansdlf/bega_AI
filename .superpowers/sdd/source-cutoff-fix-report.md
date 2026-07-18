# Frozen Checkpoint Source Window Fix Report

Date: 2026-07-18

Status: COMPLETE

## Commits

- `1a062b6` — `fix: freeze checkpoint source update windows`
- Verification documentation and this report — `docs: verify frozen checkpoint source windows`

No push, merge, network call, live database connection, or external baseball-data source was used.

## Resolved Findings

- Added nullable `source_updated_before timestamptz` to the fresh migration 004 table and an idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` upgrade in the same migration. Managed and compatibility migration ordering remains unchanged.
- Added the cutoff to the managed schema contract, durable checkpoint model, explicit row mapper, repository inserts/updates/return validation, and session state.
- Update-filtered checkpoint sessions sample the trusted source database `clock_timestamp()` once before the data SELECT. The first progress advance or zero-row completion persists that exact cutoff. Recovery reuses a persisted cutoff without resampling.
- A progressed incomplete legacy checkpoint that requires a cutoff but has none fails closed as `INGEST_CHECKPOINT_INCOMPATIBLE`. A completed legacy checkpoint still skips both source-clock and source-data SELECTs.
- Custom and generic checkpoint queries apply the configured lower watermark when present, then the same configured update column's fixed `<= source_updated_before` upper bound, then the resume predicate, with canonical keyset order and tested parameter ordering.
- A post-cutoff source update behind the cursor and a post-cutoff update in the remaining suffix are deferred together. The suffix cannot advance the current maximum past the deferred update, while the next normal run can observe both updates from the prior success watermark.
- The success watermark remains the maximum committed update timestamp and remains owned by the full-run success transaction. The `game` profile now derives its watermark timestamp from `game_updated_at`, the same source expression used by its incremental filter.
- PostgreSQL catalog types `timestamp(p) with time zone` and `timestamp(p) without time zone` map to signature-distinct `datetime` and `datetime_naive` scalars for canonical precision 0 through 6. Malformed, out-of-range, and unrelated forms fail closed.

## TDD Evidence

Focused source-window/schema/query RED before production changes:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ai_schema_migrations.py tests/test_db_schema_contract.py tests/test_ingest_checkpoints.py tests/test_ingest_query.py -q -k 'ingest_checkpoint_migration_defines_durable_progress_table or complete_contract_includes_ingest_checkpoints or valid_timestamp_precision_forms or invalid_timestamp_precision_forms or repository_load_uses_explicit_columns or advance_locks_identity or start_reuses_persisted_source_cutoff or progressed_incomplete_checkpoint_missing_required_cutoff or completed_legacy_checkpoint_without_required_cutoff or complete_can_create_zero_row_checkpoint or custom_checkpoint_query_wraps_output_aliases or freezes_update_window or generic_checkpoint_query_has_exact or generic_checkpoint_query_requires_cutoff'
```

- Exit 1: 18 failed, 7 passed, 124 deselected. Failures were the missing migration/schema field, precision mapping, durable/session cutoff API, and query upper bound.

Focused fake recovery RED before production changes:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_checkpoint_integration.py -q -k 'recovery_excludes_row_updated_behind_cursor or suffix_update_after_cutoff or crash_restart_reuses_persisted_cutoff or next_normal_run_sees_updates_deferred or zero_row_completion_persists_sampled_source_cutoff or progressed_incomplete_checkpoint_without_cutoff_fails_closed'
```

- Exit 1: 6 failed, 29 deselected, 6 existing `datetime.utcnow()` warnings. Failures were missing cutoff sampling, filtering, persistence, reuse, and fail-closed recovery.

Same-filter watermark RED before the profile fix:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_query.py -q -k 'game_watermark_uses or game_profile_uses'
```

- Exit 1: 2 failed, 36 deselected.

Focused GREEN after the minimal implementation:

- Schema/repository/query/precision command: exit 0; 27 passed, 123 deselected.
- Fake recovery command: exit 0; 6 passed, 29 deselected, 7 existing `datetime.utcnow()` warnings.
- Documentation contract RED: 2 failed, 5 deselected before design/runbook edits.
- Documentation contract GREEN: 2 passed, 5 deselected after the edits.

Expanded checkpoint/schema/query/worker regression:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_checkpoints.py tests/test_ingest_checkpoint_integration.py tests/test_ingest_query.py tests/test_ingest_results.py tests/test_ingest_worker.py tests/test_ingest_run_store.py tests/test_observability_metrics.py tests/test_ai_schema_migrations.py tests/test_ai_schema_rehearsal_script.py tests/test_db_schema_contract.py tests/test_rag_storage_schema.py tests/test_schema_startup_mode.py tests/test_validate_ai_runtime_schema.py -q
```

- Exit 0: 272 passed, 0 failed, 25 existing `datetime.utcnow()` warnings in 21.12 seconds.

## Full Verification Gates

- `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m compileall -q app scripts tests`: exit 0, no output.
- `python3 /Users/mac/project/KBO_platform/scripts/validate_baseball_data_policy.py`: exit 0, `External baseball data policy OK`.
- `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python scripts/export_openapi_contract.py --check`: exit 0, `AI OpenAPI artifacts are current`; one existing `google.generativeai` FutureWarning.
- `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/ -q`: exit 0; 1921 passed, 5 skipped, 0 failed, 33 warnings in 358.47 seconds.
- `git diff --check HEAD^ HEAD`: exit 0 with no output before the verification documentation commit.

The five skips require local operator-data migration, validation, or handoff artifacts that are absent. Warnings are existing `google.generativeai`, checkpoint-path `datetime.utcnow()`, HTTP 422 alias, and Pydantic model-field deprecations.

## Files

Runtime and schema:

- `app/core/ingest_checkpoints.py`
- `app/db/migrations/004_ai_ingest_checkpoints.sql`
- `app/db/schema_contract.py`
- `scripts/ingest_from_kbo.py`

Tests:

- `tests/test_ai_schema_migrations.py`
- `tests/test_db_schema_contract.py`
- `tests/test_ingest_checkpoints.py`
- `tests/test_ingest_checkpoint_integration.py`
- `tests/test_ingest_query.py`

Behavior and operator documentation:

- `docs/data-sync-orchestration-runbook.md`
- `docs/superpowers/specs/2026-07-18-ingest-persistent-keyset-checkpoint-design.md`
- `docs/superpowers/plans/2026-07-18-ingest-persistent-keyset-checkpoints.md`
- `.superpowers/sdd/source-cutoff-fix-report.md`

## Self-Review

- Checkpoint identity: a non-null persisted cutoff cannot be rebound or changed by repository mutation. Returned rows must contain the expected cutoff before session state advances.
- Recovery: progressed incomplete legacy rows without a required cutoff fail before source-clock or data SELECT. Completed rows return durable results without either query.
- Query safety: both builders use the same configured update-filter column for lower and upper bounds, put the upper bound before the resume predicate, retain canonical ascending keyset order, and preserve the tested season/date/resume/limit parameter order.
- Watermark safety: only update-filtered profiles use the fixed cutoff. Their committed maximum cannot include a post-cutoff suffix update. Full-scan profiles remain unbounded by a cutoff, and success-only watermark ownership is unchanged.
- Timestamp safety: valid precision-qualified catalog names are full-matched; the aware and wall-clock subtypes retain distinct payload and signature semantics.
- Baseball-data policy: no crawler, scraper, browser automation, HTTP client, public API, web-search repair, generated baseball fact, external domain, dependency, credential, or endpoint was added. Tests use fake connections and non-production row values only.

## Residual Risks

- No live PostgreSQL test was run. Fake-connection coverage does not verify migration 004 upgrade execution or psycopg adaptation of the source-clock cutoff and precision-qualified timestamp cursors against a real PostgreSQL session.
- Text keyset ordering remains dependent on the source PostgreSQL collation. The resume predicate and ascending order use the same database expressions, but no live locale/collation matrix was executed.
- The fixed cutoff deliberately leaves a safe duplicate window from the inclusive prior success watermark through the cutoff. Concurrent inserts, deletes, or mutations that do not advance the configured update timestamp are outside this guarantee and retain documented next-run/full-scan semantics.
