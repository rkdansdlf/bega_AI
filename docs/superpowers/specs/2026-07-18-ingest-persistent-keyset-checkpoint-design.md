# Persistent Keyset Checkpoints for AI Ingestion

## Summary

The AI ingestion worker currently commits `rag_chunks` in bounded batches, but it
advances the durable table watermark only when the entire run succeeds. If a
worker is restarted after one or more batch commits, recovery safely reuses
content hashes but still queries the trusted source table from the beginning.

This change adds a durable per-run, per-table keyset checkpoint. A recovered run
starts its trusted internal database query strictly after the last source row
whose destination writes and checkpoint were committed together. Static project
documents remain atomic and do not use keyset checkpoints.

The checkpoint is an execution cursor, not a successful synchronization
watermark. Existing watermarks still advance only when the full run reaches
`SUCCEEDED`.

## Goals

- Skip already committed source rows at the database query layer after lease
  recovery or process restart.
- Apply checkpointing to every internal database table that has a stable,
  non-null, unique ordering key, regardless of table size.
- Commit `rag_chunks` changes and the corresponding checkpoint in the same
  destination database transaction.
- Preserve cumulative table counts and maximum observed update time across
  recovery attempts.
- Skip a completed checkpointed table without querying its source rows again.
- Preserve completed and incomplete checkpoints for audit and incident analysis.
- Reject unsafe or incompatible resume attempts explicitly instead of silently
  restarting from the beginning.
- Preserve the existing internal-data-only baseball policy and
  `MANUAL_BASEBALL_DATA_REQUIRED` behavior.

## Non-goals

- Checkpointing static Markdown or other static project documents.
- Reusing a failed or completed run's checkpoint for a new run ID.
- Adding external baseball APIs, crawling, scraping, browser automation, or
  web-search repair.
- Automatically deleting old checkpoints. Retention automation is a later
  operational policy.
- Supporting checkpointed `row_stale_cleanup`. The orchestration worker already
  runs with cleanup disabled.
- Providing exactly-once source snapshots for inserts, deletes, or mutations that
  do not change a configured update timestamp.
- Changing the backend scheduler, frontend synchronization UI, or ingestion API
  request schema.

## Existing Behavior and Constraints

The worker invokes the synchronous ingestion function once per requested table.
The synchronous path reads a trusted internal PostgreSQL source and writes to the
AI PostgreSQL destination. The destination contains `ai_ingest_runs`,
`ai_ingest_watermarks`, and `rag_chunks`, so it can atomically fence the lease,
write chunks, and update a checkpoint.

Leased ingestion currently commits every flushed embedding buffer. A lease guard
checks the run row before each write and takes a shared lock before the write
transaction commits. This prevents lease recovery from changing ownership while
the batch is being committed.

Custom source queries have table-specific ordering, joins, and aliases. Generic
primary-key inference is not sufficient for those queries. Offset-based resume
is also unsafe when source rows are inserted or deleted. Therefore checkpointed
queries use an explicit typed keyset contract.

## Architecture

### Durable checkpoint table

Add `app/db/migrations/004_ai_ingest_checkpoints.sql` with an additive table:

```sql
CREATE TABLE IF NOT EXISTS ai_ingest_checkpoints (
    run_id uuid NOT NULL REFERENCES ai_ingest_runs(run_id),
    source_table varchar(128) NOT NULL,
    scope_key varchar(64) NOT NULL,
    cursor_version integer NOT NULL,
    cursor_signature varchar(64) NOT NULL,
    cursor_payload jsonb,
    committed_batches bigint NOT NULL DEFAULT 0,
    source_rows bigint NOT NULL DEFAULT 0,
    written_chunks bigint NOT NULL DEFAULT 0,
    reused_embeddings bigint NOT NULL DEFAULT 0,
    embedded_chunks bigint NOT NULL DEFAULT 0,
    max_updated_at timestamptz,
    source_updated_before timestamptz,
    completed boolean NOT NULL DEFAULT false,
    completed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, source_table),
    CONSTRAINT ck_ai_ingest_checkpoint_counts_nonnegative CHECK (
        committed_batches >= 0
        AND source_rows >= 0
        AND written_chunks >= 0
        AND reused_embeddings >= 0
        AND embedded_chunks >= 0
    ),
    CONSTRAINT ck_ai_ingest_checkpoint_cursor_present CHECK (
        source_rows = 0 OR cursor_payload IS NOT NULL
    ),
    CONSTRAINT ck_ai_ingest_checkpoint_completion_time CHECK (
        (completed = false AND completed_at IS NULL)
        OR (completed = true AND completed_at IS NOT NULL)
    )
);

ALTER TABLE ai_ingest_checkpoints
    ADD COLUMN IF NOT EXISTS source_updated_before timestamptz;
```

An index on `updated_at` supports later audit and retention queries. The primary
key deliberately includes `run_id`: a newly submitted run starts independently
from the last successful watermark and never resumes an earlier terminal run.

The AI startup migration sequence applies 003 before 004. The schema contract
requires all checkpoint columns. Rolling application code back does not drop the
table; older code safely ignores it.

### Checkpoint domain module

Add a focused checkpoint module under `app/core` containing:

- `CheckpointOrderField`: output field name and supported scalar type.
- `CheckpointOrder`: ordered, ascending key fields plus a deterministic
  signature.
- `CheckpointCursor`: typed values for one source row.
- `IngestCheckpoint`: durable progress, counters, maximum update time, and
  completion state.
- Cursor encode/decode, signature validation, row extraction, and keyset SQL
  helpers.
- Destination-connection helpers for loading and atomically upserting a
  checkpoint.

Checkpoint persistence uses the synchronous destination connection already owned
by `scripts/ingest_from_kbo.py`. This keeps the checkpoint update in the exact
transaction that writes `rag_chunks`; it is not delegated to the worker's async
coordination pool.

### Order resolution

All checkpoint key fields are normalized to ascending order. Existing display or
processing order in a custom query is not part of the checkpoint contract.

- Static profiles with `source_file` have no checkpoint order and retain their
  current atomic behavior.
- Generic table queries derive the order from the real PostgreSQL primary-key
  columns and their scalar types.
- Every custom SQL profile that participates in orchestration declares an
  explicit `checkpoint_order`. Its fields refer to output column aliases and
  collectively identify one output source row.
- A custom query's configured key must include enough underlying primary-key or
  sequence fields to be unique across joins.
- Cursor fields must be present and non-null in every returned row.

Supported cursor scalar types are integer, decimal, date, timezone-aware or
naive timestamp, UUID, text, and boolean. Values are encoded with explicit type
tags in JSON. Decimal, date, timestamp, and UUID values use canonical strings.
`timestamp without time zone` preserves its wall-clock fields with the
`datetime_naive` scalar. `timestamp with time zone` uses timezone-aware instants and interprets offset-free values as UTC with the `datetime` scalar. Catalog mapping accepts PostgreSQL
timestamp precision forms from `timestamp(0)` through `timestamp(6)` and rejects
malformed or out-of-range precision. The two timestamp subtypes remain signature-distinct.

The cursor signature is a SHA-256 digest of the cursor version, source table,
ordered field names, scalar types, direction, and a profile query-version value.
Changing the query shape or order contract changes the signature and prevents an
old in-flight run from resuming under different semantics.

Example payload:

```json
{
  "values": [
    {"field": "game_date", "type": "date", "value": "2026-07-18"},
    {"field": "game_id", "type": "text", "value": "20260718LGKT"}
  ]
}
```

## Query Construction

### Generic tables

The existing season, `since`, and date-bound filters remain unchanged. Resume
adds a row-value comparison and the canonical keyset order:

```sql
SELECT *
FROM source_table
WHERE <existing filters>
  AND updated_at <= source_updated_before
  AND ROW(pk_1, pk_2) > ROW(%s, %s)
ORDER BY pk_1 ASC, pk_2 ASC
```

### Custom SQL profiles

The builder removes the custom query's top-level `ORDER BY`, preserves its base
query and existing filters, and wraps it. Keyset fields refer only to wrapper
output aliases:

```sql
WITH checkpoint_source AS (
    <custom base query with season/since/date filters>
)
SELECT *
FROM checkpoint_source
WHERE updated_at <= source_updated_before
  AND ROW("cursor_field_1", "cursor_field_2") > ROW(%s, %s)
ORDER BY "cursor_field_1" ASC, "cursor_field_2" ASC
```

The same canonical `ORDER BY` is used on the first attempt when no cursor exists.
This guarantees that initial execution and recovery share one order contract.

For every checkpointed profile with `since_filter_column`, a new incomplete
table reads `clock_timestamp()` from the trusted source database before its data
SELECT. That exact value is the run-local `source_updated_before` cutoff. The
custom and generic builders apply the lower successful-watermark predicate when
present, then `updated_at <= source_updated_before`, then the resume predicate,
all against the configured update-filter column. The first progress commit or a
zero-row completion persists the cutoff. Recovery reuses the stored value and
does not sample the source clock again.

Checkpointed orchestration requires `limit=None`. A leased checkpointed call with
a limit is rejected because a per-attempt limit would not represent a stable
whole-run bound. Existing non-leased CLI behavior remains unchanged.

Before processing rows, the query result columns are checked for all configured
cursor fields. Each row's cursor must be strictly greater than the previously
observed cursor. This runtime check catches null keys, duplicate keys, query
ordering regressions, and profile configuration errors before advancing the next
checkpoint.

The source iterator keeps one row of lookahead at every prospective commit
boundary. Cursor `K` is eligible for a checkpoint commit only after the next row
is known to have a cursor strictly greater than `K`, or the source is known to be
exhausted. This prevents a duplicate key immediately after a commit boundary from
being silently skipped by `cursor > K` after a crash. Duplicate or decreasing
keys fail before either member of the unsafe boundary is checkpointed.

## Transaction and Batch Boundaries

A source row is the smallest checkpoint unit. All chunks derived from one source
row stay in the same in-memory buffer boundary; the cursor does not become
eligible until row rendering is complete and the one-row lookahead has proven a
strictly greater next cursor or end-of-stream.

Each destination commit performs these operations in order:

1. Check current lease ownership and expiry.
2. Take the existing shared lease fence for a write transaction.
3. Upsert and soft-deactivate chunk parts for every fully prepared source row in
   the buffer.
4. Upsert the checkpoint cursor, cumulative counters, and maximum observed update
   time.
5. Commit once.

The checkpoint update uses compare-and-set predicates for `scope_key`,
`cursor_version`, and `cursor_signature`. Existing progress can only advance: a
cursor or cumulative counter may not move backward.

Rows that produce no stored chunks because of quality, sensitivity, or empty
rendering still advance `source_rows` and the cursor. If a commit batch contains
only such rows, the transaction writes only the fenced checkpoint. This prevents
an endless retry of intentionally filtered rows.

When the source cursor is exhausted, a final fenced destination transaction marks
the checkpoint `completed=true` and sets `completed_at`. This marker may be a
checkpoint-only commit. A zero-row table therefore has a completed checkpoint
with a null cursor and zero counters.

The final `IngestTableResult` is built from durable cumulative checkpoint values,
not only values observed in the current process. Its `max_updated_at` is the
maximum committed value across all attempts and remains the input to the existing
success-only watermark update. The result also carries non-persisted
`attempt_source_rows` and `attempt_written_chunks` deltas for metrics. A completed
checkpoint skipped during recovery returns zero attempt deltas, so cumulative
run summaries remain accurate without double-counting the existing Prometheus
table totals.

The success watermark remains the maximum committed update timestamp. This is
safe for update-filtered profiles because every committed source row is bounded
by the fixed source cutoff. Watermark ownership remains success-only and does not
move to checkpoint advancement.

## Recovery Semantics

When a leased database table starts, ingestion loads `(run_id, source_table)`
from the destination connection.

- No checkpoint: run the canonical keyset query from the beginning.
- Compatible incomplete checkpoint: add the keyset predicate and start strictly
  after its cursor.
- Compatible completed checkpoint: return its durable result without executing
  the source timestamp or data SELECT.
- Incompatible checkpoint: fail explicitly; never reset or ignore it.

An incomplete legacy checkpoint with committed progress but no required
`source_updated_before` fails closed with `INGEST_CHECKPOINT_INCOMPATIBLE`. A
zero-progress incomplete row may initialize the cutoff on its first subsequent
progress or completion commit. A completed legacy row remains skippable because
it executes neither source query.

Crash behavior is deterministic:

- Before destination commit: neither chunks nor checkpoint survive, so recovery
  repeats the same source rows.
- After destination commit: chunks and checkpoint both survive, so recovery starts
  at the following key.
- After the last data commit but before completion marking: recovery queries after
  the last cursor, observes zero rows, and marks the table complete.
- After table completion but before run completion: recovery skips that table and
  uses its durable cumulative result.

The system provides atomic at-least-once processing at the batch boundary. It
does not hold a PostgreSQL snapshot across process restarts. A row updated after
the fixed cutoff is excluded even when it is ahead of the current cursor, so it
cannot advance the current run's maximum past an update deferred behind the
cursor. Both updates are eligible for the next normal run. The residual duplicate
window starts at the prior successful watermark and ends at the fixed cutoff
because the lower predicate remains inclusive. Inserts, deletes, and tables
without an update timestamp retain full-scan or next-run semantics. The
checkpoint never becomes the successful cross-run watermark.

## Error Contracts

The following errors terminate the current execution as `FAILED` unless noted:

- `INGEST_CHECKPOINT_CURSOR_UNAVAILABLE`: a database source has no configured or
  discoverable stable unique order.
- `INGEST_CHECKPOINT_INCOMPATIBLE`: stored scope, cursor version, or signature
  differs from the current contract.
- `INGEST_CHECKPOINT_CURSOR_TYPE_UNSUPPORTED`: a cursor value cannot be encoded
  or restored safely.
- `INGEST_CHECKPOINT_ORDER_VIOLATION`: returned rows are null, duplicate, or not
  strictly increasing under the configured order.
- `INGEST_CHECKPOINT_STALE_CLEANUP_UNSUPPORTED`: a checkpointed leased run asks
  for `row_stale_cleanup` other than `off`.

If a configured cursor field is absent from the trusted internal source result,
or a required cursor value is null because required source data is missing, the
ingestion path raises `ManualBaseballDataRequiredError` with:

```text
MANUAL_BASEBALL_DATA_REQUIRED
```

The contract names the source entity, missing cursor fields, affected range when
known, and `operator_manual_data` as the repair path. It never synthesizes a key
or seeks an external source.

A checkpoint database write failure rolls back the same transaction's chunk
changes. A lease loss prevents both chunk and checkpoint commit. Error messages
remain sanitized by the existing run-store terminal paths.

## Stale-row Cleanup Compatibility

Full stale-row cleanup requires the complete set of active source row IDs. A
resumed keyset scan intentionally omits rows committed by previous attempts, so
running cleanup from only the resumed suffix could deactivate valid earlier rows.

Therefore:

- Leased checkpointed database ingestion accepts only
  `row_stale_cleanup="off"`.
- `dry-run` and `apply` fail before the source query with
  `INGEST_CHECKPOINT_STALE_CLEANUP_UNSUPPORTED`.
- Non-leased CLI ingestion keeps its existing cleanup behavior and does not write
  checkpoints.

## Observability

Add a low-cardinality checkpoint event counter with controlled labels:

```text
ai_ingest_checkpoint_events_total{source_table,result}
```

Allowed results are `created`, `advanced`, `completed`, `resumed`,
`incompatible`, and `rejected`. Source-table labels use the existing normalized,
allowlisted table label function. Run IDs and cursor values are never metric
labels or log fields.

Logs include the run ID, normalized source table, event, committed batch count,
and cumulative source row count. They do not include raw cursor payloads, source
rows, connection representations, secrets, or baseball facts.

The final per-table run summary retains durable cumulative counts and adds
sanitized checkpoint metadata: whether the table resumed, committed batch count,
and completion state. Existing source-row and written-chunk counters increment by
the current attempt deltas rather than those cumulative values. API request and
response schemas do not change.

## Testing Strategy

Implementation follows red-green-refactor cycles. Tests cover:

1. Migration ordering, additive schema, constraints, index, and schema contract.
2. Cursor encoding and decoding for every supported type.
3. Signature determinism and incompatibility detection.
4. Generic single and composite primary-key queries.
5. Custom-query wrapping, existing filters, canonical order, and resume params.
6. Missing, null, duplicate, decreasing, and unsupported cursor values,
   including a duplicate split across a prospective commit boundary.
7. Atomic call order for lease fence, chunk writes, checkpoint upsert, and commit.
8. Rollback before commit and resume after commit.
9. Checkpoint advancement for a zero-output batch.
10. Completion after an empty suffix and source-query skipping for an already
    completed table.
11. Cumulative counts and maximum update time across recovery attempts.
12. Zero attempt deltas when a completed checkpoint is skipped, preventing metric
    double counting.
13. Limit and stale-cleanup rejection in checkpointed mode.
14. Lease loss preventing checkpoint advancement.
15. Metrics, sanitized summaries, and absence of high-cardinality labels.
16. `MANUAL_BASEBALL_DATA_REQUIRED` for missing trusted-source cursor fields.
17. Existing ingestion worker, run-store, migration, and OpenAPI regression tests.
18. Full AI test suite and `scripts/validate_baseball_data_policy.py`.
19. Fixed source cutoff sampling, persistence on first advance and zero-row
    completion, immutable recovery reuse, post-cutoff exclusion, deferred next-run
    visibility, and progressed legacy incompatibility.
20. Precision-qualified PostgreSQL timestamp subtype mapping for valid 0..6 and
    rejection of malformed or out-of-range precision.

No test uses an external baseball source, external embedding call, shared
database write, or production data. PostgreSQL transaction and keyset syntax use
deterministic test doubles by default. If a disposable local PostgreSQL runtime
is already available and separately approved, a local-only smoke test may verify
the migration, tuple comparison, crash boundary, and restart query. Otherwise the
lack of a live SQL smoke test is reported as residual risk.

## Rollout and Rollback

1. Deploy the additive migration and checkpoint-capable application together.
2. Observe checkpoint event counts, lease recovery counts, run duration, and
   watermark lag.
3. Confirm recovered runs emit `resumed` and do not repeat source rows before the
   stored cursor.
4. Preserve checkpoint rows for audit; do not add an automatic retention job in
   this change.

Application rollback restores the previous from-start recovery behavior. The 004
table and its rows remain in place for audit and for a future redeploy. Rollback
does not delete `rag_chunks`, checkpoints, run history, or watermarks. Because the
checkpoint is scoped to a run ID and older code does not read it, preserving the
table does not alter older execution behavior.

## Acceptance Criteria

- A recovered leased database ingestion query starts strictly after the last
  atomically committed cursor.
- The same transaction commits destination chunk mutations and checkpoint
  progress.
- A completed checkpointed table performs no source SELECT on recovery.
- Every checkpointed custom query has an explicit stable unique order; every
  generic query uses a real primary key.
- Unsafe cursor, cleanup, limit, or compatibility states fail explicitly.
- Static documents remain atomic and uncheckpointed.
- Watermarks advance only after full run success.
- Update-filtered checkpoint queries reuse one persisted source-clock cutoff and
  cannot let a post-cutoff suffix update outrun a deferred behind-cursor update.
- Checkpoints remain available after terminal run completion.
- Missing trusted baseball fields surface `MANUAL_BASEBALL_DATA_REQUIRED` without
  external repair.
- Focused and full AI tests pass, and the baseball-data policy validator passes.
