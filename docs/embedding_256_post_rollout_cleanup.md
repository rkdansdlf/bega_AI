# 256-d Embedding Post-Rollout Cleanup

This runbook is for the final cleanup after `rag_chunks.embedding` has already
been migrated to `vector(256)` and retrieval is using the halfvec HNSW path.
It does not re-embed rows and does not fetch external baseball data.

## Preconditions

Runtime must resolve to the rollout target:

```sh
EMBED_PROVIDER=openrouter
OPENROUTER_EMBED_MODEL=openai/text-embedding-3-small
EMBED_DIM=256
RAG_EMBEDDING_VERSION=2
AI_VECTOR_INDEX=hnsw
AI_VECTOR_QUANTIZATION=halfvec
```

Database must already have:

```sql
rag_chunks.embedding type = vector(256)
pgvector >= 0.7.0
idx_rag_chunks_embedding_halfvec_hnsw present
```

## Safe Sequence

Set a run id and the target runtime contract before starting. Load the exact
target DB environment first.

```sh
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export BEGA_SKIP_APP_INIT=1
export EMBED_PROVIDER=openrouter
export OPENROUTER_EMBED_MODEL=openai/text-embedding-3-small
export EMBED_DIM=256
export RAG_EMBEDDING_VERSION=2
export AI_VECTOR_INDEX=hnsw
export AI_VECTOR_QUANTIZATION=halfvec
```

Pause all `rag_chunks` writers before the cleanup window: ingest, re-embed,
sync, and admin write jobs. Read-only service traffic can stay up.

1. Capture the read-only pre-audit.

```sh
./.venv/bin/python scripts/audit_embedding_256_migration.py \
  --summary-output "reports/${RUN_ID}_256_migration_audit_pre_summary.json" \
  --samples-output "reports/${RUN_ID}_256_migration_audit_pre_samples.json" \
  --sample-limit 50
```

Proceed only if there are no `fail` findings, `rag_chunks.embedding` is
`vector(256)`, retrieval SQL is
`embedding::halfvec(256) <=> %s::halfvec(256)`, and
`idx_rag_chunks_embedding_halfvec_hnsw` exists.

2. Dry-run the warning cleanup and inspect the report.

```sh
./.venv/bin/python scripts/cleanup_embedding_256_warnings.py \
  --dry-run \
  --report-output "reports/${RUN_ID}_cleanup_embedding_256_dryrun.json" \
  --sample-limit 50
```

Proceed only if `runtime_errors` and `db_errors` are empty and
`before_counts.metadata_conflicts` is `0`. If both
`before_counts.metadata_fixable` and `before_counts.active_null_embeddings` are
`0`, skip the apply step.

3. Apply metadata/NULL cleanup with the exact dry-run counts.

```sh
./.venv/bin/python scripts/cleanup_embedding_256_warnings.py \
  --apply \
  --report-output "reports/${RUN_ID}_cleanup_embedding_256_apply.json" \
  --sample-limit 0 \
  --expect-metadata-fixable "<dry-run metadata_fixable>" \
  --expect-active-null-embeddings "<dry-run active_null_embeddings>" \
  --expect-metadata-conflicts 0
```

If any expected count differs from the current DB count, the script rolls back,
writes `status = count_mismatch`, exits nonzero, and performs no updates. The
expected post-cleanup report has `after_counts.metadata_fixable = 0`,
`after_counts.metadata_conflicts = 0`, and
`after_counts.active_null_embeddings = 0`.

4. Re-run the read-only audit.

```sh
./.venv/bin/python scripts/audit_embedding_256_migration.py \
  --summary-output "reports/${RUN_ID}_256_migration_audit_post_summary.json" \
  --samples-output "reports/${RUN_ID}_256_migration_audit_post_samples.json" \
  --sample-limit 50
```

The clean state is `status = pass`, no active NULL embeddings, no metadata
fixable/conflict rows, and only `idx_rag_chunks_embedding_halfvec_hnsw` in the
embedding index list.

5. If the post-audit reports both halfvec and vector HNSW indexes, dry-run the legacy
vector HNSW drop.

```sh
./.venv/bin/python scripts/create_vector_index.py --dry-run --drop-vector-hnsw
```

Apply only after the dry-run confirms `AI_VECTOR_QUANTIZATION=halfvec`,
`EMBED_DIM=256`, the halfvec index exists, and retrieval SQL is
`halfvec(256)`.

```sh
./.venv/bin/python scripts/create_vector_index.py --drop-vector-hnsw
```

Restart AI services after index topology changes if any process uses cached
`AI_VECTOR_INDEX=auto` detection.

The saved rollout reports from 2026-05-31 are old-schema evidence only. They
showed `metadata_updated = 413`, `active_null_deactivated = 2`, and a later
audit pass on that target, but every real target must still run a fresh pre-audit
and cleanup dry-run before applying.

## Residual Risks

- The cleanup soft-deactivates active NULL embedding rows. It does not delete
  rows or synthesize embeddings. If those rows represent needed source content,
  regenerate them from trusted internal/operator-provided sources.
- The 256-d rollout used prefix slicing, not semantic re-embedding. Retrieval
  should be watched with smoke/canary reports after cleanup.
- Dropping the duplicate vector HNSW index is safe only after halfvec retrieval
  is active. Recreating that index later is possible but expensive on the full
  `rag_chunks` table.
- Running services using cached `AI_VECTOR_INDEX=auto` detection should be
  restarted after index topology changes.
