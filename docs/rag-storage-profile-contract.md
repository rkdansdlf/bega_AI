# RAG storage and profile contract

## Decision

Oracle is the canonical production RAG store. PostgreSQL with pgvector is a
local acceptance or staging store only; it is not a second production source
of truth. The service does not dual-write, automatically promote a PostgreSQL
corpus to Oracle, or fail over writes between backends.

`RAG_PROFILE` and `RAG_BACKEND` are both required at the RAG factory boundary.
The URL scheme is validated after selection and never chooses the adapter.

| Profile | Backend | Writes allowed | Purpose |
| --- | --- | --- | --- |
| `test` | `fake`, or an explicitly isolated local DB | fixtures only | unit/integration |
| `local-oracle` | `oracle` | local test data only | canonical-shape Oracle E2E |
| `local-postgres` | `postgres` | local test data only | acceptance/staging |
| `production` | `oracle` | prohibited by this implementation batch | canonical production |

Local and test profiles reject non-local hosts before a pool is created.
Production requires `APP_ENV=production`, `RAG_PROFILE=production`, and
`RAG_BACKEND=oracle`. A missing or unknown profile/backend fails selection; it
never selects a production backend implicitly. `fake` ignores all configured
database URLs and opens no network connection.

## Runtime inventory

| Concern | Oracle selection | PostgreSQL selection | Owner / code path |
| --- | --- | --- | --- |
| Runtime factory | `OracleRagPool` | tagged `PostgresRagPool` | `app/deps.py`, `app/core/rag_runtime.py` |
| Manual ingest writer | `MERGE` into `rag_chunks`, refresh `rag_chunk_terms` | `INSERT .. ON CONFLICT` into `rag_chunks` | `app/routers/ingest.py`, `app/core/oracle_rag.py`, `app/core/rag_storage.py` |
| Search reader | Oracle native vector + sparse term search | pgvector + PostgreSQL FTS | `app/core/rag.py`, `app/core/retrieval.py`, `app/tools/document_query.py` |
| Readiness | Oracle connection, table/vector coverage, HNSW index | PostgreSQL connection, table, vector extension/dimension, HNSW index | `app/core/rag_readiness.py` |
| API dependency | same `get_rag_connection_pool()` singleton | same singleton | ingest, search pipeline, document/regulation tools, readiness |
| Background batch writer | disabled | allowed only when unsplit `POSTGRES_DB_URL` is the selected destination | `app/config.py`, `app/core/ingest_worker.py`, `scripts/ingest_from_kbo.py` |

The PostgreSQL worker remains PostgreSQL-specific. It writes through
`settings.database_url`, so configuration rejects the worker when
`AI_RAG_DB_URL`, `OCI_DB_URL`, or the legacy URL could split the worker from
the selected PostgreSQL runtime. Oracle and fake profiles also require the
worker to be disabled. Oracle ingest uses the authenticated manual ingest API
until a canonical Oracle worker is designed separately.

## Schema and index ownership

PostgreSQL acceptance schema artifacts are owned in the AI service:

- `app/db/schema.sql` creates pgvector `rag_chunks`, support tables, and indexes.
- `app/db/migrations/005_rag_runtime_compatibility.sql` is additive
  compatibility for an existing PostgreSQL `rag_chunks` table.
- `app/db/rag_storage_indexes_concurrent.sql` and
  `scripts/create_vector_index.py` are operator-run PostgreSQL index paths.

The inspected repository has Oracle read/write code and an expected native
index name (`IDX_RAG_CHUNKS_EMBEDDING_HNSW`), but no tracked authoritative
Oracle `rag_chunks`/`rag_chunk_terms` provisioning migration was found. That is
an explicit schema-owner gap, not something runtime startup may repair. This
batch does not create or apply either Oracle or PostgreSQL vector schema.

`app/db/schema_contract.py` still validates PostgreSQL runtime tables and says
`rag_chunks` belongs to a backend/data migration path. No matching backend
Flyway RAG migration was found, so that comment is historical/ambiguous; the
artifacts above are the only concrete PostgreSQL owner found in this inventory.

## Identity, embedding and lifecycle

- The stable upsert identity is `(source_table, source_row_id)` on both
  backends. Multipart documents append `#partN`; missing old parts are soft
  deactivated rather than physically deleted.
- `content_hash` detects changed content. Embedding reuse additionally matches
  `embedding_model`, `embedding_dim`, `embedding_version`, and
  `chunking_version`; changed content does not retain a stale vector.
- The current dimension contract is `EMBED_DIM` (default 1536). Provider/model
  come from the explicit embedding settings. Readiness checks configuration
  and dimension compatibility without making an external model request.
- PostgreSQL retrieval requires active/valid rows (`is_active`, validity and
  expiry windows). Oracle retrieval accepts `ACTIVE` and `INDEXED`; replaced
  multipart rows become `DELETED`.
- PostgreSQL readiness expects the configured HNSW form
  (`idx_rag_chunks_embedding_halfvec_hnsw` or
  `idx_rag_chunks_embedding_hnsw`). Oracle expects
  `IDX_RAG_CHUNKS_EMBEDDING_HNSW` over native vectors.

## Cleanup and promotion

The fake backend persists nothing. Local database tests must use a unique
fixture `source_table`/`source_row_id` namespace and remove only rows they
created; there is no authorized blanket cleanup of existing local data.

`scripts/sync_rag_chunks.py` is a manual PostgreSQL-to-PostgreSQL CLI and is
not an automatic production promotion path. It cannot write Oracle and must
not be pointed at production in local verification. No automatic
PostgreSQL-to-Oracle promotion or dual-write path was found.

## HTTP contract

- `/health`: process liveness only, HTTP 200 with `{"status":"UP"}`.
- `/ready`: only the selected RAG backend plus embedding configuration.
  Missing storage/schema/vector/index/model returns HTTP 503 with secret-safe
  component codes.
- `/ai/ingest/`: validation remains 422, missing or wrong internal token is
  401, expected RAG dependency failures are structured 503, and unexpected
  implementation defects remain 500.
- `/ai/search/`: selected dependencies are checked before retrieval. A zero-hit
  search remains a normal empty result, but a DB or embedding failure with no
  results is a structured 503 rather than an empty 200 response.

This contract does not claim a provisioned store, `/ready` 200 against a real
database, successful corpus ingest/search, or production readiness. Those
require an isolated provisioning and E2E batch.
