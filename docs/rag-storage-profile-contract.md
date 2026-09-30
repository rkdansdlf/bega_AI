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

### Oracle authority

The authoritative Oracle RAG schema is owned by the Backend repository at
`bega_backend/BEGA_PROJECT/src/main/resources/db/migration_rag_oracle/`.
It is intentionally separate from all three existing Flyway sets:

- primary application migrations do not necessarily target `AI_RAG_DB_URL`;
- baseball migrations target `BASEBALL_DB_URL`, which may be a different user,
  schema, or database;
- PostgreSQL acceptance migrations are not Oracle production authority.

The migration executor is Backend `RagFlywayMigrationConfig`. It uses the
dedicated `flyway_rag_schema_history` table and requires an explicit Oracle
JDBC migration URL. `RAG_FLYWAY_ENABLED` defaults to `false`; when it is false,
the executor does not construct a RAG datasource or call Flyway. The AI
service is the runtime validator and DML client only. It must never create,
alter, reconcile, or grant Oracle objects during startup, readiness, ingest,
or search.

| Responsibility | Owner |
| --- | --- |
| Schema source of truth | Backend `db/migration_rag_oracle/` |
| Migration executor | Backend `RagFlywayMigrationConfig`, explicitly enabled |
| Runtime schema/index validator | AI readiness code, read-only catalog queries |
| Isolated local provisioning | Test/release operator using the dedicated executor |
| Production promotion | Authorized DB/release operator in a separate reviewed run |
| Prohibited DDL actors | AI runtime, normal Backend startup, baseball crawler/sync paths |

The first migration is a fresh-schema contract. Existing objects are not
silently baselined or reconciled. Environment-specific grants are not in the
migration. The current runtime uses unqualified table names and validates the
vector index through `USER_INDEXES`, so `RAG_FLYWAY_USERNAME` must be the same
Oracle schema owner used by `AI_RAG_DB_URL`. A separate read/write user with
object grants would not see the owner index in `USER_INDEXES` and is therefore
not supported by the present runtime. Strict cross-user least privilege would
require a separate change to schema-qualify SQL and validate `ALL_INDEXES`;
until then, no cross-schema grants should be advertised or applied.

Checkpoint 5B local acceptance therefore formalizes a same-schema principal:
the migration owner, `RAG_ORACLE_SCHEMA`, `RAG_FLYWAY_USERNAME`, and the user
inside `AI_RAG_DB_URL` must be identical and must use a `LOCAL*` or `TEST*`
name. This is a functional local policy, not production authorization. The
principal holds DDL privileges while Flyway runs and runtime DML privileges
while the AI service runs. A production operator may be able to revoke DDL
after migration, but that control has not been validated here. Production readiness remains blocked until security owners approve same-principal use or
a separate-owner implementation is completed and tested.

### Oracle runtime/schema matrix

| Concern | Canonical migration contract | Runtime evidence |
| --- | --- | --- |
| `rag_chunks` identity | identity `id`; unique `(source_table, source_row_id)` | `MERGE` match and term identity lookup |
| Text and metadata | `title`/`content` CLOB; JSON-checked `meta`; `content_hash` | dense/sparse hydration, `JSON_VALUE`, changed-content embedding lookup |
| Vector | nullable `VECTOR(1536, FLOAT32)` | `EMBED_DIM=1536`, float-array binds, dimension readiness check |
| Lifecycle | `ACTIVE`, `INDEXED`, `DELETED` only | retrieval accepts first two; multipart replacement writes `DELETED` |
| Version/audit | `index_version`, `indexed_at`, `created_at`, `updated_at` | Oracle upsert and readiness result hydration |
| Search filters | `source_table`, `season_year`, `team_id`, `league_type_code`, `player_id` | `_bind_filter_clauses()` |
| Sparse terms | `(rag_chunk_id, token)` key, cascade FK, counts, source and game date | replace-on-upsert and per-token bounded search |
| Scalar indexes | chunk filter/lifecycle and term token/source-date indexes | dense filtering, lifecycle update, sparse candidate lookup |
| Vector index | `IDX_RAG_CHUNKS_EMBEDDING_HNSW`, cosine HNSW | readiness exact-name check and cosine distance query |

`league_type_code` is an optional manual-ingest field and a direct search
predicate. The payload trims surrounding whitespace; empty or whitespace-only
values normalize to `NULL`. A non-empty code is bound to both Oracle `MERGE`
branches. Re-ingesting the same `(source_table, source_row_id)` updates the
existing filter value, including an explicit transition to `NULL`, rather than
creating a second canonical row. `/ai/search/` accepts the internal `league`
query parameter and maps it to this same storage column.

The HNSW index and manual `MERGE` writer require an Oracle release with HNSW
DML support (23.6 or newer). Version capability, vector memory, syntax, and
actual index validity remain 5B live-provisioning checks; static 5A tests do
not claim them.

PostgreSQL acceptance schema artifacts are owned in the AI service:

- `app/db/schema.sql` creates pgvector `rag_chunks`, support tables, and indexes.
- `app/db/migrations/005_rag_runtime_compatibility.sql` is additive
  compatibility for an existing PostgreSQL `rag_chunks` table.
- `app/db/rag_storage_indexes_concurrent.sql` and
  `scripts/create_vector_index.py` are operator-run PostgreSQL index paths.

`app/db/schema_contract.py` remains a PostgreSQL-only validator. Its note that
`rag_chunks` belongs to a backend/data migration path now resolves to the
dedicated Oracle owner above only when `RAG_BACKEND=oracle`; PostgreSQL
acceptance ownership remains in the AI artifacts listed here.

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
