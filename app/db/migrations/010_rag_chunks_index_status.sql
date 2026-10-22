-- rag_chunks.index_status: chunk lifecycle shared with the KBO_playwright indexer.
--
-- KBO_playwright retires chunks by setting index_status = 'DELETED' (see its
-- rag_index_lifecycle / tombstone tools). PostgreSQL retrieval excludes the
-- retired states (DELETED, TOMBSTONED, PURGED) and treats NULL as ACTIVE, and
-- scripts/sync_rag_chunks.py copies the column between databases.
--
-- Additive and idempotent: a database where KBO_playwright already created the
-- column (varchar(24) NOT NULL DEFAULT 'ACTIVE') is left untouched. A constant
-- DEFAULT is a metadata-only change on PostgreSQL 11+ (no table rewrite).
-- Retrieval detects the column at runtime, so deploying the app before this
-- migration keeps the previous behaviour (tombstones not filtered).
--
-- ADD COLUMN takes an ACCESS EXCLUSIVE lock, even when IF NOT EXISTS turns it
-- into a no-op. scripts/migrate_ai_runtime_schema.sh runs this file on every
-- deploy, so the ALTER only runs when the column is missing: re-runs read the
-- catalog only and take no lock on rag_chunks. The check resolves `rag_chunks`
-- through the session search_path (to_regclass), exactly like the unqualified
-- ALTER below and the other migrations, and reads pg_attribute so it does not
-- depend on column privileges (information_schema.columns hides columns the
-- role cannot access).
--
-- While the first ALTER waits behind a long transaction, every new SELECT on
-- rag_chunks queues behind it, so the wait is capped: on timeout psql
-- (ON_ERROR_STOP=1) aborts and the migration is simply re-run later. The
-- setting is session-local (each file is its own psql run).
-- See docs/rag-index-status.md for rollout and rollback.

SET lock_timeout = '5s';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_attribute
        WHERE attrelid = to_regclass('rag_chunks')
          AND attname = 'index_status'
          AND NOT attisdropped
    ) THEN
        ALTER TABLE rag_chunks
            ADD COLUMN IF NOT EXISTS index_status varchar(24) DEFAULT 'ACTIVE';
    END IF;
END
$$;

RESET lock_timeout;
