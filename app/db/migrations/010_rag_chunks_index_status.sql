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

ALTER TABLE rag_chunks
    ADD COLUMN IF NOT EXISTS index_status varchar(24) DEFAULT 'ACTIVE';
