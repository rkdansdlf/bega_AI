-- Migrate stored RAG embeddings from 1536 dimensions to a 256-dimensional prefix.
-- This does not call any embedding API; it preserves existing vectors by slicing.
-- Run with psql autocommit and ON_ERROR_STOP=1. Each CONCURRENTLY statement
-- must run outside an explicit transaction.

CREATE TABLE IF NOT EXISTS rag_chunks_embedding_1536_backup_20260525 AS
SELECT id, embedding, embedding_model, embedding_dim, embedding_version
FROM rag_chunks
WHERE embedding IS NOT NULL
  AND embedding_dim = 1536;

DROP INDEX CONCURRENTLY IF EXISTS idx_rag_chunks_embedding_hnsw;
DROP INDEX CONCURRENTLY IF EXISTS idx_rag_chunks_embedding;
DROP INDEX CONCURRENTLY IF EXISTS idx_rag_chunks_embedding_halfvec_hnsw;

SET lock_timeout = '5s';
ALTER TABLE rag_chunks
ALTER COLUMN embedding TYPE vector(256)
USING subvector(embedding, 1, 256)::vector(256);
RESET lock_timeout;

UPDATE rag_chunks
SET embedding_dim = 256,
    embedding_version = 2
WHERE embedding IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_embedding_halfvec_hnsw
ON rag_chunks USING hnsw ((embedding::halfvec(256)) halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE embedding IS NOT NULL;
