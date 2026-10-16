-- Concurrent indexes for existing production rag_chunks deployments.
-- Run these outside an explicit transaction.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_active
ON rag_chunks (is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_source_type
ON rag_chunks (source_type);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_topic_key
ON rag_chunks (topic_key);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_content_hash
ON rag_chunks (content_hash);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_chunk_hash
ON rag_chunks (chunk_hash);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_embedding_model
ON rag_chunks (embedding_model);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_embedding_reuse
ON rag_chunks (content_hash, embedding_model, embedding_dim, embedding_version, chunking_version)
WHERE embedding IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_metadata
ON rag_chunks USING gin (metadata);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_active_source_type_season
ON rag_chunks (source_type, season_year)
WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_active_team_season
ON rag_chunks (team_id, season_year)
WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_active_source_season
ON rag_chunks (source_table, season_year)
WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_active_topic_key
ON rag_chunks (topic_key)
WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_retrieval_events_created_at
ON rag_retrieval_events (created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_retrieval_events_success_error
ON rag_retrieval_events (success, error_type);
