-- Create a halfvec HNSW index for 1536-dimensional RAG embeddings.
-- This does not call any embedding API; it only adds a derived pgvector index.
-- Run outside an explicit transaction because CREATE INDEX CONCURRENTLY is used.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_embedding_halfvec_hnsw
ON rag_chunks USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE embedding IS NOT NULL;
