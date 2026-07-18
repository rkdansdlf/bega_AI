-- Optional operator-controlled index. Run only when
-- CHAT_SEMANTIC_CACHE_VECTOR_INDEX_ENABLED=true.
CREATE INDEX IF NOT EXISTS idx_chat_semantic_cache_embedding_hnsw
    ON chat_semantic_response_cache
    USING hnsw (question_embedding extensions.vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
