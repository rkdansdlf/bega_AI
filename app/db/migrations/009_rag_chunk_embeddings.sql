-- Physical embedding generations (blue/green).
--
-- rag_chunks holds one inline `embedding` per chunk, so a second embedding
-- model cannot coexist with the first and a "rollback" has nothing to roll back
-- to. rag_chunk_embeddings stores one row per (generation, chunk): generation A
-- keeps serving while B is built, benchmarked, and activated; rolling back is
-- flipping the ACTIVE pointer (migration 007) back to rows that still exist.
--
-- Nothing changes until RAG_EMBEDDING_STORE=generations. Apply before deploy.

ALTER TABLE rag_embedding_generations
    ADD COLUMN IF NOT EXISTS mirror_inline boolean NOT NULL DEFAULT false;
ALTER TABLE rag_embedding_generations
    ADD COLUMN IF NOT EXISTS index_name text;
ALTER TABLE rag_embedding_generations
    ADD COLUMN IF NOT EXISTS index_ready boolean NOT NULL DEFAULT false;

-- `vector` without a dimension: generations may differ in dimension. ANN
-- indexes are per generation (partial, expression-cast to halfvec(dim)).
CREATE TABLE IF NOT EXISTS rag_chunk_embeddings (
    generation_id text        NOT NULL
        REFERENCES rag_embedding_generations (generation_id),
    chunk_id      bigint      NOT NULL
        REFERENCES rag_chunks (id) ON DELETE CASCADE,
    embedding     vector      NOT NULL,
    -- content_hash of the chunk when it was embedded. A row whose hash no longer
    -- matches rag_chunks.content_hash is stale and is never served.
    content_hash  text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (generation_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_rag_chunk_embeddings_chunk
    ON rag_chunk_embeddings (chunk_id);

-- Mirror inline writes into every generation whose signature matches the row's
-- own (embedding_model, embedding_dim, embedding_version) and that opted in
-- with mirror_inline. Existing writers (ingest, crawler) need no change to keep
-- the current generation populated; only a *new* model needs a dedicated writer.
--
-- Fires only when the embedding column is written (not on content_hash alone):
-- a content-only change must leave the old row stale (hash mismatch) instead of
-- re-pairing the old vector with the new hash.
CREATE OR REPLACE FUNCTION rag_chunks_mirror_embedding() RETURNS trigger AS $$
BEGIN
    IF NEW.embedding IS NULL THEN
        RETURN NEW;
    END IF;
    INSERT INTO rag_chunk_embeddings (generation_id, chunk_id, embedding, content_hash)
    SELECT g.generation_id, NEW.id, NEW.embedding, NEW.content_hash
    FROM rag_embedding_generations g
    WHERE g.mirror_inline
      AND g.status IN ('BUILDING', 'READY', 'ACTIVE')
      AND g.embedding_model = NEW.embedding_model
      AND g.embedding_dim = NEW.embedding_dim
      AND g.embedding_version = NEW.embedding_version
    ON CONFLICT (generation_id, chunk_id) DO UPDATE
        SET embedding = EXCLUDED.embedding,
            content_hash = EXCLUDED.content_hash,
            created_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_rag_chunks_mirror_embedding ON rag_chunks;
CREATE TRIGGER trg_rag_chunks_mirror_embedding
    AFTER INSERT OR UPDATE OF embedding ON rag_chunks
    FOR EACH ROW WHEN (NEW.embedding IS NOT NULL)
    EXECUTE FUNCTION rag_chunks_mirror_embedding();
