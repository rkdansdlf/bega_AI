-- Embedding generation registry.
--
-- A "generation" is an embedding signature (model, dim, version). Exactly one
-- generation is ACTIVE; retrieval only returns chunks whose signature matches
-- it, so old and new embeddings are never mixed in one result set. Switching
-- or rolling back is a single transaction that flips the active pointer.
--
-- This registry does NOT re-embed anything. Writing rows for a new generation
-- is owned by the crawler (KBO_playwright); activate only after the coverage
-- audit passes (scripts/manage_embedding_generation.py audit).
CREATE TABLE IF NOT EXISTS rag_embedding_generations (
    generation_id     text PRIMARY KEY,
    embedding_model   text        NOT NULL,
    embedding_dim     integer     NOT NULL,
    embedding_version integer     NOT NULL,
    status            text        NOT NULL DEFAULT 'BUILDING'
        CHECK (status IN ('BUILDING', 'READY', 'ACTIVE', 'RETIRED')),
    created_at        timestamptz NOT NULL DEFAULT now(),
    activated_at      timestamptz,
    retired_at        timestamptz,
    note              text
);

-- At most one ACTIVE generation, enforced by the database.
CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_embedding_generations_single_active
    ON rag_embedding_generations ((status))
    WHERE status = 'ACTIVE';

CREATE TABLE IF NOT EXISTS rag_embedding_generation_events (
    id            bigserial PRIMARY KEY,
    generation_id text        NOT NULL,
    action        text        NOT NULL,
    previous_id   text,
    detail        jsonb,
    created_at    timestamptz NOT NULL DEFAULT now()
);
