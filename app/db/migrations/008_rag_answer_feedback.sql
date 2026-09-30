-- User feedback on chatbot answers, joinable to retrieval events / fingerprints.
-- Stores a rating and optional free-text; never user identity. `answer_key` is
-- the cache key prefix / request id the BFF received in the answer meta.
CREATE TABLE IF NOT EXISTS rag_answer_feedback (
    feedback_id  bigserial PRIMARY KEY,
    request_id   text,
    answer_key   text,
    question     text        NOT NULL,
    rating       text        NOT NULL CHECK (rating IN ('UP', 'DOWN')),
    reason       text,
    corrected_fact text,
    fingerprint  jsonb,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rag_answer_feedback_created_at
    ON rag_answer_feedback (created_at);
CREATE INDEX IF NOT EXISTS idx_rag_answer_feedback_rating
    ON rag_answer_feedback (rating, created_at);
