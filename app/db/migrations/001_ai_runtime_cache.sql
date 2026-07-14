-- AI runtime cache schema.
-- Execute with the PostgreSQL migration/operator role before using
-- AI_DB_SCHEMA_MODE=managed.
-- The pgvector extension is a DBA/database prerequisite and is not installed
-- by this application-cache migration.

CREATE TABLE IF NOT EXISTS coach_analysis_cache (
    cache_key varchar(64) primary key,
    team_id varchar(10) not null,
    year int not null,
    prompt_version varchar(32) not null,
    model_name varchar(50) not null,
    status varchar(20) not null check (status in ('PENDING', 'COMPLETED', 'FAILED')),
    response_json jsonb,
    error_message text,
    error_code varchar(64),
    attempt_count int not null default 0,
    lease_owner varchar(80),
    lease_expires_at timestamptz,
    last_heartbeat_at timestamptz,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

ALTER TABLE coach_analysis_cache
    ALTER COLUMN prompt_version TYPE varchar(32);
ALTER TABLE coach_analysis_cache
    ADD COLUMN IF NOT EXISTS error_code varchar(64);
ALTER TABLE coach_analysis_cache
    ADD COLUMN IF NOT EXISTS attempt_count int NOT NULL DEFAULT 0;
ALTER TABLE coach_analysis_cache
    ADD COLUMN IF NOT EXISTS lease_owner varchar(80);
ALTER TABLE coach_analysis_cache
    ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz;
ALTER TABLE coach_analysis_cache
    ADD COLUMN IF NOT EXISTS last_heartbeat_at timestamptz;

CREATE INDEX IF NOT EXISTS idx_coach_cache_created_at
    ON coach_analysis_cache (created_at);
CREATE INDEX IF NOT EXISTS idx_coach_cache_team_year
    ON coach_analysis_cache (team_id, year);

CREATE TABLE IF NOT EXISTS chat_response_cache (
    cache_key varchar(64) primary key,
    question_text text not null,
    filters_json jsonb,
    intent varchar(50),
    response_text text not null,
    model_name varchar(100),
    hit_count integer not null default 0,
    created_at timestamptz not null default now(),
    expires_at timestamptz not null
);

CREATE INDEX IF NOT EXISTS idx_chat_cache_expires_at
    ON chat_response_cache (expires_at);
CREATE INDEX IF NOT EXISTS idx_chat_cache_created_at
    ON chat_response_cache (created_at);

CREATE TABLE IF NOT EXISTS chat_semantic_response_cache (
    cache_key varchar(64) primary key,
    question_text text not null,
    question_embedding extensions.vector(256) not null,
    filters_hash varchar(64) not null,
    filters_json jsonb,
    intent varchar(50),
    source_tier varchar(50),
    response_text text not null,
    model_name varchar(100),
    embedding_signature varchar(180) not null,
    hit_count integer not null default 0,
    created_at timestamptz not null default now(),
    expires_at timestamptz not null
);

ALTER TABLE chat_semantic_response_cache
    ADD COLUMN IF NOT EXISTS source_tier varchar(50);

CREATE INDEX IF NOT EXISTS idx_chat_semantic_cache_lookup
    ON chat_semantic_response_cache (filters_hash, embedding_signature, expires_at);
CREATE INDEX IF NOT EXISTS idx_chat_semantic_cache_created_at
    ON chat_semantic_response_cache (created_at);
CREATE INDEX IF NOT EXISTS idx_chat_semantic_cache_expires_at
    ON chat_semantic_response_cache (expires_at);

CREATE TABLE IF NOT EXISTS chat_semantic_cache_shadow_observation (
    id bigserial primary key,
    request_cache_key varchar(64) not null,
    candidate_cache_key varchar(64) not null,
    route varchar(20) not null,
    question_text text not null,
    filters_json jsonb,
    cached_answer text not null,
    fresh_answer text,
    similarity double precision not null,
    observed_at timestamptz not null default now(),
    completed_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_chat_semantic_shadow_observed_at
    ON chat_semantic_cache_shadow_observation (observed_at);
CREATE INDEX IF NOT EXISTS idx_chat_semantic_shadow_request_key
    ON chat_semantic_cache_shadow_observation (request_cache_key);
