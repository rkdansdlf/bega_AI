-- pgvector extension and rag_chunks table definition
create extension if not exists vector with schema extensions;

create table if not exists rag_chunks (
  id bigserial primary key,
  season_year int,
  season_id int,
  league_type_code int,
  team_id varchar(10),
  player_id varchar(20),
  source_table text not null,
  source_row_id text not null,
  title text,
  content text not null,
  content_tsv tsvector generated always as (to_tsvector('simple', coalesce(content, ''))) stored,
  embedding vector(256),
  meta jsonb default '{}'::jsonb,
  metadata jsonb default '{}'::jsonb,
  source_type text,
  source_uri text,
  topic_key text,
  content_hash text,
  chunk_hash text,
  embedding_model text,
  embedding_dim int,
  embedding_version int default 2,
  chunking_version int default 1,
  quality_score numeric default 0.50,
  is_active boolean default true,
  valid_from timestamptz,
  valid_to timestamptz,
  expires_at timestamptz,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Additive migration path for existing rag_chunks deployments.
alter table rag_chunks add column if not exists metadata jsonb default '{}'::jsonb;
alter table rag_chunks add column if not exists source_type text;
alter table rag_chunks add column if not exists source_uri text;
alter table rag_chunks add column if not exists topic_key text;
alter table rag_chunks add column if not exists content_hash text;
alter table rag_chunks add column if not exists chunk_hash text;
alter table rag_chunks add column if not exists embedding_model text;
alter table rag_chunks add column if not exists embedding_dim int;
alter table rag_chunks add column if not exists embedding_version int default 2;
alter table rag_chunks add column if not exists chunking_version int default 1;
alter table rag_chunks add column if not exists quality_score numeric default 0.50;
alter table rag_chunks add column if not exists is_active boolean default true;
alter table rag_chunks add column if not exists valid_from timestamptz;
alter table rag_chunks add column if not exists valid_to timestamptz;
alter table rag_chunks add column if not exists expires_at timestamptz;

update rag_chunks
set metadata = coalesce(meta, '{}'::jsonb)
where (metadata is null or metadata = '{}'::jsonb)
  and meta is not null
  and meta <> '{}'::jsonb;

-- Vector and text search indexes
-- halfvec HNSW 인덱스 (신규 설치 기본). 운영 마이그레이션은 scripts/create_vector_index.py 사용.
-- pgvector >= 0.7.0 필요. m=16: 레이어당 최대 연결 수, ef_construction=64: 빌드 정확도.
create index if not exists idx_rag_chunks_embedding_halfvec_hnsw on rag_chunks using hnsw ((embedding::halfvec(256)) halfvec_cosine_ops) with (m = 16, ef_construction = 64) where embedding is not null;
-- vector HNSW 인덱스는 halfvec 전환 후 운영 cleanup 대상입니다.
-- create index if not exists idx_rag_chunks_embedding_hnsw on rag_chunks using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64) where embedding is not null;
-- 기존 IVFFlat 인덱스 (레거시, 운영 마이그레이션 후 scripts/create_vector_index.py --drop-ivfflat 으로 제거)
-- create index if not exists idx_rag_chunks_embedding on rag_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 644);
create index if not exists idx_rag_chunks_content_tsv on rag_chunks using gin (content_tsv);

-- Metadata filtering indexes for performance
create index if not exists idx_rag_chunks_season_year on rag_chunks (season_year);
create index if not exists idx_rag_chunks_team_id on rag_chunks (team_id);
create index if not exists idx_rag_chunks_source_table on rag_chunks (source_table);
create index if not exists idx_rag_chunks_meta_league on rag_chunks ((meta->>'league'));
create index if not exists idx_rag_chunks_active on rag_chunks (is_active);
create index if not exists idx_rag_chunks_source_type on rag_chunks (source_type);
create index if not exists idx_rag_chunks_topic_key on rag_chunks (topic_key);
create index if not exists idx_rag_chunks_content_hash on rag_chunks (content_hash);
create index if not exists idx_rag_chunks_chunk_hash on rag_chunks (chunk_hash);
create index if not exists idx_rag_chunks_embedding_model on rag_chunks (embedding_model);
create index if not exists idx_rag_chunks_embedding_reuse
  on rag_chunks (content_hash, embedding_model, embedding_dim, embedding_version, chunking_version)
  where embedding is not null;
create index if not exists idx_rag_chunks_metadata on rag_chunks using gin (metadata);

-- Composite indexes for common filter combinations
create index if not exists idx_rag_chunks_season_team on rag_chunks (season_year, team_id);
create index if not exists idx_rag_chunks_season_source on rag_chunks (season_year, source_table);
create index if not exists idx_rag_chunks_team_source on rag_chunks (team_id, source_table);
create index if not exists idx_rag_chunks_active_source_type_season on rag_chunks (source_type, season_year)
  where is_active = true;
create index if not exists idx_rag_chunks_active_team_season on rag_chunks (team_id, season_year)
  where is_active = true;
create index if not exists idx_rag_chunks_active_source_season on rag_chunks (source_table, season_year)
  where is_active = true;
create index if not exists idx_rag_chunks_active_topic_key on rag_chunks (topic_key)
  where is_active = true;

-- New indexes for awards, movements, and game queries
create index if not exists idx_rag_chunks_award_type on rag_chunks ((meta->>'award_type'));
create index if not exists idx_rag_chunks_movement_type on rag_chunks ((meta->>'movement_type'));
create index if not exists idx_rag_chunks_game_date on rag_chunks ((meta->>'game_date'));
create index if not exists idx_rag_chunks_game_id on rag_chunks ((meta->>'game_id'));

-- Composite index for award queries by season
create index if not exists idx_rag_chunks_season_award on rag_chunks (season_year)
  where source_table = 'awards';

-- Unique constraint for data integrity and upserts
create unique index if not exists idx_rag_chunks_source on rag_chunks (source_table, source_row_id);

create table if not exists rag_retrieval_events (
  event_id bigserial primary key,
  user_query text not null,
  intent text,
  rewritten_queries jsonb default '[]'::jsonb,
  metadata_filter jsonb default '{}'::jsonb,
  retrieved_chunk_ids jsonb default '[]'::jsonb,
  selected_chunk_ids jsonb default '[]'::jsonb,
  scores jsonb default '[]'::jsonb,
  latency_ms integer,
  success boolean not null default true,
  error_type text,
  created_at timestamptz default now()
);

create index if not exists idx_rag_retrieval_events_created_at on rag_retrieval_events (created_at);
create index if not exists idx_rag_retrieval_events_success_error on rag_retrieval_events (success, error_type);

create table if not exists rag_ingest_jobs (
  job_id bigserial primary key,
  job_type text not null,
  source_tables jsonb default '[]'::jsonb,
  status text not null default 'running',
  chunk_count integer not null default 0,
  skipped_count integer not null default 0,
  reembedded_count integer not null default 0,
  error text,
  started_at timestamptz default now(),
  finished_at timestamptz
);

create index if not exists idx_rag_ingest_jobs_started_at on rag_ingest_jobs (started_at);
create index if not exists idx_rag_ingest_jobs_status on rag_ingest_jobs (status);

-- Coach Analysis Cache Table
create table if not exists coach_analysis_cache (
  cache_key varchar(64) primary key,  -- SHA256 Hash of (team_id, year, focus, question)
  team_id varchar(10) not null,
  year int not null,
  prompt_version varchar(32) not null, -- e.g. "v2"
  model_name varchar(50) not null,     -- e.g. "upstage/solar-pro-3:free"
  status varchar(20) not null check (status in ('PENDING', 'COMPLETED', 'FAILED')),
  response_json jsonb,                 -- Completed analysis result
  error_message text,                  -- Failure reason
  error_code varchar(64),
  attempt_count int not null default 0,
  lease_owner varchar(80),
  lease_expires_at timestamptz,
  last_heartbeat_at timestamptz,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

alter table coach_analysis_cache
  alter column prompt_version type varchar(32);
alter table coach_analysis_cache
  add column if not exists error_code varchar(64);
alter table coach_analysis_cache
  add column if not exists attempt_count int not null default 0;
alter table coach_analysis_cache
  add column if not exists lease_owner varchar(80);
alter table coach_analysis_cache
  add column if not exists lease_expires_at timestamptz;
alter table coach_analysis_cache
  add column if not exists last_heartbeat_at timestamptz;

-- Index for expiration and lookup
create index if not exists idx_coach_cache_created_at on coach_analysis_cache (created_at);
create index if not exists idx_coach_cache_team_year on coach_analysis_cache (team_id, year);
