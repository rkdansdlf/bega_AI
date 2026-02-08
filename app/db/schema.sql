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
  embedding vector(1536),
  meta jsonb default '{}'::jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Vector and text search indexes
create index if not exists idx_rag_chunks_embedding on rag_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);
create index if not exists idx_rag_chunks_content_tsv on rag_chunks using gin (content_tsv);

-- Metadata filtering indexes for performance
create index if not exists idx_rag_chunks_season_year on rag_chunks (season_year);
create index if not exists idx_rag_chunks_team_id on rag_chunks (team_id);
create index if not exists idx_rag_chunks_source_table on rag_chunks (source_table);
create index if not exists idx_rag_chunks_meta_league on rag_chunks using gin ((meta->>'league'));

-- Composite indexes for common filter combinations
create index if not exists idx_rag_chunks_season_team on rag_chunks (season_year, team_id);
create index if not exists idx_rag_chunks_season_source on rag_chunks (season_year, source_table);
create index if not exists idx_rag_chunks_team_source on rag_chunks (team_id, source_table);

-- New indexes for awards, movements, and game queries
create index if not exists idx_rag_chunks_award_type on rag_chunks using gin ((meta->>'award_type'));
create index if not exists idx_rag_chunks_movement_type on rag_chunks using gin ((meta->>'movement_type'));
create index if not exists idx_rag_chunks_game_date on rag_chunks using gin ((meta->>'game_date'));
create index if not exists idx_rag_chunks_game_id on rag_chunks using gin ((meta->>'game_id'));

-- Composite index for award queries by season
create index if not exists idx_rag_chunks_season_award on rag_chunks (season_year)
  where source_table = 'awards';

-- Unique constraint for data integrity and upserts
create unique index if not exists idx_rag_chunks_source on rag_chunks (source_table, source_row_id);

-- Coach Analysis Cache Table
create table if not exists coach_analysis_cache (
  cache_key varchar(64) primary key,  -- SHA256 Hash of (team_id, year, focus, question)
  team_id varchar(10) not null,
  year int not null,
  prompt_version varchar(10) not null, -- e.g. "v2"
  model_name varchar(50) not null,     -- e.g. "solar-pro-3"
  status varchar(20) not null check (status in ('PENDING', 'COMPLETED', 'FAILED')),
  response_json jsonb,                 -- Completed analysis result
  error_message text,                  -- Failure reason
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Index for expiration and lookup
create index if not exists idx_coach_cache_created_at on coach_analysis_cache (created_at);
create index if not exists idx_coach_cache_team_year on coach_analysis_cache (team_id, year);
