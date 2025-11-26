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

-- Unique constraint for data integrity and upserts
create unique index if not exists idx_rag_chunks_source on rag_chunks (source_table, source_row_id);
