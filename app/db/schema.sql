-- AI-runtime-owned tables only.
--
-- rag_chunks (and its embedding indexes) is owned by the crawler
-- (KBO_playwright). This file no longer creates, alters, indexes, or backfills
-- it. The AI service only reads it; app/db/schema_contract.py validates the
-- columns and indexes it needs at startup.
create extension if not exists vector;
set search_path to public, extensions;

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

create table if not exists operator_data_items (
  queue_id text primary key,
  priority text,
  domain text not null,
  contract_code text not null,
  question text not null,
  operator_status text not null,
  validation_status text not null,
  apply_target text,
  payload jsonb not null,
  payload_hash text not null,
  source_name text not null,
  source_checked_at timestamptz not null,
  is_verified boolean not null,
  confidence numeric not null,
  applied_at timestamptz,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_operator_data_items_domain on operator_data_items (domain);
create index if not exists idx_operator_data_items_payload_hash on operator_data_items (payload_hash);
create index if not exists idx_operator_data_items_applied_at on operator_data_items (applied_at);

create table if not exists operator_season_events (
  queue_id text primary key references operator_data_items(queue_id) on delete cascade,
  season_year int not null,
  event_name text not null,
  event_date date not null,
  stadium_name text,
  payload_hash text not null,
  source_name text not null,
  source_checked_at timestamptz not null,
  is_verified boolean not null,
  confidence numeric not null,
  applied_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_operator_season_events_lookup
  on operator_season_events (season_year, event_date);

create table if not exists operator_schedule_items (
  queue_id text primary key references operator_data_items(queue_id) on delete cascade,
  game_date date not null,
  game_id text not null,
  home_team text not null,
  away_team text not null,
  stadium_name text,
  start_time text not null,
  game_status text not null,
  payload_hash text not null,
  source_name text not null,
  source_checked_at timestamptz not null,
  is_verified boolean not null,
  confidence numeric not null,
  applied_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_operator_schedule_items_date
  on operator_schedule_items (game_date, start_time);
create index if not exists idx_operator_schedule_items_game_id
  on operator_schedule_items (game_id);
create index if not exists idx_operator_schedule_items_teams
  on operator_schedule_items (home_team, away_team);

create table if not exists operator_roster_events (
  queue_id text primary key references operator_data_items(queue_id) on delete cascade,
  season_year int not null,
  team_code text not null,
  player_name text not null,
  roster_event_type text not null,
  effective_date date not null,
  status_text text not null,
  payload_hash text not null,
  source_name text not null,
  source_checked_at timestamptz not null,
  is_verified boolean not null,
  confidence numeric not null,
  applied_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_operator_roster_events_lookup
  on operator_roster_events (season_year, team_code, effective_date);
create index if not exists idx_operator_roster_events_player
  on operator_roster_events (player_name);

-- Coach Analysis Cache Table
create table if not exists coach_analysis_cache (
  cache_key varchar(64) primary key,  -- SHA256 Hash of (team_id, year, focus, question)
  team_id varchar(10) not null,
  year int not null,
  prompt_version varchar(32) not null, -- e.g. "v2"
  model_name varchar(50) not null,     -- e.g. "openrouter/free"
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
