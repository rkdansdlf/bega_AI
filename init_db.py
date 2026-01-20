import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL")

sql_statements = [
    # 1. Extensions
    "create extension if not exists vector with schema extensions;",
    
    # 2. Table
    """
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
    """,
    
    # 3. Core Indexes
    "create index if not exists idx_rag_chunks_embedding on rag_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);",
    "create index if not exists idx_rag_chunks_content_tsv on rag_chunks using gin (content_tsv);",
    
    # 4. Metadata Indexes (Fixed: use btree for text extraction from jsonb)
    "create index if not exists idx_rag_chunks_season_year on rag_chunks (season_year);",
    "create index if not exists idx_rag_chunks_team_id on rag_chunks (team_id);",
    "create index if not exists idx_rag_chunks_source_table on rag_chunks (source_table);",
    "create index if not exists idx_rag_chunks_meta_league on rag_chunks ((meta->>'league'));",
    "create index if not exists idx_rag_chunks_season_team on rag_chunks (season_year, team_id);",
    "create index if not exists idx_rag_chunks_season_source on rag_chunks (season_year, source_table);",
    "create index if not exists idx_rag_chunks_team_source on rag_chunks (team_id, source_table);",
    "create index if not exists idx_rag_chunks_award_type on rag_chunks ((meta->>'award_type'));",
    "create index if not exists idx_rag_chunks_movement_type on rag_chunks ((meta->>'movement_type'));",
    "create index if not exists idx_rag_chunks_game_date on rag_chunks ((meta->>'game_date'));",
    "create index if not exists idx_rag_chunks_game_id on rag_chunks ((meta->>'game_id'));",
    "create unique index if not exists idx_rag_chunks_source on rag_chunks (source_table, source_row_id);"
]

try:
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()
    
    for stmt in sql_statements:
        try:
            print(f"Executing: {stmt[:50]}...")
            cur.execute(stmt)
        except Exception as e:
            print(f"Sub-error: {e}")
            
    cur.close()
    conn.close()
    print("DB Initialization complete.")
except Exception as e:
    print(f"Error: {e}")
