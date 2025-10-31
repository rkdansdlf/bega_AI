-- Example parameterised query snippets for ingestion and retrieval.

-- Upsert chunk
-- INSERT INTO rag_chunks (season_year, league_type_code, team_id, player_id, source_table, source_row_id, title, content, embedding, meta)
-- VALUES (%(season_year)s, %(league_type_code)s, %(team_id)s, %(player_id)s, %(source_table)s, %(source_row_id)s, %(title)s, %(content)s, %(embedding)s, %(meta)s)
-- ON CONFLICT (source_table, source_row_id)
-- DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, meta = EXCLUDED.meta, updated_at = now();
