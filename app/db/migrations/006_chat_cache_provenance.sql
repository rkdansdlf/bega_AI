-- Preserve answer provenance in chat caches so a cache hit reports the
-- original verified flag / sources instead of a hard-coded verified=true.
-- Legacy rows keep NULL and are served as unverified.
ALTER TABLE chat_response_cache
    ADD COLUMN IF NOT EXISTS provenance_json JSONB;
ALTER TABLE chat_semantic_response_cache
    ADD COLUMN IF NOT EXISTS provenance_json JSONB;
