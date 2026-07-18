CREATE TABLE IF NOT EXISTS ai_ingest_checkpoints (
    run_id uuid NOT NULL REFERENCES ai_ingest_runs(run_id),
    source_table varchar(128) NOT NULL,
    scope_key varchar(64) NOT NULL,
    cursor_version integer NOT NULL,
    cursor_signature varchar(64) NOT NULL,
    cursor_payload jsonb,
    committed_batches bigint NOT NULL DEFAULT 0,
    source_rows bigint NOT NULL DEFAULT 0,
    written_chunks bigint NOT NULL DEFAULT 0,
    reused_embeddings bigint NOT NULL DEFAULT 0,
    embedded_chunks bigint NOT NULL DEFAULT 0,
    max_updated_at timestamptz,
    completed boolean NOT NULL DEFAULT false,
    completed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, source_table),
    CONSTRAINT ck_ai_ingest_checkpoint_counts_nonnegative CHECK (
        committed_batches >= 0
        AND source_rows >= 0
        AND written_chunks >= 0
        AND reused_embeddings >= 0
        AND embedded_chunks >= 0
    ),
    CONSTRAINT ck_ai_ingest_checkpoint_cursor_present CHECK (
        source_rows = 0 OR cursor_payload IS NOT NULL
    ),
    CONSTRAINT ck_ai_ingest_checkpoint_completion_time CHECK (
        (completed = false AND completed_at IS NULL)
        OR (completed = true AND completed_at IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_ai_ingest_checkpoints_updated_at
    ON ai_ingest_checkpoints (updated_at);
