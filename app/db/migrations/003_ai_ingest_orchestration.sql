CREATE TABLE IF NOT EXISTS ai_ingest_runs (
    run_id uuid PRIMARY KEY,
    request_key varchar(64) NOT NULL,
    trigger_source varchar(32) NOT NULL,
    status varchar(48) NOT NULL,
    request_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    requested_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    heartbeat_at timestamptz,
    finished_at timestamptz,
    lease_owner varchar(128),
    lease_expires_at timestamptz,
    recovery_attempts integer NOT NULL DEFAULT 0,
    error_code varchar(96),
    error_message varchar(1000),
    table_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ck_ai_ingest_runs_status CHECK (
        status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'MANUAL_BASEBALL_DATA_REQUIRED')
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_ai_ingest_runs_active_request
    ON ai_ingest_runs (request_key)
    WHERE status IN ('QUEUED', 'RUNNING');

CREATE INDEX IF NOT EXISTS idx_ai_ingest_runs_status_requested
    ON ai_ingest_runs (status, requested_at);

CREATE TABLE IF NOT EXISTS ai_ingest_watermarks (
    source_table varchar(128) NOT NULL,
    scope_key varchar(64) NOT NULL,
    last_successful_updated_at timestamptz,
    last_run_id uuid REFERENCES ai_ingest_runs(run_id),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source_table, scope_key)
);
