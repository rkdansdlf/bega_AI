from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_DIR = ROOT / "app" / "db" / "migrations"
MANAGED_MIGRATION_SCRIPT = ROOT / "scripts" / "migrate_ai_runtime_schema.sh"


def test_ai_runtime_schema_migration_contains_runtime_cache_tables():
    sql = (MIGRATION_DIR / "001_ai_runtime_cache.sql").read_text(encoding="utf-8")

    for table_name in (
        "coach_analysis_cache",
        "chat_response_cache",
        "chat_semantic_response_cache",
        "chat_semantic_cache_shadow_observation",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table_name}" in sql


def test_ai_runtime_schema_vector_index_is_explicit_operator_step():
    sql = (MIGRATION_DIR / "002_chat_semantic_cache_vector_index.sql").read_text(
        encoding="utf-8"
    )

    assert "CREATE INDEX IF NOT EXISTS idx_chat_semantic_cache_embedding_hnsw" in sql
    assert "extensions.vector_cosine_ops" in sql
    assert "CREATE INDEX CONCURRENTLY" not in sql


def test_ingest_orchestration_migration_defines_durable_run_and_watermark_tables():
    sql = (MIGRATION_DIR / "003_ai_ingest_orchestration.sql").read_text(
        encoding="utf-8"
    )

    assert "CREATE TABLE IF NOT EXISTS ai_ingest_runs" in sql
    assert "CREATE TABLE IF NOT EXISTS ai_ingest_watermarks" in sql
    assert "scope_key varchar(64) NOT NULL" in sql
    assert "PRIMARY KEY (source_table, scope_key)" in sql
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in sql
    assert "WHERE status IN ('QUEUED', 'RUNNING')" in sql


def test_ingest_checkpoint_migration_defines_durable_progress_table():
    sql = (MIGRATION_DIR / "004_ai_ingest_checkpoints.sql").read_text(
        encoding="utf-8"
    )

    assert "CREATE TABLE IF NOT EXISTS ai_ingest_checkpoints" in sql
    assert "PRIMARY KEY (run_id, source_table)" in sql
    assert "cursor_signature varchar(64) NOT NULL" in sql
    assert "cursor_payload jsonb" in sql
    assert "source_rows = 0 OR cursor_payload IS NOT NULL" in sql
    assert "idx_ai_ingest_checkpoints_updated_at" in sql


def test_managed_migration_script_applies_ingest_checkpoint_migration_in_order():
    script = MANAGED_MIGRATION_SCRIPT.read_text(encoding="utf-8")
    migration_paths = (
        '"${AI_ROOT}/app/db/migrations/001_ai_runtime_cache.sql"',
        '"${AI_ROOT}/app/db/migrations/003_ai_ingest_orchestration.sql"',
        '"${AI_ROOT}/app/db/migrations/004_ai_ingest_checkpoints.sql"',
    )

    assert all(path in script for path in migration_paths)
    assert [script.index(path) for path in migration_paths] == sorted(
        script.index(path) for path in migration_paths
    )


def test_data_sync_runbook_documents_persistent_checkpoint_operations():
    text = (ROOT / "docs" / "data-sync-orchestration-runbook.md").read_text(
        encoding="utf-8"
    )
    for statement in (
        "Managed migration script applies `001_ai_runtime_cache.sql` -> "
        "`003_ai_ingest_orchestration.sql` -> "
        "`004_ai_ingest_checkpoints.sql`.",
        "Compatibility startup applies `003_ai_ingest_orchestration.sql` -> "
        "`004_ai_ingest_checkpoints.sql`.",
        "Exactly one `ai_ingest_checkpoints` row per `(run_id, source_table)` "
        "is retained after terminal completion.",
        "Resume uses a typed ascending keyset and never an offset.",
        "Chunks and their cursor commit together under the lease fence.",
        "`source_file` static documents create no checkpoint row and restart atomically.",
        "The generic terminal status error code remains "
        "`INGEST_EXECUTION_FAILED`; the five `INGEST_CHECKPOINT_*` "
        "identifiers are internal typed exception and metric classifications, "
        "not status-payload error codes.",
        "`MANUAL_BASEBALL_DATA_REQUIRED` remains the separate operator handoff.",
        "Operators observe checkpoint failures through "
        "`ai_ingest_checkpoint_events_total{source_table,result}` and AI worker "
        "logs with `run_id` and `error_type`.",
        "Checkpoint metric results are limited to `created`, `advanced`, "
        "`completed`, `resumed`, `incompatible`, and `rejected`.",
        "Rollback preserves migration 004 and retained checkpoint audit rows; "
        "older code ignores them.",
        "No automatic checkpoint retention job exists.",
        "A separately approved local-only disposable live PostgreSQL smoke is "
        "required to remove the residual PostgreSQL risk.",
    ):
        assert statement in text
