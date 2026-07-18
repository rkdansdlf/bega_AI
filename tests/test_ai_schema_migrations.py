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
    for required in (
        "004_ai_ingest_checkpoints.sql",
        "ai_ingest_checkpoints",
        "INGEST_CHECKPOINT_INCOMPATIBLE",
        "INGEST_CHECKPOINT_CURSOR_UNAVAILABLE",
        "ai_ingest_checkpoint_events_total",
        "row_stale_cleanup",
        "rollback",
    ):
        assert required in text
