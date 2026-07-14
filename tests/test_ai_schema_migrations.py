from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_DIR = ROOT / "app" / "db" / "migrations"


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
