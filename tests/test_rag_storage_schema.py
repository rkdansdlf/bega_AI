from __future__ import annotations

from pathlib import Path


def _schema_sql() -> str:
    return (
        Path(__file__).parent.parent / "app" / "db" / "schema.sql"
    ).read_text(encoding="utf-8").lower()


def test_schema_sql_has_rag_storage_metadata_columns() -> None:
    content = _schema_sql()

    for column in (
        "source_type",
        "source_uri",
        "topic_key",
        "content_hash",
        "chunk_hash",
        "embedding_model",
        "embedding_dim",
        "embedding_version",
        "chunking_version",
        "quality_score",
        "is_active",
        "valid_from",
        "valid_to",
        "expires_at",
        "metadata jsonb",
    ):
        assert column in content


def test_schema_sql_uses_256_dimensional_embeddings() -> None:
    content = _schema_sql()

    assert "embedding vector(256)" in content
    assert "embedding vector(1536)" not in content
    assert "embedding_version int default 2" in content


def test_halfvec_index_migration_sql_does_not_reembed() -> None:
    content = (
        Path(__file__).parent.parent
        / "app"
        / "db"
        / "create_halfvec_hnsw_index.sql"
    ).read_text(encoding="utf-8").lower()

    assert "create index concurrently if not exists idx_rag_chunks_embedding_halfvec_hnsw" in content
    assert "embedding::halfvec(256)" in content
    assert "halfvec_cosine_ops" in content
    assert "does not call any embedding api" in content
    assert "alter column embedding type" not in content


def test_prefix_256_migration_slices_without_api_calls() -> None:
    content = (
        Path(__file__).parent.parent
        / "app"
        / "db"
        / "migrate_rag_embeddings_256_prefix.sql"
    ).read_text(encoding="utf-8").lower()

    assert "does not call any embedding api" in content
    assert "rag_chunks_embedding_1536_backup_20260525" in content
    assert "drop index concurrently if exists idx_rag_chunks_embedding_hnsw" in content
    assert "alter column embedding type vector(256)" in content
    assert "subvector(embedding, 1, 256)::vector(256)" in content
    assert "embedding_dim = 256" in content
    assert "embedding_version = 2" in content


def test_schema_sql_has_retrieval_events_table() -> None:
    content = _schema_sql()

    assert "create table if not exists rag_retrieval_events" in content
    assert "user_query text not null" in content
    assert "metadata_filter jsonb" in content
    assert "retrieved_chunk_ids jsonb" in content
    assert "error_type text" in content


def test_schema_sql_has_operator_data_tables() -> None:
    content = _schema_sql()

    for table in (
        "create table if not exists operator_data_items",
        "create table if not exists operator_season_events",
        "create table if not exists operator_schedule_items",
        "create table if not exists operator_roster_events",
    ):
        assert table in content
    assert "payload_hash text not null" in content
    assert "source_checked_at timestamptz not null" in content
    assert "confidence numeric not null" in content


def test_concurrent_index_sql_uses_concurrently() -> None:
    content = (
        Path(__file__).parent.parent
        / "app"
        / "db"
        / "rag_storage_indexes_concurrent.sql"
    ).read_text(encoding="utf-8").lower()

    assert "create index concurrently if not exists idx_rag_chunks_content_hash" in content
    assert "create index concurrently if not exists idx_rag_chunks_topic_key" in content
    assert "create index concurrently if not exists idx_rag_chunks_active_topic_key" in content
    assert "create index concurrently if not exists idx_rag_chunks_embedding_reuse" in content
    assert "create index concurrently if not exists idx_rag_chunks_metadata" in content
    assert "create index if not exists" not in content
    assert "begin" not in content
    assert "commit" not in content
