from __future__ import annotations

from pathlib import Path


def _schema_sql() -> str:
    return (
        (Path(__file__).parent.parent / "app" / "db" / "schema.sql")
        .read_text(encoding="utf-8")
        .lower()
    )










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


