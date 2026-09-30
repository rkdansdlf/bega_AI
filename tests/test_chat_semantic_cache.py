"""Semantic response cache unit tests."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional

from app.core.chat_semantic_cache import (
    CREATE_SHADOW_OBSERVATION_TABLE_SQL,
    CREATE_VECTOR_INDEX_SQL,
    FILTERS_HASH_SCHEMA_VERSION,
    _build_filters_hash,
    _complete_shadow_observation_sync,
    _delete_semantic_by_key_sync,
    _get_semantic_sync,
    _record_shadow_observation_sync,
    _save_semantic_sync,
    get_semantic_cached_response,
)


def _run(coro):
    return asyncio.run(coro)


class _MockCursor:
    def __init__(self, row=None, rowcount=0):
        self._row = row
        self.rowcount = rowcount

    async def fetchone(self):
        return self._row


class _MockConn:
    def __init__(self, *, row=None, rowcount=0, fail: bool = False):
        self._row = row
        self._rowcount = rowcount
        self._fail = fail
        self.last_sql: Optional[str] = None
        self.last_params: Any = None
        self.executed_sql: list[str] = []

    async def execute(self, sql: str, params=None) -> _MockCursor:
        if self._fail:
            raise RuntimeError("db unavailable")
        self.last_sql = str(sql)
        self.last_params = params
        self.executed_sql.append(str(sql))
        return _MockCursor(row=self._row, rowcount=self._rowcount)


def test_build_filters_hash_is_order_stable() -> None:
    left = _build_filters_hash({"team_id": "LG", "season_year": 2025})
    right = _build_filters_hash({"season_year": 2025, "team_id": "LG"})

    assert left == right
    assert len(left) == 64


def test_filter_hash_schema_invalidates_entries_before_current_season_ttl_cap() -> None:
    assert FILTERS_HASH_SCHEMA_VERSION == "chat_semantic_filters_v2"


def test_get_semantic_sync_returns_high_similarity_row() -> None:
    expires = datetime.now(timezone.utc)
    row = (
        "semantic-key",
        "LG 흐름 알려줘",
        {"team_id": "LG"},
        "캐시 응답",
        "stats_lookup",
        "gemini-2.0-flash",
        "rag",
        4,
        expires,
        0.947,
        {"schema_version": 1, "verified": True},
    )
    settings = SimpleNamespace(embed_provider="openrouter", embed_dim=256)
    conn = _MockConn(row=row)

    result = _run(
        _get_semantic_sync(
            conn,
            question_embedding=[0.1, 0.2, 0.3],
            filters_json={"team_id": "LG"},
            settings=settings,
            threshold=0.93,
            limit=3,
        )
    )

    assert result is not None
    assert result["cache_key"] == "semantic-key"
    assert result["question_text"] == "LG 흐름 알려줘"
    assert result["filters_json"] == {"team_id": "LG"}
    assert result["response_text"] == "캐시 응답"
    assert result["source_tier"] == "rag"
    assert result["similarity"] == 0.947
    assert "chat_semantic_response_cache" in conn.last_sql
    assert "question_embedding <=> %s::vector" in conn.last_sql
    assert conn.last_params[1] == _build_filters_hash({"team_id": "LG"})
    assert conn.last_params[3] == 0.93


def test_save_semantic_sync_serializes_filters_and_embedding() -> None:
    settings = SimpleNamespace(embed_provider="openrouter", embed_dim=256)
    conn = _MockConn()

    _run(
        _save_semantic_sync(
            conn,
            cache_key="semantic-key",
            question_text="LG 흐름 알려줘",
            question_embedding=[0.1, 0.2, 0.3],
            filters_json={"season_year": 2025},
            intent="stats_lookup",
            source_tier="operator_data",
            response_text="응답",
            model_name="gemini-2.0-flash",
            settings=settings,
        )
    )

    assert "INSERT INTO chat_semantic_response_cache" in conn.last_sql
    assert conn.last_params[0] == "semantic-key"
    assert conn.last_params[2] == "[0.10000000,0.20000000,0.30000000]"
    assert json.loads(conn.last_params[4]) == {"season_year": 2025}
    assert conn.last_params[6] == "operator_data"


def test_get_semantic_sync_sets_hnsw_ef_search_when_configured() -> None:
    expires = datetime.now(timezone.utc)
    row = (
        "semantic-key",
        "LG 흐름 알려줘",
        {"team_id": "LG"},
        "캐시 응답",
        "stats_lookup",
        "gemini-2.0-flash",
        "rag",
        4,
        expires,
        0.947,
        {"schema_version": 1, "verified": True},
    )
    settings = SimpleNamespace(
        embed_provider="openrouter",
        embed_dim=256,
        chat_semantic_cache_hnsw_ef_search=64,
    )
    conn = _MockConn(row=row)

    result = _run(
        _get_semantic_sync(
            conn,
            question_embedding=[0.1, 0.2, 0.3],
            filters_json={"team_id": "LG"},
            settings=settings,
            threshold=0.93,
            limit=3,
        )
    )

    assert result is not None
    assert any("SET hnsw.ef_search = 64" in sql for sql in conn.executed_sql)


def test_semantic_cache_vector_index_sql_uses_hnsw() -> None:
    normalized = " ".join(CREATE_VECTOR_INDEX_SQL.lower().split())

    assert "using hnsw" in normalized
    assert "question_embedding vector_cosine_ops" in normalized
    assert "idx_chat_semantic_cache_embedding_hnsw" in normalized


def test_semantic_cache_ddl_creates_shadow_observation_table() -> None:
    normalized = " ".join(CREATE_SHADOW_OBSERVATION_TABLE_SQL.lower().split())

    assert (
        "create table if not exists chat_semantic_cache_shadow_observation"
        in normalized
    )
    assert "fresh_answer text" in normalized
    assert "idx_chat_semantic_shadow_observed_at" in normalized


def test_record_shadow_observation_sync_persists_candidate_pair() -> None:
    conn = _MockConn()

    _run(
        _record_shadow_observation_sync(
            conn,
            request_cache_key="request-key",
            candidate_cache_key="candidate-key",
            route="completion",
            question_text="LG 흐름은?",
            filters_json={"team_id": "LG"},
            cached_answer="후보 답변",
            similarity=0.95,
        )
    )

    assert "INSERT INTO chat_semantic_cache_shadow_observation" in conn.last_sql
    assert conn.last_params[0:4] == (
        "request-key",
        "candidate-key",
        "completion",
        "LG 흐름은?",
    )
    assert json.loads(conn.last_params[4]) == {"team_id": "LG"}
    assert conn.last_params[5:] == ("후보 답변", 0.95)


def test_complete_shadow_observation_sync_sets_fresh_answer() -> None:
    conn = _MockConn(rowcount=2)

    updated = _run(
        _complete_shadow_observation_sync(
            conn,
            request_cache_key="request-key",
            fresh_answer="새 응답",
        )
    )

    assert updated == 2
    assert "UPDATE chat_semantic_cache_shadow_observation" in conn.last_sql
    assert conn.last_params == ("새 응답", "request-key")


def test_public_semantic_lookup_treats_db_failure_as_miss() -> None:
    settings = SimpleNamespace(embed_provider="openrouter", embed_dim=256)
    conn = _MockConn(fail=True)

    result = _run(
        get_semantic_cached_response(
            conn,
            question_embedding=[0.1, 0.2, 0.3],
            filters_json=None,
            settings=settings,
            threshold=0.93,
            limit=3,
        )
    )

    assert result is None


def test_delete_semantic_by_key_sync_uses_cache_key() -> None:
    conn = _MockConn(rowcount=1)

    deleted = _run(_delete_semantic_by_key_sync(conn, "semantic-key"))

    assert deleted == 1
    assert "DELETE FROM chat_semantic_response_cache" in conn.last_sql
    assert conn.last_params == ("semantic-key",)
