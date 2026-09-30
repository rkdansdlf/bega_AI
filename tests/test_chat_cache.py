"""chat_cache.py 단위 테스트.

test_retrieval.py와 동일한 mock DB 패턴을 사용한다.
conn.execute(sql, params)가 cursor를 반환하는 psycopg3 스타일을 흉내낸다.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, List, Optional

from app.core.chat_cache import (
    _cleanup_sync,
    _delete_by_intent_sync,
    _delete_by_key_sync,
    _get_stats_sync,
    _get_sync,
    _save_sync,
    _update_hit_count_sync,
)


def _run(coro):
    return asyncio.run(coro)


# ── Mock psycopg3-style connection/cursor ─────────────────────────────────────

class _MockCursor:
    def __init__(self, row=None, rows=None, rowcount=0):
        self._row = row
        self._rows = rows or []
        self.rowcount = rowcount
        self.executed: list[tuple[str, Any]] = []

    async def fetchone(self):
        return self._row

    async def fetchall(self):
        return self._rows


class _MockConn:
    def __init__(self, *, row=None, rows=None, rowcount=0):
        self._row = row
        self._rows = rows
        self._rowcount = rowcount
        self.last_sql: Optional[str] = None
        self.last_params: Any = None

    async def execute(self, sql: str, params=None) -> _MockCursor:
        self.last_sql = str(sql)
        self.last_params = params
        return _MockCursor(row=self._row, rows=self._rows or [], rowcount=self._rowcount)


# ── TestGetCachedResponse (sync helper) ───────────────────────────────────────

class TestGetSync:
    def test_hit_returns_dict(self):
        from datetime import datetime, timezone
        expires = datetime.now(timezone.utc)
        row = ("안녕하세요 응답", "stats_lookup", "gemini-pro", 3, expires, {"schema_version": 1})
        conn = _MockConn(row=row)
        result = _run(_get_sync(conn, "abc123"))
        assert result is not None
        assert result["response_text"] == "안녕하세요 응답"
        assert result["intent"] == "stats_lookup"
        assert result["model_name"] == "gemini-pro"
        assert result["hit_count"] == 3
        assert result["provenance"] == {"schema_version": 1}

    def test_miss_returns_none(self):
        conn = _MockConn(row=None)
        result = _run(_get_sync(conn, "nonexistent_key"))
        assert result is None

    def test_select_sql_contains_cache_key_param(self):
        conn = _MockConn(row=None)
        _run(_get_sync(conn, "mykey_abc"))
        assert "chat_response_cache" in conn.last_sql
        assert conn.last_params == ("mykey_abc",)

    def test_select_sql_filters_expires_at(self):
        conn = _MockConn(row=None)
        _run(_get_sync(conn, "key"))
        assert "expires_at" in conn.last_sql


# ── TestSaveSync ──────────────────────────────────────────────────────────────

class TestSaveSync:
    def test_upsert_sql_executed(self):
        conn = _MockConn()
        _run(_save_sync(
            conn,
            cache_key="k1",
            question_text="KIA 오늘 성적?",
            filters_json={"season_year": 2025},
            intent="stats_lookup",
            response_text="KIA는 현재 1위입니다.",
            model_name="gemini-pro",
        ))
        assert "INSERT INTO chat_response_cache" in conn.last_sql
        assert "ON CONFLICT" in conn.last_sql

    def test_filters_none_serialized_as_none(self):
        conn = _MockConn()
        _run(_save_sync(
            conn,
            cache_key="k2",
            question_text="홈런왕?",
            filters_json=None,
            intent="stats_lookup",
            response_text="김도영입니다.",
            model_name=None,
        ))
        params = conn.last_params
        # filters_serialized is the 3rd param (index 2)
        assert params[2] is None

    def test_filters_dict_serialized_as_json_string(self):
        conn = _MockConn()
        _run(_save_sync(
            conn,
            cache_key="k3",
            question_text="팀 성적?",
            filters_json={"season_year": 2024},
            intent="player_profile",
            response_text="응답 텍스트",
            model_name="gemini-flash",
        ))
        params = conn.last_params
        filters_str = params[2]
        assert filters_str is not None
        parsed = json.loads(filters_str)
        assert parsed["season_year"] == 2024

    def test_intent_and_model_name_in_params(self):
        conn = _MockConn()
        _run(_save_sync(
            conn,
            cache_key="k4",
            question_text="ERA 순위?",
            filters_json=None,
            intent="comparison",
            response_text="투수 ERA 비교입니다.",
            model_name="openrouter-llama",
        ))
        params = conn.last_params
        assert "comparison" in params
        assert "openrouter-llama" in params

    def test_cache_key_is_first_param(self):
        conn = _MockConn()
        _run(_save_sync(
            conn,
            cache_key="my_key",
            question_text="질문",
            filters_json=None,
            intent=None,
            response_text="응답",
            model_name=None,
        ))
        assert conn.last_params[0] == "my_key"


# ── TestUpdateHitCount ────────────────────────────────────────────────────────

class TestUpdateHitCountSync:
    def test_update_sql_executed(self):
        conn = _MockConn()
        _run(_update_hit_count_sync(conn, "abc"))
        assert "UPDATE chat_response_cache" in conn.last_sql
        assert "hit_count" in conn.last_sql

    def test_cache_key_in_params(self):
        conn = _MockConn()
        _run(_update_hit_count_sync(conn, "target_key"))
        assert conn.last_params == ("target_key",)


# ── TestCleanupSync ───────────────────────────────────────────────────────────

class TestCleanupSync:
    def test_delete_sql_executed(self):
        conn = _MockConn(rowcount=0)
        _run(_cleanup_sync(conn))
        assert "DELETE FROM chat_response_cache" in conn.last_sql
        assert "expires_at" in conn.last_sql

    def test_returns_rowcount(self):
        conn = _MockConn(rowcount=5)
        deleted = _run(_cleanup_sync(conn))
        assert deleted == 5

    def test_returns_zero_when_nothing_deleted(self):
        conn = _MockConn(rowcount=0)
        assert _run(_cleanup_sync(conn)) == 0


# ── TestGetStatsSync ──────────────────────────────────────────────────────────

class TestGetStatsSync:
    def test_returns_list_of_dicts(self):
        rows = [
            ("stats_lookup", 42, 3.14),
            ("player_profile", 10, 1.0),
        ]
        conn = _MockConn(rows=rows)
        result = _run(_get_stats_sync(conn))
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["intent"] == "stats_lookup"
        assert result[0]["count"] == 42
        assert result[0]["avg_hits"] == pytest.approx(3.14, rel=1e-6)

    def test_empty_returns_empty_list(self):
        conn = _MockConn(rows=[])
        assert _run(_get_stats_sync(conn)) == []

    def test_null_avg_hits_defaults_to_zero(self):
        rows = [("freeform", 5, None)]
        conn = _MockConn(rows=rows)
        result = _run(_get_stats_sync(conn))
        assert result[0]["avg_hits"] == 0.0


# ── TestDeleteOperations ──────────────────────────────────────────────────────

class TestDeleteByIntentSync:
    def test_delete_sql_uses_intent(self):
        conn = _MockConn(rowcount=3)
        _run(_delete_by_intent_sync(conn, "stats_lookup"))
        assert "DELETE FROM chat_response_cache" in conn.last_sql
        assert conn.last_params == ("stats_lookup",)

    def test_returns_rowcount(self):
        conn = _MockConn(rowcount=7)
        assert _run(_delete_by_intent_sync(conn, "stats_lookup")) == 7

    def test_returns_zero_if_not_found(self):
        conn = _MockConn(rowcount=0)
        assert _run(_delete_by_intent_sync(conn, "nonexistent_intent")) == 0


class TestDeleteByKeySync:
    def test_delete_sql_uses_cache_key(self):
        conn = _MockConn(rowcount=1)
        _run(_delete_by_key_sync(conn, "specific_key"))
        assert "DELETE FROM chat_response_cache" in conn.last_sql
        assert conn.last_params == ("specific_key",)

    def test_returns_1_when_deleted(self):
        conn = _MockConn(rowcount=1)
        assert _run(_delete_by_key_sync(conn, "k")) == 1

    def test_returns_0_if_key_missing(self):
        conn = _MockConn(rowcount=0)
        assert _run(_delete_by_key_sync(conn, "missing")) == 0


import pytest
