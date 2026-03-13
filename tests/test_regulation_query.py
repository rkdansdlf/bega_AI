from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from app.tools.query_logging import (
    ACTION_GET_REGULATION_BY_CATEGORY,
    ACTION_VALIDATE_REGULATION_REFERENCE,
    REGULATION_QUERY_COMPONENT,
)
from app.tools.regulation_query import RegulationQueryTool


class _FakeCursor:
    def __init__(
        self,
        *,
        fetchone_results: list[Any] | None = None,
        fetchall_results: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self._fetchone_results = list(fetchone_results or [])
        self._fetchall_results = list(fetchall_results or [])
        self.executed: list[tuple[str, Any]] = []

    def execute(self, query: str, params: Any = None) -> None:
        self.executed.append((str(query), params))

    def fetchone(self) -> Any:
        if self._fetchone_results:
            return self._fetchone_results.pop(0)
        return None

    def fetchall(self) -> list[dict[str, Any]]:
        if self._fetchall_results:
            return self._fetchall_results.pop(0)
        return []

    def close(self) -> None:
        return None

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeConnection:
    def __init__(self, cursors: list[_FakeCursor]) -> None:
        self._cursors = list(cursors)
        self.closed = False

    def cursor(self, row_factory=None):  # noqa: ANN001
        if self._cursors:
            return self._cursors.pop(0)
        return _FakeCursor()


def _make_pool(connection):
    class _FakePool:
        def connection(self):
            class _Ctx:
                def __enter__(self_inner):
                    return connection

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

    return _FakePool()


def test_row_truthy_value_supports_mapping_and_sequence_rows() -> None:
    tool = RegulationQueryTool(_FakeConnection([]))

    assert tool._row_truthy_value({"present": True}) is True
    assert tool._row_truthy_value({"exists": False}) is False
    assert tool._row_truthy_value((True,)) is True
    assert tool._row_truthy_value(None) is False


def test_get_regulation_by_category_accepts_dict_style_exists_row() -> None:
    exists_cursor = _FakeCursor(fetchone_results=[{"present": True}])
    search_cursor = _FakeCursor(
        fetchall_results=[
            [
                {
                    "id": 1,
                    "title": "FA 보상선수",
                    "content": "관련 규정",
                    "source_table": "kbo_regulations",
                    "document_type": "rule",
                    "category": "player",
                    "regulation_code": "01-1",
                }
            ]
        ]
    )
    tool = RegulationQueryTool(_FakeConnection([exists_cursor, search_cursor]))

    result = tool.get_regulation_by_category("player", limit=3)

    assert set(result.keys()) == {
        "category",
        "regulations",
        "found",
        "total_found",
        "error",
    }
    assert result["found"] is True
    assert result["error"] is None
    assert result["total_found"] == 1
    assert result["regulations"][0]["title"] == "FA 보상선수"


def test_get_regulation_by_category_returns_error_when_rag_chunks_missing(
    caplog,
) -> None:
    exists_cursor = _FakeCursor(fetchone_results=[{"present": False}])
    tool = RegulationQueryTool(_FakeConnection([exists_cursor]))
    caplog.set_level(logging.WARNING)

    result = tool.get_regulation_by_category("player", limit=3)

    assert (
        f"[{REGULATION_QUERY_COMPONENT}] event=dependency_missing action={ACTION_GET_REGULATION_BY_CATEGORY} dependency=rag_chunks"
        in caplog.text
    )
    assert result["found"] is False
    assert result["error"] == "검색 인덱스(rag_chunks)가 준비되지 않았습니다."


def test_get_regulation_by_category_retries_with_fresh_connection(monkeypatch) -> None:
    primary_conn = SimpleNamespace(closed=False, label="primary")
    retry_conn = SimpleNamespace(closed=False, label="retry")
    tool = RegulationQueryTool(primary_conn)
    monkeypatch.setattr("app.deps.get_connection_pool", lambda: _make_pool(retry_conn))

    calls = []

    def _fake_lookup(conn, category, limit):
        calls.append((conn, category, limit))
        if conn is primary_conn:
            raise RuntimeError("connection is closed")
        return [
            {
                "id": 1,
                "title": "FA 보상선수",
                "content": "관련 규정",
                "source_table": "kbo_regulations",
                "document_type": "rule",
                "category": "player",
                "regulation_code": "01-1",
            }
        ]

    monkeypatch.setattr(tool, "_get_regulation_by_category_once", _fake_lookup)

    result = tool.get_regulation_by_category("player", limit=2)

    assert calls == [
        (primary_conn, "player", 2),
        (retry_conn, "player", 2),
    ]
    assert result["found"] is True
    assert result["regulations"][0]["title"] == "FA 보상선수"


def test_validate_regulation_reference_retries_with_fresh_connection(
    monkeypatch, caplog
) -> None:
    primary_conn = SimpleNamespace(closed=False, label="primary")
    retry_conn = SimpleNamespace(closed=False, label="retry")
    tool = RegulationQueryTool(primary_conn)
    monkeypatch.setattr("app.deps.get_connection_pool", lambda: _make_pool(retry_conn))
    caplog.set_level(logging.INFO)

    calls = []

    def _fake_validate(conn, regulation_code):
        calls.append((conn, regulation_code))
        if conn is primary_conn:
            raise RuntimeError("connection is closed")
        return {
            "id": 7,
            "title": "FA 보상선수",
            "content": "관련 규정",
            "regulation_code": regulation_code,
            "category": "player",
        }

    monkeypatch.setattr(tool, "_validate_regulation_reference_once", _fake_validate)

    result = tool.validate_regulation_reference("01-1")

    assert (
        f"[{REGULATION_QUERY_COMPONENT}] event=query_start action={ACTION_VALIDATE_REGULATION_REFERENCE} value=01-1"
        in caplog.text
    )
    assert (
        f"[{REGULATION_QUERY_COMPONENT}] event=query_retry action={ACTION_VALIDATE_REGULATION_REFERENCE} reason=connection is closed"
        in caplog.text
    )
    assert (
        f"[{REGULATION_QUERY_COMPONENT}] event=query_success action={ACTION_VALIDATE_REGULATION_REFERENCE} count=1 detail=regulation_code=01-1"
        in caplog.text
    )
    assert calls == [
        (primary_conn, "01-1"),
        (retry_conn, "01-1"),
    ]
    assert set(result.keys()) == {
        "regulation_code",
        "error",
        "exists",
        "regulation",
    }
    assert result["error"] is None
    assert result["exists"] is True
    assert result["regulation"]["regulation_code"] == "01-1"
