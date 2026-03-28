from __future__ import annotations

import logging
from types import SimpleNamespace

import psycopg

from app.core.exceptions import DBRetrievalError
from app.tools.document_query import DocumentQueryTool
from app.tools.pooled_connection import connection_scope
from app.tools.query_logging import (
    ACTION_SEARCH_DOCUMENTS,
    ACTION_SEARCH_REGULATION,
    DOCUMENT_QUERY_COMPONENT,
    REGULATION_QUERY_COMPONENT,
)
from app.tools.regulation_query import RegulationQueryTool


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


def test_connection_scope_uses_pool_when_primary_connection_is_closed(monkeypatch):
    primary_conn = SimpleNamespace(closed=True)
    pooled_conn = SimpleNamespace(closed=False, label="pooled")
    monkeypatch.setattr("app.deps.get_connection_pool", lambda: _make_pool(pooled_conn))

    with connection_scope(primary_conn) as conn:
        assert conn is pooled_conn


def test_document_query_tool_retries_with_fresh_connection(monkeypatch, caplog):
    primary_conn = SimpleNamespace(closed=False, label="primary")
    retry_conn = SimpleNamespace(closed=False, label="retry")
    tool = DocumentQueryTool(primary_conn)
    monkeypatch.setattr("app.deps.get_connection_pool", lambda: _make_pool(retry_conn))
    caplog.set_level(logging.INFO)

    calls = []

    def _fake_search(conn, query, limit):
        calls.append((conn, query, limit))
        if conn is primary_conn:
            raise RuntimeError("connection is closed")
        return [{"title": "규정", "content": "설명", "source_table": "markdown_docs"}]

    monkeypatch.setattr(tool, "_search_documents_once", _fake_search)

    result = tool.search_documents("테스트", limit=3)

    assert (
        f"[{DOCUMENT_QUERY_COMPONENT}] event=query_start action={ACTION_SEARCH_DOCUMENTS} value=테스트"
        in caplog.text
    )
    assert (
        f"[{DOCUMENT_QUERY_COMPONENT}] event=query_retry action={ACTION_SEARCH_DOCUMENTS} reason=connection is closed"
        in caplog.text
    )
    assert (
        f"[{DOCUMENT_QUERY_COMPONENT}] event=query_success action={ACTION_SEARCH_DOCUMENTS} count=1"
        in caplog.text
    )
    assert calls == [
        (primary_conn, "테스트", 3),
        (retry_conn, "테스트", 3),
    ]
    assert set(result.keys()) == {
        "query",
        "documents",
        "found",
        "error",
        "source",
        "source_tables",
    }
    assert result["found"] is True
    assert result["error"] is None
    assert result["source"] == "verified_docs"
    assert result["source_tables"] == [
        "markdown_docs",
        "kbo_definitions",
        "kbo_regulations",
    ]
    assert result["documents"][0]["title"] == "규정"


def test_document_query_tool_reuses_query_embedding_cache(monkeypatch):
    tool = DocumentQueryTool(
        SimpleNamespace(closed=False),
        settings=SimpleNamespace(embed_provider="local", embed_model="stub"),
    )
    calls = {"count": 0}

    def _fake_embed_texts(texts, settings):
        calls["count"] += 1
        return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr("app.tools.document_query.embed_texts", _fake_embed_texts)

    first = tool._embed_query("  테스트 질의 ")
    second = tool._embed_query("테스트   질의")

    assert calls["count"] == 1
    assert first == second


def test_document_query_tool_uses_exact_term_fallback_on_similarity_timeout(
    monkeypatch,
):
    tool = DocumentQueryTool(
        SimpleNamespace(closed=False),
        settings=SimpleNamespace(embed_provider="local", embed_model="stub"),
    )
    similarity_calls = []

    monkeypatch.setattr(tool, "_embed_query", lambda query: [0.1, 0.2, 0.3])

    def _fake_similarity_search(conn, embedding, *, limit, keyword, settings):
        similarity_calls.append(keyword)
        raise DBRetrievalError(
            "pgvector query timed out",
            cause=psycopg.errors.QueryCanceled("statement timeout"),
        )

    monkeypatch.setattr(
        "app.tools.document_query.similarity_search", _fake_similarity_search
    )
    monkeypatch.setattr(
        tool,
        "_search_exact_term_documents",
        lambda conn, query_lower, limit: [
            {
                "title": "WHIP 설명",
                "content": "설명",
                "source_table": "markdown_docs",
                "source_row_id": "doc-1",
                "meta": {},
                "similarity": 1.0,
                "combined_score": 1.0,
            }
        ],
    )

    docs = tool._search_documents_once(SimpleNamespace(), "WHIP 뜻이 뭐야?", limit=1)

    assert similarity_calls == ["WHIP 뜻이 뭐야?", None]
    assert len(docs) == 1
    assert docs[0]["title"] == "WHIP 설명"


def test_regulation_query_tool_retries_with_fresh_connection(monkeypatch, caplog):
    primary_conn = SimpleNamespace(closed=False, label="primary")
    retry_conn = SimpleNamespace(closed=False, label="retry")
    tool = RegulationQueryTool(primary_conn)
    monkeypatch.setattr("app.deps.get_connection_pool", lambda: _make_pool(retry_conn))
    caplog.set_level(logging.INFO)

    calls = []

    def _fake_search(conn, query, limit):
        calls.append((conn, query, limit))
        if conn is primary_conn:
            raise RuntimeError("connection is closed")
        return (
            [
                {
                    "id": 1,
                    "title": "FA 보상선수",
                    "content": "규정 본문",
                    "category": "player",
                }
            ],
            "FA 보상선수",
        )

    monkeypatch.setattr(tool, "_search_regulation_once", _fake_search)

    result = tool.search_regulation("FA 보상선수", limit=2)

    assert (
        f"[{REGULATION_QUERY_COMPONENT}] event=query_start action={ACTION_SEARCH_REGULATION} value=FA 보상선수"
        in caplog.text
    )
    assert (
        f"[{REGULATION_QUERY_COMPONENT}] event=query_retry action={ACTION_SEARCH_REGULATION} reason=connection is closed"
        in caplog.text
    )
    assert (
        f"[{REGULATION_QUERY_COMPONENT}] event=query_success action={ACTION_SEARCH_REGULATION} count=1 detail=matched_query=FA 보상선수"
        in caplog.text
    )
    assert calls == [
        (primary_conn, "FA 보상선수", 2),
        (retry_conn, "FA 보상선수", 2),
    ]
    assert set(result.keys()) == {
        "query",
        "regulations",
        "found",
        "total_found",
        "categories",
        "error",
        "matched_query",
    }
    assert result["found"] is True
    assert result["error"] is None
    assert result["total_found"] == 1
    assert result["categories"] == ["player"]
    assert result["matched_query"] == "FA 보상선수"
    assert result["regulations"][0]["title"] == "FA 보상선수"
