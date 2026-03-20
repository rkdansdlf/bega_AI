from __future__ import annotations

from typing import Any, Dict, List

from scripts import reembed_missing_rows as reembed_script
from scripts.ingest_from_kbo import TABLE_PROFILES, build_static_source_row_prefix
from scripts.verify_embedding_coverage import CoverageTarget


class _FakeCursor:
    def __init__(
        self, rows: List[Dict[str, Any]] | None = None, connection: Any = None
    ):
        self._rows = list(rows or [])
        self.connection = connection if connection is not None else self
        self.executed: List[tuple[Any, Any]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, query: Any, params: Any = None) -> None:
        self.executed.append((query, params))

    def fetchmany(self, size: int) -> List[Dict[str, Any]]:
        del size
        if not self._rows:
            return []
        rows = self._rows
        self._rows = []
        return rows

    def fetchall(self) -> List[Any]:
        return []


class _FakeDestConnection:
    def __init__(self) -> None:
        self.commit_calls = 0
        self.cursors: List[_FakeCursor] = []

    def cursor(self, *args, **kwargs) -> _FakeCursor:
        del args, kwargs
        cursor = _FakeCursor(connection=self)
        self.cursors.append(cursor)
        return cursor

    def commit(self) -> None:
        self.commit_calls += 1


class _FakeSourceConnection:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows

    def cursor(self, *args, **kwargs) -> _FakeCursor:
        del args, kwargs
        return _FakeCursor(rows=self._rows)


class _ForbiddenSourceConnection:
    def cursor(self, *args, **kwargs) -> _FakeCursor:
        del args, kwargs
        raise AssertionError(
            "source DB should not be queried for static source_file profiles"
        )


def test_build_static_target_payloads_are_stable_and_match_profile_metadata() -> None:
    target = CoverageTarget(
        table="markdown_docs_rules_terms",
        year=0,
        source_table="markdown_docs",
    )
    settings = reembed_script.get_settings()

    payloads_first = reembed_script.build_static_target_payloads(
        target,
        settings=settings,
    )
    payloads_second = reembed_script.build_static_target_payloads(
        target,
        settings=settings,
    )

    assert payloads_first
    assert {payload.source_row_id for payload in payloads_first} == {
        payload.source_row_id for payload in payloads_second
    }
    assert all(payload.table == "markdown_docs" for payload in payloads_first)
    assert all(
        payload.meta and payload.meta["source_profile"] == "markdown_docs_rules_terms"
        for payload in payloads_first
    )


def test_reembed_static_target_does_not_query_source_db(monkeypatch: Any) -> None:
    target = CoverageTarget(
        table="markdown_docs_rules_terms",
        year=0,
        source_table="markdown_docs",
    )
    dest_conn = _FakeDestConnection()
    captured: List[str] = []

    def _fake_flush_chunks(cur, settings, buffer, **kwargs):  # type: ignore[no-untyped-def]
        del cur, settings, kwargs
        captured.extend(item.source_row_id for item in buffer)
        flushed = len(buffer)
        buffer.clear()
        return flushed

    monkeypatch.setattr(reembed_script, "flush_chunks", _fake_flush_chunks)

    result = reembed_script.reembed_target_missing_rows(
        source_conn=_ForbiddenSourceConnection(),
        dest_conn=dest_conn,
        target=target,
        missing_ids={"markdown_docs:anything"},
        embed_batch_size=8,
        read_batch_size=100,
        max_concurrency=1,
        commit_interval=100,
    )

    prefix = build_static_source_row_prefix(
        "markdown_docs_rules_terms",
        TABLE_PROFILES["markdown_docs_rules_terms"],
    )
    assert result["matched_rows"] == 1
    assert result["flushed_chunks"] == len(captured)
    assert captured
    assert all(source_row_id.startswith(prefix) for source_row_id in captured)


def test_reembed_row_target_keeps_canonical_source_row_id(monkeypatch: Any) -> None:
    target = CoverageTarget(table="game", year=2025, source_table="game")
    source_conn = _FakeSourceConnection(
        rows=[
            {
                "id": 123,
                "game_id": "20250301SSLG0",
                "season_year": 2025,
                "league_type_code": 0,
            }
        ]
    )
    dest_conn = _FakeDestConnection()
    captured: List[str] = []

    def _fake_flush_chunks(cur, settings, buffer, **kwargs):  # type: ignore[no-untyped-def]
        del cur, settings, kwargs
        captured.extend(item.source_row_id for item in buffer)
        flushed = len(buffer)
        buffer.clear()
        return flushed

    monkeypatch.setattr(reembed_script, "flush_chunks", _fake_flush_chunks)
    monkeypatch.setattr(
        reembed_script, "get_primary_key_columns", lambda *_args: ["id"]
    )
    monkeypatch.setattr(
        reembed_script,
        "build_select_query",
        lambda *_args, **_kwargs: ("SELECT 1", ()),
    )
    monkeypatch.setattr(reembed_script, "build_title", lambda *_args, **_kwargs: "game")
    monkeypatch.setattr(
        reembed_script,
        "build_content",
        lambda *_args, **_kwargs: "짧은 경기 요약입니다.",
    )
    monkeypatch.setattr(
        "scripts.verify_embedding_coverage.build_expected_source_row_id",
        lambda **_kwargs: "id=123",
    )

    result = reembed_script.reembed_target_missing_rows(
        source_conn=source_conn,
        dest_conn=dest_conn,
        target=target,
        missing_ids={"id=123"},
        embed_batch_size=8,
        read_batch_size=100,
        max_concurrency=1,
        commit_interval=100,
    )

    assert result["matched_rows"] == 1
    assert captured == ["id=123"]
