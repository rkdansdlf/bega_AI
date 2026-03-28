from __future__ import annotations

from typing import Any, List

from scripts import prune_rag_chunk_extras as prune_script
from scripts.verify_embedding_coverage import CoverageTarget


class _FakeCursor:
    def __init__(self) -> None:
        self.executed: List[tuple[str, tuple[Any, ...]]] = []
        self.connection: Any = None

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((query, params))


class _FakeConnection:
    def __init__(self, cursors: List[_FakeCursor]) -> None:
        self._cursors = list(cursors)
        self.commit_calls = 0

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def cursor(self, *args, **kwargs) -> _FakeCursor:
        del args, kwargs
        cursor = self._cursors.pop(0)
        cursor.connection = self
        return cursor

    def commit(self) -> None:
        self.commit_calls += 1


def test_build_candidate_ids_adds_canonical_alias() -> None:
    candidates = prune_script.build_candidate_ids(
        raw_source_row_id="game_id=20250301SSLG0",
        table="game",
        meta={"id": 54498},
        legacy_aliases={"game_id=20250301SSLG0": "id=54498"},
    )

    assert "id=54498" in candidates


def test_prune_extra_rag_chunks_dry_run_reports_without_delete(
    monkeypatch: Any,
) -> None:
    target = CoverageTarget(table="game", year=2021, source_table="game")
    source_conn = _FakeConnection([])
    dest_init_cur = _FakeCursor()
    dest_conn = _FakeConnection([dest_init_cur])
    connections = [source_conn, dest_conn]

    monkeypatch.setattr(prune_script, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        prune_script,
        "psycopg",
        type(
            "_FakePsycopg",
            (),
            {"connect": staticmethod(lambda *args, **kwargs: connections.pop(0))},
        )(),
    )
    monkeypatch.setattr(
        prune_script,
        "collect_extra_source_row_ids",
        lambda **kwargs: {
            "table": "game",
            "source_table": "game",
            "year": 2021,
            "expected_count": 10,
            "actual_count": 12,
            "extra_source_row_ids": ["id=1", "id=2"],
        },
    )

    report = prune_script.prune_extra_rag_chunks(
        source_db_url="postgresql://source",
        dest_db_url="postgresql://dest",
        targets=[target],
        fetch_batch_size=100,
        delete_batch_size=10,
        sample_limit=5,
        execute=False,
    )

    assert report["total_extra_count"] == 2
    assert report["total_deleted_count"] == 0
    assert report["rows"][0]["sample_extra_source_row_ids"] == ["id=1", "id=2"]
    assert dest_conn.commit_calls == 0


def test_prune_extra_rag_chunks_execute_deletes_in_batches(
    monkeypatch: Any,
) -> None:
    target = CoverageTarget(table="game", year=2021, source_table="game")
    source_conn = _FakeConnection([])
    dest_init_cur = _FakeCursor()
    dest_delete_cur = _FakeCursor()
    dest_conn = _FakeConnection([dest_init_cur, dest_delete_cur])
    connections = [source_conn, dest_conn]

    monkeypatch.setattr(prune_script, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        prune_script,
        "psycopg",
        type(
            "_FakePsycopg",
            (),
            {"connect": staticmethod(lambda *args, **kwargs: connections.pop(0))},
        )(),
    )
    monkeypatch.setattr(
        prune_script,
        "collect_extra_source_row_ids",
        lambda **kwargs: {
            "table": "game",
            "source_table": "game",
            "year": 2021,
            "expected_count": 10,
            "actual_count": 13,
            "extra_source_row_ids": ["id=1", "id=2", "id=3"],
        },
    )

    report = prune_script.prune_extra_rag_chunks(
        source_db_url="postgresql://source",
        dest_db_url="postgresql://dest",
        targets=[target],
        fetch_batch_size=100,
        delete_batch_size=2,
        sample_limit=2,
        execute=True,
    )

    assert report["total_deleted_count"] == 3
    assert dest_conn.commit_calls == 2
    delete_queries = [
        query
        for query, _params in dest_delete_cur.executed
        if "DELETE FROM rag_chunks" in query
    ]
    assert len(delete_queries) == 2
