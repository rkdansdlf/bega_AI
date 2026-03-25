from __future__ import annotations

from typing import Any, List

from scripts import dedupe_rag_chunk_aliases as dedupe_script
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


def test_analyze_alias_groups_marks_deletable_aliases() -> None:
    report = dedupe_script.analyze_alias_groups(
        rows=[
            ("id=54498", {"id": 54498}),
            ("game_id=20251031LGHH0", {"id": 54498}),
        ],
        table="game",
        expected_ids={"id=54498"},
        legacy_aliases={"game_id=20251031LGHH0": "id=54498"},
        sample_limit=5,
    )

    assert report["alias_group_count"] == 1
    assert report["deletable_alias_row_count"] == 1
    assert report["blocked_alias_row_count"] == 0
    assert report["deletable_alias_source_row_ids"] == ["game_id=20251031LGHH0"]


def test_analyze_alias_groups_blocks_when_canonical_raw_is_missing() -> None:
    report = dedupe_script.analyze_alias_groups(
        rows=[
            ("game_id=20251031LGHH0", {"id": 54498}),
        ],
        table="game",
        expected_ids={"id=54498"},
        legacy_aliases={"game_id=20251031LGHH0": "id=54498"},
        sample_limit=5,
    )

    assert report["alias_group_count"] == 1
    assert report["deletable_alias_row_count"] == 0
    assert report["blocked_alias_row_count"] == 1
    assert report["blocked_alias_source_row_ids"] == ["game_id=20251031LGHH0"]


def test_dedupe_rag_chunk_aliases_execute_deletes_only_deletable(
    monkeypatch: Any,
) -> None:
    target = CoverageTarget(table="game", year=2021, source_table="game")
    source_conn = _FakeConnection([])
    dest_init_cur = _FakeCursor()
    dest_delete_cur = _FakeCursor()
    dest_conn = _FakeConnection([dest_init_cur, dest_delete_cur])
    connections = [source_conn, dest_conn]

    monkeypatch.setattr(dedupe_script, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        dedupe_script,
        "psycopg",
        type(
            "_FakePsycopg",
            (),
            {"connect": staticmethod(lambda *args, **kwargs: connections.pop(0))},
        )(),
    )
    monkeypatch.setattr(
        dedupe_script,
        "collect_alias_candidates",
        lambda **kwargs: {
            "table": "game",
            "source_table": "game",
            "year": 2021,
            "expected_count": 10,
            "actual_row_count": 12,
            "alias_group_count": 1,
            "deletable_alias_row_count": 2,
            "blocked_alias_row_count": 1,
            "deletable_alias_source_row_ids": ["game_id=1", "game_id=2"],
            "blocked_alias_source_row_ids": ["game_id=3"],
            "sample_alias_groups": [],
        },
    )

    report = dedupe_script.dedupe_rag_chunk_aliases(
        source_db_url="postgresql://source",
        dest_db_url="postgresql://dest",
        targets=[target],
        fetch_batch_size=100,
        delete_batch_size=1,
        sample_limit=5,
        execute=True,
    )

    assert report["total_deleted_count"] == 2
    assert report["total_blocked_alias_row_count"] == 1
    assert dest_conn.commit_calls == 2
    delete_queries = [
        query for query, _params in dest_delete_cur.executed if "DELETE FROM rag_chunks" in query
    ]
    assert len(delete_queries) == 2
