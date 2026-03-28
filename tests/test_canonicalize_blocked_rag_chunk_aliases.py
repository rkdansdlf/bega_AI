from __future__ import annotations

from typing import Any, List

from scripts import canonicalize_blocked_rag_chunk_aliases as canonicalize_script
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


def test_choose_primary_alias_raw_id_prefers_canonical_composite() -> None:
    chosen = canonicalize_script.choose_primary_alias_raw_id(
        canonical_id="id=3156",
        alias_raw_ids=["id=57652", "id=3156|game_id=20221020WOKT0"],
    )

    assert chosen == "id=3156|game_id=20221020WOKT0"


def test_plan_blocked_group_action_keeps_one_primary_and_secondary() -> None:
    action = canonicalize_script.plan_blocked_group_action(
        canonical_id="id=352480",
        alias_raw_ids=["game_id=20221108WOSK0|inning=1", "id=766243"],
        sample_limit=5,
    )

    assert action["can_canonicalize"] is True
    assert action["primary_alias_raw_id"] == "game_id=20221108WOSK0|inning=1"
    assert action["secondary_alias_raw_ids"] == ["id=766243"]


def test_plan_blocked_group_action_skips_multipart_alias() -> None:
    action = canonicalize_script.plan_blocked_group_action(
        canonical_id="id=1",
        alias_raw_ids=["legacy#part1"],
        sample_limit=5,
    )

    assert action["can_canonicalize"] is False
    assert action["reason"] == "multipart_alias_row"


def test_plan_destination_write_promotes_alias_when_canonical_missing() -> None:
    plan = canonicalize_script.plan_destination_write(
        canonical_id="id=352480",
        primary_alias_raw_id="game_id=20221108WOSK0|inning=1",
        secondary_alias_raw_ids=["id=766243"],
        existing_canonical_ids=set(),
    )

    assert plan == {
        "mode": "promote_alias",
        "delete_alias_raw_ids": ["id=766243"],
    }


def test_plan_destination_write_merges_when_canonical_exists() -> None:
    plan = canonicalize_script.plan_destination_write(
        canonical_id="id=352480",
        primary_alias_raw_id="game_id=20221108WOSK0|inning=1",
        secondary_alias_raw_ids=["id=766243"],
        existing_canonical_ids={"id=352480"},
    )

    assert plan == {
        "mode": "merge_into_existing",
        "delete_alias_raw_ids": [
            "game_id=20221108WOSK0|inning=1",
            "id=766243",
        ],
    }


def test_canonicalize_blocked_aliases_reports_updates(monkeypatch: Any) -> None:
    target = CoverageTarget(
        table="game_inning_scores", year=2021, source_table="game_inning_scores"
    )
    source_conn = _FakeConnection([])
    dest_init_cur = _FakeCursor()
    dest_conn = _FakeConnection([dest_init_cur])
    connections = [source_conn, dest_conn]

    monkeypatch.setattr(canonicalize_script, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        canonicalize_script,
        "psycopg",
        type(
            "_FakePsycopg",
            (),
            {"connect": staticmethod(lambda *args, **kwargs: connections.pop(0))},
        )(),
    )
    monkeypatch.setattr(
        canonicalize_script,
        "canonicalize_target",
        lambda **kwargs: {
            "table": "game_inning_scores",
            "source_table": "game_inning_scores",
            "year": 2021,
            "blocked_group_count": 2,
            "eligible_group_count": 2,
            "canonicalized_count": 2,
            "merged_into_existing_count": 1,
            "secondary_alias_row_count": 1,
            "deleted_alias_row_count": 1,
            "skipped_group_count": 0,
            "sample_skipped_groups": [],
        },
    )

    report = canonicalize_script.canonicalize_blocked_aliases(
        source_db_url="postgresql://source",
        dest_db_url="postgresql://dest",
        targets=[target],
        fetch_batch_size=100,
        read_batch_size=100,
        commit_interval=10,
        execute=True,
        sample_limit=5,
    )

    assert report["total_blocked_group_count"] == 2
    assert report["total_eligible_group_count"] == 2
    assert report["total_canonicalized_count"] == 2
    assert report["total_merged_into_existing_count"] == 1
    assert report["total_secondary_alias_row_count"] == 1
    assert report["total_deleted_alias_row_count"] == 1
