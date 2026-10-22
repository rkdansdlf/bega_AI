from __future__ import annotations

import argparse
from typing import Any, Dict, List

import pytest

from scripts import sync_rag_chunks as sync_script


class _FakeCursor:
    def __init__(
        self,
        *,
        scalar_rows: List[tuple[Any, ...]] | None = None,
        fetch_rows: List[Dict[str, Any]] | None = None,
        connection: Any = None,
    ) -> None:
        self._scalar_rows = list(scalar_rows or [])
        self._fetch_rows = list(fetch_rows or [])
        self.connection = connection if connection is not None else self
        self.executed: List[tuple[str, tuple[Any, ...]]] = []
        self.executemany_calls: List[List[tuple[Any, ...]]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((query, params))

    def executemany(self, query: str, payload: List[tuple[Any, ...]]) -> None:
        self.executed.append((query, ()))
        self.executemany_calls.append(list(payload))

    def fetchone(self) -> tuple[Any, ...] | None:
        if not self._scalar_rows:
            return None
        return self._scalar_rows.pop(0)

    def fetchmany(self, size: int) -> List[Dict[str, Any]]:
        if not self._fetch_rows:
            return []
        rows = self._fetch_rows[:size]
        self._fetch_rows = self._fetch_rows[size:]
        return rows


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


class _RaisingCursor(_FakeCursor):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((query, params))
        raise self._exc


def test_resolve_db_urls_rejects_same_db_by_default(monkeypatch: Any) -> None:
    args = argparse.Namespace(
        source_db_url="",
        dest_db_url="",
        source_env_file=".env.source",
        dest_env_file=".env.dest",
        allow_same_db=False,
    )

    class _Settings:
        def __init__(self, database_url: str) -> None:
            self.database_url = database_url

    monkeypatch.setattr(
        sync_script,
        "_load_settings_from_env_file",
        lambda _path: _Settings("postgresql://same"),
    )

    with pytest.raises(RuntimeError, match="identical"):
        sync_script.resolve_db_urls(args)


def test_build_upsert_rows_serializes_meta_and_embedding_text() -> None:
    payload = sync_script._build_upsert_rows(
        [
            {
                "meta": {"k": "v"},
                "season_year": 2025,
                "season_id": 1,
                "league_type_code": 0,
                "team_id": "LG",
                "player_id": "123",
                "source_table": "game",
                "source_row_id": "id=1",
                "title": "title",
                "content": "content",
                "embedding_text": "[0.1,0.2]",
            }
        ]
    )

    assert payload == [
        (
            '{"k": "v"}',
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            2025,
            1,
            0,
            "LG",
            "123",
            "game",
            "id=1",
            "title",
            "content",
            "[0.1,0.2]",
        )
    ]


def test_execute_scalar_supports_dict_row() -> None:
    class _ScalarCursor:
        def execute(self, query: str, params: tuple[Any, ...]) -> None:
            del query, params

        def fetchone(self) -> Dict[str, Any]:
            return {"count": 7}

    assert sync_script._execute_scalar(_ScalarCursor(), "SELECT count(*)", ()) == 7


def test_sync_rag_chunks_upserts_batches_and_commits(monkeypatch: Any) -> None:
    source_init_cur = _FakeCursor()
    source_count_cur = _FakeCursor(scalar_rows=[(3,)])
    source_read_cur = _FakeCursor(
        fetch_rows=[
            {
                "meta": {"idx": 1},
                "season_year": 2025,
                "season_id": 1,
                "league_type_code": 0,
                "team_id": "LG",
                "player_id": None,
                "source_table": "game",
                "source_row_id": "id=1",
                "title": "t1",
                "content": "c1",
                "embedding_text": "[0.1,0.2]",
            },
            {
                "meta": {"idx": 2},
                "season_year": 2025,
                "season_id": 1,
                "league_type_code": 0,
                "team_id": "LG",
                "player_id": None,
                "source_table": "game",
                "source_row_id": "id=2",
                "title": "t2",
                "content": "c2",
                "embedding_text": "[0.3,0.4]",
            },
            {
                "meta": {"idx": 3},
                "season_year": 2025,
                "season_id": 1,
                "league_type_code": 0,
                "team_id": "LG",
                "player_id": None,
                "source_table": "game",
                "source_row_id": "id=3",
                "title": "t3",
                "content": "c3",
                "embedding_text": "[0.5,0.6]",
            },
        ]
    )
    dest_init_cur = _FakeCursor()
    dest_write_cur = _FakeCursor()
    source_conn = _FakeConnection([source_count_cur, source_init_cur, source_read_cur])
    dest_conn = _FakeConnection([dest_init_cur, dest_write_cur])
    connections = [source_conn, dest_conn]

    monkeypatch.setattr(sync_script, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        sync_script,
        "psycopg",
        type(
            "_FakePsycopg",
            (),
            {
                "connect": staticmethod(lambda *args, **kwargs: connections.pop(0)),
            },
        )(),
    )

    result = sync_script.sync_rag_chunks(
        source_db_url="postgresql://source",
        dest_db_url="postgresql://dest",
        source_tables=[],
        season_year=None,
        batch_size=2,
        commit_interval=2,
        limit=None,
        known_total_rows=None,
        dry_run=False,
        truncate_dest=False,
    )

    assert result["synced_rows"] == 3
    assert dest_conn.commit_calls == 2
    assert len(dest_write_cur.executemany_calls) == 2
    assert dest_write_cur.executemany_calls[0][0][22] == "id=1"


def test_sync_rag_chunks_dry_run_only_counts(monkeypatch: Any) -> None:
    source_init_cur = _FakeCursor()
    source_count_cur = _FakeCursor(scalar_rows=[(5,)])
    dest_init_cur = _FakeCursor()
    source_conn = _FakeConnection([source_count_cur, source_init_cur])
    dest_conn = _FakeConnection([dest_init_cur])
    connections = [source_conn, dest_conn]

    monkeypatch.setattr(sync_script, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        sync_script,
        "psycopg",
        type(
            "_FakePsycopg",
            (),
            {
                "connect": staticmethod(lambda *args, **kwargs: connections.pop(0)),
            },
        )(),
    )

    result = sync_script.sync_rag_chunks(
        source_db_url="postgresql://source",
        dest_db_url="postgresql://dest",
        source_tables=["markdown_docs"],
        season_year=None,
        batch_size=100,
        commit_interval=100,
        limit=3,
        known_total_rows=None,
        dry_run=True,
        truncate_dest=False,
    )

    assert result == {
        "total_rows": 3,
        "synced_rows": 0,
        "committed_rows": 0,
        "dry_run": True,
    }


def test_sync_rag_chunks_can_truncate_destination(monkeypatch: Any) -> None:
    source_count_cur = _FakeCursor(scalar_rows=[(1,)])
    source_init_cur = _FakeCursor()
    source_read_cur = _FakeCursor(
        fetch_rows=[
            {
                "meta": {"idx": 1},
                "season_year": 2025,
                "season_id": 1,
                "league_type_code": 0,
                "team_id": "LG",
                "player_id": None,
                "source_table": "game",
                "source_row_id": "id=1",
                "title": "t1",
                "content": "c1",
                "embedding_text": "[0.1,0.2]",
            }
        ]
    )
    dest_init_cur = _FakeCursor()
    dest_write_cur = _FakeCursor()
    source_conn = _FakeConnection([source_count_cur, source_init_cur, source_read_cur])
    dest_conn = _FakeConnection([dest_init_cur, dest_write_cur])
    connections = [source_conn, dest_conn]

    monkeypatch.setattr(sync_script, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        sync_script,
        "psycopg",
        type(
            "_FakePsycopg",
            (),
            {
                "connect": staticmethod(lambda *args, **kwargs: connections.pop(0)),
            },
        )(),
    )

    sync_script.sync_rag_chunks(
        source_db_url="postgresql://source",
        dest_db_url="postgresql://dest",
        source_tables=[],
        season_year=None,
        batch_size=10,
        commit_interval=10,
        limit=None,
        known_total_rows=None,
        dry_run=False,
        truncate_dest=True,
    )

    assert any(
        "TRUNCATE TABLE rag_chunks" in query
        for query, _params in dest_init_cur.executed
    )


def test_build_select_query_adds_season_year_filter() -> None:
    query, params = sync_script.build_select_query(
        source_tables=["game"],
        season_year=2022,
        limit=10,
    )

    assert "source_table = ANY(%s)" in query
    assert "season_year = %s" in query
    assert params == (["game"], 2022, 10)


def test_count_source_rows_applies_season_year_filter() -> None:
    cur = _FakeCursor(scalar_rows=[(8,)])

    count = sync_script._count_source_rows(
        cur,
        source_tables=["game"],
        season_year=2022,
        limit=None,
    )

    assert count == 8
    assert cur.executed[0] == (
        "SELECT count(*) FROM rag_chunks WHERE source_table = ANY(%s) AND season_year = %s",
        (["game"], 2022),
    )


def test_sync_rag_chunks_uses_known_total_rows_when_count_fails(
    monkeypatch: Any,
) -> None:
    operational_error = RuntimeError("connection dropped during count")
    source_count_cur = _RaisingCursor(operational_error)
    source_init_cur = _FakeCursor()
    source_read_cur = _FakeCursor(
        fetch_rows=[
            {
                "meta": None,
                "season_year": 2020,
                "season_id": 1,
                "league_type_code": 0,
                "team_id": "LG",
                "player_id": None,
                "source_table": "game",
                "source_row_id": "id=1",
                "title": "t1",
                "content": "c1",
                "embedding_text": "[0.1,0.2]",
            }
        ]
    )
    dest_init_cur = _FakeCursor()
    dest_write_cur = _FakeCursor()
    source_conn = _FakeConnection([source_count_cur, source_init_cur, source_read_cur])
    dest_conn = _FakeConnection([dest_init_cur, dest_write_cur])
    connections = [source_conn, dest_conn]

    monkeypatch.setattr(sync_script, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        sync_script,
        "psycopg",
        type(
            "_FakePsycopg",
            (),
            {
                "connect": staticmethod(lambda *args, **kwargs: connections.pop(0)),
                "OperationalError": RuntimeError,
            },
        )(),
    )

    result = sync_script.sync_rag_chunks(
        source_db_url="postgresql://source",
        dest_db_url="postgresql://dest",
        source_tables=["game"],
        season_year=2020,
        batch_size=10,
        commit_interval=10,
        limit=None,
        known_total_rows=38788,
        dry_run=False,
        truncate_dest=False,
    )

    assert result["total_rows"] == 38788
    assert result["synced_rows"] == 1


# ---------------------------------------------------------------------------
# index_status propagation (KBO_playwright tombstones: index_status=DELETED)
# ---------------------------------------------------------------------------


def _tombstone_row(**overrides: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "meta": {"idx": 1},
        "season_year": 2026,
        "season_id": 1,
        "league_type_code": 0,
        "team_id": "SS",
        "player_id": "79402",
        "source_table": "milestone",
        "source_row_id": "2026:79402:300세이브",
        "title": "오승환 300세이브",
        "content": "c1",
        "embedding_text": "[0.1,0.2]",
        "is_active": False,
        "index_status": "DELETED",
    }
    row.update(overrides)
    return row


def _run_sync(
    monkeypatch: Any,
    *,
    rows: List[Dict[str, Any]],
    source_has_index_status: bool,
    dest_has_index_status: bool,
    truncate_dest: bool = False,
) -> tuple[Dict[str, Any], _FakeCursor, _FakeCursor, _FakeCursor]:
    source_count_cur = _FakeCursor(scalar_rows=[(len(rows),)])
    source_init_cur = _FakeCursor(scalar_rows=[(source_has_index_status,)])
    source_read_cur = _FakeCursor(fetch_rows=rows)
    dest_init_cur = _FakeCursor(scalar_rows=[(dest_has_index_status,)])
    dest_write_cur = _FakeCursor()
    source_conn = _FakeConnection([source_count_cur, source_init_cur, source_read_cur])
    dest_conn = _FakeConnection([dest_init_cur, dest_write_cur])
    connections = [source_conn, dest_conn]

    monkeypatch.setattr(sync_script, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        sync_script,
        "psycopg",
        type(
            "_FakePsycopg",
            (),
            {"connect": staticmethod(lambda *args, **kwargs: connections.pop(0))},
        )(),
    )

    result = sync_script.sync_rag_chunks(
        source_db_url="postgresql://source",
        dest_db_url="postgresql://dest",
        source_tables=[],
        season_year=None,
        batch_size=10,
        commit_interval=10,
        limit=None,
        known_total_rows=None,
        dry_run=False,
        truncate_dest=truncate_dest,
    )
    return result, source_read_cur, dest_write_cur, dest_init_cur


def test_sync_propagates_index_status_and_explicit_is_active_false(
    monkeypatch: Any,
) -> None:
    rows = [
        _tombstone_row(),
        _tombstone_row(
            source_row_id="2026:1:active", is_active=True, index_status="ACTIVE"
        ),
        _tombstone_row(source_row_id="2026:2:null", is_active=None, index_status=None),
    ]

    result, source_read_cur, dest_write_cur, _ = _run_sync(
        monkeypatch,
        rows=rows,
        source_has_index_status=True,
        dest_has_index_status=True,
    )

    assert result["synced_rows"] == 3
    select_sql, _ = source_read_cur.executed[0]
    assert "index_status" in select_sql

    upsert_sql, _ = dest_write_cur.executed[0]
    assert "index_status = EXCLUDED.index_status" in upsert_sql
    assert "COALESCE(%s, 'ACTIVE')" in upsert_sql
    # Explicit false is passed through; only NULL is defaulted to true.
    assert "COALESCE(%s, true)" in upsert_sql
    assert "is_active = EXCLUDED.is_active" in upsert_sql

    payload = dest_write_cur.executemany_calls[0]
    # Base column positions are unchanged; index_status is appended last.
    assert [row[22] for row in payload] == [
        "2026:79402:300세이브",
        "2026:1:active",
        "2026:2:null",
    ]
    assert [row[12] for row in payload] == [False, True, None]
    assert [row[-1] for row in payload] == ["DELETED", "ACTIVE", None]
    assert len(payload[0]) == len(upsert_sql.split("%s")) - 1


def test_merge_stage_sql_never_resets_deleted_to_active() -> None:
    sql = sync_script.build_merge_stage_sql(include_index_status=True)

    assert "COALESCE(index_status, 'ACTIVE')" in sql
    assert "index_status = EXCLUDED.index_status" in sql
    assert "index_status = 'ACTIVE'" not in sql
    assert "COALESCE(is_active, true)" in sql
    assert sync_script.stage_columns(include_index_status=True)[-1] == "index_status"


def test_sync_without_source_index_status_leaves_destination_status(
    monkeypatch: Any,
) -> None:
    rows = [_tombstone_row()]
    rows[0].pop("index_status")

    result, source_read_cur, dest_write_cur, _ = _run_sync(
        monkeypatch,
        rows=rows,
        source_has_index_status=False,
        dest_has_index_status=True,
    )

    assert result["synced_rows"] == 1
    assert "index_status" not in source_read_cur.executed[0][0]
    upsert_sql, _ = dest_write_cur.executed[0]
    # The destination's own lifecycle column is not touched at all.
    assert "index_status" not in upsert_sql
    assert len(dest_write_cur.executemany_calls[0][0]) == 26
    assert dest_write_cur.executemany_calls[0][0][12] is False


def test_sync_refuses_when_destination_lacks_index_status(monkeypatch: Any) -> None:
    with pytest.raises(RuntimeError, match="010_rag_chunks_index_status.sql"):
        _run_sync(
            monkeypatch,
            rows=[_tombstone_row()],
            source_has_index_status=True,
            dest_has_index_status=False,
            truncate_dest=True,
        )


def test_sync_refusal_happens_before_truncating_destination(monkeypatch: Any) -> None:
    source_count_cur = _FakeCursor(scalar_rows=[(1,)])
    source_init_cur = _FakeCursor(scalar_rows=[(True,)])
    dest_init_cur = _FakeCursor(scalar_rows=[(False,)])
    connections = [
        _FakeConnection([source_count_cur, source_init_cur]),
        _FakeConnection([dest_init_cur]),
    ]
    monkeypatch.setattr(sync_script, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        sync_script,
        "psycopg",
        type(
            "_FakePsycopg",
            (),
            {"connect": staticmethod(lambda *args, **kwargs: connections.pop(0))},
        )(),
    )

    with pytest.raises(RuntimeError):
        sync_script.sync_rag_chunks(
            source_db_url="postgresql://source",
            dest_db_url="postgresql://dest",
            source_tables=[],
            season_year=None,
            batch_size=10,
            commit_interval=10,
            limit=None,
            known_total_rows=None,
            dry_run=False,
            truncate_dest=True,
        )

    assert not any("TRUNCATE" in query for query, _ in dest_init_cur.executed)


def test_sync_refuses_truncate_when_only_destination_has_index_status(
    monkeypatch: Any,
) -> None:
    source_count_cur = _FakeCursor(scalar_rows=[(1,)])
    source_init_cur = _FakeCursor(scalar_rows=[(False,)])
    dest_init_cur = _FakeCursor(scalar_rows=[(True,)])
    connections = [
        _FakeConnection([source_count_cur, source_init_cur]),
        _FakeConnection([dest_init_cur]),
    ]
    monkeypatch.setattr(sync_script, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        sync_script,
        "psycopg",
        type(
            "_FakePsycopg",
            (),
            {"connect": staticmethod(lambda *args, **kwargs: connections.pop(0))},
        )(),
    )

    with pytest.raises(RuntimeError, match="--truncate-dest refused"):
        sync_script.sync_rag_chunks(
            source_db_url="postgresql://source",
            dest_db_url="postgresql://dest",
            source_tables=[],
            season_year=None,
            batch_size=10,
            commit_interval=10,
            limit=None,
            known_total_rows=None,
            dry_run=False,
            truncate_dest=True,
        )

    assert not any("TRUNCATE" in query for query, _ in dest_init_cur.executed)


def test_index_status_messages_match_behaviour(capsys: Any) -> None:
    assert (
        sync_script.resolve_index_status_sync(
            source_has_column=False, dest_has_column=True
        )
        is False
    )
    err = capsys.readouterr().err
    assert "existing destination rows keep their index_status" in err
    assert "left unchanged" not in err

    sync_script.resolve_index_status_sync(
        source_has_column=False, dest_has_column=False
    )
    assert "tombstones are not synced" in capsys.readouterr().err

    assert (
        sync_script.resolve_index_status_sync(
            source_has_column=True, dest_has_column=True, truncate_dest=True
        )
        is True
    )
    assert capsys.readouterr().err == ""
