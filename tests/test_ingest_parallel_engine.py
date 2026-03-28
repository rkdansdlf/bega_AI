from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Dict, List

from scripts import ingest_from_kbo as ingest_script


def test_parse_args_parallel_engine_defaults(monkeypatch: Any) -> None:
    monkeypatch.setattr(sys, "argv", ["ingest_from_kbo.py"])
    args = ingest_script.parse_args()
    assert args.parallel_engine == "thread"
    assert args.workers == 4
    assert args.embed_batch_size == 32
    assert "game_inning_scores" not in ingest_script.DEFAULT_TABLES
    assert "game_flow_summary" in ingest_script.DEFAULT_TABLES


def test_parse_args_parallel_engine_custom(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_from_kbo.py",
            "--parallel-engine",
            "subinterp",
            "--workers",
            "8",
        ],
    )
    args = ingest_script.parse_args()
    assert args.parallel_engine == "subinterp"
    assert args.workers == 8


def test_prepare_rows_for_engine_falls_back_to_thread(monkeypatch: Any) -> None:
    tasks = [
        (
            "teams",
            {"team_id": "HH", "team_name": "한화"},
            "team_id=HH",
            False,
            "2026-01-01",
        )
    ]

    def _raise(*_: Any, **__: Any) -> List[List[Dict[str, Any]]]:
        raise RuntimeError("subinterp unavailable")

    thread_result = [[{"table": "teams", "source_row_id": "team_id=HH"}]]

    monkeypatch.setattr(
        ingest_script,
        "_prepare_rows_with_subinterpreter_engine",
        _raise,
    )
    monkeypatch.setattr(
        ingest_script,
        "_prepare_rows_with_thread_engine",
        lambda _tasks, _workers: thread_result,
    )

    result, used_engine = ingest_script._prepare_rows_for_engine(
        tasks,
        parallel_engine="subinterp",
        workers=2,
    )

    assert used_engine == "thread"
    assert result == thread_result


def test_resolve_primary_key_columns_prefers_profile_override(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        ingest_script,
        "get_primary_key_columns",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("DB PK introspection should not run")
        ),
    )

    pk_columns = ingest_script.resolve_primary_key_columns(
        object(),
        "game_flow_summary",
        {"pk_columns_override": ["game_id"]},
    )

    assert pk_columns == ["game_id"]


def test_static_profile_source_row_ids_do_not_collide() -> None:
    rules_profile = ingest_script.TABLE_PROFILES["markdown_docs_rules_terms"]
    metrics_profile = ingest_script.TABLE_PROFILES["markdown_docs_strategy_metrics"]

    rules_prefix = ingest_script.build_static_source_row_prefix(
        "markdown_docs_rules_terms",
        rules_profile,
    )
    metrics_prefix = ingest_script.build_static_source_row_prefix(
        "markdown_docs_strategy_metrics",
        metrics_profile,
    )

    assert rules_prefix != metrics_prefix
    assert rules_prefix.startswith("markdown_docs:")
    assert metrics_prefix.startswith("markdown_docs:")
    assert (
        ingest_script.build_static_source_row_id(
            "markdown_docs_rules_terms",
            rules_profile,
            chunk_index=2,
            total_chunks=3,
        )
        == f"{rules_prefix}#part2"
    )


def test_ingest_sets_pgvector_search_path_on_destination_connection(
    monkeypatch: Any,
) -> None:
    class _FakeCursor:
        def __init__(self, executed: List[str]) -> None:
            self._executed = executed

        def execute(self, query: str) -> None:
            self._executed.append(query)

        def __enter__(self) -> "_FakeCursor":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class _FakeConnection:
        def __init__(self) -> None:
            self.autocommit = False
            self.executed: List[str] = []
            self.closed = False

        def cursor(self) -> _FakeCursor:
            return _FakeCursor(self.executed)

        def close(self) -> None:
            self.closed = True

    source_conn = _FakeConnection()
    dest_conn = _FakeConnection()
    connections = [source_conn, dest_conn]

    monkeypatch.setattr(
        ingest_script,
        "psycopg",
        SimpleNamespace(connect=lambda _dsn: connections.pop(0)),
    )
    monkeypatch.setattr(
        ingest_script,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://target"),
    )
    monkeypatch.setattr(
        ingest_script,
        "ingest_table",
        lambda *_args, **_kwargs: 0,
    )

    ingest_script.ingest(
        source_db_url="postgresql://source",
        tables=["teams"],
        limit=10,
        embed_batch_size=8,
        read_batch_size=50,
        season_year=2025,
        use_legacy_renderer=False,
        since=None,
        skip_embedding=True,
        max_concurrency=1,
        commit_interval=100,
        parallel_engine="thread",
        workers=2,
    )

    assert "SET statement_timeout TO 0;" in dest_conn.executed
    assert (
        f"SET search_path TO {ingest_script.PGVECTOR_SEARCH_PATH};"
        in dest_conn.executed
    )
    assert source_conn.closed is True
    assert dest_conn.closed is True
