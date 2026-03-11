from __future__ import annotations

import sys
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
