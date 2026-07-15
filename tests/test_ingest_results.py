from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.ingest_runs import IngestTableResult
from scripts import ingest_from_kbo as module


class _Cursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params=None):
        del query, params


class _UndefinedSourceCursor:
    def execute(self, query, params=None):
        del query, params
        error = RuntimeError("relation users does not exist")
        error.sqlstate = "42P01"
        raise error


class _Connection:
    def __init__(self):
        self.autocommit = False
        self.closed = False

    def cursor(self, *args, **kwargs):
        del args, kwargs
        return _Cursor()

    def close(self):
        self.closed = True


OPTIONS = {
    "limit": None,
    "embed_batch_size": 32,
    "read_batch_size": 500,
    "season_year": 2026,
    "use_legacy_renderer": False,
    "since": None,
    "skip_embedding": True,
    "max_concurrency": 1,
    "commit_interval": 500,
    "parallel_engine": "thread",
    "workers": 1,
}


def test_ingest_returns_per_table_counts(monkeypatch):
    source = _Connection()
    destination = _Connection()
    connections = iter((source, destination))
    monkeypatch.setattr(module, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://internal-destination"),
    )
    monkeypatch.setattr(module.psycopg, "connect", lambda *args, **kwargs: next(connections))
    monkeypatch.setattr(
        module,
        "ingest_table",
        lambda *args, **kwargs: IngestTableResult("game", 3, 4, 0, 0, None),
    )

    result = module.ingest(
        tables=["game"],
        source_db_url="postgresql://internal-source",
        **OPTIONS,
    )

    assert result.tables["game"].written_chunks == 3
    assert result.total_written_chunks == 3
    assert source.closed is True
    assert destination.closed is True


def test_missing_required_source_column_raises_manual_contract():
    with pytest.raises(module.ManualBaseballDataRequiredError) as raised:
        module.validate_required_source_columns(
            "game",
            {"game_id"},
            {"game_id", "game_date"},
        )

    assert raised.value.contract["code"] == "MANUAL_BASEBALL_DATA_REQUIRED"
    assert raised.value.contract["entity"] == "game"
    assert raised.value.contract["missing_fields"] == ["game_date"]
    assert raised.value.contract["import_source"] == "operator_manual_data"


def test_zero_rows_are_valid_when_required_columns_exist():
    module.validate_required_source_columns(
        "game",
        {"game_id", "game_date"},
        {"game_id", "game_date"},
    )


def test_undefined_source_schema_raises_manual_contract_before_generic_failure():
    with pytest.raises(module.ManualBaseballDataRequiredError) as raised:
        module.execute_source_select(
            _UndefinedSourceCursor(),
            "SELECT * FROM game",
            (),
            table_name="game",
            required_columns={"game_id", "game_date"},
        )

    assert raised.value.contract["code"] == "MANUAL_BASEBALL_DATA_REQUIRED"
    assert raised.value.contract["entity"] == "game"
    assert raised.value.contract["missing_fields"] == ["game_date", "game_id"]


def test_missing_trusted_static_source_raises_manual_contract(monkeypatch):
    monkeypatch.setattr(
        module,
        "build_static_profile_chunk_payloads",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("missing file")),
    )

    with pytest.raises(module.ManualBaseballDataRequiredError) as raised:
        module.load_static_profile_payloads(
            "kbo_metrics_explained",
            {"source_file": "docs/missing.md"},
            settings=SimpleNamespace(),
        )

    assert raised.value.contract["code"] == "MANUAL_BASEBALL_DATA_REQUIRED"
    assert raised.value.contract["missing_fields"] == ["source_file"]
