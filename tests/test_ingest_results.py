from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.ingest_runs import IngestLeaseLostError, IngestTableResult
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


class _LeaseCursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params=None):
        self.queries.append((query, params))

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None


class _LeaseConnection:
    def __init__(self, rows):
        self.cursor_instance = _LeaseCursor(rows)

    def cursor(self):
        return self.cursor_instance


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


def test_ingest_table_result_checkpoint_fields_default_for_legacy_callers():
    result = IngestTableResult("game", 3, 4, 1, 2, None)

    assert result.checkpoint_resumed is False
    assert result.checkpoint_committed_batches == 0
    assert result.checkpoint_completed is False
    assert result.attempt_source_rows is None
    assert result.attempt_written_chunks is None


def test_ingest_returns_per_table_counts(monkeypatch):
    source = _Connection()
    destination = _Connection()
    connections = iter((source, destination))
    captured = {}
    monkeypatch.setattr(module, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://internal-destination"),
    )
    monkeypatch.setattr(module.psycopg, "connect", lambda *args, **kwargs: next(connections))
    def _ingest_table(*args, **kwargs):
        del args
        captured.update(kwargs)
        return IngestTableResult("game", 3, 4, 0, 0, None)

    monkeypatch.setattr(module, "ingest_table", _ingest_table)

    result = module.ingest(
        tables=["game"],
        source_db_url="postgresql://internal-source",
        **OPTIONS,
    )

    assert result.tables["game"].written_chunks == 3
    assert result.total_written_chunks == 3
    assert source.closed is True
    assert destination.closed is True
    assert "checkpoint_run_id" not in captured
    assert "checkpoint_scope_key" not in captured


def test_leased_database_ingest_passes_checkpoint_identity(monkeypatch):
    source = _Connection()
    destination = _Connection()
    connections = iter((source, destination))
    captured = {}
    lease_guard = lambda _write=False: None
    monkeypatch.setattr(module, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://internal-destination"),
    )
    monkeypatch.setattr(module.psycopg, "connect", lambda *args, **kwargs: next(connections))
    monkeypatch.setattr(module, "build_ingest_lease_guard", lambda *_args: lease_guard)

    def _ingest_table(*args, **kwargs):
        del args
        captured.update(kwargs)
        return IngestTableResult("game", 3, 4, 0, 0, None)

    monkeypatch.setattr(module, "ingest_table", _ingest_table)

    module.ingest(
        tables=["game"],
        lease_run_id="run-1",
        lease_owner="worker-1",
        checkpoint_scope_key="  season:2026  ",
        source_db_url="postgresql://internal-source",
        **OPTIONS,
    )

    assert captured["checkpoint_run_id"] == "run-1"
    assert captured["checkpoint_scope_key"] == "season:2026"
    assert captured["lease_guard"] is lease_guard


def test_leased_static_ingest_preserves_non_checkpoint_path(monkeypatch):
    source = _Connection()
    destination = _Connection()
    connections = iter((source, destination))
    captured = {}
    monkeypatch.setattr(module, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://internal-destination"),
    )
    monkeypatch.setattr(module.psycopg, "connect", lambda *args, **kwargs: next(connections))
    monkeypatch.setattr(
        module,
        "build_ingest_lease_guard",
        lambda *_args: (lambda _write=False: None),
    )

    def _ingest_table(*args, **kwargs):
        del args
        captured.update(kwargs)
        return IngestTableResult("kbo_metrics_explained", 1, 1, 0, 0, None)

    monkeypatch.setattr(module, "ingest_table", _ingest_table)

    module.ingest(
        tables=["kbo_metrics_explained"],
        lease_run_id="run-1",
        lease_owner="worker-1",
        checkpoint_scope_key="season:2026",
        source_db_url="postgresql://internal-source",
        **OPTIONS,
    )

    assert "checkpoint_run_id" not in captured
    assert "checkpoint_scope_key" not in captured


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


def test_sync_lease_guard_locks_row_before_checking_actual_database_time():
    connection = _LeaseConnection(rows=[(1,), None])
    guard = module.build_ingest_lease_guard(connection, "run-1", "worker-1")

    with pytest.raises(IngestLeaseLostError):
        guard(True)

    lock_query, lock_params = connection.cursor_instance.queries[0]
    check_query, check_params = connection.cursor_instance.queries[1]
    assert "FOR SHARE" in lock_query
    assert "status = 'RUNNING'" not in lock_query
    assert lock_params == ("run-1",)
    assert "lease_expires_at > clock_timestamp()" in check_query
    assert "status = 'RUNNING'" in check_query
    assert "lease_owner = %s" in check_query
    assert "FOR SHARE" not in check_query
    assert check_params == ("run-1", "worker-1")
