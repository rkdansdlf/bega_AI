from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

from scripts import cleanup_embedding_256_warnings as cleanup


class _FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.rowcount = 0

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((query, params))
        if "UPDATE rag_chunks" in query and "embedding_model = %s" in query:
            self.rowcount = 413
        elif "UPDATE rag_chunks" in query and "is_active = false" in query:
            self.rowcount = 2


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self, *args, **kwargs) -> _FakeCursor:
        del args, kwargs
        return self._cursor


class _RunCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.rowcount = 0
        self._rows: list[dict[str, Any]] = []
        self._row: tuple[Any, ...] | None = None

    def __enter__(self) -> "_RunCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((query, params))
        self._rows = []
        self._row = None
        if "format_type" in query:
            self._rows = [{"embedding_type": "extensions.vector(256)"}]
        elif "count(*)" in query and "embedding_model <> %s" in query:
            self._row = (0,)
        elif "count(*)" in query and "embedding_dim <> %s" in query:
            self._row = (7,)
        elif "count(*)" in query and "embedding IS NULL" in query:
            self._row = (2,)
        elif "UPDATE rag_chunks" in query and "embedding_model = %s" in query:
            self.rowcount = 7
        elif "UPDATE rag_chunks" in query and "is_active = false" in query:
            self.rowcount = 2

    def fetchall(self) -> list[dict[str, Any]]:
        return self._rows

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row


class _RemainingAfterApplyCursor(_RunCursor):
    def __init__(self) -> None:
        super().__init__()
        self._after_apply = False

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((query, params))
        self._rows = []
        self._row = None
        if "format_type" in query:
            self._rows = [{"embedding_type": "extensions.vector(256)"}]
        elif "UPDATE rag_chunks" in query and "embedding_model = %s" in query:
            self.rowcount = 7
        elif "UPDATE rag_chunks" in query and "is_active = false" in query:
            self.rowcount = 2
            self._after_apply = True
        elif "count(*)" in query and "embedding_model <> %s" in query:
            self._row = (1 if self._after_apply else 0,)
        elif "count(*)" in query and "embedding_dim <> %s" in query:
            self._row = (1 if self._after_apply else 7,)
        elif "count(*)" in query and "embedding IS NULL" in query:
            self._row = (0 if self._after_apply else 2,)


class _RunConnection:
    def __init__(self, cursor: _RunCursor) -> None:
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def __enter__(self) -> "_RunConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def cursor(self, *args, **kwargs) -> _RunCursor:
        del args, kwargs
        return self._cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def _settings(**overrides: Any) -> SimpleNamespace:
    values = {
        "embed_provider": "openrouter",
        "openrouter_embed_model": "openai/text-embedding-3-small",
        "embed_model": "",
        "embed_dim": 256,
        "rag_embedding_version": 2,
        "ai_vector_index": "hnsw",
        "ai_vector_quantization": "halfvec",
        "database_url": "postgresql://example/test",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_main_defaults_to_dry_run_without_apply(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "dry_run"}

    monkeypatch.setattr(cleanup, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["cleanup"])

    assert cleanup.main() == 0
    assert captured["apply"] is False
    assert captured["report_output"] == "reports/cleanup_embedding_256_warnings.json"
    assert captured["expect_metadata_fixable"] is None
    assert captured["expect_active_null_embeddings"] is None
    assert captured["expect_metadata_conflicts"] is None


def test_main_passes_apply_count_expectations(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "applied"}

    monkeypatch.setattr(cleanup, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cleanup",
            "--apply",
            "--expect-metadata-fixable",
            "413",
            "--expect-active-null-embeddings",
            "2",
            "--expect-metadata-conflicts",
            "0",
        ],
    )

    assert cleanup.main() == 0
    assert captured["apply"] is True
    assert captured["expect_metadata_fixable"] == 413
    assert captured["expect_active_null_embeddings"] == 2
    assert captured["expect_metadata_conflicts"] == 0


def test_main_exits_nonzero_for_applied_with_remaining(monkeypatch) -> None:
    def fake_run(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"status": "applied_with_remaining"}

    monkeypatch.setattr(cleanup, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["cleanup", "--apply"])

    assert cleanup.main() == 1


def test_runtime_snapshot_includes_post_rollout_vector_contract() -> None:
    snapshot = cleanup._runtime_snapshot(_settings())

    assert snapshot["ai_vector_index"] == "hnsw"
    assert snapshot["ai_vector_quantization"] == "halfvec"


def test_validate_runtime_accepts_expected_openrouter_256_v2_hnsw_halfvec() -> None:
    assert cleanup._validate_runtime(_settings()) == []


def test_validate_runtime_rejects_unexpected_embedding_model() -> None:
    errors = cleanup._validate_runtime(_settings(openrouter_embed_model="other/model"))

    assert errors
    assert "resolved embedding model is not" in errors[0]


def test_validate_runtime_rejects_non_hnsw_halfvec_runtime() -> None:
    errors = cleanup._validate_runtime(
        _settings(ai_vector_index="auto", ai_vector_quantization="none")
    )

    assert any("AI_VECTOR_INDEX" in error for error in errors)
    assert any("AI_VECTOR_QUANTIZATION" in error for error in errors)


def test_apply_cleanup_updates_metadata_and_deactivates_nulls_only() -> None:
    cursor = _FakeCursor()
    result = cleanup._apply_cleanup(_FakeConnection(cursor))

    metadata_sql, metadata_params = cursor.executed[0]
    null_sql, null_params = cursor.executed[1]

    assert result == {"metadata_updated": 413, "active_null_deactivated": 2}
    assert "SET embedding_model = %s" in metadata_sql
    assert "embedding_dim = %s" in metadata_sql
    assert "embedding_version = %s" in metadata_sql
    assert "SET embedding =" not in metadata_sql
    assert metadata_params == (
        cleanup.EXPECTED_EMBEDDING_MODEL,
        cleanup.EXPECTED_EMBEDDING_DIM,
        cleanup.EXPECTED_EMBEDDING_VERSION,
        cleanup.EXPECTED_EMBEDDING_VERSION,
        cleanup.EXPECTED_EMBEDDING_MODEL,
        cleanup.EXPECTED_EMBEDDING_DIM,
    )
    assert "SET is_active = false" in null_sql
    assert "valid_to = COALESCE(valid_to, now())" in null_sql
    assert "DELETE FROM rag_chunks" not in null_sql
    assert null_params == ()


def test_metadata_fix_where_covers_safe_stale_metadata_only() -> None:
    assert "embedding_version = %s" in cleanup.METADATA_FIX_WHERE
    assert "embedding_model = %s" in cleanup.METADATA_FIX_WHERE
    assert "embedding_dim <> %s" in cleanup.METADATA_FIX_WHERE
    assert "btrim(embedding_model) = ''" in cleanup.METADATA_FIX_WHERE

    assert "embedding_model <> %s" in cleanup.METADATA_CONFLICT_WHERE
    assert "embedding_dim <> %s" not in cleanup.METADATA_CONFLICT_WHERE


def test_db_precondition_requires_vector_256_column() -> None:
    assert cleanup._validate_db_preconditions("extensions.vector(256)") == []
    assert cleanup._validate_db_preconditions("extensions.vector(1536)")


def test_configure_session_can_mark_dry_run_read_only() -> None:
    cursor = _FakeCursor()

    cleanup._configure_session(
        _FakeConnection(cursor),
        statement_timeout_ms=123,
        lock_timeout_ms=45,
        read_only=True,
    )

    statements = [query for query, _params in cursor.executed]
    assert statements[0] == "SET TRANSACTION READ ONLY"
    assert statements[1].startswith("SET search_path TO")


def test_count_expectation_mismatches_reports_actual_counts() -> None:
    mismatches = cleanup._count_expectation_mismatches(
        {
            "metadata_fixable": 7,
            "metadata_conflicts": 0,
            "active_null_embeddings": 2,
        },
        {
            "metadata_fixable": 413,
            "metadata_conflicts": 0,
            "active_null_embeddings": None,
        },
    )

    assert mismatches == [{"count": "metadata_fixable", "expected": 413, "actual": 7}]


def test_apply_count_mismatch_rolls_back_before_updates(monkeypatch, tmp_path) -> None:
    cursor = _RunCursor()
    connection = _RunConnection(cursor)
    monkeypatch.setattr(cleanup, "get_settings", lambda: _settings())
    monkeypatch.setattr(cleanup.psycopg, "connect", lambda *args, **kwargs: connection)

    report = cleanup.run(
        apply=True,
        report_output=str(tmp_path / "report.json"),
        sample_limit=0,
        statement_timeout_ms=120_000,
        lock_timeout_ms=5_000,
        expect_metadata_fixable=413,
        expect_active_null_embeddings=2,
        expect_metadata_conflicts=0,
    )

    assert report["status"] == "count_mismatch"
    assert report["count_mismatches"] == [
        {"count": "metadata_fixable", "expected": 413, "actual": 7}
    ]
    assert connection.rollbacks == 1
    assert connection.commits == 0
    assert not any("UPDATE rag_chunks" in query for query, _params in cursor.executed)


def test_apply_with_remaining_counts_reports_incomplete(monkeypatch, tmp_path) -> None:
    cursor = _RemainingAfterApplyCursor()
    connection = _RunConnection(cursor)
    monkeypatch.setattr(cleanup, "get_settings", lambda: _settings())
    monkeypatch.setattr(cleanup.psycopg, "connect", lambda *args, **kwargs: connection)

    report = cleanup.run(
        apply=True,
        report_output=str(tmp_path / "report.json"),
        sample_limit=0,
        statement_timeout_ms=120_000,
        lock_timeout_ms=5_000,
        expect_metadata_fixable=7,
        expect_active_null_embeddings=2,
        expect_metadata_conflicts=0,
    )

    assert report["status"] == "applied_with_remaining"
    assert report["after_counts"] == {
        "metadata_fixable": 1,
        "metadata_conflicts": 1,
        "active_null_embeddings": 0,
    }
    assert connection.commits == 1
