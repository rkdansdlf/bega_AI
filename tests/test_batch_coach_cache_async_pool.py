from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import psycopg

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "batch_coach_cache.py"
SPEC = importlib.util.spec_from_file_location(
    "batch_coach_cache_under_test", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
batch_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch_script
SPEC.loader.exec_module(batch_script)


class FakeCursor:
    def __init__(self, rows: list[tuple] | None = None, rowcount: int = 0) -> None:
        self._rows = rows or []
        self.rowcount = rowcount

    async def fetchall(self) -> list[tuple]:
        return list(self._rows)

    async def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None


class FakeConnection:
    def __init__(
        self,
        *,
        select_key_batches: list[list[str]] | None = None,
        target_existing_rows: list[tuple[str, str, bool]] | None = None,
        count_results: list[int] | None = None,
        delete_returning_keys: list[str] | None = None,
    ) -> None:
        self._key_batches = list(select_key_batches or [])
        self._target_existing_rows = target_existing_rows
        self._counts = list(count_results or [])
        self._delete_returning_keys = delete_returning_keys or []
        self._key_idx = 0
        self._count_idx = 0
        self.executed_queries: list[tuple[str, tuple]] = []
        self.committed = False

    async def execute(self, sql: str, params: tuple = ()) -> FakeCursor:
        self.executed_queries.append((sql, params))
        clean_sql = " ".join(sql.strip().split()).upper()
        if clean_sql.startswith("SET TRANSACTION READ ONLY"):
            return FakeCursor()
        if clean_sql.startswith("SELECT STATUS, RESPONSE_JSON"):
            return FakeCursor(rows=[])
        if clean_sql.startswith("SELECT COUNT(*)"):
            count = (
                self._counts[self._count_idx]
                if self._count_idx < len(self._counts)
                else 0
            )
            self._count_idx += 1
            if clean_sql.startswith("SELECT COUNT(*) FILTER"):
                # (batch-owned count, all-versions count)
                return FakeCursor(rows=[(count, count)])
            return FakeCursor(rows=[(count,)])
        if clean_sql.startswith("SELECT CACHE_KEY, STATUS, RESPONSE_JSON"):
            if self._target_existing_rows is not None:
                return FakeCursor(rows=self._target_existing_rows)
            return FakeCursor(rows=[])
        if clean_sql.startswith("SELECT CACHE_KEY"):
            keys = (
                self._key_batches[self._key_idx]
                if self._key_idx < len(self._key_batches)
                else []
            )
            self._key_idx += 1
            return FakeCursor(rows=[(k,) for k in keys])
        if clean_sql.startswith("DELETE"):
            return FakeCursor(
                rows=[(k,) for k in self._delete_returning_keys],
                rowcount=len(self._delete_returning_keys),
            )
        if clean_sql.startswith("INSERT") or clean_sql.startswith("UPDATE"):
            return FakeCursor(rowcount=1)
        return FakeCursor()

    async def commit(self) -> None:
        self.committed = True

    @contextlib.asynccontextmanager
    async def transaction(self):
        try:
            yield
        except psycopg.Rollback:
            pass


class AsyncOnlyFakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection
        self.open = AsyncMock()
        self.close = AsyncMock()

    @contextlib.asynccontextmanager
    async def connection(self):
        yield self._connection


def test_async_only_pool_fails_sync_with() -> None:
    pool = AsyncOnlyFakePool(FakeConnection())
    with pytest.raises(TypeError, match="context manager protocol"):
        with pool.connection() as conn:  # noqa: F841
            pass


@pytest.mark.asyncio
async def test_cleanup_cache_rows_with_async_pool() -> None:
    fake_conn = FakeConnection(
        select_key_batches=[
            ["k_mismatch_1"],
            ["k_legacy_1", "k_legacy_2"],
            [],
        ],
        count_results=[5],
        delete_returning_keys=["k_legacy_1", "k_legacy_2", "k_mismatch_1"],
    )
    pool = AsyncOnlyFakePool(fake_conn)
    resolver = batch_script.TeamCodeResolver()

    options = batch_script.RunOptions(
        years=[2025],
        teams=["LG", "SSG"],
        focus=[],
        only_missing=False,
        force_rebuild=False,
        quality_report=None,
        baseline=False,
        game_type="REGULAR",
        delay_seconds=0.0,
        cleanup_legacy=True,
    )

    result = await batch_script.cleanup_cache_rows(pool, options, resolver)

    assert result["prompt_mismatch_rows"] == 1
    assert result["legacy_team_rows"] == 2
    assert result["stale_pending_rows"] == 0
    assert result["cleanup_legacy_enabled"] is True
    assert result["invalid_year_rows"] == 0
    assert result["invalid_year_purge_enabled"] is False
    assert result["invalid_year_rows_total"] == 5
    assert result["force_rebuild_rows"] == 0
    assert result["force_rebuild_deletes"] == 0
    assert result["deleted_total"] == 3
    assert sorted(result["deleted_keys"]) == [
        "k_legacy_1",
        "k_legacy_2",
        "k_mismatch_1",
    ]
    assert result["would_delete_total"] == 3
    assert result["protected_prompt_version"] == batch_script.COACH_CACHE_PROMPT_VERSION

    delete_queries = [
        q
        for q in fake_conn.executed_queries
        if q[0].strip().upper().startswith("DELETE")
    ]
    assert len(delete_queries) == 1


@pytest.mark.asyncio
async def test_cleanup_cache_rows_force_rebuild_with_async_pool() -> None:
    options = batch_script.RunOptions(
        years=[2025],
        teams=["LG"],
        focus=[],
        only_missing=False,
        force_rebuild=True,
        quality_report=None,
        baseline=False,
        game_type="REGULAR",
        delay_seconds=0.0,
        cleanup_legacy=True,
    )
    target_keys = batch_script.build_target_batch_keys(options)
    fake_conn = FakeConnection(
        select_key_batches=[
            ["k_mismatch_1"],
            ["k_legacy_1"],
            [],
        ],
        target_existing_rows=[(target_keys[0], "COMPLETED", True)],
        count_results=[2],
        delete_returning_keys=["k_legacy_1", "k_mismatch_1"],
    )
    pool = AsyncOnlyFakePool(fake_conn)
    resolver = batch_script.TeamCodeResolver()

    result = await batch_script.cleanup_cache_rows(pool, options, resolver)

    assert result["force_rebuild_rows"] == 1
    assert result["force_rebuild_replace_keys"] == [target_keys[0]]
    assert result["force_rebuild_deletes"] == 0
    assert result["deleted_total"] == 2
    assert result["invalid_year_rows_total"] == 2


@pytest.mark.asyncio
async def test_summarize_results_with_async_pool() -> None:
    fake_conn = FakeConnection(count_results=[0, 0])
    pool = AsyncOnlyFakePool(fake_conn)
    resolver = batch_script.TeamCodeResolver()

    options = batch_script.RunOptions(
        years=[2025],
        teams=["LG"],
        focus=[],
        only_missing=False,
        force_rebuild=False,
        quality_report=None,
        baseline=False,
        game_type="REGULAR",
        delay_seconds=0.0,
    )
    cleanup_result = {"deleted_total": 0}
    sample_results = [
        {
            "status": "success",
            "headline": "LG analysis",
            "warnings_count": 0,
            "json_parse_ok": True,
        }
    ]

    report = await batch_script.summarize_results(
        results=sample_results,
        options=options,
        cleanup_result=cleanup_result,
        pool=pool,
        resolver=resolver,
    )

    assert report["summary"]["success"] == 1
    assert report["summary"]["failed"] == 0
    assert report["summary"]["cases"] == 1
    assert "cache_invalid_year_count" in report["summary"]
    assert "legacy_residual_total" in report["summary"]


@pytest.mark.asyncio
async def test_async_main_opens_and_closes_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conn = FakeConnection(
        select_key_batches=[[], [], []],
        count_results=[0],
    )
    pool = AsyncOnlyFakePool(fake_conn)

    async def _fake_generate(**kwargs):  # noqa: ANN003
        return {"status": "success", "headline": "ok", "json_parse_ok": True}

    monkeypatch.setattr(batch_script, "get_connection_pool", lambda: pool)
    monkeypatch.setattr(batch_script, "generate_and_cache_team", _fake_generate)
    mock_close_pool = AsyncMock()
    monkeypatch.setattr(batch_script, "close_connection_pool", mock_close_pool)

    options = batch_script.RunOptions(
        years=[2025],
        teams=["LG"],
        focus=[],
        only_missing=False,
        force_rebuild=False,
        quality_report=None,
        baseline=False,
        game_type="REGULAR",
        delay_seconds=0.0,
    )

    exit_code = await batch_script.async_main(options)

    assert exit_code == 0
    pool.open.assert_awaited_once_with(wait=True, timeout=10.0)
    mock_close_pool.assert_awaited_once()


class _NotFoundQueryTool:
    def __init__(self, conn) -> None:  # noqa: ANN001
        self.conn = conn

    async def get_team_summary(self, team_id, year):  # noqa: ANN001
        return {"found": False}

    async def get_team_advanced_metrics(self, team_id, year):  # noqa: ANN001
        return {}

    async def get_team_recent_form(self, team_id, year):  # noqa: ANN001
        return {}


@pytest.mark.asyncio
async def test_generate_and_cache_team_no_summary_with_async_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conn = FakeConnection()
    pool = AsyncOnlyFakePool(fake_conn)
    monkeypatch.setattr(batch_script, "DatabaseQueryTool", _NotFoundQueryTool)

    result = await batch_script.generate_and_cache_team(
        pool=pool,
        team_id="LG",
        year=2025,
        game_type="REGULAR",
        focus=[],
        skip_completed=True,
        only_missing=False,
    )

    assert result["status"] == "failed"
    assert result["failure_category"] == "no_team_summary"
    statements = [
        " ".join(sql.split()).upper() for sql, _ in fake_conn.executed_queries
    ]
    assert statements[0].startswith("SELECT STATUS, RESPONSE_JSON")
    assert statements[1].startswith("INSERT INTO COACH_ANALYSIS_CACHE")
    assert statements[2].startswith("UPDATE COACH_ANALYSIS_CACHE SET STATUS = 'FAILED'")
    assert fake_conn.committed is True
