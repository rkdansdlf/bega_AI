from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

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
    def __init__(self, delete_rowcount: int = 5) -> None:
        self.delete_rowcount = delete_rowcount
        self.executed_queries: list[tuple[str, tuple]] = []
        self.committed = False

    async def execute(self, sql: str, params: tuple = ()) -> FakeCursor:
        self.executed_queries.append((sql, params))
        clean_sql = " ".join(sql.strip().split()).upper()
        if clean_sql.startswith("SELECT COUNT(*)"):
            if "PROMPT_VERSION <>" in clean_sql:
                return FakeCursor(rows=[(1,)], rowcount=1)
            if "OR YEAR >" in clean_sql:
                return FakeCursor(rows=[(2,)], rowcount=1)
            if "UPPER(TEAM_ID) = ANY" in clean_sql:
                return FakeCursor(rows=[(3,)], rowcount=1)
            return FakeCursor(rows=[(0,)], rowcount=1)
        if clean_sql.startswith("DELETE"):
            return FakeCursor(rows=[], rowcount=self.delete_rowcount)
        return FakeCursor()

    async def commit(self) -> None:
        self.committed = True


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
    fake_conn = FakeConnection(delete_rowcount=6)
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
    )

    result = await batch_script.cleanup_cache_rows(pool, options, resolver)

    assert result["prompt_mismatch_rows"] == 1
    assert result["invalid_year_rows"] == 2
    assert result["legacy_team_rows"] == 3
    assert result["force_rebuild_rows"] == 0
    assert result["deleted_total"] == 6
    assert fake_conn.committed is True

    delete_queries = [
        q
        for q in fake_conn.executed_queries
        if q[0].strip().upper().startswith("DELETE")
    ]
    assert len(delete_queries) == 1


@pytest.mark.asyncio
async def test_cleanup_cache_rows_force_rebuild_with_async_pool() -> None:
    fake_conn = FakeConnection(delete_rowcount=10)
    pool = AsyncOnlyFakePool(fake_conn)
    resolver = batch_script.TeamCodeResolver()

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
    )

    result = await batch_script.cleanup_cache_rows(pool, options, resolver)

    assert result["force_rebuild_rows"] == 3
    assert result["deleted_total"] == 10
    assert fake_conn.committed is True


@pytest.mark.asyncio
async def test_summarize_results_with_async_pool() -> None:
    fake_conn = FakeConnection()
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
    fake_conn = FakeConnection()
    pool = AsyncOnlyFakePool(fake_conn)

    async def _fake_generate(**kwargs):  # noqa: ANN003
        return {"status": "success", "headline": "ok", "json_parse_ok": True}

    monkeypatch.setattr(batch_script, "get_connection_pool", lambda: pool)
    monkeypatch.setattr(batch_script, "generate_and_cache_team", _fake_generate)

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
    pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_cache_rows_delete_params_unchanged() -> None:
    """DELETE 조건/파라미터가 동기 구현과 동일한지(삭제 대상 불변) 확인."""
    fake_conn = FakeConnection(delete_rowcount=4)
    pool = AsyncOnlyFakePool(fake_conn)
    resolver = batch_script.TeamCodeResolver()
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
    )

    await batch_script.cleanup_cache_rows(pool, options, resolver)

    delete_calls = [
        (sql, params)
        for sql, params in fake_conn.executed_queries
        if sql.strip().upper().startswith("DELETE")
    ]
    assert len(delete_calls) == 1
    _, params = delete_calls[0]
    legacy_aliases = batch_script.collect_regular_aliases(resolver)
    assert params == (
        [2025],
        batch_script.PROMPT_VERSION,
        batch_script.COACH_YEAR_MIN,
        batch_script.datetime.now().year + 1,
        [2025],
        legacy_aliases,
        [2025],
        ["LG"],
    )


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
