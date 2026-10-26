from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "cleanup_expired_cache.py"
)
SPEC = importlib.util.spec_from_file_location(
    "cleanup_expired_cache_under_test", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
cleanup_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cleanup_script
SPEC.loader.exec_module(cleanup_script)


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
        select_rows: list[tuple] | None = None,
        delete_returning_rows: list[tuple] | None = None,
    ) -> None:
        self.select_rows = (
            select_rows
            if select_rows is not None
            else [
                ("key1", "LG", 2025, "COMPLETED", "2025-01-01T00:00:00"),
                ("key2", "SSG", 2025, "COMPLETED", "2025-01-02T00:00:00"),
            ]
        )
        self.delete_returning_rows = delete_returning_rows
        self.executed_queries: list[tuple[str, tuple]] = []
        self.committed = False

    async def execute(self, sql: str, params: tuple = ()) -> FakeCursor:
        self.executed_queries.append((sql, params))
        clean_sql = " ".join(sql.strip().split()).upper()
        if clean_sql.startswith("SELECT"):
            return FakeCursor(rows=self.select_rows, rowcount=len(self.select_rows))
        if clean_sql.startswith("DELETE"):
            rows = (
                self.delete_returning_rows
                if self.delete_returning_rows is not None
                else []
            )
            return FakeCursor(rows=rows, rowcount=len(rows))
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
async def test_cleanup_expired_cache_dry_run_issues_no_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_rows = [
        ("key1", "LG", 2025, "COMPLETED", "2025-01-01T00:00:00"),
        ("key2", "SSG", 2025, "COMPLETED", "2025-01-02T00:00:00"),
    ]
    fake_conn = FakeConnection(select_rows=sample_rows)
    fake_pool = AsyncOnlyFakePool(fake_conn)
    monkeypatch.setattr(cleanup_script, "get_connection_pool", lambda: fake_pool)

    result = await cleanup_script.cleanup_expired_cache(
        retention_days=7,
        dry_run=True,
        years=[2025],
    )

    assert result["would_delete"] == 2
    assert result["deleted"] == 0
    assert result["dry_run"] is True
    assert result["would_delete_keys"] == ["key1", "key2"]
    fake_pool.open.assert_awaited_once_with(wait=True, timeout=10.0)
    fake_pool.close.assert_awaited_once()

    queries = [q[0].strip().upper() for q in fake_conn.executed_queries]
    assert len(queries) == 1
    assert queries[0].startswith("SELECT")
    assert not any(q.startswith("DELETE") for q in queries)
    assert fake_conn.committed is False


@pytest.mark.asyncio
async def test_cleanup_expired_cache_non_dry_run_issues_delete_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delete_rows = [
        ("key1", "LG", 2025, "COMPLETED", "2025-01-01T00:00:00"),
        ("key2", "LG", 2025, "COMPLETED", "2025-01-02T00:00:00"),
        ("key3", "LG", 2025, "COMPLETED", "2025-01-03T00:00:00"),
    ]
    fake_conn = FakeConnection(delete_returning_rows=delete_rows)
    fake_pool = AsyncOnlyFakePool(fake_conn)
    monkeypatch.setattr(cleanup_script, "get_connection_pool", lambda: fake_pool)

    result = await cleanup_script.cleanup_expired_cache(
        retention_days=14,
        dry_run=False,
        years=[2025],
        teams=["LG"],
    )

    assert result["would_delete"] == 3
    assert result["deleted"] == 3
    assert result["dry_run"] is False
    assert result["deleted_keys"] == ["key1", "key2", "key3"]
    fake_pool.open.assert_awaited_once_with(wait=True, timeout=10.0)
    fake_pool.close.assert_awaited_once()
    assert fake_conn.committed is True

    assert len(fake_conn.executed_queries) == 1
    delete_sql, delete_params = fake_conn.executed_queries[0]
    assert delete_sql.strip().upper().startswith("DELETE")
    assert "RETURNING" in delete_sql.upper()
    assert delete_params == (14, [2025], ["LG"])


@pytest.mark.asyncio
async def test_cleanup_expired_cache_global_guard_raises() -> None:
    with pytest.raises(ValueError, match="global cleanup is blocked by default"):
        await cleanup_script.cleanup_expired_cache(
            retention_days=7,
            dry_run=True,
            years=None,
            teams=None,
            allow_global=False,
        )


def test_cleanup_expired_cache_main_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample_rows = [
        ("key1", "DB", 2025, "COMPLETED", "2025-01-01T00:00:00"),
    ]
    fake_conn = FakeConnection(select_rows=sample_rows)
    fake_pool = AsyncOnlyFakePool(fake_conn)
    monkeypatch.setattr(cleanup_script, "get_connection_pool", lambda: fake_pool)

    output_path = tmp_path / "cleanup_report.json"
    monkeypatch.setattr(
        cleanup_script.sys,
        "argv",
        [
            "cleanup_expired_cache.py",
            "--dry-run",
            "--years",
            "2025",
            "--output",
            str(output_path),
        ],
    )

    exit_code = cleanup_script.main()

    assert exit_code == 0
    fake_pool.open.assert_awaited_once_with(wait=True, timeout=10.0)
    fake_pool.close.assert_awaited_once()

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["fatal_error"] is None
    assert payload["dry_run"] is True
    assert payload["result"]["would_delete"] == 1
    assert payload["result"]["deleted"] == 0


@pytest.mark.asyncio
async def test_cleanup_expired_cache_no_rows_closes_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conn = FakeConnection(delete_returning_rows=[])
    fake_pool = AsyncOnlyFakePool(fake_conn)
    monkeypatch.setattr(cleanup_script, "get_connection_pool", lambda: fake_pool)

    result = await cleanup_script.cleanup_expired_cache(
        retention_days=7,
        dry_run=False,
        years=[2025],
    )

    assert result["deleted"] == 0
    assert result["would_delete"] == 0
    assert len(fake_conn.executed_queries) == 1
    fake_pool.close.assert_awaited_once()
