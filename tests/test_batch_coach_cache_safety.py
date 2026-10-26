from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import psycopg

import app.core.coach_cache_contract

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "batch_coach_cache.py"
SPEC = importlib.util.spec_from_file_location(
    "batch_coach_cache_safety_under_test", SCRIPT_PATH
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
        select_status_response_json: tuple[str, Any] | None = None,
        has_select_status_response: bool = False,
        select_key_batches: list[list[str]] | None = None,
        target_existing_rows: list[tuple[str, str, bool]] | None = None,
        count_results: list[int] | None = None,
        delete_returning_keys: list[str] | None = None,
        fail_update: bool = False,
    ) -> None:
        self.select_status_response_json = select_status_response_json
        self.has_select_status_response = has_select_status_response or (
            select_status_response_json is not None
        )
        self._key_batches = list(select_key_batches or [])
        self._target_existing_rows = target_existing_rows
        self._counts = list(count_results or [])
        self._delete_returning_keys = list(delete_returning_keys or [])
        self._key_idx = 0
        self._count_idx = 0
        self.executed_queries: list[tuple[str, tuple]] = []
        self.committed = False
        self.rollback_raised = False
        self.fail_update = fail_update

    async def execute(self, sql: str, params: tuple = ()) -> FakeCursor:
        self.executed_queries.append((sql, params))
        clean_sql = " ".join(sql.strip().split()).upper()

        if clean_sql.startswith("SET TRANSACTION READ ONLY"):
            return FakeCursor()

        if clean_sql.startswith("SELECT STATUS, RESPONSE_JSON"):
            if self.has_select_status_response and self.select_status_response_json:
                return FakeCursor(rows=[self.select_status_response_json])
            return FakeCursor(rows=[])

        if clean_sql.startswith("SELECT COUNT(*)"):
            count = (
                self._counts[self._count_idx]
                if self._count_idx < len(self._counts)
                else 0
            )
            self._count_idx += 1
            if clean_sql.startswith("SELECT COUNT(*) FILTER"):
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

        if clean_sql.startswith("UPDATE"):
            if self.fail_update:
                raise psycopg.DatabaseError("simulated update failure")
            return FakeCursor(rowcount=1)

        if clean_sql.startswith("INSERT"):
            return FakeCursor(rowcount=1)

        return FakeCursor()

    async def commit(self) -> None:
        self.committed = True

    @contextlib.asynccontextmanager
    async def transaction(self):
        try:
            yield
        except psycopg.Rollback:
            self.rollback_raised = True


class AsyncOnlyFakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection
        self.open = AsyncMock()
        self.close = AsyncMock()

    @contextlib.asynccontextmanager
    async def connection(self):
        yield self._connection


class FakeDatabaseQueryTool:
    def __init__(self, conn=None) -> None:
        self.conn = conn

    async def get_team_summary(self, team_id: str, year: int) -> dict[str, Any]:
        return {"found": True, "team_name": team_id}

    async def get_team_advanced_metrics(
        self, team_id: str, year: int
    ) -> dict[str, Any]:
        return {}

    async def get_team_recent_form(self, team_id: str, year: int) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# 1.a) build_cleanup_conditions checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("force_rebuild", [False, True])
def test_build_cleanup_conditions_safety(force_rebuild: bool) -> None:
    resolver = batch_script.TeamCodeResolver()
    options = batch_script.RunOptions(
        years=[2025],
        teams=batch_script.CANONICAL_TEAMS.copy(),
        focus=[],
        only_missing=False,
        force_rebuild=force_rebuild,
        quality_report=None,
        baseline=False,
        game_type="REGULAR",
        delay_seconds=0.0,
        purge_invalid_years=False,
    )

    conditions = batch_script.build_cleanup_conditions(options, resolver)
    categories = {cat for cat, _, _ in conditions}
    assert categories == {"prompt_mismatch", "legacy_team", "stale_pending"}

    for category, where_sql, params in conditions:
        assert "prompt_version <> %s" in where_sql
        assert batch_script.COACH_CACHE_PROMPT_VERSION in params
        assert "prompt_version = ANY(%s)" in where_sql

        # Find the list parameter supplied to prompt_version = ANY(%s)
        pv_any_params = [
            p
            for p in params
            if isinstance(p, list)
            and all(isinstance(v, str) for v in p)
            and any("prompt" in v or "focus" in v for v in p)
        ]
        assert len(pv_any_params) >= 1
        for pv_list in pv_any_params:
            assert batch_script.COACH_CACHE_PROMPT_VERSION not in pv_list

        assert "year = ANY(%s)" in where_sql
        assert [2025] in params


def test_build_cleanup_conditions_purge_invalid_years() -> None:
    resolver = batch_script.TeamCodeResolver()
    options = batch_script.RunOptions(
        years=[2025],
        teams=batch_script.CANONICAL_TEAMS.copy(),
        focus=[],
        only_missing=False,
        force_rebuild=False,
        quality_report=None,
        baseline=False,
        game_type="REGULAR",
        delay_seconds=0.0,
        purge_invalid_years=True,
    )

    conditions = batch_script.build_cleanup_conditions(options, resolver)
    categories = {cat for cat, _, _ in conditions}
    assert categories == {
        "prompt_mismatch",
        "legacy_team",
        "stale_pending",
        "invalid_year",
    }

    invalid_year_cond = next(c for c in conditions if c[0] == "invalid_year")
    cat, where_sql, params = invalid_year_cond
    assert "(year < %s OR year > %s)" in where_sql
    assert "prompt_version = ANY(%s)" in where_sql
    assert "prompt_version <> %s" in where_sql
    assert batch_script.COACH_CACHE_PROMPT_VERSION in params

    # Restricted to BATCH_OWNED_PROMPT_VERSIONS and excluding live version
    assert list(batch_script.BATCH_OWNED_PROMPT_VERSIONS) in params
    assert (
        batch_script.COACH_CACHE_PROMPT_VERSION
        not in batch_script.BATCH_OWNED_PROMPT_VERSIONS
    )


# ---------------------------------------------------------------------------
# 1.b) High-2 semantic check without a DB
# ---------------------------------------------------------------------------
def row_matches_cleanup_conditions(
    row: dict[str, Any],
    conditions: list[tuple[str, str, list[Any]]],
) -> bool:
    """Emulate cleanup conditions without parsing SQL generally."""
    for category, where_sql, params in conditions:
        if category == "prompt_mismatch":
            assert (
                where_sql
                == "year = ANY(%s) AND prompt_version = ANY(%s) AND prompt_version <> %s"
            )
            years, prompt_versions, excluded_version = params
            if (
                row.get("year") in years
                and row.get("prompt_version") in prompt_versions
                and row.get("prompt_version") != excluded_version
            ):
                return True
        elif category == "legacy_team":
            assert where_sql == (
                "year = ANY(%s) AND UPPER(team_id) = ANY(%s)"
                " AND prompt_version = ANY(%s) AND prompt_version <> %s"
            )
            years, aliases, prompt_versions, excluded_version = params
            if (
                row.get("year") in years
                and str(row.get("team_id", "")).upper() in aliases
                and row.get("prompt_version") in prompt_versions
                and row.get("prompt_version") != excluded_version
            ):
                return True
        elif category == "stale_pending":
            assert where_sql == (
                "year = ANY(%s) AND status = 'PENDING'"
                " AND updated_at < now() - make_interval(hours => %s)"
                " AND prompt_version = ANY(%s) AND prompt_version <> %s"
            )
            years, stale_hours, prompt_versions, excluded_version = params
            if (
                row.get("year") in years
                and row.get("status") == "PENDING"
                and row.get("stale_pending", False)
                and row.get("prompt_version") in prompt_versions
                and row.get("prompt_version") != excluded_version
            ):
                return True
        elif category == "invalid_year":
            assert where_sql == (
                "(year < %s OR year > %s)"
                " AND prompt_version = ANY(%s) AND prompt_version <> %s"
            )
            min_year, max_year, prompt_versions, excluded_version = params
            year = row.get("year", 0)
            if (
                (year < min_year or year > max_year)
                and row.get("prompt_version") in prompt_versions
                and row.get("prompt_version") != excluded_version
            ):
                return True
        else:
            raise ValueError(f"Unknown condition category: {category}")
    return False


def test_cleanup_semantic_match_evaluation() -> None:
    resolver = batch_script.TeamCodeResolver()
    options = batch_script.RunOptions(
        years=[2025],
        teams=batch_script.CANONICAL_TEAMS.copy(),
        focus=[],
        only_missing=False,
        force_rebuild=False,
        quality_report=None,
        baseline=False,
        game_type="REGULAR",
        delay_seconds=0.0,
        purge_invalid_years=True,
    )
    conditions = batch_script.build_cleanup_conditions(options, resolver)

    # Prompt mismatch allowlist check
    assert list(batch_script.BATCH_LEGACY_PROMPT_VERSIONS) == ["v3_prompt", "v4_dual"]

    # Live row must never match cleanup
    live_row = {
        "year": 2025,
        "prompt_version": batch_script.COACH_CACHE_PROMPT_VERSION,
        "team_id": "LG",
    }
    assert not row_matches_cleanup_conditions(live_row, conditions)

    # Live row with alias team must never match cleanup
    live_alias_row = {
        "year": 2025,
        "prompt_version": batch_script.COACH_CACHE_PROMPT_VERSION,
        "team_id": "OB",
    }
    assert not row_matches_cleanup_conditions(live_alias_row, conditions)

    # Canonical current version row must not match cleanup
    current_canonical_row = {
        "year": 2025,
        "prompt_version": "v5_focus",
        "team_id": "LG",
    }
    assert not row_matches_cleanup_conditions(current_canonical_row, conditions)

    # Legacy prompt row matches prompt_mismatch
    legacy_prompt_row = {
        "year": 2025,
        "prompt_version": "v4_dual",
        "team_id": "LG",
    }
    assert row_matches_cleanup_conditions(legacy_prompt_row, conditions)

    # Legacy alias row matches legacy_team
    legacy_alias_row = {
        "year": 2025,
        "prompt_version": "v5_focus",
        "team_id": "OB",
    }
    assert row_matches_cleanup_conditions(legacy_alias_row, conditions)

    # Stale pending batch-owned row matches stale_pending
    stale_pending_batch_row = {
        "year": 2025,
        "status": "PENDING",
        "stale_pending": True,
        "prompt_version": "v5_focus",
        "team_id": "LG",
    }
    assert row_matches_cleanup_conditions(stale_pending_batch_row, conditions)

    # Fresh pending batch-owned row does not match
    fresh_pending_batch_row = {
        "year": 2025,
        "status": "PENDING",
        "stale_pending": False,
        "prompt_version": "v5_focus",
        "team_id": "LG",
    }
    assert not row_matches_cleanup_conditions(fresh_pending_batch_row, conditions)

    # Stale pending live contract row does not match
    stale_pending_live_row = {
        "year": 2025,
        "status": "PENDING",
        "stale_pending": True,
        "prompt_version": batch_script.COACH_CACHE_PROMPT_VERSION,
        "team_id": "LG",
    }
    assert not row_matches_cleanup_conditions(stale_pending_live_row, conditions)


# ---------------------------------------------------------------------------
# 1.c) High-1 unit tests for generate_and_cache_team
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_and_cache_team_force_rebuild_llm_failure_keeps_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conn = FakeConnection(
        select_status_response_json=("COMPLETED", {"headline": "old"})
    )
    pool = AsyncOnlyFakePool(fake_conn)

    monkeypatch.setattr(batch_script, "DatabaseQueryTool", FakeDatabaseQueryTool)

    async def _failing_llm(messages):
        if False:
            yield ""
        raise RuntimeError("llm down")

    monkeypatch.setattr(batch_script, "get_coach_llm_generator", lambda: _failing_llm)

    result = await batch_script.generate_and_cache_team(
        pool=pool,
        team_id="LG",
        year=2025,
        game_type="REGULAR",
        focus=[],
        skip_completed=False,
        only_missing=False,
        force_rebuild=True,
    )

    executed_sqls = [
        " ".join(sql.strip().split()).upper() for sql, _ in fake_conn.executed_queries
    ]
    for sql in executed_sqls:
        assert not sql.startswith("DELETE")
        assert not sql.startswith("INSERT")
        assert not sql.startswith("UPDATE")

    assert result["status"] == "failed"
    assert result["failure_category"] == "llm_error"
    assert result["previous_kept"] is True


@pytest.mark.asyncio
async def test_generate_and_cache_team_force_rebuild_success_replaces_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conn = FakeConnection(
        select_status_response_json=("COMPLETED", {"headline": "old"})
    )
    pool = AsyncOnlyFakePool(fake_conn)

    monkeypatch.setattr(batch_script, "DatabaseQueryTool", FakeDatabaseQueryTool)

    async def _yielding_llm(messages):
        yield '{"headline": "LG Analysis", "key_metrics": []}'

    monkeypatch.setattr(batch_script, "get_coach_llm_generator", lambda: _yielding_llm)

    class FakeParsedResponse:
        headline = "LG Analysis"
        key_metrics: list = []

        def model_dump(self) -> dict[str, Any]:
            return {"headline": self.headline, "key_metrics": []}

    fake_parsed = FakeParsedResponse()
    monkeypatch.setattr(
        batch_script,
        "parse_coach_response_with_meta",
        lambda resp: (fake_parsed, None, {}),
    )

    class FakeGroundingValidation:
        reasons: list = []
        warnings: list = []

    monkeypatch.setattr(
        batch_script,
        "validate_response_against_fact_sheet",
        lambda parsed, fact_sheet: FakeGroundingValidation(),
    )
    monkeypatch.setattr(batch_script, "validate_coach_response", lambda parsed: [])

    result = await batch_script.generate_and_cache_team(
        pool=pool,
        team_id="LG",
        year=2025,
        game_type="REGULAR",
        focus=[],
        skip_completed=False,
        only_missing=False,
        force_rebuild=True,
    )

    executed_sqls = [
        " ".join(sql.strip().split()).upper() for sql, _ in fake_conn.executed_queries
    ]
    insert_queries = [s for s in executed_sqls if s.startswith("INSERT")]
    delete_queries = [s for s in executed_sqls if s.startswith("DELETE")]
    update_queries = [
        q
        for q in fake_conn.executed_queries
        if q[0].strip().upper().startswith("UPDATE")
    ]

    assert len(insert_queries) == 0
    assert len(delete_queries) == 0
    assert len(update_queries) == 1

    update_sql, update_params = update_queries[0]
    clean_update_sql = " ".join(update_sql.strip().split()).upper()
    assert "STATUS = 'COMPLETED'" in clean_update_sql
    assert "PROMPT_VERSION = %S" in clean_update_sql
    assert batch_script.PROMPT_VERSION in update_params

    assert result["replaced_previous"] is True
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_generate_and_cache_team_non_force_llm_failure_legacy_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conn = FakeConnection(
        select_status_response_json=None,
        has_select_status_response=True,
    )
    pool = AsyncOnlyFakePool(fake_conn)

    monkeypatch.setattr(batch_script, "DatabaseQueryTool", FakeDatabaseQueryTool)

    async def _failing_llm(messages):
        if False:
            yield ""
        raise RuntimeError("llm down")

    monkeypatch.setattr(batch_script, "get_coach_llm_generator", lambda: _failing_llm)

    result = await batch_script.generate_and_cache_team(
        pool=pool,
        team_id="LG",
        year=2025,
        game_type="REGULAR",
        focus=[],
        skip_completed=True,
        only_missing=False,
        force_rebuild=False,
    )

    assert result["status"] == "failed"
    assert result["failure_category"] == "llm_error"
    assert result.get("previous_kept") is not True

    executed_sqls = [
        " ".join(sql.strip().split()).upper() for sql, _ in fake_conn.executed_queries
    ]
    assert executed_sqls[0].startswith("SELECT STATUS, RESPONSE_JSON")
    assert executed_sqls[1].startswith("INSERT INTO COACH_ANALYSIS_CACHE")
    assert "PENDING" in executed_sqls[1]
    assert executed_sqls[2].startswith(
        "UPDATE COACH_ANALYSIS_CACHE SET STATUS = 'FAILED'"
    )


# ---------------------------------------------------------------------------
# 1.d) cleanup_cache_rows dry_run and force_rebuild checks
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cleanup_cache_rows_dry_run_zero_deletes() -> None:
    fake_conn = FakeConnection(
        select_key_batches=[
            ["k_mismatch_1"],
            ["k_legacy_1", "k_legacy_2"],
            [],
        ],
        count_results=[0],
    )
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
        dry_run=True,
        cleanup_legacy=True,
    )

    result = await batch_script.cleanup_cache_rows(
        pool, options, resolver, dry_run=True
    )

    assert result["deleted_total"] == 0
    assert result["would_delete_keys"] == ["k_legacy_1", "k_legacy_2", "k_mismatch_1"]
    assert result["cleanup_candidate_keys"] == [
        "k_legacy_1",
        "k_legacy_2",
        "k_mismatch_1",
    ]
    assert fake_conn.rollback_raised is True
    assert fake_conn.committed is False

    delete_queries = [
        q
        for q in fake_conn.executed_queries
        if q[0].strip().upper().startswith("DELETE")
    ]
    assert len(delete_queries) == 0


@pytest.mark.asyncio
async def test_cleanup_cache_rows_real_delete_excludes_force_rebuild_targets() -> None:
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
        dry_run=False,
        cleanup_legacy=True,
    )
    target_keys = batch_script.build_target_batch_keys(options)
    assert len(target_keys) > 0

    fake_conn = FakeConnection(
        select_key_batches=[
            ["k_mismatch_1"],
            ["k_legacy_1"],
            [],  # stale_pending
        ],
        target_existing_rows=[(tk, "COMPLETED", True) for tk in target_keys],
        count_results=[0],
        delete_returning_keys=["k_legacy_1", "k_mismatch_1"],
    )
    pool = AsyncOnlyFakePool(fake_conn)
    resolver = batch_script.TeamCodeResolver()

    result = await batch_script.cleanup_cache_rows(
        pool, options, resolver, dry_run=False
    )

    delete_queries = [
        q
        for q in fake_conn.executed_queries
        if q[0].strip().upper().startswith("DELETE")
    ]
    assert len(delete_queries) == 1

    delete_sql, delete_params = delete_queries[0]
    assert "RETURNING CACHE_KEY" in delete_sql.upper()
    assert result["deleted_total"] == 2
    assert result["deleted_keys"] == ["k_legacy_1", "k_mismatch_1"]
    assert result["force_rebuild_replace_keys"] == target_keys

    # Ensure no DELETE statement or params contain force-rebuild target keys
    for tk in target_keys:
        assert tk not in delete_sql
        assert tk not in delete_params
        for p in delete_params:
            if isinstance(p, (list, tuple, set)):
                assert tk not in p


@pytest.mark.asyncio
async def test_cleanup_legacy_gating() -> None:
    resolver = batch_script.TeamCodeResolver()

    # 1. Without cleanup_legacy and not dry-run:
    # No DELETE issued, would_delete_keys == [], cleanup_candidate_keys populated,
    # SET TRANSACTION READ ONLY executed, Rollback raised.
    fake_conn_no_legacy = FakeConnection(
        select_key_batches=[
            ["k_mismatch_1"],
            ["k_legacy_1"],
            [],
        ],
        count_results=[0],
    )
    pool_no_legacy = AsyncOnlyFakePool(fake_conn_no_legacy)
    options_no_legacy = batch_script.RunOptions(
        years=[2025],
        teams=["LG"],
        focus=[],
        only_missing=False,
        force_rebuild=False,
        quality_report=None,
        baseline=False,
        game_type="REGULAR",
        delay_seconds=0.0,
        dry_run=False,
        cleanup_legacy=False,
    )
    result_no_legacy = await batch_script.cleanup_cache_rows(
        pool_no_legacy, options_no_legacy, resolver, dry_run=False
    )

    delete_queries = [
        q
        for q in fake_conn_no_legacy.executed_queries
        if q[0].strip().upper().startswith("DELETE")
    ]
    assert len(delete_queries) == 0
    assert result_no_legacy["cleanup_legacy_enabled"] is False
    assert result_no_legacy["would_delete_keys"] == []
    assert result_no_legacy["would_delete_total"] == 0
    assert result_no_legacy["cleanup_candidate_keys"] == ["k_legacy_1", "k_mismatch_1"]
    assert result_no_legacy["deleted_keys"] == []
    assert result_no_legacy["deleted_total"] == 0
    assert fake_conn_no_legacy.rollback_raised is True
    assert fake_conn_no_legacy.committed is False
    read_only_queries = [
        q
        for q in fake_conn_no_legacy.executed_queries
        if "SET TRANSACTION READ ONLY" in q[0].strip().upper()
    ]
    assert len(read_only_queries) == 1

    # 2. With cleanup_legacy and not dry-run:
    # DELETE issued, would_delete_keys == candidate_keys, deleted_keys populated.
    fake_conn_legacy = FakeConnection(
        select_key_batches=[
            ["k_mismatch_1"],
            ["k_legacy_1"],
            [],
        ],
        count_results=[0],
        delete_returning_keys=["k_legacy_1", "k_mismatch_1"],
    )
    pool_legacy = AsyncOnlyFakePool(fake_conn_legacy)
    options_legacy = batch_script.RunOptions(
        years=[2025],
        teams=["LG"],
        focus=[],
        only_missing=False,
        force_rebuild=False,
        quality_report=None,
        baseline=False,
        game_type="REGULAR",
        delay_seconds=0.0,
        dry_run=False,
        cleanup_legacy=True,
    )
    result_legacy = await batch_script.cleanup_cache_rows(
        pool_legacy, options_legacy, resolver, dry_run=False
    )

    delete_queries_legacy = [
        q
        for q in fake_conn_legacy.executed_queries
        if q[0].strip().upper().startswith("DELETE")
    ]
    assert len(delete_queries_legacy) == 1
    assert result_legacy["cleanup_legacy_enabled"] is True
    assert result_legacy["would_delete_keys"] == ["k_legacy_1", "k_mismatch_1"]
    assert result_legacy["cleanup_candidate_keys"] == ["k_legacy_1", "k_mismatch_1"]
    assert result_legacy["deleted_keys"] == ["k_legacy_1", "k_mismatch_1"]
    assert result_legacy["deleted_total"] == 2
    assert fake_conn_legacy.committed is False


@pytest.mark.asyncio
async def test_dry_run_read_only_records_rollback() -> None:
    fake_conn = FakeConnection(
        select_key_batches=[
            ["k_mismatch_1"],
            [],
            [],
        ],
        count_results=[0],
    )
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
        dry_run=True,
        cleanup_legacy=True,
    )

    await batch_script.cleanup_cache_rows(pool, options, resolver, dry_run=True)

    assert fake_conn.rollback_raised is True
    assert fake_conn.committed is False
    assert any(
        "SET TRANSACTION READ ONLY" in q[0].strip().upper()
        for q in fake_conn.executed_queries
    )
    assert not any(
        q[0].strip().upper().startswith("DELETE") for q in fake_conn.executed_queries
    )


@pytest.mark.asyncio
async def test_per_key_isolation_in_run_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = batch_script.RunOptions(
        years=[2025],
        teams=["LG", "KIA"],
        focus=[],
        only_missing=False,
        force_rebuild=False,
        quality_report=None,
        baseline=False,
        game_type="REGULAR",
        delay_seconds=0.0,
    )

    processed_teams: list[str] = []

    async def _fake_generate(pool, team_id, year, **kwargs):
        processed_teams.append(team_id)
        if team_id == "LG":
            raise RuntimeError("team LG exploded")
        return {
            "team": team_id,
            "year": year,
            "status": "success",
            "headline": f"{team_id} ok",
            "json_parse_ok": True,
            "warnings_count": 0,
            "critical_over_limit": False,
            "validator_hard_fail": False,
            "normalization_applied": False,
            "normalization_reasons": [],
            "elapsed_seconds": 0.05,
        }

    monkeypatch.setattr(batch_script, "generate_and_cache_team", _fake_generate)
    monkeypatch.setattr(
        batch_script,
        "cleanup_cache_rows",
        AsyncMock(return_value={"deleted_total": 0, "would_delete_total": 0}),
    )

    captured_summary_results: list[dict[str, Any]] = []

    async def _fake_summarize(results, options, cleanup_result, pool, resolver):
        captured_summary_results.extend(list(results))
        return {
            "summary": {
                "all_ok": False,
                "cases": len(captured_summary_results),
                "success": 1,
                "failed": 1,
            }
        }

    monkeypatch.setattr(batch_script, "summarize_results", _fake_summarize)

    fake_conn = FakeConnection()
    pool = AsyncOnlyFakePool(fake_conn)

    exit_code = await batch_script._run_batch(options, pool)

    assert exit_code == 1
    assert processed_teams == ["LG", "KIA"]
    assert len(captured_summary_results) == 2

    failed_res = next(r for r in captured_summary_results if r["team"] == "LG")
    assert failed_res["status"] == "failed"
    assert failed_res["failure_category"] == "unexpected_error"
    assert "unexpected_error:RuntimeError: team LG exploded" in failed_res["reason"]

    success_res = next(r for r in captured_summary_results if r["team"] == "KIA")
    assert success_res["status"] == "success"


@pytest.mark.asyncio
async def test_generate_and_cache_team_cache_write_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conn = FakeConnection(
        select_status_response_json=("COMPLETED", {"headline": "old"}),
        fail_update=True,
    )
    pool = AsyncOnlyFakePool(fake_conn)

    monkeypatch.setattr(batch_script, "DatabaseQueryTool", FakeDatabaseQueryTool)

    async def _yielding_llm(messages):
        yield '{"headline": "LG Analysis", "key_metrics": []}'

    monkeypatch.setattr(batch_script, "get_coach_llm_generator", lambda: _yielding_llm)

    class FakeParsedResponse:
        headline = "LG Analysis"
        key_metrics: list = []

        def model_dump(self) -> dict[str, Any]:
            return {"headline": self.headline, "key_metrics": []}

    fake_parsed = FakeParsedResponse()
    monkeypatch.setattr(
        batch_script,
        "parse_coach_response_with_meta",
        lambda resp: (fake_parsed, None, {}),
    )

    class FakeGroundingValidation:
        reasons: list = []
        warnings: list = []

    monkeypatch.setattr(
        batch_script,
        "validate_response_against_fact_sheet",
        lambda parsed, fact_sheet: FakeGroundingValidation(),
    )
    monkeypatch.setattr(batch_script, "validate_coach_response", lambda parsed: [])

    result = await batch_script.generate_and_cache_team(
        pool=pool,
        team_id="LG",
        year=2025,
        game_type="REGULAR",
        focus=[],
        skip_completed=False,
        only_missing=False,
        force_rebuild=True,
    )

    assert result["status"] == "failed"
    assert result["failure_category"] == "cache_write_error"
    assert result["previous_kept"] is True
    assert "replaced_previous" not in result


# ---------------------------------------------------------------------------
# 1.e) Version check
# ---------------------------------------------------------------------------
def test_coach_cache_prompt_version_guard() -> None:
    assert (
        batch_script.COACH_CACHE_PROMPT_VERSION
        not in batch_script.BATCH_OWNED_PROMPT_VERSIONS
    )
    assert (
        batch_script.COACH_CACHE_PROMPT_VERSION
        is app.core.coach_cache_contract.COACH_CACHE_PROMPT_VERSION
    )
