from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import psycopg
import psycopg_pool
import pytest
import pytest_asyncio

from app.core.coach_cache_contract import COACH_CACHE_PROMPT_VERSION

pytestmark = pytest.mark.skipif(
    not os.environ.get("COACH_CACHE_TEST_DB_URL"),
    reason="COACH_CACHE_TEST_DB_URL not set; skipping live DB cleanup tests",
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "batch_coach_cache.py"
SPEC = importlib.util.spec_from_file_location(
    "batch_coach_cache_cleanup_db_under_test", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
batch_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch_script
SPEC.loader.exec_module(batch_script)

CLEANUP_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "cleanup_expired_cache.py"
)
CLEANUP_SPEC = importlib.util.spec_from_file_location(
    "cleanup_expired_cache_db_under_test", CLEANUP_SCRIPT_PATH
)
assert CLEANUP_SPEC is not None and CLEANUP_SPEC.loader is not None
cleanup_script = importlib.util.module_from_spec(CLEANUP_SPEC)
sys.modules[CLEANUP_SPEC.name] = cleanup_script
CLEANUP_SPEC.loader.exec_module(cleanup_script)

CREATE_TABLE_SQL = """
CREATE TABLE coach_analysis_cache (
    cache_key varchar(64) primary key, team_id varchar(10) not null, year int not null,
    prompt_version varchar(32) not null, model_name varchar(50) not null,
    status varchar(20) not null check (status in ('PENDING','COMPLETED','FAILED')),
    response_json jsonb, error_message text, error_code varchar(64),
    attempt_count int not null default 0, lease_owner varchar(80), lease_expires_at timestamptz,
    last_heartbeat_at timestamptz, created_at timestamptz default now(), updated_at timestamptz default now());
"""


@pytest_asyncio.fixture
async def pool():
    url = os.environ["COACH_CACHE_TEST_DB_URL"]
    schema = f"cc_test_{uuid4().hex[:10]}"

    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
        conn.execute(f'SET search_path TO "{schema}"')
        conn.execute(CREATE_TABLE_SQL)

    test_pool = psycopg_pool.AsyncConnectionPool(
        url,
        open=False,
        min_size=1,
        max_size=2,
        kwargs={"autocommit": True, "options": f"-c search_path={schema}"},
    )
    await test_pool.open(wait=True, timeout=10)
    try:
        yield test_pool
    finally:
        await test_pool.close()
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


async def insert_row(
    pool: psycopg_pool.AsyncConnectionPool,
    cache_key: str,
    team_id: str,
    year: int,
    prompt_version: str,
    status: str = "COMPLETED",
    response_json: dict[str, Any] | str | None = None,
    updated_at: datetime | None = None,
    model_name: str = "m",
) -> None:
    if isinstance(response_json, dict):
        resp_val: str | None = json.dumps(response_json)
    elif isinstance(response_json, str):
        resp_val = response_json
    elif response_json is None:
        resp_val = json.dumps({"x": 1})
    else:
        resp_val = json.dumps(response_json)

    if updated_at is None:
        updated_at = datetime.now(timezone.utc)

    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO coach_analysis_cache
                (cache_key, team_id, year, prompt_version, model_name, status, response_json, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                cache_key,
                team_id,
                year,
                prompt_version,
                model_name,
                status,
                resp_val,
                updated_at,
            ),
        )


async def fetch_cache_rows(
    pool: psycopg_pool.AsyncConnectionPool,
) -> dict[str, tuple[str, Any]]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT cache_key, status, response_json FROM coach_analysis_cache"
        )
        rows = await cur.fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}


@pytest.mark.asyncio
async def test_h2_cleanup_preserves_live_and_deletes_legacy(
    pool: psycopg_pool.AsyncConnectionPool,
) -> None:
    now = datetime.now(timezone.utc)
    live_pv = COACH_CACHE_PROMPT_VERSION

    await insert_row(
        pool,
        cache_key="live_2025",
        team_id="LG",
        year=2025,
        prompt_version=live_pv,
        status="COMPLETED",
        updated_at=now,
    )
    await insert_row(
        pool,
        cache_key="legacy_v4",
        team_id="SSG",
        year=2025,
        prompt_version="v4_dual",
        status="COMPLETED",
        updated_at=now - timedelta(days=1),
    )
    await insert_row(
        pool,
        cache_key="inv_2030",
        team_id="LG",
        year=2030,
        prompt_version=live_pv,
        status="COMPLETED",
        updated_at=now,
    )
    await insert_row(
        pool,
        cache_key="inv_1900",
        team_id="LG",
        year=1900,
        prompt_version="v5_focus",
        status="COMPLETED",
        updated_at=now,
    )
    await insert_row(
        pool,
        cache_key="alias_ob_live",
        team_id="OB",
        year=2025,
        prompt_version=live_pv,
        status="COMPLETED",
        updated_at=now,
    )
    await insert_row(
        pool,
        cache_key="alias_ob_batch",
        team_id="OB",
        year=2025,
        prompt_version="v5_focus",
        status="COMPLETED",
        updated_at=now,
    )

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
        cleanup_legacy=True,
    )
    resolver = batch_script.TeamCodeResolver()

    # Positional, pre-existing signature (non-dry-run by default) so this test
    # reproduces High-2 behaviourally against the previous implementation.
    result = await batch_script.cleanup_cache_rows(pool, options, resolver)

    rows = await fetch_cache_rows(pool)
    assert "live_2025" in rows
    assert "inv_2030" in rows
    assert "inv_1900" in rows
    assert "alias_ob_live" in rows
    assert "legacy_v4" not in rows
    assert "alias_ob_batch" not in rows
    assert result["deleted_keys"] == ["alias_ob_batch", "legacy_v4"]


@pytest.mark.asyncio
async def test_h1_force_rebuild_verify_then_replace_keeps_on_failure(
    pool: psycopg_pool.AsyncConnectionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    resolver = batch_script.TeamCodeResolver()
    # Computed with the shared key builder (not build_target_batch_keys) so the
    # test reproduces High-1 behaviourally against the previous implementation.
    target_key, _ = batch_script.build_coach_cache_key(
        schema_version=batch_script.CACHE_SCHEMA_VERSION,
        prompt_version=batch_script.PROMPT_VERSION,
        home_team_code="LG",
        away_team_code=None,
        year=2025,
        game_type="REGULAR",
        focus=[],
        question_override=None,
    )

    now = datetime.now(timezone.utc)
    live_pv = COACH_CACHE_PROMPT_VERSION

    await insert_row(
        pool,
        cache_key=target_key,
        team_id="LG",
        year=2025,
        prompt_version="v5_focus",
        status="COMPLETED",
        response_json='{"headline":"old"}',
        updated_at=now - timedelta(hours=1),
    )
    await insert_row(
        pool,
        cache_key="live_lg_fresh",
        team_id="LG",
        year=2025,
        prompt_version=live_pv,
        status="COMPLETED",
        response_json='{"headline":"fresh"}',
        updated_at=now,
    )
    await insert_row(
        pool,
        cache_key="matchup_2025_LG_vs_SSG",
        team_id="LG",
        year=2025,
        prompt_version="v5_focus",
        status="COMPLETED",
        response_json='{"headline":"matchup"}',
        updated_at=now - timedelta(hours=1),
    )
    await insert_row(
        pool,
        cache_key="focus_bullpen_2025_LG",
        team_id="LG",
        year=2025,
        prompt_version="v5_focus",
        status="COMPLETED",
        response_json='{"headline":"bullpen"}',
        updated_at=now - timedelta(hours=1),
    )

    cleanup_result = await batch_script.cleanup_cache_rows(pool, options, resolver)
    rows_after_cleanup = await fetch_cache_rows(pool)
    assert target_key in rows_after_cleanup
    assert len(rows_after_cleanup) == 4
    assert cleanup_result["deleted_total"] == 0
    assert cleanup_result["force_rebuild_replace_keys"] == [target_key]

    class FakeDatabaseQueryTool:
        def __init__(self, conn=None) -> None:
            pass

        async def get_team_summary(self, team_id: str, year: int) -> dict[str, Any]:
            return {"found": True, "team_name": team_id}

        async def get_team_advanced_metrics(
            self, team_id: str, year: int
        ) -> dict[str, Any]:
            return {}

        async def get_team_recent_form(self, team_id: str, year: int) -> dict[str, Any]:
            return {}

    async def _failing_llm(messages):
        if False:
            yield ""
        raise RuntimeError("llm down")

    monkeypatch.setattr(batch_script, "DatabaseQueryTool", FakeDatabaseQueryTool)
    monkeypatch.setattr(batch_script, "get_coach_llm_generator", lambda: _failing_llm)

    gen_result = await batch_script.generate_and_cache_team(
        pool=pool,
        team_id="LG",
        year=2025,
        game_type="REGULAR",
        focus=[],
        skip_completed=False,
        only_missing=False,
    )
    assert gen_result["status"] == "failed"
    assert gen_result.get("previous_kept") is True

    rows = await fetch_cache_rows(pool)
    assert len(rows) == 4
    for key in (
        target_key,
        "live_lg_fresh",
        "matchup_2025_LG_vs_SSG",
        "focus_bullpen_2025_LG",
    ):
        assert key in rows

    status, response_json = rows[target_key]
    assert status == "COMPLETED"
    assert response_json == {"headline": "old"}


@pytest.mark.asyncio
async def test_dry_run_parity(
    pool: psycopg_pool.AsyncConnectionPool,
) -> None:
    now = datetime.now(timezone.utc)
    live_pv = COACH_CACHE_PROMPT_VERSION

    # Seed mix: legacy rows, alias rows, live rows
    await insert_row(
        pool,
        cache_key="mix_legacy_v3",
        team_id="LG",
        year=2025,
        prompt_version="v3_prompt",
        status="COMPLETED",
        updated_at=now - timedelta(days=2),
    )
    await insert_row(
        pool,
        cache_key="mix_legacy_v4",
        team_id="KIA",
        year=2025,
        prompt_version="v4_dual",
        status="COMPLETED",
        updated_at=now - timedelta(days=2),
    )
    await insert_row(
        pool,
        cache_key="mix_alias_batch",
        team_id="OB",
        year=2025,
        prompt_version="v5_focus",
        status="COMPLETED",
        updated_at=now - timedelta(days=1),
    )
    await insert_row(
        pool,
        cache_key="mix_live_lg",
        team_id="LG",
        year=2025,
        prompt_version=live_pv,
        status="COMPLETED",
        updated_at=now,
    )
    await insert_row(
        pool,
        cache_key="mix_live_ssg",
        team_id="SSG",
        year=2025,
        prompt_version=live_pv,
        status="COMPLETED",
        updated_at=now,
    )
    await insert_row(
        pool,
        cache_key="mix_alias_live",
        team_id="OB",
        year=2025,
        prompt_version=live_pv,
        status="COMPLETED",
        updated_at=now,
    )

    rows_before = await fetch_cache_rows(pool)
    assert len(rows_before) == 6

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
        cleanup_legacy=True,
    )
    resolver = batch_script.TeamCodeResolver()

    dry = await batch_script.cleanup_cache_rows(pool, options, resolver, dry_run=True)
    assert dry["deleted_total"] == 0

    rows_after_dry = await fetch_cache_rows(pool)
    assert len(rows_after_dry) == 6

    real = await batch_script.cleanup_cache_rows(pool, options, resolver, dry_run=False)

    assert dry["would_delete_keys"] == real["deleted_keys"]
    assert real["would_delete_keys"] == real["deleted_keys"]
    assert len(real["deleted_keys"]) > 0

    rows_after_real = await fetch_cache_rows(pool)
    assert len(rows_after_real) == len(rows_before) - len(real["deleted_keys"])


@pytest.mark.asyncio
async def test_gate_metrics_ignore_live_contract_rows_after_purge(
    pool: psycopg_pool.AsyncConnectionPool,
) -> None:
    """Live-API rows survive cleanup and must not trip cache_integrity_fail."""
    live_pv = COACH_CACHE_PROMPT_VERSION
    await insert_row(pool, "live_inv_2030", "LG", 2030, live_pv)
    await insert_row(pool, "live_alias_ob", "OB", 2025, live_pv)
    await insert_row(pool, "batch_inv_1900", "LG", 1900, "v5_focus")
    await insert_row(pool, "batch_alias_ob", "OB", 2025, "v4_dual")

    options = batch_script.RunOptions(
        years=[2025],
        teams=batch_script.CANONICAL_TEAMS.copy(),
        focus=[],
        only_missing=False,
        force_rebuild=True,
        quality_report=None,
        baseline=False,
        game_type="REGULAR",
        delay_seconds=0.0,
        purge_invalid_years=True,
        cleanup_legacy=True,
    )
    resolver = batch_script.TeamCodeResolver()
    cleanup = await batch_script.cleanup_cache_rows(
        pool, options, resolver, dry_run=False
    )
    assert cleanup["deleted_keys"] == ["batch_alias_ob", "batch_inv_1900"]

    report = await batch_script.summarize_results([], options, cleanup, pool, resolver)
    summary = report["summary"]
    assert summary["cache_invalid_year_count"] == 0
    assert summary["legacy_residual_total"] == 0
    assert summary["cache_invalid_year_count_all_versions"] == 1
    assert summary["legacy_residual_total_all_versions"] == 1
    rows = await fetch_cache_rows(pool)
    assert set(rows) == {"live_inv_2030", "live_alias_ob"}


@pytest.mark.asyncio
async def test_ttl_then_rebuild_keeps_previous(
    pool: psycopg_pool.AsyncConnectionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    aged_140d = now - timedelta(days=140)

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
    )
    target_keys = batch_script.build_target_batch_keys(options)
    assert len(target_keys) == 10

    # Seed 10 batch v5_focus COMPLETED rows for 2025 canonical teams aged 140 days
    for team, key in zip(batch_script.CANONICAL_TEAMS, target_keys):
        await insert_row(
            pool,
            cache_key=key,
            team_id=team,
            year=2025,
            prompt_version="v5_focus",
            status="COMPLETED",
            response_json={"headline": f"{team} aged 140d"},
            updated_at=aged_140d,
        )

    # Plus one live v90_scheduled_llm row aged 140 days and one fresh one
    live_pv = COACH_CACHE_PROMPT_VERSION
    await insert_row(
        pool,
        cache_key="live_2025_aged_140d",
        team_id="LG",
        year=2025,
        prompt_version=live_pv,
        status="COMPLETED",
        response_json={"headline": "live old"},
        updated_at=aged_140d,
    )
    await insert_row(
        pool,
        cache_key="live_2025_fresh",
        team_id="LG",
        year=2025,
        prompt_version=live_pv,
        status="COMPLETED",
        response_json={"headline": "live fresh"},
        updated_at=now,
    )

    # Run cleanup_expired_cache (retention_days=7, years=[2025], non-dry-run)
    monkeypatch.setattr(cleanup_script, "get_connection_pool", lambda: pool)
    monkeypatch.setattr(cleanup_script, "close_connection_pool", AsyncMock())

    ttl_res = await cleanup_script.cleanup_expired_cache(
        retention_days=7,
        years=[2025],
        dry_run=False,
    )

    # Assert: 10 batch rows survive, old live row is deleted, fresh live survives
    rows_after_ttl = await fetch_cache_rows(pool)
    for key in target_keys:
        assert key in rows_after_ttl
    assert "live_2025_aged_140d" not in rows_after_ttl
    assert "live_2025_fresh" in rows_after_ttl
    assert ttl_res["deleted"] == 1
    assert ttl_res["deleted_keys"] == ["live_2025_aged_140d"]

    # Stub LLM / DB tools to fail
    class FakeDatabaseQueryTool:
        def __init__(self, conn=None) -> None:
            pass

        async def get_team_summary(self, team_id: str, year: int) -> dict[str, Any]:
            return {"found": True, "team_name": team_id}

        async def get_team_advanced_metrics(
            self, team_id: str, year: int
        ) -> dict[str, Any]:
            return {}

        async def get_team_recent_form(self, team_id: str, year: int) -> dict[str, Any]:
            return {}

    async def _failing_llm(messages):
        if False:
            yield ""
        raise RuntimeError("llm rebuild failure")

    monkeypatch.setattr(batch_script, "DatabaseQueryTool", FakeDatabaseQueryTool)
    monkeypatch.setattr(batch_script, "get_coach_llm_generator", lambda: _failing_llm)

    # Run generate_and_cache_team(force_rebuild=True, skip_completed=False) for 3 teams
    test_teams = ["LG", "SSG", "KIA"]
    for team in test_teams:
        res = await batch_script.generate_and_cache_team(
            pool=pool,
            team_id=team,
            year=2025,
            game_type="REGULAR",
            focus=[],
            skip_completed=False,
            only_missing=False,
            force_rebuild=True,
        )
        assert res["status"] == "failed"
        assert res["previous_kept"] is True

    # Assert old response_json is unchanged
    rows_after_rebuild = await fetch_cache_rows(pool)
    for team in test_teams:
        team_key = batch_script.build_target_batch_keys(
            batch_script.RunOptions(
                years=[2025],
                teams=[team],
                focus=[],
                only_missing=False,
                force_rebuild=True,
                quality_report=None,
                baseline=False,
                game_type="REGULAR",
                delay_seconds=0.0,
            )
        )[0]
        status, resp_json = rows_after_rebuild[team_key]
        assert status == "COMPLETED"
        assert resp_json == {"headline": f"{team} aged 140d"}

    # Assert that with exclude_prompt_versions=[] the same TTL call deletes them
    ttl_unexcluded = await cleanup_script.cleanup_expired_cache(
        retention_days=7,
        years=[2025],
        dry_run=False,
        exclude_prompt_versions=[],
    )
    rows_final = await fetch_cache_rows(pool)
    for key in target_keys:
        assert key not in rows_final
    assert "live_2025_fresh" in rows_final
    assert ttl_unexcluded["deleted"] == 10


@pytest.mark.asyncio
async def test_cleanup_without_flag_deletes_nothing(
    pool: psycopg_pool.AsyncConnectionPool,
) -> None:
    now = datetime.now(timezone.utc)
    await insert_row(
        pool,
        cache_key="legacy_v3_flag_test",
        team_id="LG",
        year=2025,
        prompt_version="v3_prompt",
        status="COMPLETED",
        updated_at=now,
    )
    await insert_row(
        pool,
        cache_key="legacy_v4_flag_test",
        team_id="KIA",
        year=2025,
        prompt_version="v4_dual",
        status="COMPLETED",
        updated_at=now,
    )

    resolver = batch_script.TeamCodeResolver()

    # cleanup_legacy=False non-dry-run deletes nothing
    options_no_legacy = batch_script.RunOptions(
        years=[2025],
        teams=batch_script.CANONICAL_TEAMS.copy(),
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
    res_no_legacy = await batch_script.cleanup_cache_rows(
        pool, options_no_legacy, resolver, dry_run=False
    )
    assert res_no_legacy["deleted_total"] == 0
    assert res_no_legacy["would_delete_keys"] == []
    assert sorted(res_no_legacy["cleanup_candidate_keys"]) == [
        "legacy_v3_flag_test",
        "legacy_v4_flag_test",
    ]

    rows = await fetch_cache_rows(pool)
    assert "legacy_v3_flag_test" in rows
    assert "legacy_v4_flag_test" in rows

    # cleanup_legacy=True deletes them
    options_legacy = batch_script.RunOptions(
        years=[2025],
        teams=batch_script.CANONICAL_TEAMS.copy(),
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
    res_legacy = await batch_script.cleanup_cache_rows(
        pool, options_legacy, resolver, dry_run=False
    )
    assert res_legacy["deleted_total"] == 2
    assert sorted(res_legacy["deleted_keys"]) == [
        "legacy_v3_flag_test",
        "legacy_v4_flag_test",
    ]

    rows_after = await fetch_cache_rows(pool)
    assert "legacy_v3_flag_test" not in rows_after
    assert "legacy_v4_flag_test" not in rows_after


@pytest.mark.asyncio
async def test_stale_pending_cleanup(
    pool: psycopg_pool.AsyncConnectionPool,
) -> None:
    now = datetime.now(timezone.utc)
    live_pv = COACH_CACHE_PROMPT_VERSION

    # Batch-owned PENDING row aged 5h (deleted with cleanup_legacy)
    await insert_row(
        pool,
        cache_key="stale_batch_pending_5h",
        team_id="LG",
        year=2025,
        prompt_version="v5_focus",
        status="PENDING",
        response_json=None,
        updated_at=now - timedelta(hours=5),
    )
    # PENDING row aged 10 minutes (kept)
    await insert_row(
        pool,
        cache_key="fresh_batch_pending_10m",
        team_id="SSG",
        year=2025,
        prompt_version="v5_focus",
        status="PENDING",
        response_json=None,
        updated_at=now - timedelta(minutes=10),
    )
    # Live-version PENDING aged 5h (kept)
    await insert_row(
        pool,
        cache_key="stale_live_pending_5h",
        team_id="LG",
        year=2025,
        prompt_version=live_pv,
        status="PENDING",
        response_json=None,
        updated_at=now - timedelta(hours=5),
    )

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
        dry_run=False,
        cleanup_legacy=True,
    )
    resolver = batch_script.TeamCodeResolver()

    result = await batch_script.cleanup_cache_rows(
        pool, options, resolver, dry_run=False
    )

    assert result["stale_pending_rows"] == 1
    assert result["deleted_keys"] == ["stale_batch_pending_5h"]

    rows = await fetch_cache_rows(pool)
    assert "stale_batch_pending_5h" not in rows
    assert "fresh_batch_pending_10m" in rows
    assert "stale_live_pending_5h" in rows


@pytest.mark.asyncio
async def test_plan_lists_new_generation_keys(
    pool: psycopg_pool.AsyncConnectionPool,
) -> None:
    options = batch_script.RunOptions(
        years=[2025],
        teams=batch_script.CANONICAL_TEAMS.copy(),
        focus=[],
        only_missing=False,
        force_rebuild=True,
        quality_report=None,
        baseline=False,
        game_type="REGULAR",
        delay_seconds=0.0,
        dry_run=True,
    )
    target_keys = batch_script.build_target_batch_keys(options)
    assert len(target_keys) == 10

    # Seed 7 of 10 target rows as COMPLETED
    completed_teams = batch_script.CANONICAL_TEAMS[:7]
    completed_keys = target_keys[:7]
    missing_keys = target_keys[7:]

    for team, key in zip(completed_teams, completed_keys):
        await insert_row(
            pool,
            cache_key=key,
            team_id=team,
            year=2025,
            prompt_version="v5_focus",
            status="COMPLETED",
            response_json={"headline": f"{team} ready"},
        )

    resolver = batch_script.TeamCodeResolver()
    result = await batch_script.cleanup_cache_rows(
        pool, options, resolver, dry_run=True
    )

    assert result["force_rebuild_rows"] == 7
    assert sorted(result["force_rebuild_replace_keys"]) == sorted(completed_keys)
    assert sorted(result["new_generation_keys"]) == sorted(missing_keys)
    assert len(result["new_generation_keys"]) == 3
