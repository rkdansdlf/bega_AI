from pathlib import Path


def _workflow_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / ".github" / "workflows" / name


def test_manual_cache_cleanup_is_scoped_and_uses_canonical_database_secret() -> None:
    workflow = _workflow_path("cache-cleanup.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "\n  schedule:" not in workflow
    assert "COACH_QUALITY_DB_URL_STAGING_RW" in workflow
    assert "POSTGRES_DB_URL:" in workflow
    assert "SUPABASE_DB_URL" not in workflow
    assert "--allow-global" not in workflow
    assert "--years" in workflow
    assert "--teams" in workflow
    assert "Provide at least one of workflow inputs: years or teams" in workflow
    assert 'default: "14"' in workflow
    assert "Expired cache entries (>14 days) cleaned up." not in workflow


def test_quality_workflow_validates_secrets_before_running_cache_operations() -> None:
    workflow = _workflow_path("coach-quality-2025-weekly.yml").read_text(
        encoding="utf-8"
    )

    assert "group: coach-cache-operations" in workflow
    assert "COACH_QUALITY_DB_URL_STAGING_RW" in workflow
    assert "OPENROUTER_API_KEY_STAGING" in workflow
    assert workflow.index("Validate required secrets") < workflow.index("Set up Python")
    assert "--years 2025" in workflow
    assert "--days 7" in workflow


def test_quality_workflow_has_batch_plan_dryrun_before_execute() -> None:
    workflow = _workflow_path("coach-quality-2025-weekly.yml").read_text(
        encoding="utf-8"
    )

    assert "Batch cleanup/rebuild plan dry-run" in workflow
    assert "--dry-run" in workflow
    assert workflow.index("Batch cleanup/rebuild plan dry-run") < workflow.index(
        "Cleanup cache execute"
    )


def test_warmup_workflow_uses_staging_only() -> None:
    workflow = _workflow_path("coach-warmup.yml").read_text(encoding="utf-8")

    assert "SUPABASE_DB_URL" not in workflow
    assert "COACH_QUALITY_DB_URL_STAGING_RW" in workflow
    assert "WARMUP-STAGING" in workflow
    assert "workflow_dispatch:" in workflow
    assert "\n  schedule:" not in workflow
    assert "--only-missing" in workflow


def test_weekly_workflow_cleanup_legacy_and_gate_condition() -> None:
    workflow = _workflow_path("coach-quality-2025-weekly.yml").read_text(
        encoding="utf-8"
    )

    # --cleanup-legacy passed in both plan dry-run and rebuild steps
    assert "Batch cleanup/rebuild plan dry-run" in workflow
    assert "Rebuild coach cache for 2025 regular season" in workflow
    assert "id: rebuild" in workflow
    assert workflow.count("--cleanup-legacy") >= 2

    dryrun_idx = workflow.index("Batch cleanup/rebuild plan dry-run")
    rebuild_idx = workflow.index("Rebuild coach cache for 2025 regular season")
    gate_idx = workflow.index("Evaluate coach quality gate")

    dryrun_block = workflow[dryrun_idx:rebuild_idx]
    rebuild_block = workflow[rebuild_idx:gate_idx]
    assert "--cleanup-legacy" in dryrun_block
    assert "--cleanup-legacy" in rebuild_block

    # Gate step condition
    assert "if: ${{ !cancelled() && steps.rebuild.outcome != 'skipped' }}" in workflow


def test_warmup_workflow_no_cleanup_legacy_and_dynamic_summary() -> None:
    workflow = _workflow_path("coach-warmup.yml").read_text(encoding="utf-8")

    assert "--cleanup-legacy" not in workflow
    assert "completed for all 10" not in workflow
    assert "Pre-warming completed for all 10 KBO teams." not in workflow
    assert "if: always()" in workflow
    assert "/tmp/coach_warmup_report.json" in workflow


def test_ci_workflow_has_postgres_service_and_test_db_url() -> None:
    workflow = _workflow_path("ci.yml").read_text(encoding="utf-8")

    assert "image: postgres:17" in workflow
    assert "55432:5432" in workflow
    assert "COACH_CACHE_TEST_DB_URL:" in workflow
    assert "coach_cache_test" in workflow
