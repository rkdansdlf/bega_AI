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
