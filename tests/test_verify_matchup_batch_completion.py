from typing import Any, Dict

from scripts import verify_matchup_batch_completion as vm


def test_matchup_completion_pass(monkeypatch: Any) -> None:
    report: Dict[str, Any] = {
        "summary": {
            "cases": 1,
            "success": 1,
            "generated": 1,
            "generated_success_count": 1,
            "skipped": 0,
            "skipped_count": 0,
            "failed": 0,
            "in_progress": 0,
            "warning_rate": 0.0,
            "critical_over_limit_rate": 0.0,
            "cache_invalid_year_count": 0,
            "legacy_residual_total": 0,
            "coverage_rate": 1.0,
            "target_years": [2025],
            "game_type": "REGULAR",
        },
        "options": {
            "years": [2025],
        },
        "details": [
            {
                "cache_key": "k1",
                "home_team_id": "SSG",
                "away_team_id": "HH",
                "game_type": "REGULAR",
                "status": "generated",
            }
        ],
    }

    monkeypatch.setattr(
        vm,
        "_collect_db_rows",
        lambda keys: {
            "k1": {
                "team_id": "SSG",
                "year": 2025,
                "status": "COMPLETED",
                "prompt_version": "v5_focus",
                "error_message": "",
            }
        },
    )

    result = vm._check_report(
        report,
        required_generated_success=1,
        required_years={2025},
        required_game_type="REGULAR",
        required_prompt_version="v5_focus",
        strict_game_type_check=True,
    )

    assert result["status"] == "PASS"


def test_matchup_completion_fail_on_missing_done_event(monkeypatch: Any) -> None:
    report: Dict[str, Any] = {
        "summary": {
            "cases": 1,
            "success": 0,
            "generated": 0,
            "generated_success_count": 0,
            "skipped": 0,
            "skipped_count": 0,
            "failed": 1,
            "in_progress": 0,
            "warning_rate": 0.0,
            "critical_over_limit_rate": 0.0,
            "cache_invalid_year_count": 0,
            "legacy_residual_total": 0,
            "coverage_rate": 0.0,
            "target_years": [2025],
            "game_type": "REGULAR",
        },
        "options": {"years": [2025]},
        "details": [
            {
                "cache_key": "k1",
                "home_team_id": "SSG",
                "away_team_id": "HH",
                "game_type": "REGULAR",
                "status": "failed",
                "reason": "missing_done_event",
            }
        ],
    }

    monkeypatch.setattr(vm, "_collect_db_rows", lambda keys: {})
    result = vm._check_report(
        report,
        required_generated_success=1,
        required_years={2025},
        required_game_type="REGULAR",
        required_prompt_version="v5_focus",
        strict_game_type_check=True,
    )

    assert result["status"] == "FAIL"
    assert "runtime:missing_done_event" in result["failure_codes"]
