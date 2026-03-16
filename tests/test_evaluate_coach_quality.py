from scripts.evaluate_coach_quality import evaluate_reports


def _report(summary, options=None):
    return {"path": "/tmp/sample.json", "summary": summary, "options": options or {}}


def test_evaluate_reports_pass():
    reports = [
        _report(
            {
                "cases": 30,
                "success": 30,
                "failed": 0,
                "coverage_rate": 1.0,
                "validator_fail_count": 0,
                "cache_invalid_year_count": 0,
                "legacy_residual_total": 0,
                "warning_rate": 0.1,
                "critical_over_limit_rate": 0.03,
                "drift_rate": 0.01,
            }
        )
    ]
    result = evaluate_reports(reports)
    assert result["status"] == "PASS"
    assert result["failure_codes"] == []


def test_evaluate_reports_coverage_fail():
    reports = [
        _report(
            {
                "cases": 10,
                "success": 9,
                "failed": 1,
                "validator_fail_count": 0,
                "cache_invalid_year_count": 0,
                "legacy_residual_total": 0,
                "warning_rate": 0.1,
                "critical_over_limit_rate": 0.0,
            }
        )
    ]
    result = evaluate_reports(reports)
    assert "coverage_fail" in result["failure_codes"]


def test_evaluate_reports_validator_fail():
    reports = [
        _report(
            {
                "cases": 10,
                "success": 10,
                "failed": 0,
                "validator_fail_count": 2,
                "cache_invalid_year_count": 0,
                "legacy_residual_total": 0,
                "warning_rate": 0.1,
                "critical_over_limit_rate": 0.0,
            }
        )
    ]
    result = evaluate_reports(reports)
    assert "validator_fail" in result["failure_codes"]


def test_evaluate_reports_cache_integrity_fail():
    reports = [
        _report(
            {
                "cases": 10,
                "success": 10,
                "failed": 0,
                "validator_fail_count": 0,
                "cache_invalid_year_count": 1,
                "legacy_residual_total": 0,
                "warning_rate": 0.1,
                "critical_over_limit_rate": 0.0,
            }
        )
    ]
    result = evaluate_reports(reports)
    assert "cache_integrity_fail" in result["failure_codes"]


def test_evaluate_reports_warning_and_critical_fail():
    reports = [
        _report(
            {
                "cases": 20,
                "success": 20,
                "failed": 0,
                "validator_fail_count": 0,
                "cache_invalid_year_count": 0,
                "legacy_residual_total": 0,
                "warning_rate": 0.2,
                "critical_over_limit_rate": 0.1,
            }
        )
    ]
    result = evaluate_reports(reports)
    assert "warning_rate_fail" in result["failure_codes"]
    assert "critical_over_limit_fail" in result["failure_codes"]


def test_evaluate_reports_drift_fail():
    reports = [
        _report(
            {
                "cases": 20,
                "success": 20,
                "failed": 0,
                "validator_fail_count": 0,
                "cache_invalid_year_count": 0,
                "legacy_residual_total": 0,
                "warning_rate": 0.1,
                "critical_over_limit_rate": 0.0,
                "drift_rate": 0.03,
            }
        )
    ]
    result = evaluate_reports(reports)
    assert "drift_fail" in result["failure_codes"]


def test_evaluate_reports_fresh_generation_fail():
    reports = [
        _report(
            {
                "cases": 10,
                "success": 0,
                "generated_success_count": 0,
                "skipped_count": 10,
                "failed": 0,
                "validator_fail_count": 0,
                "cache_invalid_year_count": 0,
                "legacy_residual_total": 0,
                "warning_rate": 0.0,
                "critical_over_limit_rate": 0.0,
                "target_years": [2025],
                "game_type": "REGULAR",
            }
        )
    ]
    result = evaluate_reports(reports, required_generated_success=10)
    assert "fresh_generation_fail" in result["failure_codes"]


def test_evaluate_reports_target_year_mismatch():
    reports = [
        _report(
            {
                "cases": 10,
                "success": 10,
                "generated_success_count": 10,
                "failed": 0,
                "validator_fail_count": 0,
                "cache_invalid_year_count": 0,
                "legacy_residual_total": 0,
                "warning_rate": 0.0,
                "critical_over_limit_rate": 0.0,
                "target_years": [2024],
                "game_type": "REGULAR",
            }
        )
    ]
    result = evaluate_reports(reports, required_years={2025})
    assert "target_year_mismatch" in result["failure_codes"]


def test_evaluate_reports_game_type_mismatch():
    reports = [
        _report(
            {
                "cases": 10,
                "success": 10,
                "generated_success_count": 10,
                "failed": 0,
                "validator_fail_count": 0,
                "cache_invalid_year_count": 0,
                "legacy_residual_total": 0,
                "warning_rate": 0.0,
                "critical_over_limit_rate": 0.0,
                "target_years": [2025],
                "game_type": "POST",
            }
        )
    ]
    result = evaluate_reports(reports, require_game_type="REGULAR")
    assert "game_type_mismatch" in result["failure_codes"]


def test_evaluate_reports_collects_focus_signatures():
    reports = [
        _report(
            {
                "cases": 10,
                "success": 10,
                "generated_success_count": 10,
                "failed": 0,
                "validator_fail_count": 0,
                "cache_invalid_year_count": 0,
                "legacy_residual_total": 0,
                "warning_rate": 0.0,
                "critical_over_limit_rate": 0.0,
                "target_years": [2025],
                "game_type": "REGULAR",
                "focus_signature": "recent_form+bullpen",
            }
        )
    ]
    result = evaluate_reports(reports)
    assert result["metrics"]["observed_focus_signatures"] == ["recent_form+bullpen"]


def test_evaluate_reports_generation_mix_failures():
    reports = [
        _report(
            {
                "cases": 10,
                "success": 10,
                "generated_success_count": 10,
                "failed": 0,
                "validator_fail_count": 0,
                "cache_invalid_year_count": 0,
                "legacy_residual_total": 0,
                "warning_rate": 0.0,
                "critical_over_limit_rate": 0.0,
                "llm_manual_rate": 0.0,
                "fallback_rate": 1.0,
                "focus_section_missing_rate": 0.3,
            }
        )
    ]

    result = evaluate_reports(reports)

    assert "llm_manual_rate_fail" in result["failure_codes"]
    assert "fallback_rate_fail" in result["failure_codes"]
    assert "focus_section_missing_rate_fail" in result["failure_codes"]


def test_evaluate_reports_accepts_completed_review_clutch_summary():
    reports = [
        _report(
            {
                "cases": 12,
                "success": 12,
                "generated_success_count": 12,
                "failed": 0,
                "validator_fail_count": 0,
                "cache_invalid_year_count": 0,
                "legacy_residual_total": 0,
                "warning_rate": 0.05,
                "critical_over_limit_rate": 0.0,
                "focus_section_missing_rate": 0.0,
                "llm_manual_rate": 0.5,
                "fallback_rate": 0.0,
                "target_years": [2025],
                "game_type": "POST",
                "focus_signature": "matchup+bullpen",
            },
            {"game_type": "POST"},
        )
    ]

    result = evaluate_reports(reports, require_game_type="POST")

    assert result["status"] == "PASS"


def test_evaluate_reports_tracks_preview_form_focus_signature():
    reports = [
        _report(
            {
                "cases": 8,
                "success": 8,
                "generated_success_count": 8,
                "failed": 0,
                "validator_fail_count": 0,
                "cache_invalid_year_count": 0,
                "legacy_residual_total": 0,
                "warning_rate": 0.0,
                "critical_over_limit_rate": 0.0,
                "focus_section_missing_rate": 0.0,
                "llm_manual_rate": 0.25,
                "fallback_rate": 0.0,
                "target_focus": ["batting"],
            },
            {"focus": ["batting"]},
        )
    ]

    result = evaluate_reports(reports)

    assert result["metrics"]["observed_focus_signatures"] == ["batting"]
