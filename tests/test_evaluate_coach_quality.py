import json

from scripts.evaluate_coach_quality import evaluate_reports, load_reports


def _report(summary, options=None, response_quality=None):
    return {
        "path": "/tmp/sample.json",
        "summary": summary,
        "options": options or {},
        "response_quality": response_quality or {},
    }


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


def test_evaluate_reports_accepts_coach_backfill_summary_schema():
    reports = [
        _report(
            {
                "total_targets": 40,
                "processed_targets": 40,
                "passed_targets": 40,
                "failed_targets": 0,
                "hard_failure_count": 0,
                "soft_warning_count": 0,
                "diagnosis_quality_distribution": {
                    "grounded": 5,
                    "partial": 35,
                },
                "response_quality_distribution": {
                    "grounded": 5,
                    "partial": 35,
                },
            },
            {
                "season_year": 2026,
                "status_bucket": "SCHEDULED",
            },
        )
    ]

    result = evaluate_reports(
        reports, required_years={2026}, require_game_type="SCHEDULED"
    )

    assert result["status"] == "PASS"
    assert result["metrics"]["cases"] == 40
    assert result["metrics"]["success"] == 40
    assert result["metrics"]["failed"] == 0
    assert result["metrics"]["coverage_rate"] == 1.0
    assert result["metrics"]["observed_target_years"] == [2026]
    assert result["metrics"]["observed_game_types"] == ["SCHEDULED"]


def test_evaluate_reports_fails_coach_backfill_hard_failures():
    reports = [
        _report(
            {
                "total_targets": 10,
                "processed_targets": 10,
                "passed_targets": 8,
                "failed_targets": 2,
                "hard_failure_count": 2,
                "soft_warning_count": 1,
            }
        )
    ]

    result = evaluate_reports(reports)

    assert "coverage_fail" in result["failure_codes"]
    assert "validator_fail" in result["failure_codes"]
    assert result["metrics"]["warning_rate"] == 0.125


def test_evaluate_reports_excludes_transport_failures_from_content_gate():
    reports = [
        _report(
            {
                "total_targets": 10,
                "processed_targets": 10,
                "passed_targets": 9,
                "failed_targets": 1,
                "hard_failure_count": 1,
                "content_passed_targets": 10,
                "content_failed_targets": 0,
                "content_hard_failure_count": 0,
                "transport_failed_targets": 1,
                "transport_failure_count": 1,
                "soft_warning_count": 0,
            }
        )
    ]

    result = evaluate_reports(reports)

    assert result["status"] == "PASS"
    assert result["metrics"]["success"] == 10
    assert result["metrics"]["failed"] == 0
    assert result["metrics"]["transport_failed_targets"] == 1
    assert result["metrics"]["transport_failure_count"] == 1


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


def test_evaluate_reports_flags_weak_grounding_quality_from_samples():
    reports = [
        _report(
            {
                "cases": 2,
                "success": 2,
                "failed": 0,
                "validator_fail_count": 0,
                "cache_invalid_year_count": 0,
                "legacy_residual_total": 0,
                "warning_rate": 0.0,
                "critical_over_limit_rate": 0.0,
            },
            response_quality={
                "checked_count": 2,
                "weak_grounding_count": 1,
                "uncertainty_gap_count": 1,
                "thin_markdown_count": 1,
            },
        )
    ]

    result = evaluate_reports(reports)

    assert "weak_grounding_fail" in result["failure_codes"]
    assert "uncertainty_gap_fail" in result["failure_codes"]
    assert "thin_markdown_fail" in result["failure_codes"]
    assert result["metrics"]["weak_grounding_rate"] == 0.5
    assert result["metrics"]["uncertainty_gap_rate"] == 0.5
    assert result["metrics"]["thin_markdown_rate"] == 0.5


def test_load_reports_reads_sidecar_response_quality_jsonl(tmp_path):
    report_path = tmp_path / "coach_backfill_summary_latest.json"
    results_path = tmp_path / "coach_backfill_results_latest.jsonl"
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total_targets": 1,
                    "passed_targets": 1,
                    "failed_targets": 0,
                    "hard_failure_count": 0,
                },
                "options": {},
            }
        ),
        encoding="utf-8",
    )
    results_path.write_text(
        json.dumps(
            {
                "response": {
                    "meta": {
                        "data_quality": "partial",
                        "grounding_reasons": ["missing_lineups"],
                        "structured_response": {
                            "headline": "짧은 분석",
                            "analysis": {
                                "summary": "라인업 발표 전입니다.",
                                "verdict": "라인업 발표 전이라 타순 판단은 보류합니다.",
                                "why_it_matters": [],
                                "uncertainty": [],
                            },
                            "detailed_markdown": "## 코치 판단\n- 라인업 발표 전입니다.",
                            "coach_note": "라인업 확인 뒤 타순 변수만 보강합니다.",
                        },
                    },
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    reports = load_reports([report_path])

    assert reports[0]["response_quality"] == {
        "checked_count": 1,
        "weak_grounding_count": 1,
        "uncertainty_gap_count": 1,
        "thin_markdown_count": 1,
    }


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
