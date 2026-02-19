from scripts.verify_canonical_transition import (
    evaluate_canonical_window,
    evaluate_outside_regression,
)


def test_evaluate_canonical_window_passes_when_all_green():
    output = {
        "canonical_window": {
            "totals": {"cases": 50, "all_ok": 50},
            "legacy_residuals": {"game": 0, "player_season_batting": 0},
            "runtime_seconds": 6.2,
        }
    }
    evaluation = evaluate_canonical_window(
        output,
        strict_canonical_window=True,
        strict_legacy_residual=True,
    )
    assert evaluation["passed"] is True
    assert evaluation["failed_required_checks"] == []
    assert evaluation["legacy_residual_total"] == 0


def test_evaluate_canonical_window_fails_when_matrix_has_miss():
    output = {
        "canonical_window": {
            "totals": {"cases": 50, "all_ok": 49},
            "legacy_residuals": {"game": 0},
            "runtime_seconds": 7.0,
        }
    }
    evaluation = evaluate_canonical_window(
        output,
        strict_canonical_window=True,
        strict_legacy_residual=True,
    )
    assert evaluation["passed"] is False
    assert "canonical_window_all_cases_ok" in evaluation["failed_required_checks"]


def test_evaluate_canonical_window_honors_non_strict_legacy():
    output = {
        "canonical_window": {
            "totals": {"cases": 10, "all_ok": 10},
            "legacy_residuals": {"game": 12},
            "runtime_seconds": 1.0,
        }
    }
    evaluation = evaluate_canonical_window(
        output,
        strict_canonical_window=True,
        strict_legacy_residual=False,
    )
    assert evaluation["passed"] is True
    assert "legacy_residual_total_zero" in evaluation["failed_optional_checks"]


def test_evaluate_outside_regression_default_report_only():
    output = {
        "outside_regression": {
            "total_cases": 360,
            "additional_miss_count": 3,
            "error_diff_count": 1,
            "runtime_seconds": 4.0,
        }
    }
    evaluation = evaluate_outside_regression(
        output,
        strict_outside_regression=False,
    )
    assert evaluation["passed"] is True
    assert evaluation["failed_required_checks"] == []
    assert "outside_additional_miss_zero" in evaluation["failed_optional_checks"]


def test_evaluate_outside_regression_strict_mode_fails():
    output = {
        "outside_regression": {
            "total_cases": 360,
            "additional_miss_count": 1,
            "error_diff_count": 0,
            "runtime_seconds": 4.0,
        }
    }
    evaluation = evaluate_outside_regression(
        output,
        strict_outside_regression=True,
    )
    assert evaluation["passed"] is False
    assert "outside_additional_miss_zero" in evaluation["failed_required_checks"]
