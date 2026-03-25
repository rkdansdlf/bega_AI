from __future__ import annotations

import argparse
import json

from scripts import evaluate_smoke_candidate as esc


def _baseline(
    p95_completion: float = 100.0,
    p95_stream: float = 200.0,
    *,
    first_token_p95_stream: float = 80.0,
    p95_completion_max: float | None = None,
    p95_stream_max: float | None = None,
) -> dict:
    return {
        "metrics": {
            "overall_error_rate": 0.08,
            "overall_timeout_rate": 0.02,
            "completion": {
                "error_rate": 0.10,
                "timeout_rate": 0.02,
                "fallback_ratio": 0.0,
                "latency_ms": {
                    "p95": p95_completion,
                    "p95_max": (
                        p95_completion
                        if p95_completion_max is None
                        else p95_completion_max
                    ),
                },
                "first_token_ms": {"p95": 60.0, "p95_max": 60.0},
            },
            "stream": {
                "error_rate": 0.12,
                "timeout_rate": 0.03,
                "fallback_ratio": 0.20,
                "latency_ms": {
                    "p95": p95_stream,
                    "p95_max": p95_stream if p95_stream_max is None else p95_stream_max,
                },
                "first_token_ms": {
                    "p95": first_token_p95_stream,
                    "p95_max": first_token_p95_stream,
                },
            },
        }
    }


def _candidate(
    p95_completion: float = 90.0,
    p95_stream: float = 180.0,
    *,
    first_token_p95_stream: float = 70.0,
    llm_ratio: float | None = 0.9,
) -> dict:
    summary = {
        "overall_error_rate": 0.08,
        "overall_timeout_rate": 0.02,
        "completion_metrics": {
            "error_rate": 0.10,
            "timeout_rate": 0.02,
            "latency_ms": {"p95": p95_completion},
        },
        "completion_fallback_metrics": {
            "fallback_ratio": 0.0,
        },
        "stream_metrics": {
            "error_rate": 0.12,
            "timeout_rate": 0.03,
            "latency_ms": {"p95": p95_stream},
        },
        "stream_fallback_metrics": {
            "fallback_ratio": 0.20,
        },
        "perf_metrics": {
            "completion": {"first_token_ms": {"p95": 55.0}},
            "stream": {"first_token_ms": {"p95": first_token_p95_stream}},
        },
    }
    if llm_ratio is not None:
        summary["planner_bucket_distribution"] = {
            "overall": {
                "llm": {"count": 36, "ratio": llm_ratio},
                "fast_path": {"count": 4, "ratio": 1.0 - llm_ratio},
            }
        }

    return {
        "summary": summary,
    }


def test_evaluate_candidate_pass_improved() -> None:
    report = esc.evaluate_candidate(
        _baseline(),
        _candidate(),
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=1000.0,
        candidate_memory_mb=1050.0,
        memory_increase_threshold=0.10,
    )
    assert report["status"] == "PASS_IMPROVED"
    assert report["failure_codes"] == []


def test_evaluate_candidate_fail_p95_regression() -> None:
    report = esc.evaluate_candidate(
        _baseline(),
        _candidate(p95_completion=120.0, p95_stream=180.0),
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=None,
        candidate_memory_mb=None,
        memory_increase_threshold=0.10,
    )
    assert report["status"] == "FAIL"
    assert "completion:p95_regression" in report["failure_codes"]


def test_evaluate_candidate_uses_p95_max_as_regression_anchor() -> None:
    report = esc.evaluate_candidate(
        _baseline(
            p95_completion=100.0,
            p95_stream=200.0,
            first_token_p95_stream=80.0,
            p95_completion_max=120.0,
            p95_stream_max=220.0,
        ),
        _candidate(
            p95_completion=123.0,
            p95_stream=225.0,
            first_token_p95_stream=82.0,
        ),
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=None,
        candidate_memory_mb=None,
        memory_increase_threshold=0.10,
    )

    assert report["status"] == "PASS_NO_REGRESSION"
    assert report["failure_codes"] == []
    assert report["endpoints"]["completion"]["baseline"]["p95_regression_anchor"] == 120.0


def test_evaluate_candidate_fail_error_rate_increase() -> None:
    candidate = _candidate()
    candidate["summary"]["completion_metrics"]["error_rate"] = 0.11
    report = esc.evaluate_candidate(
        _baseline(),
        candidate,
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=None,
        candidate_memory_mb=None,
        memory_increase_threshold=0.10,
    )
    assert report["status"] == "FAIL"
    assert "completion:error_rate_increase" in report["failure_codes"]


def test_evaluate_candidate_fail_stream_ttfe_regression() -> None:
    report = esc.evaluate_candidate(
        _baseline(first_token_p95_stream=80.0),
        _candidate(first_token_p95_stream=90.0),
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=None,
        candidate_memory_mb=None,
        memory_increase_threshold=0.10,
    )

    assert report["status"] == "FAIL"
    assert "stream:first_token_p95_regression" in report["failure_codes"]


def test_evaluate_candidate_pass_improved_when_stream_ttfe_improves() -> None:
    report = esc.evaluate_candidate(
        _baseline(first_token_p95_stream=100.0),
        _candidate(
            p95_completion=99.0,
            p95_stream=198.0,
            first_token_p95_stream=85.0,
        ),
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=None,
        candidate_memory_mb=None,
        memory_increase_threshold=0.10,
    )

    assert report["status"] == "PASS_IMPROVED"
    assert report["failure_codes"] == []


def test_evaluate_candidate_fail_when_overall_error_rate_increases() -> None:
    candidate = _candidate()
    candidate["summary"]["overall_error_rate"] = 0.081

    report = esc.evaluate_candidate(
        _baseline(),
        candidate,
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=None,
        candidate_memory_mb=None,
        memory_increase_threshold=0.10,
    )

    assert report["status"] == "FAIL"
    assert "overall:error_rate_increase" in report["failure_codes"]


def test_evaluate_candidate_fail_when_stream_fallback_ratio_increases() -> None:
    candidate = _candidate()
    candidate["summary"]["stream_fallback_metrics"]["fallback_ratio"] = 0.25

    report = esc.evaluate_candidate(
        _baseline(),
        candidate,
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=None,
        candidate_memory_mb=None,
        memory_increase_threshold=0.10,
    )

    assert report["status"] == "FAIL"
    assert "stream:fallback_ratio_increase" in report["failure_codes"]


def test_evaluate_candidate_fail_memory_increase() -> None:
    report = esc.evaluate_candidate(
        _baseline(),
        _candidate(),
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=1000.0,
        candidate_memory_mb=1200.0,
        memory_increase_threshold=0.10,
    )
    assert report["status"] == "FAIL"
    assert "memory:increase" in report["failure_codes"]


def test_resolve_baseline_path_for_llm_canary_preset() -> None:
    args = argparse.Namespace(baseline=None, baseline_preset="llm_canary_20")

    assert esc._resolve_baseline_path(args).endswith(
        "smoke_latency_baseline_llm_canary_20.json"
    )


def test_evaluate_candidate_enforces_llm_ratio_gate() -> None:
    report = esc.evaluate_candidate(
        _baseline(),
        _candidate(llm_ratio=0.9),
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=None,
        candidate_memory_mb=None,
        memory_increase_threshold=0.10,
        llm_ratio_min=0.85,
    )

    assert report["status"] == "PASS_IMPROVED"
    assert report["planner_checks"]["llm_ratio"] == 0.9


def test_evaluate_candidate_fails_when_llm_ratio_missing() -> None:
    report = esc.evaluate_candidate(
        _baseline(),
        _candidate(llm_ratio=None),
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=None,
        candidate_memory_mb=None,
        memory_increase_threshold=0.10,
        llm_ratio_min=0.85,
    )

    assert report["status"] == "FAIL"
    assert "planner_llm_ratio:missing" in report["failure_codes"]


def test_evaluate_candidate_fails_when_llm_ratio_below_threshold() -> None:
    report = esc.evaluate_candidate(
        _baseline(),
        _candidate(llm_ratio=0.7),
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=None,
        candidate_memory_mb=None,
        memory_increase_threshold=0.10,
        llm_ratio_min=0.85,
    )

    assert report["status"] == "FAIL"
    assert "planner_llm_ratio:below_minimum" in report["failure_codes"]


def test_main_returns_nonzero_for_fail_report(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(_baseline()), encoding="utf-8")
    candidate_path.write_text(
        json.dumps(_candidate(p95_completion=120.0, p95_stream=180.0)),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        baseline=str(baseline_path),
        baseline_preset=None,
        candidate=str(candidate_path),
        output=None,
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=None,
        candidate_memory_mb=None,
        memory_increase_threshold=0.10,
        llm_ratio_min=None,
    )

    original_parse_args = esc.parse_args
    esc.parse_args = lambda: args
    try:
        exit_code = esc.main()
    finally:
        esc.parse_args = original_parse_args

    assert exit_code == 1


def test_main_returns_zero_for_pass_report(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(_baseline()), encoding="utf-8")
    candidate_path.write_text(json.dumps(_candidate()), encoding="utf-8")

    args = argparse.Namespace(
        baseline=str(baseline_path),
        baseline_preset=None,
        candidate=str(candidate_path),
        output=None,
        p95_improve_threshold=0.05,
        p95_regression_threshold=0.05,
        error_rate_increase_threshold=0.005,
        timeout_rate_increase_threshold=0.005,
        baseline_memory_mb=None,
        candidate_memory_mb=None,
        memory_increase_threshold=0.10,
        llm_ratio_min=None,
    )

    original_parse_args = esc.parse_args
    esc.parse_args = lambda: args
    try:
        exit_code = esc.main()
    finally:
        esc.parse_args = original_parse_args

    assert exit_code == 0
