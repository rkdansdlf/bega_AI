from __future__ import annotations

from scripts import evaluate_smoke_candidate as esc


def _baseline(p95_completion: float = 100.0, p95_stream: float = 200.0) -> dict:
    return {
        "metrics": {
            "completion": {
                "error_rate": 0.10,
                "timeout_rate": 0.02,
                "latency_ms": {"p95": p95_completion},
            },
            "stream": {
                "error_rate": 0.12,
                "timeout_rate": 0.03,
                "latency_ms": {"p95": p95_stream},
            },
        }
    }


def _candidate(p95_completion: float = 90.0, p95_stream: float = 180.0) -> dict:
    return {
        "summary": {
            "completion_metrics": {
                "error_rate": 0.10,
                "timeout_rate": 0.02,
                "latency_ms": {"p95": p95_completion},
            },
            "stream_metrics": {
                "error_rate": 0.12,
                "timeout_rate": 0.03,
                "latency_ms": {"p95": p95_stream},
            },
        }
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
