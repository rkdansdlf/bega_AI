#!/usr/bin/env python3
"""Evaluate candidate smoke summary against a fixed baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
PRESET_BASELINES = {
    "regmix_100": REPORTS_DIR / "smoke_latency_baseline_regmix_100.json",
    "regulations_20": REPORTS_DIR / "smoke_latency_baseline_regulations_20.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare candidate smoke summary to baseline metrics."
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline JSON path.",
    )
    parser.add_argument(
        "--baseline-preset",
        choices=sorted(PRESET_BASELINES),
        default=None,
        help="Use canonical baseline JSON for a named preset.",
    )
    parser.add_argument(
        "--candidate", required=True, help="Candidate summary JSON path."
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    parser.add_argument(
        "--p95-improve-threshold",
        type=float,
        default=0.05,
        help="Minimum p95 improvement ratio to mark as improvement candidate (default: 0.05).",
    )
    parser.add_argument(
        "--p95-regression-threshold",
        type=float,
        default=0.05,
        help="Maximum allowed p95 regression ratio (default: 0.05).",
    )
    parser.add_argument(
        "--error-rate-increase-threshold",
        type=float,
        default=0.005,
        help="Maximum allowed error rate increase (default: 0.005 == 0.5%%p).",
    )
    parser.add_argument(
        "--timeout-rate-increase-threshold",
        type=float,
        default=0.005,
        help="Maximum allowed timeout rate increase (default: 0.005 == 0.5%%p).",
    )
    parser.add_argument("--baseline-memory-mb", type=float, default=None)
    parser.add_argument("--candidate-memory-mb", type=float, default=None)
    parser.add_argument(
        "--memory-increase-threshold",
        type=float,
        default=0.10,
        help="Maximum allowed memory increase ratio (default: 0.10 == 10%%).",
    )
    return parser.parse_args()


def _load_json(path_str: str) -> Dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def _resolve_baseline_path(args: argparse.Namespace) -> str:
    if args.baseline:
        return args.baseline
    if args.baseline_preset:
        return str(PRESET_BASELINES[args.baseline_preset])
    raise ValueError("Either --baseline or --baseline-preset must be provided.")


def _safe_ratio_delta(
    candidate: Optional[float], baseline: Optional[float]
) -> Optional[float]:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return (candidate - baseline) / baseline


def _extract_baseline_endpoint(
    baseline: Dict[str, Any], endpoint: str
) -> Dict[str, Optional[float]]:
    base = baseline.get("metrics", {}).get(endpoint, {})
    latency = base.get("latency_ms", {})
    return {
        "p95": latency.get("p95"),
        "error_rate": base.get("error_rate"),
        "timeout_rate": base.get("timeout_rate"),
    }


def _extract_candidate_endpoint(
    candidate: Dict[str, Any], endpoint: str
) -> Dict[str, Optional[float]]:
    endpoint_key = f"{endpoint}_metrics"
    base = candidate.get("summary", {}).get(endpoint_key, {})
    latency = base.get("latency_ms", {})
    return {
        "p95": latency.get("p95"),
        "error_rate": base.get("error_rate"),
        "timeout_rate": base.get("timeout_rate"),
    }


def evaluate_candidate(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    p95_improve_threshold: float,
    p95_regression_threshold: float,
    error_rate_increase_threshold: float,
    timeout_rate_increase_threshold: float,
    baseline_memory_mb: Optional[float],
    candidate_memory_mb: Optional[float],
    memory_increase_threshold: float,
) -> Dict[str, Any]:
    endpoints = ("completion", "stream")
    failures = []
    warnings = []
    endpoint_results: Dict[str, Any] = {}
    improvement_flags = []

    for endpoint in endpoints:
        baseline_metrics = _extract_baseline_endpoint(baseline, endpoint)
        candidate_metrics = _extract_candidate_endpoint(candidate, endpoint)
        p95_delta_ratio = _safe_ratio_delta(
            candidate_metrics["p95"], baseline_metrics["p95"]
        )
        error_rate_delta = None
        timeout_rate_delta = None
        if (
            baseline_metrics["error_rate"] is not None
            and candidate_metrics["error_rate"] is not None
        ):
            error_rate_delta = (
                candidate_metrics["error_rate"] - baseline_metrics["error_rate"]
            )
        if (
            baseline_metrics["timeout_rate"] is not None
            and candidate_metrics["timeout_rate"] is not None
        ):
            timeout_rate_delta = (
                candidate_metrics["timeout_rate"] - baseline_metrics["timeout_rate"]
            )

        if p95_delta_ratio is None:
            warnings.append(f"{endpoint}:missing_p95_for_comparison")
        elif p95_delta_ratio > p95_regression_threshold:
            failures.append(f"{endpoint}:p95_regression")

        if (
            error_rate_delta is not None
            and error_rate_delta > error_rate_increase_threshold
        ):
            failures.append(f"{endpoint}:error_rate_increase")
        if (
            timeout_rate_delta is not None
            and timeout_rate_delta > timeout_rate_increase_threshold
        ):
            failures.append(f"{endpoint}:timeout_rate_increase")

        improved = p95_delta_ratio is not None and p95_delta_ratio <= (
            -1.0 * p95_improve_threshold
        )
        improvement_flags.append(improved)
        endpoint_results[endpoint] = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": {
                "p95_ratio": p95_delta_ratio,
                "error_rate": error_rate_delta,
                "timeout_rate": timeout_rate_delta,
            },
            "improved": improved,
        }

    memory_result = {
        "baseline_memory_mb": baseline_memory_mb,
        "candidate_memory_mb": candidate_memory_mb,
        "delta_ratio": _safe_ratio_delta(candidate_memory_mb, baseline_memory_mb),
    }
    if (
        memory_result["delta_ratio"] is not None
        and memory_result["delta_ratio"] > memory_increase_threshold
    ):
        failures.append("memory:increase")

    if failures:
        status = "FAIL"
    elif all(improvement_flags):
        status = "PASS_IMPROVED"
    else:
        status = "PASS_NO_REGRESSION"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "failure_codes": failures,
        "warnings": warnings,
        "thresholds": {
            "p95_improve_threshold": p95_improve_threshold,
            "p95_regression_threshold": p95_regression_threshold,
            "error_rate_increase_threshold": error_rate_increase_threshold,
            "timeout_rate_increase_threshold": timeout_rate_increase_threshold,
            "memory_increase_threshold": memory_increase_threshold,
        },
        "endpoints": endpoint_results,
        "memory": memory_result,
    }


def main() -> int:
    args = parse_args()
    baseline = _load_json(_resolve_baseline_path(args))
    candidate = _load_json(args.candidate)
    report = evaluate_candidate(
        baseline,
        candidate,
        p95_improve_threshold=args.p95_improve_threshold,
        p95_regression_threshold=args.p95_regression_threshold,
        error_rate_increase_threshold=args.error_rate_increase_threshold,
        timeout_rate_increase_threshold=args.timeout_rate_increase_threshold,
        baseline_memory_mb=args.baseline_memory_mb,
        candidate_memory_mb=args.candidate_memory_mb,
        memory_increase_threshold=args.memory_increase_threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"report saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
