#!/usr/bin/env python3
"""Evaluate candidate smoke summary against a fixed baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

REPORTS_DIR = Path(
    os.environ.get(
        "REPORTS_DIR", str(Path(__file__).resolve().parents[2] / "reports")
    )
)
PRESET_BASELINES = {
    "regmix_100": REPORTS_DIR / "smoke_latency_baseline_regmix_100.json",
    "llm_canary_20": REPORTS_DIR / "smoke_latency_baseline_llm_canary_20.json",
    "regulations_20": REPORTS_DIR / "smoke_latency_baseline_regulations_20.json",
}
PRESET_PLANNER_MODE_RATIO_MIN = {
    "llm_canary_20": {
        "mode": "player_fast_path",
        "min_ratio": 0.85,
    },
}
PRESET_LLM_RATIO_MIN = {}
TTFE_IMPROVE_THRESHOLD = 0.10
TTFE_REGRESSION_THRESHOLD = 0.05
TTFE_ABSOLUTE_REGRESSION_MS = 2.0
STREAM_FIRST_MESSAGE_WARNING_THRESHOLD = 0.05
STREAM_FIRST_MESSAGE_ABSOLUTE_WARNING_MS = 5.0
LATENCY_ONLY_FAILURE_CODES = frozenset(
    {
        "completion:p95_regression",
        "stream:p95_regression",
        "stream:first_token_p95_regression",
    }
)


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
    parser.add_argument(
        "--llm-ratio-min",
        type=float,
        default=None,
        help="Optional minimum allowed overall LLM planner ratio.",
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
    first_token = base.get("first_token_ms", {})
    stream_first_message = base.get("stream_first_message_ms", {})
    p95 = latency.get("p95")
    p95_regression_anchor = latency.get("p95_max", p95)
    return {
        "p95": p95,
        "p95_regression_anchor": p95_regression_anchor,
        "first_token_p95": first_token.get("p95"),
        "first_token_p95_regression_anchor": first_token.get(
            "p95_max", first_token.get("p95")
        ),
        "stream_first_message_p95": stream_first_message.get("p95"),
        "stream_first_message_p95_regression_anchor": stream_first_message.get(
            "p95_max", stream_first_message.get("p95")
        ),
        "error_rate": base.get("error_rate"),
        "timeout_rate": base.get("timeout_rate"),
        "fallback_ratio": base.get("fallback_ratio"),
    }


def _extract_candidate_endpoint(
    candidate: Dict[str, Any], endpoint: str
) -> Dict[str, Optional[float]]:
    endpoint_key = f"{endpoint}_metrics"
    base = candidate.get("summary", {}).get(endpoint_key, {})
    latency = base.get("latency_ms", {})
    first_token = (
        candidate.get("summary", {})
        .get("perf_metrics", {})
        .get(endpoint, {})
        .get("first_token_ms", {})
    )
    stream_first_message = (
        candidate.get("summary", {})
        .get("perf_metrics", {})
        .get(endpoint, {})
        .get("stream_first_message_ms", {})
    )
    fallback_metrics = (
        candidate.get("summary", {}).get(f"{endpoint}_fallback_metrics", {})
    )
    return {
        "p95": latency.get("p95"),
        "first_token_p95": first_token.get("p95"),
        "stream_first_message_p95": stream_first_message.get("p95"),
        "error_rate": base.get("error_rate"),
        "timeout_rate": base.get("timeout_rate"),
        "fallback_ratio": fallback_metrics.get("fallback_ratio"),
    }


def _extract_baseline_overall(baseline: Dict[str, Any]) -> Dict[str, Optional[float]]:
    metrics = baseline.get("metrics", {})
    return {
        "overall_error_rate": metrics.get("overall_error_rate"),
        "overall_timeout_rate": metrics.get("overall_timeout_rate"),
    }


def _extract_candidate_overall(candidate: Dict[str, Any]) -> Dict[str, Optional[float]]:
    summary = candidate.get("summary", {})
    return {
        "overall_error_rate": summary.get("overall_error_rate"),
        "overall_timeout_rate": summary.get("overall_timeout_rate"),
    }


def _extract_candidate_llm_ratio(candidate: Dict[str, Any]) -> Optional[float]:
    overall = (
        candidate.get("summary", {})
        .get("planner_bucket_distribution", {})
        .get("overall", {})
    )
    if not isinstance(overall, dict) or not overall:
        return None

    llm_bucket = overall.get("llm")
    if not isinstance(llm_bucket, dict):
        return None

    ratio = llm_bucket.get("ratio")
    if not isinstance(ratio, (int, float)):
        return None

    return float(ratio)


def _extract_candidate_planner_mode_ratio(
    candidate: Dict[str, Any], planner_mode: str
) -> Optional[float]:
    overall = (
        candidate.get("summary", {})
        .get("planner_mode_distribution", {})
        .get("overall", {})
    )
    if not isinstance(overall, dict) or not overall:
        return None

    mode_bucket = overall.get(planner_mode)
    if not isinstance(mode_bucket, dict):
        return None

    ratio = mode_bucket.get("ratio")
    if not isinstance(ratio, (int, float)):
        return None

    return float(ratio)


def _extract_baseline_memory_mb(baseline: Dict[str, Any]) -> Optional[float]:
    memory_metrics = baseline.get("metrics", {}).get("memory_mb", {})
    if not isinstance(memory_metrics, dict):
        return None

    for key in ("p95_max", "p95", "p99_max", "avg_max", "avg"):
        value = memory_metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_candidate_memory_mb(candidate: Dict[str, Any]) -> Optional[float]:
    memory_metrics = candidate.get("summary", {}).get("memory_metrics", {})
    if not isinstance(memory_metrics, dict):
        return None
    peak_mb = memory_metrics.get("peak_mb")
    if isinstance(peak_mb, (int, float)):
        return float(peak_mb)
    return None


def is_latency_only_failure_report(report: Dict[str, Any]) -> bool:
    if report.get("status") != "FAIL":
        return False

    failures = report.get("failure_codes")
    if not isinstance(failures, list) or not failures:
        return False

    return all(code in LATENCY_ONLY_FAILURE_CODES for code in failures)


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
    llm_ratio_min: Optional[float] = None,
    planner_mode_ratio_mode: Optional[str] = None,
    planner_mode_ratio_min: Optional[float] = None,
) -> Dict[str, Any]:
    if baseline_memory_mb is None:
        baseline_memory_mb = _extract_baseline_memory_mb(baseline)
    if candidate_memory_mb is None:
        candidate_memory_mb = _extract_candidate_memory_mb(candidate)

    endpoints = ("completion", "stream")
    failures = []
    warnings = []
    endpoint_results: Dict[str, Any] = {}
    llm_ratio = _extract_candidate_llm_ratio(candidate)
    planner_mode_ratio = None
    if planner_mode_ratio_mode:
        planner_mode_ratio = _extract_candidate_planner_mode_ratio(
            candidate, planner_mode_ratio_mode
        )
    baseline_overall = _extract_baseline_overall(baseline)
    candidate_overall = _extract_candidate_overall(candidate)
    overall_result = {
        "baseline": baseline_overall,
        "candidate": candidate_overall,
        "delta": {
            "overall_error_rate": None,
            "overall_timeout_rate": None,
        },
    }
    if (
        baseline_overall["overall_error_rate"] is not None
        and candidate_overall["overall_error_rate"] is not None
    ):
        overall_result["delta"]["overall_error_rate"] = (
            candidate_overall["overall_error_rate"]
            - baseline_overall["overall_error_rate"]
        )
        if overall_result["delta"]["overall_error_rate"] > 0:
            failures.append("overall:error_rate_increase")
    if (
        baseline_overall["overall_timeout_rate"] is not None
        and candidate_overall["overall_timeout_rate"] is not None
    ):
        overall_result["delta"]["overall_timeout_rate"] = (
            candidate_overall["overall_timeout_rate"]
            - baseline_overall["overall_timeout_rate"]
        )
        if overall_result["delta"]["overall_timeout_rate"] > 0:
            failures.append("overall:timeout_rate_increase")

    stream_latency_improved = False
    stream_ttfe_improved = False

    for endpoint in endpoints:
        baseline_metrics = _extract_baseline_endpoint(baseline, endpoint)
        candidate_metrics = _extract_candidate_endpoint(candidate, endpoint)
        p95_regression_anchor = baseline_metrics.get("p95_regression_anchor")
        p95_delta_ratio = _safe_ratio_delta(
            candidate_metrics["p95"], p95_regression_anchor
        )
        first_token_regression_anchor = baseline_metrics.get(
            "first_token_p95_regression_anchor"
        )
        first_token_delta_ratio = _safe_ratio_delta(
            candidate_metrics["first_token_p95"],
            first_token_regression_anchor,
        )
        stream_first_message_regression_anchor = baseline_metrics.get(
            "stream_first_message_p95_regression_anchor"
        )
        stream_first_message_delta_ratio = _safe_ratio_delta(
            candidate_metrics["stream_first_message_p95"],
            stream_first_message_regression_anchor,
        )
        error_rate_delta = None
        timeout_rate_delta = None
        fallback_ratio_delta = None
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
        if (
            baseline_metrics["fallback_ratio"] is not None
            and candidate_metrics["fallback_ratio"] is not None
        ):
            fallback_ratio_delta = (
                candidate_metrics["fallback_ratio"] - baseline_metrics["fallback_ratio"]
            )

        if p95_delta_ratio is None:
            warnings.append(f"{endpoint}:missing_p95_for_comparison")
        elif p95_delta_ratio > p95_regression_threshold:
            failures.append(f"{endpoint}:p95_regression")

        if endpoint == "stream":
            if first_token_delta_ratio is None:
                warnings.append("stream:missing_first_token_p95_for_comparison")
            elif (
                first_token_delta_ratio > TTFE_REGRESSION_THRESHOLD
                and candidate_metrics["first_token_p95"] is not None
                and first_token_regression_anchor is not None
                and (
                    candidate_metrics["first_token_p95"]
                    - first_token_regression_anchor
                )
                > TTFE_ABSOLUTE_REGRESSION_MS
            ):
                failures.append("stream:first_token_p95_regression")
            if stream_first_message_delta_ratio is None:
                warnings.append("stream:missing_stream_first_message_p95")
            elif (
                stream_first_message_delta_ratio
                > STREAM_FIRST_MESSAGE_WARNING_THRESHOLD
                and candidate_metrics["stream_first_message_p95"] is not None
                and stream_first_message_regression_anchor is not None
                and (
                    candidate_metrics["stream_first_message_p95"]
                    - stream_first_message_regression_anchor
                )
                > STREAM_FIRST_MESSAGE_ABSOLUTE_WARNING_MS
            ):
                warnings.append("stream:stream_first_message_p95_regression")

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
        if fallback_ratio_delta is not None and fallback_ratio_delta > 0:
            failures.append(f"{endpoint}:fallback_ratio_increase")

        if endpoint == "stream":
            stream_latency_improved = p95_delta_ratio is not None and p95_delta_ratio <= (
                -1.0 * p95_improve_threshold
            )
            stream_ttfe_improved = (
                first_token_delta_ratio is not None
                and first_token_delta_ratio <= (-1.0 * TTFE_IMPROVE_THRESHOLD)
            )
        endpoint_results[endpoint] = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": {
                "p95_ratio": p95_delta_ratio,
                "first_token_p95_ratio": first_token_delta_ratio,
                "stream_first_message_p95_ratio": stream_first_message_delta_ratio,
                "error_rate": error_rate_delta,
                "timeout_rate": timeout_rate_delta,
                "fallback_ratio": fallback_ratio_delta,
            },
            "improved": (
                stream_latency_improved or stream_ttfe_improved
                if endpoint == "stream"
                else False
            ),
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

    planner_checks = {
        "llm_ratio_min": llm_ratio_min,
        "llm_ratio": llm_ratio,
        "planner_mode_ratio_mode": planner_mode_ratio_mode,
        "planner_mode_ratio_min": planner_mode_ratio_min,
        "planner_mode_ratio": planner_mode_ratio,
    }
    if planner_mode_ratio_mode == "player_fast_path":
        planner_checks["player_fast_path_ratio_min"] = planner_mode_ratio_min
        planner_checks["player_fast_path_ratio"] = planner_mode_ratio
    if llm_ratio_min is not None:
        if llm_ratio is None:
            failures.append("planner_llm_ratio:missing")
        elif llm_ratio < llm_ratio_min:
            failures.append("planner_llm_ratio:below_minimum")
    if planner_mode_ratio_min is not None and planner_mode_ratio_mode:
        failure_prefix = f"planner_{planner_mode_ratio_mode}_ratio"
        if planner_mode_ratio is None:
            failures.append(f"{failure_prefix}:missing")
        elif planner_mode_ratio < planner_mode_ratio_min:
            failures.append(f"{failure_prefix}:below_minimum")

    if failures:
        status = "FAIL"
    elif stream_latency_improved or stream_ttfe_improved:
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
            "ttfe_improve_threshold": TTFE_IMPROVE_THRESHOLD,
            "ttfe_regression_threshold": TTFE_REGRESSION_THRESHOLD,
            "ttfe_absolute_regression_ms": TTFE_ABSOLUTE_REGRESSION_MS,
            "stream_first_message_warning_threshold": STREAM_FIRST_MESSAGE_WARNING_THRESHOLD,
            "stream_first_message_absolute_warning_ms": STREAM_FIRST_MESSAGE_ABSOLUTE_WARNING_MS,
            "error_rate_increase_threshold": error_rate_increase_threshold,
            "timeout_rate_increase_threshold": timeout_rate_increase_threshold,
            "memory_increase_threshold": memory_increase_threshold,
        },
        "endpoints": endpoint_results,
        "overall": overall_result,
        "memory": memory_result,
        "planner_checks": planner_checks,
    }


def main() -> int:
    args = parse_args()
    baseline = _load_json(_resolve_baseline_path(args))
    candidate = _load_json(args.candidate)
    llm_ratio_min = args.llm_ratio_min
    if llm_ratio_min is None and args.baseline_preset:
        llm_ratio_min = PRESET_LLM_RATIO_MIN.get(args.baseline_preset)
    planner_mode_ratio_mode = None
    planner_mode_ratio_min = None
    if args.baseline_preset:
        planner_requirement = PRESET_PLANNER_MODE_RATIO_MIN.get(args.baseline_preset)
        if planner_requirement:
            planner_mode_ratio_mode = planner_requirement.get("mode")
            planner_mode_ratio_min = planner_requirement.get("min_ratio")
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
        llm_ratio_min=llm_ratio_min,
        planner_mode_ratio_mode=planner_mode_ratio_mode,
        planner_mode_ratio_min=planner_mode_ratio_min,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"report saved: {output_path}")
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
