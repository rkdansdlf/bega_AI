#!/usr/bin/env python3
"""
Evaluate Coach quality report JSON files and enforce gate conditions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

DEFAULT_THRESHOLDS = {
    "coverage_min": 1.0,
    "warning_rate_max": 0.15,
    "critical_over_limit_rate_max": 0.05,
    "drift_rate_max": 0.02,
    "validator_fail_max": 0,
    "cache_invalid_year_max": 0,
    "legacy_residual_max": 0,
}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_years_csv(raw: Optional[str]) -> Optional[Set[int]]:
    if raw is None:
        return None
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not tokens:
        return set()
    return {int(token) for token in tokens}


def collect_report_paths(raw_inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in raw_inputs:
        item = Path(raw)
        if item.is_file():
            paths.append(item)
            continue
        if item.is_dir():
            paths.extend(sorted(path for path in item.glob("*.json") if path.is_file()))
            continue
        raise FileNotFoundError(f"input path not found: {item}")

    deduped: Dict[str, Path] = {}
    for path in paths:
        deduped[str(path.resolve())] = path
    final_paths = list(deduped.values())
    if not final_paths:
        raise ValueError("no quality report json files found")
    return final_paths


def load_reports(paths: List[Path]) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"report root must be object: {path}")
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"report.summary must be object: {path}")
        reports.append(
            {
                "path": str(path),
                "summary": summary,
                "options": payload.get("options", {}),
            }
        )
    return reports


def _extract_target_years(summary: Dict[str, Any], options: Dict[str, Any]) -> Set[int]:
    years = summary.get("target_years")
    if years is None:
        years = options.get("years")
    if years is None:
        return set()
    if isinstance(years, list):
        return {_as_int(year) for year in years if year is not None}
    if isinstance(years, str):
        parsed = parse_years_csv(years)
        return parsed or set()
    return {_as_int(years)}


def _extract_game_type(summary: Dict[str, Any], options: Dict[str, Any]) -> str:
    game_type = summary.get("game_type")
    if game_type is None:
        game_type = options.get("game_type")
    if game_type is None:
        return ""
    return str(game_type).upper().strip()


def aggregate_metrics(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_cases = 0
    total_failed = 0
    total_success = 0
    total_generated_success = 0
    total_skipped = 0
    validator_fail_count = 0
    cache_invalid_year_count = 0
    legacy_residual_total = 0

    warning_rate_numerator = 0.0
    warning_rate_denominator = 0
    critical_rate_numerator = 0.0
    critical_rate_denominator = 0
    drift_rate_numerator = 0.0
    drift_rate_denominator = 0
    drift_reports = 0

    observed_years: Set[int] = set()
    observed_game_types: Set[str] = set()

    for entry in reports:
        summary = entry["summary"]
        options = entry.get("options", {})

        cases = _as_int(summary.get("cases"))
        failed = _as_int(summary.get("failed"))
        success = _as_int(summary.get("success"))
        generated_success = _as_int(
            summary.get("generated_success_count", summary.get("success"))
        )
        skipped = _as_int(summary.get("skipped_count", summary.get("skipped")))

        total_cases += cases
        total_failed += failed
        total_success += success
        total_generated_success += generated_success
        total_skipped += skipped

        validator_fail_count += _as_int(summary.get("validator_fail_count"))
        cache_invalid_year_count += _as_int(summary.get("cache_invalid_year_count"))
        legacy_residual_total += _as_int(summary.get("legacy_residual_total"))

        warning_rate = _as_float(summary.get("warning_rate"))
        if success > 0:
            warning_rate_numerator += warning_rate * success
            warning_rate_denominator += success

        critical_rate = _as_float(summary.get("critical_over_limit_rate"))
        if success > 0:
            critical_rate_numerator += critical_rate * success
            critical_rate_denominator += success

        if "drift_rate" in summary and summary.get("drift_rate") is not None:
            drift_rate = _as_float(summary.get("drift_rate"))
            drift_reports += 1
            weight = cases if cases > 0 else 1
            drift_rate_numerator += drift_rate * weight
            drift_rate_denominator += weight

        observed_years.update(_extract_target_years(summary, options))
        game_type = _extract_game_type(summary, options)
        if game_type:
            observed_game_types.add(game_type)

    coverage_rate = (
        round((total_cases - total_failed) / total_cases, 4) if total_cases > 0 else 0.0
    )
    warning_rate = (
        round(warning_rate_numerator / warning_rate_denominator, 4)
        if warning_rate_denominator > 0
        else 0.0
    )
    critical_over_limit_rate = (
        round(critical_rate_numerator / critical_rate_denominator, 4)
        if critical_rate_denominator > 0
        else 0.0
    )
    drift_rate = (
        round(drift_rate_numerator / drift_rate_denominator, 4)
        if drift_rate_denominator > 0
        else 0.0
    )

    return {
        "report_count": len(reports),
        "cases": total_cases,
        "success": total_success,
        "generated_success_count": total_generated_success,
        "skipped_count": total_skipped,
        "failed": total_failed,
        "coverage_rate": coverage_rate,
        "validator_fail_count": validator_fail_count,
        "cache_invalid_year_count": cache_invalid_year_count,
        "legacy_residual_total": legacy_residual_total,
        "warning_rate": warning_rate,
        "critical_over_limit_rate": critical_over_limit_rate,
        "drift_rate": drift_rate,
        "drift_reports": drift_reports,
        "observed_target_years": sorted(observed_years),
        "observed_game_types": sorted(observed_game_types),
    }


def evaluate_quality(
    metrics: Dict[str, Any],
    thresholds: Dict[str, Any],
    *,
    required_generated_success: int,
    required_years: Optional[Set[int]],
    require_game_type: Optional[str],
) -> List[str]:
    failure_codes: List[str] = []

    if _as_float(metrics.get("coverage_rate")) < _as_float(thresholds["coverage_min"]):
        failure_codes.append("coverage_fail")

    if _as_int(metrics.get("validator_fail_count")) > _as_int(
        thresholds["validator_fail_max"]
    ):
        failure_codes.append("validator_fail")

    cache_invalid_year_count = _as_int(metrics.get("cache_invalid_year_count"))
    legacy_residual_total = _as_int(metrics.get("legacy_residual_total"))
    if (
        cache_invalid_year_count > _as_int(thresholds["cache_invalid_year_max"])
        or legacy_residual_total > _as_int(thresholds["legacy_residual_max"])
    ):
        failure_codes.append("cache_integrity_fail")

    if _as_float(metrics.get("warning_rate")) > _as_float(thresholds["warning_rate_max"]):
        failure_codes.append("warning_rate_fail")

    if _as_float(metrics.get("critical_over_limit_rate")) > _as_float(
        thresholds["critical_over_limit_rate_max"]
    ):
        failure_codes.append("critical_over_limit_fail")

    drift_reports = _as_int(metrics.get("drift_reports"))
    if drift_reports > 0 and _as_float(metrics.get("drift_rate")) > _as_float(
        thresholds["drift_rate_max"]
    ):
        failure_codes.append("drift_fail")

    if _as_int(metrics.get("generated_success_count")) < required_generated_success:
        failure_codes.append("fresh_generation_fail")

    if required_years is not None:
        observed_years = set(metrics.get("observed_target_years", []))
        if observed_years != required_years:
            failure_codes.append("target_year_mismatch")

    if require_game_type:
        observed_game_types = set(metrics.get("observed_game_types", []))
        if observed_game_types != {require_game_type}:
            failure_codes.append("game_type_mismatch")

    return failure_codes


def evaluate_reports(
    reports: List[Dict[str, Any]],
    thresholds: Dict[str, Any] | None = None,
    *,
    required_generated_success: int = 0,
    required_years: Optional[Set[int]] = None,
    require_game_type: Optional[str] = None,
) -> Dict[str, Any]:
    effective_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        effective_thresholds.update(thresholds)

    normalized_require_game_type = (
        str(require_game_type).upper().strip() if require_game_type else None
    )

    metrics = aggregate_metrics(reports)
    failure_codes = evaluate_quality(
        metrics,
        effective_thresholds,
        required_generated_success=required_generated_success,
        required_years=required_years,
        require_game_type=normalized_require_game_type,
    )
    status = "PASS" if not failure_codes else "FAIL"

    return {
        "status": status,
        "failure_codes": failure_codes,
        "thresholds": effective_thresholds,
        "requirements": {
            "required_generated_success": required_generated_success,
            "required_years": sorted(required_years) if required_years is not None else None,
            "require_game_type": normalized_require_game_type,
        },
        "metrics": metrics,
        "reports": reports,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Coach quality report JSON files for gate decisions."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more report json files or directories containing report json files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path.",
    )
    parser.add_argument("--coverage-min", type=float, default=DEFAULT_THRESHOLDS["coverage_min"])
    parser.add_argument(
        "--warning-rate-max", type=float, default=DEFAULT_THRESHOLDS["warning_rate_max"]
    )
    parser.add_argument(
        "--critical-over-limit-rate-max",
        type=float,
        default=DEFAULT_THRESHOLDS["critical_over_limit_rate_max"],
    )
    parser.add_argument("--drift-rate-max", type=float, default=DEFAULT_THRESHOLDS["drift_rate_max"])
    parser.add_argument(
        "--required-generated-success",
        type=int,
        default=0,
        help="Minimum required generated success count (default: 0).",
    )
    parser.add_argument(
        "--require-years",
        default=None,
        help="Required target years as comma-separated values (e.g. 2025).",
    )
    parser.add_argument(
        "--require-game-type",
        default=None,
        help="Required game_type (e.g. REGULAR).",
    )
    return parser


def _write_output(path: str, data: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    thresholds = {
        "coverage_min": args.coverage_min,
        "warning_rate_max": args.warning_rate_max,
        "critical_over_limit_rate_max": args.critical_over_limit_rate_max,
        "drift_rate_max": args.drift_rate_max,
    }
    required_years = parse_years_csv(args.require_years)

    try:
        paths = collect_report_paths(args.inputs)
        reports = load_reports(paths)
        output = evaluate_reports(
            reports,
            thresholds,
            required_generated_success=max(args.required_generated_success, 0),
            required_years=required_years,
            require_game_type=args.require_game_type,
        )
        print(json.dumps(output, ensure_ascii=False, indent=2))

        if args.output:
            _write_output(args.output, output)
    except Exception as exc:
        output = {
            "status": "FAIL",
            "failure_codes": [],
            "fatal_error": str(exc),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if args.output:
            _write_output(args.output, output)
        return 2

    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
