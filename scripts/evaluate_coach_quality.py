#!/usr/bin/env python3
"""
Evaluate Coach quality report JSON files and enforce gate conditions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set

DEFAULT_THRESHOLDS = {
    "coverage_min": 1.0,
    "warning_rate_max": 0.15,
    "critical_over_limit_rate_max": 0.05,
    "drift_rate_max": 0.02,
    "llm_manual_rate_min": 0.1,
    "fallback_rate_max": 0.9,
    "focus_section_missing_rate_max": 0.2,
    "validator_fail_max": 0,
    "cache_invalid_year_max": 0,
    "legacy_residual_max": 0,
    "weak_grounding_rate_max": 0.1,
    "uncertainty_gap_rate_max": 0.05,
    "thin_markdown_rate_max": 0.1,
}
QUALITY_RESULT_FILENAMES = (
    "coach_backfill_results_latest.jsonl",
    "coach_backfill_results.jsonl",
)
MIN_MARKDOWN_SECTION_COUNT = 2


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
                "response_quality": _collect_response_quality_from_sidecar(
                    path, payload.get("options", {})
                ),
            }
        )
    return reports


def _empty_response_quality_counts() -> Dict[str, int]:
    return {
        "checked_count": 0,
        "weak_grounding_count": 0,
        "uncertainty_gap_count": 0,
        "thin_markdown_count": 0,
    }


def _merge_response_quality_counts(
    target: Dict[str, int], source: Dict[str, Any]
) -> None:
    for key in _empty_response_quality_counts():
        target[key] = target.get(key, 0) + _as_int(source.get(key))


def _extract_response_meta(entry: Dict[str, Any]) -> Dict[str, Any]:
    response = entry.get("response")
    if isinstance(response, dict) and isinstance(response.get("meta"), dict):
        return response["meta"]
    if isinstance(entry.get("meta"), dict):
        return entry["meta"]
    return {}


def _extract_structured_response(entry: Dict[str, Any]) -> Dict[str, Any]:
    structured = entry.get("structured_response")
    if isinstance(structured, dict):
        return structured

    response = entry.get("response")
    if isinstance(response, dict):
        response_structured = response.get("structured_response")
        if isinstance(response_structured, dict):
            return response_structured
        meta = response.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("structured_response"), dict):
            return meta["structured_response"]
    return {}


def _has_text_items(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(isinstance(item, str) and item.strip() for item in value)
    return False


def _markdown_section_count(markdown: str) -> int:
    return len(re.findall(r"(?m)^##\s+\S", markdown or ""))


def _score_response_quality_entry(entry: Dict[str, Any]) -> Dict[str, int]:
    counts = _empty_response_quality_counts()
    structured = _extract_structured_response(entry)
    if not structured:
        return counts

    counts["checked_count"] = 1
    analysis = structured.get("analysis")
    if not isinstance(analysis, dict):
        analysis = {}

    verdict_or_summary = _has_text_items(analysis.get("verdict")) or _has_text_items(
        analysis.get("summary")
    )
    if not verdict_or_summary or not _has_text_items(analysis.get("why_it_matters")):
        counts["weak_grounding_count"] = 1

    detailed_markdown = str(structured.get("detailed_markdown") or "")
    if not detailed_markdown.strip() or (
        _markdown_section_count(detailed_markdown) < MIN_MARKDOWN_SECTION_COUNT
    ):
        counts["thin_markdown_count"] = 1

    meta = _extract_response_meta(entry)
    grounding_reasons = meta.get("grounding_reasons")
    grounding_warnings = meta.get("grounding_warnings")
    response_notes = entry.get("response_notes")
    data_quality = str(meta.get("data_quality") or "").strip().lower()
    uncertainty_expected = (
        data_quality in {"partial", "insufficient"}
        or bool(grounding_reasons)
        or bool(grounding_warnings)
        or bool(response_notes)
    )
    if uncertainty_expected and not _has_text_items(analysis.get("uncertainty")):
        counts["uncertainty_gap_count"] = 1

    return counts


def _collect_response_quality_from_jsonl(path: Path) -> Dict[str, int]:
    counts = _empty_response_quality_counts()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                _merge_response_quality_counts(
                    counts, _score_response_quality_entry(entry)
                )
    return counts


def _candidate_result_paths(report_path: Path, options: Dict[str, Any]) -> List[Path]:
    directories: List[Path] = [report_path.parent]
    output_dir = options.get("output_dir") if isinstance(options, dict) else None
    if output_dir:
        directories.append(Path(str(output_dir)))

    candidates: Dict[str, Path] = {}
    for directory in directories:
        for filename in QUALITY_RESULT_FILENAMES:
            candidate = directory / filename
            candidates[str(candidate.resolve())] = candidate
    return list(candidates.values())


def _collect_response_quality_from_sidecar(
    report_path: Path, options: Dict[str, Any]
) -> Dict[str, int]:
    for candidate in _candidate_result_paths(report_path, options):
        if candidate.is_file():
            return _collect_response_quality_from_jsonl(candidate)
    return _empty_response_quality_counts()


def _extract_target_years(summary: Dict[str, Any], options: Dict[str, Any]) -> Set[int]:
    years = summary.get("target_years")
    if years is None:
        years = options.get("years")
    if years is None:
        years = summary.get("season_year", options.get("season_year"))
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
        game_type = summary.get("status_bucket", options.get("status_bucket"))
    if game_type is None:
        return ""
    return str(game_type).upper().strip()


def _extract_focus_signature(summary: Dict[str, Any], options: Dict[str, Any]) -> str:
    value = summary.get("focus_signature")
    if value is None:
        focus_values = summary.get("target_focus", options.get("focus"))
        if isinstance(focus_values, list):
            normalized = [
                str(item).strip().lower() for item in focus_values if str(item).strip()
            ]
            value = "+".join(normalized) if normalized else "all"
    if value is None:
        return ""
    return str(value).strip().lower()


def aggregate_metrics(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_cases = 0
    total_failed = 0
    total_success = 0
    total_generated_success = 0
    total_skipped = 0
    transport_failed_targets = 0
    transport_failure_count = 0
    validator_fail_count = 0
    cache_invalid_year_count = 0
    legacy_residual_total = 0
    response_quality_counts = _empty_response_quality_counts()

    warning_rate_numerator = 0.0
    warning_rate_denominator = 0
    critical_rate_numerator = 0.0
    critical_rate_denominator = 0
    llm_manual_rate_numerator = 0.0
    fallback_rate_numerator = 0.0
    focus_missing_rate_numerator = 0.0
    drift_rate_numerator = 0.0
    drift_rate_denominator = 0
    drift_reports = 0
    coach_generation_reports = 0

    observed_years: Set[int] = set()
    observed_game_types: Set[str] = set()
    observed_focus_signatures: Set[str] = set()

    for entry in reports:
        summary = entry["summary"]
        options = entry.get("options", {})
        _merge_response_quality_counts(
            response_quality_counts, entry.get("response_quality", {})
        )

        cases = _as_int(summary.get("cases", summary.get("total_targets")))
        raw_failed = _as_int(summary.get("failed", summary.get("failed_targets")))
        raw_success = _as_int(summary.get("success", summary.get("passed_targets")))
        failed = _as_int(summary.get("content_failed_targets"), raw_failed)
        success = _as_int(summary.get("content_passed_targets"), raw_success)
        generated_success = _as_int(
            summary.get("generated_success_count", summary.get("success"))
        )
        if "generated_success_count" not in summary and "success" not in summary:
            generated_success = success
        skipped = _as_int(summary.get("skipped_count", summary.get("skipped")))
        if not skipped and "processed_targets" in summary and cases:
            skipped = max(cases - _as_int(summary.get("processed_targets")), 0)

        total_cases += cases
        total_failed += failed
        total_success += success
        total_generated_success += generated_success
        total_skipped += skipped
        transport_failed_targets += _as_int(summary.get("transport_failed_targets"))
        transport_failure_count += _as_int(summary.get("transport_failure_count"))

        validator_fail_count += _as_int(
            summary.get(
                "validator_fail_count",
                summary.get(
                    "content_hard_failure_count",
                    summary.get("hard_failure_count"),
                ),
            )
        )
        cache_invalid_year_count += _as_int(summary.get("cache_invalid_year_count"))
        legacy_residual_total += _as_int(summary.get("legacy_residual_total"))

        if "warning_rate" in summary:
            warning_rate = _as_float(summary.get("warning_rate"))
        else:
            warning_rate = (
                _as_float(summary.get("soft_warning_count")) / success
                if success > 0
                else 0.0
            )
        if success > 0:
            warning_rate_numerator += warning_rate * success
            warning_rate_denominator += success

        if "critical_over_limit_rate" in summary:
            critical_rate = _as_float(summary.get("critical_over_limit_rate"))
        else:
            critical_rate = (
                _as_float(
                    summary.get(
                        "content_hard_failure_count",
                        summary.get("hard_failure_count"),
                    )
                )
                / cases
                if cases > 0
                else 0.0
            )
        if success > 0:
            critical_rate_numerator += critical_rate * success
            critical_rate_denominator += success

        llm_manual_rate = _as_float(summary.get("llm_manual_rate"))
        if success > 0:
            llm_manual_rate_numerator += llm_manual_rate * success

        fallback_rate = _as_float(summary.get("fallback_rate"))
        if success > 0:
            fallback_rate_numerator += fallback_rate * success

        focus_missing_rate = _as_float(summary.get("focus_section_missing_rate"))
        if success > 0:
            focus_missing_rate_numerator += focus_missing_rate * success
        if any(
            key in summary
            for key in (
                "llm_manual_rate",
                "fallback_rate",
                "focus_section_missing_rate",
            )
        ):
            coach_generation_reports += 1

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
        focus_signature = _extract_focus_signature(summary, options)
        if focus_signature:
            observed_focus_signatures.add(focus_signature)

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
    llm_manual_rate = (
        round(llm_manual_rate_numerator / warning_rate_denominator, 4)
        if warning_rate_denominator > 0
        else 0.0
    )
    fallback_rate = (
        round(fallback_rate_numerator / warning_rate_denominator, 4)
        if warning_rate_denominator > 0
        else 0.0
    )
    focus_section_missing_rate = (
        round(focus_missing_rate_numerator / warning_rate_denominator, 4)
        if warning_rate_denominator > 0
        else 0.0
    )
    drift_rate = (
        round(drift_rate_numerator / drift_rate_denominator, 4)
        if drift_rate_denominator > 0
        else 0.0
    )
    response_quality_checked_count = response_quality_counts["checked_count"]
    weak_grounding_rate = (
        round(
            response_quality_counts["weak_grounding_count"]
            / response_quality_checked_count,
            4,
        )
        if response_quality_checked_count > 0
        else 0.0
    )
    uncertainty_gap_rate = (
        round(
            response_quality_counts["uncertainty_gap_count"]
            / response_quality_checked_count,
            4,
        )
        if response_quality_checked_count > 0
        else 0.0
    )
    thin_markdown_rate = (
        round(
            response_quality_counts["thin_markdown_count"]
            / response_quality_checked_count,
            4,
        )
        if response_quality_checked_count > 0
        else 0.0
    )

    return {
        "report_count": len(reports),
        "cases": total_cases,
        "success": total_success,
        "generated_success_count": total_generated_success,
        "skipped_count": total_skipped,
        "failed": total_failed,
        "transport_failed_targets": transport_failed_targets,
        "transport_failure_count": transport_failure_count,
        "coverage_rate": coverage_rate,
        "validator_fail_count": validator_fail_count,
        "cache_invalid_year_count": cache_invalid_year_count,
        "legacy_residual_total": legacy_residual_total,
        "warning_rate": warning_rate,
        "critical_over_limit_rate": critical_over_limit_rate,
        "llm_manual_rate": llm_manual_rate,
        "fallback_rate": fallback_rate,
        "focus_section_missing_rate": focus_section_missing_rate,
        "drift_rate": drift_rate,
        "drift_reports": drift_reports,
        "coach_generation_reports": coach_generation_reports,
        "response_quality_checked_count": response_quality_checked_count,
        "weak_grounding_count": response_quality_counts["weak_grounding_count"],
        "weak_grounding_rate": weak_grounding_rate,
        "uncertainty_gap_count": response_quality_counts["uncertainty_gap_count"],
        "uncertainty_gap_rate": uncertainty_gap_rate,
        "thin_markdown_count": response_quality_counts["thin_markdown_count"],
        "thin_markdown_rate": thin_markdown_rate,
        "observed_target_years": sorted(observed_years),
        "observed_game_types": sorted(observed_game_types),
        "observed_focus_signatures": sorted(observed_focus_signatures),
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
    if cache_invalid_year_count > _as_int(
        thresholds["cache_invalid_year_max"]
    ) or legacy_residual_total > _as_int(thresholds["legacy_residual_max"]):
        failure_codes.append("cache_integrity_fail")

    if _as_float(metrics.get("warning_rate")) > _as_float(
        thresholds["warning_rate_max"]
    ):
        failure_codes.append("warning_rate_fail")

    if _as_float(metrics.get("critical_over_limit_rate")) > _as_float(
        thresholds["critical_over_limit_rate_max"]
    ):
        failure_codes.append("critical_over_limit_fail")

    if _as_int(metrics.get("coach_generation_reports")) > 0:
        if _as_float(metrics.get("llm_manual_rate")) < _as_float(
            thresholds["llm_manual_rate_min"]
        ):
            failure_codes.append("llm_manual_rate_fail")

        if _as_float(metrics.get("fallback_rate")) > _as_float(
            thresholds["fallback_rate_max"]
        ):
            failure_codes.append("fallback_rate_fail")

        if _as_float(metrics.get("focus_section_missing_rate")) > _as_float(
            thresholds["focus_section_missing_rate_max"]
        ):
            failure_codes.append("focus_section_missing_rate_fail")

    drift_reports = _as_int(metrics.get("drift_reports"))
    if drift_reports > 0 and _as_float(metrics.get("drift_rate")) > _as_float(
        thresholds["drift_rate_max"]
    ):
        failure_codes.append("drift_fail")

    if _as_int(metrics.get("response_quality_checked_count")) > 0:
        if _as_float(metrics.get("weak_grounding_rate")) > _as_float(
            thresholds["weak_grounding_rate_max"]
        ):
            failure_codes.append("weak_grounding_fail")

        if _as_float(metrics.get("uncertainty_gap_rate")) > _as_float(
            thresholds["uncertainty_gap_rate_max"]
        ):
            failure_codes.append("uncertainty_gap_fail")

        if _as_float(metrics.get("thin_markdown_rate")) > _as_float(
            thresholds["thin_markdown_rate_max"]
        ):
            failure_codes.append("thin_markdown_fail")

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
            "required_years": (
                sorted(required_years) if required_years is not None else None
            ),
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
    parser.add_argument(
        "--coverage-min", type=float, default=DEFAULT_THRESHOLDS["coverage_min"]
    )
    parser.add_argument(
        "--warning-rate-max", type=float, default=DEFAULT_THRESHOLDS["warning_rate_max"]
    )
    parser.add_argument(
        "--critical-over-limit-rate-max",
        type=float,
        default=DEFAULT_THRESHOLDS["critical_over_limit_rate_max"],
    )
    parser.add_argument(
        "--drift-rate-max", type=float, default=DEFAULT_THRESHOLDS["drift_rate_max"]
    )
    parser.add_argument(
        "--llm-manual-rate-min",
        type=float,
        default=DEFAULT_THRESHOLDS["llm_manual_rate_min"],
    )
    parser.add_argument(
        "--fallback-rate-max",
        type=float,
        default=DEFAULT_THRESHOLDS["fallback_rate_max"],
    )
    parser.add_argument(
        "--focus-section-missing-rate-max",
        type=float,
        default=DEFAULT_THRESHOLDS["focus_section_missing_rate_max"],
    )
    parser.add_argument(
        "--weak-grounding-rate-max",
        type=float,
        default=DEFAULT_THRESHOLDS["weak_grounding_rate_max"],
        help="Maximum share of structured responses missing verdict/why_it_matters.",
    )
    parser.add_argument(
        "--uncertainty-gap-rate-max",
        type=float,
        default=DEFAULT_THRESHOLDS["uncertainty_gap_rate_max"],
        help="Maximum share of partial/grounding-warning responses without uncertainty text.",
    )
    parser.add_argument(
        "--thin-markdown-rate-max",
        type=float,
        default=DEFAULT_THRESHOLDS["thin_markdown_rate_max"],
        help="Maximum share of structured responses with fewer than two detailed markdown sections.",
    )
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
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    thresholds = {
        "coverage_min": args.coverage_min,
        "warning_rate_max": args.warning_rate_max,
        "critical_over_limit_rate_max": args.critical_over_limit_rate_max,
        "drift_rate_max": args.drift_rate_max,
        "llm_manual_rate_min": args.llm_manual_rate_min,
        "fallback_rate_max": args.fallback_rate_max,
        "focus_section_missing_rate_max": args.focus_section_missing_rate_max,
        "weak_grounding_rate_max": args.weak_grounding_rate_max,
        "uncertainty_gap_rate_max": args.uncertainty_gap_rate_max,
        "thin_markdown_rate_max": args.thin_markdown_rate_max,
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
