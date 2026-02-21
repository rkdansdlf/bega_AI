#!/usr/bin/env python3
"""
Verify 2025(REGULAR) 매치업 캐시 배치 결과와 DB 상태의 일관성을 점검합니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.deps import get_connection_pool
from app.tools.team_code_resolver import CANONICAL_CODES

from scripts.evaluate_coach_quality import evaluate_reports


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_years_csv(raw: str | None) -> List[int]:
    if raw is None:
        return []
    years: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        years.append(int(token))
    return sorted(set(years))


def _status_counts(results: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"generated": 0, "skipped": 0, "failed": 0, "in_progress": 0}
    for item in results:
        status = str(item.get("status") or "").lower()
        if status in counts:
            counts[status] += 1
        else:
            counts["failed"] += 1
    return counts


def _load_report(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("report must be JSON object")
    return payload


def _collect_db_rows(cache_keys: List[str]) -> Dict[str, Dict[str, Any]]:
    if not cache_keys:
        return {}

    pool = get_connection_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT cache_key, team_id, year, status, prompt_version, error_message
            FROM coach_analysis_cache
            WHERE cache_key = ANY(%s)
            """,
            (cache_keys,),
        ).fetchall()

    records: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cache_key, team_id, year, status, prompt_version, error_message = row
        records[str(cache_key)] = {
            "team_id": str(team_id) if team_id is not None else "",
            "year": _as_int(year),
            "status": str(status or ""),
            "prompt_version": str(prompt_version or ""),
            "error_message": str(error_message or ""),
        }
    return records


def _build_failure_output(
    reasons: List[str],
    summary: Dict[str, Any],
    db_result: Dict[str, Any],
    quality_result: Dict[str, Any],
) -> Dict[str, Any]:
    if db_result.get("status") == "FAIL":
        reasons.extend([f"db:{item}" for item in db_result.get("failure_codes", [])])
    for code in quality_result.get("failure_codes", []):
        if code not in reasons:
            reasons.append(f"quality:{code}")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "failure_codes": reasons,
        "summary": summary,
        "db": db_result,
        "quality": quality_result,
    }


def _check_report(
    report: Dict[str, Any],
    *,
    required_generated_success: int,
    required_years: Set[int] | None,
    required_game_type: str | None,
    required_prompt_version: str,
    strict_game_type_check: bool,
) -> Dict[str, Any]:
    summary = report.get("summary", {})
    options = report.get("options", {})
    details = report.get("details", [])
    if not isinstance(details, list):
        details = []

    status_counts = _status_counts(details)
    cases = len(details)
    summary_cases = _as_int(summary.get("cases"), 0)

    reasons: List[str] = []
    if cases != summary_cases:
        reasons.append(
            f"summary_case_count_mismatch:{cases}!=summary_cases_{summary_cases}"
        )

    if status_counts["failed"] != 0:
        reasons.append("db:failed_exists_in_payload")
    if status_counts["in_progress"] != 0:
        reasons.append("db:in_progress_remains")
    if status_counts["generated"] + status_counts["skipped"] == 0:
        reasons.append("coverage_zero")

    failure_reasons = set(
        item.get("reason")
        for item in details
        if str(item.get("status")).lower() == "failed"
    )
    if any(str(reason).startswith("missing_done_event") for reason in failure_reasons):
        reasons.append("runtime:missing_done_event")

    cache_keys = [
        str(item.get("cache_key")) for item in details if item.get("cache_key")
    ]
    db_rows = _collect_db_rows(cache_keys)
    if len(db_rows) != len(set(cache_keys)):
        reasons.append("db:cache_key_not_found")

    status_counter: Dict[str, int] = {}
    prompt_version_mismatch = 0
    non_canonical_team_count = 0
    game_type_mismatch = 0

    for key, row in db_rows.items():
        row_status = str(row.get("status", "")).upper()
        status_counter[row_status] = status_counter.get(row_status, 0) + 1
        if row.get("prompt_version") != required_prompt_version:
            prompt_version_mismatch += 1
        team_id = str(row.get("team_id", "")).upper()
        if team_id not in CANONICAL_CODES:
            non_canonical_team_count += 1

        if strict_game_type_check:
            row_game_type = str(
                (
                    next((d for d in details if d.get("cache_key") == key), {}).get(
                        "game_type"
                    )
                )
                or ""
            ).upper()
            if required_game_type and row_game_type != required_game_type:
                game_type_mismatch += 1

    if prompt_version_mismatch:
        reasons.append(f"db:prompt_version_mismatch_{prompt_version_mismatch}")
    if non_canonical_team_count:
        reasons.append(f"db:non_canonical_team_{non_canonical_team_count}")
    if strict_game_type_check and game_type_mismatch:
        reasons.append(f"db:game_type_mismatch_{game_type_mismatch}")

    db_result = {
        "status": "PASS" if not reasons else "FAIL",
        "failure_codes": [],
        "rows_found": len(db_rows),
        "rows_expected": len(set(cache_keys)),
        "status_counter": status_counter,
        "prompt_version_mismatch": prompt_version_mismatch,
        "non_canonical_team_count": non_canonical_team_count,
        "game_type_mismatch": game_type_mismatch,
    }
    if reasons:
        db_result["failure_codes"].extend(
            code for code in reasons if code.startswith("db:")
        )

    quality_input = [{"summary": summary, "options": options}]
    quality_result = evaluate_reports(
        quality_input,
        required_generated_success=required_generated_success,
        required_years=required_years,
        require_game_type=required_game_type,
    )
    return _build_failure_output(reasons, summary, db_result, quality_result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify matchup batch completion report."
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Batch quality report JSON path.",
    )
    parser.add_argument(
        "--required-generated-success",
        type=int,
        default=90,
        help="Required generated success count.",
    )
    parser.add_argument(
        "--require-years",
        default=None,
        help="Comma-separated required years (e.g. 2025).",
    )
    parser.add_argument(
        "--require-game-type",
        default="REGULAR",
        help="Required game_type in payload.",
    )
    parser.add_argument(
        "--required-prompt-version",
        default="v5_focus",
        help="Required coach cache prompt version.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--strict-game-type",
        action="store_true",
        default=True,
        help="Enable strict game_type mismatch check from payload items.",
    )
    parser.add_argument(
        "--no-strict-game-type",
        dest="strict_game_type",
        action="store_false",
        help="Disable game_type mismatch check when payload lacks game_type.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = _load_report(args.report)
    required_years = set(parse_years_csv(args.require_years))
    check_result = _check_report(
        report,
        required_generated_success=args.required_generated_success,
        required_years=required_years if required_years else None,
        required_game_type=args.require_game_type,
        required_prompt_version=args.required_prompt_version,
        strict_game_type_check=args.strict_game_type,
    )

    check_result["inputs"] = {
        "report": args.report,
        "required_generated_success": args.required_generated_success,
        "require_years": sorted(required_years) if required_years else None,
        "require_game_type": args.require_game_type,
        "required_prompt_version": args.required_prompt_version,
        "strict_game_type": args.strict_game_type,
        "checked_at": datetime.now().isoformat(),
    }
    print(json.dumps(check_result, ensure_ascii=False, indent=2))

    if args.output:
        Path(args.output).write_text(
            json.dumps(check_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0 if check_result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
