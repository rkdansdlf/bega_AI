#!/usr/bin/env python3
"""Post-run finalizer for 2025 matchup batch.

Usage:
    python scripts/finalize_matchup_batch.py \
      --report /path/to/coach_matchup_2025_report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _run(cmd: List[str], *, timeout: int | None = None) -> int:
    proc = subprocess.run(cmd, check=False, timeout=timeout)
    return proc.returncode


def _load_json(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report is not object: {path}")
    return payload


def _ensure_completed(summary: Dict[str, Any], expected: int) -> Tuple[bool, str]:
    cases = int(summary.get("cases", 0) or 0)
    failed = int(summary.get("failed", 0) or 0)
    if cases != expected:
        return False, f"cases {cases} != expected {expected}"
    if failed != 0:
        return False, f"failed {failed} != 0"
    if summary.get("in_progress", 0):
        return False, "in_progress remains"
    return True, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize matchup batch with verification + quality + smoke checks"
    )
    parser.add_argument("--report", required=True, help="Batch quality report JSON")
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=90,
        help="Expected report summary cases (default: 90)",
    )
    parser.add_argument(
        "--required-generated-success",
        type=int,
        default=90,
        help="Required generated_success_count (default: 90)",
    )
    parser.add_argument(
        "--require-years",
        default="2025",
        help="Required years for gate check (comma-separated)",
    )
    parser.add_argument(
        "--require-game-type",
        default="REGULAR",
        help="Required game_type for gate check",
    )
    parser.add_argument(
        "--required-prompt-version",
        default="v5_focus",
        help="Required prompt_version for db integrity check",
    )
    parser.add_argument(
        "--verify-output",
        default="/tmp/coach_matchup_verify.json",
        help="Matchup completion verify output path",
    )
    parser.add_argument(
        "--quality-output",
        default="/tmp/coach_matchup_quality_gate.json",
        help="Quality gate output path",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
        help="API base URL for smoke check",
    )
    parser.add_argument(
        "--season-year",
        type=int,
        default=2025,
        help="Coach smoke season year",
    )
    parser.add_argument(
        "--smoke-samples",
        default="KT:HH,LG:LT,KIA:NC",
        help="Comma-separated sample pairs HOME:AWAY for smoke checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    report = _load_json(args.report)
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    if not isinstance(summary, dict):
        raise ValueError("report.summary must be object")

    ok, reason = _ensure_completed(summary, args.expected_cases)
    if not ok:
        print(f"[finalize] precheck failed: {reason}")
        return 1

    verify_rc = _run(
        [
            sys.executable,
            "scripts/verify_matchup_batch_completion.py",
            "--report",
            args.report,
            "--required-generated-success",
            str(args.required_generated_success),
            "--require-years",
            args.require_years,
            "--require-game-type",
            args.require_game_type,
            "--required-prompt-version",
            args.required_prompt_version,
            "--output",
            args.verify_output,
        ]
    )
    if verify_rc != 0:
        print("[finalize] verify_matchup_batch_completion failed")
        return verify_rc

    quality_rc = _run(
        [
            sys.executable,
            "scripts/evaluate_coach_quality.py",
            args.report,
            "--required-generated-success",
            str(args.required_generated_success),
            "--require-years",
            args.require_years,
            "--require-game-type",
            args.require_game_type,
            "--output",
            args.quality_output,
        ]
    )
    if quality_rc != 0:
        print("[finalize] evaluate_coach_quality failed")
        return quality_rc

    smoke_samples = [
        pair.strip() for pair in args.smoke_samples.split(",") if pair.strip()
    ]
    for pair in smoke_samples:
        if ":" not in pair:
            print(f"[finalize] skip invalid sample: {pair}")
            continue
        home, away = [value.strip() for value in pair.split(":", 1)]
        smoke_rc = _run(
            [
                sys.executable,
                "scripts/smoke_chatbot.py",
                "--base-url",
                args.base_url,
                "--season-year",
                str(args.season_year),
                "--coach-home-team",
                home,
                "--coach-away-team",
                away,
                "--coach-request-mode",
                "auto_brief",
                "--coach-focus",
                "recent_form",
                "--strict",
            ]
        )
        if smoke_rc != 0:
            print(f"[finalize] smoke failed for {home} vs {away}")
            return smoke_rc

    print("[finalize] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
