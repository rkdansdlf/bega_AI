#!/usr/bin/env python3
"""Operational rollout helper for semantic chunking re-ingest and validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest_from_kbo import DEFAULT_TABLES
from scripts.verify_embedding_coverage import (
    SEASONAL_TABLES,
    STATIC_FILE_PROFILES,
    STATIC_TABLES,
)

LOG_DIR = PROJECT_ROOT / "logs"
BENCHMARK_OUTPUT = LOG_DIR / "retrieval_benchmark_semantic_chunks.json"
VERIFY_OUTPUT_JSON = LOG_DIR / "embedding_coverage.json"
VERIFY_OUTPUT_CSV = LOG_DIR / "embedding_coverage.csv"
POST_REEMBED_VERIFY_OUTPUT_JSON = LOG_DIR / "embedding_coverage_post_reembed.json"
POST_REEMBED_VERIFY_OUTPUT_CSV = LOG_DIR / "embedding_coverage_post_reembed.csv"
ROLLOUT_SUMMARY_OUTPUT = LOG_DIR / "semantic_chunk_rollout_summary.json"

STATIC_ROLLOUT_TABLES = [
    table
    for table in DEFAULT_TABLES
    if table in STATIC_TABLES or table in STATIC_FILE_PROFILES
]
SEASONAL_ROLLOUT_TABLES = [
    table for table in DEFAULT_TABLES if table in SEASONAL_TABLES
]


@dataclass(frozen=True)
class RolloutStep:
    name: str
    command: List[str]
    report_path: Optional[Path] = None


@dataclass
class StepResult:
    name: str
    command: List[str]
    returncode: Optional[int] = None
    report_path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run semantic chunk rollout steps in a safe, repeatable order."
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--read-batch-size", type=int, default=500)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--commit-interval", type=int, default=1000)
    parser.add_argument(
        "--parallel-engine",
        choices=("thread", "subinterp"),
        default="thread",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--benchmark-limit", type=int, default=5)
    parser.add_argument("--coverage-sample-limit", type=int, default=20)
    parser.add_argument("--report-path", default=str(VERIFY_OUTPUT_JSON))
    parser.add_argument("--summary-output", default=str(ROLLOUT_SUMMARY_OUTPUT))
    parser.add_argument("--allow-benchmark-regression", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-static-ingest", action="store_true")
    parser.add_argument("--skip-seasonal-ingest", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-reembed", action="store_true")
    parser.add_argument("--skip-final-verify", action="store_true")
    return parser


def _base_ingest_command(args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "ingest_from_kbo.py"),
        "--embed-batch-size",
        str(max(1, args.embed_batch_size)),
        "--read-batch-size",
        str(max(1, args.read_batch_size)),
        "--max-concurrency",
        str(max(1, args.max_concurrency)),
        "--commit-interval",
        str(max(1, args.commit_interval)),
        "--parallel-engine",
        args.parallel_engine,
        "--workers",
        str(max(1, args.workers)),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def build_rollout_steps(args: argparse.Namespace) -> List[RolloutStep]:
    if args.start_year > args.end_year:
        raise ValueError("start-year must be less than or equal to end-year")

    report_path = str(Path(args.report_path).expanduser().resolve())
    steps: List[RolloutStep] = []
    if not args.skip_benchmark:
        steps.append(
            RolloutStep(
                name="benchmark_documents",
                command=[
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "benchmark_retrieval.py"),
                    "--variant",
                    "both",
                    "--limit",
                    str(max(1, args.benchmark_limit)),
                    "--output",
                    str(BENCHMARK_OUTPUT),
                ],
                report_path=BENCHMARK_OUTPUT,
            )
        )

    if not args.skip_static_ingest and STATIC_ROLLOUT_TABLES:
        steps.append(
            RolloutStep(
                name="ingest_static",
                command=_base_ingest_command(args)
                + ["--tables", *STATIC_ROLLOUT_TABLES],
            )
        )

    if not args.skip_seasonal_ingest and SEASONAL_ROLLOUT_TABLES:
        for year in range(args.start_year, args.end_year + 1):
            steps.append(
                RolloutStep(
                    name=f"ingest_seasonal_{year}",
                    command=_base_ingest_command(args)
                    + [
                        "--season-year",
                        str(year),
                        "--tables",
                        *SEASONAL_ROLLOUT_TABLES,
                    ],
                )
            )

    if not args.skip_verify:
        steps.append(
            RolloutStep(
                name="verify_coverage",
                command=[
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "verify_embedding_coverage.py"),
                    "--mode",
                    "all",
                    "--start-year",
                    str(args.start_year),
                    "--end-year",
                    str(args.end_year),
                    "--sample-limit",
                    str(max(0, args.coverage_sample_limit)),
                    "--output",
                    report_path,
                    "--csv-output",
                    str(VERIFY_OUTPUT_CSV),
                ],
                report_path=Path(report_path),
            )
        )

    if not args.skip_reembed:
        steps.append(
            RolloutStep(
                name="reembed_missing",
                command=[
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "reembed_missing_rows.py"),
                    "--report-path",
                    report_path,
                    "--start-year",
                    str(args.start_year),
                    "--end-year",
                    str(args.end_year),
                    "--embed-batch-size",
                    str(max(1, args.embed_batch_size)),
                    "--read-batch-size",
                    str(max(1, args.read_batch_size)),
                    "--max-concurrency",
                    str(max(1, args.max_concurrency)),
                    "--commit-interval",
                    str(max(1, args.commit_interval)),
                ],
            )
        )

    if not args.skip_final_verify:
        steps.append(
            RolloutStep(
                name="verify_coverage_post_reembed",
                command=[
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "verify_embedding_coverage.py"),
                    "--mode",
                    "all",
                    "--start-year",
                    str(args.start_year),
                    "--end-year",
                    str(args.end_year),
                    "--sample-limit",
                    str(max(0, args.coverage_sample_limit)),
                    "--output",
                    str(POST_REEMBED_VERIFY_OUTPUT_JSON),
                    "--csv-output",
                    str(POST_REEMBED_VERIFY_OUTPUT_CSV),
                ],
                report_path=POST_REEMBED_VERIFY_OUTPUT_JSON,
            )
        )

    return steps


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Expected report file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _benchmark_step_details(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    overall = payload.get("summary", {}).get("overall", {})
    acceptance = overall.get("acceptance")
    if not isinstance(acceptance, dict):
        raise RuntimeError(f"Benchmark acceptance is missing in report: {path}")
    return {
        "overall": overall,
        "acceptance": acceptance,
    }


def _coverage_step_details(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError(f"Coverage summary is missing in report: {path}")
    return {"summary": summary}


def _collect_step_details(step: RolloutStep) -> Dict[str, Any]:
    if step.report_path is None:
        return {}
    if step.name == "benchmark_documents":
        return _benchmark_step_details(step.report_path)
    if step.name.startswith("verify_coverage"):
        return _coverage_step_details(step.report_path)
    return {}


def _serialize_step_results(steps: Sequence[StepResult]) -> List[Dict[str, Any]]:
    return [asdict(step) for step in steps]


def _write_rollout_summary(
    args: argparse.Namespace,
    summary: Dict[str, Any],
    steps: Sequence[StepResult],
) -> None:
    output_path = Path(args.summary_output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    payload["steps"] = _serialize_step_results(steps)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _failure_payload(*, step: str, reason: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "step": step,
        "reason": reason,
    }
    if details:
        payload["details"] = details
    return payload


def _build_summary(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "start_year": args.start_year,
            "end_year": args.end_year,
            "limit": args.limit,
            "embed_batch_size": args.embed_batch_size,
            "read_batch_size": args.read_batch_size,
            "max_concurrency": args.max_concurrency,
            "commit_interval": args.commit_interval,
            "parallel_engine": args.parallel_engine,
            "workers": args.workers,
            "benchmark_limit": args.benchmark_limit,
            "coverage_sample_limit": args.coverage_sample_limit,
            "report_path": str(Path(args.report_path).expanduser().resolve()),
            "dry_run": args.dry_run,
            "allow_benchmark_regression": args.allow_benchmark_regression,
        },
    }


def run_rollout(args: argparse.Namespace) -> int:
    steps = build_rollout_steps(args)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    step_results: List[StepResult] = []
    summary = _build_summary(args)
    summary["status"] = "dry_run" if args.dry_run else "running"

    for index, step in enumerate(steps, start=1):
        command_str = shlex.join(step.command)
        print(f"[{index}/{len(steps)}] {step.name}: {command_str}", flush=True)
        step_result = StepResult(
            name=step.name,
            command=step.command,
            report_path=str(step.report_path) if step.report_path else None,
        )
        step_results.append(step_result)
        if args.dry_run:
            continue
        completed = subprocess.run(
            step.command,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        step_result.returncode = completed.returncode
        try:
            step_result.details = _collect_step_details(step)
        except Exception as exc:
            summary["status"] = "failed"
            summary["failure"] = _failure_payload(
                step=step.name,
                reason="report_parse_failed",
                details={"error": str(exc)},
            )
            _write_rollout_summary(args, summary, step_results)
            return 1

        if step.name == "benchmark_documents":
            acceptance = step_result.details.get("acceptance", {})
            passed = bool(acceptance.get("passed"))
            if not passed and not args.allow_benchmark_regression:
                summary["status"] = "blocked"
                summary["failure"] = _failure_payload(
                    step=step.name,
                    reason="benchmark_acceptance_failed",
                    details=acceptance,
                )
                _write_rollout_summary(args, summary, step_results)
                return 2

        if completed.returncode != 0:
            summary["status"] = "failed"
            summary["failure"] = _failure_payload(
                step=step.name,
                reason="command_failed",
                details={"returncode": completed.returncode},
            )
            _write_rollout_summary(args, summary, step_results)
            return completed.returncode

    summary["status"] = "dry_run" if args.dry_run else "completed"
    _write_rollout_summary(args, summary, step_results)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_rollout(args)


if __name__ == "__main__":
    raise SystemExit(main())
