#!/usr/bin/env python3
"""Run the read-only DB/embedding consistency audit suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ArtifactPaths:
    coverage_json: Path
    coverage_csv: Path
    storage_json: Path
    embedding_summary_json: Path
    embedding_samples_json: Path
    source_drift_json: Path
    source_drift_csv: Path
    triage_json: Path
    triage_csv: Path
    triage_md: Path
    summary_md: Path


@dataclass(frozen=True)
class AuditStep:
    name: str
    command: List[str]


@dataclass(frozen=True)
class StepResult:
    name: str
    command: List[str]
    returncode: int
    stdout: str
    stderr: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run read-only DB and RAG embedding consistency audits."
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--mode",
        choices=["all", "seasonal", "static"],
        default="all",
    )
    parser.add_argument("--source-env-file", default="")
    parser.add_argument("--source-db-url", default="")
    parser.add_argument("--dest-env-file", default="")
    parser.add_argument("--dest-db-url", default="")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--read-batch-size", type=int, default=500)
    parser.add_argument("--today", default="")
    parser.add_argument("--use-legacy-renderer", action="store_true")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--timestamp", default="")
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to invoke audit scripts.",
    )
    parser.add_argument(
        "--skip-embedding-256-audit",
        action="store_true",
        help="Skip the advisory 256-dimensional embedding migration audit.",
    )
    parser.add_argument(
        "--skip-triage",
        action="store_true",
        help="Skip the read-only triage pass over generated audit reports.",
    )
    parser.add_argument(
        "--triage-fail-on",
        choices=["never", "critical", "any"],
        default="never",
        help="Exit policy for the triage pass.",
    )
    return parser


def _timestamp(value: str = "") -> str:
    return value.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_output_dir(raw_output_dir: str) -> Path:
    output_dir = Path(raw_output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir


def build_artifact_paths(output_dir: Path, timestamp: str) -> ArtifactPaths:
    return ArtifactPaths(
        coverage_json=output_dir / f"db_embedding_coverage_{timestamp}.json",
        coverage_csv=output_dir / f"db_embedding_coverage_{timestamp}.csv",
        storage_json=output_dir / f"rag_storage_audit_{timestamp}.json",
        embedding_summary_json=output_dir
        / f"embedding_256_audit_summary_{timestamp}.json",
        embedding_samples_json=output_dir
        / f"embedding_256_audit_samples_{timestamp}.json",
        source_drift_json=output_dir / f"rag_source_drift_{timestamp}.json",
        source_drift_csv=output_dir / f"rag_source_drift_{timestamp}.csv",
        triage_json=output_dir / f"db_embedding_triage_{timestamp}.json",
        triage_csv=output_dir / f"db_embedding_triage_actions_{timestamp}.csv",
        triage_md=output_dir / f"db_embedding_triage_handoff_{timestamp}.md",
        summary_md=output_dir / f"db_embedding_consistency_summary_{timestamp}.md",
    )


def _append_db_args(command: List[str], args: argparse.Namespace) -> None:
    if args.source_env_file:
        command.extend(["--source-env-file", args.source_env_file])
    if args.source_db_url:
        command.extend(["--source-db-url", args.source_db_url])
    if args.dest_env_file:
        command.extend(["--dest-env-file", args.dest_env_file])
    if args.dest_db_url:
        command.extend(["--dest-db-url", args.dest_db_url])


def build_steps(
    args: argparse.Namespace,
    artifacts: ArtifactPaths,
) -> List[AuditStep]:
    py = args.python_executable
    coverage = [
        py,
        "scripts/verify_embedding_coverage.py",
        "--start-year",
        str(args.start_year),
        "--end-year",
        str(args.end_year),
        "--mode",
        args.mode,
        "--sample-limit",
        str(args.sample_limit),
        "--output",
        str(artifacts.coverage_json),
        "--csv-output",
        str(artifacts.coverage_csv),
    ]
    _append_db_args(coverage, args)

    storage = [
        py,
        "scripts/audit_rag_storage_hardening.py",
        "--active-only",
        "--sample-size",
        str(args.sample_limit),
        "--output",
        str(artifacts.storage_json),
    ]

    source_drift = [
        py,
        "scripts/audit_rag_chunk_source_drift.py",
        "--start-year",
        str(args.start_year),
        "--end-year",
        str(args.end_year),
        "--mode",
        args.mode,
        "--sample-limit",
        str(args.sample_limit),
        "--read-batch-size",
        str(args.read_batch_size),
        "--output",
        str(artifacts.source_drift_json),
        "--csv-output",
        str(artifacts.source_drift_csv),
    ]
    _append_db_args(source_drift, args)
    if args.today:
        source_drift.extend(["--today", args.today])
    if args.use_legacy_renderer:
        source_drift.append("--use-legacy-renderer")

    steps = [
        AuditStep("coverage", coverage),
        AuditStep("storage", storage),
        AuditStep("source_drift", source_drift),
    ]
    if not args.skip_embedding_256_audit:
        steps.insert(
            2,
            AuditStep(
                "embedding_256",
                [
                    py,
                    "scripts/audit_embedding_256_migration.py",
                    "--sample-limit",
                    str(args.sample_limit),
                    "--summary-output",
                    str(artifacts.embedding_summary_json),
                    "--samples-output",
                    str(artifacts.embedding_samples_json),
                ],
            ),
        )
    if not args.skip_triage:
        steps.append(
            AuditStep(
                "triage",
                [
                    py,
                    "scripts/triage_db_embedding_audit.py",
                    "--coverage-report",
                    str(artifacts.coverage_json),
                    "--source-drift-report",
                    str(artifacts.source_drift_json),
                    "--storage-report",
                    str(artifacts.storage_json),
                    "--embedding-256-report",
                    str(artifacts.embedding_summary_json),
                    "--output-dir",
                    str(artifacts.coverage_json.parent),
                    "--timestamp",
                    artifacts.coverage_json.stem.removeprefix(
                        "db_embedding_coverage_"
                    ),
                    "--sample-limit",
                    str(args.sample_limit),
                    "--fail-on",
                    args.triage_fail_on,
                ],
            )
        )
    return steps


def _read_env_file(path: str) -> Dict[str, str]:
    if not path:
        return {}
    env_path = Path(path).expanduser()
    if not env_path.is_absolute():
        env_path = (PROJECT_ROOT / env_path).resolve()
    values: Dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized = value.strip()
        if (
            len(normalized) >= 2
            and normalized[0] == normalized[-1]
            and normalized[0] in {"'", '"'}
        ):
            normalized = normalized[1:-1]
        values[key.strip()] = normalized
    return values


def build_subprocess_env(args: argparse.Namespace) -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("BEGA_SKIP_APP_INIT", "1")
    env_file = args.dest_env_file or args.source_env_file
    env.update(_read_env_file(env_file))
    if args.dest_db_url:
        env["OCI_DB_URL"] = args.dest_db_url
        env["POSTGRES_DB_URL"] = args.dest_db_url
    return env


def run_steps(steps: List[AuditStep], env: Dict[str, str]) -> List[StepResult]:
    results: List[StepResult] = []
    for step in steps:
        print(f"running {step.name}...", flush=True)
        completed = subprocess.run(
            step.command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        results.append(
            StepResult(
                name=step.name,
                command=step.command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        )
    return results


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _redact_command(command: List[str]) -> str:
    redacted: List[str] = []
    redact_next = False
    secret_flags = {"--source-db-url", "--dest-db-url"}
    for token in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        redacted.append(token)
        if token in secret_flags:
            redact_next = True
    return " ".join(redacted)


def render_markdown_summary(
    *,
    generated_at: str,
    args: argparse.Namespace,
    artifacts: ArtifactPaths,
    results: List[StepResult],
) -> str:
    coverage = _load_json(artifacts.coverage_json)
    storage = _load_json(artifacts.storage_json)
    embedding = _load_json(artifacts.embedding_summary_json)
    drift = _load_json(artifacts.source_drift_json)
    triage = _load_json(artifacts.triage_json)

    coverage_rows = coverage.get("rows") if isinstance(coverage.get("rows"), list) else []
    coverage_bad = sum(1 for row in coverage_rows if row.get("status") != "OK")
    storage_summary = storage.get("summary") if isinstance(storage.get("summary"), dict) else {}
    drift_summary = drift.get("summary") if isinstance(drift.get("summary"), dict) else {}
    triage_summary = triage.get("summary") if isinstance(triage.get("summary"), dict) else {}

    lines = [
        "# DB Embedding Consistency Audit",
        "",
        f"- generated_at_utc: {generated_at}",
        f"- scope: {args.mode} {args.start_year}-{args.end_year}",
        f"- sample_limit: {args.sample_limit}",
        "",
        "## Step Results",
        "",
        "| step | exit_code | command |",
        "| --- | ---: | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.returncode} | `{_redact_command(result.command)}` |"
        )

    lines.extend(
        [
            "",
            "## Findings Summary",
            "",
            f"- coverage_non_ok_targets: {coverage_bad}",
            f"- storage_missing_embedding: {storage_summary.get('missing_embedding', 'n/a')}",
            f"- storage_missing_content_hash: {storage_summary.get('missing_content_hash', 'n/a')}",
            f"- embedding_256_status: {embedding.get('status', 'skipped_or_missing')}",
            f"- source_drift_status: {drift_summary.get('status', 'missing')}",
            f"- source_drift_total: {drift_summary.get('drift_total', 'n/a')}",
            f"- triage_status: {triage_summary.get('status', 'skipped_or_missing')}",
            f"- triage_action_count: {triage_summary.get('action_count', 'n/a')}",
            f"- triage_critical_count: {triage_summary.get('critical_count', 'n/a')}",
            "",
            "## Artifacts",
            "",
            f"- coverage_json: {artifacts.coverage_json}",
            f"- coverage_csv: {artifacts.coverage_csv}",
            f"- storage_json: {artifacts.storage_json}",
            f"- embedding_summary_json: {artifacts.embedding_summary_json}",
            f"- embedding_samples_json: {artifacts.embedding_samples_json}",
            f"- source_drift_json: {artifacts.source_drift_json}",
            f"- source_drift_csv: {artifacts.source_drift_csv}",
            f"- triage_json: {artifacts.triage_json}",
            f"- triage_csv: {artifacts.triage_csv}",
            f"- triage_handoff_md: {artifacts.triage_md}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.start_year > args.end_year:
        print("start-year must be less than or equal to end-year", file=sys.stderr)
        return 1
    if args.sample_limit < 0:
        print("sample-limit must be >= 0", file=sys.stderr)
        return 1
    if args.read_batch_size <= 0:
        print("read-batch-size must be > 0", file=sys.stderr)
        return 1

    timestamp = _timestamp(args.timestamp)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = build_artifact_paths(output_dir, timestamp)
    steps = build_steps(args, artifacts)
    generated_at = datetime.now(timezone.utc).isoformat()
    results = run_steps(steps, build_subprocess_env(args))
    summary = render_markdown_summary(
        generated_at=generated_at,
        args=args,
        artifacts=artifacts,
        results=results,
    )
    artifacts.summary_md.write_text(summary, encoding="utf-8")
    print(f"summary saved: {artifacts.summary_md}")
    return 0 if all(result.returncode == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
