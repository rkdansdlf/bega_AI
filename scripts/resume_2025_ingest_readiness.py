#!/usr/bin/env python3
"""Resume the 2025 readiness run after Oracle listener recovery."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_2025_ingest_readiness as readiness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Oracle wallet aliases first, then run the full 2025 readiness "
            "pipeline when Oracle becomes available."
        )
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.0,
        help="Sleep duration between Oracle probe retries. Default: no retry delay.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Maximum Oracle probe attempts before giving up.",
    )
    parser.add_argument(
        "--latest-report-dir",
        default=str(readiness.DEFAULT_REPORT_DIR),
        help="Directory containing latest.json and latest-handoff.md pointers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the probe/readiness commands without executing them.",
    )
    return parser


def build_direct_probe_command(readiness_args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sync_kbo_data.py"),
        "--check-oracle-services-direct",
        "--wallet-dir",
        str(readiness_args.wallet_dir),
        "--oracle-timeout-seconds",
        str(max(1, int(readiness_args.oracle_timeout_seconds))),
    ]


def build_readiness_command(forwarded_args: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_2025_ingest_readiness.py"),
        *forwarded_args,
    ]


def _print_completed(process: subprocess.CompletedProcess[str]) -> None:
    if process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="", file=sys.stderr)


def _load_latest_pointer_paths(report_dir: Path) -> dict[str, str]:
    latest_pointer = report_dir / "latest.json"
    latest_handoff = report_dir / "latest-handoff.md"
    payload: dict[str, str] = {
        "latest_pointer": str(latest_pointer),
        "latest_handoff_markdown": str(latest_handoff),
    }
    if latest_pointer.exists():
        try:
            parsed = json.loads(latest_pointer.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parsed = {}
        for key in (
            "report_path",
            "artifact_dir",
            "handoff_markdown",
            "support_bundle",
            "oracle_escalation_markdown",
        ):
            value = parsed.get(key)
            if value:
                payload[key] = str(value)
    return payload


def run_resume(args: argparse.Namespace, forwarded_args: Sequence[str]) -> int:
    readiness_args = readiness.build_parser().parse_args(list(forwarded_args))
    direct_probe_command = build_direct_probe_command(readiness_args)
    readiness_command = build_readiness_command(forwarded_args)

    if args.dry_run:
        print("oracle_probe_command=" + " ".join(direct_probe_command))
        print("readiness_command=" + " ".join(readiness_command))
        return 0

    attempts = max(1, args.max_attempts)
    for attempt in range(1, attempts + 1):
        print(f"oracle_probe_attempt={attempt}/{attempts}")
        probe = subprocess.run(
            direct_probe_command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        _print_completed(probe)
        if probe.returncode == 0:
            print("oracle_probe_status=ready")
            readiness_result = subprocess.run(
                readiness_command,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            _print_completed(readiness_result)
            return int(readiness_result.returncode)

        if attempt < attempts and args.poll_seconds > 0:
            print(f"oracle_probe_status=blocked retry_in_seconds={args.poll_seconds}")
            time.sleep(max(0.0, args.poll_seconds))

    print("oracle_probe_status=blocked")
    latest_paths = _load_latest_pointer_paths(Path(args.latest_report_dir))
    for key in (
        "latest_pointer",
        "latest_handoff_markdown",
        "report_path",
        "artifact_dir",
        "handoff_markdown",
        "support_bundle",
        "oracle_escalation_markdown",
    ):
        value = latest_paths.get(key)
        if value:
            print(f"{key}={value}")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded_args = parser.parse_known_args(
        list(argv) if argv is not None else None
    )
    return run_resume(args, forwarded_args)


if __name__ == "__main__":
    raise SystemExit(main())
