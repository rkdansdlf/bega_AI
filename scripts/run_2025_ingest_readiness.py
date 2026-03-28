#!/usr/bin/env python3
"""Run the 2025 ingest readiness path and write a JSON summary report."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import psycopg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from scripts.ingest_from_kbo import DEFAULT_TABLES as DEFAULT_INGEST_TABLES
from scripts.ingest_from_kbo import TABLE_PROFILES

DEFAULT_SYNC_TABLES = (
    "teams",
    "game",
    "game_metadata",
    "game_summary",
    "game_play_by_play",
    "game_events",
)
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "ingest_readiness"

SYNC_TABLE_RE = re.compile(r"^== Syncing (?P<table>[a-zA-Z0-9_]+) ==$")
SYNC_COUNTS_RE = re.compile(
    r"^source_rows=(?P<source>\d+) target_rows=(?P<target>\d+)$"
)
SYNC_FINISHED_RE = re.compile(
    r"^finished table=(?P<table>[a-zA-Z0-9_]+) synced_rows=(?P<synced>\d+)$"
)
SYNC_SELECTED_ALIAS_RE = re.compile(r"^selected_oracle_alias=(?P<alias>[a-zA-Z0-9_]+)$")
SYNC_SELECTED_ALIAS_FALLBACK_RE = re.compile(
    r"^selected_alias_fallback=(?P<value>yes|no)$"
)
SYNC_ORACLE_FAILURE_REASON_RE = re.compile(
    r"^oracle_failure_reason=(?P<reason>[a-z_]+)$"
)
SYNC_ORACLE_RESOLUTION_STATUS_RE = re.compile(
    r"^oracle_alias_resolution_status=(?P<status>ready|blocked)$"
)
ORACLE_ALIAS_DISCOVERY_RE = re.compile(r"^Discovered (?P<count>\d+) wallet aliases .*")
ORACLE_ALIAS_STATUS_RE = re.compile(
    r"^(?P<alias>[a-zA-Z0-9_]+): (?P<status>OK|FAIL)(?: - (?P<detail>.*))?$"
)
ORACLE_LISTENER_DETAIL_RE = re.compile(
    r'Service "(?P<service>[^"]+)" is not registered with the listener at host "(?P<host>[^"]+)" port (?P<port>\d+)',
    re.IGNORECASE,
)
INGEST_TABLE_RE = re.compile(
    r"^ 테이블 '(?P<table>[^']+)'을\(를\) 수집 중입니다 \.\.\.$"
)
INGEST_FINISHED_RE = re.compile(
    r"^   -> 테이블 '(?P<table>[^']+)'에서 (?P<chunks>\d+)개 청크를 작성했습니다 "
    r"\(배치=(?P<batches>\d+), 임베딩 호출=(?P<embedding_calls>\d+), "
    r"대기 시간=(?P<sleep_seconds>[0-9.]+)초, 엔진=(?P<engine>[^,]+), "
    r"fallback=(?P<fallbacks>\d+)\)$"
)
INGEST_TOTAL_RE = re.compile(r"^총 (?P<total>\d+)개 청크 수집을 완료했습니다\.$")


@dataclass
class StepRunResult:
    name: str
    command: list[str]
    cwd: str
    duration_ms: float
    exit_code: int
    status: str
    stdout_path: str
    stderr_path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 2025 season sync + ingest + readiness validation and emit a JSON report."
    )
    parser.add_argument("--season-year", type=int, default=2025)
    parser.add_argument(
        "--since", default="", help="Incremental ingest timestamp (ISO8601)."
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional final JSON report path. Defaults to reports/ingest_readiness/<timestamp>.json",
    )
    parser.add_argument(
        "--sync-tables",
        nargs="+",
        default=list(DEFAULT_SYNC_TABLES),
        help="Source sync table list.",
    )
    parser.add_argument(
        "--ingest-tables",
        nargs="+",
        default=list(DEFAULT_INGEST_TABLES),
        help="RAG ingest table list.",
    )
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-coverage", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--sync-batch-size", type=int, default=1000)
    parser.add_argument("--sync-limit", type=int, default=0)
    parser.add_argument(
        "--wallet-dir",
        default=os.getenv(
            "ORACLE_WALLET_DIR",
            str(REPO_ROOT / "bega_backend/BEGA_PROJECT/wallet"),
        ),
    )
    parser.add_argument(
        "--oracle-service-name",
        default=os.getenv("ORACLE_SERVICE_NAME", "efh9m9c9h109963k_high"),
    )
    parser.add_argument("--oracle-timeout-seconds", type=int, default=10)
    parser.add_argument(
        "--source-db-url", default="", help="Override source PostgreSQL URL."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Optional per-table ingest row limit."
    )
    parser.add_argument("--read-batch-size", type=int, default=500)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--commit-interval", type=int, default=500)
    parser.add_argument(
        "--parallel-engine", choices=("thread", "subinterp"), default="thread"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--use-legacy-renderer", action="store_true")
    parser.add_argument(
        "--coverage-mode", choices=("all", "seasonal", "static"), default="all"
    )
    parser.add_argument("--coverage-sample-limit", type=int, default=20)
    parser.add_argument("--benchmark-limit", type=int, default=5)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--internal-api-key", default=os.getenv("AI_INTERNAL_TOKEN", "")
    )
    parser.add_argument("--smoke-batch-size", type=int, default=20)
    parser.add_argument("--smoke-question-list", default="")
    parser.add_argument("--smoke-timeout", type=float, default=180.0)
    return parser


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_output_path() -> Path:
    return DEFAULT_REPORT_DIR / f"ingest_readiness_2025_{_timestamp_slug()}.json"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_command(
    *,
    name: str,
    command: Sequence[str],
    cwd: Path,
    artifact_dir: Path,
) -> StepRunResult:
    started_at = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    stdout_path = artifact_dir / f"{name}.stdout.log"
    stderr_path = artifact_dir / f"{name}.stderr.log"
    _write_text(stdout_path, completed.stdout)
    _write_text(stderr_path, completed.stderr)
    return StepRunResult(
        name=name,
        command=list(command),
        cwd=str(cwd),
        duration_ms=duration_ms,
        exit_code=completed.returncode,
        status="ok" if completed.returncode == 0 else "failed",
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _build_smoke_command(
    args: argparse.Namespace,
    *,
    smoke_output: Path,
    smoke_summary_output: Path,
    artifact_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "smoke_chatbot_quality.py"),
        "--base-url",
        args.base_url,
        "--chat-batch-size",
        str(max(1, args.smoke_batch_size)),
        "--timeout",
        str(max(1.0, args.smoke_timeout)),
        "--lock-file",
        str(artifact_dir / "smoke.lock"),
        "--output",
        str(smoke_output),
        "--summary-output",
        str(smoke_summary_output),
    ]
    if args.internal_api_key.strip():
        command.extend(["--internal-api-key", args.internal_api_key.strip()])
    if args.smoke_question_list.strip():
        command.extend(["--chat-question-list", args.smoke_question_list.strip()])
    return command


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8") if path else ""


def parse_oracle_diagnostics_stdout(stdout: str) -> Dict[str, Any]:
    alias_count: Optional[int] = None
    aliases: list[Dict[str, Any]] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        discovered_match = ORACLE_ALIAS_DISCOVERY_RE.match(line)
        if discovered_match:
            alias_count = int(discovered_match.group("count"))
            continue

        alias_match = ORACLE_ALIAS_STATUS_RE.match(line)
        if not alias_match:
            continue

        detail = (alias_match.group("detail") or "").strip()
        parsed_alias: Dict[str, Any] = {
            "alias": alias_match.group("alias"),
            "ok": alias_match.group("status") == "OK",
            "detail": detail or None,
        }
        listener_match = ORACLE_LISTENER_DETAIL_RE.search(detail)
        if listener_match:
            parsed_alias.update(
                {
                    "listener_service_name": listener_match.group("service"),
                    "listener_host": listener_match.group("host"),
                    "listener_port": int(listener_match.group("port")),
                    "listener_registration_missing": True,
                }
            )
        normalized_detail = detail.lower()
        if (
            "listener refused connection" in normalized_detail
            or "ora-12506" in normalized_detail
            or "dpy-6000" in normalized_detail
        ):
            parsed_alias["listener_refused_connection"] = True
        aliases.append(parsed_alias)

    failed_aliases = [alias for alias in aliases if not alias.get("ok", False)]
    listener_registration_missing = bool(failed_aliases) and all(
        alias.get("listener_registration_missing", False) for alias in failed_aliases
    )
    listener_refused_connection = bool(failed_aliases) and all(
        alias.get("listener_refused_connection", False) for alias in failed_aliases
    )

    return {
        "alias_count": alias_count if alias_count is not None else len(aliases),
        "ok_count": sum(1 for alias in aliases if alias.get("ok", False)),
        "failed_count": len(failed_aliases),
        "listener_registration_missing": listener_registration_missing,
        "listener_refused_connection": listener_refused_connection,
        "aliases": aliases,
    }


def _discover_wallet_service_targets(
    *,
    wallet_dir: str,
    aliases: Sequence[str] | None = None,
) -> list[Dict[str, Any]]:
    tns_path = Path(wallet_dir) / "tnsnames.ora"
    if not tns_path.exists():
        return []

    services: list[Dict[str, Any]] = []
    current_alias: Optional[str] = None
    descriptor_lines: list[str] = []
    depth = 0
    selected_aliases = {alias for alias in aliases or [] if alias}
    alias_pattern = re.compile(r"^([A-Za-z0-9_]+)\s*=\s*(.*)$")

    def _flush() -> None:
        nonlocal current_alias, descriptor_lines, depth
        if not current_alias:
            return
        descriptor = " ".join(part.strip() for part in descriptor_lines if part.strip())
        host_match = re.search(r"HOST\s*=\s*([^)]+)", descriptor, re.IGNORECASE)
        port_match = re.search(r"PORT\s*=\s*([^)]+)", descriptor, re.IGNORECASE)
        service_match = re.search(
            r"SERVICE_NAME\s*=\s*([^)]+)",
            descriptor,
            re.IGNORECASE,
        )
        try:
            port = int(port_match.group(1).strip()) if port_match else None
        except ValueError:
            port = None
        if (
            (not selected_aliases or current_alias in selected_aliases)
            and host_match
            and port is not None
            and service_match
        ):
            services.append(
                {
                    "alias": current_alias,
                    "listener_host": host_match.group(1).strip(),
                    "listener_port": port,
                    "listener_service_name": service_match.group(1).strip(),
                }
            )
        current_alias = None
        descriptor_lines = []
        depth = 0

    for raw_line in tns_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if current_alias is None:
            match = alias_pattern.match(stripped)
            if not match:
                continue
            current_alias = match.group(1)
            remainder = match.group(2)
            descriptor_lines = [remainder] if remainder else []
            depth = remainder.count("(") - remainder.count(")")
            if descriptor_lines and depth <= 0:
                _flush()
            continue
        descriptor_lines.append(stripped)
        depth += stripped.count("(") - stripped.count(")")
        if depth <= 0:
            _flush()

    _flush()
    return services


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_support_bundle_manifest(
    *,
    artifact_dir: Path,
    report_path: Path,
    bundle_path: Path,
) -> Dict[str, Any]:
    included_files = [report_path.name]
    for candidate in sorted(artifact_dir.iterdir(), key=lambda path: path.name):
        if not candidate.is_file():
            continue
        if candidate.name == bundle_path.name:
            continue
        included_files.append(candidate.name)
    return {
        "report_path": str(report_path),
        "bundle_path": str(bundle_path),
        "included_files": included_files,
    }


def write_support_bundle(
    *,
    artifact_dir: Path,
    report_path: Path,
) -> Dict[str, str]:
    bundle_manifest_path = artifact_dir / "bundle-manifest.json"
    bundle_path = artifact_dir / "support-bundle.tar.gz"
    manifest = build_support_bundle_manifest(
        artifact_dir=artifact_dir,
        report_path=report_path,
        bundle_path=bundle_path,
    )
    _write_text(
        bundle_manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    entries = [report_path.name, bundle_manifest_path.name]
    entries.extend(
        [
            candidate.name
            for candidate in sorted(artifact_dir.iterdir(), key=lambda path: path.name)
            if candidate.is_file()
            and candidate.name not in {bundle_path.name, bundle_manifest_path.name}
        ]
    )
    with tarfile.open(bundle_path, "w:gz") as archive:
        archive.add(report_path, arcname=report_path.name)
        archive.add(bundle_manifest_path, arcname=bundle_manifest_path.name)
        for entry in entries[2:]:
            archive.add(artifact_dir / entry, arcname=entry)
    return {
        "bundle_manifest": str(bundle_manifest_path),
        "support_bundle": str(bundle_path),
    }


def build_readiness_handoff_markdown(
    *,
    report: Dict[str, Any],
    report_path: Path,
) -> str:
    readiness = report.get("readiness") or {}
    checks = readiness.get("checks") or {}
    failure_reasons = readiness.get("failure_reasons") or []
    artifacts = report.get("artifacts") or {}
    oracle_remediation = report.get("oracle_remediation") or {}
    next_action = (
        str(oracle_remediation.get("summary"))
        if oracle_remediation
        else (
            "Review the failed readiness checks before rerunning the pipeline."
            if failure_reasons
            else "Proceed with the planned ingest schedule."
        )
    )
    resume_command = oracle_remediation.get("resume_command") or []
    dba_checklist = [
        str(item)
        for item in list(oracle_remediation.get("dba_checklist") or [])
        if str(item).strip()
    ]

    lines = [
        "# Ingest Readiness Handoff",
        "",
        f"- Generated at (UTC): `{report.get('generated_at_utc', 'unknown')}`",
        f"- Season year: `{(report.get('input') or {}).get('season_year', 'unknown')}`",
        f"- Report: `{report_path}`",
        f"- Ready: `{readiness.get('ready', False)}`",
        f"- Failure reasons: `{', '.join(str(item) for item in failure_reasons) if failure_reasons else 'none'}`",
        "",
        "## Checks",
        "",
    ]
    for check_name, passed in checks.items():
        lines.append(f"- `{check_name}`: `{passed}`")
    lines.extend(
        [
            "",
            "## Next Action",
            "",
            next_action,
        ]
    )
    if resume_command:
        lines.extend(
            [
                "",
                "## Resume Command",
                "",
                "```bash",
                " ".join(str(part) for part in resume_command),
                "```",
            ]
        )
    if dba_checklist:
        lines.extend(["", "## DBA Checklist", ""])
        for item in dba_checklist:
            lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for key in (
        "oracle_escalation_markdown",
        "support_bundle",
        "bundle_manifest",
        "artifact_dir",
    ):
        value = artifacts.get(key)
        if value:
            lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def write_latest_pointer_files(
    *,
    report_dir: Path,
    report_path: Path,
    handoff_markdown: str,
    report: Dict[str, Any],
) -> Dict[str, str]:
    latest_pointer_path = report_dir / "latest.json"
    latest_handoff_path = report_dir / "latest-handoff.md"
    pointer_payload = {
        "generated_at_utc": report.get("generated_at_utc"),
        "report_path": str(report_path),
        "artifact_dir": str((report.get("artifacts") or {}).get("artifact_dir", "")),
        "handoff_markdown": str(
            (report.get("artifacts") or {}).get("handoff_markdown", "")
        ),
        "support_bundle": str(
            (report.get("artifacts") or {}).get("support_bundle", "")
        ),
        "oracle_escalation_markdown": str(
            (report.get("artifacts") or {}).get("oracle_escalation_markdown", "")
        ),
        "resume_command": list(
            (report.get("oracle_remediation") or {}).get("resume_command") or []
        ),
        "ready": bool((report.get("readiness") or {}).get("ready", False)),
        "failure_reasons": list(
            (report.get("readiness") or {}).get("failure_reasons") or []
        ),
    }
    _write_text(
        latest_pointer_path,
        json.dumps(pointer_payload, ensure_ascii=False, indent=2) + "\n",
    )
    _write_text(latest_handoff_path, handoff_markdown)
    return {
        "latest_pointer": str(latest_pointer_path),
        "latest_handoff_markdown": str(latest_handoff_path),
    }


def build_oracle_remediation(
    *,
    args: argparse.Namespace,
    sync_summary: Optional[Dict[str, Any]],
    oracle_diagnostics: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    failure_reason = (sync_summary or {}).get("oracle_failure_reason")
    if not failure_reason and (oracle_diagnostics or {}).get(
        "listener_registration_missing"
    ):
        failure_reason = "listener_registration_missing"
    if not failure_reason and (oracle_diagnostics or {}).get(
        "listener_refused_connection"
    ):
        failure_reason = "listener_refused_connection"
    if not failure_reason:
        return None

    wallet_dir = str(getattr(args, "wallet_dir", "") or "")
    timeout_seconds = int(getattr(args, "oracle_timeout_seconds", 10) or 10)
    resume_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "resume_2025_ingest_readiness.py"),
        "--season-year",
        str(getattr(args, "season_year", 2025)),
    ]
    if getattr(args, "since", ""):
        resume_command.extend(["--since", str(args.since)])

    failed_aliases = [
        str(alias.get("alias"))
        for alias in list((oracle_diagnostics or {}).get("aliases") or [])
        if alias.get("alias") and not alias.get("ok", False)
    ]
    wallet_targets = _discover_wallet_service_targets(
        wallet_dir=wallet_dir,
        aliases=failed_aliases,
    )

    if failure_reason == "listener_registration_missing":
        aliases = list((oracle_diagnostics or {}).get("aliases") or [])
        service_names = sorted(
            {
                str(alias.get("listener_service_name"))
                for alias in aliases
                if alias.get("listener_service_name")
            }
            | {
                str(target.get("listener_service_name"))
                for target in wallet_targets
                if target.get("listener_service_name")
            }
        )
        listener_hosts = sorted(
            {
                str(alias.get("listener_host"))
                for alias in aliases
                if alias.get("listener_host")
            }
            | {
                str(target.get("listener_host"))
                for target in wallet_targets
                if target.get("listener_host")
            }
        )
        listener_ports = sorted(
            {
                int(alias.get("listener_port"))
                for alias in aliases
                if alias.get("listener_port") is not None
            }
            | {
                int(target.get("listener_port"))
                for target in wallet_targets
                if target.get("listener_port") is not None
            }
        )
        verification_command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "sync_kbo_data.py"),
            "--check-oracle-services-direct",
        ]
        if wallet_dir:
            verification_command.extend(["--wallet-dir", wallet_dir])
        verification_command.extend(
            ["--oracle-timeout-seconds", str(max(1, timeout_seconds))]
        )
        remediation: Dict[str, Any] = {
            "reason": failure_reason,
            "operator_action": "contact_oracle_dba",
            "summary": "Register the wallet service names on the Oracle listener, then rerun the direct probe.",
            "dba_checklist": [
                "Confirm the database/service is fully available in the Oracle control plane.",
                "Confirm the wallet service names below are registered and exposed by the listener again.",
                "If the service map changed, reissue the wallet/tnsnames bundle and share the updated aliases.",
                "After the listener exposes the services, rerun the direct probe before resuming full readiness.",
            ],
            "listener_hosts": listener_hosts,
            "listener_ports": listener_ports,
            "service_names": service_names,
            "verification_command": verification_command,
            "resume_command": resume_command,
        }
        if len(listener_hosts) == 1:
            remediation["listener_host"] = listener_hosts[0]
        if len(listener_ports) == 1:
            remediation["listener_port"] = listener_ports[0]
        return remediation
    if failure_reason == "listener_refused_connection":
        service_names = sorted(
            {
                str(target.get("listener_service_name"))
                for target in wallet_targets
                if target.get("listener_service_name")
            }
        )
        listener_hosts = sorted(
            {
                str(target.get("listener_host"))
                for target in wallet_targets
                if target.get("listener_host")
            }
        )
        listener_ports = sorted(
            {
                int(target.get("listener_port"))
                for target in wallet_targets
                if target.get("listener_port") is not None
            }
        )
        verification_command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "sync_kbo_data.py"),
            "--check-oracle-services-direct",
        ]
        if wallet_dir:
            verification_command.extend(["--wallet-dir", wallet_dir])
        verification_command.extend(
            ["--oracle-timeout-seconds", str(max(1, timeout_seconds))]
        )
        remediation = {
            "reason": failure_reason,
            "operator_action": "contact_oracle_dba",
            "summary": "Listener is reachable but refused Oracle connections after restart. Verify the database open state, service handlers, and wallet service registration, then rerun the direct probe.",
            "dba_checklist": [
                "Confirm the Autonomous Database or target service is back in AVAILABLE/OPEN state after the restart.",
                "Confirm the wallet service names below are accepting sessions again instead of refusing connections.",
                "If services were recreated or moved during restart, refresh the wallet/tnsnames bundle and redistribute it.",
                "After Oracle-side recovery is complete, rerun the direct probe and then resume full readiness.",
            ],
            "listener_hosts": listener_hosts,
            "listener_ports": listener_ports,
            "service_names": service_names,
            "verification_command": verification_command,
            "resume_command": resume_command,
        }
        if len(listener_hosts) == 1:
            remediation["listener_host"] = listener_hosts[0]
        if len(listener_ports) == 1:
            remediation["listener_port"] = listener_ports[0]
        return remediation

    remediation_map = {
        "wallet_configuration_missing": {
            "operator_action": "validate_wallet_configuration",
            "summary": "Check wallet files and ORACLE_WALLET_DIR, then rerun the direct probe.",
        },
        "connect_timeout": {
            "operator_action": "check_network_path",
            "summary": "Check the Oracle network path and listener reachability, then rerun the direct probe.",
        },
        "listener_refused_connection": {
            "operator_action": "contact_oracle_dba",
            "summary": "Listener is reachable but refused Oracle connections after restart. Verify the database open state, service handlers, and wallet service registration, then rerun the direct probe.",
        },
        "connectivity_failed": {
            "operator_action": "check_oracle_connectivity",
            "summary": "Check Oracle listener availability and network ACLs, then rerun the direct probe.",
        },
        "mixed_connectivity_failure": {
            "operator_action": "review_oracle_probe_failures",
            "summary": "Review the per-alias probe details and fix the failing Oracle path before rerunning sync.",
        },
        "probe_exited_without_result": {
            "operator_action": "rerun_oracle_probe",
            "summary": "Rerun the direct probe and inspect per-alias failures because the subprocess exited without a result.",
        },
        "oracle_probe_failed": {
            "operator_action": "review_oracle_probe_failures",
            "summary": "Review the per-alias probe details and rerun the direct probe after fixing the Oracle path.",
        },
    }
    fallback = remediation_map.get(
        str(failure_reason),
        {
            "operator_action": "review_oracle_probe_failures",
            "summary": "Review Oracle probe details before retrying sync.",
        },
    )
    return {
        "reason": str(failure_reason),
        "resume_command": resume_command,
        **fallback,
    }


def build_oracle_escalation_markdown(
    *,
    report: Dict[str, Any],
    report_path: Path,
) -> str:
    remediation = report.get("oracle_remediation") or {}
    diagnostics = report.get("oracle_diagnostics") or {}
    sync_summary = report.get("sync") or {}
    steps = report.get("steps") or {}
    sync_step = steps.get("sync") or {}
    diagnostics_step = (diagnostics or {}).get("step") or {}
    input_payload = report.get("input") or {}

    lines = [
        "# Oracle Escalation Note",
        "",
        f"- Generated at (UTC): `{report.get('generated_at_utc', 'unknown')}`",
        f"- Season year: `{input_payload.get('season_year', 'unknown')}`",
        f"- Readiness report: `{report_path}`",
        f"- Readiness ready: `{report.get('readiness', {}).get('ready', False)}`",
        f"- Failure reason: `{sync_summary.get('oracle_failure_reason', remediation.get('reason', 'unknown'))}`",
        f"- Operator action: `{remediation.get('operator_action', 'review_oracle_probe_failures')}`",
        "",
        "## Request",
        "",
        str(
            remediation.get(
                "summary", "Review Oracle probe details before retrying sync."
            )
        ),
    ]

    listener_host = remediation.get("listener_host")
    listener_port = remediation.get("listener_port")
    service_names = remediation.get("service_names") or []
    if listener_host is not None:
        lines.append(f"- Listener host: `{listener_host}`")
    if listener_port is not None:
        lines.append(f"- Listener port: `{listener_port}`")
    if service_names:
        lines.append("- Service names:")
        for service_name in service_names:
            lines.append(f"  - `{service_name}`")

    verification_command = remediation.get("verification_command") or []
    if verification_command:
        lines.extend(
            [
                "",
                "## Verification Command",
                "",
                "```bash",
                " ".join(str(part) for part in verification_command),
                "```",
            ]
        )
    resume_command = remediation.get("resume_command") or []
    if resume_command:
        lines.extend(
            [
                "",
                "## Resume Command",
                "",
                "```bash",
                " ".join(str(part) for part in resume_command),
                "```",
            ]
        )
    dba_checklist = [
        str(item)
        for item in list(remediation.get("dba_checklist") or [])
        if str(item).strip()
    ]
    if dba_checklist:
        lines.extend(["", "## DBA Checklist", ""])
        for item in dba_checklist:
            lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- `oracle_alias_resolution_status`: `{sync_summary.get('oracle_alias_resolution_status', 'unknown')}`",
            f"- `oracle_alias_probe_ok`: `{diagnostics.get('ok_count', 0)}/{diagnostics.get('alias_count', 0)}`",
        ]
    )

    sync_stdout_path = sync_step.get("stdout_path")
    sync_stderr_path = sync_step.get("stderr_path")
    diagnostics_stdout_path = diagnostics_step.get("stdout_path")
    diagnostics_stderr_path = diagnostics_step.get("stderr_path")
    if sync_stdout_path:
        lines.append(f"- Sync stdout: `{sync_stdout_path}`")
    if sync_stderr_path:
        lines.append(f"- Sync stderr: `{sync_stderr_path}`")
    if diagnostics_stdout_path:
        lines.append(f"- Oracle diagnostics stdout: `{diagnostics_stdout_path}`")
    if diagnostics_stderr_path:
        lines.append(f"- Oracle diagnostics stderr: `{diagnostics_stderr_path}`")

    failed_aliases = [
        alias for alias in (diagnostics.get("aliases") or []) if not alias.get("ok")
    ]
    if failed_aliases:
        lines.extend(["", "## Failed Aliases", ""])
        for alias in failed_aliases:
            detail = alias.get("detail") or ""
            lines.append(f"- `{alias.get('alias', 'unknown')}`: `{detail}`")

    return "\n".join(lines) + "\n"


def parse_sync_stdout(stdout: str) -> Dict[str, Any]:
    current_table: Optional[str] = None
    per_table: Dict[str, Dict[str, Any]] = {}
    selected_oracle_alias: Optional[str] = None
    selected_alias_fallback: Optional[bool] = None
    oracle_failure_reason: Optional[str] = None
    oracle_alias_resolution_status: Optional[str] = None

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        syncing_match = SYNC_TABLE_RE.match(line)
        if syncing_match:
            current_table = syncing_match.group("table")
            per_table.setdefault(current_table, {})
            continue

        counts_match = SYNC_COUNTS_RE.match(line)
        if counts_match and current_table:
            per_table.setdefault(current_table, {}).update(
                {
                    "source_rows": int(counts_match.group("source")),
                    "target_rows_before_sync": int(counts_match.group("target")),
                }
            )
            continue

        finished_match = SYNC_FINISHED_RE.match(line)
        if finished_match:
            table = finished_match.group("table")
            per_table.setdefault(table, {}).update(
                {"synced_rows": int(finished_match.group("synced"))}
            )
            continue

        selected_alias_match = SYNC_SELECTED_ALIAS_RE.match(line)
        if selected_alias_match:
            selected_oracle_alias = selected_alias_match.group("alias")
            continue

        selected_alias_fallback_match = SYNC_SELECTED_ALIAS_FALLBACK_RE.match(line)
        if selected_alias_fallback_match:
            selected_alias_fallback = (
                selected_alias_fallback_match.group("value") == "yes"
            )
            continue

        oracle_failure_reason_match = SYNC_ORACLE_FAILURE_REASON_RE.match(line)
        if oracle_failure_reason_match:
            oracle_failure_reason = oracle_failure_reason_match.group("reason")
            continue

        oracle_resolution_status_match = SYNC_ORACLE_RESOLUTION_STATUS_RE.match(line)
        if oracle_resolution_status_match:
            oracle_alias_resolution_status = oracle_resolution_status_match.group(
                "status"
            )

    total_synced_rows = sum(row.get("synced_rows", 0) for row in per_table.values())
    summary = {
        "table_count": len(per_table),
        "total_synced_rows": total_synced_rows,
        "tables": [
            {"table": table, **values}
            for table, values in sorted(per_table.items(), key=lambda item: item[0])
        ],
    }
    if selected_oracle_alias is not None:
        summary["selected_oracle_alias"] = selected_oracle_alias
    if selected_alias_fallback is not None:
        summary["selected_alias_fallback"] = selected_alias_fallback
    if oracle_failure_reason is not None:
        summary["oracle_failure_reason"] = oracle_failure_reason
    if oracle_alias_resolution_status is not None:
        summary["oracle_alias_resolution_status"] = oracle_alias_resolution_status
    return summary


def parse_ingest_stdout(stdout: str) -> Dict[str, Any]:
    current_table: Optional[str] = None
    per_table: Dict[str, Dict[str, Any]] = {}
    total_chunks: Optional[int] = None

    for raw_line in stdout.splitlines():
        line = raw_line.rstrip()
        table_match = INGEST_TABLE_RE.match(line)
        if table_match:
            current_table = table_match.group("table")
            per_table.setdefault(current_table, {})
            continue

        finished_match = INGEST_FINISHED_RE.match(line)
        if finished_match:
            table = finished_match.group("table")
            per_table.setdefault(table, {}).update(
                {
                    "chunks_written": int(finished_match.group("chunks")),
                    "batches": int(finished_match.group("batches")),
                    "embedding_calls": int(finished_match.group("embedding_calls")),
                    "sleep_seconds": float(finished_match.group("sleep_seconds")),
                    "parallel_engine": finished_match.group("engine").strip(),
                    "parallel_engine_fallbacks": int(finished_match.group("fallbacks")),
                }
            )
            current_table = table
            continue

        total_match = INGEST_TOTAL_RE.match(line.strip())
        if total_match:
            total_chunks = int(total_match.group("total"))
            continue

        if current_table and "총 " in line and "개 청크를 처리했습니다." in line:
            processed_match = re.search(
                r"총 (?P<count>\d+)개 청크를 처리했습니다\.", line
            )
            if processed_match:
                per_table.setdefault(current_table, {}).update(
                    {"processed_chunks": int(processed_match.group("count"))}
                )

    return {
        "table_count": len(per_table),
        "total_chunks_written": (
            total_chunks
            if total_chunks is not None
            else sum(row.get("chunks_written", 0) for row in per_table.values())
        ),
        "tables": [
            {"table": table, **values}
            for table, values in sorted(per_table.items(), key=lambda item: item[0])
        ],
    }


def _resolve_source_tables(tables: Iterable[str]) -> list[str]:
    source_tables: list[str] = []
    seen: set[str] = set()
    for table in tables:
        source_table = str(TABLE_PROFILES.get(table, {}).get("source_table", table))
        if source_table in seen:
            continue
        seen.add(source_table)
        source_tables.append(source_table)
    return source_tables


def count_missing_embeddings(
    *,
    database_url: str,
    ingest_tables: Sequence[str],
    season_year: int,
    sample_limit_per_table: int = 5,
) -> Dict[str, Any]:
    source_tables = _resolve_source_tables(ingest_tables)
    if not source_tables:
        return {"total_missing_embeddings": 0, "source_tables": [], "rows": []}

    count_query = """
        SELECT source_table, COUNT(*) AS missing_count
        FROM rag_chunks
        WHERE source_table = ANY(%s)
          AND embedding IS NULL
          AND (season_year = %s OR season_year IS NULL)
        GROUP BY source_table
        ORDER BY source_table
    """
    sample_query = """
        SELECT source_table, source_row_id, title, season_year, meta->>'source_profile' AS source_profile
        FROM (
            SELECT
                source_table,
                source_row_id,
                title,
                season_year,
                meta,
                ROW_NUMBER() OVER (
                    PARTITION BY source_table
                    ORDER BY source_row_id
                ) AS row_number
            FROM rag_chunks
            WHERE source_table = ANY(%s)
              AND embedding IS NULL
              AND (season_year = %s OR season_year IS NULL)
        ) sampled
        WHERE row_number <= %s
        ORDER BY source_table, source_row_id
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(count_query, (source_tables, season_year))
            count_rows = cur.fetchall()
            cur.execute(
                sample_query, (source_tables, season_year, sample_limit_per_table)
            )
            sample_rows = cur.fetchall()

    sample_rows_by_table: Dict[str, List[Dict[str, Any]]] = {}
    for (
        source_table,
        source_row_id,
        title,
        row_season_year,
        source_profile,
    ) in sample_rows:
        sample_rows_by_table.setdefault(str(source_table), []).append(
            {
                "source_row_id": str(source_row_id),
                "title": title,
                "season_year": row_season_year,
                "source_profile": source_profile,
            }
        )

    rows = [
        {
            "source_table": source_table,
            "missing_count": int(missing_count),
            "sample_rows": sample_rows_by_table.get(str(source_table), []),
        }
        for source_table, missing_count in count_rows
    ]
    return {
        "source_tables": source_tables,
        "total_missing_embeddings": sum(row["missing_count"] for row in rows),
        "rows": rows,
    }


def _build_step_payload(
    step: Optional[StepRunResult], skipped: bool = False
) -> Dict[str, Any]:
    if skipped:
        return {"status": "skipped"}
    if step is None:
        return {"status": "not_run"}
    return {
        "status": step.status,
        "exit_code": step.exit_code,
        "duration_ms": step.duration_ms,
        "cwd": step.cwd,
        "command": step.command,
        "stdout_path": step.stdout_path,
        "stderr_path": step.stderr_path,
    }


def build_final_report(
    *,
    args: argparse.Namespace,
    steps: Dict[str, Optional[StepRunResult]],
    sync_summary: Optional[Dict[str, Any]],
    oracle_diagnostics: Optional[Dict[str, Any]],
    ingest_summary: Optional[Dict[str, Any]],
    missing_embeddings: Optional[Dict[str, Any]],
    coverage_report: Optional[Dict[str, Any]],
    benchmark_report: Optional[Dict[str, Any]],
    smoke_summary_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    coverage_summary = (coverage_report or {}).get("summary") or {}
    benchmark_summary = ((benchmark_report or {}).get("summary") or {}).get(
        "overall"
    ) or {}
    smoke_summary = (smoke_summary_report or {}).get("summary") or {}
    oracle_remediation = build_oracle_remediation(
        args=args,
        sync_summary=sync_summary,
        oracle_diagnostics=oracle_diagnostics,
    )

    readiness_checks = {
        "sync_ok": args.skip_sync
        or (steps.get("sync") is not None and steps["sync"].exit_code == 0),
        "ingest_ok": args.skip_ingest
        or (steps.get("ingest") is not None and steps["ingest"].exit_code == 0),
        "coverage_ok": args.skip_coverage
        or (
            steps.get("coverage") is not None
            and steps["coverage"].exit_code == 0
            and coverage_summary.get("total_missing_count", 0) == 0
            and coverage_summary.get("total_extra_count", 0) == 0
        ),
        "benchmark_ok": args.skip_benchmark
        or (
            steps.get("benchmark") is not None
            and steps["benchmark"].exit_code == 0
            and bool(benchmark_summary.get("acceptance", {}).get("passed", False))
        ),
        "smoke_ok": args.skip_smoke
        or (
            steps.get("smoke") is not None
            and steps["smoke"].exit_code == 0
            and smoke_summary.get("failed", 0) == 0
            and bool(smoke_summary.get("stream_fallback_ratio_ok", True))
        ),
        "missing_embeddings_ok": missing_embeddings is None
        or missing_embeddings.get("total_missing_embeddings", 0) == 0,
    }
    ready = all(readiness_checks.values())

    failures = [name for name, passed in readiness_checks.items() if not passed]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "season_year": args.season_year,
            "since": args.since or None,
            "sync_tables": args.sync_tables,
            "ingest_tables": [
                table for table in args.ingest_tables if table != "rag_chunks"
            ],
            "parallel_engine": args.parallel_engine,
            "workers": args.workers,
            "read_batch_size": args.read_batch_size,
            "embed_batch_size": args.embed_batch_size,
            "max_concurrency": args.max_concurrency,
            "commit_interval": args.commit_interval,
            "smoke_batch_size": args.smoke_batch_size,
            "benchmark_limit": args.benchmark_limit,
        },
        "readiness": {
            "ready": ready,
            "checks": readiness_checks,
            "failure_reasons": failures,
        },
        "steps": {
            "sync": _build_step_payload(steps.get("sync"), skipped=args.skip_sync),
            "ingest": _build_step_payload(
                steps.get("ingest"), skipped=args.skip_ingest
            ),
            "coverage": _build_step_payload(
                steps.get("coverage"),
                skipped=args.skip_coverage,
            ),
            "benchmark": _build_step_payload(
                steps.get("benchmark"),
                skipped=args.skip_benchmark,
            ),
            "smoke": _build_step_payload(steps.get("smoke"), skipped=args.skip_smoke),
        },
        "sync": sync_summary,
        "oracle_diagnostics": oracle_diagnostics,
        "oracle_remediation": oracle_remediation,
        "ingest": ingest_summary,
        "embeddings": missing_embeddings,
        "coverage": (
            {
                "summary": coverage_summary,
                "rows_with_gaps": [
                    row
                    for row in (coverage_report or {}).get("rows", [])
                    if row.get("missing_count", 0) > 0 or row.get("extra_count", 0) > 0
                ][:20],
            }
            if coverage_report
            else None
        ),
        "benchmark": (
            {
                "summary": benchmark_summary,
            }
            if benchmark_report
            else None
        ),
        "smoke": (
            {
                "summary": smoke_summary,
            }
            if smoke_summary_report
            else None
        ),
    }


def _resolved_source_db_url(override: str) -> str:
    if override.strip():
        return override.strip()
    settings = get_settings()
    return settings.source_db_url


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output_path()
    )
    artifact_dir = output_path.parent / output_path.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    source_db_url = _resolved_source_db_url(args.source_db_url)
    ingest_tables = [table for table in args.ingest_tables if table != "rag_chunks"]
    final_steps: Dict[str, Optional[StepRunResult]] = {
        "sync": None,
        "ingest": None,
        "coverage": None,
        "benchmark": None,
        "smoke": None,
    }

    sync_summary: Optional[Dict[str, Any]] = None
    oracle_diagnostics: Optional[Dict[str, Any]] = None
    ingest_summary: Optional[Dict[str, Any]] = None
    missing_embeddings: Optional[Dict[str, Any]] = None
    coverage_report: Optional[Dict[str, Any]] = None
    benchmark_report: Optional[Dict[str, Any]] = None
    smoke_summary_report: Optional[Dict[str, Any]] = None

    coverage_output = artifact_dir / "coverage.json"
    benchmark_output = artifact_dir / "benchmark.json"
    smoke_output = artifact_dir / "smoke.json"
    smoke_summary_output = artifact_dir / "smoke.summary.json"

    if not args.skip_sync:
        sync_command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "sync_kbo_data.py"),
            "--tables",
            ",".join(args.sync_tables),
            "--batch-size",
            str(max(1, args.sync_batch_size)),
            "--limit",
            str(max(0, args.sync_limit)),
            "--wallet-dir",
            args.wallet_dir,
            "--oracle-service-name",
            args.oracle_service_name,
            "--oracle-timeout-seconds",
            str(max(1, args.oracle_timeout_seconds)),
        ]
        final_steps["sync"] = _run_command(
            name="sync",
            command=sync_command,
            cwd=REPO_ROOT,
            artifact_dir=artifact_dir,
        )
        sync_summary = parse_sync_stdout(_read_text(final_steps["sync"].stdout_path))
        if final_steps["sync"].exit_code != 0:
            oracle_diagnostics_command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "sync_kbo_data.py"),
                "--check-oracle-services-direct",
                "--wallet-dir",
                args.wallet_dir,
                "--oracle-timeout-seconds",
                str(max(1, args.oracle_timeout_seconds)),
            ]
            oracle_diagnostics_step = _run_command(
                name="sync_oracle_diagnostics",
                command=oracle_diagnostics_command,
                cwd=REPO_ROOT,
                artifact_dir=artifact_dir,
            )
            oracle_diagnostics = parse_oracle_diagnostics_stdout(
                _read_text(oracle_diagnostics_step.stdout_path)
            )
            oracle_diagnostics["step"] = _build_step_payload(oracle_diagnostics_step)

    if not args.skip_ingest:
        ingest_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ingest_from_kbo.py"),
            "--tables",
            *ingest_tables,
            "--season-year",
            str(args.season_year),
            "--source-db-url",
            source_db_url,
            "--read-batch-size",
            str(max(1, args.read_batch_size)),
            "--embed-batch-size",
            str(max(1, args.embed_batch_size)),
            "--max-concurrency",
            str(max(1, args.max_concurrency)),
            "--commit-interval",
            str(max(1, args.commit_interval)),
            "--parallel-engine",
            args.parallel_engine,
            "--workers",
            str(max(1, args.workers)),
        ]
        if args.limit > 0:
            ingest_command.extend(["--limit", str(args.limit)])
        if args.since.strip():
            ingest_command.extend(["--since", args.since.strip()])
        if args.no_embed:
            ingest_command.append("--no-embed")
        if args.use_legacy_renderer:
            ingest_command.append("--use-legacy-renderer")
        final_steps["ingest"] = _run_command(
            name="ingest",
            command=ingest_command,
            cwd=PROJECT_ROOT,
            artifact_dir=artifact_dir,
        )
        ingest_summary = parse_ingest_stdout(
            _read_text(final_steps["ingest"].stdout_path)
        )

    if not args.skip_coverage:
        coverage_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "verify_embedding_coverage.py"),
            "--start-year",
            str(args.season_year),
            "--end-year",
            str(args.season_year),
            "--mode",
            args.coverage_mode,
            "--sample-limit",
            str(max(0, args.coverage_sample_limit)),
            "--output",
            str(coverage_output),
        ]
        final_steps["coverage"] = _run_command(
            name="coverage",
            command=coverage_command,
            cwd=PROJECT_ROOT,
            artifact_dir=artifact_dir,
        )
        coverage_report = _load_json(coverage_output)

    if not args.skip_benchmark:
        benchmark_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "benchmark_retrieval.py"),
            "--limit",
            str(max(1, args.benchmark_limit)),
            "--variant",
            "both",
            "--output",
            str(benchmark_output),
        ]
        final_steps["benchmark"] = _run_command(
            name="benchmark",
            command=benchmark_command,
            cwd=PROJECT_ROOT,
            artifact_dir=artifact_dir,
        )
        benchmark_report = _load_json(benchmark_output)

    if not args.skip_smoke:
        smoke_command = _build_smoke_command(
            args,
            smoke_output=smoke_output,
            smoke_summary_output=smoke_summary_output,
            artifact_dir=artifact_dir,
        )
        final_steps["smoke"] = _run_command(
            name="smoke",
            command=smoke_command,
            cwd=PROJECT_ROOT,
            artifact_dir=artifact_dir,
        )
        smoke_summary_report = _load_json(smoke_summary_output)

    try:
        missing_embeddings = count_missing_embeddings(
            database_url=get_settings().database_url,
            ingest_tables=ingest_tables,
            season_year=args.season_year,
        )
    except Exception as exc:
        missing_embeddings = {
            "error": str(exc),
            "source_tables": _resolve_source_tables(ingest_tables),
            "total_missing_embeddings": None,
        }

    final_report = build_final_report(
        args=args,
        steps=final_steps,
        sync_summary=sync_summary,
        oracle_diagnostics=oracle_diagnostics,
        ingest_summary=ingest_summary,
        missing_embeddings=missing_embeddings,
        coverage_report=coverage_report,
        benchmark_report=benchmark_report,
        smoke_summary_report=smoke_summary_report,
    )
    artifacts: Dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "report_path": str(output_path),
    }
    if final_report.get("oracle_remediation"):
        oracle_escalation_path = artifact_dir / "oracle-escalation.md"
        _write_text(
            oracle_escalation_path,
            build_oracle_escalation_markdown(
                report=final_report,
                report_path=output_path,
            ),
        )
        artifacts["oracle_escalation_markdown"] = str(oracle_escalation_path)
    handoff_path = artifact_dir / "handoff.md"
    artifacts["handoff_markdown"] = str(handoff_path)
    artifacts["bundle_manifest"] = str(artifact_dir / "bundle-manifest.json")
    artifacts["support_bundle"] = str(artifact_dir / "support-bundle.tar.gz")
    artifacts["latest_pointer"] = str(output_path.parent / "latest.json")
    artifacts["latest_handoff_markdown"] = str(output_path.parent / "latest-handoff.md")
    final_report["artifacts"] = artifacts
    handoff_markdown = build_readiness_handoff_markdown(
        report=final_report,
        report_path=output_path,
    )
    _write_text(handoff_path, handoff_markdown)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_support_bundle(
        artifact_dir=artifact_dir,
        report_path=output_path,
    )
    write_latest_pointer_files(
        report_dir=output_path.parent,
        report_path=output_path,
        handoff_markdown=handoff_markdown,
        report=final_report,
    )
    print(
        json.dumps(
            {"output": str(output_path), "readiness": final_report["readiness"]},
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if final_report["readiness"]["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
