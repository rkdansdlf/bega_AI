#!/usr/bin/env python3
"""Run the read-only P0 operator-data filled packet intake pipeline.

The runner orchestrates existing preflight/validation/dry-run/gate scripts.
It never fills baseball data, crawls, infers missing values, or applies DB
writes.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import audit_operator_data_p0_input_packet as audit
from scripts import check_operator_data_p0_db_prereqs as db_prereqs
from scripts import ingest_operator_data_handoff as ingest
from scripts import operator_data_recovery_gate as recovery_gate
from scripts import summarize_operator_data_p0_recovery_status as status_summary
from scripts import validate_operator_data_handoff as validation


DEFAULT_QUEUE_INPUT = audit.DEFAULT_QUEUE_INPUT
DEFAULT_FIELDS_INPUT = audit.DEFAULT_FIELDS_INPUT
DEFAULT_SOURCE_QUEUE_INPUT = audit.DEFAULT_SOURCE_QUEUE_INPUT
DEFAULT_SOURCE_FIELDS_INPUT = audit.DEFAULT_SOURCE_FIELDS_INPUT
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "reports" / "operator_data_p0_filled_intake" / "post_db_fast_path_docker_kbo500"
)
STAGE_FIELDNAMES = ["name", "status", "code", "message", "output_dir"]


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_exception(exc: Exception) -> str:
    return exc.__class__.__name__


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"required input is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _stage_result(
    *,
    name: str,
    status: str,
    output_dir: Path,
    code: str = "",
    message: str = "",
) -> dict[str, str]:
    return {
        "name": name,
        "status": status,
        "code": code,
        "message": message,
        "output_dir": str(output_dir),
    }


def _issue_records(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    records = payload.get(key)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return []
    return [record for record in records if isinstance(record, Mapping)]


def _stage_issue_detail(payload: Mapping[str, Any]) -> tuple[str, str]:
    records = [
        *_issue_records(payload, "issues"),
        *_issue_records(payload, "blockers"),
    ]
    for preferred_severity in ("error", "blocker", "warning"):
        for record in records:
            if _normalize_text(record.get("severity")) == preferred_severity:
                return (
                    _normalize_text(record.get("code")),
                    _normalize_text(record.get("message")),
                )
    if records:
        record = records[0]
        return (
            _normalize_text(record.get("code")),
            _normalize_text(record.get("message")),
        )
    return "", ""


def _stage_result_from_report(
    *,
    name: str,
    status: str,
    output_dir: Path,
    report: Mapping[str, Any],
) -> dict[str, str]:
    code, message = _stage_issue_detail(report)
    return _stage_result(
        name=name,
        status=status,
        code=code,
        message=message,
        output_dir=output_dir,
    )


def _snapshot_packet(
    *,
    queue_path: Path,
    fields_path: Path,
    source_queue_path: Path,
    source_fields_path: Path,
    output_dir: Path,
    db_url_present: bool,
    allow_empty_ready: bool,
) -> dict[str, Any]:
    queue_fields, queue_rows = _read_csv(queue_path)
    field_fields, field_rows = _read_csv(fields_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_queue = output_dir / "p0_queue.csv"
    snapshot_fields = output_dir / "p0_fields.csv"
    shutil.copyfile(queue_path, snapshot_queue)
    shutil.copyfile(fields_path, snapshot_fields)

    domain_counts = Counter(_normalize_text(row.get("domain")) for row in queue_rows)
    status_counts = Counter(_normalize_text(row.get("operator_status")) for row in queue_rows)
    non_p0_domains = sorted(set(domain_counts) - audit.P0_DOMAINS - {""})
    summary = {
        "generated_at_utc": _now_utc(),
        "total_queue_items": len(queue_rows),
        "total_field_rows": len(field_rows),
        "domain_counts": {domain: domain_counts.get(domain, 0) for domain in audit.P0_DOMAIN_ORDER},
        "status_counts": dict(sorted(status_counts.items())),
        "non_p0_domains": non_p0_domains,
        "source_queue_path": str(source_queue_path),
        "source_fields_path": str(source_fields_path),
        "filled_queue_path": str(queue_path),
        "filled_fields_path": str(fields_path),
        "snapshot_queue_path": str(snapshot_queue),
        "snapshot_fields_path": str(snapshot_fields),
        "db_url_present": db_url_present,
        "allow_empty_ready": allow_empty_ready,
        "queue_header_valid": queue_fields == audit.QUEUE_FIELDNAMES,
        "fields_header_valid": field_fields == audit.FIELDS_FIELDNAMES,
    }
    manifest = {
        "generated_at_utc": summary["generated_at_utc"],
        "input_files": {
            "queue": str(queue_path),
            "fields": str(fields_path),
            "source_queue": str(source_queue_path),
            "source_fields": str(source_fields_path),
        },
        "snapshot_files": {
            "queue": str(snapshot_queue),
            "fields": str(snapshot_fields),
        },
        "db_url_present": db_url_present,
        "allow_empty_ready": allow_empty_ready,
    }
    _write_json(output_dir / "p0_input_summary.json", summary)
    _write_json(output_dir / "intake_packet_manifest.json", manifest)
    return summary


def _write_audit_stage_error(output_dir: Path, *, code: str, message: str) -> dict[str, Any]:
    report = {
        "summary": {
            "generated_at_utc": _now_utc(),
            "status": "fail",
            "require_ready": True,
            "total_queue_items": 0,
            "total_field_rows": 0,
            "ready_or_validated_count": 0,
            "recovery_candidate_count": 0,
            "manual_fallback_count": 0,
            "blocked_ready_count": 0,
            "issue_counts": {"error": 1, "warning": 0},
        },
        "issues": [
            {
                "severity": "error",
                "code": code,
                "message": message,
                "queue_id": "",
                "domain": "",
                "field_name": "",
                "source": "filled_intake_runner",
            }
        ],
        "readiness_plan": [],
    }
    _write_json(output_dir / "p0_input_audit_summary.json", report)
    _write_csv(output_dir / "p0_input_audit_issues.csv", report["issues"], audit.ISSUE_FIELDNAMES)
    _write_csv(output_dir / "p0_input_readiness_plan.csv", [], audit.READINESS_FIELDNAMES)
    (output_dir / "p0_input_audit_handoff.md").write_text(
        "# P0 Operator Input Packet Audit\n\n- Status: `fail`\n\n"
        f"## Issues\n\n- `{code}`: {message}\n",
        encoding="utf-8",
    )
    return report


def _write_db_prereq_stage_error(output_dir: Path, *, code: str, message: str) -> dict[str, Any]:
    report = {
        "summary": {
            "generated_at_utc": _now_utc(),
            "status": "fail",
            "checked_table_count": 0,
            "missing_column_count": 0,
            "lineup_conflict_target_exists": False,
            "issue_counts": {"error": 1, "warning": 0},
        },
        "table_results": [],
        "issues": [
            {
                "severity": "error",
                "code": code,
                "message": message,
                "table_name": "",
                "missing_columns": "",
            }
        ],
    }
    _write_json(output_dir / "db_prereq_summary.json", report)
    _write_csv(output_dir / "db_prereq_issues.csv", report["issues"], db_prereqs.ISSUE_FIELDNAMES)
    _write_csv(
        output_dir / "db_prereq_tables.csv",
        [],
        [
            "table_name",
            "required_column_count",
            "available_column_count",
            "missing_column_count",
            "missing_columns",
            "status",
        ],
    )
    (output_dir / "db_prereq_handoff.md").write_text(
        "# P0 Operator Data DB Prerequisites\n\n- Status: `fail`\n\n"
        f"## Issues\n\n- `{code}`: {message}\n",
        encoding="utf-8",
    )
    return report


def _write_validation_stage_error(output_dir: Path, *, code: str, message: str) -> dict[str, Any]:
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "fail",
        "total_queue_items": 0,
        "total_field_rows": 0,
        "normalized_row_count": 0,
        "apply_plan_row_count": 0,
        "issue_counts": {"error": 1, "warning": 0},
        "priority_counts": {},
        "domain_counts": {},
        "operator_status_counts": {},
        "validation_status_counts": {},
        "apply_eligible_count": 0,
        "db_checks": {"skipped": True, "skip_reason": code},
    }
    issue = {
        "queue_id": "",
        "domain": "",
        "field_name": "",
        "severity": "error",
        "code": code,
        "message": message,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "operator_data_validation_summary.json", summary)
    _write_csv(output_dir / "operator_data_validation_issues.csv", [issue], validation.ISSUE_FIELDNAMES)
    (output_dir / "operator_data_normalized_rows.jsonl").write_text("", encoding="utf-8")
    _write_csv(output_dir / "operator_data_apply_plan.csv", [], validation.APPLY_PLAN_FIELDNAMES)
    return {"summary": summary, "issues": [issue], "normalized_rows": [], "apply_plan_rows": []}


def _write_ingest_stage_error(output_dir: Path, *, code: str, message: str) -> dict[str, Any]:
    summary = {
        "generated_at_utc": _now_utc(),
        "dry_run": True,
        "selected_domains": list(ingest.P0_DOMAINS),
        "total_rows": 0,
        "eligible_rows": 0,
        "applied_rows": 0,
        "issue_counts": {"error": 1, "warning": 0},
        "action_counts": {},
        "domain_counts": {},
        "starter_plan_count": 0,
    }
    issue = {"queue_id": "", "domain": "", "severity": "error", "code": code, "message": message}
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "operator_data_ingest_summary.json", summary)
    _write_csv(output_dir / "operator_data_ingest_plan.csv", [], ingest.PLAN_FIELDNAMES)
    _write_csv(output_dir / "operator_data_ingest_issues.csv", [issue], ingest.ISSUE_FIELDNAMES)
    _write_csv(output_dir / "operator_data_starter_plan.csv", [], ingest.STARTER_FIELDNAMES)
    return {"summary": summary, "plans": [], "issues": [issue], "starter_plan_rows": []}


def _write_gate_stage_error(output_dir: Path, *, code: str, message: str) -> dict[str, Any]:
    report = {
        "summary": {
            "generated_at_utc": _now_utc(),
            "status": "fail",
            "validation_dir": "",
            "ingest_dir": "",
            "apply_eligible_count": 0,
            "validation_error_count": 0,
            "ingest_error_count": 0,
            "issue_counts": {"error": 1},
            "ingest_action_counts": {},
        },
        "issues": [
            {
                "severity": "error",
                "code": code,
                "message": message,
                "source": "filled_intake_runner",
                "queue_id": "",
                "domain": "",
            }
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", report)
    _write_csv(output_dir / "issues.csv", report["issues"], ["severity", "code", "message", "source", "queue_id", "domain"])
    (output_dir / "handoff.md").write_text(
        "# Operator Data Recovery Gate\n\n- Status: `fail`\n\n"
        f"## Blocking Issues\n\n- `{code}`: {message}\n",
        encoding="utf-8",
    )
    return report


def _run_validation_stage(
    *,
    queue_path: Path,
    fields_path: Path,
    db_url: str,
    output_dir: Path,
) -> dict[str, Any]:
    db_checker = None
    try:
        db_checker = validation._build_db_checker(
            db_url=db_url,
            skip_db_checks=False,
        )
        return validation.validate_files(
            queue_path=queue_path,
            fields_path=fields_path,
            output_dir=output_dir,
            db_checker=db_checker,
        )
    finally:
        if db_checker is not None:
            db_checker.close()


def _run_ingest_stage(
    *,
    normalized_path: Path,
    db_url: str,
    output_dir: Path,
) -> dict[str, Any]:
    rows = ingest.load_normalized_rows(normalized_path)
    domains = list(ingest.P0_DOMAINS)
    eligible_exists = any(
        ingest._normalize_text(row.get("domain")) in set(domains)
        and ingest._row_is_apply_candidate(row)
        for row in rows
    )
    conn = None
    try:
        if eligible_exists and _normalize_text(db_url):
            conn = ingest._connect(db_url)
        report = ingest.build_ingest_report(
            rows,
            conn=conn,
            apply=False,
            allow_overwrite=False,
            domains=domains,
        )
    finally:
        if conn is not None:
            conn.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "operator_data_ingest_summary.json", report["summary"])
    _write_csv(output_dir / "operator_data_ingest_plan.csv", report["plans"], ingest.PLAN_FIELDNAMES)
    _write_csv(output_dir / "operator_data_ingest_issues.csv", report["issues"], ingest.ISSUE_FIELDNAMES)
    _write_csv(output_dir / "operator_data_starter_plan.csv", report["starter_plan_rows"], ingest.STARTER_FIELDNAMES)
    return report


def _summary_status(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else payload
    return _normalize_text(summary.get("status") if isinstance(summary, Mapping) else "")


def _intake_handoff(report: Mapping[str, Any]) -> str:
    summary = report.get("summary") or {}
    stages = report.get("stages") or []
    final = report.get("final_status_summary") or {}
    blockers = report.get("blockers") or []
    lines = [
        "# P0 Filled Intake Runner",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Exit code: `{summary.get('exit_code', '')}`",
        f"- DB URL present: `{str(summary.get('db_url_present', False)).lower()}`",
        f"- Final recovery status: `{final.get('status', 'missing')}`",
        f"- Ready/validated rows: `{final.get('ready_or_validated_count', 0)}`",
        f"- Apply eligible rows: `{final.get('apply_eligible_count', 0)}`",
        f"- Manual required rows: `{final.get('manual_required_count', 0)}`",
        "",
        "## Stages",
        "",
    ]
    for stage in stages:
        lines.append(
            "- "
            f"`{stage.get('name')}` status=`{stage.get('status')}` "
            f"code=`{stage.get('code', '')}`: {stage.get('message', '')}"
        )
    if blockers:
        lines.extend(["", "## Blocking Conditions", ""])
        for blocker in blockers:
            lines.append(
                "- "
                f"`{blocker.get('code')}` "
                f"source=`{blocker.get('source')}`: {blocker.get('message')}"
            )
    if int(final.get("manual_required_count") or 0) > 0:
        domain_counts = final.get("manual_required_domain_counts") or {}
        reason_counts = final.get("manual_required_skip_reason_counts") or {}
        format_counts = lambda counts: ", ".join(
            f"{key}={value}"
            for key, value in counts.items()
            if int(value or 0) > 0
        ) or "none"
        lines.extend(
            [
                "",
                "## MANUAL_BASEBALL_DATA_REQUIRED",
                "",
                "These P0 rows still require operator-provided data and must remain on the manual contract path.",
                "",
                f"- By domain: `{format_counts(domain_counts)}`",
                f"- By reason: `{format_counts(reason_counts)}`",
                "- Detail CSV: `gate/manual_baseball_data_required_rows.csv`",
            ]
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "If blocked, fix the listed conditions and rerun the intake. Do not apply DB writes from this runner.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _write_intake_report(output_dir: Path, report: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "intake_summary.json", report)
    _write_csv(output_dir / "intake_stages.csv", report.get("stages", []), STAGE_FIELDNAMES)
    (output_dir / "intake_handoff.md").write_text(_intake_handoff(report), encoding="utf-8")


def _write_fatal_intake_report(
    *,
    output_dir: Path,
    code: str,
    message: str,
    db_url_present: bool = False,
) -> dict[str, Any]:
    report = {
        "summary": {
            "generated_at_utc": _now_utc(),
            "status": "fatal",
            "exit_code": 2,
            "db_url_present": db_url_present,
            "output_dir": str(output_dir),
            "stage_count": 0,
            "failed_stage_count": 1,
            "nonpassing_stage_count": 1,
        },
        "stages": [
            _stage_result(
                name="intake",
                status="fatal",
                code=code,
                message=message,
                output_dir=output_dir,
            )
        ],
        "final_status_summary": {"status": "fatal"},
        "blockers": [
            {
                "severity": "blocker",
                "code": code,
                "message": message,
                "source": "filled_intake_runner",
                "queue_id": "",
                "domain": "",
            }
        ],
    }
    _write_intake_report(output_dir, report)
    return report


def run_intake(
    *,
    queue_path: Path,
    fields_path: Path,
    source_queue_path: Path,
    source_fields_path: Path,
    db_url: str,
    output_dir: Path,
    allow_empty_ready: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    db_url_present = bool(_normalize_text(db_url))
    packet_dir = output_dir / "packet_snapshot"
    audit_dir = output_dir / "audit"
    db_prereq_dir = output_dir / "db_prereqs"
    validation_dir = output_dir / "validation"
    ingest_dir = output_dir / "ingest"
    gate_dir = output_dir / "gate"
    status_dir = output_dir / "status"
    stages: list[dict[str, str]] = []

    packet_summary = _snapshot_packet(
        queue_path=queue_path,
        fields_path=fields_path,
        source_queue_path=source_queue_path,
        source_fields_path=source_fields_path,
        output_dir=packet_dir,
        db_url_present=db_url_present,
        allow_empty_ready=allow_empty_ready,
    )
    stages.append(
        _stage_result(
            name="packet_snapshot",
            status="pass",
            output_dir=packet_dir,
            message=f"{packet_summary.get('total_queue_items', 0)} queue row(s) snapshotted.",
        )
    )

    try:
        audit_report = audit.run_audit(
            queue_path=packet_dir / "p0_queue.csv",
            fields_path=packet_dir / "p0_fields.csv",
            source_queue_path=source_queue_path,
            source_fields_path=source_fields_path,
            output_dir=audit_dir,
            require_ready=not allow_empty_ready,
        )
        stages.append(
            _stage_result_from_report(
                name="audit",
                status=_summary_status(audit_report),
                output_dir=audit_dir,
                report=audit_report,
            )
        )
    except Exception as exc:
        message = f"Audit stage failed ({_safe_exception(exc)})."
        _write_audit_stage_error(audit_dir, code="audit_stage_error", message=message)
        stages.append(
            _stage_result(
                name="audit",
                status="fail",
                code="audit_stage_error",
                message=message,
                output_dir=audit_dir,
            )
        )

    try:
        db_report = db_prereqs.run_check(db_url=db_url, output_dir=db_prereq_dir)
        stages.append(
            _stage_result_from_report(
                name="db_prereqs",
                status=_summary_status(db_report),
                output_dir=db_prereq_dir,
                report=db_report,
            )
        )
    except Exception as exc:
        message = f"DB prerequisite stage failed ({_safe_exception(exc)})."
        _write_db_prereq_stage_error(db_prereq_dir, code="db_prereq_stage_error", message=message)
        stages.append(
            _stage_result(
                name="db_prereqs",
                status="fail",
                code="db_prereq_stage_error",
                message=message,
                output_dir=db_prereq_dir,
            )
        )

    try:
        validation_report = _run_validation_stage(
            queue_path=packet_dir / "p0_queue.csv",
            fields_path=packet_dir / "p0_fields.csv",
            db_url=db_url,
            output_dir=validation_dir,
        )
        stages.append(
            _stage_result_from_report(
                name="validation",
                status=_summary_status(validation_report["summary"]),
                output_dir=validation_dir,
                report=validation_report,
            )
        )
    except Exception as exc:
        message = f"Validation stage failed ({_safe_exception(exc)})."
        _write_validation_stage_error(validation_dir, code="validation_stage_error", message=message)
        stages.append(
            _stage_result(
                name="validation",
                status="fail",
                code="validation_stage_error",
                message=message,
                output_dir=validation_dir,
            )
        )

    try:
        ingest_report = _run_ingest_stage(
            normalized_path=validation_dir / "operator_data_normalized_rows.jsonl",
            db_url=db_url,
            output_dir=ingest_dir,
        )
        stages.append(
            _stage_result_from_report(
                name="ingest_dry_run",
                status="fail" if (ingest_report["summary"]["issue_counts"]["error"] > 0) else "pass",
                output_dir=ingest_dir,
                report=ingest_report,
            )
        )
    except Exception as exc:
        message = f"Ingest dry-run stage failed ({_safe_exception(exc)})."
        _write_ingest_stage_error(ingest_dir, code="ingest_stage_error", message=message)
        stages.append(
            _stage_result(
                name="ingest_dry_run",
                status="fail",
                code="ingest_stage_error",
                message=message,
                output_dir=ingest_dir,
            )
        )

    try:
        gate_report = recovery_gate.run_gate(
            validation_dir=validation_dir,
            ingest_dir=ingest_dir,
            output_dir=gate_dir,
        )
        stages.append(
            _stage_result_from_report(
                name="recovery_gate",
                status=_summary_status(gate_report),
                output_dir=gate_dir,
                report=gate_report,
            )
        )
    except Exception as exc:
        message = f"Recovery gate stage failed ({_safe_exception(exc)})."
        _write_gate_stage_error(gate_dir, code="gate_stage_error", message=message)
        stages.append(
            _stage_result(
                name="recovery_gate",
                status="fail",
                code="gate_stage_error",
                message=message,
                output_dir=gate_dir,
            )
        )

    try:
        status_report = status_summary.run_summary(
            packet_dir=packet_dir,
            audit_dir=audit_dir,
            db_prereq_dir=db_prereq_dir,
            validation_dir=validation_dir,
            ingest_dir=ingest_dir,
            gate_dir=gate_dir,
            output_dir=status_dir,
        )
        stages.append(
            _stage_result_from_report(
                name="status_summary",
                status=_summary_status(status_report),
                output_dir=status_dir,
                report=status_report,
            )
        )
    except Exception as exc:
        message = f"Status summary stage failed ({_safe_exception(exc)})."
        status_report = {
            "summary": {
                "generated_at_utc": _now_utc(),
                "status": "blocked",
                "blocker_count": 1,
                "blocker_codes": ["status_stage_error"],
            },
            "blockers": [
                {
                    "severity": "blocker",
                    "code": "status_stage_error",
                    "message": message,
                    "source": "filled_intake_runner",
                    "queue_id": "",
                    "domain": "",
                }
            ],
        }
        status_dir.mkdir(parents=True, exist_ok=True)
        _write_json(status_dir / "p0_recovery_status_summary.json", status_report)
        _write_csv(
            status_dir / "p0_recovery_status_blockers.csv",
            status_report["blockers"],
            status_summary.BLOCKER_FIELDNAMES,
        )
        (status_dir / "p0_recovery_status_handoff.md").write_text(
            "# P0 Operator Data Recovery Status\n\n- Status: `blocked`\n\n"
            f"## Blocking Conditions\n\n- `status_stage_error`: {message}\n",
            encoding="utf-8",
        )
        stages.append(
            _stage_result(
                name="status_summary",
                status="fail",
                code="status_stage_error",
                message=message,
                output_dir=status_dir,
            )
        )

    final_summary = status_report.get("summary") or {}
    final_status = _normalize_text(final_summary.get("status"))
    exit_code = 0 if final_status == "ready_for_controlled_apply" else 1
    failed_stage_count = sum(1 for stage in stages if stage.get("status") in {"fail", "fatal"})
    nonpassing_stage_count = sum(1 for stage in stages if stage.get("status") != "pass")
    intake_report = {
        "summary": {
            "generated_at_utc": _now_utc(),
            "status": final_status or "blocked",
            "exit_code": exit_code,
            "db_url_present": db_url_present,
            "allow_empty_ready": allow_empty_ready,
            "output_dir": str(output_dir),
            "stage_count": len(stages),
            "failed_stage_count": failed_stage_count,
            "nonpassing_stage_count": nonpassing_stage_count,
            "final_status_output_dir": str(status_dir),
        },
        "stages": stages,
        "final_status_summary": final_summary,
        "blockers": status_report.get("blockers", []),
    }
    _write_intake_report(output_dir, intake_report)
    return intake_report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run read-only P0 filled intake pipeline.")
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE_INPUT))
    parser.add_argument("--fields", default=str(DEFAULT_FIELDS_INPUT))
    parser.add_argument("--source-queue", default=str(DEFAULT_SOURCE_QUEUE_INPUT))
    parser.add_argument("--source-fields", default=str(DEFAULT_SOURCE_FIELDS_INPUT))
    parser.add_argument("--db-url", default="", help="PostgreSQL URL. Defaults to POSTGRES_DB_URL.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--allow-empty-ready",
        action="store_true",
        help="Allow pending-only packets to complete evidence generation; final status remains blocked.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    db_url = _normalize_text(args.db_url) or os.environ.get("POSTGRES_DB_URL", "")
    try:
        report = run_intake(
            queue_path=Path(args.queue),
            fields_path=Path(args.fields),
            source_queue_path=Path(args.source_queue),
            source_fields_path=Path(args.source_fields),
            db_url=db_url,
            output_dir=output_dir,
            allow_empty_ready=bool(args.allow_empty_ready),
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        message = f"Could not read required input or write output ({_safe_exception(exc)})."
        report = _write_fatal_intake_report(
            output_dir=output_dir,
            code="intake_io_error",
            message=message,
            db_url_present=bool(_normalize_text(db_url)),
        )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return int(report["summary"].get("exit_code") or 1)


if __name__ == "__main__":
    raise SystemExit(main())
