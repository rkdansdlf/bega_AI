#!/usr/bin/env python3
"""Summarize P0 operator-data recovery readiness from existing reports.

This script is read-only. It only collates packet/audit/validation/ingest/gate
artifacts into a single handoff summary for post-KBO500 recovery review.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET_DIR = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_operator_packet"
    / "post_db_fast_path_docker_kbo500"
)
DEFAULT_AUDIT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_operator_packet_audit"
    / "post_db_fast_path_docker_kbo500"
)
DEFAULT_DB_PREREQ_DIR = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_db_prereqs"
    / "post_db_fast_path_docker_kbo500"
)
DEFAULT_VALIDATION_DIR = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_validation"
    / "p0_packet_post_db_fast_path_docker_kbo500"
)
DEFAULT_INGEST_DIR = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_ingest"
    / "p0_packet_post_db_fast_path_docker_kbo500"
)
DEFAULT_GATE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_recovery_gate"
    / "p0_packet_post_db_fast_path_docker_kbo500"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_p0_recovery_status"
    / "post_db_fast_path_docker_kbo500"
)
BLOCKER_FIELDNAMES = ["severity", "code", "message", "source", "queue_id", "domain"]
P0_DOMAIN_ORDER = ("season_meta", "schedule_window", "game_day_lineup", "roster_news")


@dataclass(frozen=True)
class StatusBlocker:
    severity: str
    code: str
    message: str
    source: str
    queue_id: str = ""
    domain: str = ""

    def to_record(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "queue_id": self.queue_id,
            "domain": self.domain,
        }


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required report is missing: {path}")
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"report must be a JSON object: {path}")
    return dict(payload)


def _read_csv(path: Path, *, required: bool = True) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required report is missing: {path}")
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    nested = payload.get("summary")
    if isinstance(nested, Mapping):
        return dict(nested)
    return dict(payload)


def _as_int(value: Any) -> int:
    try:
        return int(str(value or "0").strip())
    except (TypeError, ValueError):
        return 0


def _issue_count(summary: Mapping[str, Any], severity: str) -> int:
    counts = summary.get("issue_counts") or {}
    return _as_int(counts.get(severity) if isinstance(counts, Mapping) else 0)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _count_rows(
    rows: Sequence[Mapping[str, str]],
    field_name: str,
    *,
    order: Sequence[str] = (),
) -> dict[str, int]:
    counts = Counter(_normalize_text(row.get(field_name)) for row in rows)
    if order:
        return {key: counts.get(key, 0) for key in order}
    return dict(sorted((key, count) for key, count in counts.items() if key))


def _summary_counts(
    summary: Mapping[str, Any],
    key: str,
    rows: Sequence[Mapping[str, str]],
    field_name: str,
    *,
    order: Sequence[str] = (),
) -> dict[str, int]:
    raw_counts = summary.get(key)
    if isinstance(raw_counts, Mapping):
        if order:
            return {item: _as_int(raw_counts.get(item)) for item in order}
        return {
            str(item_key): _as_int(item_value)
            for item_key, item_value in sorted(raw_counts.items())
        }
    return _count_rows(rows, field_name, order=order)


def _format_counts(counts: Mapping[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in counts.items() if _as_int(value) > 0]
    return ", ".join(parts) if parts else "none"


def _blocker(
    *,
    code: str,
    message: str,
    source: str,
    severity: str = "blocker",
    queue_id: str = "",
    domain: str = "",
) -> StatusBlocker:
    return StatusBlocker(
        severity=severity,
        code=code,
        message=message,
        source=source,
        queue_id=queue_id,
        domain=domain,
    )


def build_status_report(
    *,
    packet_summary: Mapping[str, Any],
    audit_report: Mapping[str, Any],
    validation_summary: Mapping[str, Any],
    ingest_summary: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    gate_issues: Sequence[Mapping[str, str]],
    packet_dir: Path,
    audit_dir: Path,
    validation_dir: Path,
    ingest_dir: Path,
    gate_dir: Path,
    db_prereq_report: Optional[Mapping[str, Any]] = None,
    db_prereq_dir: Optional[Path] = None,
    audit_issues: Sequence[Mapping[str, str]] = (),
    validation_issues: Sequence[Mapping[str, str]] = (),
    ingest_issues: Sequence[Mapping[str, str]] = (),
    gate_manual_required_rows: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    packet = _summary(packet_summary)
    audit = _summary(audit_report)
    db_prereq = _summary(db_prereq_report or {})
    validation = _summary(validation_summary)
    ingest = _summary(ingest_summary)
    gate = _summary(gate_report)
    manual_count_raw = gate.get("manual_required_count")
    manual_required_count = _as_int(manual_count_raw)
    if manual_count_raw is None and gate_manual_required_rows:
        manual_required_count = len(gate_manual_required_rows)
    manual_required_domain_counts = _summary_counts(
        gate,
        "manual_required_domain_counts",
        gate_manual_required_rows,
        "domain",
        order=P0_DOMAIN_ORDER,
    )
    manual_required_skip_reason_counts = _summary_counts(
        gate,
        "manual_required_skip_reason_counts",
        gate_manual_required_rows,
        "skip_reason",
    )
    blockers: list[StatusBlocker] = []

    if _as_int(packet.get("total_queue_items")) <= 0:
        blockers.append(
            _blocker(
                code="packet_empty",
                source="packet_summary",
                message="P0 input packet has no queue rows.",
            )
        )
    if str(audit.get("status") or "") == "fail":
        blockers.append(
            _blocker(
                code="packet_audit_failed",
                source="packet_audit",
                message="P0 input packet audit failed.",
            )
        )
    if _as_int(audit.get("ready_or_validated_count")) <= 0:
        blockers.append(
            _blocker(
                code="operator_input_missing",
                source="packet_audit",
                message="No P0 rows are marked ready_for_validation or validated.",
            )
        )
    if _as_int(audit.get("blocked_ready_count")) > 0:
        blockers.append(
            _blocker(
                code="ready_rows_blocked",
                source="packet_audit",
                message="One or more ready P0 rows have input QA errors.",
            )
        )
    for issue in audit_issues:
        if str(issue.get("severity") or "") != "error":
            continue
        blockers.append(
            _blocker(
                code=str(issue.get("code") or "packet_audit_issue"),
                source="packet_audit_issues",
                message=str(issue.get("message") or "P0 input packet audit issue."),
                queue_id=str(issue.get("queue_id") or ""),
                domain=str(issue.get("domain") or ""),
            )
        )
    if db_prereq:
        if str(db_prereq.get("status") or "") != "pass":
            blockers.append(
                _blocker(
                    code="db_prereqs_failed",
                    source="db_prereq_summary",
                    message="DB prerequisite checks have not passed.",
                )
            )
        db_prereq_issues = (
            (db_prereq_report or {}).get("issues")
            if isinstance(db_prereq_report, Mapping)
            else []
        )
        if isinstance(db_prereq_issues, Sequence):
            for issue in db_prereq_issues:
                if (
                    not isinstance(issue, Mapping)
                    or str(issue.get("severity") or "") != "error"
                ):
                    continue
                blockers.append(
                    _blocker(
                        code=str(issue.get("code") or "db_prereq_issue"),
                        source="db_prereq_summary",
                        message=str(issue.get("message") or "DB prerequisite issue."),
                    )
                )
    if _issue_count(validation, "error") > 0:
        blockers.append(
            _blocker(
                code="validation_errors",
                source="validation_summary",
                message=f"Validation reported {_issue_count(validation, 'error')} error(s).",
            )
        )
    for issue in validation_issues:
        if str(issue.get("severity") or "") != "error":
            continue
        blockers.append(
            _blocker(
                code=str(issue.get("code") or "validation_issue"),
                source="validation_issues",
                message=str(issue.get("message") or "Validation issue."),
                queue_id=str(issue.get("queue_id") or ""),
                domain=str(issue.get("domain") or ""),
            )
        )
    db_checks = validation.get("db_checks") or {}
    if isinstance(db_checks, Mapping) and db_checks.get("skipped") is True:
        blockers.append(
            _blocker(
                code="db_checks_skipped",
                source="validation_summary",
                message="Strict post-KBO500 recovery requires DB checks to run.",
            )
        )
    if _as_int(validation.get("apply_eligible_count")) <= 0:
        blockers.append(
            _blocker(
                code="no_apply_eligible_rows",
                source="validation_summary",
                message="No apply-eligible P0 rows are available.",
            )
        )
    if _issue_count(ingest, "error") > 0:
        blockers.append(
            _blocker(
                code="ingest_errors",
                source="ingest_summary",
                message=f"Ingest dry-run reported {_issue_count(ingest, 'error')} error(s).",
            )
        )
    for issue in ingest_issues:
        if str(issue.get("severity") or "") != "error":
            continue
        blockers.append(
            _blocker(
                code=str(issue.get("code") or "ingest_issue"),
                source="ingest_issues",
                message=str(issue.get("message") or "Ingest dry-run issue."),
                queue_id=str(issue.get("queue_id") or ""),
                domain=str(issue.get("domain") or ""),
            )
        )
    if str(gate.get("status") or "") != "pass":
        blockers.append(
            _blocker(
                code="recovery_gate_not_passed",
                source="recovery_gate",
                message="Recovery gate has not passed.",
            )
        )
    for issue in gate_issues:
        if str(issue.get("severity") or "") == "error":
            blockers.append(
                _blocker(
                    code=str(issue.get("code") or "gate_issue"),
                    source=str(issue.get("source") or "recovery_gate"),
                    message=str(issue.get("message") or "Recovery gate issue."),
                    queue_id=str(issue.get("queue_id") or ""),
                    domain=str(issue.get("domain") or ""),
                )
            )

    blocker_records: list[dict[str, str]] = []
    seen_blockers: set[tuple[str, str, str, str]] = set()
    for blocker in blockers:
        record = blocker.to_record()
        dedupe_key = (
            record["code"],
            record["source"],
            record["queue_id"],
            record["domain"],
        )
        if dedupe_key in seen_blockers:
            continue
        seen_blockers.add(dedupe_key)
        blocker_records.append(record)
    unique_blocker_codes = sorted({record["code"] for record in blocker_records})
    if blocker_records:
        status = "blocked"
    elif str(gate.get("status") or "") == "pass":
        status = "ready_for_controlled_apply"
    else:
        status = "incomplete"

    return {
        "summary": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "packet_dir": str(packet_dir),
            "audit_dir": str(audit_dir),
            "db_prereq_dir": str(db_prereq_dir) if db_prereq_dir else "",
            "validation_dir": str(validation_dir),
            "ingest_dir": str(ingest_dir),
            "gate_dir": str(gate_dir),
            "total_p0_queue_items": _as_int(packet.get("total_queue_items")),
            "total_p0_field_rows": _as_int(packet.get("total_field_rows")),
            "ready_or_validated_count": _as_int(audit.get("ready_or_validated_count")),
            "recovery_candidate_count": _as_int(audit.get("recovery_candidate_count")),
            "manual_fallback_count": _as_int(audit.get("manual_fallback_count")),
            "blocked_ready_count": _as_int(audit.get("blocked_ready_count")),
            "db_prereq_status": str(db_prereq.get("status") or "missing"),
            "db_prereq_error_count": (
                _issue_count(db_prereq, "error") if db_prereq else 0
            ),
            "validation_error_count": _issue_count(validation, "error"),
            "validation_warning_count": _issue_count(validation, "warning"),
            "db_checks_skipped": bool(
                isinstance(db_checks, Mapping) and db_checks.get("skipped")
            ),
            "apply_eligible_count": _as_int(validation.get("apply_eligible_count")),
            "manual_required_count": manual_required_count,
            "manual_required_domain_counts": manual_required_domain_counts,
            "manual_required_skip_reason_counts": manual_required_skip_reason_counts,
            "ingest_error_count": _issue_count(ingest, "error"),
            "ingest_action_counts": ingest.get("action_counts") or {},
            "gate_status": str(gate.get("status") or "missing"),
            "blocker_count": len(blocker_records),
            "blocker_codes": unique_blocker_codes,
        },
        "blockers": blocker_records,
        "source_summaries": {
            "packet": packet,
            "audit": audit,
            "db_prereq": db_prereq,
            "validation": validation,
            "ingest": ingest,
            "gate": gate,
            "manual_required": {
                "count": manual_required_count,
                "domain_counts": manual_required_domain_counts,
                "skip_reason_counts": manual_required_skip_reason_counts,
            },
        },
    }


def _render_handoff(report: Mapping[str, Any]) -> str:
    summary = report.get("summary") or {}
    blockers = report.get("blockers") or []
    lines = [
        "# P0 Operator Data Recovery Status",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- P0 queue rows: `{summary.get('total_p0_queue_items', 0)}`",
        f"- Ready/validated rows: `{summary.get('ready_or_validated_count', 0)}`",
        f"- Recovery candidates: `{summary.get('recovery_candidate_count', 0)}`",
        f"- Manual fallback rows: `{summary.get('manual_fallback_count', 0)}`",
        f"- Apply eligible rows: `{summary.get('apply_eligible_count', 0)}`",
        f"- Manual required rows: `{summary.get('manual_required_count', 0)}`",
        f"- DB checks skipped: `{str(summary.get('db_checks_skipped', False)).lower()}`",
        f"- Gate status: `{summary.get('gate_status', 'missing')}`",
        "",
    ]
    if _as_int(summary.get("manual_required_count")) > 0:
        lines.extend(
            [
                "## BASEBALL_DATA_SYNC_REQUIRED",
                "",
                "These P0 rows should be handed to the external trusted baseball data sync project.",
                "This repo must not crawl, synthesize, or directly enter missing baseball data.",
                "The legacy `MANUAL_BASEBALL_DATA_REQUIRED` marker remains for compatibility.",
                "",
                f"- By domain: `{_format_counts(summary.get('manual_required_domain_counts') or {})}`",
                f"- By reason: `{_format_counts(summary.get('manual_required_skip_reason_counts') or {})}`",
                f"- Sync CSV: `../gate/baseball_data_sync_required_rows.csv`",
                f"- External source: `trusted_baseball_data_project`",
                f"- Detail CSV: `../gate/manual_baseball_data_required_rows.csv`",
                "",
            ]
        )
    if blockers:
        lines.extend(["## Blocking Conditions", ""])
        for blocker in blockers:
            lines.append(
                "- "
                f"`{blocker.get('code')}` "
                f"source=`{blocker.get('source')}` "
                f"queue_id=`{blocker.get('queue_id', '')}` "
                f"domain=`{blocker.get('domain', '')}`: {blocker.get('message', '')}"
            )
        lines.extend(
            [
                "",
                "## Next Step",
                "",
                "Operators must fill intended P0 rows, rerun packet audit, then run strict validation with DB checks before ingest dry-run and recovery gate.",
            ]
        )
    else:
        lines.extend(
            [
                "## Next Step",
                "",
                "Recovery status is ready for controlled apply review. Keep `OPERATOR_DATA_FAST_PATH_ENABLED` disabled outside smoke.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def run_summary(
    *,
    packet_dir: Path,
    audit_dir: Path,
    db_prereq_dir: Path,
    validation_dir: Path,
    ingest_dir: Path,
    gate_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    report = build_status_report(
        packet_summary=_read_json(packet_dir / "p0_input_summary.json"),
        audit_report=_read_json(audit_dir / "p0_input_audit_summary.json"),
        db_prereq_report=_read_json(
            db_prereq_dir / "db_prereq_summary.json", required=False
        ),
        validation_summary=_read_json(
            validation_dir / "operator_data_validation_summary.json"
        ),
        ingest_summary=_read_json(ingest_dir / "operator_data_ingest_summary.json"),
        gate_report=_read_json(gate_dir / "summary.json"),
        gate_issues=_read_csv(gate_dir / "issues.csv"),
        audit_issues=_read_csv(audit_dir / "p0_input_audit_issues.csv", required=False),
        validation_issues=_read_csv(
            validation_dir / "operator_data_validation_issues.csv", required=False
        ),
        ingest_issues=_read_csv(
            ingest_dir / "operator_data_ingest_issues.csv", required=False
        ),
        gate_manual_required_rows=_read_csv(
            gate_dir / "manual_baseball_data_required_rows.csv",
            required=False,
        ),
        packet_dir=packet_dir,
        audit_dir=audit_dir,
        db_prereq_dir=db_prereq_dir,
        validation_dir=validation_dir,
        ingest_dir=ingest_dir,
        gate_dir=gate_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "p0_recovery_status_summary.json", report)
    _write_csv(
        output_dir / "p0_recovery_status_blockers.csv",
        report["blockers"],
        BLOCKER_FIELDNAMES,
    )
    (output_dir / "p0_recovery_status_handoff.md").write_text(
        _render_handoff(report),
        encoding="utf-8",
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize P0 recovery readiness artifacts."
    )
    parser.add_argument("--packet-dir", default=str(DEFAULT_PACKET_DIR))
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--db-prereq-dir", default=str(DEFAULT_DB_PREREQ_DIR))
    parser.add_argument("--validation-dir", default=str(DEFAULT_VALIDATION_DIR))
    parser.add_argument("--ingest-dir", default=str(DEFAULT_INGEST_DIR))
    parser.add_argument("--gate-dir", default=str(DEFAULT_GATE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_summary(
        packet_dir=Path(args.packet_dir),
        audit_dir=Path(args.audit_dir),
        db_prereq_dir=Path(args.db_prereq_dir),
        validation_dir=Path(args.validation_dir),
        ingest_dir=Path(args.ingest_dir),
        gate_dir=Path(args.gate_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 1 if report["summary"]["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
