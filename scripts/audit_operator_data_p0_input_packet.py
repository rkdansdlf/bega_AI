#!/usr/bin/env python3
"""Audit a filled P0 operator input packet before strict validation.

This script is a read-only preflight for operator-filled CSV packets. It does
not read the DB, write baseball data, crawl, infer, or repair missing values.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_INPUT = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_operator_packet"
    / "post_db_fast_path_docker_kbo500"
    / "p0_queue.csv"
)
DEFAULT_FIELDS_INPUT = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_operator_packet"
    / "post_db_fast_path_docker_kbo500"
    / "p0_fields.csv"
)
DEFAULT_SOURCE_QUEUE_INPUT = (
    PROJECT_ROOT / "reports" / "operator_data_handoff_queue_post_db_fast_path_docker_kbo500.csv"
)
DEFAULT_SOURCE_FIELDS_INPUT = (
    PROJECT_ROOT / "reports" / "operator_data_handoff_fields_post_db_fast_path_docker_kbo500.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "reports" / "operator_data_operator_packet_audit" / "post_db_fast_path_docker_kbo500"
)

P0_DOMAIN_ORDER = ("season_meta", "schedule_window", "game_day_lineup", "roster_news")
P0_DOMAINS = set(P0_DOMAIN_ORDER)
RECOVERY_STATUSES = {"ready_for_validation", "validated"}
MANUAL_FALLBACK_STATUSES = {"pending", "rejected", "applied"}
ALLOWED_STATUSES = RECOVERY_STATUSES | MANUAL_FALLBACK_STATUSES
TRUTHY_VALUES = {"1", "true", "t", "yes", "y", "verified"}
CONFIDENCE_MINIMUM = 0.70
COMMON_SOURCE_FIELDS = ("source_name", "source_checked_at", "is_verified", "confidence")
QUEUE_FIELDNAMES = [
    "queue_id",
    "priority",
    "priority_reason",
    "domain",
    "contract_code",
    "question",
    "required_fields",
    "endpoint_count",
    "endpoints",
    "sample_answer",
    "operator_status",
    "operator_owner",
    "operator_notes",
]
FIELDS_FIELDNAMES = [
    "queue_id",
    "domain",
    "contract_code",
    "question",
    "field_name",
    "field_description",
    "required",
    "operator_value",
    "operator_notes",
]
IMMUTABLE_QUEUE_FIELDS = [
    "queue_id",
    "priority",
    "priority_reason",
    "domain",
    "contract_code",
    "question",
    "required_fields",
    "endpoint_count",
    "endpoints",
    "sample_answer",
]
IMMUTABLE_FIELD_FIELDS = [
    "queue_id",
    "domain",
    "contract_code",
    "question",
    "field_name",
    "field_description",
    "required",
]
ISSUE_FIELDNAMES = ["severity", "code", "message", "queue_id", "domain", "field_name", "source"]
READINESS_FIELDNAMES = [
    "queue_id",
    "domain",
    "operator_status",
    "readiness_status",
    "manual_fallback",
    "issue_count",
    "reason",
    "question",
]


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    queue_id: str = ""
    domain: str = ""
    field_name: str = ""
    source: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "queue_id": self.queue_id,
            "domain": self.domain,
            "field_name": self.field_name,
            "source": self.source,
        }


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"required CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _issue(
    *,
    code: str,
    message: str,
    severity: str = "error",
    queue_id: str = "",
    domain: str = "",
    field_name: str = "",
    source: str = "",
) -> AuditIssue:
    return AuditIssue(
        severity=severity,
        code=code,
        message=message,
        queue_id=queue_id,
        domain=domain,
        field_name=field_name,
        source=source,
    )


def _parse_float(value: Any) -> Optional[float]:
    try:
        return float(_normalize_text(value))
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> Optional[int]:
    try:
        return int(_normalize_text(value))
    except (TypeError, ValueError):
        return None


def _parse_iso_date(value: Any) -> Optional[date]:
    raw = _normalize_text(value)
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    raw = _normalize_text(value)
    if not raw:
        return None
    try:
        if len(raw) == 10:
            return datetime.fromisoformat(f"{raw}T00:00:00")
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _valid_time(value: Any) -> bool:
    raw = _normalize_text(value)
    if not raw:
        return False
    if _parse_iso_datetime(raw) is not None:
        return True
    try:
        datetime.strptime(raw, "%H:%M")
        return True
    except ValueError:
        return False


def _is_truthy(value: Any) -> bool:
    return _normalize_text(value).lower() in TRUTHY_VALUES


def _is_required(value: Any) -> bool:
    return _normalize_text(value).lower() in TRUTHY_VALUES


def _build_source_p0(
    queue_rows: Sequence[Mapping[str, str]],
    field_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source_queue = [
        dict(row)
        for row in queue_rows
        if _normalize_text(row.get("priority")) == "P0"
        and _normalize_text(row.get("domain")) in P0_DOMAINS
    ]
    source_queue_ids = {_normalize_text(row.get("queue_id")) for row in source_queue}
    source_fields = [
        dict(row)
        for row in field_rows
        if _normalize_text(row.get("queue_id")) in source_queue_ids
        and _normalize_text(row.get("domain")) in P0_DOMAINS
    ]
    return source_queue, source_fields


def _validate_headers(
    *,
    queue_fields: Sequence[str],
    field_fields: Sequence[str],
    source_queue_fields: Sequence[str],
    source_field_fields: Sequence[str],
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    checks = [
        ("queue_header", queue_fields, QUEUE_FIELDNAMES, "packet_queue"),
        ("fields_header", field_fields, FIELDS_FIELDNAMES, "packet_fields"),
        ("source_queue_header", source_queue_fields, QUEUE_FIELDNAMES, "source_queue"),
        ("source_fields_header", source_field_fields, FIELDS_FIELDNAMES, "source_fields"),
    ]
    for field_name, actual, expected, source in checks:
        if list(actual) != expected:
            issues.append(
                _issue(
                    code="invalid_csv_header",
                    message=f"{field_name} must match {expected}; got {list(actual)}.",
                    field_name=field_name,
                    source=source,
                )
            )
    return issues


def _indexed_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    key_fields: Sequence[str],
    source: str,
) -> tuple[dict[tuple[str, ...], Mapping[str, str]], list[AuditIssue]]:
    indexed: dict[tuple[str, ...], Mapping[str, str]] = {}
    issues: list[AuditIssue] = []
    for row in rows:
        key = tuple(_normalize_text(row.get(field)) for field in key_fields)
        if any(not part for part in key):
            issues.append(
                _issue(
                    code="missing_identity_field",
                    message=f"{source} row is missing one of {list(key_fields)}.",
                    queue_id=_normalize_text(row.get("queue_id")),
                    domain=_normalize_text(row.get("domain")),
                    source=source,
                )
            )
            continue
        if key in indexed:
            issues.append(
                _issue(
                    code="duplicate_identity",
                    message=f"{source} has duplicate identity {key}.",
                    queue_id=_normalize_text(row.get("queue_id")),
                    domain=_normalize_text(row.get("domain")),
                    field_name=_normalize_text(row.get("field_name")),
                    source=source,
                )
            )
            continue
        indexed[key] = row
    return indexed, issues


def _compare_key_sets(
    *,
    packet_keys: set[tuple[str, ...]],
    source_keys: set[tuple[str, ...]],
    source: str,
    key_kind: str,
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for key in sorted(packet_keys - source_keys):
        issues.append(
            _issue(
                code=f"unexpected_{key_kind}",
                message=f"Packet contains {key_kind} not present in source P0 packet: {key}.",
                queue_id=key[0] if key else "",
                field_name=key[1] if len(key) > 1 else "",
                source=source,
            )
        )
    for key in sorted(source_keys - packet_keys):
        issues.append(
            _issue(
                code=f"missing_{key_kind}",
                message=f"Packet is missing source P0 {key_kind}: {key}.",
                queue_id=key[0] if key else "",
                field_name=key[1] if len(key) > 1 else "",
                source=source,
            )
        )
    return issues


def _validate_domain_scope(
    queue_rows: Sequence[Mapping[str, str]],
    field_rows: Sequence[Mapping[str, str]],
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for row in queue_rows:
        domain = _normalize_text(row.get("domain"))
        if domain not in P0_DOMAINS:
            issues.append(
                _issue(
                    code="non_p0_queue_domain",
                    message="P0 input packet must not contain non-P0 queue domains.",
                    queue_id=_normalize_text(row.get("queue_id")),
                    domain=domain,
                    source="packet_queue",
                )
            )
    for row in field_rows:
        domain = _normalize_text(row.get("domain"))
        if domain not in P0_DOMAINS:
            issues.append(
                _issue(
                    code="non_p0_field_domain",
                    message="P0 input packet must not contain non-P0 field domains.",
                    queue_id=_normalize_text(row.get("queue_id")),
                    domain=domain,
                    field_name=_normalize_text(row.get("field_name")),
                    source="packet_fields",
                )
            )
    return issues


def _validate_immutable_values(
    *,
    packet_queue_by_id: Mapping[tuple[str, ...], Mapping[str, str]],
    source_queue_by_id: Mapping[tuple[str, ...], Mapping[str, str]],
    packet_fields_by_key: Mapping[tuple[str, ...], Mapping[str, str]],
    source_fields_by_key: Mapping[tuple[str, ...], Mapping[str, str]],
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for key, packet_row in packet_queue_by_id.items():
        source_row = source_queue_by_id.get(key)
        if not source_row:
            continue
        for field in IMMUTABLE_QUEUE_FIELDS:
            if _normalize_text(packet_row.get(field)) != _normalize_text(source_row.get(field)):
                issues.append(
                    _issue(
                        code="immutable_queue_drift",
                        message=f"Queue field {field} differs from source handoff.",
                        queue_id=key[0],
                        domain=_normalize_text(packet_row.get("domain")),
                        field_name=field,
                        source="packet_queue",
                    )
                )
    for key, packet_row in packet_fields_by_key.items():
        source_row = source_fields_by_key.get(key)
        if not source_row:
            continue
        for field in IMMUTABLE_FIELD_FIELDS:
            if _normalize_text(packet_row.get(field)) != _normalize_text(source_row.get(field)):
                issues.append(
                    _issue(
                        code="immutable_field_drift",
                        message=f"Field row column {field} differs from source handoff.",
                        queue_id=key[0],
                        domain=_normalize_text(packet_row.get("domain")),
                        field_name=key[1],
                        source="packet_fields",
                    )
                )
    return issues


def _payload_for_queue(
    field_rows: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    payload: dict[str, str] = {}
    for row in field_rows:
        field_name = _normalize_text(row.get("field_name"))
        if not field_name:
            continue
        payload[field_name] = _normalize_text(row.get("operator_value"))
    return payload


def _field_rows_by_queue(
    field_rows: Sequence[Mapping[str, str]],
) -> dict[str, list[Mapping[str, str]]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in field_rows:
        grouped[_normalize_text(row.get("queue_id"))].append(row)
    return grouped


def _validate_ready_payload(
    *,
    queue_row: Mapping[str, str],
    field_rows: Sequence[Mapping[str, str]],
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    queue_id = _normalize_text(queue_row.get("queue_id"))
    domain = _normalize_text(queue_row.get("domain"))
    payload = _payload_for_queue(field_rows)

    for row in field_rows:
        field_name = _normalize_text(row.get("field_name"))
        if _is_required(row.get("required")) and not _normalize_text(row.get("operator_value")):
            issues.append(
                _issue(
                    code="missing_required_operator_value",
                    message="Ready P0 row has an empty required operator_value.",
                    queue_id=queue_id,
                    domain=domain,
                    field_name=field_name,
                    source="packet_fields",
                )
            )

    for field in COMMON_SOURCE_FIELDS:
        if not payload.get(field):
            issues.append(
                _issue(
                    code="missing_source_metadata",
                    message=f"Ready P0 row is missing source metadata field {field}.",
                    queue_id=queue_id,
                    domain=domain,
                    field_name=field,
                    source="packet_fields",
                )
            )
    if payload.get("source_checked_at") and _parse_iso_datetime(payload.get("source_checked_at")) is None:
        issues.append(
            _issue(
                code="invalid_source_checked_at",
                message="source_checked_at must be an ISO date or datetime.",
                queue_id=queue_id,
                domain=domain,
                field_name="source_checked_at",
                source="packet_fields",
            )
        )
    if payload.get("is_verified") and not _is_truthy(payload.get("is_verified")):
        issues.append(
            _issue(
                code="source_not_verified",
                message="Ready P0 row requires is_verified=true.",
                queue_id=queue_id,
                domain=domain,
                field_name="is_verified",
                source="packet_fields",
            )
        )
    confidence = _parse_float(payload.get("confidence"))
    if payload.get("confidence") and (confidence is None or confidence < CONFIDENCE_MINIMUM):
        issues.append(
            _issue(
                code="confidence_below_minimum",
                message=f"Ready P0 row requires confidence >= {CONFIDENCE_MINIMUM:.2f}.",
                queue_id=queue_id,
                domain=domain,
                field_name="confidence",
                source="packet_fields",
            )
        )

    issues.extend(_validate_domain_formats(queue_id=queue_id, domain=domain, payload=payload))
    return issues


def _validate_domain_formats(
    *,
    queue_id: str,
    domain: str,
    payload: Mapping[str, str],
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []

    def add(code: str, field_name: str, message: str) -> None:
        issues.append(
            _issue(
                code=code,
                message=message,
                queue_id=queue_id,
                domain=domain,
                field_name=field_name,
                source="packet_fields",
            )
        )

    if domain in {"season_meta", "roster_news"} and payload.get("season_year"):
        if _parse_int(payload.get("season_year")) is None:
            add("invalid_integer_field", "season_year", "season_year must be an integer.")
    if domain == "season_meta" and payload.get("event_date"):
        if _parse_iso_date(payload.get("event_date")) is None:
            add("invalid_date_field", "event_date", "event_date must be an ISO date.")
    if domain == "schedule_window":
        if payload.get("game_date") and _parse_iso_date(payload.get("game_date")) is None:
            add("invalid_date_field", "game_date", "game_date must be an ISO date.")
        if payload.get("start_time") and not _valid_time(payload.get("start_time")):
            add("invalid_time_field", "start_time", "start_time must be HH:MM or ISO datetime.")
    if domain == "game_day_lineup":
        if payload.get("batting_order") and _parse_int(payload.get("batting_order")) is None:
            add("invalid_integer_field", "batting_order", "batting_order must be an integer.")
        if payload.get("announced_at") and _parse_iso_datetime(payload.get("announced_at")) is None:
            add("invalid_datetime_field", "announced_at", "announced_at must be an ISO datetime.")
    if domain == "roster_news" and payload.get("effective_date"):
        if _parse_iso_date(payload.get("effective_date")) is None:
            add("invalid_date_field", "effective_date", "effective_date must be an ISO date.")
    return issues


def build_audit_report(
    *,
    queue_rows: Sequence[Mapping[str, str]],
    field_rows: Sequence[Mapping[str, str]],
    source_queue_rows: Sequence[Mapping[str, str]],
    source_field_rows: Sequence[Mapping[str, str]],
    queue_path: Path,
    fields_path: Path,
    source_queue_path: Path,
    source_fields_path: Path,
    require_ready: bool = False,
) -> dict[str, Any]:
    source_p0_queue_rows, source_p0_field_rows = _build_source_p0(
        source_queue_rows,
        source_field_rows,
    )
    issues: list[AuditIssue] = []
    issues.extend(_validate_domain_scope(queue_rows, field_rows))

    packet_queue_by_id, identity_issues = _indexed_rows(
        queue_rows,
        key_fields=["queue_id"],
        source="packet_queue",
    )
    issues.extend(identity_issues)
    source_queue_by_id, identity_issues = _indexed_rows(
        source_p0_queue_rows,
        key_fields=["queue_id"],
        source="source_queue",
    )
    issues.extend(identity_issues)
    packet_fields_by_key, identity_issues = _indexed_rows(
        field_rows,
        key_fields=["queue_id", "field_name"],
        source="packet_fields",
    )
    issues.extend(identity_issues)
    source_fields_by_key, identity_issues = _indexed_rows(
        source_p0_field_rows,
        key_fields=["queue_id", "field_name"],
        source="source_fields",
    )
    issues.extend(identity_issues)

    issues.extend(
        _compare_key_sets(
            packet_keys=set(packet_queue_by_id),
            source_keys=set(source_queue_by_id),
            source="packet_queue",
            key_kind="queue_row",
        )
    )
    issues.extend(
        _compare_key_sets(
            packet_keys=set(packet_fields_by_key),
            source_keys=set(source_fields_by_key),
            source="packet_fields",
            key_kind="field_row",
        )
    )
    issues.extend(
        _validate_immutable_values(
            packet_queue_by_id=packet_queue_by_id,
            source_queue_by_id=source_queue_by_id,
            packet_fields_by_key=packet_fields_by_key,
            source_fields_by_key=source_fields_by_key,
        )
    )

    fields_by_queue = _field_rows_by_queue(field_rows)
    issues_by_queue: dict[str, list[AuditIssue]] = defaultdict(list)
    ready_count = 0
    readiness_rows: list[dict[str, Any]] = []

    for queue_row in queue_rows:
        queue_id = _normalize_text(queue_row.get("queue_id"))
        domain = _normalize_text(queue_row.get("domain"))
        status = _normalize_text(queue_row.get("operator_status"))
        row_issues: list[AuditIssue] = []
        if status not in ALLOWED_STATUSES:
            row_issues.append(
                _issue(
                    code="invalid_operator_status",
                    message=(
                        "operator_status must be one of "
                        f"{sorted(ALLOWED_STATUSES)} for P0 input QA."
                    ),
                    queue_id=queue_id,
                    domain=domain,
                    field_name="operator_status",
                    source="packet_queue",
                )
            )
        if status in RECOVERY_STATUSES:
            ready_count += 1
            row_issues.extend(
                _validate_ready_payload(
                    queue_row=queue_row,
                    field_rows=fields_by_queue.get(queue_id, []),
                )
            )
        issues_by_queue[queue_id].extend(row_issues)
        issues.extend(row_issues)

    for queue_row in queue_rows:
        queue_id = _normalize_text(queue_row.get("queue_id"))
        status = _normalize_text(queue_row.get("operator_status"))
        queue_issue_count = sum(1 for issue in issues_by_queue.get(queue_id, []) if issue.severity == "error")
        if status in RECOVERY_STATUSES and queue_issue_count == 0:
            readiness_status = "recovery_candidate"
            reason = "ready_for_strict_validation"
            manual_fallback = "false"
        elif status in RECOVERY_STATUSES:
            readiness_status = "blocked"
            reason = "ready_row_has_input_errors"
            manual_fallback = "true"
        elif status == "rejected":
            readiness_status = "manual_fallback"
            reason = "operator_rejected"
            manual_fallback = "true"
        elif status == "applied":
            readiness_status = "manual_fallback"
            reason = "already_applied_not_recovery_candidate"
            manual_fallback = "true"
        else:
            readiness_status = "manual_fallback"
            reason = "operator_status_pending"
            manual_fallback = "true"
        readiness_rows.append(
            {
                "queue_id": queue_id,
                "domain": _normalize_text(queue_row.get("domain")),
                "operator_status": status,
                "readiness_status": readiness_status,
                "manual_fallback": manual_fallback,
                "issue_count": str(queue_issue_count),
                "reason": reason,
                "question": _normalize_text(queue_row.get("question")),
            }
        )

    if ready_count == 0:
        severity = "error" if require_ready else "warning"
        issues.append(
            _issue(
                code="no_ready_p0_rows",
                severity=severity,
                message="No P0 rows are marked ready_for_validation or validated.",
                source="packet_queue",
            )
        )

    issue_counter = Counter(issue.severity for issue in issues)
    if issue_counter.get("error", 0) > 0:
        status = "fail"
    elif issue_counter.get("warning", 0) > 0:
        status = "warning"
    else:
        status = "pass"

    domain_counts = Counter(_normalize_text(row.get("domain")) for row in queue_rows)
    status_counts = Counter(_normalize_text(row.get("operator_status")) for row in queue_rows)
    recovery_candidate_count = sum(
        1 for row in readiness_rows if row["readiness_status"] == "recovery_candidate"
    )
    manual_fallback_count = sum(1 for row in readiness_rows if row["manual_fallback"] == "true")
    blocked_ready_count = sum(1 for row in readiness_rows if row["readiness_status"] == "blocked")

    return {
        "summary": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "require_ready": require_ready,
            "queue_path": str(queue_path),
            "fields_path": str(fields_path),
            "source_queue_path": str(source_queue_path),
            "source_fields_path": str(source_fields_path),
            "total_queue_items": len(queue_rows),
            "total_field_rows": len(field_rows),
            "ready_or_validated_count": ready_count,
            "recovery_candidate_count": recovery_candidate_count,
            "manual_fallback_count": manual_fallback_count,
            "blocked_ready_count": blocked_ready_count,
            "domain_counts": {domain: domain_counts.get(domain, 0) for domain in P0_DOMAIN_ORDER},
            "status_counts": dict(sorted(status_counts.items())),
            "issue_counts": {
                "error": issue_counter.get("error", 0),
                "warning": issue_counter.get("warning", 0),
            },
        },
        "issues": [issue.to_record() for issue in issues],
        "readiness_plan": readiness_rows,
    }


def _render_handoff(report: Mapping[str, Any]) -> str:
    summary = report.get("summary") or {}
    issues = report.get("issues") or []
    lines = [
        "# P0 Operator Input Packet Audit",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Ready/validated rows: `{summary.get('ready_or_validated_count', 0)}`",
        f"- Recovery candidates: `{summary.get('recovery_candidate_count', 0)}`",
        f"- Manual fallback rows: `{summary.get('manual_fallback_count', 0)}`",
        f"- Blocked ready rows: `{summary.get('blocked_ready_count', 0)}`",
        f"- Errors: `{(summary.get('issue_counts') or {}).get('error', 0)}`",
        f"- Warnings: `{(summary.get('issue_counts') or {}).get('warning', 0)}`",
        "",
    ]
    if issues:
        lines.extend(["## Issues", ""])
        for issue in issues[:25]:
            lines.append(
                "- "
                f"`{issue.get('severity')}` `{issue.get('code')}` "
                f"queue_id=`{issue.get('queue_id', '')}` "
                f"domain=`{issue.get('domain', '')}` "
                f"field=`{issue.get('field_name', '')}`: {issue.get('message', '')}"
            )
        if len(issues) > 25:
            lines.append(f"- ... {len(issues) - 25} more issue(s) in CSV.")
        lines.append("")
    if summary.get("status") == "fail":
        lines.extend(
            [
                "## Result",
                "",
                "Fix the packet before strict validation. Missing or invalid ready rows must stay on the manual contract path.",
            ]
        )
    else:
        lines.extend(
            [
                "## Next Step",
                "",
                "Run strict validation with DB checks. Pending, rejected, or blocked rows are not recovery candidates and must keep `MANUAL_BASEBALL_DATA_REQUIRED`.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def run_audit(
    *,
    queue_path: Path,
    fields_path: Path,
    source_queue_path: Path,
    source_fields_path: Path,
    output_dir: Path,
    require_ready: bool = False,
) -> dict[str, Any]:
    queue_fields, queue_rows = _read_csv(queue_path)
    field_fields, field_rows = _read_csv(fields_path)
    source_queue_fields, source_queue_rows = _read_csv(source_queue_path)
    source_field_fields, source_field_rows = _read_csv(source_fields_path)
    header_issues = _validate_headers(
        queue_fields=queue_fields,
        field_fields=field_fields,
        source_queue_fields=source_queue_fields,
        source_field_fields=source_field_fields,
    )
    report = build_audit_report(
        queue_rows=queue_rows,
        field_rows=field_rows,
        source_queue_rows=source_queue_rows,
        source_field_rows=source_field_rows,
        queue_path=queue_path,
        fields_path=fields_path,
        source_queue_path=source_queue_path,
        source_fields_path=source_fields_path,
        require_ready=require_ready,
    )
    if header_issues:
        report["issues"] = [*report["issues"], *[issue.to_record() for issue in header_issues]]
        error_count = sum(1 for issue in report["issues"] if issue.get("severity") == "error")
        warning_count = sum(1 for issue in report["issues"] if issue.get("severity") == "warning")
        report["summary"]["issue_counts"] = {"error": error_count, "warning": warning_count}
        report["summary"]["status"] = "fail" if error_count else "warning" if warning_count else "pass"

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "p0_input_audit_summary.json", report)
    _write_csv(output_dir / "p0_input_audit_issues.csv", report["issues"], ISSUE_FIELDNAMES)
    _write_csv(
        output_dir / "p0_input_readiness_plan.csv",
        report["readiness_plan"],
        READINESS_FIELDNAMES,
    )
    (output_dir / "p0_input_audit_handoff.md").write_text(_render_handoff(report), encoding="utf-8")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a P0 operator input packet.")
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE_INPUT))
    parser.add_argument("--fields", default=str(DEFAULT_FIELDS_INPUT))
    parser.add_argument("--source-queue", default=str(DEFAULT_SOURCE_QUEUE_INPUT))
    parser.add_argument("--source-fields", default=str(DEFAULT_SOURCE_FIELDS_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Fail when no P0 rows are ready_for_validation or validated.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_audit(
        queue_path=Path(args.queue),
        fields_path=Path(args.fields),
        source_queue_path=Path(args.source_queue),
        source_fields_path=Path(args.source_fields),
        output_dir=Path(args.output_dir),
        require_ready=bool(args.require_ready),
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 1 if report["summary"]["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
