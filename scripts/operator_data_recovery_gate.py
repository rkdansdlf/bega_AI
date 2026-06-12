#!/usr/bin/env python3
"""Gate operator-data recovery readiness from validation and ingest reports.

This script is read-only with respect to baseball data. It only reads existing
validation/ingest artifacts and writes a gate summary for operator review.
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
DEFAULT_VALIDATION_DIR = (
    PROJECT_ROOT / "reports" / "operator_data_validation" / "post_db_fast_path_docker_kbo500"
)
DEFAULT_INGEST_DIR = (
    PROJECT_ROOT / "reports" / "operator_data_ingest" / "post_db_fast_path_docker_kbo500"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "reports" / "operator_data_recovery_gate" / "post_db_fast_path_docker_kbo500"
)
P0_DOMAIN_ORDER = ("season_meta", "schedule_window", "game_day_lineup", "roster_news")
P0_DOMAINS = set(P0_DOMAIN_ORDER)
APPLY_ACTIONS = {"insert", "update", "noop"}
MANUAL_CONTRACT = "MANUAL_BASEBALL_DATA_REQUIRED"
MANUAL_REQUIRED_FIELDNAMES = [
    "queue_id",
    "domain",
    "operator_status",
    "skip_reason",
    "question",
    "required_fields",
    "missing_required_fields",
    "manual_contract",
]


@dataclass(frozen=True)
class GateIssue:
    severity: str
    code: str
    message: str
    source: str = ""
    queue_id: str = ""
    domain: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "queue_id": self.queue_id,
            "domain": self.domain,
        }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required report is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"report must be a JSON object: {path}")
    return dict(payload)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"required report is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required report is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"normalized row {line_number} must be a JSON object")
            rows.append(dict(payload))
    return rows


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


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _as_int(value: Any) -> int:
    try:
        return int(str(value or "0").strip())
    except (TypeError, ValueError):
        return 0


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def _field_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_normalize_text(field) for field in value.split("|") if _normalize_text(field)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalize_text(field) for field in value if _normalize_text(field)]
    return []


def _payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("payload")
    return payload if isinstance(payload, Mapping) else {}


def _missing_required_fields(row: Mapping[str, Any], required_fields: Sequence[str]) -> list[str]:
    payload = _payload(row)
    return [field for field in required_fields if not _normalize_text(payload.get(field))]


def _manual_required_reason(row: Mapping[str, Any]) -> str:
    domain = _normalize_text(row.get("domain"))
    if domain not in P0_DOMAINS or _as_bool(row.get("apply_eligible")):
        return ""

    status = _normalize_text(row.get("operator_status"))
    skip_reason = _normalize_text(row.get("skip_reason"))
    if status not in {"ready_for_validation", "validated", "applied"}:
        return skip_reason or f"operator_status_{status or 'missing'}"
    if skip_reason in {"validation_not_passed", "not_verified", "low_confidence"}:
        return skip_reason
    return ""


def build_manual_required_rows(
    normalized_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in normalized_rows:
        skip_reason = _manual_required_reason(row)
        if not skip_reason:
            continue
        required_fields = _field_list(row.get("required_fields"))
        rows.append(
            {
                "queue_id": _normalize_text(row.get("queue_id")),
                "domain": _normalize_text(row.get("domain")),
                "operator_status": _normalize_text(row.get("operator_status")),
                "skip_reason": skip_reason,
                "question": _normalize_text(row.get("question")),
                "required_fields": "|".join(required_fields),
                "missing_required_fields": "|".join(
                    _missing_required_fields(row, required_fields)
                ),
                "manual_contract": MANUAL_CONTRACT,
            }
        )
    domain_sort = {domain: index for index, domain in enumerate(P0_DOMAIN_ORDER)}
    rows.sort(
        key=lambda row: (
            domain_sort.get(row["domain"], len(domain_sort)),
            row["queue_id"],
        )
    )
    return rows


def _manual_domain_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    counts = Counter(_normalize_text(row.get("domain")) for row in rows)
    return {domain: counts.get(domain, 0) for domain in P0_DOMAIN_ORDER}


def _manual_skip_reason_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    counts = Counter(_normalize_text(row.get("skip_reason")) for row in rows)
    return dict(sorted(counts.items()))


def _format_counts(counts: Mapping[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in counts.items() if _as_int(value) > 0]
    return ", ".join(parts) if parts else "none"


def _issue(
    *,
    code: str,
    message: str,
    source: str = "",
    queue_id: str = "",
    domain: str = "",
    severity: str = "error",
) -> GateIssue:
    return GateIssue(
        severity=severity,
        code=code,
        message=message,
        source=source,
        queue_id=queue_id,
        domain=domain,
    )


def build_gate_report(
    *,
    validation_summary: Mapping[str, Any],
    validation_apply_plan: Sequence[Mapping[str, str]],
    ingest_summary: Mapping[str, Any],
    ingest_plan: Sequence[Mapping[str, str]],
    ingest_issues: Sequence[Mapping[str, str]],
    validation_dir: Path,
    ingest_dir: Path,
    validation_issues: Sequence[Mapping[str, str]] = (),
    validation_normalized_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    issues: list[GateIssue] = []
    validation_issue_counts = validation_summary.get("issue_counts") or {}
    ingest_issue_counts = ingest_summary.get("issue_counts") or {}
    db_checks = validation_summary.get("db_checks") or {}
    manual_required_rows = build_manual_required_rows(validation_normalized_rows)
    manual_domain_counts = _manual_domain_counts(manual_required_rows)
    manual_skip_reason_counts = _manual_skip_reason_counts(manual_required_rows)

    if isinstance(db_checks, Mapping) and db_checks.get("skipped") is True:
        issues.append(
            _issue(
                code="db_checks_skipped",
                source="validation_summary",
                message="Strict recovery gate requires DB checks to run.",
            )
        )

    validation_errors = _as_int(
        validation_issue_counts.get("error") if isinstance(validation_issue_counts, Mapping) else 0
    )
    if validation_errors > 0:
        issues.append(
            _issue(
                code="validation_errors",
                source="validation_summary",
                message=f"Validation reported {validation_errors} error(s).",
            )
        )
    for row in validation_issues:
        if str(row.get("severity") or "") != "error":
            continue
        issues.append(
            _issue(
                code=str(row.get("code") or "validation_issue"),
                source="validation_issues",
                queue_id=str(row.get("queue_id") or ""),
                domain=str(row.get("domain") or ""),
                message=str(row.get("message") or "Validation issue."),
            )
        )

    ingest_errors = _as_int(
        ingest_issue_counts.get("error") if isinstance(ingest_issue_counts, Mapping) else 0
    )
    if ingest_errors > 0:
        issues.append(
            _issue(
                code="ingest_errors",
                source="ingest_summary",
                message=f"Ingest dry-run reported {ingest_errors} error(s).",
            )
        )

    for row in ingest_issues:
        if str(row.get("severity") or "") != "error":
            continue
        issues.append(
            _issue(
                code=str(row.get("code") or "ingest_issue"),
                source="ingest_issues",
                queue_id=str(row.get("queue_id") or ""),
                domain=str(row.get("domain") or ""),
                message=str(row.get("message") or "Ingest dry-run issue."),
            )
        )

    apply_eligible_count = _as_int(validation_summary.get("apply_eligible_count"))
    if apply_eligible_count <= 0:
        issues.append(
            _issue(
                code="no_apply_eligible_rows",
                source="validation_summary",
                message="No apply-eligible P0 rows are available for recovery.",
            )
        )

    for row in validation_apply_plan:
        domain = str(row.get("domain") or "").strip()
        if _as_bool(row.get("apply_eligible")) and domain not in P0_DOMAINS:
            issues.append(
                _issue(
                    code="non_p0_apply_eligible",
                    source="validation_apply_plan",
                    queue_id=str(row.get("queue_id") or ""),
                    domain=domain,
                    message="Only P0 domains may be apply-eligible in V1.",
                )
            )
        if _as_bool(row.get("apply_eligible")) and row.get("skip_reason") == "operator_data_v1_non_p0_domain":
            issues.append(
                _issue(
                    code="non_p0_policy_leaked_to_apply",
                    source="validation_apply_plan",
                    queue_id=str(row.get("queue_id") or ""),
                    domain=domain,
                    message="Non-P0 policy skip reason appeared on an apply-eligible row.",
                )
            )

    for row in ingest_plan:
        domain = str(row.get("domain") or "").strip()
        action = str(row.get("action") or "").strip()
        if action in APPLY_ACTIONS and domain not in P0_DOMAINS:
            issues.append(
                _issue(
                    code="non_p0_ingest_action",
                    source="ingest_plan",
                    queue_id=str(row.get("queue_id") or ""),
                    domain=domain,
                    message="Only P0 domains may have insert/update/noop ingest actions in V1.",
                )
            )
        if str(row.get("skip_reason") or "") == "overwrite_requires_flag":
            issues.append(
                _issue(
                    code="overwrite_requires_flag",
                    source="ingest_plan",
                    queue_id=str(row.get("queue_id") or ""),
                    domain=domain,
                    message="A different payload_hash exists and overwrite was not approved.",
                )
            )

    error_count = sum(1 for issue in issues if issue.severity == "error")
    status = "pass" if error_count == 0 else "fail"
    action_counts: dict[str, int] = {}
    for row in ingest_plan:
        action = str(row.get("action") or "").strip() or "missing"
        action_counts[action] = action_counts.get(action, 0) + 1

    return {
        "summary": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "validation_dir": str(validation_dir),
            "ingest_dir": str(ingest_dir),
            "apply_eligible_count": apply_eligible_count,
            "validation_error_count": validation_errors,
            "ingest_error_count": ingest_errors,
            "issue_counts": {"error": error_count},
            "ingest_action_counts": dict(sorted(action_counts.items())),
            "manual_required_count": len(manual_required_rows),
            "manual_required_domain_counts": manual_domain_counts,
            "manual_required_skip_reason_counts": manual_skip_reason_counts,
        },
        "issues": [issue.to_record() for issue in issues],
        "manual_required_rows": manual_required_rows,
    }


def _render_handoff(report: Mapping[str, Any]) -> str:
    summary = report.get("summary") or {}
    issues = report.get("issues") or []
    manual_required_rows = report.get("manual_required_rows") or []
    manual_domain_counts = summary.get("manual_required_domain_counts") or {}
    manual_skip_reason_counts = summary.get("manual_required_skip_reason_counts") or {}
    lines = [
        "# Operator Data Recovery Gate",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Apply eligible rows: `{summary.get('apply_eligible_count', 0)}`",
        f"- Manual required rows: `{summary.get('manual_required_count', 0)}`",
        f"- Validation errors: `{summary.get('validation_error_count', 0)}`",
        f"- Ingest errors: `{summary.get('ingest_error_count', 0)}`",
        "",
    ]
    if manual_required_rows:
        lines.extend(
            [
                "## MANUAL_BASEBALL_DATA_REQUIRED",
                "",
                "These P0 rows still require operator-provided data. Do not synthesize or crawl baseball data for them.",
                "",
                f"- CSV: `manual_baseball_data_required_rows.csv`",
                f"- By domain: `{_format_counts(manual_domain_counts)}`",
                f"- By reason: `{_format_counts(manual_skip_reason_counts)}`",
                "",
                "### Queue IDs",
                "",
            ]
        )
        rows_by_domain: dict[str, list[str]] = {domain: [] for domain in P0_DOMAIN_ORDER}
        for row in manual_required_rows:
            rows_by_domain.setdefault(_normalize_text(row.get("domain")), []).append(
                _normalize_text(row.get("queue_id"))
            )
        for domain in P0_DOMAIN_ORDER:
            queue_ids = rows_by_domain.get(domain) or []
            if queue_ids:
                lines.append(f"- `{domain}`: `{', '.join(queue_ids)}`")
        lines.append("")
    if issues:
        lines.extend(["## Blocking Issues", ""])
        for issue in issues:
            lines.append(
                "- "
                f"`{issue.get('code')}` "
                f"queue_id=`{issue.get('queue_id', '')}` "
                f"domain=`{issue.get('domain', '')}` "
                f"source=`{issue.get('source', '')}`: {issue.get('message', '')}"
            )
    else:
        lines.extend(
            [
                "## Result",
                "",
                "Recovery gate passed. Proceed to focused smoke before any production enablement.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def run_gate(*, validation_dir: Path, ingest_dir: Path, output_dir: Path) -> dict[str, Any]:
    report = build_gate_report(
        validation_summary=_read_json(validation_dir / "operator_data_validation_summary.json"),
        validation_apply_plan=_read_csv(validation_dir / "operator_data_apply_plan.csv"),
        validation_issues=_read_csv(validation_dir / "operator_data_validation_issues.csv"),
        validation_normalized_rows=_read_jsonl(validation_dir / "operator_data_normalized_rows.jsonl"),
        ingest_summary=_read_json(ingest_dir / "operator_data_ingest_summary.json"),
        ingest_plan=_read_csv(ingest_dir / "operator_data_ingest_plan.csv"),
        ingest_issues=_read_csv(ingest_dir / "operator_data_ingest_issues.csv"),
        validation_dir=validation_dir,
        ingest_dir=ingest_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", report)
    _write_csv(
        output_dir / "issues.csv",
        report["issues"],
        ["severity", "code", "message", "source", "queue_id", "domain"],
    )
    _write_csv(
        output_dir / "manual_baseball_data_required_rows.csv",
        report.get("manual_required_rows", []),
        MANUAL_REQUIRED_FIELDNAMES,
    )
    (output_dir / "handoff.md").write_text(_render_handoff(report), encoding="utf-8")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate operator-data recovery readiness.")
    parser.add_argument("--validation-dir", default=str(DEFAULT_VALIDATION_DIR))
    parser.add_argument("--ingest-dir", default=str(DEFAULT_INGEST_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        default=True,
        help="Exit 0 even when the recovery gate fails.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_gate(
        validation_dir=Path(args.validation_dir),
        ingest_dir=Path(args.ingest_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.strict and report["summary"]["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
