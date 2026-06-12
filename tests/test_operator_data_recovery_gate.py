from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts import operator_data_recovery_gate as gate


def _validation_summary(**overrides: Any) -> dict[str, Any]:
    base = {
        "issue_counts": {"error": 0, "warning": 0},
        "apply_eligible_count": 2,
        "db_checks": {"skipped": False, "skip_reason": ""},
    }
    base.update(overrides)
    return base


def _ingest_summary(**overrides: Any) -> dict[str, Any]:
    base = {
        "issue_counts": {"error": 0, "warning": 0},
        "action_counts": {"insert": 2},
    }
    base.update(overrides)
    return base


def _apply_plan(domain: str = "schedule_window", **overrides: Any) -> dict[str, str]:
    base = {
        "queue_id": "ODQ-0001",
        "priority": "P0",
        "domain": domain,
        "operator_status": "ready_for_validation",
        "validation_status": "pass",
        "apply_eligible": "true",
        "apply_target": "operator_schedule_items",
        "issue_count": "0",
        "skip_reason": "",
    }
    base.update({key: str(value) for key, value in overrides.items()})
    return base


def _ingest_plan(domain: str = "schedule_window", **overrides: Any) -> dict[str, str]:
    base = {
        "queue_id": "ODQ-0001",
        "domain": domain,
        "operator_status": "ready_for_validation",
        "validation_status": "pass",
        "apply_eligible": "true",
        "action": "insert",
        "apply_target": "operator_schedule_items",
        "payload_hash": "abc",
        "issue_count": "0",
        "skip_reason": "",
    }
    base.update({key: str(value) for key, value in overrides.items()})
    return base


def _normalized_row(**overrides: Any) -> dict[str, Any]:
    base = {
        "queue_id": "ODQ-0001",
        "priority": "P0",
        "domain": "schedule_window",
        "contract_code": "SCHEDULE_WINDOW_REQUIRED",
        "question": "오늘 KBO 경기 일정 알려줘.",
        "operator_status": "pending",
        "required_fields": [
            "game_date",
            "game_id",
            "home_team",
            "away_team",
            "stadium_name",
            "start_time",
            "game_status",
            "source_name",
            "source_checked_at",
            "is_verified",
            "confidence",
        ],
        "payload": {
            "game_date": "",
            "game_id": "",
            "home_team": "",
            "away_team": "",
            "stadium_name": "",
            "start_time": "",
            "game_status": "",
            "source_name": "",
            "source_checked_at": "",
            "is_verified": "",
            "confidence": "",
        },
        "source_metadata": {
            "source_name": "",
            "source_checked_at": "",
            "is_verified": None,
            "confidence": None,
        },
        "validation_status": "pass",
        "apply_eligible": False,
        "apply_target": "",
        "skip_reason": "operator_status_pending",
        "issue_count": 0,
    }
    base.update(overrides)
    return base


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _report(
    *,
    validation_summary: dict[str, Any] | None = None,
    validation_apply_plan: list[dict[str, str]] | None = None,
    ingest_summary: dict[str, Any] | None = None,
    ingest_plan: list[dict[str, str]] | None = None,
    ingest_issues: list[dict[str, str]] | None = None,
    validation_normalized_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return gate.build_gate_report(
        validation_summary=validation_summary or _validation_summary(),
        validation_apply_plan=validation_apply_plan or [_apply_plan()],
        ingest_summary=ingest_summary or _ingest_summary(),
        ingest_plan=ingest_plan or [_ingest_plan()],
        ingest_issues=ingest_issues or [],
        validation_normalized_rows=validation_normalized_rows or [],
        validation_dir=Path("validation"),
        ingest_dir=Path("ingest"),
    )


def _codes(report: dict[str, Any]) -> set[str]:
    return {str(issue["code"]) for issue in report["issues"]}


def test_gate_passes_clean_p0_dry_run() -> None:
    report = _report()

    assert report["summary"]["status"] == "pass"
    assert report["issues"] == []


def test_gate_fails_when_db_checks_are_skipped() -> None:
    report = _report(
        validation_summary=_validation_summary(
            db_checks={"skipped": True, "skip_reason": "--skip-db-checks was set"}
        )
    )

    assert report["summary"]["status"] == "fail"
    assert "db_checks_skipped" in _codes(report)


def test_gate_fails_on_validation_errors() -> None:
    report = _report(validation_summary=_validation_summary(issue_counts={"error": 1}))

    assert "validation_errors" in _codes(report)


def test_gate_fails_on_ingest_errors_and_overwrite_issue() -> None:
    report = _report(
        ingest_summary=_ingest_summary(issue_counts={"error": 1}),
        ingest_issues=[
            {
                "queue_id": "ODQ-0001",
                "domain": "schedule_window",
                "severity": "error",
                "code": "overwrite_requires_flag",
                "message": "overwrite",
            }
        ],
    )

    assert {"ingest_errors", "overwrite_requires_flag"}.issubset(_codes(report))


def test_gate_fails_on_non_p0_apply_target() -> None:
    report = _report(
        validation_apply_plan=[_apply_plan("venue_ticket")],
        ingest_plan=[_ingest_plan("venue_ticket", action="insert")],
    )

    assert {"non_p0_apply_eligible", "non_p0_ingest_action"}.issubset(_codes(report))


def test_gate_fails_when_no_rows_are_apply_eligible() -> None:
    report = _report(
        validation_summary=_validation_summary(apply_eligible_count=0),
        validation_apply_plan=[_apply_plan(apply_eligible="false", skip_reason="operator_status_pending")],
        ingest_plan=[_ingest_plan(action="skipped", skip_reason="operator_status_pending")],
    )

    assert "no_apply_eligible_rows" in _codes(report)


def test_gate_reports_manual_required_rows_from_pending_p0_normalized_rows() -> None:
    report = _report(
        validation_summary=_validation_summary(apply_eligible_count=0),
        validation_apply_plan=[
            _apply_plan(apply_eligible="false", skip_reason="operator_status_pending")
        ],
        ingest_plan=[_ingest_plan(action="skipped", skip_reason="operator_status_pending")],
        validation_normalized_rows=[_normalized_row()],
    )

    manual_rows = report["manual_required_rows"]
    assert report["summary"]["manual_required_count"] == 1
    assert report["summary"]["manual_required_domain_counts"]["schedule_window"] == 1
    assert report["summary"]["manual_required_skip_reason_counts"] == {
        "operator_status_pending": 1
    }
    assert manual_rows[0]["queue_id"] == "ODQ-0001"
    assert manual_rows[0]["manual_contract"] == "MANUAL_BASEBALL_DATA_REQUIRED"
    assert "game_date" in manual_rows[0]["missing_required_fields"]
    handoff = gate._render_handoff(report)
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in handoff
    assert "ODQ-0001" in handoff


def test_gate_does_not_count_ready_rows_blocked_only_by_db_checks_as_manual_required() -> None:
    report = _report(
        validation_summary=_validation_summary(
            apply_eligible_count=0,
            db_checks={"skipped": True, "skip_reason": "POSTGRES_DB_URL is not set"},
        ),
        validation_apply_plan=[_apply_plan(apply_eligible="false", skip_reason="db_checks_skipped")],
        ingest_plan=[_ingest_plan(action="skipped", skip_reason="db_checks_skipped")],
        validation_normalized_rows=[
            _normalized_row(
                operator_status="ready_for_validation",
                apply_eligible=False,
                skip_reason="db_checks_skipped",
                payload={
                    "game_date": "2026-06-07",
                    "game_id": "20260607LGDOO0",
                    "home_team": "LG",
                    "away_team": "DOO",
                    "stadium_name": "Jamsil",
                    "start_time": "18:30",
                    "game_status": "scheduled",
                    "source_name": "operator",
                    "source_checked_at": "2026-06-07",
                    "is_verified": "true",
                    "confidence": "0.95",
                },
            )
        ],
    )

    assert report["summary"]["manual_required_count"] == 0
    assert report["manual_required_rows"] == []


def test_run_gate_writes_manual_required_csv_and_handoff(tmp_path: Path) -> None:
    validation_dir = tmp_path / "validation"
    ingest_dir = tmp_path / "ingest"
    output_dir = tmp_path / "gate"
    normalized_row = _normalized_row()
    _write_json(
        validation_dir / "operator_data_validation_summary.json",
        _validation_summary(apply_eligible_count=0),
    )
    _write_csv(
        validation_dir / "operator_data_apply_plan.csv",
        [_apply_plan(apply_eligible="false", skip_reason="operator_status_pending")],
        gate.APPLY_PLAN_FIELDNAMES if hasattr(gate, "APPLY_PLAN_FIELDNAMES") else [
            "queue_id",
            "priority",
            "domain",
            "operator_status",
            "validation_status",
            "apply_eligible",
            "apply_target",
            "issue_count",
            "skip_reason",
        ],
    )
    _write_csv(
        validation_dir / "operator_data_validation_issues.csv",
        [],
        ["queue_id", "domain", "field_name", "severity", "code", "message"],
    )
    (validation_dir / "operator_data_normalized_rows.jsonl").write_text(
        json.dumps(normalized_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_json(ingest_dir / "operator_data_ingest_summary.json", _ingest_summary(action_counts={"skipped": 1}))
    _write_csv(
        ingest_dir / "operator_data_ingest_plan.csv",
        [_ingest_plan(action="skipped", skip_reason="operator_status_pending")],
        [
            "queue_id",
            "domain",
            "operator_status",
            "validation_status",
            "apply_eligible",
            "action",
            "apply_target",
            "payload_hash",
            "issue_count",
            "skip_reason",
        ],
    )
    _write_csv(
        ingest_dir / "operator_data_ingest_issues.csv",
        [],
        ["queue_id", "domain", "severity", "code", "message"],
    )

    report = gate.run_gate(
        validation_dir=validation_dir,
        ingest_dir=ingest_dir,
        output_dir=output_dir,
    )

    manual_rows = _read_csv(output_dir / "manual_baseball_data_required_rows.csv")
    handoff = (output_dir / "handoff.md").read_text(encoding="utf-8")
    assert report["summary"]["manual_required_count"] == 1
    assert manual_rows[0]["queue_id"] == "ODQ-0001"
    assert "manual_baseball_data_required_rows.csv" in handoff
