from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import summarize_operator_data_p0_recovery_status as status


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=status.BLOCKER_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in status.BLOCKER_FIELDNAMES})


def _base_inputs(**overrides: Any) -> dict[str, Any]:
    data = {
        "packet_summary": {
            "total_queue_items": 86,
            "total_field_rows": 894,
            "domain_counts": {
                "season_meta": 2,
                "schedule_window": 38,
                "game_day_lineup": 24,
                "roster_news": 22,
            },
            "status_counts": {"pending": 86},
        },
        "audit_report": {
            "summary": {
                "status": "pass",
                "ready_or_validated_count": 2,
                "recovery_candidate_count": 2,
                "manual_fallback_count": 84,
                "blocked_ready_count": 0,
                "issue_counts": {"error": 0, "warning": 0},
            },
            "issues": [],
        },
        "db_prereq_report": {
            "summary": {
                "status": "pass",
                "issue_counts": {"error": 0, "warning": 0},
            },
            "issues": [],
        },
        "validation_summary": {
            "issue_counts": {"error": 0, "warning": 0},
            "db_checks": {"skipped": False, "skip_reason": ""},
            "apply_eligible_count": 2,
        },
        "ingest_summary": {
            "issue_counts": {"error": 0, "warning": 0},
            "action_counts": {"insert": 2, "skipped": 84},
        },
        "gate_report": {
            "summary": {
                "status": "pass",
                "issue_counts": {"error": 0},
                "apply_eligible_count": 2,
            },
            "issues": [],
        },
        "gate_issues": [],
    }
    data.update(overrides)
    return data


def _report(**overrides: Any) -> dict[str, Any]:
    data = _base_inputs(**overrides)
    return status.build_status_report(
        packet_summary=data["packet_summary"],
        audit_report=data["audit_report"],
        db_prereq_report=data["db_prereq_report"],
        validation_summary=data["validation_summary"],
        ingest_summary=data["ingest_summary"],
        gate_report=data["gate_report"],
        gate_issues=data["gate_issues"],
        packet_dir=Path("packet"),
        audit_dir=Path("audit"),
        db_prereq_dir=Path("db_prereq"),
        validation_dir=Path("validation"),
        ingest_dir=Path("ingest"),
        gate_dir=Path("gate"),
    )


def _codes(report: Mapping[str, Any]) -> set[str]:
    return {str(blocker["code"]) for blocker in report["blockers"]}


def test_status_passes_when_all_artifacts_are_ready() -> None:
    report = _report()

    assert report["summary"]["status"] == "ready_for_controlled_apply"
    assert report["summary"]["blocker_count"] == 0
    assert report["blockers"] == []


def test_status_blocks_current_pending_no_db_gate_state() -> None:
    report = _report(
        audit_report={
            "summary": {
                "status": "warning",
                "ready_or_validated_count": 0,
                "recovery_candidate_count": 0,
                "manual_fallback_count": 86,
                "blocked_ready_count": 0,
                "issue_counts": {"error": 0, "warning": 1},
            }
        },
        validation_summary={
            "issue_counts": {"error": 0, "warning": 1},
            "db_checks": {"skipped": True, "skip_reason": "--skip-db-checks was set"},
            "apply_eligible_count": 0,
        },
        ingest_summary={
            "issue_counts": {"error": 0, "warning": 0},
            "action_counts": {"skipped": 86},
        },
        gate_report={
            "summary": {
                "status": "fail",
                "issue_counts": {"error": 2},
                "apply_eligible_count": 0,
            }
        },
        gate_issues=[
            {
                "severity": "error",
                "code": "db_checks_skipped",
                "message": "Strict recovery gate requires DB checks to run.",
                "source": "validation_summary",
            },
            {
                "severity": "error",
                "code": "no_apply_eligible_rows",
                "message": "No apply-eligible P0 rows are available for recovery.",
                "source": "validation_summary",
            },
        ],
    )

    assert report["summary"]["status"] == "blocked"
    assert {
        "operator_input_missing",
        "db_checks_skipped",
        "no_apply_eligible_rows",
        "recovery_gate_not_passed",
    }.issubset(_codes(report))
    assert report["summary"]["manual_fallback_count"] == 86


def test_status_propagates_manual_required_counts_from_gate() -> None:
    report = status.build_status_report(
        packet_summary=_base_inputs()["packet_summary"],
        audit_report={
            "summary": {
                "status": "warning",
                "ready_or_validated_count": 0,
                "recovery_candidate_count": 0,
                "manual_fallback_count": 86,
                "blocked_ready_count": 0,
                "issue_counts": {"error": 0, "warning": 1},
            }
        },
        db_prereq_report=_base_inputs()["db_prereq_report"],
        validation_summary={
            "issue_counts": {"error": 0, "warning": 0},
            "db_checks": {"skipped": False, "skip_reason": ""},
            "apply_eligible_count": 0,
        },
        ingest_summary={
            "issue_counts": {"error": 0, "warning": 0},
            "action_counts": {"skipped": 86},
        },
        gate_report={
            "summary": {
                "status": "fail",
                "issue_counts": {"error": 1},
                "apply_eligible_count": 0,
                "manual_required_count": 86,
                "manual_required_domain_counts": {
                    "season_meta": 2,
                    "schedule_window": 38,
                    "game_day_lineup": 24,
                    "roster_news": 22,
                },
                "manual_required_skip_reason_counts": {
                    "operator_status_pending": 86,
                },
            }
        },
        gate_issues=[
            {
                "severity": "error",
                "code": "no_apply_eligible_rows",
                "message": "No apply-eligible P0 rows are available for recovery.",
                "source": "validation_summary",
            }
        ],
        gate_manual_required_rows=[
            {
                "queue_id": "ODQ-0001",
                "domain": "season_meta",
                "operator_status": "pending",
                "skip_reason": "operator_status_pending",
                "question": "2026 KBO 개막일은 언제야?",
                "required_fields": "season_year|event_name",
                "missing_required_fields": "season_year|event_name",
                "manual_contract": "MANUAL_BASEBALL_DATA_REQUIRED",
            }
        ],
        packet_dir=Path("packet"),
        audit_dir=Path("audit"),
        db_prereq_dir=Path("db_prereq"),
        validation_dir=Path("validation"),
        ingest_dir=Path("ingest"),
        gate_dir=Path("gate"),
    )

    assert report["summary"]["manual_required_count"] == 86
    assert report["summary"]["manual_required_domain_counts"]["schedule_window"] == 38
    assert report["summary"]["manual_required_skip_reason_counts"] == {
        "operator_status_pending": 86
    }
    handoff = status._render_handoff(report)
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in handoff
    assert "season_meta=2" in handoff
    assert "../gate/manual_baseball_data_required_rows.csv" in handoff


def test_status_blocks_audit_and_validation_errors() -> None:
    report = _report(
        audit_report={
            "summary": {
                "status": "fail",
                "ready_or_validated_count": 1,
                "recovery_candidate_count": 0,
                "manual_fallback_count": 86,
                "blocked_ready_count": 1,
                "issue_counts": {"error": 2, "warning": 0},
            }
        },
        validation_summary={
            "issue_counts": {"error": 1, "warning": 0},
            "db_checks": {"skipped": False},
            "apply_eligible_count": 0,
        },
    )

    assert report["summary"]["status"] == "blocked"
    assert {"packet_audit_failed", "ready_rows_blocked", "validation_errors"}.issubset(
        _codes(report)
    )


def test_status_blocks_failed_db_prereqs() -> None:
    report = _report(
        db_prereq_report={
            "summary": {
                "status": "fail",
                "issue_counts": {"error": 1, "warning": 0},
            },
            "issues": [
                {
                    "severity": "error",
                    "code": "missing_lineup_conflict_target",
                    "message": "game_lineups conflict target missing.",
                }
            ],
        }
    )

    assert report["summary"]["status"] == "blocked"
    assert {"db_prereqs_failed", "missing_lineup_conflict_target"}.issubset(_codes(report))
    assert report["summary"]["db_prereq_status"] == "fail"


def test_run_summary_writes_json_csv_and_handoff(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    audit_dir = tmp_path / "audit"
    db_prereq_dir = tmp_path / "db_prereq"
    validation_dir = tmp_path / "validation"
    ingest_dir = tmp_path / "ingest"
    gate_dir = tmp_path / "gate"
    output_dir = tmp_path / "status"
    data = _base_inputs()
    _write_json(packet_dir / "p0_input_summary.json", data["packet_summary"])
    _write_json(audit_dir / "p0_input_audit_summary.json", data["audit_report"])
    _write_json(db_prereq_dir / "db_prereq_summary.json", data["db_prereq_report"])
    _write_json(validation_dir / "operator_data_validation_summary.json", data["validation_summary"])
    _write_json(ingest_dir / "operator_data_ingest_summary.json", data["ingest_summary"])
    _write_json(gate_dir / "summary.json", data["gate_report"])
    _write_csv(gate_dir / "issues.csv", data["gate_issues"])

    report = status.run_summary(
        packet_dir=packet_dir,
        audit_dir=audit_dir,
        db_prereq_dir=db_prereq_dir,
        validation_dir=validation_dir,
        ingest_dir=ingest_dir,
        gate_dir=gate_dir,
        output_dir=output_dir,
    )

    assert report["summary"]["status"] == "ready_for_controlled_apply"
    assert (output_dir / "p0_recovery_status_summary.json").exists()
    assert (output_dir / "p0_recovery_status_blockers.csv").exists()
    assert (output_dir / "p0_recovery_status_handoff.md").exists()
    payload = json.loads(
        (output_dir / "p0_recovery_status_summary.json").read_text(encoding="utf-8")
    )
    assert payload["summary"]["status"] == "ready_for_controlled_apply"


def test_main_returns_nonzero_when_blocked(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    audit_dir = tmp_path / "audit"
    db_prereq_dir = tmp_path / "db_prereq"
    validation_dir = tmp_path / "validation"
    ingest_dir = tmp_path / "ingest"
    gate_dir = tmp_path / "gate"
    output_dir = tmp_path / "status"
    data = _base_inputs(
        audit_report={
            "summary": {
                "status": "warning",
                "ready_or_validated_count": 0,
                "recovery_candidate_count": 0,
                "manual_fallback_count": 86,
                "blocked_ready_count": 0,
            }
        }
    )
    _write_json(packet_dir / "p0_input_summary.json", data["packet_summary"])
    _write_json(audit_dir / "p0_input_audit_summary.json", data["audit_report"])
    _write_json(db_prereq_dir / "db_prereq_summary.json", data["db_prereq_report"])
    _write_json(validation_dir / "operator_data_validation_summary.json", data["validation_summary"])
    _write_json(ingest_dir / "operator_data_ingest_summary.json", data["ingest_summary"])
    _write_json(gate_dir / "summary.json", data["gate_report"])
    _write_csv(gate_dir / "issues.csv", data["gate_issues"])

    exit_code = status.main(
        [
            "--packet-dir",
            str(packet_dir),
            "--audit-dir",
            str(audit_dir),
            "--db-prereq-dir",
            str(db_prereq_dir),
            "--validation-dir",
            str(validation_dir),
            "--ingest-dir",
            str(ingest_dir),
            "--gate-dir",
            str(gate_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 1
