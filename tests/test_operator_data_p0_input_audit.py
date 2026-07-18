from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from scripts import audit_operator_data_p0_input_packet as audit
from scripts import build_operator_data_p0_input_packet as packet


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _queue_row(
    queue_id: str,
    *,
    priority: str = "P0",
    domain: str = "schedule_window",
    question: str = "오늘 KBO 경기 일정 알려줘.",
    operator_status: str = "pending",
) -> dict[str, str]:
    return {
        "queue_id": queue_id,
        "priority": priority,
        "priority_reason": "manual baseball data required",
        "domain": domain,
        "contract_code": "MANUAL_BASEBALL_DATA_REQUIRED",
        "question": question,
        "required_fields": "source_name|source_checked_at|is_verified|confidence",
        "endpoint_count": "2",
        "endpoints": "/ai/chat/completion|/ai/chat/stream",
        "sample_answer": "",
        "operator_status": operator_status,
        "operator_owner": "",
        "operator_notes": "",
    }


def _field_row(
    queue_id: str,
    field_name: str,
    *,
    domain: str = "schedule_window",
    question: str = "오늘 KBO 경기 일정 알려줘.",
    operator_value: str = "",
) -> dict[str, str]:
    return {
        "queue_id": queue_id,
        "domain": domain,
        "contract_code": "MANUAL_BASEBALL_DATA_REQUIRED",
        "question": question,
        "field_name": field_name,
        "field_description": "Operator-provided value.",
        "required": "true",
        "operator_value": operator_value,
        "operator_notes": "",
    }


def _schedule_fields(queue_id: str, **overrides: str) -> list[dict[str, str]]:
    values = {
        "game_id": "20260607-LG-DOO-1",
        "game_date": "2026-06-07",
        "home_team": "LG",
        "away_team": "DOO",
        "game_status": "scheduled",
        "start_time": "18:30",
        "stadium_name": "Jamsil",
        "source_name": "operator internal sheet",
        "source_checked_at": "2026-06-07T09:00:00+09:00",
        "is_verified": "true",
        "confidence": "0.90",
    }
    values.update(overrides)
    return [
        _field_row(queue_id, field_name, operator_value=value)
        for field_name, value in values.items()
    ]


def _write_packet_pair(
    base: Path,
    *,
    source_queue_rows: Sequence[Mapping[str, Any]],
    source_field_rows: Sequence[Mapping[str, Any]],
    packet_queue_rows: Sequence[Mapping[str, Any]] | None = None,
    packet_field_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[Path, Path, Path, Path]:
    source_queue = base / "source_queue.csv"
    source_fields = base / "source_fields.csv"
    queue = base / "p0_queue.csv"
    fields = base / "p0_fields.csv"
    _write_csv(source_queue, audit.QUEUE_FIELDNAMES, source_queue_rows)
    _write_csv(source_fields, audit.FIELDS_FIELDNAMES, source_field_rows)
    _write_csv(queue, audit.QUEUE_FIELDNAMES, packet_queue_rows or source_queue_rows)
    _write_csv(fields, audit.FIELDS_FIELDNAMES, packet_field_rows or source_field_rows)
    return queue, fields, source_queue, source_fields


def _run(
    tmp_path: Path,
    *,
    source_queue_rows: Sequence[Mapping[str, Any]],
    source_field_rows: Sequence[Mapping[str, Any]],
    packet_queue_rows: Sequence[Mapping[str, Any]] | None = None,
    packet_field_rows: Sequence[Mapping[str, Any]] | None = None,
    require_ready: bool = False,
) -> dict[str, Any]:
    queue, fields, source_queue, source_fields = _write_packet_pair(
        tmp_path,
        source_queue_rows=source_queue_rows,
        source_field_rows=source_field_rows,
        packet_queue_rows=packet_queue_rows,
        packet_field_rows=packet_field_rows,
    )
    return audit.run_audit(
        queue_path=queue,
        fields_path=fields,
        source_queue_path=source_queue,
        source_fields_path=source_fields,
        output_dir=tmp_path / "audit",
        require_ready=require_ready,
    )


def _codes(report: Mapping[str, Any]) -> set[str]:
    return {str(issue["code"]) for issue in report["issues"]}


def test_current_all_pending_packet_warns_with_manual_fallback_86(
    tmp_path: Path,
) -> None:
    if (
        not packet.DEFAULT_QUEUE_INPUT.exists()
        or not packet.DEFAULT_FIELDS_INPUT.exists()
    ):
        pytest.skip("local operator-data handoff reports are not present")

    packet_dir = tmp_path / "packet"
    packet.build_packet(
        queue_path=packet.DEFAULT_QUEUE_INPUT,
        fields_path=packet.DEFAULT_FIELDS_INPUT,
        output_dir=packet_dir,
    )

    report = audit.run_audit(
        queue_path=packet_dir / "p0_queue.csv",
        fields_path=packet_dir / "p0_fields.csv",
        source_queue_path=packet.DEFAULT_QUEUE_INPUT,
        source_fields_path=packet.DEFAULT_FIELDS_INPUT,
        output_dir=tmp_path / "audit",
    )

    assert report["summary"]["status"] == "warning"
    assert report["summary"]["ready_or_validated_count"] == 0
    assert report["summary"]["manual_fallback_count"] == 86
    assert report["summary"]["issue_counts"] == {"error": 0, "warning": 1}
    assert _codes(report) == {"no_ready_p0_rows"}


def test_partial_ready_complete_packet_passes_with_pending_manual_fallback(
    tmp_path: Path,
) -> None:
    report = _run(
        tmp_path,
        source_queue_rows=[
            _queue_row("ODQ-0001", operator_status="pending"),
            _queue_row(
                "ODQ-0002", operator_status="pending", question="내일 KBO 일정 알려줘."
            ),
        ],
        source_field_rows=[
            *_schedule_fields("ODQ-0001"),
            *_schedule_fields("ODQ-0002", game_id="20260608-LG-DOO-1"),
        ],
        packet_queue_rows=[
            _queue_row("ODQ-0001", operator_status="ready_for_validation"),
            _queue_row(
                "ODQ-0002", operator_status="pending", question="내일 KBO 일정 알려줘."
            ),
        ],
        packet_field_rows=[
            *_schedule_fields("ODQ-0001"),
            *_schedule_fields("ODQ-0002", game_id="20260608-LG-DOO-1", source_name=""),
        ],
    )

    assert report["summary"]["status"] == "pass"
    assert report["summary"]["ready_or_validated_count"] == 1
    assert report["summary"]["recovery_candidate_count"] == 1
    assert report["summary"]["manual_fallback_count"] == 1
    assert report["issues"] == []
    plan = {row["queue_id"]: row for row in report["readiness_plan"]}
    assert plan["ODQ-0001"]["readiness_status"] == "recovery_candidate"
    assert plan["ODQ-0002"]["readiness_status"] == "manual_fallback"


def test_ready_row_missing_required_operator_value_fails(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        source_queue_rows=[_queue_row("ODQ-0001")],
        source_field_rows=_schedule_fields("ODQ-0001"),
        packet_queue_rows=[
            _queue_row("ODQ-0001", operator_status="ready_for_validation")
        ],
        packet_field_rows=_schedule_fields("ODQ-0001", game_id=""),
    )

    assert report["summary"]["status"] == "fail"
    assert "missing_required_operator_value" in _codes(report)
    assert report["summary"]["blocked_ready_count"] == 1


def test_ready_row_requires_verified_source_and_minimum_confidence(
    tmp_path: Path,
) -> None:
    report = _run(
        tmp_path,
        source_queue_rows=[_queue_row("ODQ-0001")],
        source_field_rows=_schedule_fields("ODQ-0001"),
        packet_queue_rows=[_queue_row("ODQ-0001", operator_status="validated")],
        packet_field_rows=_schedule_fields(
            "ODQ-0001",
            is_verified="false",
            confidence="0.69",
        ),
    )

    assert report["summary"]["status"] == "fail"
    assert {"source_not_verified", "confidence_below_minimum"}.issubset(_codes(report))


def test_non_p0_domain_and_immutable_drift_fail(tmp_path: Path) -> None:
    source_queue = [_queue_row("ODQ-0001")]
    source_fields = _schedule_fields("ODQ-0001")
    packet_queue = [
        _queue_row(
            "ODQ-0001",
            question="질문을 바꿔버림",
            operator_status="ready_for_validation",
        ),
        _queue_row("ODQ-0002", priority="P1", domain="venue_ticket"),
    ]
    packet_fields = [
        *_schedule_fields("ODQ-0001"),
        _field_row(
            "ODQ-0002", "venue_name", domain="venue_ticket", operator_value="Jamsil"
        ),
    ]

    report = _run(
        tmp_path,
        source_queue_rows=source_queue,
        source_field_rows=source_fields,
        packet_queue_rows=packet_queue,
        packet_field_rows=packet_fields,
    )

    assert report["summary"]["status"] == "fail"
    assert {
        "non_p0_queue_domain",
        "non_p0_field_domain",
        "immutable_queue_drift",
    }.issubset(_codes(report))


def test_require_ready_turns_all_pending_warning_into_failure(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        source_queue_rows=[_queue_row("ODQ-0001")],
        source_field_rows=_schedule_fields("ODQ-0001"),
        require_ready=True,
    )

    assert report["summary"]["status"] == "fail"
    assert "no_ready_p0_rows" in _codes(report)


def test_cli_writes_outputs_and_returns_policy_exit_codes(tmp_path: Path) -> None:
    queue, fields, source_queue, source_fields = _write_packet_pair(
        tmp_path,
        source_queue_rows=[_queue_row("ODQ-0001")],
        source_field_rows=_schedule_fields("ODQ-0001"),
    )
    output_dir = tmp_path / "audit"

    warning_exit = audit.main(
        [
            "--queue",
            str(queue),
            "--fields",
            str(fields),
            "--source-queue",
            str(source_queue),
            "--source-fields",
            str(source_fields),
            "--output-dir",
            str(output_dir),
        ]
    )
    fail_exit = audit.main(
        [
            "--queue",
            str(queue),
            "--fields",
            str(fields),
            "--source-queue",
            str(source_queue),
            "--source-fields",
            str(source_fields),
            "--output-dir",
            str(output_dir / "require_ready"),
            "--require-ready",
        ]
    )

    assert warning_exit == 0
    assert fail_exit == 1
    assert (output_dir / "p0_input_audit_summary.json").exists()
    assert (output_dir / "p0_input_audit_issues.csv").exists()
    assert (output_dir / "p0_input_readiness_plan.csv").exists()
    assert (output_dir / "p0_input_audit_handoff.md").exists()
    payload = json.loads(
        (output_dir / "p0_input_audit_summary.json").read_text(encoding="utf-8")
    )
    issues = _read_csv(output_dir / "p0_input_audit_issues.csv")
    readiness = _read_csv(output_dir / "p0_input_readiness_plan.csv")
    assert payload["summary"]["status"] == "warning"
    assert issues[0]["code"] == "no_ready_p0_rows"
    assert readiness[0]["manual_fallback"] == "true"
