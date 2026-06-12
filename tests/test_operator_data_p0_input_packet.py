from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from scripts import build_operator_data_p0_input_packet as packet


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
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


def test_packet_filters_to_p0_domains_and_preserves_contract_fields(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.csv"
    fields_path = tmp_path / "fields.csv"
    output_dir = tmp_path / "packet"
    _write_csv(
        queue_path,
        packet.QUEUE_FIELDNAMES,
        [
            _queue_row("ODQ-0001", domain="schedule_window"),
            _queue_row("ODQ-0002", domain="game_day_lineup", question="라인업 알려줘."),
            _queue_row("ODQ-0003", priority="P1", domain="operator_venue_guides"),
            _queue_row("ODQ-0004", priority="P3", domain="subjective_prediction"),
        ],
    )
    _write_csv(
        fields_path,
        packet.FIELDS_FIELDNAMES,
        [
            _field_row("ODQ-0001", "game_id", domain="schedule_window"),
            _field_row("ODQ-0001", "source_name", domain="schedule_window"),
            _field_row("ODQ-0002", "player_name", domain="game_day_lineup"),
            _field_row("ODQ-0003", "venue_name", domain="operator_venue_guides"),
            _field_row("ODQ-0004", "prediction_basis", domain="subjective_prediction"),
        ],
    )

    summary = packet.build_packet(queue_path=queue_path, fields_path=fields_path, output_dir=output_dir)
    queue_rows = _read_csv(output_dir / "p0_queue.csv")
    field_rows = _read_csv(output_dir / "p0_fields.csv")

    assert summary["total_queue_items"] == 2
    assert summary["domain_counts"]["schedule_window"] == 1
    assert summary["domain_counts"]["game_day_lineup"] == 1
    assert {row["queue_id"] for row in queue_rows} == {"ODQ-0001", "ODQ-0002"}
    assert {row["domain"] for row in queue_rows}.issubset(packet.P0_DOMAINS)
    assert {row["queue_id"] for row in field_rows} == {"ODQ-0001", "ODQ-0002"}
    assert {row["domain"] for row in field_rows}.issubset(packet.P0_DOMAINS)
    assert all(row["contract_code"] == "MANUAL_BASEBALL_DATA_REQUIRED" for row in queue_rows)


def test_packet_does_not_autofill_operator_values(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.csv"
    fields_path = tmp_path / "fields.csv"
    output_dir = tmp_path / "packet"
    _write_csv(queue_path, packet.QUEUE_FIELDNAMES, [_queue_row("ODQ-0001")])
    _write_csv(
        fields_path,
        packet.FIELDS_FIELDNAMES,
        [
            _field_row("ODQ-0001", "source_name"),
            _field_row("ODQ-0001", "operator_checked_note", operator_value="already operator-filled"),
        ],
    )

    packet.build_packet(queue_path=queue_path, fields_path=fields_path, output_dir=output_dir)
    field_rows = _read_csv(output_dir / "p0_fields.csv")

    values_by_field = {row["field_name"]: row["operator_value"] for row in field_rows}
    assert values_by_field["source_name"] == ""
    assert values_by_field["operator_checked_note"] == "already operator-filled"


def test_packet_generates_checklist_and_summary_files(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.csv"
    fields_path = tmp_path / "fields.csv"
    output_dir = tmp_path / "packet"
    _write_csv(queue_path, packet.QUEUE_FIELDNAMES, [_queue_row("ODQ-0001", domain="roster_news")])
    _write_csv(
        fields_path,
        packet.FIELDS_FIELDNAMES,
        [_field_row("ODQ-0001", "effective_date", domain="roster_news")],
    )

    packet.build_packet(queue_path=queue_path, fields_path=fields_path, output_dir=output_dir)
    summary = json.loads((output_dir / "p0_input_summary.json").read_text(encoding="utf-8"))
    checklist = (output_dir / "p0_input_checklist.md").read_text(encoding="utf-8")

    assert summary["total_queue_items"] == 1
    assert summary["total_field_rows"] == 1
    assert summary["source_queue_path"] == str(queue_path)
    assert "p0_queue.csv" in summary["generated_files"]["p0_queue"]
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in checklist
    assert "source_name" in checklist
    assert "confidence" in checklist
    assert "game_day_lineup" in checklist


def test_real_handoff_packet_extracts_current_p0_bundle(tmp_path: Path) -> None:
    queue_path = packet.DEFAULT_QUEUE_INPUT
    fields_path = packet.DEFAULT_FIELDS_INPUT
    if not queue_path.exists() or not fields_path.exists():
        pytest.skip("local operator-data handoff reports are not present")

    output_dir = tmp_path / "packet"

    summary = packet.build_packet(queue_path=queue_path, fields_path=fields_path, output_dir=output_dir)
    queue_rows = _read_csv(output_dir / "p0_queue.csv")
    field_rows = _read_csv(output_dir / "p0_fields.csv")

    assert summary["total_queue_items"] == 86
    assert summary["total_field_rows"] == 894
    assert summary["domain_counts"] == {
        "season_meta": 2,
        "schedule_window": 38,
        "game_day_lineup": 24,
        "roster_news": 22,
    }
    assert summary["status_counts"] == {"pending": 86}
    assert len(queue_rows) == 86
    assert len(field_rows) == 894
    assert {row["priority"] for row in queue_rows} == {"P0"}
    assert {row["domain"] for row in queue_rows}.issubset(packet.P0_DOMAINS)
    assert {row["domain"] for row in field_rows}.issubset(packet.P0_DOMAINS)
    assert all(row["operator_value"] == "" for row in field_rows)
