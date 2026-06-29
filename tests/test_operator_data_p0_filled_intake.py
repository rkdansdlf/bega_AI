from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import run_operator_data_p0_filled_intake as runner


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _queue_row(
    queue_id: str,
    *,
    domain: str = "schedule_window",
    question: str = "오늘 KBO 경기 일정 알려줘.",
    operator_status: str = "pending",
    priority: str = "P0",
) -> dict[str, str]:
    required_fields = (
        "game_date|game_id|home_team|away_team|stadium_name|start_time|game_status|"
        "source_name|source_checked_at|is_verified|confidence"
    )
    return {
        "queue_id": queue_id,
        "priority": priority,
        "priority_reason": "manual baseball data required",
        "domain": domain,
        "contract_code": "MANUAL_BASEBALL_DATA_REQUIRED",
        "question": question,
        "required_fields": required_fields,
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
    operator_value: str = "",
    *,
    domain: str = "schedule_window",
    question: str = "오늘 KBO 경기 일정 알려줘.",
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
        "game_date": "2026-06-07",
        "game_id": "20260607LGDOO0",
        "home_team": "LG",
        "away_team": "DOO",
        "stadium_name": "Jamsil",
        "start_time": "18:30",
        "game_status": "scheduled",
        "source_name": "operator internal sheet",
        "source_checked_at": "2026-06-07T09:00:00+09:00",
        "is_verified": "true",
        "confidence": "0.95",
    }
    values.update(overrides)
    return [
        _field_row(queue_id, field_name, value) for field_name, value in values.items()
    ]


def _write_packet(
    tmp_path: Path,
    *,
    source_queue_rows: Sequence[Mapping[str, Any]],
    source_field_rows: Sequence[Mapping[str, Any]],
    packet_queue_rows: Sequence[Mapping[str, Any]] | None = None,
    packet_field_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[Path, Path, Path, Path]:
    queue = tmp_path / "p0_queue.csv"
    fields = tmp_path / "p0_fields.csv"
    source_queue = tmp_path / "source_queue.csv"
    source_fields = tmp_path / "source_fields.csv"
    _write_csv(source_queue, runner.audit.QUEUE_FIELDNAMES, source_queue_rows)
    _write_csv(source_fields, runner.audit.FIELDS_FIELDNAMES, source_field_rows)
    _write_csv(
        queue, runner.audit.QUEUE_FIELDNAMES, packet_queue_rows or source_queue_rows
    )
    _write_csv(
        fields, runner.audit.FIELDS_FIELDNAMES, packet_field_rows or source_field_rows
    )
    return queue, fields, source_queue, source_fields


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


class FakeDbChecker:
    db_checks_skipped = False
    db_skip_reason = ""

    def get_game(self, game_id: str) -> Mapping[str, Any] | None:
        if game_id == "20260607LGDOO0":
            return {"game_id": game_id, "home_team": "LG", "away_team": "DOO"}
        return None

    def count_players(self, player_name: str) -> int:
        del player_name
        return 1

    def is_known_team_code(
        self, team_code: str, season_year: int | None = None
    ) -> bool:
        del team_code, season_year
        return True

    def close(self) -> None:
        return None


class FakeCursor:
    def __init__(self) -> None:
        self.columns_by_table = {
            table_name: set(columns)
            for table_name, columns in runner.ingest.REQUIRED_TABLE_COLUMNS.items()
        }
        self._rows: list[Mapping[str, Any]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        lowered = query.lower()
        if "select payload_hash from operator_data_items" in lowered:
            self._rows = []
        elif "from information_schema.columns" in lowered:
            table_name = str(params[0])
            self._rows = [
                {"column_name": column}
                for column in sorted(self.columns_by_table.get(table_name, set()))
            ]
        elif "from pg_index" in lowered:
            self._rows = [{"exists": 1}]
        else:
            self._rows = []

    def fetchone(self) -> Mapping[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Mapping[str, Any]]:
        return list(self._rows)


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = FakeCursor()
        self.closed = False

    def cursor(self, *args, **kwargs) -> FakeCursor:
        del args, kwargs
        return self.cursor_obj

    def close(self) -> None:
        self.closed = True


def _fake_db_prereq_pass(output_dir: Path) -> dict[str, Any]:
    report = {
        "summary": {
            "generated_at_utc": "2026-06-07T00:00:00+00:00",
            "status": "pass",
            "checked_table_count": 1,
            "missing_column_count": 0,
            "lineup_conflict_target_exists": True,
            "issue_counts": {"error": 0, "warning": 0},
        },
        "table_results": [],
        "issues": [],
    }
    runner.db_prereqs._write_report(output_dir, report)
    return report


def test_pending_only_packet_creates_full_bundle_and_exits_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    queue, fields, source_queue, source_fields = _write_packet(
        tmp_path,
        source_queue_rows=[_queue_row("ODQ-0001")],
        source_field_rows=_schedule_fields("ODQ-0001", source_name=""),
    )
    output_dir = tmp_path / "intake"
    monkeypatch.delenv("POSTGRES_DB_URL", raising=False)

    exit_code = runner.main(
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

    assert exit_code == 1
    assert (output_dir / "packet_snapshot" / "p0_queue.csv").exists()
    assert (output_dir / "audit" / "p0_input_audit_summary.json").exists()
    assert (output_dir / "db_prereqs" / "db_prereq_summary.json").exists()
    assert (
        output_dir / "validation" / "operator_data_validation_summary.json"
    ).exists()
    assert (output_dir / "ingest" / "operator_data_ingest_summary.json").exists()
    assert (output_dir / "gate" / "summary.json").exists()
    assert (output_dir / "status" / "p0_recovery_status_summary.json").exists()
    summary = json.loads(
        (output_dir / "intake_summary.json").read_text(encoding="utf-8")
    )
    assert summary["summary"]["status"] == "blocked"
    assert summary["summary"]["nonpassing_stage_count"] == 5
    stages = {row["name"]: row for row in _read_csv(output_dir / "intake_stages.csv")}
    assert stages["audit"]["code"] == "no_ready_p0_rows"
    assert (
        stages["audit"]["message"]
        == "No P0 rows are marked ready_for_validation or validated."
    )
    assert stages["db_prereqs"]["code"] == "db_url_missing"
    assert stages["validation"]["code"] == "db_checks_skipped"
    assert stages["validation"]["message"] == "POSTGRES_DB_URL is not set"
    assert stages["recovery_gate"]["code"] == "db_checks_skipped"
    assert stages["status_summary"]["code"] == "packet_audit_failed"
    validation_summary = json.loads(
        (output_dir / "validation" / "operator_data_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    gate_summary = json.loads(
        (output_dir / "gate" / "summary.json").read_text(encoding="utf-8")
    )
    manual_rows = _read_csv(
        output_dir / "gate" / "manual_baseball_data_required_rows.csv"
    )
    assert (
        validation_summary["db_checks"]["skip_reason"] == "POSTGRES_DB_URL is not set"
    )
    assert gate_summary["summary"]["manual_required_count"] == 1
    assert manual_rows[0]["queue_id"] == "ODQ-0001"
    assert manual_rows[0]["manual_contract"] == "MANUAL_BASEBALL_DATA_REQUIRED"


def test_partially_ready_valid_packet_reaches_ready_summary_with_fake_db(
    tmp_path: Path,
    monkeypatch,
) -> None:
    queue, fields, source_queue, source_fields = _write_packet(
        tmp_path,
        source_queue_rows=[_queue_row("ODQ-0001")],
        source_field_rows=_schedule_fields("ODQ-0001"),
        packet_queue_rows=[
            _queue_row("ODQ-0001", operator_status="ready_for_validation")
        ],
        packet_field_rows=_schedule_fields("ODQ-0001"),
    )
    monkeypatch.setattr(
        runner.db_prereqs,
        "run_check",
        lambda db_url, output_dir: _fake_db_prereq_pass(output_dir),
    )
    monkeypatch.setattr(
        runner.validation, "_build_db_checker", lambda **kwargs: FakeDbChecker()
    )
    monkeypatch.setattr(runner.ingest, "_connect", lambda db_url: FakeConnection())

    report = runner.run_intake(
        queue_path=queue,
        fields_path=fields,
        source_queue_path=source_queue,
        source_fields_path=source_fields,
        db_url="postgresql://user:secret@example.invalid/db",
        output_dir=tmp_path / "intake",
    )

    assert report["summary"]["status"] == "ready_for_controlled_apply"
    assert report["summary"]["exit_code"] == 0
    assert report["final_status_summary"]["apply_eligible_count"] == 1
    assert report["blockers"] == []


def test_missing_db_url_records_blocker_and_continues_to_status_summary(
    tmp_path: Path,
) -> None:
    queue, fields, source_queue, source_fields = _write_packet(
        tmp_path,
        source_queue_rows=[_queue_row("ODQ-0001")],
        source_field_rows=_schedule_fields("ODQ-0001"),
        packet_queue_rows=[
            _queue_row("ODQ-0001", operator_status="ready_for_validation")
        ],
        packet_field_rows=_schedule_fields("ODQ-0001"),
    )

    report = runner.run_intake(
        queue_path=queue,
        fields_path=fields,
        source_queue_path=source_queue,
        source_fields_path=source_fields,
        db_url="",
        output_dir=tmp_path / "intake",
    )

    blocker_codes = {blocker["code"] for blocker in report["blockers"]}
    assert report["summary"]["status"] == "blocked"
    assert "db_url_missing" in blocker_codes
    assert report["final_status_summary"]["manual_required_count"] == 0
    assert (tmp_path / "intake" / "status" / "p0_recovery_status_summary.json").exists()


def test_audit_failure_writes_downstream_summary_without_modifying_inputs(
    tmp_path: Path,
) -> None:
    source_queue_rows = [_queue_row("ODQ-0001", question="원본 질문")]
    packet_queue_rows = [
        _queue_row(
            "ODQ-0001", question="바뀐 질문", operator_status="ready_for_validation"
        )
    ]
    queue, fields, source_queue, source_fields = _write_packet(
        tmp_path,
        source_queue_rows=source_queue_rows,
        source_field_rows=_schedule_fields("ODQ-0001", question="원본 질문"),
        packet_queue_rows=packet_queue_rows,
        packet_field_rows=_schedule_fields("ODQ-0001", question="바뀐 질문"),
    )
    before = {
        _path: _hash(_path) for _path in (queue, fields, source_queue, source_fields)
    }

    report = runner.run_intake(
        queue_path=queue,
        fields_path=fields,
        source_queue_path=source_queue,
        source_fields_path=source_fields,
        db_url="",
        output_dir=tmp_path / "intake",
    )

    after = {
        _path: _hash(_path) for _path in (queue, fields, source_queue, source_fields)
    }
    assert before == after
    assert report["summary"]["status"] == "blocked"
    assert (tmp_path / "intake" / "status" / "p0_recovery_status_summary.json").exists()


def test_runner_never_calls_ingest_apply_path(tmp_path: Path, monkeypatch) -> None:
    queue, fields, source_queue, source_fields = _write_packet(
        tmp_path,
        source_queue_rows=[_queue_row("ODQ-0001")],
        source_field_rows=_schedule_fields("ODQ-0001"),
    )
    seen_apply_values: list[bool] = []
    original = runner.ingest.build_ingest_report

    def spy_build_ingest_report(*args, **kwargs):
        seen_apply_values.append(bool(kwargs.get("apply")))
        return original(*args, **kwargs)

    monkeypatch.setattr(runner.ingest, "build_ingest_report", spy_build_ingest_report)

    runner.run_intake(
        queue_path=queue,
        fields_path=fields,
        source_queue_path=source_queue,
        source_fields_path=source_fields,
        db_url="",
        output_dir=tmp_path / "intake",
    )

    assert seen_apply_values == [False]


def test_stage_exception_is_captured_as_blocker_and_handoff(
    tmp_path: Path, monkeypatch
) -> None:
    queue, fields, source_queue, source_fields = _write_packet(
        tmp_path,
        source_queue_rows=[_queue_row("ODQ-0001")],
        source_field_rows=_schedule_fields("ODQ-0001"),
    )

    def raise_audit(**kwargs):
        del kwargs
        raise RuntimeError("do not leak details")

    monkeypatch.setattr(runner.audit, "run_audit", raise_audit)

    report = runner.run_intake(
        queue_path=queue,
        fields_path=fields,
        source_queue_path=source_queue,
        source_fields_path=source_fields,
        db_url="",
        output_dir=tmp_path / "intake",
    )

    stage_codes = {stage["code"] for stage in report["stages"]}
    handoff = (tmp_path / "intake" / "intake_handoff.md").read_text(encoding="utf-8")
    assert "audit_stage_error" in stage_codes
    assert "audit_stage_error" in handoff
    assert "do not leak details" not in handoff


def test_missing_input_returns_exit_2_and_writes_intake_handoff(tmp_path: Path) -> None:
    exit_code = runner.main(
        [
            "--queue",
            str(tmp_path / "missing_queue.csv"),
            "--fields",
            str(tmp_path / "missing_fields.csv"),
            "--source-queue",
            str(tmp_path / "missing_source_queue.csv"),
            "--source-fields",
            str(tmp_path / "missing_source_fields.csv"),
            "--output-dir",
            str(tmp_path / "intake"),
        ]
    )

    assert exit_code == 2
    assert (tmp_path / "intake" / "intake_handoff.md").exists()
