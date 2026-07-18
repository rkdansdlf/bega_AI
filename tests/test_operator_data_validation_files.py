from __future__ import annotations

from pathlib import Path

import pytest

from scripts import validate_operator_data_handoff as validator
from tests.operator_data_validation_support import FakeDbChecker, _fields, _queue

def test_validate_files_writes_expected_artifacts(tmp_path: Path) -> None:
    queue_rows = [_queue("ODQ-0010", "schedule_window", status="pending")]
    field_rows = _fields("ODQ-0010", "schedule_window", {})
    queue_path = tmp_path / "queue.csv"
    fields_path = tmp_path / "fields.csv"
    output_dir = tmp_path / "out"

    validator._write_csv(queue_path, queue_rows, list(queue_rows[0].keys()))
    validator._write_csv(fields_path, field_rows, list(field_rows[0].keys()))
    report = validator.validate_files(
        queue_path=queue_path,
        fields_path=fields_path,
        output_dir=output_dir,
        db_checker=FakeDbChecker(),
    )

    assert report["summary"]["total_queue_items"] == 1
    assert (output_dir / "operator_data_validation_summary.json").exists()
    assert (output_dir / "operator_data_validation_issues.csv").exists()
    assert (output_dir / "operator_data_normalized_rows.jsonl").exists()
    assert (output_dir / "operator_data_apply_plan.csv").exists()


def test_validate_files_rejects_noncanonical_field_headers(tmp_path: Path) -> None:
    queue_rows = [_queue("ODQ-0011", "schedule_window", status="pending")]
    field_rows = _fields("ODQ-0011", "schedule_window", {})
    queue_path = tmp_path / "queue.csv"
    fields_path = tmp_path / "fields.csv"
    output_dir = tmp_path / "out"

    validator._write_csv(queue_path, queue_rows, list(queue_rows[0].keys()))
    validator._write_csv(
        fields_path,
        field_rows,
        [*field_rows[0].keys(), "source_name"],
    )
    report = validator.validate_files(
        queue_path=queue_path,
        fields_path=fields_path,
        output_dir=output_dir,
        db_checker=FakeDbChecker(),
    )

    codes = {issue["code"] for issue in report["issues"]}
    assert "invalid_fields_header" in codes
    assert report["summary"]["status"] == "fail"


def test_real_194_bundle_pending_baseline_is_read_only_and_not_apply_eligible(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).parent.parent
    queue_path = (
        project_root
        / "reports"
        / "operator_data_handoff_queue_post_db_fast_path_docker_kbo500.csv"
    )
    fields_path = (
        project_root
        / "reports"
        / "operator_data_handoff_fields_post_db_fast_path_docker_kbo500.csv"
    )
    if not queue_path.exists() or not fields_path.exists():
        pytest.skip("local operator-data handoff reports are not present")

    report = validator.validate_files(
        queue_path=queue_path,
        fields_path=fields_path,
        output_dir=tmp_path / "validation",
        db_checker=validator.NoopDbChecker("--skip-db-checks was set"),
    )

    assert report["summary"]["total_queue_items"] == 194
    assert report["summary"]["total_field_rows"] == 1857
    assert report["summary"]["normalized_row_count"] == 194
    assert report["summary"]["apply_plan_row_count"] == 194
    assert report["summary"]["issue_counts"] == {"error": 0, "warning": 1}
    assert report["summary"]["operator_status_counts"] == {"pending": 194}
    assert report["summary"]["apply_eligible_count"] == 0
    assert len(report["normalized_rows"]) == 194
    assert len(report["apply_plan_rows"]) == 194
    assert {row["skip_reason"] for row in report["normalized_rows"]} == {
        "operator_status_pending"
    }
