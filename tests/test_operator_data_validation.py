from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts import analyze_operator_data_required as analyzer
from scripts import validate_operator_data_handoff as validator


class FakeDbChecker:
    def __init__(
        self,
        *,
        games: dict[str, Mapping[str, Any]] | None = None,
        player_counts: dict[str, int] | None = None,
        known_teams: set[str] | None = None,
    ) -> None:
        self.games = games or {}
        self.player_counts = player_counts or {}
        self.known_teams = known_teams or {"LG", "KT", "KIA", "SSG", "DB", "NC", "LT", "SS", "HH", "KH"}
        self.db_checks_skipped = False
        self.db_skip_reason = ""

    def get_game(self, game_id: str) -> Mapping[str, Any] | None:
        return self.games.get(game_id)

    def count_players(self, player_name: str) -> int:
        return self.player_counts.get(player_name, 1)

    def is_known_team_code(self, team_code: str, season_year: int | None = None) -> bool:
        del season_year
        return team_code in self.known_teams

    def close(self) -> None:
        return None


def _queue(
    queue_id: str,
    domain: str,
    *,
    status: str = "ready_for_validation",
    required_fields: list[str] | None = None,
) -> dict[str, str]:
    contract = analyzer.CONTRACTS[domain]
    fields = required_fields or list(contract.required_fields)
    return {
        "queue_id": queue_id,
        "priority": "P0",
        "priority_reason": "test",
        "domain": domain,
        "contract_code": contract.contract_code,
        "question": f"{domain} question",
        "required_fields": "|".join(fields),
        "endpoint_count": "2",
        "endpoints": "/ai/chat/completion|/ai/chat/stream",
        "sample_answer": "MANUAL_BASEBALL_DATA_REQUIRED: ...",
        "operator_status": status,
        "operator_owner": "",
        "operator_notes": "",
    }


def _fields(
    queue_id: str,
    domain: str,
    values: Mapping[str, Any],
    *,
    required_fields: list[str] | None = None,
) -> list[dict[str, str]]:
    fields = required_fields or list(analyzer.CONTRACTS[domain].required_fields)
    return [
        {
            "queue_id": queue_id,
            "domain": domain,
            "contract_code": analyzer.CONTRACTS[domain].contract_code,
            "question": f"{domain} question",
            "field_name": field_name,
            "field_description": "",
            "required": "true",
            "operator_value": str(values.get(field_name, "")),
            "operator_notes": "",
        }
        for field_name in fields
    ]


def _report(
    queue_rows: list[dict[str, str]],
    field_rows: list[dict[str, str]],
    *,
    db_checker: Any | None = None,
) -> dict[str, Any]:
    return validator.build_validation_report(
        queue_rows,
        field_rows,
        db_checker=db_checker or FakeDbChecker(),
    )


def test_pending_handoff_skips_value_validation() -> None:
    queue = [_queue("ODQ-0001", "schedule_window", status="pending")]
    fields = _fields("ODQ-0001", "schedule_window", {})

    report = _report(queue, fields)

    assert report["summary"]["status"] == "pass"
    assert report["summary"]["issue_counts"] == {"error": 0, "warning": 0}
    row = report["normalized_rows"][0]
    assert row["validation_status"] == "pass"
    assert row["apply_eligible"] is False
    assert row["skip_reason"] == "operator_status_pending"


def test_ready_for_validation_requires_values_verification_and_confidence() -> None:
    queue = [_queue("ODQ-0002", "schedule_window")]
    fields = _fields(
        "ODQ-0002",
        "schedule_window",
        {
            "game_date": "2026-04-30",
            "game_id": "20260430LGKT0",
            "home_team": "LG",
            "away_team": "KT",
            "stadium_name": "잠실",
            "start_time": "18:30",
            "game_status": "scheduled",
            "source_name": "",
            "source_checked_at": "2026-06-03",
            "is_verified": "false",
            "confidence": "0.20",
        },
    )

    report = _report(queue, fields)
    codes = {issue["code"] for issue in report["issues"]}

    assert report["summary"]["status"] == "fail"
    assert {"missing_source_name", "not_verified", "low_confidence"}.issubset(codes)
    assert report["normalized_rows"][0]["apply_eligible"] is False


def test_file_shape_rejects_duplicate_orphan_and_bad_status() -> None:
    queue = [
        _queue("ODQ-0003", "schedule_window", status="bad_status"),
        _queue("ODQ-0003", "schedule_window", status="pending"),
    ]
    fields = [
        *_fields("ODQ-0003", "schedule_window", {}),
        *_fields("ODQ-9999", "schedule_window", {}),
    ]

    report = _report(queue, fields)
    codes = {issue["code"] for issue in report["issues"]}

    assert {"invalid_operator_status", "duplicate_queue_id", "orphan_field_row"}.issubset(codes)
    assert report["summary"]["status"] == "fail"


def test_file_shape_rejects_duplicate_missing_unexpected_and_mismatched_fields() -> None:
    queue = [
        _queue(
            "ODQ-0031",
            "schedule_window",
            status="pending",
            required_fields=["game_date", "source_name"],
        )
    ]
    fields = _fields(
        "ODQ-0031",
        "schedule_window",
        {},
        required_fields=["game_date", "unexpected_field"],
    )
    fields.append(dict(fields[0]))
    fields[0]["domain"] = "venue_ticket"
    fields[1]["contract_code"] = "WRONG_CONTRACT"
    fields[1]["question"] = "mismatched question"

    report = _report(queue, fields)
    codes = {issue["code"] for issue in report["issues"]}

    assert {
        "duplicate_field_row",
        "field_domain_mismatch",
        "field_contract_code_mismatch",
        "field_question_mismatch",
        "missing_field_row",
        "unexpected_field_row",
    }.issubset(codes)
    assert report["summary"]["status"] == "fail"


def test_domain_type_validation_reports_invalid_values() -> None:
    queue = [_queue("ODQ-0004", "venue_ticket")]
    fields = _fields(
        "ODQ-0004",
        "venue_ticket",
        {
            "stadium_name": "잠실",
            "topic_type": "ticket",
            "title": "예매",
            "body": "예매 정보",
            "valid_from": "2026-06-10",
            "valid_to": "2026-06-01",
            "source_name": "operator",
            "source_checked_at": "2026-06-03",
            "is_verified": "true",
            "confidence": "0.90",
        },
    )

    report = _report(queue, fields)
    codes = {issue["code"] for issue in report["issues"]}

    assert "invalid_validity_window" in codes
    assert report["summary"]["status"] == "fail"


def test_non_p0_domains_pass_validation_but_are_not_apply_eligible() -> None:
    queue = [
        _queue("ODQ-0040", "venue_ticket"),
        _queue("ODQ-0041", "unsupported_external"),
        _queue("ODQ-0042", "subjective_prediction"),
    ]
    fields = [
        *_fields(
            "ODQ-0040",
            "venue_ticket",
            {
                "stadium_name": "잠실",
                "topic_type": "ticket",
                "title": "예매",
                "body": "예매 정보",
                "valid_from": "2026-06-01",
                "valid_to": "2026-06-30",
                "source_name": "operator",
                "source_checked_at": "2026-06-03",
                "is_verified": "true",
                "confidence": "0.90",
            },
        ),
        *_fields(
            "ODQ-0041",
            "unsupported_external",
            {
                "requested_topic": "외부 광고 규정",
                "supported_by_operator": "true",
                "manual_answer_text": "운영자 승인 답변",
                "source_name": "operator",
                "source_checked_at": "2026-06-03",
                "is_verified": "true",
                "confidence": "0.90",
            },
        ),
        *_fields(
            "ODQ-0042",
            "subjective_prediction",
            {
                "as_of_date": "2026-06-03",
                "question_scope": "우승 후보",
                "selection_criteria": "운영자 기준",
                "ranked_entities": "LG, KT",
                "operator_basis": "운영자 제공 근거",
                "source_name": "operator",
                "source_checked_at": "2026-06-03",
                "is_verified": "true",
                "confidence": "0.90",
            },
        ),
    ]

    report = _report(queue, fields)
    rows = {row["queue_id"]: row for row in report["normalized_rows"]}

    assert report["summary"]["status"] == "pass"
    assert rows["ODQ-0040"]["apply_eligible"] is False
    assert rows["ODQ-0040"]["skip_reason"] == "operator_data_v1_non_p0_domain"
    assert rows["ODQ-0041"]["apply_eligible"] is False
    assert rows["ODQ-0041"]["skip_reason"] == "operator_data_v1_non_p0_domain"
    assert rows["ODQ-0042"]["apply_eligible"] is False
    assert rows["ODQ-0042"]["skip_reason"] == "policy_gate_subjective_prediction"


def test_unsupported_external_false_allows_blank_manual_answer_text() -> None:
    queue = [_queue("ODQ-0005", "unsupported_external")]
    fields = _fields(
        "ODQ-0005",
        "unsupported_external",
        {
            "requested_topic": "외부 광고 규정",
            "supported_by_operator": "false",
            "manual_answer_text": "",
            "source_name": "operator",
            "source_checked_at": "2026-06-03",
            "is_verified": "true",
            "confidence": "0.90",
        },
    )

    report = _report(queue, fields)
    row = report["normalized_rows"][0]

    assert report["summary"]["status"] == "pass"
    assert row["validation_status"] == "pass"
    assert row["apply_eligible"] is False
    assert row["skip_reason"] == "unsupported_external_not_supported"


def test_subjective_prediction_normalizes_but_is_not_apply_eligible() -> None:
    queue = [_queue("ODQ-0006", "subjective_prediction")]
    fields = _fields(
        "ODQ-0006",
        "subjective_prediction",
        {
            "as_of_date": "2026-06-03",
            "question_scope": "우승 후보",
            "selection_criteria": "운영자 기준",
            "ranked_entities": "LG, KT",
            "operator_basis": "운영자 제공 근거",
            "source_name": "operator",
            "source_checked_at": "2026-06-03",
            "is_verified": "true",
            "confidence": "0.90",
        },
    )

    report = _report(queue, fields)
    row = report["normalized_rows"][0]

    assert report["summary"]["status"] == "pass"
    assert row["apply_eligible"] is False
    assert row["skip_reason"] == "policy_gate_subjective_prediction"


def test_db_checker_validates_game_player_and_team_success() -> None:
    queue = [_queue("ODQ-0007", "game_day_lineup")]
    fields = _fields(
        "ODQ-0007",
        "game_day_lineup",
        {
            "game_id": "20260430LGKT0",
            "team_code": "LG",
            "player_name": "홍길동",
            "batting_order": "1",
            "position": "CF",
            "announced_at": "2026-04-30T16:00:00",
            "source_name": "operator",
            "source_checked_at": "2026-06-03",
            "is_verified": "true",
            "confidence": "0.95",
        },
    )
    db_checker = FakeDbChecker(
        games={"20260430LGKT0": {"game_id": "20260430LGKT0", "home_team": "LG", "away_team": "KT"}},
        player_counts={"홍길동": 1},
        known_teams={"LG"},
    )

    report = _report(queue, fields, db_checker=db_checker)
    row = report["normalized_rows"][0]

    assert report["summary"]["status"] == "pass"
    assert row["apply_eligible"] is True
    assert row["apply_target"] == "game_lineups/manual_starters"


def test_db_checker_reports_missing_game_ambiguous_player_and_unknown_team() -> None:
    queue = [_queue("ODQ-0008", "game_day_lineup")]
    fields = _fields(
        "ODQ-0008",
        "game_day_lineup",
        {
            "game_id": "MISSING",
            "team_code": "ZZZ",
            "player_name": "동명이인",
            "batting_order": "1",
            "position": "CF",
            "announced_at": "2026-04-30T16:00:00",
            "source_name": "operator",
            "source_checked_at": "2026-06-03",
            "is_verified": "true",
            "confidence": "0.95",
        },
    )
    db_checker = FakeDbChecker(games={}, player_counts={"동명이인": 2}, known_teams={"LG"})

    report = _report(queue, fields, db_checker=db_checker)
    codes = {issue["code"] for issue in report["issues"]}

    assert {"game_not_found", "player_resolution_not_unique", "unknown_team_code"}.issubset(codes)
    assert report["summary"]["status"] == "fail"


def test_db_checks_skipped_create_warning_summary() -> None:
    queue = [_queue("ODQ-0009", "schedule_window", status="pending")]
    fields = _fields("ODQ-0009", "schedule_window", {})
    db_checker = validator.NoopDbChecker("POSTGRES_DB_URL is not set")

    report = _report(queue, fields, db_checker=db_checker)

    assert report["summary"]["status"] == "warning"
    assert report["summary"]["db_checks"] == {
        "skipped": True,
        "skip_reason": "POSTGRES_DB_URL is not set",
    }
    assert any(issue["code"] == "db_checks_skipped" for issue in report["issues"])
    assert report["normalized_rows"][0]["apply_eligible"] is False


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
    assert {
        row["skip_reason"] for row in report["normalized_rows"]
    } == {"operator_status_pending"}
