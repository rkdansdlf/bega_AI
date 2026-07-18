from __future__ import annotations

import pytest

from scripts import validate_operator_data_handoff as validator
from tests.operator_data_validation_support import FakeDbChecker, _fields, _queue, _report

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

    assert {
        "invalid_operator_status",
        "duplicate_queue_id",
        "orphan_field_row",
    }.issubset(codes)
    assert report["summary"]["status"] == "fail"


def test_file_shape_rejects_duplicate_missing_unexpected_and_mismatched_fields() -> (
    None
):
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
        games={
            "20260430LGKT0": {
                "game_id": "20260430LGKT0",
                "home_team": "LG",
                "away_team": "KT",
            }
        },
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
    db_checker = FakeDbChecker(
        games={}, player_counts={"동명이인": 2}, known_teams={"LG"}
    )

    report = _report(queue, fields, db_checker=db_checker)
    codes = {issue["code"] for issue in report["issues"]}

    assert {
        "game_not_found",
        "player_resolution_not_unique",
        "unknown_team_code",
    }.issubset(codes)
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
