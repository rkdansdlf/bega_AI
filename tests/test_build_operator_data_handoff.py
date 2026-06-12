from __future__ import annotations

import csv
import json

from scripts import build_operator_data_handoff as handoff


def _record(
    question: str,
    domain: str,
    *,
    verdict: str = "operator_data_required",
    required_fields: list[str] | None = None,
) -> dict[str, object]:
    contract = handoff.taxonomy_analyzer.CONTRACTS[domain]
    return {
        "question": question,
        "domain": domain,
        "contract_code": contract.contract_code,
        "required_fields": list(required_fields or contract.required_fields),
        "final_verdict": verdict,
        "endpoint_count": 2,
        "endpoints": ["/ai/chat/completion", "/ai/chat/stream"],
        "sample_answer": "MANUAL_BASEBALL_DATA_REQUIRED: ...",
    }


def test_recovered_fast_paths_are_excluded_from_handoff_queue() -> None:
    taxonomy = {
        "records": [
            _record(
                "LG와 KT를 비교해줘?",
                "db_fast_path_candidate",
                verdict="recovered_fast_path",
            ),
            _record("오늘 KBO 경기 일정 알려줘.", "schedule_window"),
        ]
    }

    result = handoff.build_handoff(taxonomy, source_taxonomy_path="taxonomy.json")

    assert result["summary"]["total_queue_items"] == 1
    assert result["queue_rows"][0]["question"] == "오늘 KBO 경기 일정 알려줘."
    assert result["queue_rows"][0]["priority"] == "P0"


def test_default_paths_target_post_db_fast_path_bundle() -> None:
    assert str(handoff.DEFAULT_INPUT).endswith(
        "reports/operator_data_required_taxonomy_post_db_fast_path_docker_kbo500.json"
    )
    assert str(handoff.DEFAULT_QUEUE_OUTPUT).endswith(
        "reports/operator_data_handoff_queue_post_db_fast_path_docker_kbo500.csv"
    )
    assert str(handoff.DEFAULT_FIELDS_OUTPUT).endswith(
        "reports/operator_data_handoff_fields_post_db_fast_path_docker_kbo500.csv"
    )
    assert str(handoff.DEFAULT_SUMMARY_OUTPUT).endswith(
        "reports/operator_data_handoff_queue_post_db_fast_path_docker_kbo500_summary.json"
    )
    assert handoff._summary_path(handoff.DEFAULT_INPUT) == (
        "reports/operator_data_required_taxonomy_post_db_fast_path_docker_kbo500.json"
    )


def test_manual_records_sort_by_priority_domain_then_question() -> None:
    taxonomy = {
        "records": [
            _record("우승 후보는 어디야?", "subjective_prediction"),
            _record("KBO 중계는 어디서 봐?", "broadcast_media"),
            _record("2026 KBO 판정 보완은 어떻게 돼?", "unsupported_external"),
            _record("내일 KBO 경기 일정 알려줘.", "schedule_window"),
            _record("2026 KBO 개막일은 언제야?", "season_meta"),
        ]
    }

    result = handoff.build_handoff(taxonomy, source_taxonomy_path="taxonomy.json")

    assert [
        (row["queue_id"], row["priority"], row["domain"], row["question"])
        for row in result["queue_rows"]
    ] == [
        ("ODQ-0001", "P0", "season_meta", "2026 KBO 개막일은 언제야?"),
        ("ODQ-0002", "P0", "schedule_window", "내일 KBO 경기 일정 알려줘."),
        ("ODQ-0003", "P1", "broadcast_media", "KBO 중계는 어디서 봐?"),
        ("ODQ-0004", "P2", "unsupported_external", "2026 KBO 판정 보완은 어떻게 돼?"),
        ("ODQ-0005", "P3", "subjective_prediction", "우승 후보는 어디야?"),
    ]


def test_field_rows_expand_required_fields_with_blank_operator_values() -> None:
    taxonomy = {
        "records": [
            _record(
                "오늘 KBO 경기 일정 알려줘.",
                "schedule_window",
                required_fields=["game_date", "source_name"],
            )
        ]
    }

    result = handoff.build_handoff(taxonomy, source_taxonomy_path="taxonomy.json")

    assert result["field_rows"] == [
        {
            "queue_id": "ODQ-0001",
            "domain": "schedule_window",
            "contract_code": "SCHEDULE_WINDOW_REQUIRED",
            "question": "오늘 KBO 경기 일정 알려줘.",
            "field_name": "game_date",
            "field_description": "경기 날짜",
            "required": "true",
            "operator_value": "",
            "operator_notes": "",
        },
        {
            "queue_id": "ODQ-0001",
            "domain": "schedule_window",
            "contract_code": "SCHEDULE_WINDOW_REQUIRED",
            "question": "오늘 KBO 경기 일정 알려줘.",
            "field_name": "source_name",
            "field_description": "운영자가 확인한 내부/공식 원천 이름",
            "required": "true",
            "operator_value": "",
            "operator_notes": "",
        },
    ]


def test_summary_counts_match_queue_rows() -> None:
    taxonomy = {
        "records": [
            _record("2026 KBO 개막일은 언제야?", "season_meta"),
            _record("오늘 KBO 경기 일정 알려줘.", "schedule_window"),
            _record("KBO 중계는 어디서 봐?", "broadcast_media"),
            _record("2026 KBO 판정 보완은 어떻게 돼?", "unsupported_external"),
            _record("실책이 적은 선수는 누구야?", "db_fast_path_candidate"),
            _record("우승 후보는 어디야?", "subjective_prediction"),
        ]
    }

    result = handoff.build_handoff(taxonomy, source_taxonomy_path="taxonomy.json")
    summary = result["summary"]

    assert summary["total_queue_items"] == len(result["queue_rows"]) == 6
    assert summary["priority_counts"] == {
        "P0": 2,
        "P1": 1,
        "P2": 2,
        "P3": 1,
    }
    assert summary["domain_counts"]["schedule_window"] == 1
    assert summary["source_taxonomy_path"] == "taxonomy.json"


def test_report_writers_create_queue_fields_and_summary_files(tmp_path) -> None:
    taxonomy = {
        "records": [
            _record(
                "오늘 KBO 경기 일정 알려줘.",
                "schedule_window",
                required_fields=["game_date"],
            )
        ]
    }
    result = handoff.build_handoff(taxonomy, source_taxonomy_path="taxonomy.json")
    queue_path = tmp_path / "queue.csv"
    fields_path = tmp_path / "fields.csv"
    summary_path = tmp_path / "summary.json"

    handoff.write_csv(queue_path, result["queue_rows"], handoff.QUEUE_FIELDNAMES)
    handoff.write_csv(fields_path, result["field_rows"], handoff.FIELDS_FIELDNAMES)
    handoff.write_json(summary_path, result["summary"])

    with queue_path.open("r", encoding="utf-8-sig", newline="") as handle:
        queue_rows = list(csv.DictReader(handle))
    with fields_path.open("r", encoding="utf-8-sig", newline="") as handle:
        field_rows = list(csv.DictReader(handle))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert queue_rows[0]["queue_id"] == "ODQ-0001"
    assert queue_rows[0]["operator_status"] == "pending"
    assert field_rows[0]["field_name"] == "game_date"
    assert field_rows[0]["operator_value"] == ""
    assert list(field_rows[0]) == handoff.FIELDS_FIELDNAMES
    assert "source_name" not in field_rows[0]
    assert summary["total_queue_items"] == 1
