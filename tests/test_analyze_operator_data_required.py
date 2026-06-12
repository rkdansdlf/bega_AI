from __future__ import annotations

import csv
import json

from scripts import analyze_operator_data_required as analyzer


def assert_final_verdict(question: str, verdict: str, tool: str = "") -> None:
    classification = analyzer.classify_question(question)
    final = analyzer.final_verdict_for_question(question, classification)

    assert final["final_verdict"] == verdict
    assert final["final_candidate_tool"] == tool


def test_classify_representative_operator_questions() -> None:
    cases = {
        "오늘 KBO 경기 일정 알려줘.": "schedule_window",
        "2026 KBO 개막일은 언제야?": "season_meta",
        "KBO 중계는 어디서 봐?": "broadcast_media",
        "야구장 티켓은 어디서 예매해?": "venue_ticket",
        "부상자 명단은 어디서 봐?": "roster_news",
        "팀별 선발 로테이션은 어떻게 돼?": "game_day_lineup",
        "우승 후보는 어디야?": "subjective_prediction",
        "2026 KBO 광고 규정은 바뀌었어?": "unsupported_external",
        "LG와 KT를 비교해줘?": "db_fast_path_candidate",
    }

    for question, expected_domain in cases.items():
        classification = analyzer.classify_question(question)
        assert classification.domain == expected_domain
        assert classification.contract_code
        assert classification.required_fields


def test_db_fast_path_candidate_does_not_require_operator_fields() -> None:
    classification = analyzer.classify_question("LG와 KT를 비교해줘?")

    assert classification.domain == "db_fast_path_candidate"
    assert classification.recommended_status == "db_fast_path_candidate"
    assert classification.candidate_tools == ("get_team_comparison",)
    assert analyzer.final_verdict_for_question(
        "LG와 KT를 비교해줘?", classification
    ) == {
        "final_verdict": "recovered_fast_path",
        "final_reason": "현재 DB 집계 도구로 답변 복구 대상으로 분류합니다.",
        "final_candidate_tool": "get_team_comparison",
    }
    assert classification.required_fields == (
        "candidate_tool",
        "required_params",
        "validation_query",
    )


def test_db_fast_path_recovery_tools_are_specific_to_router_capability() -> None:
    cases = {
        "LG와 KT를 비교해줘?": "get_team_comparison",
        "2026년 4월 28일부터 4월 30일까지 KBO 경기표 보여줘.": "get_schedule",
        "2026년 4월 30일 KBO 경기 결과는 뭐야?": "get_games_by_date",
        "2026년 4월 30일 기준 KBO 순위는 어떻게 돼?": "get_team_standings",
        "팀별 승패는 어떻게 돼?": "get_team_standings",
    }

    for question, expected_tool in cases.items():
        assert_final_verdict(question, "recovered_fast_path", expected_tool)


def test_relative_and_live_schedule_questions_remain_operator_data_required() -> None:
    for question in (
        "오늘 KBO 경기 일정 알려줘.",
        "내일 KBO 경기 일정 알려줘.",
        "이번 주 KBO 경기표 보여줘.",
    ):
        classification = analyzer.classify_question(question)

        assert classification.domain == "schedule_window"
        assert_final_verdict(question, "operator_data_required")


def test_db_fast_path_candidate_without_basis_keeps_operator_data_verdict() -> None:
    for question in (
        "실책이 적은 선수는 누구야?",
        "도루가 많은 시즌이야?",
        "홈런이 많은 시즌이야?",
    ):
        classification = analyzer.classify_question(question)

        assert classification.domain == "db_fast_path_candidate"
        assert analyzer.final_verdict_for_question(
            question, classification
        )["final_verdict"] == "operator_data_required"


def test_build_taxonomy_dedupes_operator_questions_and_preserves_status_counts() -> None:
    report = {
        "results": [
            {
                "endpoint": "/ai/chat/completion",
                "question": "오늘 KBO 경기 일정 알려줘.",
                "answer": "MANUAL_BASEBALL_DATA_REQUIRED: ...",
                "answerability": {"status": "operator_data_required"},
            },
            {
                "endpoint": "/ai/chat/stream",
                "question": "오늘 KBO 경기 일정 알려줘.",
                "answer": "MANUAL_BASEBALL_DATA_REQUIRED: ...",
                "answerability": {"status": "operator_data_required"},
            },
            {
                "endpoint": "/ai/chat/completion",
                "question": "우리 팀 순위 몇 위야?",
                "answer": "팀명을 같이 알려주세요.",
                "answerability": {"status": "clarification_required"},
            },
            {
                "endpoint": "/ai/chat/completion",
                "question": "KBO 리그는 어떤 리그야?",
                "answer": "KBO 리그는 한국의 최상위 프로야구 리그입니다.",
                "answerability": {"status": "answerable"},
            },
        ]
    }

    taxonomy = analyzer.build_taxonomy(report)

    assert taxonomy["summary"]["total_results"] == 4
    assert taxonomy["summary"]["unique_questions_by_status"] == {
        "answerable": 1,
        "clarification_required": 1,
        "operator_data_required": 1,
    }
    assert taxonomy["summary"]["target_unique_questions"] == 1
    assert taxonomy["summary"]["domain_counts"]["schedule_window"] == 1
    assert taxonomy["summary"]["final_verdict_counts"]["operator_data_required"] == 1
    assert taxonomy["records"][0]["endpoint_count"] == 2
    assert taxonomy["records"][0]["endpoints"] == [
        "/ai/chat/completion",
        "/ai/chat/stream",
    ]


def test_build_taxonomy_counts_recovered_fast_paths_separately_from_manual_cases() -> None:
    report = {
        "results": [
            {
                "endpoint": "/ai/chat/completion",
                "question": "LG와 KT를 비교해줘?",
                "answer": "MANUAL_BASEBALL_DATA_REQUIRED: ...",
                "answerability": {"status": "operator_data_required"},
            },
            {
                "endpoint": "/ai/chat/stream",
                "question": "LG와 KT를 비교해줘?",
                "answer": "MANUAL_BASEBALL_DATA_REQUIRED: ...",
                "answerability": {"status": "operator_data_required"},
            },
            {
                "endpoint": "/ai/chat/completion",
                "question": "2026년 4월 30일 기준 KBO 순위는 어떻게 돼?",
                "answer": "MANUAL_BASEBALL_DATA_REQUIRED: ...",
                "answerability": {"status": "operator_data_required"},
            },
            {
                "endpoint": "/ai/chat/completion",
                "question": "오늘 KBO 경기 일정 알려줘.",
                "answer": "MANUAL_BASEBALL_DATA_REQUIRED: ...",
                "answerability": {"status": "operator_data_required"},
            },
            {
                "endpoint": "/ai/chat/completion",
                "question": "부상자 명단은 어디서 봐?",
                "answer": "MANUAL_BASEBALL_DATA_REQUIRED: ...",
                "answerability": {"status": "operator_data_required"},
            },
        ]
    }

    taxonomy = analyzer.build_taxonomy(report)
    records_by_question = {record["question"]: record for record in taxonomy["records"]}

    assert taxonomy["summary"]["target_unique_questions"] == 4
    assert taxonomy["summary"]["final_verdict_counts"] == {
        "operator_data_required": 2,
        "recovered_fast_path": 2,
    }
    assert records_by_question["LG와 KT를 비교해줘?"]["endpoint_count"] == 2
    assert (
        records_by_question["LG와 KT를 비교해줘?"]["final_candidate_tool"]
        == "get_team_comparison"
    )
    assert (
        records_by_question["2026년 4월 30일 기준 KBO 순위는 어떻게 돼?"][
            "final_candidate_tool"
        ]
        == "get_team_standings"
    )
    assert (
        records_by_question["오늘 KBO 경기 일정 알려줘."]["final_verdict"]
        == "operator_data_required"
    )


def test_report_writers_create_taxonomy_and_contract_template(tmp_path) -> None:
    report = {
        "results": [
            {
                "endpoint": "/ai/chat/completion",
                "question": "KBO 중계는 어디서 봐?",
                "answer": "MANUAL_BASEBALL_DATA_REQUIRED: ...",
                "answerability": {"status": "operator_data_required"},
            }
        ]
    }
    taxonomy = analyzer.build_taxonomy(report)
    taxonomy["contract_template"] = analyzer.build_contract_template_rows()

    json_path = tmp_path / "taxonomy.json"
    csv_path = tmp_path / "taxonomy.csv"
    template_path = tmp_path / "template.csv"

    analyzer.write_json(json_path, taxonomy)
    analyzer.write_records_csv(csv_path, taxonomy["records"])
    analyzer.write_template_csv(template_path, taxonomy["contract_template"])

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["target_unique_questions"] == 1

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["domain"] == "broadcast_media"
    assert "channel_name" in rows[0]["required_fields"]

    with template_path.open("r", encoding="utf-8-sig", newline="") as handle:
        template_rows = list(csv.DictReader(handle))
    domains = {row["domain"] for row in template_rows}
    assert set(analyzer.DOMAINS).issubset(domains)
