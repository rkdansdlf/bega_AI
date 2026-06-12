from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts import build_operator_data_p0_smoke_set as builder
from scripts import verify_operator_data_p0_smoke as verifier


def _row(
    queue_id: str,
    domain: str = "schedule_window",
    *,
    question: str = "오늘 KBO 경기 일정 알려줘.",
    apply_eligible: bool = True,
    validation_status: str = "pass",
    operator_status: str = "ready_for_validation",
    skip_reason: str = "",
) -> dict[str, Any]:
    return {
        "queue_id": queue_id,
        "domain": domain,
        "question": question,
        "operator_status": operator_status,
        "validation_status": validation_status,
        "apply_eligible": apply_eligible,
        "skip_reason": skip_reason,
    }


def _expectation(kind: str = "recovered") -> dict[str, Any]:
    base = {
        "queue_id": "ODQ-0001",
        "domain": "schedule_window",
        "question": "오늘 KBO 경기 일정 알려줘.",
        "expectation": kind,
    }
    if kind == "recovered":
        base.update(
            {
                "expected_strategy": "operator_data_fast_path",
                "expected_source_tier": "operator_data",
                "expected_operator_data_domain": "schedule_window",
            }
        )
    else:
        base.update(
            {
                "expected_status": "operator_data_required_or_expected_non_answer",
                "manual_contract_allowed": True,
            }
        )
    return base


def _smoke_item(
    *,
    endpoint: str,
    answer: str = "운영자 제공 데이터 기준, 확인된 항목만 정리합니다.",
    strategy: str = "operator_data_fast_path",
    source_tier: str = "operator_data",
    domain: str = "schedule_window",
    ok: bool = True,
    answerability_status: str = "answerable",
    expected_non_answer: bool = False,
) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "question": "오늘 KBO 경기 일정 알려줘.",
        "ok": ok,
        "answer": answer,
        "answerability": {
            "status": answerability_status,
            "expected_non_answer": expected_non_answer,
            "answerability_pass": True,
            "failure_markers": [],
        },
        "sample_response": {
            "strategy": strategy,
            "source_tier": source_tier,
            "operator_data_domain": domain,
        },
    }


def _report(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"results": items}


def test_pending_only_bundle_creates_manual_controls_and_recovered_warning() -> None:
    result = builder.build_smoke_set(
        [
            _row(
                "ODQ-0001",
                apply_eligible=False,
                operator_status="pending",
                skip_reason="operator_status_pending",
            )
        ]
    )

    assert result["summary"]["recovered_count"] == 0
    assert result["summary"]["manual_control_count"] == 1
    assert result["summary"]["warnings"] == ["no_apply_eligible_p0_rows_for_recovered_smoke"]
    assert result["recovered_questions"] == []
    assert result["manual_control_questions"] == ["오늘 KBO 경기 일정 알려줘."]


def test_apply_eligible_rows_create_recovered_expectations_for_p0_only() -> None:
    result = builder.build_smoke_set(
        [
            _row("ODQ-0001", "schedule_window"),
            _row("ODQ-0002", "venue_ticket", question="야구장 티켓은 어디서 예매해?"),
        ]
    )

    assert result["summary"]["recovered_count"] == 1
    assert result["expectations"][0]["expectation"] == "recovered"
    assert result["expectations"][0]["expected_operator_data_domain"] == "schedule_window"


def test_validation_fail_row_goes_to_manual_controls() -> None:
    result = builder.build_smoke_set(
        [
            _row(
                "ODQ-0001",
                apply_eligible=False,
                validation_status="fail",
                skip_reason="validation_not_passed",
            )
        ]
    )

    assert result["expectations"][0]["expectation"] == "manual_control"
    assert result["expectations"][0]["skip_reason"] == "validation_not_passed"


def test_build_files_writes_combined_questions_in_expectation_order(tmp_path: Path) -> None:
    normalized_path = tmp_path / "normalized.jsonl"
    rows = [
        _row("ODQ-0001", question="복구된 P0 질문"),
        _row(
            "ODQ-0002",
            question="수동 계약 유지 질문",
            apply_eligible=False,
            operator_status="pending",
            skip_reason="operator_status_pending",
        ),
    ]
    normalized_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    recovered_output = tmp_path / "recovered.txt"
    manual_output = tmp_path / "manual.txt"
    all_output = tmp_path / "all.txt"

    payload = builder.build_files(
        normalized_path=normalized_path,
        output_dir=tmp_path / "smoke",
        recovered_output=recovered_output,
        manual_output=manual_output,
        all_output=all_output,
    )

    assert recovered_output.read_text(encoding="utf-8").splitlines() == ["복구된 P0 질문"]
    assert manual_output.read_text(encoding="utf-8").splitlines() == ["수동 계약 유지 질문"]
    assert all_output.read_text(encoding="utf-8").splitlines() == [
        "복구된 P0 질문",
        "수동 계약 유지 질문",
    ]
    assert payload["summary"]["all_question_file"] == str(all_output)


def test_recovered_report_with_operator_fast_path_passes() -> None:
    payload = {
        "expectations": [_expectation("recovered")],
    }
    report = _report(
        [
            _smoke_item(endpoint="/ai/chat/completion"),
            _smoke_item(endpoint="/ai/chat/stream"),
        ]
    )

    result = verifier.verify_report(smoke_report=report, expectations_payload=payload)

    assert result["summary"]["status"] == "pass"
    assert result["failures"] == []


def test_recovered_report_with_manual_contract_fails() -> None:
    payload = {
        "expectations": [_expectation("recovered")],
    }
    report = _report(
        [
            _smoke_item(
                endpoint="/ai/chat/completion",
                answer="MANUAL_BASEBALL_DATA_REQUIRED: 운영자 데이터가 필요합니다.",
                strategy="manual_baseball_data_required",
                source_tier="none",
                domain="",
            ),
            _smoke_item(endpoint="/ai/chat/stream"),
        ]
    )

    result = verifier.verify_report(smoke_report=report, expectations_payload=payload)
    codes = {failure["code"] for failure in result["failures"]}

    assert result["summary"]["status"] == "fail"
    assert {"wrong_strategy", "manual_contract_returned"}.issubset(codes)


def test_manual_control_report_with_manual_contract_passes() -> None:
    payload = {
        "expectations": [_expectation("manual_control")],
    }
    report = _report(
        [
            _smoke_item(
                endpoint="/ai/chat/completion",
                answer="MANUAL_BASEBALL_DATA_REQUIRED: 운영자 데이터가 필요합니다.",
                strategy="manual_baseball_data_required",
                source_tier="none",
                domain="",
                answerability_status="operator_data_required",
                expected_non_answer=True,
            ),
            _smoke_item(
                endpoint="/ai/chat/stream",
                answer="MANUAL_BASEBALL_DATA_REQUIRED: 운영자 데이터가 필요합니다.",
                strategy="manual_baseball_data_required",
                source_tier="none",
                domain="",
                answerability_status="operator_data_required",
                expected_non_answer=True,
            ),
        ]
    )

    result = verifier.verify_report(smoke_report=report, expectations_payload=payload)

    assert result["summary"]["status"] == "pass"


def test_smoke_verifier_fails_completion_stream_mismatch() -> None:
    payload = {
        "expectations": [_expectation("recovered")],
    }
    report = _report([_smoke_item(endpoint="/ai/chat/completion")])

    result = verifier.verify_report(smoke_report=report, expectations_payload=payload)

    assert result["summary"]["status"] == "fail"
    assert result["failures"][0]["code"] == "endpoint_missing"


def test_smoke_verifier_fails_when_expected_question_is_absent() -> None:
    payload = {
        "expectations": [_expectation("recovered")],
    }

    result = verifier.verify_report(smoke_report=_report([]), expectations_payload=payload)

    assert result["summary"]["status"] == "fail"
    assert result["summary"]["expected_question_count"] == 1
    assert result["summary"]["verified_item_count"] == 0
    assert [failure["endpoint"] for failure in result["failures"]] == [
        "/ai/chat/completion",
        "/ai/chat/stream",
    ]
    assert {failure["code"] for failure in result["failures"]} == {"endpoint_missing"}
