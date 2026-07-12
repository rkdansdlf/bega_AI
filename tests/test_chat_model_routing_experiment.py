"""Pure tests for the model-routing golden cost and quality gate."""

from __future__ import annotations

import argparse
import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest

from scripts import chat_model_routing_experiment as experiment


APPROVED_GOLDEN_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "chat_quality_golden_60.json"
)


def _case(index: int, *, planner_mode: str = "default_llm_planner") -> dict[str, Any]:
    return {
        "id": f"case-{index:02d}",
        "question": f"question {index}",
        "category": "regulation",
        "expected_answerability": "answerable",
        "allowed_planner_modes": [planner_mode],
    }


def _approved_cases() -> list[dict[str, Any]]:
    return experiment.load_golden_cases(APPROVED_GOLDEN_PATH)[1]


def _answer_for_case(case: dict[str, Any]) -> str:
    expected = case["expected_answerability"]
    if expected == "operator_data_required":
        return "MANUAL_BASEBALL_DATA_REQUIRED"
    if expected == "clarification_required":
        return "팀명을 같이 알려주세요."
    if expected == "future_event_pending":
        return "미래의 기록은 공식 기록이 제공된 뒤 확인할 수 있습니다."
    return "자연스러운 답변입니다."


def _usage(
    role: str,
    cost: str,
    *,
    model: str | None = None,
    outcome: str = "success",
    pricing_source: str = "model_catalog",
) -> dict[str, Any]:
    return {
        "role": role,
        "provider": "test-provider",
        "model": model or f"explicit-{role}",
        "outcome": outcome,
        "pricing_source": pricing_source,
        "input_chars": 10,
        "output_chars": 10,
        "input_tokens": 3,
        "output_tokens": 3,
        "input_cost_usd": "0.000000000000",
        "output_cost_usd": cost,
        "total_cost_usd": cost,
    }


def _success_evidence(
    *,
    planner_cost: str = "0.000000000000",
    answer_cost: str = "0.000000000000",
    planner_mode: str = "default_llm_planner",
    planner_model: str = "explicit-planner",
    answer_model: str = "explicit-answer",
    answer: str = "자연스러운 답변입니다.",
) -> dict[str, Any]:
    return {
        "ok": True,
        "status_code": 200,
        "latency_ms": 10.0,
        "cache_bypass": True,
        "payload": {
            "answer": answer,
            "planner_mode": planner_mode,
            "fallback_triggered": False,
            "fallback_answer_used": False,
            "fallback_reason": None,
            "model_usage_complete": True,
            "model_usage": [
                _usage("planner", planner_cost, model=planner_model),
                _usage("answer", answer_cost, model=answer_model),
            ],
        },
    }


def _report(
    *,
    planner_cost: str,
    answer_cost: str = "2.000000000000",
    planner_label: str = "explicit-planner",
    answer_label: str = "explicit-answer",
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    cases = _approved_cases()
    cost_case_index = next(
        index
        for index, case in enumerate(cases)
        if any(mode in experiment.LLM_PLANNER_MODES for mode in case["allowed_planner_modes"])
    )
    results = []
    for index, case in enumerate(cases):
        planner_mode = next(
            (mode for mode in case["allowed_planner_modes"] if mode in experiment.LLM_PLANNER_MODES),
            case["allowed_planner_modes"][0],
        )
        evidence = _success_evidence(
            planner_cost=planner_cost if index == cost_case_index else "0.000000000000",
            answer_cost=answer_cost if index == cost_case_index else "0.000000000000",
            planner_model=planner_label,
            answer_model=answer_label,
            answer=_answer_for_case(case),
            planner_mode=planner_mode,
        )
        evidence["payload"]["fallback_reason"] = fallback_reason if index == 0 else None
        if planner_mode in experiment.DETERMINISTIC_PLANNER_MODES:
            evidence["payload"]["model_usage"] = []
        results.append(experiment.evaluate_case(case, evidence))
    return experiment.build_run_report(
        dataset_sha256=experiment.APPROVED_GOLDEN_SHA256,
        cases=cases,
        results=results,
        planner_model_label=planner_label,
        answer_model_label=answer_label,
        cache_bypass=True,
    )


def _report_from_details(details: list[dict[str, Any]]) -> dict[str, Any]:
    return experiment.build_run_report(
        dataset_sha256=experiment.APPROVED_GOLDEN_SHA256,
        cases=_approved_cases(),
        results=details,
        planner_model_label="explicit-planner",
        answer_model_label="explicit-answer",
        cache_bypass=True,
    )


def _passing_evidence_for_approved_case(case: dict[str, Any]) -> dict[str, Any]:
    planner_mode = next(
        (
            mode
            for mode in case["allowed_planner_modes"]
            if mode in experiment.LLM_PLANNER_MODES
        ),
        case["allowed_planner_modes"][0],
    )
    evidence = _success_evidence(
        planner_mode=planner_mode,
        answer=_answer_for_case(case),
    )
    if planner_mode in experiment.DETERMINISTIC_PLANNER_MODES:
        evidence["payload"]["model_usage"] = []
    return evidence


def test_load_golden_cases_accepts_only_the_operator_approved_asset() -> None:
    digest, cases = experiment.load_golden_cases(APPROVED_GOLDEN_PATH)

    assert digest == experiment.APPROVED_GOLDEN_SHA256
    assert len(cases) == 60
    assert cases[0]["id"] == "narrative-01"


def test_load_golden_cases_rejects_a_valid_schema_with_an_unapproved_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "golden.json"
    payload = json.loads(APPROVED_GOLDEN_PATH.read_text(encoding="utf-8"))
    payload["description"] = "different operator asset"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.load_golden_cases(path)


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2, "cases": []},
        {"schema_version": 1, "cases": "not-a-list"},
        {"schema_version": 1, "cases": [{"id": "missing-fields"}]},
        {
            "schema_version": 1,
            "cases": [_case(0), {**_case(1), "id": "case-00"}],
        },
    ],
)
def test_load_golden_cases_rejects_unsafe_schema(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.load_golden_cases(path)


def test_evaluate_case_uses_top_level_metadata_and_discards_sensitive_text() -> None:
    evidence = _success_evidence(answer="MANUAL_BASEBALL_DATA_REQUIRED")
    case = {
        **_case(0),
        "expected_answerability": "operator_data_required",
    }
    evidence["payload"].update(
        {
            "prompt": "must-not-leak",
            "tool_results": [{"secret": "must-not-leak"}],
            "headers": {"Authorization": "must-not-leak"},
        }
    )

    result = experiment.evaluate_case(case, evidence)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["answerability_status"] == "operator_data_required"
    assert result["passed"] is True
    assert "answer" not in result
    assert "MANUAL_BASEBALL_DATA_REQUIRED" not in serialized
    assert "must-not-leak" not in serialized


def test_evaluate_case_retains_answerability_failure_with_expected_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment,
        "_evaluate_answerability",
        lambda _answer, _question: {
            "status": "answerable",
            "answerability_pass": False,
        },
    )

    result = experiment.evaluate_case(_case(0), _success_evidence())

    assert result["answerability_status"] == "answerable"
    assert result["answerability_pass"] is False
    assert result["unexpected_non_answer"] is False
    assert result["failure_reasons"] == ["answerability"]


def test_answerability_failure_report_is_measured_and_exits_one() -> None:
    details = deepcopy(_report(planner_cost="1.000000000000")["details"])
    case_index = next(
        index
        for index, case in enumerate(_approved_cases())
        if case["expected_answerability"] == "answerable"
    )
    details[case_index].update(
        answerability_pass=False,
        failure_reasons=["answerability"],
        passed=False,
    )

    report = _report_from_details(details)

    assert report["details"][case_index]["answerability_pass"] is False
    assert report["summary"]["quality_passed"] is False
    assert experiment.report_exit_code(report) == 1


def test_unknown_planner_mode_report_is_measured_and_exits_one() -> None:
    details = deepcopy(_report(planner_cost="1.000000000000")["details"])
    details[0].update(
        planner_mode="unrecognized_mode",
        planner_mode_allowed=False,
        failure_reasons=["planner_mode"],
        passed=False,
    )

    report = _report_from_details(details)

    assert report["details"][0]["planner_mode"] == "unrecognized_mode"
    assert report["summary"]["quality_passed"] is False
    assert experiment.report_exit_code(report) == 1


@pytest.mark.parametrize("planner_mode", ["has whitespace", "line\nbreak", "x" * 129])
def test_evaluate_case_rejects_unsafe_planner_mode_evidence(planner_mode: str) -> None:
    evidence = _success_evidence(planner_mode=planner_mode)

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.evaluate_case(_case(0), evidence)


def test_evaluate_case_rejects_success_with_incomplete_usage() -> None:
    evidence = _success_evidence()
    evidence["payload"]["model_usage_complete"] = False

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.evaluate_case(_case(0), evidence)


def test_evaluate_case_rejects_llm_planner_without_planner_usage() -> None:
    evidence = _success_evidence()
    evidence["payload"]["model_usage"] = [_usage("answer", "1.0")]

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.evaluate_case(_case(0), evidence)


def test_evaluate_case_rejects_llm_planner_without_answer_usage() -> None:
    evidence = _success_evidence()
    evidence["payload"]["model_usage"] = [_usage("planner", "1.0")]

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.evaluate_case(_case(0), evidence)


def test_evaluate_case_allows_deterministic_mode_without_model_usage() -> None:
    case = _case(0, planner_mode="fast_path")
    evidence = _success_evidence(planner_mode="fast_path")
    evidence["payload"]["model_usage"] = []

    result = experiment.evaluate_case(case, evidence)

    assert result["passed"] is True
    assert result["model_usage"] == []


def test_evaluate_case_rejects_unpriced_successful_usage() -> None:
    evidence = _success_evidence()
    evidence["payload"]["model_usage"][0].update(
        {"pricing_source": "unpriced", "total_cost_usd": None}
    )

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.evaluate_case(_case(0), evidence)


def test_evaluate_case_accepts_independently_rounded_usage_costs() -> None:
    evidence = _success_evidence()
    usage = evidence["payload"]["model_usage"][0]
    usage.update(
        {
            "input_cost_usd": "0.000000000001",
            "output_cost_usd": "0.000000000001",
            "total_cost_usd": "0.000000000001",
        }
    )

    result = experiment.evaluate_case(_case(0), evidence)

    assert result["model_usage"][0]["input_cost_usd"] == "0.000000000001"
    assert result["model_usage"][0]["output_cost_usd"] == "0.000000000001"
    assert result["model_usage"][0]["total_cost_usd"] == "0.000000000001"


@pytest.mark.parametrize(
    "value", [True, -1, float("nan"), float("inf"), float("-inf")]
)
def test_latency_rejects_non_finite_or_negative_values(value: object) -> None:
    with pytest.raises(experiment.InvalidEvidenceError):
        experiment._latency_ms(value)


@pytest.mark.parametrize(
    "ok,status_code,failure_type",
    [
        (True, 199, None),
        (True, 300, None),
        (False, 200, "http_error"),
        (False, 99, "http_error"),
        (False, 600, "http_error"),
        (False, 503, "transport_error"),
    ],
)
def test_evaluate_case_rejects_invalid_http_status_contract(
    ok: bool, status_code: int, failure_type: str | None
) -> None:
    evidence = _success_evidence()
    evidence.update(ok=ok, status_code=status_code)
    if not ok:
        evidence["failure_type"] = failure_type

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.evaluate_case(_case(0), evidence)


@pytest.mark.parametrize("value", ["1e999999", "1e999999999999999999"])
def test_fixed_costs_fail_closed_for_huge_magnitudes(value: str) -> None:
    with pytest.raises(experiment.InvalidEvidenceError):
        experiment._fixed_usd_evidence(value, "cost")
    with pytest.raises(experiment.InvalidEvidenceError):
        experiment._fixed_usd(experiment.Decimal(value))


def test_extreme_fixed_cost_comparison_is_a_controlled_gate_failure() -> None:
    comparison = experiment.compare_reports(
        _report(planner_cost="0.000000000001"),
        _report(planner_cost="999999999999999.000000000000"),
    )

    assert comparison["gate_passed"] is False
    assert comparison["planner_reduction_percent"] == (
        "-99999999999999899999999999900.00"
    )
    assert comparison["candidate_total_cost_non_increasing"] is False
    assert experiment.report_exit_code({}, comparison) == 1


def test_overlong_fixed_cost_is_invalid_evidence() -> None:
    overlong = "9" * 200 + ".000000000000"

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment._fixed_usd_evidence(overlong, "cost")
    with pytest.raises(experiment.InvalidEvidenceError):
        experiment._fixed_usd(experiment.Decimal(overlong))


def test_increased_failed_model_attempts_are_a_new_failure() -> None:
    baseline_details = deepcopy(_report(planner_cost="1.000000000000")["details"])
    candidate_details = deepcopy(_report(planner_cost="0.800000000000")["details"])

    for details, attempt_count in (
        (baseline_details, 1),
        (candidate_details, 2),
    ):
        usage = [
            _usage("planner", "0.000000000001", outcome="failed"),
            _usage("answer", "0.000000000001"),
        ]
        if attempt_count == 2:
            usage.insert(1, _usage("planner", "0.000000000001", outcome="failed"))
        details[0].update(
            model_usage=usage,
            failed_model_attempts=attempt_count,
            failure_reasons=["failed_model_attempt"],
            passed=False,
        )

    comparison = experiment.compare_reports(
        _report_from_details(baseline_details),
        _report_from_details(candidate_details),
    )

    assert comparison["no_new_failures"] is False
    assert "candidate_introduced_new_failure" in comparison["failure_reasons"]


def test_equal_failed_model_attempts_are_not_a_new_failure() -> None:
    baseline_details = deepcopy(_report(planner_cost="1.000000000000")["details"])
    candidate_details = deepcopy(_report(planner_cost="0.800000000000")["details"])
    failed_usage = [
        _usage("planner", "0.000000000001", outcome="failed"),
        _usage("answer", "0.000000000001"),
    ]
    for details in (baseline_details, candidate_details):
        details[0].update(
            model_usage=failed_usage,
            failed_model_attempts=1,
            failure_reasons=["failed_model_attempt"],
            passed=False,
        )

    comparison = experiment.compare_reports(
        _report_from_details(baseline_details),
        _report_from_details(candidate_details),
    )

    assert comparison["no_new_failures"] is True


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda payload: payload.update(planner_mode="other_mode"), "planner_mode"),
        (lambda payload: payload.update(fallback_triggered=True), "planner_fallback"),
        (lambda payload: payload.update(fallback_answer_used=True), "answer_fallback"),
    ],
)
def test_unexpected_mode_or_fallback_is_a_measured_failure(mutation, reason: str) -> None:
    evidence = _success_evidence()
    mutation(evidence["payload"])

    result = experiment.evaluate_case(_case(0), evidence)

    assert result["passed"] is False
    assert reason in result["failure_reasons"]


def test_fallback_reason_marker_is_a_sanitized_measured_failure() -> None:
    evidence = _success_evidence()
    evidence["payload"]["fallback_reason"] = "sensitive provider failure"

    result = experiment.evaluate_case(_case(0), evidence)

    assert result["passed"] is False
    assert "fallback_reason" in result["failure_reasons"]
    assert result["fallback_reason_present"] is True
    assert "sensitive provider failure" not in json.dumps(result)


def test_fallback_reason_marker_is_a_new_failure_in_comparison() -> None:
    comparison = experiment.compare_reports(
        _report(planner_cost="1.000000000000"),
        _report(
            planner_cost="0.800000000000",
            fallback_reason="sensitive provider failure",
        ),
    )

    assert comparison["candidate_quality_passed"] is False
    assert comparison["no_new_failures"] is False
    assert "candidate_introduced_new_failure" in comparison["failure_reasons"]


def test_candidate_http_failure_is_a_new_failure_in_comparison() -> None:
    cases = _approved_cases()
    case_index = next(
        index
        for index, case in enumerate(cases)
        if not any(mode in experiment.LLM_PLANNER_MODES for mode in case["allowed_planner_modes"])
    )
    candidate_details = deepcopy(_report(planner_cost="0.800000000000")["details"])
    candidate_details[case_index] = experiment.evaluate_case(
        cases[case_index],
        {
            "ok": False,
            "status_code": 503,
            "latency_ms": 10.0,
            "failure_type": "http_error",
        },
    )

    comparison = experiment.compare_reports(
        _report(planner_cost="1.000000000000"),
        _report_from_details(candidate_details),
    )

    assert comparison["gate_passed"] is False
    assert comparison["no_new_failures"] is False


def test_candidate_quality_failure_is_a_new_failure_in_comparison() -> None:
    cases = _approved_cases()
    case_index = next(
        index
        for index, case in enumerate(cases)
        if case["expected_answerability"] == "answerable"
        and not any(mode in experiment.LLM_PLANNER_MODES for mode in case["allowed_planner_modes"])
    )
    candidate_details = deepcopy(_report(planner_cost="0.800000000000")["details"])
    evidence = _passing_evidence_for_approved_case(cases[case_index])
    evidence["payload"]["answer"] = "| 항목 | 값 |\n| --- | --- |\n| A | B |"
    candidate_details[case_index] = experiment.evaluate_case(cases[case_index], evidence)

    comparison = experiment.compare_reports(
        _report(planner_cost="1.000000000000"),
        _report_from_details(candidate_details),
    )

    assert comparison["gate_passed"] is False
    assert comparison["no_new_failures"] is False


def test_existing_case_failure_reason_is_not_new_in_comparison() -> None:
    cases = _approved_cases()
    case_index = next(
        index
        for index, case in enumerate(cases)
        if not any(mode in experiment.LLM_PLANNER_MODES for mode in case["allowed_planner_modes"])
    )
    failed_case = experiment.evaluate_case(
        cases[case_index],
        {
            "ok": False,
            "status_code": 503,
            "latency_ms": 10.0,
            "failure_type": "http_error",
        },
    )
    baseline_details = deepcopy(_report(planner_cost="1.000000000000")["details"])
    candidate_details = deepcopy(_report(planner_cost="0.800000000000")["details"])
    baseline_details[case_index] = failed_case
    candidate_details[case_index] = failed_case

    comparison = experiment.compare_reports(
        _report_from_details(baseline_details),
        _report_from_details(candidate_details),
    )

    assert comparison["gate_passed"] is False
    assert comparison["no_new_failures"] is True


def test_exact_twenty_percent_planner_reduction_passes() -> None:
    baseline = _report(planner_cost="1.000000000000")
    candidate = _report(planner_cost="0.800000000000")

    comparison = experiment.compare_reports(baseline, candidate)

    assert comparison["gate_passed"] is True
    assert comparison["planner_reduction_percent"] == "20.00"
    assert comparison["candidate_total_cost_non_increasing"] is True


def test_nineteen_point_ninety_nine_percent_reduction_fails() -> None:
    comparison = experiment.compare_reports(
        _report(planner_cost="1.000000000000"),
        _report(planner_cost="0.800100000000"),
    )

    assert comparison["planner_reduction_percent"] == "19.99"
    assert comparison["gate_passed"] is False


def test_candidate_total_cost_increase_fails() -> None:
    comparison = experiment.compare_reports(
        _report(planner_cost="1.000000000000", answer_cost="2.000000000000"),
        _report(planner_cost="0.800000000000", answer_cost="2.300000000000"),
    )

    assert comparison["candidate_total_cost_non_increasing"] is False
    assert comparison["gate_passed"] is False


def test_zero_baseline_planner_cost_is_a_measured_gate_failure() -> None:
    comparison = experiment.compare_reports(
        _report(planner_cost="0.000000000000"),
        _report(planner_cost="0.000000000000"),
    )

    assert comparison["baseline_planner_cost_positive"] is False
    assert comparison["gate_passed"] is False


@pytest.mark.parametrize("label", ["", "  ", "default", "DEFAULT"])
def test_default_or_blank_explicit_model_label_is_invalid(label: str) -> None:
    with pytest.raises(experiment.InvalidEvidenceError):
        _report(planner_cost="1.000000000000", planner_label=label)


def test_changed_answer_model_label_is_invalid() -> None:
    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.compare_reports(
            _report(planner_cost="1.000000000000"),
            _report(
                planner_cost="0.800000000000",
                answer_label="different-answer",
            ),
        )


@pytest.mark.parametrize("field", ["dataset_sha256", "ordered_case_ids"])
def test_hash_or_order_mismatch_is_invalid(field: str) -> None:
    baseline = _report(planner_cost="1.000000000000")
    candidate = _report(planner_cost="0.800000000000")
    if field == "dataset_sha256":
        candidate[field] = "b" * 64
    else:
        candidate[field] = list(reversed(candidate[field]))

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.compare_reports(baseline, candidate)


def test_build_run_report_rejects_a_constructed_report_with_the_wrong_hash() -> None:
    report = _report(planner_cost="1.000000000000")

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.build_run_report(
            dataset_sha256="a" * 64,
            cases=_approved_cases(),
            results=report["details"],
            planner_model_label="explicit-planner",
            answer_model_label="explicit-answer",
            cache_bypass=True,
        )


@pytest.mark.parametrize("mutation", ["duplicate", "missing"])
def test_duplicate_or_missing_report_case_is_invalid(mutation: str) -> None:
    candidate = _report(planner_cost="0.800000000000")
    if mutation == "duplicate":
        candidate["ordered_case_ids"][-1] = candidate["ordered_case_ids"][0]
    else:
        candidate["details"].pop()

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.compare_reports(
            _report(planner_cost="1.000000000000"), candidate
        )


def test_all_http_failures_form_a_complete_measured_failure_report() -> None:
    cases = _approved_cases()
    results = [
        experiment.evaluate_case(
            case,
            {
                "ok": False,
                "status_code": 503,
                "latency_ms": 5.0,
                "cache_bypass": True,
                "failure_type": "http_error",
            },
        )
        for case in cases
    ]

    report = experiment.build_run_report(
        dataset_sha256=experiment.APPROVED_GOLDEN_SHA256,
        cases=cases,
        results=results,
        planner_model_label="explicit-planner",
        answer_model_label="explicit-answer",
        cache_bypass=True,
    )

    assert report["summary"]["failure_count"] == 60
    assert report["summary"]["quality_passed"] is False
    assert experiment.report_exit_code(report) == 1


def test_build_report_whitelists_case_detail_fields() -> None:
    cases = _approved_cases()
    results = deepcopy(_report(planner_cost="1.000000000000")["details"])
    results[0].update(
        {
            "answer": "must-not-leak",
            "prompt": "must-not-leak",
            "headers": {"Authorization": "must-not-leak"},
            "tool_payload": {"secret": "must-not-leak"},
        }
    )

    report = experiment.build_run_report(
        dataset_sha256=experiment.APPROVED_GOLDEN_SHA256,
        cases=cases,
        results=results,
        planner_model_label="explicit-planner",
        answer_model_label="explicit-answer",
        cache_bypass=True,
    )

    assert "must-not-leak" not in json.dumps(report)


@pytest.mark.asyncio
async def test_runner_calls_once_per_case_with_cache_bypass_and_discards_answers() -> None:
    cases = [_case(0, planner_mode="fast_path"), _case(1, planner_mode="fast_path")]
    seen_questions: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_questions.append(body["question"])
        assert body["cache_bypass"] is True
        return httpx.Response(
            200,
            json={
                "answer": f"sensitive answer {body['question']}",
                "planner_mode": "fast_path",
                "fallback_triggered": False,
                "fallback_answer_used": False,
                "model_usage_complete": True,
                "model_usage": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await experiment.run_golden_cases(
            client=client,
            base_url="http://test",
            internal_api_key="secret-key",
            cases=cases,
            cache_bypass=True,
        )

    assert seen_questions == ["question 0", "question 1"]
    serialized = json.dumps(results)
    assert "sensitive answer" not in serialized
    assert "secret-key" not in serialized


@pytest.mark.asyncio
async def test_runner_continues_after_transport_and_http_failures() -> None:
    cases = [_case(index, planner_mode="fast_path") for index in range(3)]
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("credential-looking transport detail", request=request)
        if call_count == 2:
            return httpx.Response(503, json={"detail": "sensitive server detail"})
        return httpx.Response(
            200,
            json={
                "answer": "자연스러운 답변입니다.",
                "planner_mode": "fast_path",
                "fallback_triggered": False,
                "fallback_answer_used": False,
                "model_usage_complete": True,
                "model_usage": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await experiment.run_golden_cases(
            client=client,
            base_url="http://test",
            internal_api_key="secret-key",
            cases=cases,
        )

    assert [result["id"] for result in results] == [case["id"] for case in cases]
    assert [result["http_ok"] for result in results] == [False, False, True]
    assert "sensitive" not in json.dumps(results)
    assert "credential-looking" not in json.dumps(results)


def test_main_has_stable_exit_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment, "parse_args", lambda: argparse.Namespace())

    async def returns(code: int) -> int:
        return code

    for expected in (0, 1):
        monkeypatch.setattr(experiment, "async_main", lambda args, code=expected: returns(code))
        assert experiment.main() == expected

    async def invalid(_args: argparse.Namespace) -> int:
        raise experiment.InvalidEvidenceError("must not expose secret-value")

    monkeypatch.setattr(experiment, "async_main", invalid)
    assert experiment.main() == 2


def test_report_rejects_false_cache_bypass_and_observed_answer_model_mismatch() -> None:
    report = _report(planner_cost="1.000000000000")
    bad_cache = deepcopy(report)
    bad_cache["cache_bypass"] = False
    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.compare_reports(report, bad_cache)

    cases = _approved_cases()
    results = []
    for case in cases:
        planner_mode = next(
            (mode for mode in case["allowed_planner_modes"] if mode in experiment.LLM_PLANNER_MODES),
            case["allowed_planner_modes"][0],
        )
        evidence = _success_evidence(
            planner_mode=planner_mode,
            answer=_answer_for_case(case),
            answer_model="wrong-answer-model",
        )
        if planner_mode in experiment.DETERMINISTIC_PLANNER_MODES:
            evidence["payload"]["model_usage"] = []
        results.append(experiment.evaluate_case(case, evidence))
    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.build_run_report(
            dataset_sha256=experiment.APPROVED_GOLDEN_SHA256,
            cases=cases,
            results=results,
            planner_model_label="explicit-planner",
            answer_model_label="explicit-answer",
            cache_bypass=True,
        )


def test_compare_rejects_non_boolean_case_routing_flag() -> None:
    baseline = _report(planner_cost="1.000000000000")
    candidate = _report(planner_cost="0.800000000000")
    candidate["details"][0]["planner_fallback"] = "false"

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.compare_reports(baseline, candidate)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["details"][0]["quality"].update(natural_chat=False),
        lambda report: report["details"][0].update(passed=False),
        lambda report: report["details"][0].update(failure_reasons=["quality"]),
    ],
)
def test_loaded_report_rejects_forged_case_quality_or_outcome(mutate) -> None:
    candidate = _report(planner_cost="0.800000000000")
    mutate(candidate)

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.compare_reports(_report(planner_cost="1.000000000000"), candidate)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["summary"].update(latency_avg_ms=999.99),
        lambda report: report["summary"].update(total_cost_usd="9.000000000000"),
        lambda report: report["summary"].update(failure_count=False),
        lambda report: report["cost_totals_usd"].update(total="9.000000000000"),
    ],
)
def test_loaded_report_rejects_forged_summary_or_cost_totals(mutate) -> None:
    candidate = _report(planner_cost="0.800000000000")
    mutate(candidate)

    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.compare_reports(_report(planner_cost="1.000000000000"), candidate)
