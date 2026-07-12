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


def _case(index: int, *, planner_mode: str = "default_llm_planner") -> dict[str, Any]:
    return {
        "id": f"case-{index:02d}",
        "question": f"question {index}",
        "category": "regulation",
        "expected_answerability": "answerable",
        "allowed_planner_modes": [planner_mode],
    }


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
) -> dict[str, Any]:
    cases = [_case(index) for index in range(60)]
    results = []
    for index, case in enumerate(cases):
        evidence = _success_evidence(
            planner_cost=planner_cost if index == 0 else "0.000000000000",
            answer_cost=answer_cost if index == 0 else "0.000000000000",
            planner_model=planner_label,
            answer_model=answer_label,
        )
        results.append(experiment.evaluate_case(case, evidence))
    return experiment.build_run_report(
        dataset_sha256="a" * 64,
        cases=cases,
        results=results,
        planner_model_label=planner_label,
        answer_model_label=answer_label,
        cache_bypass=True,
    )


def test_load_golden_cases_returns_exact_hash_and_validated_cases(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    payload = {
        "schema_version": 1,
        "baseball_data_policy": "internal_only",
        "cases": [_case(0)],
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    path.write_bytes(raw)

    digest, cases = experiment.load_golden_cases(path)

    import hashlib

    assert digest == hashlib.sha256(raw).hexdigest()
    assert cases == payload["cases"]


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
    cases = [_case(index) for index in range(60)]
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
        dataset_sha256="a" * 64,
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
    cases = [_case(index) for index in range(60)]
    results = [
        experiment.evaluate_case(case, _success_evidence()) for case in cases
    ]
    results[0].update(
        {
            "answer": "must-not-leak",
            "prompt": "must-not-leak",
            "headers": {"Authorization": "must-not-leak"},
            "tool_payload": {"secret": "must-not-leak"},
        }
    )

    report = experiment.build_run_report(
        dataset_sha256="a" * 64,
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

    cases = [_case(index) for index in range(60)]
    results = [
        experiment.evaluate_case(
            case,
            _success_evidence(answer_model="wrong-answer-model"),
        )
        for case in cases
    ]
    with pytest.raises(experiment.InvalidEvidenceError):
        experiment.build_run_report(
            dataset_sha256="a" * 64,
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
