#!/usr/bin/env python3
"""Run and compare sanitized model-routing golden evidence."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from decimal import Decimal, DecimalException, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import httpx

from scripts.smoke_chatbot_quality import (
    _evaluate_answerability,
    _evaluate_quality,
    _quality_pass,
)


EXPECTED_CASE_COUNT = 60
REPORT_SCHEMA_VERSION = 1
APPROVED_GOLDEN_SHA256 = "8899b468d65a87a505083a5f229808a0b9b405819287f00ff707741715bb124d"
APPROVED_GOLDEN_PATH = Path(__file__).with_name("chat_quality_golden_60.json")
APPROVED_CATEGORY_COUNTS = {
    "multi_player_narrative": 20,
    "regulation": 20,
    "team_db": 14,
    "recovered_regression": 6,
}
DETERMINISTIC_PLANNER_MODES = {
    "fast_path",
    "fast_path_bundle",
    "player_fast_path",
    "predefined",
}
LLM_PLANNER_MODES = {
    "default_llm_planner",
    "player_llm_planner",
    "team_llm_planner",
}
APPROVED_PLANNER_MODES = DETERMINISTIC_PLANNER_MODES | LLM_PLANNER_MODES
APPROVED_ANSWERABILITY = {
    "answerable",
    "operator_data_required",
    "clarification_required",
    "future_event_pending",
}
USAGE_ROLES = {"planner", "answer"}
FORBIDDEN_MODEL_LABELS = {"default", "unknown"}
USD_QUANTUM = Decimal("0.000000000001")
PERCENT_QUANTUM = Decimal("0.01")
MAX_PLANNER_MODE_LENGTH = 128
HTTP_STATUS_MIN = 100
HTTP_STATUS_MAX = 599
QUALITY_KEYS = {
    "natural_chat",
    "no_table_markup",
    "no_briefing_headers",
    "no_source_line",
    "no_raw_chunk_marker",
    "no_briefing_intro",
    "no_low_data_fallback",
}
NATURAL_CHAT_KEYS = {
    "no_table_markup",
    "no_briefing_headers",
    "no_source_line",
    "no_raw_chunk_marker",
    "no_briefing_intro",
}


class InvalidEvidenceError(ValueError):
    """Raised when a run cannot provide structurally valid gate evidence."""


def _load_questions(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json", ".jsonl"}:
        questions: List[str] = []
        records: Iterable[Any]
        if path.suffix.lower() == ".json":
            parsed = json.loads(raw)
            records = (
                parsed.get("samples", parsed) if isinstance(parsed, dict) else parsed
            )
        else:
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        for item in records:
            if isinstance(item, str):
                questions.append(item)
            elif isinstance(item, dict) and item.get("question"):
                questions.append(str(item["question"]))
        return questions
    return [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return round(ordered[index], 2)


def _require_nonblank_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidEvidenceError(f"{field} must be a non-blank string")
    return value.strip()


def _explicit_model_label(value: object, field: str) -> str:
    label = _require_nonblank_string(value, field)
    if label.lower() in FORBIDDEN_MODEL_LABELS:
        raise InvalidEvidenceError(f"{field} must be explicit")
    return label


def _planner_mode(value: object, field: str) -> str:
    mode = _require_nonblank_string(value, field)
    if len(mode) > MAX_PLANNER_MODE_LENGTH or any(
        not (char.isascii() and (char.isalnum() or char in "_.:-"))
        for char in mode
    ):
        raise InvalidEvidenceError(f"{field} must be a bounded safe identifier")
    return mode


def _valid_http_status(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and HTTP_STATUS_MIN <= value <= HTTP_STATUS_MAX
    )


def _case_contract(case: object) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise InvalidEvidenceError("golden cases must be objects")
    case_id = _require_nonblank_string(case.get("id"), "case id")
    question = _require_nonblank_string(case.get("question"), "case question")
    category = _require_nonblank_string(case.get("category"), "case category")
    if category not in APPROVED_CATEGORY_COUNTS:
        raise InvalidEvidenceError("golden case category is unsupported")
    expected = _require_nonblank_string(
        case.get("expected_answerability"), "expected answerability"
    )
    if expected not in APPROVED_ANSWERABILITY:
        raise InvalidEvidenceError("expected answerability is unsupported")
    modes = case.get("allowed_planner_modes")
    if not isinstance(modes, list) or not modes:
        raise InvalidEvidenceError("allowed planner modes must be a non-empty list")
    normalized_modes = [
        _require_nonblank_string(mode, "allowed planner mode") for mode in modes
    ]
    if len(normalized_modes) != len(set(normalized_modes)):
        raise InvalidEvidenceError("allowed planner modes must be unique")
    if any(mode not in APPROVED_PLANNER_MODES for mode in normalized_modes):
        raise InvalidEvidenceError("allowed planner mode is unsupported")
    return {
        "id": case_id,
        "question": question,
        "category": category,
        "expected_answerability": expected,
        "allowed_planner_modes": normalized_modes,
    }


def _validated_golden_cases(cases: object) -> list[dict[str, Any]]:
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise InvalidEvidenceError("release evidence requires exactly 60 cases")
    validated = [_case_contract(case) for case in cases]
    case_ids = [case["id"] for case in validated]
    questions = [case["question"] for case in validated]
    if len(case_ids) != len(set(case_ids)):
        raise InvalidEvidenceError("golden case ids must be unique")
    if len(questions) != len(set(questions)):
        raise InvalidEvidenceError("golden case questions must be unique")
    if Counter(case["category"] for case in validated) != APPROVED_CATEGORY_COUNTS:
        raise InvalidEvidenceError("golden case category counts are invalid")
    return validated


def load_golden_cases(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Load schema-1 golden cases and return the hash of the exact input bytes."""

    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidEvidenceError("golden dataset could not be loaded") from exc
    if not isinstance(payload, dict):
        raise InvalidEvidenceError("golden dataset root must be an object")
    cases = payload.get("cases")
    if set(payload) != {
        "schema_version",
        "name",
        "description",
        "baseball_data_policy",
        "cases",
    }:
        raise InvalidEvidenceError("unsupported golden dataset")
    if payload.get("schema_version") != 1 or not isinstance(cases, list):
        raise InvalidEvidenceError("unsupported golden dataset")
    _require_nonblank_string(payload.get("name"), "golden dataset name")
    _require_nonblank_string(payload.get("description"), "golden dataset description")
    if payload.get("baseball_data_policy") != "internal_only":
        raise InvalidEvidenceError("golden dataset must use internal baseball data")

    validated = _validated_golden_cases(cases)
    dataset_sha256 = hashlib.sha256(raw).hexdigest()
    if dataset_sha256 != APPROVED_GOLDEN_SHA256:
        raise InvalidEvidenceError("golden dataset hash is not approved")
    return dataset_sha256, validated


def _decimal(value: object, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise InvalidEvidenceError(f"{field} must be a decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise InvalidEvidenceError(f"{field} must be a decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise InvalidEvidenceError(f"{field} must be finite and non-negative")
    return parsed


def _fixed_usd(value: Decimal) -> str:
    try:
        return format(value.quantize(USD_QUANTUM), ".12f")
    except (DecimalException, OverflowError, ValueError) as exc:
        raise InvalidEvidenceError("USD amount cannot be represented safely") from exc


def _fixed_usd_evidence(value: object, field: str) -> Decimal:
    if not isinstance(value, str):
        raise InvalidEvidenceError(f"{field} must be a fixed decimal string")
    parsed = _decimal(value, field)
    if value != _fixed_usd(parsed):
        raise InvalidEvidenceError(f"{field} must use 12 decimal places")
    return parsed


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidEvidenceError(f"{field} must be a non-negative integer")
    return value


def _sanitize_usage(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InvalidEvidenceError("model usage records must be objects")
    role = _require_nonblank_string(raw.get("role"), "usage role")
    if role not in USAGE_ROLES:
        raise InvalidEvidenceError("usage role must be planner or answer")
    provider = _explicit_model_label(raw.get("provider"), "usage provider")
    model = _explicit_model_label(raw.get("model"), "usage model")
    outcome = _require_nonblank_string(raw.get("outcome"), "usage outcome")
    if outcome not in {"success", "failed"}:
        raise InvalidEvidenceError("usage outcome is invalid")
    if raw.get("pricing_source") != "model_catalog":
        raise InvalidEvidenceError("all observed model usage must be catalog priced")

    input_cost = _fixed_usd_evidence(raw.get("input_cost_usd"), "input cost")
    output_cost = _fixed_usd_evidence(raw.get("output_cost_usd"), "output cost")
    total_cost = _fixed_usd_evidence(raw.get("total_cost_usd"), "total cost")
    if abs(input_cost + output_cost - total_cost) > USD_QUANTUM:
        raise InvalidEvidenceError("usage cost components do not equal total cost")

    return {
        "role": role,
        "provider": provider,
        "model": model,
        "outcome": outcome,
        "pricing_source": "model_catalog",
        "input_chars": _nonnegative_int(raw.get("input_chars"), "input chars"),
        "output_chars": _nonnegative_int(raw.get("output_chars"), "output chars"),
        "input_tokens": _nonnegative_int(raw.get("input_tokens"), "input tokens"),
        "output_tokens": _nonnegative_int(raw.get("output_tokens"), "output tokens"),
        "input_cost_usd": _fixed_usd(input_cost),
        "output_cost_usd": _fixed_usd(output_cost),
        "total_cost_usd": _fixed_usd(total_cost),
    }


def _latency_ms(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidEvidenceError("latency must be a non-negative number")
    try:
        latency = float(value)
    except (OverflowError, ValueError) as exc:
        raise InvalidEvidenceError("latency must be a non-negative number") from exc
    if not math.isfinite(latency) or latency < 0:
        raise InvalidEvidenceError("latency must be a non-negative number")
    return round(latency, 2)


def evaluate_case(case: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Evaluate transient HTTP/result evidence and return a sanitized case record."""

    golden = _case_contract(case)
    if not isinstance(evidence, dict):
        raise InvalidEvidenceError("case evidence must be an object")
    ok = evidence.get("ok")
    if not isinstance(ok, bool):
        raise InvalidEvidenceError("HTTP outcome must be explicit")
    latency_ms = _latency_ms(evidence.get("latency_ms"))
    status_code = evidence.get("status_code")
    if status_code is not None and not _valid_http_status(status_code):
        raise InvalidEvidenceError("HTTP status must be an integer or null")

    base = {
        "id": golden["id"],
        "category": golden["category"],
        "http_ok": ok,
        "status_code": status_code,
        "latency_ms": latency_ms,
    }
    if ok and (not _valid_http_status(status_code) or not 200 <= status_code < 300):
        raise InvalidEvidenceError("successful HTTP status is invalid")
    if not ok:
        failure_type = evidence.get("failure_type")
        if failure_type == "transport_error":
            if status_code is not None:
                raise InvalidEvidenceError("transport failure status must be null")
            safe_failure = "transport_error"
        elif failure_type == "http_error":
            if status_code is None or 200 <= status_code < 300:
                raise InvalidEvidenceError("HTTP failure status is invalid")
            safe_failure = "http_error"
        else:
            raise InvalidEvidenceError("failure type is invalid")
        return {
            **base,
            "passed": False,
            "failure_reasons": [safe_failure],
            "planner_mode": None,
            "planner_mode_allowed": False,
            "planner_fallback": False,
            "answer_fallback": False,
            "fallback_reason_present": False,
            "unexpected_non_answer": False,
            "answerability_pass": False,
            "answerability_status": None,
            "quality": {},
            "model_usage": [],
            "failed_model_attempts": 0,
        }

    payload = evidence.get("payload", evidence.get("result"))
    if not isinstance(payload, dict):
        raise InvalidEvidenceError("successful response payload must be an object")
    if payload.get("model_usage_complete") is not True:
        raise InvalidEvidenceError("successful response has incomplete model usage")
    raw_usage = payload.get("model_usage")
    if not isinstance(raw_usage, list):
        raise InvalidEvidenceError("successful response model usage must be a list")
    usage = [_sanitize_usage(record) for record in raw_usage]

    planner_mode = _planner_mode(
        payload.get("planner_mode"), "successful response planner mode"
    )
    if planner_mode in LLM_PLANNER_MODES and not any(
        record["role"] == "planner" for record in usage
    ):
        raise InvalidEvidenceError("LLM planner mode requires planner usage")
    if planner_mode in LLM_PLANNER_MODES and not any(
        record["role"] == "answer" for record in usage
    ):
        raise InvalidEvidenceError("LLM planner mode requires answer usage")

    planner_fallback = payload.get("fallback_triggered", False)
    answer_fallback = payload.get("fallback_answer_used", False)
    if not isinstance(planner_fallback, bool) or not isinstance(answer_fallback, bool):
        raise InvalidEvidenceError("fallback flags must be booleans")
    answer = payload.get("answer")
    if not isinstance(answer, str):
        raise InvalidEvidenceError("successful response answer must be text")

    quality = _evaluate_quality(answer)
    answerability = _evaluate_answerability(answer, golden["question"])
    actual_answerability = answerability.get("status")
    answerability_pass = answerability.get("answerability_pass")
    if actual_answerability not in APPROVED_ANSWERABILITY:
        raise InvalidEvidenceError("successful response answerability status is unsupported")
    if not isinstance(answerability_pass, bool):
        raise InvalidEvidenceError("successful response answerability pass is invalid")
    expected_answerability = golden["expected_answerability"]
    planner_mode_allowed = planner_mode in golden["allowed_planner_modes"]
    unexpected_non_answer = actual_answerability != expected_answerability
    failed_attempts = sum(record["outcome"] == "failed" for record in usage)
    fallback_reason_present = bool(payload.get("fallback_reason"))

    failure_reasons: list[str] = []
    if not _quality_pass(quality):
        failure_reasons.append("quality")
    if not answerability_pass or unexpected_non_answer:
        failure_reasons.append("answerability")
    if not planner_mode_allowed:
        failure_reasons.append("planner_mode")
    if planner_fallback:
        failure_reasons.append("planner_fallback")
    if answer_fallback:
        failure_reasons.append("answer_fallback")
    if fallback_reason_present:
        failure_reasons.append("fallback_reason")
    if failed_attempts:
        failure_reasons.append("failed_model_attempt")

    return {
        **base,
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "planner_mode": planner_mode,
        "planner_mode_allowed": planner_mode_allowed,
        "planner_fallback": planner_fallback,
        "answer_fallback": answer_fallback,
        "fallback_reason_present": fallback_reason_present,
        "unexpected_non_answer": unexpected_non_answer,
        "answerability_pass": answerability_pass,
        "answerability_status": actual_answerability,
        "quality": quality,
        "model_usage": usage,
        "failed_model_attempts": failed_attempts,
    }


async def _call_completion(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    internal_api_key: str,
    question: str,
    cache_bypass: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    response = await client.post(
        f"{base_url.rstrip('/')}/ai/chat/completion",
        json={"question": question, "cache_bypass": cache_bypass},
        headers={"X-Internal-Api-Key": internal_api_key},
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    if not response.is_success:
        return {
            "status_code": response.status_code,
            "ok": False,
            "latency_ms": latency_ms,
            "cache_bypass": cache_bypass,
            "failure_type": "http_error",
        }
    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        raise InvalidEvidenceError("successful response was not JSON") from exc
    return {
        "status_code": response.status_code,
        "ok": True,
        "latency_ms": latency_ms,
        "cache_bypass": cache_bypass,
        "payload": payload,
    }


async def run_golden_cases(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    internal_api_key: str,
    cases: list[dict[str, Any]],
    cache_bypass: bool = True,
) -> list[dict[str, Any]]:
    """Run cases sequentially once each and retain only evaluated evidence."""

    results: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        try:
            evidence = await _call_completion(
                client=client,
                base_url=base_url,
                internal_api_key=internal_api_key,
                question=case["question"],
                cache_bypass=cache_bypass,
            )
        except httpx.HTTPError:
            evidence = {
                "status_code": None,
                "ok": False,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "cache_bypass": cache_bypass,
                "failure_type": "transport_error",
            }
        results.append(evaluate_case(case, evidence))
    return results


def _validated_case_ids(cases: list[dict[str, Any]]) -> list[str]:
    return [case["id"] for case in _validated_golden_cases(cases)]


def _validate_observed_models(
    details: list[dict[str, Any]], planner_label: str, answer_label: str
) -> None:
    expected = {"planner": planner_label, "answer": answer_label}
    for detail in details:
        if not detail.get("http_ok"):
            continue
        for usage in detail.get("model_usage", []):
            role = usage["role"]
            if usage["model"] != expected[role]:
                raise InvalidEvidenceError(f"observed {role} model does not match label")


def _sanitize_case_detail(
    raw: object, golden_case: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InvalidEvidenceError("case details must be objects")
    failure_reasons = raw.get("failure_reasons")
    if not isinstance(failure_reasons, list) or not all(
        isinstance(reason, str) for reason in failure_reasons
    ):
        raise InvalidEvidenceError("case failure reasons must be a list of strings")
    http_ok = raw.get("http_ok")
    passed = raw.get("passed")
    if not isinstance(http_ok, bool) or not isinstance(passed, bool):
        raise InvalidEvidenceError("case outcomes must be booleans")
    status_code = raw.get("status_code")
    if status_code is not None and not _valid_http_status(status_code):
        raise InvalidEvidenceError("case HTTP status is invalid")
    planner_mode = raw.get("planner_mode")
    if planner_mode is not None:
        planner_mode = _planner_mode(planner_mode, "case planner mode")
    boolean_fields = (
        "planner_mode_allowed",
        "planner_fallback",
        "answer_fallback",
        "unexpected_non_answer",
        "answerability_pass",
        "fallback_reason_present",
    )
    if any(not isinstance(raw.get(field), bool) for field in boolean_fields):
        raise InvalidEvidenceError("case routing flags must be booleans")
    answerability_status = raw.get("answerability_status")
    if answerability_status is not None:
        answerability_status = _require_nonblank_string(
            answerability_status, "case answerability status"
        )
    model_usage = raw.get("model_usage")
    if not isinstance(model_usage, list):
        raise InvalidEvidenceError("case model usage must be a list")
    sanitized_usage = [_sanitize_usage(record) for record in model_usage]
    failed_model_attempts = _nonnegative_int(
        raw.get("failed_model_attempts"), "failed model attempts"
    )
    if failed_model_attempts != sum(
        record["outcome"] == "failed" for record in sanitized_usage
    ):
        raise InvalidEvidenceError("failed model attempt count is inconsistent")
    detail_id = _require_nonblank_string(raw.get("id"), "case detail id")
    category = _require_nonblank_string(raw.get("category"), "case detail category")
    if detail_id != golden_case["id"] or category != golden_case["category"]:
        raise InvalidEvidenceError("case detail does not match the approved dataset")

    quality = raw.get("quality")
    if http_ok:
        if not isinstance(status_code, int) or not 200 <= status_code < 300:
            raise InvalidEvidenceError("successful case HTTP status is invalid")
        if answerability_status not in APPROVED_ANSWERABILITY:
            raise InvalidEvidenceError("successful case answerability status is unsupported")
        if not isinstance(quality, dict) or set(quality) != QUALITY_KEYS or not all(
            isinstance(value, bool) for value in quality.values()
        ):
            raise InvalidEvidenceError("successful case quality evidence is invalid")
        natural_chat = all(quality[key] for key in NATURAL_CHAT_KEYS)
        if quality["natural_chat"] is not natural_chat:
            raise InvalidEvidenceError("successful case natural-chat evidence is inconsistent")
        planner_mode_allowed = planner_mode in golden_case["allowed_planner_modes"]
        unexpected_non_answer = (
            answerability_status != golden_case["expected_answerability"]
        )
        if raw["planner_mode_allowed"] is not planner_mode_allowed:
            raise InvalidEvidenceError("successful case planner-mode evidence is inconsistent")
        if raw["unexpected_non_answer"] is not unexpected_non_answer:
            raise InvalidEvidenceError("successful case answerability evidence is inconsistent")
        if planner_mode in LLM_PLANNER_MODES and not any(
            usage["role"] == "planner" for usage in sanitized_usage
        ):
            raise InvalidEvidenceError("successful case LLM planner usage is missing")
        if planner_mode in LLM_PLANNER_MODES and not any(
            usage["role"] == "answer" for usage in sanitized_usage
        ):
            raise InvalidEvidenceError("successful case LLM answer usage is missing")
        expected_failure_reasons: list[str] = []
        if not _quality_pass(quality):
            expected_failure_reasons.append("quality")
        if not raw["answerability_pass"] or unexpected_non_answer:
            expected_failure_reasons.append("answerability")
        if not planner_mode_allowed:
            expected_failure_reasons.append("planner_mode")
        if raw["planner_fallback"]:
            expected_failure_reasons.append("planner_fallback")
        if raw["answer_fallback"]:
            expected_failure_reasons.append("answer_fallback")
        if raw["fallback_reason_present"]:
            expected_failure_reasons.append("fallback_reason")
        if failed_model_attempts:
            expected_failure_reasons.append("failed_model_attempt")
        expected_passed = not expected_failure_reasons
        if failure_reasons != expected_failure_reasons or passed is not expected_passed:
            raise InvalidEvidenceError("successful case outcome evidence is inconsistent")
    else:
        if failure_reasons not in (["http_error"], ["transport_error"]):
            raise InvalidEvidenceError("failed case reason is invalid")
        if passed is not False or sanitized_usage or failed_model_attempts != 0:
            raise InvalidEvidenceError("failed case usage evidence is invalid")
        if (
            planner_mode is not None
            or raw["planner_mode_allowed"] is not False
            or raw["planner_fallback"] is not False
            or raw["answer_fallback"] is not False
            or raw["fallback_reason_present"] is not False
            or raw["unexpected_non_answer"] is not False
            or raw["answerability_pass"] is not False
            or answerability_status is not None
            or quality != {}
        ):
            raise InvalidEvidenceError("failed case routing evidence is invalid")
        if failure_reasons == ["transport_error"] and status_code is not None:
            raise InvalidEvidenceError("transport failure status is invalid")
        if failure_reasons == ["http_error"] and (
            not _valid_http_status(status_code) or 200 <= status_code < 300
        ):
            raise InvalidEvidenceError("HTTP failure status is invalid")

    return {
        "id": detail_id,
        "category": category,
        "http_ok": http_ok,
        "status_code": status_code,
        "latency_ms": _latency_ms(raw.get("latency_ms")),
        "passed": passed,
        "failure_reasons": list(failure_reasons),
        "planner_mode": planner_mode,
        "planner_mode_allowed": raw["planner_mode_allowed"],
        "planner_fallback": raw["planner_fallback"],
        "answer_fallback": raw["answer_fallback"],
        "fallback_reason_present": raw["fallback_reason_present"],
        "unexpected_non_answer": raw["unexpected_non_answer"],
        "answerability_pass": raw["answerability_pass"],
        "answerability_status": answerability_status,
        "quality": dict(quality),
        "model_usage": sanitized_usage,
        "failed_model_attempts": failed_model_attempts,
    }


def build_run_report(
    *,
    dataset_sha256: str,
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    planner_model_label: str,
    answer_model_label: str,
    cache_bypass: bool,
) -> dict[str, Any]:
    """Build a complete, sanitized 60-case report with exact Decimal totals."""

    dataset_hash = _require_nonblank_string(dataset_sha256, "dataset SHA-256")
    if dataset_hash != APPROVED_GOLDEN_SHA256:
        raise InvalidEvidenceError("dataset SHA-256 is not approved")
    planner_label = _explicit_model_label(planner_model_label, "planner model label")
    answer_label = _explicit_model_label(answer_model_label, "answer model label")
    if cache_bypass is not True:
        raise InvalidEvidenceError("release evidence requires cache bypass")

    approved_hash, approved_cases = load_golden_cases(APPROVED_GOLDEN_PATH)
    if approved_hash != dataset_hash or cases != approved_cases:
        raise InvalidEvidenceError("release cases do not match the approved dataset")
    ordered_ids = _validated_case_ids(cases)
    if len(results) != EXPECTED_CASE_COUNT:
        raise InvalidEvidenceError("release evidence is incomplete")
    sanitized_results = [
        _sanitize_case_detail(result, case) for result, case in zip(results, cases)
    ]
    result_ids = [result["id"] for result in sanitized_results]
    if result_ids != ordered_ids or len(set(result_ids)) != EXPECTED_CASE_COUNT:
        raise InvalidEvidenceError("result case ids do not match the golden order")

    _validate_observed_models(sanitized_results, planner_label, answer_label)
    role_totals = {"planner": Decimal("0"), "answer": Decimal("0")}
    for detail in sanitized_results:
        for usage in detail.get("model_usage", []):
            role_totals[usage["role"]] += _decimal(
                usage["total_cost_usd"], "usage total cost"
            )
    total = role_totals["planner"] + role_totals["answer"]
    latencies = [float(detail["latency_ms"]) for detail in sanitized_results]
    success_count = sum(bool(detail.get("http_ok")) for detail in sanitized_results)
    quality_passed = all(bool(detail.get("passed")) for detail in sanitized_results)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset_sha256": dataset_hash,
        "ordered_case_ids": ordered_ids,
        "case_count": EXPECTED_CASE_COUNT,
        "complete": True,
        "planner_model_label": planner_label,
        "answer_model_label": answer_label,
        "cache_bypass": True,
        "cost_totals_usd": {
            "planner": _fixed_usd(role_totals["planner"]),
            "answer": _fixed_usd(role_totals["answer"]),
            "total": _fixed_usd(total),
        },
        "summary": {
            "sample_count": EXPECTED_CASE_COUNT,
            "success_count": success_count,
            "failure_count": EXPECTED_CASE_COUNT - success_count,
            "passed_count": sum(
                bool(detail.get("passed")) for detail in sanitized_results
            ),
            "quality_passed": quality_passed,
            "latency_avg_ms": round(statistics.mean(latencies), 2),
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "planner_cost_usd": _fixed_usd(role_totals["planner"]),
            "answer_cost_usd": _fixed_usd(role_totals["answer"]),
            "total_cost_usd": _fixed_usd(total),
        },
        "details": sanitized_results,
    }


def _validate_report(report: object, name: str) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise InvalidEvidenceError(f"{name} report schema is invalid")
    if report.get("cache_bypass") is not True or report.get("complete") is not True:
        raise InvalidEvidenceError(f"{name} report is not complete cache-bypassed evidence")
    _explicit_model_label(report.get("planner_model_label"), f"{name} planner label")
    _explicit_model_label(report.get("answer_model_label"), f"{name} answer label")
    dataset_hash = _require_nonblank_string(
        report.get("dataset_sha256"), f"{name} dataset hash"
    )
    if dataset_hash != APPROVED_GOLDEN_SHA256:
        raise InvalidEvidenceError(f"{name} dataset hash is not approved")

    ordered_ids = report.get("ordered_case_ids")
    details = report.get("details")
    if not isinstance(ordered_ids, list) or not isinstance(details, list):
        raise InvalidEvidenceError(f"{name} case evidence is invalid")
    _, approved_cases = load_golden_cases(APPROVED_GOLDEN_PATH)
    approved_ids = [case["id"] for case in approved_cases]
    if ordered_ids != approved_ids or len(details) != EXPECTED_CASE_COUNT:
        raise InvalidEvidenceError(f"{name} report cases do not match the approved dataset")
    sanitized_details = [
        _sanitize_case_detail(detail, case)
        for detail, case in zip(details, approved_cases)
    ]
    if len(sanitized_details) != EXPECTED_CASE_COUNT:
        raise InvalidEvidenceError(f"{name} report details are incomplete")
    if [detail["id"] for detail in sanitized_details] != ordered_ids:
        raise InvalidEvidenceError(f"{name} detail order does not match case ids")
    if report.get("case_count") != EXPECTED_CASE_COUNT:
        raise InvalidEvidenceError(f"{name} case count is invalid")

    planner_label = str(report["planner_model_label"]).strip()
    answer_label = str(report["answer_model_label"]).strip()
    recomputed = {"planner": Decimal("0"), "answer": Decimal("0")}
    for detail in sanitized_details:
        for usage in detail["model_usage"]:
            expected_model = planner_label if usage["role"] == "planner" else answer_label
            if usage["model"] != expected_model:
                raise InvalidEvidenceError(f"{name} observed model does not match label")
            recomputed[usage["role"]] += _decimal(
                usage["total_cost_usd"], f"{name} usage total"
            )

    totals = report.get("cost_totals_usd")
    if not isinstance(totals, dict):
        raise InvalidEvidenceError(f"{name} cost totals are missing")
    if set(totals) != {"planner", "answer", "total"}:
        raise InvalidEvidenceError(f"{name} cost totals are invalid")
    expected_totals = {
        "planner": _fixed_usd(recomputed["planner"]),
        "answer": _fixed_usd(recomputed["answer"]),
        "total": _fixed_usd(recomputed["planner"] + recomputed["answer"]),
    }
    if totals != expected_totals:
        raise InvalidEvidenceError(f"{name} cost totals do not match usage evidence")

    latencies = [float(detail["latency_ms"]) for detail in sanitized_details]
    expected_summary = {
        "sample_count": EXPECTED_CASE_COUNT,
        "success_count": sum(detail["http_ok"] for detail in sanitized_details),
        "failure_count": sum(not detail["http_ok"] for detail in sanitized_details),
        "passed_count": sum(detail["passed"] for detail in sanitized_details),
        "quality_passed": all(detail["passed"] for detail in sanitized_details),
        "latency_avg_ms": round(statistics.mean(latencies), 2),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "planner_cost_usd": expected_totals["planner"],
        "answer_cost_usd": expected_totals["answer"],
        "total_cost_usd": expected_totals["total"],
    }
    summary = report.get("summary")
    integer_summary_fields = {
        "sample_count",
        "success_count",
        "failure_count",
        "passed_count",
    }
    latency_summary_fields = {
        "latency_avg_ms",
        "latency_p50_ms",
        "latency_p95_ms",
    }
    if (
        not isinstance(summary, dict)
        or any(
            isinstance(summary.get(field), bool)
            or not isinstance(summary.get(field), int)
            for field in integer_summary_fields
        )
        or not isinstance(summary.get("quality_passed"), bool)
        or any(
            isinstance(summary.get(field), bool)
            or not isinstance(summary.get(field), (int, float))
            for field in latency_summary_fields
        )
        or summary != expected_summary
    ):
        raise InvalidEvidenceError(f"{name} summary does not match case evidence")
    return {**report, "details": sanitized_details}


def compare_reports(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare complete reports and return measured cost/quality gate results."""

    baseline = _validate_report(baseline, "baseline")
    candidate = _validate_report(candidate, "candidate")
    if baseline["dataset_sha256"] != candidate["dataset_sha256"]:
        raise InvalidEvidenceError("dataset hash mismatch")
    if baseline["ordered_case_ids"] != candidate["ordered_case_ids"]:
        raise InvalidEvidenceError("ordered case ids mismatch")
    if baseline["answer_model_label"] != candidate["answer_model_label"]:
        raise InvalidEvidenceError("answer model label changed")

    baseline_planner = _decimal(
        baseline["cost_totals_usd"]["planner"], "baseline planner total"
    )
    candidate_planner = _decimal(
        candidate["cost_totals_usd"]["planner"], "candidate planner total"
    )
    baseline_total = _decimal(
        baseline["cost_totals_usd"]["total"], "baseline total"
    )
    candidate_total = _decimal(
        candidate["cost_totals_usd"]["total"], "candidate total"
    )

    baseline_planner_positive = baseline_planner > 0
    if baseline_planner_positive:
        planner_reduction = (
            (baseline_planner - candidate_planner) / baseline_planner * Decimal("100")
        )
    else:
        planner_reduction = Decimal("0")
    planner_reduction_passed = (
        baseline_planner_positive and planner_reduction >= Decimal("20.00")
    )
    total_non_increasing = candidate_total <= baseline_total

    no_new_failures = all(
        not (
            set(candidate_case["failure_reasons"])
            - set(baseline_case["failure_reasons"])
        )
        for baseline_case, candidate_case in zip(
            baseline["details"], candidate["details"], strict=True
        )
    )

    candidate_quality_passed = all(
        bool(detail.get("passed")) for detail in candidate["details"]
    )
    gate_passed = all(
        (
            planner_reduction_passed,
            total_non_increasing,
            candidate_quality_passed,
            no_new_failures,
        )
    )
    failure_reasons: list[str] = []
    if not baseline_planner_positive:
        failure_reasons.append("baseline_planner_cost_zero")
    elif not planner_reduction_passed:
        failure_reasons.append("planner_reduction_below_20_percent")
    if not total_non_increasing:
        failure_reasons.append("candidate_total_cost_increased")
    if not candidate_quality_passed:
        failure_reasons.append("candidate_quality_failed")
    if not no_new_failures:
        failure_reasons.append("candidate_introduced_new_failure")

    return {
        "gate_passed": gate_passed,
        "failure_reasons": failure_reasons,
        "baseline_planner_cost_positive": baseline_planner_positive,
        "planner_reduction_percent": format(
            planner_reduction.quantize(PERCENT_QUANTUM, rounding=ROUND_HALF_UP),
            ".2f",
        ),
        "planner_reduction_passed": planner_reduction_passed,
        "candidate_total_cost_non_increasing": total_non_increasing,
        "candidate_quality_passed": candidate_quality_passed,
        "no_new_failures": no_new_failures,
        "baseline_total_cost_usd": _fixed_usd(baseline_total),
        "candidate_total_cost_usd": _fixed_usd(candidate_total),
    }


def report_exit_code(
    report: dict[str, Any], comparison: dict[str, Any] | None = None
) -> int:
    if comparison is not None:
        return 0 if comparison.get("gate_passed") is True else 1
    summary = report.get("summary")
    return 0 if isinstance(summary, dict) and summary.get("quality_passed") is True else 1


def _read_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidEvidenceError("baseline report could not be loaded") from exc
    if not isinstance(payload, dict):
        raise InvalidEvidenceError("baseline report root must be an object")
    return payload


async def async_main(args: argparse.Namespace) -> int:
    dataset_sha256, all_cases = load_golden_cases(Path(args.samples))
    if args.limit <= 0:
        raise InvalidEvidenceError("limit must be positive")
    cases = all_cases[: args.limit]
    _validated_case_ids(cases)
    _explicit_model_label(args.planner_model_label, "planner model label")
    _explicit_model_label(args.answer_model_label, "answer model label")
    if args.cache_bypass is not True:
        raise InvalidEvidenceError("release evidence requires cache bypass")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise InvalidEvidenceError("timeout must be finite and positive")
    _require_nonblank_string(args.base_url, "base URL")
    internal_api_key = args.internal_api_key or os.getenv(args.internal_api_key_env, "")
    if not internal_api_key:
        raise InvalidEvidenceError("internal API key is required")

    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        results = await run_golden_cases(
            client=client,
            base_url=args.base_url,
            internal_api_key=internal_api_key,
            cases=cases,
            cache_bypass=args.cache_bypass,
        )
    report = build_run_report(
        dataset_sha256=dataset_sha256,
        cases=cases,
        results=results,
        planner_model_label=args.planner_model_label,
        answer_model_label=args.answer_model_label,
        cache_bypass=args.cache_bypass,
    )

    comparison = None
    if args.baseline_report:
        comparison = compare_reports(
            _read_report(Path(args.baseline_report)),
            report,
        )
        report["comparison"] = comparison

    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    else:
        print(output)
    return report_exit_code(report, comparison)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the chat model-routing golden cost and quality gate."
    )
    parser.add_argument("--samples", default=os.getenv("AI_MODEL_ROUTING_SAMPLES", ""))
    parser.add_argument(
        "--base-url",
        default=os.getenv("AI_MODEL_ROUTING_BASE_URL", "http://localhost:8001"),
    )
    parser.add_argument("--output", default=os.getenv("AI_MODEL_ROUTING_OUTPUT", ""))
    parser.add_argument("--baseline-report", default="")
    parser.add_argument("--internal-api-key", default=None)
    parser.add_argument("--internal-api-key-env", default="AI_INTERNAL_TOKEN")
    parser.add_argument(
        "--planner-model-label", default=os.getenv("CHAT_PLANNER_MODEL_NAME", "default")
    )
    parser.add_argument(
        "--answer-model-label", default=os.getenv("CHAT_ANSWER_MODEL_NAME", "default")
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=EXPECTED_CASE_COUNT)
    parser.add_argument("--cache-bypass", dest="cache_bypass", action="store_true")
    parser.add_argument("--no-cache-bypass", dest="cache_bypass", action="store_false")
    parser.set_defaults(cache_bypass=True)
    args = parser.parse_args()
    if not args.samples:
        parser.error("--samples or AI_MODEL_ROUTING_SAMPLES is required")
    return args


def main() -> int:
    try:
        return asyncio.run(async_main(parse_args()))
    except InvalidEvidenceError:
        print("invalid or incomplete model-routing evidence", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
