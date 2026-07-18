"""Contract tests for the version 2 AI SSE event envelope."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from app.contracts.stream_events_v2 import (
    APPROVED_EVENT_TYPES,
    AiStreamHttpError,
    parse_v2_event,
)


def _approved_examples() -> dict[str, dict[str, Any]]:
    return {
        "chat.status": {
            "version": 2,
            "type": "chat.status",
            "data": {"message": "답변을 준비했습니다."},
        },
        "chat.queue": {
            "version": 2,
            "type": "chat.queue",
            "data": {
                "state": "queued",
                "queue_position": 2,
                "estimated_wait_time": 4,
                "rpm_limit": 60,
            },
        },
        "chat.message.delta": {
            "version": 2,
            "type": "chat.message.delta",
            "data": {"delta": "첫 문장"},
        },
        "chat.meta": {
            "version": 2,
            "type": "chat.meta",
            "data": {
                "verified": True,
                "cached": False,
                "style": "markdown",
                "tool_calls": [
                    {"tool_name": "document_query", "parameters": {"team": "HT"}}
                ],
                "data_sources": [{"title": "internal-record"}],
                "answer_sources": ["internal-record"],
                "perf": {"total_ms": 12.5, "tool_count": 1},
            },
        },
        "coach.status": {
            "version": 2,
            "type": "coach.status",
            "data": {"status": "양 팀 전력 분석 중..."},
        },
        "coach.preview.chunk": {
            "version": 2,
            "type": "coach.preview.chunk",
            "data": {"text": "분석 미리보기", "attempt": 1},
        },
        "coach.preview.reset": {
            "version": 2,
            "type": "coach.preview.reset",
            "data": {"attempt": 2},
        },
        "coach.message.delta": {
            "version": 2,
            "type": "coach.message.delta",
            "data": {"delta": "상세 분석"},
        },
        "coach.meta": {
            "version": 2,
            "type": "coach.meta",
            "data": {
                "request_mode": "manual_detail",
                "analysis_type": "game_review",
                "generation_mode": "evidence_fallback",
                "data_quality": "insufficient",
                "manual_data_request": {
                    "scope": "coach_analysis",
                    "missing_items": [
                        {
                            "key": "game_record",
                            "label": "경기 기록",
                            "reason": "내부 데이터 누락",
                            "expected_format": "internal game record",
                        }
                    ],
                    "operator_message": "운영자 데이터가 필요합니다.",
                    "blocking": True,
                    "code": "MANUAL_BASEBALL_DATA_REQUIRED",
                },
                "win_probability_home": 0.55,
            },
        },
        "stream.error": {
            "version": 2,
            "type": "stream.error",
            "data": {
                "code": "COACH_ANALYSIS_FAILED",
                "message": "분석 중 오류가 발생했습니다.",
                "detail": None,
                "retryable": True,
            },
        },
        "stream.done": {
            "version": 2,
            "type": "stream.done",
            "data": {"reason": "completed"},
        },
    }


def test_chat_delta_has_exact_v2_envelope() -> None:
    event = parse_v2_event(_approved_examples()["chat.message.delta"])

    assert event.model_dump(exclude_none=True) == {
        "version": 2,
        "type": "chat.message.delta",
        "data": {"delta": "첫 문장"},
    }


def test_ai_stream_http_error_has_exact_canonical_shape() -> None:
    error = AiStreamHttpError(
        code="AI_EVENT_VERSION_UNSUPPORTED",
        message="지원하지 않는 AI 이벤트 버전입니다.",
        retryable=False,
        supported_versions=["1", "2"],
    )

    assert error.model_dump(mode="json") == {
        "code": "AI_EVENT_VERSION_UNSUPPORTED",
        "message": "지원하지 않는 AI 이벤트 버전입니다.",
        "detail": None,
        "retryable": False,
        "retry_after_seconds": None,
        "supported_versions": ["1", "2"],
    }


def test_ai_stream_http_error_rejects_invalid_retry_and_versions() -> None:
    with pytest.raises(ValidationError):
        AiStreamHttpError(
            code="AI_UPSTREAM_RATE_LIMITED",
            message="요청 한도를 초과했습니다.",
            retryable=True,
            retry_after_seconds=-1,
        )
    with pytest.raises(ValidationError):
        AiStreamHttpError(
            code="AI_EVENT_VERSION_UNSUPPORTED",
            message="지원하지 않는 버전입니다.",
            retryable=False,
            supported_versions=["3"],
        )


def test_ai_stream_http_error_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AiStreamHttpError(
            code="AI_ERROR",
            message="오류",
            retryable=False,
            success=False,
        )


@pytest.mark.parametrize("event_type", sorted(_approved_examples()))
def test_all_approved_event_types_parse(event_type: str) -> None:
    event = parse_v2_event(_approved_examples()[event_type])

    assert event.type == event_type


def test_approved_type_constant_matches_contract_examples() -> None:
    assert APPROVED_EVENT_TYPES == frozenset(_approved_examples())


def test_queue_contract_uses_snake_case_fields() -> None:
    payload = _approved_examples()["chat.queue"]
    payload["data"]["queuePosition"] = payload["data"].pop("queue_position")

    with pytest.raises(ValidationError):
        parse_v2_event(payload)


def test_v2_rejects_camel_case_coach_alias() -> None:
    payload = _approved_examples()["coach.meta"]
    payload["data"]["analysisType"] = payload["data"].pop("analysis_type")

    with pytest.raises(ValidationError):
        parse_v2_event(payload)


def test_v2_preserves_manual_baseball_data_contract() -> None:
    event = parse_v2_event(_approved_examples()["coach.meta"])

    assert event.data.manual_data_request is not None
    assert event.data.manual_data_request.code == "MANUAL_BASEBALL_DATA_REQUIRED"
    assert event.data.manual_data_request.missing_items[0].key == "game_record"


@pytest.mark.parametrize("reason", ["completed", "error", "cancelled"])
def test_stream_done_accepts_closed_reason_union(reason: str) -> None:
    payload = deepcopy(_approved_examples()["stream.done"])
    payload["data"]["reason"] = reason

    assert parse_v2_event(payload).data.reason == reason


def test_stream_done_rejects_unknown_reason() -> None:
    payload = _approved_examples()["stream.done"]
    payload["data"]["reason"] = "unknown"

    with pytest.raises(ValidationError):
        parse_v2_event(payload)


def test_v2_rejects_unknown_top_level_field() -> None:
    payload = _approved_examples()["stream.done"]
    payload["legacy"] = True

    with pytest.raises(ValidationError):
        parse_v2_event(payload)


def test_delta_must_not_be_empty() -> None:
    payload = _approved_examples()["chat.message.delta"]
    payload["data"]["delta"] = ""

    with pytest.raises(ValidationError):
        parse_v2_event(payload)
