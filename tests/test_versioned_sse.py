"""Negotiation and legacy-adapter tests for versioned AI SSE responses."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from prometheus_client import REGISTRY

from app.streaming.http_errors import AiStreamHttpException
from app.streaming.versioned_sse import (
    negotiate_event_version,
    versioned_event_source,
    versioned_events,
)


async def _source(*events: dict[str, str]) -> AsyncIterator[dict[str, str]]:
    for event in events:
        yield event


async def _collect(
    *events: dict[str, str],
    endpoint: str = "chat",
    version: int = 2,
) -> list[dict[str, str]]:
    return [
        item
        async for item in versioned_events(
            _source(*events),
            endpoint=endpoint,
            version=version,
        )
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 1), ("1", 1), (" 2 ", 2)],
)
def test_negotiates_supported_versions(raw: str | None, expected: int) -> None:
    assert negotiate_event_version(raw, endpoint="chat") == expected


def test_unsupported_version_is_406() -> None:
    with pytest.raises(AiStreamHttpException) as raised:
        negotiate_event_version("3", endpoint="coach")

    assert raised.value.status_code == 406
    assert raised.value.error.model_dump(mode="json") == {
        "code": "AI_EVENT_VERSION_UNSUPPORTED",
        "message": "지원하지 않는 AI 이벤트 버전입니다.",
        "detail": None,
        "retryable": False,
        "retry_after_seconds": None,
        "supported_versions": ["1", "2"],
    }


@pytest.mark.asyncio
async def test_v1_validates_then_yields_original_event_identity() -> None:
    original = {"event": "message", "data": '{"delta":"첫"}'}

    emitted = await _collect(original, endpoint="chat", version=1)

    assert emitted == [original]
    assert emitted[0] is original


@pytest.mark.asyncio
async def test_v1_preserves_done_sentinel() -> None:
    emitted = await _collect(
        {"event": "done", "data": "[DONE]"},
        endpoint="chat",
        version=1,
    )

    assert emitted == [{"event": "done", "data": "[DONE]"}]


@pytest.mark.asyncio
async def test_v2_maps_chat_message_to_typed_envelope() -> None:
    emitted = await _collect(
        {"event": "message", "data": '{"delta":"첫"}'},
        endpoint="chat",
        version=2,
    )

    assert emitted == [
        {
            "event": "chat.message.delta",
            "data": '{"version":2,"type":"chat.message.delta","data":{"delta":"첫"}}',
        }
    ]


@pytest.mark.asyncio
async def test_v2_canonicalizes_chat_queue_fields() -> None:
    emitted = await _collect(
        {
            "event": "queue",
            "data": json.dumps(
                {
                    "state": "queued",
                    "queuePosition": 3,
                    "estimatedWaitTime": 8,
                    "rpmLimit": 60,
                }
            ),
        },
        endpoint="chat",
        version=2,
    )

    assert json.loads(emitted[0]["data"])["data"] == {
        "state": "queued",
        "queue_position": 3,
        "estimated_wait_time": 8,
        "rpm_limit": 60,
    }


@pytest.mark.asyncio
async def test_v2_canonicalizes_coach_meta_and_manual_data_fields() -> None:
    emitted = await _collect(
        {
            "event": "meta",
            "data": json.dumps(
                {
                    "request_mode": "manual_detail",
                    "analysisType": "game_review",
                    "llmSkipReason": "data_missing",
                    "manual_data_request": {
                        "scope": "coach_analysis",
                        "missingItems": [
                            {
                                "key": "record",
                                "label": "경기 기록",
                                "reason": "내부 데이터 누락",
                                "expected_format": "internal record",
                            }
                        ],
                        "operatorMessage": "운영자 입력 필요",
                        "blocking": True,
                        "code": "MANUAL_BASEBALL_DATA_REQUIRED",
                    },
                }
            ),
        },
        endpoint="coach",
        version=2,
    )

    data = json.loads(emitted[0]["data"])["data"]
    assert data["analysis_type"] == "game_review"
    assert data["llm_skip_reason"] == "data_missing"
    assert data["manual_data_request"]["missing_items"][0]["key"] == "record"
    assert data["manual_data_request"]["operator_message"] == "운영자 입력 필요"
    assert "analysisType" not in data
    assert "llmSkipReason" not in data


@pytest.mark.asyncio
async def test_v2_maps_coach_status_message() -> None:
    emitted = await _collect(
        {"event": "status", "data": '{"message":"분석 중"}'},
        endpoint="coach",
        version=2,
    )

    assert json.loads(emitted[0]["data"]) == {
        "version": 2,
        "type": "coach.status",
        "data": {"status": "분석 중"},
    }


@pytest.mark.asyncio
async def test_v2_replaces_done_sentinel() -> None:
    emitted = await _collect(
        {"event": "done", "data": "[DONE]"},
        endpoint="coach",
        version=2,
    )

    assert json.loads(emitted[0]["data"]) == {
        "version": 2,
        "type": "stream.done",
        "data": {"reason": "completed"},
    }


@pytest.mark.asyncio
async def test_v2_marks_done_as_error_after_legacy_error() -> None:
    emitted = await _collect(
        {
            "event": "error",
            "data": json.dumps(
                {
                    "code": "COACH_INTERNAL_ERROR",
                    "message": "분석에 실패했습니다.",
                }
            ),
        },
        {"event": "done", "data": "[DONE]"},
        endpoint="coach",
        version=2,
    )

    error = json.loads(emitted[0]["data"])
    done = json.loads(emitted[1]["data"])
    assert error["data"]["retryable"] is False
    assert done["data"]["reason"] == "error"


@pytest.mark.asyncio
async def test_v2_preserves_cancelled_terminal_reason_from_legacy_meta() -> None:
    emitted = await _collect(
        {
            "event": "meta",
            "data": json.dumps({"finish_reason": "cancelled", "cancelled": True}),
        },
        {"event": "done", "data": "[DONE]"},
        endpoint="chat",
        version=2,
    )

    done = json.loads(emitted[-1]["data"])
    assert done["data"]["reason"] == "cancelled"


def test_coach_public_error_payload_exposes_retryability() -> None:
    from app.routers.coach import _coach_public_error_payload

    assert _coach_public_error_payload(
        "COACH_INTERNAL_ERROR",
        retryable=False,
    )["retryable"] is False
    assert _coach_public_error_payload(
        "COACH_INTERNAL_ERROR",
        retryable=True,
    )["retryable"] is True


@pytest.mark.asyncio
async def test_invalid_event_emits_safe_error_and_terminal_pair() -> None:
    emitted = await _collect(
        {"event": "message", "data": '{"delta":""}'},
        endpoint="chat",
        version=2,
    )

    assert [item["event"] for item in emitted] == ["stream.error", "stream.done"]
    error = json.loads(emitted[0]["data"])
    done = json.loads(emitted[1]["data"])
    assert error["data"] == {
        "code": "AI_STREAM_CONTRACT_VIOLATION",
        "message": "AI 응답 형식이 올바르지 않습니다.",
        "detail": None,
        "retryable": False,
    }
    assert done["data"]["reason"] == "error"


def test_response_factory_exposes_resolved_version() -> None:
    response = versioned_event_source(
        _source({"event": "done", "data": "[DONE]"}),
        endpoint="chat",
        version=2,
        headers={"Cache-Control": "no-cache"},
        ping=15,
    )

    assert response.headers["X-AI-Event-Version"] == "2"
    assert response.headers["Cache-Control"] == "no-cache"


def test_stream_contract_metrics_use_only_bounded_labels() -> None:
    from app.observability.stream_metrics import AI_STREAM_EVENT_TOTAL

    AI_STREAM_EVENT_TOTAL.labels(
        endpoint="chat",
        version="2",
        event_type="chat.message.delta",
    ).inc()

    matching_samples = [
        sample
        for metric in REGISTRY.collect()
        for sample in metric.samples
        if sample.name == "ai_stream_event_total"
        and sample.labels
        == {
            "endpoint": "chat",
            "version": "2",
            "event_type": "chat.message.delta",
        }
    ]
    assert matching_samples
    assert matching_samples[0].value >= 1.0
