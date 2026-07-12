"""SSE version negotiation and legacy-to-v2 event serialization."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any, Literal, cast

from fastapi import HTTPException
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse

from app.contracts.stream_events_v2 import (
    AiDataSource,
    ChatMetaData,
    CoachMetaData,
    parse_v2_event,
)
from app.observability.stream_metrics import (
    AI_STREAM_CONTRACT_FAILURE_TOTAL,
    AI_STREAM_EVENT_TOTAL,
    AI_STREAM_REQUEST_TOTAL,
    AI_STREAM_UNSUPPORTED_VERSION_TOTAL,
)

logger = logging.getLogger(__name__)

EventVersion = Literal[1, 2]
StreamEndpoint = Literal["chat", "coach"]

EVENT_VERSION_HEADER = "X-AI-Event-Version"
SUPPORTED_EVENT_VERSIONS = ("1", "2")


def negotiate_event_version(
    raw_value: str | None,
    *,
    endpoint: StreamEndpoint,
) -> EventVersion:
    """Resolve an optional version header without silently downgrading."""

    normalized = raw_value.strip() if isinstance(raw_value, str) else "1"
    if not normalized:
        normalized = "1"
    if normalized == "1":
        return 1
    if normalized == "2":
        return 2

    AI_STREAM_UNSUPPORTED_VERSION_TOTAL.labels(endpoint=endpoint).inc()
    raise HTTPException(
        status_code=406,
        detail={
            "code": "AI_EVENT_VERSION_UNSUPPORTED",
            "supported_versions": list(SUPPORTED_EVENT_VERSIONS),
        },
    )


def _json_payload(event: Mapping[str, str]) -> dict[str, Any]:
    raw_data = event.get("data", "")
    if raw_data == "[DONE]":
        return {}
    parsed = json.loads(raw_data)
    if not isinstance(parsed, dict):
        raise ValueError("SSE data must be a JSON object")
    return parsed


def _canonical_tool_calls(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    canonical: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        tool_name = item.get("tool_name") or item.get("toolName")
        if not isinstance(tool_name, str) or not tool_name:
            continue
        parameters = item.get("parameters")
        canonical.append(
            {
                "tool_name": tool_name,
                "parameters": dict(parameters) if isinstance(parameters, Mapping) else {},
            }
        )
    return canonical


def _canonical_data_sources(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    allowed = AiDataSource.model_fields
    return [
        {key: field_value for key, field_value in item.items() if key in allowed}
        for item in value
        if isinstance(item, Mapping)
    ]


def _canonical_manual_data_request(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    raw_items = value.get("missing_items", value.get("missingItems", []))
    missing_items = []
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            missing_items.append(
                {
                    "key": item.get("key"),
                    "label": item.get("label"),
                    "reason": item.get("reason"),
                    "expected_format": item.get("expected_format"),
                }
            )
    return {
        "scope": value.get("scope"),
        "missing_items": missing_items,
        "operator_message": value.get(
            "operator_message", value.get("operatorMessage")
        ),
        "blocking": value.get("blocking"),
        "code": value.get("code"),
    }


def _canonical_structured_response(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    canonical = dict(value)
    legacy_analysis_type = canonical.pop("analysisType", None)
    if "analysis_type" not in canonical and legacy_analysis_type is not None:
        canonical["analysis_type"] = legacy_analysis_type
    return canonical


def _canonical_chat_meta(payload: Mapping[str, Any]) -> dict[str, Any]:
    canonical = {
        key: value
        for key, value in payload.items()
        if key in ChatMetaData.model_fields
    }
    if "tool_calls" in payload:
        canonical["tool_calls"] = _canonical_tool_calls(payload["tool_calls"])
    if "data_sources" in payload:
        canonical["data_sources"] = _canonical_data_sources(
            payload["data_sources"]
        )
    return canonical


def _canonical_coach_meta(payload: Mapping[str, Any]) -> dict[str, Any]:
    canonical = {
        key: value
        for key, value in payload.items()
        if key in CoachMetaData.model_fields
    }
    if "analysis_type" not in canonical and "analysisType" in payload:
        canonical["analysis_type"] = payload["analysisType"]
    if "llm_skip_reason" not in canonical and "llmSkipReason" in payload:
        canonical["llm_skip_reason"] = payload["llmSkipReason"]
    if "tool_calls" in payload:
        canonical["tool_calls"] = _canonical_tool_calls(payload["tool_calls"])
    if "data_sources" in payload:
        canonical["data_sources"] = _canonical_data_sources(
            payload["data_sources"]
        )
    if "manual_data_request" in payload:
        canonical["manual_data_request"] = _canonical_manual_data_request(
            payload["manual_data_request"]
        )
    if "structured_response" in payload:
        canonical["structured_response"] = _canonical_structured_response(
            payload["structured_response"]
        )
    return canonical


def _event_object(
    event: Mapping[str, str],
    *,
    endpoint: StreamEndpoint,
) -> dict[str, Any]:
    legacy_type = event.get("event", "message")
    payload = _json_payload(event)

    if legacy_type == "done":
        return {"version": 2, "type": "stream.done", "data": {"reason": "completed"}}
    if legacy_type == "error":
        raw_code = payload.get("code") or payload.get("message")
        raw_message = payload.get("message")
        raw_detail = payload.get("detail")
        public_message = raw_detail or raw_message or "AI 응답 처리 중 오류가 발생했습니다."
        return {
            "version": 2,
            "type": "stream.error",
            "data": {
                "code": str(raw_code or "AI_STREAM_ERROR"),
                "message": str(public_message),
                "detail": str(raw_detail) if raw_detail is not None else None,
                "retryable": bool(payload.get("retryable", True)),
            },
        }

    if endpoint == "chat":
        if legacy_type == "status":
            data = {"message": payload.get("message")}
            event_type = "chat.status"
        elif legacy_type == "queue":
            data = {
                "state": payload.get("state"),
                "queue_position": payload.get(
                    "queue_position", payload.get("queuePosition")
                ),
                "estimated_wait_time": payload.get(
                    "estimated_wait_time", payload.get("estimatedWaitTime")
                ),
                "rpm_limit": payload.get("rpm_limit", payload.get("rpmLimit")),
            }
            event_type = "chat.queue"
        elif legacy_type == "message":
            data = {"delta": payload.get("delta")}
            event_type = "chat.message.delta"
        elif legacy_type == "meta":
            data = _canonical_chat_meta(payload)
            event_type = "chat.meta"
        else:
            raise ValueError(f"Unsupported chat event: {legacy_type}")
    else:
        if legacy_type == "status":
            data = {"status": payload.get("status") or payload.get("message")}
            event_type = "coach.status"
        elif legacy_type == "preview_chunk":
            data = {"text": payload.get("text"), "attempt": payload.get("attempt", 1)}
            event_type = "coach.preview.chunk"
        elif legacy_type == "preview_reset":
            data = {"attempt": payload.get("attempt", 1)}
            event_type = "coach.preview.reset"
        elif legacy_type == "message":
            data = {"delta": payload.get("delta")}
            event_type = "coach.message.delta"
        elif legacy_type == "meta":
            data = _canonical_coach_meta(payload)
            event_type = "coach.meta"
        else:
            raise ValueError(f"Unsupported coach event: {legacy_type}")

    return {"version": 2, "type": event_type, "data": data}


def _serialize_v2(event_value: dict[str, Any]) -> dict[str, str]:
    event = parse_v2_event(event_value)
    payload = event.model_dump(
        mode="json",
        exclude_none=False,
        exclude_unset=True,
    )
    return {
        "event": event.type,
        "data": json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def _safe_terminal_events(version: EventVersion) -> list[dict[str, str]]:
    if version == 1:
        return [
            {
                "event": "error",
                "data": json.dumps(
                    {
                        "code": "AI_STREAM_CONTRACT_VIOLATION",
                        "message": "AI 응답 형식이 올바르지 않습니다.",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
            {"event": "done", "data": "[DONE]"},
        ]

    return [
        _serialize_v2(
            {
                "version": 2,
                "type": "stream.error",
                "data": {
                    "code": "AI_STREAM_CONTRACT_VIOLATION",
                    "message": "AI 응답 형식이 올바르지 않습니다.",
                    "detail": None,
                    "retryable": False,
                },
            }
        ),
        _serialize_v2(
            {
                "version": 2,
                "type": "stream.done",
                "data": {"reason": "error"},
            }
        ),
    ]


async def versioned_events(
    events: AsyncIterable[dict[str, str]],
    *,
    endpoint: str,
    version: int,
) -> AsyncIterator[dict[str, str]]:
    """Validate legacy events and emit the negotiated wire representation."""

    resolved_endpoint = cast(StreamEndpoint, endpoint)
    resolved_version = cast(EventVersion, version)
    if resolved_endpoint not in {"chat", "coach"}:
        raise ValueError(f"Unsupported stream endpoint: {endpoint}")
    if resolved_version not in {1, 2}:
        raise ValueError(f"Unsupported stream version: {version}")

    async for raw_event in events:
        try:
            event_value = _event_object(raw_event, endpoint=resolved_endpoint)
            validated = _serialize_v2(event_value)
        except Exception as exc:  # noqa: BLE001
            AI_STREAM_CONTRACT_FAILURE_TOTAL.labels(
                endpoint=resolved_endpoint,
                version=str(resolved_version),
            ).inc()
            validation_fields = (
                [
                    {"location": error["loc"], "type": error["type"]}
                    for error in exc.errors(include_url=False, include_input=False)
                ]
                if isinstance(exc, ValidationError)
                else []
            )
            logger.warning(
                "AI stream contract validation failed endpoint=%s version=%s "
                "event_type=%s category=%s fields=%s",
                resolved_endpoint,
                resolved_version,
                raw_event.get("event", "message"),
                exc.__class__.__name__,
                validation_fields,
            )
            for safe_event in _safe_terminal_events(resolved_version):
                AI_STREAM_EVENT_TOTAL.labels(
                    endpoint=resolved_endpoint,
                    version=str(resolved_version),
                    event_type=safe_event["event"],
                ).inc()
                yield safe_event
            return

        emitted = raw_event if resolved_version == 1 else validated
        AI_STREAM_EVENT_TOTAL.labels(
            endpoint=resolved_endpoint,
            version=str(resolved_version),
            event_type=emitted.get("event", "message"),
        ).inc()
        yield emitted


def versioned_event_source(
    events: AsyncIterable[dict[str, str]],
    *,
    endpoint: str,
    version: int,
    headers: Mapping[str, str] | None = None,
    ping: int | None = None,
) -> EventSourceResponse:
    """Create an EventSourceResponse with negotiated serialization and headers."""

    resolved_headers = dict(headers or {})
    resolved_headers[EVENT_VERSION_HEADER] = str(version)
    AI_STREAM_REQUEST_TOTAL.labels(endpoint=endpoint, version=str(version)).inc()
    return EventSourceResponse(
        versioned_events(events, endpoint=endpoint, version=version),
        headers=resolved_headers,
        ping=ping,
    )
