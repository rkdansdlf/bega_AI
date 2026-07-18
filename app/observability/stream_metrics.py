"""Low-cardinality metrics for negotiated AI stream contracts."""

from __future__ import annotations

try:
    from prometheus_client import Counter
except ImportError:  # pragma: no cover - matches the service metrics fallback
    from app.observability.metrics import Counter

AI_STREAM_REQUEST_TOTAL = Counter(
    "ai_stream_request_total",
    "AI stream requests by endpoint and negotiated event version.",
    ["endpoint", "version"],
)

AI_STREAM_EVENT_TOTAL = Counter(
    "ai_stream_event_total",
    "AI stream events emitted by endpoint, version, and event type.",
    ["endpoint", "version", "event_type"],
)

AI_STREAM_CONTRACT_FAILURE_TOTAL = Counter(
    "ai_stream_contract_failure_total",
    "AI stream event contract validation failures.",
    ["endpoint", "version"],
)

AI_STREAM_UNSUPPORTED_VERSION_TOTAL = Counter(
    "ai_stream_unsupported_version_total",
    "Unsupported AI stream event-version requests.",
    ["endpoint"],
)
