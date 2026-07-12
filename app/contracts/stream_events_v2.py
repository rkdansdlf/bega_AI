"""Pydantic models for the closed version 2 AI SSE event contract."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AiToolCall(_StrictModel):
    tool_name: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class AiDataSource(_StrictModel):
    title: str | None = None
    url: str | None = None
    content: str | None = None


class ManualBaseballDataMissingItem(_StrictModel):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    expected_format: str = Field(min_length=1)


class ManualBaseballDataRequest(_StrictModel):
    scope: str = Field(min_length=1)
    missing_items: list[ManualBaseballDataMissingItem]
    operator_message: str = Field(min_length=1)
    blocking: bool
    code: str | None = None


class CoachRiskItem(_StrictModel):
    area: str
    level: Literal[0, 1, 2]
    description: str
    inning_label: str | None = None
    inning_start: int | None = None
    inning_end: int | None = None
    impact: str | None = None
    impact_to: Literal["home", "away", "both"] | None = None


class CoachStructuredAnalysis(_StrictModel):
    summary: str | None = None
    verdict: str | None = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    risks: list[CoachRiskItem] = Field(default_factory=list)
    why_it_matters: list[str] = Field(default_factory=list)
    swing_factors: list[str] = Field(default_factory=list)
    watch_points: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


class CoachKeyMetric(_StrictModel):
    label: str
    value: str
    status: Literal["good", "warning", "danger"]
    trend: Literal["up", "down", "neutral"]
    is_critical: bool


class CoachStructuredResponse(_StrictModel):
    headline: str
    sentiment: Literal["positive", "negative", "neutral"]
    analysis_type: Literal["game_review", "game_preview"] | None = None
    key_metrics: list[CoachKeyMetric] = Field(default_factory=list)
    analysis: CoachStructuredAnalysis
    detailed_markdown: str
    coach_note: str


class ChatStatusData(_StrictModel):
    message: str = Field(min_length=1)


class ChatQueueData(_StrictModel):
    state: Literal["queued", "processing"]
    queue_position: int = Field(ge=0)
    estimated_wait_time: int = Field(ge=0)
    rpm_limit: int = Field(ge=0)


class MessageDeltaData(_StrictModel):
    delta: str = Field(min_length=1)


class ChatMetaData(_StrictModel):
    verified: bool | None = None
    cached: bool | None = None
    semantic_cached: bool | None = None
    intent: str | None = None
    strategy: str | None = None
    style: Literal["markdown", "json", "compact"] | None = None
    planner_mode: str | None = None
    planner_cache_hit: bool | None = None
    tool_execution_mode: str | None = None
    fallback_triggered: bool | None = None
    fallback_answer_used: bool | None = None
    fallback_reason: str | None = None
    grounding_mode: str | None = None
    source_tier: str | None = None
    as_of_date: str | None = None
    finish_reason: str | None = None
    cancelled: bool | None = None
    cache_key_prefix: str | None = None
    cache_similarity: float | None = None
    error: str | None = None
    tool_calls: list[AiToolCall] = Field(default_factory=list)
    tool_results: list[JsonValue] = Field(default_factory=list)
    data_sources: list[AiDataSource] = Field(default_factory=list)
    answer_sources: list[JsonValue] = Field(default_factory=list)
    visualizations: list[JsonValue] = Field(default_factory=list)
    perf: dict[str, JsonValue] = Field(default_factory=dict)


class CoachStatusData(_StrictModel):
    status: str = Field(min_length=1)


class CoachPreviewChunkData(_StrictModel):
    text: str
    attempt: int = Field(ge=1)


class CoachPreviewResetData(_StrictModel):
    attempt: int = Field(ge=1)


class CoachMetaData(_StrictModel):
    structured_response: CoachStructuredResponse | None = None
    tool_calls: list[AiToolCall] = Field(default_factory=list)
    verified: bool | None = None
    data_sources: list[AiDataSource] = Field(default_factory=list)
    resolved_focus: list[str] = Field(default_factory=list)
    request_mode: Literal["auto_brief", "manual_detail"] | None = None
    analysis_type: Literal["game_review", "game_preview"] | None = None
    focus_signature: str | None = None
    question_signature: str | None = None
    cache_key_version: str | None = None
    cache_state: str | None = None
    validation_status: str | None = None
    in_progress: bool | None = None
    cached: bool | None = None
    llm_skip_reason: str | None = None
    focus_section_missing: bool | None = None
    missing_focus_sections: list[str] = Field(default_factory=list)
    generation_mode: Literal[
        "deterministic_auto",
        "deterministic_review",
        "deterministic_preview",
        "llm_manual",
        "evidence_fallback",
    ] | None = None
    data_quality: Literal["grounded", "partial", "insufficient"] | None = None
    used_evidence: list[str] = Field(default_factory=list)
    grounding_warnings: list[str] = Field(default_factory=list)
    grounding_reasons: list[str] = Field(default_factory=list)
    supported_fact_count: int | None = Field(default=None, ge=0)
    game_status_bucket: str | None = None
    manual_data_request: ManualBaseballDataRequest | None = None
    win_probability_home: float | None = Field(default=None, ge=0.0, le=1.0)


class StreamErrorData(_StrictModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    detail: str | None = None
    retryable: bool


class StreamDoneData(_StrictModel):
    reason: Literal["completed", "error", "cancelled"]


class ChatStatusEvent(_StrictModel):
    version: Literal[2]
    type: Literal["chat.status"]
    data: ChatStatusData


class ChatQueueEvent(_StrictModel):
    version: Literal[2]
    type: Literal["chat.queue"]
    data: ChatQueueData


class ChatMessageDeltaEvent(_StrictModel):
    version: Literal[2]
    type: Literal["chat.message.delta"]
    data: MessageDeltaData


class ChatMetaEvent(_StrictModel):
    version: Literal[2]
    type: Literal["chat.meta"]
    data: ChatMetaData


class CoachStatusEvent(_StrictModel):
    version: Literal[2]
    type: Literal["coach.status"]
    data: CoachStatusData


class CoachPreviewChunkEvent(_StrictModel):
    version: Literal[2]
    type: Literal["coach.preview.chunk"]
    data: CoachPreviewChunkData


class CoachPreviewResetEvent(_StrictModel):
    version: Literal[2]
    type: Literal["coach.preview.reset"]
    data: CoachPreviewResetData


class CoachMessageDeltaEvent(_StrictModel):
    version: Literal[2]
    type: Literal["coach.message.delta"]
    data: MessageDeltaData


class CoachMetaEvent(_StrictModel):
    version: Literal[2]
    type: Literal["coach.meta"]
    data: CoachMetaData


class StreamErrorEvent(_StrictModel):
    version: Literal[2]
    type: Literal["stream.error"]
    data: StreamErrorData


class StreamDoneEvent(_StrictModel):
    version: Literal[2]
    type: Literal["stream.done"]
    data: StreamDoneData


AiStreamV2Event = Annotated[
    ChatStatusEvent
    | ChatQueueEvent
    | ChatMessageDeltaEvent
    | ChatMetaEvent
    | CoachStatusEvent
    | CoachPreviewChunkEvent
    | CoachPreviewResetEvent
    | CoachMessageDeltaEvent
    | CoachMetaEvent
    | StreamErrorEvent
    | StreamDoneEvent,
    Field(discriminator="type"),
]

APPROVED_EVENT_TYPES = frozenset(
    {
        "chat.status",
        "chat.queue",
        "chat.message.delta",
        "chat.meta",
        "coach.status",
        "coach.preview.chunk",
        "coach.preview.reset",
        "coach.message.delta",
        "coach.meta",
        "stream.error",
        "stream.done",
    }
)

_EVENT_ADAPTER = TypeAdapter(AiStreamV2Event)


def parse_v2_event(value: object) -> AiStreamV2Event:
    """Validate an object against the closed v2 event union."""

    return _EVENT_ADAPTER.validate_python(value)


def event_schema() -> dict[str, JsonValue]:
    """Return the JSON Schema generated from the runtime event union."""

    return _EVENT_ADAPTER.json_schema(ref_template="#/components/schemas/{model}")
