"""Typed chat and coach request bodies exported with the stream contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

MAX_COACH_FOCUS_ITEMS = 6
MAX_COACH_QUESTION_OVERRIDE_LENGTH = 2000


class ChatHistoryMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant"]
    content: str


class ChatStreamRequest(BaseModel):
    """POST `/ai/chat/stream` body with the current permissive extra policy."""

    model_config = ConfigDict(extra="allow")

    question: str
    filters: dict[str, JsonValue] | None = None
    history: list[ChatHistoryMessage] | str | None = None
    cache_bypass: bool = False
    style: Literal["markdown", "json", "compact"] | None = None


class CoachAnalyzeRequest(BaseModel):
    """POST `/ai/coach/analyze` body, including existing compatibility aliases."""

    team_id: str | None = None
    home_team_id: str | None = None
    away_team_id: str | None = None
    league_context: dict[str, JsonValue] | None = None
    focus: list[str] = Field(default_factory=list)
    game_id: str | None = None
    request_mode: Literal["auto_brief", "manual_detail"] = "manual_detail"
    analysis_type: Literal["game_review", "game_preview"] | None = None
    question_override: str | None = None
    starter_signature: str | None = None
    lineup_signature: str | None = None
    expected_cache_key: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, values: Any) -> Any:
        """Preserve the route's legacy aliases and input normalization."""

        if not isinstance(values, dict):
            return values

        normalized = dict(values)
        if "analysisType" in normalized and "analysis_type" not in normalized:
            normalized["analysis_type"] = normalized["analysisType"]

        analysis_type = normalized.get("analysis_type")
        if isinstance(analysis_type, str):
            normalized_type = analysis_type.strip().lower()
            normalized["analysis_type"] = normalized_type

        focus = normalized.get("focus")
        if isinstance(focus, list) and len(focus) > MAX_COACH_FOCUS_ITEMS:
            raise ValueError(
                f"focus 항목은 최대 {MAX_COACH_FOCUS_ITEMS}개까지 허용됩니다."
            )

        question_override = normalized.get("question_override")
        if isinstance(question_override, str):
            trimmed_question = question_override.strip()
            if not trimmed_question:
                normalized["question_override"] = None
            elif len(trimmed_question) > MAX_COACH_QUESTION_OVERRIDE_LENGTH:
                raise ValueError(
                    "question_override가 너무 깁니다. "
                    f"최대 {MAX_COACH_QUESTION_OVERRIDE_LENGTH}자까지 허용됩니다."
                )
            else:
                normalized["question_override"] = trimmed_question

        if (
            normalized.get("request_mode") == "auto_brief"
            and normalized.get("question_override") is not None
        ):
            raise ValueError(
                "auto_brief 모드에서는 question_override를 사용할 수 없습니다."
            )

        if not normalized.get("home_team_id") and normalized.get("team_id"):
            normalized["home_team_id"] = normalized["team_id"]

        for signature_key in (
            "starter_signature",
            "lineup_signature",
            "expected_cache_key",
        ):
            signature_value = normalized.get(signature_key)
            if isinstance(signature_value, str):
                trimmed_signature = signature_value.strip()
                normalized[signature_key] = trimmed_signature or None

        return normalized
