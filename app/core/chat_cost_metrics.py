"""Estimated chat token and cost metrics.

Provider APIs do not consistently expose token usage on the current streaming
paths, so these counters intentionally use a deterministic character-based
estimate over user-visible question/history/answer text. System prompts, tool
payloads, and planner traffic are excluded. Cost values stay zero unless
operators configure per-1M token rates.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

from app.observability.metrics import (
    AI_CHAT_COST_ESTIMATE_BY_TYPE_USD_TOTAL,
    AI_CHAT_COST_ESTIMATE_USD_TOTAL,
    AI_CHAT_TOKEN_ESTIMATE_BY_TYPE_TOTAL,
    AI_CHAT_TOKEN_ESTIMATE_TOTAL,
)

_CHARS_PER_TOKEN = 3.5
_NARRATIVE_MARKERS = ("서사", "묶어서", "한 그룹", "한 묶음", "연결축")
_REGULATION_MARKERS = (
    "규정",
    "보상선수",
    "엔트리",
    "로스터",
    "부상자 명단",
    "육성선수",
    "군보류",
    "임의해지",
)
_TEAM_MARKERS = (
    "LG",
    "KT",
    "NC",
    "SSG",
    "KIA",
    "롯데",
    "삼성",
    "두산",
    "한화",
    "키움",
    "팀",
    "불펜",
    "선발진",
    "타선",
)
_PLAYER_ANALYSIS_MARKERS = (
    "선수",
    "타율",
    "출루율",
    "장타율",
    "홈런",
    "안타",
    "ERA",
    "기록",
    "같은 포지션",
)


def classify_chat_question_type(question: str) -> str:
    """Map user questions to bounded metric labels."""
    value = str(question or "")
    if any(marker in value for marker in _NARRATIVE_MARKERS):
        return "multi_player_narrative"
    if any(marker in value for marker in _REGULATION_MARKERS):
        return "regulation"
    if any(marker in value for marker in _TEAM_MARKERS):
        return "team_analysis"
    if any(marker in value for marker in _PLAYER_ANALYSIS_MARKERS):
        return "player_analysis"
    return "general"


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / _CHARS_PER_TOKEN)))


def estimate_history_tokens(history: Optional[Sequence[Dict[str, str]]]) -> int:
    if not history:
        return 0
    total = 0
    for item in history:
        total += estimate_text_tokens(str(item.get("role", "")))
        total += estimate_text_tokens(str(item.get("content", "")))
    return total


def resolve_chat_model_name(settings: Any) -> str:
    return str(
        getattr(settings, "chatbot_model_name", None)
        or getattr(settings, "openrouter_model", None)
        or getattr(settings, "gemini_model", None)
        or "unknown"
    )


def record_chat_token_estimate(
    settings: Any,
    *,
    route: str,
    cache_state: str,
    question: str,
    answer: str,
    history: Optional[Sequence[Dict[str, str]]] = None,
    model_name: Optional[str] = None,
    question_type: Optional[str] = None,
    planner_mode: Optional[str] = None,
) -> None:
    input_tokens = estimate_text_tokens(question) + estimate_history_tokens(history)
    output_tokens = estimate_text_tokens(answer)
    provider = str(getattr(settings, "llm_provider", None) or "unknown")
    model = str(model_name or resolve_chat_model_name(settings))
    resolved_question_type = str(question_type or classify_chat_question_type(question))
    resolved_planner_mode = str(planner_mode or "unknown")

    try:
        AI_CHAT_TOKEN_ESTIMATE_TOTAL.labels(
            route=route, token_type="input", cache_state=cache_state
        ).inc(input_tokens)
        AI_CHAT_TOKEN_ESTIMATE_TOTAL.labels(
            route=route, token_type="output", cache_state=cache_state
        ).inc(output_tokens)
    except Exception:  # noqa: BLE001
        return

    try:
        AI_CHAT_TOKEN_ESTIMATE_BY_TYPE_TOTAL.labels(
            route=route,
            question_type=resolved_question_type,
            planner_mode=resolved_planner_mode,
            token_type="input",
            cache_state=cache_state,
        ).inc(input_tokens)
        AI_CHAT_TOKEN_ESTIMATE_BY_TYPE_TOTAL.labels(
            route=route,
            question_type=resolved_question_type,
            planner_mode=resolved_planner_mode,
            token_type="output",
            cache_state=cache_state,
        ).inc(output_tokens)
    except Exception:  # noqa: BLE001
        pass

    if cache_state != "generated":
        return

    input_rate = max(
        0.0, float(getattr(settings, "chat_cost_input_usd_per_1m_tokens", 0.0) or 0.0)
    )
    output_rate = max(
        0.0, float(getattr(settings, "chat_cost_output_usd_per_1m_tokens", 0.0) or 0.0)
    )
    cost_usd = (input_tokens * input_rate / 1_000_000) + (
        output_tokens * output_rate / 1_000_000
    )
    if cost_usd <= 0:
        return

    try:
        AI_CHAT_COST_ESTIMATE_USD_TOTAL.labels(
            route=route, provider=provider, model=model
        ).inc(cost_usd)
    except Exception:  # noqa: BLE001
        pass

    try:
        AI_CHAT_COST_ESTIMATE_BY_TYPE_USD_TOTAL.labels(
            route=route,
            question_type=resolved_question_type,
            planner_mode=resolved_planner_mode,
            provider=provider,
            model=model,
        ).inc(cost_usd)
    except Exception:  # noqa: BLE001
        return
