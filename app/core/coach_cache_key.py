"""
Coach 분석 캐시 키 생성/정규화 유틸.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Sequence

FOCUS_ORDER: tuple[str, ...] = (
    "recent_form",
    "bullpen",
    "starter",
    "matchup",
    "batting",
)
FOCUS_SET = set(FOCUS_ORDER)
FOCUS_PRIORITY = {name: idx for idx, name in enumerate(FOCUS_ORDER)}


def normalize_focus(focus: Optional[Sequence[str]]) -> list[str]:
    """
    focus 입력을 허용값만 남기고 canonical order로 정렬합니다.
    """
    if not focus:
        return []

    deduped: list[str] = []
    seen = set()
    for item in focus:
        normalized = str(item or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        if normalized in FOCUS_SET:
            seen.add(normalized)
            deduped.append(normalized)

    return sorted(deduped, key=lambda x: FOCUS_PRIORITY[x])


def normalize_question_override(text: Optional[str]) -> Optional[str]:
    """
    question_override를 캐시 키 비교 가능 형태로 정규화합니다.
    """
    if text is None:
        return None
    normalized = " ".join(str(text).split())
    if not normalized:
        return None
    return normalized


def build_focus_signature(focus: Optional[Sequence[str]]) -> str:
    normalized_focus = normalize_focus(focus)
    if not normalized_focus:
        return "all"
    return "+".join(normalized_focus)


def build_question_signature(question_override: Optional[str]) -> str:
    normalized = normalize_question_override(question_override)
    if not normalized:
        return "auto"
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return f"q:{digest[:16]}"


def build_coach_cache_key(
    *,
    schema_version: str,
    prompt_version: str,
    home_team_code: str,
    away_team_code: Optional[str],
    year: int,
    game_type: str,
    focus: Optional[Sequence[str]],
    question_override: Optional[str],
    question_signature_override: Optional[str] = None,
) -> tuple[str, Dict[str, Any]]:
    """
    coach_analysis_cache PK용 해시 키와 payload를 반환합니다.
    """
    normalized_focus = normalize_focus(focus)
    payload: Dict[str, Any] = {
        "schema": schema_version,
        "prompt_version": prompt_version,
        "team_code_canonical": str(home_team_code).upper(),
        "away_team_code_canonical": str(away_team_code or "").upper(),
        "year": int(year),
        "game_type": str(game_type).upper().strip() or "UNKNOWN",
        "focus_signature": build_focus_signature(normalized_focus),
        "question_signature": (
            question_signature_override.strip().lower()
            if question_signature_override
            else build_question_signature(question_override)
        ),
    }
    cache_key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return cache_key, payload
