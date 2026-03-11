"""Coach 분석 캐시 키 생성/정규화 유틸."""

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


def _normalize_signature_part(value: Optional[str], *, fallback: str) -> str:
    normalized = " ".join(str(value or "").split()).strip().lower()
    if not normalized:
        return fallback
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return f"{fallback}:{digest[:12]}"


def build_starter_signature(
    home_pitcher: Optional[str],
    away_pitcher: Optional[str],
) -> str:
    return _normalize_signature_part(
        f"{home_pitcher or ''}|{away_pitcher or ''}",
        fallback="starter_pending",
    )


def build_lineup_signature(lineup_players: Optional[Sequence[str]]) -> str:
    if not lineup_players:
        return "lineup_pending"

    normalized_players = sorted(
        {
            " ".join(str(player or "").split()).strip().lower()
            for player in lineup_players
            if str(player or "").strip()
        }
    )
    if not normalized_players:
        return "lineup_pending"

    digest = hashlib.sha256("|".join(normalized_players).encode()).hexdigest()
    return f"lineup:{digest[:12]}"


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
    game_id: Optional[str] = None,
    league_type_code: Optional[int] = None,
    stage_label: Optional[str] = None,
    starter_signature: Optional[str] = None,
    lineup_signature: Optional[str] = None,
    request_mode: Optional[str] = None,
    game_status_bucket: Optional[str] = None,
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
        "game_id": str(game_id or "").strip(),
        "game_type": str(game_type).upper().strip() or "UNKNOWN",
        "league_type_code": (
            int(league_type_code) if league_type_code is not None else None
        ),
        "stage_label": str(stage_label or "").upper().strip() or "UNKNOWN",
        "focus_signature": build_focus_signature(normalized_focus),
        "starter_signature": str(starter_signature or "starter_pending"),
        "lineup_signature": str(lineup_signature or "lineup_pending"),
        "request_mode": str(request_mode or "manual_detail").lower().strip(),
        "game_status_bucket": str(game_status_bucket or "").upper().strip()
        or "UNKNOWN",
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
