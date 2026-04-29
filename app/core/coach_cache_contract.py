"""Shared Coach cache key contract."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .coach_cache_key import (
    build_coach_cache_key,
    build_lineup_signature,
    build_starter_signature,
)

COACH_CACHE_SCHEMA_VERSION = "v5"
COACH_CACHE_PROMPT_VERSION = "v88_grounded_copy_quality"


def _normalize_optional_signature(value: Optional[str]) -> Optional[str]:
    normalized = " ".join(str(value or "").split()).strip()
    return normalized or None


def resolve_coach_cache_signatures(
    *,
    requested_starter_signature: Optional[str],
    requested_lineup_signature: Optional[str],
    home_pitcher: Optional[str],
    away_pitcher: Optional[str],
    lineup_players: Optional[Sequence[str]],
) -> tuple[str, str]:
    starter_signature = _normalize_optional_signature(requested_starter_signature)
    lineup_signature = _normalize_optional_signature(requested_lineup_signature)
    if starter_signature and lineup_signature:
        return starter_signature, lineup_signature

    return (
        build_starter_signature(home_pitcher, away_pitcher),
        build_lineup_signature(lineup_players),
    )


def build_coach_cache_identity(
    *,
    home_team_code: str,
    away_team_code: Optional[str],
    year: int,
    game_type: str,
    focus: Optional[Sequence[str]],
    question_override: Optional[str],
    game_id: Optional[str] = None,
    league_type_code: Optional[int] = None,
    stage_label: Optional[str] = None,
    request_mode: Optional[str] = None,
    game_status_bucket: Optional[str] = None,
    question_signature_override: Optional[str] = None,
    requested_starter_signature: Optional[str] = None,
    requested_lineup_signature: Optional[str] = None,
    home_pitcher: Optional[str] = None,
    away_pitcher: Optional[str] = None,
    lineup_players: Optional[Sequence[str]] = None,
) -> tuple[str, Dict[str, Any], str, str]:
    starter_signature, lineup_signature = resolve_coach_cache_signatures(
        requested_starter_signature=requested_starter_signature,
        requested_lineup_signature=requested_lineup_signature,
        home_pitcher=home_pitcher,
        away_pitcher=away_pitcher,
        lineup_players=lineup_players,
    )
    cache_key, payload = build_coach_cache_key(
        schema_version=COACH_CACHE_SCHEMA_VERSION,
        prompt_version=COACH_CACHE_PROMPT_VERSION,
        home_team_code=home_team_code,
        away_team_code=away_team_code,
        year=year,
        game_type=game_type,
        focus=focus,
        question_override=question_override,
        game_id=game_id,
        league_type_code=league_type_code,
        stage_label=stage_label,
        starter_signature=starter_signature,
        lineup_signature=lineup_signature,
        request_mode=request_mode,
        game_status_bucket=game_status_bucket,
        question_signature_override=question_signature_override,
    )
    return cache_key, payload, starter_signature, lineup_signature
