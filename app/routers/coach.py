"""
'The Coach' 기능과 관련된 API 엔드포인트를 정의합니다.

Fast Path 최적화:
- 도구 계획 LLM 호출을 건너뛰고 focus 영역에 따라 직접 도구 호출
- 병렬 도구 실행으로 대기 시간 단축
- Coach 전용 컨텍스트 포맷팅
"""

import logging
import json
import asyncio
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from datetime import date as date_cls, datetime
from typing import List, Optional, Dict, Any, Literal, Set, AsyncIterator, Tuple
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, model_validator

from psycopg.rows import dict_row

from psycopg_pool import ConnectionPool

from ..deps import (
    get_agent,
    get_connection_pool,
    get_coach_llm_generator,
    require_ai_internal_token,
    resolve_coach_openrouter_models,
)
from ..config import get_settings
from ..agents.baseball_agent import BaseballStatisticsAgent
from ..core.coach_grounding import (
    CoachFactSheet,
    collect_numeric_tokens,
    format_coach_fact_sheet,
    validate_response_against_fact_sheet,
    extend_numeric_tokens,
    response_is_semantically_empty,
)
from ..core.prompts import COACH_PROMPT_V2
from ..core.coach_validator import (
    parse_coach_response_with_meta,
    CoachResponse,
)
from ..core.coach_cache_contract import (
    COACH_CACHE_PROMPT_VERSION,
    COACH_CACHE_SCHEMA_VERSION,
    build_coach_cache_identity,
)
from ..core.coach_cache_key import normalize_focus
from ..core.ratelimit import rate_limit_coach_dependency
from ..tools.database_query import DatabaseQueryTool
from ..tools.team_code_resolver import CANONICAL_CODES, TeamCodeResolver

logger = logging.getLogger(__name__)

COACH_YEAR_MIN = 1982
MAX_COACH_FOCUS_ITEMS = 6
MAX_COACH_QUESTION_OVERRIDE_LENGTH = 2000
PENDING_STALE_SECONDS = 90
PENDING_WAIT_TIMEOUT_SECONDS = 45
PENDING_WAIT_POLL_MS = 1000
FAILED_RETRY_AFTER_SECONDS = 30
COACH_CACHE_HEARTBEAT_INTERVAL_SECONDS = 5
COACH_CACHE_LEASE_STALE_SECONDS = 90
COACH_CACHE_MAX_RETRYABLE_ATTEMPTS = 3
COACH_PENDING_RECHECK_AFTER_SECONDS = 120
VOLATILE_CACHE_TTL_SECONDS = 300
COMPLETED_CACHE_TTL_SECONDS = 86400
COACH_LLM_STATUS_HEARTBEAT_SECONDS = 10.0
COACH_SCHEDULED_PARTIAL_MANUAL_MAX_TOKENS = 1200
COACH_SCHEDULED_PARTIAL_MANUAL_IDLE_TIMEOUT_SECONDS = 35.0
COACH_SCHEDULED_PARTIAL_MANUAL_TOTAL_TIMEOUT_SECONDS = 45.0
COACH_SCHEDULED_PARTIAL_MANUAL_REQUEST_TIMEOUT_SECONDS = 45.0
COACH_SCHEDULED_PARTIAL_MANUAL_EMPTY_CHUNK_RETRIES = 0
COACH_SCHEDULED_HALF_CONFIRMED_MANUAL_MAX_TOKENS = 900
COACH_SCHEDULED_HALF_CONFIRMED_MANUAL_IDLE_TIMEOUT_SECONDS = 15.0
COACH_SCHEDULED_HALF_CONFIRMED_MANUAL_TOTAL_TIMEOUT_SECONDS = 18.0
COACH_SCHEDULED_HALF_CONFIRMED_MANUAL_REQUEST_TIMEOUT_SECONDS = 18.0
COACH_SCHEDULED_HALF_CONFIRMED_MANUAL_FIRST_CHUNK_TIMEOUT_SECONDS = 8.0
COACH_SCHEDULED_HALF_CONFIRMED_MANUAL_EMPTY_CHUNK_RETRIES = 0
COACH_COMPLETED_MANUAL_MAX_TOKENS = 1400
COACH_COMPLETED_MANUAL_IDLE_TIMEOUT_SECONDS = 18.0
COACH_COMPLETED_MANUAL_TOTAL_TIMEOUT_SECONDS = 24.0
COACH_COMPLETED_MANUAL_REQUEST_TIMEOUT_SECONDS = 24.0
COACH_COMPLETED_MANUAL_EMPTY_CHUNK_RETRIES = 0
COACH_REQUEST_MODE_AUTO = "auto_brief"
COACH_REQUEST_MODE_MANUAL = "manual_detail"
COACH_INTERNAL_ERROR_CODE = "coach_internal_error"
COACH_INTERNAL_ERROR_MESSAGE = (
    "분석 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
)
COACH_TOOL_FETCH_FAILED_CODE = "coach_tool_fetch_failed"
COACH_DATA_INSUFFICIENT_CODE = "coach_data_insufficient"
COACH_VALIDATION_FAILED_CODE = "coach_response_validation_failed"
COACH_EMPTY_RESPONSE_ERROR_CODE = "empty_response"
COACH_NO_JSON_FOUND_ERROR_CODE = "no_json_found"
COACH_JSON_DECODE_ERROR_CODE = "json_decode_error"
COACH_STREAM_CANCELLED_ERROR_CODE = "stream_cancelled"
COACH_OCI_MAPPING_DEGRADED_ERROR_CODE = "oci_mapping_degraded"
COACH_UNSUPPORTED_ENTITY_ERROR_CODE = "unsupported_entity_name"
COACH_UNSUPPORTED_CLAIM_ERROR_CODE = "grounding_validation_failed"
COACH_NON_RETRYABLE_ERROR_CODE = "non_retryable_internal_error"
MANUAL_BASEBALL_DATA_REQUIRED_CODE = "MANUAL_BASEBALL_DATA_REQUIRED"
MANUAL_BASEBALL_DATA_REQUIRED_MESSAGE = (
    "야구 데이터 준비가 필요합니다. 운영자가 데이터를 제공하면 다시 확인할 수 있습니다."
)
COACH_CACHE_ERROR_MESSAGES: Dict[str, str] = {
    COACH_INTERNAL_ERROR_CODE: "분석 시스템 내부 오류로 캐시를 생성하지 못했습니다.",
    COACH_TOOL_FETCH_FAILED_CODE: "분석에 필요한 팀 데이터를 조회하지 못했습니다.",
    COACH_DATA_INSUFFICIENT_CODE: "분석에 필요한 데이터가 충분하지 않습니다.",
    COACH_VALIDATION_FAILED_CODE: "분석 응답 검증에 실패해 결과를 저장하지 못했습니다.",
    COACH_EMPTY_RESPONSE_ERROR_CODE: "LLM 응답이 비어 자동 복구 대기 중입니다.",
    COACH_NO_JSON_FOUND_ERROR_CODE: "LLM 응답에서 JSON을 추출하지 못해 자동 복구 대기 중입니다.",
    COACH_JSON_DECODE_ERROR_CODE: "LLM 응답 JSON 파싱에 실패해 자동 복구 대기 중입니다.",
    COACH_STREAM_CANCELLED_ERROR_CODE: "분석 스트림이 중단되어 자동 복구 대기 중입니다.",
    COACH_OCI_MAPPING_DEGRADED_ERROR_CODE: "팀 매핑 조회가 일시적으로 불안정했습니다. 잠시 후 재시도해주세요.",
    COACH_UNSUPPORTED_ENTITY_ERROR_CODE: "근거에 없는 엔티티가 감지되어 결과를 잠갔습니다.",
    COACH_UNSUPPORTED_CLAIM_ERROR_CODE: "근거 검증에 실패해 결과를 잠갔습니다.",
    COACH_NON_RETRYABLE_ERROR_CODE: "자동 복구가 불가능한 내부 오류가 발생했습니다.",
}
ALLOWED_COACH_CACHE_ERROR_CODES = set(COACH_CACHE_ERROR_MESSAGES.keys())
RETRYABLE_CACHE_ERROR_CODES = {
    COACH_EMPTY_RESPONSE_ERROR_CODE,
    COACH_NO_JSON_FOUND_ERROR_CODE,
    COACH_JSON_DECODE_ERROR_CODE,
    COACH_STREAM_CANCELLED_ERROR_CODE,
    COACH_OCI_MAPPING_DEGRADED_ERROR_CODE,
}
FOCUS_SECTION_HEADERS: Dict[str, str] = {
    "recent_form": "## 최근 전력",
    "bullpen": "## 불펜 상태",
    "starter": "## 선발 투수",
    "matchup": "## 상대 전적",
    "batting": "## 타격 생산성",
}
FOCUS_SECTION_METRIC_LABELS: Dict[str, Tuple[str, ...]] = {
    "recent_form": ("최근 흐름", "폼 진단", "득실점 마진"),
    "bullpen": ("불펜 비중", "불펜 운용", "불펜 데이터"),
    "starter": ("발표 선발",),
    "matchup": ("상대 전적", "시리즈 전적", "시리즈 맥락"),
    "batting": ("정규시즌 OPS", "팀 타격 생산성"),
}
FOCUS_SECTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "recent_form": ("최근", "흐름", "상승세", "하락세", "득실차"),
    "bullpen": ("불펜",),
    "starter": ("선발",),
    "matchup": ("상대", "전적", "시리즈"),
    "batting": ("타격", "OPS", "장타", "출루", "타순"),
}
FOCUS_SECTION_GENERIC_SUMMARIES: Dict[str, str] = {
    "recent_form": "최근 흐름은 확인 가능한 경기 표본 기준으로 제한적으로 해석해야 합니다.",
    "bullpen": "불펜 운용 근거가 제한적이어서 경기 중반 이후 대응력이 변수입니다.",
    "starter": "선발 정보가 확정되지 않아 선발 맞대결 평가는 제한됩니다.",
    "matchup": "상대 전적 근거가 충분하지 않아 직접 비교는 보수적으로 해석해야 합니다.",
    "batting": "타격 생산성 근거가 제한적이어서 장타와 출루 비교는 보수적으로 봐야 합니다.",
}
LEAGUE_TYPE_CODE_TO_STAGE: Dict[int, str] = {
    0: "REGULAR",
    1: "PRE",
    2: "WC",
    3: "SEMI_PO",
    4: "PO",
    5: "KS",
}
STAGE_TO_LEAGUE_TYPE_CODE: Dict[str, int] = {
    stage: code for code, stage in LEAGUE_TYPE_CODE_TO_STAGE.items()
}
STAGE_TO_ROUND_DISPLAY: Dict[str, str] = {
    "REGULAR": "정규시즌",
    "PRE": "시범경기",
    "WC": "와일드카드",
    "SEMI_PO": "준플레이오프",
    "PO": "플레이오프",
    "KS": "한국시리즈",
}
COACH_META_GENERATION_MODES = {
    "deterministic_auto",
    "llm_manual",
    "evidence_fallback",
}
COACH_META_DATA_QUALITIES = {"grounded", "partial", "insufficient"}
COACH_GROUNDING_REASON_MESSAGES: Dict[str, str] = {
    "missing_game_context": "경기 기본 맥락이 충분하지 않아 보수적으로 해석합니다.",
    "missing_starters": "선발 정보가 완전하지 않아 선발 관련 표현을 제한합니다.",
    "missing_lineups": "라인업이 확정되지 않아 타순 관련 단정은 피합니다.",
    "missing_summary": "경기 요약 근거가 부족해 최근 활약 서술을 제한합니다.",
    "missing_metadata": "경기 메타데이터가 부족해 일부 맥락 표현이 제한됩니다.",
    "missing_series_context": "시리즈 전황 근거가 부족해 포스트시즌 맥락을 단정하지 않습니다.",
    "missing_clutch_moments": "WPA 기반 승부처 데이터가 부족합니다.",
    "focus_data_unavailable": "요청한 focus 근거가 부족해 확인 가능한 항목만 분석하거나 보수 요약으로 전환합니다.",
    "unsupported_entity_name": "확인된 엔티티 밖의 선수명이 감지되어 보수 요약으로 전환했습니다.",
    "unsupported_entity_name_sanitized": "근거에 없는 선수명은 일반 표현으로 정리해 확인 가능한 범위만 반영했습니다.",
    "unsupported_numeric_claim": "근거 fact sheet에 없는 수치가 감지되어 보수 요약으로 전환했습니다.",
    "unsupported_numeric_claim_sanitized": "근거에 없는 수치는 일반 표현으로 정리해 확인 가능한 범위만 반영했습니다.",
    "empty_response": "LLM 응답에 실질적인 분석 내용이 없어 재시도 또는 보수 요약으로 전환했습니다.",
    "unconfirmed_starter_claim": "선발 미확정 경기에서 확정 표현이 감지되어 보수 요약으로 전환했습니다.",
    "unconfirmed_lineup_claim": "라인업 미발표 경기에서 확정 표현이 감지되어 보수 요약으로 전환했습니다.",
    "unconfirmed_series_claim": "시리즈 맥락 부족 상태에서 확정 표현이 감지되어 보수 요약으로 전환했습니다.",
    "llm_parse_failed": "LLM 응답 형식이 불안정해 보수 요약으로 전환했습니다.",
    COACH_OCI_MAPPING_DEGRADED_ERROR_CODE: "팀 매핑 조회가 일시적으로 불안정해 재연결 또는 마지막 정상 스냅샷으로 이어갔습니다.",
}
COACH_EVIDENCE_ROOT_CAUSE_ORDER = (
    "missing_game_context",
    "missing_starters",
    "missing_lineups",
    "missing_summary",
    "missing_metadata",
    "missing_series_context",
)
SCHEDULED_PARTIAL_GROUNDING_REASONS = (
    "missing_starters",
    "missing_lineups",
    "missing_summary",
    "missing_metadata",
    "focus_data_unavailable",
)


@dataclass
class EvidenceSeriesState:
    stage_label: str = "REGULAR"
    round_display: str = "정규시즌"
    game_no: Optional[int] = None
    previous_games: int = 0
    confirmed_previous_games: int = 0
    home_team_wins: int = 0
    away_team_wins: int = 0
    series_state_partial: bool = False
    series_state_hint_mismatch: bool = False

    def matchup_label(self) -> str:
        if self.game_no is None:
            return self.round_display
        return f"{self.round_display} {self.game_no}차전"

    def has_confirmed_score(self) -> bool:
        return (
            self.game_no is not None
            and self.previous_games > 0
            and not self.series_state_partial
        )

    def summary_text(self, home_team_name: str, away_team_name: str) -> str:
        if self.game_no is None:
            return f"{self.round_display} 전황 데이터가 없습니다."
        if self.previous_games <= 0:
            return f"{self.matchup_label()} 시작 전입니다."
        if self.series_state_partial:
            return (
                f"{self.matchup_label()}입니다. "
                "시리즈 전적은 DB 이력 부족으로 축약 표시합니다."
            )
        return (
            f"{self.round_display} 시리즈 전적: "
            f"{away_team_name} {self.away_team_wins}승 "
            f"{home_team_name} {self.home_team_wins}승 "
            f"(이번 경기는 {self.game_no}차전)"
        )


@dataclass
class GameEvidence:
    season_year: int
    home_team_code: str
    away_team_code: Optional[str]
    home_team_name: str
    away_team_name: str
    game_id: Optional[str] = None
    game_row_found: bool = False
    season_id: Optional[int] = None
    game_date: Optional[str] = None
    game_status: str = "UNKNOWN"
    game_status_bucket: str = "UNKNOWN"
    league_type_code: Optional[int] = None
    stage_label: str = "REGULAR"
    round_display: str = "정규시즌"
    stage_game_no_hint: Optional[int] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    winning_team_code: Optional[str] = None
    winning_team_name: Optional[str] = None
    home_pitcher: Optional[str] = None
    away_pitcher: Optional[str] = None
    lineup_announced: bool = False
    home_lineup: List[str] = field(default_factory=list)
    away_lineup: List[str] = field(default_factory=list)
    summary_items: List[str] = field(default_factory=list)
    stadium_name: Optional[str] = None
    start_time: Optional[str] = None
    weather: Optional[str] = None
    series_state: Optional[EvidenceSeriesState] = None
    used_evidence: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GameEvidenceAssessment:
    used_evidence: List[str]
    root_causes: List[str]
    expected_data_quality: str
    home_pitcher_present: bool
    away_pitcher_present: bool
    lineup_announced: bool
    summary_present: bool
    metadata_present: bool
    series_context_available: bool


def _grounding_reason_message(code: str) -> str:
    return COACH_GROUNDING_REASON_MESSAGES.get(
        code,
        "근거 범위를 벗어난 표현이 감지되어 보수적으로 처리합니다.",
    )


def _normalize_grounding_reasons(codes: List[str]) -> List[str]:
    return [code for code in dict.fromkeys(code for code in codes if code)]


def _has_confirmed_lineup_details(evidence: GameEvidence) -> bool:
    return (
        bool(evidence.lineup_announced)
        and bool(evidence.home_lineup)
        and bool(evidence.away_lineup)
    )


def _merge_grounding_warnings(
    warnings: List[str],
    reasons: List[str],
) -> List[str]:
    merged = list(warnings)
    for code in reasons:
        message = _grounding_reason_message(code)
        if message not in merged:
            merged.append(message)
    return merged


def _log_coach_stream_meta(
    payload: Dict[str, Any],
    *,
    game_id: Optional[str],
) -> None:
    logger.info(
        "[Coach Meta] request_mode=%s game_id=%s cache_state=%s validation_status=%s "
        "generation_mode=%s cached=%s in_progress=%s data_quality=%s supported_fact_count=%s "
        "grounding_reasons=%s grounding_warnings=%s used_evidence=%s resolved_focus=%s "
        "missing_focus_sections=%s",
        payload.get("request_mode"),
        game_id,
        payload.get("cache_state"),
        payload.get("validation_status"),
        payload.get("generation_mode"),
        payload.get("cached"),
        payload.get("in_progress"),
        payload.get("data_quality"),
        payload.get("supported_fact_count"),
        payload.get("grounding_reasons"),
        payload.get("grounding_warnings"),
        payload.get("used_evidence"),
        payload.get("resolved_focus"),
        payload.get("missing_focus_sections"),
    )


def _build_fact_line(label: str, value: Optional[str]) -> Optional[str]:
    clean_value = _clean_summary_text(value)
    if not clean_value:
        return None
    return f"{label}: {clean_value}"


def _safe_int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_tool_player_stats(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    stats: List[str] = []
    for key in ("ops", "avg", "era", "whip"):
        value = item.get(key)
        if value is None:
            continue
        try:
            if key in {"ops", "avg"}:
                stats.append(f"{key.upper()} {float(value):.3f}")
            else:
                stats.append(f"{key.upper()} {float(value):.2f}")
        except (TypeError, ValueError):
            stats.append(f"{key.upper()} {value}")
    for key, suffix in (
        ("home_runs", "HR"),
        ("rbi", "RBI"),
        ("wins", "승"),
        ("losses", "패"),
    ):
        value = item.get(key)
        if value is None:
            continue
        stats.append(f"{value}{suffix}")
    return ", ".join(stats[:3])


def _player_form_signals(team_data: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    signals = team_data.get("player_form_signals", {}) if team_data else {}
    values = signals.get(key, []) if isinstance(signals, dict) else []
    return [item for item in values if isinstance(item, dict)]


def _best_form_signal(team_data: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    signals = sorted(
        _player_form_signals(team_data, key),
        key=lambda item: float(item.get("form_score") or -1.0),
        reverse=True,
    )
    return signals[0] if signals else None


def _team_form_score(team_data: Dict[str, Any]) -> Optional[float]:
    scores: List[float] = []
    for key in ("batters", "pitchers"):
        signal = _best_form_signal(team_data, key)
        score = signal.get("form_score") if signal else None
        if score is None:
            continue
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            continue
    if not scores:
        return None
    return sum(scores) / len(scores)


def _form_status_label(value: Optional[str]) -> str:
    mapping = {
        "hot": "상승",
        "steady": "보합",
        "cold": "하락",
        "insufficient": "표본 부족",
    }
    return mapping.get(str(value or ""), "보합")


def _form_status_trend_label(value: Optional[str]) -> str:
    mapping = {
        "hot": "상승세",
        "steady": "보합세",
        "cold": "하락세",
        "insufficient": "표본 부족",
    }
    return mapping.get(str(value or ""), "보합세")


def _format_form_signal_fact(signal: Dict[str, Any]) -> str:
    player_name = _normalize_name_token(signal.get("player_name")) or "선수 미상"
    form_score = signal.get("form_score")
    score_text = (
        f"{float(form_score):.1f}"
        if isinstance(form_score, (int, float))
        else str(form_score or "데이터 부족")
    )
    season_metrics = signal.get("season_metrics", {}) or {}
    recent_metrics = signal.get("recent_metrics", {}) or {}
    clutch_metrics = signal.get("clutch_metrics", {}) or {}
    if "wrc_plus" in season_metrics:
        return (
            f"폼 {_form_status_label(signal.get('form_status'))}, 점수 {score_text}, "
            f"시즌 wRC+ {season_metrics.get('wrc_plus', '데이터 부족')}, "
            f"OPS+ {season_metrics.get('ops_plus', '데이터 부족')}, "
            f"최근 OPS {recent_metrics.get('ops', '데이터 부족')}, "
            f"최근 WPA/PA {clutch_metrics.get('recent_wpa_per_pa', '데이터 부족')}"
        )
    return (
        f"폼 {_form_status_label(signal.get('form_status'))}, 점수 {score_text}, "
        f"시즌 ERA+ {season_metrics.get('era_plus', '데이터 부족')}, "
        f"FIP+ {season_metrics.get('fip_plus', '데이터 부족')}, "
        f"최근 ERA {recent_metrics.get('era', '데이터 부족')}, "
        f"WPA 허용/BF {clutch_metrics.get('recent_wpa_allowed_per_bf', '데이터 부족')}"
    )


def _clutch_moments(tool_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    clutch = tool_results.get("clutch_moments", {}) or {}
    moments = clutch.get("moments", []) if isinstance(clutch, dict) else []
    return [item for item in moments if isinstance(item, dict)]


def _format_clutch_fact(moment: Dict[str, Any]) -> str:
    player_name = _normalize_name_token(moment.get("batter_name")) or "타자 미상"
    return (
        f"{moment.get('inning_label', '이닝 미상')}, "
        f"{moment.get('outs', 0)}사, "
        f"주자 {moment.get('bases_before', '-')}, "
        f"{player_name}, WPA {moment.get('wpa_delta_pct', '데이터 부족')}%p, "
        f"{moment.get('description', '설명 없음')}"
    )


SUMMARY_CLUTCH_KEYWORDS = (
    "결승",
    "끝내기",
    "역전",
    "동점",
    "쐐기",
    "세이브",
    "홀드",
)


def _summary_item_is_clutch_like(item: Optional[str]) -> bool:
    normalized = _clean_summary_text(item)
    if not normalized:
        return False
    return any(keyword in normalized for keyword in SUMMARY_CLUTCH_KEYWORDS)


def _primary_review_summary_item(evidence: GameEvidence) -> Optional[str]:
    prioritized: Optional[str] = None
    fallback: Optional[str] = None
    for raw_item in evidence.summary_items:
        item = _clean_summary_text(raw_item)
        if not item:
            continue
        if fallback is None:
            fallback = item
        if _summary_item_is_clutch_like(item):
            prioritized = item
            break
    return prioritized or fallback


def _completed_scoreline_text(evidence: GameEvidence) -> Optional[str]:
    if evidence.away_score is None or evidence.home_score is None:
        return None
    return (
        f"{evidence.away_team_name} {int(evidence.away_score)} / "
        f"{evidence.home_team_name} {int(evidence.home_score)}"
    )


def _completed_winner_code(evidence: GameEvidence) -> Optional[str]:
    explicit = str(evidence.winning_team_code or "").strip().upper()
    if explicit:
        return explicit
    if evidence.home_score is None or evidence.away_score is None:
        return None
    if int(evidence.home_score) == int(evidence.away_score):
        return None
    return (
        str(evidence.home_team_code or "").strip().upper()
        if int(evidence.home_score) > int(evidence.away_score)
        else str(evidence.away_team_code or "").strip().upper()
    )


def _completed_winner_name(evidence: GameEvidence) -> Optional[str]:
    explicit = _normalize_name_token(evidence.winning_team_name)
    if explicit:
        return explicit
    winner_code = _completed_winner_code(evidence)
    if not winner_code:
        return None
    if winner_code == str(evidence.home_team_code or "").strip().upper():
        return evidence.home_team_name
    if winner_code == str(evidence.away_team_code or "").strip().upper():
        return evidence.away_team_name
    return None


def _completed_loser_name(evidence: GameEvidence) -> Optional[str]:
    winner_code = _completed_winner_code(evidence)
    if not winner_code:
        return None
    if winner_code == str(evidence.home_team_code or "").strip().upper():
        return evidence.away_team_name
    if winner_code == str(evidence.away_team_code or "").strip().upper():
        return evidence.home_team_name
    return None


def _completed_game_is_draw(evidence: GameEvidence) -> bool:
    if evidence.home_score is None or evidence.away_score is None:
        return False
    return int(evidence.home_score) == int(evidence.away_score)


def _completed_result_margin(evidence: GameEvidence) -> Optional[int]:
    if evidence.home_score is None or evidence.away_score is None:
        return None
    return abs(int(evidence.home_score) - int(evidence.away_score))


def _completed_result_metric(evidence: GameEvidence) -> Optional[Dict[str, Any]]:
    scoreline = _completed_scoreline_text(evidence)
    if not scoreline:
        return None
    if _completed_game_is_draw(evidence):
        return {
            "label": "경기 결과",
            "value": f"{scoreline} 무승부",
            "status": "warning",
            "trend": "neutral",
            "is_critical": True,
        }
    winner_name = _completed_winner_name(evidence)
    if not winner_name:
        return {
            "label": "최종 스코어",
            "value": scoreline,
            "status": "warning",
            "trend": "neutral",
            "is_critical": True,
        }
    return {
        "label": "최종 스코어",
        "value": f"{scoreline} ({winner_name} 승)",
        "status": "good",
        "trend": "neutral",
        "is_critical": True,
    }


def _build_completed_review_why_it_matters(
    evidence: GameEvidence,
    *,
    winner_name: Optional[str],
    loser_name: Optional[str],
    result_margin: Optional[int],
    review_summary_item: Optional[str],
    review_summary_is_clutch: bool,
    clutch_moments: List[Dict[str, Any]],
    predicted_edge_team: Optional[str],
    home_name: str,
    away_name: str,
    home_recent_games: int,
    away_recent_games: int,
    home_run_diff: int,
    away_run_diff: int,
    home_bullpen: Optional[float],
    away_bullpen: Optional[float],
) -> List[str]:
    if _completed_game_is_draw(evidence):
        return [
            "승패는 갈리지 않았고, 득점 연결과 실점 억제가 비슷한 수준으로 맞섰습니다."
        ]

    reasons: List[str] = []
    if winner_name and loser_name:
        if clutch_moments:
            if result_margin == 1:
                reasons.append(
                    f"{winner_name}는 고레버리지 기회를 끝까지 지켜내며 결과를 만들었습니다."
                )
            else:
                reasons.append(
                    f"{winner_name}는 고레버리지 기회를 득점 흐름으로 연결하며 주도권을 잡았습니다."
                )
        elif review_summary_item:
            if review_summary_is_clutch:
                reasons.append(
                    f"{winner_name}는 '{review_summary_item}' 장면을 득점으로 연결했습니다."
                )
            else:
                reasons.append(
                    f"{winner_name}는 '{review_summary_item}' 장면에서 경기 흐름을 묶었습니다."
                )
        elif isinstance(result_margin, int):
            if result_margin == 1:
                reasons.append(
                    f"{winner_name}는 한 점 차 접전에서 마지막 득점 연결로 앞섰습니다."
                )
            elif result_margin <= 3:
                reasons.append(
                    f"{winner_name}는 중반 이후 추가 득점으로 {loser_name}의 추격을 끊었습니다."
                )
            else:
                reasons.append(
                    f"{winner_name}는 초중반부터 점수 차를 벌리며 {loser_name}의 운영 부담을 키웠습니다."
                )

    if winner_name and predicted_edge_team:
        if predicted_edge_team == winner_name:
            reasons.append(
                f"사전 지표의 {winner_name} 우세가 실제 결과로 이어졌습니다."
            )
        else:
            reasons.append(
                f"사전 지표는 {predicted_edge_team} 쪽이었지만, 실제 경기는 {winner_name}가 뒤집었습니다."
            )

    if (
        winner_name
        and home_recent_games >= 2
        and away_recent_games >= 2
        and home_run_diff != away_run_diff
    ):
        recent_edge_team = home_name if home_run_diff > away_run_diff else away_name
        if recent_edge_team == winner_name:
            reasons.append(
                f"최근 득실 흐름의 우세도 {winner_name} 쪽에서 유지됐습니다."
            )
        else:
            reasons.append(
                f"최근 득실 흐름은 {recent_edge_team} 쪽이었지만, 결과는 {winner_name}가 가져갔습니다."
            )
    elif (
        winner_name
        and home_bullpen
        and away_bullpen
        and abs(home_bullpen - away_bullpen) >= 3.0
    ):
        fresher_team = home_name if home_bullpen < away_bullpen else away_name
        if fresher_team == winner_name:
            reasons.append(
                f"불펜 부담이 더 낮았던 {winner_name}가 후반 운영 폭을 지켰습니다."
            )
        else:
            reasons.append(
                f"불펜 부담 지표는 {fresher_team} 쪽이 나았지만, 실제 경기는 {winner_name}가 상쇄했습니다."
            )

    if not reasons and winner_name and loser_name:
        reasons.append(
            f"{winner_name}가 중요한 득점 구간을 한 번 더 살리며 결과를 만들었습니다."
        )
    return reasons[:3]


def _summary_text_key(value: Optional[str]) -> str:
    text = _clean_summary_text(value)
    if not text:
        return ""
    return re.sub(r"[\s\.\,\!\?\"'`·:;()\[\]\-_/]+", "", text)


def _summary_texts_overlap(left: Optional[str], right: Optional[str]) -> bool:
    left_key = _summary_text_key(left)
    right_key = _summary_text_key(right)
    if not left_key or not right_key:
        return False
    return left_key == right_key or left_key in right_key or right_key in left_key


def _build_completed_review_turning_point_verdict(
    *,
    winner_name: Optional[str],
    top_moment: Optional[Dict[str, Any]],
    review_summary_item: Optional[str],
    review_summary_is_clutch: bool,
) -> str:
    subject = f"{winner_name} 승리의 분기점은 " if winner_name else "실제 분기점은 "
    if top_moment:
        inning_label = (
            _normalize_name_token(top_moment.get("inning_label")) or "핵심 이닝"
        )
        return f"{subject}{inning_label} 승부처 대응이었습니다."
    if review_summary_item:
        if review_summary_is_clutch:
            return (
                f"{subject}'{review_summary_item}' 장면을 리드로 연결한 구간이었습니다."
            )
        return f"{subject}'{review_summary_item}' 전후 운영이었습니다."
    return f"{subject}득점 연결 뒤 흐름을 지킨 운영이었습니다."


def _build_completed_review_watch_point(
    *,
    top_moment: Optional[Dict[str, Any]],
    review_summary_item: Optional[str],
) -> str:
    if top_moment:
        return "해당 승부처 직전 배터리 선택과 주자 운영을 다시 볼 필요가 있습니다."
    if review_summary_item:
        return f"'{review_summary_item}' 직전 주자 상황과 투수 교체 선택을 다시 볼 필요가 있습니다."
    return "리드를 만든 직후 투수 교체와 주자 운영이 어떻게 이어졌는지 다시 볼 필요가 있습니다."


def _append_distinct_note_part(parts: List[str], candidate: Optional[str]) -> None:
    text = _clean_summary_text(candidate)
    if not text:
        return

    normalized = re.sub(r"\s+", " ", text).strip().rstrip(".!?")
    if not normalized:
        return

    for existing in parts:
        existing_normalized = re.sub(r"\s+", " ", existing).strip().rstrip(".!?")
        if (
            normalized == existing_normalized
            or normalized in existing_normalized
            or existing_normalized in normalized
        ):
            return

    parts.append(text)


def _build_compact_coach_note(parts: List[str], *, max_length: int = 220) -> str:
    selected: List[str] = []
    selected_keys: List[str] = []
    for part in parts:
        sentence = _ensure_sentence(part)
        if not sentence:
            continue
        sentence_key = re.sub(r"\s+", " ", sentence).strip().rstrip(".!?")
        if any(
            sentence_key == existing_key
            or sentence_key in existing_key
            or existing_key in sentence_key
            for existing_key in selected_keys
        ):
            continue

        candidate = " ".join([*selected, sentence]).strip()
        if len(candidate) > max_length and selected:
            break

        selected.append(sentence)
        selected_keys.append(sentence_key)
        if len(selected) >= 2 and len(candidate) >= max_length * 0.7:
            break

    if not selected:
        return ""
    return _truncate_text_naturally(" ".join(selected), max_length=max_length)


def _append_team_fact_lines(
    fact_lines: List[str],
    numeric_tokens: Set[str],
    tool_results: Dict[str, Any],
    team_key: str,
    team_name: str,
) -> None:
    team_data = tool_results.get(team_key, {}) or {}
    recent = _recent_summary(team_data)
    advanced = _advanced_metrics(team_data)
    summary = team_data.get("summary", {}) or {}

    if recent:
        recent_games = (
            _safe_int_value(recent.get("wins"))
            + _safe_int_value(recent.get("losses"))
            + _safe_int_value(recent.get("draws"))
        )
        draws_text = (
            f" {_safe_int_value(recent.get('draws'))}무" if recent.get("draws") else ""
        )
        fact = _build_fact_line(
            f"{team_name} 최근 흐름",
            (
                f"{recent.get('wins', 0)}승 {recent.get('losses', 0)}패"
                f"{draws_text}, "
                f"득실 {'+' if _safe_int_value(recent.get('run_diff')) >= 0 else ''}{_safe_int_value(recent.get('run_diff'))}, "
                f"샘플 {recent_games}경기"
            ),
        )
        if fact:
            fact_lines.append(fact)
            extend_numeric_tokens([fact], numeric_tokens)

    batting_metrics = advanced.get("metrics", {}).get("batting", {}) or {}
    ops_value = batting_metrics.get("ops")
    if ops_value is not None:
        try:
            ops_text = f"{float(ops_value):.3f}"
        except (TypeError, ValueError):
            ops_text = str(ops_value)
        fact = _build_fact_line(f"{team_name} 정규시즌 OPS", ops_text)
        if fact:
            fact_lines.append(fact)
            extend_numeric_tokens([fact], numeric_tokens)

    bullpen_share = advanced.get("fatigue_index", {}).get("bullpen_share")
    if bullpen_share:
        fact = _build_fact_line(f"{team_name} 불펜 비중", str(bullpen_share))
        if fact:
            fact_lines.append(fact)
            extend_numeric_tokens([fact], numeric_tokens)

    top_batters = summary.get("top_batters", [])[:2]
    for player in top_batters:
        player_name = _normalize_name_token(player.get("player_name"))
        if not player_name:
            continue
        stats_text = _extract_tool_player_stats(player)
        fact = _build_fact_line(f"{team_name} 주요 타자 {player_name}", stats_text)
        if fact:
            fact_lines.append(fact)
            extend_numeric_tokens([fact], numeric_tokens)

    top_pitchers = summary.get("top_pitchers", [])[:2]
    for player in top_pitchers:
        player_name = _normalize_name_token(player.get("player_name"))
        if not player_name:
            continue
        stats_text = _extract_tool_player_stats(player)
        fact = _build_fact_line(f"{team_name} 주요 투수 {player_name}", stats_text)
        if fact:
            fact_lines.append(fact)
            extend_numeric_tokens([fact], numeric_tokens)

    for signal_key, label in (("batters", "폼 타자"), ("pitchers", "폼 투수")):
        signal = _best_form_signal(team_data, signal_key)
        if not signal:
            continue
        player_name = _normalize_name_token(signal.get("player_name"))
        if not player_name:
            continue
        fact = _build_fact_line(
            f"{team_name} {label} {player_name}",
            _format_form_signal_fact(signal),
        )
        if fact:
            fact_lines.append(fact)
            extend_numeric_tokens([fact], numeric_tokens)


def _build_coach_fact_sheet(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
    allowed_names: Set[str],
    assessment: GameEvidenceAssessment,
) -> CoachFactSheet:
    fact_lines: List[str] = []
    caveat_lines: List[str] = []
    numeric_tokens: Set[str] = set()
    reasons = list(assessment.root_causes)

    base_facts = [
        _build_fact_line(
            "매치업",
            f"{evidence.away_team_name} vs {evidence.home_team_name}",
        ),
        _build_fact_line("경기 날짜", evidence.game_date or "미상"),
        _build_fact_line(
            "경기 상태", _build_game_status_label(evidence.game_status_bucket)
        ),
        _build_fact_line("리그 구분", evidence.round_display),
    ]
    if evidence.stadium_name:
        base_facts.append(_build_fact_line("구장", evidence.stadium_name))
    if evidence.start_time:
        base_facts.append(_build_fact_line("시작 시각", evidence.start_time))
    if evidence.weather:
        base_facts.append(_build_fact_line("날씨", evidence.weather))

    for fact in base_facts:
        if fact:
            fact_lines.append(fact)
            extend_numeric_tokens([fact], numeric_tokens)

    if _is_completed_review(evidence):
        scoreline = _completed_scoreline_text(evidence)
        if scoreline:
            fact = _build_fact_line("최종 스코어", scoreline)
            if fact:
                fact_lines.append(fact)
                extend_numeric_tokens([fact], numeric_tokens)
        winner_name = _completed_winner_name(evidence)
        if winner_name:
            fact = _build_fact_line("승리 팀", winner_name)
            if fact:
                fact_lines.append(fact)
        elif _completed_game_is_draw(evidence):
            fact = _build_fact_line("경기 결과", "무승부")
            if fact:
                fact_lines.append(fact)

    if evidence.series_state and evidence.stage_label != "REGULAR":
        series_text = evidence.series_state.summary_text(
            evidence.home_team_name,
            evidence.away_team_name,
        )
        fact = _build_fact_line("시리즈 전황", series_text)
        if fact:
            fact_lines.append(fact)
            extend_numeric_tokens([fact], numeric_tokens)
    elif evidence.stage_label not in {"REGULAR", "PRE", "UNKNOWN"}:
        caveat_lines.append("시리즈 전황 데이터가 부족합니다.")

    if evidence.home_pitcher and evidence.away_pitcher:
        fact = _build_fact_line(
            "발표 선발",
            f"{evidence.away_team_name} {evidence.away_pitcher} / {evidence.home_team_name} {evidence.home_pitcher}",
        )
        if fact:
            fact_lines.append(fact)
            extend_numeric_tokens([fact], numeric_tokens)
    else:
        caveat_lines.append("선발 정보가 완전히 확정되지 않았습니다.")

    if _has_confirmed_lineup_details(evidence):
        fact = _build_fact_line(
            "발표 라인업",
            (
                f"{evidence.away_team_name} [{_summarize_lineup_players(evidence.away_lineup, 4)}] / "
                f"{evidence.home_team_name} [{_summarize_lineup_players(evidence.home_lineup, 4)}]"
            ),
        )
        if fact:
            fact_lines.append(fact)
    elif evidence.lineup_announced:
        caveat_lines.append(
            "라인업 발표 신호는 있으나 선수 구성이 확인되지 않았습니다."
        )
    else:
        caveat_lines.append("라인업이 아직 발표되지 않았습니다.")

    for item in evidence.summary_items[:3]:
        fact = _build_fact_line("경기 요약", item)
        if fact:
            fact_lines.append(fact)
            extend_numeric_tokens([fact], numeric_tokens)
    if not evidence.summary_items:
        caveat_lines.append("경기 요약 근거가 부족합니다.")

    _append_team_fact_lines(
        fact_lines, numeric_tokens, tool_results, "away", evidence.away_team_name
    )
    _append_team_fact_lines(
        fact_lines, numeric_tokens, tool_results, "home", evidence.home_team_name
    )

    matchup = tool_results.get("matchup", {}) or {}
    matchup_summary = matchup.get("summary", {}) or {}
    if matchup_summary:
        if _matchup_is_partial(matchup):
            caveat_lines.append(
                "시리즈 맞대결 전적은 DB 이력 부족으로 축약 표시합니다."
            )
        else:
            fact = _build_fact_line(
                "맞대결 전적",
                (
                    f"{evidence.away_team_name} {matchup_summary.get('team2_wins', 0)}승 / "
                    f"{evidence.home_team_name} {matchup_summary.get('team1_wins', 0)}승 / "
                    f"{matchup_summary.get('draws', 0)}무"
                ),
            )
            if fact:
                fact_lines.append(fact)
                extend_numeric_tokens([fact], numeric_tokens)

    clutch_moments = _clutch_moments(tool_results)
    if clutch_moments:
        for moment in clutch_moments[:3]:
            fact = _build_fact_line("클러치 모먼트", _format_clutch_fact(moment))
            if fact:
                fact_lines.append(fact)
                extend_numeric_tokens([fact], numeric_tokens)
    elif _is_completed_review(evidence):
        reasons.append("missing_clutch_moments")
        caveat_lines.append("WPA 기반 승부처 데이터가 부족합니다.")

    fact_lines = list(dict.fromkeys(line for line in fact_lines if line))
    caveat_lines.extend(
        _grounding_reason_message(code)
        for code in reasons
        if code.startswith("missing_")
    )
    caveat_lines = list(dict.fromkeys(line for line in caveat_lines if line))
    extend_numeric_tokens(fact_lines, numeric_tokens)

    return CoachFactSheet(
        fact_lines=fact_lines,
        caveat_lines=caveat_lines,
        allowed_entity_names={name for name in allowed_names if name},
        allowed_numeric_tokens=numeric_tokens,
        supported_fact_count=len(fact_lines),
        starters_confirmed=assessment.home_pitcher_present
        and assessment.away_pitcher_present,
        lineup_confirmed=assessment.lineup_announced,
        series_context_confirmed=assessment.series_context_available,
        require_series_context=evidence.stage_label
        not in {"REGULAR", "PRE", "UNKNOWN"},
        reasons=_normalize_grounding_reasons(reasons),
        warnings=list(caveat_lines),
    )


def _normalize_stage_label(
    league_type_code: Optional[int], round_hint: Optional[str] = None
) -> str:
    if round_hint:
        normalized_hint = str(round_hint).strip().upper()
        if normalized_hint in {"PRE", "PRESEASON", "시범경기"}:
            return "PRE"
        if normalized_hint in {"KS", "KOREAN_SERIES", "한국시리즈"}:
            return "KS"
        if normalized_hint in {"PO", "PLAYOFF", "플레이오프"}:
            return "PO"
        if normalized_hint in {"SEMI_PO", "SEMIPO", "준플레이오프", "준PO", "DS"}:
            return "SEMI_PO"
        if normalized_hint in {"WC", "WILD_CARD", "와일드카드"}:
            return "WC"
        if normalized_hint in {"REGULAR", "정규시즌"}:
            return "REGULAR"
    return LEAGUE_TYPE_CODE_TO_STAGE.get(int(league_type_code or 0), "UNKNOWN")


def _resolve_league_type_code_hint(
    league_context: Optional[Dict[str, Any]],
) -> Optional[int]:
    context = league_context or {}
    explicit_code = _parse_optional_int(context.get("league_type_code"))
    if explicit_code is not None:
        return explicit_code

    stage_label = _normalize_stage_label(
        None,
        context.get("stage_label") or context.get("round"),
    )
    if stage_label in STAGE_TO_LEAGUE_TYPE_CODE:
        return STAGE_TO_LEAGUE_TYPE_CODE[stage_label]

    league_type = str(context.get("league_type") or "").strip().upper()
    if league_type == "PRE":
        return 1
    if league_type == "REGULAR":
        return 0
    return None


def _round_display_for_stage(stage_label: str) -> str:
    return STAGE_TO_ROUND_DISPLAY.get(stage_label, "경기")


def _normalize_raw_game_status(game_status: Optional[str]) -> str:
    return str(game_status or "").strip().upper()


def _normalize_game_status_bucket(game_status: Optional[str]) -> str:
    normalized = _normalize_raw_game_status(game_status)
    if normalized in {"COMPLETED", "FINAL", "FINISHED", "DONE", "END", "E", "F"}:
        return "COMPLETED"
    if normalized in {"LIVE", "IN_PROGRESS", "INPROGRESS", "PLAYING"}:
        return "LIVE"
    if normalized in {"SCHEDULED", "READY", "NOT_STARTED", "PENDING"}:
        return "SCHEDULED"
    if normalized in {"POSTPONED", "CANCELLED", "CANCELED", "SUSPENDED"}:
        return "UNKNOWN"
    return "UNKNOWN"


def _cache_ttl_seconds_for_status_bucket(game_status_bucket: str) -> int:
    if game_status_bucket == "COMPLETED":
        return COMPLETED_CACHE_TTL_SECONDS
    return VOLATILE_CACHE_TTL_SECONDS


def _normalize_name_token(value: Optional[str]) -> Optional[str]:
    normalized = " ".join(str(value or "").split()).strip()
    return normalized or None


def _is_scheduled_partial_context(
    game_status_bucket: str,
    grounding_reasons: List[str],
) -> bool:
    if game_status_bucket != "SCHEDULED":
        return False
    return any(
        reason in grounding_reasons for reason in SCHEDULED_PARTIAL_GROUNDING_REASONS
    )


def _clean_summary_text(value: Optional[str]) -> Optional[str]:
    text = " ".join(str(value or "").split()).strip()
    return text or None


def _has_batchim(text: str) -> bool:
    for char in reversed(text.strip()):
        if "가" <= char <= "힣":
            return (ord(char) - ord("가")) % 28 != 0
    return False


def _topic_particle(text: str) -> str:
    return "은" if _has_batchim(text) else "는"


def _join_with_korean_and(left: str, right: str) -> str:
    connector = "과" if _has_batchim(left) else "와"
    return f"{left}{connector} {right}"


def _format_game_summary_item(
    summary_type: Optional[str],
    player_name: Optional[str],
    detail_text: Optional[str],
) -> Optional[str]:
    summary_label = _clean_summary_text(summary_type)
    player = _normalize_name_token(player_name)
    detail = _clean_summary_text(detail_text)

    if detail and player and detail.startswith(player):
        detail = detail[len(player) :].strip()
        detail = detail.lstrip("/-:")
        detail = detail.strip()

    parts: List[str] = []
    if summary_label:
        parts.append(summary_label)
    if player:
        parts.append(player)

    base = " ".join(parts).strip()
    if detail:
        if detail.startswith("("):
            return _clean_summary_text(f"{base} {detail}")
        if "(" in detail and ")" in detail:
            return _clean_summary_text(f"{base} {detail}" if base else detail)
        return _clean_summary_text(f"{base} ({detail})" if base else detail)
    return base or None


def _ensure_sentence(value: Optional[str]) -> Optional[str]:
    text = _clean_summary_text(value)
    if not text:
        return None
    if text[-1] in ".!?。！？":
        return text
    return f"{text}."


def _truncate_text_naturally(
    value: Optional[str],
    *,
    max_length: int,
    preserve_newlines: bool = False,
) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not preserve_newlines:
        text = " ".join(text.split()).strip()
    if len(text) <= max_length:
        return text

    floor = max(max_length // 2, max_length - 160, 1)
    for idx in range(min(len(text), max_length), floor, -1):
        if text[idx - 1] in ".!?。！？\n":
            return text[:idx].rstrip()

    last_break = max(
        text.rfind(" ", 0, max_length - 2), text.rfind("\n", 0, max_length - 2)
    )
    if last_break >= floor:
        return text[:last_break].rstrip() + "..."
    return text[: max_length - 3].rstrip() + "..."


def _summarize_lineup_players(players: List[str], limit: int = 4) -> str:
    normalized = [player for player in players if player][:limit]
    if not normalized:
        return "라인업 발표 전"
    return ", ".join(normalized)


def _build_game_status_label(game_status_bucket: str) -> str:
    labels = {
        "SCHEDULED": "경기 전",
        "LIVE": "진행 중",
        "COMPLETED": "경기 종료",
        "UNKNOWN": "상태 확인 중",
    }
    return labels.get(game_status_bucket, "상태 확인 중")


def _get_unavailable_game_status_message(game_status: Optional[str]) -> Optional[str]:
    normalized = _normalize_raw_game_status(game_status)
    if normalized in {"CANCELLED", "CANCELED"}:
        return "취소된 경기는 AI 코치 분석을 제공하지 않습니다."
    if normalized == "POSTPONED":
        return "연기된 경기는 일정 확정 후 AI 코치 분석을 제공합니다."
    if normalized == "SUSPENDED":
        return "중단된 경기는 경기 상태가 확정된 뒤 AI 코치 분석을 제공합니다."
    return None


def _has_canonical_game_team_pair(
    home_team_code: Optional[str], away_team_code: Optional[str]
) -> bool:
    return bool(home_team_code in CANONICAL_CODES and away_team_code in CANONICAL_CODES)


def _is_completed_review_bucket(game_status_bucket: Optional[str]) -> bool:
    return _normalize_game_status_bucket(game_status_bucket) == "COMPLETED"


def _is_completed_review(evidence: GameEvidence) -> bool:
    return _is_completed_review_bucket(evidence.game_status_bucket)


def _has_game_row_context(evidence: GameEvidence) -> bool:
    if getattr(evidence, "game_row_found", False):
        return True

    game_id = getattr(evidence, "game_id", None)
    game_date = getattr(evidence, "game_date", None)
    season_year = getattr(evidence, "season_year", None)
    home_team_code = getattr(evidence, "home_team_code", None)
    away_team_code = getattr(evidence, "away_team_code", None)
    stage_label = getattr(evidence, "stage_label", "UNKNOWN")
    return bool(
        game_id
        and game_date
        and season_year
        and home_team_code
        and away_team_code
        and stage_label != "UNKNOWN"
    )


def _default_used_evidence(evidence: GameEvidence) -> List[str]:
    sources: List[str] = []
    if _has_game_row_context(evidence):
        sources.extend(["game", "kbo_seasons"])
    if evidence.stadium_name or evidence.start_time or evidence.weather:
        sources.append("game_metadata")
    if _has_confirmed_lineup_details(evidence):
        sources.append("game_lineups")
    if evidence.summary_items:
        sources.append("game_summary")
    if evidence.series_state and evidence.stage_label != "REGULAR":
        sources.append("series_history")
    return sources


def assess_game_evidence(evidence: GameEvidence) -> GameEvidenceAssessment:
    used_evidence = _default_used_evidence(evidence)
    home_pitcher_present = bool(evidence.home_pitcher)
    away_pitcher_present = bool(evidence.away_pitcher)
    lineup_announced = _has_confirmed_lineup_details(evidence)
    summary_present = bool(evidence.summary_items)
    metadata_present = bool(
        evidence.stadium_name or evidence.start_time or evidence.weather
    )
    series_required = evidence.stage_label not in {"REGULAR", "PRE", "UNKNOWN"}
    series_context_available = bool(evidence.series_state)
    has_game_row_context = _has_game_row_context(evidence)
    missing_game_context = not bool(
        has_game_row_context
        and evidence.game_id
        and evidence.game_date
        and evidence.home_team_code
        and evidence.away_team_code
        and evidence.season_year
        and evidence.stage_label != "UNKNOWN"
    )

    root_causes: List[str] = []
    if missing_game_context:
        root_causes.append("missing_game_context")
    if not (home_pitcher_present and away_pitcher_present):
        root_causes.append("missing_starters")
    if not lineup_announced:
        root_causes.append("missing_lineups")
    if not summary_present:
        root_causes.append("missing_summary")
    if not metadata_present:
        root_causes.append("missing_metadata")
    if series_required and not series_context_available:
        root_causes.append("missing_series_context")

    ordered_root_causes = [
        code for code in COACH_EVIDENCE_ROOT_CAUSE_ORDER if code in root_causes
    ]
    if missing_game_context or not evidence.game_id:
        expected_data_quality = "insufficient"
    elif ordered_root_causes:
        expected_data_quality = "partial"
    else:
        expected_data_quality = "grounded"

    return GameEvidenceAssessment(
        used_evidence=used_evidence,
        root_causes=ordered_root_causes,
        expected_data_quality=expected_data_quality,
        home_pitcher_present=home_pitcher_present,
        away_pitcher_present=away_pitcher_present,
        lineup_announced=lineup_announced,
        summary_present=summary_present,
        metadata_present=metadata_present,
        series_context_available=series_context_available,
    )


def _determine_data_quality(
    evidence: GameEvidence,
    tool_results: Optional[Dict[str, Any]] = None,
) -> str:
    assessment = assess_game_evidence(evidence)
    if tool_results is None:
        return assessment.expected_data_quality

    baseline_count = 0
    if tool_results.get("home", {}).get("summary", {}).get("found"):
        baseline_count += 1
    if tool_results.get("home", {}).get("advanced", {}).get("found"):
        baseline_count += 1
    if evidence.away_team_code:
        if tool_results.get("away", {}).get("summary", {}).get("found"):
            baseline_count += 1
        if tool_results.get("away", {}).get("advanced", {}).get("found"):
            baseline_count += 1

    if assessment.expected_data_quality == "insufficient":
        if baseline_count >= 2 and evidence.game_id:
            return "partial"
        return "insufficient"
    if assessment.expected_data_quality == "partial":
        return "partial"
    if _is_completed_review(evidence) and not tool_results.get(
        "clutch_moments", {}
    ).get("found"):
        return "partial"
    if evidence.game_id and baseline_count >= 2:
        return "grounded"
    return "partial"


def _manual_data_missing_item(
    key: str,
    label: str,
    reason: str,
    expected_format: str,
) -> Dict[str, str]:
    return {
        "key": key,
        "label": label,
        "reason": reason,
        "expected_format": expected_format,
    }


def _parse_iso_date(value: Optional[str]) -> Optional[date_cls]:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    try:
        return datetime.strptime(normalized[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _lookup_stage_start_date(
    pool: ConnectionPool,
    *,
    season_year: int,
    league_type_code: int,
) -> Optional[date_cls]:
    try:
        with pool.connection() as conn:
            cursor = conn.cursor(row_factory=dict_row)
            row = cursor.execute(
                """
                SELECT start_date
                FROM kbo_seasons
                WHERE season_year = %s
                  AND league_type_code = %s
                  AND start_date IS NOT NULL
                ORDER BY season_id DESC
                LIMIT 1
                """,
                (season_year, league_type_code),
            ).fetchone()
            if row and row.get("start_date") is not None:
                start_date = row["start_date"]
                if hasattr(start_date, "date"):
                    return start_date.date()
                return start_date

            fallback_row = cursor.execute(
                """
                SELECT MIN(g.game_date) AS start_date
                FROM game g
                LEFT JOIN kbo_seasons ks ON g.season_id = ks.season_id
                WHERE COALESCE(ks.season_year, EXTRACT(YEAR FROM g.game_date)::int) = %s
                  AND COALESCE(ks.league_type_code, 0) = %s
                """,
                (season_year, league_type_code),
            ).fetchone()
            if fallback_row and fallback_row.get("start_date") is not None:
                start_date = fallback_row["start_date"]
                if hasattr(start_date, "date"):
                    return start_date.date()
                return start_date
    except Exception as exc:
        logger.warning(
            "[Coach] Failed to look up stage start date season_year=%s league_type_code=%s: %s",
            season_year,
            league_type_code,
            exc,
        )
    return None


def _has_stage_context_mismatch(
    pool: ConnectionPool,
    evidence: GameEvidence,
    game_date: Optional[date_cls],
) -> bool:
    league_type_code = _parse_optional_int(getattr(evidence, "league_type_code", None))
    season_year = _parse_optional_int(getattr(evidence, "season_year", None))
    if (
        league_type_code is None
        or league_type_code < 2
        or league_type_code > 5
        or game_date is None
        or season_year is None
    ):
        return False

    stage_start = _lookup_stage_start_date(
        pool,
        season_year=season_year,
        league_type_code=league_type_code,
    )
    if stage_start is not None:
        return game_date < stage_start
    return game_date.month < 9


def _is_past_game_missing_final_state(
    game_date: Optional[date_cls],
    game_status: str,
    *,
    home_score: Optional[int] = None,
    away_score: Optional[int] = None,
) -> bool:
    if game_date is None or game_date >= datetime.now().date():
        return False

    normalized_status = str(game_status or "").strip().upper()
    scores_present = home_score is not None and away_score is not None
    if normalized_status in {
        "COMPLETED",
        "FINAL",
        "FINISHED",
        "DONE",
        "END",
        "E",
        "F",
        "DRAW",
        "TIE",
    }:
        return not scores_present
    return normalized_status in {
        "",
        "UNKNOWN",
        "SCHEDULED",
        "LIVE",
        "IN_PROGRESS",
        "INPROGRESS",
        "DELAYED",
        "SUSPENDED",
    }


def _build_manual_data_request(
    pool: ConnectionPool,
    payload: "AnalyzeRequest",
    evidence: GameEvidence,
    assessment: GameEvidenceAssessment,
) -> Optional[Dict[str, Any]]:
    missing_items: List[Dict[str, str]] = []
    seen_keys: Set[str] = set()

    def add_item(
        key: str,
        label: str,
        reason: str,
        expected_format: str,
    ) -> None:
        if key in seen_keys:
            return
        seen_keys.add(key)
        missing_items.append(
            _manual_data_missing_item(key, label, reason, expected_format)
        )

    evidence_game_id = getattr(evidence, "game_id", None)
    evidence_game_date = getattr(evidence, "game_date", None)
    evidence_season_year = getattr(evidence, "season_year", None)
    evidence_home_team_code = getattr(evidence, "home_team_code", None)
    evidence_away_team_code = getattr(evidence, "away_team_code", None)
    evidence_stage_label = getattr(evidence, "stage_label", "UNKNOWN")
    evidence_game_status = str(getattr(evidence, "game_status", "UNKNOWN"))
    evidence_home_score = getattr(evidence, "home_score", None)
    evidence_away_score = getattr(evidence, "away_score", None)

    request_game_id = str(payload.game_id or evidence_game_id or "").strip() or None
    request_game_date = _parse_iso_date(
        evidence_game_date or (payload.league_context or {}).get("game_date")
    )
    has_game_row = _has_game_row_context(evidence)
    missing_game_row = not has_game_row
    season_year_for_context = (
        (int(evidence_season_year) if evidence_season_year is not None else None)
        if request_game_date is not None
        else None
    )
    stage_context_mismatch = (
        request_game_date is not None
        and bool(evidence_season_year)
        and season_year_for_context is not None
        and season_year_for_context != request_game_date.year
    ) or _has_stage_context_mismatch(pool, evidence, request_game_date)
    missing_final_state = _is_past_game_missing_final_state(
        request_game_date,
        evidence_game_status,
        home_score=evidence_home_score,
        away_score=evidence_away_score,
    )
    missing_critical_context = not bool(
        has_game_row
        and request_game_date
        and evidence_home_team_code
        and evidence_away_team_code
        and evidence_stage_label != "UNKNOWN"
    )

    if missing_game_row:
        add_item(
            "game_id",
            "경기 ID",
            "요청한 경기의 game row가 없어 분석 대상을 식별할 수 없습니다.",
            "예: 20260405HHOB0",
        )
        add_item(
            "game_date",
            "경기 날짜",
            "요청한 경기의 일정 날짜가 필요합니다.",
            "YYYY-MM-DD",
        )

    if stage_context_mismatch or missing_critical_context:
        add_item(
            "season_league_context",
            "시즌/리그 구분",
            "시즌 연도 또는 리그 단계가 경기 날짜와 맞지 않아 경기 단계를 확정할 수 없습니다.",
            "season_id, season_year, league_type_code",
        )

    if missing_final_state:
        add_item(
            "game_status",
            "경기 상태",
            "과거 경기의 상태가 종료 기준으로 확정되지 않았습니다.",
            "SCHEDULED, COMPLETED, CANCELLED 등",
        )
        add_item(
            "final_score",
            "최종 점수",
            "과거 경기의 최종 점수가 비어 있습니다.",
            "home_score, away_score",
        )

    if missing_game_row or missing_critical_context:
        home_pitcher_present = bool(getattr(assessment, "home_pitcher_present", False))
        away_pitcher_present = bool(getattr(assessment, "away_pitcher_present", False))
        lineup_announced = bool(getattr(assessment, "lineup_announced", False))
        if not (home_pitcher_present and away_pitcher_present):
            add_item(
                "starters",
                "선발 정보",
                "분석에 필요한 선발 정보가 부족합니다.",
                "home_pitcher, away_pitcher",
            )
        if not lineup_announced:
            add_item(
                "lineup",
                "라인업",
                "분석에 필요한 선발 라인업 정보가 부족합니다.",
                "team_code, batting_order, player_name",
            )

    if not missing_items:
        return None

    identifier_parts: List[str] = []
    if request_game_id:
        identifier_parts.append(f"경기 ID={request_game_id}")
    if request_game_date is not None:
        identifier_parts.append(f"날짜={request_game_date.isoformat()}")
    identifier_parts.extend(
        f"{item['label']}({item['reason']})" for item in missing_items
    )

    logger.warning(
        "[Coach] manual_data_required scope=%s game_id=%s missing_keys=%s",
        "coach.analyze",
        request_game_id,
        [item["key"] for item in missing_items],
    )
    return {
        "scope": "coach.analyze",
        "missingItems": missing_items,
        "operatorMessage": "다음 야구 데이터가 필요합니다: "
        + ", ".join(identifier_parts),
        "blocking": True,
        "code": MANUAL_BASEBALL_DATA_REQUIRED_CODE,
    }


def _build_meta_payload_defaults(
    *,
    generation_mode: str,
    data_quality: str,
    used_evidence: List[str],
    cache_key: Optional[str] = None,
    resolved_cache_key: Optional[str] = None,
    expected_cache_key: Optional[str] = None,
    prompt_version: Optional[str] = None,
    starter_signature: Optional[str] = None,
    lineup_signature: Optional[str] = None,
    cache_key_mismatch: bool = False,
    game_status_bucket: Optional[str] = None,
    grounding_warnings: Optional[List[str]] = None,
    grounding_reasons: Optional[List[str]] = None,
    supported_fact_count: int = 0,
    cache_error_code: Optional[str] = None,
    retryable_failure: bool = False,
    dependency_degraded: bool = False,
    attempt_count: int = 0,
    cache_row_missing: bool = False,
    recovered_from_missing_row: bool = False,
    cache_finalize_conflict: bool = False,
    lease_lost: bool = False,
) -> Dict[str, Any]:
    safe_generation_mode = (
        generation_mode
        if generation_mode in COACH_META_GENERATION_MODES
        else "evidence_fallback"
    )
    safe_data_quality = (
        data_quality if data_quality in COACH_META_DATA_QUALITIES else "partial"
    )
    return {
        "generation_mode": safe_generation_mode,
        "data_quality": safe_data_quality,
        "cache_key": str(cache_key or resolved_cache_key or "").strip() or None,
        "resolved_cache_key": (
            str(resolved_cache_key or cache_key or "").strip() or None
        ),
        "expected_cache_key": str(expected_cache_key or "").strip() or None,
        "prompt_version": str(prompt_version or "").strip() or None,
        "starter_signature": str(starter_signature or "").strip() or None,
        "lineup_signature": str(lineup_signature or "").strip() or None,
        "cache_key_mismatch": bool(cache_key_mismatch),
        "game_status_bucket": _normalize_game_status_bucket(game_status_bucket),
        "used_evidence": list(used_evidence),
        "grounding_warnings": list(grounding_warnings or []),
        "grounding_reasons": _normalize_grounding_reasons(
            list(grounding_reasons or [])
        ),
        "supported_fact_count": max(0, int(supported_fact_count)),
        "cache_error_code": str(cache_error_code or "").strip() or None,
        "retryable_failure": bool(retryable_failure),
        "dependency_degraded": bool(dependency_degraded),
        "attempt_count": _normalize_attempt_count(attempt_count),
        "cache_row_missing": bool(cache_row_missing),
        "recovered_from_missing_row": bool(recovered_from_missing_row),
        "cache_finalize_conflict": bool(cache_finalize_conflict),
        "lease_lost": bool(lease_lost),
    }


def _default_generation_mode_for_request_mode(
    *,
    request_mode: str,
    generation_mode: Optional[str] = None,
    cache_state: Optional[str] = None,
    validation_status: Optional[str] = None,
) -> str:
    normalized_generation_mode = str(generation_mode or "").strip()
    if normalized_generation_mode in COACH_META_GENERATION_MODES:
        return normalized_generation_mode

    if request_mode == COACH_REQUEST_MODE_AUTO:
        normalized_cache_state = str(cache_state or "").strip().upper()
        normalized_validation_status = str(validation_status or "").strip().lower()
        if (
            normalized_validation_status == "success"
            and normalized_cache_state
            not in {
                "FAILED_LOCKED",
                "FAILED_RETRY_WAIT",
                "PENDING_WAIT",
            }
        ):
            return "deterministic_auto"
        return "evidence_fallback"

    return "evidence_fallback"


def _ensure_stream_meta_contract(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    request_mode = str(normalized.get("request_mode") or "").strip()
    if request_mode != COACH_REQUEST_MODE_AUTO:
        return normalized

    normalized["generation_mode"] = _default_generation_mode_for_request_mode(
        request_mode=request_mode,
        generation_mode=normalized.get("generation_mode"),
        cache_state=normalized.get("cache_state"),
        validation_status=normalized.get("validation_status"),
    )
    data_quality = str(normalized.get("data_quality") or "").strip()
    normalized["data_quality"] = (
        data_quality if data_quality in COACH_META_DATA_QUALITIES else "partial"
    )
    normalized["cache_state"] = (
        str(normalized.get("cache_state") or "MISS_GENERATE").strip().upper()
    )
    normalized["cached"] = bool(normalized.get("cached"))
    normalized["in_progress"] = bool(normalized.get("in_progress"))
    normalized["used_evidence"] = list(normalized.get("used_evidence") or [])
    normalized["grounding_warnings"] = list(normalized.get("grounding_warnings") or [])
    normalized["grounding_reasons"] = _normalize_grounding_reasons(
        list(normalized.get("grounding_reasons") or [])
    )
    normalized["supported_fact_count"] = max(
        0, int(normalized.get("supported_fact_count") or 0)
    )
    return normalized


def _cached_payload_meta(cached_data: Any) -> Dict[str, Any]:
    payload = cached_data
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return {}
    if not isinstance(payload, dict):
        return {}
    meta = payload.get("_meta")
    return meta if isinstance(meta, dict) else {}


def _should_regenerate_completed_cache(
    *,
    cached_data: Any,
    request_mode: str,
) -> bool:
    if request_mode != COACH_REQUEST_MODE_MANUAL:
        return False
    payload = cached_data
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return False
    response = payload.get("response") if isinstance(payload, dict) else None
    if isinstance(response, dict) and response_is_semantically_empty(response):
        return True
    return False


def _should_short_circuit_to_deterministic_response(
    *,
    request_mode: str,
    fact_sheet: CoachFactSheet,
    game_status_bucket: str = "UNKNOWN",
    grounding_reasons: Optional[List[str]] = None,
) -> bool:
    if request_mode == COACH_REQUEST_MODE_AUTO:
        return True
    if _is_scheduled_partial_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ) and (not fact_sheet.starters_confirmed and not fact_sheet.lineup_confirmed):
        return True
    if _is_completed_manual_review_request(
        request_mode,
        game_status_bucket=game_status_bucket,
    ):
        return True
    return fact_sheet.supported_fact_count <= 0


def _is_scheduled_partial_manual_request(
    request_mode: str,
    *,
    game_status_bucket: str,
    grounding_reasons: Optional[List[str]] = None,
) -> bool:
    if request_mode != COACH_REQUEST_MODE_MANUAL:
        return False
    return _is_scheduled_partial_context(
        game_status_bucket,
        list(grounding_reasons or []),
    )


def _is_scheduled_half_confirmed_manual_request(
    request_mode: str,
    *,
    game_status_bucket: str,
    grounding_reasons: Optional[List[str]] = None,
) -> bool:
    if not _is_scheduled_partial_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ):
        return False
    reasons = set(grounding_reasons or [])
    has_missing_starters = "missing_starters" in reasons
    has_missing_lineups = "missing_lineups" in reasons
    return has_missing_starters ^ has_missing_lineups


def _is_completed_manual_review_request(
    request_mode: str,
    *,
    game_status_bucket: str,
) -> bool:
    return request_mode == COACH_REQUEST_MODE_MANUAL and _is_completed_review_bucket(
        game_status_bucket
    )


def _resolve_coach_llm_attempt_limit(
    request_mode: str,
    *,
    game_status_bucket: str = "UNKNOWN",
    grounding_reasons: Optional[List[str]] = None,
) -> int:
    if request_mode == COACH_REQUEST_MODE_AUTO:
        return 1
    if _is_scheduled_partial_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ):
        return 1
    if _is_completed_manual_review_request(
        request_mode,
        game_status_bucket=game_status_bucket,
    ):
        return 1
    return 2


def _resolve_coach_llm_max_tokens(
    default_max_tokens: int,
    *,
    request_mode: str,
    game_status_bucket: str,
    grounding_reasons: Optional[List[str]] = None,
) -> int:
    effective = max(256, int(default_max_tokens))
    if _is_scheduled_half_confirmed_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ):
        return min(effective, COACH_SCHEDULED_HALF_CONFIRMED_MANUAL_MAX_TOKENS)
    if _is_scheduled_partial_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ):
        return min(effective, COACH_SCHEDULED_PARTIAL_MANUAL_MAX_TOKENS)
    if _is_completed_manual_review_request(
        request_mode,
        game_status_bucket=game_status_bucket,
    ):
        return min(effective, COACH_COMPLETED_MANUAL_MAX_TOKENS)
    return effective


def _resolve_coach_llm_idle_timeout_seconds(
    default_timeout_seconds: float,
    *,
    request_mode: str,
    game_status_bucket: str,
    grounding_reasons: Optional[List[str]] = None,
) -> float:
    base_timeout = max(
        COACH_LLM_STATUS_HEARTBEAT_SECONDS + 1.0,
        float(default_timeout_seconds),
    )
    if _is_scheduled_half_confirmed_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ):
        return min(
            base_timeout, COACH_SCHEDULED_HALF_CONFIRMED_MANUAL_IDLE_TIMEOUT_SECONDS
        )
    if _is_scheduled_partial_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ):
        return min(base_timeout, COACH_SCHEDULED_PARTIAL_MANUAL_IDLE_TIMEOUT_SECONDS)
    if _is_completed_manual_review_request(
        request_mode,
        game_status_bucket=game_status_bucket,
    ):
        return min(base_timeout, COACH_COMPLETED_MANUAL_IDLE_TIMEOUT_SECONDS)
    return base_timeout


def _resolve_coach_llm_request_timeout_seconds(
    *,
    request_mode: str,
    game_status_bucket: str,
    grounding_reasons: Optional[List[str]] = None,
) -> float:
    if _is_scheduled_half_confirmed_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ):
        return COACH_SCHEDULED_HALF_CONFIRMED_MANUAL_REQUEST_TIMEOUT_SECONDS
    if _is_scheduled_partial_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ):
        return COACH_SCHEDULED_PARTIAL_MANUAL_REQUEST_TIMEOUT_SECONDS
    if _is_completed_manual_review_request(
        request_mode,
        game_status_bucket=game_status_bucket,
    ):
        return COACH_COMPLETED_MANUAL_REQUEST_TIMEOUT_SECONDS
    return 120.0


def _resolve_coach_llm_total_timeout_seconds(
    *,
    request_mode: str,
    game_status_bucket: str,
    grounding_reasons: Optional[List[str]] = None,
) -> Optional[float]:
    if _is_scheduled_half_confirmed_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ):
        return COACH_SCHEDULED_HALF_CONFIRMED_MANUAL_TOTAL_TIMEOUT_SECONDS
    if _is_scheduled_partial_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ):
        return COACH_SCHEDULED_PARTIAL_MANUAL_TOTAL_TIMEOUT_SECONDS
    if _is_completed_manual_review_request(
        request_mode,
        game_status_bucket=game_status_bucket,
    ):
        return COACH_COMPLETED_MANUAL_TOTAL_TIMEOUT_SECONDS
    return None


def _resolve_coach_llm_first_chunk_timeout_seconds(
    *,
    request_mode: str,
    game_status_bucket: str,
    grounding_reasons: Optional[List[str]] = None,
) -> Optional[float]:
    if _is_scheduled_half_confirmed_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ):
        return COACH_SCHEDULED_HALF_CONFIRMED_MANUAL_FIRST_CHUNK_TIMEOUT_SECONDS
    return None


def _resolve_coach_empty_chunk_retry_limit(
    *,
    request_mode: str,
    game_status_bucket: str,
    grounding_reasons: Optional[List[str]] = None,
) -> int:
    if _is_scheduled_half_confirmed_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ):
        return COACH_SCHEDULED_HALF_CONFIRMED_MANUAL_EMPTY_CHUNK_RETRIES
    if _is_scheduled_partial_manual_request(
        request_mode,
        game_status_bucket=game_status_bucket,
        grounding_reasons=grounding_reasons,
    ):
        return COACH_SCHEDULED_PARTIAL_MANUAL_EMPTY_CHUNK_RETRIES
    if _is_completed_manual_review_request(
        request_mode,
        game_status_bucket=game_status_bucket,
    ):
        return COACH_COMPLETED_MANUAL_EMPTY_CHUNK_RETRIES
    return -1


def _wrap_cached_payload(
    response_data: Dict[str, Any], meta: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "response": response_data,
        "_meta": {
            "generation_mode": meta.get("generation_mode"),
            "data_quality": meta.get("data_quality"),
            "cache_key": meta.get("cache_key"),
            "resolved_cache_key": meta.get("resolved_cache_key"),
            "expected_cache_key": meta.get("expected_cache_key"),
            "prompt_version": meta.get("prompt_version"),
            "starter_signature": meta.get("starter_signature"),
            "lineup_signature": meta.get("lineup_signature"),
            "cache_key_mismatch": bool(meta.get("cache_key_mismatch")),
            "game_status_bucket": meta.get("game_status_bucket"),
            "used_evidence": list(meta.get("used_evidence") or []),
            "grounding_warnings": list(meta.get("grounding_warnings") or []),
            "grounding_reasons": list(meta.get("grounding_reasons") or []),
            "supported_fact_count": int(meta.get("supported_fact_count") or 0),
            "cache_error_code": meta.get("cache_error_code"),
            "retryable_failure": bool(meta.get("retryable_failure")),
            "dependency_degraded": bool(meta.get("dependency_degraded")),
            "attempt_count": _normalize_attempt_count(meta.get("attempt_count")),
            "cache_row_missing": bool(meta.get("cache_row_missing")),
            "recovered_from_missing_row": bool(meta.get("recovered_from_missing_row")),
            "cache_finalize_conflict": bool(meta.get("cache_finalize_conflict")),
            "lease_lost": bool(meta.get("lease_lost")),
        },
    }


def _extract_cached_payload(
    cached_data: Dict[str, Any],
    *,
    request_mode: str = COACH_REQUEST_MODE_MANUAL,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(cached_data, dict) and isinstance(cached_data.get("response"), dict):
        response = cached_data["response"]
        meta = cached_data.get("_meta", {})
    else:
        response = cached_data
        meta = {}
    normalized = _normalize_cached_response(response)
    meta_defaults = _build_meta_payload_defaults(
        generation_mode=_default_generation_mode_for_request_mode(
            request_mode=request_mode,
            generation_mode=meta.get("generation_mode"),
            validation_status="success",
        ),
        data_quality=str(meta.get("data_quality") or "partial"),
        used_evidence=list(meta.get("used_evidence") or []),
        cache_key=meta.get("cache_key"),
        resolved_cache_key=meta.get("resolved_cache_key"),
        expected_cache_key=meta.get("expected_cache_key"),
        prompt_version=meta.get("prompt_version"),
        starter_signature=meta.get("starter_signature"),
        lineup_signature=meta.get("lineup_signature"),
        cache_key_mismatch=bool(meta.get("cache_key_mismatch")),
        game_status_bucket=str(meta.get("game_status_bucket") or "UNKNOWN"),
        grounding_warnings=list(meta.get("grounding_warnings") or []),
        grounding_reasons=list(meta.get("grounding_reasons") or []),
        supported_fact_count=int(meta.get("supported_fact_count") or 0),
        cache_error_code=meta.get("cache_error_code"),
        retryable_failure=bool(meta.get("retryable_failure")),
        dependency_degraded=bool(meta.get("dependency_degraded")),
        attempt_count=int(meta.get("attempt_count") or 0),
        cache_row_missing=bool(meta.get("cache_row_missing")),
        recovered_from_missing_row=bool(meta.get("recovered_from_missing_row")),
        cache_finalize_conflict=bool(meta.get("cache_finalize_conflict")),
        lease_lost=bool(meta.get("lease_lost")),
    )
    return normalized, meta_defaults


def _merge_cache_contract_meta(
    meta_payload: Dict[str, Any],
    contract_meta: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(meta_payload)
    for key, value in contract_meta.items():
        if merged.get(key) in (None, "", []):
            merged[key] = value
    return merged


def _cache_status_response(
    *,
    headline: str,
    coach_note: str,
    detail: str,
) -> Dict[str, Any]:
    return {
        "headline": headline,
        "sentiment": "neutral",
        "key_metrics": [],
        "analysis": {
            "strengths": [],
            "weaknesses": [],
            "risks": [],
        },
        "detailed_markdown": detail,
        "coach_note": coach_note,
    }


def _sanitize_cache_error_code(value: Any) -> str:
    if isinstance(value, str) and value in ALLOWED_COACH_CACHE_ERROR_CODES:
        return value
    return COACH_INTERNAL_ERROR_CODE


def _cache_error_message_for_user(code: str) -> str:
    normalized = _sanitize_cache_error_code(code)
    return COACH_CACHE_ERROR_MESSAGES.get(
        normalized, COACH_CACHE_ERROR_MESSAGES[COACH_INTERNAL_ERROR_CODE]
    )


def _coach_public_error_payload(
    code: str = COACH_INTERNAL_ERROR_CODE,
) -> Dict[str, str]:
    normalized = _sanitize_cache_error_code(code)
    message = (
        COACH_INTERNAL_ERROR_MESSAGE
        if normalized == COACH_INTERNAL_ERROR_CODE
        else _cache_error_message_for_user(normalized)
    )
    return {
        "code": normalized,
        "message": message,
    }


def _calc_row_age_seconds(updated_at: Any) -> float:
    if not isinstance(updated_at, datetime):
        return float("inf")
    now_ref = (
        datetime.now(updated_at.tzinfo)
        if updated_at.tzinfo is not None
        else datetime.now()
    )
    return max(0.0, (now_ref - updated_at).total_seconds())


def _normalize_attempt_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _is_retryable_cache_error_code(code: Any) -> bool:
    return str(code or "").strip().lower() in RETRYABLE_CACHE_ERROR_CODES


def _cache_error_code_from_exception(exc: Exception) -> str:
    text = str(exc or "").lower()
    if "empty response" in text or "empty_choices" in text:
        return COACH_EMPTY_RESPONSE_ERROR_CODE
    if "no json found" in text:
        return COACH_NO_JSON_FOUND_ERROR_CODE
    if "json decode error" in text or "invalid control character" in text:
        return COACH_JSON_DECODE_ERROR_CODE
    if (
        "failed to load mappings from oci" in text
        or "server closed the connection unexpectedly" in text
        or "oci" in text
        and "mapping" in text
    ):
        return COACH_OCI_MAPPING_DEGRADED_ERROR_CODE
    if "unsupported_entity_name" in text:
        return COACH_UNSUPPORTED_ENTITY_ERROR_CODE
    if "grounding_validation_failed" in text:
        return COACH_UNSUPPORTED_CLAIM_ERROR_CODE
    return COACH_INTERNAL_ERROR_CODE


def _cache_error_code_from_parse_meta(parse_meta: Dict[str, Any]) -> str:
    code = str(parse_meta.get("error_code") or "").strip().lower()
    if code in {
        COACH_EMPTY_RESPONSE_ERROR_CODE,
        COACH_NO_JSON_FOUND_ERROR_CODE,
        COACH_JSON_DECODE_ERROR_CODE,
    }:
        return code
    return COACH_INTERNAL_ERROR_CODE


async def _iterate_coach_llm_with_keepalive(
    *,
    coach_llm: Any,
    messages: List[Dict[str, str]],
    max_tokens: int,
    heartbeat_seconds: float,
    idle_timeout_seconds: float,
    first_chunk_timeout_seconds: Optional[float] = None,
    coach_llm_kwargs: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Dict[str, str]]:
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def _produce_chunks() -> None:
        try:
            async for chunk in coach_llm(
                messages=messages,
                max_tokens=max_tokens,
                **(coach_llm_kwargs or {}),
            ):
                await queue.put(("chunk", chunk))
        except Exception as exc:  # noqa: BLE001
            await queue.put(("error", exc))
        finally:
            await queue.put(("done", None))

    producer_task = asyncio.create_task(_produce_chunks())
    last_model_activity = perf_counter()
    stream_started = perf_counter()
    received_first_chunk = False
    effective_heartbeat_seconds = max(0.1, float(heartbeat_seconds))
    effective_idle_timeout_seconds = max(
        effective_heartbeat_seconds,
        float(idle_timeout_seconds),
    )

    try:
        while True:
            idle_elapsed = perf_counter() - last_model_activity
            remaining_idle = effective_idle_timeout_seconds - idle_elapsed
            if remaining_idle <= 0:
                raise TimeoutError(
                    f"coach_llm_stream_timeout_after_{int(effective_idle_timeout_seconds)}s"
                )

            remaining_first_chunk: Optional[float] = None
            if not received_first_chunk and first_chunk_timeout_seconds is not None:
                remaining_first_chunk = float(first_chunk_timeout_seconds) - (
                    perf_counter() - stream_started
                )
                if remaining_first_chunk <= 0:
                    raise TimeoutError(
                        f"coach_llm_first_chunk_timeout_after_{int(float(first_chunk_timeout_seconds))}s"
                    )

            wait_timeout = min(
                effective_heartbeat_seconds,
                remaining_idle,
                (
                    remaining_first_chunk
                    if remaining_first_chunk is not None
                    else effective_heartbeat_seconds
                ),
            )
            try:
                event_type, payload = await asyncio.wait_for(
                    queue.get(),
                    timeout=wait_timeout,
                )
            except asyncio.TimeoutError:
                yield {
                    "type": "status",
                    "message": "AI 코치가 근거를 정리하는 중입니다...",
                }
                continue

            if event_type == "chunk":
                last_model_activity = perf_counter()
                received_first_chunk = True
                yield {"type": "chunk", "chunk": str(payload)}
                continue

            if event_type == "error":
                raise payload

            if event_type == "done":
                return
    finally:
        if not producer_task.done():
            producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass


def _calc_pending_reference_age_seconds(
    updated_at: Any,
    last_heartbeat_at: Any,
    lease_expires_at: Any,
) -> float:
    age_candidates = [
        _calc_row_age_seconds(updated_at),
        _calc_row_age_seconds(last_heartbeat_at),
    ]
    lease_age = _calc_row_age_seconds(lease_expires_at)
    if lease_age != float("inf"):
        age_candidates.append(lease_age)
    return min(age_candidates) if age_candidates else float("inf")


def _is_pending_cache_stale(
    *,
    updated_at: Any,
    last_heartbeat_at: Any,
    lease_expires_at: Any,
    pending_stale_seconds: int,
) -> bool:
    if isinstance(lease_expires_at, datetime):
        now_ref = (
            datetime.now(lease_expires_at.tzinfo)
            if lease_expires_at.tzinfo is not None
            else datetime.now()
        )
        if now_ref > lease_expires_at:
            return True

    age_seconds = _calc_pending_reference_age_seconds(
        updated_at, last_heartbeat_at, lease_expires_at
    )
    return age_seconds > pending_stale_seconds


def _determine_cache_gate(
    *,
    status: Optional[str],
    has_cached_json: bool,
    updated_at: Any,
    error_code: Any = None,
    attempt_count: Any = 0,
    last_heartbeat_at: Any = None,
    lease_expires_at: Any = None,
    pending_stale_seconds: int = PENDING_STALE_SECONDS,
    completed_ttl_seconds: Optional[int] = None,
) -> str:
    if status == "COMPLETED" and has_cached_json:
        if completed_ttl_seconds is not None:
            age_seconds = _calc_row_age_seconds(updated_at)
            if age_seconds > completed_ttl_seconds:
                return "MISS_GENERATE"
        return "HIT"
    if status == "PENDING":
        if _is_pending_cache_stale(
            updated_at=updated_at,
            last_heartbeat_at=last_heartbeat_at,
            lease_expires_at=lease_expires_at,
            pending_stale_seconds=pending_stale_seconds,
        ):
            return "PENDING_STALE_TAKEOVER"
        return "PENDING_WAIT"
    if status == "FAILED":
        age_seconds = _calc_row_age_seconds(updated_at)
        attempts = _normalize_attempt_count(attempt_count)
        if str(error_code or "").strip().lower() == COACH_STREAM_CANCELLED_ERROR_CODE:
            return "MISS_GENERATE"
        if _is_retryable_cache_error_code(error_code) and (
            attempts >= COACH_CACHE_MAX_RETRYABLE_ATTEMPTS
        ):
            return "FAILED_LOCKED"
        if (
            _is_retryable_cache_error_code(error_code)
            and attempts < COACH_CACHE_MAX_RETRYABLE_ATTEMPTS
            and age_seconds > FAILED_RETRY_AFTER_SECONDS
        ):
            return "MISS_GENERATE"
        if _is_retryable_cache_error_code(error_code):
            return "FAILED_RETRY_WAIT"
        return "FAILED_LOCKED"
    return "MISS_GENERATE"


def _should_generate_from_gate(gate: str) -> bool:
    return gate in {"MISS_GENERATE", "PENDING_STALE_TAKEOVER", "ROW_RECREATED"}


async def _wait_for_cache_terminal_state(
    pool: ConnectionPool,
    cache_key: str,
    timeout_seconds: float = PENDING_WAIT_TIMEOUT_SECONDS,
    poll_ms: int = PENDING_WAIT_POLL_MS,
) -> Optional[Dict[str, Any]]:
    deadline = perf_counter() + timeout_seconds
    sleep_seconds = max(float(poll_ms), 1.0) / 1000.0

    while perf_counter() < deadline:
        await asyncio.sleep(sleep_seconds)
        try:
            with pool.connection() as conn:
                row = conn.execute(
                    """
                    SELECT status, response_json, error_message, error_code, attempt_count, updated_at,
                           lease_expires_at, last_heartbeat_at
                    FROM coach_analysis_cache
                    WHERE cache_key = %s
                    """,
                    (cache_key,),
                ).fetchone()
            if not row:
                return {"status": "MISSING_ROW"}
            (
                status,
                cached_json,
                error_message,
                error_code,
                attempt_count,
                updated_at,
                lease_expires_at,
                last_heartbeat_at,
            ) = row
            if status == "COMPLETED" and cached_json:
                return {
                    "status": "COMPLETED",
                    "response_json": cached_json,
                }
            if status == "FAILED":
                return {
                    "status": "FAILED",
                    "error_message": error_message,
                    "error_code": error_code,
                    "attempt_count": _normalize_attempt_count(attempt_count),
                    "updated_at": updated_at,
                }
            if status == "PENDING" and _is_pending_cache_stale(
                updated_at=updated_at,
                last_heartbeat_at=last_heartbeat_at,
                lease_expires_at=lease_expires_at,
                pending_stale_seconds=COACH_CACHE_LEASE_STALE_SECONDS,
            ):
                return {
                    "status": "PENDING_STALE_TAKEOVER",
                }
        except Exception as exc:
            logger.warning("[Coach] Cache wait poll failed for %s: %s", cache_key, exc)
            return None
    return None


def _build_cache_lease_owner() -> str:
    return f"coach-{uuid.uuid4().hex}"


def _dependency_degraded_from_tool_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and bool(payload.get("_dependency_degraded"))


def _collect_dependency_degraded(tool_results: Dict[str, Any]) -> bool:
    return any(
        _dependency_degraded_from_tool_payload(tool_results.get(key))
        for key in ("home", "away", "matchup")
    )


def _collect_dependency_degraded_reasons(tool_results: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    for key in ("home", "away", "matchup"):
        payload = tool_results.get(key)
        if isinstance(payload, dict):
            reason = str(payload.get("_dependency_degraded_reason") or "").strip()
            if reason:
                reasons.append(reason)
    return list(dict.fromkeys(reasons))


def _insert_pending_cache_row(
    conn: Any,
    *,
    cache_key: str,
    team_id: str,
    year: int,
    prompt_version: str,
    model_name: str,
    lease_owner: str,
    attempt_count: int,
) -> bool:
    inserted = conn.execute(
        """
        INSERT INTO coach_analysis_cache (
            cache_key, team_id, year, prompt_version, model_name, status,
            error_message, error_code, attempt_count,
            lease_owner, lease_expires_at, last_heartbeat_at, updated_at
        ) VALUES (
            %s, %s, %s, %s, %s, 'PENDING',
            NULL, NULL, %s,
            %s, now() + make_interval(secs => %s), now(), now()
        )
        ON CONFLICT (cache_key) DO NOTHING
        RETURNING cache_key
        """,
        (
            cache_key,
            team_id,
            year,
            prompt_version,
            model_name,
            max(1, _normalize_attempt_count(attempt_count)),
            lease_owner,
            COACH_CACHE_LEASE_STALE_SECONDS,
        ),
    ).fetchone()
    return bool(inserted)


async def _heartbeat_cache_lease(
    *,
    pool: ConnectionPool,
    cache_key: str,
    lease_owner: str,
    lease_lost_event: Optional[asyncio.Event] = None,
) -> None:
    while True:
        await asyncio.sleep(COACH_CACHE_HEARTBEAT_INTERVAL_SECONDS)
        try:
            with pool.connection() as conn:
                result = conn.execute(
                    """
                    UPDATE coach_analysis_cache
                    SET updated_at = now(),
                        last_heartbeat_at = now(),
                        lease_expires_at = now() + make_interval(secs => %s)
                    WHERE cache_key = %s
                      AND status = 'PENDING'
                      AND lease_owner = %s
                    """,
                    (
                        COACH_CACHE_LEASE_STALE_SECONDS,
                        cache_key,
                        lease_owner,
                    ),
                )
                conn.commit()
                if not result.rowcount:
                    if lease_lost_event is not None:
                        lease_lost_event.set()
                    logger.warning(
                        "[Coach] Cache lease lost for %s owner=%s",
                        cache_key,
                        lease_owner,
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "[Coach] Cache heartbeat update failed for %s: %s", cache_key, exc
            )


async def _cancel_heartbeat_task(task: Optional[asyncio.Task]) -> None:
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _claim_cache_generation(
    *,
    pool: ConnectionPool,
    cache_key: str,
    team_id: str,
    year: int,
    prompt_version: str,
    model_name: str,
    lease_owner: str,
    completed_ttl_seconds: Optional[int],
    request_mode: str = COACH_REQUEST_MODE_MANUAL,
) -> tuple[str, Any, Optional[str], Optional[str], int]:
    with pool.connection() as conn:
        with conn.transaction():
            inserted = _insert_pending_cache_row(
                conn,
                cache_key=cache_key,
                team_id=team_id,
                year=year,
                prompt_version=prompt_version,
                model_name=model_name,
                lease_owner=lease_owner,
                attempt_count=1,
            )
            row = conn.execute(
                """
                SELECT status, response_json, error_message, error_code, attempt_count,
                       updated_at, lease_expires_at, last_heartbeat_at
                FROM coach_analysis_cache
                WHERE cache_key = %s
                FOR UPDATE
                """,
                (cache_key,),
            ).fetchone()

            if inserted:
                return "MISS_GENERATE", None, None, None, 1

            if not row:
                recreated = _insert_pending_cache_row(
                    conn,
                    cache_key=cache_key,
                    team_id=team_id,
                    year=year,
                    prompt_version=prompt_version,
                    model_name=model_name,
                    lease_owner=lease_owner,
                    attempt_count=1,
                )
                if recreated:
                    return "ROW_RECREATED", None, None, None, 1

                row = conn.execute(
                    """
                    SELECT status, response_json, error_message, error_code, attempt_count,
                           updated_at, lease_expires_at, last_heartbeat_at
                    FROM coach_analysis_cache
                    WHERE cache_key = %s
                    FOR UPDATE
                    """,
                    (cache_key,),
                ).fetchone()
                if not row:
                    return "MISS_GENERATE", None, None, None, 1

            (
                status,
                cached_json,
                error_message,
                error_code,
                attempt_count,
                updated_at,
                lease_expires_at,
                last_heartbeat_at,
            ) = row

            cache_state = _determine_cache_gate(
                status=status,
                has_cached_json=bool(cached_json),
                updated_at=updated_at,
                error_code=error_code,
                attempt_count=attempt_count,
                last_heartbeat_at=last_heartbeat_at,
                lease_expires_at=lease_expires_at,
                completed_ttl_seconds=completed_ttl_seconds,
            )
            if cache_state == "HIT" and _should_regenerate_completed_cache(
                cached_data=cached_json,
                request_mode=request_mode,
            ):
                cache_state = "MISS_GENERATE"

            if cache_state == "HIT":
                return (
                    cache_state,
                    cached_json,
                    error_message,
                    error_code,
                    _normalize_attempt_count(attempt_count),
                )

            if cache_state in {"MISS_GENERATE", "PENDING_STALE_TAKEOVER"}:
                next_attempt = max(1, _normalize_attempt_count(attempt_count) + 1)
                conn.execute(
                    """
                    UPDATE coach_analysis_cache
                    SET status = 'PENDING',
                        team_id = %s,
                        year = %s,
                        prompt_version = %s,
                        model_name = %s,
                        error_message = NULL,
                        error_code = NULL,
                        attempt_count = %s,
                        lease_owner = %s,
                        lease_expires_at = now() + make_interval(secs => %s),
                        last_heartbeat_at = now(),
                        updated_at = now()
                    WHERE cache_key = %s
                    """,
                    (
                        team_id,
                        year,
                        prompt_version,
                        model_name,
                        next_attempt,
                        lease_owner,
                        COACH_CACHE_LEASE_STALE_SECONDS,
                        cache_key,
                    ),
                )
                return cache_state, None, None, None, next_attempt

            return (
                cache_state,
                None,
                error_message,
                str(error_code or ""),
                _normalize_attempt_count(attempt_count),
            )


def _store_completed_cache(
    *,
    pool: ConnectionPool,
    cache_key: str,
    lease_owner: str,
    team_id: str,
    year: int,
    prompt_version: str,
    model_name: str,
    response_payload: Dict[str, Any],
    meta_defaults: Dict[str, Any],
) -> Dict[str, Any]:
    wrapped_payload = json.dumps(
        _wrap_cached_payload(response_payload, meta_defaults),
        ensure_ascii=False,
    )
    with pool.connection() as conn:
        result = conn.execute(
            """
            UPDATE coach_analysis_cache
            SET status = 'COMPLETED',
                response_json = %s,
                error_message = NULL,
                error_code = NULL,
                lease_owner = NULL,
                lease_expires_at = NULL,
                last_heartbeat_at = NULL,
                updated_at = now()
            WHERE cache_key = %s
              AND status = 'PENDING'
              AND lease_owner = %s
            """,
            (
                wrapped_payload,
                cache_key,
                lease_owner,
            ),
        )
        if result.rowcount:
            conn.commit()
            return {"outcome": "updated"}

        row = conn.execute(
            """
            SELECT status, response_json, lease_owner
            FROM coach_analysis_cache
            WHERE cache_key = %s
            """,
            (cache_key,),
        ).fetchone()
        if not row:
            inserted = conn.execute(
                """
                INSERT INTO coach_analysis_cache (
                    cache_key, team_id, year, prompt_version, model_name, status,
                    response_json, error_message, error_code, attempt_count,
                    lease_owner, lease_expires_at, last_heartbeat_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, 'COMPLETED',
                    %s, NULL, NULL, %s,
                    NULL, NULL, NULL, now()
                )
                ON CONFLICT (cache_key) DO NOTHING
                RETURNING cache_key
                """,
                (
                    cache_key,
                    team_id,
                    year,
                    prompt_version,
                    model_name,
                    wrapped_payload,
                    max(
                        1, _normalize_attempt_count(meta_defaults.get("attempt_count"))
                    ),
                ),
            ).fetchone()
            if inserted:
                conn.commit()
                return {"outcome": "inserted_missing_row"}
            row = conn.execute(
                """
                SELECT status, response_json, lease_owner
                FROM coach_analysis_cache
                WHERE cache_key = %s
                """,
                (cache_key,),
            ).fetchone()
        conn.commit()
    return {
        "outcome": "finalize_conflict",
        "status": row[0] if row else None,
        "response_json": row[1] if row else None,
        "lease_owner": row[2] if row else None,
    }


def _store_failed_cache(
    *,
    pool: ConnectionPool,
    cache_key: str,
    lease_owner: str,
    team_id: str,
    year: int,
    prompt_version: str,
    model_name: str,
    attempt_count: int,
    error_code: str,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_error_code = _sanitize_cache_error_code(error_code)
    stored_error_message = str(error_message or "").strip() or normalized_error_code
    with pool.connection() as conn:
        result = conn.execute(
            """
            UPDATE coach_analysis_cache
            SET status = 'FAILED',
                response_json = NULL,
                error_message = %s,
                error_code = %s,
                lease_owner = NULL,
                lease_expires_at = NULL,
                last_heartbeat_at = NULL,
                updated_at = now()
            WHERE cache_key = %s
              AND status = 'PENDING'
              AND lease_owner = %s
            """,
            (
                stored_error_message,
                normalized_error_code,
                cache_key,
                lease_owner,
            ),
        )
        if result.rowcount:
            conn.commit()
            return {"outcome": "updated"}

        row = conn.execute(
            """
            SELECT status, response_json, lease_owner
            FROM coach_analysis_cache
            WHERE cache_key = %s
            """,
            (cache_key,),
        ).fetchone()
        if not row:
            inserted = conn.execute(
                """
                INSERT INTO coach_analysis_cache (
                    cache_key, team_id, year, prompt_version, model_name, status,
                    response_json, error_message, error_code, attempt_count,
                    lease_owner, lease_expires_at, last_heartbeat_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, 'FAILED',
                    NULL, %s, %s, %s,
                    NULL, NULL, NULL, now()
                )
                ON CONFLICT (cache_key) DO NOTHING
                RETURNING cache_key
                """,
                (
                    cache_key,
                    team_id,
                    year,
                    prompt_version,
                    model_name,
                    stored_error_message,
                    normalized_error_code,
                    max(1, _normalize_attempt_count(attempt_count)),
                ),
            ).fetchone()
            if inserted:
                conn.commit()
                return {"outcome": "inserted_missing_row"}
            row = conn.execute(
                """
                SELECT status, response_json, lease_owner
                FROM coach_analysis_cache
                WHERE cache_key = %s
                """,
                (cache_key,),
            ).fetchone()
        conn.commit()
    return {
        "outcome": "finalize_conflict",
        "status": row[0] if row else None,
        "response_json": row[1] if row else None,
        "lease_owner": row[2] if row else None,
    }


def _normalize_cached_response(cached_data: dict) -> dict:
    """
    레거시 캐시 데이터를 현재 스키마에 맞게 정규화합니다.

    CoachResponse 검증기를 통과시켜 자동으로 변환:
    - status: "주의" → "warning", "양호" → "good", "위험" → "danger"
    - area: "불펜" → "bullpen", "선발" → "starter", "타격" → "batting"
    - coach_note: 150자 초과 시 자동 truncate

    Args:
        cached_data: 캐시에서 읽은 원본 JSON 데이터

    Returns:
        정규화된 데이터 (실패 시 원본 반환)
    """
    if not cached_data:
        return cached_data

    try:
        # CoachResponse 검증기를 통과시켜 자동 정규화
        response = CoachResponse(**cached_data)
        normalized = response.model_dump()
        logger.debug("[Coach Cache] Normalized legacy data")
        return normalized
    except Exception as e:
        logger.warning(f"[Coach Cache] Failed to normalize legacy data: {e}")
        return cached_data


def _parse_explicit_year(value: Any) -> Optional[int]:
    """명시적 year 필드(예: season_year)를 정수로 파싱합니다."""
    if value is None or isinstance(value, bool):
        return None
    try:
        year = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if 1000 <= year <= 9999:
        return year
    return None


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_year_from_date_like(value: Any) -> Optional[int]:
    """YYYY-MM-DD 또는 YYYY... 형태 문자열에서 연도를 추출합니다."""
    if value is None:
        return None
    text = str(value).strip()
    if len(text) < 4 or not text[:4].isdigit():
        return None
    return int(text[:4])


def _is_valid_analysis_year(year: int) -> bool:
    return COACH_YEAR_MIN <= year <= datetime.now().year + 1


def _resolve_year_from_season_id(pool: ConnectionPool, season_id: Any) -> Optional[int]:
    if season_id is None:
        return None
    try:
        normalized_season_id = int(str(season_id).strip())
    except (TypeError, ValueError):
        return None

    try:
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT season_year FROM kbo_seasons WHERE season_id = %s LIMIT 1",
                (normalized_season_id,),
            ).fetchone()
        if not row:
            return None
        season_year = int(row[0])
        return season_year
    except Exception as exc:
        logger.warning(
            "[Coach Router] Failed to resolve season_id=%s: %s", season_id, exc
        )
        return None


def _resolve_year_from_game_context(
    pool: ConnectionPool, game_id: Optional[str], game_date: Any
) -> Optional[int]:
    explicit_game_year = _parse_year_from_date_like(game_date)
    if explicit_game_year is not None:
        return explicit_game_year

    if game_id:
        try:
            with pool.connection() as conn:
                row = conn.execute(
                    """
                    SELECT COALESCE(ks.season_year, EXTRACT(YEAR FROM g.game_date)::int) AS season_year
                    FROM game g
                    LEFT JOIN kbo_seasons ks ON g.season_id = ks.season_id
                    WHERE g.game_id = %s
                    LIMIT 1
                    """,
                    (game_id,),
                ).fetchone()
            if row and row[0]:
                return int(row[0])
        except Exception as exc:
            logger.warning(
                "[Coach Router] Failed to resolve game_id=%s: %s", game_id, exc
            )

        fallback_year = _parse_year_from_date_like(game_id)
        if fallback_year is not None:
            return fallback_year

    return None


def _resolve_target_year(
    payload: "AnalyzeRequest", pool: ConnectionPool
) -> tuple[int, str]:
    league_context = payload.league_context or {}

    if "season_year" in league_context:
        parsed_year = _parse_explicit_year(league_context.get("season_year"))
        if parsed_year is None or not _is_valid_analysis_year(parsed_year):
            raise HTTPException(
                status_code=400,
                detail="invalid_season_year_for_analysis",
            )
        return parsed_year, "league_context.season_year"

    season_id = league_context.get("season")
    season_year = _resolve_year_from_season_id(pool, season_id)
    if season_year is not None and _is_valid_analysis_year(season_year):
        return season_year, "league_context.season->kbo_seasons"

    game_year = _resolve_year_from_game_context(
        pool,
        payload.game_id,
        league_context.get("game_date"),
    )
    if game_year is not None and _is_valid_analysis_year(game_year):
        return game_year, "game_date"

    raise HTTPException(
        status_code=400,
        detail="unable_to_resolve_analysis_year",
    )


def _build_fallback_evidence(
    payload: "AnalyzeRequest",
    year: int,
    home_team_code: str,
    away_team_code: Optional[str],
    home_team_name: str,
    away_team_name: Optional[str],
) -> GameEvidence:
    league_context = payload.league_context or {}
    stage_label = _normalize_stage_label(
        league_context.get("league_type_code"),
        league_context.get("stage_label") or league_context.get("round"),
    )
    round_display = _round_display_for_stage(stage_label)
    home_pitcher = _normalize_name_token(league_context.get("home_pitcher"))
    away_pitcher = _normalize_name_token(league_context.get("away_pitcher"))
    lineup_announced = bool(league_context.get("lineup_announced"))
    evidence = GameEvidence(
        game_id=payload.game_id,
        season_year=year,
        season_id=_parse_optional_int(league_context.get("season")),
        game_date=str(league_context.get("game_date") or "").strip() or None,
        game_status="UNKNOWN",
        game_status_bucket="UNKNOWN",
        home_team_code=home_team_code,
        away_team_code=away_team_code,
        home_team_name=home_team_name,
        away_team_name=away_team_name or "상대 팀",
        league_type_code=_parse_optional_int(league_context.get("league_type_code")),
        stage_label=stage_label,
        round_display=round_display,
        stage_game_no_hint=_parse_optional_int(
            league_context.get("series_game_no") or league_context.get("game_no")
        ),
        home_pitcher=home_pitcher,
        away_pitcher=away_pitcher,
        lineup_announced=lineup_announced,
    )
    evidence.used_evidence = _default_used_evidence(evidence)
    if evidence.stage_label != "REGULAR":
        evidence.series_state = EvidenceSeriesState(
            stage_label=evidence.stage_label,
            round_display=evidence.round_display,
            game_no=evidence.stage_game_no_hint,
        )
    return evidence


def _fetch_series_state(
    conn: Any,
    evidence: GameEvidence,
) -> Optional[EvidenceSeriesState]:
    if not evidence.game_id or evidence.stage_label in {"REGULAR", "PRE", "UNKNOWN"}:
        return None

    previous_games = list(
        conn.execute(
            """
            SELECT
                g.game_id,
                g.game_date,
                UPPER(TRIM(COALESCE(g.home_team, ''))) AS home_team_code,
                UPPER(TRIM(COALESCE(g.away_team, ''))) AS away_team_code,
                g.home_score,
                g.away_score
            FROM game g
            LEFT JOIN kbo_seasons ks ON g.season_id = ks.season_id
            WHERE COALESCE(ks.season_year, EXTRACT(YEAR FROM g.game_date)::int) = %s
              AND COALESCE(ks.league_type_code, 0) = %s
              AND (
                (
                  UPPER(TRIM(COALESCE(g.home_team, ''))) = UPPER(TRIM(%s))
                  AND UPPER(TRIM(COALESCE(g.away_team, ''))) = UPPER(TRIM(%s))
                )
                OR (
                  UPPER(TRIM(COALESCE(g.home_team, ''))) = UPPER(TRIM(%s))
                  AND UPPER(TRIM(COALESCE(g.away_team, ''))) = UPPER(TRIM(%s))
                )
              )
              AND (
                UPPER(TRIM(COALESCE(g.game_status, ''))) IN ('COMPLETED', 'FINAL', 'FINISHED', 'DONE', 'END', 'E', 'F')
                OR (g.home_score IS NOT NULL AND g.away_score IS NOT NULL)
              )
              AND (
                g.game_date < %s
                OR (g.game_date = %s AND g.game_id < %s)
              )
            ORDER BY g.game_date ASC NULLS LAST, g.game_id ASC
            """,
            (
                evidence.season_year,
                evidence.league_type_code,
                evidence.home_team_code,
                evidence.away_team_code,
                evidence.away_team_code,
                evidence.home_team_code,
                evidence.game_date,
                evidence.game_date,
                evidence.game_id,
            ),
        ).fetchall()
        or []
    )
    actual_previous_games = len(previous_games)
    expected_previous_games = actual_previous_games
    game_no = actual_previous_games + 1
    hint_mismatch = False
    if evidence.stage_game_no_hint is not None:
        expected_previous_games = max(int(evidence.stage_game_no_hint) - 1, 0)
        game_no = evidence.stage_game_no_hint
        hint_mismatch = actual_previous_games != expected_previous_games

    confirmed_previous_games = min(actual_previous_games, expected_previous_games)
    home_team_wins = 0
    away_team_wins = 0
    home_team_code = str(evidence.home_team_code or "").strip().upper()
    away_team_code = str(evidence.away_team_code or "").strip().upper()
    for (
        _,
        _,
        db_home_team_code,
        db_away_team_code,
        home_score,
        away_score,
    ) in previous_games[:confirmed_previous_games]:
        if home_score is None or away_score is None or home_score == away_score:
            continue
        winner_code = (
            str(db_home_team_code or "").strip().upper()
            if home_score > away_score
            else str(db_away_team_code or "").strip().upper()
        )
        if winner_code == home_team_code:
            home_team_wins += 1
        elif winner_code == away_team_code:
            away_team_wins += 1

    return EvidenceSeriesState(
        stage_label=evidence.stage_label,
        round_display=evidence.round_display,
        game_no=game_no,
        previous_games=expected_previous_games,
        confirmed_previous_games=confirmed_previous_games,
        home_team_wins=home_team_wins,
        away_team_wins=away_team_wins,
        series_state_partial=hint_mismatch,
        series_state_hint_mismatch=hint_mismatch,
    )


def _reconcile_series_state_with_hint(
    series_state: Optional[EvidenceSeriesState],
    evidence: GameEvidence,
) -> Optional[EvidenceSeriesState]:
    if evidence.stage_game_no_hint is None:
        return series_state

    expected_previous_games = max(int(evidence.stage_game_no_hint) - 1, 0)
    if series_state is None:
        return EvidenceSeriesState(
            stage_label=evidence.stage_label,
            round_display=evidence.round_display,
            game_no=evidence.stage_game_no_hint,
            previous_games=expected_previous_games,
            confirmed_previous_games=0,
            series_state_partial=expected_previous_games > 0,
            series_state_hint_mismatch=expected_previous_games > 0,
        )

    confirmed_previous_games = min(
        int(series_state.confirmed_previous_games or series_state.previous_games or 0),
        expected_previous_games,
    )
    hint_mismatch = (
        series_state.game_no != evidence.stage_game_no_hint
        or int(series_state.previous_games or 0) != expected_previous_games
    )
    if hint_mismatch:
        logger.info(
            "[Coach] Reconciled postseason series hint game_id=%s db_game_no=%s hint=%s db_previous=%s expected_previous=%s",
            evidence.game_id,
            series_state.game_no,
            evidence.stage_game_no_hint,
            series_state.previous_games,
            expected_previous_games,
        )
    series_state.game_no = evidence.stage_game_no_hint
    series_state.previous_games = expected_previous_games
    series_state.confirmed_previous_games = confirmed_previous_games
    series_state.series_state_hint_mismatch = (
        series_state.series_state_hint_mismatch
        or confirmed_previous_games != expected_previous_games
    )
    series_state.series_state_partial = (
        series_state.series_state_partial or series_state.series_state_hint_mismatch
    )
    return series_state


def _series_score_text(evidence: GameEvidence) -> Optional[str]:
    series_state = evidence.series_state
    if not series_state or not series_state.has_confirmed_score():
        return None
    return (
        f"{evidence.away_team_name} {series_state.away_team_wins}승 vs "
        f"{evidence.home_team_name} {series_state.home_team_wins}승"
    )


def _series_context_label(evidence: GameEvidence) -> Optional[str]:
    if not evidence.series_state:
        return None
    return evidence.series_state.matchup_label()


def _matchup_is_partial(matchup: Dict[str, Any]) -> bool:
    return bool((matchup or {}).get("series_state_partial"))


def _matchup_total_games(matchup: Dict[str, Any]) -> int:
    summary = matchup.get("summary", {}) or {}
    explicit_total = _parse_optional_int(summary.get("total_games"))
    if explicit_total is not None:
        return explicit_total
    counted_total = sum(
        int(summary.get(key, 0) or 0) for key in ("team1_wins", "team2_wins", "draws")
    )
    if counted_total > 0:
        return counted_total
    games = matchup.get("games") or []
    return len(games)


def _sanitize_matchup_result_for_evidence(
    evidence: GameEvidence,
    matchup: Dict[str, Any],
) -> Dict[str, Any]:
    if (
        not isinstance(matchup, dict)
        or evidence.stage_label in {"REGULAR", "PRE", "UNKNOWN"}
        or evidence.series_state is None
    ):
        return matchup

    expected_previous_games = max(int(evidence.series_state.previous_games or 0), 0)
    confirmed_previous_games = max(
        int(
            evidence.series_state.confirmed_previous_games
            or evidence.series_state.previous_games
            or 0
        ),
        0,
    )
    expected_team1_wins = max(int(evidence.series_state.home_team_wins or 0), 0)
    expected_team2_wins = max(int(evidence.series_state.away_team_wins or 0), 0)
    expected_draws = max(
        expected_previous_games - expected_team1_wins - expected_team2_wins,
        0,
    )
    confirmed_draws = max(
        confirmed_previous_games - expected_team1_wins - expected_team2_wins,
        0,
    )
    actual_total = _matchup_total_games(matchup)
    games = matchup.get("games") or []
    summary = matchup.get("summary", {}) or {}
    sanitized = dict(matchup)
    if evidence.series_state.series_state_partial:
        sanitized["games"] = list(games[:confirmed_previous_games])
        sanitized["summary"] = {
            **summary,
            "total_games": confirmed_previous_games,
            "team1_wins": expected_team1_wins,
            "team2_wins": expected_team2_wins,
            "draws": confirmed_draws,
        }
        sanitized["found"] = confirmed_previous_games > 0
        sanitized["series_state_partial"] = True
        logger.warning(
            "[Coach] Partial postseason matchup scope game_id=%s stage=%s confirmed_total=%s expected_total=%s",
            evidence.game_id,
            evidence.stage_label,
            confirmed_previous_games,
            expected_previous_games,
        )
        return sanitized
    if (
        actual_total <= expected_previous_games
        and len(games) <= expected_previous_games
    ):
        return matchup

    sanitized["games"] = list(games[:expected_previous_games])
    sanitized["summary"] = {
        **summary,
        "total_games": expected_previous_games,
        "team1_wins": expected_team1_wins,
        "team2_wins": expected_team2_wins,
        "draws": expected_draws,
    }
    sanitized["found"] = expected_previous_games > 0
    sanitized["series_state_partial"] = False
    logger.warning(
        "[Coach] Sanitized postseason matchup scope game_id=%s stage=%s actual_total=%s expected_total=%s",
        evidence.game_id,
        evidence.stage_label,
        actual_total,
        expected_previous_games,
    )
    return sanitized


def _collect_game_evidence(
    pool: ConnectionPool,
    payload: "AnalyzeRequest",
    *,
    year: int,
    home_team_code: str,
    away_team_code: Optional[str],
    home_team_name: str,
    away_team_name: Optional[str],
    team_resolver: TeamCodeResolver,
) -> GameEvidence:
    fallback = _build_fallback_evidence(
        payload,
        year,
        home_team_code,
        away_team_code,
        home_team_name,
        away_team_name,
    )
    if not payload.game_id:
        return fallback

    try:
        league_context = payload.league_context or {}
        with pool.connection() as conn:
            cursor = conn.cursor(row_factory=dict_row)
            game_row = cursor.execute(
                """
                SELECT
                    g.game_id,
                    g.game_date,
                    g.game_status,
                    g.home_team,
                    g.away_team,
                    g.home_pitcher,
                    g.away_pitcher,
                    g.season_id,
                    g.home_score,
                    g.away_score,
                    COALESCE(ks.season_year, EXTRACT(YEAR FROM g.game_date)::int) AS season_year,
                    COALESCE(ks.league_type_code, 0) AS league_type_code,
                    gm.stadium_name,
                    gm.start_time,
                    gm.weather
                FROM game g
                LEFT JOIN kbo_seasons ks ON g.season_id = ks.season_id
                LEFT JOIN game_metadata gm ON g.game_id = gm.game_id
                WHERE g.game_id = %s
                LIMIT 1
                """,
                (payload.game_id,),
            ).fetchone()

            if not game_row:
                return fallback

            home_code = team_resolver.resolve_canonical(
                game_row["home_team"] or home_team_code, year
            )
            away_code_raw = game_row["away_team"] or away_team_code
            away_code = (
                team_resolver.resolve_canonical(away_code_raw, year)
                if away_code_raw
                else away_team_code
            )
            if not _has_canonical_game_team_pair(home_code, away_code):
                logger.warning(
                    "[Coach] Ignoring non-canonical game row game_id=%s raw_home=%s raw_away=%s resolved_home=%s resolved_away=%s",
                    payload.game_id,
                    game_row.get("home_team"),
                    game_row.get("away_team"),
                    home_code,
                    away_code,
                )
                return fallback
            hint_league_type_code = _resolve_league_type_code_hint(league_context)
            db_league_type_code = int(game_row.get("league_type_code") or 0)
            effective_league_type_code = db_league_type_code
            if (
                hint_league_type_code is not None
                and hint_league_type_code != db_league_type_code
            ):
                logger.info(
                    "[Coach] Overriding league_type_code from request hint game_id=%s db=%s hint=%s",
                    payload.game_id,
                    db_league_type_code,
                    hint_league_type_code,
                )
                effective_league_type_code = hint_league_type_code
            stage_label = _normalize_stage_label(
                effective_league_type_code,
                league_context.get("stage_label")
                or league_context.get("round")
                or league_context.get("league_type"),
            )
            resolved_game_status = str(game_row.get("game_status") or "UNKNOWN")
            if (
                resolved_game_status.strip().upper() in {"", "UNKNOWN"}
                and game_row.get("home_score") is not None
                and game_row.get("away_score") is not None
            ):
                resolved_game_status = "COMPLETED"
            home_score = _parse_optional_int(game_row.get("home_score"))
            away_score = _parse_optional_int(game_row.get("away_score"))
            winning_team_code: Optional[str] = None
            if (
                home_score is not None
                and away_score is not None
                and home_score != away_score
            ):
                winning_team_code = home_code if home_score > away_score else away_code
            evidence = GameEvidence(
                game_id=game_row["game_id"],
                game_row_found=True,
                season_year=int(game_row["season_year"] or year),
                season_id=game_row.get("season_id"),
                game_date=(
                    game_row["game_date"].strftime("%Y-%m-%d")
                    if hasattr(game_row.get("game_date"), "strftime")
                    else _normalize_name_token(game_row.get("game_date"))
                ),
                game_status=resolved_game_status,
                game_status_bucket=_normalize_game_status_bucket(resolved_game_status),
                home_team_code=home_code,
                away_team_code=away_code,
                home_team_name=team_resolver.display_name(home_code),
                away_team_name=(
                    team_resolver.display_name(away_code)
                    if away_code
                    else (away_team_name or "상대 팀")
                ),
                league_type_code=effective_league_type_code,
                stage_label=stage_label,
                round_display=_round_display_for_stage(stage_label),
                stage_game_no_hint=_parse_optional_int(
                    league_context.get("series_game_no")
                    or league_context.get("game_no")
                ),
                home_score=home_score,
                away_score=away_score,
                winning_team_code=winning_team_code,
                winning_team_name=(
                    team_resolver.display_name(winning_team_code)
                    if winning_team_code
                    else None
                ),
                home_pitcher=_normalize_name_token(game_row.get("home_pitcher"))
                or fallback.home_pitcher,
                away_pitcher=_normalize_name_token(game_row.get("away_pitcher"))
                or fallback.away_pitcher,
                stadium_name=_normalize_name_token(game_row.get("stadium_name")),
                start_time=(
                    game_row["start_time"].strftime("%H:%M")
                    if hasattr(game_row.get("start_time"), "strftime")
                    else _normalize_name_token(game_row.get("start_time"))
                ),
                weather=_normalize_name_token(game_row.get("weather")),
            )

            lineup_rows = cursor.execute(
                """
                SELECT team_code, player_name, batting_order, is_starter
                FROM game_lineups
                WHERE game_id = %s
                  AND COALESCE(is_starter, true) = true
                ORDER BY team_code, batting_order ASC NULLS LAST, player_name ASC
                """,
                (payload.game_id,),
            ).fetchall()
            for lineup_row in lineup_rows or []:
                team_code = team_resolver.resolve_canonical(
                    lineup_row["team_code"], evidence.season_year
                )
                player_name = _normalize_name_token(lineup_row.get("player_name"))
                if not player_name:
                    continue
                if team_code == evidence.home_team_code:
                    evidence.home_lineup.append(player_name)
                elif team_code == evidence.away_team_code:
                    evidence.away_lineup.append(player_name)
            evidence.lineup_announced = bool(
                evidence.home_lineup or evidence.away_lineup
            ) or bool(league_context.get("lineup_announced"))

            summary_rows = cursor.execute(
                """
                SELECT summary_type, player_name, detail_text
                FROM game_summary
                WHERE game_id = %s
                ORDER BY id ASC
                LIMIT 6
                """,
                (payload.game_id,),
            ).fetchall()
            evidence.summary_items = [
                text
                for text in (
                    _format_game_summary_item(
                        row.get("summary_type"),
                        row.get("player_name"),
                        row.get("detail_text"),
                    )
                    for row in summary_rows or []
                )
                if text
            ]

            evidence.series_state = _reconcile_series_state_with_hint(
                _fetch_series_state(conn, evidence),
                evidence,
            )

            evidence.used_evidence = _default_used_evidence(evidence)
            return evidence
    except Exception as exc:
        logger.warning(
            "[Coach] Failed to collect game evidence for %s: %s", payload.game_id, exc
        )
        return fallback


def _build_effective_league_context(
    league_context: Optional[Dict[str, Any]],
    evidence: GameEvidence,
) -> Dict[str, Any]:
    context = dict(league_context or {})
    context["season_year"] = evidence.season_year
    if evidence.season_id is not None:
        context["season"] = evidence.season_id
    if evidence.game_date:
        context["game_date"] = evidence.game_date
    context["league_type"] = (
        "POST"
        if evidence.stage_label in {"WC", "SEMI_PO", "PO", "KS"}
        else ("PRE" if evidence.stage_label == "PRE" else "REGULAR")
    )
    context["league_type_code"] = evidence.league_type_code
    context["stage_label"] = evidence.stage_label
    context["round"] = evidence.round_display
    context["game_no"] = (
        evidence.series_state.game_no
        if evidence.series_state and evidence.series_state.game_no is not None
        else evidence.stage_game_no_hint
    )
    if evidence.home_pitcher:
        context["home_pitcher"] = evidence.home_pitcher
    if evidence.away_pitcher:
        context["away_pitcher"] = evidence.away_pitcher
    context["lineup_announced"] = _has_confirmed_lineup_details(evidence)
    return context


def _format_evidence_context(evidence: GameEvidence) -> str:
    parts = [
        "## 경기 근거 데이터",
        f"- 경기 ID: {evidence.game_id or '미상'}",
        f"- 경기 날짜: {evidence.game_date or '미상'}",
        f"- 경기 상태: {_build_game_status_label(evidence.game_status_bucket)}",
        f"- 리그 구분: {evidence.round_display}",
    ]
    if evidence.stage_label != "REGULAR" and evidence.series_state:
        parts.append(
            f"- 시리즈 전황: {evidence.series_state.summary_text(evidence.home_team_name, evidence.away_team_name)}"
        )
    if evidence.home_pitcher or evidence.away_pitcher:
        parts.append(
            f"- 발표 선발: {evidence.away_team_name} {evidence.away_pitcher or '미정'} / "
            f"{evidence.home_team_name} {evidence.home_pitcher or '미정'}"
        )
    if _has_confirmed_lineup_details(evidence):
        parts.append(
            f"- 발표 라인업: {evidence.away_team_name} [{_summarize_lineup_players(evidence.away_lineup)}] / "
            f"{evidence.home_team_name} [{_summarize_lineup_players(evidence.home_lineup)}]"
        )
    elif evidence.lineup_announced:
        parts.append("- 발표 라인업: 발표 신호만 확인됐고 선수 구성은 비어 있습니다.")
    else:
        parts.append("- 발표 라인업: 미확정")
    if evidence.summary_items:
        parts.append("- 경기 요약 근거:")
        for item in evidence.summary_items[:3]:
            parts.append(f"  - {item}")
    return "\n".join(parts)


def _build_focus_section_requirements(resolved_focus: List[str]) -> str:
    """
    선택 focus에 해당하는 상세 섹션 제목 요구사항을 생성합니다.
    """
    if not resolved_focus:
        return (
            "- 선택 focus가 비어 있습니다. 종합 분석을 수행하세요.\n"
            "- 다만 detailed_markdown은 최소 2개 이상의 소제목(##)으로 구성하세요."
        )

    header_lines = [
        f"- 반드시 `{FOCUS_SECTION_HEADERS[focus]}` 제목을 포함하세요."
        for focus in resolved_focus
        if focus in FOCUS_SECTION_HEADERS
    ]
    non_selected = [
        header
        for key, header in FOCUS_SECTION_HEADERS.items()
        if key not in resolved_focus
    ]
    omit_lines = [
        f"- 미선택 focus는 가능하면 생략하세요: `{header}`" for header in non_selected
    ]
    return "\n".join(header_lines + omit_lines)


def _find_missing_focus_sections(
    response_data: Dict[str, Any], resolved_focus: List[str]
) -> List[str]:
    """
    detailed_markdown에서 선택 focus 섹션 누락 여부를 확인합니다.
    """
    if not resolved_focus:
        return []

    markdown = str(response_data.get("detailed_markdown") or "")
    missing: List[str] = []
    for focus in resolved_focus:
        header = FOCUS_SECTION_HEADERS.get(focus)
        if not header:
            continue
        if header not in markdown or not _markdown_section_has_body(markdown, header):
            missing.append(focus)
    return missing


def _has_recent_form_support(team_data: Dict[str, Any]) -> bool:
    recent = team_data.get("recent", {}) if team_data else {}
    if not recent.get("found"):
        return False
    summary = recent.get("summary", {}) or {}
    total_games = sum(
        int(summary.get(key, 0) or 0) for key in ("wins", "losses", "draws")
    )
    return total_games >= 2


def _has_matchup_support(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> bool:
    matchup = tool_results.get("matchup", {}) or {}
    games = matchup.get("games") or []
    if len(games) >= 2:
        return True
    summary = matchup.get("summary", {}) or {}
    total_games = sum(
        int(summary.get(key, 0) or 0) for key in ("team1_wins", "team2_wins", "draws")
    )
    if total_games >= 2:
        return True
    if (
        evidence.stage_label != "REGULAR"
        and evidence.series_state is not None
        and int(
            evidence.series_state.confirmed_previous_games
            or evidence.series_state.previous_games
            or 0
        )
        > 0
    ):
        return True
    return False


def _focus_has_support(
    focus: str,
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> bool:
    home_data = tool_results.get("home", {}) or {}
    away_data = tool_results.get("away", {}) or {}
    home_summary = home_data.get("summary", {}) or {}
    away_summary = away_data.get("summary", {}) or {}
    home_adv = _advanced_metrics(home_data)
    away_adv = _advanced_metrics(away_data)
    home_form = home_data.get("player_form_signals", {}) or {}
    away_form = away_data.get("player_form_signals", {}) or {}

    if focus == "recent_form":
        return _has_recent_form_support(home_data) or _has_recent_form_support(
            away_data
        )
    if focus == "matchup":
        return _has_matchup_support(evidence, tool_results)
    if focus == "starter":
        return bool(evidence.home_pitcher and evidence.away_pitcher)
    if focus == "bullpen":
        return any(
            value is not None
            for value in (
                home_adv.get("fatigue_index", {}).get("bullpen_share"),
                away_adv.get("fatigue_index", {}).get("bullpen_share"),
            )
        )
    if focus == "batting":
        return (
            any(
                value is not None
                for value in (
                    home_adv.get("metrics", {}).get("batting", {}).get("ops"),
                    away_adv.get("metrics", {}).get("batting", {}).get("ops"),
                )
            )
            or bool(home_summary.get("top_batters"))
            or bool(away_summary.get("top_batters"))
            or bool(home_form.get("batters"))
            or bool(away_form.get("batters"))
        )
    return False


def _resolve_supported_focuses(
    resolved_focus: List[str],
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> List[str]:
    return [
        focus
        for focus in resolved_focus
        if _focus_has_support(focus, evidence, tool_results)
    ]


def _focus_display_label(focus: str) -> str:
    header = FOCUS_SECTION_HEADERS.get(focus)
    if isinstance(header, str) and header.startswith("## "):
        return header[3:]
    return focus


def _build_focus_data_warning(
    unsupported_focuses: List[str],
    supported_focuses: List[str],
) -> str:
    labels = ", ".join(_focus_display_label(focus) for focus in unsupported_focuses)
    if supported_focuses:
        return f"요청한 focus 중 {labels} 근거가 부족해 확인 가능한 항목만 분석합니다."
    return f"요청한 focus({labels}) 근거가 부족해 보수 요약으로 전환합니다."


def _metric_status_from_delta(delta: float, *, reverse: bool = False) -> str:
    effective_delta = -delta if reverse else delta
    if effective_delta >= 0.025:
        return "good"
    if effective_delta <= -0.025:
        return "danger"
    return "warning"


def _safe_percent_value(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().replace("%", "")
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _recent_summary(team_data: Dict[str, Any]) -> Dict[str, Any]:
    return team_data.get("recent", {}).get("summary", {}) if team_data else {}


def _advanced_metrics(team_data: Dict[str, Any]) -> Dict[str, Any]:
    return team_data.get("advanced", {}) if team_data else {}


def _short_recent_form_text(team_name: str, team_data: Dict[str, Any]) -> str:
    summary = _recent_summary(team_data)
    wins = int(summary.get("wins", 0) or 0)
    losses = int(summary.get("losses", 0) or 0)
    draws = int(summary.get("draws", 0) or 0)
    run_diff = int(summary.get("run_diff", 0) or 0)
    draw_text = f" {draws}무" if draws else ""
    run_diff_text = f"{'+' if run_diff >= 0 else ''}{run_diff}"
    return f"{team_name} 최근 {wins}승 {losses}패{draw_text}, 득실 {run_diff_text}"


def _recent_sample_size(team_data: Dict[str, Any]) -> int:
    summary = _recent_summary(team_data)
    return sum(int(summary.get(key, 0) or 0) for key in ("wins", "losses", "draws"))


def _should_use_team_level_scheduled_fallback(evidence: GameEvidence) -> bool:
    return _normalize_game_status_bucket(
        evidence.game_status_bucket
    ) == "SCHEDULED" and not _has_confirmed_lineup_details(evidence)


def _recent_record_metric_fragment(team_name: str, team_data: Dict[str, Any]) -> str:
    summary = _recent_summary(team_data)
    total_games = _recent_sample_size(team_data)
    if total_games >= 2:
        wins = int(summary.get("wins", 0) or 0)
        losses = int(summary.get("losses", 0) or 0)
        draws = int(summary.get("draws", 0) or 0)
        draw_text = f" {draws}무" if draws else ""
        return f"{team_name} {wins}승 {losses}패{draw_text}"

    team_score = _team_form_score(team_data)
    if team_score is not None:
        signal = _best_form_signal(team_data, "batters") or _best_form_signal(
            team_data, "pitchers"
        )
        return (
            f"{team_name} {_form_status_trend_label((signal or {}).get('form_status'))} "
            f"(폼 점수 {team_score:.1f})"
        )

    return f"{team_name} 데이터 부족"


def _recent_focus_summary(team_name: str, team_data: Dict[str, Any]) -> str:
    total_games = _recent_sample_size(team_data)
    if total_games >= 2:
        return _short_recent_form_text(team_name, team_data)

    team_score = _team_form_score(team_data)
    if team_score is not None:
        signal = _best_form_signal(team_data, "batters") or _best_form_signal(
            team_data, "pitchers"
        )
        return (
            f"{team_name}는 팀 폼 점수 {team_score:.1f}점을 기록하며 최근 흐름이 "
            f"{_form_status_trend_label((signal or {}).get('form_status'))}입니다."
        )

    return f"{team_name} 최근 흐름은 확인 가능한 표본이 부족합니다."


def _scheduled_team_level_recent_focus_summary(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> str:
    away_text = _recent_focus_summary(
        evidence.away_team_name, tool_results.get("away", {}) or {}
    )
    home_text = _recent_focus_summary(
        evidence.home_team_name, tool_results.get("home", {}) or {}
    )
    return f"{away_text} / {home_text}"


def _scheduled_team_level_starter_summary(evidence: GameEvidence) -> str:
    if evidence.home_pitcher and evidence.away_pitcher:
        return (
            f"발표 선발은 {evidence.away_team_name} {evidence.away_pitcher} / "
            f"{evidence.home_team_name} {evidence.home_pitcher}입니다."
        )
    if evidence.home_pitcher or evidence.away_pitcher:
        return (
            f"발표 선발은 {evidence.away_team_name} {evidence.away_pitcher or '미정'} / "
            f"{evidence.home_team_name} {evidence.home_pitcher or '미정'}입니다."
        )
    return "양 팀 모두 선발 발표 전이라 초반 매치업은 보수적으로 봐야 합니다."


def _scheduled_team_level_bullpen_summary(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> str:
    home_adv = _advanced_metrics(tool_results.get("home", {}) or {})
    away_adv = _advanced_metrics(tool_results.get("away", {}) or {})
    home_bullpen = _safe_percent_value(
        home_adv.get("fatigue_index", {}).get("bullpen_share")
    )
    away_bullpen = _safe_percent_value(
        away_adv.get("fatigue_index", {}).get("bullpen_share")
    )

    if away_bullpen and home_bullpen:
        if abs(away_bullpen - home_bullpen) >= 3.0:
            lean_team = (
                evidence.away_team_name
                if away_bullpen < home_bullpen
                else evidence.home_team_name
            )
            return (
                f"불펜 비중은 {evidence.away_team_name} {away_bullpen:.1f}% / "
                f"{evidence.home_team_name} {home_bullpen:.1f}%로, "
                f"{lean_team} 쪽이 후반 운용 여유를 보입니다."
            )
        return (
            f"불펜 비중은 {evidence.away_team_name} {away_bullpen:.1f}% / "
            f"{evidence.home_team_name} {home_bullpen:.1f}%로 큰 격차는 아닙니다."
        )

    if away_bullpen or home_bullpen:
        known_team = (
            evidence.away_team_name if away_bullpen else evidence.home_team_name
        )
        known_value = away_bullpen if away_bullpen else home_bullpen
        return (
            f"{known_team} 불펜 비중은 {known_value:.1f}%로 확인되지만, "
            "상대 팀 운용 데이터는 부족합니다."
        )

    return (
        "양 팀 모두 불펜 운용과 최근 소모 흐름 데이터가 부족해 "
        "경기 후반 대응력은 실전에서 확인이 필요합니다."
    )


def _scheduled_team_level_bullpen_metric_value(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> str:
    home_adv = _advanced_metrics(tool_results.get("home", {}) or {})
    away_adv = _advanced_metrics(tool_results.get("away", {}) or {})
    home_bullpen = _safe_percent_value(
        home_adv.get("fatigue_index", {}).get("bullpen_share")
    )
    away_bullpen = _safe_percent_value(
        away_adv.get("fatigue_index", {}).get("bullpen_share")
    )

    if away_bullpen and home_bullpen:
        return (
            f"{evidence.away_team_name} {away_bullpen:.1f}% / "
            f"{evidence.home_team_name} {home_bullpen:.1f}%"
        )

    if away_bullpen or home_bullpen:
        known_team = (
            evidence.away_team_name if away_bullpen else evidence.home_team_name
        )
        known_value = away_bullpen if away_bullpen else home_bullpen
        return f"{known_team} {known_value:.1f}% / 상대 데이터 부족"

    return "양 팀 모두 불펜 운용 데이터 부족"


def _scheduled_team_level_batting_summary(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> str:
    home_adv = _advanced_metrics(tool_results.get("home", {}) or {})
    away_adv = _advanced_metrics(tool_results.get("away", {}) or {})
    away_ops = away_adv.get("metrics", {}).get("batting", {}).get("ops")
    home_ops = home_adv.get("metrics", {}).get("batting", {}).get("ops")

    if away_ops is not None or home_ops is not None:
        away_ops_value = float(away_ops or 0.0)
        home_ops_value = float(home_ops or 0.0)
        edge_team = (
            evidence.away_team_name
            if away_ops_value >= home_ops_value
            else evidence.home_team_name
        )
        return (
            f"{evidence.away_team_name} OPS {away_ops_value:.3f} / "
            f"{evidence.home_team_name} OPS {home_ops_value:.3f}로 "
            f"{edge_team}가 팀 타격 생산성 우위를 보입니다."
        )

    return "타격 생산성 비교에 필요한 팀 단위 OPS 데이터가 충분하지 않습니다."


def _scheduled_team_level_swing_summary(evidence: GameEvidence) -> str:
    if evidence.home_pitcher and evidence.away_pitcher:
        return "발표 선발 뒤 첫 불펜 카드가 핵심 변수입니다."
    if evidence.home_pitcher or evidence.away_pitcher:
        return "선발 한 축만 공개돼 첫 투수 교체 시점이 핵심 변수입니다."
    return "선발 발표 전이라 첫 투수 교체 시점이 핵심 변수입니다."


def _scheduled_team_level_watch_points(evidence: GameEvidence) -> List[str]:
    primary = (
        "첫 득점 직후 어느 팀이 먼저 불펜 카드로 반응하는지 확인할 필요가 있습니다."
    )
    secondary = (
        "상위 타순 출루가 실제 득점으로 이어지는지 확인할 필요가 있습니다."
        if _has_confirmed_lineup_details(evidence)
        else "경기 직전 발표되는 라인업 변화 여부를 확인할 필요가 있습니다."
    )
    return [primary, secondary]


def _recent_win_rate(summary: Dict[str, Any]) -> Optional[float]:
    wins = int(summary.get("wins", 0) or 0)
    losses = int(summary.get("losses", 0) or 0)
    draws = int(summary.get("draws", 0) or 0)
    total_games = wins + losses + draws
    if total_games <= 0:
        return None
    return (wins + 0.5 * draws) / total_games


def _scheduled_recent_edge_delta(
    home_data: Dict[str, Any],
    away_data: Dict[str, Any],
) -> float:
    home_recent_summary = _recent_summary(home_data)
    away_recent_summary = _recent_summary(away_data)
    home_recent_games = _recent_sample_size(home_data)
    away_recent_games = _recent_sample_size(away_data)
    home_form_score = _team_form_score(home_data)
    away_form_score = _team_form_score(away_data)

    delta = 0.0
    supported = False
    if home_recent_games >= 2 and away_recent_games >= 2:
        supported = True
        home_win_rate = _recent_win_rate(home_recent_summary) or 0.0
        away_win_rate = _recent_win_rate(away_recent_summary) or 0.0
        delta += (home_win_rate - away_win_rate) * 1.2

        home_run_diff_per_game = int(home_recent_summary.get("run_diff", 0) or 0) / max(
            home_recent_games, 1
        )
        away_run_diff_per_game = int(away_recent_summary.get("run_diff", 0) or 0) / max(
            away_recent_games, 1
        )
        delta += (home_run_diff_per_game - away_run_diff_per_game) * 0.10

    if home_form_score is not None or away_form_score is not None:
        supported = True
        delta += ((home_form_score or 0.0) - (away_form_score or 0.0)) / 120.0

    if not supported:
        return 0.0
    return delta


def _scheduled_edge_team_from_delta(
    delta: float,
    *,
    home_name: str,
    away_name: str,
    threshold: float,
) -> Optional[str]:
    if abs(delta) < threshold:
        return None
    return home_name if delta > 0 else away_name


def _scheduled_primary_edge_assessment(
    *,
    home_name: str,
    away_name: str,
    recent_delta: float,
    ops_delta: float,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    recent_edge_team = _scheduled_edge_team_from_delta(
        recent_delta,
        home_name=home_name,
        away_name=away_name,
        threshold=0.08,
    )
    ops_edge_team = _scheduled_edge_team_from_delta(
        ops_delta,
        home_name=home_name,
        away_name=away_name,
        threshold=0.015,
    )
    composite_delta = recent_delta * 0.72 + ops_delta * 1.30
    edge_team = _scheduled_edge_team_from_delta(
        composite_delta,
        home_name=home_name,
        away_name=away_name,
        threshold=0.10,
    )
    if not edge_team:
        return None, None, None, None

    trailing_team = away_name if edge_team == home_name else home_name
    if recent_edge_team == edge_team and ops_edge_team == edge_team:
        return edge_team, trailing_team, "recent_ops", ops_edge_team
    if recent_edge_team == edge_team:
        if ops_edge_team and ops_edge_team != edge_team:
            return edge_team, trailing_team, "recent_slight", ops_edge_team
        return edge_team, trailing_team, "recent", ops_edge_team
    if ops_edge_team == edge_team:
        if recent_edge_team and recent_edge_team != edge_team:
            return edge_team, trailing_team, "ops_slight", recent_edge_team
        return edge_team, trailing_team, "ops", recent_edge_team
    return edge_team, trailing_team, "composite", None


def _scheduled_team_level_verdict(
    *,
    edge_team: Optional[str],
    edge_source: Optional[str],
) -> str:
    if not edge_team:
        return "절대 우위를 단정하기 어렵습니다."
    if edge_source == "recent_ops":
        return f"{edge_team}가 최근 전력과 팀 타격 생산성에서 앞섭니다."
    if edge_source == "ops":
        return f"{edge_team}가 팀 타격 생산성에서 한 발 앞섭니다."
    if edge_source == "ops_slight":
        return f"{edge_team}가 팀 타격 생산성 우세로 근소하게 앞섭니다."
    if edge_source == "recent":
        return f"{edge_team}가 최근 흐름에서 한 발 앞섭니다."
    if edge_source == "recent_slight":
        return f"{edge_team}가 최근 흐름 우세로 근소하게 앞섭니다."
    if edge_source == "composite":
        return f"{edge_team}가 팀 단위 지표를 종합하면 근소하게 앞섭니다."
    return f"{edge_team}가 팀 단위 지표에서 한 발 앞섭니다."


def _build_scheduled_team_level_why_it_matters(
    *,
    edge_team: Optional[str],
    trailing_team: Optional[str],
    edge_source: Optional[str],
    secondary_edge_team: Optional[str],
    recent_reason: Optional[str],
    ops_reason: Optional[str],
    bullpen_reason: Optional[str],
) -> List[str]:
    ordered: List[str] = []
    if edge_team and edge_source == "recent_ops":
        _append_distinct_note_part(
            ordered,
            f"{edge_team}가 최근 전력과 팀 OPS를 함께 앞세워 초중반 주도권을 먼저 잡을 가능성이 있습니다.",
        )
    elif edge_team and edge_source == "recent_slight":
        _append_distinct_note_part(
            ordered,
            f"{edge_team}가 최근 흐름 우위로 경기 중반 운영 선택지를 먼저 확보할 가능성이 있습니다.",
        )
        _append_distinct_note_part(
            ordered,
            f"{secondary_edge_team or trailing_team}도 팀 OPS 우위로 초반 선취점 반격 여지는 남아 있습니다.",
        )
    elif edge_team and edge_source == "ops_slight":
        _append_distinct_note_part(
            ordered,
            f"{edge_team}가 팀 OPS 우위로 초반 선취점 압박을 먼저 걸 수 있습니다.",
        )
        _append_distinct_note_part(
            ordered,
            f"{secondary_edge_team or trailing_team}도 최근 흐름 반등으로 경기 중반 대응 여지는 남아 있습니다.",
        )
    elif edge_team and edge_source == "composite":
        _append_distinct_note_part(
            ordered,
            f"{edge_team}가 최근 흐름과 팀 타격 생산성을 종합하면 운영 선택지를 조금 더 넓게 가져갈 가능성이 있습니다.",
        )

    if edge_source in {"recent_ops", "recent_slight", "recent", "composite"}:
        _append_distinct_note_part(ordered, recent_reason)
        _append_distinct_note_part(ordered, ops_reason)
    elif edge_source in {"ops_slight", "ops"}:
        _append_distinct_note_part(ordered, ops_reason)
        _append_distinct_note_part(ordered, recent_reason)
    else:
        _append_distinct_note_part(ordered, recent_reason)
        _append_distinct_note_part(ordered, ops_reason)

    _append_distinct_note_part(ordered, bullpen_reason)
    return ordered[:3]


def _scheduled_team_level_focus_summary(
    focus: str,
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> str:
    if focus == "recent_form":
        return _scheduled_team_level_recent_focus_summary(evidence, tool_results)
    if focus == "bullpen":
        return _scheduled_team_level_bullpen_summary(evidence, tool_results)
    if focus == "starter":
        return _scheduled_team_level_starter_summary(evidence)
    if focus == "batting":
        return _scheduled_team_level_batting_summary(evidence, tool_results)
    return FOCUS_SECTION_GENERIC_SUMMARIES.get(
        focus, "확인 가능한 실데이터 기준으로 추가 근거가 제한적입니다."
    )


def _focus_metric_value(
    response_payload: Optional[Dict[str, Any]],
    focus: str,
) -> Optional[str]:
    if not isinstance(response_payload, dict):
        return None
    metric_labels = FOCUS_SECTION_METRIC_LABELS.get(focus, ())
    for metric in response_payload.get("key_metrics", []) or []:
        if not isinstance(metric, dict):
            continue
        label = str(metric.get("label") or "").strip()
        value = _clean_summary_text(str(metric.get("value") or ""))
        if label in metric_labels and value:
            return value
    return None


def _focus_warning_by_keywords(
    grounding_warnings: Optional[List[str]],
    keywords: Tuple[str, ...],
) -> Optional[str]:
    for warning in grounding_warnings or []:
        warning_text = _clean_summary_text(str(warning or ""))
        if warning_text and any(keyword in warning_text for keyword in keywords):
            return warning_text
    return None


def _analysis_focus_candidate_texts(
    response_payload: Optional[Dict[str, Any]],
    *,
    keywords: Tuple[str, ...],
) -> List[str]:
    if not isinstance(response_payload, dict):
        return []
    analysis = response_payload.get("analysis") or {}
    if not isinstance(analysis, dict):
        return []

    candidates: List[str] = []
    for item in analysis.get("strengths") or []:
        text = _clean_summary_text(item)
        if text and any(keyword in text for keyword in keywords):
            candidates.append(text)
    for item in analysis.get("weaknesses") or []:
        text = _clean_summary_text(item)
        if text and any(keyword in text for keyword in keywords):
            candidates.append(text)
    for risk in analysis.get("risks") or []:
        if not isinstance(risk, dict):
            continue
        description = _clean_summary_text(risk.get("description"))
        if description and any(keyword in description for keyword in keywords):
            candidates.append(description)
    for key in ("watch_points", "uncertainty"):
        for item in analysis.get(key) or []:
            text = _clean_summary_text(item)
            if text and any(keyword in text for keyword in keywords):
                candidates.append(text)
    return candidates


def _completed_clutch_moment_text(
    moment: Dict[str, Any],
    *,
    with_wpa: bool = False,
) -> str:
    inning_label = _normalize_name_token(moment.get("inning_label")) or "핵심 이닝"
    batter_name = _normalize_name_token(moment.get("batter_name"))
    description = _normalize_name_token(moment.get("description"))
    wpa_value = moment.get("wpa_delta_pct", "데이터 부족")

    if description and inning_label and description.startswith(inning_label):
        description = description[len(inning_label) :].strip(" -:()[]")
        description = description or None

    if batter_name:
        base = f"{inning_label} {batter_name} 타석"
    elif description:
        base = f"{inning_label} {description} 장면"
    else:
        base = f"{inning_label} 핵심 장면"

    if with_wpa:
        return f"{base}의 WPA {wpa_value}%p 변동"
    return base


def _compact_completed_focus_warning(
    focus: str,
    warning_text: Optional[str],
) -> Optional[str]:
    warning = _clean_summary_text(warning_text)
    if not warning:
        return None
    if focus == "matchup" and "상대 전적" in warning:
        return "상대 전적 표본이 부족합니다."
    if focus == "recent_form" and "최근" in warning:
        return "최근 표본이 적어 흐름 판단은 보수적으로 봐야 합니다."
    if focus == "bullpen" and "불펜" in warning:
        return "불펜 운용 데이터가 부족합니다."
    if focus == "batting" and "타격" in warning:
        return "타격 지표 표본이 부족합니다."
    return warning


def _completed_review_recent_focus_summary(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
    response_payload: Optional[Dict[str, Any]] = None,
    *,
    prefer_payload: bool = True,
) -> str:
    metric_value = _focus_metric_value(response_payload, "recent_form")
    if prefer_payload and metric_value:
        return metric_value

    away_data = tool_results.get("away", {}) or {}
    home_data = tool_results.get("home", {}) or {}
    away_games = _recent_sample_size(away_data)
    home_games = _recent_sample_size(home_data)
    if away_games > 0 or home_games > 0:
        away_text = (
            _short_recent_form_text(evidence.away_team_name, away_data)
            if away_games > 0
            else f"{evidence.away_team_name} 최근 표본이 적어 흐름 판단은 보수적으로 봐야 합니다."
        )
        home_text = (
            _short_recent_form_text(evidence.home_team_name, home_data)
            if home_games > 0
            else f"{evidence.home_team_name} 최근 표본이 적어 흐름 판단은 보수적으로 봐야 합니다."
        )
        return f"{away_text} / {home_text}"

    return "최근 표본이 적어 흐름 판단은 보수적으로 봐야 합니다."


def _completed_review_bullpen_focus_summary(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
    response_payload: Optional[Dict[str, Any]] = None,
    *,
    prefer_payload: bool = True,
) -> str:
    metric_value = _focus_metric_value(response_payload, "bullpen")
    if prefer_payload and metric_value:
        if "%" in metric_value:
            return f"불펜 비중은 {metric_value}입니다."
        return metric_value

    keywords = FOCUS_SECTION_KEYWORDS.get("bullpen", ())
    if prefer_payload:
        for candidate in _analysis_focus_candidate_texts(
            response_payload,
            keywords=keywords,
        ):
            return candidate

    home_adv = _advanced_metrics(tool_results.get("home", {}) or {})
    away_adv = _advanced_metrics(tool_results.get("away", {}) or {})
    home_bullpen = _safe_percent_value(
        home_adv.get("fatigue_index", {}).get("bullpen_share")
    )
    away_bullpen = _safe_percent_value(
        away_adv.get("fatigue_index", {}).get("bullpen_share")
    )

    if away_bullpen and home_bullpen:
        if abs(away_bullpen - home_bullpen) >= 3.0:
            return (
                f"불펜 비중은 {evidence.away_team_name} {away_bullpen:.1f}% / "
                f"{evidence.home_team_name} {home_bullpen:.1f}%로 차이가 확인됩니다."
            )
        return (
            f"불펜 비중은 {evidence.away_team_name} {away_bullpen:.1f}% / "
            f"{evidence.home_team_name} {home_bullpen:.1f}%로 큰 격차는 확인되지 않았습니다."
        )

    if away_bullpen or home_bullpen:
        known_team = (
            evidence.away_team_name if away_bullpen else evidence.home_team_name
        )
        known_value = away_bullpen if away_bullpen else home_bullpen
        return f"{known_team} 불펜 비중 {known_value:.1f}%만 확인되고 상대 팀 데이터는 부족합니다."

    return "불펜 운용 데이터가 부족해 후반 평가는 보수적으로 봐야 합니다."


def _completed_review_starter_focus_summary(
    evidence: GameEvidence,
    response_payload: Optional[Dict[str, Any]] = None,
    *,
    prefer_payload: bool = True,
) -> str:
    metric_value = _focus_metric_value(response_payload, "starter")
    if evidence.home_pitcher or evidence.away_pitcher:
        return (
            f"{evidence.away_team_name} {evidence.away_pitcher or '미정'} / "
            f"{evidence.home_team_name} {evidence.home_pitcher or '미정'}"
        )
    if prefer_payload and metric_value:
        return metric_value
    return "선발 정보가 확정되지 않아 선발 맞대결 평가는 제한됩니다."


def _completed_review_matchup_focus_summary(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
    response_payload: Optional[Dict[str, Any]] = None,
    grounding_warnings: Optional[List[str]] = None,
    *,
    prefer_payload: bool = True,
) -> str:
    metric_value = _focus_metric_value(response_payload, "matchup")
    if prefer_payload and metric_value:
        return metric_value

    if evidence.stage_label != "REGULAR" and evidence.series_state:
        return _series_score_text(evidence) or evidence.series_state.summary_text(
            evidence.home_team_name,
            evidence.away_team_name,
        )

    matchup = tool_results.get("matchup", {}) or {}
    summary = matchup.get("summary", {}) or {}
    if _matchup_total_games(matchup) > 0:
        base = (
            f"{evidence.away_team_name} {summary.get('team2_wins', 0)}승 / "
            f"{evidence.home_team_name} {summary.get('team1_wins', 0)}승 / "
            f"{summary.get('draws', 0)}무"
        )
        if _matchup_is_partial(matchup):
            return f"부분 확인 기준 상대 전적은 {base}입니다."
        return base

    warning_text = _focus_warning_by_keywords(
        grounding_warnings,
        FOCUS_SECTION_KEYWORDS.get("matchup", ()),
    )
    if warning_text:
        return _compact_completed_focus_warning("matchup", warning_text) or warning_text

    return FOCUS_SECTION_GENERIC_SUMMARIES["matchup"]


def _completed_review_batting_focus_summary(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
    response_payload: Optional[Dict[str, Any]] = None,
    *,
    prefer_payload: bool = True,
) -> str:
    metric_value = _focus_metric_value(response_payload, "batting")
    if prefer_payload and metric_value:
        return metric_value

    home_adv = _advanced_metrics(tool_results.get("home", {}) or {})
    away_adv = _advanced_metrics(tool_results.get("away", {}) or {})
    away_ops = away_adv.get("metrics", {}).get("batting", {}).get("ops")
    home_ops = home_adv.get("metrics", {}).get("batting", {}).get("ops")
    if away_ops is not None or home_ops is not None:
        away_text = f"{float(away_ops):.3f}" if away_ops is not None else "데이터 부족"
        home_text = f"{float(home_ops):.3f}" if home_ops is not None else "데이터 부족"
        return f"{evidence.away_team_name} {away_text} / {evidence.home_team_name} {home_text}"

    return "확인 가능한 타격 생산성 근거가 제한적입니다."


def _completed_review_focus_summary(
    focus: str,
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
    response_payload: Optional[Dict[str, Any]] = None,
    grounding_warnings: Optional[List[str]] = None,
    *,
    prefer_payload: bool = True,
) -> str:
    if focus == "recent_form":
        return _completed_review_recent_focus_summary(
            evidence,
            tool_results,
            response_payload=response_payload,
            prefer_payload=prefer_payload,
        )
    if focus == "bullpen":
        return _completed_review_bullpen_focus_summary(
            evidence,
            tool_results,
            response_payload=response_payload,
            prefer_payload=prefer_payload,
        )
    if focus == "starter":
        return _completed_review_starter_focus_summary(
            evidence,
            response_payload=response_payload,
            prefer_payload=prefer_payload,
        )
    if focus == "matchup":
        return _completed_review_matchup_focus_summary(
            evidence,
            tool_results,
            response_payload=response_payload,
            grounding_warnings=grounding_warnings,
            prefer_payload=prefer_payload,
        )
    if focus == "batting":
        return _completed_review_batting_focus_summary(
            evidence,
            tool_results,
            response_payload=response_payload,
            prefer_payload=prefer_payload,
        )
    return FOCUS_SECTION_GENERIC_SUMMARIES.get(
        focus, "확인 가능한 실데이터 기준으로 추가 근거가 제한적입니다."
    )


def _build_scheduled_team_level_deterministic_metrics(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    home_data = tool_results.get("home", {}) or {}
    away_data = tool_results.get("away", {}) or {}
    home_adv = _advanced_metrics(home_data)
    away_adv = _advanced_metrics(away_data)
    home_recent_summary = _recent_summary(home_data)
    away_recent_summary = _recent_summary(away_data)
    home_recent_games = _recent_sample_size(home_data)
    away_recent_games = _recent_sample_size(away_data)
    home_run_diff = int(home_recent_summary.get("run_diff", 0) or 0)
    away_run_diff = int(away_recent_summary.get("run_diff", 0) or 0)
    home_form_score = _team_form_score(home_data)
    away_form_score = _team_form_score(away_data)
    home_recent_games = _recent_sample_size(home_data)
    away_recent_games = _recent_sample_size(away_data)

    if (
        home_recent_games >= 2
        or away_recent_games >= 2
        or home_form_score is not None
        or away_form_score is not None
    ):
        recent_delta = 0.0
        if home_form_score is not None or away_form_score is not None:
            recent_delta = ((home_form_score or 0.0) - (away_form_score or 0.0)) / 100.0
        metrics.append(
            {
                "label": "최근 흐름",
                "value": (
                    f"{_recent_record_metric_fragment(evidence.away_team_name, away_data)} / "
                    f"{_recent_record_metric_fragment(evidence.home_team_name, home_data)}"
                ),
                "status": _metric_status_from_delta(recent_delta),
                "trend": "neutral",
                "is_critical": False,
            }
        )

    away_ops = away_adv.get("metrics", {}).get("batting", {}).get("ops")
    home_ops = home_adv.get("metrics", {}).get("batting", {}).get("ops")
    if away_ops is not None or home_ops is not None:
        away_ops_value = float(away_ops or 0.0)
        home_ops_value = float(home_ops or 0.0)
        metrics.append(
            {
                "label": "팀 타격 생산성",
                "value": (
                    f"{evidence.away_team_name} OPS {away_ops_value:.3f} / "
                    f"{evidence.home_team_name} OPS {home_ops_value:.3f}"
                ),
                "status": _metric_status_from_delta(home_ops_value - away_ops_value),
                "trend": "neutral",
                "is_critical": True,
            }
        )

    metrics.append(
        {
            "label": "불펜 운용",
            "value": _scheduled_team_level_bullpen_metric_value(evidence, tool_results),
            "status": "warning",
            "trend": "neutral",
            "is_critical": True,
        }
    )

    if evidence.home_pitcher or evidence.away_pitcher:
        metrics.append(
            {
                "label": "발표 선발",
                "value": (
                    f"{evidence.away_team_name} {evidence.away_pitcher or '미정'} / "
                    f"{evidence.home_team_name} {evidence.home_pitcher or '미정'}"
                ),
                "status": "warning",
                "trend": "neutral",
                "is_critical": True,
            }
        )

    return metrics[:5]


def _build_scheduled_team_level_deterministic_analysis(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> Dict[str, Any]:
    home_name = evidence.home_team_name
    away_name = evidence.away_team_name
    home_data = tool_results.get("home", {}) or {}
    away_data = tool_results.get("away", {}) or {}
    home_adv = _advanced_metrics(home_data)
    away_adv = _advanced_metrics(away_data)
    home_form_score = _team_form_score(home_data)
    away_form_score = _team_form_score(away_data)
    home_form_signal = _best_form_signal(home_data, "batters") or _best_form_signal(
        home_data, "pitchers"
    )
    away_form_signal = _best_form_signal(away_data, "batters") or _best_form_signal(
        away_data, "pitchers"
    )
    home_ops = float(home_adv.get("metrics", {}).get("batting", {}).get("ops") or 0.0)
    away_ops = float(away_adv.get("metrics", {}).get("batting", {}).get("ops") or 0.0)
    recent_delta = _scheduled_recent_edge_delta(home_data, away_data)
    ops_delta = home_ops - away_ops

    strengths: List[str] = []
    weaknesses: List[str] = []
    risks: List[Dict[str, Any]] = []
    why_it_matters: List[str] = []
    swing_factors: List[str] = []
    watch_points: List[str] = []
    uncertainty: List[str] = []
    recent_reason: Optional[str] = None
    ops_reason: Optional[str] = None
    bullpen_reason: Optional[str] = None

    edge_team: Optional[str] = None
    trailing_team: Optional[str] = None
    edge_source: Optional[str] = None
    secondary_edge_team: Optional[str] = None
    ops_edge_team = _scheduled_edge_team_from_delta(
        ops_delta,
        home_name=home_name,
        away_name=away_name,
        threshold=0.015,
    )
    recent_edge_team = _scheduled_edge_team_from_delta(
        recent_delta,
        home_name=home_name,
        away_name=away_name,
        threshold=0.08,
    )

    if ops_edge_team:
        ops_lead_value = max(home_ops, away_ops)
        ops_trailing_value = min(home_ops, away_ops)
        ops_trailing_team = home_name if ops_edge_team == away_name else away_name
        strengths.append(
            f"{ops_edge_team}는 정규시즌 OPS {ops_lead_value:.3f}로 팀 타격 생산성 우위를 보입니다."
        )
        weaknesses.append(
            f"{ops_trailing_team}는 정규시즌 OPS {ops_trailing_value:.3f}로 초반 득점 설계가 과제입니다."
        )
        ops_reason = f"{ops_edge_team}가 팀 OPS에서 앞서 초반 선취점 압박을 먼저 걸 가능성이 있습니다."

    if recent_edge_team:
        strengths.append(f"{recent_edge_team}는 최근 득실 흐름에서 앞섰습니다.")
        recent_reason = f"최근 표본에서는 {recent_edge_team}가 득실 흐름 우위를 보여 경기 중반 운영 선택지가 더 넓습니다."

    edge_team, trailing_team, edge_source, secondary_edge_team = (
        _scheduled_primary_edge_assessment(
            home_name=home_name,
            away_name=away_name,
            recent_delta=recent_delta,
            ops_delta=ops_delta,
        )
    )

    for team_name, score, signal in (
        (away_name, away_form_score, away_form_signal),
        (home_name, home_form_score, home_form_signal),
    ):
        if score is None:
            continue
        status_label = _form_status_trend_label((signal or {}).get("form_status"))
        if status_label == "하락세":
            weaknesses.append(
                f"{team_name}는 팀 폼 점수 {score:.1f}점을 기록해 최근 흐름이 {status_label}입니다."
            )
        else:
            strengths.append(
                f"{team_name}는 팀 폼 점수 {score:.1f}점을 기록하며 최근 흐름이 {status_label}입니다."
            )

    bullpen_summary = _scheduled_team_level_bullpen_summary(evidence, tool_results)
    if "데이터가 부족" in bullpen_summary or "데이터는 부족" in bullpen_summary:
        weaknesses.append("양 팀 모두 불펜 운용과 최근 소모 흐름 데이터가 부족합니다.")
        risks.append(
            {
                "area": "bullpen",
                "level": 1,
                "description": "양 팀 모두 불펜 운용과 최근 소모 흐름 데이터가 부족합니다.",
            }
        )
    else:
        bullpen_reason = bullpen_summary

    swing_factors.append(_scheduled_team_level_swing_summary(evidence))
    watch_points.extend(_scheduled_team_level_watch_points(evidence))

    uncertainty.append(
        "라인업 미발표라 타순 기반 세부 매치업은 경기 직전까지 달라질 수 있습니다."
    )
    if not evidence.home_pitcher or not evidence.away_pitcher:
        uncertainty.append(
            "선발 정보가 완전히 확정되지 않아 초반 흐름 해석은 보수적으로 봐야 합니다."
        )

    summary: str
    verdict: str
    if edge_team and trailing_team:
        if edge_source == "recent_ops":
            summary = (
                f"{edge_team}가 최근 전력과 팀 타격 생산성에서 모두 앞서 있지만, "
                f"{trailing_team}도 운영 변수 하나로 흐름을 뒤집을 여지는 남아 있습니다."
            )
        elif edge_source == "recent_slight":
            summary = (
                f"{edge_team}가 최근 전력 우세로 근소하게 앞서지만, "
                f"{(secondary_edge_team or trailing_team)}의 팀 타격 생산성 반격 여지는 남아 있습니다."
            )
        elif edge_source == "ops_slight":
            summary = (
                f"{edge_team}가 팀 타격 생산성 우세로 근소하게 앞서지만, "
                f"{(secondary_edge_team or trailing_team)}의 최근 전력 반등 가능성도 열려 있습니다."
            )
        else:
            summary = (
                f"{edge_team}가 확인된 팀 단위 지표에서 먼저 앞서지만, "
                f"{trailing_team}도 운영 변수 하나로 흐름을 뒤집을 여지는 남아 있습니다."
            )
        verdict = _scheduled_team_level_verdict(
            edge_team=edge_team,
            edge_source=edge_source,
        )
    else:
        summary = (
            "확인된 팀 단위 지표는 박빙이며, 초반 운영과 첫 번째 불펜 선택이 "
            "경기 흐름을 좌우할 가능성이 큽니다."
        )
        verdict = _scheduled_team_level_verdict(
            edge_team=None,
            edge_source=None,
        )

    why_it_matters = _build_scheduled_team_level_why_it_matters(
        edge_team=edge_team,
        trailing_team=trailing_team,
        edge_source=edge_source,
        secondary_edge_team=secondary_edge_team,
        recent_reason=recent_reason,
        ops_reason=ops_reason,
        bullpen_reason=bullpen_reason,
    )

    if not why_it_matters:
        why_it_matters.append(
            "확인된 수치가 제한적이라 절대 우위보다 경기 중 운용 변수 확인이 더 중요합니다."
        )

    return {
        "summary": summary,
        "verdict": _truncate_text_naturally(verdict, max_length=240),
        "strengths": strengths[:4],
        "weaknesses": weaknesses[:3],
        "risks": risks[:2],
        "why_it_matters": why_it_matters[:3],
        "swing_factors": swing_factors[:3],
        "watch_points": watch_points[:3],
        "uncertainty": uncertainty[:2],
    }


def _build_deterministic_metrics(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if _should_use_team_level_scheduled_fallback(evidence):
        return _build_scheduled_team_level_deterministic_metrics(
            evidence,
            tool_results,
        )

    metrics: List[Dict[str, Any]] = []
    review_mode = _is_completed_review(evidence)
    review_summary_item = (
        _primary_review_summary_item(evidence) if review_mode else None
    )
    home_data = tool_results.get("home", {})
    away_data = tool_results.get("away", {})
    home_recent = _recent_summary(home_data)
    away_recent = _recent_summary(away_data)
    home_adv = _advanced_metrics(home_data)
    away_adv = _advanced_metrics(away_data)

    if evidence.stage_label != "REGULAR" and evidence.series_state:
        series_score_text = _series_score_text(evidence)
        series_context_label = _series_context_label(evidence) or evidence.round_display
        metrics.append(
            {
                "label": "시리즈 전적" if series_score_text else "시리즈 맥락",
                "value": series_score_text or series_context_label,
                "status": "warning",
                "trend": "neutral",
                "is_critical": True,
            }
        )

    if review_mode:
        result_metric = _completed_result_metric(evidence)
        if result_metric:
            metrics.append(result_metric)

    if evidence.home_pitcher or evidence.away_pitcher:
        metrics.append(
            {
                "label": "발표 선발",
                "value": (
                    f"{evidence.away_team_name} {evidence.away_pitcher or '미정'} / "
                    f"{evidence.home_team_name} {evidence.home_pitcher or '미정'}"
                ),
                "status": "warning",
                "trend": "neutral",
                "is_critical": True,
            }
        )

    if home_recent or away_recent:
        metrics.append(
            {
                "label": "최근 흐름",
                "value": (
                    f"{evidence.away_team_name} {away_recent.get('wins', 0)}승 {away_recent.get('losses', 0)}패 / "
                    f"{evidence.home_team_name} {home_recent.get('wins', 0)}승 {home_recent.get('losses', 0)}패"
                ),
                "status": "warning",
                "trend": "neutral",
                "is_critical": False,
            }
        )

    if review_mode:
        clutch_moments = _clutch_moments(tool_results)
        if clutch_moments:
            top_moment = clutch_moments[0]
            metrics.append(
                {
                    "label": "최대 WPA 변동",
                    "value": (
                        f"{_completed_clutch_moment_text(top_moment)} "
                        f"{top_moment.get('wpa_delta_pct', '데이터 부족')}%p"
                    ),
                    "status": "warning",
                    "trend": "neutral",
                    "is_critical": True,
                }
            )
        elif review_summary_item:
            metrics.append(
                {
                    "label": (
                        "승부처 요약"
                        if _summary_item_is_clutch_like(review_summary_item)
                        else "경기 요약"
                    ),
                    "value": review_summary_item,
                    "status": "warning",
                    "trend": "neutral",
                    "is_critical": _summary_item_is_clutch_like(review_summary_item),
                }
            )
    else:
        home_form_score = _team_form_score(home_data)
        away_form_score = _team_form_score(away_data)
        if home_form_score is not None or away_form_score is not None:
            home_signal = _best_form_signal(home_data, "batters") or _best_form_signal(
                home_data, "pitchers"
            )
            away_signal = _best_form_signal(away_data, "batters") or _best_form_signal(
                away_data, "pitchers"
            )
            away_score_text = (
                f"{away_form_score:.1f}"
                if away_form_score is not None
                else "데이터 부족"
            )
            home_score_text = (
                f"{home_form_score:.1f}"
                if home_form_score is not None
                else "데이터 부족"
            )
            metrics.append(
                {
                    "label": "폼 진단",
                    "value": (
                        f"{evidence.away_team_name} {_form_status_label((away_signal or {}).get('form_status'))} "
                        f"{away_score_text} / "
                        f"{evidence.home_team_name} {_form_status_label((home_signal or {}).get('form_status'))} "
                        f"{home_score_text}"
                    ),
                    "status": _metric_status_from_delta(
                        ((home_form_score or 0.0) - (away_form_score or 0.0)) / 100.0
                    ),
                    "trend": "neutral",
                    "is_critical": False,
                }
            )

    away_ops = away_adv.get("metrics", {}).get("batting", {}).get("ops")
    home_ops = home_adv.get("metrics", {}).get("batting", {}).get("ops")
    if away_ops is not None or home_ops is not None:
        away_ops_value = float(away_ops or 0.0)
        home_ops_value = float(home_ops or 0.0)
        metrics.append(
            {
                "label": "정규시즌 OPS",
                "value": (
                    f"{evidence.away_team_name} {away_ops_value:.3f} / "
                    f"{evidence.home_team_name} {home_ops_value:.3f}"
                ),
                "status": _metric_status_from_delta(home_ops_value - away_ops_value),
                "trend": "neutral",
                "is_critical": False,
            }
        )

    away_bullpen = _safe_percent_value(
        away_adv.get("fatigue_index", {}).get("bullpen_share")
    )
    home_bullpen = _safe_percent_value(
        home_adv.get("fatigue_index", {}).get("bullpen_share")
    )
    if away_bullpen or home_bullpen:
        metrics.append(
            {
                "label": "불펜 비중",
                "value": (
                    f"{evidence.away_team_name} {away_bullpen:.1f}% / "
                    f"{evidence.home_team_name} {home_bullpen:.1f}%"
                ),
                "status": _metric_status_from_delta(
                    away_bullpen - home_bullpen, reverse=True
                ),
                "trend": "neutral",
                "is_critical": False,
            }
        )

    return metrics[:5]


def _build_deterministic_analysis(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> Dict[str, Any]:
    if _should_use_team_level_scheduled_fallback(evidence):
        return _build_scheduled_team_level_deterministic_analysis(
            evidence,
            tool_results,
        )

    review_mode = _is_completed_review(evidence)
    home_name = evidence.home_team_name
    away_name = evidence.away_team_name
    strengths: List[str] = []
    weaknesses: List[str] = []
    risks: List[Dict[str, Any]] = []
    why_it_matters: List[str] = []
    swing_factors: List[str] = []
    watch_points: List[str] = []
    uncertainty: List[str] = []
    home_data = tool_results.get("home", {})
    away_data = tool_results.get("away", {})
    home_adv = _advanced_metrics(home_data)
    away_adv = _advanced_metrics(away_data)
    home_recent_summary = _recent_summary(home_data)
    away_recent_summary = _recent_summary(away_data)
    home_recent_games = _recent_sample_size(home_data)
    away_recent_games = _recent_sample_size(away_data)
    home_run_diff = int(home_recent_summary.get("run_diff", 0) or 0)
    away_run_diff = int(away_recent_summary.get("run_diff", 0) or 0)
    edge_scores = {home_name: 0, away_name: 0}
    clutch_moments = _clutch_moments(tool_results)
    review_summary_item = (
        _primary_review_summary_item(evidence) if review_mode else None
    )
    review_summary_is_clutch = _summary_item_is_clutch_like(review_summary_item)
    home_batter_form = _best_form_signal(home_data, "batters")
    away_batter_form = _best_form_signal(away_data, "batters")
    home_pitcher_form = _best_form_signal(home_data, "pitchers")
    away_pitcher_form = _best_form_signal(away_data, "pitchers")

    home_ops = float(home_adv.get("metrics", {}).get("batting", {}).get("ops") or 0.0)
    away_ops = float(away_adv.get("metrics", {}).get("batting", {}).get("ops") or 0.0)
    if home_ops and away_ops:
        if home_ops >= away_ops:
            edge_scores[home_name] += 2
            strengths.append(
                f"{home_name} 정규시즌 OPS {home_ops:.3f}로 {away_name}({away_ops:.3f})보다 앞섭니다."
            )
            weaknesses.append(
                f"{away_name}는 정규시즌 OPS {away_ops:.3f}로 공격 선결 과제가 남아 있습니다."
            )
            why_it_matters.append(
                f"{home_name}가 출루·장타 베이스라인에서 앞서 선취점 압박을 먼저 걸 가능성이 있습니다."
            )
        else:
            edge_scores[away_name] += 2
            strengths.append(
                f"{away_name} 정규시즌 OPS {away_ops:.3f}가 {home_name}({home_ops:.3f})보다 높습니다."
            )
            weaknesses.append(
                f"{home_name}는 정규시즌 OPS {home_ops:.3f}로 선취점 압박이 큽니다."
            )
            why_it_matters.append(
                f"{away_name}가 장타 생산성에서 앞서 있어 초반 득점 루트를 더 쉽게 만들 수 있습니다."
            )

    for team_name, batter_form, opponent_name in (
        (home_name, home_batter_form, away_name),
        (away_name, away_batter_form, home_name),
    ):
        if not batter_form:
            continue
        status = str(batter_form.get("form_status") or "")
        player_name = batter_form.get("player_name") or "핵심 타자"
        score = batter_form.get("form_score")
        score_text = (
            f"{float(score):.1f}" if isinstance(score, (int, float)) else "데이터 부족"
        )
        if status == "hot":
            edge_scores[team_name] += 1
            strengths.append(
                f"{team_name}는 {player_name}의 폼 점수 {score_text}로 타선 중심축이 살아 있습니다."
            )
            why_it_matters.append(
                f"{player_name}의 최근 장타/WPA 흐름이 유지되면 {team_name}가 선취점과 빅이닝 연결 확률을 키울 수 있습니다."
            )
        elif status == "cold":
            weaknesses.append(
                f"{team_name}는 {player_name}의 폼 점수 {score_text}로 중심 타선 파괴력이 평소보다 낮습니다."
            )
            risks.append(
                {
                    "area": "batting",
                    "level": 1,
                    "description": f"{team_name} 핵심 타자 {player_name}의 최근 클러치 생산성이 둔화됐습니다.",
                }
            )
            if not review_mode:
                watch_points.append(
                    f"{team_name}는 {player_name} 앞뒤 타순에서 출루를 얼마나 이어 주는지가 공격 변동성을 줄일 포인트입니다."
                )

    for team_name, pitcher_form in (
        (home_name, home_pitcher_form),
        (away_name, away_pitcher_form),
    ):
        if not pitcher_form:
            continue
        status = str(pitcher_form.get("form_status") or "")
        player_name = pitcher_form.get("player_name") or "핵심 투수"
        score = pitcher_form.get("form_score")
        score_text = (
            f"{float(score):.1f}" if isinstance(score, (int, float)) else "데이터 부족"
        )
        if status == "hot":
            edge_scores[team_name] += 1
            strengths.append(
                f"{team_name}는 {player_name}의 폼 점수 {score_text}로 선발·핵심 투수 컨디션이 안정적입니다."
            )
        elif status == "cold":
            weaknesses.append(
                f"{team_name}는 {player_name}의 폼 점수 {score_text}로 최근 투수 운영 부담이 커졌습니다."
            )
            risks.append(
                {
                    "area": (
                        "starter"
                        if pitcher_form.get("role") == "starter"
                        else "bullpen"
                    ),
                    "level": 0,
                    "description": f"{team_name} {player_name}의 최근 실점/WPA 허용 흐름이 좋지 않습니다.",
                }
            )

    home_recent = _short_recent_form_text(home_name, home_data)
    away_recent = _short_recent_form_text(away_name, away_data)
    if home_data.get("recent", {}).get("found"):
        strengths.append(home_recent)
    if away_data.get("recent", {}).get("found"):
        strengths.append(away_recent)
    if (
        home_recent_games >= 2
        and away_recent_games >= 2
        and home_run_diff != away_run_diff
    ):
        if home_run_diff > away_run_diff:
            edge_scores[home_name] += 1
            why_it_matters.append(
                f"최근 표본에서는 {home_name}가 득실 흐름에서 앞서 경기 중반 운영 선택지가 더 넓습니다."
            )
        else:
            edge_scores[away_name] += 1
            why_it_matters.append(
                f"최근 표본에서는 {away_name}가 득실 흐름 우위를 보여 초반 실점 관리만 되면 주도권을 잡기 쉽습니다."
            )

    home_bullpen = _safe_percent_value(
        home_adv.get("fatigue_index", {}).get("bullpen_share")
    )
    away_bullpen = _safe_percent_value(
        away_adv.get("fatigue_index", {}).get("bullpen_share")
    )
    league_avg_home = _safe_percent_value(
        home_adv.get("league_averages", {}).get("bullpen_share")
    )
    league_avg_away = _safe_percent_value(
        away_adv.get("league_averages", {}).get("bullpen_share")
    )
    if home_bullpen and league_avg_home and home_bullpen - league_avg_home >= 5.0:
        risks.append(
            {
                "area": "bullpen",
                "level": 0,
                "description": (
                    f"{home_name} 불펜 비중 {home_bullpen:.1f}%로 "
                    f"리그 평균 {league_avg_home:.1f}%보다 높습니다."
                ),
            }
        )
        watch_points.append(
            (
                f"{home_name}가 5회 이전에 선발을 내린 구간이 불펜 소모 확대로 이어졌는지 복기할 필요가 있습니다."
                if review_mode
                else f"{home_name}가 5회 이전에 선발을 내리면 불펜 소모가 다시 커질 수 있습니다."
            )
        )
    if away_bullpen and league_avg_away and away_bullpen - league_avg_away >= 5.0:
        risks.append(
            {
                "area": "bullpen",
                "level": 0,
                "description": (
                    f"{away_name} 불펜 비중 {away_bullpen:.1f}%로 "
                    f"리그 평균 {league_avg_away:.1f}%보다 높습니다."
                ),
            }
        )
        watch_points.append(
            (
                f"{away_name}가 후반 접전으로 끌려간 구간에서 불펜 매치업 부담이 커졌는지 확인해야 합니다."
                if review_mode
                else f"{away_name}가 후반 접전으로 끌려가면 불펜 매치업 부담이 커질 수 있습니다."
            )
        )
    if home_bullpen and away_bullpen and abs(home_bullpen - away_bullpen) >= 3.0:
        fresher_team = home_name if home_bullpen < away_bullpen else away_name
        edge_scores[fresher_team] += 1
        why_it_matters.append(
            f"불펜 비중 차이 {abs(home_bullpen - away_bullpen):.1f}%p는 접전 후반 운용 폭 차이로 이어질 수 있습니다."
        )
    if evidence.stage_label != "REGULAR" and evidence.series_state:
        strengths.append(evidence.series_state.summary_text(home_name, away_name))
        swing_factors.append(evidence.series_state.summary_text(home_name, away_name))
    if evidence.home_pitcher or evidence.away_pitcher:
        pitcher_matchup_text = _join_with_korean_and(
            f"{away_name} {evidence.away_pitcher or '미정'}",
            f"{home_name} {evidence.home_pitcher or '미정'}",
        )
        strengths.append(
            f"발표 선발은 {away_name} {evidence.away_pitcher or '미정'} / {home_name} {evidence.home_pitcher or '미정'}입니다."
        )
        swing_factors.append(
            (
                f"초반 3이닝에서 {pitcher_matchup_text}의 이닝 소화력이 실제 흐름 차이로 이어졌는지 먼저 복기해야 합니다."
                if review_mode
                else f"초반 3이닝은 {pitcher_matchup_text}의 이닝 소화력이 좌우합니다."
            )
        )
    if _has_confirmed_lineup_details(evidence):
        strengths.append(
            f"발표 라인업 기준 상위 타순은 {away_name} [{_summarize_lineup_players(evidence.away_lineup)}], {home_name} [{_summarize_lineup_players(evidence.home_lineup)}]입니다."
        )
        watch_points.append(
            (
                "상위 타순 출루가 실제 득점으로 이어진 장면이 결과를 갈랐는지 복기할 필요가 있습니다."
                if review_mode
                else "상위 타순 출루가 먼저 연결되는 팀이 불펜 소모 전에 경기 설계를 유리하게 가져갈 가능성이 있습니다."
            )
        )
    elif evidence.lineup_announced:
        uncertainty.append(
            "라인업 발표 신호는 있으나 선수 구성이 비어 있어 타순 기반 비교는 보수적으로 봐야 합니다."
        )
    else:
        uncertainty.append(
            "라인업 미발표 경기라 타순 기반 승부처는 경기 직전까지 유동적입니다."
        )

    if not evidence.home_pitcher or not evidence.away_pitcher:
        uncertainty.append(
            "선발 정보가 완전히 확정되지 않아 초반 매치업 해석은 보수적으로 봐야 합니다."
        )

    if clutch_moments:
        top_moment = clutch_moments[0]
        top_moment_text = _completed_clutch_moment_text(top_moment, with_wpa=True)
        swing_factors.insert(
            0,
            (
                f"{top_moment_text}이 실제 승부처였습니다."
                if review_mode
                else f"{top_moment.get('inning_label', '이닝 미상')} 전후의 하이레버리지 타석이 흐름을 크게 흔들 변수입니다."
            ),
        )
        why_it_matters.append(
            f"가장 큰 WPA 변동 구간은 {top_moment.get('description', '핵심 장면')}로, 한 번의 선택이 기대 승률을 크게 흔들었다는 뜻입니다."
        )
        watch_points.insert(
            0,
            (
                _build_completed_review_watch_point(
                    top_moment=top_moment,
                    review_summary_item=review_summary_item,
                )
                if review_mode
                else f"{top_moment.get('inning_label', '이닝 미상')} 전후에 어떤 타순이 올라오는지가 체감 승부처가 됩니다."
            ),
        )
    elif review_mode:
        if review_summary_item:
            swing_factors.insert(
                0,
                (
                    f"경기 요약 기준 '{review_summary_item}' 장면이 실제 승부처로 기록됐습니다."
                    if review_summary_is_clutch
                    else f"경기 요약 기준 '{review_summary_item}' 장면이 결과 흐름을 설명하는 대표 장면으로 남았습니다."
                ),
            )
            why_it_matters.append(
                (
                    f"경기 요약에 '{review_summary_item}'가 남았다는 점은 결과를 가른 흐름을 복기할 때 해당 장면을 먼저 봐야 한다는 뜻입니다."
                )
            )
            watch_points.append(
                f"'{review_summary_item}' 직전의 주자 상황과 투수 교체 선택이 어떻게 이어졌는지 다시 볼 필요가 있습니다."
            )
            uncertainty.append(
                "WPA 수치가 없어 변동 폭은 특정할 수 없지만, 경기 요약 기준 핵심 장면은 확인됩니다."
            )
        else:
            uncertainty.append(
                "완료 경기지만 WPA 기반 세부 승부처 데이터가 없어 장면 복기는 보수적으로 봐야 합니다."
            )

    if home_recent_games < 2 and away_recent_games < 2:
        uncertainty.append(
            "최근 경기 표본이 작아 흐름 평가는 정규시즌 베이스라인 비중이 더 큽니다."
        )

    if not why_it_matters:
        why_it_matters.append(
            (
                "확인된 수치가 제한적이라 결과를 가른 운영 선택과 득점 연결 구간을 함께 복기하는 편이 더 중요합니다."
                if review_mode
                else "확인된 수치가 제한적이라 절대 우위보다 경기 중 운용 변수 확인이 더 중요합니다."
            )
        )
    if not swing_factors:
        preview_swing_factor = (
            (
                f"{away_name}와 {home_name} 모두 선발 발표 전이라, 첫 번째 투수 교체와 상위 타순 두 번째 순환 구간이 가장 큰 변수입니다."
            )
            if not evidence.home_pitcher and not evidence.away_pitcher
            else (
                f"{away_name} {evidence.away_pitcher or '선발 미정'}와 {home_name} {evidence.home_pitcher or '선발 미정'}가 내려간 뒤 첫 불펜 카드 선택이 흐름을 좌우할 수 있습니다."
            )
        )
        swing_factors.append(
            (
                "선취점 이후 불펜 투입 시점과 대응 순서가 결과를 갈랐는지 복기할 필요가 있습니다."
                if review_mode
                else preview_swing_factor
            )
        )
    if not watch_points:
        watch_points.append(
            (
                "첫 번째 득점 직후 투수 교체와 작전 선택이 어떻게 이어졌는지 다시 볼 필요가 있습니다."
                if review_mode
                else "첫 번째 득점 이후 어느 팀이 불펜 카드로 먼저 반응하는지 확인할 필요가 있습니다."
            )
        )
    if not uncertainty:
        uncertainty.append(
            (
                "완료 경기라도 확인된 실데이터 범위를 벗어난 세부 장면 평가는 보수적으로 봐야 합니다."
                if review_mode
                else "확인된 실데이터 범위 안에서는 경기 전 발표 정보 변화가 가장 큰 불확실성입니다."
            )
        )

    lead_swing_factor = _ensure_sentence(swing_factors[0] if swing_factors else None)
    lead_risk = _ensure_sentence(risks[0]["description"] if risks else None)
    home_score = edge_scores[home_name]
    away_score = edge_scores[away_name]
    if review_mode:
        scoreline = _completed_scoreline_text(evidence)
        winner_name = _completed_winner_name(evidence)
        loser_name = _completed_loser_name(evidence)
        result_margin = _completed_result_margin(evidence)
        predicted_edge_team: Optional[str] = None
        if home_score != away_score:
            predicted_edge_team = home_name if home_score > away_score else away_name
        if _completed_game_is_draw(evidence) and scoreline:
            summary = (
                f"{scoreline} 무승부로 끝났고, 확인된 실데이터 기준으로는 "
                "득점 연결과 승부처 대응이 서로 맞물린 경기였습니다."
            )
            verdict = lead_swing_factor or (
                f"무승부였지만 '{review_summary_item}' 장면 전후 대응이 "
                "가장 큰 분기점으로 남았습니다."
                if review_summary_item
                else "무승부였지만 후반 운영과 득점 연결 효율이 경기 흐름을 가장 크게 흔들었습니다."
            )
        elif winner_name and loser_name and scoreline:
            margin_text = (
                "한 점 차 접전에서"
                if result_margin == 1
                else (
                    "중반 이후 운영에서"
                    if isinstance(result_margin, int) and result_margin <= 3
                    else "득점 연결과 불펜 운용에서"
                )
            )
            summary = (
                f"{scoreline} 경기에서 {winner_name}가 이겼고, "
                f"{loser_name}{_topic_particle(loser_name)} {margin_text} 차이를 남겼습니다."
            )
            if lead_swing_factor:
                verdict = _build_completed_review_turning_point_verdict(
                    winner_name=winner_name,
                    top_moment=clutch_moments[0] if clutch_moments else None,
                    review_summary_item=review_summary_item,
                    review_summary_is_clutch=review_summary_is_clutch,
                )
            elif review_summary_item:
                verdict = _build_completed_review_turning_point_verdict(
                    winner_name=winner_name,
                    top_moment=None,
                    review_summary_item=review_summary_item,
                    review_summary_is_clutch=review_summary_is_clutch,
                )
            elif lead_risk:
                verdict = f"{winner_name}가 결과를 만든 배경에는 {lead_risk}"
            else:
                verdict = (
                    f"{winner_name}가 실제 결과를 만들었고, "
                    "후반 운영과 득점 연결 효율이 승패를 갈랐습니다."
                )
        else:
            summary = "확인된 기초 지표는 박빙이었고, 실제 결과도 한두 번의 운영 선택과 득점 연결이 승부를 갈랐을 가능성이 큽니다."
            verdict = "사전 우열보다 경기 중 승부처 대응과 득점 연결 효율이 실제 결과를 가른 쪽에 더 가깝습니다."
        why_it_matters = _build_completed_review_why_it_matters(
            evidence,
            winner_name=winner_name,
            loser_name=loser_name,
            result_margin=result_margin,
            review_summary_item=review_summary_item,
            review_summary_is_clutch=review_summary_is_clutch,
            clutch_moments=clutch_moments,
            predicted_edge_team=predicted_edge_team,
            home_name=home_name,
            away_name=away_name,
            home_recent_games=home_recent_games,
            away_recent_games=away_recent_games,
            home_run_diff=home_run_diff,
            away_run_diff=away_run_diff,
            home_bullpen=home_bullpen,
            away_bullpen=away_bullpen,
        )
    elif home_score == away_score:
        summary = "확인된 기초 지표는 박빙이며, 초반 선발 운영과 후반 불펜 선택이 승부를 가를 가능성이 큽니다."
        verdict = "숫자만으로는 절대 우위를 단정하기 어렵고, 접전으로 갈수록 불펜 운영과 한 번의 장타가 더 중요해집니다."
    else:
        edge_team = home_name if home_score > away_score else away_name
        trailing_team = away_name if edge_team == home_name else home_name
        margin = abs(home_score - away_score)
        summary = f"{edge_team}가 확인된 지표에서 먼저 앞서지만, {trailing_team}도 운용 변수 하나로 흐름을 뒤집을 여지는 남아 있습니다."
        if margin >= 2:
            verdict = f"{edge_team}가 기초 지표 우위를 갖고 출발합니다."
            if lead_risk:
                verdict = f"{verdict} {lead_risk}"
            elif lead_swing_factor:
                verdict = f"{verdict} {lead_swing_factor}"
            else:
                verdict = f"{verdict} 후반 불펜과 초반 선발 운영이 승부처로 남습니다."
        else:
            verdict = (
                f"{edge_team}가 확인된 지표에서 한 발 앞서 있지만, "
                f"{trailing_team}도 운영 선택 하나로 흐름을 뒤집을 여지가 있습니다."
            )
            if lead_swing_factor:
                verdict = f"{verdict} {lead_swing_factor}"
            else:
                verdict = f"{verdict} 초반 이닝 운영에 따라 체감 우위가 쉽게 바뀔 수 있습니다."

    return {
        "summary": summary,
        "verdict": _truncate_text_naturally(verdict, max_length=240),
        "strengths": strengths[:4],
        "weaknesses": weaknesses[:3],
        "risks": risks[:2],
        "why_it_matters": why_it_matters[:3],
        "swing_factors": swing_factors[:3],
        "watch_points": watch_points[:3],
        "uncertainty": uncertainty[:2],
    }


def _build_deterministic_headline(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> str:
    review_mode = _is_completed_review(evidence)
    clutch_moments = _clutch_moments(tool_results)
    review_summary_item = (
        _primary_review_summary_item(evidence) if review_mode else None
    )
    result_scoreline = _completed_scoreline_text(evidence)
    result_winner = _completed_winner_name(evidence)
    if evidence.stage_label != "REGULAR" and evidence.series_state:
        series_score_text = _series_score_text(evidence)
        series_context_label = _series_context_label(evidence) or evidence.round_display
        if review_mode and result_scoreline:
            if result_winner:
                return (
                    f"{series_context_label} {result_scoreline} 경기, "
                    f"{result_winner} 승리 리뷰"
                )
            if _completed_game_is_draw(evidence):
                return f"{series_context_label} {result_scoreline} 무승부 리뷰"
        if series_score_text:
            return f"{series_score_text}, {series_context_label}"
        return f"{series_context_label}, {evidence.away_team_name}-{evidence.home_team_name}"
    if review_mode and result_winner and review_summary_item:
        return f"{result_winner} 승리, {review_summary_item}"
    if review_mode and result_winner and result_scoreline:
        return f"{result_scoreline} 경기, {result_winner} 승리 리뷰"
    if review_mode and _completed_game_is_draw(evidence) and result_scoreline:
        return f"{result_scoreline} 무승부, 데이터 기반 경기 리뷰"
    if review_mode and clutch_moments:
        top_moment = clutch_moments[0]
        batter_name = _normalize_name_token(top_moment.get("batter_name")) or "승부처"
        return (
            f"{batter_name}의 {top_moment.get('inning_label', '핵심 장면')}, "
            f"{evidence.away_team_name}-{evidence.home_team_name} 경기 리뷰"
        )
    if review_mode and review_summary_item:
        return (
            f"{review_summary_item}, "
            f"{evidence.away_team_name}-{evidence.home_team_name} 경기 리뷰"
        )
    if evidence.home_pitcher or evidence.away_pitcher:
        return (
            f"{evidence.away_pitcher or evidence.away_team_name} vs "
            f"{evidence.home_pitcher or evidence.home_team_name}, "
            f"{'경기 리뷰' if review_mode else '승부처 해석'}"
        )
    if review_mode:
        return f"{evidence.away_team_name} vs {evidence.home_team_name}, 데이터 기반 경기 리뷰"
    return f"{evidence.away_team_name} vs {evidence.home_team_name}, 데이터 기반 코치 브리핑"


def _build_deterministic_markdown(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
    resolved_focus: Optional[List[str]] = None,
    grounding_warnings: Optional[List[str]] = None,
) -> str:
    if _should_use_team_level_scheduled_fallback(evidence):
        analysis = _build_scheduled_team_level_deterministic_analysis(
            evidence, tool_results
        )
        sections: List[str] = []
        if resolved_focus:
            for focus in resolved_focus:
                header = FOCUS_SECTION_HEADERS.get(focus)
                if not header:
                    continue
                sections.append(header)
                sections.append(
                    f"- {_scheduled_team_level_focus_summary(focus, evidence, tool_results)}"
                )

        sections.append("## 코치 판단")
        sections.append(f"- {analysis.get('verdict') or analysis.get('summary')}")

        if analysis.get("why_it_matters"):
            sections.append("## 왜 중요한가")
            sections.extend(f"- {item}" for item in analysis["why_it_matters"])

        if analysis.get("swing_factors"):
            sections.append("## 승부 스윙 포인트")
            sections.extend(f"- {item}" for item in analysis["swing_factors"])

        if analysis.get("watch_points"):
            sections.append("## 체크 포인트")
            sections.extend(f"- {item}" for item in analysis["watch_points"])

        if analysis.get("uncertainty"):
            sections.append("## 불확실성")
            sections.extend(f"- {item}" for item in analysis["uncertainty"])

        return _truncate_text_naturally(
            "\n".join(sections),
            max_length=1500,
            preserve_newlines=True,
        )

    review_mode = _is_completed_review(evidence)
    analysis = _build_deterministic_analysis(evidence, tool_results)
    sections: List[str] = []

    if resolved_focus:
        for focus in resolved_focus:
            header = FOCUS_SECTION_HEADERS.get(focus)
            if not header:
                continue
            sections.append(header)
            if review_mode:
                sections.append(
                    f"- {_completed_review_focus_summary(focus, evidence, tool_results, grounding_warnings=grounding_warnings)}"
                )

    sections.append("## 결과 진단" if review_mode else "## 코치 판단")
    sections.append(f"- {analysis.get('verdict') or analysis.get('summary')}")

    if analysis.get("why_it_matters"):
        sections.append("## 결과를 가른 이유" if review_mode else "## 왜 중요한가")
        sections.extend(f"- {item}" for item in analysis["why_it_matters"])

    if analysis.get("swing_factors"):
        sections.append("## 실제 전환점" if review_mode else "## 승부 스윙 포인트")
        sections.extend(f"- {item}" for item in analysis["swing_factors"])

    if analysis.get("watch_points"):
        sections.append("## 다시 볼 장면" if review_mode else "## 체크 포인트")
        sections.extend(f"- {item}" for item in analysis["watch_points"])

    if analysis.get("uncertainty"):
        sections.append("## 해석 한계" if review_mode else "## 불확실성")
        sections.extend(f"- {item}" for item in analysis["uncertainty"])

    return _truncate_text_naturally(
        "\n".join(sections),
        max_length=1500,
        preserve_newlines=True,
    )


def _markdown_section_has_body(markdown: str, header: str) -> bool:
    if not markdown or not header:
        return False
    pattern = re.compile(rf"(?ms)^{re.escape(header)}\s*\n(?P<body>.*?)(?=^##\s+|\Z)")
    match = pattern.search(markdown)
    if not match:
        return False
    body = str(match.group("body") or "").strip()
    body = re.sub(r"^[\-\*\d\.\)\s]+", "", body, flags=re.MULTILINE).strip()
    return bool(body)


def _find_markdown_section_span(
    markdown: str, header: str
) -> Optional[Tuple[int, int]]:
    if not markdown or not header:
        return None
    pattern = re.compile(rf"(?ms)^{re.escape(header)}\s*\n.*?(?=^##\s+|\Z)")
    match = pattern.search(markdown)
    if not match:
        return None
    return match.span()


def _replace_markdown_section_body(markdown: str, header: str, body: str) -> str:
    if not markdown or not header or not body:
        return str(markdown or "")
    pattern = re.compile(rf"(?ms)^{re.escape(header)}\s*\n.*?(?=^##\s+|\Z)")
    replacement = f"{header}\n{body.strip()}\n\n"
    if pattern.search(markdown):
        return pattern.sub(replacement, markdown, count=1)
    return markdown


def _iter_response_text_candidates(response_payload: Dict[str, Any]) -> List[str]:
    analysis = response_payload.get("analysis") or {}
    candidates: List[str] = []
    for key in ("summary", "verdict"):
        value = str(analysis.get(key) or "").strip()
        if value:
            candidates.append(value)
    for key in ("why_it_matters", "swing_factors", "watch_points", "uncertainty"):
        items = analysis.get(key) or []
        if isinstance(items, list):
            candidates.extend(
                str(item).strip() for item in items if str(item or "").strip()
            )
    coach_note = str(response_payload.get("coach_note") or "").strip()
    if coach_note:
        candidates.append(coach_note)
    return candidates


def _build_missing_focus_section_summary(
    response_payload: Dict[str, Any],
    focus: str,
    grounding_warnings: Optional[List[str]] = None,
    evidence: Optional[GameEvidence] = None,
    tool_results: Optional[Dict[str, Any]] = None,
) -> str:
    if (
        evidence is not None
        and tool_results is not None
        and _is_completed_review(evidence)
    ):
        return _completed_review_focus_summary(
            focus,
            evidence,
            tool_results,
            response_payload=response_payload,
            grounding_warnings=grounding_warnings,
        )

    metric_labels = FOCUS_SECTION_METRIC_LABELS.get(focus, ())
    for metric in response_payload.get("key_metrics", []) or []:
        if not isinstance(metric, dict):
            continue
        label = str(metric.get("label") or "").strip()
        value = str(metric.get("value") or "").strip()
        if label in metric_labels and value:
            return value

    keywords = FOCUS_SECTION_KEYWORDS.get(focus, ())
    for candidate in _iter_response_text_candidates(response_payload):
        if any(keyword in candidate for keyword in keywords):
            return candidate

    for warning in grounding_warnings or []:
        warning_text = str(warning or "").strip()
        if warning_text and any(keyword in warning_text for keyword in keywords):
            return warning_text

    return FOCUS_SECTION_GENERIC_SUMMARIES.get(
        focus, "확인 가능한 실데이터 기준으로 추가 근거가 제한적입니다."
    )


def _build_missing_focus_section_blocks(
    response_payload: Dict[str, Any],
    resolved_focus: Optional[List[str]] = None,
    grounding_warnings: Optional[List[str]] = None,
    evidence: Optional[GameEvidence] = None,
    tool_results: Optional[Dict[str, Any]] = None,
) -> List[str]:
    existing_md = str(response_payload.get("detailed_markdown") or "")
    blocks: List[str] = []
    for focus in resolved_focus or []:
        header = FOCUS_SECTION_HEADERS.get(focus)
        if not header or header in existing_md:
            continue
        summary = _build_missing_focus_section_summary(
            response_payload,
            focus,
            grounding_warnings=grounding_warnings,
            evidence=evidence,
            tool_results=tool_results,
        )
        blocks.append(header)
        if summary:
            blocks.append(f"- {summary}")
    return blocks


def _populate_empty_focus_sections(
    markdown: str,
    response_payload: Dict[str, Any],
    resolved_focus: Optional[List[str]] = None,
    grounding_warnings: Optional[List[str]] = None,
    evidence: Optional[GameEvidence] = None,
    tool_results: Optional[Dict[str, Any]] = None,
) -> str:
    updated = str(markdown or "")
    if not updated:
        return updated

    for focus in resolved_focus or []:
        header = FOCUS_SECTION_HEADERS.get(focus)
        if (
            not header
            or header not in updated
            or _markdown_section_has_body(updated, header)
        ):
            continue
        summary = _build_missing_focus_section_summary(
            response_payload,
            focus,
            grounding_warnings=grounding_warnings,
            evidence=evidence,
            tool_results=tool_results,
        )
        if not summary:
            continue
        pattern = re.compile(rf"(?ms)^{re.escape(header)}\s*\n(?:\s*\n)*(?=^##\s+|\Z)")
        updated = pattern.sub(f"{header}\n- {summary}\n\n", updated, count=1)

    return _normalize_detailed_markdown_layout(updated)


def _normalize_completed_review_focus_sections(
    markdown: str,
    response_payload: Dict[str, Any],
    resolved_focus: Optional[List[str]] = None,
    grounding_warnings: Optional[List[str]] = None,
    evidence: Optional[GameEvidence] = None,
    tool_results: Optional[Dict[str, Any]] = None,
) -> str:
    updated = str(markdown or "")
    if (
        not updated
        or evidence is None
        or tool_results is None
        or not _is_completed_review(evidence)
    ):
        return updated

    for focus in resolved_focus or []:
        header = FOCUS_SECTION_HEADERS.get(focus)
        if not header or header not in updated:
            continue
        summary = _completed_review_focus_summary(
            focus,
            evidence,
            tool_results,
            response_payload=response_payload,
            grounding_warnings=grounding_warnings,
            prefer_payload=False,
        )
        cleaned_summary = _clean_summary_text(summary)
        if not cleaned_summary:
            continue
        updated = _replace_markdown_section_body(
            updated,
            header,
            f"- {cleaned_summary}",
        )

    return _normalize_detailed_markdown_layout(updated)


def _ensure_completed_review_markdown_sections(
    response_payload: Dict[str, Any],
    *,
    evidence: GameEvidence,
) -> None:
    if not _is_completed_review(evidence):
        return

    markdown = str(response_payload.get("detailed_markdown") or "")
    analysis = response_payload.get("analysis") or {}
    if not isinstance(analysis, dict):
        return

    section_candidates = (
        (
            "## 결과 진단",
            _clean_summary_text(analysis.get("verdict") or analysis.get("summary")),
        ),
        (
            "## 결과를 가른 이유",
            _clean_summary_text(
                next(
                    (
                        item
                        for item in (analysis.get("why_it_matters") or [])
                        if _clean_summary_text(item)
                    ),
                    analysis.get("summary"),
                )
            ),
        ),
        (
            "## 다시 볼 장면",
            _clean_summary_text(
                next(
                    (
                        item
                        for item in (analysis.get("watch_points") or [])
                        if _clean_summary_text(item)
                    ),
                    next(
                        (
                            item
                            for item in (analysis.get("swing_factors") or [])
                            if _clean_summary_text(item)
                        ),
                        analysis.get("verdict"),
                    ),
                )
            ),
        ),
    )

    updated = markdown
    for header, candidate in section_candidates:
        if not candidate:
            continue
        if header in updated:
            if _markdown_section_has_body(updated, header):
                continue
            pattern = re.compile(
                rf"(?ms)^{re.escape(header)}\s*(?:\n(?:\s*\n)*)?(?=^##\s+|\Z)"
            )
            updated = pattern.sub(f"{header}\n- {candidate}\n\n", updated, count=1)
            continue
        updated = (updated.rstrip() + "\n\n" + f"{header}\n- {candidate}").strip()

    response_payload["detailed_markdown"] = _normalize_detailed_markdown_layout(updated)


def _normalize_completed_review_analysis_sections(
    response_payload: Dict[str, Any],
    *,
    evidence: GameEvidence,
    tool_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not _is_completed_review(evidence):
        return response_payload

    analysis = response_payload.get("analysis") or {}
    if not isinstance(analysis, dict):
        return response_payload

    updated = deepcopy(response_payload)
    analysis = updated.get("analysis") or {}
    if not isinstance(analysis, dict):
        return updated

    clutch_moments = _clutch_moments(tool_results or {})
    top_moment = clutch_moments[0] if clutch_moments else None
    top_moment_base = _completed_clutch_moment_text(top_moment) if top_moment else None
    review_summary_item = _primary_review_summary_item(evidence)
    review_summary_is_clutch = _summary_item_is_clutch_like(review_summary_item)
    winner_name = _completed_winner_name(evidence)
    result_margin = _completed_result_margin(evidence)

    swing_items = [
        _clean_summary_text(item)
        for item in (analysis.get("swing_factors") or [])
        if _clean_summary_text(item)
    ]
    watch_items = [
        _clean_summary_text(item)
        for item in (analysis.get("watch_points") or [])
        if _clean_summary_text(item)
    ]
    why_items = [
        _clean_summary_text(item)
        for item in (analysis.get("why_it_matters") or [])
        if _clean_summary_text(item)
    ]
    lead_swing_factor = swing_items[0] if swing_items else None

    verdict = _clean_summary_text(analysis.get("verdict"))
    if lead_swing_factor and _summary_texts_overlap(verdict, lead_swing_factor):
        analysis["verdict"] = _build_completed_review_turning_point_verdict(
            winner_name=winner_name,
            top_moment=top_moment,
            review_summary_item=review_summary_item,
            review_summary_is_clutch=review_summary_is_clutch,
        )

    normalized_why_items: List[str] = []
    for item in why_items:
        candidate = item
        if top_moment and (
            _summary_texts_overlap(candidate, lead_swing_factor)
            or (top_moment_base and top_moment_base in candidate)
            or ("WPA" in candidate and "승부처" in candidate)
        ):
            if result_margin == 1:
                candidate = (
                    f"{winner_name}는 고레버리지 기회를 끝까지 지켜내며 결과를 만들었습니다."
                    if winner_name
                    else "고레버리지 기회를 끝까지 지켜낸 대응이 결과를 만들었습니다."
                )
            else:
                candidate = (
                    f"{winner_name}는 고레버리지 기회를 득점 흐름으로 연결하며 주도권을 잡았습니다."
                    if winner_name
                    else "고레버리지 기회를 득점 흐름으로 연결한 대응이 주도권을 만들었습니다."
                )
        _append_distinct_note_part(normalized_why_items, candidate)
    if normalized_why_items:
        analysis["why_it_matters"] = normalized_why_items[:3]

    normalized_watch_items: List[str] = []
    if top_moment:
        _append_distinct_note_part(
            normalized_watch_items,
            _build_completed_review_watch_point(
                top_moment=top_moment,
                review_summary_item=review_summary_item,
            ),
        )
    for item in watch_items:
        candidate = item
        if top_moment and (
            _summary_texts_overlap(candidate, lead_swing_factor)
            or (top_moment_base and top_moment_base in candidate)
        ):
            candidate = _build_completed_review_watch_point(
                top_moment=top_moment,
                review_summary_item=review_summary_item,
            )
        _append_distinct_note_part(normalized_watch_items, candidate)
    if normalized_watch_items:
        analysis["watch_points"] = normalized_watch_items[:3]

    return updated


def _normalize_scheduled_partial_analysis_sections(
    response_payload: Dict[str, Any],
    *,
    evidence: GameEvidence,
    grounding_reasons: List[str],
    tool_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not tool_results or not _is_scheduled_partial_context(
        evidence.game_status_bucket,
        grounding_reasons,
    ):
        return response_payload

    updated = deepcopy(response_payload)
    template = _build_scheduled_team_level_deterministic_analysis(
        evidence, tool_results
    )
    analysis = updated.get("analysis")
    if not isinstance(analysis, dict):
        analysis = {}
        updated["analysis"] = analysis

    summary = _clean_summary_text(analysis.get("summary"))
    if not summary or _texts_are_redundant(summary, analysis.get("verdict")):
        analysis["summary"] = template.get("summary") or summary or ""

    if template.get("verdict"):
        analysis["verdict"] = template["verdict"]
    if template.get("why_it_matters"):
        analysis["why_it_matters"] = list(template["why_it_matters"][:3])
    if template.get("swing_factors"):
        analysis["swing_factors"] = list(template["swing_factors"][:3])
    if template.get("watch_points"):
        analysis["watch_points"] = list(template["watch_points"][:3])
    if template.get("uncertainty"):
        analysis["uncertainty"] = list(template["uncertainty"][:2])

    rebuilt_note = _rebuild_scheduled_coach_note(analysis)
    if rebuilt_note:
        updated["coach_note"] = rebuilt_note

    return updated


def _align_completed_review_markdown_sections(
    markdown: str,
    response_payload: Dict[str, Any],
    *,
    evidence: GameEvidence,
) -> str:
    if not _is_completed_review(evidence):
        return str(markdown or "")

    analysis = response_payload.get("analysis") or {}
    if not isinstance(analysis, dict):
        return str(markdown or "")

    updated = str(markdown or "")
    section_candidates = (
        (
            "## 결과 진단",
            _clean_summary_text(analysis.get("verdict") or analysis.get("summary")),
        ),
        (
            "## 결과를 가른 이유",
            _clean_summary_text(
                next(
                    (
                        item
                        for item in (analysis.get("why_it_matters") or [])
                        if _clean_summary_text(item)
                    ),
                    analysis.get("summary"),
                )
            ),
        ),
        (
            "## 실제 전환점",
            _clean_summary_text(
                next(
                    (
                        item
                        for item in (analysis.get("swing_factors") or [])
                        if _clean_summary_text(item)
                    ),
                    None,
                )
            ),
        ),
        (
            "## 다시 볼 장면",
            _clean_summary_text(
                next(
                    (
                        item
                        for item in (analysis.get("watch_points") or [])
                        if _clean_summary_text(item)
                    ),
                    next(
                        (
                            item
                            for item in (analysis.get("swing_factors") or [])
                            if _clean_summary_text(item)
                        ),
                        analysis.get("verdict"),
                    ),
                )
            ),
        ),
    )

    for header, candidate in section_candidates:
        if not candidate:
            continue
        body = f"- {candidate}"
        if header in updated:
            updated = _replace_markdown_section_body(updated, header, body)
        else:
            updated = f"{updated.rstrip()}\n\n{header}\n{body}\n"

    return _normalize_detailed_markdown_layout(updated)


def _align_scheduled_partial_markdown_sections(
    markdown: str,
    response_payload: Dict[str, Any],
    *,
    evidence: GameEvidence,
    grounding_reasons: List[str],
) -> str:
    if not _is_scheduled_partial_context(
        evidence.game_status_bucket,
        grounding_reasons,
    ):
        return str(markdown or "")

    analysis = response_payload.get("analysis") or {}
    if not isinstance(analysis, dict):
        return str(markdown or "")

    updated = str(markdown or "")
    section_candidates = (
        (
            "## 코치 판단",
            _clean_summary_text(analysis.get("verdict") or analysis.get("summary")),
        ),
        (
            "## 왜 중요한가",
            _clean_summary_text(
                next(
                    (
                        item
                        for item in (analysis.get("why_it_matters") or [])
                        if _clean_summary_text(item)
                    ),
                    analysis.get("summary"),
                )
            ),
        ),
        (
            "## 승부 스윙 포인트",
            _clean_summary_text(
                next(
                    (
                        item
                        for item in (analysis.get("swing_factors") or [])
                        if _clean_summary_text(item)
                    ),
                    None,
                )
            ),
        ),
        (
            "## 체크 포인트",
            _clean_summary_text(
                next(
                    (
                        item
                        for item in (analysis.get("watch_points") or [])
                        if _clean_summary_text(item)
                    ),
                    None,
                )
            ),
        ),
    )

    for header, candidate in section_candidates:
        if not candidate:
            continue
        body = f"- {candidate}"
        if header in updated:
            updated = _replace_markdown_section_body(updated, header, body)
        else:
            updated = f"{updated.rstrip()}\n\n{header}\n{body}\n"

    return _normalize_detailed_markdown_layout(updated)


def _ensure_detailed_markdown(
    response_payload: Dict[str, Any],
    resolved_focus: Optional[List[str]] = None,
    grounding_warnings: Optional[List[str]] = None,
    evidence: Optional[GameEvidence] = None,
    tool_results: Optional[Dict[str, Any]] = None,
) -> None:
    """Ensure detailed_markdown contains all focus section headers."""
    existing_md = _normalize_completed_review_focus_sections(
        str(response_payload.get("detailed_markdown") or ""),
        response_payload,
        resolved_focus=resolved_focus,
        grounding_warnings=grounding_warnings,
        evidence=evidence,
        tool_results=tool_results,
    )
    existing_md = _populate_empty_focus_sections(
        existing_md,
        response_payload,
        resolved_focus=resolved_focus,
        grounding_warnings=grounding_warnings,
        evidence=evidence,
        tool_results=tool_results,
    )
    response_payload["detailed_markdown"] = existing_md
    missing_focus_sections = _build_missing_focus_section_blocks(
        response_payload,
        resolved_focus=resolved_focus,
        grounding_warnings=grounding_warnings,
        evidence=evidence,
        tool_results=tool_results,
    )
    if not missing_focus_sections and existing_md:
        return
    if not existing_md:
        analysis = response_payload.get("analysis", {})
        sections: List[str] = list(missing_focus_sections)
        verdict = analysis.get("verdict") or analysis.get("summary")
        if verdict:
            sections.append(f"- {verdict}")
        for key in ("why_it_matters", "swing_factors", "watch_points"):
            items = analysis.get(key) or []
            for item in items[:2]:
                sections.append(f"- {item}")
        if analysis.get("uncertainty"):
            for item in analysis["uncertainty"][:1]:
                sections.append(f"- {item}")
        if sections:
            response_payload["detailed_markdown"] = _normalize_detailed_markdown_layout(
                "\n".join(sections)
            )
    else:
        prefix = "\n".join(missing_focus_sections)
        insertion_end = 0
        for focus in resolved_focus or []:
            header = FOCUS_SECTION_HEADERS.get(focus)
            if not header:
                continue
            span = _find_markdown_section_span(existing_md, header)
            if span:
                insertion_end = max(insertion_end, span[1])
        if insertion_end > 0:
            combined = (
                existing_md[:insertion_end].rstrip()
                + "\n\n"
                + prefix
                + "\n\n"
                + existing_md[insertion_end:].lstrip()
            )
        else:
            combined = prefix + "\n\n" + existing_md
        response_payload["detailed_markdown"] = _normalize_detailed_markdown_layout(
            combined
        )


def _build_deterministic_coach_response(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
    resolved_focus: Optional[List[str]] = None,
    grounding_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    analysis = _build_deterministic_analysis(evidence, tool_results)
    coach_note_parts: List[str] = []
    if analysis.get("verdict"):
        _append_distinct_note_part(coach_note_parts, str(analysis["verdict"]))
    if analysis.get("swing_factors"):
        _append_distinct_note_part(coach_note_parts, str(analysis["swing_factors"][0]))
    if analysis.get("risks"):
        _append_distinct_note_part(
            coach_note_parts, str(analysis["risks"][0]["description"])
        )
    if evidence.stage_label != "REGULAR" and evidence.series_state:
        _append_distinct_note_part(
            coach_note_parts,
            evidence.series_state.summary_text(
                evidence.home_team_name, evidence.away_team_name
            ),
        )
    if not coach_note_parts:
        _append_distinct_note_part(
            coach_note_parts,
            (
                "확인된 실데이터 기준으로는 결과를 가른 운영 장면과 득점 연결 구간을 함께 복기해야 합니다."
                if _is_completed_review(evidence)
                else "확인된 실데이터 기준으로는 최근 흐름과 정규시즌 베이스라인을 함께 봐야 합니다."
            ),
        )

    response = CoachResponse(
        headline=_build_deterministic_headline(evidence, tool_results),
        sentiment="neutral",
        key_metrics=_build_deterministic_metrics(evidence, tool_results),
        analysis=analysis,
        detailed_markdown=_build_deterministic_markdown(
            evidence,
            tool_results,
            resolved_focus,
            grounding_warnings=grounding_warnings,
        ),
        coach_note=_build_compact_coach_note(coach_note_parts),
    )
    return response.model_dump()


def _postprocess_coach_response_payload(
    response_payload: Dict[str, Any],
    *,
    evidence: GameEvidence,
    used_evidence: List[str],
    grounding_reasons: List[str],
    grounding_warnings: Optional[List[str]] = None,
    tool_results: Optional[Dict[str, Any]] = None,
    resolved_focus: Optional[List[str]] = None,
) -> Dict[str, Any]:
    processed = _sanitize_response_placeholders(
        response_payload,
        used_evidence=used_evidence,
    )
    processed = _normalize_response_team_display(
        processed,
        evidence=evidence,
    )
    processed = _soften_scheduled_partial_tone(
        processed,
        game_status_bucket=evidence.game_status_bucket,
        grounding_reasons=grounding_reasons,
    )
    processed = _normalize_response_markdown_layout(processed)
    processed = _cleanup_response_language_quality(processed)
    processed = _polish_scheduled_partial_response(
        processed,
        game_status_bucket=evidence.game_status_bucket,
        grounding_reasons=grounding_reasons,
    )
    processed = _sanitize_scheduled_unconfirmed_lineup_entities(
        processed,
        evidence=evidence,
        grounding_reasons=grounding_reasons,
        tool_results=tool_results,
        resolved_focus=resolved_focus,
    )
    processed = _normalize_scheduled_partial_analysis_sections(
        processed,
        evidence=evidence,
        grounding_reasons=grounding_reasons,
        tool_results=tool_results,
    )
    processed["detailed_markdown"] = _align_scheduled_partial_markdown_sections(
        str(processed.get("detailed_markdown") or ""),
        processed,
        evidence=evidence,
        grounding_reasons=grounding_reasons,
    )
    processed = _enforce_completed_result_anchor(
        processed,
        evidence=evidence,
        tool_results=tool_results,
        resolved_focus=resolved_focus,
        grounding_warnings=grounding_warnings,
    )
    processed = _normalize_completed_review_analysis_sections(
        processed,
        evidence=evidence,
        tool_results=tool_results,
    )
    _ensure_completed_review_markdown_sections(
        processed,
        evidence=evidence,
    )
    processed["detailed_markdown"] = _align_completed_review_markdown_sections(
        str(processed.get("detailed_markdown") or ""),
        processed,
        evidence=evidence,
    )
    processed["detailed_markdown"] = _normalize_completed_review_focus_sections(
        str(processed.get("detailed_markdown") or ""),
        processed,
        resolved_focus=resolved_focus,
        grounding_warnings=grounding_warnings,
        evidence=evidence,
        tool_results=tool_results,
    )
    return processed


def _collect_allowed_entity_names(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> Set[str]:
    names: Set[str] = {
        evidence.home_team_name,
        evidence.away_team_name,
        evidence.home_team_code,
        evidence.away_team_code or "",
    }
    for name in [
        evidence.home_pitcher,
        evidence.away_pitcher,
        *evidence.home_lineup,
        *evidence.away_lineup,
    ]:
        normalized = _normalize_name_token(name)
        if normalized:
            names.add(normalized)
    for team_key in ("home", "away"):
        summary = tool_results.get(team_key, {}).get("summary", {})
        for bucket in ("top_batters", "top_pitchers"):
            for item in summary.get(bucket, [])[:8]:
                normalized = _normalize_name_token(item.get("player_name"))
                if normalized:
                    names.add(normalized)
        form_signals = tool_results.get(team_key, {}).get("player_form_signals", {})
        for bucket in ("batters", "pitchers"):
            for item in form_signals.get(bucket, [])[:4]:
                normalized = _normalize_name_token(item.get("player_name"))
                if normalized:
                    names.add(normalized)
    # All WPA clutch moments (not just top 6) for broader coverage
    clutch_moments = _clutch_moments(tool_results)
    for item in clutch_moments:
        for key in ("batter_name", "pitcher_name"):
            normalized = _normalize_name_token(item.get(key))
            if normalized:
                names.add(normalized)
    # Head-to-head matchup player names (starting pitchers, key batters)
    matchup = tool_results.get("matchup", {}) or {}
    for game in (matchup.get("games") or [])[:10]:
        if isinstance(game, dict):
            for key in (
                "home_pitcher",
                "away_pitcher",
                "player_name",
                "batter_name",
                "pitcher_name",
                "winning_pitcher",
                "losing_pitcher",
                "save_pitcher",
            ):
                normalized = _normalize_name_token(game.get(key))
                if normalized:
                    names.add(normalized)
    for item in evidence.summary_items:
        for token in _collect_grounding_candidates(str(item or "")):
            if token in COMMON_ENTITY_STOPWORDS or token in ENGLISH_ENTITY_STOPWORDS:
                continue
            names.add(token)
    return {name for name in names if name}


COMMON_ENTITY_STOPWORDS = {
    "정규시즌",
    "포스트시즌",
    "한국시리즈",
    "와일드카드",
    "준플레이오프",
    "플레이오프",
    "선발",
    "불펜",
    "타선",
    "라인업",
    "최근",
    "경기",
    "분석",
    "브리핑",
    "시리즈",
    "실데이터",
    "관전",
    "포인트",
    "리그",
    "평균",
    "상태",
    "공격력",
    "경기력",
    "장타력",
    "수비력",
    "기동력",
    "집중력",
    "박진감",
    "선발진",
    "선취점",
    "소화력",
    "소화해",
    "연장전",
    "최하위",
    "반격할",
    "최소한",
    "유사한",
    "유사해",
    "고위험",
    "저위험",
    "부족으",
    "공개될",
}
ENGLISH_ENTITY_STOPWORDS = {
    "Raw",
    "JSON",
    "Coach",
    "Note",
    "Home",
    "Away",
    "Matchup",
    "Recent",
    "Season",
    "Series",
    "Regular",
    "Postseason",
}
GENERIC_ENTITY_ROLE_PREFIXES = ("선수", "타자", "투수")
PLAYER_CONTEXT_KEYWORDS = (
    "승부처",
    "타석",
    "타자",
    "투수",
    "선발",
    "불펜",
    "라인업",
    "타순",
    "출루",
    "장타",
    "홈런",
    "안타",
    "타격",
    "타격감",
    "구위",
    "등판",
    "실점",
    "삼진",
    "볼넷",
    "OPS",
    "ERA",
    "WPA",
    "클러치",
    "폼",
    "마운드",
)
PLAYER_NAME_PRECEDING_CUES = (
    "결승타",
    "결승포",
    "결승홈런",
    "적시타",
    "홈런",
    "안타",
    "타점",
    "장타",
    "출루",
    "선발",
    "불펜",
    "타자",
    "투수",
    "포수",
    "내야수",
    "외야수",
    "대타",
    "대주자",
)
PLAYER_NAME_FOLLOWING_CUES = (
    "OPS",
    "wRC+",
    "타석",
    "장타",
    "출루",
    "홈런",
    "안타",
    "타점",
    "볼넷",
    "삼진",
    "컨디션",
    "폼",
    "중심",
    "대응",
    "출전",
    "등판",
    "공략",
)
KOREAN_SURNAME_PATTERN = (
    "김|이|박|최|정|강|조|윤|장|임|한|오|서|신|권|황|안|송|류|전|고|문|양|"
    "손|배|백|허|유|남|심|노|하|곽|성|차|주|우|구|민|진|지|엄|채|원|천|"
    "방|공|현|함|변|염|여|추|도|소|석|선|설|마|길|표|명|기|반|왕|금|옥|"
    "육|인|맹|제|모|온|계|탁|국|연|어|은|편|용|예|경|봉|사|부|황보|독고|"
    "제갈|서문|남궁"
)
KOREAN_PLAYER_NAME_BASE_RE = re.compile(rf"^(?:{KOREAN_SURNAME_PATTERN})[가-힣]{{2}}$")
ENGLISH_PLAYER_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
KOREAN_PARTICLE_SUFFIXES = (
    "이",
    "가",
    "은",
    "는",
    "을",
    "를",
    "와",
    "과",
    "도",
    "만",
    "의",
    "에",
    "께",
    "로",
)
ENTITY_REPLACEMENT_SUFFIXES = (
    "으로",
    "에게",
    "께",
    "이",
    "가",
    "은",
    "는",
    "을",
    "를",
    "와",
    "과",
    "도",
    "만",
    "의",
    "에",
    "로",
)
TEAM_CODE_SUFFIXES = tuple(dict.fromkeys(KOREAN_PARTICLE_SUFFIXES + ("보다",)))
TEAM_DISPLAY_ALIASES_BY_CODE: Dict[str, Tuple[str, ...]] = {
    "KIA": ("기아",),
    "LG": ("LG",),
    "SSG": ("SSG", "SK"),
    "NC": ("NC",),
    "DB": ("두산",),
    "KT": ("KT",),
    "LT": ("롯데",),
    "SS": ("삼성",),
    "HH": ("한화",),
    "KH": ("키움", "넥센"),
}
ENTITY_SUFFIX_PLACEHOLDERS = {
    "으로": "핵심 선수로",
    "에게": "핵심 선수에게",
    "께": "핵심 선수께",
    "이": "핵심 선수가",
    "가": "핵심 선수가",
    "은": "핵심 선수는",
    "는": "핵심 선수는",
    "을": "핵심 선수를",
    "를": "핵심 선수를",
    "와": "핵심 선수와",
    "과": "핵심 선수와",
    "도": "핵심 선수도",
    "만": "핵심 선수만",
    "의": "핵심 선수의",
    "에": "핵심 선수에",
    "로": "핵심 선수로",
}
PLACEHOLDER_LITERAL_PATTERN = re.compile(
    r"(?:\bNone%|\bNone\b|\bN/A\b|\bn/a\b|\bnull\b|\bundefined\b|\bNaN\b|\bnan\b)"
)
TEAM_ZERO_RECORD_PLACEHOLDER_PATTERN = re.compile(
    r"([가-힣A-Za-z]{2,})\s*0승\s*0패(?=\s*/|\s*$)"
)
ZERO_RECORD_PLACEHOLDER_PATTERN = re.compile(r"0승\s*0패")
DUPLICATE_PLACEHOLDER_PHRASE_PATTERN = re.compile(r"데이터 부족(?:와|과)\s*데이터 부족")
UNSUPPORTED_RECENT_FORM_CLAUSE_PATTERN = re.compile(
    r"[^,.!?\n]*최근\s*(?:경기|흐름)[^,.!?\n]*"
    r"(?:승리|패배|연승|연패|승패|상승세|하락세|호조|부진)[^,.!?\n]*(?:[,.!?]\s*|$)"
)
UNSUPPORTED_RECENT_FORM_REPLACEMENTS = (
    (
        re.compile(r"최근 경기에서 승리를 거두지 못하고 있으며,\s*"),
        "최근 흐름 근거가 부족하며 ",
    ),
    (
        re.compile(r"최근 경기에서 승리를 거두지 못해도\s*"),
        "최근 흐름 근거가 부족하지만 ",
    ),
    (
        re.compile(r"최근 경기에서 연패 흐름이라도\s*"),
        "최근 흐름 근거가 부족하지만 ",
    ),
    (
        re.compile(r"최근 흐름이 하락세라\s*"),
        "최근 흐름 근거가 부족하며 ",
    ),
    (
        re.compile(r"최근 흐름이 상승세라\s*"),
        "최근 흐름 근거가 부족하며 ",
    ),
)
NON_KOREAN_CJK_PATTERN = re.compile(r"[\u4E00-\u9FFF]+")
LANGUAGE_ARTIFACT_REPLACEMENTS = (
    ("정규시즌开幕", "정규시즌 개막"),
    ("중盤", "중반"),
    ("中盤", "중반"),
    ("也无法 확인하여", "도 확인할 수 없어"),
    ("也无法", "도"),
    ("开幕", "개막"),
)
SCHEDULED_PARTIAL_TONE_REPLACEMENTS = (
    ("압도적 득실 마진", "뚜렷한 득실 마진"),
    ("압도적인 득실 마진", "뚜렷한 득실 마진"),
    ("압도적 우위", "뚜렷한 우세 흐름"),
    ("압도적인 우위", "우세 흐름"),
    ("압도적인 최근 승리 흐름", "뚜렷한 최근 우세 흐름"),
    ("압도적인 승리 흐름", "뚜렷한 우세 흐름"),
    ("압도적인 최근 흐름", "뚜렷한 최근 우세 흐름"),
    ("압도적인 득실 마진 우위", "뚜렷한 득실 마진 우위"),
    ("압도적으로 높습니다.", "확연히 앞섭니다."),
    ("압도적인 우위를 점하며", "우세 흐름을 보이며"),
    ("압도적인 우위로", "우세 흐름으로"),
    ("뚜렷한 우세 흐름는 명확하나", "우세 흐름은 확인되나"),
    ("우위는 명확하나", "우세 흐름은 확인되나"),
    ("승리 가능성을 높이지만", "우세 흐름을 뒷받침하지만"),
    ("승리 가능성을 높입니다.", "우세 흐름을 뒷받침합니다."),
    ("승리 가능성을 높일 수 있습니다.", "우세 흐름을 뒷받침할 수 있습니다."),
    ("실제 경기 결과에 영향을 줄 수 있습니다.", "경기 흐름에 영향을 줄 수 있습니다."),
    ("유리세를 확보했습니다.", "우세 흐름을 보입니다."),
    ("유리세를 점하고 있으나", "우세 흐름을 보이고 있으나"),
    ("유리세를 점하고", "우세 흐름을 보이며"),
    ("유리세를 확보", "우세 흐름을 보임"),
    ("유리세 확보", "우세 흐름"),
    ("유리세", "우세 흐름"),
    ("승기를 잡아야 합니다.", "우세 흐름을 이어가야 합니다."),
    ("승기를 잡아야", "우세 흐름을 이어가야"),
    ("우세한 흐름을 보입니다.", "우세 흐름을 보입니다."),
    ("경기 전 유리세", "경기 전 우세 흐름"),
)
SCHEDULED_REVIEW_LANGUAGE_REPLACEMENTS = (
    ("핵심 선수 활용", "핵심 전력 운용"),
    ("승부처로 작용했으나", "승부 변수로 볼 수 있으나"),
    ("승부처로 작용했지만", "승부 변수로 남지만"),
    ("클러치 성과", "승부처 대응"),
    ("승부처 대응가", "승부처 대응이"),
    ("클러치", "승부처"),
    ("WPA 변동", "운영 변동"),
    ("WPA", "운영 지표"),
    ("초반 선취점 이후", "초반 흐름에서"),
    ("초반 리드를 확보했으나", "초반 흐름을 선점하고 있으나"),
    ("우세를 보였으나", "우세 흐름을 보이고 있으나"),
    ("전환할 수 있었습니다.", "전환할 수 있습니다."),
    ("중요했습니다.", "중요한 변수입니다."),
    ("관전자에게는", "관전 포인트로는"),
)
SCHEDULED_USER_FACING_REPLACEMENTS = (
    ("하이 레버리지", "접전 후반"),
    ("고레버리지", "접전 후반"),
    ("운영 변수", "경기 후반 변수"),
    ("승부처", "핵심 구간"),
    ("불펜 핵심 선수", "불펜 핵심 자원"),
    ("최근 WPA 변동", "최근 운영 지표"),
    ("최근 운영 변동", "최근 운영 지표"),
    ("데이터 결여", "데이터 부족"),
    ("후반전 대응력", "경기 후반 대응력"),
    ("비교 불가능", "비교가 어렵습니다"),
    ("전략 비교 미가능", "전략 비교가 어렵습니다"),
    ("## 결과 진단", "## 코치 판단"),
    ("## 결과를 가른 이유", "## 왜 중요한가"),
    ("## 다시 볼 장면", "## 체크 포인트"),
    ("불펜 비중에 대한 공식 데이터가 없어", "불펜 운용 관련 공식 데이터가 없어"),
    ("불펜 비중 데이터가 없으며", "불펜 운용 데이터가 부족하며"),
    ("불펜 비중 데이터가 결여됨", "불펜 운용 데이터가 부족함"),
    ("불펜 비중 데이터가 결여되어", "불펜 운용 데이터가 부족해"),
    ("불펜 비중이 데이터 부족으로", "불펜 운용 데이터가 부족해"),
    ("불펜 비중이 데이터 부족", "불펜 운용 데이터가 부족"),
    ("비교 불가하다", "비교가 어렵다"),
    ("비교할 수 없다", "비교가 어렵다"),
    ("판단을 할 수 없다", "판단이 어렵다"),
    ("판단할 수 없다", "판단이 어렵다"),
    ("실제로 어떻게 활용되는지", "실제로 어떻게 쓰이는지"),
    ("실제로 어떻게 쓰이는지 확인한다.", "실제로 어떻게 쓰이는지 확인해야 합니다."),
    ("실제 대응력", "실전 대응력"),
    ("## 결론", "## 코치 판단"),
)
RESPONSE_LANGUAGE_QUALITY_REPLACEMENTS = (
    ("양팀 데이터 부족 비중", "양 팀 모두 데이터 부족"),
    ("양팀", "양 팀"),
    ("경기 경기 후반", "경기 후반"),
    ("고레버리지", "접전 후반"),
    ("불펜 비중 확인 필요", "불펜 운용 정보 확인 필요"),
    ("불펜 비중 데이터가 부족하여", "불펜 운용 데이터가 부족하여"),
    ("불펜 비중 데이터 부족으로", "불펜 운용 데이터 부족으로"),
    ("불펜 비중 데이터 부족", "불펜 운용 데이터 부족"),
    ("불펜 비중 정보가 없어", "불펜 운용 정보가 없어"),
    ("불펜 비중 정보 없음", "불펜 운용 정보 없음"),
    ("불펜 비중이 공개되지 않아", "불펜 운용 정보가 공개되지 않아"),
    ("선발 발표 시", "선발 발표 후"),
    ("실제 결과로 연결했으나", "우세 흐름으로 이어질 수 있으나"),
    ("불펜 부재", "불펜 정보 부족"),
    ("정보 부족로", "정보 부족으로"),
    ("핵심 선수 가능성", "주도 가능성"),
    ("핵심 선수 미공개", "핵심 전력 정보 미공개"),
    ("핵심 선수 예측 가능 여부", "핵심 전력 윤곽 확인 필요"),
    ("핵심 선수 예측", "핵심 전력 윤곽 판단"),
    ("핵심 선수 여부", "핵심 전력 여부"),
    ("불펜 미정형", "불펜 변수"),
    ("경기 후반 접전 후반 상황", "경기 후반 접전 상황"),
    ("핵심 구간가", "핵심 구간이"),
    ("핵심 구간를", "핵심 구간을"),
    ("walk-off", "결승타"),
    ("late inning", "경기 후반"),
)
RESPONSE_LANGUAGE_QUALITY_REGEX_REPLACEMENTS = (
    (re.compile(r"(?<![가-힣A-Za-z])선발\s+선발(?=\s|$)"), "선발"),
    (re.compile(r"예측 불가능"), "예측이 어렵습니다"),
    (re.compile(r"([가-힣])\.(?=[A-Za-z가-힣])"), r"\1 "),
    (
        re.compile(
            r"([A-Za-z]{2,5}\s+[가-힣]*?(?:랜더스|이글스|라이온즈|트윈스|베어스|위즈|다이노스|타이거즈|자이언츠|히어로즈|드래곤즈|이글|트윈|베어|위자드|다이노|타이거|자이언트|히어로))이(?=\s)"
        ),
        r"\1가",
    ),
    (
        re.compile(
            r"([가-힣]*?(?:랜더스|이글스|라이온즈|트윈스|베어스|위즈|다이노스|타이거즈|자이언츠|히어로즈|드래곤즈|이글|트윈|베어|위자드|다이노|타이거|자이언트|히어로))이(?=\s)"
        ),
        r"\1가",
    ),
)
UNCONFIRMED_LINEUP_ARTIFACT_MARKERS = (
    "상위 타선",
    "상위 타순",
    "핵심 선수",
)
COMPLETED_RESULT_DIRECTION_MARKERS = (
    "승리",
    "이겼",
    "잡았",
    "가져갔",
    "웃었",
    "실제 결과로 연결",
    "승부처를 확보",
    "경기를 가져",
    "승부를 가져",
    "흐름을 굳혔",
    "결과를 만들",
)


def _normalize_allowed_names_for_grounding(allowed_names: Set[str]) -> Set[str]:
    normalized: Set[str] = set()
    for raw_name in allowed_names:
        clean_name = _normalize_name_token(raw_name)
        if not clean_name:
            continue
        normalized.add(clean_name)
        normalized.add(clean_name.lower())
        for token in re.split(r"[\s,/()]+", clean_name):
            token = token.strip()
            if not token:
                continue
            normalized.add(token)
            normalized.add(token.lower())
    return normalized


def _collect_grounding_candidates(text: str) -> Set[str]:
    candidates: Set[str] = set()
    for token in re.findall(r"[가-힣]{2,6}", text):
        normalized = token
        if any(
            normalized.startswith(prefix) for prefix in GENERIC_ENTITY_ROLE_PREFIXES
        ):
            continue
        normalized = _strip_entity_suffix(normalized)
        if (
            KOREAN_PLAYER_NAME_BASE_RE.fullmatch(normalized)
            and normalized[-1] not in KOREAN_PARTICLE_SUFFIXES
        ):
            candidates.add(normalized)
            continue
    for token in ENGLISH_PLAYER_NAME_RE.findall(text):
        if token in ENGLISH_ENTITY_STOPWORDS:
            continue
        candidates.add(token)
    return candidates


def _strip_entity_suffix(token: str) -> str:
    normalized = str(token or "").strip()
    if len(normalized) < 3:
        return normalized
    suffixes = sorted(
        set(ENTITY_REPLACEMENT_SUFFIXES) | set(KOREAN_PARTICLE_SUFFIXES),
        key=len,
        reverse=True,
    )
    for suffix in suffixes:
        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 2:
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _segment_has_name_like_usage(segment: str, token: str) -> bool:
    if not segment or not token:
        return False
    escaped_token = re.escape(token)
    particle_pattern = "|".join(
        re.escape(suffix)
        for suffix in sorted(KOREAN_PARTICLE_SUFFIXES, key=len, reverse=True)
    )
    preceding_pattern = "|".join(
        re.escape(cue)
        for cue in sorted(PLAYER_NAME_PRECEDING_CUES, key=len, reverse=True)
    )
    following_pattern = "|".join(
        re.escape(cue)
        for cue in sorted(PLAYER_NAME_FOLLOWING_CUES, key=len, reverse=True)
    )
    patterns = (
        rf"{escaped_token}(?:{particle_pattern})(?=[^가-힣A-Za-z]|$)",
        rf"(?:{preceding_pattern})\s+{escaped_token}(?=[^가-힣A-Za-z]|$)",
        rf"{escaped_token}(?=\s*(?:{following_pattern})(?:[^가-힣A-Za-z]|$))",
        rf"{escaped_token}\s*\(",
    )
    return any(re.search(pattern, segment) for pattern in patterns)


def _segment_has_player_context(segment: str, token: str) -> bool:
    if not segment or not token:
        return False
    start = 0
    while True:
        index = segment.find(token, start)
        if index == -1:
            return False
        window = segment[max(0, index - 12) : index + len(token) + 12]
        if any(
            keyword in window for keyword in PLAYER_CONTEXT_KEYWORDS
        ) and _segment_has_name_like_usage(segment, token):
            return True
        start = index + len(token)


def _find_disallowed_entities(
    response_data: Dict[str, Any],
    allowed_names: Set[str],
    *,
    min_occurrences: int = 2,
    allow_empty_allowed_names: bool = False,
) -> List[str]:
    if not allowed_names and not allow_empty_allowed_names:
        return []
    allowed_lookup = (
        _normalize_allowed_names_for_grounding(allowed_names)
        if allowed_names
        else set()
    )
    text_segments = [
        str(response_data.get("headline") or ""),
        str(response_data.get("detailed_markdown") or ""),
        str(response_data.get("coach_note") or ""),
        str(response_data.get("analysis", {}).get("summary") or ""),
        str(response_data.get("analysis", {}).get("verdict") or ""),
        " ".join(response_data.get("analysis", {}).get("strengths", []) or []),
        " ".join(response_data.get("analysis", {}).get("weaknesses", []) or []),
        " ".join(response_data.get("analysis", {}).get("why_it_matters", []) or []),
        " ".join(response_data.get("analysis", {}).get("swing_factors", []) or []),
        " ".join(response_data.get("analysis", {}).get("watch_points", []) or []),
        " ".join(response_data.get("analysis", {}).get("uncertainty", []) or []),
    ]
    text_segments = [segment for segment in text_segments if segment]
    if not text_segments:
        return []

    token_occurrences: Dict[str, int] = {}
    for segment in text_segments:
        for token in _collect_grounding_candidates(segment):
            if not _segment_has_player_context(segment, token):
                continue
            token_occurrences[token] = token_occurrences.get(token, 0) + 1

    disallowed_tokens: List[str] = []
    for token, occurrences in token_occurrences.items():
        if token in COMMON_ENTITY_STOPWORDS:
            continue
        if token in allowed_lookup or token.lower() in allowed_lookup:
            continue
        if token.endswith("즈") or token.endswith("스") or token.endswith("스파크"):
            continue
        if occurrences < max(1, int(min_occurrences)):
            continue
        disallowed_tokens.append(token)
    return list(dict.fromkeys(disallowed_tokens))


def _collect_confirmed_lineup_entity_names(evidence: GameEvidence) -> Set[str]:
    names: Set[str] = set()
    for raw_name in (
        list(evidence.home_lineup)
        + list(evidence.away_lineup)
        + [evidence.home_pitcher, evidence.away_pitcher]
    ):
        normalized = _normalize_name_token(raw_name)
        if normalized:
            names.add(normalized)
    return names


def _find_unconfirmed_lineup_named_players(
    response_data: Dict[str, Any],
    protected_names: Set[str],
) -> List[str]:
    false_positive_tokens = {"변수지"}
    protected_lookup = _normalize_allowed_names_for_grounding(protected_names)
    text_segments = [
        str(response_data.get("headline") or ""),
        str(response_data.get("detailed_markdown") or ""),
        str(response_data.get("coach_note") or ""),
        str(response_data.get("analysis", {}).get("summary") or ""),
        str(response_data.get("analysis", {}).get("verdict") or ""),
        " ".join(response_data.get("analysis", {}).get("strengths", []) or []),
        " ".join(response_data.get("analysis", {}).get("weaknesses", []) or []),
        " ".join(response_data.get("analysis", {}).get("why_it_matters", []) or []),
        " ".join(response_data.get("analysis", {}).get("swing_factors", []) or []),
        " ".join(response_data.get("analysis", {}).get("watch_points", []) or []),
        " ".join(response_data.get("analysis", {}).get("uncertainty", []) or []),
    ]
    for metric in response_data.get("key_metrics", []) or []:
        if not isinstance(metric, dict):
            continue
        text_segments.append(str(metric.get("label") or ""))
        text_segments.append(str(metric.get("value") or ""))

    candidate_patterns = (
        re.compile(
            r"([가-힣]{2,6})\s*(?:와|과)\s*([가-힣]{2,6})(?=\s*(?:비교|의\s+시즌\s+OPS\s+비교|를\s+동시에\s+막아야))"
        ),
        re.compile(r"([가-힣]{2,6})(?=\(OPS\s*[0-9.]+\))"),
        re.compile(r"([가-힣]{2,6})(?=\(wRC\+\s*[0-9.]+\))"),
        re.compile(r"([가-힣]{2,6})(?=\s+OPS\s*[0-9.]+)"),
        re.compile(r"([가-힣]{2,6})(?=\s+폼)"),
        re.compile(r"([가-힣]{2,6})(?=\s*(?:와|과)\s+[^.\n]{0,20}?상위 타선\s+폼)"),
        re.compile(r"([가-힣]{2,6})(?=의\s+(?:높은\s+)?wRC\+)"),
        re.compile(
            r"([가-힣]{2,6})(?=의\s+[^.\n]{0,24}?(?:OPS|wRC\+|폼|타석|장타|출루))"
        ),
        re.compile(r"([가-힣]{2,6})(?=\s+첫\s+타석)"),
        re.compile(r"([가-힣]{2,6})(?=\s+장타)"),
        re.compile(r"([가-힣]{2,6})(?=\s+출루(?:율| 여부)?)"),
        re.compile(r"([가-힣]{2,6})(?=\s+중심\s+대응)"),
        re.compile(r"([가-힣]{2,6})(?=\s+컨디션)"),
        re.compile(r"([가-힣]{2,6})(?=의\s+시즌\s+OPS\s+비교)"),
    )

    tokens: List[str] = []
    for segment in text_segments:
        if not segment:
            continue
        for pattern in candidate_patterns:
            for match in pattern.finditer(segment):
                for raw_token in match.groups():
                    token = _normalize_name_token(raw_token)
                    if not token:
                        continue
                    if token in false_positive_tokens:
                        continue
                    if token in COMMON_ENTITY_STOPWORDS:
                        continue
                    if token in protected_lookup or token.lower() in protected_lookup:
                        continue
                    if (
                        token.endswith("즈")
                        or token.endswith("스")
                        or token.endswith("스파크")
                    ):
                        continue
                    tokens.append(token)
    return list(dict.fromkeys(tokens))


def _cleanup_unconfirmed_lineup_player_placeholders(
    response_data: Dict[str, Any],
) -> Dict[str, Any]:
    cleaned = deepcopy(response_data)

    def _cleanup_text(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        updated = value
        updated = re.sub(
            r"핵심 선수 OPS [0-9.]+\s*/\s*핵심 선수 OPS [0-9.]+",
            "라인업 미확정으로 개별 타자 비교 보류",
            updated,
        )
        updated = re.sub(
            r"핵심 선수\(OPS [0-9.]+\)",
            "상위 타선",
            updated,
        )
        updated = re.sub(
            r"핵심 선수\(wRC\+\s*[0-9.]+\)",
            "상위 타선",
            updated,
        )
        updated = re.sub(
            r"핵심 선수의 높은 wRC\+\s*[0-9.]+\s*및 OPS\s*[0-9.]+",
            "상위 타선의 높은 생산성",
            updated,
        )
        updated = re.sub(
            r"핵심 선수의\s+[^.\n]{0,24}?(?:OPS|wRC\+|폼)\s*\(?[0-9.]*\)?",
            "상위 타선의 핵심 생산성",
            updated,
        )
        updated = updated.replace(
            "핵심 선수 등 상위 타선의 생산성",
            "상위 타선의 생산성",
        )
        updated = updated.replace("핵심 선수", "상위 타선")
        updated = updated.replace("상위 타선와", "상위 타선과")
        updated = updated.replace("상위 타선를", "상위 타선을")
        updated = updated.replace("상위 타선 수 없으며", "파악할 수 없으며")
        updated = updated.replace(
            "개별 상위 타선 폼 비교 보류", "개별 타자 폼 비교 보류"
        )
        updated = updated.replace("극단적인 타격 상위 타선", "극단적인 타격 생산성")
        updated = updated.replace(
            "한화 이글스 양 팀 상위 타선 폼 유지 여부", "양 팀 상위 타선 폼 유지 여부"
        )
        updated = updated.replace("경기 경기 후반", "경기 후반")
        updated = updated.replace("핵심 구간가", "핵심 구간이")
        updated = updated.replace(
            "상위 타선 등 상위 타선의 생산성", "상위 타선의 생산성"
        )
        updated = updated.replace("상위 타선 등 상위 타선", "상위 타선")
        updated = updated.replace(
            "상위 타선과 상위 타선 폼 유지 여부", "양 팀 상위 타선 폼 유지 여부"
        )
        updated = updated.replace(
            "상위 타선와 상위 타선 폼 유지 여부", "양 팀 상위 타선 폼 유지 여부"
        )
        updated = re.sub(
            r"상위 타선\s*(?:와|과)\s+[A-Za-z가-힣\s]{0,20}?상위 타선\s+폼 유지 여부",
            "양 팀 상위 타선 폼 유지 여부",
            updated,
        )
        updated = re.sub(
            r"상위 타선\(wRC\+\s*[0-9.]+\)\s*(?:와|과)\s*상위 타선\(wRC\+\s*[0-9.]+\)",
            "양 팀 상위 타선",
            updated,
        )
        updated = re.sub(
            r"상위 타선\(wRC\+\s*[0-9.]+\)\s*(?:와|과)\s*상위 타선의 타격 흐름 지속 여부",
            "양 팀 상위 타선의 타격 흐름 지속 여부",
            updated,
        )
        updated = re.sub(
            r"[A-Za-z가-힣\s]{0,16}양 팀 상위 타선 폼 유지 여부",
            "양 팀 상위 타선 폼 유지 여부",
            updated,
        )
        updated = re.sub(
            r"((?:[A-Za-z]{2,5}\s+)?[가-힣]+(?:\s+[가-힣]+)?\s+상위 타선)\s+극단적인 타격 생산성",
            r"\1의 극단적인 타격 생산성",
            updated,
        )
        updated = updated.replace("상위 타선과 상위 타선 비교", "양 팀 상위 타선 비교")
        updated = updated.replace("상위 타선와 상위 타선 비교", "양 팀 상위 타선 비교")
        updated = updated.replace(
            "상위 타선과 상위 타선의 시즌 OPS 비교", "양 팀 상위 타선의 시즌 OPS 비교"
        )
        updated = updated.replace(
            "상위 타선와 상위 타선의 시즌 OPS 비교", "양 팀 상위 타선의 시즌 OPS 비교"
        )
        updated = updated.replace(
            "상위 타선과 상위 타선을 동시에 막아야 합니다.",
            "양 팀 상위 타선을 함께 제어해야 합니다.",
        )
        updated = updated.replace(
            "상위 타선와 상위 타선을 동시에 막아야 합니다.",
            "양 팀 상위 타선을 함께 제어해야 합니다.",
        )
        updated = re.sub(r"\s{2,}", " ", updated).strip()
        return updated

    for key in ("headline", "coach_note", "detailed_markdown"):
        cleaned[key] = _cleanup_text(cleaned.get(key))

    for metric in cleaned.get("key_metrics", []) or []:
        if not isinstance(metric, dict):
            continue
        metric["label"] = _cleanup_text(metric.get("label"))
        metric["value"] = _cleanup_text(metric.get("value"))
        metric_label = str(metric.get("label") or "")
        metric_value = str(metric.get("value") or "")
        if metric_value == "라인업 미확정으로 개별 타자 비교 보류":
            metric["label"] = "상위 타선 생산성"
        elif "상위 타선" in metric_label and (
            "폼" in metric_label
            or "점수" in metric_value
            or "OPS" in metric_label
            or "wRC+" in metric_label
        ):
            metric["label"] = "상위 타선 흐름"
            metric["value"] = "라인업 미확정으로 개별 타자 폼 비교 보류"
        elif (
            "OPS" in metric_value or "wRC+" in metric_value
        ) and "상위 타선" in metric_value:
            metric["label"] = "상위 타선 생산성"
            metric["value"] = "라인업 미확정으로 개별 타자 비교 보류"

    analysis = cleaned.get("analysis")
    if isinstance(analysis, dict):
        for key in ("summary", "verdict"):
            analysis[key] = _cleanup_text(analysis.get(key))
        for key in (
            "strengths",
            "weaknesses",
            "why_it_matters",
            "swing_factors",
            "watch_points",
            "uncertainty",
        ):
            items = analysis.get(key)
            if isinstance(items, list):
                analysis[key] = [_cleanup_text(item) for item in items]
        risks = analysis.get("risks")
        if isinstance(risks, list):
            for risk in risks:
                if isinstance(risk, dict):
                    risk["description"] = _cleanup_text(risk.get("description"))

    return cleaned


def _sanitize_scheduled_unconfirmed_lineup_entities(
    response_data: Dict[str, Any],
    *,
    evidence: GameEvidence,
    grounding_reasons: List[str],
    tool_results: Optional[Dict[str, Any]] = None,
    resolved_focus: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if (
        evidence.game_status_bucket != "SCHEDULED"
        or "missing_lineups" not in grounding_reasons
    ):
        return response_data

    cleaned = _cleanup_unconfirmed_lineup_player_placeholders(response_data)
    confirmed_names = _collect_confirmed_lineup_entity_names(evidence)
    protected_names = set(confirmed_names)
    protected_names.update(
        filter(
            None,
            (
                _normalize_name_token(evidence.home_team_name),
                _normalize_name_token(evidence.away_team_name),
            ),
        )
    )
    disallowed_entities = _find_disallowed_entities(
        response_data,
        protected_names,
        min_occurrences=1,
        allow_empty_allowed_names=True,
    )
    named_players = _find_unconfirmed_lineup_named_players(
        response_data, protected_names
    )
    disallowed_entities = [
        token for token in disallowed_entities if token not in {"변수지"}
    ]
    disallowed_entities = [
        token
        for token in disallowed_entities
        if token in named_players
        or KOREAN_PLAYER_NAME_BASE_RE.fullmatch(token)
        or ENGLISH_PLAYER_NAME_RE.fullmatch(token)
    ]
    disallowed_entities.extend(named_players)
    disallowed_entities = list(dict.fromkeys(disallowed_entities))
    if not disallowed_entities:
        return _repair_scheduled_unconfirmed_lineup_artifacts(
            cleaned,
            evidence=evidence,
            grounding_reasons=grounding_reasons,
            tool_results=tool_results,
            resolved_focus=resolved_focus,
        )

    sanitized = _sanitize_response_disallowed_entities(
        cleaned,
        disallowed_entities,
    )
    sanitized = _cleanup_unconfirmed_lineup_player_placeholders(sanitized)
    return _repair_scheduled_unconfirmed_lineup_artifacts(
        sanitized,
        evidence=evidence,
        grounding_reasons=grounding_reasons,
        tool_results=tool_results,
        resolved_focus=resolved_focus,
    )


def _response_contains_unconfirmed_lineup_artifacts(
    response_data: Dict[str, Any],
) -> bool:
    for candidate in _iter_response_text_candidates(response_data):
        text = str(candidate or "").strip()
        if not text:
            continue
        if any(marker in text for marker in UNCONFIRMED_LINEUP_ARTIFACT_MARKERS):
            return True
    return False


def _split_result_anchor_sentences(text: str) -> List[str]:
    parts = re.split(r"[\n.!?]+", str(text or ""))
    return [part.strip() for part in parts if part and part.strip()]


def _team_has_result_direction(
    response_data: Dict[str, Any],
    team_name: Optional[str],
) -> bool:
    team = _normalize_name_token(team_name)
    if not team:
        return False
    candidates = [
        str(response_data.get("headline") or ""),
        str(response_data.get("coach_note") or ""),
        str(response_data.get("detailed_markdown") or ""),
        str(response_data.get("analysis", {}).get("summary") or ""),
        str(response_data.get("analysis", {}).get("verdict") or ""),
    ]
    for candidate in candidates:
        for sentence in _split_result_anchor_sentences(candidate):
            if team in sentence and any(
                marker in sentence for marker in COMPLETED_RESULT_DIRECTION_MARKERS
            ):
                return True
    return False


def _ensure_completed_result_metric(
    response_data: Dict[str, Any],
    evidence: GameEvidence,
) -> Dict[str, Any]:
    metric = _completed_result_metric(evidence)
    if not metric:
        return response_data
    updated = deepcopy(response_data)
    existing_metrics = [
        item
        for item in (updated.get("key_metrics") or [])
        if isinstance(item, dict)
        and str(item.get("label") or "").strip() not in {"최종 스코어", "경기 결과"}
    ]
    updated["key_metrics"] = [metric, *existing_metrics][:5]
    return updated


def _completed_payload_needs_result_anchor(
    response_data: Dict[str, Any],
    evidence: GameEvidence,
) -> bool:
    if not _is_completed_review(evidence):
        return False
    scoreline = _completed_scoreline_text(evidence)
    if not scoreline:
        return False
    if _completed_game_is_draw(evidence):
        return "무승부" not in " ".join(_iter_response_text_candidates(response_data))

    winner_name = _completed_winner_name(evidence)
    loser_name = _completed_loser_name(evidence)
    if not winner_name:
        return False
    if not _team_has_result_direction(response_data, winner_name):
        return True
    if loser_name and _team_has_result_direction(response_data, loser_name):
        return True
    return False


def _enforce_completed_result_anchor(
    response_data: Dict[str, Any],
    *,
    evidence: GameEvidence,
    tool_results: Optional[Dict[str, Any]] = None,
    resolved_focus: Optional[List[str]] = None,
    grounding_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not _is_completed_review(evidence):
        return response_data

    updated = _ensure_completed_result_metric(response_data, evidence)
    if not tool_results or not _completed_payload_needs_result_anchor(
        updated, evidence
    ):
        return updated

    repaired = _build_deterministic_coach_response(
        evidence,
        tool_results,
        resolved_focus=resolved_focus,
        grounding_warnings=grounding_warnings,
    )
    if str(updated.get("sentiment") or "").strip():
        repaired["sentiment"] = updated["sentiment"]
    return repaired


def _repair_scheduled_unconfirmed_lineup_artifacts(
    response_data: Dict[str, Any],
    *,
    evidence: GameEvidence,
    grounding_reasons: List[str],
    tool_results: Optional[Dict[str, Any]] = None,
    resolved_focus: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if (
        evidence.game_status_bucket != "SCHEDULED"
        or "missing_lineups" not in grounding_reasons
    ):
        return response_data

    cleaned = _cleanup_unconfirmed_lineup_player_placeholders(response_data)
    if not _response_contains_unconfirmed_lineup_artifacts(cleaned) or not tool_results:
        return cleaned

    repaired = deepcopy(cleaned)
    repaired_analysis = _build_scheduled_team_level_deterministic_analysis(
        evidence, tool_results
    )
    repaired["analysis"] = repaired_analysis
    repaired["key_metrics"] = _build_scheduled_team_level_deterministic_metrics(
        evidence, tool_results
    )
    repaired["detailed_markdown"] = _build_deterministic_markdown(
        evidence,
        tool_results,
        resolved_focus=resolved_focus,
    )
    repaired["coach_note"] = _rebuild_scheduled_coach_note(repaired_analysis) or (
        repaired_analysis.get("verdict") or repaired_analysis.get("summary") or ""
    )
    headline = str(repaired.get("headline") or "")
    if any(marker in headline for marker in UNCONFIRMED_LINEUP_ARTIFACT_MARKERS):
        repaired["headline"] = _build_deterministic_headline(evidence, tool_results)

    repaired = _normalize_response_markdown_layout(repaired)
    repaired = _cleanup_response_language_quality(repaired)
    repaired = _polish_scheduled_partial_response(
        repaired,
        game_status_bucket=evidence.game_status_bucket,
        grounding_reasons=grounding_reasons,
    )
    return repaired


def _replace_disallowed_entity_names(text: str, disallowed_tokens: List[str]) -> str:
    updated = str(text or "")
    if not updated or not disallowed_tokens:
        return updated

    for token in sorted(disallowed_tokens, key=len, reverse=True):
        escaped = re.escape(token)
        suffix_pattern = "|".join(
            re.escape(suffix)
            for suffix in sorted(ENTITY_REPLACEMENT_SUFFIXES, key=len, reverse=True)
        )
        korean_pattern = re.compile(
            rf"(?<![가-힣A-Za-z]){escaped}(?P<suffix>{suffix_pattern})?"
        )

        def _replace_korean(match: re.Match[str]) -> str:
            suffix = match.group("suffix") or ""
            return ENTITY_SUFFIX_PLACEHOLDERS.get(suffix, "핵심 선수")

        updated = korean_pattern.sub(_replace_korean, updated)
        updated = re.sub(rf"\b{escaped}\b", "핵심 선수", updated)

    return updated


def _sanitize_response_disallowed_entities(
    response_data: Dict[str, Any],
    disallowed_tokens: List[str],
) -> Dict[str, Any]:
    sanitized = deepcopy(response_data)
    if not disallowed_tokens:
        return sanitized

    for key in ("headline", "detailed_markdown", "coach_note"):
        value = sanitized.get(key)
        if isinstance(value, str):
            sanitized[key] = _replace_disallowed_entity_names(value, disallowed_tokens)

    for metric in sanitized.get("key_metrics", []) or []:
        if not isinstance(metric, dict):
            continue
        for key in ("label", "value"):
            value = metric.get(key)
            if isinstance(value, str):
                metric[key] = _replace_disallowed_entity_names(value, disallowed_tokens)

    analysis = sanitized.get("analysis")
    if isinstance(analysis, dict):
        for key in ("summary", "verdict"):
            value = analysis.get(key)
            if isinstance(value, str):
                analysis[key] = _replace_disallowed_entity_names(
                    value, disallowed_tokens
                )
        for key in (
            "strengths",
            "weaknesses",
            "why_it_matters",
            "swing_factors",
            "watch_points",
            "uncertainty",
        ):
            items = analysis.get(key)
            if isinstance(items, list):
                analysis[key] = [
                    (
                        _replace_disallowed_entity_names(item, disallowed_tokens)
                        if isinstance(item, str)
                        else item
                    )
                    for item in items
                ]
        risks = analysis.get("risks")
        if isinstance(risks, list):
            for risk in risks:
                if isinstance(risk, dict) and isinstance(risk.get("description"), str):
                    risk["description"] = _replace_disallowed_entity_names(
                        risk["description"], disallowed_tokens
                    )

    return sanitized


def _replace_placeholder_literals(text: str) -> str:
    updated = PLACEHOLDER_LITERAL_PATTERN.sub("데이터 부족", str(text or ""))
    updated = updated.replace("数据 부족", "데이터 부족")
    return _cleanup_placeholder_phrase(updated)


def _cleanup_placeholder_phrase(text: str) -> str:
    updated = str(text or "")
    updated = updated.replace("데이터 부족라", "데이터 부족해")
    updated = updated.replace("데이터 부족로", "데이터 부족으로")
    updated = updated.replace("데이터 부족가", "데이터 부족이")
    updated = updated.replace("데이터 부족와", "데이터 부족과")
    updated = DUPLICATE_PLACEHOLDER_PHRASE_PATTERN.sub("결측 데이터", updated)
    return updated


def _cleanup_language_artifacts(text: str) -> str:
    updated = str(text or "")
    for source, replacement in LANGUAGE_ARTIFACT_REPLACEMENTS:
        updated = updated.replace(source, replacement)
    updated = NON_KOREAN_CJK_PATTERN.sub("", updated)
    updated = re.sub(r"\s{2,}", " ", updated)
    return updated.strip()


def _cleanup_unsupported_recent_form_text(text: str) -> str:
    updated = str(text or "")
    replaced = False
    for pattern, replacement in UNSUPPORTED_RECENT_FORM_REPLACEMENTS:
        updated, count = pattern.subn(replacement, updated)
        replaced = replaced or count > 0
    if not replaced:
        updated = UNSUPPORTED_RECENT_FORM_CLAUSE_PATTERN.sub(
            "최근 흐름 근거가 부족합니다.", updated
        )
    updated = updated.replace("근거가 부족하며 불펜", "근거가 부족하며 불펜")
    updated = re.sub(r"\s{2,}", " ", updated)
    return updated.strip()


def _replace_team_code_with_display_name(
    text: str,
    *,
    team_code: Optional[str],
    display_name: Optional[str],
) -> str:
    updated = str(text or "")
    code = str(team_code or "").strip()
    name = str(display_name or "").strip()
    if not updated or not code or not name:
        return updated

    escaped_code = re.escape(code)
    prefix_pattern = rf"\b{escaped_code}\b"
    protected_suffix = ""
    if name.upper().startswith(f"{code.upper()} "):
        protected_suffix = name[len(code) :].strip()
        if protected_suffix:
            prefix_pattern = rf"\b{escaped_code}\b(?!\s*{re.escape(protected_suffix)})"

    for suffix in sorted(set(TEAM_CODE_SUFFIXES), key=len, reverse=True):
        particle_pattern = rf"\b{escaped_code}{re.escape(suffix)}"
        if protected_suffix:
            particle_pattern += rf"(?!\s*{re.escape(protected_suffix)})"
        updated = re.sub(particle_pattern, f"{name}{suffix}", updated)
    return re.sub(prefix_pattern, name, updated)


def _replace_team_alias_with_display_name(
    text: str,
    *,
    aliases: Tuple[str, ...],
    display_name: Optional[str],
) -> str:
    updated = str(text or "")
    name = str(display_name or "").strip()
    if not updated or not name or not aliases:
        return updated

    suffix_pattern = "|".join(
        re.escape(suffix)
        for suffix in sorted(set(TEAM_CODE_SUFFIXES), key=len, reverse=True)
    )
    for alias in aliases:
        token = str(alias or "").strip()
        if not token or token == name:
            continue
        escaped_token = re.escape(token)
        protected_suffix = ""
        if name.startswith(f"{token} "):
            protected_suffix = name[len(token) :].strip()

        particle_pattern = (
            rf"(?<![가-힣A-Za-z]){escaped_token}(?P<suffix>{suffix_pattern})?(?!\s*{re.escape(protected_suffix)})"
            if protected_suffix
            else rf"(?<![가-힣A-Za-z]){escaped_token}(?P<suffix>{suffix_pattern})?"
        )

        def _replace_alias(match: re.Match[str]) -> str:
            suffix = match.group("suffix") or ""
            return f"{name}{suffix}"

        updated = re.sub(particle_pattern, _replace_alias, updated)
    return updated


def _unwrap_bracketed_team_display_name(
    text: str,
    *,
    team_code: Optional[str],
    display_name: Optional[str],
    aliases: Tuple[str, ...],
) -> str:
    updated = str(text or "")
    name = str(display_name or "").strip()
    if not updated or not name:
        return updated

    bracket_tokens: List[str] = [name]
    code = str(team_code or "").strip()
    if code:
        bracket_tokens.append(code)
    for alias in aliases:
        token = str(alias or "").strip()
        if token:
            bracket_tokens.append(token)

    for token in dict.fromkeys(bracket_tokens):
        updated = updated.replace(f"[{token}]", name)
    return updated


def _normalize_response_team_display(
    response_data: Dict[str, Any],
    *,
    evidence: GameEvidence,
) -> Dict[str, Any]:
    normalized = deepcopy(response_data)

    def _normalize_text(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        updated = value
        updated = _unwrap_bracketed_team_display_name(
            updated,
            team_code=evidence.home_team_code,
            display_name=evidence.home_team_name,
            aliases=TEAM_DISPLAY_ALIASES_BY_CODE.get(evidence.home_team_code, ()),
        )
        updated = _replace_team_code_with_display_name(
            updated,
            team_code=evidence.home_team_code,
            display_name=evidence.home_team_name,
        )
        updated = _replace_team_alias_with_display_name(
            updated,
            aliases=TEAM_DISPLAY_ALIASES_BY_CODE.get(evidence.home_team_code, ()),
            display_name=evidence.home_team_name,
        )
        updated = _unwrap_bracketed_team_display_name(
            updated,
            team_code=evidence.away_team_code,
            display_name=evidence.away_team_name,
            aliases=TEAM_DISPLAY_ALIASES_BY_CODE.get(evidence.away_team_code, ()),
        )
        updated = _replace_team_code_with_display_name(
            updated,
            team_code=evidence.away_team_code,
            display_name=evidence.away_team_name,
        )
        updated = _replace_team_alias_with_display_name(
            updated,
            aliases=TEAM_DISPLAY_ALIASES_BY_CODE.get(evidence.away_team_code, ()),
            display_name=evidence.away_team_name,
        )
        return updated

    for key in ("headline", "detailed_markdown", "coach_note"):
        normalized[key] = _normalize_text(normalized.get(key))

    for metric in normalized.get("key_metrics", []) or []:
        if not isinstance(metric, dict):
            continue
        metric["label"] = _normalize_text(metric.get("label"))
        metric["value"] = _normalize_text(metric.get("value"))

    analysis = normalized.get("analysis")
    if isinstance(analysis, dict):
        for key in ("summary", "verdict"):
            analysis[key] = _normalize_text(analysis.get(key))
        for key in (
            "strengths",
            "weaknesses",
            "why_it_matters",
            "swing_factors",
            "watch_points",
            "uncertainty",
        ):
            items = analysis.get(key)
            if isinstance(items, list):
                analysis[key] = [_normalize_text(item) for item in items]
        risks = analysis.get("risks")
        if isinstance(risks, list):
            for risk in risks:
                if isinstance(risk, dict):
                    risk["description"] = _normalize_text(risk.get("description"))

    return normalized


def _cleanup_response_language_quality(
    response_data: Dict[str, Any],
) -> Dict[str, Any]:
    cleaned = deepcopy(response_data)

    def _cleanup_text(value: Any, *, markdown: bool = False) -> Any:
        if not isinstance(value, str):
            return value
        updated = value
        for source, replacement in RESPONSE_LANGUAGE_QUALITY_REPLACEMENTS:
            updated = updated.replace(source, replacement)
        for pattern, replacement in RESPONSE_LANGUAGE_QUALITY_REGEX_REPLACEMENTS:
            updated = pattern.sub(replacement, updated)
        updated = re.sub(r"[^\S\n]{2,}", " ", updated)
        updated = updated.strip()
        if markdown:
            updated = _normalize_detailed_markdown_layout(updated)
        return updated

    for key in ("headline", "coach_note"):
        cleaned[key] = _cleanup_text(cleaned.get(key))

    cleaned["detailed_markdown"] = _cleanup_text(
        cleaned.get("detailed_markdown"), markdown=True
    )

    for metric in cleaned.get("key_metrics", []) or []:
        if not isinstance(metric, dict):
            continue
        metric["label"] = _cleanup_text(metric.get("label"))
        metric["value"] = _cleanup_text(metric.get("value"))

    analysis = cleaned.get("analysis")
    if isinstance(analysis, dict):
        for key in ("summary", "verdict"):
            analysis[key] = _cleanup_text(analysis.get(key))
        for key in (
            "strengths",
            "weaknesses",
            "why_it_matters",
            "swing_factors",
            "watch_points",
            "uncertainty",
        ):
            items = analysis.get(key)
            if isinstance(items, list):
                analysis[key] = [_cleanup_text(item) for item in items]
        risks = analysis.get("risks")
        if isinstance(risks, list):
            for risk in risks:
                if isinstance(risk, dict):
                    risk["description"] = _cleanup_text(risk.get("description"))

    return cleaned


def _normalized_text_key(value: Optional[str]) -> str:
    text = _clean_summary_text(value)
    if not text:
        return ""
    text = re.sub(r"[.,!?;:()\[\]\"'`“”‘’·]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _texts_are_redundant(left: Optional[str], right: Optional[str]) -> bool:
    left_key = _normalized_text_key(left)
    right_key = _normalized_text_key(right)
    if not left_key or not right_key:
        return False
    return left_key == right_key or left_key in right_key or right_key in left_key


def _prefixed_sentence(prefix: str, candidate: Optional[str]) -> Optional[str]:
    text = _clean_summary_text(candidate)
    if not text:
        return None
    if text.startswith(prefix):
        return _ensure_sentence(text)
    return _ensure_sentence(f"{prefix} {text}")


def _scheduled_verdict_candidate(analysis: Dict[str, Any]) -> Optional[str]:
    swing_factors = analysis.get("swing_factors") or []
    if isinstance(swing_factors, list):
        for item in swing_factors:
            sentence = _prefixed_sentence("핵심 변수는", str(item or "").strip())
            if sentence:
                return sentence

    why_it_matters = analysis.get("why_it_matters") or []
    if isinstance(why_it_matters, list):
        for item in why_it_matters:
            sentence = _ensure_sentence(str(item or "").strip())
            if sentence:
                return sentence

    watch_points = analysis.get("watch_points") or []
    if isinstance(watch_points, list):
        for item in watch_points:
            sentence = _prefixed_sentence("관전 포인트는", str(item or "").strip())
            if sentence:
                return sentence

    uncertainty = analysis.get("uncertainty") or []
    if isinstance(uncertainty, list):
        for item in uncertainty:
            sentence = _prefixed_sentence("확인 전까지는", str(item or "").strip())
            if sentence:
                return sentence

    return None


def _rebuild_scheduled_coach_note(analysis: Dict[str, Any]) -> str:
    note_parts: List[str] = []

    watch_points = analysis.get("watch_points") or []
    if isinstance(watch_points, list) and watch_points:
        _append_distinct_note_part(
            note_parts, _prefixed_sentence("관전 포인트는", str(watch_points[0]))
        )

    uncertainty = analysis.get("uncertainty") or []
    if isinstance(uncertainty, list) and uncertainty:
        _append_distinct_note_part(
            note_parts, _prefixed_sentence("확인 포인트는", str(uncertainty[0]))
        )

    swing_factors = analysis.get("swing_factors") or []
    if isinstance(swing_factors, list) and swing_factors:
        _append_distinct_note_part(note_parts, _ensure_sentence(str(swing_factors[0])))

    verdict = _clean_summary_text(analysis.get("verdict"))
    if verdict:
        _append_distinct_note_part(note_parts, verdict)

    return _build_compact_coach_note(note_parts, max_length=180)


def _polish_scheduled_partial_response(
    response_data: Dict[str, Any],
    *,
    game_status_bucket: str,
    grounding_reasons: List[str],
) -> Dict[str, Any]:
    if not _is_scheduled_partial_context(game_status_bucket, grounding_reasons):
        return response_data

    polished = deepcopy(response_data)

    def _polish_text(value: Any, *, markdown: bool = False) -> Any:
        if not isinstance(value, str):
            return value
        updated = value
        for source, replacement in SCHEDULED_USER_FACING_REPLACEMENTS:
            updated = updated.replace(source, replacement)
        updated = _cleanup_placeholder_phrase(updated)
        updated = updated.replace("핵심 자원와", "핵심 자원과")
        bullpen_copy_markers = (
            "데이터 부족",
            "데이터 결측",
            "공식 데이터",
            "예측하기 어렵",
            "판단하기 어렵",
            "미확정",
        )
        if "불펜 비중" in updated and any(
            marker in updated for marker in bullpen_copy_markers
        ):
            updated = updated.replace("불펜 비중", "불펜 운용")
        updated = re.sub(r"[^\S\n]{2,}", " ", updated).strip()
        if markdown:
            updated = _normalize_detailed_markdown_layout(updated)
        return updated

    for key in ("headline", "coach_note"):
        polished[key] = _polish_text(polished.get(key))

    polished["detailed_markdown"] = _polish_text(
        polished.get("detailed_markdown"),
        markdown=True,
    )

    for metric in polished.get("key_metrics", []) or []:
        if not isinstance(metric, dict):
            continue
        metric["label"] = _polish_text(metric.get("label"))
        metric["value"] = _polish_text(metric.get("value"))
        metric_label = _clean_summary_text(str(metric.get("label") or ""))
        metric_value = _clean_summary_text(str(metric.get("value") or ""))
        if metric_label == "불펜 비중" and (
            not metric_value
            or "데이터 부족" in metric_value
            or "비교가 어렵다" in metric_value
            or "정보 부족" in metric_value
            or "미확정" in metric_value
            or "결측" in metric_value
        ):
            metric["label"] = "불펜 운용"

    analysis = polished.get("analysis")
    if isinstance(analysis, dict):
        for key in ("summary", "verdict"):
            analysis[key] = _polish_text(analysis.get(key))
        for key in (
            "strengths",
            "weaknesses",
            "why_it_matters",
            "swing_factors",
            "watch_points",
            "uncertainty",
        ):
            items = analysis.get(key)
            if isinstance(items, list):
                analysis[key] = [_polish_text(item) for item in items]
        risks = analysis.get("risks")
        if isinstance(risks, list):
            for risk in risks:
                if isinstance(risk, dict):
                    risk["description"] = _polish_text(risk.get("description"))

        if _texts_are_redundant(
            str(analysis.get("summary") or ""),
            str(analysis.get("verdict") or ""),
        ):
            candidate_verdict = _scheduled_verdict_candidate(analysis)
            if candidate_verdict and not _texts_are_redundant(
                str(analysis.get("summary") or ""), candidate_verdict
            ):
                analysis["verdict"] = candidate_verdict

        coach_note = str(polished.get("coach_note") or "")
        if (
            not coach_note
            or _texts_are_redundant(coach_note, str(analysis.get("summary") or ""))
            or _texts_are_redundant(coach_note, str(analysis.get("verdict") or ""))
            or len(_normalized_text_key(coach_note)) < 24
        ):
            rebuilt_note = _rebuild_scheduled_coach_note(analysis)
            if rebuilt_note:
                polished["coach_note"] = rebuilt_note

    return polished


def _normalize_detailed_markdown_layout(text: str) -> str:
    updated = str(text or "").replace("\r\n", "\n")
    if not updated:
        return updated
    for header in FOCUS_SECTION_HEADERS.values():
        updated = re.sub(
            rf"(?m)^({re.escape(header)})\s*:\s*(.+?)\s*$",
            r"\1\n- \2",
            updated,
        )
    updated = re.sub(r"(?<!^)(?<!\n)\s*(##\s+)", r"\n\n\1", updated)
    updated = re.sub(r"(?m)^\-\s*$\n?", "", updated)
    updated = re.sub(r"\n{3,}", "\n\n", updated)
    lines = [line.rstrip() for line in updated.split("\n")]
    return "\n".join(lines).strip()


def _normalize_response_markdown_layout(
    response_data: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = deepcopy(response_data)
    normalized["detailed_markdown"] = _normalize_detailed_markdown_layout(
        normalized.get("detailed_markdown") or ""
    )
    return normalized


def _soften_scheduled_partial_tone(
    response_data: Dict[str, Any],
    *,
    game_status_bucket: str,
    grounding_reasons: List[str],
) -> Dict[str, Any]:
    if not _is_scheduled_partial_context(game_status_bucket, grounding_reasons):
        return response_data

    softened = deepcopy(response_data)

    def _soften_text(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        updated = value
        for source, replacement in SCHEDULED_PARTIAL_TONE_REPLACEMENTS:
            updated = updated.replace(source, replacement)
        for source, replacement in SCHEDULED_REVIEW_LANGUAGE_REPLACEMENTS:
            updated = updated.replace(source, replacement)
        return updated

    for key in ("headline", "detailed_markdown", "coach_note"):
        softened[key] = _soften_text(softened.get(key))

    analysis = softened.get("analysis")
    if isinstance(analysis, dict):
        for key in ("summary", "verdict"):
            analysis[key] = _soften_text(analysis.get(key))
        for key in (
            "strengths",
            "weaknesses",
            "why_it_matters",
            "swing_factors",
            "watch_points",
            "uncertainty",
        ):
            items = analysis.get(key)
            if isinstance(items, list):
                analysis[key] = [_soften_text(item) for item in items]
        risks = analysis.get("risks")
        if isinstance(risks, list):
            for risk in risks:
                if isinstance(risk, dict):
                    risk["description"] = _soften_text(risk.get("description"))

    for metric in softened.get("key_metrics", []) or []:
        if isinstance(metric, dict):
            metric["label"] = _soften_text(metric.get("label"))
            metric["value"] = _soften_text(metric.get("value"))

    return softened


def _sanitize_response_placeholders(
    response_data: Dict[str, Any],
    *,
    used_evidence: List[str],
) -> Dict[str, Any]:
    sanitized = deepcopy(response_data)
    recent_form_available = any(
        evidence_key in used_evidence
        for evidence_key in ("team_recent_form", "opponent_recent_form")
    )

    def _sanitize_text(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        updated = _replace_placeholder_literals(value)
        if not recent_form_available:
            updated = TEAM_ZERO_RECORD_PLACEHOLDER_PATTERN.sub(
                r"\1 데이터 부족", updated
            )
            updated = ZERO_RECORD_PLACEHOLDER_PATTERN.sub("데이터 부족", updated)
            updated = _cleanup_placeholder_phrase(updated)
            updated = _cleanup_unsupported_recent_form_text(updated)
        return _cleanup_language_artifacts(updated)

    for key in ("headline", "detailed_markdown", "coach_note"):
        sanitized[key] = _sanitize_text(sanitized.get(key))

    for metric in sanitized.get("key_metrics", []) or []:
        if not isinstance(metric, dict):
            continue
        metric["label"] = _sanitize_text(metric.get("label"))
        metric["value"] = _sanitize_text(metric.get("value"))

    analysis = sanitized.get("analysis")
    if isinstance(analysis, dict):
        for key in ("summary", "verdict"):
            analysis[key] = _sanitize_text(analysis.get(key))
        for key in (
            "strengths",
            "weaknesses",
            "why_it_matters",
            "swing_factors",
            "watch_points",
            "uncertainty",
        ):
            items = analysis.get(key)
            if isinstance(items, list):
                analysis[key] = [_sanitize_text(item) for item in items]
        risks = analysis.get("risks")
        if isinstance(risks, list):
            for risk in risks:
                if isinstance(risk, dict):
                    risk["description"] = _sanitize_text(risk.get("description"))

    return sanitized


def _text_contains_unsupported_numeric_claim(
    value: Any, unsupported_tokens: Set[str]
) -> bool:
    if not isinstance(value, str) or not value.strip() or not unsupported_tokens:
        return False
    return bool(collect_numeric_tokens(value) & unsupported_tokens)


def _strip_unsupported_numeric_lines(text: str, unsupported_tokens: Set[str]) -> str:
    if not text.strip():
        return text

    if "\n" in text:
        kept_lines: List[str] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("## "):
                kept_lines.append(raw_line)
                continue
            if _text_contains_unsupported_numeric_claim(raw_line, unsupported_tokens):
                continue
            kept_lines.append(raw_line)
        return _normalize_detailed_markdown_layout("\n".join(kept_lines))

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept_sentences = [
        sentence
        for sentence in sentences
        if sentence
        and not _text_contains_unsupported_numeric_claim(sentence, unsupported_tokens)
    ]
    if kept_sentences:
        return " ".join(kept_sentences).strip()
    return ""


def _repair_analysis_fields_after_numeric_sanitization(
    response_data: Dict[str, Any],
) -> Dict[str, Any]:
    repaired = deepcopy(response_data)
    analysis = repaired.get("analysis")
    if not isinstance(analysis, dict):
        return repaired

    candidates = _iter_response_text_candidates(repaired)
    fallback_text = next((item for item in candidates if item.strip()), "")

    summary = str(analysis.get("summary") or "").strip()
    verdict = str(analysis.get("verdict") or "").strip()
    coach_note = str(repaired.get("coach_note") or "").strip()

    if not summary and fallback_text:
        analysis["summary"] = fallback_text
        summary = fallback_text
    if not verdict and (summary or fallback_text):
        analysis["verdict"] = summary or fallback_text
        verdict = analysis["verdict"]
    if not coach_note and (verdict or summary or fallback_text):
        repaired["coach_note"] = verdict or summary or fallback_text

    return repaired


def _sanitize_response_unsupported_numeric_claims(
    response_data: Dict[str, Any],
    *,
    unsupported_tokens: List[str],
) -> Dict[str, Any]:
    sanitized = deepcopy(response_data)
    token_set = {token for token in unsupported_tokens if token}
    if not token_set:
        return sanitized

    for key in ("headline", "coach_note"):
        value = sanitized.get(key)
        if isinstance(value, str):
            sanitized[key] = _strip_unsupported_numeric_lines(value, token_set)

    detailed_markdown = sanitized.get("detailed_markdown")
    if isinstance(detailed_markdown, str):
        sanitized["detailed_markdown"] = _strip_unsupported_numeric_lines(
            detailed_markdown, token_set
        )

    key_metrics = sanitized.get("key_metrics") or []
    if isinstance(key_metrics, list):
        sanitized["key_metrics"] = [
            metric
            for metric in key_metrics
            if not (
                isinstance(metric, dict)
                and (
                    _text_contains_unsupported_numeric_claim(
                        metric.get("label"), token_set
                    )
                    or _text_contains_unsupported_numeric_claim(
                        metric.get("value"), token_set
                    )
                )
            )
        ]

    analysis = sanitized.get("analysis")
    if isinstance(analysis, dict):
        for key in ("summary", "verdict"):
            value = analysis.get(key)
            if isinstance(value, str):
                analysis[key] = _strip_unsupported_numeric_lines(value, token_set)
        for key in (
            "strengths",
            "weaknesses",
            "why_it_matters",
            "swing_factors",
            "watch_points",
            "uncertainty",
        ):
            items = analysis.get(key)
            if isinstance(items, list):
                analysis[key] = [
                    item
                    for item in items
                    if not _text_contains_unsupported_numeric_claim(item, token_set)
                ]
        risks = analysis.get("risks")
        if isinstance(risks, list):
            analysis["risks"] = [
                risk
                for risk in risks
                if not (
                    isinstance(risk, dict)
                    and _text_contains_unsupported_numeric_claim(
                        risk.get("description"), token_set
                    )
                )
            ]

    sanitized = _repair_analysis_fields_after_numeric_sanitization(sanitized)
    sanitized["detailed_markdown"] = _normalize_detailed_markdown_layout(
        sanitized.get("detailed_markdown") or ""
    )
    return sanitized


def _response_mentions_disallowed_entities(
    response_data: Dict[str, Any],
    allowed_names: Set[str],
) -> bool:
    return bool(_find_disallowed_entities(response_data, allowed_names))


router = APIRouter(prefix="/coach", tags=["coach"])


# ============================================================
# Fast Path Helper Functions
# ============================================================


def _build_coach_query(
    team_name: str,
    focus: List[str],
    opponent_name: Optional[str] = None,
    league_context: Optional[Dict[str, Any]] = None,
    game_status_bucket: Optional[str] = None,
) -> str:
    """focus 영역에 따라 Coach 질문을 구성합니다."""
    focus_text = ", ".join(focus) if focus else "종합적인 전력"
    normalized_status_bucket = _normalize_game_status_bucket(game_status_bucket)
    review_mode = normalized_status_bucket == "COMPLETED"
    scheduled_mode = normalized_status_bucket == "SCHEDULED"

    if opponent_name:
        if review_mode:
            query = (
                f"{team_name}와 {opponent_name}의 {focus_text}를 바탕으로 "
                "경기 종료 기준 리뷰를 해줘."
            )
        else:
            query = f"{team_name}와 {opponent_name}의 {focus_text}에 대해 냉철하고 다각적인 비교 분석을 수행해줘."
    else:
        if review_mode:
            query = f"{team_name}의 {focus_text}를 바탕으로 경기 종료 기준 리뷰를 해줘."
        else:
            query = (
                f"{team_name}의 {focus_text}에 대해 냉철하고 다각적인 분석을 수행해줘."
            )

    # 리그 컨텍스트 반영
    if league_context:
        season = league_context.get("season_year") or league_context.get("season")
        league_type = league_context.get("league_type")
        if league_type == "POST":
            round_name = league_context.get("round", "포스트시즌")
            game_no = league_context.get("game_no")
            if game_no is not None:
                query += f" 특히 {season}년 {round_name} {game_no}차전임을 감안하여 분석해줘."
            else:
                query += f" 특히 {season}년 {round_name} 경기 맥락을 반영해줘."
        elif league_type == "REGULAR":
            home_ctx = league_context.get("home", {})
            away_ctx = league_context.get("away", {})
            if home_ctx and away_ctx:
                home_rank = home_ctx.get("rank")
                away_rank = away_ctx.get("rank")
                if home_rank is not None and away_rank is not None:
                    rank_diff = abs(int(home_rank) - int(away_rank))
                    if rank_diff <= 2:
                        if review_mode:
                            query += " 두 팀의 순위 경쟁 구도가 실제 결과에 어떤 영향을 줬는지도 짚어줘."
                        else:
                            query += " 두 팀의 순위 경쟁이 치열한 상황이야."

    if "batting" in focus or not focus:
        if opponent_name:
            if review_mode:
                query += " 양 팀의 타격 생산성(OPS, wRC+)이 실제 득점 차이로 어떻게 이어졌는지 복기해줘."
            else:
                query += " 양 팀의 타격 생산성(OPS, wRC+)과 주요 타자들의 최근 클러치 능력을 진단해줘."
        else:
            if review_mode:
                query += " 타격 생산성(OPS, wRC+)이 실제 득점 장면과 어떻게 연결됐는지 복기해줘."
            else:
                query += " 타격 생산성(OPS, wRC+)과 주요 타자들의 최근 클러치 능력을 진단해줘."

    if "bullpen" in focus:
        if review_mode:
            query += " 불펜진의 접전 후반 대응과 소모가 실제 결과에 어떤 영향을 줬는지 분석해줘."
        elif scheduled_mode:
            query += " 불펜진의 경기 후반 접전 대응력과 최근 소모 흐름을 분석해줘."
        else:
            query += " 불펜진의 경기 후반 접전 대응력과 최근 소모 흐름을 분석해줘."

    if "recent_form" in focus or not focus:
        if review_mode:
            query += " 최근 10경기 승패 트렌드와 득실점 마진이 실제 경기 결과와 얼마나 맞물렸는지 진단해줘."
        else:
            query += " 최근 10경기 승패 트렌드와 득실점 마진을 보고 팀의 상승세/하락세를 진단해줘."

    if "starter" in focus:
        if review_mode:
            query += " 선발 로테이션의 이닝 소화력과 QS 비율, 구속 변화가 실제 경기 흐름에 어떤 영향을 줬는지 분석해줘."
        else:
            query += " 선발 로테이션의 이닝 소화력과 QS 비율, 구속 변화를 분석해줘."

    if "matchup" in focus:
        if review_mode:
            query += " 상대 전적과 시즌 매치업 경향이 실제 결과와 얼마나 맞물렸는지 비교 분석해줘."
        else:
            query += " 주요 라이벌 팀들과의 상대 전적(승률, 득실 등)을 비교 분석해줘."

    return query


async def _execute_coach_tools_parallel(
    pool: ConnectionPool,
    home_team_id: str,
    year: int,
    focus: List[str],
    away_team_id: Optional[str] = None,
    as_of_game_date: Optional[str] = None,
    exclude_game_id: Optional[str] = None,
    matchup_season_id: Optional[int] = None,
    game_id: Optional[str] = None,
    include_clutch: bool = False,
) -> Dict[str, Any]:
    """
    Coach에 필요한 도구들을 병렬로 실행합니다.
    홈팀과 원정팀 데이터를 모두 조회합니다.
    """
    loop = asyncio.get_running_loop()

    def get_team_data(team_code: str):
        """특정 팀의 모든 데이터 조회"""
        results = {}
        with pool.connection() as conn:
            db_query = DatabaseQueryTool(conn)
            results["summary"] = db_query.get_team_summary(team_code, year)
            results["advanced"] = db_query.get_team_advanced_metrics(team_code, year)
            results["player_form_signals"] = db_query.get_team_player_form_signals(
                team_code,
                year,
                as_of_game_date=as_of_game_date,
                exclude_game_id=exclude_game_id,
            )
            if "recent_form" in focus or not focus:
                results["recent"] = db_query.get_team_recent_form(
                    team_code,
                    year,
                    as_of_game_date=as_of_game_date,
                    exclude_game_id=exclude_game_id,
                )
            if "matchup" in focus and away_team_id:
                # 상대 전적은 홈팀 기준 한번만 조회해도 됨
                pass
            if getattr(db_query, "mapping_dependency_degraded", False):
                results["_dependency_degraded"] = True
                results["_dependency_degraded_reason"] = (
                    getattr(db_query, "mapping_dependency_reason", None) or "defaults"
                )
        return results

    def get_clutch_moments_sync(target_game_id: str):
        with pool.connection() as conn:
            db_query = DatabaseQueryTool(conn)
            return db_query.get_clutch_moments(target_game_id, limit=3)

    def get_matchup_stats_sync(team1: str, team2: str):
        with pool.connection() as conn:
            from app.tools.game_query import GameQueryTool

            game_query = GameQueryTool(conn)
            result = game_query.get_head_to_head(
                team1,
                team2,
                year,
                as_of_game_date=as_of_game_date,
                exclude_game_id=exclude_game_id,
                season_id=matchup_season_id,
            )
            if isinstance(result, dict) and getattr(
                game_query, "mapping_dependency_degraded", False
            ):
                result["_dependency_degraded"] = True
                result["_dependency_degraded_reason"] = (
                    getattr(game_query, "mapping_dependency_reason", None) or "defaults"
                )
            return result

    # 병렬 실행 태스크 준비
    tasks = []

    # 1. 홈팀 데이터
    tasks.append(loop.run_in_executor(None, get_team_data, home_team_id))

    # 2. 원정팀 데이터 (있을 경우)
    if away_team_id:
        tasks.append(loop.run_in_executor(None, get_team_data, away_team_id))

    # 3. 상대 전적 (Matchup focus일 경우)
    if "matchup" in focus and away_team_id:
        tasks.append(
            loop.run_in_executor(
                None, get_matchup_stats_sync, home_team_id, away_team_id
            )
        )
    if include_clutch and game_id:
        tasks.append(loop.run_in_executor(None, get_clutch_moments_sync, game_id))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    tool_results = {
        "home": {},
        "away": {},
        "matchup": {},
        "clutch_moments": {},
        "error": None,
    }

    # 홈팀 결과 처리
    if isinstance(results[0], Exception):
        tool_results["error"] = COACH_TOOL_FETCH_FAILED_CODE
        tool_results["home"] = {"error": COACH_TOOL_FETCH_FAILED_CODE}
    else:
        tool_results["home"] = results[0]

    # 원정팀 결과 처리
    if away_team_id:
        if isinstance(results[1], Exception):
            tool_results["away"] = {"error": COACH_TOOL_FETCH_FAILED_CODE}
        else:
            tool_results["away"] = results[1]

        # 상대 전적 처리
        if "matchup" in focus:
            if len(tasks) > 2 and isinstance(results[2], Exception):
                tool_results["matchup"] = {"error": COACH_TOOL_FETCH_FAILED_CODE}
            elif len(tasks) > 2:
                tool_results["matchup"] = results[2]
        clutch_index = 3 if "matchup" in focus else 2
        if include_clutch and game_id:
            if len(tasks) > clutch_index and isinstance(
                results[clutch_index], Exception
            ):
                tool_results["clutch_moments"] = {"error": COACH_TOOL_FETCH_FAILED_CODE}
            elif len(tasks) > clutch_index:
                tool_results["clutch_moments"] = results[clutch_index]
    elif include_clutch and game_id:
        clutch_index = 1
        if len(tasks) > clutch_index and isinstance(results[clutch_index], Exception):
            tool_results["clutch_moments"] = {"error": COACH_TOOL_FETCH_FAILED_CODE}
        elif len(tasks) > clutch_index:
            tool_results["clutch_moments"] = results[clutch_index]

    # 레거시 구조 호환성 유지 (단일 팀 분석 요청 시)
    if not away_team_id:
        tool_results["team_summary"] = tool_results["home"].get("summary", {})
        tool_results["advanced_metrics"] = tool_results["home"].get("advanced", {})
        tool_results["recent_form"] = tool_results["home"].get("recent", {})
        tool_results["player_form_signals"] = tool_results["home"].get(
            "player_form_signals", {}
        )

    return tool_results


def _safe_float(value: Any, default: float = 0.0) -> float:
    """None-safe float conversion for formatting."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _remove_duplicate_json_start(text: str) -> str:
    """
    LLM 스트리밍 중 발생하는 JSON 시작 부분 중복을 제거합니다.

    일부 LLM은 스트리밍 중 "content restart" 현상으로 동일한 필드를
    두 번 출력하는 경우가 있습니다. 이 함수는 중복을 감지하고 제거합니다.

    예시:
        입력: '{"headline": "A",\\n"headline": "A", "sentiment": ...'
        출력: '{"headline": "A", "sentiment": ...'

    Args:
        text: LLM의 원시 출력 텍스트

    Returns:
        중복이 제거된 텍스트
    """
    import re

    if not text or "{" not in text:
        return text

    # headline 필드 패턴 찾기
    headline_pattern = r'"headline"\s*:\s*"[^"]*"'
    matches = list(re.finditer(headline_pattern, text))

    if len(matches) < 2:
        return text

    # 두 개 이상의 headline 필드가 있으면 중복
    first_match = matches[0]
    second_match = matches[1]

    # 두 headline 값이 동일한지 확인
    first_value = text[first_match.start() : first_match.end()]
    second_value = text[second_match.start() : second_match.end()]

    if first_value == second_value:
        # 중복 발견 - 두 번째 headline 이후 내용만 유지
        logger.warning(
            "[Coach] Duplicate JSON start detected, removing first occurrence"
        )

        # { + 두 번째 headline부터의 내용
        brace_pos = text.index("{")
        clean_text = text[brace_pos : brace_pos + 1]  # '{'

        # 두 번째 headline 이후 내용
        after_second = text[second_match.start() :]
        clean_text += after_second

        return clean_text

    return text


def _build_coach_retry_prompt(
    *,
    question: str,
    context: str,
    focus_section_requirements: str,
    previous_response: str,
    failure_reasons: List[str],
) -> str:
    base_prompt = COACH_PROMPT_V2.format(
        question=question,
        context=context,
        focus_section_requirements=focus_section_requirements,
    )
    reason_text = (
        ", ".join(failure_reasons[:4]) if failure_reasons else "근거 검증 실패"
    )
    previous_excerpt = previous_response.strip()[:900]
    return (
        f"{base_prompt}\n\n"
        "## 재작성 지시\n"
        f"- 직전 응답은 다음 사유로 폐기되었습니다: {reason_text}\n"
        "- FACT SHEET에 없는 선수명, 숫자, 시리즈 맥락은 절대 넣지 마세요.\n"
        "- `analysis.verdict`와 `analysis.why_it_matters`를 반드시 채우고, 숫자와 경기 운영 의미를 직접 연결하세요.\n"
        "- `analysis.verdict`, `analysis.swing_factors`, `coach_note`에 같은 문장이나 상투 표현을 반복하지 마세요.\n"
        "- 단순 지표 나열 대신 어떤 지표가 왜 승부처인지 판단 문장으로 정리하세요.\n"
        "- 아래 이전 응답의 잘못된 표현은 반복하지 말고, 필요한 사실만 남겨 새 JSON을 작성하세요.\n"
        f"### 직전 응답\n{previous_excerpt}"
    )


def _format_team_stats(team_data: Dict[str, Any], team_role: str = "Home") -> str:
    """단일 팀 통계 포맷팅 헬퍼"""
    parts = []

    summary = team_data.get("summary", {})
    advanced = team_data.get("advanced", {})
    team_name = summary.get("team_name", "Unknown")

    parts.append(f"### [{team_role}] {team_name}")

    # 핵심 지표
    if advanced.get("metrics"):
        batting = advanced["metrics"].get("batting", {})
        pitching = advanced["metrics"].get("pitching", {})
        rankings = advanced.get("rankings", {})

        parts.append("| 지표 | 수치 | 순위 |")
        parts.append("|------|------|------|")
        if batting.get("ops"):
            parts.append(
                f"| OPS | {_safe_float(batting['ops']):.3f} | {rankings.get('batting_ops', '-')}|"
            )
        if pitching.get("avg_era"):
            parts.append(
                f"| ERA | {_safe_float(pitching['avg_era']):.2f} | {pitching.get('era_rank', '-')}|"
            )
        parts.append("")

    # 불펜
    fatigue = advanced.get("fatigue_index", {})
    if fatigue:
        bullpen_share = fatigue.get("bullpen_share")
        bullpen_load_rank = fatigue.get("bullpen_load_rank")
        parts.append(
            f"- **불펜 비중**: {bullpen_share if bullpen_share not in (None, '') else '-'}"
        )
        parts.append(
            f"- **피로도 순위**: {bullpen_load_rank if bullpen_load_rank not in (None, '') else '-'}"
        )
        parts.append("")

    # 주요 선수 (간략화)
    top_batters = summary.get("top_batters", [])[:3]
    if top_batters:
        parts.append("**주요 타자**:")
        for b in top_batters:
            parts.append(
                f"- {b['player_name']}: OPS {_safe_float(b.get('ops')):.3f}, {b.get('home_runs')}HR"
            )

    top_pitchers = summary.get("top_pitchers", [])[:3]
    if top_pitchers:
        parts.append("**주요 투수**:")
        for p in top_pitchers:
            parts.append(
                f"- {p['player_name']}: ERA {_safe_float(p.get('era')):.2f}, {p.get('wins')}승"
            )

    form_signals = team_data.get("player_form_signals", {})
    if form_signals and form_signals.get("found"):
        parts.append("**핵심 폼 진단**:")
        for signal in (
            form_signals.get("batters", [])[:1] + form_signals.get("pitchers", [])[:1]
        ):
            player_name = signal.get("player_name", "선수 미상")
            form_score = signal.get("form_score")
            score_text = (
                f"{float(form_score):.1f}"
                if isinstance(form_score, (int, float))
                else "데이터 부족"
            )
            parts.append(
                f"- {player_name}: {_form_status_label(signal.get('form_status'))} ({score_text})"
            )

    # 최근 폼 — DB schema: summary={wins,losses,draws,run_diff}, games=[{result:"Win"/"Loss"/"Draw", score:"5:3", run_diff, date, opponent}]
    recent = team_data.get("recent", {})
    if recent and recent.get("found"):
        parts.append("**최근 경기 흐름**:")
        r_summary = recent.get("summary", {})
        r_games = recent.get("games", [])
        wins = r_summary.get("wins", 0)
        losses = r_summary.get("losses", 0)
        draws = r_summary.get("draws", 0)
        parts.append(
            f"- 최근 {len(r_games)}경기: {wins}승 {losses}패{f' {draws}무' if draws else ''}"
        )
        run_diff = r_summary.get("run_diff")
        if run_diff is not None:
            parts.append(f"- 득실 마진: {'+' if run_diff >= 0 else ''}{run_diff}")
        win_rate = r_summary.get("win_rate")
        if win_rate is not None:
            parts.append(f"- 승률: {win_rate:.3f}")
        parts.append("")

    parts.append("")
    return "\n".join(parts)


def _format_coach_context(
    tool_results: Dict[str, Any],
    focus: List[str],
    game_context: Optional[str] = None,
    league_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Coach 전용 컨텍스트를 포맷합니다.
    듀얼 팀 데이터 지원.
    """
    parts = []

    # 1. 리그/경기 컨텍스트
    if league_context:
        season = league_context.get("season_year") or league_context.get("season")
        league_type = league_context.get("league_type")
        parts.append(f"## 🏟️ {season} 시즌 컨텍스트")

        if league_type == "POST":
            parts.append(
                f"**{league_context.get('round')} {league_context.get('game_no')}차전**"
            )
        else:
            home = league_context.get("home", {})
            away = league_context.get("away", {})
            parts.append(
                f"- **Home**: {home.get('rank')}위 ({home.get('gamesBehind')} GB)"
            )
            parts.append(
                f"- **Away**: {away.get('rank')}위 ({away.get('gamesBehind')} GB)"
            )
        parts.append("")

    # 2. 경기 별 모드 안내
    if game_context:
        parts.append("## ⚠️ 특정 경기 분석 모드")
        parts.append(f"**분석 대상**: {game_context}")
        parts.append("")

    # 3. 팀별 데이터
    if tool_results.get("home"):
        parts.append(_format_team_stats(tool_results["home"], "Home"))

    if tool_results.get("away"):
        parts.append(_format_team_stats(tool_results["away"], "Away"))

    # 4. 상대 전적
    matchup = tool_results.get("matchup", {})
    if matchup and matchup.get("games"):
        parts.append("### ⚔️ 맞대결 전적")
        summary = matchup.get("summary", {})
        t1 = matchup.get("team1", "팀1")
        t2 = matchup.get("team2", "팀2")
        if _matchup_is_partial(matchup):
            parts.append("- DB 이력 부족으로 시리즈 스코어 축약 표시")
        else:
            parts.append(
                f"- {t1} {summary.get('team1_wins', 0)}승 / "
                f"{t2} {summary.get('team2_wins', 0)}승 / "
                f"{summary.get('draws', 0)}무"
            )
        parts.append("| 날짜 | 스코어 | 결과 |")
        parts.append("|------|--------|------|")
        for g in matchup.get("games", [])[:3]:
            game_date = g.get("game_date", "")
            if hasattr(game_date, "strftime"):
                game_date = game_date.strftime("%Y-%m-%d")
            score = f"{g.get('home_score', 0)}:{g.get('away_score', 0)}"
            result_val = g.get("game_result", "")
            parts.append(f"| {game_date} | {score} | {result_val} |")
        parts.append("")

    clutch_moments = _clutch_moments(tool_results)
    if clutch_moments:
        parts.append("### 🎯 클러치/WPA")
        for moment in clutch_moments[:3]:
            parts.append(f"- {_format_clutch_fact(moment)}")
        parts.append("")

    return "\n".join(parts)


class AnalyzeRequest(BaseModel):
    team_id: Optional[str] = None  # deprecated — use home_team_id
    home_team_id: Optional[str] = None
    away_team_id: Optional[str] = None
    league_context: Optional[Dict[str, Any]] = None
    focus: List[str] = []
    game_id: Optional[str] = None
    request_mode: Literal[COACH_REQUEST_MODE_AUTO, COACH_REQUEST_MODE_MANUAL] = (
        COACH_REQUEST_MODE_MANUAL
    )
    question_override: Optional[str] = None
    starter_signature: Optional[str] = None
    lineup_signature: Optional[str] = None
    expected_cache_key: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def backfill_home_team_id(cls, values: Any) -> Any:
        """team_id만 보내는 기존 호출을 home_team_id로 매핑"""
        if isinstance(values, dict):
            focus = values.get("focus")
            if (
                focus is not None
                and isinstance(focus, list)
                and len(focus) > MAX_COACH_FOCUS_ITEMS
            ):
                raise ValueError(
                    f"focus 항목은 최대 {MAX_COACH_FOCUS_ITEMS}개까지 허용됩니다."
                )

            question_override = values.get("question_override")
            if isinstance(question_override, str):
                question_override_trimmed = question_override.strip()
                if not question_override_trimmed:
                    values["question_override"] = None
                elif (
                    len(question_override_trimmed) > MAX_COACH_QUESTION_OVERRIDE_LENGTH
                ):
                    raise ValueError(
                        "question_override가 너무 깁니다. "
                        f"최대 {MAX_COACH_QUESTION_OVERRIDE_LENGTH}자까지 허용됩니다."
                    )
                else:
                    values["question_override"] = question_override_trimmed

            request_mode = values.get("request_mode")
            if (
                request_mode == COACH_REQUEST_MODE_AUTO
                and values.get("question_override") is not None
            ):
                raise ValueError(
                    "auto_brief 모드에서는 question_override를 사용할 수 없습니다."
                )

            if not values.get("home_team_id") and values.get("team_id"):
                values["home_team_id"] = values["team_id"]

            for signature_key in (
                "starter_signature",
                "lineup_signature",
                "expected_cache_key",
            ):
                signature_value = values.get(signature_key)
                if isinstance(signature_value, str):
                    trimmed = signature_value.strip()
                    values[signature_key] = trimmed or None
        return values


@router.post("/analyze")
async def analyze_team(
    payload: AnalyzeRequest,
    agent: BaseballStatisticsAgent = Depends(get_agent),
    __: None = Depends(rate_limit_coach_dependency),
    _: None = Depends(require_ai_internal_token),
):
    """
    특정 팀(들)에 대한 심층 분석을 요청합니다. 'The Coach' 페르소나가 적용됩니다.
    """
    from sse_starlette.sse import EventSourceResponse

    # 하위 호환성은 model_validator에서 처리됨
    if not payload.home_team_id:
        raise HTTPException(
            status_code=400, detail="home_team_id 또는 team_id가 필요합니다."
        )

    try:
        request_id = uuid.uuid4().hex[:8]
        pool = get_connection_pool()
        team_resolver = TeamCodeResolver()

        home_team_canonical = team_resolver.resolve_canonical(payload.home_team_id)
        away_team_canonical = (
            team_resolver.resolve_canonical(payload.away_team_id)
            if payload.away_team_id
            else None
        )
        if home_team_canonical not in CANONICAL_CODES:
            raise HTTPException(
                status_code=400,
                detail="unsupported_team_for_regular_analysis",
            )
        if away_team_canonical and away_team_canonical not in CANONICAL_CODES:
            raise HTTPException(
                status_code=400,
                detail="unsupported_team_for_regular_analysis",
            )

        year, resolve_source = _resolve_target_year(payload, pool)
        if not _is_valid_analysis_year(year):
            raise HTTPException(status_code=400, detail="analysis_year_out_of_range")

        home_name = agent._convert_team_id_to_name(payload.home_team_id)
        away_name = (
            agent._convert_team_id_to_name(payload.away_team_id)
            if payload.away_team_id
            else None
        )
        game_evidence = _collect_game_evidence(
            pool,
            payload,
            year=year,
            home_team_code=home_team_canonical,
            away_team_code=away_team_canonical,
            home_team_name=home_name,
            away_team_name=away_name,
            team_resolver=team_resolver,
        )
        home_team_canonical = game_evidence.home_team_code or home_team_canonical
        away_team_canonical = game_evidence.away_team_code or away_team_canonical
        home_name = game_evidence.home_team_name or home_name
        away_name = game_evidence.away_team_name or away_name
        effective_league_context = _build_effective_league_context(
            payload.league_context,
            game_evidence,
        )
        request_mode = payload.request_mode
        is_auto_brief = request_mode == COACH_REQUEST_MODE_AUTO
        input_focus = list(payload.focus or [])
        resolved_focus = (
            ["recent_form"] if is_auto_brief else normalize_focus(input_focus)
        )
        if is_auto_brief:
            if payload.question_override:
                raise HTTPException(
                    status_code=400,
                    detail="auto_brief 요청에서는 question_override를 사용할 수 없습니다.",
                )
            effective_question_override = None
            question_signature_override = "auto"
            query = _build_coach_query(
                home_name,
                resolved_focus,
                opponent_name=away_name,
                league_context=effective_league_context,
                game_status_bucket=game_evidence.game_status_bucket,
            )
        else:
            effective_question_override = payload.question_override
            question_signature_override = None
            if payload.question_override:
                query = payload.question_override
            else:
                query = _build_coach_query(
                    home_name,
                    resolved_focus,
                    opponent_name=away_name,
                    league_context=effective_league_context,
                    game_status_bucket=game_evidence.game_status_bucket,
                )

        settings = get_settings()
        sse_ping_seconds = max(1, int(settings.chat_sse_ping_seconds))
        coach_model_candidates = resolve_coach_openrouter_models(
            settings.coach_openrouter_model or settings.openrouter_model,
            list(getattr(settings, "coach_openrouter_fallback_models", []) or []),
        )
        coach_model_name = coach_model_candidates[0]

        # Cache Key 생성
        game_type = str(
            effective_league_context.get("league_type") or "UNKNOWN"
        ).upper()
        expected_cache_key = str(payload.expected_cache_key or "").strip() or None
        cache_key, cache_key_payload, starter_signature, lineup_signature = (
            build_coach_cache_identity(
                home_team_code=home_team_canonical,
                away_team_code=away_team_canonical,
                year=year,
                game_type=game_type,
                focus=resolved_focus,
                question_override=effective_question_override,
                game_id=game_evidence.game_id,
                league_type_code=game_evidence.league_type_code,
                stage_label=game_evidence.stage_label,
                request_mode=request_mode,
                game_status_bucket=game_evidence.game_status_bucket,
                question_signature_override=question_signature_override,
                requested_starter_signature=payload.starter_signature,
                requested_lineup_signature=payload.lineup_signature,
                home_pitcher=game_evidence.home_pitcher,
                away_pitcher=game_evidence.away_pitcher,
                lineup_players=[*game_evidence.home_lineup, *game_evidence.away_lineup],
            )
        )
        cache_key_mismatch = bool(
            expected_cache_key and expected_cache_key != cache_key
        )
        cache_contract_meta = {
            "cache_key": cache_key,
            "resolved_cache_key": cache_key,
            "expected_cache_key": expected_cache_key,
            "prompt_version": COACH_CACHE_PROMPT_VERSION,
            "starter_signature": starter_signature,
            "lineup_signature": lineup_signature,
            "cache_key_mismatch": cache_key_mismatch,
        }
        focus_signature = str(cache_key_payload["focus_signature"])
        question_signature = str(cache_key_payload["question_signature"])

        logger.info(
            "[Coach Router] request_mode=%s Analyzing %s vs %s (year=%d game_id=%s stage=%s status=%s): %s... "
            "(CacheKey: %s) expected_cache_key=%s prompt_version=%s starter_signature=%s lineup_signature=%s "
            "input_season=%s resolved_year=%d resolve_source=%s input_focus=%s resolved_focus=%s "
            "focus_signature=%s question_signature=%s cache_key_version=%s",
            request_mode,
            home_name,
            away_name or "Single",
            year,
            game_evidence.game_id,
            game_evidence.stage_label,
            game_evidence.game_status_bucket,
            query[:100],
            cache_key,
            expected_cache_key,
            COACH_CACHE_PROMPT_VERSION,
            starter_signature,
            lineup_signature,
            effective_league_context.get("season"),
            year,
            resolve_source,
            input_focus,
            resolved_focus,
            focus_signature,
            question_signature,
            COACH_CACHE_SCHEMA_VERSION,
        )
        if cache_key_mismatch:
            logger.warning(
                "[Coach] Cache key mismatch detected game_id=%s expected_cache_key=%s resolved_cache_key=%s prompt_version=%s "
                "starter_signature=%s lineup_signature=%s",
                game_evidence.game_id,
                expected_cache_key,
                cache_key,
                COACH_CACHE_PROMPT_VERSION,
                starter_signature,
                lineup_signature,
            )
        evidence_assessment = assess_game_evidence(game_evidence)
        manual_data_request = _build_manual_data_request(
            pool,
            payload,
            game_evidence,
            evidence_assessment,
        )
        unavailable_game_status_message = _get_unavailable_game_status_message(
            getattr(game_evidence, "game_status", None)
        )

        async def event_generator():
            try:
                total_start = perf_counter()

                # Phase 1: 시작
                yield {
                    "event": "status",
                    "data": json.dumps(
                        {"message": "양 팀 전력 분석 중..."}, ensure_ascii=False
                    ),
                }
                # Phase 0: 캐시 확인
                cached_data = None
                cache_state = "MISS_GENERATE"
                cache_error_message = None
                cache_error_code = None
                cache_attempt_count = 0
                cache_lease_owner = _build_cache_lease_owner()
                heartbeat_task: Optional[asyncio.Task] = None
                lease_lost_event = asyncio.Event()

                def _build_stream_meta(
                    extra_payload: Optional[Dict[str, Any]] = None,
                ) -> Dict[str, Any]:
                    payload_data = {
                        "request_mode": request_mode,
                        "resolved_focus": resolved_focus,
                        "focus_signature": focus_signature,
                        "question_signature": question_signature,
                        "cache_key_version": COACH_CACHE_SCHEMA_VERSION,
                        **cache_contract_meta,
                    }
                    if extra_payload:
                        payload_data.update(extra_payload)
                    payload_data = _ensure_stream_meta_contract(payload_data)
                    _log_coach_stream_meta(
                        payload_data,
                        game_id=game_evidence.game_id,
                    )
                    return payload_data

                if unavailable_game_status_message is not None:
                    logger.info(
                        "[Coach] Skipping analysis for unavailable game status game_id=%s raw_status=%s",
                        game_evidence.game_id,
                        game_evidence.game_status,
                    )
                    yield {
                        "event": "meta",
                        "data": json.dumps(
                            _build_stream_meta(
                                {
                                    "validation_status": "game_unavailable",
                                    "generation_mode": "evidence_fallback",
                                    "data_quality": "insufficient",
                                    "used_evidence": evidence_assessment.used_evidence,
                                    "grounding_warnings": [
                                        unavailable_game_status_message
                                    ],
                                    "grounding_reasons": evidence_assessment.root_causes,
                                    "supported_fact_count": 0,
                                    "cache_state": "MISS_GENERATE",
                                    "cached": False,
                                    "in_progress": False,
                                    "game_status_bucket": game_evidence.game_status_bucket,
                                }
                            ),
                            ensure_ascii=False,
                        ),
                    }
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {
                                "code": "game_unavailable",
                                "message": unavailable_game_status_message,
                            },
                            ensure_ascii=False,
                        ),
                    }
                    return

                if manual_data_request is not None:
                    yield {
                        "event": "meta",
                        "data": json.dumps(
                            _build_stream_meta(
                                {
                                    "validation_status": "manual_data_required",
                                    "generation_mode": "evidence_fallback",
                                    "data_quality": "insufficient",
                                    "used_evidence": evidence_assessment.used_evidence,
                                    "grounding_warnings": [
                                        MANUAL_BASEBALL_DATA_REQUIRED_MESSAGE
                                    ],
                                    "grounding_reasons": evidence_assessment.root_causes,
                                    "supported_fact_count": 0,
                                    "cache_state": "MISS_GENERATE",
                                    "cached": False,
                                    "in_progress": False,
                                    "manual_data_request": manual_data_request,
                                }
                            ),
                            ensure_ascii=False,
                        ),
                    }
                    yield {"event": "done", "data": "[DONE]"}
                    return

                (
                    cache_state,
                    cached_data,
                    cache_error_message,
                    cache_error_code,
                    cache_attempt_count,
                ) = _claim_cache_generation(
                    pool=pool,
                    cache_key=cache_key,
                    team_id=home_team_canonical,
                    year=year,
                    prompt_version=COACH_CACHE_PROMPT_VERSION,
                    model_name=coach_model_name,
                    lease_owner=cache_lease_owner,
                    completed_ttl_seconds=_cache_ttl_seconds_for_status_bucket(
                        game_evidence.game_status_bucket
                    ),
                    request_mode=request_mode,
                )

                logger.info(
                    "[Coach] Cache gate=%s game_id=%s resolved_cache_key=%s expected_cache_key=%s prompt_version=%s "
                    "starter_signature=%s lineup_signature=%s request_mode=%s focus_signature=%s "
                    "question_signature=%s cache_key_version=%s attempt_count=%s error_code=%s",
                    cache_state,
                    game_evidence.game_id,
                    cache_key,
                    expected_cache_key,
                    COACH_CACHE_PROMPT_VERSION,
                    starter_signature,
                    lineup_signature,
                    request_mode,
                    focus_signature,
                    question_signature,
                    COACH_CACHE_SCHEMA_VERSION,
                    cache_attempt_count,
                    cache_error_code,
                )

                if cached_data:
                    cached_data, cached_meta = _extract_cached_payload(
                        cached_data,
                        request_mode=request_mode,
                    )
                    cached_meta = _merge_cache_contract_meta(
                        cached_meta,
                        cache_contract_meta,
                    )
                    missing_focus_sections = _find_missing_focus_sections(
                        cached_data, resolved_focus
                    )
                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {"message": "분석 데이터를 불러옵니다..."},
                            ensure_ascii=False,
                        ),
                    }
                    json_str = json.dumps(cached_data, ensure_ascii=False, indent=2)
                    yield {
                        "event": "message",
                        "data": json.dumps({"delta": json_str}, ensure_ascii=False),
                    }
                    yield {
                        "event": "meta",
                        "data": json.dumps(
                            _build_stream_meta(
                                {
                                    "validation_status": "success",
                                    "structured_response": cached_data,
                                    "fast_path": True,
                                    "cached": True,
                                    "cache_state": cache_state,
                                    "in_progress": False,
                                    "focus_section_missing": bool(
                                        missing_focus_sections
                                    ),
                                    "missing_focus_sections": missing_focus_sections,
                                    **cached_meta,
                                }
                            ),
                            ensure_ascii=False,
                        ),
                    }
                    yield {"event": "done", "data": "[DONE]"}
                    return

                wait_result: Optional[Dict[str, Any]] = None
                if not _should_generate_from_gate(cache_state):
                    if cache_state == "PENDING_WAIT":
                        wait_result = await _wait_for_cache_terminal_state(
                            pool=pool,
                            cache_key=cache_key,
                            timeout_seconds=PENDING_WAIT_TIMEOUT_SECONDS,
                            poll_ms=PENDING_WAIT_POLL_MS,
                        )
                        if (
                            wait_result
                            and wait_result.get("status") == "COMPLETED"
                            and wait_result.get("response_json")
                        ):
                            cached_wait_data, cached_wait_meta = (
                                _extract_cached_payload(
                                    wait_result["response_json"],
                                    request_mode=request_mode,
                                )
                            )
                            cached_wait_meta = _merge_cache_contract_meta(
                                cached_wait_meta,
                                cache_contract_meta,
                            )
                            missing_focus_sections = _find_missing_focus_sections(
                                cached_wait_data, resolved_focus
                            )
                            yield {
                                "event": "status",
                                "data": json.dumps(
                                    {
                                        "message": "진행 중이던 분석 결과를 불러옵니다..."
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                            json_str = json.dumps(
                                cached_wait_data, ensure_ascii=False, indent=2
                            )
                            yield {
                                "event": "message",
                                "data": json.dumps(
                                    {"delta": json_str}, ensure_ascii=False
                                ),
                            }
                            yield {
                                "event": "meta",
                                "data": json.dumps(
                                    _build_stream_meta(
                                        {
                                            "validation_status": "success",
                                            "structured_response": cached_wait_data,
                                            "fast_path": True,
                                            "cached": True,
                                            "cache_state": "PENDING_WAIT",
                                            "in_progress": False,
                                            "focus_section_missing": bool(
                                                missing_focus_sections
                                            ),
                                            "missing_focus_sections": missing_focus_sections,
                                            **cached_wait_meta,
                                        }
                                    ),
                                    ensure_ascii=False,
                                ),
                            }
                            yield {"event": "done", "data": "[DONE]"}
                            return

                        if wait_result and wait_result.get("status") in {
                            "FAILED",
                            "PENDING_STALE_TAKEOVER",
                            "MISSING_ROW",
                        }:
                            (
                                cache_state,
                                cached_data,
                                cache_error_message,
                                cache_error_code,
                                cache_attempt_count,
                            ) = _claim_cache_generation(
                                pool=pool,
                                cache_key=cache_key,
                                team_id=home_team_canonical,
                                year=year,
                                prompt_version=COACH_CACHE_PROMPT_VERSION,
                                model_name=coach_model_name,
                                lease_owner=cache_lease_owner,
                                completed_ttl_seconds=_cache_ttl_seconds_for_status_bucket(
                                    game_evidence.game_status_bucket
                                ),
                                request_mode=request_mode,
                            )
                            if cached_data:
                                cached_data, cached_meta = _extract_cached_payload(
                                    cached_data,
                                    request_mode=request_mode,
                                )
                                cached_meta = _merge_cache_contract_meta(
                                    cached_meta,
                                    {
                                        **cache_contract_meta,
                                        "cache_row_missing": True,
                                        "recovered_from_missing_row": True,
                                    },
                                )
                                missing_focus_sections = _find_missing_focus_sections(
                                    cached_data, resolved_focus
                                )
                                yield {
                                    "event": "status",
                                    "data": json.dumps(
                                        {
                                            "message": "재생성된 분석 데이터를 불러옵니다..."
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                                json_str = json.dumps(
                                    cached_data, ensure_ascii=False, indent=2
                                )
                                yield {
                                    "event": "message",
                                    "data": json.dumps(
                                        {"delta": json_str}, ensure_ascii=False
                                    ),
                                }
                                yield {
                                    "event": "meta",
                                    "data": json.dumps(
                                        _build_stream_meta(
                                            {
                                                "validation_status": "success",
                                                "structured_response": cached_data,
                                                "fast_path": True,
                                                "cached": True,
                                                "cache_state": cache_state,
                                                "in_progress": False,
                                                "focus_section_missing": bool(
                                                    missing_focus_sections
                                                ),
                                                "missing_focus_sections": missing_focus_sections,
                                                **cached_meta,
                                            }
                                        ),
                                        ensure_ascii=False,
                                    ),
                                }
                                yield {"event": "done", "data": "[DONE]"}
                                return
                            if wait_result.get("status") == "MISSING_ROW":
                                cache_error_code = cache_error_code or None

                    if cache_state in {"PENDING_WAIT", "FAILED_RETRY_WAIT"}:
                        retryable_failure = (
                            cache_state == "FAILED_RETRY_WAIT"
                            or _is_retryable_cache_error_code(cache_error_code)
                        )
                        waiting_payload = _cache_status_response(
                            headline=(
                                f"{home_name} 분석이 진행 중입니다"
                                if cache_state == "PENDING_WAIT"
                                else f"{home_name} 분석 복구를 준비 중입니다"
                            ),
                            coach_note=(
                                "잠시 후 다시 시도해주세요."
                                if cache_state == "PENDING_WAIT"
                                else "일시 오류를 자동 복구하는 중입니다. 잠시 후 다시 시도해주세요."
                            ),
                            detail=(
                                "## 캐시 준비 중\n\n동일 경기 분석 요청이 이미 진행 중입니다."
                                if cache_state == "PENDING_WAIT"
                                else (
                                    "## 자동 복구 대기 중\n\n"
                                    "일시 오류가 감지되어 다음 재생성 윈도우를 기다리고 있습니다.\n\n"
                                    f"- 오류 코드: {_sanitize_cache_error_code(cache_error_code)}\n"
                                    f"- 재시도 가능: {'예' if retryable_failure else '아니오'}"
                                )
                            ),
                        )
                        waiting_meta_defaults = _build_meta_payload_defaults(
                            generation_mode="evidence_fallback",
                            data_quality=_determine_data_quality(game_evidence),
                            used_evidence=game_evidence.used_evidence,
                            **cache_contract_meta,
                            game_status_bucket=game_evidence.game_status_bucket,
                            grounding_warnings=_merge_grounding_warnings(
                                [],
                                evidence_assessment.root_causes,
                            ),
                            grounding_reasons=evidence_assessment.root_causes,
                            supported_fact_count=0,
                            cache_error_code=cache_error_code,
                            retryable_failure=retryable_failure,
                            attempt_count=cache_attempt_count,
                            cache_row_missing=bool(
                                wait_result
                                and wait_result.get("status") == "MISSING_ROW"
                            ),
                        )
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                {
                                    "message": (
                                        "기존 분석 작업을 기다리는 중입니다..."
                                        if cache_state == "PENDING_WAIT"
                                        else "일시 오류 복구 윈도우를 기다리는 중입니다..."
                                    )
                                },
                                ensure_ascii=False,
                            ),
                        }
                        yield {
                            "event": "message",
                            "data": json.dumps(
                                {
                                    "delta": json.dumps(
                                        waiting_payload, ensure_ascii=False
                                    )
                                },
                                ensure_ascii=False,
                            ),
                        }
                        yield {
                            "event": "meta",
                            "data": json.dumps(
                                _build_stream_meta(
                                    {
                                        "validation_status": "fallback",
                                        "fast_path": True,
                                        "cached": False,
                                        "cache_state": cache_state,
                                        "in_progress": True,
                                        **waiting_meta_defaults,
                                    }
                                ),
                                ensure_ascii=False,
                            ),
                        }
                        yield {"event": "done", "data": "[DONE]"}
                        return

                    if cache_state == "FAILED_LOCKED":
                        failed_error_code = _sanitize_cache_error_code(
                            cache_error_code or cache_error_message
                        )
                        failed_payload = _cache_status_response(
                            headline=f"{home_name} 분석 캐시 갱신이 필요합니다",
                            coach_note="수동 배치로 캐시를 갱신한 뒤 다시 시도해주세요.",
                            detail=(
                                "## 캐시 잠금 상태\n\n"
                                "자동 재생성은 비활성화되어 있습니다.\n\n"
                                f"- 오류 코드: {failed_error_code}\n"
                                f"- 안내: {_cache_error_message_for_user(failed_error_code)}"
                            ),
                        )
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                {"message": "분석 캐시가 잠금 상태입니다."},
                                ensure_ascii=False,
                            ),
                        }
                        yield {
                            "event": "message",
                            "data": json.dumps(
                                {
                                    "delta": json.dumps(
                                        failed_payload, ensure_ascii=False
                                    )
                                },
                                ensure_ascii=False,
                            ),
                        }
                        failed_meta_defaults = _build_meta_payload_defaults(
                            generation_mode="evidence_fallback",
                            data_quality=_determine_data_quality(game_evidence),
                            used_evidence=game_evidence.used_evidence,
                            **cache_contract_meta,
                            game_status_bucket=game_evidence.game_status_bucket,
                            grounding_warnings=_merge_grounding_warnings(
                                [],
                                evidence_assessment.root_causes,
                            ),
                            grounding_reasons=evidence_assessment.root_causes,
                            supported_fact_count=0,
                            cache_error_code=failed_error_code,
                            retryable_failure=False,
                            attempt_count=cache_attempt_count,
                        )
                        yield {
                            "event": "meta",
                            "data": json.dumps(
                                _build_stream_meta(
                                    {
                                        "validation_status": "fallback",
                                        "fast_path": True,
                                        "cached": False,
                                        "cache_state": "FAILED_LOCKED",
                                        "in_progress": False,
                                        **failed_meta_defaults,
                                    }
                                ),
                                ensure_ascii=False,
                            ),
                        }
                        yield {"event": "done", "data": "[DONE]"}
                        return

                heartbeat_task = asyncio.create_task(
                    _heartbeat_cache_lease(
                        pool=pool,
                        cache_key=cache_key,
                        lease_owner=cache_lease_owner,
                        lease_lost_event=lease_lost_event,
                    )
                )

                # 도구 실행
                yield {
                    "event": "tool_start",
                    "data": json.dumps(
                        {"tool": "parallel_fetch_team_data"}, ensure_ascii=False
                    ),
                }

                tool_results = await _execute_coach_tools_parallel(
                    pool,
                    home_team_canonical,
                    year,
                    resolved_focus,
                    away_team_canonical,
                    as_of_game_date=game_evidence.game_date,
                    exclude_game_id=game_evidence.game_id,
                    matchup_season_id=(
                        game_evidence.season_id
                        if game_evidence.stage_label not in {"REGULAR", "UNKNOWN"}
                        else None
                    ),
                    game_id=game_evidence.game_id,
                    include_clutch=_is_completed_review(game_evidence),
                )
                matchup_scope_sanitized = False
                if tool_results.get("matchup"):
                    sanitized_matchup = _sanitize_matchup_result_for_evidence(
                        game_evidence,
                        tool_results["matchup"],
                    )
                    matchup_scope_sanitized = (
                        sanitized_matchup is not tool_results["matchup"]
                    )
                    tool_results["matchup"] = sanitized_matchup

                yield {
                    "event": "tool_result",
                    "data": json.dumps(
                        {
                            "tool": "parallel_fetch_team_data",
                            "success": True,
                            "message": "데이터 조회 완료",
                        },
                        ensure_ascii=False,
                    ),
                }

                # Phase 2: fact sheet 구성
                allowed_names = _collect_allowed_entity_names(
                    game_evidence, tool_results
                )
                used_evidence = list(game_evidence.used_evidence)
                if tool_results.get("home", {}).get("summary", {}).get("found"):
                    used_evidence.append("team_summary")
                if tool_results.get("home", {}).get("advanced", {}).get("found"):
                    used_evidence.append("team_advanced_metrics")
                if (
                    tool_results.get("home", {})
                    .get("player_form_signals", {})
                    .get("found")
                ):
                    used_evidence.append("team_player_form_signals")
                if tool_results.get("home", {}).get("recent", {}).get("found"):
                    used_evidence.append("team_recent_form")
                if tool_results.get("matchup", {}).get("found"):
                    used_evidence.append("head_to_head")
                if tool_results.get("away", {}).get("summary", {}).get("found"):
                    used_evidence.append("opponent_team_summary")
                if tool_results.get("away", {}).get("advanced", {}).get("found"):
                    used_evidence.append("opponent_team_advanced_metrics")
                if (
                    tool_results.get("away", {})
                    .get("player_form_signals", {})
                    .get("found")
                ):
                    used_evidence.append("opponent_player_form_signals")
                if tool_results.get("away", {}).get("recent", {}).get("found"):
                    used_evidence.append("opponent_recent_form")
                if tool_results.get("clutch_moments", {}).get("found"):
                    used_evidence.append("game_clutch_moments")
                used_evidence = list(dict.fromkeys(used_evidence))

                dependency_degraded = _collect_dependency_degraded(tool_results)
                dependency_degraded_reasons = _collect_dependency_degraded_reasons(
                    tool_results
                )
                data_quality = _determine_data_quality(game_evidence, tool_results)
                fact_sheet = _build_coach_fact_sheet(
                    game_evidence,
                    tool_results,
                    allowed_names,
                    evidence_assessment,
                )
                fact_sheet_context = format_coach_fact_sheet(fact_sheet)
                grounding_reasons = list(fact_sheet.reasons)
                grounding_warnings = list(fact_sheet.warnings)
                if matchup_scope_sanitized:
                    grounding_reasons.append("postseason_matchup_scope_adjusted")
                    grounding_warnings.append(
                        "포스트시즌 상대 전적은 동일 시리즈 기준으로 다시 계산했습니다."
                    )
                if dependency_degraded:
                    grounding_reasons.append(COACH_OCI_MAPPING_DEGRADED_ERROR_CODE)
                    if dependency_degraded_reasons:
                        grounding_warnings.append(
                            "매핑 조회는 복구 경로로 이어졌습니다: "
                            + ", ".join(dependency_degraded_reasons)
                        )
                llm_focus = list(resolved_focus)
                if not is_auto_brief:
                    supported_focuses = _resolve_supported_focuses(
                        resolved_focus,
                        game_evidence,
                        tool_results,
                    )
                    unsupported_focuses = [
                        focus
                        for focus in resolved_focus
                        if focus not in supported_focuses
                    ]
                    if unsupported_focuses:
                        grounding_reasons.append("focus_data_unavailable")
                        grounding_warnings.append(
                            _build_focus_data_warning(
                                unsupported_focuses,
                                supported_focuses,
                            )
                        )
                    llm_focus = supported_focuses
                grounding_reasons = _normalize_grounding_reasons(grounding_reasons)
                grounding_warnings = _merge_grounding_warnings(
                    grounding_warnings,
                    grounding_reasons,
                )
                response_focus_targets = resolved_focus if is_auto_brief else llm_focus

                if lease_lost_event.is_set():
                    waiting_payload = _cache_status_response(
                        headline=f"{home_name} 분석이 진행 중입니다",
                        coach_note="분석 소유권이 변경되어 현재 요청은 결과 저장을 중단했습니다. 잠시 후 다시 시도해주세요.",
                        detail=(
                            "## 캐시 준비 중\n\n"
                            "진행 중이던 분석 소유권이 다른 작업으로 넘어가 현재 요청은 결과 저장을 중단했습니다."
                        ),
                    )
                    waiting_meta_defaults = _build_meta_payload_defaults(
                        generation_mode="evidence_fallback",
                        data_quality=data_quality,
                        used_evidence=used_evidence,
                        **cache_contract_meta,
                        game_status_bucket=game_evidence.game_status_bucket,
                        grounding_warnings=grounding_warnings,
                        grounding_reasons=grounding_reasons,
                        supported_fact_count=fact_sheet.supported_fact_count,
                        cache_error_code=cache_error_code,
                        retryable_failure=True,
                        dependency_degraded=dependency_degraded,
                        attempt_count=cache_attempt_count,
                        lease_lost=True,
                    )
                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {
                                "message": "기존 분석 작업 상태를 다시 확인하는 중입니다..."
                            },
                            ensure_ascii=False,
                        ),
                    }
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"delta": json.dumps(waiting_payload, ensure_ascii=False)},
                            ensure_ascii=False,
                        ),
                    }
                    yield {
                        "event": "meta",
                        "data": json.dumps(
                            _build_stream_meta(
                                {
                                    "validation_status": "fallback",
                                    "fast_path": True,
                                    "cached": False,
                                    "cache_state": "PENDING_WAIT",
                                    "in_progress": True,
                                    **waiting_meta_defaults,
                                }
                            ),
                            ensure_ascii=False,
                        ),
                    }
                    yield {"event": "done", "data": "[DONE]"}
                    return

                if _should_short_circuit_to_deterministic_response(
                    request_mode=request_mode,
                    fact_sheet=fact_sheet,
                    game_status_bucket=game_evidence.game_status_bucket,
                    grounding_reasons=grounding_reasons,
                ):
                    logger.info(
                        "[Coach] Deterministic short-circuit game_id=%s cache_key=%s request_mode=%s game_status_bucket=%s grounding_reasons=%s supported_fact_count=%d",
                        game_evidence.game_id,
                        cache_key,
                        request_mode,
                        game_evidence.game_status_bucket,
                        grounding_reasons,
                        fact_sheet.supported_fact_count,
                    )
                    generation_mode = (
                        "deterministic_auto" if is_auto_brief else "evidence_fallback"
                    )
                    response_payload = _build_deterministic_coach_response(
                        game_evidence,
                        tool_results,
                        resolved_focus=resolved_focus,
                        grounding_warnings=grounding_warnings,
                    )
                    response_payload = _postprocess_coach_response_payload(
                        response_payload,
                        evidence=game_evidence,
                        used_evidence=used_evidence,
                        grounding_reasons=grounding_reasons,
                        grounding_warnings=grounding_warnings,
                        tool_results=tool_results,
                        resolved_focus=resolved_focus,
                    )
                    _ensure_detailed_markdown(
                        response_payload,
                        resolved_focus,
                        grounding_warnings=grounding_warnings,
                        evidence=game_evidence,
                        tool_results=tool_results,
                    )
                    _ensure_completed_review_markdown_sections(
                        response_payload,
                        evidence=game_evidence,
                    )
                    response_json = json.dumps(response_payload, ensure_ascii=False)
                    meta_defaults = _build_meta_payload_defaults(
                        generation_mode=generation_mode,
                        data_quality=data_quality,
                        used_evidence=used_evidence,
                        **cache_contract_meta,
                        game_status_bucket=game_evidence.game_status_bucket,
                        grounding_warnings=grounding_warnings,
                        grounding_reasons=grounding_reasons,
                        supported_fact_count=fact_sheet.supported_fact_count,
                        dependency_degraded=dependency_degraded,
                        attempt_count=cache_attempt_count,
                        cache_row_missing=cache_state == "ROW_RECREATED",
                        recovered_from_missing_row=cache_state == "ROW_RECREATED",
                        lease_lost=lease_lost_event.is_set(),
                    )
                    finalize_result = _store_completed_cache(
                        pool=pool,
                        cache_key=cache_key,
                        lease_owner=cache_lease_owner,
                        team_id=home_team_canonical,
                        year=year,
                        prompt_version=COACH_CACHE_PROMPT_VERSION,
                        model_name=coach_model_name,
                        response_payload=response_payload,
                        meta_defaults=meta_defaults,
                    )
                    finalize_outcome = str(finalize_result.get("outcome") or "")
                    if finalize_outcome == "inserted_missing_row":
                        meta_defaults["cache_row_missing"] = True
                        meta_defaults["recovered_from_missing_row"] = True
                    elif finalize_outcome == "finalize_conflict":
                        waiting_payload = _cache_status_response(
                            headline=f"{home_name} 분석이 진행 중입니다",
                            coach_note="다른 작업이 캐시 반영을 이어받아 현재 요청은 결과 저장을 중단했습니다. 잠시 후 다시 시도해주세요.",
                            detail=(
                                "## 캐시 준비 중\n\n"
                                "분석 결과 저장 시 충돌이 발생해 현재 요청은 캐시 반영을 중단했습니다."
                            ),
                        )
                        waiting_meta_defaults = _build_meta_payload_defaults(
                            generation_mode=generation_mode,
                            data_quality=data_quality,
                            used_evidence=used_evidence,
                            **cache_contract_meta,
                            game_status_bucket=game_evidence.game_status_bucket,
                            grounding_warnings=grounding_warnings,
                            grounding_reasons=grounding_reasons,
                            supported_fact_count=fact_sheet.supported_fact_count,
                            dependency_degraded=dependency_degraded,
                            attempt_count=cache_attempt_count,
                            cache_finalize_conflict=True,
                            lease_lost=lease_lost_event.is_set(),
                        )
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                {
                                    "message": "기존 분석 작업 상태를 다시 확인하는 중입니다..."
                                },
                                ensure_ascii=False,
                            ),
                        }
                        yield {
                            "event": "message",
                            "data": json.dumps(
                                {
                                    "delta": json.dumps(
                                        waiting_payload, ensure_ascii=False
                                    )
                                },
                                ensure_ascii=False,
                            ),
                        }
                        yield {
                            "event": "meta",
                            "data": json.dumps(
                                _build_stream_meta(
                                    {
                                        "validation_status": "fallback",
                                        "verified": True,
                                        "fast_path": True,
                                        "cached": False,
                                        "cache_state": "PENDING_WAIT",
                                        "in_progress": True,
                                        **waiting_meta_defaults,
                                    }
                                ),
                                ensure_ascii=False,
                            ),
                        }
                        yield {"event": "done", "data": "[DONE]"}
                        return

                    missing_focus_sections = _find_missing_focus_sections(
                        response_payload, response_focus_targets
                    )
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"delta": response_json}, ensure_ascii=False
                        ),
                    }
                    yield {
                        "event": "meta",
                        "data": json.dumps(
                            _build_stream_meta(
                                {
                                    "validation_status": (
                                        "success" if is_auto_brief else "fallback"
                                    ),
                                    "verified": True,
                                    "fast_path": True,
                                    "cached": False,
                                    "cache_state": "COMPLETED",
                                    "in_progress": False,
                                    "focus_section_missing": bool(
                                        missing_focus_sections
                                    ),
                                    "missing_focus_sections": missing_focus_sections,
                                    "structured_response": response_payload,
                                    **meta_defaults,
                                }
                            ),
                            ensure_ascii=False,
                        ),
                    }
                    yield {"event": "done", "data": "[DONE]"}
                    return

                # Phase 3: LLM 호출
                yield {
                    "event": "status",
                    "data": json.dumps(
                        {"message": "AI 코치가 분석 리포트 작성 중..."},
                        ensure_ascii=False,
                    ),
                }

                focus_section_requirements = _build_focus_section_requirements(
                    llm_focus
                )
                prompt_query = query
                if (
                    not is_auto_brief
                    and effective_question_override is None
                    and llm_focus
                    and llm_focus != resolved_focus
                ):
                    prompt_query = _build_coach_query(
                        home_name,
                        llm_focus,
                        opponent_name=away_name,
                        league_context=effective_league_context,
                        game_status_bucket=game_evidence.game_status_bucket,
                    )
                coach_prompt = COACH_PROMPT_V2.format(
                    question=prompt_query,
                    context=fact_sheet_context,
                    focus_section_requirements=focus_section_requirements,
                )
                coach_llm = get_coach_llm_generator()
                settings = get_settings()
                default_max_tokens = (
                    settings.coach_brief_max_output_tokens
                    if is_auto_brief
                    else settings.coach_max_output_tokens
                )
                effective_max_tokens = _resolve_coach_llm_max_tokens(
                    default_max_tokens,
                    request_mode=request_mode,
                    game_status_bucket=game_evidence.game_status_bucket,
                    grounding_reasons=grounding_reasons,
                )
                response_payload: Dict[str, Any]
                validation_status = "fallback"
                generation_mode = "evidence_fallback"
                runtime_grounding_reasons = list(grounding_reasons)
                runtime_grounding_warnings = list(grounding_warnings)
                response_json = ""
                last_failure_reasons: List[str] = []
                max_llm_attempts = _resolve_coach_llm_attempt_limit(
                    request_mode,
                    game_status_bucket=game_evidence.game_status_bucket,
                    grounding_reasons=grounding_reasons,
                )
                llm_idle_timeout_seconds = _resolve_coach_llm_idle_timeout_seconds(
                    float(settings.coach_llm_read_timeout),
                    request_mode=request_mode,
                    game_status_bucket=game_evidence.game_status_bucket,
                    grounding_reasons=grounding_reasons,
                )
                empty_chunk_retry_limit = _resolve_coach_empty_chunk_retry_limit(
                    request_mode=request_mode,
                    game_status_bucket=game_evidence.game_status_bucket,
                    grounding_reasons=grounding_reasons,
                )
                llm_total_timeout_seconds = _resolve_coach_llm_total_timeout_seconds(
                    request_mode=request_mode,
                    game_status_bucket=game_evidence.game_status_bucket,
                    grounding_reasons=grounding_reasons,
                )
                llm_first_chunk_timeout_seconds = (
                    _resolve_coach_llm_first_chunk_timeout_seconds(
                        request_mode=request_mode,
                        game_status_bucket=game_evidence.game_status_bucket,
                        grounding_reasons=grounding_reasons,
                    )
                )
                llm_request_timeout_seconds = (
                    _resolve_coach_llm_request_timeout_seconds(
                        request_mode=request_mode,
                        game_status_bucket=game_evidence.game_status_bucket,
                        grounding_reasons=grounding_reasons,
                    )
                )
                logger.info(
                    "[Coach] LLM config game_id=%s cache_key=%s request_mode=%s game_status_bucket=%s max_attempts=%d max_tokens=%d idle_timeout=%.1fs first_chunk_timeout=%s total_timeout=%s request_timeout=%.1fs empty_chunk_retries=%d",
                    game_evidence.game_id,
                    cache_key,
                    request_mode,
                    game_evidence.game_status_bucket,
                    max_llm_attempts,
                    effective_max_tokens,
                    llm_idle_timeout_seconds,
                    (
                        f"{llm_first_chunk_timeout_seconds:.1f}s"
                        if llm_first_chunk_timeout_seconds is not None
                        else "none"
                    ),
                    (
                        f"{llm_total_timeout_seconds:.1f}s"
                        if llm_total_timeout_seconds is not None
                        else "none"
                    ),
                    llm_request_timeout_seconds,
                    empty_chunk_retry_limit,
                )

                for attempt in range(1, max_llm_attempts + 1):
                    attempt_started = perf_counter()
                    current_prompt = coach_prompt
                    if attempt > 1:
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                {
                                    "message": "AI 코치가 근거를 재검토하며 다시 작성 중..."
                                },
                                ensure_ascii=False,
                            ),
                        }
                        current_prompt = _build_coach_retry_prompt(
                            question=prompt_query,
                            context=fact_sheet_context,
                            focus_section_requirements=focus_section_requirements,
                            previous_response=response_json,
                            failure_reasons=last_failure_reasons,
                        )
                    logger.info(
                        "[Coach] LLM attempt start game_id=%s cache_key=%s attempt=%d/%d",
                        game_evidence.game_id,
                        cache_key,
                        attempt,
                        max_llm_attempts,
                    )

                    attempt_grounding_reasons = list(grounding_reasons)
                    attempt_grounding_warnings = list(grounding_warnings)
                    failure_reasons: List[str] = []
                    response_chunks = []
                    try:
                        llm_event_stream = _iterate_coach_llm_with_keepalive(
                            coach_llm=coach_llm,
                            messages=[{"role": "user", "content": current_prompt}],
                            max_tokens=effective_max_tokens,
                            heartbeat_seconds=COACH_LLM_STATUS_HEARTBEAT_SECONDS,
                            idle_timeout_seconds=llm_idle_timeout_seconds,
                            first_chunk_timeout_seconds=llm_first_chunk_timeout_seconds,
                            coach_llm_kwargs={
                                "empty_chunk_retry_limit": empty_chunk_retry_limit,
                                "request_timeout_seconds": llm_request_timeout_seconds,
                            },
                        )
                        if llm_total_timeout_seconds is None:
                            async for llm_event in llm_event_stream:
                                if llm_event.get("type") == "status":
                                    yield {
                                        "event": "status",
                                        "data": json.dumps(
                                            {
                                                "message": llm_event.get("message")
                                                or "AI 코치가 근거를 정리하는 중입니다..."
                                            },
                                            ensure_ascii=False,
                                        ),
                                    }
                                    continue
                                response_chunks.append(
                                    str(llm_event.get("chunk") or "")
                                )
                        else:
                            async with asyncio.timeout(llm_total_timeout_seconds):
                                async for llm_event in llm_event_stream:
                                    if llm_event.get("type") == "status":
                                        yield {
                                            "event": "status",
                                            "data": json.dumps(
                                                {
                                                    "message": llm_event.get("message")
                                                    or "AI 코치가 근거를 정리하는 중입니다..."
                                                },
                                                ensure_ascii=False,
                                            ),
                                        }
                                        continue
                                    response_chunks.append(
                                        str(llm_event.get("chunk") or "")
                                    )
                    except Exception as llm_exc:  # noqa: BLE001
                        logger.warning(
                            "[Coach] LLM failed on attempt=%d cache_key=%s elapsed_sec=%.2f error=%s",
                            attempt,
                            cache_key,
                            perf_counter() - attempt_started,
                            llm_exc,
                        )
                        llm_error_code = _cache_error_code_from_exception(llm_exc)
                        if llm_error_code != COACH_INTERNAL_ERROR_CODE:
                            failure_reasons.append(llm_error_code)
                        else:
                            failure_reasons.append("coach_llm_request_failed")
                        last_failure_reasons = list(dict.fromkeys(failure_reasons))
                        runtime_grounding_reasons = _normalize_grounding_reasons(
                            attempt_grounding_reasons
                        )
                        runtime_grounding_warnings = _merge_grounding_warnings(
                            attempt_grounding_warnings,
                            runtime_grounding_reasons,
                        )
                        response_json = ""
                        continue

                    full_response = "".join(response_chunks)
                    full_response = _remove_duplicate_json_start(full_response)
                    response_json = full_response

                    parsed_response, parse_error, parse_meta = (
                        parse_coach_response_with_meta(full_response)
                    )

                    if not parsed_response:
                        logger.warning(
                            "[Coach] Parse failed on attempt=%d cache_key=%s elapsed_sec=%.2f error=%s",
                            attempt,
                            cache_key,
                            perf_counter() - attempt_started,
                            parse_error,
                        )
                        parse_error_code = _cache_error_code_from_parse_meta(parse_meta)
                        if parse_error_code != COACH_INTERNAL_ERROR_CODE:
                            failure_reasons.append(parse_error_code)
                        failure_reasons.append("llm_parse_failed")
                    else:
                        candidate_payload = parsed_response.model_dump()
                        candidate_payload = _sanitize_response_placeholders(
                            candidate_payload,
                            used_evidence=used_evidence,
                        )
                        candidate_payload = _normalize_response_team_display(
                            candidate_payload,
                            evidence=game_evidence,
                        )
                        candidate_payload = _soften_scheduled_partial_tone(
                            candidate_payload,
                            game_status_bucket=game_evidence.game_status_bucket,
                            grounding_reasons=attempt_grounding_reasons,
                        )
                        candidate_payload = _normalize_response_markdown_layout(
                            candidate_payload
                        )
                        candidate_payload = _cleanup_response_language_quality(
                            candidate_payload
                        )
                        candidate_payload = _polish_scheduled_partial_response(
                            candidate_payload,
                            game_status_bucket=game_evidence.game_status_bucket,
                            grounding_reasons=attempt_grounding_reasons,
                        )
                        candidate_payload = (
                            _sanitize_scheduled_unconfirmed_lineup_entities(
                                candidate_payload,
                                evidence=game_evidence,
                                grounding_reasons=attempt_grounding_reasons,
                                tool_results=tool_results,
                                resolved_focus=resolved_focus,
                            )
                        )
                        candidate_payload = _enforce_completed_result_anchor(
                            candidate_payload,
                            evidence=game_evidence,
                            tool_results=tool_results,
                            resolved_focus=resolved_focus,
                        )
                        disallowed_entities = _find_disallowed_entities(
                            candidate_payload, allowed_names
                        )
                        if disallowed_entities:
                            sanitized_payload = _sanitize_response_disallowed_entities(
                                candidate_payload,
                                disallowed_entities,
                            )
                            sanitized_payload = _cleanup_response_language_quality(
                                _normalize_response_team_display(
                                    sanitized_payload,
                                    evidence=game_evidence,
                                )
                            )
                            sanitized_payload = _polish_scheduled_partial_response(
                                sanitized_payload,
                                game_status_bucket=game_evidence.game_status_bucket,
                                grounding_reasons=attempt_grounding_reasons,
                            )
                            sanitized_payload = (
                                _sanitize_scheduled_unconfirmed_lineup_entities(
                                    sanitized_payload,
                                    evidence=game_evidence,
                                    grounding_reasons=attempt_grounding_reasons,
                                    tool_results=tool_results,
                                    resolved_focus=resolved_focus,
                                )
                            )
                            sanitized_payload = _enforce_completed_result_anchor(
                                sanitized_payload,
                                evidence=game_evidence,
                                tool_results=tool_results,
                                resolved_focus=resolved_focus,
                            )
                            remaining_entities = _find_disallowed_entities(
                                sanitized_payload,
                                allowed_names,
                            )
                            if not remaining_entities:
                                candidate_payload = sanitized_payload
                                attempt_grounding_reasons.append(
                                    "unsupported_entity_name_sanitized"
                                )
                                logger.info(
                                    "[Coach] Sanitized disallowed entities cache_key=%s attempt=%d tokens=%s",
                                    cache_key,
                                    attempt,
                                    disallowed_entities[:4],
                                )
                            else:
                                logger.warning(
                                    "[Coach] Grounding failed due to disallowed entity cache_key=%s attempt=%d tokens=%s",
                                    cache_key,
                                    attempt,
                                    remaining_entities[:4],
                                )
                                attempt_grounding_reasons.append(
                                    "unsupported_entity_name"
                                )
                                failure_reasons.append("unsupported_entity_name")

                        if not failure_reasons:
                            grounding_validation = validate_response_against_fact_sheet(
                                candidate_payload,
                                fact_sheet,
                            )
                            numeric_sanitized = False
                            if (
                                set(grounding_validation.reasons)
                                == {"unsupported_numeric_claim"}
                                and grounding_validation.unsupported_numeric_tokens
                            ):
                                sanitized_numeric_payload = _sanitize_response_unsupported_numeric_claims(
                                    candidate_payload,
                                    unsupported_tokens=grounding_validation.unsupported_numeric_tokens,
                                )
                                sanitized_numeric_payload = (
                                    _cleanup_response_language_quality(
                                        _normalize_response_team_display(
                                            sanitized_numeric_payload,
                                            evidence=game_evidence,
                                        )
                                    )
                                )
                                sanitized_numeric_payload = _polish_scheduled_partial_response(
                                    sanitized_numeric_payload,
                                    game_status_bucket=game_evidence.game_status_bucket,
                                    grounding_reasons=attempt_grounding_reasons,
                                )
                                sanitized_numeric_payload = (
                                    _sanitize_scheduled_unconfirmed_lineup_entities(
                                        sanitized_numeric_payload,
                                        evidence=game_evidence,
                                        grounding_reasons=attempt_grounding_reasons,
                                        tool_results=tool_results,
                                        resolved_focus=resolved_focus,
                                    )
                                )
                                sanitized_numeric_payload = (
                                    _enforce_completed_result_anchor(
                                        sanitized_numeric_payload,
                                        evidence=game_evidence,
                                        tool_results=tool_results,
                                        resolved_focus=resolved_focus,
                                    )
                                )
                                sanitized_grounding_validation = (
                                    validate_response_against_fact_sheet(
                                        sanitized_numeric_payload,
                                        fact_sheet,
                                    )
                                )
                                if not sanitized_grounding_validation.reasons:
                                    candidate_payload = sanitized_numeric_payload
                                    numeric_sanitized = True
                                    attempt_grounding_reasons.append(
                                        "unsupported_numeric_claim_sanitized"
                                    )
                                    attempt_grounding_warnings.extend(
                                        sanitized_grounding_validation.warnings
                                    )
                                    logger.info(
                                        "[Coach] Sanitized unsupported numeric claims cache_key=%s attempt=%d tokens=%s",
                                        cache_key,
                                        attempt,
                                        grounding_validation.unsupported_numeric_tokens[
                                            :6
                                        ],
                                    )

                            if not numeric_sanitized:
                                attempt_grounding_warnings.extend(
                                    grounding_validation.warnings
                                )
                                attempt_grounding_reasons.extend(
                                    grounding_validation.reasons
                                )

                            if grounding_validation.reasons and not numeric_sanitized:
                                logger.warning(
                                    "[Coach] Grounding failed due to unsupported claim cache_key=%s attempt=%d elapsed_sec=%.2f reasons=%s",
                                    cache_key,
                                    attempt,
                                    perf_counter() - attempt_started,
                                    grounding_validation.reasons,
                                )
                                failure_reasons.extend(
                                    list(grounding_validation.reasons)
                                )
                            else:
                                response_payload = candidate_payload
                                validation_status = "success"
                                generation_mode = "llm_manual"
                                runtime_grounding_reasons = (
                                    _normalize_grounding_reasons(
                                        attempt_grounding_reasons
                                    )
                                )
                                runtime_grounding_warnings = _merge_grounding_warnings(
                                    attempt_grounding_warnings,
                                    runtime_grounding_reasons,
                                )
                                response_json = json.dumps(
                                    response_payload, ensure_ascii=False
                                )
                                logger.info(
                                    "[Coach] LLM attempt success game_id=%s cache_key=%s attempt=%d elapsed_sec=%.2f",
                                    game_evidence.game_id,
                                    cache_key,
                                    attempt,
                                    perf_counter() - attempt_started,
                                )
                                break

                    last_failure_reasons = list(
                        dict.fromkeys(failure_reasons or attempt_grounding_reasons)
                    )
                    runtime_grounding_reasons = _normalize_grounding_reasons(
                        attempt_grounding_reasons + failure_reasons
                    )
                    runtime_grounding_warnings = _merge_grounding_warnings(
                        attempt_grounding_warnings,
                        runtime_grounding_reasons,
                    )
                else:
                    logger.info(
                        "[Coach] Falling back after LLM attempts exhausted game_id=%s cache_key=%s attempts=%d elapsed_sec=%.2f reasons=%s",
                        game_evidence.game_id,
                        cache_key,
                        max_llm_attempts,
                        perf_counter() - total_start,
                        last_failure_reasons,
                    )
                    response_payload = _build_deterministic_coach_response(
                        game_evidence,
                        tool_results,
                        resolved_focus=resolved_focus,
                        grounding_warnings=runtime_grounding_warnings,
                    )
                    response_payload = _postprocess_coach_response_payload(
                        response_payload,
                        evidence=game_evidence,
                        used_evidence=used_evidence,
                        grounding_reasons=runtime_grounding_reasons,
                        grounding_warnings=runtime_grounding_warnings,
                        tool_results=tool_results,
                        resolved_focus=resolved_focus,
                    )

                _ensure_detailed_markdown(
                    response_payload,
                    resolved_focus,
                    grounding_warnings=runtime_grounding_warnings,
                    evidence=game_evidence,
                    tool_results=tool_results,
                )
                _ensure_completed_review_markdown_sections(
                    response_payload,
                    evidence=game_evidence,
                )
                response_json = json.dumps(response_payload, ensure_ascii=False)
                missing_focus_sections = _find_missing_focus_sections(
                    response_payload, response_focus_targets
                )
                if missing_focus_sections:
                    logger.warning(
                        "[Coach] Missing focus sections detected focus=%s missing=%s cache_key=%s",
                        resolved_focus,
                        missing_focus_sections,
                        cache_key,
                    )

                meta_defaults = _build_meta_payload_defaults(
                    generation_mode=generation_mode,
                    data_quality=data_quality,
                    used_evidence=used_evidence,
                    **cache_contract_meta,
                    game_status_bucket=game_evidence.game_status_bucket,
                    grounding_warnings=runtime_grounding_warnings,
                    grounding_reasons=runtime_grounding_reasons,
                    supported_fact_count=fact_sheet.supported_fact_count,
                    dependency_degraded=dependency_degraded,
                    attempt_count=cache_attempt_count,
                    cache_row_missing=cache_state == "ROW_RECREATED",
                    recovered_from_missing_row=cache_state == "ROW_RECREATED",
                    lease_lost=lease_lost_event.is_set(),
                )
                finalize_result = _store_completed_cache(
                    pool=pool,
                    cache_key=cache_key,
                    lease_owner=cache_lease_owner,
                    team_id=home_team_canonical,
                    year=year,
                    prompt_version=COACH_CACHE_PROMPT_VERSION,
                    model_name=coach_model_name,
                    response_payload=response_payload,
                    meta_defaults=meta_defaults,
                )
                finalize_outcome = str(finalize_result.get("outcome") or "")
                if finalize_outcome == "inserted_missing_row":
                    meta_defaults["cache_row_missing"] = True
                    meta_defaults["recovered_from_missing_row"] = True
                elif finalize_outcome == "finalize_conflict":
                    waiting_payload = _cache_status_response(
                        headline=f"{home_name} 분석이 진행 중입니다",
                        coach_note="다른 작업이 캐시 반영을 이어받아 현재 요청은 결과 저장을 중단했습니다. 잠시 후 다시 시도해주세요.",
                        detail=(
                            "## 캐시 준비 중\n\n"
                            "분석 결과 저장 시 충돌이 발생해 현재 요청은 캐시 반영을 중단했습니다."
                        ),
                    )
                    waiting_meta_defaults = _build_meta_payload_defaults(
                        generation_mode=generation_mode,
                        data_quality=data_quality,
                        used_evidence=used_evidence,
                        **cache_contract_meta,
                        game_status_bucket=game_evidence.game_status_bucket,
                        grounding_warnings=runtime_grounding_warnings,
                        grounding_reasons=runtime_grounding_reasons,
                        supported_fact_count=fact_sheet.supported_fact_count,
                        dependency_degraded=dependency_degraded,
                        attempt_count=cache_attempt_count,
                        cache_finalize_conflict=True,
                        lease_lost=lease_lost_event.is_set(),
                    )
                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {
                                "message": "기존 분석 작업 상태를 다시 확인하는 중입니다..."
                            },
                            ensure_ascii=False,
                        ),
                    }
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"delta": json.dumps(waiting_payload, ensure_ascii=False)},
                            ensure_ascii=False,
                        ),
                    }
                    yield {
                        "event": "meta",
                        "data": json.dumps(
                            _build_stream_meta(
                                {
                                    "validation_status": "fallback",
                                    "verified": True,
                                    "fast_path": True,
                                    "cached": False,
                                    "cache_state": "PENDING_WAIT",
                                    "in_progress": True,
                                    **waiting_meta_defaults,
                                }
                            ),
                            ensure_ascii=False,
                        ),
                    }
                    yield {"event": "done", "data": "[DONE]"}
                    return

                meta_payload = _build_stream_meta(
                    {
                        "verified": True,
                        "fast_path": True,
                        "validation_status": validation_status,
                        "cache_state": "COMPLETED",
                        "in_progress": False,
                        "focus_section_missing": bool(missing_focus_sections),
                        "missing_focus_sections": missing_focus_sections,
                        "structured_response": response_payload,
                        **meta_defaults,
                    }
                )
                logger.info(
                    "[Coach] Request completed game_id=%s cache_key=%s validation_status=%s generation_mode=%s cache_state=%s elapsed_sec=%.2f grounding_reasons=%s",
                    game_evidence.game_id,
                    cache_key,
                    validation_status,
                    generation_mode,
                    meta_payload.get("cache_state"),
                    perf_counter() - total_start,
                    runtime_grounding_reasons,
                )

                yield {
                    "event": "message",
                    "data": json.dumps({"delta": response_json}, ensure_ascii=False),
                }
                yield {
                    "event": "meta",
                    "data": json.dumps(meta_payload, ensure_ascii=False),
                }

                yield {"event": "done", "data": "[DONE]"}
            except asyncio.CancelledError:
                logger.warning(
                    "[Coach Streaming Cancelled] game_id=%s resolved_cache_key=%s expected_cache_key=%s prompt_version=%s "
                    "starter_signature=%s lineup_signature=%s attempt_count=%s gate=cancelled",
                    game_evidence.game_id,
                    cache_key,
                    expected_cache_key,
                    COACH_CACHE_PROMPT_VERSION,
                    starter_signature,
                    lineup_signature,
                    cache_attempt_count,
                )
                try:
                    cancelled_store_result = _store_failed_cache(
                        pool=get_connection_pool(),
                        cache_key=cache_key,
                        lease_owner=cache_lease_owner,
                        team_id=home_team_canonical,
                        year=year,
                        prompt_version=COACH_CACHE_PROMPT_VERSION,
                        model_name=coach_model_name,
                        attempt_count=cache_attempt_count,
                        error_code=COACH_STREAM_CANCELLED_ERROR_CODE,
                        error_message="client_stream_cancelled",
                    )
                    if (
                        str(cancelled_store_result.get("outcome") or "")
                        == "inserted_missing_row"
                    ):
                        logger.info(
                            "[Coach] Cancelled stream restored missing cache row for %s",
                            cache_key,
                        )
                except Exception as cancel_exc:  # noqa: BLE001
                    logger.warning(
                        "[Coach] Failed to persist cancelled stream state for %s: %s",
                        cache_key,
                        cancel_exc,
                    )
                raise
            except Exception as e:
                logger.exception(
                    "[Coach Streaming Error] game_id=%s resolved_cache_key=%s expected_cache_key=%s prompt_version=%s "
                    "starter_signature=%s lineup_signature=%s attempt_count=%s gate=exception",
                    game_evidence.game_id,
                    cache_key,
                    expected_cache_key,
                    COACH_CACHE_PROMPT_VERSION,
                    starter_signature,
                    lineup_signature,
                    cache_attempt_count,
                )
                masked_error_code = _cache_error_code_from_exception(e)
                retryable_failure = _is_retryable_cache_error_code(masked_error_code)
                try:
                    fallback_pool = get_connection_pool()
                    failed_store_result = _store_failed_cache(
                        pool=fallback_pool,
                        cache_key=cache_key,
                        lease_owner=cache_lease_owner,
                        team_id=home_team_canonical,
                        year=year,
                        prompt_version=COACH_CACHE_PROMPT_VERSION,
                        model_name=coach_model_name,
                        attempt_count=cache_attempt_count,
                        error_code=masked_error_code,
                        error_message=str(e),
                    )
                except:  # noqa: BLE001
                    failed_store_result = {"outcome": "finalize_conflict"}
                    pass
                failure_meta_defaults = _build_meta_payload_defaults(
                    generation_mode="evidence_fallback",
                    data_quality=_determine_data_quality(game_evidence),
                    used_evidence=game_evidence.used_evidence,
                    **cache_contract_meta,
                    game_status_bucket=game_evidence.game_status_bucket,
                    grounding_warnings=_merge_grounding_warnings(
                        [],
                        evidence_assessment.root_causes,
                    ),
                    grounding_reasons=evidence_assessment.root_causes,
                    supported_fact_count=0,
                    cache_error_code=masked_error_code,
                    retryable_failure=retryable_failure,
                    attempt_count=cache_attempt_count,
                    cache_row_missing=(
                        str(failed_store_result.get("outcome") or "")
                        == "inserted_missing_row"
                    ),
                    recovered_from_missing_row=(
                        str(failed_store_result.get("outcome") or "")
                        == "inserted_missing_row"
                    ),
                    cache_finalize_conflict=(
                        str(failed_store_result.get("outcome") or "")
                        == "finalize_conflict"
                    ),
                    lease_lost=lease_lost_event.is_set(),
                )

                yield {
                    "event": "meta",
                    "data": json.dumps(
                        _build_stream_meta(
                            {
                                "validation_status": "fallback",
                                "fast_path": True,
                                "cached": False,
                                "cache_state": (
                                    "FAILED_RETRY_WAIT"
                                    if retryable_failure
                                    else "FAILED_LOCKED"
                                ),
                                "in_progress": False,
                                "focus_section_missing": False,
                                "missing_focus_sections": [],
                                **failure_meta_defaults,
                            }
                        ),
                        ensure_ascii=False,
                    ),
                }
                yield {
                    "event": "error",
                    "data": json.dumps(
                        _coach_public_error_payload(masked_error_code),
                        ensure_ascii=False,
                    ),
                }
                yield {"event": "done", "data": "[DONE]"}
            finally:
                await _cancel_heartbeat_task(heartbeat_task)

        return EventSourceResponse(
            event_generator(),
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
            ping=sse_ping_seconds,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Coach Router] Error: {e}")
        raise HTTPException(status_code=500, detail=COACH_INTERNAL_ERROR_CODE)


# ============================================================
# Legacy endpoint (기존 방식 유지, 필요 시 사용)
# ============================================================


@router.post("/analyze-legacy")
async def analyze_team_legacy(
    payload: AnalyzeRequest,
    agent: BaseballStatisticsAgent = Depends(get_agent),
    __: None = Depends(rate_limit_coach_dependency),
    _: None = Depends(require_ai_internal_token),
):
    """
    기존 방식의 Coach 분석 (전체 에이전트 파이프라인 사용).
    Fast Path에 문제가 있을 경우 대안으로 사용.
    """
    from sse_starlette.sse import EventSourceResponse

    try:
        primary_team_id = payload.home_team_id or payload.team_id
        if not primary_team_id:
            raise HTTPException(
                status_code=400, detail="home_team_id 또는 team_id가 필요합니다."
            )
        if (
            payload.request_mode == COACH_REQUEST_MODE_AUTO
            and payload.question_override
        ):
            raise HTTPException(
                status_code=400,
                detail="auto_brief 요청에서는 question_override를 사용할 수 없습니다.",
            )

        team_name = agent._convert_team_id_to_name(primary_team_id)
        resolved_focus = normalize_focus(payload.focus)

        if payload.question_override:
            query = payload.question_override
        else:
            query = _build_coach_query(team_name, resolved_focus)

        logger.info(f"[Coach Router Legacy] Analyzing for {team_name}")
        sse_ping_seconds = max(1, int(get_settings().chat_sse_ping_seconds))

        context_data = {"persona": "coach", "team_id": primary_team_id}

        async def event_generator():
            try:
                async for event in agent.process_query_stream(
                    query, context=context_data
                ):
                    if event["type"] == "status":
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                {"message": event["message"]}, ensure_ascii=False
                            ),
                        }
                    elif event["type"] == "tool_start":
                        yield {
                            "event": "tool_start",
                            "data": json.dumps(
                                {"tool": event["tool"]}, ensure_ascii=False
                            ),
                        }
                    elif event["type"] == "tool_result":
                        yield {
                            "event": "tool_result",
                            "data": json.dumps(
                                {
                                    "tool": event["tool"],
                                    "success": event["success"],
                                    "message": event["message"],
                                },
                                ensure_ascii=False,
                            ),
                        }
                    elif event["type"] == "answer_chunk":
                        yield {
                            "event": "message",
                            "data": json.dumps(
                                {"delta": event["content"]}, ensure_ascii=False
                            ),
                        }
                    elif event["type"] == "metadata":
                        meta_payload = {
                            "tool_calls": [
                                tc.to_dict() for tc in event["data"]["tool_calls"]
                            ],
                            "verified": event["data"]["verified"],
                            "data_sources": event["data"]["data_sources"],
                        }
                        yield {
                            "event": "meta",
                            "data": json.dumps(meta_payload, ensure_ascii=False),
                        }

                yield {"event": "done", "data": "[DONE]"}
            except Exception as e:
                logger.error(f"[Coach Legacy Streaming Error] {e}")
                import traceback

                logger.error(traceback.format_exc())
                yield {
                    "event": "error",
                    "data": json.dumps(
                        _coach_public_error_payload(),
                        ensure_ascii=False,
                    ),
                }
                yield {"event": "done", "data": "[DONE]"}

        return EventSourceResponse(
            event_generator(),
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
            ping=sse_ping_seconds,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Coach Router Legacy] Error: {e}")
        raise HTTPException(status_code=500, detail=COACH_INTERNAL_ERROR_CODE)
