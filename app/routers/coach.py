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
import os
import re
import socket
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from datetime import date as date_cls, datetime, time as dt_time, timedelta
from typing import (
    List,
    Optional,
    Dict,
    Any,
    Literal,
    Set,
    AsyncIterator,
    Tuple,
    Sequence,
    Coroutine,
)
from zoneinfo import ZoneInfo
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, model_validator

import psycopg
from psycopg.rows import dict_row

from psycopg_pool import AsyncConnectionPool

from ..deps import (
    get_agent,
    get_connection_pool,
    get_coach_llm_generator,
    require_ai_internal_token,
    resolve_coach_openrouter_models,
)
from ..config import get_settings
from ..agents.baseball_agent import BaseballStatisticsAgent
from ..contracts.stream_requests import CoachAnalyzeRequest
from ..core.coach_grounding import (
    CoachFactSheet,
    collect_numeric_tokens,
    format_coach_fact_sheet,
    validate_response_against_fact_sheet,
    extend_numeric_tokens,
    response_is_semantically_empty,
    _collect_response_text_segments,
)
from ..core.prompts import (
    COACH_PROMPT_V2,
    COACH_PROMPT_V2_DYNAMIC_TEMPLATE,
    COACH_PROMPT_V2_STATIC,
)
from ..core.coach_validator import (
    MAX_KEY_METRIC_VALUE_LENGTH,
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
from ..observability.metrics import (
    AI_COACH_DYNAMIC_PROMPT_CHARS,
    AI_COACH_LLM_SKIP_TOTAL,
    AI_COACH_PAYLOAD_COMPRESSION_TOTAL,
    AI_COACH_REQUEST_TOTAL,
)
from ..schemas.coach_tool_payload import CoachTeamPayload
from ..streaming.versioned_sse import (
    negotiate_event_version,
    versioned_event_source,
)
from ..tools.database_query import DatabaseQueryTool
from ..tools.team_code_resolver import CANONICAL_CODES, TeamCodeResolver

logger = logging.getLogger(__name__)
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _read_flag_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in TRUTHY_ENV_VALUES


def _read_int_env(name: str, default: str) -> int:
    raw = os.getenv(name, default).strip()
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _coach_stream_preview_enabled() -> bool:
    return _read_flag_env("COACH_STREAM_PREVIEW_ENABLED", "0")


def _coach_prompt_cache_enabled() -> bool:
    return _read_flag_env("COACH_PROMPT_CACHE_ENABLED", "1")


def _coach_prompt_cache_min_static_chars() -> int:
    return _read_int_env("COACH_PROMPT_CACHE_MIN_STATIC_CHARS", "3000")


def _coach_payload_compression_enabled() -> bool:
    """``_execute_coach_tools_parallel`` 결과를 ``CoachTeamPayload``로 압축할지.

    기본 활성화. 회귀 발견 시 ``COACH_PAYLOAD_COMPRESSION_ENABLED=0``로 옵트아웃.
    """
    return _read_flag_env("COACH_PAYLOAD_COMPRESSION_ENABLED", "1")


def _coach_payload_top_n() -> int:
    return _read_int_env("COACH_PAYLOAD_TOP_N", "3")


def _coach_fast_path_manual_recent_form_enabled() -> bool:
    return _read_flag_env("COACH_FAST_PATH_MANUAL_RECENT_FORM", "0")


COACH_YEAR_MIN = 1982
MAX_COACH_FOCUS_ITEMS = 6
MAX_COACH_QUESTION_OVERRIDE_LENGTH = 2000
PENDING_STALE_SECONDS = 90
PENDING_WAIT_TIMEOUT_SECONDS = 45
PENDING_WAIT_POLL_MS = 1000
# Coach 도구 병렬 실행 시 동시 DB 커넥션 체크아웃 상한.
# get_team_data 병렬화로 1요청당 최대 ~10개 커넥션을 동시 사용할 수 있어
# DB_POOL_MAX_SIZE(30) 고갈을 막기 위해 전역 fan-out을 제한한다.
COACH_DB_FANOUT_MAX = int(os.getenv("COACH_DB_FANOUT_MAX", "12"))
FAILED_RETRY_AFTER_SECONDS = 30
COACH_CACHE_HEARTBEAT_INTERVAL_SECONDS = 5
COACH_CACHE_LEASE_STALE_SECONDS = 90
COACH_CACHE_MAX_RETRYABLE_ATTEMPTS = 3
COACH_CACHE_CLAIM_DB_ATTEMPTS = 2
COACH_PENDING_RECHECK_AFTER_SECONDS = 120
COACH_AUTO_BRIEF_BACKGROUND_MAX_TASKS = max(
    1,
    int(os.getenv("COACH_AUTO_BRIEF_BACKGROUND_MAX_TASKS", "5")),
)
COACH_AUTO_BRIEF_BACKGROUND_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("COACH_AUTO_BRIEF_BACKGROUND_TIMEOUT_SECONDS", "120")),
)
_AUTO_BRIEF_BACKGROUND_TASKS: Set[asyncio.Task[None]] = set()
VOLATILE_CACHE_TTL_SECONDS = 300
COMPLETED_CACHE_TTL_SECONDS = 86400
COACH_LLM_STATUS_HEARTBEAT_SECONDS = 3.0
COACH_STREAM_PREVIEW_ENABLED = _coach_stream_preview_enabled()
COACH_PROMPT_CACHE_ENABLED = _coach_prompt_cache_enabled()
COACH_PROMPT_CACHE_MIN_STATIC_CHARS = _coach_prompt_cache_min_static_chars()
COACH_FAST_PATH_MANUAL_RECENT_FORM = _coach_fast_path_manual_recent_form_enabled()
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
COACH_LLM_TIMEOUT_ERROR_CODE = "coach_llm_timeout"
COACH_OCI_MAPPING_DEGRADED_ERROR_CODE = "oci_mapping_degraded"
COACH_UNSUPPORTED_ENTITY_ERROR_CODE = "unsupported_entity_name"
COACH_UNSUPPORTED_CLAIM_ERROR_CODE = "grounding_validation_failed"
COACH_NON_RETRYABLE_ERROR_CODE = "non_retryable_internal_error"
MANUAL_BASEBALL_DATA_REQUIRED_CODE = "MANUAL_BASEBALL_DATA_REQUIRED"
MANUAL_BASEBALL_DATA_REQUIRED_MESSAGE = "야구 데이터 준비가 필요합니다. trusted baseball data sync 반영 후 다시 확인할 수 있습니다."
BASEBALL_DATA_SYNC_REQUIRED_CODE = "BASEBALL_DATA_SYNC_REQUIRED"
BASEBALL_DATA_SYNC_EXTERNAL_SOURCE = "trusted_baseball_data_project"
BASEBALL_DATA_SYNC_HANDOFF = "external_trusted_baseball_data_sync"
BASEBALL_DATA_SYNC_REQUIRED_FIELDS_BY_MISSING_KEY: Dict[str, Sequence[str]] = {
    "game_id": ("game.game_id",),
    "game_date": ("game.game_date",),
    "season_league_context": (
        "game.season_id",
        "game.season_year",
        "game.league_type_code",
    ),
    "game_status": ("game.game_status",),
    "final_score": ("game.home_score", "game.away_score"),
    "starters": ("game.home_pitcher", "game.away_pitcher"),
    "lineup": (
        "game_lineups.team_code",
        "game_lineups.batting_order",
        "game_lineups.player_name",
    ),
}
COACH_STARTER_ANNOUNCEMENT_PENDING_CODE = "starter_announcement_pending"
KBO_TIMEZONE = ZoneInfo("Asia/Seoul")
KBO_STARTER_ANNOUNCEMENT_HOUR = 18
COACH_CACHE_ERROR_MESSAGES: Dict[str, str] = {
    COACH_INTERNAL_ERROR_CODE: "분석 시스템 내부 오류로 캐시를 생성하지 못했습니다.",
    COACH_TOOL_FETCH_FAILED_CODE: "분석에 필요한 팀 데이터를 조회하지 못했습니다.",
    COACH_DATA_INSUFFICIENT_CODE: "분석에 필요한 데이터가 충분하지 않습니다.",
    COACH_VALIDATION_FAILED_CODE: "분석 응답 검증에 실패해 결과를 저장하지 못했습니다.",
    COACH_EMPTY_RESPONSE_ERROR_CODE: "LLM 응답이 비어 자동 복구 대기 중입니다.",
    COACH_NO_JSON_FOUND_ERROR_CODE: "LLM 응답에서 JSON을 추출하지 못해 자동 복구 대기 중입니다.",
    COACH_JSON_DECODE_ERROR_CODE: "LLM 응답 JSON 파싱에 실패해 자동 복구 대기 중입니다.",
    COACH_STREAM_CANCELLED_ERROR_CODE: "분석 스트림이 중단되어 자동 복구 대기 중입니다.",
    COACH_LLM_TIMEOUT_ERROR_CODE: "LLM 응답 시간이 초과되어 보수 응답 또는 자동 복구가 필요합니다.",
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
    COACH_LLM_TIMEOUT_ERROR_CODE,
    COACH_OCI_MAPPING_DEGRADED_ERROR_CODE,
}
FOCUS_SECTION_HEADERS: Dict[str, str] = {
    "recent_form": "## 최근 전력",
    "bullpen": "## 불펜 상태",
    "starter": "## 선발 투수",
    "matchup": "## 상대 전적",
    "batting": "## 득점 연결력",
}
COMPLETED_REVIEW_FOCUS_SECTION_HEADERS: Dict[str, str] = {
    **FOCUS_SECTION_HEADERS,
    "batting": "## 타격 생산성",
}


def _focus_section_headers_for_status(
    game_status_bucket: Optional[str] = None,
) -> Dict[str, str]:
    if _normalize_game_status_bucket(game_status_bucket) == "COMPLETED":
        return COMPLETED_REVIEW_FOCUS_SECTION_HEADERS
    return FOCUS_SECTION_HEADERS


def _focus_section_header(
    focus: str,
    game_status_bucket: Optional[str] = None,
) -> Optional[str]:
    return _focus_section_headers_for_status(game_status_bucket).get(focus)


def _all_focus_section_headers() -> Tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [
                *FOCUS_SECTION_HEADERS.values(),
                *COMPLETED_REVIEW_FOCUS_SECTION_HEADERS.values(),
            ]
        )
    )


FOCUS_SECTION_METRIC_LABELS: Dict[str, Tuple[str, ...]] = {
    "recent_form": ("최근 흐름", "폼 진단", "득실점 마진"),
    "bullpen": ("불펜 비중", "불펜 운용", "불펜 데이터"),
    "starter": ("발표 선발", "선발 투수"),
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
    "batting": "득점 연결 근거가 제한적이어서 출루와 장타 흐름 비교는 보수적으로 봐야 합니다.",
}
COMPLETED_FOCUS_FALLBACK_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "recent_form": (
        "최근 경기 표본이 제한적이라 흐름 평가는 결과 맥락과 함께 봐야 합니다.",
        "최근 경기 근거가 제한적이라 흐름 평가는 보수적으로 봐야 합니다.",
        "최근 표본 부족으로 단기 흐름은 보조 지표로만 봐야 합니다.",
    ),
    "bullpen": (
        "불펜 운용 근거가 제한적이라 후반 평가는 결과 흐름 중심으로 봐야 합니다.",
        "불펜 근거가 부족해 중후반 운영 평가는 확인된 결과 중심으로 제한해야 합니다.",
        "불펜 운용 표본이 적어 후반 대응력 비교는 보수적으로 봐야 합니다.",
    ),
    "starter": (
        "선발 기록이 확인되지 않아 선발 맞대결 평가는 제한됩니다.",
        "선발 데이터가 비어 있어 초반 투수 비교는 보수적으로 봐야 합니다.",
        "공식 선발 기록이 부족해 선발 맞대결은 확인된 결과 중심으로만 봐야 합니다.",
    ),
    "matchup": (
        "상대 전적 근거가 충분하지 않아 직접 비교는 보수적으로 해석해야 합니다.",
        "상대 전적 표본이 부족해 팀 간 직접 비교는 제한적으로 봐야 합니다.",
        "맞대결 근거가 부족해 전적 비교보다 실제 득점 흐름을 우선 봐야 합니다.",
    ),
    "batting": (
        "확인 가능한 득점 연결 근거가 제한적입니다.",
        "출루와 장타 흐름 표본이 부족해 직접 비교는 보수적으로 봐야 합니다.",
        "공격 지표 근거가 제한적이라 득점 연결 평가는 확인된 결과 중심으로 봐야 합니다.",
    ),
    "matchup_warning": (
        "맞대결 표본이 제한적입니다.",
        "맞대결 표본이 부족해 직접 비교는 제한적입니다.",
        "상대 전적 근거가 적어 결과 중심으로 해석해야 합니다.",
    ),
    "recent_form_warning": (
        "최근 경기 표본이 제한적이라 흐름 평가는 결과 맥락과 함께 봐야 합니다.",
        "최근 경기 근거가 제한적이라 흐름 평가는 보수적으로 봐야 합니다.",
        "최근 표본 부족으로 단기 흐름은 보조 지표로만 봐야 합니다.",
    ),
    "bullpen_warning": (
        "불펜 운용 데이터가 부족합니다.",
        "불펜 운용 표본이 제한적입니다.",
        "불펜 근거가 부족해 후반 평가는 보수적입니다.",
    ),
    "batting_warning": (
        "타격 지표 표본이 부족합니다.",
        "득점 연결 근거가 제한적입니다.",
        "공격 지표 표본이 부족해 직접 비교는 제한적입니다.",
    ),
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
    "deterministic_review",
    "deterministic_preview",
    "llm_manual",
    "evidence_fallback",
}
COACH_ANALYSIS_TYPE_REVIEW = "game_review"
COACH_ANALYSIS_TYPE_PREVIEW = "game_preview"
COACH_ANALYSIS_TYPES = {COACH_ANALYSIS_TYPE_REVIEW, COACH_ANALYSIS_TYPE_PREVIEW}
COACH_META_DATA_QUALITIES = {"grounded", "partial", "insufficient"}
COACH_DATA_QUALITY_RANK = {
    "insufficient": 0,
    "partial": 1,
    "grounded": 2,
}
# 근거 가중치: recent-form summary 는 고신호(2), advanced 지표/구조 신호는 1.
# grounded 승격은 단순 tool-fetch 개수(>=2)가 아니라 가중 합산 점수로 판정해
# "실데이터 기반" 배지의 과신을 줄인다.
COACH_EVIDENCE_WEIGHTS = {
    "home_summary": 2,
    "away_summary": 2,
    "home_advanced": 1,
    "away_advanced": 1,
    "starters_pair": 1,
    "lineup": 1,
    "summary_items": 1,
}
COACH_PARTIAL_MIN_SCORE = 2
COACH_GROUNDED_MIN_SCORE = 4
# LLM 호출(manual)을 정당화하는 최소 실데이터 fact 수. 미만이면 deterministic 으로 단락.
COACH_MIN_LLM_FACT_COUNT = 2
# supported_fact_count 에서 제외하는 보일러플레이트 라벨(항상 존재 → 신뢰 신호 부풀림 방지).
COACH_CONTEXTUAL_FACT_LABELS = {
    "매치업",
    "경기 날짜",
    "경기 상태",
    "리그 구분",
    "구장",
    "시작 시각",
    "날씨",
}
# sanitize 로도 못 고친 환각성 grounding 실패 → deterministic fallback 시
# "grounded(실데이터 기반)" 배지는 과신이므로 data_quality 를 한 단계 강등한다.
# (*_sanitized 변형은 근거 내로 정리된 것이므로 강등 대상에서 제외)
COACH_GROUNDING_HARD_FAILURE_REASONS = {
    "unsupported_entity_name",
    "unsupported_numeric_claim",
    "unconfirmed_starter_claim",
    "unconfirmed_lineup_claim",
    "unconfirmed_series_claim",
    "empty_response",
    "llm_parse_failed",
}
COACH_GROUNDING_REASON_MESSAGES: Dict[str, str] = {
    "missing_game_context": "경기 기본 맥락이 충분하지 않아 보수적으로 해석합니다.",
    "missing_starters": "선발 정보가 완전하지 않아 선발 관련 표현을 제한합니다.",
    COACH_STARTER_ANNOUNCEMENT_PENDING_CODE: "공식 선발 발표 전이라 선발 맞대결은 발표 이후 보강합니다.",
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
    "scheduled_output_guard_fallback": "예정 경기 응답에 리뷰 또는 확정 표현이 감지되어 근거 기반 보수 생성으로 전환했습니다.",
    "llm_parse_failed": "LLM 응답 형식이 불안정해 보수 요약으로 전환했습니다.",
    COACH_OCI_MAPPING_DEGRADED_ERROR_CODE: "팀 매핑 조회가 일시적으로 불안정해 재연결 또는 마지막 정상 스냅샷으로 이어갔습니다.",
}
COACH_EVIDENCE_ROOT_CAUSE_ORDER = (
    "missing_game_context",
    COACH_STARTER_ANNOUNCEMENT_PENDING_CODE,
    "missing_starters",
    "missing_lineups",
    "missing_summary",
    "missing_metadata",
    "missing_series_context",
)
SCHEDULED_PARTIAL_GROUNDING_REASONS = (
    COACH_STARTER_ANNOUNCEMENT_PENDING_CODE,
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
    llm_skip_reason = str(payload.get("llm_skip_reason") or "").strip()
    if llm_skip_reason:
        AI_COACH_LLM_SKIP_TOTAL.labels(
            reason=llm_skip_reason,
            request_mode=str(payload.get("request_mode") or "unknown"),
            analysis_type=str(payload.get("analysis_type") or "unknown"),
        ).inc()
    logger.info(
        "[Coach Meta] request_mode=%s game_id=%s cache_state=%s validation_status=%s "
        "generation_mode=%s llm_skip_reason=%s cached=%s in_progress=%s data_quality=%s supported_fact_count=%s "
        "grounding_reasons=%s grounding_warnings=%s used_evidence=%s resolved_focus=%s "
        "missing_focus_sections=%s",
        payload.get("request_mode"),
        game_id,
        payload.get("cache_state"),
        payload.get("validation_status"),
        payload.get("generation_mode"),
        payload.get("llm_skip_reason"),
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


def _count_substantive_facts(fact_lines: List[str]) -> int:
    """보일러플레이트(구장/날씨/날짜 등)를 제외한 실 야구 근거 fact 수.

    supported_fact_count(=프론트 'N개 실데이터 근거' 신호)가 항상 존재하는 맥락 라벨로
    부풀지 않도록 substantive 라인만 센다.
    """
    return sum(
        1
        for line in fact_lines
        if line.split(":", 1)[0].strip() not in COACH_CONTEXTUAL_FACT_LABELS
    )


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


def _append_team_comparison_fact_lines(
    fact_lines: List[str],
    numeric_tokens: Set[str],
    tool_results: Dict[str, Any],
    home_name: str,
    away_name: str,
) -> None:
    """홈 vs 원정 핵심 지표를 한 줄 대비 형식으로 추가합니다.

    LLM이 두 팀 수치를 따로 읽고 요약하는 대신 직접 비교하도록
    "홈 X vs 원정 Y (우위 ±Z)" 형태의 fact line을 생성합니다.
    """
    home_data = tool_results.get("home", {}) or {}
    away_data = tool_results.get("away", {}) or {}
    home_adv = _advanced_metrics(home_data)
    away_adv = _advanced_metrics(away_data)
    home_recent = _recent_summary(home_data)
    away_recent = _recent_summary(away_data)

    # OPS 대비
    home_ops_raw = home_adv.get("metrics", {}).get("batting", {}).get("ops")
    away_ops_raw = away_adv.get("metrics", {}).get("batting", {}).get("ops")
    if home_ops_raw is not None and away_ops_raw is not None:
        try:
            h = float(home_ops_raw)
            a = float(away_ops_raw)
            diff = h - a
            edge_label = (
                f"{home_name} +{diff:.3f}"
                if diff >= 0
                else f"{away_name} +{abs(diff):.3f}"
            )
            fact = _build_fact_line(
                "OPS 대비",
                f"{home_name} {h:.3f} vs {away_name} {a:.3f} ({edge_label})",
            )
            if fact:
                fact_lines.append(fact)
                extend_numeric_tokens([fact], numeric_tokens)
        except (TypeError, ValueError):
            pass

    # 최근 흐름 대비
    if home_recent and away_recent:
        hw = _safe_int_value(home_recent.get("wins"))
        hl = _safe_int_value(home_recent.get("losses"))
        hd = _safe_int_value(home_recent.get("run_diff"))
        aw = _safe_int_value(away_recent.get("wins"))
        al = _safe_int_value(away_recent.get("losses"))
        ad = _safe_int_value(away_recent.get("run_diff"))
        h_str = f"{hw}승{hl}패 득실{'+' if hd >= 0 else ''}{hd}"
        a_str = f"{aw}승{al}패 득실{'+' if ad >= 0 else ''}{ad}"
        fact = _build_fact_line(
            "최근 흐름 대비",
            f"{home_name} {h_str} vs {away_name} {a_str}",
        )
        if fact:
            fact_lines.append(fact)
            extend_numeric_tokens([fact], numeric_tokens)

    # 불펜 비중 대비
    home_bs = home_adv.get("fatigue_index", {}).get("bullpen_share")
    away_bs = away_adv.get("fatigue_index", {}).get("bullpen_share")
    if home_bs is not None and away_bs is not None:
        league_avg = home_adv.get("league_averages", {}).get("bullpen_share")
        avg_str = f" 리그평균 {league_avg}%" if league_avg is not None else ""
        fact = _build_fact_line(
            "불펜 비중 대비",
            f"{home_name} {home_bs}% vs {away_name} {away_bs}%{avg_str}",
        )
        if fact:
            fact_lines.append(fact)
            extend_numeric_tokens([fact], numeric_tokens)


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
        if COACH_STARTER_ANNOUNCEMENT_PENDING_CODE in reasons:
            caveat_lines.append(
                "공식 선발 발표 전입니다. 선발 맞대결은 발표 이후 보강해야 합니다."
            )
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

    # 홈 vs 원정 핵심 지표 직접 대비 (LLM 비교 분석 유도)
    _append_team_comparison_fact_lines(
        fact_lines,
        numeric_tokens,
        tool_results,
        evidence.home_team_name,
        evidence.away_team_name,
    )
    # 팀별 상세 지표 (선수 폼, 주요 타자/투수)
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
        or code == COACH_STARTER_ANNOUNCEMENT_PENDING_CODE
    )
    caveat_lines = list(dict.fromkeys(line for line in caveat_lines if line))
    extend_numeric_tokens(fact_lines, numeric_tokens)

    return CoachFactSheet(
        fact_lines=fact_lines,
        caveat_lines=caveat_lines,
        allowed_entity_names={name for name in allowed_names if name},
        allowed_numeric_tokens=numeric_tokens,
        supported_fact_count=_count_substantive_facts(fact_lines),
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
    if normalized in {
        "COMPLETED",
        "FINAL",
        "FINISHED",
        "DONE",
        "END",
        "E",
        "F",
        "DRAW",
        "TIE",
        "D",
    }:
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


UNANNOUNCED_PITCHER_LABELS = {
    "발표 전",
    "발표전",
    "미정",
    "선발 발표 전",
    "선발발표전",
    "선발 미정",
    "선발미정",
    "선발 미발표",
    "선발미발표",
    "TBD",
    "N/A",
    "-",
}


def _normalize_pitcher_name_token(value: Optional[str]) -> Optional[str]:
    normalized = _normalize_name_token(value)
    if not normalized:
        return None
    normalized_key = normalized.upper()
    compact_key = "".join(normalized_key.split())
    if (
        normalized_key in UNANNOUNCED_PITCHER_LABELS
        or compact_key in UNANNOUNCED_PITCHER_LABELS
    ):
        return None
    return normalized


def _is_scheduled_partial_context(
    game_status_bucket: str,
    grounding_reasons: List[str],
) -> bool:
    if game_status_bucket != "SCHEDULED":
        return False
    return any(
        reason in grounding_reasons for reason in SCHEDULED_PARTIAL_GROUNDING_REASONS
    )


def _parse_game_date_for_announcement(value: Optional[str]) -> Optional[date_cls]:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    try:
        return datetime.strptime(normalized[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _coach_now_kst() -> datetime:
    override = (os.getenv("COACH_AUDIT_NOW_KST", "") or "").strip()
    if override:
        parsed = datetime.fromisoformat(override)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=KBO_TIMEZONE)
        return parsed.astimezone(KBO_TIMEZONE)
    return datetime.now(KBO_TIMEZONE)


def _starter_announcement_due_at_kst(
    game_date: Optional[str],
) -> Optional[datetime]:
    parsed_date = _parse_game_date_for_announcement(game_date)
    if parsed_date is None:
        return None
    return datetime.combine(
        parsed_date - timedelta(days=1),
        dt_time(hour=KBO_STARTER_ANNOUNCEMENT_HOUR),
        tzinfo=KBO_TIMEZONE,
    )


def _is_starter_announcement_pending(
    evidence: GameEvidence,
    *,
    now_kst: Optional[datetime] = None,
) -> bool:
    if _normalize_game_status_bucket(evidence.game_status_bucket) != "SCHEDULED":
        return False
    due_at = _starter_announcement_due_at_kst(evidence.game_date)
    if due_at is None:
        return False
    current = now_kst or _coach_now_kst()
    if current.tzinfo is None:
        current = current.replace(tzinfo=KBO_TIMEZONE)
    return current.astimezone(KBO_TIMEZONE) < due_at


def _starter_missing_reason_present(reasons: Optional[List[str]]) -> bool:
    values = set(reasons or [])
    return bool(
        "missing_starters" in values
        or COACH_STARTER_ANNOUNCEMENT_PENDING_CODE in values
    )


def _clean_summary_text(value: Optional[str]) -> Optional[str]:
    text = " ".join(str(value or "").split()).strip()
    return text or None


def _looks_like_structured_json_text(value: Optional[str]) -> bool:
    text = _clean_summary_text(value)
    if not text or text[0] not in {"{", "["}:
        return False
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(parsed, (dict, list))


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
    if _looks_like_structured_json_text(detail_text):
        return None
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


def _format_game_summary_items(
    rows: List[Dict[str, Any]],
    *,
    limit: int = 6,
) -> List[str]:
    items: List[str] = []
    for row in rows or []:
        text = _format_game_summary_item(
            row.get("summary_type"),
            row.get("player_name"),
            row.get("detail_text"),
        )
        if not text:
            continue
        items.append(text)
        if len(items) >= limit:
            break
    return items


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


def _compact_key_metric_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "데이터 확인"
    return _truncate_text_naturally(text, max_length=MAX_KEY_METRIC_VALUE_LENGTH)


def _finalize_deterministic_metrics(
    metrics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    for metric in metrics:
        if isinstance(metric, dict):
            metric["value"] = _compact_key_metric_value(metric.get("value"))
    return metrics[:5]


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


def _normalize_analysis_type(value: Optional[str]) -> Optional[str]:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in COACH_ANALYSIS_TYPES else None


def _resolve_analysis_type(
    payload: "AnalyzeRequest",
    evidence: GameEvidence,
) -> str:
    status_based = (
        COACH_ANALYSIS_TYPE_REVIEW
        if _is_completed_review(evidence)
        else COACH_ANALYSIS_TYPE_PREVIEW
    )
    requested = _normalize_analysis_type(getattr(payload, "analysis_type", None))
    if requested and requested == status_based:
        return requested
    return status_based


def _generation_mode_for_analysis_type(
    *,
    analysis_type: str,
    request_mode: str,
    fallback: str = "evidence_fallback",
) -> str:
    if _normalize_analysis_type(analysis_type) == COACH_ANALYSIS_TYPE_REVIEW:
        return "deterministic_review"
    if _normalize_analysis_type(analysis_type) == COACH_ANALYSIS_TYPE_PREVIEW:
        return "deterministic_preview"
    if request_mode == COACH_REQUEST_MODE_AUTO:
        return "deterministic_auto"
    return fallback


def _should_include_auto_brief_clutch(evidence: GameEvidence) -> bool:
    """auto_brief cache generation should include WPA moments for completed games."""
    return _is_completed_review(evidence)


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


def assess_game_evidence(
    evidence: GameEvidence,
    analysis_type: str = COACH_ANALYSIS_TYPE_REVIEW,
) -> GameEvidenceAssessment:
    normalized_analysis_type = (
        _normalize_analysis_type(analysis_type) or COACH_ANALYSIS_TYPE_REVIEW
    )
    used_evidence = _default_used_evidence(evidence)
    home_pitcher_present = bool(evidence.home_pitcher)
    away_pitcher_present = bool(evidence.away_pitcher)
    lineup_announced = _has_confirmed_lineup_details(evidence)
    summary_present = bool(evidence.summary_items)
    metadata_present = bool(
        evidence.stadium_name or evidence.start_time or evidence.weather
    )
    series_required = evidence.stage_label not in {"REGULAR", "PRE", "UNKNOWN"}
    # 부분 시리즈(DB 이력 부족으로 축약)는 '확보'로 보지 않는다 → 포스트시즌 과신 방지.
    series_context_available = bool(evidence.series_state) and not (
        evidence.series_state is not None and evidence.series_state.series_state_partial
    )
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
        if _is_starter_announcement_pending(evidence):
            root_causes.append(COACH_STARTER_ANNOUNCEMENT_PENDING_CODE)
        else:
            root_causes.append("missing_starters")
    if not lineup_announced:
        root_causes.append("missing_lineups")
    if normalized_analysis_type == COACH_ANALYSIS_TYPE_REVIEW and not summary_present:
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


def _grounding_evidence_score(
    assessment: GameEvidenceAssessment,
    tool_results: Dict[str, Any],
    evidence: GameEvidence,
) -> int:
    """근거 풍부도를 가중 합산한 점수.

    tool 로 실제 조회된 통계(summary=recent form, advanced)와 구조 신호(양 선발 확정,
    라인업 확정, 경기 요약)를 가중합한다. away 팀이 없으면 away 가중은 제외한다.
    """
    has_away = bool(evidence.away_team_code)
    score = 0
    if tool_results.get("home", {}).get("summary", {}).get("found"):
        score += COACH_EVIDENCE_WEIGHTS["home_summary"]
    if tool_results.get("home", {}).get("advanced", {}).get("found"):
        score += COACH_EVIDENCE_WEIGHTS["home_advanced"]
    if has_away:
        if tool_results.get("away", {}).get("summary", {}).get("found"):
            score += COACH_EVIDENCE_WEIGHTS["away_summary"]
        if tool_results.get("away", {}).get("advanced", {}).get("found"):
            score += COACH_EVIDENCE_WEIGHTS["away_advanced"]
    if assessment.home_pitcher_present and assessment.away_pitcher_present:
        score += COACH_EVIDENCE_WEIGHTS["starters_pair"]
    if assessment.lineup_announced:
        score += COACH_EVIDENCE_WEIGHTS["lineup"]
    if assessment.summary_present:
        score += COACH_EVIDENCE_WEIGHTS["summary_items"]
    return score


def _has_balanced_tool_coverage(
    tool_results: Dict[str, Any],
    evidence: GameEvidence,
) -> bool:
    """grounded 승격용 균형 조건: away 가 있으면 양 팀 모두 최소 1개 tool-found."""
    home_found = bool(
        tool_results.get("home", {}).get("summary", {}).get("found")
        or tool_results.get("home", {}).get("advanced", {}).get("found")
    )
    if not evidence.away_team_code:
        return home_found
    away_found = bool(
        tool_results.get("away", {}).get("summary", {}).get("found")
        or tool_results.get("away", {}).get("advanced", {}).get("found")
    )
    return home_found and away_found


def _determine_data_quality(
    evidence: GameEvidence,
    tool_results: Optional[Dict[str, Any]] = None,
    analysis_type: str = COACH_ANALYSIS_TYPE_REVIEW,
) -> str:
    assessment = assess_game_evidence(evidence, analysis_type=analysis_type)
    if tool_results is None:
        return assessment.expected_data_quality

    score = _grounding_evidence_score(assessment, tool_results, evidence)

    if assessment.expected_data_quality == "insufficient":
        if score >= COACH_PARTIAL_MIN_SCORE and evidence.game_id:
            return "partial"
        return "insufficient"
    if assessment.expected_data_quality == "partial":
        return "partial"
    if _is_completed_review(evidence) and not tool_results.get(
        "clutch_moments", {}
    ).get("found"):
        return "partial"
    # grounded 승격: 가중 점수 + 균형 조건(양 팀 커버리지) 모두 충족해야 함.
    if (
        evidence.game_id
        and score >= COACH_GROUNDED_MIN_SCORE
        and _has_balanced_tool_coverage(tool_results, evidence)
    ):
        return "grounded"
    return "partial"


def _downgrade_data_quality_after_failed_grounding(
    data_quality: str,
    failure_reasons: Optional[List[str]] = None,
) -> str:
    """환각성 grounding 실패로 deterministic fallback 된 경우 grounded→partial 강등.

    grounded 만 한 단계 강등하고 partial/insufficient 는 그대로 둔다(바닥 partial).
    sanitize 로 정리된(*_sanitized) 사유는 강등 트리거가 아니다.
    """
    if (
        COACH_DATA_QUALITY_RANK.get(data_quality, 1)
        < COACH_DATA_QUALITY_RANK["grounded"]
    ):
        return data_quality
    if any(
        reason in COACH_GROUNDING_HARD_FAILURE_REASONS
        for reason in (failure_reasons or [])
    ):
        return "partial"
    return data_quality


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


def _baseball_data_sync_required_fields(item: Dict[str, str]) -> List[str]:
    key = str(item.get("key") or "").strip()
    mapped = BASEBALL_DATA_SYNC_REQUIRED_FIELDS_BY_MISSING_KEY.get(key)
    if mapped:
        return [str(field) for field in mapped]
    expected_format = str(item.get("expected_format") or "").strip()
    if expected_format:
        return [part.strip() for part in expected_format.split(",") if part.strip()]
    return [key] if key else []


def _build_baseball_data_sync_request(
    *,
    request_game_id: Optional[str],
    request_game_date: Optional[date_cls],
    season_year: Optional[int],
    home_team_id: Optional[str],
    away_team_id: Optional[str],
    stage_label: str,
    analysis_type: str,
    missing_items: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    request_identity = (
        request_game_id
        or (request_game_date.isoformat() if request_game_date is not None else "")
        or "unknown"
    )
    return {
        "code": BASEBALL_DATA_SYNC_REQUIRED_CODE,
        "requestId": f"coach:{request_identity}:{analysis_type}",
        "consumer": "ai_coach",
        "scope": "coach.analyze",
        "analysisType": analysis_type,
        "targetSource": BASEBALL_DATA_SYNC_EXTERNAL_SOURCE,
        "handoff": BASEBALL_DATA_SYNC_HANDOFF,
        "blocking": True,
        "entity": {
            "gameId": request_game_id,
            "gameDate": (
                request_game_date.isoformat() if request_game_date is not None else None
            ),
            "seasonYear": season_year,
            "homeTeamId": home_team_id,
            "awayTeamId": away_team_id,
            "stage": stage_label,
        },
        "missingItems": [
            {
                "key": str(item.get("key") or ""),
                "label": str(item.get("label") or ""),
                "reason": str(item.get("reason") or ""),
                "expectedFormat": str(item.get("expected_format") or ""),
                "requiredFields": _baseball_data_sync_required_fields(item),
            }
            for item in missing_items
        ],
    }


def _parse_iso_date(value: Optional[str]) -> Optional[date_cls]:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    try:
        return datetime.strptime(normalized[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


async def _lookup_stage_start_date(
    pool: AsyncConnectionPool,
    *,
    season_year: int,
    league_type_code: int,
) -> Optional[date_cls]:
    try:
        async with pool.connection() as conn:
            cursor = conn.cursor(row_factory=dict_row)
            row = await (
                await cursor.execute(
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
                )
            ).fetchone()
            if row and row.get("start_date") is not None:
                start_date = row["start_date"]
                if hasattr(start_date, "date"):
                    return start_date.date()
                return start_date

            fallback_row = await (
                await cursor.execute(
                    """
                SELECT MIN(g.game_date) AS start_date
                FROM game g
                LEFT JOIN kbo_seasons ks ON g.season_id = ks.season_id
                WHERE COALESCE(ks.season_year, EXTRACT(YEAR FROM g.game_date)::int) = %s
                  AND COALESCE(ks.league_type_code, 0) = %s
                """,
                    (season_year, league_type_code),
                )
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


async def _has_stage_context_mismatch(
    pool: AsyncConnectionPool,
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

    stage_start = await _lookup_stage_start_date(
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


async def _build_manual_data_request(
    pool: AsyncConnectionPool,
    payload: "AnalyzeRequest",
    evidence: GameEvidence,
    assessment: GameEvidenceAssessment,
    analysis_type: str = COACH_ANALYSIS_TYPE_REVIEW,
) -> Optional[Dict[str, Any]]:
    normalized_analysis_type = (
        _normalize_analysis_type(analysis_type) or COACH_ANALYSIS_TYPE_REVIEW
    )
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
    parsed_evidence_season_year = _parse_optional_int(evidence_season_year)

    request_game_id = str(payload.game_id or evidence_game_id or "").strip() or None
    request_game_date = _parse_iso_date(
        evidence_game_date or (payload.league_context or {}).get("game_date")
    )
    has_game_row = _has_game_row_context(evidence)
    missing_game_row = not has_game_row
    season_year_for_context = (
        parsed_evidence_season_year if request_game_date is not None else None
    )
    stage_context_mismatch = (
        request_game_date is not None
        and (
            season_year_for_context is None
            or season_year_for_context != request_game_date.year
        )
    ) or await _has_stage_context_mismatch(pool, evidence, request_game_date)
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

    if normalized_analysis_type == COACH_ANALYSIS_TYPE_REVIEW and missing_final_state:
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
    elif (
        normalized_analysis_type == COACH_ANALYSIS_TYPE_REVIEW
        and _normalize_game_status_bucket(evidence_game_status) == "SCHEDULED"
        and not _is_starter_announcement_pending(evidence)
        and not (assessment.home_pitcher_present and assessment.away_pitcher_present)
    ):
        add_item(
            "starters",
            "선발 정보",
            "선발 발표 예정 시간(경기 전일 18시)이 지났으나 선발 정보가 입력되지 않았습니다.",
            "home_pitcher, away_pitcher",
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
    data_sync_request = _build_baseball_data_sync_request(
        request_game_id=request_game_id,
        request_game_date=request_game_date,
        season_year=parsed_evidence_season_year,
        home_team_id=evidence_home_team_code,
        away_team_id=evidence_away_team_code,
        stage_label=evidence_stage_label,
        analysis_type=normalized_analysis_type,
        missing_items=missing_items,
    )
    return {
        "scope": "coach.analyze",
        "missingItems": missing_items,
        "operatorMessage": "다음 야구 데이터가 필요합니다: "
        + ", ".join(identifier_parts),
        "blocking": True,
        "code": MANUAL_BASEBALL_DATA_REQUIRED_CODE,
        "dataSyncRequired": True,
        "dataSyncCode": BASEBALL_DATA_SYNC_REQUIRED_CODE,
        "externalSource": BASEBALL_DATA_SYNC_EXTERNAL_SOURCE,
        "dataSyncRequest": data_sync_request,
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
    analysis_type: Optional[str] = None,
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
        "analysis_type": (
            _normalize_analysis_type(analysis_type)
            or (
                COACH_ANALYSIS_TYPE_REVIEW
                if _normalize_game_status_bucket(game_status_bucket) == "COMPLETED"
                else COACH_ANALYSIS_TYPE_PREVIEW
            )
        ),
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


def _resolve_llm_skip_reason(
    *,
    request_mode: str,
    generation_mode: Optional[str],
    cache_state: Optional[str],
    cached: bool,
    in_progress: bool,
    manual_data_required: bool,
) -> Optional[str]:
    if manual_data_required:
        return "manual_data_required"
    if cached:
        return "cache_hit"

    normalized_cache_state = str(cache_state or "").strip().upper()
    if in_progress or normalized_cache_state == "PENDING_WAIT":
        return "pending_wait"
    if normalized_cache_state == "FAILED_RETRY_WAIT":
        return "failed_retry_wait"
    if normalized_cache_state == "FAILED_LOCKED":
        return "failed_locked"
    if generation_mode == "llm_manual":
        return None
    if request_mode == COACH_REQUEST_MODE_AUTO:
        return "auto_brief_deterministic"
    if generation_mode in {
        "evidence_fallback",
        "deterministic_preview",
        "deterministic_review",
        "deterministic_auto",
    }:
        return "deterministic_short_circuit"
    return None


def _default_generation_mode_for_request_mode(
    *,
    request_mode: str,
    analysis_type: Optional[str] = None,
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
            return _generation_mode_for_analysis_type(
                analysis_type=analysis_type or COACH_ANALYSIS_TYPE_PREVIEW,
                request_mode=request_mode,
            )
        return "evidence_fallback"

    return "evidence_fallback"


def _ensure_stream_meta_contract(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    request_mode = str(normalized.get("request_mode") or "").strip()
    if request_mode != COACH_REQUEST_MODE_AUTO:
        return normalized

    normalized["analysis_type"] = (
        _normalize_analysis_type(normalized.get("analysis_type"))
        or COACH_ANALYSIS_TYPE_PREVIEW
    )
    normalized["generation_mode"] = _default_generation_mode_for_request_mode(
        request_mode=request_mode,
        analysis_type=normalized.get("analysis_type"),
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
    raw_prob = normalized.get("win_probability_home")
    if isinstance(raw_prob, (int, float)) and 0.0 <= float(raw_prob) <= 1.0:
        normalized["win_probability_home"] = round(float(raw_prob), 3)
    else:
        normalized.pop("win_probability_home", None)
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
    expected_data_quality: Optional[str] = None,
    expected_used_evidence: Optional[Sequence[str]] = None,
    expected_game_status_bucket: Optional[str] = None,
    current_root_causes: Optional[Sequence[str]] = None,
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
    meta = _cached_payload_meta(payload)
    cached_bucket = _normalize_game_status_bucket(meta.get("game_status_bucket"))
    expected_bucket = _normalize_game_status_bucket(expected_game_status_bucket)
    if expected_bucket != "UNKNOWN" and cached_bucket != expected_bucket:
        return True

    cached_quality = str(meta.get("data_quality") or "partial").strip()
    expected_quality = str(expected_data_quality or "").strip()
    if (
        expected_quality in COACH_DATA_QUALITY_RANK
        and COACH_DATA_QUALITY_RANK.get(cached_quality, 1)
        < COACH_DATA_QUALITY_RANK[expected_quality]
    ):
        return True

    expected_evidence = {
        str(item) for item in expected_used_evidence or [] if str(item or "").strip()
    }
    cached_evidence = {
        str(item) for item in meta.get("used_evidence") or [] if str(item or "").strip()
    }
    if expected_evidence and not expected_evidence.issubset(cached_evidence):
        return True

    current_causes = {str(item) for item in current_root_causes or []}
    cached_stale_root_causes = {
        str(item)
        for item in meta.get("grounding_reasons") or []
        if str(item) in COACH_EVIDENCE_ROOT_CAUSE_ORDER
    }
    if cached_stale_root_causes - current_causes:
        return True
    return False


@dataclass
class _CoachLatencyTracker:
    """Lightweight per-request timing aggregator for the coach analyze path.

    Collects monotonic timestamps at key phases (request start, tool fetch,
    first preview chunk, LLM, grounding) and emits one machine-parseable
    ``coach_latency`` log line on completion. No Prometheus dependency; log
    aggregators can derive histograms from the structured fields.
    """

    request_started: float
    first_preview_at: Optional[float] = None
    tool_fetch_started_at: Optional[float] = None
    tool_fetch_completed_at: Optional[float] = None
    llm_started_at: Optional[float] = None
    llm_completed_at: Optional[float] = None
    grounding_started_at: Optional[float] = None
    grounding_completed_at: Optional[float] = None

    def mark_first_preview(self) -> None:
        if self.first_preview_at is None:
            self.first_preview_at = perf_counter()

    def mark_tool_fetch_start(self) -> None:
        self.tool_fetch_started_at = perf_counter()

    def mark_tool_fetch_complete(self) -> None:
        self.tool_fetch_completed_at = perf_counter()

    def mark_llm_start(self) -> None:
        if self.llm_started_at is None:
            self.llm_started_at = perf_counter()

    def mark_llm_complete(self) -> None:
        self.llm_completed_at = perf_counter()

    def mark_grounding_start(self) -> None:
        if self.grounding_started_at is None:
            self.grounding_started_at = perf_counter()

    def mark_grounding_complete(self) -> None:
        self.grounding_completed_at = perf_counter()

    @staticmethod
    def _delta(start: Optional[float], end: Optional[float]) -> Optional[float]:
        if start is None or end is None:
            return None
        return max(0.0, end - start)

    def build_summary(
        self,
        *,
        cache_state: Optional[str],
        request_mode: Optional[str],
        game_status_bucket: Optional[str],
        generation_mode: Optional[str],
        fast_path: bool,
        preview_enabled: bool,
        cache_enabled: bool,
    ) -> Dict[str, Any]:
        now = perf_counter()
        return {
            "coach_total_seconds": round(now - self.request_started, 4),
            "coach_ttfb_seconds": (
                round(self.first_preview_at - self.request_started, 4)
                if self.first_preview_at is not None
                else None
            ),
            "coach_tool_fetch_seconds": self._round(
                self._delta(self.tool_fetch_started_at, self.tool_fetch_completed_at)
            ),
            "coach_llm_seconds": self._round(
                self._delta(self.llm_started_at, self.llm_completed_at)
            ),
            "coach_grounding_seconds": self._round(
                self._delta(self.grounding_started_at, self.grounding_completed_at)
            ),
            "cache_state": cache_state,
            "request_mode": request_mode,
            "game_status_bucket": game_status_bucket,
            "generation_mode": generation_mode,
            "fast_path": bool(fast_path),
            "preview_enabled": bool(preview_enabled),
            "cache_enabled": bool(cache_enabled),
        }

    @staticmethod
    def _round(value: Optional[float]) -> Optional[float]:
        return None if value is None else round(value, 4)


def _emit_coach_latency_summary(
    logger_ref: Any,
    *,
    tracker: _CoachLatencyTracker,
    cache_key: Optional[str],
    game_id: Any,
    cache_state: Optional[str],
    request_mode: Optional[str],
    game_status_bucket: Optional[str],
    generation_mode: Optional[str],
    fast_path: bool,
    preview_enabled: bool,
    cache_enabled: bool,
) -> None:
    summary = tracker.build_summary(
        cache_state=cache_state,
        request_mode=request_mode,
        game_status_bucket=game_status_bucket,
        generation_mode=generation_mode,
        fast_path=fast_path,
        preview_enabled=preview_enabled,
        cache_enabled=cache_enabled,
    )
    logger_ref.info(
        "[CoachLatency] game_id=%s cache_key=%s %s",
        game_id,
        cache_key,
        " ".join(f"{k}={v}" for k, v in summary.items()),
    )
    # Coach 요청 outcome 메트릭 (cache_state x request_mode 분포)
    try:
        AI_COACH_REQUEST_TOTAL.labels(
            cache_state=str(cache_state or "unknown"),
            mode=str(request_mode or "unknown"),
        ).inc()
    except Exception:  # noqa: BLE001
        pass


def _is_manual_recent_form_fast_path_eligible(
    request_mode: str,
    resolved_focus: Optional[List[str]],
) -> bool:
    """True when `manual_detail` request resolves to exactly ``recent_form``.

    Gated behind ``COACH_FAST_PATH_MANUAL_RECENT_FORM``. The deterministic
    composer already covers ``recent_form`` output; skipping the LLM here
    saves the entire generation round-trip.
    """
    if not _coach_fast_path_manual_recent_form_enabled():
        return False
    if request_mode != COACH_REQUEST_MODE_MANUAL:
        return False
    focus = list(resolved_focus or [])
    return focus == ["recent_form"]


def _should_short_circuit_to_deterministic_response(
    *,
    request_mode: str,
    fact_sheet: CoachFactSheet,
    game_status_bucket: str = "UNKNOWN",
    grounding_reasons: Optional[List[str]] = None,
    resolved_focus: Optional[List[str]] = None,
    tool_results: Optional[Dict[str, Any]] = None,
    evidence: Optional[GameEvidence] = None,
) -> bool:
    if request_mode == COACH_REQUEST_MODE_AUTO:
        return True
    if _is_manual_recent_form_fast_path_eligible(request_mode, resolved_focus):
        return True
    # 경기 예측(SCHEDULED manual_detail)에서도 LLM을 호출한다.
    # 과거에는 선발·라인업이 모두 미확정이면 무조건 deterministic 으로 단락했으나,
    # 미래 경기는 거의 항상 라인업 미발표라 예측이 결정론 boilerplate 로 고정됐다.
    # 라인업/선발 미확정 자체로는 단락하지 않고, 아래의 데이터 충분성 게이트
    # (균형 커버리지 + 최소 fact 수)가 LLM 호출 여부를 결정하도록 위임한다.
    # 미발표 라인업 환각은 _sanitize_scheduled_unconfirmed_lineup_entities,
    # validate_response_against_fact_sheet, _scheduled_output_guard_reasons 가 방어한다.
    if _is_completed_manual_review_request(
        request_mode,
        game_status_bucket=game_status_bucket,
    ):
        return True
    # 팀 균형: away 가 있는데 한 팀만 tool 커버 → 비대칭 근거로 LLM 환각 위험 → deterministic.
    if (
        tool_results is not None
        and evidence is not None
        and not _has_balanced_tool_coverage(tool_results, evidence)
    ):
        return True
    # 최소 근거: 실데이터 fact 가 너무 적으면 LLM 호출이 무의미 → deterministic.
    if fact_sheet.supported_fact_count < COACH_MIN_LLM_FACT_COUNT:
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
    has_missing_starters = _starter_missing_reason_present(list(reasons))
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


def _attach_response_analysis_type(
    response_data: Dict[str, Any],
    analysis_type: Optional[str],
) -> Dict[str, Any]:
    normalized_analysis_type = _normalize_analysis_type(analysis_type)
    if not normalized_analysis_type:
        return response_data
    response_data["analysisType"] = normalized_analysis_type
    response_data["analysis_type"] = normalized_analysis_type
    return response_data


def _wrap_cached_payload(
    response_data: Dict[str, Any], meta: Dict[str, Any]
) -> Dict[str, Any]:
    analysis_type = _normalize_analysis_type(meta.get("analysis_type"))
    return {
        "response": _attach_response_analysis_type(response_data, analysis_type),
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
            "analysis_type": analysis_type,
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
    analysis_type: Optional[str] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(cached_data, dict) and isinstance(cached_data.get("response"), dict):
        response = cached_data["response"]
        meta = cached_data.get("_meta", {})
    else:
        response = cached_data
        meta = {}
    normalized_analysis_type = _normalize_analysis_type(
        analysis_type
    ) or _normalize_analysis_type(meta.get("analysis_type"))
    normalized = _attach_response_analysis_type(
        _normalize_cached_response(response),
        normalized_analysis_type,
    )
    meta_defaults = _build_meta_payload_defaults(
        generation_mode=_default_generation_mode_for_request_mode(
            request_mode=request_mode,
            analysis_type=normalized_analysis_type,
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
        analysis_type=normalized_analysis_type,
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
    *,
    retryable: bool = False,
) -> Dict[str, Any]:
    normalized = _sanitize_cache_error_code(code)
    message = (
        COACH_INTERNAL_ERROR_MESSAGE
        if normalized == COACH_INTERNAL_ERROR_CODE
        else _cache_error_message_for_user(normalized)
    )
    return {
        "code": normalized,
        "message": message,
        "retryable": retryable,
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
    if isinstance(exc, TimeoutError) or ("coach_llm" in text and "timeout" in text):
        return COACH_LLM_TIMEOUT_ERROR_CODE
    if "readtimeout" in text or "request timeout" in text:
        return COACH_LLM_TIMEOUT_ERROR_CODE
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
                if not received_first_chunk:
                    received_first_chunk = True
                    yield {"type": "status", "status": "first_chunk_received"}
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
    pool: AsyncConnectionPool,
    cache_key: str,
    timeout_seconds: float = PENDING_WAIT_TIMEOUT_SECONDS,
    poll_ms: int = PENDING_WAIT_POLL_MS,
) -> Optional[Dict[str, Any]]:
    _BACKOFF_MULTIPLIER = 2.0
    _MAX_SLEEP_SECONDS = 30.0
    deadline = perf_counter() + timeout_seconds
    sleep_seconds = max(float(poll_ms), 1.0) / 1000.0

    while perf_counter() < deadline:
        await asyncio.sleep(sleep_seconds)
        try:
            async with pool.connection() as conn:
                row = await (
                    await conn.execute(
                        """
                    SELECT status, response_json, error_message, error_code, attempt_count, updated_at,
                           lease_expires_at, last_heartbeat_at
                    FROM coach_analysis_cache
                    WHERE cache_key = %s
                    """,
                        (cache_key,),
                    )
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
        sleep_seconds = min(sleep_seconds * _BACKOFF_MULTIPLIER, _MAX_SLEEP_SECONDS)
    return None


def _build_cache_lease_owner() -> str:
    # hostname+pid 포함으로 멀티 워커 환경에서 어느 프로세스가 lease를 소유하는지 추적 가능
    try:
        host = socket.gethostname()[:16]
    except Exception:
        host = "unknown"
    pid = os.getpid()
    return f"coach-{host}-{pid}-{uuid.uuid4().hex[:12]}"


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


async def _insert_pending_cache_row(
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
    inserted = await (
        await conn.execute(
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
        )
    ).fetchone()
    return bool(inserted)


async def _heartbeat_cache_lease(
    *,
    pool: AsyncConnectionPool,
    cache_key: str,
    lease_owner: str,
    lease_lost_event: Optional[asyncio.Event] = None,
) -> None:
    while True:
        await asyncio.sleep(COACH_CACHE_HEARTBEAT_INTERVAL_SECONDS)
        try:
            async with pool.connection() as conn:
                result = await conn.execute(
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
                await conn.commit()
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


def _schedule_auto_brief_background_task(
    coroutine: Coroutine[Any, Any, None],
) -> asyncio.Task[None]:
    if len(_AUTO_BRIEF_BACKGROUND_TASKS) >= COACH_AUTO_BRIEF_BACKGROUND_MAX_TASKS:
        coroutine.close()
        raise TimeoutError("auto_brief_background_capacity_exceeded")

    task = asyncio.create_task(coroutine)
    _AUTO_BRIEF_BACKGROUND_TASKS.add(task)
    task.add_done_callback(_AUTO_BRIEF_BACKGROUND_TASKS.discard)
    return task


async def _generate_auto_brief_cache_background_unbounded(
    *,
    pool: AsyncConnectionPool,
    cache_key: str,
    lease_owner: str,
    home_team_canonical: str,
    away_team_canonical: str,
    home_name: str,
    away_name: str,
    year: int,
    resolved_focus: List[str],
    game_evidence: Any,
    evidence_assessment: Any,
    analysis_type: str,
    coach_model_name: str,
    cache_attempt_count: int,
) -> None:
    """auto_brief 캐시를 백그라운드에서 결정론적으로 생성합니다.

    MISS_GENERATE 상태에서 SSE 스트림을 블로킹하지 않고 캐시를 생성합니다.
    완료 시 DB에 COMPLETED로 저장되며, 다음 요청은 HIT를 받게 됩니다.
    """
    lease_lost_event = asyncio.Event()
    heartbeat_task: Optional[asyncio.Task] = asyncio.create_task(
        _heartbeat_cache_lease(
            pool=pool,
            cache_key=cache_key,
            lease_owner=lease_owner,
            lease_lost_event=lease_lost_event,
        )
    )
    try:
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
            include_clutch=_should_include_auto_brief_clutch(game_evidence),
        )

        if lease_lost_event.is_set():
            logger.warning(
                "[AutoBriefBG] Lease lost before completion cache_key=%s", cache_key
            )
            return

        if tool_results.get("matchup"):
            tool_results["matchup"] = _sanitize_matchup_result_for_evidence(
                game_evidence, tool_results["matchup"]
            )

        allowed_names = _collect_allowed_entity_names(game_evidence, tool_results)
        used_evidence = list(game_evidence.used_evidence)
        for src_key, label in (
            ("home.summary", "team_summary"),
            ("home.advanced", "team_advanced_metrics"),
            ("home.player_form_signals", "team_player_form_signals"),
            ("home.recent", "team_recent_form"),
            ("matchup", "head_to_head"),
            ("away.summary", "opponent_team_summary"),
            ("away.advanced", "opponent_team_advanced_metrics"),
            ("away.player_form_signals", "opponent_player_form_signals"),
            ("away.recent", "opponent_recent_form"),
        ):
            parts = src_key.split(".")
            obj = tool_results
            for p in parts:
                obj = obj.get(p, {}) if isinstance(obj, dict) else {}
            if isinstance(obj, dict) and obj.get("found"):
                used_evidence.append(label)
        used_evidence = list(dict.fromkeys(used_evidence))

        dependency_degraded = _collect_dependency_degraded(tool_results)
        dependency_degraded_reasons = _collect_dependency_degraded_reasons(tool_results)
        data_quality = _determine_data_quality(
            game_evidence,
            tool_results,
            analysis_type=analysis_type,
        )
        fact_sheet = _build_coach_fact_sheet(
            game_evidence, tool_results, allowed_names, evidence_assessment
        )
        grounding_reasons = list(fact_sheet.reasons)
        grounding_warnings = list(fact_sheet.warnings)
        if dependency_degraded and dependency_degraded_reasons:
            grounding_reasons.append(COACH_OCI_MAPPING_DEGRADED_ERROR_CODE)
            grounding_warnings.append(
                "매핑 조회는 복구 경로로 이어졌습니다: "
                + ", ".join(dependency_degraded_reasons)
            )
        grounding_reasons = _normalize_grounding_reasons(grounding_reasons)
        grounding_warnings = _merge_grounding_warnings(
            grounding_warnings, grounding_reasons
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
        response_payload = _attach_response_analysis_type(
            response_payload,
            analysis_type,
        )

        meta_defaults = _build_meta_payload_defaults(
            generation_mode=_generation_mode_for_analysis_type(
                analysis_type=analysis_type,
                request_mode=COACH_REQUEST_MODE_AUTO,
            ),
            data_quality=data_quality,
            used_evidence=used_evidence,
            cache_key=cache_key,
            resolved_cache_key=cache_key,
            expected_cache_key=None,
            prompt_version=COACH_CACHE_PROMPT_VERSION,
            starter_signature=None,
            lineup_signature=None,
            cache_key_mismatch=False,
            analysis_type=analysis_type,
            game_status_bucket=game_evidence.game_status_bucket,
            grounding_warnings=grounding_warnings,
            grounding_reasons=grounding_reasons,
            supported_fact_count=fact_sheet.supported_fact_count,
            dependency_degraded=dependency_degraded,
            attempt_count=cache_attempt_count,
            cache_row_missing=False,
            recovered_from_missing_row=False,
            lease_lost=lease_lost_event.is_set(),
        )
        if game_evidence.game_status_bucket == "SCHEDULED":
            pre_game_prob = _calculate_pre_game_win_probability(tool_results)
            if pre_game_prob is not None:
                meta_defaults["win_probability_home"] = pre_game_prob

        await _store_completed_cache(
            pool=pool,
            cache_key=cache_key,
            lease_owner=lease_owner,
            team_id=home_team_canonical,
            year=year,
            prompt_version=COACH_CACHE_PROMPT_VERSION,
            model_name=coach_model_name,
            response_payload=response_payload,
            meta_defaults=meta_defaults,
        )
        logger.info(
            "[AutoBriefBG] Cache stored cache_key=%s home=%s away=%s",
            cache_key,
            home_team_canonical,
            away_team_canonical,
        )
    except Exception as exc:
        failed_data_quality = _determine_data_quality(
            game_evidence,
            analysis_type=analysis_type,
        )
        logger.error(
            "[AutoBriefBG] Generation failed game_id=%s cache_key=%s "
            "cache_state=%s error_code=%s data_quality=%s: %s",
            getattr(game_evidence, "game_id", None),
            cache_key,
            "BACKGROUND_GENERATE",
            "auto_brief_bg_error",
            failed_data_quality,
            exc,
        )
        try:
            await _store_failed_cache(
                pool=pool,
                cache_key=cache_key,
                lease_owner=lease_owner,
                team_id=home_team_canonical,
                year=year,
                prompt_version=COACH_CACHE_PROMPT_VERSION,
                model_name=coach_model_name,
                attempt_count=cache_attempt_count,
                error_code="auto_brief_bg_error",
                error_message=str(exc),
            )
        except Exception as store_exc:
            logger.warning(
                "[AutoBriefBG] Failed to store error state cache_key=%s: %s",
                cache_key,
                store_exc,
            )
    finally:
        await _cancel_heartbeat_task(heartbeat_task)


async def _generate_auto_brief_cache_background(
    *,
    pool: AsyncConnectionPool,
    cache_key: str,
    lease_owner: str,
    home_team_canonical: str,
    away_team_canonical: str,
    home_name: str,
    away_name: str,
    year: int,
    resolved_focus: List[str],
    game_evidence: Any,
    evidence_assessment: Any,
    analysis_type: str,
    coach_model_name: str,
    cache_attempt_count: int,
) -> None:
    try:
        await asyncio.wait_for(
            _generate_auto_brief_cache_background_unbounded(
                pool=pool,
                cache_key=cache_key,
                lease_owner=lease_owner,
                home_team_canonical=home_team_canonical,
                away_team_canonical=away_team_canonical,
                home_name=home_name,
                away_name=away_name,
                year=year,
                resolved_focus=resolved_focus,
                game_evidence=game_evidence,
                evidence_assessment=evidence_assessment,
                analysis_type=analysis_type,
                coach_model_name=coach_model_name,
                cache_attempt_count=cache_attempt_count,
            ),
            timeout=COACH_AUTO_BRIEF_BACKGROUND_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.error(
            "[AutoBriefBG] Generation timed out game_id=%s cache_key=%s timeout_seconds=%.1f",
            getattr(game_evidence, "game_id", None),
            cache_key,
            COACH_AUTO_BRIEF_BACKGROUND_TIMEOUT_SECONDS,
        )
        try:
            await asyncio.wait_for(
                _store_failed_cache(
                    pool=pool,
                    cache_key=cache_key,
                    lease_owner=lease_owner,
                    team_id=home_team_canonical,
                    year=year,
                    prompt_version=COACH_CACHE_PROMPT_VERSION,
                    model_name=coach_model_name,
                    attempt_count=cache_attempt_count,
                    error_code=COACH_LLM_TIMEOUT_ERROR_CODE,
                    error_message="auto_brief_background_timeout",
                ),
                timeout=10.0,
            )
        except Exception as store_exc:  # noqa: BLE001
            logger.warning(
                "[AutoBriefBG] Failed to store timeout state cache_key=%s: %s",
                cache_key,
                store_exc,
            )


async def cancel_auto_brief_background_tasks() -> None:
    tasks = list(_AUTO_BRIEF_BACKGROUND_TASKS)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _AUTO_BRIEF_BACKGROUND_TASKS.difference_update(tasks)


async def _claim_cache_generation_once(
    *,
    pool: AsyncConnectionPool,
    cache_key: str,
    team_id: str,
    year: int,
    prompt_version: str,
    model_name: str,
    lease_owner: str,
    completed_ttl_seconds: Optional[int],
    request_mode: str = COACH_REQUEST_MODE_MANUAL,
    expected_data_quality: Optional[str] = None,
    expected_used_evidence: Optional[Sequence[str]] = None,
    expected_game_status_bucket: Optional[str] = None,
    current_root_causes: Optional[Sequence[str]] = None,
) -> tuple[str, Any, Optional[str], Optional[str], int]:
    async with pool.connection() as conn:
        async with conn.transaction():
            inserted = await _insert_pending_cache_row(
                conn,
                cache_key=cache_key,
                team_id=team_id,
                year=year,
                prompt_version=prompt_version,
                model_name=model_name,
                lease_owner=lease_owner,
                attempt_count=1,
            )
            row = await (
                await conn.execute(
                    """
                SELECT status, response_json, error_message, error_code, attempt_count,
                       updated_at, lease_expires_at, last_heartbeat_at
                FROM coach_analysis_cache
                WHERE cache_key = %s
                FOR UPDATE
                """,
                    (cache_key,),
                )
            ).fetchone()

            if inserted:
                return "MISS_GENERATE", None, None, None, 1

            if not row:
                recreated = await _insert_pending_cache_row(
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

                row = await (
                    await conn.execute(
                        """
                    SELECT status, response_json, error_message, error_code, attempt_count,
                           updated_at, lease_expires_at, last_heartbeat_at
                    FROM coach_analysis_cache
                    WHERE cache_key = %s
                    FOR UPDATE
                    """,
                        (cache_key,),
                    )
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
                expected_data_quality=expected_data_quality,
                expected_used_evidence=expected_used_evidence,
                expected_game_status_bucket=expected_game_status_bucket,
                current_root_causes=current_root_causes,
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
                await conn.execute(
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


async def _read_completed_cache_if_fresh(
    *,
    pool: AsyncConnectionPool,
    cache_key: str,
    completed_ttl_seconds: Optional[int],
    request_mode: str = COACH_REQUEST_MODE_MANUAL,
    expected_data_quality: Optional[str] = None,
    expected_used_evidence: Optional[Sequence[str]] = None,
    expected_game_status_bucket: Optional[str] = None,
    current_root_causes: Optional[Sequence[str]] = None,
) -> tuple[str, Any, Optional[str], Optional[str], int]:
    async with pool.connection() as conn:
        row = await (
            await conn.execute(
                """
            SELECT status, response_json, error_message, error_code, attempt_count,
                   updated_at, lease_expires_at, last_heartbeat_at
            FROM coach_analysis_cache
            WHERE cache_key = %s
            """,
                (cache_key,),
            )
        ).fetchone()

    if not row:
        return "MISS_GENERATE", None, None, None, 0

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
        expected_data_quality=expected_data_quality,
        expected_used_evidence=expected_used_evidence,
        expected_game_status_bucket=expected_game_status_bucket,
        current_root_causes=current_root_causes,
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

    return (
        cache_state,
        None,
        error_message,
        str(error_code or "") or None,
        _normalize_attempt_count(attempt_count),
    )


def _refresh_cache_pool_after_operational_error(
    pool: AsyncConnectionPool,
    *,
    cache_key: str,
    exc: Exception,
) -> None:
    check_pool = getattr(pool, "check", None)
    if not callable(check_pool):
        return
    try:
        check_pool()
    except Exception as check_exc:  # noqa: BLE001
        logger.warning(
            "[Coach] Cache pool check failed after operational error cache_key=%s error=%s check_error=%s",
            cache_key,
            exc,
            check_exc,
        )


async def _claim_cache_generation(
    *,
    pool: AsyncConnectionPool,
    cache_key: str,
    team_id: str,
    year: int,
    prompt_version: str,
    model_name: str,
    lease_owner: str,
    completed_ttl_seconds: Optional[int],
    request_mode: str = COACH_REQUEST_MODE_MANUAL,
    expected_data_quality: Optional[str] = None,
    expected_used_evidence: Optional[Sequence[str]] = None,
    expected_game_status_bucket: Optional[str] = None,
    current_root_causes: Optional[Sequence[str]] = None,
) -> tuple[str, Any, Optional[str], Optional[str], int]:
    kwargs = {
        "pool": pool,
        "cache_key": cache_key,
        "team_id": team_id,
        "year": year,
        "prompt_version": prompt_version,
        "model_name": model_name,
        "lease_owner": lease_owner,
        "completed_ttl_seconds": completed_ttl_seconds,
        "request_mode": request_mode,
        "expected_data_quality": expected_data_quality,
        "expected_used_evidence": expected_used_evidence,
        "expected_game_status_bucket": expected_game_status_bucket,
        "current_root_causes": current_root_causes,
    }
    for attempt in range(1, COACH_CACHE_CLAIM_DB_ATTEMPTS + 1):
        try:
            return await _claim_cache_generation_once(**kwargs)
        except psycopg.OperationalError as exc:
            if attempt >= COACH_CACHE_CLAIM_DB_ATTEMPTS:
                raise
            logger.warning(
                "[Coach] Cache claim DB connection failed; retrying cache_key=%s attempt=%s/%s error=%s",
                cache_key,
                attempt,
                COACH_CACHE_CLAIM_DB_ATTEMPTS,
                exc,
            )
            _refresh_cache_pool_after_operational_error(
                pool, cache_key=cache_key, exc=exc
            )
    raise RuntimeError("unreachable cache claim retry state")


async def _store_completed_cache(
    *,
    pool: AsyncConnectionPool,
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
    async with pool.connection() as conn:
        result = await conn.execute(
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
            await conn.commit()
            return {"outcome": "updated"}

        row = await (
            await conn.execute(
                """
            SELECT status, response_json, lease_owner
            FROM coach_analysis_cache
            WHERE cache_key = %s
            """,
                (cache_key,),
            )
        ).fetchone()
        if not row:
            inserted = await (
                await conn.execute(
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
                            1,
                            _normalize_attempt_count(
                                meta_defaults.get("attempt_count")
                            ),
                        ),
                    ),
                )
            ).fetchone()
            if inserted:
                await conn.commit()
                return {"outcome": "inserted_missing_row"}
            row = await (
                await conn.execute(
                    """
                SELECT status, response_json, lease_owner
                FROM coach_analysis_cache
                WHERE cache_key = %s
                """,
                    (cache_key,),
                )
            ).fetchone()
        await conn.commit()
    return {
        "outcome": "finalize_conflict",
        "status": row[0] if row else None,
        "response_json": row[1] if row else None,
        "lease_owner": row[2] if row else None,
    }


async def _store_failed_cache(
    *,
    pool: AsyncConnectionPool,
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
    async with pool.connection() as conn:
        result = await conn.execute(
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
            await conn.commit()
            return {"outcome": "updated"}

        row = await (
            await conn.execute(
                """
            SELECT status, response_json, lease_owner
            FROM coach_analysis_cache
            WHERE cache_key = %s
            """,
                (cache_key,),
            )
        ).fetchone()
        if not row:
            inserted = await (
                await conn.execute(
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
                )
            ).fetchone()
            if inserted:
                await conn.commit()
                return {"outcome": "inserted_missing_row"}
            row = await (
                await conn.execute(
                    """
                SELECT status, response_json, lease_owner
                FROM coach_analysis_cache
                WHERE cache_key = %s
                """,
                    (cache_key,),
                )
            ).fetchone()
        await conn.commit()
    return {
        "outcome": "finalize_conflict",
        "status": row[0] if row else None,
        "response_json": row[1] if row else None,
        "lease_owner": row[2] if row else None,
    }


async def _reset_failed_coach_cache_rows(
    pool: AsyncConnectionPool,
    *,
    cache_key: Optional[str] = None,
    team_id: Optional[str] = None,
    year: Optional[int] = None,
    include_stale_pending: bool = True,
    retryable_only: bool = False,
    max_rows: int = 500,
) -> Dict[str, int]:
    """
    FAILED(잠긴 캐시 포함) row를 삭제해 다음 요청에서 자연 재생성되도록 한다.

    배치 스크립트의 ``force_rebuild_delete`` 안전 규칙(활성 PENDING 보호)을 런타임
    async 경로에 맞춰 재구현한 헬퍼다. ``cache_key`` 가 주어지면 단건을, 아니면
    ``team_id``+``year`` 로 묶인 FAILED row를 일괄 처리한다. ``COMPLETED`` row와
    아직 살아있는 PENDING lease는 절대 건드리지 않는다.
    """
    stats: Dict[str, int] = {
        "matched_rows": 0,
        "deleted_failed_rows": 0,
        "deleted_stale_pending_rows": 0,
        "active_pending_blocked_count": 0,
        "completed_skipped_count": 0,
        "non_retryable_skipped_count": 0,
        "missing_cache_row_count": 0,
    }
    async with pool.connection() as conn:
        if cache_key:
            rows = await (
                await conn.execute(
                    """
                SELECT cache_key, status, error_code, updated_at,
                       lease_expires_at, last_heartbeat_at
                FROM coach_analysis_cache
                WHERE cache_key = %s
                FOR UPDATE
                """,
                    (cache_key,),
                )
            ).fetchall()
            if not rows:
                stats["missing_cache_row_count"] = 1
        else:
            rows = await (
                await conn.execute(
                    """
                SELECT cache_key, status, error_code, updated_at,
                       lease_expires_at, last_heartbeat_at
                FROM coach_analysis_cache
                WHERE team_id = %s AND year = %s
                  AND status IN ('FAILED', 'PENDING')
                ORDER BY updated_at ASC
                LIMIT %s
                FOR UPDATE
                """,
                    (team_id, year, max_rows),
                )
            ).fetchall()

        deletable: List[Tuple[str, str]] = []
        for row in rows:
            (
                row_cache_key,
                status,
                error_code,
                updated_at,
                lease_expires_at,
                last_heartbeat_at,
            ) = row
            stats["matched_rows"] += 1
            normalized_status = str(status or "").strip().upper()
            if normalized_status == "FAILED":
                if retryable_only and not _is_retryable_cache_error_code(error_code):
                    stats["non_retryable_skipped_count"] += 1
                    continue
                deletable.append(("failed", str(row_cache_key)))
            elif normalized_status == "PENDING":
                if include_stale_pending and _is_pending_cache_stale(
                    updated_at=updated_at,
                    last_heartbeat_at=last_heartbeat_at,
                    lease_expires_at=lease_expires_at,
                    pending_stale_seconds=PENDING_STALE_SECONDS,
                ):
                    deletable.append(("stale_pending", str(row_cache_key)))
                else:
                    stats["active_pending_blocked_count"] += 1
            else:
                stats["completed_skipped_count"] += 1

        delete_keys = [key for _, key in deletable]
        if delete_keys:
            await conn.execute(
                "DELETE FROM coach_analysis_cache WHERE cache_key = ANY(%s)",
                (delete_keys,),
            )
            stats["deleted_failed_rows"] = sum(
                1 for kind, _ in deletable if kind == "failed"
            )
            stats["deleted_stale_pending_rows"] = sum(
                1 for kind, _ in deletable if kind == "stale_pending"
            )
        await conn.commit()
    return stats


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


async def _resolve_year_from_season_id(
    pool: AsyncConnectionPool, season_id: Any
) -> Optional[int]:
    if season_id is None:
        return None
    try:
        normalized_season_id = int(str(season_id).strip())
    except (TypeError, ValueError):
        return None

    try:
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT season_year FROM kbo_seasons WHERE season_id = %s LIMIT 1",
                    (normalized_season_id,),
                )
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


async def _resolve_year_from_game_context(
    pool: AsyncConnectionPool, game_id: Optional[str], game_date: Any
) -> Optional[int]:
    explicit_game_year = _parse_year_from_date_like(game_date)
    if explicit_game_year is not None:
        return explicit_game_year

    if game_id:
        try:
            async with pool.connection() as conn:
                row = await (
                    await conn.execute(
                        """
                    SELECT COALESCE(ks.season_year, EXTRACT(YEAR FROM g.game_date)::int) AS season_year
                    FROM game g
                    LEFT JOIN kbo_seasons ks ON g.season_id = ks.season_id
                    WHERE g.game_id = %s
                    LIMIT 1
                    """,
                        (game_id,),
                    )
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


async def _resolve_target_year(
    payload: "AnalyzeRequest", pool: AsyncConnectionPool
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
    season_year = await _resolve_year_from_season_id(pool, season_id)
    if season_year is not None and _is_valid_analysis_year(season_year):
        return season_year, "league_context.season->kbo_seasons"

    game_year = await _resolve_year_from_game_context(
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
    game_status = _normalize_name_token(league_context.get("game_status")) or "UNKNOWN"
    game_status_bucket = _normalize_game_status_bucket(game_status)
    home_pitcher = _normalize_pitcher_name_token(league_context.get("home_pitcher"))
    away_pitcher = _normalize_pitcher_name_token(league_context.get("away_pitcher"))
    lineup_announced = bool(league_context.get("lineup_announced"))
    evidence = GameEvidence(
        game_id=payload.game_id,
        season_year=year,
        season_id=_parse_optional_int(league_context.get("season")),
        game_date=str(league_context.get("game_date") or "").strip() or None,
        game_status=game_status,
        game_status_bucket=game_status_bucket,
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


async def _fetch_series_state(
    conn: Any,
    evidence: GameEvidence,
) -> Optional[EvidenceSeriesState]:
    if not evidence.game_id or evidence.stage_label in {"REGULAR", "PRE", "UNKNOWN"}:
        return None

    previous_games = list(
        await (
            await conn.execute(
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
            )
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


async def _collect_game_evidence(
    pool: AsyncConnectionPool,
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
        async with pool.connection() as conn:
            cursor = conn.cursor(row_factory=dict_row)
            game_row = await (
                await cursor.execute(
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
                )
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
            home_score = _parse_optional_int(game_row.get("home_score"))
            away_score = _parse_optional_int(game_row.get("away_score"))
            resolved_game_status = str(game_row.get("game_status") or "UNKNOWN")
            resolved_game_status_bucket = _normalize_game_status_bucket(
                resolved_game_status
            )
            if (
                resolved_game_status_bucket == "UNKNOWN"
                and home_score is not None
                and away_score is not None
            ):
                resolved_game_status = "COMPLETED"
                resolved_game_status_bucket = "COMPLETED"
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
                game_status_bucket=resolved_game_status_bucket,
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

            lineup_rows = await (
                await cursor.execute(
                    """
                SELECT team_code, player_name, batting_order, is_starter
                FROM game_lineups
                WHERE game_id = %s
                  AND COALESCE(is_starter, true) = true
                ORDER BY team_code, batting_order ASC NULLS LAST, player_name ASC
                """,
                    (payload.game_id,),
                )
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

            summary_rows = await (
                await cursor.execute(
                    """
                SELECT summary_type, player_name, detail_text
                FROM game_summary
                WHERE game_id = %s
                ORDER BY id ASC
                LIMIT 30
                """,
                    (payload.game_id,),
                )
            ).fetchall()
            evidence.summary_items = _format_game_summary_items(summary_rows or [])

            evidence.series_state = _reconcile_series_state_with_hint(
                await _fetch_series_state(conn, evidence),
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


def _build_focus_section_requirements(
    resolved_focus: List[str],
    *,
    game_status_bucket: Optional[str] = None,
) -> str:
    """
    선택 focus에 해당하는 상세 섹션 제목 요구사항을 생성합니다.
    """
    if not resolved_focus:
        return (
            "- 선택 focus가 비어 있습니다. 종합 분석을 수행하세요.\n"
            "- 다만 detailed_markdown은 최소 2개 이상의 소제목(##)으로 구성하세요."
        )

    headers = _focus_section_headers_for_status(game_status_bucket)
    header_lines = [
        f"- 반드시 `{headers[focus]}` 제목을 포함하세요."
        for focus in resolved_focus
        if focus in headers
    ]
    non_selected = [
        header for key, header in headers.items() if key not in resolved_focus
    ]
    omit_lines = [
        f"- 미선택 focus는 가능하면 생략하세요: `{header}`" for header in non_selected
    ]
    return "\n".join(header_lines + omit_lines)


def _find_missing_focus_sections(
    response_data: Dict[str, Any],
    resolved_focus: List[str],
    game_status_bucket: Optional[str] = None,
) -> List[str]:
    """
    detailed_markdown에서 선택 focus 섹션 누락 여부를 확인합니다.
    """
    if not resolved_focus:
        return []

    effective_status_bucket = game_status_bucket or response_data.get(
        "game_status_bucket"
    )
    markdown = str(response_data.get("detailed_markdown") or "")
    missing: List[str] = []
    for focus in resolved_focus:
        header = _focus_section_header(focus, effective_status_bucket)
        if not header:
            continue
        if header not in markdown or not _markdown_section_has_body(markdown, header):
            missing.append(focus)
    return missing


def _with_focus_status_context(
    response_data: Dict[str, Any],
    game_status_bucket: Optional[str],
) -> Dict[str, Any]:
    if not game_status_bucket or response_data.get("game_status_bucket"):
        return response_data
    updated = dict(response_data)
    updated["game_status_bucket"] = game_status_bucket
    return updated


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
        if any(
            value is not None
            for value in (
                home_adv.get("fatigue_index", {}).get("bullpen_share"),
                away_adv.get("fatigue_index", {}).get("bullpen_share"),
            )
        ):
            return True
        if any(
            value is not None
            for value in (
                home_adv.get("fatigue_index", {}).get("avg_era"),
                away_adv.get("fatigue_index", {}).get("avg_era"),
            )
        ):
            return True
        for data in (home_data, away_data):
            for pitcher in data.get("player_form_signals", {}).get("pitchers", []):
                if (
                    pitcher.get("role") == "reliever"
                    and pitcher.get("season_metrics", {}).get("era") is not None
                ):
                    return True
        return False
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
    header = _focus_section_header(focus)
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
            f"{evidence.away_team_name} 출루·장타 지표 {away_ops_value:.3f} / "
            f"{evidence.home_team_name} 출루·장타 지표 {home_ops_value:.3f}로 "
            f"{edge_team}의 득점 연결력이 더 좋아 보입니다."
        )

    return "득점 연결력 비교에 필요한 출루·장타 지표가 충분하지 않습니다."


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


def _build_scheduled_team_level_snapshot_lines(
    evidence: GameEvidence,
    tool_results: Dict[str, Any],
) -> List[str]:
    """Render a bullet snapshot per team for scheduled deterministic markdown."""

    lines: List[str] = []
    for team_name, team_data in (
        (evidence.away_team_name, tool_results.get("away", {}) or {}),
        (evidence.home_team_name, tool_results.get("home", {}) or {}),
    ):
        recent = _recent_record_metric_fragment(team_name, team_data)
        adv = _advanced_metrics(team_data)
        ops_value = adv.get("metrics", {}).get("batting", {}).get("ops")
        bullpen_share = _safe_percent_value(
            adv.get("fatigue_index", {}).get("bullpen_share")
        )

        fragments: List[str] = []
        if recent and "데이터 부족" not in recent:
            fragments.append(f"최근 **{recent}**")
        if ops_value is not None:
            try:
                fragments.append(f"출루·장타 지표 **{float(ops_value):.3f}**")
            except (TypeError, ValueError):
                pass
        if bullpen_share:
            fragments.append(f"불펜 비중 **{bullpen_share:.1f}%**")

        if fragments:
            lines.append(f"- **{team_name}**: " + ", ".join(fragments))
        else:
            lines.append(f"- **{team_name}**: 최근 공개 지표가 부족해 집계 대기 중")

    if evidence.away_pitcher or evidence.home_pitcher:
        starter_line = (
            f"- 발표 선발: {evidence.away_team_name} {evidence.away_pitcher or '미정'} / "
            f"{evidence.home_team_name} {evidence.home_pitcher or '미정'}"
        )
        lines.append(starter_line)
    return lines


def _recent_win_rate(summary: Dict[str, Any]) -> Optional[float]:
    wins = int(summary.get("wins", 0) or 0)
    losses = int(summary.get("losses", 0) or 0)
    draws = int(summary.get("draws", 0) or 0)
    total_games = wins + losses + draws
    if total_games <= 0:
        return None
    return (wins + 0.5 * draws) / total_games


def _calculate_pre_game_win_probability(
    tool_results: Dict[str, Any],
    *,
    home_advantage: float = 0.04,
) -> Optional[float]:
    """Log5-based pre-game win probability for the home team.

    Uses recent-form win rate (last ~10 games) as the base statistic.
    Returns None if either team has < 3 recent games (insufficient data).
    Clamped to [0.30, 0.75] to prevent extreme outliers.
    """
    home_data = tool_results.get("home") or {}
    away_data = tool_results.get("away") or {}
    home_recent_summary = _recent_summary(home_data)
    away_recent_summary = _recent_summary(away_data)
    home_games = _recent_sample_size(home_data)
    away_games = _recent_sample_size(away_data)

    if home_games < 3 or away_games < 3:
        return None

    hw = _recent_win_rate(home_recent_summary)
    aw = _recent_win_rate(away_recent_summary)

    if hw is None or aw is None:
        return None

    # Log5 formula
    denom = hw * (1.0 - aw) + aw * (1.0 - hw) + 1e-9
    p = hw * (1.0 - aw) / denom
    p += home_advantage

    return max(0.30, min(0.75, round(p, 3)))


def _attach_scheduled_win_probability(
    meta_defaults: Dict[str, Any],
    *,
    game_status_bucket: str,
    tool_results: Dict[str, Any],
    response_payload: Optional[Dict[str, Any]] = None,
) -> None:
    """SCHEDULED 경기의 meta 에 홈팀 사전 승률을 채운다.

    프론트 다이얼로그 승률 hero 는 meta 의 ``win_probability_home`` 을 읽으므로,
    auto_brief 뿐 아니라 manual_detail(예측 다이얼로그)·LLM 경로에도 동일하게 설정한다.
    LLM 응답 payload 에 유효한 값(0~1)이 있으면 우선 사용하고, 없으면 최근 폼 기반
    Log5 계산값을 사용한다. 계산 불가(데이터 부족)면 키를 설정하지 않아
    프론트의 ``initialWinProbabilityHome`` fallback 이 동작하게 둔다.
    """
    if str(game_status_bucket or "").upper() != "SCHEDULED":
        return
    if isinstance(response_payload, dict):
        raw = response_payload.get("win_probability_home")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            value = float(raw)
            if 0.0 <= value <= 1.0:
                meta_defaults["win_probability_home"] = round(value, 3)
                return
    pre_game_prob = _calculate_pre_game_win_probability(tool_results)
    if pre_game_prob is not None:
        meta_defaults["win_probability_home"] = pre_game_prob


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
        return f"{edge_team}가 최근 흐름과 득점 연결력에서 앞섭니다."
    if edge_source == "ops":
        return f"{edge_team}가 득점 연결력에서 한 발 앞섭니다."
    if edge_source == "ops_slight":
        return f"{edge_team}가 득점 연결력 우세로 근소하게 앞섭니다."
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
            f"{edge_team}가 최근 흐름과 출루·장타 지표를 함께 앞세워 초중반 주도권을 먼저 잡을 가능성이 있습니다.",
        )
    elif edge_team and edge_source == "recent_slight":
        _append_distinct_note_part(
            ordered,
            f"{edge_team}가 최근 흐름 우위로 경기 중반 운영 선택지를 먼저 확보할 가능성이 있습니다.",
        )
        _append_distinct_note_part(
            ordered,
            f"{secondary_edge_team or trailing_team}도 출루·장타 지표 우위로 초반 선취점 반격 여지는 남아 있습니다.",
        )
    elif edge_team and edge_source == "ops_slight":
        _append_distinct_note_part(
            ordered,
            f"{edge_team}가 출루·장타 지표 우위로 초반 선취점 압박을 먼저 걸 수 있습니다.",
        )
        _append_distinct_note_part(
            ordered,
            f"{secondary_edge_team or trailing_team}도 최근 흐름 반등으로 경기 중반 대응 여지는 남아 있습니다.",
        )
    elif edge_team and edge_source == "composite":
        _append_distinct_note_part(
            ordered,
            f"{edge_team}가 최근 흐름과 득점 연결력을 종합하면 운영 선택지를 조금 더 넓게 가져갈 가능성이 있습니다.",
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


def _is_informative_clutch_description(description: Optional[str]) -> bool:
    return any(char.isalnum() for char in str(description or ""))


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
    if description and not _is_informative_clutch_description(description):
        description = None

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
    evidence: Optional[GameEvidence] = None,
) -> Optional[str]:
    warning = _clean_summary_text(warning_text)
    if not warning:
        return None
    if focus == "matchup" and "상대 전적" in warning:
        return _completed_focus_fallback_summary(evidence, "matchup_warning")
    if focus == "recent_form" and "최근" in warning:
        return _completed_focus_fallback_summary(evidence, "recent_form_warning")
    if focus == "bullpen" and "불펜" in warning:
        return _completed_focus_fallback_summary(evidence, "bullpen_warning")
    if focus == "batting" and "타격" in warning:
        return _completed_focus_fallback_summary(evidence, "batting_warning")
    return warning


def _stable_completed_copy_index(
    evidence: Optional[GameEvidence],
    salt: str,
    count: int,
) -> int:
    if count <= 1:
        return 0
    seed_parts = [
        salt,
        getattr(evidence, "game_id", None),
        getattr(evidence, "game_date", None),
        getattr(evidence, "away_team_code", None),
        getattr(evidence, "home_team_code", None),
    ]
    seed = "|".join(str(part or "") for part in seed_parts)
    value = sum((index + 1) * ord(char) for index, char in enumerate(seed))
    return value % count


def _completed_focus_fallback_summary(
    evidence: Optional[GameEvidence],
    focus: str,
    *,
    salt: Optional[str] = None,
) -> str:
    variants = COMPLETED_FOCUS_FALLBACK_VARIANTS.get(focus)
    if not variants:
        return FOCUS_SECTION_GENERIC_SUMMARIES.get(
            focus, "확인 가능한 실데이터 기준으로 추가 근거가 제한적입니다."
        )
    index = _stable_completed_copy_index(evidence, salt or focus, len(variants))
    return variants[index]


def _completed_limited_recent_text(
    evidence: GameEvidence,
    team_name: Optional[str] = None,
    *,
    salt: str = "recent_form",
) -> str:
    base = _completed_focus_fallback_summary(evidence, "recent_form", salt=salt)
    if not team_name:
        return base
    if base.startswith("최근 "):
        return f"{team_name} {base}"
    return f"{team_name}의 {base}"


def _completed_ops_weakness_text(
    evidence: GameEvidence,
    team_name: str,
    ops: float,
) -> str:
    variants = (
        f"{team_name}는 출루·장타 지표 {ops:.3f} 기준 득점 연결력 보강이 필요합니다.",
        f"{team_name}는 출루·장타 지표 {ops:.3f} 기준 득점 기회를 먼저 살려야 합니다.",
        f"{team_name}는 출루·장타 지표 {ops:.3f}라 출루 이후 장타 연결이 관건입니다.",
    )
    index = _stable_completed_copy_index(
        evidence,
        f"ops_weakness:{team_name}",
        len(variants),
    )
    return variants[index]


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
            else _completed_limited_recent_text(
                evidence,
                evidence.away_team_name,
                salt="recent_form:away",
            )
        )
        home_text = (
            _short_recent_form_text(evidence.home_team_name, home_data)
            if home_games > 0
            else _completed_limited_recent_text(
                evidence,
                evidence.home_team_name,
                salt="recent_form:home",
            )
        )
        return f"{away_text} / {home_text}"

    return _completed_limited_recent_text(evidence)


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

    return _completed_focus_fallback_summary(evidence, "bullpen")


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
    return _completed_focus_fallback_summary(evidence, "starter")


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
        return (
            _compact_completed_focus_warning("matchup", warning_text, evidence=evidence)
            or warning_text
        )

    return _completed_focus_fallback_summary(evidence, "matchup")


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

    return _completed_focus_fallback_summary(evidence, "batting")


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
    else:
        metrics.append(
            {
                "label": "발표 선발",
                "value": (
                    f"{evidence.away_team_name} 미정 / {evidence.home_team_name} 미정"
                    " (경기 직전 확인 필요)"
                ),
                "status": "warning",
                "trend": "neutral",
                "is_critical": True,
            }
        )

    existing_labels = {str(metric.get("label") or "") for metric in metrics}
    if "최근 흐름" not in existing_labels:
        metrics.insert(
            0,
            {
                "label": "최근 흐름",
                "value": (
                    f"{evidence.away_team_name} 최근 표본 부족 / "
                    f"{evidence.home_team_name} 최근 표본 부족"
                ),
                "status": "neutral",
                "trend": "neutral",
                "is_critical": False,
            },
        )
    if "팀 타격 생산성" not in existing_labels:
        metrics.append(
            {
                "label": "득점 연결력",
                "value": "출루·장타 지표 집계 부족 — 정규시즌 누적 기준 보강 대기",
                "status": "neutral",
                "trend": "neutral",
                "is_critical": False,
            }
        )

    return _finalize_deterministic_metrics(metrics)


def _ensure_minimum_risks(
    risks: List[Dict[str, Any]],
    *,
    candidates: Optional[List[Dict[str, Any]]] = None,
    fallback: Optional[Dict[str, Any]] = None,
    minimum: int = 1,
) -> List[Dict[str, Any]]:
    """리스크가 ``minimum`` 미만이면 ``candidates`` → ``fallback`` 순으로 보강합니다.

    이미 채워진 리스크는 보존하고, 부족할 때만 *이미 계산된 내부 신호*에서 유도한
    후보로 채웁니다. 외부 데이터 호출이 없으므로 Baseball Data Policy를 위반하지
    않습니다. level은 항상 0/1/2로 강제합니다.
    """
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in risks or []:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description") or "").strip()
        if not desc or desc in seen:
            continue
        result.append(item)
        seen.add(desc)
    for cand in candidates or []:
        if len(result) >= minimum:
            break
        if not isinstance(cand, dict):
            continue
        desc = str(cand.get("description") or "").strip()
        if not desc or desc in seen:
            continue
        result.append(cand)
        seen.add(desc)
    if not result and fallback and str(fallback.get("description") or "").strip():
        result.append(fallback)
    for item in result:
        if item.get("level") not in (0, 1, 2):
            item["level"] = 1
    return result


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
            f"{ops_edge_team}는 출루·장타 지표 {ops_lead_value:.3f}로 득점 연결력이 더 좋습니다."
        )
        weaknesses.append(
            f"{ops_trailing_team}는 출루·장타 지표 {ops_trailing_value:.3f}로 초반 득점 기회를 만드는 힘을 보강해야 합니다."
        )
        ops_reason = f"{ops_edge_team}가 출루·장타 지표에서 앞서 초반 선취점 압박을 먼저 걸 가능성이 있습니다."

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
        due_at = _starter_announcement_due_at_kst(evidence.game_date)
        if due_at is not None:
            due_label = due_at.strftime("%-m월 %-d일 %H시")
            uncertainty.append(
                f"선발은 {due_label}(KST) 발표 예정이며, 발표 전까지 초반 흐름 해석은 보수적으로 봐야 합니다."
            )
        else:
            uncertainty.append(
                "공식 선발 발표 전이라 초반 흐름 해석은 보수적으로 봐야 합니다."
            )
        swing_factors.append(
            "선발 발표 후 투구 스타일 매치업에 따라 초반 유리 팀이 바뀔 수 있습니다."
        )
        home_pitcher_form = _best_form_signal(home_data, "pitchers")
        away_pitcher_form = _best_form_signal(away_data, "pitchers")
        if home_pitcher_form or away_pitcher_form:
            watch_points.append(
                "선발 발표 전이라 양 팀 불펜 투수의 최근 등판 흐름이 초반 전략 단서가 됩니다."
            )
        else:
            watch_points.append(
                "선발 미확정이므로 경기 직전 발표되는 선발과 불펜 대기 구성을 확인할 필요가 있습니다."
            )

    summary: str
    verdict: str
    if edge_team and trailing_team:
        if edge_source == "recent_ops":
            summary = (
                f"{edge_team}가 최근 흐름과 득점 연결력에서 모두 앞서 있지만, "
                f"{trailing_team}도 운영 변수 하나로 흐름을 뒤집을 여지는 남아 있습니다."
            )
        elif edge_source == "recent_slight":
            summary = (
                f"{edge_team}가 최근 전력 우세로 근소하게 앞서지만, "
                f"{(secondary_edge_team or trailing_team)}의 팀 타격 생산성 반격 여지는 남아 있습니다."
            )
        elif edge_source == "ops_slight":
            summary = (
                f"{edge_team}가 득점 연결력 우세로 근소하게 앞서지만, "
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

    # 강점 최소 1건 보장: 양 팀 모두 하락세면 폼 루프가 strengths를 비울 수 있음.
    if not strengths:
        if ops_edge_team:
            strengths.append(
                f"{ops_edge_team}는 득점 연결력 우위로 초반 득점 기회를 만드는 힘이 더 좋습니다."
            )
        elif edge_team:
            strengths.append(f"{edge_team}는 확인된 팀 지표에서 우위를 보입니다.")
        else:
            strengths.append(
                "양 팀 전력이 팽팽해 초반 주도권 다툼이 관전 포인트입니다."
            )

    # 약점 최소 1건 보장: strengths는 폼 기반으로 항상 채워지지만 weaknesses는 비대칭이라 빌 수 있음.
    if not weaknesses:
        if ops_edge_team:
            ops_trailing_name = home_name if ops_edge_team == away_name else away_name
            weaknesses.append(
                f"{ops_trailing_name}는 득점 연결력 열세로 초반 득점 기회를 만드는 힘을 보강해야 합니다."
            )
        elif trailing_team:
            weaknesses.append(
                f"{trailing_team}는 확인된 우위 지표가 적어 초반 흐름 관리가 과제입니다."
            )
        else:
            weaknesses.append(
                "양 팀 모두 확인된 우위가 제한적이라 초반 변동성 관리가 공통 과제입니다."
            )

    # 리스크 최소 1건 보장: 이미 계산된 폼/OPS/라인업 신호에서 유도 (외부 호출 없음).
    min_risk_candidates: List[Dict[str, Any]] = []
    for team_name, signal in (
        (away_name, away_form_signal),
        (home_name, home_form_signal),
    ):
        if _form_status_trend_label((signal or {}).get("form_status")) == "하락세":
            min_risk_candidates.append(
                {
                    "area": "form",
                    "level": 1,
                    "description": f"{team_name}는 팀 폼이 하락세라 초반 흐름이 흔들리면 반등 동력이 부족할 수 있습니다.",
                }
            )
    if ops_edge_team:
        ops_trailing_name = home_name if ops_edge_team == away_name else away_name
        min_risk_candidates.append(
            {
                "area": "offense",
                "level": 1,
                "description": f"{ops_trailing_name}는 득점 연결력 열세로 초반 득점 기회를 살리지 못하면 추격 부담이 커집니다.",
            }
        )
    if not evidence.home_pitcher or not evidence.away_pitcher:
        min_risk_candidates.append(
            {
                "area": "lineup",
                "level": 1,
                "description": "선발·라인업 미발표라 초반 매치업 해석에 불확실성이 큽니다.",
            }
        )
    risks = _ensure_minimum_risks(
        risks,
        candidates=min_risk_candidates,
        fallback={
            "area": "overall",
            "level": 1 if edge_team else 2,
            "description": "확인된 지표상 결정적 리스크는 낮지만, 초반 선취점과 첫 불펜 선택이 경기 흐름을 좌우할 변수입니다.",
        },
        minimum=2,
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

    return _finalize_deterministic_metrics(metrics)


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
                f"{home_name} 출루·장타 지표 {home_ops:.3f}로 {away_name}({away_ops:.3f})보다 득점 연결력이 좋습니다."
            )
            weaknesses.append(
                _completed_ops_weakness_text(evidence, away_name, away_ops)
                if review_mode
                else f"{away_name}는 출루·장타 지표 {away_ops:.3f} 기준 득점 연결력 보강이 필요합니다."
            )
            why_it_matters.append(
                f"{home_name}가 출루·장타 지표에서 앞서 선취점 압박을 먼저 걸 가능성이 있습니다."
            )
        else:
            edge_scores[away_name] += 2
            strengths.append(
                f"{away_name} 출루·장타 지표 {away_ops:.3f}가 {home_name}({home_ops:.3f})보다 좋아 득점 연결력이 앞섭니다."
            )
            weaknesses.append(
                _completed_ops_weakness_text(evidence, home_name, home_ops)
                if review_mode
                else f"{home_name}는 출루·장타 지표 {home_ops:.3f} 기준 초반 득점 루트 보강이 필요합니다."
            )
            why_it_matters.append(
                f"{away_name}가 장타 흐름에서 앞서 있어 초반 득점 루트를 더 쉽게 만들 수 있습니다."
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
                f"{player_name}의 최근 장타 흐름이 유지되면 {team_name}가 선취점과 큰 득점 이닝을 만들 가능성을 키울 수 있습니다."
            )
        elif status == "cold":
            weaknesses.append(
                f"{team_name}는 {player_name}의 폼 점수 {score_text}로 중심 타선 파괴력이 평소보다 낮습니다."
            )
            risks.append(
                {
                    "area": "batting",
                    "level": 1,
                    "description": f"{team_name} 핵심 타자 {player_name}의 최근 득점 연결력이 둔화됐습니다.",
                }
            )
            if not review_mode:
                watch_points.append(
                    f"{team_name}는 {player_name} 앞뒤 타순에서 출루를 얼마나 이어 주는지가 득점 변동성을 줄일 포인트입니다."
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
                    "description": f"{team_name} {player_name}의 최근 실점 흐름이 좋지 않아 경기 흐름을 흔들 수 있습니다.",
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
            "공식 선발 발표 전이라 초반 매치업 해석은 보수적으로 봐야 합니다."
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

    # 리스크 최소 1건 보장: 이미 계산된 OPS/우열 신호에서 유도 (외부 호출 없음).
    det_edge_team = (
        home_name
        if home_score > away_score
        else away_name if away_score > home_score else None
    )
    # 강점 최소 1건 보장: 양 팀 모두 하락세면 strengths가 빌 수 있음.
    if not strengths:
        if home_ops and away_ops and home_ops != away_ops:
            ops_leader_name = home_name if home_ops > away_ops else away_name
            strengths.append(
                f"{ops_leader_name} 팀 OPS 우위로 "
                + (
                    "득점 연결 효율이 결과를 뒷받침했습니다."
                    if review_mode
                    else "초반 득점 설계에서 앞섭니다."
                )
            )
        elif det_edge_team:
            strengths.append(
                f"{det_edge_team}{_topic_particle(det_edge_team)} 확인된 지표에서 우위를 "
                + ("보이며 결과로 이어졌습니다." if review_mode else "보입니다.")
            )
        else:
            strengths.append(
                "양 팀 전력이 팽팽해 "
                + (
                    "승부처 집중력이 결과를 갈랐습니다."
                    if review_mode
                    else "초반 주도권 다툼이 관전 포인트입니다."
                )
            )

    # 약점 최소 1건 보장: weaknesses는 strengths와 달리 비대칭이라 빌 수 있음.
    if not weaknesses:
        if home_ops and away_ops and home_ops != away_ops:
            ops_trailing_name = away_name if home_ops > away_ops else home_name
            weaknesses.append(
                f"{ops_trailing_name} 팀 OPS 열세로 "
                + (
                    "득점 연결 효율이 결과에 영향을 줬습니다."
                    if review_mode
                    else "초반 득점 설계가 과제입니다."
                )
            )
        elif det_edge_team:
            trailing = away_name if det_edge_team == home_name else home_name
            weaknesses.append(
                f"{trailing}{_topic_particle(trailing)} 확인된 우위 지표가 적어 "
                + (
                    "결과를 뒤집을 동력이 부족했습니다."
                    if review_mode
                    else "초반 흐름 관리가 과제입니다."
                )
            )
        else:
            weaknesses.append(
                "양 팀 모두 확인된 우위가 제한적이라 "
                + (
                    "승부처 대응이 결과를 갈랐습니다."
                    if review_mode
                    else "초반 변동성 관리가 공통 과제입니다."
                )
            )

    min_risk_candidates: List[Dict[str, Any]] = []
    if home_ops and away_ops and home_ops != away_ops:
        ops_trailing_name = away_name if home_ops > away_ops else home_name
        min_risk_candidates.append(
            {
                "area": "offense",
                "level": 1,
                "description": (
                    f"{ops_trailing_name} 팀 OPS 열세로 "
                    + (
                        "득점 연결 효율이 결과에 영향을 줬습니다."
                        if review_mode
                        else "초반 득점 설계가 막히면 추격 부담이 커집니다."
                    )
                ),
            }
        )
    risks = _ensure_minimum_risks(
        risks,
        candidates=min_risk_candidates,
        fallback={
            "area": "overall",
            "level": 1 if det_edge_team else 2,
            "description": (
                "확인된 실데이터 기준 두드러진 리스크는 적지만, 승부처 대응과 득점 연결 효율이 결과를 갈랐습니다."
                if review_mode
                else "확인된 지표상 결정적 리스크는 낮지만, 초반 선취점과 첫 불펜 선택이 경기 흐름을 좌우할 변수입니다."
            ),
        },
        minimum=1,
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

        snapshot_lines = _build_scheduled_team_level_snapshot_lines(
            evidence, tool_results
        )
        if snapshot_lines:
            sections.append("## 경기 전 스냅샷")
            sections.extend(snapshot_lines)

        if resolved_focus:
            for focus in resolved_focus:
                header = _focus_section_header(focus, evidence.game_status_bucket)
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

    if _normalize_game_status_bucket(evidence.game_status_bucket) == "SCHEDULED":
        snapshot_lines = _build_scheduled_team_level_snapshot_lines(
            evidence, tool_results
        )
        if snapshot_lines:
            sections.append("## 경기 전 스냅샷")
            sections.extend(snapshot_lines)

    if resolved_focus:
        for focus in resolved_focus:
            header = _focus_section_header(focus, evidence.game_status_bucket)
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
    game_status_bucket = evidence.game_status_bucket if evidence is not None else None
    for focus in resolved_focus or []:
        header = _focus_section_header(focus, game_status_bucket)
        if not header or _markdown_section_has_body(existing_md, header):
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

    game_status_bucket = evidence.game_status_bucket if evidence is not None else None
    for focus in resolved_focus or []:
        header = _focus_section_header(focus, game_status_bucket)
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
        header = _focus_section_header(focus, evidence.game_status_bucket)
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

    # 빈 경우에만 결정론 template 로 백필(기존 값 보존). risks 가 비면 프론트의
    # 리스크 관리 섹션이 빈 상태 박스로만 노출되므로 최소 1건을 보장한다.
    for key, limit in (("risks", 2), ("strengths", 4), ("weaknesses", 3)):
        existing = analysis.get(key)
        if not isinstance(existing, list) or len(existing) == 0:
            template_items = template.get(key)
            if isinstance(template_items, list) and template_items:
                analysis[key] = list(template_items[:limit])

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
    tool_results: Optional[Dict[str, Any]] = None,
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
    if tool_results and "## 경기 전 스냅샷" not in updated:
        snapshot_lines = _build_scheduled_team_level_snapshot_lines(
            evidence, tool_results
        )
        if snapshot_lines:
            snapshot = "## 경기 전 스냅샷\n" + "\n".join(snapshot_lines)
            updated = (
                f"{snapshot}\n\n{updated.strip()}" if updated.strip() else snapshot
            )
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


def _ensure_scheduled_snapshot_section(
    markdown: str,
    *,
    evidence: Optional[GameEvidence],
    tool_results: Optional[Dict[str, Any]],
) -> str:
    if (
        not evidence
        or evidence.game_status_bucket != "SCHEDULED"
        or not tool_results
        or "## 경기 전 스냅샷" in str(markdown or "")
    ):
        return str(markdown or "")

    snapshot_lines = _build_scheduled_team_level_snapshot_lines(evidence, tool_results)
    if not snapshot_lines:
        return str(markdown or "")

    snapshot = "## 경기 전 스냅샷\n" + "\n".join(snapshot_lines)
    existing = str(markdown or "").strip()
    updated = f"{snapshot}\n\n{existing}" if existing else snapshot
    return _normalize_detailed_markdown_layout(updated)


def _ensure_detailed_markdown(
    response_payload: Dict[str, Any],
    resolved_focus: Optional[List[str]] = None,
    grounding_warnings: Optional[List[str]] = None,
    evidence: Optional[GameEvidence] = None,
    tool_results: Optional[Dict[str, Any]] = None,
) -> None:
    """Ensure detailed_markdown contains all focus section headers."""
    existing_md = _normalize_detailed_markdown_layout(
        str(response_payload.get("detailed_markdown") or "")
    )
    response_payload["detailed_markdown"] = existing_md
    existing_md = _normalize_completed_review_focus_sections(
        existing_md,
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
    existing_md = _ensure_scheduled_snapshot_section(
        existing_md,
        evidence=evidence,
        tool_results=tool_results,
    )
    existing_md = _normalize_detailed_markdown_layout(existing_md)
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
        game_status_bucket = (
            evidence.game_status_bucket if evidence is not None else None
        )
        for focus in resolved_focus or []:
            header = _focus_section_header(focus, game_status_bucket)
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
        tool_results=tool_results,
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
    ("대구개막", "대구 개막"),
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
    ("WPA/PA", "운영 지표"),
    ("선발 대결 분석 불가능합니다", "선발 맞대결은 발표 이후 보강해야 합니다"),
    ("선발 대결 분석 불가능", "선발 맞대결은 발표 이후 보강 필요"),
    ("선발 정보가 완전히 확정되지 않아", "공식 선발 발표 전이라"),
    ("선발 정보가 확정되지 않아", "공식 선발 발표 전이라"),
    ("선발 정보가 완전히 확정되지 않았습니다", "공식 선발 발표 전입니다"),
    ("선발 정보가 확정되지 않았습니다", "공식 선발 발표 전입니다"),
    ("선발 미확정으로", "공식 선발 발표 전이라"),
    ("선발 미발표로", "공식 선발 발표 전이라"),
    ("선발 미확정", "공식 선발 발표 전"),
    ("선발 미발표", "공식 선발 발표 전"),
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
    ("불펜 비중 데이터가 결여되어", "불펜 운용 근거가 제한돼"),
    ("불펜 비중이 데이터 부족으로", "불펜 운용 근거가 제한돼"),
    ("불펜 비중이 데이터 부족", "불펜 운용 근거가 제한적"),
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
    ("대구개막", "대구 개막"),
    ("오프ensive", "공격"),
    ("offensive", "공격"),
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
    ("긍분위기", "긍정적인 분위기"),
    ("승부를 갈릴 수", "승부가 갈릴 수"),
    ("핵심 선수 가능성", "주도 가능성"),
    ("핵심 선수 미공개", "핵심 전력 정보 미공개"),
    ("핵심 선수 예측 가능 여부", "핵심 전력 윤곽 확인 필요"),
    ("핵심 선수 예측", "핵심 전력 윤곽 판단"),
    ("핵심 선수 여부", "핵심 전력 여부"),
    ("불펜으로 핵심 선수는", "불펜 운용은"),
    ("불펜으로 핵심 선수가", "불펜 운용이"),
    ("불펜으로 핵심 선수은", "불펜 운용은"),
    ("불펜으로 핵심 선수을", "불펜 운용을"),
    ("불펜으로 핵심 선수를", "불펜 운용을"),
    ("불펜으로 핵심 선수", "불펜 운용"),
    ("불펜 미정형", "불펜 변수"),
    ("경기 후반 접전 후반 상황", "경기 후반 접전 상황"),
    ("핵심 구간는", "핵심 구간은"),
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
    (
        re.compile(
            r"([가-힣]*?(?:랜더스|이글스|라이온즈|트윈스|베어스|위즈|다이노스|타이거즈|자이언츠|히어로즈|드래곤즈|이글|트윈|베어|위자드|다이노|타이거|자이언트|히어로))은(?=\s)"
        ),
        r"\1는",
    ),
    (
        re.compile(
            r"([A-Za-z]{2,5}\s+[가-힣]*?(?:랜더스|이글스|라이온즈|트윈스|베어스|위즈|다이노스|타이거즈|자이언츠|히어로즈|드래곤즈|이글|트윈|베어|위자드|다이노|타이거|자이언트|히어로))을(?=\s)"
        ),
        r"\1를",
    ),
    (
        re.compile(
            r"([가-힣]*?(?:랜더스|이글스|라이온즈|트윈스|베어스|위즈|다이노스|타이거즈|자이언츠|히어로즈|드래곤즈|이글|트윈|베어|위자드|다이노|타이거|자이언트|히어로))을(?=\s)"
        ),
        r"\1를",
    ),
    (
        re.compile(
            r"([A-Za-z]{2,5}\s+[가-힣]*?(?:랜더스|이글스|라이온즈|트윈스|베어스|위즈|다이노스|타이거즈|자이언츠|히어로즈|드래곤즈|이글|트윈|베어|위자드|다이노|타이거|자이언트|히어로))과(?=\s)"
        ),
        r"\1와",
    ),
    (
        re.compile(
            r"([가-힣]*?(?:랜더스|이글스|라이온즈|트윈스|베어스|위즈|다이노스|타이거즈|자이언츠|히어로즈|드래곤즈|이글|트윈|베어|위자드|다이노|타이거|자이언트|히어로))과(?=\s)"
        ),
        r"\1와",
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
# 예정 경기 응답에서 '이미 끝난 경기'처럼 결과를 단정하는 표현(정규화 후 형식 변형 무관 매칭).
# 결과-프레이밍에 한정 → 일반 과거시제·최근 폼 서술은 매칭하지 않음(오탐 억제).
SCHEDULED_REVIEW_PATTERNS = (
    ("결과진단", re.compile(r"결과\s*진단")),
    ("결과를가른", re.compile(r"결과\s*를?\s*가른")),
    ("실제전환점", re.compile(r"실제\s*전환점")),
    ("다시볼장면", re.compile(r"다시\s*볼\s*장면")),
    ("우위확정", re.compile(r"우위\s*확정")),
    ("승패리뷰", re.compile(r"(승리|패배)\s*리뷰")),
)
# 양 선발이 확정됐는데도 '미확정/불가' 류 모순 표현(선발 정보 충돌).
SCHEDULED_STARTER_CONFLICT_PATTERNS = (
    ("확정되지", re.compile(r"선발\s*정보가?\s*(완전히\s*)?확정되지")),
    ("미확정", re.compile(r"선발\s*[가-힣]?\s*미확정")),
    ("미발표", re.compile(r"선발\s*[가-힣]?\s*미발표")),
    ("분석불가", re.compile(r"선발\s*(대결\s*)?분석\s*불가")),
    ("평가제한", re.compile(r"선발\s*맞?대결.{0,6}평가.{0,6}제한")),
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


def _normalize_guard_text(response_payload: Dict[str, Any]) -> str:
    """가드 매칭용 정규화: 응답 텍스트 세그먼트를 모아 마크다운 마커/구두점 제거 + 공백 축약.

    json.dumps(줄바꿈 escape·JSON 키 포함) 대신 실제 텍스트만 대상으로 해, 헤더 레벨/공백/
    구두점 변형과 무관하게 결과-프레이밍 표현을 잡는다.
    """
    text = " ".join(_collect_response_text_segments(response_payload or {}))
    text = re.sub(r"[#*>`~_\-]+", " ", text)
    text = re.sub(r"[.,!?;:()\[\]\"'“”‘’·]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _scheduled_output_guard_reasons(
    response_payload: Dict[str, Any],
    evidence: GameEvidence,
) -> List[str]:
    if str(evidence.game_status_bucket or "").upper() != "SCHEDULED":
        return []

    normalized = _normalize_guard_text(response_payload)
    reasons: List[str] = []
    for name, pattern in SCHEDULED_REVIEW_PATTERNS:
        if pattern.search(normalized):
            reasons.append(f"review_marker:{name}")

    if evidence.home_pitcher and evidence.away_pitcher:
        for name, pattern in SCHEDULED_STARTER_CONFLICT_PATTERNS:
            if pattern.search(normalized):
                reasons.append(f"starter_conflict:{name}")

    return list(dict.fromkeys(reasons))


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
    if str(game_status_bucket or "").upper() != "SCHEDULED":
        return response_data

    polished = deepcopy(response_data)

    def _polish_text(value: Any, *, markdown: bool = False) -> Any:
        if not isinstance(value, str):
            return value
        updated = value
        for source, replacement in SCHEDULED_REVIEW_LANGUAGE_REPLACEMENTS:
            updated = updated.replace(source, replacement)
        for source, replacement in SCHEDULED_USER_FACING_REPLACEMENTS:
            updated = updated.replace(source, replacement)
        for source, replacement in RESPONSE_LANGUAGE_QUALITY_REPLACEMENTS:
            updated = updated.replace(source, replacement)
        for pattern, replacement in RESPONSE_LANGUAGE_QUALITY_REGEX_REPLACEMENTS:
            updated = pattern.sub(replacement, updated)
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
    for header in _all_focus_section_headers():
        updated = re.sub(
            rf"(?m)^\s*[-*]\s*\*{{1,2}}\s*\n+\s*({re.escape(header)})\*{{1,2}}\s*:\s*(.+?)\s*$",
            r"\1\n- \2",
            updated,
        )
        updated = re.sub(
            rf"(?m)^\s*[-*]\s*\*{{0,2}}\s*({re.escape(header)})\*{{1,2}}\s*:\s*(.+?)\s*$",
            r"\1\n- \2",
            updated,
        )
        updated = re.sub(
            rf"(?m)^({re.escape(header)})\*{{1,2}}\s*:\s*(.+?)\s*$",
            r"\1\n- \2",
            updated,
        )
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


_coach_fanout_semaphore: Optional[asyncio.Semaphore] = None


def _get_coach_fanout_semaphore() -> asyncio.Semaphore:
    """Coach 도구 fan-out DB 체크아웃을 제한하는 lazy 세마포어.

    모듈 import 시점에는 실행 중인 이벤트 루프가 없으므로 첫 사용 시 생성한다.
    """
    global _coach_fanout_semaphore
    if _coach_fanout_semaphore is None:
        _coach_fanout_semaphore = asyncio.Semaphore(COACH_DB_FANOUT_MAX)
    return _coach_fanout_semaphore


def _reset_coach_fanout_semaphore_for_tests() -> None:
    """테스트 격리용. 세마포어 싱글톤을 초기화한다."""
    global _coach_fanout_semaphore
    _coach_fanout_semaphore = None


async def _execute_coach_tools_parallel(
    pool: AsyncConnectionPool,
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
    tool_timings: Dict[str, float] = {}

    def _timed(name: str, func):
        async def _wrapper(*args, **kwargs):
            started = perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                tool_timings[name] = perf_counter() - started

        return _wrapper

    async def get_team_data(team_code: str):
        """특정 팀의 모든 데이터 조회 — 4쿼리 병렬 실행"""
        degraded_state: List[Optional[str]] = [None]  # [reason] or [None]

        async def _get_summary():
            async with _get_coach_fanout_semaphore(), pool.connection() as conn:
                db_query = DatabaseQueryTool(conn)
                result = await db_query.get_team_summary(team_code, year)
                if getattr(db_query, "mapping_dependency_degraded", False):
                    degraded_state[0] = (
                        getattr(db_query, "mapping_dependency_reason", None)
                        or "defaults"
                    )
                return result

        async def _get_advanced():
            async with _get_coach_fanout_semaphore(), pool.connection() as conn:
                return await DatabaseQueryTool(conn).get_team_advanced_metrics(
                    team_code, year
                )

        async def _get_form_signals():
            async with _get_coach_fanout_semaphore(), pool.connection() as conn:
                return await DatabaseQueryTool(conn).get_team_player_form_signals(
                    team_code,
                    year,
                    as_of_game_date=as_of_game_date,
                    exclude_game_id=exclude_game_id,
                )

        async def _get_recent():
            if "recent_form" not in focus and focus:
                return None
            async with _get_coach_fanout_semaphore(), pool.connection() as conn:
                return await DatabaseQueryTool(conn).get_team_recent_form(
                    team_code,
                    year,
                    as_of_game_date=as_of_game_date,
                    exclude_game_id=exclude_game_id,
                )

        summary, advanced, form_signals, recent = await asyncio.gather(
            _get_summary(),
            _get_advanced(),
            _get_form_signals(),
            _get_recent(),
            return_exceptions=True,
        )

        results: Dict[str, Any] = {}
        for key, val in [
            ("summary", summary),
            ("advanced", advanced),
            ("player_form_signals", form_signals),
        ]:
            if not isinstance(val, BaseException):
                results[key] = val
            else:
                logger.warning("[CoachData] %s failed for %s: %s", key, team_code, val)

        if recent is not None and not isinstance(recent, BaseException):
            results["recent"] = recent
        elif isinstance(recent, BaseException):
            logger.warning(
                "[CoachData] recent_form failed for %s: %s", team_code, recent
            )

        if degraded_state[0] is not None:
            results["_dependency_degraded"] = True
            results["_dependency_degraded_reason"] = degraded_state[0]

        return results

    async def get_clutch_moments_sync(target_game_id: str):
        async with _get_coach_fanout_semaphore(), pool.connection() as conn:
            db_query = DatabaseQueryTool(conn)
            return await db_query.get_clutch_moments(target_game_id, limit=3)

    async def get_matchup_stats_sync(team1: str, team2: str):
        async with _get_coach_fanout_semaphore(), pool.connection() as conn:
            from app.tools.game_query import GameQueryTool

            game_query = GameQueryTool(conn)
            result = await game_query.get_head_to_head(
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
    gather_started = perf_counter()

    # 1. 홈팀 데이터
    tasks.append(_timed("home_team", get_team_data)(home_team_id))

    # 2. 원정팀 데이터 (있을 경우)
    if away_team_id:
        tasks.append(_timed("away_team", get_team_data)(away_team_id))

    # 3. 상대 전적 (Matchup focus일 경우)
    if "matchup" in focus and away_team_id:
        tasks.append(
            _timed("matchup", get_matchup_stats_sync)(home_team_id, away_team_id)
        )
    if include_clutch and game_id:
        tasks.append(_timed("clutch_moments", get_clutch_moments_sync)(game_id))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    gather_elapsed = perf_counter() - gather_started

    try:
        timing_fields = " ".join(
            f"{name}={duration:.3f}s" for name, duration in sorted(tool_timings.items())
        )
        logger.info(
            "[CoachToolTiming] game_id=%s focus=%s wall=%.3fs %s",
            game_id or "-",
            ",".join(focus) if focus else "-",
            gather_elapsed,
            timing_fields,
        )
    except Exception:  # noqa: BLE001 — 로그 실패가 본 흐름을 막지 않게
        logger.debug("CoachToolTiming log emission failed", exc_info=True)

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

    # Coach 페이로드 압축: 도구 결과 dict의 미사용 필드 제거 + top_n 절단
    # 환경변수 COACH_PAYLOAD_COMPRESSION_ENABLED=0으로 옵트아웃 가능
    if _coach_payload_compression_enabled():
        try:
            top_n = max(1, _coach_payload_top_n())
            for team_key, team_id_value in (
                ("home", home_team_id),
                ("away", away_team_id),
            ):
                team_data = tool_results.get(team_key) or {}
                # error-only dict는 그대로 유지 (압축할 데이터 없음)
                if not team_data or list(team_data.keys()) == ["error"]:
                    continue
                if not isinstance(team_data, dict):
                    continue
                try:
                    payload = CoachTeamPayload.from_team_data_dict(
                        team_data,
                        team_id=str(team_id_value or team_key),
                        team_name_fallback=str(team_id_value or team_key),
                        top_n=top_n,
                        form_signals_top_n=1,
                    )
                    compressed = payload.to_factsheet_dict()
                    # _dependency_degraded 플래그는 보존
                    if team_data.get("_dependency_degraded"):
                        compressed["_dependency_degraded"] = team_data[
                            "_dependency_degraded"
                        ]
                        compressed["_dependency_degraded_reason"] = team_data.get(
                            "_dependency_degraded_reason"
                        )
                    tool_results[team_key] = compressed
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[CoachPayloadCompress] %s compression failed: %s — keeping original",
                        team_key,
                        exc,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CoachPayloadCompress] compression skipped: %s", exc)

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


def _build_coach_dynamic_prompt(
    *,
    question: str,
    context: str,
    focus_section_requirements: str,
) -> str:
    return COACH_PROMPT_V2_DYNAMIC_TEMPLATE.format(
        question=question,
        context=context,
        focus_section_requirements=focus_section_requirements,
    )


def _build_coach_retry_dynamic_prompt(
    *,
    question: str,
    context: str,
    focus_section_requirements: str,
    previous_response: str,
    failure_reasons: List[str],
) -> str:
    dynamic = _build_coach_dynamic_prompt(
        question=question,
        context=context,
        focus_section_requirements=focus_section_requirements,
    )
    reason_text = (
        ", ".join(failure_reasons[:4]) if failure_reasons else "근거 검증 실패"
    )
    previous_excerpt = previous_response.strip()[:900]
    return (
        f"{dynamic}\n\n"
        "## 재작성 지시\n"
        f"- 직전 응답은 다음 사유로 폐기되었습니다: {reason_text}\n"
        "- FACT SHEET에 없는 선수명, 숫자, 시리즈 맥락은 절대 넣지 마세요.\n"
        "- `analysis.verdict`와 `analysis.why_it_matters`를 반드시 채우고, 숫자와 경기 운영 의미를 직접 연결하세요.\n"
        "- `analysis.verdict`, `analysis.swing_factors`, `coach_note`에 같은 문장이나 상투 표현을 반복하지 마세요.\n"
        "- 단순 지표 나열 대신 어떤 지표가 왜 승부처인지 판단 문장으로 정리하세요.\n"
        "- 아래 이전 응답의 잘못된 표현은 반복하지 말고, 필요한 사실만 남겨 새 JSON을 작성하세요.\n"
        f"### 직전 응답\n{previous_excerpt}"
    )


def _build_coach_retry_prompt(
    *,
    question: str,
    context: str,
    focus_section_requirements: str,
    previous_response: str,
    failure_reasons: List[str],
) -> str:
    dynamic_with_retry = _build_coach_retry_dynamic_prompt(
        question=question,
        context=context,
        focus_section_requirements=focus_section_requirements,
        previous_response=previous_response,
        failure_reasons=failure_reasons,
    )
    return COACH_PROMPT_V2_STATIC + dynamic_with_retry


def _build_coach_llm_messages(
    static_text: str,
    dynamic_text: str,
) -> List[Dict[str, Any]]:
    """Construct OpenRouter messages, applying cache_control on the static
    prefix when enabled and the static chunk is long enough to be worth caching.

    The concatenation of the returned blocks is byte-identical to
    ``static_text + dynamic_text``; upstream providers that don't support
    ``cache_control`` simply ignore the marker.
    """
    if (
        _coach_prompt_cache_enabled()
        and len(static_text) >= _coach_prompt_cache_min_static_chars()
    ):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": static_text,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": dynamic_text},
                ],
            }
        ]
    return [{"role": "user", "content": static_text + dynamic_text}]


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


AnalyzeRequest = CoachAnalyzeRequest


class CoachCacheResetRequest(BaseModel):
    """``POST /coach/cache/reset`` 입력. cache_key 단건 또는 team_id+year 일괄."""

    cache_key: Optional[str] = None
    team_id: Optional[str] = None
    year: Optional[int] = None
    include_stale_pending: bool = True
    retryable_only: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, values: Any) -> Any:
        if isinstance(values, dict):
            for key in ("cache_key", "team_id"):
                value = values.get(key)
                if isinstance(value, str):
                    values[key] = value.strip() or None
        return values


@router.post("/cache/reset")
async def reset_coach_cache(
    payload: CoachCacheResetRequest,
    _: None = Depends(require_ai_internal_token),
):
    """
    FAILED_LOCKED 상태의 Coach 분석 캐시를 운영자가 수동으로 즉시 복구한다.

    삭제된 row는 다음 분석 요청에서 자연 재생성된다. 활성 PENDING lease와
    COMPLETED row는 보호된다. cache_key 단건 또는 (team_id, year) 묶음을 받는다.
    """
    if not payload.cache_key and not (payload.team_id and payload.year is not None):
        raise HTTPException(
            status_code=400,
            detail="cache_key 또는 (team_id, year) 조합이 필요합니다.",
        )

    team_id_canonical: Optional[str] = None
    year = payload.year
    if payload.cache_key is None:
        if year is None or not _is_valid_analysis_year(year):
            raise HTTPException(status_code=400, detail="analysis_year_out_of_range")
        resolver = TeamCodeResolver()
        team_id_canonical = resolver.resolve_canonical(payload.team_id)
        if team_id_canonical not in CANONICAL_CODES:
            raise HTTPException(
                status_code=400,
                detail="unsupported_team_for_regular_analysis",
            )

    pool = get_connection_pool()
    stats = await _reset_failed_coach_cache_rows(
        pool,
        cache_key=payload.cache_key,
        team_id=team_id_canonical,
        year=year,
        include_stale_pending=payload.include_stale_pending,
        retryable_only=payload.retryable_only,
    )
    logger.info(
        "[Coach] cache reset cache_key=%s team_id=%s year=%s stats=%s",
        payload.cache_key,
        team_id_canonical,
        year,
        stats,
    )
    return {"status": "ok", "meta": stats}


@router.post("/analyze")
async def analyze_team(
    payload: AnalyzeRequest,
    agent: BaseballStatisticsAgent = Depends(get_agent),
    __: None = Depends(rate_limit_coach_dependency),
    _: None = Depends(require_ai_internal_token),
    event_version_header: str | None = Header(
        default=None,
        alias="X-AI-Event-Version",
    ),
):
    """
    특정 팀(들)에 대한 심층 분석을 요청합니다. 'The Coach' 페르소나가 적용됩니다.
    """
    event_version = negotiate_event_version(
        event_version_header,
        endpoint="coach",
    )

    route_prepare_started = perf_counter()

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

        year, resolve_source = await _resolve_target_year(payload, pool)
        if not _is_valid_analysis_year(year):
            raise HTTPException(status_code=400, detail="analysis_year_out_of_range")

        home_name = agent._convert_team_id_to_name(payload.home_team_id)
        away_name = (
            agent._convert_team_id_to_name(payload.away_team_id)
            if payload.away_team_id
            else None
        )
        game_evidence = await _collect_game_evidence(
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
        analysis_type = _resolve_analysis_type(payload, game_evidence)
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
                analysis_type=analysis_type,
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
            "analysis_type": analysis_type,
        }
        focus_signature = str(cache_key_payload["focus_signature"])
        question_signature = str(cache_key_payload["question_signature"])

        logger.info(
            "[Coach Router] request_mode=%s analysis_type=%s Analyzing %s vs %s (year=%d game_id=%s stage=%s status=%s): %s... "
            "(CacheKey: %s) expected_cache_key=%s prompt_version=%s starter_signature=%s lineup_signature=%s "
            "input_season=%s resolved_year=%d resolve_source=%s input_focus=%s resolved_focus=%s "
            "focus_signature=%s question_signature=%s cache_key_version=%s",
            request_mode,
            analysis_type,
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
        evidence_assessment = assess_game_evidence(
            game_evidence,
            analysis_type=analysis_type,
        )
        manual_data_request = await _build_manual_data_request(
            pool,
            payload,
            game_evidence,
            evidence_assessment,
            analysis_type=analysis_type,
        )
        unavailable_game_status_message = _get_unavailable_game_status_message(
            getattr(game_evidence, "game_status", None)
        )
        prepare_elapsed_sec = perf_counter() - route_prepare_started
        log_prepare = logger.warning if prepare_elapsed_sec >= 25.0 else logger.info
        log_prepare(
            "[Coach] Stream prepared game_id=%s cache_key=%s request_mode=%s analysis_type=%s "
            "game_status_bucket=%s prepare_elapsed_sec=%.2f manual_data_required=%s "
            "unavailable_status=%s",
            game_evidence.game_id,
            cache_key,
            request_mode,
            analysis_type,
            game_evidence.game_status_bucket,
            prepare_elapsed_sec,
            manual_data_request is not None,
            unavailable_game_status_message is not None,
        )

        async def event_generator():
            try:
                total_start = perf_counter()
                latency_tracker = _CoachLatencyTracker(request_started=total_start)

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
                        "analysis_type": analysis_type,
                        "resolved_focus": resolved_focus,
                        "focus_signature": focus_signature,
                        "question_signature": question_signature,
                        "cache_key_version": COACH_CACHE_SCHEMA_VERSION,
                        **cache_contract_meta,
                    }
                    if extra_payload:
                        payload_data.update(extra_payload)
                    if "llm_skip_reason" not in payload_data:
                        llm_skip_reason = _resolve_llm_skip_reason(
                            request_mode=str(
                                payload_data.get("request_mode") or request_mode
                            ),
                            generation_mode=(
                                str(payload_data.get("generation_mode"))
                                if payload_data.get("generation_mode") is not None
                                else None
                            ),
                            cache_state=(
                                str(payload_data.get("cache_state"))
                                if payload_data.get("cache_state") is not None
                                else None
                            ),
                            cached=payload_data.get("cached") is True,
                            in_progress=payload_data.get("in_progress") is True,
                            manual_data_required=(
                                payload_data.get("manual_data_request") is not None
                                or payload_data.get("validation_status")
                                == "manual_data_required"
                            ),
                        )
                        if llm_skip_reason:
                            payload_data["llm_skip_reason"] = llm_skip_reason
                    payload_analysis_type = (
                        _normalize_analysis_type(payload_data.get("analysis_type"))
                        or analysis_type
                    )
                    payload_data["analysis_type"] = payload_analysis_type
                    structured_response = payload_data.get("structured_response")
                    if isinstance(structured_response, dict):
                        enriched_response = dict(structured_response)
                        enriched_response.setdefault(
                            "analysisType", payload_analysis_type
                        )
                        enriched_response.setdefault(
                            "analysis_type", payload_analysis_type
                        )
                        payload_data["structured_response"] = enriched_response
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

                completed_ttl_seconds = _cache_ttl_seconds_for_status_bucket(
                    game_evidence.game_status_bucket
                )
                if manual_data_request is not None:
                    (
                        cache_state,
                        cached_data,
                        cache_error_message,
                        cache_error_code,
                        cache_attempt_count,
                    ) = await _read_completed_cache_if_fresh(
                        pool=pool,
                        cache_key=cache_key,
                        completed_ttl_seconds=completed_ttl_seconds,
                        request_mode=request_mode,
                        expected_data_quality=evidence_assessment.expected_data_quality,
                        expected_used_evidence=evidence_assessment.used_evidence,
                        expected_game_status_bucket=game_evidence.game_status_bucket,
                        current_root_causes=evidence_assessment.root_causes,
                    )
                    if not cached_data:
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
                                        "cache_state": cache_state,
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
                else:
                    (
                        cache_state,
                        cached_data,
                        cache_error_message,
                        cache_error_code,
                        cache_attempt_count,
                    ) = await _claim_cache_generation(
                        pool=pool,
                        cache_key=cache_key,
                        team_id=home_team_canonical,
                        year=year,
                        prompt_version=COACH_CACHE_PROMPT_VERSION,
                        model_name=coach_model_name,
                        lease_owner=cache_lease_owner,
                        completed_ttl_seconds=completed_ttl_seconds,
                        request_mode=request_mode,
                        expected_data_quality=evidence_assessment.expected_data_quality,
                        expected_used_evidence=evidence_assessment.used_evidence,
                        expected_game_status_bucket=game_evidence.game_status_bucket,
                        current_root_causes=evidence_assessment.root_causes,
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
                        analysis_type=analysis_type,
                    )
                    cached_meta = _merge_cache_contract_meta(
                        cached_meta,
                        cache_contract_meta,
                    )
                    missing_focus_sections = _find_missing_focus_sections(
                        _with_focus_status_context(
                            cached_data,
                            game_evidence.game_status_bucket,
                        ),
                        resolved_focus,
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

                # [AutoBriefBG] auto_brief MISS_GENERATE → 백그라운드 생성 후 즉시 PENDING 반환
                if _should_generate_from_gate(cache_state) and is_auto_brief:
                    auto_brief_data_quality = _determine_data_quality(
                        game_evidence,
                        analysis_type=analysis_type,
                    )
                    logger.info(
                        "[AutoBriefBG] Cache miss generation start game_id=%s "
                        "cache_key=%s cache_state=%s error_code=%s data_quality=%s",
                        game_evidence.game_id,
                        cache_key,
                        cache_state,
                        cache_error_code,
                        auto_brief_data_quality,
                    )
                    _schedule_auto_brief_background_task(
                        _generate_auto_brief_cache_background(
                            pool=pool,
                            cache_key=cache_key,
                            lease_owner=cache_lease_owner,
                            home_team_canonical=home_team_canonical,
                            away_team_canonical=away_team_canonical,
                            home_name=home_name,
                            away_name=away_name,
                            year=year,
                            resolved_focus=resolved_focus,
                            game_evidence=game_evidence,
                            evidence_assessment=evidence_assessment,
                            analysis_type=analysis_type,
                            coach_model_name=coach_model_name,
                            cache_attempt_count=cache_attempt_count,
                        )
                    )
                    bg_pending_payload = _cache_status_response(
                        headline=f"{home_name} 브리핑을 준비하고 있습니다",
                        coach_note="잠시 후 자동으로 업데이트됩니다.",
                        detail=(
                            "## 브리핑 생성 중\n\n"
                            "경기 데이터를 수집하고 분석 중입니다. 잠시 후 다시 시도해 주세요."
                        ),
                    )
                    bg_meta = _build_meta_payload_defaults(
                        generation_mode="evidence_fallback",
                        data_quality=auto_brief_data_quality,
                        used_evidence=game_evidence.used_evidence,
                        **cache_contract_meta,
                        game_status_bucket=game_evidence.game_status_bucket,
                        grounding_warnings=[],
                        grounding_reasons=evidence_assessment.root_causes,
                        supported_fact_count=0,
                        attempt_count=cache_attempt_count,
                    )
                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {"message": "브리핑 생성을 시작합니다..."},
                            ensure_ascii=False,
                        ),
                    }
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "delta": json.dumps(
                                    bg_pending_payload, ensure_ascii=False
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
                                    "cache_state": "PENDING_WAIT",
                                    "in_progress": True,
                                    **bg_meta,
                                }
                            ),
                            ensure_ascii=False,
                        ),
                    }
                    yield {"event": "done", "data": "[DONE]"}
                    return

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
                                    analysis_type=analysis_type,
                                )
                            )
                            cached_wait_meta = _merge_cache_contract_meta(
                                cached_wait_meta,
                                cache_contract_meta,
                            )
                            missing_focus_sections = _find_missing_focus_sections(
                                _with_focus_status_context(
                                    cached_wait_data,
                                    game_evidence.game_status_bucket,
                                ),
                                resolved_focus,
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
                            ) = await _claim_cache_generation(
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
                                expected_data_quality=evidence_assessment.expected_data_quality,
                                expected_used_evidence=evidence_assessment.used_evidence,
                                expected_game_status_bucket=game_evidence.game_status_bucket,
                                current_root_causes=evidence_assessment.root_causes,
                            )
                            if cached_data:
                                cached_data, cached_meta = _extract_cached_payload(
                                    cached_data,
                                    request_mode=request_mode,
                                    analysis_type=analysis_type,
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
                                    _with_focus_status_context(
                                        cached_data,
                                        game_evidence.game_status_bucket,
                                    ),
                                    resolved_focus,
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
                            data_quality=_determine_data_quality(
                                game_evidence,
                                analysis_type=analysis_type,
                            ),
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
                            data_quality=_determine_data_quality(
                                game_evidence,
                                analysis_type=analysis_type,
                            ),
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

                latency_tracker.mark_tool_fetch_start()
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
                latency_tracker.mark_tool_fetch_complete()
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
                data_quality = _determine_data_quality(
                    game_evidence,
                    tool_results,
                    analysis_type=analysis_type,
                )
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
                    resolved_focus=resolved_focus,
                    tool_results=tool_results,
                    evidence=game_evidence,
                ):
                    fast_path_hit = _is_manual_recent_form_fast_path_eligible(
                        request_mode, resolved_focus
                    )
                    logger.info(
                        "[Coach] Deterministic short-circuit game_id=%s cache_key=%s request_mode=%s game_status_bucket=%s grounding_reasons=%s supported_fact_count=%d fast_path=%s",
                        game_evidence.game_id,
                        cache_key,
                        request_mode,
                        game_evidence.game_status_bucket,
                        grounding_reasons,
                        fact_sheet.supported_fact_count,
                        fast_path_hit,
                    )
                    generation_mode = (
                        _generation_mode_for_analysis_type(
                            analysis_type=analysis_type,
                            request_mode=request_mode,
                        )
                        if is_auto_brief
                        else "evidence_fallback"
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
                    response_payload = _attach_response_analysis_type(
                        response_payload,
                        analysis_type,
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
                    # 사전 승률: auto_brief 뿐 아니라 manual_detail(예측 다이얼로그) 결정론 응답도 포함
                    _attach_scheduled_win_probability(
                        meta_defaults,
                        game_status_bucket=game_evidence.game_status_bucket,
                        tool_results=tool_results,
                        response_payload=response_payload,
                    )
                    finalize_result = await _store_completed_cache(
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
                        _with_focus_status_context(
                            response_payload,
                            game_evidence.game_status_bucket,
                        ),
                        response_focus_targets,
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
                    _emit_coach_latency_summary(
                        logger,
                        tracker=latency_tracker,
                        cache_key=cache_key,
                        game_id=game_evidence.game_id,
                        cache_state="COMPLETED",
                        request_mode=request_mode,
                        game_status_bucket=game_evidence.game_status_bucket,
                        generation_mode=generation_mode,
                        fast_path=fast_path_hit,
                        preview_enabled=_coach_stream_preview_enabled(),
                        cache_enabled=_coach_prompt_cache_enabled(),
                    )
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
                    llm_focus,
                    game_status_bucket=game_evidence.game_status_bucket,
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
                coach_prompt_dynamic = _build_coach_dynamic_prompt(
                    question=prompt_query,
                    context=fact_sheet_context,
                    focus_section_requirements=focus_section_requirements,
                )
                # 동적 프롬프트 사이즈 메트릭 + 로그 (압축 효과 추적)
                # ratio 3.5는 한국어 평균 char/token 추정.
                try:
                    _dyn_chars = len(coach_prompt_dynamic)
                    _ctx_chars = len(fact_sheet_context or "")
                    _est_tokens = int(_dyn_chars / 3.5)
                    _compress_label = (
                        "on" if _coach_payload_compression_enabled() else "off"
                    )
                    logger.info(
                        "[CoachPrompt] dynamic_chars=%d context_chars=%d est_tokens=%d compress=%s",
                        _dyn_chars,
                        _ctx_chars,
                        _est_tokens,
                        _compress_label,
                    )
                    AI_COACH_DYNAMIC_PROMPT_CHARS.labels(
                        compress=_compress_label
                    ).observe(_dyn_chars)
                    AI_COACH_PAYLOAD_COMPRESSION_TOTAL.labels(
                        enabled=_compress_label
                    ).inc()
                except Exception:  # noqa: BLE001
                    pass
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
                    current_prompt_dynamic = coach_prompt_dynamic
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
                        if _coach_stream_preview_enabled():
                            yield {
                                "event": "preview_reset",
                                "data": json.dumps(
                                    {"attempt": attempt},
                                    ensure_ascii=False,
                                ),
                            }
                        current_prompt_dynamic = _build_coach_retry_dynamic_prompt(
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
                    latency_tracker.mark_llm_start()
                    try:
                        llm_event_stream = _iterate_coach_llm_with_keepalive(
                            coach_llm=coach_llm,
                            messages=_build_coach_llm_messages(
                                COACH_PROMPT_V2_STATIC,
                                current_prompt_dynamic,
                            ),
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
                                    status_label = llm_event.get("status")
                                    if status_label:
                                        yield {
                                            "event": "status",
                                            "data": json.dumps(
                                                {"status": str(status_label)},
                                                ensure_ascii=False,
                                            ),
                                        }
                                    else:
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
                                chunk_text = str(llm_event.get("chunk") or "")
                                response_chunks.append(chunk_text)
                                if _coach_stream_preview_enabled() and chunk_text:
                                    latency_tracker.mark_first_preview()
                                    yield {
                                        "event": "preview_chunk",
                                        "data": json.dumps(
                                            {
                                                "text": chunk_text,
                                                "attempt": attempt,
                                            },
                                            ensure_ascii=False,
                                        ),
                                    }
                        else:
                            async with asyncio.timeout(llm_total_timeout_seconds):
                                async for llm_event in llm_event_stream:
                                    if llm_event.get("type") == "status":
                                        status_label = llm_event.get("status")
                                        if status_label:
                                            yield {
                                                "event": "status",
                                                "data": json.dumps(
                                                    {"status": str(status_label)},
                                                    ensure_ascii=False,
                                                ),
                                            }
                                        else:
                                            yield {
                                                "event": "status",
                                                "data": json.dumps(
                                                    {
                                                        "message": llm_event.get(
                                                            "message"
                                                        )
                                                        or "AI 코치가 근거를 정리하는 중입니다..."
                                                    },
                                                    ensure_ascii=False,
                                                ),
                                            }
                                        continue
                                    chunk_text = str(llm_event.get("chunk") or "")
                                    response_chunks.append(chunk_text)
                                    if _coach_stream_preview_enabled() and chunk_text:
                                        latency_tracker.mark_first_preview()
                                        yield {
                                            "event": "preview_chunk",
                                            "data": json.dumps(
                                                {
                                                    "text": chunk_text,
                                                    "attempt": attempt,
                                                },
                                                ensure_ascii=False,
                                            ),
                                        }
                        latency_tracker.mark_llm_complete()
                    except Exception as llm_exc:  # noqa: BLE001
                        latency_tracker.mark_llm_complete()
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
                            latency_tracker.mark_grounding_start()
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
                    # grounding 실패로 fallback 됐으면 grounded 배지는 과신 → 한 단계 강등.
                    data_quality = _downgrade_data_quality_after_failed_grounding(
                        data_quality, last_failure_reasons
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
                scheduled_guard_reasons = _scheduled_output_guard_reasons(
                    response_payload,
                    game_evidence,
                )
                if scheduled_guard_reasons:
                    logger.warning(
                        "[Coach] Scheduled output guard fallback game_id=%s cache_key=%s reasons=%s",
                        game_evidence.game_id,
                        cache_key,
                        scheduled_guard_reasons,
                    )
                    runtime_grounding_reasons = _normalize_grounding_reasons(
                        runtime_grounding_reasons + ["scheduled_output_guard_fallback"]
                    )
                    runtime_grounding_warnings = _merge_grounding_warnings(
                        runtime_grounding_warnings,
                        runtime_grounding_reasons,
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
                    generation_mode = "evidence_fallback"
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
                response_payload = _attach_response_analysis_type(
                    response_payload,
                    analysis_type,
                )
                response_json = json.dumps(response_payload, ensure_ascii=False)
                missing_focus_sections = _find_missing_focus_sections(
                    _with_focus_status_context(
                        response_payload,
                        game_evidence.game_status_bucket,
                    ),
                    response_focus_targets,
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
                # 사전 승률: SCHEDULED 예측은 LLM/결정론 fallback 모두 meta 에 승률을 채운다.
                _attach_scheduled_win_probability(
                    meta_defaults,
                    game_status_bucket=game_evidence.game_status_bucket,
                    tool_results=tool_results,
                    response_payload=response_payload,
                )
                finalize_result = await _store_completed_cache(
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
                latency_tracker.mark_grounding_complete()
                _emit_coach_latency_summary(
                    logger,
                    tracker=latency_tracker,
                    cache_key=cache_key,
                    game_id=game_evidence.game_id,
                    cache_state=meta_payload.get("cache_state"),
                    request_mode=request_mode,
                    game_status_bucket=game_evidence.game_status_bucket,
                    generation_mode=generation_mode,
                    fast_path=_is_manual_recent_form_fast_path_eligible(
                        request_mode, resolved_focus
                    ),
                    preview_enabled=_coach_stream_preview_enabled(),
                    cache_enabled=_coach_prompt_cache_enabled(),
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
                    cancelled_store_result = await _store_failed_cache(
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
                    failed_store_result = await _store_failed_cache(
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
                    data_quality=_determine_data_quality(
                        game_evidence,
                        analysis_type=analysis_type,
                    ),
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
                        _coach_public_error_payload(
                            masked_error_code,
                            retryable=retryable_failure,
                        ),
                        ensure_ascii=False,
                    ),
                }
                yield {"event": "done", "data": "[DONE]"}
            finally:
                await _cancel_heartbeat_task(heartbeat_task)

        return versioned_event_source(
            event_generator(),
            endpoint="coach",
            version=event_version,
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
            ping=sse_ping_seconds,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "[Coach Router] Error after %.2fs: %s",
            perf_counter() - route_prepare_started,
            e,
        )
        raise HTTPException(status_code=500, detail=COACH_INTERNAL_ERROR_CODE)


# ============================================================
# Legacy endpoint (기존 방식 호환용, 기본 동작 유지)
# ============================================================


@router.post("/analyze-legacy", deprecated=True)
async def analyze_team_legacy(
    payload: AnalyzeRequest,
    agent: BaseballStatisticsAgent = Depends(get_agent),
    __: None = Depends(rate_limit_coach_dependency),
    _: None = Depends(require_ai_internal_token),
):
    """
    기존 방식의 Coach 분석 (전체 에이전트 파이프라인 사용).
    Fast Path에 문제가 있을 경우 대안으로 사용.
    deprecate: COACH_ANALYZE_LEGACY_ENABLED=0 으로 비활성화 가능.
    """
    from sse_starlette.sse import EventSourceResponse

    try:
        if not _read_flag_env("COACH_ANALYZE_LEGACY_ENABLED", "1"):
            logger.warning(
                "[Coach Router] analyze-legacy is disabled. Use /ai/coach/analyze instead."
            )
            raise HTTPException(
                status_code=410,
                detail="analyze-legacy is deprecated. Use /ai/coach/analyze.",
            )

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

        logger.warning(
            "[Coach Router Legacy] Deprecated endpoint used. home_team_id=%s request_mode=%s",
            primary_team_id,
            payload.request_mode,
        )
        logger.info(f"[Coach Router Legacy] Analyzing for {team_name}")
        sse_ping_seconds = max(1, int(get_settings().chat_sse_ping_seconds))

        context_data = {
            "persona": "coach",
            "team_id": primary_team_id,
            "legacy_endpoint": True,
        }

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
                "X-Legacy-Endpoint": "analyze-legacy",
                "X-Deprecation": "true",
            },
            ping=sse_ping_seconds,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Coach Router Legacy] Error: {e}")
        raise HTTPException(status_code=500, detail=COACH_INTERNAL_ERROR_CODE)
