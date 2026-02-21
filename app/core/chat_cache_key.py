"""
챗봇 응답 캐시 키 생성 유틸.
app/core/coach_cache_key.py 패턴을 동일하게 따름.

주요 차이점 (coach vs chat):
- coach: team_id / year / focus / question_override → 경기 분석 특화
- chat:  question / filters → 범용 RAG 응답 캐싱
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Dict, Optional, Tuple

# 캐시 스키마 버전.
# 프롬프트·정규화 방식이 변경될 때 올리면 기존 캐시가 자동 미스 처리됨.
CHAT_CACHE_SCHEMA_VERSION = "v1"

# intent별 TTL (초 단위).
# stats_lookup/comparison/recent_form은 짧게, 선수 프로필·규정 설명은 길게.
INTENT_TTL_SECONDS: Dict[str, int] = {
    "stats_lookup": 3 * 3600,  # 3h - 경기 결과는 자주 변함 (기존 6h에서 단축)
    "player_profile": 48 * 3600,  # 48h - 선수 기본 정보는 상대적으로 안정적
    "recent_form": 3 * 3600,  # 3h  - 최근 폼은 빠르게 변함
    "comparison": 3 * 3600,  # 3h  - 비교도 최신성 중요
    "general_conversation": 48 * 3600,  # 48h - 인사/일반 대화는 거의 불변
    "knowledge_explanation": 48 * 3600,  # 48h - KBO 규정·설명은 안정적
    "freeform": 12 * 3600,  # 12h - 기타 (기본값)
}
DEFAULT_TTL_SECONDS = 12 * 3600

# 실시간 키워드 감지 집합.
# 아래 키워드가 포함된 질문은 캐싱을 건너뜀 (오늘 경기, 현재 순위 등).
TEMPORAL_KEYWORDS = frozenset(
    {
        "오늘",
        "어제",
        "내일",
        "지금",
        "현재",
        "최근",
        "방금",
        "막",
        "실시간",
        "라이브",
        "순위표",
        "오늘의",
    }
)

SERIES_TEMPORAL_HINTS = frozenset(
    {
        "오늘",
        "현재",
        "실시간",
        "라이브",
        "중계",
        "현황",
        "결과",
        "일정",
        "진행",
    }
)


def _normalize_question(text: str) -> str:
    """
    캐시 키 비교용 질문 정규화.

    수행 작업:
    - NFC 유니코드 정규화 (한국어 자모 합성 차이 제거)
    - 소문자 변환
    - 연속 공백을 단일 스페이스로 축약
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_filters(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    None / 빈 문자열 / 빈 리스트 값을 제거하고 키 기준으로 정렬된 필터 딕셔너리를 반환.

    정렬이 보장되어야 같은 필터가 다른 순서로 전달되어도 동일한 키가 생성됨.
    """
    if not filters:
        return {}
    return {
        k: v
        for k, v in sorted(filters.items())
        if v is not None and v != "" and v != []
    }


def get_ttl_seconds(intent: Optional[str]) -> int:
    """intent 문자열을 기반으로 TTL(초)를 반환합니다."""
    if not intent:
        return DEFAULT_TTL_SECONDS
    return INTENT_TTL_SECONDS.get(intent, DEFAULT_TTL_SECONDS)


def has_temporal_keyword(question: str) -> bool:
    """
    실시간 데이터 요청인지 감지합니다.

    True가 반환되면 캐싱을 건너뛰어야 합니다.
    예: "오늘 경기 결과", "지금 1위 팀"
    """
    normalized = question.lower()
    if any(kw in normalized for kw in TEMPORAL_KEYWORDS):
        return True
    if "시리즈" not in normalized:
        return False
    return any(hint in normalized for hint in SERIES_TEMPORAL_HINTS)


def build_chat_cache_key(
    *,
    question: str,
    filters: Optional[Dict[str, Any]] = None,
    schema_version: str = CHAT_CACHE_SCHEMA_VERSION,
) -> Tuple[str, Dict[str, Any]]:
    """
    chat_response_cache PK용 SHA256 해시 키와 payload를 반환합니다.

    coach_cache_key.build_coach_cache_key()와 동일한 구조이지만
    team/year/focus 대신 question/filters 기반으로 해시를 생성합니다.

    Args:
        question:       사용자 원본 질문 (내부적으로 정규화됨)
        filters:        메타데이터 필터 딕셔너리 (season_year, team_id 등)
        schema_version: 캐시 스키마 버전 (기본값: CHAT_CACHE_SCHEMA_VERSION)

    Returns:
        (cache_key_hex_64chars, payload_dict)
        - cache_key: 64자 16진수 SHA256 다이제스트, chat_response_cache PK로 사용
        - payload:   해시 생성에 사용된 정규화된 값들의 딕셔너리 (디버깅/로깅용)
    """
    normalized_q = _normalize_question(question)
    normalized_f = _normalize_filters(filters)

    payload: Dict[str, Any] = {
        "schema": schema_version,
        "question": normalized_q,
        "filters": normalized_f,
    }

    # coach_cache_key와 동일하게 sort_keys=True로 직렬화하여 키 순서 독립성 보장
    cache_key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    return cache_key, payload
