"""
RAG 검색 전 쿼리를 정규화하는 모듈.

처리 대상:
- 특수문자 제거/대체
- 한영 혼합 처리 (로마자 선수명 → 한글)
- 연속 공백 정리
- 한글 자음/모음 오류 보정
"""

import re
from typing import Optional

# 로마자 → 한글 선수명 매핑 (KBO 주요 선수 영문 표기)
_ROMANIZED_PLAYER_NAMES = {
    "oh seung-hwan": "오승환",
    "oh seunghwan": "오승환",
    "ryu hyun-jin": "류현진",
    "ryu hyunjin": "류현진",
    "kim ha-seong": "김하성",
    "kim haseong": "김하성",
    "park byung-ho": "박병호",
    "park byungho": "박병호",
}

# 영문 통계 → 한글 매핑
_ENGLISH_STAT_TO_KOREAN = {
    "batter": "타자",
    "pitcher": "투수",
    "batting average": "타율",
    "home run": "홈런",
    "home runs": "홈런",
    "strikeout": "삼진",
    "strikeouts": "삼진",
    "earned run average": "방어율",
    "walks plus hits per inning pitched": "WHIP",
}

# 쿼리에서 제거할 특수문자 패턴 (구조적 의미가 없는 것들)
_NOISE_CHARS_PATTERN = re.compile(r"[!?@#$%^&*=\[\]{}\|\\\"'`]")

# 한글-영문 혼합 팀명 (영문 → DB 코드)
_ENGLISH_TEAM_TO_ID = {
    "kia tigers": "KIA",
    "kia": "KIA",
    "lg twins": "LG",
    "doosan bears": "DB",
    "lotte giants": "LT",
    "samsung lions": "SS",
    "kiwoom heroes": "KH",
    "hanwha eagles": "HH",
    "kt wiz": "KT",
    "nc dinos": "NC",
    "ssg landers": "SSG",
}


def normalize_query(text: str) -> str:
    """
    쿼리 전처리 파이프라인.
    특수문자 제거, 공백 정리, 기본 정규화를 수행합니다.

    Args:
        text: 원본 사용자 쿼리

    Returns:
        정규화된 쿼리 문자열
    """
    if not text or not isinstance(text, str):
        return ""

    result = text

    # 1. 노이즈 특수문자 제거
    result = _NOISE_CHARS_PATTERN.sub(" ", result)

    # 2. 연속 공백 → 단일 공백
    result = re.sub(r"\s+", " ", result)

    # 3. 앞뒤 공백 제거
    result = result.strip()

    return result


def normalize_special_chars(text: str) -> str:
    """
    특수문자를 정리하고 쿼리를 의미 있는 형태로 변환합니다.
    괄호, 슬래시 등은 공백으로 대체합니다.
    """
    if not text:
        return ""

    result = text
    # 괄호 내용 유지하되 괄호 자체는 제거
    result = re.sub(r"[()（）\[\]]", " ", result)
    # 슬래시, 백슬래시
    result = re.sub(r"[/\\]", " ", result)
    # 마침표가 연속된 경우
    result = re.sub(r"\.{2,}", " ", result)
    # 하이픈이 연속된 경우
    result = re.sub(r"-{2,}", " ", result)

    return re.sub(r"\s+", " ", result).strip()


def detect_language_mix(text: str) -> bool:
    """
    쿼리에 한글과 영문이 혼합되어 있는지 감지합니다.

    Returns:
        True if mixed Korean and English
    """
    if not text:
        return False

    has_korean = bool(re.search(r"[가-힣]", text))
    has_english = bool(re.search(r"[a-zA-Z]", text))

    return has_korean and has_english


def normalize_romanized_player_names(text: str) -> str:
    """
    로마자로 입력된 선수명을 한글로 변환합니다.

    Args:
        text: 쿼리 (영문 선수명 포함 가능)

    Returns:
        한글 선수명으로 대체된 쿼리
    """
    if not text:
        return text

    lower_text = text.lower()
    for romanized, korean in sorted(
        _ROMANIZED_PLAYER_NAMES.items(), key=lambda x: -len(x[0])
    ):
        if romanized in lower_text:
            # 대소문자 무관하게 교체
            pattern = re.compile(re.escape(romanized), re.IGNORECASE)
            text = pattern.sub(korean, text)

    return text


def normalize_english_stats(text: str) -> str:
    """
    영문 통계 용어를 한글/표준 형태로 변환합니다.

    Args:
        text: 쿼리

    Returns:
        통계 용어가 정규화된 쿼리
    """
    if not text:
        return text

    lower_text = text.lower()
    for english, korean in sorted(
        _ENGLISH_STAT_TO_KOREAN.items(), key=lambda x: -len(x[0])
    ):
        if english in lower_text:
            pattern = re.compile(re.escape(english), re.IGNORECASE)
            text = pattern.sub(korean, text)

    return text


def correct_spacing(text: str) -> str:
    """
    한글과 영문 사이에 공백을 추가합니다.
    예: "KIA타자" → "KIA 타자"

    Args:
        text: 쿼리

    Returns:
        공백이 추가된 쿼리
    """
    if not text:
        return text

    # 영문자 다음 한글
    result = re.sub(r"([a-zA-Z])([가-힣])", r"\1 \2", text)
    # 한글 다음 영문자
    result = re.sub(r"([가-힣])([a-zA-Z])", r"\1 \2", result)
    # 숫자 다음 한글 (예: "2024년" 제외 처리 어려우므로 스킵)

    return result


def full_normalize(text: str) -> str:
    """
    모든 정규화 단계를 적용한 완전 정규화 파이프라인.

    단계:
    1. 특수문자 제거
    2. 로마자 선수명 → 한글
    3. 영문 통계 용어 → 표준화
    4. 한영 사이 공백 추가
    5. 연속 공백 정리

    Args:
        text: 원본 쿼리

    Returns:
        완전 정규화된 쿼리
    """
    if not text or not isinstance(text, str):
        return ""

    result = text
    result = _NOISE_CHARS_PATTERN.sub(" ", result)  # 노이즈 특수문자 제거
    result = normalize_special_chars(result)
    result = normalize_romanized_player_names(result)
    result = normalize_english_stats(result)
    result = correct_spacing(result)
    result = re.sub(r"\s+", " ", result).strip()

    return result
