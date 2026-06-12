#!/usr/bin/env python3
"""Classify operator-data-required chatbot questions into data contracts.

This script reads a smoke chatbot quality report and turns expected
``MANUAL_BASEBALL_DATA_REQUIRED`` answers into an operator-facing taxonomy. It
does not collect, infer, or repair baseball data.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "reports" / "post_256_release_gate_kbo500_full.json"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "reports" / "operator_data_required_taxonomy.json"
DEFAULT_CSV_OUTPUT = PROJECT_ROOT / "reports" / "operator_data_required_taxonomy.csv"
DEFAULT_TEMPLATE_OUTPUT = (
    PROJECT_ROOT / "reports" / "operator_data_contract_template.csv"
)

DOMAINS = (
    "season_meta",
    "schedule_window",
    "game_day_lineup",
    "roster_news",
    "venue_ticket",
    "broadcast_media",
    "fan_event",
    "subjective_prediction",
    "unsupported_external",
    "db_fast_path_candidate",
)

TEAM_ALIASES = (
    "lg",
    "엘지",
    "kt",
    "케이티",
    "ssg",
    "랜더스",
    "kia",
    "기아",
    "삼성",
    "두산",
    "nc",
    "엔씨",
    "롯데",
    "한화",
    "키움",
)


@dataclass(frozen=True)
class DataContract:
    domain: str
    contract_code: str
    required_fields: tuple[str, ...]
    field_descriptions: Mapping[str, str]
    description: str
    notes: str


@dataclass(frozen=True)
class Classification:
    question: str
    domain: str
    contract_code: str
    recommended_status: str
    required_fields: tuple[str, ...]
    reason: str
    candidate_tools: tuple[str, ...] = ()

    def to_record(
        self,
        *,
        source_status: str,
        endpoint_count: int,
        endpoints: Sequence[str],
        sample_answer: str,
    ) -> Dict[str, Any]:
        return {
            "question": self.question,
            "source_status": source_status,
            "recommended_status": self.recommended_status,
            "domain": self.domain,
            "contract_code": self.contract_code,
            "required_fields": list(self.required_fields),
            "candidate_tools": list(self.candidate_tools),
            **final_verdict_for_question(self.question, self),
            "reason": self.reason,
            "endpoint_count": endpoint_count,
            "endpoints": list(endpoints),
            "sample_answer": sample_answer,
        }


@dataclass(frozen=True)
class DbFastPathRecovery:
    tool: str
    reason: str


COMMON_SOURCE_FIELDS = {
    "source_name": "운영자가 확인한 내부/공식 원천 이름",
    "source_checked_at": "운영자가 원천을 확인한 시각",
    "is_verified": "운영자가 검증 완료했는지 여부",
    "confidence": "0.0~1.0 신뢰도. 자동 추론값이 아니라 운영자 판단값",
}


CONTRACTS: Dict[str, DataContract] = {
    "season_meta": DataContract(
        domain="season_meta",
        contract_code="SEASON_META_REQUIRED",
        required_fields=(
            "season_year",
            "event_name",
            "event_date",
            "stadium_name",
            *COMMON_SOURCE_FIELDS.keys(),
        ),
        field_descriptions={
            "season_year": "시즌 연도",
            "event_name": "개막일, 개막전, 올스타 브레이크 등 시즌 메타 이벤트명",
            "event_date": "이벤트 기준일",
            "stadium_name": "장소가 있는 이벤트의 구장명",
            **COMMON_SOURCE_FIELDS,
        },
        description="시즌 시작일, 개막전 장소처럼 시즌 운영 메타 데이터가 필요합니다.",
        notes="일정이 확정되지 않았으면 future_event_pending으로 남겨야 합니다.",
    ),
    "schedule_window": DataContract(
        domain="schedule_window",
        contract_code="SCHEDULE_WINDOW_REQUIRED",
        required_fields=(
            "game_date",
            "game_id",
            "home_team",
            "away_team",
            "stadium_name",
            "start_time",
            "game_status",
            *COMMON_SOURCE_FIELDS.keys(),
        ),
        field_descriptions={
            "game_date": "경기 날짜",
            "game_id": "내부 DB 기준 경기 ID",
            "home_team": "홈 팀",
            "away_team": "원정 팀",
            "stadium_name": "경기장",
            "start_time": "경기 시작 시각",
            "game_status": "예정, 진행, 종료, 취소 등 경기 상태",
            **COMMON_SOURCE_FIELDS,
        },
        description="오늘/내일/이번 주 일정, 경기 결과, 취소 여부에 기준일 데이터가 필요합니다.",
        notes="game 테이블에 이미 있는 과거 날짜는 DB fast-path 후보로 분리합니다.",
    ),
    "game_day_lineup": DataContract(
        domain="game_day_lineup",
        contract_code="GAME_DAY_LINEUP_REQUIRED",
        required_fields=(
            "game_id",
            "team_code",
            "player_name",
            "batting_order",
            "position",
            "announced_at",
            *COMMON_SOURCE_FIELDS.keys(),
        ),
        field_descriptions={
            "game_id": "내부 DB 기준 경기 ID",
            "team_code": "내부 DB 기준 팀 코드",
            "player_name": "선수명",
            "batting_order": "타순",
            "position": "수비 위치 또는 투수 역할",
            "announced_at": "라인업/선발 발표 시각",
            **COMMON_SOURCE_FIELDS,
        },
        description="선발, 라인업, 로테이션, 경기 당일 매치업 데이터가 필요합니다.",
        notes="기존 ingest_lineup_manual.py와 manual starter pipeline을 우선 재사용합니다.",
    ),
    "roster_news": DataContract(
        domain="roster_news",
        contract_code="ROSTER_NEWS_REQUIRED",
        required_fields=(
            "season_year",
            "team_code",
            "player_name",
            "roster_event_type",
            "effective_date",
            "status_text",
            *COMMON_SOURCE_FIELDS.keys(),
        ),
        field_descriptions={
            "season_year": "시즌 연도",
            "team_code": "내부 DB 기준 팀 코드",
            "player_name": "대상 선수명",
            "roster_event_type": "부상, 복귀, 콜업, 말소, 계약, 트레이드 등",
            "effective_date": "효력이 발생한 날짜",
            "status_text": "운영자가 제공한 상태 설명",
            **COMMON_SOURCE_FIELDS,
        },
        description="부상, 콜업/말소, 계약, 트레이드, 감독/코치 변동 데이터가 필요합니다.",
        notes="뉴스성 항목은 기준일과 확인 원천 없이는 답하지 않습니다.",
    ),
    "venue_ticket": DataContract(
        domain="venue_ticket",
        contract_code="VENUE_TICKET_REQUIRED",
        required_fields=(
            "stadium_name",
            "topic_type",
            "title",
            "body",
            "valid_from",
            "valid_to",
            *COMMON_SOURCE_FIELDS.keys(),
        ),
        field_descriptions={
            "stadium_name": "경기장명",
            "topic_type": "좌석, 티켓, 주차, 먹거리, 편의시설 등",
            "title": "운영자 입력 제목",
            "body": "답변 가능한 상세 내용",
            "valid_from": "정보 유효 시작일",
            "valid_to": "정보 유효 종료일",
            **COMMON_SOURCE_FIELDS,
        },
        description="티켓, 좌석, 주차, 먹거리 등 직관 정보가 필요합니다.",
        notes="구단/구장 정책이 바뀌기 쉬워 유효기간을 필수로 둡니다.",
    ),
    "broadcast_media": DataContract(
        domain="broadcast_media",
        contract_code="BROADCAST_MEDIA_REQUIRED",
        required_fields=(
            "game_date",
            "game_id",
            "team_code",
            "channel_name",
            "region",
            "media_type",
            *COMMON_SOURCE_FIELDS.keys(),
        ),
        field_descriptions={
            "game_date": "방송 대상 경기 날짜",
            "game_id": "내부 DB 기준 경기 ID",
            "team_code": "특정 팀 방송이면 팀 코드",
            "channel_name": "채널 또는 플랫폼명",
            "region": "국내, 해외, 지역 등 제공 범위",
            "media_type": "TV, 라디오, 문자중계, 다시보기, 하이라이트 등",
            **COMMON_SOURCE_FIELDS,
        },
        description="중계, 라디오, 문자중계, 다시보기, 하이라이트 정보가 필요합니다.",
        notes="외부 플랫폼 URL은 운영자가 제공한 값만 사용합니다.",
    ),
    "fan_event": DataContract(
        domain="fan_event",
        contract_code="FAN_EVENT_REQUIRED",
        required_fields=(
            "game_date",
            "game_id",
            "team_code",
            "stadium_name",
            "event_type",
            "title",
            "body",
            *COMMON_SOURCE_FIELDS.keys(),
        ),
        field_descriptions={
            "game_date": "이벤트 대상 날짜",
            "game_id": "특정 경기 이벤트면 내부 DB 기준 경기 ID",
            "team_code": "대상 팀 코드",
            "stadium_name": "경기장명",
            "event_type": "응원가, 직관 이벤트, 굿즈, 팬서비스 등",
            "title": "운영자 입력 제목",
            "body": "답변 가능한 상세 내용",
            **COMMON_SOURCE_FIELDS,
        },
        description="응원가, 직관 이벤트, 굿즈, 팬서비스 데이터가 필요합니다.",
        notes="저작권이 있는 응원가 가사는 저장하지 않고 식별 정보만 둡니다.",
    ),
    "subjective_prediction": DataContract(
        domain="subjective_prediction",
        contract_code="SUBJECTIVE_PREDICTION_REQUIRED",
        required_fields=(
            "as_of_date",
            "question_scope",
            "selection_criteria",
            "ranked_entities",
            "operator_basis",
            *COMMON_SOURCE_FIELDS.keys(),
        ),
        field_descriptions={
            "as_of_date": "판단 기준일",
            "question_scope": "우승 후보, 5강 경쟁, 팀 전망 등 질문 범위",
            "selection_criteria": "운영자가 명시한 판단 기준",
            "ranked_entities": "후보 팀/선수와 순위 또는 우선순위",
            "operator_basis": "운영자 제공 판단 근거",
            **COMMON_SOURCE_FIELDS,
        },
        description="예측, 후보, 전망처럼 운영자 기준이 필요한 주관 평가 데이터가 필요합니다.",
        notes="기록 기반 fast-path로 대체 가능하면 db_fast_path_candidate로 분리합니다.",
    ),
    "unsupported_external": DataContract(
        domain="unsupported_external",
        contract_code="UNSUPPORTED_EXTERNAL_REQUIRED",
        required_fields=(
            "requested_topic",
            "supported_by_operator",
            "manual_answer_text",
            *COMMON_SOURCE_FIELDS.keys(),
        ),
        field_descriptions={
            "requested_topic": "질문 주제",
            "supported_by_operator": "서비스에서 운영자가 지원할 주제인지 여부",
            "manual_answer_text": "지원하는 경우 운영자가 승인한 답변 텍스트",
            **COMMON_SOURCE_FIELDS,
        },
        description="현재 내부 DB와 운영자 데이터 계약 밖의 외부성 정보입니다.",
        notes="지원하지 않으면 MANUAL_BASEBALL_DATA_REQUIRED 계약으로 남깁니다.",
    ),
    "db_fast_path_candidate": DataContract(
        domain="db_fast_path_candidate",
        contract_code="DB_FAST_PATH_CANDIDATE",
        required_fields=("candidate_tool", "required_params", "validation_query"),
        field_descriptions={
            "candidate_tool": "연결 후보 도구명",
            "required_params": "도구 호출에 필요한 파라미터",
            "validation_query": "복구 여부를 확인할 대표 질문",
        },
        description="운영자 데이터 대신 현재 DB fast-path로 복구할 가능성이 있는 질문입니다.",
        notes="이번 작업에서는 즉시 답변 성공으로 바꾸지 않고 후보 리포트로만 남깁니다.",
    ),
}

DB_FAST_PATH_OPERATOR_FALLBACK_QUESTIONS = {
    "실책이 적은 선수는 누구야?",
    "도루가 많은 시즌이야?",
    "홈런이 많은 시즌이야?",
}

DB_FAST_PATH_FINAL_REASON = "현재 DB 집계 도구로 답변 복구 대상으로 분류합니다."

RELATIVE_LIVE_SCHEDULE_TOKENS = (
    "오늘",
    "내일",
    "이번 주",
    "이번주",
    "지금",
    "현재 경기",
    "현재 점수",
    "진행 중",
    "몇 회",
    "몇회",
    "다음 경기",
    "잔여 경기",
    "취소",
)

GAME_DAY_DETAIL_TOKENS = (
    "선발 라인업",
    "라인업",
    "선발투수",
    "예상 선발",
    "선발 로테이션",
    "로테이션",
    "스타팅",
    "타순",
    "선발 싸움",
    "선발 투수 ",
    "다음 경기 선발",
    "다음 타자",
    "다음 투수",
    "누가 타석",
    "결승타",
    "경기 mvp",
    "끝내기 상황",
    "득점권",
    "세이브는 누가",
    "승리 투수",
    "패전 투수",
    "홀드는 누가",
    "아웃카운트",
    "주자는 몇",
    "왜 이겼",
    "왜 졌",
    "상성",
)

TEAM_METRIC_FAST_PATH_TOKENS = (
    "불펜 era",
    "불펜 비교",
    "불펜이 좋은",
    "불펜 소모",
    "기복이 큰 팀",
    "도루가 많은 팀",
    "실책이 적은 팀",
    "팀별 era",
    "팀별 경기당 평균 득점",
    "팀별 경기당 평균 실점",
    "선발진 비교",
    "수비 비교",
    "주루 비교",
    "타선 비교",
    "팀별 홈런",
    "팀별 도루",
    "팀별 실책",
)

TEAM_FORM_FAST_PATH_TOKENS = (
    "최근 흐름",
    "연승",
    "연패",
    "홈 승률",
    "원정 승률",
    "홈/원정",
    "홈 원정",
)

STANDINGS_FAST_PATH_TOKENS = (
    "현재 kbo 순위",
    "kbo 순위",
    "순위표",
    "전체 순위",
    "팀별 승패",
    "팀별 승률",
    "팀별 승차",
)


def _contains_any(value: str, tokens: Iterable[str]) -> bool:
    return any(token in value for token in tokens)


def _compact(value: str) -> str:
    return re.sub(r"[\s?.!,~]+", "", value.lower())


def _has_named_team_pair(query_lower: str) -> bool:
    aliases = [alias for alias in TEAM_ALIASES if alias in query_lower]
    return len(set(aliases)) >= 2 and _contains_any(query_lower, ("비교", "vs", "전력"))


def _has_explicit_game_date(query: str) -> bool:
    return bool(
        re.search(r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일", query)
        or re.search(r"\d{4}-\d{1,2}-\d{1,2}", query)
    )


def _has_explicit_date_range(query: str) -> bool:
    if not _has_explicit_game_date(query):
        return False
    if not _contains_any(query, ("부터", "까지", "~")):
        return False
    month_day_count = len(re.findall(r"\d{1,2}\s*월\s*\d{1,2}\s*일", query))
    iso_date_count = len(re.findall(r"\d{4}-\d{1,2}-\d{1,2}", query))
    return month_day_count >= 2 or iso_date_count >= 2


def _is_relative_live_schedule_query(query_lower: str) -> bool:
    return _contains_any(query_lower, RELATIVE_LIVE_SCHEDULE_TOKENS)


def _is_game_day_detail_query(query_lower: str) -> bool:
    return _contains_any(query_lower, GAME_DAY_DETAIL_TOKENS)


def _resolve_db_fast_path_recovery(question: str) -> Optional[DbFastPathRecovery]:
    query_lower = question.lower().strip()

    if question in DB_FAST_PATH_OPERATOR_FALLBACK_QUESTIONS:
        return None
    if _is_game_day_detail_query(query_lower):
        return None

    if _has_named_team_pair(query_lower):
        return DbFastPathRecovery(
            tool="get_team_comparison",
            reason="두 팀명이 명시된 비교 질문은 현재 DB 팀 비교 도구로 복구할 수 있습니다.",
        )
    if _contains_any(query_lower, TEAM_METRIC_FAST_PATH_TOKENS):
        return DbFastPathRecovery(
            tool="get_team_metric_leaderboard",
            reason="팀 단위 지표 질문은 현재 DB 팀 지표 리더보드 도구로 복구할 수 있습니다.",
        )
    if _contains_any(query_lower, TEAM_FORM_FAST_PATH_TOKENS):
        return DbFastPathRecovery(
            tool="get_team_form_table",
            reason="팀 흐름/홈원정/연승연패 질문은 현재 DB 팀 흐름 도구로 복구할 수 있습니다.",
        )
    if _contains_any(query_lower, STANDINGS_FAST_PATH_TOKENS):
        return DbFastPathRecovery(
            tool="get_team_standings",
            reason="팀 순위표/승패/승률/승차 질문은 현재 DB 순위표 도구로 복구할 수 있습니다.",
        )
    if _has_explicit_date_range(question) and _contains_any(
        query_lower, ("경기표", "경기 일정", "경기일정", "일정 보여", "일정 알려")
    ):
        return DbFastPathRecovery(
            tool="get_schedule",
            reason="명시 날짜 범위 경기표 질문은 현재 DB 기간 일정 도구로 복구할 수 있습니다.",
        )
    if (
        _has_explicit_game_date(question)
        and not _is_relative_live_schedule_query(query_lower)
        and _contains_any(
            query_lower,
            ("경기 결과", "경기 일정", "경기일정", "경기표", "스코어", "점수"),
        )
    ):
        return DbFastPathRecovery(
            tool="get_games_by_date",
            reason="명시 날짜 경기 일정/결과 질문은 현재 DB 날짜별 경기 도구로 복구할 수 있습니다.",
        )
    return None


def final_verdict_for_question(
    question: str, classification: Classification
) -> Dict[str, str]:
    if classification.domain != "db_fast_path_candidate":
        return {
            "final_verdict": "operator_data_required",
            "final_reason": "운영자 제공 데이터 계약 대상입니다.",
            "final_candidate_tool": "",
        }
    recovery = _resolve_db_fast_path_recovery(question)
    if recovery is None:
        return {
            "final_verdict": "operator_data_required",
            "final_reason": "이번 pass에서는 현재 DB fast-path 근거가 부족해 수동 데이터 필요로 유지합니다.",
            "final_candidate_tool": "",
        }
    return {
        "final_verdict": "recovered_fast_path",
        "final_reason": DB_FAST_PATH_FINAL_REASON,
        "final_candidate_tool": recovery.tool,
    }


def _contract(domain: str) -> DataContract:
    if domain not in CONTRACTS:
        raise ValueError(f"unsupported operator data domain: {domain}")
    return CONTRACTS[domain]


def _build_classification(
    question: str,
    domain: str,
    reason: str,
    *,
    candidate_tools: Sequence[str] = (),
) -> Classification:
    contract = _contract(domain)
    recommended_status = (
        "db_fast_path_candidate"
        if domain == "db_fast_path_candidate"
        else "operator_data_required"
    )
    return Classification(
        question=question,
        domain=domain,
        contract_code=contract.contract_code,
        recommended_status=recommended_status,
        required_fields=contract.required_fields,
        reason=reason,
        candidate_tools=tuple(candidate_tools),
    )


def classify_question(question: str) -> Classification:
    """Classify one Korean KBO chatbot question into an operator data domain."""
    query_lower = question.lower().strip()
    query_compact = _compact(question)

    recovery = _resolve_db_fast_path_recovery(question)
    if recovery is not None:
        return _build_classification(
            question,
            "db_fast_path_candidate",
            recovery.reason,
            candidate_tools=(recovery.tool,),
        )

    if question in DB_FAST_PATH_OPERATOR_FALLBACK_QUESTIONS:
        return _build_classification(
            question,
            "db_fast_path_candidate",
            "현재 DB fast-path 근거가 부족해 운영자 데이터 필요로 유지합니다.",
        )

    if _contains_any(
        query_lower,
        (
            "개막일",
            "개막전",
            "올스타 브레이크",
            "올스타전 일정",
            "시즌 일정 발표",
        ),
    ):
        return _build_classification(
            question,
            "season_meta",
            "시즌 운영 메타 이벤트의 확정 날짜/장소가 필요합니다.",
        )

    if _contains_any(
        query_lower,
        (
            "선발 라인업",
            "라인업",
            "선발투수",
            "예상 선발",
            "선발 로테이션",
            "로테이션",
            "스타팅",
            "타순",
            "선발 싸움",
            "선발 투수 ",
            "다음 경기 선발",
            "다음 타자",
            "다음 투수",
            "누가 타석",
            "결승타",
            "경기 mvp",
            "끝내기 상황",
            "득점권",
            "세이브는 누가",
            "승리 투수",
            "패전 투수",
            "홀드는 누가",
            "아웃카운트",
            "주자는 몇",
            "왜 이겼",
            "왜 졌",
            "상성",
        ),
    ):
        return _build_classification(
            question,
            "game_day_lineup",
            "경기 당일 선발/라인업/매치업 데이터가 필요합니다.",
        )

    if _contains_any(
        query_lower,
        (
            "오늘",
            "내일",
            "이번 주",
            "이번주",
            "일정",
            "경기표",
            "경기 결과",
            "취소",
            "몇 시",
            "몇시",
            "현재 경기",
            "지금 몇 회",
            "지금 경기",
            "다음 경기 언제",
            "잔여 경기",
            "점수",
            "스코어",
        ),
    ):
        return _build_classification(
            question,
            "schedule_window",
            "기준일 또는 기간별 경기 일정/상태 데이터가 필요합니다.",
        )

    if _contains_any(
        query_lower,
        (
            "중계",
            "방송",
            "채널",
            "라디오",
            "하이라이트",
            "다시보기",
            "문자중계",
        ),
    ):
        return _build_classification(
            question,
            "broadcast_media",
            "경기별 또는 서비스별 중계/미디어 제공 데이터가 필요합니다.",
        )

    if _contains_any(
        query_lower,
        (
            "응원가",
            "직관 이벤트",
            "이벤트",
            "굿즈",
            "팬서비스",
            "치어리더",
            "응원석",
            "응원 구호",
        ),
    ):
        return _build_classification(
            question,
            "fan_event",
            "팬 이벤트/응원/굿즈 정보는 운영자 확인 데이터가 필요합니다.",
        )

    if _contains_any(
        query_lower,
        (
            "구장",
            "경기장",
            "야구장",
            "좌석",
            "티켓",
            "예매",
            "주차",
            "먹거리",
            "음식",
            "매점",
            "가족 관람",
            "가족석",
            "사진 찍기",
            "어린이와 함께",
            "대중교통",
            "매진",
            "원정 응원석",
            "원정 팬",
            "입장 마감",
            "직관 준비물",
            "테이블석",
            "휠체어석",
        ),
    ):
        return _build_classification(
            question,
            "venue_ticket",
            "구장 이용, 티켓, 좌석, 주차, 먹거리 정보가 필요합니다.",
        )

    if _contains_any(
        query_lower,
        (
            "부상",
            "복귀",
            "콜업",
            "말소",
            "엔트리",
            "등록 인원",
            "등록인원",
            "트레이드",
            "계약",
            "감독",
            "코치",
            "외국인 선수",
            "외국인선수",
            "은퇴 소식",
            "최신 kbo 이슈",
            "셋업맨",
        ),
    ):
        return _build_classification(
            question,
            "roster_news",
            "로스터/부상/계약/스태프 변동은 기준일 원천 데이터가 필요합니다.",
        )

    if _contains_any(
        query_lower,
        (
            "우승 후보",
            "우승 가능",
            "5강",
            "전망",
            "후보",
            "가능성",
            "강팀",
            "약팀",
            "에이스",
            "대표 타자",
            "마무리",
            "유망주",
            "베테랑",
            "리더십",
            "대표 선수",
            "안정적인 팀",
            "전력이 탄탄",
            "젊은 팀",
            "기대되는 팀",
            "다크호스",
            "단기전에서 강한",
            "돌풍",
            "라이벌",
            "리빌딩",
            "명문 구단",
            "반등",
            "번트가 중요한",
            "수비가 우승",
            "신인 육성",
            "관전 포인트",
            "볼 포인트",
            "변수",
            "최대 관심사",
            "인기 구단",
            "작전 야구",
            "전통이 강한",
            "타자 체력",
            "타자 친화적",
            "투수 친화적",
            "혹사 위험",
            "불펜 핵심",
            "한국시리즈 경험",
            "휴식이 필요한",
            "좋은 팀",
            "중요한가",
        ),
    ):
        return _build_classification(
            question,
            "subjective_prediction",
            "예측/평가성 질문은 운영자가 기준과 판단 근거를 제공해야 합니다.",
        )

    if _contains_any(
        query_compact,
        (
            "광고규정",
            "판정보완",
            "구종별성적",
            "war",
            "규정은바뀌었",
            "어디서봐",
            "어디서보",
        ),
    ):
        return _build_classification(
            question,
            "unsupported_external",
            "현재 내부 DB와 운영자 계약 밖의 외부성 정보입니다.",
        )

    return _build_classification(
        question,
        "unsupported_external",
        "고정 분류 규칙에 맞지 않아 운영자 검토가 필요한 외부성 질문으로 둡니다.",
    )


def _load_report(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {"results": payload, "summary": {}}
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"unsupported report payload type: {type(payload).__name__}")


def _result_status(result: Mapping[str, Any]) -> str:
    answerability = result.get("answerability")
    if isinstance(answerability, Mapping):
        status = answerability.get("status")
        if status:
            return str(status)
    status = result.get("status")
    return str(status or "unknown")


def _iter_results(report: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("report must contain a results list")
    return [result for result in results if isinstance(result, Mapping)]


def _group_target_questions(
    results: Sequence[Mapping[str, Any]],
    *,
    target_status: str,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for result in results:
        question = str(result.get("question") or "").strip()
        if not question or _result_status(result) != target_status:
            continue
        entry = grouped.setdefault(
            question,
            {
                "question": question,
                "source_status": target_status,
                "endpoint_count": 0,
                "endpoints": set(),
                "sample_answer": "",
            },
        )
        entry["endpoint_count"] += 1
        endpoint = str(result.get("endpoint") or "").strip()
        if endpoint:
            entry["endpoints"].add(endpoint)
        if not entry["sample_answer"] and result.get("answer"):
            entry["sample_answer"] = str(result.get("answer") or "")
    return grouped


def _count_unique_by_status(results: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    by_status: Dict[str, set[str]] = {}
    for result in results:
        question = str(result.get("question") or "").strip()
        if not question:
            continue
        by_status.setdefault(_result_status(result), set()).add(question)
    return {status: len(questions) for status, questions in sorted(by_status.items())}


def _count_rows_by_status(results: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for result in results:
        status = _result_status(result)
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def build_taxonomy(
    report: Mapping[str, Any],
    *,
    target_status: str = "operator_data_required",
) -> Dict[str, Any]:
    results = _iter_results(report)
    grouped = _group_target_questions(results, target_status=target_status)

    records: List[Dict[str, Any]] = []
    for question in sorted(grouped):
        entry = grouped[question]
        classification = classify_question(question)
        records.append(
            classification.to_record(
                source_status=str(entry["source_status"]),
                endpoint_count=int(entry["endpoint_count"]),
                endpoints=sorted(entry["endpoints"]),
                sample_answer=str(entry["sample_answer"]),
            )
        )

    domain_counts: Dict[str, int] = {domain: 0 for domain in DOMAINS}
    recommended_status_counts: Dict[str, int] = {}
    final_verdict_counts: Dict[str, int] = {}
    for record in records:
        domain_counts[record["domain"]] += 1
        status = str(record["recommended_status"])
        recommended_status_counts[status] = (
            recommended_status_counts.get(status, 0) + 1
        )
        verdict = str(record.get("final_verdict") or "")
        final_verdict_counts[verdict] = final_verdict_counts.get(verdict, 0) + 1

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "target_status": target_status,
        },
        "summary": {
            "total_results": len(results),
            "row_counts_by_status": _count_rows_by_status(results),
            "unique_questions_by_status": _count_unique_by_status(results),
            "target_unique_questions": len(records),
            "domain_counts": domain_counts,
            "recommended_status_counts": dict(sorted(recommended_status_counts.items())),
            "final_verdict_counts": dict(sorted(final_verdict_counts.items())),
        },
        "records": records,
    }


def build_contract_template_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for domain in DOMAINS:
        contract = _contract(domain)
        for field_name in contract.required_fields:
            rows.append(
                {
                    "domain": domain,
                    "contract_code": contract.contract_code,
                    "field_name": field_name,
                    "required": "true",
                    "description": contract.field_descriptions.get(field_name, ""),
                    "domain_description": contract.description,
                    "notes": contract.notes,
                }
            )
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_records_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "source_status",
        "recommended_status",
        "domain",
        "contract_code",
        "required_fields",
        "candidate_tools",
        "final_verdict",
        "final_reason",
        "final_candidate_tool",
        "reason",
        "endpoint_count",
        "endpoints",
        "sample_answer",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["required_fields"] = "|".join(record.get("required_fields") or [])
            row["candidate_tools"] = "|".join(record.get("candidate_tools") or [])
            row["endpoints"] = "|".join(record.get("endpoints") or [])
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_template_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "contract_code",
        "field_name",
        "required",
        "description",
        "domain_description",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify MANUAL_BASEBALL_DATA_REQUIRED smoke questions into "
            "operator data contracts."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--csv-output", default=str(DEFAULT_CSV_OUTPUT))
    parser.add_argument("--template-output", default=str(DEFAULT_TEMPLATE_OUTPUT))
    parser.add_argument("--target-status", default="operator_data_required")
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the generated summary to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    taxonomy = build_taxonomy(
        _load_report(input_path), target_status=str(args.target_status)
    )
    taxonomy["input"]["report_path"] = str(input_path)
    taxonomy["input"]["json_output"] = str(Path(args.json_output))
    taxonomy["input"]["csv_output"] = str(Path(args.csv_output))
    taxonomy["input"]["template_output"] = str(Path(args.template_output))

    template_rows = build_contract_template_rows()
    taxonomy["contract_template"] = template_rows

    write_json(Path(args.json_output), taxonomy)
    write_records_csv(Path(args.csv_output), taxonomy["records"])
    write_template_csv(Path(args.template_output), template_rows)

    if args.print_summary:
        print(json.dumps(taxonomy["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
