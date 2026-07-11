"""
RAG (Retrieval-Augmented Generation) 파이프라인의 핵심 로직을 구현한 모듈입니다.

이 모듈은 사용자 쿼리에 대해 관련성 높은 정보를 검색하고,
LLM(Large Language Model)을 사용하여 자연스러운 답변을 생성하는 과정을 담당합니다.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import random
from contextlib import asynccontextmanager, contextmanager
from datetime import date, datetime
from functools import lru_cache, wraps
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Tuple
import psycopg
from psycopg_pool import AsyncConnectionPool

import httpx

from ..config import Settings
from .embeddings import async_embed_query
from .http_clients import get_shared_httpx_client
from .prompts import FOLLOWUP_PROMPT, SYSTEM_PROMPT, HYDE_PROMPT
from .retrieval import record_retrieval_event, similarity_search
from . import kbo_metrics
from .entity_extractor import enhance_search_strategy
from .query_normalizer import full_normalize
from .query_transformer import QueryTransformer, multi_query_retrieval
from .context_formatter import ContextFormatter
from ..agents.baseball_agent import BaseballAgentRuntime
from ..agents.shared_runtime import initialize_shared_baseball_agent_runtime
from ..tools.operator_data_query import try_build_operator_fast_path_result
from .wpa_calculator import WPACalculator
from .retry_utils import llm_retry
from .exceptions import DBRetrievalError
from ..observability.metrics import (
    AI_LLM_CALL_DURATION_SECONDS,
    AI_RAG_STAGE_DURATION_SECONDS,
)
from time import perf_counter as _rag_perf_counter

logger = logging.getLogger(__name__)


# --- Constants and Helpers ---

TEAM_MAP = {
    "KIA": "KIA 타이거즈",
    "HT": "KIA 타이거즈",
    "기아": "KIA 타이거즈",
    "LG": "LG 트윈스",
    "DB": "두산 베어스",
    "DO": "두산 베어스",
    "OB": "두산 베어스",
    "두산": "두산 베어스",
    "롯데": "롯데 자이언츠",
    "삼성": "삼성 라이온즈",
    "키움": "키움 히어로즈",
    "KH": "키움 히어로즈",
    "KI": "키움 히어로즈",
    "WO": "키움 히어로즈",
    "NX": "키움 히어로즈",
    "한화": "한화 이글스",
    "KT": "KT 위즈",
    "NC": "NC 다이노스",
    "SSG": "SSG 랜더스",
    "SK": "SSG 랜더스",
}
MIN_IP_SP = 70
MIN_IP_RP = 30
MIN_PA_BATTER = 100
DB_UNAVAILABLE_PREFIX = (
    "⚠️ 현재 KBO 통계 DB에 일시적으로 접속할 수 없어 정확한 수치를 확인하지 못했습니다. "
    "아래 내용은 일반 야구 지식 기반의 참고 답변입니다."
)
EMBEDDING_FAILED_PREFIX = (
    "검색용 임베딩 생성에 실패해 저장된 KBO 근거를 확인하지 못했습니다. "
    "아래 내용은 제한적인 참고 답변입니다."
)
ZERO_HIT_PREFIX = "저장된 KBO 데이터에서는 관련 근거를 찾지 못했습니다."
_FORCE_AGENT_FAST_PATH_KEYWORDS = (
    "홈런왕",
    "홈런 1위",
    "최다 홈런",
    "최다홈런",
    "최다안타",
    "안타왕",
    "안타 1위",
    "승률 흐름",
    "현재 순위",
    "상대 전적",
    "상대전적",
    "유독 꼬이",
    "수비 실책",
    "실책 때문에",
    "같은 포지션",
)
_FORCE_TEAM_ANALYSIS_FAST_PATH_KEYWORDS = (
    "팀 타율",
    "팀 평균자책점",
    "팀 평균자책",
    "팀 era",
    "타선",
    "마운드",
    "선발진",
    "선발",
    "불펜",
    "홈런 생산력",
    "도루",
    "주루",
    "가을야구",
    "플레이오프",
    "플옵",
    "홈 경기",
    "원정 경기",
    "승패 흐름",
    "순위와 승패",
    "패턴",
    "최근",
    "5경기",
    "10경기",
    "흐름",
    "득점",
    "강점",
    "약점",
    "전력",
    "상태",
    "페이스",
    "폼",
    "가능성",
    "무너진",
    "대표 경기",
    "최다 득점",
    "역전승",
    "리드를 지킨",
    "중심타선",
)
_FORCE_PLAYER_FAST_PATH_KEYWORDS = (
    "시즌 성적",
    "주요 기록",
    "기록도",
    "강점",
    "약점",
    "같은 포지션",
    "타자야",
    "투수야",
    "타율",
    "ops",
    "홈런",
    "안타",
    "타점",
    "도루",
    "출루율",
    "장타율",
    "평균자책",
    "평균 자책",
    "era",
    "whip",
    "다승",
    "세이브",
    "홀드",
    "탈삼진",
)
_STATIC_KBO_TEAMS = (
    "LG 트윈스",
    "KT 위즈",
    "SSG 랜더스",
    "KIA 타이거즈",
    "삼성 라이온즈",
    "두산 베어스",
    "NC 다이노스",
    "롯데 자이언츠",
    "한화 이글스",
    "키움 히어로즈",
)
_STATIC_TEAM_STADIUMS = {
    "LG 트윈스": "잠실야구장",
    "두산 베어스": "잠실야구장",
    "KT 위즈": "수원 KT위즈파크",
    "SSG 랜더스": "인천 SSG랜더스필드",
    "KIA 타이거즈": "광주-KIA 챔피언스필드",
    "삼성 라이온즈": "대구 삼성 라이온즈 파크",
    "NC 다이노스": "창원NC파크",
    "롯데 자이언츠": "사직야구장",
    "한화 이글스": "대전 한화생명 볼파크",
    "키움 히어로즈": "고척스카이돔",
}
_STATIC_TEAM_ALIAS_MAP = {
    "lg": "LG 트윈스",
    "lg 트윈스": "LG 트윈스",
    "트윈스": "LG 트윈스",
    "kt": "KT 위즈",
    "kt 위즈": "KT 위즈",
    "위즈": "KT 위즈",
    "ssg": "SSG 랜더스",
    "ssg 랜더스": "SSG 랜더스",
    "랜더스": "SSG 랜더스",
    "kia": "KIA 타이거즈",
    "kia 타이거즈": "KIA 타이거즈",
    "기아": "KIA 타이거즈",
    "타이거즈": "KIA 타이거즈",
    "삼성": "삼성 라이온즈",
    "삼성 라이온즈": "삼성 라이온즈",
    "라이온즈": "삼성 라이온즈",
    "두산": "두산 베어스",
    "두산 베어스": "두산 베어스",
    "베어스": "두산 베어스",
    "nc": "NC 다이노스",
    "nc 다이노스": "NC 다이노스",
    "다이노스": "NC 다이노스",
    "롯데": "롯데 자이언츠",
    "롯데 자이언츠": "롯데 자이언츠",
    "자이언츠": "롯데 자이언츠",
    "한화": "한화 이글스",
    "한화 이글스": "한화 이글스",
    "이글스": "한화 이글스",
    "키움": "키움 히어로즈",
    "키움 히어로즈": "키움 히어로즈",
    "히어로즈": "키움 히어로즈",
}
_STATIC_EXPLAINER_ANSWERS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (
        ("야구 규칙",),
        "야구는 공격팀이 공을 치고 베이스를 돌아 득점하고, 수비팀은 타자와 주자를 아웃시켜 이닝을 끝내는 경기입니다. 한 이닝은 양 팀이 한 번씩 공격하며, 보통 9이닝 동안 더 많은 점수를 낸 팀이 이깁니다.",
    ),
    (
        ("필승조",),
        "필승조는 팀이 앞서 있거나 접전인 후반에 승리를 지키기 위해 우선 투입하는 핵심 불펜 묶음입니다. 보통 셋업맨과 마무리투수, 그리고 가장 믿을 수 있는 중간계투가 이 역할을 맡습니다.",
    ),
    (
        ("오프너",),
        "오프너 전략은 짧은 이닝만 던지는 투수를 먼저 내고, 이후 긴 이닝을 맡을 투수를 붙이는 투수 운용 방식입니다. 상대 상위 타순을 초반부터 맞춤형으로 막고 싶을 때 쓰입니다.",
    ),
    (
        ("불펜 데이",),
        "불펜 데이는 전통적인 선발투수 한 명에게 긴 이닝을 맡기지 않고 여러 불펜 투수가 나눠 던지는 경기 운영입니다. 선발 공백이나 일정 부담이 있을 때 선택하는 방식입니다.",
    ),
    (
        ("수비 시프트", "시프트"),
        "수비 시프트는 타자의 타구 방향 성향에 맞춰 야수 위치를 평소와 다르게 배치하는 전술입니다. 당겨 치는 타자나 특정 코스 타구가 많은 타자를 상대로 안타 확률을 낮추려는 목적이 큽니다.",
    ),
    (
        ("더블헤더",),
        "더블헤더는 같은 날 두 경기를 이어서 치르는 편성입니다. 우천 취소나 잔여 경기 압축 편성 때문에 생기는 경우가 많고, 투수 운용과 선수 체력 관리가 중요해집니다.",
    ),
    (
        ("서스펜디드",),
        "서스펜디드 게임은 경기가 중단된 뒤 나중에 같은 상황에서 이어서 재개되는 경기입니다. 일반 취소와 달리 이미 진행된 기록과 상황을 보존한다는 점이 핵심입니다.",
    ),
    (
        ("우천 취소",),
        "우천 취소는 날씨 때문에 경기를 정상 진행하기 어렵다고 판단될 때 경기를 열지 않거나 중단 후 취소하는 처리입니다. 취소된 경기는 잔여 경기 일정으로 다시 편성됩니다.",
    ),
    (
        ("휴식일",),
        "휴식일은 정규 편성, 이동 거리, 우천 취소 재편성, 구장 사용 조건을 함께 고려해 정해집니다. 시즌 중 실제 휴식일은 공식 일정 데이터 기준으로 확인해야 합니다.",
    ),
    (
        ("원정 연전",),
        "원정 연전은 이동과 숙소 생활이 이어져 체력 관리가 어려워지는 일정입니다. 특히 야간 경기 뒤 이동, 불펜 소모, 주전 휴식 배분이 팀 운영의 핵심 변수가 됩니다.",
    ),
    (
        ("홈 연전",),
        "홈 연전은 이동 부담이 적고 익숙한 구장 환경을 활용할 수 있다는 장점이 있습니다. 다만 실제로 언제 많은지는 시즌 일정 데이터가 필요합니다.",
    ),
    (
        ("시리즈 마지막 경기",),
        "시리즈 마지막 경기는 위닝시리즈 여부, 불펜 소모, 다음 이동 일정에 영향을 주기 때문에 중요합니다. 같은 1승이라도 연전 흐름을 바꾸는 의미가 커질 수 있습니다.",
    ),
    (
        ("시즌 막판 일정",),
        "시즌 막판 일정은 순위 경쟁, 잔여 경기 압축, 투수 체력, 부상 관리가 한꺼번에 걸려 중요합니다. 포스트시즌 경쟁권 팀일수록 한 경기의 가치가 커집니다.",
    ),
    (
        ("스트라이크", "볼"),
        "스트라이크와 볼은 투구 판정의 기본 단위입니다. 스트라이크는 타자가 치지 않았거나 헛스윙한 공이 스트라이크 존 조건을 만족한 경우이고, 볼은 그 조건을 벗어난 투구로 보면 됩니다.",
    ),
    (
        ("아웃이 되는 경우",),
        "아웃은 삼진, 뜬공 포구, 땅볼 뒤 베이스보다 먼저 송구, 태그 아웃처럼 공격 기회가 끝나는 판정입니다. 공격팀은 한 이닝에 아웃 3개를 당하면 수비로 전환합니다.",
    ),
    (
        ("홈런",),
        "홈런은 타자가 친 공이 페어 지역으로 담장을 넘어가거나 수비가 잡을 수 없는 방식으로 모든 베이스를 돌 수 있을 때 나오는 득점 플레이입니다. 주자가 있으면 주자까지 함께 득점합니다.",
    ),
    (
        ("병살",),
        "병살은 한 번의 플레이에서 아웃 두 개가 동시에 나오는 상황입니다. 주자가 있는 공격 기회가 한순간에 사라지기 때문에 흐름을 크게 끊는 결과가 됩니다.",
    ),
    (
        ("도루 성공률", "도루성공률"),
        "도루 성공률은 도루를 시도한 횟수 중 성공한 비율입니다. 보통 도루 / (도루 + 도루 실패)로 계산하며, 값이 높을수록 주루 시도가 아웃 손실 없이 득점권 진입으로 이어졌다는 뜻입니다.",
    ),
    (
        ("도루",),
        "도루는 타자의 타격 없이 주자가 다음 베이스를 훔치듯 진루하는 플레이입니다. 성공하면 득점권 기회를 만들 수 있지만 실패하면 아웃 하나를 잃는 고위험 선택입니다.",
    ),
    (
        ("세이브",),
        "세이브는 구원투수가 팀의 리드를 지키고 경기를 끝냈을 때 특정 조건에서 기록되는 투수 기록입니다. 접전 후반을 막아낸 마무리 성과를 보여주는 지표로 쓰입니다.",
    ),
    (
        ("홀드",),
        "홀드는 구원투수가 세이브 상황의 리드를 유지한 채 다음 투수에게 넘겼을 때 기록되는 지표입니다. 마무리 전 단계의 핵심 불펜 기여를 볼 때 사용합니다.",
    ),
    (
        ("타점",),
        "타점은 타자의 타격이나 희생플라이 등으로 주자가 홈을 밟아 득점했을 때 타자에게 기록되는 공격 지표입니다. 득점 생산에 직접 얼마나 관여했는지 볼 때 씁니다.",
    ),
    (
        ("war",),
        "WAR은 대체 선수와 비교해 팀 승리에 얼마나 더 기여했는지를 승수 단위로 보는 종합 가치 지표입니다. 타격, 주루, 수비, 투구 같은 여러 요소를 하나로 묶어 평가할 때 씁니다.",
    ),
    (
        ("fip",),
        "FIP는 투수가 통제하기 쉬운 홈런, 볼넷, 몸에 맞는 공, 삼진을 중심으로 투구 내용을 보는 지표입니다. 수비 영향을 줄이고 투수의 순수 투구력을 보려는 목적이 큽니다.",
    ),
    (
        ("ops",),
        "OPS는 출루율과 장타율을 더한 공격 지표입니다. 출루 능력과 장타 생산력을 함께 보기 때문에 타자의 전체 공격력을 빠르게 볼 때 많이 씁니다.",
    ),
    (
        ("whip",),
        "WHIP는 투수가 이닝당 허용한 안타와 볼넷의 합입니다. 값이 낮을수록 주자를 적게 내보냈다는 뜻이라 투수 안정성을 볼 때 유용합니다.",
    ),
    (
        ("qs",),
        "QS는 선발투수가 6이닝 이상을 던지면서 3자책점 이하로 막은 경기입니다. 선발이 경기를 무너지지 않게 책임졌는지를 보는 기본적인 안정성 지표입니다.",
    ),
    (
        ("babip",),
        "BABIP는 인플레이 타구가 안타가 되는 비율을 보는 지표입니다. 타구 질, 수비, 운이 함께 섞이기 때문에 다른 기록과 같이 해석해야 합니다.",
    ),
    (
        ("수비율",),
        "수비율은 수비 기회 중 실책 없이 아웃 처리나 보살을 기록한 비율을 보는 지표입니다. 단순히 높을수록 안정적이지만, 수비 범위나 어려운 타구 처리 능력까지 모두 설명하지는 못합니다.",
    ),
    (
        ("인필드 플라이",),
        "인필드 플라이는 특정 주자 상황에서 내야수가 평범하게 잡을 수 있는 뜬공에 대해 타자를 자동 아웃으로 선언하는 규칙입니다. 수비가 일부러 공을 떨어뜨려 병살을 노리는 것을 막기 위한 장치입니다.",
    ),
    (
        ("태그업",),
        "태그업은 플라이볼이 잡힌 뒤 주자가 원래 베이스를 다시 밟고 다음 베이스로 진루하는 플레이입니다. 공이 잡히기 전에 먼저 출발하면 아웃 위험이 생깁니다.",
    ),
    (
        ("보크",),
        "보크는 투수가 주자를 속이거나 투구 동작 규정을 어겼을 때 주자에게 진루권이 주어지는 반칙입니다. 주자가 있을 때 투수 동작의 일관성과 합법성을 보는 규정입니다.",
    ),
    (
        ("체크 스윙", "체크스윙"),
        "체크 스윙은 타자가 스윙을 하려다 멈춘 동작입니다. 배트가 충분히 돌았는지에 따라 스윙 여부를 판정하며, 상황에 따라 심판 합의나 판독 대상 논의가 생길 수 있습니다.",
    ),
    (
        ("비디오판독", "비디오 판독"),
        "비디오판독은 현장 판정이 애매한 장면을 영상으로 다시 확인해 바로잡는 절차입니다. 홈런 여부, 세이프와 아웃, 페어와 파울처럼 경기 결과에 직접 영향을 주는 장면이 핵심 대상입니다.",
    ),
    (
        ("주루 방해",),
        "주루 방해는 수비수가 공 처리와 무관하게 주자의 정상적인 주루를 막았는지 보는 규정입니다. 상황에 따라 심판이 방해가 없었다면 도달했을 베이스를 판단해 주자에게 진루권을 줍니다.",
    ),
    (
        ("판정 번복",),
        "판정 번복은 심판 합의나 비디오판독을 통해 기존 판정이 명확히 잘못됐다고 판단될 때 이뤄집니다. 실제 적용 범위와 절차는 해당 시즌 KBO 공식 규정 기준으로 확인해야 합니다.",
    ),
)
_LIVE_MANUAL_DATA_TIME_TOKENS = (
    "오늘",
    "어제",
    "내일",
    "이번 주",
    "지금",
    "현재",
    "최신",
    "최근",
    "금일",
    "방금",
)
_VAGUE_GAME_DETAIL_TOKENS = (
    "경기 스코어",
    "경기 mvp",
    "결승타",
    "승리 투수",
    "패전 투수",
    "세이브는 누가",
    "홀드는 누가",
    "경기 하이라이트",
    "경기 승부처",
    "경기의 핵심 장면",
    "오늘 최고의 선수",
    "오늘 아쉬운 선수",
    "오늘 경기의 분수령",
    "오늘 경기의 포인트",
    "다음 경기에서 볼 포인트",
    "이번 시리즈에서 볼 포인트",
    "이번 시즌 최대 관심사",
    "이번 시즌 변수",
    "이번 시즌 관전 포인트",
    "오늘 우리 팀 이겨",
    "우리 팀 순위",
    "다음 경기 언제",
    "오늘 누가 던져",
    "오늘 누가 쳐",
    "지금 경기 어때",
    "왜 졌어",
    "왜 이겼어",
    "다음 경기 전망",
    "이번 시즌 우승 가능",
    "경기 실책",
    "경기 홈런",
    "경기 역전",
    "라인업",
    "오늘 선발",
    "다음 경기 선발",
    "주자는 몇 명",
    "아웃카운트",
    "누가 타석",
    "다음 투수",
    "다음 타자",
    "득점권 상황",
    "끝내기 상황",
    "선발 투수",
    "선발 매치업",
    "팀별 선발 로테이션",
    "에이스는 누구",
    "팀 마무리 투수",
    "팀 불펜 핵심",
    "홀드왕 후보",
    "세이브왕 후보",
    "불펜 era",
    "불펜 소모",
    "셋업맨",
)
_LIVE_MANUAL_DATA_SUBJECT_TOKENS = (
    "경기 일정",
    "경기 결과",
    "경기 예상",
    "경기 승리",
    "경기 핵심",
    "경기 선발",
    "경기 불펜",
    "경기 타선",
    "경기 수비",
    "경기 후반",
    "경기 홈 어드밴티지",
    "예상 스코어",
    "승리 확률",
    "홈 어드밴티지",
    "후반 집중력",
    "점수",
    "스코어",
    "선발",
    "라인업",
    "엔트리",
    "부상",
    "트레이드",
    "뉴스",
    "이슈",
    "소식",
    "콜업",
    "말소",
    "계약",
    "교체",
    "감독 교체",
    "연장 계약",
    "은퇴",
    "몇 회",
    "주자",
    "아웃카운트",
    "타석",
    "다음 투수",
    "다음 타자",
    "득점권",
    "끝내기 상황",
)
_SCHEDULE_MANUAL_DATA_TOKENS = (
    "개막일",
    "개막전",
    "올스타전",
    "올스타 브레이크",
    "시즌 종료",
    "경기 일정",
    "경기표",
    "일정만",
    "스코어보드",
    "문자중계",
    "중계",
    "라디오",
    "하이라이트",
    "다시보기",
    "경기 알림",
    "알림 설정",
)
_FUTURE_EVENT_DEFINITIONS: Tuple[Dict[str, Any], ...] = (
    {
        "event_key": "opening_day",
        "label": "개막 이벤트",
        "tokens": ("개막일", "개막전", "정규시즌 개막"),
        "pending_until": (4, 1),
        "post_event_scope": "개막전 결과, 선발, 관중 수처럼 경기 후 확정되는 정보",
    },
    {
        "event_key": "all_star",
        "label": "올스타전",
        "tokens": (
            "올스타전",
            "올스타 브레이크",
            "올스타 mvp",
            "올스타전 mvp",
            "올스타",
        ),
        "pending_until": (8, 1),
        "post_event_scope": "올스타전 결과, MVP, 하이라이트처럼 행사 후 확정되는 정보",
    },
    {
        "event_key": "season_end",
        "label": "정규시즌 종료",
        "tokens": ("시즌 종료", "정규시즌 종료", "최종 순위"),
        "pending_until": (11, 1),
        "post_event_scope": "최종 순위와 시즌 결산처럼 시즌 종료 후 확정되는 정보",
    },
    {
        "event_key": "postseason",
        "label": "포스트시즌",
        "tokens": (
            "포스트시즌",
            "가을야구",
            "와일드카드",
            "준플레이오프",
            "플레이오프",
            "한국시리즈",
        ),
        "pending_until": (11, 30),
        "post_event_scope": "포스트시즌 대진, 시리즈 결과, 한국시리즈 우승팀처럼 진행 후 확정되는 정보",
    },
    {
        "event_key": "season_awards",
        "label": "시즌 수상",
        "tokens": ("mvp", "엠브이피", "골든글러브", "신인왕", "수상자", "수상"),
        "pending_until": (12, 31),
        "post_event_scope": "MVP, 골든글러브, 신인왕처럼 시즌 종료 후 확정되는 수상 정보",
    },
)
_FUTURE_EVENT_EXPLAINER_TOKENS = (
    "어떻게",
    "몇 경기",
    "몇경기",
    "규정",
    "룰",
    "뜻",
    "의미",
    "정의",
    "설명",
    "방식",
    "구조",
    "결정돼",
    "경험",
)
_FUTURE_EVENT_STATUS_TOKENS = (
    "언제",
    "일정",
    "날짜",
    "개최",
    "열려",
    "어디서",
    "장소",
    "결과",
    "mvp",
    "엠브이피",
    "수상",
    "수상자",
    "신인왕",
    "골든글러브",
    "우승팀",
    "진출 후보",
    "후보",
    "가능성",
    "유리한 팀",
    "종료",
    "브레이크",
    "끝나",
    "누구",
)
_FAN_EXPERIENCE_MANUAL_DATA_TOKENS = (
    "야구장 티켓",
    "티켓 예매",
    "티켓 가격",
    "매진 여부",
    "취소표",
    "할인 좌석",
    "원정 응원석",
    "원정 팬",
    "가족 관람",
    "어린이와 함께",
    "가족석",
    "테이블석",
    "응원석",
    "추천 좌석",
    "좌석 추천",
    "좌석 배치",
    "초보 직관",
    "야구장 주차",
    "근처 주차장",
    "대중교통",
    "야구장 음식",
    "근처 맛집",
    "먹거리",
    "사진 찍기",
    "직관 이벤트",
    "직관 준비물",
    "홈구장 주소",
    "휠체어석",
    "입장 마감",
    "팀별 응원가",
    "팀별 응원 구호",
    "잔여 경기 편성",
    "경기 취소는 몇 시",
)
_LEAGUE_WIDE_CURRENT_DATA_TOKENS = (
    "현재 kbo 순위",
    "1위 팀",
    "5강 경쟁",
    "가을야구 가능성",
    "최하위 팀",
    "팀별 승패",
    "팀별 승률",
    "팀별 승차",
    "최근 10경기",
    "홈 승률",
    "원정 승률",
    "팀별 득점력",
    "팀별 실점력",
    "공격과 수비",
    "연승 중인 팀",
    "연패 중인 팀",
)
_LEADERBOARD_MANUAL_DATA_TOKENS = (
    "장타율 1위",
    "ops 1위",
    "도루 1위",
    "득점 1위",
    "삼진이 많은 타자",
    "완투",
    "완봉",
    "이닝 소화",
    "피안타율",
    "실책이 적은 팀",
    "실책이 적은 선수",
    "도루가 많은 팀",
    "도루 성공률",
    "포수 도루저지율",
    "내야 수비가 좋은 팀",
    "외야 수비가 좋은 팀",
    "병살타가 많은 팀",
    "구종별 성적",
    "타구 속도",
    "존별 타격",
    "라인업 상성",
)
_ROSTER_EVALUATION_MANUAL_DATA_TOKENS = (
    "이 팀의 대표 타자",
    "이 팀의 에이스",
    "이 팀의 마무리",
    "이 팀의 유망주",
    "이 팀의 외국인 타자",
    "이 팀의 외국인 투수",
    "부상 중인 핵심 선수",
    "복귀가 기대되는 선수",
    "신인왕 후보",
    "mvp 후보",
    "우승 후보",
    "한국시리즈 진출 후보",
    "플레이오프 진출 후보",
    "준플레이오프 진출 후보",
    "와일드카드 유리한 팀",
    "다크호스 팀",
    "돌풍을 일으킬 팀",
    "반등할 팀",
    "추락 가능성",
    "단기전에서 강한 팀",
    "숫자 밖",
    "리더십",
    "팀 상징성",
    "상징성까지",
    "두 선수 성적 비교",
    "두 팀 전력 비교",
    "lg와 kt를 비교",
    "ssg와 kia를 비교",
    "삼성과 두산을 비교",
    "nc와 롯데를 비교",
    "한화와 키움을 비교",
    "타자끼리 비교",
    "투수끼리 비교",
    "선발진 비교",
    "불펜 비교",
    "타선 비교",
    "수비 비교",
    "주루 비교",
    "최근 흐름 비교",
    "팀별 경기당 평균 득점",
    "팀별 경기당 평균 실점",
    "팀별 홈런 수",
    "팀별 도루 수",
    "팀별 실책 수",
    "팀별 타율",
    "팀별 출루율",
    "팀별 장타율",
    "팀별 ops",
    "팀별 era",
    "타자 친화적 리그",
    "투수 친화적 리그",
    "홈런이 많은 시즌",
    "도루가 많은 시즌",
    "번트가 중요한 시즌",
    "작전 야구",
    "클러치 능력",
    "선발 야구",
    "불펜 야구",
    "수비가 우승",
    "휴식이 필요한 선수",
    "부상자 명단",
    "엔트리 변경",
    "콜업된 선수",
    "말소된 선수",
    "외국인 선수 교체",
    "감독 교체 소식",
    "콜업 소식",
    "말소 소식",
    "연장 계약 소식",
    "은퇴 소식",
    "투수 혹사 위험",
    "타자 체력 관리",
    "장기 부상자",
    "복귀 임박 선수",
    "신인 육성",
    "선발진이 좋은 팀",
    "불펜이 좋은 팀",
    "타선이 좋은 팀",
    "수비가 좋은 팀",
    "감독 운영이 좋은 팀",
    "우승 가능성이 있는 팀",
    "5강에 들 팀",
    "리빌딩이 필요한 팀",
    "반등이 필요한 팀",
    "내년이 더 기대되는 팀",
    "가장 안정적인 팀",
    "가장 기복이 큰 팀",
    "가장 젊은 팀",
    "가장 베테랑이 많은 팀",
    "가장 전력이 탄탄한 팀",
    "kbo 우승이 가장 많은 팀",
    "한국시리즈 경험이 많은 팀",
    "명문 구단",
    "인기 구단",
    "라이벌 관계",
    "잠실 라이벌",
    "영남 라이벌",
    "호남 라이벌",
    "원정 팬이 많은 팀",
    "전통이 강한 팀",
)
_TEAM_PAIR_COMPARISON_COMPACT_TOKENS = (
    "lg와kt를비교",
    "ssg와kia를비교",
    "삼성과두산을비교",
    "nc와롯데를비교",
    "한화와키움을비교",
)
_REGULATION_MANUAL_DATA_TOKENS = (
    "등록 인원",
    "엔트리",
    "광고 규정",
    "판정 보완",
    "광고 설치 규정",
    "덕아웃 인원",
    "외국 물질 검사",
    "판독 오류 보정",
    "경기장별 규정 차이",
    "수비 시프트 위반",
    "파울 라인 규정",
)


def _to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _ensure_answer_prefix(answer: str, prefix: str) -> str:
    answer_text = (answer or "").strip()
    if not prefix:
        return answer_text
    if answer_text.startswith(prefix):
        return answer_text
    if not answer_text:
        return prefix
    return f"{prefix}\n\n{answer_text}"


def _ensure_zero_hit_answer_prefix(answer: str) -> str:
    answer_text = (answer or "").strip()
    if answer_text.startswith("저장된"):
        return answer_text
    if not answer_text:
        return ZERO_HIT_PREFIX
    return f"{ZERO_HIT_PREFIX} {answer_text}"


def _build_static_kbo_result(
    answer: str,
    *,
    intent: str = "baseball_explainer",
    strategy: str = "static_kbo_faq",
    grounding_mode: str = "static_kbo_faq",
    source_tier: str = "internal_static",
    fallback_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "answer": answer,
        "citations": [],
        "intent": intent,
        "retrieved": [],
        "strategy": strategy,
        "verified": True,
        "tool_calls": [],
        "tool_results": [],
        "data_sources": [
            {
                "tool": strategy,
                "verified": True,
                "data_points": 1,
            }
        ],
        "visualizations": [],
        "planner_mode": "fast_path",
        "planner_cache_hit": False,
        "tool_execution_mode": "none",
        "fallback_triggered": bool(fallback_reason),
        "fallback_answer_used": False,
        "grounding_mode": grounding_mode,
        "source_tier": source_tier,
        "answer_sources": [source_tier],
        "as_of_date": None,
        "fallback_reason": fallback_reason,
        "perf": {
            "total_ms": 0.0,
            "analysis_ms": 0.0,
            "tool_ms": 0.0,
            "answer_ms": 0.0,
            "first_token_ms": 0.0,
            "tool_count": 0,
            "tool_execution_mode": "none",
            "planner_cache_hit": False,
            "planner_mode": "fast_path",
            "model": "static",
        },
    }


def _manual_baseball_data_required_answer(query: str) -> str:
    del query
    return (
        "MANUAL_BASEBALL_DATA_REQUIRED: 이 질문은 경기 당일 또는 시즌 중 변동 데이터가 필요합니다. "
        "운영자가 기준 날짜, 경기 ID, 팀명, 경기 상태, 점수, 선발, 라인업, 엔트리 변동, 관련 기록 범위를 "
        "내부 DB에 제공한 뒤 답변해야 합니다."
    )


def _extract_query_year(query: str) -> Optional[int]:
    match = re.search(r"\b(20\d{2})\b", query)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_event_pending_for_date(
    *,
    event_year: int,
    pending_until: Tuple[int, int],
    today: date,
) -> bool:
    if event_year > today.year:
        return True
    if event_year < today.year:
        return False
    pending_month, pending_day = pending_until
    return today < date(event_year, pending_month, pending_day)


def _has_explicit_single_game_date(query: str) -> bool:
    query_lower = query.lower()
    if "부터" in query_lower or "까지" in query_lower:
        return False
    if re.search(r"\b20\d{2}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일\b", query_lower):
        return True
    return bool(re.search(r"\b20\d{2}-\d{1,2}-\d{1,2}\b", query_lower))


def _has_explicit_game_date_range(query: str) -> bool:
    query_lower = query.lower()
    if not ("부터" in query_lower or "까지" in query_lower or "~" in query_lower):
        return False
    if re.search(
        r"\b20\d{2}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일.*\d{1,2}\s*월\s*\d{1,2}\s*일",
        query_lower,
    ):
        return True
    return bool(
        re.search(
            r"\b20\d{2}-\d{1,2}-\d{1,2}\b.*\b20\d{2}-\d{1,2}-\d{1,2}\b",
            query_lower,
        )
    )


def _is_db_answerable_schedule_query(query_lower: str) -> bool:
    has_supported_date = _has_explicit_single_game_date(
        query_lower
    ) or _has_explicit_game_date_range(query_lower)
    if not has_supported_date:
        return False
    if any(
        token in query_lower
        for token in (
            "중계",
            "라디오",
            "하이라이트",
            "다시보기",
            "스코어보드",
            "문자중계",
            "알림",
        )
    ):
        return False
    return any(
        token in query_lower
        for token in (
            "경기 일정",
            "경기일정",
            "경기표",
            "일정 알려",
            "일정 보여",
        )
    )


def _is_db_answerable_rank_lookup_query(query_lower: str) -> bool:
    return any(
        token in query_lower
        for token in (
            "1위 팀",
            "1위팀",
            "최하위 팀",
            "최하위팀",
        )
    )


def _is_db_answerable_standings_table_query(query_lower: str) -> bool:
    return any(
        token in query_lower
        for token in (
            "현재 kbo 순위",
            "kbo 순위",
            "순위표",
            "전체 순위",
            "팀별 승패",
            "팀별 승률",
            "팀별 승차",
        )
    ) and not any(token in query_lower for token in ("뉴스", "이슈", "소식", "가능성", "후보"))


def _is_doru_success_rate_explainer_query(query_lower: str) -> bool:
    return "도루 성공률" in query_lower and any(
        token in query_lower for token in ("어떻게", "계산", "뜻", "뭐야", "보는", "봐")
    )


def _is_db_answerable_leaderboard_or_team_form_query(query_lower: str) -> bool:
    query_compact = re.sub(r"[\s?.!,~]+", "", query_lower)
    if _is_doru_success_rate_explainer_query(query_lower):
        return False
    if any(
        token in query_lower
        for token in (
            "실책이 적은 선수",
            "홈런이 많은 시즌",
            "도루가 많은 시즌",
            "포수 도루저지율",
            "구종별 성적",
            "타구 속도",
            "존별 타격",
            "라인업 상성",
            "후보",
            "가능성",
            "뉴스",
            "이슈",
            "소식",
        )
    ):
        return False
    if any(token in query_compact for token in _TEAM_PAIR_COMPARISON_COMPACT_TOKENS):
        return True

    player_leader_tokens = (
        "장타율 1위",
        "ops 1위",
        "도루 1위",
        "득점 1위",
        "삼진이 많은 타자",
        "완투",
        "완봉",
        "이닝 소화",
        "피안타율",
        "도루 성공률",
    )
    team_metric_tokens = (
        "도루가 많은 팀",
        "실책이 적은 팀",
        "팀별 홈런",
        "팀별 도루",
        "팀별 실책",
        "팀별 타율",
        "팀별 출루율",
        "팀별 장타율",
        "팀별 ops",
        "팀별 era",
        "팀별 경기당 평균 득점",
        "팀별 경기당 평균 실점",
        "팀별 득점력",
        "팀별 실점력",
        "공격과 수비",
        "수비가 좋은 팀",
        "수비 좋은 팀",
        "내야 수비가 좋은 팀",
        "외야 수비가 좋은 팀",
        "병살타가 많은 팀",
        "불펜 era",
        "불펜 평균자책",
        "불펜 비교",
        "불펜 소모",
        "불펜이 좋은 팀",
        "선발진 비교",
        "선발진이 좋은 팀",
        "기복이 큰 팀",
        "타선 비교",
        "수비 비교",
        "주루 비교",
    )
    team_form_tokens = (
        "최근 10경기",
        "최근 5경기",
        "최근 흐름 비교",
        "홈 승률",
        "원정 승률",
        "연승 중인 팀",
        "연패 중인 팀",
    )
    return (
        any(token in query_lower for token in player_leader_tokens)
        or any(token in query_lower for token in team_metric_tokens)
        or any(token in query_lower for token in team_form_tokens)
    )


def _build_future_event_pending_result(
    query: str,
    *,
    today: Optional[date] = None,
) -> Optional[Dict[str, Any]]:
    query_lower = query.lower()
    if "올스타급" in query_lower:
        return None

    today = today or datetime.now().date()
    query_year = _extract_query_year(query)
    event_year = query_year or today.year

    for definition in _FUTURE_EVENT_DEFINITIONS:
        if not any(token in query_lower for token in definition["tokens"]):
            continue
        has_status_token = any(
            token in query_lower for token in _FUTURE_EVENT_STATUS_TOKENS
        )
        has_explainer_token = any(
            token in query_lower for token in _FUTURE_EVENT_EXPLAINER_TOKENS
        )
        if has_explainer_token and not has_status_token:
            return None
        if not has_status_token:
            return None
        if (
            definition["event_key"] == "season_awards"
            and "경기" in query_lower
            and query_year is None
            and "시즌" not in query_lower
        ):
            return None
        if not _is_event_pending_for_date(
            event_year=event_year,
            pending_until=definition["pending_until"],
            today=today,
        ):
            return None

        answer = (
            f"{event_year}년 {definition['label']}은 기준일({today.isoformat()}) 현재 "
            f"아직 진행 전인 이벤트로 분류합니다. "
            f"따라서 {definition['post_event_scope']}는 아직 확정 전입니다. "
            "개최일이나 장소처럼 사전 일정값을 답해야 하는 경우에는 내부 시즌 일정 데이터가 들어온 뒤 그 값을 기준으로 답변해야 합니다."
        )
        return _build_static_kbo_result(
            answer,
            intent="future_event_status",
            strategy="future_event_pending",
            grounding_mode="future_event_status",
            source_tier="system_clock_and_internal_policy",
            fallback_reason=None,
        )

    return None


def _is_live_manual_data_query(query: str) -> bool:
    query_lower = query.lower()
    query_compact = re.sub(r"[\s?.!,~]+", "", query_lower)
    if _is_db_answerable_schedule_query(query_lower):
        return False
    if _is_db_answerable_rank_lookup_query(query_lower):
        return False
    if _is_db_answerable_standings_table_query(query_lower):
        return False
    if _is_db_answerable_leaderboard_or_team_form_query(query_lower):
        return False
    if _is_doru_success_rate_explainer_query(query_lower):
        return False
    if any(token in query_lower for token in _REGULATION_MANUAL_DATA_TOKENS):
        return True
    if any(token in query_lower for token in _LEADERBOARD_MANUAL_DATA_TOKENS):
        return True
    if any(token in query_lower for token in _VAGUE_GAME_DETAIL_TOKENS):
        return True
    if any(token in query_lower for token in _LEAGUE_WIDE_CURRENT_DATA_TOKENS):
        return True
    if any(token in query_lower for token in _FAN_EXPERIENCE_MANUAL_DATA_TOKENS):
        return True
    if any(token in query_lower for token in _ROSTER_EVALUATION_MANUAL_DATA_TOKENS):
        return True
    if any(token in query_compact for token in _TEAM_PAIR_COMPARISON_COMPACT_TOKENS):
        return True
    if any(token in query_lower for token in _SCHEDULE_MANUAL_DATA_TOKENS):
        if any(
            marker in query_lower
            for marker in (
                "2026",
                "오늘",
                "내일",
                "이번 주",
                "어디서",
                "언제",
                "볼 수",
                "들을",
                "알려줘",
                "보여줘",
                "확인",
            )
        ):
            return True
    if not any(token.lower() in query_lower for token in _LIVE_MANUAL_DATA_TIME_TOKENS):
        return False
    if "순위" in query_lower and not any(
        token in query_lower for token in ("뉴스", "이슈", "상황", "소식")
    ):
        return False
    return any(
        token.lower() in query_lower for token in _LIVE_MANUAL_DATA_SUBJECT_TOKENS
    )


def _extract_static_team_name(query: str) -> Optional[str]:
    query_lower = query.lower()
    for alias, team_name in sorted(
        _STATIC_TEAM_ALIAS_MAP.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if alias in query_lower:
            return team_name
    return None


def _build_static_chatbot_meta_result(query: str) -> Optional[Dict[str, Any]]:
    query_lower = query.lower()

    if "데이터" in query_lower and any(
        token in query_lower for token in ("어디서 가져와", "출처", "소스")
    ):
        return _build_static_kbo_result(
            "이 서비스의 KBO 답변은 내부 DB와 운영자가 제공한 신뢰 데이터 기준으로 생성해야 합니다. "
            "일정, 중계, 부상, 뉴스처럼 실시간으로 바뀌는 항목은 운영자가 기준일과 원천 데이터를 넣은 뒤 답변하는 구조가 맞습니다.",
            intent="chatbot_meta",
            strategy="static_chatbot_meta",
            grounding_mode="static_chatbot_meta",
        )

    if "kbo api" in query_lower and any(
        token in query_lower for token in ("설계", "구성", "어떻게")
    ):
        return _build_static_kbo_result(
            "KBO API는 일정/경기, 팀, 선수, 기록, 순위, RAG 문서, 운영자 제공 데이터 상태를 분리해 설계하는 것이 좋습니다. 실시간성 데이터는 기준일과 근거 상태를 함께 내려주고, 누락 시에는 수동 데이터 필요 계약을 반환하도록 둬야 합니다.",
            intent="chatbot_meta",
            strategy="static_chatbot_meta",
            grounding_mode="static_chatbot_meta",
        )

    if "챗봇" in query_lower and any(
        token in query_lower
        for token in (
            "기능",
            "메뉴",
            "대화 흐름",
            "faq",
            "프롬프트",
            "테스트 질문",
            "성능 점검",
            "정확도 확인",
            "안내 문구",
            "문구",
            "환영 문구",
            "응답 포맷",
            "상태 관리",
            "검색 기능",
            "알림 기능",
            "캐싱",
            "성능 최적화",
            "추천 시스템",
            "로그 설계",
            "말투",
            "답변 샘플",
            "질문 샘플",
            "시나리오",
            "일정",
            "순위",
            "선수 기록",
            "경기 분석",
            "예측",
            "응원 문구",
            "요약 답변",
            "표로 정리",
            "알림 문구",
            "대화형",
        )
    ):
        return _build_static_kbo_result(
            "KBO 챗봇은 기록 조회, 팀/선수 요약, 경기 전후 체크리스트, 규정 설명, 직관 안내, 응원 문구를 메뉴로 나누는 구성이 적합합니다. "
            "실시간 일정, 부상, 뉴스, 중계 정보는 내부 DB에 기준일과 원천 데이터가 들어온 경우에만 답변하도록 제한해야 합니다.",
            intent="chatbot_meta",
            strategy="static_chatbot_meta",
            grounding_mode="static_chatbot_meta",
        )

    if any(token in query_lower for token in ("경기 알림 설정", "응원팀 알림")):
        return _build_static_kbo_result(
            "알림 관련 질문은 챗봇이 경기 일정과 응원팀 기준을 설명할 수는 있지만, 실제 푸시 알림 설정 여부는 앱의 알림 기능과 운영 설정을 확인해야 합니다. 기능형 답변은 야구 기록 데이터와 분리해 처리하는 것이 맞습니다.",
            intent="chatbot_meta",
            strategy="static_chatbot_meta",
            grounding_mode="static_chatbot_meta",
        )

    if "질문" in query_lower and any(
        token in query_lower
        for token in (
            "만들어",
            "묶어",
            "나눠",
            "바꿔",
            "다듬어",
            "변환",
            "정리",
            "faq",
            "서비스용",
            "앱용",
            "사이트용",
            "테스트용",
            "운영 매뉴얼",
            "꼭 필요한",
            "가장 많이 묻는",
            "꼭 알아야",
            "꼭 볼",
        )
    ):
        return _build_static_kbo_result(
            "질문은 일정/결과, 순위, 팀 전력, 선수 기록, 경기 분석, 규정, 직관, 응원, 데이터 근거로 나누면 운영하기 쉽습니다. "
            "예시는 '오늘 경기 일정 알려줘', 'LG 최근 승률 흐름은 어때?', '홈런 현재 선두는 누구야?', '비디오판독은 언제 가능해?'처럼 구체화하는 방식이 좋습니다.",
            intent="chatbot_meta",
            strategy="static_chatbot_meta",
            grounding_mode="static_chatbot_meta",
        )

    if "faq" in query_lower and any(
        token in query_lower for token in ("만들어", "완성", "변환", "구성")
    ):
        return _build_static_kbo_result(
            "KBO FAQ는 초보자용 규칙, 일정/결과 확인, 순위와 승률, 선수 기록, 팀 전력, 경기 분석, 직관, 응원, 데이터 근거 항목으로 구성하면 됩니다. 실시간 일정과 뉴스성 항목은 기준일 데이터가 있을 때만 답하도록 분리해야 합니다.",
            intent="chatbot_meta",
            strategy="static_chatbot_meta",
            grounding_mode="static_chatbot_meta",
        )

    if any(
        token in query_lower
        for token in (
            "답변 예시",
            "문구 예시",
            "설명 예시",
            "안내 예시",
            "응답 예시",
            "유도 예시",
            "메시지",
            "멘트 추천",
            "쓸 문구",
        )
    ):
        return _build_static_kbo_result(
            "예시는 짧은 핵심 문장 뒤에 근거를 붙이는 형식이 적합합니다. 응원 문구라면 '오늘 흐름은 아직 살아 있어요. 다음 이닝 한 번 더 밀어붙이면 됩니다.'처럼 팀명과 상황을 넣어 자연스럽게 바꾸면 됩니다.",
            intent="chatbot_meta",
            strategy="static_chatbot_meta",
            grounding_mode="static_chatbot_meta",
        )

    if any(token in query_lower for token in ("직관 팁", "비 오는 날", "여름 직관")):
        return _build_static_kbo_result(
            "직관 안내는 날씨와 구장 공지를 먼저 확인하고, 우천 가능성이 있으면 우비와 방수 보관용 봉투, 여름 경기라면 물과 모자, 보조 배터리를 챙기는 식으로 답하면 됩니다. 좌석, 입장, 이벤트 정보는 구장별 공지가 필요합니다.",
            intent="chatbot_meta",
            strategy="static_chatbot_meta",
            grounding_mode="static_chatbot_meta",
        )

    if "야구장 응원 예절" in query_lower:
        return _build_static_kbo_result(
            "야구장 응원은 주변 시야를 가리지 않고, 통로를 막지 않으며, 상대 팀과 선수에게 모욕적인 표현을 쓰지 않는 것이 기본입니다. 큰 응원 도구나 이벤트 참여 방식은 구장 공지를 확인하는 편이 안전합니다.",
            intent="chatbot_meta",
            strategy="static_chatbot_meta",
            grounding_mode="static_chatbot_meta",
        )

    if any(
        token in query_lower
        for token in (
            "응원 톤",
            "짧게 답",
            "자세히 답",
            "표로 답",
            "한 줄로 답",
            "초보자 기준",
            "데이터 중심",
            "팬 입장",
            "중립적으로",
            "재미있게",
        )
    ):
        return _build_static_kbo_result(
            "답변 톤은 질문 목적에 맞춰 바꾸면 됩니다. 경기 기록 질문은 수치와 기준일을 먼저 쓰고, 초보자 질문은 용어를 풀어 설명하며, 팬덤형 질문은 단정 대신 현재 흐름과 관전 포인트를 중심으로 답하는 방식이 안전합니다.",
            intent="chatbot_meta",
            strategy="static_chatbot_meta",
            grounding_mode="static_chatbot_meta",
        )

    return None


def _build_static_kbo_faq_result(query: str) -> Optional[Dict[str, Any]]:
    query_lower = query.lower().strip()
    compact = re.sub(r"[\s?.!,~]+", "", query_lower)
    static_team_name = _extract_static_team_name(query)

    chatbot_meta_result = _build_static_chatbot_meta_result(query)
    if chatbot_meta_result is not None:
        return chatbot_meta_result

    future_event_result = _build_future_event_pending_result(query)
    if future_event_result is not None:
        return future_event_result

    if any(token in query_lower for token in ("이 팀", "우리 팀", "해당 팀")):
        return _build_static_kbo_result(
            "어떤 팀 기준인지 팀명을 같이 알려주세요. 예: LG, KIA, KT처럼 팀명을 붙이면 바로 DB 기준으로 조회하겠습니다.",
            intent="clarification_required",
            strategy="clarification_required",
            grounding_mode="clarification_required",
            source_tier="none",
            fallback_reason="clarification_required",
        )

    if any(
        token in query_lower for token in ("두 선수", "선수 비교")
    ) and not re.search(
        r"[가-힣A-Za-z]{2,}\s*(?:과|와|vs|VS)\s*[가-힣A-Za-z]{2,}",
        query,
    ):
        return _build_static_kbo_result(
            "비교할 두 선수 이름을 같이 알려주세요. 예: 김도영과 문보경처럼 두 이름이 필요합니다.",
            intent="clarification_required",
            strategy="clarification_required",
            grounding_mode="clarification_required",
            source_tier="none",
            fallback_reason="clarification_required",
        )

    if _is_live_manual_data_query(query):
        return _build_static_kbo_result(
            _manual_baseball_data_required_answer(query),
            intent="manual_data_request",
            strategy="manual_baseball_data_required",
            grounding_mode="manual_data_request",
            source_tier="none",
            fallback_reason="manual_baseball_data_required",
        )

    if _is_db_answerable_leaderboard_or_team_form_query(query_lower):
        return None

    if "홈런" in query_lower and any(
        token in query_lower
        for token in (
            "왕",
            "1위",
            "최다",
            "선두",
            "누구",
            "몇 개",
            "몇개",
            "순위",
            "기록",
        )
    ):
        return None

    for tokens, answer in _STATIC_EXPLAINER_ANSWERS:
        if any(token in query_lower for token in tokens):
            return _build_static_kbo_result(
                answer,
                intent="baseball_explainer",
                strategy="static_baseball_explainer",
                grounding_mode="static_baseball_explainer",
            )

    if ("kbo" in query_lower and "리그" in query_lower and "어떤" in query_lower) or (
        "kbo가뭔지" in compact
    ):
        return _build_static_kbo_result(
            "KBO 리그는 한국의 최상위 프로야구 리그입니다. 10개 구단이 정규시즌을 치르고, 상위권 팀들이 포스트시즌을 거쳐 한국시리즈 우승을 다투는 구조로 보면 됩니다."
        )

    if "kbo" in query_lower and any(
        token in query_lower for token in ("언제 시작", "시작했")
    ):
        return _build_static_kbo_result(
            "KBO 리그는 1982년에 출범했습니다. 한국 프로야구의 1군 최상위 리그로 시작해 지금은 10개 구단 체제로 운영됩니다."
        )

    if "kbo" in query_lower and any(
        token in query_lower for token in ("몇 개 팀", "몇개 팀", "10개 구단")
    ):
        teams = ", ".join(_STATIC_KBO_TEAMS)
        return _build_static_kbo_result(
            f"KBO 리그는 현재 10개 구단 체제입니다. 구단은 {teams}입니다."
        )

    if static_team_name and any(
        token in query_lower for token in ("어떤 팀", "무슨 팀")
    ):
        stadium = _STATIC_TEAM_STADIUMS.get(static_team_name)
        stadium_text = f" 홈구장은 {stadium}입니다." if stadium else ""
        return _build_static_kbo_result(
            f"{static_team_name}는 KBO 리그의 10개 구단 중 하나입니다.{stadium_text} 더 정확한 시즌 전력이나 최근 성적은 순위, 승패, 타격, 투구처럼 질문 축을 좁히면 DB 기록 기준으로 이어서 볼 수 있습니다.",
            intent="team_profile",
            grounding_mode="team_profile_static",
        )

    if "각 팀" in query_lower and "홈구장" in query_lower:
        stadium_text = ", ".join(
            f"{team}는 {stadium}" for team, stadium in _STATIC_TEAM_STADIUMS.items()
        )
        return _build_static_kbo_result(
            f"KBO 10개 구단 홈구장은 {stadium_text}입니다.",
            intent="team_profile",
            grounding_mode="team_profile_static",
        )

    if "각 팀" in query_lower and any(
        token in query_lower for token in ("감독", "대표 선수")
    ):
        return _build_static_kbo_result(
            _manual_baseball_data_required_answer(query),
            intent="manual_data_request",
            strategy="manual_baseball_data_required",
            grounding_mode="manual_data_request",
            source_tier="none",
            fallback_reason="manual_baseball_data_required",
        )

    if "각 팀" in query_lower and "팀 컬러" in query_lower:
        return _build_static_kbo_result(
            "팀 컬러는 보통 구단 역사, 홈구장 분위기, 팬덤 문화, 최근 전력 성향을 묶어서 설명합니다. 정확한 시즌별 팀 컬러를 비교하려면 팀별 순위, 타격, 투구, 수비 지표를 같은 기준일로 맞춰 보는 게 좋습니다.",
            intent="team_profile",
            grounding_mode="team_profile_static",
        )

    if "정규시즌" in query_lower and any(
        token in query_lower for token in ("몇 경기", "몇경기")
    ):
        return _build_static_kbo_result(
            "KBO 정규시즌은 보통 팀당 144경기로 운영됩니다. 그래서 순위와 승률 흐름을 볼 때는 긴 시즌 누적 성적과 최근 흐름을 함께 봐야 합니다."
        )

    if "순위" in query_lower and (
        any(token in query_lower for token in ("결정", "산정"))
        or (
            "정" in query_lower
            and any(token in query_lower for token in ("어떻게", "기준"))
        )
        or "순위를정" in compact
    ):
        return _build_static_kbo_result(
            "KBO 정규시즌 순위는 기본적으로 승률을 먼저 봅니다. 승률이 같으면 상대 전적, 동률 팀 간 득점, 전년도 순위 같은 타이브레이크 기준을 차례로 적용하는 구조입니다."
        )

    if "승률" in query_lower and any(
        token in query_lower for token in ("계산", "어떻게")
    ):
        return _build_static_kbo_result(
            "KBO 승률은 승수 / (승수 + 패수)로 계산합니다. 무승부는 승률 분모에 넣지 않기 때문에, 같은 승수라도 패수가 적은 팀이 승률에서 유리해질 수 있습니다."
        )

    if "무승부" in query_lower and any(
        token in query_lower for token in ("순위", "반영", "규정")
    ):
        return _build_static_kbo_result(
            "KBO 순위에서 무승부는 승률 계산의 분모에서 제외됩니다. 예를 들어 승률은 승수 / (승수 + 패수)로 보며, 동률 상황은 별도 타이브레이크 기준을 함께 적용합니다."
        )

    if "포스트시즌" in query_lower and any(
        token in query_lower for token in ("어떻게", "진행")
    ):
        return _build_static_kbo_result(
            "KBO 포스트시즌은 정규시즌 상위 팀들이 단계적으로 맞붙는 토너먼트입니다. 와일드카드 결정전에서 시작해 준플레이오프, 플레이오프를 거쳐 한국시리즈에서 최종 우승팀을 가립니다."
        )

    if "한국시리즈" in query_lower and any(
        token in query_lower for token in ("몇 경기", "몇경기")
    ):
        return _build_static_kbo_result(
            "KBO 한국시리즈는 7전 4선승제입니다. 먼저 4승을 거둔 팀이 그 시즌 최종 우승팀이 됩니다."
        )

    if "와일드카드" in query_lower and any(
        token in query_lower for token in ("어떻게", "결정", "유리")
    ):
        return _build_static_kbo_result(
            "KBO 와일드카드 결정전은 정규시즌 4위와 5위가 맞붙는 첫 관문입니다. 4위 팀이 유리한 조건에서 시작하고, 5위 팀은 연속으로 이겨야 다음 단계에 올라가는 구조입니다."
        )

    if "연장전" in query_lower and any(
        token in query_lower for token in ("몇 회", "몇회", "규정")
    ):
        return _build_static_kbo_result(
            "2026년 KBO 정규시즌 연장전은 11회까지로 보는 규정 변경이 반영되어 있습니다. 연장에서도 승부가 나지 않으면 무승부로 처리됩니다."
        )

    if "아시아 쿼터" in query_lower:
        return _build_static_kbo_result(
            "2026년 KBO 아시아 쿼터는 기존 외국인 선수 운용에 아시아 국적 선수 슬롯을 더해 선수 수급 폭을 넓히는 제도입니다. 시스템에서는 외국인 선수 4명 체계와 함께 시즌 규정으로 분리해 다루는 항목입니다."
        )

    if "외국인 선수" in query_lower and any(
        token in query_lower for token in ("몇 명", "몇명", "규정")
    ):
        return _build_static_kbo_result(
            "2026년 KBO 외국인 선수 운용은 외국인 4명 체계로 보는 규정 변경이 반영되어 있습니다. 세부 등록과 출장 조건은 시즌 규정 항목으로 따로 관리해야 합니다."
        )

    return None


def _build_retrieval_event_filter(
    final_filters: Dict[str, Any],
    *,
    actual_filters: Optional[Dict[str, Any]] = None,
    fallback_used: bool = False,
    fallback_stage: Optional[str] = "none",
) -> Dict[str, Any]:
    event_filter = dict(final_filters or {})
    event_filter["original_filters"] = dict(final_filters or {})
    event_filter["actual_filters"] = dict(
        actual_filters if actual_filters is not None else final_filters or {}
    )
    event_filter["fallback_used"] = bool(fallback_used)
    event_filter["fallback_stage"] = fallback_stage or "none"
    return event_filter


def _build_citations(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for doc in docs:
        citation = {
            "id": doc.get("id"),
            "title": doc.get("title", ""),
            "source_table": doc.get("source_table"),
            "source_row_id": doc.get("source_row_id"),
            "source_type": doc.get("source_type"),
            "source_uri": doc.get("source_uri"),
            "topic_key": doc.get("topic_key")
            or (doc.get("meta") or {}).get("topic_key"),
            "similarity": doc.get("similarity"),
            "combined_score": doc.get("combined_score"),
            "quality_score": doc.get("quality_score"),
            "valid_from": doc.get("valid_from"),
            "valid_to": doc.get("valid_to"),
            "expires_at": doc.get("expires_at"),
        }
        citations.append(
            {key: value for key, value in citation.items() if value is not None}
        )
    return citations


def _get_safe_stat(meta: Dict, key: str, default: Any = None) -> Optional[float]:
    """메타 데이터에서 안전하게 실수 값을 가져옵니다."""
    val = meta.get(key)
    if val in (None, ""):
        return default
    try:
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val).strip())
    except (ValueError, TypeError):
        return default


def _get_team_name(raw_name: str) -> str:
    """팀 이름을 표준화합니다."""
    return TEAM_MAP.get(raw_name, raw_name)


def batter_rank_score(wrc_plus, war):
    """wRC+와 WAR을 가중 평균하여 타자 점수를 계산합니다."""
    wrc_plus_score = wrc_plus if wrc_plus is not None else 80
    war_score = (war * 20) if war is not None else 0  # WAR 1승당 wRC+ 20점 가치로 환산
    return 0.7 * wrc_plus_score + 0.3 * war_score


HISTORY_CONTEXT_LIMIT = 6
_RAG_CACHE_MAX = 10000
_LEAGUE_CONTEXT = kbo_metrics.LeagueContext()
_RRF_QUERY_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣+#]+")
_RRF_STOPWORDS = frozenset(
    {
        "알려줘",
        "알려",
        "말해줘",
        "말해",
        "해줘",
        "해주세요",
        "좀",
        "대한",
        "대해",
        "관련",
        "무엇",
        "뭐",
        "어떤",
        "누구",
        "the",
        "a",
        "an",
        "tell",
        "me",
        "about",
    }
)
_RRF_MAX_QUERY_TOKENS = 4
_RRF_LOG_PREVIEW_LEN = 140
RetrievalState = Dict[str, Any]

# --- Intent-branching keyword sets (frozenset for O(1) lookup) ---
# 각 메서드(`_is_statistical_query`, `_is_regulation_query`, `_is_game_query`,
# `_is_game_flow_narrative_query`, `_is_general_conversation`)가 매 요청마다
# 키워드 list를 새로 만들지 않도록 모듈 레벨 상수로 추출.
# 모두 query.lower() 기준으로 substring match에 사용된다.
_STAT_CHITCHAT_KEYWORDS = frozenset(
    {
        "안녕",
        "누구",
        "좋아해",
        "응원",
        "날씨",
        "어때",
        "뭐해",
        "어디",
        "언제",
        "왜",
        "어떻게",
        "고마워",
        "미안",
        "반가워",
        "잘가",
        "소개",
        "설명",
        "도움",
        "기능",
        "사용법",
    }
)
_STATISTICAL_KEYWORDS = frozenset(
    {
        "타율",
        "홈런",
        "타점",
        "득점",
        "ops",
        "era",
        "방어율",
        "whip",
        "승",
        "패",
        "세이브",
        "홀드",
        "삼진",
        "볼넷",
        "출루율",
        "장타율",
        "wrc+",
        "war",
        "fip",
        "babip",
        "몇위",
        "순위",
        "1위",
        "최고",
        "상위",
        "리더",
        "기록",
        "통계",
        "성적",
        "몇개",
        "몇점",
        "얼마나",
        "vs",
        "대",
        "비교",
        "누가",
        "더",
        "뛰어난",
        "우수한",
        "좋은",
        "맞대결",
    }
)
_STAT_SPECIFIC_REQUEST_WORDS = frozenset(
    {
        "알려줘",
        "궁금",
        "얼마",
        "몇",
        "는",
        "은",
        "의",
    }
)
_STAT_CORE_INDICATORS = frozenset(
    {
        "타율",
        "홈런",
        "ops",
        "era",
        "방어율",
    }
)
# 규정 청크가 저장된 source_table 목록. similarity_search의 source_table_in 필터에 사용.
_REGULATION_SOURCES: frozenset[str] = frozenset(
    {
        "markdown_docs",
        "kbo_regulations",
        "kbo_definitions",
    }
)
_DEFINITION_CONTEXT_MARKERS: frozenset[str] = frozenset(
    {"컬럼", "테이블", "스키마", "extra_stats"}
)


def _is_regulation_query_text(query: str) -> bool:
    query_lower = str(query or "").lower()
    if any(marker in query_lower for marker in _DEFINITION_CONTEXT_MARKERS):
        return False
    query_nospace = query_lower.replace(" ", "")
    return any(
        keyword in query_lower or keyword in query_nospace
        for keyword in _REGULATION_KEYWORDS
    )


def _context_source_priority(
    query: str, *, is_regulation: Optional[bool] = None
) -> Dict[str, int]:
    regulation_query = (
        _is_regulation_query_text(query)
        if is_regulation is None
        else is_regulation
    )
    if regulation_query:
        return {
            "kbo_regulations": 0,
            "kbo_definitions": 1,
            "markdown_docs": 2,
        }
    query_lower = str(query or "").lower()
    if any(marker in query_lower for marker in _DEFINITION_CONTEXT_MARKERS):
        return {
            "kbo_definitions": 0,
            "markdown_docs": 1,
            "kbo_regulations": 2,
        }
    return {"markdown_docs": 0, "kbo_definitions": 1, "kbo_regulations": 2}


def _sort_docs_for_context(
    docs: List[Dict[str, Any]], *, is_regulation: bool, query: str = ""
) -> List[Dict[str, Any]]:
    """Order retrieved context without overriding semantic relevance within a source."""
    source_priority = _context_source_priority(
        query,
        is_regulation=is_regulation,
    )
    return sorted(
        docs,
        key=lambda doc: source_priority.get(str(doc.get("source_table") or ""), 3),
    )

_REGULATION_KEYWORDS = frozenset(
    {
        "규정",
        "규칙",
        "룰",
        "조항",
        "가능해",
        "허용",
        "금지",
        "벌칙",
        "징계",
        "반칙",
        "파울",
        "아웃",
        "세이프",
        "스트라이크",
        "볼",
        "홈런",
        "인플레이",
        "타이브레이크",
        "지명타자",
        "연장전",
        "콜드게임",
        "더블헤더",
        "비디오판독",
        "FA",
        "자유계약",
        "외국인선수",
        "몇명까지",
        "드래프트",
        "트레이드",
        "도박",
        "폭력",
        "약물",
        "심판",
        "모독",
        "퇴장",
        "플레이오프",
        "포스트시즌",
        "와일드카드",
        "웨이버",
        "한국시리즈",
        "몇팀",
        "보크",
        "방해",
        "인필드플라이",
        "그라운드룰",
        "몸에맞는공",
        "용어",
        "뜻",
        "의미",
        "정의",
        "설명",
        "조건",
        "언제적용",
        "세이브조건",
        "승리투수",
        "왕",
        "홈런왕",
        "득점왕",
        "순위",
        # 통계 지표(wrc+, ops, era, whip, babip, war, fip)는 제거:
        # → _STATISTICAL_KEYWORDS에 이미 포함되어 있고,
        #   규정 쿼리 오탐 시 source_table_in 필터로 통계 청크가 제외되는 문제 방지.
    }
)
_GAME_KEYWORDS = frozenset(
    {
        "경기",
        "게임",
        "박스스코어",
        "스코어",
        "결과",
        "이닝별",
        "이닝별 득점",
        "몇 점",
        "7회",
        "8회",
        "9회",
        "연장",
        "오늘",
        "어제",
        "내일",
        "날짜",
        "언제",
        "몇일",
        "며칠",
        "vs",
        "대",
        "맞대결",
        "직접대결",
        "상대전적",
        "시리즈",
        "승부",
        "이겼",
        "졌",
        "무승부",
        "점수",
        "승리",
        "패배",
        "홈",
        "원정",
        "away",
        "home",
        "구장에서",
        "에서",
        "몇점",
        "득점",
        "실점",
        "타점",
        "안타",
        "홈런친",
        "투구",
        "선발",
        "등판",
        "세이브",
        "홀드",
        "승",
        "패",
    }
)
# Game query 보조: 날짜 패턴 (미리 컴파일)
_GAME_DATE_PATTERNS = (
    re.compile(r"\d{4}-\d{1,2}-\d{1,2}"),  # 2025-10-15
    re.compile(r"\d{1,2}/\d{1,2}"),  # 10/15
    re.compile(r"\d{1,2}월\s*\d{1,2}일"),  # 10월 15일
    re.compile(r"오늘|어제|내일|모레|그저께"),
)
# Game query 보조: 팀 vs 팀 (case-insensitive, 미리 컴파일)
_GAME_TEAM_VS_PATTERN = re.compile(
    r"(KIA|HT|LG|DB|DO|OB|두산|LT|롯데|SS|삼성|KH|KI|WO|NX|키움|HH|한화|KT|NC|SSG|SK)"
    r".*(vs|대|vs\.|대전|맞대결).*"
    r"(KIA|HT|LG|DB|DO|OB|두산|LT|롯데|SS|삼성|KH|KI|WO|NX|키움|HH|한화|KT|NC|SSG|SK)",
    re.IGNORECASE,
)
_GAME_FLOW_NARRATIVE_KEYWORDS = frozenset(
    {
        "경기 흐름",
        "흐름 요약",
        "승부처",
        "언제 갈렸어",
        "언제 갈렸",
        "역전",
        "동점 흐름",
        "초중후반 득점",
        "득점 양상",
    }
)
_GENERAL_BASEBALL_KEYWORDS = frozenset(
    {
        "ops",
        "wrc+",
        "war",
        "era",
        "whip",
        "babip",
        "fip",
        "골든글러브",
        "fa",
        "신인왕",
        "mvp",
        "타율",
        "방어율",
        "출루율",
        "장타율",
        "자책점",
        "세이브",
        "홀드",
        "승리투수",
        "뜻",
        "의미",
        "정의",
        "계산",
        "어떻게",
        "무엇",
        "기준",
        # 통계 질문 키워드 추가
        "홈런",
        "타점",
        "득점",
        "승",
        "패",
        "삼진",
        "볼넷",
        "몇위",
        "순위",
        "1위",
        "최고",
        "상위",
        "리더",
        "기록",
        "통계",
        "성적",
        "몇개",
        "몇점",
        "얼마나",
        "얼마",
        "vs",
        "대",
        "비교",
        "누가",
        "더",
        "뛰어난",
        "우수한",
        "좋은",
        "맞대결",
        "시즌",
        "년",
        "연도",
    }
)
_GENERAL_CONVERSATION_KEYWORDS = frozenset(
    {
        "안녕",
        "누구",
        "좋아해",
        "응원",
        "날씨",
        "어때",
        "뭐해",
        "고마워",
        "미안",
        "반가워",
        "잘가",
        "소개",
        "도움",
        "기능",
        "사용법",
    }
)


def _meta_cache_key(meta: Dict[str, Any]) -> str:
    try:
        return json.dumps(
            meta, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str
        )
    except (TypeError, ValueError):
        return str(meta)


def _preview_text(value: str, max_len: int = _RRF_LOG_PREVIEW_LEN) -> str:
    text = (value or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _append_unique_term(terms: List[str], seen: set[str], term: Any) -> None:
    normalized = str(term).strip()
    if not normalized:
        return
    key = normalized.lower()
    if key in seen:
        return
    seen.add(key)
    terms.append(normalized)


def _build_rrf_keyword(query: str, entity_filter: Optional[Any]) -> str:
    terms: List[str] = []
    seen: set[str] = set()

    if entity_filter is not None:
        player_name = getattr(entity_filter, "player_name", None)
        team_id = getattr(entity_filter, "team_id", None)
        season_year = getattr(entity_filter, "season_year", None)

        if player_name:
            _append_unique_term(terms, seen, player_name)
        if team_id:
            _append_unique_term(terms, seen, TEAM_MAP.get(str(team_id), str(team_id)))
        if season_year:
            _append_unique_term(terms, seen, season_year)

        for attr in (
            "stat_type",
            "award_type",
            "movement_type",
            "position_type",
            "game_date",
        ):
            value = getattr(entity_filter, attr, None)
            if value in (None, "", "any"):
                continue
            _append_unique_term(terms, seen, value)

    query_tokens_added = 0
    for token in _RRF_QUERY_TOKEN_PATTERN.findall(query):
        normalized = token.strip()
        lowered = normalized.lower()
        if (
            not normalized
            or lowered in _RRF_STOPWORDS
            or (len(normalized) == 1 and not normalized.isdigit())
        ):
            continue
        _append_unique_term(terms, seen, normalized)
        query_tokens_added += 1
        if query_tokens_added >= _RRF_MAX_QUERY_TOKENS:
            break

    if not terms:
        return query
    return " ".join(terms)


def _new_retrieval_state() -> RetrievalState:
    return {
        "db_error": None,
        "embedding_error": None,
        "partial_errors": [],
        "error_type": None,
    }


def _record_retrieval_state_error(
    retrieval_state: Optional[RetrievalState],
    *,
    error_type: str,
    message: str,
) -> None:
    if retrieval_state is None:
        return
    retrieval_state["error_type"] = error_type
    if error_type == "db_unavailable":
        retrieval_state["db_error"] = message
    elif error_type == "embedding_failed":
        retrieval_state["embedding_error"] = message
    partial_errors = retrieval_state.setdefault("partial_errors", [])
    if isinstance(partial_errors, list):
        partial_errors.append({"error_type": error_type, "message": message})


def _resolve_default_season_year(settings: Settings) -> int:
    configured = getattr(settings, "default_kbo_season_year", None)
    if configured:
        return int(configured)
    return datetime.now().year


class MetaWrapper:
    """
    메타 데이터를 감싸서 효율적인 캐싱을 지원하는 래퍼 클래스입니다.
    source_row_id가 있으면 이를 해시 키로 사용하고, 없으면 전체 메타 데이터의 JSON 문자열을 사용합니다.
    """

    def __init__(self, meta: Dict[str, Any]):
        self.meta = meta
        self._hash_key = self._generate_hash_key()

    def _generate_hash_key(self) -> str:
        row_id = self.meta.get("source_row_id")
        if row_id:
            content_hash_value = self.meta.get("content_hash")
            if content_hash_value:
                return f"id:{row_id}:hash:{content_hash_value}"
            updated_at = self.meta.get("updated_at")
            if updated_at:
                return f"id:{row_id}:updated:{updated_at}"

        return _meta_cache_key(self.meta)

    def __hash__(self):
        return hash(self._hash_key)

    def __eq__(self, other):
        if not isinstance(other, MetaWrapper):
            return False
        return self._hash_key == other._hash_key


@lru_cache(maxsize=_RAG_CACHE_MAX)
def _process_stat_doc_cached(
    source_table: str, meta_wrapper: MetaWrapper
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    meta = meta_wrapper.meta  # Wrapper에서 직접 meta 접근

    # [Optimized] If score is already pre-calculated in meta, use it directly
    if "score" in meta:
        return meta, None

    if source_table == "player_season_pitching":
        ip = _get_safe_stat(meta, "innings_pitched", 0.0)
        gs = _get_safe_stat(meta, "games_started", 0)

        role = "SP" if ip >= MIN_IP_SP or gs >= 10 else "RP"
        min_ip_threshold = MIN_IP_SP if role == "SP" else MIN_IP_RP

        if ip < min_ip_threshold:
            return None, (
                f"'{meta.get('player_name', 'N/A')}' 선수는 표본 부족(IP < {min_ip_threshold})으로 제외되었습니다."
            )

        era = _get_safe_stat(meta, "era", 99.0)
        whip = _get_safe_stat(meta, "whip", 99.0)
        k = _get_safe_stat(meta, "strikeouts")
        bb = _get_safe_stat(meta, "walks_allowed")
        hbp = _get_safe_stat(meta, "hit_batters")
        hr = _get_safe_stat(meta, "home_runs_allowed")
        pa = _get_safe_stat(meta, "tbf", 0)

        fip_val = kbo_metrics.fip(hr, bb, hbp, k, ip, _LEAGUE_CONTEXT) or 99.0
        era_minus_val = kbo_metrics.era_minus(era, _LEAGUE_CONTEXT) or 999.0
        fip_minus_val = kbo_metrics.fip_minus(fip_val, _LEAGUE_CONTEXT) or 999.0
        kbb_pct = kbo_metrics.k_minus_bb_pct(k, bb, pa) or -99.0

        score = kbo_metrics.pitcher_rank_score(
            era_minus_val, fip_minus_val, kbb_pct, whip, ip
        )

        return {
            "name": meta.get("player_name", "N/A"),
            "team": _get_team_name(meta.get("team_name", "N/A")),
            "role": role,
            "ip": ip,
            "era": era,
            "whip": whip,
            "kbb_pct": kbb_pct,
            "era_minus": era_minus_val,
            "fip_minus": fip_minus_val,
            "score": score,
        }, None

    if source_table == "player_season_batting":
        pa = int(_get_safe_stat(meta, "plate_appearances", 0) or 0)
        if pa < MIN_PA_BATTER:
            return None, (
                f"'{meta.get('player_name', 'N/A')}' 선수는 표본 부족(PA < {MIN_PA_BATTER})으로 제외되었습니다."
            )

        wrc_plus = _get_safe_stat(meta, "wrc_plus")
        ops_plus = _get_safe_stat(meta, "ops_plus")
        war = _get_safe_stat(meta, "war")
        ops_val = _get_safe_stat(meta, "ops")
        obp = _get_safe_stat(meta, "obp")
        slg = _get_safe_stat(meta, "slg")
        avg = _get_safe_stat(meta, "avg")

        hits = _to_int(meta.get("hits"))
        doubles = _to_int(meta.get("doubles"))
        triples = _to_int(meta.get("triples"))
        home_runs = _to_int(meta.get("home_runs") or meta.get("hr"))
        walks = _to_int(meta.get("walks"))
        ibb = _to_int(meta.get("intentional_walks"))
        hbp = _to_int(meta.get("hbp"))
        sf = _to_int(meta.get("sacrifice_flies"))
        ab = _to_int(meta.get("at_bats"))
        rbi = _to_int(meta.get("rbi"))
        steals = _to_int(meta.get("stolen_bases"))

        if ops_val is None:
            ops_val = kbo_metrics.ops(
                hits,
                walks,
                hbp,
                ab,
                sf,
                doubles,
                triples,
                home_runs,
            )

        league_ops = (
            (_LEAGUE_CONTEXT.lg_OBP + _LEAGUE_CONTEXT.lg_SLG)
            if _LEAGUE_CONTEXT.lg_OBP and _LEAGUE_CONTEXT.lg_SLG
            else None
        )
        if ops_plus is None and ops_val and league_ops:
            ops_plus = (ops_val / league_ops) * 100

        woba_val = kbo_metrics.woba(
            walks,
            ibb,
            hbp,
            hits,
            doubles,
            triples,
            home_runs,
            ab,
            sf,
            _LEAGUE_CONTEXT,
        )
        if wrc_plus is None and woba_val is not None and pa > 0:
            wrc_plus = kbo_metrics.wrc_plus(woba_val, pa, _LEAGUE_CONTEXT)

        if war is None and woba_val is not None:
            war = kbo_metrics.war_batter(
                woba_val,
                pa,
                baserunning_runs=0.0,
                fielding_runs=0.0,
                positional_runs=0.0,
                league_adj_runs=0.0,
                ctx=_LEAGUE_CONTEXT,
            )

        score = batter_rank_score(
            wrc_plus if wrc_plus is not None else 90,
            war if war is not None else 0,
        )

        return {
            "name": meta.get("player_name", "N/A"),
            "team": _get_team_name(meta.get("team_name", "N/A")),
            "pa": pa,
            "wrc_plus": wrc_plus,
            "ops_plus": ops_plus,
            "war": war,
            "ops": ops_val,
            "obp": obp,
            "slg": slg,
            "avg": avg,
            "home_runs": home_runs,
            "rbi": rbi,
            "steals": steals,
            "score": score,
        }, None

    return None, None


def _history_for_messages(
    history: Optional[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    if not history:
        return []
    trimmed = history[-HISTORY_CONTEXT_LIMIT:]
    messages: List[Dict[str, str]] = []
    for item in trimmed:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not content:
            continue
        text = content.strip()
        if not text:
            continue
        messages.append({"role": role, "content": text})
    return messages


def _history_context_block(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""
    lines: List[str] = []
    for item in history[-HISTORY_CONTEXT_LIMIT:]:
        role = "사용자" if item.get("role") == "user" else "BEGA"
        content = (item.get("content") or "").strip()
        if not content:
            continue
        snippet = content if len(content) < 400 else content[:400] + "…"
        lines.append(f"- {role}: {snippet}")
    if not lines:
        return ""
    return "### 이전 대화 맥락\n" + "\n".join(lines)


def _build_db_unavailable_context(query: str, entity_filter: Any, year: int) -> str:
    """DB 연결 불가 시 LLM 일반지식 모드에 전달할 컨텍스트를 생성합니다."""
    parts = [
        "### [시스템 안내]",
        "현재 KBO 통계 데이터베이스에 일시적으로 접속하지 못하고 있습니다.",
        "아래 질문에 대해 데이터베이스 조회 없이 LLM의 일반 야구 지식을 바탕으로 답변해주세요.",
        "",
        "**반드시 지켜야 할 사항:**",
        "- 답변 시작 부분에 다음 문구를 포함하세요:",
        "  '⚠️ 현재 KBO 통계 DB에 일시적으로 접속할 수 없어 정확한 수치를 확인하지 못했습니다. "
        "아래 내용은 일반 야구 지식 기반의 참고 답변입니다.'",
        "- 구체적인 수치(타율, ERA, 홈런수 등)를 단정적으로 제시하지 마세요.",
        "- 불확실한 정보는 '대략', '통상적으로' 등의 표현을 사용하세요.",
        "",
        "### 사용자 질문 분석",
    ]
    if entity_filter.season_year:
        parts.append(f"- 요청 연도: {entity_filter.season_year}년")
    if entity_filter.player_name:
        parts.append(f"- 선수명: {entity_filter.player_name}")
    if entity_filter.team_id:
        parts.append(f"- 팀: {entity_filter.team_id}")
    if entity_filter.stat_type:
        parts.append(f"- 통계 항목: {entity_filter.stat_type}")
    return "\n".join(parts)


def _build_embedding_failed_context(query: str, entity_filter: Any, year: int) -> str:
    """검색 임베딩 생성 실패 시 LLM에 전달할 제한 컨텍스트를 생성합니다."""
    parts = [
        "### [시스템 안내]",
        "검색용 임베딩 생성에 실패해 저장된 KBO 근거 문서를 확인하지 못했습니다.",
        "아래 질문에 대해 구체 수치를 단정하지 말고, 검색 근거 부재를 명확히 밝힌 뒤 제한적으로 답변하세요.",
        "",
        "**반드시 지켜야 할 사항:**",
        f"- 답변 시작 부분에 다음 문구를 포함하세요: '{EMBEDDING_FAILED_PREFIX}'",
        "- 저장된 RAG 근거를 확인했다고 표현하지 마세요.",
        "- 선수 기록, 순위, 규정 세부 조건처럼 변동 가능한 정보는 단정하지 마세요.",
        "",
        "### 사용자 질문 분석",
        f"- 질문: {query}",
        f"- 기준 연도: {year}년",
    ]
    if entity_filter.player_name:
        parts.append(f"- 선수명: {entity_filter.player_name}")
    if entity_filter.team_id:
        parts.append(f"- 팀: {entity_filter.team_id}")
    if entity_filter.stat_type:
        parts.append(f"- 통계 항목: {entity_filter.stat_type}")
    return "\n".join(parts)


def _observe_rag_total(coro_func):
    """RAG 코루틴 전체 소요시간을 ``total`` 스테이지로 관측하는 데코레이터."""

    @wraps(coro_func)
    async def _wrapper(*args, **kwargs):
        _start = _rag_perf_counter()
        try:
            return await coro_func(*args, **kwargs)
        finally:
            try:
                AI_RAG_STAGE_DURATION_SECONDS.labels(stage="total").observe(
                    _rag_perf_counter() - _start
                )
            except Exception:  # noqa: BLE001
                pass

    return _wrapper


def _observe_rag_total_stream(gen_func):
    """RAG 비동기 제너레이터 전체 소요시간을 ``total`` 스테이지로 관측하는 데코레이터."""

    @wraps(gen_func)
    async def _wrapper(*args, **kwargs):
        _start = _rag_perf_counter()
        try:
            async for item in gen_func(*args, **kwargs):
                yield item
        finally:
            try:
                AI_RAG_STAGE_DURATION_SECONDS.labels(stage="total").observe(
                    _rag_perf_counter() - _start
                )
            except Exception:  # noqa: BLE001
                pass

    return _wrapper


class RAGPipeline:
    """
    검색(Retrieval)과 생성(Generation)을 결합하여 답변을 생성하는 RAG 파이프라인을 관리합니다.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        connection: Optional[psycopg.AsyncConnection] = None,
        pool: Optional[AsyncConnectionPool] = None,
        agent_runtime: BaseballAgentRuntime | None = None,
        context_formatter: Optional[ContextFormatter] = None,
        wpa_calculator: Optional["WPACalculator"] = None,
    ) -> None:
        if connection is None and pool is None:
            raise ValueError("RAGPipeline requires either 'connection' or 'pool'")
        self.settings = settings
        self.connection = connection
        self._pool = pool
        self.query_transformer = QueryTransformer(self._generate)
        self.context_formatter = context_formatter or ContextFormatter()
        self.agent_runtime = agent_runtime or initialize_shared_baseball_agent_runtime(
            settings
        )
        self.baseball_agent = self.agent_runtime.shared_agent
        self._agent_fast_path_enabled = agent_runtime is not None or pool is not None
        self.wpa_calculator = wpa_calculator or WPACalculator()

    @asynccontextmanager
    async def _checkout_conn(self) -> AsyncIterator[psycopg.AsyncConnection]:
        """풀이 있으면 매 호출마다 짧게 커넥션을 빌리고, 없으면 기존 단일 커넥션 사용."""
        if self._pool is not None:
            async with self._pool.connection() as conn:
                yield conn
        else:
            yield self.connection

    async def _build_operator_or_static_kbo_result(
        self, query: str
    ) -> Optional[Dict[str, Any]]:
        if bool(getattr(self.settings, "operator_data_fast_path_enabled", False)):
            try:
                async with self._checkout_conn() as conn:
                    result = await try_build_operator_fast_path_result(conn, query)
                if result is not None:
                    logger.info("[RAG] Operator data fast-path: %s", query)
                    return result
            except Exception as exc:  # noqa: BLE001
                logger.warning("[RAG] Operator data fast-path skipped: %s", exc)
        return _build_static_kbo_faq_result(query)

    async def _process_and_enrich_docs(
        self, docs: List[Dict[str, Any]], year: int
    ) -> Dict[str, Any]:
        """
        검색된 문서를 필터링, 분류, 계산, 랭킹 매겨 LLM에 전달할 최종 컨텍스트를 생성합니다.
        """
        logger.info(f"[RAG] Processing {len(docs)} retrieved documents for year {year}")
        processed_pitchers = []
        processed_batters = []
        processed_games = []
        processed_awards = []
        processed_movements = []
        raw_docs = []
        warnings = set()
        filtered_playoff_count = 0

        for doc in docs:
            meta = doc.get("meta", {})
            if not meta:
                continue
            source_table = doc.get("source_table")

            # Filter out non-regular season data (playoffs, etc.)
            # league = meta.get("league", "")
            # if league and league != "정규시즌":
            #     filtered_playoff_count += 1
            #     continue

            # --- Pitcher / Batter Processing (cached) ---
            if source_table in {
                "player_season_pitching",
                "player_season_batting",
            }:
                league = meta.get("league", "N/A")
                if source_table == "player_season_pitching":
                    logger.info(
                        f"[RAG] Found pitcher: {meta.get('player_name')} - IP: {meta.get('innings_pitched')}, League: {league}"
                    )

                # [Optimized] Use MetaWrapper for efficient caching
                cache_meta = dict(meta)
                if doc.get("content_hash"):
                    cache_meta["content_hash"] = doc.get("content_hash")
                if doc.get("updated_at"):
                    cache_meta["updated_at"] = doc.get("updated_at")
                meta_wrapper = MetaWrapper(cache_meta)
                processed, warning = _process_stat_doc_cached(
                    source_table, meta_wrapper
                )
                if warning:
                    warnings.add(warning)
                    continue
                raw_docs.append(doc)
                if processed:
                    if source_table == "player_season_pitching":
                        processed_pitchers.append(processed)
                    else:
                        processed_batters.append(processed)
            else:
                raw_docs.append(doc)

            if source_table in [
                "game",
                "game_flow_summary",
                "game_metadata",
                "game_batting_stats",
                "game_pitching_stats",
            ]:
                processed_games.append(doc)
            elif source_table == "awards":
                processed_awards.append(doc)
            elif source_table == "player_movements":
                processed_movements.append(doc)

        # Sort by rank score (lower is better)
        processed_pitchers.sort(key=lambda p: p["score"])
        processed_batters.sort(key=lambda b: b["score"], reverse=True)

        # Check for ambiguous players (same name in batters and pitchers)
        batter_names = {p["name"] for p in processed_batters}
        pitcher_names = {p["name"] for p in processed_pitchers}
        ambiguous_names = batter_names.intersection(pitcher_names)

        if ambiguous_names:
            for name in ambiguous_names:
                warnings.add(
                    f"주의: '{name}'은(는) 투수와 타자 모두 존재합니다. "
                    "답변 시 이 모호성을 반드시 언급하고, 가능하다면 포지션(투수/타자)을 명시하여 혼동을 피하세요."
                )

        logger.info(f"[RAG] Filtered {filtered_playoff_count} playoff records")
        logger.info(
            f"[RAG] Final processed: {len(processed_pitchers)} pitchers, {len(processed_batters)} batters"
        )

        return {
            "pitchers": processed_pitchers,
            "batters": processed_batters,
            "games": processed_games,
            "awards": processed_awards,
            "movements": processed_movements,
            "raw_docs": raw_docs,
            "warnings": list(warnings),
            "context": _LEAGUE_CONTEXT,
        }

    async def retrieve(
        self,
        query: str,
        *,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        entity_filter: Optional[Any] = None,
        use_hyde: bool = False,
        intent: str = "",
        retrieval_state: Optional[RetrievalState] = None,
    ) -> List[Dict[str, Any]]:
        search_query = query
        if use_hyde:
            hyde_prompt = HYDE_PROMPT.format(question=query)
            hyde_messages = [{"role": "user", "content": hyde_prompt}]

            try:
                hypothetical_document = await self._generate(hyde_messages)
                search_query = hypothetical_document
            except Exception:
                search_query = query

        limit = limit or self.settings.default_search_limit
        _embed_start = _rag_perf_counter()
        try:
            embedding = await async_embed_query(search_query, self.settings)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[RAG] Query embedding failed; returning zero retrieval results"
            )
            _record_retrieval_state_error(
                retrieval_state,
                error_type="embedding_failed",
                message=str(exc),
            )
            return []
        finally:
            try:
                AI_RAG_STAGE_DURATION_SECONDS.labels(stage="embed").observe(
                    _rag_perf_counter() - _embed_start
                )
            except Exception:  # noqa: BLE001
                pass
        if not embedding:
            _record_retrieval_state_error(
                retrieval_state,
                error_type="embedding_failed",
                message="empty embedding",
            )
            return []

        keyword = _build_rrf_keyword(query, entity_filter)
        logger.debug(
            "[RAG] RRF keyword built='%s' from query='%s'",
            _preview_text(keyword),
            _preview_text(query),
        )
        _search_start = _rag_perf_counter()
        try:
            # 풀 모드: 매 호출마다 풀에서 짧게 커넥션을 빌림 (멀티 variation 병렬 가능)
            # 단일 커넥션 모드: 기존 방식 유지 (테스트/스크립트용)
            async with self._checkout_conn() as conn:
                docs = await similarity_search(
                    conn,
                    embedding,
                    limit=limit,
                    filters=filters,
                    keyword=keyword,
                    settings=self.settings,
                )
        except DBRetrievalError as exc:
            logger.error("[RAG] DB retrieval error in retrieve(): %s", exc)
            _record_retrieval_state_error(
                retrieval_state,
                error_type="db_unavailable",
                message=str(exc.cause),
            )
            return []
        finally:
            try:
                AI_RAG_STAGE_DURATION_SECONDS.labels(stage="search").observe(
                    _rag_perf_counter() - _search_start
                )
            except Exception:  # noqa: BLE001
                pass
        return docs

    async def retrieve_with_multi_query(
        self,
        query: str,
        entity_filter,
        *,
        filters: Optional[Dict[str, Any]] = None,
        use_llm_expansion: bool = False,
        limit: Optional[int] = None,
        intent: str = "",
        retrieval_state: Optional[RetrievalState] = None,
    ) -> List[Dict[str, Any]]:
        """
        Multi-query retrieval을 사용하여 검색 품질을 향상시킵니다.
        여러 쿼리 변형으로 검색하고 결과를 결합합니다.
        """
        logger.info(f"[RAG] Multi-query retrieval for: {query}")

        # 규칙 기반 쿼리 확장
        rule_variation_cap = max(
            1,
            int(
                getattr(
                    self.settings,
                    "retrieval_multi_query_rule_variation_max",
                    5,
                )
            ),
        )
        query_variations = self.query_transformer.expand_query_with_rules(
            query,
            entity_filter,
            max_variations=rule_variation_cap,
        )

        # LLM 기반 쿼리 확장 (선택적)
        if use_llm_expansion and len(query_variations) < 3:
            try:
                llm_variations = await self.query_transformer.llm_expand_query(query)
                query_variations.extend(llm_variations)
            except Exception as e:
                logger.warning(f"[RAG] LLM query expansion failed: {e}")

        # Multi-query retrieval 수행
        effective_limit = max(1, int(limit or self.settings.default_search_limit))
        effective_limit_per_query = max(
            int(self.settings.retrieval_multi_query_limit_per_query),
            effective_limit,
        )

        docs = await multi_query_retrieval(
            query_variations,
            self.retrieve,
            filters or {},
            entity_filter=entity_filter,
            limit_per_query=effective_limit_per_query,
            limit=effective_limit,
            intent=intent,
            retrieval_state=retrieval_state,
            settings=self.settings,
        )

        logger.info(f"[RAG] Multi-query retrieval returned {len(docs)} documents")
        return docs

    def _should_use_single_query_retrieval(
        self,
        *,
        query: str = "",
        search_strategy: Dict[str, Any],
        entity_filter: Any,
        final_filters: Dict[str, Any],
    ) -> bool:
        if not bool(
            getattr(self.settings, "retrieval_single_query_for_strict_entity", True)
        ):
            return False
        if bool(search_strategy.get("is_ranking_query")):
            return False
        if getattr(entity_filter, "player_name", None):
            query_lower = query.lower()
            return any(
                keyword in query_lower for keyword in _FORCE_PLAYER_FAST_PATH_KEYWORDS
            )
        if final_filters.get("source_table"):
            return True
        if getattr(entity_filter, "team_id", None):
            return True
        return False

    def _rerank_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """v1 rerank interface; disabled by default and uses existing scores only."""
        if not bool(getattr(self.settings, "rag_rerank_enabled", False)):
            return docs
        candidate_limit = max(
            1, int(getattr(self.settings, "rag_rerank_candidate_limit", 20) or 20)
        )
        context_limit = max(
            1, int(getattr(self.settings, "rag_context_limit", 10) or 10)
        )
        candidates = docs[:candidate_limit]
        candidates.sort(
            key=lambda doc: (
                float(doc.get("combined_score") or 0.0),
                float(doc.get("similarity") or 0.0),
                float(doc.get("quality_score") or 0.0),
            ),
            reverse=True,
        )
        return candidates[:context_limit]

    async def _record_retrieval_event(
        self,
        *,
        query: str,
        intent: Optional[str],
        final_filters: Dict[str, Any],
        docs: List[Dict[str, Any]],
        selected_docs: Optional[List[Dict[str, Any]]] = None,
        retrieval_started_at: float,
        success: bool,
        error_type: Optional[str],
        original_filters: Optional[Dict[str, Any]] = None,
        actual_filters: Optional[Dict[str, Any]] = None,
        fallback_used: bool = False,
        fallback_stage: Optional[str] = "none",
    ) -> None:
        selected = selected_docs if selected_docs is not None else docs
        rewritten_queries = [
            str(doc.get("_source_query")) for doc in docs if doc.get("_source_query")
        ]
        seen_queries: set[str] = set()
        unique_rewritten_queries: List[str] = []
        for rewritten_query in rewritten_queries:
            if rewritten_query in seen_queries:
                continue
            seen_queries.add(rewritten_query)
            unique_rewritten_queries.append(rewritten_query)

        def _doc_id(doc: Dict[str, Any]) -> Any:
            return doc.get("id")

        scores = [
            {
                "id": _doc_id(doc),
                "similarity": doc.get("similarity"),
                "combined_score": doc.get("combined_score"),
                "keyword_rank_val": doc.get("keyword_rank_val"),
                "quality_score": doc.get("quality_score"),
            }
            for doc in docs
        ]
        if original_filters is None and actual_filters is None and not fallback_used:
            metadata_filter = final_filters
        else:
            metadata_filter = _build_retrieval_event_filter(
                original_filters or final_filters,
                actual_filters=(
                    actual_filters if actual_filters is not None else final_filters
                ),
                fallback_used=fallback_used,
                fallback_stage=fallback_stage,
            )
        latency_ms = int((_rag_perf_counter() - retrieval_started_at) * 1000)
        try:
            async with self._checkout_conn() as conn:
                await record_retrieval_event(
                    conn,
                    user_query=query,
                    intent=intent,
                    rewritten_queries=unique_rewritten_queries,
                    metadata_filter=metadata_filter,
                    retrieved_chunk_ids=[_doc_id(doc) for doc in docs],
                    selected_chunk_ids=[_doc_id(doc) for doc in selected],
                    scores=scores,
                    latency_ms=latency_ms,
                    success=success,
                    error_type=error_type,
                    settings=self.settings,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RAG] Failed to record retrieval event: %s", exc)

    async def _generate(self, messages: Sequence[Dict[str, str]]) -> str:
        provider = self.settings.llm_provider
        _llm_start = _rag_perf_counter()

        try:
            if provider == "gemini":
                result = await self._generate_with_gemini(messages)
            elif provider == "openrouter":
                result = await self._generate_with_openrouter(messages)
            else:
                raise RuntimeError(f"지원되지 않는 LLM 공급자: {provider}")
            try:
                AI_LLM_CALL_DURATION_SECONDS.labels(
                    provider=provider, route="rag"
                ).observe(_rag_perf_counter() - _llm_start)
            except Exception:  # noqa: BLE001
                pass
            return result
        except Exception as e:
            logger.error(f"[RAG] Primary LLM provider '{provider}' failed: {e}")

            # Try fallback provider
            fallback_provider = "gemini" if provider == "openrouter" else "openrouter"

            # Check if fallback is available
            if fallback_provider == "gemini" and self.settings.gemini_api_key:
                logger.info(f"[RAG] Attempting fallback to Gemini")
                try:
                    fb_result = await self._generate_with_gemini(messages)
                    try:
                        AI_LLM_CALL_DURATION_SECONDS.labels(
                            provider="gemini", route="rag_fallback"
                        ).observe(_rag_perf_counter() - _llm_start)
                    except Exception:  # noqa: BLE001
                        pass
                    return fb_result
                except Exception as fallback_e:
                    logger.error(f"[RAG] Fallback to Gemini also failed: {fallback_e}")
            elif fallback_provider == "openrouter" and self.settings.openrouter_api_key:
                logger.info(f"[RAG] Attempting fallback to OpenRouter")
                try:
                    fb_result = await self._generate_with_openrouter(messages)
                    try:
                        AI_LLM_CALL_DURATION_SECONDS.labels(
                            provider="openrouter", route="rag_fallback"
                        ).observe(_rag_perf_counter() - _llm_start)
                    except Exception:  # noqa: BLE001
                        pass
                    return fb_result
                except Exception as fallback_e:
                    logger.error(
                        f"[RAG] Fallback to OpenRouter also failed: {fallback_e}"
                    )

            # All providers failed
            raise RuntimeError(
                f"모든 LLM 제공자가 실패했습니다. 주 제공자({provider}): {e}"
            )

    async def _generate_stream(
        self, messages: Sequence[Dict[str, str]]
    ) -> Iterator[str]:
        """스트리밍 모드로 답변을 생성합니다."""
        provider = self.settings.llm_provider

        try:
            if provider == "gemini":
                async for chunk in self._generate_stream_with_gemini(messages):
                    yield chunk
            elif provider == "openrouter":
                async for chunk in self._generate_stream_with_openrouter(messages):
                    yield chunk
            else:
                # 스트리밍 미지원 시 일반 생성 결과 반환
                yield await self._generate(messages)
        except Exception as e:
            logger.error(
                f"[RAG] Stream generation failed ({type(e).__name__}): {e}. Falling back to completion."
            )
            try:
                yield await self._generate(messages)
            except Exception as fe:
                logger.error(f"[RAG] Completion fallback also failed: {fe}")
                yield "죄송합니다. 답변 생성 중 오류가 발생했습니다."

    async def _generate_stream_with_gemini(
        self, messages: Sequence[Dict[str, str]]
    ) -> Iterator[str]:
        """Gemini API를 사용하여 스트리밍 답변을 생성합니다."""
        if not self.settings.gemini_api_key:
            raise RuntimeError("Gemini API 키가 없습니다.")

        gemini_contents = []
        system_instructions = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instructions += content + "\n\n"
            elif role == "user":
                user_content = content
                if system_instructions and not gemini_contents:
                    user_content = f"System Instruction:\n{system_instructions}\n\nUser Question:\n{content}"
                gemini_contents.append(
                    {"role": "user", "parts": [{"text": user_content}]}
                )
            elif role == "assistant":
                gemini_contents.append({"role": "model", "parts": [{"text": content}]})

        model = self.settings.gemini_model or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:streamGenerateContent"
        params = {"key": self.settings.gemini_api_key, "alt": "sse"}

        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": self.settings.max_output_tokens,
            },
        }

        client = get_shared_httpx_client(
            "gemini",
            timeout=httpx.Timeout(60.0, connect=10.0, read=60.0, pool=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        async with client.stream(
            "POST", url, json=payload, params=params, timeout=60.0
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        chunk = (
                            data.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        if chunk:
                            yield chunk
                    except Exception:
                        continue

    async def _generate_stream_with_openrouter(
        self, messages: Sequence[Dict[str, str]]
    ) -> Iterator[str]:
        """OpenRouter API를 사용하여 스트리밍 답변을 생성합니다."""
        if not self.settings.openrouter_api_key:
            raise RuntimeError("OpenRouter API 키가 없습니다.")

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.openrouter_referer or "",
            "X-Title": self.settings.openrouter_app_title or "",
        }
        payload = {
            "model": self.settings.openrouter_model,
            "messages": list(messages),
            "stream": True,
            "max_tokens": self.settings.max_output_tokens,
        }

        url = f"{self.settings.openrouter_base_url.rstrip('/')}/chat/completions"
        client = get_shared_httpx_client(
            "openrouter",
            timeout=httpx.Timeout(60.0, connect=10.0, read=60.0, pool=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=60.0
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                logger.error(
                    f"[RAG] OpenRouter stream error {response.status_code}: {body[:300]!r}"
                )
                response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    if line == "data: [DONE]":
                        break
                    try:
                        data = json.loads(line[6:])
                        chunk = (
                            data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if chunk:
                            yield chunk
                    except Exception:
                        continue

    @llm_retry
    async def _generate_with_openrouter(
        self, messages: Sequence[Dict[str, str]]
    ) -> str:
        if not self.settings.openrouter_api_key:
            raise RuntimeError(
                "OpenRouter를 사용하려면 OPENROUTER_API_KEY가 필요합니다."
            )

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.openrouter_referer or "",
            "X-Title": self.settings.openrouter_app_title or "",
        }
        payload = {
            "model": self.settings.openrouter_model,
            "messages": list(messages),
            "max_tokens": self.settings.max_output_tokens,
            "temperature": 0.1,
        }

        client = get_shared_httpx_client(
            "openrouter",
            timeout=httpx.Timeout(120.0, connect=10.0, read=60.0, pool=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        response = await client.post(
            f"{self.settings.openrouter_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
        )

        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            error_msg = (
                f"OpenRouter 응답에 choices가 없습니다. Keys: {list(data.keys())}"
            )
            logger.error(f"[OpenRouter] {error_msg}")
            raise RuntimeError(error_msg)

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if not content:
            error_msg = (
                f"OpenRouter 응답이 비어 있습니다. Message keys: {list(message.keys())}"
            )
            logger.error(f"[OpenRouter] {error_msg}")
            raise RuntimeError(error_msg)

        return content

    @llm_retry
    async def _generate_with_gemini(self, messages: Sequence[Dict[str, str]]) -> str:
        """Google Gemini API를 사용하여 응답을 생성합니다."""
        if not self.settings.gemini_api_key:
            raise RuntimeError("Gemini를 사용하려면 GEMINI_API_KEY가 필요합니다.")

        # Convert OpenAI format messages to Gemini format
        gemini_contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
                if not gemini_contents:
                    gemini_contents.append(
                        {
                            "role": "user",
                            "parts": [{"text": f"System: {content}\n\nUser: "}],
                        }
                    )
                else:
                    # Prepend to existing user message
                    if gemini_contents[-1]["role"] == "user":
                        gemini_contents[-1]["parts"][0]["text"] = (
                            f"System: {content}\n\n"
                            + gemini_contents[-1]["parts"][0]["text"]
                        )
            elif role == "user":
                gemini_contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                gemini_contents.append({"role": "model", "parts": [{"text": content}]})

        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "maxOutputTokens": self.settings.max_output_tokens,
                "temperature": 0.1,
            },
        }

        model = self.settings.gemini_model or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
        params = {"key": self.settings.gemini_api_key}

        client = get_shared_httpx_client(
            "gemini",
            timeout=httpx.Timeout(60.0, connect=10.0, read=60.0, pool=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        response = await client.post(url, json=payload, params=params)

        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            error_msg = (
                f"Gemini 응답에 candidates가 없습니다. Keys: {list(data.keys())}"
            )
            logger.error(f"[Gemini] {error_msg}")
            raise RuntimeError(error_msg)

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        if not parts or not parts[0].get("text"):
            error_msg = f"Gemini 응답이 비어 있습니다. Content: {content}"
            logger.warning(f"[Gemini] {error_msg}")
            return "Gemini가 응답을 생성하지 못했습니다. (빈 응답)"

        return parts[0]["text"]

    def _is_statistical_query(self, query: str, entity_filter) -> bool:
        """
        질문이 구체적인 통계 조회인지 판단합니다.
        통계 질문인 경우 야구 에이전트를 우선 사용해야 합니다.
        """
        query_lower = query.lower()

        # 1단계: 일반 대화인지 확인 (우선순위 높음)
        is_chitchat = any(keyword in query_lower for keyword in _STAT_CHITCHAT_KEYWORDS)
        if is_chitchat and not any(
            keyword in query_lower for keyword in _STATISTICAL_KEYWORDS
        ):
            return False  # 일반 대화이므로 통계 질문 아님

        # 2단계: 통계 키워드 확인
        has_stat_keywords = any(
            keyword in query_lower for keyword in _STATISTICAL_KEYWORDS
        )

        # 3단계: 구체적인 데이터 요청인지 확인
        has_specific_request = (
            entity_filter.player_name
            or entity_filter.stat_type
            or "년" in query
            or any(word in query_lower for word in _STAT_SPECIFIC_REQUEST_WORDS)
            or
            # 질문 형태나 통계 지표가 있으면 통계 질문으로 간주
            "?" in query
            or "몇" in query
            or "어떻게" in query
            or any(stat in query_lower for stat in _STAT_CORE_INDICATORS)
        )

        # 통계 키워드가 있고 구체적인 요청이면 통계 질문
        result = has_stat_keywords and has_specific_request

        # 디버깅용 로그는 DEBUG 레벨로 강등 (매 요청마다 평가되는 hot path).
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[RAG] _is_statistical_query debug: query=%r, "
                "has_stat_keywords=%s, has_specific_request=%s, "
                "player_name=%r, stat_type=%r, result=%s",
                query,
                has_stat_keywords,
                has_specific_request,
                entity_filter.player_name,
                entity_filter.stat_type,
                result,
            )
        return result

    def _is_regulation_query(self, query: str) -> bool:
        """
        질문이 KBO 규정 관련인지 판단합니다.
        """
        return _is_regulation_query_text(query)

    def _is_game_query(self, query: str) -> bool:
        """
        질문이 경기 데이터 관련인지 판단합니다.
        """
        query_lower = query.lower()

        # 키워드 매칭
        has_game_keywords = any(keyword in query_lower for keyword in _GAME_KEYWORDS)

        # 날짜 패턴 매칭 (미리 컴파일된 패턴 사용)
        has_date_pattern = any(p.search(query_lower) for p in _GAME_DATE_PATTERNS)

        # 팀 vs 팀 패턴 (미리 컴파일된 case-insensitive 패턴)
        has_team_vs_pattern = bool(_GAME_TEAM_VS_PATTERN.search(query))

        return has_game_keywords or has_date_pattern or has_team_vs_pattern

    def _is_game_flow_narrative_query(self, query: str) -> bool:
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in _GAME_FLOW_NARRATIVE_KEYWORDS)

    def _should_force_agent_fast_path(self, query: str, entity_filter) -> bool:
        """Questions with dedicated DB tools should not pay the RAG embedding cost first."""
        if getattr(entity_filter, "stat_leader", None):
            return True
        query_lower = query.lower()
        if (
            getattr(entity_filter, "player_name", None)
            and not getattr(entity_filter, "team_id", None)
            and any(
                keyword in query_lower for keyword in _FORCE_PLAYER_FAST_PATH_KEYWORDS
            )
        ):
            return True
        if getattr(entity_filter, "team_id", None) and any(
            keyword in query_lower
            for keyword in _FORCE_TEAM_ANALYSIS_FAST_PATH_KEYWORDS
        ):
            return True
        return any(
            keyword in query_lower for keyword in _FORCE_AGENT_FAST_PATH_KEYWORDS
        )

    def _is_general_conversation(self, query: str) -> bool:
        """
        일반 대화인지 판단합니다.
        """
        query_lower = query.lower()

        # 야구 관련 질문이면 일반 대화가 아님
        if any(keyword in query_lower for keyword in _GENERAL_BASEBALL_KEYWORDS):
            return False  # 야구 관련 질문이므로 일반 대화가 아님

        # 야구/통계와 무관한 일반적인 대화만 일반 대화로 분류
        return any(keyword in query_lower for keyword in _GENERAL_CONVERSATION_KEYWORDS)

    async def _try_agent_first(
        self,
        query: str,
        *,
        intent: str = "freeform",
        filters: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        야구 에이전트를 우선 사용하여 통계 질문을 처리합니다.
        에이전트가 실패하면 기존 RAG 방식으로 폴백합니다.
        """
        logger.info(f"[RAG] Attempting agent-first approach for: {query}")

        try:
            # 야구 에이전트를 통한 처리 시도
            async with self._checkout_conn() as conn:
                with self.agent_runtime.request_context(conn):
                    agent_result = await self.baseball_agent.process_query(
                        query,
                        {
                            "intent": intent,
                            "filters": filters,
                            "history": history,
                            "request_mode": "completion",
                            "persona": "chat",
                        },
                    )

            if agent_result["verified"] and not agent_result.get("error"):
                logger.info(
                    f"[RAG] Agent successfully handled query with verified data"
                )
                perf = agent_result.get("perf") or {}
                if not isinstance(perf, dict):
                    perf = {}
                planner_mode = agent_result.get("planner_mode") or "verified_agent"
                planner_cache_hit = bool(agent_result.get("planner_cache_hit", False))
                tool_execution_mode = (
                    agent_result.get("tool_execution_mode")
                    or perf.get("tool_execution_mode")
                    or "unknown"
                )
                if isinstance(perf, dict):
                    perf = {
                        **perf,
                        "planner_mode": perf.get("planner_mode") or planner_mode,
                        "planner_cache_hit": bool(
                            perf.get("planner_cache_hit", planner_cache_hit)
                        ),
                        "tool_execution_mode": tool_execution_mode,
                    }
                return {
                    "answer": agent_result["answer"],
                    "citations": [],  # 에이전트는 DB 직접 조회하므로 citations 불필요
                    "intent": intent,
                    "retrieved": agent_result.get("tool_results", []),
                    "strategy": "verified_agent",
                    "verified": True,
                    "tool_calls": agent_result.get("tool_calls", []),
                    "tool_results": agent_result.get("tool_results", []),
                    "data_sources": agent_result.get("data_sources", []),
                    "visualizations": agent_result.get("visualizations", []),
                    "planner_mode": planner_mode,
                    "planner_cache_hit": planner_cache_hit,
                    "tool_execution_mode": tool_execution_mode,
                    "fallback_triggered": bool(
                        agent_result.get("fallback_triggered", False)
                    ),
                    "fallback_answer_used": bool(
                        agent_result.get("fallback_answer_used", False)
                    ),
                    "grounding_mode": agent_result.get("grounding_mode"),
                    "source_tier": agent_result.get("source_tier"),
                    "answer_sources": agent_result.get("answer_sources", []),
                    "as_of_date": agent_result.get("as_of_date"),
                    "fallback_reason": agent_result.get("fallback_reason"),
                    "perf": perf,
                }
            else:
                logger.warning(
                    f"[RAG] Agent failed or returned unverified data: {agent_result.get('error')}"
                )
                return None  # 폴백 신호

        except Exception as e:
            logger.error(f"[RAG] Agent processing error: {e}")
            return None  # 폴백 신호

    async def _handle_general_conversation(self, query: str) -> Dict[str, Any]:
        """
        일반 대화를 처리합니다.
        """
        logger.info(f"[RAG] Handling general conversation: {query}")

        # 야구 지식 관련 질문 처리
        knowledge_responses = {
            "ops": "OPS는 출루율(OBP)과 장타율(SLG)을 더한 값입니다.\n- 계산법: OPS = 출루율 + 장타율\n- 좋은 OPS: 0.800 이상\n- 뛰어난 OPS: 0.900 이상\n- 최고 수준 OPS: 1.000 이상\n\nOPS는 타자의 종합적인 공격력을 나타내는 대표적인 지표입니다.",
            "wrc+": "wRC+는 가중출루율을 기반으로 한 공격력 지표입니다.\n- 100이 평균 (리그 평균 대비 100%)\n- 120이면 리그 평균보다 20% 우수\n- 80이면 리그 평균보다 20% 부족\n\nwRC+는 볼파크와 리그 환경을 보정한 정확한 공격력 지표입니다.",
            "war": "WAR(Wins Above Replacement)는 대체 선수 대비 승수 기여도입니다.\n- WAR 2: 평균 주전급\n- WAR 5: 올스타급\n- WAR 8+: MVP급\n\nWAR은 선수의 종합적인 가치를 하나의 숫자로 나타내는 가장 포괄적인 지표입니다.",
            "era": "ERA(자책점평균)는 투수가 9이닝당 내주는 자책점 수입니다.\n- 계산법: (자책점 × 9) ÷ 투구이닝\n- 좋은 ERA: 4.00 미만\n- 뛰어난 ERA: 3.00 미만\n- 최고 수준 ERA: 2.50 미만",
            "whip": "WHIP는 투수가 이닝당 내주는 안타와 볼넷의 합계입니다.\n- 계산법: (피안타 + 볼넷) ÷ 투구이닝\n- 좋은 WHIP: 1.30 미만\n- 뛰어난 WHIP: 1.20 미만\n- 최고 수준 WHIP: 1.10 미만",
        }

        # 간단한 대화 응답 패턴
        conversation_responses = {
            "안녕": "안녕하세요! 저는 KBO 리그 데이터 분석가 BEGA입니다. KBO 야구 통계에 대해 궁금한 것이 있으시면 언제든 물어보세요!",
            "누구": "저는 KBO 리그 전문 데이터 분석가 'BEGA'입니다. 한국 프로야구의 각종 통계와 기록을 정확하게 분석해드립니다.",
            "좋아": "네, 저는 야구를 정말 좋아합니다! 특히 KBO 리그의 흥미진진한 경기와 선수들의 기록을 분석하는 것이 제 전문 분야입니다.",
            "응원": "저는 모든 KBO 팀을 공정하게 분석합니다! 어떤 팀을 응원하시든 정확하고 객관적인 데이터를 제공해드릴게요.",
            "날씨": "죄송하지만 날씨 정보는 제공하지 않습니다. 저는 KBO 야구 통계 전문 분석가입니다. 야구 관련 질문이 있으시면 언제든 물어보세요!",
            "도움": "저는 다음과 같은 도움을 드릴 수 있습니다:\n- 선수 개인 통계 조회\n- 팀별 순위 및 기록 분석\n- 야구 지표 설명\n- KBO 리그 역사적 기록 비교\n\n궁금한 야구 통계가 있으시면 언제든 말씀해주세요!",
            "기능": "제 주요 기능은 다음과 같습니다:\n1. 선수 개인 성적 분석 (타율, 홈런, ERA 등)\n2. 팀 순위 및 리더보드 조회\n3. 고급 야구 지표 계산 및 설명\n4. 시즌별, 연도별 기록 비교\n\n구체적인 야구 통계 질문을 해보세요!",
        }

        query_lower = query.lower()

        # 야구 지식 질문 먼저 확인 (우선순위 높음)
        for keyword, response in knowledge_responses.items():
            if keyword in query_lower:
                return {
                    "answer": response,
                    "citations": [],
                    "intent": "knowledge_explanation",
                    "retrieved": [],
                    "strategy": "knowledge_handler",
                    "verified": True,
                }

        # 일반 대화 키워드 확인
        for keyword, response in conversation_responses.items():
            if keyword in query_lower:
                return {
                    "answer": response,
                    "citations": [],
                    "intent": "general_conversation",
                    "retrieved": [],
                    "strategy": "conversation_handler",
                    "verified": True,
                }

        # 기본 응답 (아무 키워드도 매칭되지 않을 때만)
        logger.info(f"[RAG] _handle_general_conversation fallback for query: {query}")
        default_response = """안녕하세요! 저는 KBO 리그 데이터 분석가 'BEGA'입니다.

KBO 야구와 관련된 다음과 같은 질문들을 도와드릴 수 있습니다:
- "2025년 김도영 타율은?"
- "홈런왕 TOP 5는?"
- "LG 트윈스 주요 선수는?"
- "OPS가 뭐야?"

야구 통계에 대해 궁금한 것이 있으시면 언제든 물어보세요!"""

        return {
            "answer": default_response,
            "citations": [],
            "intent": "general_conversation",
            "retrieved": [],
            "strategy": "conversation_handler",
            "verified": True,
        }

    @_observe_rag_total_stream
    async def run_stream(
        self,
        query: str,
        *,
        intent: str = "freeform",
        filters: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        RAG 파이프라인을 스트리밍 모드로 실행합니다.

        Returns:
            각 단계별 진행 상황 및 최종 답변 조각을 포함하는 비동기 제너레이터
        """
        # 1. 전처리 및 검색 (run 메소드의 로직과 동일하게 진행하되 결과만 다르게 구성)
        query = full_normalize(query)
        logger.info(f"[RAG] Processing query (stream): {query}")
        retrieval_state = _new_retrieval_state()
        static_kbo_result = await self._build_operator_or_static_kbo_result(query)
        if static_kbo_result is not None:
            logger.info("[RAG] Static KBO FAQ fast-path (stream): %s", query)
            yield {
                "type": "metadata",
                "data": {
                    key: value
                    for key, value in static_kbo_result.items()
                    if key != "answer"
                },
            }
            yield {
                "type": "answer_chunk",
                "content": static_kbo_result["answer"],
            }
            return

        if intent == "freeform":
            from ..ml.intent_router import predict_intent

            intent = predict_intent(query)
            logger.info(f"[RAG] Predicted intent: {intent}")

        _entity_extract_start = _rag_perf_counter()
        search_strategy = enhance_search_strategy(query)
        try:
            AI_RAG_STAGE_DURATION_SECONDS.labels(stage="entity_extract").observe(
                _rag_perf_counter() - _entity_extract_start
            )
        except Exception:  # noqa: BLE001
            pass
        entity_filter = search_strategy["entity_filter"]
        extracted_filters = search_strategy["db_filters"]

        is_game_query = self._is_game_query(query)
        is_game_flow_narrative = self._is_game_flow_narrative_query(query)
        is_statistical = self._is_statistical_query(query, entity_filter)
        is_regulation = self._is_regulation_query(query)
        force_agent_fast_path = self._should_force_agent_fast_path(query, entity_filter)

        if is_regulation and intent == "freeform":
            intent = "explanatory"

        # 메타데이터 이벤트 우선 발송
        yield {
            "type": "metadata",
            "data": {
                "intent": intent,
                "entity_filter": {
                    "season_year": entity_filter.season_year,
                    "team_id": entity_filter.team_id,
                    "player_name": entity_filter.player_name,
                    "stat_type": entity_filter.stat_type,
                    "position_type": entity_filter.position_type,
                },
            },
        }

        # 에이전트 우선 처리 로직 (스트리밍 지원하는 경우)
        # is_statistical 제외: 통계 게임 쿼리도 BaseballStatisticsAgent 경로를 사용
        if force_agent_fast_path or (
            not is_regulation and is_game_query and not is_game_flow_narrative
        ):
            # 경기 데이터 질문은 에이전트 스트리밍 시도
            # process_query_stream is on BaseballStatisticsAgent (self.baseball_agent),
            # not on BaseballAgentRuntime — use the correct reference and establish
            # request_context so DB resources resolve inside the agent.
            if hasattr(self.baseball_agent, "process_query_stream"):
                logger.info(
                    "[RAG] Agent fast-path query detected, trying agent stream first"
                )
                try:
                    async with self._checkout_conn() as conn:
                        with self.agent_runtime.request_context(conn):
                            async for event in self.baseball_agent.process_query_stream(
                                query,
                                context={
                                    "filters": filters,
                                    "history": history,
                                    "request_mode": "stream",
                                    "persona": "chat",
                                },
                            ):
                                yield event
                    return
                except Exception as e:
                    logger.error("[RAG] Agent stream error: %s", e)

        # 기존 RAG 방식으로 진행
        final_filters = {**extracted_filters, **(filters or {})}
        if is_game_query or entity_filter.game_date:
            final_filters.pop("team_id", None)
        if is_game_flow_narrative:
            final_filters["source_table"] = "game_flow_summary"

        year = (
            final_filters.get("season_year")
            or entity_filter.season_year
            or _resolve_default_season_year(self.settings)
        )

        search_limit = self.settings.default_search_limit
        if is_regulation:
            search_limit = max(search_limit, 20)
            logger.info(
                f"[RAG] Regulation query (stream): increasing search_limit to {search_limit}"
            )

        # 문서 검색 (규정 쿼리는 규정/일반 검색을 병렬 실행 후 병합)
        if is_regulation:
            reg_filters = {
                **final_filters,
                "source_table_in": list(_REGULATION_SOURCES),
            }
            _reg_result, _general_result = await asyncio.gather(
                self.retrieve(
                    query,
                    filters=reg_filters,
                    entity_filter=entity_filter,
                    limit=20,
                    retrieval_state=retrieval_state,
                ),
                self.retrieve(
                    query,
                    filters=final_filters,
                    entity_filter=entity_filter,
                    limit=search_limit,
                    retrieval_state=retrieval_state,
                ),
                return_exceptions=True,
            )
            reg_docs = _reg_result if not isinstance(_reg_result, BaseException) else []
            general_docs = (
                _general_result
                if not isinstance(_general_result, BaseException)
                else []
            )
            if len(reg_docs) >= 8:
                docs = reg_docs
            else:
                seen = {d["id"] for d in reg_docs}
                docs = reg_docs + [d for d in general_docs if d["id"] not in seen]
            logger.info(
                f"[RAG] Regulation parallel retrieval (stream): {len(docs)} docs (reg_docs={len(reg_docs)})"
            )
        else:
            docs = await self.retrieve(
                query,
                filters=final_filters,
                entity_filter=entity_filter,
                limit=search_limit,
                retrieval_state=retrieval_state,
            )

        # 검색 결과가 없으면 폴백 시도
        if not docs and final_filters:
            fallback_filters = dict(final_filters)
            if "source_table" in fallback_filters:
                fallback_filters.pop("source_table")
                docs = await self.retrieve(
                    query,
                    filters=fallback_filters,
                    limit=search_limit,
                    retrieval_state=retrieval_state,
                )

        docs = _sort_docs_for_context(
            docs,
            is_regulation=is_regulation,
            query=query,
        )

        # 규정 쿼리: LLM에 전달 전 비규정 청크 제거로 환각 억제
        if is_regulation and docs:
            regulation_sources = _REGULATION_SOURCES
            filtered = [d for d in docs if d.get("source_table") in regulation_sources]
            if filtered:
                docs = filtered[:8]

        # --- DB 연결 장애 스트림 폴백 ---
        # run()과 동일한 Baseball Data Policy 적용: DB 장애 시 LLM 스트림을 생성하지 않는다.
        if retrieval_state.get("db_error") is not None and not docs:
            logger.warning(
                "[RAG] DB was unavailable during stream retrieval. cause=%s",
                retrieval_state.get("db_error"),
            )
            answer = (
                DB_UNAVAILABLE_PREFIX
                + "\n\n현재 KBO 통계 데이터베이스에 접속할 수 없어 답변을 제공할 수 없습니다. "
                "잠시 후 다시 시도해 주세요."
            )
            yield {"type": "answer_chunk", "content": answer}
            yield {
                "type": "metadata",
                "data": {"strategy": "db_unavailable", "citations": []},
            }
            return
        # --- DB 연결 장애 스트림 폴백 끝 ---

        _format_start = _rag_perf_counter()
        processed_data = await self._process_and_enrich_docs(docs, year)
        formatted_context = self.context_formatter.format_context(
            processed_data, intent, query, entity_filter, year
        )

        if not docs:
            formatted_context = self.context_formatter.format_zero_hit_guidance(
                query, entity_filter, year, final_filters
            )
        try:
            AI_RAG_STAGE_DURATION_SECONDS.labels(stage="format").observe(
                _rag_perf_counter() - _format_start
            )
        except Exception:  # noqa: BLE001
            pass

        history_block = _history_context_block(history)
        if history_block:
            formatted_context = history_block + "\n\n" + formatted_context

        prompt = FOLLOWUP_PROMPT.format(question=query, context=formatted_context)

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 KBO 리그 야구 전문가 'BEGA'입니다. "
                    "반드시 주어진 검색 컨텍스트만 근거로 답하십시오. "
                    "컨텍스트에 없는 내용을 추론하거나 생성하지 마십시오. "
                    "2026년 규정 변화(외국인 4명, 아시아 쿼터, 11회 연장, 수비 시프트 등)가 "
                    "컨텍스트에 있으면 그것을 최신 공식 정보로 간주하여 상세히 답하십시오. "
                    "정보가 없으면 '컨텍스트에서 확인되지 않습니다'라고만 답하십시오."
                ),
            }
        ]
        messages.extend(_history_for_messages(history))
        messages.append({"role": "user", "content": prompt})

        # 스트리밍 답변 생성
        async for chunk in self._generate_stream(messages):
            yield {
                "type": "answer_chunk",
                "content": chunk,
            }

    @_observe_rag_total
    async def run(
        self,
        query: str,
        *,
        intent: str = "freeform",
        filters: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        # 1. Enhanced Entity Extraction and Search Strategy
        query = full_normalize(query)  # 특수문자 제거, 한영 정규화, 공백 정리
        logger.info(f"[RAG] Processing query: {query}")
        retrieval_state = _new_retrieval_state()
        static_kbo_result = await self._build_operator_or_static_kbo_result(query)
        if static_kbo_result is not None:
            logger.info("[RAG] Static KBO FAQ fast-path: %s", query)
            return static_kbo_result

        # 의도가 지정되지 않았거나 freeform이면 예측 시도
        if intent == "freeform":
            from ..ml.intent_router import predict_intent

            intent = predict_intent(query)
            logger.info(f"[RAG] Predicted intent: {intent}")

        # Extract entities and enhance search strategy
        _entity_extract_start = _rag_perf_counter()
        search_strategy = enhance_search_strategy(query)
        try:
            AI_RAG_STAGE_DURATION_SECONDS.labels(stage="entity_extract").observe(
                _rag_perf_counter() - _entity_extract_start
            )
        except Exception:  # noqa: BLE001
            pass
        entity_filter = search_strategy["entity_filter"]
        extracted_filters = search_strategy["db_filters"]
        raw_search_limit = search_strategy.get("search_limit")
        if raw_search_limit in (None, ""):
            search_limit = self.settings.default_search_limit
        else:
            search_limit = max(1, int(raw_search_limit))

        # 2. 통계/규정/게임 질문 여부 판별 (검색 전략에 영향)
        is_statistical = self._is_statistical_query(query, entity_filter)
        is_regulation = self._is_regulation_query(query)
        is_game_query = self._is_game_query(query)
        is_game_flow_narrative = self._is_game_flow_narrative_query(query)
        force_agent_fast_path = self._should_force_agent_fast_path(query, entity_filter)
        agent_fast_path_enabled = bool(getattr(self, "_agent_fast_path_enabled", False))
        if is_game_flow_narrative:
            force_agent_fast_path = False

        # 규정 질문이면 검색 제한을 늘려 관련 상세 조항들이 더 많이 포함되도록 함
        if is_regulation:
            search_limit = max(search_limit, 20)
            logger.info(
                f"[RAG] Regulation query: increasing search_limit to {search_limit}"
            )

        # 규정 질문이면 의도를 explanatory로 강화
        if is_regulation and intent == "freeform":
            intent = "explanatory"

        logger.info(
            f"[RAG] Path decision: is_statistical={is_statistical}, is_regulation={is_regulation}, is_game_query={is_game_query}, intent={intent}"
        )

        if force_agent_fast_path and agent_fast_path_enabled:
            logger.info(
                "[RAG] Dedicated DB fast-path query detected, trying agent first"
            )
            agent_result = await self._try_agent_first(
                query, intent=intent, filters=filters, history=history
            )
            if agent_result is not None:
                return agent_result
            logger.info("[RAG] Dedicated DB fast-path failed, falling back to RAG")

        if is_statistical:
            logger.info(
                f"[RAG] Statistical query detected, using traditional RAG directly"
            )
            # 통계 질문이면 바로 RAG로 처리 (에이전트 건너뛰기)
            pass  # 6단계로 진행

        # 3. 일반 대화인지 확인
        elif self._is_general_conversation(query):
            logger.info(f"[RAG] General conversation detected")
            return await self._handle_general_conversation(query)

        # 4. 규정 질문인지 확인
        elif is_regulation:
            logger.info(
                f"[RAG] Regulation query detected, using traditional RAG directly"
            )
            # 에이전트 대신 RAG를 직접 사용하여 벡터 검색 성능을 활용
            pass  # 6단계로 진행

        # 5. 경기 데이터 질문인지 확인
        elif is_game_query:
            if is_game_flow_narrative:
                logger.info(
                    "[RAG] Narrative game-flow query detected, skipping agent-first"
                )
            elif agent_fast_path_enabled:
                logger.info(f"[RAG] Game query detected, trying agent first")
                agent_result = await self._try_agent_first(
                    query, intent=intent, filters=filters, history=history
                )
                if agent_result is not None:
                    return agent_result
                else:
                    logger.info(
                        f"[RAG] Game agent failed, falling back to traditional RAG"
                    )
            else:
                logger.info("[RAG] Game agent fast-path disabled, using RAG")

        # 6. 기존 RAG 방식으로 폴백 또는 일반 질문 처리

        # Merge user-provided filters with extracted filters
        # User-provided filters take precedence
        final_filters = {**extracted_filters, **(filters or {})}
        if is_game_query or entity_filter.game_date:
            final_filters.pop("team_id", None)
        if is_game_flow_narrative:
            final_filters["source_table"] = "game_flow_summary"

        # Determine year for analysis
        year = (
            final_filters.get("season_year")
            or entity_filter.season_year
            or _resolve_default_season_year(self.settings)
        )
        logger.info(f"[RAG] Analysis year: {year}")
        logger.info(f"[RAG] Final filters: {final_filters}")

        # 2. Intelligent Multi-Strategy Retrieval
        docs = []
        actual_filters: Dict[str, Any] = dict(final_filters)
        fallback_used = False
        fallback_stage = "none"
        retrieval_started_at = _rag_perf_counter()

        _sig_retrieve = inspect.signature(self.retrieve).parameters
        _sig_multi_query = inspect.signature(self.retrieve_with_multi_query).parameters

        async def _run_retrieve(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
            if "retrieval_state" in _sig_retrieve:
                kwargs["retrieval_state"] = retrieval_state
            if "intent" in _sig_retrieve:
                kwargs.setdefault("intent", intent)
            return await self.retrieve(*args, **kwargs)

        async def _run_multi_query(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
            if "retrieval_state" in _sig_multi_query:
                kwargs["retrieval_state"] = retrieval_state
            if "intent" in _sig_multi_query:
                kwargs.setdefault("intent", intent)
            return await self.retrieve_with_multi_query(*args, **kwargs)

        if is_regulation:
            # 규정 쿼리: 규정 소스 필터와 일반 필터를 병렬로 검색 후 병합
            reg_filters = {
                **final_filters,
                "source_table_in": list(_REGULATION_SOURCES),
            }
            _reg_result, _general_result = await asyncio.gather(
                _run_retrieve(
                    query,
                    filters=reg_filters,
                    entity_filter=entity_filter,
                    limit=search_limit,
                ),
                _run_retrieve(
                    query,
                    filters=final_filters,
                    entity_filter=entity_filter,
                    limit=search_limit,
                ),
                return_exceptions=True,
            )
            reg_docs = _reg_result if not isinstance(_reg_result, BaseException) else []
            general_docs = (
                _general_result
                if not isinstance(_general_result, BaseException)
                else []
            )
            if len(reg_docs) >= 5:
                docs = reg_docs
            else:
                seen = {d["id"] for d in reg_docs}
                docs = reg_docs + [d for d in general_docs if d["id"] not in seen]
            actual_filters = reg_filters
            logger.info("[RAG] Regulation parallel retrieval (run): %d docs", len(docs))

        elif search_strategy["is_ranking_query"]:
            logger.info("[RAG] Ranking query detected - using multi-query retrieval")

            # For ranking queries, use multi-query retrieval for better coverage
            if not entity_filter.position_type:
                # 투수와 타자 멀티쿼리를 병렬 실행 (풀 모드에서는 DB 호출도 병렬)
                pitcher_filters = dict(final_filters)
                pitcher_filters["source_table"] = "player_season_pitching"
                batter_filters = dict(final_filters)
                batter_filters["source_table"] = "player_season_batting"

                _results = await asyncio.gather(
                    _run_multi_query(
                        query,
                        entity_filter,
                        filters=pitcher_filters,
                        limit=search_limit,
                    ),
                    _run_multi_query(
                        query,
                        entity_filter,
                        filters=batter_filters,
                        limit=search_limit,
                    ),
                    return_exceptions=True,
                )
                actual_filters = {"branch_filters": [pitcher_filters, batter_filters]}
                docs_pitchers = (
                    _results[0] if not isinstance(_results[0], BaseException) else []
                )
                docs_batters = (
                    _results[1] if not isinstance(_results[1], BaseException) else []
                )
                if isinstance(_results[0], BaseException):
                    logger.warning("[RAG] Pitcher multi-query failed: %s", _results[0])
                if isinstance(_results[1], BaseException):
                    logger.warning("[RAG] Batter multi-query failed: %s", _results[1])

                docs = docs_pitchers + docs_batters
                logger.info(
                    f"[RAG] Multi-query ranking search: {len(docs_pitchers)} pitchers + {len(docs_batters)} batters"
                )
            else:
                # Position-specific multi-query search
                docs = await _run_multi_query(
                    query,
                    entity_filter,
                    filters=final_filters,
                    limit=search_limit,
                )
                actual_filters = dict(final_filters)

        elif (not is_game_flow_narrative) and self._should_use_single_query_retrieval(
            query=query,
            search_strategy=search_strategy,
            entity_filter=entity_filter,
            final_filters=final_filters,
        ):
            single_query_filters = dict(final_filters)
            if entity_filter.player_name:
                logger.info(
                    "[RAG] Player-specific strict-entity query: %s (single-query retrieval)",
                    entity_filter.player_name,
                )
                single_query_filters.pop("source_table", None)
            elif single_query_filters.get("source_table"):
                logger.info(
                    "[RAG] Source-table constrained strict query: %s (single-query retrieval)",
                    single_query_filters.get("source_table"),
                )
            else:
                logger.info(
                    "[RAG] Team/entity strict query detected - using single-query retrieval"
                )
            docs = await _run_retrieve(
                query,
                filters=single_query_filters,
                entity_filter=entity_filter,
                limit=search_limit,
            )
            actual_filters = dict(single_query_filters)

        else:
            logger.info("[RAG] General search strategy with multi-query")
            # Use multi-query for general searches to improve coverage
            docs = await _run_multi_query(
                query,
                entity_filter,
                filters=final_filters,
                limit=search_limit,
            )
            actual_filters = dict(final_filters)

        # 3. Fallback Strategy
        if not docs and final_filters:
            logger.info("[RAG] No results with filters, attempting fallback search")
            # Remove restrictive filters one by one
            fallback_filters = dict(final_filters)

            # Try removing source_table first
            if "source_table" in fallback_filters:
                fallback_filters.pop("source_table")
                docs = await _run_retrieve(
                    query,
                    filters=fallback_filters,
                    limit=max(
                        search_limit,
                        int(self.settings.retrieval_fallback_limit_relaxed),
                    ),
                    use_hyde=False,
                )
                actual_filters = dict(fallback_filters)
                fallback_used = True
                fallback_stage = "without_source_table"
                logger.info(f"[RAG] Fallback without source_table: {len(docs)} docs")

            # If still no results, try without team filter
            if not docs and "team_id" in fallback_filters:
                fallback_filters.pop("team_id")
                docs = await _run_retrieve(
                    query,
                    filters=fallback_filters,
                    limit=max(
                        search_limit,
                        int(self.settings.retrieval_fallback_limit_relaxed),
                    ),
                    use_hyde=False,
                )
                actual_filters = dict(fallback_filters)
                fallback_used = True
                fallback_stage = "without_team_id"
                logger.info(f"[RAG] Fallback without team filter: {len(docs)} docs")

            # Final fallback: only keep year and league filters
            if not docs:
                minimal_filters = {}
                if "season_year" in final_filters:
                    minimal_filters["season_year"] = final_filters["season_year"]
                if "meta.league" in final_filters:
                    minimal_filters["meta.league"] = final_filters["meta.league"]
                docs = await _run_retrieve(
                    query,
                    filters=minimal_filters,
                    limit=max(
                        search_limit,
                        int(self.settings.retrieval_fallback_limit_minimal),
                    ),
                    use_hyde=False,
                )
                actual_filters = dict(minimal_filters)
                fallback_used = True
                fallback_stage = "minimal"
                logger.info(f"[RAG] Minimal fallback: {len(docs)} docs")

        logger.info(f"[RAG] Final retrieval result: {len(docs)} documents")

        # 규정 쿼리 top-up: 일반 검색 경로를 탄 경우에만 필요 (병렬 경로는 이미 처리됨)
        # is_regulation=True 인 경우 위의 병렬 블록에서 이미 처리되었으므로 스킵
        pass

        # --- DB 연결 장애 폴백 ---
        # docs가 있으면 일부 검색이 성공한 것이므로 에러가 있어도 폴백하지 않음
        if retrieval_state.get("db_error") is not None and not docs:
            logger.warning(
                "[RAG] DB was unavailable during retrieval. cause=%s",
                retrieval_state.get("db_error"),
            )
            db_unavailable_context = _build_db_unavailable_context(
                query, entity_filter, year
            )
            history_block = _history_context_block(history)
            if history_block:
                db_unavailable_context = history_block + "\n\n" + db_unavailable_context
            prompt = FOLLOWUP_PROMPT.format(
                question=query, context=db_unavailable_context
            )
            db_unavailable_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            db_unavailable_messages.extend(_history_for_messages(history))
            db_unavailable_messages.append({"role": "user", "content": prompt})
            answer = await self._generate(db_unavailable_messages)
            answer = _ensure_answer_prefix(answer, DB_UNAVAILABLE_PREFIX)
            asyncio.create_task(
                self._record_retrieval_event(
                    query=query,
                    intent=intent,
                    final_filters=final_filters,
                    docs=[],
                    retrieval_started_at=retrieval_started_at,
                    success=False,
                    error_type="db_unavailable",
                    original_filters=final_filters,
                    actual_filters=actual_filters,
                    fallback_used=fallback_used,
                    fallback_stage=fallback_stage,
                )
            )
            return {
                "answer": answer,
                "citations": [],
                "intent": intent,
                "retrieved": [],
                "strategy": "llm_knowledge_db_unavailable",
                "entity_filter": {
                    "season_year": entity_filter.season_year,
                    "team_id": entity_filter.team_id,
                    "player_name": entity_filter.player_name,
                    "stat_type": entity_filter.stat_type,
                    "position_type": entity_filter.position_type,
                },
            }
        # --- DB 연결 장애 폴백 끝 ---

        if retrieval_state.get("embedding_error") is not None and not docs:
            logger.warning(
                "[RAG] Query embedding failed during retrieval, using limited fallback. cause=%s",
                retrieval_state.get("embedding_error"),
            )
            embedding_failed_context = _build_embedding_failed_context(
                query, entity_filter, year
            )
            history_block = _history_context_block(history)
            if history_block:
                embedding_failed_context = (
                    history_block + "\n\n" + embedding_failed_context
                )
            prompt = FOLLOWUP_PROMPT.format(
                question=query, context=embedding_failed_context
            )
            embedding_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            embedding_messages.extend(_history_for_messages(history))
            embedding_messages.append({"role": "user", "content": prompt})
            answer = await self._generate(embedding_messages)
            answer = _ensure_answer_prefix(answer, EMBEDDING_FAILED_PREFIX)
            asyncio.create_task(
                self._record_retrieval_event(
                    query=query,
                    intent=intent,
                    final_filters=final_filters,
                    docs=[],
                    retrieval_started_at=retrieval_started_at,
                    success=False,
                    error_type="embedding_failed",
                    original_filters=final_filters,
                    actual_filters=actual_filters,
                    fallback_used=fallback_used,
                    fallback_stage=fallback_stage,
                )
            )
            return {
                "answer": answer,
                "citations": [],
                "intent": intent,
                "retrieved": [],
                "strategy": "llm_knowledge_embedding_failed",
                "entity_filter": {
                    "season_year": entity_filter.season_year,
                    "team_id": entity_filter.team_id,
                    "player_name": entity_filter.player_name,
                    "stat_type": entity_filter.stat_type,
                    "position_type": entity_filter.position_type,
                },
            }

        retrieved_docs = list(docs)
        docs = self._rerank_docs(docs)
        asyncio.create_task(
            self._record_retrieval_event(
                query=query,
                intent=intent,
                final_filters=final_filters,
                docs=retrieved_docs,
                selected_docs=docs,
                retrieval_started_at=retrieval_started_at,
                success=bool(retrieved_docs),
                error_type=(
                    None
                    if retrieved_docs
                    else retrieval_state.get("error_type") or "zero_hit"
                ),
                original_filters=final_filters,
                actual_filters=actual_filters,
                fallback_used=fallback_used,
                fallback_stage=fallback_stage,
            )
        )

        docs = _sort_docs_for_context(
            docs,
            is_regulation=is_regulation,
            query=query,
        )

        # 규정 쿼리: LLM에 전달 전 비규정 청크 제거로 환각 억제
        if is_regulation and docs:
            regulation_sources = _REGULATION_SOURCES
            filtered = [d for d in docs if d.get("source_table") in regulation_sources]
            if filtered:
                docs = filtered[:8]

        # 2. 데이터 처리 및 보강
        _format_start = _rag_perf_counter()
        processed_data = await self._process_and_enrich_docs(docs, year)

        # 3. 의도별 컨텍스트 생성 (새로운 컨텍스트 포맷터 사용)
        formatted_context = self.context_formatter.format_context(
            processed_data, intent, query, entity_filter, year
        )

        # Zero-hit 오버라이드: 모든 폴백 후에도 문서가 없으면 풍부한 가이드 컨텍스트 사용
        if not docs:
            logger.info(
                "[RAG] Zero documents after all fallbacks — using zero-hit guidance context"
            )
            formatted_context = self.context_formatter.format_zero_hit_guidance(
                query, entity_filter, year, final_filters
            )
        try:
            AI_RAG_STAGE_DURATION_SECONDS.labels(stage="format").observe(
                _rag_perf_counter() - _format_start
            )
        except Exception:  # noqa: BLE001
            pass

        # 대화 기록 컨텍스트 추가
        history_block = _history_context_block(history)
        if history_block:
            formatted_context = history_block + "\n\n" + formatted_context

        # 4. LLM 프롬프트 구성
        prompt = FOLLOWUP_PROMPT.format(question=query, context=formatted_context)

        # DEBUG: 컨텍스트 로깅
        logger.info(f"[RAG_DEBUG] Question: {query}")
        logger.info(f"[RAG_DEBUG] Formatted context length: {len(formatted_context)}")
        logger.info(f"[RAG_DEBUG] Formatted context content: {formatted_context}")

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 KBO 리그 야구 전문가 'BEGA'입니다. "
                    "반드시 주어진 검색 컨텍스트만 근거로 답하십시오. "
                    "컨텍스트에 없는 내용을 추론하거나 생성하지 마십시오. "
                    "2026년 규정 변화(외국인 4명, 아시아 쿼터, 11회 연장, 수비 시프트 등)가 "
                    "컨텍스트에 있으면 그것을 최신 공식 정보로 간주하여 상세히 답하십시오. "
                    "정보가 없으면 '컨텍스트에서 확인되지 않습니다'라고만 답하십시오."
                ),
            }
        ]
        messages.extend(_history_for_messages(history))
        messages.append({"role": "user", "content": prompt})

        # 5. LLM을 호출하여 답변 생성
        answer = await self._generate(messages)
        if not docs:
            answer = _ensure_zero_hit_answer_prefix(answer)

        # 6. 최종 결과 구성
        return {
            "answer": answer,
            "citations": _build_citations(docs),
            "intent": intent,
            "retrieved": docs,
            "strategy": "rag_v3_enhanced",  # 업데이트된 버전 명시
            "entity_filter": {
                "season_year": entity_filter.season_year,
                "team_id": entity_filter.team_id,
                "player_name": entity_filter.player_name,
                "stat_type": entity_filter.stat_type,
                "position_type": entity_filter.position_type,
            },
        }
