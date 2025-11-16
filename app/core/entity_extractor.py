"""
질문에서 핵심 엔티티(Entity)를 추출하여 메타데이터 필터링을 위한 정보를 제공하는 모듈입니다.

이 모듈은 사용자의 질문에서 연도, 팀명, 선수명, 통계 지표 등을 자동으로 인식하여
RAG 검색 시 효과적인 메타데이터 필터를 구성할 수 있도록 도와줍니다.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EntityFilter:
    """질문에서 추출된 엔티티 정보를 저장하는 필터 클래스"""
    season_year: Optional[int] = None      # 시즌 연도 (예: 2024)
    team_id: Optional[str] = None          # 팀 식별자 (예: "LG", "KIA")
    player_name: Optional[str] = None      # 선수 이름 (예: "김현수")
    stat_type: Optional[str] = None        # 통계 지표 (예: "ops", "era") 
    position_type: Optional[str] = None    # 포지션 타입 ('pitcher', 'batter')
    league_type: str = "정규시즌"           # 리그 타입 (기본값: 정규시즌)

# KBO 팀명 매핑 테이블 (사용자 입력 → 표준 팀 코드)
TEAM_MAPPING = {
    # KIA
    "KIA": "KIA", "기아": "KIA", "타이거즈": "KIA", "기아타이거즈": "KIA",
    # LG
    "LG": "LG", "엘지": "LG", "트윈스": "LG", "LG트윈스": "LG",
    # 두산
    "두산": "두산", "베어스": "두산", "두산베어스": "두산",
    # 롯데
    "롯데": "롯데", "자이언츠": "롯데", "롯데자이언츠": "롯데", "거인": "롯데",
    # 삼성
    "삼성": "삼성", "라이온즈": "삼성", "삼성라이온즈": "삼성", "사자": "삼성",
    # 키움
    "키움": "키움", "히어로즈": "키움", "키움히어로즈": "키움", "영웅": "키움",
    # 한화
    "한화": "한화", "이글스": "한화", "한화이글스": "한화", "독수리": "한화",
    # KT
    "KT": "KT", "위즈": "KT", "KT위즈": "KT", "케이티": "KT",
    # NC
    "NC": "NC", "다이노스": "NC", "NC다이노스": "NC", "공룡": "NC", "엔씨": "NC",
    # SSG
    "SSG": "SSG", "랜더스": "SSG", "SSG랜더스": "SSG", "에스에스지": "SSG",
}

# 야구 통계 지표 매핑 테이블 (사용자 입력 → 표준 지표명)
STAT_MAPPING = {
    # 타자 지표
    "OPS": "ops", "ops": "ops", "출루율+장타율": "ops",
    "타율": "avg", "AVG": "avg", "평균": "avg",
    "홈런": "home_runs", "HR": "home_runs", "hr": "home_runs", "방고": "home_runs",
    "타점": "rbi", "RBI": "rbi", "rbi": "rbi", 
    "도루": "stolen_bases", "SB": "stolen_bases", "sb": "stolen_bases",
    "WAR": "war", "war": "war", "대체선수대비승수": "war",
    "wRC+": "wrc_plus", "wrc+": "wrc_plus",
    
    # 투수 지표
    "ERA": "era", "era": "era", "평균자책": "era", "방어율": "era",
    "WHIP": "whip", "whip": "whip",
    "승수": "wins", "W": "wins", "win": "wins",
    "세이브": "saves", "SV": "saves", "save": "saves",
    "삼진": "strikeouts", "K": "strikeouts", "SO": "strikeouts",
    "이닝": "innings_pitched", "IP": "innings_pitched", "ip": "innings_pitched",
}

# 야구 포지션 및 역할 매핑 테이블
POSITION_MAPPING = {
    "투수": "pitcher", "pitcher": "pitcher", "피처": "pitcher",
    "선발": "pitcher", "선발투수": "pitcher", "SP": "pitcher",
    "불펜": "pitcher", "릴리프": "pitcher", "RP": "pitcher", "마무리": "pitcher",
    "타자": "batter", "batter": "batter", "배터": "batter", 
    "내야수": "batter", "외야수": "batter", "포수": "batter",
}

def extract_year(query: str) -> Optional[int]:
    """질문에서 연도를 추출합니다."""
    # 2020~2025년 범위 내의 4자리 숫자 찾기
    year_pattern = r'\b(202[0-5])\b'
    match = re.search(year_pattern, query)
    if match:
        return int(match.group(1))
    
    # "작년", "올해", "지난해" 등의 상대적 표현 처리
    current_year = 2025  # 시스템 기준년도
    if re.search(r'(작년|지난해)', query):
        return current_year - 1
    elif re.search(r'(올해|금년|이번해)', query):
        return current_year
    elif re.search(r'재작년', query):
        return current_year - 2
    
    return None

def extract_team(query: str) -> Optional[str]:
    """질문에서 팀명을 추출합니다."""
    # 모든 팀 이름/별칭을 체크
    for team_variant, standard_id in TEAM_MAPPING.items():
        if team_variant in query:
            return standard_id
    return None

def extract_stat_type(query: str) -> Optional[str]:
    """질문에서 통계 지표를 추출합니다."""
    for stat_variant, standard_stat in STAT_MAPPING.items():
        if stat_variant in query:
            return standard_stat
    return None

def extract_position_type(query: str) -> Optional[str]:
    """질문에서 포지션/역할을 추출합니다."""
    for pos_variant, standard_pos in POSITION_MAPPING.items():
        if pos_variant in query:
            return standard_pos
    return None

def extract_player_name(query: str) -> Optional[str]:
    """
    질문에서 선수명을 추출합니다.
    한국어 이름 패턴을 기반으로 하며, 일반적인 야구 용어들을 제외합니다.
    """
    # 2-4글자 한글 이름 패턴 (성+이름)
    name_pattern = r'[가-힣]{2,4}(?=\s|$|[^가-힣])'
    matches = re.findall(name_pattern, query)
    
    # 팀명이나 통계 용어가 아닌 것 중 첫 번째를 선수명으로 간주
    for match in matches:
        if match not in TEAM_MAPPING and match not in STAT_MAPPING:
            # 일반적인 야구 용어들도 제외
            common_terms = {
                "선수", "타자", "투수", "순위", "랭킹", "기록", "성적", "경기", "시즌",
                "이닝", "타율", "방어율", "최고", "최저", "가장", "제일", "상위", "하위",
                "리그", "야구", "올해", "작년", "금년", "시즌", "월드", "베이스", "볼넷"
            }
            if match not in common_terms:
                return match
    
    return None

def is_ranking_query(query: str) -> bool:
    """랭킹/순위 관련 질문인지 판단합니다."""
    ranking_keywords = [
        "순위", "랭킹", "상위", "하위", "1위", "2위", "3위", "4위", "5위",
        "최고", "최저", "가장", "제일", "톱", "TOP", "베스트", "worst",
        "명", "리스트", "목록"
    ]
    return any(keyword in query for keyword in ranking_keywords)

def extract_ranking_count(query: str) -> Optional[int]:
    """순위 질문에서 요청된 개수를 추출합니다."""
    # "상위 5명", "10위까지", "톱 3" 등의 패턴
    count_patterns = [
        r'상위\s*(\d+)',
        r'(\d+)명',
        r'(\d+)위까지',
        r'톱\s*(\d+)',
        r'TOP\s*(\d+)',
        r'베스트\s*(\d+)',
    ]
    
    for pattern in count_patterns:
        match = re.search(pattern, query)
        if match:
            count = int(match.group(1))
            return min(count, 20)  # 최대 20명까지 제한
    
    # 숫자가 명시되지 않은 경우 기본값
    if is_ranking_query(query):
        return 5
    
    return None

def extract_entities_from_query(query: str) -> EntityFilter:
    """질문에서 모든 엔티티를 추출하여 EntityFilter 객체로 반환합니다."""
    logger.info(f"[EntityExtractor] Extracting entities from: {query}")
    
    entity_filter = EntityFilter()
    
    # 각 엔티티 추출
    entity_filter.season_year = extract_year(query)
    entity_filter.team_id = extract_team(query)
    entity_filter.player_name = extract_player_name(query)
    entity_filter.stat_type = extract_stat_type(query)
    entity_filter.position_type = extract_position_type(query)
    
    # 로깅
    logger.info(f"[EntityExtractor] Extracted entities: "
                f"year={entity_filter.season_year}, "
                f"team={entity_filter.team_id}, "
                f"player={entity_filter.player_name}, "
                f"stat={entity_filter.stat_type}, "
                f"position={entity_filter.position_type}")
    
    return entity_filter

def convert_to_db_filters(entity_filter: EntityFilter) -> Dict[str, Any]:
    """EntityFilter를 데이터베이스 검색용 필터 딕셔너리로 변환합니다."""
    filters: Dict[str, Any] = {}
    
    # 시즌 연도 필터
    if entity_filter.season_year:
        filters["season_year"] = entity_filter.season_year
    
    # 팀 필터  
    if entity_filter.team_id:
        filters["team_id"] = entity_filter.team_id
    
    # 리그 타입 (항상 정규시즌으로 필터링)
    filters["meta.league"] = entity_filter.league_type
    
    # 포지션별 테이블 필터링
    if entity_filter.position_type == "pitcher":
        filters["source_table"] = "player_season_pitching"
    elif entity_filter.position_type == "batter":
        filters["source_table"] = "player_season_batting"
    
    return filters

def enhance_search_strategy(query: str) -> Dict[str, Any]:
    """
    질문을 분석하여 검색 전략을 결정합니다.
    
    Returns:
        strategy: 검색 전략과 파라미터를 포함한 딕셔너리
    """
    entity_filter = extract_entities_from_query(query)
    db_filters = convert_to_db_filters(entity_filter)
    
    strategy = {
        "entity_filter": entity_filter,
        "db_filters": db_filters,
        "is_ranking_query": is_ranking_query(query),
        "ranking_count": extract_ranking_count(query),
        "search_limit": 15,  # 기본 검색 제한
    }
    
    # 랭킹 쿼리인 경우 검색 제한 증가
    if strategy["is_ranking_query"]:
        strategy["search_limit"] = max(20, (strategy["ranking_count"] or 5) * 3)
    
    # 특정 선수에 대한 질문인 경우 검색 제한 감소
    if entity_filter.player_name:
        strategy["search_limit"] = 10
    
    logger.info(f"[EntityExtractor] Search strategy: {strategy}")
    
    return strategy