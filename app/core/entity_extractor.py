"""
질문에서 핵심 엔티티(Entity)를 추출하여 메타데이터 필터링을 위한 정보를 제공하는 모듈입니다.

이 모듈은 사용자의 질문에서 연도, 팀명, 선수명, 통계 지표 등을 자동으로 인식하여
RAG 검색 시 효과적인 메타데이터 필터를 구성할 수 있도록 도와줍니다.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class EntityFilter:
    """질문에서 추출된 엔티티 정보를 저장하는 필터 클래스"""

    season_year: Optional[int] = None  # 시즌 연도 (예: 2025)
    team_id: Optional[str] = None  # 팀 식별자 (예: "LG", "KIA")
    player_name: Optional[str] = None  # 선수 이름 (예: "김현수")
    stat_type: Optional[str] = None  # 통계 지표 (예: "ops", "era")
    position_type: Optional[str] = None  # 포지션 타입 ('pitcher', 'batter')
    league_type: Optional[str] = None  # 리그 타입 (None이면 모든 리그 검색)
    award_type: Optional[str] = None  # 수상 유형 (예: "mvp", "golden_glove")
    movement_type: Optional[str] = None  # 이적 유형 (예: "fa", "trade")
    game_date: Optional[str] = None  # 경기 날짜 (예: "2025-05-10")
    position_code: Optional[str] = None  # 표준 포지션 코드 (예: "1B", "SS", "PH")


# KBO 팀명 매핑 테이블 (사용자 입력 → 실제 DB team_id)
TEAM_MAPPING = {
    # KIA (DB: KIA)
    "KIA": "KIA",
    "기아": "KIA",
    "타이거즈": "KIA",
    "기아타이거즈": "KIA",
    "HT": "KIA",
    # LG (DB: LG)
    "LG": "LG",
    "엘지": "LG",
    "트윈스": "LG",
    "LG트윈스": "LG",
    # 두산 (DB: DB)
    "두산": "DB",
    "베어스": "DB",
    "두산베어스": "DB",
    "DO": "DB",
    "OB": "DB",
    # 롯데 (DB: LT)
    "롯데": "LT",
    "자이언츠": "LT",
    "롯데자이언츠": "LT",
    "거인": "LT",
    # 삼성 (DB: SS)
    "삼성": "SS",
    "라이온즈": "SS",
    "삼성라이온즈": "SS",
    "사자": "SS",
    # 키움 (DB: KH)
    "키움": "KH",
    "히어로즈": "KH",
    "키움 히어로즈": "KH",
    "영웅": "KH",
    "넥센": "KH",
    "넥센히어로즈": "KH",
    "KI": "KH",
    "NX": "KH",
    "WO": "KH",
    # 한화 (DB: HH)
    "한화": "HH",
    "이글스": "HH",
    "한화이글스": "HH",
    "독수리": "HH",
    # KT (DB: KT)
    "KT": "KT",
    "위즈": "KT",
    "KT위즈": "KT",
    "케이티": "KT",
    # NC (DB: NC)
    "NC": "NC",
    "다이노스": "NC",
    "NC다이노스": "NC",
    "공룡": "NC",
    "엔씨": "NC",
    # SSG 랜더스 (DB: SSG)
    "SSG": "SSG",
    "SK": "SSG",
    "랜더스": "SSG",
    "SSG랜더스": "SSG",
    "에스에스지": "SSG",
}

# 야구 통계 지표 매핑 테이블 (사용자 입력 → 표준 지표명)
STAT_MAPPING = {
    # 타자 지표
    "OPS": "ops",
    "ops": "ops",
    "출루율+장타율": "ops",
    "타율": "avg",
    "AVG": "avg",
    "평균": "avg",
    "홈런": "home_runs",
    "HR": "home_runs",
    "hr": "home_runs",
    "방고": "home_runs",
    "타점": "rbi",
    "RBI": "rbi",
    "rbi": "rbi",
    "도루": "stolen_bases",
    "SB": "stolen_bases",
    "sb": "stolen_bases",
    "WAR": "war",
    "war": "war",
    "대체선수대비승수": "war",
    "wRC+": "wrc_plus",
    "wrc+": "wrc_plus",
    "득점권 타율": "scoring_position_avg",
    "득점권타율": "scoring_position_avg",
    "득타율": "scoring_position_avg",
    "승률": "win_rate",
    "홈 승률": "home_win_rate",
    "홈경기 승률": "home_win_rate",
    "안방 승률": "home_win_rate",
    # 투수 지표
    "ERA": "era",
    "era": "era",
    "평균자책": "era",
    "평균자책점": "era",
    "방어율": "era",
    "WHIP": "whip",
    "whip": "whip",
    "승수": "wins",
    "W": "wins",
    "win": "wins",
    "세이브": "saves",
    "SV": "saves",
    "save": "saves",
    "삼진": "strikeouts",
    "K": "strikeouts",
    "SO": "strikeouts",
    "이닝": "innings_pitched",
    "IP": "innings_pitched",
    "ip": "innings_pitched",
    "승률": "win_rate",
    "WPCT": "win_rate",
    "득점권타율": "ops_risp",  # 임시로 ops_risp 또는 별도 필드 정의 필요
}

# 야구 포지션 및 역할 매핑 테이블 (유형별 분류)
POSITION_MAPPING = {
    "투수": "pitcher",
    "pitcher": "pitcher",
    "피처": "pitcher",
    "선발": "pitcher",
    "선발투수": "pitcher",
    "SP": "pitcher",
    "불펜": "pitcher",
    "릴리프": "pitcher",
    "RP": "pitcher",
    "마무리": "pitcher",
    "타자": "batter",
    "batter": "batter",
    "배터": "batter",
    "내야수": "batter",
    "외야수": "batter",
    "포수": "batter",
    "1B": "batter",
    "2B": "batter",
    "3B": "batter",
    "SS": "batter",
    "LF": "batter",
    "CF": "batter",
    "RF": "batter",
    "DH": "batter",
    "C": "batter",
    "P": "pitcher",
}

# 한국어 포지션 명칭 → 표준 코드 매핑
POSITION_NAME_TO_CODE = {
    "1루수": "1B",
    "일루수": "1B",
    "2루수": "2B",
    "이루수": "2B",
    "3루수": "3B",
    "삼루수": "3B",
    "유격수": "SS",
    "좌익수": "LF",
    "중견수": "CF",
    "우익수": "RF",
    "지명타자": "DH",
    "지타": "DH",
    "포수": "C",
    "투수": "P",
    "대타": "PH",
    "대주자": "PR",
}

# 한자/약어 → 한국어 명칭 매핑 (baseball.py와 동기화)
POS_ABBR_TO_NAME = {
    "一": "1루수",
    "二": "2루수",
    "三": "3루수",
    "유": "유격수",
    "좌": "좌익수",
    "우": "우익수",
    "중": "중견수",
    "포": "포수",
    "투": "투수",
    "타": "대타",
    "지": "지명타자",
    "주": "대주자",
}

# 리그 타입 매핑 테이블 (사용자 입력 → 표준 리그명)
LEAGUE_TYPE_MAPPING = {
    # 포스트시즌 (더 구체적인 것들을 먼저)
    "포스트시즌": "포스트시즌",
    "후기리그": "포스트시즌",
    # 한국시리즈
    "한국시리즈": "한국시리즈",
    "코리안시리즈": "한국시리즈",
    "KS": "한국시리즈",
    "시리즈": "한국시리즈",
    "우승결정전": "한국시리즈",
    # 와일드카드
    "와일드카드": "와일드카드",
    "wildcard": "와일드카드",
    "WC": "와일드카드",
    # 준플레이오프
    "준플레이오프": "준플레이오프",
    "준PO": "준플레이오프",
    # 플레이오프 (일반)
    "플레이오프": "플레이오프",
    "PO": "플레이오프",
    # 정규시즌 (가장 일반적인 것은 마지막에)
    "정규시즌": "정규시즌",
    "레귤러시즌": "정규시즌",
}

# 수상 유형 매핑 테이블 (사용자 입력 → 표준 수상명)
AWARD_MAPPING = {
    # MVP
    "MVP": "mvp",
    "엠브이피": "mvp",
    "최우수선수": "mvp",
    "mvp": "mvp",
    # 신인왕
    "신인왕": "rookie",
    "신인상": "rookie",
    "루키": "rookie",
    # 골든글러브
    "골든글러브": "golden_glove",
    "골글": "golden_glove",
    "황금장갑": "golden_glove",
    # 타이틀 홀더들
    "타격왕": "batting_title",
    "타율왕": "batting_title",
    "홈런왕": "hr_leader",
    "장타왕": "hr_leader",
    "타점왕": "rbi_leader",
    "도루왕": "sb_leader",
    "도루 1위": "sb_leader",
    "다승왕": "wins_leader",
    "다승": "wins_leader",
    "방어율왕": "era_leader",
    "방어율 1위": "era_leader",
    "세이브왕": "saves_leader",
    "세이브 1위": "saves_leader",
    "탈삼진왕": "so_leader",
    "탈삼진 1위": "so_leader",
}

# 선수 이동 유형 매핑 테이블 (사용자 입력 → 표준 이동 유형)
MOVEMENT_MAPPING = {
    # FA
    "FA": "fa",
    "에프에이": "fa",
    "자유계약": "fa",
    "fa": "fa",
    # 트레이드
    "트레이드": "trade",
    "교환": "trade",
    "맞트레이드": "trade",
    # 드래프트
    "드래프트": "draft",
    "신인드래프트": "draft",
    "지명": "draft",
    "1차지명": "draft",
    # 외국인 선수
    "외국인": "foreign",
    "용병": "foreign",
    "외인": "foreign",
    # 기타
    "방출": "release",
    "웨이버": "release",
    "은퇴": "retirement",
    "현역은퇴": "retirement",
    "군보류": "military",
    "군입대": "military",
    "복귀": "return",
    "군복귀": "return",
}


def extract_year(query: str) -> Optional[int]:
    """질문에서 연도를 추출합니다. (4자리 및 2자리 연도 지원)"""
    import datetime as dt

    current_year = dt.datetime.now().year

    # 1. 4자리 연도 패턴 ("2024년", "1999시즌" 등)
    full_year_patterns = [
        r"(19[8-9][0-9]|20[0-9][0-9])년",  # 1980~2099년
        r"(19[8-9][0-9]|20[0-9][0-9])시즌",
        r"(19[8-9][0-9]|20[0-9][0-9])년도",
        r"(19[8-9][0-9]|20[0-9][0-9])(?=\s)",
        r"(?<=\s)(19[8-9][0-9]|20[0-9][0-9])",
        r"^(19[8-9][0-9]|20[0-9][0-9])",
        r"(19[8-9][0-9]|20[0-9][0-9])$",
    ]

    for pattern in full_year_patterns:
        match = re.search(pattern, query)
        if match:
            year = int(match.group(1))
            if 1982 <= year <= current_year + 5:  # KBO 원년(1982)부터
                return year

    # 2. 2자리 연도 패턴 ("25년", "99시즌")
    short_year_patterns = [r"(\d{2})년", r"(\d{2})시즌", r"(\d{2})년도"]

    for pattern in short_year_patterns:
        match = re.search(pattern, query)
        if match:
            short_year = int(match.group(1))
            # KBO 문맥에 따른 연도 변환 (82~99: 1900년대, 00~현재+5: 2000년대)
            if 82 <= short_year <= 99:
                return 1900 + short_year
            elif 0 <= short_year <= (current_year + 5) % 100:
                return 2000 + short_year
            # 그 외(예: "50년")는 나이 등일 수 있으므로 무시

    # 3. 상대적 연도 표현
    if re.search(r"(작년|지난해)", query):
        return current_year - 1
    elif re.search(r"(올해|금년|이번해)", query):
        return current_year
    elif re.search(r"재작년", query):
        return current_year - 2
    elif re.search(r"내년", query):
        return current_year + 1

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
    """질문에서 포지션/역할 유형(투수/타자)을 추출합니다."""
    # 1. 포지션 매핑 테이블에서 직접 매칭
    for pos_variant, standard_pos in POSITION_MAPPING.items():
        if pos_variant in query:
            return standard_pos

    # 2. 개별 포지션 코드를 통해 역추적 (batter/pitcher)
    pos_code = extract_position_code(query)
    if pos_code:
        if pos_code == "P":
            return "pitcher"
        return "batter"

    return None


def extract_position_code(query: str) -> Optional[str]:
    """질문에서 구체적인 포지션 코드(1B, SS 등)를 추출하여 표준화합니다."""
    # 1. 풀네임 및 약어 검색
    for name in POSITION_NAME_TO_CODE.keys():
        if name in query:
            return POSITION_NAME_TO_CODE[name]

    for abbr in POS_ABBR_TO_NAME.keys():
        if abbr in query:
            name = POS_ABBR_TO_NAME[abbr]
            return POSITION_NAME_TO_CODE.get(name)

    # 2. 텍스트 직접 분석 및 표준화
    # 정규표현식으로 타순+포지션 형태(예: "9번타자", "1루수") 등은 이미 위에서 처리됨
    # "타일", "지타" 같은 복합어도 처리 필요 시 standardize_position 활용 가능

    return None


def standardize_position(position_str: str) -> Optional[str]:
    """
    다양한 포지션 표현(Hanja, Hangul, Full Name)을 표준 코드("1B", "SS", "P" 등)로 변환합니다.
    예: "一" -> "1B", "타一" -> "PH", "유격수" -> "SS"
    """
    if not position_str:
        return None

    position_str = position_str.strip()

    # 1. 이미 표준 코드인 경우
    if position_str.upper() in POSITION_NAME_TO_CODE.values():
        return position_str.upper()

    # 2. 한국어 풀네임인 경우
    if position_str in POSITION_NAME_TO_CODE:
        return POSITION_NAME_TO_CODE[position_str]

    # 3. 한자/약어 처리 (baseball.py 로직 차용)
    if len(position_str) == 1:
        name = POS_ABBR_TO_NAME.get(position_str)
        if name:
            return POSITION_NAME_TO_CODE.get(name)

    if len(position_str) == 2:
        first = position_str[0]
        second = position_str[1]
        # 서브 포지션(대타, 대주자 등)이 있으면 서브 포지션을 우선시 (PH, PR)
        if first in ["타", "주", "지"] and second in POS_ABBR_TO_NAME:
            name = POS_ABBR_TO_NAME.get(first)
            if name:
                return POSITION_NAME_TO_CODE.get(name)

    # 4. 키워드 매칭
    for name, code in POSITION_NAME_TO_CODE.items():
        if name in position_str:
            return code

    return None


def extract_league_type(query: str) -> Optional[str]:
    """질문에서 리그 타입을 추출합니다."""
    # 직접적인 리그 타입 매핑 확인
    for league_variant, standard_league in LEAGUE_TYPE_MAPPING.items():
        if league_variant in query:
            return standard_league

    # 특별한 키워드로 리그 타입 추론
    # "마지막 경기"는 보통 포스트시즌(한국시리즈)을 의미
    if re.search(r"(마지막\s*경기|최종\s*경기|우승\s*경기)", query):
        return "한국시리즈"

    # "결승"이나 "우승" 관련은 한국시리즈
    if re.search(r"(결승|우승|챔피언)", query):
        return "한국시리즈"

    # 명시적으로 정규시즌이 아닌 경우 판단
    postseason_indicators = [
        r"(플레이오프|PO)",
        r"(포스트시즌)",
        r"(와일드카드)",
        r"(준\s*플레이오프)",
    ]

    for indicator in postseason_indicators:
        if re.search(indicator, query):
            return "포스트시즌"

    return None


def extract_award_type(query: str) -> Optional[str]:
    """질문에서 수상 유형을 추출합니다."""
    for award_variant, standard_award in AWARD_MAPPING.items():
        if award_variant in query:
            return standard_award

    # 수상 관련 키워드 추가 탐지
    if re.search(r"(수상|상을?\s*받|상을?\s*탔)", query):
        return "any"  # 어떤 수상이든

    return None


def extract_movement_type(query: str) -> Optional[str]:
    """질문에서 선수 이동/이적 유형을 추출합니다."""
    for movement_variant, standard_movement in MOVEMENT_MAPPING.items():
        if movement_variant in query:
            return standard_movement

    # 이적 관련 키워드 추가 탐지
    if re.search(r"(이적|영입|계약)", query):
        return "any"  # 어떤 이동이든

    return None


def extract_game_date(query: str) -> Optional[str]:
    """질문에서 경기 날짜를 추출합니다."""
    import datetime as dt

    today = dt.date.today()

    # 1. 상대적 날짜 표현
    if re.search(r"(어제|지난\s*경기)", query):
        return (today - dt.timedelta(days=1)).isoformat()
    elif re.search(r"(오늘|금일)", query):
        return today.isoformat()
    elif re.search(r"(그저께|그제)", query):
        return (today - dt.timedelta(days=2)).isoformat()
    elif re.search(r"(내일)", query):
        return (today + dt.timedelta(days=1)).isoformat()

    # 2. YYYY-MM-DD 또는 YYYY.MM.DD 패턴
    date_patterns = [
        r"(\d{4})[-.\/](\d{1,2})[-.\/](\d{1,2})",  # 2025-04-12, 2025/4/12
    ]

    for pattern in date_patterns:
        match = re.search(pattern, query)
        if match:
            year, month, day = match.groups()
            try:
                parsed_date = dt.date(int(year), int(month), int(day))
                return parsed_date.isoformat()
            except ValueError:
                continue

    # 3. MM월 DD일 패턴 (현재 연도 가정)
    month_day_pattern = r"(\d{1,2})월\s*(\d{1,2})일"
    match = re.search(month_day_pattern, query)
    if match:
        month, day = match.groups()
        try:
            # 현재 연도로 가정, 미래 날짜면 작년으로
            parsed_date = dt.date(today.year, int(month), int(day))
            if parsed_date > today:
                parsed_date = dt.date(today.year - 1, int(month), int(day))
            return parsed_date.isoformat()
        except ValueError:
            pass

    return None


def extract_player_name(query: str) -> Optional[str]:
    """
    질문에서 선수명을 추출합니다.
    한국어 이름과 외국인 선수명을 모두 인식합니다.
    """
    # 외국인 선수명 목록 (2024-2025 시즌 주요 선수들)
    foreign_players = {
        # 타자
        "디아즈",
        "무니에",
        "소크라테스",
        "로사리오",
        "나바로",
        "페렐",
        "말라도나비치",
        "오리엘리",
        "수아레즈",
        "알카트라즈",
        "리베르",
        "라모스",
        "하스",
        "맥가리",
        "바나거즈",
        "루이즈",
        "무라타",
        "우싱",
        "오캬마",
        "이치히라",
        "페르난데스",
        # 투수
        "폰세",
        "플라허티",
        "하이데이",
        "드뤼",
        "윌커슨",
        "켈리",
        "반헤켄",
        "라우덴밀크",
        "울리시스",
        "멘데스",
        "가르시아",
        "반레",
        "도미",
        "바비",
        "키쇼",
        "이구치",
        "미야자키",
        "코야마",
        "리처드",
        "캐슬",
        "에스피노",
    }

    # 1. 외국인 선수명 우선 검색 (조사 제거 전에 긴 이름부터 매칭)
    # 긴 이름부터 검색하여 부분 매칭 문제 방지
    foreign_players_sorted = sorted(foreign_players, key=len, reverse=True)
    for player in foreign_players_sorted:
        if player in query:
            return player

    # 2. 한국어 조사 제거 후 외국인 선수명 재검색
    query_normalized = query
    korean_particles = [
        "의",
        "가",
        "은",
        "는",
        "을",
        "를",
        "이",
        "에",
        "에서",
        "로",
        "으로",
        "와",
        "과",
    ]
    for particle in korean_particles:
        query_normalized = query_normalized.replace(particle, "")

    for player in foreign_players_sorted:
        if player in query_normalized:
            return player

    # 3. 한글 이름 패턴 검색 (2-4글자 한글) - 조사가 붙은 경우 고려
    # 조사가 붙기 전의 순수한 이름 추출
    name_patterns = [
        r"([가-힣]{2,4})(?:의|가|은|는|을|를|이|에|에서|로|으로|와|과)",  # 조사가 붙은 이름
        r"([가-힣]{2,4})(?=\s|$|[^가-힣])",  # 조사가 없는 이름
    ]

    matches = []
    for pattern in name_patterns:
        found = re.findall(pattern, query)
        matches.extend(found)

    # 중복 제거하면서 순서 유지
    unique_matches = []
    for match in matches:
        if match not in unique_matches:
            unique_matches.append(match)

    # 팀명이나 통계 용어가 아닌 것 중 첫 번째를 선수명으로 간주
    for match in unique_matches:
        if match not in TEAM_MAPPING and match not in STAT_MAPPING:
            # 일반적인 야구 용어들도 제외
            common_terms = {
                "선수",
                "타자",
                "투수",
                "순위",
                "랭킹",
                "기록",
                "성적",
                "경기",
                "시즌",
                "이닝",
                "타율",
                "방어율",
                "최고",
                "최저",
                "가장",
                "제일",
                "상위",
                "하위",
                "리그",
                "야구",
                "올해",
                "작년",
                "금년",
                "시즌",
                "월드",
                "베이스",
                "볼넷",
                "평균",
                "통산",
                "개수",
                "몇개",
                "몇점",
                "몇승",
                "몇패",
                "시절",
                "때문",
                "대표내년",
                "재작년",
                "지난해",
                "어제",
                "오늘",
                "내일",
                "최근",
                "현재",
                "마지막",
                "최종",
                "결승",
                "우승",
                "준우승",
                "시합",
                "대결",
                "승리",
                "패배",
                "무승부",
                "연장",
                "취소",
                "종료",
                "시작",
                "우승팀",
                "챔피언",
                "등수",
                "순위표",
                "주요",
                "선수단",
                "로스터",
                "라인업",
                "명단",
                "구단",
                "감독",
                "코치",
                "홈런",
                "안타",
                "득점",
                "실점",
                "승률",
                "세이브",
                "타점",
                "출루율",
                "장타율",
                "도루",
                "삼진",
                "볼넷",
                "타수",
                "타석",
                "등판",
                "완투",
                "조건",
                "자격",
                "규정",
                "규칙",
                "제도",
                "방법",
                "방식",
                "기준",
                "정보",
                "내용",
                "결과",
                "현황",
                "상황",
                "전망",
                "예상",
                "분석",
                "소식",
                "뉴스",
                "발표",
                "공식",
                "확정",
                "변경",
                "연기",
                "어디",
                "누구",
                "언제",
                "어떻게",
                "왜",
                "무엇",
                "얼마",
                "몇",
                "어떤",
                "무슨",
                "포수",
                "내야수",
                "외야수",
                "유격수",
                "중견수",
                "좌익수",
                "우익수",
                "올스타",
                "신인왕",
                "골든글러브",
                "베스트",
                "엠브이피",
                "타이틀",
                "트레이드",
                "이적",
                "영입",
                "방출",
                "등록",
                "말소",
                "계약",
                "연봉",
                "알려줘",
                "설명해줘",
                "보여줘",
                "부탁해",
                "어딨어",
                "누구야",
                "작년",
                "올해",
                "재작년",
                "내년",
                "순위",
                "성적",
                "기록",
                "랭킹",
                "정규시즌",
                "승률",
                "몇승",
                "몇패",
                "방어율",
                "평균자책",
                "평균자책점",
                "타율",
                "홈런",
                "타점",
                "도루",
                "포스트시즌",
                "준플레이오프",
                "플레이오프",
                "한국시리즈",
                "와일드카드",
                "누구",
                "어디",
                "언제",
                "어떻게",
                "얼마나",
                "어떠니",
                "궁금해",
                "그려줘",
                "표",
                "상대",
                "특정",
                "결과",
                "대결",
                "승부",
                "위가",
                "경기",
            }
            if match not in common_terms:
                return match

    return None


def normalize_player_name(name: str) -> str:
    """선수명을 정규화합니다. (공백 제거, 소문자 변환 등)"""
    if not name:
        return ""

    # 한국어 조사 제거
    korean_particles = [
        "의",
        "가",
        "은",
        "는",
        "을",
        "를",
        "이",
        "에",
        "에서",
        "로",
        "으로",
        "와",
        "과",
    ]
    normalized = name
    for particle in korean_particles:
        normalized = normalized.replace(particle, "")

    # 공백 제거 및 소문자 변환 (외국인 선수명 고려)
    normalized = re.sub(r"\s+", "", normalized).strip()

    return normalized


def calculate_name_similarity(name1: str, name2: str) -> float:
    """두 선수명 간의 유사도를 계산합니다. (0.0 ~ 1.0)"""
    if not name1 or not name2:
        return 0.0

    norm_name1 = normalize_player_name(name1).lower()
    norm_name2 = normalize_player_name(name2).lower()

    if norm_name1 == norm_name2:
        return 1.0

    # SequenceMatcher를 사용한 유사도 계산
    similarity = SequenceMatcher(None, norm_name1, norm_name2).ratio()

    # 부분 문자열 매칭 보너스
    if norm_name1 in norm_name2 or norm_name2 in norm_name1:
        similarity = max(similarity, 0.8)

    # 한글 이름의 경우 2글자 이상 일치하면 높은 점수
    if len(norm_name1) >= 2 and len(norm_name2) >= 2:
        if norm_name1[:2] == norm_name2[:2]:  # 성이 같은 경우
            similarity = max(similarity, 0.7)

    return similarity


def find_similar_player_names(
    query_name: str, known_players: List[str], threshold: float = 0.6
) -> List[Tuple[str, float]]:
    """
    주어진 선수명과 유사한 선수들을 찾습니다.

    Args:
        query_name: 검색할 선수명
        known_players: 알려진 선수명 목록
        threshold: 유사도 임계값 (기본값: 0.6)

    Returns:
        (선수명, 유사도) 튜플 리스트 (유사도 높은 순으로 정렬)
    """
    if not query_name or not known_players:
        return []

    similarities = []
    for player in known_players:
        similarity = calculate_name_similarity(query_name, player)
        if similarity >= threshold:
            similarities.append((player, similarity))

    # 유사도 높은 순으로 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities


def extract_player_name_with_fuzzy_matching(
    query: str, known_players: Optional[List[str]] = None
) -> Optional[str]:
    """
    퍼지 매칭을 활용한 고급 선수명 추출.

    Args:
        query: 검색 쿼리
        known_players: 알려진 선수명 목록 (없으면 기본 추출만 수행)

    Returns:
        추출된 선수명 (가장 유사도가 높은 것)
    """
    # 1. 기본 선수명 추출
    extracted_name = extract_player_name(query)
    if not extracted_name:
        return None

    # 2. 알려진 선수 목록이 없으면 기본 추출 결과 반환
    if not known_players:
        return extracted_name

    # 3. 정확히 일치하는 선수가 있으면 반환
    if extracted_name in known_players:
        return extracted_name

    # 4. 유사한 선수명 찾기
    similar_players = find_similar_player_names(
        extracted_name, known_players, threshold=0.7
    )

    if similar_players:
        best_match, similarity = similar_players[0]
        logger.info(
            f"[PlayerNameMatcher] '{extracted_name}' -> '{best_match}' (similarity: {similarity:.3f})"
        )
        return best_match

    # 5. 유사한 선수가 없으면 원본 반환
    return extracted_name


def is_ranking_query(query: str) -> bool:
    """랭킹/순위 관련 질문인지 판단합니다."""
    ranking_keywords = [
        "순위",
        "랭킹",
        "상위",
        "하위",
        "1위",
        "2위",
        "3위",
        "4위",
        "5위",
        "최고",
        "최저",
        "가장",
        "제일",
        "톱",
        "TOP",
        "베스트",
        "worst",
        "명",
        "리스트",
        "목록",
    ]
    return any(keyword in query for keyword in ranking_keywords)


def extract_ranking_count(query: str) -> Optional[int]:
    """순위 질문에서 요청된 개수를 추출합니다."""
    # "상위 5명", "10위까지", "톱 3" 등의 패턴
    count_patterns = [
        r"상위\s*(\d+)",
        r"(\d+)명",
        r"(\d+)위까지",
        r"톱\s*(\d+)",
        r"TOP\s*(\d+)",
        r"베스트\s*(\d+)",
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
    entity_filter.position_code = extract_position_code(query)
    entity_filter.league_type = extract_league_type(query)
    entity_filter.award_type = extract_award_type(query)
    entity_filter.movement_type = extract_movement_type(query)
    entity_filter.game_date = extract_game_date(query)

    # 로깅
    logger.info(
        f"[EntityExtractor] Extracted entities: "
        f"year={entity_filter.season_year}, "
        f"team={entity_filter.team_id}, "
        f"player={entity_filter.player_name}, "
        f"stat={entity_filter.stat_type}, "
        f"position={entity_filter.position_type}, "
        f"league={entity_filter.league_type}, "
        f"award={entity_filter.award_type}, "
        f"movement={entity_filter.movement_type}, "
        f"game_date={entity_filter.game_date}, "
        f"position_code={entity_filter.position_code}"
    )

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

    # 리그 타입 필터 (특정 리그가 감지된 경우에만 적용)
    if entity_filter.league_type:
        filters["meta.league"] = entity_filter.league_type

    # 수상 유형 필터
    if entity_filter.award_type and entity_filter.award_type != "any":
        filters["meta.award_type"] = entity_filter.award_type
        # 수상 관련 질문은 awards 테이블에서 검색
        filters["source_table"] = "awards"

    # 선수 이동 유형 필터
    if entity_filter.movement_type and entity_filter.movement_type != "any":
        filters["meta.movement_type"] = entity_filter.movement_type
        # 이동 관련 질문은 player_movements 테이블에서 검색
        filters["source_table"] = "player_movements"

    # 경기 날짜 필터
    if entity_filter.game_date:
        filters["meta.game_date"] = entity_filter.game_date

    # 포지션별 테이블 필터링 (다른 필터가 없을 때만)
    if "source_table" not in filters:
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
