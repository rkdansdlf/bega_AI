"""
야구 통계 전문 에이전트입니다.

이 에이전트는 LLM의 환각을 방지하기 위해 모든 통계 질문에 대해 
반드시 실제 DB를 조회한 결과만 사용합니다.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union
from psycopg2.extensions import connection as PgConnection
from decimal import Decimal
from datetime import date, datetime

from ..tools.database_query import DatabaseQueryTool
from ..tools.regulation_query import RegulationQueryTool
from ..tools.game_query import GameQueryTool
from ..tools.document_query import DocumentQueryTool
from ..core.tools.datetime_tool import get_current_datetime, get_baseball_season_info # 신규 도구 임포트
from .tool_caller import ToolCaller, ToolCall, ToolResult

logger = logging.getLogger(__name__)

from ..core.prompts import SYSTEM_PROMPT # SYSTEM_PROMPT 임포트
from ..core.entity_extractor import extract_entities_from_query # 엔티티 추출 임포트

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, Decimal):  
            return float(obj)
        return super().default(obj)
    
def clean_json_response(response: str) -> str:
    """LLM 응답에서 순수 JSON 추출 및 정제"""
    # 코드 블록 제거
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    
    # 주석 제거 (// 와 /* */)
    response = re.sub(r'//.*?$', '', response, flags=re.MULTILINE)
    response = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)
    
    # 후행 쉼표 제거
    response = re.sub(r',(\s*[}\]])', r'\1', response)
    
    return response.strip()

TEAM_CODE_TO_NAME = {
    "KIA": "KIA 타이거즈", "기아": "KIA 타이거즈",
    "LG": "LG 트윈스",
    "SSG": "SSG 랜더스",
    "NC": "NC 다이노스",
    "두산": "두산 베어스",
    "KT": "KT 위즈",
    "롯데": "롯데 자이언츠",
    "삼성": "삼성 라이온즈",
    "한화": "한화 이글스",
    "키움": "키움 히어로즈",
    "키움": "키움 히어로즈",
}

def _replace_team_codes(data: Any) -> Any:
    """Recursively replace team codes with full names in a data structure."""
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            # 키에 'team'이 포함되어 있고, 값이 코드 사전에 있는 경우 변환
            if 'team' in k and isinstance(v, str) and v in TEAM_CODE_TO_NAME:
                new_dict[k] = TEAM_CODE_TO_NAME[v]
            else:
                new_dict[k] = _replace_team_codes(v)
        return new_dict
    elif isinstance(data, list):
        return [_replace_team_codes(item) for item in data]
    else:
        # 값 자체가 팀 코드인 경우도 변환 (예: winning_team: "LG")
        if isinstance(data, str) and data in TEAM_CODE_TO_NAME:
            return TEAM_CODE_TO_NAME[data]
        return data

class BaseballStatisticsAgent:
    """
    야구 통계 전문 에이전트
    
    이 에이전트는 다음 원칙을 따릅니다:
    1. 모든 통계 질문은 반드시 실제 DB 도구를 통해 조회
    2. 도구 결과가 없으면 "데이터 없음"으로 명확히 응답  
    3. LLM 지식 기반 추측 절대 금지
    4. 검증된 데이터만 사용하여 답변 생성
    """
    
    def __init__(self, connection: PgConnection, llm_generator):
        self.connection = connection
        self.llm_generator = llm_generator
        self.db_query_tool = DatabaseQueryTool(connection)
        self.regulation_query_tool = RegulationQueryTool(connection)
        self.game_query_tool = GameQueryTool(connection)
        self.document_query_tool = DocumentQueryTool(connection) # 신규 도구 인스턴스 생성
        self.tool_caller = ToolCaller()
        
        # 팀명 매핑 캐시 초기화
        self._team_name_cache = None
        
        # 등록 가능한 도구들
        self._register_tools()
        
    def _register_tools(self):
        """사용 가능한 도구들을 등록합니다."""
        
        # 선수 개별 통계 조회 도구
        self.tool_caller.register_tool(
            "get_player_stats",
            "선수의 개별 시즌 통계를 실제 DB에서 조회합니다. 타율, 홈런, ERA 등 개인 기록 질문에 사용하세요.",
            {
                "player_name": "선수명 (부분 매칭 가능)",
                "year": "시즌 년도", 
                "position": "batting(타자), pitching(투수), 또는 both(둘다, 기본값)"
            },
            self._tool_get_player_stats
        )
        
        # 선수 통산 통계 조회 도구  
        self.tool_caller.register_tool(
            "get_career_stats",
            "선수의 통산(커리어) 통계를 실제 DB에서 조회합니다. '통산', '총', '전체' 기록 질문에 사용하세요.",
            {
                "player_name": "선수명 (부분 매칭 가능)",
                "position": "batting(타자), pitching(투수), 또는 both(둘다, 기본값)"
            },
            self._tool_get_career_stats
        )
        
        # 리더보드/순위 조회 도구
        self.tool_caller.register_tool(
            "get_leaderboard", 
            "특정 통계 지표의 순위/리더보드를 실제 DB에서 조회합니다. '최고', '상위', '1위' 등의 질문에 사용하세요.",
            {
                "stat_name": "통계 지표명 (ops, era, home_runs, 타율, 홈런 등)",
                "year": "시즌 년도",
                "position": "batting(타자) 또는 pitching(투수)", 
                "team_filter": "특정 팀만 조회 (선택적, 예: KIA, LG)",
                "limit": "상위 몇 명까지 (선택적, 기본 10명)"
            },
            self._tool_get_leaderboard
        )
        
        # 선수 존재 여부 확인 도구
        self.tool_caller.register_tool(
            "validate_player",
            "선수가 해당 연도에 실제로 기록이 있는지 DB에서 확인합니다. 선수명 오타나 존재하지 않는 선수 질문 시 사용하세요.",
            {
                "player_name": "선수명",
                "year": "시즌 년도 (기본값: current_year = {current_year})"
            },
            self._tool_validate_player
        )
        
        # 팀 요약 정보 조회 도구
        self.tool_caller.register_tool(
            "get_team_summary",
            "특정 팀의 주요 선수들과 팀 통계를 실제 DB에서 조회합니다. 팀 관련 질문에 사용하세요.",
            {
                "team_name": "팀명 (KIA, LG, 두산 등)",
                "year": "시즌 년도"
            },
            self._tool_get_team_summary
        )
        
        # 포지션 정보 조회 도구
        self.tool_caller.register_tool(
            "get_position_info",
            "포지션 약어를 전체 포지션명으로 변환합니다. 포지션 관련 질문에 사용하세요.",
            {
                "position_abbr": "포지션 약어 (지, 타, 주, 중, 좌, 우, 一, 二, 三, 유, 포)"
            },
            self._tool_get_position_info
        )
        
        # 팀 기본 정보 조회 도구
        self.tool_caller.register_tool(
            "get_team_basic_info",
            "팀의 기본 정보를 조회합니다. 홈구장, 마스코트, 창단연도 등의 질문에 사용하세요.",
            {
                "team_name": "팀명 (KIA, LG, 두산 등)"
            },
            self._tool_get_team_basic_info
        )
        
        # 수비 통계 조회 도구
        self.tool_caller.register_tool(
            "get_defensive_stats",
            "선수의 수비 통계를 조회합니다. 수비율, 오류, 어시스트 등의 질문에 사용하세요.",
            {
                "player_name": "선수명",
                "year": "시즌 년도 (선택적, 생략하면 통산)"
            },
            self._tool_get_defensive_stats
        )
        
        # 구속 데이터 조회 도구
        self.tool_caller.register_tool(
            "get_velocity_data",
            "투수의 구속 데이터를 조회합니다. 직구, 변화구 구속 등의 질문에 사용하세요.",
            {
                "player_name": "선수명",
                "year": "시즌 년도 (선택적, 생략하면 최근 데이터)"
            },
            self._tool_get_velocity_data
        )
        
        # KBO 규정 검색 도구
        self.tool_caller.register_tool(
            "search_regulations",
            "KBO 규정을 검색합니다. 야구 규칙, 제도, 판정 기준 등의 질문에 사용하세요.",
            {
                "query": "검색할 규정 내용 (예: 타이브레이크, FA 조건, 인필드 플라이)"
            },
            self._tool_search_regulations
        )
        
        # 규정 카테고리별 조회 도구
        self.tool_caller.register_tool(
            "get_regulations_by_category",
            "특정 카테고리의 규정들을 조회합니다. 체계적인 규정 설명이 필요할 때 사용하세요.",
            {
                "category": "규정 카테고리 (basic, player, game, technical, discipline, postseason, special, terms)"
            },
            self._tool_get_regulations_by_category
        )
        
        # 경기 박스스코어 조회 도구
        self.tool_caller.register_tool(
            "get_game_box_score",
            "특정 경기의 박스스코어와 상세 정보를 조회합니다. 경기 결과, 이닝별 득점 등의 질문에 사용하세요.",
            {
                "game_id": "경기 고유 ID (선택적)",
                "date": "경기 날짜 (YYYY-MM-DD, 선택적)",
                "home_team": "홈팀명 (선택적)",
                "away_team": "원정팀명 (선택적)"
            },
            self._tool_get_game_box_score
        )
        
        # 날짜별 경기 조회 도구
        self.tool_caller.register_tool(
            "get_games_by_date",
            "특정 날짜의 모든 경기를 조회합니다. '오늘 경기', '어제 경기' 등의 질문에 사용하세요.",
            {
                "date": "경기 날짜 (YYYY-MM-DD)"
            },
            self._tool_get_games_by_date
        )
        
        # 팀 간 직접 대결 조회 도구
        self.tool_caller.register_tool(
            "get_head_to_head",
            "두 팀 간의 직접 대결 기록을 조회합니다. 맞대결 성적, 승부 현황 등의 질문에 사용하세요.",
            {
                "team1": "팀1 이름",
                "team2": "팀2 이름",
                "year": "시즌 년도 (선택적)",
                "limit": "최근 몇 경기까지 (선택적, 기본 10경기)"
            },
            self._tool_get_head_to_head
        )
        
        # 선수 경기 성적 조회 도구
        self.tool_caller.register_tool(
            "get_player_game_performance",
            "특정 선수의 개별 경기 성적을 조회합니다. 특정 경기에서의 선수 활약 등의 질문에 사용하세요.",
            {
                "player_name": "선수명",
                "date": "경기 날짜 (선택적)",
                "recent_games": "최근 몇 경기까지 (선택적, 기본 5경기)"
            },
            self._tool_get_player_game_performance
        )
        
        # 선수 비교 도구
        self.tool_caller.register_tool(
            "compare_players",
            "두 선수의 통계를 비교 분석합니다. 'A vs B', 'A와 B 중 누가' 등 선수 비교 질문에 사용하세요.",
            {
                "player1": "첫 번째 선수명",
                "player2": "두 번째 선수명",
                "comparison_type": "career(통산 비교, 기본값) 또는 season(특정 시즌 비교)",
                "year": "특정 시즌 비교 시 연도 (선택적)",
                "position": "batting(타자), pitching(투수), 또는 both(둘다, 기본값)"
            },
            self._tool_compare_players
        )

        # 시즌 마지막 경기 정보 조회 도구 (통합 버전)
        self.tool_caller.register_tool(
            "get_season_final_game_date",
            "특정 시즌의 마지막 경기 날짜와 경기 결과를 한 번에 조회합니다. '마지막 경기', '최종전', '한국시리즈 마지막', '작년 마지막 경기 결과' 등의 질문에 사용하세요.",
            {
                "year": "시즌 년도",
                "league_type": "'regular_season'(정규시즌) 또는 'korean_series'(한국시리즈, 기본값)"
            },
            self._tool_get_season_final_game_date
        )
        
        # 팀 순위 조회 도구
        self.tool_caller.register_tool(
            "get_team_rank",
            "특정 시즌의 팀 최종 순위를 조회합니다. '몇 등', '순위', '시즌 마무리' 등의 질문에 사용하세요.",
            {
                "team_name": "팀명 (예: 'KIA', '기아', 'SSG')",
                "year": "시즌 년도"
            },
            self._tool_get_team_rank
        )
        
        # 지능적 팀별 마지막 경기 조회 도구
        self.tool_caller.register_tool(
            "get_team_last_game",
            "특정 팀의 실제 마지막 경기를 지능적으로 조회합니다. 팀 순위를 확인하여 포스트시즌 진출팀(1-5위)은 한국시리즈, 미진출팀(6-10위)은 정규시즌 마지막 경기를 자동으로 찾습니다.",
            {
                "team_name": "팀명 (예: 'SSG', '기아', 'KIA')",
                "year": "시즌 년도"
            },
            self._tool_get_team_last_game
        )
        
        # 한국시리즈 우승팀 조회 도구
        self.tool_caller.register_tool(
            "get_korean_series_winner",
            """특정 시즌의 한국시리즈 우승팀을 조회합니다.

다음 질문에서 반드시 사용하세요:
- '우승팀', '챔피언', '한국시리즈 우승' 등의 질문
- '작년 우승팀', '지난해 챔피언' (시간 확인 후 연도 변환하여 호출)
- 'X년 우승팀은?' 

이 도구는 한국시리즈 마지막 경기의 승리팀을 자동으로 식별하여 우승팀을 판단합니다.""",
            {
                "year": "시즌 년도"
            },
            self._tool_get_korean_series_winner
        )
        
        # 현재 날짜 및 시간 조회 도구
        self.tool_caller.register_tool(
            "get_current_datetime",
            """현재 날짜와 시간을 한국 시간 기준으로 조회합니다.

다음 상황에서 반드시 이 도구를 먼저 사용해야 합니다:
- '작년', '지난해', '올해', '금년' 등 상대적 시간 표현이 포함된 질문
- '오늘', '지금' 등 현재 시점 기준 질문
- 우승팀, 시즌 기록 등에서 정확한 연도가 필요한 질문
- 예: '작년 우승팀은?' → 먼저 현재 날짜 확인하여 '작년' = '2024년'임을 파악

중요: 상대적 시간 표현이 있으면 절대적 연도로 변환하기 위해 반드시 이 도구를 호출하세요.""",
            {},
            self._tool_get_current_datetime
        )

        self.tool_caller.register_tool(
            "get_baseball_season_info",
            "현재 KBO 야구 시즌 정보 조회합니다. '지금 야구 시즌이야?', '현재 시즌 상태는?' 등의 질문에 사용하세요.",
            {},
            self._tool_get_baseball_season_info
        )

        # 문서 검색 도구 (신규 추가)
        self.tool_caller.register_tool(
            "search_documents",
            "KBO 리그 규정, 용어 정의, 선수 관련 스토리 등 비정형 텍스트 문서를 검색합니다. 'ABS가 뭐야?', 'FA 규정 알려줘'와 같은 설명형/정의형 질문에 사용하세요.",
            {
                "query": "검색할 질문 또는 키워드",
                "limit": "반환할 최대 결과 수 (선택적, 기본값 10)"
            },
            self._tool_search_documents
        )
    
    def _tool_search_documents(self, query: str, limit: int = 10) -> ToolResult:
        """문서 검색 도구의 래퍼 함수"""
        try:
            result = self.document_query_tool.search_documents(query, limit)
            
            if result.get("error"):
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"문서 검색 오류: {result['error']}"
                )
            
            if not result.get("found"):
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"'{query}'와 관련된 문서를 찾을 수 없습니다."
                )
            
            # 답변 생성에 용이하도록 컨텍스트 포맷팅
            formatted_docs = []
            for doc in result["documents"]:
                title = doc.get('title', '정보 조각')
                content = doc.get('content', '')
                formatted_docs.append(f"문서명: {title}\n내용: {content}")
            
            return ToolResult(
                success=True,
                data={"documents": formatted_docs},
                message=f"'{query}' 관련 문서를 {len(formatted_docs)}개 찾았습니다."
            )
            
        except Exception as e:
            logger.error(f"[BaseballAgent] Document search tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"문서 검색 도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_current_datetime(self, **kwargs) -> ToolResult:
        """현재 날짜 및 시간 조회 도구"""
        try:
            datetime_info = get_current_datetime()
            return ToolResult(
                success=True,
                data=datetime_info,
                message=f"현재 시간은 {datetime_info['formatted_date']} {datetime_info['formatted_time']}입니다."
            )
        except Exception as e:
            logger.error(f"Current datetime tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"현재 시간 조회 중 오류 발생: {e}"
            )

    def _tool_get_baseball_season_info(self, **kwargs) -> ToolResult:
        """현재 야구 시즌 정보 조회 도구"""
        try:
            season_info = get_baseball_season_info()
            return ToolResult(
                success=True,
                data=season_info,
                message=f"현재 {season_info['current_year']}년 야구 시즌은 '{season_info['season_status']}' 상태입니다."
            )
        except Exception as e:
            logger.error(f"Baseball season info tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"야구 시즌 정보 조회 중 오류 발생: {e}"
            )

    def _load_team_name_mapping(self) -> Dict[str, str]:
        """팀 ID와 팀명 매핑을 데이터베이스에서 로드합니다."""
        if self._team_name_cache is not None:
            return self._team_name_cache
            
        try:
            with self.connection.cursor() as cursor:
                # 현재 활성화된 팀들의 매핑 정보를 조회 (가장 적절한 팀명 선택)
                cursor.execute("""
                    SELECT team_id, full_name 
                    FROM team_name_mapping 
                    WHERE full_name IN (
                        '한화 이글스', 'KIA 타이거즈', 'KT 위즈', 'LG 트윈스', 
                        '롯데 자이언츠', 'NC 다이노스', '두산 베어스', 'SSG 랜더스', 
                        '삼성 라이언즈', '키움 히어로즈'
                    )
                    ORDER BY team_id, full_name
                """)
                
                mapping = {}
                for team_id, full_name in cursor.fetchall():
                    if team_id not in mapping:  # 각 team_id당 첫 번째 매핑만 사용
                        mapping[team_id] = full_name
                
                # 누락된 team_id에 대한 폴백 매핑
                fallback_mapping = {
                    'HH': '한화 이글스',
                    'HT': 'KIA 타이거즈', 
                    'KT': 'KT 위즈',
                    'LG': 'LG 트윈스',
                    'LT': '롯데 자이언츠',
                    'NC': 'NC 다이노스',
                    'OB': '두산 베어스',
                    'SK': 'SSG 랜더스',
                    'SS': '삼성 라이언즈',
                    'WO': '키움 히어로즈'
                }
                
                # 누락된 매핑 보완
                for team_id, team_name in fallback_mapping.items():
                    if team_id not in mapping:
                        mapping[team_id] = team_name
                
                self._team_name_cache = mapping
                logger.info(f"[BaseballAgent] Loaded team mappings: {mapping}")
                return mapping
                
        except Exception as e:
            logger.error(f"[BaseballAgent] Failed to load team mappings: {e}")
            # 에러 시 기본 매핑 사용
            fallback_mapping = {
                'HH': '한화 이글스', 'HT': 'KIA 타이거즈', 'KT': 'KT 위즈',
                'LG': 'LG 트윈스', 'LT': '롯데 자이언츠', 'NC': 'NC 다이노스',
                'OB': '두산 베어스', 'SK': 'SSG 랜더스', 'SS': '삼성 라이언즈',
                'WO': '키움 히어로즈'
            }
            self._team_name_cache = fallback_mapping
            return fallback_mapping
    
    def _convert_team_id_to_name(self, team_id: str) -> str:
        """팀 ID를 팀명으로 변환합니다."""
        if not team_id:
            return team_id
            
        mapping = self._load_team_name_mapping()
        return mapping.get(team_id, team_id)  # 매핑이 없으면 원래 ID 반환
    
    def _format_game_info_with_team_names(self, game_info: Dict[str, Any]) -> Dict[str, Any]:
        """게임 정보에서 팀 ID를 팀명으로 변환하여 포맷팅합니다."""
        formatted = game_info.copy()
        
        if 'home_team' in formatted:
            formatted['home_team_name'] = self._convert_team_id_to_name(formatted['home_team'])
        if 'away_team' in formatted:
            formatted['away_team_name'] = self._convert_team_id_to_name(formatted['away_team'])
            
        return formatted
    
    def _format_league_type_to_korean(self, league_type: str) -> str:
        """리그 타입을 한국어로 변환합니다."""
        league_mapping = {
            'korean_series': '한국시리즈',
            'regular_season': '정규시즌', 
            'postseason': '포스트시즌',
            'wild_card': '와일드카드',
            'semi_playoff': '준플레이오프',
            'playoff': '플레이오프'
        }
        return league_mapping.get(league_type, league_type)
    
    def _format_game_status_to_korean(self, status: str) -> str:
        """게임 상태를 한국어로 변환하거나 불필요한 상태는 제거합니다."""
        # COMPLETED 같은 과거 경기 상태는 표시하지 않음 (이미 지난 경기이므로)
        status_mapping = {
            'COMPLETED': '',  # 완료된 경기는 상태 표시하지 않음
            'SCHEDULED': '예정',
            'LIVE': '진행 중',
            'CANCELLED': '취소됨',
            'POSTPONED': '연기됨'
        }
        formatted_status = status_mapping.get(status, status)
        return formatted_status if formatted_status else None
    
    def _format_stadium_name(self, stadium: str) -> str:
        """경기장명을 사용자 친화적으로 포맷팅합니다."""
        if not stadium:
            return stadium
        
        # 경기장명 정규화
        stadium_mapping = {
            '광주': '광주-기아 챔피언스 필드',
            '잠실': '잠실야구장',
            '문학': '인천 SSG 랜더스필드',
            '대구': '대구 삼성 라이온즈 파크',
            '창원': '창원 NC 파크',
            '수원': '수원 KT 위즈 파크',
            '고척': '고척 스카이돔',
            '사직': '사직야구장'
        }
        
        return stadium_mapping.get(stadium, stadium)
        
    def _tool_get_season_final_game_date(self, year: int, league_type: str = 'korean_series') -> ToolResult:
        """시즌 마지막 경기 정보 조회 도구 (날짜 + 경기 결과)"""
        try:
            # 1단계: 마지막 경기 날짜 조회
            date_result = self.game_query_tool.get_season_final_game_date(year, league_type)
            
            if date_result["error"]:
                return ToolResult(
                    success=False,
                    data=date_result,
                    message=f"마지막 경기 날짜 조회 오류: {date_result['error']}"
                )
            
            if not date_result["found"]:
                return ToolResult(
                    success=False,
                    data=date_result,
                    message=f"{year}년 {league_type}의 마지막 경기 날짜를 찾을 수 없습니다."
                )
            
            final_date = date_result['final_game_date']
            
            # 2단계: 해당 날짜의 경기 결과 조회
            games_result = self.game_query_tool.get_games_by_date(final_date)
            
            combined_result = {
                "year": year,
                "league_type": league_type,
                "final_date": final_date,
                "games": games_result.get("games", []),
                "total_games": games_result.get("total_games", 0)
            }
            
            if games_result.get("found") and games_result.get("games"):
                game_info = []
                formatted_games = []
                
                # 리그 타입을 한국어로 변환
                league_name_korean = self._format_league_type_to_korean(league_type)
                
                for game in games_result["games"]:
                    # 팀명 매핑 적용
                    formatted_game = self._format_game_info_with_team_names(game)
                    formatted_games.append(formatted_game)
                    
                    # 팀명을 사용한 게임 정보 생성
                    away_name = formatted_game.get('away_team_name', formatted_game['away_team'])
                    home_name = formatted_game.get('home_team_name', formatted_game['home_team'])
                    
                    # 경기장명 포맷팅
                    stadium_name = self._format_stadium_name(game.get('stadium', ''))
                    
                    # 게임 상태 포맷팅 (COMPLETED는 표시하지 않음)
                    game_status = self._format_game_status_to_korean(game.get('game_status', ''))
                    
                    # 기본 경기 정보
                    game_desc = f"{away_name} {game['away_score']}-{game['home_score']} {home_name}"
                    
                    # 경기장 정보 추가
                    if stadium_name:
                        game_desc += f" ({stadium_name})"
                    
                    # 상태 정보 추가 (COMPLETED가 아닌 경우에만)
                    if game_status:
                        game_desc += f" - {game_status}"
                    
                    game_info.append(game_desc)
                
                # combined_result에도 formatted_games 추가
                combined_result["formatted_games"] = formatted_games
                
                message = f"{year}년 {league_name_korean} 마지막 경기 ({final_date}):\n" + "\n".join(game_info)
            else:
                league_name_korean = self._format_league_type_to_korean(league_type)
                message = f"{year}년 {league_name_korean}의 마지막 경기 날짜는 {final_date}이지만 경기 상세 정보를 찾을 수 없습니다."
            
            return ToolResult(
                success=True,
                data=combined_result,
                message=message
            )
            
        except Exception as e:
            logger.error(f"Final game tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_player_stats(self, player_name: str, year: int, position: str = "both") -> ToolResult:
        """선수 개별 통계 조회 도구"""
        try:
            result = self.db_query_tool.get_player_season_stats(player_name, year, position)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"DB 조회 오류: {result['error']}"
                )
            
            if not result["found"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"{year}년 '{player_name}' 선수의 기록을 찾을 수 없습니다. 선수명 확인이나 다른 연도를 시도해보세요."
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"{year}년 {player_name} 선수 통계를 성공적으로 조회했습니다."
            )
            
        except Exception as e:
            logger.error(f"Player stats tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )
    
    def _tool_get_career_stats(self, player_name: str, position: str = "both") -> ToolResult:
        """선수 통산 통계 조회 도구"""
        try:
            result = self.db_query_tool.get_player_career_stats(player_name, position)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"DB 조회 오류: {result['error']}"
                )
            
            if not result["found"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"'{player_name}' 선수의 통산 기록을 찾을 수 없습니다. 선수명을 확인해주세요."
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"{player_name} 선수 통산 통계를 성공적으로 조회했습니다."
            )
            
        except Exception as e:
            logger.error(f"Career stats tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )
    
    def _tool_get_leaderboard(
        self, 
        stat_name: str, 
        year: int, 
        position: str, 
        team_filter: str = None, 
        limit: int = 10
    ) -> ToolResult:
        """리더보드 조회 도구"""
        try:
            result = self.db_query_tool.get_team_leaderboard(
                stat_name, year, position, team_filter, limit
            )
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"DB 조회 오류: {result['error']}"
                )
            
            if not result["found"] or not result["leaderboard"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"{year}년 {stat_name} {position} 리더보드 데이터를 찾을 수 없습니다."
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"{year}년 {stat_name} {position} 리더보드를 성공적으로 조회했습니다 (총 {len(result['leaderboard'])}명)."
            )
            
        except Exception as e:
            logger.error(f"Leaderboard tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )
    
    def _tool_validate_player(self, player_name: str, year: int = None) -> ToolResult:
        """선수 존재 여부 확인 도구"""
        try:
            if year is None:
                import datetime as dt
                year = dt.datetime.now().year
            result = self.db_query_tool.validate_player_exists(player_name, year)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"DB 조회 오류: {result['error']}"
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"선수 검색 완료: {len(result['found_players'])}명의 유사한 선수를 찾았습니다." if result["exists"] else "해당 선수를 찾을 수 없습니다."
            )
            
        except Exception as e:
            logger.error(f"Player validation tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )
    
    def _tool_get_team_summary(self, team_name: str, year: int) -> ToolResult:
        """팀 요약 정보 조회 도구"""
        try:
            result = self.db_query_tool.get_team_summary(team_name, year)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"DB 조회 오류: {result['error']}"
                )
            
            if not result["found"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"{year}년 {team_name} 팀 데이터를 찾을 수 없습니다."
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"{year}년 {team_name} 팀 주요 선수 정보를 성공적으로 조회했습니다."
            )
            
        except Exception as e:
            logger.error(f"Team summary tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_position_info(self, position_abbr: str) -> ToolResult:
        """포지션 정보 조회 도구"""
        try:
            result = self.db_query_tool.get_position_info(position_abbr)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"포지션 조회 오류: {result['error']}"
                )
            
            if not result["found"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"'{position_abbr}' 포지션 약어를 찾을 수 없습니다."
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"'{position_abbr}' 포지션 정보: {result['position_name']}"
            )
            
        except Exception as e:
            logger.error(f"Position info tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_team_basic_info(self, team_name: str) -> ToolResult:
        """팀 기본 정보 조회 도구"""
        try:
            result = self.db_query_tool.get_team_basic_info(team_name)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"팀 정보 조회 오류: {result['error']}"
                )
            
            if not result["found"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"'{team_name}' 팀의 기본 정보를 찾을 수 없습니다."
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"{result['team_name']} 팀 기본 정보를 성공적으로 조회했습니다."
            )
            
        except Exception as e:
            logger.error(f"Team basic info tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_defensive_stats(self, player_name: str, year: int = None) -> ToolResult:
        """선수 수비 통계 조회 도구"""
        try:
            result = self.db_query_tool.get_player_defensive_stats(player_name, year)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"수비 통계 조회 오류: {result['error']}"
                )
            
            if not result["found"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=result["message"]  # 데이터베이스에 없다는 메시지
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"{player_name} 선수 수비 통계를 성공적으로 조회했습니다."
            )
            
        except Exception as e:
            logger.error(f"Defensive stats tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_velocity_data(self, player_name: str, year: int = None) -> ToolResult:
        """투수 구속 데이터 조회 도구"""
        try:
            result = self.db_query_tool.get_pitcher_velocity_data(player_name, year)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"구속 데이터 조회 오류: {result['error']}"
                )
            
            if not result["found"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=result["message"]  # 데이터베이스에 없다는 메시지
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"{player_name} 선수 구속 데이터를 성공적으로 조회했습니다."
            )
            
        except Exception as e:
            logger.error(f"Velocity data tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_search_regulations(self, query: str) -> ToolResult:
        """KBO 규정 검색 도구"""
        try:
            result = self.regulation_query_tool.search_regulation(query)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"규정 검색 오류: {result['error']}"
                )
            
            if not result["found"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"'{query}' 관련 규정을 찾을 수 없습니다."
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"'{query}' 관련 규정을 {result['total_found']}개 찾았습니다."
            )
            
        except Exception as e:
            logger.error(f"Regulation search tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_regulations_by_category(self, category: str) -> ToolResult:
        """규정 카테고리별 조회 도구"""
        try:
            result = self.regulation_query_tool.get_regulation_by_category(category)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"카테고리 조회 오류: {result['error']}"
                )
            
            if not result["found"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"'{category}' 카테고리의 규정을 찾을 수 없습니다."
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"'{category}' 카테고리 규정을 {result['total_found']}개 조회했습니다."
            )
            
        except Exception as e:
            logger.error(f"Regulation category tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_game_box_score(
        self, 
        game_id: str = None, 
        date: str = None,
        home_team: str = None,
        away_team: str = None
    ) -> ToolResult:
        """경기 박스스코어 조회 도구"""
        try:
            result = self.game_query_tool.get_game_box_score(game_id, date, home_team, away_team)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"박스스코어 조회 오류: {result['error']}"
                )
            
            if not result["found"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message="조건에 맞는 경기를 찾을 수 없습니다. 날짜나 팀명을 확인해주세요."
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"{result['total_games']}개 경기의 박스스코어를 성공적으로 조회했습니다."
            )
            
        except Exception as e:
            logger.error(f"Game box score tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_games_by_date(self, date: str, **kwargs) -> ToolResult:
        """날짜별 경기 조회 도구"""
        try:
            team = kwargs.get('team', None)
            result = self.game_query_tool.get_games_by_date(date, team)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"날짜별 경기 조회 오류: {result['error']}"
                )
            
            if not result["found"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"{date}에 경기가 없습니다." + (f" ({team} 팀 포함)" if team else "")
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"{date}에 {result['total_games']}개 경기를 성공적으로 조회했습니다."
            )
            
        except Exception as e:
            logger.error(f"Games by date tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_head_to_head(
        self, 
        team1: str, 
        team2: str, 
        year: int = None,
        limit: int = 10
    ) -> ToolResult:
        """팀 간 직접 대결 조회 도구"""
        try:
            result = self.game_query_tool.get_head_to_head(team1, team2, year, limit)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"팀 간 대결 기록 조회 오류: {result['error']}"
                )
            
            if not result["found"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"{team1} vs {team2} 맞대결 기록을 찾을 수 없습니다." + (f" ({year}년)" if year else "")
                )
            
            summary = result["summary"]
            return ToolResult(
                success=True,
                data=result,
                message=f"{team1} vs {team2} 맞대결: {summary['total_games']}경기 (승부: {summary['team1_wins']}-{summary['team2_wins']})"
            )
            
        except Exception as e:
            logger.error(f"Head to head tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_player_game_performance(
        self, 
        player_name: str, 
        date: str = None,
        recent_games: int = 5
    ) -> ToolResult:
        """선수 경기 성적 조회 도구"""
        try:
            result = self.game_query_tool.get_player_game_performance(player_name, date, recent_games)
            
            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"선수 경기 성적 조회 오류: {result['error']}"
                )
            
            if not result["found"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=result.get("message", f"{player_name} 선수의 경기 성적을 찾을 수 없습니다.")
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"{player_name} 선수의 {result['total_games']}경기 성적을 성공적으로 조회했습니다."
            )
            
        except Exception as e:
            logger.error(f"Player game performance tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"도구 실행 중 오류 발생: {e}"
            )
    
    def _tool_compare_players(
        self, 
        player1: str, 
        player2: str, 
        comparison_type: str = "career",
        year: int = None,
        position: str = "both"
    ) -> ToolResult:
        """선수 비교 도구"""
        try:
            logger.info(f"[BaseballAgent] Comparing players: {player1} vs {player2} ({comparison_type})")
            
            # 두 선수의 통계를 모두 조회
            if comparison_type == "season" and year:
                # 특정 시즌 비교
                player1_result = self.db_query_tool.get_player_season_stats(player1, year, position)
                player2_result = self.db_query_tool.get_player_season_stats(player2, year, position)
                comparison_label = f"{year}년 시즌"
            else:
                # 통산 비교
                player1_result = self.db_query_tool.get_player_career_stats(player1, position)
                player2_result = self.db_query_tool.get_player_career_stats(player2, position)
                comparison_label = "통산"
            
            # 오류 처리
            if player1_result["error"] or player2_result["error"]:
                return ToolResult(
                    success=False,
                    data={
                        "player1_result": player1_result,
                        "player2_result": player2_result
                    },
                    message=f"데이터 조회 오류: {player1_result.get('error') or player2_result.get('error')}"
                )
            
            # 두 선수 중 하나라도 데이터가 없으면 실패
            if not player1_result["found"]:
                return ToolResult(
                    success=False,
                    data={
                        "player1_result": player1_result,
                        "player2_result": player2_result
                    },
                    message=f"{comparison_label} '{player1}' 선수의 기록을 찾을 수 없습니다."
                )
            
            if not player2_result["found"]:
                return ToolResult(
                    success=False,
                    data={
                        "player1_result": player1_result,
                        "player2_result": player2_result
                    },
                    message=f"{comparison_label} '{player2}' 선수의 기록을 찾을 수 없습니다."
                )
            
            # 비교 분석 데이터 구성
            comparison_data = {
                "comparison_type": comparison_label,
                "player1": {
                    "name": player1,
                    "data": player1_result
                },
                "player2": {
                    "name": player2,
                    "data": player2_result
                },
                "analysis": self._analyze_player_comparison(player1_result, player2_result, position)
            }
            
            return ToolResult(
                success=True,
                data=comparison_data,
                message=f"{player1} vs {player2} {comparison_label} 비교 분석 완료"
            )
            
        except Exception as e:
            logger.error(f"Player comparison tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"선수 비교 도구 실행 중 오류 발생: {e}"
            )
    
    def _tool_get_team_rank(self, team_name: str, year: int) -> ToolResult:
        """팀 순위 조회 도구"""
        try:
            with self.connection.cursor() as cursor:
                # 팀명을 team_id로 매핑
                from ..core.entity_extractor import TEAM_MAPPING
                team_id = None
                for variant, mapped_id in TEAM_MAPPING.items():
                    if variant in team_name:
                        team_id = mapped_id
                        break
                
                if not team_id:
                    team_id = team_name  # 직접 매핑 실패시 원본 사용
                
                # v_team_rank_all 뷰에서 팀 순위 조회 (올바른 컬럼명 사용)
                cursor.execute("""
                    SELECT season_rank, team_name
                    FROM v_team_rank_all 
                    WHERE (team_id = %s OR team_name LIKE %s) 
                    AND season_year = %s
                """, [team_id, f'%{team_name}%', year])
                
                result = cursor.fetchone()
                
                if result:
                    season_rank, full_team_name = result
                    return ToolResult(
                        success=True,
                        data={
                            "team_name": full_team_name,
                            "team_rank": season_rank,  # API 호환성을 위해 team_rank로 반환
                            "season_rank": season_rank,  # 실제 DB 컬럼명
                            "year": year,
                            "found": True
                        },
                        message=f"{full_team_name}의 {year}년 최종 순위: {season_rank}등"
                    )
                else:
                    return ToolResult(
                        success=False,
                        data={
                            "team_name": team_name,
                            "year": year,
                            "found": False
                        },
                        message=f"{team_name}의 {year}년 순위 정보를 찾을 수 없습니다"
                    )
                    
        except Exception as e:
            logger.error(f"[BaseballAgent] Team rank query error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"팀 순위 조회 중 오류 발생: {e}"
            )
    
    def _tool_get_team_last_game(self, team_name: str, year: int) -> ToolResult:
        """지능적 팀별 마지막 경기 조회 도구"""
        try:
            # 1단계: 팀 순위 조회로 포스트시즌 진출 여부 확인
            rank_result = self._tool_get_team_rank(team_name, year)
            
            if not rank_result.success:
                # 순위 정보를 찾을 수 없는 경우, 기본적으로 한국시리즈로 시도
                logger.warning(f"[BaseballAgent] 팀 순위를 찾을 수 없어 한국시리즈로 조회: {team_name}")
                league_type = "korean_series"
                team_rank = None
            else:
                team_rank = rank_result.data.get("team_rank")
                # 상위 5팀은 포스트시즌 진출, 6위 이하는 정규시즌에서 마무리
                league_type = "korean_series" if team_rank <= 5 else "regular_season"
            
            logger.info(f"[BaseballAgent] {team_name} {year}년 순위: {team_rank}, 리그 타입: {league_type}")
            
            # 2단계: 해당 리그 타입의 마지막 경기 조회
            final_game_result = self._tool_get_season_final_game_date(year, league_type)
            
            if not final_game_result.success:
                return ToolResult(
                    success=False,
                    data={
                        "team_name": team_name,
                        "year": year,
                        "team_rank": team_rank,
                        "league_type": league_type
                    },
                    message=f"{team_name}의 {year}년 마지막 경기를 찾을 수 없습니다"
                )
            
            # 3단계: 해당 팀의 경기만 필터링
            games = final_game_result.data.get("formatted_games", [])
            team_games = []
            
            # 팀명을 동적으로 매핑
            from ..core.entity_extractor import extract_team
            team_id = extract_team(team_name)
            
            if not team_id:
                team_id = team_name  # 매핑 실패시 원본 사용
            
            for game in games:
                if (game.get("home_team") == team_id or game.get("away_team") == team_id):
                    team_games.append(game)
            
            # 결과 구성
            combined_data = {
                "team_name": team_name,
                "year": year,
                "team_rank": team_rank,
                "league_type": league_type,
                "final_date": final_game_result.data.get("final_date"),
                "team_games": team_games,
                "all_games": games,
                "postseason_qualified": team_rank <= 5 if team_rank else None
            }
            
            # 메시지 생성
            league_name = "한국시리즈" if league_type == "korean_series" else "정규시즌"
            rank_info = f"최종 순위 {team_rank}등" if team_rank else "순위 정보 없음"
            
            if team_games:
                game_count = len(team_games)
                game_summary = f"{game_count}경기"
                message = f"{team_name}의 {year}년 {league_name} 마지막 경기 조회 완료 ({rank_info}, {game_summary})"
            else:
                message = f"{team_name}의 {year}년 {league_name} 마지막 경기를 찾을 수 없습니다 ({rank_info})"
            
            return ToolResult(
                success=True,
                data=combined_data,
                message=message
            )
            
        except Exception as e:
            logger.error(f"[BaseballAgent] Team last game query error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"팀 마지막 경기 조회 중 오류 발생: {e}"
            )
    
    def _tool_get_korean_series_winner(self, year: int) -> ToolResult:
        """한국시리즈 우승팀 조회 도구"""
        try:
            # 1단계: 한국시리즈 마지막 경기 조회
            final_game_result = self._tool_get_season_final_game_date(year, "korean_series")
            
            if not final_game_result.success:
                return ToolResult(
                    success=False,
                    data={"year": year},
                    message=f"{year}년 한국시리즈 정보를 찾을 수 없습니다"
                )
            
            games = final_game_result.data.get("formatted_games", [])
            if not games:
                return ToolResult(
                    success=False,
                    data={"year": year},
                    message=f"{year}년 한국시리즈 경기 결과를 찾을 수 없습니다"
                )
            
            # 2단계: 우승팀 식별
            # 한국시리즈는 7전 4선승제이므로 마지막 경기의 승리팀이 우승팀
            final_game = games[-1]  # 마지막 경기
            
            winner_team_id = None
            winner_team_name = None
            
            # winning_team 필드가 있으면 사용
            if 'winning_team' in final_game:
                winner_team_id = final_game['winning_team']
            else:
                # 점수 비교로 승리팀 결정
                home_score = final_game.get('home_score', 0)
                away_score = final_game.get('away_score', 0)
                
                if home_score > away_score:
                    winner_team_id = final_game.get('home_team')
                elif away_score > home_score:
                    winner_team_id = final_game.get('away_team')
            
            # 팀 ID를 팀명으로 변환
            if winner_team_id:
                winner_team_name = self._convert_team_id_to_name(winner_team_id)
            
            if not winner_team_name:
                return ToolResult(
                    success=False,
                    data={
                        "year": year,
                        "final_game": final_game
                    },
                    message=f"{year}년 한국시리즈 우승팀을 정확히 식별할 수 없습니다"
                )
            
            # 우승팀 순위 정보도 함께 조회
            rank_result = self._tool_get_team_rank(winner_team_name, year)
            winner_rank = rank_result.data.get("team_rank") if rank_result.success else None
            
            result_data = {
                "year": year,
                "winner_team_id": winner_team_id,
                "winner_team_name": winner_team_name,
                "winner_rank": winner_rank,
                "final_game": final_game,
                "series_type": "한국시리즈"
            }
            
            rank_text = f" (정규시즌 {winner_rank}위)" if winner_rank else ""
            message = f"{year}년 한국시리즈 우승팀: {winner_team_name}{rank_text}"
            
            return ToolResult(
                success=True,
                data=result_data,
                message=message
            )
            
        except Exception as e:
            logger.error(f"[BaseballAgent] Korean Series winner query error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"한국시리즈 우승팀 조회 중 오류 발생: {e}"
            )

    def _tool_get_current_datetime(self, **kwargs) -> ToolResult:
        """현재 날짜 및 시간 조회 도구"""
        try:
            datetime_info = get_current_datetime()
            return ToolResult(
                success=True,
                data=datetime_info,
                message=f"현재 시간은 {datetime_info['formatted_date']} {datetime_info['formatted_time']}입니다."
            )
        except Exception as e:
            logger.error(f"Current datetime tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"현재 시간 조회 중 오류 발생: {e}"
            )

    def _tool_get_baseball_season_info(self, **kwargs) -> ToolResult:
        """현재 야구 시즌 정보 조회 도구"""
        try:
            season_info = get_baseball_season_info()
            return ToolResult(
                success=True,
                data=season_info,
                message=f"현재 {season_info['current_year']}년 야구 시즌은 '{season_info['season_status']}' 상태입니다."
            )
        except Exception as e:
            logger.error(f"Baseball season info tool error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"야구 시즌 정보 조회 중 오류 발생: {e}"
            )
    
    def _analyze_player_comparison(self, player1_data: Dict, player2_data: Dict, position: str) -> Dict:
        """두 선수 데이터를 분석하여 비교 결과를 생성합니다."""
        analysis = {
            "summary": "",
            "key_stats": {},
            "strengths": {
                "player1": [],
                "player2": []
            }
        }
        
        try:
            # 타자 비교 분석
            if position in ["batting", "both"] and "batting_stats" in player1_data and "batting_stats" in player2_data:
                p1_batting = player1_data["batting_stats"]
                p2_batting = player2_data["batting_stats"]
                
                # 주요 타격 지표 비교
                key_batting_stats = ["avg", "ops", "home_runs", "rbi", "runs", "hits"]
                
                for stat in key_batting_stats:
                    if stat in p1_batting and stat in p2_batting:
                        p1_val = float(p1_batting[stat] or 0)
                        p2_val = float(p2_batting[stat] or 0)
                        
                        analysis["key_stats"][stat] = {
                            "player1": p1_val,
                            "player2": p2_val,
                            "difference": p1_val - p2_val,
                            "better_player": "player1" if p1_val > p2_val else "player2" if p2_val > p1_val else "tie"
                        }
                        
                        # 장점 분석
                        if p1_val > p2_val:
                            analysis["strengths"]["player1"].append(f"{stat}: {p1_val}")
                        elif p2_val > p1_val:
                            analysis["strengths"]["player2"].append(f"{stat}: {p2_val}")
            
            # 투수 비교 분석
            if position in ["pitching", "both"] and "pitching_stats" in player1_data and "pitching_stats" in player2_data:
                p1_pitching = player1_data["pitching_stats"]
                p2_pitching = player2_data["pitching_stats"]
                
                # 주요 투구 지표 비교 (ERA, WHIP은 낮을수록 좋음)
                key_pitching_stats = ["era", "whip", "wins", "strikeouts", "innings_pitched"]
                
                for stat in key_pitching_stats:
                    if stat in p1_pitching and stat in p2_pitching:
                        p1_val = float(p1_pitching[stat] or 0)
                        p2_val = float(p2_pitching[stat] or 0)
                        
                        # ERA, WHIP은 낮을수록 좋음
                        if stat in ["era", "whip"]:
                            better_player = "player1" if p1_val < p2_val else "player2" if p2_val < p1_val else "tie"
                        else:
                            better_player = "player1" if p1_val > p2_val else "player2" if p2_val > p1_val else "tie"
                        
                        analysis["key_stats"][stat] = {
                            "player1": p1_val,
                            "player2": p2_val,
                            "difference": p1_val - p2_val,
                            "better_player": better_player
                        }
                        
                        # 장점 분석
                        if better_player == "player1":
                            analysis["strengths"]["player1"].append(f"{stat}: {p1_val}")
                        elif better_player == "player2":
                            analysis["strengths"]["player2"].append(f"{stat}: {p2_val}")
            
            # 요약 생성
            p1_advantages = len(analysis["strengths"]["player1"])
            p2_advantages = len(analysis["strengths"]["player2"])
            
            if p1_advantages > p2_advantages:
                analysis["summary"] = f"선수1이 {p1_advantages}개 지표에서 우세, 선수2가 {p2_advantages}개 지표에서 우세"
            elif p2_advantages > p1_advantages:
                analysis["summary"] = f"선수2가 {p2_advantages}개 지표에서 우세, 선수1이 {p1_advantages}개 지표에서 우세"
            else:
                analysis["summary"] = f"두 선수 모두 {p1_advantages}개씩 지표에서 우세하여 비슷한 수준"
                
        except Exception as e:
            logger.error(f"Player comparison analysis error: {e}")
            analysis["summary"] = "비교 분석 중 오류 발생"
        
        return analysis

    def _is_chitchat(self, query: str) -> bool:
        """간단한 일상 대화인지 키워드 기반으로 확인합니다."""
        query_lower = query.lower().strip()
        
        # 야구 관련 키워드가 있으면 일상 대화 아님
        baseball_keywords = [
            "우승", "챔피언", "선수", "팀", "경기", "시즌", "년",
            "성적", "기록", "통산", "타율", "홈런", "투수", "타자"
        ]
        
        if any(keyword in query_lower for keyword in baseball_keywords):
            return False
        
        # 선수 관련 질문 패턴 ("김도영이 누구야" 같은 질문)
        import re
        if re.search(r'[가-힣]{2,4}(이가|이는|이)?\s*(누구|뭐)', query_lower):
            return False
    
        # 일상 대화 키워드
        chitchat_keywords = ["안녕", "고마워", "반가워", "도움", "기능"]
    
        return any(keyword in query_lower for keyword in chitchat_keywords)

    def _get_chitchat_response(self, query: str) -> Optional[str]:
        """미리 정의된 일상 대화 응답을 반환합니다."""
        query_lower = query.lower()
        if "안녕" in query_lower:
            return "안녕하세요! 저는 KBO 리그 데이터 분석가 BEGA입니다. 야구에 대해 궁금한 점이 있으시면 무엇이든 물어보세요!"
        if "누구" in query_lower or "이름이" in query_lower:
            return "저는 KBO 리그 전문 데이터 분석가 'BEGA'입니다. 선수 기록, 경기 결과, 리그 규정 등 야구에 대한 모든 것을 알려드릴 수 있습니다."
        if "고마워" in query_lower:
            return "천만에요! 더 궁금한 점이 있으시면 언제든지 다시 물어보세요."
        if "도움" in query_lower or "기능" in query_lower:
            return """저는 KBO 리그와 관련된 다양한 질문에 답변할 수 있어요. 예를 들어, 다음과 같이 질문해보세요.
- "어제 LG 경기 결과 알려줘"
- "김도영 2024년 성적은 어땠어?"
- "ABS 규정에 대해 설명해줘"
"""
        return None

    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        통계 질문을 처리하고 실제 DB 데이터를 사용하여 답변을 생성합니다.
        """
        logger.info(f"[BaseballAgent] Processing query: {query}")
        
        # --- 신규 추가: 일상 대화 처리기 ---
        if self._is_chitchat(query):
            response = self._get_chitchat_response(query)
            if response:
                logger.info("[BaseballAgent] 일상 대화로 처리합니다.")
                return {
                    "answer": response,
                    "tool_calls": [],
                    "tool_results": [],
                    "verified": True,
                    "data_sources": ["predefined"],
                    "error": None
                }
        # --- 일상 대화 처리기 끝 ---

        # 1단계: 질문 분석 및 필요한 도구 결정
        analysis_result = await self._analyze_query_and_plan_tools(query, context)
        
        if analysis_result["error"]:
            return {
                "answer": "질문 분석 중 오류가 발생했습니다.",
                "tool_calls": [],
                "verified": False,
                "error": analysis_result["error"]
            }
        
        # 2단계: 도구 실행을 통한 데이터 수집
        tool_results = []
        for tool_call in analysis_result["tool_calls"]:
            logger.info(f"[BaseballAgent] Executing tool: {tool_call.tool_name}")
            result = self.tool_caller.execute_tool(tool_call)
            tool_results.append(result)
            
        # 3단계: 수집된 실제 데이터를 바탕으로 답변 생성
        answer_result = await self._generate_verified_answer(query, tool_results, context)
        
        return {
            "answer": answer_result["answer"],
            "tool_calls": analysis_result["tool_calls"],
            "tool_results": tool_results,
            "verified": answer_result["verified"],
            "data_sources": answer_result["data_sources"],
            "error": answer_result.get("error")
        }
    
    async def _analyze_query_and_plan_tools(
        self, 
        query: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        사용자 질문을 분석하고 필요한 도구 호출을 계획합니다.
        '작년', '올해' 등 상대적인 시간 표현을 미리 처리합니다.
        """
        logger.info(f"[BaseballAgent] Analyzing query for tool planning: {query}")
        
        # 시간 표현 전처리
        now = datetime.now()
        current_year = now.year
        current_date = now.strftime("%Y년 %m월 %d일")
        year_replacements = {
            "재작년": str(now.year - 2),
            "작년": str(now.year - 1),
            "올해": str(now.year),
            "내년": str(now.year + 1),
        }

        processed_query = query
        for keyword, year_str in year_replacements.items():
            if keyword in processed_query:
                processed_query = processed_query.replace(keyword, f"{year_str}년")

        # LLM을 사용하여 질문을 분석하고 도구 사용 계획 수립
        
        # 0. 엔티티 추출 (LLM 호출 전)
        from ..core.entity_extractor import extract_entities_from_query
        entity_filter = extract_entities_from_query(processed_query)

        # 추출된 엔티티를 프롬프트에 제공할 컨텍스트로 포맷팅
        entity_context_parts = []
        if entity_filter.season_year:
            entity_context_parts.append(f"- 연도: {entity_filter.season_year}년")
        if entity_filter.player_name:
            entity_context_parts.append(f"- 선수명: {entity_filter.player_name}")
        if entity_filter.team_id:
            entity_context_parts.append(f"- 팀명: {entity_filter.team_id}")
        if entity_filter.stat_type:
            entity_context_parts.append(f"- 통계 지표: {entity_filter.stat_type}")
        if entity_filter.league_type:
            entity_context_parts.append(f"- 리그 타입: {entity_filter.league_type}")
        
        entity_context = ""
        if entity_context_parts:
            entity_context = "\n\n### 질문에서 분석된 정보:\n" + "\n".join(entity_context_parts)

        query_text = processed_query # 전처리된 쿼리 사용
        analysis_prompt_template = """
당신은 야구 통계 전문 에이전트입니다. 사용자의 질문을 분석하고 실제 데이터베이스에서 정확한 답변을 얻기 위해 어떤 도구들을 사용해야 하는지 결정해야 합니다.

**현재날짜: {current_date}**
**현재년도: {current_year}년**
작년: {last_year}년
재작년: {two_years_ago}년

질문: {query_text}

사용 가능한 도구들과 정확한 매개변수:

1. **get_player_stats**: 특정 선수의 개별 시즌 통계 조회
   - player_name (필수): 선수명
   - year (필수): 시즌 년도 (기본값: current_year = {current_year})
   - position (선택): "batting", "pitching", "both" 중 하나 (기본값: "both")

2. **get_leaderboard**: 통계 지표별 순위/리더보드 조회  
   - stat_name (필수): 통계 지표명 (예: "home_runs", "era", "ops", "타율")
   - year (필수): 시즌 년도 (기본값: current_year = {current_year})
   - position (필수): "batting" 또는 "pitching"
   - team_filter (선택): 특정 팀명 (예: "KIA", "LG")
   - limit (선택): 상위 몇 명까지 (기본값: 10)

3. **validate_player**: 선수 존재 여부 및 정확한 이름 확인
   - player_name (필수): 선수명
   - year (선택): 시즌 년도 (기본값: current_year = {current_year})

4. **get_career_stats**: 선수의 통산(커리어) 통계 조회
   - player_name (필수): 선수명
   - position (선택): "batting", "pitching", "both" 중 하나 (기본값: "both")

5. **get_team_summary**: 팀의 주요 선수들과 통계 조회
   - team_name (필수): 팀명 (예: "KIA", "기아")
   - year (필수): 시즌 년도 (기본값: current_year = {current_year})

6. **get_position_info**: 포지션 약어를 전체 포지션명으로 변환
   - position_abbr (필수): 포지션 약어 (예: "지", "포", "一", "二", "三")

7. **get_team_basic_info**: 팀의 기본 정보 조회
   - team_name (필수): 팀명 (예: "KIA", "LG", "두산")

8. **get_defensive_stats**: 선수의 수비 통계 조회
   - player_name (필수): 선수명
   - year (선택): 시즌 년도 (생략하면 통산) 

9. **get_velocity_data**: 투수의 구속 데이터 조회
   - player_name (필수): 선수명
   - year (선택): 시즌 년도 (생략하면 최근 데이터)

10. **search_regulations**: KBO 규정 검색
    - query (필수): 검색할 규정 내용 (예: "타이브레이크", "FA 조건", "인필드 플라이")

11. **get_regulations_by_category**: 카테고리별 규정 조회
    - category (필수): 규정 카테고리 (basic, player, game, technical, discipline, postseason, special, terms)

12. **get_game_box_score**: 특정 경기의 박스스코어와 상세 정보 조회
    - game_id (선택): 경기 고유 ID
    - date (선택): 경기 날짜 (YYYY-MM-DD)
    - home_team (선택): 홈팀명
    - away_team (선택): 원정팀명

13. **get_games_by_date**: 특정 날짜의 모든 경기 조회
    - date (필수): 경기 날짜 (YYYY-MM-DD)
    - team (선택): 특정 팀만 조회

14. **get_head_to_head**: 두 팀 간의 직접 대결 기록 조회
    - team1 (필수): 팀1 이름
    - team2 (필수): 팀2 이름
    - year (선택): 시즌 년도
    - limit (선택): 최근 몇 경기까지 (기본 10경기)

15. **get_player_game_performance**: 특정 선수의 개별 경기 성적 조회
    - player_name (필수): 선수명
    - date (선택): 경기 날짜
    - recent_games (선택): 최근 몇 경기까지 (기본 5경기)

16. **compare_players**: 두 선수의 통계를 비교 분석
    - player1 (필수): 첫 번째 선수명
    - player2 (필수): 두 번째 선수명
    - comparison_type (선택): "career"(통산 비교, 기본값) 또는 "season"(특정 시즌 비교)
    - year (선택): 특정 시즌 비교 시 연도
    - position (선택): "batting", "pitching", "both" 중 하나 (기본값: "both")

17. **get_season_final_game_date**: 특정 시즌의 마지막 경기 날짜를 조회
   - year (필수): 시즌 년도 (기본값: current_year = {current_year})
   - league_type (선택): "regular_season" 또는 "korean_series" (기본값: "korean_series")

18. **get_team_rank**: 특정 시즌의 팀 최종 순위를 조회
   - team_name (필수): 팀명 (예: "KIA", "기아", "SSG")
   - year (필수): 시즌 년도 (기본값: current_year = {current_year})

19. **get_team_last_game**: 특정 팀의 실제 마지막 경기를 지능적으로 조회
   - team_name (필수): 팀명 (예: "SSG", "기아", "KIA")
   - year (필수): 시즌 년도 (기본값: current_year = {current_year})
   - 자동으로 팀 순위를 확인하여 포스트시즌 진출팀(1-5위)은 한국시리즈, 미진출팀(6-10위)은 정규시즌 마지막 경기를 조회

20. **get_korean_series_winner**: 특정 시즌의 한국시리즈 우승팀을 조회
   - year (필수): 시즌 년도 (기본값: current_year = {current_year})
   - 우승팀과 함께 정규시즌 순위 정보도 제공

21. **get_current_datetime**: 현재 날짜와 시간 조회
    - "지금 몇 시", "오늘 날짜" 질문에 사용

22. **get_baseball_season_info**: 현재 야구 시즌 정보 조회
    - "지금 몇 시", "오늘 날짜" 질문에 사용
    - "지금 야구 시즌이야?", "시즌 중" 질문에 사용

23. **search_documents**: KBO 리그 규정, 용어 정의, 선수 관련 스토리 등 비정형 텍스트 문서를 검색합니다.
    - query (필수): 검색할 질문 또는 키워드
    - 'ABS가 뭐야?', 'FA 규정 알려줘'와 같은 설명형/정의형 질문에 사용

23. **search_documents**: KBO 리그 규정, 용어 정의 등 비정형 텍스트 문서 검색
    - query (필수): 검색할 질문 또는 키워드
    - 'ABS가 뭐야?', 'FA 규정 알려줘'와 같은 설명형/정의형 질문에 사용

질문: {query}

**중요한 규칙:**
- "우승팀", "챔피언", "한국시리즈 우승" 질문은 get_korean_series_winner 사용 (자동으로 우승팀과 순위 정보 제공)
- 시즌이 명시되지 않으면 current_year({current_year})을 기본값으로 사용
- "통산", "커리어", "총", "KBO 리그 통산" 키워드가 있으면 반드시 get_career_stats 사용
- "세이브" 키워드가 포함된 통산 기록 질문은 get_career_stats 사용
- "몇 년", "2025년" 등 구체적 연도가 있으면 get_player_stats 사용
- "가장 많은", "최고", "언제", "어느 시즌" 등 최고 기록 시즌을 묻는 질문:
  * 먼저 get_career_stats로 통산 기록 확인
  * 필요시 여러 연도의 get_player_stats로 연도별 비교
- "마지막 경기", "최종전" 질문: 특정 팀이 언급되면 get_team_last_game을 우선 사용 (자동으로 순위 확인 후 적절한 리그의 마지막 경기 조회). 전체 리그 마지막 경기는 get_season_final_game_date 사용
- "결승전", "우승" 질문은 get_season_final_game_date(league_type='korean_series') 사용
- 팀 순위 질문("몇 등", "순위", "몇 위")은 get_team_rank 사용
- 순위/리더보드 질문은 get_leaderboard 사용
- 경기 결과, 박스스코어 질문은 get_game_box_score 사용
- 특정 날짜 경기 질문("5월 5일 경기", "어린이날")은 get_games_by_date 사용
- 시즌 일정 질문("언제부터 시작", "시범경기 일정")은 get_games_by_date 사용
- 팀 간 맞대결 질문은 get_head_to_head 사용
- 포스트시즌("한국시리즈", "플레이오프") 질문은 get_games_by_date 사용
- 선수 개별 경기 활약 질문은 get_player_game_performance 사용
- 선수 비교 질문("A vs B", "A와 B 중 누가", "더 뛰어난")은 compare_players 사용
- 통산 기록 비교는 comparison_type="career", 특정 시즌 비교는 comparison_type="season"

위 질문에 정확히 답변하기 위해 어떤 도구들을 어떤 순서로 호출해야 하는지 JSON 형식으로 계획을 세워주세요.
**절대 금지사항**: 
- "DATE_FROM_STEP_1", "YEAR_FROM_CONTEXT", "<date_from_relevant_get_season_final_game_date>" 같은 플레이스홀더 텍스트를 절대 사용하지 마세요
- 매개변수 값은 반드시 실제 구체적인 값을 사용하세요

**질문 유형별 도구 선택 예시**:
- "작년 SSG 마지막 경기" → get_team_last_game(team_name="SSG", year: {last_year})
- "2025시즌 정규시즌 최종전" → get_season_final_game_date(year=2025, league_type="regular_season")  
- "기아 마지막 경기" → get_team_last_game(team_name="기아", year: {last_year})
- "한국시리즈 마지막 경기" → get_season_final_game_date(year: {last_year}, league_type="korean_series")
- "작년 우승팀은?" → get_korean_series_winner(year: {last_year})
- "2024년 한국시리즈 챔피언" → get_korean_series_winner(year=2024)

중요한 원칙:
- 반드시 실제 데이터베이스 조회가 필요한 경우만 도구를 사용하세요
- 선수명이 불확실한 경우 먼저 validate_player로 확인하세요
- 리그 전체 순위("최고", "상위", "1위")는 get_leaderboard를 사용하세요
- 특정 선수의 개별 시즌 통계는 get_player_stats를 사용하세요
- 특정 선수의 통산/커리어 기록은 get_career_stats를 사용하세요
- 특정 선수의 "가장 좋은 시즌" 질문은 get_career_stats + 여러 연도 get_player_stats 조합
- 경기 일정/결과는 get_games_by_date 또는 get_game_box_score 사용하세요
- 날짜 형식은 YYYY-MM-DD로 변환하세요 (기본값: current_year = {current_year}, 
예: "5월 5일" → "{current_year}-05-05")
- 연도 정보가 없는 경우 현재 연도를 기본값으로 사용하세요
- "재작년", "제작년" → {two_years_ago}, 
"작년", "지난해" → {last_year}, 
"올해" → {current_year}로 자동 변환하세요
- 상대적 연도 표현은 현재 연도를 기반으로 동적으로 계산하세요

**반드시 다음 JSON 형식으로만 응답하세요:**
```json
{{
    "analysis": "질문 분석 내용",
    "tool_calls": [
        {{
            "tool_name": "도구명",
            "parameters": {{
                "매개변수명": "값"
            }},
            "reasoning": "이 도구를 사용하는 이유"
        }}
    ],
    "expected_result": "예상되는 답변 유형"
}}
```
"""
        analysis_prompt = analysis_prompt_template.format(
            current_date=current_date,
            current_year=current_year,
            last_year=current_year - 1,
            two_years_ago=current_year - 2,
            query_text=query_text,
            query=query
            )
        
        logger.info(f"[TEST] query: {query_text}")

        try:
            # LLM 호출하여 분석 결과 받기
            analysis_messages = [{"role": "user", "content": analysis_prompt}]
            raw_response = await self.llm_generator(analysis_messages)
            
            logger.info(f"[BaseballAgent] Raw LLM response: {raw_response[:200]}...")
            
            # JSON 블록 추출 (```json ... ``` 형태인 경우)
            if '```json' in raw_response:
                start = raw_response.find('```json') + 7
                end = raw_response.find('```', start)
                json_content = raw_response[start:end].strip() if end != -1 else raw_response[start:].strip()
            elif raw_response.strip().startswith('{'):
                json_content = raw_response.strip()
            else:
                # JSON이 아닌 응답인 경우 기본 분석 제공
                logger.warning(f"[BaseballAgent] Non-JSON response, providing fallback analysis")
                return {
                    "analysis": f"'{query}' 질문을 리더보드로 분석",
                    "tool_calls": [ToolCall(
                        tool_name="get_leaderboard",
                        parameters={
                            "stat_name": "ops",
                            "year": current_year,
                            "position": "batting",
                            "limit": 10
                        }
                    )],
                    "expected_result": "상위 타자 순위",
                    "error": None
                }
            
            # JSON 파싱
            try:
                cleaned_json = clean_json_response(json_content)
                analysis_data = json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                logger.error(f"[BaseballAgent] JSON parsing error: {e}")
                logger.error(f"[BaseballAgent] Original content: {json_content}")
                logger.error(f"[BaseballAgent] Cleaned content: {cleaned_json}")
                raise
            
            # ToolCall 객체들로 변환
            tool_calls = []
            for call_data in analysis_data.get("tool_calls", []):
                tool_call = ToolCall(
                    tool_name=call_data["tool_name"],
                    parameters=call_data["parameters"]
                )
                tool_calls.append(tool_call)
            
            return {
                "analysis": analysis_data.get("analysis", ""),
                "tool_calls": tool_calls,
                "expected_result": analysis_data.get("expected_result", ""),
                "error": None
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"[BaseballAgent] JSON parsing error in query analysis: {e}")
            logger.error(f"[BaseballAgent] Failed response content: {raw_response}")
            
            # 현재 연도 계산 및 엔티티 추출
            import datetime as dt
            from ..core.entity_extractor import extract_entities_from_query
            
            current_year = dt.datetime.now().year
            entity_filter = extract_entities_from_query(query)
            
            # 질문 유형에 따른 스마트 폴백
            query_lower = query.lower()
            
            # 선수명 추출 시도
            import re
            korean_names = re.findall(r'[가-힣]{2,4}', query)
            potential_player_name = korean_names[0] if korean_names else entity_filter.player_name
            
            # 질문에서 추출된 값들 사용
            extracted_year = entity_filter.season_year or current_year
            extracted_stat = entity_filter.stat_type or "ops"  # 기본값
            extracted_position = entity_filter.position_type or "batting"  # 기본값
            
            # 통산/커리어 질문 감지
            if any(word in query_lower for word in ["통산", "커리어", "총", "kbo 리그"]):
                if potential_player_name:
                    fallback_tool = ToolCall(
                        tool_name="get_career_stats",
                        parameters={
                            "player_name": potential_player_name,
                            "position": "both"
                        }
                    )
                    analysis = f"{potential_player_name} 선수의 통산 기록 조회"
                else:
                    fallback_tool = ToolCall(
                        tool_name="get_leaderboard",
                        parameters={
                            "stat_name": extracted_stat,
                            "year": extracted_year,
                            "position": extracted_position,
                            "limit": 10
                        }
                    )
                    analysis = f"통산 기록 관련 질문으로 판단하여 상위 {extracted_stat} 조회"
                    
            # 경기 일정/결과 질문 감지
            elif any(word in query_lower for word in ["경기", "일정", "결과", "어린이날", "한국시리즈", "시범경기", "언제부터", "우승"]):
                # 날짜 추출 시도
                import re
                date_patterns = [
                    r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',  # 2025년 5월 5일
                    r'(\d{4})-(\d{1,2})-(\d{1,2})',  # 2025-05-05
                ]
                extracted_date = None
                for pattern in date_patterns:
                    match = re.search(pattern, query)
                    if match:
                        year, month, day = match.groups()
                        extracted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        break
                
                if extracted_date:
                    fallback_tool = ToolCall(
                        tool_name="get_games_by_date",
                        parameters={
                            "date": extracted_date
                        }
                    )
                    analysis = f"{extracted_date} 경기 결과 조회"
                else:
                    # 이미 추출된 연도 사용
                    fallback_tool = ToolCall(
                        tool_name="get_games_by_date", 
                        parameters={
                            "date": f"{extracted_year}-03-01"  # 기본값으로 시즌 시작 시점
                        }
                    )
                    analysis = f"{extracted_year}년 경기 정보 조회"
                    
            # 가장 많은/최고 시즌 질문 감지
            elif any(word in query_lower for word in ["가장 많은", "최고", "언제", "어느 시즌", "어떤 년도"]):
                if potential_player_name:
                    fallback_tool = ToolCall(
                        tool_name="get_career_stats",
                        parameters={
                            "player_name": potential_player_name,
                            "position": "both"
                        }
                    )
                    analysis = f"{potential_player_name} 선수의 최고 시즌 조회를 위한 통산 기록 확인"
                else:
                    # 질문에서 "홈런" 언급이 있으면 home_runs, 없으면 기본값
                    stat_for_max = "home_runs" if "홈런" in query_lower else extracted_stat
                    fallback_tool = ToolCall(
                        tool_name="get_leaderboard",
                        parameters={
                            "stat_name": stat_for_max,
                            "year": extracted_year,
                            "position": extracted_position,
                            "limit": 10
                        }
                    )
                    analysis = f"최고 기록 관련 질문으로 판단하여 {stat_for_max} 순위 조회"
                    
            # 투수 질문 감지  
            elif any(word in query_lower for word in ["투수", "투구", "방어율", "era", "whip", "승", "세이브"]):
                if potential_player_name:
                    fallback_tool = ToolCall(
                        tool_name="get_career_stats",
                        parameters={
                            "player_name": potential_player_name,
                            "position": "pitching"
                        }
                    )
                    analysis = f"{potential_player_name} 투수의 통산 기록 조회"
                else:
                    # 투수 질문에서 특정 통계가 언급되면 사용, 없으면 ERA 기본값
                    pitcher_stat = extracted_stat if extracted_stat in ["era", "whip", "wins", "saves", "strikeouts", "innings_pitched"] else "era"
                    fallback_tool = ToolCall(
                        tool_name="get_leaderboard",
                        parameters={
                            "stat_name": pitcher_stat,
                            "year": extracted_year,
                            "position": "pitching",
                            "limit": 10
                        }
                    )
                    analysis = f"투수 관련 질문으로 판단하여 {pitcher_stat} 기준 상위 투수 조회"
            
            # 타자 질문 감지 (기본값)
            else:
                if potential_player_name:
                    fallback_tool = ToolCall(
                        tool_name="get_career_stats",
                        parameters={
                            "player_name": potential_player_name,
                            "position": "batting"
                        }
                    )
                    analysis = f"{potential_player_name} 선수의 타격 통계 조회"
                else:
                    # 타자 질문에서 특정 통계가 언급되면 사용, 없으면 OPS 기본값  
                    batter_stat = extracted_stat if extracted_stat in ["ops", "avg", "home_runs", "rbi", "stolen_bases", "war", "wrc_plus"] else "ops"
                    fallback_tool = ToolCall(
                        tool_name="get_leaderboard", 
                        parameters={
                            "stat_name": batter_stat,
                            "year": extracted_year,
                            "position": "batting",
                            "limit": 10
                        }
                    )
                    analysis = f"타자 관련 질문으로 판단하여 {batter_stat} 기준 상위 타자 조회"
            
            return {
                "analysis": analysis,
                "tool_calls": [fallback_tool],
                "expected_result": "리더보드 순위",
                "error": None
            }
        except Exception as e:
            logger.error(f"[BaseballAgent] Error in query analysis: {e}")
            return {
                "analysis": "",
                "tool_calls": [],
                "expected_result": "",
                "error": f"질문 분석 오류: {e}"
            }
    
    async def _generate_verified_answer(
        self, 
        query: str, 
        tool_results: List[Dict[str, Any]], 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        도구 실행 결과를 바탕으로 검증된 답변을 생성합니다.
        """
        logger.info(f"[BaseballAgent] Generating verified answer with {len(tool_results)} tool results")
        
        # 시간 컨텍스트 생성 
        now = datetime.now()
        current_year = now.year

        time_context = ""
        if "재작년" in query:
            actual_year = current_year - 2
            time_context = f"\n\n**중요**: 사용자가 '재작년'이라고 했고, 현재가 {current_year}년이므로 조회된 데이터는 {actual_year}년입니다. 답변할 때 '{actual_year}년'으로 명시하세요."
        elif "작년" in query:
            actual_year = current_year - 1
            time_context = f"\n\n**중요**: 사용자가 '작년'이라고 했고, 현재가 {current_year}년이므로 조회된 데이터는 {actual_year}년입니다. 답변할 때 '{actual_year}년'으로 명시하세요."
        elif "올해" in query:
            time_context = f"\n\n**중요**: 사용자가 '올해'라고 했고, 현재는 {current_year}년입니다."

        # 도구 실행 결과를 텍스트로 변환
        tool_data_summary = []
        data_sources = []
        
        for i, result in enumerate(tool_results):
            # result는 이제 dict 형태 (ToolResult.to_dict() 결과)
            if result.success:
                tool_data_summary.append(f"도구 {i+1} 결과: {result.message}")
                try:
                    # 팀 코드를 전체 이름으로 변환하는 로직 추가
                    sanitized_data = _replace_team_codes(result.data)
                    data_json = json.dumps(
                        sanitized_data, 
                        ensure_ascii=False, 
                        indent=2,
                        cls=DateTimeEncoder
                    )
                    tool_data_summary.append(f"데이터: {data_json}")
                except Exception as e:
                    logger.error(f"[BaseballAgent] JSON serialization error: {e}")
                    tool_data_summary.append(f"데이터: (직렬화 실패)")
                
                result_data = result.data if result.data else {}
                data_sources.append({
                    "tool": result_data.get("source", "database") if isinstance(result_data, dict) else "database",
                    "verified": True,
                    "data_points": len(result_data) if isinstance(result_data, list) else 1
                })
            else:
                tool_data_summary.append(f"도구 {i+1} 실패: {result.message}")
                data_sources.append({
                    "tool": "failed",
                    "verified": False,
                    "error": result.message
                })
        
        # 검증된 데이터만 사용하여 답변 생성
        answer_prompt = f"""
안녕하세요! KBO 야구 데이터를 다루는 BEGA입니다.

사용자 질문: {query}{time_context}

조회된 기록:
{chr(10).join(tool_data_summary)}

다음 가이드라인에 따라 자연스럽고 친근하게 답변해주세요:

**중요 - 연도 처리:**
- 조회된 데이터의 연도(year 필드)를 그대로 사용하세요
- 절대로 연도를 재계산하거나 추측하지 마세요
- 데이터에 "year: 2023"이 있으면 → "2023년"이라고 답변
- "재작년", "작년" 같은 표현은 이미 정확한 연도로 변환되어 조회되었습니다

답변 스타일:
- 친구에게 설명하듯 편안하고 자연스럽게
- "기록을 확인해보니", "데이터를 살펴보면", "최신 정보로는" 등의 자연스러운 표현
- 실제 야구 해설위원이나 스포츠 기자가 답변하는 톤

팀 순위 정보가 있는 경우:
- 마지막 경기 결과와 함께 팀의 최종 순위를 자연스럽게 언급
- 예: "~팀의 최종순위는 ~등으로 시즌을 마무리 했습니다"
- 포스트시즌 진출 여부(상위 5등 이내)도 함께 언급
- 한국시리즈 우승팀인 경우 "우승"이라는 표현 사용

우승팀 식별 방법:
- 한국시리즈 마지막 경기에서 승리한 팀이 우승팀
- 경기 데이터의 'winning_team' 필드 또는 점수 비교로 우승팀 판단
- "우승", "챔피언", "한국시리즈 우승팀" 등의 표현으로 답변

절대 사용 금지:
- "핵심:", "설명:", "요약:" 같은 구조화된 표현  
- "제공된 DB에서", "데이터베이스 기준", "제시된 검색 결과" 같은 기술적 용어
- 지나치게 격식적인 공문서 톤
- **조회된 연도를 임의로 변경하거나 재계산**

데이터 없는 경우:
- "아쉽게도 해당 경기 기록을 찾을 수 없네요"
- "죄송해요, 그 정보는 현재 확인이 어려워요"  
- "해당 데이터가 아직 업데이트되지 않은 것 같아요"

정확성 원칙:
- 위의 조회 데이터만 사용 (추측 금지)
- **조회된 연도(year 필드)를 절대 변경하지 마세요!**
- 불확실하면 솔직하게 모른다고 표현

자연스럽고 친근한 대화체로 답변해주세요!
"""

        try:
            # 검증된 데이터 기반 답변 생성
            answer_messages = [{"role": "user", "content": answer_prompt}]
            answer = await self.llm_generator(answer_messages)
            
            # 성공한 도구가 하나라도 있는지 확인
            has_verified_data = any(result.success for result in tool_results)
            
            return {
                "answer": answer,
                "verified": has_verified_data,
                "data_sources": data_sources,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"[BaseballAgent] Error generating verified answer: {e}")
            return {
                "answer": "답변 생성 중 오류가 발생했습니다. 제공된 DB 조회 결과를 확인할 수 없습니다.",
                "verified": False,
                "data_sources": [],
                "error": f"답변 생성 오류: {e}"
            }