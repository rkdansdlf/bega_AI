"""
야구 통계 전문 에이전트입니다.

이 에이전트는 LLM의 환각을 방지하기 위해 모든 통계 질문에 대해 
반드시 실제 DB를 조회한 결과만 사용합니다.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from psycopg2.extensions import connection as PgConnection

from ..tools.database_query import DatabaseQueryTool
from ..tools.regulation_query import RegulationQueryTool
from ..tools.game_query import GameQueryTool
from .tool_caller import ToolCaller, ToolCall, ToolResult

logger = logging.getLogger(__name__)

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
        self.tool_caller = ToolCaller()
        
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
                "year": "시즌 년도 (기본값: 2025)"
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
    
    def _tool_validate_player(self, player_name: str, year: int = 2025) -> ToolResult:
        """선수 존재 여부 확인 도구"""
        try:
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

    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        통계 질문을 처리하고 실제 DB 데이터를 사용하여 답변을 생성합니다.
        
        Args:
            query: 사용자 질문
            context: 추가 컨텍스트 정보
            
        Returns:
            처리 결과와 검증된 답변
        """
        logger.info(f"[BaseballAgent] Processing query: {query}")
        
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
        """
        logger.info(f"[BaseballAgent] Analyzing query for tool planning: {query}")
        
        # LLM을 사용하여 질문을 분석하고 도구 사용 계획 수립
        query_text = query
        # 템플릿을 분리하여 f-string 문제 해결
        analysis_prompt_template = """
당신은 야구 통계 전문 에이전트입니다. 사용자의 질문을 분석하고 실제 데이터베이스에서 정확한 답변을 얻기 위해 어떤 도구들을 사용해야 하는지 결정해야 합니다.

질문: {query_text}

사용 가능한 도구들과 정확한 매개변수:

1. **get_player_stats**: 특정 선수의 개별 시즌 통계 조회
   - player_name (필수): 선수명
   - year (필수): 시즌 년도 (예: 2025)
   - position (선택): "batting", "pitching", "both" 중 하나 (기본값: "both")

2. **get_leaderboard**: 통계 지표별 순위/리더보드 조회  
   - stat_name (필수): 통계 지표명 (예: "home_runs", "era", "ops", "타율")
   - year (필수): 시즌 년도
   - position (필수): "batting" 또는 "pitching"
   - team_filter (선택): 특정 팀명 (예: "KIA", "LG")
   - limit (선택): 상위 몇 명까지 (기본값: 10)

3. **validate_player**: 선수 존재 여부 및 정확한 이름 확인
   - player_name (필수): 선수명
   - year (선택): 시즌 년도 (기본값: 2025)

4. **get_career_stats**: 선수의 통산(커리어) 통계 조회
   - player_name (필수): 선수명
   - position (선택): "batting", "pitching", "both" 중 하나 (기본값: "both")

5. **get_team_summary**: 팀의 주요 선수들과 통계 조회
   - team_name (필수): 팀명 (예: "KIA", "기아")
   - year (필수): 시즌 년도

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

질문: {query}

**중요한 규칙:**
- 시즌이 명시되지 않으면 2025년을 기본값으로 사용
- "통산", "커리어", "총", "KBO 리그 통산" 키워드가 있으면 반드시 get_career_stats 사용
- "세이브" 키워드가 포함된 통산 기록 질문은 get_career_stats 사용
- "몇 년", "2023년" 등 구체적 연도가 있으면 get_player_stats 사용
- "가장 많은", "최고", "언제", "어느 시즌" 등 최고 기록 시즌을 묻는 질문:
  * 먼저 get_career_stats로 통산 기록 확인
  * 필요시 여러 연도의 get_player_stats로 연도별 비교
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
**중요**: 매개변수명을 정확히 사용하세요!

중요한 원칙:
- 반드시 실제 데이터베이스 조회가 필요한 경우만 도구를 사용하세요
- 선수명이 불확실한 경우 먼저 validate_player로 확인하세요
- 리그 전체 순위("최고", "상위", "1위")는 get_leaderboard를 사용하세요
- 특정 선수의 개별 시즌 통계는 get_player_stats를 사용하세요
- 특정 선수의 통산/커리어 기록은 get_career_stats를 사용하세요
- 특정 선수의 "가장 좋은 시즌" 질문은 get_career_stats + 여러 연도 get_player_stats 조합
- 경기 일정/결과는 get_games_by_date 또는 get_game_box_score 사용하세요
- 날짜 형식은 YYYY-MM-DD로 변환하세요 (예: "5월 5일" → "2023-05-05")
- 연도 정보가 없는 경우 2025년을 기본값으로 사용하세요

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
        analysis_prompt = analysis_prompt_template.format(query_text=query_text, query=query)

        try:
            # LLM 호출하여 분석 결과 받기
            analysis_messages = [{"role": "user", "content": analysis_prompt}]
            raw_response = await self.llm_generator(analysis_messages)
            
            logger.info(f"[BaseballAgent] Raw LLM response: {raw_response[:200]}...")
            
            # JSON 블록 추출 (```json ... ``` 형태인 경우)
            if '```json' in raw_response:
                start = raw_response.find('```json') + 7
                end = raw_response.find('```', start)
                if end != -1:
                    json_content = raw_response[start:end].strip()
                else:
                    json_content = raw_response[start:].strip()
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
                            "year": 2025,
                            "position": "batting",
                            "limit": 10
                        }
                    )],
                    "expected_result": "상위 타자 순위",
                    "error": None
                }
            
            # JSON 파싱
            analysis_data = json.loads(json_content)
            
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
            
            # 질문 유형에 따른 스마트 폴백
            query_lower = query.lower()
            
            # 선수명 추출 시도
            import re
            korean_names = re.findall(r'[가-힣]{2,4}', query)
            potential_player_name = korean_names[0] if korean_names else None
            
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
                            "stat_name": "ops",
                            "year": 2025,
                            "position": "batting",
                            "limit": 10
                        }
                    )
                    analysis = "통산 기록 관련 질문으로 판단하여 상위 타자 조회"
                    
            # 경기 일정/결과 질문 감지
            elif any(word in query_lower for word in ["경기", "일정", "결과", "어린이날", "한국시리즈", "시범경기", "언제부터", "우승"]):
                # 날짜 추출 시도
                import re
                date_patterns = [
                    r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',  # 2023년 5월 5일
                    r'(\d{4})-(\d{1,2})-(\d{1,2})',  # 2023-05-05
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
                    # 연도 추출
                    year_match = re.search(r'(\d{4})', query)
                    year = int(year_match.group(1)) if year_match else 2025
                    
                    fallback_tool = ToolCall(
                        tool_name="get_games_by_date", 
                        parameters={
                            "date": f"{year}-03-01"  # 기본값으로 시즌 시작 시점
                        }
                    )
                    analysis = f"{year}년 경기 정보 조회"
                    
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
                    fallback_tool = ToolCall(
                        tool_name="get_leaderboard",
                        parameters={
                            "stat_name": "home_runs",
                            "year": 2025,
                            "position": "batting",
                            "limit": 10
                        }
                    )
                    analysis = "최고 기록 관련 질문으로 판단하여 홈런 순위 조회"
                    
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
                    fallback_tool = ToolCall(
                        tool_name="get_leaderboard",
                        parameters={
                            "stat_name": "era",
                            "year": 2025,
                            "position": "pitching",
                            "limit": 10
                        }
                    )
                    analysis = "투수 관련 질문으로 판단하여 ERA 기준 상위 투수 조회"
            
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
                    fallback_tool = ToolCall(
                        tool_name="get_leaderboard", 
                        parameters={
                            "stat_name": "ops",
                            "year": 2025,
                            "position": "batting",
                            "limit": 10
                        }
                    )
                    analysis = "타자 관련 질문으로 판단하여 OPS 기준 상위 타자 조회"
            
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
        tool_results: List[ToolResult], 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        도구 실행 결과를 바탕으로 검증된 답변을 생성합니다.
        """
        logger.info(f"[BaseballAgent] Generating verified answer with {len(tool_results)} tool results")
        
        # 도구 실행 결과를 텍스트로 변환
        tool_data_summary = []
        data_sources = []
        
        for i, result in enumerate(tool_results):
            if result.success:
                tool_data_summary.append(f"도구 {i+1} 결과: {result.message}")
                tool_data_summary.append(f"데이터: {json.dumps(result.data, ensure_ascii=False, indent=2)}")
                data_sources.append({
                    "tool": result.data.get("source", "database"),
                    "verified": True,
                    "data_points": len(result.data) if isinstance(result.data, list) else 1
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
당신은 KBO 리그 데이터 분석가 'BEGA'입니다. 다음 원칙을 반드시 준수하세요:

### 핵심 원칙 (절대 위반 금지)
1. **오직 아래 제공된 실제 DB 조회 결과만 사용하세요**
2. **데이터가 없으면 "제공된 데이터에서 확인할 수 없습니다"라고 명확히 답변하세요**  
3. **절대로 추측하거나 일반적인 야구 지식을 사용하지 마세요**
4. **모든 통계 수치는 아래 DB 결과에서만 인용하세요**

사용자 질문: {query}

실제 DB 조회 결과:
{chr(10).join(tool_data_summary)}

### 답변 생성 규칙
1. 자연스러운 톤: 마치 야구 전문가가 답변하는 것처럼 자연스럽고 친근하게 작성
2. 범위 명시: 필요시 분석 범위(예: 2025년 정규시즌 기준) 자연스럽게 포함
3. 근거 제시: "최신 기록을 확인해보니...", "현재 시즌 기준으로..." 등 자연스러운 표현 사용
4. 데이터 부족 시: "죄송하지만 해당 정보를 찾을 수 없습니다" 등 친근한 표현 사용
5. 지표 설명: 복잡한 지표는 한글로 쉽게 설명하되 자연스럽게 포함
   - OPS (출루율+장타율), wRC+ (조정 득점 생산력), ERA- (조정 방어율), WAR (대체 선수 대비 승수) 등
6. 정확성 우선: 확실한 정보만 제공하되 딱딱하지 않은 톤으로 전달

위 DB 조회 결과만을 사용하여 정확하고 간결한 답변을 작성하세요.
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