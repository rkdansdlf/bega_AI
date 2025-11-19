"""
경기 데이터 조회를 위한 전용 도구입니다.

이 도구는 box_score 테이블과 관련 경기 테이블들을 조회하여
경기별 상세 정보, 팀 간 대결 기록, 날짜별 경기 등을 제공합니다.
"""

import logging
from typing import Dict, List, Any, Optional
from psycopg2.extensions import connection as PgConnection
import psycopg2.extras
from datetime import datetime

logger = logging.getLogger(__name__)

class GameQueryTool:
    """
    경기 데이터 전용 조회 도구
    
    이 도구는 다음 원칙을 따릅니다:
    1. box_score, game, game_summary 테이블을 활용한 경기 데이터 조회
    2. 정확한 경기 정보와 통계 제공
    3. 추측이나 해석 없이 실제 DB 데이터만 반환
    """
    
    def __init__(self, connection: PgConnection):
        self.connection = connection
        
        self.TEAM_CODE_TO_NAME = {
            "KIA": "KIA 타이거즈",
            "LG": "LG 트윈스",
            "OB": "두산 베어스",
            "LT": "롯데 자이언츠",
            "SS": "삼성 라이온즈",
            "WO": "키움 히어로즈",
            "HH": "한화 이글스",
            "KT": "KT 위즈",
            "NC": "NC 다이노스",
            "SK": "SSG 랜더스"
        }

        self.NAME_TO_CODE = {
            "KIA": "KIA", "기아": "KIA", "KIA 타이거즈": "KIA", "타이거즈": "KIA",
            "LG": "LG", "LG 트윈스": "LG", "트윈스": "LG",
            "두산": "OB", "OB": "OB", "두산 베어스": "OB", "베어스": "OB",
            "롯데": "LT", "LT": "LT", "롯데 자이언츠": "LT", "자이언츠": "LT",
            "삼성": "SS", "SS": "SS", "삼성 라이온즈": "SS", "라이온즈": "SS",
            "키움": "WO", "WO": "WO", "키움 히어로즈": "WO", "히어로즈": "WO", "우리": "WO",
            "한화": "HH", "HH": "HH", "한화 이글스": "HH", "이글스": "HH",
            "KT": "KT", "KT 위즈": "KT", "위즈": "KT",
            "NC": "NC", "NC 다이노스": "NC", "다이노스": "NC",
            "SSG": "SK", "SK": "SK", "SSG 랜더스": "SK", "랜더스": "SK"
        }

    def get_team_name(self, team_code: str) -> str:
        return self.TEAM_CODE_TO_NAME.get(team_code, team_code)
    
    def get_team_code(self, team_input: str) -> str:
        return self.NAME_TO_CODE.get(team_input, team_input)
    
    def _normalize_team_name(self, team_name: str) -> str:
        """팀명을 정규화합니다."""
        team_name = team_name.strip()
        
        # for standard_name, variations in self.team_mapping.items():
        #     if team_name in variations:
        #         return standard_name
        
        # return team_name
        return self.get_team_code(team_name)
    
    def _format_game_response(self, game_dict: Dict) -> Dict:
        """
        경기 데이터에 팀 정식 명칭 추가
        
        Args:
            game_dict: 원본 경기 데이터
            
        Returns:
            팀 이름이 추가된 경기 데이터
        """
        if 'home_team' in game_dict:
            game_dict['home_team_code'] = game_dict['home_team']
            game_dict['home_team_name'] = self.get_team_name(game_dict['home_team'])
        
        if 'away_team' in game_dict:
            game_dict['away_team_code'] = game_dict['away_team']
            game_dict['away_team_name'] = self.get_team_name(game_dict['away_team'])
        
        if 'winning_team' in game_dict and game_dict['winning_team']:
            game_dict['winning_team_code'] = game_dict['winning_team']
            game_dict['winning_team_name'] = self.get_team_name(game_dict['winning_team'])
        
        return game_dict
    
    def get_game_box_score(
        self, 
        game_id: str = None, 
        date: str = None,
        home_team: str = None,
        away_team: str = None
    ) -> Dict[str, Any]:
        """
        특정 경기의 박스스코어를 조회합니다.
        
        Args:
            game_id: 경기 고유 ID (우선순위 높음)
            date: 경기 날짜 (YYYY-MM-DD)
            home_team: 홈팀명
            away_team: 원정팀명
            
        Returns:
            경기 박스스코어 결과
        """
        logger.info(f"[GameQuery] Box score query - ID: {game_id}, Date: {date}, Teams: {home_team} vs {away_team}")
        
        result = {
            "query_params": {
                "game_id": game_id,
                "date": date, 
                "home_team": home_team,
                "away_team": away_team
            },
            "games": [],
            "found": False,
            "total_games": 0,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 쿼리 조건 구성
            where_conditions = []
            query_params = []
            
            if game_id:
                # game_id가 있으면 최우선 사용
                where_conditions.append("g.game_id = %s")
                query_params.append(game_id)
            else:
                # 다른 조건들로 검색
                if date:
                    where_conditions.append("DATE(g.game_date) = %s")
                    query_params.append(date)
                
                if home_team:
                    normalized_home = self._normalize_team_name(home_team)
                    where_conditions.append("g.home_team = %s")
                    query_params.append(normalized_home)
                
                if away_team:
                    normalized_away = self._normalize_team_name(away_team)
                    where_conditions.append("g.away_team = %s")
                    query_params.append(normalized_away)
            
            if not where_conditions:
                result["error"] = "검색 조건이 필요합니다 (game_id, date, 또는 팀명)"
                return result
            
            # 기본 경기 정보 조회 쿼리 (실제 스키마에 맞게 수정)
            game_query = f"""
                SELECT 
                    g.game_id,
                    g.game_date,
                    g.home_team,
                    g.away_team,
                    g.home_score,
                    g.away_score,
                    g.game_status,
                    g.stadium,
                    g.winning_team,
                    g.home_pitcher,
                    g.away_pitcher
                FROM game g
                WHERE {' AND '.join(where_conditions)}
                ORDER BY g.game_date DESC, g.game_id
                LIMIT 10;
            """
            
            cursor.execute(game_query, query_params)
            games = cursor.fetchall()
            
            if not games:
                result["error"] = "조건에 맞는 경기를 찾을 수 없습니다"
                return result
            
            # 각 경기의 박스스코어 상세 정보 조회
            for game in games:
                game_dict = dict(game)
                game_dict = self._format_game_response(game_dict)
                
                # 박스스코어 상세 정보 조회 (실제 스키마 기준)
                box_score_query = """
                    SELECT 
                        game_id,
                        stadium,
                        crowd,
                        start_time,
                        end_time,
                        game_time,
                        away_record,
                        home_record,
                        away_1, away_2, away_3, away_4, away_5, away_6, away_7, away_8, away_9,
                        home_1, home_2, home_3, home_4, home_5, home_6, home_7, home_8, home_9,
                        away_r, away_h, away_e,
                        home_r, home_h, home_e
                    FROM box_score 
                    WHERE game_id = %s;
                """
                
                cursor.execute(box_score_query, (game_dict['game_id'],))
                box_score = cursor.fetchone()
                
                if box_score:
                    game_dict['box_score'] = dict(box_score)
                else:
                    game_dict['box_score'] = {}
                
                # 게임 요약 정보 조회 (실제 스키마 기준)
                summary_query = """
                    SELECT 
                        summary_type,
                        player_name,
                        detail_text
                    FROM game_summary
                    WHERE game_id = %s;
                """
                
                cursor.execute(summary_query, (game_dict['game_id'],))
                summaries = cursor.fetchall()
                
                if summaries:
                    game_dict['summary'] = [dict(summary) for summary in summaries]
                else:
                    game_dict['summary'] = []
                
                result["games"].append(game_dict)
            
            result["found"] = True
            result["total_games"] = len(result["games"])
            logger.info(f"[GameQuery] Found {len(result['games'])} games")
            
        except Exception as e:
            logger.error(f"[GameQuery] Box score query error: {e}")
            result["error"] = f"박스스코어 조회 오류: {e}"
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result
    
    def get_games_by_date(self, date: str, team: str = None) -> Dict[str, Any]:
        """
        특정 날짜의 모든 경기를 조회합니다.
        
        Args:
            date: 경기 날짜 (YYYY-MM-DD)
            team: 특정 팀만 조회 (선택적)
            
        Returns:
            해당 날짜 경기 목록
        """
        logger.info(f"[GameQuery] Games by date: {date}, Team: {team}")
        
        result = {
            "date": date,
            "team_filter": team,
            "games": [],
            "found": False,
            "total_games": 0,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 쿼리 조건 구성
            where_conditions = ["DATE(g.game_date) = %s"]
            query_params = [date]
            
            if team:
                normalized_team = self._normalize_team_name(team)
                where_conditions.append("(g.home_team = %s OR g.away_team = %s)")
                query_params.extend([normalized_team, normalized_team])
            
            query = f"""
                SELECT 
                    g.game_id,
                    g.game_date,
                    g.home_team,
                    g.away_team,
                    g.home_score,
                    g.away_score,
                    g.game_status,
                    g.stadium,
                    g.winning_team,
                    g.home_pitcher,
                    g.away_pitcher
                FROM game g
                WHERE {' AND '.join(where_conditions)}
                ORDER BY g.game_date, g.game_id;
            """
            
            cursor.execute(query, query_params)
            games = cursor.fetchall()
            
            if games:
                result["games"] = [dict(game) for game in games]
                result["found"] = True
                result["total_games"] = len(games)
                logger.info(f"[GameQuery] Found {len(games)} games on {date}")
            else:
                logger.warning(f"[GameQuery] No games found on {date}")
                
        except Exception as e:
            logger.error(f"[GameQuery] Date query error: {e}")
            result["error"] = f"날짜별 경기 조회 오류: {e}"
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result
    
    def get_head_to_head(
        self, 
        team1: str, 
        team2: str, 
        year: int = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        두 팀 간의 직접 대결 기록을 조회합니다.
        
        Args:
            team1: 팀1 이름
            team2: 팀2 이름
            year: 시즌 년도 (선택적)
            limit: 최근 N경기 (기본값: 10)
            
        Returns:
            팀 간 대결 기록
        """
        logger.info(f"[GameQuery] Head to head: {team1} vs {team2}, Year: {year}")
        
        team1_normalized = self._normalize_team_name(team1)
        team2_normalized = self._normalize_team_name(team2)
        team1_name = self.get_team_name(team1_normalized)
        team2_name = self.get_team_name(team2_normalized)
        
        result = {
            "team1": team1_name,
            "team2": team2_name,
            "year": year,
            "games": [],
            "summary": {
                "total_games": 0,
                "team1_wins": 0,
                "team2_wins": 0,
                "draws": 0
            },
            "found": False,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 쿼리 조건 구성
            where_conditions = [
                "((g.home_team = %s AND g.away_team = %s) OR "
                "(g.home_team = %s AND g.away_team = %s))"
            ]
            query_params = [team1_normalized, team2_normalized, team2_normalized, team1_normalized]
            
            if year:
                where_conditions.append("EXTRACT(YEAR FROM g.game_date) = %s")
                query_params.append(year)
            
            # 상세 경기 기록 조회
            games_query = f"""
                SELECT 
                    g.game_id,
                    g.game_date,
                    g.home_team,
                    g.away_team,
                    g.home_score,
                    g.away_score,
                    g.game_status,
                    g.stadium,
                    g.winning_team,
                    CASE 
                        WHEN g.winning_team = %s THEN 'team1_win'
                        WHEN g.winning_team = %s THEN 'team2_win'
                        WHEN g.home_score = g.away_score THEN 'draw'
                        ELSE 'unknown'
                    END as game_result
                FROM game g
                WHERE {' AND '.join(where_conditions)}
                AND g.game_status = 'COMPLETED'
                ORDER BY g.game_date DESC
                LIMIT %s;
            """
            
            games_params = query_params + [team1_normalized, team2_normalized, limit]
            cursor.execute(games_query, games_params)
            games = cursor.fetchall()
            
            if games:
                result["games"] = [self._format_game_response(dict(game)) for game in games]
                result["found"] = True
                
                # 요약 통계 계산
                result["summary"]["total_games"] = len(games)
                for game in games:
                    if game['game_result'] == 'team1_win':
                        result["summary"]["team1_wins"] += 1
                    elif game['game_result'] == 'team2_win':
                        result["summary"]["team2_wins"] += 1
                    elif game['game_result'] == 'draw':
                        result["summary"]["draws"] += 1
                
                logger.info(f"[GameQuery] Found {len(games)} head-to-head games")
            else:
                logger.warning(f"[GameQuery] No head-to-head games found")
                
        except Exception as e:
            logger.error(f"[GameQuery] Head-to-head query error: {e}")
            result["error"] = f"팀 간 대결 기록 조회 오류: {e}"
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result
    
    def get_schedule(
        self, 
        start_date: str, 
        end_date: str, 
        team: str = None
    ) -> Dict[str, Any]:
        """
        특정 기간 동안의 경기 일정을 조회합니다.
        
        Args:
            start_date: 조회 시작 날짜 (YYYY-MM-DD)
            end_date: 조회 종료 날짜 (YYYY-MM-DD)
            team: 특정 팀의 일정만 조회 (선택적)
            
        Returns:
            경기 일정 목록
        """
        logger.info(f"[GameQuery] Schedule query: {start_date} to {end_date}, Team: {team}")
        
        result = {
            "start_date": start_date,
            "end_date": end_date,
            "team_filter": team,
            "games": [],
            "found": False,
            "total_games": 0,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_conditions = ["g.game_date BETWEEN %s AND %s"]
            query_params = [start_date, end_date]
            
            if team:
                normalized_team = self._normalize_team_name(team)
                where_conditions.append("(g.home_team = %s OR g.away_team = %s)")
                query_params.extend([normalized_team, normalized_team])
            
            query = f"""
                SELECT 
                    g.game_id,
                    g.game_date,
                    g.home_team,
                    g.away_team,
                    g.game_status,
                    g.stadium
                FROM game g
                WHERE {' AND '.join(where_conditions)}
                ORDER BY g.game_date, g.game_id;
            """
            
            cursor.execute(query, query_params)
            games = cursor.fetchall()
            
            if games:
                result["games"] = [self._format_game_response(dict(game)) for game in games]
                result["found"] = True
                result["total_games"] = len(games)
                logger.info(f"[GameQuery] Found {len(games)} scheduled games")
            else:
                logger.warning(f"[GameQuery] No scheduled games found in the period")
                
        except Exception as e:
            logger.error(f"[GameQuery] Schedule query error: {e}")
            result["error"] = f"경기 일정 조회 오류: {e}"
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result
    
    def get_player_game_performance(
        self, 
        player_name: str, 
        date: str = None,
        recent_games: int = 5
    ) -> Dict[str, Any]:
        """
        특정 선수의 개별 경기 성적을 조회합니다.
        
        Args:
            player_name: 선수명
            date: 특정 경기 날짜 (선택적)
            recent_games: 최근 N경기 (기본값: 5)
            
        Returns:
            선수별 경기 성적
        """
        logger.info(f"[GameQuery] Player game performance: {player_name}, Date: {date}")
        
        result = {
            "player_name": player_name,
            "date_filter": date,
            "performances": [],
            "found": False,
            "total_games": 0,
            "error": None,
            "message": ""
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 선수별 경기 성적 테이블이 있는지 확인
            # 실제 스키마에 따라 조정 필요
            table_check_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name IN ('player_game_stats', 'player_game_batting', 'player_game_pitching')
                AND table_schema = 'public';
            """
            
            cursor.execute(table_check_query)
            available_tables = [row[0] for row in cursor.fetchall()]
            
            if not available_tables:
                result["error"] = "선수별 경기 성적 테이블을 찾을 수 없습니다"
                result["message"] = "현재 데이터베이스에는 개별 경기 성적 데이터가 없습니다. 시즌 통계만 이용 가능합니다."
                return result
            
            # 사용 가능한 테이블에서 데이터 조회
            # 이 부분은 실제 스키마에 맞게 구현 필요
            result["message"] = "개별 경기 성적 기능은 데이터베이스 스키마 확인 후 구현 예정입니다."
            
        except Exception as e:
            logger.error(f"[GameQuery] Player performance query error: {e}")
            result["error"] = f"선수 경기 성적 조회 오류: {e}"
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result
    
    def validate_game_exists(self, game_id: str = None, date: str = None) -> Dict[str, Any]:
        """
        경기 존재 여부를 확인합니다.
        
        Args:
            game_id: 경기 ID
            date: 경기 날짜
            
        Returns:
            경기 존재 여부 확인 결과
        """
        logger.info(f"[GameQuery] Validating game - ID: {game_id}, Date: {date}")
        
        result = {
            "game_id": game_id,
            "date": date,
            "exists": False,
            "game_count": 0,
            "games": [],
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_conditions = []
            query_params = []
            
            if game_id:
                where_conditions.append("game_id = %s")
                query_params.append(game_id)
            
            if date:
                where_conditions.append("DATE(game_date) = %s")
                query_params.append(date)
            
            if not where_conditions:
                result["error"] = "game_id 또는 date 중 하나는 필요합니다"
                return result
            
            query = f"""
                SELECT game_id, game_date, home_team, away_team, game_status
                FROM game 
                WHERE {' AND '.join(where_conditions)}
                ORDER BY game_date;
            """
            
            cursor.execute(query, query_params)
            games = cursor.fetchall()
            
            if games:
                result["exists"] = True
                result["game_count"] = len(games)
                result["games"] = [dict(game) for game in games]
                logger.info(f"[GameQuery] Found {len(games)} matching games")
            else:
                logger.warning(f"[GameQuery] No matching games found")
                
        except Exception as e:
            logger.error(f"[GameQuery] Validation error: {e}")
            result["error"] = f"경기 검증 오류: {e}"
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result