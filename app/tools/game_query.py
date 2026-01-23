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
        
        # 기본 매핑 (폴백용)
        self.TEAM_CODE_TO_NAME = {
            "HT": "KIA 타이거즈", "KIA": "KIA 타이거즈",
            "LG": "LG 트윈스",
            "OB": "두산 베어스",
            "LT": "롯데 자이언츠",
            "SS": "삼성 라이온즈",
            "WO": "키움 히어로즈",
            "HH": "한화 이글스",
            "KT": "KT 위즈",
            "NC": "NC 다이노스",
            "SSG": "SSG 랜더스",
            "SK": "SSG 랜더스"
        }

        self.NAME_TO_CODE = {
            "KIA": "HT", "기아": "HT", "KIA 타이거즈": "HT", "타이거즈": "HT", "HT": "HT",
            "LG": "LG", "LG 트윈스": "LG", "트윈스": "LG",
            "두산": "OB", "OB": "OB", "두산 베어스": "OB", "베어스": "OB",
            "롯데": "LT", "LT": "LT", "롯데 자이언츠": "LT", "자이언츠": "LT",
            "삼성": "SS", "SS": "SS", "삼성 라이온즈": "SS", "라이온즈": "SS",
            "키움": "WO", "WO": "WO", "키움 히어로즈": "WO", "히어로즈": "WO", "우리": "WO",
            "한화": "HH", "HH": "HH", "한화 이글스": "HH", "이글스": "HH",
            "KT": "KT", "KT 위즈": "KT", "위즈": "KT",
            "NC": "NC", "NC 다이노스": "NC", "다이노스": "NC",
            "SSG": "SSG", "SK": "SSG", "SSG 랜더스": "SSG", "랜더스": "SSG"
        }
        
        # DB에서 최신 매핑 로드
        self._load_team_mappings()

    def _load_team_mappings(self):
        """OCI DB의 teams 테이블과 franchise_id를 활용하여 팀 매핑 정보를 동적으로 로드합니다."""
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            query = """
                SELECT team_id, team_name, franchise_id, founded_year 
                FROM teams 
                WHERE franchise_id IS NOT NULL 
                ORDER BY franchise_id, founded_year DESC;
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if rows:
                logger.info(f"[GameQuery] Syncing {len(rows)} franchise entries from OCI")
                
                # 프랜차이즈별로 그룹화
                franchises = {}
                for row in rows:
                    f_id = row['franchise_id']
                    if f_id not in franchises:
                        franchises[f_id] = []
                    franchises[f_id].append(row)
                
                for f_id, members in franchises.items():
                    # 1. 현대적 브랜드명 선정 (DESC 정렬이므로 첫 번째가 최신)
                    modern_team = members[0]
                    modern_name = modern_team['team_name']
                    
                    # 2. 경기(game) 테이블 매핑 원칙: 
                    # SSG는 'SSG' 사용, KIA는 'HT' 사용, 키움은 'WO' 사용 확인됨
                    if modern_name == "SSG 랜더스":
                        target_code = "SSG"
                    elif modern_name == "KIA 타이거즈":
                        target_code = "HT"
                    else:
                        # 기본적으로는 members 중 가장 오래된 혹은 안정된 ID 선택 시도
                        target_code = next((m['team_id'] for m in members if m['team_id'] in ['HT', 'WO', 'OB', 'SS', 'LT', 'HH', 'KT', 'NC', 'LG']), modern_team['team_id'])
                    
                    # 3. 매핑 데이터 업데이트
                    for member in members:
                        m_id = member['team_id']
                        m_name = member['team_name']
                        
                        self.NAME_TO_CODE[m_id] = target_code
                        self.NAME_TO_CODE[m_name] = target_code
                        self.NAME_TO_CODE[m_name.split()[0]] = target_code
                        
                        self.TEAM_CODE_TO_NAME[m_id] = modern_name
                        self.TEAM_CODE_TO_NAME[target_code] = modern_name
                
                logger.info("[GameQuery] Game Team codes synchronized using OCI franchise IDs.")
            
            cursor.close()
        except Exception as e:
            logger.warning(f"[GameQuery] Dynamic mapping failed from OCI: {e}. Using defaults.")

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
                game_id = game_dict['game_id']
                
                # 1. 이닝별 점수 조회
                inning_query = """
                    SELECT inning_number, home_score, away_score 
                    FROM game_inning_scores 
                    WHERE game_id = %s 
                    ORDER BY inning_number;
                """
                cursor.execute(inning_query, (game_id,))
                innings = cursor.fetchall()
                
                box_score = {
                    "game_id": game_id,
                    "away_r": game_dict.get('away_score', 0),
                    "home_r": game_dict.get('home_score', 0),
                    "away_h": 0, "home_h": 0,  # 타격 스탯에서 집계 필요
                    "away_e": 0, "home_e": 0   # 실책 정보는 현재 스키마에 없으면 0 처리
                }
                
                # 이닝 점수 매핑
                for inning in innings:
                    idx = inning['inning_number']
                    box_score[f'away_{idx}'] = inning['away_score']
                    box_score[f'home_{idx}'] = inning['home_score']
                    
                game_dict['box_score'] = box_score
                
                # 2. 타격 기록 요약 (안타 수 집계 등)
                batting_query = """
                    SELECT team_code, COUNT(*) as hits
                    FROM game_batting_stats
                    WHERE game_id = %s AND hit_type IS NOT NULL
                    GROUP BY team_code
                """ 
                # Note: hit_type 컬럼이 있는지 확인 필요. 
                # 단순 안타수 집계가 어렵다면 game_batting_stats에서 hits 컬럼을 sum 
                # (테이블 구조 확인이 안되므로 일반적인 구조 가정: hits 컬럼 존재 시)
                
                stats_check_query = """
                    SELECT team_code, SUM(hits) as total_hits, SUM(rbi) as total_rbi
                    FROM game_batting_stats
                    WHERE game_id = %s
                    GROUP BY team_code
                """
                try:
                    cursor.execute(stats_check_query, (game_id,))
                    team_stats = cursor.fetchall()
                    for stat in team_stats:
                        # team_code가 home_team인지 away_team인지 확인
                        # (DB에 저장된 team_code와 game 테이블의 팀 코드가 일치한다고 가정)
                        # normalize를 통해 비교
                        t_code = self._normalize_team_name(stat['team_code'])
                        if t_code == self._normalize_team_name(game_dict.get('home_team', '')):
                            box_score['home_h'] = stat['total_hits']
                        else:
                            box_score['away_h'] = stat['total_hits']
                except Exception:
                    # 컬럼이 없거나 오류 발생시 0으로 유지 (로그 생략 가능)
                    pass

                result["games"].append(game_dict)
            
            result["found"] = True
            result["total_games"] = len(result["games"])
            logger.info(f"[GameQuery] Found {len(result['games'])} games (stats aggregated)")
            
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
            
            # 파라미터 순서: [CASE_WHEN_team1, CASE_WHEN_team2] + [WHERE_params] + [LIMIT]
            games_params = [team1_normalized, team2_normalized] + query_params + [limit]
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
    
    def get_team_ranking(self, year: int = 2025) -> Dict[str, Any]:
        """
        특정 연도의 팀 순위를 조회합니다.
        [실제 구현] 이 함수는 GET /team/ranking API를 호출해야 합니다.
        
        Args:
            year: 조회할 시즌 년도 (기본값: 2025)
            
        Returns:
            팀 순위 정보
        """
        logger.info(f"[GameQuery] Getting team ranking for {year}")
        
        # NOTE: 이것은 모의 데이터입니다. 실제로는 API를 호출해야 합니다.
        # KB V2.0 문서의 "3.1 2025 정규시즌 최종 순위 및 승률표" 기반
        mock_ranking_2025 = [
            {"rank": 1, "team_name": "LG 트윈스", "wins": 85, "losses": 56, "draws": 3, "win_rate": 0.603, "games_behind": 0.0},
            {"rank": 2, "team_name": "한화 이글스", "wins": 83, "losses": 57, "draws": 4, "win_rate": 0.593, "games_behind": 1.5},
            {"rank": 3, "team_name": "SSG 랜더스", "wins": 75, "losses": 65, "draws": 4, "win_rate": 0.536, "games_behind": 9.5},
            {"rank": 4, "team_name": "삼성 라이온즈", "wins": 74, "losses": 68, "draws": 2, "win_rate": 0.521, "games_behind": 11.5},
            {"rank": 5, "team_name": "NC 다이노스", "wins": 71, "losses": 67, "draws": 6, "win_rate": 0.514, "games_behind": 12.5},
            {"rank": 6, "team_name": "KT 위즈", "wins": 70, "losses": 72, "draws": 2, "win_rate": 0.493, "games_behind": 15.5},
            {"rank": 7, "team_name": "롯데 자이언츠", "wins": 68, "losses": 74, "draws": 2, "win_rate": 0.479, "games_behind": 17.5},
            {"rank": 8, "team_name": "KIA 타이거즈", "wins": 65, "losses": 76, "draws": 3, "win_rate": 0.461, "games_behind": 20.0},
            {"rank": 9, "team_name": "두산 베어스", "wins": 61, "losses": 80, "draws": 3, "win_rate": 0.433, "games_behind": 24.0},
            {"rank": 10, "team_name": "키움 히어로즈", "wins": 58, "losses": 85, "draws": 1, "win_rate": 0.406, "games_behind": 28.0},
        ]
        
        result = {
            "year": year,
            "ranking": [],
            "found": False,
            "error": None
        }
        
        if year == 2025:
            result["ranking"] = mock_ranking_2025
            result["found"] = True
            logger.info(f"[GameQuery] Returned mock ranking for {year}")
        else:
            result["error"] = f"{year}년의 순위 데이터는 현재 사용할 수 없습니다."
            logger.warning(f"[GameQuery] No ranking data available for {year}")
        
        return result

    def get_season_final_game_date(self, year: int, league_type: str = 'korean_series') -> Dict[str, Any]:
        """
        특정 시즌의 마지막 경기 날짜를 조회합니다.
        
        Args:
            year: 시즌 년도
            league_type: 'regular_season' 또는 'korean_series' (기본값)
            
        Returns:
            마지막 경기 날짜 정보
        """
        logger.info(f"[GameQuery] Getting final game date for {year} {league_type}")
        
        result = {
            "year": year,
            "league_type": league_type,
            "final_game_date": None,
            "found": False,
            "error": None
        }
        
        # 리그 타입에 따른 league_type_code 매핑
        league_code_map = {
            'regular_season': 0,
            'korean_series': 5
        }
        
        league_code = league_code_map.get(league_type)
        if league_code is None:
            result["error"] = f"잘못된 리그 타입입니다: {league_type}"
            return result
            
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            query = """
                SELECT MAX(g.game_date) as final_game_date
                FROM game g
                LEFT JOIN kbo_seasons ks ON g.season_id = ks.season_id
                WHERE ks.season_year = %s
                  AND ks.league_type_code = %s
                  AND g.game_status = 'COMPLETED';
            """
            
            cursor.execute(query, (year, league_code))
            row = cursor.fetchone()
            
            if row and row['final_game_date']:
                final_date = row['final_game_date']
                # datetime.date 객체를 YYYY-MM-DD 형식의 문자열로 변환
                result["final_game_date"] = final_date.strftime('%Y-%m-%d')
                result["found"] = True
                logger.info(f"[GameQuery] Found final game date: {result['final_game_date']}")
            else:
                logger.warning(f"[GameQuery] No final game found for {year} {league_type}")
                
        except Exception as e:
            logger.error(f"[GameQuery] Final game date query error: {e}")
            result["error"] = f"마지막 경기 날짜 조회 오류: {e}"
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
        [실제 구현] 이 함수는 GET /player/info/{id} API와 연동되어야 합니다.
        
        Args:
            player_name: 선수명
            date: 특정 경기 날짜 (선택적)
            recent_games: 최근 N경기 (기본값: 5)
            
        Returns:
            선수별 경기 성적
        """
        logger.info(f"[GameQuery] Player game performance: {player_name}, Date: {date}, Recent: {recent_games}")
        
        # NOTE: 이것은 모의 데이터입니다. 실제로는 API를 호출하여
        #       선수의 최근 경기 기록, 당일 라인업 포함 여부 등을 가져와야 합니다.
        
        result = {
            "player_name": player_name,
            "date_filter": date,
            "is_in_lineup_today": True, # 모의 데이터
            "performances": [],
            "found": False,
            "total_games": 0,
            "error": None,
            "message": ""
        }

        # '안현민' 선수에 대한 모의 데이터 생성
        if "안현민" in player_name:
            result["found"] = True
            result["total_games"] = 5
            result["performances"] = [
                {"date": "2025-10-01", "opponent": "LG", "at_bats": 4, "hits": 2, "home_runs": 1, "rbi": 3},
                {"date": "2025-09-29", "opponent": "SSG", "at_bats": 5, "hits": 3, "home_runs": 0, "rbi": 1},
                {"date": "2025-09-28", "opponent": "NC", "at_bats": 3, "hits": 1, "home_runs": 0, "rbi": 0},
                {"date": "2025-09-27", "opponent": "두산", "at_bats": 4, "hits": 0, "home_runs": 0, "rbi": 0},
                {"date": "2025-09-25", "opponent": "KIA", "at_bats": 4, "hits": 2, "home_runs": 1, "rbi": 2},
            ]
            logger.info(f"Returned mock performance data for {player_name}")
        else:
            result["message"] = f"'{player_name}' 선수의 최근 경기 성적을 찾을 수 없습니다."
            logger.warning(f"No mock performance data available for {player_name}")

        return result
    
    def get_game_lineup(
        self, 
        game_id: str = None, 
        date: str = None,
        team_name: str = None
    ) -> Dict[str, Any]:
        """
        특정 경기의 선발 라인업을 조회합니다.
        
        Args:
            game_id: 경기 고유 ID
            date: 경기 날짜 (YYYY-MM-DD)
            team_name: 팀명
            
        Returns:
            라인업 정보 결과
        """
        logger.info(f"[GameQuery] Lineup query - ID: {game_id}, Date: {date}, Team: {team_name}")
        
        result = {
            "query_params": {
                "game_id": game_id,
                "date": date,
                "team_name": team_name
            },
            "lineups": [],
            "found": False,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 1. 경기 ID 찾기 (ID가 없는 경우)
            if not game_id and date:
                where_clause = "DATE(game_date) = %s"
                params = [date]
                if team_name:
                    normalized_team = self._normalize_team_name(team_name)
                    where_clause += " AND (home_team = %s OR away_team = %s)"
                    params.extend([normalized_team, normalized_team])
                
                cursor.execute(f"SELECT game_id FROM game WHERE {where_clause} LIMIT 1", params)
                row = cursor.fetchone()
                if row:
                    game_id = row['game_id']
            
            if not game_id:
                result["error"] = "경기를 찾을 수 없거나 game_id가 제공되지 않았습니다."
                return result
            
            # 2. 라인업 조회 (새로 추가된 player_name, is_starter 컬럼 사용)
            lineup_query = """
                SELECT 
                    team_code,
                    player_name,
                    position,
                    batting_order,
                    is_starter
                FROM game_lineups
                WHERE game_id = %s
            """
            params = [game_id]
            
            if team_name:
                normalized_team = self._normalize_team_name(team_name)
                lineup_query += " AND team_code = %s"
                params.append(normalized_team)
                
            lineup_query += " ORDER BY team_code, batting_order"
            
            cursor.execute(lineup_query, params)
            rows = cursor.fetchall()
            
            if rows:
                result["lineups"] = [dict(row) for row in rows]
                result["found"] = True
                
                # 팀 코드를 이름으로 변환
                for entry in result["lineups"]:
                    entry["team_name"] = self.get_team_name(entry["team_code"])
                    
                logger.info(f"[GameQuery] Found {len(rows)} lineup entries for game {game_id}")
            else:
                logger.warning(f"[GameQuery] No lineup found for game {game_id}")
                
        except Exception as e:
            logger.error(f"[GameQuery] Lineup query error: {e}")
            result["error"] = f"라인업 조회 오류: {e}"
        finally:
            if 'cursor' in locals():
                cursor.close()
                
        return result

    def get_team_last_game_date(self, team_name: str, year: int, league_type: str = 'regular_season') -> Dict[str, Any]:
        """
        특정 팀의 마지막 경기 날짜를 조회합니다.
        
        Args:
            team_name: 팀명
            year: 시즌 년도
            league_type: 'regular_season' 또는 'korean_series' 등
            
        Returns:
            팀의 마지막 경기 날짜 정보
        """
        logger.info(f"[GameQuery] Getting last game date for {team_name} in {year} ({league_type})")
        
        normalized_team = self._normalize_team_name(team_name)
        
        result = {
            "team_name": team_name,
            "team_id": normalized_team,
            "year": year,
            "last_game_date": None,
            "found": False,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 리그 타입 코드 매핑
            league_code_map = {'regular_season': 0, 'korean_series': 5}
            league_code = league_code_map.get(league_type, 0)
            
            query = """
                SELECT MAX(g.game_date) as last_game_date
                FROM game g
                LEFT JOIN kbo_seasons ks ON g.season_id = ks.season_id
                WHERE ks.season_year = %s
                  AND ks.league_type_code = %s
                  AND (g.home_team = %s OR g.away_team = %s)
                  AND g.game_status = 'COMPLETED';
            """
            
            cursor.execute(query, (year, league_code, normalized_team, normalized_team))
            row = cursor.fetchone()
            
            if row and row['last_game_date']:
                result["last_game_date"] = row['last_game_date'].strftime('%Y-%m-%d')
                result["found"] = True
                logger.info(f"[GameQuery] Found last game for {team_name}: {result['last_game_date']}")
            else:
                logger.warning(f"[GameQuery] No last game found for {team_name} in {year} {league_type}")
                
        except Exception as e:
            logger.error(f"[GameQuery] Team last game query error: {e}")
            result["error"] = str(e)
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