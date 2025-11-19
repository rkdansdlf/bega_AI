"""
직접 데이터베이스 쿼리를 통해 정확한 통계를 조회하는 도구입니다.

이 모듈은 LLM의 환각(hallucination)을 방지하기 위해 
실제 DB에서 정확한 데이터만을 조회하여 반환합니다.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from psycopg2.extensions import connection as PgConnection
import psycopg2.extras

logger = logging.getLogger(__name__)

class DatabaseQueryTool:
    """
    실제 데이터베이스에서 KBO 통계를 직접 조회하는 도구입니다.
    
    이 도구는 다음과 같은 환각 방지 원칙을 따릅니다:
    1. 모든 통계는 실제 DB 쿼리로만 조회
    2. 데이터가 없으면 명확하게 "데이터 없음" 반환 
    3. 추정이나 계산은 실제 DB 필드만 사용
    4. 검색 결과가 비어있으면 절대 추측하지 않음
    """
    
    def __init__(self, connection: PgConnection):
        self.connection = connection
        
        # KBO 팀 매핑 (정확한 DB 데이터와 매칭)
        self.team_mapping = {
            "KIA": "KIA 타이거즈", "기아": "KIA 타이거즈",
            "LG": "LG 트윈스",
            "두산": "두산 베어스", 
            "롯데": "롯데 자이언츠",
            "삼성": "삼성 라이온즈",
            "키움": "키움 히어로즈",
            "한화": "한화 이글스",
            "KT": "KT 위즈",
            "NC": "NC 다이노스",
            "SSG": "SSG 랜더스"
        }
        
        # KBO 포지션 약어 매핑
        self.position_mapping = {
            "지": "지명타자",
            "타": "대타", 
            "주": "대주자",
            "중": "중견수",
            "좌": "좌익수",
            "우": "우익수",
            "一": "1루수",
            "二": "2루수", 
            "三": "3루수",
            "유": "유격수",
            "포": "포수"
        }
    
    def get_player_career_stats(
        self, 
        player_name: str, 
        position: str = "both"  # "batting", "pitching", "both"
    ) -> Dict[str, Any]:
        """
        특정 선수의 통산 통계를 실제 DB에서 계산하여 조회합니다.
        
        Args:
            player_name: 선수명 (부분 매칭 지원)
            position: 조회할 포지션 ("batting", "pitching", "both")
            
        Returns:
            실제 DB에서 계산된 선수 통산 통계 딕셔너리
        """
        logger.info(f"[DatabaseQuery] Querying career stats: {player_name}, {position}")
        
        result = {
            "player_name": player_name,
            "career": True,
            "batting_stats": None,
            "pitching_stats": None,
            "found": False,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 우선 선수가 존재하는지 확인
            check_query = """
                SELECT DISTINCT name FROM player_basic 
                WHERE (LOWER(name) = LOWER(%s) OR LOWER(name) LIKE LOWER(%s))
                LIMIT 5
            """
            cursor.execute(check_query, (player_name, f'%{player_name}%'))
            existing_players = [row[0] for row in cursor.fetchall()]
            
            if not existing_players:
                result["error"] = f"선수 '{player_name}'을(를) 찾을 수 없습니다. 정확한 선수명을 확인해주세요."
                logger.warning(f"[DatabaseQuery] No player found matching: {player_name}")
                return result
            else:
                logger.info(f"[DatabaseQuery] Found matching players: {existing_players}")
                # 정확히 일치하는 선수명이 있으면 그것을 사용, 없으면 첫 번째 결과 사용
                exact_match = next((name for name in existing_players if name.lower() == player_name.lower()), None)
                matched_name = exact_match if exact_match else existing_players[0]
                player_name = matched_name  # 실제 데이터베이스의 선수명으로 업데이트
            
            # 타격 통산 통계 조회 (합계)
            if position in ["batting", "both"]:
                batting_query = """
                    SELECT 
                        pb.name as player_name,
                        COUNT(DISTINCT psb.season) as seasons_played,
                        SUM(psb.games) as total_games,
                        SUM(psb.plate_appearances) as total_pa,
                        SUM(psb.at_bats) as total_ab,
                        SUM(psb.hits) as total_hits,
                        SUM(psb.doubles) as total_doubles,
                        SUM(psb.triples) as total_triples,
                        SUM(psb.home_runs) as total_home_runs,
                        SUM(psb.rbi) as total_rbi,
                        SUM(psb.runs) as total_runs,
                        SUM(psb.walks) as total_walks,
                        SUM(psb.strikeouts) as total_strikeouts,
                        SUM(psb.stolen_bases) as total_stolen_bases,
                        ROUND((CASE WHEN SUM(psb.at_bats) > 0 THEN SUM(psb.hits)::decimal / SUM(psb.at_bats) ELSE 0 END)::numeric, 3) as career_avg,
                        ROUND((CASE WHEN SUM(psb.at_bats + psb.walks + psb.hbp + psb.sacrifice_flies) > 0 
                                   THEN (SUM(psb.hits) + SUM(psb.walks) + SUM(psb.hbp))::decimal / 
                                        SUM(psb.at_bats + psb.walks + psb.hbp + psb.sacrifice_flies) 
                                   ELSE 0 END)::numeric, 3) as career_obp,
                        ROUND((CASE WHEN SUM(psb.at_bats) > 0 
                                   THEN (SUM(psb.hits) + SUM(psb.doubles) + 2*SUM(psb.triples) + 3*SUM(psb.home_runs))::decimal / SUM(psb.at_bats)
                                   ELSE 0 END)::numeric, 3) as career_slg
                    FROM player_season_batting psb
                    JOIN player_basic pb ON psb.player_id = pb.player_id
                    JOIN kbo_seasons ks ON psb.season = ks.season_year
                    WHERE (LOWER(pb.name) = LOWER(%s) OR LOWER(pb.name) LIKE LOWER(%s))
                    AND ks.league_type_code = '0'
                    AND psb.plate_appearances >= 10  -- 최소 기준
                    GROUP BY pb.player_id, pb.name
                    ORDER BY total_home_runs DESC
                    LIMIT 1
                """
                cursor.execute(batting_query, (player_name, f'%{player_name}%'))
                batting_row = cursor.fetchone()
                
                if batting_row:
                    batting_stats = dict(batting_row)
                    # OPS 계산
                    if batting_stats['career_obp'] and batting_stats['career_slg']:
                        batting_stats['career_ops'] = round(batting_stats['career_obp'] + batting_stats['career_slg'], 3)
                    
                    result["batting_stats"] = batting_stats
                    result["found"] = True
                    logger.info(f"[DatabaseQuery] Found career batting stats for {player_name}")
            
            # 투구 통산 통계 조회 (합계 및 평균)
            if position in ["pitching", "both"]:
                pitching_query = """
                    SELECT 
                        pb.name as player_name,
                        COUNT(DISTINCT psp.season) as seasons_played,
                        SUM(psp.games) as total_games,
                        SUM(psp.games_started) as total_games_started,
                        SUM(psp.wins) as total_wins,
                        SUM(psp.losses) as total_losses,
                        SUM(psp.saves) as total_saves,
                        SUM(psp.holds) as total_holds,
                        SUM(psp.innings_pitched) as total_innings_pitched,
                        SUM(psp.strikeouts) as total_strikeouts,
                        SUM(psp.walks_allowed) as total_walks_allowed,
                        SUM(psp.hits_allowed) as total_hits_allowed,
                        SUM(psp.home_runs_allowed) as total_home_runs_allowed,
                        SUM(psp.earned_runs) as total_earned_runs,
                        ROUND((CASE WHEN SUM(psp.innings_pitched) > 0 
                                   THEN (SUM(psp.earned_runs) * 9.0) / SUM(psp.innings_pitched) 
                                   ELSE 0 END)::numeric, 2) as career_era,
                        ROUND((CASE WHEN SUM(psp.innings_pitched) > 0 
                                   THEN (SUM(psp.hits_allowed) + SUM(psp.walks_allowed)) / SUM(psp.innings_pitched) 
                                   ELSE 0 END)::numeric, 2) as career_whip
                    FROM player_season_pitching psp
                    JOIN player_basic pb ON psp.player_id = pb.player_id
                    JOIN kbo_seasons ks ON psp.season = ks.season_year
                    WHERE (LOWER(pb.name) = LOWER(%s) OR LOWER(pb.name) LIKE LOWER(%s))
                    AND ks.league_type_code = '0'
                    AND psp.innings_pitched >= 1  -- 최소 기준
                    GROUP BY pb.player_id, pb.name
                    ORDER BY total_wins DESC
                    LIMIT 1
                """
                cursor.execute(pitching_query, (player_name, f'%{player_name}%'))
                pitching_row = cursor.fetchone()
                
                if pitching_row:
                    result["pitching_stats"] = dict(pitching_row)
                    result["found"] = True
                    logger.info(f"[DatabaseQuery] Found career pitching stats for {player_name}")
                    
        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying career stats: {e}")
            result["error"] = str(e)
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result

    def get_player_season_stats(
        self, 
        player_name: str, 
        year: int, 
        position: str = "both"  # "batting", "pitching", "both"
    ) -> Dict[str, Any]:
        """
        특정 선수의 시즌 통계를 실제 DB에서 조회합니다.
        
        Args:
            player_name: 선수명 (부분 매칭 지원)
            year: 시즌 년도
            position: 조회할 포지션 ("batting", "pitching", "both")
            
        Returns:
            실제 DB에서 조회된 선수 통계 딕셔너리
        """
        logger.info(f"[DatabaseQuery] Querying player stats: {player_name}, {year}, {position}")
        
        result = {
            "player_name": player_name,
            "year": year,
            "batting_stats": None,
            "pitching_stats": None,
            "found": False,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 타격 통계 조회
            if position in ["batting", "both"]:
                batting_query = """
                    SELECT DISTINCT
                        pb.name as player_name, 
                        t.team_name,
                        psb.season,
                        psb.plate_appearances, psb.at_bats, psb.hits, psb.doubles, psb.triples, psb.home_runs,
                        psb.rbi, psb.runs, psb.walks, psb.strikeouts, psb.stolen_bases, psb.caught_stealing,
                        psb.avg, psb.obp, psb.slg, psb.ops, psb.babip
                    FROM player_season_batting psb
                    JOIN player_basic pb ON psb.player_id = pb.player_id
                    JOIN kbo_seasons ks ON psb.season = ks.season_year
                    LEFT JOIN teams t ON psb.team_code = t.team_id
                    WHERE LOWER(pb.name) LIKE LOWER(%s) 
                    AND psb.season = %s 
                    AND ks.league_type_code = '0'
                    ORDER BY psb.plate_appearances DESC
                    LIMIT 1
                """
                cursor.execute(batting_query, (f'%{player_name}%', year))
                batting_row = cursor.fetchone()
                
                if batting_row:
                    result["batting_stats"] = dict(batting_row)
                    result["found"] = True
                    logger.info(f"[DatabaseQuery] Found batting stats for {player_name}")
            
            # 투구 통계 조회  
            if position in ["pitching", "both"]:
                pitching_query = """
                    SELECT DISTINCT
                        pb.name as player_name,
                        t.team_name,
                        psp.season,
                        psp.games, psp.games_started, psp.wins, psp.losses, psp.saves, psp.holds,
                        psp.innings_pitched, psp.hits_allowed, psp.runs_allowed, psp.earned_runs,
                        psp.home_runs_allowed, psp.walks_allowed, psp.strikeouts, psp.era, psp.whip
                    FROM player_season_pitching psp
                    JOIN player_basic pb ON psp.player_id = pb.player_id
                    JOIN kbo_seasons ks ON psp.season = ks.season_year
                    LEFT JOIN teams t ON psp.team_code = t.team_id
                    WHERE LOWER(pb.name) LIKE LOWER(%s) 
                    AND psp.season = %s 
                    AND ks.league_type_code = '0'
                    ORDER BY psp.innings_pitched DESC
                    LIMIT 1
                """
                cursor.execute(pitching_query, (f'%{player_name}%', year))
                pitching_row = cursor.fetchone()
                
                if pitching_row:
                    result["pitching_stats"] = dict(pitching_row)
                    result["found"] = True
                    logger.info(f"[DatabaseQuery] Found pitching stats for {player_name}")
                    
        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying player stats: {e}")
            result["error"] = str(e)
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result
    
    def get_team_leaderboard(
        self, 
        stat_name: str, 
        year: int, 
        position: str, 
        team_filter: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        특정 통계 지표의 리더보드를 실제 DB에서 조회합니다.
        
        Args:
            stat_name: 통계 지표명 (예: "ops", "era", "home_runs")
            year: 시즌 년도
            position: "batting" 또는 "pitching"
            team_filter: 특정 팀만 필터링 (선택적)
            limit: 상위 몇 명까지 조회할지
            
        Returns:
            실제 DB에서 조회된 리더보드 딕셔너리
        """
        logger.info(f"[DatabaseQuery] Querying leaderboard: {stat_name}, {year}, {position}")
        
        result = {
            "stat_name": stat_name,
            "year": year,
            "position": position,
            "team_filter": team_filter,
            "leaderboard": [],
            "found": False,
            "total_qualified_players": 0,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 통계 지표에 따른 쿼리 구성
            if position == "batting":
                # 타격 통계 리더보드
                stat_mapping = {
                    "ops": ("ops", "DESC", 100),  # (컬럼명, 정렬순서, 최소 타석)
                    "타율": ("avg", "DESC", 100),
                    "avg": ("avg", "DESC", 100),
                    "홈런": ("home_runs", "DESC", 100),
                    "home_runs": ("home_runs", "DESC", 100),
                    "타점": ("rbi", "DESC", 100),
                    "rbi": ("rbi", "DESC", 100),
                    "출루율": ("obp", "DESC", 100),
                    "obp": ("obp", "DESC", 100),
                    "장타율": ("slg", "DESC", 100),
                    "slg": ("slg", "DESC", 100),
                    "wrc_plus": ("wrc_plus", "DESC", 100),
                    "war": ("war", "DESC", 100)
                }
                
                if stat_name.lower() not in stat_mapping:
                    result["error"] = f"지원하지 않는 타격 통계: {stat_name}"
                    return result
                
                db_column, sort_order, min_pa = stat_mapping[stat_name.lower()]
                
                # 팀 필터 조건 구성
                team_condition = ""
                params = [year]
                if team_filter:
                    team_name = self.team_mapping.get(team_filter, team_filter)
                    team_condition = "AND team_name = %s"
                    params.append(team_name)
                
                params.extend([min_pa, limit])
                
                query = f"""
                    SELECT 
                        pb.name as player_name, 
                        t.team_name, 
                        psb.{db_column} as stat_value,
                        psb.plate_appearances, psb.avg, psb.obp, psb.slg, psb.ops, psb.home_runs, psb.rbi
                    FROM player_season_batting psb
                    JOIN player_basic pb ON psb.player_id = pb.player_id
                    JOIN kbo_seasons ks ON psb.season = ks.season_year
                    LEFT JOIN teams t ON psb.team_code = t.team_id
                    WHERE psb.season = %s 
                    AND ks.league_type_code = '0'
                    {team_condition}
                    AND psb.plate_appearances >= %s 
                    AND psb.{db_column} IS NOT NULL
                    ORDER BY psb.{db_column} {sort_order}
                    LIMIT %s
                """
                
            elif position == "pitching":
                # 투구 통계 리더보드
                stat_mapping = {
                    "era": ("era", "ASC", 30),  # (컬럼명, 정렬순서, 최소 이닝)
                    "방어율": ("era", "ASC", 30),
                    "whip": ("whip", "ASC", 30), 
                    "승": ("wins", "DESC", 30),
                    "wins": ("wins", "DESC", 30),
                    "삼진": ("strikeouts", "DESC", 30),
                    "strikeouts": ("strikeouts", "DESC", 30),
                    "세이브": ("saves", "DESC", 0),
                    "saves": ("saves", "DESC", 0),
                    "war": ("war", "DESC", 30)
                }
                
                if stat_name.lower() not in stat_mapping:
                    result["error"] = f"지원하지 않는 투구 통계: {stat_name}"
                    return result
                
                db_column, sort_order, min_ip = stat_mapping[stat_name.lower()]
                
                # 팀 필터 조건 구성
                team_condition = ""
                params = [year]
                if team_filter:
                    team_name = self.team_mapping.get(team_filter, team_filter)
                    team_condition = "AND team_name = %s"
                    params.append(team_name)
                
                params.extend([min_ip, limit])
                
                query = f"""
                    SELECT 
                        pb.name as player_name, 
                        t.team_name, 
                        psp.{db_column} as stat_value,
                        psp.innings_pitched, psp.era, psp.whip, psp.wins, psp.losses, psp.saves, psp.strikeouts
                    FROM player_season_pitching psp
                    JOIN player_basic pb ON psp.player_id = pb.player_id
                    JOIN kbo_seasons ks ON psp.season = ks.season_year
                    LEFT JOIN teams t ON psp.team_code = t.team_id
                    WHERE psp.season = %s 
                    AND ks.league_type_code = '0'
                    {team_condition}
                    AND psp.innings_pitched >= %s 
                    AND psp.{db_column} IS NOT NULL
                    ORDER BY psp.{db_column} {sort_order}
                    LIMIT %s
                """
            else:
                result["error"] = f"지원하지 않는 포지션: {position}"
                return result
            
            # 쿼리 실행
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if rows:
                result["leaderboard"] = [dict(row) for row in rows]
                result["found"] = True
                result["total_qualified_players"] = len(rows)
                logger.info(f"[DatabaseQuery] Found {len(rows)} players in leaderboard")
            else:
                logger.warning(f"[DatabaseQuery] No data found for {stat_name} leaderboard")
                
        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying leaderboard: {e}")
            result["error"] = str(e)
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result
    
    def validate_player_exists(self, player_name: str, year: int) -> Dict[str, Any]:
        """
        선수가 해당 연도에 실제로 존재하는지 DB에서 확인합니다.
        
        Args:
            player_name: 선수명
            year: 시즌 년도
            
        Returns:
            선수 존재 여부와 상세 정보
        """
        logger.info(f"[DatabaseQuery] Validating player existence: {player_name}, {year}")
        
        result = {
            "player_name": player_name,
            "year": year,
            "exists": False,
            "found_players": [],
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 타격 테이블과 투구 테이블 모두에서 검색
            search_query = """
                SELECT DISTINCT pb.name as player_name, t.team_name, 'batting' as position_type
                FROM player_season_batting psb
                JOIN player_basic pb ON psb.player_id = pb.player_id
                LEFT JOIN teams t ON psb.team_code = t.team_id
                WHERE LOWER(pb.name) LIKE LOWER(%s) 
                AND psb.season = %s 
                AND psb.league = '정규시즌'
                
                UNION
                
                SELECT DISTINCT pb.name as player_name, t.team_name, 'pitching' as position_type
                FROM player_season_pitching psp
                JOIN player_basic pb ON psp.player_id = pb.player_id
                LEFT JOIN teams t ON psp.team_code = t.team_id
                WHERE LOWER(pb.name) LIKE LOWER(%s) 
                AND psp.season = %s 
                AND psp.league = '정규시즌'
                
                ORDER BY player_name
            """
            
            cursor.execute(search_query, (f'%{player_name}%', year, f'%{player_name}%', year))
            rows = cursor.fetchall()
            
            if rows:
                result["exists"] = True
                result["found_players"] = [dict(row) for row in rows]
                logger.info(f"[DatabaseQuery] Found {len(rows)} matching players")
            else:
                logger.warning(f"[DatabaseQuery] No player found matching {player_name} in {year}")
                
        except Exception as e:
            logger.error(f"[DatabaseQuery] Error validating player: {e}")
            result["error"] = str(e)
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result
    
    def get_position_info(self, position_abbr: str) -> Dict[str, Any]:
        """
        포지션 약어를 전체 포지션명으로 변환합니다.
        
        Args:
            position_abbr: 포지션 약어 (지, 타, 주, 중, 좌, 우, 一, 二, 三, 유, 포)
            
        Returns:
            포지션 정보 딕셔너리
        """
        logger.info(f"[DatabaseQuery] Looking up position: {position_abbr}")
        
        result = {
            "position_abbr": position_abbr,
            "position_name": None,
            "found": False,
            "error": None
        }
        
        try:
            full_position = self.position_mapping.get(position_abbr)
            if full_position:
                result["position_name"] = full_position
                result["found"] = True
                logger.info(f"[DatabaseQuery] Position mapping: {position_abbr} -> {full_position}")
            else:
                logger.warning(f"[DatabaseQuery] Unknown position abbreviation: {position_abbr}")
                
        except Exception as e:
            logger.error(f"[DatabaseQuery] Error looking up position: {e}")
            result["error"] = str(e)
        
        return result
    
    def get_player_defensive_stats(
        self, 
        player_name: str, 
        year: int = None
    ) -> Dict[str, Any]:
        """
        특정 선수의 수비 통계를 조회합니다.
        
        Args:
            player_name: 선수명 (부분 매칭 지원)
            year: 시즌 년도 (None이면 통산)
            
        Returns:
            선수 수비 통계 딕셔너리
        """
        logger.info(f"[DatabaseQuery] Querying defensive stats: {player_name}, {year}")
        
        result = {
            "player_name": player_name,
            "year": year,
            "defensive_stats": None,
            "found": False,
            "error": None,
            "message": "수비 통계 데이터가 현재 데이터베이스에 없습니다."
        }
        
        # 현재 데이터베이스에 수비 통계가 없다고 가정
        # 실제 데이터베이스에 수비 통계 테이블이 있다면 다음과 같은 쿼리를 사용:
        """
        예상되는 수비 통계 쿼리:
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if year:
                query = '''
                    SELECT 
                        pb.name as player_name,
                        t.team_name,
                        psd.season,
                        psd.position,
                        psd.games_played,
                        psd.innings_played,
                        psd.putouts,
                        psd.assists,
                        psd.errors,
                        psd.double_plays,
                        psd.fielding_percentage,
                        psd.range_factor
                    FROM player_season_defense psd
                    JOIN player_basic pb ON psd.player_id = pb.player_id
                    LEFT JOIN teams t ON psd.team_code = t.team_id
                    WHERE LOWER(pb.name) LIKE LOWER(%s) 
                    AND psd.season = %s 
                    AND psd.league = '정규시즌'
                    ORDER BY psd.games_played DESC
                    LIMIT 1
                '''
                cursor.execute(query, (f'%{player_name}%', year))
            else:
                # 통산 수비 통계
                query = '''
                    SELECT 
                        pb.name as player_name,
                        COUNT(DISTINCT psd.season) as seasons_played,
                        SUM(psd.games_played) as total_games,
                        SUM(psd.innings_played) as total_innings,
                        SUM(psd.putouts) as total_putouts,
                        SUM(psd.assists) as total_assists,
                        SUM(psd.errors) as total_errors,
                        SUM(psd.double_plays) as total_double_plays,
                        ROUND((CASE WHEN SUM(psd.putouts + psd.assists + psd.errors) > 0 
                                   THEN SUM(psd.putouts + psd.assists)::decimal / SUM(psd.putouts + psd.assists + psd.errors) 
                                   ELSE 0 END)::numeric, 3) as career_fielding_pct
                    FROM player_season_defense psd
                    JOIN player_basic pb ON psd.player_id = pb.player_id
                    WHERE LOWER(pb.name) LIKE LOWER(%s) 
                    AND psd.league = '정규시즌'
                    GROUP BY pb.player_id, pb.name
                    LIMIT 1
                '''
                cursor.execute(query, (f'%{player_name}%',))
            
            row = cursor.fetchone()
            if row:
                result["defensive_stats"] = dict(row)
                result["found"] = True
                result["message"] = f"수비 통계를 성공적으로 조회했습니다."
        """
        
        logger.info(f"[DatabaseQuery] Defensive stats not available for {player_name}")
        return result
    
    def get_pitcher_velocity_data(
        self, 
        player_name: str, 
        year: int = None
    ) -> Dict[str, Any]:
        """
        투수의 구속 데이터를 조회합니다.
        
        Args:
            player_name: 선수명 (부분 매칭 지원)
            year: 시즌 년도 (None이면 최근 데이터)
            
        Returns:
            구속 데이터 딕셔너리
        """
        logger.info(f"[DatabaseQuery] Querying pitch velocity: {player_name}, {year}")
        
        result = {
            "player_name": player_name,
            "year": year,
            "velocity_data": None,
            "found": False,
            "error": None,
            "message": "구속 데이터가 현재 데이터베이스에 없습니다."
        }
        
        # 현재 데이터베이스에 구속 데이터가 없다고 가정
        # 실제 데이터베이스에 구속 데이터 테이블이 있다면 다음과 같은 쿼리를 사용:
        """
        예상되는 구속 데이터 쿼리:
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            query = '''
                SELECT 
                    pb.name as player_name,
                    t.team_name,
                    ppv.season,
                    ppv.fastball_avg_velocity,
                    ppv.fastball_max_velocity,
                    ppv.slider_avg_velocity,
                    ppv.curveball_avg_velocity,
                    ppv.changeup_avg_velocity,
                    ppv.cutter_avg_velocity,
                    ppv.splitter_avg_velocity,
                    ppv.total_pitches
                FROM player_pitch_velocity ppv
                JOIN player_basic pb ON ppv.player_id = pb.player_id
                LEFT JOIN teams t ON ppv.team_code = t.team_id
                WHERE LOWER(pb.name) LIKE LOWER(%s) 
                AND ppv.season = %s 
                AND ppv.league = '정규시즌'
                LIMIT 1
            '''
            
            if year:
                cursor.execute(query, (f'%{player_name}%', year))
            else:
                # 가장 최근 데이터
                query += ' ORDER BY ppv.season DESC LIMIT 1'
                cursor.execute(query.replace('AND ppv.season = %s', ''), (f'%{player_name}%',))
            
            row = cursor.fetchone()
            if row:
                result["velocity_data"] = dict(row)
                result["found"] = True
                result["message"] = f"구속 데이터를 성공적으로 조회했습니다."
        """
        
        logger.info(f"[DatabaseQuery] Pitch velocity data not available for {player_name}")
        return result
    
    def get_team_basic_info(self, team_name: str) -> Dict[str, Any]:
        """
        특정 팀의 기본 정보를 조회합니다 (홈구장, 마스코트, 감독 등).
        
        Args:
            team_name: 팀명
            
        Returns:
            팀 기본 정보 딕셔너리
        """
        logger.info(f"[DatabaseQuery] Querying team basic info: {team_name}")
        
        full_team_name = self.team_mapping.get(team_name, team_name)
        
        result = {
            "team_name": full_team_name,
            "basic_info": {
                "home_stadium": None,
                "mascot": None,
                "manager": None,
                "founded": None,
                "city": None
            },
            "found": False,
            "error": None
        }
        
        # 현재는 하드코딩된 정보 제공 (추후 DB 테이블이 있을 경우 쿼리로 변경)
        team_info_mapping = {
            "KIA 타이거즈": {
                "home_stadium": "기아 챔피언스 필드",
                "mascot": "호돌이",
                "city": "광주",
                "founded": "1982"
            },
            "LG 트윈스": {
                "home_stadium": "잠실야구장",
                "mascot": "루키",
                "city": "서울",
                "founded": "1982"
            },
            "두산 베어스": {
                "home_stadium": "잠실야구장",
                "mascot": "비바",
                "city": "서울", 
                "founded": "1982"
            },
            "롯데 자이언츠": {
                "home_stadium": "사직야구장",
                "mascot": "누리",
                "city": "부산",
                "founded": "1975"
            },
            "삼성 라이온즈": {
                "home_stadium": "대구 삼성 라이온즈 파크",
                "mascot": "레오",
                "city": "대구",
                "founded": "1982"
            },
            "키움 히어로즈": {
                "home_stadium": "고척스카이돔",
                "mascot": "턱돌이",
                "city": "서울",
                "founded": "2008"
            },
            "한화 이글스": {
                "home_stadium": "한화생명 이글스파크",
                "mascot": "수리 (Suri)",
                "city": "대전",
                "founded": "1985"
            },
            "KT 위즈": {
                "home_stadium": "수원 KT 위즈파크",
                "mascot": "위즈키",
                "city": "수원",
                "founded": "2015"
            },
            "NC 다이노스": {
                "home_stadium": "창원 NC파크",
                "mascot": "단디 & 쎄리",
                "city": "창원",
                "founded": "2013"
            },
            "SSG 랜더스": {
                "home_stadium": "문학야구장",
                "mascot": "랜디 (Landy)",
                "city": "인천",
                "founded": "2000"
            }
        }
        
        try:
            info = team_info_mapping.get(full_team_name)
            if info:
                result["basic_info"] = info
                result["found"] = True
                logger.info(f"[DatabaseQuery] Found basic info for {full_team_name}")
            else:
                logger.warning(f"[DatabaseQuery] No basic info found for {full_team_name}")
                
        except Exception as e:
            logger.error(f"[DatabaseQuery] Error getting team basic info: {e}")
            result["error"] = str(e)
        
        return result
    
    def get_team_summary(self, team_name: str, year: int) -> Dict[str, Any]:
        """
        특정 팀의 주요 선수들과 팀 통계를 실제 DB에서 조회합니다.
        
        Args:
            team_name: 팀명
            year: 시즌 년도
            
        Returns:
            실제 DB에서 조회된 팀 요약 정보
        """
        logger.info(f"[DatabaseQuery] Querying team summary: {team_name}, {year}")
        
        full_team_name = self.team_mapping.get(team_name, team_name)
        
        result = {
            "team_name": full_team_name,
            "year": year,
            "top_batters": [],
            "top_pitchers": [],
            "found": False,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 팀 상위 타자들 조회 (OPS 기준)
            batters_query = """
                SELECT pb.name as player_name, psb.avg, psb.obp, psb.slg, psb.ops, psb.home_runs, psb.rbi, psb.plate_appearances
                FROM player_season_batting psb
                JOIN player_basic pb ON psb.player_id = pb.player_id
                LEFT JOIN teams t ON psb.team_code = t.team_id
                WHERE t.team_name = %s 
                AND psb.season = %s 
                AND psb.league = '정규시즌'
                AND psb.plate_appearances >= 100
                AND psb.ops IS NOT NULL
                ORDER BY psb.ops DESC
                LIMIT 5
            """
            cursor.execute(batters_query, (full_team_name, year))
            batters = cursor.fetchall()
            
            if batters:
                result["top_batters"] = [dict(row) for row in batters]
                result["found"] = True
            
            # 팀 상위 투수들 조회 (ERA 기준)
            pitchers_query = """
                SELECT pb.name as player_name, psp.era, psp.whip, psp.wins, psp.losses, psp.saves, psp.innings_pitched, psp.strikeouts
                FROM player_season_pitching psp
                JOIN player_basic pb ON psp.player_id = pb.player_id
                LEFT JOIN teams t ON psp.team_code = t.team_id
                WHERE t.team_name = %s 
                AND psp.season = %s 
                AND psp.league = '정규시즌'
                AND psp.innings_pitched >= 30
                AND psp.era IS NOT NULL
                ORDER BY psp.era ASC
                LIMIT 5
            """
            cursor.execute(pitchers_query, (full_team_name, year))
            pitchers = cursor.fetchall()
            
            if pitchers:
                result["top_pitchers"] = [dict(row) for row in pitchers]
                result["found"] = True
                
            logger.info(f"[DatabaseQuery] Found team data: {len(batters)} batters, {len(pitchers)} pitchers")
                
        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying team summary: {e}")
            result["error"] = str(e)
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result