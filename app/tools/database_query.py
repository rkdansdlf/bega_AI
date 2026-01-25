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
        
        # 1. NAME_TO_STATS_CODE: 통계 테이블(player_season_stats 등) 조회용
        # KBO 통계 관례상 SK, HT, WO, OB 등을 선호함
        self.NAME_TO_STATS_CODE = {
            "KIA": "HT", "기아": "HT", "HT": "HT",
            "LG": "LG",
            "SSG": "SK", "SK": "SK",
            "NC": "NC",
            "두산": "OB", "OB": "OB",
            "KT": "KT",
            "롯데": "LT", "LT": "LT",
            "삼성": "SS", "SS": "SS",
            "한화": "HH", "HH": "HH",
            "키움": "WO", "WO": "WO"
        }
        
        # 2. NAME_TO_GAME_CODE: 경기/순위 테이블(game 등) 조회용
        # game 테이블 PK인 SSG, HT, WO 등을 선호함
        self.NAME_TO_GAME_CODE = {
            "KIA": "HT", "HT": "HT",
            "SSG": "SSG", "SK": "SSG",
            "키움": "WO", "WO": "WO"
        }

        self.TEAM_CODE_TO_NAME = {
            "HT": "KIA 타이거즈", "KIA": "KIA 타이거즈",
            "LG": "LG 트윈스",
            "SSG": "SSG 랜더스", "SK": "SSG 랜더스",
            "NC": "NC 다이노스",
            "OB": "두산 베어스", "두산": "두산 베어스",
            "KT": "kt wiz",
            "LT": "롯데 자이언츠", "롯데": "롯데 자이언츠",
            "SS": "삼성 라이온즈", "삼성": "삼성 라이온즈",
            "HH": "한화 이글스", "한화": "한화 이글스",
            "WO": "키움 히어로즈", "키움": "키움 히어로즈"
        }
        
        # DB에서 최신 매핑 로드하여 위 딕셔너리 정밀 업데이트
        self._load_team_mappings()
        
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

    def _load_team_mappings(self):
        """OCI DB의 teams 테이블과 franchise_id를 활용하여 팀 매핑 정보를 동적으로 로드합니다."""
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # franchise_id가 있는 팀들 조회 (최신 창단순 정렬)
            query = """
                SELECT team_id, team_name, franchise_id, founded_year 
                FROM teams 
                WHERE franchise_id IS NOT NULL 
                ORDER BY franchise_id, founded_year DESC;
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if rows:
                logger.info(f"[DatabaseQuery] Loading mappings for {len(rows)} franchise entries from OCI")
                
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
                    modern_id = modern_team['team_id']
                    
                    # 2. 코드 라우팅 전략
                    # (A) 통계용: SK, HT, WO, OB 유지 시도
                    stats_code = next((m['team_id'] for m in members if m['team_id'] in ['SK', 'HT', 'WO', 'OB']), modern_id)
                    
                    # (B) 경기/순위용: SSG, HT, WO 등 최신 테이블 PK 기준
                    game_code = modern_id if modern_id != 'SK' else 'SSG'
                    if modern_name == "KIA 타이거즈": game_code = "HT"
                    if modern_name == "키움 히어로즈": game_code = "WO"
                    
                    # 3. 매핑 데이터 업데이트
                    for member in members:
                        m_id = member['team_id']
                        
                        # (1) 통계 쿼리용 (NAME_TO_STATS_CODE)
                        self.NAME_TO_STATS_CODE[m_id] = stats_code
                        
                        # (2) 경기/순위 쿼리용 (NAME_TO_GAME_CODE)
                        self.NAME_TO_GAME_CODE[m_id] = game_code
                        
                        # (3) 표시용 (TEAM_CODE_TO_NAME)
                        self.TEAM_CODE_TO_NAME[m_id] = modern_name
                        self.TEAM_CODE_TO_NAME[stats_code] = modern_name
                        self.TEAM_CODE_TO_NAME[game_code] = modern_name
                    
                    # 수동 추가: 이름 매칭 강화
                    self.NAME_TO_STATS_CODE[modern_name] = stats_code
                    self.NAME_TO_STATS_CODE[modern_name.split()[0]] = stats_code
                    self.NAME_TO_GAME_CODE[modern_name] = game_code
                    self.NAME_TO_GAME_CODE[modern_name.split()[0]] = game_code
                
                logger.info("[DatabaseQuery] SQL Team mappings (Stats vs Game) synchronized using OCI.")
            
            cursor.close()
        except Exception as e:
            logger.warning(f"[DatabaseQuery] Failed to load mappings from OCI: {e}. Using defaults.")

    def get_team_name(self, team_code: str) -> str:
        return self.TEAM_CODE_TO_NAME.get(team_code, team_code)
    
    def get_team_code(self, team_input: str) -> str:
        """통계 테이블용 코드를 반환합니다 (기본값)"""
        return self.NAME_TO_STATS_CODE.get(team_input, team_input)

    def get_game_team_code(self, team_input: str) -> str:
        """경기/순위 테이블용 코드를 반환합니다"""
        return self.NAME_TO_GAME_CODE.get(team_input, team_input)
    
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
            existing_players = [row['name'] for row in cursor.fetchall()]
            
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
                        psb.avg, psb.obp, psb.slg, psb.ops, psb.babip, psb.iso, psb.extra_stats
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
                    batting_dict = dict(batting_row)
                    # extra_stats에서 XR 추출
                    if batting_dict.get('extra_stats') and isinstance(batting_dict['extra_stats'], dict):
                        batting_dict['xr'] = batting_dict['extra_stats'].get('xr')
                    
                    result["batting_stats"] = batting_dict
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
                        psp.home_runs_allowed, psp.walks_allowed, psp.strikeouts, psp.era, psp.whip,
                        psp.k_per_nine, psp.bb_per_nine, psp.kbb, psp.extra_stats
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
                    pitching_dict = dict(pitching_row)
                    # extra_stats에서 FIP 추출 (독립 컬럼에 없을 경우 대비)
                    if pitching_dict.get('extra_stats') and isinstance(pitching_dict['extra_stats'], dict):
                        pitching_dict['fip_extra'] = pitching_dict['extra_stats'].get('fip')
                        # 만약 fip가 독립 컬럼으로도 조회되게 하고 싶다면 쿼리에 추가 가능
                    
                    result["pitching_stats"] = pitching_dict
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
                    "도루": ("stolen_bases", "DESC", 0),
                    "stolen_bases": ("stolen_bases", "DESC", 0),
                    "득점권타율": ("scoring_position_avg", "DESC", 0),
                    "득점권 타율": ("scoring_position_avg", "DESC", 0),
                    "scoring_position_avg": ("scoring_position_avg", "DESC", 0),
                    "홈승률": ("home_win_rate", "DESC", 0),
                    "홈 승률": ("home_win_rate", "DESC", 0),
                    "home_win_rate": ("home_win_rate", "DESC", 0)
                }
                
                if stat_name.lower() not in stat_mapping:
                    result["error"] = f"지원하지 않는 타격 통계: {stat_name}"
                    return result
                
                # DB에 없는 통계 컬럼 예외 처리
                # home_win_rate는 팀별 계산이 가능하므로 여기서 별도 처리
                unsupported_cols = ["scoring_position_avg", "wrc_plus"]
                db_column, sort_order, min_pa = stat_mapping[stat_name.lower()]
                
                if db_column == "home_win_rate":
                    # 홈 승률 계산용 특수 쿼리 (팀별)
                    query = """
                        SELECT 
                            t.team_name,
                            ROUND(SUM(CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END)::numeric / COUNT(*), 3) as win_rate,
                            COUNT(*) as games,
                            SUM(CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END) as wins
                        FROM game g
                        JOIN teams t ON g.home_team = t.team_id
                        WHERE g.season = %s
                        AND g.status IN ('S', 'E')
                        GROUP BY t.team_name
                        ORDER BY win_rate DESC
                        LIMIT %s
                    """
                    cursor.execute(query, (year, limit))
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        result["leaderboard"].append({
                            "team_name": row[0],
                            "stat_value": float(row[1]),
                            "details": {
                                "games": row[2],
                                "wins": row[3],
                                "type": "team_home_win_rate"
                            }
                        })
                    return result

                if db_column in unsupported_cols:
                    result["error"] = f"현재 '{stat_name}'(컬럼: {db_column}) 데이터는 데이터베이스에서 원격으로 지원하지 않는 지표입니다."
                    return result
                
                # 팀 필터 조건 구성
                team_condition = ""
                params = [year]
                if team_filter:
                    team_code = self.get_team_code(team_filter)
                    team_condition = "AND psb.team_code = %s"
                    params.append(team_code)
                
                params.extend([min_pa, limit])
                
                query = f"""
                    SELECT 
                        pb.name as player_name, 
                        psb.team_code, 
                        psb.{db_column} as stat_value,
                        psb.plate_appearances, psb.avg, psb.obp, psb.slg, psb.ops, psb.home_runs, psb.rbi
                    FROM (
                        SELECT DISTINCT ON (player_id, team_code) *
                        FROM player_season_batting
                        WHERE season = %s
                        ORDER BY player_id, team_code, plate_appearances DESC
                    ) psb
                    JOIN player_basic pb ON psb.player_id = pb.player_id
                    JOIN kbo_seasons ks ON psb.season = ks.season_year
                    WHERE psb.season = %s 
                    AND ks.league_type_code = '0'
                    {team_condition}
                    AND psb.plate_appearances >= %s 
                    AND psb.{db_column} IS NOT NULL
                    ORDER BY psb.{db_column} {sort_order}
                    LIMIT %s
                """
                # params order needs to be adjusted because season is used twice now
                # params previously: [year, min_pa, limit] (if no team filter) 
                # or [year, team_code, min_pa, limit]
                final_params = [year, year]
                if team_filter:
                    final_params.append(self.get_team_code(team_filter))
                final_params.extend([min_pa, limit])
                params = final_params
                
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
                    "홀드": ("holds", "DESC", 0),
                    "holds": ("holds", "DESC", 0),
                    "innings_pitched": ("innings_pitched", "DESC", 0),
                    "innings": ("innings_pitched", "DESC", 0),
                    "이닝": ("innings_pitched", "DESC", 0),
                    "ip": ("innings_pitched", "DESC", 0),
                    "quality_starts": ("quality_starts", "DESC", 0),
                    "qs": ("quality_starts", "DESC", 0)
                }
                
                if stat_name.lower() not in stat_mapping:
                    result["error"] = f"지원하지 않는 투구 통계: {stat_name}"
                    return result
                
                db_column, sort_order, min_ip = stat_mapping[stat_name.lower()]
                
                # 팀 필터 조건 구성
                team_condition = ""
                params = [year]
                if team_filter:
                    team_code = self.get_team_code(team_filter)
                    team_condition = "AND psp.team_code = %s"
                    params.append(team_code)
                
                params.extend([min_ip, limit])
                
                query = f"""
                    SELECT 
                        pb.name as player_name, 
                        psp.team_code, 
                        psp.{db_column} as stat_value,
                        psp.innings_pitched, psp.era, psp.whip, psp.wins, psp.losses, psp.saves, psp.strikeouts, psp.quality_starts
                    FROM (
                        SELECT DISTINCT ON (player_id, team_code) *
                        FROM player_season_pitching
                        WHERE season = %s
                        ORDER BY player_id, team_code, innings_pitched DESC
                    ) psp
                    JOIN player_basic pb ON psp.player_id = pb.player_id
                    JOIN kbo_seasons ks ON psp.season = ks.season_year
                    WHERE psp.season = %s 
                    AND ks.league_type_code = '0'
                    {team_condition}
                    AND psp.innings_pitched >= %s 
                    AND psp.{db_column} IS NOT NULL
                    ORDER BY psp.{db_column} {sort_order}
                    LIMIT %s
                """
                final_params = [year, year]
                if team_filter:
                    final_params.append(self.get_team_code(team_filter))
                final_params.extend([min_ip, limit])
                params = final_params
            else:
                result["error"] = f"지원하지 않는 포지션: {position}"
                return result
            
            # 쿼리 실행
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if rows:
                for row in rows:
                    team_code = row.get("team_code")
                    display_team_name = self.get_team_name(team_code) or team_code
                    
                    player_data = dict(row)
                    player_data["team_name"] = display_team_name
                    result["leaderboard"].append(player_data)
                
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
                AND psb.league = 'REGULAR'
                
                UNION
                
                SELECT DISTINCT pb.name as player_name, t.team_name, 'pitching' as position_type
                FROM player_season_pitching psp
                JOIN player_basic pb ON psp.player_id = pb.player_id
                LEFT JOIN teams t ON psp.team_code = t.team_id
                WHERE LOWER(pb.name) LIKE LOWER(%s) 
                AND psp.season = %s 
                AND psp.league = 'REGULAR'
                
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
                    AND psd.league = 'REGULAR'
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
                    AND psd.league = 'REGULAR'
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
                AND ppv.league = 'REGULAR'
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
    
    def get_team_season_rank(self, team_name: str, year: int) -> Dict[str, Any]:
        """
        특정 팀의 시즌 순위를 계산하여 반환합니다.
        MISSING VIEW v_team_rank_all 대체 구현
        """
        logger.info(f"[DatabaseQuery] Calculating team rank for {team_name} ({year})")
        
        team_code = self.get_team_code(team_name)
        result = {
            "team_name": team_name,
            "year": year,
            "rank": None,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "found": False,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 모든 팀의 승무패 집계 (정규시즌 기준)
            # season_id를 통해 정확한 시즌 필터링
            rank_query = """
                WITH team_stats AS (
                    SELECT 
                        team,
                        SUM(CASE WHEN winning_team = team THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN winning_team IS NOT NULL AND winning_team != team THEN 1 ELSE 0 END) as losses,
                        SUM(CASE WHEN winning_team IS NULL AND home_score = away_score THEN 1 ELSE 0 END) as draws
                    FROM (
                        SELECT home_team as team, winning_team, home_score, away_score 
                        FROM game g
                        JOIN kbo_seasons ks ON g.season_id = ks.season_id
                        JOIN teams t ON g.home_team = t.team_id
                        WHERE ks.season_year = %s AND ks.league_type_code = '0' -- 정규시즌
                        AND g.game_status = 'COMPLETED'
                        AND t.franchise_id IS NOT NULL
                        
                        UNION ALL
                        
                        SELECT away_team as team, winning_team, home_score, away_score 
                        FROM game g
                        JOIN kbo_seasons ks ON g.season_id = ks.season_id
                        JOIN teams t ON g.away_team = t.team_id
                        WHERE ks.season_year = %s AND ks.league_type_code = '0'
                        AND g.game_status = 'COMPLETED'
                        AND t.franchise_id IS NOT NULL
                    ) all_games
                    GROUP BY team
                )
                SELECT 
                    team, wins, losses, draws,
                    RANK() OVER (ORDER BY (wins::float / NULLIF(wins + losses, 0)) DESC) as rank
                FROM team_stats
            """
            
            cursor.execute(rank_query, (year, year))
            rankings = cursor.fetchall()
            
            target_team_code = self.get_game_team_code(team_name)
            
            # 1차 시도: 팀 코드로 매칭
            for row in rankings:
                if row['team'] == target_team_code:
                    result['rank'] = row['rank']
                    result['wins'] = row['wins']
                    result['losses'] = row['losses']
                    result['draws'] = row['draws']
                    result['found'] = True
                    break
            
            # 2차 시도: 매칭 실패 시 팀명으로 재확인 (혹시 DB에 한글로 저장된 경우)
            if not result['found']:
                for row in rankings:
                    if self.get_team_name(row['team']) == team_name:
                         result['rank'] = row['rank']
                         result['wins'] = row['wins']
                         result['found'] = True
                         break
            
            # 찾는 팀이 순위표에 없으면 (신생팀이거나 이름 불일치)
            if not result['found'] and rankings:
                 logger.warning(f"[DatabaseQuery] Team {team_name}({target_team_code}) not found in rankings. Available: {[r['team'] for r in rankings]}")

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error calculating team rank: {e}")
            result["error"] = str(e)
        finally:
             if 'cursor' in locals():
                cursor.close()
                
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
        
        team_code = self.get_team_code(team_name) if team_name else None
        full_team_name = self.get_team_name(team_code) if team_code else None
        
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

        team_code = self.get_team_code(team_name) if team_name else None
        full_team_name = self.get_team_name(team_code) if team_code else None
        
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
                WHERE psb.team_code = %s 
                AND psb.season = %s 
                AND psb.league = 'REGULAR'
                AND psb.plate_appearances >= 100
                AND psb.ops IS NOT NULL
                ORDER BY psb.ops DESC
                LIMIT 5
            """
            cursor.execute(batters_query, (team_code, year))
            batters = cursor.fetchall()
            
            if batters:
                for row in batters:
                    player_data = dict(row)
                    # team_name 필드가 쿼리에 포함되지 않았으므로 수동 추가
                    player_data["team_name"] = full_team_name
                    result["top_batters"].append(player_data)
                result["found"] = True
            
            # 팀 상위 투수들 조회 (ERA 기준)
            pitchers_query = """
                SELECT pb.name as player_name, psp.era, psp.whip, psp.wins, psp.losses, psp.saves, psp.innings_pitched, psp.strikeouts
                FROM player_season_pitching psp
                JOIN player_basic pb ON psp.player_id = pb.player_id
                WHERE psp.team_code = %s 
                AND psp.season = %s 
                AND psp.league = 'REGULAR'
                AND psp.innings_pitched >= 30
                AND psp.era IS NOT NULL
                ORDER BY psp.era ASC
                LIMIT 5
            """
            cursor.execute(pitchers_query, (team_code, year))
            pitchers = cursor.fetchall()
            
            if pitchers:
                for row in pitchers:
                    player_data = dict(row)
                    player_data["team_name"] = full_team_name
                    result["top_pitchers"].append(player_data)
                result["found"] = True
                
            logger.info(f"[DatabaseQuery] Found team data: {len(batters)} batters, {len(pitchers)} pitchers")
                
        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying team summary: {e}")
            result["error"] = str(e)
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result

    def get_pitcher_starting_win_rate(
        self, 
        player_name: str, 
        year: int
    ) -> Dict[str, Any]:
        """
        특정 투수가 선발 등판했을 때 팀의 승률을 계산합니다.
        
        Args:
            player_name: 투수 이름
            year: 시즌 년도
            
        Returns:
            투수 선발 시 팀 승률 정보
        """
        logger.info(f"[DatabaseQuery] Querying pitcher starting win rate: {player_name}, {year}")
        
        result = {
            "player_name": player_name,
            "year": year,
            "found": False,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 1. 먼저 player_id 조회
            player_query = """
                SELECT player_id, name, team_id 
                FROM player_basic 
                WHERE name LIKE %s
                LIMIT 1
            """
            cursor.execute(player_query, (f"%{player_name}%",))
            player = cursor.fetchone()
            
            if not player:
                result["error"] = f"선수 '{player_name}'을(를) 찾을 수 없습니다."
                return result
            
            player_id = player["player_id"]
            result["player_name"] = player["name"]
            result["team"] = player.get("team_id")
            
            # 2. 해당 투수가 선발로 등판한 경기 조회 (home_pitcher 또는 away_pitcher)
            games_query = """
                SELECT 
                    g.game_id,
                    g.game_date,
                    g.home_team,
                    g.away_team,
                    g.home_score,
                    g.away_score,
                    g.winning_team,
                    CASE 
                        WHEN g.home_pitcher = %s THEN g.home_team
                        WHEN g.away_pitcher = %s THEN g.away_team
                    END as pitcher_team,
                    CASE 
                        WHEN g.home_pitcher = %s THEN 'home'
                        WHEN g.away_pitcher = %s THEN 'away'
                    END as pitcher_side
                FROM game g
                WHERE g.season_id = %s
                AND (g.home_pitcher = %s OR g.away_pitcher = %s)
                AND g.game_status IN ('S', 'E')
                ORDER BY g.game_date
            """
            cursor.execute(games_query, (player_id, player_id, player_id, player_id, year, player_id, player_id))
            games = cursor.fetchall()
            
            if not games:
                result["error"] = f"{year}년에 '{player_name}' 선수의 선발 등판 기록을 찾을 수 없습니다."
                return result
            
            # 3. 승패 계산
            total_games = len(games)
            wins = 0
            losses = 0
            no_decision = 0
            
            for game in games:
                pitcher_team = game["pitcher_team"]
                winning_team = game["winning_team"]
                
                if winning_team == pitcher_team:
                    wins += 1
                elif winning_team and winning_team != pitcher_team:
                    losses += 1
                else:
                    no_decision += 1
            
            win_rate = round(wins / total_games, 3) if total_games > 0 else 0
            
            result["found"] = True
            result["stats"] = {
                "total_starts": total_games,
                "team_wins": wins,
                "team_losses": losses,
                "no_decision": no_decision,
                "team_win_rate": win_rate,
                "team_win_rate_pct": f"{win_rate * 100:.1f}%"
            }
            result["message"] = f"{player_name} 선수가 {year}년 선발 등판한 {total_games}경기 중 팀 승률: {win_rate * 100:.1f}% ({wins}승 {losses}패)"
            
            logger.info(f"[DatabaseQuery] Pitcher starting stats: {total_games} games, {wins}W-{losses}L")
            
        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying pitcher starting win rate: {e}")
            result["error"] = str(e)
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result

    def get_advanced_stats(
        self, 
        player_name: str, 
        year: int,
        position: str = "both"
    ) -> Dict[str, Any]:
        """
        특정 선수의 고급 통계(ERA+, OPS+, FIP, QS 등)를 계산하여 조회합니다.
        """
        logger.info(f"[DatabaseQuery] Querying advanced stats: {player_name}, {year}")
        
        result = {
            "player_name": player_name,
            "year": year,
            "found": False,
            "batting_advanced": None,
            "pitching_advanced": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            avg_query = """
                SELECT 
                    ROUND(AVG(avg)::numeric, 3) as lg_avg,
                    ROUND(AVG(obp)::numeric, 3) as lg_obp,
                    ROUND(AVG(slg)::numeric, 3) as lg_slg,
                    ROUND(AVG(ops)::numeric, 3) as lg_ops,
                    ROUND((SUM(earned_runs) * 9.0 / NULLIF(SUM(innings_pitched), 0))::numeric, 2) as lg_era
                FROM (
                    SELECT avg, obp, slg, ops, NULL as earned_runs, NULL as innings_pitched 
                    FROM player_season_batting WHERE season = %s AND plate_appearances >= 50
                    UNION ALL
                    SELECT NULL, NULL, NULL, NULL, earned_runs, innings_pitched 
                    FROM player_season_pitching WHERE season = %s AND innings_pitched >= 10
                ) combined
            """
            cursor.execute(avg_query, (year, year))
            lg_avg = cursor.fetchone()
            
            if position in ["batting", "both"]:
                bat_query = """
                    SELECT 
                        avg, obp, slg, ops, iso, babip, plate_appearances, team_code
                    FROM player_season_batting psb
                    JOIN player_basic pb ON psb.player_id = pb.player_id
                    WHERE (LOWER(pb.name) = LOWER(%s) OR LOWER(pb.name) LIKE LOWER(%s))
                    AND psb.season = %s
                    LIMIT 1
                """
                cursor.execute(bat_query, (player_name, f'%{player_name}%', year))
                bat_row = cursor.fetchone()
                
                if bat_row and lg_avg and lg_avg.get('lg_ops') and bat_row.get('ops') is not None:
                    try:
                        bat_row['ops_plus'] = round((float(bat_row['ops']) / float(lg_avg['lg_ops'])) * 100, 1)
                    except (TypeError, ValueError, ZeroDivisionError):
                        bat_row['ops_plus'] = None
                    result["batting_advanced"] = dict(bat_row)
                    result["found"] = True

            if position in ["pitching", "both"]:
                pitch_query = """
                    SELECT 
                        era, whip, fip, quality_starts, innings_pitched, strikeouts, team_code
                    FROM player_season_pitching psp
                    JOIN player_basic pb ON psp.player_id = pb.player_id
                    WHERE (LOWER(pb.name) = LOWER(%s) OR LOWER(pb.name) LIKE LOWER(%s))
                    AND psp.season = %s
                    LIMIT 1
                """
                cursor.execute(pitch_query, (player_name, f'%{player_name}%', year))
                pitch_row = cursor.fetchone()
                
                if pitch_row and lg_avg and lg_avg.get('lg_era') and pitch_row.get('era') is not None:
                    try:
                        # ERA+ = (League ERA / Pitcher ERA) * 100
                        if float(pitch_row['era']) > 0:
                            pitch_row['era_plus'] = round((float(lg_avg['lg_era']) / float(pitch_row['era'])) * 100, 1)
                        else:
                            pitch_row['era_plus'] = 0
                    except (TypeError, ValueError, ZeroDivisionError):
                        pitch_row['era_plus'] = None
                    result["pitching_advanced"] = dict(pitch_row)
                    result["found"] = True

            result["league_averages"] = dict(lg_avg) if lg_avg else None
            
        except Exception as e:
            logger.error(f"[DatabaseQuery] Error in get_advanced_stats: {e}")
            result["error"] = str(e)
        finally:
            if 'cursor' in locals():
                cursor.close()
        return result

    def get_team_advanced_metrics(self, team_name: str, year: int) -> Dict[str, Any]:
        """
        팀의 전반적인 성적 지표(ERA, OPS, AVG 등)와 리그 내 순위, 
        그리고 '불펜 과부하 지표(Bullpen Share)'를 조회하여 객관적 진단을 돕습니다.
        """
        logger.info(f"[DatabaseQuery] Querying advanced metrics for {team_name} in {year}")
        
        team_code = self.get_team_code(team_name)
        result = {
            "team_name": self.get_team_name(team_code),
            "year": year,
            "metrics": {},
            "league_averages": {},
            "rankings": {},
            "fatigue_index": {}
        }

        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # 1. 팀 타격 지표 및 순위
            batting_query = """
                WITH team_batting AS (
                    SELECT
                        team_code,
                        ROUND(AVG(avg)::numeric, 3) as avg,
                        ROUND(AVG(ops)::numeric, 3) as ops,
                        SUM(home_runs) as total_hr,
                        SUM(rbi) as total_rbi
                    FROM player_season_batting
                    WHERE season = %s AND plate_appearances > 50
                    GROUP BY team_code
                ),
                ranked_batting AS (
                    SELECT 
                        *,
                        RANK() OVER (ORDER BY ops DESC) as ops_rank,
                        RANK() OVER (ORDER BY avg DESC) as avg_rank
                    FROM team_batting
                )
                SELECT * FROM ranked_batting WHERE team_code = %s;
            """
            cursor.execute(batting_query, (year, team_code))
            bat_row = cursor.fetchone()
            if bat_row:
                result["metrics"]["batting"] = {
                    "avg": float(bat_row['avg']),
                    "ops": float(bat_row['ops']),
                    "total_hr": int(bat_row['total_hr']),
                    "total_rbi": int(bat_row['total_rbi'])
                }
                result["rankings"]["batting_ops"] = f"{bat_row['ops_rank']}위"
                result["rankings"]["batting_avg"] = f"{bat_row['avg_rank']}위"

            # 2. 팀 투구 및 '과부하' 지표
            # gs > 0 (선발), gs = 0 (불펜) 구분하여 이닝 비중 계산
            pitching_query = """
                WITH team_pitching_raw AS (
                    SELECT 
                        team_code,
                        -- 선발 요건: GS > 0 이거나 QS > 0 이거나 경기당 3이닝 이상 투구
                        SUM(CASE WHEN (COALESCE(games_started, 0) > 0 OR COALESCE(quality_starts, 0) > 0 OR (innings_pitched / NULLIF(games, 0)) >= 3) THEN innings_pitched ELSE 0 END) as starter_ip,
                        SUM(CASE WHEN NOT (COALESCE(games_started, 0) > 0 OR COALESCE(quality_starts, 0) > 0 OR (innings_pitched / NULLIF(games, 0)) >= 3) THEN innings_pitched ELSE 0 END) as bullpen_ip,
                        SUM(innings_pitched) as total_ip,
                        SUM(quality_starts) as total_qs,
                        SUM(CASE WHEN (COALESCE(games_started, 0) > 0) THEN games_started 
                                 WHEN (COALESCE(quality_starts, 0) > 0 OR (innings_pitched / NULLIF(games, 0)) >= 3) THEN games 
                                 ELSE 0 END) as total_gs,
                        ROUND(AVG(era)::numeric, 2) as avg_era
                    FROM player_season_pitching
                    WHERE season = %s
                    GROUP BY team_code
                ),
                fatigue_calc AS (
                    SELECT
                        *,
                        ROUND((bullpen_ip / NULLIF(total_ip, 0) * 100)::numeric, 1) as bullpen_share,
                        ROUND(((total_qs::numeric / NULLIF(total_gs, 0)) * 100)::numeric, 1) as qs_rate
                    FROM team_pitching_raw
                ),
                ranked_pitching AS (
                    SELECT 
                        *,
                        RANK() OVER (ORDER BY avg_era ASC) as era_rank,
                        RANK() OVER (ORDER BY bullpen_share DESC) as load_rank
                    FROM fatigue_calc
                )
                SELECT * FROM ranked_pitching WHERE team_code = %s;
            """
            cursor.execute(pitching_query, (year, team_code))
            pitch_row = cursor.fetchone()
            if pitch_row:
                result["metrics"]["pitching"] = {
                    "era_rank": f"{pitch_row['era_rank']}위",
                    "qs_rate": f"{pitch_row['qs_rate']}%",
                    "avg_era": float(pitch_row['avg_era'])
                }
                result["fatigue_index"] = {
                    "bullpen_share": f"{pitch_row['bullpen_share']}%",
                    "bullpen_load_rank": f"{pitch_row['load_rank']}위 (높을수록 과부하)"
                }

            # 3. 리그 평균 비교군 (불펜 비중 리그 평균)
            league_avg_query = """
                SELECT
                    ROUND(AVG(bullpen_share)::numeric, 1) as avg_bullpen_share,
                    ROUND(AVG(avg_era)::numeric, 2) as avg_league_era
                FROM (
                    SELECT 
                        SUM(CASE WHEN NOT (COALESCE(games_started, 0) > 0 OR COALESCE(quality_starts, 0) > 0 OR (innings_pitched / NULLIF(games, 0)) >= 3) THEN innings_pitched ELSE 0 END) / NULLIF(SUM(innings_pitched), 0) * 100 as bullpen_share,
                        AVG(era) as avg_era
                    FROM player_season_pitching
                    WHERE season = %s
                    GROUP BY team_code
                ) as league_stats;
            """
            cursor.execute(league_avg_query, (year,))
            l_avg = cursor.fetchone()
            if l_avg:
                result["league_averages"]["bullpen_share"] = f"{l_avg['avg_bullpen_share']}%"
                result["league_averages"]["era"] = float(l_avg['avg_league_era'])

            return result

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error in advanced metrics: {e}")
            result["error"] = str(e)
            return result
        finally:
            if 'cursor' in locals():
                cursor.close()

    # ========================================
    # WPA (Win Probability Added) Query Methods
    # ========================================
    
    def get_player_wpa_leaders(
        self,
        year: int,
        position: str = "both",  # "batter", "pitcher", "both"
        team_filter: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        WPA(승리 확률 기여도) 리더보드를 조회합니다.
        
        Args:
            year: 시즌 년도
            position: "batter", "pitcher", "both"
            team_filter: 특정 팀만 필터링 (선택적)
            limit: 상위 몇 명까지 조회할지
            
        Returns:
            WPA 리더보드 딕셔너리
        """
        logger.info(f"[DatabaseQuery] Querying WPA leaders: {year}, {position}, team={team_filter}")
        
        result = {
            "year": year,
            "position": position,
            "team_filter": team_filter,
            "batter_leaders": [],
            "pitcher_leaders": [],
            "found": False,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 팀 필터 조건 구성
            team_condition = ""
            if team_filter:
                team_code = self.get_team_code(team_filter)
                team_condition = f"AND ge.team_code = '{team_code}'"
            
            # 타자 WPA 리더보드 (batter_id가 있는 이벤트)
            if position in ["batter", "both"]:
                batter_query = f"""
                    SELECT 
                        pb.name as player_name,
                        ge.team_code,
                        ROUND(SUM(ge.wpa)::numeric, 3) as total_wpa,
                        ROUND(SUM(CASE WHEN ge.wpa > 0 THEN ge.wpa ELSE 0 END)::numeric, 3) as wpa_positive,
                        ROUND(SUM(CASE WHEN ge.wpa < 0 THEN ge.wpa ELSE 0 END)::numeric, 3) as wpa_negative,
                        COUNT(*) as plate_appearances,
                        COUNT(CASE WHEN ge.wpa > 0.05 THEN 1 END) as clutch_plays
                    FROM game_events ge
                    JOIN game g ON ge.game_id = g.game_id
                    JOIN player_basic pb ON ge.batter_id = pb.player_id
                    WHERE g.season = %s
                    AND ge.wpa IS NOT NULL
                    AND ge.batter_id IS NOT NULL
                    {team_condition}
                    GROUP BY pb.player_id, pb.name, ge.team_code
                    HAVING COUNT(*) >= 50
                    ORDER BY SUM(ge.wpa) DESC
                    LIMIT %s
                """
                cursor.execute(batter_query, (year, limit))
                batter_rows = cursor.fetchall()
                
                for row in batter_rows:
                    team_code = row.get("team_code")
                    display_team_name = self.get_team_name(team_code) or team_code
                    
                    result["batter_leaders"].append({
                        "player_name": row["player_name"],
                        "team_name": display_team_name,
                        "total_wpa": float(row["total_wpa"]),
                        "wpa_positive": float(row["wpa_positive"]),
                        "wpa_negative": float(row["wpa_negative"]),
                        "plate_appearances": row["plate_appearances"],
                        "clutch_plays": row["clutch_plays"]
                    })
                
                if batter_rows:
                    result["found"] = True
            
            # 투수 WPA 리더보드 (pitcher_id가 있는 이벤트)
            if position in ["pitcher", "both"]:
                pitcher_query = f"""
                    SELECT 
                        pb.name as player_name,
                        ge.team_code,
                        ROUND(SUM(-ge.wpa)::numeric, 3) as total_wpa,
                        ROUND(SUM(CASE WHEN ge.wpa < 0 THEN -ge.wpa ELSE 0 END)::numeric, 3) as wpa_positive,
                        ROUND(SUM(CASE WHEN ge.wpa > 0 THEN -ge.wpa ELSE 0 END)::numeric, 3) as wpa_negative,
                        COUNT(*) as batters_faced,
                        COUNT(CASE WHEN ge.wpa < -0.05 THEN 1 END) as clutch_outs
                    FROM game_events ge
                    JOIN game g ON ge.game_id = g.game_id
                    JOIN player_basic pb ON ge.pitcher_id = pb.player_id
                    WHERE g.season = %s
                    AND ge.wpa IS NOT NULL
                    AND ge.pitcher_id IS NOT NULL
                    {team_condition}
                    GROUP BY pb.player_id, pb.name, ge.team_code
                    HAVING COUNT(*) >= 50
                    ORDER BY SUM(-ge.wpa) DESC
                    LIMIT %s
                """
                cursor.execute(pitcher_query, (year, limit))
                pitcher_rows = cursor.fetchall()
                
                for row in pitcher_rows:
                    team_code = row.get("team_code")
                    display_team_name = self.get_team_name(team_code) or team_code
                    
                    result["pitcher_leaders"].append({
                        "player_name": row["player_name"],
                        "team_name": display_team_name,
                        "total_wpa": float(row["total_wpa"]),
                        "wpa_positive": float(row["wpa_positive"]),
                        "wpa_negative": float(row["wpa_negative"]),
                        "batters_faced": row["batters_faced"],
                        "clutch_outs": row["clutch_outs"]
                    })
                
                if pitcher_rows:
                    result["found"] = True
            
            logger.info(f"[DatabaseQuery] Found {len(result['batter_leaders'])} batter leaders, {len(result['pitcher_leaders'])} pitcher leaders")
            
        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying WPA leaders: {e}")
            result["error"] = str(e)
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result
    
    def get_clutch_moments(
        self,
        game_id: Optional[str] = None,
        date: Optional[str] = None,
        year: Optional[int] = None,
        team_filter: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        결정적인 순간(WPA가 높은 플레이)을 조회합니다.
        
        Args:
            game_id: 특정 경기 ID (선택적)
            date: 경기 날짜 YYYY-MM-DD (선택적)
            year: 시즌 년도 (선택적)
            team_filter: 특정 팀만 필터링 (선택적)
            limit: 상위 몇 개까지 조회할지
            
        Returns:
            클러치 순간 목록 딕셔너리
        """
        logger.info(f"[DatabaseQuery] Querying clutch moments: game_id={game_id}, date={date}, year={year}")
        
        result = {
            "game_id": game_id,
            "date": date,
            "year": year,
            "clutch_moments": [],
            "found": False,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 조건 구성
            conditions = ["ge.wpa IS NOT NULL"]
            params = []
            
            if game_id:
                conditions.append("ge.game_id = %s")
                params.append(game_id)
            if date:
                conditions.append("g.game_date = %s")
                params.append(date)
            if year:
                conditions.append("g.season = %s")
                params.append(year)
            if team_filter:
                team_code = self.get_team_code(team_filter)
                conditions.append("(g.home_team = %s OR g.away_team = %s)")
                params.extend([team_code, team_code])
            
            where_clause = " AND ".join(conditions)
            params.append(limit)
            
            query = f"""
                SELECT 
                    ge.game_id,
                    g.game_date,
                    g.home_team,
                    g.away_team,
                    pb.name as batter_name,
                    pp.name as pitcher_name,
                    ge.event_type,
                    ge.event_description,
                    ge.inning,
                    ge.inning_half,
                    ROUND(ge.wpa::numeric, 3) as wpa,
                    ROUND(ge.win_expectancy_before::numeric, 3) as we_before,
                    ROUND(ge.win_expectancy_after::numeric, 3) as we_after,
                    ge.base_state,
                    ge.home_score,
                    ge.away_score,
                    ge.score_diff
                FROM game_events ge
                JOIN game g ON ge.game_id = g.game_id
                LEFT JOIN player_basic pb ON ge.batter_id = pb.player_id
                LEFT JOIN player_basic pp ON ge.pitcher_id = pp.player_id
                WHERE {where_clause}
                ORDER BY ABS(ge.wpa) DESC
                LIMIT %s
            """
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            for row in rows:
                # 주자 상황 해석
                base_state = row.get("base_state", 0) or 0
                runners = []
                if base_state & 1:
                    runners.append("1루")
                if base_state & 2:
                    runners.append("2루")
                if base_state & 4:
                    runners.append("3루")
                runners_str = ", ".join(runners) if runners else "주자 없음"
                
                result["clutch_moments"].append({
                    "game_id": row["game_id"],
                    "game_date": str(row["game_date"]) if row["game_date"] else None,
                    "home_team": self.get_team_name(row["home_team"]),
                    "away_team": self.get_team_name(row["away_team"]),
                    "batter": row["batter_name"],
                    "pitcher": row["pitcher_name"],
                    "event_type": row["event_type"],
                    "description": row["event_description"],
                    "inning": row["inning"],
                    "inning_half": "초" if row["inning_half"] == "top" else "말",
                    "wpa": float(row["wpa"]),
                    "we_before": float(row["we_before"]) if row["we_before"] else None,
                    "we_after": float(row["we_after"]) if row["we_after"] else None,
                    "situation": {
                        "home_score": row["home_score"],
                        "away_score": row["away_score"],
                        "score_diff": row["score_diff"],
                        "runners": runners_str
                    }
                })
            
            if rows:
                result["found"] = True
                logger.info(f"[DatabaseQuery] Found {len(rows)} clutch moments")
            
        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying clutch moments: {e}")
            result["error"] = str(e)
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result
    
    def get_player_wpa_stats(
        self,
        player_name: str,
        year: int,
        position: str = "both"  # "batter", "pitcher", "both"
    ) -> Dict[str, Any]:
        """
        특정 선수의 WPA 상세 통계를 조회합니다.
        
        Args:
            player_name: 선수명
            year: 시즌 년도
            position: "batter", "pitcher", "both"
            
        Returns:
            선수 WPA 통계 딕셔너리
        """
        logger.info(f"[DatabaseQuery] Querying player WPA stats: {player_name}, {year}")
        
        result = {
            "player_name": player_name,
            "year": year,
            "batting_wpa": None,
            "pitching_wpa": None,
            "found": False,
            "error": None
        }
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 타자로서 WPA
            if position in ["batter", "both"]:
                batter_query = """
                    SELECT 
                        pb.name as player_name,
                        ROUND(SUM(ge.wpa)::numeric, 3) as total_wpa,
                        ROUND(SUM(CASE WHEN ge.wpa > 0 THEN ge.wpa ELSE 0 END)::numeric, 3) as wpa_positive,
                        ROUND(SUM(CASE WHEN ge.wpa < 0 THEN ge.wpa ELSE 0 END)::numeric, 3) as wpa_negative,
                        COUNT(*) as plate_appearances,
                        COUNT(CASE WHEN ge.wpa > 0.05 THEN 1 END) as clutch_hits,
                        COUNT(CASE WHEN ge.wpa < -0.05 THEN 1 END) as clutch_fails,
                        ROUND(AVG(ge.wpa)::numeric, 4) as avg_wpa_per_pa,
                        MAX(ge.wpa) as best_moment_wpa,
                        MIN(ge.wpa) as worst_moment_wpa
                    FROM game_events ge
                    JOIN game g ON ge.game_id = g.game_id
                    JOIN player_basic pb ON ge.batter_id = pb.player_id
                    WHERE g.season = %s
                    AND ge.wpa IS NOT NULL
                    AND LOWER(pb.name) LIKE LOWER(%s)
                    GROUP BY pb.player_id, pb.name
                """
                cursor.execute(batter_query, (year, f'%{player_name}%'))
                batter_row = cursor.fetchone()
                
                if batter_row:
                    result["batting_wpa"] = {
                        "player_name": batter_row["player_name"],
                        "total_wpa": float(batter_row["total_wpa"]),
                        "wpa_positive": float(batter_row["wpa_positive"]),
                        "wpa_negative": float(batter_row["wpa_negative"]),
                        "plate_appearances": batter_row["plate_appearances"],
                        "clutch_hits": batter_row["clutch_hits"],
                        "clutch_fails": batter_row["clutch_fails"],
                        "avg_wpa_per_pa": float(batter_row["avg_wpa_per_pa"]),
                        "best_moment_wpa": float(batter_row["best_moment_wpa"]) if batter_row["best_moment_wpa"] else None,
                        "worst_moment_wpa": float(batter_row["worst_moment_wpa"]) if batter_row["worst_moment_wpa"] else None
                    }
                    result["found"] = True
            
            # 투수로서 WPA (부호 반전: 타자의 WPA가 음수면 투수에게 긍정적)
            if position in ["pitcher", "both"]:
                pitcher_query = """
                    SELECT 
                        pb.name as player_name,
                        ROUND(SUM(-ge.wpa)::numeric, 3) as total_wpa,
                        ROUND(SUM(CASE WHEN ge.wpa < 0 THEN -ge.wpa ELSE 0 END)::numeric, 3) as wpa_positive,
                        ROUND(SUM(CASE WHEN ge.wpa > 0 THEN -ge.wpa ELSE 0 END)::numeric, 3) as wpa_negative,
                        COUNT(*) as batters_faced,
                        COUNT(CASE WHEN ge.wpa < -0.05 THEN 1 END) as clutch_outs,
                        COUNT(CASE WHEN ge.wpa > 0.05 THEN 1 END) as clutch_fails,
                        ROUND(AVG(-ge.wpa)::numeric, 4) as avg_wpa_per_bf,
                        MIN(ge.wpa) as best_moment_wpa,
                        MAX(ge.wpa) as worst_moment_wpa
                    FROM game_events ge
                    JOIN game g ON ge.game_id = g.game_id
                    JOIN player_basic pb ON ge.pitcher_id = pb.player_id
                    WHERE g.season = %s
                    AND ge.wpa IS NOT NULL
                    AND LOWER(pb.name) LIKE LOWER(%s)
                    GROUP BY pb.player_id, pb.name
                """
                cursor.execute(pitcher_query, (year, f'%{player_name}%'))
                pitcher_row = cursor.fetchone()
                
                if pitcher_row:
                    result["pitching_wpa"] = {
                        "player_name": pitcher_row["player_name"],
                        "total_wpa": float(pitcher_row["total_wpa"]),
                        "wpa_positive": float(pitcher_row["wpa_positive"]),
                        "wpa_negative": float(pitcher_row["wpa_negative"]),
                        "batters_faced": pitcher_row["batters_faced"],
                        "clutch_outs": pitcher_row["clutch_outs"],
                        "clutch_fails": pitcher_row["clutch_fails"],
                        "avg_wpa_per_bf": float(pitcher_row["avg_wpa_per_bf"]),
                        "best_moment_wpa": float(-pitcher_row["best_moment_wpa"]) if pitcher_row["best_moment_wpa"] else None,
                        "worst_moment_wpa": float(-pitcher_row["worst_moment_wpa"]) if pitcher_row["worst_moment_wpa"] else None
                    }
                    result["found"] = True
            
            if result["found"]:
                logger.info(f"[DatabaseQuery] Found WPA stats for {player_name}")
            else:
                logger.warning(f"[DatabaseQuery] No WPA data found for {player_name} in {year}")
            
        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying player WPA stats: {e}")
            result["error"] = str(e)
        finally:
            if 'cursor' in locals():
                cursor.close()
        
        return result