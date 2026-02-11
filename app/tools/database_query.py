"""
직접 데이터베이스 쿼리를 통해 정확한 통계를 조회하는 도구입니다.

이 모듈은 LLM의 환각(hallucination)을 방지하기 위해
실제 DB에서 정확한 데이터만을 조회하여 반환합니다.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import threading
import psycopg
from psycopg.rows import dict_row
from psycopg import Connection as PgConnection

logger = logging.getLogger(__name__)


# ============================================================
# TTL Cache for Coach Performance Optimization
# ============================================================


class TTLCache:
    """
    Thread-safe TTL (Time-To-Live) 캐시.

    팀 통계와 같이 자주 변경되지 않는 데이터를 캐싱하여
    반복적인 DB 쿼리를 줄입니다.

    사용 예:
        cache = TTLCache(ttl_seconds=3600)  # 1시간
        cache.set("team_summary:KIA:2024", data)
        cached = cache.get("team_summary:KIA:2024")
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 100):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값을 조회합니다. 만료된 경우 None 반환."""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    logger.debug(f"[TTLCache] Hit: {key}")
                    return value
                else:
                    # 만료된 항목 제거
                    del self._cache[key]
                    logger.debug(f"[TTLCache] Expired: {key}")
            return None

    def set(self, key: str, value: Any) -> None:
        """캐시에 값을 저장합니다."""
        with self._lock:
            # 캐시 크기 제한
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            self._cache[key] = (value, time.time())
            logger.debug(f"[TTLCache] Set: {key}")

    def _evict_oldest(self) -> None:
        """가장 오래된 항목을 제거합니다."""
        if not self._cache:
            return
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]
        logger.debug(f"[TTLCache] Evicted: {oldest_key}")

    def clear(self) -> None:
        """캐시를 비웁니다."""
        with self._lock:
            self._cache.clear()
            logger.info("[TTLCache] Cache cleared")

    def stats(self) -> Dict[str, Any]:
        """캐시 통계를 반환합니다."""
        with self._lock:
            now = time.time()
            valid_count = sum(
                1 for _, (_, ts) in self._cache.items() if now - ts < self.ttl
            )
            return {
                "total_entries": len(self._cache),
                "valid_entries": valid_count,
                "expired_entries": len(self._cache) - valid_count,
                "ttl_seconds": self.ttl,
                "max_size": self.max_size,
            }


# 전역 캐시 인스턴스 (Coach 최적화용)
_coach_cache = TTLCache(ttl_seconds=3600, max_size=100)  # 1시간 TTL


def get_coach_cache() -> TTLCache:
    """전역 Coach 캐시 인스턴스를 반환합니다."""
    return _coach_cache


def clear_coach_cache() -> None:
    """Coach 캐시를 비웁니다."""
    _coach_cache.clear()


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
        # 정규 코드(SS, LT, LG, DB, KIA, KH, HH, SSG, NC, KT) 기준
        # NOTICE: OCI DB 테이블(player_season_*)이 아직 Legacy Code(HT, OB, WO, SK)를 사용하므로
        #         쿼리용 코드는 Legacy로 매핑합니다.
        self.NAME_TO_STATS_CODE = {
            "KIA": "HT",
            "기아": "HT",
            "HT": "HT",
            "LG": "LG",
            "SSG": "SK",
            "SK": "SK",
            "NC": "NC",
            "두산": "OB",
            "DB": "OB",
            "DO": "OB",
            "OB": "OB",
            "KT": "KT",
            "롯데": "LT",
            "LT": "LT",
            "삼성": "SS",
            "SS": "SS",
            "한화": "HH",
            "HH": "HH",
            "키움": "WO",
            "넥센": "WO",
            "KH": "WO",
            "KI": "WO",
            "WO": "WO",
            "NX": "WO",
        }

        # 2. NAME_TO_GAME_CODE: 경기/순위 테이블(game 등) 조회용
        # 정규 코드(SS, LT, LG, DB, KIA, KH, HH, SSG, NC, KT) 기준
        # NOTICE: game 테이블도 Legacy Code를 사용 중
        self.NAME_TO_GAME_CODE = {
            "KIA": "HT",
            "HT": "HT",
            "SSG": "SK",
            "SK": "SK",
            "키움": "WO",
            "넥센": "WO",
            "KH": "WO",
            "KI": "WO",
            "WO": "WO",
            "NX": "WO",
            "두산": "OB",
            "DB": "OB",
            "DO": "OB",
            "OB": "OB",
        }

        self.TEAM_CODE_TO_NAME = {
            "KIA": "KIA 타이거즈",
            "HT": "KIA 타이거즈",
            "LG": "LG 트윈스",
            "SSG": "SSG 랜더스",
            "SK": "SSG 랜더스",
            "NC": "NC 다이노스",
            "DB": "두산 베어스",
            "DO": "두산 베어스",
            "OB": "두산 베어스",
            "두산": "두산 베어스",
            "KT": "kt wiz",
            "LT": "롯데 자이언츠",
            "롯데": "롯데 자이언츠",
            "SS": "삼성 라이온즈",
            "삼성": "삼성 라이온즈",
            "HH": "한화 이글스",
            "한화": "한화 이글스",
            "KH": "키움 히어로즈",
            "KI": "키움 히어로즈",
            "WO": "키움 히어로즈",
            "키움": "키움 히어로즈",
            "NX": "키움 히어로즈",
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
            "포": "포수",
        }

    def _load_team_mappings(self):
        """OCI DB의 teams 테이블과 franchise_id를 활용하여 팀 매핑 정보를 동적으로 로드합니다."""
        try:
            cursor = self.connection.cursor(row_factory=dict_row)

            # franchise_id가 있는 팀들 조회 (최신 창단순 정렬)
            query = """
                SELECT team_id, team_name, franchise_id, founded_year, is_active
                FROM teams
                WHERE franchise_id IS NOT NULL
                ORDER BY franchise_id, is_active DESC, founded_year DESC;
            """
            cursor.execute(query)
            rows = cursor.fetchall()

            if rows:
                logger.info(
                    f"[DatabaseQuery] Loading mappings for {len(rows)} franchise entries from OCI"
                )

                # 프랜차이즈별로 그룹화
                franchises = {}
                for row in rows:
                    f_id = row["franchise_id"]
                    if f_id not in franchises:
                        franchises[f_id] = []
                    franchises[f_id].append(row)

                for f_id, members in franchises.items():
                    # 1. 현대적 브랜드명 선정 (DESC 정렬이므로 첫 번째가 최신)
                    modern_team = members[0]
                    modern_name = modern_team["team_name"]
                    modern_id = modern_team["team_id"]

                    # 2. 통계 테이블용 코드 결정
                    # 기존 하드코딩된 매핑이 있으면 그것을 사용 (통계 테이블 코드가 일관되지 않음)
                    # 예: LG 프랜차이즈 - MBC(1982), LG(1990) 중 통계는 LG 사용
                    #     한화 프랜차이즈 - BE(1986), HH(1993) 중 통계는 HH 사용
                    existing_stats_code = self.NAME_TO_STATS_CODE.get(modern_id)
                    if existing_stats_code:
                        stats_code = existing_stats_code
                    else:
                        # 하드코딩된 매핑이 없으면 modern_id 사용
                        stats_code = modern_id

                    game_code = stats_code

                    # 3. 매핑 데이터 업데이트 (기존 매핑 덮어쓰지 않음)
                    for member in members:
                        m_id = member["team_id"]

                        # (1) 통계 쿼리용 - 기존 매핑이 없을 때만 추가
                        if m_id not in self.NAME_TO_STATS_CODE:
                            self.NAME_TO_STATS_CODE[m_id] = stats_code

                        # (2) 경기/순위 쿼리용 - 기존 매핑이 없을 때만 추가
                        if m_id not in self.NAME_TO_GAME_CODE:
                            self.NAME_TO_GAME_CODE[m_id] = game_code

                        # (3) 표시용 (TEAM_CODE_TO_NAME) - 항상 현대 이름으로 업데이트
                        self.TEAM_CODE_TO_NAME[m_id] = modern_name
                        self.TEAM_CODE_TO_NAME[stats_code] = modern_name

                    # 수동 추가: 이름 매칭 강화 (기존 매핑이 없을 때만)
                    if modern_name not in self.NAME_TO_STATS_CODE:
                        self.NAME_TO_STATS_CODE[modern_name] = stats_code
                    short_name = modern_name.split()[0]
                    if short_name not in self.NAME_TO_STATS_CODE:
                        self.NAME_TO_STATS_CODE[short_name] = stats_code
                    if modern_name not in self.NAME_TO_GAME_CODE:
                        self.NAME_TO_GAME_CODE[modern_name] = game_code
                    if short_name not in self.NAME_TO_GAME_CODE:
                        self.NAME_TO_GAME_CODE[short_name] = game_code

                logger.info(
                    "[DatabaseQuery] SQL Team mappings (Stats vs Game) synchronized using OCI."
                )

            cursor.close()
        except Exception as e:
            logger.warning(
                f"[DatabaseQuery] Failed to load mappings from OCI: {e}. Using defaults."
            )

    def get_team_name(self, team_code: str) -> str:
        return self.TEAM_CODE_TO_NAME.get(team_code, team_code)

    def safe_float(self, value: Any, default: float = 0.0) -> float:
        """NULL 값을 안전하게 float로 변환합니다."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def safe_int(self, value: Any, default: int = 0) -> int:
        """NULL 값을 안전하게 int로 변환합니다."""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_team_code(self, team_input: str) -> str:
        """통계 테이블용 코드를 반환합니다 (기본값)"""
        code = self.NAME_TO_STATS_CODE.get(team_input)
        if not code:
            code = self.NAME_TO_STATS_CODE.get(team_input.upper())
        return code or team_input

    def get_game_team_code(self, team_input: str) -> str:
        """경기/순위 테이블용 코드를 반환합니다"""
        code = self.NAME_TO_GAME_CODE.get(team_input)
        if not code:
            code = self.NAME_TO_GAME_CODE.get(team_input.upper())
        return code or team_input

    def get_player_career_stats(
        self,
        player_name: str,
        position: str = "both",  # "batting", "pitching", "both"
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
            "error": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

            # 우선 선수가 존재하는지 확인
            check_query = """
                SELECT DISTINCT name FROM player_basic 
                WHERE (LOWER(name) = LOWER(%s) OR LOWER(name) LIKE LOWER(%s))
                LIMIT 5
            """
            cursor.execute(check_query, (player_name, f"%{player_name}%"))
            existing_players = [row["name"] for row in cursor.fetchall()]

            if not existing_players:
                result["error"] = (
                    f"선수 '{player_name}'을(를) 찾을 수 없습니다. 정확한 선수명을 확인해주세요."
                )
                logger.warning(
                    f"[DatabaseQuery] No player found matching: {player_name}"
                )
                return result
            else:
                logger.info(
                    f"[DatabaseQuery] Found matching players: {existing_players}"
                )
                # 정확히 일치하는 선수명이 있으면 그것을 사용, 없으면 첫 번째 결과 사용
                exact_match = next(
                    (
                        name
                        for name in existing_players
                        if name.lower() == player_name.lower()
                    ),
                    None,
                )
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
                    JOIN (SELECT DISTINCT ON (season_year) * FROM kbo_seasons WHERE league_type_code = '0' ORDER BY season_year, season_id) ks 
                        ON psb.season = ks.season_year
                    WHERE (LOWER(pb.name) = LOWER(%s) OR LOWER(pb.name) LIKE LOWER(%s))
                    AND psb.plate_appearances >= 10  -- 최소 기준
                    GROUP BY pb.player_id, pb.name
                    ORDER BY total_home_runs DESC
                    LIMIT 1
                """
                cursor.execute(batting_query, (player_name, f"%{player_name}%"))
                batting_row = cursor.fetchone()

                if batting_row:
                    batting_stats = dict(batting_row)
                    # OPS 계산
                    if batting_stats["career_obp"] and batting_stats["career_slg"]:
                        batting_stats["career_ops"] = round(
                            batting_stats["career_obp"] + batting_stats["career_slg"], 3
                        )

                    result["batting_stats"] = batting_stats
                    result["found"] = True
                    logger.info(
                        f"[DatabaseQuery] Found career batting stats for {player_name}"
                    )

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
                    JOIN (SELECT DISTINCT ON (season_year) * FROM kbo_seasons WHERE league_type_code = '0' ORDER BY season_year, season_id) ks 
                        ON psp.season = ks.season_year
                    WHERE (LOWER(pb.name) = LOWER(%s) OR LOWER(pb.name) LIKE LOWER(%s))
                    AND psp.innings_pitched >= 1  -- 최소 기준
                    GROUP BY pb.player_id, pb.name
                    ORDER BY total_wins DESC
                    LIMIT 1
                """
                cursor.execute(pitching_query, (player_name, f"%{player_name}%"))
                pitching_row = cursor.fetchone()

                if pitching_row:
                    result["pitching_stats"] = dict(pitching_row)
                    result["found"] = True
                    logger.info(
                        f"[DatabaseQuery] Found career pitching stats for {player_name}"
                    )

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying career stats: {e}")
            result["error"] = str(e)
        finally:
            if "cursor" in locals():
                cursor.close()

        return result

    def get_player_season_stats(
        self,
        player_name: str,
        year: int,
        position: str = "both",  # "batting", "pitching", "both"
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
        logger.info(
            f"[DatabaseQuery] Querying player stats: {player_name}, {year}, {position}"
        )

        result = {
            "player_name": player_name,
            "year": year,
            "batting_stats": None,
            "pitching_stats": None,
            "found": False,
            "error": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

            # 타격 통계 조회
            if position in ["batting", "both"]:
                batting_query = """
                    SELECT DISTINCT
                        pb.name as player_name, 
                        t.team_name,
                        psb.season as season_year,
                        psb.plate_appearances, psb.at_bats, psb.hits, psb.doubles, psb.triples, psb.home_runs,
                        psb.rbi, psb.runs, psb.walks, psb.strikeouts, psb.stolen_bases, psb.caught_stealing,
                        psb.avg, psb.obp, psb.slg, psb.ops, psb.babip
                    FROM player_season_batting psb
                    JOIN player_basic pb ON psb.player_id = pb.player_id
                    LEFT JOIN teams t ON psb.team_code = t.team_id
                    WHERE LOWER(pb.name) LIKE LOWER(%s) 
                    AND psb.season = %s 
                    ORDER BY psb.plate_appearances DESC
                    LIMIT 1
                """
                cursor.execute(batting_query, (f"%{player_name}%", year))
                batting_row = cursor.fetchone()

                if batting_row:
                    result["batting_stats"] = dict(batting_row)
                    result["found"] = True
                    logger.info(
                        f"[DatabaseQuery] Found batting stats for {player_name}"
                    )

            # 투구 통계 조회
            if position in ["pitching", "both"]:
                pitching_query = """
                    SELECT DISTINCT
                        pb.name as player_name,
                        t.team_name,
                        psp.season as season_year,
                        psp.games, psp.games_started, psp.wins, psp.losses, psp.saves, psp.holds,
                        psp.innings_pitched, psp.hits_allowed, psp.runs_allowed, psp.earned_runs,
                        psp.home_runs_allowed, psp.walks_allowed, psp.strikeouts, psp.era, psp.whip
                    FROM player_season_pitching psp
                    JOIN player_basic pb ON psp.player_id = pb.player_id
                    LEFT JOIN teams t ON psp.team_code = t.team_id
                    WHERE LOWER(pb.name) LIKE LOWER(%s) 
                    AND psp.season = %s 
                    ORDER BY psp.innings_pitched DESC
                    LIMIT 1
                """
                cursor.execute(pitching_query, (f"%{player_name}%", year))
                pitching_row = cursor.fetchone()

                if pitching_row:
                    result["pitching_stats"] = dict(pitching_row)
                    result["found"] = True
                    logger.info(
                        f"[DatabaseQuery] Found pitching stats for {player_name}"
                    )

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying player stats: {e}")
            result["error"] = str(e)
        finally:
            if "cursor" in locals():
                cursor.close()

        return result

    def get_team_leaderboard(
        self,
        stat_name: str,
        year: int,
        position: str,
        team_filter: Optional[str] = None,
        limit: int = 10,
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
        logger.info(
            f"[DatabaseQuery] Querying leaderboard: {stat_name}, {year}, {position}"
        )

        result = {
            "stat_name": stat_name,
            "year": year,
            "position": position,
            "team_filter": team_filter,
            "leaderboard": [],
            "found": False,
            "total_qualified_players": 0,
            "error": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

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
                    "home_win_rate": ("home_win_rate", "DESC", 0),
                }

                if stat_name.lower() not in stat_mapping:
                    result["error"] = f"지원하지 않는 타격 통계: {stat_name}"
                    return result

                # DB에 없는 통계 컬럼 예외 처리
                # home_win_rate는 팀별 계산이 가능하므로 여기서 별도 처리
                unsupported_cols = ["scoring_position_avg", "wrc_plus"]
                db_column, sort_order, min_pa = stat_mapping[stat_name.lower()]

                # Defense-in-depth: validate sort_order even though it comes from whitelist
                if sort_order not in ("ASC", "DESC"):
                    raise ValueError(f"Invalid sort order: {sort_order}")

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
                        result["leaderboard"].append(
                            {
                                "team_name": row[0],
                                "stat_value": float(row[1]),
                                "details": {
                                    "games": row[2],
                                    "wins": row[3],
                                    "type": "team_home_win_rate",
                                },
                            }
                        )
                    return result

                if db_column in unsupported_cols:
                    result["error"] = (
                        f"현재 '{stat_name}'(컬럼: {db_column}) 데이터는 데이터베이스에서 원격으로 지원하지 않는 지표입니다."
                    )
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
                    SELECT DISTINCT ON (pb.player_id, psb.team_code)
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
                    WHERE psb.season = %s 
                    {team_condition}
                    AND psb.plate_appearances >= %s 
                    AND psb.{db_column} IS NOT NULL
                    ORDER BY pb.player_id, psb.team_code, psb.{db_column} {sort_order}
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
                    "qs": ("quality_starts", "DESC", 0),
                }

                if stat_name.lower() not in stat_mapping:
                    result["error"] = f"지원하지 않는 투구 통계: {stat_name}"
                    return result

                db_column, sort_order, min_ip = stat_mapping[stat_name.lower()]

                # Defense-in-depth: validate sort_order even though it comes from whitelist
                if sort_order not in ("ASC", "DESC"):
                    raise ValueError(f"Invalid sort order: {sort_order}")

                # 팀 필터 조건 구성
                team_condition = ""
                params = [year]
                if team_filter:
                    team_code = self.get_team_code(team_filter)
                    team_condition = "AND psp.team_code = %s"
                    params.append(team_code)

                params.extend([min_ip, limit])

                query = f"""
                    SELECT DISTINCT ON (pb.player_id, psp.team_code)
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
                    WHERE psp.season = %s 
                    {team_condition}
                    AND psp.innings_pitched >= %s 
                    AND psp.{db_column} IS NOT NULL
                    ORDER BY pb.player_id, psp.team_code, psp.{db_column} {sort_order}
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
                logger.warning(
                    f"[DatabaseQuery] No data found for {stat_name} leaderboard"
                )

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying leaderboard: {e}")
            result["error"] = str(e)
        finally:
            if "cursor" in locals():
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
        logger.info(
            f"[DatabaseQuery] Validating player existence: {player_name}, {year}"
        )

        result = {
            "player_name": player_name,
            "year": year,
            "exists": False,
            "found_players": [],
            "error": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

            # 타격 테이블과 투구 테이블 모두에서 검색
            search_query = """
                SELECT DISTINCT pb.name as player_name, t.team_name, 'batting' as position_type
                FROM player_season_batting psb
                JOIN player_basic pb ON psb.player_id = pb.player_id
                LEFT JOIN teams t ON psb.team_code = t.team_id
                WHERE LOWER(pb.name) LIKE LOWER(%s) 
                AND psb.season = %s 
                
                UNION
                
                SELECT DISTINCT pb.name as player_name, t.team_name, 'pitching' as position_type
                FROM player_season_pitching psp
                JOIN player_basic pb ON psp.player_id = pb.player_id
                LEFT JOIN teams t ON psp.team_code = t.team_id
                WHERE LOWER(pb.name) LIKE LOWER(%s) 
                AND psp.season = %s 
                
                ORDER BY player_name
            """

            cursor.execute(
                search_query, (f"%{player_name}%", year, f"%{player_name}%", year)
            )
            rows = cursor.fetchall()

            if rows:
                result["exists"] = True
                result["found_players"] = [dict(row) for row in rows]
                logger.info(f"[DatabaseQuery] Found {len(rows)} matching players")
            else:
                logger.warning(
                    f"[DatabaseQuery] No player found matching {player_name} in {year}"
                )

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error validating player: {e}")
            result["error"] = str(e)
        finally:
            if "cursor" in locals():
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
            "error": None,
        }

        try:
            full_position = self.position_mapping.get(position_abbr)
            if full_position:
                result["position_name"] = full_position
                result["found"] = True
                logger.info(
                    f"[DatabaseQuery] Position mapping: {position_abbr} -> {full_position}"
                )
            else:
                logger.warning(
                    f"[DatabaseQuery] Unknown position abbreviation: {position_abbr}"
                )

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error looking up position: {e}")
            result["error"] = str(e)

        return result

    def get_player_defensive_stats(
        self, player_name: str, year: int = None
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
            "message": "수비 통계 데이터가 현재 데이터베이스에 없습니다.",
        }

        # 현재 데이터베이스에 수비 통계가 없다고 가정
        # 실제 데이터베이스에 수비 통계 테이블이 있다면 다음과 같은 쿼리를 사용:
        """
        예상되는 수비 통계 쿼리:
        
        try:
            cursor = self.connection.cursor(row_factory=dict_row)
            
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
        self, player_name: str, year: int = None
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
            "message": "구속 데이터가 현재 데이터베이스에 없습니다.",
        }

        # 현재 데이터베이스에 구속 데이터가 없다고 가정
        # 실제 데이터베이스에 구속 데이터 테이블이 있다면 다음과 같은 쿼리를 사용:
        """
        예상되는 구속 데이터 쿼리:
        
        try:
            cursor = self.connection.cursor(row_factory=dict_row)
            
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

        logger.info(
            f"[DatabaseQuery] Pitch velocity data not available for {player_name}"
        )
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
            "error": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

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
                if row["team"] == target_team_code:
                    result["rank"] = row["rank"]
                    result["wins"] = row["wins"]
                    result["losses"] = row["losses"]
                    result["draws"] = row["draws"]
                    result["found"] = True
                    break

            # 2차 시도: 매칭 실패 시 팀명으로 재확인 (혹시 DB에 한글로 저장된 경우)
            if not result["found"]:
                for row in rankings:
                    if self.get_team_name(row["team"]) == team_name:
                        result["rank"] = row["rank"]
                        result["wins"] = row["wins"]
                        result["found"] = True
                        break

            # 찾는 팀이 순위표에 없으면 (신생팀이거나 이름 불일치)
            if not result["found"] and rankings:
                logger.warning(
                    f"[DatabaseQuery] Team {team_name}({target_team_code}) not found in rankings. Available: {[r['team'] for r in rankings]}"
                )

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error calculating team rank: {e}")
            result["error"] = str(e)
        finally:
            if "cursor" in locals():
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
                "city": None,
            },
            "found": False,
            "error": None,
        }

        # 현재는 하드코딩된 정보 제공 (추후 DB 테이블이 있을 경우 쿼리로 변경)
        team_info_mapping = {
            "KIA 타이거즈": {
                "home_stadium": "기아 챔피언스 필드",
                "mascot": "호돌이",
                "city": "광주",
                "founded": "1982",
            },
            "LG 트윈스": {
                "home_stadium": "잠실야구장",
                "mascot": "루키",
                "city": "서울",
                "founded": "1982",
            },
            "두산 베어스": {
                "home_stadium": "잠실야구장",
                "mascot": "비바",
                "city": "서울",
                "founded": "1982",
            },
            "롯데 자이언츠": {
                "home_stadium": "사직야구장",
                "mascot": "누리",
                "city": "부산",
                "founded": "1975",
            },
            "삼성 라이온즈": {
                "home_stadium": "대구 삼성 라이온즈 파크",
                "mascot": "레오",
                "city": "대구",
                "founded": "1982",
            },
            "키움 히어로즈": {
                "home_stadium": "고척스카이돔",
                "mascot": "턱돌이",
                "city": "서울",
                "founded": "2008",
            },
            "한화 이글스": {
                "home_stadium": "한화생명 이글스파크",
                "mascot": "수리 (Suri)",
                "city": "대전",
                "founded": "1985",
            },
            "KT 위즈": {
                "home_stadium": "수원 KT 위즈파크",
                "mascot": "위즈키",
                "city": "수원",
                "founded": "2015",
            },
            "NC 다이노스": {
                "home_stadium": "창원 NC파크",
                "mascot": "단디 & 쎄리",
                "city": "창원",
                "founded": "2013",
            },
            "SSG 랜더스": {
                "home_stadium": "문학야구장",
                "mascot": "랜디 (Landy)",
                "city": "인천",
                "founded": "2000",
            },
        }

        try:
            info = team_info_mapping.get(full_team_name)
            if info:
                result["basic_info"] = info
                result["found"] = True
                logger.info(f"[DatabaseQuery] Found basic info for {full_team_name}")
            else:
                logger.warning(
                    f"[DatabaseQuery] No basic info found for {full_team_name}"
                )

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

        # TTL 캐시 확인 (Coach 최적화)
        cache_key = f"team_summary:{team_name}:{year}"
        cached_result = _coach_cache.get(cache_key)
        if cached_result is not None:
            logger.info(
                f"[DatabaseQuery] Cache hit for team summary: {team_name}, {year}"
            )
            return cached_result

        team_code = self.get_team_code(team_name) if team_name else None
        full_team_name = self.get_team_name(team_code) if team_code else None

        result = {
            "team_name": full_team_name,
            "year": year,
            "top_batters": [],
            "top_pitchers": [],
            "found": False,
            "error": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

            # 팀 상위 타자들 조회 (OPS 기준, 역할 분류 포함)
            batters_query = """
                SELECT pb.name as player_name, psb.avg, psb.obp, psb.slg, psb.ops,
                       psb.home_runs, psb.rbi, psb.plate_appearances,
                       CASE
                           WHEN psb.plate_appearances >= 400 THEN 'regular'
                           WHEN psb.plate_appearances >= 200 THEN 'platoon'
                           ELSE 'bench'
                       END as role
                FROM player_season_batting psb
                JOIN player_basic pb ON psb.player_id = pb.player_id
                LEFT JOIN teams t ON psb.team_code = t.team_id
                WHERE psb.team_code = %s
                AND psb.season = %s
                AND psb.league = 'REGULAR'
                AND psb.plate_appearances >= 50
                AND psb.ops IS NOT NULL
                ORDER BY psb.ops DESC
                LIMIT 8
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

            # 팀 상위 투수들 조회 (ERA 기준, 역할 분류 포함)
            pitchers_query = """
                SELECT pb.name as player_name, psp.era, psp.whip, psp.wins, psp.losses,
                       psp.saves, psp.holds, psp.innings_pitched, psp.strikeouts,
                       psp.games_started,
                       CASE
                           WHEN psp.games_started >= 10 THEN 'starter'
                           WHEN psp.saves >= 5 THEN 'closer'
                           WHEN psp.holds >= 10 THEN 'setup'
                           ELSE 'middle_reliever'
                       END as role
                FROM player_season_pitching psp
                JOIN player_basic pb ON psp.player_id = pb.player_id
                WHERE psp.team_code = %s
                AND psp.season = %s
                AND psp.league = 'REGULAR'
                AND psp.innings_pitched >= 20
                AND psp.era IS NOT NULL
                ORDER BY psp.era ASC
                LIMIT 8
            """
            cursor.execute(pitchers_query, (team_code, year))
            pitchers = cursor.fetchall()

            if pitchers:
                for row in pitchers:
                    player_data = dict(row)
                    player_data["team_name"] = full_team_name
                    result["top_pitchers"].append(player_data)
                result["found"] = True

            logger.info(
                f"[DatabaseQuery] Found team data: {len(batters)} batters, {len(pitchers)} pitchers"
            )

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying team summary: {e}")
            result["error"] = str(e)
        finally:
            if "cursor" in locals():
                cursor.close()

        # 성공적인 결과만 캐시 (Coach 최적화)
        if result.get("found") and not result.get("error"):
            _coach_cache.set(cache_key, result)

        return result

    def get_pitcher_starting_win_rate(
        self, player_name: str, year: int
    ) -> Dict[str, Any]:
        """
        특정 투수가 선발 등판했을 때 팀의 승률을 계산합니다.

        Args:
            player_name: 투수 이름
            year: 시즌 년도

        Returns:
            투수 선발 시 팀 승률 정보
        """
        logger.info(
            f"[DatabaseQuery] Querying pitcher starting win rate: {player_name}, {year}"
        )

        result = {
            "player_name": player_name,
            "year": year,
            "found": False,
            "error": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

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
            cursor.execute(
                games_query,
                (
                    player_id,
                    player_id,
                    player_id,
                    player_id,
                    year,
                    player_id,
                    player_id,
                ),
            )
            games = cursor.fetchall()

            if not games:
                result["error"] = (
                    f"{year}년에 '{player_name}' 선수의 선발 등판 기록을 찾을 수 없습니다."
                )
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
                "team_win_rate_pct": f"{win_rate * 100:.1f}%",
            }
            result["message"] = (
                f"{player_name} 선수가 {year}년 선발 등판한 {total_games}경기 중 팀 승률: {win_rate * 100:.1f}% ({wins}승 {losses}패)"
            )

            logger.info(
                f"[DatabaseQuery] Pitcher starting stats: {total_games} games, {wins}W-{losses}L"
            )

        except Exception as e:
            logger.error(
                f"[DatabaseQuery] Error querying pitcher starting win rate: {e}"
            )
            result["error"] = str(e)
        finally:
            if "cursor" in locals():
                cursor.close()

        return result

    def get_advanced_stats(
        self, player_name: str, year: int, position: str = "both"
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
            "pitching_advanced": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

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
                cursor.execute(bat_query, (player_name, f"%{player_name}%", year))
                bat_row = cursor.fetchone()

                if (
                    bat_row
                    and lg_avg
                    and lg_avg.get("lg_ops")
                    and bat_row.get("ops") is not None
                ):
                    try:
                        bat_row["ops_plus"] = round(
                            (float(bat_row["ops"]) / float(lg_avg["lg_ops"])) * 100, 1
                        )
                    except (TypeError, ValueError, ZeroDivisionError):
                        bat_row["ops_plus"] = None
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
                cursor.execute(pitch_query, (player_name, f"%{player_name}%", year))
                pitch_row = cursor.fetchone()

                if (
                    pitch_row
                    and lg_avg
                    and lg_avg.get("lg_era")
                    and pitch_row.get("era") is not None
                ):
                    try:
                        # ERA+ = (League ERA / Pitcher ERA) * 100
                        if float(pitch_row["era"]) > 0:
                            pitch_row["era_plus"] = round(
                                (float(lg_avg["lg_era"]) / float(pitch_row["era"]))
                                * 100,
                                1,
                            )
                        else:
                            pitch_row["era_plus"] = 0
                    except (TypeError, ValueError, ZeroDivisionError):
                        pitch_row["era_plus"] = None
                    result["pitching_advanced"] = dict(pitch_row)
                    result["found"] = True

            result["league_averages"] = dict(lg_avg) if lg_avg else None

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error in get_advanced_stats: {e}")
            result["error"] = str(e)
        finally:
            if "cursor" in locals():
                cursor.close()
        return result

    def get_team_advanced_metrics(self, team_name: str, year: int) -> Dict[str, Any]:
        """
        팀의 전반적인 성적 지표(ERA, OPS, AVG 등)와 리그 내 순위,
        그리고 '불펜 과부하 지표(Bullpen Share)'를 조회하여 객관적 진단을 돕습니다.
        """
        logger.info(
            f"[DatabaseQuery] Querying advanced metrics for {team_name} in {year}"
        )

        # TTL 캐시 확인 (Coach 최적화)
        cache_key = f"team_advanced_metrics:{team_name}:{year}"
        cached_result = _coach_cache.get(cache_key)
        if cached_result is not None:
            logger.info(
                f"[DatabaseQuery] Cache hit for advanced metrics: {team_name}, {year}"
            )
            return cached_result

        team_code = self.get_team_code(team_name)
        result = {
            "team_name": self.get_team_name(team_code),
            "year": year,
            "metrics": {},
            "league_averages": {},
            "rankings": {},
            "fatigue_index": {},
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

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
                    "avg": self.safe_float(bat_row["avg"]),
                    "ops": self.safe_float(bat_row["ops"]),
                    "total_hr": self.safe_int(bat_row["total_hr"]),
                    "total_rbi": self.safe_int(bat_row["total_rbi"]),
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
                    "era_rank": (
                        f"{pitch_row['era_rank']}위" if pitch_row["era_rank"] else "-"
                    ),
                    "qs_rate": (
                        f"{pitch_row['qs_rate']}%" if pitch_row["qs_rate"] else "0%"
                    ),
                    "avg_era": self.safe_float(pitch_row["avg_era"]),
                }
                result["fatigue_index"] = {
                    "bullpen_share": f"{pitch_row['bullpen_share']}%",
                    "bullpen_load_rank": f"{pitch_row['load_rank']}위 (높을수록 과부하)",
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
                result["league_averages"]["bullpen_share"] = (
                    f"{l_avg['avg_bullpen_share']}%"
                    if l_avg["avg_bullpen_share"]
                    else "0%"
                )
                result["league_averages"]["era"] = (
                    float(l_avg["avg_league_era"])
                    if l_avg["avg_league_era"] is not None
                    else 0.0
                )

            # 성공적인 결과 캐시 (Coach 최적화)
            _coach_cache.set(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error in advanced metrics: {e}")
            result["error"] = str(e)
            return result
        finally:
            if "cursor" in locals():
                cursor.close()

    def get_game_info(self, game_id: str) -> Dict[str, Any]:
        """
        특정 경기의 상세 정보(대진, 스코어, 일시, 장소 등)를 조회합니다.

        Args:
            game_id: 경기 ID (예: '20250322WOLG0')

        Returns:
            경기 상세 정보 딕셔너리
        """
        result = {
            "game_id": game_id,
            "found": False,
        }

        try:
            # 1. 기본 경기 정보 조회 (팀, 점수, 일시)
            query = """
            SELECT 
                g.game_date, g.home_team, g.away_team, 
                g.home_score, g.away_score, g.game_status,
                h.team_name as home_team_name,
                a.team_name as away_team_name,
                s.stadium_name
            FROM game g
            LEFT JOIN teams h ON g.home_team = h.team_id
            LEFT JOIN teams a ON g.away_team = a.team_id
            LEFT JOIN stadiums s ON g.stadium_id = s.stadium_id
            WHERE g.game_id = %s
            """
            row = self.conn.execute(query, (game_id,)).fetchone()

            if row:
                result.update(
                    {
                        "found": True,
                        "date": row[0].strftime("%Y-%m-%d") if row[0] else None,
                        "home_team": row[1],
                        "away_team": row[2],
                        "home_score": row[3],
                        "away_score": row[4],
                        "status": row[5],
                        "home_team_name": row[6],
                        "away_team_name": row[7],
                        "stadium": row[8],
                    }
                )

                # 2. 선발 투수 정보 조회 (있을 경우)
                pitcher_query = """
                SELECT player_name, team_id
                FROM game_lineups
                WHERE game_id = %s AND position = 'SP'
                """
                pitchers = self.conn.execute(pitcher_query, (game_id,)).fetchall()
                for p_name, p_team in pitchers:
                    if p_team == result["home_team"]:
                        result["home_starter"] = p_name
                    else:
                        result["away_starter"] = p_name

                # 3. 경기 요약 정보 조회 (결승타 등)
                summary_query = """
                SELECT summary_type, player_name, detail_text
                FROM game_summary
                WHERE game_id = %s
                """
                summaries = self.conn.execute(summary_query, (game_id,)).fetchall()
                if summaries:
                    result["summaries"] = [
                        {"type": s[0], "player": s[1], "detail": s[2]}
                        for s in summaries
                    ]

            return result
        except Exception as e:
            logger.error(f"[DatabaseQueryTool] Error fetching game info: {e}")
            return result

    def get_team_recent_form(
        self, team_name: str, year: int, limit: int = 10
    ) -> Dict[str, Any]:
        """
        팀의 최근 경기 결과(승패, 득실점)를 조회합니다.

        Args:
            team_name: 팀명
            year: 시즌 년도
            limit: 최근 n경기 (기본 10)

        Returns:
            최근 경기 결과 요약
        """
        logger.info(f"[DatabaseQuery] Querying recent form for {team_name} in {year}")

        # TTL 캐시 확인
        cache_key = f"recent_form:{team_name}:{year}:{limit}"
        cached_result = _coach_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        team_code = self.get_team_code(team_name)
        result = {
            "team_name": self.get_team_name(team_code),
            "year": year,
            "games": [],
            "summary": {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "win_rate": 0.0,
                "run_diff": 0,
            },
            "found": False,
            "error": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

            # 최근 경기 조회 (완료된 경기만)
            query = """
                SELECT 
                    g.game_date,
                    CASE 
                        WHEN g.home_team = %s THEN 'home' 
                        ELSE 'away' 
                    END as side,
                    CASE 
                        WHEN g.home_team = %s THEN t_away.team_name 
                        ELSE t_home.team_name 
                    END as opponent,
                    g.home_score,
                    g.away_score,
                    g.winning_team,
                    g.stadium_id
                FROM game g
                JOIN teams t_home ON g.home_team = t_home.team_id
                JOIN teams t_away ON g.away_team = t_away.team_id
                WHERE (g.home_team = %s OR g.away_team = %s)
                AND CAST(g.season_id AS TEXT) LIKE %s
                AND g.home_score IS NOT NULL -- 완료된 경기(점수 존재)
                ORDER BY g.game_date DESC
                LIMIT %s
            """
            season_pattern = f"{year}%"

            cursor.execute(
                query,
                (team_code, team_code, team_code, team_code, season_pattern, limit),
            )
            games = cursor.fetchall()

            if games:
                wins = 0
                losses = 0
                draws = 0
                total_runs = 0
                total_allowed = 0

                for g in games:
                    game_data = {
                        "date": g["game_date"].strftime("%Y-%m-%d"),
                        "opponent": g["opponent"],
                        "score": "",
                        "result": "",
                        "run_diff": 0,
                    }

                    my_score = (
                        g["home_score"] if g["side"] == "home" else g["away_score"]
                    )
                    opp_score = (
                        g["away_score"] if g["side"] == "home" else g["home_score"]
                    )

                    game_data["score"] = f"{my_score}:{opp_score}"
                    game_data["run_diff"] = my_score - opp_score

                    total_runs += my_score
                    total_allowed += opp_score

                    if g["winning_team"] == team_code:
                        game_data["result"] = "Win"
                        wins += 1
                    elif g["winning_team"] is None and my_score == opp_score:
                        game_data["result"] = "Draw"
                        draws += 1
                    else:
                        game_data["result"] = "Loss"
                        losses += 1

                    result["games"].append(game_data)

                result["summary"]["wins"] = wins
                result["summary"]["losses"] = losses
                result["summary"]["draws"] = draws
                result["summary"]["run_diff"] = total_runs - total_allowed
                total_games = wins + losses + draws
                if total_games > 0:
                    result["summary"]["win_rate"] = round(
                        wins / (wins + losses) if (wins + losses) > 0 else 0, 3
                    )

                result["found"] = True

                _coach_cache.set(cache_key, result)

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying recent form: {e}")
            result["error"] = str(e)

        return result

    def get_team_monthly_trend(self, team_name: str, year: int) -> Dict[str, Any]:
        """
        팀의 월별 승률 트렌드를 조회합니다.

        Args:
            team_name: 팀명
            year: 시즌 년도

        Returns:
            월별 트렌드 데이터
        """
        logger.info(f"[DatabaseQuery] Querying monthly trend for {team_name} in {year}")

        cache_key = f"monthly_trend:{team_name}:{year}"
        cached_result = _coach_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        team_code = self.get_team_code(team_name)
        result = {
            "team_name": self.get_team_name(team_code),
            "year": year,
            "monthly_stats": [],
            "found": False,
            "error": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

            # Using EXTRACT(MONTH) for filtering
            query = """
                SELECT 
                    EXTRACT(MONTH FROM g.game_date) as month,
                    COUNT(*) as games,
                    SUM(CASE WHEN g.winning_team = %s THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN g.winning_team != %s AND g.winning_team IS NOT NULL THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN g.winning_team IS NULL THEN 1 ELSE 0 END) as draws,
                    AVG(
                        CASE WHEN g.home_team = %s THEN g.home_score ELSE g.away_score END
                    ) as avg_runs_scored,
                    AVG(
                        CASE WHEN g.home_team = %s THEN g.away_score ELSE g.home_score END
                    ) as avg_runs_allowed
                FROM game g
                WHERE (g.home_team = %s OR g.away_team = %s)
                AND CAST(g.season_id AS TEXT) LIKE %s
                AND g.home_score IS NOT NULL
                GROUP BY month
                ORDER BY month
            """
            season_pattern = f"{year}%"
            cursor.execute(
                query,
                (
                    team_code,
                    team_code,
                    team_code,
                    team_code,
                    team_code,
                    team_code,
                    season_pattern,
                ),
            )
            rows = cursor.fetchall()

            if rows:
                for row in rows:
                    month_data = dict(row)
                    total_decisions = month_data["wins"] + month_data["losses"]
                    month_data["win_rate"] = (
                        round(month_data["wins"] / total_decisions, 3)
                        if total_decisions > 0
                        else 0.0
                    )
                    month_data["avg_runs_scored"] = round(
                        float(month_data["avg_runs_scored"]), 1
                    )
                    month_data["avg_runs_allowed"] = round(
                        float(month_data["avg_runs_allowed"]), 1
                    )
                    result["monthly_stats"].append(month_data)

                result["found"] = True
                _coach_cache.set(cache_key, result)

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying monthly trend: {e}")
            result["error"] = str(e)

        return result

    def get_team_matchup_stats(self, team_name: str, year: int) -> Dict[str, Any]:
        """
        특정 팀의 상대 전적을 조회합니다.

        Args:
            team_name: 팀명
            year: 시즌 년도

        Returns:
            상대 팀별 전적
        """
        logger.info(f"[DatabaseQuery] Querying matchup stats for {team_name} in {year}")

        cache_key = f"matchup_stats:{team_name}:{year}"
        cached_result = _coach_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        team_code = self.get_team_code(team_name)
        result = {
            "team_name": self.get_team_name(team_code),
            "year": year,
            "matchups": {},
            "found": False,
            "error": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

            query = """
                SELECT 
                    CASE 
                        WHEN g.home_team = %s THEN t_away.team_name 
                        ELSE t_home.team_name 
                    END as opponent,
                    COUNT(*) as games,
                    SUM(CASE WHEN g.winning_team = %s THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN g.winning_team != %s AND g.winning_team IS NOT NULL THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN g.winning_team IS NULL THEN 1 ELSE 0 END) as draws
                FROM game g
                JOIN teams t_home ON g.home_team = t_home.team_id
                JOIN teams t_away ON g.away_team = t_away.team_id
                WHERE (g.home_team = %s OR g.away_team = %s)
                AND CAST(g.season_id AS TEXT) LIKE %s
                AND g.home_score IS NOT NULL
                GROUP BY opponent
                ORDER BY wins DESC
            """
            season_pattern = f"{year}%"
            cursor.execute(
                query,
                (team_code, team_code, team_code, team_code, team_code, season_pattern),
            )
            rows = cursor.fetchall()

            if rows:
                for row in rows:
                    opp = row["opponent"]
                    data = dict(row)
                    total = data["wins"] + data["losses"]
                    data["win_rate"] = (
                        round(data["wins"] / total, 3) if total > 0 else 0.0
                    )
                    result["matchups"][opp] = data

                result["found"] = True
                _coach_cache.set(cache_key, result)

        except Exception as e:
            logger.error(f"[DatabaseQuery] Error querying matchup stats: {e}")
            result["error"] = str(e)

        return result
