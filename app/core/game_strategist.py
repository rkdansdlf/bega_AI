"""
경기 전략 수립 및 불펜 운용 조언을 제공하는 모듈입니다.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import date, timedelta, datetime
import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


class GameStrategist:
    def __init__(self, connection: psycopg.Connection):
        self.connection = connection

    def _get_team_id(self, team_name: str) -> Optional[str]:
        """팀 이름으로 ID 조회"""
        try:
            cursor = self.connection.cursor(row_factory=dict_row)
            # 매핑 로직은 간단하게 처리하거나 DB 조회
            # 여기서는 DB 조회 사용
            query = "SELECT team_id FROM teams WHERE team_name = %s LIMIT 1"
            cursor.execute(query, (team_name,))
            row = cursor.fetchone()
            if row:
                return row["team_id"]

            # Fallback: 이미 코드일 경우
            return team_name
        except Exception:
            return team_name
        finally:
            cursor.close()

    def check_bullpen_availability(
        self, team_name: str, target_date: str = None
    ) -> Dict[str, Any]:
        """
        특정 팀의 불펜 투수 가용성 분석

        로직:
        1. 최근 3일간 투구 수 조회
        2. 연투 여부 확인
        3. 피로도 등급 산정 (Safe, Warning, Danger)
        """
        if not target_date:
            target_date = date.today().strftime("%Y-%m-%d")

        team_id = self._get_team_id(team_name)
        target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()

        # 최근 3일 데이터 조회 범위
        start_date = target_dt - timedelta(days=3)

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

            # 해당 팀의 불펜 투수 명단 조회 (선발 제외, 혹은 최근 구원 등판 기록 있는 선수)
            # 여기서는 'pitcher' 포지션 선수 중 선발(GS) 비중이 적은 선수 필터링 등의 로직이 필요하지만,
            # 단순화를 위해 최근 기록이 있는 모든 투수를 대상으로 함.

            # 최근 3일간 투구 기록 조회
            query = """
                SELECT 
                    pb.name, 
                    pb.player_id,
                    g.game_date,
                    gps.innings_pitched as ip,
                    gps.earned_runs as er,
                    -- 투구수 컬럼이 있다고 가정 (없으면 이닝 기반 추정)
                    -- gps.pitch_count 
                    (gps.innings_pitched * 15) as estimated_pitch_count -- 임시 추정치
                FROM game_pitching_stats gps
                JOIN game g ON gps.game_id = g.game_id
                JOIN player_basic pb ON gps.player_id = pb.player_id
                WHERE gps.team_code = %s 
                AND g.game_date BETWEEN %s AND %s
                AND g.game_date < %s -- 당일 제외
                ORDER BY pb.name, g.game_date DESC
            """
            cursor.execute(query, (team_id, start_date, target_date, target_date))
            rows = cursor.fetchall()

            pitcher_status = {}

            for row in rows:
                p_id = row["player_id"]
                name = row["name"]
                game_date = row["game_date"]
                days_ago = (target_dt - game_date).days
                pitches = row["estimated_pitch_count"]

                if name not in pitcher_status:
                    pitcher_status[name] = {
                        "recent_pitches": [],
                        "days_rest": (
                            days_ago if days_ago > 0 else 0
                        ),  # 가장 최근 등판일 기준
                        "total_pitches_3days": 0,
                        "status": "Available",
                    }

                # 데이터 축적
                status = pitcher_status[name]
                status["recent_pitches"].append(
                    {"days_ago": days_ago, "pitches": pitches}
                )
                status["total_pitches_3days"] += pitches
                # days_rest는 처음에 한 번 세팅된 값(가장 최근) 유지

            # 분석 및 상태 결정
            results = []
            for name, info in pitcher_status.items():
                # 간단한 피로도 로직
                # 1. 어제 30구 이상 던졌으면 휴식 권장
                # 2. 3일간 50구 이상이면 주의
                # 3. 3연투면 휴식 필수

                is_consecutive_3days = False
                days_pitched = [p["days_ago"] for p in info["recent_pitches"]]
                if 1 in days_pitched and 2 in days_pitched and 3 in days_pitched:
                    is_consecutive_3days = True

                status_label = "Available"
                reason = "충분한 휴식"

                if info["days_rest"] == 1:  # 어제 던짐
                    yesterday_pitches = next(
                        (
                            p["pitches"]
                            for p in info["recent_pitches"]
                            if p["days_ago"] == 1
                        ),
                        0,
                    )
                    if yesterday_pitches > 45:
                        status_label = "Unavailable"
                        reason = f"전날 투구수 과다 ({int(yesterday_pitches)}구)"
                    elif yesterday_pitches > 25:
                        status_label = "Warning"
                        reason = f"전날 투구 ({int(yesterday_pitches)}구)"

                if is_consecutive_3days:
                    status_label = "Unavailable"
                    reason = "3일 연투"

                if info["total_pitches_3days"] > 70:
                    status_label = "High Risk"
                    reason = (
                        f"최근 3일 투구수 과다 ({int(info['total_pitches_3days'])}구)"
                    )

                results.append(
                    {
                        "name": name,
                        "status": status_label,
                        "reason": reason,
                        "details": info,
                    }
                )

            # 등판했던 선수들 외에 로스터에 있는 다른 선수들도 추가해야 하지만,
            # 현재 로스터 테이블 접근이 어려우므로 최근 기록 있는 선수만 반환

            return {"team": team_name, "date": target_date, "bullpen_status": results}

        except Exception as e:
            logger.error(f"[GameStrategist] Availability check failed: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()

    def recommend_pitcher(self, team_name: str, situation: str) -> Dict[str, Any]:
        """
        상황별 투수 추천
        situation: "winning_close" (필승조), "losing" (추격조), "lefty_batter" (좌타자 상대)
        """
        # 1. 가용 자원 확인
        availability = self.check_bullpen_availability(team_name)
        if "error" in availability:
            return availability

        candidates = [
            p
            for p in availability["bullpen_status"]
            if p["status"] in ["Available", "Warning"]
        ]

        # 2. 상황별 필터링 (여기서는 단순 ERA 기반으로 모의 구현)
        # 실제로는 좌/우 스플릿, 탈삼진 능력 등을 봐야 함

        # 모의 데이터: ERA를 DB에서 가져오는 로직 추가 필요하지만 복잡도를 줄이기 위해
        # 가용 선수 중 랜덤 or 이름 기반 모의 로직 대신
        # 실제로는 `get_player_season_stats` 등을 호출해서 데이터를 보강해야 함.

        return {
            "situation": situation,
            "recommended": candidates[
                :3
            ],  # 단순히 가용한 선수 상위 3명 리턴 (개선 필요)
            "message": f"{situation} 상황에서 추천하는 가용 투수 목록입니다.",
        }
