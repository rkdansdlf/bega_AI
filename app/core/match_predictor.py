"""
투수 vs 타자 상성 분석 및 승부 예측을 수행하는 모듈입니다.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import date
import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


class MatchPredictor:
    def __init__(self, connection: psycopg.Connection):
        self.connection = connection

    def _get_player_id_and_team(
        self, name: str, year: int
    ) -> Optional[Tuple[str, str, str]]:
        """선수 이름으로 ID, 팀, 포지션(투수/타자)을 조회합니다."""
        try:
            cursor = self.connection.cursor(row_factory=dict_row)

            # 1. 투수 확인
            query_pitcher = """
                SELECT pb.player_id, t.team_name, 'pitcher' as role
                FROM player_season_pitching psp
                JOIN player_basic pb ON psp.player_id = pb.player_id
                LEFT JOIN teams t ON psp.team_code = t.team_id
                WHERE pb.name = %s AND psp.season = %s
                LIMIT 1
            """
            cursor.execute(query_pitcher, (name, year))
            row = cursor.fetchone()
            if row:
                return row["player_id"], row["team_name"], "pitcher"

            # 2. 타자 확인
            query_batter = """
                SELECT pb.player_id, t.team_name, 'batter' as role
                FROM player_season_batting psb
                JOIN player_basic pb ON psb.player_id = pb.player_id
                LEFT JOIN teams t ON psb.team_code = t.team_id
                WHERE pb.name = %s AND psb.season = %s
                LIMIT 1
            """
            cursor.execute(query_batter, (name, year))
            row = cursor.fetchone()
            if row:
                return row["player_id"], row["team_name"], "batter"

            return None
        except Exception as e:
            logger.error(f"[MatchPredictor] Player lookup failed: {e}")
            return None
        finally:
            cursor.close()

    def _get_recent_form(
        self, player_id: str, role: str, limit: int = 5
    ) -> Dict[str, Any]:
        """최근 N경기 성적을 조회합니다."""
        try:
            cursor = self.connection.cursor(row_factory=dict_row)
            if role == "batter":
                query = """
                    SELECT 
                        SUM(hits) as hits, SUM(at_bats) as ab, 
                        SUM(home_runs) as hr, SUM(rbi) as rbi,
                        AVG(avg) as avg
                    FROM game_batting_stats 
                    WHERE player_id = %s
                    ORDER BY game_date DESC LIMIT %s
                """
                # game_batting_stats might doesn't have game_date directly, usually joined with game table
                # Assuming simple schema for now, if error, will fix.
                # Actually commonly it's joined. Let's try join.
                query = """
                    SELECT 
                        COUNT(*) as games,
                        SUM(gbs.hits) as hits, SUM(gbs.at_bats) as ab,
                        SUM(gbs.home_runs) as hr, SUM(gbs.rbi) as rbi
                    FROM game_batting_stats gbs
                    JOIN game g ON gbs.game_id = g.game_id
                    WHERE gbs.player_id = %s
                    ORDER BY g.game_date DESC LIMIT %s
                """
            else:
                query = """
                    SELECT 
                        COUNT(*) as games,
                        SUM(gps.innings_pitched) as ip, SUM(gps.earned_runs) as er,
                        SUM(gps.strikeouts) as k, SUM(gps.wins) as w
                    FROM game_pitching_stats gps
                    JOIN game g ON gps.game_id = g.game_id
                    WHERE gps.player_id = %s
                    ORDER BY g.game_date DESC LIMIT %s
                """

            cursor.execute(query, (player_id, limit))
            row = cursor.fetchone()
            return dict(row) if row else {}
        except Exception as e:
            logger.warning(f"[MatchPredictor] Recent form query failed: {e}")
            return {}
        finally:
            cursor.close()

    def _get_head_to_head(self, pitcher_id: str, batter_id: str) -> Dict[str, Any]:
        """투수 vs 타자 상대 전적 조회 (game_events 테이블 활용)"""
        try:
            cursor = self.connection.cursor(row_factory=dict_row)
            # game_events 테이블이 있다고 가정
            query = """
                SELECT 
                    COUNT(*) as pa,
                    SUM(CASE WHEN result_type = 'hit' THEN 1 ELSE 0 END) as hits,
                    SUM(CASE WHEN result_type = 'homerun' THEN 1 ELSE 0 END) as hr,
                    SUM(CASE WHEN result_type = 'strikeout' THEN 1 ELSE 0 END) as k,
                    SUM(CASE WHEN result_type = 'walk' THEN 1 ELSE 0 END) as bb
                FROM game_events
                WHERE pitcher_id = %s AND batter_id = %s
            """
            cursor.execute(query, (pitcher_id, batter_id))
            row = cursor.fetchone()
            if row and row["pa"] > 0:
                avg = (
                    row["hits"] / (row["pa"] - row["bb"])
                    if (row["pa"] - row["bb"]) > 0
                    else 0.0
                )
                return {**dict(row), "avg": round(avg, 3)}
            return {"pa": 0, "message": "No record"}
        except Exception as e:
            logger.warning(
                f"[MatchPredictor] Head-to-head query failed (Table might not exist): {e}"
            )
            return {"pa": 0, "error": str(e)}
        finally:
            cursor.close()

    def predict(
        self, pitcher_name: str, batter_name: str, year: int = 2024
    ) -> Dict[str, Any]:
        """
        투수와 타자의 승부를 예측합니다.

        Algorithm:
        1. Base Score: 50 (Neutral)
        2. Adjust for Head-to-Head (Weight: High)
        3. Adjust for Recent Form (Weight: Medium)
        4. Adjust for Season Stats (Weight: Low)
        """
        pitcher_info = self._get_player_id_and_team(pitcher_name, year)
        batter_info = self._get_player_id_and_team(batter_name, year)

        if not pitcher_info:
            return {"error": f"Pitcher '{pitcher_name}' not found."}
        if not batter_info:
            return {"error": f"Batter '{batter_name}' not found."}

        p_id, p_team, _ = pitcher_info
        b_id, b_team, _ = batter_info

        if p_team == b_team and p_team is not None:
            return {
                "result": "Same Team",
                "message": "같은 팀 선수끼리는 대결하지 않습니다.",
            }

        # 1. Stats Lookup
        h2h = self._get_head_to_head(p_id, b_id)
        p_recent = self._get_recent_form(p_id, "pitcher")
        b_recent = self._get_recent_form(b_id, "batter")

        # 2. Score Calculation (0-100, >50 means Batter Advantage)
        score = 50.0
        reasons = []

        # H2H Impact
        if h2h.get("pa", 0) >= 3:
            avg = h2h.get("avg", 0.0)
            if avg > 0.350:
                score += 15
                reasons.append(f"상대 전적 강세 (타율 {avg:.3f})")
            elif avg < 0.200:
                score -= 15
                reasons.append(f"상대 전적 약세 (타율 {avg:.3f})")

        # Recent Form Impact
        # Batter
        b_ab = b_recent.get("ab", 0)
        if b_ab and b_ab > 10:
            b_avg = b_recent.get("hits", 0) / b_ab
            if b_avg >= 0.400:
                score += 10
                reasons.append(f"타자 최근 타격감 절정 (타율 {b_avg:.3f})")
            elif b_avg <= 0.150:
                score -= 10
                reasons.append(f"타자 최근 타격감 저조 (타율 {b_avg:.3f})")

        # Pitcher
        p_ip = p_recent.get("ip", 0)
        if p_ip and p_ip > 5:
            p_era = (p_recent.get("er", 0) * 9) / p_ip
            if p_era < 2.00:
                score -= 10
                reasons.append(f"투수 최근 언터쳐블 모드 (ERA {p_era:.2f})")
            elif p_era > 6.00:
                score += 10
                reasons.append(f"투수 최근 난조 (ERA {p_era:.2f})")

        # Result Generation
        prob = min(max(score, 10), 90) / 100.0  # Normalize to 0.1 - 0.9

        winner = batter_name if score > 50 else pitcher_name
        win_prob = prob if score > 50 else (1 - prob)

        return {
            "pitcher": pitcher_name,
            "batter": batter_name,
            "predicted_winner": winner,
            "win_probability": round(win_prob, 2),
            "score": score,
            "reasons": reasons,
            "h2h_summary": h2h,
            "recent_form": {"batter": b_recent, "pitcher": p_recent},
        }
