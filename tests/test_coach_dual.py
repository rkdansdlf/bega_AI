"""
Coach Dual-Team Analysis Tests
"""

import pytest
from unittest.mock import Mock, patch
from app.routers.coach import _build_coach_query, _format_coach_context


class TestCoachDualAnalysis:
    """듀얼 팀 분석 및 리그 컨텍스트 테스트"""

    def test_build_coach_query_dual(self):
        """듀얼 팀 쿼리 생성 테스트"""

        # 1. 기본 듀얼 팀 쿼리
        query = _build_coach_query(
            team_name="KIA", focus=["matchup"], opponent_name="LG"
        )
        assert "KIA와 LG" in query
        assert "비교 분석" in query
        assert "상대 전적" in query

        # 2. 리그 컨텍스트 포함 (정규시즌)
        league_ctx_reg = {
            "season": 2024,
            "league_type": "REGULAR",
            "home": {"rank": 1, "gamesBehind": 0},
            "away": {"rank": 2, "gamesBehind": 1.5},
        }
        query_reg = _build_coach_query("KIA", ["batting"], "LG", league_ctx_reg)
        assert "순위 경쟁이 치열한 상황" in query_reg

        # 3. 리그 컨텍스트 포함 (포스트시즌)
        league_ctx_post = {
            "season": 2024,
            "league_type": "POST",
            "round": "한국시리즈",
            "game_no": 1,
        }
        query_post = _build_coach_query("KIA", [], "Samsung", league_ctx_post)
        assert "한국시리즈 1차전" in query_post

    def test_format_coach_context_dual(self):
        """듀얼 팀 컨텍스트 포맷팅 테스트"""

        tool_results = {
            "home": {
                "summary": {
                    "team_name": "KIA",
                    "top_batters": [{"player_name": "Kim", "ops": 1.0}],
                },
                "advanced": {"metrics": {"batting": {"ops": 0.9}}},
            },
            "away": {
                "summary": {
                    "team_name": "LG",
                    "top_batters": [{"player_name": "Lee", "ops": 0.9}],
                },
                "advanced": {"metrics": {"batting": {"ops": 0.8}}},
            },
            "matchup": {
                "team1": "KIA",
                "team2": "LG",
                "summary": {
                    "total_games": 10,
                    "team1_wins": 5,
                    "team2_wins": 5,
                    "draws": 0,
                },
                "games": [
                    {
                        "game_date": "2024-05-01",
                        "home_score": 5,
                        "away_score": 4,
                        "game_result": "team1_win",
                    }
                ],
            },
        }

        league_ctx = {
            "season": 2024,
            "league_type": "REGULAR",
            "home": {"rank": 1, "gamesBehind": 0},
            "away": {"rank": 2, "gamesBehind": 1.5},
        }

        context = _format_coach_context(
            tool_results, ["matchup"], league_context=league_ctx
        )

        # 검증
        assert "2024 시즌 컨텍스트" in context
        assert "Home" in context and "1위" in context
        assert "Away" in context and "2위" in context
        assert "[Home] KIA" in context
        assert "[Away] LG" in context
        assert "맞대결 전적" in context
        assert "KIA 5승" in context
        assert "LG 5승" in context


class TestCoachEdgeCases:
    """P0/P1 엣지 케이스 검증"""

    def test_rank_none_does_not_crash(self):
        """[P0] rank=None이면 TypeError 없이 정상 동작"""
        league_ctx = {
            "season": 2024,
            "league_type": "REGULAR",
            "home": {"rank": None, "gamesBehind": 0},
            "away": {"rank": 1, "gamesBehind": 0},
        }
        # 예외 없이 반환되어야 함
        query = _build_coach_query("KIA", ["batting"], "LG", league_ctx)
        assert "KIA" in query
        # rank=None이므로 순위 경쟁 문구가 나오면 안 됨
        assert "순위 경쟁이 치열한 상황" not in query

    def test_single_team_no_dual_wording(self):
        """[P1] 단일 팀 분석에 '양 팀' 문구가 포함되지 않아야 함"""
        query = _build_coach_query("KIA", ["batting"])
        assert "양 팀" not in query
        assert "타격 생산성" in query

    def test_recent_form_in_context(self):
        """[P1] recent_form 데이터가 컨텍스트 문자열에 포함되어야 함"""
        from app.routers.coach import _format_team_stats

        team_data = {
            "summary": {"team_name": "KIA", "top_batters": [], "top_pitchers": []},
            "advanced": {},
            "recent": {
                "found": True,
                "summary": {
                    "wins": 2,
                    "losses": 1,
                    "draws": 0,
                    "run_diff": 5,
                    "win_rate": 0.667,
                },
                "games": [
                    {
                        "date": "2024-09-01",
                        "opponent": "LG",
                        "score": "5:3",
                        "result": "Win",
                        "run_diff": 2,
                    },
                    {
                        "date": "2024-09-02",
                        "opponent": "LG",
                        "score": "2:4",
                        "result": "Loss",
                        "run_diff": -2,
                    },
                    {
                        "date": "2024-09-03",
                        "opponent": "LG",
                        "score": "6:1",
                        "result": "Win",
                        "run_diff": 5,
                    },
                ],
            },
        }
        ctx = _format_team_stats(team_data, "Home")
        assert "최근 경기 흐름" in ctx
        assert "3경기" in ctx
        assert "2승 1패" in ctx
        assert "득실 마진: +5" in ctx
        assert "승률: 0.667" in ctx
