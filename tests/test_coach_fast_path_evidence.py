"""
Coach 기능 테스트 모듈.

Fast Path, 캐싱, 응답 검증기 등을 테스트합니다.
"""

import pytest
import json
import asyncio
import logging
import os
import re
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from pydantic import ValidationError

from tests.coach_test_support import (
    _build_game_evidence,
    _collect_sse_text,
    _extract_sse_meta_events,
    _install_coach_endpoint_cache_hit,
)

class TestCoachFastPathEvidence:
    def test_focus_section_requirements(self):
        """선택 focus별 섹션 요구사항 생성 테스트"""
        from app.routers.coach import _build_focus_section_requirements

        requirements = _build_focus_section_requirements(["recent_form", "bullpen"])
        assert "## 최근 전력" in requirements
        assert "## 불펜 상태" in requirements
        assert "미선택 focus" in requirements

    def test_find_missing_focus_sections(self):
        """detailed_markdown 섹션 누락 감지 테스트"""
        from app.routers.coach import _find_missing_focus_sections

        response_data = {
            "detailed_markdown": "## 최근 전력\n- 승률 0.700\n## 선발 투수\n- 선발 ERA 3.20"
        }
        missing = _find_missing_focus_sections(
            response_data, ["recent_form", "bullpen", "starter"]
        )
        assert missing == ["bullpen"]

    def test_resolve_supported_focuses_skips_empty_recent_and_matchup(self):
        from app.routers.coach import _resolve_supported_focuses

        evidence = _build_game_evidence(
            stage_label="PRE",
            round_display="시범경기",
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
        )
        tool_results = {
            "home": {
                "summary": {"found": True, "top_batters": []},
                "advanced": {
                    "found": True,
                    "metrics": {"batting": {"ops": 0.754}},
                    "fatigue_index": {"bullpen_share": "31.0%"},
                },
                "recent": {
                    "found": True,
                    "summary": {"wins": 0, "losses": 0, "draws": 0, "run_diff": 0},
                },
            },
            "away": {
                "summary": {"found": True, "top_batters": []},
                "advanced": {
                    "found": True,
                    "metrics": {"batting": {"ops": 0.731}},
                    "fatigue_index": {"bullpen_share": "28.5%"},
                },
                "recent": {"found": False, "summary": {}},
            },
            "matchup": {"found": False, "games": [], "summary": {}},
        }

        supported = _resolve_supported_focuses(
            ["recent_form", "matchup", "starter", "bullpen", "batting"],
            evidence,
            tool_results,
        )

        assert supported == ["starter", "bullpen", "batting"]

    def test_resolve_supported_focuses_requires_minimum_recent_and_matchup_samples(
        self,
    ):
        from app.routers.coach import _resolve_supported_focuses

        evidence = _build_game_evidence(
            stage_label="PRE",
            round_display="시범경기",
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
        )
        tool_results = {
            "home": {
                "summary": {"found": True, "top_batters": []},
                "advanced": {
                    "found": True,
                    "metrics": {"batting": {"ops": 0.754}},
                    "fatigue_index": {"bullpen_share": "31.0%"},
                },
                "recent": {
                    "found": True,
                    "summary": {"wins": 1, "losses": 0, "draws": 0, "run_diff": 1},
                },
            },
            "away": {
                "summary": {"found": True, "top_batters": []},
                "advanced": {
                    "found": True,
                    "metrics": {"batting": {"ops": 0.731}},
                    "fatigue_index": {"bullpen_share": "28.5%"},
                },
                "recent": {
                    "found": True,
                    "summary": {"wins": 0, "losses": 1, "draws": 0, "run_diff": -1},
                },
            },
            "matchup": {
                "found": True,
                "games": [{"game_id": "prev1"}],
                "summary": {"team1_wins": 1, "team2_wins": 0, "draws": 0},
            },
        }

        supported = _resolve_supported_focuses(
            ["recent_form", "matchup"],
            evidence,
            tool_results,
        )

        assert supported == []

    def test_sanitize_matchup_result_for_postseason_uses_series_scope(self):
        from app.routers.coach import (
            EvidenceSeriesState,
            _sanitize_matchup_result_for_evidence,
        )

        evidence = _build_game_evidence(
            stage_label="KS",
            round_display="한국시리즈",
            league_type_code=5,
            series_state=EvidenceSeriesState(
                stage_label="KS",
                round_display="한국시리즈",
                game_no=5,
                previous_games=4,
                home_team_wins=1,
                away_team_wins=3,
            ),
        )
        matchup = {
            "found": True,
            "games": [{"game_id": f"g{i}"} for i in range(10)],
            "summary": {
                "total_games": 10,
                "team1_wins": 6,
                "team2_wins": 4,
                "draws": 0,
            },
        }

        sanitized = _sanitize_matchup_result_for_evidence(evidence, matchup)

        assert sanitized["summary"]["total_games"] == 4
        assert sanitized["summary"]["team1_wins"] == 1
        assert sanitized["summary"]["team2_wins"] == 3
        assert sanitized["summary"]["draws"] == 0
        assert len(sanitized["games"]) == 4
        assert sanitized["found"] is True

    def test_sanitize_matchup_result_marks_partial_when_series_history_is_short(self):
        from app.routers.coach import (
            EvidenceSeriesState,
            _sanitize_matchup_result_for_evidence,
        )

        evidence = _build_game_evidence(
            stage_label="KS",
            round_display="한국시리즈",
            league_type_code=5,
            series_state=EvidenceSeriesState(
                stage_label="KS",
                round_display="한국시리즈",
                game_no=5,
                previous_games=4,
                confirmed_previous_games=3,
                home_team_wins=1,
                away_team_wins=2,
                series_state_partial=True,
                series_state_hint_mismatch=True,
            ),
        )
        matchup = {
            "found": True,
            "games": [{"game_id": f"g{i}"} for i in range(10)],
            "summary": {
                "total_games": 10,
                "team1_wins": 6,
                "team2_wins": 4,
                "draws": 0,
            },
        }

        sanitized = _sanitize_matchup_result_for_evidence(evidence, matchup)

        assert sanitized["series_state_partial"] is True
        assert sanitized["summary"]["total_games"] == 3
        assert sanitized["summary"]["team1_wins"] == 1
        assert sanitized["summary"]["team2_wins"] == 2
        assert len(sanitized["games"]) == 3

    def test_postseason_headline_uses_round_only_when_series_score_is_partial(self):
        from app.routers.coach import (
            EvidenceSeriesState,
            _build_deterministic_headline,
        )

        evidence = _build_game_evidence(
            away_team_name="한화 이글스",
            home_team_name="LG 트윈스",
            stage_label="KS",
            round_display="한국시리즈",
            league_type_code=5,
            series_state=EvidenceSeriesState(
                stage_label="KS",
                round_display="한국시리즈",
                game_no=5,
                previous_games=4,
                confirmed_previous_games=3,
                home_team_wins=1,
                away_team_wins=2,
                series_state_partial=True,
                series_state_hint_mismatch=True,
            ),
        )

        headline = _build_deterministic_headline(evidence, {})

        assert "5차전" in headline
        assert "승 vs" not in headline

    def test_execute_coach_tools_parallel_passes_postseason_season_id(
        self, monkeypatch
    ):
        from app.routers import coach as coach_router

        captured = {}

        class _FakeConn:
            pass

        class _FakePool:
            def connection(self):
                class _Ctx:
                    async def __aenter__(self_inner):
                        return _FakeConn()

                    async def __aexit__(self_inner, exc_type, exc, tb):
                        return False

                return _Ctx()

        class _FakeDatabaseQueryTool:
            def __init__(self, _conn):
                pass

            async def get_team_summary(self, *_args, **_kwargs):
                return {"found": False}

            async def get_team_advanced_metrics(self, *_args, **_kwargs):
                return {"found": False}

            async def get_team_player_form_signals(self, *_args, **_kwargs):
                return {"found": False, "batters": [], "pitchers": []}

            async def get_team_recent_form(self, *_args, **_kwargs):
                return {"found": False}

        class _FakeGameQueryTool:
            def __init__(self, _conn):
                pass

            async def get_head_to_head(self, *_args, **kwargs):
                captured.update(kwargs)
                return {"found": False, "games": [], "summary": {}}

        monkeypatch.setattr(
            coach_router,
            "DatabaseQueryTool",
            _FakeDatabaseQueryTool,
        )
        import app.tools.game_query as game_query_module

        monkeypatch.setattr(
            game_query_module,
            "GameQueryTool",
            _FakeGameQueryTool,
        )

        async def _run():
            return await coach_router._execute_coach_tools_parallel(
                _FakePool(),
                "LG",
                2025,
                ["matchup"],
                "HH",
                as_of_game_date="2025-10-31",
                exclude_game_id="20251031LGHH0",
                matchup_season_id=264,
            )

        result = asyncio.run(_run())

        assert result["matchup"]["found"] is False
        assert captured["season_id"] == 264
        assert captured["as_of_game_date"] == "2025-10-31"
        assert captured["exclude_game_id"] == "20251031LGHH0"

    def test_execute_coach_tools_parallel_bounds_concurrent_db_checkouts(
        self, monkeypatch
    ):
        """fan-out 세마포어가 동시 DB 커넥션 체크아웃을 상한 이하로 제한한다."""
        from app.routers import coach as coach_router

        # 세마포어 상한을 작게 설정하고 싱글톤 초기화
        monkeypatch.setattr(coach_router, "COACH_DB_FANOUT_MAX", 2)
        coach_router._reset_coach_fanout_semaphore_for_tests()

        active = 0
        peak = 0
        lock = asyncio.Lock()

        class _FakeConn:
            pass

        class _FakePool:
            def connection(self):
                class _Ctx:
                    async def __aenter__(self_inner):
                        nonlocal active, peak
                        async with lock:
                            active += 1
                            peak = max(peak, active)
                        # 동시성 창을 벌리기 위해 잠깐 양보
                        await asyncio.sleep(0.01)
                        return _FakeConn()

                    async def __aexit__(self_inner, exc_type, exc, tb):
                        nonlocal active
                        async with lock:
                            active -= 1
                        return False

                return _Ctx()

        class _FakeDatabaseQueryTool:
            def __init__(self, _conn):
                pass

            async def get_team_summary(self, *_a, **_k):
                await asyncio.sleep(0.01)
                return {"found": False}

            async def get_team_advanced_metrics(self, *_a, **_k):
                await asyncio.sleep(0.01)
                return {"found": False}

            async def get_team_player_form_signals(self, *_a, **_k):
                await asyncio.sleep(0.01)
                return {"found": False, "batters": [], "pitchers": []}

            async def get_team_recent_form(self, *_a, **_k):
                await asyncio.sleep(0.01)
                return {"found": False}

        monkeypatch.setattr(
            coach_router, "DatabaseQueryTool", _FakeDatabaseQueryTool
        )

        async def _run():
            # 홈+원정 = get_team_data 2회 × 4쿼리 = 8개 fan-out, 상한 2로 제한되어야 함
            return await coach_router._execute_coach_tools_parallel(
                _FakePool(),
                "LG",
                2025,
                [],
                "HH",
            )

        asyncio.run(_run())
        coach_router._reset_coach_fanout_semaphore_for_tests()

        assert peak <= 2, f"동시 체크아웃 peak={peak} 가 상한 2를 초과"

    def test_build_focus_data_warning_mentions_fallback_when_all_missing(self):
        from app.routers.coach import _build_focus_data_warning

        warning = _build_focus_data_warning(["recent_form", "matchup"], [])

        assert "보수 요약" in warning
        assert "최근 전력" in warning
        assert "상대 전적" in warning

    def test_format_coach_context(self):
        """Coach 컨텍스트 포맷팅 테스트"""
        from app.routers.coach import _format_coach_context

        tool_results = {
            "home": {
                "summary": {
                    "team_name": "KIA 타이거즈",
                    "year": 2024,
                    "top_batters": [
                        {
                            "player_name": "김도영",
                            "avg": 0.312,
                            "obp": 0.380,
                            "slg": 0.520,
                            "ops": 0.900,
                            "home_runs": 25,
                            "rbi": 80,
                        }
                    ],
                    "top_pitchers": [
                        {
                            "player_name": "양현종",
                            "era": 3.45,
                            "whip": 1.12,
                            "wins": 12,
                            "losses": 5,
                            "saves": 0,
                            "innings_pitched": 150.0,
                        }
                    ],
                    "found": True,
                },
                "advanced": {
                    "team_name": "KIA 타이거즈",
                    "year": 2024,
                    "metrics": {
                        "batting": {"ops": 0.750, "avg": 0.280},
                        "pitching": {
                            "avg_era": 4.20,
                            "qs_rate": "45%",
                            "era_rank": "5위",
                        },
                    },
                    "fatigue_index": {
                        "bullpen_share": "35%",
                        "bullpen_load_rank": "3위 (높을수록 과부하)",
                    },
                    "league_averages": {"bullpen_share": "30%", "era": 4.00},
                    "rankings": {"batting_ops": "4위", "batting_avg": "3위"},
                },
            }
        }

        context = _format_coach_context(tool_results, ["batting", "bullpen"])

        assert "KIA 타이거즈" in context
        assert "김도영" in context
        assert "양현종" in context
        assert "불펜 비중" in context
        assert "35%" in context  # 팀 불펜 비중

    def test_format_coach_context_normalizes_missing_fatigue_values(self):
        from app.routers.coach import _format_coach_context

        tool_results = {
            "home": {
                "summary": {"team_name": "한화 이글스", "year": 2026, "found": True},
                "advanced": {
                    "team_name": "한화 이글스",
                    "year": 2026,
                    "metrics": {"batting": {}, "pitching": {}},
                    "fatigue_index": {
                        "bullpen_share": None,
                        "bullpen_load_rank": None,
                    },
                },
            }
        }

        context = _format_coach_context(tool_results, ["bullpen"])

        assert "**불펜 비중**: -" in context
        assert "**피로도 순위**: -" in context
        assert "None%" not in context

    def test_build_coach_fact_sheet_tracks_confirmed_facts_and_caveats(self):
        from app.routers.coach import (
            _build_coach_fact_sheet,
            _collect_allowed_entity_names,
            assess_game_evidence,
        )

        evidence = _build_game_evidence(
            home_pitcher=None,
            lineup_announced=False,
            home_lineup=[],
            away_lineup=[],
            summary_items=[],
        )
        tool_results = {
            "home": {
                "summary": {
                    "found": True,
                    "top_batters": [
                        {"player_name": "홍창기", "ops": 0.901, "home_runs": 3}
                    ],
                    "top_pitchers": [{"player_name": "임찬규", "era": 3.21, "wins": 1}],
                },
                "advanced": {
                    "found": True,
                    "metrics": {"batting": {"ops": 0.754}},
                    "fatigue_index": {"bullpen_share": "31.0%"},
                },
                "recent": {
                    "found": True,
                    "summary": {"wins": 6, "losses": 4, "draws": 0, "run_diff": 8},
                },
                "player_form_signals": {
                    "found": True,
                    "batters": [
                        {
                            "player_name": "홍창기",
                            "form_score": 68.4,
                            "form_status": "hot",
                            "season_metrics": {"wrc_plus": 141.2, "ops_plus": 132.4},
                            "recent_metrics": {"ops": 1.022},
                            "clutch_metrics": {"recent_wpa_per_pa": 0.0081},
                        }
                    ],
                    "pitchers": [
                        {
                            "player_name": "임찬규",
                            "form_score": 59.2,
                            "form_status": "steady",
                            "season_metrics": {"era_plus": 122.5, "fip_plus": 117.3},
                            "recent_metrics": {"era": 3.11},
                            "clutch_metrics": {"recent_wpa_allowed_per_bf": -0.0034},
                        }
                    ],
                },
            },
            "away": {
                "summary": {
                    "found": True,
                    "top_batters": [
                        {"player_name": "강백호", "ops": 0.877, "home_runs": 4}
                    ],
                    "top_pitchers": [
                        {"player_name": "쿠에바스", "era": 2.98, "wins": 2}
                    ],
                },
                "advanced": {
                    "found": True,
                    "metrics": {"batting": {"ops": 0.731}},
                    "fatigue_index": {"bullpen_share": "28.5%"},
                },
                "recent": {
                    "found": True,
                    "summary": {"wins": 4, "losses": 6, "draws": 0, "run_diff": -5},
                },
                "player_form_signals": {
                    "found": True,
                    "batters": [
                        {
                            "player_name": "강백호",
                            "form_score": 44.1,
                            "form_status": "cold",
                            "season_metrics": {"wrc_plus": 109.4, "ops_plus": 111.2},
                            "recent_metrics": {"ops": 0.641},
                            "clutch_metrics": {"recent_wpa_per_pa": -0.0042},
                        }
                    ],
                    "pitchers": [
                        {
                            "player_name": "쿠에바스",
                            "form_score": 47.7,
                            "form_status": "steady",
                            "season_metrics": {"era_plus": 128.2, "fip_plus": 121.4},
                            "recent_metrics": {"era": 4.72},
                            "clutch_metrics": {"recent_wpa_allowed_per_bf": 0.0041},
                        }
                    ],
                },
            },
            "matchup": {"summary": {"team1_wins": 5, "team2_wins": 3, "draws": 0}},
            "clutch_moments": {
                "found": True,
                "moments": [
                    {
                        "inning_label": "8회말",
                        "outs": 1,
                        "bases_before": "1,2루",
                        "description": "홍창기 결승 2루타",
                        "wpa_delta_pct": 18.4,
                        "batter_name": "홍창기",
                        "pitcher_name": "박영현",
                    }
                ],
            },
        }

        allowed_names = _collect_allowed_entity_names(evidence, tool_results)
        fact_sheet = _build_coach_fact_sheet(
            evidence,
            tool_results,
            allowed_names,
            assess_game_evidence(evidence),
        )

        assert fact_sheet.supported_fact_count >= 8
        assert fact_sheet.starters_confirmed is False
        assert fact_sheet.lineup_confirmed is False
        assert any("선발 정보" in item for item in fact_sheet.caveat_lines)
        assert any("라인업" in item for item in fact_sheet.caveat_lines)
        assert "강백호" in fact_sheet.allowed_entity_names
        assert any("폼 타자" in line for line in fact_sheet.fact_lines)
        assert any("클러치 모먼트" in line for line in fact_sheet.fact_lines)
        assert "0.754" in fact_sheet.allowed_numeric_tokens
        assert (
            "31%" in fact_sheet.allowed_numeric_tokens
            or "31.0%" in fact_sheet.allowed_numeric_tokens
        )

    def test_hint_only_lineup_signal_does_not_count_as_confirmed_lineup(self):
        from app.routers.coach import (
            _build_coach_fact_sheet,
            _collect_allowed_entity_names,
            _should_use_team_level_scheduled_fallback,
            assess_game_evidence,
        )

        evidence = _build_game_evidence(
            lineup_announced=True,
            home_lineup=[],
            away_lineup=[],
            summary_items=[],
        )
        tool_results = {
            "home": {"summary": {}, "advanced": {}, "recent": {}},
            "away": {"summary": {}, "advanced": {}, "recent": {}},
            "matchup": {},
            "clutch_moments": {"found": False, "moments": []},
        }

        assessment = assess_game_evidence(evidence)
        allowed_names = _collect_allowed_entity_names(evidence, tool_results)
        fact_sheet = _build_coach_fact_sheet(
            evidence,
            tool_results,
            allowed_names,
            assessment,
        )

        assert assessment.lineup_announced is False
        assert "missing_lineups" in assessment.root_causes
        assert "game_lineups" not in assessment.used_evidence
        assert not any(
            line.startswith("발표 라인업:") for line in fact_sheet.fact_lines
        )
        assert (
            "라인업 발표 신호는 있으나 선수 구성이 확인되지 않았습니다."
            in fact_sheet.caveat_lines
        )
        assert _should_use_team_level_scheduled_fallback(evidence) is True

    def test_collect_allowed_entity_names_includes_summary_item_players(self):
        from app.routers.coach import _collect_allowed_entity_names

        evidence = _build_game_evidence(summary_items=["결승타 김도영", "오스틴 3안타"])
        tool_results = {
            "home": {"summary": {"top_batters": [], "top_pitchers": []}},
            "away": {"summary": {"top_batters": [], "top_pitchers": []}},
        }

        allowed_names = _collect_allowed_entity_names(evidence, tool_results)

        assert "김도영" in allowed_names
        assert "오스틴" in allowed_names

    def test_validate_response_against_fact_sheet_rejects_unsupported_numeric_claim(
        self,
    ):
        from app.core.coach_grounding import validate_response_against_fact_sheet
        from app.routers.coach import (
            _build_coach_fact_sheet,
            _collect_allowed_entity_names,
            assess_game_evidence,
        )

        evidence = _build_game_evidence()
        tool_results = {
            "home": {
                "summary": {"found": True, "top_batters": [], "top_pitchers": []},
                "advanced": {
                    "found": True,
                    "metrics": {"batting": {"ops": 0.754}},
                    "fatigue_index": {"bullpen_share": "31.0%"},
                },
                "recent": {
                    "found": True,
                    "summary": {"wins": 6, "losses": 4, "draws": 0, "run_diff": 8},
                },
            },
            "away": {
                "summary": {"found": True, "top_batters": [], "top_pitchers": []},
                "advanced": {
                    "found": True,
                    "metrics": {"batting": {"ops": 0.731}},
                    "fatigue_index": {"bullpen_share": "28.5%"},
                },
                "recent": {
                    "found": True,
                    "summary": {"wins": 4, "losses": 6, "draws": 0, "run_diff": -5},
                },
            },
            "matchup": {},
        }
        fact_sheet = _build_coach_fact_sheet(
            evidence,
            tool_results,
            _collect_allowed_entity_names(evidence, tool_results),
            assess_game_evidence(evidence),
        )

        validation = validate_response_against_fact_sheet(
            {
                "headline": "LG가 0.812 OPS 흐름으로 근소 우세",
                "detailed_markdown": "## 최근 전력\n- LG OPS 0.812가 변수입니다.",
                "coach_note": "0.812 수치가 유지되면 흐름을 잡을 수 있습니다.",
                "analysis": {"strengths": [], "weaknesses": [], "risks": []},
                "key_metrics": [],
            },
            fact_sheet,
        )

        assert "unsupported_numeric_claim" in validation.reasons
        assert "0.812" in validation.unsupported_numeric_tokens

    def test_validate_response_against_fact_sheet_rejects_unconfirmed_lineup_claim(
        self,
    ):
        from app.core.coach_grounding import validate_response_against_fact_sheet
        from app.routers.coach import (
            _build_coach_fact_sheet,
            _collect_allowed_entity_names,
            assess_game_evidence,
        )

        evidence = _build_game_evidence(
            lineup_announced=False,
            home_lineup=[],
            away_lineup=[],
        )
        tool_results = {
            "home": {
                "summary": {"found": True},
                "advanced": {"found": True},
                "recent": {"found": False},
            },
            "away": {
                "summary": {"found": True},
                "advanced": {"found": True},
                "recent": {"found": False},
            },
            "matchup": {},
        }
        fact_sheet = _build_coach_fact_sheet(
            evidence,
            tool_results,
            _collect_allowed_entity_names(evidence, tool_results),
            assess_game_evidence(evidence),
        )

        validation = validate_response_against_fact_sheet(
            {
                "headline": "라인업이 승부처를 좌우한다",
                "detailed_markdown": "## 최근 전력\n- 라인업 상위 타순의 장타력이 핵심입니다.",
                "coach_note": "라인업 짜임새가 좋습니다.",
                "analysis": {"strengths": [], "weaknesses": [], "risks": []},
                "key_metrics": [],
            },
            fact_sheet,
        )

        assert "unconfirmed_lineup_claim" in validation.reasons

    def test_validate_response_against_fact_sheet_rejects_semantically_empty_response(
        self,
    ):
        from app.core.coach_grounding import validate_response_against_fact_sheet
        from app.routers.coach import (
            _build_coach_fact_sheet,
            _collect_allowed_entity_names,
            assess_game_evidence,
        )

        evidence = _build_game_evidence()
        tool_results = {
            "home": {
                "summary": {"found": True},
                "advanced": {"found": True},
                "recent": {"found": True, "summary": {"wins": 6, "losses": 4}},
            },
            "away": {
                "summary": {"found": True},
                "advanced": {"found": True},
                "recent": {"found": True, "summary": {"wins": 4, "losses": 6}},
            },
            "matchup": {},
        }
        fact_sheet = _build_coach_fact_sheet(
            evidence,
            tool_results,
            _collect_allowed_entity_names(evidence, tool_results),
            assess_game_evidence(evidence),
        )

        validation = validate_response_against_fact_sheet(
            {
                "headline": "AI 코치 분석 요약",
                "detailed_markdown": "## 최근 전력\n## 불펜 상태",
                "coach_note": "",
                "analysis": {
                    "summary": "",
                    "verdict": "",
                    "strengths": [],
                    "weaknesses": [],
                    "risks": [],
                    "why_it_matters": [],
                    "swing_factors": [],
                    "watch_points": [],
                    "uncertainty": [],
                },
                "key_metrics": [],
            },
            fact_sheet,
        )

        assert "empty_response" in validation.reasons

    def test_disallowed_entity_detection_ignores_generic_korean_nouns(self):
        from app.routers.coach import _response_mentions_disallowed_entities

        response_data = {
            "headline": "실데이터 기반 매치업",
            "detailed_markdown": (
                "## 최근 흐름\n"
                "- 초반 흐름과 상승세를 확인해야 합니다.\n"
                "- 선취점과 공격력, 연장전 변수도 봐야 합니다."
            ),
            "coach_note": "경기 초반 흐름과 이닝 소화력이 중요하고 반격할 타이밍이 필요합니다.",
            "analysis": {
                "strengths": ["최근 흐름이 안정적이고 선발진 부담이 낮습니다."],
                "weaknesses": ["최하위권 공격력 변수와 후반 변수가 있습니다."],
            },
        }

        assert (
            _response_mentions_disallowed_entities(
                response_data,
                {"한화 이글스", "LG 트윈스", "폰세", "오스틴"},
            )
            is False
        )

    def test_disallowed_entity_detection_ignores_generic_adjectives(self):
        from app.routers.coach import _find_disallowed_entities

        response_data = {
            "headline": "최소한 불펜 변수는 체크해야 한다",
            "detailed_markdown": (
                "## 코치 판단\n"
                "- 유사한 불펜 운영 패턴과 고위험 상황을 먼저 봐야 합니다.\n"
                "- 최소한 장타 허용은 줄여야 합니다."
            ),
            "coach_note": "유사해 보이는 흐름이라도 결측 데이터가 많습니다.",
            "analysis": {
                "strengths": [],
                "weaknesses": ["고위험 구간 판단은 보수적으로 가야 합니다."],
            },
        }

        assert (
            _find_disallowed_entities(response_data, {"한화 이글스", "SSG 랜더스"})
            == []
        )

    def test_disallowed_entity_detection_ignores_general_terms_with_suffixes(self):
        from app.routers.coach import _find_disallowed_entities

        response_data = {
            "headline": "불펜 운영 변수",
            "detailed_markdown": (
                "## 체크 포인트\n"
                "- 불펜 비중 데이터가 공개될 경우 하이 레버리지 상황을 다시 봐야 합니다.\n"
                "- 득실점 마진으로 경기 중반 운영 선택지가 달라질 수 있습니다."
            ),
            "coach_note": "불펜 카드가 공개될 경우와 득실점 마진으로 이어지는 운영 흐름을 함께 봐야 합니다.",
            "analysis": {
                "strengths": [],
                "weaknesses": [
                    "득실점 마진으로 접전 운영 선택지가 달라질 수 있습니다."
                ],
            },
        }

        assert (
            _find_disallowed_entities(response_data, {"한화 이글스", "SSG 랜더스"})
            == []
        )

    def test_disallowed_entity_detection_ignores_narrative_tokens_near_player_context(
        self,
    ):
        from app.routers.coach import _find_disallowed_entities

        response_data = {
            "headline": "한화 이글스 승리, 결승타 최재훈이 흐름을 바꿨다",
            "coach_note": "결승타로 이어진 장면과 최재훈의 타석 운영을 함께 복기해야 합니다.",
            "analysis": {
                "summary": "한화 이글스는 최재훈의 장타와 후속 득점 연결로 주도권을 잡았습니다.",
                "watch_points": [
                    "'결승타 최재훈 (2회 2사 1,2루서 좌월 홈런)' 직전의 주자 상황과 투수 교체 선택이 어떻게 이어졌는지 다시 볼 필요가 있습니다."
                ],
                "strengths": [],
                "weaknesses": [],
            },
        }

        assert (
            _find_disallowed_entities(
                response_data,
                {"한화 이글스", "NC 다이노스", "최재훈"},
            )
            == []
        )

    def test_disallowed_entity_detection_rejects_unknown_player_name(self):
        from app.routers.coach import _response_mentions_disallowed_entities

        response_data = {
            "headline": "김도영이 승부처를 만든다",
            "detailed_markdown": "## 관전 포인트\n- 김도영의 장타력이 변수입니다.",
            "coach_note": "김도영 중심 대응이 필요합니다.",
            "analysis": {
                "strengths": ["김도영의 최근 타격감이 좋습니다."],
                "weaknesses": [],
            },
        }

        assert (
            _response_mentions_disallowed_entities(
                response_data,
                {"한화 이글스", "LG 트윈스", "폰세", "오스틴"},
            )
            is True
        )

    def test_sanitize_disallowed_entity_names_rewrites_unknown_player_tokens(self):
        from app.routers.coach import (
            _find_disallowed_entities,
            _response_mentions_disallowed_entities,
            _sanitize_response_disallowed_entities,
        )

        response_data = {
            "headline": "김도영이 승부처를 만든다",
            "detailed_markdown": "## 관전 포인트\n- 김도영의 장타력이 변수입니다.",
            "coach_note": "김도영과 오스틴을 함께 묶어 보면 안 됩니다.",
            "analysis": {
                "summary": "김도영이 흐름을 흔들 수 있습니다.",
                "verdict": "김도영의 초반 타석이 중요합니다.",
                "strengths": ["김도영이 최근 타격감이 좋습니다."],
                "weaknesses": [],
                "risks": [
                    {
                        "area": "batting",
                        "level": 1,
                        "description": "김도영의 장타 허용이 변수입니다.",
                    }
                ],
                "why_it_matters": ["김도영과의 승부를 피하기 어렵습니다."],
                "swing_factors": ["김도영이 출루하면 흐름이 달라집니다."],
                "watch_points": ["김도영의 첫 타석 결과"],
                "uncertainty": [],
            },
            "key_metrics": [
                {
                    "label": "폼 진단",
                    "value": "김도영 상승세",
                    "status": "good",
                    "trend": "up",
                    "is_critical": True,
                }
            ],
        }
        allowed_names = {"한화 이글스", "LG 트윈스", "오스틴"}

        disallowed = _find_disallowed_entities(response_data, allowed_names)
        sanitized = _sanitize_response_disallowed_entities(response_data, disallowed)

        assert "김도영" in disallowed
        assert sanitized["headline"] == "핵심 선수가 승부처를 만든다"
        assert "핵심 선수의 장타력" in sanitized["detailed_markdown"]
        assert "핵심 선수와 오스틴" in sanitized["coach_note"]
        assert sanitized["analysis"]["verdict"] == "핵심 선수의 초반 타석이 중요합니다."
        assert sanitized["key_metrics"][0]["value"] == "핵심 선수 상승세"
        assert _response_mentions_disallowed_entities(sanitized, allowed_names) is False

    def test_scheduled_unconfirmed_lineup_sanitizer_removes_unconfirmed_players(self):
        from app.routers.coach import (
            _sanitize_scheduled_unconfirmed_lineup_entities,
        )

        evidence = _build_game_evidence(
            home_team_code="HH",
            away_team_code="SSG",
            home_team_name="한화 이글스",
            away_team_name="SSG 랜더스",
            home_pitcher=None,
            away_pitcher=None,
            lineup_announced=False,
            home_lineup=[],
            away_lineup=[],
            summary_items=[],
        )
        response_data = {
            "headline": "한화 이글스 vs SSG 랜더스, 불펜 데이터 부재 속 타격 생산성 변수",
            "detailed_markdown": (
                "## 코치 판단\n"
                "- 페라자(OPS 1.163)의 생산성은 변수지만 라인업은 아직 공개되지 않았습니다.\n"
                "- 상위 타선(wRC+ 287.6)과 페라자(wRC+ 213.9)의 타격 흐름 지속 여부"
            ),
            "coach_note": "페라자와 강백호 비교는 아직 이릅니다.",
            "analysis": {
                "summary": "페라자와 강백호의 시즌 OPS 비교는 참고 수준입니다.",
                "verdict": (
                    "라인업 미확정 상황이라도 한화 이글스의 페라자(OPS 1.163) 등 "
                    "상위 타선의 생산성이 변수가 될 전망입니다."
                ),
                "strengths": ["한화 이글스 페라자의 높은 wRC+ 216.7 및 OPS 1.163"],
                "weaknesses": [],
                "risks": [],
                "why_it_matters": ["페라자와 강백호를 동시에 막아야 합니다."],
                "swing_factors": ["페라자 첫 타석 결과"],
                "watch_points": [
                    "강백호 장타 허용 여부",
                    "페라자와 SSG 랜더스 상위 타선 폼 유지 여부",
                ],
                "uncertainty": ["라인업 미확정"],
            },
            "key_metrics": [
                {
                    "label": "한화 이글스 핵심 타격",
                    "value": "페라자 OPS 1.163 / 강백호 OPS 0.840",
                    "status": "warning",
                    "trend": "neutral",
                    "is_critical": True,
                },
                {
                    "label": "한화 이글스 페라자 폼",
                    "value": "상승 (점수 88.4)",
                    "status": "good",
                    "trend": "up",
                    "is_critical": True,
                },
            ],
        }

        sanitized = _sanitize_scheduled_unconfirmed_lineup_entities(
            response_data,
            evidence=evidence,
            grounding_reasons=[
                "missing_lineups",
                "missing_starters",
                "missing_summary",
            ],
        )
        serialized = json.dumps(sanitized, ensure_ascii=False)

        assert "페라자" not in serialized
        assert "강백호" not in serialized
        assert sanitized["key_metrics"][0]["label"] == "상위 타선 생산성"
        assert (
            sanitized["key_metrics"][0]["value"]
            == "라인업 미확정으로 개별 타자 비교 보류"
        )
        assert sanitized["key_metrics"][1]["label"] == "상위 타선 흐름"
        assert (
            sanitized["key_metrics"][1]["value"]
            == "라인업 미확정으로 개별 타자 폼 비교 보류"
        )
        assert "상위 타선" in sanitized["analysis"]["verdict"]

    def test_scheduled_unconfirmed_lineup_sanitizer_keeps_confirmed_lineup_players(
        self,
    ):
        from app.routers.coach import (
            _sanitize_scheduled_unconfirmed_lineup_entities,
        )

        evidence = _build_game_evidence(
            home_team_code="HH",
            away_team_code="SSG",
            home_team_name="한화 이글스",
            away_team_name="SSG 랜더스",
            lineup_announced=True,
            home_lineup=["페라자"],
            away_lineup=[],
            summary_items=[],
        )
        response_data = {
            "headline": "페라자 컨디션 체크",
            "detailed_markdown": "## 체크 포인트\n- 페라자 첫 타석 결과",
            "coach_note": "페라자 중심 대응",
            "analysis": {
                "summary": "페라자 출루 여부가 중요합니다.",
                "verdict": "페라자 타석 운영이 핵심입니다.",
                "strengths": ["페라자 컨디션 양호"],
                "weaknesses": [],
                "risks": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "key_metrics": [
                {
                    "label": "중심 타자",
                    "value": "페라자 출루율 체크",
                    "status": "good",
                    "trend": "up",
                    "is_critical": True,
                }
            ],
        }

        sanitized = _sanitize_scheduled_unconfirmed_lineup_entities(
            response_data,
            evidence=evidence,
            grounding_reasons=["missing_lineups"],
        )
        serialized = json.dumps(sanitized, ensure_ascii=False)

        assert "페라자" in serialized

    def test_scheduled_unconfirmed_lineup_sanitizer_repairs_placeholder_artifacts_with_team_level_copy(
        self,
    ):
        from app.routers.coach import (
            _find_missing_focus_sections,
            _sanitize_scheduled_unconfirmed_lineup_entities,
        )

        evidence = _build_game_evidence(
            home_team_code="HH",
            away_team_code="SSG",
            home_team_name="한화 이글스",
            away_team_name="SSG 랜더스",
            home_pitcher=None,
            away_pitcher=None,
            lineup_announced=False,
            home_lineup=[],
            away_lineup=[],
            summary_items=[],
        )
        tool_results = {
            "home": {
                "summary": {"found": True},
                "advanced": {"metrics": {"batting": {"ops": 0.860}}},
                "recent": {"found": False},
                "player_form_signals": {
                    "found": True,
                    "batters": [
                        {
                            "player_name": "페라자",
                            "form_score": 83.1,
                            "form_status": "hot",
                        }
                    ],
                    "pitchers": [],
                },
            },
            "away": {
                "summary": {"found": True},
                "advanced": {"metrics": {"batting": {"ops": 1.152}}},
                "recent": {"found": False},
                "player_form_signals": {
                    "found": True,
                    "batters": [
                        {
                            "player_name": "최정",
                            "form_score": 93.0,
                            "form_status": "hot",
                        }
                    ],
                    "pitchers": [],
                },
            },
            "matchup": {},
        }
        response_data = {
            "headline": "한화 이글스 vs SSG 랜더스, 불펜 데이터 결측에 따른 변수 분석",
            "coach_note": (
                "불펜 데이터 결측으로 인해 타격 지표가 운영의 핵심 변수이며, "
                "SSG 랜더스의 상위 타선 생산력이 상대 불펜의 과부하를 상위 타선 가능성이 큽니다."
            ),
            "detailed_markdown": (
                "## 코치 판단\n"
                "- 불펜 관련 상위 타선 데이터가 모두 결측되어, 수치 기반의 대응력 비교가 불가능한 상황입니다.\n\n"
                "## 왜 중요한가\n"
                "- 불펜 지표가 없는 상태에서 유일한 비교 가능 지표는 타격 생산성입니다.\n\n"
                "## 불펜 상태\n"
                "- 한화 이글스와 SSG 랜더스 모두 불펜 비중 데이터가 확인되지 않아, 최근 소모 흐름이나 접전 대응력을 분석할 근거가 부족합니다.\n\n"
                "## 최근 전력\n"
                "- 양 팀 모두 불펜 비중 및 소모 흐름에 대한 구체적인 수치가 확인되지 않아 운영 예측에 한계가 있습니다."
            ),
            "analysis": {
                "summary": "양 팀 모두 불펜 비중 및 소모 흐름에 대한 구체적인 수치가 확인되지 않아 운영 예측에 한계가 있습니다.",
                "verdict": (
                    "불펜의 상위 타선 지표가 결측된 상태이므로, 현재로서는 SSG 랜더스가 "
                    "팀 OPS 1.152의 타격 우위를 통해 불펜 소모를 상위 타선 가능성이 높습니다."
                ),
                "strengths": [
                    "SSG 랜더스는 상위 타선과 상위 타선을 필두로 한 상위 타선 타격 생산성을 보유하고 있습니다."
                ],
                "weaknesses": [
                    "한화 이글스는 SSG 랜더스 대비 상대적으로 낮은 팀 OPS(0.860)를 기록하고 있습니다."
                ],
                "risks": [
                    {
                        "area": "overall",
                        "level": 1,
                        "description": "양 팀 불펜 비중 및 최근 소모 데이터가 모두 결측되어 접전 대응력 판단이 불가능합니다.",
                    }
                ],
                "why_it_matters": [
                    "불펜의 가용 자원과 소모도가 확인되지 않은 상태에서 타격 생산성의 차이는 경기 후반 투수 교체 타이밍과 과부하 정도에 직접적인 영향을 줍니다."
                ],
                "swing_factors": [
                    "데이터가 상위 타선 불펜진이 SSG 랜더스의 상위 타선 타선(상위 타선 폼 점수 93.0)을 얼마나 억제하느냐가 핵심 구간이 될 것입니다."
                ],
                "watch_points": [
                    "경기 중 실제 불펜 투입 시점과 투구 수 변화를 통해 실시간 소모 흐름을 체크해야 합니다."
                ],
                "uncertainty": [
                    "양 팀 불펜의 구체적인 비중, 최근 투구 이력, 선발 투수 정보가 미확정 상태입니다."
                ],
            },
            "key_metrics": [
                {
                    "label": "한화 이글스 불펜 비중",
                    "value": "데이터 부족",
                    "status": "warning",
                    "trend": "neutral",
                    "is_critical": True,
                },
                {
                    "label": "SSG 랜더스 불펜 비중",
                    "value": "데이터 부족",
                    "status": "warning",
                    "trend": "neutral",
                    "is_critical": True,
                },
                {
                    "label": "팀 OPS 비교",
                    "value": "한화 이글스 0.860 / SSG 랜더스 1.152",
                    "status": "good",
                    "trend": "neutral",
                    "is_critical": False,
                },
            ],
        }

        sanitized = _sanitize_scheduled_unconfirmed_lineup_entities(
            response_data,
            evidence=evidence,
            grounding_reasons=[
                "missing_lineups",
                "missing_starters",
                "missing_summary",
            ],
            tool_results=tool_results,
            resolved_focus=["recent_form", "bullpen"],
        )
        serialized = json.dumps(sanitized, ensure_ascii=False)

        assert "페라자" not in serialized
        assert "최정" not in serialized
        assert "상위 타선" not in serialized
        assert "핵심 선수" not in serialized
        assert "## 최근 전력\n-" in sanitized["detailed_markdown"]
        assert "## 불펜 상태\n-" in sanitized["detailed_markdown"]
        assert _find_missing_focus_sections(sanitized, ["recent_form", "bullpen"]) == []
