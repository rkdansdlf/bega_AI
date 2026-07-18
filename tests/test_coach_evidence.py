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

class TestCoachEvidenceHelpers:
    """Coach 경기 근거 헬퍼 테스트"""

    def test_resolve_league_type_code_hint_accepts_preseason_context(self):
        from app.routers.coach import _resolve_league_type_code_hint

        hint_code = _resolve_league_type_code_hint(
            {"league_type": "PRE", "league_type_code": 1}
        )

        assert hint_code == 1

    def test_reconcile_series_state_with_hint_updates_game_no(self):
        from app.routers.coach import (
            EvidenceSeriesState,
            GameEvidence,
            _reconcile_series_state_with_hint,
        )

        evidence = GameEvidence(
            season_year=2025,
            home_team_code="SS",
            away_team_code="SSG",
            home_team_name="삼성 라이온즈",
            away_team_name="SSG 랜더스",
            game_id="20251011SSSK0",
            league_type_code=5,
            stage_label="KS",
            round_display="한국시리즈",
            stage_game_no_hint=2,
        )
        series_state = EvidenceSeriesState(
            stage_label="KS",
            round_display="한국시리즈",
            game_no=1,
            previous_games=0,
        )

        reconciled = _reconcile_series_state_with_hint(series_state, evidence)

        assert reconciled is not None
        assert reconciled.game_no == 2
        assert reconciled.previous_games == 1
        assert reconciled.confirmed_previous_games == 0
        assert reconciled.series_state_partial is True
        assert reconciled.series_state_hint_mismatch is True

    def test_postponed_game_status_bucket_is_not_completed(self):
        from app.routers.coach import _normalize_game_status_bucket

        assert _normalize_game_status_bucket("POSTPONED") == "UNKNOWN"
        assert _normalize_game_status_bucket("CANCELLED") == "UNKNOWN"

    def test_draw_game_status_bucket_is_completed(self):
        from app.routers.coach import _normalize_game_status_bucket

        assert _normalize_game_status_bucket("DRAW") == "COMPLETED"
        assert _normalize_game_status_bucket("TIE") == "COMPLETED"

    def test_unavailable_game_status_message_is_defined_for_cancelled_and_postponed(
        self,
    ):
        from app.routers.coach import _get_unavailable_game_status_message

        assert (
            _get_unavailable_game_status_message("CANCELLED")
            == "취소된 경기는 AI 코치 분석을 제공하지 않습니다."
        )
        assert (
            _get_unavailable_game_status_message("POSTPONED")
            == "연기된 경기는 일정 확정 후 AI 코치 분석을 제공합니다."
        )
        assert (
            _get_unavailable_game_status_message("SUSPENDED")
            == "중단된 경기는 경기 상태가 확정된 뒤 AI 코치 분석을 제공합니다."
        )
        assert _get_unavailable_game_status_message("SCHEDULED") is None

    def test_has_canonical_game_team_pair_requires_both_teams_to_be_canonical(self):
        from app.routers.coach import _has_canonical_game_team_pair

        assert _has_canonical_game_team_pair("KT", "NC") is True
        assert _has_canonical_game_team_pair("롯데0", "0LG") is False
        assert _has_canonical_game_team_pair("KT", None) is False

    def test_completed_review_without_clutch_data_is_partial(self):
        from app.routers.coach import _determine_data_quality

        evidence = _build_game_evidence(
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
        )
        tool_results = {
            "home": {
                "summary": {"found": True},
                "advanced": {"found": True},
            },
            "away": {
                "summary": {"found": True},
                "advanced": {"found": True},
            },
            "clutch_moments": {"found": False, "moments": []},
        }

        assert _determine_data_quality(evidence, tool_results) == "partial"

    def test_completed_review_fact_sheet_marks_missing_clutch_reason(self):
        from app.routers.coach import _build_coach_fact_sheet, assess_game_evidence

        evidence = _build_game_evidence(
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
        )
        tool_results = {
            "home": {
                "summary": {"found": True},
                "advanced": {"found": True},
            },
            "away": {
                "summary": {"found": True},
                "advanced": {"found": True},
            },
            "clutch_moments": {"found": False, "moments": []},
        }

        fact_sheet = _build_coach_fact_sheet(
            evidence,
            tool_results,
            allowed_names=set(),
            assessment=assess_game_evidence(evidence),
        )

        assert "missing_clutch_moments" in fact_sheet.reasons
        assert "WPA 기반 승부처 데이터가 부족합니다." in fact_sheet.warnings

    def test_log_coach_stream_meta_includes_grounding_context(self, caplog):
        from app.routers.coach import _log_coach_stream_meta

        payload = {
            "request_mode": "manual_detail",
            "cache_state": "COMPLETED",
            "validation_status": "success",
            "generation_mode": "llm_manual",
            "cached": False,
            "in_progress": False,
            "data_quality": "partial",
            "supported_fact_count": 6,
            "grounding_reasons": ["missing_starters", "focus_data_unavailable"],
            "grounding_warnings": [
                "선발 정보가 완전하지 않아 선발 관련 표현을 제한합니다.",
                "요청한 focus 중 상대 전적 근거가 부족해 확인 가능한 항목만 분석합니다.",
            ],
            "used_evidence": ["game", "team_recent_form"],
            "resolved_focus": ["starter", "matchup"],
            "missing_focus_sections": ["matchup"],
        }

        with caplog.at_level(logging.INFO):
            _log_coach_stream_meta(payload, game_id="202503120001")

        assert "game_id=202503120001" in caplog.text
        assert "data_quality=partial" in caplog.text
        assert (
            "grounding_reasons=['missing_starters', 'focus_data_unavailable']"
            in caplog.text
        )

    def test_preview_verdict_keeps_pitcher_sentence_well_formed(self):
        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence(
            home_team_name="KT 위즈",
            away_team_name="LG 트윈스",
            home_pitcher="패트릭",
            away_pitcher="송승기",
        )
        tool_results = {
            "home": {
                "summary": {"found": True},
                "advanced": {"metrics": {"batting": {"ops": 0.790}}},
                "recent": {"found": False},
                "player_form_signals": {},
            },
            "away": {
                "summary": {"found": True},
                "advanced": {"metrics": {"batting": {"ops": 0.765}}},
                "recent": {"found": False},
                "player_form_signals": {},
            },
            "matchup": {},
            "clutch_moments": {"found": False, "moments": []},
        }

        payload = _build_deterministic_coach_response(evidence, tool_results)

        assert "좌우합니다.에 따라" not in payload["analysis"]["verdict"]
        assert "송승기과" not in payload["analysis"]["verdict"]
        assert "송승기와 KT 위즈 패트릭" in payload["analysis"]["verdict"]

    def test_preview_close_margin_fallback_avoids_stock_phrases(self):
        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence(
            home_team_name="한화 이글스",
            away_team_name="두산 베어스",
            home_pitcher=None,
            away_pitcher=None,
            lineup_announced=False,
            home_lineup=[],
            away_lineup=[],
        )
        tool_results = {
            "home": {
                "summary": {"found": True},
                "advanced": {"found": False},
                "recent": {
                    "found": True,
                    "summary": {"wins": 4, "losses": 1, "draws": 0, "run_diff": 6},
                },
                "player_form_signals": {},
            },
            "away": {
                "summary": {"found": True},
                "advanced": {"found": False},
                "recent": {
                    "found": True,
                    "summary": {"wins": 2, "losses": 3, "draws": 0, "run_diff": -2},
                },
                "player_form_signals": {},
            },
            "matchup": {},
            "clutch_moments": {"found": False, "moments": []},
        }

        payload = _build_deterministic_coach_response(evidence, tool_results)

        assert (
            "근소 우세지만 격차는 크지 않습니다." not in payload["analysis"]["verdict"]
        )
        assert (
            "초반 선취점 이후 불펜 투입 시점이 경기 방향을 크게 흔들 변수입니다."
            not in payload["analysis"]["swing_factors"][0]
        )
        assert "한화 이글스" in payload["analysis"]["verdict"]
        assert payload["analysis"]["verdict"].endswith("한 발 앞섭니다.")
        assert (
            payload["analysis"]["watch_points"][0]
            == "첫 득점 직후 어느 팀이 먼저 불펜 카드로 반응하는지 확인할 필요가 있습니다."
        )

    def test_compact_coach_note_does_not_repeat_or_cut_mid_sentence(self):
        from app.routers.coach import _build_compact_coach_note

        coach_note = _build_compact_coach_note(
            [
                "LG 트윈스가 기초 지표 우위를 실제 결과로 연결했습니다.",
                "1회초 김현수 타석의 WPA -10.5%p 변동이 실제 승부처였습니다.",
                "1회초 김현수 타석의 WPA -10.5%p 변동이 실제 승부처였습니다.",
            ],
            max_length=140,
        )

        assert coach_note.endswith(".")
        assert coach_note.count("실제 승부처였습니다.") == 1

    def test_rebuild_scheduled_coach_note_avoids_duplicate_prefix_and_stays_compact(
        self,
    ):
        from app.routers.coach import _rebuild_scheduled_coach_note

        note = _rebuild_scheduled_coach_note(
            {
                "watch_points": [
                    "첫 득점 직후 어느 팀이 먼저 불펜 카드로 반응하는지 확인할 필요가 있습니다."
                ],
                "uncertainty": [
                    "라인업 미발표라 타순 기반 세부 매치업은 경기 직전까지 달라질 수 있습니다."
                ],
                "swing_factors": [
                    "선발 발표 전이라 첫 투수 교체 시점이 핵심 변수입니다."
                ],
                "verdict": "롯데 자이언츠가 팀 타격 생산성에서 한 발 앞섭니다.",
            }
        )

        assert "핵심 변수는 선발 발표 전이라" not in note
        assert "선발 발표 전이라 첫 투수 교체 시점이 핵심 변수입니다." in note
        assert len(note) <= 180

    def test_completed_deterministic_review_avoids_unknown_batter_placeholder(self):
        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence(
            home_team_code="LG",
            away_team_code="KIA",
            home_team_name="LG 트윈스",
            away_team_name="KIA 타이거즈",
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            home_score=7,
            away_score=2,
            winning_team_code="LG",
            winning_team_name="LG 트윈스",
            home_pitcher="송승기",
            away_pitcher="양현종",
            summary_items=["결승타 문성주 (1회 1사 만루서 밀어내기 4구)"],
        )
        tool_results = {
            "home": {
                "recent": {
                    "found": True,
                    "summary": {"wins": 3, "losses": 7, "draws": 0, "run_diff": -12},
                },
                "advanced": {"metrics": {"batting": {"ops": 0.740}}},
                "summary": {"found": True},
            },
            "away": {
                "recent": {
                    "found": True,
                    "summary": {"wins": 3, "losses": 6, "draws": 1, "run_diff": -9},
                },
                "advanced": {"metrics": {"batting": {"ops": 0.779}}},
                "summary": {"found": True},
            },
            "matchup": {
                "found": True,
                "summary": {"team2_wins": 1, "team1_wins": 0, "draws": 0},
            },
            "clutch_moments": {
                "found": True,
                "moments": [
                    {
                        "inning_label": "1회말",
                        "batter_name": None,
                        "description": "8번타자 이재원",
                        "wpa_delta_pct": 21.2,
                    }
                ],
            },
        }

        payload = _build_deterministic_coach_response(
            evidence,
            tool_results,
            resolved_focus=["recent_form", "bullpen", "starter", "matchup", "batting"],
            grounding_warnings=[],
        )

        rendered = json.dumps(payload, ensure_ascii=False)
        assert "타자 미상" not in rendered
        assert (
            "1회말 8번타자 이재원 장면의 WPA 21.2%p 변동이 실제 승부처였습니다."
            in rendered
        )

    def test_completed_clutch_moment_text_deduplicates_inning_prefix(self):
        from app.routers.coach import _completed_clutch_moment_text

        moment = {
            "inning_label": "6회초",
            "description": "6회초 SSG 공격",
            "wpa_delta_pct": -16.4,
        }

        assert _completed_clutch_moment_text(moment) == "6회초 SSG 공격 장면"
        assert (
            _completed_clutch_moment_text(moment, with_wpa=True)
            == "6회초 SSG 공격 장면의 WPA -16.4%p 변동"
        )

    def test_completed_clutch_moment_text_ignores_separator_description(self):
        from app.routers.coach import _completed_clutch_moment_text

        moment = {
            "inning_label": "9회말",
            "description": "=" * 80,
            "wpa_delta_pct": -50.0,
        }

        assert _completed_clutch_moment_text(moment) == "9회말 핵심 장면"
        assert (
            _completed_clutch_moment_text(moment, with_wpa=True)
            == "9회말 핵심 장면의 WPA -50.0%p 변동"
        )

    def test_completed_deterministic_metrics_compacts_long_wpa_values(self):
        from app.core.coach_validator import MAX_KEY_METRIC_VALUE_LENGTH
        from app.routers.coach import _build_deterministic_metrics

        evidence = _build_game_evidence(
            home_team_code="HH",
            away_team_code="SK",
            home_team_name="한화 이글스",
            away_team_name="SSG 랜더스",
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            home_score=8,
            away_score=9,
            winning_team_code="SK",
            winning_team_name="SSG 랜더스",
        )
        tool_results = {
            "home": {
                "recent": {
                    "found": True,
                    "summary": {"wins": 2, "losses": 3, "draws": 0, "run_diff": -2},
                },
                "advanced": {"metrics": {"batting": {"ops": 0.711}}},
            },
            "away": {
                "recent": {
                    "found": True,
                    "summary": {"wins": 3, "losses": 2, "draws": 0, "run_diff": 4},
                },
                "advanced": {"metrics": {"batting": {"ops": 0.752}}},
            },
            "clutch_moments": {
                "found": True,
                "moments": [
                    {
                        "inning_label": "9회말",
                        "batter_name": None,
                        "description": "=" * 80,
                        "wpa_delta_pct": -50.0,
                    }
                ],
            },
        }

        metrics = _build_deterministic_metrics(evidence, tool_results)

        assert any(metric["label"] == "최대 WPA 변동" for metric in metrics)
        assert all(
            len(metric["value"]) <= MAX_KEY_METRIC_VALUE_LENGTH for metric in metrics
        )

    def test_completed_deterministic_review_separates_section_roles(self):
        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence(
            home_team_code="LG",
            away_team_code="KIA",
            home_team_name="LG 트윈스",
            away_team_name="KIA 타이거즈",
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            home_score=7,
            away_score=2,
            winning_team_code="LG",
            winning_team_name="LG 트윈스",
            home_pitcher="송승기",
            away_pitcher="양현종",
            summary_items=["결승타 문성주 (1회 1사 만루서 밀어내기 4구)"],
        )
        tool_results = {
            "home": {
                "recent": {
                    "found": True,
                    "summary": {"wins": 3, "losses": 7, "draws": 0, "run_diff": -12},
                },
                "advanced": {"metrics": {"batting": {"ops": 0.740}}},
                "summary": {"found": True},
            },
            "away": {
                "recent": {
                    "found": True,
                    "summary": {"wins": 3, "losses": 6, "draws": 1, "run_diff": -9},
                },
                "advanced": {"metrics": {"batting": {"ops": 0.779}}},
                "summary": {"found": True},
            },
            "matchup": {
                "found": True,
                "summary": {"team2_wins": 1, "team1_wins": 0, "draws": 0},
            },
            "clutch_moments": {
                "found": True,
                "moments": [
                    {
                        "inning_label": "1회말",
                        "batter_name": None,
                        "description": "8번타자 이재원",
                        "wpa_delta_pct": 21.2,
                    }
                ],
            },
        }

        payload = _build_deterministic_coach_response(
            evidence,
            tool_results,
            resolved_focus=["recent_form", "bullpen", "starter", "matchup", "batting"],
            grounding_warnings=[],
        )

        assert (
            payload["analysis"]["verdict"]
            == "LG 트윈스 승리의 분기점은 1회말 승부처 대응이었습니다."
        )
        assert (
            payload["analysis"]["swing_factors"][0]
            == "1회말 8번타자 이재원 장면의 WPA 21.2%p 변동이 실제 승부처였습니다."
        )
        assert (
            payload["analysis"]["watch_points"][0]
            == "해당 승부처 직전 배터리 선택과 주자 운영을 다시 볼 필요가 있습니다."
        )
        assert (
            "## 결과 진단\n- LG 트윈스 승리의 분기점은 1회말 승부처 대응이었습니다."
            in payload["detailed_markdown"]
        )
        assert (
            "## 실제 전환점\n- 1회말 8번타자 이재원 장면의 WPA 21.2%p 변동이 실제 승부처였습니다."
            in payload["detailed_markdown"]
        )
        assert (
            "## 다시 볼 장면\n- 해당 승부처 직전 배터리 선택과 주자 운영을 다시 볼 필요가 있습니다."
            in payload["detailed_markdown"]
        )

    def test_scheduled_manual_detail_with_supported_facts_does_not_short_circuit(self):
        from app.core.coach_grounding import CoachFactSheet
        from app.routers.coach import (
            COACH_REQUEST_MODE_MANUAL,
            _should_short_circuit_to_deterministic_response,
        )

        fact_sheet = CoachFactSheet(
            fact_lines=["홈팀 OPS 0.812", "원정팀 OPS 0.905"],
            caveat_lines=["라인업 미확정"],
            allowed_entity_names={"LG 트윈스"},
            allowed_numeric_tokens={"0.812", "0.905"},
            supported_fact_count=2,
            starters_confirmed=False,
            lineup_confirmed=False,
            series_context_confirmed=True,
            require_series_context=False,
        )

        assert (
            _should_short_circuit_to_deterministic_response(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                fact_sheet=fact_sheet,
                game_status_bucket="SCHEDULED",
                grounding_reasons=[],
            )
            is False
        )

    def test_scheduled_partial_manual_calls_llm_when_facts_sufficient_despite_missing_lineup(
        self,
    ):
        # 정책 변경: 선발·라인업이 모두 미확정이어도 팀 단위 실데이터가 충분하면
        # 경기 예측에서도 LLM 을 호출한다(결정론 boilerplate 로 고정되지 않도록).
        from app.core.coach_grounding import CoachFactSheet
        from app.routers.coach import (
            COACH_REQUEST_MODE_MANUAL,
            _should_short_circuit_to_deterministic_response,
        )

        fact_sheet = CoachFactSheet(
            fact_lines=["홈팀 OPS 0.740", "원정팀 OPS 0.905"],
            caveat_lines=["선발 정보 미확정", "라인업 미확정"],
            allowed_entity_names={"LG 트윈스", "롯데 자이언츠"},
            allowed_numeric_tokens={"0.740", "0.905"},
            supported_fact_count=2,
            starters_confirmed=False,
            lineup_confirmed=False,
            series_context_confirmed=True,
            require_series_context=False,
            reasons=["missing_starters", "missing_lineups", "missing_summary"],
            warnings=[],
        )

        assert (
            _should_short_circuit_to_deterministic_response(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                fact_sheet=fact_sheet,
                game_status_bucket="SCHEDULED",
                grounding_reasons=[
                    "missing_starters",
                    "missing_lineups",
                    "missing_summary",
                ],
            )
            is False
        )

    def test_scheduled_partial_manual_short_circuits_when_facts_insufficient(self):
        # 안전 경계 유지: 선발·라인업 미확정 + 실데이터 fact 가 빈약하면 결정론으로 단락.
        from app.core.coach_grounding import CoachFactSheet
        from app.routers.coach import (
            COACH_REQUEST_MODE_MANUAL,
            _should_short_circuit_to_deterministic_response,
        )

        fact_sheet = CoachFactSheet(
            fact_lines=["홈팀 OPS 0.740"],
            caveat_lines=["선발 정보 미확정", "라인업 미확정"],
            allowed_entity_names={"LG 트윈스", "롯데 자이언츠"},
            allowed_numeric_tokens={"0.740"},
            supported_fact_count=1,
            starters_confirmed=False,
            lineup_confirmed=False,
            series_context_confirmed=True,
            require_series_context=False,
            reasons=["missing_starters", "missing_lineups", "missing_summary"],
            warnings=[],
        )

        assert (
            _should_short_circuit_to_deterministic_response(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                fact_sheet=fact_sheet,
                game_status_bucket="SCHEDULED",
                grounding_reasons=[
                    "missing_starters",
                    "missing_lineups",
                    "missing_summary",
                ],
            )
            is True
        )

    def test_scheduled_partial_manual_keeps_llm_when_starter_is_confirmed(self):
        from app.core.coach_grounding import CoachFactSheet
        from app.routers.coach import (
            COACH_REQUEST_MODE_MANUAL,
            _should_short_circuit_to_deterministic_response,
        )

        fact_sheet = CoachFactSheet(
            fact_lines=["홈팀 OPS 0.740", "발표 선발 롯데 반즈 / LG 손주영"],
            caveat_lines=["라인업 미확정"],
            allowed_entity_names={"LG 트윈스", "롯데 자이언츠", "반즈", "손주영"},
            allowed_numeric_tokens={"0.740"},
            supported_fact_count=2,
            starters_confirmed=True,
            lineup_confirmed=False,
            series_context_confirmed=True,
            require_series_context=False,
            reasons=["missing_lineups", "missing_summary"],
            warnings=[],
        )

        assert (
            _should_short_circuit_to_deterministic_response(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                fact_sheet=fact_sheet,
                game_status_bucket="SCHEDULED",
                grounding_reasons=["missing_lineups", "missing_summary"],
            )
            is False
        )

    def test_live_manual_detail_with_supported_facts_does_not_short_circuit(self):
        from app.core.coach_grounding import CoachFactSheet
        from app.routers.coach import (
            COACH_REQUEST_MODE_MANUAL,
            _should_short_circuit_to_deterministic_response,
        )

        fact_sheet = CoachFactSheet(
            fact_lines=["홈팀 최근 10경기 7승 3패", "원정팀 최근 10경기 5승 5패"],
            caveat_lines=["경기 진행 중"],
            allowed_entity_names={"LG 트윈스", "KT 위즈"},
            allowed_numeric_tokens={"10", "7", "3", "5"},
            supported_fact_count=2,
            starters_confirmed=True,
            lineup_confirmed=True,
            series_context_confirmed=True,
            require_series_context=False,
            reasons=[],
            warnings=[],
        )

        assert (
            _should_short_circuit_to_deterministic_response(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                fact_sheet=fact_sheet,
                game_status_bucket="LIVE",
                grounding_reasons=[],
                resolved_focus=["recent_form", "matchup"],
            )
            is False
        )

    def test_manual_recent_form_fast_path_requires_flag(self, monkeypatch):
        from app.core.coach_grounding import CoachFactSheet
        from app.routers import coach as coach_module

        fact_sheet = CoachFactSheet(
            fact_lines=["홈팀 최근 10경기 7승 3패", "원정팀 최근 10경기 5승 5패"],
            caveat_lines=[],
            allowed_entity_names={"LG 트윈스"},
            allowed_numeric_tokens={"7", "3", "5"},
            supported_fact_count=2,
            starters_confirmed=True,
            lineup_confirmed=True,
            series_context_confirmed=True,
            require_series_context=False,
        )

        monkeypatch.delenv("COACH_FAST_PATH_MANUAL_RECENT_FORM", raising=False)
        assert coach_module._coach_fast_path_manual_recent_form_enabled() is False
        assert (
            coach_module._should_short_circuit_to_deterministic_response(
                request_mode=coach_module.COACH_REQUEST_MODE_MANUAL,
                fact_sheet=fact_sheet,
                game_status_bucket="LIVE",
                grounding_reasons=[],
                resolved_focus=["recent_form"],
            )
            is False
        )

    def test_manual_recent_form_fast_path_triggers_when_flag_on(self, monkeypatch):
        from app.core.coach_grounding import CoachFactSheet
        from app.routers import coach as coach_module

        fact_sheet = CoachFactSheet(
            fact_lines=["홈팀 최근 10경기 7승 3패", "원정팀 최근 10경기 5승 5패"],
            caveat_lines=[],
            allowed_entity_names={"LG 트윈스"},
            allowed_numeric_tokens={"7", "3", "5"},
            supported_fact_count=2,
            starters_confirmed=True,
            lineup_confirmed=True,
            series_context_confirmed=True,
            require_series_context=False,
        )

        monkeypatch.setenv("COACH_FAST_PATH_MANUAL_RECENT_FORM", "1")
        assert coach_module._coach_fast_path_manual_recent_form_enabled() is True
        assert (
            coach_module._should_short_circuit_to_deterministic_response(
                request_mode=coach_module.COACH_REQUEST_MODE_MANUAL,
                fact_sheet=fact_sheet,
                game_status_bucket="LIVE",
                grounding_reasons=[],
                resolved_focus=["recent_form"],
            )
            is True
        )
        # Mixed focus should NOT trigger the fast path.
        assert (
            coach_module._should_short_circuit_to_deterministic_response(
                request_mode=coach_module.COACH_REQUEST_MODE_MANUAL,
                fact_sheet=fact_sheet,
                game_status_bucket="LIVE",
                grounding_reasons=[],
                resolved_focus=["recent_form", "pitcher_matchup"],
            )
            is False
        )
        # Empty focus should NOT trigger the fast path either.
        assert (
            coach_module._should_short_circuit_to_deterministic_response(
                request_mode=coach_module.COACH_REQUEST_MODE_MANUAL,
                fact_sheet=fact_sheet,
                game_status_bucket="LIVE",
                grounding_reasons=[],
                resolved_focus=[],
            )
            is False
        )

    def test_completed_manual_short_circuits_even_with_supported_facts(self):
        from app.core.coach_grounding import CoachFactSheet
        from app.routers.coach import (
            COACH_REQUEST_MODE_MANUAL,
            _should_short_circuit_to_deterministic_response,
        )

        fact_sheet = CoachFactSheet(
            fact_lines=["최종 스코어 7-2", "발표 선발 KIA 양현종 / LG 송승기"],
            caveat_lines=["WPA 기반 승부처 데이터가 부족합니다."],
            allowed_entity_names={"LG 트윈스", "KIA 타이거즈", "양현종", "송승기"},
            allowed_numeric_tokens={"7", "2"},
            supported_fact_count=2,
            starters_confirmed=True,
            lineup_confirmed=True,
            series_context_confirmed=True,
            require_series_context=False,
            reasons=[],
            warnings=[],
        )

        assert (
            _should_short_circuit_to_deterministic_response(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                fact_sheet=fact_sheet,
                game_status_bucket="COMPLETED",
                grounding_reasons=[],
            )
            is True
        )

    def test_auto_brief_and_empty_manual_still_short_circuit(self):
        from app.core.coach_grounding import CoachFactSheet
        from app.routers.coach import (
            COACH_REQUEST_MODE_AUTO,
            COACH_REQUEST_MODE_MANUAL,
            _should_short_circuit_to_deterministic_response,
        )

        empty_fact_sheet = CoachFactSheet(
            fact_lines=[],
            caveat_lines=["경기 요약 근거 부족"],
            allowed_entity_names=set(),
            allowed_numeric_tokens=set(),
            supported_fact_count=0,
            starters_confirmed=False,
            lineup_confirmed=False,
            series_context_confirmed=False,
            require_series_context=False,
        )

        assert (
            _should_short_circuit_to_deterministic_response(
                request_mode=COACH_REQUEST_MODE_AUTO,
                fact_sheet=empty_fact_sheet,
                game_status_bucket="UNKNOWN",
                grounding_reasons=[],
            )
            is True
        )
        assert (
            _should_short_circuit_to_deterministic_response(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                fact_sheet=empty_fact_sheet,
                game_status_bucket="UNKNOWN",
                grounding_reasons=[],
            )
            is True
        )

    def test_auto_brief_stream_meta_contract_backfills_success_defaults(self):
        from app.routers.coach import _ensure_stream_meta_contract

        payload = _ensure_stream_meta_contract(
            {
                "request_mode": "auto_brief",
                "validation_status": "success",
                "cache_state": "COMPLETED",
            }
        )

        assert payload["generation_mode"] == "deterministic_preview"
        assert payload["analysis_type"] == "game_preview"
        assert payload["data_quality"] == "partial"
        assert payload["cache_state"] == "COMPLETED"
        assert payload["cached"] is False
        assert payload["in_progress"] is False
        assert payload["used_evidence"] == []
        assert payload["grounding_warnings"] == []
        assert payload["grounding_reasons"] == []
        assert payload["supported_fact_count"] == 0

    def test_auto_brief_stream_meta_contract_preserves_failed_locked_fallback(self):
        from app.routers.coach import _ensure_stream_meta_contract

        payload = _ensure_stream_meta_contract(
            {
                "request_mode": "auto_brief",
                "validation_status": "fallback",
                "cache_state": "FAILED_LOCKED",
                "data_quality": "insufficient",
            }
        )

        assert payload["generation_mode"] == "evidence_fallback"
        assert payload["data_quality"] == "insufficient"
        assert payload["cache_state"] == "FAILED_LOCKED"

    def test_extract_cached_payload_defaults_auto_brief_generation_mode(self):
        from app.routers.coach import (
            COACH_REQUEST_MODE_AUTO,
            _extract_cached_payload,
        )

        _, meta = _extract_cached_payload(
            {"response": {"headline": "cached"}},
            request_mode=COACH_REQUEST_MODE_AUTO,
        )

        assert meta["generation_mode"] == "deterministic_preview"
        assert meta["analysis_type"] == "game_preview"
        assert meta["data_quality"] == "partial"

    def test_auto_brief_llm_attempt_limit_is_single_pass(self):
        from app.routers.coach import (
            COACH_REQUEST_MODE_AUTO,
            COACH_REQUEST_MODE_MANUAL,
            _resolve_coach_llm_attempt_limit,
        )

        assert _resolve_coach_llm_attempt_limit(COACH_REQUEST_MODE_AUTO) == 1
        assert _resolve_coach_llm_attempt_limit(COACH_REQUEST_MODE_MANUAL) == 2
        assert (
            _resolve_coach_llm_attempt_limit(
                COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="SCHEDULED",
                grounding_reasons=["missing_starters", "missing_lineups"],
            )
            == 1
        )

    def test_scheduled_partial_manual_llm_limits_are_compact(self):
        from app.routers.coach import (
            COACH_REQUEST_MODE_MANUAL,
            _resolve_coach_empty_chunk_retry_limit,
            _resolve_coach_llm_idle_timeout_seconds,
            _resolve_coach_llm_max_tokens,
            _resolve_coach_llm_request_timeout_seconds,
            _resolve_coach_llm_total_timeout_seconds,
        )

        grounding_reasons = ["missing_starters", "missing_lineups", "missing_summary"]

        assert (
            _resolve_coach_llm_max_tokens(
                2000,
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="SCHEDULED",
                grounding_reasons=grounding_reasons,
            )
            == 1200
        )
        assert (
            _resolve_coach_llm_idle_timeout_seconds(
                60.0,
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="SCHEDULED",
                grounding_reasons=grounding_reasons,
            )
            == 35.0
        )
        assert (
            _resolve_coach_llm_request_timeout_seconds(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="SCHEDULED",
                grounding_reasons=grounding_reasons,
            )
            == 45.0
        )
        assert (
            _resolve_coach_llm_total_timeout_seconds(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="SCHEDULED",
                grounding_reasons=grounding_reasons,
            )
            == 45.0
        )
        assert (
            _resolve_coach_empty_chunk_retry_limit(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="SCHEDULED",
                grounding_reasons=grounding_reasons,
            )
            == 0
        )

    def test_scheduled_half_confirmed_manual_llm_limits_are_more_aggressive(self):
        from app.routers.coach import (
            COACH_REQUEST_MODE_MANUAL,
            _resolve_coach_empty_chunk_retry_limit,
            _resolve_coach_llm_first_chunk_timeout_seconds,
            _resolve_coach_llm_idle_timeout_seconds,
            _resolve_coach_llm_max_tokens,
            _resolve_coach_llm_request_timeout_seconds,
            _resolve_coach_llm_total_timeout_seconds,
        )

        grounding_reasons = ["missing_lineups", "missing_summary"]

        assert (
            _resolve_coach_llm_max_tokens(
                2000,
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="SCHEDULED",
                grounding_reasons=grounding_reasons,
            )
            == 900
        )
        assert (
            _resolve_coach_llm_idle_timeout_seconds(
                60.0,
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="SCHEDULED",
                grounding_reasons=grounding_reasons,
            )
            == 15.0
        )
        assert (
            _resolve_coach_llm_request_timeout_seconds(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="SCHEDULED",
                grounding_reasons=grounding_reasons,
            )
            == 18.0
        )
        assert (
            _resolve_coach_llm_total_timeout_seconds(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="SCHEDULED",
                grounding_reasons=grounding_reasons,
            )
            == 18.0
        )
        assert (
            _resolve_coach_llm_first_chunk_timeout_seconds(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="SCHEDULED",
                grounding_reasons=grounding_reasons,
            )
            == 8.0
        )
        assert (
            _resolve_coach_empty_chunk_retry_limit(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="SCHEDULED",
                grounding_reasons=grounding_reasons,
            )
            == 0
        )

    def test_completed_manual_llm_limits_are_compact(self):
        from app.routers.coach import (
            COACH_REQUEST_MODE_MANUAL,
            _resolve_coach_empty_chunk_retry_limit,
            _resolve_coach_llm_attempt_limit,
            _resolve_coach_llm_idle_timeout_seconds,
            _resolve_coach_llm_max_tokens,
            _resolve_coach_llm_request_timeout_seconds,
            _resolve_coach_llm_total_timeout_seconds,
        )

        assert (
            _resolve_coach_llm_attempt_limit(
                COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="COMPLETED",
            )
            == 1
        )
        assert (
            _resolve_coach_llm_max_tokens(
                2000,
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="COMPLETED",
                grounding_reasons=[],
            )
            == 1400
        )
        assert (
            _resolve_coach_llm_idle_timeout_seconds(
                60.0,
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="COMPLETED",
                grounding_reasons=[],
            )
            == 18.0
        )
        assert (
            _resolve_coach_llm_request_timeout_seconds(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="COMPLETED",
                grounding_reasons=[],
            )
            == 24.0
        )
        assert (
            _resolve_coach_llm_total_timeout_seconds(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="COMPLETED",
                grounding_reasons=[],
            )
            == 24.0
        )
        assert (
            _resolve_coach_empty_chunk_retry_limit(
                request_mode=COACH_REQUEST_MODE_MANUAL,
                game_status_bucket="COMPLETED",
                grounding_reasons=[],
            )
            == 0
        )

    def test_manual_detail_reuses_cached_evidence_fallback(self):
        from app.routers.coach import (
            COACH_REQUEST_MODE_AUTO,
            COACH_REQUEST_MODE_MANUAL,
            _should_regenerate_completed_cache,
        )

        fallback_payload = {
            "response": {"headline": "fallback"},
            "_meta": {"generation_mode": "evidence_fallback"},
        }
        llm_payload = {
            "response": {"headline": "llm"},
            "_meta": {"generation_mode": "llm_manual"},
        }

        assert (
            _should_regenerate_completed_cache(
                cached_data=fallback_payload,
                request_mode=COACH_REQUEST_MODE_MANUAL,
            )
            is False
        )
        assert (
            _should_regenerate_completed_cache(
                cached_data=llm_payload,
                request_mode=COACH_REQUEST_MODE_MANUAL,
            )
            is False
        )
        assert (
            _should_regenerate_completed_cache(
                cached_data=fallback_payload,
                request_mode=COACH_REQUEST_MODE_AUTO,
            )
            is False
        )

    def test_manual_detail_regenerates_cache_when_current_evidence_is_richer(self):
        from app.routers.coach import (
            COACH_REQUEST_MODE_MANUAL,
            _should_regenerate_completed_cache,
        )

        stale_payload = {
            "response": {
                "headline": "KIA 타이거즈 vs NC 다이노스, 부분 근거 분석",
                "detailed_markdown": "## 최근 전력\n- 데이터 부족",
                "coach_note": "선발과 라인업 확인 후 보강이 필요합니다.",
                "analysis": {
                    "summary": "데이터가 부족합니다.",
                    "verdict": "보수적으로 봐야 합니다.",
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
            "_meta": {
                "data_quality": "partial",
                "game_status_bucket": "UNKNOWN",
                "used_evidence": ["game", "kbo_seasons"],
                "grounding_reasons": [
                    "missing_starters",
                    "missing_lineups",
                    "missing_summary",
                    "missing_metadata",
                ],
            },
        }

        assert (
            _should_regenerate_completed_cache(
                cached_data=stale_payload,
                request_mode=COACH_REQUEST_MODE_MANUAL,
                expected_data_quality="grounded",
                expected_used_evidence=[
                    "game",
                    "kbo_seasons",
                    "game_metadata",
                    "game_lineups",
                    "game_summary",
                ],
                expected_game_status_bucket="SCHEDULED",
                current_root_causes=[],
            )
            is True
        )

    def test_manual_detail_keeps_cache_when_evidence_contract_matches(self):
        from app.routers.coach import (
            COACH_REQUEST_MODE_MANUAL,
            _should_regenerate_completed_cache,
        )

        cached_payload = {
            "response": {
                "headline": "KIA 타이거즈 vs NC 다이노스, 근거 기반 분석",
                "detailed_markdown": "## 최근 전력\n- KIA 4승 5패 / NC 4승 6패",
                "coach_note": "선발 기준으로 봅니다.",
                "analysis": {
                    "summary": "확인된 근거 기준 분석입니다.",
                    "verdict": "확인된 근거를 우선 봅니다.",
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
            "_meta": {
                "data_quality": "grounded",
                "game_status_bucket": "SCHEDULED",
                "used_evidence": [
                    "game",
                    "kbo_seasons",
                    "game_metadata",
                    "game_lineups",
                    "game_summary",
                ],
                "grounding_reasons": [],
            },
        }

        assert (
            _should_regenerate_completed_cache(
                cached_data=cached_payload,
                request_mode=COACH_REQUEST_MODE_MANUAL,
                expected_data_quality="grounded",
                expected_used_evidence=[
                    "game",
                    "kbo_seasons",
                    "game_metadata",
                    "game_lineups",
                    "game_summary",
                ],
                expected_game_status_bucket="SCHEDULED",
                current_root_causes=[],
            )
            is False
        )

    def test_determine_cache_gate_keeps_completed_manual_fallback_as_hit(self):
        from app.routers.coach import (
            COACH_REQUEST_MODE_MANUAL,
            _determine_cache_gate,
            _should_regenerate_completed_cache,
        )

        cached_data = {
            "response": {
                "headline": "LG 트윈스 승리",
                "detailed_markdown": "## 결과 진단\n- LG 트윈스 승리",
                "coach_note": "승부처 복기",
                "analysis": {
                    "summary": "LG 트윈스가 이겼습니다.",
                    "verdict": "LG 트윈스가 승부처를 가져갔습니다.",
                    "strengths": ["타격 생산성"],
                    "weaknesses": [],
                    "risks": [],
                    "why_it_matters": ["실제 경기 결과로 연결됐습니다."],
                    "swing_factors": ["결승타 장면"],
                    "watch_points": ["득점 직전 상황"],
                    "uncertainty": [],
                },
                "key_metrics": [{"label": "최종 스코어", "value": "LG 14 / 삼성 13"}],
            },
            "_meta": {"generation_mode": "evidence_fallback"},
        }

        gate = _determine_cache_gate(
            status="COMPLETED",
            has_cached_json=True,
            updated_at=None,
            completed_ttl_seconds=None,
        )

        assert gate == "HIT"
        assert (
            _should_regenerate_completed_cache(
                cached_data=cached_data,
                request_mode=COACH_REQUEST_MODE_MANUAL,
            )
            is False
        )

    def test_iterate_coach_llm_with_keepalive_emits_status_before_chunk(self):
        from app.routers.coach import _iterate_coach_llm_with_keepalive

        async def _fake_coach_llm(*_args, **_kwargs):
            await asyncio.sleep(0.15)
            yield "첫 응답"

        async def _run():
            events = []
            async for event in _iterate_coach_llm_with_keepalive(
                coach_llm=_fake_coach_llm,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=64,
                heartbeat_seconds=0.01,
                idle_timeout_seconds=0.4,
            ):
                events.append(event)
            return events

        events = asyncio.run(_run())

        assert any(event["type"] == "status" for event in events)
        assert events[-1] == {"type": "chunk", "chunk": "첫 응답"}

    def test_iterate_coach_llm_with_keepalive_times_out_when_model_stalls(self):
        from app.routers.coach import _iterate_coach_llm_with_keepalive

        async def _stalled_coach_llm(*_args, **_kwargs):
            await asyncio.sleep(0.2)
            if False:
                yield ""

        async def _run():
            async for _ in _iterate_coach_llm_with_keepalive(
                coach_llm=_stalled_coach_llm,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=64,
                heartbeat_seconds=0.01,
                idle_timeout_seconds=0.03,
            ):
                pass

        with pytest.raises(TimeoutError):
            asyncio.run(_run())

    def test_iterate_coach_llm_with_keepalive_forwards_llm_kwargs(self):
        from app.routers.coach import _iterate_coach_llm_with_keepalive

        captured = {}

        async def _fake_coach_llm(*_args, **kwargs):
            captured.update(kwargs)
            yield "응답"

        async def _run():
            events = []
            async for event in _iterate_coach_llm_with_keepalive(
                coach_llm=_fake_coach_llm,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=64,
                heartbeat_seconds=0.01,
                idle_timeout_seconds=0.4,
                coach_llm_kwargs={
                    "empty_chunk_retry_limit": 0,
                    "request_timeout_seconds": 45.0,
                },
            ):
                events.append(event)
            return events

        events = asyncio.run(_run())

        assert captured["empty_chunk_retry_limit"] == 0
        assert captured["request_timeout_seconds"] == 45.0
        assert events[-1] == {"type": "chunk", "chunk": "응답"}

    def test_iterate_coach_llm_with_keepalive_times_out_before_first_chunk_deadline(
        self,
    ):
        from app.routers.coach import _iterate_coach_llm_with_keepalive

        async def _slow_first_chunk_llm(*_args, **_kwargs):
            await asyncio.sleep(0.08)
            yield "늦은 응답"

        async def _run():
            async for _ in _iterate_coach_llm_with_keepalive(
                coach_llm=_slow_first_chunk_llm,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=64,
                heartbeat_seconds=0.01,
                idle_timeout_seconds=0.20,
                first_chunk_timeout_seconds=0.03,
            ):
                pass

        with pytest.raises(TimeoutError, match="first_chunk_timeout"):
            asyncio.run(_run())

    def test_coach_stream_preview_flag_defaults_to_disabled(self, monkeypatch):
        from app.routers import coach as coach_module

        monkeypatch.delenv("COACH_STREAM_PREVIEW_ENABLED", raising=False)
        assert coach_module._coach_stream_preview_enabled() is False

    @pytest.mark.parametrize(
        "env_value,expected",
        [
            ("1", True),
            ("true", True),
            ("TRUE", True),
            ("yes", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("", False),
            ("no", False),
        ],
    )
    def test_coach_stream_preview_flag_env_parsing(
        self, monkeypatch, env_value, expected
    ):
        from app.routers import coach as coach_module

        monkeypatch.setenv("COACH_STREAM_PREVIEW_ENABLED", env_value)
        assert coach_module._coach_stream_preview_enabled() is expected

    def test_coach_prompt_v2_split_is_byte_identical(self):
        from app.core.prompts import (
            COACH_PROMPT_V2,
            COACH_PROMPT_V2_DYNAMIC_TEMPLATE,
            COACH_PROMPT_V2_STATIC,
        )

        assert (
            COACH_PROMPT_V2_STATIC + COACH_PROMPT_V2_DYNAMIC_TEMPLATE == COACH_PROMPT_V2
        )
        assert "{focus_section_requirements}" not in COACH_PROMPT_V2_STATIC
        assert "{question}" not in COACH_PROMPT_V2_STATIC
        assert "{context}" not in COACH_PROMPT_V2_STATIC

    def test_build_coach_llm_messages_disabled_returns_plain_string(self, monkeypatch):
        from app.routers import coach as coach_module

        monkeypatch.setenv("COACH_PROMPT_CACHE_ENABLED", "0")
        messages = coach_module._build_coach_llm_messages(
            "STATIC_PREFIX_",
            "DYNAMIC_SUFFIX",
        )
        assert messages == [{"role": "user", "content": "STATIC_PREFIX_DYNAMIC_SUFFIX"}]

    def test_build_coach_llm_messages_enabled_applies_cache_control(self, monkeypatch):
        from app.routers import coach as coach_module

        monkeypatch.setenv("COACH_PROMPT_CACHE_ENABLED", "1")
        monkeypatch.setenv("COACH_PROMPT_CACHE_MIN_STATIC_CHARS", "10")
        static_text = "A" * 50
        dynamic_text = "DYNAMIC"
        messages = coach_module._build_coach_llm_messages(static_text, dynamic_text)
        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["text"] == static_text
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        assert content[1]["text"] == dynamic_text
        assert "cache_control" not in content[1]
        # Byte-identical when concatenated back.
        assert content[0]["text"] + content[1]["text"] == static_text + dynamic_text

    def test_coach_latency_tracker_records_phase_timings(self):
        from app.routers.coach import _CoachLatencyTracker

        tracker = _CoachLatencyTracker(request_started=0.0)
        tracker.mark_tool_fetch_start()
        tracker.mark_tool_fetch_complete()
        tracker.mark_llm_start()
        tracker.mark_first_preview()
        tracker.mark_first_preview()  # idempotent — should not overwrite
        first_preview_initial = tracker.first_preview_at
        tracker.mark_first_preview()
        assert tracker.first_preview_at == first_preview_initial
        tracker.mark_llm_complete()
        tracker.mark_grounding_start()
        tracker.mark_grounding_complete()

        summary = tracker.build_summary(
            cache_state="COMPLETED",
            request_mode="manual_detail",
            game_status_bucket="LIVE",
            generation_mode="llm_manual",
            fast_path=False,
            preview_enabled=True,
            cache_enabled=False,
        )

        assert summary["cache_state"] == "COMPLETED"
        assert summary["request_mode"] == "manual_detail"
        assert summary["fast_path"] is False
        assert summary["preview_enabled"] is True
        assert summary["cache_enabled"] is False
        assert summary["coach_total_seconds"] is not None
        assert summary["coach_ttfb_seconds"] is not None
        assert summary["coach_llm_seconds"] is not None
        assert summary["coach_grounding_seconds"] is not None
        assert summary["coach_tool_fetch_seconds"] is not None

    def test_coach_latency_tracker_summary_handles_missing_phases(self):
        from app.routers.coach import _CoachLatencyTracker

        tracker = _CoachLatencyTracker(request_started=0.0)
        summary = tracker.build_summary(
            cache_state="CACHED",
            request_mode="auto_brief",
            game_status_bucket="FINAL",
            generation_mode="deterministic_auto",
            fast_path=False,
            preview_enabled=False,
            cache_enabled=True,
        )
        assert summary["coach_ttfb_seconds"] is None
        assert summary["coach_llm_seconds"] is None
        assert summary["coach_grounding_seconds"] is None
        assert summary["coach_tool_fetch_seconds"] is None
        assert summary["cache_enabled"] is True

    def test_build_coach_llm_messages_enabled_skips_below_threshold(self, monkeypatch):
        from app.routers import coach as coach_module

        monkeypatch.setenv("COACH_PROMPT_CACHE_ENABLED", "1")
        monkeypatch.setenv("COACH_PROMPT_CACHE_MIN_STATIC_CHARS", "9999")
        messages = coach_module._build_coach_llm_messages("short_static", "dynamic")
        assert messages == [{"role": "user", "content": "short_staticdynamic"}]


# ============================================================
# TTL Cache Tests
# ============================================================
