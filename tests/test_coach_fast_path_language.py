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

class TestCoachFastPathLanguage:
    def test_placeholder_sanitizer_replaces_none_and_zero_record_literals(self):
        from app.routers.coach import _sanitize_response_placeholders

        response_data = {
            "headline": "한화 vs SSG, 불펜 비교",
            "coach_note": "None%라 판단이 어렵습니다.",
            "key_metrics": [
                {
                    "label": "최근 전력",
                    "value": "한화 0승 0패 / SSG 0승 0패",
                    "status": "warning",
                    "trend": "neutral",
                    "is_critical": False,
                },
                {
                    "label": "불펜 비중",
                    "value": "한화 None% / SSG None%",
                    "status": "warning",
                    "trend": "neutral",
                    "is_critical": False,
                },
            ],
            "analysis": {
                "summary": "불펜 비중 None%로 비교가 어렵습니다.",
                "verdict": "최근 흐름 0승 0패 기준 비교는 무의미합니다.",
                "strengths": [],
                "weaknesses": [],
                "risks": [],
                "why_it_matters": ["0승 0패 표기는 결측일 뿐입니다."],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "detailed_markdown": "None%와 0승 0패 대신 결측을 설명해야 합니다.",
        }

        sanitized = _sanitize_response_placeholders(
            response_data,
            used_evidence=[
                "game",
                "team_advanced_metrics",
                "opponent_team_advanced_metrics",
            ],
        )

        assert sanitized["coach_note"] == "데이터 부족해 판단이 어렵습니다."
        assert (
            sanitized["key_metrics"][0]["value"] == "한화 데이터 부족 / SSG 데이터 부족"
        )
        assert (
            sanitized["key_metrics"][1]["value"] == "한화 데이터 부족 / SSG 데이터 부족"
        )
        assert (
            sanitized["analysis"]["summary"]
            == "불펜 비중 데이터 부족으로 비교가 어렵습니다."
        )
        assert (
            sanitized["analysis"]["verdict"]
            == "최근 흐름 데이터 부족 기준 비교는 무의미합니다."
        )
        assert (
            sanitized["detailed_markdown"] == "결측 데이터 대신 결측을 설명해야 합니다."
        )

    def test_placeholder_sanitizer_normalizes_mixed_language_artifacts(self):
        from app.routers.coach import _sanitize_response_placeholders

        response_data = {
            "headline": "정규시즌开幕 전 불펜 비교",
            "coach_note": "WPA 변동也无法 확인하여 판단이 어렵습니다.",
            "key_metrics": [],
            "analysis": {
                "summary": "경기 중盤 운영 전략 비교가 제한됩니다.",
                "verdict": "정규시즌开幕 전이라 보수적으로 봐야 합니다.",
                "strengths": [],
                "weaknesses": [],
                "risks": [],
                "why_it_matters": ["WPA 변동也无法 확인하여 사후 분석도 제한됩니다."],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "detailed_markdown": "정규시즌开幕 전 데이터만 있습니다.",
        }

        sanitized = _sanitize_response_placeholders(
            response_data,
            used_evidence=["game", "team_advanced_metrics"],
        )

        assert sanitized["headline"] == "정규시즌 개막 전 불펜 비교"
        assert sanitized["coach_note"] == "WPA 변동도 확인할 수 없어 판단이 어렵습니다."
        assert (
            sanitized["analysis"]["summary"] == "경기 중반 운영 전략 비교가 제한됩니다."
        )
        assert (
            sanitized["analysis"]["verdict"]
            == "정규시즌 개막 전이라 보수적으로 봐야 합니다."
        )
        assert (
            sanitized["analysis"]["why_it_matters"][0]
            == "WPA 변동도 확인할 수 없어 사후 분석도 제한됩니다."
        )
        assert sanitized["detailed_markdown"] == "정규시즌 개막 전 데이터만 있습니다."

    def test_placeholder_sanitizer_rewrites_recent_form_claims_without_recent_evidence(
        self,
    ):
        from app.routers.coach import _sanitize_response_placeholders

        response_data = {
            "headline": "불펜 비교",
            "coach_note": "최근 경기에서 승리를 거두지 못해도 불펜 변수는 남습니다.",
            "key_metrics": [],
            "analysis": {
                "summary": "SSG와 한화 모두 최근 경기에서 승리를 거두지 못하고 있으며, 불펜 비중 정보가 없어 판단이 어렵습니다.",
                "verdict": "최근 흐름이 하락세라 경기 후반 불펜 운영이 중요합니다.",
                "strengths": [],
                "weaknesses": [],
                "risks": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "detailed_markdown": "최근 경기에서 연패 흐름이라도 불펜 대응은 봐야 합니다.",
        }

        sanitized = _sanitize_response_placeholders(
            response_data,
            used_evidence=[
                "game",
                "team_advanced_metrics",
                "opponent_team_advanced_metrics",
            ],
        )

        assert (
            sanitized["coach_note"]
            == "최근 흐름 근거가 부족하지만 불펜 변수는 남습니다."
        )
        assert (
            sanitized["analysis"]["summary"]
            == "SSG와 한화 모두 최근 흐름 근거가 부족하며 불펜 비중 정보가 없어 판단이 어렵습니다."
        )
        assert (
            sanitized["analysis"]["verdict"]
            == "최근 흐름 근거가 부족하며 경기 후반 불펜 운영이 중요합니다."
        )
        assert (
            sanitized["detailed_markdown"]
            == "최근 흐름 근거가 부족하지만 불펜 대응은 봐야 합니다."
        )

    def test_soften_scheduled_partial_tone_rewrites_overconfident_phrasing(self):
        from app.routers.coach import _soften_scheduled_partial_tone

        response_data = {
            "headline": "SSG vs 한화, 경기 전 유리세 분석",
            "coach_note": "SSG가 경기 전 유리세를 확보했습니다.",
            "key_metrics": [
                {"label": "최근 흐름", "value": "SSG 유리세 확보", "status": "good"}
            ],
            "analysis": {
                "summary": "SSG가 압도적 우위를 점하며 경기 전 유리세를 확보했습니다.",
                "verdict": "SSG가 압도적 우위는 명확하나 승부를 가져갈 가능성이 높습니다.",
                "strengths": ["SSG가 압도적인 우위를 점합니다."],
                "weaknesses": [],
                "risks": [{"description": "경기 전 유리세를 유지할지가 변수입니다."}],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "detailed_markdown": "경기 전 유리세를 확보했습니다.",
        }

        softened = _soften_scheduled_partial_tone(
            response_data,
            game_status_bucket="SCHEDULED",
            grounding_reasons=["missing_starters", "missing_lineups"],
        )

        assert "압도적인 우위" not in softened["analysis"]["summary"]
        assert "압도적 우위" not in softened["analysis"]["summary"]
        assert "유리세 확보" not in softened["analysis"]["summary"]
        assert "우세 흐름" in softened["analysis"]["summary"]
        assert softened["coach_note"] == "SSG가 경기 전 우세 흐름을 보입니다."
        assert softened["key_metrics"][0]["value"] == "SSG 우세 흐름"
        assert "우세 흐름은 확인되나" in softened["analysis"]["verdict"]

    def test_soften_scheduled_partial_tone_skips_grounded_completed_games(self):
        from app.routers.coach import _soften_scheduled_partial_tone

        response_data = {
            "headline": "완료 경기 요약",
            "coach_note": "유리세를 확보했습니다.",
            "analysis": {
                "summary": "압도적인 우위를 점했습니다.",
                "verdict": "압도적인 우위입니다.",
                "strengths": [],
                "weaknesses": [],
                "risks": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "key_metrics": [],
            "detailed_markdown": "압도적인 우위",
        }

        softened = _soften_scheduled_partial_tone(
            response_data,
            game_status_bucket="COMPLETED",
            grounding_reasons=["missing_starters"],
        )

        assert softened == response_data

    def test_normalize_response_team_display_expands_team_codes(self):
        from app.routers.coach import (
            GameEvidence,
            _normalize_response_team_display,
        )

        evidence = GameEvidence(
            season_year=2026,
            home_team_code="HH",
            away_team_code="SSG",
            home_team_name="한화 이글스",
            away_team_name="SSG 랜더스",
        )
        response_data = {
            "headline": "SSG vs HH, 경기 프리뷰",
            "coach_note": "HH가 반격할 수 있습니다.",
            "key_metrics": [
                {"label": "최근 흐름", "value": "SSG 8승 2패 / HH 6승 4패"}
            ],
            "analysis": {
                "summary": "SSG가 HH보다 최근 흐름이 좋습니다.",
                "verdict": "SSG vs HH 구도입니다.",
                "strengths": ["SSG의 최근 흐름 우세"],
                "weaknesses": ["HH 불펜 변수"],
                "risks": [{"description": "HH의 불펜 부담"}],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "detailed_markdown": "## 최근 전력\nSSG 8승 2패 / HH 6승 4패",
        }

        normalized = _normalize_response_team_display(response_data, evidence=evidence)

        assert normalized["headline"] == "SSG 랜더스 vs 한화 이글스, 경기 프리뷰"
        assert normalized["coach_note"] == "한화 이글스가 반격할 수 있습니다."
        assert (
            normalized["key_metrics"][0]["value"]
            == "SSG 랜더스 8승 2패 / 한화 이글스 6승 4패"
        )
        assert (
            normalized["analysis"]["summary"]
            == "SSG 랜더스가 한화 이글스보다 최근 흐름이 좋습니다."
        )

    def test_normalize_response_markdown_layout_inserts_section_breaks(self):
        from app.routers.coach import _normalize_response_markdown_layout

        response_data = {
            "detailed_markdown": (
                "## 최근 전력\nSSG 8승 2패 / 한화 이글스 6승 4패 "
                "## 불펜 상태\n불펜 비중 데이터 부족 "
                "## 다시 볼 장면\n불펜 투입 시점 관전"
            )
        }

        normalized = _normalize_response_markdown_layout(response_data)

        assert (
            normalized["detailed_markdown"]
            == "## 최근 전력\nSSG 8승 2패 / 한화 이글스 6승 4패\n\n## 불펜 상태\n불펜 비중 데이터 부족\n\n## 다시 볼 장면\n불펜 투입 시점 관전"
        )

    def test_normalize_response_markdown_layout_normalizes_inline_focus_headers(self):
        from app.routers.coach import _normalize_response_markdown_layout

        response_data = {
            "detailed_markdown": (
                "## 최근 전력: SSG 랜더스 +11 / 한화 이글스 +8\n"
                "## 불펜 상태: 데이터 부족\n"
                "## 체크 포인트\n-\n"
                "불펜 교체 시점 확인"
            )
        }

        normalized = _normalize_response_markdown_layout(response_data)

        assert (
            normalized["detailed_markdown"]
            == "## 최근 전력\n- SSG 랜더스 +11 / 한화 이글스 +8\n\n## 불펜 상태\n- 데이터 부족\n\n## 체크 포인트\n불펜 교체 시점 확인"
        )

    def test_normalize_response_markdown_layout_repairs_broken_bold_focus_headers(
        self,
    ):
        from app.routers.coach import (
            _find_missing_focus_sections,
            _normalize_response_markdown_layout,
        )

        response_data = {
            "detailed_markdown": (
                "## 체크 포인트\n"
                "- **\n\n"
                "## 최근 전력**: SSG 랜더스 8승 2패 / 한화 이글스 4승 6패\n"
                "- **\n\n"
                "## 불펜 상태**: 데이터 부족으로 후반 운용은 보수적으로 봐야 합니다.\n"
                "- **\n\n"
                "## 선발 투수**: 최민준 vs 왕옌청"
            )
        }

        normalized = _normalize_response_markdown_layout(response_data)

        assert "## 최근 전력**" not in normalized["detailed_markdown"]
        assert "- **" not in normalized["detailed_markdown"]
        assert (
            "## 최근 전력\n- SSG 랜더스 8승 2패 / 한화 이글스 4승 6패"
            in normalized["detailed_markdown"]
        )
        assert (
            "## 불펜 상태\n- 데이터 부족으로 후반 운용은 보수적으로 봐야 합니다."
            in normalized["detailed_markdown"]
        )
        assert "## 선발 투수\n- 최민준 vs 왕옌청" in normalized["detailed_markdown"]
        assert (
            _find_missing_focus_sections(
                normalized, ["recent_form", "bullpen", "starter"]
            )
            == []
        )

    def test_normalize_response_team_display_expands_korean_short_aliases(self):
        from app.routers.coach import _normalize_response_team_display

        evidence = _build_game_evidence(
            home_team_code="HH",
            away_team_code="SSG",
            home_team_name="한화 이글스",
            away_team_name="SSG 랜더스",
        )
        response_data = {
            "headline": "한화 vs SSG 랜더스, 경기 프리뷰",
            "coach_note": "한화가 반격할 수 있습니다.",
            "key_metrics": [
                {"label": "최근 흐름", "value": "SSG 랜더스 8승 2패 / 한화 6승 4패"}
            ],
            "analysis": {
                "summary": "SSG 랜더스가 한화보다 최근 흐름이 좋습니다.",
                "verdict": "한화는 후반 운영이 변수입니다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
                "risks": [],
            },
            "detailed_markdown": "## 최근 전력\n한화 vs SSG 랜더스",
        }

        normalized = _normalize_response_team_display(response_data, evidence=evidence)

        assert normalized["headline"] == "한화 이글스 vs SSG 랜더스, 경기 프리뷰"
        assert normalized["coach_note"] == "한화 이글스가 반격할 수 있습니다."
        assert (
            normalized["key_metrics"][0]["value"]
            == "SSG 랜더스 8승 2패 / 한화 이글스 6승 4패"
        )
        assert (
            normalized["analysis"]["summary"]
            == "SSG 랜더스가 한화 이글스보다 최근 흐름이 좋습니다."
        )

    def test_normalize_response_team_display_unwraps_bracketed_team_names(self):
        from app.routers.coach import _normalize_response_team_display

        evidence = _build_game_evidence(
            home_team_code="HH",
            away_team_code="SSG",
            home_team_name="한화 이글스",
            away_team_name="SSG 랜더스",
        )
        response_data = {
            "headline": "[한화 이글스] vs [SSG], 경기 프리뷰",
            "coach_note": "[HH]가 반격할 수 있습니다.",
            "key_metrics": [
                {"label": "최근 흐름", "value": "[SSG 랜더스] 8승 2패 / [한화] 6승 4패"}
            ],
            "analysis": {
                "summary": "[SSG]가 [한화 이글스]보다 최근 흐름이 좋습니다.",
                "verdict": "[SSG 랜더스]가 초반 흐름을 잡으면 유리합니다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
                "risks": [{"description": "[HH] 불펜 변수"}],
            },
            "detailed_markdown": "## 최근 전력\n[SSG] 8승 2패 / [한화] 6승 4패",
        }

        normalized = _normalize_response_team_display(response_data, evidence=evidence)

        assert normalized["headline"] == "한화 이글스 vs SSG 랜더스, 경기 프리뷰"
        assert normalized["coach_note"] == "한화 이글스가 반격할 수 있습니다."
        assert (
            normalized["key_metrics"][0]["value"]
            == "SSG 랜더스 8승 2패 / 한화 이글스 6승 4패"
        )
        assert (
            normalized["analysis"]["summary"]
            == "SSG 랜더스가 한화 이글스보다 최근 흐름이 좋습니다."
        )
        assert (
            normalized["analysis"]["risks"][0]["description"] == "한화 이글스 불펜 변수"
        )

    def test_cleanup_response_language_quality_fixes_awkward_phrases(self):
        from app.routers.coach import _cleanup_response_language_quality

        response_data = {
            "headline": "한화 이글스 vs SSG 랜더스, 불펜 미정형 승부처",
            "coach_note": "불펜 핵심 선수 미공개로 중반 이후 승부처 예측 불가능",
            "key_metrics": [{"label": "불펜 부담", "value": "양팀 데이터 부족 비중"}],
            "analysis": {
                "summary": "SSG 랜더스의 최근 폼과 득실차 우세로 핵심 선수 가능성 높으나 불펜 부담 미확정",
                "verdict": "SSG 랜더스 선발 선발 발표 시 불펜 핵심 선수 예측 가능 여부",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [
                    "SSG 랜더스의 불펜 부재로 인해 클러치 상황 대비 변수가 됩니다."
                ],
                "watch_points": [],
                "uncertainty": [],
                "risks": [],
            },
            "detailed_markdown": "## 결과 진단\n- 불펜 핵심 선수 미공개로 중반 이후 승부처 예측 불가능",
        }

        cleaned = _cleanup_response_language_quality(response_data)

        assert cleaned["headline"] == "한화 이글스 vs SSG 랜더스, 불펜 변수 승부처"
        assert (
            cleaned["coach_note"]
            == "불펜 핵심 전력 정보 미공개로 중반 이후 승부처 예측이 어렵습니다"
        )
        assert (
            cleaned["analysis"]["summary"]
            == "SSG 랜더스의 최근 폼과 득실차 우세로 주도 가능성 높으나 불펜 부담 미확정"
        )
        assert (
            cleaned["analysis"]["verdict"]
            == "SSG 랜더스 선발 발표 후 불펜 핵심 전력 윤곽 확인 필요"
        )
        assert (
            cleaned["analysis"]["swing_factors"][0]
            == "SSG 랜더스의 불펜 정보 부족으로 인해 클러치 상황 대비 변수가 됩니다."
        )
        assert cleaned["key_metrics"][0]["value"] == "양 팀 모두 데이터 부족"
        assert (
            cleaned["detailed_markdown"]
            == "## 결과 진단\n- 불펜 핵심 전력 정보 미공개로 중반 이후 승부처 예측이 어렵습니다"
        )

    def test_cleanup_response_language_quality_fixes_particles_and_duplicates(self):
        from app.routers.coach import _cleanup_response_language_quality

        response_data = {
            "headline": "SSG 랜더스 최근 상승세, 한화 이글스 대비 우위 점검",
            "coach_note": "불펜 데이터 부족을 변수로 두고, 경기 중 양 팀 불펜 운용 패턴을 체크해야 합니다.",
            "key_metrics": [],
            "analysis": {
                "summary": "SSG 랜더스의 최근 상승세와 득실점 우위가 먼저 보이지만, 불펜 데이터 부족으로 변수가 남습니다.",
                "verdict": "SSG 랜더스이 최근 9경기 7승 2패, 득실 +11로 한화 이글스 대비 우위를 보이지만, 불펜 비중 데이터 부족으로 경기 후반 변수가 될 수 있습니다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
                "risks": [],
            },
            "detailed_markdown": (
                "## 다시 볼 장면\n"
                "- 양 팀 불펜 투입 시점과 경기 후반 접전 후반 상황 대응이 핵심 구간가 될 수 있습니다."
            ),
        }

        cleaned = _cleanup_response_language_quality(response_data)

        assert cleaned["analysis"]["verdict"] == (
            "SSG 랜더스가 최근 9경기 7승 2패, 득실 +11로 한화 이글스 대비 우위를 보이지만, "
            "불펜 운용 데이터 부족으로 경기 후반 변수가 될 수 있습니다."
        )
        assert (
            cleaned["detailed_markdown"]
            == "## 다시 볼 장면\n- 양 팀 불펜 투입 시점과 경기 후반 접전 상황 대응이 핵심 구간이 될 수 있습니다."
        )

    def test_cleanup_response_language_quality_collapses_duplicate_game_phrase(self):
        from app.routers.coach import _cleanup_response_language_quality

        response_data = {
            "headline": "예정 경기 분석",
            "coach_note": "경기 경기 후반 변수를 지켜봐야 합니다.",
            "key_metrics": [],
            "analysis": {
                "summary": "경기 경기 후반 운영이 변수입니다.",
                "verdict": "선발 미확정으로 경기 경기 후반 변수 존재",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
                "risks": [{"description": "선발 미확정으로 경기 경기 후반 변수 존재"}],
            },
            "detailed_markdown": "## 코치 판단\n경기 경기 후반 운영 변수를 확인해야 합니다.",
        }

        cleaned = _cleanup_response_language_quality(response_data)

        assert cleaned["coach_note"] == "경기 후반 변수를 지켜봐야 합니다."
        assert cleaned["analysis"]["summary"] == "경기 후반 운영이 변수입니다."
        assert cleaned["analysis"]["verdict"] == "선발 미확정으로 경기 후반 변수 존재"
        assert (
            cleaned["analysis"]["risks"][0]["description"]
            == "선발 미확정으로 경기 후반 변수 존재"
        )
        assert (
            cleaned["detailed_markdown"]
            == "## 코치 판단\n경기 후반 운영 변수를 확인해야 합니다."
        )

    def test_cleanup_response_language_quality_rewrites_bullpen_share_copy(self):
        from app.routers.coach import _cleanup_response_language_quality

        response_data = {
            "headline": "한화 이글스 vs SSG 랜더스, 불펜 비중 확인 필요",
            "coach_note": "불펜 비중 데이터 부족으로 인해 경기 후반 운영에 주의해야 합니다.",
            "key_metrics": [],
            "analysis": {
                "summary": "양 팀 모두 불펜 비중 데이터가 부족하여 접전 후반 상황 비교가 어렵습니다.",
                "verdict": "불펜 비중이 공개되지 않아 운영 판단이 제한됩니다.",
                "strengths": [],
                "weaknesses": ["양 팀 모두 불펜 비중 데이터 부족"],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
                "risks": [],
            },
            "detailed_markdown": "## 불펜 상태\n- 양 팀 모두 불펜 비중 데이터가 부족하여 접전 후반 상황에서의 팀 기량 비교가 어렵습니다.",
        }

        cleaned = _cleanup_response_language_quality(response_data)

        assert (
            cleaned["headline"] == "한화 이글스 vs SSG 랜더스, 불펜 운용 정보 확인 필요"
        )
        assert (
            cleaned["coach_note"]
            == "불펜 운용 데이터 부족으로 인해 경기 후반 운영에 주의해야 합니다."
        )
        assert (
            cleaned["analysis"]["summary"]
            == "양 팀 모두 불펜 운용 데이터가 부족하여 접전 후반 상황 비교가 어렵습니다."
        )
        assert (
            cleaned["analysis"]["verdict"]
            == "불펜 운용 정보가 공개되지 않아 운영 판단이 제한됩니다."
        )
        assert (
            cleaned["analysis"]["weaknesses"][0] == "양 팀 모두 불펜 운용 데이터 부족"
        )
        assert (
            cleaned["detailed_markdown"]
            == "## 불펜 상태\n- 양 팀 모두 불펜 운용 데이터가 부족하여 접전 후반 상황에서의 팀 기량 비교가 어렵습니다."
        )

    def test_cleanup_response_language_quality_rewrites_go_leverage(self):
        from app.routers.coach import _cleanup_response_language_quality

        response_data = {
            "headline": "SSG 랜더스 vs 한화 이글스, 최근 흐름과 불펜 가용성 비교",
            "coach_note": "경기 후반 불펜 운용과 고레버리지 상황 대응을 주시하세요.",
            "key_metrics": [
                {
                    "label": "불펜 상태",
                    "value": "고레버리지 상황에서의 과부하 여부 판단이 어렵습니다.",
                }
            ],
            "analysis": {
                "summary": "양 팀 모두 고레버리지 상황 대응 근거가 부족합니다.",
                "verdict": "고레버리지 상황에서 불펜 운용 차이를 확인해야 합니다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": ["고레버리지 상황 대응을 지켜봅니다."],
                "uncertainty": [],
                "risks": [],
            },
            "detailed_markdown": "## 다시 볼 장면\n- 선발 교체 뒤 고레버리지 상황 대응을 봅니다.",
        }

        cleaned = _cleanup_response_language_quality(response_data)

        assert (
            cleaned["coach_note"]
            == "경기 후반 불펜 운용과 접전 후반 상황 대응을 주시하세요."
        )
        assert (
            cleaned["key_metrics"][0]["value"]
            == "접전 후반 상황에서의 과부하 여부 판단이 어렵습니다."
        )
        assert (
            cleaned["analysis"]["summary"]
            == "양 팀 모두 접전 후반 상황 대응 근거가 부족합니다."
        )
        assert (
            cleaned["analysis"]["watch_points"][0]
            == "접전 후반 상황 대응을 지켜봅니다."
        )
        assert (
            cleaned["detailed_markdown"]
            == "## 다시 볼 장면\n- 선발 교체 뒤 접전 후반 상황 대응을 봅니다."
        )

    def test_cleanup_response_language_quality_fixes_key_section_object_particle(self):
        from app.routers.coach import _cleanup_response_language_quality

        response_data = {
            "headline": "한화 이글스 vs SSG 랜더스, 불펜전 승패 갈림길",
            "coach_note": "선발 발표 후 핵심 구간를 다시 보세요.",
            "key_metrics": [],
            "analysis": {
                "summary": "핵심 구간를 단정하기 어렵습니다.",
                "verdict": "불펜 정보가 부족하여 핵심 구간를 단정하기 어렵습니다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
                "risks": [],
            },
            "detailed_markdown": "## 다시 볼 장면\n선발 투수 발표 후 핵심 구간를 다시 분석해야 합니다.",
        }

        cleaned = _cleanup_response_language_quality(response_data)

        assert cleaned["coach_note"] == "선발 발표 후 핵심 구간을 다시 보세요."
        assert cleaned["analysis"]["summary"] == "핵심 구간을 단정하기 어렵습니다."
        assert (
            cleaned["analysis"]["verdict"]
            == "불펜 정보가 부족하여 핵심 구간을 단정하기 어렵습니다."
        )
        assert (
            cleaned["detailed_markdown"]
            == "## 다시 볼 장면\n선발 투수 발표 후 핵심 구간을 다시 분석해야 합니다."
        )

    def test_cleanup_response_language_quality_fixes_sanitized_entity_artifacts(self):
        from app.routers.coach import _cleanup_response_language_quality

        response_data = {
            "headline": "SSG 랜더스 vs 삼성 라이온즈, 상승세 맞불 대구개막",
            "coach_note": (
                "선발 이닝 소화 후 불펜으로 핵심 선수는 경기 운영에 영향을 준다."
            ),
            "key_metrics": [],
            "analysis": {
                "summary": (
                    "선발 이닝 소화 후 불펜으로 핵심 선수가 경기 운영에 영향을 준다. 핵심 구간는 경기 후반입니다."
                ),
                "verdict": "긍분위기에서 불펜으로 핵심 선수를 확인해야 합니다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
                "risks": [],
            },
            "detailed_markdown": (
                "## 불펜 상태\n"
                "- 선발 이닝 소화 후 불펜으로 핵심 선수는 경기 운영에 영향을 준다."
            ),
        }

        cleaned = _cleanup_response_language_quality(response_data)

        assert (
            cleaned["headline"] == "SSG 랜더스 vs 삼성 라이온즈, 상승세 맞불 대구 개막"
        )
        assert (
            cleaned["coach_note"]
            == "선발 이닝 소화 후 불펜 운용은 경기 운영에 영향을 준다."
        )
        assert (
            cleaned["analysis"]["summary"]
            == "선발 이닝 소화 후 불펜 운용이 경기 운영에 영향을 준다. 핵심 구간은 경기 후반입니다."
        )
        assert (
            cleaned["analysis"]["verdict"]
            == "긍정적인 분위기에서 불펜 운용을 확인해야 합니다."
        )
        assert (
            cleaned["detailed_markdown"]
            == "## 불펜 상태\n- 선발 이닝 소화 후 불펜 운용은 경기 운영에 영향을 준다."
        )

    def test_scheduled_output_guard_detects_review_and_starter_conflicts(self):
        from types import SimpleNamespace

        from app.routers.coach import _scheduled_output_guard_reasons

        payload = {
            "headline": "최대 10경기 성적 분석",
            "detailed_markdown": (
                "## 선발 투수\n"
                "- 선발 미확정으로 선발 대결 분석 불가능합니다.\n\n"
                "## 결과 진단\n"
                "- SSG 랜더스의 우위 확정."
            ),
        }
        evidence = SimpleNamespace(
            game_status_bucket="SCHEDULED",
            home_pitcher="후라도",
            away_pitcher="최민준",
        )

        reasons = _scheduled_output_guard_reasons(payload, evidence)

        assert any(reason.startswith("review_marker:") for reason in reasons)
        assert any(reason.startswith("starter_conflict:") for reason in reasons)

    def test_polish_scheduled_partial_response_also_cleans_grounded_scheduled_copy(
        self,
    ):
        from app.routers.coach import _polish_scheduled_partial_response

        response_data = {
            "headline": "SSG 랜더스의 오프ensive 우위",
            "coach_note": "",
            "key_metrics": [{"label": "최근 WPA 변동", "value": "WPA/PA 데이터 부족"}],
            "analysis": {
                "summary": "키움 히어로즈은 OPS 우위이지만 WPA/PA는 변수입니다. 두산 베어스을 압박하고 한화 이글스과 맞섭니다.",
                "verdict": "SSG 랜더스의 오프ensive 흐름이 변수입니다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": ["WPA 변동은 후반 운용 판단에 중요했습니다."],
                "swing_factors": ["불펜의 WPA/PA 데이터는 미확정입니다."],
                "watch_points": ["경기 후반 불펜 투입 시점"],
                "uncertainty": [],
                "risks": [
                    {
                        "area": "bullpen",
                        "level": 1,
                        "description": "선발 미확정으로 WPA/PA 데이터 부족",
                    }
                ],
            },
            "detailed_markdown": (
                "## 다시 볼 장면\n"
                "- 선발 정보가 완전히 확정되지 않아 SSG 랜더스 오프ensive 흐름과 WPA/PA를 확인합니다."
            ),
        }

        polished = _polish_scheduled_partial_response(
            response_data,
            game_status_bucket="SCHEDULED",
            grounding_reasons=[],
        )
        blob = str(polished)

        assert "WPA" not in blob
        assert "오프ensive" not in blob
        assert "히어로즈은" not in blob
        assert "베어스을" not in blob
        assert "이글스과" not in blob
        assert "선발 미확정" not in blob
        assert "선발 정보가 완전히 확정되지" not in blob
        assert "공식 선발 발표 전이라" in blob
        assert "두산 베어스를" in blob
        assert "한화 이글스와" in blob
        assert polished["coach_note"]
        assert polished["key_metrics"][0]["label"] == "최근 운영 지표"
        assert polished["detailed_markdown"].startswith("## 체크 포인트")

    def test_polish_scheduled_partial_response_rewrites_jargon(self):
        from app.routers.coach import _polish_scheduled_partial_response

        response_data = {
            "headline": "SSG 랜더스 vs 한화 이글스 전략 분석",
            "coach_note": "하이 레버리지 상황 대응을 지켜봐야 합니다.",
            "key_metrics": [{"label": "불펜 비중", "value": "비교 불가하다"}],
            "analysis": {
                "summary": "불펜 비중이 공개되지 않아 하이 레버리지 상황 처리 능력은 비교 불가하다.",
                "verdict": "불펜 비중이 공개되지 않아 하이 레버리지 상황 처리 능력은 비교 불가하다.",
                "strengths": [],
                "weaknesses": ["하이 레버리지 상황 판단을 할 수 없다."],
                "why_it_matters": [],
                "swing_factors": [
                    "불펜 비중이 공개되지 않아 운영 변수 확인이 어렵습니다."
                ],
                "watch_points": ["실제로 어떻게 활용되는지 확인한다."],
                "uncertainty": [
                    "선발이 미정이라 구체적인 경기 운영 변수는 남아 있습니다."
                ],
                "risks": [],
            },
            "detailed_markdown": "## 결론\n하이 레버리지 상황은 비교 불가하다.",
        }

        polished = _polish_scheduled_partial_response(
            response_data,
            game_status_bucket="SCHEDULED",
            grounding_reasons=[
                "missing_starters",
                "missing_lineups",
                "missing_summary",
            ],
        )

        assert polished["key_metrics"][0]["label"] == "불펜 운용"
        assert polished["key_metrics"][0]["value"] == "비교가 어렵다"
        assert (
            polished["analysis"]["summary"]
            == "불펜 운용 정보가 공개되지 않아 접전 후반 상황 처리 능력은 비교가 어렵다."
        )
        assert (
            polished["analysis"]["verdict"]
            == "핵심 변수는 불펜 운용 정보가 공개되지 않아 경기 후반 변수 확인이 어렵습니다."
        )
        assert polished["analysis"]["weaknesses"][0] == "접전 후반 상황 판단이 어렵다."
        assert (
            polished["detailed_markdown"]
            == "## 코치 판단\n접전 후반 상황은 비교가 어렵다."
        )
        assert polished["coach_note"].startswith("관전 포인트는")

    def test_polish_scheduled_partial_response_rewrites_live_bullpen_share_copy(self):
        from app.routers.coach import _polish_scheduled_partial_response

        response_data = {
            "headline": "SSG 랜더스 vs 한화 이글스, 최근 흐름·불펜 비교",
            "coach_note": "SSG 랜더스의 상승 흐름과 한화 이글스 페라자 폼을 동시에 주시하며, 후반 불펜 투입 타이밍이 승부를 좌우할 것이다.",
            "key_metrics": [
                {
                    "label": "최근 흐름",
                    "value": "SSG 랜더스 8승 2패 (+29) / 한화 이글스 6승 4패 (+2) (10경기)",
                },
                {"label": "불펜 비중", "value": "데이터 부족"},
            ],
            "analysis": {
                "summary": "SSG 랜더스가 최근 승·패와 득실 차에서 확연히 우위이며, 한화 이글스는 페라자 폼 상승으로 타격 잠재력이 높다.",
                "verdict": "SSG 랜더스는 10경기 득실 +29라는 압도적 흐름을 바탕으로 경기 후반 점수 확대 가능성이 크며, 한화 이글스는 페라자 상승 폼이 핵심 타격 포인트가 될 것이다.",
                "strengths": [],
                "weaknesses": [
                    "양 팀 모두 불펜 비중 데이터가 없으며, 선발·라인업 미확정"
                ],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": ["경기 후반 불펜 투입 시점과 양 팀 불펜 실제 활용도"],
                "uncertainty": [
                    "선발 투수와 라인업 발표가 없으며, 불펜 비중 데이터가 결여됨"
                ],
                "risks": [
                    {
                        "area": "overall",
                        "level": 1,
                        "description": "불펜 활용 현황과 선발 로테이션이 불투명해 경기 후반 변수가 큼",
                    }
                ],
            },
            "detailed_markdown": (
                "## 불펜 상태\n"
                "- 양 팀 모두 불펜 비중에 대한 공식 데이터가 없어 현재 불펜 활용 능력을 정확히 판단하기 어렵다."
            ),
        }

        polished = _polish_scheduled_partial_response(
            response_data,
            game_status_bucket="SCHEDULED",
            grounding_reasons=[
                "missing_starters",
                "missing_lineups",
                "missing_summary",
            ],
        )

        assert polished["key_metrics"][1]["label"] == "불펜 운용"
        assert polished["key_metrics"][1]["value"] == "데이터 부족"
        assert (
            polished["analysis"]["weaknesses"][0]
            == "양 팀 모두 불펜 운용 데이터가 부족하며, 선발·라인업 미확정"
        )
        assert (
            polished["analysis"]["uncertainty"][0]
            == "선발 투수와 라인업 발표가 없으며, 불펜 운용 데이터가 부족함"
        )
        assert polished["detailed_markdown"] == (
            "## 불펜 상태\n"
            "- 양 팀 모두 불펜 운용 관련 공식 데이터가 없어 현재 불펜 활용 능력을 정확히 판단하기 어렵다."
        )

    def test_polish_scheduled_partial_response_rewrites_bullpen_share_missing_phrase(
        self,
    ):
        from app.routers.coach import _polish_scheduled_partial_response

        response_data = {
            "headline": "SSG 랜더스 vs 한화 이글스, 불펜전 승패 갈림길",
            "coach_note": "후반 불펜 투입 타이밍이 승부를 좌우할 것이다.",
            "key_metrics": [],
            "analysis": {
                "summary": "최근 흐름상 SSG 랜더스가 우세하다.",
                "verdict": "경기 후반 접전 대응력이 변수다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
                "risks": [],
            },
            "detailed_markdown": (
                "## 불펜 상태\n"
                "- 두 팀 모두 불펜 비중이 데이터 부족으로, 불펜전의 중요성이 더욱 부각되고 있습니다."
            ),
        }

        polished = _polish_scheduled_partial_response(
            response_data,
            game_status_bucket="SCHEDULED",
            grounding_reasons=[
                "missing_starters",
                "missing_lineups",
                "missing_summary",
            ],
        )

        assert polished["detailed_markdown"] == (
            "## 불펜 상태\n"
            "- 두 팀 모두 불펜 운용 근거가 제한돼, 불펜전의 중요성이 더욱 부각되고 있습니다."
        )

    def test_polish_scheduled_partial_response_rewrites_bullpen_share_marked_missing_variants(
        self,
    ):
        from app.routers.coach import _polish_scheduled_partial_response

        response_data = {
            "headline": "예정 경기 분석",
            "coach_note": "불펜 비중 데이터 결측으로 경기 후반 예측이 어렵습니다.",
            "key_metrics": [],
            "analysis": {
                "summary": "불펜 비중 및 소모 흐름에 대한 데이터 부족으로 경기 후반 대응력 예측이 어렵습니다.",
                "verdict": "불펜 비중 데이터 결측이 커서 보수적으로 봐야 합니다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": ["불펜 비중 데이터 결측"],
                "risks": [],
            },
            "detailed_markdown": (
                "## 불펜 상태\n"
                "- 현재 불펜 비중 및 소모 흐름에 대한 데이터가 부족하여 경기 후반 대응력을 예측하기 어렵습니다."
            ),
        }

        polished = _polish_scheduled_partial_response(
            response_data,
            game_status_bucket="SCHEDULED",
            grounding_reasons=[
                "missing_starters",
                "missing_lineups",
                "missing_summary",
            ],
        )

        assert (
            polished["coach_note"]
            == "불펜 운용 데이터 결측으로 경기 후반 예측이 어렵습니다."
        )
        assert (
            polished["analysis"]["summary"]
            == "불펜 운용 및 소모 흐름에 대한 데이터 부족으로 경기 후반 대응력 예측이 어렵습니다."
        )
        assert (
            polished["analysis"]["verdict"]
            == "불펜 운용 데이터 결측이 커서 보수적으로 봐야 합니다."
        )
        assert polished["analysis"]["uncertainty"][0] == "불펜 운용 데이터 결측"
        assert polished["detailed_markdown"] == (
            "## 불펜 상태\n"
            "- 현재 불펜 운용 및 소모 흐름에 대한 데이터가 부족하여 경기 후반 대응력을 예측하기 어렵습니다."
        )

    def test_polish_scheduled_partial_response_relabels_bullpen_share_metric_when_value_is_unconfirmed(
        self,
    ):
        from app.routers.coach import _polish_scheduled_partial_response

        response_data = {
            "headline": "예정 경기 분석",
            "coach_note": "불펜 정보 확인이 필요합니다.",
            "key_metrics": [
                {"label": "불펜 비중", "value": "SSG 랜더스 불펜 데이터 미확정"}
            ],
            "analysis": {
                "summary": "불펜 데이터 미확정으로 후반 대응력 판단이 어렵습니다.",
                "verdict": "불펜 데이터가 더 공개돼야 합니다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
                "risks": [],
            },
            "detailed_markdown": "## 불펜 상태\n- SSG 랜더스 불펜 데이터 미확정",
        }

        polished = _polish_scheduled_partial_response(
            response_data,
            game_status_bucket="SCHEDULED",
            grounding_reasons=[
                "missing_starters",
                "missing_lineups",
                "missing_summary",
            ],
        )

        assert polished["key_metrics"][0]["label"] == "불펜 운용"
        assert polished["key_metrics"][0]["value"] == "SSG 랜더스 불펜 데이터 미확정"

    def test_polish_scheduled_partial_response_rewrites_review_style_scheduled_copy(
        self,
    ):
        from app.routers.coach import _polish_scheduled_partial_response

        response_data = {
            "headline": "한화 이글스 vs SSG 랜더스 불펜, 데이터 결여로 전략 불확실",
            "coach_note": "선발·라인업 발표 전 불펜 핵심 선수와 운영 지표 데이터 미공개로 전략 비교 미가능",
            "key_metrics": [
                {
                    "label": "불펜 핵심 선수",
                    "value": "한화 이글스 데이터 부족 / SSG 랜더스 데이터 부족",
                },
                {"label": "최근 WPA 변동", "value": "데이터 미확정"},
            ],
            "analysis": {
                "summary": "불펜 핵심 선수와 운영 지표 데이터 미공개로 양 팀의 후반전 대응력 비교 불가능",
                "verdict": "불펜 전략 비교 미가능 - 선발·라인업 미발표로 변수 과다",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [
                    "불펜 핵심 전력 정보 미공개로 선발·불펜 분배 전략 파악 불가"
                ],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [
                    "운영 지표/PA 데이터 미확정으로 핵심 구간 상황 분석 한계"
                ],
                "risks": [
                    {
                        "area": "overall",
                        "level": 1,
                        "description": "불펜 핵심 선수와 운영 지표 수치 미공개로 핵심 구간 분석 한계",
                    }
                ],
            },
            "detailed_markdown": (
                "## 결과 진단\n"
                "- 선발·라인업 미발표로 불펜 핵심 선수 미확정\n\n"
                "## 결과를 가른 이유\n"
                "- FACT SHEET에 불펜 핵심 선수(데이터 부족)과 운영 지표/PA 데이터 미공개로 양 팀 불펜 분석 불가능\n\n"
                "## 다시 볼 장면\n"
                "- 선발 발표 후 불펜 핵심 선수와 운영 지표 수치 공개 여부 확인 필요"
            ),
        }

        polished = _polish_scheduled_partial_response(
            response_data,
            game_status_bucket="SCHEDULED",
            grounding_reasons=[
                "missing_starters",
                "missing_lineups",
                "missing_summary",
            ],
        )

        assert (
            polished["headline"]
            == "한화 이글스 vs SSG 랜더스 불펜, 데이터 부족으로 전략 불확실"
        )
        assert (
            polished["coach_note"]
            == "선발·라인업 발표 전 불펜 핵심 자원과 운영 지표 데이터 미공개로 전략 비교가 어렵습니다"
        )
        assert polished["key_metrics"][0]["label"] == "불펜 핵심 자원"
        assert polished["key_metrics"][1]["label"] == "최근 운영 지표"
        assert (
            polished["analysis"]["summary"]
            == "불펜 핵심 자원과 운영 지표 데이터 미공개로 양 팀의 경기 후반 대응력 비교가 어렵습니다"
        )
        assert (
            polished["analysis"]["verdict"]
            == "불펜 전략 비교가 어렵습니다 - 선발·라인업 미발표로 변수 과다"
        )
        assert (
            polished["analysis"]["risks"][0]["description"]
            == "불펜 핵심 자원과 운영 지표 수치 미공개로 핵심 구간 분석 한계"
        )
        assert polished["detailed_markdown"].startswith(
            "## 코치 판단\n- 선발·라인업 미발표로 불펜 핵심 자원 미확정"
        )
        assert "## 왜 중요한가" in polished["detailed_markdown"]
        assert "## 체크 포인트" in polished["detailed_markdown"]

    def test_polish_scheduled_partial_response_rebuilds_duplicate_note(self):
        from app.routers.coach import _polish_scheduled_partial_response

        response_data = {
            "headline": "SSG 랜더스 vs 한화 이글스, 상승세 대비 불펜 불확실성",
            "coach_note": "SSG 랜더스의 상승세가 이어지고 있습니다.",
            "key_metrics": [],
            "analysis": {
                "summary": "SSG 랜더스의 상승세가 이어지고 있습니다.",
                "verdict": "SSG 랜더스의 상승세가 이어지고 있습니다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [
                    "불펜 비중이 공개되지 않아 경기 후반 변수 확인이 어렵습니다."
                ],
                "watch_points": ["경기 중 불펜 교체 시점을 확인해야 합니다."],
                "uncertainty": ["라인업 발표 전까지는 타순 변수도 남아 있습니다."],
                "risks": [],
            },
            "detailed_markdown": "## 결론\nSSG 랜더스의 상승세가 이어지고 있습니다.",
        }

        polished = _polish_scheduled_partial_response(
            response_data,
            game_status_bucket="SCHEDULED",
            grounding_reasons=["missing_starters", "missing_lineups"],
        )

        assert polished["analysis"]["verdict"].startswith("핵심 변수는")
        assert polished["analysis"]["verdict"] != polished["analysis"]["summary"]
        assert polished["coach_note"].startswith("관전 포인트는")
        assert polished["coach_note"] != polished["analysis"]["summary"]

    def test_soften_scheduled_partial_tone_rewrites_strong_scheduled_claims(self):
        from app.routers.coach import _soften_scheduled_partial_tone

        response_data = {
            "headline": "한화 이글스 vs SSG 랜더스, 불펜 승부 예측",
            "coach_note": "SSG 랜더스의 압도적인 최근 흐름을 고려하되 승기를 잡아야 합니다.",
            "key_metrics": [],
            "analysis": {
                "summary": "SSG 랜더스는 압도적인 승리 흐름을 보이고 있습니다.",
                "verdict": "SSG 랜더스가 유리세를 점하고 있으나 변수도 있습니다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [
                    "SSG 랜더스의 압도적인 득실 마진 우위는 실제 경기 결과에 영향을 줄 수 있습니다."
                ],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
                "risks": [],
            },
            "detailed_markdown": "## 최근 전력\nSSG 랜더스는 압도적인 승리 흐름을 보이고 있습니다.",
        }

        softened = _soften_scheduled_partial_tone(
            response_data,
            game_status_bucket="SCHEDULED",
            grounding_reasons=["missing_starters", "missing_lineups"],
        )

        assert "압도적인" not in softened["analysis"]["summary"]
        assert "유리세" not in softened["analysis"]["verdict"]
        assert "승기" not in softened["coach_note"]
        assert "뚜렷한 우세 흐름" in softened["analysis"]["summary"]
        assert "우세 흐름을 보이고 있으나" in softened["analysis"]["verdict"]
        assert (
            softened["analysis"]["why_it_matters"][0]
            == "SSG 랜더스의 뚜렷한 득실 마진 우위는 경기 흐름에 영향을 줄 수 있습니다."
        )
        assert (
            softened["coach_note"]
            == "SSG 랜더스의 뚜렷한 최근 우세 흐름을 고려하되 우세 흐름을 이어가야 합니다."
        )

    def test_soften_scheduled_partial_tone_rewrites_live_intense_margin_copy(self):
        from app.routers.coach import _soften_scheduled_partial_tone

        response_data = {
            "headline": "SSG 랜더스 vs 한화 이글스, 압도적 득실 마진의 SSG 랜더스와 페라자의 한화 이글스",
            "coach_note": "SSG 랜더스의 압도적인 득실 마진을 한화 이글스의 페라자가 얼마나 상쇄할 수 있느냐가 관건입니다.",
            "key_metrics": [],
            "analysis": {
                "summary": "SSG 랜더스의 압도적인 득실 마진과 한화 이글스의 핵심 타자 페라자의 고효율 타격이 맞붙는 구도입니다.",
                "verdict": "SSG 랜더스가 최근 10경기 8승 2패와 +29의 압도적 득실 마진을 통해 팀 전체의 화력과 운영 우위를 점하고 있습니다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
                "risks": [],
            },
            "detailed_markdown": "## 코치 판단\nSSG 랜더스의 압도적인 득실 마진이 눈에 띕니다.",
        }

        softened = _soften_scheduled_partial_tone(
            response_data,
            game_status_bucket="SCHEDULED",
            grounding_reasons=[
                "missing_starters",
                "missing_lineups",
                "missing_summary",
            ],
        )

        assert "압도적 득실 마진" not in softened["headline"]
        assert "압도적인 득실 마진" not in softened["analysis"]["summary"]
        assert "압도적 득실 마진" not in softened["analysis"]["verdict"]
        assert "뚜렷한 득실 마진" in softened["headline"]
        assert "뚜렷한 득실 마진" in softened["analysis"]["summary"]
        assert "뚜렷한 득실 마진" in softened["detailed_markdown"]

    def test_soften_scheduled_partial_tone_rewrites_review_style_language(self):
        from app.routers.coach import _soften_scheduled_partial_tone

        response_data = {
            "headline": "SSG 랜더스 vs 한화 이글스 전략 분석",
            "coach_note": "SSG 랜더스의 폼 진단과 불펜의 핵심 선수 활용이 승부처로 작용했지만, 관전자에게는 판단이 어렵습니다.",
            "key_metrics": [],
            "analysis": {
                "summary": "SSG 랜더스의 최근 흐름과 불펜의 핵심 선수 활용이 승부처로 작용했으나, 한화 이글스의 클러치 성과가 변수입니다.",
                "verdict": "SSG 랜더스가 초반 선취점 이후 불펜 투입 시점에서 우세를 보였으나, WPA 변동이 변수입니다.",
                "strengths": [],
                "weaknesses": ["불펜의 WPA 변동이 클러치에 영향"],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": ["관전자에게는 불펜의 핵심 선수 활용을 봐야 합니다."],
                "uncertainty": [],
                "risks": [],
            },
            "detailed_markdown": "초반 리드를 확보했으나 불펜의 WPA 변동이 중요했습니다.",
        }

        softened = _soften_scheduled_partial_tone(
            response_data,
            game_status_bucket="SCHEDULED",
            grounding_reasons=["missing_starters", "missing_lineups"],
        )

        assert "핵심 선수 활용" not in softened["analysis"]["summary"]
        assert "승부처로 작용" not in softened["analysis"]["summary"]
        assert "클러치" not in softened["analysis"]["summary"]
        assert "WPA" not in softened["analysis"]["verdict"]
        assert (
            softened["analysis"]["summary"]
            == "SSG 랜더스의 최근 흐름과 불펜의 핵심 전력 운용이 승부 변수로 볼 수 있으나, 한화 이글스의 승부처 대응이 변수입니다."
        )
        assert (
            softened["analysis"]["verdict"]
            == "SSG 랜더스가 초반 흐름에서 불펜 투입 시점에서 우세 흐름을 보이고 있으나, 운영 변동이 변수입니다."
        )
        assert (
            softened["analysis"]["weaknesses"][0] == "불펜의 운영 변동이 승부처에 영향"
        )
        assert (
            softened["analysis"]["watch_points"][0]
            == "관전 포인트로는 불펜의 핵심 전력 운용을 봐야 합니다."
        )
        assert (
            softened["detailed_markdown"]
            == "초반 흐름을 선점하고 있으나 불펜의 운영 변동이 중요한 변수입니다."
        )

    def test_soften_scheduled_partial_tone_rewrites_metric_labels(self):
        from app.routers.coach import _soften_scheduled_partial_tone

        response_data = {
            "headline": "예정 경기 분석",
            "coach_note": "WPA 변동과 클러치 대응을 봐야 합니다.",
            "key_metrics": [
                {"label": "최근 WPA 변동", "value": "고레버리지 대응 확인 필요"},
                {"label": "불펜 핵심 선수", "value": "핵심 선수 활용 여부 확인"},
            ],
            "analysis": {
                "summary": "WPA 변동과 클러치 대응이 변수입니다.",
                "verdict": "불펜 핵심 선수 활용이 중요합니다.",
                "strengths": [],
                "weaknesses": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
                "risks": [],
            },
            "detailed_markdown": "## 다시 볼 장면\n- WPA 변동과 클러치 대응을 확인합니다.",
        }

        softened = _soften_scheduled_partial_tone(
            response_data,
            game_status_bucket="SCHEDULED",
            grounding_reasons=["missing_starters", "missing_lineups"],
        )

        assert softened["key_metrics"][0]["label"] == "최근 운영 변동"
        assert softened["key_metrics"][0]["value"] == "고레버리지 대응 확인 필요"
        assert softened["key_metrics"][1]["label"] == "불펜 핵심 선수"
        assert softened["coach_note"] == "운영 변동과 승부처 대응을 봐야 합니다."

    def test_markdown_section_has_body_rejects_empty_focus_headers(self):
        from app.routers.coach import _markdown_section_has_body

        markdown = "## 최근 전력\n## 불펜 상태\n- 불펜 비중 데이터 부족"

        assert not _markdown_section_has_body(markdown, "## 최근 전력")
        assert _markdown_section_has_body(markdown, "## 불펜 상태")
