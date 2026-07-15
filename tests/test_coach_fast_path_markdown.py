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

class TestCoachFastPathMarkdown:
    def test_find_missing_focus_sections_treats_empty_headers_as_missing(self):
        from app.routers.coach import _find_missing_focus_sections

        response_data = {
            "detailed_markdown": "## 최근 전력\n## 불펜 상태\n- 불펜 비중 데이터 부족"
        }

        assert _find_missing_focus_sections(
            response_data, ["recent_form", "bullpen"]
        ) == ["recent_form"]

    def test_ensure_detailed_markdown_fills_missing_focus_sections_from_key_metrics(
        self,
    ):
        from app.routers.coach import _ensure_detailed_markdown

        response_payload = {
            "key_metrics": [
                {
                    "label": "최근 흐름",
                    "value": "SSG 랜더스 8승 2패 / 한화 이글스 6승 4패",
                },
                {"label": "불펜 비중", "value": "양 팀 모두 불펜 비중 데이터 부족"},
            ],
            "analysis": {
                "summary": "SSG 랜더스의 득실차가 앞서지만 불펜 변수는 남아 있습니다.",
                "verdict": "SSG 랜더스의 우세 흐름이 보이지만 격차는 제한적입니다.",
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "detailed_markdown": "## 결과 진단\n- SSG 랜더스의 득실차가 앞서지만 불펜 변수는 남아 있습니다.",
        }

        _ensure_detailed_markdown(response_payload, ["recent_form", "bullpen"])

        assert (
            response_payload["detailed_markdown"]
            == "## 최근 전력\n- SSG 랜더스 8승 2패 / 한화 이글스 6승 4패\n\n## 불펜 상태\n- 양 팀 모두 불펜 비중 데이터 부족\n\n## 결과 진단\n- SSG 랜더스의 득실차가 앞서지만 불펜 변수는 남아 있습니다."
        )

    def test_ensure_detailed_markdown_populates_existing_empty_focus_headers(self):
        from app.routers.coach import _ensure_detailed_markdown

        response_payload = {
            "key_metrics": [
                {
                    "label": "최근 흐름",
                    "value": "한화 이글스 6승 4패 / SSG 랜더스 8승 2패",
                },
                {"label": "불펜 비중", "value": "한화/SSG 랜더스 모두 데이터 부족"},
            ],
            "analysis": {
                "summary": "SSG 랜더스의 최근 상승세와 한화의 불펜 부담이 잠재적 승부처입니다.",
                "verdict": "SSG 랜더스의 최근 상승세와 한화의 불펜 부담이 잠재적 승부처입니다.",
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "detailed_markdown": "## 최근 전력\n## 불펜 상태\n## 결과 진단\n- SSG 랜더스의 최근 상승세와 한화의 불펜 부담이 잠재적 승부처입니다.",
        }

        _ensure_detailed_markdown(response_payload, ["recent_form", "bullpen"])

        assert (
            response_payload["detailed_markdown"]
            == "## 최근 전력\n- 한화 이글스 6승 4패 / SSG 랜더스 8승 2패\n\n## 불펜 상태\n- 한화/SSG 랜더스 모두 데이터 부족\n\n## 결과 진단\n- SSG 랜더스의 최근 상승세와 한화의 불펜 부담이 잠재적 승부처입니다."
        )

    def test_ensure_detailed_markdown_populates_empty_focus_headers_with_blank_lines(
        self,
    ):
        from app.routers.coach import _ensure_detailed_markdown

        response_payload = {
            "key_metrics": [
                {
                    "label": "최근 흐름",
                    "value": "한화 이글스 상승 83.1 / SSG 랜더스 상승 93.0",
                },
                {"label": "불펜 운용", "value": "양 팀 모두 불펜 운용 데이터 부족"},
            ],
            "analysis": {
                "summary": "확인된 팀 단위 지표는 박빙입니다.",
                "verdict": "첫 번째 불펜 선택이 가장 큰 변수입니다.",
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "detailed_markdown": (
                "## 최근 전력\n\n"
                "## 불펜 상태\n\n"
                "## 코치 판단\n"
                "- 첫 번째 불펜 선택이 가장 큰 변수입니다."
            ),
        }

        _ensure_detailed_markdown(response_payload, ["recent_form", "bullpen"])

        assert (
            response_payload["detailed_markdown"]
            == "## 최근 전력\n- 한화 이글스 상승 83.1 / SSG 랜더스 상승 93.0\n\n## 불펜 상태\n- 양 팀 모두 불펜 운용 데이터 부족\n\n## 코치 판단\n- 첫 번째 불펜 선택이 가장 큰 변수입니다."
        )

    def test_ensure_detailed_markdown_backfills_malformed_focus_headers(self):
        from app.routers.coach import _ensure_detailed_markdown

        response_payload = {
            "key_metrics": [
                {
                    "label": "최근 흐름",
                    "value": "SSG 랜더스 8승 2패 / 한화 이글스 4승 6패",
                },
                {"label": "불펜 운용", "value": "양 팀 모두 불펜 운용 데이터 부족"},
                {"label": "선발 투수", "value": "최민준 / 왕옌청"},
            ],
            "analysis": {
                "summary": "SSG 랜더스가 확인된 지표에서 앞섭니다.",
                "verdict": "SSG 랜더스가 앞서지만 후반 변수는 남아 있습니다.",
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "detailed_markdown": (
                "## 체크 포인트\n"
                "- **\n\n"
                "## 최근 전력**: SSG 랜더스 8승 2패 / 한화 이글스 4승 6패\n"
                "- **\n\n"
                "## 불펜 상태**: 데이터 부족으로 후반 운용은 보수적으로 봐야 합니다.\n"
                "- **\n\n"
                "## 선발 투수**: 최민준 vs 왕옌청"
            ),
        }

        _ensure_detailed_markdown(
            response_payload, ["recent_form", "bullpen", "starter"]
        )

        markdown = response_payload["detailed_markdown"]
        assert "## 최근 전력**" not in markdown
        assert "- **" not in markdown
        assert "## 최근 전력\n- SSG 랜더스 8승 2패 / 한화 이글스 4승 6패" in markdown
        assert (
            "## 불펜 상태\n- 데이터 부족으로 후반 운용은 보수적으로 봐야 합니다."
            in markdown
        )
        assert "## 선발 투수\n- 최민준 vs 왕옌청" in markdown

    def test_ensure_detailed_markdown_inserts_missing_focus_after_existing_focus_sections(
        self,
    ):
        from app.routers.coach import _ensure_detailed_markdown

        response_payload = {
            "key_metrics": [
                {"label": "불펜 비중", "value": "불펜 비중 정보 없음"},
            ],
            "analysis": {
                "summary": "SSG 랜더스의 최근 우세 흐름이 유지되고 있습니다.",
                "verdict": "SSG 랜더스가 우세 흐름을 보이지만 변수는 남아 있습니다.",
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "detailed_markdown": (
                "## 최근 전력\n"
                "- SSG 랜더스 8승 2패 / 한화 이글스 6승 4패\n\n"
                "## 결과 진단\n"
                "- SSG 랜더스의 최근 우세 흐름이 유지되고 있습니다."
            ),
        }

        _ensure_detailed_markdown(response_payload, ["recent_form", "bullpen"])

        assert (
            response_payload["detailed_markdown"]
            == "## 최근 전력\n- SSG 랜더스 8승 2패 / 한화 이글스 6승 4패\n\n## 불펜 상태\n- 불펜 비중 정보 없음\n\n## 결과 진단\n- SSG 랜더스의 최근 우세 흐름이 유지되고 있습니다."
        )

    def test_completed_deterministic_markdown_populates_focus_sections_immediately(
        self,
    ):
        from app.routers.coach import _build_deterministic_markdown

        evidence = _build_game_evidence(
            home_team_code="HH",
            away_team_code="NC",
            home_team_name="한화 이글스",
            away_team_name="NC 다이노스",
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            home_score=11,
            away_score=4,
            winning_team_code="HH",
            winning_team_name="한화 이글스",
            home_pitcher="류현진",
            away_pitcher="김태경",
            summary_items=["결승타 최재훈 (2회 2사 1,2루서 좌월 홈런)"],
        )
        tool_results = {
            "home": {
                "recent": {
                    "summary": {"wins": 4, "losses": 6, "draws": 0, "run_diff": -9}
                },
                "advanced": {
                    "metrics": {"batting": {"ops": 0.783}},
                    "fatigue_index": {"bullpen_share": 38.0},
                },
            },
            "away": {
                "recent": {
                    "summary": {"wins": 4, "losses": 5, "draws": 0, "run_diff": -1}
                },
                "advanced": {
                    "metrics": {"batting": {"ops": 0.799}},
                    "fatigue_index": {"bullpen_share": 31.0},
                },
            },
            "matchup": {"summary": {}},
            "clutch_moments": {"found": False, "moments": []},
        }

        markdown = _build_deterministic_markdown(
            evidence,
            tool_results,
            resolved_focus=["recent_form", "bullpen", "starter", "matchup", "batting"],
        )

        assert (
            "## 최근 전력\n- NC 다이노스 최근 4승 5패, 득실 -1 / 한화 이글스 최근 4승 6패, 득실 -9"
            in markdown
        )
        assert (
            "## 불펜 상태\n- 불펜 비중은 NC 다이노스 31.0% / 한화 이글스 38.0%로 차이가 확인됩니다."
            in markdown
        )
        assert "## 선발 투수\n- NC 다이노스 김태경 / 한화 이글스 류현진" in markdown
        assert "## 상대 전적\n-" in markdown
        assert "상대" in markdown or "맞대결" in markdown
        assert "## 타격 생산성\n- NC 다이노스 0.799 / 한화 이글스 0.783" in markdown

    def test_completed_focus_fallback_copy_varies_by_game_id(self):
        from app.routers.coach import (
            _completed_review_bullpen_focus_summary,
            _completed_review_matchup_focus_summary,
            _completed_review_recent_focus_summary,
            _completed_review_starter_focus_summary,
        )

        tool_results = {
            "home": {"recent": {"found": False}, "advanced": {}, "summary": {}},
            "away": {"recent": {"found": False}, "advanced": {}, "summary": {}},
            "matchup": {"summary": {}},
        }
        evidences = [
            _build_game_evidence(
                game_id=f"20260427LGKT{index}",
                game_status="COMPLETED",
                game_status_bucket="COMPLETED",
                home_pitcher=None,
                away_pitcher=None,
            )
            for index in range(12)
        ]

        bullpen_summaries = {
            _completed_review_bullpen_focus_summary(evidence, tool_results)
            for evidence in evidences
        }
        recent_summaries = {
            _completed_review_recent_focus_summary(evidence, tool_results)
            for evidence in evidences
        }
        starter_summaries = {
            _completed_review_starter_focus_summary(evidence) for evidence in evidences
        }
        matchup_summaries = {
            _completed_review_matchup_focus_summary(evidence, tool_results)
            for evidence in evidences
        }

        assert len(bullpen_summaries) > 1
        assert len(recent_summaries) > 1
        assert len(starter_summaries) > 1
        assert len(matchup_summaries) > 1
        assert all("불펜" in summary for summary in bullpen_summaries)
        assert all("최근" in summary for summary in recent_summaries)
        assert all("선발" in summary for summary in starter_summaries)
        assert all(
            "상대" in summary or "맞대결" in summary for summary in matchup_summaries
        )
        assert _completed_review_bullpen_focus_summary(
            evidences[0], tool_results
        ) == _completed_review_bullpen_focus_summary(evidences[0], tool_results)

    def test_ensure_detailed_markdown_uses_completed_focus_specific_summaries(self):
        from app.routers.coach import _ensure_detailed_markdown

        evidence = _build_game_evidence(
            home_team_code="HH",
            away_team_code="NC",
            home_team_name="한화 이글스",
            away_team_name="NC 다이노스",
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            home_score=11,
            away_score=4,
            winning_team_code="HH",
            winning_team_name="한화 이글스",
            home_pitcher="류현진",
            away_pitcher="김태경",
            summary_items=["결승타 최재훈 (2회 2사 1,2루서 좌월 홈런)"],
        )
        tool_results = {
            "home": {
                "recent": {
                    "summary": {"wins": 4, "losses": 6, "draws": 0, "run_diff": -9}
                },
                "advanced": {
                    "metrics": {"batting": {"ops": 0.783}},
                    "fatigue_index": {"bullpen_share": 38.0},
                },
            },
            "away": {
                "recent": {
                    "summary": {"wins": 4, "losses": 5, "draws": 0, "run_diff": -1}
                },
                "advanced": {
                    "metrics": {"batting": {"ops": 0.799}},
                    "fatigue_index": {"bullpen_share": 31.0},
                },
            },
            "matchup": {"summary": {}},
            "clutch_moments": {"found": False, "moments": []},
        }
        response_payload = {
            "key_metrics": [
                {
                    "label": "승부처 요약",
                    "value": "결승타 최재훈 (2회 2사 1,2루서 좌월 홈런)",
                },
            ],
            "analysis": {
                "summary": "NC 다이노스 4 / 한화 이글스 11 경기에서 한화 이글스가 이겼고, NC 다이노스는 득점 연결과 불펜 운용에서 차이를 남겼습니다.",
                "verdict": "한화 이글스 승리의 분기점은 경기 요약 기준 '결승타 최재훈 (2회 2사 1,2루서 좌월 홈런)' 장면이 실제 승부처로 기록됐습니다.",
                "why_it_matters": [
                    "한화 이글스는 '결승타 최재훈 (2회 2사 1,2루서 좌월 홈런)' 장면처럼 필요한 득점을 실제 결과로 연결했습니다."
                ],
                "swing_factors": [
                    "경기 요약 기준 '결승타 최재훈 (2회 2사 1,2루서 좌월 홈런)' 장면이 실제 승부처로 기록됐습니다."
                ],
                "watch_points": [
                    "'결승타 최재훈 (2회 2사 1,2루서 좌월 홈런)' 직전의 주자 상황과 투수 교체 선택이 어떻게 이어졌는지 다시 볼 필요가 있습니다."
                ],
                "uncertainty": [
                    "WPA 수치가 없어 변동 폭은 특정할 수 없지만, 경기 요약 기준 핵심 장면은 확인됩니다."
                ],
                "strengths": [],
                "weaknesses": [],
                "risks": [],
            },
            "detailed_markdown": (
                "## 최근 전력\n\n"
                "## 불펜 상태\n\n"
                "## 선발 투수\n\n"
                "## 상대 전적\n\n"
                "## 타격 생산성\n\n"
                "## 결과 진단\n"
                "- 한화 이글스 승리의 분기점은 경기 요약 기준 '결승타 최재훈 (2회 2사 1,2루서 좌월 홈런)' 장면이 실제 승부처로 기록됐습니다."
            ),
        }

        _ensure_detailed_markdown(
            response_payload,
            ["recent_form", "bullpen", "starter", "matchup", "batting"],
            grounding_warnings=[
                "요청한 focus 중 상대 전적 근거가 부족해 확인 가능한 항목만 분석합니다."
            ],
            evidence=evidence,
            tool_results=tool_results,
        )

        markdown = response_payload["detailed_markdown"]
        assert (
            "## 최근 전력\n- NC 다이노스 최근 4승 5패, 득실 -1 / 한화 이글스 최근 4승 6패, 득실 -9"
            in markdown
        )
        assert (
            "## 불펜 상태\n- 불펜 비중은 NC 다이노스 31.0% / 한화 이글스 38.0%로 차이가 확인됩니다."
            in markdown
        )
        assert "## 선발 투수\n- NC 다이노스 김태경 / 한화 이글스 류현진" in markdown
        assert "## 상대 전적\n-" in markdown
        assert any(
            phrase in markdown for phrase in ("상대 전적", "맞대결 표본", "직접 비교")
        )
        assert "## 타격 생산성\n- NC 다이노스 0.799 / 한화 이글스 0.783" in markdown
        assert "## 불펜 상태\n- NC 다이노스 4 / 한화 이글스 11 경기에서" not in markdown

    def test_ensure_detailed_markdown_rewrites_completed_focus_sections_to_thematic_bodies(
        self,
    ):
        from app.routers.coach import _ensure_detailed_markdown

        evidence = _build_game_evidence(
            home_team_code="HH",
            away_team_code="NC",
            home_team_name="한화 이글스",
            away_team_name="NC 다이노스",
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            home_score=11,
            away_score=4,
            winning_team_code="HH",
            winning_team_name="한화 이글스",
            home_pitcher="류현진",
            away_pitcher="김태경",
            summary_items=["결승타 최재훈 (2회 2사 1,2루서 좌월 홈런)"],
        )
        tool_results = {
            "home": {
                "recent": {
                    "summary": {"wins": 4, "losses": 6, "draws": 0, "run_diff": -9}
                },
                "advanced": {
                    "metrics": {"batting": {"ops": 0.783}},
                    "fatigue_index": {"bullpen_share": 38.0},
                },
            },
            "away": {
                "recent": {
                    "summary": {"wins": 4, "losses": 5, "draws": 1, "run_diff": -21}
                },
                "advanced": {
                    "metrics": {"batting": {"ops": 0.799}},
                    "fatigue_index": {"bullpen_share": 31.0},
                },
            },
            "matchup": {"summary": {}},
            "clutch_moments": {"found": False, "moments": []},
        }
        response_payload = {
            "key_metrics": [
                {
                    "label": "발표 선발",
                    "value": "NC 다이노스 김태경 / 한화 이글스 류현진",
                },
                {
                    "label": "팀 타격 생산성",
                    "value": "NC 다이노스 0.799 / 한화 이글스 0.783",
                },
            ],
            "analysis": {
                "summary": "NC 다이노스 4 / 한화 이글스 11 경기에서 한화 이글스가 이겼습니다.",
                "verdict": "한화 이글스가 결승타 구간을 실제 결과로 연결했습니다.",
                "why_it_matters": ["경기 후반 추가 득점이 승부를 갈랐습니다."],
                "swing_factors": ["2회 최재훈 타석이 핵심이었습니다."],
                "watch_points": ["추가 득점 직전 주자 운영을 다시 볼 필요가 있습니다."],
                "uncertainty": [],
                "strengths": [],
                "weaknesses": [],
                "risks": [],
            },
            "detailed_markdown": (
                "## 최근 전력\n"
                "- NC 다이노스 OPS 우세가 먼저 보였습니다.\n\n"
                "## 불펜 상태\n"
                "- NC 다이노스 4 / 한화 이글스 11 경기에서 한화 이글스가 이겼습니다.\n\n"
                "## 선발 투수\n"
                "- 경기 후반 추가 득점이 승부를 갈랐습니다.\n\n"
                "## 상대 전적\n"
                "- 타격 생산성 우위가 계속 유지됐습니다.\n\n"
                "## 타격 생산성\n"
                "- NC 다이노스 최근 4승 5패 1무, 득실 -21 / 한화 이글스 최근 4승 6패, 득실 -9\n\n"
                "## 결과 진단\n"
                "- 한화 이글스가 결승타 구간을 실제 결과로 연결했습니다."
            ),
        }

        _ensure_detailed_markdown(
            response_payload,
            ["recent_form", "bullpen", "starter", "matchup", "batting"],
            grounding_warnings=[
                "요청한 focus 중 상대 전적 근거가 부족해 확인 가능한 항목만 분석합니다."
            ],
            evidence=evidence,
            tool_results=tool_results,
        )

        markdown = response_payload["detailed_markdown"]
        assert (
            "## 최근 전력\n- NC 다이노스 최근 4승 5패 1무, 득실 -21 / 한화 이글스 최근 4승 6패, 득실 -9"
            in markdown
        )
        assert (
            "## 불펜 상태\n- 불펜 비중은 NC 다이노스 31.0% / 한화 이글스 38.0%로 차이가 확인됩니다."
            in markdown
        )
        assert "## 선발 투수\n- NC 다이노스 김태경 / 한화 이글스 류현진" in markdown
        assert "## 상대 전적\n-" in markdown
        assert any(
            phrase in markdown for phrase in ("상대 전적", "맞대결 표본", "직접 비교")
        )
        assert "## 타격 생산성\n- NC 다이노스 0.799 / 한화 이글스 0.783" in markdown
        assert "OPS 우세가 먼저 보였습니다" not in markdown
        assert "NC 다이노스 4 / 한화 이글스 11 경기에서" not in markdown

    def test_sanitize_response_unsupported_numeric_claims_repairs_payload(self):
        from app.core.coach_grounding import (
            CoachFactSheet,
            validate_response_against_fact_sheet,
        )
        from app.routers.coach import _sanitize_response_unsupported_numeric_claims

        response_payload = {
            "headline": "한화 이글스 vs SSG 랜더스, 데이터 기반 코치 브리핑",
            "sentiment": "neutral",
            "key_metrics": [
                {"label": "불펜 비중", "value": "한화 60% / SSG 랜더스 80%"}
            ],
            "analysis": {
                "summary": "불펜 가동률 80%는 부담입니다. 첫 번째 투수 교체가 변수입니다.",
                "verdict": "운영 리스크가 60% 수준입니다. 첫 번째 투수 교체가 변수입니다.",
                "strengths": [],
                "weaknesses": [],
                "risks": [],
                "why_it_matters": ["첫 번째 투수 교체가 변수입니다."],
                "swing_factors": [],
                "watch_points": ["첫 번째 투수 교체가 변수입니다."],
                "uncertainty": [],
            },
            "detailed_markdown": (
                "## 불펜 상태\n"
                "- 한화 60% / SSG 랜더스 80%\n"
                "## 체크 포인트\n"
                "- 첫 번째 투수 교체가 변수입니다."
            ),
            "coach_note": "불펜 가동률 80%보다 운영 선택을 봐야 합니다.",
        }
        fact_sheet = CoachFactSheet(
            fact_lines=[],
            caveat_lines=[],
            allowed_entity_names={"한화 이글스", "SSG 랜더스"},
            allowed_numeric_tokens=set(),
            supported_fact_count=2,
            starters_confirmed=False,
            lineup_confirmed=False,
            series_context_confirmed=True,
            require_series_context=False,
        )

        sanitized = _sanitize_response_unsupported_numeric_claims(
            response_payload,
            unsupported_tokens=["60", "60%", "80", "80%"],
        )
        validation = validate_response_against_fact_sheet(sanitized, fact_sheet)

        assert sanitized["key_metrics"] == []
        assert sanitized["analysis"]["summary"] == "첫 번째 투수 교체가 변수입니다."
        assert sanitized["analysis"]["verdict"] == "첫 번째 투수 교체가 변수입니다."
        assert sanitized["coach_note"] == "첫 번째 투수 교체가 변수입니다."
        assert "80%" not in sanitized["detailed_markdown"]
        assert validation.reasons == []

    def test_coach_prompt_example_avoids_literal_sample_numbers(self):
        from app.core.prompts import COACH_PROMPT_V2

        assert "52.0" not in COACH_PROMPT_V2
        assert "64.5" not in COACH_PROMPT_V2
        assert "18.4%p" not in COACH_PROMPT_V2
        assert "홍길동" not in COACH_PROMPT_V2
        assert "[폼 점수]" in COACH_PROMPT_V2
        assert "[WPA 변화]" in COACH_PROMPT_V2

    def test_should_regenerate_completed_cache_for_semantically_empty_manual_payload(
        self,
    ):
        from app.routers.coach import _should_regenerate_completed_cache

        cached_data = {
            "response": {
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
            "_meta": {"generation_mode": "llm_manual"},
        }

        assert (
            _should_regenerate_completed_cache(
                cached_data=cached_data,
                request_mode="manual_detail",
            )
            is True
        )

    def test_deterministic_response_omits_starter_and_lineup_when_missing(self):
        from app.routers.coach import GameEvidence, _build_deterministic_coach_response

        evidence = GameEvidence(
            season_year=2025,
            home_team_code="HH",
            away_team_code="LG",
            home_team_name="한화 이글스",
            away_team_name="LG 트윈스",
            game_id="202510200001",
            game_date="2025-10-20",
            game_status_bucket="SCHEDULED",
        )
        tool_results = {
            "home": {"recent": {"found": False}, "advanced": {}, "summary": {}},
            "away": {"recent": {"found": False}, "advanced": {}, "summary": {}},
            "matchup": {},
        }

        payload = _build_deterministic_coach_response(evidence, tool_results)

        assert "발표 선발" not in payload["detailed_markdown"]
        assert "발표 라인업" not in payload["detailed_markdown"]
        assert payload["analysis"]["verdict"]
        assert payload["analysis"]["why_it_matters"]

    def test_scheduled_missing_starters_before_announcement_is_preview_pending(
        self, monkeypatch
    ):
        from app.routers.coach import (
            COACH_STARTER_ANNOUNCEMENT_PENDING_CODE,
            _build_coach_fact_sheet,
            assess_game_evidence,
        )

        monkeypatch.setenv("COACH_AUDIT_NOW_KST", "2026-04-23T17:00:00+09:00")
        evidence = _build_game_evidence(
            game_date="2026-04-24",
            home_pitcher=None,
            away_pitcher=None,
        )

        assessment = assess_game_evidence(evidence)
        fact_sheet = _build_coach_fact_sheet(
            evidence,
            {"home": {}, "away": {}, "matchup": {}},
            set(),
            assessment,
        )

        assert COACH_STARTER_ANNOUNCEMENT_PENDING_CODE in assessment.root_causes
        assert "missing_starters" not in assessment.root_causes
        assert assessment.expected_data_quality == "partial"
        assert COACH_STARTER_ANNOUNCEMENT_PENDING_CODE in fact_sheet.reasons
        assert any("공식 선발 발표 전" in warning for warning in fact_sheet.warnings)

    def test_scheduled_missing_starters_after_announcement_requires_data(
        self, monkeypatch
    ):
        from app.routers.coach import (
            COACH_STARTER_ANNOUNCEMENT_PENDING_CODE,
            assess_game_evidence,
        )

        monkeypatch.setenv("COACH_AUDIT_NOW_KST", "2026-04-23T19:00:00+09:00")
        evidence = _build_game_evidence(
            game_date="2026-04-24",
            home_pitcher=None,
            away_pitcher=None,
        )

        assessment = assess_game_evidence(evidence)

        assert "missing_starters" in assessment.root_causes
        assert COACH_STARTER_ANNOUNCEMENT_PENDING_CODE not in assessment.root_causes
        assert assessment.expected_data_quality == "partial"

    def test_build_manual_data_request_returns_stage_mismatch_payload(self):
        from app.routers.coach import (
            AnalyzeRequest,
            _build_manual_data_request,
            assess_game_evidence,
        )

        evidence = _build_game_evidence(
            game_row_found=True,
            season_year=2019,
            season_id=265,
            game_date="2026-04-05",
            game_status="SCHEDULED",
            game_status_bucket="SCHEDULED",
            stage_label="KOREAN_SERIES",
        )
        payload = AnalyzeRequest(
            home_team_id="HH",
            away_team_id="OB",
            game_id="20260405HHOB0",
            league_context={"game_date": "2026-04-05"},
        )

        manual_request = asyncio.run(_build_manual_data_request(
            None,
            payload,
            evidence,
            assess_game_evidence(evidence),
        ))

        assert manual_request is not None
        assert manual_request["scope"] == "coach.analyze"
        assert manual_request["blocking"] is True
        assert manual_request["code"] == "MANUAL_BASEBALL_DATA_REQUIRED"
        assert {item["key"] for item in manual_request["missingItems"]} == {
            "season_league_context",
            "game_status",
            "final_score",
        }
        assert "경기 ID=20260405HHOB0" in manual_request["operatorMessage"]
        assert "날짜=2026-04-05" in manual_request["operatorMessage"]

    def test_build_manual_data_request_handles_invalid_season_year(self):
        from app.routers.coach import (
            AnalyzeRequest,
            _build_manual_data_request,
            assess_game_evidence,
        )

        evidence = _build_game_evidence(
            game_row_found=True,
            season_year="UNKNOWN",
            game_date="2099-04-05",
            game_status="SCHEDULED",
            game_status_bucket="SCHEDULED",
        )
        payload = AnalyzeRequest(
            home_team_id="LG",
            away_team_id="KT",
            game_id="20990405LGKT0",
            league_context={"game_date": "2099-04-05"},
        )

        manual_request = asyncio.run(_build_manual_data_request(
            None,
            payload,
            evidence,
            assess_game_evidence(evidence),
        ))

        assert manual_request is not None
        assert manual_request["code"] == "MANUAL_BASEBALL_DATA_REQUIRED"
        assert {item["key"] for item in manual_request["missingItems"]} == {
            "season_league_context"
        }

    def test_build_manual_data_request_accepts_numeric_string_season_year(self):
        from app.routers.coach import (
            AnalyzeRequest,
            _build_manual_data_request,
            assess_game_evidence,
        )

        evidence = _build_game_evidence(
            game_row_found=True,
            season_year="2099",
            game_date="2099-04-05",
            game_status="SCHEDULED",
            game_status_bucket="SCHEDULED",
        )
        payload = AnalyzeRequest(
            home_team_id="LG",
            away_team_id="KT",
            game_id="20990405LGKT0",
            league_context={"game_date": "2099-04-05"},
        )

        manual_request = asyncio.run(_build_manual_data_request(
            None,
            payload,
            evidence,
            assess_game_evidence(evidence),
        ))

        assert manual_request is None

    def test_build_manual_data_request_does_not_block_completed_game_with_scores(self):
        from app.routers.coach import (
            AnalyzeRequest,
            _build_manual_data_request,
            assess_game_evidence,
        )

        evidence = _build_game_evidence(
            game_row_found=True,
            season_year=2026,
            season_id=266,
            game_date="2026-03-22",
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            away_team_name="LG 트윈스",
            home_team_name="삼성 라이온즈",
            away_score=14,
            home_score=13,
            winning_team_code="LG",
            winning_team_name="LG 트윈스",
        )
        payload = AnalyzeRequest(
            home_team_id="SS",
            away_team_id="LG",
            game_id="20260322LGSS0",
            league_context={"game_date": "2026-03-22"},
        )

        manual_request = asyncio.run(_build_manual_data_request(
            None,
            payload,
            evidence,
            assess_game_evidence(evidence),
        ))

        assert manual_request is None

    def test_build_manual_data_request_allows_preview_without_final_score(self):
        from app.routers.coach import (
            AnalyzeRequest,
            COACH_ANALYSIS_TYPE_PREVIEW,
            _build_manual_data_request,
            assess_game_evidence,
        )

        evidence = _build_game_evidence(
            game_row_found=True,
            season_year=2026,
            season_id=266,
            game_date="2026-04-05",
            game_status="SCHEDULED",
            game_status_bucket="SCHEDULED",
            home_score=None,
            away_score=None,
        )
        payload = AnalyzeRequest(
            home_team_id="LG",
            away_team_id="KT",
            game_id="20260405LGKT0",
            league_context={"game_date": "2026-04-05"},
            request_mode="manual_detail",
            analysis_type=COACH_ANALYSIS_TYPE_PREVIEW,
        )
        assessment = assess_game_evidence(
            evidence,
            analysis_type=COACH_ANALYSIS_TYPE_PREVIEW,
        )

        manual_request = asyncio.run(_build_manual_data_request(
            None,
            payload,
            evidence,
            assessment,
            analysis_type=COACH_ANALYSIS_TYPE_PREVIEW,
        ))

        assert manual_request is None

    def test_build_fallback_evidence_preserves_scheduled_status_from_payload(self):
        from app.routers.coach import AnalyzeRequest, _build_fallback_evidence

        payload = AnalyzeRequest(
            home_team_id="LG",
            away_team_id="KT",
            game_id="20260405LGKT0",
            league_context={
                "game_date": "2026-04-05",
                "game_status": "SCHEDULED",
                "league_type_code": 0,
            },
            request_mode="auto_brief",
            analysis_type="game_preview",
        )

        evidence = _build_fallback_evidence(
            payload,
            2026,
            "LG",
            "KT",
            "LG 트윈스",
            "KT 위즈",
        )

        assert evidence.game_status == "SCHEDULED"
        assert evidence.game_status_bucket == "SCHEDULED"

    def test_build_fallback_evidence_ignores_unannounced_pitcher_labels(
        self, monkeypatch
    ):
        from app.routers.coach import (
            AnalyzeRequest,
            _build_fallback_evidence,
            assess_game_evidence,
        )

        monkeypatch.setenv("COACH_AUDIT_NOW_KST", "2026-04-04T19:00:00+09:00")
        payload = AnalyzeRequest(
            home_team_id="LG",
            away_team_id="KT",
            game_id="20260405LGKT0",
            league_context={
                "game_date": "2026-04-05",
                "game_status": "SCHEDULED",
                "league_type_code": 0,
                "home_pitcher": "발표 전",
                "away_pitcher": "미정",
            },
            request_mode="auto_brief",
            analysis_type="game_preview",
        )

        evidence = _build_fallback_evidence(
            payload,
            2026,
            "LG",
            "KT",
            "LG 트윈스",
            "KT 위즈",
        )
        assessment = assess_game_evidence(
            evidence,
            analysis_type="game_preview",
        )

        assert evidence.home_pitcher is None
        assert evidence.away_pitcher is None
        assert "missing_starters" in assessment.root_causes

    def test_build_manual_data_request_requires_final_score_for_review(self):
        from app.routers.coach import (
            AnalyzeRequest,
            BASEBALL_DATA_SYNC_REQUIRED_CODE,
            COACH_ANALYSIS_TYPE_REVIEW,
            MANUAL_BASEBALL_DATA_REQUIRED_CODE,
            _build_manual_data_request,
            assess_game_evidence,
        )

        evidence = _build_game_evidence(
            game_row_found=True,
            season_year=2026,
            season_id=266,
            game_date="2026-04-05",
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            home_score=None,
            away_score=None,
        )
        payload = AnalyzeRequest(
            home_team_id="LG",
            away_team_id="KT",
            game_id="20260405LGKT0",
            league_context={"game_date": "2026-04-05"},
            request_mode="manual_detail",
            analysis_type=COACH_ANALYSIS_TYPE_REVIEW,
        )
        assessment = assess_game_evidence(
            evidence,
            analysis_type=COACH_ANALYSIS_TYPE_REVIEW,
        )

        manual_request = asyncio.run(_build_manual_data_request(
            None,
            payload,
            evidence,
            assessment,
            analysis_type=COACH_ANALYSIS_TYPE_REVIEW,
        ))

        assert manual_request is not None
        assert manual_request["code"] == MANUAL_BASEBALL_DATA_REQUIRED_CODE
        assert manual_request["dataSyncRequired"] is True
        assert manual_request["dataSyncCode"] == BASEBALL_DATA_SYNC_REQUIRED_CODE
        assert manual_request["externalSource"] == "trusted_baseball_data_project"
        assert manual_request["dataSyncRequest"] == {
            "code": BASEBALL_DATA_SYNC_REQUIRED_CODE,
            "requestId": "coach:20260405LGKT0:game_review",
            "consumer": "ai_coach",
            "scope": "coach.analyze",
            "analysisType": COACH_ANALYSIS_TYPE_REVIEW,
            "targetSource": "trusted_baseball_data_project",
            "handoff": "external_trusted_baseball_data_sync",
            "blocking": True,
            "entity": {
                "gameId": "20260405LGKT0",
                "gameDate": "2026-04-05",
                "seasonYear": 2026,
                "homeTeamId": "LG",
                "awayTeamId": "KT",
                "stage": "REGULAR",
            },
            "missingItems": [
                {
                    "key": "game_status",
                    "label": "경기 상태",
                    "reason": "과거 경기의 상태가 종료 기준으로 확정되지 않았습니다.",
                    "expectedFormat": "SCHEDULED, COMPLETED, CANCELLED 등",
                    "requiredFields": ["game.game_status"],
                },
                {
                    "key": "final_score",
                    "label": "최종 점수",
                    "reason": "과거 경기의 최종 점수가 비어 있습니다.",
                    "expectedFormat": "home_score, away_score",
                    "requiredFields": ["game.home_score", "game.away_score"],
                },
            ],
        }
        assert {item["key"] for item in manual_request["missingItems"]} == {
            "game_status",
            "final_score",
        }

    def test_format_game_summary_item_normalizes_db_row(self):
        from app.routers.coach import _format_game_summary_item

        assert (
            _format_game_summary_item(
                "결승타",
                "박동원",
                "박동원(1회 2사 1,2루서 중전 안타)",
            )
            == "결승타 박동원 (1회 2사 1,2루서 중전 안타)"
        )
        assert (
            _format_game_summary_item(
                "홈런",
                "박건우",
                "박건우1호(8회1점 왕옌청)",
            )
            == "홈런 박건우 1호(8회1점 왕옌청)"
        )

    def test_postprocess_completed_payload_cleans_language_artifacts_and_fills_sections(
        self,
    ):
        from app.routers.coach import _postprocess_coach_response_payload

        evidence = _build_game_evidence(
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            away_team_name="NC 다이노스",
            home_team_name="한화 이글스",
            away_score=4,
            home_score=11,
            winning_team_code="HH",
            winning_team_name="한화 이글스",
            summary_items=["결승타 최재훈 (2회 2사 1,2루서 좌월 홈런)"],
        )
        response_payload = {
            "headline": "한화 이글스 11-4 승리, 박건우 홈런과 최재훈.walk-off 결정적",
            "sentiment": "positive",
            "key_metrics": [
                {
                    "label": "타격 생산성",
                    "value": "한화 이글스 0.817 OPS / NC 다이노스 0.836 OPS",
                    "status": "good",
                    "trend": "neutral",
                    "is_critical": True,
                }
            ],
            "analysis": {
                "summary": "한화 이글스의 고 OPS 타자와 late inning 점수가 승부처를 결정했습니다.",
                "verdict": "한화 이글스의 고 OPS 타자와 walk-off 상황이 승부처를 결정했습니다.",
                "strengths": [],
                "weaknesses": [],
                "risks": [],
                "why_it_matters": ["경기 후반 추가 득점이 승부를 갈랐습니다."],
                "swing_factors": [],
                "watch_points": ["2회 최재훈 타석과 8회 박건우 홈런 타이밍"],
                "uncertainty": [],
            },
            "detailed_markdown": (
                "## 결과 진단\n\n"
                "## 결과를 가른 이유\n"
                "- 경기 후반 추가 득점이 승부를 갈랐습니다.\n\n"
                "## 다시 볼 장면"
            ),
            "coach_note": "한화 이글스의 late inning 대응이 좋았습니다.",
        }

        processed = _postprocess_coach_response_payload(
            response_payload,
            evidence=evidence,
            used_evidence=["game", "game_summary"],
            grounding_reasons=[],
            tool_results=None,
            resolved_focus=["recent_form", "bullpen"],
        )
        serialized = json.dumps(processed, ensure_ascii=False)

        assert "walk-off" not in serialized
        assert "late inning" not in serialized
        assert "최재훈 결승타" in processed["headline"]
        assert "## 결과 진단\n-" in processed["detailed_markdown"]
        assert "## 다시 볼 장면\n-" in processed["detailed_markdown"]

    def test_postprocess_completed_payload_realigns_llm_focus_sections(self):
        from app.routers.coach import _postprocess_coach_response_payload

        evidence = _build_game_evidence(
            home_team_code="HH",
            away_team_code="NC",
            home_team_name="한화 이글스",
            away_team_name="NC 다이노스",
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            home_score=11,
            away_score=4,
            winning_team_code="HH",
            winning_team_name="한화 이글스",
            home_pitcher="류현진",
            away_pitcher="김태경",
            summary_items=["결승타 최재훈 (2회 2사 1,2루서 좌월 홈런)"],
        )
        tool_results = {
            "home": {
                "recent": {
                    "summary": {"wins": 4, "losses": 6, "draws": 0, "run_diff": -9}
                },
                "advanced": {
                    "metrics": {"batting": {"ops": 0.783}},
                    "fatigue_index": {"bullpen_share": 38.0},
                },
                "summary": {},
            },
            "away": {
                "recent": {
                    "summary": {"wins": 4, "losses": 5, "draws": 1, "run_diff": -21}
                },
                "advanced": {
                    "metrics": {"batting": {"ops": 0.799}},
                    "fatigue_index": {"bullpen_share": 31.0},
                },
                "summary": {},
            },
            "matchup": {"summary": {}},
            "clutch_moments": {"found": False, "moments": []},
        }
        response_payload = {
            "headline": "한화 이글스 승리, 데이터 기반 경기 리뷰",
            "sentiment": "positive",
            "key_metrics": [
                {
                    "label": "팀 타격 생산성",
                    "value": "NC 다이노스 0.799 / 한화 이글스 0.783",
                    "status": "warning",
                    "trend": "neutral",
                    "is_critical": True,
                }
            ],
            "analysis": {
                "summary": "NC 다이노스 4 / 한화 이글스 11 경기에서 한화 이글스가 이겼습니다.",
                "verdict": "한화 이글스가 결승타 구간을 실제 결과로 연결했습니다.",
                "strengths": [],
                "weaknesses": [],
                "risks": [],
                "why_it_matters": ["경기 후반 추가 득점이 승부를 갈랐습니다."],
                "swing_factors": ["2회 최재훈 타석이 핵심이었습니다."],
                "watch_points": ["추가 득점 직전 주자 운영을 다시 볼 필요가 있습니다."],
                "uncertainty": [],
            },
            "detailed_markdown": (
                "## 최근 전력\n"
                "- OPS 우위가 먼저 확인됐습니다.\n\n"
                "## 불펜 상태\n"
                "- NC 다이노스 4 / 한화 이글스 11 경기에서 한화 이글스가 이겼습니다.\n\n"
                "## 선발 투수\n"
                "- 경기 후반 추가 득점이 승부를 갈랐습니다.\n\n"
                "## 결과 진단\n"
                "- 한화 이글스가 결승타 구간을 실제 결과로 연결했습니다."
            ),
            "coach_note": "한화 이글스가 결과를 가져갔습니다.",
        }

        processed = _postprocess_coach_response_payload(
            response_payload,
            evidence=evidence,
            used_evidence=["game", "recent"],
            grounding_reasons=[],
            grounding_warnings=[
                "요청한 focus 중 상대 전적 근거가 부족해 확인 가능한 항목만 분석합니다."
            ],
            tool_results=tool_results,
            resolved_focus=["recent_form", "bullpen", "starter", "matchup", "batting"],
        )

        markdown = processed["detailed_markdown"]
        assert (
            "## 최근 전력\n- NC 다이노스 최근 4승 5패 1무, 득실 -21 / 한화 이글스 최근 4승 6패, 득실 -9"
            in markdown
        )
        assert (
            "## 불펜 상태\n- 불펜 비중은 NC 다이노스 31.0% / 한화 이글스 38.0%로 차이가 확인됩니다."
            in markdown
        )
        assert "## 선발 투수\n- NC 다이노스 김태경 / 한화 이글스 류현진" in markdown
        assert "OPS 우위가 먼저 확인됐습니다." not in markdown
        assert (
            "NC 다이노스 4 / 한화 이글스 11 경기에서 한화 이글스가 이겼습니다."
            not in markdown
        )

    def test_postprocess_completed_payload_compacts_repeated_clutch_sections(self):
        from app.routers.coach import _postprocess_coach_response_payload

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
            summary_items=["결승타 문성주 (1회 1사 만루서 밀어내기 4구)"],
        )
        tool_results = {
            "home": {"recent": {"found": False}, "advanced": {}, "summary": {}},
            "away": {"recent": {"found": False}, "advanced": {}, "summary": {}},
            "matchup": {"summary": {}},
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
        response_payload = {
            "headline": "LG 트윈스 승리, 데이터 기반 경기 리뷰",
            "sentiment": "positive",
            "key_metrics": [],
            "analysis": {
                "summary": "KIA 타이거즈 2 / LG 트윈스 7 경기에서 LG 트윈스가 이겼습니다.",
                "verdict": "LG 트윈스 승리의 분기점은 1회말 8번타자 이재원 장면의 WPA 21.2%p 변동이 실제 승부처였습니다.",
                "strengths": [],
                "weaknesses": [],
                "risks": [],
                "why_it_matters": [
                    "1회말 8번타자 이재원 장면의 WPA 21.2%p 변동이 실제 승부처였습니다."
                ],
                "swing_factors": [
                    "1회말 8번타자 이재원 장면의 WPA 21.2%p 변동이 실제 승부처였습니다."
                ],
                "watch_points": [
                    "1회말 8번타자 이재원 장면 직전 배터리 선택과 작전 흐름을 다시 볼 필요가 있습니다."
                ],
                "uncertainty": [],
            },
            "detailed_markdown": (
                "## 결과 진단\n"
                "- LG 트윈스 승리의 분기점은 1회말 8번타자 이재원 장면의 WPA 21.2%p 변동이 실제 승부처였습니다.\n\n"
                "## 결과를 가른 이유\n"
                "- 1회말 8번타자 이재원 장면의 WPA 21.2%p 변동이 실제 승부처였습니다.\n\n"
                "## 실제 전환점\n"
                "- 1회말 8번타자 이재원 장면의 WPA 21.2%p 변동이 실제 승부처였습니다.\n\n"
                "## 다시 볼 장면\n"
                "- 1회말 8번타자 이재원 장면 직전 배터리 선택과 작전 흐름을 다시 볼 필요가 있습니다."
            ),
            "coach_note": "LG 트윈스가 결과를 가져갔습니다.",
        }

        processed = _postprocess_coach_response_payload(
            response_payload,
            evidence=evidence,
            used_evidence=["game", "game_summary"],
            grounding_reasons=[],
            grounding_warnings=[],
            tool_results=tool_results,
            resolved_focus=["recent_form", "bullpen"],
        )

        assert (
            processed["analysis"]["verdict"]
            == "LG 트윈스 승리의 분기점은 1회말 승부처 대응이었습니다."
        )
        assert (
            processed["analysis"]["why_it_matters"][0]
            == "LG 트윈스는 고레버리지 기회를 득점 흐름으로 연결하며 주도권을 잡았습니다."
        )
        assert (
            processed["analysis"]["swing_factors"][0]
            == "1회말 8번타자 이재원 장면의 WPA 21.2%p 변동이 실제 승부처였습니다."
        )
        assert (
            processed["analysis"]["watch_points"][0]
            == "해당 승부처 직전 배터리 선택과 주자 운영을 다시 볼 필요가 있습니다."
        )
        assert (
            "## 결과 진단\n- LG 트윈스 승리의 분기점은 1회말 승부처 대응이었습니다."
            in processed["detailed_markdown"]
        )
        assert (
            "## 결과를 가른 이유\n- LG 트윈스는 고레버리지 기회를 득점 흐름으로 연결하며 주도권을 잡았습니다."
            in processed["detailed_markdown"]
        )
        assert (
            "## 다시 볼 장면\n- 해당 승부처 직전 배터리 선택과 주자 운영을 다시 볼 필요가 있습니다."
            in processed["detailed_markdown"]
        )

    def test_postprocess_scheduled_partial_payload_realigns_llm_sections(self):
        from app.routers.coach import _postprocess_coach_response_payload

        evidence = _build_game_evidence(
            game_status="SCHEDULED",
            game_status_bucket="SCHEDULED",
            home_team_code="LG",
            away_team_code="LT",
            home_team_name="LG 트윈스",
            away_team_name="롯데 자이언츠",
            home_pitcher=None,
            away_pitcher=None,
            lineup_announced=False,
        )
        tool_results = {
            "home": {
                "recent": {
                    "summary": {"wins": 9, "losses": 1, "draws": 0, "run_diff": 26}
                },
                "advanced": {"metrics": {"batting": {"ops": 0.740}}},
                "summary": {},
            },
            "away": {
                "recent": {
                    "summary": {"wins": 3, "losses": 7, "draws": 0, "run_diff": -19}
                },
                "advanced": {"metrics": {"batting": {"ops": 0.905}}},
                "summary": {},
            },
            "matchup": {},
        }
        response_payload = {
            "headline": "LG 트윈스 vs 롯데 자이언츠 예정 경기 분석",
            "sentiment": "neutral",
            "key_metrics": [
                {
                    "label": "최근 흐름",
                    "value": "롯데 자이언츠 최근 3승 7패 / LG 트윈스 최근 9승 1패",
                },
                {
                    "label": "팀 타격 생산성",
                    "value": "롯데 자이언츠 OPS 0.905 / LG 트윈스 OPS 0.740",
                },
            ],
            "analysis": {
                "summary": "롯데 자이언츠의 장타 생산성과 LG 트윈스의 최근 상승 흐름이 함께 변수입니다.",
                "verdict": "롯데 자이언츠가 타격에서 앞서지만 선발 발표 전이라 첫 투수 교체 시점과 라인업 변화까지 함께 봐야 합니다.",
                "strengths": [],
                "weaknesses": [],
                "risks": [],
                "why_it_matters": [
                    "롯데 자이언츠의 장타 생산성이 경기 초반 흐름을 좌우할 수 있습니다."
                ],
                "swing_factors": [
                    "라인업 변화와 첫 투수 교체 시점을 함께 봐야 합니다."
                ],
                "watch_points": ["라인업 변화와 첫 투수 교체 시점을 함께 봐야 합니다."],
                "uncertainty": ["라인업 발표 전까지는 변수도 큽니다."],
            },
            "detailed_markdown": (
                "## 최근 전력\n"
                "- 롯데 자이언츠 최근 3승 7패 / LG 트윈스 최근 9승 1패\n\n"
                "## 코치 판단\n"
                "- 롯데 자이언츠가 타격에서 앞서지만 선발 발표 전이라 첫 투수 교체 시점과 라인업 변화까지 함께 봐야 합니다.\n\n"
                "## 승부 스윙 포인트\n"
                "- 라인업 변화와 첫 투수 교체 시점을 함께 봐야 합니다.\n\n"
                "## 체크 포인트\n"
                "- 라인업 변화와 첫 투수 교체 시점을 함께 봐야 합니다."
            ),
            "coach_note": "라인업 변화와 첫 투수 교체 시점을 함께 봐야 합니다.",
        }

        processed = _postprocess_coach_response_payload(
            response_payload,
            evidence=evidence,
            used_evidence=["game", "team_recent_form", "opponent_recent_form"],
            grounding_reasons=[
                "missing_starters",
                "missing_lineups",
                "missing_summary",
            ],
            grounding_warnings=[],
            tool_results=tool_results,
            resolved_focus=["recent_form", "bullpen", "starter"],
        )

        assert (
            processed["analysis"]["verdict"]
            == "LG 트윈스가 최근 흐름 우세로 근소하게 앞섭니다."
        )
        assert (
            processed["analysis"]["why_it_matters"][0]
            == "LG 트윈스가 최근 흐름 우위로 경기 중반 운영 선택지를 먼저 확보할 가능성이 있습니다."
        )
        assert (
            processed["analysis"]["swing_factors"][0]
            == "선발 발표 전이라 첫 투수 교체 시점이 핵심 변수입니다."
        )
        assert (
            processed["analysis"]["watch_points"][0]
            == "첫 득점 직후 어느 팀이 먼저 불펜 카드로 반응하는지 확인할 필요가 있습니다."
        )
        assert "핵심 변수는 선발 발표 전이라" not in processed["coach_note"]
        assert (
            "## 코치 판단\n- LG 트윈스가 최근 흐름 우세로 근소하게 앞섭니다."
            in processed["detailed_markdown"]
        )
        assert (
            "## 왜 중요한가\n- LG 트윈스가 최근 흐름 우위로 경기 중반 운영 선택지를 먼저 확보할 가능성이 있습니다."
            in processed["detailed_markdown"]
        )
        assert (
            "롯데 자이언츠의 팀 타격 생산성 반격 여지는 남아 있습니다."
            in processed["analysis"]["summary"]
        )
        assert (
            "## 승부 스윙 포인트\n- 선발 발표 전이라 첫 투수 교체 시점이 핵심 변수입니다."
            in processed["detailed_markdown"]
        )
        assert (
            "## 체크 포인트\n- 첫 득점 직후 어느 팀이 먼저 불펜 카드로 반응하는지 확인할 필요가 있습니다."
            in processed["detailed_markdown"]
        )

    def test_scheduled_deterministic_preview_has_snapshot_and_guaranteed_metrics(self):
        """Scheduled games (no starters, no lineup) must still return a rich preview.

        Regression: scheduled deterministic responses used to leave
        ``swing_factors`` / ``watch_points`` / ``uncertainty`` sparse and omit
        a snapshot section, which collapsed whole insight cards on the
        frontend. The scheduled team-level builder should now always emit a
        '## 경기 전 스냅샷' section and at least three key_metrics.
        """

        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence(
            home_team_name="KT 위즈",
            away_team_name="LG 트윈스",
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
                "advanced": {"metrics": {"batting": {"ops": 0.740}}},
                "recent": {
                    "found": True,
                    "summary": {"wins": 5, "losses": 2, "draws": 0, "run_diff": 6},
                },
                "player_form_signals": {},
            },
            "away": {
                "summary": {"found": True},
                "advanced": {"metrics": {"batting": {"ops": 0.715}}},
                "recent": {
                    "found": True,
                    "summary": {"wins": 3, "losses": 4, "draws": 0, "run_diff": -1},
                },
                "player_form_signals": {},
            },
            "matchup": {},
            "clutch_moments": {"found": False, "moments": []},
        }

        payload = _build_deterministic_coach_response(evidence, tool_results)

        assert "## 경기 전 스냅샷" in payload["detailed_markdown"]
        assert "**KT 위즈**" in payload["detailed_markdown"]
        assert "**LG 트윈스**" in payload["detailed_markdown"]

        assert payload["analysis"]["swing_factors"]
        assert payload["analysis"]["watch_points"]
        assert payload["analysis"]["uncertainty"]

        assert len(payload["key_metrics"]) >= 3
        metric_labels = {metric["label"] for metric in payload["key_metrics"]}
        assert "최근 흐름" in metric_labels
        assert "팀 타격 생산성" in metric_labels
        assert "발표 선발" in metric_labels

    def test_scheduled_deterministic_preview_keeps_snapshot_with_lineup_context(self):
        """Scheduled partial responses keep the snapshot even when lineup rows exist."""

        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence(
            home_team_name="KT 위즈",
            away_team_name="SSG 랜더스",
            home_pitcher=None,
            away_pitcher=None,
            lineup_announced=True,
            home_lineup=["상위 타선"],
            away_lineup=["상위 타선"],
            summary_items=[],
        )
        tool_results = {
            "home": {
                "summary": {"found": True},
                "advanced": {"metrics": {"batting": {"ops": 0.740}}},
                "recent": {
                    "found": True,
                    "summary": {"wins": 4, "losses": 3, "draws": 0, "run_diff": 3},
                },
                "player_form_signals": {},
            },
            "away": {
                "summary": {"found": True},
                "advanced": {"metrics": {"batting": {"ops": 0.721}}},
                "recent": {
                    "found": True,
                    "summary": {"wins": 3, "losses": 4, "draws": 0, "run_diff": -2},
                },
                "player_form_signals": {},
            },
            "matchup": {},
            "clutch_moments": {"found": False, "moments": []},
        }

        payload = _build_deterministic_coach_response(
            evidence,
            tool_results,
            resolved_focus=["recent_form", "bullpen"],
        )

        assert "## 경기 전 스냅샷" in payload["detailed_markdown"]
        assert "## 최근 전력" in payload["detailed_markdown"]
        assert "## 불펜 상태" in payload["detailed_markdown"]
        assert "- 발표 선발:" not in payload["detailed_markdown"]

    def test_scheduled_manual_detail_ensure_markdown_adds_snapshot_when_grounded(self):
        """Grounded scheduled LLM responses must also keep the snapshot section."""

        from app.routers.coach import _ensure_detailed_markdown

        evidence = _build_game_evidence(
            home_team_name="KT 위즈",
            away_team_name="LG 트윈스",
            home_pitcher="사우어",
            away_pitcher="웰스",
            lineup_announced=False,
            home_lineup=[],
            away_lineup=[],
            summary_items=[],
        )
        tool_results = {
            "home": {
                "summary": {"found": True},
                "advanced": {"metrics": {"batting": {"ops": 0.866}}},
                "recent": {
                    "found": True,
                    "summary": {"wins": 8, "losses": 2, "draws": 0, "run_diff": 24},
                },
                "player_form_signals": {},
            },
            "away": {
                "summary": {"found": True},
                "advanced": {"metrics": {"batting": {"ops": 0.723}}},
                "recent": {
                    "found": True,
                    "summary": {"wins": 8, "losses": 2, "draws": 0, "run_diff": 23},
                },
                "player_form_signals": {},
            },
            "matchup": {},
            "clutch_moments": {"found": False, "moments": []},
        }
        payload = {
            "headline": "LG 트윈스 vs KT 위즈, 득점력 대결 전망",
            "sentiment": "neutral",
            "key_metrics": [],
            "analysis": {
                "summary": "양 팀 모두 최근 흐름이 좋습니다.",
                "strengths": [],
                "weaknesses": [],
                "risks": [],
            },
            "detailed_markdown": (
                "## 최근 전력\n"
                "- 양 팀 모두 최근 흐름이 좋습니다.\n\n"
                "## 불펜 상태\n"
                "- 불펜 운용 데이터는 제한적입니다."
            ),
            "coach_note": "근거 기반 상세 분석입니다.",
        }

        _ensure_detailed_markdown(
            payload,
            ["recent_form", "bullpen"],
            evidence=evidence,
            tool_results=tool_results,
        )

        markdown = payload["detailed_markdown"]
        assert markdown.startswith("## 경기 전 스냅샷")
        assert "**LG 트윈스**" in markdown
        assert "**KT 위즈**" in markdown
        assert "- 발표 선발: LG 트윈스 웰스 / KT 위즈 사우어" in markdown
        assert "## 최근 전력" in markdown
        assert "## 불펜 상태" in markdown


# ============================================================
# _attach_scheduled_win_probability (예측 다이얼로그 승률 hero)
# ============================================================
