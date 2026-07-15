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

class TestCoachFastPath:
    """Coach Fast Path 통합 테스트"""

    def test_cache_prompt_version_fits_cache_column(self):
        from app.core.coach_cache_contract import COACH_CACHE_PROMPT_VERSION

        assert len(COACH_CACHE_PROMPT_VERSION) <= 32

    def test_claim_cache_generation_retries_once_after_operational_error(
        self, monkeypatch
    ):
        from app.routers import coach

        calls: list[dict[str, object]] = []

        class _Pool:
            check_count = 0

            def check(self):
                self.check_count += 1

        async def _fake_claim_once(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise coach.psycopg.OperationalError("server closed connection")
            return "HIT", {"ok": True}, None, None, 2

        pool = _Pool()
        monkeypatch.setattr(coach, "_claim_cache_generation_once", _fake_claim_once)

        result = asyncio.run(coach._claim_cache_generation(
            pool=pool,
            cache_key="cache-key",
            team_id="HH",
            year=2026,
            prompt_version="v-test",
            model_name="model",
            lease_owner="owner",
            completed_ttl_seconds=None,
        ))

        assert result == ("HIT", {"ok": True}, None, None, 2)
        assert len(calls) == 2
        assert pool.check_count == 1

    def test_build_coach_query(self):
        """Coach 쿼리 빌드 테스트"""
        from app.routers.coach import _build_coach_query

        # 기본 쿼리
        query = _build_coach_query("KIA 타이거즈", [])
        assert "KIA 타이거즈" in query
        assert "종합적인 전력" in query

        # 특정 focus
        query = _build_coach_query("두산 베어스", ["bullpen", "batting"])
        assert "bullpen" in query or "불펜" in query
        assert "batting" in query or "타격" in query

    def test_build_coach_query_uses_review_tone_for_completed_game(self):
        from app.routers.coach import _build_coach_query

        query = _build_coach_query(
            "LG 트윈스",
            ["recent_form", "matchup"],
            opponent_name="KT 위즈",
            game_status_bucket="COMPLETED",
        )

        assert "경기 종료 기준 리뷰" in query
        assert "실제 경기 결과" in query
        assert "상승세/하락세" not in query

    def test_build_coach_query_avoids_bullpen_jargon_for_scheduled_game(self):
        from app.routers.coach import _build_coach_query

        query = _build_coach_query(
            "SSG 랜더스",
            ["recent_form", "bullpen"],
            opponent_name="한화 이글스",
            game_status_bucket="SCHEDULED",
        )

        assert "하이 레버리지" not in query
        assert "과부하 지표" not in query
        assert "경기 후반 접전 대응력" in query
        assert "최근 소모 흐름" in query

    def test_completed_deterministic_response_uses_review_labels(self):
        from app.routers.coach import (
            _build_deterministic_headline,
            _build_deterministic_markdown,
        )

        evidence = _build_game_evidence(
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
        )

        headline = _build_deterministic_headline(evidence, {})
        markdown = _build_deterministic_markdown(evidence, {})

        assert "경기 리뷰" in headline
        assert "## 결과 진단" in markdown
        assert "## 다시 볼 장면" in markdown

    def test_completed_deterministic_response_includes_clutch_signal(self):
        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence(
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
        )
        tool_results = {
            "home": {"recent": {"found": False}, "advanced": {}, "summary": {}},
            "away": {"recent": {"found": False}, "advanced": {}, "summary": {}},
            "matchup": {},
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
                    }
                ],
            },
        }

        payload = _build_deterministic_coach_response(evidence, tool_results)

        assert any(
            metric["label"] == "최대 WPA 변동" for metric in payload["key_metrics"]
        )
        assert any("WPA" in item for item in payload["analysis"]["swing_factors"])
        assert "8회말" in payload["coach_note"]

    def test_completed_deterministic_response_uses_summary_fallback_without_wpa(self):
        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence(
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            summary_items=["결승타 김도영", "오스틴 3안타"],
        )
        tool_results = {
            "home": {"recent": {"found": False}, "advanced": {}, "summary": {}},
            "away": {"recent": {"found": False}, "advanced": {}, "summary": {}},
            "matchup": {},
            "clutch_moments": {"found": False, "moments": []},
        }

        payload = _build_deterministic_coach_response(evidence, tool_results)

        assert payload["headline"].startswith("결승타 김도영")
        assert any(
            metric["label"] == "승부처 요약" and "결승타 김도영" in metric["value"]
            for metric in payload["key_metrics"]
        )
        assert any(
            "결승타 김도영" in item for item in payload["analysis"]["swing_factors"]
        )
        assert any(
            "WPA 수치가 없어" in item for item in payload["analysis"]["uncertainty"]
        )
        assert "결승타 김도영" in payload["coach_note"]
        assert not any(
            metric["label"] == "최대 WPA 변동" for metric in payload["key_metrics"]
        )

    def test_completed_summary_item_skips_structured_preview_json(self):
        from app.routers.coach import (
            _build_deterministic_headline,
            _format_game_summary_item,
        )

        preview_summary = _format_game_summary_item(
            "프리뷰",
            None,
            '{"game_id":"20260426LGOB0","home_lineup":[{"player_name":"김민석"}]}',
        )
        evidence = _build_game_evidence(
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            home_score=4,
            away_score=3,
            winning_team_code="DB",
            winning_team_name="두산 베어스",
            summary_items=[item for item in [preview_summary] if item],
        )

        headline = _build_deterministic_headline(evidence, {})

        assert preview_summary is None
        assert "프리뷰" not in headline
        assert "game_id" not in headline
        assert "두산 베어스 승리 리뷰" in headline

    def test_completed_summary_items_filter_json_before_limit(self):
        from app.routers.coach import _format_game_summary_items

        rows = [
            {
                "summary_type": "프리뷰",
                "player_name": None,
                "detail_text": '{"game_id":"20260426LGOB0"}',
            }
            for _ in range(9)
        ]
        rows.extend(
            [
                {
                    "summary_type": "결승타",
                    "player_name": "양의지",
                    "detail_text": "10회말 좌전 적시타",
                },
                {
                    "summary_type": "리뷰_WPA",
                    "player_name": None,
                    "detail_text": "10회말 핵심 장면 WPA 50.0%p",
                },
            ]
        )

        items = _format_game_summary_items(rows, limit=2)

        assert items == [
            "결승타 양의지 (10회말 좌전 적시타)",
            "리뷰_WPA (10회말 핵심 장면 WPA 50.0%p)",
        ]

    def test_completed_deterministic_response_anchors_actual_winner(self):
        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence(
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            away_team_name="KT 위즈",
            home_team_name="LG 트윈스",
            away_score=7,
            home_score=3,
            winning_team_code="KT",
            winning_team_name="KT 위즈",
            summary_items=["강백호 결승타"],
        )
        tool_results = {
            "home": {
                "recent": {
                    "summary": {"wins": 5, "losses": 0, "draws": 0, "run_diff": 12}
                },
                "advanced": {
                    "metrics": {"batting": {"ops": 0.845}},
                    "fatigue_index": {"bullpen_share": 28.0},
                },
                "summary": {},
            },
            "away": {
                "recent": {
                    "summary": {"wins": 1, "losses": 4, "draws": 0, "run_diff": -8}
                },
                "advanced": {
                    "metrics": {"batting": {"ops": 0.691}},
                    "fatigue_index": {"bullpen_share": 41.5},
                },
                "summary": {},
            },
            "matchup": {},
            "clutch_moments": {"found": False, "moments": []},
        }

        payload = _build_deterministic_coach_response(evidence, tool_results)
        serialized = json.dumps(payload, ensure_ascii=False)

        assert payload["headline"].startswith("KT 위즈 승리")
        assert payload["analysis"]["summary"].startswith(
            "KT 위즈 7 / LG 트윈스 3 경기에서 KT 위즈가 이겼고"
        )
        assert payload["analysis"]["verdict"].startswith("KT 위즈")
        assert payload["analysis"]["why_it_matters"]
        assert all(
            "LG 트윈스가 출루·장타 베이스라인에서 앞서" not in item
            for item in payload["analysis"]["why_it_matters"]
        )
        assert any(
            "실제 경기 결과" in item or "실제 경기" in item
            for item in payload["analysis"]["why_it_matters"]
        )
        assert payload["key_metrics"][0]["label"] == "최종 스코어"
        assert "LG 트윈스 승리" not in serialized

    def test_completed_fact_sheet_includes_final_score_and_winner(self):
        from app.routers.coach import (
            _build_coach_fact_sheet,
            _collect_allowed_entity_names,
            assess_game_evidence,
        )

        evidence = _build_game_evidence(
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            away_team_name="KT 위즈",
            home_team_name="LG 트윈스",
            away_score=7,
            home_score=3,
            winning_team_code="KT",
            winning_team_name="KT 위즈",
            summary_items=["강백호 결승타"],
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

        assert "최종 스코어: KT 위즈 7 / LG 트윈스 3" in fact_sheet.fact_lines
        assert "승리 팀: KT 위즈" in fact_sheet.fact_lines

    def test_postprocess_completed_payload_repairs_incorrect_result_direction(self):
        from app.routers.coach import _postprocess_coach_response_payload

        evidence = _build_game_evidence(
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            away_team_name="KT 위즈",
            home_team_name="LG 트윈스",
            away_score=7,
            home_score=3,
            winning_team_code="KT",
            winning_team_name="KT 위즈",
            summary_items=["강백호 결승타"],
        )
        tool_results = {
            "home": {
                "recent": {
                    "summary": {"wins": 5, "losses": 0, "draws": 0, "run_diff": 12}
                },
                "advanced": {"metrics": {"batting": {"ops": 0.845}}},
                "summary": {},
            },
            "away": {
                "recent": {
                    "summary": {"wins": 1, "losses": 4, "draws": 0, "run_diff": -8}
                },
                "advanced": {"metrics": {"batting": {"ops": 0.691}}},
                "summary": {},
            },
            "matchup": {},
            "clutch_moments": {"found": False, "moments": []},
        }
        response_payload = {
            "headline": "LG 트윈스 승리, 데이터 기반 경기 리뷰",
            "sentiment": "neutral",
            "key_metrics": [
                {
                    "label": "정규시즌 OPS",
                    "value": "KT 위즈 0.691 / LG 트윈스 0.845",
                    "status": "warning",
                    "trend": "neutral",
                    "is_critical": False,
                }
            ],
            "analysis": {
                "summary": "LG 트윈스가 기초 지표 우위를 실제 결과로 연결했습니다.",
                "verdict": "LG 트윈스가 폼과 불펜 우위로 승부처를 확보했습니다.",
                "strengths": [],
                "weaknesses": [],
                "risks": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "detailed_markdown": (
                "## 결과 진단\n"
                "- LG 트윈스가 기초 지표 우위를 실제 결과로 연결했습니다."
            ),
            "coach_note": "LG 트윈스가 승부처를 가져갔습니다.",
        }

        processed = _postprocess_coach_response_payload(
            response_payload,
            evidence=evidence,
            used_evidence=["game", "recent"],
            grounding_reasons=[],
            tool_results=tool_results,
            resolved_focus=["recent_form", "bullpen"],
        )
        serialized = json.dumps(processed, ensure_ascii=False)

        assert processed["headline"].startswith("KT 위즈 승리")
        assert processed["analysis"]["summary"].startswith(
            "KT 위즈 7 / LG 트윈스 3 경기에서 KT 위즈가 이겼고"
        )
        assert processed["analysis"]["verdict"].startswith("KT 위즈")
        assert processed["analysis"]["why_it_matters"]
        assert all(
            "LG 트윈스가 출루·장타 베이스라인에서 앞서" not in item
            for item in processed["analysis"]["why_it_matters"]
        )
        assert processed["key_metrics"][0]["label"] == "최종 스코어"
        assert "LG 트윈스가 폼과 불펜 우위로 승부처를 확보했습니다." not in serialized

    def test_preview_deterministic_response_includes_form_signal_metric(self):
        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence()
        tool_results = {
            "home": {
                "summary": {"found": True},
                "advanced": {"metrics": {"batting": {"ops": 0.771}}},
                "recent": {"found": False},
                "player_form_signals": {
                    "found": True,
                    "batters": [
                        {
                            "player_name": "홍창기",
                            "form_score": 68.4,
                            "form_status": "hot",
                        }
                    ],
                    "pitchers": [
                        {
                            "player_name": "임찬규",
                            "form_score": 61.2,
                            "form_status": "steady",
                            "role": "starter",
                        }
                    ],
                },
            },
            "away": {
                "summary": {"found": True},
                "advanced": {"metrics": {"batting": {"ops": 0.742}}},
                "recent": {"found": False},
                "player_form_signals": {
                    "found": True,
                    "batters": [
                        {
                            "player_name": "강백호",
                            "form_score": 44.1,
                            "form_status": "cold",
                        }
                    ],
                    "pitchers": [
                        {
                            "player_name": "쿠에바스",
                            "form_score": 49.7,
                            "form_status": "steady",
                            "role": "starter",
                        }
                    ],
                },
            },
            "matchup": {},
        }

        payload = _build_deterministic_coach_response(evidence, tool_results)
        form_metric = next(
            metric for metric in payload["key_metrics"] if metric["label"] == "폼 진단"
        )

        assert "상승" in form_metric["value"]
        assert "하락" in form_metric["value"]
        assert any("폼 점수" in item for item in payload["analysis"]["strengths"])

    def test_scheduled_preview_fallback_uses_team_level_sentences_when_lineups_missing(
        self,
    ):
        from app.routers.coach import (
            _build_deterministic_coach_response,
            _find_missing_focus_sections,
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

        payload = _build_deterministic_coach_response(
            evidence,
            tool_results,
            resolved_focus=["recent_form", "bullpen"],
        )
        serialized = json.dumps(payload, ensure_ascii=False)

        assert "페라자" not in serialized
        assert "최정" not in serialized
        assert "## 최근 전력\n-" in payload["detailed_markdown"]
        assert "## 불펜 상태\n-" in payload["detailed_markdown"]
        assert _find_missing_focus_sections(payload, ["recent_form", "bullpen"]) == []
        assert "선발 발표 상위 타선" not in serialized
        assert "라인업 미발표 상위 타선 타순 기반 핵심 구간는" not in serialized
        assert "상위 타선 기초 지표" not in serialized
        assert "상위 타선 선취점" not in serialized
        assert "상승세" in serialized
        assert "상승 83.1" not in serialized
        assert "93.0로" not in serialized
        assert "점을 기록하며" in serialized

    def test_preview_deterministic_response_avoids_sentence_gluing(self):
        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence()
        tool_results = {
            "home": {
                "recent": {
                    "summary": {"wins": 3, "losses": 2, "draws": 0, "run_diff": 5}
                },
                "advanced": {"metrics": {"batting": {"ops": 0.790}}},
                "summary": {},
            },
            "away": {
                "recent": {
                    "summary": {"wins": 2, "losses": 3, "draws": 0, "run_diff": -3}
                },
                "advanced": {"metrics": {"batting": {"ops": 0.720}}},
                "summary": {},
            },
            "matchup": {},
        }

        payload = _build_deterministic_coach_response(evidence, tool_results)

        assert "좌우합니다.에 따라" not in payload["analysis"]["verdict"]
        assert "좌우합니다.에 따라" not in payload["detailed_markdown"]

    def test_scheduled_deterministic_response_separates_section_roles(self):
        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence(
            game_status="SCHEDULED",
            game_status_bucket="SCHEDULED",
            home_team_code="HH",
            away_team_code="SSG",
            home_team_name="한화 이글스",
            away_team_name="SSG 랜더스",
            home_pitcher="문동주",
            away_pitcher="김광현",
            lineup_announced=False,
        )
        tool_results = {
            "home": {
                "recent": {
                    "summary": {"wins": 2, "losses": 3, "draws": 0, "run_diff": -2}
                },
                "advanced": {"metrics": {"batting": {"ops": 0.790}}},
                "summary": {},
            },
            "away": {
                "recent": {
                    "summary": {"wins": 4, "losses": 1, "draws": 0, "run_diff": 8}
                },
                "advanced": {"metrics": {"batting": {"ops": 0.840}}},
                "summary": {},
            },
            "matchup": {},
            "clutch_moments": {"found": False, "moments": []},
        }

        payload = _build_deterministic_coach_response(
            evidence,
            tool_results,
            resolved_focus=["recent_form", "starter", "bullpen"],
        )

        assert (
            payload["analysis"]["verdict"]
            == "SSG 랜더스가 최근 흐름과 득점 연결력에서 앞섭니다."
        )
        assert (
            payload["analysis"]["why_it_matters"][0]
            == "SSG 랜더스가 최근 흐름과 출루·장타 지표를 함께 앞세워 초중반 주도권을 먼저 잡을 가능성이 있습니다."
        )
        assert (
            payload["analysis"]["swing_factors"][0]
            == "발표 선발 뒤 첫 불펜 카드가 핵심 변수입니다."
        )
        assert (
            payload["analysis"]["watch_points"][0]
            == "첫 득점 직후 어느 팀이 먼저 불펜 카드로 반응하는지 확인할 필요가 있습니다."
        )
        assert (
            "## 코치 판단\n- SSG 랜더스가 최근 흐름과 득점 연결력에서 앞섭니다."
            in payload["detailed_markdown"]
        )
        assert (
            "## 승부 스윙 포인트\n- 발표 선발 뒤 첫 불펜 카드가 핵심 변수입니다."
            in payload["detailed_markdown"]
        )
        assert (
            "## 체크 포인트\n- 첫 득점 직후 어느 팀이 먼저 불펜 카드로 반응하는지 확인할 필요가 있습니다."
            in payload["detailed_markdown"]
        )

    def test_scheduled_deterministic_response_uses_plain_risk_language(self):
        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence(
            game_status="SCHEDULED",
            game_status_bucket="SCHEDULED",
            home_team_code="HH",
            away_team_code="SSG",
            home_team_name="한화 이글스",
            away_team_name="SSG 랜더스",
            home_pitcher=None,
            away_pitcher=None,
            lineup_announced=False,
        )
        tool_results = {
            "home": {
                "recent": {
                    "summary": {"wins": 2, "losses": 3, "draws": 0, "run_diff": -2}
                },
                "advanced": {"metrics": {"batting": {"ops": 0.790}}},
                "summary": {},
            },
            "away": {
                "recent": {
                    "summary": {"wins": 4, "losses": 1, "draws": 0, "run_diff": 8}
                },
                "advanced": {"metrics": {"batting": {"ops": 0.840}}},
                "summary": {},
            },
            "matchup": {},
            "clutch_moments": {"found": False, "moments": []},
        }

        payload = _build_deterministic_coach_response(
            evidence,
            tool_results,
            resolved_focus=["recent_form", "starter", "batting"],
        )
        analysis = payload["analysis"]
        narrative_blob = "\n".join(
            [
                analysis["summary"],
                analysis["verdict"],
                payload["detailed_markdown"],
                payload["coach_note"],
                *analysis["strengths"],
                *analysis["weaknesses"],
                *[risk["description"] for risk in analysis["risks"]],
                *analysis["why_it_matters"],
                *analysis["swing_factors"],
                *analysis["watch_points"],
                *analysis["uncertainty"],
            ]
        )

        assert analysis["risks"]
        assert any("득점 연결력" in risk["description"] for risk in analysis["risks"])
        assert "공격 생산성" not in narrative_blob
        assert "타격 생산성" not in narrative_blob
        assert "클러치 생산성" not in narrative_blob
        assert not re.search(r"\bOPS\b", narrative_blob)
        assert not re.search(r"\bWPA\b", narrative_blob)

    def test_normalize_coach_payload_backfills_risks_with_plain_language(self):
        from app.core.coach_validator import normalize_coach_payload

        payload = {
            "headline": "테스트 분석",
            "sentiment": "neutral",
            "key_metrics": [],
            "analysis": {
                "summary": "양 팀 지표는 박빙입니다.",
                "verdict": "득점 연결력이 변수입니다.",
                "strengths": ["SSG 랜더스는 최근 득점 흐름이 좋습니다."],
                "weaknesses": [
                    "한화 이글스는 팀 OPS 열세로 초반 득점 설계가 과제입니다."
                ],
                "risks": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": ["선발 발표 전이라 초반 흐름 해석은 보수적입니다."],
            },
            "detailed_markdown": "## 코치 판단\n- 득점 연결력이 변수입니다.",
            "coach_note": "득점 연결력이 변수입니다.",
        }

        normalized, reasons = normalize_coach_payload(payload)
        risks = normalized["analysis"]["risks"]

        assert "backfill_empty_risks" in reasons
        assert risks
        assert "득점 연결력" in risks[0]["description"]
        assert not re.search(r"\bOPS\b", risks[0]["description"])
        assert "타격 생산성" not in risks[0]["description"]

    def test_completed_deterministic_response_avoids_sentence_gluing(self):
        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence(
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
            summary_items=["결승타 김도영"],
        )
        tool_results = {
            "home": {
                "recent": {
                    "summary": {"wins": 4, "losses": 1, "draws": 0, "run_diff": 7}
                },
                "advanced": {"metrics": {"batting": {"ops": 0.801}}},
                "summary": {},
            },
            "away": {
                "recent": {
                    "summary": {"wins": 2, "losses": 3, "draws": 0, "run_diff": -2}
                },
                "advanced": {"metrics": {"batting": {"ops": 0.734}}},
                "summary": {},
            },
            "matchup": {},
            "clutch_moments": {"found": False, "moments": []},
        }

        payload = _build_deterministic_coach_response(evidence, tool_results)

        assert "기록됐습니다.이 결과를" not in payload["analysis"]["verdict"]
        assert "기록됐습니다.이 결과를" not in payload["detailed_markdown"]
        assert "기록됐습니다.이 결과를" not in payload["coach_note"]

    def test_deterministic_response_deduplicates_coach_note_clutch_sentence(self):
        from app.routers.coach import _build_deterministic_coach_response

        evidence = _build_game_evidence(
            game_status="COMPLETED",
            game_status_bucket="COMPLETED",
        )
        tool_results = {
            "home": {
                "recent": {
                    "summary": {"wins": 4, "losses": 1, "draws": 0, "run_diff": 8}
                },
                "advanced": {
                    "metrics": {"batting": {"ops": 0.812}},
                    "fatigue_index": {"bullpen_share": 41.0},
                },
                "summary": {},
            },
            "away": {
                "recent": {
                    "summary": {"wins": 1, "losses": 4, "draws": 0, "run_diff": -6}
                },
                "advanced": {
                    "metrics": {"batting": {"ops": 0.701}},
                    "fatigue_index": {"bullpen_share": 49.8},
                },
                "summary": {},
            },
            "matchup": {},
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
                    }
                ],
            },
        }

        payload = _build_deterministic_coach_response(evidence, tool_results)

        assert payload["coach_note"].count("8회말 홍창기 타석") == 1
