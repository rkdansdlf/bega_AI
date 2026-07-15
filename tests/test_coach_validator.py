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

class TestCoachValidator:
    """Coach 응답 검증기 테스트"""

    def test_extract_json_from_pure_json(self):
        """순수 JSON 추출 테스트"""
        from app.core.coach_validator import extract_json_from_response

        raw = '{"headline": "테스트", "sentiment": "neutral"}'
        result = extract_json_from_response(raw)
        assert result is not None
        assert "headline" in result

    def test_extract_json_from_code_block(self):
        """코드 블록에서 JSON 추출 테스트"""
        from app.core.coach_validator import extract_json_from_response

        raw = """여기 분석 결과입니다:

```json
{"headline": "선발 붕괴가 불펜 과부하로 직결", "sentiment": "negative"}
```

위 분석을 참고하세요."""

        result = extract_json_from_response(raw)
        assert result is not None
        data = json.loads(result)
        assert data["headline"] == "선발 붕괴가 불펜 과부하로 직결"
        assert data["sentiment"] == "negative"

    def test_extract_json_from_mixed_text(self):
        """혼합 텍스트에서 JSON 추출 테스트"""
        from app.core.coach_validator import extract_json_from_response

        raw = """분석 시작
{"headline": "테스트 헤드라인", "sentiment": "positive", "key_metrics": []}
분석 종료"""

        result = extract_json_from_response(raw)
        assert result is not None
        data = json.loads(result)
        assert data["headline"] == "테스트 헤드라인"

    def test_parse_valid_coach_response(self):
        """유효한 Coach 응답 파싱 테스트"""
        from app.core.coach_validator import parse_coach_response

        raw = """{
            "headline": "KIA 타이거즈 선발진 안정화 필요",
            "sentiment": "negative",
            "key_metrics": [
                {"label": "팀 ERA", "value": "4.52", "status": "주의", "trend": "up", "is_critical": true}
            ],
            "analysis": {
                "summary": "선발 안정감이 흔들려 후반 운영 부담이 커진 상태입니다.",
                "verdict": "선발 이닝이 짧아지면 불펜 과부하가 먼저 터질 가능성이 큽니다.",
                "strengths": ["타선 생산력 양호"],
                "weaknesses": ["불펜 과부하"],
                "risks": [{"area": "불펜", "level": 0, "description": "리그 평균 대비 7%p 높은 불펜 비중"}],
                "why_it_matters": ["불펜 소모는 접전 후반 카드 선택 폭을 줄입니다."],
                "swing_factors": ["선발이 5회를 버티는지가 핵심입니다."],
                "watch_points": ["초반 출루 이후 번트/강공 선택을 봐야 합니다."],
                "uncertainty": ["라인업 발표 전이라 중심 타선 배치는 변수입니다."]
            },
            "detailed_markdown": "## 상세 분석\\n테스트 내용",
            "coach_note": "선발 로테이션 재정비가 시급합니다."
        }"""

        response, error = parse_coach_response(raw)
        assert response is not None
        assert response.headline == "KIA 타이거즈 선발진 안정화 필요"
        assert response.sentiment == "negative"
        assert len(response.key_metrics) == 1
        assert response.key_metrics[0].is_critical is True
        assert (
            response.analysis.verdict
            == "선발 이닝이 짧아지면 불펜 과부하가 먼저 터질 가능성이 큽니다."
        )
        assert response.analysis.why_it_matters == [
            "불펜 소모는 접전 후반 카드 선택 폭을 줄입니다."
        ]
        assert len(response.analysis.risks) == 1
        assert response.analysis.risks[0].level == 0

    def test_parse_stable_trend_is_normalized(self):
        """key_metrics.trend='stable'을 neutral로 정규화"""
        from app.core.coach_validator import parse_coach_response

        raw = """{
            "headline": "KT 위즈 2025시즌 점검",
            "sentiment": "neutral",
            "key_metrics": [
                {"label": "불펜 ERA", "value": "4.21", "status": "warning", "trend": "stable", "is_critical": false}
            ],
            "analysis": {"strengths": [], "weaknesses": [], "risks": []},
            "detailed_markdown": "",
            "coach_note": "불펜 운용은 보합세입니다."
        }"""

        response, error = parse_coach_response(raw)
        assert error is None
        assert response is not None
        assert response.key_metrics[0].trend == "neutral"

    def test_parse_invalid_json(self):
        """잘못된 JSON 파싱 시 None 반환 테스트"""
        from app.core.coach_validator import parse_coach_response

        raw = "이것은 JSON이 아닙니다."
        raw = "이것은 JSON이 아닙니다."
        response, error = parse_coach_response(raw)
        assert response is None
        assert error is not None

    def test_parse_response_sanitizes_control_chars(self):
        """제어문자가 섞인 응답도 정리 후 파싱"""
        from app.core.coach_validator import parse_coach_response_with_meta

        raw = (
            "{\x0b"
            '"headline": "제어문자 테스트", '
            '"sentiment": "neutral", '
            '"key_metrics": [], '
            '"analysis": {"strengths": [], "weaknesses": [], "risks": []}, '
            '"coach_note": "정상 파싱"}'
        )

        response, error, meta = parse_coach_response_with_meta(raw)
        assert error is None
        assert response is not None
        assert response.headline == "제어문자 테스트"
        assert meta["normalization_applied"] is True
        assert "sanitize_control_chars" in meta["normalization_reasons"]

    def test_parse_unquoted_json_keys_are_recovered(self):
        """따옴표 없는 JSON 키도 보정 후 파싱"""
        from app.core.coach_validator import parse_coach_response_with_meta

        raw = """{
            headline: "언쿼티드 키 테스트",
            sentiment: "neutral",
            key_metrics: [
                {label: "OPS", value: "0.812", status: "good", trend: "up", is_critical: true}
            ],
            analysis: {strengths: [], weaknesses: [], risks: []},
            coach_note: "정상 파싱"
        }"""

        response, error, meta = parse_coach_response_with_meta(raw)
        assert error is None
        assert response is not None
        assert response.headline == "언쿼티드 키 테스트"
        assert meta["normalization_applied"] is True
        assert "quote_unquoted_json_keys" in meta["normalization_reasons"]

    def test_extract_json_from_prose_then_object(self):
        """설명 문구 뒤 첫 JSON 객체를 salvage"""
        from app.core.coach_validator import extract_json_from_response

        raw = (
            "분석 요약입니다. 아래 JSON만 사용하세요.\n"
            '{"headline": "살베지 테스트", "sentiment": "neutral", "key_metrics": []}\n'
            "추가 설명은 무시합니다."
        )

        result = extract_json_from_response(raw)
        assert result is not None
        data = json.loads(result)
        assert data["headline"] == "살베지 테스트"

    def test_parse_missing_headline_is_auto_recovered(self):
        """headline 누락 시 자동 복구 후 파싱 성공"""
        from app.core.coach_validator import parse_coach_response_with_meta

        raw = """{
            "sentiment": "neutral",
            "key_metrics": [
                {"label": "OPS", "value": "0.812", "status": "good", "is_critical": true}
            ],
            "analysis": {
                "strengths": ["타선의 출루율이 꾸준히 유지됨"],
                "weaknesses": [],
                "risks": []
            },
            "coach_note": "중심 타선 유지가 필요합니다."
        }"""

        response, error, meta = parse_coach_response_with_meta(raw)
        assert error is None
        assert response is not None
        assert response.headline != ""
        assert meta["normalization_applied"] is True
        assert "derive_headline" in meta["normalization_reasons"]

    def test_parse_string_key_metrics_are_coerced_to_objects(self):
        """문자열 key_metrics 배열을 구조화 지표로 복구"""
        from app.core.coach_validator import parse_coach_response_with_meta

        raw = """{
            "headline": "문자열 지표 테스트",
            "sentiment": "neutral",
            "key_metrics": [
                "최근 흐름 4승 1패",
                "최대 WPA 변동 +18.4%p",
                "불펜 비중 0%"
            ],
            "analysis": {"strengths": [], "weaknesses": [], "risks": []},
            "coach_note": "핵심 지표를 복구합니다."
        }"""

        response, error, meta = parse_coach_response_with_meta(raw)
        assert error is None
        assert response is not None
        assert response.key_metrics[0].label == "최근 흐름"
        assert response.key_metrics[0].value == "4승 1패"
        assert response.key_metrics[1].label == "최대 WPA 변동"
        assert response.key_metrics[1].value == "+18.4%p"
        assert response.key_metrics[1].trend == "up"
        assert response.key_metrics[2].label == "불펜 비중"
        assert response.key_metrics[2].value == "0%"
        assert any(
            reason.startswith("coerce_key_metric_0_from_string")
            for reason in meta["normalization_reasons"]
        )

    def test_parse_key_metric_value_is_truncated_before_validation(self):
        """key_metrics.value가 50자 초과 시 사전 절단"""
        from app.core.coach_validator import parse_coach_response_with_meta

        long_value = "x" * 80
        raw = f"""{{
            "headline": "테스트 헤드라인",
            "sentiment": "neutral",
            "key_metrics": [
                {{"label": "장문지표", "value": "{long_value}", "status": "warning", "is_critical": false}}
            ],
            "analysis": {{"strengths": [], "weaknesses": [], "risks": []}},
            "coach_note": "테스트 노트입니다."
        }}"""

        response, error, meta = parse_coach_response_with_meta(raw)
        assert error is None
        assert response is not None
        assert len(response.key_metrics[0].value) <= 50
        assert response.key_metrics[0].value.endswith("...")
        assert meta["normalization_applied"] is True
        assert any(
            reason.startswith("truncate_key_metrics_value_")
            for reason in meta["normalization_reasons"]
        )

    def test_parse_is_critical_is_capped_to_two(self):
        """is_critical=true가 3개 이상이면 2개로 보정"""
        from app.core.coach_validator import parse_coach_response_with_meta

        raw = """{
            "headline": "테스트 헤드라인",
            "sentiment": "neutral",
            "key_metrics": [
                {"label": "지표1", "value": "1", "status": "danger", "is_critical": true},
                {"label": "지표2", "value": "2", "status": "danger", "is_critical": true},
                {"label": "지표3", "value": "3", "status": "danger", "is_critical": true}
            ],
            "analysis": {"strengths": [], "weaknesses": [], "risks": []},
            "coach_note": "테스트 노트입니다."
        }"""

        response, error, meta = parse_coach_response_with_meta(raw)
        assert error is None
        assert response is not None
        critical_count = sum(1 for metric in response.key_metrics if metric.is_critical)
        assert critical_count == 2
        assert "reduce_is_critical_3_to_2" in meta["normalization_reasons"]

    def test_parse_out_of_range_risk_level_is_coerced(self):
        """analysis.risks[].level이 범위를 벗어나면 주의(1)로 보정"""
        from app.core.coach_validator import parse_coach_response_with_meta

        raw = """{
            "headline": "리스크 레벨 보정 테스트",
            "sentiment": "neutral",
            "key_metrics": [],
            "analysis": {
                "strengths": [],
                "weaknesses": [],
                "risks": [{"area": "불펜", "level": 3, "description": "과부하 위험"}]
            },
            "coach_note": "불펜 관리가 필요합니다."
        }"""

        response, error, meta = parse_coach_response_with_meta(raw)
        assert error is None
        assert response is not None
        assert response.analysis.risks[0].level == 1
        assert "coerce_risk_level_0_to_1" in meta["normalization_reasons"]

    def test_parse_non_recoverable_shape_still_fails(self):
        """정규화로 복구 불가능한 형태는 기존처럼 실패"""
        from app.core.coach_validator import parse_coach_response

        raw = """{
            "headline": "테스트",
            "sentiment": "neutral",
            "key_metrics": [],
            "analysis": "invalid_type",
            "coach_note": "테스트 노트입니다."
        }"""

        response, error = parse_coach_response(raw)
        assert response is None
        assert error is not None

    def test_validate_coach_response_warnings(self):
        """Coach 응답 검증 경고 테스트"""
        from app.core.coach_validator import (
            CoachResponse,
            validate_coach_response,
            KeyMetric,
            AnalysisSection,
        )

        # 핵심 지표가 4개 이상인 경우 경고
        response = CoachResponse(
            headline="테스트 헤드라인",
            sentiment="neutral",
            key_metrics=[
                KeyMetric(label=f"지표{i}", value="1", status="양호", is_critical=True)
                for i in range(4)
            ],
            analysis=AnalysisSection(),
            coach_note="짧음",  # 20자 미만
        )

        warnings = validate_coach_response(response)
        assert len(warnings) >= 2  # 핵심 지표 초과 + coach_note 짧음

    def test_format_coach_response_as_markdown(self):
        """Coach 응답 마크다운 변환 테스트"""
        from app.core.coach_validator import (
            CoachResponse,
            format_coach_response_as_markdown,
            KeyMetric,
            AnalysisSection,
            RiskItem,
        )

        response = CoachResponse(
            headline="팀 상태 양호",
            sentiment="positive",
            key_metrics=[
                KeyMetric(
                    label="OPS",
                    value=".850",
                    status="양호",
                    trend="up",
                    is_critical=True,
                )
            ],
            analysis=AnalysisSection(
                summary="타격과 불펜 흐름이 동시에 살아 있습니다.",
                verdict="공격 우세는 확인되지만 후반 불펜 소모 관리가 남아 있습니다.",
                strengths=["강력한 타선"],
                weaknesses=["불펜 불안"],
                risks=[RiskItem(area="불펜", level=1, description="주의 필요")],
                why_it_matters=["OPS 우위가 선취점 확률을 끌어올립니다."],
                swing_factors=["7회 전후 불펜 카드가 승부처입니다."],
                watch_points=["상위 타순 두 번째 순환을 체크해야 합니다."],
                uncertainty=["라인업 변경 가능성은 남아 있습니다."],
            ),
            detailed_markdown="## 상세 분석\n테스트 내용",
            coach_note="지속적인 모니터링을 권장합니다.",
        )

        markdown = format_coach_response_as_markdown(response)
        assert "🟢" in markdown  # positive sentiment
        assert "OPS" in markdown
        assert "강력한 타선" in markdown
        assert "코치 판정" in markdown
        assert "왜 중요한가" in markdown
        assert "Coach's Note" in markdown

    def test_analyze_request_strips_whitespace_question_override(self):
        """manual 모드의 question_override 앞뒤 공백이 정리된다."""
        from app.routers.coach import AnalyzeRequest

        payload = AnalyzeRequest(
            home_team_id="HH",
            request_mode="manual_detail",
            question_override="  오늘 경기 핵심을 정리해줘  ",
            starter_signature="  starter:abc  ",
            lineup_signature="  lineup:def  ",
            expected_cache_key="  cache123  ",
        )

        assert payload.question_override == "오늘 경기 핵심을 정리해줘"
        assert payload.starter_signature == "starter:abc"
        assert payload.lineup_signature == "lineup:def"
        assert payload.expected_cache_key == "cache123"

    def test_analyze_request_auto_mode_rejects_question_override(self):
        """auto_brief 모드에서 question_override는 검증 예외가 발생한다."""
        from app.routers.coach import AnalyzeRequest

        with pytest.raises(ValidationError):
            AnalyzeRequest(
                home_team_id="HH",
                request_mode="auto_brief",
                question_override="강제로 설정하려고 한 질문",
            )

    def test_analyze_request_rejects_unknown_request_mode(self):
        """정의되지 않은 request_mode는 검증 예외가 발생한다."""
        from app.routers.coach import AnalyzeRequest

        with pytest.raises(ValidationError):
            AnalyzeRequest(
                home_team_id="HH",
                request_mode="experimental",
            )


# ============================================================
# Coach Error Masking Helper Tests
# ============================================================


class TestCoachErrorMaskingHelpers:
    """Coach 에러 마스킹 헬퍼 테스트"""

    def test_sanitize_cache_error_code_allows_known_codes(self):
        from app.routers.coach import (
            COACH_DATA_INSUFFICIENT_CODE,
            COACH_LLM_TIMEOUT_ERROR_CODE,
            COACH_STREAM_CANCELLED_ERROR_CODE,
            _sanitize_cache_error_code,
        )

        assert _sanitize_cache_error_code(COACH_DATA_INSUFFICIENT_CODE) == (
            COACH_DATA_INSUFFICIENT_CODE
        )
        assert _sanitize_cache_error_code(COACH_STREAM_CANCELLED_ERROR_CODE) == (
            COACH_STREAM_CANCELLED_ERROR_CODE
        )
        assert _sanitize_cache_error_code(COACH_LLM_TIMEOUT_ERROR_CODE) == (
            COACH_LLM_TIMEOUT_ERROR_CODE
        )

    def test_sanitize_cache_error_code_falls_back_for_unknown(self):
        from app.routers.coach import (
            COACH_INTERNAL_ERROR_CODE,
            _sanitize_cache_error_code,
        )

        assert _sanitize_cache_error_code("db_timeout_with_internal_trace") == (
            COACH_INTERNAL_ERROR_CODE
        )
        assert _sanitize_cache_error_code(None) == COACH_INTERNAL_ERROR_CODE

    def test_cache_error_message_for_user_uses_safe_mapping(self):
        from app.routers.coach import (
            COACH_INTERNAL_ERROR_CODE,
            _cache_error_message_for_user,
        )

        message = _cache_error_message_for_user("db_timeout_with_internal_trace")
        assert message
        assert "db_timeout_with_internal_trace" not in message
        assert message == _cache_error_message_for_user(COACH_INTERNAL_ERROR_CODE)

    def test_cache_error_code_from_exception_identifies_coach_llm_timeout(self):
        from app.routers.coach import (
            COACH_LLM_TIMEOUT_ERROR_CODE,
            _cache_error_code_from_exception,
        )

        assert (
            _cache_error_code_from_exception(
                TimeoutError("coach_llm_stream_timeout_after_18s")
            )
            == COACH_LLM_TIMEOUT_ERROR_CODE
        )

    def test_coach_public_error_payload_masks_unknown_code(self):
        from app.routers.coach import (
            COACH_INTERNAL_ERROR_CODE,
            _coach_public_error_payload,
        )

        payload = _coach_public_error_payload("internal_stacktrace_xxx")
        assert payload["code"] == COACH_INTERNAL_ERROR_CODE
        assert payload["message"]
        assert "internal_stacktrace_xxx" not in payload["message"]

    def test_coach_public_error_payload_exposes_safe_known_message(self):
        from app.routers.coach import (
            COACH_DATA_INSUFFICIENT_CODE,
            _coach_public_error_payload,
        )

        payload = _coach_public_error_payload(COACH_DATA_INSUFFICIENT_CODE)
        assert payload["code"] == COACH_DATA_INSUFFICIENT_CODE
        assert payload["message"] == "분석에 필요한 데이터가 충분하지 않습니다."


# ============================================================
# Coach Evidence Helpers Tests
# ============================================================
