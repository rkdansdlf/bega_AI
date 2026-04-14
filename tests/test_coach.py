"""
Coach 기능 테스트 모듈.

Fast Path, 캐싱, 응답 검증기 등을 테스트합니다.
"""

import pytest
import json
import asyncio
import logging
from pydantic import ValidationError


def _build_game_evidence(**overrides):
    from app.routers.coach import GameEvidence

    base = dict(
        season_year=2025,
        home_team_code="LG",
        away_team_code="KT",
        home_team_name="LG 트윈스",
        away_team_name="KT 위즈",
        game_id="202503120001",
        season_id=20250,
        game_date="2025-03-12",
        game_status="SCHEDULED",
        game_status_bucket="SCHEDULED",
        league_type_code=0,
        stage_label="REGULAR",
        round_display="정규시즌",
        home_pitcher="임찬규",
        away_pitcher="쿠에바스",
        lineup_announced=True,
        home_lineup=["홍창기", "박해민"],
        away_lineup=["강백호", "로하스"],
        summary_items=["결승타 홍창기"],
        stadium_name="잠실",
        start_time="18:30",
        weather="맑음",
    )
    base.update(overrides)
    return GameEvidence(**base)


# ============================================================
# Coach Validator Tests
# ============================================================


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
            COACH_STREAM_CANCELLED_ERROR_CODE,
            _sanitize_cache_error_code,
        )

        assert _sanitize_cache_error_code(COACH_DATA_INSUFFICIENT_CODE) == (
            COACH_DATA_INSUFFICIENT_CODE
        )
        assert _sanitize_cache_error_code(COACH_STREAM_CANCELLED_ERROR_CODE) == (
            COACH_STREAM_CANCELLED_ERROR_CODE
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
            fact_lines=["홈팀 OPS 0.812"],
            caveat_lines=["라인업 미확정"],
            allowed_entity_names={"LG 트윈스"},
            allowed_numeric_tokens={"0.812"},
            supported_fact_count=1,
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

    def test_scheduled_partial_manual_short_circuits_when_starter_and_lineup_are_both_missing(
        self,
    ):
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

        assert payload["generation_mode"] == "deterministic_auto"
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

        assert meta["generation_mode"] == "deterministic_auto"
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


# ============================================================
# TTL Cache Tests
# ============================================================


class TestTTLCache:
    """TTL 캐시 테스트"""

    def test_cache_set_and_get(self):
        """캐시 설정 및 조회 테스트"""
        from app.tools.database_query import TTLCache

        cache = TTLCache(ttl_seconds=60)
        cache.set("test_key", {"data": "value"})

        result = cache.get("test_key")
        assert result is not None
        assert result["data"] == "value"

    def test_cache_miss(self):
        """캐시 미스 테스트"""
        from app.tools.database_query import TTLCache

        cache = TTLCache(ttl_seconds=60)
        result = cache.get("nonexistent_key")
        assert result is None

    def test_cache_expiry(self):
        """캐시 만료 테스트"""
        import time
        from app.tools.database_query import TTLCache

        cache = TTLCache(ttl_seconds=1)  # 1초 TTL
        cache.set("expire_key", "value")

        # 즉시 조회 - 존재해야 함
        assert cache.get("expire_key") == "value"

        # 1.5초 대기 후 조회 - 만료됨
        time.sleep(1.5)
        assert cache.get("expire_key") is None

    def test_cache_max_size(self):
        """캐시 최대 크기 테스트"""
        from app.tools.database_query import TTLCache

        cache = TTLCache(ttl_seconds=60, max_size=3)

        # 4개 항목 추가 (최대 크기 초과)
        for i in range(4):
            cache.set(f"key_{i}", f"value_{i}")

        # 최대 3개만 유지
        stats = cache.stats()
        assert stats["total_entries"] <= 3

    def test_cache_stats(self):
        """캐시 통계 테스트"""
        from app.tools.database_query import TTLCache

        cache = TTLCache(ttl_seconds=60, max_size=10)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.stats()
        assert stats["total_entries"] == 2
        assert stats["ttl_seconds"] == 60
        assert stats["max_size"] == 10


# ============================================================
# Tool Caller Parallel Execution Tests
# ============================================================


class TestToolCallerParallel:
    """병렬 도구 실행 테스트"""

    def test_execute_multiple_tools_parallel(self):
        """
        병렬 도구 실행 테스트

        [P3 Fix] 시간 기반 검증 대신 Mock 기반 동시성 검증으로 수정.
        두 도구가 거의 동시에 시작되었는지 확인하여 CI 환경에서도 안정적으로 통과.
        """

        async def _async_test():
            import time
            import threading
            from app.agents.tool_caller import ToolCaller, ToolCall

            caller = ToolCaller()
            call_times = []
            call_lock = threading.Lock()

            # 시작 시간을 기록하는 추적 도구
            def tracking_tool_1(param1: str) -> dict:
                with call_lock:
                    call_times.append(("tool_1", time.time()))
                time.sleep(0.1)  # 작업 시뮬레이션
                return {"result": f"tool1: {param1}"}

            def tracking_tool_2(param2: str) -> dict:
                with call_lock:
                    call_times.append(("tool_2", time.time()))
                time.sleep(0.1)  # 작업 시뮬레이션
                return {"result": f"tool2: {param2}"}

            caller.register_tool(
                "tracking_tool_1",
                "Tracking tool 1",
                {"param1": "Parameter 1"},
                tracking_tool_1,
            )
            caller.register_tool(
                "tracking_tool_2",
                "Tracking tool 2",
                {"param2": "Parameter 2"},
                tracking_tool_2,
            )

            # 병렬 실행
            tool_calls = [
                ToolCall("tracking_tool_1", {"param1": "test1"}),
                ToolCall("tracking_tool_2", {"param2": "test2"}),
            ]

            results = await caller.execute_multiple_tools_parallel(tool_calls)

            # 검증: 두 도구 모두 호출됨
            assert len(call_times) == 2
            assert len(results) == 2
            assert all(r.success for r in results)

            # [P3 Fix] 동시성 검증: 두 도구가 거의 동시에 시작되었는지 확인
            # 순차 실행이면 0.1초 이상 차이, 병렬이면 0.05초 이내
            time_diff = abs(call_times[0][1] - call_times[1][1])
            assert (
                time_diff < 0.05
            ), f"Tools not started concurrently: {time_diff:.3f}s apart"

        asyncio.run(_async_test())

    def test_parallel_execution_with_failure(self):
        """병렬 실행 중 일부 실패 처리 테스트"""

        async def _async_test():
            from app.agents.tool_caller import ToolCaller, ToolCall

            caller = ToolCaller()

            def success_tool() -> dict:
                return {"status": "ok"}

            def fail_tool() -> dict:
                raise ValueError("Intentional failure")

            caller.register_tool("success_tool", "Success", {}, success_tool)
            caller.register_tool("fail_tool", "Fail", {}, fail_tool)

            tool_calls = [
                ToolCall("success_tool", {}),
                ToolCall("fail_tool", {}),
            ]

            results = await caller.execute_multiple_tools_parallel(tool_calls)

            assert results[0].success is True
            assert results[1].success is False

        asyncio.run(_async_test())


# ============================================================
# Coach Fast Path Integration Tests
# ============================================================


class TestCoachFastPath:
    """Coach Fast Path 통합 테스트"""

    def test_cache_prompt_version_fits_cache_column(self):
        from app.core.coach_cache_contract import COACH_CACHE_PROMPT_VERSION

        assert len(COACH_CACHE_PROMPT_VERSION) <= 32

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
            == "SSG 랜더스가 최근 전력과 팀 타격 생산성에서 앞섭니다."
        )
        assert (
            payload["analysis"]["why_it_matters"][0]
            == "SSG 랜더스가 최근 전력과 팀 OPS를 함께 앞세워 초중반 주도권을 먼저 잡을 가능성이 있습니다."
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
            "## 코치 판단\n- SSG 랜더스가 최근 전력과 팀 타격 생산성에서 앞섭니다."
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
                    def __enter__(self_inner):
                        return _FakeConn()

                    def __exit__(self_inner, exc_type, exc, tb):
                        return False

                return _Ctx()

        class _FakeDatabaseQueryTool:
            def __init__(self, _conn):
                pass

            def get_team_summary(self, *_args, **_kwargs):
                return {"found": False}

            def get_team_advanced_metrics(self, *_args, **_kwargs):
                return {"found": False}

            def get_team_player_form_signals(self, *_args, **_kwargs):
                return {"found": False, "batters": [], "pitchers": []}

            def get_team_recent_form(self, *_args, **_kwargs):
                return {"found": False}

        class _FakeGameQueryTool:
            def __init__(self, _conn):
                pass

            def get_head_to_head(self, *_args, **kwargs):
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
            == "불펜 비중이 공개되지 않아 접전 후반 상황 처리 능력은 비교가 어렵다."
        )
        assert (
            polished["analysis"]["verdict"]
            == "핵심 변수는 불펜 비중이 공개되지 않아 경기 후반 변수 확인이 어렵습니다."
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
            "- 두 팀 모두 불펜 운용 데이터가 부족해, 불펜전의 중요성이 더욱 부각되고 있습니다."
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
        assert (
            "## 상대 전적\n- 상대 전적 근거가 충분하지 않아 직접 비교는 보수적으로 해석해야 합니다."
            in markdown
        )
        assert "## 타격 생산성\n- NC 다이노스 0.799 / 한화 이글스 0.783" in markdown

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
        assert "## 상대 전적\n- 상대 전적 표본이 부족합니다." in markdown
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
        assert "## 상대 전적\n- 상대 전적 표본이 부족합니다." in markdown
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

        manual_request = _build_manual_data_request(
            None,
            payload,
            evidence,
            assess_game_evidence(evidence),
        )

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

        manual_request = _build_manual_data_request(
            None,
            payload,
            evidence,
            assess_game_evidence(evidence),
        )

        assert manual_request is None

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


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
