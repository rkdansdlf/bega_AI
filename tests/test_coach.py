"""
Coach 기능 테스트 모듈.

Fast Path, 캐싱, 응답 검증기 등을 테스트합니다.
"""

import pytest
import json
import asyncio
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
        )

        assert payload.question_override == "오늘 경기 핵심을 정리해줘"

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
            _sanitize_cache_error_code,
        )

        assert _sanitize_cache_error_code(COACH_DATA_INSUFFICIENT_CODE) == (
            COACH_DATA_INSUFFICIENT_CODE
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
            },
            "matchup": {"summary": {"team1_wins": 5, "team2_wins": 3, "draws": 0}},
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
        assert "0.754" in fact_sheet.allowed_numeric_tokens
        assert (
            "31%" in fact_sheet.allowed_numeric_tokens
            or "31.0%" in fact_sheet.allowed_numeric_tokens
        )

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


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
