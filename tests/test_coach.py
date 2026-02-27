"""
Coach 기능 테스트 모듈.

Fast Path, 캐싱, 응답 검증기 등을 테스트합니다.
"""

import pytest
import json
import asyncio
from pydantic import ValidationError

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
                "strengths": ["타선 생산력 양호"],
                "weaknesses": ["불펜 과부하"],
                "risks": [{"area": "불펜", "level": 0, "description": "리그 평균 대비 7%p 높은 불펜 비중"}]
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
                strengths=["강력한 타선"],
                weaknesses=["불펜 불안"],
                risks=[RiskItem(area="불펜", level=1, description="주의 필요")],
            ),
            detailed_markdown="## 상세 분석\n테스트 내용",
            coach_note="지속적인 모니터링을 권장합니다.",
        )

        markdown = format_coach_response_as_markdown(response)
        assert "🟢" in markdown  # positive sentiment
        assert "OPS" in markdown
        assert "강력한 타선" in markdown
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


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
