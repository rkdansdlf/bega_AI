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
            import asyncio
            from app.agents.tool_caller import ToolCaller, ToolCall

            caller = ToolCaller()
            call_times = []

            # 시작 시간을 기록하는 추적 도구 (native async — await 지점에서 진짜 동시 실행)
            async def tracking_tool_1(param1: str) -> dict:
                call_times.append(("tool_1", time.time()))
                await asyncio.sleep(0.1)  # 비동기 I/O 시뮬레이션
                return {"result": f"tool1: {param1}"}

            async def tracking_tool_2(param2: str) -> dict:
                call_times.append(("tool_2", time.time()))
                await asyncio.sleep(0.1)  # 비동기 I/O 시뮬레이션
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
