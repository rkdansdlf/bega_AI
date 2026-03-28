"""
도구 호출을 관리하는 시스템입니다.

이 모듈은 LLM이 실제 도구들을 안전하게 호출할 수 있도록
인터페이스를 제공하고 실행 결과를 관리합니다.
"""

import logging
import re
import asyncio
from datetime import datetime
from types import MappingProxyType
from typing import Dict, List, Any, Callable, Optional, Iterable, Mapping
from dataclasses import dataclass
import inspect

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """도구 호출 요청을 나타내는 클래스"""

    tool_name: str
    parameters: Dict[str, Any]

    def __str__(self):
        return f"ToolCall(tool_name='{self.tool_name}', parameters={self.parameters})"

    def to_dict(self) -> Dict[str, Any]:
        """JSON serialization을 위한 딕셔너리 변환"""
        return {"tool_name": self.tool_name, "parameters": self.parameters}


@dataclass
class ToolResult:
    """도구 실행 결과를 나타내는 클래스"""

    success: bool
    data: Any
    message: str

    def __str__(self):
        return f"ToolResult(success={self.success}, message='{self.message}')"

    def to_dict(self) -> Dict[str, Any]:
        """JSON serialization을 위한 딕셔너리 변환"""
        return {"success": self.success, "data": self.data, "message": self.message}


@dataclass(frozen=True)
class ToolDefinition:
    """공유 가능한 불변 도구 정의."""

    tool_name: str
    description: str
    parameters_schema: Mapping[str, str]
    handler_attr: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters_schema",
            MappingProxyType(dict(self.parameters_schema)),
        )


class ToolCaller:
    """
    도구 호출을 관리하는 클래스

    이 클래스는 다음과 같은 역할을 합니다:
    1. 사용 가능한 도구들을 등록하고 관리
    2. 도구 호출 요청을 안전하게 실행
    3. 실행 결과를 표준화된 형태로 반환
    4. 오류 처리 및 로깅
    """

    def __init__(
        self,
        *,
        tools: Optional[Dict[str, Dict[str, Any]]] = None,
        tool_functions: Optional[Dict[str, Callable]] = None,
        handler_attrs: Optional[Mapping[str, str]] = None,
        bound_target: Any = None,
        tool_descriptions: Optional[str] = None,
    ):
        self.tools: Dict[str, Dict[str, Any]] = tools or {}
        self._tool_functions: Dict[str, Callable] = tool_functions or {}
        self._handler_attrs: Dict[str, str] = dict(handler_attrs or {})
        self._bound_target = bound_target
        self._tool_descriptions = tool_descriptions
        if not self._tool_functions and not self._handler_attrs:
            for tool_name, tool_info in self.tools.items():
                function = tool_info.get("function")
                if callable(function):
                    self._tool_functions[tool_name] = function

    @classmethod
    def from_definitions(
        cls,
        definitions: Iterable[ToolDefinition],
        handler_resolver: Callable[[str], Callable] | None = None,
        *,
        bound_target: Any = None,
        tool_descriptions: Optional[str] = None,
    ) -> "ToolCaller":
        definition_items = tuple(definitions)
        shared_tools: Dict[str, Dict[str, Any]] = {}
        tool_functions: Dict[str, Callable] = {}
        handler_attrs: Dict[str, str] = {}

        for definition in definition_items:
            shared_tools[definition.tool_name] = {
                "description": definition.description,
                "parameters_schema": definition.parameters_schema,
            }
            handler_attrs[definition.tool_name] = definition.handler_attr
            if handler_resolver is not None:
                tool_functions[definition.tool_name] = handler_resolver(
                    definition.handler_attr
                )

        return cls(
            tools=shared_tools,
            tool_functions=tool_functions or None,
            handler_attrs=handler_attrs,
            bound_target=bound_target,
            tool_descriptions=tool_descriptions
            or cls.describe_definitions(definition_items),
        )

    def bind(self, bound_target: Any) -> "ToolCaller":
        return ToolCaller(
            tools=self.tools,
            handler_attrs=self._handler_attrs,
            bound_target=bound_target,
            tool_descriptions=self._tool_descriptions,
        )

    @staticmethod
    def describe_definitions(definitions: Iterable[ToolDefinition]) -> str:
        descriptions = []
        for definition in definitions:
            descriptions.append(
                f"**{definition.tool_name}**: {definition.description}"
            )
            if definition.parameters_schema:
                param_lines = []
                for param_name, param_desc in definition.parameters_schema.items():
                    param_lines.append(f"  - {param_name}: {param_desc}")
                descriptions.append("\n".join(param_lines))
            descriptions.append("")
        return "\n".join(descriptions)

    @staticmethod
    def _coerce_year(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            match = re.search(r"\d{4}", value)
            if match:
                return int(match.group(0))
        return None

    @classmethod
    def _default_tool_value(cls, tool_name: str, param_name: str) -> Any | None:
        year_defaults = {
            "get_leaderboard",
            "get_team_summary",
            "get_team_advanced_metrics",
            "get_team_rank",
            "get_team_by_rank",
            "get_team_last_game",
            "get_korean_series_winner",
            "get_award_winners",
            "get_team_basic_info",
            "get_player_stats",
            "get_career_stats",
            "validate_player",
        }

        if param_name == "year" and tool_name in year_defaults:
            return datetime.now().year
        if param_name == "position" and tool_name == "get_leaderboard":
            return "batting"
        if param_name == "comparison_type" and tool_name == "compare_players":
            return "career"
        if param_name == "stat_name" and tool_name == "get_leaderboard":
            return "ops"
        if param_name == "limit" and tool_name == "get_leaderboard":
            return 10
        return None

    def _normalize_parameters(
        self, tool_name: str, tool_function: Callable, parameters: Dict[str, Any]
    ) -> tuple[Dict[str, Any], set[str], set[str]]:
        sig = inspect.signature(tool_function)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        # 불필요한 인자 제거 (LLM이 임의 키를 넣는 케이스 방지)
        if accepts_kwargs:
            normalized = dict(parameters)
        else:
            normalized = {
                key: value for key, value in parameters.items() if key in sig.parameters
            }
        dropped = set(parameters.keys()) - set(normalized.keys())

        # 누락 필수/필수 유사 파라미터 채우기
        missing_required: set[str] = set()
        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            if name not in normalized:
                fallback = self._default_tool_value(tool_name, name)
                if fallback is not None:
                    normalized[name] = fallback
                elif param.default == inspect.Parameter.empty:
                    missing_required.add(name)

        # year 문자열 보정("2025년", 공백 등)
        if "year" in normalized:
            coerced_year = self._coerce_year(normalized["year"])
            if coerced_year is not None:
                normalized["year"] = coerced_year
            elif normalized.get("year") is None:
                normalized["year"] = datetime.now().year

        # limit 정수 정규화
        if "limit" in normalized and isinstance(normalized["limit"], str):
            stripped = normalized["limit"].strip()
            if stripped.isdigit():
                normalized["limit"] = int(stripped)

        return normalized, missing_required, dropped

    def register_tool(
        self,
        tool_name: str,
        description: str,
        parameters_schema: Dict[str, str],
        function: Callable,
    ) -> None:
        """
        새로운 도구를 등록합니다.

        Args:
            tool_name: 도구 이름
            description: 도구 설명
            parameters_schema: 매개변수 스키마 {param_name: description}
            function: 실제 실행할 함수
        """
        self.tools[tool_name] = {
            "description": description,
            "parameters_schema": parameters_schema,
            "function": function,
        }
        self._tool_functions[tool_name] = function
        self._handler_attrs[tool_name] = getattr(function, "__name__", tool_name)
        self._tool_descriptions = None
        logger.debug(f"[ToolCaller] Registered tool: {tool_name}")

    def _resolve_tool_function(self, tool_name: str) -> Optional[Callable]:
        tool_function = self._tool_functions.get(tool_name)
        if tool_function is not None:
            return tool_function

        handler_attr = self._handler_attrs.get(tool_name)
        if handler_attr and self._bound_target is not None:
            return getattr(self._bound_target, handler_attr, None)

        return None

    def get_tool_descriptions(self) -> str:
        """등록된 모든 도구들의 설명을 반환합니다."""
        if self._tool_descriptions is None:
            descriptions = []
            for tool_name, tool_info in self.tools.items():
                descriptions.append(f"**{tool_name}**: {tool_info['description']}")
                if tool_info["parameters_schema"]:
                    param_lines = []
                    for param_name, param_desc in tool_info["parameters_schema"].items():
                        param_lines.append(f"  - {param_name}: {param_desc}")
                    descriptions.append("\n".join(param_lines))
                descriptions.append("")
            self._tool_descriptions = "\n".join(descriptions)
        return self._tool_descriptions

    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        도구 호출을 실행합니다.

        Args:
            tool_call: 실행할 도구 호출 정보

        Returns:
            도구 실행 결과
        """
        logger.info(f"[ToolCaller] Executing: {tool_call}")

        # 도구 존재 여부 확인
        if tool_call.tool_name not in self.tools:
            error_msg = f"존재하지 않는 도구입니다: {tool_call.tool_name}"
            logger.error(f"[ToolCaller] {error_msg}")
            return ToolResult(success=False, data={}, message=error_msg)

        tool_info = self.tools[tool_call.tool_name]
        tool_function = self._resolve_tool_function(tool_call.tool_name)
        if tool_function is None:
            error_msg = f"실행 함수가 연결되지 않았습니다: {tool_call.tool_name}"
            logger.error(f"[ToolCaller] {error_msg}")
            return ToolResult(success=False, data={}, message=error_msg)

        try:
            normalized_parameters, missing_required, dropped_params = (
                self._normalize_parameters(
                    tool_call.tool_name, tool_function, tool_call.parameters
                )
            )

            if dropped_params:
                logger.warning(
                    "[ToolCaller] Dropped unsupported parameters: "
                    f"{tool_call.tool_name}: {sorted(dropped_params)}"
                )

            if missing_required:
                logger.warning(
                    f"[ToolCaller] Missing REQUIRED parameters: {missing_required}"
                )
                return ToolResult(
                    success=False,
                    data={},
                    message=f"매개변수 누락: {', '.join(sorted(missing_required))}",
                )

            # 도구 함수 실행
            result = tool_function(**normalized_parameters)

            # 결과 타입 확인
            if isinstance(result, ToolResult):
                logger.info(
                    f"[ToolCaller] Tool executed successfully: {tool_call.tool_name}"
                )
                return result
            else:
                # 일반 함수 결과를 ToolResult로 래핑
                logger.info(
                    f"[ToolCaller] Tool executed, wrapping result: {tool_call.tool_name}"
                )
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"{tool_call.tool_name} 실행 완료",
                )

        except TypeError as e:
            error_msg = f"매개변수 오류: {e}"
            logger.error(f"[ToolCaller] Parameter error for {tool_call.tool_name}: {e}")
            return ToolResult(success=False, data={}, message=error_msg)
        except Exception as e:
            error_msg = f"도구 실행 중 오류: {e}"
            logger.error(f"[ToolCaller] Execution error for {tool_call.tool_name}: {e}")
            return ToolResult(success=False, data={}, message=error_msg)

    def execute_multiple_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        여러 도구를 순차적으로 실행합니다.

        Args:
            tool_calls: 실행할 도구 호출 목록

        Returns:
            각 도구의 실행 결과 목록
        """
        logger.info(f"[ToolCaller] Executing {len(tool_calls)} tools")

        results = []
        for i, tool_call in enumerate(tool_calls):
            logger.info(
                f"[ToolCaller] Executing tool {i + 1}/{len(tool_calls)}: {tool_call.tool_name}"
            )
            result = self.execute_tool(tool_call)
            results.append(result)

            # 실패 시 로깅 (계속 진행)
            if not result.success:
                logger.warning(
                    f"[ToolCaller] Tool {tool_call.tool_name} failed: {result.message}"
                )

        return results

    async def execute_tool_async(self, tool_call: ToolCall) -> ToolResult:
        """단일 도구를 event loop 밖에서 실행합니다."""
        return await asyncio.to_thread(self.execute_tool, tool_call)

    async def execute_multiple_tools_parallel(
        self, tool_calls: List[ToolCall], max_concurrency: int | None = None
    ) -> List[ToolResult]:
        """
        여러 도구를 병렬로 실행합니다.

        독립적인 도구 호출들을 asyncio.gather()로 병렬 실행하여
        총 대기 시간을 줄입니다.

        Args:
            tool_calls: 실행할 도구 호출 목록

        Returns:
            각 도구의 실행 결과 목록 (입력 순서 유지)
        """
        logger.info(
            "[ToolCaller] Executing %d tools in parallel (max_concurrency=%s)",
            len(tool_calls),
            max_concurrency,
        )

        semaphore = (
            asyncio.Semaphore(max_concurrency)
            if isinstance(max_concurrency, int) and max_concurrency > 0
            else None
        )

        async def execute_single_async(tool_call: ToolCall) -> ToolResult:
            """단일 도구를 비동기적으로 실행합니다."""
            if semaphore is None:
                return await self.execute_tool_async(tool_call)

            async with semaphore:
                return await self.execute_tool_async(tool_call)

        # 병렬 실행
        tasks = [execute_single_async(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"[ToolCaller] Parallel execution failed for {tool_calls[i].tool_name}: {result}"
                )
                processed_results.append(
                    ToolResult(
                        success=False, data={}, message=f"병렬 실행 중 오류: {result}"
                    )
                )
            else:
                processed_results.append(result)

        logger.info(
            f"[ToolCaller] Parallel execution completed: "
            f"{sum(1 for r in processed_results if r.success)}/{len(processed_results)} succeeded"
        )

        return processed_results

    def list_available_tools(self) -> List[str]:
        """사용 가능한 도구 목록을 반환합니다."""
        return list(self.tools.keys())

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """특정 도구의 스키마 정보를 반환합니다."""
        if tool_name not in self.tools:
            return None

        return {
            "name": tool_name,
            "description": self.tools[tool_name]["description"],
            "parameters_schema": dict(self.tools[tool_name]["parameters_schema"]),
        }
