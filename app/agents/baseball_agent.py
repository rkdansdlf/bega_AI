"""
야구 통계/지식 응답용 에이전트입니다.

이 에이전트는 환각을 줄이기 위해 실제 DB, 검증된 문서, 최신 검색 결과를
우선 순위에 따라 사용합니다.
"""

import re
import json
import logging
import asyncio
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import psycopg
from decimal import Decimal
from datetime import date, datetime

from ..tools.database_query import DatabaseQueryTool
from ..tools.regulation_query import RegulationQueryTool
from ..tools.game_query import GameQueryTool
from ..tools.document_query import DocumentQueryTool
from ..tools.team_display import replace_team_codes
from ..core.tools.datetime_tool import (
    get_current_datetime,
    get_baseball_season_info,
)  # 신규 도구 임포트
from .chat_intent_router import ChatIntent, ChatIntentRouter, IntentDecision
from .chat_renderers import ChatRendererRegistry
from .tool_caller import ToolCaller, ToolCall, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)
_PLANNER_CACHE_LOCK = RLock()
_PLANNER_CACHE: "OrderedDict[str, tuple[float, Dict[str, Any]]]" = OrderedDict()
_AGENT_RUNTIME_LOCK = RLock()
_AGENT_RUNTIME_COUNTER = 0
_TOOL_DEFINITION_LOCK = RLock()
_DEFAULT_TOOL_DEFINITIONS: tuple["ToolDefinition", ...] | None = None
_REQUEST_CONTEXT: ContextVar["AgentRequestContext | None"] = ContextVar(
    "baseball_agent_request_context",
    default=None,
)
_REQUEST_RESOURCE_ATTRS = frozenset(
    {
        "connection",
        "db_query_tool",
        "regulation_query_tool",
        "game_query_tool",
        "document_query_tool",
        "game_strategist",
        "match_predictor",
    }
)

from ..core.prompts import (
    SYSTEM_PROMPT,
    FOLLOWUP_PROMPT,
    COACH_PROMPT,
    DEFAULT_ANSWER_PROMPT,
    EXPLAINER_ANSWER_PROMPT,
    LATEST_ANSWER_PROMPT,
)  # SYSTEM_PROMPT 임포트
from ..core.entity_extractor import (
    extract_entities_from_query,
    extract_player_names,
    extract_team,
)  # 엔티티 추출 임포트
from ..config import Settings


@dataclass(slots=True)
class AgentRequestContext:
    runtime_id: int
    connection: psycopg.Connection
    db_query_tool: DatabaseQueryTool
    regulation_query_tool: RegulationQueryTool
    game_query_tool: GameQueryTool
    document_query_tool: DocumentQueryTool
    game_strategist: Any
    match_predictor: Any

    @classmethod
    def create(
        cls,
        *,
        runtime_id: int,
        connection: psycopg.Connection,
        settings: Optional[Settings],
    ) -> "AgentRequestContext":
        from ..core.game_strategist import GameStrategist
        from ..core.match_predictor import MatchPredictor

        return cls(
            runtime_id=runtime_id,
            connection=connection,
            db_query_tool=DatabaseQueryTool(connection),
            regulation_query_tool=RegulationQueryTool(connection),
            game_query_tool=GameQueryTool(connection),
            document_query_tool=DocumentQueryTool(
                connection,
                settings=settings,
            ),
            game_strategist=GameStrategist(connection),
            match_predictor=MatchPredictor(connection),
        )


@dataclass(slots=True)
class AgentRequestContextHandle:
    request_context: AgentRequestContext
    previous_request_context: AgentRequestContext | None
    context_count: int


class BaseballAgentRuntime:
    """Shared runtime for a singleton baseball agent plus request-bound contexts."""

    def __init__(
        self,
        *,
        llm_generator,
        settings: Optional[Settings] = None,
        fast_path_enabled: bool = True,
        fast_path_scope: str = "team",
        fast_path_min_messages: int = 1,
        fast_path_tool_cap: int = 2,
        fast_path_fallback_on_empty: bool = True,
        chat_dynamic_token_enabled: bool = True,
        chat_analysis_max_tokens: int = 350,
        chat_answer_max_tokens_short: int = 1400,
        chat_answer_max_tokens_long: int = 2600,
        chat_answer_max_tokens_team: int = 900,
        chat_tool_result_max_chars: int = 2200,
        chat_tool_result_max_items: int = 8,
        chat_first_token_watchdog_seconds: float = 20.0,
        chat_first_token_retry_max_attempts: int = 1,
        chat_stream_first_token_watchdog_seconds: float = 10.0,
        chat_stream_first_token_retry_max_attempts: int = 0,
        tool_definitions: tuple[ToolDefinition, ...] | None = None,
        tool_description_text: str | None = None,
    ) -> None:
        global _AGENT_RUNTIME_COUNTER

        with _AGENT_RUNTIME_LOCK:
            _AGENT_RUNTIME_COUNTER += 1
            self.runtime_id = _AGENT_RUNTIME_COUNTER

        self._lock = RLock()
        self._request_context_count = 0
        self._team_name_cache: Dict[str, str] | None = None
        self.llm_generator = llm_generator
        self.settings = settings
        self.fast_path_enabled = fast_path_enabled
        self.fast_path_scope = fast_path_scope
        self.fast_path_min_messages = max(0, int(fast_path_min_messages))
        self.fast_path_tool_cap = max(1, int(fast_path_tool_cap))
        self.fast_path_fallback_on_empty = fast_path_fallback_on_empty
        self.chat_dynamic_token_enabled = chat_dynamic_token_enabled
        self.chat_analysis_max_tokens = max(64, int(chat_analysis_max_tokens))
        self.chat_answer_max_tokens_short = max(256, int(chat_answer_max_tokens_short))
        self.chat_answer_max_tokens_long = max(
            self.chat_answer_max_tokens_short, int(chat_answer_max_tokens_long)
        )
        requested_team_tokens = int(chat_answer_max_tokens_team)
        self.chat_answer_max_tokens_team = max(
            256,
            min(self.chat_answer_max_tokens_short, requested_team_tokens),
        )
        self.chat_tool_result_max_chars = max(400, int(chat_tool_result_max_chars))
        self.chat_tool_result_max_items = max(1, int(chat_tool_result_max_items))
        self.chat_planner_timeout_seconds = max(
            0.1,
            float(
                getattr(
                    self.settings,
                    "chat_planner_timeout_seconds",
                    DEFAULT_LLM_PLANNER_TIMEOUT_SECONDS,
                )
            ),
        )
        self.chat_compact_planner_timeout_seconds = max(
            0.1,
            float(
                getattr(
                    self.settings,
                    "chat_compact_planner_timeout_seconds",
                    COMPACT_LLM_PLANNER_TIMEOUT_SECONDS,
                )
            ),
        )
        self.chat_first_token_watchdog_seconds = max(
            1.0, float(chat_first_token_watchdog_seconds)
        )
        self.chat_first_token_retry_max_attempts = max(
            0, int(chat_first_token_retry_max_attempts)
        )
        self.chat_stream_first_token_watchdog_seconds = max(
            1.0, float(chat_stream_first_token_watchdog_seconds)
        )
        self.chat_stream_first_token_retry_max_attempts = max(
            0, int(chat_stream_first_token_retry_max_attempts)
        )
        self.chat_tool_parallel_enabled = bool(
            getattr(self.settings, "chat_tool_parallel_enabled", True)
        )
        self.chat_tool_parallel_split_batch_enabled = bool(
            getattr(self.settings, "chat_tool_parallel_split_batch_enabled", True)
        )
        configured_serial_tools = list(
            getattr(self.settings, "chat_tool_parallel_serial_tools", [])
        )
        if configured_serial_tools:
            self.chat_tool_parallel_serial_tools = {
                str(tool_name).strip()
                for tool_name in configured_serial_tools
                if str(tool_name).strip()
            }
        else:
            self.chat_tool_parallel_serial_tools = set(SERIAL_DB_TOOL_NAMES)
        self.chat_tool_parallel_max_concurrency = max(
            1, int(getattr(self.settings, "chat_tool_parallel_max_concurrency", 2))
        )
        self.chat_planner_cache_ttl_seconds = max(
            0, int(getattr(self.settings, "chat_planner_cache_ttl_seconds", 60))
        )
        self.chat_planner_cache_max_entries = max(
            32, int(getattr(self.settings, "chat_planner_cache_max_entries", 512))
        )
        self.tool_definitions = tool_definitions or _get_default_tool_definitions()
        self.tool_description_text = (
            tool_description_text
            if tool_description_text is not None
            else ToolCaller.describe_definitions(self.tool_definitions)
        )
        self.chat_intent_router = ChatIntentRouter(
            fast_path_enabled=self.fast_path_enabled,
            fast_path_scope=self.fast_path_scope,
        )
        self.chat_renderer_registry = ChatRendererRegistry()
        self.tool_caller_factory = ToolCaller.from_definitions(
            self.tool_definitions,
            tool_descriptions=self.tool_description_text,
        )
        self.shared_agent = BaseballStatisticsAgent.__new__(BaseballStatisticsAgent)
        self.shared_agent._apply_shared_runtime(self)

        logger.info(
            "[AgentRuntime] event=created runtime_id=%d provider=%s model=%s",
            self.runtime_id,
            getattr(self.settings, "llm_provider", "unknown"),
            self._resolve_perf_model_name(),
        )
        if self.chat_answer_max_tokens_team != requested_team_tokens:
            logger.info(
                "[AnswerConfig] team_tokens_requested=%d clamped_to=%d short_cap=%d",
                requested_team_tokens,
                self.chat_answer_max_tokens_team,
                self.chat_answer_max_tokens_short,
            )
        logger.info(
            "[PlannerConfig] fast_path_enabled=%s scope=%s min_messages=%d tool_cap=%d fallback_on_empty=%s",
            self.fast_path_enabled,
            self.fast_path_scope,
            self.fast_path_min_messages,
            self.fast_path_tool_cap,
            self.fast_path_fallback_on_empty,
        )
        logger.info(
            "[AnswerConfig] dynamic_token=%s analysis_max=%d answer_short=%d answer_team=%d answer_long=%d tool_chars=%d tool_items=%d",
            self.chat_dynamic_token_enabled,
            self.chat_analysis_max_tokens,
            self.chat_answer_max_tokens_short,
            self.chat_answer_max_tokens_team,
            self.chat_answer_max_tokens_long,
            self.chat_tool_result_max_chars,
            self.chat_tool_result_max_items,
        )
        logger.info(
            "[AnswerWatchdog] default_watchdog_seconds=%.1f default_retry_max_attempts=%d stream_watchdog_seconds=%.1f stream_retry_max_attempts=%d",
            self.chat_first_token_watchdog_seconds,
            self.chat_first_token_retry_max_attempts,
            self.chat_stream_first_token_watchdog_seconds,
            self.chat_stream_first_token_retry_max_attempts,
        )
        logger.info(
            "[ExecutionConfig] tool_parallel_enabled=%s split_batch=%s tool_parallel_max_concurrency=%d serial_tool_count=%d planner_cache_ttl_seconds=%d planner_cache_max_entries=%d",
            self.chat_tool_parallel_enabled,
            self.chat_tool_parallel_split_batch_enabled,
            self.chat_tool_parallel_max_concurrency,
            len(self.chat_tool_parallel_serial_tools),
            self.chat_planner_cache_ttl_seconds,
            self.chat_planner_cache_max_entries,
        )
        logger.info(
            "[ToolRegistry] runtime_id=%d tool_count=%d shared_description_chars=%d",
            self.runtime_id,
            len(self.tool_definitions),
            len(self.tool_description_text),
        )
        logger.info(
            "[PlannerTimeoutConfig] default_timeout=%.1fs compact_timeout=%.1fs",
            self.chat_planner_timeout_seconds,
            self.chat_compact_planner_timeout_seconds,
        )

    def _resolve_perf_model_name(self) -> str:
        if self.settings is None:
            return "unknown"
        if getattr(self.settings, "llm_provider", None) == "gemini":
            return getattr(self.settings, "gemini_model", "unknown")
        return getattr(self.settings, "openrouter_model", "unknown")

    def get_team_name_cache(self) -> Dict[str, str] | None:
        with self._lock:
            return self._team_name_cache

    def set_team_name_cache(self, mapping: Dict[str, str]) -> None:
        with self._lock:
            self._team_name_cache = dict(mapping)

    def enter_request_context(
        self, connection: psycopg.Connection
    ) -> AgentRequestContextHandle:
        with self._lock:
            self._request_context_count += 1
            request_context_count = self._request_context_count

        request_context = AgentRequestContext.create(
            runtime_id=self.runtime_id,
            connection=connection,
            settings=self.settings,
        )
        previous_request_context = _REQUEST_CONTEXT.get()
        _REQUEST_CONTEXT.set(request_context)
        logger.info(
            "[AgentRuntime] event=request_context runtime_id=%d context_count=%d conn_id=%s tool_registry_reused=true tool_count=%d",
            self.runtime_id,
            request_context_count,
            id(connection),
            len(self.tool_definitions),
        )
        return AgentRequestContextHandle(
            request_context=request_context,
            previous_request_context=previous_request_context,
            context_count=request_context_count,
        )

    def exit_request_context(self, handle: AgentRequestContextHandle) -> None:
        current_request_context = _REQUEST_CONTEXT.get()
        if current_request_context is handle.request_context:
            _REQUEST_CONTEXT.set(handle.previous_request_context)
            return

        logger.warning(
            "[AgentRuntime] event=request_context_cleanup_mismatch runtime_id=%d context_count=%d current_runtime_id=%s restore_skipped=true",
            self.runtime_id,
            handle.context_count,
            getattr(current_request_context, "runtime_id", None),
        )

    @contextmanager
    def request_context(self, connection: psycopg.Connection):
        handle = self.enter_request_context(connection)
        try:
            yield handle.request_context
        finally:
            self.exit_request_context(handle)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


class _ToolDefinitionRecorder:
    def __init__(self) -> None:
        self.definitions: list[ToolDefinition] = []

    def register_tool(
        self,
        tool_name: str,
        description: str,
        parameters_schema: Dict[str, str],
        function,
    ) -> None:
        handler_attr = getattr(function, "__name__", "")
        if not handler_attr:
            raise RuntimeError(f"Unable to resolve handler attribute for {tool_name}")
        self.definitions.append(
            ToolDefinition(
                tool_name=tool_name,
                description=description,
                parameters_schema=parameters_schema,
                handler_attr=handler_attr,
            )
        )


def _get_default_tool_definitions() -> tuple[ToolDefinition, ...]:
    global _DEFAULT_TOOL_DEFINITIONS
    with _TOOL_DEFINITION_LOCK:
        if _DEFAULT_TOOL_DEFINITIONS is None:
            _DEFAULT_TOOL_DEFINITIONS = BaseballStatisticsAgent.build_tool_definitions()
    return _DEFAULT_TOOL_DEFINITIONS


def clean_json_response(response: str) -> str:
    """LLM 응답에서 순수 JSON 추출 및 정제"""
    # 코드 블록 제거
    response = re.sub(r"```json\s*", "", response)
    response = re.sub(r"```\s*", "", response)

    # 주석 제거 (// 와 /* */)
    response = re.sub(r"//.*?$", "", response, flags=re.MULTILINE)
    response = re.sub(r"/\*.*?\*/", "", response, flags=re.DOTALL)

    # 후행 쉼표 제거
    response = re.sub(r",(\s*[}\]])", r"\1", response)

    # 이중 중괄호 {{ }} 처리 (LLM 실수 보정)
    # 단순히 replace를 하면 {{}} -> {} 가 되지만,
    # {{{...}}} 같은 경우도 고려해야 하므로,
    # 맨 앞과 맨 뒤의 {{, }} 만 제거하는 것이 안전합니다.
    # 하지만 더 간단하고 강력한 방법은 json.tool 처럼
    # 가장 바깥쪽의 { 와 } 를 찾는 것입니다.
    # 여기서는 간단히 이중 중괄호만 단일 중괄호로 치환합니다.
    # (내부 데이터에 {{ }} 가 있는 경우는 드물다고 가정)
    if response.startswith("{{") and response.endswith("}}"):
        response = response[1:-1]

    return response.strip()


TEAM_CODE_TO_NAME = {
    "KIA": "KIA 타이거즈",
    "HT": "KIA 타이거즈",
    "기아": "KIA 타이거즈",
    "LG": "LG 트윈스",
    "SSG": "SSG 랜더스",
    "SK": "SSG 랜더스",
    "NC": "NC 다이노스",
    "DB": "두산 베어스",
    "DO": "두산 베어스",
    "OB": "두산 베어스",
    "두산": "두산 베어스",
    "KT": "KT 위즈",
    "LT": "롯데 자이언츠",
    "LOT": "롯데 자이언츠",
    "롯데": "롯데 자이언츠",
    "SS": "삼성 라이온즈",
    "삼성": "삼성 라이온즈",
    "HH": "한화 이글스",
    "한화": "한화 이글스",
    "KH": "키움 히어로즈",
    "KI": "키움 히어로즈",
    "WO": "키움 히어로즈",
    "NX": "키움 히어로즈",
    "키움": "키움 히어로즈",
}

INVALID_PLAYER_NAME_TOKENS = {
    "보면",
    "요약",
    "흐름",
    "강점",
    "약점",
    "리스크",
    "페이스",
    "가을야구",
    "플레이오프",
    "분석",
    "시즌",
    "최근",
    "젊은",
}

PLAYER_NAME_NORMALIZATION_SUFFIXES = (
    "묶음으로",
    "묶음은",
    "묶음을",
    "라인으로",
    "라인은",
    "라인을",
    "축으로",
    "축에서",
    "축은",
    "축을",
    "처럼",
    "같은",
    "같이",
    "으로는",
    "으로",
    "이라고",
    "이라",
    "라고",
    "에게는",
    "에게",
    "에서",
    "과",
    "와",
    "을",
    "를",
    "은",
    "는",
    "이",
    "가",
    "도",
    "만",
)

TEAM_LLM_ALLOWED_TOOLS = {
    "get_team_summary",
    "get_team_advanced_metrics",
    "get_team_rank",
    "get_team_last_game",
}

PLAYER_LLM_ALLOWED_TOOLS = {
    "validate_player",
    "get_player_stats",
    "get_career_stats",
}

MULTI_PLAYER_EXPANDABLE_TOOLS = {
    "get_advanced_stats",
    "get_career_stats",
    "get_defensive_stats",
    "get_pitcher_starting_win_rate",
    "get_player_game_performance",
    "get_player_stats",
    "get_player_wpa_stats",
    "get_velocity_data",
    "validate_player",
}

COMPACT_LLM_PLANNER_TOKEN_CAP = 128
COMPACT_LLM_PLANNER_TIMEOUT_SECONDS = 24.0
DEFAULT_LLM_PLANNER_TIMEOUT_SECONDS = 30.0
PLAYER_LLM_MULTI_NAME_CAP = 4
REFERENCE_YEAR_DEFAULT_TOOLS = {
    "get_advanced_stats",
    "get_award_winners",
    "get_defensive_stats",
    "get_korean_series_winner",
    "get_leaderboard",
    "get_pitcher_starting_win_rate",
    "get_player_stats",
    "get_player_wpa_stats",
    "get_team_advanced_metrics",
    "get_team_last_game",
    "get_team_rank",
    "get_team_summary",
    "get_velocity_data",
    "validate_player",
}

SERIAL_DB_TOOL_NAMES = {
    "compare_players",
    "get_advanced_stats",
    "get_award_winners",
    "get_career_stats",
    "get_defensive_stats",
    "get_games_by_date",
    "get_head_to_head",
    "get_korean_series_winner",
    "get_leaderboard",
    "get_pitcher_starting_win_rate",
    "get_player_game_performance",
    "get_player_stats",
    "get_season_final_game_date",
    "get_team_advanced_metrics",
    "get_team_by_rank",
    "get_team_last_game",
    "get_team_rank",
    "get_team_summary",
    "get_velocity_data",
    "get_player_wpa_stats",
    "search_documents",
    "search_regulations",
    "validate_player",
}


class BaseballStatisticsAgent:
    """
    야구 통계 전문 에이전트

    이 에이전트는 다음 원칙을 따릅니다:
    1. 모든 통계 질문은 반드시 실제 DB 도구를 통해 조회
    2. 도구 결과가 없으면 "데이터 없음"으로 명확히 응답
    3. LLM 지식 기반 추측 절대 금지
    4. 검증된 데이터만 사용하여 답변 생성
    """

    @classmethod
    def build_tool_definitions(cls) -> tuple[ToolDefinition, ...]:
        agent = cls.__new__(cls)
        recorder = _ToolDefinitionRecorder()
        agent.tool_caller = recorder
        agent._register_tools()
        return tuple(recorder.definitions)

    def __init__(
        self,
        connection: psycopg.Connection,
        llm_generator,
        settings: Optional[Settings] = None,
        fast_path_enabled: bool = True,
        fast_path_scope: str = "team",
        fast_path_min_messages: int = 1,
        fast_path_tool_cap: int = 2,
        fast_path_fallback_on_empty: bool = True,
        chat_dynamic_token_enabled: bool = True,
        chat_analysis_max_tokens: int = 350,
        chat_answer_max_tokens_short: int = 1400,
        chat_answer_max_tokens_long: int = 2600,
        chat_answer_max_tokens_team: int = 900,
        chat_tool_result_max_chars: int = 2200,
        chat_tool_result_max_items: int = 8,
        chat_first_token_watchdog_seconds: float = 20.0,
        chat_first_token_retry_max_attempts: int = 1,
        chat_stream_first_token_watchdog_seconds: float = 10.0,
        chat_stream_first_token_retry_max_attempts: int = 0,
    ):
        del (
            connection,
            llm_generator,
            settings,
            fast_path_enabled,
            fast_path_scope,
            fast_path_min_messages,
            fast_path_tool_cap,
            fast_path_fallback_on_empty,
            chat_dynamic_token_enabled,
            chat_analysis_max_tokens,
            chat_answer_max_tokens_short,
            chat_answer_max_tokens_long,
            chat_answer_max_tokens_team,
            chat_tool_result_max_chars,
            chat_tool_result_max_items,
            chat_first_token_watchdog_seconds,
            chat_first_token_retry_max_attempts,
            chat_stream_first_token_watchdog_seconds,
            chat_stream_first_token_retry_max_attempts,
        )
        raise RuntimeError(
            "BaseballStatisticsAgent is runtime-managed. Use BaseballAgentRuntime.shared_agent inside runtime.request_context(...)."
        )

    def _apply_shared_runtime(self, runtime: BaseballAgentRuntime) -> None:
        self._runtime = runtime
        self.runtime_id = runtime.runtime_id
        self.llm_generator = runtime.llm_generator
        self.settings = runtime.settings
        self.fast_path_enabled = runtime.fast_path_enabled
        self.fast_path_scope = runtime.fast_path_scope
        self.fast_path_min_messages = runtime.fast_path_min_messages
        self.fast_path_tool_cap = runtime.fast_path_tool_cap
        self.fast_path_fallback_on_empty = runtime.fast_path_fallback_on_empty
        self.chat_dynamic_token_enabled = runtime.chat_dynamic_token_enabled
        self.chat_analysis_max_tokens = runtime.chat_analysis_max_tokens
        self.chat_answer_max_tokens_short = runtime.chat_answer_max_tokens_short
        self.chat_answer_max_tokens_long = runtime.chat_answer_max_tokens_long
        self.chat_answer_max_tokens_team = runtime.chat_answer_max_tokens_team
        self.chat_tool_result_max_chars = runtime.chat_tool_result_max_chars
        self.chat_tool_result_max_items = runtime.chat_tool_result_max_items
        self.chat_planner_timeout_seconds = runtime.chat_planner_timeout_seconds
        self.chat_compact_planner_timeout_seconds = (
            runtime.chat_compact_planner_timeout_seconds
        )
        self.chat_first_token_watchdog_seconds = (
            runtime.chat_first_token_watchdog_seconds
        )
        self.chat_first_token_retry_max_attempts = (
            runtime.chat_first_token_retry_max_attempts
        )
        self.chat_stream_first_token_watchdog_seconds = (
            runtime.chat_stream_first_token_watchdog_seconds
        )
        self.chat_stream_first_token_retry_max_attempts = (
            runtime.chat_stream_first_token_retry_max_attempts
        )
        self.chat_tool_parallel_enabled = runtime.chat_tool_parallel_enabled
        self.chat_tool_parallel_split_batch_enabled = (
            runtime.chat_tool_parallel_split_batch_enabled
        )
        self.chat_tool_parallel_serial_tools = set(
            runtime.chat_tool_parallel_serial_tools
        )
        self.chat_tool_parallel_max_concurrency = (
            runtime.chat_tool_parallel_max_concurrency
        )
        self.chat_planner_cache_ttl_seconds = runtime.chat_planner_cache_ttl_seconds
        self.chat_planner_cache_max_entries = runtime.chat_planner_cache_max_entries
        self.tool_definitions = runtime.tool_definitions
        self.tool_description_text = runtime.tool_description_text
        self.tool_caller_factory = runtime.tool_caller_factory
        self.tool_caller = runtime.tool_caller_factory.bind(self)
        self._team_name_cache = runtime.get_team_name_cache()
        self.chat_intent_router = runtime.chat_intent_router.bind(self)
        self.chat_renderer_registry = runtime.chat_renderer_registry.bind(self)

    def _maybe_current_request_context(self) -> AgentRequestContext | None:
        request_context = _REQUEST_CONTEXT.get()
        if (
            request_context is not None
            and request_context.runtime_id == self.runtime_id
        ):
            return request_context
        return None

    def _current_request_context(self) -> AgentRequestContext:
        request_context = self._maybe_current_request_context()
        if request_context is None:
            raise RuntimeError(
                "AgentRequestContext is not active. Ensure the runtime request context is entered."
            )
        return request_context

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in _REQUEST_RESOURCE_ATTRS:
            request_context = self._maybe_current_request_context()
            if request_context is not None:
                return getattr(request_context, attr_name)
        raise AttributeError(attr_name)

    def _settings_int(self, attr_name: str, default: int) -> int:
        if self.settings is None:
            return default
        value = getattr(self.settings, attr_name, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _resolve_perf_model_name(self) -> str:
        if self.settings is None:
            return "unknown"
        if self.settings.llm_provider == "gemini":
            return self.settings.gemini_model
        return self.settings.openrouter_model

    @staticmethod
    def _normalize_planner_query(query: str) -> str:
        return re.sub(r"\s+", " ", str(query or "")).strip().casefold()

    def _build_planner_cache_key(
        self, query: str, context: Optional[Dict[str, Any]]
    ) -> str:
        normalized_query = self._normalize_planner_query(query)
        messages = context.get("messages") if isinstance(context, dict) else None
        history = context.get("history") if isinstance(context, dict) else None
        has_history = bool(messages or history)
        return f"{normalized_query}|history={int(has_history)}"

    def _get_cached_planner_plan(
        self, query: str, context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if self.chat_planner_cache_ttl_seconds <= 0:
            return None

        cache_key = self._build_planner_cache_key(query, context)
        now = time.monotonic()
        with _PLANNER_CACHE_LOCK:
            cached_entry = _PLANNER_CACHE.get(cache_key)
            if cached_entry is None:
                return None
            cached_at, cached_plan = cached_entry
            age_seconds = now - cached_at
            if age_seconds > self.chat_planner_cache_ttl_seconds:
                _PLANNER_CACHE.pop(cache_key, None)
                return None
            _PLANNER_CACHE.move_to_end(cache_key)

        plan = deepcopy(cached_plan)
        plan["planner_cache_hit"] = True
        plan["planner_cache_age_ms"] = round(age_seconds * 1000, 2)
        if "analysis_ms" not in plan:
            plan["analysis_ms"] = 0.0
        return plan

    def _store_planner_plan_cache(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        plan: Dict[str, Any],
    ) -> None:
        if self.chat_planner_cache_ttl_seconds <= 0 or plan.get("error"):
            return

        cache_key = self._build_planner_cache_key(query, context)
        cached_plan = deepcopy(plan)
        cached_plan.pop("planner_cache_hit", None)
        cached_plan.pop("planner_cache_age_ms", None)

        with _PLANNER_CACHE_LOCK:
            _PLANNER_CACHE[cache_key] = (time.monotonic(), cached_plan)
            _PLANNER_CACHE.move_to_end(cache_key)
            while len(_PLANNER_CACHE) > self.chat_planner_cache_max_entries:
                _PLANNER_CACHE.popitem(last=False)

    @staticmethod
    def _has_hallucinated_tool_parameters(parameters: Dict[str, Any]) -> bool:
        hallucination_indicators = [
            "DATE_FROM_STEP",
            "YEAR_FROM_CONTEXT",
            "STEP",
            "FROM",
            "{{",
            "}}",
            "추출된 날짜",
            "추출된 선수명",
            "확인된 선수명",
            "DATE_FROM_",
            "YEAR_FROM_",
        ]
        for param_val in parameters.values():
            if isinstance(param_val, str) and any(
                indicator in param_val.upper() for indicator in hallucination_indicators
            ):
                return True
        return False

    def _normalize_player_name_candidate(self, raw_name: Any) -> str:
        name = str(raw_name or "").strip()
        if not name:
            return ""

        name = re.sub(r"^[\s\"'“”‘’\[\]\(\),./]+|[\s\"'“”‘’\[\]\(\),./]+$", "", name)
        if not name:
            return ""

        normalized = name
        for suffix in PLAYER_NAME_NORMALIZATION_SUFFIXES:
            if normalized.endswith(suffix):
                candidate = normalized[: -len(suffix)].strip()
                if len(candidate) >= 2:
                    normalized = candidate
                    break

        normalized = re.sub(
            r"^[\s\"'“”‘’\[\]\(\),./]+|[\s\"'“”‘’\[\]\(\),./]+$",
            "",
            normalized,
        )
        if len(normalized) >= 2:
            return normalized
        return name

    def _eligible_player_stats_batch_spec(
        self, tool_calls: List[ToolCall]
    ) -> Optional[Dict[str, Any]]:
        if not 2 <= len(tool_calls) <= PLAYER_LLM_MULTI_NAME_CAP:
            return None
        if any(call.tool_name != "get_player_stats" for call in tool_calls):
            return None

        player_names: List[str] = []
        years: set[int] = set()
        positions: set[str] = set()

        for call in tool_calls:
            player_name = self._normalize_player_name_candidate(
                call.parameters.get("player_name")
            )
            year = call.parameters.get("year")
            position = str(call.parameters.get("position") or "both").strip() or "both"
            if len(player_name) < 2 or not isinstance(year, int):
                return None
            player_names.append(player_name)
            years.add(year)
            positions.add(position)

        if len(years) != 1 or len(positions) != 1:
            return None

        return {
            "player_names": player_names,
            "year": next(iter(years)),
            "position": next(iter(positions)),
        }

    async def _execute_batched_player_stats_tool_calls_async(
        self, tool_calls: List[ToolCall]
    ) -> Optional[List[ToolResult]]:
        batch_spec = self._eligible_player_stats_batch_spec(tool_calls)
        db_query_tool = getattr(self, "db_query_tool", None)
        if batch_spec is None or not hasattr(
            db_query_tool, "get_player_season_stats_batch"
        ):
            return None

        try:
            raw_results = await asyncio.to_thread(
                db_query_tool.get_player_season_stats_batch,
                batch_spec["player_names"],
                batch_spec["year"],
                batch_spec["position"],
            )
        except Exception as exc:
            logger.warning(
                "[PlayerBatchLookup] used=false reason=execution_error error=%s",
                exc,
            )
            return None

        if not isinstance(raw_results, list) or len(raw_results) != len(tool_calls):
            logger.warning(
                "[PlayerBatchLookup] used=false reason=invalid_result_shape count=%s expected=%d",
                len(raw_results) if isinstance(raw_results, list) else "n/a",
                len(tool_calls),
            )
            return None

        logger.info(
            "[PlayerBatchLookup] used=true count=%d year=%s position=%s",
            len(tool_calls),
            batch_spec["year"],
            batch_spec["position"],
        )

        tool_results: List[ToolResult] = []
        for tool_call, raw_result in zip(tool_calls, raw_results):
            if not isinstance(raw_result, dict):
                return None

            requested_name = (
                self._normalize_player_name_candidate(
                    tool_call.parameters.get("player_name")
                )
                or str(tool_call.parameters.get("player_name") or "").strip()
            )
            if raw_result.get("error"):
                tool_results.append(
                    ToolResult(
                        success=False,
                        data=raw_result,
                        message=f"DB 조회 오류: {raw_result['error']}",
                    )
                )
                continue

            if not raw_result.get("found"):
                tool_results.append(
                    ToolResult(
                        success=True,
                        data=raw_result,
                        message=(
                            f"{batch_spec['year']}년 '{requested_name}' 선수의 기록을 찾을 수 없습니다. "
                            "선수명 확인이나 다른 연도를 시도해보세요."
                        ),
                    )
                )
                continue

            tool_results.append(
                ToolResult(
                    success=True,
                    data=raw_result,
                    message=(
                        f"{batch_spec['year']}년 {requested_name} 선수 통계를 성공적으로 조회했습니다."
                    ),
                )
            )

        return tool_results

    async def _execute_tool_call_async(self, tool_call: ToolCall) -> ToolResult:
        if self._has_hallucinated_tool_parameters(tool_call.parameters):
            return ToolResult(
                success=False,
                data={},
                message=f"매개변수 오류: {tool_call.tool_name}",
            )
        return await self.tool_caller.execute_tool_async(tool_call)

    async def _execute_tool_batch_async(
        self, tool_calls: List[ToolCall]
    ) -> List[ToolResult]:
        if not tool_calls:
            return []

        batched_player_results = (
            await self._execute_batched_player_stats_tool_calls_async(tool_calls)
        )
        if batched_player_results is not None:
            return batched_player_results

        execution_mode = self._resolve_tool_execution_mode(tool_calls)
        if execution_mode == "sequential":
            return [
                await self._execute_tool_call_async(tool_call)
                for tool_call in tool_calls
            ]
        if execution_mode == "mixed":
            return await self._execute_tool_batch_mixed_async(tool_calls)

        return await self._execute_parallel_tool_batch_async(tool_calls)

    async def _execute_parallel_tool_batch_async(
        self, tool_calls: List[ToolCall]
    ) -> List[ToolResult]:
        executable_calls: List[ToolCall] = []
        executable_indexes: List[int] = []
        results: List[Optional[ToolResult]] = [None] * len(tool_calls)

        for index, tool_call in enumerate(tool_calls):
            if self._has_hallucinated_tool_parameters(tool_call.parameters):
                results[index] = ToolResult(
                    success=False,
                    data={},
                    message=f"매개변수 오류: {tool_call.tool_name}",
                )
            else:
                executable_calls.append(tool_call)
                executable_indexes.append(index)

        parallel_results = await self.tool_caller.execute_multiple_tools_parallel(
            executable_calls,
            max_concurrency=self.chat_tool_parallel_max_concurrency,
        )
        for index, result in zip(executable_indexes, parallel_results):
            results[index] = result

        return [
            (
                result
                if result is not None
                else ToolResult(success=False, data={}, message="도구 실행 중 오류")
            )
            for result in results
        ]

    async def _execute_tool_batch_mixed_async(
        self, tool_calls: List[ToolCall]
    ) -> List[ToolResult]:
        results: List[Optional[ToolResult]] = [None] * len(tool_calls)
        parallel_calls: List[ToolCall] = []
        parallel_indexes: List[int] = []

        for index, tool_call in enumerate(tool_calls):
            if self._has_hallucinated_tool_parameters(tool_call.parameters):
                results[index] = ToolResult(
                    success=False,
                    data={},
                    message=f"매개변수 오류: {tool_call.tool_name}",
                )
                continue

            if self._is_serial_tool_call(tool_call):
                results[index] = await self._execute_tool_call_async(tool_call)
                continue

            parallel_calls.append(tool_call)
            parallel_indexes.append(index)

        if parallel_calls:
            parallel_results = await self.tool_caller.execute_multiple_tools_parallel(
                parallel_calls,
                max_concurrency=self.chat_tool_parallel_max_concurrency,
            )
            for index, result in zip(parallel_indexes, parallel_results):
                results[index] = result

        return [
            (
                result
                if result is not None
                else ToolResult(success=False, data={}, message="도구 실행 중 오류")
            )
            for result in results
        ]

    def _is_serial_tool_call(self, tool_call: ToolCall) -> bool:
        serial_tools = getattr(
            self, "chat_tool_parallel_serial_tools", SERIAL_DB_TOOL_NAMES
        )
        return tool_call.tool_name in serial_tools

    def _resolve_tool_execution_mode(self, tool_calls: List[ToolCall]) -> str:
        if not tool_calls:
            return "none"
        if len(tool_calls) == 1 or not self.chat_tool_parallel_enabled:
            return "sequential"
        has_serial_tool = any(self._is_serial_tool_call(call) for call in tool_calls)
        has_parallel_safe_tool = any(
            not self._is_serial_tool_call(call) for call in tool_calls
        )
        if has_serial_tool and has_parallel_safe_tool:
            if getattr(self, "chat_tool_parallel_split_batch_enabled", True):
                return "mixed"
            return "sequential"
        if has_serial_tool:
            return "sequential"
        return "parallel"

    def _register_tools(self):
        """사용 가능한 도구들을 등록합니다."""

        # 선수 개별 통계 조회 도구
        self.tool_caller.register_tool(
            "get_player_stats",
            "선수의 개별 시즌 통계를 실제 DB에서 조회합니다. 타율, 홈런, ERA 등 개인 기록 질문에 사용하세요.",
            {
                "player_name": "선수명 (부분 매칭 가능)",
                "year": "시즌 년도",
                "position": "batting(타자), pitching(투수), 또는 both(둘다, 기본값)",
            },
            self._tool_get_player_stats,
        )

        # 선수 통산 통계 조회 도구
        self.tool_caller.register_tool(
            "get_career_stats",
            "선수의 통산(커리어) 통계를 실제 DB에서 조회합니다. '통산', '총', '전체' 기록 질문에 사용하세요.",
            {
                "player_name": "선수명 (부분 매칭 가능)",
                "position": "batting(타자), pitching(투수), 또는 both(둘다, 기본값)",
            },
            self._tool_get_career_stats,
        )

        # 리더보드/순위 조회 도구
        self.tool_caller.register_tool(
            "get_leaderboard",
            "특정 통계 지표의 순위/리더보드를 실제 DB에서 조회합니다. '최고', '상위', '1위' 등의 질문에 사용하세요.",
            {
                "stat_name": "통계 지표명 (ops, era, home_runs, 타율, 홈런 등)",
                "year": "시즌 년도",
                "position": "batting(타자) 또는 pitching(투수)",
                "team_filter": "특정 팀만 조회 (선택적, 예: KIA, LG)",
                "limit": "상위 몇 명까지 (선택적, 기본 10명)",
            },
            self._tool_get_leaderboard,
        )

        # 선수 존재 여부 확인 도구
        self.tool_caller.register_tool(
            "validate_player",
            "선수가 해당 연도에 실제로 기록이 있는지 DB에서 확인합니다. 선수명 오타나 존재하지 않는 선수 질문 시 사용하세요.",
            {
                "player_name": "선수명",
                "year": "시즌 년도 (기본값: current_year = {current_year})",
            },
            self._tool_validate_player,
        )

        # 팀 요약 정보 조회 도구
        self.tool_caller.register_tool(
            "get_team_summary",
            "특정 팀의 시즌 성적, 경기 기록 요약, 주요 선수 정보를 조회합니다. '삼성 경기 기록 어때?', '롯데 잘해?' 등의 질문에 사용하세요.",
            {"team_name": "팀명 (KIA, LG, 두산 등)", "year": "시즌 년도"},
            self._tool_get_team_summary,
        )

        # 팀 심층 지표 조회 도구 (과부하 진단용)
        self.tool_caller.register_tool(
            "get_team_advanced_metrics",
            "팀의 전반적인 성적 순위(ERA 1위 등)와 불펜 과부하 지표(Bullpen Share)를 조회합니다. 팀의 강/약점 분석 및 '과부하' 여부를 판단할 때 반드시 사용하세요.",
            {"team_name": "팀명 (KIA, LG, 두산 등)", "year": "시즌 년도"},
            self._tool_get_team_advanced_metrics,
        )

        # 포지션 정보 조회 도구
        self.tool_caller.register_tool(
            "get_position_info",
            "포지션 약어를 전체 포지션명으로 변환합니다. 포지션 관련 질문에 사용하세요.",
            {
                "position_abbr": "포지션 약어 (지, 타, 주, 중, 좌, 우, 一, 二, 三, 유, 포)"
            },
            self._tool_get_position_info,
        )

        # 팀 기본 정보 조회 도구
        self.tool_caller.register_tool(
            "get_team_basic_info",
            "팀의 기본 정보를 조회합니다. 홈구장, 마스코트, 창단연도 등의 질문에 사용하세요.",
            {"team_name": "팀명 (KIA, LG, 두산 등)"},
            self._tool_get_team_basic_info,
        )

        # 수비 통계 조회 도구
        self.tool_caller.register_tool(
            "get_defensive_stats",
            "선수의 수비 통계를 조회합니다. 수비율, 오류, 어시스트 등의 질문에 사용하세요.",
            {"player_name": "선수명", "year": "시즌 년도 (선택적, 생략하면 통산)"},
            self._tool_get_defensive_stats,
        )

        # 구속 데이터 조회 도구
        self.tool_caller.register_tool(
            "get_velocity_data",
            "투수의 구속 데이터를 조회합니다. 직구, 변화구 구속 등의 질문에 사용하세요.",
            {
                "player_name": "선수명",
                "year": "시즌 년도 (선택적, 생략하면 최근 데이터)",
            },
            self._tool_get_velocity_data,
        )

        # 고급 통계 조회 도구 (신규)
        self.tool_caller.register_tool(
            "get_advanced_stats",
            "선수의 고급 통계(ERA+, OPS+, FIP, QS 등)를 조회합니다. 전문가 수준의 분석이나 세부 지표 질문에 사용하세요.",
            {
                "player_name": "선수명",
                "year": "시즌 년도",
                "position": "batting(타자), pitching(투수), 또는 both(둘다, 기본값)",
            },
            self._tool_get_advanced_stats,
        )

        # KBO 규정 검색 도구
        self.tool_caller.register_tool(
            "search_regulations",
            "KBO 규정을 검색합니다. 야구 규칙, 제도, 판정 기준 등의 질문에 사용하세요.",
            {"query": "검색할 규정 내용 (예: 타이브레이크, FA 조건, 인필드 플라이)"},
            self._tool_search_regulations,
        )

        # 규정 카테고리별 조회 도구
        self.tool_caller.register_tool(
            "get_regulations_by_category",
            "특정 카테고리의 규정들을 조회합니다. 체계적인 규정 설명이 필요할 때 사용하세요.",
            {
                "category": "규정 카테고리 (basic, player, game, technical, discipline, postseason, special, terms)"
            },
            self._tool_get_regulations_by_category,
        )

        # 경기 박스스코어 조회 도구
        self.tool_caller.register_tool(
            "get_game_box_score",
            "특정 경기의 박스스코어와 상세 정보를 SQL로 조회합니다. 경기 결과, 박스스코어, 이닝별 득점, '7회에 몇 점 났어?' 같은 질문에 우선 사용하세요.",
            {
                "game_id": "경기 고유 ID (선택적)",
                "date": "경기 날짜 (YYYY-MM-DD, 선택적)",
                "home_team": "홈팀명 (선택적)",
                "away_team": "원정팀명 (선택적)",
            },
            self._tool_get_game_box_score,
        )

        # 날짜별 경기 조회 도구
        self.tool_caller.register_tool(
            "get_games_by_date",
            "특정 날짜의 모든 경기를 조회합니다. '오늘 경기', '어제 경기' 등의 질문에 사용하세요.",
            {"date": "경기 날짜 (YYYY-MM-DD)"},
            self._tool_get_games_by_date,
        )

        # 팀 최근 경기 조회 도구 (신규 추가)
        self.tool_caller.register_tool(
            "get_recent_games_by_team",
            "특정 팀의 최근 경기 기록을 조회합니다. '최근 5경기', '최근 성적', '요즘 경기 결과' 등의 질문에 사용하세요.",
            {
                "team_name": "팀명 (예: KT, KIA, LG)",
                "limit": "조회할 경기 수 (선택적, 기본값 5)",
                "year": "시즌 연도 (선택적)",
            },
            self._tool_get_recent_games_by_team,
        )

        # 경기 라인업 조회 도구
        self.tool_caller.register_tool(
            "get_game_lineup",
            "특정 경기의 선발 라인업(타순, 포지션, 선수명)을 조회합니다. '누가 나와?', '라인업 알려줘', '선발 누구요?' 등의 질문에 사용하세요.",
            {
                "game_id": "경기 고유 ID (선택적)",
                "date": "경기 날짜 (YYYY-MM-DD, 선택적)",
                "team_name": "팀명 (선택적)",
            },
            self._tool_get_game_lineup,
        )

        # 팀 간 직접 대결 조회 도구
        self.tool_caller.register_tool(
            "get_head_to_head",
            "두 팀 간의 직접 대결 기록을 조회합니다. 맞대결 성적, 승부 현황 등의 질문에 사용하세요.",
            {
                "team1": "팀1 이름",
                "team2": "팀2 이름",
                "year": "시즌 년도 (선택적)",
                "limit": "최근 몇 경기까지 (선택적, 기본 10경기)",
            },
            self._tool_get_head_to_head,
        )

        # 선수 경기 성적 조회 도구
        self.tool_caller.register_tool(
            "get_player_game_performance",
            "특정 선수의 개별 경기 성적을 조회합니다. 특정 경기에서의 선수 활약 등의 질문에 사용하세요.",
            {
                "player_name": "선수명",
                "date": "경기 날짜 (선택적)",
                "recent_games": "최근 몇 경기까지 (선택적, 기본 5경기)",
            },
            self._tool_get_player_game_performance,
        )

        # 선수 비교 도구
        self.tool_caller.register_tool(
            "compare_players",
            "두 선수의 통계를 비교 분석합니다. 'A vs B', 'A와 B 중 누가' 등 선수 비교 질문에 사용하세요.",
            {
                "player1": "첫 번째 선수명",
                "player2": "두 번째 선수명",
                "comparison_type": "career(통산 비교, 기본값) 또는 season(특정 시즌 비교)",
                "year": "특정 시즌 비교 시 연도 (선택적)",
                "position": "batting(타자), pitching(투수), 또는 both(둘다, 기본값)",
            },
            self._tool_compare_players,
        )

        # 시즌 마지막 경기 정보 조회 도구 (통합 버전)
        self.tool_caller.register_tool(
            "get_season_final_game_date",
            "특정 시즌의 마지막 경기 날짜와 경기 결과를 한 번에 조회합니다. '마지막 경기', '최종전', '한국시리즈 마지막', '작년 마지막 경기 결과' 등의 질문에 사용하세요.",
            {
                "year": "시즌 년도",
                "league_type": "'regular_season'(정규시즌) 또는 'korean_series'(한국시리즈, 기본값)",
            },
            self._tool_get_season_final_game_date,
        )

        # 팀 순위 조회 도구
        self.tool_caller.register_tool(
            "get_team_rank",
            "특정 시즌의 팀 최종 순위를 조회합니다. '몇 등', '순위', '시즌 마무리' 등의 질문에 사용하세요.",
            {"team_name": "팀명 (예: 'KIA', '기아', 'SSG')", "year": "시즌 년도"},
            self._tool_get_team_rank,
        )

        self.tool_caller.register_tool(
            "get_team_by_rank",
            "특정 시즌 정규시즌에서 몇 위를 한 팀이 누구인지 조회합니다. '2024년 3위 팀', '작년 1위 팀' 같은 질문에 사용하세요.",
            {
                "year": "시즌 년도",
                "rank": "조회할 최종 순위",
                "season_phase": "'regular'(기본값) 또는 'regular_season'",
            },
            self._tool_get_team_by_rank,
        )

        # 지능적 팀별 마지막 경기 조회 도구
        self.tool_caller.register_tool(
            "get_team_last_game",
            "특정 팀의 실제 마지막 경기를 지능적으로 조회합니다. 팀 순위를 확인하여 포스트시즌 진출팀(1-5위)은 한국시리즈, 미진출팀(6-10위)은 정규시즌 마지막 경기를 자동으로 찾습니다.",
            {"team_name": "팀명 (예: 'SSG', '기아', 'KIA')", "year": "시즌 년도"},
            self._tool_get_team_last_game,
        )

        # 한국시리즈 우승팀 조회 도구
        self.tool_caller.register_tool(
            "get_korean_series_winner",
            """특정 시즌의 한국시리즈 우승팀을 조회합니다.

다음 질문에서 반드시 사용하세요:
- '우승팀', '챔피언', '한국시리즈 우승' 등의 질문
- '작년 우승팀', '지난해 챔피언' (시간 확인 후 연도 변환하여 호출)
- 'X년 우승팀은?' 

이 도구는 한국시리즈 마지막 경기의 승리팀을 자동으로 식별하여 우승팀을 판단합니다.""",
            {"year": "시즌 년도"},
            self._tool_get_korean_series_winner,
        )

        # 시즌 수상자 조회 도구
        self.tool_caller.register_tool(
            "get_award_winners",
            "특정 시즌의 MVP, 신인왕, 골든글러브 등 수상자를 조회합니다. '2025 MVP 누구야?', '신인왕 알려줘' 같은 질문에 사용하세요.",
            {
                "year": "시즌 년도",
                "award_type": "수상 종류 (mvp, rookie, golden_glove, korean_series_mvp, all_star_mvp)",
            },
            self._tool_get_award_winners,
        )

        # 현재 날짜 및 시간 조회 도구
        self.tool_caller.register_tool(
            "get_current_datetime",
            """현재 날짜와 시간을 한국 시간 기준으로 조회합니다.

다음 상황에서 반드시 이 도구를 먼저 사용해야 합니다:
- '작년', '지난해', '올해', '금년' 등 상대적 시간 표현이 포함된 질문
- '오늘', '지금' 등 현재 시점 기준 질문
- 우승팀, 시즌 기록 등에서 정확한 연도가 필요한 질문
- 예: '작년 우승팀은?' → 먼저 현재 날짜 확인하여 '작년' = '2024년'임을 파악

중요: 상대적 시간 표현이 있으면 절대적 연도로 변환하기 위해 반드시 이 도구를 호출하세요.""",
            {},
            self._tool_get_current_datetime,
        )

        self.tool_caller.register_tool(
            "get_baseball_season_info",
            "현재 KBO 야구 시즌 정보 조회합니다. '지금 야구 시즌이야?', '현재 시즌 상태는?' 등의 질문에 사용하세요.",
            {},
            self._tool_get_baseball_season_info,
        )

        # 문서 검색 도구 (신규 추가)
        self.tool_caller.register_tool(
            "search_documents",
            "KBO/야구 설명형 문서와 지식 베이스를 검색합니다. 'ABS가 뭐야?', 'wRC+ 뜻', '마스코트', '응원 문화' 같은 설명형 질문에 사용하세요.",
            {
                "query": "검색할 질문 또는 키워드",
                "limit": "반환할 최대 결과 수 (선택적, 기본값 10)",
            },
            self._tool_search_documents,
        )

        self.tool_caller.register_tool(
            "get_regulations",
            "규정/용어/설명형 질문용 legacy alias입니다. search_documents와 동일하게 사용하세요.",
            {
                "query": "검색할 질문 또는 키워드",
                "limit": "반환할 최대 결과 수 (선택적, 기본값 5)",
            },
            self._tool_search_documents,
        )

        # WPA(승리 확률) 계산 도구 (신규 추가)
        self.tool_caller.register_tool(
            "calculate_win_probability",
            "특정 경기 상황(이닝, 점수차, 주자, 아웃)에서의 승리 확률을 계산합니다. '9회말 무사 1루 승률은?', '이 상황에서 역전 확률은?' 등의 질문에 사용하세요.",
            {
                "inning": "이닝 (1-12)",
                "is_top": "초/말 여부 (true=초, false=말)",
                "score_diff": "점수차 (홈팀 기준, 홈스코어 - 원정스코어)",
                "outs": "아웃 카운트 (0-2)",
                "runner_on_1st": "1루 주자 유무 (true/false)",
                "runner_on_2nd": "2루 주자 유무 (true/false)",
                "runner_on_3rd": "3루 주자 유무 (true/false)",
            },
            self._tool_calculate_win_probability,
        )

        # 승부 예측 도구 (신규 추가: Agentic Reasoning)
        self.tool_caller.register_tool(
            "predict_matchup",
            "투수와 타자의 맞대결 승부를 예측합니다. '류현진 vs 김도영', '오늘 대결 누가 이길까?' 등의 질문에 사용하세요. 단순 비교가 아닌 승리 확률을 추론합니다.",
            {
                "pitcher_name": "투수 이름",
                "batter_name": "타자 이름",
                "year": "기준 연도 (기본값: 현재)",
            },
            self._tool_predict_matchup,
        )

        # WPA 리더보드 조회 도구
        self.tool_caller.register_tool(
            "get_player_wpa_leaders",
            "가장 높은 WPA(승리 확률 기여도)를 기록한 선수를 조회합니다. '클러치 히터', '해결사', 'WPA 1위' 등의 질문에 사용하세요.",
            {
                "year": "조회할 시즌 (기본값: current_year)",
                "limit": "조회할 상위 선수 수 (기본값: 10)",
                "team_name": "특정 팀 필터링 (선택 사항)",
            },
            self._tool_get_player_wpa_leaders,
        )

        # 클러치 상황 조회 도구
        self.tool_caller.register_tool(
            "get_clutch_moments",
            "특정 경기에서 승부의 결정적인 순간(WPA 변화가 컸던 순간)들을 조회합니다. '승부처가 언제였어?', '어떤 상황이 결정적이었어?' 등의 질문에 사용하세요.",
            {
                "game_id": "경기 ID",
                "limit": "조회할 상위 상황 수 (기본값: 5)",
            },
            self._tool_get_clutch_moments,
        )

        # 선수 개별 WPA 통계 조회 도구
        self.tool_caller.register_tool(
            "get_player_wpa_stats",
            "특정 선수의 WPA 관련 통계를 상세 조회합니다. '김도영 선수가 중요한 순간에 잘 쳤어?', '이 선수의 WPA 어때?' 등의 질문에 사용하세요.",
            {
                "player_name": "선수 명",
                "year": "조회할 시즌 (기본값: current_year)",
            },
            self._tool_get_player_wpa_stats,
        )

        # 불펜 가용성 확인 도구 (신규 추가: Game Strategy)
        self.tool_caller.register_tool(
            "check_bullpen_availability",
            "특정 팀의 불펜 투수 가용성을 확인합니다. '오늘 LG 불펜 누구 나올 수 있어?', '투수 혹사 상태 확인해줘' 등의 질문에 사용하세요.",
            {
                "team_name": "팀 이름",
                "date": "기준 날짜 (YYYY-MM-DD, 생략 시 오늘)",
            },
            self._tool_check_bullpen,
        )

        # 투수 교체 추천 도구 (신규 추가: Game Strategy)
        self.tool_caller.register_tool(
            "recommend_pitcher_change",
            "경기 상황에 맞는 구원 투수를 추천합니다. '지금 누구 올려야 해?', '좌타자 상대로 누구 낼까?' 등의 질문에 사용하세요.",
            {
                "team_name": "팀 이름",
                "situation": "경기 상황 (winning_close, losing, lefty_batter 등)",
            },
            self._tool_recommend_pitcher,
        )

    def _tool_check_bullpen(self, team_name: str, date: str = None) -> ToolResult:
        """불펜 가용성 확인 도구"""
        try:
            strategist = self._current_request_context().game_strategist
            result = strategist.check_bullpen_availability(team_name, date)

            if "error" in result:
                return ToolResult(success=False, data=result, message=result["error"])

            # 메시지 포맷팅
            status_list = result.get("bullpen_status", [])
            available = [p["name"] for p in status_list if p["status"] == "Available"]
            warning = [
                f"{p['name']}({p['reason']})"
                for p in status_list
                if p["status"] == "Warning"
            ]
            unavailable = [
                f"{p['name']}({p['reason']})"
                for p in status_list
                if p["status"] in ["Unavailable", "High Risk"]
            ]

            msg = f"[{result['team']} 불펜 현황]\n"
            msg += f"✅ 가용: {', '.join(available) if available else '없음'}\n"
            msg += f"⚠️ 주의: {', '.join(warning) if warning else '없음'}\n"
            msg += f"⛔ 불가: {', '.join(unavailable) if unavailable else '없음'}"

            return ToolResult(success=True, data=result, message=msg)
        except Exception as e:
            logger.error(f"Bullpen check error: {e}")
            return ToolResult(success=False, data={}, message=f"불펜 확인 중 오류: {e}")

    def _tool_recommend_pitcher(self, team_name: str, situation: str) -> ToolResult:
        """투수 추천 도구"""
        try:
            strategist = self._current_request_context().game_strategist
            result = strategist.recommend_pitcher(team_name, situation)

            if "error" in result:
                return ToolResult(success=False, data=result, message=result["error"])

            recs = result.get("recommended", [])
            rec_names = ", ".join([p["name"] for p in recs])

            return ToolResult(
                success=True,
                data=result,
                message=f"[{team_name} {situation} 상황 추천]\n추천 투수: {rec_names}\n(최근 휴식일과 기록을 고려한 추천입니다)",
            )
        except Exception as e:
            logger.error(f"Pitcher recommendation error: {e}")
            return ToolResult(success=False, data={}, message=f"투수 추천 중 오류: {e}")

    def _tool_predict_matchup(
        self, pitcher_name: str, batter_name: str, year: int = None
    ) -> ToolResult:
        """투타 맞대결 예측 도구"""
        if year is None:
            year = date.today().year

        try:
            predictor = self._current_request_context().match_predictor
            result = predictor.predict(pitcher_name, batter_name, year)

            if "error" in result:
                return ToolResult(success=False, data=result, message=result["error"])

            winner = result["predicted_winner"]
            prob = result["win_probability"]
            reasons = (
                ", ".join(result["reasons"])
                if result["reasons"]
                else "데이터 부족으로 인한 기본 확률"
            )

            msg = f"예측 결과: {winner} 우세 (확률 {prob:.0%})\n근거: {reasons}"
            return ToolResult(success=True, data=result, message=msg)

        except Exception as e:
            logger.error(f"Matchup prediction error: {e}")
            return ToolResult(success=False, data={}, message=f"예측 중 오류 발생: {e}")

    def _tool_calculate_win_probability(
        self,
        inning: int,
        is_top: bool,
        score_diff: int,
        outs: int,
        runner_on_1st: bool = False,
        runner_on_2nd: bool = False,
        runner_on_3rd: bool = False,
    ) -> ToolResult:
        """승리 확률 계산 도구"""
        from ..core.wpa_calculator import WPACalculator

        try:
            calc = WPACalculator()
            runners = (runner_on_1st, runner_on_2nd, runner_on_3rd)
            prob = calc.calculate_win_probability(
                inning, is_top, score_diff, outs, runners
            )

            situation_desc = (
                f"{inning}회{'초' if is_top else '말'} {score_diff}점차 {outs}아웃"
            )
            if any(runners):
                r_desc = []
                if runner_on_1st:
                    r_desc.append("1루")
                if runner_on_2nd:
                    r_desc.append("2루")
                if runner_on_3rd:
                    r_desc.append("3루")
                situation_desc += f" 주자 {'/'.join(r_desc)}"
            else:
                situation_desc += " 주자 없음"

            return ToolResult(
                success=True,
                data={"win_probability": prob, "percent": f"{prob:.1%}"},
                message=f"[{situation_desc}] 상황의 홈팀 승리 확률은 약 {prob:.1%} 입니다.",
            )
        except Exception as e:
            logger.error(f"WPA tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"승리 확률 계산 중 오류: {e}"
            )

    def _tool_search_documents(self, query: str, limit: int = 10) -> ToolResult:
        """문서 검색 도구의 래퍼 함수"""
        try:
            effective_limit = max(1, min(limit, 2))
            result = self.document_query_tool.search_documents(query, effective_limit)

            if result.get("error"):
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"문서 검색 오류: {result['error']}",
                )

            if not result.get("found"):
                return ToolResult(
                    success=True,
                    data={
                        "source": result.get("source", "verified_docs"),
                        "found": False,
                        "documents": [],
                        "results": [],
                    },
                    message=f"'{query}'와 관련된 문서를 찾을 수 없습니다.",
                )

            return ToolResult(
                success=True,
                data={
                    "source": result.get("source", "verified_docs"),
                    "found": True,
                    "documents": result.get("documents", [])[:effective_limit],
                    "results": result.get("documents", [])[:effective_limit],
                    "source_tables": result.get("source_tables", []),
                },
                message=f"'{query}' 관련 문서를 {len(result.get('documents', []))}개 찾았습니다.",
            )

        except Exception as e:
            logger.error(f"[BaseballAgent] Document search tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"문서 검색 도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_current_datetime(self, **kwargs) -> ToolResult:
        """현재 날짜 및 시간 조회 도구"""
        try:
            datetime_info = get_current_datetime()
            return ToolResult(
                success=True,
                data=datetime_info,
                message=f"현재 시간은 {datetime_info['formatted_date']} {datetime_info['formatted_time']}입니다.",
            )
        except Exception as e:
            logger.error(f"Current datetime tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"현재 시간 조회 중 오류 발생: {e}"
            )

    def _tool_get_baseball_season_info(self, **kwargs) -> ToolResult:
        """현재 야구 시즌 정보 조회 도구"""
        try:
            season_info = get_baseball_season_info()
            return ToolResult(
                success=True,
                data=season_info,
                message=f"현재 {season_info['current_year']}년 야구 시즌은 '{season_info['season_status']}' 상태입니다.",
            )
        except Exception as e:
            logger.error(f"Baseball season info tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"야구 시즌 정보 조회 중 오류 발생: {e}"
            )

    def _load_team_name_mapping(self) -> Dict[str, str]:
        """팀 ID와 팀명 매핑을 로드합니다."""
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            shared_cache = runtime.get_team_name_cache()
            if shared_cache is not None:
                self._team_name_cache = shared_cache
                return shared_cache

        if self._team_name_cache is not None:
            return self._team_name_cache

        mapping = dict(TEAM_CODE_TO_NAME)

        resolver = getattr(getattr(self, "db_query_tool", None), "team_resolver", None)
        if resolver is not None:
            for team_code, display_name in getattr(
                resolver, "code_to_name", {}
            ).items():
                if isinstance(team_code, str) and isinstance(display_name, str):
                    mapping[team_code] = display_name

            for alias in getattr(resolver, "name_to_canonical", {}).keys():
                if not isinstance(alias, str) or not alias.strip():
                    continue
                display_name = resolver.display_name(alias)
                if isinstance(display_name, str) and display_name.strip():
                    mapping[alias] = display_name

        self._team_name_cache = mapping
        if runtime is not None:
            runtime.set_team_name_cache(mapping)
        return mapping

    def _convert_team_id_to_name(self, team_id: str) -> str:
        """팀 ID를 팀명으로 변환합니다."""
        if not team_id:
            return team_id

        db_query_tool = getattr(self, "db_query_tool", None)
        if db_query_tool is not None and hasattr(db_query_tool, "get_team_name"):
            try:
                resolved_name = db_query_tool.get_team_name(team_id)
            except Exception as exc:
                logger.debug(
                    "[BaseballAgent] Failed to resolve team name via DatabaseQueryTool for %s: %s",
                    team_id,
                    exc,
                )
            else:
                if (
                    isinstance(resolved_name, str)
                    and resolved_name.strip()
                    and resolved_name != team_id
                ):
                    return resolved_name

        mapping = self._load_team_name_mapping()
        return mapping.get(team_id, team_id)

    def _format_game_info_with_team_names(
        self, game_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """게임 정보에서 팀 ID를 팀명으로 변환하여 포맷팅합니다."""
        formatted = game_info.copy()

        if "home_team" in formatted:
            formatted["home_team_name"] = self._convert_team_id_to_name(
                formatted["home_team"]
            )
        if "away_team" in formatted:
            formatted["away_team_name"] = self._convert_team_id_to_name(
                formatted["away_team"]
            )

        return formatted

    def _format_league_type_to_korean(self, league_type: str) -> str:
        """리그 타입을 한국어로 변환합니다."""
        league_mapping = {
            "korean_series": "한국시리즈",
            "regular_season": "정규시즌",
            "postseason": "포스트시즌",
            "wild_card": "와일드카드",
            "semi_playoff": "준플레이오프",
            "playoff": "플레이오프",
        }
        return league_mapping.get(league_type, league_type)

    def _format_game_status_to_korean(self, status: str) -> Optional[str]:
        """게임 상태를 한국어로 변환하거나 불필요한 상태는 제거합니다."""
        # COMPLETED 같은 과거 경기 상태는 표시하지 않음 (이미 지난 경기이므로)
        status_mapping = {
            "COMPLETED": "",  # 완료된 경기는 상태 표시하지 않음
            "SCHEDULED": "예정",
            "LIVE": "진행 중",
            "CANCELLED": "취소됨",
            "POSTPONED": "연기됨",
        }
        formatted_status = status_mapping.get(status, status)
        return formatted_status if formatted_status else None

    def _format_stadium_name(self, stadium: str) -> str:
        """경기장명을 사용자 친화적으로 포맷팅합니다."""
        if not stadium:
            return stadium

        # 경기장명 정규화
        stadium_mapping = {
            "광주": "광주-기아 챔피언스 필드",
            "잠실": "잠실야구장",
            "문학": "인천 SSG 랜더스필드",
            "대구": "대구 삼성 라이온즈 파크",
            "창원": "창원 NC 파크",
            "수원": "수원 KT 위즈 파크",
            "고척": "고척 스카이돔",
            "사직": "사직야구장",
        }

        return stadium_mapping.get(stadium, stadium)

    def _tool_get_season_final_game_date(
        self, year: int, league_type: str = "korean_series"
    ) -> ToolResult:
        """시즌 마지막 경기 정보 조회 도구 (날짜 + 경기 결과)"""
        try:
            # 1단계: 마지막 경기 날짜 조회
            date_result = self.game_query_tool.get_season_final_game_date(
                year, league_type
            )

            if date_result["error"]:
                return ToolResult(
                    success=False,
                    data=date_result,
                    message=f"마지막 경기 날짜 조회 오류: {date_result['error']}",
                )

            if not date_result["found"]:
                return ToolResult(
                    success=True,
                    data=date_result,
                    message=f"{year}년 {league_type}의 마지막 경기 날짜를 찾을 수 없습니다.",
                )

            final_date = date_result["final_game_date"]

            # 2단계: 해당 날짜의 경기 결과 조회
            games_result = self.game_query_tool.get_games_by_date(final_date)

            combined_result = {
                "year": year,
                "league_type": league_type,
                "final_date": final_date,
                "games": games_result.get("games", []),
                "total_games": games_result.get("total_games", 0),
            }

            if games_result.get("found") and games_result.get("games"):
                game_info = []
                formatted_games = []

                # 리그 타입을 한국어로 변환
                league_name_korean = self._format_league_type_to_korean(league_type)

                for game in games_result["games"]:
                    # 팀명 매핑 적용
                    formatted_game = self._format_game_info_with_team_names(game)
                    formatted_games.append(formatted_game)

                    # 팀명을 사용한 게임 정보 생성
                    away_name = formatted_game.get(
                        "away_team_name", formatted_game["away_team"]
                    )
                    home_name = formatted_game.get(
                        "home_team_name", formatted_game["home_team"]
                    )

                    # 경기장명 포맷팅
                    stadium_name = self._format_stadium_name(game.get("stadium", ""))

                    # 게임 상태 포맷팅 (COMPLETED는 표시하지 않음)
                    game_status = self._format_game_status_to_korean(
                        game.get("game_status", "")
                    )

                    # 기본 경기 정보
                    game_desc = f"{away_name} {game['away_score']}-{game['home_score']} {home_name}"

                    # 경기장 정보 추가
                    if stadium_name:
                        game_desc += f" ({stadium_name})"

                    # 상태 정보 추가 (COMPLETED가 아닌 경우에만)
                    if game_status:
                        game_desc += f" - {game_status}"

                    game_info.append(game_desc)

                # combined_result에도 formatted_games 추가
                combined_result["formatted_games"] = formatted_games

                message = (
                    f"{year}년 {league_name_korean} 마지막 경기 ({final_date}):\n"
                    + "\n".join(game_info)
                )
            else:
                league_name_korean = self._format_league_type_to_korean(league_type)
                message = f"{year}년 {league_name_korean}의 마지막 경기 날짜는 {final_date}이지만 경기 상세 정보를 찾을 수 없습니다."

            return ToolResult(success=True, data=combined_result, message=message)

        except Exception as e:
            logger.error(f"Final game tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_game_lineup(
        self, game_id: str = None, date: str = None, team_name: str = None
    ) -> ToolResult:
        """경기 라인업 조회 도구"""
        try:
            result = self.game_query_tool.get_game_lineup(game_id, date, team_name)

            if result.get("error"):
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"라인업 조회 오류: {result['error']}",
                )

            if not result["found"]:
                query_info = f"ID: {game_id}" if game_id else f"날짜: {date}"
                if team_name:
                    query_info += f", 팀: {team_name}"
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"{query_info}에 대한 라인업 정보를 찾을 수 없습니다.",
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"라인업 조회를 성공했습니다. ({len(result['lineups'])}명의 선수)",
            )

        except Exception as e:
            logger.error(f"Lineup tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_player_stats(
        self, player_name: str, year: int, position: str = "both"
    ) -> ToolResult:
        """선수 개별 통계 조회 도구"""
        try:
            result = self.db_query_tool.get_player_season_stats(
                player_name, year, position
            )

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"DB 조회 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"{year}년 '{player_name}' 선수의 기록을 찾을 수 없습니다. 선수명 확인이나 다른 연도를 시도해보세요.",
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"{year}년 {player_name} 선수 통계를 성공적으로 조회했습니다.",
            )

        except Exception as e:
            logger.error(f"Player stats tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_career_stats(
        self, player_name: str, position: str = "both"
    ) -> ToolResult:
        """선수 통산 통계 조회 도구"""
        try:
            result = self.db_query_tool.get_player_career_stats(player_name, position)

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"DB 조회 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"'{player_name}' 선수의 통산 기록을 찾을 수 없습니다. 선수명을 확인해주세요.",
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"{player_name} 선수 통산 통계를 성공적으로 조회했습니다.",
            )

        except Exception as e:
            logger.error(f"Career stats tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _normalize_leaderboard_position(self, position: str) -> str | None:
        normalized = (position or "").strip().lower()
        if normalized in {"batting", "batter", "hitter", "타자"}:
            return "batting"
        if normalized in {"pitching", "pitcher", "투수"}:
            return "pitching"
        return None

    def _tool_get_leaderboard(
        self,
        stat_name: str,
        year: int,
        position: str,
        team_filter: str = None,
        limit: int = 10,
    ) -> ToolResult:
        """리더보드 조회 도구"""
        try:
            normalized_position = self._normalize_leaderboard_position(position)
            if not normalized_position:
                result = {
                    "stat_name": stat_name,
                    "year": year,
                    "position": position,
                    "team_filter": team_filter,
                    "leaderboard": [],
                    "found": False,
                    "total_qualified_players": 0,
                    "error": "get_leaderboard position은 batting 또는 pitching만 허용됩니다.",
                }
                return ToolResult(
                    success=False,
                    data=result,
                    message="도구 가드레일: get_leaderboard position은 batting 또는 pitching만 허용됩니다.",
                )

            result = self.db_query_tool.get_team_leaderboard(
                stat_name, year, normalized_position, team_filter, limit
            )

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"DB 조회 오류: {result['error']}",
                )

            if not result["found"] or not result["leaderboard"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"{year}년 {stat_name} {normalized_position} 리더보드 데이터를 찾을 수 없습니다.",
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"{year}년 {stat_name} {normalized_position} 리더보드를 성공적으로 조회했습니다 (총 {len(result['leaderboard'])}명).",
            )

        except Exception as e:
            logger.error(f"Leaderboard tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_validate_player(self, player_name: str, year: int = None) -> ToolResult:
        """선수 존재 여부 확인 도구"""
        try:
            if year is None:
                import datetime as dt

                year = dt.datetime.now().year
            result = self.db_query_tool.validate_player_exists(player_name, year)

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"DB 조회 오류: {result['error']}",
                )

            return ToolResult(
                success=True,
                data=result,
                message=(
                    f"선수 검색 완료: {len(result['found_players'])}명의 유사한 선수를 찾았습니다."
                    if result["exists"]
                    else "해당 선수를 찾을 수 없습니다."
                ),
            )

        except Exception as e:
            logger.error(f"Player validation tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_team_summary(self, team_name: str, year: int) -> ToolResult:
        """팀 요약 정보 조회 도구"""
        try:
            result = self.db_query_tool.get_team_summary(team_name, year)

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"DB 조회 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"{year}년 {team_name} 팀 데이터를 찾을 수 없습니다.",
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"{year}년 {team_name} 팀 주요 선수 정보를 성공적으로 조회했습니다.",
            )

        except Exception as e:
            logger.error(f"Team summary tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_team_advanced_metrics(self, team_name: str, year: int) -> ToolResult:
        """팀 심층 지표 조회 도구 래퍼"""
        try:
            result = self.db_query_tool.get_team_advanced_metrics(team_name, year)

            if result.get("error"):
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"심층 지표 조회 오류: {result['error']}",
                )

            # AI가 판단하기 쉽게 불펜 과부하 여부를 포함한 메시지 생성
            share = result.get("fatigue_index", {}).get("bullpen_share", "0%")
            avg_share = result.get("league_averages", {}).get("bullpen_share", "0%")

            return ToolResult(
                success=True,
                data=result,
                message=f"{year}년 {team_name} 심층 지표 조회 완료. (불펜 비중: {share}, 리그 평균: {avg_share})",
            )

        except Exception as e:
            logger.error(f"Team advanced metrics tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_position_info(self, position_abbr: str) -> ToolResult:
        """포지션 정보 조회 도구"""
        try:
            result = self.db_query_tool.get_position_info(position_abbr)

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"포지션 조회 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"'{position_abbr}' 포지션 약어를 찾을 수 없습니다.",
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"'{position_abbr}' 포지션 정보: {result['position_name']}",
            )

        except Exception as e:
            logger.error(f"Position info tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_team_basic_info(self, team_name: str) -> ToolResult:
        """팀 기본 정보 조회 도구"""
        try:
            result = self.db_query_tool.get_team_basic_info(team_name)

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"팀 정보 조회 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"'{team_name}' 팀의 기본 정보를 찾을 수 없습니다.",
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"{result['team_name']} 팀 기본 정보를 성공적으로 조회했습니다.",
            )

        except Exception as e:
            logger.error(f"Team basic info tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_defensive_stats(
        self, player_name: str, year: int = None
    ) -> ToolResult:
        """선수 수비 통계 조회 도구"""
        try:
            result = self.db_query_tool.get_player_defensive_stats(player_name, year)

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"수비 통계 조회 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=result["message"],  # 데이터베이스에 없다는 메시지
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"{player_name} 선수 수비 통계를 성공적으로 조회했습니다.",
            )

        except Exception as e:
            logger.error(f"Defensive stats tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_velocity_data(self, player_name: str, year: int = None) -> ToolResult:
        """투수 구속 데이터 조회 도구"""
        try:
            result = self.db_query_tool.get_pitcher_velocity_data(player_name, year)

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"구속 데이터 조회 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=result["message"],  # 데이터베이스에 없다는 메시지
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"{player_name} 선수 구속 데이터를 성공적으로 조회했습니다.",
            )

        except Exception as e:
            logger.error(f"Velocity data tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_search_regulations(self, query: str) -> ToolResult:
        """KBO 규정 검색 도구"""
        try:
            result = self.regulation_query_tool.search_regulation(query)

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"규정 검색 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"'{query}' 관련 규정을 찾을 수 없습니다.",
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"'{query}' 관련 규정을 {result['total_found']}개 찾았습니다.",
            )

        except Exception as e:
            logger.error(f"Regulation search tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_regulations_by_category(self, category: str) -> ToolResult:
        """규정 카테고리별 조회 도구"""
        try:
            result = self.regulation_query_tool.get_regulation_by_category(category)

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"카테고리 조회 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"'{category}' 카테고리의 규정을 찾을 수 없습니다.",
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"'{category}' 카테고리 규정을 {result['total_found']}개 조회했습니다.",
            )

        except Exception as e:
            logger.error(f"Regulation category tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_game_box_score(
        self,
        game_id: str = None,
        date: str = None,
        home_team: str = None,
        away_team: str = None,
    ) -> ToolResult:
        """경기 박스스코어 조회 도구"""
        try:
            result = self.game_query_tool.get_game_box_score(
                game_id, date, home_team, away_team
            )

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"박스스코어 조회 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message="조건에 맞는 경기를 찾을 수 없습니다. 날짜나 팀명을 확인해주세요.",
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"{result['total_games']}개 경기의 박스스코어를 성공적으로 조회했습니다.",
            )

        except Exception as e:
            logger.error(f"Game box score tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_games_by_date(self, date: str, **kwargs) -> ToolResult:
        """날짜별 경기 조회 도구"""
        try:
            team = kwargs.get("team", None)
            result = self.game_query_tool.get_games_by_date(date, team)

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"날짜별 경기 조회 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"{date}에 경기가 없습니다."
                    + (f" ({team} 팀 포함)" if team else ""),
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"{date}에 {result['total_games']}개 경기를 성공적으로 조회했습니다.",
            )

        except Exception as e:
            logger.error(f"Games by date tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_recent_games_by_team(
        self,
        team_name: str,
        limit: int = 5,
        year: int | None = None,
    ) -> ToolResult:
        """팀 최근 경기 조회 도구"""
        try:
            result = self.game_query_tool.get_team_recent_games(team_name, limit, year)

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"최근 경기 조회 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"{team_name} 팀의 최근 경기 기록을 찾을 수 없습니다.",
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"{team_name} 팀의 최근 {len(result['games'])}경기 기록을 성공적으로 조회했습니다.",
            )

        except Exception as e:
            logger.error(f"Team recent games tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_advanced_stats(
        self, player_name: str, year: int, position: str = "both"
    ) -> ToolResult:
        """고급 통계 조회 도구"""
        try:
            result = self.db_query_tool.get_advanced_stats(player_name, year, position)

            if "error" in result and result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"고급 통계 조회 중 오류 발생: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"{year}년에 '{player_name}' 선수의 고급 통계 데이터를 찾을 수 없습니다.",
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"{year}년 {player_name} 선수의 고급 통계(ERA+, OPS+, FIP 등)를 성공적으로 조회했습니다.",
            )

        except Exception as e:
            logger.error(f"Advanced stats tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"고급 통계 도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_head_to_head(
        self, team1: str, team2: str, year: int = None, limit: int = 10
    ) -> ToolResult:
        """팀 간 직접 대결 조회 도구"""
        try:
            result = self.game_query_tool.get_head_to_head(team1, team2, year, limit)

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"팀 간 대결 기록 조회 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"{team1} vs {team2} 맞대결 기록을 찾을 수 없습니다."
                    + (f" ({year}년)" if year else ""),
                )

            summary = result["summary"]
            return ToolResult(
                success=True,
                data=result,
                message=f"{team1} vs {team2} 맞대결: {summary['total_games']}경기 (승부: {summary['team1_wins']}-{summary['team2_wins']})",
            )

        except Exception as e:
            logger.error(f"Head to head tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_player_game_performance(
        self, player_name: str, date: str = None, recent_games: int = 5
    ) -> ToolResult:
        """선수 경기 성적 조회 도구"""
        try:
            result = self.game_query_tool.get_player_game_performance(
                player_name, date, recent_games
            )

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"선수 경기 성적 조회 오류: {result['error']}",
                )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=result.get(
                        "message", f"{player_name} 선수의 경기 성적을 찾을 수 없습니다."
                    ),
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"{player_name} 선수의 {result['total_games']}경기 성적을 성공적으로 조회했습니다.",
            )

        except Exception as e:
            logger.error(f"Player game performance tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"도구 실행 중 오류 발생: {e}"
            )

    def _tool_compare_players(
        self,
        player1: str,
        player2: str,
        comparison_type: str = "career",
        year: int = None,
        position: str = "both",
    ) -> ToolResult:
        """선수 비교 도구"""
        try:
            logger.info(
                f"[BaseballAgent] Comparing players: {player1} vs {player2} ({comparison_type})"
            )

            # 두 선수의 통계를 모두 조회
            if comparison_type == "season" and year:
                # 특정 시즌 비교
                player1_result = self.db_query_tool.get_player_season_stats(
                    player1, year, position
                )
                player2_result = self.db_query_tool.get_player_season_stats(
                    player2, year, position
                )
                comparison_label = f"{year}년 시즌"
            else:
                # 통산 비교
                player1_result = self.db_query_tool.get_player_career_stats(
                    player1, position
                )
                player2_result = self.db_query_tool.get_player_career_stats(
                    player2, position
                )
                comparison_label = "통산"

            # 오류 처리
            if player1_result["error"] or player2_result["error"]:
                return ToolResult(
                    success=False,
                    data={
                        "player1_result": player1_result,
                        "player2_result": player2_result,
                    },
                    message=f"데이터 조회 오류: {player1_result.get('error') or player2_result.get('error')}",
                )

            # 두 선수 중 하나라도 데이터가 없으면 실패
            if not player1_result["found"]:
                return ToolResult(
                    success=True,
                    data={
                        "player1_result": player1_result,
                        "player2_result": player2_result,
                    },
                    message=f"{comparison_label} '{player1}' 선수의 기록을 찾을 수 없습니다.",
                )

            if not player2_result["found"]:
                return ToolResult(
                    success=True,
                    data={
                        "player1_result": player1_result,
                        "player2_result": player2_result,
                    },
                    message=f"{comparison_label} '{player2}' 선수의 기록을 찾을 수 없습니다.",
                )

            # 비교 분석 데이터 구성
            comparison_data = {
                "comparison_type": comparison_label,
                "player1": {"name": player1, "data": player1_result},
                "player2": {"name": player2, "data": player2_result},
                "analysis": self._analyze_player_comparison(
                    player1_result, player2_result, position
                ),
            }

            return ToolResult(
                success=True,
                data=comparison_data,
                message=f"{player1} vs {player2} {comparison_label} 비교 분석 완료",
            )

        except Exception as e:
            logger.error(f"Player comparison tool error: {e}")
            return ToolResult(
                success=False, data={}, message=f"선수 비교 도구 실행 중 오류 발생: {e}"
            )

    def _tool_get_team_rank(self, team_name: str, year: int) -> ToolResult:
        """팀 순위 조회 도구"""
        try:
            request_context = self._current_request_context()
            with request_context.connection.cursor() as cursor:
                # 팀명을 team_id로 매핑
                from ..core.entity_extractor import TEAM_MAPPING

                team_id = None
                for variant, mapped_id in TEAM_MAPPING.items():
                    if variant in team_name:
                        team_id = mapped_id
                        break

                if not team_id:
                    team_id = team_name  # 직접 매핑 실패시 원본 사용

                # v_team_rank_all 뷰 대신 동적 계산 도구 사용
                rank_result = self.db_query_tool.get_team_season_rank(team_name, year)

                if rank_result["found"]:
                    return ToolResult(
                        success=True,
                        data={
                            "team_name": rank_result["team_name"],
                            "team_rank": rank_result["rank"],
                            "season_rank": rank_result["rank"],
                            "wins": rank_result["wins"],
                            "losses": rank_result["losses"],
                            "year": year,
                            "found": True,
                        },
                        message=f"{rank_result['team_name']}의 {year}년 정규시즌 최종 순위: {rank_result['rank']}위 ({rank_result['wins']}승 {rank_result['losses']}패)",
                    )
                else:
                    return ToolResult(
                        success=True,
                        data={"team_name": team_name, "year": year, "found": False},
                        message=f"{team_name}의 {year}년 순위 정보를 찾을 수 없습니다",
                    )

        except Exception as e:
            logger.error(f"[BaseballAgent] Team rank query error: {e}")
            return ToolResult(
                success=False, data={}, message=f"팀 순위 조회 중 오류 발생: {e}"
            )

    def _tool_get_team_by_rank(
        self, year: int, rank: int, season_phase: str = "regular", **kwargs
    ) -> ToolResult:
        """정규시즌 순위 역질의 도구"""
        del kwargs
        try:
            normalized_phase = str(season_phase or "regular").strip().lower()
            if normalized_phase not in {"regular", "regular_season"}:
                return ToolResult(
                    success=True,
                    data={
                        "year": year,
                        "rank": rank,
                        "season_phase": normalized_phase,
                        "found": False,
                    },
                    message="정규시즌 순위 역질의만 지원합니다.",
                )

            rank_result = self.db_query_tool.get_team_by_season_rank(year, rank)
            if rank_result["found"]:
                return ToolResult(
                    success=True,
                    data={
                        "team_name": rank_result["team_name"],
                        "team_code": rank_result["team_code"],
                        "team_rank": rank_result["rank"],
                        "season_rank": rank_result["rank"],
                        "wins": rank_result["wins"],
                        "losses": rank_result["losses"],
                        "draws": rank_result["draws"],
                        "year": year,
                        "found": True,
                    },
                    message=(
                        f"{year}년 정규시즌 {rank_result['rank']}위 팀: "
                        f"{rank_result['team_name']} "
                        f"({rank_result['wins']}승 {rank_result['losses']}패)"
                    ),
                )

            return ToolResult(
                success=True,
                data={"year": year, "rank": rank, "found": False},
                message=f"{year}년 정규시즌 {rank}위 팀 정보를 찾을 수 없습니다",
            )

        except Exception as e:
            logger.error(f"[BaseballAgent] Team by rank query error: {e}")
            return ToolResult(
                success=False, data={}, message=f"순위 기반 팀 조회 중 오류 발생: {e}"
            )

    def _tool_get_team_last_game(self, team_name: str, year: int) -> ToolResult:
        """지능적 팀별 마지막 경기 조회 도구"""
        try:
            # 1단계: 팀 순위 조회로 포스트시즌 진출 여부 확인
            rank_result = self._tool_get_team_rank(team_name, year)
            rank_found = (
                bool(rank_result.data.get("found")) if rank_result.data else False
            )

            if not rank_result.success or not rank_found:
                # 순위 정보를 찾을 수 없는 경우, 기본적으로 한국시리즈로 시도
                logger.warning(
                    f"[BaseballAgent] 팀 순위를 찾을 수 없어 한국시리즈로 조회: {team_name}"
                )
                league_type = "korean_series"
                team_rank = None
            else:
                team_rank = rank_result.data.get("team_rank")
                # 상위 5팀은 포스트시즌 진출, 6위 이하는 정규시즌에서 마무리
                league_type = (
                    "korean_series"
                    if isinstance(team_rank, int) and team_rank <= 5
                    else "regular_season"
                )

            logger.info(
                f"[BaseballAgent] {team_name} {year}년 순위: {team_rank}, 리그 타입: {league_type}"
            )

            # 2단계: 해당 팀의 실제 마지막 경기 날짜 조회
            final_date_result = self.game_query_tool.get_team_last_game_date(
                team_name, year, league_type
            )

            if not final_date_result.get("found"):
                # 팀별 마지막 날짜를 못 찾으면 전체 시즌 마지막 날짜로 폴백
                logger.warning(
                    f"[BaseballAgent] 팀별 마지막 경기 날짜를 못 찾음 ({team_name}, {year}). 전체 시즌 날짜로 시도."
                )
                final_game_result = self.game_query_tool.get_season_final_game_date(
                    year, league_type
                )
                if not final_game_result.get("found"):
                    return ToolResult(
                        success=True,
                        data={"team_name": team_name, "year": year, "found": False},
                        message=f"{team_name}의 {year}년 {league_type} 마지막 경기 정보를 찾을 수 없습니다",
                    )
                final_date = final_game_result["final_game_date"]
            else:
                final_date = final_date_result["last_game_date"]

            # 3단계: 해당 날짜의 경기 결과 조회
            games_result = self.game_query_tool.get_games_by_date(final_date)

            if not games_result.get("found"):
                return ToolResult(
                    success=True,
                    data={
                        "team_name": team_name,
                        "year": year,
                        "team_rank": team_rank,
                        "final_date": final_date,
                        "found": False,
                    },
                    message=f"{team_name}의 {year}년 {league_type} 마지막 경기({final_date}) 결과 조회 실패",
                )

            # 3단계: 해당 팀의 경기만 필터링
            games = games_result.get("games", [])
            team_games = []

            # 팀명을 동적으로 매핑
            from ..core.entity_extractor import extract_team

            team_id = extract_team(team_name)

            if not team_id:
                team_id = team_name  # 매핑 실패시 원본 사용

            for game in games:
                if game.get("home_team") == team_id or game.get("away_team") == team_id:
                    team_games.append(game)

            # 결과 구성
            combined_data = {
                "team_name": team_name,
                "year": year,
                "team_rank": team_rank,
                "league_type": league_type,
                "final_date": final_date,
                "team_games": team_games,
                "all_games": games,
                "postseason_qualified": team_rank <= 5 if team_rank else None,
            }

            # 메시지 생성
            league_name = "한국시리즈" if league_type == "korean_series" else "정규시즌"
            rank_info = f"최종 순위 {team_rank}등" if team_rank else "순위 정보 없음"

            if team_games:
                game_count = len(team_games)
                game_summary = f"{game_count}경기"
                message = f"{team_name}의 {year}년 {league_name} 마지막 경기 조회 완료 ({rank_info}, {game_summary})"
            else:
                message = f"{team_name}의 {year}년 {league_name} 마지막 경기를 찾을 수 없습니다 ({rank_info})"

            return ToolResult(success=True, data=combined_data, message=message)

        except Exception as e:
            logger.error(f"[BaseballAgent] Team last game query error: {e}")
            return ToolResult(
                success=False, data={}, message=f"팀 마지막 경기 조회 중 오류 발생: {e}"
            )

    def _tool_get_korean_series_winner(self, year: int) -> ToolResult:
        """한국시리즈 우승팀 조회 도구"""
        try:
            # 1단계: 한국시리즈 마지막 경기 조회
            final_game_result = self._tool_get_season_final_game_date(
                year, "korean_series"
            )

            if not final_game_result.success:
                return ToolResult(
                    success=False,
                    data={"year": year},
                    message=f"{year}년 한국시리즈 정보 조회 중 오류가 발생했습니다",
                )

            if not final_game_result.data.get("found", True):
                return ToolResult(
                    success=True,
                    data={"year": year, "found": False},
                    message=f"{year}년 한국시리즈 정보를 찾을 수 없습니다",
                )

            games = final_game_result.data.get("formatted_games", [])
            if not games:
                return ToolResult(
                    success=True,
                    data={"year": year, "found": False},
                    message=f"{year}년 한국시리즈 경기 결과를 찾을 수 없습니다",
                )

            # 2단계: 우승팀 식별
            # 한국시리즈는 7전 4선승제이므로 마지막 경기의 승리팀이 우승팀
            final_game = games[-1]  # 마지막 경기

            winner_team_id = None
            winner_team_name = None

            # winning_team 필드가 있으면 사용
            if "winning_team" in final_game:
                winner_team_id = final_game["winning_team"]
            else:
                # 점수 비교로 승리팀 결정
                home_score = final_game.get("home_score", 0)
                away_score = final_game.get("away_score", 0)

                if home_score > away_score:
                    winner_team_id = final_game.get("home_team")
                elif away_score > home_score:
                    winner_team_id = final_game.get("away_team")

            # 팀 ID를 팀명으로 변환
            if winner_team_id:
                winner_team_name = self._convert_team_id_to_name(winner_team_id)

            if not winner_team_name:
                return ToolResult(
                    success=False,
                    data={"year": year, "final_game": final_game},
                    message=f"{year}년 한국시리즈 우승팀을 정확히 식별할 수 없습니다",
                )

            # 우승팀 순위 정보도 함께 조회
            rank_result = self._tool_get_team_rank(winner_team_name, year)
            winner_rank = (
                rank_result.data.get("team_rank") if rank_result.success else None
            )

            result_data = {
                "year": year,
                "winner_team_id": winner_team_id,
                "winner_team_name": winner_team_name,
                "winner_rank": winner_rank,
                "final_game": final_game,
                "series_type": "한국시리즈",
            }

            rank_text = f" (정규시즌 {winner_rank}위)" if winner_rank else ""
            final_score = (
                f"{final_game.get('away_score')}:{final_game.get('home_score')}"
            )
            stadium_text = (
                f" ({final_game.get('stadium')})" if final_game.get("stadium") else ""
            )

            message = f"{year}년 한국시리즈 우승팀: {winner_team_name}{rank_text}. "
            message += f"결승전 결과: {final_game.get('away_team_name')} {final_score} {final_game.get('home_team_name')}{stadium_text} ({final_game.get('game_date')})"

            return ToolResult(success=True, data=result_data, message=message)

        except Exception as e:
            logger.error(f"[BaseballAgent] Korean Series winner query error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"한국시리즈 우승팀 조회 중 오류 발생: {e}",
            )

    @staticmethod
    def _display_award_type(award_type: Optional[str]) -> str:
        award_display_map = {
            "mvp": "MVP",
            "rookie": "신인왕",
            "golden_glove": "골든글러브",
            "korean_series_mvp": "한국시리즈 MVP",
            "all_star_mvp": "올스타전 MVP",
            "any": "수상",
            None: "수상",
            "": "수상",
        }
        return award_display_map.get(award_type, str(award_type))

    def _resolve_award_query_type(
        self, query: str, entity_filter: Any
    ) -> Optional[str]:
        query_lower = query.lower()
        if (
            any(keyword in query for keyword in ["한국시리즈", "코리안시리즈"])
            or "ks mvp" in query_lower
        ) and ("mvp" in query_lower or "엠브이피" in query):
            return "korean_series_mvp"
        if (
            any(keyword in query for keyword in ["올스타", "올스타전"])
            or "all-star mvp" in query_lower
            or "all star mvp" in query_lower
        ) and ("mvp" in query_lower or "엠브이피" in query):
            return "all_star_mvp"

        award_type = getattr(entity_filter, "award_type", None)
        if isinstance(award_type, str) and award_type.strip():
            return award_type

        if any(keyword in query_lower for keyword in ["수상", "수상자"]):
            return "any"

        return None

    def _tool_get_award_winners(self, year: int, award_type: str = None) -> ToolResult:
        """시즌 수상자 조회 도구"""
        try:
            result = self.db_query_tool.get_award_winners(year, award_type)

            if result["error"]:
                return ToolResult(
                    success=False,
                    data=result,
                    message=f"DB 조회 오류: {result['error']}",
                )

            display_award = self._display_award_type(
                result.get("award_type") or award_type
            )

            if not result["found"]:
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"{year}년 {display_award} 수상 정보를 찾을 수 없습니다.",
                )

            awards = result.get("awards", [])
            if len(awards) == 1:
                winner = awards[0]
                team_text = (
                    f" ({winner['team_name']})" if winner.get("team_name") else ""
                )
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"{year}년 {display_award} 수상자는 {winner['player_name']}{team_text}입니다.",
                )

            return ToolResult(
                success=True,
                data=result,
                message=f"{year}년 {display_award} 수상 기록 {len(awards)}건을 조회했습니다.",
            )
        except Exception as e:
            logger.error(f"[BaseballAgent] Award query error: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"수상자 조회 중 오류 발생: {e}",
            )

    def _analyze_player_comparison(
        self, player1_data: Dict, player2_data: Dict, position: str
    ) -> Dict:
        """두 선수 데이터를 분석하여 비교 결과를 생성합니다."""
        analysis = {
            "summary": "",
            "key_stats": {},
            "strengths": {"player1": [], "player2": []},
        }

        try:
            # 타자 비교 분석
            if (
                position in ["batting", "both"]
                and "batting_stats" in player1_data
                and "batting_stats" in player2_data
            ):
                p1_batting = player1_data["batting_stats"]
                p2_batting = player2_data["batting_stats"]

                # 주요 타격 지표 비교
                key_batting_stats = ["avg", "ops", "home_runs", "rbi", "runs", "hits"]

                for stat in key_batting_stats:
                    if stat in p1_batting and stat in p2_batting:
                        p1_val = float(p1_batting[stat] or 0)
                        p2_val = float(p2_batting[stat] or 0)

                        analysis["key_stats"][stat] = {
                            "player1": p1_val,
                            "player2": p2_val,
                            "difference": p1_val - p2_val,
                            "better_player": (
                                "player1"
                                if p1_val > p2_val
                                else "player2" if p2_val > p1_val else "tie"
                            ),
                        }

                        # 장점 분석
                        if p1_val > p2_val:
                            analysis["strengths"]["player1"].append(f"{stat}: {p1_val}")
                        elif p2_val > p1_val:
                            analysis["strengths"]["player2"].append(f"{stat}: {p2_val}")

            # 투수 비교 분석
            if (
                position in ["pitching", "both"]
                and "pitching_stats" in player1_data
                and "pitching_stats" in player2_data
            ):
                p1_pitching = player1_data["pitching_stats"]
                p2_pitching = player2_data["pitching_stats"]

                # 주요 투구 지표 비교 (ERA, WHIP은 낮을수록 좋음)
                key_pitching_stats = [
                    "era",
                    "whip",
                    "wins",
                    "strikeouts",
                    "innings_pitched",
                ]

                for stat in key_pitching_stats:
                    if stat in p1_pitching and stat in p2_pitching:
                        p1_val = float(p1_pitching[stat] or 0)
                        p2_val = float(p2_pitching[stat] or 0)

                        # ERA, WHIP은 낮을수록 좋음
                        if stat in ["era", "whip"]:
                            better_player = (
                                "player1"
                                if p1_val < p2_val
                                else "player2" if p2_val < p1_val else "tie"
                            )
                        else:
                            better_player = (
                                "player1"
                                if p1_val > p2_val
                                else "player2" if p2_val > p1_val else "tie"
                            )

                        analysis["key_stats"][stat] = {
                            "player1": p1_val,
                            "player2": p2_val,
                            "difference": p1_val - p2_val,
                            "better_player": better_player,
                        }

                        # 장점 분석
                        if better_player == "player1":
                            analysis["strengths"]["player1"].append(f"{stat}: {p1_val}")
                        elif better_player == "player2":
                            analysis["strengths"]["player2"].append(f"{stat}: {p2_val}")

            # 요약 생성
            p1_advantages = len(analysis["strengths"]["player1"])
            p2_advantages = len(analysis["strengths"]["player2"])

            if p1_advantages > p2_advantages:
                analysis["summary"] = (
                    f"선수1이 {p1_advantages}개 지표에서 우세, 선수2가 {p2_advantages}개 지표에서 우세"
                )
            elif p2_advantages > p1_advantages:
                analysis["summary"] = (
                    f"선수2가 {p2_advantages}개 지표에서 우세, 선수1이 {p1_advantages}개 지표에서 우세"
                )
            else:
                analysis["summary"] = (
                    f"두 선수 모두 {p1_advantages}개씩 지표에서 우세하여 비슷한 수준"
                )

        except Exception as e:
            logger.error(f"Player comparison analysis error: {e}")
            analysis["summary"] = "비교 분석 중 오류 발생"

        return analysis

    def _is_chitchat(self, query: str) -> bool:
        """간단한 일상 대화인지 키워드 기반으로 확인합니다."""
        query_lower = query.lower().strip()

        # 야구 관련 키워드가 있으면 일상 대화 아님
        baseball_keywords = [
            "우승",
            "챔피언",
            "선수",
            "팀",
            "경기",
            "시즌",
            "년",
            "성적",
            "기록",
            "통산",
            "타율",
            "홈런",
            "투수",
            "타자",
        ]

        if any(keyword in query_lower for keyword in baseball_keywords):
            return False

        # 선수 관련 질문 패턴 ("김도영이 누구야" 같은 질문)
        import re

        if re.search(r"[가-힣]{2,4}(이가|이는|이)?\s*(누구|뭐)", query_lower):
            return False

        # 일상 대화 키워드
        chitchat_keywords = ["안녕", "고마워", "반가워", "도움", "기능"]

        return any(keyword in query_lower for keyword in chitchat_keywords)

    def _get_chitchat_response(self, query: str) -> Optional[str]:
        """미리 정의된 일상 대화 응답을 반환합니다."""
        query_lower = query.lower()
        if "안녕" in query_lower:
            return "안녕하세요! 저는 KBO 리그 데이터 분석가 BEGA입니다. 야구에 대해 궁금한 점이 있으시면 무엇이든 물어보세요!"
        if "누구" in query_lower or "이름이" in query_lower:
            return "저는 KBO 리그 전문 데이터 분석가 'BEGA'입니다. 선수 기록, 경기 결과, 리그 규정 등 야구에 대한 모든 것을 알려드릴 수 있습니다."
        if "고마워" in query_lower:
            return "천만에요! 더 궁금한 점이 있으시면 언제든지 다시 물어보세요."
        if "도움" in query_lower or "기능" in query_lower:
            return """저는 KBO 리그와 관련된 다양한 질문에 답변할 수 있어요. 예를 들어, 다음과 같이 질문해보세요.
- "어제 LG 경기 결과 알려줘"
- "김도영 2024년 성적은 어땠어?"
- "ABS 규정에 대해 설명해줘"
"""
        return None

    def _detect_team_alias_from_query(self, query: str) -> Optional[str]:
        query_upper = (query or "").upper()
        if "KIA" in query_upper or "기아" in query:
            return "KIA"
        if "LG" in query_upper:
            return "LG"
        if "SSG" in query_upper:
            return "SSG"
        if "KT" in query_upper:
            return "KT"
        if "NC" in query_upper:
            return "NC"
        if "키움" in query:
            return "키움"
        if "두산" in query:
            return "두산"
        if "한화" in query:
            return "한화"
        if "롯데" in query:
            return "롯데"
        if "삼성" in query:
            return "삼성"
        return None

    def _is_team_analysis_query(self, query: str, entity_filter: Any) -> bool:
        decision = self._resolve_chat_intent(query, entity_filter)
        return decision.intent == ChatIntent.TEAM_ANALYSIS

    def _is_team_metric_query_text(self, query: str) -> bool:
        query_lower = query.lower()
        team_name = extract_team(query) or self._detect_team_alias_from_query(query)
        if not team_name:
            return False
        if any(token in query_lower for token in ["팀 내", "팀내"]):
            return False
        if any(
            token in query_lower
            for token in [
                "1위",
                "2위",
                "3위",
                "상위",
                "리더",
                "누가",
                "누구",
                "최고",
                "제일",
                "가장",
            ]
        ):
            return False
        return any(token in query_lower for token in ["팀", "구단"]) and any(
            token in query_lower
            for token in [
                "타율",
                "ops",
                "평균자책",
                "평균 자책",
                "평균자책점",
                "방어율",
                "era",
                "홈런",
                "타점",
            ]
        )

    def _resolve_reference_year(self, query: str, entity_filter: Any) -> int:
        now = datetime.now()
        current_year = now.year

        extracted_year = getattr(entity_filter, "season_year", None)
        if isinstance(extracted_year, int):
            return extracted_year

        query_lower = query.lower()
        if "재작년" in query_lower:
            return current_year - 2
        if "작년" in query_lower:
            return current_year - 1

        if any(word in query_lower for word in ["올해", "이번 시즌", "최신"]):
            if now.month <= 3:
                return current_year - 1
            return current_year

        if now.month <= 3:
            return current_year - 1
        return current_year

    def _resolve_chat_intent(self, query: str, entity_filter: Any) -> IntentDecision:
        return self.chat_intent_router.resolve(query, entity_filter)

    def _intent_decision_to_plan(
        self, decision: IntentDecision
    ) -> Optional[Dict[str, Any]]:
        tool_calls = self._prioritize_and_cap_tool_calls(
            decision.tool_calls,
            limit=3 if decision.planner_mode == "fast_path_bundle" else None,
            preserve_input_order=decision.planner_mode == "fast_path_bundle",
        )
        if decision.direct_answer:
            return {
                "analysis": decision.analysis,
                "tool_calls": tool_calls,
                "confidence": decision.confidence,
                "intent": decision.intent.value,
                "planner_mode": decision.planner_mode,
                "search_keywords": [],
                "error": None,
                "direct_answer": decision.direct_answer,
                "grounding_mode": decision.grounding_mode,
                "source_tier": decision.source_tier,
                "fallback_reason": decision.fallback_reason,
            }
        if not tool_calls or decision.intent == ChatIntent.UNKNOWN:
            return None
        return {
            "analysis": decision.analysis,
            "tool_calls": tool_calls,
            "confidence": decision.confidence,
            "intent": decision.intent.value,
            "planner_mode": decision.planner_mode,
            "search_keywords": [],
            "error": None,
            "grounding_mode": decision.grounding_mode,
            "source_tier": decision.source_tier,
            "fallback_reason": decision.fallback_reason,
        }

    def _extract_llm_planner_player_name(self, entity_filter: Any) -> Optional[str]:
        for attr_name in ("player_name", "person_name", "name"):
            value = getattr(entity_filter, attr_name, None)
            if isinstance(value, str) and value.strip():
                normalized = self._normalize_player_name_candidate(value)
                if normalized:
                    return normalized
        return None

    def _extract_llm_planner_player_names(
        self, query: str, entity_filter: Any
    ) -> List[str]:
        player_names: List[str] = []
        seen: set[str] = set()

        delimited_names: List[str] = []
        query_segments = [
            segment.strip() for segment in re.split(r"[,/]", query) if segment.strip()
        ]
        if len(query_segments) >= 2:
            for segment in query_segments[: PLAYER_LLM_MULTI_NAME_CAP * 2]:
                first_token = re.split(r"\s+", segment, maxsplit=1)[0]
                normalized_name = self._normalize_player_name_candidate(first_token)
                if (
                    len(normalized_name) < 2
                    or normalized_name in INVALID_PLAYER_NAME_TOKENS
                    or normalized_name in delimited_names
                ):
                    continue
                delimited_names.append(normalized_name)
                if len(delimited_names) >= PLAYER_LLM_MULTI_NAME_CAP:
                    break

        explicit_player_name = self._extract_llm_planner_player_name(entity_filter)
        extracted_names = (
            delimited_names
            if len(delimited_names) >= 2
            else extract_player_names(query, limit=PLAYER_LLM_MULTI_NAME_CAP)
        )
        for raw_name in [explicit_player_name, *extracted_names]:
            name = self._normalize_player_name_candidate(raw_name)
            if len(name) < 2 or name in INVALID_PLAYER_NAME_TOKENS or name in seen:
                continue
            player_names.append(name)
            seen.add(name)
            if len(player_names) >= PLAYER_LLM_MULTI_NAME_CAP:
                break

        return player_names

    def _has_explicit_player_name(self, entity_filter: Any) -> bool:
        player_name = self._extract_llm_planner_player_name(entity_filter)
        return (
            isinstance(player_name, str)
            and len(player_name) >= 2
            and player_name not in INVALID_PLAYER_NAME_TOKENS
        )

    def _build_team_llm_planner_prompt(
        self,
        *,
        query_text: str,
        current_date: str,
        current_year: int,
        team_name: str,
        entity_context: str,
    ) -> str:
        return (
            "너는 KBO 도구 라우터다. JSON 한 줄만 출력한다.\n"
            "목표: 팀 분석 질문에 필요한 최소 도구만 계획한다.\n"
            "규칙: "
            "tool_calls 최대 2개. "
            "허용 도구: get_team_summary, get_team_advanced_metrics, get_team_rank, get_team_last_game. "
            "우선순위: get_team_summary, get_team_advanced_metrics. "
            "선수명이 없으면 선수 도구 금지. "
            "parameters.year는 질문 연도 없으면 기본 시즌 연도 사용.\n"
            "출력: "
            '{"analysis":"짧게","tool_calls":[{"tool_name":"...","parameters":{}}],"expected_result":"짧게"}'
            "\n"
            "analysis와 expected_result는 12자 이내로 짧게 쓰고 설명문은 금지한다.\n"
            f"현재 날짜: {current_date}\n"
            f"기본 시즌 연도: {current_year}\n"
            f"팀: {team_name}\n"
            f"질문: {query_text}{entity_context}\n"
        )

    def _build_player_llm_planner_prompt(
        self,
        *,
        query_text: str,
        current_date: str,
        current_year: int,
        player_name: str,
        player_names: List[str],
        entity_context: str,
    ) -> str:
        prefers_career = any(
            token in query_text.lower() for token in ["통산", "커리어", "총"]
        )
        primary_tool = "get_career_stats" if prefers_career else "get_player_stats"
        secondary_tool = "get_player_stats" if prefers_career else "get_career_stats"
        tool_call_cap = max(2, min(PLAYER_LLM_MULTI_NAME_CAP, len(player_names or [])))
        multi_player_rules = ""
        player_scope_line = f"선수: {player_name}\n"
        if len(player_names) >= 2:
            joined_player_names = ", ".join(player_names)
            multi_player_rules = (
                f"질문 등장 선수: {joined_player_names}. "
                "여러 선수 질문이면 각 선수마다 별도 tool_call을 만든다. "
                "player_names 배열은 금지하고 각 tool_call에는 player_name 하나만 넣는다. "
                "질문에 나온 선수명을 누락하지 마라. "
            )
            player_scope_line = f"선수들: {joined_player_names}\n"
        return (
            "너는 KBO 도구 라우터다. JSON 한 줄만 출력한다.\n"
            "목표: 선수 분석 질문에 필요한 최소 도구만 계획한다.\n"
            "규칙: "
            f"tool_calls 최대 {tool_call_cap}개. "
            "허용 도구: validate_player, get_player_stats, get_career_stats. "
            f"우선순위: {primary_tool}, {secondary_tool}. "
            "이름이 불확실할 때만 validate_player 사용. "
            "통산/커리어/총 질문이면 get_career_stats 우선. "
            "그 외에는 get_player_stats 우선, year는 질문 연도 없으면 기본 시즌 연도 사용. "
            "팀/리더보드/규정/문서 검색 도구는 금지. "
            f"{multi_player_rules}\n"
            "출력: "
            '{"analysis":"짧게","tool_calls":[{"tool_name":"...","parameters":{}}],"expected_result":"짧게"}'
            "\n"
            "analysis와 expected_result는 12자 이내로 짧게 쓰고 설명문은 금지한다.\n"
            f"현재 날짜: {current_date}\n"
            f"기본 시즌 연도: {current_year}\n"
            f"{player_scope_line}"
            f"질문: {query_text}{entity_context}\n"
        )

    def _resolve_llm_planner_max_tokens(self, planner_mode: str) -> Optional[int]:
        analysis_max_tokens = (
            self.chat_analysis_max_tokens if self.chat_dynamic_token_enabled else None
        )
        if not isinstance(analysis_max_tokens, int):
            return analysis_max_tokens
        if planner_mode in {"team_llm_planner", "player_llm_planner"}:
            return min(COMPACT_LLM_PLANNER_TOKEN_CAP, analysis_max_tokens)
        return analysis_max_tokens

    def _resolve_llm_planner_timeout_seconds(self, planner_mode: str) -> float:
        default_timeout = max(
            0.1,
            float(
                getattr(
                    self,
                    "chat_planner_timeout_seconds",
                    DEFAULT_LLM_PLANNER_TIMEOUT_SECONDS,
                )
            ),
        )
        compact_timeout = max(
            0.1,
            float(
                getattr(
                    self,
                    "chat_compact_planner_timeout_seconds",
                    COMPACT_LLM_PLANNER_TIMEOUT_SECONDS,
                )
            ),
        )
        if planner_mode in {"team_llm_planner", "player_llm_planner"}:
            return min(default_timeout, compact_timeout)
        return default_timeout

    def _select_llm_planner_prompt(
        self,
        *,
        query_text: str,
        query: str,
        entity_filter: Any,
        current_date: str,
        current_year: int,
        entity_context: str,
    ) -> tuple[str, str]:
        is_team_query = self._is_team_analysis_query(query_text, entity_filter)
        detected_team = (
            getattr(entity_filter, "team_id", None)
            or extract_team(query_text)
            or self._detect_team_alias_from_query(query_text)
        )
        query_player_names = self._extract_llm_planner_player_names(
            query_text, entity_filter
        )
        reference_year = self._resolve_reference_year(query_text, entity_filter)
        if is_team_query and detected_team:
            return (
                self._build_team_llm_planner_prompt(
                    query_text=query_text,
                    current_date=current_date,
                    current_year=reference_year,
                    team_name=detected_team,
                    entity_context=entity_context,
                ),
                "team_llm_planner",
            )

        if (
            self._has_explicit_player_name(entity_filter)
            or len(query_player_names) >= 2
        ):
            player_name = (
                query_player_names[0]
                if query_player_names
                else self._extract_llm_planner_player_name(entity_filter)
            )
            return (
                self._build_player_llm_planner_prompt(
                    query_text=query_text,
                    current_date=current_date,
                    current_year=reference_year,
                    player_name=player_name,
                    player_names=query_player_names or [player_name],
                    entity_context=entity_context,
                ),
                "player_llm_planner",
            )

        return (
            SYSTEM_PROMPT.format(
                current_date=current_date,
                current_year=current_year,
                last_year=current_year - 1,
                two_years_ago=current_year - 2,
                query_text=query_text,
                query=query,
            ),
            "default_llm_planner",
        )

    def _coerce_tool_call(self, raw_call: Any) -> Optional[ToolCall]:
        if isinstance(raw_call, ToolCall):
            return raw_call
        if not isinstance(raw_call, dict):
            return None

        tool_name = raw_call.get("tool_name")
        parameters = raw_call.get("parameters", {})

        if not isinstance(tool_name, str) or not tool_name.strip():
            return None
        if not isinstance(parameters, dict):
            return None

        return ToolCall(tool_name=tool_name, parameters=parameters)

    def _extract_player_name_batch(self, parameters: Dict[str, Any]) -> List[str]:
        raw_names = parameters.get("player_names")
        if not isinstance(raw_names, list):
            return []

        normalized: List[str] = []
        seen: set[str] = set()
        for value in raw_names:
            name = self._normalize_player_name_candidate(value)
            if len(name) < 2 or name in INVALID_PLAYER_NAME_TOKENS or name in seen:
                continue
            normalized.append(name)
            seen.add(name)
            if len(normalized) >= PLAYER_LLM_MULTI_NAME_CAP:
                break
        return normalized

    def _coerce_tool_calls(self, raw_call: Any) -> List[ToolCall]:
        call = self._coerce_tool_call(raw_call)
        if call is None:
            return []

        if call.tool_name not in MULTI_PLAYER_EXPANDABLE_TOOLS:
            return [call]

        player_name = str(call.parameters.get("player_name") or "").strip()
        if player_name:
            return [call]

        player_names = self._extract_player_name_batch(call.parameters)
        if not player_names:
            return [call]

        expanded_calls: List[ToolCall] = []
        for name in player_names:
            expanded_parameters = {
                key: value
                for key, value in call.parameters.items()
                if key != "player_names"
            }
            expanded_parameters["player_name"] = name
            expanded_calls.append(
                ToolCall(tool_name=call.tool_name, parameters=expanded_parameters)
            )

        logger.info(
            "[Planner] expand multi-player tool_call tool=%s count=%d",
            call.tool_name,
            len(expanded_calls),
        )
        return expanded_calls

    def _supplement_missing_player_tool_calls(
        self,
        filtered: List[ToolCall],
        *,
        query: str,
        entity_filter: Any,
        query_player_names: List[str],
        is_career_query: bool,
        tool_cap: int,
    ) -> List[ToolCall]:
        if len(query_player_names) < 2 or tool_cap <= 0:
            return filtered

        preferred_tool_name = (
            "get_career_stats" if is_career_query else "get_player_stats"
        )
        alternate_tool_name = (
            "get_player_stats" if is_career_query else "get_career_stats"
        )

        template_parameters: Optional[Dict[str, Any]] = None
        covered_names: set[str] = set()
        for tool_name in (preferred_tool_name, alternate_tool_name):
            for call in filtered:
                if call.tool_name != tool_name:
                    continue
                template_parameters = dict(call.parameters)
                preferred_tool_name = tool_name
                break
            if template_parameters is not None:
                break

        if template_parameters is None:
            template_parameters = {
                "position": getattr(entity_filter, "position_type", None) or "both",
            }
            if preferred_tool_name == "get_player_stats":
                template_parameters["year"] = self._resolve_reference_year(
                    query, entity_filter
                )

        for call in filtered:
            if call.tool_name != preferred_tool_name:
                continue
            player_name = str(call.parameters.get("player_name") or "").strip()
            if player_name:
                covered_names.add(player_name)

        supplemented = list(filtered)
        appended = 0
        for player_name in query_player_names:
            if len(covered_names) >= tool_cap:
                break
            if player_name in covered_names:
                continue
            parameters = {
                key: value
                for key, value in template_parameters.items()
                if key != "player_names"
            }
            parameters["player_name"] = player_name
            supplemented.append(
                ToolCall(tool_name=preferred_tool_name, parameters=parameters)
            )
            covered_names.add(player_name)
            appended += 1

        if appended:
            logger.info(
                "[Planner] supplemented multi-player tool_calls added=%d covered=%d",
                appended,
                len(covered_names),
            )

        return supplemented

    def _apply_reference_year_default(
        self, call: ToolCall, *, query: str, entity_filter: Any
    ) -> ToolCall:
        if call.tool_name not in REFERENCE_YEAR_DEFAULT_TOOLS:
            return call

        year = call.parameters.get("year")
        reference_year = self._resolve_reference_year(query, entity_filter)
        current_year = datetime.now().year
        normalized_year = year
        if isinstance(normalized_year, str):
            match = re.search(r"\d{4}", normalized_year)
            if match:
                normalized_year = int(match.group(0))

        if normalized_year not in (None, ""):
            if (
                isinstance(normalized_year, int)
                and reference_year < current_year
                and normalized_year == current_year
            ):
                normalized_parameters = dict(call.parameters)
                normalized_parameters["year"] = reference_year
                return ToolCall(
                    tool_name=call.tool_name, parameters=normalized_parameters
                )
            return call

        normalized_parameters = dict(call.parameters)
        normalized_parameters["year"] = reference_year
        return ToolCall(tool_name=call.tool_name, parameters=normalized_parameters)

    def _prioritize_and_cap_tool_calls(
        self,
        tool_calls: List[Any],
        *,
        limit: Optional[int] = None,
        preferred_order: Optional[List[str]] = None,
        preserve_input_order: bool = False,
    ) -> List[ToolCall]:
        if not tool_calls:
            return []

        preferred_order = preferred_order or [
            "get_team_summary",
            "get_team_advanced_metrics",
        ]
        preferred_rank = {name: idx for idx, name in enumerate(preferred_order)}

        deduped: List[ToolCall] = []
        seen = set()
        for raw_call in tool_calls:
            coerced_calls = self._coerce_tool_calls(raw_call)
            if not coerced_calls:
                logger.warning("[Planner] skip invalid tool_call format: %s", raw_call)
                continue
            for call in coerced_calls:
                key = (
                    call.tool_name,
                    json.dumps(call.parameters, ensure_ascii=False, sort_keys=True),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(call)

        if not preserve_input_order:
            deduped.sort(
                key=lambda c: (
                    preferred_rank.get(c.tool_name, len(preferred_order)),
                    c.tool_name,
                )
            )
        effective_limit = max(1, int(limit or self.fast_path_tool_cap))
        return deduped[:effective_limit]

    def _build_team_fast_path_tool_calls(
        self, query: str, team_name: str, year: int
    ) -> List[ToolCall]:
        query_lower = query.lower()
        tool_calls: List[ToolCall] = [
            ToolCall(
                tool_name="get_team_summary",
                parameters={"team_name": team_name, "year": year},
            )
        ]
        advanced_needed_keywords = [
            "불펜",
            "리스크",
            "강점",
            "약점",
            "전력",
            "가을야구",
            "플레이오프",
            "플옵",
            "페이스",
            "흐름",
            "득점",
            "타선",
            "선발",
            "선발진",
            "수비",
            "상대 전적",
            "상대전적",
            "최근",
            "5경기",
            "10경기",
            "살아난",
            "식은",
            "올라오는",
            "내려가는",
            "믿고",
            "팀 타율",
            "타율",
            "팀 ops",
            "ops",
            "평균자책",
            "평균 자책",
            "평균자책점",
            "방어율",
            "era",
            "홈런",
            "타점",
        ]
        if any(keyword in query_lower for keyword in advanced_needed_keywords):
            tool_calls.append(
                ToolCall(
                    tool_name="get_team_advanced_metrics",
                    parameters={"team_name": team_name, "year": year},
                )
            )
        return self._prioritize_and_cap_tool_calls(tool_calls)

    def _build_player_fast_path_tool_calls(
        self,
        query: str,
        player_names: List[str],
        year: int,
        entity_filter: Any,
    ) -> List[ToolCall]:
        query_lower = query.lower()
        position = getattr(entity_filter, "position_type", None) or "both"
        if any(
            token in query_lower
            for token in [
                "투수",
                "선발",
                "불펜",
                "마무리",
                "로테이션",
                "구위",
                "에이스",
            ]
        ):
            position = "pitching"
        elif any(
            token in query_lower
            for token in [
                "타자",
                "타선",
                "장타",
                "포수",
                "내야",
                "클린업",
                "출루",
                "컨택",
                "홈런",
            ]
        ):
            position = "batting"
        tool_calls = [
            ToolCall(
                tool_name="get_player_stats",
                parameters={
                    "player_name": self._normalize_player_name_candidate(player_name),
                    "year": year,
                    "position": position,
                },
            )
            for player_name in player_names
            if self._normalize_player_name_candidate(player_name)
        ]
        return self._prioritize_and_cap_tool_calls(
            tool_calls,
            limit=max(2, len(tool_calls)),
            preserve_input_order=True,
        )

    def _is_player_fast_path_blocked_keyword(
        self, query_lower: str, keyword: str
    ) -> bool:
        start_index = query_lower.find(keyword)
        while start_index != -1:
            trailing_window = query_lower[start_index : start_index + len(keyword) + 18]
            if "말고" not in trailing_window and "제외" not in trailing_window:
                return True
            start_index = query_lower.find(keyword, start_index + len(keyword))
        return False

    def _collect_multi_player_stats_payloads(
        self, tool_results: List[ToolResult]
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        seen_requested: set[str] = set()

        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue
            data = result.data
            is_player_stats_payload = (
                "batting_stats" in data
                or "pitching_stats" in data
                or data.get("batch_lookup") is True
                or (
                    "found" in data
                    and "year" in data
                    and "player_name" in data
                    and "found_players" not in data
                )
            )
            if not is_player_stats_payload:
                continue

            requested_name = self._normalize_player_name_candidate(
                data.get("requested_player_name") or data.get("player_name")
            )
            if len(requested_name) < 2 or requested_name in seen_requested:
                continue

            batting_stats = (
                data.get("batting_stats")
                if isinstance(data.get("batting_stats"), dict)
                else None
            )
            pitching_stats = (
                data.get("pitching_stats")
                if isinstance(data.get("pitching_stats"), dict)
                else None
            )
            resolved_name = self._normalize_player_name_candidate(
                data.get("resolved_player_name")
                or (batting_stats or {}).get("player_name")
                or (pitching_stats or {}).get("player_name")
                or requested_name
            )
            team_name = (batting_stats or {}).get("team_name") or (
                pitching_stats or {}
            ).get("team_name")

            entries.append(
                {
                    "requested_name": requested_name,
                    "player_name": resolved_name or requested_name,
                    "team_name": team_name,
                    "found": bool(data.get("found")),
                    "batting_stats": batting_stats,
                    "pitching_stats": pitching_stats,
                }
            )
            seen_requested.add(requested_name)

        return entries

    def _classify_multi_player_narrative_query(
        self, query: str, *, pitcher_mode: bool
    ) -> str:
        query_lower = query.lower()
        if any(token in query_lower for token in ["성장 신호", "성장", "젊은", "표본"]):
            return "growth_signals"
        if any(token in query_lower for token in ["안정감", "폭발력", "균형"]):
            return "stability_vs_explosiveness"
        if pitcher_mode and any(
            token in query_lower
            for token in ["로테이션", "구위", "압박", "하위 유형", "유형", "선발 운영"]
        ):
            return "pitcher_archetype"
        if any(token in query_lower for token in ["장타", "파워", "홈런", "장타 생산"]):
            return "power_profile"
        if any(
            token in query_lower
            for token in ["흐름", "연결축", "리더", "포수", "리드", "역할", "가치"]
        ):
            return "game_context_role"
        return "generic_grouping"

    def _multi_player_entry_metric(
        self, entry: Dict[str, Any], keys: List[str]
    ) -> Optional[Any]:
        batting = entry.get("batting_stats") or {}
        pitching = entry.get("pitching_stats") or {}
        for source in (batting, pitching):
            value = self._pick_metric_value(source, keys)
            if value is not None:
                return value
        return None

    def _multi_player_batter_style_line(self, entry: Dict[str, Any]) -> str:
        batting = entry.get("batting_stats") or {}
        player_name = entry.get("player_name") or entry.get("requested_name") or "선수"
        avg = self._pick_metric_value(batting, ["avg", "batting_avg"])
        obp = self._pick_metric_value(batting, ["obp"])
        slg = self._pick_metric_value(batting, ["slg"])
        ops = self._pick_metric_value(batting, ["ops"])
        home_runs = self._pick_metric_value(batting, ["home_runs", "hr"])
        doubles = self._pick_metric_value(batting, ["doubles"])
        walks = self._pick_metric_value(batting, ["walks"])
        strikeouts = self._pick_metric_value(batting, ["strikeouts"])
        plate_appearances = self._pick_metric_value(
            batting, ["plate_appearances", "pa"]
        )

        try:
            obp_value = float(obp) if obp is not None else None
        except (TypeError, ValueError):
            obp_value = None
        try:
            slg_value = float(slg) if slg is not None else None
        except (TypeError, ValueError):
            slg_value = None
        try:
            avg_value = float(avg) if avg is not None else None
        except (TypeError, ValueError):
            avg_value = None

        traits: List[str] = []
        if obp_value is not None and obp_value >= 0.37:
            traits.append("출루 안정감")
        if slg_value is not None and slg_value >= 0.5:
            traits.append("장타 폭발력")
        if avg_value is not None and avg_value >= 0.3:
            traits.append("정교한 컨택")
        if doubles not in (None, "", 0):
            traits.append("갭파워")
        if not traits:
            traits.append("균형형")

        metrics: List[str] = []
        for label, value in (
            ("AVG", avg),
            ("OBP", obp),
            ("SLG", slg),
            ("OPS", ops),
            ("HR", home_runs),
            ("2루타", doubles),
            ("BB", walks),
            ("K", strikeouts),
            ("PA", plate_appearances),
        ):
            if value not in (None, ""):
                metrics.append(f"{label} {self._format_deterministic_metric(value)}")
            if len(metrics) >= 4:
                break

        metrics_text = ", ".join(metrics) if metrics else "표면 지표 확인"
        team_name = self._format_deterministic_metric(entry.get("team_name"))
        team_prefix = "" if team_name == "확인 불가" else f"{team_name} "
        return (
            f"{team_prefix}{player_name}은 {', '.join(traits[:2])} 쪽이 먼저 읽히고, "
            f"확인된 수치는 {metrics_text}입니다."
        )

    def _multi_player_pitcher_style_line(self, entry: Dict[str, Any]) -> str:
        pitching = entry.get("pitching_stats") or {}
        player_name = entry.get("player_name") or entry.get("requested_name") or "선수"
        era = self._pick_metric_value(pitching, ["era"])
        whip = self._pick_metric_value(pitching, ["whip"])
        strikeouts = self._pick_metric_value(pitching, ["strikeouts", "so", "k"])
        innings = self._pick_metric_value(
            pitching, ["innings_pitched", "ip", "innings"]
        )
        games_started = self._pick_metric_value(pitching, ["games_started", "gs"])
        saves = self._pick_metric_value(pitching, ["saves", "save"])
        holds = self._pick_metric_value(pitching, ["holds", "hold"])

        try:
            innings_value = float(innings) if innings is not None else None
            strikeout_value = float(strikeouts) if strikeouts is not None else None
            games_started_value = (
                float(games_started) if games_started is not None else None
            )
            saves_value = float(saves) if saves is not None else 0.0
            holds_value = float(holds) if holds is not None else 0.0
        except (TypeError, ValueError):
            innings_value = strikeout_value = games_started_value = None
            saves_value = holds_value = 0.0

        traits: List[str] = []
        if saves_value + holds_value >= 8 and (games_started_value or 0) < 5:
            traits.append("불펜 압박형")
        if (
            innings_value is not None
            and innings_value >= 100
            or games_started_value is not None
            and games_started_value >= 16
        ):
            traits.append("선발 소화형")
        if (
            strikeout_value is not None
            and innings_value not in (None, 0)
            and strikeout_value / innings_value >= 1.0
        ):
            traits.append("구위 압박형")
        if not traits:
            traits.append("운영형")
        traits = list(dict.fromkeys(traits))

        metrics: List[str] = []
        for label, value in (
            ("ERA", era),
            ("WHIP", whip),
            ("K", strikeouts),
            ("IP", innings),
            ("GS", games_started),
            ("SV", saves),
            ("HLD", holds),
        ):
            if value not in (None, ""):
                metrics.append(f"{label} {self._format_deterministic_metric(value)}")
            if len(metrics) >= 4:
                break

        metrics_text = ", ".join(metrics) if metrics else "표면 지표 확인"
        team_name = self._format_deterministic_metric(entry.get("team_name"))
        team_prefix = "" if team_name == "확인 불가" else f"{team_name} "
        return (
            f"{team_prefix}{player_name}은 {', '.join(traits[:2])} 쪽으로 읽히고, "
            f"확인된 수치는 {metrics_text}입니다."
        )

    def _build_multi_player_narrative_answer(
        self,
        query: str,
        tool_results: List[ToolResult],
        *,
        chat_mode: bool,
    ) -> Optional[str]:
        entries = self._collect_multi_player_stats_payloads(tool_results)
        if len(entries) < 2:
            return None

        usable_entries = [
            entry
            for entry in entries
            if entry.get("found")
            and (entry.get("batting_stats") or entry.get("pitching_stats"))
        ]
        if len(usable_entries) < 2:
            return None

        query_entities = extract_entities_from_query(query)
        query_player_names = self._extract_llm_planner_player_names(
            query, query_entities
        )
        missing_names = [
            name
            for name in query_player_names
            if not any(
                entry.get("requested_name") == name and entry.get("found")
                for entry in entries
            )
        ]

        batting_entries = [
            entry for entry in usable_entries if entry.get("batting_stats")
        ]
        pitching_entries = [
            entry for entry in usable_entries if entry.get("pitching_stats")
        ]
        query_lower = query.lower()
        pitcher_mode = bool(
            any(
                token in query_lower
                for token in ["투수", "구위", "로테이션", "선발", "불펜"]
            )
            or (
                len(pitching_entries) >= 2
                and len(pitching_entries) >= len(batting_entries)
            )
        )
        selected_entries = pitching_entries if pitcher_mode else batting_entries
        if len(selected_entries) < 2:
            return None

        narrative_kind = self._classify_multi_player_narrative_query(
            query, pitcher_mode=pitcher_mode
        )
        display_names = [
            entry.get("player_name") or entry.get("requested_name")
            for entry in selected_entries
        ]
        player_group = ", ".join(display_names)

        if pitcher_mode:
            intro_map = {
                "pitcher_archetype": (
                    f"{player_group}처럼 강한 구위를 공유하는 투수 묶음도 실제 역할은 다릅니다. "
                    "선발 이닝 소화, 높은 레버리지 불펜 압박, 실점 억제 안정감 중 어디가 먼저 보이는지로 나누는 편이 빠릅니다."
                ),
                "game_context_role": (
                    f"{player_group}은 같은 파워형 투수라도 경기 운영에서 가치가 커지는 장면이 다릅니다."
                ),
                "generic_grouping": (
                    f"{player_group}은 같은 파워 축으로 보여도 이닝 소화형, 압박형, 불펜형으로 결이 갈립니다."
                ),
            }
            intro = intro_map.get(narrative_kind, intro_map["generic_grouping"])
            detail_lines = [
                self._multi_player_pitcher_style_line(entry)
                for entry in selected_entries[:PLAYER_LLM_MULTI_NAME_CAP]
            ]
            insight_lines = [
                "- 투수 묶음은 ERA 하나보다 WHIP, 탈삼진, 이닝/보직을 같이 봐야 유형 분리가 안정적입니다.",
                "- 같은 구위형이라도 선발과 불펜은 가치가 커지는 타이밍이 달라서 역할 축을 함께 보는 편이 안전합니다.",
            ]
        else:
            intro_map = {
                "growth_signals": (
                    f"{player_group}처럼 젊은 타자를 볼 때는 결과값보다 먼저 "
                    "1) 타석 접근 안정감, 2) 장타 신호, 3) 최종 생산성 순서로 읽는 편이 안전합니다."
                ),
                "stability_vs_explosiveness": (
                    f"{player_group}을 한 묶음으로 보면 안정감은 출루/삼진 관리에서, 폭발력은 장타율과 홈런 생산에서 갈립니다."
                ),
                "power_profile": (
                    f"{player_group}의 장타 생산은 같은 홈런 수보다 장타율, 2루타, 출루 동반 여부까지 함께 봐야 결이 분명해집니다."
                ),
                "game_context_role": (
                    f"{player_group}은 같은 타자 묶음이어도 경기 흐름을 잇는 출루 축, 장면을 뒤집는 장타 축, 높은 레버리지 대응 축으로 역할이 갈립니다."
                ),
                "generic_grouping": (
                    f"{player_group}은 같은 묶음으로 보여도 출루 안정감, 장타 폭발력, 변동성 관리에서 차이가 납니다."
                ),
            }
            intro = intro_map.get(narrative_kind, intro_map["generic_grouping"])
            detail_lines = [
                self._multi_player_batter_style_line(entry)
                for entry in selected_entries[:PLAYER_LLM_MULTI_NAME_CAP]
            ]
            insight_lines = [
                "- 타자 묶음은 타율보다 OBP/SLG, 홈런/2루타, 삼진/볼넷 방향을 같이 봐야 과장 없이 해석할 수 있습니다.",
                "- 표본이 짧을수록 타점 같은 결과값보다 접근 안정감과 장타 신호를 먼저 읽는 편이 안전합니다.",
            ]

        if missing_names:
            detail_lines.append(
                f"현재 시즌 매칭이 안 된 선수는 {', '.join(missing_names)}이라 이번 정리에서는 제외했습니다."
            )

        table_rows: List[str] = []
        for entry in selected_entries[:PLAYER_LLM_MULTI_NAME_CAP]:
            player_name = (
                entry.get("player_name") or entry.get("requested_name") or "선수"
            )
            if pitcher_mode:
                era = self._format_deterministic_metric(
                    self._multi_player_entry_metric(entry, ["era"])
                )
                whip = self._format_deterministic_metric(
                    self._multi_player_entry_metric(entry, ["whip"])
                )
                strikeouts = self._format_deterministic_metric(
                    self._multi_player_entry_metric(entry, ["strikeouts", "so", "k"])
                )
                table_rows.append(f"| {player_name} | {era} | {whip} | {strikeouts} |")
            else:
                obp = self._format_deterministic_metric(
                    self._multi_player_entry_metric(entry, ["obp"])
                )
                slg = self._format_deterministic_metric(
                    self._multi_player_entry_metric(entry, ["slg"])
                )
                home_runs = self._format_deterministic_metric(
                    self._multi_player_entry_metric(entry, ["home_runs", "hr"])
                )
                table_rows.append(f"| {player_name} | {obp} | {slg} | {home_runs} |")

        if chat_mode:
            lines = [intro, *detail_lines[:3]]
            if missing_names:
                lines.append(
                    f"빠진 선수는 {', '.join(missing_names)}이고, 확인된 선수 기준으로만 정리했습니다."
                )
            return "\n\n".join(lines[:5])

        if pitcher_mode:
            table_header = (
                "| 선수 | ERA | WHIP | 탈삼진 |\n" "| --- | --- | --- | --- |\n"
            )
        else:
            table_header = "| 선수 | OBP | SLG | 홈런 |\n" "| --- | --- | --- | --- |\n"

        return (
            "## 요약\n"
            f"{intro}\n\n"
            "## 상세 내역\n"
            f"{chr(10).join('- ' + line for line in detail_lines[:4])}\n\n"
            "## 핵심 지표\n"
            f"{table_header}{chr(10).join(table_rows[:4])}\n\n"
            "## 인사이트\n"
            f"{chr(10).join(insight_lines[:2])}\n"
            "출처: DB 조회 결과"
        )

    def _should_defer_incomplete_multi_player_answer(
        self, query: str, tool_results: List[ToolResult]
    ) -> bool:
        query_player_names = self._extract_llm_planner_player_names(
            query, extract_entities_from_query(query)
        )
        if len(query_player_names) < 2:
            return False

        entries = self._collect_multi_player_stats_payloads(tool_results)
        if not entries:
            return False

        usable_entries = [
            entry
            for entry in entries
            if entry.get("found")
            and (entry.get("batting_stats") or entry.get("pitching_stats"))
        ]
        return len(usable_entries) < 2

    def _build_reference_fast_path_plan(
        self, query: str, entity_filter: Any
    ) -> Optional[Dict[str, Any]]:
        decision = self._resolve_chat_intent(query, entity_filter)
        if decision.intent != ChatIntent.UNKNOWN:
            return self._intent_decision_to_plan(decision)

        query_lower = query.lower()
        year = self._resolve_reference_year(query, entity_filter)
        team_name = (
            getattr(entity_filter, "team_id", None)
            or extract_team(query)
            or self._detect_team_alias_from_query(query)
        )
        query_player_names = self._extract_llm_planner_player_names(
            query, entity_filter
        )
        player_name = (
            getattr(entity_filter, "player_name", None)
            or getattr(entity_filter, "person_name", None)
            or getattr(entity_filter, "name", None)
        )

        regulation_keywords = [
            "규정",
            "규칙",
            "판정",
            "스트라이크존",
            "fa",
            "보상선수",
            "보상 선수",
            "등록일수",
            "선수 등록",
            "드래프트",
            "엔트리",
            "로스터",
            "외국인 선수",
            "외국인선수",
            "부상자",
            "il",
            "육성선수",
            "육성 선수",
            "군보류",
            "임의해지",
            "특수 신분",
        ]
        if any(keyword in query_lower for keyword in regulation_keywords):
            return {
                "analysis": "규정성 질문으로 판단되어 규정 검색 fast-path를 사용합니다.",
                "tool_calls": [ToolCall("search_regulations", {"query": query})],
                "confidence": 0.95,
                "intent": "regulation_lookup",
                "planner_mode": "fast_path",
                "search_keywords": [],
                "error": None,
            }

        award_type = self._resolve_award_query_type(query, entity_filter)
        if award_type:
            award_parameters: Dict[str, Any] = {"year": year}
            if award_type != "any":
                award_parameters["award_type"] = award_type
            return {
                "analysis": "수상 질문으로 판단되어 awards fast-path를 사용합니다.",
                "tool_calls": [ToolCall("get_award_winners", award_parameters)],
                "confidence": 0.95,
                "intent": "award_lookup",
                "planner_mode": "fast_path",
                "search_keywords": [],
                "error": None,
            }

        champion_keywords = [
            "우승팀",
            "챔피언",
            "한국시리즈 우승",
            "코시 우승",
            "우승한 팀",
        ]
        champion_future_keywords = [
            "가능성",
            "할까",
            "할 수",
            "예상",
            "예측",
            "후보",
        ]
        if any(keyword in query_lower for keyword in champion_keywords) and not any(
            keyword in query_lower for keyword in champion_future_keywords
        ):
            return {
                "analysis": "우승팀 질문으로 판단되어 한국시리즈 우승팀 fast-path를 사용합니다.",
                "tool_calls": [ToolCall("get_korean_series_winner", {"year": year})],
                "confidence": 0.95,
                "intent": "champion_lookup",
                "planner_mode": "fast_path",
                "search_keywords": [],
                "error": None,
            }

        import re

        extracted_date = None
        if "오늘" in query_lower and any(
            keyword in query_lower
            for keyword in ["경기", "일정", "중계", "몇 시", "몇시"]
        ):
            extracted_date = datetime.now().strftime("%Y-%m-%d")
        else:
            date_patterns = [
                r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일",
                r"(\d{4})-(\d{1,2})-(\d{1,2})",
            ]
            for pattern in date_patterns:
                match = re.search(pattern, query)
                if match:
                    year_str, month_str, day_str = match.groups()
                    extracted_date = (
                        f"{year_str}-{month_str.zfill(2)}-{day_str.zfill(2)}"
                    )
                    break

        game_id_match = re.search(r"\b\d{8}[A-Z]{4}\d\b", query)
        game_flow_narrative_keywords = [
            "경기 흐름",
            "흐름 요약",
            "승부처",
            "언제 갈렸어",
            "언제 갈렸",
            "역전",
            "동점 흐름",
            "초중후반 득점",
            "득점 양상",
        ]
        is_game_flow_narrative = any(
            keyword in query_lower for keyword in game_flow_narrative_keywords
        )
        box_score_keywords = [
            "박스스코어",
            "box score",
            "이닝별",
            "이닝별 득점",
            "몇 점",
            "7회",
            "8회",
            "9회",
            "연장",
        ]
        if any(keyword in query_lower for keyword in box_score_keywords):
            box_score_parameters: Dict[str, Any] = {}
            if game_id_match:
                box_score_parameters["game_id"] = game_id_match.group(0)
            elif extracted_date:
                box_score_parameters["date"] = extracted_date

            if box_score_parameters:
                return {
                    "analysis": "박스스코어/이닝 질문으로 판단되어 SQL 경기 조회 fast-path를 사용합니다.",
                    "tool_calls": [
                        ToolCall("get_game_box_score", box_score_parameters)
                    ],
                    "confidence": 0.94,
                    "intent": "game_lookup",
                    "planner_mode": "fast_path",
                    "search_keywords": [],
                    "error": None,
                }

        if extracted_date and is_game_flow_narrative:
            return None

        if extracted_date:
            game_parameters: Dict[str, Any] = {"date": extracted_date}
            if team_name:
                game_parameters["team"] = team_name
            return {
                "analysis": "날짜/일정 질문으로 판단되어 경기 일정 fast-path를 사용합니다.",
                "tool_calls": [ToolCall("get_games_by_date", game_parameters)],
                "confidence": 0.93,
                "intent": "games_by_date_lookup",
                "planner_mode": "fast_path",
                "search_keywords": [],
                "error": None,
            }

        leaderboard_trigger_keywords = [
            "리더보드",
            "랭킹",
            "순위",
            "1위",
            "top",
            "탑",
        ]
        leaderboard_stat_map = [
            ("홈런", ("home_runs", "batting")),
            ("타율", ("avg", "batting")),
            ("ops", ("ops", "batting")),
            ("타점", ("rbi", "batting")),
            ("도루", ("stolen_bases", "batting")),
            ("era", ("era", "pitching")),
            ("평균자책", ("era", "pitching")),
            ("whip", ("whip", "pitching")),
            ("세이브", ("saves", "pitching")),
            ("홀드", ("holds", "pitching")),
            ("탈삼진", ("strikeouts", "pitching")),
        ]
        if any(keyword in query_lower for keyword in leaderboard_trigger_keywords):
            for keyword, (stat_name, position) in leaderboard_stat_map:
                if keyword in query_lower:
                    return {
                        "analysis": "리더보드 질문으로 판단되어 순위 fast-path를 사용합니다.",
                        "tool_calls": [
                            ToolCall(
                                "get_leaderboard",
                                {
                                    "stat_name": stat_name,
                                    "year": year,
                                    "position": position,
                                    "limit": 10,
                                },
                            )
                        ],
                        "confidence": 0.92,
                        "intent": "leaderboard_lookup",
                        "planner_mode": "fast_path",
                        "search_keywords": [],
                        "error": None,
                    }

        if team_name:
            rank_keywords = [
                "순위",
                "몇 위",
                "몇위",
                "승률",
                "승패",
            ]
            if any(keyword in query_lower for keyword in rank_keywords):
                return {
                    "analysis": "팀 순위/성적 질문으로 판단되어 팀 순위 fast-path를 사용합니다.",
                    "tool_calls": [
                        ToolCall(
                            "get_team_rank",
                            {"team_name": team_name, "year": year},
                        )
                    ],
                    "confidence": 0.94,
                    "intent": "team_rank_lookup",
                    "planner_mode": "fast_path",
                    "search_keywords": [],
                    "error": None,
                }

            last_game_keywords = [
                "마지막 경기",
                "최근 경기 언제",
                "마지막으로 경기",
                "언제 마지막으로",
                "최종전",
            ]
            if any(keyword in query_lower for keyword in last_game_keywords):
                return {
                    "analysis": "마지막 경기 날짜 질문으로 판단되어 팀 마지막 경기 fast-path를 사용합니다.",
                    "tool_calls": [
                        ToolCall(
                            "get_team_last_game",
                            {"team_name": team_name, "year": year},
                        )
                    ],
                    "confidence": 0.93,
                    "intent": "team_last_game_lookup",
                    "planner_mode": "fast_path",
                    "search_keywords": [],
                    "error": None,
                }

        player_fast_path_block_keywords = [
            "맞대결",
            "상대전적",
            "승부",
            "예측",
            "누가 이길",
            "누가 유리",
            "수상",
            "순위",
            "랭킹",
            "리더보드",
            "규정",
            "판정",
            "일정",
            "박스스코어",
        ]
        if len(query_player_names) >= 2 and not any(
            self._is_player_fast_path_blocked_keyword(query_lower, keyword)
            for keyword in player_fast_path_block_keywords
        ):
            player_tool_calls = self._build_player_fast_path_tool_calls(
                query,
                query_player_names,
                year,
                entity_filter,
            )
            if player_tool_calls:
                return {
                    "analysis": "복수 선수 설명형 질문으로 판단되어 선수 fast-path를 사용합니다.",
                    "tool_calls": player_tool_calls,
                    "confidence": 0.94,
                    "intent": "freeform",
                    "planner_mode": "player_fast_path",
                    "grounding_mode": "structured_kbo",
                    "source_tier": "database",
                    "search_keywords": [],
                    "error": None,
                }

        if isinstance(player_name, str) and player_name.strip():
            player_stat_keywords = [
                "성적",
                "기록",
                "타율",
                "ops",
                "출루율",
                "장타율",
                "홈런",
                "타점",
                "도루",
                "war",
                "era",
                "whip",
                "세이브",
                "홀드",
            ]
            if any(keyword in query_lower for keyword in player_stat_keywords):
                return {
                    "analysis": "선수 기록 질문으로 판단되어 선수 성적 fast-path를 사용합니다.",
                    "tool_calls": [
                        ToolCall(
                            "get_player_stats",
                            {"player_name": player_name, "year": year},
                        )
                    ],
                    "confidence": 0.94,
                    "intent": "player_stats_lookup",
                    "planner_mode": "fast_path",
                    "search_keywords": [],
                    "error": None,
                }

        return None

    def _build_fast_path_plan(
        self, query: str, entity_filter: Any, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        decision = self._resolve_chat_intent(query, entity_filter)
        if decision.intent != ChatIntent.TEAM_ANALYSIS:
            return self._build_reference_fast_path_plan(query, entity_filter)

        messages = context.get("messages") if isinstance(context, dict) else None
        history = context.get("history") if isinstance(context, dict) else None
        message_count = 0
        if isinstance(messages, list):
            message_count = len(messages)
        elif isinstance(history, list):
            message_count = len(history)
        if message_count < self.fast_path_min_messages:
            explicit_team = self._detect_team_alias_from_query(query)
            if explicit_team:
                logger.info(
                    "[PlannerDecision] fast_path_gate_override explicit_team=%s message_count=%d",
                    explicit_team,
                    message_count,
                )
            else:
                logger.info(
                    "[PlannerDecision] mode=llm reason=insufficient_messages_for_fast_path count=%s min_required=%d",
                    message_count,
                    self.fast_path_min_messages,
                )
                return None

        planned_team_path = self._intent_decision_to_plan(decision)
        if planned_team_path:
            return planned_team_path

        team_name = (
            getattr(entity_filter, "team_id", None)
            or extract_team(query)
            or self._detect_team_alias_from_query(query)
        )
        if not team_name:
            return self._build_reference_fast_path_plan(query, entity_filter)
        year = self._resolve_reference_year(query, entity_filter)
        tool_calls = self._build_team_fast_path_tool_calls(query, team_name, year)
        if not tool_calls:
            return self._build_reference_fast_path_plan(query, entity_filter)

        return {
            "analysis": f"{team_name} 팀 분석 질문으로 판단되어 fast-path를 사용합니다.",
            "tool_calls": tool_calls,
            "confidence": 0.96,
            "intent": "team_analysis",
            "planner_mode": "fast_path",
            "search_keywords": [],
            "error": None,
        }

    def _soft_filter_llm_tool_calls(
        self,
        tool_calls: List[Any],
        *,
        query: str,
        entity_filter: Any,
        planner_mode: str = "default_llm_planner",
    ) -> List[ToolCall]:
        if not tool_calls:
            return []

        filtered: List[ToolCall] = []
        seen = set()
        team_metric_query = self._is_team_metric_query_text(query)
        team_query = (
            self._is_team_analysis_query(query, entity_filter) or team_metric_query
        )
        low_value_for_team = {"get_current_datetime", "get_baseball_season_info"}
        player_focused_tools = {
            "get_career_stats",
            "get_player_stats",
            "get_defensive_stats",
            "get_velocity_data",
            "get_advanced_stats",
            "validate_player",
        }
        team_core_tools = {
            "get_team_summary",
            "get_team_advanced_metrics",
            "get_team_rank",
            "get_team_last_game",
        }
        query_player_names = self._extract_llm_planner_player_names(
            query, entity_filter
        )
        player_name = (
            query_player_names[0]
            if query_player_names
            else self._extract_llm_planner_player_name(entity_filter)
        )
        has_valid_player = bool(query_player_names)
        multi_player_batch_size = 0
        for raw_call in tool_calls:
            if isinstance(raw_call, ToolCall):
                parameters = raw_call.parameters
            elif isinstance(raw_call, dict):
                parameters = raw_call.get("parameters", {})
            else:
                parameters = {}
            if isinstance(parameters, dict):
                multi_player_batch_size = max(
                    multi_player_batch_size,
                    len(self._extract_player_name_batch(parameters)),
                )
        no_explicit_player = not has_valid_player and multi_player_batch_size == 0
        is_career_query = any(
            token in query.lower() for token in ["통산", "커리어", "총"]
        )
        allowed_tools: Optional[set[str]] = None
        preferred_order: List[str] = []
        tool_cap: Optional[int] = None

        if planner_mode == "team_llm_planner":
            allowed_tools = TEAM_LLM_ALLOWED_TOOLS
            preferred_order = [
                "get_team_summary",
                "get_team_advanced_metrics",
                "get_team_rank",
                "get_team_last_game",
            ]
            tool_cap = 2
        elif planner_mode == "player_llm_planner":
            allowed_tools = PLAYER_LLM_ALLOWED_TOOLS
            preferred_order = (
                ["get_career_stats", "get_player_stats", "validate_player"]
                if is_career_query
                else ["get_player_stats", "get_career_stats", "validate_player"]
            )
            tool_cap = max(
                2,
                min(
                    PLAYER_LLM_MULTI_NAME_CAP,
                    max(len(query_player_names), multi_player_batch_size, 1),
                ),
            )

        for raw_call in tool_calls:
            coerced_calls = self._coerce_tool_calls(raw_call)
            if not coerced_calls:
                logger.warning(
                    "[Planner] skip invalid llm tool_call format: %s", raw_call
                )
                continue
            for call in coerced_calls:
                call = self._apply_reference_year_default(
                    call, query=query, entity_filter=entity_filter
                )
                if allowed_tools is not None and call.tool_name not in allowed_tools:
                    logger.info(
                        "[Planner] drop tool_call outside planner scope mode=%s tool=%s",
                        planner_mode,
                        call.tool_name,
                    )
                    continue
                if team_query and call.tool_name in low_value_for_team:
                    logger.info(
                        "[Planner] drop low-value team tool_call: %s", call.tool_name
                    )
                    continue
                if team_metric_query and call.tool_name == "get_leaderboard":
                    logger.info(
                        "[Planner] drop leaderboard tool_call for team metric query: %s",
                        call.tool_name,
                    )
                    continue
                if (
                    team_query
                    and no_explicit_player
                    and call.tool_name in player_focused_tools
                ):
                    logger.info(
                        "[Planner] drop player-focused tool_call for team query: %s",
                        call.tool_name,
                    )
                    continue
                if (
                    planner_mode == "player_llm_planner"
                    and no_explicit_player
                    and call.tool_name != "validate_player"
                ):
                    logger.info(
                        "[Planner] drop player tool_call without validated player: %s",
                        call.tool_name,
                    )
                    continue

                key = (
                    call.tool_name,
                    json.dumps(call.parameters, ensure_ascii=False, sort_keys=True),
                )
                if key in seen:
                    continue
                seen.add(key)
                filtered.append(call)

        if team_query:
            has_team_core = any(call.tool_name in team_core_tools for call in filtered)
            if not has_team_core:
                team_name = (
                    getattr(entity_filter, "team_id", None)
                    or extract_team(query)
                    or self._detect_team_alias_from_query(query)
                )
                if team_name:
                    year = self._resolve_reference_year(query, entity_filter)
                    fallback_team_calls = self._build_team_fast_path_tool_calls(
                        query, team_name, year
                    )
                    logger.info(
                        "[Planner] replace llm tool_calls with team fallback tools count=%d",
                        len(fallback_team_calls),
                    )
                    return fallback_team_calls

        if planner_mode == "player_llm_planner" and filtered:
            filtered = self._supplement_missing_player_tool_calls(
                filtered,
                query=query,
                entity_filter=entity_filter,
                query_player_names=query_player_names,
                is_career_query=is_career_query,
                tool_cap=tool_cap or 2,
            )

        if planner_mode == "player_llm_planner" and not filtered and query_player_names:
            fallback_tool_name = (
                "get_career_stats" if is_career_query else "get_player_stats"
            )
            fallback_names = query_player_names[: max(1, tool_cap or 2)]
            filtered = []
            for fallback_name in fallback_names:
                fallback_parameters: Dict[str, Any] = {
                    "player_name": fallback_name,
                    "position": getattr(entity_filter, "position_type", None) or "both",
                }
                if fallback_tool_name == "get_player_stats":
                    fallback_parameters["year"] = self._resolve_reference_year(
                        query, entity_filter
                    )
                filtered.append(
                    ToolCall(
                        tool_name=fallback_tool_name,
                        parameters=fallback_parameters,
                    )
                )

        if preferred_order and filtered:
            preferred_rank = {
                tool_name: idx for idx, tool_name in enumerate(preferred_order)
            }
            filtered.sort(
                key=lambda call: (
                    preferred_rank.get(call.tool_name, len(preferred_order)),
                    call.tool_name,
                )
            )
        if tool_cap is not None:
            filtered = filtered[:tool_cap]

        return filtered

    def _is_meaningful_tool_data(self, data: Any) -> bool:
        if data is None:
            return False
        if isinstance(data, list):
            return len(data) > 0
        if isinstance(data, dict):
            if "exists" in data and "found_players" in data:
                if data.get("exists") is False:
                    found_players = data.get("found_players") or []
                    return len(found_players) > 0
                return True
            found = data.get("found")
            if found is not False:
                return True
            core_list_keys = [
                "top_batters",
                "top_pitchers",
                "leaderboard",
                "games",
                "results",
                "items",
            ]
            return any(
                isinstance(data.get(key), list) and len(data.get(key, [])) > 0
                for key in core_list_keys
            )
        return bool(data)

    def _has_meaningful_tool_results(self, tool_results: List[ToolResult]) -> bool:
        for result in tool_results:
            if not result.success:
                continue
            if self._is_meaningful_tool_data(result.data):
                return True
        return False

    def _build_stats_lookup_followup_tool_call(
        self,
        query: str,
        analysis_result: Dict[str, Any],
        tool_results: List[ToolResult],
    ) -> Optional[ToolCall]:
        query_lower = query.lower()
        stat_tokens = [
            "평균자책",
            "평균 자책",
            "era",
            "whip",
            "ops",
            "타율",
            "홈런",
            "타점",
            "세이브",
            "save",
            "홀드",
            "hold",
            "탈삼진",
            "삼진",
            "안타",
            "출루율",
            "장타율",
            "승리",
            "다승",
            "wins",
            "war",
            "기록",
            "성적",
        ]
        if not any(token in query_lower for token in stat_tokens):
            return None

        original_tool_calls = analysis_result.get("tool_calls") or []
        if any(
            getattr(tool_call, "tool_name", "") == "get_player_stats"
            for tool_call in original_tool_calls
        ):
            return None

        if any(
            isinstance(result.data, dict)
            and (
                result.data.get("batting_stats") is not None
                or result.data.get("pitching_stats") is not None
            )
            for result in tool_results
            if result.success
        ):
            return None

        validate_result: Optional[ToolResult] = None
        for tool_call, result in zip(original_tool_calls, tool_results):
            if getattr(tool_call, "tool_name", "") == "validate_player":
                validate_result = result
                break

        if (
            validate_result is None
            or not validate_result.success
            or not isinstance(validate_result.data, dict)
        ):
            return None

        validate_data = validate_result.data
        if not validate_data.get("exists"):
            return None

        found_players = validate_data.get("found_players") or []
        if len(found_players) != 1 or not isinstance(found_players[0], dict):
            return None

        player_info = found_players[0]
        player_name = str(player_info.get("player_name") or "").strip()
        if not player_name:
            return None

        year = validate_data.get("year")
        if not isinstance(year, int):
            return None

        raw_position = str(player_info.get("position_type") or "").strip().lower()
        if raw_position == "pitching":
            position = "pitching"
        elif raw_position == "batting":
            position = "batting"
        else:
            position = "both"

        return ToolCall(
            "get_player_stats",
            {
                "player_name": player_name,
                "year": year,
                "position": position,
            },
        )

    def _tool_results_have_source_tier(
        self, tool_results: List[ToolResult], source_tier: str
    ) -> bool:
        target_tier = self._normalize_source_tier(source_tier)
        for result in tool_results:
            if not isinstance(result.data, dict):
                continue
            result_tier = self._normalize_source_tier(
                str(result.data.get("source") or result.data.get("source_tier") or "")
            )
            if result_tier == target_tier:
                return True
        return False

    @staticmethod
    def _analysis_tool_names(analysis_result: Dict[str, Any]) -> List[str]:
        tool_names: List[str] = []
        for tool_call in analysis_result.get("tool_calls") or []:
            tool_name = getattr(tool_call, "tool_name", "")
            if not tool_name and isinstance(tool_call, dict):
                tool_name = str(
                    tool_call.get("tool_name") or tool_call.get("name") or ""
                )
            tool_name = str(tool_name or "").strip()
            if tool_name:
                tool_names.append(tool_name)
        return tool_names

    def _should_use_document_fallback(
        self,
        query: str,
        analysis_result: Dict[str, Any],
        tool_results: List[ToolResult],
    ) -> bool:
        if self._has_meaningful_tool_results(tool_results):
            return False
        if self._tool_results_have_source_tier(tool_results, "docs"):
            return False

        intent = str(analysis_result.get("intent") or "").lower()
        grounding_mode = str(analysis_result.get("grounding_mode") or "").lower()
        if intent == "latest_info" or grounding_mode == "latest_info":
            return False
        if intent in {
            "player_lookup",
            "leaderboard_lookup",
            "team_analysis",
            "schedule_lookup",
            "season_standing_lookup_by_rank",
        }:
            return False

        query_lower = query.lower()
        explicit_document_tokens = [
            "뜻",
            "의미",
            "원리",
            "규정",
            "룰",
            "규칙",
            "abs",
            "whip",
            "wrc+",
            "war",
            "ops",
            "fip",
            "babip",
            "qs",
            "피치클락",
            "전술",
            "전략",
            "플래툰",
            "번트",
            "히트앤런",
            "마스코트",
            "응원",
            "팬 문화",
            "구장",
            "홈구장",
            "역사",
            "전통",
        ]
        structured_lookup_tools = {
            "validate_player",
            "get_player_stats",
            "get_career_stats",
            "get_leaderboard",
            "get_team_summary",
            "get_team_advanced_metrics",
            "get_team_rank",
            "get_team_last_game",
            "get_team_by_rank",
            "get_games_by_date",
            "get_award_winners",
            "get_korean_series_winner",
        }
        analysis_tool_names = self._analysis_tool_names(analysis_result)
        if any(
            tool_name in structured_lookup_tools for tool_name in analysis_tool_names
        ) and not any(token in query_lower for token in explicit_document_tokens):
            return False

        document_fallback_tokens = [
            "뜻",
            "의미",
            "설명",
            "해설",
            "원리",
            "규정",
            "룰",
            "규칙",
            "abs",
            "whip",
            "wrc+",
            "war",
            "ops",
            "fip",
            "babip",
            "qs",
            "피치클락",
            "전술",
            "전략",
            "플래툰",
            "번트",
            "히트앤런",
            "마스코트",
            "응원",
            "팬 문화",
            "구장",
            "홈구장",
            "역사",
            "전통",
            "우승",
            "우승팀",
            "한국시리즈",
            "플레이오프",
            "포스트시즌",
            "최종전",
            "마지막 경기",
            "몇 위",
            "몇등",
            "순위",
            "리더보드",
            "랭킹",
            "다승",
            "홈런",
            "타율",
            "ops",
            "era",
            "세이브",
            "홀드",
            "탈삼진",
            "mvp",
            "신인왕",
            "골든글러브",
            "수상",
        ]
        if any(token in query_lower for token in document_fallback_tokens):
            return True

        return grounding_mode in {
            "baseball_explainer",
            "long_tail_entity",
        } or intent in {
            "baseball_explainer",
            "long_tail_entity",
            "season_result_lookup",
        }

    def _build_low_grounding_fallback_plan(
        self,
        query: str,
        analysis_result: Dict[str, Any],
        tool_results: List[ToolResult],
    ) -> List[Dict[str, Any]]:
        fallback_plan: List[Dict[str, Any]] = []

        if self._should_use_document_fallback(query, analysis_result, tool_results):
            fallback_plan.append(
                {
                    "tool_call": ToolCall(
                        "search_documents",
                        {"query": query, "limit": 5},
                    ),
                    "grounding_mode": "baseball_explainer",
                    "source_tier": "docs",
                    "fallback_reason": "internal_lookup_returned_no_results",
                }
            )

        return fallback_plan

    def _normalize_source_tier(self, raw_source: str) -> str:
        normalized = (raw_source or "").strip().lower()
        if normalized in {"web_search", "web", "news", "official_api"}:
            return "web"
        if normalized in {
            "verified_docs",
            "markdown_docs",
            "kbo_definitions",
            "kbo_regulations",
            "document",
            "docs",
        }:
            return "docs"
        if normalized in {"predefined", "none", "cache", "mixed"}:
            return normalized
        return "db"

    def _source_tier_from_tool_results(
        self,
        tool_results: List[ToolResult],
        explicit_source_tier: Optional[str] = None,
    ) -> str:
        tiers: List[str] = []
        if explicit_source_tier:
            tiers.append(self._normalize_source_tier(explicit_source_tier))
        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue
            tiers.append(
                self._normalize_source_tier(
                    str(
                        result.data.get("source")
                        or result.data.get("source_tier")
                        or "database"
                    )
                )
            )
        unique_tiers = [tier for tier in dict.fromkeys(tiers) if tier]
        if not unique_tiers:
            return "db"
        if len(unique_tiers) == 1:
            return unique_tiers[0]
        if "web" in unique_tiers or "docs" in unique_tiers:
            return "mixed"
        return unique_tiers[0]

    def _resolve_grounding_mode(
        self,
        predicted_intent: str,
        analysis_result: Dict[str, Any],
        tool_results: List[ToolResult],
    ) -> str:
        explicit_mode = analysis_result.get("grounding_mode")
        if explicit_mode:
            return str(explicit_mode)
        if predicted_intent in {
            "baseball_explainer",
            "latest_info",
            "long_tail_entity",
            "unsupported",
        }:
            return predicted_intent
        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue
            source_tier = self._normalize_source_tier(
                str(result.data.get("source") or result.data.get("source_tier") or "")
            )
            if source_tier == "web":
                return "latest_info"
            if source_tier == "docs":
                return "baseball_explainer"
        return "structured_kbo"

    def _build_answer_sources(
        self, tool_results: List[ToolResult]
    ) -> List[Dict[str, Any]]:
        answer_sources: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue
            source_tier = self._normalize_source_tier(
                str(result.data.get("source") or result.data.get("source_tier") or "")
            )
            raw_items = result.data.get("results")
            items: List[Dict[str, Any]] = []
            if isinstance(raw_items, list):
                items = [item for item in raw_items if isinstance(item, dict)]
            elif isinstance(result.data.get("documents"), list):
                items = [
                    item
                    for item in result.data.get("documents", [])
                    if isinstance(item, dict)
                ]

            for item in items:
                meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
                source_name = (
                    item.get("source_name")
                    or meta.get("source_file")
                    or item.get("source_table")
                    or result.data.get("source")
                )
                ref = str(
                    item.get("url")
                    or item.get("source_row_id")
                    or item.get("title")
                    or source_name
                )
                dedupe_key = (source_tier, ref)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                answer_sources.append(
                    {
                        "source_tier": source_tier,
                        "title": item.get("title"),
                        "source_name": source_name,
                        "url": item.get("url"),
                        "published_at": item.get("published_at"),
                    }
                )
                if len(answer_sources) >= 5:
                    return answer_sources
        return answer_sources

    def _resolve_as_of_date(self, tool_results: List[ToolResult]) -> str:
        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue
            if result.data.get("as_of_date"):
                return str(result.data.get("as_of_date"))
        return datetime.now().date().isoformat()

    async def process_query_stream(
        self, query: str, context: Dict[str, Any] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        통계 질문을 처리하고 진행 상황(이벤트)을 스트리밍합니다.
        """
        logger.info(f"[BaseballAgent] Processing query stream: {query}")
        if context is None:
            context = {}
        elif not isinstance(context, dict):
            context = {}
        request_mode = str(context.get("request_mode", "stream"))
        process_started_at = time.perf_counter()
        analysis_started_at = process_started_at

        # --- 신규 추가: 일상 대화 처리기 ---
        if self._is_chitchat(query):
            response = self._get_chitchat_response(query)
            if response:
                yield {"type": "answer_chunk", "content": response}
                yield {
                    "type": "metadata",
                    "data": {
                        "tool_calls": [],
                        "tool_results": [],
                        "verified": True,
                        "data_sources": ["predefined"],
                        "intent": "general_conversation",
                        "planner_mode": "predefined",
                        "grounding_mode": "predefined",
                        "source_tier": "predefined",
                        "answer_sources": [],
                        "as_of_date": datetime.now().date().isoformat(),
                        "fallback_reason": None,
                    },
                }
                return

        # 1. 의도 파악
        from ..ml.intent_router import predict_intent

        intent = predict_intent(query)

        if intent == "match_prediction":
            disclaimer = "\n\n[주의] 이 예측은 과거 데이터를 기반으로 한 확률적 추정일 뿐이며, 실제 경기 결과와 다를 수 있습니다. 도박이나 금전적 베팅의 근거로 사용할 수 없습니다."
            context["prompt_override"] = (
                "당신은 KBO 승부 예측 AI입니다. 다음 도구 실행 결과(데이터)를 바탕으로 승부를 예측하세요.\n"
                "분석 결과와 승리 확률을 명확히 제시하고, 그 근거(최근 성적, 상대 전적 등)를 설명하세요.\n"
                f"답변의 마지막에는 반드시 다음 면책 조항을 포함하세요: {disclaimer}\n\n"
                "질문: {question}\n\n데이터:\n{context}"
            )
        elif intent == "game_analysis":
            context["prompt_override"] = (
                "당신은 KBO 전문 데이터 분석가입니다. 제공된 WPA(승리 확률 기여도)나 클러치 데이터를 바탕으로 심층적인 분석을 제공하세요.\n"
                "단순한 수치 나열보다는, 그 수치가 의미하는 경기 상황의 중요도나 선수의 해결사 능력을 설명하는 데 집중하세요.\n\n"
                "질문: {question}\n\n데이터:\n{context}"
            )

        # 1단계: 도구 계획
        yield {"type": "status", "message": "질문 의도를 분석하고 있습니다..."}
        analysis_result = await self._analyze_query_and_plan_tools(query, context)
        analysis_ms = round((time.perf_counter() - analysis_started_at) * 1000, 2)
        planner_mode = analysis_result.get("planner_mode", "default_llm_planner")
        fallback_triggered = False
        intent = analysis_result.get("intent") or intent

        if analysis_result.get("direct_answer"):
            direct_answer = str(analysis_result.get("direct_answer", "")).strip()
            planner_cache_hit = bool(analysis_result.get("planner_cache_hit", False))
            yield {"type": "answer_chunk", "content": direct_answer}
            yield {
                "type": "metadata",
                "data": {
                    "tool_calls": [],
                    "tool_results": [],
                    "visualizations": [],
                    "verified": True,
                    "data_sources": [],
                    "intent": intent,
                    "error": None,
                    "planner_mode": planner_mode,
                    "planner_cache_hit": planner_cache_hit,
                    "tool_execution_mode": "none",
                    "grounding_mode": analysis_result.get(
                        "grounding_mode", "unsupported"
                    ),
                    "source_tier": analysis_result.get("source_tier", "none"),
                    "answer_sources": [],
                    "as_of_date": datetime.now().date().isoformat(),
                    "fallback_reason": analysis_result.get("fallback_reason"),
                },
            }
            return

        if analysis_result["error"]:
            internal_error = str(analysis_result["error"])
            logger.warning(
                "[BaseballAgent] Analysis failed query=%s error=%s",
                query[:120],
                internal_error,
            )
            heuristic_answer = self._build_known_explainer_answer(
                query,
                grounding_mode="unsupported",
                source_tier="none",
            )
            heuristic_verified = False
            heuristic_tool_calls: List[ToolCall] = []
            heuristic_tool_results: List[ToolResult] = []
            heuristic_data_sources = [
                {
                    "tool": "analysis",
                    "verified": False,
                    "error": "analysis_temporarily_unavailable",
                }
            ]

            if heuristic_answer:
                heuristic_verified = True
                heuristic_data_sources = [
                    {"tool": "builtin_knowledge", "verified": True}
                ]

            if not heuristic_answer:
                heuristic_answer = self._build_known_latest_answer(
                    query,
                    grounding_mode="latest_info",
                    source_tier="web",
                    as_of_date=datetime.now().date().isoformat(),
                )
                if heuristic_answer:
                    heuristic_verified = True
                    heuristic_data_sources = [
                        {"tool": "builtin_latest", "verified": True}
                    ]

            if not heuristic_answer:
                query_lower = query.lower()
                if any(token in query_lower for token in ["팀", "구단"]) and any(
                    token in query_lower
                    for token in [
                        "타율",
                        "ops",
                        "평균자책",
                        "평균 자책",
                        "평균자책점",
                        "방어율",
                        "era",
                        "홈런",
                        "타점",
                    ]
                ):
                    team_name = self._detect_team_alias_from_query(query)
                    year_match = re.search(r"(20\d{2})년", query)
                    year = (
                        int(year_match.group(1)) if year_match else datetime.now().year
                    )
                    if team_name:
                        heuristic_tool_calls = self._build_team_fast_path_tool_calls(
                            query, team_name, year
                        )
                        heuristic_tool_results = await self._execute_tool_batch_async(
                            heuristic_tool_calls
                        )
                        heuristic_answer = self._build_fast_path_answer(
                            query, heuristic_tool_results, chat_mode=True
                        )
                        if heuristic_answer:
                            heuristic_verified = True
                            heuristic_data_sources = [
                                {
                                    "tool": (
                                        result.data.get("source", "database")
                                        if isinstance(result.data, dict)
                                        else "database"
                                    ),
                                    "verified": bool(result.success),
                                }
                                for result in heuristic_tool_results
                            ]

            structured_error_answer = heuristic_answer or (
                "지금은 질문 분석 단계가 잠시 불안정해서 정확한 답변을 만들지 못했습니다.\n\n"
                "같은 질문을 한 번 더 보내주시면 대부분 바로 복구됩니다."
            )
            heuristic_tool_execution_mode = self._resolve_tool_execution_mode(
                heuristic_tool_calls
            )
            planner_status = (
                "analysis_error_fallback" if heuristic_verified else "analysis_error"
            )
            yield {
                "type": "answer_chunk",
                "content": structured_error_answer,
            }
            yield {
                "type": "metadata",
                "data": {
                    "tool_calls": [
                        {
                            "tool_name": tool_call.tool_name,
                            "parameters": tool_call.parameters,
                        }
                        for tool_call in heuristic_tool_calls
                    ],
                    "tool_results": [
                        {
                            "success": result.success,
                            "data": result.data,
                            "message": result.message,
                        }
                        for result in heuristic_tool_results
                    ],
                    "visualizations": [],
                    "verified": heuristic_verified,
                    "data_sources": heuristic_data_sources,
                    "intent": intent,
                    "error": (
                        None
                        if heuristic_verified
                        else "analysis_temporarily_unavailable"
                    ),
                    "planner_mode": planner_mode,
                    "planner_status": planner_status,
                    "planner_cache_hit": False,
                    "tool_execution_mode": heuristic_tool_execution_mode,
                    "grounding_mode": (
                        (
                            "structured_kbo"
                            if heuristic_tool_results
                            else "baseball_explainer"
                        )
                        if heuristic_verified
                        else "unsupported"
                    ),
                    "source_tier": (
                        ("database" if heuristic_tool_results else "builtin")
                        if heuristic_verified
                        else "none"
                    ),
                    "answer_sources": [],
                    "as_of_date": datetime.now().date().isoformat(),
                    "fallback_reason": (
                        (
                            "analysis_error_fast_path_recovery"
                            if heuristic_tool_results
                            else "analysis_error_builtin_recovery"
                        )
                        if heuristic_verified
                        else "analysis_temporarily_unavailable"
                    ),
                },
            }
            return

        logger.info(
            "[Planner] mode=%s fallback_triggered=%s tool_count=%d analysis_ms=%s",
            planner_mode,
            fallback_triggered,
            len(analysis_result.get("tool_calls", [])),
            analysis_result.get("analysis_ms"),
        )

        # 2단계: 도구 실행
        tool_results = []
        tool_started_at = time.perf_counter()
        for tool_call in analysis_result["tool_calls"]:
            yield {
                "type": "tool_start",
                "tool": tool_call.tool_name,
                "params": tool_call.parameters,
            }

        tool_results = await self._execute_tool_batch_async(
            analysis_result.get("tool_calls", [])
        )
        tool_execution_mode = self._resolve_tool_execution_mode(
            analysis_result.get("tool_calls", [])
        )
        for tool_call, result in zip(analysis_result["tool_calls"], tool_results):
            yield {
                "type": "tool_result",
                "tool": tool_call.tool_name,
                "success": result.success,
                "message": result.message,
            }

        stats_followup_tool_call = self._build_stats_lookup_followup_tool_call(
            query, analysis_result, tool_results
        )
        if stats_followup_tool_call is not None:
            yield {
                "type": "tool_start",
                "tool": stats_followup_tool_call.tool_name,
                "params": stats_followup_tool_call.parameters,
            }
            followup_result = await self._execute_tool_call_async(
                stats_followup_tool_call
            )
            tool_results.append(followup_result)
            yield {
                "type": "tool_result",
                "tool": stats_followup_tool_call.tool_name,
                "success": followup_result.success,
                "message": followup_result.message,
            }

        # Fast-Path 품질 방어: 데이터가 빈약하면 LLM 분석 경로로 1회 폴백
        if (
            planner_mode == "fast_path"
            and analysis_result.get("intent") == "team_analysis"
            and self.fast_path_fallback_on_empty
            and request_mode != "completion"
            and not self._has_meaningful_tool_results(tool_results)
        ):
            fallback_triggered = True
            logger.info(
                "[Planner] mode=%s fallback_triggered=%s reason=insufficient_tool_data request_mode=%s",
                planner_mode,
                fallback_triggered,
                request_mode,
            )
            llm_fallback_plan = await self._analyze_query_with_llm(query, context)
            if not llm_fallback_plan.get("error"):
                fallback_tool_calls = self._prioritize_and_cap_tool_calls(
                    llm_fallback_plan.get("tool_calls", [])
                )

                fallback_results: List[ToolResult] = []
                for tool_call in fallback_tool_calls:
                    yield {
                        "type": "tool_start",
                        "tool": tool_call.tool_name,
                        "params": tool_call.parameters,
                    }
                fallback_results = await self._execute_tool_batch_async(
                    fallback_tool_calls
                )
                for tool_call, fallback_result in zip(
                    fallback_tool_calls, fallback_results
                ):
                    yield {
                        "type": "tool_result",
                        "tool": tool_call.tool_name,
                        "success": fallback_result.success,
                        "message": fallback_result.message,
                    }

                if self._has_meaningful_tool_results(fallback_results):
                    analysis_result = llm_fallback_plan
                    analysis_result["planner_mode"] = llm_fallback_plan.get(
                        "planner_mode", "default_llm_planner"
                    )
                    tool_results = fallback_results
                    planner_mode = analysis_result["planner_mode"]

        fallback_plan = self._build_low_grounding_fallback_plan(
            query, analysis_result, tool_results
        )
        for fallback_step in fallback_plan:
            fallback_tool_call = fallback_step["tool_call"]
            yield {
                "type": "tool_start",
                "tool": fallback_tool_call.tool_name,
                "params": fallback_tool_call.parameters,
            }
            fallback_result = await self._execute_tool_call_async(fallback_tool_call)
            tool_results.append(fallback_result)
            yield {
                "type": "tool_result",
                "tool": fallback_tool_call.tool_name,
                "success": fallback_result.success,
                "message": fallback_result.message,
            }
            if self._has_meaningful_tool_results([fallback_result]):
                fallback_triggered = True
                analysis_result["grounding_mode"] = fallback_step["grounding_mode"]
                analysis_result["source_tier"] = fallback_step["source_tier"]
                analysis_result["fallback_reason"] = fallback_step["fallback_reason"]
                break
        tool_ms = round((time.perf_counter() - tool_started_at) * 1000, 2)

        # 3단계: 답변 생성
        yield {
            "type": "status",
            "message": "분석된 데이터를 바탕으로 답변을 생성하고 있습니다...",
        }
        context["planner_mode"] = planner_mode
        grounding_mode = self._resolve_grounding_mode(
            intent, analysis_result, tool_results
        )
        source_tier = self._source_tier_from_tool_results(
            tool_results, analysis_result.get("source_tier")
        )
        answer_sources = self._build_answer_sources(tool_results)
        as_of_date = self._resolve_as_of_date(tool_results)
        fallback_reason = analysis_result.get("fallback_reason")
        context["grounding_mode"] = grounding_mode
        context["source_tier"] = source_tier
        context["as_of_date"] = as_of_date
        context["fallback_reason"] = fallback_reason
        answer_started_at = time.perf_counter()

        answer_error = None
        answer_verified = False
        answer_data_sources = []
        first_token_timeout_reason: Optional[str] = None
        fallback_answer_used = False

        def _build_safe_stream_error_answer(reason: str) -> str:
            del reason
            return (
                "답변 생성 중 연결이 잠시 불안정해 응답이 중단되었습니다.\n\n"
                "같은 질문을 다시 보내주시면 이어서 확인하겠습니다."
            )

        answer_attempt = 0
        prefetched_chunks: List[str] = []
        answer_iterator: Optional[AsyncGenerator[str, None]] = None
        if request_mode == "completion":
            effective_watchdog_seconds = self.chat_first_token_watchdog_seconds
            if planner_mode == "fast_path":
                effective_watchdog_seconds = max(
                    effective_watchdog_seconds,
                    30.0,
                )
            effective_retry_max_attempts = self.chat_first_token_retry_max_attempts
        elif request_mode == "stream":
            effective_watchdog_seconds = self.chat_stream_first_token_watchdog_seconds
            effective_retry_max_attempts = (
                self.chat_stream_first_token_retry_max_attempts
            )
        else:
            effective_watchdog_seconds = self.chat_first_token_watchdog_seconds
            effective_retry_max_attempts = self.chat_first_token_retry_max_attempts

        answer_context = dict(context or {})
        answer_context["planner_mode"] = planner_mode
        answer_context["intent"] = intent
        answer_context["grounding_mode"] = grounding_mode
        answer_context["source_tier"] = source_tier
        answer_context["as_of_date"] = as_of_date

        while True:
            answer_result = await self._generate_verified_answer(
                query, tool_results, answer_context
            )
            answer_error = answer_result.get("error")
            answer_verified = answer_result["verified"]
            answer_data_sources = answer_result.get("data_sources", [])
            prefetched_chunks = []
            answer_iterator = None

            retry_triggered = False
            answer_content = answer_result["answer"]
            if hasattr(answer_content, "__aiter__"):
                answer_iterator = answer_content.__aiter__()
                try:
                    async with asyncio.timeout(effective_watchdog_seconds):
                        first_chunk = await answer_iterator.__anext__()
                    if first_chunk:
                        prefetched_chunks.append(first_chunk)
                except TimeoutError:
                    if answer_attempt < effective_retry_max_attempts:
                        answer_attempt += 1
                        retry_triggered = True
                        logger.warning(
                            "[AnswerWatchdog] first_token_timeout retry=%d/%d timeout=%.1fs mode=%s",
                            answer_attempt,
                            effective_retry_max_attempts,
                            effective_watchdog_seconds,
                            request_mode,
                        )
                    else:
                        recovery = self._build_answer_generation_recovery_answer(
                            query,
                            tool_results,
                            answer_context,
                        )
                        if recovery:
                            logger.warning(
                                "[AnswerWatchdog] timeout_recovered query=%s",
                                query[:120],
                            )
                            first_token_timeout_reason = "first_token_timeout_recovered"
                            answer_error = None
                            answer_verified = bool(recovery.get("verified", False))
                            answer_iterator = None
                            fallback_answer_used = True
                            fallback_reason = (
                                fallback_reason or "answer_generation_recovery"
                            )
                            prefetched_chunks = [str(recovery.get("answer") or "")]
                        else:
                            timeout_message = (
                                "답변 첫 토큰 지연으로 재시도 한도를 초과했습니다."
                            )
                            logger.warning(
                                "[AnswerWatchdog] timeout_exhausted query=%s detail=%s",
                                query[:120],
                                timeout_message,
                            )
                            first_token_timeout_reason = "first_token_timeout_exhausted"
                            answer_error = "temporary_generation_issue"
                            answer_verified = False
                            answer_iterator = None
                            fallback_answer_used = True
                            prefetched_chunks = [
                                _build_safe_stream_error_answer(timeout_message)
                            ]
                except StopAsyncIteration:
                    answer_iterator = None
                except asyncio.CancelledError:
                    logger.info(
                        "[BaseballAgent] Answer stream prefetch cancelled query=%s",
                        query[:120],
                    )
                    raise
                except Exception as exc:
                    if answer_attempt < effective_retry_max_attempts:
                        answer_attempt += 1
                        retry_triggered = True
                        logger.warning(
                            "[AnswerWatchdog] first_token_error retry=%d/%d error=%s mode=%s",
                            answer_attempt,
                            effective_retry_max_attempts,
                            exc,
                            request_mode,
                        )
                    else:
                        logger.error(
                            "[BaseballAgent] Answer stream prefetch failed: %s", exc
                        )
                        recovery = self._build_answer_generation_recovery_answer(
                            query,
                            tool_results,
                            answer_context,
                        )
                        if recovery:
                            answer_error = None
                            answer_verified = bool(recovery.get("verified", False))
                            answer_iterator = None
                            fallback_answer_used = True
                            fallback_reason = (
                                fallback_reason or "answer_generation_recovery"
                            )
                            prefetched_chunks = [str(recovery.get("answer") or "")]
                        else:
                            answer_error = "temporary_generation_issue"
                            answer_verified = False
                            answer_iterator = None
                            fallback_answer_used = True
                            prefetched_chunks = [
                                _build_safe_stream_error_answer(str(exc))
                            ]
            else:
                prefetched_chunks = [str(answer_content)]

            if retry_triggered:
                continue
            break
        first_token_ms = (
            round((time.perf_counter() - answer_started_at) * 1000, 2)
            if prefetched_chunks
            else None
        )
        answer_ms = round((time.perf_counter() - answer_started_at) * 1000, 2)
        total_ms = round((time.perf_counter() - process_started_at) * 1000, 2)

        public_answer_error = None
        if answer_error:
            logger.warning(
                "[BaseballAgent] user_error_hidden query=%s error=%s",
                query[:120],
                answer_error,
            )
            public_answer_error = "temporary_generation_issue"

        def _build_metadata() -> Dict[str, Any]:
            serialized_tool_calls = []
            for call in analysis_result.get("tool_calls", []):
                if isinstance(call, ToolCall):
                    serialized_tool_calls.append(
                        {"tool_name": call.tool_name, "parameters": call.parameters}
                    )
                elif isinstance(call, dict):
                    serialized_tool_calls.append(call)

            serialized_tool_results = []
            for result in tool_results:
                if isinstance(result, ToolResult):
                    serialized_tool_results.append(
                        {
                            "success": result.success,
                            "data": result.data,
                            "message": result.message,
                        }
                    )
                elif isinstance(result, dict):
                    serialized_tool_results.append(result)

            return {
                "tool_calls": serialized_tool_calls,
                "tool_results": serialized_tool_results,
                "visualizations": self._generate_visualizations(tool_results),
                "verified": answer_verified,
                "data_sources": answer_data_sources,
                "intent": intent,
                "error": public_answer_error,
                "planner_mode": planner_mode,
                "planner_cache_hit": bool(
                    analysis_result.get("planner_cache_hit", False)
                ),
                "tool_execution_mode": tool_execution_mode,
                "fallback_triggered": fallback_triggered,
                "fallback_answer_used": fallback_answer_used,
                "grounding_mode": grounding_mode,
                "source_tier": source_tier,
                "answer_sources": answer_sources,
                "as_of_date": as_of_date,
                "fallback_reason": fallback_reason,
                "perf": {
                    "total_ms": total_ms,
                    "analysis_ms": analysis_ms,
                    "tool_ms": tool_ms,
                    "answer_ms": answer_ms,
                    "first_token_ms": first_token_ms,
                    "tool_count": len(analysis_result.get("tool_calls", [])),
                    "tool_execution_mode": tool_execution_mode,
                    "planner_cache_hit": bool(
                        analysis_result.get("planner_cache_hit", False)
                    ),
                    "answer_retry_count": answer_attempt,
                    "request_mode": request_mode,
                    "first_token_watchdog_seconds": effective_watchdog_seconds,
                    "first_token_retry_max_attempts": effective_retry_max_attempts,
                    "first_token_timeout_reason": first_token_timeout_reason,
                    "planner_mode": planner_mode,
                    "model": self._resolve_perf_model_name(),
                },
            }

        emit_answer_before_metadata = (
            request_mode == "stream"
            and planner_mode in {"fast_path", "player_fast_path"}
            and any(bool(chunk) for chunk in prefetched_chunks)
        )

        if emit_answer_before_metadata:
            for chunk in prefetched_chunks:
                if chunk:
                    yield {"type": "answer_chunk", "content": chunk}
            yield {"type": "metadata", "data": _build_metadata()}
        else:
            yield {"type": "metadata", "data": _build_metadata()}
            for chunk in prefetched_chunks:
                if chunk:
                    yield {"type": "answer_chunk", "content": chunk}

        if answer_iterator is not None:
            try:
                async for chunk in answer_iterator:
                    yield {"type": "answer_chunk", "content": chunk}
            except asyncio.CancelledError:
                logger.info(
                    "[BaseballAgent] Answer stream iteration cancelled query=%s",
                    query[:120],
                )
                raise
            except Exception as exc:
                logger.error("[BaseballAgent] Answer stream iteration failed: %s", exc)
                yield {
                    "type": "answer_chunk",
                    "content": _build_safe_stream_error_answer(str(exc)),
                }

    async def process_query(
        self, query: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Legacy wrapper for process_query_stream.
        """
        metadata = {}
        stream = self.process_query_stream(query, context)
        answer_chunks_buffer = []

        # Consume stream until metadata
        async for event in stream:
            if event["type"] == "metadata":
                metadata.update(event["data"])
                break
            elif event["type"] == "answer_chunk":
                answer_chunks_buffer.append(event["content"])

        async def combined_answer_generator():
            for chunk in answer_chunks_buffer:
                yield chunk
            async for event in stream:
                if event["type"] == "answer_chunk":
                    yield event["content"]

        return {"answer": combined_answer_generator(), **metadata}

    def _generate_visualizations(
        self, tool_results: List[ToolResult]
    ) -> List[Dict[str, Any]]:
        """도구 실행 결과를 바탕으로 프론트엔드용 시각화 데이터를 생성합니다."""
        viz_list = []

        for res in tool_results:
            if not res.success or not res.data:
                continue

            # 1. 승부 예측 (Match Prediction) -> 확률형 차트
            if "win_probability" in res.data and "predicted_winner" in res.data:
                viz_list.append(
                    {
                        "type": "match_prediction",
                        "title": "승부 예측 결과",
                        "data": {
                            "winner": res.data.get("predicted_winner"),
                            "probability": res.data.get("win_probability"),
                            "pitcher": res.data.get("pitcher"),
                            "batter": res.data.get("batter"),
                            "history": res.data.get("head_to_head_summary"),
                        },
                    }
                )

            # 2. 불펜 현황 (Bullpen Status) -> 상태 리스트/게이지
            if "bullpen_status" in res.data:
                viz_list.append(
                    {
                        "type": "bullpen_status",
                        "title": f"{res.data.get('team', '')} 불펜 가용성",
                        "data": {
                            "team": res.data.get("team"),
                            "status_list": res.data.get("bullpen_status"),
                        },
                    }
                )

            # 3. WPA (Win Probability) -> 게이지/텍스트
            if "predicted_winner" in res.data and "win_probability" in res.data:
                pass
            elif "win_probability" in res.data and isinstance(
                res.data.get("percent"), str
            ):
                viz_list.append(
                    {
                        "type": "wpa_gauge",
                        "title": "승리 확률",
                        "data": {
                            "probability": res.data.get("win_probability"),
                            "label": res.data.get("percent"),
                        },
                    }
                )

        return viz_list

    def _compact_tool_payload_for_prompt(self, data: Any, depth: int = 0) -> Any:
        """최종 답변 프롬프트에 넣을 도구 결과를 경량화합니다."""
        if isinstance(data, dict):
            if depth >= 3:
                keys = list(data.keys())
                return {
                    "_truncated_fields": len(keys),
                    "_sample_keys": keys[: min(3, len(keys))],
                }

            items = list(data.items())
            compact: Dict[str, Any] = {}
            for key, value in items[: self.chat_tool_result_max_items]:
                compact[key] = self._compact_tool_payload_for_prompt(value, depth + 1)
            if len(items) > self.chat_tool_result_max_items:
                compact["_truncated_fields"] = (
                    len(items) - self.chat_tool_result_max_items
                )
            return compact

        if isinstance(data, list):
            if depth >= 3:
                return {"_truncated_items": len(data)}

            clipped = [
                self._compact_tool_payload_for_prompt(item, depth + 1)
                for item in data[: self.chat_tool_result_max_items]
            ]
            if len(data) > self.chat_tool_result_max_items:
                clipped.append(
                    {"_truncated_items": len(data) - self.chat_tool_result_max_items}
                )
            return clipped

        if isinstance(data, str) and len(data) > 300:
            return f"{data[:300]}...(truncated)"

        return data

    def _serialize_tool_data_for_prompt(self, data: Any) -> str:
        """도구 결과를 길이 제한 내 JSON 문자열로 직렬화합니다."""
        sanitized_data = replace_team_codes(
            data,
            team_name_resolver=self._convert_team_id_to_name,
        )
        compact_data = self._compact_tool_payload_for_prompt(sanitized_data)
        data_json = json.dumps(
            compact_data,
            ensure_ascii=False,
            cls=DateTimeEncoder,
            separators=(",", ":"),
        )
        if len(data_json) > self.chat_tool_result_max_chars:
            data_json = f"{data_json[: self.chat_tool_result_max_chars]}...(truncated)"
        return data_json

    def _clean_answer_prompt_snippet(self, text: Any, max_chars: int = 260) -> str:
        if text is None:
            return ""
        snippet = str(text)
        snippet = re.sub(r"```.*?```", " ", snippet, flags=re.DOTALL)
        snippet = re.sub(r"(^|\n)\s*#+\s*", " ", snippet, flags=re.MULTILINE)
        snippet = re.sub(r"(^|\n)\s*[-*]\s*", " ", snippet, flags=re.MULTILINE)
        snippet = re.sub(r"`+", "", snippet)
        snippet = snippet.replace("**", " ")
        snippet = snippet.replace("|", " ")
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rsplit(" ", 1)[0].rstrip()
            snippet = f"{snippet}..."
        return snippet

    def _doc_contains_query_focus(self, query: str, doc: Dict[str, Any]) -> bool:
        focus_terms = self.document_query_tool._focus_terms(query.lower())
        if not focus_terms:
            return False

        searchable_text = " ".join(
            [
                str(doc.get("title", "") or "").lower(),
                str(doc.get("content", "") or "").lower(),
                str(doc.get("source_row_id", "") or "").lower(),
            ]
        )
        compact_text = re.sub(r"\s+", "", searchable_text)

        for term in focus_terms:
            compact_term = re.sub(r"\s+", "", term)
            if len(compact_term) < 2:
                continue
            if compact_term in compact_text or term in searchable_text:
                return True
        return False

    def _prepare_tool_data_for_answer_prompt(
        self,
        query: str,
        data: Any,
        *,
        grounding_mode: str,
        source_tier: str,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        effective_source_tier = self._normalize_source_tier(
            str(data.get("source") or data.get("source_tier") or source_tier or "")
        )

        if effective_source_tier == "docs" or grounding_mode in {
            "baseball_explainer",
            "long_tail_entity",
        }:
            raw_docs = data.get("documents") or data.get("results") or []
            if not isinstance(raw_docs, list):
                raw_docs = []

            focus_docs = [
                doc
                for doc in raw_docs
                if isinstance(doc, dict) and self._doc_contains_query_focus(query, doc)
            ]
            selected_docs = focus_docs[:2] if focus_docs else raw_docs[:1]

            compact_docs: List[Dict[str, Any]] = []
            for doc in selected_docs:
                if not isinstance(doc, dict):
                    continue
                excerpt = self._clean_answer_prompt_snippet(doc.get("content"))
                if not excerpt:
                    continue
                meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
                compact_doc: Dict[str, Any] = {"excerpt": excerpt}
                knowledge_type = meta.get("knowledge_type")
                freshness = meta.get("freshness")
                if knowledge_type:
                    compact_doc["knowledge_type"] = knowledge_type
                if freshness:
                    compact_doc["freshness"] = freshness
                compact_docs.append(compact_doc)

            return {
                "source": data.get("source", "verified_docs"),
                "found": bool(data.get("found")) and bool(compact_docs),
                "documents": compact_docs,
            }

        if effective_source_tier == "web" or grounding_mode == "latest_info":
            raw_results = data.get("results") or data.get("items") or []
            if not isinstance(raw_results, list):
                raw_results = []

            compact_results: List[Dict[str, Any]] = []
            for item in raw_results[:2]:
                if not isinstance(item, dict):
                    continue
                summary = self._clean_answer_prompt_snippet(
                    " ".join(
                        part
                        for part in [
                            str(item.get("title", "") or "").strip(),
                            str(item.get("snippet", "") or "").strip(),
                        ]
                        if part
                    ),
                    max_chars=240,
                )
                if not summary:
                    continue
                compact_item: Dict[str, Any] = {"summary": summary}
                published_at = item.get("published_at")
                if published_at:
                    compact_item["published_at"] = published_at
                compact_results.append(compact_item)

            compact_payload: Dict[str, Any] = {
                "source": data.get("source", "manual_request"),
                "found": bool(data.get("found", compact_results)),
                "results": compact_results,
            }
            if data.get("as_of_date"):
                compact_payload["as_of_date"] = data.get("as_of_date")
            return compact_payload

        return data

    def _optimize_team_prompt_payload(self, data: Any) -> Any:
        """팀 분석 프롬프트용 payload를 추가 압축해 답변 지연을 줄입니다."""
        if not isinstance(data, dict):
            return data

        optimized = dict(data)

        # 타자/투수 상위 표본은 2명만 유지하고 핵심 필드만 남깁니다.
        top_batters = optimized.get("top_batters")
        if isinstance(top_batters, list):
            compact_batters = []
            for row in top_batters[:2]:
                if not isinstance(row, dict):
                    continue
                compact_batters.append(
                    {
                        "player_name": row.get("player_name"),
                        "avg": row.get("avg"),
                        "ops": row.get("ops"),
                        "home_runs": row.get("home_runs"),
                    }
                )
            optimized["top_batters"] = compact_batters

        top_pitchers = optimized.get("top_pitchers")
        if isinstance(top_pitchers, list):
            compact_pitchers = []
            for row in top_pitchers[:2]:
                if not isinstance(row, dict):
                    continue
                compact_pitchers.append(
                    {
                        "player_name": row.get("player_name"),
                        "era": row.get("era"),
                        "whip": row.get("whip"),
                        "wins": row.get("wins"),
                        "saves": row.get("saves"),
                    }
                )
            optimized["top_pitchers"] = compact_pitchers

        # 일반 리스트형 필드는 앞부분만 유지
        for key in ("leaderboard", "games", "results", "items"):
            value = optimized.get(key)
            if isinstance(value, list) and len(value) > 4:
                optimized[key] = value[:4]

        # metrics는 핵심만 유지
        metrics = optimized.get("metrics")
        if isinstance(metrics, dict):
            compact_metrics: Dict[str, Any] = {}
            batting = metrics.get("batting")
            pitching = metrics.get("pitching")
            if isinstance(batting, dict):
                compact_metrics["batting"] = {
                    "avg": batting.get("avg"),
                    "ops": batting.get("ops"),
                    "total_hr": batting.get("total_hr"),
                }
            if isinstance(pitching, dict):
                compact_metrics["pitching"] = {
                    "era_rank": pitching.get("era_rank"),
                    "qs_rate": pitching.get("qs_rate"),
                }
            if compact_metrics:
                optimized["metrics"] = compact_metrics

        return optimized

    def _summarize_team_tool_data_for_prompt(self, data: Any) -> Any:
        """팀 분석 fast-path에서 프롬프트 입력용 핵심 필드만 유지합니다."""
        if not isinstance(data, dict):
            return data

        team_name = data.get("team_name")
        year = data.get("year")

        if "top_batters" in data or "top_pitchers" in data:
            top_batters = []
            for row in data.get("top_batters", [])[:3]:
                if not isinstance(row, dict):
                    continue
                top_batters.append(
                    {
                        "player_name": row.get("player_name"),
                        "avg": row.get("avg"),
                        "ops": row.get("ops"),
                        "home_runs": row.get("home_runs"),
                        "rbi": row.get("rbi"),
                    }
                )

            top_pitchers = []
            for row in data.get("top_pitchers", [])[:3]:
                if not isinstance(row, dict):
                    continue
                top_pitchers.append(
                    {
                        "player_name": row.get("player_name"),
                        "era": row.get("era"),
                        "whip": row.get("whip"),
                        "wins": row.get("wins"),
                        "saves": row.get("saves"),
                        "holds": row.get("holds"),
                    }
                )

            return {
                "team_name": team_name,
                "year": year,
                "top_batters": top_batters,
                "top_pitchers": top_pitchers,
                "found": data.get("found"),
            }

        if "metrics" in data and isinstance(data.get("metrics"), dict):
            metrics = data.get("metrics", {})
            batting = metrics.get("batting", {}) if isinstance(metrics, dict) else {}
            pitching = metrics.get("pitching", {}) if isinstance(metrics, dict) else {}
            return {
                "team_name": team_name,
                "year": year,
                "batting": {
                    "avg": batting.get("avg"),
                    "ops": batting.get("ops"),
                    "total_hr": batting.get("total_hr"),
                    "total_rbi": batting.get("total_rbi"),
                },
                "pitching": {
                    "era_rank": pitching.get("era_rank"),
                    "qs_rate": pitching.get("qs_rate"),
                    "avg_era": pitching.get("avg_era"),
                },
                "rankings": data.get("rankings", {}),
                "found": data.get("found"),
            }

        return data

    def _build_team_metric_fast_path_answer(
        self,
        query: str,
        team_name: Any,
        year: Any,
        batting: Dict[str, Any],
        pitching: Dict[str, Any],
        rankings: Dict[str, Any],
    ) -> Optional[str]:
        query_lower = query.lower()
        team_label = self._format_team_display_name(team_name)
        if "두산" in query:
            team_label = "두산 베어스"
        elif "lg" in query_lower:
            team_label = "LG 트윈스"
        elif "ssg" in query_lower:
            team_label = "SSG 랜더스"
        elif "kia" in query_lower:
            team_label = "KIA 타이거즈"
        year_label = self._format_deterministic_metric(year)
        season_label = f"{year_label}년 {team_label}"

        batting_avg = batting.get("avg")
        batting_ops = batting.get("ops")
        batting_avg_rank = rankings.get("batting_avg")
        batting_ops_rank = rankings.get("batting_ops")
        total_hr = batting.get("total_hr")
        total_rbi = batting.get("total_rbi")
        avg_era = pitching.get("avg_era")
        era_rank = pitching.get("era_rank")
        qs_rate = pitching.get("qs_rate")

        if "타율" in query_lower:
            if batting_avg is None:
                return None
            answer = f"{season_label} 팀 타율은 {self._format_deterministic_metric(batting_avg)}입니다."
            if batting_avg_rank:
                answer += f" 팀 타율 순위는 {self._format_deterministic_metric(batting_avg_rank)}입니다."
            if batting_ops is not None:
                answer += f" 같이 보면 팀 OPS는 {self._format_deterministic_metric(batting_ops)}입니다."
            return answer

        if "ops" in query_lower:
            if batting_ops is None:
                return None
            answer = f"{season_label} 팀 OPS는 {self._format_deterministic_metric(batting_ops)}입니다."
            if batting_ops_rank:
                answer += f" OPS 순위는 {self._format_deterministic_metric(batting_ops_rank)}입니다."
            if batting_avg is not None:
                answer += f" 팀 타율은 {self._format_deterministic_metric(batting_avg)}로 확인됩니다."
            return answer

        if any(
            token in query_lower
            for token in ["평균자책", "평균 자책", "평균자책점", "방어율", "era"]
        ):
            if avg_era is None:
                return None
            answer = f"{season_label} 팀 평균자책점은 {self._format_deterministic_metric(avg_era)}입니다."
            if era_rank:
                answer += (
                    f" 리그 순위는 {self._format_deterministic_metric(era_rank)}입니다."
                )
            if qs_rate:
                answer += f" 선발 QS 비율은 {self._format_deterministic_metric(qs_rate)}입니다."
            return answer

        if "홈런" in query_lower:
            if total_hr is None:
                return None
            answer = f"{season_label} 팀 홈런은 {self._format_deterministic_metric(total_hr)}개입니다."
            if batting_ops is not None:
                answer += (
                    f" 팀 OPS는 {self._format_deterministic_metric(batting_ops)}입니다."
                )
            if batting_ops_rank:
                answer += f" OPS 순위는 {self._format_deterministic_metric(batting_ops_rank)}입니다."
            return answer

        if "타점" in query_lower:
            if total_rbi is None:
                return None
            answer = f"{season_label} 팀 타점은 {self._format_deterministic_metric(total_rbi)}점입니다."
            if batting_avg is not None:
                answer += f" 팀 타율은 {self._format_deterministic_metric(batting_avg)}입니다."
            if batting_ops is not None:
                answer += (
                    f" 팀 OPS는 {self._format_deterministic_metric(batting_ops)}입니다."
                )
            return answer

        return None

    def _build_known_explainer_answer(
        self,
        query: str,
        *,
        grounding_mode: str,
        source_tier: str,
    ) -> Optional[str]:
        del grounding_mode
        del source_tier
        query_lower = query.lower()

        if "babip" in query_lower:
            return (
                "BABIP는 인플레이된 타구가 안타가 될 비율을 보는 지표입니다.\n\n"
                "보통 홈런과 삼진처럼 수비가 개입하지 않는 결과는 빼고, 필드 안으로 들어간 타구가 얼마나 안타로 이어졌는지를 봅니다.\n\n"
                "즉 타자의 타구 질, 수비, 운이 함께 섞이는 지표라서 타율이나 장타율 같은 다른 기록과 같이 해석하는 게 중요합니다."
            )

        if "whip" in query_lower:
            return (
                "WHIP는 투수가 1이닝당 몇 명의 주자를 내보냈는지 보는 지표입니다.\n\n"
                "피안타와 볼넷으로 허용한 주자를 합쳐 투구 이닝으로 나누기 때문에, 값이 낮을수록 주자 허용이 적다는 뜻입니다.\n\n"
                "즉 평균자책점과 함께 보면 실점뿐 아니라 이닝마다 얼마나 깔끔하게 주자를 막았는지도 같이 판단할 수 있습니다."
            )

        if "qs" in query_lower:
            return (
                "QS는 선발투수가 6이닝 이상을 던지면서 3자책점 이하로 막았다는 뜻입니다.\n\n"
                "핵심은 선발이 최소 6이닝을 책임지고, 경기 운영을 무너뜨리지 않을 정도로 실점을 억제했는지를 보는 기준이라는 점입니다.\n\n"
                "즉 선발투수의 안정적인 기본 역할 수행 여부를 빠르게 확인할 때 자주 쓰는 지표로 이해하면 됩니다."
            )

        if (
            "부상자 명단" in query_lower
            or query_lower.strip() == "il"
            or " il " in f" {query_lower} "
        ):
            return (
                "IL은 부상 때문에 바로 경기에 뛰기 어려운 선수를 관리하는 부상자 명단 개념으로 이해하면 됩니다.\n\n"
                "팬 입장에서는 선수가 일시적으로 엔트리 운영에서 빠지고, 복귀 여부는 공식 엔트리 변동과 등록·말소 기록으로 확인한다고 보면 가장 정확합니다.\n\n"
                "다만 최소 말소 일수나 세부 운영 방식은 시즌 규정이 바뀔 수 있으니, 최종 확인은 해당 시즌 공식 공시와 운영 규정을 기준으로 보는 게 맞습니다."
            )

        if "체인지업" in query_lower:
            return (
                "체인지업은 직구와 비슷한 폼으로 던지지만 실제 구속은 더 느리게 들어오는 변화구입니다.\n\n"
                "타자는 직구 타이밍으로 스윙을 시작했다가 공이 늦게 들어오면 중심이 무너지기 쉬워서 헛스윙이나 약한 타구가 나오기 쉽습니다.\n\n"
                "즉 빠른 공처럼 보이게 속인 뒤 속도 차로 타이밍을 빼앗는 공이라고 보면 됩니다."
            )

        if (
            "wpa" in query_lower
            or "승리확률기여도" in query_lower
            or "승리 확률 기여도" in query_lower
        ):
            return (
                "WPA는 한 플레이가 팀의 승리 확률을 얼마나 올리거나 내렸는지 보여주는 지표입니다.\n\n"
                "같은 안타라도 9회 동점 상황에서 나온 안타는 승리 확률을 크게 바꾸기 때문에 WPA가 크게 오르고, 점수 차가 큰 상황의 안타는 변화폭이 작습니다.\n\n"
                "즉 단순 누적 기록이 아니라 경기 상황의 중요도까지 반영해서, 누가 결정적인 순간에 승리에 더 크게 기여했는지를 보는 지표라고 이해하면 됩니다."
            )

        if "히트앤런" in query_lower or "히트 앤 런" in query_lower:
            return (
                "히트앤런은 주자가 미리 스타트를 끊는 동시에 타자가 반드시 배트를 내 공을 맞혀 주자를 보내는 작전입니다.\n\n"
                "도루와 달리 타자가 스윙을 해야 한다는 점이 핵심이라서, 주자 진루를 돕는 대신 타자가 헛스윙하면 주자가 잡히기 쉽습니다.\n\n"
                "즉 수비를 흔들면서 주자를 적극적으로 움직이기 위한 고위험 작전이라고 이해하면 됩니다."
            )

        if "필승조" in query_lower:
            return (
                "필승조는 팀이 앞서 있을 때 승리를 지키기 위해 7회 이후 중요한 상황에 주로 투입하는 핵심 불펜입니다.\n\n"
                "보통 가장 믿는 셋업맨과 마무리투수, 또는 그 직전 구간을 맡는 투수들이 이 역할을 담당합니다.\n\n"
                "즉 접전 리드를 끝까지 지켜야 할 때 먼저 떠올리는 불펜 묶음이라고 보면 됩니다."
            )

        if (
            "단디" in query_lower
            or "쎄리" in query_lower
            or ("nc" in query_lower and "마스코트" in query_lower)
        ):
            return (
                "NC 다이노스의 대표 마스코트는 단디와 쎄리입니다.\n\n"
                "단디는 푸른색 티라노사우루스를 모티브로 한 캐릭터이고, 이름은 경상도 사투리로 '야무지게'라는 느낌에서 왔다고 이해하면 됩니다.\n\n"
                "쎄리는 목이 긴 브라키오사우루스 계열 이미지의 캐릭터로, 공을 세게 때리자는 의미를 담은 이름입니다.\n\n"
                "즉 NC는 공룡 구단 정체성을 살려 두 마스코트를 함께 쓰는 팀이라고 보면 됩니다."
            )

        if "프레이밍" in query_lower:
            return (
                "포수 프레이밍은 포수가 공을 받는 동작으로 볼과 스트라이크의 경계 판정을 더 유리하게 보이게 만드는 기술입니다.\n\n"
                "핵심은 스트라이크존 근처로 들어온 공을 포수가 최대한 자연스럽게 잡아 심판의 스트라이크 판정 가능성을 높이는 데 있습니다.\n\n"
                "즉 포수의 수비 기술이 스트라이크 판정에 간접적으로 영향을 줄 수 있는 영역이라고 이해하면 됩니다."
            )

        if "수비 시프트" in query_lower or (
            "시프트" in query_lower and "수비" in query_lower
        ):
            return (
                "수비 시프트는 타자의 타구 방향 성향에 맞춰 야수들의 수비 위치를 평소와 다르게 옮겨 두는 전술입니다.\n\n"
                "예를 들어 당겨 치는 타자라면 내야수나 외야수를 그 방향으로 더 붙여서 자주 가는 타구를 먼저 막으려는 의도가 큽니다.\n\n"
                "즉 타자별 타구 패턴을 보고 수비 위치를 조정해 안타 확률을 낮추는 전략이라고 이해하면 됩니다."
            )

        if "번트" in query_lower:
            return (
                "번트는 강하게 치지 않고 공을 짧게 죽여 주자를 진루시키거나 타자가 살아나가려는 타격입니다.\n\n"
                "희생번트는 타자 아웃을 감수하고 주자 진루를 얻는 선택이라서, 한 점이 중요한 상황에서 자주 검토됩니다.\n\n"
                "즉 번트는 아웃 하나와 주자 이동을 맞바꾸며 득점 기회를 설계하는 작전이라고 보면 됩니다."
            )

        if "병살타" in query_lower:
            return (
                "병살타가 치명적인 이유는 한 번의 타구로 아웃 2개가 동시에 나오기 쉽기 때문입니다.\n\n"
                "주자가 있는 상황에서 병살이 나오면 주자도 사라지고 타자도 아웃돼서, 순식간에 득점 기회가 크게 줄어듭니다.\n\n"
                "즉 공격 흐름이 끊기고 남은 이닝 아웃카운트 부담이 커져서, 좋은 찬스를 한 번에 잃는 결과가 되기 쉽습니다."
            )

        if "불펜 과부하" in query_lower:
            return (
                "불펜 과부하는 특정 불펜 투수들이 너무 자주 등판하거나 짧은 휴식으로 반복 투입돼 부담이 커진 상태를 뜻합니다.\n\n"
                "보통 최근 연투 횟수, 등판 간 휴식일, 특정 필승조에게 이닝이 몰리는지, 접전에서 같은 투수들이 계속 나오는지를 함께 봅니다.\n\n"
                "이 상태가 심해지면 구위와 제구가 떨어지고, 시즌 후반으로 갈수록 접전 운영이 흔들릴 가능성이 커집니다.\n\n"
                "즉 불펜 과부하는 한두 경기 결과보다도 불펜 운용이 특정 투수에게 얼마나 쏠려 있는지를 보고 판단한다고 이해하면 됩니다."
            )

        if "득점권 생산성" in query_lower:
            return (
                "득점권 생산성은 주자가 2루나 3루에 있을 때 팀이나 타자가 실제로 점수를 만들어내는 효율을 보는 개념입니다.\n\n"
                "단순 득점권 타율만 보는 것보다 적시타, 희생플라이, 병살 회피, 볼넷으로 이어지는 타석 내용까지 함께 보는 게 더 정확합니다.\n\n"
                "표본이 짧으면 운의 영향도 커지기 때문에, 시즌 전체 득점력이나 출루율, 장타력과 같이 놓고 해석해야 과대평가를 줄일 수 있습니다.\n\n"
                "즉 득점권 생산성은 찬스에서 얼마나 점수를 실속 있게 가져오느냐를 보는 지표라고 이해하면 됩니다."
            )

        if ("좌타자" in query_lower and "우타자" in query_lower) or (
            "좌타" in query_lower and "우타" in query_lower
        ):
            return (
                "좌타자와 우타자의 차이는 단순히 서는 방향만이 아니라 수비 시프트, 투수 매치업, 타구 방향, 주루 이점까지 경기 운영에 영향을 준다는 점에 있습니다.\n\n"
                "예를 들어 좌타자는 1루까지 한 걸음이 짧아 내야안타나 병살 회피에서 조금 유리할 수 있고, 우타자는 좌완·우완 상대 체감이나 밀어치기 패턴이 다르게 나타나기도 합니다.\n\n"
                "감독 입장에서는 상대 선발 유형, 불펜 좌우 조합, 구장 특성까지 함께 보면서 좌타·우타 배치를 조정합니다.\n\n"
                "즉 좌타자와 우타자 차이는 개인 습관의 문제가 아니라 라인업 구성과 매치업 전략에 직접 연결되는 요소라고 보면 됩니다."
            )

        if "투수 교체" in query_lower or "교체 타이밍" in query_lower:
            return (
                "투수 교체 타이밍은 보통 투수 구위가 떨어지는 신호와 경기 상황을 함께 보고 판단합니다.\n\n"
                "투구 수가 많아졌는지, 구속이나 제구가 흔들리는지, 같은 타순을 세 번째로 상대하는지처럼 투수 쪽 신호를 먼저 봅니다.\n\n"
                "여기에 점수 차, 주자 상황, 다음 타자 유형, 불펜 가용 여부까지 겹쳐서 지금 버티게 할지 바로 바꿀지를 정합니다.\n\n"
                "즉 한 가지 숫자만 보는 게 아니라 투수 컨디션과 경기 맥락을 같이 묶어 보는 게 일반적인 교체 기준이라고 이해하면 됩니다."
            )

        if "인필드 플라이" in query_lower:
            return (
                "인필드 플라이는 특정 상황의 내야 뜬공에 대해 타자를 자동 아웃으로 선언하는 규칙입니다.\n\n"
                "보통 무사나 1사에 주자 1, 2루 또는 만루일 때 내야수가 평범하게 처리할 수 있는 뜬공이면, 수비가 고의로 공을 떨어뜨려 이중 플레이를 노리지 못하게 타자 아웃을 먼저 선언합니다.\n\n"
                "즉 내야 뜬공 상황에서 수비의 편법을 막기 위해 타자 아웃을 바로 인정하는 보호 규칙이라고 이해하면 됩니다."
            )

        if "태그업" in query_lower:
            return (
                "태그업은 플라이볼이 잡힌 뒤 주자가 원래 있던 베이스를 다시 밟고 다음 베이스로 진루하는 플레이입니다.\n\n"
                "주자는 수비가 공을 잡기 전에 먼저 출발할 수는 있지만, 실제로 진루하려면 플라이가 잡힌 뒤 원래 베이스를 다시 터치해야 합니다.\n\n"
                "즉 플라이 아웃 상황에서 주자가 베이스를 다시 밟고 움직여야 합법적으로 진루할 수 있는 규칙이라고 보면 됩니다."
            )

        if "비디오 판독" in query_lower:
            return (
                "비디오 판독은 현장 판정이 애매한 장면을 영상으로 다시 확인해 바로잡는 절차입니다.\n\n"
                "대표적으로 홈런 여부, 세이프와 아웃 판정, 페어와 파울처럼 경기 결과에 직접 영향을 주는 플레이가 주요 판독 대상입니다.\n\n"
                "즉 핵심 장면의 판정을 더 정확하게 하기 위해 홈런과 아웃 여부 같은 상황을 다시 확인하는 제도라고 보면 됩니다."
            )

        if "승리투수" in query_lower or "승리 투수" in query_lower:
            return (
                "승리투수는 자기 팀이 리드를 잡아 최종적으로 승리한 경기에서 가장 직접적으로 승리에 기여한 투수에게 주어집니다.\n\n"
                "선발투수는 보통 5이닝 이상을 던져야 기본 승리 요건을 충족하고, 그보다 짧게 던졌다면 공식 기록원이 가장 효과적이었다고 판단한 구원투수가 승리투수가 될 수 있습니다.\n\n"
                "즉 선발투수의 5이닝 기준과 실제 승리 기여도를 함께 봐서 승리투수를 결정한다고 이해하면 됩니다."
            )

        if "보크" in query_lower:
            return (
                "보크는 투수가 주자를 속이거나 투구 동작 규정을 어겨 주자에게 진루권이 주어지는 반칙입니다.\n\n"
                "핵심은 투구 동작을 하다가 멈추거나, 견제 동작을 속이듯 가져가거나, 세트 포지션 규칙을 어기는 식으로 주자를 혼란스럽게 만드는 상황입니다.\n\n"
                "즉 주자가 있을 때 투수 동작의 합법성을 엄격하게 보는 규정이라고 이해하면 됩니다."
            )

        if "풀카운트" in query_lower:
            return (
                "풀카운트는 볼 3개, 스트라이크 2개가 된 상태를 말합니다.\n\n"
                "이 상황에서는 다음 공 하나로 볼넷이나 삼진, 인플레이 결과가 바로 갈릴 수 있어서 투수와 타자 모두 선택이 더 극단적으로 중요해집니다.\n\n"
                "주자가 있으면 보통 스타트를 끊기 쉬워서 타자는 맞히는 능력, 투수는 승부구 완성도가 특히 크게 작용합니다.\n\n"
                "즉 풀카운트는 한 공의 가치가 가장 커지는 대표적인 승부 카운트라고 이해하면 됩니다."
            )

        if "라인드라이브" in query_lower and "뜬공" in query_lower:
            return (
                "라인드라이브 성향과 뜬공 성향의 차이는 타구 각도와 그에 따른 결과 기대값에서 갈립니다.\n\n"
                "라인드라이브는 낮고 강하게 뻗는 타구라 안타로 이어질 확률이 비교적 높고, 뜬공은 멀리 보내면 장타나 홈런이 될 수 있지만 평범하면 아웃으로 끝나기 쉽습니다.\n\n"
                "그래서 같은 타구 질이라도 선수 유형에 따라 라인드라이브는 안정적인 출루, 뜬공은 장타 잠재력 쪽으로 해석하는 경우가 많습니다.\n\n"
                "즉 라인드라이브는 안타 생산성, 뜬공은 장타 위험과 보상을 함께 가진 타구 성향이라고 보면 됩니다."
            )

        if "낫아웃" in query_lower:
            return (
                "낫아웃은 삼진이 나왔더라도 포수가 공을 완전히 포구하지 못하면 타자가 1루로 뛸 수 있는 규정입니다.\n\n"
                "보통 1루가 비어 있거나 2아웃일 때 성립하고, 포수가 공을 제대로 잡으면 그대로 삼진 아웃으로 끝납니다.\n\n"
                "즉 삼진이 곧바로 플레이 종료를 뜻하는 게 아니라, 포구 여부에 따라 타자가 살아날 수 있는 예외 규정입니다."
            )

        if "아웃카운트" in query_lower:
            return (
                "야구에서 한 이닝 공격은 3아웃이 되면 끝나고 공수 교대가 일어납니다.\n\n"
                "3아웃 구조 덕분에 공격 기회가 너무 짧지도 길지도 않게 유지되고, 이닝마다 흐름과 전략이 분명하게 나뉩니다.\n\n"
                "즉 3아웃은 한 팀의 공격 단위를 끊어 이닝과 공수 교대 리듬을 만드는 기본 규칙이라고 보면 됩니다."
            )

        if "대표 라이벌" in query_lower or (
            "라이벌" in query_lower and "매치업" in query_lower
        ):
            return (
                "KBO에서 대표 라이벌 매치업으로 가장 자주 거론되는 건 두산과 LG의 잠실 라이벌입니다.\n\n"
                "같은 잠실구장을 쓰고 수도권 팬층이 겹쳐서 경기 자체의 긴장감과 화제성이 꾸준히 큰 편입니다.\n\n"
                "그 밖에 롯데와 삼성 같은 영남권 맞대결, 롯데와 KIA처럼 인기 구단끼리 붙는 전통 대진도 자주 라이벌전으로 언급됩니다.\n\n"
                "다만 KBO가 공식적으로 몇 쌍만 딱 지정한 구조는 아니라서, 시대와 성적에 따라 팬들이 체감하는 대표 라이벌은 조금씩 달라질 수 있습니다."
            )

        return None

    def _build_known_latest_answer(
        self,
        query: str,
        *,
        grounding_mode: str,
        source_tier: str,
        as_of_date: str,
    ) -> Optional[str]:
        if grounding_mode != "latest_info" and source_tier != "web":
            return None

        if "맞대결" not in query:
            query_lower = query.lower()
            if any(
                token in query_lower
                for token in ["최근 경기", "최근 활약", "요즘 활약"]
            ):
                player_match = re.search(
                    r"([A-Za-z가-힣]+)\s+최근", re.sub(r"\d{4}년", "", query)
                )
                if not player_match:
                    return None
                player_name = player_match.group(1).strip()
                as_of_label = as_of_date or datetime.now().strftime("%Y-%m-%d")
                return (
                    f"{as_of_label} 기준으로 {player_name}의 최근 경기 활약은 현재 연결된 최신 자료에서 직접 확인되지 않았습니다.\n\n"
                    f"지금 확보된 자료가 {player_name}의 실제 최근 경기 기록이나 활약 요약이 아니라서, 최근 폼을 추정해서 말하진 않겠습니다.\n\n"
                    "최근 경기 로그나 공식 기록 자료가 붙으면 경기별 활약 흐름으로 다시 정리할 수 있습니다."
                )
            if "5위" in query_lower and any(
                token in query_lower for token in ["싸움", "순위", "경쟁"]
            ):
                as_of_label = as_of_date or datetime.now().strftime("%Y-%m-%d")
                return (
                    f"{as_of_label} 기준으로 현재 5위 순위 경쟁 상황은 연결된 최신 자료에서 직접 확인되지 않았습니다.\n\n"
                    "지금 확보된 자료만으로는 어느 팀이 5위 싸움에서 앞선다고 단정할 수 없어서, 순위 경쟁 상황을 추정해서 말하진 않겠습니다.\n\n"
                    "최신 순위표나 최근 경기 결과가 붙으면 5위 경쟁 구도를 바로 다시 정리할 수 있습니다."
                )
            return None

        cleaned_query = re.sub(r"\d{4}년", "", query)
        matchup = re.search(
            r"([A-Za-z가-힣]+)\s*(?:와|과|vs\.?|VS\.?)\s*([A-Za-z가-힣]+)",
            cleaned_query,
        )
        if not matchup:
            return None

        team1 = matchup.group(1).strip()
        team2 = matchup.group(2).strip()
        as_of_label = as_of_date or datetime.now().strftime("%Y-%m-%d")
        return (
            f"{as_of_label} 기준으로 {team1}와 {team2}의 최근 맞대결 기록은 현재 연결된 최신 자료에서 직접 확인되지 않았습니다.\n\n"
            f"지금 확보된 자료가 두 팀의 실제 맞대결 결과가 아니라서, {team1}와 {team2} 중 어느 쪽이 우세하다고 추정해서 말하진 않겠습니다.\n\n"
            "맞대결 전적이나 경기 결과가 확인되는 자료가 붙으면 최근 경기 순서대로 다시 정리할 수 있습니다."
        )

    def _game_matches_team_code(
        self, game: Dict[str, Any], team_code: Optional[str]
    ) -> bool:
        if not team_code:
            return True
        candidates = {
            str(game.get("home_team") or "").strip(),
            str(game.get("away_team") or "").strip(),
            str(game.get("home_team_code") or "").strip(),
            str(game.get("away_team_code") or "").strip(),
        }
        return team_code in candidates

    def _summarize_recent_team_games(
        self,
        recent_data: Dict[str, Any],
        team_name: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        games = recent_data.get("games")
        if not isinstance(games, list) or not games:
            return None

        wins = 0
        losses = 0
        draws = 0
        previews: List[str] = []
        streak_kind: Optional[str] = None
        streak_count = 0

        for game in games:
            if not isinstance(game, dict):
                continue
            team_result = str(game.get("team_result") or "").strip().lower()
            if team_result == "win":
                wins += 1
                current_kind = "win"
            elif team_result == "loss":
                losses += 1
                current_kind = "loss"
            else:
                draws += 1
                current_kind = "draw"

            if streak_kind is None:
                streak_kind = current_kind
                streak_count = 1
            elif streak_kind == current_kind:
                streak_count += 1

            opponent = self._format_team_display_name(
                game.get("opponent_team")
                or game.get("opponent_team_code")
                or game.get("opponent_team_name")
            )
            if opponent == "확인 불가":
                opponent = "상대팀"
            game_date = self._format_deterministic_metric(game.get("game_date"))
            home_score = self._format_deterministic_metric(game.get("home_score"))
            away_score = self._format_deterministic_metric(game.get("away_score"))
            result_label = {"win": "승", "loss": "패", "draw": "무"}.get(
                team_result, "결과 확인 필요"
            )
            previews.append(
                f"{game_date} {opponent}전 {away_score}-{home_score} ({result_label})"
            )

        sample_size = len([game for game in games if isinstance(game, dict)])
        if sample_size <= 0:
            return None

        record_text = f"{wins}승 {losses}패"
        if draws:
            record_text += f" {draws}무"
        streak_text = None
        if streak_kind in {"win", "loss"} and streak_count > 0:
            streak_text = f"{streak_count}연{'승' if streak_kind == 'win' else '패'}"

        return {
            "team_name": team_name or recent_data.get("team_name"),
            "record_text": record_text,
            "sample_size": sample_size,
            "streak_text": streak_text,
            "recent_preview": ", ".join(previews[:3]),
        }

    def _group_lineups_by_team(
        self, lineups: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for lineup in lineups:
            team_label = self._format_team_display_name(
                lineup.get("team_code") or lineup.get("team_name")
            )
            if team_label == "확인 불가":
                team_label = str(
                    lineup.get("team_name") or lineup.get("team_code") or "팀"
                )
            grouped.setdefault(team_label, []).append(lineup)

        for entries in grouped.values():
            entries.sort(
                key=lambda item: (
                    item.get("batting_order") is None,
                    item.get("batting_order") or 99,
                    str(item.get("player_name") or ""),
                )
            )
        return grouped

    def _build_team_bundle_answer(
        self,
        query: str,
        tool_results: List[ToolResult],
        *,
        chat_mode: bool,
    ) -> Optional[str]:
        has_recent_games = False
        has_team_summary = False
        has_team_metrics = False
        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue
            data = result.data
            if "games" in data and data.get("team_name") and "date" not in data:
                has_recent_games = True
            if "top_batters" in data or "top_pitchers" in data:
                has_team_summary = True
            if "metrics" in data or "fatigue_index" in data:
                has_team_metrics = True

        if not has_recent_games or not (has_team_summary or has_team_metrics):
            return None
        return self._build_fast_path_answer(query, tool_results, chat_mode=chat_mode)

    def _build_schedule_bundle_answer(
        self,
        query: str,
        tool_results: List[ToolResult],
    ) -> Optional[str]:
        query_lower = query.lower()
        if not any(
            token in query_lower for token in ["라인업", "선발", "스타팅", "타순"]
        ):
            return None

        schedule_data: Optional[Dict[str, Any]] = None
        lineup_data: Optional[Dict[str, Any]] = None
        box_score_data: Optional[Dict[str, Any]] = None
        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue
            data = result.data
            if "lineups" in data:
                lineup_data = data
            elif "games" in data and "date" in data:
                schedule_data = data
            elif "games" in data and any(
                isinstance(game, dict) and isinstance(game.get("box_score"), dict)
                for game in data.get("games", [])
            ):
                box_score_data = data

        if lineup_data is None and box_score_data is None:
            return None

        team_code = self._detect_team_alias_from_query(query)
        schedule_games = list((schedule_data or {}).get("games") or [])
        if team_code:
            schedule_games = [
                game
                for game in schedule_games
                if isinstance(game, dict)
                and self._game_matches_team_code(game, team_code)
            ]

        box_games = list((box_score_data or {}).get("games") or [])
        if team_code:
            box_games = [
                game
                for game in box_games
                if isinstance(game, dict)
                and self._game_matches_team_code(game, team_code)
            ]

        lineup_entries = [
            entry
            for entry in list((lineup_data or {}).get("lineups") or [])
            if isinstance(entry, dict)
        ]
        grouped_lineups = self._group_lineups_by_team(lineup_entries)
        if team_code:
            filtered_groups = {
                team_label: entries
                for team_label, entries in grouped_lineups.items()
                if any(
                    str(entry.get("team_code") or "").strip() == team_code
                    for entry in entries
                )
            }
            if filtered_groups:
                grouped_lineups = filtered_groups

        date_label = self._format_deterministic_metric(
            (schedule_data or {}).get("date")
            or ((lineup_data or {}).get("query_params") or {}).get("date")
            or ((box_score_data or {}).get("query_params") or {}).get("date")
        )
        if date_label == "확인 불가":
            date_label = "해당 경기일"

        if not team_code and len(schedule_games) > 1 and (grouped_lineups or box_games):
            return (
                f"{date_label}에는 라인업/박스스코어를 볼 수 있는 경기가 여러 개라 한 경기로 바로 못 좁히겠습니다.\n\n"
                "팀명이나 game_id를 같이 주시면 일정, 선발 라인업, 박스스코어를 한 번에 이어서 정리해드리겠습니다."
            )

        lines = [
            f"{date_label} 경기 기준으로 확인된 일정과 상세 정보는 이렇게 보입니다."
        ]
        if schedule_games:
            first_game = schedule_games[0]
            away_team = self._format_team_display_name(first_game.get("away_team"))
            home_team = self._format_team_display_name(first_game.get("home_team"))
            stadium = self._format_deterministic_metric(first_game.get("stadium"))
            lines.append(
                f"일정상 매치는 {away_team} 대 {home_team}이고, 장소는 {stadium}입니다."
            )

        for team_label, entries in list(grouped_lineups.items())[:2]:
            preview = ", ".join(
                f"{entry.get('batting_order', '?')}번 {entry.get('player_name', '선수')}"
                for entry in entries[:3]
            )
            if preview:
                lines.append(f"{team_label} 선발 라인업은 {preview} 순으로 확인됩니다.")

        if box_games:
            game = box_games[0]
            away_team = self._format_team_display_name(
                game.get("away_team_code") or game.get("away_team")
            )
            home_team = self._format_team_display_name(
                game.get("home_team_code") or game.get("home_team")
            )
            away_score = self._format_deterministic_metric(game.get("away_score"))
            home_score = self._format_deterministic_metric(game.get("home_score"))
            lines.append(
                f"박스스코어는 {away_team} {away_score}-{home_score} {home_team} 기준으로 연결되고, 이닝별 상세 득점 정보까지 확인 가능합니다."
            )

        return "\n\n".join(lines[:4]) if len(lines) > 1 else None

    def _match_leaderboard_entry(
        self,
        leaderboard: List[Dict[str, Any]],
        *,
        player_name: Optional[str],
        validation_data: Optional[Dict[str, Any]],
    ) -> Optional[tuple[int, Dict[str, Any]]]:
        candidate_names = {str(player_name or "").strip().casefold()}
        for row in (validation_data or {}).get("found_players", []):
            if not isinstance(row, dict):
                continue
            resolved_name = str(row.get("player_name") or "").strip().casefold()
            if resolved_name:
                candidate_names.add(resolved_name)

        for index, entry in enumerate(leaderboard, start=1):
            entry_name = str(
                entry.get("player_name")
                or entry.get("name")
                or entry.get("player")
                or ""
            ).strip()
            if entry_name.casefold() in candidate_names:
                return index, entry
        return None

    def _build_player_bundle_answer(
        self,
        tool_results: List[ToolResult],
    ) -> Optional[str]:
        validation_data: Optional[Dict[str, Any]] = None
        stats_data: Optional[Dict[str, Any]] = None
        leaderboard_data: Optional[Dict[str, Any]] = None

        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue
            data = result.data
            if "found_players" in data and "exists" in data:
                validation_data = data
            elif "batting_stats" in data or "pitching_stats" in data:
                stats_data = data
            elif "leaderboard" in data:
                leaderboard_data = data

        if stats_data is None or leaderboard_data is None:
            return None

        if validation_data and validation_data.get("exists") is False:
            return (
                f"{validation_data.get('year', '해당 시즌')}년 기준으로는 "
                f"{validation_data.get('player_name', '해당 선수')} 기록이 바로 확인되지 않습니다.\n\n"
                "선수명 표기나 시즌을 한 번 더 좁혀주시면 검증 후 기록과 순위를 다시 묶어서 볼 수 있습니다."
            )

        resolved_player_name = (
            ((stats_data.get("batting_stats") or {}).get("player_name"))
            or ((stats_data.get("pitching_stats") or {}).get("player_name"))
            or stats_data.get("player_name")
        )
        stats_answer = self._build_player_stats_chat_answer(stats_data)
        if not stats_answer:
            return None

        leaderboard = leaderboard_data.get("leaderboard") or []
        metric_key = str(leaderboard_data.get("stat_name") or "").lower()
        metric_label = {
            "ops": "OPS",
            "home_runs": "홈런",
            "avg": "타율",
            "rbi": "타점",
            "era": "ERA",
            "whip": "WHIP",
            "strikeouts": "탈삼진",
            "wins": "다승",
            "saves": "세이브",
            "holds": "홀드",
        }.get(metric_key, metric_key)
        matched_entry = self._match_leaderboard_entry(
            leaderboard,
            player_name=resolved_player_name,
            validation_data=validation_data,
        )
        if matched_entry:
            rank, entry = matched_entry
            metric_value = self._format_deterministic_metric(
                self._extract_leaderboard_value(entry, metric_key)
            )
            rank_line = f"리그 {metric_label} 리더보드에서는 현재 {rank}위이고, 값은 {metric_value}입니다."
        else:
            rank_line = (
                f"리그 {metric_label} 상위 {len(leaderboard)}명 리더보드에는 "
                f"{resolved_player_name or '해당 선수'} 이름이 바로 보이지 않습니다."
            )

        candidate_rows = (validation_data or {}).get("found_players") or []
        if len(candidate_rows) > 1:
            teams = [
                str(row.get("team_name") or "").strip()
                for row in candidate_rows[:3]
                if isinstance(row, dict) and str(row.get("team_name") or "").strip()
            ]
            if teams:
                return "\n\n".join(
                    [
                        stats_answer,
                        rank_line,
                        "검증 단계에서는 "
                        + ", ".join(teams)
                        + " 소속 후보까지 함께 확인했습니다.",
                    ]
                )

        return "\n\n".join([stats_answer, rank_line])

    def _build_player_comparison_answer(
        self,
        query: str,
        tool_results: List[ToolResult],
        *,
        chat_mode: bool,
    ) -> Optional[str]:
        comparison_data: Optional[Dict[str, Any]] = None
        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue
            data = result.data
            if "player1" in data and "player2" in data and "analysis" in data:
                comparison_data = data
                break

        if comparison_data is None:
            return None

        player1 = comparison_data.get("player1") or {}
        player2 = comparison_data.get("player2") or {}
        player1_name = str(player1.get("name") or "선수1").strip() or "선수1"
        player2_name = str(player2.get("name") or "선수2").strip() or "선수2"
        player_pair = f"{player1_name}과 {player2_name}"
        comparison_label = str(comparison_data.get("comparison_type") or "기준 기록")
        analysis = (
            comparison_data.get("analysis")
            if isinstance(comparison_data.get("analysis"), dict)
            else {}
        )
        player1_data = (
            player1.get("data") if isinstance(player1.get("data"), dict) else {}
        )
        player2_data = (
            player2.get("data") if isinstance(player2.get("data"), dict) else {}
        )
        query_lower = query.lower()

        def _safe_float(value: Any) -> Optional[float]:
            if value in (None, ""):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _pick(data: Dict[str, Any], keys: List[str]) -> Optional[Any]:
            return self._pick_metric_value(data, keys)

        def _rate(numerator: Any, denominator: Any) -> Optional[float]:
            numerator_value = _safe_float(numerator)
            denominator_value = _safe_float(denominator)
            if numerator_value is None or denominator_value in (None, 0):
                return None
            return numerator_value / denominator_value

        def _leader_name(
            value1: Optional[float],
            value2: Optional[float],
            *,
            lower_is_better: bool = False,
        ) -> Optional[str]:
            if value1 is None or value2 is None or value1 == value2:
                return None
            if lower_is_better:
                return player1_name if value1 < value2 else player2_name
            return player1_name if value1 > value2 else player2_name

        batting1 = (
            player1_data.get("batting_stats")
            if isinstance(player1_data.get("batting_stats"), dict)
            else None
        )
        batting2 = (
            player2_data.get("batting_stats")
            if isinstance(player2_data.get("batting_stats"), dict)
            else None
        )
        pitching1 = (
            player1_data.get("pitching_stats")
            if isinstance(player1_data.get("pitching_stats"), dict)
            else None
        )
        pitching2 = (
            player2_data.get("pitching_stats")
            if isinstance(player2_data.get("pitching_stats"), dict)
            else None
        )

        if batting1 and batting2:
            obp1 = _safe_float(_pick(batting1, ["obp", "career_obp"]))
            obp2 = _safe_float(_pick(batting2, ["obp", "career_obp"]))
            slg1 = _safe_float(_pick(batting1, ["slg", "career_slg"]))
            slg2 = _safe_float(_pick(batting2, ["slg", "career_slg"]))
            ops1 = _safe_float(_pick(batting1, ["ops", "career_ops"]))
            ops2 = _safe_float(_pick(batting2, ["ops", "career_ops"]))
            walks1 = _safe_float(_pick(batting1, ["walks", "total_walks"]))
            walks2 = _safe_float(_pick(batting2, ["walks", "total_walks"]))
            strikeouts1 = _safe_float(
                _pick(batting1, ["strikeouts", "total_strikeouts"])
            )
            strikeouts2 = _safe_float(
                _pick(batting2, ["strikeouts", "total_strikeouts"])
            )
            pa1 = _safe_float(_pick(batting1, ["plate_appearances", "total_pa"]))
            pa2 = _safe_float(_pick(batting2, ["plate_appearances", "total_pa"]))
            home_runs1 = _safe_float(_pick(batting1, ["home_runs", "total_home_runs"]))
            home_runs2 = _safe_float(_pick(batting2, ["home_runs", "total_home_runs"]))
            doubles1 = _safe_float(_pick(batting1, ["doubles", "total_doubles"]))
            doubles2 = _safe_float(_pick(batting2, ["doubles", "total_doubles"]))

            walk_rate1 = _rate(walks1, pa1)
            walk_rate2 = _rate(walks2, pa2)
            strikeout_rate1 = _rate(strikeouts1, pa1)
            strikeout_rate2 = _rate(strikeouts2, pa2)

            approach_lines: List[str] = []
            power_lines: List[str] = []
            table_rows: List[str] = []
            insight_lines: List[str] = []

            obp_leader = _leader_name(obp1, obp2)
            if obp_leader:
                approach_lines.append(
                    f"타석 접근은 출루율 기준으로 {obp_leader} 쪽이 더 안정적입니다 "
                    f"({player1_name} {self._format_deterministic_metric(obp1)}, "
                    f"{player2_name} {self._format_deterministic_metric(obp2)})."
                )
                table_rows.append(
                    f"| 출루율(OBP) | {self._format_deterministic_metric(obp1)} | "
                    f"{self._format_deterministic_metric(obp2)} | {obp_leader} 우세 |"
                )

            walk_rate_leader = _leader_name(walk_rate1, walk_rate2)
            if walk_rate_leader:
                approach_lines.append(
                    f"볼넷 비중은 {walk_rate_leader} 쪽이 높습니다 "
                    f"({player1_name} {self._format_deterministic_metric(walk_rate1)}, "
                    f"{player2_name} {self._format_deterministic_metric(walk_rate2)})."
                )
                table_rows.append(
                    f"| 볼넷 비율 | {self._format_deterministic_metric(walk_rate1)} | "
                    f"{self._format_deterministic_metric(walk_rate2)} | {walk_rate_leader} 우세 |"
                )

            contact_leader = _leader_name(
                strikeout_rate1, strikeout_rate2, lower_is_better=True
            )
            if contact_leader:
                approach_lines.append(
                    f"삼진 억제는 {contact_leader} 쪽이 낫습니다 "
                    f"({player1_name} {self._format_deterministic_metric(strikeout_rate1)}, "
                    f"{player2_name} {self._format_deterministic_metric(strikeout_rate2)})."
                )

            slg_leader = _leader_name(slg1, slg2)
            if slg_leader:
                power_lines.append(
                    f"장타 생산은 장타율 기준으로 {slg_leader} 쪽이 한 단계 위입니다 "
                    f"({player1_name} {self._format_deterministic_metric(slg1)}, "
                    f"{player2_name} {self._format_deterministic_metric(slg2)})."
                )
                table_rows.append(
                    f"| 장타율(SLG) | {self._format_deterministic_metric(slg1)} | "
                    f"{self._format_deterministic_metric(slg2)} | {slg_leader} 우세 |"
                )

            home_run_leader = _leader_name(home_runs1, home_runs2)
            doubles_leader = _leader_name(doubles1, doubles2)
            if home_run_leader:
                power_lines.append(
                    f"홈런 생산은 {home_run_leader} 쪽이 앞섭니다 "
                    f"({player1_name} {self._format_deterministic_metric(home_runs1)}, "
                    f"{player2_name} {self._format_deterministic_metric(home_runs2)})."
                )
            if doubles_leader and doubles_leader != home_run_leader:
                power_lines.append(
                    f"반대로 2루타는 {doubles_leader} 쪽이 더 많아 갭파워 색깔은 조금 다르게 보입니다."
                )

            ops_leader = _leader_name(ops1, ops2)
            if not approach_lines:
                approach_lines.append(
                    f"타석 접근은 {player1_name} 또는 {player2_name} 한쪽의 출루율/볼넷/삼진 세부 수치가 비어 있어 지금 payload만으로 단정하기 어렵습니다."
                )
            if not power_lines and home_run_leader:
                power_lines.append(
                    f"장타 생산은 홈런 기준으로 {home_run_leader} 쪽이 더 강하게 보입니다 "
                    f"({player1_name} {self._format_deterministic_metric(home_runs1)}, "
                    f"{player2_name} {self._format_deterministic_metric(home_runs2)})."
                )

            productivity_summary = (
                f"전체 생산성은 {ops_leader} 쪽이 조금 더 앞서고"
                if ops_leader
                else "전체 생산성은 한쪽 세부 출루/장타 수치가 비어 있어 단정하기 어렵고"
            )
            style_summary = (
                "타석 접근과 장타 생산 방식은 분명한 결 차이가 있습니다."
                if any("단정하기 어렵습니다" not in line for line in approach_lines)
                and power_lines
                else "확인 가능한 범위 안에서만 차이를 읽는 편이 안전합니다."
            )
            summary_line = (
                f"{comparison_label} 기준으로 {player_pair}의 기록을 비교하면 "
                f"{productivity_summary}, {style_summary}"
            )
            if analysis.get("summary"):
                insight_lines.append(
                    f"- 내부 비교 요약은 `{analysis['summary']}` 기준으로 잡힙니다."
                )
            if any(token in query_lower for token in ["타석", "접근", "선구안"]):
                insight_lines.append(
                    "- 타석 접근 차이는 OBP, 볼넷 비율, 삼진 비율을 같이 봐야 과장 없이 읽을 수 있습니다."
                )
            if any(token in query_lower for token in ["장타", "홈런", "파워"]):
                insight_lines.append(
                    "- 장타력은 홈런 수 하나보다 SLG와 2루타 분포까지 함께 볼 때 해석이 안정적입니다."
                )

            if chat_mode:
                lines = [summary_line]
                if approach_lines:
                    lines.append(approach_lines[0])
                if power_lines:
                    lines.append(power_lines[0])
                if len(power_lines) > 1:
                    lines.append(power_lines[1])
                return "\n\n".join(lines[:4])

            detail_lines = (approach_lines[:2] + power_lines[:2]) or [
                "두 선수 모두 시즌 타격 데이터는 확인되지만, 세부 차이를 요약할 포인트가 아직 충분히 잡히지 않았습니다."
            ]
            if not table_rows:
                table_rows.append(
                    f"| OPS | {self._format_deterministic_metric(ops1)} | "
                    f"{self._format_deterministic_metric(ops2)} | 전체 생산성 비교 |"
                )
            insight_lines = insight_lines[:2] or [
                "- 비교 질문은 시즌을 고정하면 스타일 차이와 결과 차이를 분리해서 읽기 쉽습니다.",
                "- 같은 선수라도 통산과 단일 시즌의 결론이 달라질 수 있으니 목적에 맞게 구간을 고르는 편이 안전합니다.",
            ]
            return (
                "## 요약\n"
                f"{summary_line}\n\n"
                "## 상세 내역\n"
                f"{chr(10).join('- ' + line for line in detail_lines[:4])}\n\n"
                "## 핵심 지표\n"
                "| 항목 | "
                f"{player1_name} | {player2_name} | 해석 |\n"
                "| --- | --- | --- | --- |\n"
                f"{chr(10).join(table_rows[:4])}\n\n"
                "## 인사이트\n"
                f"{chr(10).join(insight_lines)}\n"
                "출처: DB 조회 결과"
            )

        if pitching1 and pitching2:
            era1 = _safe_float(_pick(pitching1, ["era", "career_era"]))
            era2 = _safe_float(_pick(pitching2, ["era", "career_era"]))
            whip1 = _safe_float(_pick(pitching1, ["whip", "career_whip"]))
            whip2 = _safe_float(_pick(pitching2, ["whip", "career_whip"]))
            strikeouts1 = _safe_float(
                _pick(pitching1, ["strikeouts", "total_strikeouts"])
            )
            strikeouts2 = _safe_float(
                _pick(pitching2, ["strikeouts", "total_strikeouts"])
            )
            innings1 = _safe_float(
                _pick(pitching1, ["innings_pitched", "total_innings_pitched"])
            )
            innings2 = _safe_float(
                _pick(pitching2, ["innings_pitched", "total_innings_pitched"])
            )

            era_leader = _leader_name(era1, era2, lower_is_better=True)
            whip_leader = _leader_name(whip1, whip2, lower_is_better=True)
            strikeout_leader = _leader_name(strikeouts1, strikeouts2)
            summary_line = (
                f"{comparison_label} 기준으로 {player_pair}의 기록을 비교하면 "
                f"실점 억제는 {era_leader or '두 선수'} 쪽, 이닝 소화와 탈삼진은 "
                f"{strikeout_leader or '두 선수'} 쪽이 조금 더 강하게 보입니다."
            )

            if chat_mode:
                lines = [summary_line]
                if era_leader:
                    lines.append(
                        f"ERA는 {era_leader} 쪽이 더 낮습니다 "
                        f"({player1_name} {self._format_deterministic_metric(era1)}, "
                        f"{player2_name} {self._format_deterministic_metric(era2)})."
                    )
                if whip_leader:
                    lines.append(
                        f"주자 관리도 WHIP 기준으로 {whip_leader} 쪽이 더 안정적입니다."
                    )
                return "\n\n".join(lines[:3])

            return (
                "## 요약\n"
                f"{summary_line}\n\n"
                "## 상세 내역\n"
                f"- ERA 기준 우세는 {era_leader or '동률'}입니다.\n"
                f"- WHIP 기준 우세는 {whip_leader or '동률'}입니다.\n"
                f"- 이닝/탈삼진은 {strikeout_leader or '동률'} 쪽이 조금 더 강합니다.\n\n"
                "## 핵심 지표\n"
                f"| 항목 | {player1_name} | {player2_name} | 해석 |\n"
                "| --- | --- | --- | --- |\n"
                f"| ERA | {self._format_deterministic_metric(era1)} | {self._format_deterministic_metric(era2)} | 실점 억제 |\n"
                f"| WHIP | {self._format_deterministic_metric(whip1)} | {self._format_deterministic_metric(whip2)} | 주자 관리 |\n"
                f"| 탈삼진 | {self._format_deterministic_metric(strikeouts1)} | {self._format_deterministic_metric(strikeouts2)} | 헛스윙 유도 |\n"
                f"| 이닝 | {self._format_deterministic_metric(innings1)} | {self._format_deterministic_metric(innings2)} | 소화 이닝 |\n\n"
                "## 인사이트\n"
                "- 투수 비교는 ERA와 WHIP의 방향이 같은지 먼저 보는 편이 안전합니다.\n"
                "- 탈삼진과 이닝이 함께 높으면 경기 지배력이 강한 타입으로 읽을 수 있습니다.\n"
                "출처: DB 조회 결과"
            )

        summary_line = (
            f"{comparison_label} 기준으로 {player_pair} 비교 데이터는 확보됐지만, "
            "타격/투구 세부 지표가 충분히 잡히지 않아 스타일 차이는 더 좁힌 질문에서 확실해집니다."
        )
        if chat_mode:
            return summary_line
        return (
            "## 요약\n"
            f"{summary_line}\n\n"
            "## 상세 내역\n"
            "- compare_players 결과는 확보됐습니다.\n"
            "- 다만 현재 payload만으로는 타격/투구 세부 차이를 더 깊게 분해하기 어렵습니다.\n\n"
            "## 핵심 지표\n"
            "| 항목 | 상태 |\n"
            "| --- | --- |\n"
            "| 비교 기준 | 확보 |\n"
            "| 세부 지표 | 일부 부족 |\n\n"
            "## 인사이트\n"
            "- 시즌, 포지션, 원하는 비교 축을 같이 주면 deterministic 비교 답변이 더 선명해집니다.\n"
            "- 예: `김도영과 문보경 2025 타석 접근 비교`처럼 좁히면 됩니다.\n"
            "출처: DB 조회 결과"
        )

    def _build_bundle_reference_answer(
        self,
        query: str,
        tool_results: List[ToolResult],
        *,
        chat_mode: bool,
    ) -> Optional[str]:
        team_bundle_answer = self._build_team_bundle_answer(
            query,
            tool_results,
            chat_mode=chat_mode,
        )
        if team_bundle_answer:
            return team_bundle_answer

        schedule_bundle_answer = self._build_schedule_bundle_answer(query, tool_results)
        if schedule_bundle_answer:
            return schedule_bundle_answer

        return self._build_player_bundle_answer(tool_results)

    def _build_fast_path_answer(
        self, query: str, tool_results: List[ToolResult], chat_mode: bool = False
    ) -> Optional[str]:
        """Fast-path 팀 질문은 LLM 없이 DB 결과만으로 즉시 답변을 생성합니다."""
        summary_data: Dict[str, Any] = {}
        advanced_data: Dict[str, Any] = {}
        recent_data: Dict[str, Any] = {}

        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue
            data = result.data
            if "top_batters" in data or "top_pitchers" in data:
                summary_data = data
            if "metrics" in data or "fatigue_index" in data:
                advanced_data = data
            if "games" in data and data.get("team_name") and "date" not in data:
                recent_data = data

        team_name = summary_data.get("team_name") or advanced_data.get("team_name")
        year = summary_data.get("year") or advanced_data.get("year")
        if not team_name or not year:
            return None

        top_batters = summary_data.get("top_batters") or []
        top_pitchers = summary_data.get("top_pitchers") or []
        metrics = advanced_data.get("metrics") or {}
        batting = metrics.get("batting") or {}
        pitching = metrics.get("pitching") or {}
        rankings = advanced_data.get("rankings") or {}
        fatigue_index = advanced_data.get("fatigue_index") or {}
        league_averages = advanced_data.get("league_averages") or {}
        recent_form = self._summarize_recent_team_games(recent_data, team_name)

        query_lower = query.lower()
        direct_metric_answer = self._build_team_metric_fast_path_answer(
            query,
            team_name,
            year,
            batting,
            pitching,
            rankings,
        )
        if direct_metric_answer:
            return direct_metric_answer

        unavailable_topics = []
        if "상대 전적" in query_lower or "상대전적" in query_lower:
            unavailable_topics.append("상대 전적")
        if "실책" in query_lower or "수비" in query_lower:
            unavailable_topics.append("실책/수비")
        if "큰 경기" in query_lower:
            unavailable_topics.append("클러치/큰 경기")

        ops = batting.get("ops")
        avg = batting.get("avg")
        total_hr = batting.get("total_hr")
        avg_era = pitching.get("avg_era")
        qs_rate = pitching.get("qs_rate")
        era_rank = pitching.get("era_rank")
        ops_rank = rankings.get("batting_ops")
        bullpen_share = fatigue_index.get("bullpen_share")
        bullpen_load_rank = fatigue_index.get("bullpen_load_rank")
        league_bullpen_share = league_averages.get("bullpen_share")

        summary_line = (
            f"{year}년 {team_name}는 "
            f"팀 OPS {ops if ops is not None else '확인 불가'}, "
            f"평균자책 {avg_era if avg_era is not None else '확인 불가'} 기준으로 보면 "
            "지금 전력 흐름은 읽힙니다."
        )

        if (
            "가을야구" in query_lower
            or "플레이오프" in query_lower
            or "플옵" in query_lower
        ):
            summary_line = (
                f"{year}년 {team_name}는 타격 순위 {ops_rank or '확인 불가'}, "
                f"평균자책 관련 지표 {era_rank or '확인 불가'} 수준이라 "
                "가을야구 경쟁력은 충분하지만 마운드 안정성이 핵심 변수입니다."
            )
        elif "불펜" in query_lower or "필승조" in query_lower:
            summary_line = (
                f"{year}년 {team_name} 불펜 비중은 {bullpen_share or '확인 불가'}로 "
                f"리그 평균 {league_bullpen_share or '확인 불가'} 대비 비교가 가능하며, "
                "현재는 불펜 의존도와 선발 이닝 소화력이 같이 봐야 하는 상태입니다."
            )
        elif "선발" in query_lower:
            summary_line = (
                f"{year}년 {team_name} 선발진은 QS 비율 {qs_rate or '확인 불가'}, "
                f"평균자책 {avg_era if avg_era is not None else '확인 불가'} 기준으로 보면 "
                "완전히 불안정하다고 보긴 어렵지만 꾸준함은 더 확인이 필요합니다."
            )
        elif recent_form and any(
            token in query_lower
            for token in ["최근", "흐름", "폼", "페이스", "5경기", "10경기"]
        ):
            summary_line = (
                f"{year}년 {team_name}는 최근 {recent_form['sample_size']}경기 "
                f"{recent_form['record_text']} 흐름이고, "
                f"불펜 비중 {bullpen_share or '확인 불가'}까지 같이 보면 "
                "상승세인지 버티는 흐름인지 더 분명하게 읽힙니다."
            )
        elif (
            "타선" in query_lower
            or "득점" in query_lower
            or "침묵" in query_lower
            or "터질" in query_lower
        ):
            summary_line = (
                f"{year}년 {team_name} 타선은 팀 OPS {ops if ops is not None else '확인 불가'}, "
                f"팀 타율 {avg if avg is not None else '확인 불가'}, 홈런 {total_hr if total_hr is not None else '확인 불가'} 기준으로 "
                "전체 화력 방향은 확인되지만, 경기별 득점 변동성은 추가 경기 로그가 있어야 더 정확합니다."
            )

        starter_names = [
            pitcher.get("player_name")
            for pitcher in top_pitchers
            if pitcher.get("role") == "starter" and pitcher.get("player_name")
        ][:2]

        if chat_mode:
            chat_lines = [summary_line]

            if top_batters:
                batter = top_batters[0]
                batter_name = batter.get("player_name", "주축 타자")
                chat_lines.append(
                    f"타선 쪽에서는 {batter_name}{self._select_subject_particle(batter_name)} "
                    f"OPS {batter.get('ops', '확인 불가')} / 홈런 {batter.get('home_runs', '확인 불가')}로 중심을 잡고 있습니다."
                )
            if len(top_batters) > 1:
                batter2 = top_batters[1]
                batter2_name = batter2.get("player_name", "주축 타자")
                chat_lines.append(
                    f"옆에서는 {batter2_name}{self._select_subject_particle(batter2_name)} "
                    f"타율 {batter2.get('avg', '확인 불가')} / OPS {batter2.get('ops', '확인 불가')}로 받쳐주고 있습니다."
                )
            if starter_names:
                starter_label = ", ".join(starter_names)
                chat_lines.append(
                    f"선발 쪽은 {starter_label} 중심으로 보고 있고, QS 비율은 {qs_rate or '확인 불가'}입니다."
                )
            elif top_pitchers:
                pitcher = top_pitchers[0]
                pitcher_name = pitcher.get("player_name", "핵심 투수")
                chat_lines.append(
                    f"마운드는 {pitcher_name}{self._select_subject_particle(pitcher_name)} "
                    f"ERA {pitcher.get('era', '확인 불가')} / WHIP {pitcher.get('whip', '확인 불가')}로 중심을 잡고 있습니다."
                )
            if bullpen_share or bullpen_load_rank:
                chat_lines.append(
                    f"불펜 비중은 {bullpen_share or '확인 불가'}이고, 과부하 순위는 {bullpen_load_rank or '확인 불가'}라서 "
                    "불펜 부담도 같이 봐야 합니다."
                )
            if recent_form:
                recent_line = (
                    f"최근 {recent_form['sample_size']}경기 흐름은 {recent_form['record_text']}이고, "
                    f"가장 가까운 경기들은 {recent_form['recent_preview']} 쪽입니다."
                )
                if recent_form.get("streak_text"):
                    recent_line += (
                        f" 지금은 {recent_form['streak_text']} 흐름으로도 읽힙니다."
                    )
                chat_lines.append(recent_line)
            if unavailable_topics:
                chat_lines.append(
                    f"다만 {', '.join(unavailable_topics)} 쪽은 지금 붙은 fast-path 데이터만으로는 단정하기 어렵습니다."
                )

            if ops_rank and era_rank:
                chat_lines.append(
                    f"숫자만 놓고 보면 타격은 {ops_rank}, 마운드는 {era_rank}라서 한쪽이 완전히 무너진 팀으로 보긴 어렵습니다."
                )
            elif ops is not None or avg_era is not None:
                chat_lines.append(
                    "지금 단계에서는 완전한 붕괴보다는 기복 관리가 더 중요한 팀으로 보입니다."
                )

            if bullpen_share and league_bullpen_share:
                chat_lines.append(
                    f"불펜 비중 {bullpen_share}는 리그 평균 {league_bullpen_share}와 비교하면서 계속 보는 게 좋겠습니다."
                )

            return "\n\n".join(chat_lines[:6])

        detail_lines = []
        if top_batters:
            batter = top_batters[0]
            detail_lines.append(
                f"- 타선 핵심 타자는 **{batter.get('player_name', '주축 타자')}**이고, OPS {batter.get('ops', '확인 불가')} / 홈런 {batter.get('home_runs', '확인 불가')} 수준입니다."
            )
        if len(top_batters) > 1:
            batter2 = top_batters[1]
            detail_lines.append(
                f"- 보조 화력은 **{batter2.get('player_name', '주축 타자')}**이 받치고 있고, 타율 {batter2.get('avg', '확인 불가')} / OPS {batter2.get('ops', '확인 불가')} 수준입니다."
            )
        if starter_names:
            detail_lines.append(
                f"- 선발 축은 **{', '.join(starter_names)}** 중심으로 보이며, QS 비율은 {qs_rate or '확인 불가'}입니다."
            )
        elif top_pitchers:
            pitcher = top_pitchers[0]
            detail_lines.append(
                f"- 마운드 중심은 **{pitcher.get('player_name', '핵심 투수')}**이고, ERA {pitcher.get('era', '확인 불가')} / WHIP {pitcher.get('whip', '확인 불가')}입니다."
            )
        if bullpen_share or bullpen_load_rank:
            detail_lines.append(
                f"- 불펜 비중은 {bullpen_share or '확인 불가'}, 과부하 순위는 {bullpen_load_rank or '확인 불가'}로 집계됩니다."
            )
        if recent_form:
            recent_detail = (
                f"- 최근 {recent_form['sample_size']}경기는 {recent_form['record_text']} 흐름이고, "
                f"직전 경기 묶음은 {recent_form['recent_preview']} 기준으로 확인됩니다."
            )
            if recent_form.get("streak_text"):
                recent_detail += f" 현재는 {recent_form['streak_text']} 흐름입니다."
            detail_lines.append(recent_detail)
        if unavailable_topics:
            detail_lines.append(
                f"- 질문의 핵심인 **{', '.join(unavailable_topics)}** 직접 데이터는 현재 fast-path 도구셋에 없어 여기서는 추정하지 않았습니다."
            )
        detail_lines = detail_lines[:4]
        if not detail_lines:
            detail_lines.append(
                "- 현재 확보된 DB 결과 기준으로 팀 전력의 큰 흐름만 확인 가능하며, 세부 원인 분해는 제한적입니다."
            )

        table_rows = [
            ("팀 타율", avg if avg is not None else "확인 불가", "타선 정확도"),
            (
                "팀 OPS",
                ops if ops is not None else "확인 불가",
                f"리그 순위 {ops_rank or '확인 불가'}",
            ),
            (
                "선발 QS 비율",
                qs_rate or "확인 불가",
                f"평균자책 {avg_era if avg_era is not None else '확인 불가'}",
            ),
            (
                "불펜 비중",
                bullpen_share or "확인 불가",
                f"리그 평균 {league_bullpen_share or '확인 불가'}",
            ),
            (
                "최근 흐름",
                recent_form["record_text"] if recent_form else "확인 불가",
                (
                    recent_form.get("streak_text", "최근 경기 추세")
                    if recent_form
                    else "최근 경기 추세"
                ),
            ),
        ]

        insight_lines = []
        if ops_rank and era_rank:
            insight_lines.append(
                f"- 타격은 {ops_rank}, 마운드는 {era_rank} 기준이라 한쪽으로 완전히 무너진 전력은 아닙니다."
            )
        elif ops is not None or avg_era is not None:
            insight_lines.append(
                "- 현재 확인 가능한 범위에서는 타선과 마운드 둘 다 완전한 붕괴보다는 기복 관리가 더 중요해 보입니다."
            )
        if bullpen_share and league_bullpen_share:
            insight_lines.append(
                f"- 불펜 비중 {bullpen_share}는 리그 평균 {league_bullpen_share}와 비교해 운용 부담을 판단할 수 있는 지점입니다."
            )
        if unavailable_topics:
            insight_lines.append(
                "- 실책, 상대 전적, 큰 경기 대응력처럼 세부 맥락이 필요한 항목은 전용 도구를 붙이면 더 정확해집니다."
            )
        if not insight_lines:
            insight_lines.append(
                "- 현재 도구 결과만으로는 기본 전력 흐름까지는 확인 가능하지만 세부 원인 진단은 제한적입니다."
            )
        insight_lines = insight_lines[:2]

        table_text = "\n".join(
            f"| {label} | {value} | {meaning} |" for label, value, meaning in table_rows
        )

        return (
            "## 요약\n"
            f"{summary_line}\n\n"
            "## 상세 내역\n" + "\n".join(detail_lines) + "\n\n## 핵심 지표\n"
            "| 항목 | 수치 | 해석 |\n"
            "| --- | --- | --- |\n"
            f"{table_text}\n\n"
            "## 인사이트\n" + "\n".join(insight_lines) + "\n\n출처: DB 조회 결과"
        )

    def _format_chatbot_table_row(self, cells: List[str]) -> Optional[str]:
        normalized = [str(cell).strip() for cell in cells if str(cell).strip()]
        if not normalized:
            return None
        label = normalized[0]
        particle = self._select_topic_particle(label)
        if len(normalized) >= 3:
            return (
                f"{label}{particle} {normalized[1]}이고, "
                f"{normalized[2]} 정도로 보면 됩니다."
            )
        if len(normalized) == 2:
            return f"{label}{particle} {normalized[1]}입니다."
        return normalized[0]

    def _extract_particle_target(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("**") and cleaned.endswith("**") and len(cleaned) > 4:
            cleaned = cleaned[2:-2].strip()
        cleaned = re.sub(r"[`*_~]+", "", cleaned)
        return cleaned.strip()

    def _has_batchim(self, text: str) -> bool:
        cleaned = self._extract_particle_target(text)
        if not cleaned:
            return False
        last_char = cleaned[-1]
        if "가" <= last_char <= "힣":
            return (ord(last_char) - ord("가")) % 28 != 0
        return False

    def _select_topic_particle(self, text: str) -> str:
        return "은" if self._has_batchim(text) else "는"

    def _select_subject_particle(self, text: str) -> str:
        return "이" if self._has_batchim(text) else "가"

    def _select_direction_particle(self, text: str) -> str:
        cleaned = self._extract_particle_target(text)
        if not cleaned:
            return "로"
        last_char = cleaned[-1]
        if "가" <= last_char <= "힣":
            final_consonant = (ord(last_char) - ord("가")) % 28
            if final_consonant == 0 or final_consonant == 8:
                return "로"
            return "으로"
        return "로"

    def _postprocess_chatbot_answer_text(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            return normalized

        normalized = re.sub(
            r"(^|\n)\s*##\s+[^\n]+",
            r"\1",
            normalized,
            flags=re.MULTILINE,
        )
        normalized = re.sub(
            r"(^|\n)\s*\*\*분석 결과:\*\*\s*",
            r"\1",
            normalized,
            flags=re.MULTILINE,
        )
        normalized = re.sub(
            r"(^|\n)\s*분석 결과:\s*",
            r"\1",
            normalized,
            flags=re.MULTILINE,
        )
        normalized = re.sub(
            r"^\s*안녕하세요[!,. ]*(?:저는 [^.!\n]+입니다[!.]?)?\s*",
            "",
            normalized,
            count=1,
        )
        normalized = re.sub(
            r"^\s*질문하신\s+`?[^`\n]+`?\s*(?:기준으로 보면|관련해서는)?\s*",
            "",
            normalized,
            count=1,
        )
        normalized = re.sub(
            r"^\s*(?:\*\*)?분석 결과(?:\*\*)?\s*[:：]\s*",
            "",
            normalized,
            count=1,
        )
        normalized = re.sub(
            r"^\s*규정은 이렇게 이해하면 됩니다\.?\s*",
            "",
            normalized,
            count=1,
        )
        normalized = re.sub(
            r"(^|\n)\s*(?:KBO\s+)?[^.\n]{0,80}?(?:개요|해설|스토리라인)\s*$",
            r"\1",
            normalized,
            flags=re.MULTILINE,
        )
        normalized = re.sub(
            r"(^|\n)\s*(?:[-*]\s*)?(?:\*\*)?[A-Za-z0-9_./-]+\.md(?:\*\*)?\s*[:：]\s*",
            r"\1",
            normalized,
            flags=re.MULTILINE,
        )
        normalized = re.sub(
            r"(^|\n)\s*(?:[-*]\s*)?(?:\*\*)?[A-Za-z0-9_./-]+\.md(?:\*\*)?\s*$",
            r"\1",
            normalized,
            flags=re.MULTILINE,
        )
        normalized = re.sub(
            r"(^|\n)\s*(?:[-*]\s*)?(?:출처|source)\s*[:：].*$",
            r"\1",
            normalized,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        normalized = re.sub(
            r"(^|\n)\s*근거 요약\s*$",
            r"\1",
            normalized,
            flags=re.MULTILINE,
        )

        normalized = re.sub(
            r"^(.*?는) 타선은 ([^,]+), 마운드는 ([^.]+) 기준으로 현재 전력의 방향성은 확인 가능합니다\.$",
            r"\1 \2의 타선과 \3의 마운드를 보면 지금 전력 흐름은 읽힙니다.",
            normalized,
            count=1,
            flags=re.MULTILINE,
        )

        def replace_bold_particle(match: re.Match[str]) -> str:
            token = match.group(1)
            particle = match.group(2)
            if particle in {"은", "는"}:
                fixed = self._select_topic_particle(token)
            elif particle in {"이", "가"}:
                fixed = self._select_subject_particle(token)
            else:
                fixed = self._select_direction_particle(token)
            return f"{token}{fixed}"

        normalized = re.sub(
            r"(\*\*[^*]+\*\*)(은|는|이|가|로)(?=[\s,.)!?]|$)",
            replace_bold_particle,
            normalized,
        )
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
        return normalized

    def _normalize_chatbot_answer_text(self, text: str) -> str:
        if not text:
            return ""

        paragraphs: List[str] = []
        table_rows: List[List[str]] = []
        table_headers: List[str] = []

        def flush_table() -> None:
            nonlocal table_rows, table_headers
            if not table_rows:
                return
            table_sentences: List[str] = []
            for row in table_rows[:4]:
                sentence = self._format_chatbot_table_row(row)
                if sentence:
                    table_sentences.append(sentence)
            if table_sentences:
                paragraphs.append(" ".join(table_sentences))
            table_rows = []
            table_headers = []

        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                flush_table()
                continue

            lowered = stripped.lower()
            if stripped.startswith("## ") or stripped.startswith("### "):
                flush_table()
                continue
            if stripped.startswith("질문하신 "):
                flush_table()
                continue
            if stripped.startswith("분석 결과:") or stripped.startswith(
                "**분석 결과:**"
            ):
                flush_table()
                continue
            if lowered.startswith("출처:") or lowered.startswith("- 출처:"):
                flush_table()
                continue
            if lowered.startswith("source:") or lowered.startswith("- source:"):
                flush_table()
                continue
            if re.fullmatch(
                r"(?:[-*]\s*)?(?:\*\*)?[A-Za-z0-9_./-]+\.md(?:\*\*)?",
                stripped,
            ):
                flush_table()
                continue

            if stripped.startswith("|"):
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                is_divider = all(
                    cell and set(cell) <= {"-", ":", " "} for cell in cells
                )
                if is_divider:
                    continue
                if not table_headers:
                    table_headers = cells
                else:
                    table_rows.append(cells)
                continue

            flush_table()

            if stripped.startswith("- ") or stripped.startswith("* "):
                stripped = stripped[2:].strip()
            if stripped:
                paragraphs.append(stripped)

        flush_table()
        normalized = "\n\n".join(part for part in paragraphs if part).strip()
        return self._postprocess_chatbot_answer_text(normalized or text.strip())

    def _resolve_answer_max_tokens(
        self, query: str, tool_results: List[ToolResult]
    ) -> Optional[int]:
        """최종 답변 max_tokens를 질문 유형별로 동적 계산합니다."""
        if not self.chat_dynamic_token_enabled:
            return None

        entity_filter = extract_entities_from_query(query)
        if self._is_team_analysis_query(query, entity_filter):
            return self.chat_answer_max_tokens_team

        has_heavy_rows = False
        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue
            for key in ("results", "players", "games", "rows", "leaders"):
                value = result.data.get(key)
                if isinstance(value, list) and len(value) > 4:
                    has_heavy_rows = True
                    break
            if has_heavy_rows:
                break

        long_keywords = ("상세", "자세히", "비교", "분석", "리포트", "전체", "추세")
        if has_heavy_rows or len(query) > 45 or any(k in query for k in long_keywords):
            return self.chat_answer_max_tokens_long
        return self.chat_answer_max_tokens_short

    async def _analyze_query_and_plan_tools(
        self, query: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        질문 분석 진입점.
        - 팀 분석 패턴이면 Fast-Path(규칙 기반) 적용
        - 그 외에는 기존 LLM 분석 경로 사용
        """
        if context is None:
            context = {}
        elif not isinstance(context, dict):
            context = {}

        entity_filter = extract_entities_from_query(query)
        fast_path_plan = self._build_fast_path_plan(query, entity_filter, context)
        if fast_path_plan:
            logger.info(
                "[PlannerDecision] mode=%s team_query=%s tool_count=%d",
                fast_path_plan.get("planner_mode", "fast_path"),
                self._is_team_analysis_query(query, entity_filter),
                len(fast_path_plan.get("tool_calls", [])),
            )
            return fast_path_plan

        cached_plan = self._get_cached_planner_plan(query, context)
        if cached_plan is not None:
            logger.info(
                "[PlannerDecision] mode=%s cache_hit=true tool_count=%d",
                cached_plan.get("planner_mode", "default_llm_planner"),
                len(cached_plan.get("tool_calls", [])),
            )
            return cached_plan

        llm_analysis_started = datetime.now()
        llm_plan = await self._analyze_query_with_llm(query, context)
        analysis_ms = round(
            (datetime.now() - llm_analysis_started).total_seconds() * 1000, 2
        )
        if "planner_mode" not in llm_plan:
            llm_plan["planner_mode"] = "default_llm_planner"
        llm_plan["analysis_ms"] = analysis_ms
        original_count = len(llm_plan.get("tool_calls", []))
        llm_plan["tool_calls"] = self._soft_filter_llm_tool_calls(
            llm_plan.get("tool_calls", []),
            query=query,
            entity_filter=entity_filter,
            planner_mode=llm_plan["planner_mode"],
        )
        logger.info(
            "[PlannerDecision] mode=%s team_query=%s tool_count=%d->%d",
            llm_plan["planner_mode"],
            self._is_team_analysis_query(query, entity_filter),
            original_count,
            len(llm_plan.get("tool_calls", [])),
        )
        self._store_planner_plan_cache(query, context, llm_plan)
        return llm_plan

    async def _analyze_query_with_llm(
        self, query: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        사용자 질문을 분석하고 필요한 도구 호출을 계획합니다.
        '작년', '올해' 등 상대적인 시간 표현을 미리 처리합니다.
        """
        logger.info(f"[BaseballAgent] Analyzing query for tool planning: {query}")

        # 시간 표현 전처리
        now = datetime.now()
        current_year = now.year
        current_date = now.strftime("%Y년 %m월 %d일")
        # 시간 표현 전처리 (프롬프트에서 처리하므로 파이썬 측 치환 로직 제거)
        processed_query = query

        # LLM을 사용하여 질문을 분석하고 도구 사용 계획 수립

        # 0. 엔티티 추출 (LLM 호출 전)
        from ..core.entity_extractor import extract_entities_from_query

        entity_filter = extract_entities_from_query(processed_query)

        # 추출된 엔티티를 프롬프트에 제공할 컨텍스트로 포맷팅
        entity_context_parts = []
        if entity_filter.season_year:
            entity_context_parts.append(f"- 연도: {entity_filter.season_year}년")
        query_player_names = self._extract_llm_planner_player_names(
            processed_query, entity_filter
        )
        if len(query_player_names) >= 2:
            entity_context_parts.append(
                f"- 선수명 목록: {', '.join(query_player_names)}"
            )
        elif entity_filter.player_name:
            entity_context_parts.append(f"- 선수명: {entity_filter.player_name}")
        if entity_filter.team_id:
            entity_context_parts.append(f"- 팀명: {entity_filter.team_id}")
        if entity_filter.stat_type:
            entity_context_parts.append(f"- 통계 지표: {entity_filter.stat_type}")
        if entity_filter.league_type:
            entity_context_parts.append(f"- 리그 타입: {entity_filter.league_type}")

        entity_context = ""
        if entity_context_parts:
            entity_context = "\n힌트: " + ", ".join(entity_context_parts)

        query_text = processed_query  # 전처리된 쿼리 사용
        analysis_prompt, planner_mode = self._select_llm_planner_prompt(
            query_text=query_text,
            query=query,
            entity_filter=entity_filter,
            current_date=current_date,
            current_year=current_year,
            entity_context=entity_context,
        )
        logger.info(
            "[Planner] selected llm planner mode=%s query=%s",
            planner_mode,
            query_text,
        )

        try:
            # LLM 호출하여 분석 결과 받기
            analysis_messages = [{"role": "user", "content": analysis_prompt}]
            analysis_max_tokens = self._resolve_llm_planner_max_tokens(planner_mode)
            analysis_timeout_seconds = self._resolve_llm_planner_timeout_seconds(
                planner_mode
            )
            logger.info(
                "[Planner] mode=%s prompt_chars=%d max_tokens=%s timeout=%.1fs",
                planner_mode,
                len(analysis_prompt),
                analysis_max_tokens,
                analysis_timeout_seconds,
            )

            # 스트리밍 API인 경우 전체 응답을 모아서 처리
            raw_response = ""
            try:
                async with asyncio.timeout(analysis_timeout_seconds):
                    async for chunk in self.llm_generator(
                        analysis_messages, max_tokens=analysis_max_tokens
                    ):
                        if chunk:
                            raw_response += chunk
            except TimeoutError:
                logger.warning(
                    "[Planner] timeout mode=%s timeout=%.1fs query=%s",
                    planner_mode,
                    analysis_timeout_seconds,
                    query_text[:120],
                )
                return {
                    "analysis": "",
                    "tool_calls": [],
                    "expected_result": "",
                    "error": "질문 분석 오류: planner timeout",
                    "planner_mode": planner_mode,
                }

            logger.info(f"[BaseballAgent] Raw LLM response: {raw_response[:200]}...")

            # JSON 블록 추출 (```json ... ``` 형태인 경우)
            if "```json" in raw_response:
                start = raw_response.find("```json") + 7
                end = raw_response.find("```", start)
                json_content = (
                    raw_response[start:end].strip()
                    if end != -1
                    else raw_response[start:].strip()
                )
            elif raw_response.strip().startswith("{"):
                json_content = raw_response.strip()
            else:
                # JSON이 아닌 응답인 경우 기본 분석 제공
                logger.warning(
                    f"[BaseballAgent] Non-JSON response, providing fallback analysis"
                )
                return {
                    "analysis": f"'{query}' 질문을 리더보드로 분석",
                    "tool_calls": [
                        ToolCall(
                            tool_name="get_leaderboard",
                            parameters={
                                "stat_name": "ops",
                                "year": current_year,
                                "position": "batting",
                                "limit": 10,
                            },
                        )
                    ],
                    "expected_result": "상위 타자 순위",
                    "error": None,
                    "planner_mode": planner_mode,
                }

            # JSON 파싱
            try:
                cleaned_json = clean_json_response(json_content)
                analysis_data = json.loads(cleaned_json)

                # --- Year Correction Logic ---
                try:
                    from ..core.entity_extractor import extract_entities_from_query
                    import datetime as dt

                    entity_filter = extract_entities_from_query(query)
                    now = dt.datetime.now()
                    current_year = now.year

                    # 명시적으로 추출된 연도가 있는 경우 (예: "작년" -> 2025)
                    if entity_filter.season_year:
                        extracted_year = entity_filter.season_year
                        for call_data in analysis_data.get("tool_calls", []):
                            params = call_data.get("parameters", {})
                            if "year" in params:
                                llm_year = params["year"]
                                if isinstance(llm_year, str) and llm_year.isdigit():
                                    llm_year = int(llm_year)
                                    params["year"] = llm_year
                                # 1. LLM이 현재/미래 연도를 제시했으나 추출된 연도는 과거인 경우 보정 (예: "작년" -> 2025)
                                if (
                                    isinstance(llm_year, int)
                                    and llm_year >= current_year
                                    and extracted_year < current_year
                                ):
                                    logger.info(
                                        f"[BaseballAgent] Correcting year {llm_year} -> {extracted_year} based on entity extraction"
                                    )
                                    params["year"] = extracted_year

                                # 2. Pre-season 로직: 1-3월인데 올해(2026)를 조회하려는 경우 (데이터가 없으므로 2025로 강제 보정)
                                # 단, 명시적으로 미래를 묻는 쿼리가 아닐 때만
                                elif (
                                    isinstance(llm_year, int)
                                    and llm_year == current_year
                                    and now.month <= 3
                                ):
                                    # 통계 관련 도구인 경우에만 적용
                                    stat_related_tools = [
                                        "get_leaderboard",
                                        "get_player_stats",
                                        "get_team_summary",
                                        "get_team_rank",
                                    ]
                                    if call_data.get("tool_name") in stat_related_tools:
                                        # "내년", "2026년" 같은 명시적 표현이 없는지 확인
                                        if not any(
                                            word in query
                                            for word in [
                                                "내년",
                                                "다가오는",
                                                str(current_year),
                                            ]
                                        ):
                                            logger.info(
                                                f"[BaseballAgent] Pre-season override: {llm_year} -> {current_year - 1}"
                                            )
                                            params["year"] = current_year - 1
                except Exception as e:
                    logger.warning(f"[BaseballAgent] Year correction failed: {e}")
                # -----------------------------
            except json.JSONDecodeError as e:
                logger.error(f"[BaseballAgent] JSON parsing error: {e}")
                logger.error(f"[BaseballAgent] Original content: {json_content}")
                logger.error(f"[BaseballAgent] Cleaned content: {cleaned_json}")
                raise

            # ToolCall 객체들로 변환
            tool_calls = []
            for call_data in analysis_data.get("tool_calls", []):
                tool_call = ToolCall(
                    tool_name=call_data["tool_name"], parameters=call_data["parameters"]
                )
                tool_calls.append(tool_call)

            return {
                "analysis": analysis_data.get("analysis", ""),
                "tool_calls": tool_calls,
                "expected_result": analysis_data.get("expected_result", ""),
                "error": None,
                "planner_mode": planner_mode,
            }

        except json.JSONDecodeError as e:
            logger.error(f"[BaseballAgent] JSON parsing error in query analysis: {e}")
            logger.error(f"[BaseballAgent] Failed response content: {raw_response}")

            # 현재 연도 계산 및 엔티티 추출
            import datetime as dt
            from ..core.entity_extractor import extract_entities_from_query

            current_year = dt.datetime.now().year
            entity_filter = extract_entities_from_query(query)

            # 질문 유형에 따른 스마트 폴백
            query_lower = query.lower()

            # 선수명 추출 시도
            from ..core.entity_extractor import TEAM_MAPPING

            potential_player_name = entity_filter.player_name

            if not potential_player_name:
                import re

                words = re.findall(r"[가-힣]{2,4}", query)
                # 공통 용어 및 팀명 제외
                from ..core.entity_extractor import extract_player_name

                # extract_player_name의 내부 로직을 활용하거나 직접 필터링
                common_terms = {
                    "순위",
                    "성적",
                    "기록",
                    "랭킹",
                    "정규시즌",
                    "데이터",
                    "확인",
                    "투수",
                    "타자",
                    "선수",
                    "팀명",
                    "구단",
                    "홈런",
                    "안타",
                    "타점",
                    "알려줘",
                    "설명해줘",
                    "보여줘",
                    "부탁해",
                    "어딨어",
                    "누구야",
                    "작년",
                    "올해",
                    "재작년",
                    "내년",
                    "승률",
                    "몇승",
                    "몇패",
                    "상대",
                    "특정",
                    "결과",
                    "대결",
                    "승부",
                    "위가",
                    "경기",
                    "어떤",
                    "무슨",
                }

                for word in words:
                    # 팀 이름 매핑 키 필터링
                    if word not in common_terms and word not in TEAM_MAPPING:
                        potential_player_name = word
                        break

            # 질문에서 추출된 값들 사용
            extracted_year = entity_filter.season_year or current_year
            extracted_stat = entity_filter.stat_type or "ops"
            extracted_position = entity_filter.position_type or "both"
            resolved_award_type = self._resolve_award_query_type(query, entity_filter)
            fallback_tool = None
            analysis = "LLM 응답 분석 실패로 인한 규칙 기반 Fallback 도구 선택"

            if resolved_award_type:
                award_parameters = {"year": extracted_year}
                if resolved_award_type != "any":
                    award_parameters["award_type"] = resolved_award_type
                fallback_tool = ToolCall(
                    tool_name="get_award_winners",
                    parameters=award_parameters,
                )
                analysis = (
                    f"{extracted_year}년 "
                    f"{self._display_award_type(resolved_award_type)} 수상 조회"
                )

            # 규정/규칙 질문 감지
            regulation_keywords = [
                "규정",
                "규칙",
                "제도",
                "요건",
                "자격",
                "기준",
                "공식",
                "FA",
                "연봉",
                "드래프트",
                "엔트리",
                "피치클락",
                "ABS",
                "로봇심판",
                "베이스",
                "시프트",
            ]
            if fallback_tool is None and any(
                keyword in query for keyword in regulation_keywords
            ):
                fallback_tool = ToolCall(
                    tool_name="search_regulations", parameters={"query": query}
                )
                analysis = f"규정/규칙 관련 질문: '{query}'"

            # 팀 순위 질문 감지
            elif (
                fallback_tool is None
                and entity_filter.team_id
                and any(
                    word in query_lower
                    for word in ["순위", "성적", "기록", "랭킹", "정규시즌"]
                )
            ):
                fallback_tool = ToolCall(
                    tool_name="get_team_rank",
                    parameters={
                        "team_name": entity_filter.team_id,
                        "year": extracted_year,
                    },
                )
                analysis = (
                    f"'{entity_filter.team_id}' 팀의 {extracted_year}년 성적 조회"
                )

            # 선수 성적 질문 감지
            elif fallback_tool is None and potential_player_name:
                # 통산/커리어 질문 감지
                if any(
                    word in query_lower for word in ["통산", "커리어", "총", "kbo 리그"]
                ):
                    fallback_tool = ToolCall(
                        tool_name="get_career_stats",
                        parameters={
                            "player_name": potential_player_name,
                            "position": "both",
                        },
                    )
                    analysis = f"{potential_player_name} 선수의 통산 기록 조회"
                else:
                    fallback_tool = ToolCall(
                        tool_name="get_leaderboard",
                        parameters={
                            "stat_name": extracted_stat,
                            "year": extracted_year,
                            "position": extracted_position,
                            "limit": 10,
                        },
                    )
                    analysis = (
                        f"통산 기록 관련 질문으로 판단하여 상위 {extracted_stat} 조회"
                    )

            # 경기 일정/결과 질문 감지
            elif fallback_tool is None and any(
                word in query_lower
                for word in [
                    "경기",
                    "일정",
                    "결과",
                    "어린이날",
                    "한국시리즈",
                    "시범경기",
                    "언제부터",
                    "우승",
                ]
            ):
                # 날짜 추출 시도
                import re

                date_patterns = [
                    r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일",  # 2025년 5월 5일
                    r"(\d{4})-(\d{1,2})-(\d{1,2})",  # 2025-05-05
                ]
                extracted_date = None
                for pattern in date_patterns:
                    match = re.search(pattern, query)
                    if match:
                        year, month, day = match.groups()
                        extracted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        break
                if extracted_date:
                    fallback_tool = ToolCall(
                        tool_name="get_games_by_date",
                        parameters={"date": extracted_date},
                    )
                    analysis = f"{extracted_date} 날짜의 경기 일정/결과 조회"

            # --- 고도화된 Fallback 로직 시작 ---

            # 1. 한국시리즈/우승팀 질문 (get_korean_series_winner)
            if fallback_tool is None and any(
                word in query_lower
                for word in ["우승", "한국시리즈", "코리안시리즈", "결승"]
            ):
                fallback_tool = ToolCall(
                    tool_name="get_korean_series_winner",
                    parameters={"year": extracted_year},
                )
                analysis = f"{extracted_year}년 한국시리즈/우승팀 정보 조회 (Fallback)"

            # 2. 팀 순위/성적 질문 (get_team_rank)
            elif fallback_tool is None and any(
                word in query_lower
                for word in ["순위", "성적", "기록", "랭킹", "승률", "몇승", "몇패"]
            ):
                # 팀이 명시되었거나 "팀별"인 경우
                team_name = (
                    potential_player_name
                    if potential_player_name
                    and any(
                        t in query_lower
                        for t in [
                            "기아",
                            "삼성",
                            "엘지",
                            "두산",
                            "롯데",
                            "키움",
                            "한화",
                            "엔씨",
                            "에스에스지",
                            "케이티",
                        ]
                    )
                    else entity_filter.team_id
                )

                if team_name:
                    fallback_tool = ToolCall(
                        tool_name="get_team_rank",
                        parameters={"team_name": team_name, "year": extracted_year},
                    )
                    analysis = (
                        f"{extracted_year}년 {team_name} 팀 성적/순위 조회 (Fallback)"
                    )
                else:
                    # 팀이 명시되지 않은 전체 순위 질문이면 리더보드로 (정규시즌 순위 개념)
                    fallback_tool = ToolCall(
                        tool_name="get_leaderboard",
                        parameters={
                            "stat_name": "ops",  # 팀 순위 대용 (정규시즌 리더보드)
                            "year": extracted_year,
                            "position": "batting",
                            "limit": 10,
                        },
                    )
                    analysis = (
                        f"{extracted_year}년 전체 팀 순위/리더보드 조회 (Fallback)"
                    )

            # 3. 선수 개인 기록 질문
            elif (
                fallback_tool is None
                and potential_player_name
                and not any(
                    word in query_lower for word in ["순위", "리더보드", "랭킹"]
                )
            ):
                fallback_tool = ToolCall(
                    tool_name="get_career_stats",
                    parameters={
                        "player_name": potential_player_name,
                        "position": entity_filter.position_type or "both",
                    },
                )
                analysis = f"{potential_player_name} 선수의 통계 조회 (Fallback)"

            # 4. 특정 통계 리더보드 질문
            elif fallback_tool is None and entity_filter.stat_type:
                pos = entity_filter.position_type
                if not pos:
                    # 지표로 포지션 추측
                    if entity_filter.stat_type in [
                        "era",
                        "whip",
                        "wins",
                        "saves",
                        "strikeouts",
                        "innings_pitched",
                    ]:
                        pos = "pitching"
                    else:
                        pos = "batting"

                fallback_tool = ToolCall(
                    tool_name="get_leaderboard",
                    parameters={
                        "stat_name": entity_filter.stat_type,
                        "year": extracted_year,
                        "position": pos,
                        "limit": 10,
                    },
                )
                analysis = f"{extracted_year}년 {entity_filter.stat_type} 리더보드 조회 (Fallback)"

            # 5. 최후의 보루 (OPS 리더보드)
            if not fallback_tool:
                fallback_tool = ToolCall(
                    tool_name="get_leaderboard",
                    parameters={
                        "stat_name": "ops",
                        "year": extracted_year,
                        "position": "batting",
                        "limit": 10,
                    },
                )
                analysis = "질문 의도 파악이 어려워 기본 리더보드(OPS) 조회 (Fallback)"

            return {
                "analysis": analysis,
                "tool_calls": [fallback_tool],
                "expected_result": "조회 결과",
                "error": None,
                "planner_mode": planner_mode,
            }
        except Exception as e:
            logger.exception(f"[BaseballAgent] Error in query analysis: {e}")
            return {
                "analysis": "",
                "tool_calls": [],
                "expected_result": "",
                "error": f"질문 분석 오류: {e}",
                "planner_mode": planner_mode,
            }

    def _format_deterministic_metric(self, value: Any) -> str:
        if value is None or value == "":
            return "확인 불가"
        if isinstance(value, float):
            text = f"{value:.3f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return text
        return str(value)

    def _build_award_answer(self, data: Dict[str, Any]) -> Optional[str]:
        awards = data.get("awards")
        if not isinstance(awards, list) or not awards:
            return None

        year = self._format_deterministic_metric(data.get("year"))
        requested_type = data.get("award_type")
        top_entries = awards[:5]
        rows = []
        for award in top_entries:
            rows.append(
                "| "
                + f"{self._display_award_type(award.get('award_type'))} | "
                + f"{self._format_deterministic_metric(award.get('player_name'))} | "
                + f"{self._format_deterministic_metric(award.get('team_name') or '-')} | "
                + f"{self._format_deterministic_metric(award.get('position') or '-')} |"
            )

        if requested_type and requested_type != "any" and len(awards) == 1:
            winner = awards[0]
            team_text = f" ({winner['team_name']})" if winner.get("team_name") else ""
            summary = (
                f"{year}년 KBO {self._display_award_type(requested_type)} 수상자는 "
                f"{winner['player_name']}{team_text}입니다."
            )
            detail_lines = [
                f"- 조회된 수상 유형은 {self._display_award_type(requested_type)}입니다.",
                f"- 포지션 표기는 {self._format_deterministic_metric(winner.get('position'))}입니다.",
            ]
        else:
            lead = awards[0]
            summary = (
                f"{year}년 KBO 수상 기록은 {len(awards)}건 확인됐고, "
                f"대표적으로 {self._display_award_type(lead.get('award_type'))}는 "
                f"{self._format_deterministic_metric(lead.get('player_name'))}입니다."
            )
            detail_lines = [
                f"- 요청 연도는 {year}년입니다.",
                f"- 표에는 상위 {len(top_entries)}건만 요약했습니다.",
            ]

        return (
            "## 요약\n"
            f"{summary}\n\n"
            "## 상세 내역\n"
            f"{chr(10).join(detail_lines)}\n\n"
            "## 핵심 지표\n"
            "| 수상 | 선수 | 팀 | 포지션 |\n"
            "| --- | --- | --- | --- |\n"
            f"{chr(10).join(rows)}\n\n"
            "## 인사이트\n"
            "- 수상 질문은 시즌 연도와 상 이름이 함께 들어가면 가장 안정적으로 확인됩니다.\n"
            "- 동일 시즌의 다른 타이틀 홀더가 궁금하면 같은 연도로 바로 이어서 조회하면 됩니다.\n"
            "출처: DB 조회 결과"
        )

    def _build_player_stats_answer(self, data: Dict[str, Any]) -> Optional[str]:
        batting = (
            data.get("batting_stats")
            if isinstance(data.get("batting_stats"), dict)
            else None
        )
        pitching = (
            data.get("pitching_stats")
            if isinstance(data.get("pitching_stats"), dict)
            else None
        )
        if not batting and not pitching:
            return None

        player_name = (
            (batting or {}).get("player_name")
            or (pitching or {}).get("player_name")
            or data.get("player_name")
        )
        season_label = (
            "통산" if data.get("career") else f"{data.get('year', '해당')}시즌"
        )
        rows: List[str] = []
        detail_lines: List[str] = []

        if batting:
            avg_key = "career_avg" if data.get("career") else "avg"
            ops_key = "career_ops" if data.get("career") else "ops"
            hr_key = "total_home_runs" if data.get("career") else "home_runs"
            rbi_key = "total_rbi" if data.get("career") else "rbi"
            rows.extend(
                [
                    f"| 타격 타율 | {self._format_deterministic_metric(batting.get(avg_key))} | {season_label} 타격 지표 |",
                    f"| 타격 OPS | {self._format_deterministic_metric(batting.get(ops_key))} | 장타력/출루 생산성 |",
                    f"| 홈런 | {self._format_deterministic_metric(batting.get(hr_key))} | 장타 생산 |",
                    f"| 타점 | {self._format_deterministic_metric(batting.get(rbi_key))} | 득점 기여 |",
                ]
            )
            detail_lines.append(
                f"- 타격 데이터 기준 팀은 {self._format_deterministic_metric(batting.get('team_name'))}이고, 기준 구간은 {season_label}입니다."
            )

        if pitching:
            era_key = "career_era" if data.get("career") else "era"
            whip_key = "career_whip" if data.get("career") else "whip"
            win_key = "total_wins" if data.get("career") else "wins"
            so_key = "total_strikeouts" if data.get("career") else "strikeouts"
            rows.extend(
                [
                    f"| 투구 ERA | {self._format_deterministic_metric(pitching.get(era_key))} | 실점 억제력 |",
                    f"| 투구 WHIP | {self._format_deterministic_metric(pitching.get(whip_key))} | 주자 허용 지표 |",
                    f"| 승수 | {self._format_deterministic_metric(pitching.get(win_key))} | 승리 기여 |",
                    f"| 삼진 | {self._format_deterministic_metric(pitching.get(so_key))} | 탈삼진 생산 |",
                ]
            )
            detail_lines.append(
                f"- 투구 데이터 기준 팀은 {self._format_deterministic_metric(pitching.get('team_name'))}이고, 기준 구간은 {season_label}입니다."
            )

        rows = rows[:4]
        detail_lines = detail_lines[:2]
        summary = (
            f"{player_name}의 {season_label} 기록은 DB 기준으로 바로 확인 가능합니다."
        )
        if batting and not pitching:
            summary = (
                f"{player_name}의 {season_label} 타격 기록은 "
                f"타율 {self._format_deterministic_metric(batting.get('career_avg' if data.get('career') else 'avg'))}, "
                f"OPS {self._format_deterministic_metric(batting.get('career_ops' if data.get('career') else 'ops'))} 수준입니다."
            )
        elif pitching and not batting:
            summary = (
                f"{player_name}의 {season_label} 투구 기록은 "
                f"ERA {self._format_deterministic_metric(pitching.get('career_era' if data.get('career') else 'era'))}, "
                f"WHIP {self._format_deterministic_metric(pitching.get('career_whip' if data.get('career') else 'whip'))} 기준으로 확인됩니다."
            )

        return (
            "## 요약\n"
            f"{summary}\n\n"
            "## 상세 내역\n"
            f"{chr(10).join(detail_lines) if detail_lines else '- 타격/투구 중 조회 가능한 기록만 반영했습니다.'}\n\n"
            "## 핵심 지표\n"
            "| 항목 | 값 | 해석 |\n"
            "| --- | --- | --- |\n"
            f"{chr(10).join(rows)}\n\n"
            "## 인사이트\n"
            f"- 이 답변은 {season_label} 공식 DB 집계만 사용했습니다.\n"
            "- 타격과 투구가 함께 잡힌 경우, 포지션별 역할 차이는 표 수치로 분리해 보는 편이 정확합니다.\n"
            "출처: DB 조회 결과"
        )

    def _build_leaderboard_answer(self, data: Dict[str, Any]) -> Optional[str]:
        leaderboard = data.get("leaderboard")
        if not isinstance(leaderboard, list) or not leaderboard:
            return None

        stat_name = self._format_deterministic_metric(data.get("stat_name"))
        year = self._format_deterministic_metric(data.get("year"))
        position = self._format_deterministic_metric(data.get("position"))
        team_filter = data.get("team_filter")
        top_entries = leaderboard[:3]
        leader = top_entries[0]
        leader_name = leader.get("player_name") or leader.get("team_name") or "1위"
        leader_value = self._format_deterministic_metric(leader.get("stat_value"))
        rows = []
        for idx, entry in enumerate(top_entries, start=1):
            rows.append(
                "| "
                + f"{idx}위 | "
                + f"{self._format_deterministic_metric(entry.get('player_name') or entry.get('team_name'))} | "
                + f"{self._format_deterministic_metric(entry.get('team_name'))} | "
                + f"{self._format_deterministic_metric(entry.get('stat_value'))} |"
            )

        detail_lines = [
            f"- 조회 지표는 {year}년 {position} {stat_name} 기준입니다.",
            f"- 표에는 상위 {len(top_entries)}명만 요약했고, 전체 적격 인원은 {self._format_deterministic_metric(data.get('total_qualified_players'))}명입니다.",
        ]
        if team_filter:
            detail_lines[1] = (
                f"- 팀 필터는 {team_filter}로 적용됐고, 적격 인원은 {self._format_deterministic_metric(data.get('total_qualified_players'))}명입니다."
            )

        return (
            "## 요약\n"
            f"{year}년 {stat_name} 리더보드 상단은 {leader_name}({leader_value})가 잡힙니다.\n\n"
            "## 상세 내역\n"
            f"{chr(10).join(detail_lines)}\n\n"
            "## 핵심 지표\n"
            "| 순위 | 선수/팀 | 소속 | 수치 |\n"
            "| --- | --- | --- | --- |\n"
            f"{chr(10).join(rows)}\n\n"
            "## 인사이트\n"
            f"- {stat_name}처럼 순위형 질문은 상단 3명만 봐도 리그 흐름을 빠르게 읽을 수 있습니다.\n"
            f"- {leader_name}가 선두인 만큼, 비교 질문이면 바로 아래 경쟁자와 격차를 추가 조회하는 방식이 좋습니다.\n"
            "출처: DB 조회 결과"
        )

    def _build_regulation_answer(self, data: Dict[str, Any]) -> Optional[str]:
        regulations = data.get("regulations")
        if not isinstance(regulations, list) or not regulations:
            return None

        top_entries = regulations[:3]
        query_or_category = data.get("query") or data.get("category") or "규정"
        lead = top_entries[0]
        preview = " ".join(str(lead.get("content", "")).split())
        preview = preview[:120] + ("..." if len(preview) > 120 else "")
        rows = []
        for entry in top_entries:
            rows.append(
                "| "
                + f"{self._format_deterministic_metric(entry.get('regulation_code'))} | "
                + f"{self._format_deterministic_metric(entry.get('title'))} | "
                + f"{self._format_deterministic_metric(entry.get('category') or entry.get('document_type'))} |"
            )

        return (
            "## 요약\n"
            f"'{query_or_category}' 관련 규정은 DB 검색 기준으로 {len(regulations)}건 확인됐고, 우선순위가 가장 높은 조항부터 정리합니다.\n\n"
            "## 상세 내역\n"
            f"- 최상단 규정 제목은 {self._format_deterministic_metric(lead.get('title'))}입니다.\n"
            f"- 본문 미리보기: {preview or '확인 불가'}\n\n"
            "## 핵심 지표\n"
            "| 규정 코드 | 제목 | 분류 |\n"
            "| --- | --- | --- |\n"
            f"{chr(10).join(rows)}\n\n"
            "## 인사이트\n"
            "- 규정 질문은 해석보다 원문 제목과 조항 코드를 먼저 확인하는 편이 안전합니다.\n"
            "- 세부 적용 사례가 필요하면 같은 키워드로 조항 본문을 더 좁혀 재검색하는 방식이 좋습니다.\n"
            "출처: DB 조회 결과"
        )

    def _build_games_by_date_answer(self, data: Dict[str, Any]) -> Optional[str]:
        games = data.get("games")
        if not isinstance(games, list):
            return None

        date = self._format_deterministic_metric(data.get("date"))
        if date == "확인 불가":
            return None

        team_filter = self._format_team_display_name(data.get("team_filter"))
        if not games:
            detail_lines = [f"- 조회 날짜는 {date}입니다."]
            if team_filter != "확인 불가":
                detail_lines.append(f"- 팀 필터는 {team_filter} 기준으로 적용됐습니다.")
                summary_line = (
                    f"{date} 일정 중 {team_filter} 기준으로 잡힌 경기는 없습니다."
                )
            else:
                summary_line = f"{date}에는 DB 기준으로 잡힌 KBO 경기가 없습니다."
            detail_lines.append("- 확인된 경기 수는 0경기입니다.")

            return (
                "## 요약\n"
                f"{summary_line}\n\n"
                "## 상세 내역\n"
                f"{chr(10).join(detail_lines)}\n\n"
                "## 핵심 지표\n"
                "| 항목 | 값 |\n"
                "| --- | --- |\n"
                f"| 조회 날짜 | {date} |\n"
                f"| 팀 필터 | {team_filter} |\n"
                "| 확인된 경기 수 | 0 |\n\n"
                "## 인사이트\n"
                "- 해당 날짜에는 편성된 경기가 없는 날일 수 있습니다.\n"
                "- 특정 팀 일정이 필요하면 다른 날짜나 팀명을 같이 지정해 다시 조회하면 됩니다.\n"
                "출처: DB 조회 결과"
            )

        rows = []
        completed_count = 0
        for game in games[:4]:
            status = self._format_deterministic_metric(game.get("game_status"))
            home_score = self._format_deterministic_metric(game.get("home_score"))
            away_score = self._format_deterministic_metric(game.get("away_score"))
            if status == "COMPLETED":
                completed_count += 1
            score_text = (
                f"{away_score}-{home_score}"
                if game.get("home_score") is not None
                and game.get("away_score") is not None
                else "-"
            )
            rows.append(
                "| "
                + f"{self._format_team_display_name(game.get('away_team'))} @ {self._format_team_display_name(game.get('home_team'))} | "
                + f"{score_text} | "
                + f"{status} | "
                + f"{self._format_deterministic_metric(game.get('stadium'))} |"
            )

        detail_lines = [f"- 조회 날짜는 {date}입니다."]
        if team_filter != "확인 불가":
            detail_lines.append(f"- 팀 필터는 {team_filter} 기준으로 적용됐습니다.")
        detail_lines.append(
            f"- 확인된 경기 수는 {len(games)}경기이며, 종료 상태로 잡힌 경기는 {completed_count}경기입니다."
        )

        return (
            "## 요약\n"
            f"{date} 경기 일정/결과는 DB 기준으로 {len(games)}경기 확인됩니다.\n\n"
            "## 상세 내역\n"
            f"{chr(10).join(detail_lines)}\n\n"
            "## 핵심 지표\n"
            "| 경기 | 스코어 | 상태 | 구장 |\n"
            "| --- | --- | --- | --- |\n"
            f"{chr(10).join(rows)}\n\n"
            "## 인사이트\n"
            "- 날짜별 질의는 경기 수와 종료 여부를 먼저 보면 일정 확인과 결과 확인을 한 번에 처리할 수 있습니다.\n"
            "- 특정 팀 흐름이 목적이면 같은 날짜 기준으로 상대팀과 스코어를 이어서 비교하는 게 효율적입니다.\n"
            "출처: DB 조회 결과"
        )

    def _build_team_last_game_date_answer(self, data: Dict[str, Any]) -> Optional[str]:
        if not data.get("last_game_date"):
            return None

        team_name = self._format_team_display_name(data.get("team_name"))
        year = self._format_deterministic_metric(data.get("year"))
        last_game_date = self._format_deterministic_metric(data.get("last_game_date"))

        return (
            "## 요약\n"
            f"{team_name}의 {year}년 마지막 경기 날짜는 {last_game_date}로 확인됩니다.\n\n"
            "## 상세 내역\n"
            f"- 조회 대상 팀은 {team_name}입니다.\n"
            f"- 시즌 기준 연도는 {year}년입니다.\n\n"
            "## 핵심 지표\n"
            "| 항목 | 값 | 비고 |\n"
            "| --- | --- | --- |\n"
            f"| 팀 | {team_name} | 조회 대상 |\n"
            f"| 시즌 | {year} | 기준 연도 |\n"
            f"| 마지막 경기 날짜 | {last_game_date} | 완료 경기 기준 |\n\n"
            "## 인사이트\n"
            "- 마지막 경기 날짜는 시즌 종료 시점 확인이나 포스트시즌 연결 질문의 기준점으로 쓰기 좋습니다.\n"
            "- 상대팀이나 경기 결과까지 필요하면 같은 날짜로 상세 경기 조회를 이어서 붙이면 됩니다.\n"
            "출처: DB 조회 결과"
        )

    def _pick_metric_value(
        self, data: Dict[str, Any], candidate_keys: List[str]
    ) -> Optional[Any]:
        for key in candidate_keys:
            value = data.get(key)
            if value is not None and value != "":
                return value
        return None

    def _shorten_chat_excerpt(self, text: Any, limit: int = 90) -> str:
        raw = str(text or "").strip()
        if not raw:
            return "확인 가능한 내용이 아직 충분히 잡히지 않았습니다."
        normalized = " ".join(raw.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: limit - 1].rstrip()}…"

    def _format_team_display_name(self, team_value: Any) -> str:
        return self._format_deterministic_metric(
            replace_team_codes(
                team_value,
                team_name_resolver=self._convert_team_id_to_name,
            )
        )

    def _build_player_stats_chat_answer(self, data: Dict[str, Any]) -> Optional[str]:
        player_name = self._format_deterministic_metric(data.get("player_name"))
        if player_name == "확인 불가":
            return None

        year = self._format_deterministic_metric(data.get("year"))
        team_name = data.get("team_name")
        batting_stats = data.get("batting_stats") or {}
        pitching_stats = data.get("pitching_stats") or {}

        intro = f"{player_name} 기록은 현재 이렇게 보입니다."
        if team_name and year != "확인 불가":
            intro = f"{year}년 {team_name} {player_name} 기록은 현재 이렇게 보입니다."
        elif year != "확인 불가":
            intro = f"{year}년 {player_name} 기록은 현재 이렇게 보입니다."

        lines = [intro]

        if batting_stats:
            batting_bits = []
            avg = self._pick_metric_value(batting_stats, ["avg", "batting_avg"])
            ops = self._pick_metric_value(batting_stats, ["ops"])
            home_runs = self._pick_metric_value(
                batting_stats, ["home_runs", "hr", "homeruns"]
            )
            rbi = self._pick_metric_value(batting_stats, ["rbi", "runs_batted_in"])
            if avg is not None:
                batting_bits.append(f"타율 {avg}")
            if ops is not None:
                batting_bits.append(f"OPS {ops}")
            if home_runs is not None:
                batting_bits.append(f"홈런 {home_runs}")
            if rbi is not None:
                batting_bits.append(f"타점 {rbi}")
            if batting_bits:
                lines.append(
                    f"타격 쪽에서는 {', '.join(batting_bits[:4])} 정도가 바로 눈에 들어옵니다."
                )

        if pitching_stats:
            pitching_bits = []
            era = self._pick_metric_value(pitching_stats, ["era"])
            whip = self._pick_metric_value(pitching_stats, ["whip"])
            wins = self._pick_metric_value(pitching_stats, ["wins", "win"])
            saves = self._pick_metric_value(pitching_stats, ["saves", "save"])
            holds = self._pick_metric_value(pitching_stats, ["holds", "hold"])
            strikeouts = self._pick_metric_value(
                pitching_stats, ["strikeouts", "so", "k"]
            )
            if era is not None:
                pitching_bits.append(f"평균자책 {era}")
            if whip is not None:
                pitching_bits.append(f"WHIP {whip}")
            if wins is not None:
                pitching_bits.append(f"{wins}승")
            if saves is not None:
                pitching_bits.append(f"{saves}세이브")
            if holds is not None:
                pitching_bits.append(f"{holds}홀드")
            if strikeouts is not None:
                pitching_bits.append(f"탈삼진 {strikeouts}")
            if pitching_bits:
                lines.append(
                    f"투수 기록으로 보면 {', '.join(pitching_bits[:4])} 정도로 정리됩니다."
                )

        if len(lines) == 1:
            lines.append(
                "지금 붙은 기록에서는 세부 지표가 충분히 안 잡혀서 시즌과 지표를 같이 물어보면 더 정확하게 볼 수 있습니다."
            )

        return "\n\n".join(lines[:3])

    def _extract_leaderboard_value(self, entry: Dict[str, Any], stat_name: str) -> Any:
        stat_key = str(stat_name or "").lower()
        candidate_keys = [
            "value",
            "stat_value",
            "metric_value",
            stat_key,
            stat_key.replace(" ", "_"),
            "avg",
            "ops",
            "era",
            "home_runs",
            "rbi",
            "saves",
            "holds",
            "strikeouts",
            "wins",
            "whip",
        ]
        for key in candidate_keys:
            if key in entry and entry.get(key) is not None:
                return entry.get(key)
        return "확인 불가"

    def _build_leaderboard_chat_answer(
        self, query: str, data: Dict[str, Any]
    ) -> Optional[str]:
        leaderboard = data.get("leaderboard") or data.get("leaders") or []
        if not leaderboard:
            return None

        decision = self._resolve_chat_intent(query, extract_entities_from_query(query))
        registry_answer = self.chat_renderer_registry.render_leaderboard(
            query, data, decision
        )
        if registry_answer:
            return registry_answer

        year = self._format_deterministic_metric(data.get("year"))
        raw_stat_name = self._format_deterministic_metric(data.get("stat_name"))
        stat_key = re.sub(r"[^a-z0-9가-힣_]+", "", str(raw_stat_name).lower())
        stat_name_map = {
            "era": "평균자책점(ERA)",
            "avg": "타율",
            "battingavg": "타율",
            "ops": "OPS",
            "hr": "홈런",
            "homeruns": "홈런",
            "rbi": "타점",
            "wins": "다승",
            "win": "다승",
            "whip": "WHIP",
            "saves": "세이브",
            "save": "세이브",
            "holds": "홀드",
            "hold": "홀드",
            "strikeouts": "탈삼진",
            "so": "탈삼진",
            "war": "WAR",
        }
        stat_name = stat_name_map.get(stat_key, raw_stat_name)
        season_label = f"{year}년" if year != "확인 불가" else "해당 시즌"
        query_lower = query.lower()

        def _player_label(entry: Dict[str, Any]) -> str:
            player_name = self._format_deterministic_metric(
                entry.get("player_name") or entry.get("name")
            )
            team_name = self._format_deterministic_metric(
                entry.get("team_name") or entry.get("team")
            )
            if team_name != "확인 불가":
                return f"{player_name}({team_name})"
            return player_name

        top_entry = leaderboard[0]
        top_player = _player_label(top_entry)
        top_value = self._format_deterministic_metric(
            self._extract_leaderboard_value(top_entry, raw_stat_name)
        )

        asks_superlative = any(
            token in query_lower
            for token in ["최고", "제일", "가장", "누가", "누구야", "잘한"]
        )
        role_label = "선수"
        if "투수" in query_lower:
            role_label = "투수"
        elif "타자" in query_lower:
            role_label = "타자"

        if asks_superlative:
            return (
                f"{season_label} 최고 {role_label}를 한 명만 꼽자면, 지금 조회된 기록에선 "
                f"{stat_name} 기준 1위인 {top_player}입니다.\n\n"
                f"현재 확인된 수치는 {stat_name} {top_value}이고, 지금 답변은 이 지표 하나를 기준으로 잡은 결과입니다.\n\n"
                "다만 '최고'라는 말은 기준에 따라 달라집니다. 원하면 "
                f"{stat_name} 기준, WAR 기준, 팬 체감 기준으로 나눠서 다시 정리해드릴게요."
            )

        intro = f"{season_label} {stat_name} 기준으로 보면 상위권은 이렇게 보입니다."
        lines = [intro]

        for index, entry in enumerate(leaderboard[:3], start=1):
            player_name = _player_label(entry)
            stat_value = self._format_deterministic_metric(
                self._extract_leaderboard_value(entry, raw_stat_name)
            )
            lines.append(
                f"{index}위는 {player_name}이고, {stat_name}은 {stat_value}입니다."
            )

        return "\n\n".join(lines[:4])

    def _build_regulation_chat_answer(
        self, query: str, data: Dict[str, Any]
    ) -> Optional[str]:
        regulations = data.get("regulations") or []
        if not regulations:
            return None

        query_lower = query.lower()

        if any(keyword in query_lower for keyword in ["fa", "보상선수", "보상 선수"]):
            return (
                "FA와 보상선수 기준은 선수 규정에서 핵심으로 다루는 항목입니다.\n\n"
                "지금 붙은 규정 기준으로는 보상선수 여부와 보상 방식은 시즌별 공식 규정과 공시를 같이 봐야 안전합니다.\n\n"
                "특히 연차, 등록일수, 등급별 보상 기준처럼 숫자로 끊기는 부분은 시즌마다 손볼 수 있어서, 그 수치를 여기서 단정하진 않겠습니다."
            )

        if any(
            keyword in query_lower
            for keyword in ["선수 등록", "등록 규정", "등록 현황", "등말소", "등·말소"]
        ):
            return (
                "선수 등록 규정은 공식 선수 등록 현황과 선수 기록 페이지를 같이 보는 흐름으로 잡으면 됩니다.\n\n"
                "지금 붙은 규정 기준으로도 구단별 등록 현황, 퓨처스 등록 현황, 선수별 엔트리 등록·말소 일수를 먼저 확인하고, 세부 숫자 기준은 시즌 공지와 규정 원문으로 다시 맞춰보라는 방향에 가깝습니다.\n\n"
                "즉 화면에 보이는 등록 상태를 먼저 보고, 인원 기준이나 운영 세칙은 최신 시즌 규정으로 한 번 더 대조하는 방식이 가장 안전합니다."
            )

        if any(
            keyword in query_lower
            for keyword in ["엔트리", "등록일수", "말소", "부상자", "il"]
        ):
            if any(keyword in query_lower for keyword in ["부상자", "il"]):
                return (
                    "부상자 명단은 선수 상태가 바뀌었을 때 엔트리에서 빠지고 다시 복귀하는 흐름으로 이해하면 됩니다.\n\n"
                    "지금 붙은 규정 기준으로도 핵심은 선수 기록 페이지의 등록·말소 일수와 공식 등록 현황을 함께 보는 쪽에 가깝습니다.\n\n"
                    "최소 일수나 세부 분류처럼 숫자로 잘라 말해야 하는 부분은 시즌 운영 규정이 바뀔 수 있어서, 최신 시즌 공지와 규정 원문을 같이 보는 게 안전합니다."
                )
            return (
                "엔트리 등록일수와 말소 일수는 선수 기록 페이지와 공식 등록 현황에서 확인하는 흐름으로 이해하면 됩니다.\n\n"
                "지금 잡힌 규정도 선수별 등록/말소 내역을 먼저 보고, 세부 운영 기준은 시즌 공지와 규정 원문으로 다시 대조하라는 방향에 가깝습니다.\n\n"
                "즉, 기록실에서 일수와 등·말소 내역을 확인하고, 최신 시즌 규정으로 숫자 기준을 한 번 더 맞춰보는 방식이 가장 안전합니다."
            )

        if "abs" in query_lower or "자동 투구 판정" in query_lower:
            return (
                "ABS는 공의 궤적이 스트라이크존을 통과했는지 추적 시스템으로 판정하는 자동 투구 판정 시스템입니다.\n\n"
                "핵심은 심판의 육안만으로 볼·스트라이크를 판단하는 대신, 시스템이 스트라이크존 통과 여부를 일관되게 계산해 준다는 점입니다.\n\n"
                "즉 타자별 기준 스트라이크존과 투구 궤적 데이터를 결합해 판정 정확도와 일관성을 높이는 방식으로 이해하면 됩니다."
            )

        if "타이브레이크" in query_lower:
            return (
                "타이브레이크는 연장전에서 승부를 빨리 가리기 위해 공격 시작을 무사 1,2루로 두는 제도입니다.\n\n"
                "핵심은 연장 이닝마다 주자를 미리 두고 시작해서 득점 가능성을 높이고, 경기 시간이 과도하게 길어지는 걸 줄이는 데 있습니다.\n\n"
                "즉 일반 이닝처럼 빈 베이스에서 시작하는 연장이 아니라, 무사 1,2루라는 특수 조건에서 시작하는 연장전 규정으로 보면 됩니다."
            )

        if "와일드카드" in query_lower:
            return (
                "KBO 포스트시즌 와일드카드는 정규시즌 4위와 5위가 맞붙어 준플레이오프 진출팀을 가리는 단계입니다.\n\n"
                "핵심은 정규시즌 4위 팀이 1승 어드밴티지를 안고 시작하고, 5위 팀은 시리즈를 뒤집으려면 연속으로 이겨야 한다는 점입니다.\n\n"
                "즉 정규시즌 순위 우위를 반영해 4위가 더 유리한 조건에서 출발하는 포스트시즌 관문이라고 이해하면 됩니다."
            )

        if "연장전" in query_lower:
            return (
                "연장전은 9회까지 승부가 나지 않았을 때 이어서 치르는 추가 이닝입니다.\n\n"
                "KBO 1군 정규시즌은 12회까지 진행하고도 승패가 안 갈리면 무승부로 끝내는 것이 기본입니다.\n\n"
                "즉 일반적으로는 9회 이후에도 승부를 이어가되, 정규시즌은 12회가 끝나면 더 하지 않고 종료한다고 보면 됩니다."
            )

        if "지명타자" in query_lower:
            return (
                "지명타자 제도는 투수 대신 타석에 서는 전담 타자를 두는 제도입니다.\n\n"
                "지명타자는 공격에서는 타격만 맡고 수비 포지션은 소화하지 않으므로, 투수 타석을 공격력 좋은 선수로 대체하는 효과가 있습니다.\n\n"
                "즉 투수의 타석 부담을 줄이고 타선 생산력을 높이기 위한 제도로 이해하면 됩니다."
            )

        if "사구" in query_lower and "고의4구" in query_lower:
            return (
                "사구는 투구가 타자의 몸에 맞아 1루에 나가는 경우이고, 고의4구는 투수가 의도적으로 타자를 볼넷으로 내보내는 경우입니다.\n\n"
                "사구는 몸에 맞는 공이라는 접촉이 핵심이고, 고의4구는 접촉 없이 볼넷을 허용한다는 점이 가장 큰 차이입니다.\n\n"
                "즉 결과는 둘 다 타자가 1루로 나간다는 점에서 비슷하지만, 사유와 판정 방식은 분명히 다릅니다."
            )

        if "낫아웃" in query_lower:
            return (
                "낫아웃은 삼진이 나왔더라도 포수가 공을 완전히 포구하지 못하면 타자가 1루로 뛸 수 있는 규정입니다.\n\n"
                "보통 1루가 비어 있거나 2아웃일 때 성립하고, 포수가 공을 제대로 잡으면 그대로 삼진 아웃으로 끝납니다.\n\n"
                "즉 삼진이 곧바로 플레이 종료를 뜻하는 게 아니라, 포구 여부에 따라 타자가 살아날 수 있는 예외 규정으로 이해하면 됩니다."
            )

        if "스트라이크존" in query_lower:
            return (
                "스트라이크존은 홈플레이트 위를 통과한 공이 타자의 무릎과 어깨 기준 높이 사이를 지났는지로 판단합니다.\n\n"
                "핵심은 공이 단순히 가운데로 왔는지가 아니라, 홈플레이트 폭 안과 타자 신체 기준 높이를 함께 충족했는지 보는 데 있습니다.\n\n"
                "즉 홈플레이트 위를 지나면서 무릎에서 어깨 부근 사이 높이를 통과한 공이 기본적인 스트라이크존이라고 이해하면 됩니다."
            )

        if "비디오 판독" in query_lower:
            return (
                "비디오 판독은 현장 판정이 애매한 플레이를 영상으로 다시 확인해 바로잡는 절차입니다.\n\n"
                "대표적으로 홈런 여부, 세이프와 아웃 판정, 페어와 파울, 포스나 태그 상황처럼 경기 결과에 직접 영향을 주는 장면이 주요 대상입니다.\n\n"
                "즉 육안으로 확신하기 어려운 핵심 플레이를 영상으로 다시 확인해 판정을 정정하는 제도라고 보면 됩니다."
            )

        if any(keyword in query_lower for keyword in ["외국인 선수", "외국인선수"]):
            return (
                "외국인 선수 규정은 등록과 출전 기준을 시즌별 운영 규정으로 확인하는 쪽이 가장 안전합니다.\n\n"
                "지금 붙은 문서도 세부 제한을 단정하기보다, 시즌별 규정 변경 공지와 리그 운영 규정을 우선 보라고 정리하고 있습니다.\n\n"
                "즉 숫자나 보유 한도를 바로 잘라 말하기보다, 해당 시즌 공식 공지와 규정・자료실 문서를 같이 보는 방식으로 이해하면 됩니다."
            )

        if any(
            keyword in query_lower
            for keyword in ["육성선수", "육성 선수", "군보류", "임의해지", "특수 신분"]
        ):
            return (
                "육성선수나 군보류, 임의해지처럼 특수 신분 선수는 공시 상태와 시즌별 선수 등록 규정을 같이 보는 게 핵심입니다.\n\n"
                "지금 잡힌 규정 기준으로도 정식 선수 전환 여부나 복귀 가능 여부는 구단 공시와 시즌별 등록 규정으로 확인하라는 방향에 가깝습니다.\n\n"
                "즉 선수 상태가 어떻게 공시됐는지 먼저 보고, 세부 복귀 조건이나 전환 기준은 최신 시즌 규정으로 다시 맞춰보는 방식이 가장 안전합니다."
            )

        lines: List[str] = []
        for regulation in regulations[:2]:
            content = (
                regulation.get("content")
                or regulation.get("summary")
                or regulation.get("description")
                or regulation.get("text")
                or ""
            )
            cleaned = re.sub(r"[#*_`>-]+", " ", str(content))
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            cleaned = re.sub(
                r"^(?:KBO\s+)?[^.]{0,80}?(?:개요|해설|스토리라인)\s+",
                "",
                cleaned,
            )
            cleaned = re.sub(
                r"^(?:이 문서는|챗봇은)\s+[^.]+?\.\s*",
                "",
                cleaned,
            )
            cleaned = re.sub(r"^세부 조항\s*", "", cleaned)
            if cleaned:
                lines.append(self._shorten_chat_excerpt(cleaned))

        if not lines:
            return None
        lines.append(
            "예외 조항까지 보려면 규정 이름을 조금 더 좁혀서 다시 물어보는 편이 정확합니다."
        )
        return "\n\n".join(lines[:4])

    def _build_games_by_date_chat_answer(self, data: Dict[str, Any]) -> Optional[str]:
        games = data.get("games") or []
        date = self._format_deterministic_metric(data.get("date"))
        if date == "확인 불가":
            return None

        team_filter = self._format_team_display_name(data.get("team_filter"))
        if not games:
            if team_filter != "확인 불가":
                return (
                    f"{date} 일정 중 {team_filter} 기준으로 잡힌 경기는 없습니다.\n\n"
                    "DB 기준으로는 그 날짜에 해당 팀 경기가 편성되지 않은 상태입니다."
                )
            return (
                f"{date} 기준으로 DB에 잡힌 KBO 경기는 없습니다.\n\n"
                "현재 일정 자료상 그 날짜에는 편성된 경기가 없는 상태입니다."
            )

        intro = f"{date} 일정은 DB 기준으로 {len(games)}경기 잡혀 있습니다."
        if team_filter != "확인 불가":
            intro = f"{date} 일정 중 {team_filter} 기준으로 보면 {len(games)}경기 잡혀 있습니다."

        lines = [intro]
        for game in games[:2]:
            away_team = self._format_team_display_name(game.get("away_team"))
            home_team = self._format_team_display_name(game.get("home_team"))
            stadium = self._format_deterministic_metric(game.get("stadium"))
            status_value = game.get("status") or game.get("game_status")
            status = self._format_game_status_to_korean(
                status_value
            ) or self._format_deterministic_metric(status_value)
            raw_status = str(status_value or "").strip().upper()
            completed_status = raw_status == "COMPLETED" or status in {
                "완료",
                "종료",
                "경기 종료",
            }
            scheduled_status = raw_status in {"SCHEDULED", "PRE_GAME"} or status in {
                "예정",
                "경기 전",
            }
            live_status = raw_status in {"IN_PROGRESS", "LIVE"} or status in {
                "진행 중",
                "경기 중",
            }
            home_score = game.get("home_score")
            away_score = game.get("away_score")
            if home_score is not None and away_score is not None:
                if status == "확인 불가" or completed_status:
                    lines.append(
                        f"{away_team} 대 {home_team} 경기는 {stadium}에서 열렸고, 스코어는 {away_score}-{home_score}입니다."
                    )
                elif live_status:
                    lines.append(
                        f"{away_team} 대 {home_team} 경기는 {stadium}에서 진행 중이고, 현재 스코어는 {away_score}-{home_score}입니다."
                    )
                else:
                    lines.append(
                        f"{away_team} 대 {home_team} 경기는 {stadium}에서 열렸고, 스코어는 {away_score}-{home_score}입니다. 현재 상태는 {status}입니다."
                    )
            else:
                if status == "확인 불가" or scheduled_status:
                    lines.append(
                        f"{away_team} 대 {home_team} 경기는 {stadium}에서 열릴 예정입니다."
                    )
                elif live_status:
                    lines.append(
                        f"{away_team} 대 {home_team} 경기는 {stadium}에서 진행 중입니다."
                    )
                else:
                    lines.append(
                        f"{away_team} 대 {home_team} 경기는 {stadium}에서 열리는 일정으로 잡혀 있습니다."
                    )

        return "\n\n".join(lines[:3])

    def _build_team_last_game_date_chat_answer(
        self, data: Dict[str, Any]
    ) -> Optional[str]:
        last_game_date = data.get("last_game_date") or data.get("final_date")
        if not last_game_date:
            return None

        team_name = self._format_team_display_name(data.get("team_name"))
        year = self._format_deterministic_metric(data.get("year"))
        team_games = data.get("team_games") or data.get("all_games") or []
        first_game = (
            team_games[0] if isinstance(team_games, list) and team_games else {}
        )

        if not isinstance(first_game, dict) or not first_game:
            return f"{year} {team_name}의 마지막 경기는 {last_game_date}입니다."

        home_team = self._format_team_display_name(first_game.get("home_team"))
        away_team = self._format_team_display_name(first_game.get("away_team"))
        stadium = self._format_deterministic_metric(first_game.get("stadium"))
        home_score = first_game.get("home_score")
        away_score = first_game.get("away_score")

        team_score = None
        opponent_score = None
        opponent_team = "상대팀"

        if team_name == home_team:
            team_score = home_score
            opponent_score = away_score
            opponent_team = away_team
        elif team_name == away_team:
            team_score = away_score
            opponent_score = home_score
            opponent_team = home_team
        elif away_team != "확인 불가":
            opponent_team = away_team if away_team != team_name else home_team

        lead = f"{year} {team_name}의 마지막 경기는 {last_game_date}이었습니다."
        if stadium != "확인 불가":
            lead = f"{year} {team_name}의 마지막 경기는 {last_game_date} {stadium}에서 열렸습니다."

        if isinstance(team_score, (int, float)) and isinstance(
            opponent_score, (int, float)
        ):
            if team_score > opponent_score:
                result_text = f"{opponent_team}를 상대로 {team_score}-{opponent_score}로 이기고 시즌을 마쳤습니다."
            elif team_score < opponent_score:
                result_text = f"{opponent_team}를 상대로 {team_score}-{opponent_score}로 졌습니다."
            else:
                result_text = (
                    f"{opponent_team}와 {team_score}-{opponent_score} 무승부였습니다."
                )

            if (
                data.get("league_type") == "korean_series"
                and data.get("team_rank") == 1
                and team_score > opponent_score
            ):
                result_text = f"{opponent_team}를 상대로 {team_score}-{opponent_score}로 이기면서 우승으로 시즌을 마쳤습니다."

            return f"{lead} {result_text}"

        if opponent_team != "상대팀":
            return f"{lead} 상대는 {opponent_team}였습니다."

        return lead

    def _build_team_rank_chat_answer(self, data: Dict[str, Any]) -> Optional[str]:
        team_name = self._format_team_display_name(data.get("team_name"))
        year = self._format_deterministic_metric(data.get("year"))
        team_rank = data.get("team_rank") or data.get("season_rank")
        if team_name == "확인 불가" or year == "확인 불가" or not team_rank:
            return None

        wins = data.get("wins")
        losses = data.get("losses")
        draws = data.get("draws")
        record_bits = []
        if isinstance(wins, int):
            record_bits.append(f"{wins}승")
        if isinstance(losses, int):
            record_bits.append(f"{losses}패")
        if isinstance(draws, int) and draws > 0:
            record_bits.append(f"{draws}무")

        answer = f"{year}년 정규시즌 {team_rank}위는 {team_name}입니다."
        if record_bits:
            answer += f" 시즌 기록은 {' '.join(record_bits)}입니다."
        return answer

    def _build_team_rank_answer(self, data: Dict[str, Any]) -> Optional[str]:
        team_name = self._format_team_display_name(data.get("team_name"))
        year = self._format_deterministic_metric(data.get("year"))
        team_rank = data.get("team_rank") or data.get("season_rank")
        if team_name == "확인 불가" or year == "확인 불가" or not team_rank:
            return None

        wins = self._format_deterministic_metric(data.get("wins"))
        losses = self._format_deterministic_metric(data.get("losses"))
        draws = self._format_deterministic_metric(data.get("draws"))

        return (
            "## 요약\n"
            f"{year}년 정규시즌 {team_rank}위 팀은 {team_name}입니다.\n\n"
            "## 상세 내역\n"
            f"- 시즌 연도는 {year}년입니다.\n"
            f"- 최종 순위는 {team_rank}위입니다.\n\n"
            "## 핵심 지표\n"
            "| 항목 | 값 |\n"
            "| --- | --- |\n"
            f"| 팀 | {team_name} |\n"
            f"| 순위 | {team_rank}위 |\n"
            f"| 승리 | {wins} |\n"
            f"| 패배 | {losses} |\n"
            f"| 무승부 | {draws} |\n\n"
            "## 인사이트\n"
            "- 정규시즌 순위는 포스트시즌 진출 경로와 홈 어드밴티지 해석의 기준이 됩니다.\n"
            "- 같은 시즌 다른 순위 팀도 이어서 조회하면 경쟁 구도를 바로 비교할 수 있습니다.\n"
            "출처: DB 조회 결과"
        )

    def _build_korean_series_winner_chat_answer(
        self, data: Dict[str, Any]
    ) -> Optional[str]:
        winner_team_name = self._format_team_display_name(data.get("winner_team_name"))
        year = self._format_deterministic_metric(data.get("year"))
        if winner_team_name == "확인 불가" or year == "확인 불가":
            return None

        final_game = data.get("final_game") or {}
        if not isinstance(final_game, dict):
            final_game = {}

        stadium = self._format_deterministic_metric(final_game.get("stadium"))
        game_date = self._format_deterministic_metric(
            final_game.get("game_date") or data.get("final_date")
        )

        away_team = self._format_team_display_name(
            final_game.get("away_team_name") or final_game.get("away_team")
        )
        home_team = self._format_team_display_name(
            final_game.get("home_team_name") or final_game.get("home_team")
        )
        away_score = final_game.get("away_score")
        home_score = final_game.get("home_score")
        winner_rank = data.get("winner_rank")

        intro = f"{year}년 한국시리즈 우승팀은 {winner_team_name}입니다."
        if isinstance(winner_rank, int):
            intro = (
                f"{year}년 한국시리즈 우승팀은 {winner_team_name}이고, "
                f"정규시즌 {winner_rank}위 팀이었습니다."
            )

        if (
            away_team != "확인 불가"
            and home_team != "확인 불가"
            and isinstance(away_score, (int, float))
            and isinstance(home_score, (int, float))
        ):
            detail = (
                f"마지막 경기는 {game_date} {stadium}에서 열렸고, "
                f"{away_team}와 {home_team}가 맞붙어 {away_score}-{home_score}로 끝났습니다."
            )
            if winner_team_name in {away_team, home_team}:
                if winner_team_name == away_team and away_score > home_score:
                    detail = (
                        f"마지막 경기는 {game_date} {stadium}에서 열렸고, "
                        f"{winner_team_name}가 {home_team}를 {away_score}-{home_score}로 꺾고 우승을 확정했습니다."
                    )
                elif winner_team_name == home_team and home_score > away_score:
                    detail = (
                        f"마지막 경기는 {game_date} {stadium}에서 열렸고, "
                        f"{winner_team_name}가 {away_team}를 {home_score}-{away_score}로 꺾고 우승을 확정했습니다."
                    )
            return f"{intro} {detail}"

        if game_date != "확인 불가":
            if stadium != "확인 불가":
                return f"{intro} 우승을 확정한 마지막 경기는 {game_date} {stadium}에서 열렸습니다."
            return f"{intro} 우승을 확정한 마지막 경기는 {game_date}였습니다."

        return intro

    def _build_award_chat_answer(self, data: Dict[str, Any]) -> Optional[str]:
        awards = data.get("awards")
        if not isinstance(awards, list) or not awards:
            return None

        year = self._format_deterministic_metric(data.get("year"))
        requested_type = data.get("award_type")

        if requested_type and requested_type != "any" and len(awards) == 1:
            winner = awards[0]
            team_text = f" ({winner['team_name']})" if winner.get("team_name") else ""
            return (
                f"{year}년 {self._display_award_type(requested_type)} 수상자는 "
                f"{winner['player_name']}{team_text}입니다."
            )

        lead = awards[0]
        return (
            f"{year}년 수상 기록은 {len(awards)}건 확인됐고, "
            f"대표적으로 {self._display_award_type(lead.get('award_type'))}는 "
            f"{self._format_deterministic_metric(lead.get('player_name'))}입니다."
        )

    def _build_chat_reference_answer(
        self, query: str, tool_results: List[ToolResult]
    ) -> Optional[str]:
        bundle_answer = self._build_bundle_reference_answer(
            query,
            tool_results,
            chat_mode=True,
        )
        if bundle_answer:
            return bundle_answer

        comparison_answer = self._build_player_comparison_answer(
            query,
            tool_results,
            chat_mode=True,
        )
        if comparison_answer:
            return comparison_answer

        multi_player_answer = self._build_multi_player_narrative_answer(
            query,
            tool_results,
            chat_mode=True,
        )
        if multi_player_answer:
            logger.info(
                "[BaseballAgent] Using multi-player narrative renderer query=%s",
                query[:120],
            )
            return multi_player_answer
        if self._should_defer_incomplete_multi_player_answer(query, tool_results):
            return None

        decision = self._resolve_chat_intent(query, extract_entities_from_query(query))
        registry_answer = self.chat_renderer_registry.render_reference(
            query, tool_results, decision
        )
        if registry_answer:
            return registry_answer

        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue

            data = result.data

            if "batting_stats" in data or "pitching_stats" in data:
                answer = self._build_player_stats_chat_answer(data)
                if answer:
                    return answer

            if "awards" in data:
                answer = self._build_award_chat_answer(data)
                if answer:
                    return answer

            if "leaderboard" in data:
                answer = self._build_leaderboard_chat_answer(query, data)
                if answer:
                    return answer

            if "regulations" in data:
                answer = self._build_regulation_chat_answer(query, data)
                if answer:
                    return answer

            if "games" in data and "date" in data:
                answer = self._build_games_by_date_chat_answer(data)
                if answer:
                    return answer

            if "team_name" in data and (
                data.get("team_rank") is not None or data.get("season_rank") is not None
            ):
                answer = self._build_team_rank_chat_answer(data)
                if answer:
                    return answer

            if data.get("found") is False:
                continue

            if "winner_team_name" in data and "series_type" in data:
                answer = self._build_korean_series_winner_chat_answer(data)
                if answer:
                    return answer

            if "last_game_date" in data or "final_date" in data:
                answer = self._build_team_last_game_date_chat_answer(data)
                if answer:
                    return answer

        return None

    def _build_chat_low_data_answer(
        self, query: str, tool_results: List[ToolResult]
    ) -> str:
        decision = self._resolve_chat_intent(query, extract_entities_from_query(query))
        return self.chat_renderer_registry.render_low_data(
            query, tool_results, decision
        )

    def _build_structured_deterministic_answer(
        self, query: str, tool_results: List[ToolResult]
    ) -> Optional[str]:
        bundle_answer = self._build_bundle_reference_answer(
            query,
            tool_results,
            chat_mode=False,
        )
        if bundle_answer:
            return bundle_answer

        comparison_answer = self._build_player_comparison_answer(
            query,
            tool_results,
            chat_mode=False,
        )
        if comparison_answer:
            return comparison_answer

        multi_player_answer = self._build_multi_player_narrative_answer(
            query,
            tool_results,
            chat_mode=False,
        )
        if multi_player_answer:
            return multi_player_answer
        if self._should_defer_incomplete_multi_player_answer(query, tool_results):
            return None

        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue

            data = result.data

            if "batting_stats" in data or "pitching_stats" in data:
                answer = self._build_player_stats_answer(data)
                if answer:
                    return answer

            if "awards" in data:
                answer = self._build_award_answer(data)
                if answer:
                    return answer

            if "leaderboard" in data:
                answer = self._build_leaderboard_answer(data)
                if answer:
                    return answer

            if "regulations" in data:
                answer = self._build_regulation_answer(data)
                if answer:
                    return answer

            if "games" in data and "date" in data:
                answer = self._build_games_by_date_answer(data)
                if answer:
                    return answer

            if "team_name" in data and (
                data.get("team_rank") is not None or data.get("season_rank") is not None
            ):
                answer = self._build_team_rank_answer(data)
                if answer:
                    return answer

            if data.get("found") is False:
                continue

            if "last_game_date" in data:
                answer = self._build_team_last_game_date_answer(data)
                if answer:
                    return answer

        return None

    def _build_answer_generation_recovery_answer(
        self,
        query: str,
        tool_results: List[ToolResult],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        persona = context.get("persona") if isinstance(context, dict) else None
        planner_mode = (
            str(context.get("planner_mode", "")) if isinstance(context, dict) else ""
        )

        if planner_mode == "fast_path":
            fast_path_answer = self._build_fast_path_answer(
                query,
                tool_results,
                chat_mode=(persona == "chat"),
            )
            if fast_path_answer:
                if persona == "chat":
                    fast_path_answer = self._normalize_chatbot_answer_text(
                        fast_path_answer
                    )
                return {"answer": fast_path_answer, "verified": True}

        if persona == "chat":
            chat_reference_answer = self._build_chat_reference_answer(
                query, tool_results
            )
            if chat_reference_answer:
                return {"answer": chat_reference_answer, "verified": True}

        deterministic_answer = self._build_structured_deterministic_answer(
            query, tool_results
        )
        if deterministic_answer:
            if persona == "chat":
                deterministic_answer = self._normalize_chatbot_answer_text(
                    deterministic_answer
                )
            return {"answer": deterministic_answer, "verified": True}

        if not self._has_meaningful_tool_results(tool_results):
            if persona == "chat":
                return {
                    "answer": self._build_chat_low_data_answer(query, tool_results),
                    "verified": False,
                }
            return {
                "answer": (
                    "현재 연결된 자료만으로는 질문에 대해 확인된 답을 만들지 못했습니다.\n\n"
                    "추정하지 않고 확인된 범위에서만 말씀드리겠습니다."
                ),
                "verified": False,
            }

        return None

    async def _generate_verified_answer(
        self,
        query: str,
        tool_results: List[ToolResult],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        도구 실행 결과를 바탕으로 검증된 답변을 생성합니다.
        """
        logger.info(
            f"[BaseballAgent] Generating verified answer with {len(tool_results)} tool results"
        )

        # 시간 컨텍스트 생성
        now = datetime.now()
        current_year = now.year
        is_team_query = self._is_team_analysis_query(
            query, extract_entities_from_query(query)
        )
        planner_mode = (
            str(context.get("planner_mode", "")) if isinstance(context, dict) else ""
        )
        grounding_mode = (
            str(context.get("grounding_mode", "")) if isinstance(context, dict) else ""
        )
        explicit_source_tier = (
            str(context.get("source_tier", "")) if isinstance(context, dict) else ""
        )
        as_of_date = (
            str(context.get("as_of_date", "")) if isinstance(context, dict) else ""
        )
        is_fast_path_answer = planner_mode == "fast_path"
        request_mode = (
            str(context.get("request_mode", "stream"))
            if isinstance(context, dict)
            else "stream"
        )
        source_tier = self._source_tier_from_tool_results(
            tool_results, explicit_source_tier or None
        )
        if not grounding_mode:
            if source_tier == "web":
                grounding_mode = "latest_info"
            elif source_tier == "docs":
                grounding_mode = "baseball_explainer"
            else:
                grounding_mode = "structured_kbo"
        if not as_of_date:
            as_of_date = self._resolve_as_of_date(tool_results)

        time_context = ""
        if "재작년" in query:
            actual_year = current_year - 2
            time_context = f"\n\n**중요**: 사용자가 '재작년'이라고 했고, 현재가 {current_year}년이므로 조회된 데이터는 {actual_year}년입니다. 답변할 때 '{actual_year}년'으로 명시하세요."
        elif "작년" in query:
            actual_year = current_year - 1
            time_context = f"\n\n**중요**: 사용자가 '작년'이라고 했고, 현재가 {current_year}년이므로 조회된 데이터는 {actual_year}년입니다. 답변할 때 '{actual_year}년'으로 명시하세요."
        elif "올해" in query:
            time_context = f"\n\n**중요**: 사용자가 '올해'라고 했고, 현재는 {current_year}년입니다."

        # 도구 실행 결과를 텍스트로 변환
        tool_data_summary = []
        data_sources = []
        failed_tool_messages: List[str] = []

        for i, result in enumerate(tool_results):
            # result는 이제 dict 형태 (ToolResult.to_dict() 결과)
            if result.success:
                tool_data_summary.append(f"도구 {i + 1} 결과: {result.message}")
                result_data = result.data if result.data else {}
                try:
                    prompt_data = (
                        self._summarize_team_tool_data_for_prompt(result_data)
                        if is_team_query
                        else result_data
                    )
                    if is_team_query:
                        prompt_data = self._optimize_team_prompt_payload(prompt_data)
                    else:
                        prompt_data = self._prepare_tool_data_for_answer_prompt(
                            query,
                            prompt_data,
                            grounding_mode=grounding_mode,
                            source_tier=source_tier,
                        )
                    data_json = self._serialize_tool_data_for_prompt(prompt_data)
                    tool_data_summary.append(f"데이터: {data_json}")
                except Exception as e:
                    logger.error(f"[BaseballAgent] JSON serialization error: {e}")
                    tool_data_summary.append(f"데이터: (직렬화 실패)")

                found_flag = (
                    result_data.get("found") if isinstance(result_data, dict) else None
                )
                data_sources.append(
                    {
                        "tool": (
                            result_data.get("source", "database")
                            if isinstance(result_data, dict)
                            else "database"
                        ),
                        "verified": True,
                        "data_points": (
                            len(result_data)
                            if isinstance(result_data, list)
                            else (0 if found_flag is False else 1)
                        ),
                    }
                )
            else:
                failed_tool_messages.append(str(result.message))
                data_sources.append(
                    {"tool": "failed", "verified": False, "error": result.message}
                )

        if failed_tool_messages:
            sampled_failures = "; ".join(msg[:120] for msg in failed_tool_messages[:2])
            tool_data_summary.append(
                f"참고: 일부 도구 조회가 실패했습니다(총 {len(failed_tool_messages)}건). 실패 요약: {sampled_failures}"
            )

        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue
            result_data = result.data
            metrics = result_data.get("metrics") or {}
            if not metrics or not result_data.get("team_name"):
                continue
            direct_metric_answer = self._build_team_metric_fast_path_answer(
                query,
                result_data.get("team_name"),
                result_data.get("year"),
                metrics.get("batting") or {},
                metrics.get("pitching") or {},
                result_data.get("rankings") or {},
            )
            if direct_metric_answer:
                return {
                    "answer": direct_metric_answer,
                    "verified": True,
                    "data_sources": data_sources,
                    "error": None,
                }

        direct_explainer_answer = self._build_known_explainer_answer(
            query,
            grounding_mode=grounding_mode,
            source_tier=source_tier,
        )
        if direct_explainer_answer:
            return {
                "answer": direct_explainer_answer,
                "verified": True,
                "data_sources": data_sources,
                "error": None,
            }

        direct_latest_answer = self._build_known_latest_answer(
            query,
            grounding_mode=grounding_mode,
            source_tier=source_tier,
            as_of_date=as_of_date,
        )
        if direct_latest_answer:
            return {
                "answer": direct_latest_answer,
                "verified": True,
                "data_sources": data_sources,
                "error": None,
            }

        # 프롬프트 선택 로직
        persona = context.get("persona") if context else None
        prompt_override = context.get("prompt_override") if context else None

        tool_data_text = chr(10).join(tool_data_summary)
        if is_fast_path_answer:
            team_context_cap = 560 if request_mode == "completion" else 420
        else:
            team_context_cap = 850 if request_mode == "completion" else 720
        if is_team_query and len(tool_data_text) > team_context_cap:
            tool_data_text = (
                f"{tool_data_text[:team_context_cap]}...(team-context-truncated)"
            )

        if prompt_override:
            # 프롬프트 오버라이드 사용
            answer_prompt = prompt_override.format(
                question=f"{query}{time_context}", context=tool_data_text
            )
        elif persona == "coach":
            # 코치 페르소나 사용
            answer_prompt = COACH_PROMPT.format(
                question=f"{query}{time_context}", context=tool_data_text
            )
        elif grounding_mode == "latest_info" or source_tier == "web":
            answer_prompt = LATEST_ANSWER_PROMPT.format(
                question=f"{query}{time_context}",
                context=tool_data_text,
                as_of_date=as_of_date,
            )
        elif grounding_mode in {"baseball_explainer", "long_tail_entity"}:
            answer_prompt = EXPLAINER_ANSWER_PROMPT.format(
                question=f"{query}{time_context}",
                context=tool_data_text,
            )
        elif persona == "chat":
            answer_prompt = (
                "당신은 KBO 챗봇 BEGA입니다.\n"
                "아래 검증된 검색 결과만 사용해 자연스럽고 짧은 채팅처럼 답하세요.\n"
                "제목, 섹션명, 표, '출처:' 문장, 브리핑체 표현은 쓰지 마세요.\n"
                "없는 정보는 솔직하게 확인이 어렵다고 말하고, 아는 척하지 마세요.\n"
                "답변은 2~5문장 안팎으로 간결하게 마무리하세요.\n"
                "팬이 묻는 말투에 맞춰 부드럽고 직설적으로 답하세요.\n\n"
                "### 사용자 질문\n"
                f"{query}{time_context}\n\n"
                "### 검증된 검색 결과\n"
                f"{tool_data_text}"
            )
        elif is_fast_path_answer:
            answer_prompt = (
                "당신은 KBO 분석가 BEGA입니다.\n"
                "아래 DB 결과만 사용해 짧고 정확하게 답하세요.\n"
                "없는 정보는 '확인 불가'라고 쓰세요.\n"
                "출력 형식은 반드시 아래 순서를 지키세요.\n"
                "## 요약\n"
                "- 결론 1~2문장\n"
                "## 상세 내역\n"
                "- bullet 2~3개\n"
                "## 핵심 지표\n"
                "- 표 1개 필수, 최대 3행\n"
                "## 인사이트\n"
                "- bullet 2개\n"
                "- 마지막 줄은 반드시 `출처: DB 조회 결과`\n"
                "길이는 430자 안팎으로 짧게 유지하세요.\n\n"
                "### 사용자 질문\n"
                f"{query}{time_context}\n\n"
                "### DB 검색 결과\n"
                f"{tool_data_text}"
            )
        elif is_team_query:
            answer_prompt = (
                "당신은 KBO 전문 분석가 BEGA입니다.\n"
                "아래 DB 결과만 사용해 정확하고 읽기 쉬운 답변을 작성하세요.\n"
                "DB 결과에 없는 수치/선수/순위는 추정하지 말고 '확인 불가'로 표시하세요.\n"
                "출력 형식은 아래 순서를 반드시 지키세요:\n"
                "## 요약\n"
                "- 질문에 대한 직접 결론 1~2문장 (팀명/연도/핵심 수치 포함)\n\n"
                "## 상세 내역\n"
                "- 핵심 근거 bullet 2~4개\n\n"
                "## 핵심 지표\n"
                "- 마크다운 표 1개 이상 필수\n"
                "- 표 컬럼 예시: 항목 | 수치 | 해석\n\n"
                "## 인사이트\n"
                "- 데이터 해석 bullet 2~4개\n"
                "- 마지막 bullet은 다음 경기 관전 포인트 1개\n\n"
                "## 길이 제한\n"
                "- 답변은 650자 내외, bullet 최대 5개, 표는 최대 4행으로 간결하게 작성\n\n"
                "### 사용자 질문\n"
                f"{query}{time_context}\n\n"
                "### DB 검색 결과\n"
                f"{tool_data_text}"
            )
        else:
            # 기본 BEGA 페르소나 (기존 로직 유지)
            answer_prompt = DEFAULT_ANSWER_PROMPT.format(
                question=f"{query}{time_context}", context=tool_data_text
            )

        has_meaningful_data = self._has_meaningful_tool_results(tool_results)
        if not has_meaningful_data:
            logger.info(
                "[AnswerQuality] insufficient_data_fallback query=%s tool_count=%d",
                query[:120],
                len(tool_results),
            )
            query_lower = query.lower()
            team_tokens = [
                "lg",
                "엘지",
                "트윈스",
                "kt",
                "케이티",
                "위즈",
                "ssg",
                "랜더스",
                "기아",
                "kia",
                "타이거즈",
                "두산",
                "베어스",
                "롯데",
                "자이언츠",
                "한화",
                "이글스",
                "삼성",
                "라이온즈",
                "nc",
                "엔씨",
                "다이노스",
                "키움",
                "히어로즈",
            ]
            contains_team_token = any(token in query_lower for token in team_tokens)

            summary_line = (
                "지금 잡힌 기록만으로 여기서 딱 잘라 말하면 좀 무리입니다.\n"
                "괜히 아는 척하지 않고, 확인된 것만 팬 눈높이로 정리해드릴게요."
            )
            detail_lines = [
                "- 조회는 돌렸는데, 지금 손에 잡힌 값만으로는 답을 시원하게 못 박기 어렵습니다.",
                "- 팀, 선수, 시즌 중 하나만 더 콕 집어주시면 답변이 훨씬 또렷해집니다.",
            ]
            table_rows = [
                ("질문 유형", "일반 조회"),
                ("조회 도구 수", str(len(tool_results))),
                ("유효 데이터 건수", "0"),
                ("다음 액션", "시점 또는 대상을 더 구체화"),
            ]
            insight_lines = [
                "- 질문을 `대상 + 시즌 + 지표`로 던지면 바로 써먹을 만한 답이 나올 확률이 높습니다.",
                "- 같은 질문도 잠깐 뒤에 다시 보내면 조회가 풀리는 경우가 있습니다.",
            ]

            if any(
                keyword in query_lower
                for keyword in [
                    "규정",
                    "규칙",
                    "fa",
                    "드래프트",
                    "엔트리",
                    "로스터",
                    "판정",
                ]
            ):
                summary_line = (
                    "규정은 설명할 수 있는데, 지금 잡힌 조항만으로 핵심을 딱 잘라 말하기엔 살짝 모자랍니다.\n"
                    "헷갈릴 만한 부분은 빼고, 확인된 선에서만 정리해드릴게요."
                )
                detail_lines = [
                    "- 규정 검색은 들어갔지만, 질문 범위가 넓어서 바로 핵심으로 꽂히는 조항이 덜 모였습니다.",
                    "- `FA 보상선수`, `등록일수`, `1군 엔트리`처럼 주제를 좁혀주시면 훨씬 또렷하게 풀어드릴 수 있습니다.",
                ]
                table_rows = [
                    ("질문 유형", "규정/제도"),
                    ("조회 도구 수", str(len(tool_results))),
                    ("확인 상태", "규정 근거 부족"),
                    ("추천 재질문", "예: FA 보상선수 규정 알려줘"),
                ]
                insight_lines = [
                    "- 규정 질문은 키워드가 넓으면 안 봐도 될 조항까지 같이 딸려옵니다.",
                    "- 제도명 하나만 정확히 찍어주셔도 답변 선명도가 확 올라갑니다.",
                ]
            elif any(
                keyword in query_lower
                for keyword in ["마지막 경기", "최근 경기 언제", "최종전"]
            ):
                summary_line = (
                    "현재 연결된 자료에서는 마지막 경기 날짜를 확정하지 못했습니다.\n"
                    "확인되지 않은 날짜를 추정해서 답하지는 않겠습니다."
                )
                detail_lines = [
                    "- 마지막 경기 조회는 수행했지만, 시즌 범위나 경기 유형까지 확정할 값이 비어 있었습니다.",
                    "- 현재 자료만으로는 정규시즌/포스트시즌 여부를 함께 확정하기 어렵습니다.",
                ]
                table_rows = [
                    ("질문 유형", "팀 마지막 경기"),
                    ("조회 도구 수", str(len(tool_results))),
                    ("확인 상태", "마지막 경기 기록 부족"),
                    ("추천 재질문", "예: 2025 LG 마지막 경기 날짜"),
                ]
                insight_lines = [
                    "- 마지막 경기 질문은 정규시즌인지 포스트시즌인지 빠지면 빈값이 나오기 쉽습니다.",
                    "- 시즌 연도와 경기 범위를 같이 주는 방식이 제일 안정적입니다.",
                ]
            elif any(
                keyword in query_lower
                for keyword in ["오늘", "일정", "중계", "몇 시", "몇시"]
            ):
                summary_line = (
                    "현재 연결된 일정 자료에서는 질문에 맞는 경기 정보를 확정하지 못했습니다.\n"
                    "없는 경기를 있는 것처럼 답하지는 않겠습니다."
                )
                detail_lines = [
                    "- 날짜 기준 일정 조회는 수행했지만, 질문에 맞는 경기 데이터가 충분히 모이지 않았습니다.",
                    "- 현재 자료만으로는 경기 존재 여부와 시각을 함께 확정하기 어렵습니다.",
                ]
                table_rows = [
                    ("질문 유형", "경기 일정"),
                    ("조회 도구 수", str(len(tool_results))),
                    ("확인 상태", "일정 데이터 부족"),
                    ("추천 재질문", "예: 오늘 LG 경기 일정 알려줘"),
                ]
                insight_lines = [
                    "- 일정 질문은 날짜만 있어도 되지만 팀까지 같이 주면 잡음이 확 줄어듭니다.",
                    "- 비시즌이거나 경기 없는 날이면 빈 결과가 정상일 수 있습니다.",
                ]
            elif contains_team_token and any(
                keyword in query_lower
                for keyword in ["순위", "몇 위", "몇위", "승률", "승패"]
            ):
                summary_line = (
                    "당장 몇 위인지 딱 찍고 싶은데, 지금 붙은 순위표로는 그 말을 자신 있게 못 하겠습니다.\n"
                    "괜히 헛짚지 않고 확인된 범위만 말씀드릴게요."
                )
                detail_lines = [
                    "- 팀 순위 조회는 수행됐지만, 해당 시즌 순위 레코드가 비어 있거나 덜 채워진 상태였습니다.",
                    "- `2025 LG 순위 알려줘`처럼 시즌 연도를 같이 주시면 같은 질문도 훨씬 안정적으로 답할 수 있습니다.",
                ]
                table_rows = [
                    ("질문 유형", "팀 순위"),
                    ("조회 도구 수", str(len(tool_results))),
                    ("확인 상태", "순위 데이터 부족"),
                    ("추천 재질문", "예: 2025 LG 순위 알려줘"),
                ]
                insight_lines = [
                    "- 개막 직후나 비시즌에는 순위표가 비어 있는 경우가 제법 있습니다.",
                    "- 팀 질문은 시즌 연도 하나 붙여주는 것만으로 품질 차이가 크게 납니다.",
                ]
            elif not contains_team_token and any(
                keyword in query_lower
                for keyword in [
                    "리더보드",
                    "랭킹",
                    "홈런",
                    "타율",
                    "ops",
                    "era",
                    "세이브",
                    "홀드",
                    "탈삼진",
                ]
            ):
                summary_line = (
                    "순위표 바로 펼치고 싶은데, 지금 모인 값만으로는 랭킹을 자신 있게 세우기 어렵습니다.\n"
                    "빈칸을 추정으로 메우지 않고 확인된 것만 남기겠습니다."
                )
                detail_lines = [
                    "- 리더보드 조회는 들어갔지만, 질문 지표에 맞는 유효 순위 데이터가 충분히 모이지 않았습니다.",
                    "- `2025 홈런 랭킹`, `2025 평균자책 순위`처럼 연도와 지표를 같이 주시면 바로 또렷해집니다.",
                ]
                table_rows = [
                    ("질문 유형", "리더보드"),
                    ("조회 도구 수", str(len(tool_results))),
                    ("확인 상태", "순위표 데이터 부족"),
                    ("추천 재질문", "예: 2025 홈런 랭킹 보여줘"),
                ]
                insight_lines = [
                    "- 리더보드는 연도와 타격/투수 지표가 같이 있어야 정확도가 안정적입니다.",
                    "- 지표 이름을 구체적으로 적을수록 바로 써먹을 수 있는 답으로 바뀝니다.",
                ]
            elif any(
                keyword in query_lower
                for keyword in [
                    "성적",
                    "기록",
                    "타율",
                    "ops",
                    "홈런",
                    "타점",
                    "era",
                    "whip",
                    "세이브",
                    "홀드",
                ]
            ):
                summary_line = (
                    "선수 기록 바로 꺼내고 싶은데, 이번 조회에선 선수명이나 시즌 매칭이 흐릿합니다.\n"
                    "없는 기록을 지어내진 않고 확인된 범위만 말씀드릴게요."
                )
                detail_lines = [
                    "- 선수 기록 조회는 수행됐지만, 선수명 매칭이나 시즌 데이터가 충분히 잡히지 않았습니다.",
                    "- `오스틴 2025 홈런 몇 개야?`, `문보경 2025 OPS 알려줘`처럼 시즌과 지표를 같이 주시면 정확도가 더 올라갑니다.",
                ]
                table_rows = [
                    ("질문 유형", "선수 기록"),
                    ("조회 도구 수", str(len(tool_results))),
                    ("확인 상태", "선수/시즌 데이터 부족"),
                    ("추천 재질문", "예: 오스틴 2025 기록 알려줘"),
                ]
                insight_lines = [
                    "- 선수 질문은 동명이인, 영문명, 시즌 누락 때문에 빈 결과가 자주 납니다.",
                    "- 선수명 + 시즌 + 지표 조합이 제일 안정적인 질문 형태입니다.",
                ]

            table_text = "\n".join(
                f"| {label} | {value} |" for label, value in table_rows
            )
            low_data_answer = (
                "## 요약\n"
                f"{summary_line}\n\n"
                "## 상세 내역\n" + "\n".join(detail_lines) + "\n\n## 핵심 지표\n"
                "| 항목 | 상태 |\n"
                "| --- | --- |\n"
                f"{table_text}\n\n"
                "## 인사이트\n" + "\n".join(insight_lines) + "\n\n출처: DB 조회 결과"
            )
            if persona == "chat":
                low_data_answer = self._build_chat_low_data_answer(query, tool_results)
            return {
                "answer": low_data_answer,
                "verified": False,
                "data_sources": data_sources,
                "error": None,
            }

        if is_fast_path_answer:
            fast_path_answer = self._build_fast_path_answer(
                query,
                tool_results,
                chat_mode=(persona == "chat"),
            )
            if fast_path_answer:
                if persona == "chat" and (
                    "## " in fast_path_answer or "출처:" in fast_path_answer
                ):
                    fast_path_answer = self._normalize_chatbot_answer_text(
                        fast_path_answer
                    )
                logger.info(
                    "[BaseballAgent] Using deterministic fast-path answer renderer query=%s",
                    query[:120],
                )
                return {
                    "answer": fast_path_answer,
                    "verified": True,
                    "data_sources": data_sources,
                    "error": None,
                }

        deterministic_answer = self._build_structured_deterministic_answer(
            query, tool_results
        )
        if persona == "chat":
            chat_reference_answer = self._build_chat_reference_answer(
                query, tool_results
            )
            if chat_reference_answer:
                logger.info(
                    "[BaseballAgent] Using chat reference answer renderer query=%s planner_mode=%s",
                    query[:120],
                    planner_mode or "unknown",
                )
                return {
                    "answer": chat_reference_answer,
                    "verified": True,
                    "data_sources": data_sources,
                    "error": None,
                }
        if deterministic_answer:
            if persona == "chat":
                deterministic_answer = self._normalize_chatbot_answer_text(
                    deterministic_answer
                )
            logger.info(
                "[BaseballAgent] Using structured deterministic answer renderer query=%s planner_mode=%s",
                query[:120],
                planner_mode or "unknown",
            )
            return {
                "answer": deterministic_answer,
                "verified": True,
                "data_sources": data_sources,
                "error": None,
            }

        try:
            # 검증된 데이터 기반 답변 생성
            answer_prompt_preview = (
                answer_prompt
                if len(answer_prompt) <= 1200
                else f"{answer_prompt[:1200]}...(truncated)"
            )
            logger.info(
                "[BaseballAgent] Final Answer Prompt (preview): %s",
                answer_prompt_preview,
            )
            answer_messages = [{"role": "user", "content": answer_prompt}]
            answer_max_tokens = self._resolve_answer_max_tokens(query, tool_results)
            if (
                self.chat_dynamic_token_enabled
                and is_team_query
                and isinstance(answer_max_tokens, int)
            ):
                team_cap = self._settings_int("chat_team_answer_cap_base", 520)
                if len(tool_results) > 2:
                    team_cap = self._settings_int("chat_team_answer_cap_heavy", 650)
                compact_query = query.replace(" ", "")
                brief_query = len(compact_query) <= 24
                low_complexity_markers = (
                    "요약",
                    "흐름",
                    "상태",
                    "분위기",
                    "폼",
                    "부진",
                    "좋아",
                )
                high_complexity_markers = (
                    "비교",
                    "세부",
                    "상세",
                    "리포트",
                    "심층",
                    "자세히",
                )
                if is_fast_path_answer:
                    if request_mode == "stream":
                        team_cap = self._settings_int(
                            "chat_team_answer_cap_fast_path_stream", 360
                        )
                    else:
                        team_cap = self._settings_int(
                            "chat_team_answer_cap_fast_path_completion", 460
                        )
                if brief_query and any(m in query for m in low_complexity_markers):
                    team_cap = min(
                        team_cap,
                        self._settings_int("chat_team_answer_cap_brief", 460),
                    )
                if any(m in query for m in high_complexity_markers):
                    team_cap = max(
                        team_cap,
                        self._settings_int("chat_team_answer_cap_high_complexity", 600),
                    )
                if request_mode == "stream":
                    team_cap = min(
                        team_cap,
                        self._settings_int("chat_team_answer_cap_stream", 420),
                    )
                if is_fast_path_answer and request_mode == "stream":
                    team_cap = min(
                        team_cap,
                        self._settings_int(
                            "chat_team_answer_cap_fast_path_stream", 360
                        ),
                    )
                if is_fast_path_answer and request_mode == "completion":
                    team_cap = min(
                        team_cap,
                        self._settings_int(
                            "chat_team_answer_cap_fast_path_completion", 460
                        ),
                    )
                answer_max_tokens = min(answer_max_tokens, team_cap)
            if (
                self.chat_dynamic_token_enabled
                and request_mode == "completion"
                and isinstance(answer_max_tokens, int)
            ):
                completion_cap = self._settings_int(
                    (
                        "chat_completion_answer_cap_team"
                        if is_team_query
                        else "chat_completion_answer_cap_general"
                    ),
                    880 if is_team_query else 1200,
                )
                answer_max_tokens = min(answer_max_tokens, completion_cap)
            logger.info(
                "[AnswerBudget] max_tokens=%s dynamic=%s query_len=%d tool_count=%d team_query=%s mode=%s",
                answer_max_tokens,
                self.chat_dynamic_token_enabled,
                len(query),
                len(tool_results),
                is_team_query,
                request_mode,
            )
            # 스트리밍을 위해 await 제거하고 제너레이터 반환
            answer = self.llm_generator(answer_messages, max_tokens=answer_max_tokens)

            # 의미 있는 데이터가 확인된 경우만 verified 처리
            has_verified_data = has_meaningful_data

            return {
                "answer": answer,
                "verified": has_verified_data,
                "data_sources": data_sources,
                "error": None,
            }

        except Exception as e:
            logger.error(f"[BaseballAgent] Error generating verified answer: {e}")
            error_answer = (
                "## 요약\n"
                "방금 답변 생성 타이밍이 흔들려서 정식 분석 문장을 완성하지 못했습니다.\n\n"
                "## 상세 내역\n"
                "- 내부 생성 단계에서 일시 오류가 발생했습니다.\n"
                "- DB 조회 파이프라인과는 분리된 생성 단계 이슈일 수 있습니다.\n\n"
                "## 핵심 지표\n"
                "| 항목 | 상태 |\n"
                "| --- | --- |\n"
                "| 답변 생성 | 일시 오류 |\n"
                "| 재시도 권장 | 예 |\n\n"
                "## 인사이트\n"
                "- 동일 질문 재요청 시 정상 복구되는 경우가 많습니다.\n"
                "- 반복 시 모델/네트워크 상태를 점검하겠습니다.\n"
            )
            if persona == "chat":
                error_answer = (
                    "방금 답변 문장을 만드는 타이밍이 잠깐 꼬였습니다. "
                    "같은 질문을 한 번 더 보내주시면 바로 다시 이어서 볼게요."
                )
            return {
                "answer": error_answer,
                "verified": False,
                "data_sources": [],
                "error": f"답변 생성 오류: {e}",
            }

    def _tool_get_player_wpa_leaders(
        self, year: int = None, limit: int = 10, team_name: str = None
    ) -> ToolResult:
        """WPA 순위 조회 도구 wrapper"""
        if year is None:
            import datetime

            year = datetime.datetime.now().year

        try:
            result = self.db_query_tool.get_player_wpa_leaders(
                year=year, limit=limit, team_name=team_name
            )
            return ToolResult(
                success=True,
                data=result,
                message=f"{year}년 WPA 순위를 성공적으로 조회했습니다.",
            )
        except Exception as e:
            logger.error(f"Error in _tool_get_player_wpa_leaders: {e}")
            return ToolResult(
                success=False, data={}, message=f"WPA 순위 조회 실패: {e}"
            )

    def _tool_get_clutch_moments(self, game_id: str, limit: int = 5) -> ToolResult:
        """승부처 조회 도구 wrapper"""
        try:
            result = self.db_query_tool.get_clutch_moments(game_id=game_id, limit=limit)
            return ToolResult(
                success=True,
                data=result,
                message=f"경기 {game_id}의 결정적 순간들을 성공적으로 조회했습니다.",
            )
        except Exception as e:
            logger.error(f"Error in _tool_get_clutch_moments: {e}")
            return ToolResult(success=False, data={}, message=f"승부처 조회 실패: {e}")

    def _tool_get_player_wpa_stats(
        self, player_name: str, year: int = None
    ) -> ToolResult:
        """선수별 WPA 통계 조회 도구 wrapper"""
        if year is None:
            import datetime

            year = datetime.datetime.now().year

        try:
            result = self.db_query_tool.get_player_wpa_stats(
                player_name=player_name, year=year
            )
            return ToolResult(
                success=True,
                data=result,
                message=f"{year}년 {player_name} 선수의 WPA 통계를 성공적으로 조회했습니다.",
            )
        except Exception as e:
            logger.error(f"Error in _tool_get_player_wpa_stats: {e}")
            return ToolResult(
                success=False, data={}, message=f"선수 WPA 통계 조회 실패: {e}"
            )
