import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.agents.baseball_agent import BaseballStatisticsAgent, SERIAL_DB_TOOL_NAMES
from app.agents.chat_intent_router import ChatIntent
from app.agents.tool_caller import ToolCall, ToolResult
from app.core.entity_extractor import extract_entities_from_query, extract_player_names


def _build_agent() -> BaseballStatisticsAgent:
    agent = BaseballStatisticsAgent.__new__(BaseballStatisticsAgent)
    agent.settings = None
    agent.fast_path_tool_cap = 2
    agent.chat_dynamic_token_enabled = True
    agent.chat_analysis_max_tokens = 350
    agent.chat_planner_timeout_seconds = 30.0
    agent.chat_compact_planner_timeout_seconds = 24.0
    agent.chat_first_token_watchdog_seconds = 20.0
    agent.chat_first_token_retry_max_attempts = 1
    agent.chat_stream_first_token_watchdog_seconds = 10.0
    agent.chat_stream_first_token_retry_max_attempts = 0
    agent.chat_tool_parallel_enabled = True
    agent.chat_tool_parallel_split_batch_enabled = True
    agent.chat_tool_parallel_serial_tools = set(SERIAL_DB_TOOL_NAMES)
    agent.chat_tool_parallel_max_concurrency = 2
    agent.chat_planner_cache_ttl_seconds = 60
    agent.chat_planner_cache_max_entries = 512
    agent._is_team_analysis_query = (
        lambda query, entity_filter: bool(getattr(entity_filter, "team_id", None))
        and "분석" in query
    )
    return agent


def test_select_llm_planner_prompt_uses_team_mode_for_team_analysis() -> None:
    agent = _build_agent()
    query = "LG 팀 분석해줘"
    entity_filter = extract_entities_from_query(query)

    prompt, planner_mode = agent._select_llm_planner_prompt(
        query_text=query,
        query=query,
        entity_filter=entity_filter,
        current_date="2026년 03월 12일",
        current_year=2026,
        entity_context="",
    )

    assert planner_mode == "team_llm_planner"
    assert (
        "허용 도구: get_team_summary, get_team_advanced_metrics, get_team_rank, get_team_last_game"
        in prompt
    )


def test_select_llm_planner_prompt_uses_player_mode_for_player_analysis() -> None:
    agent = _build_agent()
    query = "김현수 분석해줘"
    entity_filter = extract_entities_from_query(query)

    prompt, planner_mode = agent._select_llm_planner_prompt(
        query_text=query,
        query=query,
        entity_filter=entity_filter,
        current_date="2026년 03월 12일",
        current_year=2026,
        entity_context="",
    )

    assert planner_mode == "player_llm_planner"
    assert "허용 도구: validate_player, get_player_stats, get_career_stats" in prompt
    assert "JSON 한 줄만 출력한다." in prompt
    assert "analysis와 expected_result는 12자 이내로 짧게 쓰고 설명문은 금지한다." in prompt


def test_select_llm_planner_prompt_prioritizes_career_tool_for_career_queries() -> None:
    agent = _build_agent()
    query = "김현수 통산 분석해줘"
    entity_filter = extract_entities_from_query(query)

    prompt, planner_mode = agent._select_llm_planner_prompt(
        query_text=query,
        query=query,
        entity_filter=entity_filter,
        current_date="2026년 03월 12일",
        current_year=2026,
        entity_context="",
    )

    assert planner_mode == "player_llm_planner"
    assert "우선순위: get_career_stats, get_player_stats" in prompt


def test_extract_player_names_preserves_multi_player_query_order() -> None:
    query = "김도영, 문보경, 노시환을 묶어서 보면 타석 접근법을 각각 어떻게 설명할 수 있어?"

    assert extract_player_names(query, limit=3) == ["김도영", "문보경", "노시환"]


def test_extract_llm_planner_player_names_normalizes_suffixes_and_order() -> None:
    agent = _build_agent()
    query = "안현민, 윤도현, 김도영처럼 젊은 타자를 볼 때 성장 신호 순서를 설명해줘"
    entity_filter = extract_entities_from_query(query)

    assert agent._extract_llm_planner_player_names(query, entity_filter) == [
        "안현민",
        "윤도현",
        "김도영",
    ]


def test_select_llm_planner_prompt_includes_multi_player_scope() -> None:
    agent = _build_agent()
    agent._resolve_reference_year = lambda query, entity_filter: 2025
    query = "김도영, 문보경, 노시환을 같이 보면 어떤 유형의 타자인지 설명해줘"
    entity_filter = extract_entities_from_query(query)

    prompt, planner_mode = agent._select_llm_planner_prompt(
        query_text=query,
        query=query,
        entity_filter=entity_filter,
        current_date="2026년 03월 12일",
        current_year=2026,
        entity_context="",
    )

    assert planner_mode == "player_llm_planner"
    assert "기본 시즌 연도: 2025" in prompt
    assert "tool_calls 최대 3개." in prompt
    assert "질문 등장 선수: 김도영, 문보경, 노시환." in prompt
    assert "player_names 배열은 금지" in prompt


def test_soft_filter_llm_tool_calls_limits_team_planner_scope() -> None:
    agent = _build_agent()
    query = "LG 팀 분석해줘"
    entity_filter = extract_entities_from_query(query)

    filtered = agent._soft_filter_llm_tool_calls(
        [
            ToolCall(
                tool_name="get_team_last_game",
                parameters={"team_name": "LG", "year": 2025},
            ),
            ToolCall(
                tool_name="get_player_stats",
                parameters={"player_name": "김현수", "year": 2025},
            ),
            ToolCall(
                tool_name="get_team_summary",
                parameters={"team_name": "LG", "year": 2025},
            ),
            ToolCall(
                tool_name="get_team_rank",
                parameters={"team_name": "LG", "year": 2025},
            ),
        ],
        query=query,
        entity_filter=entity_filter,
        planner_mode="team_llm_planner",
    )

    assert [call.tool_name for call in filtered] == [
        "get_team_summary",
        "get_team_rank",
    ]


def test_soft_filter_llm_tool_calls_prioritizes_player_stats_for_player_planner() -> (
    None
):
    agent = _build_agent()
    query = "김현수 통산 분석해줘"
    entity_filter = extract_entities_from_query(query)

    filtered = agent._soft_filter_llm_tool_calls(
        [
            ToolCall(
                tool_name="get_team_summary",
                parameters={"team_name": "LG", "year": 2025},
            ),
            ToolCall(
                tool_name="get_player_stats",
                parameters={
                    "player_name": "김현수",
                    "year": 2025,
                    "position": "both",
                },
            ),
            ToolCall(
                tool_name="get_career_stats",
                parameters={"player_name": "김현수", "position": "both"},
            ),
        ],
        query=query,
        entity_filter=entity_filter,
        planner_mode="player_llm_planner",
    )

    assert [call.tool_name for call in filtered] == [
        "get_career_stats",
        "get_player_stats",
    ]


def test_soft_filter_llm_tool_calls_expands_multi_player_batch_for_player_planner() -> (
    None
):
    agent = _build_agent()
    agent._resolve_reference_year = lambda query, entity_filter: 2025
    query = "김도영, 문보경, 노시환을 묶어서 보면 타석 접근법을 각각 어떤 유형의 타자로 설명할 수 있어?"
    entity_filter = extract_entities_from_query(query)

    filtered = agent._soft_filter_llm_tool_calls(
        [
            {
                "tool_name": "get_player_stats",
                "parameters": {
                    "player_names": ["김도영", "문보경", "노시환"],
                    "year": 2026,
                },
            },
            {
                "tool_name": "get_career_stats",
                "parameters": {
                    "player_names": ["김도영", "문보경", "노시환"],
                },
            },
        ],
        query=query,
        entity_filter=entity_filter,
        planner_mode="player_llm_planner",
    )

    assert [call.tool_name for call in filtered] == [
        "get_player_stats",
        "get_player_stats",
        "get_player_stats",
    ]
    assert [call.parameters["player_name"] for call in filtered] == [
        "김도영",
        "문보경",
        "노시환",
    ]
    assert all(call.parameters["year"] == 2025 for call in filtered)


def test_soft_filter_llm_tool_calls_supplements_missing_multi_player_names() -> None:
    agent = _build_agent()
    agent._resolve_reference_year = lambda query, entity_filter: 2025
    query = "김도영, 문보경, 노시환을 묶어서 보면 타석 접근법을 각각 어떤 유형의 타자로 설명할 수 있어?"
    entity_filter = extract_entities_from_query(query)

    filtered = agent._soft_filter_llm_tool_calls(
        [
            ToolCall(
                tool_name="get_player_stats",
                parameters={"player_name": "김도영", "year": 2026, "position": "both"},
            ),
            ToolCall(
                tool_name="get_player_stats",
                parameters={"player_name": "문보경", "year": 2026, "position": "both"},
            ),
        ],
        query=query,
        entity_filter=entity_filter,
        planner_mode="player_llm_planner",
    )

    assert [call.tool_name for call in filtered] == [
        "get_player_stats",
        "get_player_stats",
        "get_player_stats",
    ]
    assert [call.parameters["player_name"] for call in filtered] == [
        "김도영",
        "문보경",
        "노시환",
    ]
    assert all(call.parameters["year"] == 2025 for call in filtered)


def test_soft_filter_llm_tool_calls_applies_reference_year_to_missing_player_year() -> (
    None
):
    agent = _build_agent()
    agent._resolve_reference_year = lambda query, entity_filter: 2025
    query = "김현수 분석해줘"
    entity_filter = extract_entities_from_query(query)

    filtered = agent._soft_filter_llm_tool_calls(
        [
            ToolCall(
                tool_name="get_player_stats",
                parameters={"player_name": "김현수", "position": "both"},
            )
        ],
        query=query,
        entity_filter=entity_filter,
        planner_mode="player_llm_planner",
    )

    assert filtered[0].parameters["year"] == 2025


def test_analyze_query_and_plan_tools_uses_player_fast_path_for_multi_player_explainer() -> (
    None
):
    agent = _build_agent()
    agent._resolve_chat_intent = lambda query, entity_filter: SimpleNamespace(
        intent=ChatIntent.UNKNOWN
    )
    agent._resolve_reference_year = lambda query, entity_filter: 2025

    llm_calls = {"count": 0}

    async def _fake_llm(query, context):
        llm_calls["count"] += 1
        return {
            "analysis": "llm plan",
            "tool_calls": [],
            "planner_mode": "player_llm_planner",
            "error": None,
        }

    agent._analyze_query_with_llm = _fake_llm

    async def _run():
        return await agent._analyze_query_and_plan_tools(
            "박동원, 양의지, 유강남을 타격형 포수 묶음으로 볼 때 경기 전개 속에서 각자의 가치가 커지는 장면을 풀어줘.",
            {},
        )

    plan = asyncio.run(_run())

    assert llm_calls["count"] == 0
    assert plan["planner_mode"] == "player_fast_path"
    assert [call.tool_name for call in plan["tool_calls"]] == [
        "get_player_stats",
        "get_player_stats",
        "get_player_stats",
    ]
    assert [call.parameters["player_name"] for call in plan["tool_calls"]] == [
        "박동원",
        "양의지",
        "유강남",
    ]
    assert all(call.parameters["year"] == 2025 for call in plan["tool_calls"])


def test_analyze_query_and_plan_tools_uses_player_fast_path_when_ranking_phrase_is_negated() -> None:
    agent = _build_agent()
    agent._resolve_chat_intent = lambda query, entity_filter: SimpleNamespace(
        intent=ChatIntent.UNKNOWN
    )
    agent._resolve_reference_year = lambda query, entity_filter: 2025

    llm_calls = {"count": 0}

    async def _fake_llm(query, context):
        del query, context
        llm_calls["count"] += 1
        return {
            "analysis": "llm plan",
            "tool_calls": [],
            "planner_mode": "player_llm_planner",
            "error": None,
        }

    agent._analyze_query_with_llm = _fake_llm

    async def _run():
        return await agent._analyze_query_and_plan_tools(
            "김도영, 문보경, 노시환을 묶어서 보면 타석 접근법을 각각 어떤 유형의 타자로 설명할 수 있어? 단순 순위나 랭킹 말고 서사적으로 풀어줘.",
            {},
        )

    plan = asyncio.run(_run())

    assert llm_calls["count"] == 0
    assert plan["planner_mode"] == "player_fast_path"
    assert all(call.parameters["position"] == "batting" for call in plan["tool_calls"])


def test_build_reference_fast_path_plan_skips_player_fast_path_for_multi_player_ranking_query() -> None:
    agent = _build_agent()
    agent._resolve_chat_intent = lambda query, entity_filter: SimpleNamespace(
        intent=ChatIntent.UNKNOWN
    )
    agent._resolve_reference_year = lambda query, entity_filter: 2025
    query = "김도영, 문보경, 노시환 홈런 순위와 리더보드 판도를 설명해줘"

    plan = agent._build_reference_fast_path_plan(
        query,
        extract_entities_from_query(query),
    )

    assert plan is not None
    assert plan["planner_mode"] != "player_fast_path"


def test_analyze_query_and_plan_tools_uses_cached_llm_plan(monkeypatch) -> None:
    agent = _build_agent()
    agent.chat_planner_cache_ttl_seconds = 30
    agent.chat_planner_cache_max_entries = 16
    agent._build_fast_path_plan = lambda query, entity_filter, context=None: None

    llm_calls = {"count": 0}

    async def _fake_llm(query, context):
        llm_calls["count"] += 1
        return {
            "analysis": "llm plan",
            "tool_calls": [
                ToolCall(
                    tool_name="get_leaderboard",
                    parameters={
                        "stat_name": "ops",
                        "year": 2025,
                        "position": "batting",
                        "limit": 10,
                    },
                )
            ],
            "planner_mode": "default_llm_planner",
            "error": None,
        }

    agent._analyze_query_with_llm = _fake_llm

    async def _run():
        first = await agent._analyze_query_and_plan_tools("김도영 순위 알려줘", {})
        second = await agent._analyze_query_and_plan_tools("김도영 순위 알려줘", {})
        return first, second

    import asyncio

    first_plan, second_plan = asyncio.run(_run())

    assert llm_calls["count"] == 1
    assert first_plan["tool_calls"][0].tool_name == "get_leaderboard"
    assert second_plan["planner_cache_hit"] is True
    assert second_plan["tool_calls"][0].tool_name == "get_leaderboard"


def test_analyze_query_and_plan_tools_keeps_cache_alive_for_completion_stream_gap(
    monkeypatch,
) -> None:
    agent = _build_agent()
    agent.chat_planner_cache_ttl_seconds = 60
    agent.chat_planner_cache_max_entries = 16
    agent._build_fast_path_plan = lambda query, entity_filter, context=None: None

    llm_calls = {"count": 0}
    monotonic_now = {"value": 100.0}

    async def _fake_llm(query, context):
        llm_calls["count"] += 1
        return {
            "analysis": "llm plan",
            "tool_calls": [
                ToolCall(
                    tool_name="get_player_stats",
                    parameters={"player_name": "강민호", "year": 2025},
                )
            ],
            "planner_mode": "player_llm_planner",
            "error": None,
        }

    monkeypatch.setattr("app.agents.baseball_agent.time.monotonic", lambda: monotonic_now["value"])
    agent._analyze_query_with_llm = _fake_llm

    async def _run():
        first = await agent._analyze_query_and_plan_tools(
            "강민호, 양의지, 박동원 비교해줘", {}
        )
        monotonic_now["value"] = 124.0
        second = await agent._analyze_query_and_plan_tools(
            "강민호, 양의지, 박동원 비교해줘", {}
        )
        return first, second

    first_plan, second_plan = asyncio.run(_run())

    assert llm_calls["count"] == 1
    assert first_plan["planner_mode"] == "player_llm_planner"
    assert second_plan["planner_cache_hit"] is True
    assert second_plan["planner_cache_age_ms"] == 24000.0


def test_process_query_stream_preserves_attempted_planner_mode_on_analysis_error(
    monkeypatch,
) -> None:
    agent = _build_agent()
    agent._is_chitchat = lambda query: False
    agent._analyze_query_and_plan_tools = lambda query, context: _analysis_error_plan()
    agent._build_known_explainer_answer = lambda *args, **kwargs: None
    agent._build_known_latest_answer = lambda *args, **kwargs: None
    agent._resolve_tool_execution_mode = lambda tool_calls: "none"

    monkeypatch.setattr("app.ml.intent_router.predict_intent", lambda query: "freeform")

    async def _run():
        return [
            event
            async for event in agent.process_query_stream(
                "양의지와 박동원을 비교해줘", {}
            )
        ]

    async def _analysis_error_plan():
        return {
            "analysis": "",
            "tool_calls": [],
            "expected_result": "",
            "error": "질문 분석 오류: provider empty chunk",
            "planner_mode": "player_llm_planner",
        }

    import asyncio

    events = asyncio.run(_run())
    metadata = next(event["data"] for event in events if event["type"] == "metadata")

    assert metadata["planner_mode"] == "player_llm_planner"
    assert metadata["planner_status"] == "analysis_error"
    assert metadata["error"] == "analysis_temporarily_unavailable"


def test_process_query_stream_emits_fast_path_answer_before_metadata_for_stream(
    monkeypatch,
) -> None:
    agent = _build_agent()
    agent._is_chitchat = lambda query: False
    agent.fast_path_fallback_on_empty = False
    agent._generate_visualizations = lambda tool_results: []
    agent._resolve_tool_execution_mode = lambda tool_calls: "sequential"
    agent._build_stats_lookup_followup_tool_call = (
        lambda query, analysis_result, tool_results: None
    )
    agent._build_low_grounding_fallback_plan = (
        lambda query, analysis_result, tool_results: []
    )
    agent._resolve_grounding_mode = (
        lambda intent, analysis_result, tool_results: "structured_kbo"
    )
    agent._source_tier_from_tool_results = (
        lambda tool_results, default_tier=None: default_tier or "database"
    )
    agent._build_answer_sources = lambda tool_results: []
    agent._resolve_as_of_date = lambda tool_results: None
    agent._resolve_perf_model_name = lambda: "test-model"

    async def _fast_path_plan():
        return {
            "analysis": "ok",
            "tool_calls": [],
            "expected_result": "",
            "error": None,
            "planner_mode": "fast_path",
            "intent": "team_analysis",
            "source_tier": "database",
        }

    async def _empty_results():
        return []

    async def _fast_answer(query, tool_results, context):
        del query, tool_results, context
        return {
            "answer": "빠른 응답",
            "verified": True,
            "data_sources": [{"tool": "database", "verified": True}],
            "error": None,
        }

    agent._analyze_query_and_plan_tools = lambda query, context: _fast_path_plan()
    agent._execute_tool_batch_async = lambda tool_calls: _empty_results()
    agent._generate_verified_answer = _fast_answer

    monkeypatch.setattr(
        "app.ml.intent_router.predict_intent", lambda query: "team_analysis"
    )

    async def _run():
        return [
            event
            async for event in agent.process_query_stream(
                "LG 흐름 정리해줘", {"request_mode": "stream"}
            )
        ]

    events = asyncio.run(_run())
    event_types = [event["type"] for event in events]

    assert event_types.index("answer_chunk") < event_types.index("metadata")


def test_process_query_stream_preserves_metadata_before_answer_for_completion_fast_path(
    monkeypatch,
) -> None:
    agent = _build_agent()
    agent._is_chitchat = lambda query: False
    agent.fast_path_fallback_on_empty = False
    agent._generate_visualizations = lambda tool_results: []
    agent._resolve_tool_execution_mode = lambda tool_calls: "sequential"
    agent._build_stats_lookup_followup_tool_call = (
        lambda query, analysis_result, tool_results: None
    )
    agent._build_low_grounding_fallback_plan = (
        lambda query, analysis_result, tool_results: []
    )
    agent._resolve_grounding_mode = (
        lambda intent, analysis_result, tool_results: "structured_kbo"
    )
    agent._source_tier_from_tool_results = (
        lambda tool_results, default_tier=None: default_tier or "database"
    )
    agent._build_answer_sources = lambda tool_results: []
    agent._resolve_as_of_date = lambda tool_results: None
    agent._resolve_perf_model_name = lambda: "test-model"

    async def _fast_path_plan():
        return {
            "analysis": "ok",
            "tool_calls": [],
            "expected_result": "",
            "error": None,
            "planner_mode": "fast_path",
            "intent": "team_analysis",
            "source_tier": "database",
        }

    async def _empty_results():
        return []

    async def _fast_answer(query, tool_results, context):
        del query, tool_results, context
        return {
            "answer": "빠른 응답",
            "verified": True,
            "data_sources": [{"tool": "database", "verified": True}],
            "error": None,
        }

    agent._analyze_query_and_plan_tools = lambda query, context: _fast_path_plan()
    agent._execute_tool_batch_async = lambda tool_calls: _empty_results()
    agent._generate_verified_answer = _fast_answer

    monkeypatch.setattr(
        "app.ml.intent_router.predict_intent", lambda query: "team_analysis"
    )

    async def _run():
        return [
            event
            async for event in agent.process_query_stream(
                "LG 흐름 정리해줘", {"request_mode": "completion"}
            )
        ]

    events = asyncio.run(_run())
    event_types = [event["type"] for event in events]

    assert event_types.index("metadata") < event_types.index("answer_chunk")


def test_process_query_stream_emits_player_fast_path_answer_before_metadata_for_stream(
    monkeypatch,
) -> None:
    agent = _build_agent()
    agent._is_chitchat = lambda query: False
    agent.fast_path_fallback_on_empty = False
    agent._generate_visualizations = lambda tool_results: []
    agent._resolve_tool_execution_mode = lambda tool_calls: "sequential"
    agent._build_stats_lookup_followup_tool_call = (
        lambda query, analysis_result, tool_results: None
    )
    agent._build_low_grounding_fallback_plan = (
        lambda query, analysis_result, tool_results: []
    )
    agent._resolve_grounding_mode = (
        lambda intent, analysis_result, tool_results: "structured_kbo"
    )
    agent._source_tier_from_tool_results = (
        lambda tool_results, default_tier=None: default_tier or "database"
    )
    agent._build_answer_sources = lambda tool_results: []
    agent._resolve_as_of_date = lambda tool_results: None
    agent._resolve_perf_model_name = lambda: "test-model"

    async def _player_fast_path_plan():
        return {
            "analysis": "ok",
            "tool_calls": [],
            "expected_result": "",
            "error": None,
            "planner_mode": "player_fast_path",
            "intent": "freeform",
            "source_tier": "database",
            "grounding_mode": "structured_kbo",
        }

    async def _empty_results():
        return []

    async def _fast_answer(query, tool_results, context):
        del query, tool_results, context
        return {
            "answer": "빠른 선수 응답",
            "verified": True,
            "data_sources": [{"tool": "database", "verified": True}],
            "error": None,
        }

    agent._analyze_query_and_plan_tools = lambda query, context: _player_fast_path_plan()
    agent._execute_tool_batch_async = lambda tool_calls: _empty_results()
    agent._generate_verified_answer = _fast_answer

    monkeypatch.setattr("app.ml.intent_router.predict_intent", lambda query: "freeform")

    async def _run():
        return [
            event
            async for event in agent.process_query_stream(
                "박동원, 양의지, 유강남 해석해줘",
                {"request_mode": "stream"},
            )
        ]

    events = asyncio.run(_run())
    event_types = [event["type"] for event in events]

    assert event_types.index("answer_chunk") < event_types.index("metadata")


def test_resolve_llm_planner_max_tokens_caps_compact_player_and_team_modes() -> None:
    agent = _build_agent()

    assert agent._resolve_llm_planner_max_tokens("player_llm_planner") == 128
    assert agent._resolve_llm_planner_max_tokens("team_llm_planner") == 128
    assert agent._resolve_llm_planner_max_tokens("default_llm_planner") == 350


def test_resolve_llm_planner_timeout_seconds_caps_compact_player_and_team_modes() -> (
    None
):
    agent = _build_agent()

    assert agent._resolve_llm_planner_timeout_seconds("player_llm_planner") == 24.0
    assert agent._resolve_llm_planner_timeout_seconds("team_llm_planner") == 24.0
    assert agent._resolve_llm_planner_timeout_seconds("default_llm_planner") == 30.0


def test_resolve_tool_execution_mode_forces_sequential_for_db_backed_batches() -> None:
    agent = _build_agent()

    assert (
        agent._resolve_tool_execution_mode(
            [
                ToolCall("get_player_stats", {"player_name": "김현수", "year": 2025}),
                ToolCall("get_career_stats", {"player_name": "김현수"}),
            ]
        )
        == "sequential"
    )
    assert (
        agent._resolve_tool_execution_mode(
            [
                ToolCall("get_current_datetime", {}),
                ToolCall("get_baseball_season_info", {}),
            ]
        )
        == "parallel"
    )


def test_resolve_tool_execution_mode_returns_mixed_for_split_batches() -> None:
    agent = _build_agent()

    assert (
        agent._resolve_tool_execution_mode(
            [
                ToolCall("get_current_datetime", {}),
                ToolCall("get_player_stats", {"player_name": "김현수", "year": 2025}),
            ]
        )
        == "mixed"
    )


def test_resolve_tool_execution_mode_falls_back_to_sequential_when_split_disabled() -> (
    None
):
    agent = _build_agent()
    agent.chat_tool_parallel_split_batch_enabled = False

    assert (
        agent._resolve_tool_execution_mode(
            [
                ToolCall("get_current_datetime", {}),
                ToolCall("get_player_stats", {"player_name": "김현수", "year": 2025}),
            ]
        )
        == "sequential"
    )


def test_execute_tool_batch_async_mixed_mode_preserves_original_order() -> None:
    agent = _build_agent()
    agent.tool_caller = SimpleNamespace()

    async def _serial_execute(tool_call: ToolCall) -> ToolResult:
        return ToolResult(
            success=True,
            data={"tool": tool_call.tool_name, "mode": "serial"},
            message="ok",
        )

    async def _parallel_execute(tool_calls, max_concurrency=None):
        del max_concurrency
        return [
            ToolResult(
                success=True,
                data={"tool": tool_call.tool_name, "mode": "parallel"},
                message="ok",
            )
            for tool_call in tool_calls
        ]

    agent._execute_tool_call_async = AsyncMock(side_effect=_serial_execute)
    agent.tool_caller.execute_multiple_tools_parallel = _parallel_execute

    results = asyncio.run(
        agent._execute_tool_batch_async(
            [
                ToolCall("get_current_datetime", {}),
                ToolCall("get_player_stats", {"player_name": "김현수", "year": 2025}),
                ToolCall("get_baseball_season_info", {}),
            ]
        )
    )

    assert [result.data["tool"] for result in results] == [
        "get_current_datetime",
        "get_player_stats",
        "get_baseball_season_info",
    ]
    assert [result.data["mode"] for result in results] == [
        "parallel",
        "serial",
        "parallel",
    ]
    assert agent._execute_tool_call_async.await_count == 1


def test_execute_tool_batch_async_uses_batched_player_stats_lookup() -> None:
    agent = _build_agent()
    agent.db_query_tool = SimpleNamespace(
        get_player_season_stats_batch=lambda player_names, year, position: [
            {
                "requested_player_name": "안현민",
                "player_name": "안현민",
                "resolved_player_name": "안현민",
                "year": year,
                "batting_stats": {
                    "player_name": "안현민",
                    "team_name": "KT",
                    "obp": 0.392,
                    "slg": 0.522,
                    "home_runs": 16,
                },
                "pitching_stats": None,
                "found": True,
                "error": None,
                "batch_lookup": True,
            },
            {
                "requested_player_name": "윤도현",
                "player_name": "윤도현",
                "resolved_player_name": "윤도현",
                "year": year,
                "batting_stats": {
                    "player_name": "윤도현",
                    "team_name": "KIA",
                    "obp": 0.355,
                    "slg": 0.441,
                    "home_runs": 8,
                },
                "pitching_stats": None,
                "found": True,
                "error": None,
                "batch_lookup": True,
            },
            {
                "requested_player_name": "김도영",
                "player_name": "김도영",
                "resolved_player_name": None,
                "year": year,
                "batting_stats": None,
                "pitching_stats": None,
                "found": False,
                "error": None,
                "batch_lookup": True,
            },
        ]
    )
    agent._execute_tool_call_async = AsyncMock(
        side_effect=AssertionError("individual tool execution should not run")
    )

    results = asyncio.run(
        agent._execute_tool_batch_async(
            [
                ToolCall(
                    "get_player_stats",
                    {"player_name": "안현민", "year": 2025, "position": "batting"},
                ),
                ToolCall(
                    "get_player_stats",
                    {"player_name": "윤도현", "year": 2025, "position": "batting"},
                ),
                ToolCall(
                    "get_player_stats",
                    {"player_name": "김도영", "year": 2025, "position": "batting"},
                ),
            ]
        )
    )

    assert [result.success for result in results] == [True, True, True]
    assert results[0].data["resolved_player_name"] == "안현민"
    assert results[1].data["resolved_player_name"] == "윤도현"
    assert results[2].data["found"] is False
    assert "찾을 수 없습니다" in results[2].message


def test_build_player_fast_path_tool_calls_infers_pitching_from_query() -> None:
    agent = _build_agent()

    tool_calls = agent._build_player_fast_path_tool_calls(
        "문동주, 김택연, 곽빈을 강한 구위 축으로 묶었을 때 어떤 하위 유형으로 나뉘는지 설명해줘.",
        ["문동주", "김택연", "곽빈"],
        2025,
        SimpleNamespace(position_type=None),
    )

    assert [call.parameters["position"] for call in tool_calls] == [
        "pitching",
        "pitching",
        "pitching",
    ]


def test_build_structured_deterministic_answer_for_multi_player_batter_narrative() -> None:
    agent = _build_agent()

    answer = agent._build_structured_deterministic_answer(
        "안현민, 윤도현, 김도영처럼 젊은 타자를 볼 때 표본이 적어도 먼저 읽어야 할 성장 신호의 순서를 설명해줘.",
        [
            ToolResult(
                success=True,
                data={
                    "requested_player_name": "안현민",
                    "player_name": "안현민",
                    "resolved_player_name": "안현민",
                    "year": 2025,
                    "batting_stats": {
                        "player_name": "안현민",
                        "team_name": "KT",
                        "plate_appearances": 210,
                        "avg": 0.311,
                        "obp": 0.392,
                        "slg": 0.551,
                        "ops": 0.943,
                        "home_runs": 13,
                        "doubles": 12,
                        "walks": 24,
                        "strikeouts": 39,
                    },
                    "pitching_stats": None,
                    "found": True,
                    "error": None,
                    "batch_lookup": True,
                },
                message="ok",
            ),
            ToolResult(
                success=True,
                data={
                    "requested_player_name": "윤도현",
                    "player_name": "윤도현",
                    "resolved_player_name": "윤도현",
                    "year": 2025,
                    "batting_stats": {
                        "player_name": "윤도현",
                        "team_name": "KIA",
                        "plate_appearances": 188,
                        "avg": 0.287,
                        "obp": 0.349,
                        "slg": 0.448,
                        "ops": 0.797,
                        "home_runs": 7,
                        "doubles": 14,
                        "walks": 18,
                        "strikeouts": 41,
                    },
                    "pitching_stats": None,
                    "found": True,
                    "error": None,
                    "batch_lookup": True,
                },
                message="ok",
            ),
            ToolResult(
                success=True,
                data={
                    "requested_player_name": "김도영",
                    "player_name": "김도영",
                    "resolved_player_name": "김도영",
                    "year": 2025,
                    "batting_stats": {
                        "player_name": "김도영",
                        "team_name": "KIA",
                        "plate_appearances": 240,
                        "avg": 0.334,
                        "obp": 0.409,
                        "slg": 0.603,
                        "ops": 1.012,
                        "home_runs": 18,
                        "doubles": 16,
                        "walks": 30,
                        "strikeouts": 35,
                    },
                    "pitching_stats": None,
                    "found": True,
                    "error": None,
                    "batch_lookup": True,
                },
                message="ok",
            ),
        ],
    )

    assert answer is not None
    assert "타석 접근 안정감" in answer
    assert "안현민" in answer
    assert "윤도현" in answer
    assert "김도영" in answer
    assert "출처: DB 조회 결과" in answer


def test_build_chat_reference_answer_for_multi_player_pitcher_narrative_mentions_missing_player() -> None:
    agent = _build_agent()

    answer = agent._build_chat_reference_answer(
        "문동주, 김택연, 곽빈을 강한 구위 축으로 묶었을 때 파워형 투수 안에서도 어떤 하위 유형으로 나눠볼 수 있어?",
        [
            ToolResult(
                success=True,
                data={
                    "requested_player_name": "문동주",
                    "player_name": "문동주",
                    "resolved_player_name": "문동주",
                    "year": 2025,
                    "batting_stats": None,
                    "pitching_stats": {
                        "player_name": "문동주",
                        "team_name": "한화",
                        "games_started": 18,
                        "innings_pitched": 112.2,
                        "strikeouts": 128,
                        "era": 3.41,
                        "whip": 1.18,
                    },
                    "found": True,
                    "error": None,
                    "batch_lookup": True,
                },
                message="ok",
            ),
            ToolResult(
                success=True,
                data={
                    "requested_player_name": "김택연",
                    "player_name": "김택연",
                    "resolved_player_name": "김택연",
                    "year": 2025,
                    "batting_stats": None,
                    "pitching_stats": {
                        "player_name": "김택연",
                        "team_name": "두산",
                        "games_started": 0,
                        "innings_pitched": 58.1,
                        "strikeouts": 76,
                        "saves": 12,
                        "holds": 10,
                        "era": 2.91,
                        "whip": 1.05,
                    },
                    "found": True,
                    "error": None,
                    "batch_lookup": True,
                },
                message="ok",
            ),
            ToolResult(
                success=True,
                data={
                    "requested_player_name": "곽빈",
                    "player_name": "곽빈",
                    "resolved_player_name": None,
                    "year": 2025,
                    "batting_stats": None,
                    "pitching_stats": None,
                    "found": False,
                    "error": None,
                    "batch_lookup": True,
                },
                message="miss",
            ),
        ],
    )

    assert answer is not None
    assert "문동주" in answer
    assert "김택연" in answer
    assert "곽빈" in answer
    assert "제외" in answer or "빠진 선수" in answer


def test_build_structured_deterministic_answer_returns_none_when_multi_player_usable_count_is_low() -> None:
    agent = _build_agent()

    answer = agent._build_structured_deterministic_answer(
        "안현민, 윤도현, 김도영처럼 젊은 타자를 볼 때 성장 신호 순서를 설명해줘",
        [
            ToolResult(
                success=True,
                data={
                    "requested_player_name": "안현민",
                    "player_name": "안현민",
                    "resolved_player_name": "안현민",
                    "year": 2025,
                    "batting_stats": {
                        "player_name": "안현민",
                        "team_name": "KT",
                        "obp": 0.392,
                        "slg": 0.522,
                        "home_runs": 16,
                    },
                    "pitching_stats": None,
                    "found": True,
                    "error": None,
                    "batch_lookup": True,
                },
                message="ok",
            ),
            ToolResult(
                success=True,
                data={
                    "requested_player_name": "윤도현",
                    "player_name": "윤도현",
                    "resolved_player_name": None,
                    "year": 2025,
                    "batting_stats": None,
                    "pitching_stats": None,
                    "found": False,
                    "error": None,
                    "batch_lookup": True,
                },
                message="miss",
            ),
        ],
    )

    assert answer is None


def test_analyze_query_with_llm_uses_compact_player_planner_token_cap() -> None:
    agent = _build_agent()
    captured = {}

    async def _fake_llm(messages, max_tokens=None):
        captured["prompt"] = messages[0]["content"]
        captured["max_tokens"] = max_tokens
        yield '{"analysis":"짧게","tool_calls":[],"expected_result":"짧게"}'

    agent.llm_generator = _fake_llm

    async def _run():
        return await agent._analyze_query_with_llm("김현수 분석해줘", {})

    import asyncio

    result = asyncio.run(_run())

    assert result["planner_mode"] == "player_llm_planner"
    assert captured["max_tokens"] == 128
    assert "JSON 한 줄만 출력한다." in captured["prompt"]


def test_analyze_query_with_llm_returns_analysis_error_on_planner_timeout() -> None:
    agent = _build_agent()
    agent.chat_planner_timeout_seconds = 0.01
    agent.chat_compact_planner_timeout_seconds = 0.01

    async def _slow_llm(messages, max_tokens=None):
        del messages, max_tokens
        import asyncio

        await asyncio.sleep(0.2)
        yield '{"analysis":"짧게","tool_calls":[],"expected_result":"짧게"}'

    agent.llm_generator = _slow_llm

    async def _run():
        return await agent._analyze_query_with_llm("김현수 분석해줘", {})

    import asyncio

    result = asyncio.run(_run())

    assert result["planner_mode"] == "player_llm_planner"
    assert result["tool_calls"] == []
    assert result["error"] == "질문 분석 오류: planner timeout"


def test_process_query_stream_retries_once_on_answer_prefetch_error_for_completion(
    monkeypatch,
) -> None:
    agent = _build_agent()
    agent._is_chitchat = lambda query: False
    agent._generate_visualizations = lambda tool_results: []
    agent._execute_tool_batch_async = lambda tool_calls: _empty_results()
    agent._resolve_tool_execution_mode = lambda tool_calls: "none"
    agent._analyze_query_and_plan_tools = lambda query, context: _basic_plan()

    attempt_counter = {"count": 0}

    async def _broken_answer():
        raise RuntimeError("provider empty chunk")
        yield ""

    async def _good_answer():
        yield "정상 복구 답변"

    async def _fake_generate_verified_answer(query, tool_results, context):
        del query, tool_results, context
        attempt_counter["count"] += 1
        if attempt_counter["count"] == 1:
            return {
                "answer": _broken_answer(),
                "verified": False,
                "data_sources": [],
                "error": None,
            }
        return {
            "answer": _good_answer(),
            "verified": True,
            "data_sources": [{"tool": "database", "verified": True}],
            "error": None,
        }

    async def _basic_plan():
        return {
            "analysis": "ok",
            "tool_calls": [],
            "expected_result": "",
            "error": None,
            "planner_mode": "default_llm_planner",
        }

    async def _empty_results():
        return []

    agent._generate_verified_answer = _fake_generate_verified_answer
    monkeypatch.setattr("app.ml.intent_router.predict_intent", lambda query: "freeform")

    async def _run():
        return [
            event
            async for event in agent.process_query_stream(
                "김현수 분석해줘", {"request_mode": "completion"}
            )
        ]

    import asyncio

    events = asyncio.run(_run())
    metadata = next(event["data"] for event in events if event["type"] == "metadata")
    answer_chunks = [event["content"] for event in events if event["type"] == "answer_chunk"]

    assert answer_chunks == ["정상 복구 답변"]
    assert metadata["error"] is None
    assert metadata["fallback_answer_used"] is False
    assert metadata["perf"]["answer_retry_count"] == 1


def test_process_query_stream_recovers_with_deterministic_answer_on_prefetch_error(
    monkeypatch,
) -> None:
    agent = _build_agent()
    agent._is_chitchat = lambda query: False
    agent.chat_first_token_retry_max_attempts = 0
    agent._generate_visualizations = lambda tool_results: []
    agent._resolve_tool_execution_mode = lambda tool_calls: "sequential"
    agent._analyze_query_and_plan_tools = lambda query, context: _player_plan()
    agent._execute_tool_batch_async = lambda tool_calls: _player_results()

    async def _broken_answer():
        raise RuntimeError("provider empty chunk")
        yield ""

    async def _fake_generate_verified_answer(query, tool_results, context):
        del query, tool_results, context
        return {
            "answer": _broken_answer(),
            "verified": False,
            "data_sources": [{"tool": "database", "verified": True}],
            "error": None,
        }

    async def _player_plan():
        return {
            "analysis": "ok",
            "tool_calls": [
                ToolCall(
                    "get_player_stats",
                    {"player_name": "김현수", "year": 2025, "position": "both"},
                )
            ],
            "expected_result": "",
            "error": None,
            "planner_mode": "player_llm_planner",
        }

    async def _player_results():
        return [
            ToolResult(
                success=True,
                data={
                    "player_name": "김현수",
                    "year": 2025,
                    "batting_stats": {
                        "player_name": "김현수",
                        "team_name": "LG",
                        "avg": 0.321,
                        "ops": 0.912,
                        "home_runs": 22,
                        "rbi": 88,
                    },
                    "pitching_stats": None,
                    "found": True,
                },
                message="2025년 김현수 선수 통계를 성공적으로 조회했습니다.",
            )
        ]

    agent._generate_verified_answer = _fake_generate_verified_answer
    monkeypatch.setattr("app.ml.intent_router.predict_intent", lambda query: "freeform")

    async def _run():
        return [
            event
            async for event in agent.process_query_stream(
                "김현수 분석해줘", {"request_mode": "completion"}
            )
        ]

    import asyncio

    events = asyncio.run(_run())
    metadata = next(event["data"] for event in events if event["type"] == "metadata")
    answer_chunks = [event["content"] for event in events if event["type"] == "answer_chunk"]

    assert metadata["error"] is None
    assert metadata["fallback_answer_used"] is True
    assert metadata["fallback_reason"] == "answer_generation_recovery"
    assert metadata["verified"] is True
    assert "김현수" in "".join(answer_chunks)
