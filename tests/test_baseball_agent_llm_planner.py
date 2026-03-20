from app.agents.baseball_agent import BaseballStatisticsAgent
from app.agents.tool_caller import ToolCall
from app.core.entity_extractor import extract_entities_from_query


def _build_agent() -> BaseballStatisticsAgent:
    agent = BaseballStatisticsAgent.__new__(BaseballStatisticsAgent)
    agent.fast_path_tool_cap = 2
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
