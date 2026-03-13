from app.agents.baseball_agent import BaseballStatisticsAgent
from app.agents.chat_intent_router import ChatIntentRouter
from app.core.entity_extractor import extract_entities_from_query


def _build_agent_for_fast_path() -> BaseballStatisticsAgent:
    agent = BaseballStatisticsAgent.__new__(BaseballStatisticsAgent)
    agent._resolve_award_query_type = lambda query, entity_filter: None
    agent._detect_team_alias_from_query = lambda query: None
    agent.chat_intent_router = ChatIntentRouter(
        resolve_reference_year=agent._resolve_reference_year,
        detect_team_alias=agent._detect_team_alias_from_query,
        resolve_award_query_type=agent._resolve_award_query_type,
        build_team_tool_calls=lambda query, team_name, season_year: [],
        fast_path_enabled=True,
        fast_path_scope="all",
    )
    return agent


def test_box_score_query_prefers_sql_fast_path() -> None:
    agent = _build_agent_for_fast_path()
    query = "2025년 5월 1일 경기 박스스코어 알려줘"

    plan = agent._build_reference_fast_path_plan(
        query,
        extract_entities_from_query(query),
    )

    assert plan is not None
    assert plan["tool_calls"][0].tool_name == "get_game_box_score"
    assert plan["tool_calls"][0].parameters == {"date": "2025-05-01"}


def test_games_by_date_query_still_uses_schedule_fast_path() -> None:
    agent = _build_agent_for_fast_path()
    query = "2025년 5월 1일 경기 일정 알려줘"

    plan = agent._build_reference_fast_path_plan(
        query,
        extract_entities_from_query(query),
    )

    assert plan is not None
    assert plan["tool_calls"][0].tool_name == "get_games_by_date"
    assert plan["tool_calls"][0].parameters == {"date": "2025-05-01"}


def test_game_flow_query_does_not_collapse_to_schedule_fast_path() -> None:
    agent = _build_agent_for_fast_path()
    query = "2025년 5월 1일 경기 흐름 요약해줘"

    plan = agent._build_reference_fast_path_plan(
        query,
        extract_entities_from_query(query),
    )

    assert plan is not None
    assert plan["tool_calls"][0].tool_name == "get_game_box_score"
    assert plan["tool_calls"][0].parameters == {"date": "2025-05-01"}


def test_team_scoped_game_flow_query_uses_game_box_score_fast_path() -> None:
    agent = _build_agent_for_fast_path()
    agent._detect_team_alias_from_query = lambda query: "KT" if "KT" in query else None
    agent.chat_intent_router = ChatIntentRouter(
        resolve_reference_year=agent._resolve_reference_year,
        detect_team_alias=agent._detect_team_alias_from_query,
        resolve_award_query_type=agent._resolve_award_query_type,
        build_team_tool_calls=lambda query, team_name, season_year: [],
        fast_path_enabled=True,
        fast_path_scope="all",
    )
    query = "2025년 5월 1일 KT 경기 흐름 요약해줘"

    plan = agent._build_reference_fast_path_plan(
        query,
        extract_entities_from_query(query),
    )

    assert plan is not None
    assert plan["tool_calls"][0].tool_name == "get_game_box_score"
    assert plan["tool_calls"][0].parameters == {"date": "2025-05-01"}
