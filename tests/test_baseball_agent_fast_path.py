from app.agents.baseball_agent import BaseballStatisticsAgent
from app.agents.chat_intent_router import ChatIntent, ChatIntentRouter, IntentDecision
from app.agents.tool_caller import ToolCall, ToolResult
from app.core.entity_extractor import extract_entities_from_query


def _build_agent_for_fast_path(
    *, fast_path_scope: str = "all"
) -> BaseballStatisticsAgent:
    agent = BaseballStatisticsAgent.__new__(BaseballStatisticsAgent)
    agent._resolve_award_query_type = lambda query, entity_filter: None
    agent._detect_team_alias_from_query = lambda query: (
        "LG" if "LG" in query else ("KT" if "KT" in query else None)
    )
    agent.fast_path_tool_cap = 2
    agent._team_name_cache = {}
    agent._convert_team_id_to_name = lambda team_id: team_id
    agent.chat_intent_router = ChatIntentRouter(
        resolve_reference_year=agent._resolve_reference_year,
        detect_team_alias=agent._detect_team_alias_from_query,
        resolve_award_query_type=agent._resolve_award_query_type,
        build_team_tool_calls=lambda query, team_name, season_year: [],
        fast_path_enabled=True,
        fast_path_scope=fast_path_scope,
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


def test_team_bundle_query_uses_deterministic_bundle_fast_path() -> None:
    agent = _build_agent_for_fast_path(fast_path_scope="team")
    query = "LG 최근 5경기 흐름이랑 불펜 상태 같이 봐줘"

    plan = agent._build_reference_fast_path_plan(
        query,
        extract_entities_from_query(query),
    )

    assert plan is not None
    assert plan["planner_mode"] == "fast_path_bundle"
    assert [call.tool_name for call in plan["tool_calls"]] == [
        "get_team_summary",
        "get_team_advanced_metrics",
        "get_recent_games_by_team",
    ]
    assert plan["tool_calls"][2].parameters["limit"] == 5


def test_schedule_bundle_query_uses_schedule_lineup_box_score_bundle() -> None:
    agent = _build_agent_for_fast_path()
    query = "2025년 5월 1일 LG 경기 일정이랑 라인업, 박스스코어 같이 보여줘"

    plan = agent._build_reference_fast_path_plan(
        query,
        extract_entities_from_query(query),
    )

    assert plan is not None
    assert plan["planner_mode"] == "fast_path_bundle"
    assert [call.tool_name for call in plan["tool_calls"]] == [
        "get_games_by_date",
        "get_game_lineup",
        "get_game_box_score",
    ]


def test_player_bundle_query_uses_validate_stats_leaderboard_bundle() -> None:
    agent = _build_agent_for_fast_path()
    query = "김도영 2025 홈런 몇 위고 시즌 기록 어때?"

    plan = agent._build_reference_fast_path_plan(
        query,
        extract_entities_from_query(query),
    )

    assert plan is not None
    assert plan["planner_mode"] == "fast_path_bundle"
    assert [call.tool_name for call in plan["tool_calls"]] == [
        "validate_player",
        "get_player_stats",
        "get_leaderboard",
    ]


def test_player_comparison_query_uses_compare_players_fast_path() -> None:
    agent = _build_agent_for_fast_path()
    query = "2025년 김도영과 문보경을 비교할 때 타석 접근법과 장타 생산 방식 차이를 설명해줘"

    plan = agent._build_reference_fast_path_plan(
        query,
        extract_entities_from_query(query),
    )

    assert plan is not None
    assert plan["planner_mode"] == "fast_path"
    assert [call.tool_name for call in plan["tool_calls"]] == ["compare_players"]
    assert plan["tool_calls"][0].parameters == {
        "player1": "김도영",
        "player2": "문보경",
        "comparison_type": "season",
        "position": "batting",
        "year": 2025,
    }


def test_latest_info_query_returns_manual_data_request_answer() -> None:
    agent = _build_agent_for_fast_path()
    query = "오늘 선발 변경된 팀 있어?"

    decision = agent.chat_intent_router.resolve(
        query,
        extract_entities_from_query(query),
        agent=agent,
    )

    assert decision.intent == ChatIntent.LATEST_INFO
    assert decision.tool_calls == []
    assert decision.metric_policy == "manual_data_request"
    assert decision.grounding_mode == "manual_data_request"
    assert decision.source_tier == "none"
    assert decision.fallback_reason == "manual_baseball_data_required"
    assert decision.direct_answer is not None
    assert "외부 야구 웹 조회는 사용하지 않습니다." in decision.direct_answer


def test_fast_path_bundle_dedupes_and_caps_to_three_tools() -> None:
    agent = _build_agent_for_fast_path()

    plan = agent._intent_decision_to_plan(
        IntentDecision(
            intent=ChatIntent.PLAYER_LOOKUP,
            planner_mode="fast_path_bundle",
            tool_calls=[
                ToolCall("validate_player", {"player_name": "김도영", "year": 2025}),
                ToolCall("validate_player", {"player_name": "김도영", "year": 2025}),
                ToolCall("get_player_stats", {"player_name": "김도영", "year": 2025}),
                ToolCall(
                    "get_leaderboard",
                    {
                        "stat_name": "home_runs",
                        "year": 2025,
                        "position": "batting",
                        "limit": 5,
                    },
                ),
                ToolCall("get_career_stats", {"player_name": "김도영"}),
            ],
        )
    )

    assert plan is not None
    assert [call.tool_name for call in plan["tool_calls"]] == [
        "validate_player",
        "get_player_stats",
        "get_leaderboard",
    ]


def test_player_bundle_answer_mentions_stats_and_rank() -> None:
    agent = _build_agent_for_fast_path()

    answer = agent._build_bundle_reference_answer(
        "김도영 2025 홈런 몇 위고 시즌 기록 어때?",
        [
            ToolResult(
                success=True,
                data={
                    "player_name": "김도영",
                    "year": 2025,
                    "exists": True,
                    "found_players": [
                        {
                            "player_name": "김도영",
                            "team_name": "KIA 타이거즈",
                            "position_type": "batting",
                        }
                    ],
                },
                message="ok",
            ),
            ToolResult(
                success=True,
                data={
                    "player_name": "김도영",
                    "year": 2025,
                    "batting_stats": {
                        "player_name": "김도영",
                        "avg": 0.312,
                        "ops": 0.955,
                        "home_runs": 16,
                        "rbi": 48,
                    },
                },
                message="ok",
            ),
            ToolResult(
                success=True,
                data={
                    "stat_name": "home_runs",
                    "year": 2025,
                    "leaderboard": [
                        {"player_name": "김도영", "home_runs": 16},
                        {"player_name": "오스틴", "home_runs": 15},
                    ],
                },
                message="ok",
            ),
        ],
        chat_mode=True,
    )

    assert answer is not None
    assert "김도영 기록은 현재 이렇게 보입니다." in answer
    assert "리그 홈런 리더보드에서는 현재 1위" in answer


def test_player_comparison_answer_mentions_approach_and_power() -> None:
    agent = _build_agent_for_fast_path()

    answer = agent._build_player_comparison_answer(
        "김도영과 문보경을 비교할 때 타석 접근법과 장타 생산 방식 차이를 설명해줘",
        [
            ToolResult(
                success=True,
                data={
                    "comparison_type": "2025년 시즌",
                    "player1": {
                        "name": "김도영",
                        "data": {
                            "batting_stats": {
                                "player_name": "김도영",
                                "plate_appearances": 520,
                                "walks": 62,
                                "strikeouts": 88,
                                "obp": 0.401,
                                "slg": 0.571,
                                "ops": 0.972,
                                "home_runs": 28,
                                "doubles": 24,
                            }
                        },
                    },
                    "player2": {
                        "name": "문보경",
                        "data": {
                            "batting_stats": {
                                "player_name": "문보경",
                                "plate_appearances": 540,
                                "walks": 48,
                                "strikeouts": 104,
                                "obp": 0.367,
                                "slg": 0.522,
                                "ops": 0.889,
                                "home_runs": 22,
                                "doubles": 31,
                            }
                        },
                    },
                    "analysis": {
                        "summary": "선수1이 4개 지표에서 우세, 선수2가 1개 지표에서 우세"
                    },
                },
                message="ok",
            )
        ],
        chat_mode=True,
    )

    assert answer is not None
    assert "타석 접근" in answer
    assert "장타 생산" in answer
    assert "홈런 생산" in answer


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
