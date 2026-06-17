from app.agents.baseball_agent import BaseballStatisticsAgent
from app.agents.chat_intent_router import ChatIntent, ChatIntentRouter, IntentDecision
from app.agents.tool_caller import ToolCall, ToolResult
from app.core.entity_extractor import extract_entities_from_query
from app.tools.database_query import DatabaseQueryTool


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


def test_named_team_comparison_query_uses_team_comparison_fast_path() -> None:
    agent = _build_agent_for_fast_path()
    query = "2026년 LG와 KT를 비교해줘?"

    plan = agent._build_reference_fast_path_plan(
        query,
        extract_entities_from_query(query),
    )

    assert plan is not None
    assert plan["planner_mode"] == "fast_path"
    assert [call.tool_name for call in plan["tool_calls"]] == [
        "get_team_comparison"
    ]
    assert plan["tool_calls"][0].parameters == {
        "team1": "LG",
        "team2": "KT",
        "year": 2026,
        "recent_limit": 10,
    }


def test_db_fast_path_candidate_team_metric_queries_route_to_expected_metrics() -> None:
    agent = _build_agent_for_fast_path()
    cases = {
        "2026년 도루가 많은 팀은 어디야?": ("stolen_bases", "DESC"),
        "2026년 실책이 적은 팀은 어디야?": ("errors", "ASC"),
        "2026년 팀별 ERA는?": ("era", "ASC"),
        "2026년 팀별 경기당 평균 득점은?": ("runs_per_game", "DESC"),
        "2026년 팀별 경기당 평균 실점은?": ("runs_allowed_per_game", "ASC"),
        "2026년 타선 비교해줘.": ("ops", "DESC"),
        "2026년 수비 비교해줘.": ("fielding_pct", "DESC"),
        "2026년 주루 비교해줘.": ("stolen_bases", "DESC"),
        "2026년 가장 기복이 큰 팀은?": ("run_margin_volatility", "DESC"),
        "2026년 불펜 소모가 많은 팀은 어디야?": ("bullpen_share", "DESC"),
        "2026년 불펜 비교해줘.": ("bullpen_era", "ASC"),
        "2026년 선발진 비교해줘.": ("starter_qs_rate", "DESC"),
    }

    for query, (metric_name, sort_order) in cases.items():
        plan = agent._build_reference_fast_path_plan(
            query,
            extract_entities_from_query(query),
        )

        assert plan is not None, query
        assert [call.tool_name for call in plan["tool_calls"]] == [
            "get_team_metric_leaderboard"
        ]
        assert plan["tool_calls"][0].parameters["metric_name"] == metric_name
        assert plan["tool_calls"][0].parameters["sort_order"] == sort_order


def test_recent_flow_comparison_routes_to_team_form_table() -> None:
    agent = _build_agent_for_fast_path()
    query = "2026년 최근 흐름 비교해줘."

    plan = agent._build_reference_fast_path_plan(
        query,
        extract_entities_from_query(query),
    )

    assert plan is not None
    assert [call.tool_name for call in plan["tool_calls"]] == ["get_team_form_table"]
    assert plan["tool_calls"][0].parameters["form_type"] == "recent"


def test_get_team_comparison_combines_rank_metrics_and_recent_form() -> None:
    tool = DatabaseQueryTool.__new__(DatabaseQueryTool)
    tool.get_team_code = lambda team, year=None: team
    tool.get_team_name = lambda team, year=None: f"{team} 트윈스" if team == "LG" else f"{team} 위즈"
    tool.get_team_variants = lambda team, year=None: [team]
    tool.get_team_form_table = lambda **kwargs: {
        "form_rows": [
            {
                "team_code": "LG",
                "games": 10,
                "wins": 6,
                "losses": 4,
                "draws": 0,
                "win_pct": 0.6,
            },
            {
                "team_code": "KT",
                "games": 10,
                "wins": 5,
                "losses": 5,
                "draws": 0,
                "win_pct": 0.5,
            },
        ]
    }
    tool.get_team_season_rank = lambda team, year: {
        "found": True,
        "rank": 1 if team == "LG" else 2,
        "wins": 30,
        "losses": 20,
        "draws": 1,
        "win_pct": 0.6,
        "as_of_date": "2026-05-31",
    }
    tool.get_team_advanced_metrics = lambda team, year: {
        "found": True,
        "metrics": {
            "batting": {"ops": 0.755, "avg": 0.274, "total_hr": 52},
            "pitching": {"avg_era": 3.71, "era_rank": "1위", "qs_rate": "52.0%"},
        },
        "fatigue_index": {
            "bullpen_share": "38.1%",
            "bullpen_load_rank": "4위 (높을수록 과부하)",
        },
    }

    result = tool.get_team_comparison("LG", "KT", 2026)

    assert result["found"] is True
    assert len(result["teams"]) == 2
    assert result["teams"][0]["rank"] == 1
    assert result["teams"][0]["ops"] == 0.755
    assert result["teams"][0]["recent"]["wins"] == 6
    assert result["teams"][1]["team_code"] == "KT"


def test_team_comparison_renderer_mentions_db_basis_and_core_metrics() -> None:
    agent = _build_agent_for_fast_path()

    answer = agent._build_team_comparison_chat_answer(
        {
            "year": 2026,
            "team1": "LG",
            "team2": "KT",
            "teams": [
                {
                    "team_name": "LG 트윈스",
                    "rank": 1,
                    "wins": 30,
                    "losses": 20,
                    "draws": 1,
                    "win_pct": 0.6,
                    "ops": 0.755,
                    "era": 3.71,
                    "qs_rate": "52.0%",
                    "bullpen_share": "38.1%",
                    "recent": {
                        "games": 10,
                        "wins": 6,
                        "losses": 4,
                        "draws": 0,
                        "win_pct": 0.6,
                    },
                },
                {
                    "team_name": "KT 위즈",
                    "rank": 2,
                    "wins": 29,
                    "losses": 21,
                    "draws": 1,
                    "win_pct": 0.58,
                    "ops": 0.732,
                    "era": 3.95,
                    "qs_rate": "49.0%",
                    "bullpen_share": "40.0%",
                    "recent": {
                        "games": 10,
                        "wins": 5,
                        "losses": 5,
                        "draws": 0,
                        "win_pct": 0.5,
                    },
                },
            ],
        }
    )

    assert answer is not None
    assert "DB 기준" in answer
    assert "LG 트윈스" in answer
    assert "KT 위즈" in answer
    assert "OPS" in answer
    assert "불펜 비중" in answer


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


def test_team_metric_leaderboard_answer_is_natural_chat() -> None:
    agent = _build_agent_for_fast_path()

    answer = agent._build_team_metric_leaderboard_chat_answer(
        {
            "year": 2026,
            "metric_name": "home_runs",
            "metric_label": "홈런",
            "sort_order": "DESC",
            "team_metric_leaderboard": [
                {"team_name": "LG", "stat_value": 42, "details": {"games": 30}},
                {"team_name": "KT", "stat_value": 38, "details": {"games": 30}},
            ],
        }
    )

    assert answer is not None
    assert "2026년 팀별 홈런" in answer
    assert "1위 LG" in answer
    assert "MANUAL_BASEBALL_DATA_REQUIRED" not in answer


def test_team_form_table_answer_renders_home_away_split() -> None:
    agent = _build_agent_for_fast_path()

    answer = agent._build_team_form_table_chat_answer(
        {
            "year": 2026,
            "form_type": "home_away",
            "form_rows": [
                {
                    "team_name": "LG",
                    "home_win_pct": 0.667,
                    "away_win_pct": 0.571,
                }
            ],
        }
    )

    assert answer is not None
    assert "홈/원정 승률" in answer
    assert "홈 승률 0.667" in answer
    assert "원정 승률 0.571" in answer


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


def test_player_stats_fast_path_renders_without_llm() -> None:
    agent = _build_agent_for_fast_path()

    answer = agent._build_fast_path_answer(
        "2026년 구자욱 시즌 성적 핵심만 알려줘.",
        [
            ToolResult(
                success=True,
                data={
                    "player_name": "구자욱",
                    "year": 2026,
                    "batting_stats": {
                        "team_name": "삼성 라이온즈",
                        "avg": 0.333,
                        "ops": 0.705,
                        "home_runs": 0,
                        "rbi": 25,
                    },
                    "pitching_stats": None,
                    "found": True,
                },
                message="ok",
            )
        ],
        chat_mode=True,
    )

    assert answer is not None
    assert "2026년" in answer
    assert "구자욱" in answer
    assert "타율 0.333" in answer
    assert "출처" not in answer
    assert "다른 출처" not in answer


def test_big_game_team_fast_path_avoids_audit_failure_fallback_phrase() -> None:
    agent = _build_agent_for_fast_path()

    answer = agent._build_fast_path_answer(
        "LG 큰 경기만 가면 흔들리는 구간이 어디인지 콕 집어줘.",
        [
            ToolResult(
                success=True,
                data={
                    "team_name": "LG",
                    "year": 2026,
                    "top_batters": [
                        {
                            "player_name": "오스틴",
                            "ops": 0.931,
                            "home_runs": 12,
                        }
                    ],
                    "top_pitchers": [],
                },
                message="ok",
            ),
            ToolResult(
                success=True,
                data={
                    "team_name": "LG",
                    "year": 2026,
                    "metrics": {
                        "batting": {
                            "ops": 0.781,
                            "avg": 0.281,
                            "total_hr": 39,
                        },
                        "pitching": {
                            "avg_era": 3.86,
                            "qs_rate": 0.54,
                            "era_rank": "3위",
                        },
                    },
                    "rankings": {"batting_ops": "2위"},
                    "fatigue_index": {
                        "bullpen_share": "41.2%",
                        "bullpen_load_rank": "4위",
                    },
                    "league_averages": {"bullpen_share": "39.0%"},
                },
                message="ok",
            ),
        ],
        chat_mode=True,
    )

    assert answer is not None
    assert "클러치/큰 경기" in answer
    assert "최근 흐름·선발·불펜·타선 지표 중심" in answer
    assert "단정하기 어렵" not in answer
    assert "자료만으로는" not in answer
    assert "현재 연결된 자료" not in answer


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
