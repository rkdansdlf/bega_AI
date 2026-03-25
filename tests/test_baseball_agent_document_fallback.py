from app.agents.baseball_agent import BaseballStatisticsAgent
from app.agents.tool_caller import ToolCall, ToolResult


def _build_agent() -> BaseballStatisticsAgent:
    return BaseballStatisticsAgent.__new__(BaseballStatisticsAgent)


def test_document_fallback_disabled_for_player_lookup_stat_miss() -> None:
    agent = _build_agent()

    should_fallback = agent._should_use_document_fallback(
        "오스틴드라군 2025 홈런 몇 개야?",
        {"intent": "player_lookup", "grounding_mode": "structured_kbo"},
        [],
    )

    assert should_fallback is False


def test_document_fallback_kept_for_baseball_explainer() -> None:
    agent = _build_agent()

    should_fallback = agent._should_use_document_fallback(
        "WHIP 뜻이 뭐야?",
        {"intent": "baseball_explainer", "grounding_mode": "baseball_explainer"},
        [],
    )

    assert should_fallback is True


def test_document_fallback_disabled_for_structured_player_comparison_miss() -> None:
    agent = _build_agent()

    should_fallback = agent._should_use_document_fallback(
        "김도영과 문보경을 비교할 때 타석 접근법과 장타 생산 방식 차이를 설명해줘.",
        {
            "intent": "baseball_explainer",
            "grounding_mode": "baseball_explainer",
            "tool_calls": [
                ToolCall(
                    "get_player_stats",
                    {"player_name": "김도영", "year": 2026},
                ),
                ToolCall(
                    "get_player_stats",
                    {"player_name": "문보경", "year": 2026},
                ),
            ],
        },
        [],
    )

    assert should_fallback is False


def test_validate_player_empty_miss_is_not_meaningful_tool_data() -> None:
    agent = _build_agent()

    assert (
        agent._is_meaningful_tool_data(
            {
                "player_name": "곽빈",
                "year": 2026,
                "exists": False,
                "found_players": [],
                "error": None,
            }
        )
        is False
    )


def test_validate_player_empty_miss_does_not_count_as_meaningful_tool_result() -> None:
    agent = _build_agent()

    assert (
        agent._has_meaningful_tool_results(
            [
                ToolResult(
                    success=True,
                    data={
                        "player_name": "곽빈",
                        "year": 2026,
                        "exists": False,
                        "found_players": [],
                        "error": None,
                    },
                    message="해당 선수를 찾을 수 없습니다.",
                )
            ]
        )
        is False
    )
