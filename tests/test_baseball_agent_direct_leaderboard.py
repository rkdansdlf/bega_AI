from __future__ import annotations

from app.agents.baseball_agent import BaseballStatisticsAgent
from app.agents.chat_renderers import ChatRendererRegistry
from app.agents.tool_caller import ToolResult


def _agent() -> BaseballStatisticsAgent:
    # 런타임 생성자는 직접 호출을 막으므로, 리더보드 렌더링에 필요한 부분만 붙인다.
    agent = BaseballStatisticsAgent.__new__(BaseballStatisticsAgent)
    agent._resolve_chat_intent = lambda query, entity_filter: None
    agent.chat_renderer_registry = ChatRendererRegistry().bind(agent)
    return agent


def _leaderboard_result() -> ToolResult:
    data = {
        "found": True,
        "year": 2025,
        "stat_name": "home_runs",
        "position": "batting",
        "total_qualified_players": 10,
        "leaderboard": [
            {
                "player_id": 68525,
                "player_name": "선수A",
                "team_code": "LT",
                "team_name": "롯데 자이언츠",
                "stat_value": 27,
            },
            {
                "player_id": 68106,
                "player_name": "선수B",
                "team_code": "LG",
                "team_name": "LG 트윈스",
                "stat_value": 26,
            },
        ],
    }
    return ToolResult(success=True, data=data, message="leaderboard ok")


def test_direct_leaderboard_answer_uses_tool_values_for_chat_persona() -> None:
    agent = _agent()
    answer = agent._build_direct_leaderboard_answer(
        "2025년 KBO 홈런 1위 선수는 누구인가요?",
        [_leaderboard_result()],
        context={"persona": "chat"},
    )
    assert answer is not None
    assert "27개" in answer
    assert "26개" in answer
    assert "0개" not in answer


def test_direct_leaderboard_answer_skipped_for_non_chat_persona() -> None:
    agent = _agent()
    answer = agent._build_direct_leaderboard_answer(
        "2025년 KBO 홈런 1위 선수는 누구인가요?",
        [_leaderboard_result()],
        context={"persona": "analyst"},
    )
    assert answer is None


def test_direct_leaderboard_answer_skipped_without_leaderboard_data() -> None:
    agent = _agent()
    failed = ToolResult(success=False, data={}, message="failed")
    answer = agent._build_direct_leaderboard_answer(
        "2025년 KBO 홈런 1위 선수는 누구인가요?",
        [failed],
        context={"persona": "chat"},
    )
    assert answer is None
