from __future__ import annotations

from app.agents.baseball_agent import (
    BaseballStatisticsAgent,
    _leaderboard_data_quality_issue,
)
from app.agents.chat_renderers import ChatRendererRegistry


def _rows(values):
    return [
        {"player_name": f"선수{i}", "team_name": "팀", "stat_value": v}
        for i, v in enumerate(values, start=1)
    ]


def _data(stat, values, **extra):
    data = {
        "found": True,
        "year": 2025,
        "stat_name": stat,
        "position": "batting",
        "leaderboard": _rows(values),
    }
    data.update(extra)
    return data


def test_healthy_counting_leaderboard_passes() -> None:
    assert (
        _leaderboard_data_quality_issue(_data("home_runs", [27, 26, 19, 16, 13, 11]))
        is None
    )


def test_zero_in_top_five_is_flagged() -> None:
    issue = _leaderboard_data_quality_issue(_data("home_runs", [0, 0, 0, 0, 0, 0]))
    assert issue is not None
    assert "5명" in issue or "상위" in issue


def test_zero_outside_top_five_is_not_flagged() -> None:
    assert (
        _leaderboard_data_quality_issue(_data("home_runs", [27, 26, 19, 16, 13, 0]))
        is None
    )


def test_rate_stat_zero_is_not_flagged() -> None:
    assert (
        _leaderboard_data_quality_issue(_data("avg", [0.0, 0.0, 0.0, 0.0, 0.0])) is None
    )


def test_ascending_counting_query_is_not_flagged() -> None:
    data = _data("home_runs", [0, 0, 0, 0, 0], sort_order="ASC")
    assert _leaderboard_data_quality_issue(data) is None


def test_too_few_rows_is_not_flagged() -> None:
    assert _leaderboard_data_quality_issue(_data("home_runs", [0, 0, 0])) is None


def _agent() -> BaseballStatisticsAgent:
    agent = BaseballStatisticsAgent.__new__(BaseballStatisticsAgent)
    agent._resolve_chat_intent = lambda query, entity_filter: None
    agent.chat_renderer_registry = ChatRendererRegistry().bind(agent)
    return agent


def test_chat_answer_reports_data_check_instead_of_ranking() -> None:
    answer = _agent()._build_leaderboard_chat_answer(
        "2025년 KBO 홈런 1위 선수는 누구인가요?",
        _data("home_runs", [0, 0, 0, 0, 0, 0]),
    )
    assert answer is not None
    assert "확인이 필요" in answer
    assert "순위를 확정할 수 없습니다" in answer
    assert "1위는" not in answer


def test_chat_answer_unchanged_for_healthy_data() -> None:
    answer = _agent()._build_leaderboard_chat_answer(
        "2025년 KBO 홈런 1위 선수는 누구인가요?",
        _data("home_runs", [27, 26, 19, 16, 13, 11]),
    )
    assert answer is not None
    assert "27개" in answer
    assert "확인이 필요" not in answer
