from app.agents.baseball_agent import BaseballStatisticsAgent


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
