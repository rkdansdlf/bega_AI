from app.agents.chat_intent_router import ChatIntent, IntentDecision
from app.agents.chat_renderers import ChatRendererRegistry


class _StubAgent:
    def _detect_team_alias_from_query(self, query: str) -> str | None:
        query_lower = query.lower()
        if "lg" in query_lower or "엘지" in query:
            return "LG"
        if "kt" in query_lower:
            return "KT"
        return None

    def _format_deterministic_metric(self, value):
        if value in (None, ""):
            return "확인 불가"
        return str(value)

    def _format_team_display_name(self, team_name: str | None) -> str:
        if team_name == "LG":
            return "LG 트윈스"
        if team_name == "KT":
            return "KT 위즈"
        if team_name == "DB":
            return "두산 베어스"
        if team_name == "LT":
            return "롯데 자이언츠"
        if team_name == "KH":
            return "키움 히어로즈"
        return "확인 불가"

    def _normalize_chatbot_answer_text(self, text: str) -> str:
        return text


def test_render_low_data_team_analysis_uses_query_without_unboundlocalerror() -> None:
    registry = ChatRendererRegistry(_StubAgent())
    decision = IntentDecision(intent=ChatIntent.TEAM_ANALYSIS, subject_type="team")

    rendered = registry.render_low_data("LG 팀 OPS 어때?", [], decision)

    assert "LG 트윈스" in rendered
    assert "팀 OPS" in rendered


def test_render_low_data_generic_metric_fallback_uses_query_without_unboundlocalerror() -> (
    None
):
    registry = ChatRendererRegistry(_StubAgent())
    decision = IntentDecision(intent=ChatIntent.UNKNOWN, subject_type=None)

    rendered = registry.render_low_data("LG 팀 OPS 알려줘", [], decision)

    assert "LG 트윈스" in rendered
    assert "팀 OPS" in rendered


def test_render_game_flow_single_box_score_renders_narrative() -> None:
    registry = ChatRendererRegistry(_StubAgent())

    rendered = registry.render_game_flow(
        "2025년 5월 1일 경기 흐름 요약해줘",
        {
            "query_params": {"date": "2025-05-01"},
            "games": [
                {
                    "game_id": "20250501KTOB0",
                    "game_date": "2025-05-01",
                    "home_team": "DB",
                    "away_team": "KT",
                    "home_team_code": "DB",
                    "away_team_code": "KT",
                    "home_team_name": "두산 베어스",
                    "away_team_name": "KT 위즈",
                    "home_score": 3,
                    "away_score": 3,
                    "winning_team": None,
                    "box_score": {
                        "away_1": 1,
                        "home_1": 0,
                        "away_4": 0,
                        "home_4": 1,
                        "away_6": 0,
                        "home_6": 2,
                        "away_9": 2,
                        "home_9": 0,
                    },
                }
            ],
        },
    )

    assert rendered is not None
    assert "2025-05-01" in rendered
    assert "리드 체인지" in rendered
    assert "[META]" not in rendered
    assert "[출처]" not in rendered


def test_render_game_flow_multiple_box_scores_requests_matchup_clarification() -> None:
    registry = ChatRendererRegistry(_StubAgent())

    rendered = registry.render_game_flow(
        "2025년 5월 1일 경기 흐름 요약해줘",
        {
            "query_params": {"date": "2025-05-01"},
            "games": [
                {
                    "game_id": "20250501KTOB0",
                    "game_date": "2025-05-01",
                    "home_team_code": "DB",
                    "away_team_code": "KT",
                    "home_score": 3,
                    "away_score": 3,
                    "box_score": {"away_1": 1, "home_1": 0},
                },
                {
                    "game_id": "20250501LTWO0",
                    "game_date": "2025-05-01",
                    "home_team_code": "KH",
                    "away_team_code": "LT",
                    "home_score": 0,
                    "away_score": 5,
                    "box_score": {"away_1": 2, "home_1": 0},
                },
            ],
        },
    )

    assert rendered is not None
    assert "어떤 경기를" in rendered
    assert "KT 위즈 3-3 두산 베어스" in rendered
    assert "OB 베어스" not in rendered
    assert "우리 히어로즈" not in rendered


def test_render_game_flow_filters_multiple_games_by_query_team() -> None:
    registry = ChatRendererRegistry(_StubAgent())

    rendered = registry.render_game_flow(
        "2025년 5월 1일 KT 경기 흐름 요약해줘",
        {
            "query_params": {"date": "2025-05-01"},
            "games": [
                {
                    "game_id": "20250501KTOB0",
                    "game_date": "2025-05-01",
                    "home_team_code": "DB",
                    "away_team_code": "KT",
                    "home_team_name": "두산 베어스",
                    "away_team_name": "KT 위즈",
                    "home_score": 3,
                    "away_score": 3,
                    "winning_team": None,
                    "box_score": {
                        "away_1": 1,
                        "home_1": 0,
                        "home_4": 1,
                        "home_6": 2,
                        "away_9": 2,
                    },
                },
                {
                    "game_id": "20250501LTWO0",
                    "game_date": "2025-05-01",
                    "home_team_code": "KH",
                    "away_team_code": "LT",
                    "home_score": 0,
                    "away_score": 5,
                    "box_score": {"away_1": 2, "home_1": 0},
                },
            ],
        },
    )

    assert rendered is not None
    assert "어떤 경기를" not in rendered
    assert "KT 위즈" in rendered
    assert "OB 베어스" not in rendered
