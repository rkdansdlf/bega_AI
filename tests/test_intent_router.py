from app.ml.intent_router import rule_intent


def test_rule_intent_routes_box_score_questions_to_game_lookup() -> None:
    assert rule_intent("2025년 5월 1일 경기 박스스코어 알려줘") == "game_lookup"


def test_rule_intent_keeps_general_stats_lookup() -> None:
    assert rule_intent("2024년 평균자책점 1위 누구야?") == "stats_lookup"
