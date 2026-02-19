from types import SimpleNamespace

from app.core.rag import _build_rrf_keyword


def test_rrf_keyword_includes_entity_and_intent_terms() -> None:
    entity_filter = SimpleNamespace(
        player_name="김도영",
        team_id="KIA",
        season_year=2024,
        stat_type="home_runs",
        award_type=None,
        movement_type=None,
        position_type="batter",
        game_date=None,
    )

    keyword = _build_rrf_keyword("김도영 2024년 홈런 순위 알려줘", entity_filter)

    assert "김도영" in keyword
    assert "KIA 타이거즈" in keyword
    assert "2024" in keyword
    assert "home_runs" in keyword
    assert "batter" in keyword


def test_rrf_keyword_adds_query_support_tokens_for_entity_only_case() -> None:
    entity_filter = SimpleNamespace(
        player_name="김도영",
        team_id="KIA",
        season_year=None,
        stat_type=None,
        award_type=None,
        movement_type=None,
        position_type=None,
        game_date=None,
    )

    keyword = _build_rrf_keyword("김도영 슬럼프 원인 분석", entity_filter)

    assert "김도영" in keyword
    assert "KIA 타이거즈" in keyword
    assert "슬럼프" in keyword


def test_rrf_keyword_falls_back_to_query_when_entity_absent() -> None:
    keyword = _build_rrf_keyword("야구 규칙 설명해줘", None)
    assert "야구" in keyword


def test_rrf_keyword_changes_with_multi_query_variations() -> None:
    entity_filter = SimpleNamespace(
        player_name="김도영",
        team_id="KIA",
        season_year=2024,
        stat_type=None,
        award_type=None,
        movement_type=None,
        position_type=None,
        game_date=None,
    )

    keyword_hr = _build_rrf_keyword("김도영 홈런 추이", entity_filter)
    keyword_ops = _build_rrf_keyword("김도영 OPS 추이", entity_filter)

    assert keyword_hr != keyword_ops
    assert "홈런" in keyword_hr
    assert "OPS" in keyword_ops
