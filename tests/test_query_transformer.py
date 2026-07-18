"""
QueryTransformer 단위 테스트.
expand_query_with_rules(), 중복 제거, 변형 수 제한을 검증한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.query_transformer import QueryTransformer, QueryVariation
from app.core.entity_extractor import EntityFilter


# ── 기존 통합 테스트 (호환 유지) ──────────────────────────────────────────────

def test_expand_query_with_rules_respects_variation_cap_and_keeps_original() -> None:
    transformer = QueryTransformer()
    entity_filter = SimpleNamespace(
        stat_type="ops",
        team_id="LG",
        player_name="김현수",
        position_type="batter",
    )
    variations = transformer.expand_query_with_rules(
        "김현수 OPS 알려줘",
        entity_filter,
        max_variations=3,
    )
    assert len(variations) == 3
    assert variations[0].query == "김현수 OPS 알려줘"
    assert variations[0].variation_type == "original"
    assert len({variation.query for variation in variations}) == len(variations)


# ── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def transformer() -> QueryTransformer:
    return QueryTransformer()


def make_filter(**kwargs) -> EntityFilter:
    f = EntityFilter()
    for key, value in kwargs.items():
        setattr(f, key, value)
    return f


# ── TestQueryVariation ────────────────────────────────────────────────────────

class TestQueryVariation:
    def test_defaults(self):
        v = QueryVariation("홈런왕은?", "original")
        assert v.query == "홈런왕은?"
        assert v.variation_type == "original"
        assert v.weight == 1.0

    def test_custom_weight(self):
        v = QueryVariation("ERA 낮은 투수", "statistical", 0.8)
        assert v.weight == 0.8

    def test_variation_type_strings(self):
        for vtype in ["original", "expanded", "statistical", "contextual"]:
            v = QueryVariation("test", vtype, 1.0)
            assert v.variation_type == vtype


# ── TestExpandQueryWithRules ──────────────────────────────────────────────────

class TestExpandQueryWithRules:
    def test_always_includes_original(self, transformer):
        ef = make_filter()
        variations = transformer.expand_query_with_rules("2024년 타율 1위", ef)
        assert variations[0].query == "2024년 타율 1위"
        assert variations[0].variation_type == "original"
        assert variations[0].weight == 1.0

    def test_empty_query_returns_zero_or_one(self, transformer):
        # 빈 쿼리는 dedup에서 공백 제거 후 0개 또는 original 1개 반환
        ef = make_filter()
        variations = transformer.expand_query_with_rules("", ef)
        assert len(variations) <= 1

    # 통계 확장 ---

    def test_ops_stat_generates_statistical_variations(self, transformer):
        ef = make_filter(stat_type="ops")
        variations = transformer.expand_query_with_rules("OPS 순위", ef)
        types = [v.variation_type for v in variations]
        assert "statistical" in types

    def test_era_stat_generates_statistical_variations(self, transformer):
        ef = make_filter(stat_type="era")
        variations = transformer.expand_query_with_rules("ERA 낮은 투수", ef)
        types = [v.variation_type for v in variations]
        assert "statistical" in types

    def test_home_runs_stat_generates_statistical_variations(self, transformer):
        ef = make_filter(stat_type="home_runs")
        variations = transformer.expand_query_with_rules("홈런 1위", ef)
        types = [v.variation_type for v in variations]
        assert "statistical" in types

    def test_war_stat_generates_statistical_variations(self, transformer):
        ef = make_filter(stat_type="war")
        variations = transformer.expand_query_with_rules("WAR 높은 선수", ef)
        types = [v.variation_type for v in variations]
        assert "statistical" in types

    def test_unknown_stat_no_statistical_expansion(self, transformer):
        ef = make_filter(stat_type="babip")  # 매핑에 없는 지표
        variations = transformer.expand_query_with_rules("BABIP 순위", ef)
        stat_vars = [v for v in variations if v.variation_type == "statistical"]
        assert len(stat_vars) == 0

    # 컨텍스트 확장 ---

    def test_team_id_generates_contextual_variations(self, transformer):
        ef = make_filter(team_id="KIA")
        variations = transformer.expand_query_with_rules("타율 1위", ef)
        types = [v.variation_type for v in variations]
        assert "contextual" in types

    def test_player_name_generates_contextual_variations(self, transformer):
        ef = make_filter(player_name="류현진")
        variations = transformer.expand_query_with_rules("선수 기록", ef)
        types = [v.variation_type for v in variations]
        assert "contextual" in types

    def test_team_and_player_both_generate_contextual(self, transformer):
        ef = make_filter(team_id="LG", player_name="오지환")
        variations = transformer.expand_query_with_rules("수비 기록", ef)
        contextual = [v for v in variations if v.variation_type == "contextual"]
        assert len(contextual) >= 2

    def test_contextual_includes_full_team_name(self, transformer):
        ef = make_filter(team_id="KIA")
        variations = transformer.expand_query_with_rules("투수 ERA", ef)
        queries = [v.query for v in variations]
        assert any("KIA 타이거즈" in q for q in queries)

    # 랭킹 확장 ---

    def test_ranking_query_generates_expanded(self, transformer):
        ef = make_filter()
        variations = transformer.expand_query_with_rules("홈런 순위 알려줘", ef)
        types = [v.variation_type for v in variations]
        assert "expanded" in types

    def test_non_ranking_query_no_expanded(self, transformer):
        ef = make_filter()
        variations = transformer.expand_query_with_rules("류현진 커리어 통계", ef)
        expanded = [v for v in variations if v.variation_type == "expanded"]
        assert len(expanded) == 0

    def test_pitcher_position_ranking_expansion(self, transformer):
        ef = make_filter(position_type="pitcher")
        variations = transformer.expand_query_with_rules("순위 투수", ef)
        queries = [v.query for v in variations]
        assert any("ERA" in q or "WHIP" in q for q in queries)

    def test_batter_position_ranking_expansion(self, transformer):
        ef = make_filter(position_type="batter")
        variations = transformer.expand_query_with_rules("타율 랭킹", ef)
        queries = [v.query for v in variations]
        assert any("OPS" in q or "타율" in q for q in queries)


# ── TestDuplicateRemoval ──────────────────────────────────────────────────────

class TestDuplicateRemoval:
    def test_no_duplicate_queries(self, transformer):
        ef = make_filter(stat_type="ops", team_id="LG")
        variations = transformer.expand_query_with_rules("OPS 순위", ef)
        queries = [v.query for v in variations]
        assert len(queries) == len(set(queries))

    def test_empty_string_filtered(self, transformer):
        ef = make_filter()
        variations = transformer.expand_query_with_rules("   ", ef)
        for v in variations:
            assert v.query.strip() != ""

    def test_dedup_preserves_order(self, transformer):
        ef = make_filter()
        variations = transformer.expand_query_with_rules("2024년 MVP", ef)
        queries = [v.query for v in variations]
        unique = list(dict.fromkeys(queries))
        assert queries == unique


# ── TestVariationCap ──────────────────────────────────────────────────────────

class TestVariationCap:
    def test_default_cap_is_5(self, transformer):
        ef = make_filter(stat_type="ops", team_id="KIA", position_type="batter")
        variations = transformer.expand_query_with_rules("OPS 순위", ef)
        assert len(variations) <= 5

    def test_cap_1_returns_only_original(self, transformer):
        ef = make_filter(stat_type="era", team_id="LG", position_type="pitcher")
        variations = transformer.expand_query_with_rules("ERA 1위", ef, max_variations=1)
        assert len(variations) == 1
        assert variations[0].variation_type == "original"

    def test_cap_2_returns_at_most_2(self, transformer):
        ef = make_filter(stat_type="home_runs", team_id="SS", position_type="batter")
        variations = transformer.expand_query_with_rules("홈런 순위", ef, max_variations=2)
        assert len(variations) <= 2

    def test_cap_10_never_exceeded(self, transformer):
        ef = make_filter(stat_type="war", team_id="NC", player_name="손아섭", position_type="batter")
        variations = transformer.expand_query_with_rules("WAR 순위", ef, max_variations=10)
        assert len(variations) <= 10

    def test_negative_cap_treated_as_1(self, transformer):
        ef = make_filter()
        variations = transformer.expand_query_with_rules("타율", ef, max_variations=-5)
        assert len(variations) >= 1


# ── TestGetFullTeamName ───────────────────────────────────────────────────────

class TestGetFullTeamName:
    @pytest.mark.parametrize("team_id,expected", [
        ("KIA", "KIA 타이거즈"),
        ("LG", "LG 트윈스"),
        ("DB", "두산 베어스"),
        ("KT", "KT 위즈"),
        ("NC", "NC 다이노스"),
        ("SSG", "SSG 랜더스"),
        ("SK", "SSG 랜더스"),
        ("KH", "키움 히어로즈"),
    ])
    def test_known_team_ids(self, transformer, team_id, expected):
        assert transformer._get_full_team_name(team_id) == expected

    def test_unknown_team_id_returns_as_is(self, transformer):
        assert transformer._get_full_team_name("UNKNOWN") == "UNKNOWN"


# ── TestIsRankingQuery ────────────────────────────────────────────────────────

class TestIsRankingQuery:
    @pytest.mark.parametrize("query", [
        "타율 순위는?",
        "홈런 랭킹 1위",
        "OPS 최고 선수",
        "ERA TOP5",
        "베스트 타자",
        "톱 투수 순위",
    ])
    def test_ranking_queries_detected(self, transformer, query):
        assert transformer._is_ranking_query(query) is True

    @pytest.mark.parametrize("query", [
        "류현진 커리어 통계",
        "KIA 팀 분석",
        "두산 베어스 2024년",
        "ERA가 뭐야?",
    ])
    def test_non_ranking_queries(self, transformer, query):
        assert transformer._is_ranking_query(query) is False


# ── TestWeightOrdering ────────────────────────────────────────────────────────

class TestWeightOrdering:
    def test_original_has_highest_weight(self, transformer):
        ef = make_filter(stat_type="era", team_id="LG", position_type="pitcher")
        variations = transformer.expand_query_with_rules("ERA 순위", ef)
        original_weight = next(v.weight for v in variations if v.variation_type == "original")
        for v in variations:
            assert v.weight <= original_weight

    def test_all_weights_in_valid_range(self, transformer):
        ef = make_filter(stat_type="ops")
        variations = transformer.expand_query_with_rules("OPS 분석", ef)
        for v in variations:
            assert 0.0 < v.weight <= 1.0
