from __future__ import annotations

from types import SimpleNamespace

from app.core.query_transformer import QueryTransformer


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
