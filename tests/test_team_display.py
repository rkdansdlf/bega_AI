from app.tools.team_display import replace_team_codes, resolve_team_display_name


def test_resolve_team_display_name_uses_default_resolver_for_legacy_code() -> None:
    assert resolve_team_display_name("SK") == "SK 와이번스"


def test_resolve_team_display_name_prefers_custom_resolver() -> None:
    assert (
        resolve_team_display_name(
            "SSG",
            team_name_resolver=lambda team_code: "커스텀 랜더스",
        )
        == "커스텀 랜더스"
    )


def test_replace_team_codes_recursively_rewrites_nested_team_fields() -> None:
    replaced = replace_team_codes(
        {
            "home_team": "SK",
            "matchup": {"away_team": "HH"},
            "games": [{"winning_team": "LG"}],
        }
    )

    assert replaced == {
        "home_team": "SK 와이번스",
        "matchup": {"away_team": "한화 이글스"},
        "games": [{"winning_team": "LG 트윈스"}],
    }
