from scripts.apply_manual_starters import (
    GameStarterSnapshot,
    build_update_plan,
    parse_manual_starter_inputs,
)


def _existing(**overrides):
    base = {
        "game_id": "20260424KTSK0",
        "game_date": "2026-04-24",
        "game_status": "SCHEDULED",
        "home_team_id": "SSG",
        "away_team_id": "KT",
        "home_pitcher": "",
        "away_pitcher": "",
    }
    base.update(overrides)
    return GameStarterSnapshot(**base)


def test_parse_manual_starter_inputs_skips_empty_template_row():
    inputs, issues = parse_manual_starter_inputs(
        [
            {
                "game_id": "20260424KTSK0",
                "home_pitcher": "",
                "away_pitcher": "",
            }
        ]
    )

    assert inputs == []
    assert [(item.severity, item.code) for item in issues] == [
        ("warning", "empty_starter_pair")
    ]


def test_parse_manual_starter_inputs_rejects_incomplete_pair():
    inputs, issues = parse_manual_starter_inputs(
        [
            {
                "game_id": "20260424KTSK0",
                "home_pitcher": "김광현",
                "away_pitcher": "",
            }
        ]
    )

    assert inputs == []
    assert [(item.severity, item.code) for item in issues] == [
        ("error", "incomplete_starter_pair")
    ]


def test_build_update_plan_for_missing_scheduled_starters():
    inputs, parse_issues = parse_manual_starter_inputs(
        [
            {
                "game_id": "20260424KTSK0",
                "game_date": "2026-04-24",
                "home_team_id": "SSG",
                "away_team_id": "KT",
                "home_pitcher": "김광현",
                "away_pitcher": "고영표",
            }
        ]
    )
    plans, plan_issues = build_update_plan(
        inputs,
        {"20260424KTSK0": _existing()},
    )

    assert parse_issues == []
    assert plan_issues == []
    assert len(plans) == 1
    assert plans[0].changed is True
    assert plans[0].new_home_pitcher == "김광현"
    assert plans[0].new_away_pitcher == "고영표"


def test_build_update_plan_rejects_team_mismatch():
    inputs, _issues = parse_manual_starter_inputs(
        [
            {
                "game_id": "20260424KTSK0",
                "home_team_id": "HH",
                "away_team_id": "KT",
                "home_pitcher": "김광현",
                "away_pitcher": "고영표",
            }
        ]
    )
    plans, issues = build_update_plan(
        inputs,
        {"20260424KTSK0": _existing()},
    )

    assert plans == []
    assert [(item.severity, item.code) for item in issues] == [
        ("error", "home_team_mismatch")
    ]


def test_build_update_plan_requires_overwrite_flag():
    inputs, _issues = parse_manual_starter_inputs(
        [
            {
                "game_id": "20260424KTSK0",
                "home_team_id": "SSG",
                "away_team_id": "KT",
                "home_pitcher": "김광현",
                "away_pitcher": "고영표",
            }
        ]
    )
    plans, issues = build_update_plan(
        inputs,
        {
            "20260424KTSK0": _existing(
                home_pitcher="다른선발",
                away_pitcher="고영표",
            )
        },
    )

    assert plans == []
    assert [(item.severity, item.code) for item in issues] == [
        ("error", "starter_overwrite_requires_flag")
    ]


def test_build_update_plan_allows_overwrite_with_flag():
    inputs, _issues = parse_manual_starter_inputs(
        [
            {
                "game_id": "20260424KTSK0",
                "home_team_id": "SSG",
                "away_team_id": "KT",
                "home_pitcher": "김광현",
                "away_pitcher": "고영표",
            }
        ]
    )
    plans, issues = build_update_plan(
        inputs,
        {
            "20260424KTSK0": _existing(
                home_pitcher="다른선발",
                away_pitcher="고영표",
            )
        },
        allow_overwrite=True,
    )

    assert issues == []
    assert len(plans) == 1
    assert plans[0].changed is True
