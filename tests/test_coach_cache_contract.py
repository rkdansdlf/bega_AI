from app.core.coach_cache_contract import (
    COACH_CACHE_PROMPT_VERSION,
    COACH_CACHE_SCHEMA_VERSION,
    build_coach_cache_identity,
)


def test_build_coach_cache_identity_uses_shared_contract_versions() -> None:
    cache_key, payload, starter_signature, lineup_signature = (
        build_coach_cache_identity(
            home_team_code="SS",
            away_team_code="NC",
            year=2025,
            game_type="REGULAR",
            focus=["matchup", "recent_form"],
            question_override=None,
            game_id="20250708SSNC0",
            league_type_code=0,
            stage_label="REGULAR",
            request_mode="manual_detail",
            game_status_bucket="COMPLETED",
            home_pitcher="후라도",
            away_pitcher="로건",
            lineup_players=["김지찬", "구자욱", "손아섭", "데이비슨"],
        )
    )

    assert cache_key
    assert payload["schema"] == COACH_CACHE_SCHEMA_VERSION
    assert payload["prompt_version"] == COACH_CACHE_PROMPT_VERSION
    assert payload["starter_signature"] == starter_signature
    assert payload["lineup_signature"] == lineup_signature


def test_build_coach_cache_identity_prefers_requested_signatures_only_when_both_exist() -> (
    None
):
    provided_key, provided_payload, provided_starter, provided_lineup = (
        build_coach_cache_identity(
            home_team_code="KT",
            away_team_code="SK",
            year=2025,
            game_type="REGULAR",
            focus=["matchup"],
            question_override=None,
            game_id="20250708KTSK0",
            league_type_code=0,
            stage_label="REGULAR",
            request_mode="manual_detail",
            game_status_bucket="COMPLETED",
            requested_starter_signature="starter:provided",
            requested_lineup_signature="lineup:provided",
            home_pitcher="쿠에바스",
            away_pitcher="김광현",
            lineup_players=["강백호", "로하스", "최정"],
        )
    )
    fallback_key, fallback_payload, fallback_starter, fallback_lineup = (
        build_coach_cache_identity(
            home_team_code="KT",
            away_team_code="SK",
            year=2025,
            game_type="REGULAR",
            focus=["matchup"],
            question_override=None,
            game_id="20250708KTSK0",
            league_type_code=0,
            stage_label="REGULAR",
            request_mode="manual_detail",
            game_status_bucket="COMPLETED",
            requested_starter_signature="starter:provided",
            requested_lineup_signature=None,
            home_pitcher="쿠에바스",
            away_pitcher="김광현",
            lineup_players=["강백호", "로하스", "최정"],
        )
    )

    assert provided_starter == "starter:provided"
    assert provided_lineup == "lineup:provided"
    assert provided_payload["starter_signature"] == "starter:provided"
    assert provided_payload["lineup_signature"] == "lineup:provided"

    assert fallback_starter != "starter:provided"
    assert fallback_lineup != "lineup:provided"
    assert fallback_payload["starter_signature"] == fallback_starter
    assert fallback_payload["lineup_signature"] == fallback_lineup
    assert provided_key != fallback_key
