from app.core.coach_cache_key import (
    build_coach_cache_key,
    build_focus_signature,
    normalize_focus,
)


def test_normalize_focus_dedup_and_sort():
    resolved = normalize_focus(["starter", "recent_form", "starter", "unknown"])
    assert resolved == ["recent_form", "starter"]


def test_build_coach_cache_key_same_focus_different_order_same_key():
    key1, payload1 = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="SSG",
        away_team_code="LG",
        year=2025,
        game_type="REGULAR",
        focus=["recent_form", "bullpen"],
        question_override=None,
    )
    key2, payload2 = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="SSG",
        away_team_code="LG",
        year=2025,
        game_type="regular",
        focus=["bullpen", "recent_form"],
        question_override=None,
    )

    assert key1 == key2
    assert (
        payload1["focus_signature"]
        == payload2["focus_signature"]
        == "recent_form+bullpen"
    )


def test_build_coach_cache_key_diff_focus_different_key():
    key1, _ = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="SSG",
        away_team_code="LG",
        year=2025,
        game_type="REGULAR",
        focus=["recent_form"],
        question_override=None,
    )
    key2, _ = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="SSG",
        away_team_code="LG",
        year=2025,
        game_type="REGULAR",
        focus=["bullpen"],
        question_override=None,
    )
    assert key1 != key2


def test_build_coach_cache_key_same_focus_diff_question_different_key():
    key1, payload1 = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="DB",
        away_team_code=None,
        year=2025,
        game_type="REGULAR",
        focus=["recent_form"],
        question_override="두산 최근 전력 분석",
    )
    key2, payload2 = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="DB",
        away_team_code=None,
        year=2025,
        game_type="REGULAR",
        focus=["recent_form"],
        question_override="두산 불펜 집중 분석",
    )
    assert key1 != key2
    assert payload1["question_signature"] != payload2["question_signature"]


def test_build_coach_cache_key_auto_brief_question_signature_fixed():
    key1, payload1 = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="OB",
        away_team_code="LG",
        year=2025,
        game_type="REGULAR",
        focus=["recent_form", "bullpen"],
        question_override="LG 최근 전력 분석",
        question_signature_override="auto",
    )
    key2, payload2 = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="OB",
        away_team_code="LG",
        year=2025,
        game_type="regular",
        focus=["bullpen", "recent_form"],
        question_override="LG 불펜 리스크 분석",
        question_signature_override="auto",
    )

    assert key1 == key2
    assert payload1["question_signature"] == "auto"
    assert payload2["question_signature"] == "auto"


def test_build_coach_cache_key_auto_brief_stable_with_focus_order_and_case():
    key1, payload1 = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="HH",
        away_team_code="LG",
        year=2025,
        game_type="regular",
        focus=["bullpen", "recent_form", "starter", "BULLPEN"],
        question_override="HH 최근 전력 분석",
        question_signature_override="auto",
    )
    key2, payload2 = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="HH",
        away_team_code="LG",
        year=2025,
        game_type="REGULAR",
        focus=["recent_form", "bullpen", "starter"],
        question_override="HH 불펜 리스크 분석",
        question_signature_override="auto",
    )

    assert key1 == key2
    assert payload1["focus_signature"] == "recent_form+bullpen+starter"
    assert payload2["focus_signature"] == "recent_form+bullpen+starter"
    assert payload1["question_signature"] == "auto"
    assert payload2["question_signature"] == "auto"


def test_build_coach_cache_key_manual_question_change_always_changes_key():
    key1, payload1 = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="OB",
        away_team_code="LG",
        year=2025,
        game_type="REGULAR",
        focus=["recent_form"],
        question_override="OB 전력 분석",
    )
    key2, payload2 = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="OB",
        away_team_code="LG",
        year=2025,
        game_type="REGULAR",
        focus=["recent_form"],
        question_override="OB 경기력 분석",
    )

    assert key1 != key2
    assert payload1["question_signature"] != payload2["question_signature"]


def test_unknown_focus_ignored_signature_all():
    assert build_focus_signature(["unknown", "legacy_only"]) == "all"
