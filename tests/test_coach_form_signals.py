from app.tools.database_query import (
    _classify_form_score,
    _compute_batter_form_score,
    _compute_pitcher_form_score,
)


def test_compute_batter_form_score_marks_hot_profile():
    result = _compute_batter_form_score(
        wrc_plus=152.0,
        ops_plus=145.0,
        season_ops=0.892,
        season_iso=0.221,
        recent_ops=1.084,
        recent_iso=0.338,
        recent_pa=27,
        season_wpa_per_pa=0.0032,
        recent_wpa_per_pa=0.0118,
    )

    assert result["recent_score"] is not None
    assert result["clutch_score"] is not None
    assert result["form_score"] is not None
    assert result["form_score"] > 62.0
    assert _classify_form_score(result["form_score"]) == "hot"


def test_compute_batter_form_score_hides_recent_component_for_small_sample():
    result = _compute_batter_form_score(
        wrc_plus=118.0,
        ops_plus=112.0,
        season_ops=0.801,
        season_iso=0.166,
        recent_ops=0.944,
        recent_iso=0.250,
        recent_pa=9,
        season_wpa_per_pa=0.0011,
        recent_wpa_per_pa=0.0045,
    )

    assert result["season_score"] is not None
    assert result["recent_score"] is None
    assert result["form_score"] is not None


def test_compute_pitcher_form_score_marks_cold_profile():
    result = _compute_pitcher_form_score(
        era_plus=96.0,
        fip_plus=91.0,
        whip=1.39,
        kbb=2.1,
        season_era=4.21,
        season_whip=1.39,
        recent_era=6.75,
        recent_whip=1.88,
        recent_kbb=1.2,
        season_wpa_allowed_per_bf=0.0018,
        recent_wpa_allowed_per_bf=0.0102,
    )

    assert result["season_score"] is not None
    assert result["recent_score"] is not None
    assert result["clutch_score"] is not None
    assert result["form_score"] is not None
    assert result["form_score"] < 45.0
    assert _classify_form_score(result["form_score"]) == "cold"
