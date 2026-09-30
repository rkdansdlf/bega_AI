from app.core import kbo_metrics
from app.core.context_formatter import ContextFormatter


def test_default_context_is_not_season_calibrated():
    ctx = kbo_metrics.LeagueContext()
    assert ctx.constants_source == "default_placeholder"
    assert not ctx.is_season_calibrated


def test_calibrated_context():
    ctx = kbo_metrics.LeagueContext(season=2025, constants_source="kbo_official_2025")
    assert ctx.is_season_calibrated


def test_metric_provenance_marks_estimate_and_version():
    prov = kbo_metrics.metric_provenance(["WAR", "wRC+", "WAR"])
    assert prov["estimated"] is True
    assert prov["estimated_metrics"] == ["WAR", "wRC+"]
    assert prov["metric_version"] == kbo_metrics.METRIC_METHOD_VERSION
    assert prov["league_context_source"] == "default_placeholder"
    assert prov["season_calibrated"] is False


def test_formatter_labels_estimated_wrc():
    f = ContextFormatter.__new__(ContextFormatter)
    line = f._format_batter_line(
        {
            "name": "A",
            "team": "LG",
            "pa": 500,
            "ops": 0.9,
            "wrc_plus": 130.0,
            "estimated_metrics": ["wRC+"],
        },
        1,
        focus_stat="ops",
    )
    assert "wRC+(추정)" in line
    official = f._format_batter_line(
        {"name": "A", "team": "LG", "pa": 500, "ops": 0.9, "wrc_plus": 130.0},
        1,
        focus_stat="ops",
    )
    assert "(추정)" not in official
