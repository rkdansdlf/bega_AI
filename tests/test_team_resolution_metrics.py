from app.tools.team_code_resolver import TeamCodeResolver
from app.tools.team_resolution_metrics import get_team_resolution_metrics


def test_outside_window_fallback_rate(monkeypatch):
    metrics = get_team_resolution_metrics()
    metrics.reset()

    monkeypatch.setenv("TEAM_CODE_READ_MODE", "canonical_only")
    monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_START", "2021")
    monkeypatch.setenv("TEAM_CODE_CANONICAL_WINDOW_END", "2025")
    monkeypatch.setenv("TEAM_CODE_OUTSIDE_WINDOW_MODE", "dual")

    resolver = TeamCodeResolver()
    resolver.query_variants("SSG", 2024)
    resolver.query_variants("SSG", 2019)

    snapshot = metrics.snapshot()
    assert snapshot["resolution_total"] >= 2
    assert snapshot["outside_window_total"] >= 1
    assert snapshot["outside_window_fallback_total"] >= 1
    assert snapshot["outside_window_fallback_rate"] == 1.0


def test_team_resolution_miss_rate():
    metrics = get_team_resolution_metrics()
    metrics.reset()

    metrics.record_query_result(
        source="DatabaseQueryTool.get_team_summary",
        season_year=2024,
        found=True,
        error=None,
    )
    metrics.record_query_result(
        source="DatabaseQueryTool.get_team_summary",
        season_year=2024,
        found=False,
        error="unsupported_team_for_regular_analysis",
    )

    snapshot = metrics.snapshot()
    assert snapshot["query_total"] == 2
    assert snapshot["query_miss_total"] == 1
    assert snapshot["team_resolution_miss_rate"] == 0.5
