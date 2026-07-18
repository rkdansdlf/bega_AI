"""team_resolution_metrics.py 단위 테스트.

TeamResolutionMetrics 클래스의 순수 로직을 독립 인스턴스로 검증한다.
전역 싱글턴 대신 fresh 인스턴스를 사용해 테스트 간 상태 격리를 보장한다.
"""
from __future__ import annotations

from app.tools.team_resolution_metrics import TeamResolutionMetrics, get_team_resolution_metrics


# ── TestYearBucket ────────────────────────────────────────────────────────────

class TestYearBucket:
    def test_all_years_within_window_return_window_label(self):
        for year in range(2021, 2026):
            assert TeamResolutionMetrics._year_bucket(year) == "window_2021_2025"

    def test_year_below_window_returns_outside(self):
        assert TeamResolutionMetrics._year_bucket(2020) == "outside_window"

    def test_year_above_window_returns_outside(self):
        assert TeamResolutionMetrics._year_bucket(2026) == "outside_window"

    def test_none_returns_unknown(self):
        assert TeamResolutionMetrics._year_bucket(None) == "unknown"

    def test_boundary_2021_included(self):
        assert TeamResolutionMetrics._year_bucket(2021) == "window_2021_2025"

    def test_boundary_2025_included(self):
        assert TeamResolutionMetrics._year_bucket(2025) == "window_2021_2025"


# ── TestRecordResolutionEvent ─────────────────────────────────────────────────

class TestRecordResolutionEvent:
    def test_resolution_total_incremented(self):
        m = TeamResolutionMetrics()
        m.record_resolution_event(season_year=2024, query_mode="canonical", outside_window=False, fallback_used=False)
        assert m.snapshot()["resolution_total"] == 1

    def test_outside_window_true_increments_counter(self):
        m = TeamResolutionMetrics()
        m.record_resolution_event(season_year=2019, query_mode="canonical", outside_window=True, fallback_used=False)
        assert m.snapshot()["outside_window_total"] == 1

    def test_outside_window_false_does_not_increment(self):
        m = TeamResolutionMetrics()
        m.record_resolution_event(season_year=2024, query_mode="canonical", outside_window=False, fallback_used=False)
        assert m.snapshot()["outside_window_total"] == 0

    def test_fallback_used_true_increments_counter(self):
        m = TeamResolutionMetrics()
        m.record_resolution_event(season_year=2019, query_mode="dual", outside_window=True, fallback_used=True)
        assert m.snapshot()["outside_window_fallback_total"] == 1

    def test_mode_counts_aggregated(self):
        m = TeamResolutionMetrics()
        m.record_resolution_event(season_year=2024, query_mode="canonical", outside_window=False, fallback_used=False)
        m.record_resolution_event(season_year=2024, query_mode="canonical", outside_window=False, fallback_used=False)
        m.record_resolution_event(season_year=2024, query_mode="dual", outside_window=False, fallback_used=False)
        snap = m.snapshot()
        assert snap["mode_counts"]["canonical"] == 2
        assert snap["mode_counts"]["dual"] == 1

    def test_multiple_events_accumulate(self):
        m = TeamResolutionMetrics()
        for _ in range(5):
            m.record_resolution_event(season_year=2024, query_mode="canonical", outside_window=False, fallback_used=False)
        assert m.snapshot()["resolution_total"] == 5


# ── TestRecordQueryResult ─────────────────────────────────────────────────────

class TestRecordQueryResult:
    def test_found_true_does_not_increment_miss(self):
        m = TeamResolutionMetrics()
        m.record_query_result(source="tool.get_team", season_year=2024, found=True, error=None)
        assert m.snapshot()["query_miss_total"] == 0

    def test_found_false_increments_miss(self):
        m = TeamResolutionMetrics()
        m.record_query_result(source="tool.get_team", season_year=2024, found=False, error=None)
        assert m.snapshot()["query_miss_total"] == 1

    def test_error_present_counts_as_miss_even_if_found_true(self):
        m = TeamResolutionMetrics()
        m.record_query_result(source="tool.get_team", season_year=2024, found=True, error="db_error")
        assert m.snapshot()["query_miss_total"] == 1

    def test_query_total_incremented(self):
        m = TeamResolutionMetrics()
        m.record_query_result(source="s", season_year=2024, found=True, error=None)
        m.record_query_result(source="s", season_year=2024, found=True, error=None)
        assert m.snapshot()["query_total"] == 2

    def test_source_totals_aggregated(self):
        m = TeamResolutionMetrics()
        m.record_query_result(source="src_a", season_year=2024, found=True, error=None)
        m.record_query_result(source="src_a", season_year=2024, found=False, error=None)
        m.record_query_result(source="src_b", season_year=2024, found=True, error=None)
        snap = m.snapshot()
        assert snap["source_totals"]["src_a"] == 2
        assert snap["source_totals"]["src_b"] == 1

    def test_source_misses_aggregated(self):
        m = TeamResolutionMetrics()
        m.record_query_result(source="src_a", season_year=2024, found=False, error=None)
        m.record_query_result(source="src_a", season_year=2024, found=True, error=None)
        snap = m.snapshot()
        assert snap["source_misses"]["src_a"] == 1


# ── TestSnapshot ──────────────────────────────────────────────────────────────

class TestSnapshot:
    def test_initial_state_all_zeros(self):
        m = TeamResolutionMetrics()
        snap = m.snapshot()
        assert snap["resolution_total"] == 0
        assert snap["outside_window_total"] == 0
        assert snap["query_total"] == 0
        assert snap["query_miss_total"] == 0

    def test_miss_rate_calculated_correctly(self):
        m = TeamResolutionMetrics()
        m.record_query_result(source="s", season_year=2024, found=True, error=None)
        m.record_query_result(source="s", season_year=2024, found=False, error=None)
        assert m.snapshot()["team_resolution_miss_rate"] == 0.5

    def test_miss_rate_zero_when_no_queries(self):
        m = TeamResolutionMetrics()
        assert m.snapshot()["team_resolution_miss_rate"] == 0.0

    def test_fallback_rate_calculated_correctly(self):
        m = TeamResolutionMetrics()
        m.record_resolution_event(season_year=2019, query_mode="dual", outside_window=True, fallback_used=True)
        m.record_resolution_event(season_year=2019, query_mode="dual", outside_window=True, fallback_used=False)
        assert m.snapshot()["outside_window_fallback_rate"] == 0.5

    def test_fallback_rate_zero_when_no_outside_window(self):
        m = TeamResolutionMetrics()
        m.record_resolution_event(season_year=2024, query_mode="canonical", outside_window=False, fallback_used=False)
        assert m.snapshot()["outside_window_fallback_rate"] == 0.0

    def test_snapshot_returns_dict(self):
        assert isinstance(TeamResolutionMetrics().snapshot(), dict)

    def test_snapshot_contains_required_keys(self):
        snap = TeamResolutionMetrics().snapshot()
        required = [
            "resolution_total", "outside_window_total", "outside_window_fallback_total",
            "outside_window_fallback_rate", "query_total", "query_miss_total",
            "team_resolution_miss_rate", "mode_counts", "year_bucket_counts",
            "source_totals", "source_misses",
        ]
        for key in required:
            assert key in snap

    def test_year_bucket_counts_reflect_events(self):
        m = TeamResolutionMetrics()
        m.record_resolution_event(season_year=2024, query_mode="canonical", outside_window=False, fallback_used=False)
        m.record_resolution_event(season_year=2019, query_mode="canonical", outside_window=True, fallback_used=False)
        snap = m.snapshot()
        assert snap["year_bucket_counts"]["window_2021_2025"] >= 1
        assert snap["year_bucket_counts"]["outside_window"] >= 1


# ── TestReset ─────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_resolution_total(self):
        m = TeamResolutionMetrics()
        m.record_resolution_event(season_year=2024, query_mode="canonical", outside_window=False, fallback_used=False)
        m.reset()
        assert m.snapshot()["resolution_total"] == 0

    def test_reset_clears_query_total(self):
        m = TeamResolutionMetrics()
        m.record_query_result(source="s", season_year=2024, found=True, error=None)
        m.reset()
        assert m.snapshot()["query_total"] == 0

    def test_reset_clears_mode_counts(self):
        m = TeamResolutionMetrics()
        m.record_resolution_event(season_year=2024, query_mode="canonical", outside_window=False, fallback_used=False)
        m.reset()
        assert m.snapshot()["mode_counts"] == {}

    def test_reset_allows_fresh_accumulation(self):
        m = TeamResolutionMetrics()
        m.record_resolution_event(season_year=2024, query_mode="canonical", outside_window=False, fallback_used=False)
        m.reset()
        m.record_resolution_event(season_year=2024, query_mode="canonical", outside_window=False, fallback_used=False)
        assert m.snapshot()["resolution_total"] == 1


# ── Legacy integration tests (preserved from original file) ───────────────────

def test_outside_window_fallback_rate(monkeypatch):
    from app.tools.team_code_resolver import TeamCodeResolver

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
