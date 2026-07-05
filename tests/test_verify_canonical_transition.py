import asyncio
import inspect

from scripts import verify_canonical_transition as guard
from scripts.verify_canonical_transition import (
    LEGACY_CODES,
    evaluate_canonical_window,
    evaluate_outside_regression,
    run_canonical_window_smoke,
    run_outside_window_regression,
)


class _FakeAsyncCursor:
    def __init__(self, rows=None):
        self._rows = list(rows or [])
        self._index = 0
        self.executed = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def execute(self, query, params=None):
        self.executed.append((query, params))

    async def fetchone(self):
        if self._index >= len(self._rows):
            return (0,)
        row = self._rows[self._index]
        self._index += 1
        return row


class _FakeAsyncConnection:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.cursors = []

    def cursor(self):
        cursor = _FakeAsyncCursor(self.rows)
        self.cursors.append(cursor)
        return cursor


class _FakeDatabaseQueryTool:
    calls = []

    def __init__(self, conn):
        self.conn = conn

    async def get_team_summary(self, team, year):
        type(self).calls.append(("summary", team, year))
        return {"found": True, "error": None}

    async def get_team_advanced_metrics(self, team, year):
        type(self).calls.append(("advanced", team, year))
        return {"found": True, "error": None}


class _FakeGameQueryTool:
    calls = []

    def __init__(self, conn):
        self.conn = conn

    async def get_team_last_game_date(self, team, year, league_type):
        type(self).calls.append(("last_game", team, year, league_type))
        return {"found": True, "error": None}


def _install_fake_tools(monkeypatch):
    _FakeDatabaseQueryTool.calls = []
    _FakeGameQueryTool.calls = []
    monkeypatch.setattr(guard, "DatabaseQueryTool", _FakeDatabaseQueryTool)
    monkeypatch.setattr(guard, "GameQueryTool", _FakeGameQueryTool)
    monkeypatch.setattr(guard, "clear_coach_cache", lambda: None)


def _contains_coroutine(value):
    if inspect.iscoroutine(value):
        return True
    if isinstance(value, dict):
        return any(_contains_coroutine(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_coroutine(item) for item in value)
    return False


def test_legacy_residual_domain_matches_backend_guard():
    assert set(LEGACY_CODES) == {
        "SK",
        "OB",
        "HT",
        "WO",
        "DO",
        "KI",
        "KW",
        "NX",
        "SL",
        "BE",
        "MBC",
        "LOT",
    }


def test_evaluate_canonical_window_passes_when_all_green():
    output = {
        "canonical_window": {
            "totals": {"cases": 50, "all_ok": 50},
            "legacy_residuals": {"game": 0, "player_season_batting": 0},
            "runtime_seconds": 6.2,
        }
    }
    evaluation = evaluate_canonical_window(
        output,
        strict_canonical_window=True,
        strict_legacy_residual=True,
    )
    assert evaluation["passed"] is True
    assert evaluation["failed_required_checks"] == []
    assert evaluation["legacy_residual_total"] == 0


def test_evaluate_canonical_window_fails_when_matrix_has_miss():
    output = {
        "canonical_window": {
            "totals": {"cases": 50, "all_ok": 49},
            "legacy_residuals": {"game": 0},
            "runtime_seconds": 7.0,
        }
    }
    evaluation = evaluate_canonical_window(
        output,
        strict_canonical_window=True,
        strict_legacy_residual=True,
    )
    assert evaluation["passed"] is False
    assert "canonical_window_all_cases_ok" in evaluation["failed_required_checks"]


def test_evaluate_canonical_window_honors_non_strict_legacy():
    output = {
        "canonical_window": {
            "totals": {"cases": 10, "all_ok": 10},
            "legacy_residuals": {"game": 12},
            "runtime_seconds": 1.0,
        }
    }
    evaluation = evaluate_canonical_window(
        output,
        strict_canonical_window=True,
        strict_legacy_residual=False,
    )
    assert evaluation["passed"] is True
    assert "legacy_residual_total_zero" in evaluation["failed_optional_checks"]


def test_evaluate_outside_regression_default_report_only():
    output = {
        "outside_regression": {
            "total_cases": 360,
            "additional_miss_count": 3,
            "error_diff_count": 1,
            "runtime_seconds": 4.0,
        }
    }
    evaluation = evaluate_outside_regression(
        output,
        strict_outside_regression=False,
    )
    assert evaluation["passed"] is True
    assert evaluation["failed_required_checks"] == []
    assert "outside_additional_miss_zero" in evaluation["failed_optional_checks"]


def test_evaluate_outside_regression_strict_mode_fails():
    output = {
        "outside_regression": {
            "total_cases": 360,
            "additional_miss_count": 1,
            "error_diff_count": 0,
            "runtime_seconds": 4.0,
        }
    }
    evaluation = evaluate_outside_regression(
        output,
        strict_outside_regression=True,
    )
    assert evaluation["passed"] is False
    assert "outside_additional_miss_zero" in evaluation["failed_required_checks"]


def test_run_canonical_window_smoke_awaits_db_tools_and_populates_totals(monkeypatch):
    _install_fake_tools(monkeypatch)
    conn = _FakeAsyncConnection(rows=[(0,)] * 7)

    result = asyncio.run(
        run_canonical_window_smoke(
            conn,
            teams=["SSG"],
            years=[2024],
            window_start=2021,
            window_end=2025,
            outside_mode="dual",
        )
    )

    assert result["totals"] == {
        "cases": 1,
        "summary_ok": 1,
        "advanced_ok": 1,
        "last_game_ok": 1,
        "all_ok": 1,
    }
    assert set(result["legacy_residuals"]) == {
        "game",
        "player_season_batting",
        "player_season_pitching",
        "game_lineups",
        "game_batting_stats",
        "game_pitching_stats",
        "team_daily_roster",
    }
    assert sum(result["legacy_residuals"].values()) == 0
    assert _FakeDatabaseQueryTool.calls == [
        ("summary", "SSG", 2024),
        ("advanced", "SSG", 2024),
    ]
    assert _FakeGameQueryTool.calls == [
        ("last_game", "SSG", 2024, "regular_season"),
    ]
    assert len(conn.cursors) == 1
    assert len(conn.cursors[0].executed) == 7
    assert not _contains_coroutine(result)


def test_run_outside_window_regression_awaits_dual_and_canonical_matrices(
    monkeypatch,
):
    _install_fake_tools(monkeypatch)

    result = asyncio.run(
        run_outside_window_regression(
            _FakeAsyncConnection(),
            teams=["SSG"],
            years=[2019],
            window_start=2021,
            window_end=2025,
            outside_mode="dual",
        )
    )

    assert result["total_cases"] == 3
    assert result["dual_found_total"] == 3
    assert result["canonical_found_total"] == 3
    assert result["additional_miss_count"] == 0
    assert result["additional_hit_count"] == 0
    assert result["error_diff_count"] == 0
    assert _FakeDatabaseQueryTool.calls == [
        ("summary", "SSG", 2019),
        ("advanced", "SSG", 2019),
        ("summary", "SSG", 2019),
        ("advanced", "SSG", 2019),
    ]
    assert _FakeGameQueryTool.calls == [
        ("last_game", "SSG", 2019, "regular_season"),
        ("last_game", "SSG", 2019, "regular_season"),
    ]
    assert not _contains_coroutine(result)
