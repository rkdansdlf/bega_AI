from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from scripts import batch_coach_cache


def test_parse_args_requires_quality_report_for_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        batch_coach_cache.sys,
        "argv",
        ["batch_coach_cache.py", "--baseline"],
    )

    with pytest.raises(SystemExit) as exc_info:
        batch_coach_cache.parse_args()

    assert exc_info.value.code == 2


def test_parse_args_accepts_baseline_with_quality_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        batch_coach_cache.sys,
        "argv",
        [
            "batch_coach_cache.py",
            "--baseline",
            "--quality-report",
            "/tmp/coach-quality-report.json",
        ],
    )

    options = batch_coach_cache.parse_args()
    assert options.baseline is True
    assert options.quality_report == "/tmp/coach-quality-report.json"


def test_run_baseline_evaluation_uses_required_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Completed:
        returncode = 0

    def _fake_run(cmd, check=False):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["check"] = check
        return _Completed()

    monkeypatch.setattr(batch_coach_cache.subprocess, "run", _fake_run)
    report_path = Path("/tmp/coach-quality-report.json")

    rc = batch_coach_cache.run_baseline_evaluation(report_path, [2024, 2025], "regular")

    assert rc == 0
    assert captured["check"] is False
    assert captured["cmd"] == [
        batch_coach_cache.sys.executable,
        "scripts/evaluate_coach_quality.py",
        "/tmp/coach-quality-report.json",
        "--require-years",
        "2024,2025",
        "--require-game-type",
        "REGULAR",
    ]


def test_async_main_returns_failure_when_baseline_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "quality.json"
    calls = {"baseline": 0}

    async def _fake_generate_and_cache_team(**kwargs):  # noqa: ANN003
        return {"status": "success", "headline": "ok"}

    def _fake_summarize_results(*args, **kwargs):  # noqa: ANN002, ANN003
        return {"summary": {"failed": 0}, "details": []}

    def _fake_run_baseline(report: Path, years, game_type: str):  # noqa: ANN001
        calls["baseline"] += 1
        assert report == report_path
        assert years == [2025]
        assert game_type == "REGULAR"
        return 1

    monkeypatch.setattr(batch_coach_cache, "get_connection_pool", lambda: object())
    monkeypatch.setattr(batch_coach_cache, "TeamCodeResolver", lambda: object())
    monkeypatch.setattr(
        batch_coach_cache, "cleanup_cache_rows", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        batch_coach_cache,
        "generate_and_cache_team",
        _fake_generate_and_cache_team,
    )
    monkeypatch.setattr(batch_coach_cache, "summarize_results", _fake_summarize_results)
    monkeypatch.setattr(
        batch_coach_cache,
        "run_baseline_evaluation",
        _fake_run_baseline,
    )

    options = batch_coach_cache.RunOptions(
        years=[2025],
        teams=["LG"],
        focus=[],
        only_missing=False,
        force_rebuild=False,
        quality_report=str(report_path),
        baseline=True,
        game_type="REGULAR",
        delay_seconds=0.0,
    )

    rc = asyncio.run(batch_coach_cache.async_main(options))

    assert rc == 1
    assert report_path.exists()
    assert calls["baseline"] == 1
