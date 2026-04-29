import json
from pathlib import Path
from typing import Any

from scripts import run_scheduled_manual_starter_pipeline as pipeline


def test_grounded_game_ids_from_results_selects_only_ready_scheduled_games():
    rows = [
        {
            "target": {
                "game_id": "20260423HHLG0",
                "game_status_bucket": "SCHEDULED",
            },
            "diagnosis": {
                "expected_data_quality": "grounded",
                "root_causes": [],
            },
        },
        {
            "target": {
                "game_id": "20260424KTSK0",
                "game_status_bucket": "SCHEDULED",
            },
            "diagnosis": {
                "expected_data_quality": "partial",
                "root_causes": ["missing_starters"],
            },
        },
        {
            "target": {
                "game_id": "20260419WOKT0",
                "game_status_bucket": "COMPLETED",
            },
            "diagnosis": {
                "expected_data_quality": "grounded",
                "root_causes": [],
            },
        },
    ]

    assert pipeline.grounded_game_ids_from_results(rows) == ["20260423HHLG0"]


def test_build_apply_step_keeps_apply_flag_explicit(tmp_path: Path):
    args = pipeline.build_parser().parse_args(
        [
            "--csv-path",
            "reports/manual.csv",
            "--date-from",
            "2026-04-24",
            "--date-to",
            "2026-04-30",
            "--apply-starters",
        ]
    )

    step = pipeline.build_apply_step(args, tmp_path)

    assert step is not None
    assert step.name == "apply_manual_starters"
    assert "--apply" in step.command


def test_build_backfill_step_requires_flag(tmp_path: Path):
    args = pipeline.build_parser().parse_args(
        [
            "--date-from",
            "2026-04-24",
            "--date-to",
            "2026-04-30",
        ]
    )

    assert pipeline.build_backfill_step(args, tmp_path, ["20260424KTSK0"]) is None


def test_build_backfill_step_uses_grounded_ids_and_cache_probe(tmp_path: Path):
    args = pipeline.build_parser().parse_args(
        [
            "--date-from",
            "2026-04-24",
            "--date-to",
            "2026-04-30",
            "--backfill",
        ]
    )

    step = pipeline.build_backfill_step(
        args,
        tmp_path,
        ["20260423HHLG0", "20260423HTKT0"],
    )

    assert step is not None
    assert step.name == "backfill_scheduled_grounded"
    assert "--verify-cache-hit" in step.command
    assert "20260423HHLG0,20260423HTKT0" in step.command


def test_run_pipeline_plan_only_does_not_execute(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    args = pipeline.build_parser().parse_args(
        [
            "--csv-path",
            "reports/manual.csv",
            "--date-from",
            "2026-04-24",
            "--date-to",
            "2026-04-30",
            "--output-dir",
            str(tmp_path),
            "--plan-only",
        ]
    )

    def _unexpected_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("subprocess.run should not be called in plan-only mode")

    monkeypatch.setattr(pipeline.subprocess, "run", _unexpected_run)

    assert pipeline.run_pipeline(args) == 0
    summary_path = tmp_path / "scheduled_manual_starter_pipeline_summary_latest.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["readiness_blockers"]["status"] == "not_evaluated"
    assert summary["readiness_blockers"]["manual_baseball_data_required_count"] is None
    assert summary["next_action"]["code"] == "RUN_PIPELINE"


def test_latest_report_summaries_expose_manual_required_blockers(tmp_path: Path) -> None:
    readiness_dir = tmp_path / "readiness"
    readiness_dir.mkdir()
    (readiness_dir / "coach_backfill_summary_latest.json").write_text(
        json.dumps(
            {
                "summary": {
                    "manual_baseball_data_required_count": 30,
                    "missing_data_distribution": {"missing_starters": 30},
                    "diagnosis_quality_distribution": {"partial": 30},
                },
                "paths": {
                    "latest_manual_baseball_data_required_csv": (
                        "/app/reports/manual_required.csv"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    report_summaries = pipeline._latest_report_summaries(tmp_path)
    blockers = pipeline._readiness_blockers(report_summaries["readiness_summary"])

    assert blockers["status"] == "manual_data_required"
    assert blockers["manual_baseball_data_required_count"] == 30
    assert blockers["missing_data_distribution"] == {"missing_starters": 30}
    assert report_summaries["readiness_summary"]["paths"][
        "latest_manual_baseball_data_required_csv"
    ] == "/app/reports/manual_required.csv"


def test_readiness_blockers_expose_official_announcement_pending() -> None:
    blockers = pipeline._readiness_blockers(
        {
            "summary": {
                "manual_baseball_data_required_count": 0,
                "starter_announcement_pending_count": 30,
                "missing_data_distribution": {
                    "starter_announcement_pending": 30,
                },
                "diagnosis_quality_distribution": {"partial": 30},
            }
        }
    )

    assert blockers["status"] == "official_announcement_pending"
    assert blockers["starter_announcement_pending_count"] == 30


def test_readiness_blockers_expose_mixed_manual_and_pending_status() -> None:
    blockers = pipeline._readiness_blockers(
        {
            "summary": {
                "manual_baseball_data_required_count": 5,
                "starter_announcement_pending_count": 25,
                "missing_data_distribution": {
                    "missing_starters": 5,
                    "starter_announcement_pending": 25,
                },
                "diagnosis_quality_distribution": {"partial": 30},
            }
        }
    )

    assert blockers["status"] == "mixed_manual_required_and_announcement_pending"
    assert blockers["manual_baseball_data_required_count"] == 5
    assert blockers["starter_announcement_pending_count"] == 25


def test_next_action_waits_for_official_starter_announcement() -> None:
    args = pipeline.build_parser().parse_args(
        [
            "--date-from",
            "2026-04-24",
            "--date-to",
            "2026-04-30",
        ]
    )

    next_action = pipeline._next_action(
        status="completed",
        args=args,
        grounded_game_ids=[],
        readiness_blockers={"status": "official_announcement_pending"},
        backfill_summary={},
    )

    assert next_action["code"] == "WAIT_FOR_OFFICIAL_STARTER_ANNOUNCEMENT"


def test_next_action_handles_mixed_manual_and_pending_status() -> None:
    args = pipeline.build_parser().parse_args(
        [
            "--date-from",
            "2026-04-24",
            "--date-to",
            "2026-04-30",
        ]
    )

    next_action = pipeline._next_action(
        status="completed",
        args=args,
        grounded_game_ids=[],
        readiness_blockers={"status": "mixed_manual_required_and_announcement_pending"},
        backfill_summary={},
    )

    assert next_action["code"] == "RESOLVE_DUE_STARTERS_AND_WAIT_PENDING"


def test_next_action_requests_backfill_for_grounded_games(tmp_path: Path) -> None:
    args = pipeline.build_parser().parse_args(
        [
            "--date-from",
            "2026-04-24",
            "--date-to",
            "2026-04-30",
        ]
    )

    next_action = pipeline._next_action(
        status="completed",
        args=args,
        grounded_game_ids=["20260424KTSK0"],
        readiness_blockers={"status": "grounded"},
        backfill_summary={},
    )

    assert next_action["code"] == "RUN_BACKFILL"
