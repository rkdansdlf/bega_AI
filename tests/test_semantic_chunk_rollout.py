from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts import run_semantic_chunk_rollout as rollout


def test_build_rollout_steps_full_plan_orders_benchmark_ingest_verify_reembed() -> None:
    args = rollout.build_parser().parse_args(
        [
            "--start-year",
            "2024",
            "--end-year",
            "2025",
        ]
    )

    steps = rollout.build_rollout_steps(args)
    names = [step.name for step in steps]

    assert names[0] == "benchmark_documents"
    assert names[1] == "ingest_static"
    assert "ingest_seasonal_2024" in names
    assert "ingest_seasonal_2025" in names
    assert names[-3:] == [
        "verify_coverage",
        "reembed_missing",
        "verify_coverage_post_reembed",
    ]
    assert "markdown_docs_rules_terms" in steps[1].command


def test_build_rollout_steps_respects_skip_flags() -> None:
    args = rollout.build_parser().parse_args(
        [
            "--skip-benchmark",
            "--skip-static-ingest",
            "--skip-seasonal-ingest",
            "--skip-reembed",
            "--skip-final-verify",
        ]
    )

    steps = rollout.build_rollout_steps(args)

    assert [step.name for step in steps] == ["verify_coverage"]


def test_run_rollout_dry_run_does_not_execute_subprocess(
    monkeypatch: Any,
    capsys: Any,
    tmp_path: Path,
) -> None:
    args = rollout.build_parser().parse_args(
        [
            "--dry-run",
            "--skip-benchmark",
            "--skip-static-ingest",
            "--skip-seasonal-ingest",
            "--skip-reembed",
            "--skip-final-verify",
            "--summary-output",
            str(tmp_path / "dry_run_summary.json"),
        ]
    )

    def _unexpected_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("subprocess.run should not be called during dry-run")

    monkeypatch.setattr(rollout.subprocess, "run", _unexpected_run)

    assert rollout.run_rollout(args) == 0
    captured = capsys.readouterr().out
    assert "verify_coverage" in captured
    summary = json.loads(
        (tmp_path / "dry_run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "dry_run"


def test_run_rollout_blocks_on_benchmark_acceptance_failure(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    benchmark_output = tmp_path / "benchmark.json"
    summary_output = tmp_path / "rollout_summary.json"
    monkeypatch.setattr(rollout, "BENCHMARK_OUTPUT", benchmark_output)

    args = rollout.build_parser().parse_args(
        [
            "--skip-static-ingest",
            "--skip-seasonal-ingest",
            "--skip-verify",
            "--skip-reembed",
            "--skip-final-verify",
            "--summary-output",
            str(summary_output),
        ]
    )
    calls: list[list[str]] = []

    def _fake_run(command, cwd, check):  # type: ignore[no-untyped-def]
        del cwd, check
        calls.append(list(command))
        benchmark_output.write_text(
            json.dumps(
                {
                    "summary": {
                        "overall": {
                            "acceptance": {
                                "passed": False,
                                "checks": {
                                    "zero_hit_not_worse": True,
                                    "quality_improved": False,
                                    "p95_within_budget": True,
                                },
                            }
                        }
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(rollout.subprocess, "run", _fake_run)

    assert rollout.run_rollout(args) == 2
    assert len(calls) == 1
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["failure"]["reason"] == "benchmark_acceptance_failed"


def test_run_rollout_allows_benchmark_regression_with_flag(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    benchmark_output = tmp_path / "benchmark.json"
    summary_output = tmp_path / "rollout_summary.json"
    monkeypatch.setattr(rollout, "BENCHMARK_OUTPUT", benchmark_output)

    args = rollout.build_parser().parse_args(
        [
            "--allow-benchmark-regression",
            "--skip-static-ingest",
            "--skip-seasonal-ingest",
            "--skip-verify",
            "--skip-reembed",
            "--skip-final-verify",
            "--summary-output",
            str(summary_output),
        ]
    )

    def _fake_run(command, cwd, check):  # type: ignore[no-untyped-def]
        del command, cwd, check
        benchmark_output.write_text(
            json.dumps(
                {
                    "summary": {
                        "overall": {
                            "acceptance": {
                                "passed": False,
                                "checks": {
                                    "zero_hit_not_worse": True,
                                    "quality_improved": False,
                                    "p95_within_budget": True,
                                },
                            }
                        }
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(rollout.subprocess, "run", _fake_run)

    assert rollout.run_rollout(args) == 0
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
