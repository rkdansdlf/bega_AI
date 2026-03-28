from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from scripts import resume_2025_ingest_readiness as resume_readiness


def test_build_direct_probe_command_uses_readiness_args() -> None:
    readiness_args = resume_readiness.readiness.build_parser().parse_args(
        [
            "--wallet-dir",
            "/tmp/wallet",
            "--oracle-timeout-seconds",
            "7",
        ]
    )

    command = resume_readiness.build_direct_probe_command(readiness_args)

    assert command[0] == resume_readiness.sys.executable
    assert command[1].endswith("/scripts/sync_kbo_data.py")
    assert command[-4:] == [
        "--wallet-dir",
        "/tmp/wallet",
        "--oracle-timeout-seconds",
        "7",
    ]


def test_run_resume_prints_latest_paths_when_oracle_is_still_blocked(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    latest_pointer = report_dir / "latest.json"
    latest_pointer.write_text(
        json.dumps(
            {
                "report_path": "/tmp/readiness.json",
                "artifact_dir": "/tmp/artifacts",
                "handoff_markdown": "/tmp/artifacts/handoff.md",
                "support_bundle": "/tmp/artifacts/support-bundle.tar.gz",
                "oracle_escalation_markdown": "/tmp/artifacts/oracle-escalation.md",
            }
        ),
        encoding="utf-8",
    )

    def _fake_run(command, cwd, capture_output, text, check):  # type: ignore[no-untyped-def]
        del cwd, capture_output, text, check
        assert command[1].endswith("/scripts/sync_kbo_data.py")
        return subprocess.CompletedProcess(command, 1, stdout="blocked\n", stderr="")

    monkeypatch.setattr(resume_readiness.subprocess, "run", _fake_run)

    exit_code = resume_readiness.main(
        [
            "--latest-report-dir",
            str(report_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "oracle_probe_status=blocked" in captured.out
    assert f"latest_pointer={report_dir / 'latest.json'}" in captured.out
    assert "support_bundle=/tmp/artifacts/support-bundle.tar.gz" in captured.out


def test_run_resume_runs_full_readiness_after_successful_probe(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    calls: list[list[str]] = []

    def _fake_run(command, cwd, capture_output, text, check):  # type: ignore[no-untyped-def]
        del cwd, capture_output, text, check
        calls.append(list(command))
        if command[1].endswith("/scripts/sync_kbo_data.py"):
            return subprocess.CompletedProcess(
                command, 0, stdout="probe ok\n", stderr=""
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"output":"/tmp/readiness.json"}\n',
            stderr="",
        )

    monkeypatch.setattr(resume_readiness.subprocess, "run", _fake_run)

    exit_code = resume_readiness.main(
        [
            "--season-year",
            "2025",
            "--skip-smoke",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls[0][1].endswith("/scripts/sync_kbo_data.py")
    assert calls[1][1].endswith("/scripts/run_2025_ingest_readiness.py")
    assert "--season-year" in calls[1]
    assert "2025" in calls[1]
    assert "--skip-smoke" in calls[1]
    assert "oracle_probe_status=ready" in captured.out
    assert '{"output":"/tmp/readiness.json"}' in captured.out


def test_run_resume_retries_probe_until_success(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    probe_calls = {"count": 0}
    sleep_calls: list[float] = []

    def _fake_run(command, cwd, capture_output, text, check):  # type: ignore[no-untyped-def]
        del cwd, capture_output, text, check
        if command[1].endswith("/scripts/sync_kbo_data.py"):
            probe_calls["count"] += 1
            if probe_calls["count"] == 1:
                return subprocess.CompletedProcess(
                    command, 1, stdout="blocked\n", stderr=""
                )
            return subprocess.CompletedProcess(
                command, 0, stdout="probe ok\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout="ready\n", stderr="")

    monkeypatch.setattr(resume_readiness.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        resume_readiness.time, "sleep", lambda seconds: sleep_calls.append(seconds)
    )

    exit_code = resume_readiness.main(
        [
            "--poll-seconds",
            "2.5",
            "--max-attempts",
            "2",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert probe_calls["count"] == 2
    assert sleep_calls == [2.5]
    assert "retry_in_seconds=2.5" in captured.out
