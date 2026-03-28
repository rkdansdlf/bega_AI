from __future__ import annotations

import json
import tarfile
from argparse import Namespace
from pathlib import Path

from scripts import run_2025_ingest_readiness as readiness


def test_parse_sync_stdout_extracts_row_counts() -> None:
    stdout = """
== Syncing game ==
source_rows=720 target_rows=640
  upserted=80
finished table=game synced_rows=80

== Syncing game_events ==
source_rows=900 target_rows=850
  upserted=50
finished table=game_events synced_rows=50
""".strip()

    summary = readiness.parse_sync_stdout(stdout)

    assert summary["table_count"] == 2
    assert summary["total_synced_rows"] == 130
    assert summary["tables"][0]["table"] == "game"
    assert summary["tables"][0]["source_rows"] == 720
    assert summary["tables"][1]["synced_rows"] == 50


def test_parse_sync_stdout_extracts_oracle_alias_resolution_metadata() -> None:
    stdout = """
Oracle alias resolution
preferred_oracle_alias=efh9m9c9h109963k_high
wallet_alias_count=5
oracle_alias_probed=5
oracle_alias_probe_ok=1/5
selected_oracle_alias=efh9m9c9h109963k_medium
selected_alias_fallback=yes
oracle_alias_resolution_status=ready
== Syncing game ==
source_rows=720 target_rows=640
finished table=game synced_rows=80
""".strip()

    summary = readiness.parse_sync_stdout(stdout)

    assert summary["selected_oracle_alias"] == "efh9m9c9h109963k_medium"
    assert summary["selected_alias_fallback"] is True
    assert summary["oracle_alias_resolution_status"] == "ready"
    assert summary["table_count"] == 1


def test_parse_ingest_stdout_extracts_chunk_metrics() -> None:
    stdout = """
 테이블 'game'을(를) 수집 중입니다 ...
      총 48개 청크를 처리했습니다.
   -> 테이블 'game'에서 48개 청크를 작성했습니다 (배치=2, 임베딩 호출=2, 대기 시간=0.14초, 엔진=thread, fallback=0)
 테이블 'game_flow_summary'을(를) 수집 중입니다 ...
      총 12개 청크를 처리했습니다.
   -> 테이블 'game_flow_summary'에서 12개 청크를 작성했습니다 (배치=1, 임베딩 호출=1, 대기 시간=0.02초, 엔진=subinterp, fallback=1)
총 60개 청크 수집을 완료했습니다.
""".strip()

    summary = readiness.parse_ingest_stdout(stdout)

    assert summary["table_count"] == 2
    assert summary["total_chunks_written"] == 60
    assert summary["tables"][0]["chunks_written"] == 48
    assert summary["tables"][1]["parallel_engine"] == "subinterp"
    assert summary["tables"][1]["parallel_engine_fallbacks"] == 1


def test_parse_oracle_diagnostics_stdout_extracts_listener_registration_failures() -> (
    None
):
    stdout = """
Discovered 2 wallet aliases in /tmp/wallet (direct probe)
efh9m9c9h109963k_high: FAIL - DPY-6005: cannot connect to database. | DPY-6001: Service "gdc7f6256e3d53b_efh9m9c9h109963k_high.adb.oraclecloud.com" is not registered with the listener at host "146.56.121.170" port 1522. (Similar to ORA-12514)
efh9m9c9h109963k_medium: OK
""".strip()

    summary = readiness.parse_oracle_diagnostics_stdout(stdout)

    assert summary["alias_count"] == 2
    assert summary["ok_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["listener_registration_missing"] is True
    assert (
        summary["aliases"][0]["listener_service_name"]
        == "gdc7f6256e3d53b_efh9m9c9h109963k_high.adb.oraclecloud.com"
    )
    assert summary["aliases"][0]["listener_host"] == "146.56.121.170"
    assert summary["aliases"][0]["listener_port"] == 1522


def test_parse_oracle_diagnostics_stdout_extracts_listener_refused_connection() -> None:
    stdout = """
Discovered 2 wallet aliases in /tmp/wallet (direct probe)
efh9m9c9h109963k_high: FAIL - DPY-6005: cannot connect to database. | DPY-6000: Listener refused connection. (Similar to ORA-12506)
efh9m9c9h109963k_medium: FAIL - DPY-6005: cannot connect to database. | DPY-6000: Listener refused connection. (Similar to ORA-12506)
""".strip()

    summary = readiness.parse_oracle_diagnostics_stdout(stdout)

    assert summary["alias_count"] == 2
    assert summary["ok_count"] == 0
    assert summary["failed_count"] == 2
    assert summary["listener_registration_missing"] is False
    assert summary["listener_refused_connection"] is True
    assert summary["aliases"][0]["listener_refused_connection"] is True


def test_build_final_report_marks_readiness_failures() -> None:
    args = Namespace(
        season_year=2025,
        since="2025-03-22T09:00:00Z",
        sync_tables=["game"],
        ingest_tables=["game", "game_flow_summary"],
        parallel_engine="thread",
        workers=4,
        read_batch_size=500,
        embed_batch_size=32,
        max_concurrency=2,
        commit_interval=500,
        smoke_batch_size=20,
        benchmark_limit=5,
        wallet_dir="/tmp/wallet",
        oracle_timeout_seconds=7,
        skip_sync=False,
        skip_ingest=False,
        skip_coverage=False,
        skip_benchmark=False,
        skip_smoke=False,
    )
    steps = {
        "sync": readiness.StepRunResult(
            name="sync",
            command=["python", "sync.py"],
            cwd="/tmp",
            duration_ms=100.0,
            exit_code=0,
            status="ok",
            stdout_path="/tmp/sync.out",
            stderr_path="/tmp/sync.err",
        ),
        "ingest": readiness.StepRunResult(
            name="ingest",
            command=["python", "ingest.py"],
            cwd="/tmp",
            duration_ms=100.0,
            exit_code=0,
            status="ok",
            stdout_path="/tmp/ingest.out",
            stderr_path="/tmp/ingest.err",
        ),
        "coverage": readiness.StepRunResult(
            name="coverage",
            command=["python", "coverage.py"],
            cwd="/tmp",
            duration_ms=100.0,
            exit_code=1,
            status="failed",
            stdout_path="/tmp/coverage.out",
            stderr_path="/tmp/coverage.err",
        ),
        "benchmark": readiness.StepRunResult(
            name="benchmark",
            command=["python", "benchmark.py"],
            cwd="/tmp",
            duration_ms=100.0,
            exit_code=0,
            status="ok",
            stdout_path="/tmp/benchmark.out",
            stderr_path="/tmp/benchmark.err",
        ),
        "smoke": readiness.StepRunResult(
            name="smoke",
            command=["python", "smoke.py"],
            cwd="/tmp",
            duration_ms=100.0,
            exit_code=0,
            status="ok",
            stdout_path="/tmp/smoke.out",
            stderr_path="/tmp/smoke.err",
        ),
    }

    report = readiness.build_final_report(
        args=args,
        steps=steps,
        sync_summary={"total_synced_rows": 100},
        oracle_diagnostics={
            "alias_count": 1,
            "ok_count": 0,
            "failed_count": 1,
            "listener_registration_missing": True,
            "aliases": [
                {
                    "alias": "efh9m9c9h109963k_high",
                    "ok": False,
                    "detail": "listener registration missing",
                    "listener_service_name": "gdc7f6256e3d53b_efh9m9c9h109963k_high.adb.oraclecloud.com",
                    "listener_host": "146.56.121.170",
                    "listener_port": 1522,
                    "listener_registration_missing": True,
                }
            ],
        },
        ingest_summary={"total_chunks_written": 200},
        missing_embeddings={"total_missing_embeddings": 3, "rows": []},
        coverage_report={
            "summary": {
                "total_missing_count": 2,
                "total_extra_count": 0,
            },
            "rows": [{"table": "game", "missing_count": 2, "extra_count": 0}],
        },
        benchmark_report={
            "summary": {
                "overall": {
                    "acceptance": {"passed": True},
                }
            }
        },
        smoke_summary_report={
            "summary": {
                "failed": 0,
                "stream_fallback_ratio_ok": True,
            }
        },
    )

    assert report["readiness"]["ready"] is False
    assert "coverage_ok" in report["readiness"]["failure_reasons"]
    assert "missing_embeddings_ok" in report["readiness"]["failure_reasons"]
    assert report["oracle_diagnostics"]["listener_registration_missing"] is True
    assert report["oracle_remediation"]["reason"] == "listener_registration_missing"
    assert report["oracle_remediation"]["operator_action"] == "contact_oracle_dba"
    assert report["oracle_remediation"]["listener_host"] == "146.56.121.170"
    assert report["oracle_remediation"]["listener_port"] == 1522
    assert report["oracle_remediation"]["resume_command"] == [
        readiness.sys.executable,
        str(readiness.PROJECT_ROOT / "scripts" / "resume_2025_ingest_readiness.py"),
        "--season-year",
        "2025",
        "--since",
        "2025-03-22T09:00:00Z",
    ]
    assert (
        report["oracle_remediation"]["service_names"][0]
        == "gdc7f6256e3d53b_efh9m9c9h109963k_high.adb.oraclecloud.com"
    )
    assert report["coverage"]["rows_with_gaps"][0]["table"] == "game"


def test_build_final_report_marks_listener_refused_connection_with_wallet_targets(
    tmp_path: Path,
) -> None:
    wallet_dir = tmp_path / "wallet"
    wallet_dir.mkdir()
    (wallet_dir / "tnsnames.ora").write_text(
        """
efh9m9c9h109963k_high=(DESCRIPTION=(ADDRESS=(PROTOCOL=TCPS)(HOST=146.56.121.170)(PORT=1522))(CONNECT_DATA=(SERVICE_NAME=gdc7f6256e3d53b_efh9m9c9h109963k_high.adb.oraclecloud.com)))
efh9m9c9h109963k_tp=(DESCRIPTION=(ADDRESS=(PROTOCOL=TCPS)(HOST=146.56.121.170)(PORT=1522))(CONNECT_DATA=(SERVICE_NAME=gdc7f6256e3d53b_efh9m9c9h109963k_tp.adb.oraclecloud.com)))
""".strip(),
        encoding="utf-8",
    )

    args = Namespace(
        season_year=2025,
        since="",
        sync_tables=["game"],
        ingest_tables=["game"],
        parallel_engine="thread",
        workers=4,
        read_batch_size=500,
        embed_batch_size=32,
        max_concurrency=2,
        commit_interval=500,
        smoke_batch_size=20,
        benchmark_limit=5,
        wallet_dir=str(wallet_dir),
        oracle_timeout_seconds=7,
        skip_sync=False,
        skip_ingest=True,
        skip_coverage=True,
        skip_benchmark=True,
        skip_smoke=True,
    )
    steps = {
        "sync": readiness.StepRunResult(
            name="sync",
            command=["python", "sync.py"],
            cwd="/tmp",
            duration_ms=100.0,
            exit_code=1,
            status="failed",
            stdout_path="/tmp/sync.out",
            stderr_path="/tmp/sync.err",
        )
    }

    report = readiness.build_final_report(
        args=args,
        steps=steps,
        sync_summary={
            "total_synced_rows": 0,
            "oracle_failure_reason": "listener_refused_connection",
            "oracle_alias_resolution_status": "blocked",
        },
        oracle_diagnostics={
            "alias_count": 2,
            "ok_count": 0,
            "failed_count": 2,
            "listener_registration_missing": False,
            "listener_refused_connection": True,
            "aliases": [
                {
                    "alias": "efh9m9c9h109963k_high",
                    "ok": False,
                    "detail": "DPY-6000: Listener refused connection.",
                    "listener_refused_connection": True,
                },
                {
                    "alias": "efh9m9c9h109963k_tp",
                    "ok": False,
                    "detail": "DPY-6000: Listener refused connection.",
                    "listener_refused_connection": True,
                },
            ],
        },
        ingest_summary=None,
        missing_embeddings=None,
        coverage_report=None,
        benchmark_report=None,
        smoke_summary_report=None,
    )

    assert report["oracle_remediation"]["reason"] == "listener_refused_connection"
    assert report["oracle_remediation"]["operator_action"] == "contact_oracle_dba"
    assert report["oracle_remediation"]["listener_host"] == "146.56.121.170"
    assert report["oracle_remediation"]["listener_port"] == 1522
    assert report["oracle_remediation"]["service_names"] == [
        "gdc7f6256e3d53b_efh9m9c9h109963k_high.adb.oraclecloud.com",
        "gdc7f6256e3d53b_efh9m9c9h109963k_tp.adb.oraclecloud.com",
    ]
    assert report["oracle_remediation"]["dba_checklist"][0].startswith(
        "Confirm the Autonomous Database or target service"
    )


def test_build_oracle_escalation_markdown_includes_dba_request() -> None:
    report = {
        "generated_at_utc": "2026-03-23T13:00:26Z",
        "input": {"season_year": 2025},
        "readiness": {"ready": False},
        "sync": {
            "oracle_failure_reason": "listener_registration_missing",
            "oracle_alias_resolution_status": "blocked",
        },
        "steps": {
            "sync": {
                "stdout_path": "/tmp/readiness/sync.stdout.log",
                "stderr_path": "/tmp/readiness/sync.stderr.log",
            }
        },
        "oracle_diagnostics": {
            "alias_count": 5,
            "ok_count": 0,
            "aliases": [
                {
                    "alias": "efh9m9c9h109963k_high",
                    "ok": False,
                    "detail": "DPY-6001: Service not registered",
                }
            ],
            "step": {
                "stdout_path": "/tmp/readiness/sync_oracle_diagnostics.stdout.log",
                "stderr_path": "/tmp/readiness/sync_oracle_diagnostics.stderr.log",
            },
        },
        "oracle_remediation": {
            "reason": "listener_registration_missing",
            "operator_action": "contact_oracle_dba",
            "summary": "Register the wallet service names on the Oracle listener, then rerun the direct probe.",
            "dba_checklist": [
                "Confirm the database/service is fully available in the Oracle control plane.",
                "Confirm the wallet service names below are registered and exposed by the listener again.",
            ],
            "listener_host": "146.56.121.170",
            "listener_port": 1522,
            "service_names": [
                "gdc7f6256e3d53b_efh9m9c9h109963k_high.adb.oraclecloud.com"
            ],
            "verification_command": [
                "/tmp/python",
                "/tmp/sync_kbo_data.py",
                "--check-oracle-services-direct",
            ],
            "resume_command": [
                "/tmp/python",
                "/tmp/resume_2025_ingest_readiness.py",
                "--season-year",
                "2025",
            ],
        },
    }

    markdown = readiness.build_oracle_escalation_markdown(
        report=report,
        report_path=Path("/tmp/readiness/report.json"),
    )

    assert "# Oracle Escalation Note" in markdown
    assert "Failure reason: `listener_registration_missing`" in markdown
    assert "Operator action: `contact_oracle_dba`" in markdown
    assert "Listener host: `146.56.121.170`" in markdown
    assert "Listener port: `1522`" in markdown
    assert "gdc7f6256e3d53b_efh9m9c9h109963k_high.adb.oraclecloud.com" in markdown
    assert "/tmp/readiness/sync.stdout.log" in markdown
    assert "/tmp/readiness/sync_oracle_diagnostics.stdout.log" in markdown
    assert (
        "/tmp/python /tmp/sync_kbo_data.py --check-oracle-services-direct" in markdown
    )
    assert (
        "/tmp/python /tmp/resume_2025_ingest_readiness.py --season-year 2025"
        in markdown
    )
    assert "## DBA Checklist" in markdown
    assert (
        "Confirm the database/service is fully available in the Oracle control plane."
        in markdown
    )


def test_write_support_bundle_writes_manifest_and_archive(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    report_path = tmp_path / "report.json"
    report_path.write_text('{"ready": false}\n', encoding="utf-8")
    (artifact_dir / "oracle-escalation.md").write_text(
        "# Oracle Escalation Note\n", encoding="utf-8"
    )
    (artifact_dir / "sync.stdout.log").write_text(
        "oracle_alias_resolution_status=blocked\n", encoding="utf-8"
    )

    bundle_paths = readiness.write_support_bundle(
        artifact_dir=artifact_dir,
        report_path=report_path,
    )

    manifest_path = Path(bundle_paths["bundle_manifest"])
    bundle_path = Path(bundle_paths["support_bundle"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["report_path"] == str(report_path)
    assert manifest["bundle_path"] == str(bundle_path)
    assert manifest["included_files"] == [
        "report.json",
        "oracle-escalation.md",
        "sync.stdout.log",
    ]
    with tarfile.open(bundle_path, "r:gz") as archive:
        names = sorted(archive.getnames())
    assert names == [
        "bundle-manifest.json",
        "oracle-escalation.md",
        "report.json",
        "sync.stdout.log",
    ]


def test_build_readiness_handoff_markdown_includes_failure_and_artifacts() -> None:
    report = {
        "generated_at_utc": "2026-03-24T00:10:00Z",
        "input": {"season_year": 2025},
        "readiness": {
            "ready": False,
            "checks": {"sync_ok": False, "missing_embeddings_ok": True},
            "failure_reasons": ["sync_ok"],
        },
        "oracle_remediation": {
            "summary": "Register the wallet service names on the Oracle listener, then rerun the direct probe.",
            "resume_command": [
                "/tmp/python",
                "/tmp/resume_2025_ingest_readiness.py",
                "--season-year",
                "2025",
            ],
            "dba_checklist": [
                "Confirm the database/service is fully available in the Oracle control plane.",
            ],
        },
        "artifacts": {
            "oracle_escalation_markdown": "/tmp/artifacts/oracle-escalation.md",
            "support_bundle": "/tmp/artifacts/support-bundle.tar.gz",
            "bundle_manifest": "/tmp/artifacts/bundle-manifest.json",
            "artifact_dir": "/tmp/artifacts",
        },
    }

    markdown = readiness.build_readiness_handoff_markdown(
        report=report,
        report_path=Path("/tmp/report.json"),
    )

    assert "# Ingest Readiness Handoff" in markdown
    assert "Failure reasons: `sync_ok`" in markdown
    assert "Register the wallet service names on the Oracle listener" in markdown
    assert "/tmp/artifacts/support-bundle.tar.gz" in markdown
    assert "`sync_ok`: `False`" in markdown
    assert (
        "/tmp/python /tmp/resume_2025_ingest_readiness.py --season-year 2025"
        in markdown
    )
    assert "## DBA Checklist" in markdown
    assert (
        "Confirm the database/service is fully available in the Oracle control plane."
        in markdown
    )


def test_write_latest_pointer_files_writes_pointer_and_handoff(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    report_path = report_dir / "readiness.json"
    handoff_markdown = "# Ingest Readiness Handoff\n"
    report = {
        "generated_at_utc": "2026-03-24T00:12:00Z",
        "readiness": {"ready": False, "failure_reasons": ["sync_ok"]},
        "artifacts": {
            "artifact_dir": "/tmp/artifacts",
            "handoff_markdown": "/tmp/artifacts/handoff.md",
            "support_bundle": "/tmp/artifacts/support-bundle.tar.gz",
            "oracle_escalation_markdown": "/tmp/artifacts/oracle-escalation.md",
        },
        "oracle_remediation": {
            "resume_command": [
                "/tmp/python",
                "/tmp/resume_2025_ingest_readiness.py",
                "--season-year",
                "2025",
            ]
        },
    }

    pointer_paths = readiness.write_latest_pointer_files(
        report_dir=report_dir,
        report_path=report_path,
        handoff_markdown=handoff_markdown,
        report=report,
    )

    latest_pointer = Path(pointer_paths["latest_pointer"])
    latest_handoff = Path(pointer_paths["latest_handoff_markdown"])
    pointer_payload = json.loads(latest_pointer.read_text(encoding="utf-8"))

    assert latest_handoff.read_text(encoding="utf-8") == handoff_markdown
    assert pointer_payload["report_path"] == str(report_path)
    assert pointer_payload["handoff_markdown"] == "/tmp/artifacts/handoff.md"
    assert pointer_payload["support_bundle"] == "/tmp/artifacts/support-bundle.tar.gz"
    assert pointer_payload["resume_command"] == [
        "/tmp/python",
        "/tmp/resume_2025_ingest_readiness.py",
        "--season-year",
        "2025",
    ]
    assert pointer_payload["failure_reasons"] == ["sync_ok"]


def test_build_smoke_command_uses_artifact_scoped_lock_file() -> None:
    args = Namespace(
        base_url="http://127.0.0.1:8001",
        smoke_batch_size=5,
        smoke_timeout=180.0,
        internal_api_key="internal-token",
        smoke_question_list="scripts/questions.txt",
    )

    command = readiness._build_smoke_command(
        args,
        smoke_output=Path("/tmp/smoke.json"),
        smoke_summary_output=Path("/tmp/smoke.summary.json"),
        artifact_dir=Path("/tmp/readiness-artifacts"),
    )

    assert "--lock-file" in command
    lock_index = command.index("--lock-file")
    assert command[lock_index + 1] == "/tmp/readiness-artifacts/smoke.lock"
    assert "--internal-api-key" in command
    assert "--chat-question-list" in command


def test_count_missing_embeddings_includes_sample_rows(monkeypatch) -> None:
    class FakeCursor:
        def __init__(self) -> None:
            self._results = [
                [("kbo_regulations", 2)],
                [
                    (
                        "kbo_regulations",
                        "reg_dummy_001",
                        "KBO 타이브레이크 규정 (예시)",
                        2025,
                        None,
                    ),
                    (
                        "kbo_regulations",
                        "reg_dummy_002",
                        "KBO 등록 규정 (예시)",
                        2025,
                        "kbo_regulations_basic",
                    ),
                ],
            ]

        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def execute(self, query, params) -> None:
            del query, params

        def fetchall(self):
            return self._results.pop(0)

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

    monkeypatch.setattr(
        readiness.psycopg, "connect", lambda database_url: FakeConnection()
    )

    report = readiness.count_missing_embeddings(
        database_url="postgresql://example",
        ingest_tables=["kbo_regulations_basic"],
        season_year=2025,
    )

    assert report["total_missing_embeddings"] == 2
    assert report["rows"][0]["source_table"] == "kbo_regulations"
    assert report["rows"][0]["sample_rows"][0]["source_row_id"] == "reg_dummy_001"
    assert (
        report["rows"][0]["sample_rows"][1]["source_profile"] == "kbo_regulations_basic"
    )
