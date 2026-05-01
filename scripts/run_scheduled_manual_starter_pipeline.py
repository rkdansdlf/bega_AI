#!/usr/bin/env python3
"""Run the safe scheduled Coach manual-starter pipeline.

The pipeline uses only operator-provided starter CSV data and internal Coach
audit scripts. It does not collect or infer baseball data from external
sources.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "coach_scheduled_manual_pipeline"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: List[str]
    report_path: Optional[Path] = None


@dataclass
class PipelineStepResult:
    name: str
    command: List[str]
    returncode: Optional[int] = None
    report_path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply operator-filled scheduled starters, re-diagnose scheduled "
            "Coach readiness, and optionally backfill grounded games."
        )
    )
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--season-year", type=int, default=datetime.now().year)
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--request-interval-seconds", type=float, default=0.2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=10.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--apply-starters",
        action="store_true",
        help="Persist starter CSV updates before readiness diagnosis.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow starter CSV values to overwrite non-empty DB starter values.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Run Coach backfill for currently grounded scheduled games.",
    )
    parser.add_argument(
        "--no-verify-cache-hit",
        dest="verify_cache_hit",
        action="store_false",
        default=True,
        help="Do not require a second cache HIT probe during backfill.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print and write the pipeline plan without executing commands.",
    )
    return parser


def _resolve_output_dir(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _script_command(script_name: str) -> List[str]:
    return [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)]


def _step_output_dir(output_dir: Path, name: str) -> Path:
    return output_dir / name


def build_apply_step(
    args: argparse.Namespace, output_dir: Path
) -> Optional[PipelineStep]:
    if not args.csv_path:
        return None

    command = [
        *_script_command("apply_manual_starters.py"),
        str(args.csv_path),
        "--output-dir",
        str(_step_output_dir(output_dir, "manual_starters")),
    ]
    if args.apply_starters:
        command.append("--apply")
    if args.allow_overwrite:
        command.append("--allow-overwrite")
    return PipelineStep(name="apply_manual_starters", command=command)


def build_readiness_step(args: argparse.Namespace, output_dir: Path) -> PipelineStep:
    report_dir = _step_output_dir(output_dir, "readiness")
    return PipelineStep(
        name="diagnose_scheduled_readiness",
        command=[
            *_script_command("coach_backfill_audit.py"),
            "--season-year",
            str(args.season_year),
            "--date-from",
            str(args.date_from),
            "--date-to",
            str(args.date_to),
            "--status-bucket",
            "SCHEDULED",
            "--limit",
            str(args.limit),
            "--dry-run",
            "--output-dir",
            str(report_dir),
        ],
        report_path=report_dir / "coach_backfill_results_latest.jsonl",
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            trimmed = line.strip()
            if not trimmed:
                continue
            rows.append(json.loads(trimmed))
    return rows


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _compact_report_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload:
        return {}
    compact: Dict[str, Any] = {}
    for key in ("summary", "paths", "options"):
        value = payload.get(key)
        if value:
            compact[key] = value
    return compact


def _latest_report_summaries(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    return {
        "manual_starter_apply_summary": _compact_report_payload(
            _load_json(
                _step_output_dir(output_dir, "manual_starters")
                / "coach_manual_starter_apply_summary_latest.json"
            )
        ),
        "readiness_summary": _compact_report_payload(
            _load_json(
                _step_output_dir(output_dir, "readiness")
                / "coach_backfill_summary_latest.json"
            )
        ),
        "backfill_summary": _compact_report_payload(
            _load_json(
                _step_output_dir(output_dir, "scheduled_backfill")
                / "coach_backfill_summary_latest.json"
            )
        ),
    }


def _readiness_blockers(readiness_summary: Dict[str, Any]) -> Dict[str, Any]:
    if not readiness_summary:
        return {
            "status": "not_evaluated",
            "manual_baseball_data_required_count": None,
            "starter_announcement_pending_count": None,
            "missing_data_distribution": {},
            "missing_data_distribution_by_source": {},
            "diagnosis_quality_distribution": {},
        }

    summary = readiness_summary.get("summary") or {}
    manual_required = summary.get("manual_baseball_data_required_count") or 0
    try:
        manual_required_count = int(manual_required)
    except (TypeError, ValueError):
        manual_required_count = 0
    starter_pending = summary.get("starter_announcement_pending_count") or 0
    try:
        starter_pending_count = int(starter_pending)
    except (TypeError, ValueError):
        starter_pending_count = 0
    missing_data_distribution = summary.get("missing_data_distribution") or {}
    diagnosis_quality_distribution = summary.get("diagnosis_quality_distribution") or {}
    if manual_required_count > 0 and starter_pending_count > 0:
        status = "mixed_manual_required_and_announcement_pending"
    elif manual_required_count > 0:
        status = "manual_data_required"
    elif starter_pending_count > 0:
        status = "official_announcement_pending"
    elif missing_data_distribution:
        status = "missing_data"
    elif diagnosis_quality_distribution.get("grounded"):
        status = "grounded"
    else:
        status = "clear"
    return {
        "status": status,
        "manual_baseball_data_required_count": manual_required_count,
        "starter_announcement_pending_count": starter_pending_count,
        "missing_data_distribution": missing_data_distribution,
        "missing_data_distribution_by_source": (
            summary.get("missing_data_distribution_by_source") or {}
        ),
        "diagnosis_quality_distribution": diagnosis_quality_distribution,
    }


def grounded_game_ids_from_results(rows: Iterable[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    seen: set[str] = set()
    for row in rows:
        diagnosis = row.get("diagnosis") or {}
        target = row.get("target") or {}
        game_id = str(target.get("game_id") or diagnosis.get("game_id") or "").strip()
        if not game_id or game_id in seen:
            continue
        bucket = str(
            target.get("game_status_bucket")
            or diagnosis.get("game_status_bucket")
            or ""
        ).upper()
        quality = str(
            diagnosis.get("expected_data_quality")
            or target.get("expected_data_quality")
            or ""
        ).lower()
        root_causes = {str(value) for value in diagnosis.get("root_causes") or []}
        if bucket == "SCHEDULED" and quality == "grounded" and not root_causes:
            seen.add(game_id)
            ids.append(game_id)
    return ids


def build_backfill_step(
    args: argparse.Namespace,
    output_dir: Path,
    game_ids: Sequence[str],
) -> Optional[PipelineStep]:
    if not args.backfill or not game_ids:
        return None

    command = [
        *_script_command("coach_backfill_audit.py"),
        "--season-year",
        str(args.season_year),
        "--game-ids",
        ",".join(game_ids),
        "--status-bucket",
        "SCHEDULED",
        "--limit",
        str(args.limit),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--request-interval-seconds",
        str(args.request_interval_seconds),
        "--max-attempts",
        str(args.max_attempts),
        "--retry-backoff-seconds",
        str(args.retry_backoff_seconds),
        "--output-dir",
        str(_step_output_dir(output_dir, "scheduled_backfill")),
    ]
    if args.verify_cache_hit:
        command.append("--verify-cache-hit")
    return PipelineStep(name="backfill_scheduled_grounded", command=command)


def _next_action(
    *,
    status: str,
    args: argparse.Namespace,
    grounded_game_ids: Sequence[str],
    readiness_blockers: Dict[str, Any],
    backfill_summary: Dict[str, Any],
) -> Dict[str, str]:
    blocker_status = str(readiness_blockers.get("status") or "")
    if status == "plan_only" or blocker_status == "not_evaluated":
        return {
            "code": "RUN_PIPELINE",
            "message": "plan-only 상태입니다. readiness 진단을 실행해 실제 차단 사유를 확인하세요.",
        }
    if blocker_status == "manual_data_required":
        return {
            "code": "FILL_MANUAL_BASEBALL_DATA",
            "message": (
                "운영자 제공 선발 투수 CSV를 채운 뒤 --apply-starters로 적용하세요."
            ),
        }
    if blocker_status == "mixed_manual_required_and_announcement_pending":
        return {
            "code": "RESOLVE_DUE_STARTERS_AND_WAIT_PENDING",
            "message": (
                "발표 예정 시각이 지난 경기는 선발 동기화/수동 보강을 확인하고, "
                "나머지는 공식 발표 예정 시각 이후 다시 확인하세요."
            ),
        }
    if blocker_status == "official_announcement_pending":
        return {
            "code": "WAIT_FOR_OFFICIAL_STARTER_ANNOUNCEMENT",
            "message": "공식 선발 발표 예정 시각 이후 내부 동기화가 반영되는지 다시 확인하세요.",
        }
    if blocker_status == "missing_data":
        return {
            "code": "RESOLVE_MISSING_DATA",
            "message": "readiness_summary의 missing_data_distribution을 기준으로 누락 데이터를 보강하세요.",
        }
    if grounded_game_ids and not args.backfill:
        return {
            "code": "RUN_BACKFILL",
            "message": "grounded 예정경기가 있으므로 --backfill로 Coach 캐시를 생성하세요.",
        }
    if backfill_summary:
        return {
            "code": "REVIEW_BACKFILL_SUMMARY",
            "message": "backfill_summary에서 실패 수와 cache HIT 검증 결과를 확인하세요.",
        }
    return {
        "code": "NO_ACTION",
        "message": "추가 차단 사유가 없습니다.",
    }


def _run_step(step: PipelineStep) -> PipelineStepResult:
    completed = subprocess.run(step.command, cwd=PROJECT_ROOT, check=False)
    return PipelineStepResult(
        name=step.name,
        command=step.command,
        returncode=completed.returncode,
        report_path=str(step.report_path) if step.report_path else None,
    )


def _write_summary(
    output_dir: Path,
    *,
    status: str,
    args: argparse.Namespace,
    steps: Sequence[PipelineStepResult],
    grounded_game_ids: Sequence[str],
    manual_starter_apply_summary: Optional[Dict[str, Any]] = None,
    readiness_summary: Optional[Dict[str, Any]] = None,
    backfill_summary: Optional[Dict[str, Any]] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    readiness_payload = readiness_summary or {}
    backfill_payload = backfill_summary or {}
    readiness_blockers = _readiness_blockers(readiness_payload)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "options": {
            "csv_path": args.csv_path,
            "season_year": args.season_year,
            "date_from": args.date_from,
            "date_to": args.date_to,
            "limit": args.limit,
            "apply_starters": bool(args.apply_starters),
            "allow_overwrite": bool(args.allow_overwrite),
            "backfill": bool(args.backfill),
            "verify_cache_hit": bool(args.verify_cache_hit),
            "plan_only": bool(args.plan_only),
        },
        "grounded_game_ids": list(grounded_game_ids),
        "grounded_count": len(grounded_game_ids),
        "readiness_blockers": readiness_blockers,
        "next_action": _next_action(
            status=status,
            args=args,
            grounded_game_ids=grounded_game_ids,
            readiness_blockers=readiness_blockers,
            backfill_summary=backfill_payload,
        ),
        "manual_starter_apply_summary": manual_starter_apply_summary or {},
        "readiness_summary": readiness_payload,
        "backfill_summary": backfill_payload,
        "steps": [asdict(step) for step in steps],
    }
    path = output_dir / "scheduled_manual_starter_pipeline_summary_latest.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_pipeline(args: argparse.Namespace) -> int:
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    planned_steps: List[PipelineStep] = []
    apply_step = build_apply_step(args, output_dir)
    if apply_step:
        planned_steps.append(apply_step)
    readiness_step = build_readiness_step(args, output_dir)
    planned_steps.append(readiness_step)

    if args.plan_only:
        plan_results = [
            PipelineStepResult(
                name=step.name,
                command=step.command,
                returncode=None,
                report_path=str(step.report_path) if step.report_path else None,
            )
            for step in planned_steps
        ]
        summary_path = _write_summary(
            output_dir,
            status="plan_only",
            args=args,
            steps=plan_results,
            grounded_game_ids=[],
            **_latest_report_summaries(output_dir),
        )
        print(summary_path)
        return 0

    results: List[PipelineStepResult] = []
    for step in planned_steps:
        result = _run_step(step)
        results.append(result)
        if result.returncode != 0:
            summary_path = _write_summary(
                output_dir,
                status="failed",
                args=args,
                steps=results,
                grounded_game_ids=[],
                **_latest_report_summaries(output_dir),
            )
            print(summary_path)
            return result.returncode or 1

    grounded_ids = grounded_game_ids_from_results(
        _load_jsonl(readiness_step.report_path)
    )
    backfill_step = build_backfill_step(args, output_dir, grounded_ids)
    if backfill_step:
        result = _run_step(backfill_step)
        results.append(result)
        if result.returncode != 0:
            summary_path = _write_summary(
                output_dir,
                status="failed",
                args=args,
                steps=results,
                grounded_game_ids=grounded_ids,
                **_latest_report_summaries(output_dir),
            )
            print(summary_path)
            return result.returncode or 1

    summary_path = _write_summary(
        output_dir,
        status="completed",
        args=args,
        steps=results,
        grounded_game_ids=grounded_ids,
        **_latest_report_summaries(output_dir),
    )
    print(summary_path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_pipeline(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
