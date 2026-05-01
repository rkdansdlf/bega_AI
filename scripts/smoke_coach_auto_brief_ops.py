#!/usr/bin/env python3
"""Lightweight ops gate for Prediction auto_brief health and warm-path smoke."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx

os.environ.setdefault("BEGA_SKIP_APP_INIT", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.routers.coach_auto_brief_ops import (
    CoachAutoBriefOpsGateThresholdsInput as GateThresholds,
    _build_auto_brief_health_snapshot_from_inputs,
    _resolve_requested_window,
    build_auto_brief_ops_gate,
    collect_auto_brief_health_inputs,
)
from scripts.batch_coach_auto_brief import _resolve_cache_state_label
from scripts.batch_coach_matchup_cache import (
    WORKSPACE_ROOT,
    MatchupTarget,
    call_analyze,
    resolve_default_internal_api_key,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8001"


@dataclass(frozen=True)
class WarmPathSmokeResult:
    attempted: bool
    ok: bool
    reason: str
    game_id: str | None = None
    cache_key: str | None = None
    status: str | None = None
    cache_state: str | None = None
    cached: bool | None = None
    elapsed_ms: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize today's Prediction auto_brief ops state and fail if the "
            "lightweight gate thresholds are exceeded."
        )
    )
    parser.add_argument(
        "--window",
        default="today",
        choices=["today", "tomorrow", "custom"],
        help="Date window selector for the ops snapshot.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Required for --window custom. Inclusive YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional for --window custom. Inclusive YYYY-MM-DD.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of unresolved targets to retain in the summary.",
    )
    parser.add_argument(
        "--max-unresolved",
        type=int,
        default=0,
        help="Fail when unresolved_count exceeds this threshold.",
    )
    parser.add_argument(
        "--max-failed-locked",
        type=int,
        default=0,
        help="Fail when FAILED_LOCKED count exceeds this threshold.",
    )
    parser.add_argument(
        "--max-pending-wait",
        type=int,
        default=None,
        help="Optional additional threshold for PENDING_WAIT count.",
    )
    parser.add_argument(
        "--max-insufficient-ratio",
        type=float,
        default=None,
        help="Optional threshold for insufficient / selected_target_count.",
    )
    parser.add_argument(
        "--min-selected-targets",
        type=int,
        default=0,
        help=(
            "Optional minimum selected_target_count. Use 0 to allow off-days, "
            "or 1+ when the window is expected to contain live targets."
        ),
    )
    parser.add_argument(
        "--fail-on-missing-report",
        action="store_true",
        help=(
            "Fail when latest prewarm report is missing for a non-empty selected window."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Coach AI base URL used for the warm-path smoke request.",
    )
    parser.add_argument(
        "--internal-api-key",
        default=resolve_default_internal_api_key(WORKSPACE_ROOT),
        help="Value for X-Internal-Api-Key when hitting the AI service directly.",
    )
    parser.add_argument(
        "--smoke-timeout",
        type=float,
        default=15.0,
        help="Warm-path smoke timeout in seconds.",
    )
    parser.add_argument(
        "--skip-warm-smoke",
        action="store_true",
        help="Skip the live auto_brief warm-path smoke request.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def _parse_iso_date(raw: str | None) -> date | None:
    if raw is None:
        return None
    normalized = str(raw).strip()
    if not normalized:
        return None
    return date.fromisoformat(normalized)


def _select_warm_smoke_target(
    selected_targets: list[MatchupTarget],
    results: list[dict[str, Any]],
) -> MatchupTarget | None:
    for target, item in zip(selected_targets, results):
        if _resolve_cache_state_label(item) == "COMPLETED":
            return target
    return None


async def run_warm_path_smoke(
    *,
    target: MatchupTarget | None,
    base_url: str,
    internal_api_key: str,
    timeout_seconds: float,
) -> WarmPathSmokeResult:
    if target is None:
        return WarmPathSmokeResult(
            attempted=False,
            ok=False,
            reason="no_completed_target_for_smoke",
        )

    timeout = httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds))
    headers: dict[str, str] = {}
    if internal_api_key:
        headers["X-Internal-Api-Key"] = internal_api_key

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        item = await call_analyze(
            client=client,
            base_url=base_url,
            target=target,
        )
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))

    meta = dict(item.get("meta") or {})
    status = str(item.get("status") or "")
    cache_state = str(meta.get("cache_state") or _resolve_cache_state_label(item))
    cached = meta.get("cached")
    ok = status == "skipped" and bool(cached) and cache_state == "COMPLETED"

    return WarmPathSmokeResult(
        attempted=True,
        ok=ok,
        reason=str(item.get("reason") or ("cache_hit" if ok else "warm_path_miss")),
        game_id=target.game_id,
        cache_key=target.cache_key,
        status=status or None,
        cache_state=cache_state or None,
        cached=bool(cached) if cached is not None else None,
        elapsed_ms=elapsed_ms,
    )


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    raise TypeError(f"Unsupported model type: {type(model)!r}")


def build_gate_report(
    *,
    snapshot: Any,
    thresholds: GateThresholds,
    smoke_result: WarmPathSmokeResult,
) -> dict[str, Any]:
    gate = build_auto_brief_ops_gate(
        summary=snapshot.summary,
        latest_report=snapshot.latest_report,
        thresholds=thresholds,
    )
    failed_checks = list(gate.checks.failed)
    warnings = list(gate.checks.warnings)

    if smoke_result.attempted and not smoke_result.ok:
        failed_checks.append(
            "warm_path_smoke "
            f"failed ({smoke_result.reason}, status={smoke_result.status}, "
            f"cache_state={smoke_result.cache_state}, cached={smoke_result.cached})"
        )
    elif not smoke_result.attempted and smoke_result.reason != "skipped_by_flag":
        warnings.append(f"warm_path_smoke skipped ({smoke_result.reason})")

    verdict = "FAIL" if failed_checks else "WARN" if warnings else "PASS"
    return {
        "generated_at_utc": snapshot.generated_at_utc.isoformat(),
        "window": snapshot.window,
        "date_window": snapshot.date_window,
        "verdict": verdict,
        "thresholds": asdict(thresholds),
        "checks": {
            "failed": failed_checks,
            "warnings": warnings,
        },
        "health": _model_dump(snapshot),
        "derived": {
            "failed_locked_count": gate.failed_locked_count,
            "pending_wait_count": gate.pending_wait_count,
            "insufficient_count": gate.insufficient_count,
            "insufficient_ratio": gate.insufficient_ratio,
        },
        "warm_path_smoke": asdict(smoke_result),
        "latest_report": (
            _model_dump(snapshot.latest_report)
            if snapshot.latest_report is not None
            else None
        ),
    }


def render_console_summary(report: dict[str, Any]) -> list[str]:
    health = report["health"]
    summary = health["summary"]
    derived = report["derived"]
    smoke = report["warm_path_smoke"]

    lines = [
        (
            "[auto-brief-ops] "
            f"verdict={report['verdict']} "
            f"window={report['window']} "
            f"date_window={report['date_window']}"
        ),
        (
            "[auto-brief-ops] "
            f"selected={summary['selected_target_count']} "
            f"unresolved={summary['unresolved_count']} "
            f"failed_locked={derived['failed_locked_count']} "
            f"pending_wait={derived['pending_wait_count']} "
            f"insufficient_ratio={derived['insufficient_ratio']:.3f}"
        ),
    ]

    latest_report = report.get("latest_report")
    if latest_report:
        lines.append(
            "[auto-brief-ops] "
            f"latest_report={latest_report['path']} "
            f"report_unresolved={latest_report['unresolved_count']} "
            f"report_completed={latest_report['completed_count']}"
        )
    else:
        lines.append("[auto-brief-ops] latest_report=missing")

    if not smoke["attempted"]:
        smoke_label = "SKIP"
    else:
        smoke_label = "PASS" if smoke["ok"] else "FAIL"

    lines.append(
        "[auto-brief-ops] "
        f"warm_path_smoke={smoke_label} "
        f"attempted={smoke['attempted']} "
        f"reason={smoke['reason']} "
        f"game_id={smoke['game_id']} "
        f"status={smoke['status']} "
        f"elapsed_ms={smoke['elapsed_ms']}"
    )

    for item in report["checks"]["warnings"]:
        lines.append(f"[auto-brief-ops] warning={item}")
    for item in report["checks"]["failed"]:
        lines.append(f"[auto-brief-ops] failed_check={item}")
    return lines


async def async_main(args: argparse.Namespace) -> int:
    try:
        start, end = _resolve_requested_window(
            window=args.window,
            start_date=_parse_iso_date(args.start_date),
            end_date=_parse_iso_date(args.end_date),
        )
    except ValueError as exc:
        print(f"[auto-brief-ops] invalid_args={exc}", file=sys.stderr)
        return 2

    try:
        inputs = collect_auto_brief_health_inputs(start=start, end=end)
        snapshot = _build_auto_brief_health_snapshot_from_inputs(
            window=args.window,
            start=start,
            end=end,
            sample_size=max(1, args.sample_size),
            inputs=inputs,
        )
    except Exception as exc:
        print(f"[auto-brief-ops] health_collection_failed={exc}", file=sys.stderr)
        return 2

    if args.skip_warm_smoke:
        smoke_result = WarmPathSmokeResult(
            attempted=False,
            ok=True,
            reason="skipped_by_flag",
        )
    else:
        smoke_target = _select_warm_smoke_target(
            inputs.selected_targets,
            inputs.results,
        )
        smoke_result = await run_warm_path_smoke(
            target=smoke_target,
            base_url=args.base_url,
            internal_api_key=args.internal_api_key,
            timeout_seconds=args.smoke_timeout,
        )

    report = build_gate_report(
        snapshot=snapshot,
        thresholds=GateThresholds(
            max_unresolved=max(0, args.max_unresolved),
            max_failed_locked=max(0, args.max_failed_locked),
            max_pending_wait=args.max_pending_wait,
            max_insufficient_ratio=args.max_insufficient_ratio,
            min_selected_targets=max(0, args.min_selected_targets),
            fail_on_missing_report=bool(args.fail_on_missing_report),
        ),
        smoke_result=smoke_result,
    )

    for line in render_console_summary(report):
        print(line)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[auto-brief-ops] output={output_path}")

    return 0 if report["verdict"] != "FAIL" else 1


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
