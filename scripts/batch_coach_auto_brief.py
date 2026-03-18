#!/usr/bin/env python3
"""
Prediction 페이지 CoachBriefing 컴포넌트가 사용하는 auto_brief 캐시를 배치로 적재합니다.

manual_detail 배치 스크립트(batch_coach_matchup_cache.py)와 동일한 인프라를 재사용하되,
request_mode="auto_brief" / focus=["recent_form"] 으로 호출합니다.

사용법:
    python scripts/batch_coach_auto_brief.py \
      --years 2025 \
      --status-bucket COMPLETED \
      --limit 50 \
      --order desc \
      --delay-seconds 2.0 \
      --timeout 120 \
      --quality-report reports/coach_2025_auto_brief_50.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.batch_coach_matchup_cache import (
    AUTO_BRIEF_FOCUS,
    PENDING_RECHECK_SECONDS,
    PROJECT_ROOT,
    VALID_STATUS_BUCKET_FILTERS,
    VALID_TARGET_ORDERS,
    VALID_CACHE_STATE_FILTERS,
    WORKSPACE_ROOT,
    MatchupTarget,
    _annotate_cache_resolution,
    _build_result_shell,
    _cache_resolution_log_suffix,
    _collect_retryable_replay_targets,
    _filter_targets_by_cache_state,
    _normalize_recovered_pending_result,
    call_analyze_with_retries,
    collect_cache_verification_results,
    load_targets,
    parse_cache_state_filter,
    parse_status_bucket_filter,
    parse_target_order,
    parse_years,
    resolve_default_internal_api_key,
    summarize_matchup_results,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

AUTO_BRIEF_MODE = "auto_brief"


def _extract_brief_headline(item: Dict[str, Any]) -> str | None:
    """결과에서 auto_brief headline을 추출합니다."""
    meta = item.get("meta") or {}

    cached_response = meta.get("cached_response") or {}
    if isinstance(cached_response, dict):
        sr = cached_response.get("structured_response") or {}
        if isinstance(sr, dict):
            headline = sr.get("headline")
            if isinstance(headline, str) and headline.strip():
                return headline.strip()

    return None


def _extract_data_quality(item: Dict[str, Any]) -> str:
    """결과에서 data_quality를 추출합니다."""
    meta = item.get("meta") or {}
    return str(meta.get("data_quality") or meta.get("generation_mode") or "unknown")


def _extract_grounding_warnings(item: Dict[str, Any]) -> List[str]:
    """grounding_warnings 추출."""
    meta = item.get("meta") or {}
    warnings = meta.get("grounding_warnings")
    if isinstance(warnings, list):
        return [str(w) for w in warnings if w]
    return []


async def async_main(args: argparse.Namespace) -> int:
    years = parse_years(args.years)
    target_order = parse_target_order(args.order)
    status_bucket_filter = parse_status_bucket_filter(args.status_bucket)
    cache_state_filter = parse_cache_state_filter(args.cache_state_filter)

    targets = load_targets(
        years=years,
        league_type="REGULAR",
        request_focus=list(AUTO_BRIEF_FOCUS),
        request_mode=AUTO_BRIEF_MODE,
        question_override=None,
        offset=max(0, args.offset),
        limit=args.limit,
        order=target_order,
        status_bucket_filter=status_bucket_filter,
    )
    targets = _filter_targets_by_cache_state(targets, cache_state_filter)

    if not targets:
        print("No auto_brief targets found.")
        return 0

    print(
        f"=== auto_brief batch: {len(targets)} targets "
        f"(years={years}, status={status_bucket_filter}, order={target_order}) ==="
    )

    if args.verify_cache_only:
        results = collect_cache_verification_results(targets)
        for idx, (target, item) in enumerate(zip(targets, results), start=1):
            headline = _extract_brief_headline(item) or "-"
            quality = _extract_data_quality(item)
            print(
                f"[{idx}/{len(targets)}] {target.game_date} "
                f"{target.away_team_id}@{target.home_team_id} "
                f"-> {item['status']} (quality={quality})"
            )
            if headline != "-":
                print(f"  headline: {headline[:80]}")
    else:
        start_time = datetime.now()
        timeout = httpx.Timeout(args.timeout, connect=min(10.0, args.timeout))
        default_headers: Dict[str, str] = {}
        if args.internal_api_key:
            default_headers["X-Internal-Api-Key"] = args.internal_api_key

        results: List[Dict[str, Any]] = []
        async with httpx.AsyncClient(
            timeout=timeout, headers=default_headers
        ) as client:
            per_target_timeout = max(args.timeout + 35.0, args.timeout)
            for idx, target in enumerate(targets, start=1):
                item = await call_analyze_with_retries(
                    client=client,
                    base_url=args.base_url,
                    target=target,
                    timeout_seconds=per_target_timeout,
                    max_attempts=args.max_attempts,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                )
                results.append(item)

                cache_hit = bool((item.get("meta") or {}).get("cached"))
                quality = _extract_data_quality(item)
                headline = _extract_brief_headline(item)
                warnings = _extract_grounding_warnings(item)

                print(
                    f"[{idx}/{len(targets)}] {target.game_date} "
                    f"{target.away_team_id}@{target.home_team_id} "
                    f"-> {item['status']} (cache_hit={cache_hit}, quality={quality})"
                )
                if headline:
                    print(f"  headline: {headline[:100]}")
                if warnings:
                    print(f"  grounding_warnings: {warnings}")

                if idx < len(targets):
                    await asyncio.sleep(max(0.0, args.delay_seconds))

            # Recovery passes for pending/retryable failures
            result_by_cache_key = {
                str(item.get("cache_key") or ""): item for item in results
            }
            for recovery_pass in range(1, args.recovery_pass_count + 1):
                pending_targets = [
                    t
                    for t in targets
                    if (result_by_cache_key.get(t.cache_key) or {}).get("status")
                    == "in_progress"
                ]
                retryable = _collect_retryable_replay_targets(
                    targets, result_by_cache_key
                )
                if not pending_targets and not retryable:
                    break

                if pending_targets:
                    await asyncio.sleep(max(0.0, args.pending_recheck_seconds))
                    rechecked = collect_cache_verification_results(
                        pending_targets, previous_results=result_by_cache_key
                    )
                    for item in rechecked:
                        recovered = _normalize_recovered_pending_result(item)
                        recovered_meta = dict(recovered.get("meta") or {})
                        recovered_meta["recovery_pass"] = recovery_pass
                        recovered["meta"] = recovered_meta
                        result_by_cache_key[recovered["cache_key"]] = recovered
                        print(
                            f"[recovery:{recovery_pass}] {recovered['cache_key'][:16]}.. "
                            f"-> {recovered['status']} ({recovered.get('reason')})"
                        )

                replay = _collect_retryable_replay_targets(targets, result_by_cache_key)
                for target in replay:
                    item = await call_analyze_with_retries(
                        client=client,
                        base_url=args.base_url,
                        target=target,
                        timeout_seconds=per_target_timeout,
                        max_attempts=args.max_attempts,
                        retry_backoff_seconds=args.retry_backoff_seconds,
                    )
                    item_meta = dict(item.get("meta") or {})
                    item_meta["recovery_pass"] = recovery_pass
                    item["meta"] = item_meta
                    result_by_cache_key[target.cache_key] = item
                    print(
                        f"[retry:{recovery_pass}] {target.game_date} "
                        f"{target.away_team_id}@{target.home_team_id} "
                        f"-> {item['status']} ({item.get('reason')})"
                    )
                    await asyncio.sleep(max(0.0, args.delay_seconds))

            results = [
                result_by_cache_key.get(t.cache_key, _build_result_shell(t))
                for t in targets
            ]

    summary = summarize_matchup_results(
        results=results,
        years=years,
        league_type="REGULAR",
        focus=list(AUTO_BRIEF_FOCUS),
    )
    if not args.verify_cache_only:
        elapsed = datetime.now() - start_time
        summary["runtime_seconds"] = round(elapsed.total_seconds(), 3)

    # Print summary
    generated = summary.get("generated_success_count", 0)
    skipped = summary.get("skipped_count", 0)
    failed = summary.get("failed", 0)
    in_progress = summary.get("in_progress", 0)
    print(
        f"\n=== Summary ===\n"
        f"total={summary['cases']}, generated={generated}, "
        f"cache_hit={skipped}, failed={failed}, in_progress={in_progress}\n"
        f"coverage_rate={summary.get('coverage_rate', 0):.1%}"
    )

    if args.quality_report:
        report = {
            "summary": summary,
            "options": {
                "years": years,
                "mode": AUTO_BRIEF_MODE,
                "focus": list(AUTO_BRIEF_FOCUS),
                "status_bucket": status_bucket_filter,
                "order": target_order,
                "offset": args.offset,
                "limit": args.limit,
                "cache_state_filter": cache_state_filter,
                "verify_cache_only": args.verify_cache_only,
            },
            "details": results,
            "run_started_at": (
                start_time.isoformat() if not args.verify_cache_only else None
            ),
            "run_finished_at": datetime.now().isoformat(),
        }
        report_path = Path(args.quality_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Quality report: {report_path}")

    return 0 if failed == 0 and in_progress == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coach auto_brief batch runner for Prediction page CoachBriefing."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
        help="Coach API base URL.",
    )
    parser.add_argument(
        "--internal-api-key",
        default=resolve_default_internal_api_key(WORKSPACE_ROOT),
        help="Value for X-Internal-Api-Key header.",
    )
    parser.add_argument(
        "--years",
        default="2025",
        help="Comma-separated season years.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Target offset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max targets to process.",
    )
    parser.add_argument(
        "--order",
        default="desc",
        choices=VALID_TARGET_ORDERS,
        help="Target order: asc or desc (default: desc for latest first).",
    )
    parser.add_argument(
        "--status-bucket",
        default="COMPLETED",
        choices=VALID_STATUS_BUCKET_FILTERS,
        help="Filter by game status (default: COMPLETED).",
    )
    parser.add_argument(
        "--cache-state-filter",
        default="ANY",
        choices=VALID_CACHE_STATE_FILTERS,
        help="Filter by cache row state.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=2.0,
        help="Delay between requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout seconds.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max retry attempts per target.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Backoff seconds for retries.",
    )
    parser.add_argument(
        "--quality-report",
        default=None,
        help="Output quality report JSON path.",
    )
    parser.add_argument(
        "--verify-cache-only",
        action="store_true",
        help="Skip API calls; verify existing cache rows only.",
    )
    parser.add_argument(
        "--recovery-pass-count",
        type=int,
        default=1,
        help="Recovery passes for pending/retryable failures.",
    )
    parser.add_argument(
        "--pending-recheck-seconds",
        type=float,
        default=PENDING_RECHECK_SECONDS,
        help="Wait before rechecking pending targets.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        return asyncio.run(async_main(args))
    except Exception as exc:
        logger.error("batch_coach_auto_brief failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
