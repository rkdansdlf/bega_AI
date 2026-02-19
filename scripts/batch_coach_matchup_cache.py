#!/usr/bin/env python3
"""
Prediction 자동 조회와 동일한 매치업 캐시 키를 수동 배치로 생성합니다.

기본 정책:
- 런타임 `/coach/analyze`는 자동 재생성을 하지 않습니다.
- 캐시 재생성은 이 스크립트 같은 수동 배치에서만 수행합니다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.coach_cache_key import (
    build_coach_cache_key,
    build_focus_signature,
    normalize_focus,
)
from app.deps import get_connection_pool
from app.tools.team_code_resolver import CANONICAL_CODES, TeamCodeResolver

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

LEAGUE_TYPE_TO_CODE = {
    "REGULAR": 0,
    "PRE": 2,
    "POST": 5,
}
LEAGUE_CODE_TO_TYPE = {
    0: "REGULAR",
    2: "PRE",
    5: "POST",
}
MATCHUP_FOCUS = ["matchup", "recent_form"]
AUTO_BRIEF_FOCUS = ["recent_form"]
VALID_REQUEST_MODES = ("auto_brief", "manual_detail")
DONE_WAIT_SECONDS = 20.0
DONE_WAIT_INTERVAL_SECONDS = 1.0
COACH_YEAR_MIN = 1982


@dataclass
class MatchupTarget:
    cache_key: str
    game_id: str
    season_id: int
    season_year: int
    game_date: str
    game_type: str
    home_team_id: str
    away_team_id: str
    request_focus: List[str]
    request_mode: str
    question_override: Optional[str]


def parse_years(raw: str) -> List[int]:
    years = sorted({int(token.strip()) for token in raw.split(",") if token.strip()})
    if not years:
        raise ValueError("at least one year is required")
    return years


def parse_league_type(raw: str) -> str:
    normalized = str(raw or "REGULAR").strip().upper()
    if normalized not in {"REGULAR", "PRE", "POST", "ANY"}:
        raise ValueError("league-type must be one of REGULAR, PRE, POST, ANY")
    return normalized


def parse_focus(raw: str | None, *, default: List[str]) -> List[str]:
    if not raw:
        return normalize_focus(default)

    normalized = normalize_focus([token.strip() for token in raw.split(",")])
    return normalized or normalize_focus(default)


def parse_request_mode(raw: str | None) -> str:
    mode = str(raw or "auto_brief").strip().lower()
    if mode not in VALID_REQUEST_MODES:
        raise ValueError(
            f"mode must be one of {', '.join(VALID_REQUEST_MODES)} (got={raw!r})"
        )
    return mode


def parse_question_override(raw: str | None) -> Optional[str]:
    if raw is None:
        return None
    normalized = " ".join(raw.split())
    return normalized or None


def _is_done_marker(data_str: str) -> bool:
    normalized = data_str.strip().lower()
    return normalized in {"[done]", "done", '"[done]"'}


def _extract_error_message(data_str: str) -> str | None:
    data = data_str.strip()
    if not data:
        return None

    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return data

    if not isinstance(parsed, dict):
        return data

    for key in ("error", "message", "detail", "reason"):
        value = parsed.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text

    return None


def _normalize_status_counts(results: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "generated": 0,
        "skipped": 0,
        "in_progress": 0,
        "failed": 0,
    }
    for item in results:
        status = item.get("status")
        if status in summary:
            summary[status] += 1
        else:
            summary["failed"] += 1
    return summary


def _fetch_cache_state(
    cache_key: str,
) -> Optional[tuple[str, Any, Optional[str]]]:
    try:
        with get_connection_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT status, response_json, error_message
                FROM coach_analysis_cache
                WHERE cache_key = %s
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        status = str(row[0] or "")
        response_json = row[1]
        error_message = row[2]
        return status, response_json, error_message
    except Exception:
        return None


async def _wait_cache_completion(
    cache_key: str,
) -> tuple[str | None, Any, Optional[str], bool]:
    attempts = max(1, int(DONE_WAIT_SECONDS / DONE_WAIT_INTERVAL_SECONDS))
    for _ in range(attempts):
        row = _fetch_cache_state(cache_key)
        if row is None:
            return None, None, None, False

        status, response_json, error_message = row
        if status in {"COMPLETED", "FAILED"}:
            return status, response_json, error_message, True
        if status in {"PENDING", "PENDING_WAIT", "FAILED_LOCKED"}:
            # continue polling
            pass
        await asyncio.sleep(DONE_WAIT_INTERVAL_SECONDS)
    return "TIMEOUT", None, "cache_not_ready", False


def load_targets(
    *,
    years: List[int],
    league_type: str,
    request_focus: List[str],
    request_mode: str,
    question_override: Optional[str],
    offset: int,
    limit: Optional[int],
) -> List[MatchupTarget]:
    resolver = TeamCodeResolver()
    pool = get_connection_pool()

    where_parts = ["ks.season_year = ANY(%s)"]
    params: List[Any] = [years]
    if league_type != "ANY":
        where_parts.append("ks.league_type_code = %s")
        params.append(LEAGUE_TYPE_TO_CODE[league_type])

    where_sql = " AND ".join(where_parts)
    query = f"""
        SELECT
            g.game_id,
            g.home_team,
            g.away_team,
            g.game_date,
            g.season_id,
            ks.season_year,
            ks.league_type_code
        FROM game g
        JOIN kbo_seasons ks ON g.season_id = ks.season_id
        WHERE {where_sql}
        ORDER BY g.game_date ASC, g.game_id ASC
    """

    with pool.connection() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()

    deduped: List[MatchupTarget] = []
    seen = set()
    for row in rows:
        (
            game_id,
            home_team,
            away_team,
            game_date,
            season_id,
            season_year,
            league_code,
        ) = row
        home_canonical = resolver.resolve_canonical(home_team)
        away_canonical = resolver.resolve_canonical(away_team)

        if (
            home_canonical not in CANONICAL_CODES
            or away_canonical not in CANONICAL_CODES
        ):
            continue

        game_type = LEAGUE_CODE_TO_TYPE.get(int(league_code), "UNKNOWN")
        dedupe_key = (home_canonical, away_canonical, int(season_year), game_type)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        cache_key, _payload = build_coach_cache_key(
            schema_version="v3",
            prompt_version="v5_focus",
            home_team_code=home_canonical,
            away_team_code=away_canonical,
            year=int(season_year),
            game_type=game_type,
            focus=request_focus,
            question_override=(
                None if request_mode == "auto_brief" else question_override
            ),
        )
        deduped.append(
            MatchupTarget(
                cache_key=cache_key,
                game_id=str(game_id),
                season_id=int(season_id),
                season_year=int(season_year),
                game_date=(
                    game_date.isoformat()
                    if hasattr(game_date, "isoformat")
                    else str(game_date)
                ),
                game_type=game_type,
                home_team_id=home_canonical,
                away_team_id=away_canonical,
                request_focus=list(request_focus),
                request_mode=request_mode,
                question_override=question_override,
            )
        )

    sliced = deduped[offset:]
    if limit is not None and limit >= 0:
        sliced = sliced[:limit]
    return sliced


def force_rebuild_delete(cache_keys: List[str]) -> int:
    if not cache_keys:
        return 0
    pool = get_connection_pool()
    with pool.connection() as conn:
        result = conn.execute(
            "DELETE FROM coach_analysis_cache WHERE cache_key = ANY(%s)",
            (cache_keys,),
        )
        conn.commit()
    return result.rowcount or 0


async def call_analyze(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    target: MatchupTarget,
) -> Dict[str, Any]:
    payload = {
        "home_team_id": target.home_team_id,
        "away_team_id": target.away_team_id,
        "league_context": {
            "season": target.season_id,
            "season_year": target.season_year,
            "game_date": target.game_date,
            "league_type": target.game_type,
            "round": None,
            "game_no": None,
        },
        "focus": target.request_focus,
        "request_mode": target.request_mode,
        "game_id": target.game_id,
    }
    if target.request_mode != "auto_brief" and target.question_override:
        payload["question_override"] = target.question_override

    result: Dict[str, Any] = {
        "cache_key": target.cache_key,
        "game_id": target.game_id,
        "home_team_id": target.home_team_id,
        "away_team_id": target.away_team_id,
        "year": target.season_year,
        "game_type": target.game_type,
        "status": "failed",
        "reason": None,
        "meta": {},
    }

    current_event = "message"
    saw_done = False
    error_message = None
    meta_payload: Dict[str, Any] = {}
    saw_message_event = False
    saw_meta_event = False
    saw_any_event = False
    saw_error_event = False

    try:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/coach/analyze",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                result["reason"] = (
                    f"http_{response.status_code}:{body.decode('utf-8', errors='replace')[:160]}"
                )
                return result

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip().lower()
                    continue
                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if not data_str:
                    continue

                saw_any_event = True
                if _is_done_marker(data_str):
                    saw_done = True
                    continue

                if current_event == "meta":
                    try:
                        parsed = json.loads(data_str)
                        if isinstance(parsed, dict):
                            meta_payload = parsed
                    except json.JSONDecodeError:
                        pass
                elif current_event == "error":
                    error_message = _extract_error_message(data_str)
                    saw_error_event = True
                elif current_event == "message":
                    saw_message_event = True
                    if data_str and data_str.startswith('{"error"'):
                        possible_error = _extract_error_message(data_str)
                        if possible_error:
                            error_message = possible_error

                if current_event == "meta":
                    saw_meta_event = True

    except Exception as exc:
        result["reason"] = str(exc) or exc.__class__.__name__
        return result

    if not saw_done:
        status, response_json, cache_error, was_terminal = await _wait_cache_completion(
            target.cache_key
        )
        if was_terminal:
            if status == "COMPLETED" and response_json:
                result["status"] = "generated"
                result["reason"] = "generated_without_done_event"
                result["meta"] = {"cache_state": "COMPLETED", "in_progress": False}
                return result
            if status == "FAILED":
                result["status"] = "failed"
                result["reason"] = str(cache_error or "failed_without_done_event")
                return result

        if status == "TIMEOUT":
            result["status"] = "failed"
            result["reason"] = "missing_done_event_timeout"
            return result

        if status == "FAILED_LOCKED":
            result["status"] = "failed"
            result["reason"] = "failed_locked_without_done_event"
            return result

        if saw_error_event and error_message:
            result["reason"] = error_message
            return result
        if not saw_any_event:
            result["reason"] = "empty_response_stream"
        elif not saw_message_event and not saw_meta_event:
            result["reason"] = "missing_done_event_no_payload"
        else:
            result["reason"] = "missing_done_event"
        return result
    if error_message:
        result["reason"] = error_message
        return result

    result["meta"] = meta_payload
    cache_state = str(meta_payload.get("cache_state") or "")
    validation_status = str(meta_payload.get("validation_status") or "")

    if cache_state == "FAILED_LOCKED":
        result["status"] = "failed"
        result["reason"] = "failed_locked"
    elif meta_payload.get("in_progress") is True:
        result["status"] = "in_progress"
        result["reason"] = "pending_wait"
    elif meta_payload.get("cached") is True:
        result["status"] = "skipped"
        result["reason"] = "cache_hit"
    elif validation_status == "success":
        result["status"] = "generated"
        result["reason"] = "generated"
    else:
        result["status"] = "failed"
        result["reason"] = cache_state.lower() if cache_state else "validation_fallback"
    return result


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts = _normalize_status_counts(results)
    total = (
        status_counts["generated"]
        + status_counts["skipped"]
        + status_counts["failed"]
        + status_counts["in_progress"]
    )

    return {
        "cases": total,
        "all_ok": status_counts["failed"] == 0 and status_counts["in_progress"] == 0,
        "success": status_counts["generated"],
        "skipped": status_counts["skipped"],
        "in_progress": status_counts["in_progress"],
        "failed": status_counts["failed"],
        "generated_success_count": status_counts["generated"],
        "skipped_count": status_counts["skipped"],
    }


def collect_matchup_integrity_metrics(
    years: List[int],
    years_alias: List[str] | None = None,
) -> tuple[int, int]:
    legacy_aliases = years_alias or []
    pool = get_connection_pool()
    with pool.connection() as conn:
        cache_invalid_year_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM coach_analysis_cache
            WHERE year < %s OR year > %s
            """,
            (COACH_YEAR_MIN, datetime.now().year + 1),
        ).fetchone()[0]

        legacy_residual_total = 0
        if legacy_aliases:
            legacy_residual_total = conn.execute(
                """
                SELECT COUNT(*)
                FROM coach_analysis_cache
                WHERE year = ANY(%s)
                  AND UPPER(team_id) = ANY(%s)
                """,
                (years, legacy_aliases),
            ).fetchone()[0]

    return int(cache_invalid_year_count), int(legacy_residual_total)


def summarize_matchup_results(
    results: List[Dict[str, Any]],
    years: List[int],
    league_type: str,
    focus: List[str],
) -> Dict[str, Any]:
    status_counts = _normalize_status_counts(results)
    total = (
        status_counts["generated"]
        + status_counts["skipped"]
        + status_counts["failed"]
        + status_counts["in_progress"]
    )
    warnings_total = 0
    critical_over_limit_count = 0
    seen_cache_keys: set[str] = set()
    target_teams: set[str] = set()
    failure_reasons: list[str] = []

    for item in results:
        warnings = item.get("warnings_count")
        if isinstance(warnings, (int, float)):
            warnings_total += int(warnings)
        if item.get("critical_over_limit"):
            critical_over_limit_count += 1

        cache_key = item.get("cache_key")
        if isinstance(cache_key, str):
            seen_cache_keys.add(cache_key)

        home_team_id = item.get("home_team_id")
        away_team_id = item.get("away_team_id")
        if isinstance(home_team_id, str):
            target_teams.add(home_team_id.upper())
        if isinstance(away_team_id, str):
            target_teams.add(away_team_id.upper())

        if item.get("status") == "failed":
            reason = item.get("reason")
            if isinstance(reason, str) and reason:
                failure_reasons.append(reason)

    focus_signature = build_focus_signature(focus)

    failure_category_counts: Dict[str, int] = {}
    for item in results:
        if item.get("status") != "failed":
            continue
        reason = item.get("reason") or "unknown"
        failure_category_counts[str(reason)] = (
            failure_category_counts.get(str(reason), 0) + 1
        )

    try:
        from app.tools.team_code_resolver import TeamCodeResolver

        resolver = TeamCodeResolver()
        legacy_aliases: List[str] = []
        for aliases in resolver.team_variants.values():
            for code in aliases:
                if str(code).upper() not in CANONICAL_CODES:
                    legacy_aliases.append(str(code).upper())
        for aliases in getattr(resolver, "legacy_code_map", {}).keys():
            legacy_aliases.append(str(aliases).upper())
        legacy_aliases = sorted(set(legacy_aliases))
        cache_invalid_year_count, legacy_residual_total = (
            collect_matchup_integrity_metrics(
                years,
                legacy_aliases,
            )
        )
    except Exception as exc:
        logger.warning("Failed to collect integrity metrics: %s", exc)
        cache_invalid_year_count = 0
        legacy_residual_total = 0

    return {
        "cases": total,
        "all_ok": status_counts["failed"] == 0 and status_counts["in_progress"] == 0,
        "target_count": total,
        "success": status_counts["generated"],
        "skipped": status_counts["skipped"],
        "in_progress": status_counts["in_progress"],
        "failed": status_counts["failed"],
        "generated_success_count": status_counts["generated"],
        "skipped_count": status_counts["skipped"],
        "target_years": sorted(set(years)),
        "target_teams": sorted(target_teams),
        "focus_signature": focus_signature,
        "cache_key_count": len(seen_cache_keys),
        "game_type": league_type.upper(),
        "coverage_rate": (
            round((status_counts["generated"] + status_counts["skipped"]) / total, 4)
            if total
            else 0.0
        ),
        "json_parse_success_rate": 0.0,
        "warning_rate": (
            round(warnings_total / status_counts["generated"], 4)
            if status_counts["generated"]
            else 0.0
        ),
        "critical_over_limit_rate": (
            round(critical_over_limit_count / status_counts["generated"], 4)
            if status_counts["generated"]
            else 0.0
        ),
        "cache_invalid_year_count": cache_invalid_year_count,
        "legacy_residual_total": legacy_residual_total,
        "failure_reasons": sorted(set(failure_reasons)),
        "failure_category_counts": failure_category_counts,
    }


async def async_main(args: argparse.Namespace) -> int:
    years = parse_years(args.years)
    league_type = parse_league_type(args.league_type)
    request_mode = parse_request_mode(args.mode)
    request_focus = parse_focus(
        args.focus,
        default=AUTO_BRIEF_FOCUS if request_mode == "auto_brief" else MATCHUP_FOCUS,
    )
    question_override = (
        parse_question_override(args.question_override)
        if request_mode != "auto_brief"
        else None
    )

    targets = load_targets(
        years=years,
        league_type=league_type,
        request_focus=request_focus,
        request_mode=request_mode,
        question_override=question_override,
        offset=max(0, args.offset),
        limit=args.limit,
    )

    if not targets:
        print("No matchup targets found.")
        return 0

    if args.force_rebuild:
        deleted = force_rebuild_delete([target.cache_key for target in targets])
        print(f"Force rebuild enabled. deleted_cache_rows={deleted}")

    start_time = datetime.now()
    timeout = httpx.Timeout(args.timeout, connect=min(10.0, args.timeout))
    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for idx, target in enumerate(targets, start=1):
            item = await call_analyze(
                client=client,
                base_url=args.base_url,
                target=target,
            )
            results.append(item)
            print(
                f"[{idx}/{len(targets)}] {target.season_year} {target.home_team_id} vs {target.away_team_id} "
                f"-> {item['status']} ({item.get('reason')})"
            )
            if idx < len(targets):
                await asyncio.sleep(max(0.0, args.delay_seconds))

    summary = summarize_matchup_results(
        results=results,
        years=years,
        league_type=league_type,
        focus=request_focus,
    )
    elapsed = datetime.now() - start_time
    summary["runtime_seconds"] = round(elapsed.total_seconds(), 3)
    report = {
        "summary": summary,
        "options": {
            "years": years,
            "league_type": league_type,
            "focus": request_focus,
            "mode": request_mode,
            "force_rebuild": args.force_rebuild,
            "offset": args.offset,
            "limit": args.limit,
            "question_override": question_override,
        },
        "details": results,
        "run_started_at": start_time.isoformat(),
        "run_finished_at": datetime.now().isoformat(),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.quality_report:
        report_path = Path(args.quality_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Quality report written: {report_path}")
    return 0 if summary["failed"] == 0 and summary["in_progress"] == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coach matchup cache batch runner for Prediction page payloads."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
        help="Coach API base URL.",
    )
    parser.add_argument(
        "--years",
        default="2025",
        help="Comma-separated season years (e.g. 2025 or 2024,2025).",
    )
    parser.add_argument(
        "--league-type",
        default="REGULAR",
        help="REGULAR, PRE, POST, or ANY (default: REGULAR).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Target offset after matchup dedupe.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max matchup targets to process.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Delay between requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout seconds.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete selected matchup cache keys before replaying requests.",
    )
    parser.add_argument(
        "--mode",
        default="auto_brief",
        choices=VALID_REQUEST_MODES,
        help="Request mode: auto_brief or manual_detail.",
    )
    parser.add_argument(
        "--focus",
        help="Comma-separated focus list (e.g. recent_form,bullpen). Defaults by mode.",
    )
    parser.add_argument(
        "--question-override",
        help="Manual detail only: question override text for payload and cache key.",
    )
    parser.add_argument(
        "--quality-report",
        default=None,
        help="Output quality report JSON.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        return asyncio.run(async_main(args))
    except Exception as exc:
        logger.error("batch_coach_matchup_cache failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
