#!/usr/bin/env python3
"""
Coach 분석 배치 캐시 생성 스크립트.

기본 동작:
- 지정 연도/팀 조합의 캐시를 생성하거나 갱신
- canonical 키 스키마(v2)로 캐시 저장
- 품질 리포트(JSON) 출력 지원
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.coach_validator import (
    classify_parse_error,
    parse_coach_response_with_meta,
    validate_coach_response,
)
from app.core.prompts import COACH_PROMPT_V2
from app.deps import get_coach_llm_generator, get_connection_pool
from app.routers.coach import _remove_duplicate_json_start
from app.tools.database_query import DatabaseQueryTool
from app.tools.team_code_resolver import CANONICAL_CODES, TeamCodeResolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CANONICAL_TEAMS = ["SS", "LT", "LG", "DB", "KIA", "KH", "HH", "SSG", "NC", "KT"]
PROMPT_VERSION = "v4_dual"
MODEL_NAME = "solar-pro-3"
CACHE_SCHEMA_VERSION = "v2"
COACH_YEAR_MIN = 1982


@dataclass
class RunOptions:
    years: List[int]
    teams: List[str]
    only_missing: bool
    force_rebuild: bool
    quality_report: str | None
    game_type: str
    delay_seconds: float


def parse_years(years_arg: str) -> List[int]:
    years: List[int] = []
    for token in years_arg.split(","):
        token = token.strip()
        if not token:
            continue
        year = int(token)
        years.append(year)
    deduped = sorted(set(years))
    if not deduped:
        raise ValueError("at least one year must be provided")
    return deduped


def parse_teams(teams_arg: str | None, resolver: TeamCodeResolver) -> List[str]:
    if not teams_arg:
        return CANONICAL_TEAMS.copy()

    parsed: List[str] = []
    for token in teams_arg.split(","):
        raw = token.strip()
        if not raw:
            continue
        canonical = resolver.resolve_canonical(raw)
        if canonical in CANONICAL_CODES and canonical not in parsed:
            parsed.append(canonical)
        else:
            logger.warning("Ignoring unsupported regular team input: %s -> %s", raw, canonical)

    if not parsed:
        raise ValueError("no valid canonical regular teams were resolved from --teams")
    return parsed


def is_valid_year(year: int) -> bool:
    return COACH_YEAR_MIN <= year <= datetime.now().year + 1


def build_cache_key(team_id: str, year: int, game_type: str) -> str:
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "team_code_canonical": team_id,
        "away_team_code_canonical": "",
        "year": year,
        "game_type": game_type.upper(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def collect_regular_aliases(resolver: TeamCodeResolver) -> List[str]:
    aliases = set()

    for canonical in CANONICAL_TEAMS:
        for code in resolver.team_variants.get(canonical, []):
            upper_code = str(code).upper()
            if upper_code not in CANONICAL_CODES:
                aliases.add(upper_code)

    for name_or_code, canonical in resolver.name_to_canonical.items():
        if canonical in CANONICAL_CODES:
            normalized = str(name_or_code).strip().upper()
            if normalized and normalized not in CANONICAL_CODES:
                aliases.add(normalized)

    return sorted(aliases)


def cleanup_cache_rows(pool, options: RunOptions, resolver: TeamCodeResolver) -> Dict[str, Any]:
    max_year = datetime.now().year + 1
    legacy_aliases = collect_regular_aliases(resolver)

    result: Dict[str, Any] = {
        "prompt_mismatch_rows": 0,
        "invalid_year_rows": 0,
        "legacy_team_rows": 0,
        "force_rebuild_rows": 0,
        "deleted_total": 0,
    }

    with pool.connection() as conn:
        result["prompt_mismatch_rows"] = conn.execute(
            """
            SELECT COUNT(*)
            FROM coach_analysis_cache
            WHERE year = ANY(%s)
              AND prompt_version <> %s
            """,
            (options.years, PROMPT_VERSION),
        ).fetchone()[0]

        result["invalid_year_rows"] = conn.execute(
            """
            SELECT COUNT(*)
            FROM coach_analysis_cache
            WHERE year < %s OR year > %s
            """,
            (COACH_YEAR_MIN, max_year),
        ).fetchone()[0]

        result["legacy_team_rows"] = conn.execute(
            """
            SELECT COUNT(*)
            FROM coach_analysis_cache
            WHERE year = ANY(%s)
              AND UPPER(team_id) = ANY(%s)
            """,
            (options.years, legacy_aliases),
        ).fetchone()[0]

        if options.force_rebuild:
            result["force_rebuild_rows"] = conn.execute(
                """
                SELECT COUNT(*)
                FROM coach_analysis_cache
                WHERE year = ANY(%s)
                  AND UPPER(team_id) = ANY(%s)
                """,
                (options.years, options.teams),
            ).fetchone()[0]

        delete_query = """
            DELETE FROM coach_analysis_cache
            WHERE (
                year = ANY(%s)
                AND prompt_version <> %s
            )
            OR (year < %s OR year > %s)
            OR (
                year = ANY(%s)
                AND UPPER(team_id) = ANY(%s)
            )
        """
        delete_params: List[Any] = [
            options.years,
            PROMPT_VERSION,
            COACH_YEAR_MIN,
            max_year,
            options.years,
            legacy_aliases,
        ]

        if options.force_rebuild:
            delete_query += """
            OR (
                year = ANY(%s)
                AND UPPER(team_id) = ANY(%s)
            )
            """
            delete_params.extend([options.years, options.teams])

        deleted = conn.execute(delete_query, tuple(delete_params))
        result["deleted_total"] = deleted.rowcount or 0
        conn.commit()

    return result


def format_batch_context(tool_results: Dict[str, Any], year: int) -> str:
    team_summary = tool_results.get("team_summary", {})
    advanced = tool_results.get("advanced_metrics", {})
    recent = tool_results.get("recent_form", {})

    team_name = team_summary.get("team_name") or advanced.get("team_name", "Unknown")
    lines = [f"## {team_name} {year}시즌 분석 데이터", ""]

    batting = advanced.get("metrics", {}).get("batting", {})
    pitching = advanced.get("metrics", {}).get("pitching", {})
    rankings = advanced.get("rankings", {})

    lines.append("### 핵심 지표")
    lines.append(f"- OPS: {batting.get('ops', 'N/A')} (rank={rankings.get('batting_ops', 'N/A')})")
    lines.append(f"- 팀 타율: {batting.get('avg', 'N/A')} (rank={rankings.get('batting_avg', 'N/A')})")
    lines.append(f"- 팀 ERA: {pitching.get('avg_era', 'N/A')} (rank={pitching.get('era_rank', 'N/A')})")
    lines.append("")

    if recent.get("found"):
        summary = recent.get("summary", {})
        lines.append("### 최근 10경기")
        lines.append(
            f"- {summary.get('wins', 0)}승 {summary.get('losses', 0)}패 {summary.get('draws', 0)}무"
        )
        lines.append(f"- 득실 마진: {summary.get('run_diff', 0)}")
        lines.append("")

    top_batters = team_summary.get("top_batters", [])[:5]
    if top_batters:
        lines.append("### 주요 타자")
        for player in top_batters:
            lines.append(
                f"- {player.get('player_name', 'N/A')} | AVG {player.get('avg', 'N/A')} | OPS {player.get('ops', 'N/A')}"
            )
        lines.append("")

    top_pitchers = team_summary.get("top_pitchers", [])[:5]
    if top_pitchers:
        lines.append("### 주요 투수")
        for player in top_pitchers:
            lines.append(
                f"- {player.get('player_name', 'N/A')} | ERA {player.get('era', 'N/A')} | W {player.get('wins', 'N/A')}"
            )
        lines.append("")

    return "\n".join(lines)


async def generate_and_cache_team(
    pool,
    team_id: str,
    year: int,
    game_type: str,
    skip_completed: bool,
    only_missing: bool,
) -> Dict[str, Any]:
    start = datetime.now()
    cache_key = build_cache_key(team_id, year, game_type)

    result: Dict[str, Any] = {
        "team": team_id,
        "year": year,
        "cache_key": cache_key,
        "status": "failed",
        "reason": None,
        "failure_category": None,
        "cache_hit": False,
        "json_parse_ok": False,
        "warnings_count": 0,
        "critical_over_limit": False,
        "validator_hard_fail": False,
        "normalization_applied": False,
        "normalization_reasons": [],
    }

    with pool.connection() as conn:
        row = conn.execute(
            "SELECT status, response_json FROM coach_analysis_cache WHERE cache_key = %s",
            (cache_key,),
        ).fetchone()

        if row:
            status = row[0]
            cached_json = row[1]
            if only_missing:
                result["status"] = "skipped"
                result["reason"] = "existing_cache_row"
                result["cache_hit"] = status == "COMPLETED" and bool(cached_json)
                result["json_parse_ok"] = bool(cached_json)
                result["elapsed_seconds"] = (datetime.now() - start).total_seconds()
                return result

            if status == "COMPLETED" and cached_json and skip_completed:
                result["status"] = "skipped"
                result["reason"] = "already_cached"
                result["cache_hit"] = True
                result["json_parse_ok"] = True
                result["elapsed_seconds"] = (datetime.now() - start).total_seconds()
                return result

        conn.execute(
            """
            INSERT INTO coach_analysis_cache (cache_key, team_id, year, prompt_version, model_name, status)
            VALUES (%s, %s, %s, %s, %s, 'PENDING')
            ON CONFLICT (cache_key) DO UPDATE
                SET status = 'PENDING',
                    team_id = EXCLUDED.team_id,
                    year = EXCLUDED.year,
                    prompt_version = EXCLUDED.prompt_version,
                    model_name = EXCLUDED.model_name,
                    updated_at = now()
            """,
            (cache_key, team_id, year, PROMPT_VERSION, MODEL_NAME),
        )
        conn.commit()

    with pool.connection() as conn:
        db_query = DatabaseQueryTool(conn)
        tool_results = {
            "team_summary": db_query.get_team_summary(team_id, year),
            "advanced_metrics": db_query.get_team_advanced_metrics(team_id, year),
            "recent_form": db_query.get_team_recent_form(team_id, year),
        }

    if not tool_results["team_summary"].get("found"):
        with pool.connection() as conn:
            conn.execute(
                """
                UPDATE coach_analysis_cache
                SET status = 'FAILED', error_message = %s, updated_at = now()
                WHERE cache_key = %s
                """,
                (f"No team summary for {team_id} {year}", cache_key),
            )
            conn.commit()
        result["reason"] = "no_team_summary"
        result["failure_category"] = "no_team_summary"
        result["validator_hard_fail"] = True
        result["elapsed_seconds"] = (datetime.now() - start).total_seconds()
        return result

    context = format_batch_context(tool_results, year)
    question = f"{team_id} {year}시즌 종합 분석"
    coach_prompt = COACH_PROMPT_V2.format(question=question, context=context)
    messages = [{"role": "user", "content": coach_prompt}]

    coach_llm = get_coach_llm_generator()

    try:
        chunks: List[str] = []
        async for chunk in coach_llm(messages):
            chunks.append(chunk)
        full_response = "".join(chunks)
    except Exception as exc:
        with pool.connection() as conn:
            conn.execute(
                """
                UPDATE coach_analysis_cache
                SET status = 'FAILED', error_message = %s, updated_at = now()
                WHERE cache_key = %s
                """,
                (str(exc), cache_key),
            )
            conn.commit()
        result["reason"] = f"llm_error:{exc}"
        result["failure_category"] = "llm_error"
        result["validator_hard_fail"] = True
        result["elapsed_seconds"] = (datetime.now() - start).total_seconds()
        return result

    full_response = _remove_duplicate_json_start(full_response)
    parsed_response, parse_error, parse_meta = parse_coach_response_with_meta(full_response)
    result["normalization_applied"] = bool(parse_meta.get("normalization_applied"))
    result["normalization_reasons"] = list(parse_meta.get("normalization_reasons", []))

    with pool.connection() as conn:
        if parsed_response:
            payload_json = json.dumps(parsed_response.model_dump(), ensure_ascii=False)
            conn.execute(
                """
                UPDATE coach_analysis_cache
                SET status = 'COMPLETED', response_json = %s, error_message = NULL, updated_at = now()
                WHERE cache_key = %s
                """,
                (payload_json, cache_key),
            )
            conn.commit()

            warnings = validate_coach_response(parsed_response)
            critical_count = sum(1 for metric in parsed_response.key_metrics if metric.is_critical)

            result["status"] = "success"
            result["headline"] = parsed_response.headline
            result["json_parse_ok"] = True
            result["warnings_count"] = len(warnings)
            result["warnings"] = warnings
            result["critical_over_limit"] = critical_count > 2
        else:
            error_msg = parse_error or "JSON parsing failed"
            conn.execute(
                """
                UPDATE coach_analysis_cache
                SET status = 'FAILED', error_message = %s, updated_at = now()
                WHERE cache_key = %s
                """,
                (error_msg, cache_key),
            )
            conn.commit()
            result["reason"] = error_msg
            result["failure_category"] = (
                parse_meta.get("error_code")
                or classify_parse_error(error_msg)
            )
            result["validator_hard_fail"] = True

    result["elapsed_seconds"] = (datetime.now() - start).total_seconds()
    return result


def summarize_results(
    results: Iterable[Dict[str, Any]],
    options: RunOptions,
    cleanup_result: Dict[str, Any],
    pool,
    resolver: TeamCodeResolver,
) -> Dict[str, Any]:
    rows = list(results)
    total = len(rows)
    success = sum(1 for row in rows if row["status"] == "success")
    skipped = sum(1 for row in rows if row["status"] == "skipped")
    failed = sum(1 for row in rows if row["status"] == "failed")

    warnings_total = sum(int(row.get("warnings_count", 0)) for row in rows if row["status"] == "success")
    critical_over_limit_count = sum(1 for row in rows if row.get("critical_over_limit"))
    validator_fail_count = sum(1 for row in rows if row.get("validator_hard_fail"))
    parse_success_count = sum(1 for row in rows if row.get("json_parse_ok"))
    normalization_applied_count = sum(
        1 for row in rows if row.get("normalization_applied")
    )
    elapsed_total = sum(float(row.get("elapsed_seconds", 0.0)) for row in rows)
    failure_category_counts: Dict[str, int] = {}
    for row in rows:
        if row.get("status") != "failed":
            continue
        category = row.get("failure_category") or "unknown"
        failure_category_counts[category] = failure_category_counts.get(category, 0) + 1

    legacy_aliases = collect_regular_aliases(resolver)
    with pool.connection() as conn:
        invalid_year_rows = conn.execute(
            """
            SELECT COUNT(*)
            FROM coach_analysis_cache
            WHERE year < %s OR year > %s
            """,
            (COACH_YEAR_MIN, datetime.now().year + 1),
        ).fetchone()[0]

        legacy_team_rows = conn.execute(
            """
            SELECT COUNT(*)
            FROM coach_analysis_cache
            WHERE year = ANY(%s)
              AND UPPER(team_id) = ANY(%s)
            """,
            (options.years, legacy_aliases),
        ).fetchone()[0]

    cases_ok = success + skipped
    summary = {
        "cases": total,
        "all_ok": failed == 0,
        "success": success,
        "skipped": skipped,
        "generated_success_count": success,
        "skipped_count": skipped,
        "failed": failed,
        "target_years": options.years,
        "target_teams": options.teams,
        "game_type": options.game_type.upper(),
        "coverage_rate": round(cases_ok / total, 4) if total else 0.0,
        "json_parse_success_rate": round(parse_success_count / total, 4) if total else 0.0,
        "validator_fail_count": validator_fail_count,
        "normalization_applied_count": normalization_applied_count,
        "warning_rate": round(warnings_total / success, 4) if success else 0.0,
        "critical_over_limit_rate": round(critical_over_limit_count / success, 4) if success else 0.0,
        "cache_hit_rate": round(skipped / total, 4) if total else 0.0,
        "runtime_seconds": round(elapsed_total, 3),
        "cache_invalid_year_count": int(invalid_year_rows),
        "legacy_residual_total": int(legacy_team_rows),
        "failure_category_counts": failure_category_counts,
    }

    return {
        "summary": summary,
        "options": {
            "years": options.years,
            "teams": options.teams,
            "only_missing": options.only_missing,
            "force_rebuild": options.force_rebuild,
            "game_type": options.game_type,
            "delay_seconds": options.delay_seconds,
        },
        "cleanup": cleanup_result,
        "details": rows,
    }


def parse_args() -> RunOptions:
    parser = argparse.ArgumentParser(description="Coach batch cache generator with quality report")
    parser.add_argument(
        "--years",
        default="2025",
        help="Comma-separated year list (e.g. 2018,2019,2020)",
    )
    parser.add_argument(
        "--teams",
        default=None,
        help="Comma-separated team list (canonical or aliases). Default: canonical 10 teams",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only create cache rows that do not already exist (status regardless)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete target-year canonical cache and rebuild",
    )
    parser.add_argument(
        "--quality-report",
        default=None,
        help="Output JSON report path",
    )
    parser.add_argument(
        "--game-type",
        default="REGULAR",
        help="Cache key game_type value (default: REGULAR)",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.5,
        help="Delay between LLM calls (default: 1.5)",
    )
    args = parser.parse_args()

    resolver = TeamCodeResolver()
    years = parse_years(args.years)
    for year in years:
        if not is_valid_year(year):
            raise ValueError(f"invalid year in --years: {year}")

    teams = parse_teams(args.teams, resolver)
    return RunOptions(
        years=years,
        teams=teams,
        only_missing=args.only_missing,
        force_rebuild=args.force_rebuild,
        quality_report=args.quality_report,
        game_type=args.game_type,
        delay_seconds=args.delay_seconds,
    )


async def async_main(options: RunOptions) -> int:
    run_start = datetime.now()
    pool = get_connection_pool()
    resolver = TeamCodeResolver()

    print("=" * 72)
    print("Coach 배치 캐시 시작")
    print(f"Years: {options.years}")
    print(f"Teams: {options.teams}")
    print(f"Prompt: {PROMPT_VERSION} | Schema: {CACHE_SCHEMA_VERSION} | GameType: {options.game_type}")
    print(f"Options: only_missing={options.only_missing}, force_rebuild={options.force_rebuild}")
    print("=" * 72)

    cleanup_result = cleanup_cache_rows(pool, options, resolver)
    logger.info("Cleanup result: %s", cleanup_result)

    results: List[Dict[str, Any]] = []
    total_cases = len(options.years) * len(options.teams)
    case_index = 0

    for year in options.years:
        for team in options.teams:
            case_index += 1
            print(f"[{case_index}/{total_cases}] {year} {team} processing...")
            item = await generate_and_cache_team(
                pool=pool,
                team_id=team,
                year=year,
                game_type=options.game_type,
                skip_completed=(not options.force_rebuild),
                only_missing=(options.only_missing and not options.force_rebuild),
            )
            results.append(item)

            status_icon = {
                "success": "✓",
                "skipped": "⊘",
                "failed": "✗",
            }.get(item["status"], "?")
            reason = item.get("headline") or item.get("reason") or ""
            print(f"  {status_icon} {item['status']}: {reason}")

            if case_index < total_cases:
                await asyncio.sleep(max(options.delay_seconds, 0.0))

    report = summarize_results(results, options, cleanup_result, pool, resolver)
    report["run_started_at"] = run_start.isoformat()
    report["run_finished_at"] = datetime.now().isoformat()

    print("\n" + "=" * 72)
    print("완료 요약")
    print("=" * 72)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    if options.quality_report:
        report_path = Path(options.quality_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Quality report written: {report_path}")

    return 0 if report["summary"]["failed"] == 0 else 1


def main() -> int:
    try:
        options = parse_args()
        return asyncio.run(async_main(options))
    except Exception as exc:
        logger.error("Batch failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
