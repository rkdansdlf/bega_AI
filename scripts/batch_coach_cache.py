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
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import psycopg

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.coach_validator import (
    classify_parse_error,
    parse_coach_response_with_meta,
    validate_coach_response,
)
from app.core.coach_grounding import (
    CoachFactSheet,
    collect_numeric_tokens,
    validate_response_against_fact_sheet,
)
from app.core.prompts import COACH_PROMPT_V2
from app.core.coach_cache_contract import (
    COACH_BATCH_LEGACY_PROMPT_VERSIONS,
    COACH_BATCH_OWNED_PROMPT_VERSIONS,
    COACH_BATCH_PROMPT_VERSION,
    COACH_CACHE_PROMPT_VERSION,
)
from app.core.coach_cache_key import (
    build_coach_cache_key,
    build_focus_signature,
    normalize_focus,
)
from app.deps import (
    close_connection_pool,
    get_coach_llm_generator,
    get_connection_pool,
)
from app.routers.coach import _remove_duplicate_json_start
from app.tools.database_query import DatabaseQueryTool
from app.tools.team_code_resolver import CANONICAL_CODES, TeamCodeResolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CANONICAL_TEAMS = ["SS", "LT", "LG", "DB", "KIA", "KH", "HH", "SSG", "NC", "KT"]
PROMPT_VERSION = COACH_BATCH_PROMPT_VERSION
MODEL_NAME = "openrouter/free"
CACHE_SCHEMA_VERSION = "v3"
COACH_YEAR_MIN = 1982
STALE_PENDING_HOURS = 3
FOCUS_SECTION_HEADERS: Dict[str, str] = {
    "recent_form": "## 최근 전력",
    "bullpen": "## 불펜 상태",
    "starter": "## 선발 투수",
    "matchup": "## 상대 전적",
    "batting": "## 타격 생산성",
}

BATCH_LEGACY_PROMPT_VERSIONS: tuple = COACH_BATCH_LEGACY_PROMPT_VERSIONS
BATCH_OWNED_PROMPT_VERSIONS: tuple = COACH_BATCH_OWNED_PROMPT_VERSIONS

if COACH_CACHE_PROMPT_VERSION in BATCH_OWNED_PROMPT_VERSIONS:
    raise RuntimeError(
        f"COACH_CACHE_PROMPT_VERSION {COACH_CACHE_PROMPT_VERSION!r} must not be "
        f"in BATCH_OWNED_PROMPT_VERSIONS {BATCH_OWNED_PROMPT_VERSIONS!r}; "
        "batch must never own the live contract version"
    )


@dataclass
class RunOptions:
    years: List[int]
    teams: List[str]
    focus: List[str]
    only_missing: bool
    force_rebuild: bool
    quality_report: str | None
    baseline: bool
    game_type: str
    delay_seconds: float
    dry_run: bool = False
    purge_invalid_years: bool = False
    cleanup_legacy: bool = False


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
            logger.warning(
                "Ignoring unsupported regular team input: %s -> %s", raw, canonical
            )

    if not parsed:
        raise ValueError("no valid canonical regular teams were resolved from --teams")
    return parsed


def parse_focus(focus_arg: str | None) -> List[str]:
    if not focus_arg:
        return []
    return normalize_focus([token.strip() for token in focus_arg.split(",")])


def is_valid_year(year: int) -> bool:
    return COACH_YEAR_MIN <= year <= datetime.now().year + 1


def build_focus_section_requirements(resolved_focus: List[str]) -> str:
    if not resolved_focus:
        return (
            "- 선택 focus가 비어 있습니다. 종합 분석을 수행하세요.\n"
            "- 다만 detailed_markdown은 최소 2개 이상의 소제목(##)으로 구성하세요."
        )
    selected = [
        f"- 반드시 `{FOCUS_SECTION_HEADERS[item]}` 제목을 포함하세요."
        for item in resolved_focus
        if item in FOCUS_SECTION_HEADERS
    ]
    omitted = [
        f"- 미선택 focus는 가능하면 생략하세요: `{header}`"
        for key, header in FOCUS_SECTION_HEADERS.items()
        if key not in resolved_focus
    ]
    return "\n".join(selected + omitted)


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


def build_cleanup_conditions(
    options: RunOptions, resolver: TeamCodeResolver
) -> List[tuple[str, str, List[Any]]]:
    conditions: List[tuple[str, str, List[Any]]] = []
    conditions.append(
        (
            "prompt_mismatch",
            "year = ANY(%s) AND prompt_version = ANY(%s) AND prompt_version <> %s",
            [
                options.years,
                list(BATCH_LEGACY_PROMPT_VERSIONS),
                COACH_CACHE_PROMPT_VERSION,
            ],
        )
    )
    conditions.append(
        (
            "legacy_team",
            (
                "year = ANY(%s) AND UPPER(team_id) = ANY(%s)"
                " AND prompt_version = ANY(%s) AND prompt_version <> %s"
            ),
            [
                options.years,
                collect_regular_aliases(resolver),
                list(BATCH_OWNED_PROMPT_VERSIONS),
                COACH_CACHE_PROMPT_VERSION,
            ],
        )
    )
    # A run killed mid-generation leaves its new key PENDING (a COMPLETED row is
    # never downgraded thanks to verify-then-replace). PENDING rows older than
    # STALE_PENDING_HOURS cannot belong to a live run: the weekly job times out
    # after 90 minutes and shares the coach-cache-operations concurrency group.
    conditions.append(
        (
            "stale_pending",
            (
                "year = ANY(%s) AND status = 'PENDING'"
                " AND updated_at < now() - make_interval(hours => %s)"
                " AND prompt_version = ANY(%s) AND prompt_version <> %s"
            ),
            [
                options.years,
                STALE_PENDING_HOURS,
                list(BATCH_OWNED_PROMPT_VERSIONS),
                COACH_CACHE_PROMPT_VERSION,
            ],
        )
    )
    if options.purge_invalid_years:
        max_year = datetime.now().year + 1
        conditions.append(
            (
                "invalid_year",
                (
                    "(year < %s OR year > %s)"
                    " AND prompt_version = ANY(%s) AND prompt_version <> %s"
                ),
                [
                    COACH_YEAR_MIN,
                    max_year,
                    list(BATCH_OWNED_PROMPT_VERSIONS),
                    COACH_CACHE_PROMPT_VERSION,
                ],
            )
        )
    return conditions


def build_target_batch_keys(options: RunOptions) -> List[str]:
    keys: List[str] = []
    for year in options.years:
        for team in options.teams:
            cache_key, _ = build_coach_cache_key(
                schema_version=CACHE_SCHEMA_VERSION,
                prompt_version=PROMPT_VERSION,
                home_team_code=team,
                away_team_code=None,
                year=year,
                game_type=options.game_type,
                focus=options.focus,
                question_override=None,
            )
            keys.append(cache_key)
    return keys


async def cleanup_cache_rows(
    pool,
    options: RunOptions,
    resolver: TeamCodeResolver,
    dry_run: bool | None = None,
) -> Dict[str, Any]:
    """Plan (and, with --cleanup-legacy and not dry-run, apply) batch cleanup.

    Deletes only happen when ``options.cleanup_legacy`` is set and the run is
    not a dry-run. Otherwise the same plan is computed read-only and the
    candidates are reported under ``cleanup_candidate_keys``.
    """
    if dry_run is None:
        dry_run = options.dry_run
    delete_enabled = bool(options.cleanup_legacy) and not dry_run

    conditions = build_cleanup_conditions(options, resolver)
    target_keys = build_target_batch_keys(options)

    category_keys: Dict[str, List[str]] = {
        "prompt_mismatch": [],
        "legacy_team": [],
        "stale_pending": [],
        "invalid_year": [],
    }
    force_rebuild_replace_keys: List[str] = []
    new_generation_keys: List[str] = []
    skip_existing_keys: List[str] = []
    invalid_year_rows_total = 0
    deleted_keys: List[str] = []

    async with pool.connection() as conn:
        async with conn.transaction():
            if not delete_enabled:
                await conn.execute("SET TRANSACTION READ ONLY")

            for category, where_sql, params in conditions:
                cur = await conn.execute(
                    f"SELECT cache_key FROM coach_analysis_cache"
                    f" WHERE {where_sql} ORDER BY cache_key",
                    tuple(params),
                )
                rows = await cur.fetchall()
                category_keys[category] = [r[0] for r in rows]

            candidate_keys = sorted(
                {key for keys in category_keys.values() for key in keys}
            )

            cur = await conn.execute(
                "SELECT cache_key, status, response_json IS NOT NULL"
                " FROM coach_analysis_cache WHERE cache_key = ANY(%s)",
                (target_keys,),
            )
            existing = {r[0]: (r[1], bool(r[2])) for r in await cur.fetchall()}
            # Simulate the deletion in --dry-run --cleanup-legacy so the plan
            # matches what the real run would do.
            removed = set(candidate_keys) if options.cleanup_legacy else set()
            effective_only_missing = options.only_missing and not options.force_rebuild
            for key in target_keys:
                state = None if key in removed else existing.get(key)
                completed = bool(state) and state[0] == "COMPLETED" and state[1]
                if effective_only_missing:
                    if state is None:
                        new_generation_keys.append(key)
                    else:
                        skip_existing_keys.append(key)
                elif completed and options.force_rebuild:
                    force_rebuild_replace_keys.append(key)
                elif completed:
                    skip_existing_keys.append(key)
                else:
                    new_generation_keys.append(key)

            max_year = datetime.now().year + 1
            cur = await conn.execute(
                "SELECT COUNT(*) FROM coach_analysis_cache"
                " WHERE year < %s OR year > %s",
                (COACH_YEAR_MIN, max_year),
            )
            row = await cur.fetchone()
            invalid_year_rows_total = row[0]

            if delete_enabled and candidate_keys:
                where_parts: List[str] = []
                all_params: List[Any] = []
                for _, where_sql, params in conditions:
                    where_parts.append(f"({where_sql})")
                    all_params.extend(params)
                combined_where = " OR ".join(where_parts)
                cur = await conn.execute(
                    f"DELETE FROM coach_analysis_cache"
                    f" WHERE {combined_where} RETURNING cache_key",
                    tuple(all_params),
                )
                rows = await cur.fetchall()
                deleted_keys = sorted(r[0] for r in rows)

            if not delete_enabled:
                # Read-only plan: end with ROLLBACK, never COMMIT.
                raise psycopg.Rollback()

    would_delete_keys = candidate_keys if options.cleanup_legacy else []
    return {
        "dry_run": dry_run,
        "cleanup_legacy_enabled": bool(options.cleanup_legacy),
        "prompt_mismatch_rows": len(category_keys["prompt_mismatch"]),
        "legacy_team_rows": len(category_keys["legacy_team"]),
        "stale_pending_rows": len(category_keys["stale_pending"]),
        "invalid_year_rows": len(category_keys["invalid_year"]),
        "invalid_year_purge_enabled": options.purge_invalid_years,
        "invalid_year_rows_total": invalid_year_rows_total,
        "cleanup_candidate_keys": candidate_keys,
        "force_rebuild_rows": len(force_rebuild_replace_keys),
        "force_rebuild_replace_keys": force_rebuild_replace_keys,
        "force_rebuild_deletes": 0,
        "new_generation_keys": new_generation_keys,
        "skip_existing_keys": skip_existing_keys,
        "would_delete_total": len(would_delete_keys),
        "would_delete_keys": would_delete_keys,
        "deleted_total": len(deleted_keys),
        "deleted_keys": deleted_keys,
        "protected_prompt_version": COACH_CACHE_PROMPT_VERSION,
    }


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
    lines.append(
        f"- OPS: {batting.get('ops', 'N/A')} (rank={rankings.get('batting_ops', 'N/A')})"
    )
    lines.append(
        f"- 팀 타율: {batting.get('avg', 'N/A')} (rank={rankings.get('batting_avg', 'N/A')})"
    )
    lines.append(
        f"- 팀 ERA: {pitching.get('avg_era', 'N/A')} (rank={pitching.get('era_rank', 'N/A')})"
    )
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


def collect_batch_allowed_entity_names(tool_results: Dict[str, Any]) -> set[str]:
    names: set[str] = set()
    team_summary = tool_results.get("team_summary", {})
    advanced = tool_results.get("advanced_metrics", {})
    team_name = str(
        team_summary.get("team_name") or advanced.get("team_name") or ""
    ).strip()
    if team_name:
        names.add(team_name)
    for bucket in ("top_batters", "top_pitchers"):
        for item in team_summary.get(bucket, [])[:8]:
            player_name = str(item.get("player_name") or "").strip()
            if player_name:
                names.add(player_name)
    return names


def build_batch_fact_sheet(
    context: str, tool_results: Dict[str, Any]
) -> CoachFactSheet:
    fact_lines = [
        line.lstrip("- ").strip()
        for line in context.splitlines()
        if line.strip().startswith("-")
    ]
    numeric_tokens = collect_numeric_tokens(context)
    return CoachFactSheet(
        fact_lines=fact_lines,
        caveat_lines=[],
        allowed_entity_names=collect_batch_allowed_entity_names(tool_results),
        allowed_numeric_tokens=numeric_tokens,
        supported_fact_count=len(fact_lines),
        starters_confirmed=True,
        lineup_confirmed=True,
        series_context_confirmed=True,
        require_series_context=False,
    )


async def _mark_failed(pool, cache_key: str, message: str) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE coach_analysis_cache"
            " SET status = 'FAILED', error_message = %s, updated_at = now()"
            " WHERE cache_key = %s",
            (message, cache_key),
        )
        await conn.commit()


async def generate_and_cache_team(
    pool,
    team_id: str,
    year: int,
    game_type: str,
    focus: List[str],
    skip_completed: bool,
    only_missing: bool,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    start = datetime.now()
    cache_key, cache_key_payload = build_coach_cache_key(
        schema_version=CACHE_SCHEMA_VERSION,
        prompt_version=PROMPT_VERSION,
        home_team_code=team_id,
        away_team_code=None,
        year=year,
        game_type=game_type,
        focus=focus,
        question_override=None,
    )
    focus_signature = cache_key_payload["focus_signature"]
    question_signature = cache_key_payload["question_signature"]

    result: Dict[str, Any] = {
        "team": team_id,
        "year": year,
        "cache_key": cache_key,
        "focus_signature": focus_signature,
        "question_signature": question_signature,
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

    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT status, response_json FROM coach_analysis_cache"
            " WHERE cache_key = %s",
            (cache_key,),
        )
        row = await cur.fetchone()

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

    keep_previous = (
        bool(row) and row[0] == "COMPLETED" and bool(row[1]) and not skip_completed
    )

    if not keep_previous:
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO coach_analysis_cache
                    (cache_key, team_id, year, prompt_version, model_name, status)
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
            await conn.commit()

    async with pool.connection() as conn:
        db_query = DatabaseQueryTool(conn)
        tool_results = {
            "team_summary": await db_query.get_team_summary(team_id, year),
            "advanced_metrics": await db_query.get_team_advanced_metrics(team_id, year),
            "recent_form": await db_query.get_team_recent_form(team_id, year),
        }

    if not tool_results["team_summary"].get("found"):
        if keep_previous:
            result["previous_kept"] = True
        else:
            await _mark_failed(pool, cache_key, f"No team summary for {team_id} {year}")
        result["reason"] = "no_team_summary"
        result["failure_category"] = "no_team_summary"
        result["validator_hard_fail"] = True
        result["elapsed_seconds"] = (datetime.now() - start).total_seconds()
        return result

    context = format_batch_context(tool_results, year)
    fact_sheet = build_batch_fact_sheet(context, tool_results)
    question = (
        f"{team_id} {year}시즌 종합 분석"
        if not focus
        else f"{team_id} {year}시즌 {'/'.join(focus)} 집중 분석"
    )
    focus_section_requirements = build_focus_section_requirements(focus)
    coach_prompt = COACH_PROMPT_V2.format(
        question=question,
        context=context,
        focus_section_requirements=focus_section_requirements,
    )
    messages = [{"role": "user", "content": coach_prompt}]

    coach_llm = get_coach_llm_generator()

    try:
        chunks: List[str] = []
        async for chunk in coach_llm(messages):
            chunks.append(chunk)
        full_response = "".join(chunks)
    except Exception as exc:
        if keep_previous:
            result["previous_kept"] = True
        else:
            await _mark_failed(pool, cache_key, str(exc))
        result["reason"] = f"llm_error:{exc}"
        result["failure_category"] = "llm_error"
        result["validator_hard_fail"] = True
        result["elapsed_seconds"] = (datetime.now() - start).total_seconds()
        return result

    full_response = _remove_duplicate_json_start(full_response)
    parsed_response, parse_error, parse_meta = parse_coach_response_with_meta(
        full_response
    )
    result["normalization_applied"] = bool(parse_meta.get("normalization_applied"))
    result["normalization_reasons"] = list(parse_meta.get("normalization_reasons", []))

    if parsed_response:
        parsed_payload = parsed_response.model_dump()
        grounding_validation = validate_response_against_fact_sheet(
            parsed_payload,
            fact_sheet,
        )
        if grounding_validation.reasons:
            error_msg = ",".join(grounding_validation.reasons)
            if keep_previous:
                result["previous_kept"] = True
            else:
                await _mark_failed(pool, cache_key, error_msg)
            result["reason"] = error_msg
            result["failure_category"] = "grounding_validation_fail"
            result["validator_hard_fail"] = True
        else:
            payload_json = json.dumps(parsed_payload, ensure_ascii=False)
            try:
                async with pool.connection() as conn:
                    if keep_previous:
                        await conn.execute(
                            "UPDATE coach_analysis_cache"
                            " SET status = 'COMPLETED',"
                            " response_json = %s,"
                            " prompt_version = %s,"
                            " model_name = %s,"
                            " error_message = NULL,"
                            " updated_at = now()"
                            " WHERE cache_key = %s",
                            (payload_json, PROMPT_VERSION, MODEL_NAME, cache_key),
                        )
                        result["replaced_previous"] = True
                    else:
                        await conn.execute(
                            "UPDATE coach_analysis_cache"
                            " SET status = 'COMPLETED',"
                            " response_json = %s,"
                            " error_message = NULL,"
                            " updated_at = now()"
                            " WHERE cache_key = %s",
                            (payload_json, cache_key),
                        )
                    await conn.commit()
            except Exception as exc:
                # A single-statement UPDATE is atomic: on error the previous
                # row (COMPLETED or PENDING) is left as-is. Isolate per key.
                logger.exception("Cache write failed for %s", cache_key)
                result["reason"] = f"cache_write_error:{type(exc).__name__}: {exc}"
                result["failure_category"] = "cache_write_error"
                if keep_previous:
                    result["previous_kept"] = True
                    result.pop("replaced_previous", None)
                result["elapsed_seconds"] = (datetime.now() - start).total_seconds()
                return result

            warnings = validate_coach_response(parsed_response)
            warnings.extend(grounding_validation.warnings)
            critical_count = sum(
                1 for metric in parsed_response.key_metrics if metric.is_critical
            )

            result["status"] = "success"
            result["headline"] = parsed_response.headline
            result["json_parse_ok"] = True
            result["warnings_count"] = len(warnings)
            result["warnings"] = warnings
            result["critical_over_limit"] = critical_count > 2
    else:
        error_msg = parse_error or "JSON parsing failed"
        if keep_previous:
            result["previous_kept"] = True
        else:
            await _mark_failed(pool, cache_key, error_msg)
        result["reason"] = error_msg
        result["failure_category"] = parse_meta.get(
            "error_code"
        ) or classify_parse_error(error_msg)
        result["validator_hard_fail"] = True

    result["elapsed_seconds"] = (datetime.now() - start).total_seconds()
    return result


async def summarize_results(
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

    warnings_total = sum(
        int(row.get("warnings_count", 0)) for row in rows if row["status"] == "success"
    )
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
    owned_versions = list(BATCH_OWNED_PROMPT_VERSIONS)
    max_year = datetime.now().year + 1
    async with pool.connection() as conn:
        # Gate metrics only count rows this batch owns (and can clean up).
        # Rows written under the live API contract are reported separately as
        # informational totals and never fail the batch quality gate.
        cur = await conn.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE prompt_version = ANY(%s) AND prompt_version <> %s
                ),
                COUNT(*)
            FROM coach_analysis_cache
            WHERE year < %s OR year > %s
            """,
            (owned_versions, COACH_CACHE_PROMPT_VERSION, COACH_YEAR_MIN, max_year),
        )
        row = await cur.fetchone()
        invalid_year_rows = row[0]
        invalid_year_rows_all_versions = row[1]

        cur = await conn.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE prompt_version = ANY(%s) AND prompt_version <> %s
                ),
                COUNT(*)
            FROM coach_analysis_cache
            WHERE year = ANY(%s)
              AND UPPER(team_id) = ANY(%s)
            """,
            (
                owned_versions,
                COACH_CACHE_PROMPT_VERSION,
                options.years,
                legacy_aliases,
            ),
        )
        row = await cur.fetchone()
        legacy_team_rows = row[0]
        legacy_team_rows_all_versions = row[1]

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
        "target_focus": options.focus,
        "focus_signature": build_focus_signature(options.focus),
        "game_type": options.game_type.upper(),
        "coverage_rate": round(cases_ok / total, 4) if total else 0.0,
        "json_parse_success_rate": (
            round(parse_success_count / total, 4) if total else 0.0
        ),
        "validator_fail_count": validator_fail_count,
        "normalization_applied_count": normalization_applied_count,
        "warning_rate": round(warnings_total / success, 4) if success else 0.0,
        "critical_over_limit_rate": (
            round(critical_over_limit_count / success, 4) if success else 0.0
        ),
        "cache_hit_rate": round(skipped / total, 4) if total else 0.0,
        "runtime_seconds": round(elapsed_total, 3),
        "cache_invalid_year_count": int(invalid_year_rows),
        "legacy_residual_total": int(legacy_team_rows),
        "cache_invalid_year_count_all_versions": int(invalid_year_rows_all_versions),
        "legacy_residual_total_all_versions": int(legacy_team_rows_all_versions),
        "failure_category_counts": failure_category_counts,
    }

    return {
        "summary": summary,
        "options": {
            "years": options.years,
            "teams": options.teams,
            "focus": options.focus,
            "only_missing": options.only_missing,
            "force_rebuild": options.force_rebuild,
            "game_type": options.game_type,
            "delay_seconds": options.delay_seconds,
            "dry_run": options.dry_run,
            "cleanup_legacy": options.cleanup_legacy,
            "purge_invalid_years": options.purge_invalid_years,
        },
        "cleanup": cleanup_result,
        "details": rows,
    }


def parse_args() -> RunOptions:
    parser = argparse.ArgumentParser(
        description="Coach batch cache generator with quality report"
    )
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
        "--focus",
        default=None,
        help="Comma-separated focus profile (e.g. recent_form,bullpen). Default: empty(all)",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only create cache rows that do not already exist (status regardless)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help=(
            "Regenerate target batch keys even if COMPLETED; previous row is"
            " replaced only after the new result passes validation"
            " (verify-then-replace). Deletes nothing up front."
        ),
    )
    parser.add_argument(
        "--quality-report",
        default=None,
        help="Output JSON report path",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline quality evaluation after completing batch",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Plan cleanup only: report what would be deleted/replaced,"
            " no DELETE, no LLM, no writes"
        ),
    )
    parser.add_argument(
        "--purge-invalid-years",
        action="store_true",
        help=(
            "Opt-in deletion of batch-owned rows whose year is outside"
            " [COACH_YEAR_MIN, now.year+1]"
        ),
    )
    parser.add_argument(
        "--cleanup-legacy",
        action="store_true",
        help=(
            "Actually DELETE batch-owned cleanup candidates (prompt_mismatch,"
            " legacy_team, stale_pending, and invalid_year with"
            " --purge-invalid-years). Without it, cleanup is report-only."
            " Only the weekly quality workflow passes this flag."
        ),
    )
    args = parser.parse_args()

    resolver = TeamCodeResolver()
    years = parse_years(args.years)
    for year in years:
        if not is_valid_year(year):
            raise ValueError(f"invalid year in --years: {year}")
    if args.baseline and not args.quality_report:
        parser.error("--baseline requires --quality-report")

    teams = parse_teams(args.teams, resolver)
    focus = parse_focus(args.focus)
    return RunOptions(
        years=years,
        teams=teams,
        focus=focus,
        only_missing=args.only_missing,
        force_rebuild=args.force_rebuild,
        quality_report=args.quality_report,
        baseline=args.baseline,
        game_type=args.game_type,
        delay_seconds=args.delay_seconds,
        dry_run=args.dry_run,
        purge_invalid_years=args.purge_invalid_years,
        cleanup_legacy=args.cleanup_legacy,
    )


def run_baseline_evaluation(report_path: Path, years: List[int], game_type: str) -> int:
    cmd = [
        sys.executable,
        "scripts/evaluate_coach_quality.py",
        str(report_path),
        "--require-years",
        ",".join(map(str, years)),
        "--require-game-type",
        str(game_type).upper(),
    ]
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def build_unexpected_error_result(
    team: str, year: int, options: RunOptions, exc: Exception
) -> Dict[str, Any]:
    cache_key, payload = build_coach_cache_key(
        schema_version=CACHE_SCHEMA_VERSION,
        prompt_version=PROMPT_VERSION,
        home_team_code=team,
        away_team_code=None,
        year=year,
        game_type=options.game_type,
        focus=options.focus,
        question_override=None,
    )
    return {
        "team": team,
        "year": year,
        "cache_key": cache_key,
        "focus_signature": payload["focus_signature"],
        "question_signature": payload["question_signature"],
        "status": "failed",
        "reason": f"unexpected_error:{type(exc).__name__}: {exc}",
        "failure_category": "unexpected_error",
        "cache_hit": False,
        "json_parse_ok": False,
        "warnings_count": 0,
        "critical_over_limit": False,
        "validator_hard_fail": False,
        "normalization_applied": False,
        "normalization_reasons": [],
        "elapsed_seconds": 0.0,
    }


async def _run_batch(options: RunOptions, pool) -> int:
    if options.baseline and not options.quality_report:
        raise ValueError("--baseline requires --quality-report")

    run_start = datetime.now()
    resolver = TeamCodeResolver()

    print("=" * 72)
    print("Coach 배치 캐시 시작")
    print(f"Years: {options.years}")
    print(f"Teams: {options.teams}")
    print(f"Focus: {options.focus or ['all']}")
    print(
        f"Prompt: {PROMPT_VERSION} | Schema: {CACHE_SCHEMA_VERSION} | GameType: {options.game_type}"
    )
    print(
        f"Options: only_missing={options.only_missing},"
        f" force_rebuild={options.force_rebuild},"
        f" dry_run={options.dry_run},"
        f" purge_invalid_years={options.purge_invalid_years},"
        f" cleanup_legacy={options.cleanup_legacy}"
    )
    print("=" * 72)

    cleanup_result = await cleanup_cache_rows(pool, options, resolver)
    logger.info("Cleanup result: %s", cleanup_result)

    if options.dry_run:
        report = {
            "dry_run": True,
            "generated_at": datetime.now().isoformat(),
            "options": {
                "years": options.years,
                "teams": options.teams,
                "focus": options.focus,
                "only_missing": options.only_missing,
                "force_rebuild": options.force_rebuild,
                "game_type": options.game_type,
                "delay_seconds": options.delay_seconds,
                "purge_invalid_years": options.purge_invalid_years,
                "cleanup_legacy": options.cleanup_legacy,
                "dry_run": options.dry_run,
            },
            "cleanup": cleanup_result,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if options.quality_report:
            report_path = Path(options.quality_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return 0

    results: List[Dict[str, Any]] = []
    total_cases = len(options.years) * len(options.teams)
    case_index = 0

    for year in options.years:
        for team in options.teams:
            case_index += 1
            print(f"[{case_index}/{total_cases}] {year} {team} processing...")
            try:
                item = await generate_and_cache_team(
                    pool=pool,
                    team_id=team,
                    year=year,
                    game_type=options.game_type,
                    focus=options.focus,
                    skip_completed=(not options.force_rebuild),
                    only_missing=(options.only_missing and not options.force_rebuild),
                    force_rebuild=options.force_rebuild,
                )
            except Exception as exc:  # isolate DB/replace errors per key
                logger.exception("Team %s %s failed with unexpected error", year, team)
                item = build_unexpected_error_result(team, year, options, exc)
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

    report = await summarize_results(results, options, cleanup_result, pool, resolver)
    report["run_started_at"] = run_start.isoformat()
    report["run_finished_at"] = datetime.now().isoformat()

    print("\n" + "=" * 72)
    print("완료 요약")
    print("=" * 72)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    baseline_failed = False
    if options.quality_report:
        report_path = Path(options.quality_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Quality report written: {report_path}")

        if options.baseline:
            print("\n" + "-" * 72)
            print("기준 품질 평가(Baseline Evaluation) 실행 중...")
            baseline_rc = run_baseline_evaluation(
                report_path, options.years, options.game_type
            )
            if baseline_rc != 0:
                baseline_failed = True
                print(f"Baseline evaluation failed (exit code={baseline_rc})")

    return 0 if report["summary"]["failed"] == 0 and not baseline_failed else 1


async def async_main(options: RunOptions) -> int:
    # get_connection_pool()은 open=False로 생성된 AsyncConnectionPool이므로
    # 이벤트 루프 안에서 명시적으로 열고, 종료 시 닫는다.
    pool = get_connection_pool()
    try:
        await pool.open(wait=True, timeout=10.0)
        return await _run_batch(options, pool)
    finally:
        await close_connection_pool()


def main() -> int:
    try:
        options = parse_args()
        return asyncio.run(async_main(options))
    except Exception as exc:
        logger.error("Batch failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
