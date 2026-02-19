#!/usr/bin/env python3
"""
Validate team-code runtime policy for canonical window and outside-window fallback.

Usage:
  python scripts/verify_canonical_transition.py --mode all
  python scripts/verify_canonical_transition.py --mode canonical_window --output /tmp/window.json
  python scripts/verify_canonical_transition.py --mode outside_regression --outside-years 2001-2009,2018-2020
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Tuple

import psycopg

# Avoid importing full FastAPI app while reusing app modules.
os.environ.setdefault("PYTEST_CURRENT_TEST", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.tools.database_query import DatabaseQueryTool, clear_coach_cache
from app.tools.game_query import GameQueryTool

CANONICAL_TEAMS = ["SS", "LT", "LG", "DB", "KIA", "KH", "HH", "SSG", "NC", "KT"]
LEGACY_CODES = ["SK", "OB", "HT", "WO", "DO", "KI", "KW"]


def parse_year_ranges(spec: str) -> List[int]:
    years: List[int] = []
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if start <= end:
                years.extend(range(start, end + 1))
            else:
                years.extend(range(start, end - 1, -1))
            continue
        years.append(int(token))
    return sorted(set(years))


@contextmanager
def temporary_team_code_env(
    *,
    read_mode: str,
    window_start: int,
    window_end: int,
    outside_mode: str,
):
    keys = [
        "TEAM_CODE_READ_MODE",
        "TEAM_CODE_CANONICAL_WINDOW_START",
        "TEAM_CODE_CANONICAL_WINDOW_END",
        "TEAM_CODE_OUTSIDE_WINDOW_MODE",
    ]
    previous = {k: os.environ.get(k) for k in keys}
    os.environ["TEAM_CODE_READ_MODE"] = read_mode
    os.environ["TEAM_CODE_CANONICAL_WINDOW_START"] = str(window_start)
    os.environ["TEAM_CODE_CANONICAL_WINDOW_END"] = str(window_end)
    os.environ["TEAM_CODE_OUTSIDE_WINDOW_MODE"] = outside_mode
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _dsn_host(dsn: str) -> str | None:
    if "@" not in dsn:
        return None
    return dsn.split("@", 1)[1].split("/", 1)[0]


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def evaluate_canonical_window(
    output: Dict[str, object],
    *,
    strict_canonical_window: bool,
    strict_legacy_residual: bool,
) -> Dict[str, object]:
    canonical = output.get("canonical_window")
    checks: List[Dict[str, object]] = []

    if not isinstance(canonical, dict):
        checks.extend(
            [
                {
                    "name": "canonical_window_result_present",
                    "passed": False,
                    "enforced": strict_canonical_window,
                    "details": "canonical_window result missing",
                },
                {
                    "name": "legacy_residual_result_present",
                    "passed": False,
                    "enforced": strict_legacy_residual,
                    "details": "legacy_residuals missing",
                },
            ]
        )
        failed_required = [
            c["name"] for c in checks if c["enforced"] and not c["passed"]
        ]
        return {
            "passed": len(failed_required) == 0,
            "strict_canonical_window": strict_canonical_window,
            "strict_legacy_residual": strict_legacy_residual,
            "failed_required_checks": failed_required,
            "failed_optional_checks": [
                c["name"] for c in checks if (not c["enforced"]) and (not c["passed"])
            ],
            "checks": checks,
            "cases": 0,
            "all_ok": 0,
            "legacy_residual_total": 0,
            "runtime_seconds": None,
        }

    totals = canonical.get("totals", {})
    if not isinstance(totals, dict):
        totals = {}
    cases = _safe_int(totals.get("cases"))
    all_ok = _safe_int(totals.get("all_ok"))
    all_cases_ok = cases > 0 and all_ok == cases

    legacy_residuals = canonical.get("legacy_residuals", {})
    if isinstance(legacy_residuals, dict):
        legacy_residual_total = sum(_safe_int(v) for v in legacy_residuals.values())
    else:
        legacy_residual_total = 0

    checks.extend(
        [
            {
                "name": "canonical_window_all_cases_ok",
                "passed": all_cases_ok,
                "enforced": strict_canonical_window,
                "details": f"all_ok={all_ok}, cases={cases}",
            },
            {
                "name": "legacy_residual_total_zero",
                "passed": legacy_residual_total == 0,
                "enforced": strict_legacy_residual,
                "details": f"legacy_residual_total={legacy_residual_total}",
            },
        ]
    )

    failed_required = [c["name"] for c in checks if c["enforced"] and not c["passed"]]
    failed_optional = [
        c["name"] for c in checks if (not c["enforced"]) and (not c["passed"])
    ]
    return {
        "passed": len(failed_required) == 0,
        "strict_canonical_window": strict_canonical_window,
        "strict_legacy_residual": strict_legacy_residual,
        "failed_required_checks": failed_required,
        "failed_optional_checks": failed_optional,
        "checks": checks,
        "cases": cases,
        "all_ok": all_ok,
        "legacy_residual_total": legacy_residual_total,
        "runtime_seconds": canonical.get("runtime_seconds"),
    }


def evaluate_outside_regression(
    output: Dict[str, object],
    *,
    strict_outside_regression: bool,
) -> Dict[str, object]:
    outside = output.get("outside_regression")
    checks: List[Dict[str, object]] = []

    if not isinstance(outside, dict):
        checks.append(
            {
                "name": "outside_regression_result_present",
                "passed": False,
                "enforced": strict_outside_regression,
                "details": "outside_regression result missing",
            }
        )
        failed_required = [
            c["name"] for c in checks if c["enforced"] and not c["passed"]
        ]
        return {
            "passed": len(failed_required) == 0,
            "strict_outside_regression": strict_outside_regression,
            "failed_required_checks": failed_required,
            "failed_optional_checks": [
                c["name"] for c in checks if (not c["enforced"]) and (not c["passed"])
            ],
            "checks": checks,
            "total_cases": 0,
            "additional_miss_count": 0,
            "error_diff_count": 0,
            "runtime_seconds": None,
        }

    additional_miss_count = _safe_int(outside.get("additional_miss_count"))
    error_diff_count = _safe_int(outside.get("error_diff_count"))
    total_cases = _safe_int(outside.get("total_cases"))

    checks.extend(
        [
            {
                "name": "outside_additional_miss_zero",
                "passed": additional_miss_count == 0,
                "enforced": strict_outside_regression,
                "details": f"additional_miss_count={additional_miss_count}",
            },
            {
                "name": "outside_error_diff_zero",
                "passed": error_diff_count == 0,
                "enforced": strict_outside_regression,
                "details": f"error_diff_count={error_diff_count}",
            },
        ]
    )

    failed_required = [c["name"] for c in checks if c["enforced"] and not c["passed"]]
    failed_optional = [
        c["name"] for c in checks if (not c["enforced"]) and (not c["passed"])
    ]
    return {
        "passed": len(failed_required) == 0,
        "strict_outside_regression": strict_outside_regression,
        "failed_required_checks": failed_required,
        "failed_optional_checks": failed_optional,
        "checks": checks,
        "total_cases": total_cases,
        "additional_miss_count": additional_miss_count,
        "error_diff_count": error_diff_count,
        "runtime_seconds": outside.get("runtime_seconds"),
    }


def _append_github_step_summary(lines: List[str]) -> None:
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def _build_step_summary(output: Dict[str, object]) -> List[str]:
    lines: List[str] = [
        "## Canonical Guard Summary",
        "",
    ]

    canonical_eval = output.get("evaluation", {}).get("canonical_window")
    canonical_result = output.get("canonical_window")
    if isinstance(canonical_eval, dict) and isinstance(canonical_result, dict):
        lines.extend(
            [
                "### Canonical Window",
                "",
                "| key | value |",
                "|---|---:|",
                f"| cases | {canonical_eval.get('cases', 0)} |",
                f"| all_ok | {canonical_eval.get('all_ok', 0)} |",
                f"| legacy_residual_total | {canonical_eval.get('legacy_residual_total', 0)} |",
                f"| runtime_seconds | {canonical_eval.get('runtime_seconds', canonical_result.get('runtime_seconds'))} |",
                f"| passed | {canonical_eval.get('passed')} |",
                "",
            ]
        )
        failed_required = canonical_eval.get("failed_required_checks") or []
        if failed_required:
            lines.append(f"- required_failures: {', '.join(map(str, failed_required))}")
        failed_optional = canonical_eval.get("failed_optional_checks") or []
        if failed_optional:
            lines.append(f"- optional_failures: {', '.join(map(str, failed_optional))}")
        if failed_required or failed_optional:
            lines.append("")

    outside_eval = output.get("evaluation", {}).get("outside_regression")
    if isinstance(outside_eval, dict):
        lines.extend(
            [
                "### Outside Regression",
                "",
                "| key | value |",
                "|---|---:|",
                f"| total_cases | {outside_eval.get('total_cases', 0)} |",
                f"| additional_miss_count | {outside_eval.get('additional_miss_count', 0)} |",
                f"| error_diff_count | {outside_eval.get('error_diff_count', 0)} |",
                f"| runtime_seconds | {outside_eval.get('runtime_seconds')} |",
                f"| strict_enforced | {outside_eval.get('strict_outside_regression')} |",
                f"| passed | {outside_eval.get('passed')} |",
                "",
            ]
        )

    if output.get("fatal_error"):
        lines.extend(
            [
                "### Fatal Error",
                "",
                f"- {output['fatal_error']}",
                "",
            ]
        )

    return lines


def _resolve_database_url() -> str:
    from_guard_env = os.getenv("CANONICAL_GUARD_DB_URL_RO")
    if from_guard_env and from_guard_env.strip():
        return from_guard_env.strip()
    return get_settings().database_url


def _run_matrix(
    conn: psycopg.Connection,
    *,
    teams: Iterable[str],
    years: Iterable[int],
) -> Dict[Tuple[int, str, str], Dict[str, object]]:
    clear_coach_cache()
    db_tool = DatabaseQueryTool(conn)
    game_tool = GameQueryTool(conn)

    out: Dict[Tuple[int, str, str], Dict[str, object]] = {}
    for year in years:
        for team in teams:
            summary = db_tool.get_team_summary(team, year)
            advanced = db_tool.get_team_advanced_metrics(team, year)
            last_game = game_tool.get_team_last_game_date(team, year, "regular_season")

            out[(year, team, "summary")] = {
                "found": bool(summary.get("found")),
                "error": summary.get("error"),
            }
            out[(year, team, "advanced")] = {
                "found": bool(advanced.get("found")),
                "error": advanced.get("error"),
            }
            out[(year, team, "last_game")] = {
                "found": bool(last_game.get("found")),
                "error": last_game.get("error"),
            }
    return out


def run_canonical_window_smoke(
    conn: psycopg.Connection,
    *,
    teams: List[str],
    years: List[int],
    window_start: int,
    window_end: int,
    outside_mode: str,
) -> Dict[str, object]:
    started = time.time()
    result: Dict[str, object] = {
        "years": years,
        "teams": teams,
        "totals": {
            "cases": 0,
            "summary_ok": 0,
            "advanced_ok": 0,
            "last_game_ok": 0,
            "all_ok": 0,
        },
        "failures": [],
        "legacy_residuals": {},
        "runtime_seconds": None,
    }

    with temporary_team_code_env(
        read_mode="canonical_only",
        window_start=window_start,
        window_end=window_end,
        outside_mode=outside_mode,
    ):
        clear_coach_cache()
        db_tool = DatabaseQueryTool(conn)
        game_tool = GameQueryTool(conn)

        for year in years:
            for team in teams:
                result["totals"]["cases"] += 1

                summary = db_tool.get_team_summary(team, year)
                advanced = db_tool.get_team_advanced_metrics(team, year)
                last_game = game_tool.get_team_last_game_date(
                    team, year, "regular_season"
                )

                ok_summary = bool(summary.get("found")) and not summary.get("error")
                ok_advanced = bool(advanced.get("found")) and not advanced.get("error")
                ok_last = bool(last_game.get("found")) and not last_game.get("error")

                if ok_summary:
                    result["totals"]["summary_ok"] += 1
                if ok_advanced:
                    result["totals"]["advanced_ok"] += 1
                if ok_last:
                    result["totals"]["last_game_ok"] += 1
                if ok_summary and ok_advanced and ok_last:
                    result["totals"]["all_ok"] += 1
                else:
                    result["failures"].append(
                        {
                            "year": year,
                            "team": team,
                            "summary": {
                                "found": bool(summary.get("found")),
                                "error": summary.get("error"),
                            },
                            "advanced": {
                                "found": bool(advanced.get("found")),
                                "error": advanced.get("error"),
                            },
                            "last_game": {
                                "found": bool(last_game.get("found")),
                                "error": last_game.get("error"),
                            },
                        }
                    )

        queries = {
            "game": """
                SELECT COUNT(*)::bigint
                FROM game g
                JOIN kbo_seasons ks ON g.season_id = ks.season_id
                WHERE ks.season_year BETWEEN %s AND %s
                  AND (
                    g.home_team = ANY(%s)
                    OR g.away_team = ANY(%s)
                    OR g.winning_team = ANY(%s)
                  )
            """,
            "player_season_batting": """
                SELECT COUNT(*)::bigint
                FROM player_season_batting
                WHERE season BETWEEN %s AND %s
                  AND team_code = ANY(%s)
            """,
            "player_season_pitching": """
                SELECT COUNT(*)::bigint
                FROM player_season_pitching
                WHERE season BETWEEN %s AND %s
                  AND team_code = ANY(%s)
            """,
            "game_lineups": """
                SELECT COUNT(*)::bigint
                FROM game_lineups gl
                JOIN game g ON gl.game_id = g.game_id
                JOIN kbo_seasons ks ON g.season_id = ks.season_id
                WHERE ks.season_year BETWEEN %s AND %s
                  AND gl.team_code = ANY(%s)
            """,
            "game_batting_stats": """
                SELECT COUNT(*)::bigint
                FROM game_batting_stats gs
                JOIN game g ON gs.game_id = g.game_id
                JOIN kbo_seasons ks ON g.season_id = ks.season_id
                WHERE ks.season_year BETWEEN %s AND %s
                  AND gs.team_code = ANY(%s)
            """,
            "game_pitching_stats": """
                SELECT COUNT(*)::bigint
                FROM game_pitching_stats gs
                JOIN game g ON gs.game_id = g.game_id
                JOIN kbo_seasons ks ON g.season_id = ks.season_id
                WHERE ks.season_year BETWEEN %s AND %s
                  AND gs.team_code = ANY(%s)
            """,
            "team_daily_roster": """
                SELECT COUNT(*)::bigint
                FROM team_daily_roster
                WHERE EXTRACT(YEAR FROM roster_date) BETWEEN %s AND %s
                  AND team_code = ANY(%s)
            """,
        }

        with conn.cursor() as cur:
            for table, query in queries.items():
                if table == "game":
                    cur.execute(
                        query,
                        (
                            window_start,
                            window_end,
                            LEGACY_CODES,
                            LEGACY_CODES,
                            LEGACY_CODES,
                        ),
                    )
                else:
                    cur.execute(query, (window_start, window_end, LEGACY_CODES))
                result["legacy_residuals"][table] = int(cur.fetchone()[0])

    result["runtime_seconds"] = round(time.time() - started, 2)
    return result


def run_outside_window_regression(
    conn: psycopg.Connection,
    *,
    teams: List[str],
    years: List[int],
    window_start: int,
    window_end: int,
    outside_mode: str,
) -> Dict[str, object]:
    started = time.time()
    result: Dict[str, object] = {
        "years": years,
        "teams": teams,
        "total_cases": len(years) * len(teams) * 3,
        "dual_found_total": 0,
        "canonical_found_total": 0,
        "additional_miss_count": 0,
        "additional_hit_count": 0,
        "error_diff_count": 0,
        "additional_miss_samples": [],
        "additional_hit_samples": [],
        "error_diff_samples": [],
        "runtime_seconds": None,
    }

    with temporary_team_code_env(
        read_mode="dual",
        window_start=window_start,
        window_end=window_end,
        outside_mode=outside_mode,
    ):
        dual = _run_matrix(conn, teams=teams, years=years)

    with temporary_team_code_env(
        read_mode="canonical_only",
        window_start=window_start,
        window_end=window_end,
        outside_mode=outside_mode,
    ):
        canonical = _run_matrix(conn, teams=teams, years=years)

    additional_miss = []
    additional_hit = []
    error_diff = []

    for key, dual_row in dual.items():
        canonical_row = canonical[key]

        if dual_row["found"]:
            result["dual_found_total"] += 1
        if canonical_row["found"]:
            result["canonical_found_total"] += 1

        if dual_row["found"] and (not canonical_row["found"]):
            additional_miss.append(
                {
                    "year": key[0],
                    "team": key[1],
                    "fn": key[2],
                    "dual_error": dual_row["error"],
                    "canonical_error": canonical_row["error"],
                }
            )
        if (not dual_row["found"]) and canonical_row["found"]:
            additional_hit.append({"year": key[0], "team": key[1], "fn": key[2]})
        if dual_row["error"] != canonical_row["error"]:
            error_diff.append(
                {
                    "year": key[0],
                    "team": key[1],
                    "fn": key[2],
                    "dual_error": dual_row["error"],
                    "canonical_error": canonical_row["error"],
                }
            )

    result["additional_miss_count"] = len(additional_miss)
    result["additional_hit_count"] = len(additional_hit)
    result["error_diff_count"] = len(error_diff)
    result["additional_miss_samples"] = additional_miss[:20]
    result["additional_hit_samples"] = additional_hit[:20]
    result["error_diff_samples"] = error_diff[:20]
    result["runtime_seconds"] = round(time.time() - started, 2)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify canonical window runtime policy and outside-window fallback."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "canonical_window", "outside_regression"],
        default="all",
    )
    parser.add_argument(
        "--window-start",
        type=int,
        default=2021,
        help="Canonical window start year.",
    )
    parser.add_argument(
        "--window-end",
        type=int,
        default=2025,
        help="Canonical window end year.",
    )
    parser.add_argument(
        "--outside-mode",
        choices=["dual", "canonical_only"],
        default="dual",
        help="Mode for years outside canonical window when read mode is canonical_only.",
    )
    parser.add_argument(
        "--outside-years",
        default="2001-2009,2018-2020",
        help="Outside-window year ranges for dual-vs-canonical regression checks.",
    )
    parser.add_argument(
        "--teams",
        default=",".join(CANONICAL_TEAMS),
        help="Comma-separated canonical team codes.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON file path.",
    )
    parser.add_argument(
        "--strict-canonical-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when canonical window matrix has any miss/error.",
    )
    parser.add_argument(
        "--strict-legacy-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when legacy residual rows are greater than 0.",
    )
    parser.add_argument(
        "--strict-outside-regression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail on outside-window regression miss/error diffs.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    teams = [token.strip() for token in args.teams.split(",") if token.strip()]
    window_years = list(
        range(
            min(args.window_start, args.window_end),
            max(args.window_start, args.window_end) + 1,
        )
    )
    outside_years = parse_year_ranges(args.outside_years)

    database_url = _resolve_database_url()
    output: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_host": _dsn_host(database_url),
        "input": {
            "mode": args.mode,
            "window_start": args.window_start,
            "window_end": args.window_end,
            "outside_mode": args.outside_mode,
            "window_years": window_years,
            "outside_years": outside_years,
            "teams": teams,
            "strict_canonical_window": args.strict_canonical_window,
            "strict_legacy_residual": args.strict_legacy_residual,
            "strict_outside_regression": args.strict_outside_regression,
        },
    }
    exit_code = 0

    try:
        with psycopg.connect(database_url, autocommit=True) as conn:
            if args.mode in {"all", "canonical_window"}:
                output["canonical_window"] = run_canonical_window_smoke(
                    conn,
                    teams=teams,
                    years=window_years,
                    window_start=args.window_start,
                    window_end=args.window_end,
                    outside_mode=args.outside_mode,
                )

            if args.mode in {"all", "outside_regression"}:
                output["outside_regression"] = run_outside_window_regression(
                    conn,
                    teams=teams,
                    years=outside_years,
                    window_start=args.window_start,
                    window_end=args.window_end,
                    outside_mode=args.outside_mode,
                )
    except Exception as exc:
        output["fatal_error"] = str(exc)
        exit_code = 1

    evaluations: Dict[str, object] = {}
    if args.mode in {"all", "canonical_window"}:
        canonical_eval = evaluate_canonical_window(
            output,
            strict_canonical_window=args.strict_canonical_window,
            strict_legacy_residual=args.strict_legacy_residual,
        )
        evaluations["canonical_window"] = canonical_eval
        if not canonical_eval["passed"]:
            exit_code = 1

    if args.mode in {"all", "outside_regression"}:
        outside_eval = evaluate_outside_regression(
            output,
            strict_outside_regression=args.strict_outside_regression,
        )
        evaluations["outside_regression"] = outside_eval
        if not outside_eval["passed"]:
            exit_code = 1

    output["evaluation"] = evaluations

    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote report: {out_path}")

    _append_github_step_summary(_build_step_summary(output))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
