#!/usr/bin/env python3
"""Apply operator-provided starter pitchers from a Coach audit CSV.

This script only consumes an operator-filled internal CSV. It does not collect
or infer baseball data from external sources.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

os.environ.setdefault("BEGA_SKIP_APP_INIT", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "coach_manual_starters"
REQUIRED_COLUMNS = {"game_id", "home_pitcher", "away_pitcher"}


@dataclass(frozen=True)
class ManualStarterInput:
    row_number: int
    game_id: str
    game_date: str
    home_team_id: str
    away_team_id: str
    home_pitcher: str
    away_pitcher: str


@dataclass(frozen=True)
class GameStarterSnapshot:
    game_id: str
    game_date: str
    game_status: str
    home_team_id: str
    away_team_id: str
    home_pitcher: str
    away_pitcher: str


@dataclass(frozen=True)
class StarterUpdatePlan:
    row_number: int
    game_id: str
    game_date: str
    home_team_id: str
    away_team_id: str
    current_home_pitcher: str
    current_away_pitcher: str
    new_home_pitcher: str
    new_away_pitcher: str
    changed: bool


@dataclass(frozen=True)
class StarterApplyIssue:
    row_number: int
    game_id: str
    severity: str
    code: str
    message: str


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _issue(
    *,
    row_number: int,
    game_id: str,
    severity: str,
    code: str,
    message: str,
) -> StarterApplyIssue:
    return StarterApplyIssue(
        row_number=row_number,
        game_id=game_id,
        severity=severity,
        code=code,
        message=message,
    )


def _read_csv_rows(path: Path) -> tuple[List[Dict[str, str]], List[StarterApplyIssue]]:
    issues: List[StarterApplyIssue] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing_columns:
            issues.append(
                _issue(
                    row_number=1,
                    game_id="",
                    severity="error",
                    code="missing_required_columns",
                    message=f"CSV missing required columns: {', '.join(missing_columns)}",
                )
            )
            return [], issues
        return [dict(row) for row in reader], issues


def parse_manual_starter_inputs(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[List[ManualStarterInput], List[StarterApplyIssue]]:
    inputs: List[ManualStarterInput] = []
    issues: List[StarterApplyIssue] = []
    seen: Dict[str, ManualStarterInput] = {}

    for index, row in enumerate(rows, start=2):
        game_id = _normalize_text(row.get("game_id"))
        home_pitcher = _normalize_text(row.get("home_pitcher"))
        away_pitcher = _normalize_text(row.get("away_pitcher"))
        if not game_id:
            issues.append(
                _issue(
                    row_number=index,
                    game_id="",
                    severity="error",
                    code="missing_game_id",
                    message="game_id is required.",
                )
            )
            continue
        if not home_pitcher and not away_pitcher:
            issues.append(
                _issue(
                    row_number=index,
                    game_id=game_id,
                    severity="warning",
                    code="empty_starter_pair",
                    message="Both home_pitcher and away_pitcher are empty; row skipped.",
                )
            )
            continue
        if not home_pitcher or not away_pitcher:
            issues.append(
                _issue(
                    row_number=index,
                    game_id=game_id,
                    severity="error",
                    code="incomplete_starter_pair",
                    message="Both home_pitcher and away_pitcher must be provided together.",
                )
            )
            continue

        parsed = ManualStarterInput(
            row_number=index,
            game_id=game_id,
            game_date=_normalize_text(row.get("game_date")),
            home_team_id=_normalize_text(row.get("home_team_id")),
            away_team_id=_normalize_text(row.get("away_team_id")),
            home_pitcher=home_pitcher,
            away_pitcher=away_pitcher,
        )
        previous = seen.get(game_id)
        if previous:
            if (
                previous.home_pitcher == parsed.home_pitcher
                and previous.away_pitcher == parsed.away_pitcher
            ):
                issues.append(
                    _issue(
                        row_number=index,
                        game_id=game_id,
                        severity="warning",
                        code="duplicate_same_starters",
                        message="Duplicate game_id with identical starters; row skipped.",
                    )
                )
            else:
                issues.append(
                    _issue(
                        row_number=index,
                        game_id=game_id,
                        severity="error",
                        code="duplicate_conflicting_starters",
                        message="Duplicate game_id has conflicting starter values.",
                    )
                )
            continue
        seen[game_id] = parsed
        inputs.append(parsed)

    return inputs, issues


def _normalize_snapshot(row: Mapping[str, Any]) -> GameStarterSnapshot:
    return GameStarterSnapshot(
        game_id=_normalize_text(row.get("game_id")),
        game_date=_normalize_text(row.get("game_date")),
        game_status=_normalize_text(row.get("game_status")).upper(),
        home_team_id=_normalize_text(row.get("home_team")),
        away_team_id=_normalize_text(row.get("away_team")),
        home_pitcher=_normalize_text(row.get("home_pitcher")),
        away_pitcher=_normalize_text(row.get("away_pitcher")),
    )


def build_update_plan(
    inputs: Sequence[ManualStarterInput],
    existing_games: Mapping[str, GameStarterSnapshot],
    *,
    allow_non_scheduled: bool = False,
    allow_overwrite: bool = False,
) -> tuple[List[StarterUpdatePlan], List[StarterApplyIssue]]:
    plans: List[StarterUpdatePlan] = []
    issues: List[StarterApplyIssue] = []

    for item in inputs:
        existing = existing_games.get(item.game_id)
        if existing is None:
            issues.append(
                _issue(
                    row_number=item.row_number,
                    game_id=item.game_id,
                    severity="error",
                    code="game_not_found",
                    message="No matching game row exists.",
                )
            )
            continue
        if existing.game_status != "SCHEDULED" and not allow_non_scheduled:
            issues.append(
                _issue(
                    row_number=item.row_number,
                    game_id=item.game_id,
                    severity="error",
                    code="non_scheduled_game",
                    message=f"Game status is {existing.game_status}; expected SCHEDULED.",
                )
            )
            continue
        if item.home_team_id and item.home_team_id != existing.home_team_id:
            issues.append(
                _issue(
                    row_number=item.row_number,
                    game_id=item.game_id,
                    severity="error",
                    code="home_team_mismatch",
                    message=(
                        f"CSV home_team_id={item.home_team_id} does not match "
                        f"DB home_team={existing.home_team_id}."
                    ),
                )
            )
            continue
        if item.away_team_id and item.away_team_id != existing.away_team_id:
            issues.append(
                _issue(
                    row_number=item.row_number,
                    game_id=item.game_id,
                    severity="error",
                    code="away_team_mismatch",
                    message=(
                        f"CSV away_team_id={item.away_team_id} does not match "
                        f"DB away_team={existing.away_team_id}."
                    ),
                )
            )
            continue

        would_overwrite = (
            existing.home_pitcher
            and existing.home_pitcher != item.home_pitcher
        ) or (
            existing.away_pitcher
            and existing.away_pitcher != item.away_pitcher
        )
        if would_overwrite and not allow_overwrite:
            issues.append(
                _issue(
                    row_number=item.row_number,
                    game_id=item.game_id,
                    severity="error",
                    code="starter_overwrite_requires_flag",
                    message="Existing starter differs; rerun with --allow-overwrite to change it.",
                )
            )
            continue

        changed = (
            existing.home_pitcher != item.home_pitcher
            or existing.away_pitcher != item.away_pitcher
        )
        if not changed:
            issues.append(
                _issue(
                    row_number=item.row_number,
                    game_id=item.game_id,
                    severity="warning",
                    code="starter_values_unchanged",
                    message="Starter values already match the CSV.",
                )
            )
        plans.append(
            StarterUpdatePlan(
                row_number=item.row_number,
                game_id=item.game_id,
                game_date=existing.game_date or item.game_date,
                home_team_id=existing.home_team_id,
                away_team_id=existing.away_team_id,
                current_home_pitcher=existing.home_pitcher,
                current_away_pitcher=existing.away_pitcher,
                new_home_pitcher=item.home_pitcher,
                new_away_pitcher=item.away_pitcher,
                changed=changed,
            )
        )

    return plans, issues


def _fetch_existing_games(game_ids: Sequence[str]) -> Dict[str, GameStarterSnapshot]:
    if not game_ids:
        return {}
    from app.deps import get_connection_pool
    from psycopg.rows import dict_row

    sql = """
        SELECT
            game_id,
            game_date,
            game_status,
            home_team,
            away_team,
            home_pitcher,
            away_pitcher
        FROM game
        WHERE game_id = ANY(%s)
    """
    pool = get_connection_pool()
    with pool.connection() as conn:
        rows = conn.cursor(row_factory=dict_row).execute(sql, (list(game_ids),)).fetchall()
    return {_normalize_text(row.get("game_id")): _normalize_snapshot(row) for row in rows}


def _apply_updates(plans: Sequence[StarterUpdatePlan]) -> int:
    changed_plans = [plan for plan in plans if plan.changed]
    if not changed_plans:
        return 0
    from app.deps import get_connection_pool

    sql = """
        UPDATE game
        SET
            home_pitcher = %s,
            away_pitcher = %s,
            updated_at = NOW()
        WHERE game_id = %s
    """
    pool = get_connection_pool()
    applied = 0
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            for plan in changed_plans:
                cursor.execute(
                    sql,
                    (plan.new_home_pitcher, plan.new_away_pitcher, plan.game_id),
                )
                applied += int(cursor.rowcount or 0)
        conn.commit()
    return applied


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _plan_rows(plans: Sequence[StarterUpdatePlan]) -> List[Dict[str, Any]]:
    return [
        {
            "row_number": plan.row_number,
            "game_id": plan.game_id,
            "game_date": plan.game_date,
            "away_team_id": plan.away_team_id,
            "home_team_id": plan.home_team_id,
            "current_away_pitcher": plan.current_away_pitcher,
            "current_home_pitcher": plan.current_home_pitcher,
            "new_away_pitcher": plan.new_away_pitcher,
            "new_home_pitcher": plan.new_home_pitcher,
            "changed": plan.changed,
        }
        for plan in plans
    ]


def _issue_rows(issues: Sequence[StarterApplyIssue]) -> List[Dict[str, Any]]:
    return [
        {
            "row_number": issue.row_number,
            "game_id": issue.game_id,
            "severity": issue.severity,
            "code": issue.code,
            "message": issue.message,
        }
        for issue in issues
    ]


def _write_reports(
    *,
    output_dir: Path,
    options: Mapping[str, Any],
    plans: Sequence[StarterUpdatePlan],
    issues: Sequence[StarterApplyIssue],
    applied_updates: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "options": dict(options),
        "summary": {
            "planned_rows": len(plans),
            "changed_rows": sum(1 for plan in plans if plan.changed),
            "unchanged_rows": sum(1 for plan in plans if not plan.changed),
            "applied_updates": applied_updates,
            "error_count": error_count,
            "warning_count": warning_count,
        },
    }

    summary_path = output_dir / f"coach_manual_starter_apply_summary_{timestamp}.json"
    plan_path = output_dir / f"coach_manual_starter_apply_plan_{timestamp}.csv"
    issues_path = output_dir / f"coach_manual_starter_apply_issues_{timestamp}.csv"
    latest_summary_path = output_dir / "coach_manual_starter_apply_summary_latest.json"
    latest_plan_path = output_dir / "coach_manual_starter_apply_plan_latest.csv"
    latest_issues_path = output_dir / "coach_manual_starter_apply_issues_latest.csv"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(
        plan_path,
        _plan_rows(plans),
        [
            "row_number",
            "game_id",
            "game_date",
            "away_team_id",
            "home_team_id",
            "current_away_pitcher",
            "current_home_pitcher",
            "new_away_pitcher",
            "new_home_pitcher",
            "changed",
        ],
    )
    _write_csv(
        issues_path,
        _issue_rows(issues),
        ["row_number", "game_id", "severity", "code", "message"],
    )

    shutil.copyfile(summary_path, latest_summary_path)
    shutil.copyfile(plan_path, latest_plan_path)
    shutil.copyfile(issues_path, latest_issues_path)

    return {
        "summary": summary["summary"],
        "paths": {
            "summary_json": str(summary_path),
            "plan_csv": str(plan_path),
            "issues_csv": str(issues_path),
            "latest_summary_json": str(latest_summary_path),
            "latest_plan_csv": str(latest_plan_path),
            "latest_issues_csv": str(latest_issues_path),
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply operator-filled starter pitcher CSV rows to game.",
    )
    parser.add_argument("csv_path", help="Operator-filled manual starter CSV path.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist changes. Without this flag the script only writes a dry-run plan.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow changing non-empty starter values that differ from the CSV.",
    )
    parser.add_argument(
        "--allow-non-scheduled",
        action="store_true",
        help="Allow updates when game_status is not SCHEDULED.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Relative paths resolve under bega_AI.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        default=True,
        help="Exit 0 even when validation errors are present.",
    )
    return parser.parse_args(argv)


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else Path.cwd() / path


def _resolve_output_dir(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    csv_path = _resolve_path(args.csv_path)
    output_dir = _resolve_output_dir(args.output_dir)

    rows, read_issues = _read_csv_rows(csv_path)
    inputs, parse_issues = parse_manual_starter_inputs(rows)
    existing = _fetch_existing_games([item.game_id for item in inputs])
    plans, plan_issues = build_update_plan(
        inputs,
        existing,
        allow_non_scheduled=args.allow_non_scheduled,
        allow_overwrite=args.allow_overwrite,
    )
    issues = [*read_issues, *parse_issues, *plan_issues]
    error_count = sum(1 for issue in issues if issue.severity == "error")
    applied_updates = 0
    if args.apply and error_count == 0:
        applied_updates = _apply_updates(plans)

    output = _write_reports(
        output_dir=output_dir,
        options={
            "csv_path": str(csv_path),
            "apply": bool(args.apply),
            "allow_overwrite": bool(args.allow_overwrite),
            "allow_non_scheduled": bool(args.allow_non_scheduled),
        },
        plans=plans,
        issues=issues,
        applied_updates=applied_updates,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.strict and error_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
