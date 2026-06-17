#!/usr/bin/env python3
"""Check DB prerequisites for P0 operator-data recovery.

This script is read-only. It only checks target table columns and the lineup
upsert conflict target needed before strict validation / ingest apply.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

try:
    import psycopg
    from psycopg.rows import dict_row
except ModuleNotFoundError as exc:
    psycopg = None
    dict_row = None
    _PSYCOPG_IMPORT_ERROR = exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import ingest_operator_data_handoff as ingest

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_db_prereqs"
    / "post_db_fast_path_docker_kbo500"
)
ISSUE_FIELDNAMES = ["severity", "code", "message", "table_name", "missing_columns"]
BASE_READER_TABLE_COLUMNS = {
    "game": {"game_id", "home_team", "away_team", "game_date", "game_status"},
    "player_basic": {"player_id", "name"},
}
REQUIRED_TABLE_COLUMNS = {
    **ingest.REQUIRED_TABLE_COLUMNS,
    **BASE_READER_TABLE_COLUMNS,
}
LINEUP_CONFLICT_COLUMNS = ("game_id", "team_code", "batting_order")


@dataclass(frozen=True)
class DbPrereqIssue:
    severity: str
    code: str
    message: str
    table_name: str = ""
    missing_columns: str = ""

    def to_record(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "table_name": self.table_name,
            "missing_columns": self.missing_columns,
        }


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _issue(
    *,
    code: str,
    message: str,
    severity: str = "error",
    table_name: str = "",
    missing_columns: Sequence[str] = (),
) -> DbPrereqIssue:
    return DbPrereqIssue(
        severity=severity,
        code=code,
        message=message,
        table_name=table_name,
        missing_columns=",".join(missing_columns),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _table_columns(cur: Any, table_name: str) -> set[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
        """,
        (table_name,),
    )
    columns: set[str] = set()
    for row in list(cur.fetchall() or []):
        columns.add(
            _normalize_text(
                row.get("column_name") if isinstance(row, Mapping) else row[0]
            )
        )
    return columns


def _has_unique_conflict_target(
    cur: Any, table_name: str, columns: Sequence[str]
) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM pg_index idx
        JOIN pg_class tbl ON tbl.oid = idx.indrelid
        WHERE tbl.relname = %s
          AND idx.indisunique = true
          AND (
            SELECT array_agg(att.attname::text ORDER BY keys.ordinality)
            FROM regexp_split_to_table(idx.indkey::text, ' ')
              WITH ORDINALITY AS keys(attnum_text, ordinality)
            JOIN pg_attribute att
              ON att.attrelid = tbl.oid
             AND att.attnum = keys.attnum_text::smallint
          ) = %s::text[]
        LIMIT 1
        """,
        (table_name, list(columns)),
    )
    return cur.fetchone() is not None


def build_db_prereq_report(
    *,
    cur: Any,
    output_dir: Path,
    checked_tables: Mapping[str, set[str]] = REQUIRED_TABLE_COLUMNS,
) -> dict[str, Any]:
    del output_dir
    issues: list[DbPrereqIssue] = []
    table_results: list[dict[str, Any]] = []
    missing_column_total = 0

    for table_name, required_columns in sorted(checked_tables.items()):
        available_columns = _table_columns(cur, table_name)
        missing_columns = sorted(required_columns - available_columns)
        missing_column_total += len(missing_columns)
        table_results.append(
            {
                "table_name": table_name,
                "required_column_count": len(required_columns),
                "available_column_count": len(available_columns),
                "missing_column_count": len(missing_columns),
                "missing_columns": missing_columns,
                "status": "pass" if not missing_columns else "fail",
            }
        )
        if missing_columns:
            code = (
                "missing_table" if not available_columns else "schema_missing_columns"
            )
            issues.append(
                _issue(
                    code=code,
                    table_name=table_name,
                    missing_columns=missing_columns,
                    message=(
                        f"{table_name} is missing required columns: "
                        f"{','.join(missing_columns)}"
                    ),
                )
            )

    lineup_conflict_target_exists = _has_unique_conflict_target(
        cur,
        "game_lineups",
        LINEUP_CONFLICT_COLUMNS,
    )
    if not lineup_conflict_target_exists:
        issues.append(
            _issue(
                code="missing_lineup_conflict_target",
                table_name="game_lineups",
                message=(
                    "game_lineups must have a unique or primary-key conflict target on "
                    "(game_id, team_code, batting_order)."
                ),
            )
        )

    issue_counts = Counter(issue.severity for issue in issues)
    status = "fail" if issue_counts.get("error", 0) else "pass"
    return {
        "summary": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "checked_table_count": len(checked_tables),
            "missing_column_count": missing_column_total,
            "lineup_conflict_target_exists": lineup_conflict_target_exists,
            "issue_counts": {
                "error": issue_counts.get("error", 0),
                "warning": issue_counts.get("warning", 0),
            },
        },
        "table_results": table_results,
        "issues": [issue.to_record() for issue in issues],
    }


def _build_unavailable_report(*, code: str, message: str) -> dict[str, Any]:
    issue = _issue(code=code, message=message)
    return {
        "summary": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "fail",
            "checked_table_count": 0,
            "missing_column_count": 0,
            "lineup_conflict_target_exists": False,
            "issue_counts": {"error": 1, "warning": 0},
        },
        "table_results": [],
        "issues": [issue.to_record()],
    }


def _render_handoff(report: Mapping[str, Any]) -> str:
    summary = report.get("summary") or {}
    issues = report.get("issues") or []
    lines = [
        "# P0 Operator Data DB Prerequisites",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Checked tables: `{summary.get('checked_table_count', 0)}`",
        f"- Missing columns: `{summary.get('missing_column_count', 0)}`",
        f"- Lineup conflict target: `{str(summary.get('lineup_conflict_target_exists', False)).lower()}`",
        f"- Errors: `{(summary.get('issue_counts') or {}).get('error', 0)}`",
        "",
    ]
    if issues:
        lines.extend(["## Issues", ""])
        for issue in issues:
            lines.append(
                "- "
                f"`{issue.get('code')}` "
                f"table=`{issue.get('table_name', '')}` "
                f"missing=`{issue.get('missing_columns', '')}`: {issue.get('message', '')}"
            )
        lines.extend(
            [
                "",
                "## Next Step",
                "",
                "Fix DB prerequisites before strict validation, ingest dry-run, or controlled apply.",
            ]
        )
    else:
        lines.extend(
            [
                "## Next Step",
                "",
                "DB prerequisites passed. Continue with strict validation using DB checks enabled.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _write_report(output_dir: Path, report: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "db_prereq_summary.json", report)
    _write_csv(output_dir / "db_prereq_issues.csv", report["issues"], ISSUE_FIELDNAMES)
    _write_csv(
        output_dir / "db_prereq_tables.csv",
        report["table_results"],
        [
            "table_name",
            "required_column_count",
            "available_column_count",
            "missing_column_count",
            "missing_columns",
            "status",
        ],
    )
    (output_dir / "db_prereq_handoff.md").write_text(
        _render_handoff(report), encoding="utf-8"
    )


def _connect(db_url: str) -> Any:
    if psycopg is None:
        raise RuntimeError(f"psycopg is required: {_PSYCOPG_IMPORT_ERROR}")
    return psycopg.connect(db_url, row_factory=dict_row)


def run_check(*, db_url: str, output_dir: Path) -> dict[str, Any]:
    if not _normalize_text(db_url):
        report = _build_unavailable_report(
            code="db_url_missing",
            message="POSTGRES_DB_URL or --db-url is required for DB prerequisite checks.",
        )
        _write_report(output_dir, report)
        return report
    conn = None
    try:
        conn = _connect(db_url)
        with conn.cursor() as cur:
            report = build_db_prereq_report(cur=cur, output_dir=output_dir)
    except Exception as exc:
        report = _build_unavailable_report(
            code="db_connection_failed",
            message=f"Could not run DB prerequisite checks ({exc.__class__.__name__}).",
        )
    finally:
        if conn is not None:
            conn.close()
    _write_report(output_dir, report)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check P0 operator-data DB prerequisites."
    )
    parser.add_argument(
        "--db-url", default="", help="PostgreSQL URL. Defaults to POSTGRES_DB_URL."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        default=True,
        help="Exit 0 even when DB prerequisites fail.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    db_url = _normalize_text(args.db_url) or os.environ.get("POSTGRES_DB_URL", "")
    report = run_check(db_url=db_url, output_dir=Path(args.output_dir))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.strict and report["summary"]["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
