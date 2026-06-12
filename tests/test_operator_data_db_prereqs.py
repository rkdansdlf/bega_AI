from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts import check_operator_data_p0_db_prereqs as prereqs


class FakeCursor:
    def __init__(self) -> None:
        self.columns_by_table: dict[str, set[str]] = {
            table_name: set(columns)
            for table_name, columns in prereqs.REQUIRED_TABLE_COLUMNS.items()
        }
        self.has_lineup_conflict_target = True
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self._rows: list[Mapping[str, Any]] = []

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((query, params))
        lowered = query.lower()
        if "from information_schema.columns" in lowered:
            table_name = str(params[0])
            self._rows = [
                {"column_name": column}
                for column in sorted(self.columns_by_table.get(table_name, set()))
            ]
        elif "from pg_index" in lowered:
            self._rows = [{"exists": 1}] if self.has_lineup_conflict_target else []
        else:
            self._rows = []

    def fetchone(self) -> Mapping[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Mapping[str, Any]]:
        return list(self._rows)


def _codes(report: Mapping[str, Any]) -> set[str]:
    return {str(issue["code"]) for issue in report["issues"]}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_db_prereqs_pass_when_schema_and_conflict_target_exist(tmp_path: Path) -> None:
    cursor = FakeCursor()

    report = prereqs.build_db_prereq_report(cur=cursor, output_dir=tmp_path)

    assert report["summary"]["status"] == "pass"
    assert report["summary"]["checked_table_count"] == len(prereqs.REQUIRED_TABLE_COLUMNS)
    assert report["summary"]["missing_column_count"] == 0
    assert report["summary"]["lineup_conflict_target_exists"] is True
    assert report["issues"] == []
    conflict_queries = [
        query for query, _params in cursor.executed if "FROM pg_index" in query
    ]
    assert conflict_queries
    assert "array_agg(att.attname::text ORDER BY keys.ordinality)" in conflict_queries[0]


def test_db_prereqs_fail_on_missing_columns() -> None:
    cursor = FakeCursor()
    cursor.columns_by_table["operator_schedule_items"].remove("game_status")

    report = prereqs.build_db_prereq_report(cur=cursor, output_dir=Path("unused"))

    assert report["summary"]["status"] == "fail"
    assert "schema_missing_columns" in _codes(report)
    issue = report["issues"][0]
    assert issue["table_name"] == "operator_schedule_items"
    assert issue["missing_columns"] == "game_status"


def test_db_prereqs_fail_on_missing_table_and_lineup_conflict_target() -> None:
    cursor = FakeCursor()
    cursor.columns_by_table["operator_roster_events"] = set()
    cursor.has_lineup_conflict_target = False

    report = prereqs.build_db_prereq_report(cur=cursor, output_dir=Path("unused"))

    assert report["summary"]["status"] == "fail"
    assert {"missing_table", "missing_lineup_conflict_target"}.issubset(_codes(report))


def test_run_check_without_db_url_writes_failure_outputs(tmp_path: Path) -> None:
    report = prereqs.run_check(db_url="", output_dir=tmp_path)

    assert report["summary"]["status"] == "fail"
    assert _codes(report) == {"db_url_missing"}
    assert (tmp_path / "db_prereq_summary.json").exists()
    assert (tmp_path / "db_prereq_issues.csv").exists()
    assert (tmp_path / "db_prereq_tables.csv").exists()
    assert (tmp_path / "db_prereq_handoff.md").exists()
    payload = json.loads((tmp_path / "db_prereq_summary.json").read_text(encoding="utf-8"))
    issues = _read_csv(tmp_path / "db_prereq_issues.csv")
    assert payload["summary"]["status"] == "fail"
    assert issues[0]["code"] == "db_url_missing"


def test_main_no_strict_returns_zero_for_missing_db_url(tmp_path: Path) -> None:
    exit_code = prereqs.main(["--output-dir", str(tmp_path), "--no-strict"])

    assert exit_code == 0


def test_backend_postgresql_migration_contains_operator_data_prereqs() -> None:
    migration = (
        Path(__file__).resolve().parents[2]
        / "bega_backend"
        / "BEGA_PROJECT"
        / "src"
        / "main"
        / "resources"
        / "db"
        / "migration_postgresql"
        / "V150__create_operator_data_p0_tables.sql"
    )
    if not migration.exists():
        pytest.skip("backend PostgreSQL operator-data migration is not present")

    sql = migration.read_text(encoding="utf-8").lower()

    expected_created_tables = set(prereqs.REQUIRED_TABLE_COLUMNS) - {"game", "player_basic"}
    for table_name in expected_created_tables:
        assert f"create table if not exists {table_name}" in sql
    assert "ux_game_lineups_game_team_batting_order" in sql
    assert "on game_lineups (game_id, team_code, batting_order)" in sql
