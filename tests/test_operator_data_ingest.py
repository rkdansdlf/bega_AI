from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts import ingest_operator_data_handoff as ingest


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.existing_hashes: dict[str, str] = {}
        self.player_rows: list[Mapping[str, Any]] = [{"player_id": "1001"}]
        self.columns_by_table: dict[str, set[str]] = {
            table_name: set(columns)
            for table_name, columns in ingest.REQUIRED_TABLE_COLUMNS.items()
        }
        self.columns_by_table["game_lineups"].update(
            {
                "player_name",
                "is_starter",
                "notes",
            }
        )
        self.columns = self.columns_by_table["game_lineups"]
        self.games: dict[str, Mapping[str, Any]] = {
            "20260605LGKT0": {
                "game_id": "20260605LGKT0",
                "home_team": "LG",
                "away_team": "KT",
            }
        }
        self.has_lineup_conflict_target = True
        self._rows: list[Mapping[str, Any]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((query, params))
        lowered = query.lower()
        if "select payload_hash from operator_data_items" in lowered:
            queue_id = str(params[0])
            payload_hash = self.existing_hashes.get(queue_id)
            self._rows = [{"payload_hash": payload_hash}] if payload_hash else []
        elif "from player_basic" in lowered:
            self._rows = list(self.player_rows)
        elif "from information_schema.columns" in lowered:
            table_name = str(params[0])
            columns = self.columns_by_table.get(table_name, set())
            self._rows = [{"column_name": column} for column in sorted(columns)]
        elif "from pg_index" in lowered:
            self._rows = [{"exists": 1}] if self.has_lineup_conflict_target else []
        elif "from game where game_id" in lowered:
            game_id = str(params[0])
            row = self.games.get(game_id)
            self._rows = [row] if row else []
        else:
            self._rows = []

    def fetchone(self) -> Mapping[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Mapping[str, Any]]:
        return list(self._rows)


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_obj = cursor
        self.commit_calls = 0
        self.rollback_calls = 0

    def cursor(self, *args, **kwargs) -> FakeCursor:
        del args, kwargs
        return self.cursor_obj

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1


def _row(domain: str = "schedule_window", **overrides: Any) -> dict[str, Any]:
    base = {
        "queue_id": "ODQ-0001",
        "priority": "P0",
        "domain": domain,
        "contract_code": "SCHEDULE_WINDOW_REQUIRED",
        "question": "오늘 KBO 경기 일정 알려줘.",
        "operator_status": "ready_for_validation",
        "validation_status": "pass",
        "apply_eligible": True,
        "apply_target": "operator_schedule_items",
        "payload": {
            "game_date": "2026-06-05",
            "game_id": "20260605LGKT0",
            "home_team": "LG",
            "away_team": "KT",
            "stadium_name": "잠실",
            "start_time": "18:30",
            "game_status": "scheduled",
        },
        "source_metadata": {
            "source_name": "operator",
            "source_checked_at": "2026-06-05",
            "is_verified": True,
            "confidence": 0.95,
        },
    }
    base.update(overrides)
    return base


def _lineup_row(**overrides: Any) -> dict[str, Any]:
    base = _row(
        "game_day_lineup",
        contract_code="GAME_DAY_LINEUP_REQUIRED",
        apply_target="game_lineups/manual_starters",
        payload={
            "game_id": "20260605LGKT0",
            "team_code": "LG",
            "player_name": "홍길동",
            "batting_order": "1",
            "position": "CF",
            "announced_at": "2026-06-05T16:00:00",
        },
    )
    base.update(overrides)
    return base


def test_pending_bundle_rows_skip_without_db() -> None:
    row = _row(
        operator_status="pending",
        apply_eligible=False,
        skip_reason="operator_status_pending",
    )

    report = ingest.build_ingest_report([row], conn=None)

    assert report["summary"]["eligible_rows"] == 0
    assert report["plans"][0]["action"] == "skipped"
    assert report["plans"][0]["skip_reason"] == "operator_status_pending"


def test_dry_run_checks_existing_but_does_not_write() -> None:
    cursor = FakeCursor()
    conn = FakeConnection(cursor)

    report = ingest.build_ingest_report([_row()], conn=conn, apply=False)

    assert report["plans"][0]["action"] == "insert"
    assert conn.commit_calls == 0
    write_sql = [
        query
        for query, _params in cursor.executed
        if query.strip().lower().startswith(("insert", "update", "delete"))
    ]
    assert write_sql == []


def test_dry_run_reports_update_with_overwrite_flag_without_write() -> None:
    cursor = FakeCursor()
    cursor.existing_hashes["ODQ-0001"] = "different"
    conn = FakeConnection(cursor)

    report = ingest.build_ingest_report(
        [_row()],
        conn=conn,
        apply=False,
        allow_overwrite=True,
    )

    assert report["plans"][0]["action"] == "update"
    assert conn.commit_calls == 0
    write_sql = [
        query
        for query, _params in cursor.executed
        if query.strip().lower().startswith(("insert", "update", "delete"))
    ]
    assert write_sql == []


def test_eligible_rows_without_db_report_db_required() -> None:
    report = ingest.build_ingest_report([_row()], conn=None, apply=False)

    assert report["summary"]["issue_counts"]["error"] == 1
    assert report["issues"][0]["code"] == "db_required"
    assert report["plans"][0]["skip_reason"] == "db_required"


def test_same_payload_hash_is_noop() -> None:
    row = _row()
    cursor = FakeCursor()
    cursor.existing_hashes[row["queue_id"]] = ingest._payload_hash(row)
    conn = FakeConnection(cursor)

    report = ingest.build_ingest_report([row], conn=conn, apply=True)

    assert report["plans"][0]["action"] == "noop"
    assert report["plans"][0]["skip_reason"] == "same_payload_hash"
    assert conn.commit_calls == 1


def test_different_payload_hash_requires_overwrite_flag() -> None:
    cursor = FakeCursor()
    cursor.existing_hashes["ODQ-0001"] = "different"
    conn = FakeConnection(cursor)

    report = ingest.build_ingest_report([_row()], conn=conn, apply=True)

    assert report["summary"]["issue_counts"]["error"] == 1
    assert report["issues"][0]["code"] == "overwrite_requires_flag"
    assert report["plans"][0]["action"] == "skipped"
    assert conn.rollback_calls == 1


def test_skipped_lineup_rows_do_not_create_starter_plan_rows() -> None:
    home = _lineup_row(
        queue_id="ODQ-HOME",
        payload={
            "game_id": "20260605LGKT0",
            "team_code": "LG",
            "player_name": "홍길동",
            "batting_order": "1",
            "position": "P",
            "announced_at": "2026-06-05T16:00:00",
        },
    )
    away = _lineup_row(
        queue_id="ODQ-AWAY",
        payload={
            "game_id": "20260605LGKT0",
            "team_code": "KT",
            "player_name": "김철수",
            "batting_order": "1",
            "position": "P",
            "announced_at": "2026-06-05T16:00:00",
        },
    )
    cursor = FakeCursor()
    cursor.existing_hashes["ODQ-HOME"] = "different"
    cursor.existing_hashes["ODQ-AWAY"] = "different"
    conn = FakeConnection(cursor)

    report = ingest.build_ingest_report([home, away], conn=conn, apply=False)

    assert [plan["action"] for plan in report["plans"]] == ["skipped", "skipped"]
    assert {plan["skip_reason"] for plan in report["plans"]} == {"overwrite_requires_flag"}
    assert {issue["code"] for issue in report["issues"]} == {"overwrite_requires_flag"}
    assert report["starter_plan_rows"] == []


def test_non_p0_domain_is_skipped_before_db() -> None:
    row = _row(
        "venue_ticket",
        apply_target="operator_venue_guides",
        payload={"stadium_name": "잠실"},
    )

    report = ingest.build_ingest_report([row], conn=None, domains=["venue_ticket"])

    assert report["summary"]["eligible_rows"] == 0
    assert report["plans"][0]["skip_reason"] == "operator_data_v1_non_p0_domain"


def test_unverified_or_low_confidence_rows_skip_before_db() -> None:
    unverified = _row(
        queue_id="ODQ-0100",
        source_metadata={
            "source_name": "operator",
            "source_checked_at": "2026-06-05",
            "is_verified": False,
            "confidence": 0.95,
        },
    )
    low_confidence = _row(
        queue_id="ODQ-0101",
        source_metadata={
            "source_name": "operator",
            "source_checked_at": "2026-06-05",
            "is_verified": True,
            "confidence": 0.69,
        },
    )

    report = ingest.build_ingest_report([unverified, low_confidence], conn=None)

    assert report["summary"]["eligible_rows"] == 0
    assert [plan["skip_reason"] for plan in report["plans"]] == [
        "not_verified",
        "low_confidence",
    ]


def test_game_day_lineup_requires_exact_player_resolution() -> None:
    cursor = FakeCursor()
    cursor.player_rows = []
    conn = FakeConnection(cursor)
    row = _lineup_row()

    report = ingest.build_ingest_report([row], conn=conn, apply=False)

    assert report["summary"]["issue_counts"]["error"] == 1
    assert report["issues"][0]["code"] == "player_resolution_not_unique"


def test_game_day_lineup_requires_reader_visible_columns() -> None:
    cursor = FakeCursor()
    cursor.columns_by_table["game_lineups"].remove("notes")
    cursor.columns_by_table["game_lineups"].remove("player_name")
    conn = FakeConnection(cursor)

    report = ingest.build_ingest_report([_lineup_row()], conn=conn, apply=False)

    assert report["summary"]["issue_counts"]["error"] == 1
    assert report["issues"][0]["code"] == "schema_missing_columns"
    assert "notes,player_name" in report["issues"][0]["message"]
    assert report["plans"][0]["action"] == "skipped"


def test_game_day_lineup_requires_upsert_conflict_target() -> None:
    cursor = FakeCursor()
    cursor.has_lineup_conflict_target = False
    conn = FakeConnection(cursor)

    report = ingest.build_ingest_report([_lineup_row()], conn=conn, apply=False)

    assert report["summary"]["issue_counts"]["error"] == 1
    assert report["issues"][0]["code"] == "missing_lineup_conflict_target"
    assert report["plans"][0]["action"] == "skipped"
    conflict_queries = [
        query for query, _params in cursor.executed if "FROM pg_index" in query
    ]
    assert conflict_queries
    assert "array_agg(att.attname::text ORDER BY keys.ordinality)" in conflict_queries[0]


def test_dry_run_reports_schema_conflicts_without_writes() -> None:
    cursor = FakeCursor()
    cursor.columns_by_table["operator_schedule_items"].remove("game_status")
    conn = FakeConnection(cursor)

    report = ingest.build_ingest_report([_row()], conn=conn, apply=False)

    assert report["summary"]["issue_counts"]["error"] == 1
    assert report["issues"][0]["code"] == "schema_missing_columns"
    assert report["plans"][0]["action"] == "skipped"
    write_sql = [
        query
        for query, _params in cursor.executed
        if query.strip().lower().startswith(("insert", "update", "delete"))
    ]
    assert write_sql == []


def test_default_p0_dry_run_over_real_194_baseline_without_db() -> None:
    normalized_path = (
        Path(__file__).parent.parent
        / "reports"
        / "operator_data_validation"
        / "post_db_fast_path_docker_kbo500"
        / "operator_data_normalized_rows.jsonl"
    )
    if not normalized_path.exists():
        pytest.skip("local operator-data validation report is not present")

    rows = ingest.load_normalized_rows(normalized_path)

    report = ingest.build_ingest_report(rows, conn=None, apply=False)

    assert report["summary"]["total_rows"] == 194
    assert report["summary"]["selected_domains"] == list(ingest.P0_DOMAINS)
    assert report["summary"]["eligible_rows"] == 0
    assert report["summary"]["action_counts"] == {"skipped": 194}
    assert report["summary"]["issue_counts"] == {"error": 0, "warning": 0}
