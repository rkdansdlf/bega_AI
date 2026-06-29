from __future__ import annotations

import asyncio

from datetime import date
from typing import Any, Mapping

from app.tools.operator_data_query import (
    PARTIAL_NOTICE,
    try_build_operator_fast_path_result,
)


class FakeCursor:
    def __init__(self, rows_by_table: dict[str, list[Mapping[str, Any]]]) -> None:
        self.rows_by_table = rows_by_table
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self._rows: list[Mapping[str, Any]] = []

    async def __aenter__(self) -> "FakeCursor":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((query, params))
        lowered = query.lower()
        table = ""
        for candidate in (
            "operator_schedule_items",
            "operator_season_events",
            "operator_roster_events",
            "game_lineups",
        ):
            if candidate in lowered:
                table = candidate
                break
        self._rows = list(self.rows_by_table.get(table, []))

    async def fetchall(self) -> list[Mapping[str, Any]]:
        return list(self._rows)


class FakeConnection:
    def __init__(self, rows_by_table: dict[str, list[Mapping[str, Any]]]) -> None:
        self.cursor_obj = FakeCursor(rows_by_table)

    def cursor(self, *args, **kwargs) -> FakeCursor:
        del args, kwargs
        return self.cursor_obj


class BrokenCursor:
    async def __aenter__(self) -> "BrokenCursor":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        del query, params
        raise RuntimeError("schema missing")


class BrokenConnection:
    def cursor(self, *args, **kwargs) -> BrokenCursor:
        del args, kwargs
        return BrokenCursor()


def test_schedule_fast_path_returns_partial_operator_answer() -> None:
    conn = FakeConnection(
        {
            "operator_schedule_items": [
                {
                    "queue_id": "ODQ-0001",
                    "game_date": "2026-06-05",
                    "game_id": "20260605LGKT0",
                    "home_team": "LG",
                    "away_team": "KT",
                    "stadium_name": "잠실",
                    "start_time": "18:30",
                    "game_status": "scheduled",
                    "source_checked_at": "2026-06-05",
                    "confidence": 0.95,
                }
            ]
        }
    )

    result = asyncio.run(
        try_build_operator_fast_path_result(
            conn,
            "오늘 KBO 경기 일정 알려줘.",
            today=date(2026, 6, 5),
        )
    )

    assert result is not None
    assert result["strategy"] == "operator_data_fast_path"
    assert result["grounding_mode"] == "operator_provided"
    assert result["source_tier"] == "operator_data"
    assert result["operator_data_partial"] is True
    assert PARTIAL_NOTICE in result["answer"]
    assert "18:30" in result["answer"]
    assert "LG" in result["answer"]


def test_roster_fast_path_uses_verified_operator_rows_only() -> None:
    conn = FakeConnection(
        {
            "operator_roster_events": [
                {
                    "queue_id": "ODQ-0020",
                    "season_year": 2026,
                    "team_code": "LG",
                    "player_name": "홍길동",
                    "roster_event_type": "부상",
                    "effective_date": "2026-06-04",
                    "status_text": "엔트리 제외",
                    "source_checked_at": "2026-06-05",
                    "confidence": 0.9,
                }
            ]
        }
    )

    result = asyncio.run(
        try_build_operator_fast_path_result(conn, "2026년 LG 부상자 명단은 어디서 봐?")
    )

    assert result is not None
    assert result["operator_data_domain"] == "roster_news"
    assert PARTIAL_NOTICE in result["answer"]
    assert "홍길동" in result["answer"]


def test_roster_fast_path_requires_team_and_year_scope() -> None:
    conn = FakeConnection(
        {
            "operator_roster_events": [
                {
                    "queue_id": "ODQ-0020",
                    "season_year": 2026,
                    "team_code": "LG",
                    "player_name": "홍길동",
                    "roster_event_type": "부상",
                    "effective_date": "2026-06-04",
                    "status_text": "엔트리 제외",
                    "source_checked_at": "2026-06-05",
                    "confidence": 0.9,
                }
            ]
        }
    )

    no_year = asyncio.run(
        try_build_operator_fast_path_result(conn, "LG 부상자 명단은 어디서 봐?")
    )
    no_team = asyncio.run(
        try_build_operator_fast_path_result(conn, "2026년 부상자 명단은 어디서 봐?")
    )

    assert no_year is None
    assert no_team is None


def test_lineup_fast_path_requires_date_scope() -> None:
    conn = FakeConnection(
        {
            "game_lineups": [
                {
                    "game_id": "20260605LGKT0",
                    "team_code": "LG",
                    "player_name": "홍길동",
                    "position": "CF",
                    "batting_order": 1,
                    "notes": {
                        "source_type": "manual_lineup",
                        "queue_id": "ODQ-0030",
                        "is_verified": True,
                        "confidence": 0.9,
                    },
                    "game_date": "2026-06-05",
                    "home_team": "LG",
                    "away_team": "KT",
                }
            ]
        }
    )

    result = asyncio.run(try_build_operator_fast_path_result(conn, "LG 라인업 알려줘."))

    assert result is None


def test_no_operator_rows_returns_none_for_manual_fallback() -> None:
    conn = FakeConnection({})

    result = asyncio.run(
        try_build_operator_fast_path_result(
            conn,
            "오늘 KBO 경기 일정 알려줘.",
            today=date(2026, 6, 5),
        )
    )

    assert result is None


def test_invalid_operator_rows_return_none_for_manual_fallback() -> None:
    conn = FakeConnection(
        {
            "operator_schedule_items": [
                {
                    "queue_id": "ODQ-0001",
                    "game_date": "2026-06-05",
                    "game_id": "20260605LGKT0",
                    "home_team": "LG",
                    "away_team": "KT",
                    "stadium_name": "잠실",
                    "start_time": "18:30",
                    "game_status": "scheduled",
                    "source_checked_at": "2026-06-05",
                    "is_verified": False,
                    "confidence": 0.95,
                },
                {
                    "queue_id": "ODQ-0002",
                    "game_date": "2026-06-05",
                    "game_id": "20260605KISS0",
                    "home_team": "KIA",
                    "away_team": "SSG",
                    "stadium_name": "광주",
                    "start_time": "18:30",
                    "game_status": "scheduled",
                    "source_checked_at": "2026-06-05",
                    "is_verified": True,
                    "confidence": 0.69,
                },
            ]
        }
    )

    result = asyncio.run(
        try_build_operator_fast_path_result(
            conn,
            "오늘 KBO 경기 일정 알려줘.",
            today=date(2026, 6, 5),
        )
    )

    assert result is None


def test_schema_errors_return_none_for_manual_fallback() -> None:
    result = asyncio.run(
        try_build_operator_fast_path_result(
            BrokenConnection(),
            "오늘 KBO 경기 일정 알려줘.",
            today=date(2026, 6, 5),
        )
    )

    assert result is None
