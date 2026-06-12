from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from app.core.rag import RAGPipeline


class _OperatorCursor:
    def __init__(self, rows: list[Mapping[str, Any]]) -> None:
        self.rows = rows

    def __enter__(self) -> "_OperatorCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        del query, params

    def fetchall(self) -> list[Mapping[str, Any]]:
        return list(self.rows)


class _OperatorConn:
    def __init__(self, rows: list[Mapping[str, Any]]) -> None:
        self.rows = rows

    def cursor(self, *args, **kwargs) -> _OperatorCursor:
        del args, kwargs
        return _OperatorCursor(self.rows)


def _operator_pipeline(enabled: bool, rows: list[Mapping[str, Any]]) -> RAGPipeline:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = SimpleNamespace(operator_data_fast_path_enabled=enabled)
    pipeline.connection = _OperatorConn(rows)
    pipeline._pool = None
    return pipeline


def test_operator_data_fast_path_flag_off_keeps_manual_contract() -> None:
    pipeline = _operator_pipeline(
        False,
        [
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
        ],
    )

    result = pipeline._build_operator_or_static_kbo_result("오늘 KBO 경기 일정 알려줘.")

    assert result is not None
    assert result["strategy"] == "manual_baseball_data_required"
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in result["answer"]


def test_operator_data_fast_path_flag_on_returns_operator_answer() -> None:
    pipeline = _operator_pipeline(
        True,
        [
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
        ],
    )

    result = pipeline._build_operator_or_static_kbo_result("오늘 KBO 경기 일정 알려줘.")

    assert result is not None
    assert result["strategy"] == "operator_data_fast_path"
    assert result["source_tier"] == "operator_data"
    assert result["operator_data_partial"] is True
    assert "확인된 항목만" in result["answer"]


def test_operator_data_fast_path_without_rows_keeps_manual_contract() -> None:
    pipeline = _operator_pipeline(True, [])

    result = pipeline._build_operator_or_static_kbo_result("오늘 KBO 경기 일정 알려줘.")

    assert result is not None
    assert result["strategy"] == "manual_baseball_data_required"
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in result["answer"]


def test_operator_data_fast_path_underspecified_rows_keep_manual_contract() -> None:
    pipeline = _operator_pipeline(
        True,
        [
            {
                "queue_id": "ODQ-0030",
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
                "season_year": 2026,
                "roster_event_type": "부상",
                "effective_date": "2026-06-04",
                "status_text": "엔트리 제외",
                "source_checked_at": "2026-06-05",
                "confidence": 0.9,
            }
        ],
    )

    lineup = pipeline._build_operator_or_static_kbo_result("LG 라인업 알려줘.")
    roster = pipeline._build_operator_or_static_kbo_result("LG 부상자 명단은 어디서 봐?")

    assert lineup is not None
    assert lineup["strategy"] == "manual_baseball_data_required"
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in lineup["answer"]
    assert roster is not None
    assert roster["strategy"] == "manual_baseball_data_required"
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in roster["answer"]


def test_operator_data_fast_path_malformed_lineup_notes_keep_manual_contract() -> None:
    pipeline = _operator_pipeline(
        True,
        [
            {
                "queue_id": "ODQ-0030",
                "game_id": "20260605LGKT0",
                "team_code": "LG",
                "player_name": "홍길동",
                "position": "P",
                "batting_order": 1,
                "notes": "manual_lineup malformed metadata",
                "game_date": "2026-06-05",
                "home_team": "LG",
                "away_team": "KT",
            }
        ],
    )

    result = pipeline._build_operator_or_static_kbo_result("2026-06-05 LG 라인업 알려줘.")

    assert result is not None
    assert result["strategy"] == "manual_baseball_data_required"
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in result["answer"]
