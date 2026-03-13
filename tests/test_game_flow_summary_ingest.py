from __future__ import annotations

from scripts import ingest_from_kbo as ingest_script


def test_game_flow_summary_single_chunk_skips_smart_chunks(monkeypatch) -> None:
    def _raise(_text: str):
        raise AssertionError("smart_chunks should not run for single_chunk profiles")

    monkeypatch.setattr(ingest_script, "smart_chunks", _raise)

    row = {
        "game_id": "20250501LGHH0",
        "game_date": "2025-05-01",
        "season_year": 2025,
        "season_id": 2025,
        "league_type_code": 0,
        "home_team": "LG",
        "away_team": "HH",
        "home_team_name": "LG 트윈스",
        "away_team_name": "한화 이글스",
        "home_score": 4,
        "away_score": 1,
        "winning_team": "LG",
        "inning_lines_json": [
            {"inning": 1, "home_runs": 0, "away_runs": 1, "is_extra": False},
            {"inning": 5, "home_runs": 4, "away_runs": 0, "is_extra": False},
        ],
    }

    payloads = ingest_script._build_chunk_payload_dicts_for_row(
        table_name="game_flow_summary",
        row=row,
        source_row_id="game_id=20250501LGHH0",
        use_legacy_renderer=False,
        today_str="2026-03-07",
    )

    assert len(payloads) == 1
    assert payloads[0]["table"] == "game_flow_summary"
    assert payloads[0]["source_row_id"] == "game_id=20250501LGHH0"
    assert payloads[0]["content"].startswith("[TL;DR]")
