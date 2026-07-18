from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts import analyze_operator_data_required as analyzer
from scripts import validate_operator_data_handoff as validator


class FakeDbChecker:
    def __init__(
        self,
        *,
        games: dict[str, Mapping[str, Any]] | None = None,
        player_counts: dict[str, int] | None = None,
        known_teams: set[str] | None = None,
    ) -> None:
        self.games = games or {}
        self.player_counts = player_counts or {}
        self.known_teams = known_teams or {
            "LG",
            "KT",
            "KIA",
            "SSG",
            "DB",
            "NC",
            "LT",
            "SS",
            "HH",
            "KH",
        }
        self.db_checks_skipped = False
        self.db_skip_reason = ""

    def get_game(self, game_id: str) -> Mapping[str, Any] | None:
        return self.games.get(game_id)

    def count_players(self, player_name: str) -> int:
        return self.player_counts.get(player_name, 1)

    def is_known_team_code(
        self, team_code: str, season_year: int | None = None
    ) -> bool:
        del season_year
        return team_code in self.known_teams

    def close(self) -> None:
        return None


def _queue(
    queue_id: str,
    domain: str,
    *,
    status: str = "ready_for_validation",
    required_fields: list[str] | None = None,
) -> dict[str, str]:
    contract = analyzer.CONTRACTS[domain]
    fields = required_fields or list(contract.required_fields)
    return {
        "queue_id": queue_id,
        "priority": "P0",
        "priority_reason": "test",
        "domain": domain,
        "contract_code": contract.contract_code,
        "question": f"{domain} question",
        "required_fields": "|".join(fields),
        "endpoint_count": "2",
        "endpoints": "/ai/chat/completion|/ai/chat/stream",
        "sample_answer": "MANUAL_BASEBALL_DATA_REQUIRED: ...",
        "operator_status": status,
        "operator_owner": "",
        "operator_notes": "",
    }


def _fields(
    queue_id: str,
    domain: str,
    values: Mapping[str, Any],
    *,
    required_fields: list[str] | None = None,
) -> list[dict[str, str]]:
    fields = required_fields or list(analyzer.CONTRACTS[domain].required_fields)
    return [
        {
            "queue_id": queue_id,
            "domain": domain,
            "contract_code": analyzer.CONTRACTS[domain].contract_code,
            "question": f"{domain} question",
            "field_name": field_name,
            "field_description": "",
            "required": "true",
            "operator_value": str(values.get(field_name, "")),
            "operator_notes": "",
        }
        for field_name in fields
    ]


def _report(
    queue_rows: list[dict[str, str]],
    field_rows: list[dict[str, str]],
    *,
    db_checker: Any | None = None,
) -> dict[str, Any]:
    return validator.build_validation_report(
        queue_rows,
        field_rows,
        db_checker=db_checker or FakeDbChecker(),
    )
