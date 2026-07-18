#!/usr/bin/env python3
"""Safely ingest operator-provided manual game lineup rows.

This script is intentionally not a crawler and does not infer missing data. It
only accepts verified operator input and resolves player IDs from the internal
player_basic table before writing to game_lineups.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    import psycopg
    from psycopg.rows import dict_row
except ModuleNotFoundError as exc:
    psycopg = None
    dict_row = None
    _PSYCOPG_IMPORT_ERROR = exc

REQUIRED_FIELDS = {
    "game_id",
    "team_code",
    "player_name",
    "batting_order",
    "position",
    "source_url",
    "source_name",
    "source_checked_at",
    "manual_override_reason",
    "is_verified",
    "confidence",
}
CONFIDENCE_MINIMUM = 0.70


class ManualBaseballDataRequired(RuntimeError):
    """Raised when operator-provided baseball data is incomplete or ambiguous."""


@dataclass(frozen=True)
class ManualLineupRow:
    game_id: str
    team_code: str
    player_name: str
    batting_order: int
    position: str
    source_url: str
    source_name: str
    source_checked_at: str
    manual_override_reason: str
    is_verified: bool
    confidence: float


@dataclass(frozen=True)
class ResolvedLineupRow:
    row: ManualLineupRow
    player_id: str

    @property
    def quality_score(self) -> float:
        return min(self.row.confidence, 0.70)


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required to run ingest_lineup_manual.py. "
            f"Detail: {_PSYCOPG_IMPORT_ERROR}"
        )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "t", "yes", "y", "verified"}


def _load_payload(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        common = {
            key: payload.get(key)
            for key in (
                "source_url",
                "source_name",
                "source_checked_at",
                "manual_override_reason",
                "is_verified",
                "confidence",
            )
            if key in payload
        }
        rows = payload.get("rows") or payload.get("lineups")
        if isinstance(rows, list):
            return [{**common, **dict(row)} for row in rows]
    raise ManualBaseballDataRequired(
        "MANUAL_BASEBALL_DATA_REQUIRED: input must be a CSV, JSON list, or JSON object with rows/lineups"
    )


def _parse_row(raw: Dict[str, Any], index: int) -> ManualLineupRow:
    missing = sorted(field for field in REQUIRED_FIELDS if raw.get(field) in (None, ""))
    if missing:
        raise ManualBaseballDataRequired(
            f"MANUAL_BASEBALL_DATA_REQUIRED: row {index} missing {','.join(missing)}"
        )
    confidence = float(raw["confidence"])
    if confidence < CONFIDENCE_MINIMUM:
        raise ManualBaseballDataRequired(
            f"MANUAL_BASEBALL_DATA_REQUIRED: row {index} confidence below {CONFIDENCE_MINIMUM}"
        )
    if not _truthy(raw["is_verified"]):
        raise ManualBaseballDataRequired(
            f"MANUAL_BASEBALL_DATA_REQUIRED: row {index} is_verified must be true"
        )
    return ManualLineupRow(
        game_id=str(raw["game_id"]).strip(),
        team_code=str(raw["team_code"]).strip(),
        player_name=str(raw["player_name"]).strip(),
        batting_order=int(raw["batting_order"]),
        position=str(raw["position"]).strip(),
        source_url=str(raw["source_url"]).strip(),
        source_name=str(raw["source_name"]).strip(),
        source_checked_at=str(raw["source_checked_at"]).strip(),
        manual_override_reason=str(raw["manual_override_reason"]).strip(),
        is_verified=True,
        confidence=confidence,
    )


def load_manual_rows(path: Path) -> List[ManualLineupRow]:
    return [
        _parse_row(row, idx) for idx, row in enumerate(_load_payload(path), start=1)
    ]


def _fetch_rows(cur: Any) -> List[Any]:
    rows = cur.fetchall()
    return list(rows or [])


def lookup_player_id(cur: Any, player_name: str) -> str:
    cur.execute(
        """
        SELECT player_id
        FROM player_basic
        WHERE name = %s
        ORDER BY player_id
        """,
        (player_name,),
    )
    rows = _fetch_rows(cur)
    if len(rows) != 1:
        raise ManualBaseballDataRequired(
            "MANUAL_BASEBALL_DATA_REQUIRED: "
            f"player_name={player_name!r} resolved to {len(rows)} player_basic rows"
        )
    row = rows[0]
    player_id = row.get("player_id") if isinstance(row, dict) else row[0]
    if player_id in (None, ""):
        raise ManualBaseballDataRequired(
            f"MANUAL_BASEBALL_DATA_REQUIRED: player_name={player_name!r} has no player_id"
        )
    return str(player_id)


def resolve_lineups(
    cur: Any, rows: Sequence[ManualLineupRow]
) -> List[ResolvedLineupRow]:
    return [
        ResolvedLineupRow(row=row, player_id=lookup_player_id(cur, row.player_name))
        for row in rows
    ]


def _table_columns(cur: Any) -> set[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'game_lineups'
        """,
        (),
    )
    columns = set()
    for row in _fetch_rows(cur):
        columns.add(str(row.get("column_name") if isinstance(row, dict) else row[0]))
    return columns


def _notes_payload(row: ResolvedLineupRow) -> str:
    payload = {
        "source_type": "manual_lineup",
        "source_url": row.row.source_url,
        "source_name": row.row.source_name,
        "source_checked_at": row.row.source_checked_at,
        "manual_override_reason": row.row.manual_override_reason,
        "is_verified": row.row.is_verified,
        "confidence": row.row.confidence,
        "quality_score": row.quality_score,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _insert_payload(row: ResolvedLineupRow, columns: set[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "game_id": row.row.game_id,
        "team_code": row.row.team_code,
        "player_id": row.player_id,
        "batting_order": row.row.batting_order,
        "position": row.row.position,
    }
    if "player_name" in columns:
        payload["player_name"] = row.row.player_name
    if "is_starter" in columns:
        payload["is_starter"] = True
    if "notes" in columns:
        payload["notes"] = _notes_payload(row)
    return payload


def apply_lineups(cur: Any, rows: Sequence[ResolvedLineupRow]) -> int:
    columns = _table_columns(cur)
    required = {"game_id", "team_code", "player_id", "batting_order", "position"}
    missing = required - columns
    if missing:
        raise RuntimeError(f"game_lineups missing required columns: {sorted(missing)}")
    updated = 0
    for row in rows:
        payload = _insert_payload(row, columns)
        column_names = list(payload.keys())
        placeholders = ", ".join(["%s"] * len(column_names))
        assignments = ", ".join(
            f"{column} = EXCLUDED.{column}"
            for column in column_names
            if column not in {"game_id", "team_code", "batting_order"}
        )
        sql = (
            f"INSERT INTO game_lineups ({', '.join(column_names)}) "
            f"VALUES ({placeholders}) "
            "ON CONFLICT (game_id, team_code, batting_order) DO UPDATE SET "
            f"{assignments}"
        )
        cur.execute(sql, tuple(payload[column] for column in column_names))
        updated += 1
    return updated


def ingest_lineup(*, input_path: Path, apply: bool, db_url: str) -> Dict[str, Any]:
    _require_psycopg()
    rows = load_manual_rows(input_path)
    with psycopg.connect(db_url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            resolved = resolve_lineups(cur, rows)
            if apply:
                updated = apply_lineups(cur, resolved)
                conn.commit()
            else:
                updated = 0
    return {
        "input_rows": len(rows),
        "resolved_rows": len(resolved),
        "updated_rows": updated,
        "dry_run": not apply,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely ingest verified operator-provided game_lineups rows."
    )
    parser.add_argument(
        "--input", required=True, help="Operator-provided CSV or JSON file."
    )
    parser.add_argument(
        "--apply", action="store_true", help="Persist rows. Default is dry-run."
    )
    parser.add_argument(
        "--db-url",
        default="",
        help="PostgreSQL URL override. Defaults to POSTGRES_DB_URL.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    db_url = args.db_url.strip() or os.environ.get("POSTGRES_DB_URL", "")
    if not db_url:
        raise RuntimeError("POSTGRES_DB_URL is required")
    result = ingest_lineup(
        input_path=Path(args.input),
        apply=bool(args.apply),
        db_url=db_url,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
