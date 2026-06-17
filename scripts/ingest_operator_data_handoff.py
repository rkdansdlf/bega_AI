#!/usr/bin/env python3
"""Dry-run/apply verified operator-data handoff rows for P0 domains.

The script only consumes ``operator_data_normalized_rows.jsonl`` produced by
``validate_operator_data_handoff.py``. It does not crawl, infer, repair, or
collect baseball data.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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

P0_DOMAINS = ("season_meta", "schedule_window", "game_day_lineup", "roster_news")
P0_DOMAIN_SET = set(P0_DOMAINS)
DEFAULT_NORMALIZED_INPUT = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_validation"
    / "post_db_fast_path_docker_kbo500"
    / "operator_data_normalized_rows.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_ingest"
    / "post_db_fast_path_docker_kbo500"
)
CONFIDENCE_MINIMUM = 0.70
STARTER_POSITIONS = {"P", "SP", "투수", "선발", "선발투수"}

PLAN_FIELDNAMES = [
    "queue_id",
    "domain",
    "operator_status",
    "validation_status",
    "apply_eligible",
    "action",
    "apply_target",
    "payload_hash",
    "issue_count",
    "skip_reason",
]
ISSUE_FIELDNAMES = ["queue_id", "domain", "severity", "code", "message"]
STARTER_FIELDNAMES = ["game_id", "home_pitcher", "away_pitcher", "source_queue_ids"]

REQUIRED_TABLE_COLUMNS = {
    "operator_data_items": {
        "queue_id",
        "priority",
        "domain",
        "contract_code",
        "question",
        "operator_status",
        "validation_status",
        "apply_target",
        "payload",
        "payload_hash",
        "source_name",
        "source_checked_at",
        "is_verified",
        "confidence",
        "applied_at",
        "updated_at",
    },
    "operator_season_events": {
        "queue_id",
        "season_year",
        "event_name",
        "event_date",
        "stadium_name",
        "payload_hash",
        "source_name",
        "source_checked_at",
        "is_verified",
        "confidence",
        "applied_at",
        "updated_at",
    },
    "operator_schedule_items": {
        "queue_id",
        "game_date",
        "game_id",
        "home_team",
        "away_team",
        "stadium_name",
        "start_time",
        "game_status",
        "payload_hash",
        "source_name",
        "source_checked_at",
        "is_verified",
        "confidence",
        "applied_at",
        "updated_at",
    },
    "operator_roster_events": {
        "queue_id",
        "season_year",
        "team_code",
        "player_name",
        "roster_event_type",
        "effective_date",
        "status_text",
        "payload_hash",
        "source_name",
        "source_checked_at",
        "is_verified",
        "confidence",
        "applied_at",
        "updated_at",
    },
    "game_lineups": {
        "game_id",
        "team_code",
        "player_id",
        "player_name",
        "batting_order",
        "position",
        "notes",
    },
}

DOMAIN_TARGET_TABLES = {
    "season_meta": "operator_season_events",
    "schedule_window": "operator_schedule_items",
    "game_day_lineup": "game_lineups",
    "roster_news": "operator_roster_events",
}


@dataclass(frozen=True)
class IngestIssue:
    queue_id: str
    domain: str
    severity: str
    code: str
    message: str

    def to_record(self) -> Dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "domain": self.domain,
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }


def load_normalized_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"normalized row {line_number} must be an object")
            rows.append(dict(payload))
    return rows


def build_ingest_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    conn: Any = None,
    apply: bool = False,
    allow_overwrite: bool = False,
    domains: Sequence[str] = P0_DOMAINS,
) -> Dict[str, Any]:
    selected_domains = [
        str(domain).strip() for domain in domains if str(domain).strip()
    ]
    selected_domain_set = set(selected_domains)
    plans: List[Dict[str, Any]] = []
    issues: List[IngestIssue] = []
    eligible_rows: List[Mapping[str, Any]] = []
    lineup_rows_for_starters: List[Mapping[str, Any]] = []
    starter_rows: List[Dict[str, Any]] = []

    for row in rows:
        domain = _normalize_text(row.get("domain"))
        queue_id = _normalize_text(row.get("queue_id"))
        payload_hash = _payload_hash(row)
        plan = _base_plan(row, payload_hash=payload_hash)
        if domain not in P0_DOMAIN_SET:
            plan["action"] = "skipped"
            plan["skip_reason"] = "operator_data_v1_non_p0_domain"
        elif domain not in selected_domain_set:
            plan["action"] = "skipped"
            plan["skip_reason"] = "domain_not_selected"
        elif not _row_is_apply_candidate(row):
            plan["action"] = "skipped"
            plan["skip_reason"] = _candidate_skip_reason(row)
        else:
            plan["action"] = "pending_db_check"
            eligible_rows.append(row)
        plans.append(plan)

    if eligible_rows and conn is None:
        for row in eligible_rows:
            queue_id = _normalize_text(row.get("queue_id"))
            domain = _normalize_text(row.get("domain"))
            issues.append(
                IngestIssue(
                    queue_id=queue_id,
                    domain=domain,
                    severity="error",
                    code="db_required",
                    message="A DB connection is required for eligible operator-data rows.",
                )
            )
            _update_plan(plans, queue_id, action="skipped", skip_reason="db_required")
    elif eligible_rows:
        with conn.cursor(row_factory=dict_row) as cur:
            table_columns_cache: Dict[str, set[str]] = {}
            game_lineup_columns: Optional[set[str]] = None
            for row in eligible_rows:
                row_issues: List[IngestIssue] = []
                queue_id = _normalize_text(row.get("queue_id"))
                domain = _normalize_text(row.get("domain"))
                payload_hash = _payload_hash(row)
                row_issues.extend(
                    _schema_issues_for_row(
                        cur,
                        row,
                        table_columns_cache=table_columns_cache,
                    )
                )
                if row_issues:
                    issues.extend(row_issues)
                    _update_plan(
                        plans,
                        queue_id,
                        action="skipped",
                        issue_count=len(row_issues),
                        skip_reason=_skip_reason_for_issues(row_issues),
                    )
                    continue

                existing_hash = _fetch_existing_payload_hash(cur, queue_id)
                if existing_hash == payload_hash:
                    _update_plan(
                        plans,
                        queue_id,
                        action="noop",
                        skip_reason="same_payload_hash",
                    )
                    continue
                if existing_hash and not allow_overwrite:
                    row_issues.append(
                        IngestIssue(
                            queue_id=queue_id,
                            domain=domain,
                            severity="error",
                            code="overwrite_requires_flag",
                            message="queue_id already exists with a different payload_hash.",
                        )
                    )

                resolved_lineup_row: Optional[Mapping[str, Any]] = None
                if domain == "game_day_lineup":
                    conflict_issue = _lineup_conflict_target_issue(cur, row)
                    if conflict_issue is not None:
                        row_issues.append(conflict_issue)
                    player_id = _lookup_player_id(cur, row)
                    if not player_id:
                        row_issues.append(
                            IngestIssue(
                                queue_id=queue_id,
                                domain=domain,
                                severity="error",
                                code="player_resolution_not_unique",
                                message="player_name must resolve to exactly one player_basic row.",
                            )
                        )
                    else:
                        resolved_lineup_row = {
                            **dict(row),
                            "_resolved_player_id": player_id,
                        }
                        if game_lineup_columns is None:
                            game_lineup_columns = _columns_for_table(
                                cur,
                                "game_lineups",
                                table_columns_cache=table_columns_cache,
                            )
                if row_issues:
                    issues.extend(row_issues)
                    _update_plan(
                        plans,
                        queue_id,
                        action="skipped",
                        issue_count=len(row_issues),
                        skip_reason=_skip_reason_for_issues(row_issues),
                    )
                    continue
                if resolved_lineup_row is not None:
                    row = resolved_lineup_row
                    lineup_rows_for_starters.append(row)

                action = "update" if existing_hash else "insert"
                if apply:
                    _apply_staging_row(cur, row, payload_hash)
                    _apply_domain_row(cur, row, payload_hash, game_lineup_columns)
                _update_plan(plans, queue_id, action=action)

            starter_rows = _build_starter_plan_rows(cur, lineup_rows_for_starters)

    if apply and eligible_rows and conn is not None and not _has_error(issues):
        conn.commit()
    elif apply and eligible_rows and conn is not None:
        conn.rollback()

    issue_counts = Counter(issue.severity for issue in issues)
    action_counts = Counter(str(plan.get("action") or "") for plan in plans)
    domain_counts = Counter(str(row.get("domain") or "") for row in rows)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": not apply,
        "selected_domains": selected_domains,
        "total_rows": len(rows),
        "eligible_rows": len(eligible_rows),
        "applied_rows": sum(
            1 for plan in plans if apply and plan.get("action") in {"insert", "update"}
        ),
        "issue_counts": {
            "error": issue_counts.get("error", 0),
            "warning": issue_counts.get("warning", 0),
        },
        "action_counts": dict(sorted(action_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "starter_plan_count": len(starter_rows),
    }
    return {
        "summary": summary,
        "plans": plans,
        "issues": [issue.to_record() for issue in issues],
        "starter_plan_rows": starter_rows,
    }


def _row_is_apply_candidate(row: Mapping[str, Any]) -> bool:
    source = row.get("source_metadata") or {}
    if not isinstance(source, Mapping):
        source = {}
    return (
        bool(row.get("apply_eligible"))
        and _normalize_text(row.get("validation_status")) == "pass"
        and _normalize_text(row.get("operator_status"))
        in {"ready_for_validation", "validated"}
        and source.get("is_verified") is True
        and _as_float(source.get("confidence")) is not None
        and float(source.get("confidence")) >= CONFIDENCE_MINIMUM
    )


def _candidate_skip_reason(row: Mapping[str, Any]) -> str:
    if not row.get("apply_eligible"):
        return _normalize_text(row.get("skip_reason")) or "not_apply_eligible"
    if _normalize_text(row.get("validation_status")) != "pass":
        return "validation_not_passed"
    source = row.get("source_metadata") or {}
    if not isinstance(source, Mapping) or source.get("is_verified") is not True:
        return "not_verified"
    confidence = _as_float(source.get("confidence"))
    if confidence is None or confidence < CONFIDENCE_MINIMUM:
        return "low_confidence"
    return "not_apply_eligible"


def _skip_reason_for_issues(row_issues: Sequence[IngestIssue]) -> str:
    for issue in row_issues:
        if issue.severity == "error" and issue.code:
            return issue.code
    for issue in row_issues:
        if issue.code:
            return issue.code
    return "validation_not_passed"


def _base_plan(row: Mapping[str, Any], *, payload_hash: str) -> Dict[str, Any]:
    return {
        "queue_id": _normalize_text(row.get("queue_id")),
        "domain": _normalize_text(row.get("domain")),
        "operator_status": _normalize_text(row.get("operator_status")),
        "validation_status": _normalize_text(row.get("validation_status")),
        "apply_eligible": str(bool(row.get("apply_eligible"))).lower(),
        "action": "",
        "apply_target": _normalize_text(row.get("apply_target")),
        "payload_hash": payload_hash,
        "issue_count": 0,
        "skip_reason": "",
    }


def _update_plan(
    plans: Sequence[Dict[str, Any]],
    queue_id: str,
    *,
    action: str,
    skip_reason: str = "",
    issue_count: Optional[int] = None,
) -> None:
    for plan in plans:
        if plan.get("queue_id") != queue_id:
            continue
        plan["action"] = action
        plan["skip_reason"] = skip_reason
        if issue_count is not None:
            plan["issue_count"] = issue_count
        return


def _fetch_existing_payload_hash(cur: Any, queue_id: str) -> str:
    cur.execute(
        "SELECT payload_hash FROM operator_data_items WHERE queue_id = %s",
        (queue_id,),
    )
    row = cur.fetchone()
    if not row:
        return ""
    return _normalize_text(
        row.get("payload_hash") if isinstance(row, Mapping) else row[0]
    )


def _lookup_player_id(cur: Any, row: Mapping[str, Any]) -> str:
    payload = _payload(row)
    player_name = _normalize_text(payload.get("player_name"))
    if not player_name:
        return ""
    cur.execute(
        """
        SELECT player_id
        FROM player_basic
        WHERE name = %s
        ORDER BY player_id
        """,
        (player_name,),
    )
    rows = list(cur.fetchall() or [])
    if len(rows) != 1:
        return ""
    player_id = rows[0].get("player_id") if isinstance(rows[0], Mapping) else rows[0][0]
    return _normalize_text(player_id)


def _table_columns(cur: Any, table_name: str) -> set[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
        """,
        (table_name,),
    )
    columns = set()
    for row in list(cur.fetchall() or []):
        columns.add(
            _normalize_text(
                row.get("column_name") if isinstance(row, Mapping) else row[0]
            )
        )
    return columns


def _columns_for_table(
    cur: Any,
    table_name: str,
    *,
    table_columns_cache: Dict[str, set[str]],
) -> set[str]:
    if table_name not in table_columns_cache:
        table_columns_cache[table_name] = _table_columns(cur, table_name)
    return table_columns_cache[table_name]


def _schema_issues_for_row(
    cur: Any,
    row: Mapping[str, Any],
    *,
    table_columns_cache: Dict[str, set[str]],
) -> List[IngestIssue]:
    queue_id = _normalize_text(row.get("queue_id"))
    domain = _normalize_text(row.get("domain"))
    tables = ["operator_data_items"]
    target_table = DOMAIN_TARGET_TABLES.get(domain)
    if target_table:
        tables.append(target_table)

    issues: List[IngestIssue] = []
    for table_name in tables:
        required_columns = REQUIRED_TABLE_COLUMNS.get(table_name, set())
        available_columns = _columns_for_table(
            cur,
            table_name,
            table_columns_cache=table_columns_cache,
        )
        missing_columns = sorted(required_columns - available_columns)
        if missing_columns:
            issues.append(
                IngestIssue(
                    queue_id=queue_id,
                    domain=domain,
                    severity="error",
                    code="schema_missing_columns",
                    message=(
                        f"{table_name} is missing required columns: "
                        f"{','.join(missing_columns)}"
                    ),
                )
            )
    return issues


def _lineup_conflict_target_issue(
    cur: Any,
    row: Mapping[str, Any],
) -> Optional[IngestIssue]:
    if _has_unique_conflict_target(
        cur,
        "game_lineups",
        ("game_id", "team_code", "batting_order"),
    ):
        return None
    return IngestIssue(
        queue_id=_normalize_text(row.get("queue_id")),
        domain=_normalize_text(row.get("domain")),
        severity="error",
        code="missing_lineup_conflict_target",
        message=(
            "game_lineups must have a unique or primary-key conflict target on "
            "(game_id, team_code, batting_order)."
        ),
    )


def _has_unique_conflict_target(
    cur: Any,
    table_name: str,
    columns: Sequence[str],
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


def _apply_staging_row(cur: Any, row: Mapping[str, Any], payload_hash: str) -> None:
    source = _source_metadata(row)
    values = (
        _normalize_text(row.get("queue_id")),
        _normalize_text(row.get("priority")),
        _normalize_text(row.get("domain")),
        _normalize_text(row.get("contract_code")),
        _normalize_text(row.get("question")),
        _normalize_text(row.get("operator_status")),
        _normalize_text(row.get("validation_status")),
        _normalize_text(row.get("apply_target")),
        json.dumps(_payload(row), ensure_ascii=False, sort_keys=True),
        payload_hash,
        source["source_name"],
        source["source_checked_at"],
        source["is_verified"],
        source["confidence"],
    )
    cur.execute(
        """
        INSERT INTO operator_data_items (
          queue_id, priority, domain, contract_code, question, operator_status,
          validation_status, apply_target, payload, payload_hash, source_name,
          source_checked_at, is_verified, confidence, applied_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, now())
        ON CONFLICT (queue_id) DO UPDATE SET
          priority = EXCLUDED.priority,
          domain = EXCLUDED.domain,
          contract_code = EXCLUDED.contract_code,
          question = EXCLUDED.question,
          operator_status = EXCLUDED.operator_status,
          validation_status = EXCLUDED.validation_status,
          apply_target = EXCLUDED.apply_target,
          payload = EXCLUDED.payload,
          payload_hash = EXCLUDED.payload_hash,
          source_name = EXCLUDED.source_name,
          source_checked_at = EXCLUDED.source_checked_at,
          is_verified = EXCLUDED.is_verified,
          confidence = EXCLUDED.confidence,
          applied_at = now(),
          updated_at = now()
        """,
        values,
    )


def _apply_domain_row(
    cur: Any,
    row: Mapping[str, Any],
    payload_hash: str,
    game_lineup_columns: Optional[set[str]],
) -> None:
    domain = _normalize_text(row.get("domain"))
    if domain == "season_meta":
        _apply_season_event(cur, row, payload_hash)
    elif domain == "schedule_window":
        _apply_schedule_item(cur, row, payload_hash)
    elif domain == "game_day_lineup":
        _apply_game_lineup(cur, row, game_lineup_columns or set())
    elif domain == "roster_news":
        _apply_roster_event(cur, row, payload_hash)


def _apply_season_event(cur: Any, row: Mapping[str, Any], payload_hash: str) -> None:
    payload = _payload(row)
    source = _source_metadata(row)
    cur.execute(
        """
        INSERT INTO operator_season_events (
          queue_id, season_year, event_name, event_date, stadium_name, payload_hash,
          source_name, source_checked_at, is_verified, confidence, applied_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (queue_id) DO UPDATE SET
          season_year = EXCLUDED.season_year,
          event_name = EXCLUDED.event_name,
          event_date = EXCLUDED.event_date,
          stadium_name = EXCLUDED.stadium_name,
          payload_hash = EXCLUDED.payload_hash,
          source_name = EXCLUDED.source_name,
          source_checked_at = EXCLUDED.source_checked_at,
          is_verified = EXCLUDED.is_verified,
          confidence = EXCLUDED.confidence,
          applied_at = now(),
          updated_at = now()
        """,
        (
            _normalize_text(row.get("queue_id")),
            _as_int(payload.get("season_year")),
            _normalize_text(payload.get("event_name")),
            _normalize_text(payload.get("event_date")),
            _normalize_text(payload.get("stadium_name")),
            payload_hash,
            source["source_name"],
            source["source_checked_at"],
            source["is_verified"],
            source["confidence"],
        ),
    )


def _apply_schedule_item(cur: Any, row: Mapping[str, Any], payload_hash: str) -> None:
    payload = _payload(row)
    source = _source_metadata(row)
    cur.execute(
        """
        INSERT INTO operator_schedule_items (
          queue_id, game_date, game_id, home_team, away_team, stadium_name,
          start_time, game_status, payload_hash, source_name, source_checked_at,
          is_verified, confidence, applied_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (queue_id) DO UPDATE SET
          game_date = EXCLUDED.game_date,
          game_id = EXCLUDED.game_id,
          home_team = EXCLUDED.home_team,
          away_team = EXCLUDED.away_team,
          stadium_name = EXCLUDED.stadium_name,
          start_time = EXCLUDED.start_time,
          game_status = EXCLUDED.game_status,
          payload_hash = EXCLUDED.payload_hash,
          source_name = EXCLUDED.source_name,
          source_checked_at = EXCLUDED.source_checked_at,
          is_verified = EXCLUDED.is_verified,
          confidence = EXCLUDED.confidence,
          applied_at = now(),
          updated_at = now()
        """,
        (
            _normalize_text(row.get("queue_id")),
            _normalize_text(payload.get("game_date")),
            _normalize_text(payload.get("game_id")),
            _normalize_text(payload.get("home_team")),
            _normalize_text(payload.get("away_team")),
            _normalize_text(payload.get("stadium_name")),
            _normalize_text(payload.get("start_time")),
            _normalize_text(payload.get("game_status")),
            payload_hash,
            source["source_name"],
            source["source_checked_at"],
            source["is_verified"],
            source["confidence"],
        ),
    )


def _apply_roster_event(cur: Any, row: Mapping[str, Any], payload_hash: str) -> None:
    payload = _payload(row)
    source = _source_metadata(row)
    cur.execute(
        """
        INSERT INTO operator_roster_events (
          queue_id, season_year, team_code, player_name, roster_event_type,
          effective_date, status_text, payload_hash, source_name,
          source_checked_at, is_verified, confidence, applied_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (queue_id) DO UPDATE SET
          season_year = EXCLUDED.season_year,
          team_code = EXCLUDED.team_code,
          player_name = EXCLUDED.player_name,
          roster_event_type = EXCLUDED.roster_event_type,
          effective_date = EXCLUDED.effective_date,
          status_text = EXCLUDED.status_text,
          payload_hash = EXCLUDED.payload_hash,
          source_name = EXCLUDED.source_name,
          source_checked_at = EXCLUDED.source_checked_at,
          is_verified = EXCLUDED.is_verified,
          confidence = EXCLUDED.confidence,
          applied_at = now(),
          updated_at = now()
        """,
        (
            _normalize_text(row.get("queue_id")),
            _as_int(payload.get("season_year")),
            _normalize_text(payload.get("team_code")),
            _normalize_text(payload.get("player_name")),
            _normalize_text(payload.get("roster_event_type")),
            _normalize_text(payload.get("effective_date")),
            _normalize_text(payload.get("status_text")),
            payload_hash,
            source["source_name"],
            source["source_checked_at"],
            source["is_verified"],
            source["confidence"],
        ),
    )


def _apply_game_lineup(
    cur: Any,
    row: Mapping[str, Any],
    columns: set[str],
) -> None:
    required = {
        "game_id",
        "team_code",
        "player_id",
        "player_name",
        "batting_order",
        "position",
        "notes",
    }
    missing = required - columns
    if missing:
        raise RuntimeError(f"game_lineups missing required columns: {sorted(missing)}")
    payload = _payload(row)
    source = _source_metadata(row)
    insert_payload: Dict[str, Any] = {
        "game_id": _normalize_text(payload.get("game_id")),
        "team_code": _normalize_text(payload.get("team_code")),
        "player_id": _normalize_text(row.get("_resolved_player_id")),
        "player_name": _normalize_text(payload.get("player_name")),
        "batting_order": _as_int(payload.get("batting_order")),
        "position": _normalize_text(payload.get("position")),
        "notes": json.dumps(
            {
                "source_type": "manual_lineup",
                "queue_id": _normalize_text(row.get("queue_id")),
                "source_name": source["source_name"],
                "source_checked_at": source["source_checked_at"],
                "is_verified": source["is_verified"],
                "confidence": source["confidence"],
                "quality_score": min(float(source["confidence"]), 0.70),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    if "is_starter" in columns:
        insert_payload["is_starter"] = True

    column_names = list(insert_payload.keys())
    placeholders = ", ".join(["%s"] * len(column_names))
    assignments = ", ".join(
        f"{column} = EXCLUDED.{column}"
        for column in column_names
        if column not in {"game_id", "team_code", "batting_order"}
    )
    cur.execute(
        (
            f"INSERT INTO game_lineups ({', '.join(column_names)}) "
            f"VALUES ({placeholders}) "
            "ON CONFLICT (game_id, team_code, batting_order) DO UPDATE SET "
            f"{assignments}"
        ),
        tuple(insert_payload[column] for column in column_names),
    )


def _build_starter_plan_rows(
    cur: Any,
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_game: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        payload = _payload(row)
        position = _normalize_text(payload.get("position")).upper().replace(" ", "")
        if position not in STARTER_POSITIONS:
            continue
        game_id = _normalize_text(payload.get("game_id"))
        team_code = _normalize_text(payload.get("team_code"))
        if game_id and team_code:
            by_game[game_id][team_code] = row

    starter_rows: List[Dict[str, Any]] = []
    for game_id, team_rows in by_game.items():
        game = _fetch_game(cur, game_id)
        home_team = _normalize_text(game.get("home_team"))
        away_team = _normalize_text(game.get("away_team"))
        home_row = team_rows.get(home_team)
        away_row = team_rows.get(away_team)
        if not home_row or not away_row:
            continue
        home_payload = _payload(home_row)
        away_payload = _payload(away_row)
        starter_rows.append(
            {
                "game_id": game_id,
                "home_pitcher": _normalize_text(home_payload.get("player_name")),
                "away_pitcher": _normalize_text(away_payload.get("player_name")),
                "source_queue_ids": "|".join(
                    [
                        _normalize_text(home_row.get("queue_id")),
                        _normalize_text(away_row.get("queue_id")),
                    ]
                ),
            }
        )
    return starter_rows


def _fetch_game(cur: Any, game_id: str) -> Mapping[str, Any]:
    cur.execute(
        "SELECT game_id, home_team, away_team FROM game WHERE game_id = %s",
        (game_id,),
    )
    row = cur.fetchone()
    return dict(row) if isinstance(row, Mapping) else {}


def _payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("payload") or {}
    return payload if isinstance(payload, Mapping) else {}


def _source_metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    source = row.get("source_metadata") or {}
    if not isinstance(source, Mapping):
        source = {}
    return {
        "source_name": _normalize_text(source.get("source_name")),
        "source_checked_at": _normalize_text(source.get("source_checked_at")),
        "is_verified": source.get("is_verified") is True,
        "confidence": float(_as_float(source.get("confidence")) or 0.0),
    }


def _payload_hash(row: Mapping[str, Any]) -> str:
    payload = {
        "domain": _normalize_text(row.get("domain")),
        "payload": _payload(row),
        "source_metadata": _source_metadata(row),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _has_error(issues: Sequence[IngestIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(_normalize_text(value))
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(_normalize_text(value))
    except (TypeError, ValueError):
        return None


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _connect(db_url: str) -> Any:
    if psycopg is None:
        raise RuntimeError(f"psycopg is required: {_PSYCOPG_IMPORT_ERROR}")
    return psycopg.connect(db_url, row_factory=dict_row)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest verified operator-data handoff normalized rows."
    )
    parser.add_argument("--normalized", default=str(DEFAULT_NORMALIZED_INPUT))
    parser.add_argument(
        "--apply", action="store_true", help="Persist rows. Default is dry-run."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit no-op alias for the default dry-run mode.",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--domains",
        default=",".join(P0_DOMAINS),
        help="Comma-separated domains to consider. Defaults to P0 domains.",
    )
    parser.add_argument(
        "--db-url",
        default="",
        help="PostgreSQL URL override. Defaults to POSTGRES_DB_URL.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        default=True,
        help="Exit 0 even when ingest validation errors are present.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = load_normalized_rows(Path(args.normalized))
    domains = [
        item.strip() for item in str(args.domains or "").split(",") if item.strip()
    ]
    eligible_exists = any(
        _normalize_text(row.get("domain")) in set(domains)
        and _row_is_apply_candidate(row)
        for row in rows
    )
    db_url = _normalize_text(args.db_url) or os.environ.get("POSTGRES_DB_URL", "")
    conn = None
    if eligible_exists:
        if not db_url:
            report = build_ingest_report(
                rows,
                conn=None,
                apply=bool(args.apply),
                allow_overwrite=bool(args.allow_overwrite),
                domains=domains,
            )
        else:
            conn = _connect(db_url)
            try:
                report = build_ingest_report(
                    rows,
                    conn=conn,
                    apply=bool(args.apply),
                    allow_overwrite=bool(args.allow_overwrite),
                    domains=domains,
                )
            finally:
                conn.close()
    else:
        report = build_ingest_report(
            rows,
            conn=None,
            apply=bool(args.apply),
            allow_overwrite=bool(args.allow_overwrite),
            domains=domains,
        )

    output_dir = Path(args.output_dir)
    _write_json(output_dir / "operator_data_ingest_summary.json", report["summary"])
    _write_csv(
        output_dir / "operator_data_ingest_plan.csv", report["plans"], PLAN_FIELDNAMES
    )
    _write_csv(
        output_dir / "operator_data_ingest_issues.csv",
        report["issues"],
        ISSUE_FIELDNAMES,
    )
    _write_csv(
        output_dir / "operator_data_starter_plan.csv",
        report["starter_plan_rows"],
        STARTER_FIELDNAMES,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.strict and report["summary"]["issue_counts"]["error"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
