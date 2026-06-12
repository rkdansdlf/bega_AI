#!/usr/bin/env python3
"""Validate operator-data handoff CSVs without mutating baseball data.

This script only reads operator-provided CSVs and, when configured, performs
read-only DB checks. It does not crawl, infer, repair, or write baseball data.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.tools.team_code_resolver import CANONICAL_CODES, TeamCodeResolver
from scripts import analyze_operator_data_required as taxonomy_analyzer
from scripts import build_operator_data_handoff as handoff_builder

DEFAULT_QUEUE_INPUT = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_handoff_queue_post_db_fast_path_docker_kbo500.csv"
)
DEFAULT_FIELDS_INPUT = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_handoff_fields_post_db_fast_path_docker_kbo500.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_validation"
    / "post_db_fast_path_docker_kbo500"
)

ALLOWED_STATUSES = {
    "pending",
    "ready_for_validation",
    "validated",
    "applied",
    "rejected",
}
VALUE_REQUIRED_STATUSES = {"ready_for_validation", "validated"}
TRUTHY_VALUES = {"1", "true", "t", "yes", "y", "verified"}
FALSEY_VALUES = {"0", "false", "f", "no", "n", "unverified"}
CONFIDENCE_MINIMUM = 0.70
COMMON_SOURCE_FIELDS = {"source_name", "source_checked_at", "is_verified", "confidence"}
P0_DOMAINS = {"season_meta", "schedule_window", "game_day_lineup", "roster_news"}

ALLOWED_GAME_STATUSES = {
    "scheduled",
    "completed",
    "cancelled",
    "canceled",
    "postponed",
    "in_progress",
    "예정",
    "진행",
    "종료",
    "완료",
    "취소",
    "연기",
    "경기전",
}
ALLOWED_ROSTER_EVENT_TYPES = {
    "injury",
    "return",
    "callup",
    "senddown",
    "contract",
    "trade",
    "coach_change",
    "manager_change",
    "registration",
    "entry",
    "other",
    "부상",
    "복귀",
    "콜업",
    "말소",
    "계약",
    "트레이드",
    "감독",
    "코치",
    "등록",
    "엔트리",
    "기타",
}
ALLOWED_TOPIC_TYPES = {
    "seat",
    "ticket",
    "parking",
    "food",
    "facility",
    "transport",
    "other",
    "좌석",
    "티켓",
    "주차",
    "먹거리",
    "편의시설",
    "교통",
    "기타",
}
ALLOWED_MEDIA_TYPES = {
    "tv",
    "radio",
    "text",
    "replay",
    "highlight",
    "streaming",
    "other",
    "TV",
    "라디오",
    "문자중계",
    "다시보기",
    "하이라이트",
    "스트리밍",
    "기타",
}
ALLOWED_FAN_EVENT_TYPES = {
    "cheer",
    "gift",
    "goods",
    "fan_service",
    "away_section",
    "other",
    "응원",
    "선물",
    "굿즈",
    "팬서비스",
    "원정응원석",
    "기타",
}

APPLY_TARGETS = {
    "season_meta": "operator_season_events",
    "schedule_window": "operator_schedule_items",
    "game_day_lineup": "game_lineups/manual_starters",
    "roster_news": "operator_roster_events",
    "venue_ticket": "operator_venue_guides",
    "broadcast_media": "operator_broadcast_items",
    "fan_event": "operator_fan_events",
    "unsupported_external": "operator_data_items",
    "db_fast_path_candidate": "",
    "subjective_prediction": "operator_data_items_policy_gate",
}

ISSUE_FIELDNAMES = [
    "queue_id",
    "domain",
    "field_name",
    "severity",
    "code",
    "message",
]
APPLY_PLAN_FIELDNAMES = [
    "queue_id",
    "priority",
    "domain",
    "operator_status",
    "validation_status",
    "apply_eligible",
    "apply_target",
    "issue_count",
    "skip_reason",
]
QUEUE_INPUT_FIELDNAMES = handoff_builder.QUEUE_FIELDNAMES
FIELDS_INPUT_FIELDNAMES = handoff_builder.FIELDS_FIELDNAMES


@dataclass(frozen=True)
class ValidationIssue:
    queue_id: str
    domain: str
    field_name: str
    severity: str
    code: str
    message: str

    def to_record(self) -> Dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "domain": self.domain,
            "field_name": self.field_name,
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }


class DbChecker(Protocol):
    db_checks_skipped: bool
    db_skip_reason: str

    def get_game(self, game_id: str) -> Optional[Mapping[str, Any]]: ...

    def count_players(self, player_name: str) -> int: ...

    def is_known_team_code(
        self, team_code: str, season_year: Optional[int] = None
    ) -> bool: ...

    def close(self) -> None: ...


class NoopDbChecker:
    def __init__(self, reason: str) -> None:
        self.db_checks_skipped = True
        self.db_skip_reason = reason

    def get_game(self, game_id: str) -> Optional[Mapping[str, Any]]:
        del game_id
        return None

    def count_players(self, player_name: str) -> int:
        del player_name
        return 0

    def is_known_team_code(
        self, team_code: str, season_year: Optional[int] = None
    ) -> bool:
        del team_code, season_year
        return True

    def close(self) -> None:
        return None


class PsycopgDbChecker:
    def __init__(self, db_url: str) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"psycopg is required for DB checks: {exc}") from exc

        self._psycopg = psycopg
        self._dict_row = dict_row
        self._conn = psycopg.connect(db_url, row_factory=dict_row)
        self._resolver = TeamCodeResolver()
        self.db_checks_skipped = False
        self.db_skip_reason = ""

    def get_game(self, game_id: str) -> Optional[Mapping[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT game_id, home_team, away_team, game_date, game_status
                FROM game
                WHERE game_id = %s
                """,
                (game_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def count_players(self, player_name: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS count FROM player_basic WHERE name = %s",
                (player_name,),
            )
            row = cur.fetchone() or {}
        return int(row.get("count") or 0)

    def is_known_team_code(
        self, team_code: str, season_year: Optional[int] = None
    ) -> bool:
        canonical = self._resolver.resolve_canonical(team_code, season_year)
        return str(canonical or "").upper() in CANONICAL_CODES

    def close(self) -> None:
        self._conn.close()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _issue(
    *,
    queue_id: str,
    domain: str = "",
    field_name: str = "",
    severity: str,
    code: str,
    message: str,
) -> ValidationIssue:
    return ValidationIssue(
        queue_id=queue_id,
        domain=domain,
        field_name=field_name,
        severity=severity,
        code=code,
        message=message,
    )


def _read_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def _split_fields(raw: Any) -> List[str]:
    if isinstance(raw, str):
        return [
            _normalize_text(field) for field in raw.split("|") if _normalize_text(field)
        ]
    if isinstance(raw, Sequence):
        return [_normalize_text(field) for field in raw if _normalize_text(field)]
    return []


def _parse_bool(value: Any) -> Optional[bool]:
    normalized = _normalize_text(value).lower()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSEY_VALUES:
        return False
    return None


def _parse_float(value: Any) -> Optional[float]:
    try:
        return float(_normalize_text(value))
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> Optional[int]:
    try:
        return int(_normalize_text(value))
    except (TypeError, ValueError):
        return None


def _parse_iso_date(value: Any) -> Optional[date]:
    raw = _normalize_text(value)
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    raw = _normalize_text(value)
    if not raw:
        return None
    try:
        if len(raw) == 10:
            return datetime.fromisoformat(f"{raw}T00:00:00")
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _valid_time(value: Any) -> bool:
    raw = _normalize_text(value)
    if not raw:
        return False
    if _parse_iso_datetime(raw) is not None:
        return True
    try:
        datetime.strptime(raw, "%H:%M")
        return True
    except ValueError:
        return False


def _is_false(value: Any) -> bool:
    return _parse_bool(value) is False


def _source_metadata(payload: Mapping[str, str]) -> Dict[str, Any]:
    confidence = _parse_float(payload.get("confidence"))
    return {
        "source_name": _normalize_text(payload.get("source_name")),
        "source_checked_at": _normalize_text(payload.get("source_checked_at")),
        "is_verified": _parse_bool(payload.get("is_verified")),
        "confidence": confidence,
    }


def _validation_status(issues: Sequence[ValidationIssue]) -> str:
    if any(issue.severity == "error" for issue in issues):
        return "fail"
    if any(issue.severity == "warning" for issue in issues):
        return "warning"
    return "pass"


def _contract_required_fields(domain: str) -> List[str]:
    contract = taxonomy_analyzer.CONTRACTS.get(domain)
    return list(contract.required_fields) if contract else []


def _base_apply_target(domain: str) -> str:
    return APPLY_TARGETS.get(domain, "operator_data_items")


def _domain_apply_eligible(domain: str, payload: Mapping[str, str]) -> tuple[bool, str]:
    if domain == "subjective_prediction":
        return False, "policy_gate_subjective_prediction"
    if domain == "db_fast_path_candidate":
        return False, "db_fast_path_candidate_not_operator_fact"
    if domain == "unsupported_external" and _is_false(
        payload.get("supported_by_operator")
    ):
        return False, "unsupported_external_not_supported"
    if domain not in P0_DOMAINS:
        return False, "operator_data_v1_non_p0_domain"
    return True, ""


def _build_payload(field_rows: Sequence[Mapping[str, str]]) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for row in field_rows:
        field_name = _normalize_text(row.get("field_name"))
        if not field_name:
            continue
        payload[field_name] = _normalize_text(row.get("operator_value"))
    return payload


def _validate_headers(
    *,
    queue_fieldnames: Optional[Sequence[str]],
    field_fieldnames: Optional[Sequence[str]],
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if (
        queue_fieldnames is not None
        and list(queue_fieldnames) != QUEUE_INPUT_FIELDNAMES
    ):
        issues.append(
            _issue(
                queue_id="",
                field_name="queue_header",
                severity="error",
                code="invalid_queue_header",
                message=(
                    "queue CSV header must exactly match "
                    f"{QUEUE_INPUT_FIELDNAMES}; got {list(queue_fieldnames)}."
                ),
            )
        )
    if (
        field_fieldnames is not None
        and list(field_fieldnames) != FIELDS_INPUT_FIELDNAMES
    ):
        issues.append(
            _issue(
                queue_id="",
                field_name="fields_header",
                severity="error",
                code="invalid_fields_header",
                message=(
                    "fields CSV header must exactly match "
                    f"{FIELDS_INPUT_FIELDNAMES}; got {list(field_fieldnames)}."
                ),
            )
        )
    return issues


def _validate_file_shape(
    queue_rows: Sequence[Mapping[str, str]],
    fields_by_queue: Mapping[str, List[Mapping[str, str]]],
    *,
    queue_fieldnames: Optional[Sequence[str]] = None,
    field_fieldnames: Optional[Sequence[str]] = None,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = _validate_headers(
        queue_fieldnames=queue_fieldnames,
        field_fieldnames=field_fieldnames,
    )
    seen_queue_ids: set[str] = set()
    queue_by_id: Dict[str, Mapping[str, str]] = {}

    for row in queue_rows:
        queue_id = _normalize_text(row.get("queue_id"))
        domain = _normalize_text(row.get("domain"))
        if not queue_id:
            issues.append(
                _issue(
                    queue_id="",
                    domain=domain,
                    severity="error",
                    code="missing_queue_id",
                    message="queue_id is required.",
                )
            )
            continue
        if queue_id in seen_queue_ids:
            issues.append(
                _issue(
                    queue_id=queue_id,
                    domain=domain,
                    severity="error",
                    code="duplicate_queue_id",
                    message="queue_id must be unique.",
                )
            )
        else:
            queue_by_id[queue_id] = row
        seen_queue_ids.add(queue_id)

        if domain not in taxonomy_analyzer.CONTRACTS:
            issues.append(
                _issue(
                    queue_id=queue_id,
                    domain=domain,
                    severity="error",
                    code="unknown_domain",
                    message=f"unsupported operator data domain: {domain}",
                )
            )
            continue

        status = _normalize_text(row.get("operator_status")) or "pending"
        if status not in ALLOWED_STATUSES:
            issues.append(
                _issue(
                    queue_id=queue_id,
                    domain=domain,
                    field_name="operator_status",
                    severity="error",
                    code="invalid_operator_status",
                    message=f"operator_status must be one of {sorted(ALLOWED_STATUSES)}.",
                )
            )

        queue_required_fields = _split_fields(row.get("required_fields"))
        contract_required_fields = _contract_required_fields(domain)
        required_fields = queue_required_fields or contract_required_fields
        if not required_fields:
            issues.append(
                _issue(
                    queue_id=queue_id,
                    domain=domain,
                    field_name="required_fields",
                    severity="error",
                    code="missing_required_fields",
                    message="required_fields must not be empty.",
                )
            )

    valid_queue_ids = {
        _normalize_text(row.get("queue_id"))
        for row in queue_rows
        if _normalize_text(row.get("queue_id"))
    }
    seen_field_keys: set[tuple[str, str]] = set()
    for queue_id, field_rows in fields_by_queue.items():
        if queue_id not in valid_queue_ids:
            for field_row in field_rows:
                issues.append(
                    _issue(
                        queue_id=queue_id,
                        domain=_normalize_text(field_row.get("domain")),
                        field_name=_normalize_text(field_row.get("field_name")),
                        severity="error",
                        code="orphan_field_row",
                        message="fields CSV contains a queue_id not present in queue CSV.",
                    )
                )
            continue

        queue_row = queue_by_id.get(queue_id)
        if queue_row is None:
            continue
        domain = _normalize_text(queue_row.get("domain"))
        contract_code = _normalize_text(queue_row.get("contract_code"))
        question = _normalize_text(queue_row.get("question"))

        field_names: List[str] = []
        for field_row in field_rows:
            field_name = _normalize_text(field_row.get("field_name"))
            if not field_name:
                issues.append(
                    _issue(
                        queue_id=queue_id,
                        domain=_normalize_text(field_row.get("domain")),
                        severity="error",
                        code="missing_field_name",
                        message="field_name is required in fields CSV.",
                    )
                )
                continue
            field_names.append(field_name)
            field_key = (queue_id, field_name)
            if field_key in seen_field_keys:
                issues.append(
                    _issue(
                        queue_id=queue_id,
                        domain=domain,
                        field_name=field_name,
                        severity="error",
                        code="duplicate_field_row",
                        message="fields CSV must not repeat the same queue_id/field_name pair.",
                    )
                )
            seen_field_keys.add(field_key)

            field_domain = _normalize_text(field_row.get("domain"))
            if field_domain != domain:
                issues.append(
                    _issue(
                        queue_id=queue_id,
                        domain=field_domain,
                        field_name=field_name,
                        severity="error",
                        code="field_domain_mismatch",
                        message=f"field row domain={field_domain} does not match queue domain={domain}.",
                    )
                )
            field_contract_code = _normalize_text(field_row.get("contract_code"))
            if field_contract_code != contract_code:
                issues.append(
                    _issue(
                        queue_id=queue_id,
                        domain=domain,
                        field_name=field_name,
                        severity="error",
                        code="field_contract_code_mismatch",
                        message=(
                            f"field row contract_code={field_contract_code} "
                            f"does not match queue contract_code={contract_code}."
                        ),
                    )
                )
            field_question = _normalize_text(field_row.get("question"))
            if field_question != question:
                issues.append(
                    _issue(
                        queue_id=queue_id,
                        domain=domain,
                        field_name=field_name,
                        severity="error",
                        code="field_question_mismatch",
                        message="field row question does not match queue question.",
                    )
                )

        required_fields = _split_fields(
            queue_row.get("required_fields")
        ) or _contract_required_fields(domain)
        expected_field_names = set(required_fields)
        actual_field_names = set(field_names)
        for field_name in sorted(expected_field_names - actual_field_names):
            issues.append(
                _issue(
                    queue_id=queue_id,
                    domain=domain,
                    field_name=field_name,
                    severity="error",
                    code="missing_field_row",
                    message="required field row is missing from fields CSV.",
                )
            )
        for field_name in sorted(actual_field_names - expected_field_names):
            issues.append(
                _issue(
                    queue_id=queue_id,
                    domain=domain,
                    field_name=field_name,
                    severity="error",
                    code="unexpected_field_row",
                    message="fields CSV contains a field_name not listed in queue required_fields.",
                )
            )

    return issues


def _validate_common_values(
    *,
    queue_id: str,
    domain: str,
    required_fields: Sequence[str],
    payload: Mapping[str, str],
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for field_name in required_fields:
        if (
            domain == "unsupported_external"
            and field_name == "manual_answer_text"
            and _is_false(payload.get("supported_by_operator"))
        ):
            continue
        if _normalize_text(payload.get(field_name)) == "":
            issues.append(
                _issue(
                    queue_id=queue_id,
                    domain=domain,
                    field_name=field_name,
                    severity="error",
                    code="missing_operator_value",
                    message="operator_value is required for this field.",
                )
            )

    metadata = _source_metadata(payload)
    if not metadata["source_name"]:
        issues.append(
            _issue(
                queue_id=queue_id,
                domain=domain,
                field_name="source_name",
                severity="error",
                code="missing_source_name",
                message="source_name is required for validation-ready rows.",
            )
        )
    if not metadata["source_checked_at"]:
        issues.append(
            _issue(
                queue_id=queue_id,
                domain=domain,
                field_name="source_checked_at",
                severity="error",
                code="missing_source_checked_at",
                message="source_checked_at is required for validation-ready rows.",
            )
        )
    elif _parse_iso_datetime(metadata["source_checked_at"]) is None:
        issues.append(
            _issue(
                queue_id=queue_id,
                domain=domain,
                field_name="source_checked_at",
                severity="error",
                code="invalid_source_checked_at",
                message="source_checked_at must be an ISO date or datetime.",
            )
        )
    if metadata["is_verified"] is not True:
        issues.append(
            _issue(
                queue_id=queue_id,
                domain=domain,
                field_name="is_verified",
                severity="error",
                code="not_verified",
                message="is_verified must be true.",
            )
        )
    if metadata["confidence"] is None:
        issues.append(
            _issue(
                queue_id=queue_id,
                domain=domain,
                field_name="confidence",
                severity="error",
                code="invalid_confidence",
                message="confidence must be numeric.",
            )
        )
    elif metadata["confidence"] < CONFIDENCE_MINIMUM:
        issues.append(
            _issue(
                queue_id=queue_id,
                domain=domain,
                field_name="confidence",
                severity="error",
                code="low_confidence",
                message=f"confidence must be at least {CONFIDENCE_MINIMUM}.",
            )
        )
    return issues


def _validate_domain_values(
    *,
    queue_id: str,
    domain: str,
    payload: Mapping[str, str],
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    def invalid(field_name: str, code: str, message: str) -> None:
        issues.append(
            _issue(
                queue_id=queue_id,
                domain=domain,
                field_name=field_name,
                severity="error",
                code=code,
                message=message,
            )
        )

    if domain == "season_meta":
        if _parse_int(payload.get("season_year")) is None:
            invalid("season_year", "invalid_integer", "season_year must be an integer.")
        if _parse_iso_date(payload.get("event_date")) is None:
            invalid("event_date", "invalid_date", "event_date must be an ISO date.")

    elif domain == "schedule_window":
        if _parse_iso_date(payload.get("game_date")) is None:
            invalid("game_date", "invalid_date", "game_date must be an ISO date.")
        if not _valid_time(payload.get("start_time")):
            invalid(
                "start_time",
                "invalid_time",
                "start_time must be HH:MM or ISO datetime.",
            )
        status = _normalize_text(payload.get("game_status")).lower()
        if status and status not in {item.lower() for item in ALLOWED_GAME_STATUSES}:
            invalid("game_status", "invalid_enum", "game_status is not supported.")

    elif domain == "game_day_lineup":
        batting_order = _parse_int(payload.get("batting_order"))
        if batting_order is None or batting_order <= 0:
            invalid(
                "batting_order",
                "invalid_integer",
                "batting_order must be a positive integer.",
            )
        if _parse_iso_datetime(payload.get("announced_at")) is None:
            invalid(
                "announced_at",
                "invalid_datetime",
                "announced_at must be an ISO date or datetime.",
            )

    elif domain == "roster_news":
        if _parse_int(payload.get("season_year")) is None:
            invalid("season_year", "invalid_integer", "season_year must be an integer.")
        if _parse_iso_date(payload.get("effective_date")) is None:
            invalid(
                "effective_date", "invalid_date", "effective_date must be an ISO date."
            )
        event_type = _normalize_text(payload.get("roster_event_type")).lower()
        if event_type and event_type not in {
            item.lower() for item in ALLOWED_ROSTER_EVENT_TYPES
        }:
            invalid(
                "roster_event_type",
                "invalid_enum",
                "roster_event_type is not supported.",
            )

    elif domain == "venue_ticket":
        valid_from = _parse_iso_datetime(payload.get("valid_from"))
        valid_to = _parse_iso_datetime(payload.get("valid_to"))
        if valid_from is None:
            invalid(
                "valid_from",
                "invalid_datetime",
                "valid_from must be an ISO date or datetime.",
            )
        if valid_to is None:
            invalid(
                "valid_to",
                "invalid_datetime",
                "valid_to must be an ISO date or datetime.",
            )
        if valid_from is not None and valid_to is not None and valid_from > valid_to:
            invalid(
                "valid_to",
                "invalid_validity_window",
                "valid_to must be greater than or equal to valid_from.",
            )
        topic = _normalize_text(payload.get("topic_type")).lower()
        if topic and topic not in {item.lower() for item in ALLOWED_TOPIC_TYPES}:
            invalid("topic_type", "invalid_enum", "topic_type is not supported.")

    elif domain == "broadcast_media":
        if _parse_iso_date(payload.get("game_date")) is None:
            invalid("game_date", "invalid_date", "game_date must be an ISO date.")
        media_type = _normalize_text(payload.get("media_type")).lower()
        if media_type and media_type not in {
            item.lower() for item in ALLOWED_MEDIA_TYPES
        }:
            invalid("media_type", "invalid_enum", "media_type is not supported.")

    elif domain == "fan_event":
        if _parse_iso_date(payload.get("game_date")) is None:
            invalid("game_date", "invalid_date", "game_date must be an ISO date.")
        event_type = _normalize_text(payload.get("event_type")).lower()
        if event_type and event_type not in {
            item.lower() for item in ALLOWED_FAN_EVENT_TYPES
        }:
            invalid("event_type", "invalid_enum", "event_type is not supported.")

    elif domain == "unsupported_external":
        supported = _parse_bool(payload.get("supported_by_operator"))
        if supported is None:
            invalid(
                "supported_by_operator",
                "invalid_boolean",
                "supported_by_operator must be boolean.",
            )

    elif domain == "subjective_prediction":
        if _parse_iso_date(payload.get("as_of_date")) is None:
            invalid("as_of_date", "invalid_date", "as_of_date must be an ISO date.")

    return issues


def _validate_db_values(
    *,
    queue_id: str,
    domain: str,
    payload: Mapping[str, str],
    db_checker: DbChecker,
) -> List[ValidationIssue]:
    if db_checker.db_checks_skipped:
        return []
    issues: List[ValidationIssue] = []
    game_id = _normalize_text(payload.get("game_id"))
    team_code = _normalize_text(payload.get("team_code"))
    player_name = _normalize_text(payload.get("player_name"))
    season_year = _parse_int(payload.get("season_year"))

    game_row: Optional[Mapping[str, Any]] = None
    if game_id:
        game_row = db_checker.get_game(game_id)
        if game_row is None:
            issues.append(
                _issue(
                    queue_id=queue_id,
                    domain=domain,
                    field_name="game_id",
                    severity="error",
                    code="game_not_found",
                    message="game_id does not exist in game table.",
                )
            )

    if domain == "schedule_window" and game_row is not None:
        for payload_field, db_field in (
            ("home_team", "home_team"),
            ("away_team", "away_team"),
        ):
            payload_team = _normalize_text(payload.get(payload_field))
            db_team = _normalize_text(game_row.get(db_field))
            if payload_team and db_team and payload_team != db_team:
                issues.append(
                    _issue(
                        queue_id=queue_id,
                        domain=domain,
                        field_name=payload_field,
                        severity="error",
                        code="game_team_conflict",
                        message=f"{payload_field}={payload_team} conflicts with game.{db_field}={db_team}.",
                    )
                )

    if team_code and not db_checker.is_known_team_code(team_code, season_year):
        issues.append(
            _issue(
                queue_id=queue_id,
                domain=domain,
                field_name="team_code",
                severity="error",
                code="unknown_team_code",
                message="team_code cannot be resolved to a supported KBO team.",
            )
        )

    if player_name:
        player_count = db_checker.count_players(player_name)
        if player_count != 1:
            issues.append(
                _issue(
                    queue_id=queue_id,
                    domain=domain,
                    field_name="player_name",
                    severity="error",
                    code="player_resolution_not_unique",
                    message=f"player_name resolved to {player_count} player_basic rows.",
                )
            )

    return issues


def _detect_lineup_conflicts(
    normalized_rows: Sequence[Mapping[str, Any]],
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    seen: Dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in normalized_rows:
        if row.get("domain") != "game_day_lineup":
            continue
        payload = row.get("payload") or {}
        if not isinstance(payload, Mapping):
            continue
        key = (
            _normalize_text(payload.get("game_id")),
            _normalize_text(payload.get("team_code")),
            _normalize_text(payload.get("batting_order")),
        )
        if not all(key):
            continue
        player_name = _normalize_text(payload.get("player_name"))
        previous = seen.get(key)
        if previous is not None:
            previous_payload = previous.get("payload") or {}
            previous_player = _normalize_text(
                previous_payload.get("player_name")
                if isinstance(previous_payload, Mapping)
                else ""
            )
            if player_name != previous_player:
                issues.append(
                    _issue(
                        queue_id=str(row.get("queue_id") or ""),
                        domain="game_day_lineup",
                        field_name="batting_order",
                        severity="error",
                        code="lineup_slot_conflict",
                        message=(
                            "Duplicate game_id/team_code/batting_order has conflicting player_name "
                            f"with queue_id={previous.get('queue_id')}."
                        ),
                    )
                )
        else:
            seen[key] = row
    return issues


def build_validation_report(
    queue_rows: Sequence[Mapping[str, str]],
    field_rows: Sequence[Mapping[str, str]],
    *,
    db_checker: DbChecker,
    queue_fieldnames: Optional[Sequence[str]] = None,
    field_fieldnames: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    fields_by_queue: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for field_row in field_rows:
        fields_by_queue[_normalize_text(field_row.get("queue_id"))].append(field_row)

    issues: List[ValidationIssue] = _validate_file_shape(
        queue_rows,
        fields_by_queue,
        queue_fieldnames=queue_fieldnames,
        field_fieldnames=field_fieldnames,
    )
    normalized_rows: List[Dict[str, Any]] = []
    issues_by_queue: Dict[str, List[ValidationIssue]] = defaultdict(list)
    for issue in issues:
        issues_by_queue[issue.queue_id].append(issue)

    if db_checker.db_checks_skipped:
        warning = _issue(
            queue_id="",
            severity="warning",
            code="db_checks_skipped",
            message=db_checker.db_skip_reason or "DB checks were skipped.",
        )
        issues.append(warning)
        issues_by_queue[""].append(warning)

    for queue_row in queue_rows:
        queue_id = _normalize_text(queue_row.get("queue_id"))
        domain = _normalize_text(queue_row.get("domain"))
        status = _normalize_text(queue_row.get("operator_status")) or "pending"
        priority = _normalize_text(queue_row.get("priority"))
        required_fields = _split_fields(
            queue_row.get("required_fields")
        ) or _contract_required_fields(domain)
        payload = _build_payload(fields_by_queue.get(queue_id, []))

        row_issues = list(issues_by_queue.get(queue_id, []))
        if status in VALUE_REQUIRED_STATUSES and domain in taxonomy_analyzer.CONTRACTS:
            row_issues.extend(
                _validate_common_values(
                    queue_id=queue_id,
                    domain=domain,
                    required_fields=required_fields,
                    payload=payload,
                )
            )
            row_issues.extend(
                _validate_domain_values(
                    queue_id=queue_id,
                    domain=domain,
                    payload=payload,
                )
            )
            row_issues.extend(
                _validate_db_values(
                    queue_id=queue_id,
                    domain=domain,
                    payload=payload,
                    db_checker=db_checker,
                )
            )

        issues.extend(issue for issue in row_issues if issue not in issues)
        row_status = _validation_status(row_issues)
        domain_eligible, domain_skip_reason = _domain_apply_eligible(domain, payload)
        apply_target = _base_apply_target(domain)
        apply_eligible = (
            status in VALUE_REQUIRED_STATUSES
            and row_status == "pass"
            and domain_eligible
            and not db_checker.db_checks_skipped
        )
        skip_reason = ""
        if not apply_eligible:
            if status not in VALUE_REQUIRED_STATUSES:
                skip_reason = f"operator_status_{status or 'missing'}"
            elif row_status != "pass":
                skip_reason = "validation_not_passed"
            elif not domain_eligible:
                skip_reason = domain_skip_reason
            elif db_checker.db_checks_skipped:
                skip_reason = "db_checks_skipped"

        normalized_rows.append(
            {
                "queue_id": queue_id,
                "priority": priority,
                "domain": domain,
                "contract_code": _normalize_text(queue_row.get("contract_code")),
                "question": _normalize_text(queue_row.get("question")),
                "operator_status": status,
                "required_fields": required_fields,
                "payload": payload,
                "source_metadata": _source_metadata(payload),
                "validation_status": row_status,
                "apply_eligible": apply_eligible,
                "apply_target": apply_target if apply_eligible else "",
                "skip_reason": skip_reason,
                "issue_count": len(row_issues),
            }
        )

    conflict_issues = _detect_lineup_conflicts(normalized_rows)
    if conflict_issues:
        issues.extend(conflict_issues)
        conflict_count_by_queue = Counter(issue.queue_id for issue in conflict_issues)
        for row in normalized_rows:
            extra_count = conflict_count_by_queue.get(str(row.get("queue_id") or ""), 0)
            if extra_count:
                row["issue_count"] = int(row.get("issue_count") or 0) + extra_count
                row["validation_status"] = "fail"
                row["apply_eligible"] = False
                row["apply_target"] = ""
                row["skip_reason"] = "validation_not_passed"

    severity_counts = Counter(issue.severity for issue in issues)
    priority_counts = Counter(str(row.get("priority") or "") for row in normalized_rows)
    domain_counts = Counter(str(row.get("domain") or "") for row in normalized_rows)
    status_counts = Counter(
        str(row.get("operator_status") or "") for row in normalized_rows
    )
    validation_counts = Counter(
        str(row.get("validation_status") or "") for row in normalized_rows
    )
    summary_status = _validation_status(issues)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": summary_status,
        "total_queue_items": len(queue_rows),
        "total_field_rows": len(field_rows),
        "normalized_row_count": len(normalized_rows),
        "apply_plan_row_count": len(normalized_rows),
        "issue_counts": {
            "error": severity_counts.get("error", 0),
            "warning": severity_counts.get("warning", 0),
        },
        "priority_counts": dict(sorted(priority_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "operator_status_counts": dict(sorted(status_counts.items())),
        "validation_status_counts": dict(sorted(validation_counts.items())),
        "apply_eligible_count": sum(
            1 for row in normalized_rows if row.get("apply_eligible")
        ),
        "db_checks": {
            "skipped": bool(db_checker.db_checks_skipped),
            "skip_reason": db_checker.db_skip_reason,
        },
    }
    return {
        "summary": summary,
        "issues": [issue.to_record() for issue in issues],
        "normalized_rows": normalized_rows,
        "apply_plan_rows": [
            {
                "queue_id": row["queue_id"],
                "priority": row["priority"],
                "domain": row["domain"],
                "operator_status": row["operator_status"],
                "validation_status": row["validation_status"],
                "apply_eligible": str(bool(row["apply_eligible"])).lower(),
                "apply_target": row["apply_target"],
                "issue_count": row["issue_count"],
                "skip_reason": row["skip_reason"],
            }
            for row in normalized_rows
        ],
    }


def validate_files(
    *,
    queue_path: Path,
    fields_path: Path,
    output_dir: Path,
    db_checker: DbChecker,
) -> Dict[str, Any]:
    queue_fieldnames, queue_rows = _read_csv(queue_path)
    field_fieldnames, field_rows = _read_csv(fields_path)
    report = build_validation_report(
        queue_rows,
        field_rows,
        db_checker=db_checker,
        queue_fieldnames=queue_fieldnames,
        field_fieldnames=field_fieldnames,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "operator_data_validation_summary.json", report["summary"])
    _write_csv(
        output_dir / "operator_data_validation_issues.csv",
        report["issues"],
        ISSUE_FIELDNAMES,
    )
    _write_jsonl(
        output_dir / "operator_data_normalized_rows.jsonl",
        report["normalized_rows"],
    )
    _write_csv(
        output_dir / "operator_data_apply_plan.csv",
        report["apply_plan_rows"],
        APPLY_PLAN_FIELDNAMES,
    )
    return report


def _build_db_checker(*, db_url: str, skip_db_checks: bool) -> DbChecker:
    if skip_db_checks:
        return NoopDbChecker("--skip-db-checks was set")
    if not db_url:
        return NoopDbChecker("POSTGRES_DB_URL is not set")
    return PsycopgDbChecker(db_url)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate operator-data handoff queue/fields CSV files."
    )
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE_INPUT))
    parser.add_argument("--fields", default=str(DEFAULT_FIELDS_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--db-url",
        default="",
        help="PostgreSQL URL override. Defaults to POSTGRES_DB_URL.",
    )
    parser.add_argument(
        "--skip-db-checks",
        action="store_true",
        help="Skip read-only game/player/team DB checks and emit a warning.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        default=True,
        help="Exit 0 even when validation errors are present.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    db_url = _normalize_text(args.db_url) or os.environ.get("POSTGRES_DB_URL", "")
    db_checker = _build_db_checker(
        db_url=db_url,
        skip_db_checks=bool(args.skip_db_checks),
    )
    try:
        report = validate_files(
            queue_path=Path(args.queue),
            fields_path=Path(args.fields),
            output_dir=Path(args.output_dir),
            db_checker=db_checker,
        )
    finally:
        db_checker.close()
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.strict and report["summary"]["issue_counts"]["error"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
