#!/usr/bin/env python3
"""Diagnose coach grounding readiness and validate live coach responses."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _detect_workspace_root(start: Path) -> Path:
    current = start.resolve()
    while True:
        if (current / "docker-compose.yml").exists() or (
            current / ".env.prod"
        ).exists():
            return current
        if current.parent == current:
            return start.resolve()
        current = current.parent


def _read_env_file_entries(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    entries: Dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            trimmed = line.strip()
            if not trimmed or trimmed.startswith("#") or "=" not in trimmed:
                continue
            key, value = trimmed.split("=", 1)
            normalized = value.strip()
            if (normalized.startswith('"') and normalized.endswith('"')) or (
                normalized.startswith("'") and normalized.endswith("'")
            ):
                normalized = normalized[1:-1]
            entries[key.strip()] = normalized
    except OSError:
        return {}
    return entries


def _preload_workspace_env(project_root: Path) -> None:
    for filename in (".env.prod", ".env"):
        for key, value in _read_env_file_entries(project_root / filename).items():
            os.environ.setdefault(key, value)


WORKSPACE_ROOT = _detect_workspace_root(PROJECT_ROOT)
_preload_workspace_env(WORKSPACE_ROOT)

import httpx
from psycopg.rows import dict_row

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.deps import get_connection_pool
from app.core.coach_grounding import (
    CoachFactSheet,
    detect_unconfirmed_data_claims as detect_unconfirmed_data_claims_from_fact_sheet,
)
from app.routers.coach import (
    COACH_EVIDENCE_ROOT_CAUSE_ORDER,
    GameEvidence,
    _fetch_series_state,
    _normalize_game_status_bucket,
    _normalize_stage_label,
    _round_display_for_stage,
    assess_game_evidence,
)
from app.tools.team_code_resolver import TeamCodeResolver

DEFAULT_BACKEND_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "coach"
DEFAULT_VALIDATION_SAMPLE_SIZE = 4
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_COACH_TIMEOUT_SECONDS = 90.0
LOCAL_TOKEN_ENV_FILES = (".env.prod", ".env")
POSTSEASON_STAGE_LABELS = {"WC", "SEMI_PO", "PO", "KS"}
REQUEST_MODE_AUTO = "auto_brief"
REQUEST_MODE_MANUAL = "manual_detail"
BACKEND_POSTSEASON_CODE = {
    "WC": 2,
    "SEMI_PO": 3,
    "PO": 4,
    "KS": 5,
}


@dataclass(frozen=True)
class BackendMatchMeta:
    game_id: str
    season_id: Optional[int]
    league_type: Optional[str]
    post_season_series: Optional[str]
    series_game_no: Optional[int]
    home_pitcher: Optional[str]
    away_pitcher: Optional[str]
    game_status: Optional[str]
    detail_status_code: Optional[int] = None
    detail_error: Optional[str] = None


def _read_env_file_value(path: Path, key: str) -> str:
    return _read_env_file_entries(path).get(key, "")


def resolve_default_internal_api_key(project_root: Path = WORKSPACE_ROOT) -> str:
    env_value = (os.getenv("AI_INTERNAL_TOKEN", "") or "").strip()
    if env_value:
        return env_value
    for filename in LOCAL_TOKEN_ENV_FILES:
        token = _read_env_file_value(
            project_root / filename, "AI_INTERNAL_TOKEN"
        ).strip()
        if token:
            return token
    return ""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose coach grounding readiness and validate live coach responses."
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--season-year",
        type=int,
        default=datetime.now().year,
        help="Season year filter for diagnosis queries.",
    )
    common.add_argument(
        "--date-from",
        default=None,
        help="Inclusive start date (YYYY-MM-DD).",
    )
    common.add_argument(
        "--date-to",
        default=None,
        help="Inclusive end date (YYYY-MM-DD).",
    )
    common.add_argument(
        "--game-ids",
        default=None,
        help="Comma-separated game IDs.",
    )
    common.add_argument(
        "--max-games",
        type=int,
        default=100,
        help="Maximum games to load for diagnosis or validation input selection.",
    )
    common.add_argument(
        "--backend-base-url",
        default=DEFAULT_BACKEND_BASE_URL,
        help="Backend base URL used for validation calls.",
    )
    common.add_argument(
        "--output-dir",
        default="reports/coach",
        help="Output directory relative to bega_AI root unless absolute.",
    )
    common.add_argument(
        "--internal-api-key",
        default=resolve_default_internal_api_key(),
        help="Optional X-Internal-Api-Key header value.",
    )
    common.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout used for backend and AI proxy validation calls.",
    )
    common.add_argument(
        "--coach-timeout-seconds",
        type=float,
        default=DEFAULT_COACH_TIMEOUT_SECONDS,
        help="Read timeout used for coach SSE validation calls.",
    )
    common.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Exit with code 1 when hard failures are detected (default).",
    )
    common.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Always exit with code 0 unless the script crashes.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "diagnose",
        parents=[common],
        help="Diagnose data readiness from game-related tables only.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        parents=[common],
        help="Validate live coach responses against diagnosed game evidence.",
    )
    validate_parser.add_argument(
        "--diagnosis-json",
        default=None,
        help="Path to a previous diagnosis JSON report.",
    )

    all_parser = subparsers.add_parser(
        "all",
        parents=[common],
        help="Run diagnosis, select four validation samples, and validate them.",
    )
    all_parser.add_argument(
        "--diagnosis-json",
        default=None,
        help="Optional diagnosis JSON to reuse instead of querying the DB.",
    )

    return parser.parse_args(argv)


def parse_game_ids(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    seen: set[str] = set()
    values: List[str] = []
    for token in raw.split(","):
        normalized = token.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return values


def resolve_output_dir(raw: str) -> Path:
    base = Path(raw)
    if base.is_absolute():
        return base
    return PROJECT_ROOT / base


def _stringify_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "strftime"):
        if value.__class__.__name__ == "time":
            return value.strftime("%H:%M")
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    return text or None


def _stringify_time(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M")
    text = str(value).strip()
    return text or None


def _normalize_optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_optional_text(value: Any) -> Optional[str]:
    text = " ".join(str(value or "").split()).strip()
    return text or None


def _is_postseason_stage(stage_label: Optional[str]) -> bool:
    return str(stage_label or "").upper() in POSTSEASON_STAGE_LABELS


def _league_bucket(stage_label: Optional[str]) -> str:
    normalized = str(stage_label or "").upper()
    if normalized in POSTSEASON_STAGE_LABELS:
        return "postseason"
    if normalized == "PRE":
        return "preseason"
    if normalized == "REGULAR":
        return "regular"
    return "unknown"


def _expected_backend_league_type(stage_label: Optional[str]) -> Optional[str]:
    normalized = str(stage_label or "").upper()
    if normalized in POSTSEASON_STAGE_LABELS:
        return "POST"
    if normalized == "PRE":
        return "PRE"
    if normalized == "REGULAR":
        return "REGULAR"
    return None


def _resolve_backend_league_type_code(backend_meta: BackendMatchMeta) -> Optional[int]:
    if backend_meta.league_type == "PRE":
        return 1
    if backend_meta.league_type == "REGULAR":
        return 0
    if backend_meta.league_type == "POST":
        return BACKEND_POSTSEASON_CODE.get(
            str(backend_meta.post_season_series or "").strip().upper()
        )
    return None


def _matchup_key(record: Dict[str, Any]) -> tuple[Any, ...]:
    teams = sorted(
        [str(record.get("home_team_id") or ""), str(record.get("away_team_id") or "")]
    )
    return (
        record.get("season_year"),
        tuple(teams),
    )


def _sorted_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: (
            str(item.get("game_date") or ""),
            str(item.get("game_id") or ""),
        ),
    )


def _build_game_evidence_from_row(
    row: Dict[str, Any],
    resolver: TeamCodeResolver,
    conn: Any,
) -> GameEvidence:
    season_year = int(row["season_year"])
    home_team_code = resolver.resolve_canonical(row["home_team"], season_year)
    away_team_code = resolver.resolve_canonical(row["away_team"], season_year)
    stage_label = _normalize_stage_label(row.get("league_type_code"), None)
    evidence = GameEvidence(
        game_id=str(row["game_id"]),
        season_id=row.get("season_id"),
        season_year=season_year,
        game_date=_stringify_date(row.get("game_date")),
        game_status=str(row.get("game_status") or "UNKNOWN"),
        game_status_bucket=_normalize_game_status_bucket(row.get("game_status")),
        home_team_code=home_team_code,
        away_team_code=away_team_code,
        home_team_name=resolver.display_name(home_team_code),
        away_team_name=resolver.display_name(away_team_code),
        league_type_code=int(row.get("league_type_code") or 0),
        stage_label=stage_label,
        round_display=_round_display_for_stage(stage_label),
        home_pitcher=_normalize_optional_text(row.get("home_pitcher")),
        away_pitcher=_normalize_optional_text(row.get("away_pitcher")),
        lineup_announced=bool(row.get("lineup_count")),
        summary_items=["summary"] if int(row.get("summary_count") or 0) > 0 else [],
        stadium_name=_normalize_optional_text(row.get("stadium_name")),
        start_time=_stringify_time(row.get("start_time")),
        weather=_normalize_optional_text(row.get("weather")),
    )
    evidence.series_state = _fetch_series_state(conn, evidence)
    evidence.used_evidence = list(assess_game_evidence(evidence).used_evidence)
    return evidence


def diagnose_games(
    *,
    season_year: Optional[int],
    date_from: Optional[str],
    date_to: Optional[str],
    game_ids: Sequence[str],
    max_games: int,
) -> List[Dict[str, Any]]:
    where_parts: List[str] = []
    params: List[Any] = []
    if game_ids:
        where_parts.append("g.game_id = ANY(%s)")
        params.append(list(game_ids))
    else:
        if season_year is not None:
            where_parts.append(
                "COALESCE(ks.season_year, EXTRACT(YEAR FROM g.game_date)::int) = %s"
            )
            params.append(int(season_year))
        if date_from:
            where_parts.append("g.game_date >= %s")
            params.append(date_from)
        if date_to:
            where_parts.append("g.game_date <= %s")
            params.append(date_to)

    sql = """
        SELECT
            g.game_id,
            g.game_date,
            g.game_status,
            g.home_team,
            g.away_team,
            g.home_pitcher,
            g.away_pitcher,
            g.season_id,
            COALESCE(ks.season_year, EXTRACT(YEAR FROM g.game_date)::int) AS season_year,
            COALESCE(ks.league_type_code, 0) AS league_type_code,
            gm.stadium_name,
            gm.start_time,
            gm.weather,
            COALESCE(lineups.lineup_count, 0) AS lineup_count,
            COALESCE(summaries.summary_count, 0) AS summary_count
        FROM game g
        LEFT JOIN kbo_seasons ks ON g.season_id = ks.season_id
        LEFT JOIN game_metadata gm ON g.game_id = gm.game_id
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS lineup_count
            FROM game_lineups gl
            WHERE gl.game_id = g.game_id
              AND COALESCE(gl.is_starter, true) = true
        ) lineups ON true
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS summary_count
            FROM game_summary gs
            WHERE gs.game_id = g.game_id
        ) summaries ON true
    """
    if where_parts:
        sql += "\nWHERE " + "\n  AND ".join(where_parts)
    sql += "\nORDER BY g.game_date ASC NULLS LAST, g.game_id ASC"
    if max_games > 0:
        sql += "\nLIMIT %s"
        params.append(max_games)

    pool = get_connection_pool()
    resolver = TeamCodeResolver()
    diagnosis: List[Dict[str, Any]] = []
    with pool.connection() as conn:
        cursor = conn.cursor(row_factory=dict_row)
        rows = cursor.execute(sql, tuple(params)).fetchall()
        for row in rows or []:
            evidence = _build_game_evidence_from_row(row, resolver, conn)
            assessment = assess_game_evidence(evidence)
            series_state = evidence.series_state
            diagnosis.append(
                {
                    "game_id": evidence.game_id,
                    "season_id": evidence.season_id,
                    "season_year": evidence.season_year,
                    "game_date": evidence.game_date,
                    "game_status": evidence.game_status,
                    "game_status_bucket": evidence.game_status_bucket,
                    "home_team_id": evidence.home_team_code,
                    "away_team_id": evidence.away_team_code,
                    "home_team_name": evidence.home_team_name,
                    "away_team_name": evidence.away_team_name,
                    "league_type_code": evidence.league_type_code,
                    "stage_label": evidence.stage_label,
                    "home_pitcher_present": assessment.home_pitcher_present,
                    "away_pitcher_present": assessment.away_pitcher_present,
                    "lineup_announced": assessment.lineup_announced,
                    "summary_present": assessment.summary_present,
                    "metadata_present": assessment.metadata_present,
                    "series_context_available": assessment.series_context_available,
                    "series_game_no": series_state.game_no if series_state else None,
                    "series_score": (
                        {
                            "home_team_wins": series_state.home_team_wins,
                            "away_team_wins": series_state.away_team_wins,
                            "previous_games": series_state.previous_games,
                            "confirmed_previous_games": series_state.confirmed_previous_games,
                        }
                        if series_state
                        else None
                    ),
                    "series_state_partial": bool(
                        series_state.series_state_partial if series_state else False
                    ),
                    "series_state_hint_mismatch": bool(
                        series_state.series_state_hint_mismatch if series_state else False
                    ),
                    "expected_data_quality": assessment.expected_data_quality,
                    "root_causes": list(assessment.root_causes),
                    "used_evidence": list(assessment.used_evidence),
                }
            )
    return diagnosis


def build_diagnosis_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    quality_counts = Counter(
        str(item.get("expected_data_quality") or "unknown") for item in records
    )
    root_cause_counts = Counter(
        code for item in records for code in (item.get("root_causes") or [])
    )
    league_counts = Counter(_league_bucket(item.get("stage_label")) for item in records)
    stage_counts = Counter(
        str(item.get("stage_label") or "UNKNOWN") for item in records
    )
    series_state_partial_count = sum(
        1 for item in records if bool(item.get("series_state_partial"))
    )
    series_state_hint_mismatch_count = sum(
        1 for item in records if bool(item.get("series_state_hint_mismatch"))
    )
    return {
        "total_games": len(records),
        "quality_distribution": {
            "grounded": quality_counts.get("grounded", 0),
            "partial": quality_counts.get("partial", 0),
            "insufficient": quality_counts.get("insufficient", 0),
        },
        "root_cause_distribution": {
            code: root_cause_counts.get(code, 0)
            for code in COACH_EVIDENCE_ROOT_CAUSE_ORDER
        },
        "league_distribution": {
            "regular": league_counts.get("regular", 0),
            "postseason": league_counts.get("postseason", 0),
            "preseason": league_counts.get("preseason", 0),
            "unknown": league_counts.get("unknown", 0),
        },
        "stage_distribution": dict(sorted(stage_counts.items())),
        "series_state_partial_count": series_state_partial_count,
        "series_state_hint_mismatch_count": series_state_hint_mismatch_count,
    }


def select_validation_samples(
    records: Sequence[Dict[str, Any]],
    *,
    limit: int = DEFAULT_VALIDATION_SAMPLE_SIZE,
) -> List[Dict[str, Any]]:
    sorted_records = _sorted_records(records)
    if limit <= 0:
        return []

    matchup_groups: Dict[tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for record in sorted_records:
        matchup_groups[_matchup_key(record)].append(record)

    pair_candidates = [
        group[:2] for group in matchup_groups.values() if len(group) >= 2
    ]
    pair_candidates.sort(
        key=lambda group: (
            str(group[0].get("game_date") or ""),
            str(group[0].get("game_id") or ""),
            str(group[1].get("game_date") or ""),
            str(group[1].get("game_id") or ""),
        )
    )
    reserved_pair = pair_candidates[0] if pair_candidates else None

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_record(record: Dict[str, Any], reason: str) -> None:
        copied = deepcopy(record)
        copied["selection_reason"] = reason
        selected.append(copied)
        selected_ids.add(str(record.get("game_id")))

    def first_match(predicate: Any) -> Optional[Dict[str, Any]]:
        for candidate in sorted_records:
            game_id = str(candidate.get("game_id"))
            if game_id in selected_ids:
                continue
            if predicate(candidate):
                return candidate
        return None

    reserved_slots = 1 if reserved_pair and limit >= 2 else 0
    category_specs = [
        (
            "postseason_grounded",
            lambda item: _is_postseason_stage(item.get("stage_label"))
            and item.get("expected_data_quality") == "grounded",
        ),
        (
            "postseason_partial",
            lambda item: _is_postseason_stage(item.get("stage_label"))
            and item.get("expected_data_quality") == "partial",
        ),
        (
            "regular_grounded",
            lambda item: item.get("stage_label") == "REGULAR"
            and item.get("expected_data_quality") == "grounded",
        ),
    ]

    category_capacity = max(0, limit - reserved_slots)
    for reason, predicate in category_specs:
        if len(selected) >= category_capacity:
            break
        candidate = None
        if reserved_pair:
            anchor = reserved_pair[0]
            if str(anchor.get("game_id")) not in selected_ids and predicate(anchor):
                candidate = anchor
        if candidate is None:
            candidate = first_match(predicate)
        if candidate is not None:
            add_record(candidate, reason)

    if reserved_pair and limit >= 2:
        anchor, partner = reserved_pair
        anchor_game_id = str(anchor.get("game_id"))
        partner_game_id = str(partner.get("game_id"))
        if anchor_game_id not in selected_ids:
            if len(selected) >= max(1, limit - 1) and selected:
                removed = selected.pop()
                selected_ids.remove(str(removed.get("game_id")))
            add_record(anchor, "same_matchup_anchor")
        if len(selected) < limit and partner_game_id not in selected_ids:
            add_record(partner, "same_matchup_pair")

    filler_specs = [
        ("fill_partial", lambda item: item.get("expected_data_quality") == "partial"),
        ("fill_grounded", lambda item: item.get("expected_data_quality") == "grounded"),
        ("fill_any", lambda item: True),
    ]
    for reason, predicate in filler_specs:
        while len(selected) < limit:
            candidate = first_match(predicate)
            if candidate is None:
                break
            add_record(candidate, reason)
        if len(selected) >= limit:
            break

    return selected[:limit]


def load_diagnosis_report(path: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    diagnosis = payload.get("diagnosis")
    if not isinstance(diagnosis, list):
        raise ValueError("diagnosis_json_missing_diagnosis_array")
    return diagnosis


def _build_internal_headers(token: str) -> Dict[str, str]:
    return {"X-Internal-Api-Key": token} if token else {}


def ensure_backend_available(client: httpx.Client, base_url: str) -> None:
    response = client.get(f"{base_url.rstrip('/')}/api/matches/bounds")
    if response.status_code != 200:
        raise RuntimeError(
            f"backend_unavailable status={response.status_code} body={response.text[:200]}"
        )


def fetch_backend_match_meta(
    client: httpx.Client,
    base_url: str,
    record: Dict[str, Any],
) -> BackendMatchMeta:
    game_id = str(record["game_id"])
    game_date = str(record["game_date"])
    match_response = client.get(
        f"{base_url.rstrip('/')}/api/matches",
        params={"date": game_date},
    )
    if match_response.status_code != 200:
        raise RuntimeError(
            f"match_lookup_failed game_id={game_id} status={match_response.status_code}"
        )
    matches = match_response.json()
    if not isinstance(matches, list):
        raise RuntimeError(f"match_lookup_invalid_payload game_id={game_id}")
    match_payload = next(
        (item for item in matches if str(item.get("gameId")) == game_id),
        None,
    )
    if match_payload is None:
        raise RuntimeError(f"match_not_found_in_date_lookup game_id={game_id}")

    detail_status_code: Optional[int] = None
    detail_error: Optional[str] = None
    detail_payload: Dict[str, Any] = {}
    detail_response = client.get(f"{base_url.rstrip('/')}/api/matches/{game_id}")
    detail_status_code = detail_response.status_code
    if detail_response.status_code == 200:
        payload = detail_response.json()
        if isinstance(payload, dict):
            detail_payload = payload
        else:
            detail_error = "match_detail_invalid_payload"
    else:
        detail_error = f"match_detail_failed game_id={game_id} status={detail_response.status_code}"

    home_pitcher = None
    away_pitcher = None
    if isinstance(match_payload.get("homePitcher"), dict):
        home_pitcher = _normalize_optional_text(
            match_payload.get("homePitcher", {}).get("name")
        )
    if isinstance(match_payload.get("awayPitcher"), dict):
        away_pitcher = _normalize_optional_text(
            match_payload.get("awayPitcher", {}).get("name")
        )

    return BackendMatchMeta(
        game_id=game_id,
        season_id=_normalize_optional_int(match_payload.get("seasonId")),
        league_type=_normalize_optional_text(match_payload.get("leagueType")),
        post_season_series=_normalize_optional_text(
            match_payload.get("postSeasonSeries")
        ),
        series_game_no=_normalize_optional_int(match_payload.get("seriesGameNo")),
        home_pitcher=_normalize_optional_text(detail_payload.get("homePitcher"))
        or home_pitcher,
        away_pitcher=_normalize_optional_text(detail_payload.get("awayPitcher"))
        or away_pitcher,
        game_status=_normalize_optional_text(detail_payload.get("gameStatus"))
        or _normalize_optional_text(record.get("game_status")),
        detail_status_code=detail_status_code,
        detail_error=detail_error,
    )


def build_league_context(
    record: Dict[str, Any],
    backend_meta: BackendMatchMeta,
) -> Dict[str, Any]:
    backend_league_type_code = _resolve_backend_league_type_code(backend_meta)
    stage_label = (
        _normalize_optional_text(backend_meta.post_season_series)
        or ("PRE" if backend_meta.league_type == "PRE" else None)
        or _normalize_optional_text(record.get("stage_label"))
    )
    return {
        "season": backend_meta.season_id or record.get("season_id"),
        "season_year": record.get("season_year"),
        "league_type": backend_meta.league_type,
        "league_type_code": backend_league_type_code or record.get("league_type_code"),
        "stage_label": stage_label,
        "series_game_no": backend_meta.series_game_no or record.get("series_game_no"),
        "game_date": record.get("game_date"),
        "home_pitcher": backend_meta.home_pitcher,
        "away_pitcher": backend_meta.away_pitcher,
        "lineup_announced": bool(record.get("lineup_announced")),
    }


def build_request_payload(
    record: Dict[str, Any],
    backend_meta: BackendMatchMeta,
    request_mode: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "home_team_id": record.get("home_team_id"),
        "away_team_id": record.get("away_team_id"),
        "game_id": record.get("game_id"),
        "league_context": build_league_context(record, backend_meta),
        "request_mode": request_mode,
    }
    if request_mode == REQUEST_MODE_MANUAL:
        payload["focus"] = ["matchup", "recent_form"]
    return payload


def parse_sse_stream(response: httpx.Response) -> Dict[str, Any]:
    current_event = "message"
    sample_lines: List[str] = []
    answer_chunks: List[str] = []
    meta_payload: Dict[str, Any] = {}
    error_payload: Any = None
    event_sequence: List[str] = []
    done_seen = False

    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
        sample_lines.append(line)
        if len(sample_lines) > 40:
            sample_lines = sample_lines[-40:]

        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
            event_sequence.append(current_event)
            continue

        if not line.startswith("data:"):
            continue

        data_str = line.split(":", 1)[1].strip()
        if data_str == "[DONE]":
            done_seen = True
            continue

        parsed: Any
        try:
            parsed = json.loads(data_str)
        except json.JSONDecodeError:
            parsed = data_str

        if current_event == "message" and isinstance(parsed, dict):
            delta = parsed.get("delta")
            if isinstance(delta, str):
                answer_chunks.append(delta)
        elif current_event == "meta" and isinstance(parsed, dict):
            meta_payload = parsed
        elif current_event == "error":
            error_payload = parsed

        current_event = "message"

    structured_response = (
        meta_payload.get("structured_response")
        if isinstance(meta_payload.get("structured_response"), dict)
        else None
    )
    return {
        "status_code": response.status_code,
        "done_seen": done_seen,
        "event_sequence": event_sequence,
        "answer": "".join(answer_chunks),
        "meta": meta_payload,
        "structured_response": structured_response,
        "error_payload": error_payload,
        "sample_response": "\n".join(sample_lines),
    }


def call_coach_analyze(
    client: httpx.Client,
    base_url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
) -> Dict[str, Any]:
    request_headers = {"Accept": "text/event-stream"}
    request_headers.update(headers)
    endpoint = f"{base_url.rstrip('/')}/api/ai/coach/analyze"
    try:
        with client.stream(
            "POST",
            endpoint,
            json=payload,
            headers=request_headers,
        ) as response:
            if response.status_code != 200:
                error_body = response.read().decode("utf-8", errors="replace")
                return {
                    "status_code": response.status_code,
                    "done_seen": False,
                    "event_sequence": [],
                    "answer": "",
                    "meta": {},
                    "structured_response": None,
                    "error_payload": error_body,
                    "sample_response": error_body[:400],
                }
            return parse_sse_stream(response)
    except httpx.TimeoutException as exc:
        error_text = f"coach_request_timeout: {exc}"
        return {
            "status_code": 0,
            "done_seen": False,
            "event_sequence": [],
            "answer": "",
            "meta": {},
            "structured_response": None,
            "error_payload": error_text,
            "sample_response": error_text,
        }
    except httpx.HTTPError as exc:
        error_text = f"coach_request_error: {exc}"
        return {
            "status_code": 0,
            "done_seen": False,
            "event_sequence": [],
            "answer": "",
            "meta": {},
            "structured_response": None,
            "error_payload": error_text,
            "sample_response": error_text,
        }


def compare_backend_meta(
    record: Dict[str, Any],
    backend_meta: BackendMatchMeta,
) -> List[str]:
    failures: List[str] = []
    expected_league_type = _expected_backend_league_type(record.get("stage_label"))
    if expected_league_type != backend_meta.league_type:
        failures.append(
            "leagueType mismatch"
            f" expected={expected_league_type!r} actual={backend_meta.league_type!r}"
        )

    expected_series = (
        str(record.get("stage_label"))
        if _is_postseason_stage(record.get("stage_label"))
        else None
    )
    if expected_series != backend_meta.post_season_series:
        failures.append(
            "postSeasonSeries mismatch"
            f" expected={expected_series!r} actual={backend_meta.post_season_series!r}"
        )

    expected_series_game_no = _normalize_optional_int(record.get("series_game_no"))
    if expected_series_game_no != backend_meta.series_game_no:
        failures.append(
            "seriesGameNo mismatch"
            f" expected={expected_series_game_no!r} actual={backend_meta.series_game_no!r}"
        )

    return failures


def detect_unconfirmed_data_claims(
    record: Dict[str, Any],
    response_payload: Dict[str, Any],
) -> List[str]:
    fact_sheet = CoachFactSheet(
        fact_lines=[],
        caveat_lines=[],
        allowed_entity_names=set(),
        allowed_numeric_tokens=set(),
        supported_fact_count=0,
        starters_confirmed=bool(record.get("home_pitcher_present"))
        and bool(record.get("away_pitcher_present")),
        lineup_confirmed=bool(record.get("lineup_announced")),
        series_context_confirmed="missing_series_context"
        not in (record.get("root_causes") or []),
        require_series_context=str(record.get("stage_label") or "").upper()
        in POSTSEASON_STAGE_LABELS,
    )
    return detect_unconfirmed_data_claims_from_fact_sheet(
        fact_sheet,
        response_payload,
    )


def validate_capture(
    *,
    record: Dict[str, Any],
    request_mode: str,
    capture: Dict[str, Any],
) -> tuple[List[str], List[str]]:
    hard_failures: List[str] = []
    soft_warnings: List[str] = []
    if capture.get("status_code") != 200:
        error_payload = str(capture.get("error_payload") or "").strip()
        if error_payload:
            hard_failures.append(
                f"{request_mode} HTTP 실패 status={capture.get('status_code')} detail={error_payload}"
            )
        else:
            hard_failures.append(
                f"{request_mode} HTTP 실패 status={capture.get('status_code')}"
            )
        return hard_failures, soft_warnings

    if not capture.get("done_seen"):
        hard_failures.append(f"{request_mode} SSE done 누락")

    meta = capture.get("meta")
    if not isinstance(meta, dict) or not meta:
        hard_failures.append(f"{request_mode} meta 누락")
        return hard_failures, soft_warnings

    structured_response = capture.get("structured_response")
    if not isinstance(structured_response, dict):
        hard_failures.append(f"{request_mode} structured_response 누락")

    for field_name in (
        "generation_mode",
        "data_quality",
        "used_evidence",
        "grounding_warnings",
        "grounding_reasons",
        "supported_fact_count",
    ):
        if field_name not in meta:
            hard_failures.append(f"{request_mode} meta.{field_name} 누락")

    if meta.get("data_quality") == "insufficient" and not meta.get("used_evidence"):
        soft_warnings.append(
            f"{request_mode} insufficient 대비 used_evidence 비어 있음"
        )

    if (
        record.get("expected_data_quality") == "grounded"
        and meta.get("data_quality") == "partial"
    ):
        soft_warnings.append(f"{request_mode} grounded 진단 대비 partial 응답")

    fallback_mode = (
        request_mode == REQUEST_MODE_MANUAL
        and meta.get("generation_mode") == "evidence_fallback"
    )
    if fallback_mode:
        soft_warnings.append("manual_detail fallback 발생")

    grounding_reasons = {
        str(item)
        for item in (meta.get("grounding_reasons") or [])
        if isinstance(item, str)
    }
    if "unsupported_entity_name" in grounding_reasons:
        target = soft_warnings if fallback_mode else hard_failures
        target.append(f"{request_mode} unsupported entity name detected")
    if "unsupported_numeric_claim" in grounding_reasons:
        target = soft_warnings if fallback_mode else hard_failures
        target.append(f"{request_mode} unsupported numeric claim detected")

    if isinstance(structured_response, dict):
        soft_warnings.extend(
            detect_unconfirmed_data_claims(record, structured_response)
        )

    return hard_failures, soft_warnings


def detect_reused_matchup_briefs(
    results: Sequence[Dict[str, Any]],
) -> Dict[str, List[str]]:
    grouped: Dict[tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[_matchup_key(result["diagnosis"])].append(result)

    reused: Dict[str, List[str]] = defaultdict(list)
    for group in grouped.values():
        if len(group) < 2:
            continue
        signatures: Dict[str, List[str]] = defaultdict(list)
        for item in group:
            auto_result = item.get("responses", {}).get(REQUEST_MODE_AUTO, {})
            structured = auto_result.get("structured_response")
            if not isinstance(structured, dict):
                continue
            signature = json.dumps(structured, ensure_ascii=False, sort_keys=True)
            signatures[signature].append(str(item["diagnosis"].get("game_id")))
        for shared_ids in signatures.values():
            if len(shared_ids) < 2:
                continue
            for game_id in shared_ids:
                reused[game_id].append(
                    "동일 매치업 다른 game_id에서 auto_brief structured_response가 동일함"
                )
    return reused


def validate_records(
    *,
    records: Sequence[Dict[str, Any]],
    backend_base_url: str,
    internal_api_key: str,
    timeout_seconds: float,
    coach_timeout_seconds: float,
) -> Dict[str, Any]:
    headers = _build_internal_headers(internal_api_key)
    results: List[Dict[str, Any]] = []
    coach_timeout = httpx.Timeout(
        connect=min(timeout_seconds, 10.0),
        read=max(coach_timeout_seconds, timeout_seconds),
        write=max(timeout_seconds, 10.0),
        pool=max(timeout_seconds, 10.0),
    )
    with httpx.Client(
        timeout=timeout_seconds, follow_redirects=True
    ) as backend_client, httpx.Client(
        timeout=coach_timeout,
        follow_redirects=True,
    ) as coach_client:
        ensure_backend_available(backend_client, backend_base_url)
        for record in _sorted_records(records):
            backend_meta = fetch_backend_match_meta(
                backend_client, backend_base_url, record
            )
            payload_auto = build_request_payload(
                record, backend_meta, REQUEST_MODE_AUTO
            )
            payload_manual = build_request_payload(
                record, backend_meta, REQUEST_MODE_MANUAL
            )
            auto_capture = call_coach_analyze(
                coach_client, backend_base_url, payload_auto, headers
            )
            manual_capture = call_coach_analyze(
                coach_client, backend_base_url, payload_manual, headers
            )

            hard_failures = compare_backend_meta(record, backend_meta)
            series_game_no_mismatch = any(
                "seriesGameNo mismatch" in failure for failure in hard_failures
            )
            soft_warnings: List[str] = []

            for request_mode, capture in (
                (REQUEST_MODE_AUTO, auto_capture),
                (REQUEST_MODE_MANUAL, manual_capture),
            ):
                mode_failures, mode_warnings = validate_capture(
                    record=record,
                    request_mode=request_mode,
                    capture=capture,
                )
                hard_failures.extend(mode_failures)
                soft_warnings.extend(mode_warnings)

            results.append(
                {
                    "diagnosis": deepcopy(record),
                    "backend_meta": {
                        "game_id": backend_meta.game_id,
                        "season_id": backend_meta.season_id,
                        "league_type": backend_meta.league_type,
                        "post_season_series": backend_meta.post_season_series,
                        "series_game_no": backend_meta.series_game_no,
                        "home_pitcher": backend_meta.home_pitcher,
                        "away_pitcher": backend_meta.away_pitcher,
                        "game_status": backend_meta.game_status,
                        "detail_status_code": backend_meta.detail_status_code,
                        "detail_error": backend_meta.detail_error,
                    },
                    "requests": {
                        REQUEST_MODE_AUTO: payload_auto,
                        REQUEST_MODE_MANUAL: payload_manual,
                    },
                    "responses": {
                        REQUEST_MODE_AUTO: auto_capture,
                        REQUEST_MODE_MANUAL: manual_capture,
                    },
                    "hard_failures": hard_failures,
                    "soft_warnings": soft_warnings,
                    "series_game_no_mismatch": series_game_no_mismatch,
                    "ok": not hard_failures,
                }
            )

    reused = detect_reused_matchup_briefs(results)
    for result in results:
        game_id = str(result["diagnosis"].get("game_id"))
        if game_id in reused:
            result["hard_failures"].extend(reused[game_id])
            result["ok"] = False
        detail_error = result["backend_meta"].get("detail_error")
        if detail_error:
            result["soft_warnings"].append(str(detail_error))

    hard_failure_count = sum(len(item["hard_failures"]) for item in results)
    soft_warning_count = sum(len(item["soft_warnings"]) for item in results)
    return {
        "targets": [str(item["game_id"]) for item in _sorted_records(records)],
        "results": results,
        "summary": {
            "total_targets": len(records),
            "hard_failure_count": hard_failure_count,
            "soft_warning_count": soft_warning_count,
            "passed_targets": sum(1 for item in results if item["ok"]),
            "failed_targets": sum(1 for item in results if not item["ok"]),
            "series_game_no_mismatch_count": sum(
                1 for item in results if item.get("series_game_no_mismatch")
            ),
        },
    }


def build_recommendations(records: Sequence[Dict[str, Any]]) -> List[str]:
    root_cause_counts = Counter(
        code for item in records for code in (item.get("root_causes") or [])
    )
    recommendations: List[str] = []
    if root_cause_counts:
        top_code, _ = root_cause_counts.most_common(1)[0]
        if top_code == "missing_lineups":
            recommendations.append("라인업 적재 파이프라인을 우선 점검하세요.")
        elif top_code == "missing_starters":
            recommendations.append("선발 적재 파이프라인을 우선 점검하세요.")

    postseason_missing_series = any(
        item.get("stage_label") in POSTSEASON_STAGE_LABELS
        and "missing_series_context" in (item.get("root_causes") or [])
        for item in records
    )
    if postseason_missing_series:
        recommendations.append(
            "포스트시즌 season/stage 매핑과 series 계산 쿼리를 점검하세요."
        )
    if any(bool(item.get("series_state_partial")) for item in records):
        recommendations.append(
            "포스트시즌 DB 이력이 부족한 경기는 시리즈 스코어를 축약 표시하므로 series 백필 범위를 점검하세요."
        )

    if not recommendations:
        recommendations.append("즉시 조치할 데이터 적재 경고는 감지되지 않았습니다.")
    return recommendations


def build_report_payload(
    *,
    command: str,
    options: Dict[str, Any],
    diagnosis: Sequence[Dict[str, Any]],
    validation: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    diagnosis_summary = build_diagnosis_summary(diagnosis)
    validation_summary = (validation or {}).get(
        "summary",
        {
            "total_targets": 0,
            "hard_failure_count": 0,
            "soft_warning_count": 0,
            "passed_targets": 0,
            "failed_targets": 0,
            "series_game_no_mismatch_count": 0,
        },
    )
    recommendations = build_recommendations(diagnosis)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "options": options,
        "summary": {
            "command": command,
            "diagnosis": diagnosis_summary,
            "validation": validation_summary,
            "recommendations": recommendations,
        },
        "diagnosis": list(diagnosis),
        "validation": validation
        or {
            "targets": [],
            "results": [],
            "summary": validation_summary,
        },
    }


def render_markdown_report(report: Dict[str, Any]) -> str:
    diagnosis_summary = report["summary"]["diagnosis"]
    validation_summary = report["summary"]["validation"]
    recommendations = report["summary"]["recommendations"]
    lines = [
        "## 요약",
        f"- 실행 모드: {report['summary']['command']}",
        f"- 진단 경기 수: {diagnosis_summary['total_games']}",
        (
            "- 검증 결과: "
            f"{validation_summary['passed_targets']} 통과 / "
            f"{validation_summary['failed_targets']} 실패 / "
            f"{validation_summary['soft_warning_count']} 경고"
        ),
        "",
        "## 데이터 누락 진단",
        (
            "- 품질 분포: "
            f"grounded {diagnosis_summary['quality_distribution']['grounded']}, "
            f"partial {diagnosis_summary['quality_distribution']['partial']}, "
            f"insufficient {diagnosis_summary['quality_distribution']['insufficient']}"
        ),
        (
            "- 리그 분포: "
            f"정규 {diagnosis_summary['league_distribution']['regular']}, "
            f"포스트시즌 {diagnosis_summary['league_distribution']['postseason']}, "
            f"시범 {diagnosis_summary['league_distribution']['preseason']}, "
            f"기타 {diagnosis_summary['league_distribution']['unknown']}"
        ),
        (
            "- 시리즈 진단: "
            f"partial {diagnosis_summary['series_state_partial_count']}, "
            f"hint_mismatch {diagnosis_summary['series_state_hint_mismatch_count']}, "
            f"validation_seriesGameNo_mismatch {validation_summary.get('series_game_no_mismatch_count', 0)}"
        ),
        (
            "- 주요 원인: "
            + ", ".join(
                f"{code}={count}"
                for code, count in diagnosis_summary["root_cause_distribution"].items()
            )
        ),
        "",
        "## 실경기 검증",
    ]

    validation = report.get("validation") or {}
    results = validation.get("results") or []
    if not results:
        lines.append("- 검증 실행 없음")
    else:
        for item in results:
            diagnosis = item["diagnosis"]
            lines.append(
                (
                    f"- {diagnosis['game_id']} {diagnosis['away_team_id']}@{diagnosis['home_team_id']}: "
                    f"hard={len(item['hard_failures'])}, soft={len(item['soft_warnings'])}, ok={item['ok']}"
                )
            )
            if item["hard_failures"]:
                lines.append(
                    "  hard: " + " | ".join(str(text) for text in item["hard_failures"])
                )
            if item["soft_warnings"]:
                lines.append(
                    "  soft: " + " | ".join(str(text) for text in item["soft_warnings"])
                )

    lines.extend(["", "## 권장 조치"])
    for recommendation in recommendations:
        lines.append(f"- {recommendation}")
    lines.append("")
    return "\n".join(lines)


def write_report_files(report: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"coach_grounding_{timestamp}.json"
    markdown_path = output_dir / f"coach_grounding_{timestamp}.md"
    latest_json_path = output_dir / "coach_grounding_latest.json"
    latest_markdown_path = output_dir / "coach_grounding_latest.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    shutil.copyfile(json_path, latest_json_path)
    shutil.copyfile(markdown_path, latest_markdown_path)

    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "latest_json": str(latest_json_path),
        "latest_markdown": str(latest_markdown_path),
    }


def load_validation_input(
    *,
    season_year: Optional[int],
    date_from: Optional[str],
    date_to: Optional[str],
    game_ids: Sequence[str],
    max_games: int,
    diagnosis_json: Optional[str],
) -> List[Dict[str, Any]]:
    if diagnosis_json:
        records = load_diagnosis_report(diagnosis_json)
        if game_ids:
            requested = set(game_ids)
            records = [
                item for item in records if str(item.get("game_id")) in requested
            ]
        if max_games > 0:
            return _sorted_records(records)[:max_games]
        return _sorted_records(records)

    if not game_ids:
        raise ValueError("validate_requires_game_ids_or_diagnosis_json")

    return diagnose_games(
        season_year=season_year,
        date_from=date_from,
        date_to=date_to,
        game_ids=game_ids,
        max_games=max_games,
    )


def _build_options_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "command": args.command,
        "season_year": args.season_year,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "game_ids": parse_game_ids(args.game_ids),
        "max_games": args.max_games,
        "backend_base_url": args.backend_base_url,
        "output_dir": str(resolve_output_dir(args.output_dir)),
        "strict": bool(args.strict),
        "internal_api_key_provided": bool(args.internal_api_key),
        "timeout_seconds": args.timeout_seconds,
        "coach_timeout_seconds": args.coach_timeout_seconds,
        "diagnosis_json": getattr(args, "diagnosis_json", None),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    options = _build_options_dict(args)
    output_dir = resolve_output_dir(args.output_dir)
    game_ids = parse_game_ids(args.game_ids)
    diagnosis: List[Dict[str, Any]]
    validation: Optional[Dict[str, Any]] = None

    try:
        if args.command == "diagnose":
            diagnosis = diagnose_games(
                season_year=args.season_year,
                date_from=args.date_from,
                date_to=args.date_to,
                game_ids=game_ids,
                max_games=args.max_games,
            )
        elif args.command == "validate":
            diagnosis = load_validation_input(
                season_year=args.season_year,
                date_from=args.date_from,
                date_to=args.date_to,
                game_ids=game_ids,
                max_games=args.max_games,
                diagnosis_json=args.diagnosis_json,
            )
            validation = validate_records(
                records=diagnosis,
                backend_base_url=args.backend_base_url,
                internal_api_key=args.internal_api_key,
                timeout_seconds=args.timeout_seconds,
                coach_timeout_seconds=args.coach_timeout_seconds,
            )
        else:
            if args.diagnosis_json:
                diagnosis = load_diagnosis_report(args.diagnosis_json)
                if game_ids:
                    requested = set(game_ids)
                    diagnosis = [
                        item
                        for item in diagnosis
                        if str(item.get("game_id")) in requested
                    ]
                if args.max_games > 0:
                    diagnosis = _sorted_records(diagnosis)[: args.max_games]
            else:
                diagnosis = diagnose_games(
                    season_year=args.season_year,
                    date_from=args.date_from,
                    date_to=args.date_to,
                    game_ids=game_ids,
                    max_games=args.max_games,
                )
            selected = select_validation_samples(
                diagnosis,
                limit=min(DEFAULT_VALIDATION_SAMPLE_SIZE, len(diagnosis)),
            )
            validation = validate_records(
                records=selected,
                backend_base_url=args.backend_base_url,
                internal_api_key=args.internal_api_key,
                timeout_seconds=args.timeout_seconds,
                coach_timeout_seconds=args.coach_timeout_seconds,
            )

        report = build_report_payload(
            command=args.command,
            options=options,
            diagnosis=diagnosis,
            validation=validation,
        )
        paths = write_report_files(report, output_dir)
        print(
            json.dumps(
                {"summary": report["summary"], "paths": paths},
                ensure_ascii=False,
                indent=2,
            )
        )

        hard_failure_count = report["summary"]["validation"]["hard_failure_count"]
        if args.strict and hard_failure_count > 0:
            return 1
        return 0
    except Exception as exc:
        error_report = build_report_payload(
            command=args.command,
            options=options,
            diagnosis=diagnosis if "diagnosis" in locals() else [],
            validation=(validation or {})
            | {
                "error": str(exc),
                "summary": (
                    (validation or {}).get(
                        "summary",
                        {
                            "total_targets": 0,
                            "hard_failure_count": 1,
                            "soft_warning_count": 0,
                            "passed_targets": 0,
                            "failed_targets": 0,
                        },
                    )
                ),
            },
        )
        error_paths = write_report_files(error_report, output_dir)
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "paths": error_paths,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
