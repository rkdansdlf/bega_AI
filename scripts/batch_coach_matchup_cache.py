#!/usr/bin/env python3
"""
Prediction 자동 조회와 동일한 매치업 캐시 키를 수동 배치로 생성합니다.

기본 정책:
- 런타임 `/coach/analyze`는 자동 재생성을 하지 않습니다.
- 캐시 재생성은 이 스크립트 같은 수동 배치에서만 수행합니다.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.coach_cache_key import (
    build_coach_cache_key,
    build_focus_signature,
    build_lineup_signature,
    build_starter_signature,
    normalize_focus,
)
from app.deps import get_connection_pool
from app.tools.team_code_resolver import CANONICAL_CODES, TeamCodeResolver

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

CACHE_SCHEMA_VERSION = "v5"
PROMPT_VERSION = "v10_postseason_matchup_scope"
LOCAL_TOKEN_ENV_FILES = (".env.prod", ".env")
LEAGUE_TYPE_TO_CODES = {
    "REGULAR": (0,),
    "PRE": (1,),
    "POST": (2, 3, 4, 5),
}
LEAGUE_CODE_TO_TYPE = {
    0: "REGULAR",
    1: "PRE",
    2: "POST",
    3: "POST",
    4: "POST",
    5: "POST",
}
MATCHUP_FOCUS = ["matchup", "recent_form"]
AUTO_BRIEF_FOCUS = ["recent_form"]
VALID_REQUEST_MODES = ("manual_detail",)
VALID_TARGET_ORDERS = ("asc", "desc")
VALID_STATUS_BUCKET_FILTERS = ("ANY", "COMPLETED", "LIVE", "SCHEDULED")
VALID_CACHE_STATE_FILTERS = (
    "ANY",
    "MISSING",
    "COMPLETED",
    "FAILED",
    "PENDING",
    "UNRESOLVED",
)
RETRYABLE_FAILURE_PREFIXES = (
    "http_429:",
    "http_500:",
    "http_502:",
    "http_503:",
    "http_504:",
    "readtimeout",
    "readerror",
    "server disconnected",
    "peer closed connection",
    "coach_internal_error",
    "target_wall_timeout",
    "failed_locked",
    "missing_done_event_timeout",
    "empty_response_stream",
    "missing_done_event",
    "분석 처리 중 오류가 발생했습니다.",
)
DONE_WAIT_SECONDS = 20.0
DONE_WAIT_INTERVAL_SECONDS = 1.0
TARGET_TIMEOUT_BUFFER_SECONDS = 15.0
COACH_YEAR_MIN = 1982
POSTSEASON_LEAGUE_TYPES = {"POST", "ANY"}
LEAGUE_CODE_TO_STAGE = {
    0: "REGULAR",
    1: "PRE",
    2: "WC",
    3: "SEMI_PO",
    4: "PO",
    5: "KS",
}


def _read_env_file_entries(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    entries: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        entries[key] = value.strip().strip('"').strip("'")
    return entries


def _preload_workspace_env(project_root: Path) -> None:
    for filename in LOCAL_TOKEN_ENV_FILES:
        for key, value in _read_env_file_entries(project_root / filename).items():
            os.environ.setdefault(key, value)


def resolve_default_internal_api_key(project_root: Path) -> str:
    env_value = (os.getenv("AI_INTERNAL_TOKEN", "") or "").strip()
    if env_value:
        return env_value

    for filename in LOCAL_TOKEN_ENV_FILES:
        token = (
            _read_env_file_entries(project_root / filename).get("AI_INTERNAL_TOKEN", "")
            or ""
        ).strip()
        if token:
            return token
    return ""


def _league_codes_for_type(league_type: str) -> tuple[int, ...]:
    return LEAGUE_TYPE_TO_CODES[league_type]


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


WORKSPACE_ROOT = _detect_workspace_root(PROJECT_ROOT)
_preload_workspace_env(WORKSPACE_ROOT)
_POSTSEASON_REPAIR_MODULE: Any = None


def _normalize_name_token(value: Optional[str]) -> Optional[str]:
    normalized = " ".join(str(value or "").split()).strip()
    return normalized or None


def _normalize_game_status_bucket(game_status: Optional[str]) -> str:
    normalized = str(game_status or "").strip().upper()
    if normalized in {"COMPLETED", "FINAL", "FINISHED", "DONE", "END", "E", "F"}:
        return "COMPLETED"
    if normalized in {"LIVE", "IN_PROGRESS", "INPROGRESS", "PLAYING"}:
        return "LIVE"
    if normalized in {"SCHEDULED", "READY", "NOT_STARTED", "PENDING"}:
        return "SCHEDULED"
    if normalized in {"POSTPONED", "CANCELLED", "CANCELED", "SUSPENDED"}:
        return "COMPLETED"
    return "UNKNOWN"


def _load_postseason_repair_module() -> Any:
    global _POSTSEASON_REPAIR_MODULE
    if _POSTSEASON_REPAIR_MODULE is not None:
        return _POSTSEASON_REPAIR_MODULE

    module_path = WORKSPACE_ROOT / "scripts" / "repair_postseason_season_ids.py"
    spec = importlib.util.spec_from_file_location(
        "root_repair_postseason_season_ids",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed_to_load_repair_module path={module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _POSTSEASON_REPAIR_MODULE = module
    return module


def _postseason_mismatch_error_message(mismatches: List[Any]) -> str:
    if not mismatches:
        return ""
    sample_ids = ", ".join(str(item.game_id) for item in mismatches[:3])
    return (
        "postseason_season_id_mismatch_detected "
        f"count={len(mismatches)} sample={sample_ids} "
        "run scripts/repair_postseason_season_ids.py --years <year> --apply first"
    )


def _ensure_postseason_stage_integrity(years: List[int]) -> None:
    repair_module = _load_postseason_repair_module()
    pool = get_connection_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            stages_by_year = repair_module.load_season_stages(cur, years)
            games = repair_module.load_games(cur, years)
    mismatches = repair_module.collect_mismatches(games, stages_by_year)
    if mismatches:
        raise RuntimeError(_postseason_mismatch_error_message(mismatches))


@dataclass
class MatchupTarget:
    cache_key: str
    game_id: str
    season_id: int
    season_year: int
    game_date: str
    game_type: str
    home_team_id: str
    away_team_id: str
    league_type_code: int
    stage_label: str
    series_game_no: Optional[int]
    game_status_bucket: str
    starter_signature: str
    lineup_signature: str
    request_focus: List[str]
    request_mode: str
    question_override: Optional[str]


def _build_analyze_payload(target: MatchupTarget) -> Dict[str, Any]:
    league_context: Dict[str, Any] = {
        "season": target.season_id,
        "season_year": target.season_year,
        "game_date": target.game_date,
        "league_type": target.game_type,
        "league_type_code": target.league_type_code,
        "round": target.stage_label,
        "stage_label": target.stage_label,
    }
    if target.series_game_no is not None:
        league_context["game_no"] = target.series_game_no
        league_context["series_game_no"] = target.series_game_no

    payload: Dict[str, Any] = {
        "home_team_id": target.home_team_id,
        "away_team_id": target.away_team_id,
        "league_context": league_context,
        "focus": target.request_focus,
        "request_mode": target.request_mode,
        "game_id": target.game_id,
    }
    if target.question_override:
        payload["question_override"] = target.question_override
    return payload


def parse_years(raw: str) -> List[int]:
    years = sorted({int(token.strip()) for token in raw.split(",") if token.strip()})
    if not years:
        raise ValueError("at least one year is required")
    return years


def parse_league_type(raw: str) -> str:
    normalized = str(raw or "REGULAR").strip().upper()
    if normalized not in {"REGULAR", "PRE", "POST", "ANY"}:
        raise ValueError("league-type must be one of REGULAR, PRE, POST, ANY")
    return normalized


def parse_focus(raw: str | None, *, default: List[str]) -> List[str]:
    if not raw:
        return normalize_focus(default)

    normalized = normalize_focus([token.strip() for token in raw.split(",")])
    return normalized or normalize_focus(default)


def parse_request_mode(raw: str | None) -> str:
    mode = str(raw or "manual_detail").strip().lower()
    if mode not in VALID_REQUEST_MODES:
        raise ValueError(
            f"mode must be one of {', '.join(VALID_REQUEST_MODES)} (got={raw!r})"
        )
    return mode


def parse_target_order(raw: str | None) -> str:
    normalized = str(raw or "asc").strip().lower()
    if normalized not in VALID_TARGET_ORDERS:
        raise ValueError(
            f"order must be one of {', '.join(VALID_TARGET_ORDERS)} (got={raw!r})"
        )
    return normalized


def parse_status_bucket_filter(raw: str | None) -> str:
    normalized = str(raw or "ANY").strip().upper()
    if normalized not in VALID_STATUS_BUCKET_FILTERS:
        raise ValueError(
            "status-bucket must be one of "
            f"{', '.join(VALID_STATUS_BUCKET_FILTERS)} (got={raw!r})"
        )
    return normalized


def parse_cache_state_filter(raw: str | None) -> str:
    normalized = str(raw or "ANY").strip().upper()
    if normalized not in VALID_CACHE_STATE_FILTERS:
        raise ValueError(
            "cache-state-filter must be one of "
            f"{', '.join(VALID_CACHE_STATE_FILTERS)} (got={raw!r})"
        )
    return normalized


def parse_question_override(raw: str | None) -> Optional[str]:
    if raw is None:
        return None
    normalized = " ".join(raw.split())
    return normalized or None


def load_failed_cache_keys(report_path: str | None) -> set[str]:
    if not report_path:
        return set()

    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    details = payload.get("details")
    if not isinstance(details, list):
        raise ValueError("quality report details must be a list")

    failed_cache_keys: set[str] = set()
    for item in details:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").strip().lower() != "failed":
            continue
        cache_key = str(item.get("cache_key") or "").strip()
        if cache_key:
            failed_cache_keys.add(cache_key)
    return failed_cache_keys


def _is_done_marker(data_str: str) -> bool:
    normalized = data_str.strip().lower()
    return normalized in {"[done]", "done", '"[done]"'}


def _extract_error_message(data_str: str) -> str | None:
    data = data_str.strip()
    if not data:
        return None

    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return data

    if not isinstance(parsed, dict):
        return data

    for key in ("error", "message", "detail", "reason"):
        value = parsed.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text

    return None


def _classify_meta_result(meta_payload: Dict[str, Any]) -> tuple[str, str]:
    cache_state = str(meta_payload.get("cache_state") or "")
    validation_status = str(meta_payload.get("validation_status") or "")
    generation_mode = str(meta_payload.get("generation_mode") or "")

    if cache_state == "FAILED_LOCKED":
        return "failed", "failed_locked"
    if meta_payload.get("in_progress") is True:
        return "in_progress", "pending_wait"
    if meta_payload.get("cached") is True:
        return "skipped", "cache_hit"
    if validation_status == "success":
        return "generated", "generated"
    if validation_status == "fallback" and generation_mode == "evidence_fallback":
        return "generated", "generated_fallback"
    return "failed", cache_state.lower() if cache_state else "validation_fallback"


def _normalize_status_counts(results: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "generated": 0,
        "skipped": 0,
        "in_progress": 0,
        "failed": 0,
    }
    for item in results:
        status = item.get("status")
        if status in summary:
            summary[status] += 1
        else:
            summary["failed"] += 1
    return summary


def _sort_targets(
    targets: List[MatchupTarget],
    *,
    order: str,
) -> List[MatchupTarget]:
    reverse = order == "desc"
    return sorted(
        targets,
        key=lambda target: (target.game_date, target.game_id),
        reverse=reverse,
    )


def _fetch_cache_state(
    cache_key: str,
) -> Optional[tuple[str, Any, Optional[str]]]:
    try:
        with get_connection_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT status, response_json, error_message
                FROM coach_analysis_cache
                WHERE cache_key = %s
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        status = str(row[0] or "")
        response_json = row[1]
        error_message = row[2]
        return status, response_json, error_message
    except Exception:
        return None


def _fetch_cache_rows(
    cache_keys: List[str],
) -> Dict[str, tuple[str, Any, Optional[str]]]:
    if not cache_keys:
        return {}

    with get_connection_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT cache_key, status, response_json, error_message
            FROM coach_analysis_cache
            WHERE cache_key = ANY(%s)
            """,
            (cache_keys,),
        ).fetchall()

    return {
        str(row[0]): (
            str(row[1] or ""),
            row[2],
            row[3],
        )
        for row in rows
    }


def _normalize_cache_state_label(
    row: Optional[tuple[str, Any, Optional[str]]],
) -> str:
    if row is None:
        return "MISSING"

    status = str(row[0] or "").strip().upper()
    if status == "COMPLETED":
        return "COMPLETED"
    if status == "FAILED":
        return "FAILED"
    if status in {"PENDING", "PENDING_WAIT", "FAILED_LOCKED"}:
        return "PENDING"
    return "FAILED"


def _filter_targets_by_cache_state(
    targets: List[MatchupTarget],
    cache_state_filter: str,
) -> List[MatchupTarget]:
    if cache_state_filter == "ANY" or not targets:
        return targets

    row_by_cache_key = _fetch_cache_rows([target.cache_key for target in targets])
    filtered: List[MatchupTarget] = []
    for target in targets:
        state = _normalize_cache_state_label(row_by_cache_key.get(target.cache_key))
        if cache_state_filter == "UNRESOLVED":
            if state != "COMPLETED":
                filtered.append(target)
            continue
        if state == cache_state_filter:
            filtered.append(target)
    return filtered


def _extract_cached_meta(response_json: Any) -> Dict[str, Any]:
    if not isinstance(response_json, dict):
        return {}

    meta = response_json.get("meta")
    if isinstance(meta, dict):
        return dict(meta)
    return {}


def _build_cache_verification_result(
    target: MatchupTarget,
    row: Optional[tuple[str, Any, Optional[str]]],
) -> Dict[str, Any]:
    result = _build_result_shell(target)
    if row is None:
        result["reason"] = "missing_cache_row"
        return result

    cache_state, response_json, error_message = row
    normalized_state = str(cache_state or "").strip().upper()

    if normalized_state == "COMPLETED":
        meta = _extract_cached_meta(response_json)
        meta.setdefault("cached", True)
        meta.setdefault("in_progress", False)
        meta.setdefault("cache_state", normalized_state)
        result["meta"] = meta
        result["status"], result["reason"] = _classify_meta_result(meta)
        return result

    if normalized_state in {"PENDING", "PENDING_WAIT"}:
        result["status"] = "in_progress"
        result["reason"] = normalized_state.lower()
        result["meta"] = {"cache_state": normalized_state, "in_progress": True}
        return result

    if normalized_state == "FAILED_LOCKED":
        result["reason"] = "failed_locked"
        result["meta"] = {"cache_state": normalized_state}
        return result

    if normalized_state == "FAILED":
        result["reason"] = str(error_message or "failed_cache_row")
        result["meta"] = {"cache_state": normalized_state}
        return result

    result["reason"] = (
        f"unexpected_cache_state:{normalized_state.lower()}"
        if normalized_state
        else "unexpected_cache_state"
    )
    result["meta"] = {"cache_state": normalized_state}
    return result


def _build_terminal_cache_result_if_available(
    target: MatchupTarget,
) -> Optional[Dict[str, Any]]:
    row = _fetch_cache_state(target.cache_key)
    if row is None:
        return None

    cache_state = str(row[0] or "").strip().upper()
    if cache_state in {"COMPLETED", "FAILED", "FAILED_LOCKED"}:
        return _build_cache_verification_result(target, row)
    return None


def _is_retryable_failure_reason(reason: str | None) -> bool:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return False
    return normalized.startswith(RETRYABLE_FAILURE_PREFIXES)


def collect_cache_verification_results(
    targets: List[MatchupTarget],
) -> List[Dict[str, Any]]:
    row_by_cache_key = _fetch_cache_rows([target.cache_key for target in targets])
    return [
        _build_cache_verification_result(target, row_by_cache_key.get(target.cache_key))
        for target in targets
    ]


async def _wait_cache_completion(
    cache_key: str,
) -> tuple[str | None, Any, Optional[str], bool]:
    attempts = max(1, int(DONE_WAIT_SECONDS / DONE_WAIT_INTERVAL_SECONDS))
    for _ in range(attempts):
        row = _fetch_cache_state(cache_key)
        if row is None:
            return None, None, None, False

        status, response_json, error_message = row
        if status in {"COMPLETED", "FAILED"}:
            return status, response_json, error_message, True
        if status in {"PENDING", "PENDING_WAIT", "FAILED_LOCKED"}:
            # continue polling
            pass
        await asyncio.sleep(DONE_WAIT_INTERVAL_SECONDS)
    return "TIMEOUT", None, "cache_not_ready", False


def load_targets(
    *,
    years: List[int],
    league_type: str,
    request_focus: List[str],
    request_mode: str,
    question_override: Optional[str],
    offset: int,
    limit: Optional[int],
    order: str = "asc",
    status_bucket_filter: str = "ANY",
) -> List[MatchupTarget]:
    resolver = TeamCodeResolver()
    pool = get_connection_pool()

    where_parts = ["ks.season_year = ANY(%s)"]
    params: List[Any] = [years]
    if league_type != "ANY":
        where_parts.append("ks.league_type_code = ANY(%s)")
        params.append(list(_league_codes_for_type(league_type)))

    where_sql = " AND ".join(where_parts)
    query = f"""
        SELECT
            g.game_id,
            g.home_team,
            g.away_team,
            g.game_date,
            g.game_status,
            g.home_pitcher,
            g.away_pitcher,
            g.home_score,
            g.away_score,
            g.season_id,
            ks.season_year,
            ks.league_type_code
        FROM game g
        JOIN kbo_seasons ks ON g.season_id = ks.season_id
        WHERE {where_sql}
        ORDER BY g.game_date ASC, g.game_id ASC
    """

    with pool.connection() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
        lineup_rows = conn.execute(
            """
            SELECT game_id, team_code, player_name, batting_order
            FROM game_lineups
            WHERE game_id = ANY(%s)
              AND COALESCE(is_starter, true) = true
            ORDER BY game_id ASC, team_code ASC, batting_order ASC NULLS LAST, player_name ASC
            """,
            ([str(row[0]) for row in rows],),
        ).fetchall()

    lineup_by_game: Dict[str, Dict[str, List[str]]] = {}
    for lineup_row in lineup_rows or []:
        game_id = str(lineup_row[0])
        team_code = resolver.resolve_canonical(lineup_row[1])
        player_name = _normalize_name_token(lineup_row[2])
        if not team_code or not player_name:
            continue
        game_entry = lineup_by_game.setdefault(game_id, {})
        game_entry.setdefault(team_code, []).append(player_name)

    deduped: List[MatchupTarget] = []
    seen = set()
    postseason_game_numbers: Dict[tuple[int, str, str], int] = {}
    for row in rows:
        (
            game_id,
            home_team,
            away_team,
            game_date,
            game_status,
            home_pitcher,
            away_pitcher,
            home_score,
            away_score,
            season_id,
            season_year,
            league_code,
        ) = row
        home_canonical = resolver.resolve_canonical(home_team)
        away_canonical = resolver.resolve_canonical(away_team)

        if (
            home_canonical not in CANONICAL_CODES
            or away_canonical not in CANONICAL_CODES
        ):
            continue

        normalized_league_code = int(league_code)
        game_type = LEAGUE_CODE_TO_TYPE.get(normalized_league_code, "UNKNOWN")
        stage_label = LEAGUE_CODE_TO_STAGE.get(normalized_league_code, "UNKNOWN")
        resolved_game_status = str(game_status or "UNKNOWN")
        if (
            resolved_game_status.strip().upper() in {"", "UNKNOWN"}
            and home_score is not None
            and away_score is not None
        ):
            resolved_game_status = "COMPLETED"
        game_status_bucket = _normalize_game_status_bucket(resolved_game_status)
        if status_bucket_filter != "ANY" and game_status_bucket != status_bucket_filter:
            continue
        game_lineups = lineup_by_game.get(str(game_id), {})
        lineup_signature = build_lineup_signature(
            [
                *game_lineups.get(home_canonical, []),
                *game_lineups.get(away_canonical, []),
            ]
        )
        starter_signature = build_starter_signature(
            _normalize_name_token(home_pitcher),
            _normalize_name_token(away_pitcher),
        )
        series_game_no: Optional[int] = None
        if normalized_league_code in {2, 3, 4, 5}:
            matchup_key = (
                int(season_id),
                min(home_canonical, away_canonical),
                max(home_canonical, away_canonical),
            )
            previous_game_count = postseason_game_numbers.get(matchup_key, 0)
            series_game_no = previous_game_count + 1
            postseason_game_numbers[matchup_key] = series_game_no
        dedupe_key = str(game_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        cache_key, _payload = build_coach_cache_key(
            schema_version=CACHE_SCHEMA_VERSION,
            prompt_version=PROMPT_VERSION,
            home_team_code=home_canonical,
            away_team_code=away_canonical,
            year=int(season_year),
            game_type=game_type,
            focus=request_focus,
            question_override=question_override,
            game_id=str(game_id),
            league_type_code=normalized_league_code,
            stage_label=stage_label,
            starter_signature=starter_signature,
            lineup_signature=lineup_signature,
            request_mode=request_mode,
            game_status_bucket=game_status_bucket,
        )
        deduped.append(
            MatchupTarget(
                cache_key=cache_key,
                game_id=str(game_id),
                season_id=int(season_id),
                season_year=int(season_year),
                game_date=(
                    game_date.isoformat()
                    if hasattr(game_date, "isoformat")
                    else str(game_date)
                ),
                game_type=game_type,
                home_team_id=home_canonical,
                away_team_id=away_canonical,
                league_type_code=normalized_league_code,
                stage_label=stage_label,
                series_game_no=series_game_no,
                game_status_bucket=game_status_bucket,
                starter_signature=starter_signature,
                lineup_signature=lineup_signature,
                request_focus=list(request_focus),
                request_mode=request_mode,
                question_override=question_override,
            )
        )

    ordered = _sort_targets(deduped, order=order)
    sliced = ordered[offset:]
    if limit is not None and limit >= 0:
        sliced = sliced[:limit]
    return sliced


def force_rebuild_delete(cache_keys: List[str]) -> int:
    if not cache_keys:
        return 0
    pool = get_connection_pool()
    with pool.connection() as conn:
        result = conn.execute(
            "DELETE FROM coach_analysis_cache WHERE cache_key = ANY(%s)",
            (cache_keys,),
        )
        conn.commit()
    return result.rowcount or 0


def _build_result_shell(target: MatchupTarget) -> Dict[str, Any]:
    return {
        "cache_key": target.cache_key,
        "game_id": target.game_id,
        "home_team_id": target.home_team_id,
        "away_team_id": target.away_team_id,
        "year": target.season_year,
        "game_type": target.game_type,
        "status": "failed",
        "reason": None,
        "meta": {},
    }


async def call_analyze(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    target: MatchupTarget,
) -> Dict[str, Any]:
    payload = _build_analyze_payload(target)
    result: Dict[str, Any] = _build_result_shell(target)

    current_event = "message"
    saw_done = False
    error_message = None
    meta_payload: Dict[str, Any] = {}
    saw_message_event = False
    saw_meta_event = False
    saw_any_event = False
    saw_error_event = False

    try:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/coach/analyze",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                result["reason"] = (
                    f"http_{response.status_code}:{body.decode('utf-8', errors='replace')[:160]}"
                )
                return result

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip().lower()
                    continue
                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if not data_str:
                    continue

                saw_any_event = True
                if _is_done_marker(data_str):
                    saw_done = True
                    continue

                if current_event == "meta":
                    try:
                        parsed = json.loads(data_str)
                        if isinstance(parsed, dict):
                            meta_payload = parsed
                    except json.JSONDecodeError:
                        pass
                elif current_event == "error":
                    error_message = _extract_error_message(data_str)
                    saw_error_event = True
                elif current_event == "message":
                    saw_message_event = True
                    if data_str and data_str.startswith('{"error"'):
                        possible_error = _extract_error_message(data_str)
                        if possible_error:
                            error_message = possible_error

                if current_event == "meta":
                    saw_meta_event = True

    except Exception as exc:
        result["reason"] = str(exc) or exc.__class__.__name__
        return result

    if not saw_done:
        status, response_json, cache_error, was_terminal = await _wait_cache_completion(
            target.cache_key
        )
        if was_terminal:
            if status == "COMPLETED" and response_json:
                result["status"] = "generated"
                result["reason"] = "generated_without_done_event"
                result["meta"] = {"cache_state": "COMPLETED", "in_progress": False}
                return result
            if status == "FAILED":
                result["status"] = "failed"
                result["reason"] = str(cache_error or "failed_without_done_event")
                return result

        if status == "TIMEOUT":
            result["status"] = "failed"
            result["reason"] = "missing_done_event_timeout"
            return result

        if status == "FAILED_LOCKED":
            result["status"] = "failed"
            result["reason"] = "failed_locked_without_done_event"
            return result

        if saw_error_event and error_message:
            result["reason"] = error_message
            return result
        if not saw_any_event:
            result["reason"] = "empty_response_stream"
        elif not saw_message_event and not saw_meta_event:
            result["reason"] = "missing_done_event_no_payload"
        else:
            result["reason"] = "missing_done_event"
        return result
    if error_message:
        result["reason"] = error_message
        return result

    result["meta"] = meta_payload
    result["status"], result["reason"] = _classify_meta_result(meta_payload)
    return result


async def call_analyze_with_deadline(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    target: MatchupTarget,
    timeout_seconds: float,
) -> Dict[str, Any]:
    try:
        return await asyncio.wait_for(
            call_analyze(
                client=client,
                base_url=base_url,
                target=target,
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        recovered = _build_terminal_cache_result_if_available(target)
        if recovered is not None:
            recovered_meta = dict(recovered.get("meta") or {})
            recovered_meta.setdefault("timeout_seconds", round(timeout_seconds, 3))
            recovered["meta"] = recovered_meta
            return recovered
        result = _build_result_shell(target)
        result["reason"] = "target_wall_timeout"
        result["meta"] = {"timeout_seconds": round(timeout_seconds, 3)}
        return result


async def call_analyze_with_retries(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    target: MatchupTarget,
    timeout_seconds: float,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> Dict[str, Any]:
    attempts_allowed = max(1, int(max_attempts))
    backoff_seconds = max(0.0, float(retry_backoff_seconds))
    last_result = _build_result_shell(target)

    for attempt in range(1, attempts_allowed + 1):
        item = await call_analyze_with_deadline(
            client=client,
            base_url=base_url,
            target=target,
            timeout_seconds=timeout_seconds,
        )
        item_meta = dict(item.get("meta") or {})
        item_meta["attempt"] = attempt
        item["meta"] = item_meta
        last_result = item

        if item.get("status") != "failed":
            return item

        recovered = _build_terminal_cache_result_if_available(target)
        if recovered is not None:
            recovered_meta = dict(recovered.get("meta") or {})
            recovered_meta["attempt"] = attempt
            recovered["meta"] = recovered_meta
            return recovered

        reason = str(item.get("reason") or "")
        if attempt >= attempts_allowed or not _is_retryable_failure_reason(reason):
            return item

        if reason.startswith("failed_locked"):
            deleted = force_rebuild_delete([target.cache_key])
            item["meta"]["retry_deleted_cache_rows"] = deleted

        await asyncio.sleep(backoff_seconds * attempt)

    return last_result


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts = _normalize_status_counts(results)
    total = (
        status_counts["generated"]
        + status_counts["skipped"]
        + status_counts["failed"]
        + status_counts["in_progress"]
    )

    return {
        "cases": total,
        "all_ok": status_counts["failed"] == 0 and status_counts["in_progress"] == 0,
        "success": status_counts["generated"],
        "skipped": status_counts["skipped"],
        "in_progress": status_counts["in_progress"],
        "failed": status_counts["failed"],
        "generated_success_count": status_counts["generated"],
        "skipped_count": status_counts["skipped"],
    }


def collect_matchup_integrity_metrics(
    years: List[int],
    years_alias: List[str] | None = None,
) -> tuple[int, int]:
    legacy_aliases = years_alias or []
    pool = get_connection_pool()
    with pool.connection() as conn:
        cache_invalid_year_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM coach_analysis_cache
            WHERE year < %s OR year > %s
            """,
            (COACH_YEAR_MIN, datetime.now().year + 1),
        ).fetchone()[0]

        legacy_residual_total = 0
        if legacy_aliases:
            legacy_residual_total = conn.execute(
                """
                SELECT COUNT(*)
                FROM coach_analysis_cache
                WHERE year = ANY(%s)
                  AND UPPER(team_id) = ANY(%s)
                """,
                (years, legacy_aliases),
            ).fetchone()[0]

    return int(cache_invalid_year_count), int(legacy_residual_total)


def summarize_matchup_results(
    results: List[Dict[str, Any]],
    years: List[int],
    league_type: str,
    focus: List[str],
) -> Dict[str, Any]:
    status_counts = _normalize_status_counts(results)
    total = (
        status_counts["generated"]
        + status_counts["skipped"]
        + status_counts["failed"]
        + status_counts["in_progress"]
    )
    warnings_total = 0
    critical_over_limit_count = 0
    llm_manual_count = 0
    evidence_fallback_count = 0
    deterministic_auto_count = 0
    focus_section_missing_count = 0
    seen_cache_keys: set[str] = set()
    target_teams: set[str] = set()
    failure_reasons: list[str] = []

    for item in results:
        warnings = item.get("warnings_count")
        if isinstance(warnings, (int, float)):
            warnings_total += int(warnings)
        if item.get("critical_over_limit"):
            critical_over_limit_count += 1

        meta = item.get("meta") or {}
        generation_mode = str(meta.get("generation_mode") or "")
        if generation_mode == "llm_manual":
            llm_manual_count += 1
        elif generation_mode == "evidence_fallback":
            evidence_fallback_count += 1
        elif generation_mode == "deterministic_auto":
            deterministic_auto_count += 1

        if bool(meta.get("focus_section_missing")):
            focus_section_missing_count += 1

        cache_key = item.get("cache_key")
        if isinstance(cache_key, str):
            seen_cache_keys.add(cache_key)

        home_team_id = item.get("home_team_id")
        away_team_id = item.get("away_team_id")
        if isinstance(home_team_id, str):
            target_teams.add(home_team_id.upper())
        if isinstance(away_team_id, str):
            target_teams.add(away_team_id.upper())

        if item.get("status") == "failed":
            reason = item.get("reason")
            if isinstance(reason, str) and reason:
                failure_reasons.append(reason)

    focus_signature = build_focus_signature(focus)

    failure_category_counts: Dict[str, int] = {}
    for item in results:
        if item.get("status") != "failed":
            continue
        reason = item.get("reason") or "unknown"
        failure_category_counts[str(reason)] = (
            failure_category_counts.get(str(reason), 0) + 1
        )

    try:
        from app.tools.team_code_resolver import TeamCodeResolver

        resolver = TeamCodeResolver()
        legacy_aliases: List[str] = []
        for aliases in resolver.team_variants.values():
            for code in aliases:
                if str(code).upper() not in CANONICAL_CODES:
                    legacy_aliases.append(str(code).upper())
        for aliases in getattr(resolver, "legacy_code_map", {}).keys():
            legacy_aliases.append(str(aliases).upper())
        legacy_aliases = sorted(set(legacy_aliases))
        cache_invalid_year_count, legacy_residual_total = (
            collect_matchup_integrity_metrics(
                years,
                legacy_aliases,
            )
        )
    except Exception as exc:
        logger.warning("Failed to collect integrity metrics: %s", exc)
        cache_invalid_year_count = 0
        legacy_residual_total = 0

    return {
        "cases": total,
        "all_ok": status_counts["failed"] == 0 and status_counts["in_progress"] == 0,
        "target_count": total,
        "success": status_counts["generated"],
        "skipped": status_counts["skipped"],
        "in_progress": status_counts["in_progress"],
        "failed": status_counts["failed"],
        "generated_success_count": status_counts["generated"],
        "skipped_count": status_counts["skipped"],
        "target_years": sorted(set(years)),
        "target_teams": sorted(target_teams),
        "focus_signature": focus_signature,
        "cache_key_count": len(seen_cache_keys),
        "game_type": league_type.upper(),
        "coverage_rate": (
            round((status_counts["generated"] + status_counts["skipped"]) / total, 4)
            if total
            else 0.0
        ),
        "json_parse_success_rate": 0.0,
        "warning_rate": (
            round(warnings_total / status_counts["generated"], 4)
            if status_counts["generated"]
            else 0.0
        ),
        "llm_manual_count": llm_manual_count,
        "evidence_fallback_count": evidence_fallback_count,
        "deterministic_auto_count": deterministic_auto_count,
        "llm_manual_rate": (
            round(llm_manual_count / status_counts["generated"], 4)
            if status_counts["generated"]
            else 0.0
        ),
        "fallback_rate": (
            round(evidence_fallback_count / status_counts["generated"], 4)
            if status_counts["generated"]
            else 0.0
        ),
        "focus_section_missing_count": focus_section_missing_count,
        "focus_section_missing_rate": (
            round(focus_section_missing_count / status_counts["generated"], 4)
            if status_counts["generated"]
            else 0.0
        ),
        "critical_over_limit_rate": (
            round(critical_over_limit_count / status_counts["generated"], 4)
            if status_counts["generated"]
            else 0.0
        ),
        "cache_invalid_year_count": cache_invalid_year_count,
        "legacy_residual_total": legacy_residual_total,
        "failure_reasons": sorted(set(failure_reasons)),
        "failure_category_counts": failure_category_counts,
    }


async def async_main(args: argparse.Namespace) -> int:
    years = parse_years(args.years)
    league_type = parse_league_type(args.league_type)
    request_mode = parse_request_mode(args.mode)
    request_focus = parse_focus(args.focus, default=MATCHUP_FOCUS)
    target_order = parse_target_order(args.order)
    status_bucket_filter = parse_status_bucket_filter(args.status_bucket)
    cache_state_filter = parse_cache_state_filter(args.cache_state_filter)
    question_override = parse_question_override(args.question_override)

    if (
        league_type in POSTSEASON_LEAGUE_TYPES
        and not args.allow_postseason_stage_mismatch
    ):
        _ensure_postseason_stage_integrity(years)

    targets = load_targets(
        years=years,
        league_type=league_type,
        request_focus=request_focus,
        request_mode=request_mode,
        question_override=question_override,
        offset=max(0, args.offset),
        limit=args.limit,
        order=target_order,
        status_bucket_filter=status_bucket_filter,
    )
    failed_cache_keys = load_failed_cache_keys(args.retry_failures_from_report)
    if failed_cache_keys:
        targets = [
            target for target in targets if target.cache_key in failed_cache_keys
        ]
    targets = _filter_targets_by_cache_state(targets, cache_state_filter)

    if not targets:
        print("No matchup targets found.")
        return 0

    if args.verify_cache_only and args.force_rebuild:
        raise ValueError("force-rebuild cannot be combined with verify-cache-only")

    if args.force_rebuild:
        deleted = force_rebuild_delete([target.cache_key for target in targets])
        print(f"Force rebuild enabled. deleted_cache_rows={deleted}")

    start_time = datetime.now()
    results: List[Dict[str, Any]]
    if args.verify_cache_only:
        results = collect_cache_verification_results(targets)
        for idx, (target, item) in enumerate(zip(targets, results), start=1):
            print(
                f"[{idx}/{len(targets)}] {target.season_year} {target.home_team_id} vs {target.away_team_id} "
                f"-> {item['status']} ({item.get('reason')})"
            )
    else:
        timeout = httpx.Timeout(args.timeout, connect=min(10.0, args.timeout))
        default_headers: Dict[str, str] = {}
        if args.internal_api_key:
            default_headers["X-Internal-Api-Key"] = args.internal_api_key
        results = []
        async with httpx.AsyncClient(
            timeout=timeout, headers=default_headers
        ) as client:
            per_target_timeout = max(
                args.timeout + DONE_WAIT_SECONDS + TARGET_TIMEOUT_BUFFER_SECONDS,
                args.timeout,
            )
            for idx, target in enumerate(targets, start=1):
                item = await call_analyze_with_retries(
                    client=client,
                    base_url=args.base_url,
                    target=target,
                    timeout_seconds=per_target_timeout,
                    max_attempts=args.max_attempts,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                )
                results.append(item)
                print(
                    f"[{idx}/{len(targets)}] {target.season_year} {target.home_team_id} vs {target.away_team_id} "
                    f"-> {item['status']} ({item.get('reason')})"
                )
                if idx < len(targets):
                    await asyncio.sleep(max(0.0, args.delay_seconds))

    summary = summarize_matchup_results(
        results=results,
        years=years,
        league_type=league_type,
        focus=request_focus,
    )
    elapsed = datetime.now() - start_time
    summary["runtime_seconds"] = round(elapsed.total_seconds(), 3)
    report = {
        "summary": summary,
        "options": {
            "years": years,
            "league_type": league_type,
            "focus": request_focus,
            "mode": request_mode,
            "force_rebuild": args.force_rebuild,
            "max_attempts": args.max_attempts,
            "retry_backoff_seconds": args.retry_backoff_seconds,
            "offset": args.offset,
            "limit": args.limit,
            "order": target_order,
            "status_bucket": status_bucket_filter,
            "cache_state_filter": cache_state_filter,
            "question_override": question_override,
            "allow_postseason_stage_mismatch": args.allow_postseason_stage_mismatch,
            "retry_failures_from_report": args.retry_failures_from_report,
            "verify_cache_only": args.verify_cache_only,
        },
        "details": results,
        "run_started_at": start_time.isoformat(),
        "run_finished_at": datetime.now().isoformat(),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.quality_report:
        report_path = Path(args.quality_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Quality report written: {report_path}")
    return 0 if summary["failed"] == 0 and summary["in_progress"] == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coach matchup cache batch runner for Prediction page payloads."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
        help="Coach API base URL.",
    )
    parser.add_argument(
        "--internal-api-key",
        default=resolve_default_internal_api_key(WORKSPACE_ROOT),
        help="Value for X-Internal-Api-Key when calling protected AI endpoints.",
    )
    parser.add_argument(
        "--years",
        default="2025",
        help="Comma-separated season years (e.g. 2025 or 2024,2025).",
    )
    parser.add_argument(
        "--league-type",
        default="REGULAR",
        help="REGULAR, PRE, POST, or ANY (default: REGULAR).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Target offset after matchup dedupe.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max matchup targets to process.",
    )
    parser.add_argument(
        "--order",
        default="asc",
        choices=VALID_TARGET_ORDERS,
        help="Target order after dedupe: asc (oldest first) or desc (latest first).",
    )
    parser.add_argument(
        "--status-bucket",
        default="ANY",
        choices=VALID_STATUS_BUCKET_FILTERS,
        help="Filter targets by normalized game status bucket.",
    )
    parser.add_argument(
        "--cache-state-filter",
        default="ANY",
        choices=VALID_CACHE_STATE_FILTERS,
        help="Filter targets by cache row state: ANY, MISSING, COMPLETED, FAILED, PENDING, UNRESOLVED.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Delay between requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout seconds.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete selected matchup cache keys before replaying requests.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max attempts per target for retryable failures.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Base backoff seconds before retrying a retryable failure.",
    )
    parser.add_argument(
        "--mode",
        default="manual_detail",
        choices=VALID_REQUEST_MODES,
        help="Request mode for per-game prewarm. auto_brief is intentionally excluded.",
    )
    parser.add_argument(
        "--focus",
        help="Comma-separated focus list (e.g. recent_form,bullpen). Defaults by mode.",
    )
    parser.add_argument(
        "--question-override",
        help="Manual detail only: question override text for payload and cache key.",
    )
    parser.add_argument(
        "--quality-report",
        default=None,
        help="Output quality report JSON.",
    )
    parser.add_argument(
        "--retry-failures-from-report",
        default=None,
        help="Replay only failed cache keys from a prior quality report JSON.",
    )
    parser.add_argument(
        "--verify-cache-only",
        action="store_true",
        help="Skip coach API calls and verify target cache rows directly from the database.",
    )
    parser.add_argument(
        "--allow-postseason-stage-mismatch",
        action="store_true",
        help="Bypass postseason season_id/stage integrity preflight guard.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        return asyncio.run(async_main(args))
    except Exception as exc:
        logger.error("batch_coach_matchup_cache failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
