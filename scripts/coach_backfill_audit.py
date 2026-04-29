#!/usr/bin/env python3
"""Backfill Coach analysis cache and audit response quality.

The script intentionally uses only internal DB/readiness signals and the Coach
SSE API. It does not collect or repair baseball data from external sources.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo

os.environ.setdefault("BEGA_SKIP_APP_INIT", "1")

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "coach_backfill"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_REQUEST_INTERVAL_SECONDS = 2.6
MIN_VERIFY_CACHE_HIT_REQUEST_INTERVAL_SECONDS = 3.0
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRYABLE_SSE_ERROR_CODES = {
    "coach_request_error",
    "coach_request_timeout",
    "oci_mapping_degraded",
}
COMPLETED_FOCUS = ["recent_form", "bullpen", "starter", "matchup", "batting"]
SCHEDULED_FOCUS = ["recent_form", "bullpen", "starter"]
FOCUS_HEADERS = {
    "recent_form": "최근 전력",
    "bullpen": "불펜 상태",
    "starter": "선발 투수",
    "matchup": "상대 전적",
    "batting": "타격 생산성",
}
STATUS_BUCKETS = {"ANY", "COMPLETED", "LIVE", "SCHEDULED"}
OFFICIAL_GAME_ID_TEAM_CODES = {
    "KIA": "HT",
    "DB": "OB",
    "KH": "WO",
    "SSG": "SK",
}
OUTPUT_QUALITY_FIELDS = (
    "headline",
    "coach_note",
    "detailed_markdown",
    "key_metrics",
    "analysis",
)
BANNED_FRAGMENTS = (
    "선발 발표 상위 타선",
    "라인업 미발표 상위 타선 타순 기반 핵심 구간는",
    "상위 타선 기초 지표",
    "상위 타선 선취점",
    "핵심 구간는",
    "경기 경기 후반",
    "근소 우세지만 격차는 크지 않습니다",
)
SCHEDULED_OUTCOME_FRAGMENTS = (
    "## 결과 진단",
    "## 결과를 가른 이유",
    "## 실제 전환점",
    "## 다시 볼 장면",
    "결과를 가른",
    "우위 확정",
    "결승타",
    "끝내기",
    "승리,",
    "패배,",
)
SCHEDULED_CONFIRMED_STARTER_CONFLICT_FRAGMENTS = (
    "선발 정보가 확정되지",
    "선발 정보가 완전히 확정되지",
    "선발 맞대결 평가는 제한",
    "선발 미확정",
    "선발 미발표",
    "선발 대결 분석 불가능",
)
SCHEDULED_METRIC_JARGON_FRAGMENTS = (
    "WPA",
    "WPA/PA",
    "오프ensive",
    "offensive",
)
TRANSPORT_FAILURE_PREFIXES = (
    "HTTP 실패 status=0",
    "HTTP 실패 status=429",
    "SSE transport error 이벤트 감지:",
)
COMPLETED_PREDICTION_FRAGMENTS = (
    "경기 예측",
    "예상됩니다",
    "관전 포인트",
)
ROOT_CAUSE_LABELS = {
    "missing_starters": "선발 투수 정보 부족",
    "starter_announcement_pending": "공식 선발 발표 대기",
    "missing_lineups": "라인업 정보 부족",
    "missing_summary": "경기 요약 정보 부족",
    "missing_metadata": "구장/시간/날씨 메타데이터 부족",
    "missing_clutch_moments": "승부처 데이터 부족",
    "missing_game_context": "기본 경기 정보 부족",
    "missing_series_context": "포스트시즌 시리즈 맥락 부족",
    "focus_data_unavailable": "요청 focus 근거 부족",
}
NON_MISSING_RESPONSE_REASON_LABELS = {
    "empty_response": "LLM 빈 응답 후 보수 생성",
    "focus_data_unavailable": "요청 focus 직접 근거 제한",
    "json_decode_error": "LLM JSON 파싱 오류 후 보수 생성",
    "llm_parse_failed": "LLM 응답 파싱 실패 후 보수 생성",
    "no_json_found": "LLM JSON 미검출 후 보수 생성",
    "unsupported_entity_name": "근거 밖 엔티티 감지",
    "unsupported_entity_name_sanitized": "근거 밖 엔티티 안전 정리",
    "unsupported_numeric_claim": "근거 밖 수치 감지",
    "unsupported_numeric_claim_sanitized": "근거 밖 수치 안전 정리",
    "scheduled_output_guard_fallback": "예정 경기 리뷰/확정 표현 보수 생성 전환",
    "unconfirmed_lineup_claim": "라인업 미확정 단정 정리",
    "unconfirmed_series_claim": "시리즈 맥락 단정 정리",
    "unconfirmed_starter_claim": "선발 미확정 단정 정리",
}
LOCAL_TOKEN_ENV_FILES = (".env.prod", ".env")
MANUAL_BASEBALL_DATA_REQUIRED_CODE = "MANUAL_BASEBALL_DATA_REQUIRED"
MANUAL_REQUIRED_FIELDS_BY_MISSING_KEY: Dict[str, Sequence[str]] = {
    "game_id": ("game.game_id",),
    "game_date": ("game.game_date",),
    "season_league_context": (
        "game.season_id",
        "game.season_year",
        "game.league_type_code",
    ),
    "game_status": ("game.game_status",),
    "final_score": ("game.home_score", "game.away_score"),
    "starters": ("game.home_pitcher", "game.away_pitcher"),
    "lineup": (
        "game_lineups.team_code",
        "game_lineups.batting_order",
        "game_lineups.player_name",
    ),
}
STARTER_ANNOUNCEMENT_PENDING_CODE = "starter_announcement_pending"
STARTER_ANNOUNCEMENT_PENDING_LABEL = "공식 선발 발표 대기"
KBO_TIMEZONE = ZoneInfo("Asia/Seoul")
KBO_STARTER_ANNOUNCEMENT_HOUR = 18


def _read_env_file_entries(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    entries: Dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            normalized = value.strip()
            if (normalized.startswith('"') and normalized.endswith('"')) or (
                normalized.startswith("'") and normalized.endswith("'")
            ):
                normalized = normalized[1:-1]
            entries[key.strip()] = normalized
    except OSError:
        return {}
    return entries


def parse_game_ids(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    values: List[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        normalized = token.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return values


def _parse_game_date(value: Any) -> Optional[date]:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    try:
        return date.fromisoformat(normalized[:10])
    except ValueError:
        return None


def _now_kst() -> datetime:
    override = (os.getenv("COACH_AUDIT_NOW_KST", "") or "").strip()
    if override:
        parsed = datetime.fromisoformat(override)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=KBO_TIMEZONE)
        return parsed.astimezone(KBO_TIMEZONE)
    return datetime.now(KBO_TIMEZONE)


def starter_announcement_due_at_kst(game_date: Any) -> Optional[datetime]:
    parsed = _parse_game_date(game_date)
    if parsed is None:
        return None
    announcement_date = parsed - timedelta(days=1)
    return datetime.combine(
        announcement_date,
        dt_time(hour=KBO_STARTER_ANNOUNCEMENT_HOUR),
        tzinfo=KBO_TIMEZONE,
    )


def is_starter_announcement_pending(
    record: Dict[str, Any],
    *,
    now_kst: Optional[datetime] = None,
) -> bool:
    if str(record.get("game_status_bucket") or "").upper() != "SCHEDULED":
        return False
    root_causes = {str(code) for code in record.get("root_causes") or []}
    if STARTER_ANNOUNCEMENT_PENDING_CODE in root_causes:
        return True
    if "missing_starters" not in root_causes:
        return False
    due_at = starter_announcement_due_at_kst(record.get("game_date"))
    if due_at is None:
        return False
    current = now_kst or _now_kst()
    if current.tzinfo is None:
        current = current.replace(tzinfo=KBO_TIMEZONE)
    return current.astimezone(KBO_TIMEZONE) < due_at


def effective_request_interval_seconds(
    request_interval_seconds: float,
    *,
    verify_cache_hit: bool,
) -> float:
    """Clamp cache-hit verification to a rate-limit-safe interval."""

    normalized = max(0.0, float(request_interval_seconds or 0.0))
    if verify_cache_hit:
        return max(normalized, MIN_VERIFY_CACHE_HIT_REQUEST_INTERVAL_SECONDS)
    return normalized


def resolve_default_internal_api_key() -> str:
    env_value = (os.getenv("AI_INTERNAL_TOKEN", "") or "").strip()
    if env_value:
        return env_value
    workspace_root = PROJECT_ROOT.parent
    token_paths = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / ".env.prod",
        *(workspace_root / filename for filename in LOCAL_TOKEN_ENV_FILES),
    ]
    for path in token_paths:
        token = _read_env_file_entries(path).get("AI_INTERNAL_TOKEN", "")
        if token.strip():
            return token.strip()
    return ""


def _sorted_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: (
            str(item.get("game_date") or ""),
            str(item.get("game_id") or ""),
        ),
    )


def _official_game_id_team_code(team_code: Any) -> str:
    normalized = str(team_code or "").strip().upper()
    return OFFICIAL_GAME_ID_TEAM_CODES.get(normalized, normalized)


def _game_id_date_prefix(record: Dict[str, Any]) -> str:
    game_date = str(record.get("game_date") or "").replace("-", "")
    if len(game_date) == 8 and game_date.isdigit():
        return game_date
    game_id = str(record.get("game_id") or "")
    return game_id[:8]


def _preferred_game_id_prefix(record: Dict[str, Any]) -> str:
    return (
        _game_id_date_prefix(record)
        + _official_game_id_team_code(record.get("away_team_id"))
        + _official_game_id_team_code(record.get("home_team_id"))
    )


def _is_preferred_game_id(record: Dict[str, Any]) -> bool:
    game_id = str(record.get("game_id") or "").strip().upper()
    prefix = _preferred_game_id_prefix(record).upper()
    return bool(game_id and prefix and game_id.startswith(prefix))


def _game_id_suffix_bucket(record: Dict[str, Any]) -> str:
    game_id = str(record.get("game_id") or "").strip().upper()
    if game_id and game_id[-1].isdigit():
        return game_id[-1]
    return ""


def dedupe_official_game_ids(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for record in records:
        key = (
            _game_id_date_prefix(record),
            str(record.get("away_team_id") or ""),
            str(record.get("home_team_id") or ""),
            str(record.get("stage_label") or ""),
            _game_id_suffix_bucket(record),
        )
        grouped.setdefault(key, []).append(record)

    deduped: List[Dict[str, Any]] = []
    for group in grouped.values():
        preferred = [item for item in group if _is_preferred_game_id(item)]
        if preferred:
            deduped.append(_sorted_records(preferred)[0])
        else:
            deduped.append(_sorted_records(group)[0])
    return _sorted_records(deduped)


def _diagnose_games(
    *,
    season_year: int,
    date_from: Optional[str],
    date_to: Optional[str],
    game_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    from scripts.coach_grounding_audit import diagnose_games

    return diagnose_games(
        season_year=season_year,
        date_from=date_from,
        date_to=date_to,
        game_ids=game_ids,
        max_games=0,
    )


def _detect_unconfirmed_data_claims(
    record: Dict[str, Any],
    structured_response: Dict[str, Any],
) -> List[str]:
    try:
        from scripts.coach_grounding_audit import detect_unconfirmed_data_claims
    except Exception:
        return []
    return detect_unconfirmed_data_claims(record, structured_response)


@dataclass(frozen=True)
class BackfillCapture:
    status_code: int
    elapsed_seconds: float
    response_headers: Dict[str, str]
    done_seen: bool
    event_sequence: List[str]
    message_text: str
    meta: Dict[str, Any]
    structured_response: Optional[Dict[str, Any]]
    error_payload: Any
    sample_response: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Coach manual_detail responses and audit output quality."
    )
    parser.add_argument("--season-year", type=int, default=datetime.now().year)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--game-ids", default=None)
    parser.add_argument("--status-bucket", default="ANY", choices=sorted(STATUS_BUCKETS))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--order", default="asc", choices=("asc", "desc"))
    parser.add_argument(
        "--include-noncanonical-game-ids",
        action="store_true",
        help="Keep duplicate alias game IDs instead of preferring official KBO game IDs.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--internal-api-key",
        default=resolve_default_internal_api_key(),
        help="X-Internal-Api-Key for direct AI Coach calls.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Relative paths resolve under bega_AI.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=DEFAULT_REQUEST_INTERVAL_SECONDS,
        help=(
            "Minimum interval between Coach HTTP requests. "
            "When --verify-cache-hit is used, values below 3.0 are clamped."
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only diagnose target readiness; do not call Coach.",
    )
    parser.add_argument(
        "--verify-cache-hit",
        action="store_true",
        help="Call each target twice and require the second response to be cache HIT.",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Exit with code 1 when hard quality failures are found.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Always exit 0 unless the script crashes.",
    )
    return parser.parse_args(argv)


def resolve_output_dir(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def select_records(
    records: Sequence[Dict[str, Any]],
    *,
    status_bucket: str,
    offset: int,
    limit: Optional[int],
    order: str,
    include_noncanonical_game_ids: bool = False,
) -> List[Dict[str, Any]]:
    selected = [
        dict(item)
        for item in records
        if status_bucket == "ANY"
        or str(item.get("game_status_bucket") or "").upper() == status_bucket
    ]
    if not include_noncanonical_game_ids:
        selected = dedupe_official_game_ids(selected)
    selected = _sorted_records(selected)
    if order == "desc":
        selected = list(reversed(selected))
    start = max(0, int(offset or 0))
    selected = selected[start:]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


def focus_for_record(record: Dict[str, Any]) -> List[str]:
    bucket = str(record.get("game_status_bucket") or "").upper()
    if bucket == "COMPLETED":
        return list(COMPLETED_FOCUS)
    return list(SCHEDULED_FOCUS)


def build_request_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    stage_label = str(record.get("stage_label") or "").strip() or None
    league_context: Dict[str, Any] = {
        "season": record.get("season_id"),
        "season_year": record.get("season_year"),
        "game_date": record.get("game_date"),
        "league_type_code": record.get("league_type_code"),
        "stage_label": stage_label,
        "round": stage_label,
        "lineup_announced": bool(record.get("lineup_announced")),
    }
    if record.get("series_game_no") is not None:
        league_context["game_no"] = record.get("series_game_no")
        league_context["series_game_no"] = record.get("series_game_no")

    return {
        "home_team_id": record.get("home_team_id"),
        "away_team_id": record.get("away_team_id"),
        "game_id": record.get("game_id"),
        "request_mode": "manual_detail",
        "focus": focus_for_record(record),
        "league_context": league_context,
    }


def _safe_json_loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_sse_stream(response: httpx.Response, elapsed_seconds: float) -> BackfillCapture:
    current_event = "message"
    event_sequence: List[str] = []
    message_chunks: List[str] = []
    sample_lines: List[str] = []
    meta_payload: Dict[str, Any] = {}
    error_payload: Any = None
    done_seen = False

    for raw_line in response.iter_lines():
        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
        if not line:
            current_event = "message"
            continue
        sample_lines.append(line)
        if len(sample_lines) > 60:
            sample_lines = sample_lines[-60:]
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip().lower()
            event_sequence.append(current_event)
            if current_event == "done":
                done_seen = True
            continue
        if not line.startswith("data:"):
            continue

        data_str = line.split(":", 1)[1].strip()
        if not data_str:
            continue
        if data_str == "[DONE]":
            done_seen = True
            continue
        parsed = _safe_json_loads(data_str)
        if current_event == "message":
            if isinstance(parsed, dict) and isinstance(parsed.get("delta"), str):
                message_chunks.append(parsed["delta"])
            else:
                message_chunks.append(str(parsed))
        elif current_event == "meta" and isinstance(parsed, dict):
            meta_payload = parsed
        elif current_event == "error":
            error_payload = parsed

    structured_response = meta_payload.get("structured_response")
    if not isinstance(structured_response, dict):
        parsed_message = _safe_json_loads("".join(message_chunks).strip())
        structured_response = parsed_message if isinstance(parsed_message, dict) else None

    return BackfillCapture(
        status_code=response.status_code,
        elapsed_seconds=elapsed_seconds,
        response_headers=dict(response.headers),
        done_seen=done_seen,
        event_sequence=event_sequence,
        message_text="".join(message_chunks),
        meta=meta_payload,
        structured_response=structured_response,
        error_payload=error_payload,
        sample_response="\n".join(sample_lines),
    )


def call_coach(
    client: httpx.Client,
    *,
    base_url: str,
    payload: Dict[str, Any],
    internal_api_key: str,
) -> BackfillCapture:
    headers = {"Accept": "text/event-stream"}
    if internal_api_key:
        headers["X-Internal-Api-Key"] = internal_api_key
    started = time.monotonic()
    try:
        with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/coach/analyze",
            json=payload,
            headers=headers,
        ) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", errors="replace")
                return BackfillCapture(
                    status_code=response.status_code,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    response_headers=dict(response.headers),
                    done_seen=False,
                    event_sequence=[],
                    message_text="",
                    meta={},
                    structured_response=None,
                    error_payload=body,
                    sample_response=body[:800],
                )
            return parse_sse_stream(
                response,
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
    except httpx.TimeoutException as exc:
        return BackfillCapture(
            status_code=0,
            elapsed_seconds=round(time.monotonic() - started, 3),
            response_headers={},
            done_seen=False,
            event_sequence=[],
            message_text="",
            meta={},
            structured_response=None,
            error_payload=f"coach_request_timeout: {exc}",
            sample_response=str(exc),
        )
    except httpx.HTTPError as exc:
        return BackfillCapture(
            status_code=0,
            elapsed_seconds=round(time.monotonic() - started, 3),
            response_headers={},
            done_seen=False,
            event_sequence=[],
            message_text="",
            meta={},
            structured_response=None,
            error_payload=f"coach_request_error: {exc}",
            sample_response=str(exc),
        )


def _sleep_for_request_interval(state: Dict[str, float], interval_seconds: float) -> None:
    interval = max(0.0, float(interval_seconds or 0.0))
    if interval <= 0:
        state["last_request_at"] = time.monotonic()
        return
    now = time.monotonic()
    last_request_at = state.get("last_request_at")
    if last_request_at is not None:
        wait_seconds = interval - (now - last_request_at)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
    state["last_request_at"] = time.monotonic()


def _retry_delay_seconds(
    capture: BackfillCapture,
    *,
    retry_backoff_seconds: float,
    attempt: int,
) -> float:
    retry_after = capture.response_headers.get("retry-after", "")
    if retry_after:
        try:
            return max(float(retry_after), retry_backoff_seconds)
        except ValueError:
            pass
    return max(retry_backoff_seconds, retry_backoff_seconds * attempt)


def _capture_error_code(capture: Optional[BackfillCapture]) -> str:
    if capture is None:
        return ""
    payload = capture.error_payload
    if isinstance(payload, dict):
        return str(payload.get("code") or "").strip().lower()
    normalized = str(payload or "").strip().lower()
    for code in RETRYABLE_SSE_ERROR_CODES:
        if code in normalized:
            return code
    meta = capture.meta if isinstance(capture.meta, dict) else {}
    return str(meta.get("cache_error_code") or "").strip().lower()


def is_retryable_capture(capture: Optional[BackfillCapture]) -> bool:
    if capture is None:
        return False
    if capture.status_code in RETRYABLE_STATUS_CODES:
        return True
    if capture.status_code != 200:
        return False

    meta = capture.meta if isinstance(capture.meta, dict) else {}
    cache_state = str(meta.get("cache_state") or "").strip().upper()
    error_code = _capture_error_code(capture)
    if error_code in RETRYABLE_SSE_ERROR_CODES:
        return True
    if cache_state in {"FAILED_RETRY_WAIT", "PENDING_WAIT"}:
        return bool(meta.get("retryable_failure") or cache_state == "PENDING_WAIT")
    return False


def call_coach_with_retries(
    client: httpx.Client,
    *,
    base_url: str,
    payload: Dict[str, Any],
    internal_api_key: str,
    request_interval_state: Dict[str, float],
    request_interval_seconds: float,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> BackfillCapture:
    attempts = max(1, int(max_attempts or 1))
    last_capture: Optional[BackfillCapture] = None
    for attempt in range(1, attempts + 1):
        _sleep_for_request_interval(
            request_interval_state,
            request_interval_seconds,
        )
        last_capture = call_coach(
            client,
            base_url=base_url,
            payload=payload,
            internal_api_key=internal_api_key,
        )
        if not is_retryable_capture(last_capture):
            return last_capture
        if attempt < attempts:
            time.sleep(
                _retry_delay_seconds(
                    last_capture,
                    retry_backoff_seconds=retry_backoff_seconds,
                    attempt=attempt,
                )
            )
    return last_capture or BackfillCapture(
        status_code=0,
        elapsed_seconds=0.0,
        response_headers={},
        done_seen=False,
        event_sequence=[],
        message_text="",
        meta={},
        structured_response=None,
        error_payload="coach_request_not_attempted",
        sample_response="coach_request_not_attempted",
    )


def _text_blob(structured_response: Optional[Dict[str, Any]]) -> str:
    if not isinstance(structured_response, dict):
        return ""
    return json.dumps(structured_response, ensure_ascii=False)


def _dedupe_messages(messages: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        deduped.append(message)
    return deduped


def is_transport_failure_message(message: str) -> bool:
    normalized = str(message or "")
    return any(
        normalized.startswith(prefix) for prefix in TRANSPORT_FAILURE_PREFIXES
    )


def _failure_category(message: str) -> str:
    return "transport" if is_transport_failure_message(message) else "content"


def _markdown(structured_response: Optional[Dict[str, Any]]) -> str:
    if not isinstance(structured_response, dict):
        return ""
    return str(structured_response.get("detailed_markdown") or "")


def _has_section(markdown: str, header: str) -> bool:
    return f"## {header}" in markdown


def collect_missing_data(
    record: Dict[str, Any],
    meta: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    starter_pending = is_starter_announcement_pending(record)
    for code in record.get("root_causes") or []:
        normalized_code = str(code)
        if normalized_code == "missing_starters" and starter_pending:
            normalized_code = STARTER_ANNOUNCEMENT_PENDING_CODE
        rows.append(
            {
                "source": "diagnosis",
                "code": normalized_code,
                "label": ROOT_CAUSE_LABELS.get(normalized_code, normalized_code),
            }
        )
    if isinstance(meta, dict):
        for code in meta.get("grounding_reasons") or []:
            normalized_code = str(code)
            if normalized_code == "missing_starters" and starter_pending:
                normalized_code = STARTER_ANNOUNCEMENT_PENDING_CODE
            if (
                not normalized_code.startswith("missing_")
                and normalized_code != STARTER_ANNOUNCEMENT_PENDING_CODE
            ):
                continue
            rows.append(
                {
                    "source": "response",
                    "code": normalized_code,
                    "label": ROOT_CAUSE_LABELS.get(normalized_code, normalized_code),
                }
            )
    deduped: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["source"], row["code"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def collect_response_notes(meta: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not isinstance(meta, dict):
        return []

    rows: List[Dict[str, str]] = []
    for code in meta.get("grounding_reasons") or []:
        normalized_code = str(code)
        if normalized_code.startswith("missing_"):
            continue
        rows.append(
            {
                "source": "response",
                "code": normalized_code,
                "label": NON_MISSING_RESPONSE_REASON_LABELS.get(
                    normalized_code,
                    ROOT_CAUSE_LABELS.get(normalized_code, normalized_code),
                ),
            }
        )

    deduped: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["source"], row["code"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def assess_quality(
    record: Dict[str, Any],
    capture: Optional[BackfillCapture],
    *,
    cache_hit_probe: bool = False,
) -> tuple[List[str], List[str]]:
    hard_failures: List[str] = []
    soft_warnings: List[str] = []
    if capture is None:
        return hard_failures, soft_warnings

    if capture.status_code != 200:
        hard_failures.append(f"HTTP 실패 status={capture.status_code}")
        return hard_failures, soft_warnings
    if not capture.done_seen:
        hard_failures.append("SSE done 누락")
    if capture.error_payload:
        if is_retryable_capture(capture):
            hard_failures.append(
                f"SSE transport error 이벤트 감지: {capture.error_payload}"
            )
            return hard_failures, soft_warnings
        hard_failures.append(f"SSE error 이벤트 감지: {capture.error_payload}")
    if not isinstance(capture.meta, dict) or not capture.meta:
        hard_failures.append("meta 누락")
    if not isinstance(capture.structured_response, dict):
        hard_failures.append("structured_response 누락")
        return hard_failures, soft_warnings

    meta = capture.meta
    structured = capture.structured_response
    markdown = _markdown(structured)
    blob = _text_blob(structured)
    bucket = str(record.get("game_status_bucket") or "").upper()
    response_bucket = str(meta.get("game_status_bucket") or "").upper()

    if bucket in {"COMPLETED", "SCHEDULED"} and response_bucket != bucket:
        hard_failures.append(
            f"응답 경기 상태 불일치: expected={bucket} actual={response_bucket or 'missing'}"
        )

    for field_name in OUTPUT_QUALITY_FIELDS:
        value = structured.get(field_name)
        if value in (None, "", [], {}):
            soft_warnings.append(f"structured_response.{field_name} 비어 있음")

    for focus in focus_for_record(record):
        header = FOCUS_HEADERS.get(focus)
        if header and not _has_section(markdown, header):
            hard_failures.append(f"focus 섹션 누락: {header}")

    if meta.get("focus_section_missing"):
        hard_failures.append(
            "focus_section_missing=true "
            f"missing={meta.get('missing_focus_sections') or []}"
        )

    data_quality = str(meta.get("data_quality") or "")
    if data_quality == "insufficient":
        hard_failures.append("data_quality=insufficient")
    if (
        record.get("expected_data_quality") == "grounded"
        and data_quality == "partial"
    ):
        soft_warnings.append("진단은 grounded이나 응답은 partial")

    cache_state = str(meta.get("cache_state") or "").upper()
    if cache_hit_probe and cache_state != "HIT":
        hard_failures.append(f"재호출 cache_state가 HIT가 아님: {cache_state or 'missing'}")

    for fragment in BANNED_FRAGMENTS:
        if fragment in blob:
            hard_failures.append(f"금지 문구 포함: {fragment}")

    if bucket == "SCHEDULED":
        for fragment in SCHEDULED_OUTCOME_FRAGMENTS:
            if fragment in blob:
                hard_failures.append(f"예정 경기 결과 확정 표현 포함: {fragment}")
        for fragment in SCHEDULED_METRIC_JARGON_FRAGMENTS:
            if fragment in blob:
                hard_failures.append(f"예정 경기 사용자 문구에 지표/언어 잔여 포함: {fragment}")
        if record.get("home_pitcher_present") and record.get("away_pitcher_present"):
            for fragment in SCHEDULED_CONFIRMED_STARTER_CONFLICT_FRAGMENTS:
                if fragment in blob:
                    hard_failures.append(
                        f"예정 경기 선발 확정 상태와 충돌: {fragment}"
                    )
    elif bucket == "COMPLETED":
        for fragment in COMPLETED_PREDICTION_FRAGMENTS:
            if fragment in blob:
                soft_warnings.append(f"완료 경기 예측 톤 의심 문구 포함: {fragment}")

    unconfirmed_claims = _detect_unconfirmed_data_claims(record, structured)
    if unconfirmed_claims:
        target = (
            hard_failures
            if "missing_lineups" in set(record.get("root_causes") or [])
            else soft_warnings
        )
        target.extend(f"미확정 데이터 단정 의심: {item}" for item in unconfirmed_claims)

    return hard_failures, soft_warnings


def _capture_to_dict(capture: Optional[BackfillCapture]) -> Optional[Dict[str, Any]]:
    if capture is None:
        return None
    return {
        "status_code": capture.status_code,
        "elapsed_seconds": capture.elapsed_seconds,
        "response_headers": capture.response_headers,
        "done_seen": capture.done_seen,
        "event_sequence": capture.event_sequence,
        "meta": capture.meta,
        "structured_response": capture.structured_response,
        "error_payload": capture.error_payload,
        "sample_response": capture.sample_response,
    }


def run_backfill(
    *,
    records: Sequence[Dict[str, Any]],
    base_url: str,
    internal_api_key: str,
    timeout_seconds: float,
    delay_seconds: float,
    request_interval_seconds: float,
    max_attempts: int,
    retry_backoff_seconds: float,
    dry_run: bool,
    verify_cache_hit: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    timeout = httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds))
    request_interval_state: Dict[str, float] = {}
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for index, record in enumerate(records, start=1):
            payload = build_request_payload(record)
            capture: Optional[BackfillCapture] = None
            cache_capture: Optional[BackfillCapture] = None
            if not dry_run:
                capture = call_coach_with_retries(
                    client,
                    base_url=base_url,
                    payload=payload,
                    internal_api_key=internal_api_key,
                    request_interval_state=request_interval_state,
                    request_interval_seconds=request_interval_seconds,
                    max_attempts=max_attempts,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                if (
                    verify_cache_hit
                    and capture.status_code == 200
                    and not capture.error_payload
                ):
                    cache_capture = call_coach_with_retries(
                        client,
                        base_url=base_url,
                        payload=payload,
                        internal_api_key=internal_api_key,
                        request_interval_state=request_interval_state,
                        request_interval_seconds=request_interval_seconds,
                        max_attempts=max_attempts,
                        retry_backoff_seconds=retry_backoff_seconds,
                    )

            hard_failures, soft_warnings = assess_quality(record, capture)
            cache_failures, cache_warnings = assess_quality(
                record,
                cache_capture,
                cache_hit_probe=True,
            )
            hard_failures = _dedupe_messages([*hard_failures, *cache_failures])
            soft_warnings = _dedupe_messages([*soft_warnings, *cache_warnings])

            meta = capture.meta if capture else {}
            structured = capture.structured_response if capture else None
            result = {
                "index": index,
                "target": {
                    "game_id": record.get("game_id"),
                    "game_date": record.get("game_date"),
                    "game_status_bucket": record.get("game_status_bucket"),
                    "home_team_id": record.get("home_team_id"),
                    "away_team_id": record.get("away_team_id"),
                    "stage_label": record.get("stage_label"),
                    "expected_data_quality": record.get("expected_data_quality"),
                },
                "diagnosis": record,
                "request": payload,
                "response": _capture_to_dict(capture),
                "cache_probe_response": _capture_to_dict(cache_capture),
                "missing_data": collect_missing_data(record, meta),
                "response_notes": collect_response_notes(meta),
                "quality": {
                    "hard_failures": hard_failures,
                    "soft_warnings": soft_warnings,
                    "ok": not hard_failures,
                },
                "output_summary": {
                    "headline": (
                        structured.get("headline")
                        if isinstance(structured, dict)
                        else None
                    ),
                    "detailed_markdown_len": len(_markdown(structured)),
                },
            }
            results.append(result)

            print(
                "[{}/{}] {} {}@{} status={} quality_ok={} cache_state={} cache_probe_state={}".format(
                    index,
                    len(records),
                    record.get("game_id"),
                    record.get("away_team_id"),
                    record.get("home_team_id"),
                    record.get("game_status_bucket"),
                    not hard_failures,
                    meta.get("cache_state") if isinstance(meta, dict) else None,
                    (
                        cache_capture.meta.get("cache_state")
                        if cache_capture and isinstance(cache_capture.meta, dict)
                        else None
                    ),
                )
            )
            if index < len(records) and not dry_run:
                time.sleep(max(0.0, delay_seconds))
    return results


def summarize_results(
    records: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    diagnosis_quality = Counter(
        str(item.get("expected_data_quality") or "unknown") for item in records
    )
    response_quality = Counter(
        str(((item.get("response") or {}).get("meta") or {}).get("data_quality") or "none")
        for item in results
    )
    cache_states = Counter(
        str(((item.get("response") or {}).get("meta") or {}).get("cache_state") or "none")
        for item in results
    )
    cache_probe_states = Counter(
        str(
            ((item.get("cache_probe_response") or {}).get("meta") or {}).get(
                "cache_state"
            )
            or "none"
        )
        for item in results
    )
    missing_codes = Counter(
        row["code"] for item in results for row in item.get("missing_data") or []
    )
    missing_codes_by_source = Counter(
        f"{row['source']}:{row['code']}"
        for item in results
        for row in item.get("missing_data") or []
    )
    response_notes = Counter(
        row["code"] for item in results for row in item.get("response_notes") or []
    )
    response_notes_by_source = Counter(
        f"{row['source']}:{row['code']}"
        for item in results
        for row in item.get("response_notes") or []
    )
    manual_required_rows = _manual_baseball_data_required_rows(results)
    manual_required_count = len(
        {row.get("game_id") for row in manual_required_rows if row.get("game_id")}
    )
    starter_announcement_pending_count = len(
        _starter_announcement_pending_rows(results)
    )
    hard_failure_count = sum(
        len((item.get("quality") or {}).get("hard_failures") or [])
        for item in results
    )
    transport_failure_count = sum(
        1
        for item in results
        for message in (item.get("quality") or {}).get("hard_failures") or []
        if is_transport_failure_message(message)
    )
    content_hard_failure_count = hard_failure_count - transport_failure_count
    transport_failed_targets = sum(
        1
        for item in results
        if any(
            is_transport_failure_message(message)
            for message in (item.get("quality") or {}).get("hard_failures") or []
        )
    )
    content_failed_targets = sum(
        1
        for item in results
        if any(
            not is_transport_failure_message(message)
            for message in (item.get("quality") or {}).get("hard_failures") or []
        )
    )
    soft_warning_count = sum(
        len((item.get("quality") or {}).get("soft_warnings") or [])
        for item in results
    )
    return {
        "total_targets": len(records),
        "processed_targets": len(results),
        "passed_targets": sum(1 for item in results if item["quality"]["ok"]),
        "failed_targets": sum(1 for item in results if not item["quality"]["ok"]),
        "hard_failure_count": hard_failure_count,
        "content_passed_targets": len(results) - content_failed_targets,
        "content_failed_targets": content_failed_targets,
        "content_hard_failure_count": content_hard_failure_count,
        "transport_failed_targets": transport_failed_targets,
        "transport_failure_count": transport_failure_count,
        "soft_warning_count": soft_warning_count,
        "diagnosis_quality_distribution": dict(sorted(diagnosis_quality.items())),
        "response_quality_distribution": dict(sorted(response_quality.items())),
        "cache_state_distribution": dict(sorted(cache_states.items())),
        "cache_probe_state_distribution": dict(sorted(cache_probe_states.items())),
        "missing_data_distribution": dict(sorted(missing_codes.items())),
        "missing_data_distribution_by_source": dict(
            sorted(missing_codes_by_source.items())
        ),
        "response_note_distribution": dict(sorted(response_notes.items())),
        "response_note_distribution_by_source": dict(
            sorted(response_notes_by_source.items())
        ),
        "manual_baseball_data_required_count": manual_required_count,
        "manual_baseball_data_required_row_count": len(manual_required_rows),
        "starter_announcement_pending_count": starter_announcement_pending_count,
    }


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_summary_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "game_id",
        "game_date",
        "status_bucket",
        "matchup",
        "expected_data_quality",
        "data_quality",
        "generation_mode",
        "cache_state",
        "cached",
        "cache_probe_state",
        "cache_probe_cached",
        "focus_section_missing",
        "missing_focus_sections",
        "hard_failure_count",
        "soft_warning_count",
        "quality_ok",
        "missing_data_codes",
        "response_note_codes",
        "headline",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            target = item["target"]
            response = item.get("response") or {}
            meta = response.get("meta") or {}
            cache_probe = item.get("cache_probe_response") or {}
            cache_probe_meta = cache_probe.get("meta") or {}
            writer.writerow(
                {
                    "game_id": target.get("game_id"),
                    "game_date": target.get("game_date"),
                    "status_bucket": target.get("game_status_bucket"),
                    "matchup": f"{target.get('away_team_id')}@{target.get('home_team_id')}",
                    "expected_data_quality": target.get("expected_data_quality"),
                    "data_quality": meta.get("data_quality"),
                    "generation_mode": meta.get("generation_mode"),
                    "cache_state": meta.get("cache_state"),
                    "cached": meta.get("cached"),
                    "cache_probe_state": cache_probe_meta.get("cache_state"),
                    "cache_probe_cached": cache_probe_meta.get("cached"),
                    "focus_section_missing": meta.get("focus_section_missing"),
                    "missing_focus_sections": "|".join(
                        str(value) for value in meta.get("missing_focus_sections") or []
                    ),
                    "hard_failure_count": len(item["quality"]["hard_failures"]),
                    "soft_warning_count": len(item["quality"]["soft_warnings"]),
                    "quality_ok": item["quality"]["ok"],
                    "missing_data_codes": "|".join(
                        row["code"] for row in item.get("missing_data") or []
                    ),
                    "response_note_codes": "|".join(
                        row["code"] for row in item.get("response_notes") or []
                    ),
                    "headline": item.get("output_summary", {}).get("headline"),
                }
            )


def _write_missing_data_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    fields = ["game_id", "game_date", "status_bucket", "source", "code", "label"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            target = item["target"]
            for row in item.get("missing_data") or []:
                writer.writerow(
                    {
                        "game_id": target.get("game_id"),
                        "game_date": target.get("game_date"),
                        "status_bucket": target.get("game_status_bucket"),
                        "source": row.get("source"),
                        "code": row.get("code"),
                        "label": row.get("label"),
                    }
                )


def _write_response_notes_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    fields = ["game_id", "game_date", "status_bucket", "source", "code", "label"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            target = item["target"]
            for row in item.get("response_notes") or []:
                writer.writerow(
                    {
                        "game_id": target.get("game_id"),
                        "game_date": target.get("game_date"),
                        "status_bucket": target.get("game_status_bucket"),
                        "source": row.get("source"),
                        "code": row.get("code"),
                        "label": row.get("label"),
                    }
                )


def _manual_baseball_data_required_rows(
    results: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    def manual_requests(item: Dict[str, Any]) -> List[Dict[str, Any]]:
        requests: List[Dict[str, Any]] = []
        for capture_key in ("response", "cache_probe_response"):
            capture = item.get(capture_key) or {}
            meta = capture.get("meta") or {}
            if isinstance(meta, dict):
                request = meta.get("manual_data_request")
                if isinstance(request, dict):
                    requests.append(request)
            error_payload = capture.get("error_payload")
            if isinstance(error_payload, dict):
                request = error_payload.get("manual_data_request")
                if isinstance(request, dict):
                    requests.append(request)
                if error_payload.get("code") == MANUAL_BASEBALL_DATA_REQUIRED_CODE:
                    requests.append(error_payload)
        return requests

    def required_fields_for_missing_item(item: Dict[str, Any]) -> List[str]:
        key = str(item.get("key") or "").strip()
        mapped = MANUAL_REQUIRED_FIELDS_BY_MISSING_KEY.get(key)
        if mapped:
            return [str(field) for field in mapped]
        expected = str(item.get("expected_format") or "").strip()
        if expected:
            return [part.strip() for part in expected.split(",") if part.strip()]
        return [key] if key else []

    rows: List[Dict[str, str]] = []
    seen_rows: set[tuple[str, str, str]] = set()
    manual_keys_by_game: Dict[str, set[str]] = {}

    for item in results:
        target = item["target"]
        target_game_id = str(target.get("game_id") or "")
        for request in manual_requests(item):
            if request.get("code") != MANUAL_BASEBALL_DATA_REQUIRED_CODE:
                continue
            missing_items = [
                missing_item
                for missing_item in request.get("missingItems") or []
                if isinstance(missing_item, dict)
            ]
            missing_keys = [
                str(missing_item.get("key") or "").strip()
                for missing_item in missing_items
                if str(missing_item.get("key") or "").strip()
            ]
            if not missing_keys:
                missing_keys = ["manual_data_required"]
            manual_keys_by_game.setdefault(target_game_id, set()).update(missing_keys)

            required_fields: List[str] = []
            for missing_item in missing_items:
                for field in required_fields_for_missing_item(missing_item):
                    if field and field not in required_fields:
                        required_fields.append(field)
            if not required_fields:
                required_fields = ["operator.provided_baseball_data"]

            missing_code = "|".join(missing_keys)
            required_fields_text = "|".join(required_fields)
            row_key = (target_game_id, missing_code, required_fields_text)
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            rows.append(
                {
                    "game_id": target_game_id,
                    "game_date": str(target.get("game_date") or ""),
                    "status_bucket": str(target.get("game_status_bucket") or ""),
                    "away_team_id": str(target.get("away_team_id") or ""),
                    "home_team_id": str(target.get("home_team_id") or ""),
                    "contract_code": MANUAL_BASEBALL_DATA_REQUIRED_CODE,
                    "missing_code": missing_code,
                    "required_fields": required_fields_text,
                    "home_pitcher": "",
                    "away_pitcher": "",
                    "operator_message": str(request.get("operatorMessage") or ""),
                }
            )

    for item in results:
        target = item["target"]
        diagnosis = item.get("diagnosis") or {}
        target_game_id = str(target.get("game_id") or "")
        missing_codes = {
            str(row.get("code") or "")
            for row in item.get("missing_data") or []
        }
        missing_codes.update(str(code) for code in diagnosis.get("root_causes") or [])
        if "missing_starters" not in missing_codes:
            continue
        if "starters" in manual_keys_by_game.get(target_game_id, set()):
            continue
        if is_starter_announcement_pending(diagnosis):
            continue

        required_fields: List[str] = []
        if not diagnosis.get("home_pitcher_present"):
            required_fields.append("game.home_pitcher")
        if not diagnosis.get("away_pitcher_present"):
            required_fields.append("game.away_pitcher")
        if not required_fields:
            required_fields = ["game.home_pitcher", "game.away_pitcher"]

        row_key = (
            target_game_id,
            "missing_starters",
            "|".join(required_fields),
        )
        if row_key in seen_rows:
            continue
        seen_rows.add(row_key)
        rows.append(
            {
                "game_id": target_game_id,
                "game_date": str(target.get("game_date") or ""),
                "status_bucket": str(target.get("game_status_bucket") or ""),
                "away_team_id": str(target.get("away_team_id") or ""),
                "home_team_id": str(target.get("home_team_id") or ""),
                "contract_code": MANUAL_BASEBALL_DATA_REQUIRED_CODE,
                "missing_code": "missing_starters",
                "required_fields": "|".join(required_fields),
                "home_pitcher": "",
                "away_pitcher": "",
                "operator_message": (
                    "다음 야구 데이터가 필요합니다: 선발 투수 정보 "
                    f"({', '.join(required_fields)})"
                ),
            }
        )
    return rows


def _starter_announcement_pending_rows(
    results: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in results:
        target = item["target"]
        diagnosis = item.get("diagnosis") or {}
        if not is_starter_announcement_pending(diagnosis):
            continue
        due_at = starter_announcement_due_at_kst(diagnosis.get("game_date"))
        rows.append(
            {
                "game_id": str(target.get("game_id") or ""),
                "game_date": str(target.get("game_date") or ""),
                "status_bucket": str(target.get("game_status_bucket") or ""),
                "away_team_id": str(target.get("away_team_id") or ""),
                "home_team_id": str(target.get("home_team_id") or ""),
                "status_code": STARTER_ANNOUNCEMENT_PENDING_CODE,
                "expected_announcement_at_kst": (
                    due_at.isoformat(timespec="minutes") if due_at else ""
                ),
                "operator_message": (
                    "공식 선발 발표 전입니다. 발표 예정 시각 이후 내부 동기화가 "
                    "반영되는지 다시 확인하세요."
                ),
            }
        )
    return rows


def _write_manual_baseball_data_required_csv(
    path: Path,
    results: Sequence[Dict[str, Any]],
) -> None:
    fields = [
        "game_id",
        "game_date",
        "status_bucket",
        "away_team_id",
        "home_team_id",
        "contract_code",
        "missing_code",
        "required_fields",
        "home_pitcher",
        "away_pitcher",
        "operator_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in _manual_baseball_data_required_rows(results):
            writer.writerow(row)


def _write_starter_announcement_pending_csv(
    path: Path,
    results: Sequence[Dict[str, Any]],
) -> None:
    fields = [
        "game_id",
        "game_date",
        "status_bucket",
        "away_team_id",
        "home_team_id",
        "status_code",
        "expected_announcement_at_kst",
        "operator_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in _starter_announcement_pending_rows(results):
            writer.writerow(row)


def _write_quality_failures_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "game_id",
        "game_date",
        "status_bucket",
        "severity",
        "category",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            target = item["target"]
            for message in item["quality"]["hard_failures"]:
                writer.writerow(
                    {
                        "game_id": target.get("game_id"),
                        "game_date": target.get("game_date"),
                        "status_bucket": target.get("game_status_bucket"),
                        "severity": "hard",
                        "category": _failure_category(message),
                        "message": message,
                    }
                )
            for message in item["quality"]["soft_warnings"]:
                writer.writerow(
                    {
                        "game_id": target.get("game_id"),
                        "game_date": target.get("game_date"),
                        "status_bucket": target.get("game_status_bucket"),
                        "severity": "soft",
                        "category": "content",
                        "message": message,
                    }
                )


def write_reports(
    *,
    output_dir: Path,
    options: Dict[str, Any],
    diagnosis: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = summarize_results(diagnosis, results)
    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "options": options,
        "summary": summary,
        "diagnosis_count": len(diagnosis),
    }

    jsonl_path = output_dir / f"coach_backfill_results_{timestamp}.jsonl"
    summary_json_path = output_dir / f"coach_backfill_summary_{timestamp}.json"
    summary_csv_path = output_dir / f"coach_backfill_summary_{timestamp}.csv"
    missing_csv_path = output_dir / f"coach_missing_data_report_{timestamp}.csv"
    response_notes_csv_path = output_dir / f"coach_response_notes_report_{timestamp}.csv"
    manual_required_csv_path = (
        output_dir / f"coach_manual_baseball_data_required_{timestamp}.csv"
    )
    starter_pending_csv_path = (
        output_dir / f"coach_starter_announcement_pending_{timestamp}.csv"
    )
    quality_csv_path = output_dir / f"coach_quality_failures_{timestamp}.csv"
    latest_jsonl_path = output_dir / "coach_backfill_results_latest.jsonl"
    latest_summary_json_path = output_dir / "coach_backfill_summary_latest.json"
    latest_summary_csv_path = output_dir / "coach_backfill_summary_latest.csv"
    latest_missing_csv_path = output_dir / "coach_missing_data_report_latest.csv"
    latest_response_notes_csv_path = output_dir / "coach_response_notes_report_latest.csv"
    latest_manual_required_csv_path = (
        output_dir / "coach_manual_baseball_data_required_latest.csv"
    )
    latest_starter_pending_csv_path = (
        output_dir / "coach_starter_announcement_pending_latest.csv"
    )
    latest_quality_csv_path = output_dir / "coach_quality_failures_latest.csv"

    _write_jsonl(jsonl_path, results)
    summary_json_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_csv(summary_csv_path, results)
    _write_missing_data_csv(missing_csv_path, results)
    _write_response_notes_csv(response_notes_csv_path, results)
    _write_manual_baseball_data_required_csv(manual_required_csv_path, results)
    _write_starter_announcement_pending_csv(starter_pending_csv_path, results)
    _write_quality_failures_csv(quality_csv_path, results)

    shutil.copyfile(jsonl_path, latest_jsonl_path)
    shutil.copyfile(summary_json_path, latest_summary_json_path)
    shutil.copyfile(summary_csv_path, latest_summary_csv_path)
    shutil.copyfile(missing_csv_path, latest_missing_csv_path)
    shutil.copyfile(response_notes_csv_path, latest_response_notes_csv_path)
    shutil.copyfile(manual_required_csv_path, latest_manual_required_csv_path)
    shutil.copyfile(starter_pending_csv_path, latest_starter_pending_csv_path)
    shutil.copyfile(quality_csv_path, latest_quality_csv_path)

    return {
        "results_jsonl": str(jsonl_path),
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
        "missing_data_csv": str(missing_csv_path),
        "response_notes_csv": str(response_notes_csv_path),
        "manual_baseball_data_required_csv": str(manual_required_csv_path),
        "starter_announcement_pending_csv": str(starter_pending_csv_path),
        "quality_failures_csv": str(quality_csv_path),
        "latest_results_jsonl": str(latest_jsonl_path),
        "latest_summary_json": str(latest_summary_json_path),
        "latest_summary_csv": str(latest_summary_csv_path),
        "latest_missing_data_csv": str(latest_missing_csv_path),
        "latest_response_notes_csv": str(latest_response_notes_csv_path),
        "latest_manual_baseball_data_required_csv": str(
            latest_manual_required_csv_path
        ),
        "latest_starter_announcement_pending_csv": str(
            latest_starter_pending_csv_path
        ),
        "latest_quality_failures_csv": str(latest_quality_csv_path),
    }


def _options_dict(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    applied_request_interval_seconds: float,
) -> Dict[str, Any]:
    return {
        "season_year": args.season_year,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "game_ids": parse_game_ids(args.game_ids),
        "status_bucket": args.status_bucket,
        "offset": args.offset,
        "limit": args.limit,
        "order": args.order,
        "include_noncanonical_game_ids": args.include_noncanonical_game_ids,
        "base_url": args.base_url,
        "output_dir": str(output_dir),
        "timeout_seconds": args.timeout_seconds,
        "delay_seconds": args.delay_seconds,
        "request_interval_seconds": applied_request_interval_seconds,
        "requested_request_interval_seconds": args.request_interval_seconds,
        "min_verify_cache_hit_request_interval_seconds": (
            MIN_VERIFY_CACHE_HIT_REQUEST_INTERVAL_SECONDS
        ),
        "max_attempts": args.max_attempts,
        "retry_backoff_seconds": args.retry_backoff_seconds,
        "dry_run": args.dry_run,
        "verify_cache_hit": args.verify_cache_hit,
        "strict": args.strict,
        "internal_api_key_provided": bool(args.internal_api_key),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = resolve_output_dir(args.output_dir)
    game_ids = parse_game_ids(args.game_ids)
    applied_request_interval_seconds = effective_request_interval_seconds(
        args.request_interval_seconds,
        verify_cache_hit=args.verify_cache_hit,
    )
    options = _options_dict(
        args,
        output_dir,
        applied_request_interval_seconds=applied_request_interval_seconds,
    )
    if applied_request_interval_seconds != args.request_interval_seconds:
        print(
            "Adjusted request interval from "
            f"{args.request_interval_seconds:.1f}s to "
            f"{applied_request_interval_seconds:.1f}s for cache HIT verification."
        )

    diagnosis = _diagnose_games(
        season_year=args.season_year,
        date_from=args.date_from,
        date_to=args.date_to,
        game_ids=game_ids,
    )
    targets = select_records(
        diagnosis,
        status_bucket=args.status_bucket,
        offset=args.offset,
        limit=args.limit,
        order=args.order,
        include_noncanonical_game_ids=args.include_noncanonical_game_ids,
    )
    if not targets:
        print("No Coach backfill targets found.")
        paths = write_reports(
            output_dir=output_dir,
            options=options,
            diagnosis=[],
            results=[],
        )
        print(json.dumps({"summary": summarize_results([], []), "paths": paths}, ensure_ascii=False, indent=2))
        return 0

    results = run_backfill(
        records=targets,
        base_url=args.base_url,
        internal_api_key=args.internal_api_key,
        timeout_seconds=args.timeout_seconds,
        delay_seconds=args.delay_seconds,
        request_interval_seconds=applied_request_interval_seconds,
        max_attempts=args.max_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        dry_run=args.dry_run,
        verify_cache_hit=args.verify_cache_hit,
    )
    paths = write_reports(
        output_dir=output_dir,
        options=options,
        diagnosis=targets,
        results=results,
    )
    summary = summarize_results(targets, results)
    print(json.dumps({"summary": summary, "paths": paths}, ensure_ascii=False, indent=2))

    if args.strict and summary["hard_failure_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
