"""Read-only operator-provided P0 data fast-path for chatbot answers.

This module never collects or infers baseball data. It only reads verified
operator-provided rows already applied to the internal DB. Missing rows return
``None`` so callers can preserve the MANUAL_BASEBALL_DATA_REQUIRED contract.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import date, datetime, timedelta
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from psycopg.rows import dict_row

from app.tools.team_code_resolver import TeamCodeResolver
from app.tools.team_display import resolve_team_display_name

logger = logging.getLogger(__name__)

CONFIDENCE_MINIMUM = 0.70
MAX_ROWS = 8
PARTIAL_NOTICE = "운영자 제공 데이터 기준, 확인된 항목만 정리합니다."

P0_DOMAINS = {"season_meta", "schedule_window", "game_day_lineup", "roster_news"}

_SCHEDULE_TOKENS = (
    "경기 일정",
    "경기일정",
    "경기표",
    "스코어보드",
    "오늘 경기",
    "내일 경기",
    "이번 주",
    "이번주",
)
_SEASON_EVENT_TOKENS = (
    "개막",
    "올스타",
    "시즌 종료",
    "정규시즌 종료",
)
_LINEUP_TOKENS = (
    "라인업",
    "선발 라인업",
    "타순",
    "오늘 선발",
    "선발투수",
    "선발 투수",
)
_ROSTER_TOKENS = (
    "부상",
    "복귀",
    "콜업",
    "말소",
    "엔트리",
    "로스터",
    "계약",
    "트레이드",
    "외국인 선수",
)


def try_build_operator_fast_path_result(
    conn: Any,
    query: str,
    *,
    today: Optional[date] = None,
) -> Optional[Dict[str, Any]]:
    """Return an operator-provided fast-path answer or ``None``.

    Any DB/schema/runtime failure falls back to ``None`` to avoid replacing the
    existing manual-data contract with an operational error.
    """

    normalized_query = _normalize_text(query)
    if not normalized_query:
        return None

    try:
        resolver = TeamCodeResolver()
        today_value = today or date.today()
        if _is_lineup_query(normalized_query):
            result = _lineup_result(conn, normalized_query, resolver, today_value)
            if result is not None:
                return result
        if _is_schedule_query(normalized_query):
            result = _schedule_result(conn, normalized_query, resolver, today_value)
            if result is not None:
                return result
        if _is_roster_query(normalized_query):
            result = _roster_result(conn, normalized_query, resolver)
            if result is not None:
                return result
        if _is_season_event_query(normalized_query):
            result = _season_event_result(conn, normalized_query)
            if result is not None:
                return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("[OperatorData] fast-path skipped: %s", exc)
        return None

    return None


def _build_result(
    *,
    answer: str,
    domain: str,
    rows: Sequence[Mapping[str, Any]],
    partial: bool = True,
) -> Dict[str, Any]:
    queue_ids = [
        str(row.get("queue_id") or "").strip()
        for row in rows
        if str(row.get("queue_id") or "").strip()
    ]
    source_checked_at = next(
        (
            str(row.get("source_checked_at") or "").strip()
            for row in rows
            if str(row.get("source_checked_at") or "").strip()
        ),
        None,
    )
    return {
        "answer": answer,
        "citations": [],
        "intent": "operator_data_fast_path",
        "retrieved": [],
        "strategy": "operator_data_fast_path",
        "verified": True,
        "tool_calls": [],
        "tool_results": [],
        "data_sources": [
            {
                "tool": f"operator_data.{domain}",
                "verified": True,
                "data_points": len(rows),
            }
        ],
        "visualizations": [],
        "planner_mode": "fast_path",
        "planner_cache_hit": False,
        "tool_execution_mode": "none",
        "fallback_triggered": False,
        "fallback_answer_used": False,
        "grounding_mode": "operator_provided",
        "source_tier": "operator_data",
        "answer_sources": ["operator_data"],
        "as_of_date": source_checked_at,
        "operator_data_domain": domain,
        "operator_data_partial": bool(partial),
        "operator_data_queue_ids": queue_ids,
        "perf": {
            "total_ms": 0.0,
            "analysis_ms": 0.0,
            "tool_ms": 0.0,
            "answer_ms": 0.0,
            "first_token_ms": 0.0,
            "tool_count": 0,
            "tool_execution_mode": "none",
            "planner_cache_hit": False,
            "planner_mode": "fast_path",
            "model": "operator_data",
        },
    }


def _schedule_result(
    conn: Any,
    query: str,
    resolver: TeamCodeResolver,
    today: date,
) -> Optional[Dict[str, Any]]:
    date_range = _extract_date_range(query, today)
    if date_range is None:
        return None
    team_code = _extract_team_code(query, resolver)
    start_date, end_date = date_range
    rows = _fetch_schedule_rows(conn, start_date, end_date, team_code)
    if not rows:
        return None

    lines = [PARTIAL_NOTICE]
    for row in rows:
        game_date = _format_date(row.get("game_date"))
        start_time = _format_time(row.get("start_time"))
        away = _display_team(row.get("away_team"), resolver)
        home = _display_team(row.get("home_team"), resolver)
        stadium = _normalize_text(row.get("stadium_name") or row.get("stadium"))
        status = _normalize_text(row.get("game_status"))
        line = f"- {game_date} {start_time} {away} @ {home}"
        if stadium:
            line += f", {stadium}"
        if status:
            line += f", 상태 {status}"
        lines.append(line)

    return _build_result(
        answer="\n".join(lines),
        domain="schedule_window",
        rows=rows,
        partial=True,
    )


def _season_event_result(conn: Any, query: str) -> Optional[Dict[str, Any]]:
    year = _extract_year(query)
    event_token = _season_event_token(query)
    rows = _fetch_season_event_rows(conn, year, event_token)
    if not rows:
        return None

    lines = [PARTIAL_NOTICE]
    for row in rows:
        event_name = _normalize_text(row.get("event_name"))
        event_date = _format_date(row.get("event_date"))
        stadium = _normalize_text(row.get("stadium_name"))
        line = f"- {event_name}: {event_date}"
        if stadium:
            line += f", {stadium}"
        lines.append(line)

    return _build_result(
        answer="\n".join(lines),
        domain="season_meta",
        rows=rows,
        partial=True,
    )


def _roster_result(
    conn: Any,
    query: str,
    resolver: TeamCodeResolver,
) -> Optional[Dict[str, Any]]:
    year = _extract_year(query)
    team_code = _extract_team_code(query, resolver)
    if year is None or team_code is None:
        return None
    rows = _fetch_roster_rows(conn, year, team_code)
    if not rows:
        return None

    lines = [PARTIAL_NOTICE]
    for row in rows:
        effective_date = _format_date(row.get("effective_date"))
        team = _display_team(row.get("team_code"), resolver)
        player = _normalize_text(row.get("player_name"))
        event_type = _normalize_text(row.get("roster_event_type"))
        status_text = _normalize_text(row.get("status_text"))
        line = f"- {effective_date} {team} {player} {event_type}"
        if status_text:
            line += f": {status_text}"
        lines.append(line)

    return _build_result(
        answer="\n".join(lines),
        domain="roster_news",
        rows=rows,
        partial=True,
    )


def _lineup_result(
    conn: Any,
    query: str,
    resolver: TeamCodeResolver,
    today: date,
) -> Optional[Dict[str, Any]]:
    date_range = _extract_date_range(query, today)
    if date_range is None:
        return None
    team_code = _extract_team_code(query, resolver)
    rows = _fetch_lineup_rows(conn, date_range, team_code)
    if not rows:
        return None

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("game_id") or ""), []).append(row)

    lines = [PARTIAL_NOTICE]
    for game_id, game_rows in grouped.items():
        first = game_rows[0]
        game_date = _format_date(first.get("game_date"))
        away = _display_team(first.get("away_team"), resolver)
        home = _display_team(first.get("home_team"), resolver)
        lines.append(f"- {game_date} {away} @ {home} ({game_id})")
        for row in game_rows[:6]:
            team = _display_team(row.get("team_code"), resolver)
            order = _normalize_text(row.get("batting_order"))
            player = _normalize_text(row.get("player_name"))
            position = _normalize_text(row.get("position"))
            order_text = f"{order}번 " if order else ""
            position_text = f" {position}" if position else ""
            lines.append(f"  - {team}: {order_text}{player}{position_text}")

    return _build_result(
        answer="\n".join(lines),
        domain="game_day_lineup",
        rows=rows,
        partial=True,
    )


def _fetch_schedule_rows(
    conn: Any,
    start_date: date,
    end_date: date,
    team_code: Optional[str],
) -> List[Mapping[str, Any]]:
    query = """
        SELECT queue_id, game_date, game_id, home_team, away_team, stadium_name,
               start_time, game_status, source_name, source_checked_at, confidence
        FROM operator_schedule_items
        WHERE game_date BETWEEN %s AND %s
          AND is_verified = true
          AND confidence >= %s
    """
    params: List[Any] = [start_date.isoformat(), end_date.isoformat(), CONFIDENCE_MINIMUM]
    if team_code:
        query += " AND (home_team = %s OR away_team = %s)"
        params.extend([team_code, team_code])
    query += " ORDER BY game_date, start_time, game_id LIMIT %s"
    params.append(MAX_ROWS)
    return _verified_source_rows(_fetch_rows(conn, query, params))


def _fetch_season_event_rows(
    conn: Any,
    year: Optional[int],
    event_token: Optional[str],
) -> List[Mapping[str, Any]]:
    query = """
        SELECT queue_id, season_year, event_name, event_date, stadium_name,
               source_name, source_checked_at, confidence
        FROM operator_season_events
        WHERE is_verified = true
          AND confidence >= %s
    """
    params: List[Any] = [CONFIDENCE_MINIMUM]
    if year:
        query += " AND season_year = %s"
        params.append(year)
    if event_token:
        query += " AND event_name ILIKE %s"
        params.append(f"%{event_token}%")
    query += " ORDER BY event_date, event_name LIMIT %s"
    params.append(MAX_ROWS)
    return _verified_source_rows(_fetch_rows(conn, query, params))


def _fetch_roster_rows(
    conn: Any,
    year: Optional[int],
    team_code: Optional[str],
) -> List[Mapping[str, Any]]:
    query = """
        SELECT queue_id, season_year, team_code, player_name, roster_event_type,
               effective_date, status_text, source_name, source_checked_at, confidence
        FROM operator_roster_events
        WHERE is_verified = true
          AND confidence >= %s
    """
    params: List[Any] = [CONFIDENCE_MINIMUM]
    if year:
        query += " AND season_year = %s"
        params.append(year)
    if team_code:
        query += " AND team_code = %s"
        params.append(team_code)
    query += " ORDER BY effective_date DESC, queue_id LIMIT %s"
    params.append(MAX_ROWS)
    return _verified_source_rows(_fetch_rows(conn, query, params))


def _fetch_lineup_rows(
    conn: Any,
    date_range: Optional[tuple[date, date]],
    team_code: Optional[str],
) -> List[Mapping[str, Any]]:
    query = """
        SELECT gl.game_id, gl.team_code, gl.player_name, gl.position,
               gl.batting_order, gl.notes, g.game_date, g.home_team, g.away_team
        FROM game_lineups gl
        JOIN game g ON g.game_id = gl.game_id
        WHERE gl.notes::text LIKE %s
    """
    params: List[Any] = ["%manual_lineup%"]
    if date_range is not None:
        start_date, end_date = date_range
        query += " AND DATE(g.game_date) BETWEEN %s AND %s"
        params.extend([start_date.isoformat(), end_date.isoformat()])
    if team_code:
        query += " AND gl.team_code = %s"
        params.append(team_code)
    query += " ORDER BY g.game_date DESC, gl.game_id, gl.team_code, gl.batting_order LIMIT %s"
    params.append(MAX_ROWS)
    rows = _fetch_rows(conn, query, params)
    normalized: List[Mapping[str, Any]] = []
    for row in rows:
        notes = _notes_payload(row.get("notes"))
        if notes.get("source_type") != "manual_lineup":
            continue
        if notes.get("is_verified") is not True:
            continue
        confidence = _as_float(notes.get("confidence"))
        if confidence is None or confidence < CONFIDENCE_MINIMUM:
            continue
        normalized.append(
            {
                **dict(row),
                "queue_id": str(notes.get("queue_id") or row.get("game_id") or ""),
                "source_name": notes.get("source_name", ""),
                "source_checked_at": notes.get("source_checked_at", ""),
                "confidence": confidence if confidence is not None else CONFIDENCE_MINIMUM,
            }
        )
    return normalized


def _fetch_rows(conn: Any, query: str, params: Sequence[Any]) -> List[Mapping[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, tuple(params))
        return [dict(row) for row in list(cur.fetchall() or [])]


def _verified_source_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    verified_rows: List[Mapping[str, Any]] = []
    for row in rows:
        if "is_verified" in row and row.get("is_verified") is not True:
            continue
        confidence = _as_float(row.get("confidence"))
        if confidence is None or confidence < CONFIDENCE_MINIMUM:
            continue
        verified_rows.append(row)
    return verified_rows


def _notes_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    with suppress(Exception):
        parsed = raw if isinstance(raw, str) else str(raw)
        value = json.loads(parsed)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _extract_date_range(query: str, today: date) -> Optional[tuple[date, date]]:
    explicit = _extract_explicit_date(query, today)
    query_lower = query.lower()
    if explicit is not None:
        if _contains_any(query_lower, ("부터", "까지", "~")):
            dates = _extract_all_explicit_dates(query, today)
            if len(dates) >= 2:
                return min(dates), max(dates)
        return explicit, explicit
    if _contains_any(query_lower, ("오늘", "금일")):
        return today, today
    if "내일" in query_lower:
        target = today + timedelta(days=1)
        return target, target
    if "어제" in query_lower:
        target = today - timedelta(days=1)
        return target, target
    if _contains_any(query_lower, ("이번 주", "이번주")):
        return today, today + timedelta(days=6)
    return None


def _extract_explicit_date(query: str, today: date) -> Optional[date]:
    dates = _extract_all_explicit_dates(query, today)
    return dates[0] if dates else None


def _extract_all_explicit_dates(query: str, today: date) -> List[date]:
    dates: List[date] = []
    for match in re.finditer(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", query):
        with suppress(ValueError):
            dates.append(date(int(match.group(1)), int(match.group(2)), int(match.group(3))))
    for match in re.finditer(r"\b(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", query):
        with suppress(ValueError):
            dates.append(date(int(match.group(1)), int(match.group(2)), int(match.group(3))))
    if not dates:
        for match in re.finditer(r"\b(\d{1,2})\s*월\s*(\d{1,2})\s*일", query):
            with suppress(ValueError):
                dates.append(date(today.year, int(match.group(1)), int(match.group(2))))
    return dates


def _extract_year(query: str) -> Optional[int]:
    match = re.search(r"(20\d{2})\s*년", query)
    if match:
        with suppress(ValueError):
            return int(match.group(1))
    match = re.search(r"\b(20\d{2})\b", query)
    if not match:
        return None
    with suppress(ValueError):
        return int(match.group(1))
    return None


def _extract_team_code(query: str, resolver: TeamCodeResolver) -> Optional[str]:
    compact_query = query.lower().replace(" ", "")
    aliases = sorted(resolver.name_to_canonical, key=len, reverse=True)
    for alias in aliases:
        normalized_alias = str(alias or "").lower().replace(" ", "")
        if normalized_alias and normalized_alias in compact_query:
            code = resolver.resolve_canonical(alias)
            if code:
                return str(code)
    for code in resolver.code_to_name:
        normalized_code = str(code or "").lower()
        if normalized_code and normalized_code in compact_query:
            return str(code)
    return None


def _season_event_token(query: str) -> Optional[str]:
    for token in _SEASON_EVENT_TOKENS:
        if token in query:
            return token
    return None


def _is_schedule_query(query: str) -> bool:
    return _contains_any(query, _SCHEDULE_TOKENS)


def _is_season_event_query(query: str) -> bool:
    return _contains_any(query, _SEASON_EVENT_TOKENS)


def _is_lineup_query(query: str) -> bool:
    return _contains_any(query, _LINEUP_TOKENS)


def _is_roster_query(query: str) -> bool:
    return _contains_any(query, _ROSTER_TOKENS)


def _contains_any(value: str, tokens: Iterable[str]) -> bool:
    return any(token in value for token in tokens)


def _display_team(value: Any, resolver: TeamCodeResolver) -> str:
    text = _normalize_text(value)
    if not text:
        return "팀 미상"
    return str(resolve_team_display_name(text, resolver.display_name))


def _format_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _normalize_text(value)
    return text[:10] if len(text) >= 10 else text


def _format_time(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%H:%M")
    text = _normalize_text(value)
    if "T" in text:
        return text.split("T", 1)[1][:5]
    return text[:5] if re.match(r"^\d{1,2}:\d{2}", text) else text


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(_normalize_text(value))
    except (TypeError, ValueError):
        return None
