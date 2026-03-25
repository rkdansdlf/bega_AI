from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, List

from psycopg import Connection as PgConnection
from psycopg.rows import dict_row

TEAM_MAPPING_ROWS_QUERY = """
    SELECT
        t.team_id,
        t.team_name,
        t.franchise_id,
        t.founded_year,
        t.is_active,
        tf.current_code
    FROM teams t
    JOIN team_franchises tf ON tf.id = t.franchise_id
    WHERE t.franchise_id IS NOT NULL
    ORDER BY
        t.franchise_id,
        CASE WHEN t.team_id = tf.current_code THEN 0 ELSE 1 END,
        t.is_active DESC,
        t.founded_year DESC,
        t.team_id ASC;
"""

_TEAM_MAPPING_CACHE_TTL_SECONDS = 300.0
_team_mapping_cache_lock = RLock()
_team_mapping_cache_rows: List[Dict[str, Any]] | None = None
_team_mapping_cache_loaded_at = 0.0


@dataclass(frozen=True)
class TeamMappingLoadResult:
    degraded: bool
    reason: str | None


def update_cached_team_mapping_rows(rows: List[Dict[str, Any]]) -> None:
    copied_rows = [dict(row) for row in rows]
    with _team_mapping_cache_lock:
        global _team_mapping_cache_rows
        global _team_mapping_cache_loaded_at
        _team_mapping_cache_rows = copied_rows
        _team_mapping_cache_loaded_at = time.monotonic()


def load_cached_team_mapping_rows(
    max_age_seconds: float = _TEAM_MAPPING_CACHE_TTL_SECONDS,
) -> List[Dict[str, Any]] | None:
    with _team_mapping_cache_lock:
        if _team_mapping_cache_rows is None:
            return None
        if max_age_seconds > 0:
            age = time.monotonic() - _team_mapping_cache_loaded_at
            if age > max_age_seconds:
                return None
        return [dict(row) for row in _team_mapping_cache_rows]


def fetch_team_mapping_rows(connection: PgConnection) -> List[Dict[str, Any]]:
    cursor = connection.cursor(row_factory=dict_row)
    try:
        cursor.execute(TEAM_MAPPING_ROWS_QUERY)
        return cursor.fetchall()
    finally:
        cursor.close()


def load_team_mappings_with_retry(
    *,
    connection: PgConnection,
    fetch_rows: Callable[[PgConnection], List[Dict[str, Any]]],
    apply_rows: Callable[[List[Dict[str, Any]], str], None],
    apply_snapshot_rows: Callable[[List[Dict[str, Any]]], None],
    load_snapshot: Callable[[], List[Dict[str, Any]] | None],
    logger: logging.Logger,
    primary_source: str,
    primary_failure_message: str,
    retry_source: str,
    retry_failure_message: str,
    snapshot_warning_message: str,
    defaults_warning_message: str,
) -> TeamMappingLoadResult:
    try:
        rows = fetch_rows(connection)
        if rows:
            apply_rows(rows, primary_source)
            update_cached_team_mapping_rows(rows)
        return TeamMappingLoadResult(degraded=False, reason=None)
    except Exception as exc:
        logger.warning(primary_failure_message, exc)

    try:
        from app.deps import get_connection_pool

        with get_connection_pool().connection() as retry_conn:
            rows = fetch_rows(retry_conn)
        if rows:
            apply_rows(rows, retry_source)
            update_cached_team_mapping_rows(rows)
            return TeamMappingLoadResult(
                degraded=True,
                reason="oci_retry_recovered",
            )
    except Exception as retry_exc:
        logger.warning(retry_failure_message, retry_exc)

    snapshot_rows = load_snapshot()
    if snapshot_rows:
        apply_snapshot_rows(snapshot_rows)
        logger.warning(snapshot_warning_message, len(snapshot_rows))
        return TeamMappingLoadResult(
            degraded=True,
            reason="last_good_snapshot",
        )

    logger.warning(defaults_warning_message)
    return TeamMappingLoadResult(
        degraded=True,
        reason="defaults",
    )
