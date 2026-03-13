from __future__ import annotations

import logging
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TeamMappingLoadResult:
    degraded: bool
    reason: str | None


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
        return TeamMappingLoadResult(degraded=False, reason=None)
    except Exception as exc:
        logger.warning(primary_failure_message, exc)

    try:
        from app.deps import get_connection_pool

        with get_connection_pool().connection() as retry_conn:
            rows = fetch_rows(retry_conn)
        if rows:
            apply_rows(rows, retry_source)
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
