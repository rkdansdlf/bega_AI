"""Invalidate chat answer caches when ingested source data changes.

TTL alone serves stale answers until expiry after new data lands. On a
successful ingest we delete cached answers whose scope overlaps the change:

* rows whose ``filters_json`` names an affected season / team / player, and
* "floating" rows with no season filter (they mean "the current season"),
  except intents that are stable regardless of data (explanations, chit-chat).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

CACHE_TABLES = ("chat_response_cache", "chat_semantic_response_cache")
STABLE_INTENTS = ("general_conversation", "knowledge_explanation")


def _as_text_list(values: Iterable[Any]) -> list:
    return sorted({str(v) for v in values if v is not None and str(v) != ""})


def build_invalidation_sql(table: str) -> str:
    if table not in CACHE_TABLES:
        raise ValueError(f"unsupported cache table: {table!r}")
    return f"""
        DELETE FROM {table}
        WHERE (%(everything)s AND COALESCE(intent, '') <> ALL(%(stable)s))
           OR (filters_json->>'season_year') = ANY(%(seasons)s)
           OR (filters_json->>'team_id') = ANY(%(teams)s)
           OR (filters_json->>'player_id') = ANY(%(players)s)
           OR (
                (filters_json IS NULL OR filters_json->>'season_year' IS NULL)
                AND COALESCE(intent, '') <> ALL(%(stable)s)
           )
    """


async def invalidate_chat_caches(
    conn: Any,
    *,
    season_years: Sequence[Any] = (),
    team_ids: Sequence[Any] = (),
    player_ids: Sequence[Any] = (),
    everything: bool = False,
) -> Dict[str, int]:
    """Delete affected rows from both chat caches; returns per-table counts."""
    params = {
        "seasons": _as_text_list(season_years),
        "teams": _as_text_list(team_ids),
        "players": _as_text_list(player_ids),
        "stable": list(STABLE_INTENTS),
        "everything": bool(everything),
    }
    deleted: Dict[str, int] = {}
    for table in CACHE_TABLES:
        cur = await conn.execute(build_invalidation_sql(table), params)
        deleted[table] = int(getattr(cur, "rowcount", 0) or 0)
    return deleted


async def invalidate_for_ingest_run(
    request: Any, *, pool: Optional[Any] = None
) -> Dict[str, int]:
    """Best-effort invalidation for a succeeded ingest ``request``.

    Never raises: a failed invalidation must not fail the ingest run.
    """
    try:
        if pool is None:
            from ..deps import get_connection_pool

            pool = get_connection_pool()
        seasons = [request.season_year] if request.season_year is not None else []
        async with pool.connection() as conn:
            deleted = await invalidate_chat_caches(
                conn,
                season_years=seasons,
                # A run without a season can touch any season.
                everything=request.season_year is None,
            )
        logger.info(
            "[CacheInvalidation] ingest tables=%s season=%s deleted=%s",
            list(getattr(request, "tables", ())),
            request.season_year,
            deleted,
        )
        return deleted
    except Exception as exc:  # noqa: BLE001
        logger.warning("[CacheInvalidation] failed: %s", type(exc).__name__)
        return {}
