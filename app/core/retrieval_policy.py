"""Retrieval filter safety policy.

Two guarantees live here:

1. Only known ``rag_chunks`` columns / ``meta.*`` JSON keys may become SQL
   identifiers. Filter *values* are always parameterised, but the key is
   interpolated, so it must come from an allowlist.
2. Retrieval fallback may relax *how* we search (``source_table``) but never
   *who/when* we search for (player, team, season) on factual queries —
   otherwise a "2025 KIA" question can be answered from 2024 LG chunks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

ALLOWED_FILTER_COLUMNS = frozenset(
    {
        "season_year",
        "season_id",
        "league_type_code",
        "team_id",
        "player_id",
        "source_table",
        "source_type",
        "topic_key",
        "source_row_id",
    }
)
ALLOWED_JSON_COLUMNS = {"meta": "metadata", "metadata": "metadata"}
_JSON_KEY_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")

# Keys identifying the entity a factual answer is about. Never relaxed.
CORE_ENTITY_KEYS = ("player_id", "team_id", "season_year")
# Intents whose answers are not about a specific player/team/season.
RELAXABLE_INTENTS = frozenset(
    {"knowledge_explanation", "baseball_explainer", "general_conversation"}
)


class InvalidRetrievalFilter(ValueError):
    """Raised when a retrieval filter key is not on the allowlist."""


def resolve_filter_column(key: str) -> Tuple[str, Optional[str]]:
    """Return ``(sql_column, json_key_or_None)`` for an allowlisted filter key."""
    if not isinstance(key, str):
        raise InvalidRetrievalFilter(f"filter key must be str: {key!r}")
    if "." in key:
        json_field, json_key = key.split(".", 1)
        column = ALLOWED_JSON_COLUMNS.get(json_field)
        if column is None or not _JSON_KEY_RE.match(json_key):
            raise InvalidRetrievalFilter(f"unsupported filter key: {key!r}")
        return column, json_key
    if key not in ALLOWED_FILTER_COLUMNS:
        raise InvalidRetrievalFilter(f"unsupported filter key: {key!r}")
    return key, None


def protected_filter_keys(
    filters: Mapping[str, Any], *, intent: str = "", is_regulation: bool = False
) -> frozenset:
    """Filter keys that fallback must not remove for this query."""
    if is_regulation or intent in RELAXABLE_INTENTS:
        return frozenset()
    return frozenset(k for k in CORE_ENTITY_KEYS if filters.get(k) is not None)


def relaxed_fields(
    original: Mapping[str, Any], effective: Optional[Mapping[str, Any]]
) -> list:
    """Filter keys present in ``original`` but missing from ``effective``."""
    effective = effective or {}
    return sorted(k for k in original if k not in effective)


def assert_core_constraints_kept(
    original: Mapping[str, Any],
    effective: Optional[Mapping[str, Any]],
    protected: Iterable[str],
) -> None:
    dropped = set(relaxed_fields(original, effective)) & set(protected)
    if dropped:
        raise AssertionError(f"core entity constraints relaxed: {sorted(dropped)}")


def annotate_relaxation(
    original: Mapping[str, Any], effective: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    fields = relaxed_fields(original, effective)
    return {"constraint_relaxed": bool(fields), "relaxed_fields": fields}
