"""Deterministic relevance guard for retrieved chunks.

``quality_score`` measures source quality, not whether a chunk answers *this*
question. The guard drops chunks whose season / team / player explicitly
conflict with what the query asked for. A chunk that carries no value for a
dimension is kept (unknown is not a conflict).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

GUARDED_DIMENSIONS = ("season_year", "team_id", "player_id")


def _doc_value(doc: Mapping[str, Any], key: str) -> Any:
    value = doc.get(key)
    if value is None:
        meta = doc.get("meta") or doc.get("metadata") or {}
        if isinstance(meta, Mapping):
            value = meta.get(key)
    return value


def _same(a: Any, b: Any) -> bool:
    return str(a).strip().upper() == str(b).strip().upper()


@dataclass
class GuardResult:
    kept: List[Dict[str, Any]]
    dropped: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def all_dropped(self) -> bool:
        return bool(self.dropped) and not self.kept


def conflicts(
    doc: Mapping[str, Any], constraints: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    for dim in GUARDED_DIMENSIONS:
        wanted = constraints.get(dim)
        if wanted is None:
            continue
        actual = _doc_value(doc, dim)
        if actual is None:
            continue
        if not _same(wanted, actual):
            return {
                "id": doc.get("id"),
                "dimension": dim,
                "wanted": wanted,
                "actual": actual,
            }
    return None


def apply_relevance_guard(
    docs: Sequence[Dict[str, Any]], constraints: Mapping[str, Any]
) -> GuardResult:
    """Split ``docs`` into kept/dropped against explicit entity constraints."""
    active = {k: v for k, v in constraints.items() if k in GUARDED_DIMENSIONS and v}
    if not active:
        return GuardResult(kept=list(docs))
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    reasons: List[Dict[str, Any]] = []
    for doc in docs:
        reason = conflicts(doc, active)
        if reason is None:
            kept.append(doc)
        else:
            dropped.append(doc)
            reasons.append(reason)
    return GuardResult(kept=kept, dropped=dropped, reasons=reasons)
