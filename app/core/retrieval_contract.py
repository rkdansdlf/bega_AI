"""Backend-neutral retrieval contract shared by the PostgreSQL and Oracle paths.

Quality guarantees (filter allowlist, entity constraints, relevance guard,
generation gate) must not depend on which database served the query. Every
backend therefore has to:

1. resolve filter keys through :func:`resolve_backend_filters`, which rejects
   anything unknown or unsupported *instead of silently ignoring it* (an
   ignored filter widens the search scope invisibly);
2. return rows that pass :func:`enforce_result_contract` — entity scope
   (``season_year``/``team_id``/``player_id``) and provenance keys present so
   the relevance guard and provenance layers can act on them;
3. declare what it can enforce via :func:`backend_capabilities`, which readiness
   uses to fail closed when a required capability is missing.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .retrieval_policy import InvalidRetrievalFilter, resolve_filter_column

# Internal (non-column) filter keys understood by every backend.
INTERNAL_FILTER_KEYS = frozenset(
    {
        "_include_game_inning_scores",
        "_exclude_source_tables",
        "source_table_in",
    }
)

# Keys every result row must carry (value may be None when unknown).
ENTITY_SCOPE_KEYS = ("season_year", "team_id", "player_id")
REQUIRED_RESULT_KEYS = (
    (
        "id",
        "source_table",
        "source_row_id",
        "title",
        "content",
        "meta",
        "similarity",
        "combined_score",
    )
    + ENTITY_SCOPE_KEYS
    + ("retrieval_backend", "index_generation")
)

# Capabilities a backend must have to serve production traffic.
REQUIRED_CAPABILITIES = ("filter_allowlist", "entity_scope", "generation_gate")


class RetrievalContractViolation(RuntimeError):
    """A backend returned rows (or is configured) outside the shared contract."""


class UnsupportedBackendFilter(InvalidRetrievalFilter):
    """Filter key is allowed by policy but this backend cannot enforce it."""


def resolve_backend_filters(
    filters: Optional[Mapping[str, Any]],
    *,
    backend: str,
    supported_columns: Iterable[str],
    extra_keys: Iterable[str] = (),
) -> Dict[str, Any]:
    """Validate ``filters`` for ``backend`` and return the enforceable subset.

    Raises :class:`InvalidRetrievalFilter` for keys outside the shared policy
    and :class:`UnsupportedBackendFilter` for policy-allowed keys the backend
    cannot enforce. ``None`` values are dropped (matching PostgreSQL).
    """
    supported = set(supported_columns)
    extra = set(extra_keys)
    resolved: Dict[str, Any] = {}
    for key, value in (filters or {}).items():
        if key in INTERNAL_FILTER_KEYS or key in extra:
            resolved[key] = value
            continue
        if value is None:
            continue
        column, json_key = resolve_filter_column(key)  # allowlist / injection guard
        if json_key is None and column not in supported:
            raise UnsupportedBackendFilter(
                f"{backend} cannot enforce filter {key!r}; refusing to ignore it"
            )
        resolved[key] = value
    return resolved


def backend_capabilities(
    *,
    backend: str,
    supported_columns: Iterable[str],
    generation_gate: bool,
    temporal_filters: bool,
) -> Dict[str, Any]:
    supported = set(supported_columns)
    return {
        "backend": backend,
        "filter_allowlist": True,
        "entity_scope": all(k in supported for k in ENTITY_SCOPE_KEYS),
        "generation_gate": bool(generation_gate),
        # Informational: not required (Oracle relies on index_status).
        "temporal_filters": bool(temporal_filters),
    }


def missing_required_capabilities(capabilities: Mapping[str, Any]) -> List[str]:
    return [c for c in REQUIRED_CAPABILITIES if not capabilities.get(c)]


def normalize_result(
    row: Mapping[str, Any], *, backend: str, index_generation: Optional[str]
) -> Dict[str, Any]:
    """Fill contract keys so downstream code never branches on backend."""
    out = dict(row)
    meta = out.get("meta")
    if not isinstance(meta, dict) or (not meta and out.get("metadata")):
        meta = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
    out["meta"] = meta
    out["metadata"] = meta
    for key in ENTITY_SCOPE_KEYS:
        if out.get(key) is None:
            out[key] = meta.get(key) if isinstance(meta, dict) else None
        out.setdefault(key, None)
    out.setdefault("title", None)
    out.setdefault("content", None)
    out.setdefault("similarity", 0.0)
    out.setdefault("combined_score", out.get("similarity") or 0.0)
    out["retrieval_backend"] = backend
    out["index_generation"] = index_generation
    return out


def enforce_result_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    backend: str,
    require_generation: bool = False,
    constraints: Optional[Mapping[str, Any]] = None,
) -> None:
    """Fail closed if any row breaks the contract.

    ``constraints`` (e.g. the requested season/team/player) are re-verified on
    the returned rows: a backend that ignored a filter must not get its rows
    into the context.
    """
    for row in rows:
        missing = [k for k in REQUIRED_RESULT_KEYS if k not in row]
        if missing:
            raise RetrievalContractViolation(
                f"{backend} row {row.get('id')!r} missing contract keys: {missing}"
            )
        if require_generation and not row.get("index_generation"):
            raise RetrievalContractViolation(
                f"{backend} row {row.get('id')!r} has no index generation"
            )
        for key, wanted in (constraints or {}).items():
            if key not in ENTITY_SCOPE_KEYS or wanted is None:
                continue
            actual = row.get(key)
            if actual is not None and str(actual).upper() != str(wanted).upper():
                raise RetrievalContractViolation(
                    f"{backend} row {row.get('id')!r} violates {key}="
                    f"{wanted!r} (got {actual!r})"
                )


def entity_constraints(filters: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {k: v for k, v in (filters or {}).items() if k in ENTITY_SCOPE_KEYS and v}
