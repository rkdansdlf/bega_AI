"""Provenance carried through the chat response caches.

A cache hit must describe the *original* answer, not the cache itself:
a hit may only report ``verified=True`` if the fresh generation did, and it
must return the same ``data_sources`` / ``answer_sources`` / ``as_of_date``.
Rows written before provenance existed carry none, so they restore as
unverified instead of being promoted to verified.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional

PROVENANCE_SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_cache_provenance(
    meta: Mapping[str, Any], fingerprint: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Snapshot the provenance fields of a fresh generation result/meta."""
    body: Dict[str, Any] = {
        "verified": bool(meta.get("verified", False)),
        "data_sources": _json_safe(meta.get("data_sources") or []),
        "answer_sources": _json_safe(meta.get("answer_sources") or []),
        "as_of_date": _json_safe(meta.get("as_of_date")),
        "grounding_mode": meta.get("grounding_mode"),
        "source_tier": meta.get("source_tier"),
        "fallback_reason": meta.get("fallback_reason"),
        "fallback_answer_used": bool(meta.get("fallback_answer_used", False)),
    }
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        **body,
        "fingerprint": _json_safe(dict(fingerprint)) if fingerprint else None,
        "provenance_hash": _hash(body),
    }


def is_cacheable_provenance(provenance: Optional[Mapping[str, Any]]) -> bool:
    """Fallback answers are never worth serving from cache."""
    if not provenance:
        return False
    return not provenance.get("fallback_answer_used", False)


def restore_cache_provenance(raw: Any) -> Dict[str, Any]:
    """Provenance fields for a cache hit payload.

    Missing/unparseable provenance (legacy rows) restores as unverified.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    if not isinstance(raw, Mapping) or raw.get("schema_version") != (
        PROVENANCE_SCHEMA_VERSION
    ):
        return {
            "verified": False,
            "data_sources": [],
            "answer_sources": [],
            "as_of_date": None,
            "origin_grounding_mode": None,
            "origin_source_tier": None,
            "fallback_reason": "cache_provenance_missing",
            "provenance_hash": None,
            "fingerprint": None,
        }
    return {
        "verified": bool(raw.get("verified", False)),
        "data_sources": list(raw.get("data_sources") or []),
        "answer_sources": list(raw.get("answer_sources") or []),
        "as_of_date": raw.get("as_of_date"),
        "origin_grounding_mode": raw.get("grounding_mode"),
        "origin_source_tier": raw.get("source_tier"),
        "fallback_reason": raw.get("fallback_reason"),
        "provenance_hash": raw.get("provenance_hash"),
        "fingerprint": raw.get("fingerprint"),
    }
