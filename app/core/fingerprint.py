"""Canonical per-response fingerprint for quality-regression analysis.

Answers ("why is it worse than last week?") are only comparable when every
component that shapes them is recorded: prompt text, planner/retrieval logic,
reranker, generation model and embedding model. Bump the version constants
when the corresponding logic changes behaviour.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional

PLANNER_VERSION = "planner_v1"
# v2: entity-constraint-preserving fallback + relevance guard.
RETRIEVAL_VERSION = "retrieval_v2"


@lru_cache(maxsize=1)
def prompt_hash() -> str:
    """Hash of every public str constant in ``app.core.prompts``."""
    from . import prompts

    parts = []
    for name, value in sorted(vars(prompts).items()):
        if name.startswith("_") or not isinstance(value, str):
            continue
        parts.append(f"{name}\x00{value}")
    digest = hashlib.sha256("\x01".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


def prompt_version() -> str:
    return f"prompts-{prompt_hash()}"


def _reranker_version(settings: Any) -> str:
    if not bool(getattr(settings, "rag_rerank_enabled", False)):
        return "none"
    provider = str(getattr(settings, "rag_reranker_provider", "score") or "score")
    if provider == "http" and getattr(settings, "rag_reranker_model", None):
        return f"http:{settings.rag_reranker_model}"
    return "score_v1"


def _embedding_signature(settings: Any) -> Optional[str]:
    try:
        from .embeddings import _embed_signature

        return _embed_signature(settings)
    except Exception:  # noqa: BLE001
        return None


def _model_from_result(result: Mapping[str, Any], settings: Any) -> Optional[str]:
    usage = result.get("model_usage") or []
    for entry in usage:
        if isinstance(entry, Mapping) and entry.get("model"):
            return str(entry["model"])
    return (
        getattr(settings, "coach_openrouter_model", None)
        or getattr(settings, "openrouter_model", None)
        or getattr(settings, "gemini_model", None)
    )


def build_response_fingerprint(
    settings: Any, result: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    result = result or {}
    return {
        "prompt_version": prompt_version(),
        "prompt_hash": prompt_hash(),
        "planner_version": PLANNER_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "reranker_version": _reranker_version(settings),
        "model": _model_from_result(result, settings),
        "embedding_signature": _embedding_signature(settings),
    }
