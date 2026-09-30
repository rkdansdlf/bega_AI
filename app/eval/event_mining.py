"""Classify ``rag_retrieval_events`` rows into failure modes and turn them into
golden-dataset candidates.

Pure functions only; ``scripts/mine_retrieval_events.py`` does the DB read.
Candidates are *unlabeled*: a human (or operator-provided data) must fill
``relevant_doc_keys`` before a candidate enters ``evals/rag_retrieval_v1.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional

LOW_SIMILARITY_THRESHOLD = 0.45

CATEGORIES = (
    "negative_feedback",
    "error",
    "zero_hit",
    "constraint_relaxed",
    "fallback",
    "bad_source",
    "low_similarity",
)


def _loads(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return default
    return default if value is None else value


def max_similarity(scores: Iterable[Mapping[str, Any]]) -> Optional[float]:
    values = [
        float(s["similarity"])
        for s in scores
        if isinstance(s, Mapping) and s.get("similarity") is not None
    ]
    return max(values) if values else None


def classify_event(
    event: Mapping[str, Any], *, low_similarity: float = LOW_SIMILARITY_THRESHOLD
) -> List[str]:
    """Failure-mode labels for one event (empty list == healthy)."""
    meta = _loads(event.get("metadata_filter"), {})
    retrieved = _loads(event.get("retrieved_chunk_ids"), [])
    scores = _loads(event.get("scores"), [])
    labels: List[str] = []
    if not event.get("success", True):
        labels.append("error")
        return labels
    if not retrieved:
        labels.append("zero_hit")
    if isinstance(meta, Mapping):
        if meta.get("constraint_relaxed"):
            labels.append("constraint_relaxed")
        if meta.get("fallback_used"):
            labels.append("fallback")
        if int(meta.get("relevance_guard_dropped") or 0) > 0:
            labels.append("bad_source")
    top = max_similarity(scores)
    if retrieved and top is not None and top < low_similarity:
        labels.append("low_similarity")
    return labels


def merge_feedback(
    report: Dict[str, Any], feedback_rows: Iterable[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Fold DOWN ratings into the candidate list (and add questions not seen
    as unhealthy retrieval events). UP ratings are ignored."""
    by_question = {c["question"]: c for c in report["candidates"]}
    negatives = 0
    for row in feedback_rows:
        if str(row.get("rating")) != "DOWN":
            continue
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        negatives += 1
        cand = by_question.get(question)
        if cand is None:
            cand = {
                "id": candidate_id(question),
                "question": question,
                "filters": {},
                "relevant_doc_keys": [],
                "status": "needs_label",
                "observed_count": 0,
                "failure_modes": [],
                "intent": None,
            }
            by_question[question] = cand
            report["candidates"].append(cand)
        cand["observed_count"] += 1
        if "negative_feedback" not in cand["failure_modes"]:
            cand["failure_modes"] = sorted(
                cand["failure_modes"] + ["negative_feedback"]
            )
        if row.get("corrected_fact"):
            cand.setdefault("corrected_facts", []).append(str(row["corrected_fact"]))
    report["candidates"].sort(key=lambda c: (-c["observed_count"], c["question"]))
    report["summary"]["by_failure_mode"]["negative_feedback"] = negatives
    report["summary"]["candidate_questions"] = len(report["candidates"])
    return report


def candidate_id(question: str) -> str:
    return "mined-" + hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:10]


def build_candidates(
    events: Iterable[Mapping[str, Any]],
    *,
    min_occurrences: int = 1,
    low_similarity: float = LOW_SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """Group unhealthy events by normalised question; rank by frequency."""
    groups: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "labels": Counter(), "intents": Counter(), "filters": None}
    )
    totals: Counter = Counter()
    total_events = 0
    for event in events:
        total_events += 1
        labels = classify_event(event, low_similarity=low_similarity)
        for label in labels:
            totals[label] += 1
        if not labels:
            continue
        question = str(event.get("user_query") or "").strip()
        if not question:
            continue
        g = groups[question]
        g["count"] += 1
        g["labels"].update(labels)
        if event.get("intent"):
            g["intents"][str(event["intent"])] += 1
        if g["filters"] is None:
            meta = _loads(event.get("metadata_filter"), {})
            if isinstance(meta, Mapping):
                g["filters"] = {
                    k: meta[k]
                    for k in ("season_year", "team_id", "player_id", "source_table")
                    if k in meta
                }

    candidates = []
    for question, g in groups.items():
        if g["count"] < min_occurrences:
            continue
        candidates.append(
            {
                "id": candidate_id(question),
                "question": question,
                "filters": g["filters"] or {},
                "relevant_doc_keys": [],
                "status": "needs_label",
                "observed_count": g["count"],
                "failure_modes": sorted(g["labels"]),
                "intent": g["intents"].most_common(1)[0][0] if g["intents"] else None,
            }
        )
    candidates.sort(key=lambda c: (-c["observed_count"], c["question"]))
    return {
        "summary": {
            "events": total_events,
            "by_failure_mode": {k: totals.get(k, 0) for k in CATEGORIES},
            "candidate_questions": len(candidates),
        },
        "candidates": candidates,
    }
