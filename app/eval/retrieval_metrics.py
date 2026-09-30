"""Retrieval-quality metrics over golden relevance judgments.

Documents are identified by a stable key ``"<source_table>:<source_row_id>"``
(never a serial ``rag_chunks.id``, which changes on re-ingest).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def doc_key(doc: Mapping[str, Any]) -> str:
    return f"{doc.get('source_table')}:{doc.get('source_row_id')}"


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    question: str
    relevant: frozenset  # doc keys judged relevant
    filters: Dict[str, Any] = field(default_factory=dict)
    required_source_tables: frozenset = frozenset()
    must_not_sources: frozenset = frozenset()  # doc keys OR source_table names
    expect_no_answer: bool = False  # nothing in the corpus should match

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "RetrievalCase":
        return RetrievalCase(
            id=str(raw["id"]),
            question=str(raw["question"]),
            relevant=frozenset(raw.get("relevant_doc_keys") or ()),
            filters=dict(raw.get("filters") or {}),
            required_source_tables=frozenset(raw.get("required_source_tables") or ()),
            must_not_sources=frozenset(raw.get("must_not_sources") or ()),
            expect_no_answer=bool(raw.get("expect_no_answer", False)),
        )


def recall_at_k(
    retrieved: Sequence[str], relevant: frozenset, k: int
) -> Optional[float]:
    if not relevant:
        return None
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: frozenset) -> Optional[float]:
    if not relevant:
        return None
    for rank, key in enumerate(retrieved, start=1):
        if key in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: frozenset, k: int) -> Optional[float]:
    """Binary-gain nDCG@k."""
    if not relevant:
        return None
    dcg = sum(
        1.0 / math.log2(i + 2) for i, key in enumerate(retrieved[:k]) if key in relevant
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def is_wrong_source(
    docs: Sequence[Mapping[str, Any]], case: RetrievalCase, k: int = 5
) -> bool:
    """True if the top-k contains a forbidden source or misses a required table."""
    top = list(docs[:k])
    for doc in top:
        if (
            doc_key(doc) in case.must_not_sources
            or str(doc.get("source_table")) in case.must_not_sources
        ):
            return True
    if case.required_source_tables and top:
        tables = {str(d.get("source_table")) for d in top}
        if not tables & case.required_source_tables:
            return True
    return False


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def evaluate_case(
    case: RetrievalCase, docs: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    keys = [doc_key(d) for d in docs]
    return {
        "id": case.id,
        "retrieved": keys[:10],
        "zero_hit": len(docs) == 0,
        "recall@5": recall_at_k(keys, case.relevant, 5),
        "recall@10": recall_at_k(keys, case.relevant, 10),
        "mrr": reciprocal_rank(keys, case.relevant),
        "ndcg@10": ndcg_at_k(keys, case.relevant, 10),
        "wrong_source": is_wrong_source(docs, case),
        # expect_no_answer cases pass only when retrieval returns nothing.
        "no_answer_violation": bool(case.expect_no_answer and docs),
    }


def aggregate(
    rows: Sequence[Mapping[str, Any]], cases: Sequence[RetrievalCase]
) -> Dict[str, Any]:
    answerable = {c.id for c in cases if not c.expect_no_answer}
    ans_rows = [r for r in rows if r["id"] in answerable]
    n = len(rows) or 1
    return {
        "cases": len(rows),
        "recall@5": _mean(r["recall@5"] for r in ans_rows),
        "recall@10": _mean(r["recall@10"] for r in ans_rows),
        "mrr": _mean(r["mrr"] for r in ans_rows),
        "ndcg@10": _mean(r["ndcg@10"] for r in ans_rows),
        "wrong_source_rate": sum(1 for r in rows if r["wrong_source"]) / n,
        "zero_hit_rate": (
            sum(1 for r in ans_rows if r["zero_hit"]) / len(ans_rows)
            if ans_rows
            else 0.0
        ),
        "no_answer_violations": sum(1 for r in rows if r["no_answer_violation"]),
    }


# Direction of "better" per metric; used by regression gating.
HIGHER_IS_BETTER = {"recall@5", "recall@10", "mrr", "ndcg@10"}
LOWER_IS_BETTER = {"wrong_source_rate", "zero_hit_rate", "no_answer_violations"}


def compare_to_baseline(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    tolerance: float = 0.02,
) -> List[str]:
    """Return human-readable regressions (empty list == pass)."""
    problems: List[str] = []
    for name in sorted(HIGHER_IS_BETTER):
        cur, base = current.get(name), baseline.get(name)
        if cur is None or base is None:
            continue
        if cur < base - tolerance:
            problems.append(f"{name} regressed: {cur:.3f} < baseline {base:.3f}")
    for name in sorted(LOWER_IS_BETTER):
        cur, base = current.get(name), baseline.get(name)
        if cur is None or base is None:
            continue
        if cur > base + tolerance:
            problems.append(f"{name} regressed: {cur:.3f} > baseline {base:.3f}")
    return problems
