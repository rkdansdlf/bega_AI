"""Deterministic claim ↔ source grounding and hallucination metrics.

No LLM judge: a sentence is *supported* by a source when every number in it
appears in that source's text and its content tokens overlap enough. This is
deliberately strict on numbers, which is where baseball answers go wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

_NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")
_TOKEN_RE = re.compile(r"[가-힣A-Za-z]{2,}")
_STOP = frozenset({"입니다", "합니다", "있습니다", "했습니다", "그리고", "하지만"})
MIN_TOKEN_OVERLAP = 0.5


def extract_numbers(text: str) -> Set[str]:
    """Normalised numbers: '1,234' -> '1234', '0.300' -> '0.3', '.300' kept as '300'."""
    out: Set[str] = set()
    for raw in _NUM_RE.findall(text or ""):
        cleaned = raw.replace(",", "")
        if "." in cleaned:
            cleaned = cleaned.rstrip("0").rstrip(".")
        out.add(cleaned)
    return out


def _tokens(text: str) -> Set[str]:
    return {t for t in _TOKEN_RE.findall(text or "") if t not in _STOP}


def _covered(token: str, source_text: str) -> bool:
    """Token appears in the source, tolerating trailing Korean particles
    (승률은 → 승률, 기록했습니다 → 기록)."""
    if token in source_text:
        return True
    for cut in (1, 2):
        stem = token[:-cut]
        if len(stem) >= 2 and stem in source_text:
            return True
    return False


def split_claims(answer: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(answer or "") if len(s.strip()) >= 4]


@dataclass
class ClaimGrounding:
    claim: str
    supporting_sources: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    unsupported_numbers: List[str] = field(default_factory=list)
    supported: bool = False


def ground_claims(answer: str, sources: Mapping[str, str]) -> List[ClaimGrounding]:
    """Map each answer sentence to the source ids that support it.

    ``sources`` maps a source id (e.g. ``"team_summary:LG|2025"``) to its text.
    """
    src_numbers = {sid: extract_numbers(text) for sid, text in sources.items()}
    all_numbers: Set[str] = set().union(*src_numbers.values()) if src_numbers else set()
    results: List[ClaimGrounding] = []
    for claim in split_claims(answer):
        nums = extract_numbers(claim)
        toks = _tokens(claim)
        supporting: List[str] = []
        for sid in sources:
            if not nums <= src_numbers[sid]:
                continue
            overlap = (
                sum(1 for t in toks if _covered(t, sources[sid])) / len(toks)
                if toks
                else 1.0
            )
            if overlap >= MIN_TOKEN_OVERLAP:
                supporting.append(sid)
        results.append(
            ClaimGrounding(
                claim=claim,
                supporting_sources=supporting,
                numbers=sorted(nums),
                unsupported_numbers=sorted(nums - all_numbers),
                supported=bool(supporting),
            )
        )
    return results


def citation_scores(
    cited: Iterable[str], required: Iterable[str], available: Iterable[str]
) -> Dict[str, Optional[float]]:
    """Precision: cited ids that exist in context. Recall: required ids cited."""
    cited_set, required_set, avail = set(cited), set(required), set(available)
    precision = len(cited_set & avail) / len(cited_set) if cited_set else None
    recall = len(cited_set & required_set) / len(required_set) if required_set else None
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "phantom_citations": sorted(cited_set - avail),
    }


def evaluate_generation(
    case: Mapping[str, Any], answer: str, cited_sources: Sequence[str]
) -> Dict[str, Any]:
    """Score one recorded answer against a generation golden case.

    case keys: sources{id:text}, required_sources[], required_numbers[],
    forbidden_numbers[], forbidden_phrases[], allowed_entities[]?,
    known_entities[]?, expect_non_answer, non_answer_markers[]
    """
    sources: Mapping[str, str] = case.get("sources") or {}
    grounding = ground_claims(answer, sources)
    answer_numbers = extract_numbers(answer)
    context_numbers: Set[str] = set()
    for text in sources.values():
        context_numbers |= extract_numbers(text)

    numeric_hallucinations = sorted(answer_numbers - context_numbers)
    missing_numbers = sorted(set(case.get("required_numbers") or ()) - answer_numbers)
    forbidden_hits = sorted(
        {n for n in case.get("forbidden_numbers") or () if n in answer_numbers}
        | {p for p in case.get("forbidden_phrases") or () if p in answer}
    )
    known = set(case.get("known_entities") or ())
    context_blob = " ".join(sources.values())
    entity_hallucinations = sorted(
        e for e in known if e in answer and e not in context_blob
    )
    markers = case.get("non_answer_markers") or ()
    # A refusal sentence makes no factual claim, so it cannot be unsupported.
    unsupported = [
        g.claim
        for g in grounding
        if not g.supported and not any(m in g.claim for m in markers)
    ]
    non_answer = any(m in answer for m in markers)
    expect_non = bool(case.get("expect_non_answer"))

    scores = citation_scores(cited_sources, case.get("required_sources") or (), sources)
    passed = (
        not numeric_hallucinations
        and not entity_hallucinations
        and not forbidden_hits
        and not missing_numbers
        and not scores["phantom_citations"]
        and (non_answer if expect_non else not non_answer)
    )
    return {
        "id": case.get("id"),
        "passed": passed,
        "claims": len(grounding),
        "unsupported_claims": unsupported,
        "numeric_hallucinations": numeric_hallucinations,
        "entity_hallucinations": entity_hallucinations,
        "missing_numbers": missing_numbers,
        "forbidden_hits": forbidden_hits,
        "non_answer_correct": non_answer if expect_non else not non_answer,
        **scores,
        "grounding": [
            {"claim": g.claim, "sources": g.supporting_sources} for g in grounding
        ],
    }


def aggregate_generation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows) or 1
    claims = sum(r["claims"] for r in rows) or 1
    prec = [
        r["citation_precision"] for r in rows if r["citation_precision"] is not None
    ]
    rec = [r["citation_recall"] for r in rows if r["citation_recall"] is not None]
    return {
        "cases": len(rows),
        "pass_rate": sum(1 for r in rows if r["passed"]) / n,
        "unsupported_claim_rate": sum(len(r["unsupported_claims"]) for r in rows)
        / claims,
        "numeric_hallucination_rate": sum(
            1 for r in rows if r["numeric_hallucinations"]
        )
        / n,
        "entity_hallucination_rate": sum(1 for r in rows if r["entity_hallucinations"])
        / n,
        "citation_precision": sum(prec) / len(prec) if prec else None,
        "citation_recall": sum(rec) / len(rec) if rec else None,
    }


def build_runtime_grounding(
    answer: str,
    docs: Sequence[Mapping[str, Any]],
    *,
    max_claims: int = 30,
    max_source_chars: int = 4000,
) -> Dict[str, Any]:
    """Claim → ``rag_chunks`` id mapping for a live answer.

    Claims with no supporting chunk are reported (not removed): the caller
    decides whether to surface, flag, or gate on them.
    """
    sources = {
        str(doc["id"]): str(doc.get("content") or doc.get("title") or "")[
            :max_source_chars
        ]
        for doc in docs
        if doc.get("id") is not None
    }
    grounded = ground_claims(answer, sources)[:max_claims]
    unsupported_numbers = sorted({n for g in grounded for n in g.unsupported_numbers})
    unsupported = [g for g in grounded if not g.supported]
    return {
        "claims": [
            {
                "claim": g.claim,
                "source_ids": g.supporting_sources,
                "supported": g.supported,
            }
            for g in grounded
        ],
        "total_claims": len(grounded),
        "unsupported_claims": len(unsupported),
        "unsupported_numbers": unsupported_numbers,
        "coverage": (
            round((len(grounded) - len(unsupported)) / len(grounded), 4)
            if grounded
            else None
        ),
    }
