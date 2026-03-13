"""릴리스 결정 초안용 deterministic eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .release_decision_agent import ReleaseDecisionDraft


class ReleaseDecisionEvalCase(BaseModel):
    case_id: str
    scenario: str
    task_prompt: str
    seed_paths: list[str] = Field(default_factory=list)
    expected_decision: Literal["GO", "NO_GO", "PENDING"]
    required_keywords: list[str] = Field(default_factory=list)
    required_sources: list[str] = Field(default_factory=list)


class ReleaseDecisionEvalResult(BaseModel):
    case_id: str
    status: Literal["PASS", "FAIL"]
    decision_ok: bool
    keyword_hits: dict[str, bool]
    source_hits: dict[str, bool]
    missing_keywords: list[str]
    missing_sources: list[str]


def load_eval_cases(path: Path) -> list[ReleaseDecisionEvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("eval cases file must contain a JSON list")
    return [ReleaseDecisionEvalCase.model_validate(item) for item in payload]


def evaluate_release_decision(
    draft: ReleaseDecisionDraft,
    case: ReleaseDecisionEvalCase,
) -> ReleaseDecisionEvalResult:
    cited_sources = {item.source.strip().lower() for item in draft.evidence}
    corpus = "\n".join(
        [
            draft.summary,
            *draft.blockers,
            *draft.risks,
            *draft.next_actions,
            *(item.claim for item in draft.evidence),
            *(item.excerpt for item in draft.evidence),
        ]
    ).lower()

    decision_ok = draft.decision == case.expected_decision
    keyword_hits = {
        keyword: keyword.lower() in corpus for keyword in case.required_keywords
    }
    source_hits = {
        source: source.lower() in cited_sources for source in case.required_sources
    }
    missing_keywords = [key for key, hit in keyword_hits.items() if not hit]
    missing_sources = [key for key, hit in source_hits.items() if not hit]
    status = (
        "PASS"
        if decision_ok and not missing_keywords and not missing_sources
        else "FAIL"
    )
    return ReleaseDecisionEvalResult(
        case_id=case.case_id,
        status=status,
        decision_ok=decision_ok,
        keyword_hits=keyword_hits,
        source_hits=source_hits,
        missing_keywords=missing_keywords,
        missing_sources=missing_sources,
    )
