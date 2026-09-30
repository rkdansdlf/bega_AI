"""Scope/provenance equivalence checks for semantic-cache candidates.

Token overlap and matching numbers are necessary but not sufficient: "2025 LG"
and "2024 LG" answers overlap heavily and can even share numbers. These checks
compare *what the answer is about* (season, team), *where it came from*
(sources), *when* (as_of_date) and *whether it answers at all*.
"""

from __future__ import annotations

import re
from typing import Any, List, Mapping, Optional, Set

from .entity_extractor import TEAM_MAPPING

_YEAR_RE = re.compile(r"(?<!\d)(19[89]\d|20\d{2})(?!\d)")
NON_ANSWER_MARKERS = (
    "MANUAL_BASEBALL_DATA_REQUIRED",
    "확인할 수 없습니다",
    "데이터가 부족",
    "제공할 수 없습니다",
)


def extract_years(text: str) -> Set[int]:
    return {int(y) for y in _YEAR_RE.findall(text or "")}


def extract_teams(text: str) -> Set[str]:
    # longest alias first so "기아타이거즈" is not shadowed by shorter aliases
    found: Set[str] = set()
    remaining = text or ""
    for alias in sorted(TEAM_MAPPING, key=len, reverse=True):
        if alias in remaining:
            found.add(TEAM_MAPPING[alias])
            remaining = remaining.replace(alias, " ")
    return found


def is_non_answer(text: str) -> bool:
    return any(marker in (text or "") for marker in NON_ANSWER_MARKERS)


def _source_ids(value: Any) -> Set[str]:
    ids: Set[str] = set()
    for item in value or ():
        if isinstance(item, Mapping):
            key = (
                item.get("id")
                or item.get("source_uri")
                or f"{item.get('source_table')}:{item.get('source_row_id')}"
            )
            ids.add(str(key))
        elif item is not None:
            ids.add(str(item))
    return ids


def check_equivalence(
    *,
    request_question: str = "",
    cached_question: str = "",
    cached_answer: str = "",
    fresh_answer: str = "",
    cached_provenance: Optional[Mapping[str, Any]] = None,
    fresh_provenance: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Return failure reasons (empty list == equivalent)."""
    reasons: List[str] = []

    if request_question and cached_question:
        if extract_years(request_question) != extract_years(cached_question):
            reasons.append("season_scope_mismatch")
        if extract_teams(request_question) != extract_teams(cached_question):
            reasons.append("team_scope_mismatch")

    if request_question:
        asked_years = extract_years(request_question)
        answer_years = extract_years(cached_answer)
        if asked_years and answer_years and not asked_years & answer_years:
            reasons.append("answer_season_mismatch")
        asked_teams = extract_teams(request_question)
        answer_teams = extract_teams(cached_answer)
        if asked_teams and answer_teams and not asked_teams & answer_teams:
            reasons.append("answer_team_mismatch")

    if is_non_answer(cached_answer) != is_non_answer(fresh_answer):
        reasons.append("answerability_mismatch")

    cp, fp = cached_provenance or {}, fresh_provenance or {}
    for field, reason in (
        ("data_sources", "source_provenance_mismatch"),
        ("answer_sources", "source_provenance_mismatch"),
    ):
        a, b = _source_ids(cp.get(field)), _source_ids(fp.get(field))
        if a and b and a != b and reason not in reasons:
            reasons.append(reason)

    a_date, b_date = cp.get("as_of_date"), fp.get("as_of_date")
    if a_date and b_date and str(a_date) != str(b_date):
        reasons.append("freshness_mismatch")

    return reasons
