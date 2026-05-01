"""
Coach grounding helpers.

The Coach 응답이 확인된 사실 범위를 벗어나지 않도록
허용 엔티티, 수치, 결측 경고를 fact sheet로 묶고 검증합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Dict, Iterable, List, Sequence, Set

NUMERIC_TOKEN_RE = re.compile(r"(?<![A-Za-z])(?:\d+\.\d+|\d+|\.\d+)%?")
GROUNDING_DISCLAIMER_TOKENS = (
    "미정",
    "발표 전",
    "미발표",
    "확정 전",
    "확정되지",
    "데이터 부족",
    "부족",
    "확인 필요",
    "제한",
    "보수",
    "가능성",
    "변수",
)
SERIES_REFERENCE_TOKENS = (
    "시리즈",
    "차전",
    "전적",
    "1승",
    "2승",
    "3승",
    "4승",
)
GENERIC_EMPTY_HEADLINES = {
    "AI 코치 분석 요약",
    "AI 코치 브리핑",
    "전략 코치 리뷰",
}


@dataclass(frozen=True)
class CoachFactSheet:
    fact_lines: List[str]
    caveat_lines: List[str]
    allowed_entity_names: Set[str]
    allowed_numeric_tokens: Set[str]
    supported_fact_count: int
    starters_confirmed: bool
    lineup_confirmed: bool
    series_context_confirmed: bool
    require_series_context: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CoachGroundingValidationResult:
    warnings: List[str]
    reasons: List[str]
    unsupported_numeric_tokens: List[str] = field(default_factory=list)


def _strip_numeric_token(token: str) -> str:
    return token.strip().rstrip(".,)")


def _normalize_decimal_text(text: str) -> str | None:
    candidate = text.strip()
    if not candidate:
        return None
    if candidate.startswith("."):
        candidate = f"0{candidate}"

    try:
        normalized = format(Decimal(candidate).normalize(), "f")
    except (InvalidOperation, ValueError):
        return None

    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def expand_numeric_token_variants(token: str) -> Set[str]:
    cleaned = _strip_numeric_token(token)
    if not cleaned:
        return set()

    suffix = "%" if cleaned.endswith("%") else ""
    core = cleaned[:-1] if suffix else cleaned
    variants = {cleaned}
    normalized = _normalize_decimal_text(core)
    if normalized is not None:
        variants.add(normalized)
        if normalized.startswith("0."):
            variants.add(normalized[1:])
        if suffix:
            variants.add(f"{normalized}%")
    if suffix:
        variants.add(core)
    return {value for value in variants if value}


def collect_numeric_tokens(text: str) -> Set[str]:
    tokens: Set[str] = set()
    for raw_token in NUMERIC_TOKEN_RE.findall(text or ""):
        tokens.update(expand_numeric_token_variants(raw_token))
    return tokens


def format_coach_fact_sheet(fact_sheet: CoachFactSheet) -> str:
    lines: List[str] = [
        "## FACT SHEET",
        "- 아래 FACT SHEET에 적힌 사실만 사용해 응답을 재작성하세요.",
        "- FACT SHEET에 없는 선수명, 수치, 경기 맥락은 절대 단정하지 마세요.",
        "- 판단 문장은 확인된 근거와 경기 운영 의미가 함께 보이도록 짧게 연결하세요.",
    ]
    if fact_sheet.fact_lines:
        lines.append("")
        lines.append("### 확인된 사실")
        lines.extend(f"- {line}" for line in fact_sheet.fact_lines)
    if fact_sheet.caveat_lines:
        lines.append("")
        lines.append("### 결측 및 주의")
        lines.extend(f"- {line}" for line in fact_sheet.caveat_lines)
        lines.append(
            "- 위 한계 중 경기 해석에 가장 큰 영향을 주는 항목은 `analysis.uncertainty`나 상세 마크다운의 한계 섹션에 반영하세요."
        )
    if fact_sheet.allowed_entity_names:
        lines.append("")
        lines.append("### 허용 엔티티")
        lines.append("- " + ", ".join(sorted(fact_sheet.allowed_entity_names)[:32]))
    return "\n".join(lines)


def _collect_response_text_segments(response_data: Dict[str, Any]) -> List[str]:
    analysis = response_data.get("analysis") or {}
    key_metrics = response_data.get("key_metrics") or []
    segments: List[str] = [
        str(response_data.get("headline") or ""),
        str(response_data.get("detailed_markdown") or ""),
        str(response_data.get("coach_note") or ""),
        str(analysis.get("summary") or ""),
        str(analysis.get("verdict") or ""),
        " ".join(str(item) for item in analysis.get("strengths", []) or []),
        " ".join(str(item) for item in analysis.get("weaknesses", []) or []),
        " ".join(
            str(item.get("description") or "")
            for item in analysis.get("risks", []) or []
        ),
        " ".join(str(item) for item in analysis.get("why_it_matters", []) or []),
        " ".join(str(item) for item in analysis.get("swing_factors", []) or []),
        " ".join(str(item) for item in analysis.get("watch_points", []) or []),
        " ".join(str(item) for item in analysis.get("uncertainty", []) or []),
        " ".join(
            str(item.get("value") or "")
            for item in key_metrics
            if isinstance(item, dict)
        ),
    ]
    return [segment for segment in segments if segment]


def response_numeric_tokens(response_data: Dict[str, Any]) -> Set[str]:
    combined = " ".join(_collect_response_text_segments(response_data))
    return collect_numeric_tokens(combined)


def _strip_markdown_scaffold(text: str) -> str:
    kept_lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("## "):
            continue
        line = re.sub(r"^[-*]\s*", "", line).strip()
        if line:
            kept_lines.append(line)
    return " ".join(kept_lines).strip()


def response_is_semantically_empty(response_data: Dict[str, Any]) -> bool:
    analysis = response_data.get("analysis") or {}
    key_metrics = response_data.get("key_metrics") or []

    has_key_metric = any(
        isinstance(metric, dict)
        and (
            str(metric.get("label") or "").strip()
            or str(metric.get("value") or "").strip()
        )
        for metric in key_metrics
    )
    if has_key_metric:
        return False

    text_fields = [
        str(analysis.get("summary") or "").strip(),
        str(analysis.get("verdict") or "").strip(),
        str(response_data.get("coach_note") or "").strip(),
        _strip_markdown_scaffold(response_data.get("detailed_markdown") or ""),
    ]
    if any(text_fields):
        return False

    for key in (
        "strengths",
        "weaknesses",
        "why_it_matters",
        "swing_factors",
        "watch_points",
        "uncertainty",
    ):
        items = analysis.get(key) or []
        if any(str(item or "").strip() for item in items):
            return False

    risks = analysis.get("risks") or []
    if any(
        isinstance(item, dict) and str(item.get("description") or "").strip()
        for item in risks
    ):
        return False

    headline = str(response_data.get("headline") or "").strip()
    return not headline or headline in GENERIC_EMPTY_HEADLINES


def has_grounding_disclaimer(text: str) -> bool:
    return any(token in text for token in GROUNDING_DISCLAIMER_TOKENS)


def detect_unconfirmed_data_claims(
    fact_sheet: CoachFactSheet,
    response_payload: Dict[str, Any],
) -> List[str]:
    warnings: List[str] = []
    response_text = " ".join(_collect_response_text_segments(response_payload))

    if (
        not fact_sheet.lineup_confirmed
        and "라인업" in response_text
        and not has_grounding_disclaimer(response_text)
    ):
        warnings.append("라인업 미발표 경기에서 확정 표현 가능성")

    if (
        not fact_sheet.starters_confirmed
        and "선발" in response_text
        and not has_grounding_disclaimer(response_text)
    ):
        warnings.append("선발 미확정 경기에서 확정 표현 가능성")

    if (
        fact_sheet.require_series_context
        and not fact_sheet.series_context_confirmed
        and any(token in response_text for token in SERIES_REFERENCE_TOKENS)
        and not has_grounding_disclaimer(response_text)
    ):
        warnings.append("시리즈 맥락 부족 경기에서 확정 표현 가능성")

    return warnings


def validate_response_against_fact_sheet(
    response_data: Dict[str, Any],
    fact_sheet: CoachFactSheet,
) -> CoachGroundingValidationResult:
    warnings: List[str] = []
    reasons: List[str] = []

    if response_is_semantically_empty(response_data):
        reasons.append("empty_response")
        warnings.append(
            "LLM 응답에 실질적인 분석 내용이 없어 재시도 또는 보수 요약으로 전환합니다."
        )

    unsupported_numeric_tokens = sorted(
        token
        for token in response_numeric_tokens(response_data)
        if token not in fact_sheet.allowed_numeric_tokens
    )
    if unsupported_numeric_tokens:
        reasons.append("unsupported_numeric_claim")
        warnings.append(
            "근거 fact sheet에 없는 수치가 감지되었습니다: "
            + ", ".join(unsupported_numeric_tokens[:6])
        )

    claim_warnings = detect_unconfirmed_data_claims(fact_sheet, response_data)
    if claim_warnings:
        warnings.extend(claim_warnings)
        if any("라인업" in item for item in claim_warnings):
            reasons.append("unconfirmed_lineup_claim")
        if any("선발" in item for item in claim_warnings):
            reasons.append("unconfirmed_starter_claim")
        if any("시리즈" in item for item in claim_warnings):
            reasons.append("unconfirmed_series_claim")

    return CoachGroundingValidationResult(
        warnings=warnings,
        reasons=list(dict.fromkeys(reasons)),
        unsupported_numeric_tokens=unsupported_numeric_tokens,
    )


def extend_numeric_tokens(values: Iterable[str], bucket: Set[str]) -> None:
    for value in values:
        bucket.update(collect_numeric_tokens(value))
