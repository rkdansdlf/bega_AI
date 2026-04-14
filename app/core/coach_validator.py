"""
Coach 응답 검증 모듈.

COACH_PROMPT_V2의 JSON 출력을 Pydantic 모델로 검증하고 파싱합니다.
LLM 출력의 일관성을 보장하고, 잘못된 형식의 응답을 감지합니다.
"""

import json
import logging
import re
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Union, Tuple
from pydantic import BaseModel, Field, field_validator, BeforeValidator
from typing_extensions import Annotated

logger = logging.getLogger(__name__)
MAX_KEY_METRIC_VALUE_LENGTH = 50
MAX_KEY_METRIC_LABEL_LENGTH = 30
MAX_CRITICAL_METRICS = 2
FALLBACK_HEADLINE = "AI 코치 분석 요약"
MAX_ANALYSIS_TEXT_LENGTH = 240
MAX_ANALYSIS_LIST_ITEMS = 4
MAX_ANALYSIS_LIST_ITEM_LENGTH = 140
MAX_DETAILED_MARKDOWN_LENGTH = 900
MAX_COACH_NOTE_LENGTH = 220
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"[.!?。！？](?:\s|$)")


def _truncate_text(value: Any, *, max_length: int, collapse_whitespace: bool) -> str:
    if not isinstance(value, str):
        return ""

    text = " ".join(value.split()).strip() if collapse_whitespace else value.strip()
    if len(text) <= max_length:
        return text

    hard_limit = max(0, max_length - 3)
    minimum_boundary = max(0, hard_limit // 2)
    boundary_candidates: List[int] = []

    prefix = text[: hard_limit + 1]
    for match in _SENTENCE_BOUNDARY_PATTERN.finditer(prefix):
        boundary = match.end()
        if match.group(0)[-1].isspace():
            boundary -= 1
        if boundary >= minimum_boundary:
            boundary_candidates.append(boundary)

    newline_boundary = prefix.rfind("\n")
    if newline_boundary >= minimum_boundary:
        boundary_candidates.append(newline_boundary)

    space_boundary = prefix.rfind(" ")
    if space_boundary >= minimum_boundary:
        boundary_candidates.append(space_boundary)

    cutoff = max(boundary_candidates) if boundary_candidates else hard_limit
    suffix = text[cutoff:].strip()
    if suffix and text[:cutoff].rstrip().endswith((".", "!", "?", "。", "！", "？")):
        return text[:cutoff].rstrip()
    return text[:cutoff].rstrip() + "..."


def _normalize_short_text(value: Any, *, max_length: int) -> str:
    return _truncate_text(
        value,
        max_length=max_length,
        collapse_whitespace=True,
    )


def _normalize_text_list(
    value: Any,
    *,
    max_items: int = MAX_ANALYSIS_LIST_ITEMS,
    max_item_length: int = MAX_ANALYSIS_LIST_ITEM_LENGTH,
) -> List[str]:
    if not isinstance(value, list):
        return []

    normalized: List[str] = []
    for item in value:
        text = _normalize_short_text(item, max_length=max_item_length)
        if text:
            normalized.append(text)
        if len(normalized) >= max_items:
            break
    return normalized


# ============================================================
# Pydantic Models for Coach Response
# ============================================================


class KeyMetric(BaseModel):
    """핵심 지표 모델"""

    label: str = Field(..., max_length=30, description="지표명")
    # 문자열로 자동 변환 (LLM이 숫자로 주는 경우 대비)
    value: Annotated[Union[str, int, float], BeforeValidator(str)] = Field(
        ..., max_length=MAX_KEY_METRIC_VALUE_LENGTH, description="수치"
    )
    status: Literal["good", "warning", "danger"] = Field(
        default="warning", description="평가 (good/warning/danger)"
    )
    trend: Literal["up", "down", "neutral"] = Field(default="neutral")
    is_critical: bool = Field(default=False)

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: str) -> str:
        """상태 값을 영어로 정규화 (한글→영어 통일)"""
        if not isinstance(v, str):
            return "warning"
        normalized = v.lower().strip()
        if normalized in ["양호", "good", "positive", "최상"]:
            return "good"
        elif normalized in ["주의", "warning", "caution", "보통"]:
            return "warning"
        elif normalized in ["위험", "danger", "critical", "bad"]:
            return "danger"
        return "warning"  # 알 수 없는 값은 warning으로 기본 처리

    @field_validator("trend", mode="before")
    @classmethod
    def normalize_trend(cls, v: str) -> str:
        """추세 값을 허용 스키마로 정규화합니다."""
        if not isinstance(v, str):
            return "neutral"
        normalized = v.lower().strip()
        if normalized in ["up", "상승", "increase", "rising"]:
            return "up"
        if normalized in ["down", "하락", "decrease", "falling"]:
            return "down"
        if normalized in ["neutral", "보합", "유지", "stable", "flat"]:
            return "neutral"
        return "neutral"


class RiskItem(BaseModel):
    """위험 요소 모델"""

    area: str = Field(..., description="영역 (bullpen/starter/batting/defense)")
    level: Literal[0, 1, 2] = Field(..., description="위험도 (0=위험, 1=주의, 2=양호)")
    description: str = Field(..., max_length=150, description="위험 설명 (최대 150자)")

    @field_validator("area", mode="before")
    @classmethod
    def normalize_area(cls, v: str) -> str:
        """영역 값을 영어로 정규화"""
        if not isinstance(v, str):
            return "overall"
        normalized = v.lower().strip()
        area_mapping = {
            "불펜": "bullpen",
            "bullpen": "bullpen",
            "릴리프": "bullpen",
            "선발": "starter",
            "starter": "starter",
            "starting": "starter",
            "타격": "batting",
            "batting": "batting",
            "타선": "batting",
            "수비": "defense",
            "defense": "defense",
            "전체": "overall",
            "overall": "overall",
        }
        return area_mapping.get(normalized, "overall")

    @field_validator("level", mode="before")
    @classmethod
    def normalize_level(cls, v: Any) -> int:
        """위험도 값을 허용 범위(0/1/2)로 보수 정규화"""
        if isinstance(v, bool):
            return 1
        if isinstance(v, (int, float)):
            normalized = int(v)
            return normalized if normalized in (0, 1, 2) else 1
        if isinstance(v, str):
            normalized = v.strip().lower()
            level_mapping = {
                "0": 0,
                "danger": 0,
                "critical": 0,
                "high": 0,
                "위험": 0,
                "1": 1,
                "warning": 1,
                "caution": 1,
                "medium": 1,
                "moderate": 1,
                "주의": 1,
                "2": 2,
                "good": 2,
                "safe": 2,
                "low": 2,
                "양호": 2,
            }
            return level_mapping.get(normalized, 1)
        return 1


class AnalysisSection(BaseModel):
    """분석 섹션 모델"""

    summary: str = Field(default="", max_length=MAX_ANALYSIS_TEXT_LENGTH)
    verdict: str = Field(default="", max_length=MAX_ANALYSIS_TEXT_LENGTH)
    strengths: List[str] = Field(default_factory=list, description="강점 목록")
    weaknesses: List[str] = Field(default_factory=list, description="약점 목록")
    risks: List[RiskItem] = Field(default_factory=list, description="위험 요소 목록")
    why_it_matters: List[str] = Field(
        default_factory=list, description="판단 근거 목록"
    )
    swing_factors: List[str] = Field(default_factory=list, description="승부 변수 목록")
    watch_points: List[str] = Field(
        default_factory=list, description="체크 포인트 목록"
    )
    uncertainty: List[str] = Field(default_factory=list, description="불확실성 목록")

    @field_validator("summary", "verdict", mode="before")
    @classmethod
    def truncate_analysis_text(cls, v: Any) -> str:
        return _normalize_short_text(v, max_length=MAX_ANALYSIS_TEXT_LENGTH)

    @field_validator(
        "strengths",
        "weaknesses",
        "why_it_matters",
        "swing_factors",
        "watch_points",
        "uncertainty",
        mode="before",
    )
    @classmethod
    def normalize_analysis_lists(cls, v: Any) -> List[str]:
        return _normalize_text_list(v)

    @field_validator("risks", mode="before")
    @classmethod
    def limit_risks(cls, v: Any) -> List[Any]:
        if not isinstance(v, list):
            return []
        return v[:3]


class CoachResponse(BaseModel):
    """
    Coach 응답 전체 모델.

    COACH_PROMPT_V2의 JSON 스키마와 일치해야 합니다.
    """

    headline: str = Field(
        ..., min_length=5, max_length=60, description="한 줄 진단 (최대 60자)"
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(default="neutral")
    key_metrics: List[KeyMetric] = Field(
        default_factory=list, description="핵심 지표 목록 (최대 6개)"
    )
    analysis: AnalysisSection = Field(default_factory=AnalysisSection)
    detailed_markdown: str = Field(
        default="",
        max_length=MAX_DETAILED_MARKDOWN_LENGTH,
        description="상세 분석 마크다운",
    )
    coach_note: str = Field(
        default="",
        max_length=MAX_COACH_NOTE_LENGTH,
        description="전략적 제언",
    )

    @field_validator("headline", mode="before")
    @classmethod
    def truncate_headline(cls, v: str) -> str:
        """headline 길이 제한 및 정리"""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("headline은 비어있을 수 없습니다.")
        v = " ".join(v.split()).strip()

        # [Fix] "headline": "Title" 형태의 중복 키 패턴 제거
        # LLM이 JSON 형식을 값 안에 포함시키는 환각 방지
        import re

        dup_pattern = r'^"headline"\s*:\s*"(.*)"$'
        match = re.match(dup_pattern, v)
        if match:
            v = match.group(1)

        # [Fix] 따옴표로 감싸진 경우 제거
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]

        return _truncate_text(v, max_length=60, collapse_whitespace=True)

    @field_validator("detailed_markdown", mode="before")
    @classmethod
    def truncate_markdown(cls, v: str) -> str:
        """detailed_markdown 길이 제한"""
        return _truncate_text(
            v,
            max_length=MAX_DETAILED_MARKDOWN_LENGTH,
            collapse_whitespace=False,
        )

    @field_validator("coach_note", mode="before")
    @classmethod
    def truncate_coach_note(cls, v: str) -> str:
        """coach_note 길이 제한"""
        return _truncate_text(
            v, max_length=MAX_COACH_NOTE_LENGTH, collapse_whitespace=True
        )

    @field_validator("key_metrics", mode="before")
    @classmethod
    def limit_metrics(cls, v: list) -> list:
        """key_metrics 개수 제한 (최대 6개)"""
        if not isinstance(v, list):
            return []
        return v[:6]


# ============================================================
# Parser Functions
# ============================================================


_TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")
_UNQUOTED_JSON_KEY_PATTERN = re.compile(
    r"(?P<prefix>[{,]\s*)(?P<key>[A-Za-z_][A-Za-z0-9_-]*)(?P<suffix>\s*:)"
)
_METRIC_VALUE_TOKEN_PATTERN = re.compile(r"(?P<value>[^\s]*\d[^\s]*)$")
_METRIC_LABEL_PREFIXES = (
    "최대 WPA 변동",
    "정규시즌 OPS",
    "정규시즌 ERA",
    "최근 흐름",
    "폼 진단",
    "불펜 비중",
    "불펜 ERA",
    "선발 ERA",
    "팀 OPS",
    "팀 ERA",
    "OPS",
    "ERA",
)


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (common LLM JSON mistake)."""
    return _TRAILING_COMMA_PATTERN.sub(r"\1", text)


def _quote_unquoted_json_keys(text: str) -> str:
    """Quote bare object keys like {headline: ...} into valid JSON."""
    return _UNQUOTED_JSON_KEY_PATTERN.sub(
        lambda match: (
            f'{match.group("prefix")}"{match.group("key")}"{match.group("suffix")}'
        ),
        text,
    )


def _try_auto_close_json(text: str) -> Optional[str]:
    """Attempt to close incomplete JSON by balancing braces/brackets."""
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            open_braces += 1
        elif ch == "}":
            open_braces -= 1
        elif ch == "[":
            open_brackets += 1
        elif ch == "]":
            open_brackets -= 1

    if open_braces <= 0 and open_brackets <= 0:
        return None  # Already balanced or over-closed

    # Only attempt closing if imbalance is small (likely truncation)
    if open_braces > 3 or open_brackets > 3:
        return None

    closed = text.rstrip().rstrip(",")
    closed += "]" * max(0, open_brackets)
    closed += "}" * max(0, open_braces)
    return closed


def extract_json_from_response(raw_response: str) -> Optional[str]:
    """
    LLM 응답에서 JSON 부분을 추출합니다.

    다양한 형식을 처리합니다:
    - 순수 JSON
    - ```json ... ``` 코드 블록
    - 앞뒤 텍스트가 있는 JSON
    - trailing comma 제거
    - 불완전 JSON 자동 닫기
    """
    if not raw_response:
        return None

    text = _CONTROL_CHAR_PATTERN.sub(" ", raw_response).strip()

    # Case 1: ```json 코드 블록 (prefer the largest dict-looking block)
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(json_block_pattern, text)
    if matches:
        # Pick the first block that looks like a JSON object
        for match in matches:
            candidate = _strip_trailing_commas(match.strip())
            if candidate.startswith("{"):
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass
        # Fall back to first match with trailing comma fix
        return _strip_trailing_commas(matches[0].strip())

    # Try with trailing comma removal first
    cleaned = _strip_trailing_commas(text)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, end_idx = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return cleaned[idx : idx + end_idx]

    # Case 2: { ... } JSON 객체 직접 찾기 (brace matching)
    brace_count = 0
    start_idx = -1
    end_idx = -1

    for i, char in enumerate(cleaned):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                end_idx = i + 1
                break

    if start_idx != -1 and end_idx != -1:
        return cleaned[start_idx:end_idx]

    # Case 3: Incomplete JSON - try auto-closing
    if start_idx != -1 and end_idx == -1:
        fragment = cleaned[start_idx:]
        auto_closed = _try_auto_close_json(fragment)
        if auto_closed:
            try:
                parsed = json.loads(auto_closed)
                if isinstance(parsed, dict):
                    logger.info(
                        "[CoachValidator] Auto-closed incomplete JSON successfully"
                    )
                    return auto_closed
            except json.JSONDecodeError:
                pass

    return None


def _load_json_object_with_repairs(
    json_text: str,
) -> Tuple[Optional[Dict[str, Any]], List[str], Optional[str]]:
    """Load a JSON object with lightweight repairs for common LLM mistakes."""
    candidates: List[Tuple[str, List[str]]] = [(json_text, [])]
    stripped = _strip_trailing_commas(json_text)
    if stripped != json_text:
        candidates.append((stripped, ["strip_trailing_commas"]))
    quoted = _quote_unquoted_json_keys(stripped)
    if quoted != stripped:
        repair_reasons: List[str] = []
        if stripped != json_text:
            repair_reasons.append("strip_trailing_commas")
        repair_reasons.append("quote_unquoted_json_keys")
        candidates.append((quoted, repair_reasons))

    last_error: Optional[str] = None
    seen_candidates = set()
    for candidate, reasons in candidates:
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"JSON decode error: {exc}"
            continue

        if not isinstance(data, dict):
            return None, reasons, "Validation error: root JSON must be an object"
        return data, reasons, None

    return None, [], last_error or "JSON decode error: failed to load object"


def derive_headline(data: Dict[str, Any]) -> str:
    """헤드라인 누락 시 대체 헤드라인을 생성합니다."""
    raw_headline = data.get("headline")
    if isinstance(raw_headline, str) and raw_headline.strip():
        return raw_headline.strip()

    key_metrics = data.get("key_metrics")
    if isinstance(key_metrics, list) and key_metrics:
        first_metric = key_metrics[0]
        if isinstance(first_metric, dict):
            label = str(first_metric.get("label", "")).strip()
            value = str(first_metric.get("value", "")).strip()
            if label and value:
                return f"{label} {value} 중심 분석"
            if label:
                return f"{label} 중심 분석"

    analysis = data.get("analysis")
    if isinstance(analysis, dict):
        for key in ("strengths", "weaknesses"):
            items = analysis.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str) and item.strip():
                        return item.strip()

    return FALLBACK_HEADLINE


def _split_string_metric_item(text: str) -> Tuple[str, str]:
    normalized_text = " ".join(str(text).split()).strip()
    if not normalized_text:
        return "핵심 지표", "데이터 확인"

    for prefix in _METRIC_LABEL_PREFIXES:
        if normalized_text.startswith(prefix):
            remainder = normalized_text[len(prefix) :].strip(" :")
            if remainder:
                return prefix, remainder

    colon_match = re.match(
        r"(?P<label>[^:：]+)\s*[:：]\s*(?P<value>.+)", normalized_text
    )
    if colon_match:
        label = colon_match.group("label").strip()
        value = colon_match.group("value").strip()
        if label and value:
            return label, value

    value_match = _METRIC_VALUE_TOKEN_PATTERN.search(normalized_text)
    if value_match:
        value = value_match.group("value").strip()
        label = normalized_text[: value_match.start()].strip(" -:()")
        if label and value:
            return label, value

    return "핵심 지표", normalized_text


def _infer_metric_status(label: str, value: str) -> str:
    text = f"{label} {value}"
    if re.search(r"하락|부진|부족|열세|불안|과부하|악화|위험", text):
        return "danger"
    if re.search(r"상승|반등|우세|양호|호조|개선", text):
        return "good"
    return "warning"


def _infer_metric_trend(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("+"):
        return "up"
    if stripped.startswith("-"):
        return "down"
    return "neutral"


def coerce_string_key_metrics(data: Dict[str, Any]) -> List[str]:
    """Convert string key_metrics entries into structured metric objects."""
    reasons: List[str] = []
    metrics = data.get("key_metrics")
    if not isinstance(metrics, list):
        return reasons

    for idx, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        raw_text = " ".join(metric.split()).strip()
        if not raw_text:
            metrics[idx] = {
                "label": "핵심 지표",
                "value": "데이터 확인",
                "status": "warning",
                "trend": "neutral",
                "is_critical": False,
            }
            reasons.append(f"coerce_key_metric_{idx}_empty_string")
            continue

        label, value = _split_string_metric_item(raw_text)
        normalized_label = (
            _normalize_short_text(label, max_length=MAX_KEY_METRIC_LABEL_LENGTH)
            or "핵심 지표"
        )
        normalized_value = (
            _normalize_short_text(value, max_length=MAX_KEY_METRIC_VALUE_LENGTH)
            or "데이터 확인"
        )
        metrics[idx] = {
            "label": normalized_label,
            "value": normalized_value,
            "status": _infer_metric_status(normalized_label, normalized_value),
            "trend": _infer_metric_trend(normalized_value),
            "is_critical": "wpa" in raw_text.lower(),
        }
        reasons.append(f"coerce_key_metric_{idx}_from_string")

    return reasons


def normalize_key_metric_values(data: Dict[str, Any]) -> List[str]:
    """key_metrics[].value를 문자열화하고 길이를 제한합니다."""
    reasons: List[str] = []
    metrics = data.get("key_metrics")
    if not isinstance(metrics, list):
        return reasons

    for idx, metric in enumerate(metrics):
        if not isinstance(metric, dict):
            continue
        if "value" not in metric:
            continue
        value = str(metric.get("value", "")).strip()
        if len(value) > MAX_KEY_METRIC_VALUE_LENGTH:
            metric["value"] = value[: MAX_KEY_METRIC_VALUE_LENGTH - 3] + "..."
            reasons.append(f"truncate_key_metrics_value_{idx}")
        else:
            metric["value"] = value
    return reasons


def normalize_critical_flags(
    data: Dict[str, Any], max_critical: int = MAX_CRITICAL_METRICS
) -> List[str]:
    """is_critical=true를 최대 개수만 유지합니다."""
    reasons: List[str] = []
    metrics = data.get("key_metrics")
    if not isinstance(metrics, list):
        return reasons

    critical_indices: List[int] = []
    for idx, metric in enumerate(metrics):
        if isinstance(metric, dict) and bool(metric.get("is_critical")):
            critical_indices.append(idx)

    if len(critical_indices) > max_critical:
        for idx in critical_indices[max_critical:]:
            metric = metrics[idx]
            if isinstance(metric, dict):
                metric["is_critical"] = False
        reasons.append(f"reduce_is_critical_{len(critical_indices)}_to_{max_critical}")
    return reasons


def normalize_risk_levels(data: Dict[str, Any]) -> List[str]:
    """analysis.risks[].level을 허용 범위(0/1/2)로 정규화합니다."""
    reasons: List[str] = []
    analysis = data.get("analysis")
    if not isinstance(analysis, dict):
        return reasons

    risks = analysis.get("risks")
    if not isinstance(risks, list):
        return reasons

    for idx, risk in enumerate(risks):
        if not isinstance(risk, dict) or "level" not in risk:
            continue

        raw_level = risk.get("level")
        normalized_level = RiskItem.normalize_level(raw_level)
        if raw_level != normalized_level:
            risk["level"] = normalized_level
            reasons.append(f"coerce_risk_level_{idx}_to_{normalized_level}")

    return reasons


def normalize_coach_payload(data: dict) -> Tuple[dict, List[str]]:
    """LLM 응답 payload를 검증 전에 정규화합니다."""
    if not isinstance(data, dict):
        return data, []

    normalized = deepcopy(data)
    reasons: List[str] = []

    if not isinstance(normalized.get("key_metrics"), list):
        normalized["key_metrics"] = []
        reasons.append("coerce_key_metrics_to_empty_list")

    reasons.extend(coerce_string_key_metrics(normalized))
    headline = normalized.get("headline")
    if not isinstance(headline, str) or not headline.strip():
        normalized["headline"] = derive_headline(normalized)
        reasons.append("derive_headline")

    reasons.extend(normalize_key_metric_values(normalized))
    reasons.extend(normalize_critical_flags(normalized))
    reasons.extend(normalize_risk_levels(normalized))
    return normalized, reasons


def classify_parse_error(error_message: Optional[str]) -> str:
    """파싱 실패 메시지를 표준 카테고리로 분류합니다."""
    if not error_message:
        return "unknown"

    lowered = error_message.lower()
    if "empty response" in lowered:
        return "empty_response"
    if "no json found" in lowered:
        return "no_json_found"
    if "json decode error" in lowered:
        return "json_decode_error"
    if "headline" in lowered and "field required" in lowered:
        return "schema_missing_headline"
    if (
        "key_metrics" in lowered
        and "value" in lowered
        and ("at most 50" in lowered or "too_long" in lowered)
    ):
        return "metric_value_too_long"
    if "validation error" in lowered:
        return "schema_validation_error"
    return "unknown"


def parse_coach_response_with_meta(
    raw_response: str,
) -> Tuple[Optional[CoachResponse], Optional[str], Dict[str, Any]]:
    """
    parse_coach_response의 확장 버전입니다.
    정규화 적용 여부와 실패 카테고리를 메타데이터로 함께 반환합니다.
    """
    meta: Dict[str, Any] = {
        "normalization_applied": False,
        "normalization_reasons": [],
        "error_code": None,
    }

    if not raw_response or not raw_response.strip():
        error = "Empty response"
        meta["error_code"] = classify_parse_error(error)
        return None, error, meta

    sanitized_response = _CONTROL_CHAR_PATTERN.sub(" ", raw_response)
    if sanitized_response != raw_response:
        meta["normalization_applied"] = True
        meta["normalization_reasons"].append("sanitize_control_chars")

    try:
        json_str = extract_json_from_response(sanitized_response)

        if not json_str and sanitized_response.strip().startswith('"headline"'):
            logger.warning(
                "[CoachValidator] Missing braces detected, attempting to wrap with {}"
            )
            temp_json = "{" + sanitized_response.strip()
            if not temp_json.endswith("}"):
                temp_json += "}"
            try:
                data, repair_reasons, load_error = _load_json_object_with_repairs(
                    temp_json
                )
                if load_error:
                    error = load_error
                    meta["error_code"] = classify_parse_error(error)
                    return None, error, meta
                normalized_data, reasons = normalize_coach_payload(data)
                meta["normalization_reasons"].extend(repair_reasons)
                meta["normalization_reasons"].extend(reasons)
                meta["normalization_applied"] = bool(meta["normalization_reasons"])
                return CoachResponse(**normalized_data), None, meta
            except Exception as e:
                logger.warning(f"Fallback parsing failed: {e}")

        if not json_str:
            error = "No JSON found"
            meta["error_code"] = classify_parse_error(error)
            return None, error, meta

        data, repair_reasons, load_error = _load_json_object_with_repairs(json_str)
        if load_error:
            logger.warning(f"[CoachValidator] {load_error}")
            error = load_error
            meta["error_code"] = classify_parse_error(error)
            return None, error, meta

        meta["normalization_reasons"].extend(repair_reasons)
        normalized_data, reasons = normalize_coach_payload(data)
        meta["normalization_reasons"].extend(reasons)
        meta["normalization_applied"] = bool(meta["normalization_reasons"])

        try:
            return CoachResponse(**normalized_data), None, meta
        except Exception as e:
            logger.warning(f"[CoachValidator] Pydantic validation error: {e}")
            error = f"Validation error: {e}"
            meta["error_code"] = classify_parse_error(error)
            return None, error, meta

    except Exception as e:
        logger.error(f"[CoachValidator] Failed to parse coach response: {e}")
        error = f"Unknown error: {e}"
        meta["error_code"] = classify_parse_error(error)
        return None, error, meta


def parse_coach_response(
    raw_response: str,
) -> Tuple[Optional[CoachResponse], Optional[str]]:
    """
    LLM 응답을 파싱하여 CoachResponse 객체로 변환합니다.

    Args:
        raw_response: LLM의 원시 응답 문자열

    Returns:
        CoachResponse 객체. 파싱 실패 시 None 반환 (캐시 FAILED 처리용).
    """
    response, error, _ = parse_coach_response_with_meta(raw_response)
    return response, error


def validate_coach_response(response: CoachResponse) -> List[str]:
    """
    CoachResponse의 데이터 품질을 검증합니다.

    Returns:
        경고 메시지 목록 (비어있으면 양호)
    """
    warnings = []

    # 핵심 지표 개수 확인
    critical_count = sum(1 for m in response.key_metrics if m.is_critical)
    if critical_count > 2:
        warnings.append(
            f"핵심 지표(is_critical=true)가 {critical_count}개입니다. 최대 2개를 권장합니다."
        )

    # 분석 내용 확인
    if (
        not response.analysis.summary
        and not response.analysis.verdict
        and not response.analysis.strengths
        and not response.analysis.weaknesses
    ):
        warnings.append("강점과 약점이 모두 비어있습니다.")

    if not response.analysis.verdict:
        warnings.append("analysis.verdict가 비어있습니다. 판단 문장을 권장합니다.")

    if not response.analysis.why_it_matters:
        warnings.append(
            "analysis.why_it_matters가 비어있습니다. 판단 근거 보강이 필요합니다."
        )

    # coach_note 길이 확인
    if len(response.coach_note) < 20:
        warnings.append("coach_note가 너무 짧습니다. 구체적인 전략 제언을 권장합니다.")

    # 선수명 포함 여부 확인 (품질 지표)
    all_text = " ".join(
        [
            response.analysis.summary,
            response.analysis.verdict,
            *response.analysis.strengths,
            *response.analysis.weaknesses,
            *response.analysis.why_it_matters,
            *response.analysis.swing_factors,
        ]
    )
    if all_text:
        # 한글 이름 패턴 (2-4글자 한글 이름)
        korean_name_pattern = r"[가-힣]{2,4}"
        if not re.search(korean_name_pattern, all_text):
            warnings.append("분석에 선수명이 포함되지 않았습니다. 구체성이 부족합니다.")

        # 수치 데이터 포함 여부 확인
        number_pattern = r"\d+\.?\d*"
        if not re.search(number_pattern, all_text):
            warnings.append("분석에 수치 데이터가 포함되지 않았습니다.")

    return warnings


def format_coach_response_as_markdown(response: CoachResponse) -> str:
    """
    CoachResponse를 마크다운 형식으로 변환합니다.

    JSON 응답을 프론트엔드에서 렌더링하기 좋은 형식으로 변환합니다.
    """
    parts = []

    # 헤드라인
    sentiment_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(
        response.sentiment, "⚪"
    )
    parts.append(f"## {sentiment_emoji} {response.headline}\n")

    # 핵심 지표 테이블
    if response.key_metrics:
        parts.append("### 핵심 지표")
        parts.append("| 지표 | 수치 | 상태 | 추세 |")
        parts.append("|------|------|------|------|")

        trend_symbol = {"up": "📈", "down": "📉", "neutral": "➡️"}
        for m in response.key_metrics:
            critical_mark = "**" if m.is_critical else ""
            trend = trend_symbol.get(m.trend, "")
            parts.append(
                f"| {critical_mark}{m.label}{critical_mark} | {m.value} | {m.status} | {trend} |"
            )
        parts.append("")

    # 분석 섹션
    if response.analysis.summary:
        parts.append("### 한 줄 판단")
        parts.append(response.analysis.summary)
        parts.append("")

    if response.analysis.verdict:
        parts.append("### 코치 판정")
        parts.append(f"- {response.analysis.verdict}")
        parts.append("")

    if response.analysis.strengths:
        parts.append("### 💪 강점")
        for s in response.analysis.strengths:
            parts.append(f"- {s}")
        parts.append("")

    if response.analysis.weaknesses:
        parts.append("### ⚠️ 약점")
        for w in response.analysis.weaknesses:
            parts.append(f"- {w}")
        parts.append("")

    if response.analysis.risks:
        parts.append("### 🚨 위험 요소")
        risk_emoji = {0: "🔴", 1: "🟡", 2: "🟢"}
        for r in response.analysis.risks:
            emoji = risk_emoji.get(r.level, "⚪")
            parts.append(f"- {emoji} **{r.area}**: {r.description}")
        parts.append("")

    if response.analysis.why_it_matters:
        parts.append("### 왜 중요한가")
        for item in response.analysis.why_it_matters:
            parts.append(f"- {item}")
        parts.append("")

    if response.analysis.swing_factors:
        parts.append("### 승부 변수")
        for item in response.analysis.swing_factors:
            parts.append(f"- {item}")
        parts.append("")

    if response.analysis.watch_points:
        parts.append("### 체크 포인트")
        for item in response.analysis.watch_points:
            parts.append(f"- {item}")
        parts.append("")

    if response.analysis.uncertainty:
        parts.append("### 불확실성")
        for item in response.analysis.uncertainty:
            parts.append(f"- {item}")
        parts.append("")

    # 상세 분석 (이미 마크다운)
    if response.detailed_markdown:
        parts.append(response.detailed_markdown)
        parts.append("")

    # Coach's Note
    if response.coach_note:
        parts.append("### 💡 Coach's Note")
        parts.append(response.coach_note)
        parts.append("")

    return "\n".join(parts)
