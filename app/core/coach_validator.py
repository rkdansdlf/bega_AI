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
MAX_CRITICAL_METRICS = 2
FALLBACK_HEADLINE = "AI 코치 분석 요약"


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


class AnalysisSection(BaseModel):
    """분석 섹션 모델"""

    strengths: List[str] = Field(default_factory=list, description="강점 목록")
    weaknesses: List[str] = Field(default_factory=list, description="약점 목록")
    risks: List[RiskItem] = Field(default_factory=list, description="위험 요소 목록")


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
        default="", max_length=500, description="상세 분석 마크다운 (최대 500자)"
    )
    coach_note: str = Field(
        default="", max_length=120, description="전략적 제언 (최대 120자)"
    )

    @field_validator("headline", mode="before")
    @classmethod
    def truncate_headline(cls, v: str) -> str:
        """headline 길이 제한 및 정리"""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("headline은 비어있을 수 없습니다.")
        v = v.strip()

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

        if len(v) > 60:
            v = v[:57] + "..."
        return v

    @field_validator("detailed_markdown", mode="before")
    @classmethod
    def truncate_markdown(cls, v: str) -> str:
        """detailed_markdown 길이 제한 (프롬프트 규칙: 최대 500자)"""
        if not isinstance(v, str):
            return ""
        v = v.strip()
        if len(v) > 500:
            v = v[:497] + "..."
        return v

    @field_validator("coach_note", mode="before")
    @classmethod
    def truncate_coach_note(cls, v: str) -> str:
        """coach_note 길이 제한 (프롬프트 규칙: 최대 120자)"""
        if not isinstance(v, str):
            return ""
        v = v.strip()
        if len(v) > 120:
            v = v[:117] + "..."
        return v

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


def extract_json_from_response(raw_response: str) -> Optional[str]:
    """
    LLM 응답에서 JSON 부분을 추출합니다.

    다양한 형식을 처리합니다:
    - 순수 JSON
    - ```json ... ``` 코드 블록
    - 앞뒤 텍스트가 있는 JSON
    """
    if not raw_response:
        return None

    text = raw_response.strip()

    # Case 1: ```json 코드 블록
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(json_block_pattern, text)
    if matches:
        return matches[0].strip()

    # Case 2: { ... } JSON 객체 직접 찾기
    # 가장 바깥쪽 중괄호 매칭
    brace_count = 0
    start_idx = -1
    end_idx = -1

    for i, char in enumerate(text):
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
        return text[start_idx:end_idx]

    return None


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


def normalize_coach_payload(data: dict) -> Tuple[dict, List[str]]:
    """LLM 응답 payload를 검증 전에 정규화합니다."""
    if not isinstance(data, dict):
        return data, []

    normalized = deepcopy(data)
    reasons: List[str] = []

    if not isinstance(normalized.get("key_metrics"), list):
        normalized["key_metrics"] = []
        reasons.append("coerce_key_metrics_to_empty_list")

    headline = normalized.get("headline")
    if not isinstance(headline, str) or not headline.strip():
        normalized["headline"] = derive_headline(normalized)
        reasons.append("derive_headline")

    reasons.extend(normalize_key_metric_values(normalized))
    reasons.extend(normalize_critical_flags(normalized))
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

    try:
        json_str = extract_json_from_response(raw_response)

        if not json_str and raw_response.strip().startswith('"headline"'):
            logger.warning(
                "[CoachValidator] Missing braces detected, attempting to wrap with {}"
            )
            temp_json = "{" + raw_response.strip()
            if not temp_json.endswith("}"):
                temp_json += "}"
            try:
                data = json.loads(temp_json)
                if not isinstance(data, dict):
                    error = "Validation error: root JSON must be an object"
                    meta["error_code"] = classify_parse_error(error)
                    return None, error, meta
                normalized_data, reasons = normalize_coach_payload(data)
                meta["normalization_reasons"] = reasons
                meta["normalization_applied"] = len(reasons) > 0
                return CoachResponse(**normalized_data), None, meta
            except Exception as e:
                logger.warning(f"Fallback parsing failed: {e}")

        if not json_str:
            error = "No JSON found"
            meta["error_code"] = classify_parse_error(error)
            return None, error, meta

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"[CoachValidator] JSON decode error: {e}")
            error = f"JSON decode error: {e}"
            meta["error_code"] = classify_parse_error(error)
            return None, error, meta

        if not isinstance(data, dict):
            error = "Validation error: root JSON must be an object"
            meta["error_code"] = classify_parse_error(error)
            return None, error, meta

        normalized_data, reasons = normalize_coach_payload(data)
        meta["normalization_reasons"] = reasons
        meta["normalization_applied"] = len(reasons) > 0

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


def _create_fallback_response(error_reason: str, original_text: str) -> CoachResponse:
    """
    파싱 실패 시 원본 텍스트를 보존하는 fallback 응답 생성.

    에러가 발생해도 사용자에게 최소한의 정보를 제공합니다.
    """
    # 원본 텍스트에서 의미 있는 첫 줄 추출 시도
    first_meaningful_line = ""
    if original_text:
        for line in original_text.strip().split("\n"):
            cleaned = line.strip()
            # [Fix] '{', '```', '#' 뿐만 아니라 '"headline"' 같은 JSON fragment도 무시
            if cleaned and not cleaned.startswith(("{", "```", "#", '"')):
                first_meaningful_line = cleaned[:100]  # 최대 100자
                break

    headline = first_meaningful_line or "AI 분석 결과"

    # 원본 텍스트 정리 (마크다운 코드블록, JSON 잔해 제거)
    cleaned_text = original_text.strip() if original_text else ""
    for prefix in ["```json", "```", "{"]:
        if cleaned_text.startswith(prefix):
            cleaned_text = cleaned_text[len(prefix) :]
    for suffix in ["```", "}"]:
        if cleaned_text.endswith(suffix):
            cleaned_text = cleaned_text[: -len(suffix)]
    cleaned_text = cleaned_text.strip()

    return CoachResponse(
        headline=headline,
        sentiment="neutral",
        key_metrics=[],
        analysis=AnalysisSection(strengths=[], weaknesses=[], risks=[]),
        detailed_markdown="",
        coach_note=(
            cleaned_text[:2000] if cleaned_text else f"형식 변환 실패: {error_reason}"
        ),
    )


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
    if not response.analysis.strengths and not response.analysis.weaknesses:
        warnings.append("강점과 약점이 모두 비어있습니다.")

    # coach_note 길이 확인
    if len(response.coach_note) < 20:
        warnings.append("coach_note가 너무 짧습니다. 구체적인 전략 제언을 권장합니다.")

    # 선수명 포함 여부 확인 (품질 지표)
    all_text = " ".join(response.analysis.strengths + response.analysis.weaknesses)
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
