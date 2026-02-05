"""
Coach 응답 검증 모듈.

COACH_PROMPT_V2의 JSON 출력을 Pydantic 모델로 검증하고 파싱합니다.
LLM 출력의 일관성을 보장하고, 잘못된 형식의 응답을 감지합니다.
"""

import json
import logging
import re
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, BeforeValidator
from typing_extensions import Annotated

logger = logging.getLogger(__name__)


# ============================================================
# Pydantic Models for Coach Response
# ============================================================

class KeyMetric(BaseModel):
    """핵심 지표 모델"""
    label: str = Field(..., max_length=30, description="지표명")
    # 문자열로 자동 변환 (LLM이 숫자로 주는 경우 대비)
    value: Annotated[Union[str, int, float], BeforeValidator(str)] = Field(
        ..., max_length=20, description="수치"
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
            "불펜": "bullpen", "bullpen": "bullpen", "릴리프": "bullpen",
            "선발": "starter", "starter": "starter", "starting": "starter",
            "타격": "batting", "batting": "batting", "타선": "batting",
            "수비": "defense", "defense": "defense",
            "전체": "overall", "overall": "overall"
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
    headline: str = Field(..., min_length=5, max_length=60, description="한 줄 진단 (최대 60자)")
    sentiment: Literal["positive", "negative", "neutral"] = Field(default="neutral")
    key_metrics: List[KeyMetric] = Field(default_factory=list, description="핵심 지표 목록 (최대 6개)")
    analysis: AnalysisSection = Field(default_factory=AnalysisSection)
    detailed_markdown: str = Field(default="", max_length=10000, description="상세 분석 마크다운 (최대 10000자)")
    coach_note: str = Field(default="", max_length=2000, description="전략적 제언 (최대 2000자)")

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
        """detailed_markdown 길이 제한"""
        if not isinstance(v, str):
            return ""
        v = v.strip()
        if len(v) > 10000:
            v = v[:9997] + "..."
        return v

    @field_validator("coach_note", mode="before")
    @classmethod
    def truncate_coach_note(cls, v: str) -> str:
        """coach_note 길이 제한"""
        if not isinstance(v, str):
            return ""
        v = v.strip()
        if len(v) > 2000:
            v = v[:1997] + "..."
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
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                end_idx = i + 1
                break

    if start_idx != -1 and end_idx != -1:
        return text[start_idx:end_idx]

    return None


def parse_coach_response(raw_response: str) -> Optional[CoachResponse]:
    """
    LLM 응답을 파싱하여 CoachResponse 객체로 변환합니다.

    Args:
        raw_response: LLM의 원시 응답 문자열

    Returns:
        CoachResponse 객체. 파싱 실패 시 None 반환 (캐시 FAILED 처리용).
    """
    if not raw_response or not raw_response.strip():
        logger.warning("[CoachValidator] Empty response received")
        return None

    try:
        # JSON 추출
        json_str = extract_json_from_response(raw_response)
        
        # [Fix] JSON을 못 찾았지만, 텍스트가 "headline": ... 형태로 시작하는 경우 (Missing braces)
        # Solar Pro 모델이 가끔 여는 괄호를 빼먹는 경우 처리
        if not json_str and raw_response.strip().startswith('"headline"'):
            logger.warning("[CoachValidator] Missing braces detected, attempting to wrap with {}")
            # 맨 뒤에 }가 있는지 확인하고 없으면 추가
            temp_json = "{" + raw_response.strip()
            if not temp_json.endswith("}"):
                temp_json += "}"
            
            # 다시 시도
            try:
                data = json.loads(temp_json)
                return CoachResponse(**data)
            except:
                pass # 실패하면 원래 로직대로 진행

        if not json_str:
            logger.warning("[CoachValidator] No JSON found in response")
            return None

        # JSON 파싱
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"[CoachValidator] JSON decode error: {e}")
            return None

        # Pydantic 검증
        return CoachResponse(**data)

    except Exception as e:
        logger.error(f"[CoachValidator] Failed to parse coach response: {e}")
        return None


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
            cleaned_text = cleaned_text[len(prefix):]
    for suffix in ["```", "}"]:
        if cleaned_text.endswith(suffix):
            cleaned_text = cleaned_text[:-len(suffix)]
    cleaned_text = cleaned_text.strip()

    return CoachResponse(
        headline=headline,
        sentiment="neutral",
        key_metrics=[],
        analysis=AnalysisSection(
            strengths=[],
            weaknesses=[],
            risks=[]
        ),
        detailed_markdown="",
        coach_note=cleaned_text[:2000] if cleaned_text else f"형식 변환 실패: {error_reason}"
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
        warnings.append(f"핵심 지표(is_critical=true)가 {critical_count}개입니다. 최대 2개를 권장합니다.")

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
        korean_name_pattern = r'[가-힣]{2,4}'
        if not re.search(korean_name_pattern, all_text):
            warnings.append("분석에 선수명이 포함되지 않았습니다. 구체성이 부족합니다.")

        # 수치 데이터 포함 여부 확인
        number_pattern = r'\d+\.?\d*'
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
