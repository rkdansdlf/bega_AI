from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Literal

import httpx
from google import genai
from google.genai import types as genai_types
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from typing_extensions import TypedDict

from ..config import Settings, get_settings
from ..core.http_clients import get_shared_httpx_client
from ..deps import require_ai_internal_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/moderation", tags=["moderation"])

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


class ModerationResult(TypedDict):
    category: str
    reason: str
    action: Literal["ALLOW", "BLOCK"]
    decisionSource: Literal["RULE", "MODEL", "FALLBACK"]
    riskLevel: Literal["LOW", "MEDIUM", "HIGH"]


@router.post("/safety-check")
async def safety_check(
    payload: Dict[str, Any] = Body(...),
    _: None = Depends(require_ai_internal_token),
) -> ModerationResult:
    """
    게시글/댓글 텍스트를 점진 Hybrid 정책(RULE + MODEL)으로 검사합니다.
    """
    content = str(payload.get("content", "")).strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content is required")

    settings = get_settings()
    rule_result = _evaluate_rule(content, settings)

    provider = settings.llm_provider
    provider_configured = bool(
        (provider == "openrouter" and settings.openrouter_api_key)
        or (provider == "gemini" and settings.gemini_api_key)
    )
    if not provider_configured:
        logger.warning(
            "%s moderation API key 미설정 상태입니다. RULE 기반 FALLBACK으로 처리합니다.",
            provider,
        )
        fallback_result = _fallback_from_rule(rule_result)
        _log_result(fallback_result)
        return fallback_result

    try:
        if provider == "openrouter":
            model_result = await _evaluate_openrouter_model(content, settings)
        else:
            model_result = await run_in_threadpool(_evaluate_model, content, settings)
        final_result = _merge_rule_and_model(rule_result, model_result)
        _log_result(final_result)
        return final_result
    except Exception:
        logger.exception(
            "모델 기반 moderation 실패. RULE 기반 FALLBACK으로 전환합니다."
        )
        fallback_result = _fallback_from_rule(rule_result)
        _log_result(fallback_result)
        return fallback_result


def _evaluate_rule(content: str, settings: Settings) -> ModerationResult:
    normalized = content.lower()
    high_risk_keywords = settings.moderation_high_risk_keywords
    spam_keywords = settings.moderation_spam_keywords

    if _contains_any(normalized, high_risk_keywords):
        return _build_result(
            category="INAPPROPRIATE",
            reason="안전 정책에 따라 제한되는 표현이 감지되었습니다.",
            action="BLOCK",
            decision_source="RULE",
            risk_level="HIGH",
        )

    spam_score = 0
    url_count = len(URL_PATTERN.findall(normalized))
    if url_count >= settings.moderation_spam_url_threshold:
        spam_score += 2

    if _has_repeated_chars(normalized, settings.moderation_repeated_char_threshold):
        spam_score += 2

    if _contains_any(normalized, spam_keywords):
        spam_score += 1

    if spam_score >= settings.moderation_spam_block_score:
        return _build_result(
            category="SPAM",
            reason="스팸 가능성이 높은 패턴이 감지되었습니다.",
            action="BLOCK",
            decision_source="RULE",
            risk_level="HIGH",
        )

    if spam_score >= settings.moderation_spam_medium_score:
        return _build_result(
            category="SPAM",
            reason="주의 패턴이 감지되었습니다.",
            action="ALLOW",
            decision_source="RULE",
            risk_level="MEDIUM",
        )

    return _build_result(
        category="SAFE",
        reason="",
        action="ALLOW",
        decision_source="RULE",
        risk_level="LOW",
    )


def _build_moderation_prompt(content: str) -> str:
    return f"""
    당신은 KBO(한국 프로야구) 커뮤니티의 콘텐츠 관리자입니다.
    다음 텍스트를 분석해 JSON으로만 답변하세요.

    분류 카테고리:
    - SAFE: 정상 콘텐츠
    - INAPPROPRIATE: 욕설/혐오/공격적 표현
    - SPAM: 광고/도배/홍보
    - SPOILER: 경기 핵심 결과 스포일러

    텍스트: "{content}"

    아래 JSON 스키마만 반환:
    {{
      "category": "SAFE" | "INAPPROPRIATE" | "SPAM" | "SPOILER",
      "reason": "한국어로 된 짧은 사유",
      "action": "ALLOW" | "BLOCK",
      "riskLevel": "LOW" | "MEDIUM" | "HIGH"
    }}
    """


def _model_result_from_text(response_text: str) -> ModerationResult:
    parsed = _parse_json_payload(response_text)

    action = _normalize_action(str(parsed.get("action", "ALLOW")))
    risk_level = _normalize_risk_level(str(parsed.get("riskLevel", "LOW")), action)
    category = str(parsed.get("category", "SAFE")).strip().upper()
    reason = str(parsed.get("reason", "")).strip()

    return _build_result(
        category=category or "SAFE",
        reason=reason,
        action=action,
        decision_source="MODEL",
        risk_level=risk_level,
    )


async def _evaluate_openrouter_model(
    content: str, settings: Settings
) -> ModerationResult:
    model = str(settings.openrouter_model or "").strip()
    if not model:
        raise RuntimeError("OpenRouter moderation model is not configured")

    client = get_shared_httpx_client(
        "moderation-openrouter",
        timeout=httpx.Timeout(30.0, connect=10.0, read=20.0, pool=5.0),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )
    response = await client.post(
        f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.openrouter_referer or "https://kbo-platform.com",
            "X-Title": settings.openrouter_app_title or "KBO Platform Moderation",
        },
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": _build_moderation_prompt(content)}
            ],
            "temperature": 0.0,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        },
    )
    response.raise_for_status()
    payload = response.json()
    response_text = payload["choices"][0]["message"]["content"]
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError("OpenRouter moderation response content is empty")
    return _model_result_from_text(response_text)


def _evaluate_model(content: str, settings: Settings) -> ModerationResult:
    prompt = _build_moderation_prompt(content)

    with genai.Client(api_key=settings.gemini_api_key) as client:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
    return _model_result_from_text(getattr(response, "text", ""))


def _merge_rule_and_model(
    rule_result: ModerationResult, model_result: ModerationResult
) -> ModerationResult:
    if rule_result["action"] == "BLOCK":
        return _build_result(
            category=rule_result["category"],
            reason=rule_result["reason"],
            action="BLOCK",
            decision_source="RULE",
            risk_level="HIGH",
        )

    if model_result["action"] == "BLOCK":
        return _build_result(
            category=model_result["category"],
            reason=model_result["reason"],
            action="BLOCK",
            decision_source="MODEL",
            risk_level=_normalize_risk_level(model_result["riskLevel"], "BLOCK"),
        )

    if rule_result["riskLevel"] == "MEDIUM":
        return _build_result(
            category="SPAM",
            reason="주의 패턴이 감지되었습니다.",
            action="ALLOW",
            decision_source="RULE",
            risk_level="MEDIUM",
        )

    return _build_result(
        category=model_result["category"],
        reason=model_result["reason"],
        action="ALLOW",
        decision_source="MODEL",
        risk_level=_normalize_risk_level(model_result["riskLevel"], "ALLOW"),
    )


def _fallback_from_rule(rule_result: ModerationResult) -> ModerationResult:
    if rule_result["riskLevel"] == "HIGH":
        return _build_result(
            category=rule_result["category"],
            reason="안전 정책에 따라 게시가 제한되었습니다.",
            action="BLOCK",
            decision_source="FALLBACK",
            risk_level="HIGH",
        )

    if rule_result["riskLevel"] == "MEDIUM":
        return _build_result(
            category="SPAM",
            reason="주의 패턴이 감지되어 규칙 기반으로 처리되었습니다.",
            action="ALLOW",
            decision_source="FALLBACK",
            risk_level="MEDIUM",
        )

    return _build_result(
        category="SAFE",
        reason="규칙 기반 검사를 통과했습니다.",
        action="ALLOW",
        decision_source="FALLBACK",
        risk_level="LOW",
    )


def _build_result(
    category: str,
    reason: str,
    action: Literal["ALLOW", "BLOCK"],
    decision_source: Literal["RULE", "MODEL", "FALLBACK"],
    risk_level: Literal["LOW", "MEDIUM", "HIGH"],
) -> ModerationResult:
    return {
        "category": category,
        "reason": reason,
        "action": action,
        "decisionSource": decision_source,
        "riskLevel": risk_level,
    }


def _parse_json_payload(payload: str) -> Dict[str, Any]:
    payload = (payload or "").strip()
    if not payload:
        raise ValueError("Empty moderation payload")

    try:
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("Moderation payload is not an object")
        return parsed
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(payload[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Moderation payload is not an object")
        return parsed


def _normalize_action(raw_action: str) -> Literal["ALLOW", "BLOCK"]:
    return "BLOCK" if raw_action.strip().upper() == "BLOCK" else "ALLOW"


def _normalize_risk_level(
    raw_risk_level: str, action: Literal["ALLOW", "BLOCK"]
) -> Literal["LOW", "MEDIUM", "HIGH"]:
    normalized = raw_risk_level.strip().upper()
    if normalized in {"LOW", "MEDIUM", "HIGH"}:
        return normalized  # type: ignore[return-value]
    return "HIGH" if action == "BLOCK" else "LOW"


def _contains_any(content: str, keywords: list[str]) -> bool:
    return any(keyword in content for keyword in keywords)


def _has_repeated_chars(content: str, threshold: int) -> bool:
    min_repeats = max(2, threshold)
    pattern = re.compile(r"(.)\1{" + str(min_repeats - 1) + ",}")
    return pattern.search(content) is not None


def _log_result(result: ModerationResult) -> None:
    logger.info(
        "Moderation completed: action=%s source=%s risk=%s category=%s",
        result["action"],
        result["decisionSource"],
        result["riskLevel"],
        result["category"],
    )
