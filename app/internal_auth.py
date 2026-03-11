"""경량 내부 토큰 인증 dependency."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request, status

from .config import get_settings
from .core.security_metrics import record_security_event


def _extract_internal_token_from_authorization(authorization: str) -> str:
    candidate = (authorization or "").strip()
    if not candidate:
        return ""
    if candidate.lower().startswith("bearer "):
        return candidate[7:].strip()
    return candidate


def require_ai_internal_token(
    request: Request,
    x_internal_api_key: str = Header(default="", alias="X-Internal-Api-Key"),
    authorization: str = Header(default="", alias="Authorization"),
) -> None:
    settings = get_settings()
    expected_token = (getattr(settings, "resolved_ai_internal_token", "") or "").strip()
    endpoint = request.url.path if request is not None else "unknown"

    if not expected_token:
        record_security_event(
            "AI_INTERNAL_AUTH_MISCONFIGURED",
            endpoint=endpoint,
            detail="missing_ai_internal_token",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI internal authentication is not configured",
        )

    provided_token = (
        (x_internal_api_key or "").strip()
        or _extract_internal_token_from_authorization(authorization)
    )
    if not provided_token or not secrets.compare_digest(provided_token, expected_token):
        record_security_event(
            "AI_INTERNAL_AUTH_REJECT",
            endpoint=endpoint,
            detail="missing_or_invalid_token",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API token",
        )
