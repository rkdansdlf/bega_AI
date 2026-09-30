"""Shared wire contract for unavailable AI dependencies."""

from __future__ import annotations

from typing import Literal

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class AIDependencyUnavailableResponse(BaseModel):
    code: Literal["AI_DEPENDENCY_UNAVAILABLE"]
    message: str
    retryable: bool


class AIDependencyUnavailable(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable


async def ai_dependency_unavailable_handler(
    _request: Request,
    exception: AIDependencyUnavailable,
) -> JSONResponse:
    payload = AIDependencyUnavailableResponse(
        code="AI_DEPENDENCY_UNAVAILABLE",
        message=exception.message,
        retryable=exception.retryable,
    )
    return JSONResponse(status_code=503, content=payload.model_dump())
