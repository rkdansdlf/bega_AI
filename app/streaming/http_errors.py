"""Safe ordinary-HTTP errors for AI stream setup failures."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.contracts.stream_events_v2 import AiStreamHttpError


class AiStreamHttpException(Exception):
    def __init__(
        self,
        status_code: int,
        error: AiStreamHttpError,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(error.code)
        self.status_code = status_code
        self.error = error
        self.headers = dict(headers or {})


async def ai_stream_http_exception_handler(
    _request: Request,
    exception: AiStreamHttpException,
) -> JSONResponse:
    return JSONResponse(
        status_code=exception.status_code,
        content=exception.error.model_dump(mode="json"),
        headers=exception.headers,
    )


def install_ai_stream_http_error_handler(app: FastAPI) -> None:
    app.add_exception_handler(
        AiStreamHttpException,
        ai_stream_http_exception_handler,
    )
