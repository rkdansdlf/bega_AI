from __future__ import annotations

import logging
from dataclasses import dataclass

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeprecatedApiOperation:
    method: str
    path: str
    canonical_method: str
    canonical_path: str


DEPRECATED_API_OPERATIONS = (
    DeprecatedApiOperation("POST", "/coach/analyze", "POST", "/ai/coach/analyze"),
    DeprecatedApiOperation(
        "POST", "/coach/cache/reset", "POST", "/ai/coach/cache/reset"
    ),
    DeprecatedApiOperation(
        "POST", "/coach/analyze-legacy", "POST", "/ai/coach/analyze"
    ),
    DeprecatedApiOperation(
        "POST", "/ai/coach/analyze-legacy", "POST", "/ai/coach/analyze"
    ),
    DeprecatedApiOperation("POST", "/vision/ticket", "POST", "/ai/vision/ticket"),
    DeprecatedApiOperation(
        "POST",
        "/vision/seat-view-classify",
        "POST",
        "/ai/vision/seat-view-classify",
    ),
    DeprecatedApiOperation("GET", "/ai/chat/stream", "POST", "/ai/chat/stream"),
)

_DEPRECATED_BY_METHOD_PATH = {
    (operation.method, operation.path): operation
    for operation in DEPRECATED_API_OPERATIONS
}


class LegacyApiDeprecationMiddleware:
    """Attach migration metadata without buffering normal or SSE responses."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        operation = _DEPRECATED_BY_METHOD_PATH.get(
            (str(scope.get("method", "")).upper(), str(scope.get("path", "")))
        )
        if operation is None:
            await self.app(scope, receive, send)
            return

        logger.warning(
            "deprecated_api_operation method=%s legacy_path=%s "
            "canonical_method=%s canonical_path=%s",
            operation.method,
            operation.path,
            operation.canonical_method,
            operation.canonical_path,
        )

        async def send_with_deprecation_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Deprecation"] = "true"
                headers["X-Deprecation"] = "true"
                if "X-Legacy-Endpoint" not in headers:
                    headers["X-Legacy-Endpoint"] = (
                        f"{operation.method} {operation.path}"
                    )
                headers.append(
                    "Link",
                    f'<{operation.canonical_path}>; rel="successor-version"',
                )
            await send(message)

        await self.app(scope, receive, send_with_deprecation_headers)
