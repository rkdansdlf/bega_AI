"""Low-cardinality HTTP metrics for the FastAPI service."""

from __future__ import annotations

import time
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from app.observability.metrics import (
    AI_HTTP_REQUEST_DURATION_SECONDS,
    AI_HTTP_REQUESTS_TOTAL,
)


_EXCLUDED_PATHS = frozenset({"/metrics", "/metrics/", "/ai/metrics", "/ai/metrics/"})


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    return str(route_path or "unmatched")


def _status_group(status_code: int) -> str:
    group = int(status_code) // 100
    return f"{group}xx" if 1 <= group <= 5 else "other"


def install_http_metrics(app: FastAPI) -> None:
    """Install request metrics once without measuring Prometheus scrapes."""

    if getattr(app.state, "http_metrics_installed", False):
        return
    app.state.http_metrics_installed = True

    @app.middleware("http")
    async def record_http_metrics(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        started_at = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            labels = {
                "method": request.method.upper(),
                "route": _route_template(request),
                "status_group": _status_group(status_code),
            }
            AI_HTTP_REQUESTS_TOTAL.labels(**labels).inc()
            AI_HTTP_REQUEST_DURATION_SECONDS.labels(**labels).observe(
                max(0.0, time.perf_counter() - started_at)
            )
