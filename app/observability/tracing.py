"""Request correlation + lightweight spans (HTTP → retrieval → tool → LLM).

* ``request_id_var`` carries the id for the current request. It is bound into
  structlog contextvars (so every log line has it), tagged on the Sentry scope,
  and echoed in the ``X-Request-ID`` response header so the BFF/frontend can
  join logs across services.
* ``span()`` opens a Sentry child span when Sentry is initialised and an
  OpenTelemetry span when ``opentelemetry`` happens to be installed; otherwise
  it is a no-op. No hard dependency is added.
"""

from __future__ import annotations

import functools
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Iterator, Optional

from app.core.request_scope import begin_request_scope, end_request_scope

REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex


def sanitize_request_id(raw: Optional[str]) -> str:
    """Accept a caller-supplied id only if it is a short safe token."""
    if raw and _SAFE_ID.match(raw.strip()):
        return raw.strip()
    return new_request_id()


def current_request_id() -> Optional[str]:
    return request_id_var.get()


def _bind(request_id: str) -> None:
    try:
        import structlog

        structlog.contextvars.bind_contextvars(request_id=request_id)
    except Exception:  # noqa: BLE001
        pass
    try:
        import sentry_sdk

        sentry_sdk.set_tag("request_id", request_id)
    except Exception:  # noqa: BLE001
        pass


def _unbind() -> None:
    try:
        import structlog

        structlog.contextvars.unbind_contextvars("request_id")
    except Exception:  # noqa: BLE001
        pass


class RequestIdMiddleware:
    """Pure-ASGI middleware (safe with SSE streaming responses)."""

    def __init__(self, app: Callable) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        raw = headers.get(REQUEST_ID_HEADER.lower().encode(), b"").decode(
            "latin-1", "ignore"
        )
        request_id = sanitize_request_id(raw)
        token = request_id_var.set(request_id)
        scope_token = begin_request_scope()
        _bind(request_id)

        async def send_with_id(message: dict) -> None:
            if message.get("type") == "http.response.start":
                hdrs = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() != REQUEST_ID_HEADER.lower().encode()
                ]
                hdrs.append((REQUEST_ID_HEADER.lower().encode(), request_id.encode()))
                message = {**message, "headers": hdrs}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            end_request_scope(scope_token)
            request_id_var.reset(token)
            _unbind()


@contextmanager
def span(op: str, name: str, **data: Any) -> Iterator[None]:
    """Child span named ``op``/``name``; attributes go in ``data``."""
    sentry_cm = None
    otel_cm = None
    try:
        import sentry_sdk

        if sentry_sdk.get_client().is_active():
            sentry_cm = sentry_sdk.start_span(op=op, name=name)
    except Exception:  # noqa: BLE001
        sentry_cm = None
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]

        otel_cm = trace.get_tracer("bega_ai").start_as_current_span(f"{op}.{name}")
    except Exception:  # noqa: BLE001
        otel_cm = None

    entered = []
    try:
        for cm in (sentry_cm, otel_cm):
            if cm is None:
                continue
            obj = cm.__enter__()
            entered.append((cm, obj))
            for key, value in data.items():
                try:
                    if hasattr(obj, "set_data"):
                        obj.set_data(key, value)
                    elif hasattr(obj, "set_attribute"):
                        obj.set_attribute(key, value)
                except Exception:  # noqa: BLE001
                    pass
        rid = request_id_var.get()
        if rid:
            for _, obj in entered:
                try:
                    if hasattr(obj, "set_data"):
                        obj.set_data("request_id", rid)
                except Exception:  # noqa: BLE001
                    pass
        yield
    except BaseException as exc:
        for cm, _ in reversed(entered):
            try:
                cm.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:  # noqa: BLE001
                pass
        raise
    else:
        for cm, _ in reversed(entered):
            try:
                cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass


def traced(op: str, name: Optional[str] = None) -> Callable:
    """Decorator form of :func:`span` for ``async def`` functions."""

    def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        span_name = name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(op, span_name):
                return await fn(*args, **kwargs)

        return wrapper

    return decorator
