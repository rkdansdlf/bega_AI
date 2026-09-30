"""Per-request scratch space shared by every task the request spawns.

A ``ContextVar`` holding one *mutable* dict is shared by child tasks (they copy
the context, not the dict). That is what lets all retrievals of one chat request
— including parallel multi-query / HyDE tasks created with ``asyncio.gather`` —
agree on a single embedding generation even if an operator activates another one
mid-request. Outside a request (scripts, tests, workers) there is no scope and
callers fall back to their normal, unpinned behaviour.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Dict, Optional

_request_scope: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "request_scope", default=None
)


def begin_request_scope() -> Token:
    return _request_scope.set({})


def end_request_scope(token: Token) -> None:
    _request_scope.reset(token)


def current_scope() -> Optional[Dict[str, Any]]:
    return _request_scope.get()
