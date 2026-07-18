"""Shared HTTP client registry for outbound AI-service requests."""

from __future__ import annotations

import asyncio
import inspect
from threading import RLock

import httpx

_client_lock = RLock()
_shared_clients: dict[str, httpx.AsyncClient] = {}


def _is_reusable_client(client: object) -> bool:
    if client is None:
        return False

    if getattr(client, "is_closed", False):
        return False

    return callable(getattr(client, "aclose", None))


def _build_client(
    *,
    timeout: httpx.Timeout | float,
    limits: httpx.Limits | None,
    follow_redirects: bool,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=follow_redirects,
    )


def get_shared_httpx_client(
    name: str,
    *,
    timeout: httpx.Timeout | float,
    limits: httpx.Limits | None = None,
    follow_redirects: bool = False,
) -> httpx.AsyncClient:
    """Return a lazily created shared AsyncClient for a stable use case."""
    with _client_lock:
        client = _shared_clients.get(name)
        if not _is_reusable_client(client):
            client = _build_client(
                timeout=timeout,
                limits=limits,
                follow_redirects=follow_redirects,
            )
            _shared_clients[name] = client
        return client


async def close_shared_httpx_clients() -> None:
    """Close and forget every shared AsyncClient."""
    with _client_lock:
        clients = list(_shared_clients.values())
        _shared_clients.clear()

    if not clients:
        return

    close_coroutines = []
    for client in clients:
        async_close = getattr(client, "aclose", None)
        if callable(async_close):
            result = async_close()
            if inspect.isawaitable(result):
                close_coroutines.append(result)
            continue

        sync_close = getattr(client, "close", None)
        if callable(sync_close):
            result = sync_close()
            if inspect.isawaitable(result):
                close_coroutines.append(result)

    if close_coroutines:
        await asyncio.gather(*close_coroutines)


def reset_shared_httpx_clients_for_tests() -> None:
    """테스트 격리용. 공유 클라이언트 레지스트리를 동기적으로 비운다.

    선행 테스트가 다른 이벤트 루프에 바인딩된 클라이언트를 레지스트리에 남기면
    이후 테스트의 재사용/종료 불변식이 깨질 수 있다. ``aclose()``를 호출하지 않고
    레지스트리만 비워 테스트를 순서 독립적으로 만든다.
    """
    with _client_lock:
        _shared_clients.clear()
