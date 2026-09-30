"""Explicit RAG storage/profile selection and pool identity contract."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

import psycopg
from fastapi import Request
from fastapi.responses import JSONResponse
from psycopg_pool import PoolTimeout


class RagBackend(str, Enum):
    ORACLE = "oracle"
    POSTGRES = "postgres"
    FAKE = "fake"


class RagProfile(str, Enum):
    TEST = "test"
    LOCAL_ORACLE = "local-oracle"
    LOCAL_POSTGRES = "local-postgres"
    PRODUCTION = "production"


class RagConfigurationError(RuntimeError):
    """Safe configuration failure that never includes a credential-bearing URL."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RagDependencyUnavailable(RuntimeError):
    """Expected RAG dependency failure with a secret-safe wire contract."""

    def __init__(
        self,
        code: str,
        message: str = "RAG dependency is unavailable",
        *,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


async def rag_dependency_unavailable_handler(
    _request: Request,
    exception: RagDependencyUnavailable,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "code": exception.code,
            "message": exception.message,
            "retryable": exception.retryable,
        },
    )


def classify_rag_dependency_error(exception: Exception) -> str | None:
    """Classify known DB dependency failures; leave code defects as 500s."""

    if isinstance(exception, RagConfigurationError):
        return exception.code
    if isinstance(
        exception,
        (
            PoolTimeout,
            psycopg.OperationalError,
            psycopg.InterfaceError,
            TimeoutError,
            OSError,
        ),
    ):
        return "RAG_STORAGE_UNAVAILABLE"
    if isinstance(
        exception,
        (
            psycopg.errors.UndefinedTable,
            psycopg.errors.UndefinedColumn,
            psycopg.errors.InvalidSchemaName,
        ),
    ):
        return "RAG_SCHEMA_NOT_READY"
    if isinstance(
        exception,
        (
            psycopg.errors.UndefinedFunction,
            psycopg.errors.UndefinedObject,
        ),
    ):
        return "RAG_VECTOR_CAPABILITY_NOT_READY"

    # python-oracledb exposes the numeric ORA code on exception.args[0].code.
    oracle_code = None
    if getattr(exception, "args", None):
        oracle_code = getattr(exception.args[0], "code", None)
    if oracle_code in {942, 4043}:
        return "RAG_SCHEMA_NOT_READY"
    if oracle_code in {29855, 29902, 51803, 51804, 51805}:
        return "RAG_VECTOR_CAPABILITY_NOT_READY"
    if isinstance(oracle_code, int):
        return "RAG_STORAGE_UNAVAILABLE"
    return None


@dataclass(frozen=True)
class RagSelection:
    profile: RagProfile
    backend: RagBackend
    url: str | None


_LOCAL_RAG_HOSTS = {
    "localhost",
    "127.0.0.1",
    "::1",
    "postgres",
    "pgvector-db",
    "oracle",
    "oracle-db",
}

_PROFILE_BACKENDS = {
    RagProfile.LOCAL_ORACLE: {RagBackend.ORACLE},
    RagProfile.LOCAL_POSTGRES: {RagBackend.POSTGRES},
    RagProfile.PRODUCTION: {RagBackend.ORACLE},
    # A test may use a fake or an explicitly isolated local database.
    RagProfile.TEST: {RagBackend.FAKE, RagBackend.ORACLE, RagBackend.POSTGRES},
}


def _required_enum(value: str | None, enum_type: type[Enum], *, code: str, key: str):
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise RagConfigurationError(code, f"{key} must be configured explicitly")
    try:
        return enum_type(normalized)
    except ValueError as exc:  # Settings normally rejects this first.
        raise RagConfigurationError(
            "RAG_CONFIGURATION_INVALID",
            f"{key} contains an unsupported value",
        ) from exc


def _selected_url(settings: Any, backend: RagBackend) -> str | None:
    if backend is RagBackend.FAKE:
        return None
    if backend is RagBackend.ORACLE:
        return (getattr(settings, "ai_rag_db_url", None) or "").strip() or None
    # Deliberately exclude OCI_DB_URL and SUPABASE_DB_URL. RAG storage may use
    # only the explicit split URL or the canonical PostgreSQL setting.
    return (
        getattr(settings, "ai_rag_db_url", None)
        or getattr(settings, "postgres_db_url", None)
        or ""
    ).strip() or None


def _validate_url_scheme(url: str, backend: RagBackend) -> None:
    scheme = urlsplit(url).scheme.lower()
    expected = (
        {"oracle", "oracle+oracledb"}
        if backend is RagBackend.ORACLE
        else {"postgres", "postgresql"}
    )
    if scheme not in expected:
        raise RagConfigurationError(
            "RAG_BACKEND_URL_MISMATCH",
            "RAG storage URL scheme does not match RAG_BACKEND",
        )


def _validate_local_url(url: str) -> None:
    try:
        host = (urlsplit(url).hostname or "").strip().lower()
    except ValueError as exc:
        raise RagConfigurationError(
            "RAG_STORAGE_URL_INVALID",
            "RAG storage URL is invalid",
        ) from exc
    if host not in _LOCAL_RAG_HOSTS and not host.endswith(".local"):
        raise RagConfigurationError(
            "RAG_REMOTE_ENDPOINT_FORBIDDEN",
            "local/test RAG profiles may use local endpoints only",
        )


def resolve_rag_selection(settings: Any) -> RagSelection:
    """Resolve exactly one backend without URL inference or cross-backend fallback."""

    profile = _required_enum(
        getattr(settings, "rag_profile", None),
        RagProfile,
        code="RAG_PROFILE_NOT_SELECTED",
        key="RAG_PROFILE",
    )
    backend = _required_enum(
        getattr(settings, "rag_backend", None),
        RagBackend,
        code="RAG_BACKEND_NOT_SELECTED",
        key="RAG_BACKEND",
    )
    if backend not in _PROFILE_BACKENDS[profile]:
        raise RagConfigurationError(
            "RAG_PROFILE_BACKEND_MISMATCH",
            "RAG_PROFILE and RAG_BACKEND are not compatible",
        )

    app_env = str(getattr(settings, "app_env", "") or "").strip().lower()
    production_env = app_env in {"prod", "production"}
    if production_env != (profile is RagProfile.PRODUCTION):
        raise RagConfigurationError(
            "RAG_PROFILE_ENV_MISMATCH",
            "RAG_PROFILE does not match APP_ENV",
        )

    url = _selected_url(settings, backend)
    if backend is not RagBackend.FAKE:
        if not url:
            raise RagConfigurationError(
                "RAG_STORAGE_URL_MISSING",
                "the selected RAG backend requires an explicit storage URL",
            )
        _validate_url_scheme(url, backend)
        if profile is not RagProfile.PRODUCTION:
            _validate_local_url(url)

    return RagSelection(profile=profile, backend=backend, url=url)


class FakeRagConnection:
    backend = "fake"


class FakeRagPool:
    """No-network pool used only by the explicit test profile."""

    backend = "fake"

    def __init__(self, selection: RagSelection) -> None:
        if selection.backend is not RagBackend.FAKE:
            raise ValueError("FakeRagPool requires the fake backend")
        self.selection = selection
        self.network_connection_attempts = 0

    async def open(self, *, wait: bool = True, timeout: float | None = None) -> None:
        del wait, timeout

    @asynccontextmanager
    async def connection(
        self, timeout: float | None = None
    ) -> AsyncIterator[FakeRagConnection]:
        del timeout
        yield FakeRagConnection()

    async def close(self) -> None:
        return None

    def get_stats(self) -> dict[str, Any]:
        return {"backend": self.backend, "network_connection_attempts": 0}


class PostgresRagPool:
    """Tag the psycopg pool so every RAG consumer sees one backend identity."""

    backend = "postgres"

    def __init__(self, pool: Any, selection: RagSelection) -> None:
        self._pool = pool
        self.selection = selection

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)

    async def open(self, *args: Any, **kwargs: Any) -> Any:
        return await self._pool.open(*args, **kwargs)

    def connection(self, *args: Any, **kwargs: Any) -> Any:
        return self._pool.connection(*args, **kwargs)

    async def close(self) -> Any:
        return await self._pool.close()

    def get_stats(self) -> dict[str, Any]:
        get_stats = getattr(self._pool, "get_stats", None)
        return get_stats() if callable(get_stats) else {}
