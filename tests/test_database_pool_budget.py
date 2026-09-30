"""Database pool budget configuration tests."""

import pytest

from app.config import Settings


def test_default_pool_allocation_fits_budget(monkeypatch) -> None:
    for name in (
        "AI_DB_POOL_MAX_SIZE",
        "AI_INGEST_DB_POOL_MAX_SIZE",
        "AI_BASEBALL_DB_POOL_MAX_SIZE",
        "AI_RAG_DB_POOL_MAX_SIZE",
        "AI_DB_CONNECTION_BUDGET",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.configured_db_connection_total == 32
    settings.validate_database_pool_budget()


def test_pool_allocation_over_budget_fails_fast(monkeypatch) -> None:
    monkeypatch.setenv("AI_DB_POOL_MAX_SIZE", "20")
    monkeypatch.setenv("AI_DB_CONNECTION_BUDGET", "20")

    settings = Settings()

    with pytest.raises(RuntimeError, match="exceeds AI_DB_CONNECTION_BUDGET"):
        settings.validate_database_pool_budget()


def test_pool_factories_use_configured_limits(monkeypatch) -> None:
    from app import deps

    settings = Settings(
        app_env="test",
        rag_profile="test",
        rag_backend="postgres",
        ai_rag_db_url="postgresql://local:local@postgres:5432/test",
        ingest_worker_enabled=False,
        db_pool_max_size=11,
        ingest_db_pool_max_size=2,
        baseball_db_pool_max_size=5,
        rag_db_pool_max_size=7,
        db_connection_budget=25,
    )
    created: list[tuple[int, int, str | None]] = []

    def fake_create(*, min_size: int, max_size: int, conninfo=None):
        created.append((min_size, max_size, conninfo))
        return object()

    monkeypatch.setattr(deps, "get_settings", lambda: settings)
    monkeypatch.setattr(deps, "_create_async_connection_pool", fake_create)
    monkeypatch.setattr(
        deps, "_format_connection_pool_stats", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(deps, "_connection_pool", None)
    monkeypatch.setattr(deps, "_ingest_connection_pool", None)
    monkeypatch.setattr(deps, "_baseball_connection_pool", None)
    monkeypatch.setattr(deps, "_rag_connection_pool", None)

    deps.get_connection_pool()
    deps.get_ingest_connection_pool()
    deps.get_baseball_connection_pool()
    deps.get_rag_connection_pool()

    assert [entry[1] for entry in created] == [11, 2, 5, 7]
