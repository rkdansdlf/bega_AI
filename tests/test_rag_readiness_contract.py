from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.core import rag_readiness
from app.core.rag_runtime import FakeRagPool, resolve_rag_selection


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "RAG_BACKEND",
        "RAG_PROFILE",
        "AI_RAG_DB_URL",
        "POSTGRES_DB_URL",
        "OCI_DB_URL",
        "SUPABASE_DB_URL",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "EMBED_PROVIDER",
        "EMBED_MODEL",
        "EMBED_DIM",
        "AI_INGEST_WORKER_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)


def _settings(**updates: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "app_env": "test",
        "rag_profile": "test",
        "rag_backend": "fake",
        "embed_provider": "local",
        "embed_dim": 1536,
        "ingest_worker_enabled": False,
        "postgres_db_url": "postgresql://local:local@postgres:5432/test",
    }
    values.update(updates)
    return Settings(**values)


def _ready_storage_components() -> dict[str, dict[str, object]]:
    return {
        "rag_storage": {"status": "UP", "code": "RAG_STORAGE_READY"},
        "rag_schema": {"status": "UP", "code": "RAG_SCHEMA_READY"},
        "rag_vector": {"status": "UP", "code": "RAG_VECTOR_READY"},
        "rag_index": {"status": "UP", "code": "RAG_VECTOR_INDEX_READY"},
    }


@pytest.mark.asyncio
async def test_fake_backend_can_report_up_without_a_network_connection() -> None:
    settings = _settings()
    pool = FakeRagPool(resolve_rag_selection(settings))

    report = await rag_readiness.build_rag_readiness_report(settings, pool)

    assert report["status"] == "UP"
    assert report["components"]["rag_backend"] == {
        "status": "UP",
        "backend": "fake",
        "profile": "test",
    }
    assert report["components"]["embedding"]["status"] == "UP"
    assert pool.network_connection_attempts == 0


@pytest.mark.asyncio
async def test_readiness_rejects_pool_identity_different_from_selection() -> None:
    settings = _settings()
    wrong_pool = SimpleNamespace(backend="postgres")

    report = await rag_readiness.build_rag_readiness_report(settings, wrong_pool)

    assert report["status"] == "NOT_READY"
    assert report["components"]["rag_backend"]["code"] == (
        "RAG_BACKEND_IDENTITY_MISMATCH"
    )


@pytest.mark.parametrize(
    ("component", "code"),
    [
        ("rag_storage", "RAG_STORAGE_UNAVAILABLE"),
        ("rag_schema", "RAG_SCHEMA_NOT_READY"),
        ("rag_vector", "RAG_VECTOR_CAPABILITY_NOT_READY"),
        ("rag_index", "RAG_VECTOR_INDEX_NOT_READY"),
    ],
)
@pytest.mark.asyncio
async def test_each_storage_dependency_makes_readiness_not_ready(
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    code: str,
) -> None:
    settings = _settings(
        app_env="local",
        rag_profile="local-postgres",
        rag_backend="postgres",
        ai_rag_db_url="postgresql://local:local@postgres:5432/rag",
    )
    pool = SimpleNamespace(backend="postgres")
    components = _ready_storage_components()
    components[component] = {"status": "DOWN", "code": code}

    async def probe(_pool, _settings, _selection):
        return components

    monkeypatch.setattr(rag_readiness, "_probe_selected_backend", probe)

    report = await rag_readiness.build_rag_readiness_report(settings, pool)

    assert report["status"] == "NOT_READY"
    assert report["components"][component] == {"status": "DOWN", "code": code}


@pytest.mark.asyncio
async def test_missing_embedding_configuration_is_a_component_503_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(embed_provider="openrouter", openrouter_api_key=None)
    pool = FakeRagPool(resolve_rag_selection(settings))

    report = await rag_readiness.build_rag_readiness_report(settings, pool)

    assert report["status"] == "NOT_READY"
    assert report["components"]["embedding"] == {
        "status": "DOWN",
        "code": "RAG_EMBEDDING_NOT_READY",
        "provider": "openrouter",
        "dimension": 1536,
    }


@pytest.mark.asyncio
async def test_probe_failure_does_not_leak_a_credential_bearing_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        app_env="local",
        rag_profile="local-postgres",
        rag_backend="postgres",
        ai_rag_db_url="postgresql://local:super-secret@postgres:5432/rag",
    )
    pool = SimpleNamespace(backend="postgres")

    async def probe(_pool, _settings, _selection):
        raise RuntimeError(
            "could not connect to postgresql://local:super-secret@postgres:5432/rag"
        )

    monkeypatch.setattr(rag_readiness, "_probe_selected_backend", probe)

    report = await rag_readiness.build_rag_readiness_report(settings, pool)
    serialized = str(report)

    assert report["status"] == "NOT_READY"
    assert report["components"]["rag_storage"]["code"] == ("RAG_STORAGE_UNAVAILABLE")
    assert "super-secret" not in serialized
    assert "postgresql://" not in serialized


@pytest.mark.asyncio
async def test_oracle_missing_table_probe_reports_schema_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        app_env="local",
        rag_profile="local-oracle",
        rag_backend="oracle",
        ai_rag_db_url="oracle+oracledb://local:local@oracle-db/FREEPDB1",
    )
    pool = SimpleNamespace(backend="oracle")

    class _OracleError(Exception):
        pass

    async def probe(_pool, _settings, _selection):
        raise _OracleError(SimpleNamespace(code=942))

    monkeypatch.setattr(rag_readiness, "_probe_selected_backend", probe)

    report = await rag_readiness.build_rag_readiness_report(settings, pool)

    assert report["status"] == "NOT_READY"
    assert report["components"]["rag_storage"]["status"] == "UP"
    assert report["components"]["rag_schema"] == {
        "status": "DOWN",
        "code": "RAG_SCHEMA_NOT_READY",
    }
