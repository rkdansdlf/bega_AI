from __future__ import annotations

from types import SimpleNamespace

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core.embeddings import EmbeddingError
from app.core.rag_runtime import RagDependencyUnavailable


@pytest.fixture
def app_and_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("RAG_PROFILE", "test")
    monkeypatch.setenv("RAG_BACKEND", "fake")
    monkeypatch.setenv("AI_INGEST_WORKER_ENABLED", "false")
    monkeypatch.setenv("EMBED_PROVIDER", "local")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "local-rag-test-token")
    monkeypatch.setenv("POSTGRES_DB_URL", "postgresql://local:local@postgres/test")
    get_settings.cache_clear()

    from app import main as main_module

    app = main_module.create_app()
    client = TestClient(app, raise_server_exceptions=False)
    return app, client, main_module


def _payload() -> dict[str, object]:
    return {
        "title": "Local contract fixture",
        "content": "로컬 계약 검증을 위한 충분히 긴 내부 테스트 문서입니다.",
        "source_table": "contract_fixture",
        "source_row_id": "fixture-1",
    }


def test_health_is_process_liveness_only(app_and_client) -> None:
    _app, client, _main = app_and_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_ready_returns_component_level_not_ready_contract(
    app_and_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app, client, main_module = app_and_client

    async def not_ready():
        return {
            "status": "NOT_READY",
            "components": {
                "rag_backend": {
                    "status": "UP",
                    "backend": "postgres",
                    "profile": "local-postgres",
                },
                "rag_schema": {
                    "status": "DOWN",
                    "code": "RAG_SCHEMA_NOT_READY",
                },
                "embedding": {
                    "status": "DOWN",
                    "code": "RAG_EMBEDDING_NOT_READY",
                },
            },
        }

    monkeypatch.setattr(main_module, "get_readiness_report", not_ready)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "code": "AI_DEPENDENCY_UNAVAILABLE",
        "message": "mandatory AI dependencies are unavailable",
        "retryable": True,
        "status": "NOT_READY",
        "components": {
            "rag_backend": {
                "status": "UP",
                "backend": "postgres",
                "profile": "local-postgres",
            },
            "rag_schema": {
                "status": "DOWN",
                "code": "RAG_SCHEMA_NOT_READY",
            },
            "embedding": {
                "status": "DOWN",
                "code": "RAG_EMBEDDING_NOT_READY",
            },
        },
    }














@pytest.mark.parametrize(
    ("state_key", "code"),
    [
        ("db_error", "RAG_STORAGE_UNAVAILABLE"),
        ("embedding_error", "RAG_EMBEDDING_NOT_READY"),
    ],
)
def test_search_dependency_failure_is_not_disguised_as_empty_results(
    app_and_client,
    state_key: str,
    code: str,
) -> None:
    app, client, _main = app_and_client
    from app.routers import search

    class _Pipeline:
        async def retrieve(self, *_args, retrieval_state=None, **_kwargs):
            retrieval_state[state_key] = "credential-bearing dependency detail"
            return []

    app.dependency_overrides[search.get_rag_pipeline] = lambda: _Pipeline()
    app.dependency_overrides[search.require_rag_read_ready] = lambda: None

    response = client.get(
        "/ai/search/",
        params={"q": "로컬 계약", "use_multi_query": "false"},
        headers={"X-Internal-Api-Key": "local-rag-test-token"},
    )

    assert response.status_code == 503
    assert response.json()["code"] == code
    assert "credential-bearing" not in response.text
