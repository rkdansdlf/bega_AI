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
    ("payload", "expected_status"),
    [
        ({"title": "missing fields"}, 422),
        ({**_payload(), "season_year": "not-an-integer"}, 422),
        ({**_payload(), "league_type_code": 123}, 422),
    ],
)
def test_ingest_validation_errors_remain_422(
    app_and_client,
    payload: dict[str, object],
    expected_status: int,
) -> None:
    app, client, _main = app_and_client
    from app.routers import ingest

    async def no_connection_needed():
        yield object()

    app.dependency_overrides[ingest.get_rag_connection] = no_connection_needed
    app.dependency_overrides[ingest.require_rag_write_ready] = lambda: None

    response = client.post(
        "/ai/ingest/",
        json=payload,
        headers={"X-Internal-Api-Key": "local-rag-test-token"},
    )

    assert response.status_code == expected_status


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Internal-Api-Key": "wrong-token"}],
)
def test_ingest_missing_or_wrong_token_is_401(app_and_client, headers) -> None:
    app, client, _main = app_and_client
    from app.routers import ingest

    async def no_connection_needed():
        yield object()

    app.dependency_overrides[ingest.get_rag_connection] = no_connection_needed
    app.dependency_overrides[ingest.require_rag_write_ready] = lambda: None

    response = client.post("/ai/ingest/", json=_payload(), headers=headers)

    assert response.status_code == 401


@pytest.mark.parametrize(
    "code",
    [
        "RAG_STORAGE_UNAVAILABLE",
        "RAG_SCHEMA_NOT_READY",
        "RAG_VECTOR_INDEX_NOT_READY",
    ],
)
def test_ingest_readiness_dependency_failures_are_structured_503(
    app_and_client,
    code: str,
) -> None:
    app, client, _main = app_and_client
    from app.routers import ingest

    async def no_connection_needed():
        yield object()

    async def dependency_down():
        raise RagDependencyUnavailable(code)

    app.dependency_overrides[ingest.get_rag_connection] = no_connection_needed
    app.dependency_overrides[ingest.require_rag_write_ready] = dependency_down

    response = client.post(
        "/ai/ingest/",
        json=_payload(),
        headers={"X-Internal-Api-Key": "local-rag-test-token"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "code": code,
        "message": "RAG dependency is unavailable",
        "retryable": True,
    }


def test_ingest_embedding_failure_is_structured_503(
    app_and_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client, _main = app_and_client
    from app.routers import ingest

    class _Cursor:
        async def execute(self, *_args, **_kwargs):
            return None

        async def fetchall(self):
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class _Connection:
        def cursor(self):
            return _Cursor()

    async def connection():
        yield _Connection()

    async def embedding_down(*_args, **_kwargs):
        raise EmbeddingError("provider response containing a-secret-value")

    app.dependency_overrides[ingest.get_rag_connection] = connection
    app.dependency_overrides[ingest.require_rag_write_ready] = lambda: None
    monkeypatch.setattr(
        ingest,
        "smart_chunks",
        lambda _text, settings=None: [
            "로컬 RAG 계약 검증을 위한 문서이며 외부 데이터 없이 임베딩 의존성 "
            "실패가 구조화된 응답으로 변환되는지 확인할 만큼 충분히 긴 내용입니다."
        ],
    )
    monkeypatch.setattr(ingest, "async_embed_texts", embedding_down)

    response = client.post(
        "/ai/ingest/",
        json=_payload(),
        headers={"X-Internal-Api-Key": "local-rag-test-token"},
    )

    assert response.status_code == 503
    assert response.json()["code"] == "RAG_EMBEDDING_NOT_READY"
    assert "a-secret-value" not in response.text


def test_ingest_unexpected_code_defect_remains_500(
    app_and_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client, _main = app_and_client
    from app.routers import ingest

    async def connection():
        yield object()

    app.dependency_overrides[ingest.get_rag_connection] = connection
    app.dependency_overrides[ingest.require_rag_write_ready] = lambda: None
    monkeypatch.setattr(
        ingest,
        "smart_chunks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("unexpected implementation defect")
        ),
    )

    response = client.post(
        "/ai/ingest/",
        json=_payload(),
        headers={"X-Internal-Api-Key": "local-rag-test-token"},
    )

    assert response.status_code == 500
    assert "unexpected implementation defect" not in response.text


def test_connection_acquisition_failure_is_structured_503(
    app_and_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app, client, _main = app_and_client
    from app import deps

    class _Context:
        async def __aenter__(self):
            raise psycopg.OperationalError("postgresql://user:secret@postgres/rag")

        async def __aexit__(self, *_args):
            return False

    pool = SimpleNamespace(backend="fake", connection=lambda: _Context())
    monkeypatch.setattr(deps, "get_rag_connection_pool", lambda: pool)

    response = client.post(
        "/ai/ingest/",
        json=_payload(),
        headers={"X-Internal-Api-Key": "local-rag-test-token"},
    )

    assert response.status_code == 503
    assert response.json()["code"] == "RAG_STORAGE_UNAVAILABLE"
    assert "secret" not in response.text


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
