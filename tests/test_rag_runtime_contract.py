from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.core.rag_runtime import (
    FakeRagPool,
    RagBackend,
    RagConfigurationError,
    RagDependencyUnavailable,
    RagProfile,
    resolve_rag_selection,
)


@pytest.fixture(autouse=True)
def _isolate_rag_contract_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "RAG_BACKEND",
        "RAG_PROFILE",
        "OCI_DB_URL",
        "POSTGRES_DB_URL",
        "SUPABASE_DB_URL",
        "AI_RAG_DB_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def _settings(**updates: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "app_env": "test",
        "rag_profile": "test",
        "rag_backend": "fake",
        "embed_provider": "local",
        "ingest_worker_enabled": False,
        "postgres_db_url": "postgresql://local:local@postgres:5432/test",
    }
    values.update(updates)
    return Settings(**values)


def test_explicit_env_only_mode_does_not_read_cwd_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / ".env").write_text(
        "AI_INTERNAL_TOKEN=dotenv-secret-that-must-not-load\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BEGA_EXPLICIT_ENV_ONLY", "true")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.delenv("AI_INTERNAL_TOKEN", raising=False)
    get_settings.cache_clear()

    try:
        settings = get_settings()
        assert settings.app_env == "test"
        assert settings.ai_internal_token is None
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("backend", ["oracle", "postgres", "fake"])
def test_settings_accepts_only_declared_rag_backends(backend: str) -> None:
    settings = _settings(rag_backend=backend)

    assert settings.rag_backend == backend


def test_settings_rejects_unknown_rag_backend_before_pool_creation() -> None:
    with pytest.raises(ValidationError, match="RAG_BACKEND"):
        _settings(rag_backend="auto")


@pytest.mark.parametrize(
    ("profile", "backend", "url", "expected_backend"),
    [
        ("test", "fake", None, RagBackend.FAKE),
        (
            "local-postgres",
            "postgres",
            "postgresql://local:local@postgres:5432/rag",
            RagBackend.POSTGRES,
        ),
        (
            "local-oracle",
            "oracle",
            "oracle+oracledb://local:local@oracle-db:1521/FREEPDB1",
            RagBackend.ORACLE,
        ),
    ],
)
def test_selection_uses_the_explicit_profile_and_backend_only(
    profile: str,
    backend: str,
    url: str | None,
    expected_backend: RagBackend,
) -> None:
    updates: dict[str, object] = {
        "app_env": "test" if profile == "test" else "local",
        "rag_profile": profile,
        "rag_backend": backend,
    }
    if url:
        updates["ai_rag_db_url"] = url

    selection = resolve_rag_selection(_settings(**updates))

    assert selection.profile is RagProfile(profile)
    assert selection.backend is expected_backend
    assert selection.url == url


def test_missing_backend_does_not_fall_back_to_a_database_url() -> None:
    settings = _settings(rag_backend=None, rag_profile="local-postgres")

    with pytest.raises(RagConfigurationError) as raised:
        resolve_rag_selection(settings)

    assert raised.value.code == "RAG_BACKEND_NOT_SELECTED"


def test_postgres_selection_does_not_fall_back_to_oci_or_legacy_urls() -> None:
    settings = _settings(
        rag_profile="local-postgres",
        rag_backend="postgres",
        postgres_db_url=None,
        ai_rag_db_url=None,
        oci_db_url="postgresql://remote:secret@prod-db.example:5432/prod",
        legacy_source_db_url="postgresql://legacy:secret@legacy.example:5432/prod",
    )

    with pytest.raises(RagConfigurationError) as raised:
        resolve_rag_selection(settings)

    assert raised.value.code == "RAG_STORAGE_URL_MISSING"
    assert "secret" not in str(raised.value)


def test_local_profile_rejects_remote_endpoint_without_leaking_credentials() -> None:
    settings = _settings(
        app_env="local",
        rag_profile="local-postgres",
        rag_backend="postgres",
        ai_rag_db_url=(
            "postgresql://production_user:super-secret@prod-db.example:5432/prod"
        ),
    )

    with pytest.raises(RagConfigurationError) as raised:
        resolve_rag_selection(settings)

    assert raised.value.code == "RAG_REMOTE_ENDPOINT_FORBIDDEN"
    assert "production_user" not in str(raised.value)
    assert "super-secret" not in str(raised.value)


def test_production_profile_requires_oracle_without_opening_a_connection() -> None:
    settings = _settings(
        app_env="production",
        rag_profile="production",
        rag_backend="postgres",
        ai_rag_db_url="postgresql://user:secret@prod-db.example:5432/rag",
    )

    with pytest.raises(RagConfigurationError) as raised:
        resolve_rag_selection(settings)

    assert raised.value.code == "RAG_PROFILE_BACKEND_MISMATCH"


def test_postgres_batch_worker_rejects_a_split_rag_destination() -> None:
    with pytest.raises(RuntimeError, match="same selected PostgreSQL RAG URL"):
        _settings(
            app_env="local",
            rag_profile="local-postgres",
            rag_backend="postgres",
            postgres_db_url="postgresql://local:local@postgres:5432/general",
            ai_rag_db_url="postgresql://local:local@postgres:5432/rag",
            ingest_worker_enabled=True,
        )


def test_postgres_batch_worker_is_allowed_only_on_the_unsplit_selected_url() -> None:
    settings = _settings(
        app_env="local",
        rag_profile="local-postgres",
        rag_backend="postgres",
        postgres_db_url="postgresql://local:local@postgres:5432/rag",
        ai_rag_db_url=None,
        ingest_worker_enabled=True,
    )

    selection = resolve_rag_selection(settings)

    assert selection.url == settings.database_url


def test_fake_selection_ignores_configured_urls_and_opens_no_network_pool() -> None:
    settings = _settings(
        rag_profile="test",
        rag_backend="fake",
        ai_rag_db_url="oracle+oracledb://user:secret@prod-db.example/prod",
    )

    selection = resolve_rag_selection(settings)
    pool = FakeRagPool(selection)

    assert selection.url is None
    assert pool.backend == "fake"
    assert pool.network_connection_attempts == 0


@pytest.mark.parametrize(
    ("backend", "profile", "url", "expected_factory"),
    [
        (
            "oracle",
            "local-oracle",
            "oracle+oracledb://u:p@oracle-db/FREEPDB1",
            "oracle",
        ),
        ("postgres", "local-postgres", "postgresql://u:p@postgres/rag", "postgres"),
        ("fake", "test", None, "fake"),
    ],
)
def test_rag_pool_factory_never_constructs_an_unselected_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    profile: str,
    url: str | None,
    expected_factory: str,
) -> None:
    from app import deps

    calls: list[str] = []

    class _Pool:
        def __init__(self, label: str) -> None:
            self.backend = label

        def get_stats(self) -> dict[str, object]:
            return {}

    settings = _settings(
        app_env="test" if profile == "test" else "local",
        rag_backend=backend,
        rag_profile=profile,
        ai_rag_db_url=url,
    )
    monkeypatch.setattr(deps, "get_settings", lambda: settings)
    monkeypatch.setattr(deps, "_rag_connection_pool", None)
    monkeypatch.setattr(
        deps,
        "OracleRagPool",
        lambda *_args, **_kwargs: calls.append("oracle") or _Pool("oracle"),
    )
    monkeypatch.setattr(
        deps,
        "_create_async_connection_pool",
        lambda **_kwargs: calls.append("postgres") or _Pool("postgres"),
    )

    pool = deps.get_rag_connection_pool()

    assert pool.backend == backend
    assert calls == ([] if expected_factory == "fake" else [expected_factory])


def test_writer_reader_and_readiness_resolve_the_same_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import deps

    selected_pool = SimpleNamespace(backend="oracle")
    general_pool = SimpleNamespace(backend="general")
    baseball_pool = SimpleNamespace(backend="baseball")
    captured: dict[str, object] = {}

    class _Pipeline:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(deps, "get_settings", lambda: _settings())
    monkeypatch.setattr(deps, "get_rag_connection_pool", lambda: selected_pool)
    monkeypatch.setattr(deps, "get_connection_pool", lambda: general_pool)
    monkeypatch.setattr(deps, "get_baseball_connection_pool", lambda: baseball_pool)
    monkeypatch.setattr(deps, "get_shared_baseball_agent_runtime", lambda: object())
    monkeypatch.setattr(deps, "_get_shared_context_formatter", lambda: object())
    monkeypatch.setattr(deps, "_get_shared_wpa_calculator", lambda: object())
    monkeypatch.setattr(deps, "RAGPipeline", _Pipeline)

    deps.get_rag_pipeline()

    assert captured["rag_pool"] is selected_pool
    assert captured["pool"] is general_pool


def test_search_pipeline_maps_selection_failure_to_dependency_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import deps

    def missing_selection():
        raise RagConfigurationError(
            "RAG_BACKEND_NOT_SELECTED",
            "RAG_BACKEND must be configured explicitly",
        )

    monkeypatch.setattr(deps, "get_rag_connection_pool", missing_selection)

    with pytest.raises(RagDependencyUnavailable) as raised:
        deps.get_rag_pipeline()

    assert raised.value.code == "RAG_BACKEND_NOT_SELECTED"
