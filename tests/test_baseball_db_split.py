"""DB 도메인 분리 배선 검증.

캐시(자체 테이블) · RAG(rag_chunks) · 야구 세 도메인이 각자 URL 을 갖되,
전용 환경변수를 설정하기 전에는 모두 같은 DB 로 폴백해야 한다. 그 경계를 고정한다.
"""

import pytest

from app.config import Settings


@pytest.fixture
def build_settings(monkeypatch):
    """실제 .env / 프로세스 환경을 격리한 Settings 팩토리.

    database_url 은 OCI_DB_URL 을 먼저 보므로, 개발자 환경에 그 값이 있으면
    테스트가 조용히 그것을 집는다. 관련 변수를 모두 지우고 시작한다.
    """

    def _factory(**env) -> Settings:
        for name in (
            "OCI_DB_URL",
            "POSTGRES_DB_URL",
            "SUPABASE_DB_URL",
            "AI_BASEBALL_DB_URL",
            "AI_RAG_DB_URL",
        ):
            monkeypatch.delenv(name, raising=False)
        env.setdefault("POSTGRES_DB_URL", "postgresql://user:pw@rag-host:5432/rag")
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return Settings(_env_file=None)

    return _factory


def test_baseball_url_defaults_to_the_rag_url(build_settings) -> None:
    settings = build_settings()

    # 미설정 상태에서 갈라지면 분리 전 배포가 조용히 다른 DB 를 보게 된다.
    assert settings.baseball_db_url == settings.database_url
    assert settings.source_db_url == settings.database_url


def test_baseball_url_splits_only_when_configured(build_settings) -> None:
    settings = build_settings(AI_BASEBALL_DB_URL="postgresql://user:pw@bb-host:5432/bb")

    assert settings.database_url.endswith("/rag")
    assert settings.baseball_db_url.endswith("/bb")
    # 인제스트는 source_db_url 로 읽고 database_url 로 쓴다. 이 둘이 갈려야
    # 경계를 넘는 적재가 성립한다.
    assert settings.source_db_url == settings.baseball_db_url


def test_oci_url_still_wins_for_the_rag_database(build_settings) -> None:
    settings = build_settings(OCI_DB_URL="postgresql://user:pw@oci-host:5432/oci")

    assert settings.database_url.endswith("/oci")
    assert settings.baseball_db_url.endswith("/oci")


@pytest.mark.parametrize(
    ("domain", "expected_pool"),
    [("cache", "general"), ("rag", "rag"), ("baseball", "baseball")],
)
@pytest.mark.asyncio
async def test_connection_scope_selects_the_pool_for_its_domain(
    monkeypatch, domain: str, expected_pool: str
) -> None:
    from app.tools.pooled_connection import connection_scope

    class _FakeConn:
        closed = False

        def __init__(self, label: str) -> None:
            self.label = label

    class _FakePool:
        def __init__(self, label: str) -> None:
            self.label = label

        def connection(self):
            conn = _FakeConn(self.label)

            class _Ctx:
                async def __aenter__(self):
                    return conn

                async def __aexit__(self, *exc):
                    return False

            return _Ctx()

    monkeypatch.setattr("app.deps.get_connection_pool", lambda: _FakePool("general"))
    monkeypatch.setattr(
        "app.deps.get_baseball_connection_pool", lambda: _FakePool("baseball")
    )
    monkeypatch.setattr("app.deps.get_rag_connection_pool", lambda: _FakePool("rag"))

    async with connection_scope(None, domain=domain) as conn:
        assert conn.label == expected_pool


@pytest.mark.asyncio
async def test_connection_scope_defaults_to_the_local_cache_pool(monkeypatch) -> None:
    """태깅을 빠뜨린 호출부가 원격 DB(RAG·야구)로 새지 않아야 한다."""

    from app.tools.pooled_connection import connection_scope

    calls: list[str] = []

    class _Ctx:
        async def __aenter__(self):
            class _Conn:
                closed = False

            return _Conn()

        async def __aexit__(self, *exc):
            return False

    class _FakePool:
        def __init__(self, label: str) -> None:
            self.label = label

        def connection(self):
            calls.append(self.label)
            return _Ctx()

    monkeypatch.setattr("app.deps.get_connection_pool", lambda: _FakePool("general"))
    monkeypatch.setattr(
        "app.deps.get_baseball_connection_pool", lambda: _FakePool("baseball")
    )
    monkeypatch.setattr("app.deps.get_rag_connection_pool", lambda: _FakePool("rag"))

    async with connection_scope(None):
        pass

    assert calls == ["general"]


def test_rag_url_defaults_to_the_cache_database(build_settings) -> None:
    settings = build_settings()

    assert settings.rag_db_url == settings.database_url


def test_rag_url_splits_without_dragging_the_caches_along(build_settings) -> None:
    """A-2 의 요점: rag_chunks 만 원격으로 가고 캐시는 남는다."""

    settings = build_settings(
        AI_RAG_DB_URL="postgresql://user:pw@rag-remote:5432/ragdb",
        AI_BASEBALL_DB_URL="postgresql://user:pw@bb-host:5432/bb",
    )

    assert settings.rag_db_url.endswith("/ragdb")
    # 캐시는 요청마다 쓰기가 발생하므로 원격으로 따라가면 안 된다.
    assert settings.database_url.endswith("/rag")
    assert settings.baseball_db_url.endswith("/bb")
