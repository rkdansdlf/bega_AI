"""야구 DB 분리 배선 검증.

분리 전(AI_BASEBALL_DB_URL 미설정)에는 두 풀이 같은 DB 를 가리켜야 하고,
설정한 순간에만 갈라져야 한다. 그 경계를 고정한다.
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
    [("rag", "general"), ("baseball", "baseball")],
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

    async with connection_scope(None, domain=domain) as conn:
        assert conn.label == expected_pool


@pytest.mark.asyncio
async def test_connection_scope_defaults_to_the_rag_pool(monkeypatch) -> None:
    """태깅을 빠뜨린 호출부가 야구 풀로 새지 않아야 한다."""

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

    async with connection_scope(None):
        pass

    assert calls == ["general"]
