import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.observability import tracing


def _app():
    app = FastAPI()
    app.add_middleware(tracing.RequestIdMiddleware)

    @app.get("/echo")
    async def echo():
        return {"request_id": tracing.current_request_id()}

    return app


def test_generates_request_id_and_exposes_it_to_handlers_and_header():
    res = TestClient(_app()).get("/echo")
    rid = res.headers["x-request-id"]
    assert len(rid) == 32
    assert res.json()["request_id"] == rid


def test_accepts_safe_caller_request_id():
    res = TestClient(_app()).get("/echo", headers={"X-Request-ID": "bff-abc12345"})
    assert res.headers["x-request-id"] == "bff-abc12345"
    assert res.json()["request_id"] == "bff-abc12345"


@pytest.mark.parametrize(
    "bad", ["short", "has space in it!!", "x" * 200, "a/b\r\nInjected: 1"]
)
def test_replaces_unsafe_request_id(bad):
    assert tracing.sanitize_request_id(bad) != bad
    assert tracing.sanitize_request_id(bad)


def test_request_id_does_not_leak_between_requests():
    client = TestClient(_app())
    a = client.get("/echo", headers={"X-Request-ID": "request-aaaa1"}).json()
    b = client.get("/echo").json()
    assert a["request_id"] == "request-aaaa1"
    assert b["request_id"] != "request-aaaa1"
    assert tracing.current_request_id() is None


def test_span_is_noop_without_sentry_and_propagates_errors():
    with tracing.span("test", "ok", key="v"):
        pass
    with pytest.raises(ValueError):
        with tracing.span("test", "boom"):
            raise ValueError("x")


@pytest.mark.asyncio
async def test_traced_decorator_preserves_result():
    @tracing.traced("test")
    async def fn(x):
        return x * 2

    assert await fn(3) == 6
    assert fn.__name__ == "fn"


def test_sentry_sample_rates_are_configurable_and_validated(monkeypatch):
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")
    monkeypatch.setenv("SENTRY_PROFILES_SAMPLE_RATE", "0")
    s = Settings()
    assert s.sentry_traces_sample_rate == 0.1
    assert s.sentry_profiles_sample_rate == 0.0
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "2")
    with pytest.raises(Exception):
        Settings()


def test_sentry_init_uses_configured_rates(monkeypatch):
    import sentry_sdk

    captured = {}
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: captured.update(kw))
    monkeypatch.setenv("SENTRY_DSN", "https://key@example.invalid/1")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.25")
    monkeypatch.setenv("SENTRY_PROFILES_SAMPLE_RATE", "0.05")
    from app import config, main

    config.get_settings.cache_clear()
    try:
        main.create_app()
    finally:
        config.get_settings.cache_clear()
    assert captured["traces_sample_rate"] == 0.25
    assert captured["profiles_sample_rate"] == 0.05
