from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from app.agents import shared_runtime
from app import deps


def test_create_baseball_agent_runtime_delegates_to_factory(monkeypatch):
    captured: dict[str, object] = {}
    expected_runtime = object()

    def _fake_factory(settings=None):
        captured["settings"] = settings
        return expected_runtime

    monkeypatch.setattr(deps, "create_baseball_agent_runtime", _fake_factory)

    custom_settings = object()

    assert deps._create_baseball_agent_runtime(custom_settings) is expected_runtime
    assert captured["settings"] is custom_settings


def test_initialize_shared_baseball_agent_runtime_is_idempotent(monkeypatch):
    created: list[object] = []

    def _fake_create(settings=None):
        runtime = SimpleNamespace(runtime_id=len(created) + 1)
        created.append(runtime)
        return runtime

    monkeypatch.setattr(shared_runtime, "_default_runtime", None)
    monkeypatch.setattr(shared_runtime, "_runtime_by_settings_id", {})
    monkeypatch.setattr(shared_runtime, "create_baseball_agent_runtime", _fake_create)

    runtime_one = shared_runtime.initialize_shared_baseball_agent_runtime()
    runtime_two = shared_runtime.initialize_shared_baseball_agent_runtime()

    assert runtime_one is runtime_two
    assert len(created) == 1


@pytest.mark.asyncio
async def test_get_agent_uses_request_context_and_shared_agent(monkeypatch):
    request_context_calls: list[tuple[str, object]] = []
    shared_agent = object()

    class _FakeRuntime:
        def __init__(self):
            self.shared_agent = shared_agent

        @contextmanager
        def request_context(self, connection):
            request_context_calls.append(("enter", connection))
            try:
                yield
            finally:
                request_context_calls.append(("exit", connection))

    monkeypatch.setattr(
        deps, "get_shared_baseball_agent_runtime", lambda: _FakeRuntime()
    )

    conn_one = object()
    agent_dependency = deps.get_agent(conn_one)

    assert await anext(agent_dependency) is shared_agent
    assert request_context_calls == [("enter", conn_one)]
    with pytest.raises(StopAsyncIteration):
        await anext(agent_dependency)
    assert request_context_calls == [("enter", conn_one), ("exit", conn_one)]


def test_get_rag_pipeline_uses_shared_runtime(monkeypatch):
    captured: dict[str, object] = {}
    fake_runtime = object()
    fake_pool = object()

    class _FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(deps, "get_shared_baseball_agent_runtime", lambda: fake_runtime)
    monkeypatch.setattr(deps, "RAGPipeline", _FakePipeline)
    monkeypatch.setattr(deps, "get_connection_pool", lambda: fake_pool)

    pipeline = deps.get_rag_pipeline()

    assert pipeline is not None
    assert captured["agent_runtime"] is fake_runtime
    assert captured["pool"] is fake_pool
