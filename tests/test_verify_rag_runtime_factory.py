from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

from scripts import verify_rag


def test_verify_rag_uses_runtime_factory(monkeypatch, capsys) -> None:
    fake_settings = SimpleNamespace(database_url="postgresql://example/test")
    fake_connection = object()
    captured: dict[str, object] = {}

    @contextmanager
    def _fake_connect(_db_url):
        yield fake_connection

    class _FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def retrieve(self, query: str, limit: int = 5):
            captured["query"] = query
            captured["limit"] = limit
            return []

    monkeypatch.setattr(verify_rag, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(verify_rag, "create_baseball_agent_runtime", lambda settings: "runtime")
    monkeypatch.setattr(verify_rag, "RAGPipeline", _FakePipeline)
    monkeypatch.setattr(verify_rag.psycopg, "connect", _fake_connect)

    asyncio.run(verify_rag.verify_retrieval("테스트 질의"))

    assert captured["settings"] is fake_settings
    assert captured["connection"] is fake_connection
    assert captured["agent_runtime"] == "runtime"
    assert captured["query"] == "테스트 질의"
    assert captured["limit"] == 5
    assert "Found 0 documents." in capsys.readouterr().out
