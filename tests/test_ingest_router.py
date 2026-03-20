import asyncio
from types import SimpleNamespace

from app.routers import ingest


class _FakeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, query: str, params: tuple[object, ...]) -> None:
        self.calls.append((query, params))

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj


def test_ingest_document_keeps_single_chunk_identity(monkeypatch) -> None:
    async def fake_embed_texts(chunks, settings):
        return [[0.1, 0.2] for _ in chunks]

    monkeypatch.setattr(
        ingest,
        "smart_chunks",
        lambda text, settings=None: ["single chunk"],
    )
    monkeypatch.setattr(ingest, "async_embed_texts", fake_embed_texts)
    monkeypatch.setattr(ingest, "get_settings", lambda: SimpleNamespace())

    conn = _FakeConnection()
    payload = ingest.IngestPayload(
        title="Single Doc",
        content="body",
        source_table="kbo_regulations",
        source_row_id="rule-1",
    )

    result = asyncio.run(ingest.ingest_document(payload, conn, None, None))

    params = conn.cursor_obj.calls[0][1]
    assert params[4] == "rule-1"
    assert params[5] == "Single Doc"
    assert result == {"status": "ok", "chunks": 1}


def test_ingest_document_suffixes_multi_chunk_source_rows(monkeypatch) -> None:
    async def fake_embed_texts(chunks, settings):
        return [[0.1, 0.2] for _ in chunks]

    monkeypatch.setattr(
        ingest,
        "smart_chunks",
        lambda text, settings=None: ["part one", "part two"],
    )
    monkeypatch.setattr(ingest, "async_embed_texts", fake_embed_texts)
    monkeypatch.setattr(ingest, "get_settings", lambda: SimpleNamespace())

    conn = _FakeConnection()
    payload = ingest.IngestPayload(
        title="Multipart Doc",
        content="body",
        source_table="kbo_regulations",
        source_row_id="rule-2",
    )

    result = asyncio.run(ingest.ingest_document(payload, conn, None, None))

    params = [call[1] for call in conn.cursor_obj.calls]
    assert [item[4] for item in params] == ["rule-2#part1", "rule-2#part2"]
    assert [item[5] for item in params] == [
        "Multipart Doc (분할 1)",
        "Multipart Doc (분할 2)",
    ]
    assert result == {"status": "ok", "chunks": 2}
