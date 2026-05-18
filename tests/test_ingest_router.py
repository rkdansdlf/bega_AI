import asyncio
from types import ModuleType
from types import SimpleNamespace
import sys

from fastapi import BackgroundTasks

from app.routers import ingest


class _FakeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...] | None]] = []

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self.calls.append((query, params))

    def fetchall(self) -> list[tuple[object, ...]]:
        return []

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
    monkeypatch.setattr(
        ingest,
        "get_settings",
        lambda: SimpleNamespace(rag_quality_min_chars=1),
    )

    conn = _FakeConnection()
    payload = ingest.IngestPayload(
        title="Single Doc",
        content="body",
        source_table="kbo_regulations",
        source_row_id="rule-1",
    )

    result = asyncio.run(ingest.ingest_document(payload, conn, None, None))

    assert (
        conn.cursor_obj.calls[0][0]
        == "SET search_path TO public, extensions, security;"
    )
    params = conn.cursor_obj.calls[3][1]
    assert params[21] == "rule-1"
    assert params[22] == "Single Doc"
    assert result == {"status": "ok", "chunks": 1, "skipped": 0}


def test_ingest_document_suffixes_multi_chunk_source_rows(monkeypatch) -> None:
    async def fake_embed_texts(chunks, settings):
        return [[0.1, 0.2] for _ in chunks]

    monkeypatch.setattr(
        ingest,
        "smart_chunks",
        lambda text, settings=None: ["part one", "part two"],
    )
    monkeypatch.setattr(ingest, "async_embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        ingest,
        "get_settings",
        lambda: SimpleNamespace(rag_quality_min_chars=1),
    )

    conn = _FakeConnection()
    payload = ingest.IngestPayload(
        title="Multipart Doc",
        content="body",
        source_table="kbo_regulations",
        source_row_id="rule-2",
    )

    result = asyncio.run(ingest.ingest_document(payload, conn, None, None))

    assert (
        conn.cursor_obj.calls[0][0]
        == "SET search_path TO public, extensions, security;"
    )
    params = [call[1] for call in conn.cursor_obj.calls if call[1] and len(call[1]) == 25]
    assert [item[21] for item in params] == ["rule-2#part1", "rule-2#part2"]
    assert [item[22] for item in params] == [
        "Multipart Doc (분할 1)",
        "Multipart Doc (분할 2)",
    ]
    assert result == {"status": "ok", "chunks": 2, "skipped": 0}


def test_ingest_document_embeds_duplicate_batch_content_once(monkeypatch) -> None:
    captured_embed_inputs: list[list[str]] = []

    async def fake_embed_texts(chunks, settings):
        captured_embed_inputs.append(list(chunks))
        return [[0.1, 0.2] for _ in chunks]

    duplicate_chunk = "KBO 규정 설명과 경기 운영 기준을 충분히 담은 검색 가능한 문장입니다."
    monkeypatch.setattr(
        ingest,
        "smart_chunks",
        lambda text, settings=None: [duplicate_chunk, duplicate_chunk],
    )
    monkeypatch.setattr(ingest, "async_embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        ingest,
        "get_settings",
        lambda: SimpleNamespace(rag_quality_min_chars=1),
    )

    conn = _FakeConnection()
    payload = ingest.IngestPayload(
        title="Dedup Doc",
        content="body",
        source_table="kbo_regulations",
        source_row_id="rule-3",
    )

    result = asyncio.run(ingest.ingest_document(payload, conn, None, None))

    assert captured_embed_inputs == [[duplicate_chunk]]
    assert result == {"status": "ok", "chunks": 2, "skipped": 0}


def test_ingest_document_skips_sensitive_chunk_and_stores_valid_chunk(monkeypatch) -> None:
    captured_embed_inputs: list[list[str]] = []

    async def fake_embed_texts(chunks, settings):
        captured_embed_inputs.append(list(chunks))
        return [[0.1, 0.2] for _ in chunks]

    valid_chunk = "KBO 규정 설명과 경기 운영 기준을 충분히 담은 검색 가능한 문장입니다."
    monkeypatch.setattr(
        ingest,
        "smart_chunks",
        lambda text, settings=None: ["api_key=sk-live-secret-value", valid_chunk],
    )
    monkeypatch.setattr(ingest, "async_embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        ingest,
        "get_settings",
        lambda: SimpleNamespace(rag_quality_min_chars=1),
    )

    conn = _FakeConnection()
    payload = ingest.IngestPayload(
        title="Mixed Doc",
        content="body",
        source_table="kbo_regulations",
        source_row_id="rule-4",
    )

    result = asyncio.run(ingest.ingest_document(payload, conn, None, None))

    assert captured_embed_inputs == [[valid_chunk]]
    assert result == {"status": "ok", "chunks": 1, "skipped": 1}


def test_ingest_document_returns_zero_when_all_chunks_sensitive(monkeypatch) -> None:
    captured_embed_inputs: list[list[str]] = []

    async def fake_embed_texts(chunks, settings):
        captured_embed_inputs.append(list(chunks))
        return [[0.1, 0.2] for _ in chunks]

    monkeypatch.setattr(
        ingest,
        "smart_chunks",
        lambda text, settings=None: [
            "password=super-secret-value",
            "Authorization: Bearer abc.def.ghi",
        ],
    )
    monkeypatch.setattr(ingest, "async_embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        ingest,
        "get_settings",
        lambda: SimpleNamespace(rag_quality_min_chars=1),
    )

    conn = _FakeConnection()
    payload = ingest.IngestPayload(
        title="Secrets Doc",
        content="body",
        source_table="kbo_regulations",
        source_row_id="rule-5",
    )

    result = asyncio.run(ingest.ingest_document(payload, conn, None, None))

    assert captured_embed_inputs == []
    assert conn.cursor_obj.calls == []
    assert result == {"status": "ok", "chunks": 0, "skipped": 2}


def test_run_ingestion_job_forwards_incremental_parallel_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_ingest(**kwargs) -> None:
        captured.update(kwargs)

    fake_module = ModuleType("scripts.ingest_from_kbo")
    fake_module.ingest = fake_ingest
    fake_module.DEFAULT_TABLES = ["teams", "rag_chunks", "game"]
    monkeypatch.setitem(sys.modules, "scripts.ingest_from_kbo", fake_module)

    payload = ingest.RunIngestPayload(
        tables=None,
        season_year=2025,
        since="2025-03-22T09:00:00Z",
        read_batch_size=250,
        embed_batch_size=16,
        max_concurrency=3,
        commit_interval=200,
        parallel_engine="subinterp",
        workers=6,
        no_embed=True,
    )
    settings = SimpleNamespace(source_db_url="postgresql://source-db")
    background_tasks = BackgroundTasks()

    result = asyncio.run(
        ingest.run_ingestion_job(payload, background_tasks, settings, None, None)
    )

    assert result == {
        "status": "accepted",
        "message": "Ingestion job started in background.",
        "tables": ["teams", "game"],
    }
    assert len(background_tasks.tasks) == 1

    task = background_tasks.tasks[0]
    task.func(*task.args, **task.kwargs)

    assert captured["source_db_url"] == "postgresql://source-db"
    assert captured["tables"] == ["teams", "game"]
    assert captured["season_year"] == 2025
    assert captured["read_batch_size"] == 250
    assert captured["embed_batch_size"] == 16
    assert captured["max_concurrency"] == 3
    assert captured["commit_interval"] == 200
    assert captured["skip_embedding"] is True
    assert captured["parallel_engine"] == "subinterp"
    assert captured["workers"] == 6
    assert str(captured["since"]) == "2025-03-22 09:00:00+00:00"
