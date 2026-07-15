import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from app.core.ingest_runs import (
    IngestRunRecord,
    IngestRunRequest,
    IngestRunStatus,
    build_request_key,
)
from app.routers import ingest


class _FakeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...] | None]] = []

    async def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self.calls.append((query, params))

    async def fetchall(self) -> list[tuple[object, ...]]:
        return []

    async def __aenter__(self) -> "_FakeCursor":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
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


RUN_ID = UUID("44444444-4444-4444-8444-444444444444")
REQUESTED_AT = datetime(2026, 7, 15, 4, 30, tzinfo=UTC)


def _run_record(status=IngestRunStatus.QUEUED, *, table_summary=None):
    request = IngestRunRequest(
        tables=("game", "game_metadata"),
        season_year=2026,
        trigger_source="BACKEND_SCHEDULED",
    )
    return IngestRunRecord(
        run_id=RUN_ID,
        request_key=build_request_key(request),
        request=request,
        status=status,
        requested_at=REQUESTED_AT,
        finished_at=REQUESTED_AT if status not in {IngestRunStatus.QUEUED, IngestRunStatus.RUNNING} else None,
        error_code=(
            "MANUAL_BASEBALL_DATA_REQUIRED"
            if status is IngestRunStatus.MANUAL_BASEBALL_DATA_REQUIRED
            else None
        ),
        error_message=None,
        table_summary=table_summary or {},
    )


class _RunStore:
    def __init__(self, record=None, *, deduplicated=False):
        self.record = record
        self.deduplicated = deduplicated
        self.request = None

    async def create_or_get_active(self, request):
        self.request = request
        return self.record, self.deduplicated

    async def get(self, run_id):
        return self.record if self.record and self.record.run_id == run_id else None


def test_run_ingestion_job_persists_queue_request_without_background_task() -> None:
    store = _RunStore(_run_record())
    payload = ingest.RunIngestPayload(
        tables=["game_metadata", "game"],
        season_year=2026,
        mode="incremental",
        trigger_source="backend_scheduled",
    )

    response = asyncio.run(
        ingest.run_ingestion_job(payload, store, None, None)
    )

    assert response.status_code == 202
    assert json.loads(response.body) == {
        "run_id": str(RUN_ID),
        "status": "QUEUED",
        "deduplicated": False,
    }
    assert store.request.tables == ("game", "game_metadata")
    assert store.request.mode.value == "INCREMENTAL"
    assert store.request.trigger_source == "BACKEND_SCHEDULED"


def test_get_ingestion_run_returns_sanitized_manual_contract() -> None:
    contract = {
        "code": "MANUAL_BASEBALL_DATA_REQUIRED",
        "entity": "game",
        "missing_fields": ["game_date"],
        "import_source": "operator_manual_data",
    }
    store = _RunStore(
        _run_record(
            IngestRunStatus.MANUAL_BASEBALL_DATA_REQUIRED,
            table_summary={"error_contract": contract},
        )
    )

    result = asyncio.run(ingest.get_ingestion_run(RUN_ID, store, None, None))

    assert result["status"] == "MANUAL_BASEBALL_DATA_REQUIRED"
    assert result["error"] == contract
    assert "request_key" not in result
    assert "request_payload" not in result


def test_get_ingestion_run_returns_404_for_unknown_run() -> None:
    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            ingest.get_ingestion_run(uuid4(), _RunStore(record=None), None, None)
        )

    assert raised.value.status_code == 404
