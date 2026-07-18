"""단일 문서를 벡터 DB에 수동 업서트하기 위한 API 라우터."""

import logging
from collections.abc import Mapping
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from ..config import get_settings
from ..core.chunking import smart_chunks
from ..core.embeddings import async_embed_texts
from ..core.ingest_runs import IngestRunMode, IngestRunRequest, IngestRunStatus
from ..observability.metrics import (
    AI_INGEST_SUBMISSIONS_TOTAL,
    normalize_ingest_trigger_source,
)
from ..core.rag_storage import (
    RAG_CHUNKS_UPSERT_SQL,
    base_source_row_id,
    build_chunk_storage_fields,
    build_upsert_tuple,
    fetch_existing_embedding_texts_async,
    is_search_worthy_content,
    resolve_embedding_model,
    resolve_embedding_version,
    scan_sensitive_content,
    soft_deactivate_missing_parts_async,
    vector_literal,
)
from ..deps import (
    get_db_connection,
    get_ingest_run_store,
    require_ai_internal_token,
)
from ..core.ratelimit import rate_limit_debug_dependency

router = APIRouter(prefix="/ai/ingest", tags=["ingest"])
PGVECTOR_SEARCH_PATH = "public, extensions, security"
logger = logging.getLogger(__name__)


class IngestPayload(BaseModel):
    title: str
    content: str
    season_year: Optional[int] = None
    team_id: Optional[str] = None
    player_id: Optional[str] = None
    source_table: str
    source_row_id: str
    source_type: Optional[str] = None
    source_uri: Optional[str] = None


@router.post("/")
async def ingest_document(
    payload: IngestPayload,
    conn=Depends(get_db_connection),
    __: None = Depends(require_ai_internal_token),
    _: None = Depends(rate_limit_debug_dependency),
):
    settings = get_settings()
    chunks = smart_chunks(payload.content, settings=settings)
    chunk_count = len(chunks)
    embedding_model = resolve_embedding_model(settings)
    embedding_version = resolve_embedding_version(settings)
    min_chars = int(getattr(settings, "rag_quality_min_chars", 50) or 50)

    records = []
    for idx, chunk in enumerate(chunks, start=1):
        source_row_id = payload.source_row_id
        title = payload.title
        if chunk_count > 1:
            source_row_id = f"{payload.source_row_id}#part{idx}"
            title = f"{payload.title} (분할 {idx})"
        sensitive_findings = scan_sensitive_content(chunk)
        if sensitive_findings:
            logger.warning(
                "Skipping sensitive RAG ingest chunk source_table=%s source_row_id=%s findings=%s",
                payload.source_table,
                source_row_id,
                ",".join(sorted(set(sensitive_findings))),
            )
            continue
        if not is_search_worthy_content(chunk, min_chars=min_chars):
            continue
        try:
            storage_fields = build_chunk_storage_fields(
                settings=settings,
                source_table=payload.source_table,
                source_row_id=source_row_id,
                content=chunk,
                meta=None,
                source_type=payload.source_type,
                source_uri=payload.source_uri,
                embedding_model=embedding_model,
                embedding_version=embedding_version,
            )
        except ValueError as exc:
            logger.warning(
                "Skipping blocked RAG ingest chunk source_table=%s source_row_id=%s reason=%s",
                payload.source_table,
                source_row_id,
                exc,
            )
            continue
        records.append((idx, source_row_id, title, chunk, storage_fields))

    if not records:
        return {"status": "ok", "chunks": 0, "skipped": chunk_count}

    async with conn.cursor() as cur:
        await cur.execute(f"SET search_path TO {PGVECTOR_SEARCH_PATH};")
        existing_embeddings = (
            await fetch_existing_embedding_texts_async(
                cur,
                content_hashes=(record[4]["content_hash"] for record in records),
                embedding_model=embedding_model,
                embedding_dim=records[0][4]["embedding_dim"],
                embedding_version=embedding_version,
                chunking_version=records[0][4]["chunking_version"],
            )
            if bool(getattr(settings, "rag_storage_dedup_enabled", True))
            else {}
        )

    vector_literals: list[Optional[str]] = [None] * len(records)
    embed_indices: list[int] = []
    embed_texts: list[str] = []
    pending_hashes: dict[str, int] = {}
    duplicate_links: list[tuple[int, int]] = []
    for idx, record in enumerate(records):
        storage_fields = record[4]
        existing = existing_embeddings.get(storage_fields["content_hash"])
        if existing:
            vector_literals[idx] = existing
            continue
        pending_idx = pending_hashes.get(storage_fields["content_hash"])
        if pending_idx is not None:
            duplicate_links.append((idx, pending_idx))
            continue
        pending_hashes[storage_fields["content_hash"]] = idx
        embed_indices.append(idx)
        embed_texts.append(record[3])

    if embed_texts:
        embeddings = await async_embed_texts(embed_texts, settings)
        for idx, embedding in zip(embed_indices, embeddings):
            vector_literals[idx] = vector_literal(embedding)
        for duplicate_idx, original_idx in duplicate_links:
            vector_literals[duplicate_idx] = vector_literals[original_idx]

    async with conn.cursor() as cur:
        await cur.execute(f"SET search_path TO {PGVECTOR_SEARCH_PATH};")
        active_source_row_ids: list[str] = []
        for record, embedding_text in zip(records, vector_literals):
            _idx, source_row_id, title, chunk, storage_fields = record
            active_source_row_ids.append(source_row_id)
            await cur.execute(
                RAG_CHUNKS_UPSERT_SQL,
                build_upsert_tuple(
                    meta=None,
                    storage_fields=storage_fields,
                    season_year=payload.season_year,
                    season_id=None,
                    league_type_code=None,
                    team_id=payload.team_id,
                    player_id=payload.player_id,
                    source_table=payload.source_table,
                    source_row_id=source_row_id,
                    title=title,
                    content=chunk,
                    embedding_text=embedding_text,
                ),
            )
        await soft_deactivate_missing_parts_async(
            cur,
            source_table=payload.source_table,
            source_prefix=base_source_row_id(payload.source_row_id),
            active_source_row_ids=active_source_row_ids,
        )
    return {
        "status": "ok",
        "chunks": len(records),
        "skipped": chunk_count - len(records),
    }


class RunIngestPayload(BaseModel):
    tables: Optional[list[str]] = None
    season_year: Optional[int] = None
    mode: str = "INCREMENTAL"
    trigger_source: str = "MANUAL_API"
    since: Optional[str] = None

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        return IngestRunMode(value.strip().upper()).value

    @field_validator("trigger_source")
    @classmethod
    def _validate_trigger_source(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"BACKEND_SCHEDULED", "MANUAL_API", "CLI_RECOVERY"}:
            raise ValueError("unsupported ingestion trigger_source")
        return normalized


@router.post("/run")
async def run_ingestion_job(
    payload: RunIngestPayload,
    store=Depends(get_ingest_run_store),
    _: None = Depends(rate_limit_debug_dependency),
    __: None = Depends(require_ai_internal_token),
):
    """Persist or deduplicate a durable internal-DB ingestion run."""

    from scripts.ingest_from_kbo import DEFAULT_TABLES

    tables = payload.tables
    if tables is None:
        tables = [table for table in DEFAULT_TABLES if table != "rag_chunks"]
    try:
        request = IngestRunRequest(
            tables=tuple(tables),
            season_year=payload.season_year,
            mode=payload.mode,
            trigger_source=payload.trigger_source,
            since=payload.since,
        ).normalized()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    record, deduplicated = await store.create_or_get_active(request)
    AI_INGEST_SUBMISSIONS_TOTAL.labels(
        trigger_source=normalize_ingest_trigger_source(request.trigger_source),
        result="deduplicated" if deduplicated else "created",
    ).inc()
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "run_id": str(record.run_id),
            "status": record.status.value,
            "deduplicated": deduplicated,
        },
    )


def _iso_or_none(value):
    return value.isoformat() if value is not None else None


def _sanitize_table_summary(summary: Mapping[str, object]) -> dict[str, object]:
    allowed_fields = {
        "source_rows",
        "written_chunks",
        "reused_embeddings",
        "embedded_chunks",
        "max_updated_at",
    }
    sanitized = {}
    for source_table, raw_value in summary.items():
        if source_table == "error_contract" or not isinstance(raw_value, Mapping):
            continue
        sanitized[source_table] = {
            key: raw_value[key]
            for key in allowed_fields
            if key in raw_value
            and isinstance(raw_value[key], (str, int, float, type(None)))
        }
    return sanitized


def _sanitize_error(record) -> dict[str, object] | None:
    if record.status is IngestRunStatus.MANUAL_BASEBALL_DATA_REQUIRED:
        contract = record.table_summary.get("error_contract")
        if isinstance(contract, Mapping):
            allowed = {
                "code",
                "scope",
                "entity",
                "range",
                "missing_fields",
                "import_source",
                "operator_message",
                "message",
                "blocking",
            }
            return {key: contract[key] for key in allowed if key in contract}
    if record.error_code:
        return {"code": record.error_code, "message": record.error_message}
    return None


@router.get("/runs/{run_id}")
async def get_ingestion_run(
    run_id: UUID,
    store=Depends(get_ingest_run_store),
    _: None = Depends(rate_limit_debug_dependency),
    __: None = Depends(require_ai_internal_token),
):
    """Return only sanitized durable run status fields."""

    record = await store.get(run_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return {
        "run_id": str(record.run_id),
        "status": record.status.value,
        "trigger_source": record.request.trigger_source,
        "requested_at": _iso_or_none(record.requested_at),
        "started_at": _iso_or_none(record.started_at),
        "heartbeat_at": _iso_or_none(record.heartbeat_at),
        "finished_at": _iso_or_none(record.finished_at),
        "recovery_attempts": record.recovery_attempts,
        "tables": _sanitize_table_summary(record.table_summary),
        "error": _sanitize_error(record),
    }
