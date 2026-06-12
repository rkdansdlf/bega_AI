"""단일 문서를 벡터 DB에 수동 업서트하기 위한 API 라우터."""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel

from ..config import get_settings
from ..core.chunking import smart_chunks
from ..core.embeddings import async_embed_texts
from ..core.rag_storage import (
    RAG_CHUNKS_UPSERT_SQL,
    base_source_row_id,
    build_chunk_storage_fields,
    build_upsert_tuple,
    fetch_existing_embedding_texts,
    is_search_worthy_content,
    resolve_embedding_model,
    resolve_embedding_version,
    scan_sensitive_content,
    soft_deactivate_missing_parts,
    vector_literal,
)
from ..deps import get_db_connection, require_ai_internal_token
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

    with conn.cursor() as cur:
        cur.execute(f"SET search_path TO {PGVECTOR_SEARCH_PATH};")
        existing_embeddings = (
            fetch_existing_embedding_texts(
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

    with conn.cursor() as cur:
        cur.execute(f"SET search_path TO {PGVECTOR_SEARCH_PATH};")
        active_source_row_ids: list[str] = []
        for record, embedding_text in zip(records, vector_literals):
            _idx, source_row_id, title, chunk, storage_fields = record
            active_source_row_ids.append(source_row_id)
            cur.execute(
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
        soft_deactivate_missing_parts(
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
    limit: Optional[int] = None
    since: Optional[str] = None
    read_batch_size: int = 500
    embed_batch_size: int = 32
    max_concurrency: int = 2
    commit_interval: int = 500
    parallel_engine: str = "thread"
    workers: int = 4
    no_embed: bool = False
    use_legacy_renderer: bool = False


@router.post("/run")
async def run_ingestion_job(
    payload: RunIngestPayload,
    background_tasks: BackgroundTasks,
    settings=Depends(get_settings),
    __: None = Depends(require_ai_internal_token),
    _: None = Depends(rate_limit_debug_dependency),
):
    """KBO 데이터 인덱싱(Ingestion) 작업을 백그라운드에서 실행합니다."""

    # Lazy import to avoid circular dependency issues at startup if scripts logic changes
    from scripts.ingest_from_kbo import ingest, DEFAULT_TABLES

    tables_to_run = payload.tables if payload.tables else DEFAULT_TABLES

    # Remove rag_chunks if present (safety check)
    tables_to_run = [t for t in tables_to_run if t != "rag_chunks"]

    def _parse_since_value(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))

    def _run_ingest_wrapper():
        try:
            logger.info("[IngestWorker] Starting ingestion for tables: %s", tables_to_run)
            ingest(
                source_db_url=settings.source_db_url,
                tables=tables_to_run,
                limit=payload.limit,
                embed_batch_size=payload.embed_batch_size,
                read_batch_size=payload.read_batch_size,
                season_year=payload.season_year,
                use_legacy_renderer=payload.use_legacy_renderer,
                since=_parse_since_value(payload.since),
                skip_embedding=payload.no_embed,
                max_concurrency=max(1, payload.max_concurrency),
                commit_interval=max(1, payload.commit_interval),
                parallel_engine=payload.parallel_engine,
                workers=max(1, payload.workers),
            )
            logger.info("[IngestWorker] Ingestion completed successfully.")
        except Exception as e:
            logger.exception("[IngestWorker] Ingestion failed: %s", e)

    background_tasks.add_task(_run_ingest_wrapper)

    return {
        "status": "accepted",
        "message": "Ingestion job started in background.",
        "tables": tables_to_run,
    }
