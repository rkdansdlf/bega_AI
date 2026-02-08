"""단일 문서를 벡터 DB에 수동 업서트하기 위한 API 라우터."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..config import get_settings
from ..core.chunking import smart_chunks
from ..core.embeddings import async_embed_texts
from ..db import SCHEMA_SQL
from ..deps import get_db_connection

router = APIRouter(prefix="/ai/ingest", tags=["ingest"])


class IngestPayload(BaseModel):
    title: str
    content: str
    season_year: Optional[int] = None
    team_id: Optional[str] = None
    player_id: Optional[str] = None
    source_table: str
    source_row_id: str


@router.post("/")
async def ingest_document(payload: IngestPayload, conn=Depends(get_db_connection)):
    settings = get_settings()
    chunks = smart_chunks(payload.content)
    embeddings = await async_embed_texts(chunks, settings)

    with conn.cursor() as cur:
        for chunk, embedding in zip(chunks, embeddings):
            vector_literal = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
            cur.execute(
                """
                INSERT INTO rag_chunks (season_year, team_id, player_id, source_table, source_row_id, title, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (source_table, source_row_id)
                DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, updated_at = now()
                """,
                (
                    payload.season_year,
                    payload.team_id,
                    payload.player_id,
                    payload.source_table,
                    payload.source_row_id,
                    payload.title,
                    chunk,
                    vector_literal,
                ),
            )
    return {"status": "ok", "chunks": len(chunks)}


class RunIngestPayload(BaseModel):
    tables: Optional[list[str]] = None
    season_year: Optional[int] = None
    limit: Optional[int] = None
    read_batch_size: int = 500
    embed_batch_size: int = 32
    max_concurrency: int = 2
    commit_interval: int = 500
    no_embed: bool = False
    use_legacy_renderer: bool = False


@router.post("/run")
async def run_ingestion_job(
    payload: RunIngestPayload,
    background_tasks: BackgroundTasks,
    settings=Depends(get_settings),
):
    """KBO 데이터 인덱싱(Ingestion) 작업을 백그라운드에서 실행합니다."""

    # Lazy import to avoid circular dependency issues at startup if scripts logic changes
    from scripts.ingest_from_kbo import ingest, DEFAULT_TABLES

    tables_to_run = payload.tables if payload.tables else DEFAULT_TABLES

    # Remove rag_chunks if present (safety check)
    tables_to_run = [t for t in tables_to_run if t != "rag_chunks"]

    def _run_ingest_wrapper():
        try:
            print(f"[IngestWorker] Starting ingestion for tables: {tables_to_run}")
            ingest(
                tables=tables_to_run,
                limit=payload.limit,
                embed_batch_size=payload.embed_batch_size,
                read_batch_size=payload.read_batch_size,
                season_year=payload.season_year,
                use_legacy_renderer=payload.use_legacy_renderer,
                since=None,  # Incremental update via 'since' not yet exposed in payload for simplicity
                skip_embedding=payload.no_embed,
                max_concurrency=payload.max_concurrency,
                commit_interval=payload.commit_interval,
            )
            print(f"[IngestWorker] Ingestion completed successfully.")
        except Exception as e:
            print(f"[IngestWorker] Ingestion failed: {e}")

    background_tasks.add_task(_run_ingest_wrapper)

    return {
        "status": "accepted",
        "message": "Ingestion job started in background.",
        "tables": tables_to_run,
    }
