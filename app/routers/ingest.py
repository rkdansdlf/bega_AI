"""단일 문서를 벡터 DB에 수동 업서트하기 위한 API 라우터."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
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
