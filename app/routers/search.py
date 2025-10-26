"""검색 디버깅용 엔드포인트를 제공하는 라우터."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from psycopg2.extensions import connection as PgConnection

from ..deps import get_db_connection, get_rag_pipeline

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/")
async def search_chunks(
    q: str = Query(..., min_length=2, description="질문 또는 키워드"),
    limit: int = Query(5, ge=1, le=20),
    pipeline=Depends(get_rag_pipeline),
):
    docs = await pipeline.retrieve(q, limit=limit)
    return {"query": q, "results": docs}
