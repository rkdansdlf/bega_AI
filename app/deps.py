"""FastAPI 의존성 주입을 위한 공통 헬퍼를 정의하는 모듈."""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager

import psycopg2
from psycopg2.extensions import connection as PgConnection
from fastapi import Depends

from .config import get_settings
from .core.rag import RAGPipeline
from .ml.intent_router import predict_intent, load_clf


@asynccontextmanager
async def lifespan(app):
    load_clf()
    yield


def get_db_connection() -> Generator[PgConnection, None, None]:
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url)
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


def get_rag_pipeline(
    conn: PgConnection = Depends(get_db_connection),
) -> RAGPipeline:
    settings = get_settings()
    return RAGPipeline(settings=settings, connection=conn)


def get_intent_router():
    return predict_intent
