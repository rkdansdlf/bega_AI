"""FastAPI 의존성 주입을 위한 공통 헬퍼를 정의하는 모듈."""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from typing import Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extensions import connection as PgConnection
from fastapi import Depends

from .config import get_settings
from .core.rag import RAGPipeline
from .ml.intent_router import predict_intent, load_clf
from .agents.baseball_agent import BaseballStatisticsAgent
from .core.prompts import SYSTEM_PROMPT

# 전역 커넥션 풀 (앱 시작 시 한 번만 생성)
_connection_pool: Optional[pool.SimpleConnectionPool] = None


def get_connection_pool() -> pool.SimpleConnectionPool:
    """커넥션 풀을 가져오거나 생성합니다."""
    global _connection_pool
    
    if _connection_pool is None:
        settings = get_settings()
        _connection_pool = pool.SimpleConnectionPool(
            minconn=1,        # 최소 1개 커넥션 유지
            maxconn=5,        # 최대 5개 커넥션 (Supabase 무료 플랜 고려)
            dsn=settings.database_url
        )
    
    return _connection_pool


def close_connection_pool():
    """앱 종료 시 커넥션 풀을 닫습니다."""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None


@asynccontextmanager
async def lifespan(app):
    """앱 시작/종료 시 실행되는 lifespan 이벤트"""
    # 시작 시
    load_clf()
    get_connection_pool()  # 커넥션 풀 초기화
    
    yield
    
    # 종료 시
    close_connection_pool()  # 모든 커넥션 정리


def get_db_connection() -> Generator[PgConnection, None, None]:
    """커넥션 풀에서 커넥션을 가져와서 사용 후 반환합니다."""
    pool_instance = get_connection_pool()
    conn = pool_instance.getconn()  # 풀에서 재사용 가능한 커넥션 가져오기
    conn.autocommit = True
    
    try:
        yield conn
    finally:
        pool_instance.putconn(conn)  # 사용 후 풀에 반환 (닫지 않음!)


def get_rag_pipeline(
    conn: PgConnection = Depends(get_db_connection),
) -> RAGPipeline:
    settings = get_settings()
    return RAGPipeline(settings=settings, connection=conn)


def get_agent(
    conn: PgConnection = Depends(get_db_connection),
) -> BaseballStatisticsAgent:
    """Dependency to get an instance of the BaseballStatisticsAgent."""
    settings = get_settings()
    
    async def llm_generator(messages):
        import httpx
        if not settings.openrouter_api_key:
            raise RuntimeError("OpenRouter API key is required.")
        
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.openrouter_referer or "",
            "X-Title": settings.openrouter_app_title or "",
        }
        payload = {
            "model": settings.openrouter_model,
            "messages": list(messages),
            "max_tokens": settings.max_output_tokens,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("OpenRouter response is empty.")
        return content

    return BaseballStatisticsAgent(connection=conn, llm_generator=llm_generator)


def get_intent_router():
    return predict_intent