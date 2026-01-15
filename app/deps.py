"""FastAPI 의존성 주입을 위한 공통 헬퍼를 정의하는 모듈."""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from typing import Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extensions import connection as PgConnection
from fastapi import Depends
from fastapi.params import Depends as DependsClass

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
            minconn=1,
            maxconn=5,
            dsn=settings.database_url,
            # TCP keepalive 옵션 추가
            keepalives=1,
            keepalives_idle=30,      # 30초 유휴 후 keepalive 시작
            keepalives_interval=10,  # 10초마다 keepalive 패킷
            keepalives_count=5       # 5번 실패하면 연결 끊김으로 판단
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
    conn = pool_instance.getconn()
    
    # 연결 상태 확인 (stale connection 방지)
    try:
        # 간단한 쿼리로 연결이 살아있는지 확인
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.rollback()  # 트랜잭션 종료 (autocommit 설정 전에 필요)
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        # 연결이 끊어졌으면 풀에서 제거하고 새 연결 생성
        pool_instance.putconn(conn, close=True)
        conn = pool_instance.getconn()
    
    conn.autocommit = True
    
    try:
        yield conn
    finally:
        pool_instance.putconn(conn)

def get_rag_pipeline(
    conn: PgConnection = Depends(get_db_connection),
) -> RAGPipeline:
    settings = get_settings()
    return RAGPipeline(settings=settings, connection=conn)


def get_agent(
    conn: PgConnection = Depends(get_db_connection),
) -> BaseballStatisticsAgent:
    """Dependency to get an instance of the BaseballStatisticsAgent.
    
    NOTE: If called directly (outside FastAPI request context), 
    conn will be a Depends object, not a connection. We detect this
    and manually obtain a connection from the pool.
    """
    # Handle direct calls (not via FastAPI DI)
    if isinstance(conn, DependsClass):
        pool_instance = get_connection_pool()
        conn = pool_instance.getconn()
        conn.autocommit = True
    settings = get_settings()
    
    async def llm_generator(messages):
        import httpx
        import json
        if not settings.openrouter_api_key:
            raise RuntimeError("OpenRouter API key is required.")
        
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.openrouter_referer or "",
            "X-Title": settings.openrouter_app_title or "",
        }
        # 스트리밍 활성화
        payload = {
            "model": settings.openrouter_model,
            "messages": list(messages),
            "max_tokens": settings.max_output_tokens,
            "stream": True 
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                yield delta
                        except json.JSONDecodeError:
                            continue

    return BaseballStatisticsAgent(connection=conn, llm_generator=llm_generator)


def get_intent_router():
    return predict_intent