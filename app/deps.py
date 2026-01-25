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
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=30,
            dsn=settings.database_url,
            # TCP keepalive 옵션 추가
            keepalives=1,
            keepalives_idle=30,  # 30초 유휴 후 keepalive 시작
            keepalives_interval=10,  # 10초마다 keepalive 패킷
            keepalives_count=5,  # 5번 실패하면 연결 끊김으로 판단
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

    # tenacity settings
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception,
        before_sleep_log,
    )
    import httpx
    import logging
    import json

    logger = logging.getLogger("BaseballAgent")

    def is_server_error(exception):
        """Return True if exception is a 5xx server error."""
        return (
            isinstance(exception, httpx.HTTPStatusError)
            and exception.response.status_code >= 500
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_server_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def fetch_completion_stream(payload, headers):
        """Helper function to fetch stream with retry logic."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    yield line

    async def llm_generator(messages):
        if not settings.openrouter_api_key:
            raise RuntimeError("OpenRouter API key is required.")

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.openrouter_referer or "",
            "X-Title": settings.openrouter_app_title or "",
        }

        # Combine primary model with fallbacks
        models_to_try = [
            settings.openrouter_model
        ] + settings.openrouter_fallback_models

        last_exception = None

        for i, model in enumerate(models_to_try):
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": 0.1,
                "max_tokens": settings.max_output_tokens,
            }
            is_fallback = i > 0

            if is_fallback:
                logger.warning(
                    f"Switching to model {i}: {model} (Previous error: {last_exception})"
                )

            try:
                chunk_count = 0
                total_chars = 0
                # Reuse the same helper (retry on 5xx, fail fast on 429)
                async for line in fetch_completion_stream(payload, headers):
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = (
                                data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                chunk_count += 1
                                total_chars += len(delta)
                                yield delta
                        except json.JSONDecodeError:
                            continue

                # 스트림 완료 후 청크 수 로깅
                if chunk_count == 0:
                    logger.warning(
                        f"[LLM Generator] Stream completed but received 0 chunks from model {model}"
                    )
                else:
                    logger.debug(
                        f"[LLM Generator] Stream completed: {chunk_count} chunks, {total_chars} chars from model {model}"
                    )
                return  # Success!

            except Exception as e:
                logger.error(f"Model {model} failed: {e}")
                last_exception = e
                # Continue to next model in loop

        # If all models fail
        logger.error(f"All models failed. details: {last_exception}")
        raise last_exception

    return BaseballStatisticsAgent(connection=conn, llm_generator=llm_generator)


def get_intent_router():
    return predict_intent
