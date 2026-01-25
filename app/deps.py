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
    """Dependency to get an instance of the BaseballStatisticsAgent."""
    if isinstance(conn, DependsClass):
        pool_instance = get_connection_pool()
        conn = pool_instance.getconn()
        conn.autocommit = True
    settings = get_settings()

    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, before_sleep_log
    import httpx
    import logging
    import json
    import asyncio
    
    logger = logging.getLogger("BaseballAgent")

    def is_server_error(exception):
        return (
            isinstance(exception, httpx.HTTPStatusError) and 
            exception.response.status_code >= 500
        )

    async def llm_generator(messages):
        provider = settings.llm_provider
        
        if provider == "gemini":
            import google.generativeai as genai
            if not settings.gemini_api_key:
                raise RuntimeError("Gemini API key is required.")
            
            genai.configure(api_key=settings.gemini_api_key)
            
            # Gemini format conversion
            gemini_contents = []
            system_instruction = ""
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "system":
                    system_instruction += content + "\n"
                elif role == "user":
                    gemini_contents.append({"role": "user", "parts": [{"text": content}]})
                elif role == "assistant":
                    gemini_contents.append({"role": "model", "parts": [{"text": content}]})
            
            model_name = settings.gemini_model or "gemini-1.5-flash"
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction if system_instruction else None
                )
                
                # generate_content stream=True returns a response iterative object
                response = model.generate_content(
                    gemini_contents,
                    stream=True,
                    generation_config={
                        "max_output_tokens": settings.max_output_tokens,
                        "temperature": 0.1,
                    }
                )
                
                for chunk in response:
                    # In some SDK versions, chunk might have an attribute 'text'
                    try:
                        if chunk.text:
                            yield chunk.text
                    except Exception:
                        # Sometimes text is not available if safety filters block it
                        continue
                return
            except Exception as e:
                logger.error(f"Gemini SDK streaming failed: {e}")
                # Fallback to older format or simpler initialization
                try:
                    if system_instruction and gemini_contents:
                        gemini_contents[0]["parts"][0]["text"] = f"System: {system_instruction}\n\n{gemini_contents[0]['parts'][0]['text']}"
                    
                    model = genai.GenerativeModel(model_name=model_name)
                    response = model.generate_content(
                        gemini_contents,
                        stream=True,
                        generation_config={
                            "max_output_tokens": settings.max_output_tokens,
                            "temperature": 0.1,
                        }
                    )
                    for chunk in response:
                        try:
                            if chunk.text:
                                yield chunk.text
                        except Exception:
                            continue
                    return
                except Exception as fallback_e:
                    logger.error(f"Gemini SDK fallback failed: {fallback_e}")
                    raise fallback_e

        # Default to OpenRouter/OpenAI logic if provider is openrouter
        if not settings.openrouter_api_key:
            raise RuntimeError("OpenRouter API key is required.")

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.openrouter_referer or "",
            "X-Title": settings.openrouter_app_title or "",
        }

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception(is_server_error),
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )
        async def fetch_completion_stream(payload, headers):
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

        models_to_try = [settings.openrouter_model] + settings.openrouter_fallback_models
        last_exception = None

        for i, model in enumerate(models_to_try):
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": 0.1,
                "max_tokens": settings.max_output_tokens
            }
            try:
                async for line in fetch_completion_stream(payload, headers):
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
                return
            except Exception as e:
                logger.error(f"Model {model} failed: {e}")
                last_exception = e

        raise last_exception

    return BaseballStatisticsAgent(connection=conn, llm_generator=llm_generator)


def get_intent_router():
    return predict_intent