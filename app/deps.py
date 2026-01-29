"""FastAPI 의존성 주입을 위한 공통 헬퍼를 정의하는 모듈."""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from typing import Optional

import psycopg
from psycopg_pool import ConnectionPool
from fastapi import Depends
from fastapi.params import Depends as DependsClass

from .config import get_settings
from .core.rag import RAGPipeline
from .ml.intent_router import predict_intent, load_clf
from .agents.baseball_agent import BaseballStatisticsAgent
from .core.prompts import SYSTEM_PROMPT

# 전역 커넥션 풀 (앱 시작 시 한 번만 생성)
_connection_pool: Optional[ConnectionPool] = None


def get_connection_pool() -> ConnectionPool:
    """커넥션 풀을 가져오거나 생성합니다."""
    global _connection_pool

    if _connection_pool is None:
        settings = get_settings()
        _connection_pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=30,
            # TCP keepalive 옵션 및 기타 설정
            kwargs={
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
                "autocommit": True,
            },
        )

    return _connection_pool


def close_connection_pool():
    """앱 종료 시 커넥션 풀을 닫습니다."""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.close()
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


def get_db_connection() -> Generator[psycopg.Connection, None, None]:
    """커넥션 풀에서 커넥션을 가져와서 사용 후 반환합니다."""
    pool_instance = get_connection_pool()

    # psycopg_pool은 context manager를 지원하여 안전하게 반환함
    with pool_instance.connection() as conn:
        # 연결 상태 확인은 psycopg3에서 더 지능적으로 처리되지만
        # 명시적인 확인이 필요한 경우 execute("SELECT 1") 등을 사용 가능
        yield conn


def get_rag_pipeline(
    conn: psycopg.Connection = Depends(get_db_connection),
) -> RAGPipeline:
    settings = get_settings()
    return RAGPipeline(settings=settings, connection=conn)


def get_agent(
    conn: psycopg.Connection = Depends(get_db_connection),
) -> BaseballStatisticsAgent:
    """Dependency to get an instance of the BaseballStatisticsAgent."""
    # Note: depends handles the context manager yield for us.
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

    async def openrouter_generator(messages):
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

    async def gemini_generator(messages):
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig

        if not settings.gemini_api_key:
            raise RuntimeError("Gemini API key is required.")

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)

        # Convert messages to Gemini format
        gemini_messages = []
        system_instruction = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction = content
            else:
                gemini_role = "user" if role == "user" else "model"
                gemini_messages.append({"role": gemini_role, "parts": [content]})

        if system_instruction:
            model = genai.GenerativeModel(
                model_name=settings.gemini_model, system_instruction=system_instruction
            )

        try:
            response = await model.generate_content_async(
                gemini_messages,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=settings.max_output_tokens,
                ),
                stream=True,
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise e

    # Select generator based on provider
    if settings.llm_provider == "gemini":
        llm_generator = gemini_generator
    else:
        llm_generator = openrouter_generator

    return BaseballStatisticsAgent(connection=conn, llm_generator=llm_generator)


def get_intent_router():
    return predict_intent
