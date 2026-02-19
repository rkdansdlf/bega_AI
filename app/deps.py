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
    pool = get_connection_pool()  # 커넥션 풀 초기화

    # [Coach Caching] 캐시 테이블 자동 생성 (편의성)
    try:
        with pool.connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS coach_analysis_cache (
                cache_key varchar(64) primary key,  -- SHA256 Hash of (team_id, year, focus, question)
                team_id varchar(10) not null,
                year int not null,
                prompt_version varchar(10) not null, -- e.g. "v2"
                model_name varchar(50) not null,     -- e.g. "upstage/solar-pro-3:free"
                status varchar(20) not null check (status in ('PENDING', 'COMPLETED', 'FAILED')),
                response_json jsonb,                 -- Completed analysis result
                error_message text,                  -- Failure reason
                created_at timestamptz default now(),
                updated_at timestamptz default now()
            );
            CREATE INDEX IF NOT EXISTS idx_coach_cache_created_at ON coach_analysis_cache (created_at);
            CREATE INDEX IF NOT EXISTS idx_coach_cache_team_year ON coach_analysis_cache (team_id, year);
            """)
            # psycopg3 in pool context might need explicit commit if autocommit is not set?
            # connection pool is created with autocommit=True in get_connection_pool
    except Exception as e:
        print(f"[Warning] Failed to ensure coach_analysis_cache table: {e}")

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
        stop=stop_after_attempt(2),  # 5 -> 2: 빠른 실패
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(is_server_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def fetch_completion_stream(payload, headers):
        """Helper function to fetch stream with retry logic."""
        # 타임아웃: read=10s, total=20s (기존 60s에서 대폭 단축)
        timeout_config = httpx.Timeout(10.0, connect=5.0, read=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            async with client.stream(
                "POST",
                f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                # Log error response body for 4xx errors before raising
                if response.status_code >= 400 and response.status_code < 500:
                    error_body = await response.aread()
                    logger.error(
                        f"[OpenRouter 4xx] Status: {response.status_code}, Body: {error_body.decode('utf-8', errors='replace')}"
                    )
                response.raise_for_status()
                async for line in response.aiter_lines():
                    yield line

    async def openrouter_generator(messages, max_tokens=None):
        """OpenRouter LLM generator with optional max_tokens override."""
        if not settings.openrouter_api_key:
            raise RuntimeError("OpenRouter API key is required.")

        # max_tokens 오버라이드 지원
        effective_max_tokens = max_tokens or settings.max_output_tokens

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.openrouter_referer or "",
            "X-Title": settings.openrouter_app_title or "",
        }

        # Build list of models to try: primary + fallbacks
        # [PATCH] openrouter/free 제거 - 무료 라우터는 큐잉/속도 저하 심각
        primary_model = settings.openrouter_model
        fallback_models = [
            m
            for m in settings.openrouter_fallback_models
            if m not in ("openrouter/free", "openrouter/auto")
        ]
        models_to_try = [primary_model] + fallback_models
        logger.info(
            f"[LLM] Models to try (filtered): {models_to_try}, max_tokens={effective_max_tokens}"
        )

        last_exception = None

        for i, model in enumerate(models_to_try):
            is_fallback = i > 0
            if is_fallback:
                logger.warning(
                    f"[LLM Fallback] Trying model {i+1}/{len(models_to_try)}: {model}"
                )
            else:
                logger.info(
                    f"[LLM] Primary: {model}, Fallbacks available: {fallback_models}"
                )

            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": 0.1,
                "max_tokens": effective_max_tokens,
            }

            try:
                chunk_count = 0
                total_chars = 0

                async for line in fetch_completion_stream(payload, headers):
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:].strip()
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
                    else:
                        # Log non-data lines (might be errors or metadata)
                        if line and not line.startswith(":"):
                            logger.info(f"[OpenRouter Raw] {model}: {line}")
                            if "error" in line.lower():
                                logger.error(f"[OpenRouter Error Detail] {line}")

                # Check if we got a valid response
                if chunk_count == 0:
                    error_msg = f"Empty response (0 chunks) from {model}. Check filters or token limits."
                    logger.warning(f"[LLM] {error_msg}")
                    last_exception = RuntimeError(error_msg)
                    continue  # Try next model
                else:
                    logger.info(f"[LLM] Success: {chunk_count} chunks from {model}")
                    return  # Success, exit the loop

            except Exception as e:
                logger.error(f"[LLM] Model {model} failed: {e}")
                last_exception = e
                continue  # Try next model

        # All models failed
        logger.error(
            f"[LLM] All {len(models_to_try)} models failed. Last error: {last_exception}"
        )
        raise last_exception or RuntimeError("All models failed")

    async def gemini_generator(messages, max_tokens=None):
        """Gemini LLM generator with optional max_tokens override."""
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig

        if not settings.gemini_api_key:
            raise RuntimeError("Gemini API key is required.")

        # max_tokens 오버라이드 지원
        effective_max_tokens = max_tokens or settings.max_output_tokens

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
                    max_output_tokens=effective_max_tokens,
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


def get_coach_llm_generator():
    """
    Coach 전용 LLM generator를 반환합니다.

    OpenRouter만 지원합니다. openrouter_model 및 openrouter_fallback_models
    설정을 통해 사용할 모델을 지정할 수 있습니다.
    """
    import logging
    import json
    import httpx

    logger = logging.getLogger("CoachLLM")
    settings = get_settings()

    async def coach_openrouter_generator(messages, max_tokens: int):
        """Coach 전용 OpenRouter generator."""
        if not settings.openrouter_api_key:
            raise RuntimeError("OpenRouter API key is required for Coach.")

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.openrouter_referer or "",
            "X-Title": settings.openrouter_app_title or "",
        }

        primary_model = settings.coach_openrouter_model or settings.openrouter_model
        # [PATCH] openrouter/free 제거 - 무료 라우터는 빈 응답/큐잉 문제 발생
        fallback_models = [
            m
            for m in settings.coach_openrouter_fallback_models
            if m not in ("openrouter/free", "openrouter/auto")
        ]
        models_to_try = [primary_model] + fallback_models

        logger.info(
            "[Coach LLM] OpenRouter models=%s, max_tokens=%d",
            models_to_try,
            max_tokens,
        )

        last_exception = None
        for i, model in enumerate(models_to_try):
            is_fallback = i > 0
            if is_fallback:
                logger.warning(
                    "[Coach LLM Fallback] Trying model %d/%d: %s",
                    i + 1,
                    len(models_to_try),
                    model,
                )

            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": 0.1,
                "top_p": 0.5,
                "max_tokens": max_tokens,
            }

            try:
                chunk_count = 0
                timeout_config = httpx.Timeout(
                    settings.coach_llm_read_timeout,
                    connect=5.0,
                    read=settings.coach_llm_read_timeout,
                    pool=5.0,
                )
                async with httpx.AsyncClient(timeout=timeout_config) as client:
                    async with client.stream(
                        "POST",
                        f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
                        json=payload,
                        headers=headers,
                    ) as response:
                        if response.status_code >= 400 and response.status_code < 500:
                            error_body = await response.aread()
                            logger.error(
                                "[Coach OpenRouter 4xx] Status: %s, Body: %s",
                                response.status_code,
                                error_body.decode("utf-8", errors="replace"),
                            )
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
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
                                        yield delta
                                except json.JSONDecodeError:
                                    continue
                            else:
                                if line and not line.startswith(":"):
                                    logger.info(
                                        "[Coach OpenRouter Raw] %s: %s", model, line
                                    )
                                    if "error" in line.lower():
                                        logger.error(
                                            "[Coach OpenRouter Error Detail] %s", line
                                        )

                if chunk_count == 0:
                    error_msg = f"Empty response (0 chunks) from {model}."
                    logger.warning("[Coach LLM] %s", error_msg)
                    last_exception = RuntimeError(error_msg)
                    continue
                logger.info(
                    "[Coach LLM] Success: %d chunks from %s", chunk_count, model
                )
                return
            except Exception as e:
                logger.error("[Coach LLM] OpenRouter model %s failed: %s", model, e)
                last_exception = e
                continue

        raise last_exception or RuntimeError("All OpenRouter models failed")

    async def coach_llm(messages, max_tokens=None):
        """Coach LLM entrypoint (OpenRouter only).

        Note: Coach feature only supports OpenRouter.
        COACH_OPENROUTER_MODEL/COACH_OPENROUTER_FALLBACK_MODELS settings
        control which models are used.
        """
        effective_max_tokens = max_tokens or settings.coach_max_output_tokens

        try:
            async for chunk in coach_openrouter_generator(
                messages, effective_max_tokens
            ):
                yield chunk
        except Exception as e:
            logger.error("[Coach LLM] OpenRouter failed: %s", e)
            raise

    return coach_llm


def get_intent_router():
    return predict_intent
