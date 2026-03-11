"""FastAPI 의존성 주입을 위한 공통 헬퍼를 정의하는 모듈."""

import asyncio
from collections.abc import Generator
from contextlib import asynccontextmanager
import logging
import secrets
from typing import Any, Optional

import psycopg
from psycopg_pool import ConnectionPool, PoolTimeout
from fastapi import Depends, Header, HTTPException, Request, status

logger = logging.getLogger(__name__)

from .config import get_settings
from .core.rag import RAGPipeline
from .ml.intent_router import predict_intent, load_clf
from .agents.baseball_agent import BaseballStatisticsAgent
from .core.chat_cache import CREATE_TABLE_SQL as CHAT_CACHE_DDL
from .core.chat_cache import cleanup_expired as _cleanup_expired_cache
from .core.security_metrics import record_security_event

# 전역 커넥션 풀 (앱 시작 시 한 번만 생성)
_connection_pool: Optional[ConnectionPool] = None
COACH_OPENROUTER_BLOCKED_MODELS = {
    "openrouter/auto",
    "upstage/solar-pro-3:free",
}
COACH_OPENROUTER_RETRY_LIMIT = 1
COACH_OPENROUTER_RETRY_BACKOFF_SECONDS = 0.75
COACH_OPENROUTER_MAX_TOKENS = 4000


def _extract_text_from_openrouter_content(content: Any) -> str:
    """OpenRouter delta/message content를 안전하게 문자열로 변환합니다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value:
                    parts.append(text_value)
                    continue
                nested_content = item.get("content")
                if isinstance(nested_content, str) and nested_content:
                    parts.append(nested_content)
        return "".join(parts)
    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
    return ""


def _parse_openrouter_stream_delta(payload: Any) -> tuple[str, str]:
    """SSE payload에서 텍스트 delta를 추출하고 파싱 상태 코드를 반환합니다."""
    if not isinstance(payload, dict):
        return "", "non_object_payload"

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return "", "missing_choices"
    if not choices:
        return "", "empty_choices"

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return "", "malformed_choice"

    delta_obj = first_choice.get("delta")
    if isinstance(delta_obj, dict):
        delta_text = _extract_text_from_openrouter_content(delta_obj.get("content"))
        if delta_text:
            return delta_text, "ok"

    message_obj = first_choice.get("message")
    if isinstance(message_obj, dict):
        message_text = _extract_text_from_openrouter_content(message_obj.get("content"))
        if message_text:
            return message_text, "ok"

    text_value = first_choice.get("text")
    if isinstance(text_value, str) and text_value:
        return text_value, "ok"

    if first_choice.get("finish_reason"):
        return "", "finished"
    return "", "empty_content"


def resolve_coach_openrouter_models(
    primary_model: str, fallback_models: list[str]
) -> list[str]:
    candidates: list[str] = []
    for model in [primary_model] + list(fallback_models):
        normalized = str(model or "").strip()
        if not normalized:
            continue
        if normalized in COACH_OPENROUTER_BLOCKED_MODELS:
            logger.warning("[Coach LLM] Skipping blocked model: %s", normalized)
            continue
        if normalized not in candidates:
            candidates.append(normalized)

    if candidates:
        return candidates

    fallback = str(primary_model or "").strip()
    if fallback:
        return [fallback]
    return ["openrouter/free"]


def is_retryable_coach_openrouter_error(exc: Exception) -> bool:
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500

    return isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            httpx.TransportError,
            httpx.ReadError,
        ),
    )


def clamp_coach_openrouter_max_tokens(requested_tokens: int) -> int:
    normalized = max(256, int(requested_tokens))
    return min(normalized, COACH_OPENROUTER_MAX_TOKENS)


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
                "target_session_attrs": "read-write",  # standby/recovery 서버 연결 거부
            },
        )

    return _connection_pool


def close_connection_pool():
    """앱 종료 시 커넥션 풀을 닫습니다."""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.close()
        _connection_pool = None


async def _chat_cache_cleanup_loop(interval_seconds: int = 3600) -> None:
    """1시간마다 만료된 chat_response_cache 항목을 삭제하는 백그라운드 루프.

    첫 실행은 interval_seconds 후 (앱 시작 직후 DB 부하 방지).
    일시적 DB 오류 발생 시 경고 로그만 남기고 루프를 계속 유지합니다.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            pool = get_connection_pool()
            with pool.connection() as conn:
                deleted = await _cleanup_expired_cache(conn)
            if deleted:
                logger.info("[ChatCache] Cleanup: %d expired entries deleted", deleted)
        except Exception as exc:
            logger.warning("[ChatCache] Cleanup loop error: %s", exc)


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
                prompt_version varchar(32) not null, -- e.g. "v2"
                model_name varchar(50) not null,     -- e.g. "upstage/solar-pro-3:free"
                status varchar(20) not null check (status in ('PENDING', 'COMPLETED', 'FAILED')),
                response_json jsonb,                 -- Completed analysis result
                error_message text,                  -- Failure reason
                created_at timestamptz default now(),
                updated_at timestamptz default now()
            );
            ALTER TABLE coach_analysis_cache
            ALTER COLUMN prompt_version TYPE varchar(32);
            CREATE INDEX IF NOT EXISTS idx_coach_cache_created_at ON coach_analysis_cache (created_at);
            CREATE INDEX IF NOT EXISTS idx_coach_cache_team_year ON coach_analysis_cache (team_id, year);
            """)
            # psycopg3 in pool context might need explicit commit if autocommit is not set?
            # connection pool is created with autocommit=True in get_connection_pool
    except Exception as e:
        print(f"[Warning] Failed to ensure coach_analysis_cache table: {e}")

    # [Chat Caching] chat_response_cache 테이블 자동 생성
    try:
        with pool.connection() as conn:
            conn.execute(CHAT_CACHE_DDL)
        logger.info("[Lifespan] chat_response_cache table ensured")
    except Exception as exc:
        logger.warning("[Lifespan] chat_response_cache DDL failed: %s", exc)

    # [Chat Caching] 만료 항목 주기적 삭제 백그라운드 태스크 시작
    cleanup_task = asyncio.create_task(_chat_cache_cleanup_loop())
    logger.info("[Lifespan] chat_response_cache cleanup task started (interval=1h)")

    yield

    # 종료 시: cleanup 태스크 취소 후 커넥션 풀 정리
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    close_connection_pool()  # 모든 커넥션 정리


def get_db_connection() -> Generator[psycopg.Connection, None, None]:
    """커넥션 풀에서 커넥션을 가져와서 사용 후 반환합니다."""
    pool_instance = get_connection_pool()

    # psycopg_pool은 context manager를 지원하여 안전하게 반환함
    try:
        with pool_instance.connection() as conn:
            # 연결 상태 확인은 psycopg3에서 더 지능적으로 처리되지만
            # 명시적인 확인이 필요한 경우 execute("SELECT 1") 등을 사용 가능
            yield conn
    except PoolTimeout as exc:
        logger.error("[DB] Pool timeout while acquiring connection: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="데이터베이스 연결이 일시적으로 불안정합니다. 잠시 후 다시 시도해주세요.",
        ) from exc
    except psycopg.OperationalError as exc:
        logger.error("[DB] Operational error while acquiring connection: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="데이터베이스 연결이 복구 중입니다. 잠시 후 다시 시도해주세요.",
        ) from exc


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

    def _resolve_model_candidates(
        primary_model: str, fallback_models: list[str]
    ) -> list[str]:
        blocked = {"openrouter/auto"}
        candidates: list[str] = []
        for model in [primary_model] + list(fallback_models):
            if not model:
                continue
            if model in blocked:
                logger.warning(f"[LLM] Skipping blocked model: {model}")
                continue
            if model not in candidates:
                candidates.append(model)

        if candidates:
            return candidates

        if primary_model:
            logger.warning(
                f"[LLM] All configured models are blocked; fallback to primary: {primary_model}"
            )
            return [primary_model]

        return [m for m in fallback_models if m]

    @retry(
        stop=stop_after_attempt(2),  # 5 -> 2: 빠른 실패
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(is_server_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def fetch_completion_stream(payload, headers):
        """Helper function to fetch stream with retry logic."""
        # Timeout: allow longer streaming windows before marking model as failed.
        timeout_config = httpx.Timeout(30.0, connect=5.0, read=30.0, pool=5.0)
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

        # Build model list to try (블록된 모델은 건너뛰고, 없으면 primary로 복귀)
        primary_model = settings.openrouter_model
        fallback_models = settings.openrouter_fallback_models
        models_to_try = _resolve_model_candidates(primary_model, fallback_models)
        logger.info(
            f"[LLM] Models to try (filtered): {models_to_try}, max_tokens={effective_max_tokens}"
        )

        last_exception = None
        empty_chunk_retries = max(0, int(settings.chat_openrouter_empty_chunk_retries))
        empty_chunk_backoff_ms = max(
            50, int(settings.chat_openrouter_empty_chunk_backoff_ms)
        )

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

            for retry_index in range(empty_chunk_retries + 1):
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.1,
                    "max_tokens": effective_max_tokens,
                }

                try:
                    chunk_count = 0
                    empty_choice_count = 0
                    malformed_chunk_count = 0

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
                                delta, parse_reason = _parse_openrouter_stream_delta(
                                    data
                                )
                                if parse_reason in {"missing_choices", "empty_choices"}:
                                    empty_choice_count += 1
                                elif parse_reason in {
                                    "non_object_payload",
                                    "malformed_choice",
                                }:
                                    malformed_chunk_count += 1
                                if delta:
                                    chunk_count += 1
                                    yield delta
                            except json.JSONDecodeError:
                                malformed_chunk_count += 1
                                continue
                        else:
                            # Log non-data lines (might be errors or metadata)
                            if line and not line.startswith(":"):
                                logger.info(f"[OpenRouter Raw] {model}: {line}")
                                if "error" in line.lower():
                                    logger.error(f"[OpenRouter Error Detail] {line}")

                    if chunk_count == 0:
                        if malformed_chunk_count > 0:
                            empty_chunk_reason = "malformed_stream_payload"
                        elif empty_choice_count > 0:
                            empty_chunk_reason = "empty_choices"
                        else:
                            empty_chunk_reason = "empty_chunk"
                        error_msg = (
                            f"Empty response (0 chunks) from {model}. "
                            "Check filters or token limits."
                        )
                        logger.warning(
                            "[LLM EmptyChunk] model=%s retry_index=%d max_retries=%d max_tokens=%s reason=%s empty_choices=%d malformed_chunks=%d",
                            model,
                            retry_index,
                            empty_chunk_retries,
                            effective_max_tokens,
                            empty_chunk_reason,
                            empty_choice_count,
                            malformed_chunk_count,
                        )
                        last_exception = RuntimeError(error_msg)
                        if retry_index < empty_chunk_retries:
                            backoff_seconds = (empty_chunk_backoff_ms / 1000.0) * (
                                2**retry_index
                            )
                            await asyncio.sleep(backoff_seconds)
                            continue
                        break

                    if empty_choice_count > 0 or malformed_chunk_count > 0:
                        logger.info(
                            "[LLM StreamParse] model=%s chunk_count=%d empty_choices=%d malformed_chunks=%d",
                            model,
                            chunk_count,
                            empty_choice_count,
                            malformed_chunk_count,
                        )
                    logger.info(f"[LLM] Success: {chunk_count} chunks from {model}")
                    return  # Success, exit the loop

                except Exception as e:
                    logger.error(f"[LLM] Model {model} failed: {e}")
                    last_exception = e
                    break  # Try next model

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

    return BaseballStatisticsAgent(
        connection=conn,
        llm_generator=llm_generator,
        fast_path_enabled=settings.chat_fast_path_enabled,
        fast_path_scope=settings.chat_fast_path_scope,
        fast_path_min_messages=settings.chat_fast_path_min_messages,
        fast_path_tool_cap=settings.chat_fast_path_tool_cap,
        fast_path_fallback_on_empty=settings.chat_fast_path_fallback_on_empty,
        chat_dynamic_token_enabled=settings.chat_dynamic_token_enabled,
        chat_analysis_max_tokens=settings.chat_analysis_max_tokens,
        chat_answer_max_tokens_short=settings.chat_answer_max_tokens_short,
        chat_answer_max_tokens_long=settings.chat_answer_max_tokens_long,
        chat_answer_max_tokens_team=settings.chat_answer_max_tokens_team,
        chat_tool_result_max_chars=settings.chat_tool_result_max_chars,
        chat_tool_result_max_items=settings.chat_tool_result_max_items,
        chat_first_token_watchdog_seconds=settings.chat_first_token_watchdog_seconds,
        chat_first_token_retry_max_attempts=settings.chat_first_token_retry_max_attempts,
        chat_stream_first_token_watchdog_seconds=settings.chat_stream_first_token_watchdog_seconds,
        chat_stream_first_token_retry_max_attempts=settings.chat_stream_first_token_retry_max_attempts,
    )


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

        effective_max_tokens = clamp_coach_openrouter_max_tokens(max_tokens)
        if effective_max_tokens != max_tokens:
            logger.warning(
                "[Coach LLM] Clamped max_tokens from %d to %d",
                max_tokens,
                effective_max_tokens,
            )

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.openrouter_referer or "",
            "X-Title": settings.openrouter_app_title or "",
        }

        primary_model = settings.coach_openrouter_model or settings.openrouter_model
        fallback_models = list(settings.coach_openrouter_fallback_models)
        models_to_try = resolve_coach_openrouter_models(primary_model, fallback_models)

        logger.info(
            "[Coach LLM] OpenRouter models=%s, max_tokens=%d",
            models_to_try,
            effective_max_tokens,
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

            for retry_index in range(COACH_OPENROUTER_RETRY_LIMIT + 1):
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.1,
                    "top_p": 0.5,
                    "max_tokens": effective_max_tokens,
                }

                try:
                    chunk_count = 0
                    empty_choice_count = 0
                    malformed_chunk_count = 0
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
                            if (
                                response.status_code >= 400
                                and response.status_code < 500
                            ):
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
                                        delta, parse_reason = (
                                            _parse_openrouter_stream_delta(data)
                                        )
                                        if parse_reason in {
                                            "missing_choices",
                                            "empty_choices",
                                        }:
                                            empty_choice_count += 1
                                        elif parse_reason in {
                                            "non_object_payload",
                                            "malformed_choice",
                                        }:
                                            malformed_chunk_count += 1
                                        if delta:
                                            chunk_count += 1
                                            yield delta
                                    except json.JSONDecodeError:
                                        malformed_chunk_count += 1
                                        continue
                                else:
                                    if line and not line.startswith(":"):
                                        logger.info(
                                            "[Coach OpenRouter Raw] %s: %s", model, line
                                        )
                                        if "error" in line.lower():
                                            logger.error(
                                                "[Coach OpenRouter Error Detail] %s",
                                                line,
                                            )

                    if chunk_count == 0:
                        if malformed_chunk_count > 0:
                            empty_chunk_reason = "malformed_stream_payload"
                        elif empty_choice_count > 0:
                            empty_chunk_reason = "empty_choices"
                        else:
                            empty_chunk_reason = "empty_chunk"
                        error_msg = (
                            f"Empty response (0 chunks) from {model}. "
                            f"reason={empty_chunk_reason}, empty_choices={empty_choice_count}, malformed={malformed_chunk_count}"
                        )
                        last_exception = RuntimeError(error_msg)
                        if retry_index < COACH_OPENROUTER_RETRY_LIMIT:
                            logger.warning(
                                "[Coach LLM Retry] model=%s retry_index=%d/%d reason=%s",
                                model,
                                retry_index + 1,
                                COACH_OPENROUTER_RETRY_LIMIT + 1,
                                empty_chunk_reason,
                            )
                            await asyncio.sleep(
                                COACH_OPENROUTER_RETRY_BACKOFF_SECONDS
                                * (2**retry_index)
                            )
                            continue
                        logger.warning("[Coach LLM] %s", error_msg)
                        break
                    if empty_choice_count > 0 or malformed_chunk_count > 0:
                        logger.info(
                            "[Coach LLM StreamParse] model=%s chunk_count=%d empty_choices=%d malformed_chunks=%d",
                            model,
                            chunk_count,
                            empty_choice_count,
                            malformed_chunk_count,
                        )
                    logger.info(
                        "[Coach LLM] Success: %d chunks from %s", chunk_count, model
                    )
                    return
                except Exception as e:
                    retryable = is_retryable_coach_openrouter_error(e)
                    logger.error(
                        "[Coach LLM] OpenRouter model %s failed attempt=%d/%d retryable=%s: %s",
                        model,
                        retry_index + 1,
                        COACH_OPENROUTER_RETRY_LIMIT + 1,
                        retryable,
                        e,
                    )
                    last_exception = e
                    if retryable and retry_index < COACH_OPENROUTER_RETRY_LIMIT:
                        await asyncio.sleep(
                            COACH_OPENROUTER_RETRY_BACKOFF_SECONDS * (2**retry_index)
                        )
                        continue
                    break

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


def _extract_internal_token_from_authorization(authorization: str) -> str:
    candidate = (authorization or "").strip()
    if not candidate:
        return ""
    if candidate.lower().startswith("bearer "):
        return candidate[7:].strip()
    return candidate


def require_ai_internal_token(
    request: Request,
    x_internal_api_key: str = Header(default="", alias="X-Internal-Api-Key"),
    authorization: str = Header(default="", alias="Authorization"),
) -> None:
    settings = get_settings()
    expected_token = (getattr(settings, "resolved_ai_internal_token", "") or "").strip()
    endpoint = request.url.path if request is not None else "unknown"

    if not expected_token:
        logger.error("AI internal token is not configured.")
        record_security_event(
            "AI_INTERNAL_AUTH_MISCONFIGURED",
            endpoint=endpoint,
            detail="missing_ai_internal_token",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI internal authentication is not configured",
        )

    provided_token = (
        x_internal_api_key or ""
    ).strip() or _extract_internal_token_from_authorization(authorization)
    if not provided_token or not secrets.compare_digest(provided_token, expected_token):
        record_security_event(
            "AI_INTERNAL_AUTH_REJECT",
            endpoint=endpoint,
            detail="missing_or_invalid_token",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API token",
        )
