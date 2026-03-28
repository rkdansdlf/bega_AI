"""FastAPI 의존성 주입을 위한 공통 헬퍼를 정의하는 모듈."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
import json
import logging
import secrets
from typing import Any, Optional

import psycopg
import httpx
from psycopg_pool import ConnectionPool, PoolTimeout
from fastapi import Depends, Header, HTTPException, Request, status

logger = logging.getLogger(__name__)

from .config import get_settings
from .core.http_clients import close_shared_httpx_clients, get_shared_httpx_client
from .core.rag import RAGPipeline
from .ml.intent_router import predict_intent, load_clf
from .agents.baseball_agent import BaseballAgentRuntime, BaseballStatisticsAgent
from .agents.shared_runtime import (
    get_shared_baseball_agent_runtime as _get_shared_baseball_agent_runtime,
    initialize_shared_baseball_agent_runtime as _initialize_cached_baseball_agent_runtime,
    reset_shared_baseball_agent_runtime,
)
from .agents.runtime_factory import create_baseball_agent_runtime
from .core.chat_cache import CREATE_TABLE_SQL as CHAT_CACHE_DDL
from .core.chat_cache import cleanup_expired as _cleanup_expired_cache
from .core.security_metrics import record_security_event

# 전역 커넥션 풀 (앱 시작 시 한 번만 생성)
_connection_pool: Optional[ConnectionPool] = None
DB_POOL_MIN_SIZE = 1
DB_POOL_MAX_SIZE = 30

# 전역 싱글톤: stateless 컴포넌트 (앱 수명 동안 재사용)
_shared_context_formatter = None
_shared_wpa_calculator = None


def _get_shared_context_formatter():
    """ContextFormatter 싱글톤 반환."""
    global _shared_context_formatter
    if _shared_context_formatter is None:
        from .core.rag import ContextFormatter
        _shared_context_formatter = ContextFormatter()
    return _shared_context_formatter


def _get_shared_wpa_calculator():
    """WPACalculator 싱글톤 반환."""
    global _shared_wpa_calculator
    if _shared_wpa_calculator is None:
        from .core.rag import WPACalculator
        _shared_wpa_calculator = WPACalculator()
    return _shared_wpa_calculator


def _create_baseball_agent_runtime(
    settings: Optional[Any] = None,
) -> BaseballAgentRuntime:
    return create_baseball_agent_runtime(settings)


def _initialize_shared_baseball_agent_runtime() -> BaseballAgentRuntime:
    runtime = _initialize_cached_baseball_agent_runtime()
    logger.info(
        "[Lifespan] baseball_agent_runtime initialized runtime_id=%d",
        runtime.runtime_id,
    )
    return runtime


def get_shared_baseball_agent_runtime() -> BaseballAgentRuntime:
    return _get_shared_baseball_agent_runtime()
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
            min_size=DB_POOL_MIN_SIZE,
            max_size=DB_POOL_MAX_SIZE,
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
        logger.info(
            "[DB] Connection pool initialized pool_stats=%s",
            _format_connection_pool_stats(_connection_pool),
        )

    return _connection_pool


def _snapshot_connection_pool_stats(
    pool_instance: Optional[Any] = None,
) -> dict[str, Any]:
    pool = pool_instance or _connection_pool
    snapshot: dict[str, Any] = {
        "min_size": DB_POOL_MIN_SIZE,
        "max_size": DB_POOL_MAX_SIZE,
        "pool_available": pool is not None,
    }
    if pool is None:
        return snapshot

    pool_name = getattr(pool, "name", None)
    if isinstance(pool_name, str) and pool_name:
        snapshot["name"] = pool_name

    get_stats = getattr(pool, "get_stats", None)
    if callable(get_stats):
        try:
            stats = get_stats()
        except Exception as exc:  # noqa: BLE001
            snapshot["stats_error"] = str(exc)
        else:
            if isinstance(stats, dict):
                snapshot["stats"] = stats

    return snapshot


def _format_connection_pool_stats(pool_instance: Optional[Any] = None) -> str:
    return json.dumps(
        _snapshot_connection_pool_stats(pool_instance),
        ensure_ascii=False,
        sort_keys=True,
    )


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
    _initialize_shared_baseball_agent_runtime()

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
                error_code varchar(64),
                attempt_count int not null default 0,
                lease_owner varchar(80),
                lease_expires_at timestamptz,
                last_heartbeat_at timestamptz,
                created_at timestamptz default now(),
                updated_at timestamptz default now()
            );
            ALTER TABLE coach_analysis_cache
            ALTER COLUMN prompt_version TYPE varchar(32);
            ALTER TABLE coach_analysis_cache
            ADD COLUMN IF NOT EXISTS error_code varchar(64);
            ALTER TABLE coach_analysis_cache
            ADD COLUMN IF NOT EXISTS attempt_count int NOT NULL DEFAULT 0;
            ALTER TABLE coach_analysis_cache
            ADD COLUMN IF NOT EXISTS lease_owner varchar(80);
            ALTER TABLE coach_analysis_cache
            ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz;
            ALTER TABLE coach_analysis_cache
            ADD COLUMN IF NOT EXISTS last_heartbeat_at timestamptz;
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

    # 종료 시: cleanup 태스크 취소 후 리소스 정리
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # httpx 클라이언트 정리
    await close_shared_httpx_clients()

    reset_shared_baseball_agent_runtime()
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
        logger.error(
            "[DB] Pool timeout while acquiring connection: %s pool_stats=%s",
            exc,
            _format_connection_pool_stats(pool_instance),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="데이터베이스 연결이 일시적으로 불안정합니다. 잠시 후 다시 시도해주세요.",
        ) from exc
    except psycopg.OperationalError as exc:
        logger.error(
            "[DB] Operational error while acquiring connection: %s pool_stats=%s",
            exc,
            _format_connection_pool_stats(pool_instance),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="데이터베이스 연결이 복구 중입니다. 잠시 후 다시 시도해주세요.",
        ) from exc


def get_rag_pipeline(
    conn: psycopg.Connection = Depends(get_db_connection),
) -> RAGPipeline:
    settings = get_settings()
    return RAGPipeline(
        settings=settings,
        connection=conn,
        agent_runtime=get_shared_baseball_agent_runtime(),
        context_formatter=_get_shared_context_formatter(),
        wpa_calculator=_get_shared_wpa_calculator(),
    )


async def get_agent(
    conn: psycopg.Connection = Depends(get_db_connection),
) -> AsyncGenerator[BaseballStatisticsAgent, None]:
    """Dependency to get an instance of the BaseballStatisticsAgent."""
    runtime = get_shared_baseball_agent_runtime()
    with runtime.request_context(conn):
        yield runtime.shared_agent


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
                    client = get_shared_httpx_client(
                        "openrouter",
                        timeout=120.0,
                        limits=httpx.Limits(
                            max_connections=20,
                            max_keepalive_connections=10,
                        ),
                    )
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
