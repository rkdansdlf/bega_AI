"""FastAPI 의존성 주입을 위한 공통 헬퍼를 정의하는 모듈."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import psycopg
import httpx
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from fastapi import Depends, HTTPException, status

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
from .core.chat_semantic_cache import CREATE_TABLE_SQL as CHAT_SEMANTIC_CACHE_DDL
from .core.chat_semantic_cache import (
    CREATE_SHADOW_OBSERVATION_TABLE_SQL as CHAT_SEMANTIC_SHADOW_OBSERVATION_DDL,
    CREATE_VECTOR_INDEX_SQL as CHAT_SEMANTIC_CACHE_VECTOR_INDEX_DDL,
)
from .core.chat_semantic_cache import cleanup_expired as _cleanup_expired_semantic_cache
from .core.ingest_run_store import IngestRunStore
from .core.ingest_worker import IngestWorker
from .db.schema_contract import validate_schema_contract
from .internal_auth import require_ai_internal_token

# 전역 커넥션 풀 (앱 시작 시 한 번만 생성). 전 계층이 async psycopg3로 통일됨.
_connection_pool: Optional[AsyncConnectionPool] = None
_ingest_connection_pool: Optional[AsyncConnectionPool] = None
DB_POOL_MIN_SIZE = 1
DB_POOL_MAX_SIZE = 30
INGEST_DB_POOL_MIN_SIZE = 1
INGEST_DB_POOL_MAX_SIZE = 2
INGEST_ORCHESTRATION_MIGRATION_SQLS = tuple(
    (
        Path(__file__).resolve().parent / "db" / "migrations" / filename
    ).read_text(encoding="utf-8")
    for filename in (
        "003_ai_ingest_orchestration.sql",
        "004_ai_ingest_checkpoints.sql",
    )
)

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


def _create_async_connection_pool(
    *,
    min_size: int,
    max_size: int,
) -> AsyncConnectionPool:
    settings = get_settings()
    return AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=min_size,
        max_size=max_size,
        check=AsyncConnectionPool.check_connection,
        open=False,
        kwargs={
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
            "autocommit": True,
            "target_session_attrs": "read-write",
        },
    )


def get_connection_pool() -> AsyncConnectionPool:
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = _create_async_connection_pool(
            min_size=DB_POOL_MIN_SIZE,
            max_size=DB_POOL_MAX_SIZE,
        )
        logger.info(
            "[DB] General connection pool created (open pending) pool_stats=%s",
            _format_connection_pool_stats(_connection_pool),
        )
    return _connection_pool


def get_ingest_connection_pool() -> AsyncConnectionPool:
    global _ingest_connection_pool
    if _ingest_connection_pool is None:
        _ingest_connection_pool = _create_async_connection_pool(
            min_size=INGEST_DB_POOL_MIN_SIZE,
            max_size=INGEST_DB_POOL_MAX_SIZE,
        )
        logger.info(
            "[DB] Ingest coordination pool created (open pending) pool_stats=%s",
            _format_connection_pool_stats(
                _ingest_connection_pool,
                min_size=INGEST_DB_POOL_MIN_SIZE,
                max_size=INGEST_DB_POOL_MAX_SIZE,
            ),
        )
    return _ingest_connection_pool


def _snapshot_connection_pool_stats(
    pool_instance: Optional[Any] = None,
    *,
    min_size: int = DB_POOL_MIN_SIZE,
    max_size: int = DB_POOL_MAX_SIZE,
) -> dict[str, Any]:
    pool = pool_instance or _connection_pool
    snapshot: dict[str, Any] = {
        "min_size": min_size,
        "max_size": max_size,
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


def _format_connection_pool_stats(
    pool_instance: Optional[Any] = None,
    *,
    min_size: int = DB_POOL_MIN_SIZE,
    max_size: int = DB_POOL_MAX_SIZE,
) -> str:
    return json.dumps(
        _snapshot_connection_pool_stats(
            pool_instance,
            min_size=min_size,
            max_size=max_size,
        ),
        ensure_ascii=False,
        sort_keys=True,
    )


async def close_connection_pool() -> None:
    """앱 종료 시 커넥션 풀을 닫습니다."""
    global _connection_pool
    pool = _connection_pool
    _connection_pool = None
    if pool is not None:
        await pool.close()


async def close_ingest_connection_pool() -> None:
    global _ingest_connection_pool
    pool = _ingest_connection_pool
    _ingest_connection_pool = None
    if pool is not None:
        await pool.close()


async def _chat_cache_cleanup_loop(interval_seconds: int = 3600) -> None:
    """1시간마다 만료된 chat_response_cache 항목을 삭제하는 백그라운드 루프.

    첫 실행은 interval_seconds 후 (앱 시작 직후 DB 부하 방지).
    일시적 DB 오류 발생 시 경고 로그만 남기고 루프를 계속 유지합니다.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            pool = get_connection_pool()
            async with pool.connection() as conn:
                deleted = await _cleanup_expired_cache(conn)
                semantic_deleted = await _cleanup_expired_semantic_cache(conn)
            if deleted:
                logger.info("[ChatCache] Cleanup: %d expired entries deleted", deleted)
            if semantic_deleted:
                logger.info(
                    "[ChatSemanticCache] Cleanup: %d expired entries deleted",
                    semantic_deleted,
                )
        except Exception as exc:
            logger.warning("[ChatCache] Cleanup loop error: %s", exc)


async def _db_pool_metrics_loop(interval_seconds: int = 30) -> None:
    """주기적으로 DB 풀 stats를 Prometheus gauge에 publish.

    psycopg_pool stats 키 (예시): pool_size, pool_available, requests_waiting,
    pool_min, pool_max. 모든 키를 일일이 매핑하지 않고 주요 4개만 노출.
    """
    try:
        from app.observability.metrics import AI_DB_POOL_SIZE
    except ImportError:
        return
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            pool = get_connection_pool()
            get_stats = getattr(pool, "get_stats", None)
            if not callable(get_stats):
                continue
            stats = get_stats() or {}
            mapping = {
                "max": stats.get("pool_max") or DB_POOL_MAX_SIZE,
                "min": stats.get("pool_min") or DB_POOL_MIN_SIZE,
                "available": stats.get("pool_available", 0),
                "requests_waiting": stats.get("requests_waiting", 0),
            }
            for state, value in mapping.items():
                try:
                    AI_DB_POOL_SIZE.labels(state=state).set(float(value or 0))
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("[DBPoolMetrics] publish loop error: %s", exc)


async def _coach_failed_cache_recovery_loop() -> None:
    """FAILED_LOCKED Coach 캐시를 주기적으로 풀어주는 백그라운드 루프.

    재시도 가능(retryable) 오류로 최대 시도 횟수에 도달해 잠긴 row만 대상으로 하며,
    쿨다운(updated_at 경과)과 사이클당 처리 상한으로 무한 재생성/LLM 비용 폭주를
    방지한다. non-retryable 오류 row와 살아있는 PENDING/COMPLETED는 건드리지 않는다.
    삭제된 row는 다음 분석 요청에서 자연 재생성된다. 기본 비활성(env로 opt-in).
    """
    settings = get_settings()
    interval_seconds = max(60, int(settings.coach_failed_recovery_interval_seconds))
    cooldown_seconds = max(0, int(settings.coach_failed_recovery_cooldown_seconds))
    max_rows = max(1, int(settings.coach_failed_recovery_max_rows_per_cycle))

    # coach 모듈은 deps를 import하므로 순환 방지를 위해 지연 import한다.
    from .routers.coach import (
        RETRYABLE_CACHE_ERROR_CODES,
        COACH_CACHE_MAX_RETRYABLE_ATTEMPTS,
    )

    retryable_codes = list(RETRYABLE_CACHE_ERROR_CODES)
    logger.info(
        "[CoachRecovery] FAILED_LOCKED recovery loop started "
        "(interval=%ss, cooldown=%ss, max_rows=%s)",
        interval_seconds,
        cooldown_seconds,
        max_rows,
    )
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            pool = get_connection_pool()
            async with pool.connection() as conn:
                rows = await (
                    await conn.execute(
                        """
                    DELETE FROM coach_analysis_cache
                    WHERE cache_key IN (
                        SELECT cache_key FROM coach_analysis_cache
                        WHERE status = 'FAILED'
                          AND error_code = ANY(%s)
                          AND attempt_count >= %s
                          AND updated_at < now() - make_interval(secs => %s)
                        ORDER BY updated_at ASC
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING cache_key
                    """,
                        (
                            retryable_codes,
                            COACH_CACHE_MAX_RETRYABLE_ATTEMPTS,
                            cooldown_seconds,
                            max_rows,
                        ),
                    )
                ).fetchall()
                await conn.commit()
            if rows:
                logger.info(
                    "[CoachRecovery] released %d FAILED_LOCKED row(s) for regeneration",
                    len(rows),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CoachRecovery] recovery loop error: %s", exc)


async def _ensure_startup_schema(pool, settings) -> None:
    """Keep the compatibility startup DDL path for local/dev environments."""

    # [Coach Caching] 캐시 테이블 자동 생성 (편의성)
    try:
        async with pool.connection() as conn:
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS coach_analysis_cache (
                cache_key varchar(64) primary key,  -- SHA256 Hash of (team_id, year, focus, question)
                team_id varchar(10) not null,
                year int not null,
                prompt_version varchar(32) not null, -- e.g. "v2"
                model_name varchar(50) not null,     -- e.g. "openrouter/free"
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to ensure coach_analysis_cache table: %s", exc)

    # [Chat Caching] chat_response_cache 테이블 자동 생성
    try:
        async with pool.connection() as conn:
            await conn.execute(CHAT_CACHE_DDL)
        logger.info("[Lifespan] chat_response_cache table ensured")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Lifespan] chat_response_cache DDL failed: %s", exc)

    # Shadow observations do not depend on pgvector and must remain available
    # even when semantic serving is disabled on a local or degraded database.
    try:
        async with pool.connection() as conn:
            await conn.execute(CHAT_SEMANTIC_SHADOW_OBSERVATION_DDL)
        logger.info("[Lifespan] chat semantic shadow observation table ensured")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[Lifespan] chat semantic shadow observation DDL failed: %s", exc
        )

    # [Chat Semantic Caching] chat_semantic_response_cache 테이블 자동 생성
    try:
        async with pool.connection() as conn:
            await conn.execute(CHAT_SEMANTIC_CACHE_DDL)
        logger.info("[Lifespan] chat_semantic_response_cache table ensured")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Lifespan] chat_semantic_response_cache DDL failed: %s", exc)
    if bool(getattr(settings, "chat_semantic_cache_vector_index_enabled", False)):
        try:
            async with pool.connection() as conn:
                await conn.execute(CHAT_SEMANTIC_CACHE_VECTOR_INDEX_DDL)
            logger.info("[Lifespan] chat_semantic_response_cache HNSW index ensured")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[Lifespan] chat_semantic_response_cache HNSW index DDL failed: %s",
                exc,
            )

    await _ensure_ingest_orchestration_schema(pool)


async def _ensure_ingest_orchestration_schema(pool) -> None:
    """Provision the durable queue and checkpoints in compatibility mode."""

    async with pool.connection() as conn:
        for migration_sql in INGEST_ORCHESTRATION_MIGRATION_SQLS:
            await conn.execute(migration_sql)
    logger.info("[Lifespan] AI ingest orchestration tables ensured")


async def _prepare_schema(pool, settings) -> None:
    """Prepare or validate the AI schema according to the runtime mode."""

    if settings.ai_db_schema_mode == "managed":
        try:
            async with pool.connection() as conn:
                await validate_schema_contract(
                    conn,
                    require_vector_index=settings.chat_semantic_cache_vector_index_enabled,
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[Lifespan] managed AI DB schema contract validation failed"
            )
            raise
        logger.info("[Lifespan] managed AI DB schema contract validated")
        return

    await _ensure_startup_schema(pool, settings)


async def _prepare_required_database_pools(
    settings: Any,
) -> tuple[AsyncConnectionPool, AsyncConnectionPool]:
    general_pool = get_connection_pool()
    ingest_pool = get_ingest_connection_pool()
    try:
        await general_pool.open(wait=True, timeout=10.0)
        await ingest_pool.open(wait=True, timeout=10.0)
        await _prepare_schema(general_pool, settings)
    except BaseException as exc:  # noqa: BLE001
        logger.error(
            "[Lifespan] required DB pool preparation failed "
            "error_type=%s general_pool_stats=%s ingest_pool_stats=%s",
            type(exc).__name__,
            _format_connection_pool_stats(general_pool),
            _format_connection_pool_stats(
                ingest_pool,
                min_size=INGEST_DB_POOL_MIN_SIZE,
                max_size=INGEST_DB_POOL_MAX_SIZE,
            ),
        )
        for resource_name, closer in (
            ("ingest_pool", close_ingest_connection_pool),
            ("general_pool", close_connection_pool),
        ):
            try:
                await closer()
            except BaseException as cleanup_exc:  # noqa: BLE001
                logger.error(
                    "[Lifespan] startup cleanup failed "
                    "resource=%s error_type=%s",
                    resource_name,
                    type(cleanup_exc).__name__,
                )
        raise
    logger.info("[Lifespan] required database pools opened")
    return general_pool, ingest_pool


def get_ingest_run_store() -> IngestRunStore:
    """Build an ingestion store over its dedicated coordination pool."""

    settings = get_settings()
    return IngestRunStore(
        get_ingest_connection_pool(),
        lease_seconds=settings.ingest_worker_lease_seconds,
        max_recovery_attempts=settings.ingest_worker_max_recovery_attempts,
    )


@asynccontextmanager
async def lifespan(app):
    """앱 시작/종료 시 실행되는 lifespan 이벤트"""
    del app
    background_tasks: list[asyncio.Task[Any]] = []
    ingest_worker_stop_event: Optional[asyncio.Event] = None
    embedding_backend: Any = None
    database_resources_started = False
    runtime_initialized = False
    primary_exception = False
    try:
        settings = get_settings()
        load_clf()
        database_resources_started = True
        await _prepare_required_database_pools(settings)
        _initialize_shared_baseball_agent_runtime()
        runtime_initialized = True

        cleanup_task = asyncio.create_task(_chat_cache_cleanup_loop())
        background_tasks.append(cleanup_task)
        logger.info(
            "[Lifespan] chat_response_cache cleanup task started (interval=1h)"
        )

        db_pool_metrics_task = asyncio.create_task(_db_pool_metrics_loop())
        background_tasks.append(db_pool_metrics_task)
        logger.info("[Lifespan] db pool metrics publisher started (interval=30s)")

        if settings.coach_failed_recovery_enabled:
            coach_recovery_task = asyncio.create_task(
                _coach_failed_cache_recovery_loop()
            )
            background_tasks.append(coach_recovery_task)
            logger.info("[Lifespan] coach FAILED_LOCKED recovery loop started")

        if settings.ingest_worker_enabled:
            ingest_store = get_ingest_run_store()
            ingest_worker_stop_event = asyncio.Event()
            ingest_worker = IngestWorker(store=ingest_store, settings=settings)
            recovered, failed = await ingest_worker.recover_expired_once()
            ingest_worker_task = asyncio.create_task(
                ingest_worker.run_forever(ingest_worker_stop_event)
            )
            background_tasks.append(ingest_worker_task)
            ingest_recovery_task = asyncio.create_task(
                ingest_worker.run_recovery_forever(ingest_worker_stop_event)
            )
            background_tasks.append(ingest_recovery_task)
            logger.info(
                "[Lifespan] ingest worker started recovered=%d failed=%d",
                recovered,
                failed,
            )

        try:
            from .core.embedding_cache import get_backend as _get_embed_backend

            embedding_backend = await _get_embed_backend()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[Lifespan] embedding cache backend init failed error_type=%s",
                type(exc).__name__,
            )

        yield
    except BaseException:  # noqa: BLE001
        primary_exception = True
        raise
    finally:
        cleanup_errors: list[tuple[str, BaseException]] = []
        if ingest_worker_stop_event is not None:
            ingest_worker_stop_event.set()
        for task in background_tasks:
            task.cancel()
        for task in background_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except BaseException as exc:  # noqa: BLE001
                cleanup_errors.append(("background_task", exc))

        async def attempt_async_cleanup(resource_name: str, closer: Any) -> None:
            try:
                await closer()
            except BaseException as exc:  # noqa: BLE001
                cleanup_errors.append((resource_name, exc))

        if database_resources_started:
            await attempt_async_cleanup("http_clients", close_shared_httpx_clients)

            if embedding_backend is not None:
                try:
                    from .core.embedding_cache import (
                        RedisEmbeddingBackend as _RedisBackend,
                    )

                    if isinstance(embedding_backend, _RedisBackend):
                        client = getattr(embedding_backend, "_client", None)
                        if client is not None and hasattr(client, "aclose"):
                            await client.aclose()
                except BaseException as exc:  # noqa: BLE001
                    cleanup_errors.append(("embedding_cache", exc))

            if runtime_initialized:
                try:
                    reset_shared_baseball_agent_runtime()
                except BaseException as exc:  # noqa: BLE001
                    cleanup_errors.append(("baseball_agent_runtime", exc))

            await attempt_async_cleanup(
                "ingest_pool",
                close_ingest_connection_pool,
            )
            await attempt_async_cleanup("general_pool", close_connection_pool)

        for resource_name, exc in cleanup_errors:
            logger.error(
                "[Lifespan] cleanup failed resource=%s error_type=%s",
                resource_name,
                type(exc).__name__,
            )
        if cleanup_errors and not primary_exception:
            raise cleanup_errors[0][1]


async def get_db_connection() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """커넥션 풀에서 비동기 커넥션을 가져와서 사용 후 반환합니다."""
    pool_instance = get_connection_pool()

    # AsyncConnectionPool은 async context manager를 지원하여 안전하게 반환함
    try:
        async with pool_instance.connection() as conn:
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


def get_rag_pipeline() -> RAGPipeline:
    """RAGPipeline에 ConnectionPool을 주입하여 멀티쿼리가 진정 병렬로 실행되도록 한다.

    각 DB 호출은 풀에서 짧게 커넥션을 빌리므로, 멀티 variation 검색이
    더 이상 단일 커넥션으로 직렬화되지 않는다.
    """
    settings = get_settings()
    pool_instance = get_connection_pool()
    return RAGPipeline(
        settings=settings,
        pool=pool_instance,
        agent_runtime=get_shared_baseball_agent_runtime(),
        context_formatter=_get_shared_context_formatter(),
        wpa_calculator=_get_shared_wpa_calculator(),
    )


async def get_agent(
    conn: psycopg.AsyncConnection = Depends(get_db_connection),
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

    async def coach_openrouter_generator(
        messages,
        max_tokens: int,
        empty_chunk_retry_limit: Optional[int] = None,
        request_timeout_seconds: Optional[float] = None,
    ):
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

        effective_retry_limit = (
            COACH_OPENROUTER_RETRY_LIMIT
            if empty_chunk_retry_limit is None
            else max(0, int(empty_chunk_retry_limit))
        )
        effective_request_timeout_seconds = (
            120.0
            if request_timeout_seconds is None
            else max(5.0, float(request_timeout_seconds))
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

            for retry_index in range(effective_retry_limit + 1):
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.2,
                    "top_p": 0.6,
                    "max_tokens": effective_max_tokens,
                }

                try:
                    attempt_started = perf_counter()
                    logger.info(
                        "[Coach LLM] Attempt start model=%s attempt=%d/%d timeout=%.1fs max_tokens=%d",
                        model,
                        retry_index + 1,
                        effective_retry_limit + 1,
                        effective_request_timeout_seconds,
                        effective_max_tokens,
                    )
                    chunk_count = 0
                    empty_choice_count = 0
                    malformed_chunk_count = 0
                    client = get_shared_httpx_client(
                        (
                            "openrouter"
                            if request_timeout_seconds is None
                            else f"openrouter:coach:{int(effective_request_timeout_seconds)}"
                        ),
                        timeout=effective_request_timeout_seconds,
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
                        if retry_index < effective_retry_limit:
                            logger.warning(
                                "[Coach LLM Retry] model=%s retry_index=%d/%d reason=%s",
                                model,
                                retry_index + 1,
                                effective_retry_limit + 1,
                                empty_chunk_reason,
                            )
                            await asyncio.sleep(
                                COACH_OPENROUTER_RETRY_BACKOFF_SECONDS
                                * (2**retry_index)
                            )
                            continue
                        logger.warning(
                            "[Coach LLM] Empty response model=%s elapsed_sec=%.2f reason=%s empty_choices=%d malformed=%d",
                            model,
                            perf_counter() - attempt_started,
                            empty_chunk_reason,
                            empty_choice_count,
                            malformed_chunk_count,
                        )
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
                        "[Coach LLM] Success model=%s elapsed_sec=%.2f chunk_count=%d empty_choices=%d malformed_chunks=%d",
                        model,
                        perf_counter() - attempt_started,
                        chunk_count,
                        empty_choice_count,
                        malformed_chunk_count,
                    )
                    return
                except Exception as e:
                    retryable = is_retryable_coach_openrouter_error(e)
                    logger.error(
                        "[Coach LLM] OpenRouter model %s failed attempt=%d/%d retryable=%s elapsed_sec=%.2f: %s",
                        model,
                        retry_index + 1,
                        effective_retry_limit + 1,
                        retryable,
                        perf_counter() - attempt_started,
                        e,
                    )
                    last_exception = e
                    if retryable and retry_index < effective_retry_limit:
                        await asyncio.sleep(
                            COACH_OPENROUTER_RETRY_BACKOFF_SECONDS * (2**retry_index)
                        )
                        continue
                    break

        raise last_exception or RuntimeError("All OpenRouter models failed")

    async def coach_llm(
        messages,
        max_tokens=None,
        empty_chunk_retry_limit: Optional[int] = None,
        request_timeout_seconds: Optional[float] = None,
    ):
        """Coach LLM entrypoint (OpenRouter only).

        Note: Coach feature only supports OpenRouter.
        COACH_OPENROUTER_MODEL/COACH_OPENROUTER_FALLBACK_MODELS settings
        control which models are used.
        """
        effective_max_tokens = max_tokens or settings.coach_max_output_tokens

        try:
            async for chunk in coach_openrouter_generator(
                messages,
                effective_max_tokens,
                empty_chunk_retry_limit=empty_chunk_retry_limit,
                request_timeout_seconds=request_timeout_seconds,
            ):
                yield chunk
        except Exception as e:
            logger.error("[Coach LLM] OpenRouter failed: %s", e)
            raise

    return coach_llm


def get_intent_router():
    return predict_intent
