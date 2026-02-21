"""
챗봇 응답 DB 캐시 CRUD 모듈.

DB 접근 패턴은 app/routers/coach.py와 동일하게 따릅니다:
- psycopg3 (psycopg 패키지)
- psycopg_pool.ConnectionPool, pool.connection() context manager
- conn.execute(sql, (param, ...)) — %s 플레이스홀더, 튜플 파라미터
- 풀 생성 시 autocommit=True 설정 → 개별 DML에 conn.commit() 불필요
- 비동기 컨텍스트에서 동기 DB 호출은 run_in_threadpool로 래핑

DDL 자동 적용:
    deps.py lifespan()에서 CREATE_TABLE_SQL을 실행하도록 추가해야 합니다.
    (coach_analysis_cache 생성 방식과 동일)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi.concurrency import run_in_threadpool

from .chat_cache_key import get_ttl_seconds

logger = logging.getLogger(__name__)

# ─── DDL ────────────────────────────────────────────────────────────────────────
# deps.py lifespan()에서 pool.connection()으로 실행합니다.
# scripts/create_chat_cache_table.sql과 동일한 스키마입니다.
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_response_cache (
    cache_key      VARCHAR(64)  PRIMARY KEY,
    question_text  TEXT         NOT NULL,
    filters_json   JSONB,
    intent         VARCHAR(50),
    response_text  TEXT         NOT NULL,
    model_name     VARCHAR(100),
    hit_count      INTEGER      NOT NULL DEFAULT 0,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at     TIMESTAMPTZ  NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_cache_expires_at ON chat_response_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_chat_cache_created_at ON chat_response_cache(created_at);
"""


# ─── 동기 헬퍼 (run_in_threadpool로 실행) ──────────────────────────────────────
# psycopg3 API:
#   conn.execute(sql, params_tuple) → cursor
#   cursor.fetchone() → Optional[tuple]
#   cursor.fetchall() → list[tuple]
#   cursor.rowcount  → int (DML 영향 행 수)
# autocommit=True이므로 conn.commit() 생략.


def _get_sync(conn, cache_key: str) -> Optional[Dict[str, Any]]:
    """
    유효한 캐시 항목을 조회합니다.

    expires_at > now() 조건으로 만료 항목을 자동 제외합니다.
    항목이 없거나 만료된 경우 None을 반환합니다.
    """
    row = conn.execute(
        """
        SELECT response_text, intent, model_name, hit_count, expires_at
        FROM   chat_response_cache
        WHERE  cache_key = %s
          AND  expires_at > now()
        """,
        (cache_key,),
    ).fetchone()

    if row is None:
        return None

    response_text, intent, model_name, hit_count, expires_at = row
    return {
        "response_text": response_text,
        "intent": intent,
        "model_name": model_name,
        "hit_count": hit_count,
        "expires_at": expires_at,
    }


def _save_sync(
    conn,
    *,
    cache_key: str,
    question_text: str,
    filters_json: Optional[Dict[str, Any]],
    intent: Optional[str],
    response_text: str,
    model_name: Optional[str],
) -> None:
    """
    캐시 항목을 저장합니다.

    ON CONFLICT (cache_key) DO UPDATE로 동일 키를 원자적으로 갱신합니다.
    만료된 기존 row가 남아 있는 경우에도 즉시 재활성화할 수 있습니다.

    filters_json은 psycopg3 %s 플레이스홀더에 ::jsonb 캐스트를 적용합니다.
    (coach.py의 response_json 저장 방식과 동일)
    """
    ttl_secs = get_ttl_seconds(intent)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_secs)

    # JSONB 직렬화 — None이면 DB에 NULL 저장
    filters_serialized: Optional[str] = (
        json.dumps(filters_json, ensure_ascii=False, sort_keys=True)
        if filters_json
        else None
    )

    conn.execute(
        """
        INSERT INTO chat_response_cache
            (cache_key, question_text, filters_json, intent,
             response_text, model_name, expires_at)
        VALUES
            (%s, %s, %s::jsonb, %s, %s, %s, %s)
        ON CONFLICT (cache_key) DO UPDATE
        SET
            question_text = EXCLUDED.question_text,
            filters_json = EXCLUDED.filters_json,
            intent = EXCLUDED.intent,
            response_text = EXCLUDED.response_text,
            model_name = EXCLUDED.model_name,
            expires_at = EXCLUDED.expires_at,
            hit_count = 0,
            created_at = now()
        """,
        (
            cache_key,
            question_text,
            filters_serialized,
            intent,
            response_text,
            model_name,
            expires_at,
        ),
    )


def _update_hit_count_sync(conn, cache_key: str) -> None:
    """hit_count를 1 증가시킵니다. 존재하지 않는 키는 무시됩니다."""
    conn.execute(
        """
        UPDATE chat_response_cache
        SET    hit_count = hit_count + 1
        WHERE  cache_key = %s
        """,
        (cache_key,),
    )


def _cleanup_sync(conn) -> int:
    """
    만료된 캐시 항목을 삭제합니다.

    Returns:
        삭제된 행 수
    """
    result = conn.execute("DELETE FROM chat_response_cache WHERE expires_at <= now()")
    # psycopg3 cursor.rowcount: 영향받은 행 수 (DML 실행 후 유효)
    deleted: int = getattr(result, "rowcount", 0) or 0
    if deleted:
        logger.info("[ChatCache] Cleaned up %d expired entries", deleted)
    return deleted


def _get_stats_sync(conn) -> List[Dict[str, Any]]:
    """
    유효한 캐시 항목의 intent별 통계를 반환합니다.

    Returns:
        [{"intent": str, "count": int, "avg_hits": float}, ...]
    """
    rows = conn.execute("""
        SELECT intent,
               COUNT(*)                        AS cnt,
               ROUND(AVG(hit_count)::numeric, 2) AS avg_hits
        FROM   chat_response_cache
        WHERE  expires_at > now()
        GROUP  BY intent
        ORDER  BY cnt DESC
        """).fetchall()

    return [
        {
            "intent": row[0],
            "count": row[1],
            "avg_hits": float(row[2] or 0),
        }
        for row in rows
    ]


def _delete_by_intent_sync(conn, intent: str) -> int:
    """특정 intent의 모든 캐시 항목을 삭제합니다 (만료 여부 무관)."""
    result = conn.execute(
        "DELETE FROM chat_response_cache WHERE intent = %s",
        (intent,),
    )
    return getattr(result, "rowcount", 0) or 0


def _delete_by_key_sync(conn, cache_key: str) -> int:
    """특정 cache_key 항목을 삭제합니다."""
    result = conn.execute(
        "DELETE FROM chat_response_cache WHERE cache_key = %s",
        (cache_key,),
    )
    return getattr(result, "rowcount", 0) or 0


# ─── 비동기 퍼블릭 API ──────────────────────────────────────────────────────────
# 비동기 라우터(chat_stream.py)에서 직접 import하여 사용합니다.
# 모든 함수는 conn을 첫 번째 인자로 받습니다.
# conn은 호출자 측에서 pool.connection() 컨텍스트로 관리합니다.
#
# 사용 예시 (chat_stream.py):
#   pool = get_connection_pool()
#   with pool.connection() as conn:
#       hit = await get_cached_response(conn, cache_key)
#       if hit:
#           await update_hit_count(conn, cache_key)
#           return hit["response_text"]
#   ...LLM 호출...
#   with pool.connection() as conn:
#       await save_to_cache(conn, cache_key=..., ...)


async def get_cached_response(conn, cache_key: str) -> Optional[Dict[str, Any]]:
    """
    유효한 캐시 항목을 비동기로 조회합니다.

    Returns:
        캐시 히트 시 {"response_text", "intent", "model_name",
                       "hit_count", "expires_at"} 딕셔너리,
        미스 또는 만료 시 None.
    """
    return await run_in_threadpool(_get_sync, conn, cache_key)


async def save_to_cache(
    conn,
    *,
    cache_key: str,
    question_text: str,
    filters_json: Optional[Dict[str, Any]],
    intent: Optional[str],
    response_text: str,
    model_name: Optional[str],
) -> None:
    """
    LLM 응답을 캐시에 저장합니다.

    저장 실패 시 경고 로그만 남기고 예외를 상위로 전파하지 않습니다.
    캐시 저장 실패가 챗봇 응답 흐름을 중단시키지 않도록 하기 위함입니다.
    """
    try:
        await run_in_threadpool(
            _save_sync,
            conn,
            cache_key=cache_key,
            question_text=question_text,
            filters_json=filters_json,
            intent=intent,
            response_text=response_text,
            model_name=model_name,
        )
    except Exception as exc:
        # 키 앞 8자만 로깅 (전체 해시 노출 방지)
        logger.warning("[ChatCache] save failed for key %.8s: %s", cache_key, exc)


async def update_hit_count(conn, cache_key: str) -> None:
    """
    캐시 히트 시 hit_count를 비동기로 증가시킵니다.

    실패해도 응답 흐름에 영향을 주지 않도록 예외를 삼킵니다.
    """
    try:
        await run_in_threadpool(_update_hit_count_sync, conn, cache_key)
    except Exception as exc:
        logger.warning("[ChatCache] hit_count update failed: %s", exc)


async def cleanup_expired(conn) -> int:
    """
    만료된 캐시 항목을 비동기로 삭제합니다.

    스케줄러 또는 관리 엔드포인트에서 주기적으로 호출하세요.

    Returns:
        삭제된 행 수
    """
    return await run_in_threadpool(_cleanup_sync, conn)


async def get_stats(conn) -> List[Dict[str, Any]]:
    """
    유효한 캐시의 intent별 통계를 반환합니다.

    관리 엔드포인트(/admin/cache/stats)에서 사용합니다.
    """
    return await run_in_threadpool(_get_stats_sync, conn)


async def delete_by_intent(conn, intent: str) -> int:
    """
    특정 intent의 캐시를 전체 삭제합니다.

    예: intent="stats_lookup" 캐시를 강제 무효화할 때 사용.

    Returns:
        삭제된 행 수
    """
    return await run_in_threadpool(_delete_by_intent_sync, conn, intent)


async def delete_by_key(conn, cache_key: str) -> int:
    """
    특정 cache_key 항목을 삭제합니다.

    Returns:
        삭제된 행 수 (0이면 존재하지 않았음)
    """
    return await run_in_threadpool(_delete_by_key_sync, conn, cache_key)
