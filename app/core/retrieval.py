"""pgvector 기반의 유사도 검색 기능을 제공하는 모듈입니다.

이 모듈은 PostgreSQL 데이터베이스와 pgvector 확장을 사용하여
벡터 임베딩 간의 코사인 유사도를 계산하고, 관련성 높은 문서를 검색하는 기능을 구현합니다.

환경 변수 USE_FIRESTORE_SEARCH 설정은 과거 호환 전용이며 현재는 PostgreSQL pgvector 검색만 지원합니다.
"""

import logging
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.errors import QueryCanceled, UndefinedTable
from psycopg import OperationalError as PsycopgOperationalError
from psycopg import InterfaceError as PsycopgInterfaceError
from ..config import Settings, get_settings
from .exceptions import DBRetrievalError
from ..observability.metrics import AI_RETRIEVAL_FALLBACK_LEVEL_TOTAL

logger = logging.getLogger(__name__)

_INTERNAL_FILTER_INCLUDE_INNING_SCORES = "_include_game_inning_scores"
_INTERNAL_FILTER_EXCLUDE_SOURCE_TABLES = "_exclude_source_tables"
_SUPPRESSED_SOURCE_TABLES = ("game_inning_scores",)
_PGVECTOR_SEARCH_PATH = "public, extensions, security"

# rag_chunks 테이블 존재 여부 캐시 (프로세스 수명 동안 유효)
# 테이블은 배포 후 변경되지 않으므로 프로세스 재시작 시 자동 무효화됨
_rag_chunks_table_exists: Optional[bool] = None

# 감지된 벡터 인덱스 타입 캐시 (프로세스 수명 동안 유효, "hnsw" 또는 "ivfflat")
# AI_VECTOR_INDEX=auto 모드에서 최초 쿼리 시 pg_indexes를 조회한 후 캐싱한다.
# 인덱스 변경 시 서비스 재시작 필요.
_detected_vector_index: Optional[str] = None


def _vector_literal(vector: Sequence[float]) -> str:
    """Python 리스트 형태의 벡터를 pgvector가 인식하는 문자열 형태로 변환합니다.

    예: [0.1, 0.2, 0.3] -> '[0.10000000,0.20000000,0.30000000]'
    """
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"


def _json_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


async def _ensure_pgvector_search_path(conn: psycopg.AsyncConnection) -> None:
    """pgvector 타입과 연산자를 찾을 수 있도록 search_path를 보정합니다."""
    async with conn.cursor() as cursor:
        await cursor.execute(f"SET search_path TO {_PGVECTOR_SEARCH_PATH}")


def _resolve_settings(settings: Optional[Settings]) -> Settings:
    if settings is not None:
        return settings
    return get_settings()


async def _detect_active_index(conn: psycopg.AsyncConnection) -> str:
    """pg_indexes에서 rag_chunks embedding 인덱스 타입을 감지합니다.

    결과를 모듈 레벨 변수에 캐싱하여 반복 조회를 방지합니다.
    HNSW 인덱스(idx_rag_chunks_embedding_halfvec_hnsw 또는 vector HNSW)가 있으면 "hnsw"를,
    없으면 "ivfflat"을 반환합니다.
    """
    global _detected_vector_index
    if _detected_vector_index is not None:
        return _detected_vector_index
    try:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT indexdef
                FROM pg_indexes
                WHERE tablename = 'rag_chunks'
                  AND indexname LIKE '%embedding%'
                """)
            rows = await cur.fetchall()
            for row in rows:
                idx_def = (row[0] or "").lower()
                if "hnsw" in idx_def:
                    _detected_vector_index = "hnsw"
                    logger.info(
                        "[VectorIndex] auto-detect: HNSW index found — "
                        "using hnsw.ef_search session GUC."
                    )
                    return "hnsw"
        _detected_vector_index = "ivfflat"
        logger.info(
            "[VectorIndex] auto-detect: no HNSW index found — "
            "falling back to ivfflat.probes session GUC."
        )
    except Exception as exc:
        logger.debug(
            "[VectorIndex] auto-detect failed (%s); defaulting to ivfflat.", exc
        )
        _detected_vector_index = "ivfflat"
    return _detected_vector_index  # type: ignore[return-value]


def _embedding_distance_sql(
    settings: Settings,
    *,
    table_alias: Optional[str] = None,
) -> str:
    column = f"{table_alias}.embedding" if table_alias else "embedding"
    quantization = (
        str(getattr(settings, "ai_vector_quantization", "none") or "none")
        .lower()
        .strip()
    )
    if quantization == "halfvec":
        embed_dim = max(1, int(getattr(settings, "embed_dim", 256) or 256))
        return f"{column}::halfvec({embed_dim}) <=> %s::halfvec({embed_dim})"
    return f"{column} <=> %s::vector"


async def _ensure_pgvector_session(
    conn: psycopg.AsyncConnection,
    settings: Optional[Settings] = None,
) -> None:
    """pgvector 검색 세션 설정을 보정합니다.

    AI_VECTOR_INDEX 환경변수 값에 따라 인덱스별 GUC를 설정합니다:
      - "hnsw"    → hnsw.ef_search만 설정
      - "ivfflat" → ivfflat.probes만 설정
      - "auto"    → pg_indexes 감지 후 위 둘 중 하나 적용 (캐싱됨)
    """
    active_settings = _resolve_settings(settings)
    mode = (active_settings.ai_vector_index or "auto").lower().strip()
    if mode == "auto":
        mode = await _detect_active_index(conn)

    async with conn.cursor() as cursor:
        await cursor.execute(f"SET search_path TO {_PGVECTOR_SEARCH_PATH}")
        if mode == "hnsw":
            ef_search = max(1, int(active_settings.retrieval_hnsw_ef_search))
            await cursor.execute(f"SET hnsw.ef_search = {ef_search}")
        else:
            probes = max(1, int(active_settings.retrieval_ivfflat_probes))
            await cursor.execute(f"SET ivfflat.probes = {probes}")


async def _rag_chunks_exists(
    conn: psycopg.AsyncConnection,
    settings: Optional[Settings] = None,
) -> bool:
    """`rag_chunks` 테이블이 존재하는지 확인합니다.

    결과를 모듈 레벨 변수에 캐싱하여 information_schema 반복 조회를 방지합니다.
    존재가 확인된 경우에만 캐싱하며, False는 캐싱하지 않아 초기화 시나리오(테이블 생성 전
    요청이 먼저 들어오는 경우)에도 올바르게 동작합니다.
    """
    global _rag_chunks_table_exists
    if _rag_chunks_table_exists is True:
        return True
    try:
        await _ensure_pgvector_session(conn, settings)
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT EXISTS(
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name = 'rag_chunks'
                );
                """)
            row = await cursor.fetchone()
            result = bool(row[0]) if row else False
            if result:
                _rag_chunks_table_exists = True
            return result
    except Exception:
        return False


async def similarity_search(
    conn: psycopg.AsyncConnection,
    embedding: Sequence[float],
    *,
    limit: int,
    filters: Optional[Dict[str, Any]] = None,
    keyword: Optional[str] = None,
    settings: Optional[Settings] = None,
    intent: str = "",
) -> List[Dict[str, Any]]:
    """주어진 임베딩과 유사한 문서를 데이터베이스에서 검색합니다.

    환경 변수 USE_FIRESTORE_SEARCH=true이면 과거 동작이었으나 현재는 PostgreSQL pgvector만 사용합니다.

    Args:
    conn: PostgreSQL 데이터베이스 연결 객체.
        embedding: 검색의 기준이 될 벡터 임베딩.
        limit: 반환할 최대 문서 수.
        filters: 검색 결과를 필터링할 조건 (예: {'source_table': 'news'}).
        keyword: 텍스트 검색(Full-Text Search)에 사용할 키워드.

    Returns:
        유사도 순으로 정렬된 문서 리스트. 각 문서는 사전(dict) 형태로 반환됩니다.
    """
    if os.getenv("USE_FIRESTORE_SEARCH", "false").lower() == "true":
        raise NotImplementedError(
            "Firestore search has been removed. PostgreSQL pgvector search is supported only."
        )

    active_settings = _resolve_settings(settings)

    if not await _rag_chunks_exists(conn, active_settings):
        logger.warning("[Search] rag_chunks table is not available.")
        return []

    include_game_inning_scores = False
    exclude_source_tables: List[str] = []
    cleaned_filters: Dict[str, Any] = {}
    if filters:
        include_game_inning_scores = bool(
            filters.get(_INTERNAL_FILTER_INCLUDE_INNING_SCORES)
        )
        raw_excluded_tables = filters.get(_INTERNAL_FILTER_EXCLUDE_SOURCE_TABLES)
        if isinstance(raw_excluded_tables, str):
            exclude_source_tables = [raw_excluded_tables]
        elif isinstance(raw_excluded_tables, Sequence):
            exclude_source_tables = [
                str(value) for value in raw_excluded_tables if value
            ]
        cleaned_filters = {
            key: value
            for key, value in filters.items()
            if key
            not in {
                _INTERNAL_FILTER_INCLUDE_INNING_SCORES,
                _INTERNAL_FILTER_EXCLUDE_SOURCE_TABLES,
            }
        }
        if cleaned_filters.get("source_table") == "game_inning_scores":
            include_game_inning_scores = True

    # 기본값: PostgreSQL pgvector 사용
    filter_clauses: List[str] = ["embedding IS NOT NULL"]  # 임베딩이 없는 문서는 제외
    if bool(getattr(active_settings, "rag_retrieval_active_filter_enabled", True)):
        filter_clauses.extend(
            [
                "COALESCE(is_active, true) = true",
                "(expires_at IS NULL OR expires_at > now())",
                "(valid_from IS NULL OR valid_from <= now())",
                "(valid_to IS NULL OR valid_to > now())",
            ]
        )
    filter_params: List[Any] = []

    suppressed_tables: List[str] = []
    if not include_game_inning_scores:
        suppressed_tables.extend(_SUPPRESSED_SOURCE_TABLES)
    for source_table in exclude_source_tables:
        if source_table not in suppressed_tables:
            suppressed_tables.append(source_table)

    for source_table in suppressed_tables:
        filter_clauses.append("source_table <> %s")
        filter_params.append(source_table)

    # 제공된 필터 조건을 SQL WHERE 절로 변환합니다.
    if cleaned_filters:
        for key, value in cleaned_filters.items():
            if value is None:
                continue
            # JSON field filtering support (e.g., "meta.league")
            if "." in key:
                json_field, json_key = key.rsplit(".", 1)
                if json_field == "meta":
                    json_field = "metadata"
                filter_clauses.append(f"{json_field}->>%s = %s")
                filter_params.extend([json_key, value])
            else:
                filter_clauses.append(f"{key} = %s")
                filter_params.append(value)

    where_clause = " AND ".join(filter_clauses)

    # 벡터를 SQL 쿼리에 직접 삽입할 수 있는 문자열 형태로 변환합니다.
    vector_str = _vector_literal(embedding)
    embedding_distance = _embedding_distance_sql(active_settings)
    row_embedding_distance = _embedding_distance_sql(active_settings, table_alias="r")

    # 최종 SQL 쿼리를 구성합니다.
    # <=> 연산자: pgvector에서 코사인 거리(1 - 코사인 유사도)를 계산합니다.
    # Reciprocal Rank Fusion (RRF) 스타일의 가중합을 위해 rank를 계산합니다.
    rrf_k = _resolve_rrf_k(intent)

    if keyword:
        sql = f"""
        WITH keyword_query AS (
            SELECT plainto_tsquery('simple', %s) AS query
        ),
        vector_search AS (
            SELECT
                id,
                ROW_NUMBER() OVER (ORDER BY {embedding_distance} ASC) AS vector_rank
            FROM rag_chunks
            WHERE {where_clause}
            LIMIT %s * 2
        ),
        keyword_search AS (
            SELECT
                r.id,
                ts_rank(r.content_tsv, kq.query) AS ts_rank_val,
                ROW_NUMBER() OVER (ORDER BY ts_rank(r.content_tsv, kq.query) DESC) AS keyword_rank
            FROM rag_chunks r
            CROSS JOIN keyword_query kq
            WHERE {where_clause} AND r.content_tsv @@ kq.query
            LIMIT %s * 2
        ),
        candidates AS (
            SELECT id FROM vector_search
            UNION
            SELECT id FROM keyword_search
        )
        SELECT
            c.id,
            r.title,
            r.content,
            r.source_table,
            r.source_row_id,
            COALESCE(NULLIF(r.metadata, '{{}}'::jsonb), r.meta, '{{}}'::jsonb) AS meta,
            r.metadata,
            r.source_type,
            r.source_uri,
            r.topic_key,
            r.content_hash,
            r.valid_from,
            r.valid_to,
            r.expires_at,
            r.updated_at,
            r.quality_score,
            (1 - ({row_embedding_distance})) AS similarity,
            COALESCE(k.ts_rank_val, 0) AS keyword_rank_val,
            (
                COALESCE(1.0 / ({rrf_k} + v.vector_rank), 0) +
                COALESCE(1.0 / ({rrf_k} + k.keyword_rank), 0)
            ) AS combined_score
        FROM candidates c
        JOIN rag_chunks r ON r.id = c.id
        LEFT JOIN vector_search v ON v.id = c.id
        LEFT JOIN keyword_search k ON k.id = c.id
        ORDER BY combined_score DESC, similarity DESC
        LIMIT %s
        """

        # Parameter order:
        # keyword_query -> vector_search(vector+filters+limit) ->
        # keyword_search(filters+limit) -> final similarity vector+limit.
        final_params = (
            [keyword, vector_str]
            + filter_params
            + [limit]
            + filter_params
            + [limit, vector_str, limit]
        )
    else:
        # 키워드 없는 경우 순수 벡터 검색
        sql = f"""
        SELECT
               id,
               title,
               content,
               source_table,
               source_row_id,
               COALESCE(NULLIF(metadata, '{{}}'::jsonb), meta, '{{}}'::jsonb) AS meta,
               metadata,
               source_type,
               source_uri,
               topic_key,
               content_hash,
               valid_from,
               valid_to,
               expires_at,
               updated_at,
               quality_score,
               (1 - ({embedding_distance})) as similarity
        FROM rag_chunks
        WHERE {where_clause}
        ORDER BY {embedding_distance} ASC
        LIMIT %s
        """
        final_params = [vector_str] + filter_params + [vector_str, limit]

    start_time = time.perf_counter()

    try:
        # _ensure_pgvector_session이 인덱스 종류에 맞는 GUC(hnsw.ef_search 또는
        # ivfflat.probes)를 설정한다. 여기서는 쿼리 타임아웃만 별도로 제한한다.
        await _ensure_pgvector_session(conn, active_settings)
        async with conn.cursor(row_factory=dict_row) as cur:
            statement_timeout_ms = max(
                1, int(active_settings.retrieval_statement_timeout_ms)
            )
            await cur.execute(f"SET LOCAL statement_timeout = {statement_timeout_ms};")
            await cur.execute(sql, final_params)
            rows = await cur.fetchall()
    except UndefinedTable:
        return []
    except QueryCanceled as exc:
        logger.warning(
            "[Search] similarity_search timed out after %dms (hybrid=%s): %s",
            int(active_settings.retrieval_statement_timeout_ms),
            bool(keyword),
            exc,
        )
        raise DBRetrievalError("pgvector query timed out", cause=exc) from exc
    except (PsycopgOperationalError, PsycopgInterfaceError, TimeoutError) as exc:
        logger.error("[Search] DB unreachable during similarity_search: %s", exc)
        raise DBRetrievalError(
            "pgvector query failed — DB unreachable", cause=exc
        ) from exc
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "[Search] Hybrid RRF search took %.2fms (results=%d, hybrid=%s)",
        elapsed_ms,
        len(rows),
        bool(keyword),
    )

    return [dict(row) for row in rows]


# intent별 RRF k 값: 낮을수록 상위 결과에 집중, 높을수록 더 넓은 풀
_RRF_K_BY_INTENT: Dict[str, int] = {
    "player_profile": 30,  # 특정 선수 데이터 → 상위 결과 집중
    "stats_lookup": 40,  # 통계 순위 → 약간 더 넓은 풀
    "comparison": 50,  # 비교 쌍 검색 → 균형
}


def _resolve_rrf_k(intent: str) -> int:
    """intent에 맞는 RRF k 상수를 반환한다. 알 수 없는 intent는 기본값 60."""
    return _RRF_K_BY_INTENT.get(intent, 60)


# 점진적으로 완화되는 필터 키 목록 (제거 순서: 선택성 낮은 것부터)
_FALLBACK_FILTER_KEYS = ("source_table", "team_id", "season_year")


async def similarity_search_with_fallback(
    conn: psycopg.AsyncConnection,
    embedding: Sequence[float],
    *,
    limit: int,
    filters: Optional[Dict[str, Any]] = None,
    keyword: Optional[str] = None,
    settings: Optional[Settings] = None,
    min_results: int = 1,
    intent: str = "",
) -> tuple[List[Dict[str, Any]], str]:
    """
    필터를 점진적으로 완화하며 similarity_search()를 호출한다.

    결과 수가 min_results 미만이면 다음 단계 필터로 재시도:
      Level 1: 모든 필터 적용 (season_year + team_id + source_table)
      Level 2: source_table 제거
      Level 3: team_id 제거
      Level 4: season_year 제거 (내부 필터만 유지)

    Returns:
        (결과 리스트, 사용된 레벨 설명 문자열)
    """
    base_filters = dict(filters) if filters else {}

    # 내부(private) 필터는 모든 레벨에서 유지
    internal_keys = {
        _INTERNAL_FILTER_INCLUDE_INNING_SCORES,
        _INTERNAL_FILTER_EXCLUDE_SOURCE_TABLES,
    }
    internal_filters = {k: v for k, v in base_filters.items() if k in internal_keys}

    # 사용자 필터에서 단계별로 제거할 키 목록
    removable_keys = list(_FALLBACK_FILTER_KEYS)
    current_filters = {k: v for k, v in base_filters.items() if k not in internal_keys}

    for level in range(len(removable_keys) + 1):
        level_name = f"level_{level + 1}"
        merged = {**current_filters, **internal_filters}
        results = await similarity_search(
            conn,
            embedding,
            limit=limit,
            filters=merged if merged else None,
            keyword=keyword,
            settings=settings,
            intent=intent,
        )
        if len(results) >= min_results:
            if level > 0:
                logger.info(
                    "[Search] Fallback %s returned %d results (removed: %s)",
                    level_name,
                    len(results),
                    removable_keys[:level],
                )
            try:
                AI_RETRIEVAL_FALLBACK_LEVEL_TOTAL.labels(level=level_name).inc()
            except Exception:  # noqa: BLE001
                pass
            return results, level_name

        if level < len(removable_keys):
            removed_key = removable_keys[level]
            current_filters.pop(removed_key, None)
            logger.info(
                "[Search] Fallback: 0 results at %s, retrying without '%s'",
                level_name,
                removed_key,
            )

    try:
        AI_RETRIEVAL_FALLBACK_LEVEL_TOTAL.labels(level="exhausted").inc()
    except Exception:  # noqa: BLE001
        pass
    return [], f"level_{len(removable_keys) + 1}_exhausted"


async def record_retrieval_event(
    conn: psycopg.AsyncConnection,
    *,
    user_query: str,
    intent: Optional[str],
    rewritten_queries: Sequence[str],
    metadata_filter: Dict[str, Any],
    retrieved_chunk_ids: Sequence[Any],
    selected_chunk_ids: Sequence[Any],
    scores: Sequence[Dict[str, Any]],
    latency_ms: Optional[int],
    success: bool,
    error_type: Optional[str],
    settings: Optional[Settings] = None,
) -> None:
    """Best-effort retrieval event logging for RAG quality analysis."""
    active_settings = _resolve_settings(settings)
    if not bool(getattr(active_settings, "rag_retrieval_event_logging_enabled", True)):
        return
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"SET search_path TO {_PGVECTOR_SEARCH_PATH}")
            await cur.execute(
                """
                INSERT INTO rag_retrieval_events (
                    user_query,
                    intent,
                    rewritten_queries,
                    metadata_filter,
                    retrieved_chunk_ids,
                    selected_chunk_ids,
                    scores,
                    latency_ms,
                    success,
                    error_type
                ) VALUES (
                    %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
                    %s::jsonb, %s, %s, %s
                )
                """,
                (
                    user_query,
                    intent,
                    _json_payload(list(rewritten_queries)),
                    _json_payload(metadata_filter or {}),
                    _json_payload(list(retrieved_chunk_ids)),
                    _json_payload(list(selected_chunk_ids)),
                    _json_payload(list(scores)),
                    latency_ms,
                    success,
                    error_type,
                ),
            )
    except UndefinedTable:
        logger.debug("[Search] rag_retrieval_events table is not available yet.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Search] Failed to record retrieval event: %s", exc)
