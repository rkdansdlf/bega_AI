"""pgvector 기반의 유사도 검색 기능을 제공하는 모듈입니다.

이 모듈은 PostgreSQL 데이터베이스와 pgvector 확장을 사용하여
벡터 임베딩 간의 코사인 유사도를 계산하고, 관련성 높은 문서를 검색하는 기능을 구현합니다.

환경 변수 USE_FIRESTORE_SEARCH 설정은 과거 호환 전용이며 현재는 PostgreSQL pgvector 검색만 지원합니다.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.errors import UndefinedTable

logger = logging.getLogger(__name__)

_INTERNAL_FILTER_INCLUDE_INNING_SCORES = "_include_game_inning_scores"
_INTERNAL_FILTER_EXCLUDE_SOURCE_TABLES = "_exclude_source_tables"
_SUPPRESSED_SOURCE_TABLES = ("game_inning_scores",)
_PGVECTOR_SEARCH_PATH = "public, extensions, security"
_IVFFLAT_PROBES = 512


def _vector_literal(vector: Sequence[float]) -> str:
    """Python 리스트 형태의 벡터를 pgvector가 인식하는 문자열 형태로 변환합니다.

    예: [0.1, 0.2, 0.3] -> '[0.10000000,0.20000000,0.30000000]'
    """
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"


def _ensure_pgvector_search_path(conn: psycopg.Connection) -> None:
    """pgvector 타입과 연산자를 찾을 수 있도록 search_path를 보정합니다."""
    with conn.cursor() as cursor:
        cursor.execute(f"SET search_path TO {_PGVECTOR_SEARCH_PATH}")


def _ensure_pgvector_session(conn: psycopg.Connection) -> None:
    """pgvector 검색 세션 설정을 보정합니다."""
    with conn.cursor() as cursor:
        cursor.execute(f"SET search_path TO {_PGVECTOR_SEARCH_PATH}")
        cursor.execute(f"SET ivfflat.probes = {_IVFFLAT_PROBES}")


def _rag_chunks_exists(conn: psycopg.Connection) -> bool:
    """`rag_chunks` 테이블이 존재하는지 확인합니다."""
    try:
        _ensure_pgvector_session(conn)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name = 'rag_chunks'
                );
                """)
            row = cursor.fetchone()
            return bool(row[0]) if row else False
    except Exception:
        return False


def similarity_search(
    conn: psycopg.Connection,
    embedding: Sequence[float],
    *,
    limit: int,
    filters: Optional[Dict[str, Any]] = None,
    keyword: Optional[str] = None,
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

    if not _rag_chunks_exists(conn):
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
                filter_clauses.append(f"{json_field}->>%s = %s")
                filter_params.extend([json_key, value])
            else:
                filter_clauses.append(f"{key} = %s")
                filter_params.append(value)

    where_clause = " AND ".join(filter_clauses)

    # 벡터를 SQL 쿼리에 직접 삽입할 수 있는 문자열 형태로 변환합니다.
    vector_str = _vector_literal(embedding)

    # 최종 SQL 쿼리를 구성합니다.
    # <=> 연산자: pgvector에서 코사인 거리(1 - 코사인 유사도)를 계산합니다.
    # Reciprocal Rank Fusion (RRF) 스타일의 가중합을 위해 rank를 계산합니다.
    rrf_k = 60  # RRF standard constant

    if keyword:
        sql = f"""
        WITH keyword_query AS (
            SELECT plainto_tsquery('simple', %s) AS query
        ),
        vector_search AS (
            SELECT
                id,
                ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector ASC) AS vector_rank
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
            r.meta,
            (1 - (r.embedding <=> %s::vector)) AS similarity,
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
               meta,
               (1 - (embedding <=> %s::vector)) as similarity
        FROM rag_chunks
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s
        """
        final_params = [vector_str] + filter_params + [vector_str, limit]

    start_time = time.perf_counter()

    try:
        _ensure_pgvector_session(conn)
        with conn.cursor(row_factory=dict_row) as cur:
            # HNSW 검색 정확도 튜닝 (Recall 향상)
            cur.execute("SET LOCAL hnsw.ef_search = 100;")
            cur.execute(sql, final_params)
            rows = cur.fetchall()
    except UndefinedTable:
        return []
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "[Search] Hybrid RRF search took %.2fms (results=%d, hybrid=%s)",
        elapsed_ms,
        len(rows),
        bool(keyword),
    )

    return [dict(row) for row in rows]
