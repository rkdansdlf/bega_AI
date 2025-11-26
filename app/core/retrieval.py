"""pgvector 기반의 유사도 검색 기능을 제공하는 모듈입니다.

이 모듈은 PostgreSQL 데이터베이스와 pgvector 확장을 사용하여
벡터 임베딩 간의 코사인 유사도를 계산하고, 관련성 높은 문서를 검색하는 기능을 구현합니다.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import RealDictCursor

from ..config import Settings


def _vector_literal(vector: Sequence[float]) -> str:
    """Python 리스트 형태의 벡터를 pgvector가 인식하는 문자열 형태로 변환합니다.
    
    예: [0.1, 0.2, 0.3] -> '[0.10000000,0.20000000,0.30000000]'
    """
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"


def similarity_search(
    conn: PgConnection,
    embedding: Sequence[float],
    *,
    limit: int,
    filters: Optional[Dict[str, Any]] = None,
    keyword: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """주어진 임베딩과 유사한 문서를 데이터베이스에서 검색합니다.

    Args:
        conn: PostgreSQL 데이터베이스 연결 객체.
        embedding: 검색의 기준이 될 벡터 임베딩.
        limit: 반환할 최대 문서 수.
        filters: 검색 결과를 필터링할 조건 (예: {'source_table': 'news'}).
        keyword: 텍스트 검색(Full-Text Search)에 사용할 키워드.

    Returns:
        유사도 순으로 정렬된 문서 리스트. 각 문서는 사전(dict) 형태로 반환됩니다.
    """
    filter_clauses: List[str] = ["embedding IS NOT NULL"]  # 임베딩이 없는 문서는 제외
    filter_params: List[Any] = []

    # 제공된 필터 조건을 SQL WHERE 절로 변환합니다.
    if filters:
        for key, value in filters.items():
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
    
    # 키워드 검색이 요청된 경우, 텍스트 검색 순위(ts_rank)를 계산하는 부분을 추가합니다.
    ts_part = ""
    keyword_param: List[str] = []
    if keyword:
        ts_part = ", ts_rank(content_tsv, plainto_tsquery('simple', %s)) as keyword_rank"
        keyword_param.append(keyword)

    # 최종 SQL 쿼리를 구성합니다.
    # <=> 연산자: pgvector에서 코사인 거리(1 - 코사인 유사도)를 계산합니다.
    # (1 - (embedding <=> ...))를 통해 코사인 유사도를 구합니다.
    sql = f"""
    SELECT 
           id,
           title,
           content,
           source_table,
           source_row_id,
           meta,
           (1 - (embedding <=> %s::vector)) as similarity
           {ts_part}
    FROM rag_chunks
    WHERE {where_clause}
    ORDER BY similarity DESC
    LIMIT %s
    """
    # SQL 쿼리에 사용될 파라미터를 최종적으로 조합합니다.
    # 순서: [벡터 문자열, (키워드), ...필터 값, LIMIT 값]
    final_params = [vector_str] + keyword_param + filter_params + [limit]

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, final_params)
        rows = cur.fetchall()
        
    return [dict(row) for row in rows]