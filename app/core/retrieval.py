"""pgvector 기반 유사도 검색 기능을 제공하는 모듈."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import RealDictCursor

from ..config import Settings


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"


def similarity_search(
    conn: PgConnection,
    embedding: Sequence[float],
    *,
    limit: int,
    filters: Optional[Dict[str, Any]] = None,
    keyword: Optional[str] = None,
) -> List[Dict[str, Any]]:
    filter_clauses: List[str] = ["embedding IS NOT NULL"]
    params: List[Any] = []

    if filters:
        for key, value in filters.items():
            if value is None:
                continue
            filter_clauses.append(f"{key} = %s")
            params.append(value)

    where_clause = " AND ".join(filter_clauses)
    vector = _vector_literal(embedding)
    ts_part = ""
    if keyword:
        ts_part = ", ts_rank(content_tsv, plainto_tsquery('simple', %s)) as keyword_rank"
        params.append(keyword)

    sql = f"""
    SELECT id,
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
    params = [vector] + params + [limit]

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [dict(row) for row in rows]
