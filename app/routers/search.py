"""검색 성능 디버깅 및 분석을 위한 고급 엔드포인트 라우터"""

from typing import Any, Dict, Optional, List
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from psycopg2.extensions import connection as PgConnection

from ..deps import get_db_connection, get_rag_pipeline
from ..core.entity_extractor import enhance_search_strategy

router = APIRouter(prefix="/search", tags=["search"])


class SearchAnalysisResponse(BaseModel):
    """검색 분석 결과를 담는 응답 모델"""
    query: str
    execution_time_ms: float
    entity_analysis: Dict[str, Any]
    search_strategy: Dict[str, Any]
    results: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]


@router.get("/", response_model=SearchAnalysisResponse)
async def debug_search(
    q: str = Query(..., min_length=2, description="분석할 질문 또는 키워드"),
    limit: int = Query(10, ge=1, le=50, description="검색 결과 개수"),
    year: Optional[int] = Query(None, description="시즌 연도 (예: 2024)"),
    team: Optional[str] = Query(None, description="팀명 (예: LG, KIA)"),
    use_multi_query: bool = Query(True, description="다중 쿼리 검색 사용 여부"),
    pipeline=Depends(get_rag_pipeline),
):
    """
    검색 알고리즘을 자세히 분석하고 성능을 측정하는 디버깅 엔드포인트입니다.
    개발자가 검색 품질을 개선하기 위해 사용합니다.
    """
    start_time = time.time()
    
    # 1. 엔티티 추출 및 검색 전략 분석
    search_strategy = enhance_search_strategy(q)
    entity_filter = search_strategy["entity_filter"]
    
    # 사용자 제공 필터 적용
    filters = search_strategy["db_filters"].copy()
    if year:
        filters["season_year"] = year
    if team:
        filters["team_id"] = team.upper()
    
    # 2. 검색 수행
    if use_multi_query:
        docs = await pipeline.retrieve_with_multi_query(
            q, entity_filter, filters=filters
        )
        search_method = "multi_query_retrieval"
    else:
        docs = await pipeline.retrieve(q, filters=filters, limit=limit)
        search_method = "single_query_retrieval"
    
    # 3. 성능 메트릭 계산
    execution_time = (time.time() - start_time) * 1000  # ms
    
    # 결과 품질 분석
    similarity_scores = [doc.get('similarity', 0) for doc in docs if doc.get('similarity')]
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    min_similarity = min(similarity_scores) if similarity_scores else 0
    max_similarity = max(similarity_scores) if similarity_scores else 0
    
    # 메타데이터 커버리지 분석
    unique_teams = set(doc.get('team_id') for doc in docs if doc.get('team_id'))
    unique_years = set(doc.get('season_year') for doc in docs if doc.get('season_year'))
    unique_tables = set(doc.get('source_table') for doc in docs if doc.get('source_table'))
    
    return SearchAnalysisResponse(
        query=q,
        execution_time_ms=round(execution_time, 2),
        entity_analysis={
            "extracted_year": entity_filter.season_year,
            "extracted_team": entity_filter.team_id,
            "extracted_player": entity_filter.player_name,
            "extracted_stat": entity_filter.stat_type,
            "extracted_position": entity_filter.position_type,
            "is_ranking_query": search_strategy["is_ranking_query"],
            "ranking_count": search_strategy.get("ranking_count")
        },
        search_strategy={
            "method": search_method,
            "applied_filters": filters,
            "search_limit": search_strategy["search_limit"],
            "total_results_found": len(docs)
        },
        results=docs[:limit],  # 요청된 개수만 반환
        performance_metrics={
            "execution_time_ms": round(execution_time, 2),
            "results_count": len(docs),
            "avg_similarity": round(avg_similarity, 4),
            "similarity_range": {
                "min": round(min_similarity, 4),
                "max": round(max_similarity, 4)
            },
            "data_coverage": {
                "unique_teams": list(unique_teams),
                "unique_years": list(unique_years),
                "unique_tables": list(unique_tables)
            }
        }
    )


@router.get("/test-entity-extraction")
async def test_entity_extraction(
    q: str = Query(..., min_length=2, description="엔티티 추출을 테스트할 질문")
):
    """
    질문에서 엔티티 추출이 올바르게 작동하는지 테스트하는 엔드포인트입니다.
    """
    search_strategy = enhance_search_strategy(q)
    entity_filter = search_strategy["entity_filter"]
    
    return {
        "query": q,
        "extracted_entities": {
            "season_year": entity_filter.season_year,
            "team_id": entity_filter.team_id,
            "player_name": entity_filter.player_name,
            "stat_type": entity_filter.stat_type,
            "position_type": entity_filter.position_type,
            "league_type": entity_filter.league_type
        },
        "search_strategy": {
            "is_ranking_query": search_strategy["is_ranking_query"],
            "ranking_count": search_strategy.get("ranking_count"),
            "search_limit": search_strategy["search_limit"]
        },
        "generated_filters": search_strategy["db_filters"]
    }


@router.get("/compare-methods")
async def compare_search_methods(
    q: str = Query(..., min_length=2, description="비교 테스트할 질문"),
    limit: int = Query(10, ge=1, le=20),
    pipeline=Depends(get_rag_pipeline),
):
    """
    단일 쿼리 검색과 다중 쿼리 검색 성능을 비교하는 엔드포인트입니다.
    """
    search_strategy = enhance_search_strategy(q)
    entity_filter = search_strategy["entity_filter"]
    filters = search_strategy["db_filters"]
    
    # 단일 쿼리 검색
    start_single = time.time()
    docs_single = await pipeline.retrieve(q, filters=filters, limit=limit)
    time_single = (time.time() - start_single) * 1000
    
    # 다중 쿼리 검색
    start_multi = time.time()
    docs_multi = await pipeline.retrieve_with_multi_query(
        q, entity_filter, filters=filters
    )
    time_multi = (time.time() - start_multi) * 1000
    
    # 결과 비교 분석
    single_similarities = [doc.get('similarity', 0) for doc in docs_single]
    multi_similarities = [doc.get('similarity', 0) for doc in docs_multi[:limit]]
    
    return {
        "query": q,
        "comparison": {
            "single_query": {
                "execution_time_ms": round(time_single, 2),
                "results_count": len(docs_single),
                "avg_similarity": round(sum(single_similarities) / len(single_similarities) if single_similarities else 0, 4),
                "results_preview": [doc.get('title', doc.get('content', '')[:100]) for doc in docs_single[:3]]
            },
            "multi_query": {
                "execution_time_ms": round(time_multi, 2),
                "results_count": len(docs_multi),
                "avg_similarity": round(sum(multi_similarities) / len(multi_similarities) if multi_similarities else 0, 4),
                "results_preview": [doc.get('title', doc.get('content', '')[:100]) for doc in docs_multi[:3]]
            }
        },
        "recommendation": "multi_query" if len(docs_multi) > len(docs_single) else "single_query"
    }
