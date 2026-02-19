"""
Query Transformation과 Multi-query Retrieval을 위한 모듈입니다.

사용자의 짧은 질문을 다양한 관점에서 확장하고 변형하여
벡터 검색의 정확도를 높이는 기능을 제공합니다.
"""

import asyncio
import logging
from typing import List, Dict, Any, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryVariation:
    """쿼리 변형 정보와 메타데이터를 저장하는 클래스"""

    query: str  # 변형된 질문 텍스트
    variation_type: (
        str  # 변형 유형: 'original', 'expanded', 'statistical', 'contextual'
    )
    weight: float = 1.0  # 검색 결과 가중치 (0.1 ~ 1.0)


class QueryTransformer:
    """질문 변형 및 다중 쿼리 검색을 관리하는 클래스"""

    def __init__(self, llm_generate_func=None):
        """
        쿼리 변형기 초기화

        인수:
            llm_generate_func: LLM 텍스트 생성 함수 (선택사항)
        """
        self.llm_generate = llm_generate_func

    def expand_query_with_rules(
        self, original_query: str, entity_filter
    ) -> List[QueryVariation]:
        """
        규칙 기반으로 쿼리를 확장합니다.
        빠르고 예측 가능한 방법으로 다양한 쿼리 변형을 생성합니다.
        """
        variations = [QueryVariation(original_query, "original", 1.0)]

        # 1. 통계 지표 확장
        if entity_filter.stat_type:
            stat_expansions = self._expand_stat_queries(original_query, entity_filter)
            variations.extend(stat_expansions)

        # 2. 팀/선수 컨텍스트 확장
        if entity_filter.team_id or entity_filter.player_name:
            context_expansions = self._expand_context_queries(
                original_query, entity_filter
            )
            variations.extend(context_expansions)

        # 3. 순위/랭킹 쿼리 확장
        if self._is_ranking_query(original_query):
            ranking_expansions = self._expand_ranking_queries(
                original_query, entity_filter
            )
            variations.extend(ranking_expansions)

        logger.info(f"[QueryTransformer] Generated {len(variations)} query variations")
        return variations[:5]  # 최대 5개로 제한

    def _expand_stat_queries(self, query: str, entity_filter) -> List[QueryVariation]:
        """통계 지표 관련 쿼리를 확장합니다."""
        variations = []
        stat = entity_filter.stat_type

        # 통계 지표별 상세 확장
        if stat == "ops":
            variations.extend(
                [
                    QueryVariation(f"{query} 출루율 장타율", "statistical", 0.9),
                    QueryVariation(f"OPS 높은 선수 순위", "statistical", 0.8),
                ]
            )
        elif stat == "era":
            variations.extend(
                [
                    QueryVariation(f"{query} 평균자책 방어율", "statistical", 0.9),
                    QueryVariation(f"ERA 낮은 투수 랭킹", "statistical", 0.8),
                ]
            )
        elif stat == "home_runs":
            variations.extend(
                [
                    QueryVariation(f"{query} 홈런 순위 타자", "statistical", 0.9),
                    QueryVariation(f"장타력 홈런왕", "statistical", 0.7),
                ]
            )
        elif stat == "war":
            variations.extend(
                [
                    QueryVariation(f"{query} WAR 대체선수대비승수", "statistical", 0.9),
                    QueryVariation(f"팀 기여도 높은 선수", "statistical", 0.8),
                ]
            )

        return variations

    def _expand_context_queries(
        self, query: str, entity_filter
    ) -> List[QueryVariation]:
        """팀이나 선수 컨텍스트를 확장합니다."""
        variations = []

        if entity_filter.team_id:
            team_name = self._get_full_team_name(entity_filter.team_id)
            variations.extend(
                [
                    QueryVariation(f"{team_name} {query}", "contextual", 0.8),
                    QueryVariation(
                        f"{entity_filter.team_id} 소속 선수 기록", "contextual", 0.7
                    ),
                ]
            )

        if entity_filter.player_name:
            variations.extend(
                [
                    QueryVariation(
                        f"{entity_filter.player_name} 선수 개인 기록", "contextual", 0.8
                    ),
                    QueryVariation(
                        f"{entity_filter.player_name} 시즌 성적", "contextual", 0.7
                    ),
                ]
            )

        return variations

    def _expand_ranking_queries(
        self, query: str, entity_filter
    ) -> List[QueryVariation]:
        """순위/랭킹 관련 쿼리를 확장합니다."""
        variations = []

        # 순위 관련 동의어 확장
        ranking_synonyms = ["상위", "톱", "best", "최고", "1위", "리더"]
        for synonym in ranking_synonyms:
            if synonym not in query:
                variations.append(QueryVariation(f"{synonym} {query}", "expanded", 0.6))

        # 포지션별 랭킹 확장
        if entity_filter.position_type == "pitcher":
            variations.extend(
                [
                    QueryVariation(f"투수 {query} ERA WHIP", "expanded", 0.7),
                    QueryVariation(f"선발 불펜 투수 순위", "expanded", 0.6),
                ]
            )
        elif entity_filter.position_type == "batter":
            variations.extend(
                [
                    QueryVariation(f"타자 {query} OPS 타율", "expanded", 0.7),
                    QueryVariation(f"타격 순위 홈런 타점", "expanded", 0.6),
                ]
            )

        return variations

    def _get_full_team_name(self, team_id: str) -> str:
        """팀 ID를 전체 팀명으로 변환합니다."""
        team_names = {
            "KIA": "KIA 타이거즈",
            "HT": "KIA 타이거즈",
            "LG": "LG 트윈스",
            "DB": "두산 베어스",
            "OB": "두산 베어스",
            "DO": "두산 베어스",
            "두산": "두산 베어스",
            "롯데": "롯데 자이언츠",
            "삼성": "삼성 라이온즈",
            "키움": "키움 히어로즈",
            "KH": "키움 히어로즈",
            "KI": "키움 히어로즈",
            "WO": "키움 히어로즈",
            "NX": "키움 히어로즈",
            "한화": "한화 이글스",
            "KT": "KT 위즈",
            "NC": "NC 다이노스",
            "SSG": "SSG 랜더스",
            "SK": "SSG 랜더스",
        }
        return team_names.get(team_id, team_id)

    def _is_ranking_query(self, query: str) -> bool:
        """순위/랭킹 관련 질문인지 판단합니다."""
        ranking_keywords = [
            "순위",
            "랭킹",
            "상위",
            "1위",
            "최고",
            "톱",
            "TOP",
            "베스트",
        ]
        return any(keyword in query for keyword in ranking_keywords)

    async def llm_expand_query(self, original_query: str) -> List[QueryVariation]:
        """
        LLM을 사용하여 쿼리를 확장합니다.
        더 창의적이고 복합적인 쿼리 변형을 생성할 수 있습니다.
        """
        if not self.llm_generate:
            return []

        expansion_prompt = f"""다음 KBO 야구 질문을 3가지 다른 방식으로 다시 표현해주세요. 각각은 같은 의도이지만 다른 키워드를 사용해야 합니다.

원본 질문: {original_query}

1. 상세 확장 버전: (통계 용어나 전문 표현 추가)
2. 맥락 추가 버전: (시즌, 리그 등 맥락 정보 포함)  
3. 동의어 버전: (다른 표현으로 바꿔서)

각 버전은 한 줄로만 작성하고, 번호와 설명 없이 질문만 작성해주세요."""

        try:
            messages = [{"role": "user", "content": expansion_prompt}]
            response = await self.llm_generate(messages)

            # 응답을 파싱하여 변형 쿼리 추출
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            variations = []

            for i, line in enumerate(lines[:3]):  # 최대 3개
                if line and len(line) > 10:  # 너무 짧은 응답 제외
                    variation_type = ["detailed", "contextual", "synonym"][i]
                    variations.append(
                        QueryVariation(line, variation_type, 0.8 - i * 0.1)
                    )

            logger.info(
                f"[QueryTransformer] LLM generated {len(variations)} variations"
            )
            return variations

        except Exception as e:
            logger.warning(f"[QueryTransformer] LLM expansion failed: {e}")
            return []


async def multi_query_retrieval(
    query_variations: List[QueryVariation],
    retrieve_func,
    filters: Dict[str, Any],
    limit_per_query: int = 5,
) -> List[Dict[str, Any]]:
    """
    여러 쿼리 변형으로 병렬 검색을 수행하고 결과를 결합합니다.

    Args:
        query_variations: 변형된 쿼리 목록
        retrieve_func: 검색 함수
        filters: 검색 필터
        limit_per_query: 각 쿼리당 검색 제한

    Returns:
        가중치가 적용되고 중복이 제거된 최종 문서 목록
    """
    logger.info(
        f"[MultiQuery] Starting retrieval with {len(query_variations)} variations"
    )

    # 병렬로 모든 쿼리 변형에 대해 검색 수행
    tasks = []
    for variation in query_variations:
        task = retrieve_func(variation.query, filters=filters, limit=limit_per_query)
        tasks.append((variation, task))

    # 모든 검색 작업을 병렬 실행
    all_results = []
    for variation, task in tasks:
        try:
            docs = await task
            # 각 문서에 가중치 정보 추가
            for doc in docs:
                doc["_query_weight"] = variation.weight
                doc["_query_type"] = variation.variation_type
                doc["_source_query"] = variation.query
            all_results.extend(docs)
        except Exception as e:
            logger.warning(
                f"[MultiQuery] Failed to retrieve for '{variation.query}': {e}"
            )

    # 중복 제거 및 점수 결합
    unique_docs = {}
    for doc in all_results:
        doc_id = doc.get("id")
        if doc_id not in unique_docs:
            unique_docs[doc_id] = doc
        else:
            # 기존 문서와 가중치를 결합하여 더 높은 점수 계산
            existing = unique_docs[doc_id]
            existing_score = existing.get("similarity", 0) * existing.get(
                "_query_weight", 1
            )
            new_score = doc.get("similarity", 0) * doc.get("_query_weight", 1)

            if new_score > existing_score:
                unique_docs[doc_id] = doc
            else:
                # 가중 평균으로 점수 결합
                combined_weight = (
                    existing.get("_query_weight", 1) + doc.get("_query_weight", 1)
                ) / 2
                existing["_query_weight"] = combined_weight

    # 최종 점수로 정렬 (유사도 * 쿼리 가중치)
    final_docs = list(unique_docs.values())
    final_docs.sort(
        key=lambda d: d.get("similarity", 0) * d.get("_query_weight", 1), reverse=True
    )

    logger.info(
        f"[MultiQuery] Combined {len(all_results)} results into {len(final_docs)} unique docs"
    )
    return final_docs
