"""
KBO 규정 검색을 위한 전용 도구입니다.

이 도구는 벡터 데이터베이스에 저장된 KBO 규정집 문서를 검색하여
정확하고 신뢰할 수 있는 규정 정보를 제공합니다.
"""

import logging
from typing import Dict, List, Any, Optional
import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


class RegulationQueryTool:
    """
    KBO 규정 전용 검색 도구

    이 도구는 다음 원칙을 따릅니다:
    1. 벡터 DB에 저장된 공식 규정 문서만 검색
    2. 정확한 조항과 출처 제공
    3. 추측이나 해석 없이 공식 규정만 인용
    """

    def __init__(self, connection: psycopg.Connection):
        self.connection = connection

        # 규정 카테고리별 키워드 매핑
        self.regulation_categories = {
            "basic": [
                "리그구성",
                "경기시간",
                "이닝",
                "타이브레이크",
                "지명타자",
                "비디오판독",
            ],
            "player": [
                "선수등록",
                "외국인선수",
                "FA",
                "드래프트",
                "트레이드",
                "웨이버",
            ],
            "game": [
                "경기진행",
                "투구시계",
                "방해",
                "보크",
                "홈런",
                "인필드플라이",
                "우천",
            ],
            "technical": ["기록", "안타", "실책", "승리투수", "세이브", "홀드", "심판"],
            "discipline": ["징계", "폭력", "도박", "약물", "처벌", "벌금", "출장정지"],
            "postseason": [
                "포스트시즌",
                "플레이오프",
                "와일드카드",
                "한국시리즈",
                "4승어드밴티지",
            ],
            "special": ["코로나19", "우천", "태풍", "비상상황", "국제대회", "특별규정"],
            "terms": ["용어정의", "타율", "방어율", "포지션", "심판판정", "기록용어"],
        }

    def search_regulation(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        규정 관련 질문을 검색합니다.

        Args:
            query: 검색할 질문
            limit: 반환할 최대 결과 수

        Returns:
            검색 결과 딕셔너리
        """
        logger.info(f"[RegulationQuery] Searching regulations for: {query}")

        result = {
            "query": query,
            "regulations": [],
            "found": False,
            "total_found": 0,
            "categories": [],
            "error": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

            # 검색 키워드 준비
            search_pattern = f"%{query}%"

            # 텍스트 및 벡터 검색을 모두 활용 (ILIKE로 1차 필터링, embedding으로 순위 결정)
            # 참고: 임베딩 컬럼이 없어, 현재는 텍스트 검색만 수행됩니다.
            text_search_query = """
                SELECT 
                    id,
                    title,
                    content,
                    source_table,
                    meta->>'document_type' as document_type,
                    meta->>'category' as category,
                    meta->>'regulation_code' as regulation_code,
                    CASE 
                        WHEN title ILIKE %s THEN 0.9
                        WHEN content ILIKE %s THEN 0.7
                        ELSE 0.5
                    END as similarity_score
                FROM rag_chunks 
                WHERE source_table IN ('kbo_regulations', 'markdown_docs')
                AND (
                    content ILIKE %s 
                    OR title ILIKE %s
                )
                ORDER BY similarity_score DESC, title
                LIMIT %s;
            """

            cursor.execute(
                text_search_query,
                (
                    search_pattern,  # for similarity_score title
                    search_pattern,  # for similarity_score content
                    search_pattern,  # for WHERE content
                    search_pattern,  # for WHERE title
                    limit,
                ),
            )

            rows = cursor.fetchall()

            if rows:
                result["regulations"] = [dict(row) for row in rows]
                result["found"] = True
                result["total_found"] = len(rows)

                # 카테고리 추출
                categories = set()
                for row in rows:
                    if row.get("category"):
                        categories.add(row["category"])
                result["categories"] = list(categories)

                logger.info(
                    f"[RegulationQuery] Found {len(rows)} regulations via text search"
                )
            else:
                logger.warning(f"[RegulationQuery] No regulations found for: {query}")

        except Exception as e:
            logger.error(f"[RegulationQuery] Database error: {e}")
            result["error"] = f"데이터베이스 오류: {e}"
        finally:
            if "cursor" in locals():
                cursor.close()

        return result

    def get_regulation_by_category(
        self, category: str, limit: int = 10
    ) -> Dict[str, Any]:
        """
        특정 카테고리의 규정들을 조회합니다.

        Args:
            category: 규정 카테고리 (basic, player, game 등)
            limit: 반환할 최대 결과 수

        Returns:
            카테고리별 규정 결과
        """
        logger.info(f"[RegulationQuery] Getting regulations for category: {category}")

        result = {
            "category": category,
            "regulations": [],
            "found": False,
            "total_found": 0,
            "error": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

            # 카테고리별 키워드 가져오기
            keywords = self.regulation_categories.get(category, [])

            if not keywords:
                result["error"] = f"알 수 없는 카테고리: {category}"
                return result

            # 키워드 기반 검색
            keyword_conditions = " OR ".join([f"content ILIKE %s" for _ in keywords])
            query = f"""
                SELECT 
                    id,
                    title,
                    content,
                    source_table,
                    meta->>'document_type' as document_type,
                    meta->>'category' as category,
                    meta->>'regulation_code' as regulation_code
                FROM rag_chunks 
                WHERE source_table = 'kbo_regulations'
                AND ({keyword_conditions})
                ORDER BY title
                LIMIT %s;
            """

            # 검색 패턴 준비
            search_patterns = [f"%{keyword}%" for keyword in keywords] + [limit]

            cursor.execute(query, search_patterns)
            rows = cursor.fetchall()

            if rows:
                result["regulations"] = [dict(row) for row in rows]
                result["found"] = True
                result["total_found"] = len(rows)
                logger.info(
                    f"[RegulationQuery] Found {len(rows)} regulations for category {category}"
                )
            else:
                logger.warning(
                    f"[RegulationQuery] No regulations found for category: {category}"
                )

        except Exception as e:
            logger.error(f"[RegulationQuery] Category search error: {e}")
            result["error"] = f"카테고리 검색 오류: {e}"
        finally:
            if "cursor" in locals():
                cursor.close()

        return result

    def find_related_regulations(self, topic: str) -> Dict[str, Any]:
        """
        특정 주제와 관련된 규정들을 찾습니다.

        Args:
            topic: 검색 주제

        Returns:
            관련 규정 검색 결과
        """
        logger.info(f"[RegulationQuery] Finding related regulations for topic: {topic}")

        # 주제에 따른 카테고리 매핑
        topic_to_category = {
            "선수": "player",
            "경기": "game",
            "기록": "technical",
            "징계": "discipline",
            "플레이오프": "postseason",
            "포스트시즌": "postseason",
            "용어": "terms",
            "특별": "special",
        }

        # 주제에서 카테고리 찾기
        matched_category = None
        for key, category in topic_to_category.items():
            if key in topic:
                matched_category = category
                break

        if matched_category:
            # 카테고리별 검색
            return self.get_regulation_by_category(matched_category)
        else:
            # 일반 검색
            return self.search_regulation(topic)

    def validate_regulation_reference(self, regulation_code: str) -> Dict[str, Any]:
        """
        규정 조항 번호로 해당 규정이 존재하는지 확인합니다.

        Args:
            regulation_code: 규정 조항 번호 (예: "01-1", "02-3")

        Returns:
            규정 유효성 검증 결과
        """
        logger.info(f"[RegulationQuery] Validating regulation: {regulation_code}")

        result = {
            "regulation_code": regulation_code,
            "exists": False,
            "regulation": None,
            "error": None,
        }

        try:
            cursor = self.connection.cursor(row_factory=dict_row)

            query = """
                SELECT 
                    id,
                    title,
                    content,
                    meta->>'regulation_code' as regulation_code,
                    meta->>'category' as category
                FROM rag_chunks 
                WHERE source_table = 'kbo_regulations'
                AND meta->>'regulation_code' = %s
                LIMIT 1;
            """

            cursor.execute(query, (regulation_code,))
            row = cursor.fetchone()

            if row:
                result["exists"] = True
                result["regulation"] = dict(row)
                logger.info(f"[RegulationQuery] Found regulation: {regulation_code}")
            else:
                logger.warning(
                    f"[RegulationQuery] Regulation not found: {regulation_code}"
                )

        except Exception as e:
            logger.error(f"[RegulationQuery] Validation error: {e}")
            result["error"] = f"검증 오류: {e}"
        finally:
            if "cursor" in locals():
                cursor.close()

        return result
