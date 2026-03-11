"""
KBO 규정 검색을 위한 전용 도구입니다.

이 도구는 벡터 데이터베이스에 저장된 KBO 규정집 문서를 검색하여
정확하고 신뢰할 수 있는 규정 정보를 제공합니다.
"""

from contextlib import contextmanager
import logging
from typing import Dict, Any, List
import psycopg
from psycopg.rows import dict_row
from psycopg.errors import UndefinedTable

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

    @contextmanager
    def _connection_scope(self, force_fresh: bool = False):
        conn = self.connection
        if (
            not force_fresh
            and conn is not None
            and not bool(getattr(conn, "closed", False))
        ):
            yield conn
            return

        from ..deps import get_connection_pool

        with get_connection_pool().connection() as pooled_conn:
            yield pooled_conn

    def _normalize_regulation_query(self, query: str) -> str:
        text = " ".join((query or "").strip().split())
        if not text:
            return ""

        removable_tokens = [
            "팬 말투로",
            "팬한테",
            "팬처럼",
            "쉽게",
            "풀어줘",
            "설명해줘",
            "설명 좀",
            "설명",
            "정리해줘",
            "정리 좀",
            "정리",
            "알려줘",
            "말해줘",
            "말해 줘",
            "한 번",
            "한번",
            "좀",
            "다시",
            "지금",
        ]
        normalized = text
        for token in removable_tokens:
            normalized = normalized.replace(token, " ")

        normalized = normalized.replace("보상 선수", "보상선수")
        normalized = normalized.replace("에프에이", "FA")
        normalized = " ".join(normalized.split())
        return normalized

    def _build_regulation_search_terms(self, query: str) -> List[str]:
        original = " ".join((query or "").strip().split())
        normalized = self._normalize_regulation_query(original)
        compact = normalized.replace(" ", "")
        query_lower = original.lower()

        stopwords = {
            "규정",
            "규칙",
            "설명",
            "정리",
            "질문",
            "내용",
            "관련",
            "말투",
            "팬",
            "지금",
            "다시",
            "쉽게",
        }

        terms: List[str] = []

        def add(term: str) -> None:
            candidate = " ".join((term or "").strip().split())
            if len(candidate) < 2:
                return
            if candidate not in terms:
                terms.append(candidate)

        add(original)
        add(normalized)
        if compact and compact != normalized:
            add(compact)

        if "fa" in query_lower or "에프에이" in original:
            add("FA")
            add("자유계약")
            add("FA 보상선수")
        if "보상선수" in compact:
            add("보상선수")
            add("보상 선수")
        if "등록일수" in compact:
            add("등록일수")
        if "엔트리" in original:
            add("엔트리")
            add("1군 엔트리")
        if "말소" in original:
            add("말소")
        if "드래프트" in original:
            add("드래프트")

        for token in normalized.split():
            if len(token) < 2 or token in stopwords:
                continue
            add(token)

        return sorted(terms, key=lambda value: (-len(value), value))

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

            def _run_once(conn: psycopg.Connection):
                with conn.cursor(row_factory=dict_row) as cursor:
                    cursor.execute("""
                        SELECT EXISTS(
                            SELECT 1
                            FROM information_schema.tables
                            WHERE table_schema = 'public'
                              AND table_name = 'rag_chunks'
                        );
                        """)
                    row = cursor.fetchone()
                    table_exists = False
                    if row:
                        if hasattr(row, "get"):
                            for key in ("exists", "?column?", "present"):
                                if key in row:
                                    table_exists = bool(row.get(key))
                                    break
                            if not table_exists:
                                try:
                                    table_exists = bool(next(iter(row.values())))
                                except Exception:
                                    table_exists = False
                        else:
                            try:
                                table_exists = bool(row[0])
                            except Exception:
                                table_exists = bool(row)
                    if not table_exists:
                        return None, None

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
                    search_terms = self._build_regulation_search_terms(query)
                    rows = []
                    matched_term = None

                    for search_term in search_terms:
                        search_pattern = f"%{search_term}%"
                        cursor.execute(
                            text_search_query,
                            (
                                search_pattern,
                                search_pattern,
                                search_pattern,
                                search_pattern,
                                limit,
                            ),
                        )
                        rows = cursor.fetchall()
                        if rows:
                            matched_term = search_term
                            break

                    return rows, matched_term

            try:
                with self._connection_scope() as conn:
                    rows, matched_term = _run_once(conn)
            except Exception as exc:
                if "connection is closed" not in str(exc).lower():
                    raise
                logger.warning(
                    "[RegulationQuery] Retrying with a fresh connection: %s",
                    exc,
                )
                with self._connection_scope(force_fresh=True) as conn:
                    rows, matched_term = _run_once(conn)

            if rows is None:
                logger.warning("[RegulationQuery] rag_chunks table is not available")
                result["error"] = "검색 인덱스(rag_chunks)가 준비되지 않았습니다."
                return result

            if rows:
                result["regulations"] = [dict(row) for row in rows]
                result["found"] = True
                result["total_found"] = len(rows)

                categories = set()
                for row in rows:
                    if row.get("category"):
                        categories.add(row["category"])
                result["categories"] = list(categories)
                result["matched_query"] = matched_term

                logger.info(
                    "[RegulationQuery] Found %s regulations via text search (matched=%s)",
                    len(rows),
                    matched_term,
                )
            else:
                logger.warning(
                    "[RegulationQuery] No regulations found for: %s",
                    query,
                )

        except Exception as e:
            if isinstance(e, UndefinedTable):
                logger.error(f"[RegulationQuery] rag_chunks table not found: {e}")
                result["error"] = "검색 인덱스(rag_chunks)가 준비되지 않았습니다."
            else:
                logger.error(f"[RegulationQuery] Database error: {e}")
                result["error"] = f"데이터베이스 오류: {e}"

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

            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name = 'rag_chunks'
                );
                """)
            row = cursor.fetchone()
            if not (row and row[0]):
                logger.warning("[RegulationQuery] rag_chunks table is not available")
                result["error"] = "검색 인덱스(rag_chunks)가 준비되지 않았습니다."
                return result

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
            if isinstance(e, UndefinedTable):
                logger.error(f"[RegulationQuery] rag_chunks table not found: {e}")
                result["error"] = "검색 인덱스(rag_chunks)가 준비되지 않았습니다."
            else:
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
