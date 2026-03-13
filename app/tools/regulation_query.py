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
from .pooled_connection import connection_scope, run_with_fresh_connection_retry
from .query_logging import (
    ACTION_FIND_RELATED_REGULATIONS,
    ACTION_GET_REGULATION_BY_CATEGORY,
    ACTION_SEARCH_REGULATION,
    ACTION_VALIDATE_REGULATION_REFERENCE,
    REGULATION_QUERY_COMPONENT,
    build_retry_warning_message,
    log_dependency_missing,
    log_query_empty,
    log_query_error,
    log_query_start,
    log_query_success,
)
from .query_result import apply_error, apply_list_results, build_search_result

logger = logging.getLogger(__name__)


class RegulationQueryTool:
    """
    KBO 규정 전용 검색 도구

    이 도구는 다음 원칙을 따릅니다:
    1. 벡터 DB에 저장된 공식 규정 문서만 검색
    2. 정확한 조항과 출처 제공
    3. 추측이나 해석 없이 공식 규정만 인용
    """

    COMPONENT = REGULATION_QUERY_COMPONENT

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
        with connection_scope(self.connection, force_fresh=force_fresh) as conn:
            yield conn

    def _retry_warning_message(self, action: str) -> str:
        return build_retry_warning_message(self.COMPONENT, action)

    def _log_query_start(self, action: str, value: str) -> None:
        log_query_start(
            logger,
            component=self.COMPONENT,
            action=action,
            value=value,
        )

    def _log_query_success(
        self, action: str, count: int, detail: str | None = None
    ) -> None:
        log_query_success(
            logger,
            component=self.COMPONENT,
            action=action,
            count=count,
            detail=detail,
        )

    def _log_query_empty(self, action: str, value: str) -> None:
        log_query_empty(
            logger,
            component=self.COMPONENT,
            action=action,
            value=value,
        )

    def _log_dependency_missing(self, action: str, dependency: str) -> None:
        log_dependency_missing(
            logger,
            component=self.COMPONENT,
            action=action,
            dependency=dependency,
        )

    def _log_query_error(self, action: str, error: str) -> None:
        log_query_error(
            logger,
            component=self.COMPONENT,
            action=action,
            error=error,
        )

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

    @staticmethod
    def _row_truthy_value(row: Any) -> bool:
        if not row:
            return False
        if hasattr(row, "get"):
            for key in ("exists", "?column?", "present"):
                if key in row:
                    return bool(row.get(key))
            try:
                return bool(next(iter(row.values())))
            except Exception:
                return False
        try:
            return bool(row[0])
        except Exception:
            return bool(row)

    def _rag_chunks_table_available(self, conn: psycopg.Connection) -> bool:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name = 'rag_chunks'
                ) AS present;
                """)
            return self._row_truthy_value(cursor.fetchone())

    def _search_regulation_once(
        self, conn: psycopg.Connection, query: str, limit: int
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        if not self._rag_chunks_table_available(conn):
            return None, None

        with conn.cursor(row_factory=dict_row) as cursor:
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
            rows: list[dict[str, Any]] = []
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

    def search_regulation(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        규정 관련 질문을 검색합니다.

        Args:
            query: 검색할 질문
            limit: 반환할 최대 결과 수

        Returns:
            검색 결과 딕셔너리
        """
        self._log_query_start(ACTION_SEARCH_REGULATION, query)

        result = build_search_result(
            query=query,
            regulations=[],
            total_found=0,
            categories=[],
        )

        try:
            rows, matched_term = run_with_fresh_connection_retry(
                connection=self.connection,
                operation=lambda conn: self._search_regulation_once(conn, query, limit),
                logger=logger,
                retry_warning_message=self._retry_warning_message(
                    ACTION_SEARCH_REGULATION
                ),
            )

            if rows is None:
                self._log_dependency_missing(ACTION_SEARCH_REGULATION, "rag_chunks")
                apply_error(result, "검색 인덱스(rag_chunks)가 준비되지 않았습니다.")
                return result

            if rows:
                categories = set()
                for row in rows:
                    if row.get("category"):
                        categories.add(row["category"])
                apply_list_results(
                    result,
                    field="regulations",
                    rows=rows,
                    total_field="total_found",
                    row_mapper=dict,
                    extra_updates={
                        "categories": list(categories),
                        "matched_query": matched_term,
                    },
                )

                self._log_query_success(
                    ACTION_SEARCH_REGULATION,
                    count=len(rows),
                    detail=f"matched_query={matched_term}",
                )
            else:
                self._log_query_empty(ACTION_SEARCH_REGULATION, query)

        except Exception as e:
            if isinstance(e, UndefinedTable):
                self._log_query_error(ACTION_SEARCH_REGULATION, str(e))
                apply_error(result, "검색 인덱스(rag_chunks)가 준비되지 않았습니다.")
            else:
                self._log_query_error(ACTION_SEARCH_REGULATION, str(e))
                apply_error(result, f"데이터베이스 오류: {e}")

        return result

    def _get_regulation_by_category_once(
        self, conn: psycopg.Connection, category: str, limit: int
    ) -> list[dict[str, Any]] | None:
        if not self._rag_chunks_table_available(conn):
            return None

        keywords = self.regulation_categories.get(category, [])
        if not keywords:
            raise ValueError(f"알 수 없는 카테고리: {category}")

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

        search_patterns = [f"%{keyword}%" for keyword in keywords] + [limit]

        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, search_patterns)
            return cursor.fetchall()

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
        self._log_query_start(ACTION_GET_REGULATION_BY_CATEGORY, category)

        result = build_search_result(
            category=category,
            regulations=[],
            total_found=0,
        )

        try:
            rows = run_with_fresh_connection_retry(
                connection=self.connection,
                operation=lambda conn: self._get_regulation_by_category_once(
                    conn, category, limit
                ),
                logger=logger,
                retry_warning_message=self._retry_warning_message(
                    ACTION_GET_REGULATION_BY_CATEGORY
                ),
            )

            if rows is None:
                self._log_dependency_missing(
                    ACTION_GET_REGULATION_BY_CATEGORY,
                    "rag_chunks",
                )
                apply_error(result, "검색 인덱스(rag_chunks)가 준비되지 않았습니다.")
                return result

            if rows:
                apply_list_results(
                    result,
                    field="regulations",
                    rows=rows,
                    total_field="total_found",
                    row_mapper=dict,
                )
                self._log_query_success(
                    ACTION_GET_REGULATION_BY_CATEGORY,
                    count=len(rows),
                    detail=f"category={category}",
                )
            else:
                self._log_query_empty(ACTION_GET_REGULATION_BY_CATEGORY, category)

        except Exception as e:
            if isinstance(e, UndefinedTable):
                self._log_query_error(ACTION_GET_REGULATION_BY_CATEGORY, str(e))
                apply_error(result, "검색 인덱스(rag_chunks)가 준비되지 않았습니다.")
            elif isinstance(e, ValueError):
                self._log_query_error(ACTION_GET_REGULATION_BY_CATEGORY, str(e))
                apply_error(result, str(e))
            else:
                self._log_query_error(ACTION_GET_REGULATION_BY_CATEGORY, str(e))
                apply_error(result, f"카테고리 검색 오류: {e}")

        return result

    def find_related_regulations(self, topic: str) -> Dict[str, Any]:
        """
        특정 주제와 관련된 규정들을 찾습니다.

        Args:
            topic: 검색 주제

        Returns:
            관련 규정 검색 결과
        """
        self._log_query_start(ACTION_FIND_RELATED_REGULATIONS, topic)

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

    def _validate_regulation_reference_once(
        self, conn: psycopg.Connection, regulation_code: str
    ) -> dict[str, Any] | None:
        with conn.cursor(row_factory=dict_row) as cursor:
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
            return dict(row) if row else None

    def validate_regulation_reference(self, regulation_code: str) -> Dict[str, Any]:
        """
        규정 조항 번호로 해당 규정이 존재하는지 확인합니다.

        Args:
            regulation_code: 규정 조항 번호 (예: "01-1", "02-3")

        Returns:
            규정 유효성 검증 결과
        """
        self._log_query_start(ACTION_VALIDATE_REGULATION_REFERENCE, regulation_code)

        result = {
            "regulation_code": regulation_code,
            "exists": False,
            "regulation": None,
            "error": None,
        }

        try:
            row = run_with_fresh_connection_retry(
                connection=self.connection,
                operation=lambda conn: self._validate_regulation_reference_once(
                    conn, regulation_code
                ),
                logger=logger,
                retry_warning_message=self._retry_warning_message(
                    ACTION_VALIDATE_REGULATION_REFERENCE
                ),
            )

            if row:
                result["exists"] = True
                result["regulation"] = row
                self._log_query_success(
                    ACTION_VALIDATE_REGULATION_REFERENCE,
                    count=1,
                    detail=f"regulation_code={regulation_code}",
                )
            else:
                self._log_query_empty(
                    ACTION_VALIDATE_REGULATION_REFERENCE,
                    regulation_code,
                )

        except Exception as e:
            self._log_query_error(ACTION_VALIDATE_REGULATION_REFERENCE, str(e))
            apply_error(result, f"검증 오류: {e}")

        return result
