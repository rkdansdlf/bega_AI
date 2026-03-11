"""Vector DB에서 설명형 야구 문서를 검색하는 전용 도구입니다."""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Any, Dict, List

import psycopg
from psycopg.rows import dict_row

from ..config import Settings
from ..core.embeddings import embed_texts
from ..core.retrieval import similarity_search

logger = logging.getLogger(__name__)

DOCUMENT_SOURCE_TABLES = ("markdown_docs", "kbo_definitions", "kbo_regulations")
TEMPORAL_KEYWORDS = ("오늘", "지금", "현재", "최신", "최근", "요즘")
RULE_TERMS_KEYWORDS = ("규정", "규칙", "fa", "abs", "룰", "용어", "뜻", "의미")
STRATEGY_METRIC_KEYWORDS = (
    "war",
    "wrc+",
    "ops",
    "whip",
    "fip",
    "babip",
    "세이버",
    "전술",
    "전략",
    "포지션",
    "플래툰",
    "번트",
    "히트앤런",
    "지표",
)
CULTURE_HISTORY_KEYWORDS = (
    "마스코트",
    "응원가",
    "별명",
    "엠블럼",
    "유니폼",
    "팬 문화",
    "응원 문화",
    "구단 역사",
    "전통",
    "라이벌",
    "홈구장",
    "구장",
)


class DocumentQueryTool:
    """비정형 문서(Markdown 등) 검색 전용 도구."""

    def __init__(self, connection: psycopg.Connection):
        self.connection = connection
        self.settings = Settings()

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

    @staticmethod
    def _matches_any(query_lower: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in query_lower for keyword in keywords)

    @staticmethod
    def _focus_terms(query_lower: str) -> list[str]:
        compact = re.sub(r"\s+", "", query_lower)
        for suffix in (
            "설명해줘",
            "알려줘",
            "어떤공이야",
            "어떤공이냐",
            "어떤뜻이야",
            "뜻이야",
            "의미야",
            "뭐야",
            "뭔데",
            "왜치명적이야",
            "왜중요해",
            "어떻게봐",
            "어떻게보나",
        ):
            if compact.endswith(suffix):
                compact = compact[: -len(suffix)]
                break
        compact = re.sub(r"(이|가|은|는|을|를|와|과|의|도|만|중)+$", "", compact)

        stopwords = {
            "뭐야",
            "뭔데",
            "설명해줘",
            "알려줘",
            "뜻",
            "의미",
            "어떤",
            "왜",
            "어떻게",
            "보통",
            "관련",
            "중",
            "공이야",
        }
        terms: list[str] = []
        if len(compact) >= 2:
            terms.append(compact)
        for token in re.split(r"\s+", query_lower):
            cleaned = re.sub(r"[?!.,\"'`]", "", token).strip()
            cleaned = re.sub(r"(이|가|은|는|을|를|와|과|의|도|만|야|요)$", "", cleaned)
            if len(cleaned) < 2 or cleaned in stopwords:
                continue
            terms.append(cleaned)

        deduped: list[str] = []
        for term in terms:
            if term not in deduped:
                deduped.append(term)
        return deduped

    def _lexical_match_boost(self, query_lower: str, doc: Dict[str, Any]) -> float:
        focus_terms = self._focus_terms(query_lower)
        if not focus_terms:
            return 0.0

        searchable_text = " ".join(
            [
                str(doc.get("title", "") or "").lower(),
                str(doc.get("content", "") or "").lower(),
                str(doc.get("source_row_id", "") or "").lower(),
            ]
        )
        compact_text = re.sub(r"\s+", "", searchable_text)

        best_boost = 0.0
        for term in focus_terms:
            compact_term = re.sub(r"\s+", "", term)
            if len(compact_term) < 2:
                continue
            if compact_term in compact_text:
                best_boost = max(best_boost, 0.28 + min(len(compact_term) * 0.01, 0.12))
            elif term in searchable_text:
                best_boost = max(best_boost, 0.16)
        return best_boost

    def _has_direct_focus_term(self, query_lower: str, doc: Dict[str, Any]) -> bool:
        focus_terms = self._focus_terms(query_lower)
        if not focus_terms:
            return False

        searchable_text = " ".join(
            [
                str(doc.get("title", "") or "").lower(),
                str(doc.get("content", "") or "").lower(),
                str(doc.get("source_row_id", "") or "").lower(),
            ]
        )
        compact_text = re.sub(r"\s+", "", searchable_text)

        for term in focus_terms:
            compact_term = re.sub(r"\s+", "", term)
            if len(compact_term) < 2:
                continue
            if compact_term in compact_text or term in searchable_text:
                return True
        return False

    def _search_exact_term_documents(
        self, conn: psycopg.Connection, query_lower: str, limit: int
    ) -> list[Dict[str, Any]]:
        focus_terms = self._focus_terms(query_lower)
        if not focus_terms:
            return []

        clauses: list[str] = []
        params: list[Any] = [list(DOCUMENT_SOURCE_TABLES)]
        for term in focus_terms[:4]:
            like_value = f"%{term}%"
            clauses.append(
                "(lower(title) LIKE %s OR lower(content) LIKE %s OR lower(source_row_id) LIKE %s)"
            )
            params.extend([like_value, like_value, like_value])
        params.append(limit)

        sql = f"""
        SELECT
            id,
            title,
            content,
            source_table,
            source_row_id,
            meta,
            1.0 AS similarity,
            1.0 AS combined_score
        FROM rag_chunks
        WHERE source_table = ANY(%s)
          AND ({' OR '.join(clauses)})
        LIMIT %s
        """

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def _score_document(self, query_lower: str, doc: Dict[str, Any]) -> float:
        score = float(doc.get("combined_score") or doc.get("similarity") or 0.0)
        meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
        knowledge_type = str(meta.get("knowledge_type", ""))
        freshness = str(meta.get("freshness", ""))

        if self._matches_any(query_lower, TEMPORAL_KEYWORDS) and freshness in {
            "seasonal",
            "recent",
            "live",
        }:
            score += 0.08
        if self._matches_any(query_lower, RULE_TERMS_KEYWORDS) and knowledge_type in {
            "rules_terms",
            "regulation",
        }:
            score += 0.06
        if self._matches_any(
            query_lower, STRATEGY_METRIC_KEYWORDS
        ) and knowledge_type in {
            "strategy_metrics",
            "metric_explainer",
        }:
            score += 0.06
        if self._matches_any(
            query_lower, CULTURE_HISTORY_KEYWORDS
        ) and knowledge_type in {
            "culture_history",
            "long_tail_entity",
        }:
            score += 0.06
        if doc.get("source_table") == "markdown_docs":
            score += 0.01
        score += self._lexical_match_boost(query_lower, doc)
        return score

    def _normalize_document(
        self, query_lower: str, doc: Dict[str, Any]
    ) -> Dict[str, Any]:
        normalized = dict(doc)
        meta = (
            normalized.get("meta") if isinstance(normalized.get("meta"), dict) else {}
        )
        content = str(normalized.get("content", "") or "").strip()
        if len(content) > 720:
            content = content[:720].rsplit(" ", 1)[0] + "..."
        normalized["content"] = content
        normalized["source_name"] = (
            meta.get("source_file")
            or meta.get("source_name")
            or normalized.get("source_table")
            or "verified_docs"
        )
        normalized["_score"] = self._score_document(query_lower, normalized)
        return normalized

    def _search_documents_once(
        self, conn: psycopg.Connection, query: str, limit: int
    ) -> list[Dict[str, Any]]:
        embeddings = embed_texts([query], self.settings)
        if not embeddings:
            raise RuntimeError("질문을 임베딩하는 데 실패했습니다.")

        query_lower = query.lower()
        per_source_limit = max(limit * 2, 5)
        collected: List[Dict[str, Any]] = []
        for source_table in DOCUMENT_SOURCE_TABLES:
            docs = similarity_search(
                conn,
                embeddings[0],
                limit=per_source_limit,
                filters={"source_table": source_table},
                keyword=query,
            )
            if not docs:
                docs = similarity_search(
                    conn,
                    embeddings[0],
                    limit=per_source_limit,
                    filters={"source_table": source_table},
                    keyword=None,
                )
            collected.extend(docs)

        collected.extend(
            self._search_exact_term_documents(
                conn,
                query_lower,
                limit=per_source_limit,
            )
        )

        deduped: List[Dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for doc in sorted(
            collected,
            key=lambda item: (
                1 if self._has_direct_focus_term(query_lower, item) else 0,
                self._score_document(query_lower, item),
            ),
            reverse=True,
        ):
            key = (
                str(doc.get("source_table", "")),
                str(doc.get("source_row_id", "")),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(self._normalize_document(query_lower, doc))
            if len(deduped) >= limit:
                break

        return deduped

    def search_documents(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """사용자 질문과 가장 관련성 높은 문서 조각(chunk)을 검색합니다."""
        logger.info("[DocumentQueryTool] Searching documents for: %s", query)

        result = {
            "query": query,
            "documents": [],
            "found": False,
            "error": None,
            "source": "verified_docs",
            "source_tables": list(DOCUMENT_SOURCE_TABLES),
        }

        try:
            try:
                with self._connection_scope() as conn:
                    deduped = self._search_documents_once(conn, query, limit)
            except Exception as exc:
                if "connection is closed" not in str(exc).lower():
                    raise
                logger.warning(
                    "[DocumentQueryTool] Retrying with a fresh connection: %s",
                    exc,
                )
                with self._connection_scope(force_fresh=True) as conn:
                    deduped = self._search_documents_once(conn, query, limit)

            if deduped:
                result["documents"] = deduped
                result["found"] = True
                logger.info(
                    "[DocumentQueryTool] Found %d relevant document chunks.",
                    len(deduped),
                )
            else:
                logger.warning(
                    "[DocumentQueryTool] No relevant documents found for: %s",
                    query,
                )

        except Exception as e:
            logger.error("[DocumentQueryTool] Document search error: %s", e)
            result["error"] = f"문서 검색 중 오류 발생: {e}"

        return result
