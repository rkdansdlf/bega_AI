"""
Vector DB에서 문서(비정형 텍스트) 검색을 위한 전용 도구입니다.

주로 규정, 용어 설명, 팬 문화 등 정형화된 DB로 답변하기 어려운
설명형 질문에 대한 답변 근거를 찾을 때 사용됩니다.
"""

import logging
from typing import Dict, List, Any, Optional
import psycopg
from ..core.embeddings import async_embed_texts, embed_texts
from ..config import Settings
from ..core.retrieval import similarity_search

logger = logging.getLogger(__name__)


class DocumentQueryTool:
    """
    비정형 문서(Markdown 등) 검색 전용 도구
    """

    def __init__(self, connection: psycopg.Connection):
        self.connection = connection
        self.settings = Settings()

    def search_documents(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        사용자 질문과 가장 관련성 높은 문서 조각(chunk)을 Vector DB에서 검색합니다.

        Args:
            query: 검색할 사용자 원본 질문
            limit: 반환할 최대 결과 수

        Returns:
            검색 결과 딕셔너리
        """
        logger.info(f"[DocumentQueryTool] Searching documents for: {query}")

        result = {"query": query, "documents": [], "found": False, "error": None}

        try:
            # 질문을 임베딩하여 검색
            embeddings = embed_texts([query], self.settings)
            if not embeddings:
                result["error"] = "질문을 임베딩하는 데 실패했습니다."
                return result

            # 유사도 검색 실행
            # 이 때, 'markdown_docs'와 같이 비정형 텍스트 소스를 주로 검색하도록 필터링 할 수 있습니다.
            docs = similarity_search(
                self.connection,
                embeddings[0],
                limit=limit,
                filters={
                    "source_table": "markdown_docs"
                },  # 설명형 문서는 markdown_docs에 저장됨
                keyword=query,
            )

            if docs:
                result["documents"] = docs
                result["found"] = True
                logger.info(
                    f"[DocumentQueryTool] Found {len(docs)} relevant document chunks."
                )
            else:
                logger.warning(
                    f"[DocumentQueryTool] No relevant documents found for: {query}"
                )

        except Exception as e:
            logger.error(f"[DocumentQueryTool] Document search error: {e}")
            result["error"] = f"문서 검색 중 오류 발생: {e}"

        return result
