"""레거시 시스템과의 호환성을 위해 RAG 파이프라인을 감싸는 챗봇 어댑터 모듈입니다.

이 클래스는 기존 시스템의 인터페이스를 유지하면서, 내부적으로는 새로운 RAG 파이프라인을
호출하여 질문에 대한 답변을 생성합니다. 동기적인 메소드 호출을 비동기 로직으로
연결하는 역할을 합니다.
"""

import asyncio
from typing import Dict

from psycopg2 import connect

from app.config import get_settings
from app.core.rag import RAGPipeline
from app.ml.intent_router import predict_intent


class KBOChatbot:
    """KBO 챗봇의 레거시 인터페이스를 제공하는 클래스."""

    def __init__(self) -> None:
        """KBOChatbot 인스턴스를 초기화합니다."""
        self.settings = get_settings()

    async def _process_async(self, question: str) -> Dict[str, str]:
        """질문 처리의 핵심 비동기 로직입니다.

        RAG 파이프라인을 초기화하고 실행하여 답변을 생성합니다.
        """
        # 데이터베이스 연결을 생성합니다.
        # 참고: 매번 연결을 새로 생성하는 것은 비효율적일 수 있으나,
        # 레거시 동기 인터페이스와의 호환성을 위해 이 구조를 유지합니다.
        conn = connect(self.settings.database_url)
        try:
            pipeline = RAGPipeline(settings=self.settings, connection=conn)
            intent = predict_intent(question)  # 질문의 의도를 파악합니다.
            result = await pipeline.run(question, intent=intent)  # RAG 파이프라인 실행
        finally:
            conn.close()  # 연결을 반드시 닫습니다.
        
        return {
            "answer": result["answer"],
            "query_executed": None,  # 레거시 필드, 현재 사용되지 않음
            "execution_time": None,  # 레거시 필드, 현재 사용되지 않음
        }

    def process_question(self, question: str) -> Dict[str, str]:
        """사용자의 질문을 동기적으로 처리합니다.

        내부적으로는 비동기 `_process_async` 메소드를 실행하고 그 결과를 반환합니다.
        """
        return asyncio.run(self._process_async(question))

    def test_connection(self) -> bool:
        """레거시 연결 테스트 메소드. 항상 True를 반환합니다."""
        return True


# 전역 챗봇 인스턴스 생성
chatbot = KBOChatbot()