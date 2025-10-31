"""레거시 호환을 위해 RAG 파이프라인을 감싸는 챗봇 어댑터 모듈."""

import asyncio
from typing import Dict

from psycopg2 import connect

from app.config import get_settings
from app.core.rag import RAGPipeline
from app.ml.intent_router import predict_intent


class KBOChatbot:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def _process_async(self, question: str) -> Dict[str, str]:
        conn = connect(self.settings.database_url)
        pipeline = RAGPipeline(settings=self.settings, connection=conn)
        intent = predict_intent(question)
        result = await pipeline.run(question, intent=intent)
        conn.close()
        return {
            "answer": result["answer"],
            "query_executed": None,
            "execution_time": None,
        }

    def process_question(self, question: str) -> Dict[str, str]:
        return asyncio.run(self._process_async(question))

    def test_connection(self) -> bool:
        return True


chatbot = KBOChatbot()
