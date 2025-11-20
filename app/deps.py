"""FastAPI 의존성 주입을 위한 공통 헬퍼를 정의하는 모듈."""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager

import psycopg2
from psycopg2.extensions import connection as PgConnection
from fastapi import Depends

from .config import get_settings
from .core.rag import RAGPipeline
from .ml.intent_router import predict_intent, load_clf
from .agents.baseball_agent import BaseballStatisticsAgent
from .core.prompts import SYSTEM_PROMPT


@asynccontextmanager
async def lifespan(app):
    load_clf()
    yield


def get_db_connection() -> Generator[PgConnection, None, None]:
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url)
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


def get_rag_pipeline(
    conn: PgConnection = Depends(get_db_connection),
) -> RAGPipeline:
    settings = get_settings()
    return RAGPipeline(settings=settings, connection=conn)


def get_agent(
    conn: PgConnection = Depends(get_db_connection),
) -> BaseballStatisticsAgent:
    """Dependency to get an instance of the BaseballStatisticsAgent."""
    settings = get_settings()
    
    # 에이전트가 LLM을 직접 호출할 수 있도록 생성기 함수를 전달해야 합니다.
    # 이 부분은 실제 LLM 호출 로직에 따라 달라집니다.
    # 여기서는 RAGPipeline에 있던 _generate_with_openrouter를 임시로 가져왔다고 가정합니다.
    # 실제로는 LLM 호출을 위한 별도의 클라이언트를 만드는 것이 좋습니다.
    
    # 임시 LLM 생성기 함수
    # To-Do: Refactor LLM client into a separate dependency
    async def llm_generator(messages):
        import httpx
        if not settings.openrouter_api_key:
            raise RuntimeError("OpenRouter API key is required.")
        
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.openrouter_referer or "",
            "X-Title": settings.openrouter_app_title or "",
        }
        payload = {
            "model": settings.openrouter_model,
            "messages": list(messages),
            "max_tokens": settings.max_output_tokens,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("OpenRouter response is empty.")
        return content

    return BaseballStatisticsAgent(connection=conn, llm_generator=llm_generator)


def get_intent_router():
    return predict_intent
