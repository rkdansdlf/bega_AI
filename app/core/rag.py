"""
RAG (Retrieval-Augmented Generation) 파이프라인의 핵심 로직을 구현한 모듈입니다.

이 모듈은 사용자 쿼리에 대해 관련성 높은 정보를 검색하고, 
LLM(Large Language Model)을 사용하여 자연스러운 답변을 생성하는 과정을 담당합니다.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence
from psycopg2.extensions import connection as PgConnection

import httpx

from ..config import Settings
from .embeddings import async_embed_texts
from .prompts import FOLLOWUP_PROMPT, SYSTEM_PROMPT
from .retrieval import similarity_search
from .tools import try_answer_with_sql


class RAGPipeline:
    """
    검색(Retrieval)과 생성(Generation)을 결합하여 답변을 생성하는 RAG 파이프라인을 관리합니다.

    이 클래스는 다음과 같은 주요 기능을 수행합니다:
    1. 사용자 쿼리를 임베딩하여 벡터로 변환합니다.
    2. 변환된 벡터를 사용하여 데이터베이스에서 유사도 높은 문서를 검색합니다.
    3. 검색된 문서와 사용자 쿼리를 기반으로 LLM에 전달할 프롬프트를 구성합니다.
    4. LLM(Gemini 또는 OpenRouter)을 호출하여 최종 답변을 생성합니다.
    5. 필요한 경우, SQL 쿼리를 통해 직접 답변을 시도하는 도구(tool)를 사용합니다.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        connection: PgConnection,
    ) -> None:
        """
        RAGPipeline 인스턴스를 초기화합니다.

        Args:
            settings: 애플리케이션의 설정을 담고 있는 객체.
            connection: PostgreSQL 데이터베이스 연결 객체.
        """
        self.settings = settings
        self.connection = connection

    async def retrieve(
        self,
        query: str,
        *,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        주어진 쿼리와 가장 관련성 높은 문서를 데이터베이스에서 검색합니다.

        Args:
            query: 사용자의 원본 질문.
            limit: 검색할 최대 문서 수. 지정하지 않으면 기본값을 사용합니다.
            filters: 검색 결과를 필터링하기 위한 조건.

        Returns:
            검색된 문서들의 리스트. 각 문서는 사전(dict) 형태로 제공됩니다.
        """
        # 검색할 문서 수를 설정하거나 기본값을 사용합니다.
        limit = limit or self.settings.default_search_limit
        
        # 쿼리를 임베딩 벡터로 변환합니다.
        embeddings = await async_embed_texts([query], self.settings)
        if not embeddings:
            return []

        # 쿼리가 짧은 경우(5단어 이하), 키워드 검색을 함께 활용하여 정확도를 높입니다.
        keyword = query if len(query.split()) <= 5 else None
        
        # 유사도 검색을 수행합니다.
        docs = similarity_search(
            self.connection,
            embeddings[0],
            limit=limit,
            filters=filters,
            keyword=keyword,
        )
        return docs

    async def _generate_with_gemini(
        self,
        messages: Sequence[Dict[str, str]],
    ) -> str:
        """
        Google Gemini 모델을 사용하여 답변을 생성합니다.

        Args:
            messages: 모델에 전달될 메시지 시퀀스 (시스템 프롬프트, 사용자 프롬프트 등).

        Returns:
            생성된 답변 텍스트.

        Raises:
            RuntimeError: API 키가 없거나, `google-generativeai` 패키지가 설치되지 않았거나,
                          API 호출에 실패한 경우 발생합니다.
        """
        if not self.settings.gemini_api_key:
            raise RuntimeError("Gemini를 사용하려면 GEMINI_API_KEY가 필요합니다.")
        try:
            import google.generativeai as genai
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "google-generativeai 패키지가 설치되지 않았습니다. Gemini를 사용하려면 설치해주세요."
            ) from exc

        # Gemini API를 설정합니다.
        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(self.settings.gemini_model)

        # 메시지를 시스템과 사용자 역할로 분리하여 프롬프트를 구성합니다.
        system_parts = [
            msg["content"] for msg in messages if msg.get("role") == "system"
        ]
        user_parts = [
            msg["content"] for msg in messages if msg.get("role") == "user"
        ]
        combined_prompt = "\n\n".join(system_parts + user_parts).strip()

        def _invoke() -> str:  # noqa: E302
            """동기적인 API 호출을 비동기적으로 실행하기 위한 래퍼 함수."""
            response = model.generate_content(combined_prompt)
            if hasattr(response, "text"):
                return response.text
            if isinstance(response, dict):
                return response.get("text", "")
            return ""

        # 동기 함수를 별도의 스레드에서 실행하여 비동기 컨텍스트를 차단하지 않도록 합니다.
        answer = await asyncio.to_thread(_invoke)
        if not answer:
            raise RuntimeError("Gemini가 빈 응답을 반환했습니다.")
        return answer

    async def _generate_with_openrouter(
        self,
        messages: Sequence[Dict[str, str]],
    ) -> str:
        """
        OpenRouter를 통해 LLM을 호출하여 답변을 생성합니다.

        Args:
            messages: 모델에 전달될 메시지 시퀀스.

        Returns:
            생성된 답변 텍스트.

        Raises:
            RuntimeError: API 키가 없거나 API 요청에 실패한 경우 발생합니다.
        """
        if not self.settings.openrouter_api_key:
            raise RuntimeError("OpenRouter를 사용하려면 OPENROUTER_API_KEY가 필요합니다.")

        base_url = self.settings.openrouter_base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        
        # HTTP 요청 헤더를 설정합니다.
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.openrouter_referer:
            headers["HTTP-Referer"] = self.settings.openrouter_referer
        if self.settings.openrouter_app_title:
            headers["X-Title"] = self.settings.openrouter_app_title

        # 요청 페이로드를 구성합니다.
        payload: Dict[str, Any] = {
            "model": self.settings.openrouter_model,
            "messages": list(messages),
        }
        if self.settings.max_output_tokens:
            payload["max_tokens"] = self.settings.max_output_tokens

        # 비동기 HTTP 클라이언트를 사용하여 API에 요청을 보냅니다.
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        try:
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            snippet = response.text[:200]
            raise RuntimeError(f"OpenRouter 요청 실패: {snippet}") from exc

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter가 응답을 반환하지 않았습니다: {data}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not content:
            raise RuntimeError("OpenRouter 응답이 비어 있습니다.")
        return content

    async def _generate(
        self,
        messages: Sequence[Dict[str, str]],
    ) -> str:
        """
        설정에 지정된 LLM 공급자(provider)를 사용하여 답변을 생성합니다.

        Args:
            messages: 모델에 전달될 메시지 시퀀스.

        Returns:
            생성된 답변 텍스트.

        Raises:
            RuntimeError: 지원되지 않는 LLM 공급자가 지정된 경우 발생합니다.
        """
        provider = self.settings.llm_provider
        if provider == "gemini":
            return await self._generate_with_gemini(messages)
        if provider == "openrouter":
            return await self._generate_with_openrouter(messages)
        raise RuntimeError(f"지원되지 않는 LLM 공급자: {provider}")

    async def run(
        self,
        query: str,
        *,
        intent: str = "freeform",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        전체 RAG 파이프라인을 실행하여 사용자 쿼리에 대한 답변을 생성합니다.

        Args:
            query: 사용자의 원본 질문.
            intent: 쿼리의 의도 (예: 'freeform', 'stats').
            filters: 검색 필터링 조건.

        Returns:
            답변, 인용, 검색된 문서 등 RAG 파이프라인의 최종 결과를 담은 사전.
        """
        # 1. 관련 문서 검색
        docs = await self.retrieve(query, filters=filters)

        # 2. SQL 도구를 사용하여 직접 답변 시도 (주로 통계 관련 쿼리)
        tool_answer = try_answer_with_sql(self.connection, query, filters or {})
        if tool_answer:
            return {
                "answer": tool_answer["text"],
                "citations": tool_answer["citations"],
                "intent": intent,
                "retrieved": docs,
                "strategy": "direct_sql",  # 사용된 전략 명시
            }

        # 3. RAG 기반 답변 생성
        # 검색된 문서들을 컨텍스트로 결합합니다.
        joined_context = "\n\n".join(
            f"[#{doc['id']}] {doc.get('title') or ''}\n{doc['content']}"
            for doc in docs
        )
        # LLM에 전달할 프롬프트를 생성합니다.
        prompt = FOLLOWUP_PROMPT.format(question=query, context=joined_context)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        # LLM을 호출하여 답변을 생성합니다.
        answer = await self._generate(messages)

        # 4. 최종 결과 구성
        citations = [{"id": doc["id"], "title": doc.get("title", "")} for doc in docs]
        return {
            "answer": answer,
            "citations": citations,
            "intent": intent,
            "retrieved": docs,
            "strategy": "rag",  # 사용된 전략 명시
        }