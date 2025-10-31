"""RAG 파이프라인의 검색·생성 로직을 구현한 핵심 모듈."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence
from psycopg2.extensions import connection as PgConnection

from ..config import Settings
from .embeddings import async_embed_texts
from .prompts import FOLLOWUP_PROMPT, SYSTEM_PROMPT
from .retrieval import similarity_search
from .tools import try_answer_with_sql


class RAGPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        connection: PgConnection,
    ) -> None:
        self.settings = settings
        self.connection = connection

    async def retrieve(
        self,
        query: str,
        *,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        limit = limit or self.settings.default_search_limit
        embeddings = await async_embed_texts([query], self.settings)
        if not embeddings:
            return []
        keyword = query if len(query.split()) <= 5 else None
        docs = similarity_search(
            self.connection,
            embeddings[0],
            limit=limit,
            filters=filters,
            keyword=keyword,
        )
        return docs

    async def _generate_with_gemini(
        self, messages: Sequence[Dict[str, str]]
    ) -> str:
        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required when using Gemini.")
        try:
            import google.generativeai as genai
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "google-generativeai package not installed; install it to use Gemini."
            ) from exc

        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(self.settings.gemini_model)

        system_parts = [
            msg["content"] for msg in messages if msg.get("role") == "system"
        ]
        user_parts = [
            msg["content"] for msg in messages if msg.get("role") == "user"
        ]
        combined_prompt = "\n\n".join(system_parts + user_parts).strip()

        def _invoke() -> str:
            response = model.generate_content(combined_prompt)
            if hasattr(response, "text"):
                return response.text
            if isinstance(response, dict):
                return response.get("text", "")
            return ""

        answer = await asyncio.to_thread(_invoke)
        if not answer:
            raise RuntimeError("Gemini returned an empty response.")
        return answer

    async def _generate(
        self, messages: Sequence[Dict[str, str]]
    ) -> str:
        return await self._generate_with_gemini(messages)

    async def run(
        self,
        query: str,
        *,
        intent: str = "freeform",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        docs = await self.retrieve(query, filters=filters)

        # tool-based attempt first for stats queries
        tool_answer = try_answer_with_sql(self.connection, query, filters or {})
        if tool_answer:
            return {
                "answer": tool_answer["text"],
                "citations": tool_answer["citations"],
                "intent": intent,
                "retrieved": docs,
                "strategy": "direct_sql",
            }

        joined_context = "\n\n".join(
            f"[#{doc['id']}] {doc.get('title') or ''}\n{doc['content']}"
            for doc in docs
        )
        prompt = FOLLOWUP_PROMPT.format(question=query, context=joined_context)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        answer = await self._generate(messages)

        citations = [{"id": doc["id"], "title": doc.get("title", "")} for doc in docs]
        return {
            "answer": answer,
            "citations": citations,
            "intent": intent,
            "retrieved": docs,
            "strategy": "rag",
        }
