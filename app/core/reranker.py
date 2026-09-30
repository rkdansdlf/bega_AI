"""Reranker providers for the RAG pipeline.

``score``  – reorders by the scores retrieval already produced (v1 behaviour,
             no model involved; NOT a semantic reranker).
``http``   – a real cross-encoder / rerank API (Cohere/Jina/TEI-style
             ``POST <url>`` with ``{model, query, documents}`` returning
             ``results: [{index, relevance_score}]``). Sends only chunk text,
             never baseball data fetched from elsewhere.

Any provider failure degrades to the score ordering (fail-open) and is
reported through ``RerankOutcome.degraded`` so it shows up in metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

import httpx

logger = logging.getLogger(__name__)

RERANKER_VERSION_SCORE = "score_v1"
_MAX_DOC_CHARS = 2000


@dataclass
class RerankOutcome:
    docs: List[Dict[str, Any]]
    reranker: str
    version: str
    degraded: bool = False
    reason: Optional[str] = None
    scores: List[float] = field(default_factory=list)


class Reranker(Protocol):
    name: str
    version: str

    async def rerank(
        self, query: str, docs: Sequence[Dict[str, Any]], top_n: int
    ) -> RerankOutcome: ...


def _score_key(doc: Dict[str, Any]) -> tuple:
    return (
        float(doc.get("combined_score") or 0.0),
        float(doc.get("similarity") or 0.0),
        float(doc.get("quality_score") or 0.0),
    )


class ScoreReranker:
    name = "score"
    version = RERANKER_VERSION_SCORE

    async def rerank(
        self, query: str, docs: Sequence[Dict[str, Any]], top_n: int
    ) -> RerankOutcome:
        ordered = sorted(docs, key=_score_key, reverse=True)[:top_n]
        return RerankOutcome(ordered, self.name, self.version)


class HTTPCrossEncoderReranker:
    name = "http"

    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_s: float = 3.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.url = url
        self.model = model
        self.version = f"http:{model}"
        self._api_key = api_key
        self._timeout = timeout_s
        self._client = client
        self._fallback = ScoreReranker()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._client is not None:
            resp = await self._client.post(
                self.url, json=payload, headers=self._headers(), timeout=self._timeout
            )
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    self.url, json=payload, headers=self._headers()
                )
        resp.raise_for_status()
        return resp.json()

    async def rerank(
        self, query: str, docs: Sequence[Dict[str, Any]], top_n: int
    ) -> RerankOutcome:
        if not docs:
            return RerankOutcome([], self.name, self.version)
        payload = {
            "model": self.model,
            "query": query,
            "documents": [
                str(d.get("content") or d.get("title") or "")[:_MAX_DOC_CHARS]
                for d in docs
            ],
            "top_n": min(top_n, len(docs)),
        }
        try:
            body = await self._post(payload)
            results = body.get("results") or body.get("data") or []
            scored = []
            for item in results:
                idx = int(item["index"])
                if not 0 <= idx < len(docs):
                    raise ValueError(f"rerank index out of range: {idx}")
                scored.append(
                    (idx, float(item.get("relevance_score", item.get("score", 0.0))))
                )
            if not scored:
                raise ValueError("empty rerank response")
        except Exception as exc:  # noqa: BLE001 - fail open to score ordering
            logger.warning("[Reranker] provider failed, using score order: %s", exc)
            fallback = await self._fallback.rerank(query, docs, top_n)
            fallback.degraded = True
            fallback.reason = type(exc).__name__
            fallback.version = f"{self.version}->{fallback.version}"
            return fallback

        scored.sort(key=lambda pair: pair[1], reverse=True)
        picked = scored[:top_n]
        out_docs = []
        for idx, score in picked:
            doc = dict(docs[idx])
            doc["rerank_score"] = score
            out_docs.append(doc)
        return RerankOutcome(
            out_docs,
            self.name,
            self.version,
            scores=[score for _, score in picked],
        )


def build_reranker(settings: Any) -> Optional[Reranker]:
    """Reranker for ``settings`` or None when reranking is disabled."""
    if not bool(getattr(settings, "rag_rerank_enabled", False)):
        return None
    provider = str(getattr(settings, "rag_reranker_provider", "score") or "score")
    if provider == "http":
        url = getattr(settings, "rag_reranker_url", None)
        model = getattr(settings, "rag_reranker_model", None)
        if not url or not model:
            logger.warning(
                "[Reranker] http provider needs RAG_RERANKER_URL and "
                "RAG_RERANKER_MODEL; using score reranker"
            )
            return ScoreReranker()
        return HTTPCrossEncoderReranker(
            url=url,
            model=model,
            api_key=getattr(settings, "rag_reranker_api_key", None),
            timeout_s=float(getattr(settings, "rag_reranker_timeout_s", 3.0) or 3.0),
        )
    return ScoreReranker()
