from __future__ import annotations

"""Embedding provider selection module for the KBO RAG system.

Supported providers:
    - `gemini`: Google Gemini embeddings via batchEmbedContents
    - `hf`: HuggingFace Sentence-Transformers models
    - `openai`: OpenAI embeddings API (direct)
    - `openrouter`: OpenRouter embeddings endpoint
    - `local`: Deterministic sine-based embeddings for tests
"""

import asyncio
import json
import logging
import math
import os
import time
from typing import List, Optional, Sequence, Tuple

import httpx

from ..config import Settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


def _ensure_dimension(
    vectors: Sequence[Sequence[float]],
    expected: Optional[int],
) -> None:
    """Warn if vector dimensions mismatch the configured expectation."""
    if not vectors or not expected:
        return
    mismatched = {len(vec) for vec in vectors if len(vec) != expected}
    if mismatched:
        logger.warning(
            "Embedding dimension mismatch detected. expected=%s, observed=%s",
            expected,
            sorted(list(mismatched))[:3],
        )


async def _embed_local(texts: Sequence[str]) -> List[List[float]]:
    """Generate deterministic sine-based vectors (64d) for local testing."""
    vectors: List[List[float]] = []
    for text in texts:
        seed = hash(text) % 1000
        vector = [math.sin(idx + seed) for idx in range(64)]
        vectors.append(vector)
    return vectors


async def _embed_hf(
    texts: Sequence[str],
    settings: Settings,
    *,
    max_concurrency: int,
) -> List[List[float]]:
    """Embed using a Sentence-Transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise EmbeddingError(
            "sentence-transformers 패키지가 필요합니다. `pip install sentence-transformers` 후 다시 시도하세요."
        ) from exc

    env_model = getattr(settings, "hf_embed_model", None) or os.getenv("HF_EMBED_MODEL")
    model_name = settings.embed_model or env_model or "intfloat/multilingual-e5-large"
    env_batch = getattr(settings, "hf_embed_batch", None) or os.getenv("HF_BATCH")
    batch_size = int(env_batch) if env_batch else 16
    if batch_size <= 0:
        batch_size = len(texts) or 1

    model = SentenceTransformer(model_name)
    embeddings = await asyncio.to_thread(
        model.encode,
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    if isinstance(embeddings, list):
        return [list(map(float, vec)) for vec in embeddings]
    return [list(map(float, embeddings[idx])) for idx in range(len(texts))]


async def _embed_gemini(
    texts: Sequence[str],
    settings: Settings,
    *,
    max_concurrency: int,
) -> List[List[float]]:
    """Embed with Gemini batchEmbedContents API."""
    if not settings.gemini_api_key:
        raise EmbeddingError("GEMINI_API_KEY가 설정되어 있지 않습니다.")

    raw_model = settings.gemini_embed_model or "text-embedding-004"
    model_path = raw_model if raw_model.startswith("models/") else f"models/{raw_model}"
    env_dim = getattr(settings, "embed_dim", None) or os.getenv("EMBED_DIM")
    embed_dim = int(env_dim) if env_dim else 1536
    env_batch = getattr(settings, "embed_batch_size", None) or os.getenv("EMBED_BATCH_SIZE")
    batch_size = int(env_batch) if env_batch else 32
    if batch_size <= 0:
        batch_size = len(texts) or 1
    env_token_limit = os.getenv("GEMINI_MAX_TOKENS")
    max_tokens = int(env_token_limit) if env_token_limit else 3072
    rpm = int(os.getenv("GEMINI_RPM") or 60)
    min_delay = 60.0 / rpm if rpm > 0 else 0.0
    max_retries = 5

    url = f"https://generativelanguage.googleapis.com/v1/{model_path}:batchEmbedContents"
    params = {"key": settings.gemini_api_key}
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": settings.gemini_api_key,
    }

    effective_concurrency = max(max_concurrency, 1)
    limits = httpx.Limits(
        max_connections=effective_concurrency,
        max_keepalive_connections=1,
    )

    async def post_chunk(chunk: Sequence[str]) -> List[List[float]]:
        prepared_texts: List[str] = []
        for text in chunk:
            trimmed_text, trimmed = _ensure_token_limit(text, max_tokens)
            if trimmed:
                logger.debug(
                    "Trimmed text to respect token limit (approx %s tokens -> %s)",
                    _estimate_tokens(text),
                    _estimate_tokens(trimmed_text),
                )
            prepared_texts.append(trimmed_text)

        payload = {
            "model": model_path,
            "requests": [
                {
                    "model": model_path,
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": embed_dim,
                }
                for text in prepared_texts
            ],
        }
        async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
            response = await client.post(
                url,
                params=params,
                headers=headers,
                content=json.dumps(payload, ensure_ascii=False),
            )
        try:
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            snippet = response.text[:200]
            raise EmbeddingError(f"Gemini 응답 파싱 실패: {snippet}") from exc

        if "error" in data:
            raise EmbeddingError(f"Gemini 오류: {data['error']}")

        embeddings: List[List[float]] = []
        if "responses" in data:
            for item in data.get("responses", []):
                values = item.get("embedding", {}).get("values")
                if values:
                    embeddings.append(list(map(float, values)))
        if "embeddings" in data:
            for item in data.get("embeddings", []):
                values = item.get("values") or item.get("embedding", {}).get("values")
                if values:
                    embeddings.append(list(map(float, values)))

        if not embeddings:
            raise EmbeddingError(f"Gemini가 임베딩을 반환하지 않았습니다: {data}")
        return embeddings

    batches = [list(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    results: List[List[float]] = []
    last_request_at = 0.0

    for idx, chunk in enumerate(batches):
        if min_delay > 0 and idx > 0:
            elapsed = time.perf_counter() - last_request_at
            if elapsed < min_delay:
                await asyncio.sleep(min_delay - elapsed)

        attempt = 0
        backoff = 1.0
        while True:
            try:
                chunk_vectors = await post_chunk(chunk)
                results.extend(chunk_vectors)
                last_request_at = time.perf_counter()
                break
            except EmbeddingError as exc:
                attempt += 1
                if attempt >= max_retries:
                    raise
                sleep_for = backoff + (0.1 * attempt)
                logger.warning(
                    "Gemini embedding retry %s/%s due to %s; sleeping %.1fs",
                    attempt,
                    max_retries,
                    exc,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)
                backoff = min(backoff * 2, 30.0)
            except httpx.HTTPError as exc:  # pragma: no cover
                attempt += 1
                if attempt >= max_retries:
                    raise EmbeddingError(f"Gemini HTTP 오류: {exc}") from exc
                sleep_for = backoff + (0.1 * attempt)
                logger.warning(
                    "Gemini HTTP 오류 재시도 %s/%s (%s); %.1fs 후 재시도",
                    attempt,
                    max_retries,
                    exc,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)
                backoff = min(backoff * 2, 30.0)

    if len(results) != len(texts):
        raise EmbeddingError(
            f"Gemini 응답 수가 입력 수와 일치하지 않습니다. input={len(texts)} output={len(results)}"
        )
    _ensure_dimension(results, embed_dim)
    return results


async def _embed_openai(
    texts: Sequence[str],
    settings: Settings,
    *,
    max_concurrency: int,
) -> List[List[float]]:
    """Embed using OpenAI API directly."""
    if not settings.openai_api_key:
        raise EmbeddingError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    model = settings.openai_embed_model or settings.embed_model or "text-embedding-3-small"
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": list(texts),
    }

    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        snippet = (response.text or "")[:300]
        raise EmbeddingError(
            f"OpenAI API 오류 {response.status_code}: {snippet}"
        )

    try:
        data = response.json()
    except Exception as exc:
        snippet = (response.text or "")[:300]
        raise EmbeddingError(f"OpenAI JSON 파싱 실패: {snippet}") from exc

    embeddings: List[List[float]] = []
    for item in data.get("data", []):
        vec = item.get("embedding")
        if vec:
            embeddings.append(list(map(float, vec)))

    if not embeddings:
        raise EmbeddingError(f"OpenAI가 임베딩을 반환하지 않았습니다: {data}")

    env_dim = getattr(settings, "embed_dim", None) or os.getenv("EMBED_DIM")
    expected_dim = int(env_dim) if env_dim else None
    _ensure_dimension(embeddings, expected_dim)
    return embeddings


async def _embed_openrouter(
    texts: Sequence[str],
    settings: Settings,
    *,
    max_concurrency: int,
) -> List[List[float]]:
    """Embed using OpenRouter embeddings API."""
    if not settings.openrouter_api_key:
        raise EmbeddingError("OPENROUTER_API_KEY가 설정되어 있지 않습니다.")

    model = (
        settings.openrouter_embed_model
        or settings.embed_model
        or "openai/text-embedding-3-small"
    )
    base_url = settings.openrouter_base_url.rstrip("/")
    url = f"{base_url}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if settings.openrouter_referer:
        headers["HTTP-Referer"] = settings.openrouter_referer
    if settings.openrouter_app_title:
        headers["X-Title"] = settings.openrouter_app_title
    else:
        headers.setdefault("X-Title", "KBO-Embedding")

    payload = {
        "model": model,
        "input": list(texts),
    }

    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        response = await client.post(url, json=payload, headers=headers)

    content_type = response.headers.get("content-type", "")
    if response.status_code != 200:
        snippet = (response.text or "")[:300]
        raise EmbeddingError(
            f"OpenRouter {response.status_code} {content_type}: {snippet}"
        )
    if "application/json" not in content_type:
        snippet = (response.text or "")[:300]
        raise EmbeddingError(
            f"OpenRouter 비JSON 응답: {content_type}: {snippet}"
        )

    try:
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        snippet = (response.text or "")[:300]
        raise EmbeddingError(f"OpenRouter JSON 파싱 실패: {snippet}") from exc

    embeddings: List[List[float]] = []
    for item in data.get("data", []):
        vec = item.get("embedding")
        if vec:
            embeddings.append(list(map(float, vec)))

    if not embeddings:
        raise EmbeddingError(f"OpenRouter가 임베딩을 반환하지 않았습니다: {data}")

    env_dim = getattr(settings, "embed_dim", None) or os.getenv("EMBED_DIM")
    expected_dim = int(env_dim) if env_dim else None
    _ensure_dimension(embeddings, expected_dim)
    return embeddings


async def async_embed_texts(
    texts: Sequence[str],
    settings: Settings,
    max_concurrency: int = 1,
) -> List[List[float]]:
    """Asynchronously embed texts according to the configured provider."""
    if not texts:
        return []

    provider = settings.embed_provider
    if provider == "local":
        return await _embed_local(texts)
    if provider == "hf":
        vectors = await _embed_hf(texts, settings, max_concurrency=max_concurrency)
        env_dim = getattr(settings, "embed_dim", None) or os.getenv("EMBED_DIM")
        expected_dim = int(env_dim) if env_dim else None
        _ensure_dimension(vectors, expected_dim)
        return vectors
    if provider == "gemini":
        return await _embed_gemini(texts, settings, max_concurrency=max_concurrency)
    if provider == "openai":
        return await _embed_openai(texts, settings, max_concurrency=max_concurrency)
    if provider == "openrouter":
        return await _embed_openrouter(texts, settings, max_concurrency=max_concurrency)

    raise EmbeddingError(f"Unsupported provider: {provider}")


def embed_texts(
    texts: Sequence[str],
    settings: Settings,
    *,
    max_concurrency: int = 1,
) -> List[List[float]]:
    """Synchronous wrapper for async_embed_texts()."""
    return asyncio.run(async_embed_texts(texts, settings, max_concurrency=max_concurrency))
def _estimate_tokens(text: str) -> int:
    """Rudimentary token approximation (4 chars ≈ 1 token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _ensure_token_limit(text: str, max_tokens: int) -> Tuple[str, bool]:
    """Trim text to stay under max token budget, returns (trimmed_text, trimmed?)."""
    if max_tokens <= 0:
        return text, False
    approx = _estimate_tokens(text)
    if approx <= max_tokens:
        return text, False
    # Rough trimming: keep proportional characters.
    target_chars = max_tokens * 4
    trimmed = text[:target_chars]
    return trimmed, True
