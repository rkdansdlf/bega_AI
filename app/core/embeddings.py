from __future__ import annotations

"""KBO RAG 시스템을 위한 임베딩 프로바이더 선택 모듈입니다.

이 모듈은 다양한 임베딩 생성 API 또는 라이브러리를 추상화하여 일관된 인터페이스를 제공합니다.
이를 통해 필요에 따라 임베딩 모델을 쉽게 교체할 수 있습니다.

지원되는 프로바이더:
    - `gemini`: Google Gemini (batchEmbedContents API 사용)
    - `openai`: OpenAI 임베딩 API
    - `openrouter`: OpenRouter 임베딩 엔드포인트
    - `hf`: HuggingFace Sentence-Transformers 라이브러리 (로컬 모델)
    - `local`: 테스트용 로컬 임베딩 (결정론적 사인파 기반)
"""

import asyncio
import json
import logging
import math
import os
import re
import time
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import httpx

from ..config import Settings

logger = logging.getLogger(__name__)

_QUERY_EMBED_CACHE_MAX = int(os.getenv("EMBED_QUERY_CACHE_MAX", "2048"))
_QUERY_EMBED_CACHE: "OrderedDict[str, List[float]]" = OrderedDict()
_QUERY_EMBED_LOCK = asyncio.Lock()
_QUERY_WHITESPACE_RE = re.compile(r"\s+")


class EmbeddingError(RuntimeError):
    """임베딩 생성 과정에서 오류가 발생했을 때 사용하는 예외 클래스."""


def _normalize_query(text: str) -> str:
    if not text:
        return ""
    return _QUERY_WHITESPACE_RE.sub(" ", text).strip().casefold()


def _embed_signature(settings: Settings) -> str:
    provider = settings.embed_provider or "unknown"
    env_dim = getattr(settings, "embed_dim", None) or os.getenv("EMBED_DIM")
    embed_dim = str(env_dim) if env_dim else ""

    if provider == "openai":
        model = settings.openai_embed_model or settings.embed_model or "text-embedding-3-small"
        return f"{provider}:{model}:{embed_dim}"
    if provider == "openrouter":
        model = (
            settings.openrouter_embed_model
            or settings.embed_model
            or "openai/text-embedding-3-small"
        )
        return f"{provider}:{model}:{embed_dim}"
    if provider == "gemini":
        model = settings.gemini_embed_model or settings.embed_model or ""
        return f"{provider}:{model}:{embed_dim}"
    if provider == "hf":
        env_model = getattr(settings, "hf_embed_model", None) or os.getenv("HF_EMBED_MODEL")
        model = settings.embed_model or env_model or "intfloat/multilingual-e5-large"
        return f"{provider}:{model}"
    if provider == "local":
        return f"{provider}:local"

    model = settings.embed_model or ""
    return f"{provider}:{model}:{embed_dim}"


async def _get_cached_query_embedding(cache_key: str) -> Optional[List[float]]:
    if _QUERY_EMBED_CACHE_MAX <= 0:
        return None
    async with _QUERY_EMBED_LOCK:
        cached = _QUERY_EMBED_CACHE.get(cache_key)
        if cached is None:
            return None
        _QUERY_EMBED_CACHE.move_to_end(cache_key)
        return cached


async def _set_cached_query_embedding(cache_key: str, embedding: List[float]) -> None:
    if _QUERY_EMBED_CACHE_MAX <= 0:
        return
    async with _QUERY_EMBED_LOCK:
        _QUERY_EMBED_CACHE[cache_key] = embedding
        _QUERY_EMBED_CACHE.move_to_end(cache_key)
        while len(_QUERY_EMBED_CACHE) > _QUERY_EMBED_CACHE_MAX:
            _QUERY_EMBED_CACHE.popitem(last=False)


def _ensure_dimension(
    vectors: Sequence[Sequence[float]],
    expected: Optional[int],
) -> None:
    """생성된 벡터의 차원이 설정된 기대값과 일치하는지 확인하고, 다를 경우 경고를 로깅합니다."""
    if not vectors or not expected:
        return
    mismatched = {len(vec) for vec in vectors if len(vec) != expected}
    if mismatched:
        logger.warning(
            "임베딩 차원 불일치 감지. 기대값=%s, 실제값=%s",
            expected,
            sorted(list(mismatched))[:3],
        )


async def _embed_local(texts: Sequence[str], settings: Settings) -> List[List[float]]:
    """로컬 테스트를 위해 결정론적인 사인파 기반 벡터를 생성합니다."""
    env_dim = getattr(settings, "embed_dim", None) or os.getenv("EMBED_DIM")
    dim = int(env_dim) if env_dim else 1536
    
    vectors: List[List[float]] = []
    for text in texts:
        seed = hash(text) % 1000
        # simple deterministic vector
        vector = [math.sin(idx + seed) for idx in range(dim)]
        vectors.append(vector)
    return vectors


async def _embed_hf(
    texts: Sequence[str],
    settings: Settings,
    *,
    max_concurrency: int,
) -> List[List[float]]:
    """HuggingFace의 Sentence-Transformers 모델을 사용하여 텍스트를 임베딩합니다."""
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise EmbeddingError(
            "sentence-transformers 패키지가 필요합니다. `pip install sentence-transformers` 후 다시 시도하세요."
        ) from exc

    # 설정 또는 환경 변수에서 모델 이름과 배치 크기를 가져옵니다.
    env_model = getattr(settings, "hf_embed_model", None) or os.getenv("HF_EMBED_MODEL")
    model_name = settings.embed_model or env_model or "intfloat/multilingual-e5-large"
    env_batch = getattr(settings, "hf_embed_batch", None) or os.getenv("HF_BATCH")
    batch_size = int(env_batch) if env_batch else 16
    if batch_size <= 0:
        batch_size = len(texts) or 1

    # 모델을 로드하고 임베딩을 생성합니다.
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
    """Google Gemini의 `batchEmbedContents` API를 사용하여 텍스트를 임베딩합니다."""
    if not settings.gemini_api_key:
        raise EmbeddingError("GEMINI_API_KEY가 설정되어 있지 않습니다.")

    # API 요청에 필요한 파라미터를 설정합니다.
    raw_model = settings.gemini_embed_model or settings.embed_model
    if not raw_model:
        raise EmbeddingError("GEMINI_EMBED_MODEL이 설정되어 있지 않습니다.")
    model_path = raw_model if raw_model.startswith("models/") else f"models/{raw_model}"
    env_dim = getattr(settings, "embed_dim", None) or os.getenv("EMBED_DIM")
    embed_dim = int(env_dim) if env_dim else 1536
    env_batch = getattr(settings, "embed_batch_size", None) or os.getenv("EMBED_BATCH_SIZE")
    batch_size = int(env_batch) if env_batch else 32
    if batch_size <= 0:
        batch_size = len(texts) or 1
    env_token_limit = os.getenv("GEMINI_MAX_TOKENS")
    max_tokens = int(env_token_limit) if env_token_limit else 3072
    rpm = int(os.getenv("GEMINI_RPM") or 60)  # 분당 요청 수 제한
    min_delay = 60.0 / rpm if rpm > 0 else 0.0
    max_retries = 5

    url = f"https://generativelanguage.googleapis.com/v1/{model_path}:batchEmbedContents"
    params = {"key": settings.gemini_api_key}
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": settings.gemini_api_key,
    }

    # 동시성 제어를 위한 HTTP 클라이언트 설정을 구성합니다.
    effective_concurrency = max(max_concurrency, 1)
    limits = httpx.Limits(
        max_connections=effective_concurrency,
        max_keepalive_connections=1,
    )

    async def post_chunk(chunk: Sequence[str]) -> List[List[float]]:
        """텍스트 청크를 API에 전송하고 임베딩 결과를 받습니다."""
        prepared_texts: List[str] = []
        for text in chunk:
            # 텍스트가 토큰 제한을 초과하지 않도록 자릅니다.
            trimmed_text, trimmed = _ensure_token_limit(text, max_tokens)
            if trimmed:
                logger.debug(
                    "토큰 제한을 위해 텍스트를 잘랐습니다 (약 %s 토큰 -> %s 토큰)",
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
        except Exception as exc:
            snippet = response.text[:200]
            raise EmbeddingError(f"Gemini 응답 파싱 실패: {snippet}") from exc

        if "error" in data:
            raise EmbeddingError(f"Gemini 오류: {data['error']}")

        embeddings: List[List[float]] = []
        # API 응답에서 임베딩 값을 추출합니다.
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

    # 전체 텍스트를 배치 크기만큼 나누어 처리합니다.
    batches = [list(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    results: List[List[float]] = []
    last_request_at = 0.0

    # 각 배치를 순회하며 API 요청을 보냅니다.
    for idx, chunk in enumerate(batches):
        # RPM 제한을 준수하기 위해 필요한 경우 대기합니다.
        if min_delay > 0 and idx > 0:
            elapsed = time.perf_counter() - last_request_at
            if elapsed < min_delay:
                await asyncio.sleep(min_delay - elapsed)

        # API 요청 실패 시 재시도 로직 (지수 백오프 사용)
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
                    "Gemini 임베딩 재시도 %s/%s. 원인: %s. %.1fs 후 재시도합니다.",
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
                    "Gemini HTTP 오류 발생. %s/%s 재시도. %.1fs 후 재시도합니다. (%s)",
                    attempt,
                    max_retries,
                    sleep_for,
                    exc,
                )
                await asyncio.sleep(sleep_for)
                backoff = min(backoff * 2, 30.0)

    if len(results) != len(texts):
        raise EmbeddingError(
            f"Gemini 응답 수가 입력 수와 일치하지 않습니다. 입력={len(texts)}, 출력={len(results)}"
        )
    _ensure_dimension(results, embed_dim)
    return results


async def _embed_openai(
    texts: Sequence[str],
    settings: Settings,
    *,
    max_concurrency: int,
) -> List[List[float]]:
    """OpenAI의 임베딩 API를 직접 사용하여 텍스트를 임베딩합니다."""
    if not settings.openai_api_key:
        raise EmbeddingError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    model = settings.openai_embed_model or settings.embed_model or "text-embedding-3-small"
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    
    env_batch = getattr(settings, "embed_batch_size", None) or os.getenv("EMBED_BATCH_SIZE")
    batch_size = int(env_batch) if env_batch else 32
    if batch_size <= 0:
        batch_size = len(texts) or 1
    
    max_retries = 5
    
    effective_concurrency = max(max_concurrency, 1)
    limits = httpx.Limits(
        max_connections=effective_concurrency,
        max_keepalive_connections=1,
    )
    timeout = httpx.Timeout(60.0, connect=10.0)

    async def post_chunk(chunk: Sequence[str], client: httpx.AsyncClient) -> List[List[float]]:
        payload = {
            "model": model,
            "input": list(chunk),
        }
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
        
        return embeddings

    batches = [list(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    results: List[List[float]] = []
    
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for chunk in batches:
            attempt = 0
            backoff = 1.0
            while True:
                try:
                    chunk_vectors = await post_chunk(chunk, client)
                    results.extend(chunk_vectors)
                    break
                except (EmbeddingError, httpx.HTTPError) as exc:
                    attempt += 1
                    if attempt >= max_retries:
                        raise EmbeddingError(f"OpenAI 임베딩 실패 후 최대 재시도 도달: {exc}") from exc
                    
                    sleep_for = backoff + (0.1 * attempt)
                    logger.warning(
                        "OpenAI 임베딩 재시도 %s/%s. 원인: %s. %.1fs 후 재시도합니다.",
                        attempt,
                        max_retries,
                        exc,
                        sleep_for,
                    )
                    await asyncio.sleep(sleep_for)
                    backoff = min(backoff * 2, 30.0)

    if len(results) != len(texts):
        raise EmbeddingError(
            f"OpenAI 응답 수가 입력 수와 일치하지 않습니다. 입력={len(texts)}, 출력={len(results)}"
        )

    env_dim = getattr(settings, "embed_dim", None) or os.getenv("EMBED_DIM")
    expected_dim = int(env_dim) if env_dim else None
    _ensure_dimension(results, expected_dim)
    return results


async def _embed_openrouter(
    texts: Sequence[str],
    settings: Settings,
    *,
    max_concurrency: int,
) -> List[List[float]]:
    """OpenRouter의 임베딩 API를 사용하여 텍스트를 임베딩합니다."""
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
            f"OpenRouter 오류 {response.status_code} ({content_type}): {snippet}"
        )
    if "application/json" not in content_type:
        snippet = (response.text or "")[:300]
        raise EmbeddingError(
            f"OpenRouter가 JSON이 아닌 응답을 반환했습니다: {content_type}: {snippet}"
        )

    try:
        data = response.json()
    except Exception as exc:
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
    """설정된 프로바이더에 따라 비동기적으로 텍스트 임베딩을 수행합니다."""
    if not texts:
        return []

    provider = settings.embed_provider
    if provider == "local":
        return await _embed_local(texts, settings)
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

    raise EmbeddingError(f"지원되지 않는 프로바이더입니다: {provider}")


async def async_embed_query(
    query: str,
    settings: Settings,
    max_concurrency: int = 1,
) -> List[float]:
    """단일 검색 질의를 임베딩하고, 정규화 키 기준으로 LRU 캐시를 사용합니다."""
    if not query:
        return []
    normalized = _normalize_query(query)
    cache_key = f"{_embed_signature(settings)}:{normalized}"

    cached = await _get_cached_query_embedding(cache_key)
    if cached is not None:
        logger.debug("[Embeddings] Cache HIT for query (len=%d)", len(query))
        return cached

    start_time = time.perf_counter()
    vectors = await async_embed_texts([query], settings, max_concurrency=max_concurrency)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info("[Embeddings] Query embedding took %.2fms (len=%d)", elapsed_ms, len(query))
    
    if not vectors:
        return []

    embedding = vectors[0]
    await _set_cached_query_embedding(cache_key, embedding)
    return embedding


def embed_texts(
    texts: Sequence[str],
    settings: Settings,
    *,
    max_concurrency: int = 1,
) -> List[List[float]]:
    """`async_embed_texts` 함수의 동기적 래퍼(wrapper)입니다.
    이미 이벤트 루프가 실행 중인 경우(예: Uvicorn), 별도 스레드에서 실행하여 RuntimeError를 방지합니다.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
        
    if loop and loop.is_running():
        # 실행 중인 루프가 있으면 별도 스레드에서 실행
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run, 
                async_embed_texts(texts, settings, max_concurrency=max_concurrency)
            )
            return future.result()
    else:
        # 실행 중인 루프가 없으면 바로 실행
        return asyncio.run(async_embed_texts(texts, settings, max_concurrency=max_concurrency))

def _estimate_tokens(text: str) -> int:
    """간단한 토큰 수 추정 (4 글자 ≈ 1 토큰)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _ensure_token_limit(text: str, max_tokens: int) -> Tuple[str, bool]:
    """텍스트가 최대 토큰 예산을 초과하지 않도록 자릅니다. (잘린 텍스트, 잘림 여부)를 반환합니다."""
    if max_tokens <= 0:
        return text, False
    approx = _estimate_tokens(text)
    if approx <= max_tokens:
        return text, False
    # 대략적인 자르기: 비율에 맞춰 글자 수를 유지합니다.
    target_chars = max_tokens * 4
    trimmed = text[:target_chars]
    return trimmed, True
