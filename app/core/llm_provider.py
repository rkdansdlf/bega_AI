"""LLM provider adapters, routing, and provider-reported usage.

``rag.py`` used to own the Gemini/OpenRouter HTTP transport. Transport now
lives here behind :class:`LLMProvider`; RAG only asks a :class:`ProviderRouter`
to ``complete``/``stream``. The router skips providers whose circuit is OPEN or
that have no credentials, and records per-provider health.

Token usage is taken from the provider response when present
(``usage_source="provider"``) and only estimated when it is not.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import httpx

from ..observability.tracing import span
from .http_clients import get_shared_httpx_client
from .provider_circuit import ProviderCircuit
from .retry_utils import llm_retry

logger = logging.getLogger(__name__)

Messages = Sequence[Dict[str, str]]


@dataclass
class Usage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    source: str = "estimate"  # "provider" | "estimate"

    @property
    def is_reported(self) -> bool:
        return self.source == "provider"


@dataclass
class CallAttribution:
    """Which provider/model *actually* served a call, versus what was asked.

    ``fallback_depth`` is the actual provider's position in the configured
    order (0 = the requested primary served it). ``fallback_reason`` is why the
    providers ahead of it were not used: an error class, ``circuit_open`` or
    ``not_configured``. Fingerprints, cost, traces and cache provenance must
    all use this instead of the configured ``LLM_PROVIDER``.
    """

    requested_provider: Optional[str] = None
    requested_model: Optional[str] = None
    actual_provider: Optional[str] = None
    actual_model: Optional[str] = None
    fallback_depth: int = 0
    fallback_reason: Optional[str] = None
    attempts: List[Dict[str, str]] = field(default_factory=list)

    @property
    def fell_back(self) -> bool:
        return self.fallback_depth > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_provider": self.requested_provider,
            "requested_model": self.requested_model,
            "actual_provider": self.actual_provider,
            "actual_model": self.actual_model,
            "fallback_depth": self.fallback_depth,
            "fallback_reason": self.fallback_reason,
            "attempts": list(self.attempts),
        }


# Per-request (per async context) record of the most recent call's attribution.
# A ContextVar, not an instance attribute: RAG pipelines are shared across
# concurrent requests.
_last_attribution: ContextVar[Optional[CallAttribution]] = ContextVar(
    "llm_last_attribution", default=None
)


def get_call_attribution() -> Optional[CallAttribution]:
    return _last_attribution.get()


@dataclass
class LLMResult:
    text: str
    provider: str
    model: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    fallback_from: Optional[str] = None
    attribution: Optional[CallAttribution] = None


@dataclass
class UsageSink:
    """Collects usage/attribution from a stream (generators cannot return)."""

    usage: Usage = field(default_factory=Usage)
    attribution: Optional[CallAttribution] = None


def estimate_tokens(text: str) -> int:
    # Coarse, language-agnostic: ~1 token per 3 chars (Korean-heavy text).
    return max(1, len(text or "") // 3)


def usage_from_openai(data: Dict[str, Any]) -> Usage:
    raw = data.get("usage") or {}
    if not raw:
        return Usage()
    prompt = raw.get("prompt_tokens")
    completion = raw.get("completion_tokens")
    total = raw.get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return Usage(prompt, completion, total, source="provider")


def usage_from_gemini(data: Dict[str, Any]) -> Usage:
    raw = data.get("usageMetadata") or {}
    if not raw:
        return Usage()
    prompt = raw.get("promptTokenCount")
    completion = raw.get("candidatesTokenCount")
    total = raw.get("totalTokenCount")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return Usage(prompt, completion, total, source="provider")


class LLMProvider(Protocol):
    name: str

    def is_configured(self) -> bool: ...

    async def complete(self, messages: Messages) -> LLMResult: ...

    def stream(
        self, messages: Messages, sink: Optional[UsageSink] = None
    ) -> AsyncIterator[str]: ...

    def health(self) -> Dict[str, Any]: ...


def _timeout(total: float) -> httpx.Timeout:
    return httpx.Timeout(total, connect=10.0, read=60.0, pool=10.0)


_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10)


class OpenRouterProvider:
    name = "openrouter"

    def __init__(self, settings: Any) -> None:
        self._s = settings

    def is_configured(self) -> bool:
        return bool(getattr(self._s, "openrouter_api_key", None))

    @property
    def model(self) -> Optional[str]:
        return getattr(self._s, "openrouter_model", None)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._s.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._s.openrouter_referer or "",
            "X-Title": self._s.openrouter_app_title or "",
        }

    def _url(self) -> str:
        return f"{self._s.openrouter_base_url.rstrip('/')}/chat/completions"

    def _client(self, total: float):
        return get_shared_httpx_client(
            "openrouter", timeout=_timeout(total), limits=_LIMITS
        )

    @llm_retry
    async def complete(self, messages: Messages) -> LLMResult:
        if not self.is_configured():
            raise RuntimeError(
                "OpenRouter를 사용하려면 OPENROUTER_API_KEY가 필요합니다."
            )
        payload = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": self._s.max_output_tokens,
            "temperature": 0.1,
        }
        response = await self._client(120.0).post(
            self._url(), json=payload, headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"OpenRouter 응답에 choices가 없습니다. Keys: {list(data.keys())}"
            )
        content = (choices[0].get("message") or {}).get("content", "")
        if not content:
            raise RuntimeError("OpenRouter 응답이 비어 있습니다.")
        return LLMResult(content, self.name, self.model, usage_from_openai(data))

    async def stream(
        self, messages: Messages, sink: Optional[UsageSink] = None
    ) -> AsyncIterator[str]:
        if not self.is_configured():
            raise RuntimeError("OpenRouter API 키가 없습니다.")
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": self._s.max_output_tokens,
        }
        async with self._client(60.0).stream(
            "POST", self._url(), json=payload, headers=self._headers(), timeout=60.0
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                logger.error(
                    "[LLM] OpenRouter stream error %s: %r",
                    response.status_code,
                    body[:300],
                )
                response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                try:
                    data = json.loads(line[6:])
                except ValueError:
                    continue
                if sink is not None and data.get("usage"):
                    sink.usage = usage_from_openai(data)
                choices = data.get("choices") or [{}]
                chunk = (choices[0].get("delta") or {}).get("content", "")
                if chunk:
                    yield chunk

    def health(self) -> Dict[str, Any]:
        return {"provider": self.name, "configured": self.is_configured()}


def _to_gemini_contents(messages: Messages) -> List[Dict[str, Any]]:
    """OpenAI-style messages → Gemini contents (system folded into first user)."""
    system_parts: List[str] = []
    contents: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            text = content
            if system_parts and not contents:
                text = (
                    "System Instruction:\n"
                    + "\n\n".join(system_parts)
                    + f"\n\nUser Question:\n{content}"
                )
            contents.append({"role": "user", "parts": [{"text": text}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
    return contents


class GeminiProvider:
    name = "gemini"

    def __init__(self, settings: Any) -> None:
        self._s = settings

    def is_configured(self) -> bool:
        return bool(getattr(self._s, "gemini_api_key", None))

    @property
    def model(self) -> str:
        return getattr(self._s, "gemini_model", None) or "gemini-1.5-flash"

    def _url(self, method: str) -> str:
        return (
            "https://generativelanguage.googleapis.com/v1/models/"
            f"{self.model}:{method}"
        )

    def _payload(self, messages: Messages) -> Dict[str, Any]:
        return {
            "contents": _to_gemini_contents(messages),
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": self._s.max_output_tokens,
            },
        }

    def _client(self):
        return get_shared_httpx_client("gemini", timeout=_timeout(60.0), limits=_LIMITS)

    @llm_retry
    async def complete(self, messages: Messages) -> LLMResult:
        if not self.is_configured():
            raise RuntimeError("Gemini를 사용하려면 GEMINI_API_KEY가 필요합니다.")
        response = await self._client().post(
            self._url("generateContent"),
            json=self._payload(messages),
            params={"key": self._s.gemini_api_key},
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(
                f"Gemini 응답에 candidates가 없습니다. Keys: {list(data.keys())}"
            )
        parts = (candidates[0].get("content") or {}).get("parts", [])
        if not parts or not parts[0].get("text"):
            raise RuntimeError("Gemini 응답이 비어 있습니다.")
        return LLMResult(
            parts[0]["text"], self.name, self.model, usage_from_gemini(data)
        )

    async def stream(
        self, messages: Messages, sink: Optional[UsageSink] = None
    ) -> AsyncIterator[str]:
        if not self.is_configured():
            raise RuntimeError("Gemini API 키가 없습니다.")
        async with self._client().stream(
            "POST",
            self._url("streamGenerateContent"),
            json=self._payload(messages),
            params={"key": self._s.gemini_api_key, "alt": "sse"},
            timeout=60.0,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except ValueError:
                    continue
                if sink is not None and data.get("usageMetadata"):
                    sink.usage = usage_from_gemini(data)
                parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get(
                    "parts"
                ) or [{}]
                chunk = parts[0].get("text", "")
                if chunk:
                    yield chunk

    def health(self) -> Dict[str, Any]:
        return {"provider": self.name, "configured": self.is_configured()}


class AllProvidersFailed(RuntimeError):
    def __init__(self, attempts: List[Dict[str, str]]):
        self.attempts = attempts
        super().__init__(
            "모든 LLM 제공자가 실패했습니다: "
            + "; ".join(f"{a['provider']}={a['error']}" for a in attempts)
        )


class ProviderRouter:
    """Ordered failover across providers with per-provider circuits."""

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        circuits: Optional[Dict[str, ProviderCircuit]] = None,
    ) -> None:
        self.providers = list(providers)
        self.circuits = circuits or {p.name: ProviderCircuit(p.name) for p in providers}

    def _plan(self) -> Tuple[List[LLMProvider], Dict[str, str]]:
        """Usable providers in order, plus why each skipped one was skipped."""
        usable: List[LLMProvider] = []
        skipped: Dict[str, str] = {}
        for provider in self.providers:
            if not provider.is_configured():
                skipped[provider.name] = "not_configured"
            elif not self.circuits[provider.name].allow_request():
                logger.warning("[LLM] circuit OPEN, skipping %s", provider.name)
                skipped[provider.name] = "circuit_open"
            else:
                usable.append(provider)
        return usable, skipped

    def _attribution(
        self,
        actual: LLMProvider,
        attempts: List[Dict[str, str]],
        skipped: Dict[str, str],
    ) -> CallAttribution:
        primary = self.providers[0]
        depth = self.providers.index(actual)
        reason: Optional[str] = None
        if depth:
            ahead = self.providers[0].name
            reason = skipped.get(ahead) or next(
                (a["error"] for a in attempts if a["provider"] == ahead), None
            )
        history = [
            {"provider": name, "outcome": why} for name, why in skipped.items()
        ] + [{"provider": a["provider"], "outcome": a["error"]} for a in attempts]
        history.append({"provider": actual.name, "outcome": "ok"})
        return CallAttribution(
            requested_provider=primary.name,
            requested_model=getattr(primary, "model", None),
            actual_provider=actual.name,
            actual_model=getattr(actual, "model", None),
            fallback_depth=depth,
            fallback_reason=reason,
            attempts=history,
        )

    async def complete(self, messages: Messages) -> LLMResult:
        attempts: List[Dict[str, str]] = []
        candidates, skipped = self._plan()
        for provider in candidates:
            circuit = self.circuits[provider.name]
            try:
                with span("llm", "complete", provider=provider.name):
                    result = await provider.complete(messages)
            except Exception as exc:  # noqa: BLE001
                circuit.record_failure(exc)
                attempts.append(
                    {"provider": provider.name, "error": type(exc).__name__}
                )
                logger.error("[LLM] %s failed: %s", provider.name, exc)
                continue
            circuit.record_success()
            attribution = self._attribution(provider, attempts, skipped)
            result.attribution = attribution
            _last_attribution.set(attribution)
            result.provider = provider.name
            result.model = result.model or attribution.actual_model
            if attribution.fell_back:
                result.fallback_from = attribution.requested_provider
            return result
        raise AllProvidersFailed(
            attempts or [{"provider": "none", "error": "no_available_provider"}]
        )

    async def stream(
        self, messages: Messages, sink: Optional[UsageSink] = None
    ) -> AsyncIterator[str]:
        attempts: List[Dict[str, str]] = []
        candidates, skipped = self._plan()
        for provider in candidates:
            circuit = self.circuits[provider.name]
            yielded = False
            try:
                async for chunk in provider.stream(messages, sink):
                    if not yielded and sink is not None:
                        # Recorded at the first chunk so a mid-stream failure
                        # is still attributed to the provider that spoke.
                        sink.attribution = self._attribution(
                            provider, attempts, skipped
                        )
                        _last_attribution.set(sink.attribution)
                    yielded = True
                    yield chunk
            except Exception as exc:  # noqa: BLE001
                circuit.record_failure(exc)
                attempts.append(
                    {"provider": provider.name, "error": type(exc).__name__}
                )
                logger.error("[LLM] %s stream failed: %s", provider.name, exc)
                if yielded:
                    # Cannot splice a second provider into a half-sent answer.
                    raise
                continue
            circuit.record_success()
            if not yielded and sink is not None:
                sink.attribution = self._attribution(provider, attempts, skipped)
                _last_attribution.set(sink.attribution)
            return
        raise AllProvidersFailed(
            attempts or [{"provider": "none", "error": "no_available_provider"}]
        )

    def health(self) -> Dict[str, Any]:
        return {
            "providers": [
                {**p.health(), **self.circuits[p.name].snapshot()}
                for p in self.providers
            ]
        }


# Circuits are process-wide: outage memory must outlive a single request/pipeline.
_CIRCUITS: Dict[str, ProviderCircuit] = {}


def reset_circuits() -> None:
    _CIRCUITS.clear()


def _shared_circuit(name: str) -> ProviderCircuit:
    if name not in _CIRCUITS:
        _CIRCUITS[name] = ProviderCircuit(name)
    return _CIRCUITS[name]


def build_router(settings: Any) -> ProviderRouter:
    """Primary (``llm_provider``) first, the other as fallback."""
    providers: Dict[str, LLMProvider] = {
        "gemini": GeminiProvider(settings),
        "openrouter": OpenRouterProvider(settings),
    }
    primary = str(getattr(settings, "llm_provider", "openrouter"))
    if primary not in providers:
        raise RuntimeError(f"지원되지 않는 LLM 공급자: {primary}")
    order = [primary] + [n for n in providers if n != primary]
    return ProviderRouter(
        [providers[n] for n in order],
        {n: _shared_circuit(n) for n in order},
    )


def provider_health() -> Dict[str, Any]:
    """Snapshot of every provider circuit seen so far (for /ready, metrics)."""
    return {"circuits": [c.snapshot() for c in _CIRCUITS.values()]}
