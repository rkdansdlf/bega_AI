import asyncio

import httpx
import pytest

from app.core.llm_provider import (
    AllProvidersFailed,
    LLMResult,
    ProviderRouter,
    Usage,
    UsageSink,
    usage_from_gemini,
    usage_from_openai,
)
from app.core.provider_circuit import (
    CircuitState,
    ProviderCircuit,
    is_availability_failure,
)


def _status_error(code):
    req = httpx.Request("POST", "http://x")
    return httpx.HTTPStatusError(
        "e", request=req, response=httpx.Response(code, request=req)
    )


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeProvider:
    def __init__(self, name, *, configured=True, error=None, text="ok", chunks=None):
        self.name = name
        self._configured = configured
        self._error = error
        self._text = text
        self._chunks = chunks if chunks is not None else ["a", "b"]
        self.calls = 0

    def is_configured(self):
        return self._configured

    async def complete(self, messages):
        self.calls += 1
        if self._error:
            raise self._error
        return LLMResult(self._text, self.name)

    async def stream(self, messages, sink=None):
        self.calls += 1
        if self._error:
            raise self._error
        for c in self._chunks:
            yield c

    def health(self):
        return {"provider": self.name, "configured": self._configured}


def _run(coro):
    return asyncio.run(coro)


# --- circuit breaker --------------------------------------------------------
def _circuit(clock, **kw):
    return ProviderCircuit(
        "p",
        clock=clock,
        min_requests=4,
        failure_rate_threshold=0.5,
        cooldown_seconds=30,
        **kw
    )


def test_circuit_opens_on_failure_rate_then_half_open_then_closes():
    clock = FakeClock()
    c = _circuit(clock)
    for _ in range(4):
        c.record_failure(_status_error(503))
    assert c.state is CircuitState.OPEN
    assert c.allow_request() is False
    clock.t = 31
    assert c.state is CircuitState.HALF_OPEN
    assert c.allow_request() is True  # single probe
    assert c.allow_request() is False
    c.record_success()
    assert c.state is CircuitState.CLOSED


def test_half_open_failure_reopens():
    clock = FakeClock()
    c = _circuit(clock)
    for _ in range(4):
        c.record_failure(_status_error(429))
    clock.t = 31
    assert c.allow_request()
    c.record_failure(_status_error(500))
    assert c.state is CircuitState.OPEN


def test_client_errors_do_not_trip_circuit():
    c = _circuit(FakeClock())
    for _ in range(10):
        c.record_failure(_status_error(401))
    assert c.state is CircuitState.CLOSED
    assert is_availability_failure(httpx.ReadTimeout("t"))
    assert not is_availability_failure(ValueError("x"))


def test_old_failures_age_out_of_window():
    clock = FakeClock()
    c = _circuit(clock)
    for _ in range(3):
        c.record_failure(_status_error(503))
    clock.t = 120
    c.record_failure(_status_error(503))
    assert c.state is CircuitState.CLOSED  # only 1 event left in window


# --- router -----------------------------------------------------------------
def test_router_falls_back_and_marks_origin():
    primary = FakeProvider("gemini", error=_status_error(503))
    fallback = FakeProvider("openrouter", text="from-fallback")
    r = ProviderRouter([primary, fallback])
    out = _run(r.complete([]))
    assert out.text == "from-fallback"
    assert out.fallback_from == "gemini"


def test_router_skips_unconfigured_and_open_circuit_providers():
    clock = FakeClock()
    bad = FakeProvider("gemini", error=_status_error(503))
    good = FakeProvider("openrouter")
    circuits = {
        "gemini": ProviderCircuit(
            "gemini", clock=clock, min_requests=2, cooldown_seconds=30
        ),
        "openrouter": ProviderCircuit("openrouter", clock=clock),
    }
    r = ProviderRouter([bad, good], circuits)
    for _ in range(2):
        _run(r.complete([]))
    assert bad.calls == 2
    _run(r.complete([]))
    assert bad.calls == 2  # circuit OPEN: no more calls to the failing provider
    unconfigured = ProviderRouter([FakeProvider("gemini", configured=False)])
    with pytest.raises(AllProvidersFailed):
        _run(unconfigured.complete([]))


def test_router_stream_fails_over_before_first_chunk_only():
    async def collect(router):
        return [c async for c in router.stream([])]

    ok = ProviderRouter(
        [FakeProvider("gemini", error=_status_error(503)), FakeProvider("openrouter")]
    )
    assert _run(collect(ok)) == ["a", "b"]

    class Half(FakeProvider):
        async def stream(self, messages, sink=None):
            yield "partial"
            raise _status_error(503)

    half = ProviderRouter([Half("gemini"), FakeProvider("openrouter")])

    async def run_half():
        got = []
        with pytest.raises(httpx.HTTPStatusError):
            async for c in half.stream([]):
                got.append(c)
        return got

    assert _run(run_half()) == ["partial"]  # never splices a second provider


# --- usage ------------------------------------------------------------------
def test_usage_extraction_prefers_provider_numbers():
    u = usage_from_openai({"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (10, 5, 15)
    assert u.source == "provider" and u.is_reported
    g = usage_from_gemini(
        {
            "usageMetadata": {
                "promptTokenCount": 7,
                "candidatesTokenCount": 3,
                "totalTokenCount": 10,
            }
        }
    )
    assert g.total_tokens == 10 and g.is_reported
    assert usage_from_openai({}).source == "estimate"
    assert usage_from_gemini({}).source == "estimate"
    assert UsageSink().usage == Usage()


# --- real transports over a mock HTTP layer ---------------------------------
from types import SimpleNamespace  # noqa: E402

from app.core import llm_provider as lp  # noqa: E402

_OR = SimpleNamespace(
    openrouter_api_key="k",
    openrouter_referer="",
    openrouter_app_title="",
    openrouter_model="m",
    openrouter_base_url="https://or.test/v1",
    max_output_tokens=32,
)
_GM = SimpleNamespace(gemini_api_key="k", gemini_model="g", max_output_tokens=32)


def _mock_client(monkeypatch, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(lp, "get_shared_httpx_client", lambda *a, **k: client)
    return client


def test_openrouter_complete_reports_provider_usage(monkeypatch):
    def handler(request):
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "답변"}}],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 4,
                    "total_tokens": 15,
                },
            },
        )

    _mock_client(monkeypatch, handler)
    out = _run(lp.OpenRouterProvider(_OR).complete([{"role": "user", "content": "q"}]))
    assert out.text == "답변" and out.usage.total_tokens == 15 and out.usage.is_reported


def test_gemini_empty_response_raises_so_router_can_fail_over(monkeypatch):
    _mock_client(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"candidates": [{"content": {"parts": []}}]}
        ),
    )
    with pytest.raises(RuntimeError):
        _run(lp.GeminiProvider(_GM).complete([{"role": "user", "content": "q"}]))


def test_openrouter_stream_collects_usage_from_final_chunk(monkeypatch):
    body = (
        'data: {"choices":[{"delta":{"content":"안"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"녕"}}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    _mock_client(monkeypatch, lambda request: httpx.Response(200, text=body))
    sink = UsageSink()

    async def go():
        return [c async for c in lp.OpenRouterProvider(_OR).stream([], sink)]

    assert _run(go()) == ["안", "녕"]
    assert sink.usage.total_tokens == 5 and sink.usage.is_reported


def test_gemini_message_conversion_folds_system_into_first_user():
    contents = lp._to_gemini_contents(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
    )
    assert contents[0]["role"] == "user" and "SYS" in contents[0]["parts"][0]["text"]
    assert contents[1]["role"] == "model"


# --- actual provider/model attribution --------------------------------------
from app.core.fingerprint import build_response_fingerprint  # noqa: E402
from app.core.llm_provider import get_call_attribution  # noqa: E402


class ModelProvider(FakeProvider):
    def __init__(self, name, model, **kw):
        super().__init__(name, **kw)
        self.model = model


def _router(primary, fallback, clock=None):
    kw = {"clock": clock} if clock else {}
    return ProviderRouter(
        [primary, fallback],
        {
            primary.name: ProviderCircuit(primary.name, min_requests=2, **kw),
            fallback.name: ProviderCircuit(fallback.name, **kw),
        },
    )


def test_primary_success_is_depth_zero_with_requested_equals_actual():
    r = _router(ModelProvider("gemini", "g-1"), ModelProvider("openrouter", "or-1"))
    out = _run(r.complete([]))
    a = out.attribution
    assert (a.requested_provider, a.actual_provider) == ("gemini", "gemini")
    assert (a.requested_model, a.actual_model) == ("g-1", "g-1")
    assert a.fallback_depth == 0 and a.fallback_reason is None and not a.fell_back
    assert out.fallback_from is None


def test_fallback_records_actual_provider_model_and_reason():
    r = _router(
        ModelProvider("gemini", "g-1", error=_status_error(503)),
        ModelProvider("openrouter", "or-deepseek"),
    )
    out = _run(r.complete([]))
    a = out.attribution
    assert a.requested_provider == "gemini" and a.requested_model == "g-1"
    assert a.actual_provider == "openrouter" and a.actual_model == "or-deepseek"
    assert a.fallback_depth == 1 and a.fell_back
    assert a.fallback_reason == "HTTPStatusError"
    assert out.provider == "openrouter" and out.fallback_from == "gemini"
    assert [h["outcome"] for h in a.attempts] == ["HTTPStatusError", "ok"]


def test_skipped_providers_are_explained_by_circuit_or_config():
    unconfigured = _router(
        ModelProvider("gemini", "g-1", configured=False),
        ModelProvider("openrouter", "or-1"),
    )
    a = _run(unconfigured.complete([])).attribution
    assert a.fallback_depth == 1 and a.fallback_reason == "not_configured"

    clock = FakeClock()
    bad = ModelProvider("gemini", "g-1", error=_status_error(503))
    r = _router(bad, ModelProvider("openrouter", "or-1"), clock)
    for _ in range(2):
        _run(r.complete([]))
    a = _run(r.complete([])).attribution
    assert a.fallback_reason == "circuit_open" and bad.calls == 2


def test_stream_attribution_reports_fallback_provider():
    r = _router(
        ModelProvider("gemini", "g-1", error=_status_error(429)),
        ModelProvider("openrouter", "or-1"),
    )
    sink = UsageSink()

    async def go():
        return [c async for c in r.stream([], sink)]

    assert _run(go()) == ["a", "b"]
    assert sink.attribution.actual_provider == "openrouter"
    assert sink.attribution.actual_model == "or-1"
    assert sink.attribution.fallback_depth == 1


def test_mid_stream_failure_is_still_attributed_to_the_provider_that_spoke():
    class Half(ModelProvider):
        async def stream(self, messages, sink=None):
            yield "partial"
            raise _status_error(503)

    r = _router(Half("gemini", "g-1"), ModelProvider("openrouter", "or-1"))
    sink = UsageSink()

    async def go():
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in r.stream([], sink):
                pass

    _run(go())
    assert sink.attribution.actual_provider == "gemini"


def test_attribution_is_visible_through_the_request_context():
    r = _router(
        ModelProvider("gemini", "g-1", error=_status_error(503)),
        ModelProvider("openrouter", "or-1"),
    )

    async def go():
        await r.complete([])
        return get_call_attribution()

    assert _run(go()).actual_provider == "openrouter"


def test_fingerprint_uses_actual_model_not_configured_provider():
    settings = SimpleNamespace(
        llm_provider="gemini",
        gemini_model="configured-gemini",
        rag_rerank_enabled=False,
        embed_provider="openrouter",
        embed_dim=1536,
    )
    attribution = {
        "requested_provider": "gemini",
        "actual_provider": "openrouter",
        "actual_model": "deepseek/x",
        "fallback_depth": 1,
        "fallback_reason": "HTTPStatusError",
    }
    fp = build_response_fingerprint(settings, {"llm_attribution": attribution})
    assert fp["model"] == "deepseek/x"
    assert fp["llm_actual_provider"] == "openrouter"
    assert fp["llm_requested_provider"] == "gemini"
    assert fp["llm_fallback_depth"] == 1
    assert fp["llm_fallback_reason"] == "HTTPStatusError"
    plain = build_response_fingerprint(settings, {})
    assert plain["model"] == "configured-gemini" and plain["llm_fallback_depth"] == 0


def test_rag_stream_accounting_uses_actual_provider_after_fallback():
    from app.core.rag import RAGPipeline, _attribution_dict

    router = _router(
        ModelProvider("gemini", "g-1", error=_status_error(503)),
        ModelProvider("openrouter", "or-deepseek"),
    )
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = SimpleNamespace(llm_provider="gemini")
    pipeline._llm_router = lambda: router
    captured = {}

    def fake_account(result, messages):
        captured["result"] = result
        return {}

    pipeline._account_usage = fake_account

    async def go():
        chunks = [c async for c in pipeline._generate_stream([])]
        return chunks, _attribution_dict()

    chunks, attribution = _run(go())
    assert chunks == ["a", "b"]
    assert captured["result"].provider == "openrouter"  # not LLM_PROVIDER
    assert captured["result"].model == "or-deepseek"
    assert attribution["actual_provider"] == "openrouter"
    assert attribution["requested_provider"] == "gemini"
