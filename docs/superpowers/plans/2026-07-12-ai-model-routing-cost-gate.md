# AI Model Routing Cost Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible request-level planner/answer cost evidence and a golden-60 release gate that requires at least 20% planner savings without total-cost or quality regression.

**Architecture:** A pure pricing and usage module calculates deterministic token and `Decimal` cost estimates. The existing request context collects attempt-level observations emitted by the production LLM generators; API metadata, Prometheus, and the experiment report consume the same records. The experiment runs the immutable internal-only golden set once per baseline/candidate and fails closed on incomplete evidence.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic Settings v2, `prometheus_client`, `httpx`, pytest.

## Global Constraints

- Work from `origin/main` revision `c1bae4fa2aead216b81126bfd5f9469ff53a040d` in the isolated worktree.
- Do not use CodeGraph; inspect files directly with `rg`, `sed`, and normal Git commands.
- Do not add external baseball crawling, scraping, web-search repair, or external baseball API calls.
- Missing or inconsistent baseball data must preserve `MANUAL_BASEBALL_DATA_REQUIRED`.
- Do not log pricing JSON, prompts, answers, tool payloads, credentials, headers, or user identifiers.
- Do not change model selection, prompts, routing rules, cache serving flags, or answer behavior.
- `CHAT_MODEL_PRICING_JSON` is optional for normal serving; malformed non-empty configuration fails startup.
- Release-gate evidence accepts only exact provider/model catalog pricing, explicit planner/answer model labels, cache bypass, and the immutable 60-case input hash.
- Cost arithmetic uses `Decimal`; reports serialize USD as fixed decimal strings.
- Exit code `0` means pass, `1` means valid measured failure, and `2` means invalid/incomplete evidence.
- No package installation, live model call, deployment, or external service call without separate approval.

---

## File Map

- Create `app/core/chat_model_usage.py`: pure pricing schema, token estimator, usage record, and serialization.
- Modify `app/config.py`: optional pricing JSON and startup validation.
- Create `tests/test_chat_model_usage.py`: pure catalog and estimator tests.
- Modify `app/agents/runtime_factory.py`: emit one observation per actual provider attempt.
- Modify `tests/test_llm_model_candidates.py`: observer coverage for success, retry, fallback, and failure.
- Modify `app/agents/baseball_agent.py`: request-scoped collector, model-role binding, and metadata reference.
- Modify `tests/test_baseball_agent_llm_planner.py`: planner/answer collection and retry aggregation tests.
- Modify `app/observability/metrics.py`: bounded model-usage counters.
- Modify `app/core/chat_cost_metrics.py`: emit counters from shared usage records while preserving legacy metrics.
- Modify `tests/test_chat_cost_metrics.py` and `tests/test_observability_metrics.py`: counter contracts.
- Modify `app/routers/chat_stream.py`: completion, stream, static, and cache metadata contracts.
- Modify `tests/test_chat_api_smoke.py`: sanitized `model_usage` and completion marker tests.
- Add `scripts/chat_quality_golden_60.json`: immutable operator-provided 60-case source asset.
- Create `tests/test_chat_quality_golden_dataset.py`: dataset policy, schema, categories, uniqueness, and hash.
- Modify `scripts/chat_model_routing_experiment.py`: golden execution, report generation, comparison, and exit codes.
- Create `tests/test_chat_model_routing_experiment.py`: pure report/gate and HTTP-result tests.
- Modify `.env.example`: pricing schema example without real prices or secrets.
- Modify `docs/ai-optimization-rollout-runbook.md`: baseline/candidate commands and decision rules.
- Modify `.github/workflows/ai-pr-gate.yml`: include deterministic new tests.

---

### Task 1: Pure Pricing Catalog And Usage Estimator

**Files:**
- Create: `app/core/chat_model_usage.py`
- Modify: `app/config.py:215-251,636-644`
- Create: `tests/test_chat_model_usage.py`
- Modify: `tests/test_config_internal_token.py:260-275`

**Interfaces:**
- Produces: `ModelPrice`, `ModelPricingCatalog`, `ModelUsageEstimate`, `estimate_model_usage()`, `serialize_messages()`, and `format_usd()`.
- Produces: `Settings.chat_model_pricing_json: Optional[str]` and validated catalog parsing.
- Consumes: no application service or metrics module; this task stays pure.

- [ ] **Step 1: Write failing catalog and estimator tests**

Add tests with these exact assertions:

```python
from decimal import Decimal

import pytest

from app.core.chat_model_usage import ModelPricingCatalog, estimate_model_usage


PRICING_JSON = """{
  "openrouter": {
    "vendor/planner": {
      "input_usd_per_1m_tokens": "1.00",
      "output_usd_per_1m_tokens": "2.00"
    }
  }
}"""


def test_catalog_uses_exact_provider_and_model_lookup() -> None:
    catalog = ModelPricingCatalog.from_json(PRICING_JSON)
    price = catalog.lookup("openrouter", "vendor/planner")

    assert price is not None
    assert price.input_usd_per_1m_tokens == Decimal("1.00")
    assert catalog.lookup("openrouter", "vendor/other") is None


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        "[]",
        '{"openrouter":{"vendor/planner":{"input_usd_per_1m_tokens":"-1","output_usd_per_1m_tokens":"2"}}}',
        '{"openrouter":{"vendor/planner":{"input_usd_per_1m_tokens":"NaN","output_usd_per_1m_tokens":"2"}}}',
        '{"openrouter":{"vendor/planner":{"input_usd_per_1m_tokens":"1"}}}',
    ],
)
def test_catalog_rejects_invalid_pricing(raw: str) -> None:
    with pytest.raises(ValueError):
        ModelPricingCatalog.from_json(raw)


def test_usage_estimate_is_deterministic_and_sanitized() -> None:
    catalog = ModelPricingCatalog.from_json(PRICING_JSON)
    record = estimate_model_usage(
        catalog,
        role="planner",
        provider="openrouter",
        model="vendor/planner",
        messages=[{"role": "user", "content": "abcd"}],
        output_text="1234567",
        outcome="success",
    )

    assert record.input_tokens > 0
    assert record.output_tokens == 2
    assert record.pricing_source == "model_catalog"
    assert record.total_cost_usd == (
        Decimal(record.input_tokens) / Decimal("1000000")
        + Decimal("0.000004")
    )
    payload = record.to_dict()
    assert payload["role"] == "planner"
    assert payload["total_cost_usd"] == format(record.total_cost_usd, ".12f")
    assert "abcd" not in str(payload)
    assert "1234567" not in str(payload)
```

Add config tests that set a valid `CHAT_MODEL_PRICING_JSON`, then assert invalid non-empty JSON raises `ValueError` containing `CHAT_MODEL_PRICING_JSON`.

- [ ] **Step 2: Run tests and confirm the expected import/config failures**

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_chat_model_usage.py \
  tests/test_config_internal_token.py::test_chat_model_pricing_json_is_validated -q
```

Expected: FAIL because `app.core.chat_model_usage` and the setting do not exist.

- [ ] **Step 3: Implement the pure model and config validator**

Create these public shapes in `app/core/chat_model_usage.py`:

```python
ModelRole = Literal["planner", "answer"]
ModelCallOutcome = Literal["success", "failed"]
PricingSource = Literal["model_catalog", "unpriced"]


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_usd_per_1m_tokens: Decimal
    output_usd_per_1m_tokens: Decimal


@dataclass(frozen=True, slots=True)
class ModelUsageEstimate:
    role: ModelRole
    provider: str
    model: str
    outcome: ModelCallOutcome
    pricing_source: PricingSource
    input_chars: int
    output_chars: int
    input_tokens: int
    output_tokens: int
    input_cost_usd: Decimal | None
    output_cost_usd: Decimal | None
    total_cost_usd: Decimal | None

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "outcome": self.outcome,
            "pricing_source": self.pricing_source,
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost_usd": format_usd(self.input_cost_usd),
            "output_cost_usd": format_usd(self.output_cost_usd),
            "total_cost_usd": format_usd(self.total_cost_usd),
        }


@dataclass(frozen=True, slots=True)
class ModelPricingCatalog:
    prices: Mapping[tuple[str, str], ModelPrice]

    @classmethod
    def from_json(cls, raw: str | None) -> "ModelPricingCatalog":
        if raw is None or not raw.strip():
            return cls(prices={})
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("pricing root must be an object")
        return cls(prices=MappingProxyType(_parse_price_entries(parsed)))

    def lookup(self, provider: str, model: str) -> ModelPrice | None:
        return self.prices.get((provider.strip(), model.strip()))
```

Implement `_parse_price_entries()` with object/type checks, exact key checks,
string-only price values, `Decimal(value)`, `is_finite()`, and non-negative
validation. Keep
`_CHARS_PER_TOKEN = Decimal("3.5")`. Implement `serialize_messages()` as compact
stable JSON using `json.dumps(messages, ensure_ascii=False, sort_keys=True,
separators=(",", ":"))`. Implement `estimate_model_usage()` by counting
serialized input and generated output characters, applying `ceil(chars / 3.5)`,
exact catalog lookup, and `Decimal(tokens) * rate / Decimal("1000000")`.

In `app/config.py`, add:

```python
chat_model_pricing_json: Optional[str] = Field(
    None, validation_alias="CHAT_MODEL_PRICING_JSON"
)

@field_validator("chat_model_pricing_json")
def _validate_chat_model_pricing_json(cls, value: Optional[str]) -> Optional[str]:
    try:
        ModelPricingCatalog.from_json(value)
    except ValueError as exc:
        raise ValueError(f"CHAT_MODEL_PRICING_JSON is invalid: {exc}") from exc
    return value
```

- [ ] **Step 4: Run focused tests**

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_chat_model_usage.py \
  tests/test_config_internal_token.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the pure pricing unit**

```bash
git add app/core/chat_model_usage.py app/config.py \
  tests/test_chat_model_usage.py tests/test_config_internal_token.py
git commit -m "feat(ai): add model pricing catalog"
```

---

### Task 2: Attempt-Level Provider Observations

**Files:**
- Modify: `app/agents/runtime_factory.py:120-335`
- Modify: `tests/test_llm_model_candidates.py`

**Interfaces:**
- Consumes: `ModelCallOutcome` from Task 1.
- Produces: optional generator keyword `usage_observer` receiving keyword-only `provider`, `model`, `messages`, `output_text`, and `outcome`.
- Guarantees: one callback for every OpenRouter model/retry attempt and one for each Gemini generation attempt.

- [ ] **Step 1: Add failing observer tests**

Extend the existing fake OpenRouter tests:

```python
observations: list[dict[str, object]] = []

def observe(**payload: object) -> None:
    observations.append(dict(payload))

chunks = [
    chunk
    async for chunk in generator(
        [{"role": "user", "content": "hello"}],
        model_override="openrouter/cheap-planner",
        usage_observer=observe,
    )
]

assert chunks == ["override response"]
assert observations == [
    {
        "provider": "openrouter",
        "model": "openrouter/cheap-planner",
        "messages": [{"role": "user", "content": "hello"}],
        "output_text": "override response",
        "outcome": "success",
    }
]
```

Add a test where the first attempt returns empty choices and the second succeeds.
Assert two observations in order: failed empty output, then successful output.
Extend the existing 429 fallback test to assert a failed primary observation
followed by a successful fallback-model observation.

- [ ] **Step 2: Run observer tests and verify failure**

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_llm_model_candidates.py::test_openrouter_generator_uses_model_override_without_fallback \
  tests/test_llm_model_candidates.py::test_openrouter_429_falls_back_to_gpt_oss -q
```

Expected: FAIL because `usage_observer` is not accepted or invoked.

- [ ] **Step 3: Add attempt notification to both providers**

Add this local helper without logging payload content:

```python
def _notify_usage_observer(
    observer,
    *,
    provider: str,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    output_text: str,
    outcome: str,
) -> None:
    if observer is None:
        return
    observer(
        provider=provider,
        model=model,
        messages=messages,
        output_text=output_text,
        outcome=outcome,
    )
```

Change both generated functions to accept `usage_observer=None`. For every
OpenRouter retry/model loop, collect `attempt_chunks`; notify `failed` before an
empty retry or exception and `success` before returning. For Gemini, collect
chunks and notify in `try/except/else`. Preserve current yielded text, retry,
fallback metrics, and exceptions.

- [ ] **Step 4: Run all runtime generator tests**

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_llm_model_candidates.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit provider observation support**

```bash
git add app/agents/runtime_factory.py tests/test_llm_model_candidates.py
git commit -m "feat(ai): observe routed model attempts"
```

---

### Task 3: Request-Scoped Agent Usage Collection

**Files:**
- Modify: `app/agents/baseball_agent.py:1-120,130-260,4069-4090,6622-6685,9190-9215,11890-11910`
- Modify: `app/agents/runtime_factory.py:340-380`
- Modify: `tests/test_baseball_agent_llm_planner.py`

**Interfaces:**
- Consumes: `ModelPricingCatalog.from_json()`, `ModelUsageEstimate`, and `estimate_model_usage()` from Task 1; provider observations from Task 2.
- Produces: `AgentRequestContext.model_usage_records: list[ModelUsageEstimate]`.
- Produces: `_call_llm_stream(messages, *, max_tokens, model_override, usage_role)` and live `model_usage` metadata list.

- [ ] **Step 1: Write failing request-scope and retry tests**

Add a test that enters `runtime.request_context(fake_connection)`, invokes the
planner with a fake generator accepting `usage_observer`, and asserts one
planner record. Add an answer-retry test that triggers two answer generator
invocations and asserts both answer records remain in order:

```python
assert [record.role for record in request_context.model_usage_records] == [
    "planner",
    "answer",
    "answer",
]
assert all(
    record.pricing_source == "model_catalog"
    for record in request_context.model_usage_records
)
```

- [ ] **Step 2: Run the two new test node IDs**

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_baseball_agent_llm_planner.py::test_model_usage_collects_planner_and_answer_attempts \
  tests/test_baseball_agent_llm_planner.py::test_model_usage_keeps_answer_retry_costs -q
```

Expected: FAIL because the context has no usage records and `_call_llm_stream`
has no role binding.

- [ ] **Step 3: Add the collector and role-bound observer**

Import `field` from `dataclasses`, then extend the context:

```python
@dataclass(slots=True)
class AgentRequestContext:
    runtime_id: int
    connection: psycopg.AsyncConnection
    db_query_tool: DatabaseQueryTool
    regulation_query_tool: RegulationQueryTool
    game_query_tool: GameQueryTool
    document_query_tool: DocumentQueryTool
    game_strategist: Any
    match_predictor: Any
    model_usage_records: list[ModelUsageEstimate] = field(default_factory=list)
```

Parse the catalog once in `BaseballAgentRuntime.__init__` and expose it to the
shared agent. Add `BaseballStatisticsAgent._observe_model_attempt()` that calls
`estimate_model_usage()`, appends to the active request context when present,
without emitting metrics yet; Task 4 adds the metrics call after defining that
interface.

Change `_call_llm_stream` to require `usage_role`, bind it into the observer,
and pass `usage_observer` only when the generator signature supports it.
Preserve compatibility with simple test generators by wrapping their returned
async iterator and emitting one configured-model observation at completion.

Pass `usage_role="planner"` at `_analyze_query_with_llm` and
`usage_role="answer"` at `_generate_verified_answer`.

In `_build_metadata`, expose the live list reference without copying it:

```python
request_context = self._maybe_current_request_context()
model_usage_records = (
    request_context.model_usage_records if request_context is not None else []
)
```

Add `"model_usage": model_usage_records` to the existing returned metadata and
preserve every current key.

- [ ] **Step 4: Run planner/answer focused tests**

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_baseball_agent_llm_planner.py \
  tests/test_llm_model_candidates.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit request-scoped collection**

```bash
git add app/agents/baseball_agent.py app/agents/runtime_factory.py \
  tests/test_baseball_agent_llm_planner.py tests/test_llm_model_candidates.py
git commit -m "feat(ai): collect request model usage"
```

---

### Task 4: Metrics And Internal API Metadata

**Files:**
- Modify: `app/observability/metrics.py:110-150`
- Modify: `app/core/chat_cost_metrics.py`
- Modify: `app/agents/baseball_agent.py`
- Modify: `app/routers/chat_stream.py:671-700,933-975,1250-1305,1500-1560,1859-2090`
- Modify: `tests/test_chat_cost_metrics.py`
- Modify: `tests/test_observability_metrics.py`
- Modify: `tests/test_chat_api_smoke.py`

**Interfaces:**
- Consumes: `ModelUsageEstimate` records from Task 3.
- Produces: `record_model_usage_estimate(record) -> None`.
- Produces: response fields `model_usage: list[dict]` and `model_usage_complete: bool`.

- [ ] **Step 1: Write failing metric and API contract tests**

Add recorder-based tests asserting a priced successful planner record emits:

```python
assert token_counter.calls[0][0] == {
    "role": "planner",
    "provider": "openrouter",
    "model": "vendor/planner",
    "token_type": "input",
    "outcome": "success",
}
assert outcome_counter.calls[0][0]["result"] == "priced"
assert cost_counter.calls[0][1] == pytest.approx(float(record.total_cost_usd))
```

Assert an unpriced record emits token/outcome but no cost. Assert a failed
record uses `result="failed"`. Update `_FakeAgent` in
`tests/test_chat_api_smoke.py` to return a fake usage object with `to_dict()`.
Assert generated completion returns sanitized records and
`model_usage_complete is True`; exact and semantic cache tests must assert an
empty usage list and completion true.

- [ ] **Step 2: Run focused tests and confirm missing counters/fields**

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_chat_cost_metrics.py \
  tests/test_observability_metrics.py::test_metric_objects_accept_labels_and_observations \
  tests/test_chat_api_smoke.py -q
```

Expected: FAIL on missing counters and response fields.

- [ ] **Step 3: Implement bounded counters and final metadata**

Add counters without altering existing label schemas:

```python
AI_MODEL_USAGE_TOKEN_ESTIMATE_TOTAL = Counter(
    "ai_model_usage_token_estimate_total",
    "Estimated model-call tokens by role and routed model.",
    ["role", "provider", "model", "token_type", "outcome"],
)
AI_MODEL_USAGE_COST_ESTIMATE_USD_TOTAL = Counter(
    "ai_model_usage_cost_estimate_usd_total",
    "Catalog-priced estimated model-call cost in USD.",
    ["role", "provider", "model"],
)
AI_MODEL_USAGE_OUTCOME_TOTAL = Counter(
    "ai_model_usage_outcome_total",
    "Model usage pricing and call outcomes.",
    ["role", "provider", "model", "result"],
)
```

Implement `record_model_usage_estimate()` to emit both token directions, one
outcome, and cost for every `model_catalog` record with non-null cost, including
failed attempts with observable input or partial output. Bound provider/model
values before labels. Preserve
`record_chat_token_estimate()` unchanged for existing dashboards.

Call the new function from the agent observation method. Add `model_usage=[]`
to cached/static payload builders. Completion sets
`model_usage_complete=True` only after fully consuming the answer generator.
Stream final metadata copies `result.get("model_usage", [])` after stream
consumption and sets completion false on cancellation, true otherwise. Serialize
through existing `_safe_serialize()` and `to_dict()`.

- [ ] **Step 4: Run metrics and route tests**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit observable API evidence**

```bash
git add app/observability/metrics.py app/core/chat_cost_metrics.py \
  app/agents/baseball_agent.py app/routers/chat_stream.py \
  tests/test_chat_cost_metrics.py tests/test_observability_metrics.py \
  tests/test_chat_api_smoke.py
git commit -m "feat(ai): expose routed model usage evidence"
```

---

### Task 5: Promote And Lock The Operator Golden 60

**Files:**
- Create: `scripts/chat_quality_golden_60.json`
- Create: `tests/test_chat_quality_golden_dataset.py`

**Interfaces:**
- Produces: immutable schema-version-1 dataset with `cases` consumed by Task 6.
- Requires SHA-256 `8899b468d65a87a505083a5f229808a0b9b405819287f00ff707741715bb124d`.

- [ ] **Step 1: Verify the operator source before adding it**

```bash
shasum -a 256 /Users/mac/project/KBO_platform/bega_AI/scripts/chat_quality_golden_60.json
jq '{count:(.cases|length), policy:.baseball_data_policy}' \
  /Users/mac/project/KBO_platform/bega_AI/scripts/chat_quality_golden_60.json
```

Expected: exact approved hash, `count=60`, and `policy="internal_only"`.
Stop with `MANUAL_BASEBALL_DATA_REQUIRED` if the source is absent or the hash
differs; do not reconstruct or edit the cases.

- [ ] **Step 2: Add the immutable file and contract test**

Add the verified file verbatim using `apply_patch`; do not change ordering,
questions, expectations, or whitespace. Add tests asserting:

```python
assert dataset["schema_version"] == 1
assert dataset["baseball_data_policy"] == "internal_only"
assert len(dataset["cases"]) == 60
assert Counter(case["category"] for case in dataset["cases"]) == {
    "multi_player_narrative": 20,
    "regulation": 20,
    "team_db": 14,
    "recovered_regression": 6,
}
assert hashlib.sha256(GOLDEN_DATASET_PATH.read_bytes()).hexdigest() == (
    "8899b468d65a87a505083a5f229808a0b9b405819287f00ff707741715bb124d"
)
```

Also assert IDs/questions are unique, no question contains `http://` or
`https://`, every allowed planner-mode list is non-empty, and expected
answerability belongs to the four approved values.

- [ ] **Step 3: Run dataset and policy tests**

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_chat_quality_golden_dataset.py -q
python3 scripts/validate_baseball_data_policy.py
```

Expected: PASS and `External baseball data policy OK`.

- [ ] **Step 4: Commit the attributable operator asset**

```bash
git add scripts/chat_quality_golden_60.json tests/test_chat_quality_golden_dataset.py
git commit -m "test(ai): add internal model routing golden set"
```

---

### Task 6: Golden Runner And Baseline/Candidate Comparator

**Files:**
- Modify: `scripts/chat_model_routing_experiment.py`
- Create: `tests/test_chat_model_routing_experiment.py`
- Modify: `tests/test_ai_optimization_ops_scripts.py`

**Interfaces:**
- Consumes: Task 4 response metadata and Task 5 golden schema.
- Produces: `load_golden_cases()`, `evaluate_case()`, `build_run_report()`, `compare_reports()`, and stable CLI exit codes.
- Reuses: `_evaluate_quality`, `_quality_pass`, and `_evaluate_answerability` from `scripts.smoke_chatbot_quality`.

- [ ] **Step 1: Write failing pure gate tests**

Build minimal 60-case reports with explicit labels and priced usage. Assert:

```python
comparison = compare_reports(baseline, candidate)
assert comparison["gate_passed"] is True
assert comparison["planner_reduction_percent"] == "20.00"
assert comparison["candidate_total_cost_non_increasing"] is True
```

Use baseline planner cost `1.000000000000`, candidate planner cost
`0.800000000000`, and equal answer cost. Add parameterized failures for:

- `19.99` percent reduction
- candidate total greater than baseline
- zero baseline planner cost
- changed answer label
- `default` model label
- dataset hash or ordered ID mismatch
- duplicate/missing case
- successful response with `model_usage_complete=false`
- successful LLM planner case without planner usage
- unpriced usage
- unexpected planner mode or fallback

Assert structural problems raise `InvalidEvidenceError` and map to exit `2`;
measured gate failures return exit `1`.

- [ ] **Step 2: Run pure tests and verify failure**

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_chat_model_routing_experiment.py -q
```

Expected: FAIL on missing golden/report/comparison interfaces.

- [ ] **Step 3: Implement golden loading and single-run evaluation**

Define:

```python
class InvalidEvidenceError(ValueError):
    pass


def load_golden_cases(path: Path) -> tuple[str, list[dict[str, Any]]]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    cases = payload.get("cases")
    if payload.get("schema_version") != 1 or not isinstance(cases, list):
        raise InvalidEvidenceError("unsupported golden dataset")
    return hashlib.sha256(raw).hexdigest(), cases
```

Run exactly one cache-bypassed completion request for each case. Continue after
HTTP failures so the report has 60 ordered results. Store no answer text. For
successful responses, evaluate natural chat, answerability, allowed planner
modes, fallback flags, usage completeness, and catalog pricing. Include dataset
hash, ordered case IDs, explicit model labels, cache bypass, latency summary,
per-role cost totals, and sanitized case details.

- [ ] **Step 4: Implement comparison and exit mapping**

Use `Decimal` for report costs:

```python
planner_reduction = (
    (baseline_planner - candidate_planner)
    / baseline_planner
    * Decimal("100")
)
planner_pass = planner_reduction >= Decimal("20.00")
total_pass = candidate_total <= baseline_total
gate_passed = planner_pass and total_pass and quality_passed
```

Validate structure and comparability before arithmetic. A report with complete
60-case HTTP/model failures is a measured failure (`1`); a successful case with
missing/unknown usage is invalid evidence (`2`). Default `--limit` to 60 and
require 60 for comparison. Default cache bypass to true and reject a comparison
report generated without it.

CLI arguments remain or add these exact names:

```text
--samples
--base-url
--output
--baseline-report
--planner-model-label
--answer-model-label
--internal-api-key-env
--timeout
--limit
```

- [ ] **Step 5: Run script tests and legacy operation tests**

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_chat_model_routing_experiment.py \
  tests/test_ai_optimization_ops_scripts.py \
  tests/test_chat_quality_golden_dataset.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the executable cost gate**

```bash
git add scripts/chat_model_routing_experiment.py \
  tests/test_chat_model_routing_experiment.py \
  tests/test_ai_optimization_ops_scripts.py
git commit -m "feat(ai): gate model routing cost and quality"
```

---

### Task 7: Operator Contract, CI, And Full Verification

**Files:**
- Modify: `.env.example`
- Modify: `docs/ai-optimization-rollout-runbook.md:20-55,75-85,125-165`
- Modify: `.github/workflows/ai-pr-gate.yml`
- Test: all files changed by Tasks 1-6

**Interfaces:**
- Consumes: all completed instrumentation and gate commands.
- Produces: attributable operator procedure and deterministic CI coverage.

- [ ] **Step 1: Add a failing documentation/workflow contract test**

Add assertions to `tests/test_ai_optimization_ops_scripts.py` that the runbook
contains `CHAT_MODEL_PRICING_JSON`, the golden path, baseline command,
candidate command, 20% planner threshold, total-cost non-increase rule, and exit
codes `0/1/2`. Assert `ai-pr-gate.yml` includes:

```text
tests/test_chat_model_usage.py
tests/test_chat_model_routing_experiment.py
tests/test_chat_quality_golden_dataset.py
```

- [ ] **Step 2: Run the exact new contract test**

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_ai_optimization_ops_scripts.py::test_model_routing_runbook_and_ci_contract -q
```

Expected: FAIL because docs and CI do not reference the new gate.

- [ ] **Step 3: Document configuration and two-run operation**

Add a redacted JSON example to `.env.example`; use non-production provider/model
names and decimal string prices, never real credentials.

Update the runbook with exact operator commands:

```bash
AI_MODEL_ROUTING_SAMPLES=scripts/chat_quality_golden_60.json \
AI_MODEL_ROUTING_OUTPUT=outputs/model-routing/baseline.json \
./.venv/bin/python scripts/chat_model_routing_experiment.py \
  --planner-model-label "$CHAT_PLANNER_MODEL_NAME" \
  --answer-model-label "$CHAT_ANSWER_MODEL_NAME"

AI_MODEL_ROUTING_SAMPLES=scripts/chat_quality_golden_60.json \
AI_MODEL_ROUTING_OUTPUT=outputs/model-routing/candidate.json \
./.venv/bin/python scripts/chat_model_routing_experiment.py \
  --baseline-report outputs/model-routing/baseline.json \
  --planner-model-label "$CHAT_PLANNER_MODEL_NAME" \
  --answer-model-label "$CHAT_ANSWER_MODEL_NAME"
```

State that each configuration must use explicit model names and a catalog
covering them, the answer model must remain fixed, live calls need separate
approval, and no passing report authorizes deployment. Add the three
deterministic test files to the existing AI PR gate test command.

- [ ] **Step 4: Run changed-file formatting/static checks**

Run the repository's existing formatting/lint commands from
`.github/workflows/ai-pr-gate.yml` against changed Python files, then run:

```bash
git diff --check
```

Expected: PASS with no changed-file formatting or whitespace errors.

- [ ] **Step 5: Run focused regression**

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_config_internal_token.py \
  tests/test_chat_model_usage.py \
  tests/test_chat_cost_metrics.py \
  tests/test_llm_model_candidates.py \
  tests/test_baseball_agent_llm_planner.py \
  tests/test_chat_api_smoke.py \
  tests/test_observability_metrics.py \
  tests/test_ai_optimization_ops_scripts.py \
  tests/test_chat_model_routing_experiment.py \
  tests/test_chat_quality_golden_dataset.py -q
```

Expected: PASS.

- [ ] **Step 6: Run full AI and policy verification**

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/
python3 scripts/validate_baseball_data_policy.py
```

Expected: full pytest PASS and `External baseball data policy OK`.

- [ ] **Step 7: Review the final diff against the approved spec**

```bash
rg -n "CHAT_MODEL_PRICING_JSON|model_usage_complete|planner_reduction_percent" \
  app scripts tests docs .env.example
rg -n "http://|https://" scripts/chat_quality_golden_60.json
git diff --check origin/main...HEAD
git status --short
```

Expected: implementation symbols are present, golden questions contain no URLs,
diff check is clean, and only intended files are modified. Record the commands
and results in commit/PR evidence.

- [ ] **Step 8: Commit operator and CI contracts**

```bash
git add .env.example docs/ai-optimization-rollout-runbook.md \
  .github/workflows/ai-pr-gate.yml tests/test_ai_optimization_ops_scripts.py
git commit -m "docs(ai): document model routing cost gate"
```

---

## Completion Evidence

Implementation is locally complete only when every task commit exists and all
focused/full tests plus the baseball-data policy check pass. The implementation
must not claim the 20% optimization target itself has been achieved until an
approved baseline and candidate live run produce comparable reports. Frontend
promotion, target-host release gates, deployment, and post-deploy observation
remain separate approvals and are not part of this plan.
