# AI Model Routing Cost Gate Design

**Status:** Design approved; written specification awaiting final review
**Date:** 2026-07-12
**Base revision:** `c1bae4fa2aead216b81126bfd5f9469ff53a040d`

## Context

The AI optimization runbook requires the planner/answer model split to preserve
route quality while reducing estimated cost by at least 20%. The merged
`scripts/chat_model_routing_experiment.py` records latency and configured model
labels, but it does not measure planner cost or compare a baseline and candidate.
The existing chat cost metric also estimates only visible request/answer text,
so it cannot prove the planner-specific exit criterion.

This design adds request-scoped model usage evidence and a deterministic cost
gate. It does not change model selection, prompts, tool routing, answer content,
or baseball-data sources.

## Goals

- Estimate each planner and answer model call from the complete message payload
  and generated response using one shared calculator.
- Apply model-specific input and output prices selected by provider and exact
  model name.
- Reuse the same usage records for internal response metadata, Prometheus
  counters, and experiment reports.
- Compare identical baseline and candidate golden runs.
- Pass only when planner cost falls by at least 20%, total model cost does not
  increase, the answer model stays fixed, and the quality contract passes.
- Keep missing or inconsistent baseball data on the
  `MANUAL_BASEBALL_DATA_REQUIRED` path.

## Non-Goals

- Provider billing reconciliation or provider usage API integration.
- Provider-tokenizer accuracy. The result is a deterministic estimate, not an
  invoice.
- Changing the planner model, answer model, prompt, tool policy, or cache
  serving settings as part of the instrumentation patch.
- Production rollout, external baseball lookup, scraping, or data repair.
- Treating semantic-cache shadow or canary evidence as complete.

## Approved Decisions

1. Cost source: estimated request tokens multiplied by configured model prices.
2. Cost gate: planner cost reduction must be at least 20%; total model cost must
   be less than or equal to baseline.
3. Quality gate: the approved golden 60 must pass, no new misroute or fallback
   may appear, and the answer model must remain unchanged.
4. Pricing source: a provider/model pricing catalog supplied as JSON
   configuration.
5. Architecture: one shared usage calculator produces request-level evidence
   for both reports and Prometheus.

## Golden Dataset Provenance

The operator-provided source workspace contains
`scripts/chat_quality_golden_60.json` with these immutable properties:

- schema versioned object with 60 cases
- categories: 20 multi-player narrative, 6 recovered regression, 20
  regulation, and 14 team database cases
- each case includes `id`, `question`, `category`,
  `expected_answerability`, and `allowed_planner_modes`
- `baseball_data_policy` is `internal_only`
- SHA-256:
  `8899b468d65a87a505083a5f229808a0b9b405819287f00ff707741715bb124d`

Implementation will add this existing operator-provided file to the canonical
AI branch without changing its questions or expected results. The gate records
the input file hash and ordered case IDs. It must not synthesize missing facts
or add an external baseball data source.

## Configuration Contract

Add optional `CHAT_MODEL_PRICING_JSON`. Its schema is keyed first by provider
and then by the exact model name:

```json
{
  "openrouter": {
    "vendor/planner-model": {
      "input_usd_per_1m_tokens": "0.10",
      "output_usd_per_1m_tokens": "0.30"
    },
    "vendor/answer-model": {
      "input_usd_per_1m_tokens": "1.00",
      "output_usd_per_1m_tokens": "3.00"
    }
  }
}
```

Prices are decimal strings in USD per one million tokens. Parsing rejects:

- malformed JSON or a non-object root
- blank provider or model keys
- missing input or output prices
- booleans, non-numeric values, negative values, NaN, or infinity
- unknown fields in a price entry

The catalog remains optional so an existing installation without pricing still
starts and records token counters. If the variable is present but invalid,
settings validation fails at startup. The existing global
`CHAT_COST_INPUT_USD_PER_1M_TOKENS` and
`CHAT_COST_OUTPUT_USD_PER_1M_TOKENS` settings remain supported as legacy
answer-metric configuration during migration, but a routing release gate
accepts only `model_catalog` pricing.

## Components

### Pricing Catalog

A focused core module owns JSON parsing, normalized provider/model lookup, and
`Decimal` prices. Lookup is exact after trimming surrounding whitespace; it
does not alias, guess, or silently fall back to another model.

### Usage Estimator

The shared estimator receives:

- role: `planner` or `answer`
- provider and exact configured model
- complete message list supplied to the model call
- concatenated generated text
- outcome: success or failure

It serializes the message list with stable compact JSON and estimates tokens as
`ceil(character_count / 3.5)`, matching the existing deterministic estimator.
It calculates money with `Decimal` and retains enough precision for small
golden runs. JSON reports serialize costs as fixed decimal strings;
Prometheus converts the final amount to `float` only at the counter boundary.

The immutable usage record contains only bounded metadata:

- role, provider, model, outcome, and pricing source
- input/output character and estimated-token counts
- input/output/total estimated USD

It never contains prompts, questions, answers, tool results, credentials, or
pricing JSON.

### Request-Scoped Collector

The existing `AgentRequestContext` is already isolated by the runtime's
`_REQUEST_CONTEXT` `ContextVar`. Add the collector as a field on that context,
and let planner and answer generation append through the current request
context. Do not add another `ContextVar`, process-global usage list, or
singleton mutable collector.

Multiple answer attempts are all retained, so retry cost cannot disappear from
the total. A failed or partial model attempt is retained as an unsuccessful
usage record when locally observable. Any failed model attempt invalidates the
candidate quality gate even when a later retry succeeds.

Baseline and candidate runs must set explicit planner and answer model names.
The existing generator disables provider fallback candidates when a model
override is present, so a successful explicitly routed call is attributable to
that model. A run using the implicit `default` label is invalid release
evidence.

### Internal Response Metadata

Completion and stream metadata expose a `model_usage` array containing the
sanitized usage records. Exact and semantic cache hits return an empty array
and their existing cache markers; they do not fabricate model cost.

The experiment forces cache bypass. Deterministic `fast_path`, `player_fast_path`,
or `predefined` cases may legitimately have no planner usage. A case whose
planner mode requires an LLM must have a priced planner record. Every report
also states whether all observed model calls were captured and priced.

### Prometheus Metrics

The existing token and cost metrics are fed by the shared usage records and
gain bounded role/provider/model attribution where the current metric contract
does not already provide it. No question, case ID, user ID, raw route, or error
text becomes a label.

Token counters increment even when pricing is unavailable. Cost counters
increment for every catalog-priced attempt, including a failed attempt with
observable input or partial output, so retries are not hidden. A separate
bounded outcome counter distinguishes `priced`, `unpriced`, and `failed`
records so missing pricing is visible without inventing a zero-cost success.

### Golden Runner And Comparator

Enhance `scripts/chat_model_routing_experiment.py` to load the golden JSON case
schema and run one cache-bypassed completion request per case. It applies the
existing natural-chat and answerability helpers, then additionally checks:

- actual planner mode is in `allowed_planner_modes`
- actual answerability equals `expected_answerability`
- no unexpected planner or answer fallback metadata is present
- all model calls are captured and catalog-priced

The report omits answer text but includes case ID, category, pass/fail reasons,
planner mode, fallback flags, latency, and sanitized usage records.

A baseline report is generated first. A candidate run accepts the baseline
report and validates the same dataset hash, ordered IDs, explicit answer model,
and complete 60-case execution before comparing costs.

## Gate Semantics

The candidate passes only when all conditions hold:

1. Both reports contain exactly 60 ordered cases from the same dataset hash.
2. Both reports used explicit catalog-priced planner and answer models.
3. Candidate answer model equals baseline answer model.
4. Every candidate case passes HTTP, natural-chat, answerability, and allowed
   planner-mode checks.
5. Candidate introduces no per-case misroute, planner fallback, answer
   fallback, failed call, or unexpected non-answer relative to baseline.
6. Baseline planner cost is greater than zero.
7. `planner_reduction_percent >= 20.00`.
8. `candidate_total_cost_usd <= baseline_total_cost_usd`.

Fast-path cases with no model call are valid when their allowed planner mode is
deterministic and the response metadata confirms no call occurred. Cost totals
sum every captured call, including retries.

Exit codes are stable:

- `0`: valid evidence and all gates pass
- `1`: valid evidence but quality or cost criteria fail
- `2`: invalid or incomplete evidence, configuration, input, or report

## Error Handling

- Invalid configured pricing fails settings validation with a field-scoped
  error and never logs the raw JSON.
- An unknown provider/model produces an unpriced usage record during normal
  service operation; it does not abort a user request solely for observability.
- The experiment fails closed with exit `2` for unpriced records, missing
  metadata, implicit model labels, duplicate/missing case IDs, dataset mismatch,
  answer-model mismatch that prevents comparison, or incomplete execution.
- HTTP/model/planner/answer failures are measured gate failures with exit `1`
  when the report remains structurally complete.
- Existing user-facing temporary failure and `MANUAL_BASEBALL_DATA_REQUIRED`
  contracts remain unchanged.

## Security And Privacy

- Pricing configuration contains no credentials and must not be logged.
- Usage metadata exposes counts and model labels only on existing internally
  authenticated AI routes.
- Reports do not retain answer text, prompts, tool payloads, API keys, headers,
  or user identifiers.
- Model and provider labels are normalized and bounded before metric use.
- No external baseball endpoint, web search, scraper, or synthetic data repair
  path is introduced.

## Test Strategy

### Unit Tests

- valid multi-provider pricing catalog and exact lookup
- malformed JSON, schema errors, non-finite/negative prices, and unknown fields
- deterministic message serialization and `3.5` chars-per-token rounding
- `Decimal` input, output, and total cost calculations
- priced, unpriced, failed, retry, and multi-call aggregation
- sanitized JSON serialization with no source text leakage

### Agent And API Tests

- planner and answer calls append role-correct usage records
- deterministic fast paths emit an empty but complete usage array
- retries accumulate rather than replace cost
- completion and stream metadata preserve existing fields and add sanitized
  usage
- cache hits do not fabricate generated-model cost
- missing catalog does not break normal requests

### Metrics Tests

- bounded role/provider/model labels
- token counters for priced and unpriced calls
- cost counters only for catalog-priced successful calls
- priced/unpriced/failed outcome counter behavior

### Experiment Tests

- exact 20% planner reduction boundary passes
- reduction below 20%, total cost increase, and baseline zero planner cost
- answer-model change, implicit model labels, unknown prices, missing usage,
  dataset hash mismatch, reordered/duplicate/missing cases
- allowed planner modes, expected answerability, natural-chat failures,
  misroutes, and new fallback behavior
- valid deterministic no-call cases
- exit codes `0`, `1`, and `2`
- operator-provided golden file schema, count, policy, categories, and hash

### Verification

Run focused tests while implementing, then:

```bash
./.venv/bin/python -m pytest \
  tests/test_chat_cost_metrics.py \
  tests/test_baseball_agent_llm_planner.py \
  tests/test_chat_api_smoke.py \
  tests/test_ai_optimization_ops_scripts.py \
  tests/test_chat_quality_golden_dataset.py \
  tests/test_observability_metrics.py
./.venv/bin/python -m pytest tests/
python3 scripts/validate_baseball_data_policy.py
```

No live model call, package installation, deployment, or external service call
is part of implementation verification without separate approval.

## Rollout And Compatibility

1. Merge instrumentation with no pricing catalog configured; serving behavior
   remains unchanged and token metrics continue.
2. Configure and validate the catalog in a controlled environment.
3. Generate a baseline golden report with explicit planner and answer models.
4. Restart with only the planner model changed and generate the candidate
   report from the same golden hash.
5. Review the automatic gate and failed-case summary before any model-routing
   rollout decision.

This design does not authorize production rollout. Production use still
requires the existing runbook stages, target-host access, and separately
approved live model calls.
