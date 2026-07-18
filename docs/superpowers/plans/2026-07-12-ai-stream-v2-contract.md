# AI Stream v2 Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a typed `version/type/data` SSE v2 contract to the existing chat and coach paths, retain v1 through header negotiation, transparently forward negotiation through Spring, and consume generated discriminated unions in React.

**Architecture:** FastAPI owns Pydantic request/event models, a deterministic OpenAPI artifact, and an endpoint-aware v1/v2 serializer. Spring forwards only `X-AI-Event-Version` and leaves stream bytes untouched. The frontend vendors the AI artifact, generates TypeScript with its existing `openapi-typescript` dependency, and selects strict v2 or legacy v1 parsing through `VITE_AI_EVENT_VERSION`.

**Tech Stack:** Python 3/FastAPI/Pydantic 2/sse-starlette/pytest/prometheus-client; Java 21/Spring Boot/WebClient/JUnit 5/MockMvc; TypeScript 5/React 18/Vite/Node test runner/openapi-typescript.

## Global Constraints

- Existing public paths remain `/api/ai/chat/stream` and `/api/ai/coach/analyze` through Spring and `/ai/chat/stream` and `/ai/coach/analyze` in FastAPI.
- Header absent or `1` returns v1; header `2` returns v2; every other explicit value returns HTTP 406 with `AI_EVENT_VERSION_UNSUPPORTED` before streaming.
- Every v2 frame has exactly `version`, `type`, and `data`; numeric `version` is `2`; SSE `event` must equal JSON `type`.
- v2 wire fields use canonical snake_case; v1 aliases and `[DONE]` remain unchanged.
- Spring never parses, transforms, or buffers event semantics and forwards no arbitrary browser headers.
- Frontend defaults to v2 after rollout and supports `VITE_AI_EVENT_VERSION=1` rollback.
- Do not add external baseball crawling, scraping, web-search repair, external baseball APIs, or synthesized facts; preserve `MANUAL_BASEBALL_DATA_REQUIRED` and `manual_data_request`.
- Preserve unrelated dirty work in all repositories and stage only files named by each task.
- For AI files already dirty before this plan (`app/observability/metrics.py`, `tests/test_observability_metrics.py`, `app/routers/chat_stream.py`, and `tests/test_chat_api_smoke.py`), use interactive hunk staging and inspect the cached diff; never stage the whole file.
- Every production-code task follows test-first RED, observed failure, minimal GREEN, focused verification, then a narrow commit.

## File and Responsibility Map

### AI repository (`bega_AI`)

- Create `app/contracts/__init__.py`: contract package exports.
- Create `app/contracts/stream_events_v2.py`: closed Pydantic data/envelope union and runtime decoder.
- Create `app/contracts/stream_requests.py`: chat and coach request models with existing compatibility validation.
- Create `app/streaming/__init__.py`: streaming package exports.
- Create `app/streaming/versioned_sse.py`: negotiation, legacy normalization, v1 passthrough, v2 serialization, terminal handling, and common response factory.
- Create `contracts/ai-stream-v2.openapi.json`: deterministic generated contract artifact.
- Create `scripts/export_ai_stream_contract.py`: offline artifact exporter/checker.
- Create `tests/test_stream_events_v2.py`: envelope/model validation.
- Create `tests/test_versioned_sse.py`: negotiation and adapter behavior.
- Create `tests/test_ai_stream_contract_export.py`: deterministic artifact freshness.
- Modify `app/routers/chat_stream.py`: typed chat request validation and versioned response factory on every active chat stream path.
- Modify `app/routers/coach.py`: shared coach request model and versioned response factory on `/analyze` only; `/analyze-legacy` stays v1-only and deprecated.
- Modify `app/observability/metrics.py`: bounded stream-contract counters.
- Modify focused existing chat/coach tests for route-level v1/v2 characterization.

### Backend repository (`bega_backend`)

- Modify `BEGA_PROJECT/src/main/java/com/example/ai/controller/AiProxyController.java`: read optional negotiation header for chat and coach streams.
- Modify `BEGA_PROJECT/src/main/java/com/example/ai/service/AiProxyService.java`: request-header forwarding overload and response-header allow-list.
- Modify `BEGA_PROJECT/src/main/java/com/example/auth/config/SecurityConfig.java`: allow/expose only the negotiation header in CORS.
- Modify `BEGA_PROJECT/src/test/java/com/example/ai/controller/AiProxyControllerTest.java`: controller forwarding and response-header tests.
- Modify `BEGA_PROJECT/src/test/java/com/example/ai/service/AiProxyServiceTest.java`: upstream request, retry, and transparent-body tests.
- Modify `BEGA_PROJECT/src/test/java/com/example/auth/config/CorsIntegrationTest.java`: preflight and exposed-header assertions.

### Frontend repository (`bega_frontend`)

- Create `contracts/ai-stream-v2.openapi.json`: read-only vendored AI artifact.
- Create `contracts/ai-stream-v2.metadata.json`: source, schema version, and SHA-256.
- Create `scripts/sync-ai-stream-contract.mjs`: sibling copy/hash metadata/generation orchestration.
- Create `scripts/generate-ai-stream-types.mjs`: deterministic openapi-typescript generation and `--check`.
- Create `src/api/generated/aiStreamV2.ts`: generated TypeScript.
- Create `src/api/aiStreamContract.ts`: strict runtime v2 decoder and stable contract error.
- Create `src/api/aiStreamContract.test.ts`: every discriminator and failure class.
- Modify `package.json`: AI contract sync/generate/check scripts.
- Modify `src/api/sse.ts` and `src/api/sse.test.ts`: caller-controlled terminal recognition for typed v2 done while preserving `[DONE]`.
- Modify `src/api/chatbot.ts` and `src/api/chatbot.test.ts`: send version header and exhaustively consume chat v2.
- Modify `src/api/coach.ts` and `src/api/coach.test.ts`: generated request type, exhaustive coach v2, snake-case mapping, and v1 rollback.

---

### Task 1: Define the closed AI v2 event model

**Files:**
- Create: `app/contracts/__init__.py`
- Create: `app/contracts/stream_events_v2.py`
- Test: `tests/test_stream_events_v2.py`

**Interfaces:**
- Consumes: Pydantic 2 already declared in `requirements.txt`.
- Produces: `AiStreamV2Event`, `parse_v2_event(value: object) -> AiStreamV2Event`, `event_schema_components() -> dict[str, object]`, and concrete envelope classes discriminated by `type`.

- [ ] **Step 1: Write failing model tests**

Create tests that import the wished-for API and assert the exact top-level contract, the closed event union, snake-case queue fields, coach metadata aliases being rejected, manual-data metadata preservation, terminal reason validation, and unknown top-level field rejection:

```python
from pydantic import ValidationError
import pytest

from app.contracts.stream_events_v2 import parse_v2_event


def test_chat_delta_has_exact_v2_envelope() -> None:
    event = parse_v2_event({
        "version": 2,
        "type": "chat.message.delta",
        "data": {"delta": "첫 문장"},
    })
    assert event.model_dump(exclude_none=True) == {
        "version": 2,
        "type": "chat.message.delta",
        "data": {"delta": "첫 문장"},
    }


@pytest.mark.parametrize("event_type", [
    "chat.status", "chat.queue", "chat.message.delta", "chat.meta",
    "coach.status", "coach.preview.chunk", "coach.preview.reset",
    "coach.message.delta", "coach.meta", "stream.error", "stream.done",
])
def test_all_approved_event_types_parse(event_type: str) -> None:
    assert event_type in approved_example_payloads()
    assert parse_v2_event(approved_example_payloads()[event_type]).type == event_type


def test_v2_rejects_camel_case_coach_alias() -> None:
    with pytest.raises(ValidationError):
        parse_v2_event({
            "version": 2,
            "type": "coach.meta",
            "data": {"analysisType": "game_review"},
        })


def test_v2_rejects_unknown_top_level_field() -> None:
    with pytest.raises(ValidationError):
        parse_v2_event({
            "version": 2,
            "type": "stream.done",
            "data": {"reason": "completed"},
            "legacy": True,
        })
```

The test helper contains one valid complete payload for every approved event type, including a `coach.meta` example with canonical v2 `manual_data_request` fields `scope`, `missing_items`, `operator_message`, `blocking`, and optional `code`. The v1 adapter leaves the existing camel-case nested fields unchanged for legacy consumers.

- [ ] **Step 2: Run tests and observe RED**

Run: `./.venv/bin/python -m pytest tests/test_stream_events_v2.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'app.contracts'`.

- [ ] **Step 3: Implement Pydantic event models**

Use `ConfigDict(extra="forbid")` on envelopes and concrete data models. Define reusable typed models for tool calls, data sources, performance values, manual-data requests, structured coach response, and coach risk items. Preserve flexible leaf values only where the existing contract is intentionally open, such as tool parameters and performance values; do not use `Dict[str, Any]` as an event data model. Define each envelope with `version: Literal[2]`, a literal `type`, and its concrete `data`. Build the concrete union with a discriminating `Field` and validate through `TypeAdapter`:

```python
AiStreamV2Event = Annotated[
    ChatStatusEvent
    | ChatQueueEvent
    | ChatMessageDeltaEvent
    | ChatMetaEvent
    | CoachStatusEvent
    | CoachPreviewChunkEvent
    | CoachPreviewResetEvent
    | CoachMessageDeltaEvent
    | CoachMetaEvent
    | StreamErrorEvent
    | StreamDoneEvent,
    Field(discriminator="type"),
]

_EVENT_ADAPTER = TypeAdapter(AiStreamV2Event)


def parse_v2_event(value: object) -> AiStreamV2Event:
    return _EVENT_ADAPTER.validate_python(value)


def event_schema_components() -> dict[str, object]:
    return _EVENT_ADAPTER.json_schema(ref_template="#/components/schemas/{model}")
```

Canonical enums are closed: queue state `queued|processing`, chat style `markdown|json|compact`, coach request mode `auto_brief|manual_detail`, analysis type `game_review|game_preview`, generation mode `deterministic_auto|deterministic_review|deterministic_preview|llm_manual|evidence_fallback`, data quality `grounded|partial|insufficient`, and done reason `completed|error|cancelled`.

- [ ] **Step 4: Run tests and observe GREEN**

Run: `./.venv/bin/python -m pytest tests/test_stream_events_v2.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add app/contracts/__init__.py app/contracts/stream_events_v2.py tests/test_stream_events_v2.py
git commit -m "feat: define AI stream v2 event contract"
```

### Task 2: Add request contracts and deterministic export

**Files:**
- Create: `app/contracts/stream_requests.py`
- Create: `scripts/export_ai_stream_contract.py`
- Create: `contracts/ai-stream-v2.openapi.json`
- Create: `tests/test_ai_stream_contract_export.py`
- Modify: `app/routers/coach.py`
- Test: `tests/test_coach.py`
- Test: `tests/test_coach_router_year_resolution.py`

**Interfaces:**
- Consumes: `AiStreamV2Event` schema from Task 1 and current coach validation constants.
- Produces: `ChatStreamRequest`, `CoachAnalyzeRequest`, `build_contract_document() -> dict[str, object]`, and CLI `--check`.

- [ ] **Step 1: Write failing request and exporter tests**

Assert that chat fields match current inputs, extra chat fields remain accepted initially, coach `team_id` and `analysisType` compatibility still normalize, auto brief rejects `question_override`, and exporter output is byte-stable:

```python
def test_chat_request_preserves_current_extra_field_policy() -> None:
    request = ChatStreamRequest.model_validate({
        "question": "테스트",
        "history": None,
        "client_marker": "legacy",
    })
    assert request.question == "테스트"
    assert request.model_extra == {"client_marker": "legacy"}


def test_coach_request_backfills_legacy_fields() -> None:
    request = CoachAnalyzeRequest.model_validate({
        "team_id": "HT",
        "analysisType": "game_review",
        "request_mode": "manual_detail",
    })
    assert request.home_team_id == "HT"
    assert request.analysis_type == "game_review"


def test_committed_contract_is_current() -> None:
    expected = render_contract_json(build_contract_document())
    assert CONTRACT_PATH.read_text(encoding="utf-8") == expected
```

- [ ] **Step 2: Run tests and observe RED**

Run: `./.venv/bin/python -m pytest tests/test_ai_stream_contract_export.py -q`

Expected: import failure for `app.contracts.stream_requests` or missing contract artifact.

- [ ] **Step 3: Implement request models and exporter**

Move the existing coach request fields and `model_validator(mode="before")` behavior into `CoachAnalyzeRequest`; preserve the route-local name with `AnalyzeRequest = CoachAnalyzeRequest` during migration. Define `ChatStreamRequest` with `ConfigDict(extra="allow")`, current request fields, bounded style literals, and history entry models. Keep existing route-level 400/413 validation until equivalence tests prove safe replacement.

The exporter builds OpenAPI `3.1.0`, sets title `Bega AI Stream Contract`, version `2.0.0`, places request schemas and the named union under `components.schemas`, and emits sorted UTF-8 JSON with two-space indentation and one trailing newline. `--check` compares generated bytes without writing; default mode writes atomically.

- [ ] **Step 4: Generate artifact and run GREEN tests**

Run:

```bash
./.venv/bin/python scripts/export_ai_stream_contract.py
./.venv/bin/python scripts/export_ai_stream_contract.py --check
./.venv/bin/python -m pytest tests/test_ai_stream_contract_export.py -q
./.venv/bin/python -m pytest tests/test_coach.py tests/test_coach_router_year_resolution.py -q -k 'request or payload or analysis_type or question_override'
```

Expected: exporter reports the committed artifact is current and the focused request compatibility tests pass.

- [ ] **Step 5: Commit Task 2 files**

```bash
git add app/contracts/stream_requests.py scripts/export_ai_stream_contract.py contracts/ai-stream-v2.openapi.json tests/test_ai_stream_contract_export.py app/routers/coach.py
git commit -m "feat: export typed AI stream contract"
```

### Task 3: Build negotiation, adapter, terminal handling, and metrics

**Files:**
- Create: `app/streaming/__init__.py`
- Create: `app/streaming/versioned_sse.py`
- Create: `tests/test_versioned_sse.py`
- Modify: `app/observability/metrics.py`
- Modify: `tests/test_observability_metrics.py`

**Interfaces:**
- Produces: `EventVersion = Literal[1, 2]`, `negotiate_event_version(raw)`, `versioned_events(events, endpoint, version)`, and `versioned_event_source(events, endpoint, version, headers, ping)`.

- [ ] **Step 1: Write failing negotiation and serialization tests**

Cover absent/1/2/whitespace/unsupported values, v1 object identity and `[DONE]`, chat and coach event mapping, event/type equality, error termination on validation failure, response version header, and bounded metrics:

```python
@pytest.mark.parametrize(("raw", "expected"), [(None, 1), ("1", 1), (" 2 ", 2)])
def test_negotiates_supported_versions(raw: str | None, expected: int) -> None:
    assert negotiate_event_version(raw) == expected


def test_unsupported_version_is_406() -> None:
    with pytest.raises(HTTPException) as raised:
        negotiate_event_version("3")
    assert raised.value.status_code == 406
    assert raised.value.detail == {
        "code": "AI_EVENT_VERSION_UNSUPPORTED",
        "supported_versions": ["1", "2"],
    }


@pytest.mark.asyncio
async def test_v1_validates_then_yields_original_event() -> None:
    original = {"event": "message", "data": '{"delta":"첫"}'}
    assert [item async for item in versioned_events(source(original), endpoint="chat", version=1)] == [original]


@pytest.mark.asyncio
async def test_v2_maps_chat_message_to_typed_envelope() -> None:
    emitted = [item async for item in versioned_events(
        source({"event": "message", "data": '{"delta":"첫"}'}),
        endpoint="chat",
        version=2,
    )]
    assert emitted == [{
        "event": "chat.message.delta",
        "data": '{"version":2,"type":"chat.message.delta","data":{"delta":"첫"}}',
    }]
```

- [ ] **Step 2: Run tests and observe RED**

Run: `./.venv/bin/python -m pytest tests/test_versioned_sse.py tests/test_observability_metrics.py -q`

Expected: missing `app.streaming.versioned_sse` imports or missing metrics.

- [ ] **Step 3: Implement adapter and bounded counters**

Parse legacy `data` JSON, map by endpoint and legacy event name, canonicalize known aliases only for v2, validate with Task 1 models, and yield the untouched original for v1. Convert `done/[DONE]` to `stream.done` only for v2. On validation failure after start, log only endpoint/version/event/validation category, increment the failure counter, emit a safe version-specific error and terminal pair, then stop.

Add counters with labels limited to `endpoint`, `version`, and closed `event_type`:

```python
AI_STREAM_REQUEST_TOTAL = Counter(
    "ai_stream_request_total",
    "AI stream requests by endpoint and negotiated event version.",
    ["endpoint", "version"],
)
AI_STREAM_EVENT_TOTAL = Counter(
    "ai_stream_event_total",
    "AI stream events emitted by endpoint, version, and event type.",
    ["endpoint", "version", "event_type"],
)
AI_STREAM_CONTRACT_FAILURE_TOTAL = Counter(
    "ai_stream_contract_failure_total",
    "AI stream event contract validation failures.",
    ["endpoint", "version"],
)
AI_STREAM_UNSUPPORTED_VERSION_TOTAL = Counter(
    "ai_stream_unsupported_version_total",
    "Unsupported AI stream event-version requests.",
    ["endpoint"],
)
```

- [ ] **Step 4: Run tests and observe GREEN**

Run: `./.venv/bin/python -m pytest tests/test_versioned_sse.py tests/test_observability_metrics.py -q`

Expected: all focused tests pass with no payload values appearing in metric labels or captured logs.

- [ ] **Step 5: Commit Task 3 files**

```bash
git add app/streaming/__init__.py app/streaming/versioned_sse.py tests/test_versioned_sse.py
git add -p app/observability/metrics.py tests/test_observability_metrics.py
git diff --cached --check
git diff --cached -- app/observability/metrics.py tests/test_observability_metrics.py
git commit -m "feat: negotiate versioned AI stream events"
```

Accept only contract-counter and contract-test hunks during `git add -p`; leave all pre-existing optimization and observability hunks unstaged.

### Task 4: Integrate chat with the versioned response factory

**Files:**
- Modify: `app/routers/chat_stream.py`
- Modify: `tests/test_chat_api_smoke.py`
- Modify: `tests/test_chat_stream_live.py`
- Modify: `tests/test_chat_stream_cancellation.py`

**Interfaces:**
- Consumes: `negotiate_event_version` and `versioned_event_source` from Task 3.
- Produces: v1/v2 chat behavior on the existing POST path and resolved response header.

- [ ] **Step 1: Write failing route characterization tests**

Add route tests for absent header v1, explicit v2 static/cache/queue/live/error/done sequences, unsupported version 406, and response header. Decode SSE frames and assert v1 still contains `[DONE]`, while v2 contains `chat.*`/`stream.*` envelopes and no `[DONE]`.

- [ ] **Step 2: Run tests and observe RED**

Run: `./.venv/bin/python -m pytest tests/test_chat_api_smoke.py tests/test_chat_stream_live.py tests/test_chat_stream_cancellation.py -q -k 'event_version or v2 or legacy_v1'`

Expected: v2 assertions fail because the route ignores the version header.

- [ ] **Step 3: Thread the negotiated version through every active chat response path**

Read the header with FastAPI `Header(default=None, alias="X-AI-Event-Version")` on POST `/stream`, negotiate before static/cache/queue selection, and add `event_version` parameters to `_make_static_sse_response`, `_make_cached_sse_response`, `_stream_response`, and `_make_queued_stream_response`. Replace their direct `EventSourceResponse` construction with `versioned_event_source`, passing the existing event generator, `endpoint="chat"`, the negotiated version, existing headers, and existing ping interval.

Validate `ChatStreamRequest` after the existing byte-size check and preserve current 400/413 public details. Do not alter completion, voice, GET debug stream, cache decisions, agent context, cancellation, or queue reservation behavior. The GET debug stream remains v1 unless explicitly brought into scope by a later design.

- [ ] **Step 4: Run chat tests and observe GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_chat_api_smoke.py tests/test_chat_stream_live.py tests/test_chat_stream_cancellation.py tests/test_versioned_sse.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 4 files**

```bash
git add -p app/routers/chat_stream.py tests/test_chat_api_smoke.py
git add tests/test_chat_stream_live.py tests/test_chat_stream_cancellation.py
git diff --cached --check
git diff --cached -- app/routers/chat_stream.py tests/test_chat_api_smoke.py
git commit -m "feat: serve negotiated chat stream v2"
```

Accept only version negotiation, response-factory, request-contract, and v2 characterization hunks during `git add -p`; keep all pre-existing chat optimization hunks unstaged.

### Task 5: Integrate coach with the versioned response factory

**Files:**
- Modify: `app/routers/coach.py`
- Modify: `tests/test_coach.py`
- Modify: `tests/test_coach_auto_brief_recovery.py`

**Interfaces:**
- Consumes: `CoachAnalyzeRequest`, `negotiate_event_version`, and `versioned_event_source`.
- Produces: v1/v2 coach behavior on `/analyze`; deprecated `/analyze-legacy` remains unchanged v1.

- [ ] **Step 1: Write failing coach v2 route tests**

Cover representative auto-brief, manual-detail, cached, in-progress, preview chunk/reset, metadata, public error, manual-data metadata, and terminal paths. Assert v2 emits only canonical `analysis_type` and `llm_skip_reason`, preserves `manual_data_request`, and produces typed error/done. Assert absent and explicit v1 still use legacy event names and `[DONE]`.

- [ ] **Step 2: Run tests and observe RED**

Run: `./.venv/bin/python -m pytest tests/test_coach.py tests/test_coach_auto_brief_recovery.py -q -k 'event_version or stream_v2 or legacy_v1'`

Expected: v2 assertions fail because `/analyze` ignores the version header.

- [ ] **Step 3: Integrate the common factory**

Accept the optional header on `analyze_team`, negotiate before `event_generator` is returned, and replace the `/analyze` `EventSourceResponse` with `versioned_event_source`, passing `event_generator()`, `endpoint="coach"`, the negotiated version, existing headers, and the existing ping interval. Use `CoachAnalyzeRequest` as the route payload type while retaining the local `AnalyzeRequest` alias for imports and tests. Leave `/analyze-legacy` direct and v1-only.

The adapter maps legacy coach `status` payloads that use `message` to v2 `coach.status.data.status`, maps `message` to `coach.message.delta`, maps preview events, and canonicalizes metadata aliases without changing the original v1 dictionaries.

- [ ] **Step 4: Run coach and contract tests and observe GREEN**

Run the focused coach files plus:

```bash
./.venv/bin/python -m pytest tests/test_stream_events_v2.py tests/test_versioned_sse.py tests/test_ai_stream_contract_export.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5 files**

Stage only the coach route and focused coach test files, then commit:

```bash
git add app/routers/coach.py tests/test_coach.py tests/test_coach_auto_brief_recovery.py
git commit -m "feat: serve negotiated coach stream v2"
```

### Task 6: Forward only the negotiation header through Spring

**Files:**
- Modify: `BEGA_PROJECT/src/main/java/com/example/ai/controller/AiProxyController.java`
- Modify: `BEGA_PROJECT/src/main/java/com/example/ai/service/AiProxyService.java`
- Modify: `BEGA_PROJECT/src/test/java/com/example/ai/controller/AiProxyControllerTest.java`
- Modify: `BEGA_PROJECT/src/test/java/com/example/ai/service/AiProxyServiceTest.java`

**Interfaces:**
- Produces: `forwardJsonStream(String uri, String payload, @Nullable String eventVersion)` while retaining the two-argument overload for current callers/tests.

- [ ] **Step 1: Write failing controller and service tests**

In controller tests, send header `2` and verify the three-argument service call; omit it and verify `null`. In service tests, capture MockWebServer requests and assert only `X-AI-Event-Version` is forwarded, is retained on internal-token retry, response header is retained, and SSE bytes match exactly.

- [ ] **Step 2: Run tests and observe RED**

Run from `bega_backend/BEGA_PROJECT`:

```bash
./gradlew test --tests "*AiProxyControllerTest" --tests "*AiProxyServiceTest"
```

Expected: compilation fails because the three-argument overload does not exist or verification fails because the header is absent.

- [ ] **Step 3: Implement minimal forwarding**

Add a constant `AI_EVENT_VERSION_HEADER = "X-AI-Event-Version"`. Controller methods read `@RequestHeader(value = AI_EVENT_VERSION_HEADER, required = false) String eventVersion`. The service overload applies the header only when `StringUtils.hasText(eventVersion)`, using `eventVersion.trim()`, after internal auth is applied. Add the same constant to `PASS_THROUGH_HEADERS`. Do not accept a general `HttpHeaders` argument.

Retain:

```java
public ProxyStreamResponse forwardJsonStream(String uri, String payload) {
    return forwardJsonStream(uri, payload, null);
}
```

- [ ] **Step 4: Run tests and observe GREEN**

Run the same focused Gradle command.

Expected: both test classes pass.

- [ ] **Step 5: Commit Task 6 files**

```bash
git add BEGA_PROJECT/src/main/java/com/example/ai/controller/AiProxyController.java BEGA_PROJECT/src/main/java/com/example/ai/service/AiProxyService.java BEGA_PROJECT/src/test/java/com/example/ai/controller/AiProxyControllerTest.java BEGA_PROJECT/src/test/java/com/example/ai/service/AiProxyServiceTest.java
git commit -m "feat: forward AI stream event version"
```

### Task 7: Allow and expose the negotiation header through backend CORS

**Files:**
- Modify: `BEGA_PROJECT/src/main/java/com/example/auth/config/SecurityConfig.java`
- Modify: `BEGA_PROJECT/src/test/java/com/example/auth/config/CorsIntegrationTest.java`

**Interfaces:**
- Consumes: header constant or literal from Task 6.
- Produces: browser preflight acceptance and readable response negotiation header without exposing credentials.

- [ ] **Step 1: Write failing CORS tests**

Add a preflight with `Access-Control-Request-Headers: content-type,x-ai-event-version` and assert the allow header contains both. Extend the exposed-header test to require `X-AI-Event-Version` while still rejecting `Set-Cookie` and `Authorization`.

- [ ] **Step 2: Run and observe RED**

Run: `./gradlew test --tests "*CorsIntegrationTest"`

Expected: requested or exposed version-header assertion fails.

- [ ] **Step 3: Update the narrow CORS lists**

Add `X-AI-Event-Version` to existing allowed and exposed header lists only. Do not introduce `*`, expose authorization headers, or alter allowed origins, methods, credentials, cookies, or auth rules.

- [ ] **Step 4: Run and observe GREEN**

Run: `./gradlew test --tests "*CorsIntegrationTest" --tests "*AiProxyControllerTest" --tests "*AiProxyServiceTest"`

Expected: all focused tests pass.

- [ ] **Step 5: Commit Task 7 files**

```bash
git add BEGA_PROJECT/src/main/java/com/example/auth/config/SecurityConfig.java BEGA_PROJECT/src/test/java/com/example/auth/config/CorsIntegrationTest.java
git commit -m "feat: expose AI event version header"
```

### Task 8: Vendor the AI contract and generate frontend types

**Files:**
- Create: `contracts/ai-stream-v2.openapi.json`
- Create: `contracts/ai-stream-v2.metadata.json`
- Create: `scripts/sync-ai-stream-contract.mjs`
- Create: `scripts/generate-ai-stream-types.mjs`
- Create: `src/api/generated/aiStreamV2.ts`
- Create: `scripts/ai-stream-contract.test.mjs`
- Modify: `package.json`

**Interfaces:**
- Consumes: `../bega_AI/contracts/ai-stream-v2.openapi.json` for explicit sync only.
- Produces: generated OpenAPI `components` types and offline `npm run api:ai-stream:check`.

- [ ] **Step 1: Write failing script tests**

Test that sync copies exact bytes, metadata SHA-256 matches the vendored artifact, generation `--check` detects drift, and scripts never fetch a network URL.

- [ ] **Step 2: Run and observe RED**

Run: `node --test scripts/ai-stream-contract.test.mjs`

Expected: missing script/module failure.

- [ ] **Step 3: Implement offline sync and generation**

Use Node built-ins `fs`, `crypto`, `path`, `os`, and `child_process`. The sync source defaults to `../bega_AI/contracts/ai-stream-v2.openapi.json` and may be overridden only by a local filesystem argument. Construct metadata from the copied bytes with this exact object shape:

```javascript
const metadata = {
  source_repository: 'BegaBaseball/AI',
  schema_version: '2.0.0',
  sha256: createHash('sha256').update(contractBytes).digest('hex'),
};
```

Generation invokes local `node_modules/.bin/openapi-typescript` with the vendored file. `--check` generates to a temporary directory and compares bytes. Add scripts:

```json
{
  "api:ai-stream:sync": "node scripts/sync-ai-stream-contract.mjs",
  "api:ai-stream:types": "node scripts/generate-ai-stream-types.mjs",
  "api:ai-stream:check": "node scripts/generate-ai-stream-types.mjs --check"
}
```

- [ ] **Step 4: Sync, generate, and observe GREEN**

Run:

```bash
npm run api:ai-stream:sync
npm run api:ai-stream:check
node --test scripts/ai-stream-contract.test.mjs
```

Expected: snapshot hash matches and generated types are current.

- [ ] **Step 5: Commit Task 8 files**

```bash
git add contracts/ai-stream-v2.openapi.json contracts/ai-stream-v2.metadata.json scripts/sync-ai-stream-contract.mjs scripts/generate-ai-stream-types.mjs scripts/ai-stream-contract.test.mjs src/api/generated/aiStreamV2.ts package.json
git commit -m "build: generate AI stream contract types"
```

### Task 9: Add strict frontend v2 decoding and terminal support

**Files:**
- Create: `src/api/aiStreamContract.ts`
- Create: `src/api/aiStreamContract.test.ts`
- Modify: `src/api/sse.ts`
- Modify: `src/api/sse.test.ts`

**Interfaces:**
- Produces: `AiStreamContractError`, `decodeAiStreamV2Event(event: SseEvent)`, `getAiEventVersion() -> '1'|'2'`, and `isTypedDone(event) -> boolean`.

- [ ] **Step 1: Write failing decoder tests**

Use one fixture per generated discriminator. Assert malformed JSON, version mismatch, unknown type, missing required data, extra top-level field, and SSE event/type mismatch throw `AiStreamContractError`. Assert typed done causes the low-level reader to cancel without waiting for EOF while legacy `[DONE]` still works.

- [ ] **Step 2: Run and observe RED**

Run: `node --import tsx --test src/api/aiStreamContract.test.ts src/api/sse.test.ts`

Expected: module import or typed terminal assertion fails.

- [ ] **Step 3: Implement decoder and caller-controlled completion**

Generated TypeScript supplies compile-time unions. Runtime decoder uses explicit object/string/number/boolean/array guards for the closed event set and returns the generated union type only after validation. `getAiEventVersion` accepts only env value `1` or `2`, defaults to `2`, and throws on other configured values.

Extend `ConsumeSseStreamOptions` with optional `isTerminalEvent?: (event: SseEvent) => boolean`. After delivering a non-legacy event, mark `sawDone` when this predicate returns true. Preserve existing line parsing, keepalive behavior, event reset, timeout, abort, cancellation, and `[DONE]` handling.

- [ ] **Step 4: Run and observe GREEN**

Run the same Node test command.

Expected: all decoder and SSE tests pass.

- [ ] **Step 5: Commit Task 9 files**

```bash
git add src/api/aiStreamContract.ts src/api/aiStreamContract.test.ts src/api/sse.ts src/api/sse.test.ts
git commit -m "feat: decode typed AI stream v2 events"
```

### Task 10: Migrate frontend chat to v2 with v1 rollback

**Files:**
- Modify: `src/api/chatbot.ts`
- Modify: `src/api/chatbot.test.ts`
- Modify: `src/types/chatbot.ts`

**Interfaces:**
- Consumes: generated `ChatStreamRequest`, strict decoder, version configuration, and typed terminal predicate.
- Produces: exhaustive chat event handling with the existing `onDelta`, `onMeta`, and `onQueueStatus` APIs.

- [ ] **Step 1: Write failing v2 chat tests**

Return `X-AI-Event-Version: 2` and typed status, queue, delta, meta, error, and done frames. Assert request header `2`, snake-case queue mapping, meta normalization, error mapping, and no wait for EOF. Add response-version mismatch and `VITE_AI_EVENT_VERSION=1` legacy rollback tests.

- [ ] **Step 2: Run and observe RED**

Run: `node --import tsx --test src/api/chatbot.test.ts src/api/aiStreamContract.test.ts src/api/sse.test.ts`

Expected: request header or typed event handling assertions fail.

- [ ] **Step 3: Implement exhaustive v2 chat dispatch**

Set `X-AI-Event-Version` on the existing request. For configured v2, require response header `2`, decode each frame, and switch on `chat.status`, `chat.queue`, `chat.message.delta`, `chat.meta`, `stream.error`, and `stream.done`. Reject coach types on the chat endpoint. Map wire queue snake-case to the existing camel-case callback. Reuse `normalizeAiStreamMeta` for the concrete chat meta data. For configured v1, retain the existing parser in a named legacy helper. Replace the handwritten request interface with the generated request type while preserving the existing exported application alias if callers need it.

- [ ] **Step 4: Run and observe GREEN**

Run the same Node test command.

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 10 files**

```bash
git add src/api/chatbot.ts src/api/chatbot.test.ts src/types/chatbot.ts
git commit -m "feat: consume chat stream v2 contract"
```

### Task 11: Migrate frontend coach to v2 with v1 rollback

**Files:**
- Modify: `src/api/coach.ts`
- Modify: `src/api/coach.test.ts`

**Interfaces:**
- Consumes: generated `CoachAnalyzeRequest`, strict decoder, and typed terminal predicate.
- Produces: existing `CoachAnalyzeResponse` application model populated from canonical v2 coach data.

- [ ] **Step 1: Write failing v2 coach tests**

Cover typed status, preview chunk/reset, delta, meta, error, done, manual-data metadata, payload-too-large error, response-version mismatch, terminal recovery policy, and configured v1 rollback. Assert the request contains only canonical `analysis_type` after existing normalization.

- [ ] **Step 2: Run and observe RED**

Run: `node --import tsx --test src/api/coach.test.ts src/api/aiStreamContract.test.ts src/api/sse.test.ts`

Expected: typed coach events are ignored or fail existing raw parsing.

- [ ] **Step 3: Implement exhaustive v2 coach dispatch**

Set the version header and require matching response negotiation in v2. Switch on `coach.status`, `coach.preview.chunk`, `coach.preview.reset`, `coach.message.delta`, `coach.meta`, `stream.error`, and `stream.done`; reject chat types. Replace `AiStreamMetaPayload & Record<string, unknown>` with generated data types. Map canonical `analysis_type` and `llm_skip_reason` into existing application aliases only at the return boundary. Preserve auth reissue, retries, payload limit, timeout, abort, terminal-meta recovery, manual-data request, and win-probability validation. Keep current v1 logic in a named legacy helper selected only when configured version is `1`.

- [ ] **Step 4: Run and observe GREEN**

Run the same Node test command.

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 11 files**

```bash
git add src/api/coach.ts src/api/coach.test.ts
git commit -m "feat: consume coach stream v2 contract"
```

### Task 12: Cross-service verification, policy gate, and reviews

**Files:**
- Modify only files required by a newly reproduced defect, and only after adding a failing regression test.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: evidence that the approved design is complete and compatible.

- [ ] **Step 1: Verify the AI repository**

Run from `bega_AI`:

```bash
./.venv/bin/python scripts/export_ai_stream_contract.py --check
./.venv/bin/python -m pytest tests/test_stream_events_v2.py tests/test_versioned_sse.py tests/test_ai_stream_contract_export.py -q
./.venv/bin/python -m pytest tests/ -q
```

Expected: artifact current; focused and full suites pass.

- [ ] **Step 2: Verify the backend repository**

Run from `bega_backend/BEGA_PROJECT`:

```bash
./gradlew test --tests "*AiProxyControllerTest" --tests "*AiProxyServiceTest" --tests "*CorsIntegrationTest"
./gradlew test
./gradlew migrationSafetyCheck
```

Expected: focused and full suites plus migration gate pass.

- [ ] **Step 3: Verify the frontend repository**

Run from `bega_frontend`:

```bash
npm run api:ai-stream:check
node --import tsx --test src/api/aiStreamContract.test.ts src/api/sse.test.ts src/api/chatbot.test.ts src/api/coach.test.ts
npm run test:unit
npm run build
```

Expected: contract check, focused/unit tests, and production build pass.

- [ ] **Step 4: Verify the baseball-data policy**

Run from `/Users/mac/project/KBO_platform`:

```bash
python3 scripts/validate_baseball_data_policy.py
```

Expected: `External baseball data policy OK`.

- [ ] **Step 5: Audit intentional changes**

In each repository run `git status --short`, `git diff --check`, and `git log --oneline --decorate -15`. Confirm every contract commit contains only files listed in its task and all pre-existing dirty files remain present and uncommitted unless intentionally changed by this plan.

- [ ] **Step 6: Run independent reviews**

Use the project-required general code reviewer for AI/backend/cross-service changes, frontend reviewer for React/TypeScript changes, and security reviewer for header, CORS, error, logging, and proxy behavior. Treat every P0-P2 finding as requiring a failing regression test and fix before completion; resolve or explicitly document lower-priority findings.

- [ ] **Step 7: Final compatibility audit**

Prove each success criterion from the design with file/test evidence: header negotiation, exact v2 envelope, v1 output, schema generation, frontend generated types, Spring transparency, explicit failures, response header, v1 rollback, policy preservation, and clean scoped commits. Do not declare completion from green focused tests alone.
