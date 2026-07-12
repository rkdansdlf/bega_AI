# AI Stream v2 Contract Design

## Context

Chat and coach streaming cross three independently versioned repositories:

- the React frontend opens an SSE request and manually narrows parsed JSON;
- the Spring backend authenticates, rate-limits, and byte-proxies the request;
- the FastAPI service produces endpoint-specific SSE events.

The current public paths are `/api/ai/chat/stream` and `/api/ai/coach/analyze` at the Spring boundary, backed by `/ai/chat/stream` and `/ai/coach/analyze` in FastAPI. The Spring proxy intentionally treats request and response bodies as raw strings and bytes. FastAPI currently builds many SSE events as untyped dictionaries, while the frontend intersects broad handwritten interfaces and silently ignores some parse or event mismatches. Coach metadata also carries legacy snake-case and camel-case aliases.

This makes the wire contract implicit and permits producer and consumer drift. The repositories are separate Git repositories, so the design must also support independently reproducible CI without assuming a shared root repository or a running sibling service.

The AI worktree contains substantial unrelated in-progress changes, including changes in `app/routers/chat_stream.py`. Implementation must treat those files as authoritative, make narrow edits around them, and never include unrelated changes in contract commits.

## Goals

1. Introduce a version 2 SSE envelope with the exact top-level shape `version`, `type`, and `data`.
2. Negotiate v2 on the existing endpoints through `X-AI-Event-Version: 2`.
3. Preserve v1 behavior for requests with no version header or an explicit value of `1`.
4. Make FastAPI the canonical owner of request and stream-event schemas.
5. Generate frontend request and discriminated-union event types from a committed AI-owned contract artifact.
6. Keep Spring as a transparent byte-stream proxy that knows only the negotiation header, not AI event semantics.
7. Fail explicitly on malformed, unknown, mismatched, or unsupported v2 contracts.
8. Preserve authentication, rate limits, queue behavior, timeouts, retries, caching, cancellation, public messages, and the `MANUAL_BASEBALL_DATA_REQUIRED` policy.
9. Provide a reversible rollout in which the frontend can return to v1 without changing endpoint paths.

## Non-goals

- Removing v1 or setting a v1 removal date.
- Introducing new endpoint paths.
- Parsing or transforming SSE payloads in Spring.
- Converting the backend proxy into a domain-aware AI client.
- Rewriting all chat or coach generation logic.
- Changing coach analysis, grounding, cache, or baseball-data behavior.
- Adding external baseball crawling, scraping, web-search repair, or external baseball API access.
- Creating a shared package registry or a fourth contract repository in this phase.
- Typing unrelated JSON, multipart, voice, ingest, moderation, vision, or admin endpoints.

## Chosen Architecture

FastAPI owns typed request and event models and performs version-specific serialization. Spring forwards one allow-listed negotiation header and continues to stream bytes without inspecting the body. The frontend sends v2 negotiation, consumes generated request/event types, and maps canonical snake-case wire data to its existing UI-facing models.

```text
React frontend
  -> X-AI-Event-Version: 2
  -> Spring auth/rate-limit/concurrency boundary
  -> transparent request + header forwarding
  -> FastAPI request model
  -> typed internal stream event
  -> v1 or v2 serializer
  -> transparent Spring byte stream
  -> generated frontend discriminated union parser
```

This is preferred over transforming SSE in Spring because a transforming proxy would own AI semantics, add buffering and parsing failure modes, and couple backend deployment to every event addition. It is preferred over emitting v1 and v2 fields in one payload because duplicate aliases would remain ambiguous and would not establish a meaningful typed boundary.

## Version Negotiation

The existing request paths remain unchanged.

| Request header | Result |
| --- | --- |
| absent | v1 |
| `X-AI-Event-Version: 1` | v1 |
| `X-AI-Event-Version: 2` | v2 |
| any other value | HTTP 406 before the stream starts |

FastAPI is authoritative for validation. Unsupported values return a stable JSON error with code `AI_EVENT_VERSION_UNSUPPORTED` and supported versions `1` and `2`. Whitespace is trimmed, but values are otherwise matched exactly. Version fallback is never implicit for an unsupported explicit value.

Successful streaming responses include `X-AI-Event-Version` with the resolved value. Spring forwards that response header. CORS configuration allows the request header and exposes the response header where cross-origin browser access is enabled.

When the browser omits the header, Spring also omits it upstream so existing requests retain their current shape. When the browser supplies `1` or `2`, Spring forwards only that single normalized header value. It does not forward arbitrary incoming headers.

## v2 Envelope

Every v2 SSE frame has a JSON object with exactly these top-level fields:

```json
{
  "version": 2,
  "type": "chat.message.delta",
  "data": {
    "delta": "분석 결과입니다."
  }
}
```

Rules:

- `version` is the numeric literal `2`.
- `type` is a closed string-literal union.
- `data` is a type-specific object, not an untyped map.
- The SSE `event` field equals the JSON `type` value.
- Wire field names are canonical snake_case.
- Unknown top-level fields are rejected by the v2 model.
- Endpoint-specific metadata is not collapsed into one cross-endpoint catch-all type.
- A v2 consumer treats an `event`/`type` mismatch as a contract failure.

The closed v2 type set is:

```text
chat.status
chat.queue
chat.message.delta
chat.meta
coach.status
coach.preview.chunk
coach.preview.reset
coach.message.delta
coach.meta
stream.error
stream.done
```

### Chat event data

- `chat.status`: stable public `message` text.
- `chat.queue`: `state` (`queued` or `processing`), `queue_position`, `estimated_wait_time`, and `rpm_limit`.
- `chat.message.delta`: non-empty `delta` text.
- `chat.meta`: the existing tool, source, verification, visualization, style, cache, planner, grounding, fallback, completion, and performance fields consumed by `normalizeAiStreamMeta` and chat UI code. Optionality matches real producer branches rather than making every field universally optional.

### Coach event data

- `coach.status`: stable status code in `status`; display text remains a frontend concern unless the current producer already supplies it.
- `coach.preview.chunk`: `text` and positive integer `attempt`.
- `coach.preview.reset`: positive integer `attempt`.
- `coach.message.delta`: `delta` text.
- `coach.meta`: the existing structured response, tools, verification, sources, focus signatures, request mode, analysis type, cache state, LLM skip reason, focus coverage, generation mode, data quality, grounding, supported fact count, game status, validation status, manual-data request, and home win-probability fields consumed by the frontend.

For v2, coach aliases are removed from the wire. `analysis_type` replaces `analysisType`, and `llm_skip_reason` replaces `llmSkipReason`. The frontend may continue exposing camel-case application properties after explicit mapping.

### Shared terminal events

`stream.error` uses:

```json
{
  "version": 2,
  "type": "stream.error",
  "data": {
    "code": "COACH_ANALYSIS_FAILED",
    "message": "분석 중 오류가 발생했습니다.",
    "detail": null,
    "retryable": true
  }
}
```

`code`, `message`, and `retryable` are required. `detail` is nullable and contains only safe public detail. Internal exception messages and validation payloads are never exposed.

`stream.done` replaces the v1 `[DONE]` sentinel for v2:

```json
{
  "version": 2,
  "type": "stream.done",
  "data": {
    "reason": "completed"
  }
}
```

`reason` is a closed union of `completed`, `error`, and `cancelled`.

## Request Contracts

The contract artifact also includes typed request schemas.

`ChatStreamRequest` covers the currently accepted `question`, `filters`, `history`, `cache_bypass`, and `style` fields. Its validation and extra-field policy initially match the existing `_validate_chat_payload`, question validation, and history decoding behavior so adopting the model does not create an accidental input-breaking change.

The existing coach `AnalyzeRequest` remains the runtime authority for `/ai/coach/analyze`. It is moved or re-exported into the contract surface without changing its compatibility validator, accepted aliases, or current request behavior. v2 affects response serialization; it does not require a different request body.

The Spring backend continues to enforce existing payload-size and coarse JSON limits before proxying. It does not duplicate the FastAPI domain request models.

## AI Contract Ownership

Pydantic v2 discriminated unions under an AI contract module are the runtime source of truth. A deterministic export script produces a committed OpenAPI 3 contract artifact containing:

- `ChatStreamRequest`;
- `CoachAnalyzeRequest`;
- every concrete v2 envelope and data schema;
- a named `AiStreamV2Event` `oneOf` union with a discriminator on `type`;
- stable negotiation and error schemas.

Proposed paths:

```text
bega_AI/app/contracts/stream_events_v2.py
bega_AI/app/contracts/stream_requests.py
bega_AI/contracts/ai-stream-v2.openapi.json
bega_AI/scripts/export_ai_stream_contract.py
```

The exporter uses stable ordering and formatting. AI CI regenerates into a temporary file and fails if the committed artifact differs. The artifact is generated from runtime models and is never edited by hand.

## v1 Compatibility Adapter

Existing chat and coach generators currently yield legacy event dictionaries through many branches. A shared adapter provides incremental migration without a big-bang rewrite:

```text
legacy event dict
  -> endpoint-aware normalizer
  -> concrete Pydantic event validation
  -> v1 passthrough or v2 serialization
```

For v1, the adapter validates a normalized copy and then yields the original dictionary unchanged. Existing event names, payload aliases, JSON construction, ordering, and `[DONE]` remain under the existing `EventSourceResponse` behavior.

For v2, the adapter serializes the validated typed event into the envelope and sets the SSE `event` field to the envelope type. Endpoint context distinguishes chat `status`, `message`, and `meta` from coach equivalents. New producer code constructs typed events directly; legacy dictionary production can be replaced incrementally after this phase.

All chat and coach `EventSourceResponse` construction paths use the common versioned response factory, including static, cache, queue, regular streaming, auto-brief, manual-detail, error, and early terminal branches. A route must not be able to bypass negotiation by returning an unwrapped event source.

## Spring Proxy Boundary

Spring changes are deliberately small:

- `AiProxyController` reads the optional `X-AI-Event-Version` header on chat and coach streaming endpoints.
- `AiProxyService.forwardJsonStream` accepts an optional version value and attaches only that header to the FastAPI request.
- Internal API authentication remains service-owned and cannot be overridden by the browser.
- `X-AI-Event-Version` is added to the response pass-through allow-list.
- The service continues to return `Flux<DataBuffer>` and `StreamingResponseBody` without event parsing or transformation.
- Existing retry-on-internal-token behavior preserves the event-version header on every attempt.

Focused tests prove request forwarding, absence behavior, response forwarding, retry preservation, and byte-for-byte body transparency.

## Frontend Contract Consumption

Because the frontend and AI service are separate repositories, the frontend commits a read-only vendored snapshot of the AI contract plus source metadata and SHA-256. It uses the already installed `openapi-typescript` tool to generate types.

Proposed paths:

```text
bega_frontend/contracts/ai-stream-v2.openapi.json
bega_frontend/contracts/ai-stream-v2.metadata.json
bega_frontend/src/api/generated/aiStreamV2.ts
bega_frontend/scripts/sync-ai-stream-contract.mjs
bega_frontend/scripts/generate-ai-stream-types.mjs
```

The metadata records the source repository, schema version, and artifact SHA-256. Generated TypeScript and vendored contract files are not hand-edited. Frontend CI verifies that generated TypeScript matches the committed snapshot. A local coordinated-release command copies the sibling AI artifact, updates metadata, regenerates types, and verifies the source and vendored hashes.

The frontend adds a typed v2 decoder above the low-level line-oriented SSE reader. The decoder:

1. parses JSON;
2. requires version `2`;
3. checks the closed `type` union;
4. verifies SSE `event` equals envelope `type`;
5. validates required fields before dispatch;
6. returns a discriminated union to chat or coach handlers.

The chat and coach API modules stop intersecting a broad metadata interface with `Record<string, unknown>`. They switch exhaustively on generated event types and then map canonical wire fields into existing application-facing types.

The frontend sends `X-AI-Event-Version` from a narrow configuration value. Its post-rollout default is `2`; setting `VITE_AI_EVENT_VERSION=1` restores the existing v1 parser and request behavior without changing routes. For v2, a missing or mismatched response version header is a visible contract error rather than a silent fallback.

## Error Handling

An unsupported version fails before `EventSourceResponse` starts, so the client receives an ordinary HTTP 406 JSON response.

If internal event validation fails after streaming has started, FastAPI:

1. records a sanitized error containing endpoint, negotiated version, event name, and validation category but no payload or user question;
2. increments a low-cardinality contract-failure metric;
3. emits the version-appropriate public error event;
4. emits the version-appropriate terminal event;
5. stops the generator.

For v1 this means the existing `error` shape followed by `[DONE]`. For v2 it means `stream.error` followed by `stream.done` with reason `error`.

The frontend no longer silently ignores malformed JSON, unknown v2 events, missing required fields, or event/type mismatches. It raises a stable client contract error. Existing abort, connect-timeout, read-timeout, HTTP status, retry, payload-too-large, authentication-expired, and queue handling remain separate from contract failures.

`MANUAL_BASEBALL_DATA_REQUIRED` remains a domain data-availability contract. Existing `manual_data_request` metadata is preserved in typed coach metadata and is not converted into an automated data fallback or an external lookup.

## Observability

Add low-cardinality metrics for:

- stream requests by endpoint and negotiated version;
- events emitted by endpoint, version, and closed event type;
- contract validation failures by endpoint and version;
- unsupported-version requests.

Logs may include request ID, endpoint, negotiated version, event type, and validation category. They must not include questions, answers, event payloads, tokens, secrets, or baseball data. The negotiated response header supports client diagnostics without inspecting stream contents.

## Rollout and Rollback

Deployment order is:

1. FastAPI with v2 models, serializers, v1 adapter, schema artifact, and v1 default.
2. Spring with optional request-header forwarding and response-header pass-through.
3. Frontend with the vendored matching schema, generated types, v2 decoder, and v2 request configuration.

Before step 3, all traffic remains v1. A frontend rollback sets `VITE_AI_EVENT_VERSION=1`; no backend or AI rollback is required. AI and Spring retain v1 indefinitely for this phase. Removing v1 requires a later design, usage evidence, and explicit approval.

Commits remain repository-local and narrowly scoped. No commit may absorb unrelated dirty worktree files.

## Testing Strategy

Implementation follows red-green-refactor in each repository.

### AI tests

- negotiation: absent, `1`, `2`, whitespace, and unsupported values;
- every concrete Pydantic envelope and discriminator;
- event/type equality and closed top-level fields;
- deterministic contract export and artifact freshness;
- v1 adapter output for representative static, cache, queue, normal, error, and terminal chat branches;
- v1 adapter output for representative auto-brief, manual-detail, cached, in-progress, preview, error, and terminal coach branches;
- v2 serialization for the same representative branches;
- mid-stream validation failure termination;
- contract metrics without payload labels;
- request model behavior equivalence.

### Backend tests

- chat and coach forwarding of explicit version `1` and `2`;
- missing header remains missing upstream;
- version header survives internal-token retry;
- response version header passes through;
- non-allow-listed browser headers do not pass through;
- streamed bytes are unchanged;
- existing security, concurrency, payload-limit, timeout, and error behavior remains green.

### Frontend tests

- contract snapshot SHA and generated-type freshness;
- valid parsing for every v2 discriminator;
- exhaustive chat and coach dispatch;
- malformed JSON, unknown version, unknown type, missing field, and event/type mismatch failures;
- v2 typed done and error behavior;
- v1 rollback parser behavior;
- chat delta, metadata, queue, retry, timeout, and abort regressions;
- coach preview, status, delta, metadata, terminal recovery, manual-data request, retry, timeout, and abort regressions.

### Completion verification

- focused AI contract and route tests;
- full `python -m pytest tests/` in the AI environment;
- focused backend proxy tests;
- backend `./gradlew test` and `./gradlew migrationSafetyCheck`;
- focused frontend unit tests and contract-generation check;
- frontend `npm run build`;
- `python3 scripts/validate_baseball_data_policy.py` from the workspace root;
- independent general code review, frontend review, and security review;
- intentional-change audit in all three repositories.

## Success Criteria

The phase is complete only when:

- existing chat and coach paths negotiate v2 through the approved header;
- absent and explicit v1 requests retain their legacy wire behavior;
- every v2 frame conforms to the closed `version/type/data` union;
- FastAPI runtime models deterministically generate the committed contract artifact;
- frontend generated types match the vendored AI contract and handwritten broad stream intersections are removed from chat and coach consumers;
- Spring forwards only the approved version header and never parses the SSE body;
- unsupported versions, malformed v2 events, and contract violations fail explicitly;
- the response exposes the resolved version;
- the frontend can roll back to v1 through configuration;
- all focused and full verification gates pass;
- no external baseball data path is added and `MANUAL_BASEBALL_DATA_REQUIRED` remains intact;
- unrelated user work is preserved and excluded from contract commits.
