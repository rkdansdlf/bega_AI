# Unified AI Stream HTTP Errors and v2 Default Design

## Context

The chat and coach streams already share an additive SSE v2 contract across FastAPI, Spring Boot, and React. FastAPI owns the stream event schemas, Spring forwards the event-version header and response bytes, and the frontend vendors the generated OpenAPI artifact. Version 1 remains available for rollback.

Two gaps remain before the browser can safely default to v2:

1. failures that occur before an SSE response starts do not have one wire shape; FastAPI's unsupported-version response is wrapped under `detail`, while Spring commonly returns its general `ApiResponse` error shape;
2. chat and coach decode those failures independently, discard different fields, and still default the browser request to event version 1.

The existing v2 in-stream `stream.error` event already provides `code`, `message`, nullable `detail`, and `retryable`. This design preserves that event and adds a canonical ordinary-HTTP error contract for failures that happen before streaming begins.

The AI worktree also contains unrelated security changes in `app/main.py`, configuration, dependencies, and tests. The backend worktree contains unrelated realtime changes. Implementation must preserve those edits, use narrowly scoped patches, and stage only this feature's hunks and files.

## Goals

1. Add a canonical `AiStreamHttpError` response for chat and coach failures before the stream starts.
2. Return that shape directly from FastAPI for unsupported event versions instead of wrapping it under `detail`.
3. Normalize Spring AI proxy pre-stream failures into the same safe shape.
4. Decode HTTP errors once in the frontend and use the result consistently in chat and coach.
5. Preserve `code`, `message`, `detail`, retryability, retry timing, and supported-version information through the UI-facing error boundary.
6. Change the browser's absent or empty event-version configuration default from version 1 to version 2.
7. Keep explicit `VITE_AI_EVENT_VERSION=1` as an immediate rollback with no endpoint change.
8. Preserve v1 streaming, the existing v2 envelope and event names, authentication, rate limits, cancellation, and baseball-data safety policy.

## Non-goals

- Removing event version 1 or setting a removal date.
- Introducing SSE v3 or changing the `version/type/data` v2 envelope.
- Adding or renaming chat or coach endpoints.
- Making Spring parse or transform SSE frames.
- Replacing the general backend `ApiResponse` contract outside AI stream endpoints.
- Exposing upstream validation bodies, exception messages, secrets, internal URLs, or stack traces.
- Changing coach analysis, chat generation, grounding, caching, or queue behavior.
- Adding external baseball crawling, scraping, web-search repair, external baseball APIs, or synthesized baseball facts.

## Chosen Architecture

FastAPI remains the canonical schema owner. It adds `AiStreamHttpError` alongside the existing stream contract models and exports the schema in `contracts/ai-stream-v2.openapi.json`. The additive contract artifact version becomes `2.1.0`.

Spring defines a matching serialization DTO and uses it only for failures before an AI stream is established. Successful SSE responses remain transparent byte streams. The frontend syncs the generated contract, decodes canonical and narrowly defined rolling-deployment compatibility shapes into one internal error type, and maps that type to the existing chat and coach UI error classes.

```text
React chat or coach request
  -> X-AI-Event-Version: 2 by default
  -> Spring authentication, limits, and proxy boundary
  -> FastAPI version negotiation

Success:
  -> unchanged SSE v2 stream

Failure before stream:
  -> AiStreamHttpError JSON
  -> shared frontend decoder
  -> existing chat/coach UI-facing error type
```

This is preferred over adding error fields to every SSE event because an HTTP failure and an in-stream terminal event occur at different protocol phases. It is preferred over keeping separate chat and coach decoders because those decoders have already drifted. Spring does not become the canonical owner; its DTO mirrors the generated AI schema and is verified by focused tests.

## Canonical HTTP Error Contract

The ordinary HTTP response body has this exact snake-case shape:

```json
{
  "code": "AI_EVENT_VERSION_UNSUPPORTED",
  "message": "지원하지 않는 AI 이벤트 버전입니다.",
  "detail": null,
  "retryable": false,
  "retry_after_seconds": null,
  "supported_versions": ["1", "2"]
}
```

The canonical Pydantic model is:

```python
class AiStreamHttpError(_StrictModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    detail: str | None = None
    retryable: bool
    retry_after_seconds: int | None = Field(default=None, ge=0)
    supported_versions: list[Literal["1", "2"]] = Field(default_factory=list)
```

Contract rules:

- `code`, `message`, and `retryable` are required.
- `detail` and `retry_after_seconds` are nullable and are serialized as `null` when absent.
- `supported_versions` is always an array. It is `["1", "2"]` only when version negotiation fails and is empty for unrelated failures.
- `retry_after_seconds` is a non-negative integer derived from a valid integer `Retry-After` value. An invalid or date-form header is not guessed and produces `null`.
- Unknown properties are rejected by the canonical Pydantic model.
- The body does not add a `success` property; HTTP status and the error type already express failure.
- Messages and details are public, stable text. Raw exception messages and upstream bodies are never copied into the response.

The content type is `application/json`. When retry timing is known, the ordinary `Retry-After` response header is preserved in addition to the body field.

## Status and Retry Semantics

The producer or proxy keeps the original meaningful HTTP status. Retryability is derived from the public failure class, not from arbitrary body input:

| Failure class | Typical status | `retryable` | Additional fields |
| --- | ---: | --- | --- |
| Unsupported event version | 406 | `false` | `supported_versions: ["1", "2"]` |
| Authentication or authorization | 401/403 | `false` | no internal auth detail |
| Invalid or oversized request | 400/413/422 | `false` | only safe public detail |
| Rate limit or capacity limit | 429 | `true` | valid `retry_after_seconds` when known |
| Upstream connection or service failure | 502/503 | `true` | stable service-unavailable message |
| Upstream timeout | 504 | `true` | stable timeout message |
| Unexpected proxy failure | 500 | `false` unless the failure is known transient | generic public message |

Spring must not trust an upstream `retryable` value for an unrecognized or malformed response. It derives the safe fallback from the status and known proxy exception type. FastAPI's 406 response is already canonical and may be forwarded after strict parsing; malformed upstream content is replaced by Spring's own canonical fallback.

## FastAPI Behavior

`negotiate_event_version` continues to resolve an absent header as v1 for non-browser and rolling-deployment compatibility. Explicit `1` resolves v1, explicit `2` resolves v2, and any other explicit value fails before `EventSourceResponse` is created.

Unsupported values raise a focused AI stream HTTP exception carrying status 406 and an `AiStreamHttpError`. A registered FastAPI exception handler serializes the model directly as the top-level JSON body and sets no stream headers. The generic FastAPI `HTTPException` response shape is not used for this case, avoiding the old `{"detail": {...}}` wrapper.

The exception and handler live in a focused streaming HTTP-error module. `app/main.py` only registers the handler. Because `app/main.py` already contains unrelated work, the implementation must stage only the registration hunk after reviewing the cached diff.

The deterministic exporter includes `AiStreamHttpError` as a named component and updates the contract metadata version to `2.1.0`. The existing event union and `StreamErrorData` schema remain wire-compatible.

## Spring Proxy Behavior

Spring introduces a dedicated AI stream error response DTO with the same snake-case fields and null/empty serialization behavior as the canonical model. It does not reuse or modify the platform-wide `ApiResponse` DTO.

For chat and coach requests that fail before response streaming starts, the proxy returns the dedicated DTO. This includes local proxy validation/capacity failures, upstream non-2xx responses, connection failures, and stream-header timeouts. The proxy:

1. preserves a meaningful safe status;
2. preserves a valid integer `Retry-After` header and mirrors it as `retry_after_seconds`;
3. strictly recognizes a canonical upstream error body;
4. forwards a recognized 406 `AI_EVENT_VERSION_UNSUPPORTED` payload without adding or removing supported versions;
5. replaces malformed, unknown, or unsafe upstream bodies with a status-derived public code and message;
6. never logs or returns internal service tokens or raw exception text.

Once an SSE response has started, Spring remains a byte-transparent proxy. It does not convert in-stream `stream.error` events into HTTP errors and cannot change the HTTP status after headers are committed.

## Frontend Error Decoding

The frontend adds one shared AI stream request-error module. Its canonical internal value carries:

```typescript
interface AiStreamErrorDetails {
  code: string;
  message: string;
  detail: string | null;
  retryable: boolean;
  retryAfterSeconds: number | null;
  supportedVersions: AiEventVersion[];
}
```

`AiStreamRequestError` extends `Error`, records the HTTP status, and exposes those details. A single async decoder reads an unsuccessful `Response` body once and returns that error.

For a coordinated rolling deployment, the decoder accepts only three input families:

1. the canonical top-level snake-case `AiStreamHttpError`;
2. the previous FastAPI `{"detail": {"code": ..., "supported_versions": [...]}}` unsupported-version response;
3. the existing Spring `ApiResponse` fields `code`, `message`, and optional `data` when the canonical proxy has not deployed yet.

All three are normalized into `AiStreamErrorDetails`. Unknown, invalid, HTML, or empty bodies use a status-derived safe fallback and never surface the raw body. The compatibility branches are explicitly tested and documented as deployment adapters, not alternate canonical contracts.

Chat and coach call this decoder for every non-OK pre-stream response. They no longer construct raw `HTTP error! status ..., body ...` messages or maintain separate payload parsers. Their existing UI-facing classes remain available:

- chat maps the shared details to `RateLimitError` or `ChatStreamEventError` as appropriate;
- coach maps them to `RateLimitError` or `CoachAnalyzeError` as appropriate;
- both preserve upstream `code`, `detail`, `retryable`, retry seconds, and supported versions instead of discarding them.

For failures after streaming begins, both modules normalize v2 `stream.error.data` into the same internal detail shape, using `null` retry timing and an empty supported-version list because those fields are not part of the existing SSE event schema. This unifies application handling without changing the v2 event wire contract.

`AI_EVENT_VERSION_HEADER` is exported once from `aiStreamContract.ts` and imported by chat and coach. Duplicate local constants are removed.

## v2 Default and Rollback

The browser configuration resolver follows this exact table:

| `VITE_AI_EVENT_VERSION` | Browser request version |
| --- | --- |
| absent | `2` |
| empty or whitespace | `2` |
| `2` | `2` |
| `1` | `1` |
| any other value | fail fast with `AiStreamContractError` |

The checked-in frontend `.env.example` documents `VITE_AI_EVENT_VERSION=2`. Real `.env` files are not modified.

Rollback requires only setting `VITE_AI_EVENT_VERSION=1` and rebuilding/redeploying the frontend. FastAPI still treats an absent header as v1, Spring still transparently forwards either supported header, and all existing v1 parsing remains in place. No endpoint, backend, or AI rollback is required.

## Contract Ownership and Release Order

The release dependency is:

```text
1. FastAPI model, handler, and contract artifact 2.1.0
2. Spring matching error normalization
3. Frontend artifact sync and shared decoder
4. Frontend default switch to v2
```

The frontend default switch is committed only after its tests cover explicit v1 rollback and compatibility with both old and new pre-stream error bodies. Independent deployment remains safe because the frontend sends a header already supported by the deployed v2 services, and the decoder accepts the immediately preceding error shapes during rollout.

The AI artifact is the source of truth. Frontend vendored JSON, metadata hash, and generated TypeScript must be regenerated, not hand-edited. Spring parity is enforced with serialization and proxy tests rather than by introducing a new shared runtime package.

## Observability and Safety

- Existing bounded stream-version metrics continue to distinguish v1 and v2 requests.
- Existing proxy metrics and logs continue to record status classes and failure categories without response bodies.
- Unsupported-version and fallback normalization logs use bounded error codes; they do not log user prompts, tokens, or raw upstream payloads.
- No new external service call or baseball-data source is introduced.
- Missing or inconsistent baseball facts continue to surface `MANUAL_BASEBALL_DATA_REQUIRED` through existing stream metadata and error behavior.

## Testing Strategy

Implementation follows RED-GREEN TDD in each repository.

### FastAPI

- model tests assert exact fields, nullability, forbidden extras, non-negative retry seconds, and closed supported versions;
- negotiation tests assert direct top-level 406 JSON and no `detail` wrapper;
- exporter tests assert deterministic artifact `2.1.0` freshness and named schema inclusion;
- existing v1/v2 stream tests prove event envelopes are unchanged.

### Spring Boot

- DTO tests assert exact snake-case JSON;
- service tests cover canonical upstream 406, malformed upstream fallback, 429 retry timing, 5xx/timeout retryability, and raw-body non-disclosure;
- controller tests cover local pre-stream failures and content type;
- existing proxy tests prove successful streams remain byte-transparent and version headers remain forwarded.

### React

- decoder tests cover canonical, old FastAPI, old Spring, invalid, HTML, and empty bodies;
- chat and coach tests prove both use the shared decoder and preserve fields;
- in-stream error tests prove code, detail, and retryability survive UI mapping;
- resolver tests prove absent/empty defaults to v2, explicit v1 rolls back, and invalid values fail;
- contract generation/check and production build remain green.

### Cross-service release gate

- regenerate and check the AI artifact;
- sync and check the frontend vendored artifact and generated types;
- run focused AI, backend, and frontend suites, then the appropriate full suites;
- run the repository baseball-data policy validator;
- inspect staged and committed diffs for unrelated dirty-worktree changes;
- obtain general, frontend, and security review before declaring the phase complete.

## Acceptance Criteria

1. FastAPI 406 uses top-level `AiStreamHttpError` JSON with no `detail` wrapper.
2. Spring pre-stream AI failures use the same fields and never expose raw upstream bodies.
3. Chat and coach share one decoder and preserve all approved error information.
4. Existing v2 SSE envelope/event schemas and successful byte streams remain unchanged.
5. Browser requests default to event version 2 when configuration is absent or empty.
6. Explicit `VITE_AI_EVENT_VERSION=1` still uses the existing v1 request and parser path.
7. The AI contract artifact and frontend generated types identify schema version `2.1.0` and pass drift checks.
8. Focused and release verification pass without staging or overwriting unrelated work.
