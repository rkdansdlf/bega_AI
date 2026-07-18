# Unified AI Stream HTTP Errors and v2 Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give FastAPI and Spring AI stream preflight failures one safe `AiStreamHttpError` shape, make chat and coach consume it through one frontend decoder, and default browser SSE negotiation to v2 while retaining explicit v1 rollback.

**Architecture:** FastAPI owns a Pydantic `AiStreamHttpError` and exports it in contract artifact `2.1.0`. Spring mirrors the shape for upstream and local pre-stream proxy failures without parsing successful SSE frames. React syncs the generated schema, normalizes canonical and immediately previous rollout shapes into one internal error value, maps it to existing UI error classes, and changes only the browser's missing/empty configuration default to v2.

**Tech Stack:** Python 3, FastAPI, Pydantic 2, pytest; Java 21, Spring Boot, WebClient, Jackson, MockMvc, JUnit 5; TypeScript 5, Vite, Node test runner, openapi-typescript.

## Global Constraints

- Existing paths remain `/api/ai/chat/stream`, `/api/ai/coach/analyze`, `/ai/chat/stream`, and `/ai/coach/analyze`.
- Keep SSE v1 and the existing SSE v2 `version/type/data` envelope; do not add v3 or remove v1.
- FastAPI still resolves an absent request header to v1; only the browser configuration default changes to v2.
- The canonical HTTP body fields are exactly `code`, `message`, `detail`, `retryable`, `retry_after_seconds`, and `supported_versions` in snake_case.
- `code`, `message`, and `retryable` are required; nullable fields serialize as `null`; `supported_versions` always serializes as an array.
- `supported_versions` is `["1", "2"]` only for `AI_EVENT_VERSION_UNSUPPORTED`; otherwise it is empty.
- Preserve only non-negative integer `Retry-After` seconds. Do not infer date-form values.
- Never expose raw upstream bodies, validation payloads, exception messages, secrets, tokens, or internal URLs.
- Spring remains byte-transparent for successful SSE responses and does not parse `stream.error` frames.
- Frontend accepts the old FastAPI `detail` wrapper and old Spring `ApiResponse` only as tested rolling-deployment adapters.
- `VITE_AI_EVENT_VERSION=1` remains a tested rollback. Real `.env` files are not modified.
- Do not add external baseball crawling, scraping, web-search repair, external baseball APIs, or synthesized facts. Preserve `MANUAL_BASEBALL_DATA_REQUIRED`.
- Do not install dependencies or change package/lock files.
- Preserve all unrelated dirty work. In AI, `.env.example`, `app/config.py`, `app/deps.py`, `app/main.py`, and several security tests are already modified; only this plan's `app/main.py` hunk may be staged interactively.
- Backend has unrelated realtime work elsewhere; stage only exact AI proxy files. Frontend target files are currently clean.
- Every production task follows RED, observed expected failure, minimal GREEN, focused tests, cached-diff inspection, and a narrow commit.

## File and Responsibility Map

### AI repository (`bega_AI`)

- Modify `app/contracts/stream_events_v2.py`: canonical `AiStreamHttpError` Pydantic model; existing SSE event union remains unchanged.
- Create `app/streaming/http_errors.py`: focused exception, handler, and FastAPI installer.
- Modify `app/streaming/versioned_sse.py`: raise the focused 406 exception from negotiation.
- Modify `app/main.py`: register the handler; partial-stage only this hunk.
- Modify `scripts/export_ai_stream_contract.py`: export the named HTTP error schema and set contract version `2.1.0`.
- Regenerate `contracts/ai-stream-v2.openapi.json`: deterministic canonical artifact.
- Modify `tests/test_stream_events_v2.py`: model constraints.
- Modify `tests/test_ai_stream_contract_export.py`: artifact version and named component.
- Modify `tests/test_versioned_sse.py`: focused exception assertions.
- Create `tests/test_stream_http_errors.py`: direct top-level HTTP response integration.
- Modify `tests/test_chat_stream_versioning.py`: route-level direct 406 and existing v1/v2 behavior.
- Modify `tests/test_coach_contracts.py`: route-level direct 406 and existing v1/v2 behavior.

### Backend repository (`bega_backend/BEGA_PROJECT`)

- Create `src/main/java/com/example/ai/dto/AiStreamHttpErrorResponse.java`: exact JSON DTO and safe status-based factories.
- Modify `src/main/java/com/example/ai/service/AiProxyService.java`: normalize upstream stream failures and valid integer retry timing; leave byte endpoints unchanged.
- Modify `src/main/java/com/example/ai/controller/AiProxyController.java`: map local stream setup `AiProxyException` values to the canonical body while preserving permit cleanup and coach metrics.
- Create `src/test/java/com/example/ai/dto/AiStreamHttpErrorResponseTest.java`: exact serialization.
- Modify `src/test/java/com/example/ai/service/AiProxyServiceTest.java`: canonical upstream/fallback/retry tests and successful-body transparency.
- Modify `src/test/java/com/example/ai/controller/AiProxyControllerTest.java`: local 413, capacity, connection, and timeout response shape.

### Frontend repository (`bega_frontend`)

- Regenerate `contracts/ai-stream-v2.openapi.json`, `contracts/ai-stream-v2.metadata.json`, and `src/api/generated/aiStreamV2.ts`: vendored `2.1.0` contract and generated type.
- Modify `src/api/aiStreamContract.ts`: export the one negotiation-header constant and change missing/empty resolution to v2 in the final rollout task.
- Modify `src/api/aiStreamContract.test.ts`: header export and version resolver characterization.
- Create `src/api/aiStreamError.ts`: shared HTTP decoder, rollout adapters, in-stream normalizer, `AiStreamRequestError`, and shared `RateLimitError`.
- Create `src/api/aiStreamError.test.ts`: canonical, compatibility, fallback, and no-raw-body cases.
- Modify `src/api/chatbot.ts` and `src/api/chatbot.test.ts`: shared pre-stream decoder and complete v2 `stream.error` preservation.
- Modify `src/api/coach.ts` and `src/api/coach.test.ts`: remove custom body parser, share decoder, and preserve error details.
- Modify `.env.example`: document browser default version 2 and explicit version 1 rollback.

### Integrated documentation (`KBO_platform` root)

- Modify `docs/API_REFERENCE.md`: document canonical pre-stream error JSON, contract version `2.1.0`, browser v2 default, and v1 rollback.
- Create `.superpowers/sdd/ai-stream-error-progress.md`: record per-task commits and verification without touching the unrelated `.superpowers/sdd/progress.md`.

---

### Task 1: Add the AI-owned HTTP error schema and artifact 2.1.0

**Files:**
- Modify: `app/contracts/stream_events_v2.py`
- Modify: `scripts/export_ai_stream_contract.py`
- Modify: `tests/test_stream_events_v2.py`
- Modify: `tests/test_ai_stream_contract_export.py`
- Regenerate: `contracts/ai-stream-v2.openapi.json`

**Interfaces:**
- Consumes: existing `_StrictModel`, `Literal`, `Field`, `_extract_schema`.
- Produces: `AiStreamHttpError`; OpenAPI component `components.schemas.AiStreamHttpError`; artifact info version `2.1.0`.

- [ ] **Step 1: Write failing Pydantic model tests**

Add these tests to `tests/test_stream_events_v2.py`:

```python
from pydantic import ValidationError

from app.contracts.stream_events_v2 import AiStreamHttpError


def test_ai_stream_http_error_has_exact_canonical_shape() -> None:
    error = AiStreamHttpError(
        code="AI_EVENT_VERSION_UNSUPPORTED",
        message="지원하지 않는 AI 이벤트 버전입니다.",
        retryable=False,
        supported_versions=["1", "2"],
    )

    assert error.model_dump(mode="json") == {
        "code": "AI_EVENT_VERSION_UNSUPPORTED",
        "message": "지원하지 않는 AI 이벤트 버전입니다.",
        "detail": None,
        "retryable": False,
        "retry_after_seconds": None,
        "supported_versions": ["1", "2"],
    }


def test_ai_stream_http_error_rejects_invalid_retry_and_versions() -> None:
    with pytest.raises(ValidationError):
        AiStreamHttpError(
            code="AI_UPSTREAM_RATE_LIMITED",
            message="요청 한도를 초과했습니다.",
            retryable=True,
            retry_after_seconds=-1,
        )
    with pytest.raises(ValidationError):
        AiStreamHttpError(
            code="AI_EVENT_VERSION_UNSUPPORTED",
            message="지원하지 않는 버전입니다.",
            retryable=False,
            supported_versions=["3"],
        )


def test_ai_stream_http_error_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AiStreamHttpError(
            code="AI_ERROR",
            message="오류",
            retryable=False,
            success=False,
        )
```

- [ ] **Step 2: Run the model tests and observe RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_stream_events_v2.py -q
```

Expected: collection fails because `AiStreamHttpError` does not exist.

- [ ] **Step 3: Implement the canonical Pydantic model**

Add immediately before `StreamErrorData` in `app/contracts/stream_events_v2.py`:

```python
class AiStreamHttpError(_StrictModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    detail: str | None = None
    retryable: bool
    retry_after_seconds: int | None = Field(default=None, ge=0)
    supported_versions: list[Literal["1", "2"]] = Field(default_factory=list)
```

Do not add this model to `AiStreamV2Event`; it describes ordinary HTTP failures, not SSE frames.

- [ ] **Step 4: Add failing exporter assertions**

Update `test_contract_document_contains_named_requests_and_event_union` in `tests/test_ai_stream_contract_export.py`:

```python
assert document["info"]["version"] == "2.1.0"
assert "AiStreamHttpError" in schemas
http_error = schemas["AiStreamHttpError"]
assert set(http_error["required"]) == {"code", "message", "retryable"}
assert http_error["additionalProperties"] is False
```

- [ ] **Step 5: Run the exporter test and observe RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_ai_stream_contract_export.py::test_contract_document_contains_named_requests_and_event_union -q
```

Expected: FAIL because the document is still `2.0.0` and lacks `AiStreamHttpError`.

- [ ] **Step 6: Export the named model and bump the additive artifact version**

In `scripts/export_ai_stream_contract.py`, import `AiStreamHttpError`, generate its schema, and add it through `_extract_schema`:

```python
from app.contracts.stream_events_v2 import AiStreamHttpError, event_schema

http_error = AiStreamHttpError.model_json_schema(
    ref_template="#/components/schemas/{model}"
)
schemas["AiStreamHttpError"] = _extract_schema(http_error, schemas)
```

Set `info.version` to `2.1.0`. Keep stable sorting and all existing event schemas unchanged.

- [ ] **Step 7: Regenerate and verify the artifact**

Run:

```bash
./.venv/bin/python scripts/export_ai_stream_contract.py
./.venv/bin/python scripts/export_ai_stream_contract.py --check
./.venv/bin/python -m pytest tests/test_stream_events_v2.py tests/test_ai_stream_contract_export.py -q
```

Expected: artifact reports current and all focused tests pass.

- [ ] **Step 8: Commit only Task 1 files**

```bash
git add app/contracts/stream_events_v2.py scripts/export_ai_stream_contract.py tests/test_stream_events_v2.py tests/test_ai_stream_contract_export.py contracts/ai-stream-v2.openapi.json
git diff --cached --check
git diff --cached --name-only
git commit -m "feat: define AI stream HTTP error contract"
```

Expected staged names: exactly the five files listed above.

---

### Task 2: Return a direct top-level FastAPI 406 response

**Files:**
- Create: `app/streaming/http_errors.py`
- Modify: `app/streaming/versioned_sse.py`
- Modify: `app/main.py` (partial stage only)
- Modify: `tests/test_versioned_sse.py`
- Create: `tests/test_stream_http_errors.py`
- Modify: `tests/test_chat_stream_versioning.py`
- Modify: `tests/test_coach_contracts.py`

**Interfaces:**
- Consumes: `AiStreamHttpError`, `FastAPI.add_exception_handler`, `JSONResponse`.
- Produces: `AiStreamHttpException(status_code: int, error: AiStreamHttpError, headers: Mapping[str, str] | None)`; `install_ai_stream_http_error_handler(app: FastAPI) -> None`.

- [ ] **Step 1: Change the negotiation unit test to the focused exception**

Replace the `HTTPException` assertion in `tests/test_versioned_sse.py` with:

```python
from app.streaming.http_errors import AiStreamHttpException


def test_unsupported_version_is_406() -> None:
    with pytest.raises(AiStreamHttpException) as raised:
        negotiate_event_version("3", endpoint="coach")

    assert raised.value.status_code == 406
    assert raised.value.error.model_dump(mode="json") == {
        "code": "AI_EVENT_VERSION_UNSUPPORTED",
        "message": "지원하지 않는 AI 이벤트 버전입니다.",
        "detail": None,
        "retryable": False,
        "retry_after_seconds": None,
        "supported_versions": ["1", "2"],
    }
```

- [ ] **Step 2: Add a failing direct HTTP integration test**

Create `tests/test_stream_http_errors.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.streaming.http_errors import install_ai_stream_http_error_handler
from app.streaming.versioned_sse import negotiate_event_version


def test_unsupported_version_returns_top_level_canonical_json() -> None:
    app = FastAPI()
    install_ai_stream_http_error_handler(app)

    @app.get("/stream")
    async def stream(version: str) -> dict[str, int]:
        return {"version": negotiate_event_version(version, endpoint="chat")}

    response = TestClient(app).get("/stream", params={"version": "3"})

    assert response.status_code == 406
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "code": "AI_EVENT_VERSION_UNSUPPORTED",
        "message": "지원하지 않는 AI 이벤트 버전입니다.",
        "detail": None,
        "retryable": False,
        "retry_after_seconds": None,
        "supported_versions": ["1", "2"],
    }
```

Update `test_unsupported_version_fails_before_stream` in `tests/test_chat_stream_versioning.py` and `test_endpoint_stream_rejects_unsupported_version_before_work` in `tests/test_coach_contracts.py` to assert the same top-level body and explicitly assert `not isinstance(response.json().get("detail"), dict)`. Keep their existing assertions that producer work did not start.

- [ ] **Step 3: Run both tests and observe RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_versioned_sse.py::test_unsupported_version_is_406 tests/test_stream_http_errors.py -q
```

Expected: collection fails because `app.streaming.http_errors` does not exist.

- [ ] **Step 4: Implement the focused exception and installer**

Create `app/streaming/http_errors.py`:

```python
"""Safe ordinary-HTTP errors for AI stream setup failures."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.contracts.stream_events_v2 import AiStreamHttpError


class AiStreamHttpException(Exception):
    def __init__(
        self,
        status_code: int,
        error: AiStreamHttpError,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(error.code)
        self.status_code = status_code
        self.error = error
        self.headers = dict(headers or {})


async def ai_stream_http_exception_handler(
    _request: Request,
    exception: AiStreamHttpException,
) -> JSONResponse:
    return JSONResponse(
        status_code=exception.status_code,
        content=exception.error.model_dump(mode="json"),
        headers=exception.headers,
    )


def install_ai_stream_http_error_handler(app: FastAPI) -> None:
    app.add_exception_handler(
        AiStreamHttpException,
        ai_stream_http_exception_handler,
    )
```

- [ ] **Step 5: Raise the canonical 406 exception**

Remove the FastAPI `HTTPException` import from `app/streaming/versioned_sse.py`, import `AiStreamHttpError` and `AiStreamHttpException`, and replace the final raise with:

```python
raise AiStreamHttpException(
    status_code=406,
    error=AiStreamHttpError(
        code="AI_EVENT_VERSION_UNSUPPORTED",
        message="지원하지 않는 AI 이벤트 버전입니다.",
        retryable=False,
        supported_versions=list(SUPPORTED_EVENT_VERSIONS),
    ),
)
```

- [ ] **Step 6: Register the handler in the real app**

Import `install_ai_stream_http_error_handler` in `app/main.py` and call it immediately after `FastAPI(...)` construction:

```python
install_ai_stream_http_error_handler(app)
```

Do not alter or stage the existing internal-router security changes in this dirty file.

- [ ] **Step 7: Run focused and stream regression tests**

```bash
./.venv/bin/python -m pytest tests/test_versioned_sse.py tests/test_stream_http_errors.py tests/test_chat_stream_versioning.py tests/test_coach_contracts.py -q
```

Expected: all tests pass and existing v1/v2 stream event assertions remain unchanged.

- [ ] **Step 8: Partial-stage `app/main.py` and commit**

```bash
git add app/streaming/http_errors.py app/streaming/versioned_sse.py tests/test_versioned_sse.py tests/test_stream_http_errors.py tests/test_chat_stream_versioning.py tests/test_coach_contracts.py
git add -p app/main.py
git diff --cached --check
git diff --cached --name-only
git diff --cached -- app/main.py
git commit -m "feat: standardize AI stream negotiation errors"
```

Accept only the handler import and `install_ai_stream_http_error_handler(app)` hunk from `app/main.py`. Reject every security/configuration hunk.

---

### Task 3: Normalize upstream Spring stream failures

**Files:**
- Create: `src/main/java/com/example/ai/dto/AiStreamHttpErrorResponse.java`
- Modify: `src/main/java/com/example/ai/service/AiProxyService.java`
- Create: `src/test/java/com/example/ai/dto/AiStreamHttpErrorResponseTest.java`
- Modify: `src/test/java/com/example/ai/service/AiProxyServiceTest.java`

**Interfaces:**
- Consumes: `HttpStatusCode`, filtered upstream headers, optional upstream bytes, Jackson `ObjectMapper`.
- Produces: exact JSON DTO; `buildStandardizedStreamErrorBody(HttpStatusCode, HttpHeaders, byte[])`; public `serializeStreamError(AiStreamHttpErrorResponse)` for the controller; integer `retry_after_seconds`; canonical 406 adapter.

- [ ] **Step 1: Write a failing exact-serialization DTO test**

Create `AiStreamHttpErrorResponseTest.java` with:

```java
package com.example.ai.dto;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import org.junit.jupiter.api.Test;

class AiStreamHttpErrorResponseTest {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void serializesExactCanonicalSnakeCaseShape() throws Exception {
        var response = new AiStreamHttpErrorResponse(
                "AI_EVENT_VERSION_UNSUPPORTED",
                "지원하지 않는 AI 이벤트 버전입니다.",
                null,
                false,
                null,
                List.of("1", "2"));

        assertThat(objectMapper.readTree(objectMapper.writeValueAsBytes(response)))
                .isEqualTo(objectMapper.readTree("""
                        {
                          "code":"AI_EVENT_VERSION_UNSUPPORTED",
                          "message":"지원하지 않는 AI 이벤트 버전입니다.",
                          "detail":null,
                          "retryable":false,
                          "retry_after_seconds":null,
                          "supported_versions":["1","2"]
                        }
                        """));
    }
}
```

- [ ] **Step 2: Run the DTO test and observe RED**

```bash
./gradlew test --tests "*AiStreamHttpErrorResponseTest"
```

Expected: compilation fails because the DTO does not exist.

- [ ] **Step 3: Implement the exact DTO**

Create `AiStreamHttpErrorResponse.java`:

```java
package com.example.ai.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

@JsonInclude(JsonInclude.Include.ALWAYS)
public record AiStreamHttpErrorResponse(
        String code,
        String message,
        String detail,
        boolean retryable,
        @JsonProperty("retry_after_seconds") Long retryAfterSeconds,
        @JsonProperty("supported_versions") List<String> supportedVersions) {

    public AiStreamHttpErrorResponse {
        supportedVersions = supportedVersions == null ? List.of() : List.copyOf(supportedVersions);
    }
}
```

- [ ] **Step 4: Replace upstream stream-error tests with canonical assertions**

In `AiProxyServiceTest`, characterize these cases with JSON-tree equality:

```java
@Test
void forwardJsonStreamNormalizesCanonicalUnsupportedVersion() throws Exception {
    String body = """
            {"code":"AI_EVENT_VERSION_UNSUPPORTED","message":"지원하지 않는 AI 이벤트 버전입니다.","detail":null,"retryable":false,"retry_after_seconds":null,"supported_versions":["1","2"]}
            """;
    server = startServer("/ai/chat/stream", exchange ->
            writeResponse(exchange, 406, body, "application/json"));

    ProxyStreamResponse response = newService(Duration.ofSeconds(5), "stream-token")
            .forwardJsonStream("/ai/chat/stream", "{\"test\":true}", "3");

    assertThat(response.status().value()).isEqualTo(406);
    assertThat(json(response.errorBody())).isEqualTo(json(body));
}

@Test
void forwardJsonStreamAdaptsLegacyWrappedUnsupportedVersion() throws Exception {
    server = startServer("/ai/chat/stream", exchange -> writeResponse(
            exchange,
            406,
            "{\"detail\":{\"code\":\"AI_EVENT_VERSION_UNSUPPORTED\",\"supported_versions\":[\"1\",\"2\"]}}",
            "application/json"));

    ProxyStreamResponse response = newService(Duration.ofSeconds(5), "stream-token")
            .forwardJsonStream("/ai/chat/stream", "{}", "3");

    assertThat(json(response.errorBody()).get("code").asText())
            .isEqualTo("AI_EVENT_VERSION_UNSUPPORTED");
    assertThat(json(response.errorBody()).get("detail").isNull()).isTrue();
}

@Test
void forwardJsonStreamDoesNotExposeMalformedUpstreamBody() throws Exception {
    server = startServer("/ai/coach/analyze", exchange -> writeResponse(
            exchange, 503, "secret-bearing upstream body", "text/plain"));

    ProxyStreamResponse response = newService(Duration.ofSeconds(5), "stream-token")
            .forwardJsonStream("/ai/coach/analyze", "{}");
    String body = new String(response.errorBody(), StandardCharsets.UTF_8);

    assertThat(body).contains("\"code\":\"AI_UPSTREAM_UNAVAILABLE\"");
    assertThat(body).contains("\"retryable\":true");
    assertThat(body).doesNotContain("secret-bearing");
}
```

Add a local `json(byte[] value)` helper that calls the test `ObjectMapper.readTree`.

Add a 429 case whose upstream response has `Retry-After: 37` and assert both the filtered response header and body `retry_after_seconds` equal 37. Add an invalid/date-form `Retry-After` case and assert the body field is null.

- [ ] **Step 5: Run service tests and observe RED**

```bash
./gradlew test --tests "*AiProxyServiceTest"
```

Expected: canonical field assertions fail because the service still emits `ApiResponse` or raw 406 bytes.

- [ ] **Step 6: Implement safe upstream normalization**

In `AiProxyService`:

1. Keep `buildStandardizedErrorBody` unchanged for non-stream byte requests.
2. Change the `WebClientResponseException` stream branch to always set JSON content type and call:

```java
buildStandardizedStreamErrorBody(e.getStatusCode(), headers, e.getResponseBodyAsByteArray())
```

3. Implement a strict 406 adapter using `ObjectMapper.readTree`. Accept only `AI_EVENT_VERSION_UNSUPPORTED` from either the top-level object or an object-valued `detail`, and always emit the canonical safe message, `retryable=false`, null detail/timing, and `List.of("1", "2")`.
4. For all other statuses ignore the upstream body and map safe codes/messages:

```java
private AiStreamHttpErrorResponse resolveStreamError(HttpStatusCode status, HttpHeaders headers) {
    AiUpstreamError safe = resolveUpstreamError(status);
    return new AiStreamHttpErrorResponse(
            safe.code(),
            safe.message(),
            null,
            status.value() == 429 || status.is5xxServerError(),
            parseRetryAfterSeconds(headers.getFirst(HttpHeaders.RETRY_AFTER)),
            List.of());
}

private Long parseRetryAfterSeconds(String value) {
    if (!StringUtils.hasText(value) || !value.trim().matches("\\d+")) {
        return null;
    }
    try {
        return Long.parseLong(value.trim());
    } catch (NumberFormatException ignored) {
        return null;
    }
}
```

Serialize with the existing object mapper. The fallback literal must contain all six canonical fields, including explicit nulls and `supported_versions:[]`.

Expose the serializer used by both upstream and local setup errors:

```java
public byte[] serializeStreamError(AiStreamHttpErrorResponse error) {
    try {
        return OBJECT_MAPPER.writeValueAsBytes(error);
    } catch (JsonProcessingException exception) {
        log.error("Failed to serialize AI stream error response. code={}", error.code(), exception);
        String fallback = "{\"code\":\"AI_STREAM_REQUEST_FAILED\",\"message\":\"AI 요청을 처리할 수 없습니다. 잠시 후 다시 시도해주세요.\",\"detail\":null,\"retryable\":false,\"retry_after_seconds\":null,\"supported_versions\":[]}";
        return fallback.getBytes(StandardCharsets.UTF_8);
    }
}
```

`buildStandardizedStreamErrorBody` calls this method after normalization; it never appends the upstream byte array to logs or fallback text.

- [ ] **Step 7: Run DTO and service tests, then prove success transparency**

```bash
./gradlew test --tests "*AiStreamHttpErrorResponseTest" --tests "*AiProxyServiceTest"
```

Expected: all tests pass, including the existing byte-for-byte successful SSE test and event-version retry test.

- [ ] **Step 8: Commit only Task 3 backend files**

From `bega_backend/BEGA_PROJECT`:

```bash
git add src/main/java/com/example/ai/dto/AiStreamHttpErrorResponse.java src/main/java/com/example/ai/service/AiProxyService.java src/test/java/com/example/ai/dto/AiStreamHttpErrorResponseTest.java src/test/java/com/example/ai/service/AiProxyServiceTest.java
git diff --cached --check
git diff --cached --name-only
git commit -m "feat: normalize AI stream upstream errors"
```

---

### Task 4: Normalize local Spring stream setup failures

**Files:**
- Modify: `src/main/java/com/example/ai/controller/AiProxyController.java`
- Modify: `src/test/java/com/example/ai/controller/AiProxyControllerTest.java`

**Interfaces:**
- Consumes: `AiProxyException`, `AiStreamHttpErrorResponse`, existing permit and coach duration cleanup.
- Produces: canonical `StreamingResponseBody` for local payload, capacity, connection, configuration, and timeout failures before stream headers are returned.

- [ ] **Step 1: Write failing controller assertions for local 413 and 504**

Change the existing coach oversized-payload test to assert:

```java
.andExpect(status().isPayloadTooLarge())
.andExpect(jsonPath("$.code").value(AiProxyRequestLimits.PAYLOAD_TOO_LARGE_CODE))
.andExpect(jsonPath("$.message").value("AI 요청 본문이 너무 큽니다."))
.andExpect(jsonPath("$.detail").isEmpty())
.andExpect(jsonPath("$.retryable").value(false))
.andExpect(jsonPath("$.retry_after_seconds").isEmpty())
.andExpect(jsonPath("$.supported_versions").isArray())
.andExpect(jsonPath("$.success").doesNotExist());
```

Add a chat-stream test where `forwardJsonStream` throws `AiProxyException(HttpStatus.GATEWAY_TIMEOUT, "AI_UPSTREAM_TIMEOUT", "AI 응답 시간이 초과되었습니다.")` and assert status 504, `retryable=true`, empty supported versions, and the stable message.

Add a capacity test where `streamConcurrencyLimiter.acquire("chat_stream")` throws status 503/code `AI_PROXY_STREAMS_BUSY`; assert the same canonical fields and verify no proxy call.

Because the controller's declared body remains `StreamingResponseBody`, collect each local error with `request().asyncStarted()` and assert it through `asyncDispatch(result)`, matching the existing upstream non-2xx controller tests:

```java
MvcResult result = limitedMockMvc.perform(post("/api/ai/coach/analyze")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"home_team_id\":\"HH\",\"away_team_id\":\"SS\"}"))
        .andExpect(request().asyncStarted())
        .andReturn();

limitedMockMvc.perform(asyncDispatch(result))
        .andExpect(status().isPayloadTooLarge())
        .andExpect(jsonPath("$.code").value(AiProxyRequestLimits.PAYLOAD_TOO_LARGE_CODE));
```

- [ ] **Step 2: Run controller tests and observe RED**

```bash
./gradlew test --tests "*AiProxyControllerTest"
```

Expected: local exceptions still resolve through the general `ApiResponse` handler, so `success` exists and canonical fields are absent.

- [ ] **Step 3: Add a local-exception stream response helper**

Refactor only `chatStream` and `coachAnalyze` so their outer try blocks include payload validation and permit acquisition. Catch `AiProxyException` before the generic runtime catch. Convert it through:

```java
private ResponseEntity<StreamingResponseBody> toLocalStreamErrorResponse(AiProxyException exception) {
    AiStreamHttpErrorResponse error = new AiStreamHttpErrorResponse(
            exception.getCode(),
            exception.getStatus().is5xxServerError()
                    ? safeProxyMessage(exception.getCode())
                    : exception.getMessage(),
            null,
            isRetryableProxySetupFailure(exception),
            null,
            List.of());
    byte[] body = serializeStreamError(error);
    return ResponseEntity.status(exception.getStatus())
            .contentType(MediaType.APPLICATION_JSON)
            .body(outputStream -> outputStream.write(body));
}
```

Both methods keep `Permit streamPermit = null` outside the try. In the `AiProxyException` catch, close it only when non-null before returning the canonical response. In the generic runtime catch, keep the existing close-and-rethrow behavior. Coach records setup duration with `exception.getStatus().value()` before returning.

Use explicit stable messages for `AI_UPSTREAM_TIMEOUT`, `AI_UPSTREAM_CONNECTION_FAILED`, `AI_PROXY_STREAMS_BUSY`, service URL/configuration failures, and a generic fallback. Do not use arbitrary `exception.getMessage()` for 5xx. `isRetryableProxySetupFailure` returns true only for known transient codes (`AI_UPSTREAM_TIMEOUT`, `AI_UPSTREAM_CONNECTION_FAILED`, `AI_UPSTREAM_EMPTY_RESPONSE`, and `AI_PROXY_STREAMS_BUSY`) or status 429; configuration and internal-auth misconfiguration codes are false even when their status is 503.

Preserve these cleanup invariants:

- close an acquired permit exactly once;
- do not close when acquisition failed;
- record coach setup duration once using the error status;
- do not call `writeStream` for an error body;
- keep the successful path unchanged.

Expose a public `serializeStreamError(AiStreamHttpErrorResponse)` method from `AiProxyService` and cover it in Task 3 tests rather than creating another mapper in the controller package.

- [ ] **Step 4: Run controller and service tests**

```bash
./gradlew test --tests "*AiProxyControllerTest" --tests "*AiProxyServiceTest" --tests "*AiStreamHttpErrorResponseTest"
```

Expected: all pass; existing coach metrics and successful async-stream tests remain green.

- [ ] **Step 5: Commit only Task 4 files**

```bash
git add src/main/java/com/example/ai/controller/AiProxyController.java src/test/java/com/example/ai/controller/AiProxyControllerTest.java
git diff --cached --check
git diff --cached --name-only
git commit -m "feat: standardize local AI stream failures"
```

---

### Task 5: Sync the contract and add the shared frontend decoder

**Files:**
- Regenerate: `contracts/ai-stream-v2.openapi.json`
- Regenerate: `contracts/ai-stream-v2.metadata.json`
- Regenerate: `src/api/generated/aiStreamV2.ts`
- Modify: `src/api/aiStreamContract.ts`
- Create: `src/api/aiStreamError.ts`
- Create: `src/api/aiStreamError.test.ts`

**Interfaces:**
- Consumes: generated `AiStreamHttpError`, `StreamErrorData`, `AiEventVersion`.
- Produces: `AI_EVENT_VERSION_HEADER`; `AiStreamErrorDetails`; `AiStreamRequestError`; `RateLimitError`; `decodeAiStreamHttpError(response: Response) -> Promise<AiStreamRequestError>`; `normalizeAiStreamEventError(data: StreamErrorData) -> AiStreamErrorDetails`.

- [ ] **Step 1: Sync the AI artifact and verify generated types**

From `bega_frontend`:

```bash
npm run api:ai-stream:sync
npm run api:ai-stream:check
```

Expected: metadata reports schema version `2.1.0`; generated components include `AiStreamHttpError`; source and vendored SHA-256 values match.

- [ ] **Step 2: Export the single header constant**

Add to `src/api/aiStreamContract.ts`:

```typescript
export const AI_EVENT_VERSION_HEADER = 'X-AI-Event-Version';
```

Do not switch the default in this task.

- [ ] **Step 3: Write failing shared-decoder tests**

Create `src/api/aiStreamError.test.ts` with focused cases:

```typescript
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  decodeAiStreamHttpError,
  normalizeAiStreamEventError,
  RateLimitError,
} from './aiStreamError';

test('decodes the canonical top-level error without losing fields', async () => {
  const error = await decodeAiStreamHttpError(new Response(JSON.stringify({
    code: 'AI_EVENT_VERSION_UNSUPPORTED',
    message: '지원하지 않는 AI 이벤트 버전입니다.',
    detail: null,
    retryable: false,
    retry_after_seconds: null,
    supported_versions: ['1', '2'],
  }), { status: 406, headers: { 'content-type': 'application/json' } }));

  assert.equal(error.statusCode, 406);
  assert.equal(error.code, 'AI_EVENT_VERSION_UNSUPPORTED');
  assert.deepEqual(error.supportedVersions, ['1', '2']);
  assert.equal(error.retryable, false);
});

test('adapts old FastAPI detail wrapper', async () => {
  const error = await decodeAiStreamHttpError(new Response(JSON.stringify({
    detail: { code: 'AI_EVENT_VERSION_UNSUPPORTED', supported_versions: ['1', '2'] },
  }), { status: 406 }));
  assert.equal(error.code, 'AI_EVENT_VERSION_UNSUPPORTED');
  assert.deepEqual(error.supportedVersions, ['1', '2']);
});

test('adapts old Spring ApiResponse', async () => {
  const error = await decodeAiStreamHttpError(new Response(JSON.stringify({
    success: false,
    code: 'AI_UPSTREAM_TIMEOUT',
    message: '서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.',
  }), { status: 504 }));
  assert.equal(error.code, 'AI_UPSTREAM_TIMEOUT');
  assert.equal(error.retryable, true);
});

test('invalid body uses safe fallback and never exposes raw text', async () => {
  const error = await decodeAiStreamHttpError(new Response(
    'secret-bearing upstream html',
    { status: 503, headers: { 'Retry-After': '17' } },
  ));
  assert.equal(error.code, 'AI_STREAM_REQUEST_FAILED');
  assert.equal(error.retryAfterSeconds, 17);
  assert.equal(error.retryable, true);
  assert.doesNotMatch(error.message, /secret-bearing/);
});

test('normalizes in-stream error without inventing HTTP-only fields', () => {
  assert.deepEqual(normalizeAiStreamEventError({
    code: 'COACH_ANALYSIS_FAILED',
    message: '분석 실패',
    detail: '안전한 상세',
    retryable: true,
  }), {
    code: 'COACH_ANALYSIS_FAILED',
    message: '분석 실패',
    detail: '안전한 상세',
    retryable: true,
    retryAfterSeconds: null,
    supportedVersions: [],
  });
});
```

Also test canonical body `retry_after_seconds` taking precedence over a header, invalid/date-form `Retry-After` producing null, invalid supported versions being discarded, and `new RateLimitError(details)` preserving details plus a default 10-second retry when timing is absent.

- [ ] **Step 4: Run decoder tests and observe RED**

```bash
node --import tsx --test src/api/aiStreamError.test.ts
```

Expected: module-not-found failure.

- [ ] **Step 5: Implement the shared decoder**

Create `src/api/aiStreamError.ts` with these exact public shapes:

```typescript
import type { components } from './generated/aiStreamV2';
import type { AiEventVersion } from './aiStreamContract';

type StreamErrorData = components['schemas']['StreamErrorData'];

export interface AiStreamErrorDetails {
  code: string;
  message: string;
  detail: string | null;
  retryable: boolean;
  retryAfterSeconds: number | null;
  supportedVersions: AiEventVersion[];
}

export class AiStreamRequestError extends Error implements AiStreamErrorDetails {
  readonly statusCode: number;
  readonly code: string;
  readonly detail: string | null;
  readonly retryable: boolean;
  readonly retryAfterSeconds: number | null;
  readonly supportedVersions: AiEventVersion[];

  constructor(statusCode: number, details: AiStreamErrorDetails) {
    super(details.message);
    this.name = 'AiStreamRequestError';
    this.statusCode = statusCode;
    this.code = details.code;
    this.detail = details.detail;
    this.retryable = details.retryable;
    this.retryAfterSeconds = details.retryAfterSeconds;
    this.supportedVersions = [...details.supportedVersions];
  }
}
```

Implement narrow record/string/boolean/non-negative-integer guards. Parse `response.text()` once. Select canonical top-level first, old object-valued `detail` second, and old `ApiResponse` third. Use status-derived safe fallbacks:

```typescript
const fallbackForStatus = (status: number): AiStreamErrorDetails => ({
  code: 'AI_STREAM_REQUEST_FAILED',
  message: status === 504
    ? 'AI 서비스 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.'
    : status === 503 || status === 502
      ? 'AI 서비스가 현재 사용할 수 없습니다. 잠시 후 다시 시도해주세요.'
      : 'AI 요청을 처리할 수 없습니다. 잠시 후 다시 시도해주세요.',
  detail: null,
  retryable: status === 429 || status >= 500,
  retryAfterSeconds: null,
  supportedVersions: [],
});
```

`normalizeAiStreamEventError` maps the four existing SSE fields and supplies null/empty HTTP-only fields. Move `RateLimitError` here, preserve its `name`, public `retryAfterSeconds`, and Korean display message, and retain a 10-second fallback.

Use a compatibility-preserving constructor so the existing mock-rate-limit call remains valid while canonical details are retained:

```typescript
export class RateLimitError extends Error {
  readonly code: string;
  readonly detail: string | null;
  readonly retryable: boolean;
  readonly retryAfterSeconds: number;
  readonly supportedVersions: AiEventVersion[];

  constructor(value: AiStreamErrorDetails | number) {
    const details = typeof value === 'number'
      ? {
          code: 'RATE_LIMITED',
          message: '요청이 많아 잠시 후 다시 시도해주세요.',
          detail: null,
          retryable: true,
          retryAfterSeconds: value,
          supportedVersions: [] as AiEventVersion[],
        }
      : value;
    super(details.message);
    this.name = 'RateLimitError';
    this.code = details.code;
    this.detail = details.detail;
    this.retryable = details.retryable;
    this.retryAfterSeconds = details.retryAfterSeconds ?? 10;
    this.supportedVersions = [...details.supportedVersions];
  }
}
```

- [ ] **Step 6: Run decoder and contract checks**

```bash
node --import tsx --test src/api/aiStreamError.test.ts
npm run api:ai-stream:check
```

Expected: all tests pass and no generated drift.

- [ ] **Step 7: Commit generated and shared decoder files**

```bash
git add contracts/ai-stream-v2.openapi.json contracts/ai-stream-v2.metadata.json src/api/generated/aiStreamV2.ts src/api/aiStreamContract.ts src/api/aiStreamError.ts src/api/aiStreamError.test.ts
git diff --cached --check
git diff --cached --name-only
git commit -m "feat: add shared AI stream error decoder"
```

---

### Task 6: Migrate chat and coach to the shared error path

**Files:**
- Modify: `src/api/chatbot.ts`
- Modify: `src/api/chatbot.test.ts`
- Modify: `src/api/coach.ts`
- Modify: `src/api/coach.test.ts`

**Interfaces:**
- Consumes: `decodeAiStreamHttpError`, `normalizeAiStreamEventError`, `RateLimitError`, `AI_EVENT_VERSION_HEADER`.
- Produces: chat and coach UI errors retaining `upstreamCode`, `detail`, `retryable`, `retryAfterSeconds`, and `supportedVersions`.

- [ ] **Step 1: Write failing chat pre-stream and in-stream preservation tests**

Add tests to `chatbot.test.ts` for a canonical 406 and a v2 `stream.error`:

```typescript
test('chat preserves canonical pre-stream version error fields', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({
    code: 'AI_EVENT_VERSION_UNSUPPORTED',
    message: '지원하지 않는 AI 이벤트 버전입니다.',
    detail: null,
    retryable: false,
    retry_after_seconds: null,
    supported_versions: ['1', '2'],
  }), { status: 406 }));

  await assert.rejects(
    () => sendChatMessageStream({ question: '버전', history: null }, () => undefined),
    (error) => {
      assert.ok(error instanceof ChatStreamEventError);
      assert.equal(error.eventCode, 'AI_EVENT_VERSION_UNSUPPORTED');
      assert.deepEqual(error.supportedVersions, ['1', '2']);
      assert.equal(error.retryable, false);
      return true;
    },
  );
});
```

For v2 `stream.error`, assert `eventCode`, safe detail, and `retryable=true`. Extend the existing 429 test to use a canonical body and assert the shared rate-limit error keeps `code` and timing.

- [ ] **Step 2: Write failing coach HTTP and in-stream preservation tests**

Add a canonical 504 test and extend the existing payload-limit stream test:

```typescript
assert.ok(error instanceof CoachAnalyzeError);
assert.equal(error.code, 'STREAM_TIMEOUT');
assert.equal(error.upstreamCode, 'AI_UPSTREAM_TIMEOUT');
assert.equal(error.retryable, true);
assert.equal(error.detail, null);
```

Add a 406 test that asserts `CoachAnalyzeError.code === 'REQUEST_FAILED'` and `supportedVersions` equals `['1', '2']`. Add a 429 test that asserts `RateLimitError` and body/header timing preservation.

- [ ] **Step 3: Run the four focused frontend test files and observe RED**

```bash
node --import tsx --test src/api/aiStreamError.test.ts src/api/aiStreamContract.test.ts src/api/chatbot.test.ts src/api/coach.test.ts
```

Expected: new fields or mappings fail because chat and coach still use separate logic.

- [ ] **Step 4: Migrate chat**

In `chatbot.ts`:

- import and re-export `RateLimitError` from `aiStreamError.ts`;
- import `AI_EVENT_VERSION_HEADER` from `aiStreamContract.ts` and remove the local constant;
- remove `parseRetryAfterSeconds`;
- decode every non-OK `Response` once with `decodeAiStreamHttpError`;
- throw `RateLimitError` for status 429;
- retry only when the normalized error is retryable and attempts remain;
- throw `ChatStreamEventError` with the normalized details on the terminal attempt;
- normalize v2 `stream.error.data` through `normalizeAiStreamEventError`.

Extend `ChatStreamEventError` without changing its class name:

```typescript
export class ChatStreamEventError extends Error {
  readonly eventCode: string;
  readonly detail: string | null;
  readonly retryable: boolean;
  readonly retryAfterSeconds: number | null;
  readonly supportedVersions: AiEventVersion[];

  constructor(details: AiStreamErrorDetails) {
    super(CHATBOT_STREAM_TEMPORARY_ERROR);
    this.name = 'ChatStreamEventError';
    this.eventCode = details.code;
    this.detail = details.detail ?? details.message;
    this.retryable = details.retryable;
    this.retryAfterSeconds = details.retryAfterSeconds;
    this.supportedVersions = [...details.supportedVersions];
  }
}
```

Legacy v1 `error` frames must also be converted to a complete `AiStreamErrorDetails` object with `retryable=true`, null timing, and empty versions; do not retain the old positional constructor.

- [ ] **Step 5: Migrate coach**

In `coach.ts`:

- import the shared decoder, event normalizer, `RateLimitError`, and common header;
- remove `ParsedCoachErrorPayload`, `readCoachErrorPayload`, `resolveCoachRequestFailureMessage`, and the local header constant;
- preserve auth reissue by storing one decoded `AiStreamRequestError` for a 401 and reusing it after retries;
- use code/status mapping only at the UI boundary: 413 or `AI_PROXY_PAYLOAD_TOO_LARGE` -> `PAYLOAD_TOO_LARGE`; 504 or `AI_UPSTREAM_TIMEOUT` -> `STREAM_TIMEOUT`; all others -> `REQUEST_FAILED`; 429 -> shared `RateLimitError`;
- normalize v2 `stream.error.data` through the shared function.

Extend `CoachAnalyzeError`:

```typescript
export class CoachAnalyzeError extends Error {
  readonly code: CoachAnalyzeErrorCode;
  readonly statusCode: number | null;
  readonly upstreamCode: string | null;
  readonly detail: string | null;
  readonly retryable: boolean;
  readonly retryAfterSeconds: number | null;
  readonly supportedVersions: AiEventVersion[];

  constructor(
    code: CoachAnalyzeErrorCode,
    message: string,
    statusCode: number | null = null,
    details?: AiStreamErrorDetails,
  ) {
    super(message);
    this.name = 'CoachAnalyzeError';
    this.code = code;
    this.statusCode = statusCode;
    this.upstreamCode = details?.code ?? null;
    this.detail = details?.detail ?? null;
    this.retryable = details?.retryable ?? false;
    this.retryAfterSeconds = details?.retryAfterSeconds ?? null;
    this.supportedVersions = [...(details?.supportedVersions ?? [])];
  }
}
```

Keep payload-limit display copy and authentication-expired behavior unchanged.

- [ ] **Step 6: Run focused tests and type/build checks**

```bash
node --import tsx --test src/api/aiStreamError.test.ts src/api/aiStreamContract.test.ts src/api/chatbot.test.ts src/api/coach.test.ts
npm run api:ai-stream:check
npm run build
```

Expected: all focused tests and the production build pass; no raw-body error string remains in chat or coach.

- [ ] **Step 7: Commit only chat/coach migration files**

```bash
git add src/api/chatbot.ts src/api/chatbot.test.ts src/api/coach.ts src/api/coach.test.ts
git diff --cached --check
git diff --cached --name-only
git commit -m "refactor: unify AI stream error handling"
```

---

### Task 7: Switch the browser default to v2 and document rollback

**Files:**
- Modify: `src/api/aiStreamContract.ts`
- Modify: `src/api/aiStreamContract.test.ts`
- Modify: `.env.example`
- Modify: `../docs/API_REFERENCE.md` from the workspace root
- Create: `../.superpowers/sdd/ai-stream-error-progress.md` from the workspace root

**Interfaces:**
- Consumes: supported frontend versions `'1' | '2'`.
- Produces: absent/empty/whitespace browser config -> `2`; explicit `1` -> v1 rollback.

- [ ] **Step 1: Change the resolver test first**

Replace the legacy-default test in `aiStreamContract.test.ts` with:

```typescript
test('resolveAiEventVersion defaults to v2 and preserves explicit v1 rollback', () => {
  assert.equal(resolveAiEventVersion(undefined), '2');
  assert.equal(resolveAiEventVersion(null), '2');
  assert.equal(resolveAiEventVersion(''), '2');
  assert.equal(resolveAiEventVersion('   '), '2');
  assert.equal(resolveAiEventVersion('1'), '1');
  assert.equal(resolveAiEventVersion(' 2 '), '2');
  assert.throws(() => resolveAiEventVersion('3'), /VITE_AI_EVENT_VERSION/);
});
```

Add one request assertion each in chat and coach tests proving the default emitted header is `X-AI-Event-Version: 2`. Retain at least one explicit-v1 test in each suite proving the legacy parser path and `[DONE]` still work.

The test files currently set `process.env.VITE_AI_EVENT_VERSION = '1'` at module scope. In each default-header test, save the prior value, delete the environment key before invoking the request, and restore it with `t.after`; do not remove the module-level v1 setting because the existing v1 characterization tests depend on it.

- [ ] **Step 2: Run tests and observe RED**

```bash
node --import tsx --test src/api/aiStreamContract.test.ts src/api/chatbot.test.ts src/api/coach.test.ts
```

Expected: missing/empty values still resolve to `1`.

- [ ] **Step 3: Implement the exact resolver table**

Use:

```typescript
export const resolveAiEventVersion = (value: unknown): AiEventVersion => {
  if (value === undefined || value === null) return '2';
  if (typeof value !== 'string') {
    return fail('VITE_AI_EVENT_VERSION must be 1 or 2');
  }
  const normalized = value.trim();
  if (!normalized) return '2';
  if (normalized === '1' || normalized === '2') return normalized;
  return fail('VITE_AI_EVENT_VERSION must be 1 or 2');
};
```

- [ ] **Step 4: Update checked-in environment guidance**

Change only the frontend `.env.example` lines to:

```dotenv
# 기본값은 SSE v2입니다. 긴급 롤백 시에만 VITE_AI_EVENT_VERSION=1로 변경합니다.
VITE_AI_EVENT_VERSION=2
```

Do not modify AI `.env.example` or any real `.env` file.

- [ ] **Step 5: Run frontend release checks**

```bash
node --import tsx --test src/api/aiStreamError.test.ts src/api/aiStreamContract.test.ts src/api/chatbot.test.ts src/api/coach.test.ts
npm run api:ai-stream:check
npm run test:unit
npm run build
```

Expected: focused tests, full unit suite, generated-contract check, and production build pass.

- [ ] **Step 6: Commit the frontend rollout switch**

```bash
git add src/api/aiStreamContract.ts src/api/aiStreamContract.test.ts src/api/chatbot.test.ts src/api/coach.test.ts .env.example
git diff --cached --check
git diff --cached --name-only
git commit -m "feat: default AI streams to event version 2"
```

- [ ] **Step 7: Update integrated documentation and progress ledger**

In root `docs/API_REFERENCE.md`, add the six-field HTTP error example, state that the AI contract artifact is `2.1.0`, and document:

```text
Browser default: X-AI-Event-Version: 2
Rollback: set VITE_AI_EVENT_VERSION=1 and rebuild/redeploy only the frontend
FastAPI absent-header compatibility: v1
```

Create `.superpowers/sdd/ai-stream-error-progress.md` with each task's repository, commit SHA, focused command, and result. Do not modify `.superpowers/sdd/progress.md`.

- [ ] **Step 8: Commit only integrated documentation**

From `/Users/mac/project/KBO_platform`:

```bash
git add docs/API_REFERENCE.md .superpowers/sdd/ai-stream-error-progress.md
git diff --cached --check
git diff --cached --name-only
git commit -m "docs: publish AI stream error contract"
```

---

### Task 8: Run the cross-service release gate and reviews

**Files:**
- Modify only `.superpowers/sdd/ai-stream-error-progress.md` if verification results need recording.

**Interfaces:**
- Consumes: all prior task commits.
- Produces: evidence that contract generation, behavior, builds, safety policy, and dirty-worktree isolation are acceptable.

- [ ] **Step 1: Verify AI contract drift and focused behavior**

From `bega_AI`:

```bash
./.venv/bin/python scripts/export_ai_stream_contract.py --check
./.venv/bin/python -m pytest tests/test_stream_events_v2.py tests/test_ai_stream_contract_export.py tests/test_versioned_sse.py tests/test_stream_http_errors.py tests/test_chat_stream_versioning.py tests/test_coach_contracts.py -q
```

Expected: artifact current; all focused tests pass.

- [ ] **Step 2: Run the full AI suite**

```bash
./.venv/bin/python -m pytest tests/ -q
```

Expected: full suite passes. Record exact pass/warning counts.

- [ ] **Step 3: Verify backend focused and full suites**

From `bega_backend/BEGA_PROJECT`:

```bash
./gradlew test --tests "*AiStreamHttpErrorResponseTest" --tests "*AiProxyServiceTest" --tests "*AiProxyControllerTest" --tests "*CorsIntegrationTest"
./gradlew test
```

Expected: focused suite and full backend suite pass. If unrelated dirty realtime tests fail, capture the exact failure and prove all scoped tests remain green; do not alter unrelated files.

- [ ] **Step 4: Verify frontend contract, tests, and build**

From `bega_frontend`:

```bash
npm run api:ai-stream:check
node --import tsx --test src/api/aiStreamError.test.ts src/api/aiStreamContract.test.ts src/api/chatbot.test.ts src/api/coach.test.ts
npm run test:unit
npm run build
```

Expected: generated contract current, focused/full unit tests pass, production build passes.

- [ ] **Step 5: Run repository safety and diff gates**

From the workspace root:

```bash
python3 scripts/validate_baseball_data_policy.py
git diff --check
git -C bega_AI diff --check
git -C bega_backend diff --check
git -C bega_frontend diff --check
```

Expected: baseball policy passes and all repositories report no whitespace errors.

- [ ] **Step 6: Audit generated parity and secrets**

Confirm:

```bash
rg -n '"version": "2.1.0"|"AiStreamHttpError"' bega_AI/contracts/ai-stream-v2.openapi.json bega_frontend/contracts/ai-stream-v2.openapi.json bega_frontend/contracts/ai-stream-v2.metadata.json
rg -n 'HTTP error! status:|body: \$\{errorText\}|secret-bearing' bega_frontend/src/api/chatbot.ts bega_frontend/src/api/coach.ts bega_backend/BEGA_PROJECT/src/main/java/com/example/ai
```

Expected: both artifacts and metadata identify `2.1.0`; production files contain no raw-body error construction or test secret marker.

- [ ] **Step 7: Request independent reviews**

Run the repository-required reviewers against the complete cross-service diff:

- `code-reviewer`: contract consistency, retries, cleanup, maintainability, and test quality;
- `frontend-code-reviewer`: React/TypeScript error class compatibility, retry semantics, and v1 rollback;
- `security-reviewer`: raw-body disclosure, status/message safety, auth boundary, CORS/header behavior, and secret handling.

Resolve every blocking or high-priority finding with a new RED/GREEN test and narrow follow-up commit. Re-run the affected focused suite after each fix.

- [ ] **Step 8: Perform the release-verification completion audit**

Use `kbo-release-verification` and `verification-before-completion`. Compare every acceptance criterion in the design to a passing test or inspected artifact. Update `.superpowers/sdd/ai-stream-error-progress.md` with final commands, results, review verdicts, and commit SHAs, then commit only that ledger update if changed:

```bash
git add .superpowers/sdd/ai-stream-error-progress.md
git diff --cached --check
git commit -m "docs: record AI stream release verification"
```

Do not declare the phase complete unless all scoped gates are green and unrelated dirty work remains unstaged.
