# KBO AI Service Endpoints

> This file is generated. Do not edit directly.
> Source: `contracts/openapi.json`
> Regenerate with: `python scripts/export_openapi_contract.py`

Version: `0.1.0`
Paths: **32**
Operations: **33**

## chat

### DELETE `/ai/chat/cache`
Flush Cache By Intent

특정 intent의 캐시 항목을 모두 삭제합니다.
- Operation ID: `flush_cache_by_intent_ai_chat_cache_delete`
- Tags: `chat`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `intent` | query | yes | `string` | 삭제할 intent |  |
| `Authorization` | header | no | `string` |  |  |
| `X-Cache-Admin-Token` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### GET `/ai/chat/cache/stats`
Chat Cache Stats

캐시 현황 통계를 반환합니다.
- Operation ID: `chat_cache_stats_ai_chat_cache_stats_get`
- Tags: `chat`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Cache-Admin-Token` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### DELETE `/ai/chat/cache/{cache_key}`
Invalidate Cache Entry

특정 캐시 키를 무효화합니다.
- Operation ID: `invalidate_cache_entry_ai_chat_cache__cache_key__delete`
- Tags: `chat`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `cache_key` | path | yes | `string` |  |  |
| `Authorization` | header | no | `string` |  |  |
| `X-Cache-Admin-Token` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/ai/chat/completion`
Chat Completion

단일 JSON 응답으로 전체 채팅 답변을 반환하는 엔드포인트입니다.
- Operation ID: `chat_completion_ai_chat_completion_post`
- Tags: `chat`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: `object`

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### GET `/ai/chat/stream`
Chat Stream Get

GET 요청을 통해 질문을 받고, 답변을 SSE 스트림으로 반환합니다.
- Operation ID: `chat_stream_get_ai_chat_stream_get`
- Tags: `chat`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `q` | query | no | `string` | 질문 텍스트 |  |
| `style` | query | no | `string` |  |  |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/ai/chat/stream`
Chat Stream Post

POST 요청을 통해 질문을 받고, 답변을 SSE 스트림으로 반환합니다.
- Operation ID: `chat_stream_post_ai_chat_stream_post`
- Tags: `chat`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `style` | query | no | `string` |  |  |
| `Authorization` | header | no | `string` |  |  |
| `X-AI-Event-Version` | header | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "X-Ai-Event-Version" }` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: `object`

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/ai/chat/voice`
Transcribe Audio
- Operation ID: `transcribe_audio_ai_chat_voice_post`
- Tags: `chat`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `multipart/form-data`
- Schema: [Body_transcribe_audio_ai_chat_voice_post](api-schemas.md#body-transcribe-audio-ai-chat-voice-post)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

## coach

### POST `/ai/coach/analyze`
Analyze Team

특정 팀(들)에 대한 심층 분석을 요청합니다. 'The Coach' 페르소나가 적용됩니다.
- Operation ID: `analyze_team_ai_coach_analyze_post`
- Tags: `coach`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-AI-Event-Version` | header | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "X-Ai-Event-Version" }` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: [CoachAnalyzeRequest](api-schemas.md#coachanalyzerequest)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/ai/coach/analyze-legacy`
Analyze Team Legacy

기존 방식의 Coach 분석 (전체 에이전트 파이프라인 사용).
Fast Path에 문제가 있을 경우 대안으로 사용.
deprecate: COACH_ANALYZE_LEGACY_ENABLED=0 으로 비활성화 가능.
- Operation ID: `analyze_team_legacy_ai_coach_analyze_legacy_post`
- Tags: `coach`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: [CoachAnalyzeRequest](api-schemas.md#coachanalyzerequest)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/ai/coach/cache/reset`
Reset Coach Cache

FAILED_LOCKED 상태의 Coach 분석 캐시를 운영자가 수동으로 즉시 복구한다.

삭제된 row는 다음 분석 요청에서 자연 재생성된다. 활성 PENDING lease와
COMPLETED row는 보호된다. cache_key 단건 또는 (team_id, year) 묶음을 받는다.
- Operation ID: `reset_coach_cache_ai_coach_cache_reset_post`
- Tags: `coach`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: [CoachCacheResetRequest](api-schemas.md#coachcacheresetrequest)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/coach/analyze`
Analyze Team

특정 팀(들)에 대한 심층 분석을 요청합니다. 'The Coach' 페르소나가 적용됩니다.
- Operation ID: `analyze_team_coach_analyze_post`
- Tags: `coach`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-AI-Event-Version` | header | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "X-Ai-Event-Version" }` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: [CoachAnalyzeRequest](api-schemas.md#coachanalyzerequest)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/coach/analyze-legacy`
Analyze Team Legacy

기존 방식의 Coach 분석 (전체 에이전트 파이프라인 사용).
Fast Path에 문제가 있을 경우 대안으로 사용.
deprecate: COACH_ANALYZE_LEGACY_ENABLED=0 으로 비활성화 가능.
- Operation ID: `analyze_team_legacy_coach_analyze_legacy_post`
- Tags: `coach`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: [CoachAnalyzeRequest](api-schemas.md#coachanalyzerequest)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/coach/cache/reset`
Reset Coach Cache

FAILED_LOCKED 상태의 Coach 분석 캐시를 운영자가 수동으로 즉시 복구한다.

삭제된 row는 다음 분석 요청에서 자연 재생성된다. 활성 PENDING lease와
COMPLETED row는 보호된다. cache_key 단건 또는 (team_id, year) 묶음을 받는다.
- Operation ID: `reset_coach_cache_coach_cache_reset_post`
- Tags: `coach`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: [CoachCacheResetRequest](api-schemas.md#coachcacheresetrequest)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

## coach-auto-brief-ops

### GET `/ai/coach/auto-brief/ops/health`
Get Auto Brief Health
- Operation ID: `get_auto_brief_health_ai_coach_auto_brief_ops_health_get`
- Tags: `coach-auto-brief-ops`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `end_date` | query | no | `{   "anyOf": [     {       "format": "date",       "type": "string"     },     {       "type": "null"     }   ],   "title": "End Date" }` |  |  |
| `sample_size` | query | no | `integer` |  |  |
| `start_date` | query | no | `{   "anyOf": [     {       "format": "date",       "type": "string"     },     {       "type": "null"     }   ],   "title": "Start Date" }` |  |  |
| `window` | query | no | `string` |  |  |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: [CoachAutoBriefOpsHealthResponse](api-schemas.md#coachautobriefopshealthresponse)

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

## ingest

### POST `/ai/ingest/`
Ingest Document
- Operation ID: `ingest_document_ai_ingest__post`
- Tags: `ingest`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: [IngestPayload](api-schemas.md#ingestpayload)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/ai/ingest/run`
Run Ingestion Job

Persist or deduplicate a durable internal-DB ingestion run.
- Operation ID: `run_ingestion_job_ai_ingest_run_post`
- Tags: `ingest`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: [RunIngestPayload](api-schemas.md#runingestpayload)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### GET `/ai/ingest/runs/{run_id}`
Get Ingestion Run

Return only sanitized durable run status fields.
- Operation ID: `get_ingestion_run_ai_ingest_runs__run_id__get`
- Tags: `ingest`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `run_id` | path | yes | `string (uuid)` |  |  |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

## moderation

### POST `/moderation/safety-check`
Safety Check

게시글/댓글 텍스트를 점진 Hybrid 정책(RULE + MODEL)으로 검사합니다.
- Operation ID: `safety_check_moderation_safety_check_post`
- Tags: `moderation`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: `object`

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: [ModerationResult](api-schemas.md#moderationresult)

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

## release-decision

### GET `/ai/release-decision/artifacts`
List Release Decision Artifacts
- Operation ID: `list_release_decision_artifacts_ai_release_decision_artifacts_get`
- Tags: `release-decision`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `array`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### GET `/ai/release-decision/artifacts/{artifact_id}`
Get Release Decision Artifact
- Operation ID: `get_release_decision_artifact_ai_release_decision_artifacts__artifact_id__get`
- Tags: `release-decision`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `artifact_id` | path | yes | `string` |  |  |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: [ReleaseDecisionArtifactRecord](api-schemas.md#releasedecisionartifactrecord)

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/ai/release-decision/draft`
Draft Release Decision
- Operation ID: `draft_release_decision_ai_release_decision_draft_post`
- Tags: `release-decision`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: [ReleaseDecisionDraftRequest](api-schemas.md#releasedecisiondraftrequest)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: [ReleaseDecisionDraftResponse](api-schemas.md#releasedecisiondraftresponse)

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### GET `/ai/release-decision/eval-cases`
List Release Decision Eval Cases
- Operation ID: `list_release_decision_eval_cases_ai_release_decision_eval_cases_get`
- Tags: `release-decision`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `array`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/ai/release-decision/evaluate`
Evaluate Release Decision Draft
- Operation ID: `evaluate_release_decision_draft_ai_release_decision_evaluate_post`
- Tags: `release-decision`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: [ReleaseDecisionEvaluateRequest](api-schemas.md#releasedecisionevaluaterequest)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: [ReleaseDecisionEvaluateResponse](api-schemas.md#releasedecisionevaluateresponse)

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### GET `/ai/release-decision/presets`
List Release Decision Presets
- Operation ID: `list_release_decision_presets_ai_release_decision_presets_get`
- Tags: `release-decision`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `array`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/ai/release-decision/save`
Save Release Decision Artifact
- Operation ID: `save_release_decision_artifact_ai_release_decision_save_post`
- Tags: `release-decision`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `application/json`
- Schema: [ReleaseDecisionSaveRequest](api-schemas.md#releasedecisionsaverequest)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: [ReleaseDecisionArtifactSummary](api-schemas.md#releasedecisionartifactsummary)

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

## search

### GET `/ai/search/`
Debug Search

검색 알고리즘을 자세히 분석하고 성능을 측정하는 디버깅 엔드포인트입니다.
개발자가 검색 품질을 개선하기 위해 사용합니다.
- Operation ID: `debug_search_ai_search__get`
- Tags: `search`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `limit` | query | no | `integer` | 검색 결과 개수 |  |
| `q` | query | yes | `string` | 분석할 질문 또는 키워드 |  |
| `team` | query | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "description": "팀명 (예: LG, KIA)",   "title": "Team" }` | 팀명 (예: LG, KIA) |  |
| `use_multi_query` | query | no | `boolean` | 다중 쿼리 검색 사용 여부 |  |
| `year` | query | no | `{   "anyOf": [     {       "type": "integer"     },     {       "type": "null"     }   ],   "description": "시즌 연도 (예: 2025)",   "title": "Year" }` | 시즌 연도 (예: 2025) |  |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: [SearchAnalysisResponse](api-schemas.md#searchanalysisresponse)

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### GET `/ai/search/compare-methods`
Compare Search Methods

단일 쿼리 검색과 다중 쿼리 검색 성능을 비교하는 엔드포인트입니다.
- Operation ID: `compare_search_methods_ai_search_compare_methods_get`
- Tags: `search`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `limit` | query | no | `integer` |  |  |
| `q` | query | yes | `string` | 비교 테스트할 질문 |  |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### GET `/ai/search/test-entity-extraction`
Test Entity Extraction

질문에서 엔티티 추출이 올바르게 작동하는지 테스트하는 엔드포인트입니다.
- Operation ID: `test_entity_extraction_ai_search_test_entity_extraction_get`
- Tags: `search`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `q` | query | yes | `string` | 엔티티 추출을 테스트할 질문 |  |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

## system

### GET `/health`
Health

애플리케이션의 상태를 확인하는 헬스 체크 엔드포인트.
- Operation ID: `health_health_get`
- Tags: `system`
- Security: Not specified in OpenAPI
- Deprecated: no

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: `{}`

## vision

### POST `/ai/vision/seat-view-classify`
Classify Seat View Image

Classify an uploaded baseball-related image for seat-view moderation.
- Operation ID: `classify_seat_view_image_ai_vision_seat_view_classify_post`
- Tags: `vision`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `multipart/form-data`
- Schema: [Body_classify_seat_view_image_ai_vision_seat_view_classify_post](api-schemas.md#body-classify-seat-view-image-ai-vision-seat-view-classify-post)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: [SeatViewClassification](api-schemas.md#seatviewclassification)

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/ai/vision/ticket`
Analyze Ticket Image

Analyzes an uploaded ticket image using Gemini Vision (Native or OpenRouter) to extract details.
- Operation ID: `analyze_ticket_image_ai_vision_ticket_post`
- Tags: `vision`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `multipart/form-data`
- Schema: [Body_analyze_ticket_image_ai_vision_ticket_post](api-schemas.md#body-analyze-ticket-image-ai-vision-ticket-post)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: [TicketInfo](api-schemas.md#ticketinfo)

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/vision/seat-view-classify`
Classify Seat View Image

Classify an uploaded baseball-related image for seat-view moderation.
- Operation ID: `classify_seat_view_image_vision_seat_view_classify_post`
- Tags: `vision`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `multipart/form-data`
- Schema: [Body_classify_seat_view_image_vision_seat_view_classify_post](api-schemas.md#body-classify-seat-view-image-vision-seat-view-classify-post)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: [SeatViewClassification](api-schemas.md#seatviewclassification)

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)

### POST `/vision/ticket`
Analyze Ticket Image

Analyzes an uploaded ticket image using Gemini Vision (Native or OpenRouter) to extract details.
- Operation ID: `analyze_ticket_image_vision_ticket_post`
- Tags: `vision`
- Security: `InternalApiKey`
- Deprecated: no

#### Parameters

| Name | In | Required | Schema | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `Authorization` | header | no | `string` |  |  |
| `X-Internal-Api-Key` | header | no | `string` |  |  |

### Request body
- Required: **yes**

#### Media type: `multipart/form-data`
- Schema: [Body_analyze_ticket_image_vision_ticket_post](api-schemas.md#body-analyze-ticket-image-vision-ticket-post)

### Responses

### Response `200`
Successful Response

#### Media type: `application/json`
- Schema: [TicketInfo](api-schemas.md#ticketinfo)

### Response `422`
Validation Error

#### Media type: `application/json`
- Schema: [HTTPValidationError](api-schemas.md#httpvalidationerror)
