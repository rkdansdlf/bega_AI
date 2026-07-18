# AI legacy endpoint deprecation runbook

작성일: 2026-06-07
Phase 1 갱신일: 2026-07-18

## 목적

FastAPI의 7개 legacy operation을 canonical `/ai/**` 경로로 이전한다. Phase 1에서는
모든 route를 계속 제공하면서 호출자를 이전하고 사용량을 관측한다. 연속 7일간
legacy 호출과 Backend vision fallback이 0건인 경우에만 별도 승인을 받아 Phase 2
비활성화를 진행하고, 이후 추가 14일간 0건이면 Phase 3 삭제를 판단한다.

## Phase 1 고정 대상

| Legacy operation | Canonical successor |
|---|---|
| `POST /coach/analyze` | `POST /ai/coach/analyze` |
| `POST /coach/cache/reset` | `POST /ai/coach/cache/reset` |
| `POST /coach/analyze-legacy` | `POST /ai/coach/analyze` |
| `POST /ai/coach/analyze-legacy` | `POST /ai/coach/analyze` |
| `POST /vision/ticket` | `POST /ai/vision/ticket` |
| `POST /vision/seat-view-classify` | `POST /ai/vision/seat-view-classify` |
| `GET /ai/chat/stream` | `POST /ai/chat/stream` |

AI OpenAPI는 Phase 1 동안 33개 operation을 유지하며, 위 7개에만
`deprecated: true`를 표시한다. canonical operation은 deprecated가 아니다.

## 식별 계약

legacy 응답에는 다음 migration metadata가 붙는다.

| Header/log | 값 |
|---|---|
| `Deprecation` | `true` |
| `Link` | `<canonical-path>; rel="successor-version"` |
| `X-Deprecation` | `true` |
| `X-Legacy-Endpoint` | legacy method/path. 기존 `analyze-legacy` 값은 호환을 위해 유지할 수 있음 |
| 사용 로그 | `deprecated_api_operation method=... legacy_path=... canonical_method=... canonical_path=...` |

운영자가 실제 비활성화 날짜를 승인하기 전에는 `Sunset` 헤더를 발행하지 않는다.
`COACH_ANALYZE_LEGACY_ENABLED` 기본값은 Phase 1에서 `1`로 유지한다. 이 flag를
`0`으로 설정했을 때 기존 두 `analyze-legacy` operation이 HTTP 410을 반환하는
기능도 그대로 유지한다.

## 보류 조건

- 7일 관측 기간 중 legacy operation 호출이 1건 이상 발생한다.
- 호출 클라이언트가 내부 배치인지 외부 클라이언트인지 확인되지 않았다.
- Backend가 canonical `/ai/vision/**`에서 404를 받고 `/vision/**`으로 fallback한다.
- canonical Coach/Vision/POST stream의 5xx, timeout 또는 SSE 완료율이 기준선보다 악화된다.
- canonical 응답에서 `Deprecation`, `X-Deprecation` 또는 `X-Legacy-Endpoint`가 발견된다.
- Phase 2 이후 HTTP 410이 반복 발생한다.

## 1. Phase 1 관측

1. 운영 로그에서 legacy 사용 로그를 operation별로 연속 7일간 집계한다.

```bash
rg "deprecated_api_operation method=" <ai-log-path>
```

2. Backend 로그에서 vision canonical 404 fallback을 같은 기간 집계한다.

```bash
rg "fallback to legacy path|fallback to legacy endpoint" <backend-log-path>
```

3. 같은 기간 canonical Coach/Vision/POST stream의 호출량, 5xx, timeout, SSE 완료율을 기록한다.
4. 다음 조건이 모두 충족될 때만 Phase 2 승인을 요청한다.

   - 7개 legacy operation 호출 0건
   - Backend vision fallback 0건
   - canonical 오류율 증가 없음

한 조건이라도 충족되지 않으면 7일 관측 창을 다시 시작한다.

## 2. Phase 2 비활성화

Phase 2는 자동으로 진행하지 않는다. 운영 지표와 실제 비활성화 날짜를 첨부해 별도
승인을 받은 뒤 staging부터 적용한다.

1. `analyze-legacy`는 staging에서 `COACH_ANALYZE_LEGACY_ENABLED=0`을 적용해 HTTP 410을 확인한다.
2. GET stream과 prefix alias의 비활성화 방식은 호출자·배포 호환성 근거를 바탕으로 별도 구현한다.
3. canonical operation은 기존 JSON/SSE 동작과 인증 계약을 유지해야 한다.
4. staging 검증 후 운영에 적용하고 24시간 동안 410, canonical 5xx, timeout, SSE 완료율을 집중 관측한다.
5. 반복 호출 또는 회귀가 있으면 route 삭제를 보류하고 호출자를 추적한다.

## 3. Phase 3 삭제 판단

1. Phase 2 적용 후 추가 14일 동안 legacy/410 호출이 0건인지 확인한다.
2. 조건을 만족하면 7개 operation과 Backend vision fallback 삭제를 위한 별도 작업과 승인을 만든다.
3. 삭제 후 AI OpenAPI operation 수는 33개에서 26개가 되어야 한다.
4. 팀 코드 legacy resolver, 데이터베이스 환경변수 fallback, 다른 호환 계층은 이 삭제 범위에 포함하지 않는다.

## 테스트 체크리스트

| 테스트 | Phase 1 기대 결과 |
|---|---|
| AI OpenAPI operation 수 | 33 |
| AI OpenAPI deprecated 수 | 정확히 7 |
| canonical operation metadata | `deprecated` 아님 |
| legacy 응답 | `Deprecation`, `Link`, `X-Deprecation`, `X-Legacy-Endpoint` 포함 |
| canonical 응답 | deprecation 관련 header 없음 |
| `COACH_ANALYZE_LEGACY_ENABLED=0` legacy 호출 | 기존처럼 HTTP 410 |
| 내부 batch/smoke/backfill Coach 주소 | `/ai/coach/analyze` |
| Backend vision 호출 | `/ai/vision/**` 우선, 404에서만 `/vision/**` fallback |

## 관측 기록 양식

| 날짜 | 환경 | operation | legacy 호출 | vision fallback | 410 | canonical 5xx | canonical SSE 완료율 | 판단 |
|---|---|---|---:|---:|---:|---:|---:|---|
| YYYY-MM-DD | staging/prod | method + path | 0 | 0 | 0 | 0 | 100% | continue/hold |
