# Coach legacy endpoint deprecation runbook

작성일: 2026-06-07

## 목적

`/coach/analyze-legacy`와 `/ai/coach/analyze-legacy` 호출을 운영에서 분리 추적하고, 호출량 0건 유지가 확인되면 `COACH_ANALYZE_LEGACY_ENABLED=0`으로 비활성화한 뒤 최종 삭제 판단까지 진행한다.

현재 legacy 경로는 `COACH_ANALYZE_LEGACY_ENABLED=0`일 때 HTTP 410을 반환한다. 정상 legacy 응답에는 `X-Legacy-Endpoint: analyze-legacy`와 `X-Deprecation: true` 헤더가 붙는다.

## 고정 기준

| 항목 | 값 |
|---|---|
| 대상 legacy 경로 | `/coach/analyze-legacy`, `/ai/coach/analyze-legacy` |
| 정상 경로 | `/coach/analyze`, `/ai/coach/analyze` |
| legacy 사용 로그 | `[Coach Router Legacy] Deprecated endpoint used` |
| legacy 비활성화 로그 | `[Coach Router] analyze-legacy is disabled. Use /ai/coach/analyze instead.` |
| legacy 식별 헤더 | `X-Legacy-Endpoint: analyze-legacy`, `X-Deprecation: true` |
| 비활성화 응답 | HTTP 410, `analyze-legacy is deprecated. Use /ai/coach/analyze.` |
| 전환 기준 | 운영에서 연속 7일간 legacy 호출 0건 |
| 삭제 판단 기준 | 운영 비활성화 후 추가 14일간 legacy 호출 0건 |

## 보류 조건

- 7일 관측 기간 중 legacy 호출이 1건 이상 발생한다.
- 호출 클라이언트가 내부 배치인지 외부 클라이언트인지 확인되지 않았다.
- 운영 비활성화 후 HTTP 410이 반복 발생한다.
- 정상 `/coach/analyze` 또는 `/ai/coach/analyze`의 5xx, timeout, Coach 응답 실패율이 증가한다.
- 정상 경로 응답에서 `X-Legacy-Endpoint` 헤더가 발견된다.

## 1. 관측 단계

1. 운영 로그에서 legacy 사용 로그를 7일간 집계한다.

```bash
rg "\\[Coach Router Legacy\\] Deprecated endpoint used" <log-path>
```

2. `/coach/analyze-legacy`와 `/ai/coach/analyze-legacy`를 분리 집계한다.
3. 같은 기간 정상 `/coach/analyze`, `/ai/coach/analyze`의 호출량, 5xx, timeout, Coach 응답 실패율을 같이 기록한다.
4. 7일 연속 0건이면 스테이징 비활성화 단계로 이동한다.

## 2. 스테이징 비활성화 단계

1. 스테이징 환경에 다음 값을 적용한다.

```bash
COACH_ANALYZE_LEGACY_ENABLED=0
```

2. legacy 경로는 HTTP 410을 반환해야 한다.
3. 정상 `/ai/coach/analyze`는 기존처럼 SSE 응답을 반환해야 한다.
4. 정상 경로 응답에는 `X-Legacy-Endpoint`가 없어야 한다.
5. legacy 비활성화 로그가 남는지 확인한다.

## 3. 운영 비활성화 단계

1. 운영에서 7일 연속 legacy 호출 0건이 확인된 후 다음 값을 적용한다.

```bash
COACH_ANALYZE_LEGACY_ENABLED=0
```

2. 적용 후 24시간 동안 다음 지표를 확인한다.

| 지표 | 기대값 |
|---|---|
| legacy HTTP 410 발생량 | 0건 유지, 발생 시 클라이언트 추적 |
| Coach 5xx | 기존 기준선 대비 증가 없음 |
| 정상 `/analyze` 호출량 | 급감 없음 |
| 정상 `/analyze` SSE 완료율 | 기존 기준선 대비 하락 없음 |

3. HTTP 410이 반복 발생하면 호출 클라이언트를 추적하고 라우트 삭제 판단은 보류한다. 운영 비활성화는 유지한다.

## 4. 제거 판단 단계

1. 운영 비활성화 후 추가 14일 동안 legacy 호출 0건을 확인한다.
2. 조건을 만족하면 별도 삭제 작업을 생성한다.
3. 삭제 대상은 `/analyze-legacy` 라우트, 관련 로그 문구, legacy endpoint 응답 헤더로 제한한다.
4. 팀코드 legacy resolver와 `SUPABASE_DB_URL` fallback은 이번 삭제 대상에 포함하지 않는다.

## 테스트 체크리스트

| 테스트 | 기대 결과 |
|---|---|
| `/ai/coach/analyze` 헤더 확인 | `X-Legacy-Endpoint` 없음 |
| `/ai/coach/analyze-legacy` 헤더 확인 | `X-Legacy-Endpoint: analyze-legacy`, `X-Deprecation: true` |
| `COACH_ANALYZE_LEGACY_ENABLED=0`에서 legacy 호출 | HTTP 410 |
| 같은 설정에서 정상 `/ai/coach/analyze` 호출 | 기존 SSE 흐름 유지 |
| legacy 호출 로그 확인 | `[Coach Router Legacy] Deprecated endpoint used` |
| legacy 비활성화 로그 확인 | disabled warning 기록 |

## 기록 양식

| 날짜 | 환경 | legacy 호출량 | 410 발생량 | 정상 analyze 5xx | 정상 SSE 완료율 | 판단 |
|---|---|---:|---:|---:|---:|---|
| YYYY-MM-DD | staging/prod | 0 | 0 | 0 | 100% | continue |
