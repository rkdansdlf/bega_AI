# 데이터 동기화 오케스트레이션 운영 런북

이 런북은 내부 KBO 데이터베이스에서 AI `rag_chunks`로 반영되는 내구성 있는 동기화 작업을 운영하는 절차를 설명합니다. 외부 야구 API, 크롤링, 웹 검색 복구는 사용하지 않습니다. 내부 데이터가 없거나 불일치하면 `MANUAL_BASEBALL_DATA_REQUIRED` 상태로 운영자에게 넘깁니다.

## 단일 스케줄러 원칙

운영 정기 실행 주체는 backend JobRunr의 `ai-rag-ingestion` 반복 작업 하나입니다.

- backend `AI_INGEST_ENABLED=true`일 때만 반복 작업을 등록합니다.
- `AI_INGEST_CRON`의 기본값은 KST 기준 서버 설정에 따른 `30 4 * * *`입니다. 배포 환경의 JobRunr 시간대를 먼저 확인합니다.
- `scripts/daily_ingest_kbo.sh`는 `MANUAL RECOVERY ONLY`입니다. 두 번째 cron이나 별도 주기 작업으로 등록하지 않습니다.
- 여러 backend 인스턴스가 같은 JobRunr 저장소를 사용해야 동일 recurring-job ID의 단일 실행 특성이 유지됩니다.

## 배포 및 마이그레이션 순서

1. DDL 권한이 있는 migration role로 AI 스키마를 먼저 적용합니다.

   ```bash
   AI_SCHEMA_DB_URL='postgresql://...' ./scripts/migrate_ai_runtime_schema.sh
   ```

   Managed migration script applies `001_ai_runtime_cache.sql` -> `003_ai_ingest_orchestration.sql` -> `004_ai_ingest_checkpoints.sql`. Compatibility startup applies `003_ai_ingest_orchestration.sql` -> `004_ai_ingest_checkpoints.sql`. 003은 `ai_ingest_runs`, 활성 요청 유일 인덱스, 범위별 `ai_ingest_watermarks`를 생성하고, 004는 내구성 진행 상태용 `ai_ingest_checkpoints`를 생성합니다. Fresh schema와 이전 004 schema 모두에 `source_updated_before timestamptz` stores the immutable source-clock cutoff; 같은 004가 `ADD COLUMN IF NOT EXISTS`로 기존 테이블을 안전하게 보강합니다.

2. AI 서비스를 `AI_DB_SCHEMA_MODE=managed`로 시작합니다. 작업자는 기본 활성화되며 다음 값으로 조정합니다.

   ```env
   AI_INGEST_WORKER_ENABLED=true
   AI_INGEST_WORKER_POLL_SECONDS=2
   AI_INGEST_WORKER_LEASE_SECONDS=120
   AI_INGEST_WORKER_MAX_RECOVERY_ATTEMPTS=1
   ```

   AI ingest coordination uses a dedicated PostgreSQL pool with one minimum and two maximum connections per AI process. The pool is not configurable independently: its bounded size is part of the heartbeat isolation contract. Failure to open this required pool fails AI startup instead of starting a worker without durable coordination. 시작 또는 종료 중 오류가 나도 생성된 모든 background task를 취소·대기하고 두 DB pool을 각각 닫으므로 한쪽 close 실패가 다른 쪽 정리를 건너뛰지 않습니다.

3. AI 서비스의 내부 상태 및 metrics 경로가 정상인지 확인합니다.
4. backend를 처음에는 `AI_INGEST_ENABLED=false`로 배포하고 AI 연결·내부 토큰·JobRunr 저장소를 확인합니다.
5. 다음 backend 설정으로 반복 제출을 활성화합니다.

   ```env
   AI_INGEST_ENABLED=true
   AI_INGEST_CRON=30 4 * * *
   AI_INGEST_TABLES=game,game_metadata,game_summary
   AI_INGEST_CHECK_INTERVAL=30s
   AI_INGEST_MONITORING_DURATION=2h
   ```

   `AI_INGEST_SEASON_YEAR`는 운영자가 명시적으로 고정해야 할 때만 설정합니다. 코드가 야구 시즌을 추정하지 않습니다.

## 실행 상태와 중복 제거

`POST /ai/ingest/run`은 `202 Accepted`와 `run_id`, `status`, `deduplicated`를 반환합니다. 같은 테이블·시즌·범위·모드의 `QUEUED` 또는 `RUNNING` 요청은 활성 유일 인덱스로 하나의 실행에 합쳐집니다. 요청 테이블은 코드에 고정된 내부 야구 DB/정적 문서 허용 목록으로 제한되며 사용자·인증·운영 테이블은 수집할 수 없습니다.

상태 전이는 다음과 같습니다.

- `QUEUED`: DB 큐에 저장되어 worker claim 대기
- `RUNNING`: worker가 lease를 소유하고 heartbeat 갱신
- `SUCCEEDED`: 테이블 결과 저장과 watermark 갱신이 같은 트랜잭션에서 완료
- `FAILED`: 실행 실패 또는 lease 복구 한도 초과
- `MANUAL_BASEBALL_DATA_REQUIRED`: 내부 데이터 누락/불일치로 운영자 입력 필요

backend는 실행을 다시 제출하거나 프로세스 안에서 대기하지 않습니다. 상태 확인 하나마다 JobRunr one-shot 작업 하나를 예약합니다. 성공한 경우에만 일정·순위·홈 캐시를 비웁니다.

## 상태 조회

내부 토큰을 쉘 기록이나 로그에 출력하지 말고 환경변수로 전달합니다.

```bash
curl -sS \
  -H "X-Internal-Api-Key: ${AI_INTERNAL_TOKEN}" \
  "${AI_SERVICE_URL%/}/ai/ingest/runs/<run_id>"
```

응답에는 상태, 요청/시작/heartbeat/완료 시각, 복구 횟수, 허용된 테이블별 처리량, 정제된 오류 계약만 포함됩니다.

## 수동 복구

우선 내구성 큐를 통한 복구를 사용합니다.

```bash
curl -sS -X POST \
  -H "Content-Type: application/json" \
  -H "X-Internal-Api-Key: ${AI_INTERNAL_TOKEN}" \
  -d '{"tables":["game","game_metadata","game_summary"],"mode":"INCREMENTAL","trigger_source":"CLI_RECOVERY"}' \
  "${AI_SERVICE_URL%/}/ai/ingest/run"
```

AI HTTP 경로 자체를 사용할 수 없는 비상 복구 창에서만 아래 직접 명령을 사용합니다. 먼저 backend 정기 제출을 비활성화하고 `ai_ingest_runs`에 `QUEUED`/`RUNNING` 실행이 없음을 확인합니다.

```bash
cd /path/to/bega_AI
./scripts/daily_ingest_kbo.sh
```

이 스크립트는 내부 `OCI_DB_URL` 또는 `POSTGRES_DB_URL`만 읽습니다. cron으로 등록하지 않습니다. 운영자가 제공하지 않은 야구 사실을 생성하거나 외부에서 복구하지 않습니다.

## 리스 만료와 재시작 복구

AI 서비스는 시작 시와 실행 중 주기적으로 만료된 `RUNNING` lease를 확인합니다. heartbeat는 정상 상태에서 lease 시간의 1/3 간격으로 실행됩니다. `psycopg.OperationalError`, `psycopg.InterfaceError`, `psycopg_pool.PoolTimeout`처럼 식별된 일시적 연결·트랜잭션 오류만 지수 백오프로 재시도합니다. 안전 여유는 최대 5초이며 짧은 lease에서는 함께 축소됩니다. 각 heartbeat 호출 자체도 남은 안전 시간으로 제한되고, 그 시각에 도달하면 `exhausted`를 정확히 한 번 기록한 뒤 lease 상실 상태가 됩니다. 프로그래밍·매핑·인증·설정 오류는 재시도하지 않고 즉시 lease 상실로 처리합니다.

heartbeat와 성공·실패·`MANUAL_BASEBALL_DATA_REQUIRED` 종료는 모두 DB에서 `RUNNING`, owner 일치, `lease_expires_at > clock_timestamp()`를 만족해야 합니다. 트랜잭션 시작 시각에 고정되는 `now()`는 lease 판단에 사용하지 않습니다. 동기 ingest 경로와 store mutation은 run row lock을 먼저 얻은 뒤 실제 DB 시각으로 owner·만료 조건을 확인합니다. 만료된 owner는 recovery가 실행되기 전이라도 lease를 되살리거나 terminal 상태를 기록할 수 없습니다.

`AI_INGEST_WORKER_MAX_RECOVERY_ATTEMPTS` 미만이면 만료 실행을 `QUEUED`로 되돌리고, 한도에 도달하면 `FAILED` 및 `INGEST_LEASE_EXPIRED`로 종결합니다. watermark는 시즌과 명시적 `since` 범위별로 분리되고 전체 run이 `SUCCEEDED`가 되는 `finish_success` 트랜잭션에서만 단조 증가합니다. table의 batch checkpoint 성공만으로 watermark가 전진하지 않으므로, 실패·재시작 중 watermark는 변경되지 않습니다.

## Persistent checkpoint resume

Leased database-table ingest는 `ai_ingest_checkpoints`에 `(run_id, source_table)`당 정확히 한 행을 사용합니다. Exactly one `ai_ingest_checkpoints` row per `(run_id, source_table)` is retained after terminal completion. Resume uses a typed ascending keyset and never an offset. 체크포인트가 있으면 source query는 저장된 typed cursor의 모든 필드에 대해 ascending keyset (`>` row-value comparison)으로 재개합니다. custom query는 명시적 고유 cursor order를, generic PostgreSQL table은 실제 지원되는 primary key를 요구합니다.

Checkpoint update-window eligibility uses the same resolver for session startup and query construction. Incremental runs (`since` supplied) preserve the legacy `updated_at` default when a profile omits `since_filter_column`; FULL runs (`since` omitted) freeze only profiles that explicitly declare a nonempty `since_filter_column`. 따라서 `kbo_seasons`처럼 해당 필드를 선언하지 않은 FULL profile은 source clock을 읽거나 기본 `updated_at` 상한을 만들지 않습니다. The `teams` and `stadiums` profiles explicitly bind both FULL cutoff filtering and watermark extraction to `updated_at`. Configured `watermark_fields` are exclusive: ingestion checks only those fields in declared order, returns no row watermark when they are all NULL or missing, and uses generic timestamp aliases only when the profile has no `watermark_fields` configuration. `kbo_seasons` and other timestamp-free profiles declare an empty field list, so unrelated projected timestamps cannot advance their watermark. Eligible new checkpoints read trusted source DB `clock_timestamp()` once before the source data SELECT. Custom and generic incremental queries apply inclusive lower and upper predicates to the same effective column, excluding NULL values. FULL queries use `(column IS NULL OR column <= source_updated_before)` so nullable or left-joined update fields remain eligible. 첫 progress commit 또는 zero-row completion이 정확한 cutoff를 checkpoint row에 저장하고, 재시작은 저장된 cutoff를 재사용합니다. Any progressed eligible legacy checkpoint missing the cutoff fails closed as `INGEST_CHECKPOINT_INCOMPATIBLE`, whether incomplete or completed. A completed legacy checkpoint may skip the source clock and data SELECT without a cutoff only when its durable cursor, counters, and watermark prove zero progress.

Chunks and their cursor commit together under the lease fence. 청크 upsert/비활성화와 cursor advancement는 같은 destination transaction에서 lease fence 확인 후 함께 commit됩니다. 출력 청크가 0개인 source batch도 cursor progress는 fence 아래 commit되어 재시작 시 같은 행을 무한 재처리하지 않습니다. duplicate 또는 역행 cursor는 다음 행 lookahead가 unsafe boundary를 감지한 뒤 commit 전에 거부합니다. 이미 `completed`인 table checkpoint를 복구하면 source data `SELECT` 없이 누적 결과를 반환합니다.

The success watermark remains the maximum committed non-NULL source update timestamp and still advances only in full-run `finish_success`. Every non-NULL update-filtered source row is bounded by the fixed cutoff, while FULL runs deliberately retain NULL update values. 따라서 뒤쪽 cursor의 post-cutoff update가 앞쪽 cursor의 deferred update를 건너뛰게 만들 수 없습니다. The incremental residual duplicate window starts at the prior successful watermark and ends at the fixed cutoff because the lower predicate is inclusive; 이는 안전한 중복 재평가이며 누락 복구를 자동 외부 소스로 대체하지 않습니다.

`source_file` static documents create no checkpoint row and restart atomically. 정적 문서는 전체 profile을 하나의 atomic transaction으로 다시 시작·commit하므로 database-table cursor resume과 섞지 않습니다.

Checkpointed run은 `limit=None` 및 `row_stale_cleanup=off`여야 합니다. `row_stale_cleanup`의 `dry-run` 또는 `apply`는 checkpointed run에서 허용되지 않습니다.

### Checkpoint failure contracts

The generic terminal status error code remains `INGEST_EXECUTION_FAILED`; the five `INGEST_CHECKPOINT_*` identifiers are internal typed exception and metric classifications, not status-payload error codes. `MANUAL_BASEBALL_DATA_REQUIRED` remains the separate operator handoff. 일반 checkpoint 실패의 상태 응답 `error.code`는 `INGEST_EXECUTION_FAILED`이며, worker log의 `run_id`와 `error_type`를 함께 확인합니다. cursor를 수동으로 수정하지 말고 구성 또는 내부 trusted source를 바로잡은 뒤 재시도합니다.

- `INGEST_CHECKPOINT_CURSOR_UNAVAILABLE`: checkpoint 실행에 cursor를 만들 수 없음(예: `limit`이 `None`이 아님, 정적 문서에 checkpoint 요청).
- `INGEST_CHECKPOINT_INCOMPATIBLE`: 저장된 cursor version/signature/scope 또는 durable checkpoint 상태가 현재 order와 호환되지 않음.
- `INGEST_CHECKPOINT_CURSOR_TYPE_UNSUPPORTED`: source 또는 저장 cursor 값이 선언된 typed cursor scalar와 맞지 않음.
- `INGEST_CHECKPOINT_ORDER_VIOLATION`: cursor가 strictly ascending이 아니거나 duplicate여서 안전한 경계를 보장할 수 없음.
- `INGEST_CHECKPOINT_STALE_CLEANUP_UNSUPPORTED`: checkpointed run에 `row_stale_cleanup=off` 이외의 값을 지정함.

필수 internal source field 또는 cursor field가 누락·null이면 `MANUAL_BASEBALL_DATA_REQUIRED`를 반환합니다. `scope=ai_ingest_source_schema`, `missing_fields`, `import_source=operator_manual_data`를 포함한 contract를 운영 티켓에 전달하고, 운영자 검증 데이터 또는 trusted internal sync로만 보완합니다. 외부 야구 API, 크롤링, 웹 검색 복구로 값을 만들지 않습니다.

## MANUAL_BASEBALL_DATA_REQUIRED 인계

상태 응답의 `error`에서 다음 허용 필드만 운영 티켓으로 복사합니다.

- `code`, `scope`, `entity`, `range`
- `missing_fields`, `import_source`
- `operator_message` 또는 `message`
- `blocking`

운영자는 내부 DB 또는 승인된 수동 데이터 입력 경로로 값을 제공한 뒤 새 실행을 제출합니다. 누락값을 추정하거나 외부 야구 사이트에서 채우지 않습니다.

## 관측 지표

다음 Prometheus 지표를 실행 ID 상태 조회 및 JobRunr 기록과 함께 확인합니다.

- `ai_ingest_submissions_total{trigger_source,result}`: 생성/중복 제출
- `ai_ingest_active_runs{trigger_source}`: DB에 저장된 현재 `RUNNING` 실행
- `ai_ingest_queued_runs{trigger_source}`: DB에 저장된 현재 `QUEUED` 실행
- `ai_ingest_run_completions_total{status,trigger_source}`: terminal 결과
- `ai_ingest_run_duration_seconds{status,trigger_source}`: 실행 시간
- `ai_ingest_table_duration_seconds{source_table}`: 허용된 테이블별 처리 시간
- `ai_ingest_table_source_rows_total{source_table}`: 허용된 테이블별 처리 원본 행 수
- `ai_ingest_table_written_chunks_total{source_table}`: 허용된 테이블별 작성 청크
- `ai_ingest_watermark_lag_seconds{source_table}`: 허용된 테이블별 최근 성공 watermark 지연
- `ai_ingest_lease_recoveries_total{result}`: 만료 리스 재대기/최종 실패 수
- `ai_ingest_heartbeats_total{result}`: `success`, `retry`, `rejected`, `exhausted` heartbeat 결과
- `ai_ingest_checkpoint_events_total{source_table,result}`: durable checkpoint lifecycle. `created`는 첫 checkpoint row 생성, `advanced`는 fenced batch cursor commit, `completed`는 table completion commit, `resumed`는 기존 row 사용, `incompatible`은 `IngestCheckpointIncompatibleError` classification, `rejected`는 그 밖의 `IngestCheckpointError` classification을 뜻합니다. Checkpoint metric results are limited to `created`, `advanced`, `completed`, `resumed`, `incompatible`, and `rejected`. Operators observe checkpoint failures through `ai_ingest_checkpoint_events_total{source_table,result}` and AI worker logs with `run_id` and `error_type`. `source_table`과 `result`는 고정된 bounded label 집합입니다.

Backend Prometheus에서는 다음 지표를 함께 확인합니다.

- `backend_ai_ingest_submissions_total{result,deduplicated}`: 제출 성공/실패와 중복 제거 결과
- `backend_ai_ingest_monitor_terminal_total{status}`: backend가 관측한 terminal 상태
- `backend_ai_ingest_orchestration_duration_seconds{status}`: backend 기준 전체 오케스트레이션 시간
- `backend_ai_ingest_cache_invalidations_total{result}`: 성공 후 읽기 캐시 무효화 결과
- `backend_ai_ingest_manual_data_required_total`: 운영자 수동 데이터 인계 횟수

라벨은 고정 집합으로 정규화되며 run ID, 토큰, 오류 본문을 라벨에 넣지 않습니다.

## 롤백 (rollback)

전용 coordination pool 변경만 비활성화하는 환경 플래그는 없습니다. 코드 롤백 시 이전 버전의 단일 공용 풀이 복원되지만 `ai_ingest_runs`, watermark, `rag_chunks` 데이터는 그대로 호환됩니다.

1. `AI_INGEST_ENABLED=false`로 backend를 재배포해 새 반복 제출을 중단합니다.
2. 이미 생성된 실행은 terminal 상태가 될 때까지 AI worker를 유지합니다. 긴급 중단 시 run ID와 lease 만료 시각을 기록합니다.
3. Rollback preserves migration 004 and retained checkpoint audit rows; older code ignores them. 애플리케이션 코드를 롤백하되 003과 `004_ai_ingest_checkpoints.sql`의 테이블·인덱스는 즉시 삭제하지 않습니다. No automatic checkpoint retention job exists.
4. 캐시가 성공 후 비워졌는지 확인하고, 필요하면 backend의 기존 캐시 운영 절차로만 무효화합니다.
5. 재활성화 전에 JobRunr에 `ai-rag-ingestion` recurring 작업이 하나뿐인지 확인합니다.

## 검증 범위와 잔여 위험

이 절차의 unit/integration tests는 disposable fake connection으로 cursor, transaction, lease fence, resume을 검증합니다. A separately approved local-only disposable live PostgreSQL smoke is required to remove the residual PostgreSQL risk. 별도 승인 없이는 이 smoke를 실행하지 않으며, shared 또는 production database에서 실행하지 않습니다.

2026-07-22 검증에서는 로컬 `postgres:16-alpine` disposable database에 003과 `004_ai_ingest_checkpoints.sql`을 적용하고 004 재적용의 멱등성까지 확인했습니다. 운영 `ingest_table`/checkpoint repository/run store/heartbeat loop를 임베딩 없이 실행해 첫 2행 commit 직후 중단, 저장 cursor 뒤 4행 keyset 재개, 완료 checkpoint의 source SELECT 생략, 고정 cutoff 이후 행 제외, lease 만료 후 재큐잉·owner 인계, transient heartbeat 재시도 성공을 확인했습니다. 컨테이너와 one-shot harness는 검증 직후 제거했습니다. 이 검증은 checkpoint PostgreSQL binding 위험을 제거하지만 pgvector 포함 전체 001 schema migration과 locale/collation matrix는 다루지 않습니다.
