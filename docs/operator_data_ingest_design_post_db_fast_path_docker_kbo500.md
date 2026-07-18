# Operator Data Input, Validation, and Ingest Design

## Scope

이 문서는 Docker KBO 500 감사 후 남은 `operator_data_required` 194개 고유 문항을 운영자가 어떻게 입력하고, 시스템이 어떻게 검증하고, 어떤 경로로 ingest할지 정의한다.

V1 실행 경로는 `P0` 도메인(`season_meta`, `schedule_window`, `game_day_lineup`, `roster_news`)만 post-KBO500 recovery 대상으로 삼는다. `P1` 이상 도메인은 검증 산출물에는 남기지만 target schema와 query path가 완성되기 전까지 `apply_eligible=false`로 유지한다.

이번 설계는 외부 야구 크롤링, 웹 검색, 임의 데이터 합성을 포함하지 않는다. 모든 값은 운영자가 제공한 내부/공식 확인 데이터만 사용한다.

## Current Inputs

기준 산출물은 다음 3개 파일이다. 경로와 명령은 `bega_AI` 디렉터리에서 실행하는 것을 기준으로 한다.

- `reports/operator_data_handoff_queue_post_db_fast_path_docker_kbo500.csv`
- `reports/operator_data_handoff_fields_post_db_fast_path_docker_kbo500.csv`
- `reports/operator_data_handoff_queue_post_db_fast_path_docker_kbo500_summary.json`

현재 분포는 다음과 같다.

| Priority | Count | 처리 의미 |
| --- | ---: | --- |
| P0 | 86 | 시즌 메타, 일정, 라인업, 로스터처럼 우선 구조화할 수 있는 데이터 |
| P1 | 38 | 구장, 중계, 팬 이벤트처럼 운영자 확인 후 답변 가능한 서비스성 데이터 |
| P2 | 20 | 지원 여부나 추가 도구화 판단이 필요한 항목 |
| P3 | 50 | 예측/전망처럼 운영 기준 합의가 먼저 필요한 주관 항목 |

| Domain | Count | 기본 ingest 정책 |
| --- | ---: | --- |
| `season_meta` | 2 | 전용 operator season event 테이블 또는 RAG 보조 청크 |
| `schedule_window` | 38 | `game` row와 대조 후 일정 override/staging |
| `game_day_lineup` | 24 | 기존 `ingest_lineup_manual.py`, `apply_manual_starters.py` 우선 재사용 |
| `roster_news` | 22 | 전용 roster event 테이블 |
| `venue_ticket` | 26 | 전용 venue guide 테이블 및 유효기간 기반 조회 |
| `broadcast_media` | 7 | 전용 broadcast 테이블 |
| `fan_event` | 5 | 전용 fan event 테이블 |
| `unsupported_external` | 17 | 지원 여부만 staging, 기본은 manual contract 유지 |
| `db_fast_path_candidate` | 3 | DB 도구 복구 가능성만 추적, operator fact로 ingest하지 않음 |
| `subjective_prediction` | 50 | 운영 정책 승인 전에는 fact ingest 금지 |

기준 산출물은 다음 명령으로 재생성한다. 이 명령은 외부 데이터 수집이나 DB write를 수행하지 않는다.

```bash
.venv/bin/python scripts/build_operator_data_handoff.py --print-summary
```

예상 count는 `total_queue_items=194`, `total_field_rows=1857`, `P0=86`, `P1=38`, `P2=20`, `P3=50`이다.

## Operator Input Contract

Field CSV header는 다음 9개 컬럼과 순서를 정확히 유지한다.

```text
queue_id,domain,contract_code,question,field_name,field_description,required,operator_value,operator_notes
```

운영자는 `fields` CSV에서 각 `field_name` row의 `operator_value`를 채운다. source metadata는 별도 컬럼이 아니라 다음 `field_name` row의 `operator_value`로 입력한다.

- `source_name`: 운영자가 확인한 내부/공식 원천 이름
- `source_checked_at`: ISO 시각 또는 날짜
- `is_verified`: `true`
- `confidence`: `0.70` 이상. 자동 추론값이 아니라 운영자 판단값
- `operator_notes`: 선택 입력. 충돌, 보류, 유효기간 설명을 남긴다

`queue` CSV의 `operator_status`는 다음 값만 허용한다.

| Status | 의미 |
| --- | --- |
| `pending` | 아직 입력 전 |
| `ready_for_validation` | 운영자 입력 완료, 검증 대기 |
| `validated` | 검증 통과, apply 가능 |
| `applied` | DB 또는 staging 반영 완료 |
| `rejected` | 운영자가 지원하지 않거나 정책상 답변하지 않음 |

DB 적용 대상은 `ready_for_validation` 또는 `validated`만 허용한다. `pending`, `rejected`는 ingest에서 제외한다.

## Validation Design

새 검증 스크립트 `scripts/validate_operator_data_handoff.py`를 둔다. 기본은 읽기 전용이며 DB write를 하지 않는다.

입력:

```bash
BEGA_SKIP_APP_INIT=1 .venv/bin/python scripts/validate_operator_data_handoff.py \
  --queue reports/operator_data_handoff_queue_post_db_fast_path_docker_kbo500.csv \
  --fields reports/operator_data_handoff_fields_post_db_fast_path_docker_kbo500.csv \
  --output-dir reports/operator_data_validation/post_db_fast_path_docker_kbo500
```

검증 단계:

1. 파일 구조 검증
   - Queue/Field CSV header는 정의된 컬럼과 순서를 정확히 맞춰야 함
   - `queue_id` 중복 금지
   - `fields` row가 존재하지 않는 `queue_id` 금지
   - 같은 `queue_id`, `field_name` 중복 금지
   - domain별 required field 누락 금지
   - `required_fields`에 없는 예상 외 `field_name` 금지
   - field row의 `domain`, `contract_code`, `question`은 queue row와 일치해야 함
   - `operator_status` 허용값 검증

2. 공통 메타데이터 검증
   - ingest 대상 row는 모든 required field의 `operator_value`가 비어 있으면 안 된다
   - `is_verified=true`
   - `confidence >= 0.70`
   - `source_name`, `source_checked_at` 필수
   - 날짜/시각은 ISO 형식 또는 명시적으로 허용한 KST 날짜 형식만 허용

3. 도메인별 형식 검증
   - `season_meta`: `season_year` 정수, `event_date` 날짜, `stadium_name` 문자열
   - `schedule_window`: `game_date`, `game_id`, `home_team`, `away_team`, `start_time`, `game_status`
   - `game_day_lineup`: `batting_order` 정수, `team_code` 내부 팀 코드, `position` 비어 있지 않음
   - `roster_news`: `roster_event_type` 허용 enum, `effective_date` 날짜
   - `venue_ticket`: `valid_from <= valid_to`, `topic_type` 허용 enum
   - `broadcast_media`: `media_type` 허용 enum, `game_id` 또는 날짜/팀 조합 필요
   - `fan_event`: `event_type` 허용 enum, 특정 경기 이벤트면 `game_id` 필요
   - `subjective_prediction`: `ranked_entities`, `operator_basis`, `selection_criteria`가 모두 있어도 정책 승인 전에는 `validated`까지만 허용하고 apply 금지

4. 내부 DB 대조 검증
   - `game_id`는 `game` 테이블에 존재해야 한다
   - `team_code`는 내부 팀 resolver 또는 팀 테이블 기준으로 해석 가능해야 한다
   - `player_name`은 `player_basic`에서 정확히 1건으로 해석되어야 한다. 0건 또는 다건이면 `MANUAL_BASEBALL_DATA_REQUIRED` 유지
   - `schedule_window`의 home/away team은 기존 `game` row와 충돌하면 error
   - `game_day_lineup`은 같은 `game_id`, `team_code`, `batting_order`에 서로 다른 선수가 중복되면 error

출력:

- `operator_data_validation_summary.json`
- `operator_data_validation_issues.csv`
- `operator_data_normalized_rows.jsonl`
- `operator_data_apply_plan.csv`

`operator_data_normalized_rows.jsonl`은 ingest 스크립트의 유일한 입력으로 사용한다. 사람이 채운 CSV를 바로 DB에 쓰지 않는다.

## Staging and Ingest Design

새 ingest 스크립트 `scripts/ingest_operator_data_handoff.py`를 둔다. 기본은 `--dry-run`, 실제 적용은 `--apply`가 있어야만 수행한다.

```bash
BEGA_SKIP_APP_INIT=1 .venv/bin/python scripts/ingest_operator_data_handoff.py \
  --normalized reports/operator_data_validation/post_db_fast_path_docker_kbo500/operator_data_normalized_rows.jsonl \
  --dry-run
```

적용 시에는 검증 산출물의 `validation_status=pass` row만 처리한다.

```bash
BEGA_SKIP_APP_INIT=1 .venv/bin/python scripts/ingest_operator_data_handoff.py \
  --normalized reports/operator_data_validation/post_db_fast_path_docker_kbo500/operator_data_normalized_rows.jsonl \
  --apply
```

공통 staging 테이블을 먼저 둔다.

```sql
create table if not exists operator_data_items (
  queue_id text primary key,
  priority text,
  domain text not null,
  contract_code text not null,
  question text not null,
  operator_status text not null,
  validation_status text not null,
  apply_target text,
  payload jsonb not null,
  payload_hash text not null,
  source_name text not null,
  source_checked_at timestamptz not null,
  is_verified boolean not null,
  confidence numeric not null,
  applied_at timestamptz,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);
```

idempotency 기준:

- 같은 `queue_id`와 같은 `payload_hash`는 재실행해도 no-op
- 같은 `queue_id`에 다른 `payload_hash`가 있으면 `--allow-overwrite` 없이는 실패
- domain target table에 이미 다른 값이 있으면 conflict report만 만들고 기본은 실패

## Domain Apply Targets

| Domain | Apply target | 정책 |
| --- | --- | --- |
| `season_meta` | `operator_season_events` 또는 `rag_chunks(source_table='operator_season_events')` | 확정 이벤트만 적용. 미확정 이벤트는 `future_event_pending` 유지 |
| `schedule_window` | `operator_schedule_items` staging, 기존 `game`과 충돌 없을 때만 optional update | 운영자 값이 기존 `game`과 다르면 overwrite 금지 |
| `game_day_lineup` | 기존 `game_lineups`, `game.home_pitcher`, `game.away_pitcher` | `ingest_lineup_manual.py`, `apply_manual_starters.py` 재사용 |
| `roster_news` | `operator_roster_events` | 기준일, 팀, 선수, 이벤트 타입 필수 |
| `venue_ticket` | V1 future only | `operator_venue_guides` schema/query path 전까지 apply 금지 |
| `broadcast_media` | V1 future only | `operator_broadcast_items` schema/query path 전까지 apply 금지 |
| `fan_event` | V1 future only | `operator_fan_events` schema/query path 전까지 apply 금지 |
| `unsupported_external` | V1 no ingest | `supported_by_operator=false`면 답변 복구하지 않음 |
| `db_fast_path_candidate` | ingest 없음 | 도구 복구 backlog로만 유지 |
| `subjective_prediction` | V1 no ingest | 운영 기준 합의 전 answerable fact로 쓰지 않음 |

## RAG Policy

operator-data는 먼저 구조화 테이블로 답한다. RAG 청크는 다음 경우에만 보조로 만든다.

- `venue_ticket`, `fan_event`, `season_meta`처럼 짧은 설명형 답변이 필요한 경우
- `valid_from`, `valid_to`, `expires_at`, `quality_score`, `source_type='operator_provided'`를 모두 채울 수 있는 경우
- embedding은 월간 배치 정책을 따른다. 즉시 API 호출이 필요한 ingest는 기본 금지한다

RAG 청크 생성 전에도 `operator_data_items`의 `payload_hash`를 기준으로 content hash를 만들어 중복 생성을 막는다.

## Chatbot Recovery Path

V1 ingest 후 답변 경로는 다음 순서로 둔다.

1. operator P0 domain table fast-path, `OPERATOR_DATA_FAST_PATH_ENABLED=true`인 smoke/rollout 환경에서만 활성화
2. 기존 static/manual gate 또는 DB fast-path
3. `MANUAL_BASEBALL_DATA_REQUIRED`

`operator_data_required` 문항이 자동으로 answerable이 되는 것은 아니다. 해당 도메인에 검증 완료 row가 있고, 기준일/팀/경기/선수 조건이 질문과 매칭될 때만 답한다.
`MANUAL_BASEBALL_DATA_REQUIRED`는 기존 호환 marker이며, 새 산출물에서는 `BASEBALL_DATA_SYNC_REQUIRED`와 `trusted_baseball_data_project` handoff 정보를 함께 노출한다.

답변에는 다음을 포함한다.

- 기준일 또는 유효기간
- 검증된 trusted baseball data sync 기준이라는 표현
- 값이 만료되었거나 검증되지 않았으면 답하지 않고 manual contract 유지

## Rollout Plan

1. 검증기만 구현
   - 194개 CSV를 읽어 issue report를 만든다
   - DB write 없음

2. P0 dry-run
   - `season_meta`, `schedule_window`, `game_day_lineup`, `roster_news` 86건만 apply plan 생성
   - error 0건 전까지 apply 금지

3. P0 apply
   - `operator_data_items` staging 반영
   - 라인업/선발은 기존 manual scripts로 분기 적용
   - schedule은 기존 `game`과 충돌 없을 때만 별도 target에 반영

4. P0 focused audit
   - P0 질문 subset completion/stream 실행
   - answerability failure 0
   - 검증되지 않은 row에 대한 답변 생성 0

5. P1 apply
   - venue, broadcast, fan event를 유효기간 기반으로 적용
   - 만료 데이터는 답변하지 않는지 확인

6. P2/P3 policy gate
   - `unsupported_external`, `subjective_prediction`은 운영자 정책 승인 없이는 validation 산출물까지만 허용하고 DB apply 금지
   - 승인된 경우에도 답변 문구에 운영 기준과 기준일을 명시

## V1 Implementation Status

- `validate_operator_data_handoff.py`는 CSV shape, source metadata, domain values, DB cross-check, P0-only apply eligibility를 구현했다.
- `ingest_operator_data_handoff.py`는 normalized JSONL만 입력으로 받고, 기본 dry-run, `--apply` 명시 적용, idempotent payload hash, P0-only domain selection, lineup schema/conflict-target checks를 구현했다.
- `operator_data_query.py`는 `season_meta`, `schedule_window`, `game_day_lineup`, `roster_news` read-only fast-path를 제공한다. missing row, schema error, low confidence, underspecified lineup/roster query는 `None`을 반환해 manual contract를 유지한다.
- `operator_data_recovery_gate.py`는 validation/ingest dry-run 산출물을 읽어 apply 가능 여부를 read-only로 판정하고, 미충족 row는 `baseball_data_sync_required_rows.csv`로 외부 trusted sync 요구사항을 발행한다.
- `build_operator_data_p0_input_packet.py`는 원본 handoff CSV에서 P0 도메인 4개만 별도 operator input packet으로 복사한다. `operator_value`는 자동으로 채우지 않는다.
- `audit_operator_data_p0_input_packet.py`는 운영자가 채운 P0 packet을 strict validation 전에 read-only로 점검한다. 일부 P0 row만 ready여도 허용하며, pending/rejected/blocked row는 manual contract 대상으로 남긴다.
- `check_operator_data_p0_db_prereqs.py`는 DB URL이 주어졌을 때 P0 target/read table columns와 `game_lineups(game_id, team_code, batting_order)` conflict target을 read-only로 확인한다.
- `build_operator_data_p0_smoke_set.py`와 `verify_operator_data_p0_smoke.py`는 P0 recovered/manual-control smoke 입력과 결과 검증을 담당한다.
- `summarize_operator_data_p0_recovery_status.py`는 P0 packet/audit/validation/ingest/gate 산출물을 하나의 post-KBO500 recovery status handoff로 묶는다.
- `run_operator_data_p0_filled_intake.py`는 운영자가 채운 P0 packet을 snapshot한 뒤 audit, DB prereq, strict validation, ingest dry-run, recovery gate, status summary를 read-only로 한 번에 재생한다.
- `OPERATOR_DATA_FAST_PATH_ENABLED` 기본값은 `false`다.

## Post-KBO500 Recovery Runbook

1. P0 input packet을 생성한다. 출력은 `season_meta`, `schedule_window`, `game_day_lineup`, `roster_news`만 포함해야 한다.
2. P0 row 데이터는 외부 trusted baseball data sync 프로젝트가 수집/검증해 내부 DB 또는 handoff 산출물로 반영한다. 미수집, 불일치, 미검증 row는 복구 대상이 아니며 `MANUAL_BASEBALL_DATA_REQUIRED` 호환 경로와 `BASEBALL_DATA_SYNC_REQUIRED` handoff를 유지한다.
3. P0 packet QA preflight를 실행한다. `status=pass` 또는 `warning`이면 strict validation으로 갈 수 있고, `fail`이면 packet을 먼저 고친다. 운영 gate에서 최소 1개 ready row를 강제하려면 `--require-ready`를 사용한다.
4. Filled intake runner를 실행해 packet snapshot, P0 packet QA, DB prerequisite, strict validation, ingest dry-run, recovery gate, status summary 산출물을 한 번에 생성한다. DB URL 누락, target schema 누락, lineup conflict target 누락이 있으면 status는 `blocked`여야 한다.
5. DB 체크를 켠 strict validation을 실행한다. `db_checks.skipped=false`, `issue_counts.error=0`, `apply_eligible_count > 0`이어야 다음 단계로 간다.
6. ingest dry-run을 실행한다. `operator_data_ingest_issues.csv`가 header-only이고 plan action이 의도한 insert/update/noop인지 확인한다.
7. recovery gate를 실행한다. gate status가 `pass`일 때만 apply 또는 focused smoke로 넘어간다.
8. P0 recovery status summary를 생성해 packet/audit/db-prereq/validation/ingest/gate 증적과 blocker를 한 파일에서 확인한다.
9. `--apply`는 dry-run이 통과한 같은 normalized JSONL에만 실행한다. 다른 payload hash overwrite는 `--allow-overwrite` 승인 전 금지한다.
10. P0 smoke set을 생성한다. `apply_eligible=true` row는 recovered expectation, 나머지 P0 row는 manual-control expectation으로 분리한다.
11. smoke 환경에서만 `OPERATOR_DATA_FAST_PATH_ENABLED=true`를 켜고 completion/stream audit을 실행한다.
12. smoke verifier로 recovered P0 질문은 operator-data 답변을 반환하고, 미수집/미검증/범위 불일치 질문은 `MANUAL_BASEBALL_DATA_REQUIRED` 호환 marker와 `BASEBALL_DATA_SYNC_REQUIRED` handoff를 유지하는지 확인한다.

## Verification Commands

정적 검증:

```bash
BEGA_SKIP_APP_INIT=1 .venv/bin/python -m py_compile \
  scripts/validate_operator_data_handoff.py \
  scripts/ingest_operator_data_handoff.py \
  scripts/operator_data_recovery_gate.py \
  scripts/build_operator_data_p0_input_packet.py \
  scripts/audit_operator_data_p0_input_packet.py \
  scripts/check_operator_data_p0_db_prereqs.py \
  scripts/run_operator_data_p0_filled_intake.py \
  scripts/build_operator_data_p0_smoke_set.py \
  scripts/verify_operator_data_p0_smoke.py \
  scripts/summarize_operator_data_p0_recovery_status.py \
  scripts/analyze_operator_data_required.py \
  scripts/build_operator_data_handoff.py
```

단위 테스트:

```bash
BEGA_SKIP_APP_INIT=1 .venv/bin/python -m pytest \
  tests/test_analyze_operator_data_required.py \
  tests/test_build_operator_data_handoff.py \
  tests/test_operator_data_validation.py \
  tests/test_operator_data_ingest.py \
  tests/test_operator_data_recovery_gate.py \
  tests/test_operator_data_p0_input_packet.py \
  tests/test_operator_data_p0_input_audit.py \
  tests/test_operator_data_db_prereqs.py \
  tests/test_operator_data_p0_filled_intake.py \
  tests/test_operator_data_p0_recovery_status.py \
  tests/test_operator_data_p0_smoke.py \
  tests/test_smoke_chatbot_quality_summary.py
```

운영 dry-run:

```bash
BEGA_SKIP_APP_INIT=1 .venv/bin/python scripts/build_operator_data_p0_input_packet.py \
  --queue reports/operator_data_handoff_queue_post_db_fast_path_docker_kbo500.csv \
  --fields reports/operator_data_handoff_fields_post_db_fast_path_docker_kbo500.csv \
  --output-dir reports/operator_data_operator_packet/post_db_fast_path_docker_kbo500

BEGA_SKIP_APP_INIT=1 .venv/bin/python scripts/audit_operator_data_p0_input_packet.py \
  --queue reports/operator_data_operator_packet/post_db_fast_path_docker_kbo500/p0_queue.csv \
  --fields reports/operator_data_operator_packet/post_db_fast_path_docker_kbo500/p0_fields.csv \
  --source-queue reports/operator_data_handoff_queue_post_db_fast_path_docker_kbo500.csv \
  --source-fields reports/operator_data_handoff_fields_post_db_fast_path_docker_kbo500.csv \
  --output-dir reports/operator_data_operator_packet_audit/post_db_fast_path_docker_kbo500

BEGA_SKIP_APP_INIT=1 .venv/bin/python scripts/check_operator_data_p0_db_prereqs.py \
  --db-url "$POSTGRES_DB_URL" \
  --output-dir reports/operator_data_db_prereqs/post_db_fast_path_docker_kbo500

BEGA_SKIP_APP_INIT=1 .venv/bin/python scripts/run_operator_data_p0_filled_intake.py \
  --queue reports/operator_data_operator_packet/post_db_fast_path_docker_kbo500/p0_queue.csv \
  --fields reports/operator_data_operator_packet/post_db_fast_path_docker_kbo500/p0_fields.csv \
  --source-queue reports/operator_data_handoff_queue_post_db_fast_path_docker_kbo500.csv \
  --source-fields reports/operator_data_handoff_fields_post_db_fast_path_docker_kbo500.csv \
  --db-url "$POSTGRES_DB_URL" \
  --output-dir reports/operator_data_p0_filled_intake/post_db_fast_path_docker_kbo500
```

Filled-intake recovery readiness requires DB checks. Do not use
`--skip-db-checks` for the recovery gate; that mode is only for structural
baseline inspection of an unfilled packet.

P0 focused smoke:

```bash
BEGA_SKIP_APP_INIT=1 .venv/bin/python scripts/build_operator_data_p0_smoke_set.py \
  --normalized reports/operator_data_p0_filled_intake/post_db_fast_path_docker_kbo500/validation/operator_data_normalized_rows.jsonl \
  --output-dir reports/operator_data_smoke

OPERATOR_DATA_FAST_PATH_ENABLED=true .venv/bin/python scripts/smoke_chatbot_quality.py \
  --chat-question-list scripts/smoke_chatbot_operator_data_p0_all.txt \
  --chat-batch-size <recovered_count + manual_control_count> \
  --output reports/operator_data_smoke/p0_all_full.json \
  --summary-output reports/operator_data_smoke/p0_all_summary.json

BEGA_SKIP_APP_INIT=1 .venv/bin/python scripts/verify_operator_data_p0_smoke.py \
  --report reports/operator_data_smoke/p0_all_full.json \
  --expectations reports/operator_data_smoke/operator_data_p0_smoke_expectations.json \
  --output-dir reports/operator_data_smoke
```

## Acceptance Criteria

- 194개 queue item이 `pending`, `ready_for_validation`, `validated`, `applied`, `rejected` 중 하나로 상태 관리된다
- 검증 대상 row의 required field 누락 0건
- `is_verified=false` 또는 `confidence < 0.70` row는 apply되지 않는다
- DB target 충돌이 있는 row는 apply되지 않고 conflict report에 남는다
- `game_day_lineup`은 player name이 내부 `player_basic`에서 정확히 1건으로 해석될 때만 적용된다
- P0 apply 후 P0 focused audit에서 answerability failure 0건
- P1 apply 후 만료/미검증 venue, broadcast, fan event가 답변에 사용되지 않는다
- P2/P3는 운영 정책 승인 전 fact 답변으로 전환되지 않는다

## Remaining Before Production Use

1. 운영자가 P0 input packet의 실제 값을 채워야 한다. 현재 기준 원본 bundle은 194개 모두 `pending`이고 P0 packet은 86개 모두 `pending`이다.
2. 실제 DB strict validation과 ingest dry-run/apply 증적이 필요하다.
3. P0 focused smoke set으로 completion/stream 양쪽에서 recovery와 manual fallback을 확인해야 한다.
4. P1 venue/broadcast/fan-event target schema와 query path는 V1 범위 밖이다.
