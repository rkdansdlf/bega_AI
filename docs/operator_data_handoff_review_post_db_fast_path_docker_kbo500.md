# Operator Data Handoff Review: post_db_fast_path_docker_kbo500

## 기준 산출물
- Source taxonomy: `reports/operator_data_required_taxonomy_post_db_fast_path_docker_kbo500.json`
- Queue CSV: `reports/operator_data_handoff_queue_post_db_fast_path_docker_kbo500.csv`
- Field CSV: `reports/operator_data_handoff_fields_post_db_fast_path_docker_kbo500.csv`
- Summary JSON: `reports/operator_data_handoff_queue_post_db_fast_path_docker_kbo500_summary.json`

## 요약
- 총 운영자 큐 항목: `194`
- 총 운영자 입력 필드 row: `1857`
- Field CSV는 source metadata를 별도 컬럼으로 두지 않는다. `source_name`, `source_checked_at`, `is_verified`, `confidence`는 모두 `field_name` row이며 운영자는 해당 row의 `operator_value`를 채운다.
- Docker KBO 500 audit 기준 answerability failure는 `0`건이다.
- 이 산출물은 운영자 입력 준비용이며 DB ingest, 외부 크롤링, 웹 검색, 자동 데이터 보정을 수행하지 않는다.

## Regeneration Path

`bega_AI` 디렉터리에서 기본 인자만으로 이 bundle을 다시 생성한다.

```bash
.venv/bin/python scripts/build_operator_data_handoff.py --print-summary
```

예상 summary:

- `source_taxonomy_path`: `reports/operator_data_required_taxonomy_post_db_fast_path_docker_kbo500.json`
- `total_queue_items`: `194`
- `total_field_rows`: `1857`
- `priority_counts`: `P0=86`, `P1=38`, `P2=20`, `P3=50`

검증은 읽기 전용이다. DB 접속 없이 구조 검증만 확인할 때는 다음 명령을 사용한다.

```bash
.venv/bin/python scripts/validate_operator_data_handoff.py --skip-db-checks --no-strict
```

빈 운영자 queue 기준 예상 결과:

- `issue_counts.error`: `0`
- `issue_counts.warning`: `1` (`db_checks_skipped`)
- `operator_status_counts.pending`: `194`
- `apply_eligible_count`: `0`

## Field CSV Contract

Field CSV header는 다음 9개 컬럼과 순서를 정확히 유지해야 한다.

```text
queue_id,domain,contract_code,question,field_name,field_description,required,operator_value,operator_notes
```

운영자는 각 `queue_id`의 모든 `field_name` row를 확인하고 `operator_value`에 값을 입력한다. 검증 대상으로 넘기려면 Queue CSV의 `operator_status`를 `ready_for_validation` 또는 `validated`로 바꾸고, 다음 metadata row도 함께 채운다.

- `source_name`: 운영자가 확인한 내부/공식 원천 이름
- `source_checked_at`: ISO 날짜 또는 datetime
- `is_verified`: `true`
- `confidence`: `0.70` 이상

## Count Breakdown

### Priority Counts
| Priority | Count |
| --- | ---: |
| P0 | 86 |
| P1 | 38 |
| P2 | 20 |
| P3 | 50 |

### Domain Counts
| Domain | Count |
| --- | ---: |
| season_meta | 2 |
| schedule_window | 38 |
| game_day_lineup | 24 |
| roster_news | 22 |
| venue_ticket | 26 |
| broadcast_media | 7 |
| fan_event | 5 |
| unsupported_external | 17 |
| db_fast_path_candidate | 3 |
| subjective_prediction | 50 |

## 운영자 처리 순서
1. `P0`부터 처리한다: 시즌 메타, 일정 범위, 경기 당일 라인업, 로스터 뉴스.
2. `schedule_window`는 기준일, 경기 ID, 팀명, 경기 상태, 점수, 시작 시각을 내부 데이터와 맞춘다.
3. `game_day_lineup`은 선발, 라인업, 투수 결과, 경기 상황성 질문을 분리해서 채운다.
4. `roster_news`는 부상, 복귀, 콜업/말소, 계약, 트레이드의 기준일과 확인 원천을 남긴다.
5. `P3` subjective 항목은 별도 운영 기준 합의 후 처리한다.

## V1 구현 상태
- `validate_operator_data_handoff.py`는 CSV 구조, source metadata, domain별 값, DB 대조, P0-only apply eligibility를 검증한다.
- `ingest_operator_data_handoff.py`는 validation normalized JSONL만 입력으로 받으며 기본 dry-run, 명시적 `--apply`, payload hash idempotency, P0-only apply, lineup schema/conflict-target guard를 제공한다.
- `operator_data_query.py`는 `season_meta`, `schedule_window`, `game_day_lineup`, `roster_news`만 read-only fast-path로 조회한다.
- `operator_data_recovery_gate.py`는 validation/ingest dry-run 산출물을 읽어 apply 가능 여부를 판정한다.
- `build_operator_data_p0_input_packet.py`는 P0 도메인 4개만 운영자 입력용 packet으로 분리하고 `operator_value`를 자동으로 채우지 않는다.
- `audit_operator_data_p0_input_packet.py`는 P0 packet이 원본 handoff 구조를 유지하는지, ready row가 최소 입력 조건을 만족하는지 strict validation 전에 read-only로 점검한다.
- `check_operator_data_p0_db_prereqs.py`는 P0 target/read table schema와 lineup conflict target을 DB URL 기준으로 read-only 확인한다.
- `build_operator_data_p0_smoke_set.py`와 `verify_operator_data_p0_smoke.py`는 P0 focused smoke 입력(recovered/manual-control/combined)과 strict 결과 검증을 제공한다.
- `summarize_operator_data_p0_recovery_status.py`는 packet/audit/validation/ingest/gate 결과를 post-KBO500 recovery handoff로 요약한다.
- `run_operator_data_p0_filled_intake.py`는 filled P0 packet을 read-only intake pipeline으로 재생하고 stage별 evidence bundle을 만든다.
- fast-path는 기본 비활성화(`OPERATOR_DATA_FAST_PATH_ENABLED=false`)이며, row가 없거나 질문 범위가 부족하면 `MANUAL_BASEBALL_DATA_REQUIRED` 경로를 유지한다.

## Post-KBO500 Runbook
1. P0 input packet을 생성해 `season_meta`, `schedule_window`, `game_day_lineup`, `roster_news` 86개만 운영자 입력 대상으로 분리한다.
2. 운영자가 P0 row를 채우고 `operator_status=ready_for_validation` 또는 `validated`로 변경한다.
3. 미입력, 불일치, 미검증 row는 복구 대상이 아니며 `MANUAL_BASEBALL_DATA_REQUIRED` 경로를 유지한다.
4. P0 packet QA preflight를 실행해 구조 drift, non-P0 혼입, ready row 입력 누락을 먼저 막는다.
5. Filled intake runner로 packet snapshot, DB prerequisite, strict validation, ingest dry-run, recovery gate, status summary를 한 번에 재생한다.
6. DB 체크를 켠 strict validation을 실행하고 `db_checks.skipped=false`, `issue_counts.error=0`을 확인한다.
7. ingest dry-run으로 insert/update/noop plan과 issue CSV를 확인한다.
8. recovery gate status가 `pass`인지 확인한다.
9. P0 recovery status summary로 전체 blocker와 증적 경로를 확인한다.
10. dry-run 통과 후 같은 normalized JSONL에만 `--apply`를 실행한다.
11. P0 focused smoke set을 만들고 smoke 환경에서만 `OPERATOR_DATA_FAST_PATH_ENABLED=true`를 켠다.
12. smoke verifier가 completion/stream 양쪽에서 recovered/manual-control expectation을 모두 통과해야 한다.

## 아직 남은 것
- 현재 기준 원본 bundle은 194개 모두 `pending`이고 P0 input packet은 86개 모두 `pending`이므로 실제 운영자 입력이 필요하다.
- 실 DB strict validation, P0 ingest dry-run/apply, recovery gate, completion/stream smoke verifier 증적이 아직 필요하다.
- P1 venue/broadcast/fan-event target schema와 query path는 V1 범위 밖이다.
- 외부 야구 크롤링, 웹 검색 기반 수집, 임의 데이터 합성은 계속 금지한다.
