# Operator Data Handoff Review: post_256_release_gate_kbo500

## 기준 산출물
- Source taxonomy: `reports/operator_data_required_taxonomy_post_256_release_gate_kbo500.json`
- Queue CSV: `reports/operator_data_handoff_queue_post_256_release_gate_kbo500.csv`
- Field CSV: `reports/operator_data_handoff_fields_post_256_release_gate_kbo500.csv`
- Summary JSON: `reports/operator_data_handoff_queue_post_256_release_gate_kbo500_summary.json`

## 요약
- 총 운영자 큐 항목: `195`
- 총 운영자 입력 필드 row: `1866`
- `recovered_fast_path` 항목은 handoff CSV에서 제외했다.
- 이 산출물은 운영자 입력 준비용이며 DB ingest 또는 자동 데이터 보정은 수행하지 않는다.

## 우선순위 기준
- `P0`: 바로 구조화 입력이 가능한 핵심 야구 데이터
  - `season_meta`
  - `schedule_window`
  - `game_day_lineup`
  - `roster_news`
- `P1`: 운영자 확인이 필요한 서비스/구장/팬 경험 정보
  - `venue_ticket`
  - `broadcast_media`
  - `fan_event`
- `P2`: 지원 여부 또는 fast-path 보강 여부를 운영자가 판정해야 하는 항목
  - `unsupported_external`
  - `db_fast_path_candidate`
- `P3`: 주관 평가 기준 합의가 필요한 항목
  - `subjective_prediction`

## Count Breakdown

### Priority Counts
| Priority | Count |
| --- | ---: |
| P0 | 86 |
| P1 | 38 |
| P2 | 20 |
| P3 | 51 |

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
| subjective_prediction | 51 |

## Queue CSV 컬럼
- `queue_id`: 운영자 큐의 안정 ID
- `priority`: actionability 기준 우선순위
- `priority_reason`: 우선순위 부여 이유
- `domain`: taxonomy domain
- `contract_code`: 운영자 데이터 계약 코드
- `question`: smoke report의 원 질문
- `required_fields`: 운영자가 채워야 하는 계약 필드 목록
- `endpoint_count`: completion/stream 등 동일 질문이 감지된 endpoint 수
- `endpoints`: 해당 질문이 나온 endpoint 목록
- `sample_answer`: MANUAL_BASEBALL_DATA_REQUIRED 샘플 응답
- `operator_status`: 기본값 `pending`
- `operator_owner`: 담당자 입력용 빈 칸
- `operator_notes`: 운영자 메모용 빈 칸

## Field CSV 컬럼
- `queue_id`: Queue CSV와 연결되는 안정 ID
- `domain`, `contract_code`, `question`: 원 큐 항목 식별 정보
- `field_name`: 운영자가 입력해야 하는 필드명
- `field_description`: 필드 설명
- `required`: 현재 모든 계약 필드는 `true`
- `operator_value`: 운영자 입력값
- `source_name`: 운영자가 확인한 내부/공식 원천 이름
- `source_checked_at`: 원천 확인 시각
- `is_verified`: 운영자 검증 완료 여부
- `confidence`: 운영자 판단 신뢰도
- `operator_notes`: 필드 단위 메모

## 운영자 처리 순서
1. `P0`부터 처리한다.
2. `schedule_window`는 기준일, 경기 ID, 팀명, 경기 상태, 점수, 시작 시각이 내부 데이터와 일치하는지 먼저 확인한다.
3. `game_day_lineup`은 선발, 라인업, 투수 결과, 경기 상황성 질문을 분리해서 확인한다.
4. `roster_news`는 부상, 복귀, 콜업/말소, 계약, 트레이드의 기준일과 확인 원천을 반드시 남긴다.
5. `P3` subjective 항목은 별도 운영 기준 합의 후 처리한다.

## 다음 단계
- CSV가 채워진 뒤 별도 작업으로 ingest 설계를 진행해야 한다.
- ingest는 operator-provided data만 사용해야 하며 외부 야구 크롤링, 웹 검색 기반 수집, 자동 합성 보정은 추가하지 않는다.
