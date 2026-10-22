# rag_chunks.index_status (tombstone) 운영 가이드

KBO_playwright([#57](https://github.com/rkdansdlf/KBO_playwright/pull/57) 등)는 RAG 청크를
`index_status='DELETED'`로 tombstone 합니다. bega_AI PostgreSQL 검색은
`COALESCE(index_status, 'ACTIVE') NOT IN ('DELETED', 'TOMBSTONED', 'PURGED')` 로 이 청크를
제외하고, `scripts/sync_rag_chunks.py` 는 상태를 대상 DB로 전파합니다.

## 적용 (migration 010)

```bash
AI_SCHEMA_DB_URL='postgresql://...' ./scripts/migrate_ai_runtime_schema.sh
# 또는 단독 실행
psql --set ON_ERROR_STOP=1 -f app/db/migrations/010_rag_chunks_index_status.sql "$DB_URL"
```

- additive / 멱등. 컬럼이 이미 있으면(KBO_playwright 관리 DB) no-op.
- `ADD COLUMN` 은 ACCESS EXCLUSIVE 락을 잡습니다. 긴 트랜잭션 뒤에서 락을 기다리는 동안
  이후의 모든 `SELECT` 가 그 뒤에 줄을 서므로(검증 시 SELECT 6초 대기 관측), 파일 안에서
  `SET lock_timeout = '5s'` 로 대기를 제한합니다. 시간 초과 시 psql 이 실패하므로
  `pg_stat_activity` 에서 긴 트랜잭션을 확인한 뒤 다시 실행하면 됩니다. 수동으로 ALTER 를
  실행할 때도 반드시 `SET lock_timeout = '5s';` 를 먼저 실행하세요.
- 앱은 컬럼을 런타임에 감지합니다. migration 전에 배포해도 검색은 실패하지 않고
  (필터 생략 + 경고 로그), 적용 후 최대 5분(음성 캐시 TTL) 안에 재시작 없이 필터가 켜집니다.

## sync_rag_chunks.py

| source 컬럼 | dest 컬럼 | 동작 |
|---|---|---|
| 있음 | 있음 | 그대로 복사(NULL→ACTIVE). source 의 DELETED 는 dest 에서도 DELETED |
| 있음 | 없음 | 쓰기 전에 거부 — dest 에 010 을 먼저 적용 |
| 없음 | 있음 | 기존 dest 행의 `index_status` 유지(UPDATE 에서 제외), 새 행은 dest 기본값(ACTIVE). **`--truncate-dest` 는 거부**(모든 tombstone 이 ACTIVE 로 부활하므로) |
| 없음 | 없음 | 전파할 상태 없음(경고) |

`is_active` 는 `COALESCE(is_active, true)` 로 복사됩니다(명시적 false 유지). source 에 컬럼이
없는 경우 dest 에서 tombstone 된 행의 `is_active=false` 는 source 값으로 덮일 수 있지만,
`index_status='DELETED'` 가 유지되므로 검색에서는 계속 제외됩니다.

## 롤백

- **앱 코드 롤백**: 컬럼은 남겨 두세요. 이전 버전은 컬럼을 무시하므로 호환됩니다
  (tombstone 필터만 사라짐 — `is_active=false` 로 보조 차단되는 행만 숨겨짐).
- **컬럼 삭제(`ALTER TABLE rag_chunks DROP COLUMN index_status`)**: 권장하지 않습니다.
  KBO_playwright 도구가 이 컬럼을 씁니다. 꼭 삭제해야 한다면:
  - 앱은 컬럼 존재를 프로세스 수명 동안 캐시합니다. 삭제 후 첫 검색이 `UndefinedColumn`
    을 만나면 캐시를 무효화하고 필터 없이 한 번 재시도합니다(autocommit 연결 — 기본 풀).
    호출자 트랜잭션 안의 연결이면 그 요청은 실패하고 다음 요청부터 필터 없이 동작합니다.
  - 가장 확실한 방법은 **컬럼 삭제 후 AI 서비스를 재시작**하는 것입니다(롤백 절차에 포함).
  - 삭제에도 `SET lock_timeout = '5s';` 를 먼저 실행하세요.
