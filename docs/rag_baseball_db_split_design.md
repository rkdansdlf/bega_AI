# AI 서비스 2-DB 분리 설계 (RAG ↔ 야구)

작성 2026-07-26. 상태: **설계안, 미구현.**

## 배경

AI 서비스와 백엔드가 같은 PostgreSQL 데이터베이스(`bega_backend`, 3.5GB)를 공유한다.
RAG 임베딩을 별도 호스트로 옮기려 했으나, AI 서비스가 같은 커넥션에서 야구 테이블도
조회하고 있어 `rag_chunks`만 떼어낼 수 없다는 것이 확인됐다.

이 문서는 그 결합을 끊는 설계다.

## 현재 상태 (2026-07-26 실측)

```
pgvector-db 컨테이너 (서버 /var/lib/docker/volumes/ubuntu_pgvector_data)
└── bega_backend  3495 MB
    ├── rag_chunks                    2233 MB   ← AI 소유, 64%
    ├── game_events / game_play_by_play 685 MB  ← 야구
    ├── team_daily_roster               87 MB   ← 야구
    └── 그 외 야구 테이블               ~490 MB
```

두 소비자가 같은 DB를 가리킨다.

| 소비자 | 환경변수 | 값 |
|---|---|---|
| AI 서비스 | `POSTGRES_DB_URL` | `postgresql://…@pgvector-db:5432/bega_backend` |
| 백엔드(야구) | `BASEBALL_DB_URL` | `jdbc:postgresql://pgvector-db:5432/bega_backend` |

AI 서비스가 직접 조회하는 야구 테이블: `game`(28곳), `player_season_pitching`(14),
`player_season_batting`(13), `game_events`(5), `game_batting_stats`(5),
`team_standings_daily`(4).

## 실현 가능성 — 교차 조인이 없다

분리의 성패는 "한 SQL 안에서 두 도메인을 조인하는가"에 달렸다. **없다.**

`rag_chunks`와 야구 테이블을 함께 언급하는 파일은 `core/rag_storage.py`와
`tools/regulation_query.py` 둘뿐인데, 두 파일의 SQL 블록 26개를 검사한 결과
**두 도메인을 함께 참조하는 블록은 0개**다. 같은 파일 안에 도메인별 쿼리가
나란히 있을 뿐이다.

→ **쿼리 재작성 없이 라우팅만으로 분리된다.**

## 분리 지점이 하나다

모든 DB 접근이 단일 초크포인트를 지난다.

```
app/deps.py:_create_async_connection_pool()   settings.database_url 로 풀 1개 생성
        ↑
app/deps.py:get_connection_pool()             전역 싱글턴
        ↑
app/tools/pooled_connection.py:connection_scope()   커넥션 획득 표준 경로
```

획득 지점은 총 26곳이며 도메인 분포는 다음과 같다.

| 파일 | 획득 | 도메인 |
|---|---|---|
| `routers/chat_stream.py` | 13 | RAG |
| `routers/coach.py` | 4 | **혼재** (RAG 18 / 야구 21 참조) |
| `core/chat_cache.py` | 1 | RAG |
| `tools/team_mapping_loader.py` | 1 | 야구 |
| `tools/regulation_query.py` | 2 | 혼재 |
| `tools/document_query.py` | 2 | RAG |
| `tools/pooled_connection.py` | 1 | (초크포인트) |

혼재 파일이 둘 있으나 교차 조인이 없으므로 **쿼리 단위로 도메인이 결정된다.**

## 목표 구조

```
AI 서비스
├── RAG 풀      POSTGRES_DB_URL          → rag_chunks, chat_*_cache, ai_ingest_*, …
└── 야구 풀     AI_BASEBALL_DB_URL(신규)  → game, player_season_*, …  (읽기 전용)

백엔드
└── 야구 풀     BASEBALL_DB_URL          → 변경 없음
```

AI 서비스의 야구 접근은 전부 조회다. 야구 풀은 **읽기 전용**으로 열어
`target_session_attrs`를 낮추고 권한도 `SELECT`만 부여한다.

## 테이블 배치

**RAG DB로 이동** (AI 소유, `app/db/schema.sql` + `migrations/`):

```
rag_chunks  rag_ingest_jobs  rag_retrieval_events
chat_response_cache  chat_semantic_response_cache
chat_semantic_cache_shadow_observation
coach_analysis_cache
ai_ingest_runs  ai_ingest_watermarks  ai_ingest_checkpoints
```

**야구 DB에 잔류**: `game*`, `player_*`, `team_*`, `matchup_*`, `stat_rankings` 등.

**미결정 — `operator_*` 4종** (`operator_data_items`, `operator_roster_events`,
`operator_schedule_items`, `operator_season_events`): 스키마상 AI 소유지만 내용은
운영자가 제공한 **야구 데이터**이며 `MANUAL_BASEBALL_DATA_REQUIRED` 계약의 일부다.
`tools/operator_data_query.py`가 야구 테이블을 1회 참조하므로 배치에 따라
그 쿼리의 풀이 달라진다. **결정 필요** (아래 미결 사항 참조).

## 코드 변경

1. **`app/config.py`** — `baseball_db_url` 필드 추가(`AI_BASEBALL_DB_URL`).
   미설정 시 `database_url`로 폴백해 **기존 단일 DB 배포와 하위호환**을 유지한다.
   이 폴백이 있어야 분리 전후로 같은 코드가 돈다.

2. **`app/deps.py`** — `_create_async_connection_pool(conninfo=…)`로 일반화하고
   `get_connection_pool()` / `get_baseball_connection_pool()` 두 개를 노출.
   기동·종료 시 두 풀을 함께 open/close.

3. **`app/tools/pooled_connection.py`** — `connection_scope(conn, *, domain="rag")`.
   `domain`으로 풀을 고르되 기본값은 RAG로 두어 호출부 누락 시 현행 동작을 유지한다.

4. **호출부 26곳 태깅** — 야구 쿼리에만 `domain="baseball"`을 명시. 대부분은
   `chat_stream.py`(13)처럼 파일 전체가 한 도메인이라 기계적이다. `coach.py`와
   `regulation_query.py`만 쿼리별 판단이 필요하다.

5. **`app/db/schema.sql`** — 야구 테이블 참조가 있다면 RAG DB 스키마에서 제외.

## 데이터 이관

분리 자체는 코드 배포만으로 끝나지 않는다. RAG 테이블을 새 DB로 옮겨야 한다.

```bash
# 1) RAG 테이블만 덤프 (약 2.3GB)
pg_dump -h <현재> -U <user> -d bega_backend -Fc \
  -t rag_chunks -t rag_ingest_jobs -t rag_retrieval_events \
  -t chat_response_cache -t chat_semantic_response_cache \
  -t chat_semantic_cache_shadow_observation -t coach_analysis_cache \
  -t ai_ingest_runs -t ai_ingest_watermarks -t ai_ingest_checkpoints \
  -f rag.dump

# 2) 대상 DB 준비 — pgvector 확장이 먼저 있어야 한다
psql -h <대상> -U <user> -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 3) 복원
pg_restore -h <대상> -U <user> -d rag --no-owner --no-privileges -j 4 rag.dump

# 4) 검증 — 행 수와 인덱스가 모두 넘어왔는지
psql -h <대상> -U <user> -d rag -c "SELECT count(*) FROM rag_chunks;"
psql -h <대상> -U <user> -d rag -c "\di idx_rag_chunks_*"
```

`rag_chunks`의 pgvector 인덱스(`idx_rag_chunks_season_year`, `_team_id`,
`_meta_league`)는 복원 후 재생성에 시간이 걸린다. `-j 4`로 병렬화하고,
복원 완료까지 AI 서비스는 기존 DB를 계속 보게 둔다.

## 전환과 롤백

폴백 설계 덕에 무중단 전환이 가능하다.

1. 코드 배포 (`AI_BASEBALL_DB_URL` 미설정 → 두 풀이 같은 DB를 가리킴, **동작 무변화**)
2. RAG 데이터를 새 DB로 복사 (원본 유지)
3. `POSTGRES_DB_URL`을 새 RAG DB로, `AI_BASEBALL_DB_URL`을 기존 DB로 지정 후 재기동
4. 검증
5. 문제 시 3단계의 환경변수만 되돌리고 재기동 — **데이터 롤백 불필요**

원본 RAG 테이블은 안정화가 확인될 때까지 삭제하지 않는다.

## 미결 사항

1. **`operator_*` 4종의 배치** — AI 소유 vs 야구 데이터. 운영자 입력 흐름
   (`MANUAL_BASEBALL_DATA_REQUIRED`)이 어느 쪽 DB를 정본으로 볼지 결정해야 한다.
2. **RAG DB 호스트** — 서버 pgvector 컨테이너에 DB만 추가할지, 별도 호스트로 뺄지.
   노트북 호스팅을 검토했다면 가용성 영향은 챗봇 품질 저하로 한정된다(야구 기능은
   서버 DB에 남으므로 무관). 이 분리의 실질적 이득이 그 격리다.
3. **야구 풀 권한** — 읽기 전용 롤을 새로 만들지, 기존 자격증명을 재사용할지.

## 이 작업이 풀지 못하는 것

**Oracle의 `ORA-65114`와 무관하다.** 그것은 Autonomous Database의 할당량 문제이고
이 분리는 PostgreSQL 쪽 이야기다. B2 재배포 차단은 이 작업으로 해소되지 않는다.
이 분리의 목적은 RAG 데이터의 배치 자유도를 얻는 것이지 용량 확보가 아니다.
