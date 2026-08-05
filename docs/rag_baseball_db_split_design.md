# AI 서비스 DB 도메인 분리

작성 2026-07-26, 갱신 2026-08-03.
상태: **코드 구현 완료(`418cf2e`, `518fea4`). 데이터 이관·전환 대기.**

## 무엇을 왜 나눴나

AI 서비스와 백엔드가 하나의 PostgreSQL(`bega_backend`, 3.5GB)을 공유하고 있었다.
`rag_chunks`(2233MB)를 다른 호스트로 옮기려 했으나, AI 서비스가 같은 커넥션으로
야구 테이블도 조회해서 떼어낼 수 없었다.

이후 대량 임베딩과 `rag_chunks` 적재가 **별도 프로젝트로 이관**되면서 이 서비스는
`rag_chunks`에 대해 읽기 전용이 되었다. 그에 맞춰 분리 단위를 셋으로 잡았다.

| 도메인 | 환경변수 | 내용 | 위치 |
|---|---|---|---|
| cache | `POSTGRES_DB_URL` | chat/coach 캐시, 인제스트 상태 | **로컬 고정** |
| rag | `AI_RAG_DB_URL` | `rag_chunks`, `rag_retrieval_events` | 원격 가능 |
| baseball | `AI_BASEBALL_DB_URL` | `game`, `player_season_*` 등 | 원격 가능 |

**캐시를 rag 와 함께 보내지 않는 것이 요점이다.** 캐시는 거의 모든 요청에서 쓰이므로
원격 왕복이 응답 지연에 그대로 얹힌다. 반면 `rag_chunks`는 조회뿐이고
`rag_retrieval_events` 쓰기는 `asyncio.create_task` 로 fire-and-forget 이라
원격에 둬도 응답을 막지 않는다.

## 분리가 가능했던 이유

**교차 조인이 없다.** 두 도메인 테이블을 함께 언급하는 파일은 `core/rag_storage.py`와
`tools/regulation_query.py` 둘뿐인데, 그 안의 SQL 블록 26개 중 두 도메인을 함께
참조하는 것은 0개였다. 같은 파일에 도메인별 쿼리가 나란히 있을 뿐이다.
→ 쿼리 재작성 없이 **라우팅만으로** 분리된다.

**획득 지점이 하나다.** 모든 DB 접근이
`app/tools/pooled_connection.py:connection_scope()` 를 지나고, RAG 검색은
`app/deps.py:get_rag_pipeline()` 한 곳에서 풀을 주입받는다.

## 구현된 것

```
app/config.py                database_url / rag_db_url / baseball_db_url
                             뒤 둘은 전용 환경변수 미설정 시 database_url 로 폴백
app/deps.py                  풀 3개(+인제스트 조정 풀) 생성·기동·종료
                             get_rag_pipeline() 이 RAG 풀을 주입
app/tools/pooled_connection.py
                             connection_scope(domain="cache"|"rag"|"baseball")
                             기본값 "cache" — 태깅 누락이 원격으로 새지 않게
```

도메인 태깅이 실제로 필요했던 곳:

| 위치 | 도메인 | 근거 |
|---|---|---|
| `tools/document_query.py`, `tools/regulation_query.py` | rag | `FROM rag_chunks` |
| `deps.py:get_rag_pipeline()` | rag | 검색·리트리벌 |
| `tools/team_mapping_loader.py` | baseball | `FROM teams` |
| `routers/coach.py` — `_resolve_target_year`, `_collect_game_evidence`, `_build_manual_data_request` | baseball | `FROM game`, `game_lineups`, `game_summary`, `kbo_seasons` |
| `routers/coach.py` — 나머지 8곳 | cache | `coach_analysis_cache` |

나머지 호출부는 전부 캐시/RAG 라 기본값으로 덮인다.

검증: `tests/test_baseball_db_split.py` 가 폴백과 도메인별 풀 선택을 고정한다.
전체 스위트 4073건 통과.

## 남은 단계

세 URL 이 모두 폴백 상태이므로 현재 코드는 **운영에 배포해도 동작이 바뀌지 않는다.**
아래는 실제로 도메인을 갈라내는 절차다. 각 단계가 환경변수 하나씩이라 개별 롤백된다.

### R-1. 사전 확인 (외부 RAG DB 준비 측)

- pgvector 확장 존재: `CREATE EXTENSION IF NOT EXISTS vector;`
- `rag_chunks` 스키마가 `app/db/schema.sql` 과 일치하는지
- **임베딩 정합성** — 적재에 쓴 모델·차원과 이 서비스의 `EMBED_MODEL`·`EMBED_DIM`
  이 같아야 한다. 다르면 질의 벡터와 저장 벡터의 공간이 달라 검색이 무의미해진다.
  현재 스키마는 `vector(256)` 고정이므로 차원이 바뀌면 컬럼 타입 변경이 선행된다.
- 인덱스: `idx_rag_chunks_*` 15종. 대량 적재 후 생성해야 빠르다.

### R-2. RAG 전환

```bash
# .env.prod 에 추가 (env_file 로 컨테이너에 전달된다)
AI_RAG_DB_URL=postgresql://user:pw@<rag-host>:5432/<db>

cd /home/ubuntu && ./compose.sh up -d --no-build ai-chatbot
```

기동 로그에서 `[DB] RAG connection pool created ... separated=True` 확인.

검증 순서:
1. `GET /health`
2. 챗봇 질의 1건 — RAG 검색 경로가 살아 있는지
3. `scripts/smoke_chatbot.py` 로 **p95 지연을 전환 전과 비교** — 원격 왕복이
   붙으므로 여기서 수용 가능한지 판단한다
4. 코치 분석 1건 — 캐시(로컬)와 야구(로컬)가 여전히 정상인지

롤백: `AI_RAG_DB_URL` 을 지우고 재기동. 데이터 롤백 불필요.

### R-3. 야구 전환 (R-2 안정화 후)

```bash
AI_BASEBALL_DB_URL=postgresql://user:pw@<baseball-host>:5432/<db>
```

같은 방식으로 재기동·검증. 로그의 `[DB] Baseball connection pool created ...
separated=True` 확인. 검증은 코치 분석(야구 조회가 가장 많은 경로)을 중심으로.

주의: 백엔드의 `BASEBALL_DB_URL` 은 **별개**다. 이 단계는 AI 서비스만 옮기므로,
백엔드가 여전히 기존 DB 를 본다면 야구 데이터가 두 곳에 존재하게 된다.
정본을 하나로 유지할 계획이 없으면 두 사본이 갈라진다.

### R-4. 정리

전환이 안정되면 기존 DB 의 `rag_chunks` 를 삭제한다. 그 전까지는 남겨 롤백 여지를
유지한다.

## 미결 사항

1. **`operator_*` 4종** — `operator_data_items`, `operator_roster_events`,
   `operator_schedule_items`, `operator_season_events`. 스키마상 AI 소유지만 내용은
   운영자가 제공한 야구 데이터이며 `MANUAL_BASEBALL_DATA_REQUIRED` 계약의 일부다.
   현재는 캐시 도메인(기본값)에 있다. 야구가 원격으로 가면 정본 위치를 정해야 한다.
2. **야구 사본 정본화** — R-3 주의 참조.
3. **인제스트 코드의 처분** — 적재가 외부로 나가면서 `scripts/ingest_from_kbo.py`,
   `app/routers/ingest.py`, ingest worker, `ai_ingest_*` 테이블이 쓰이지 않게 된다.
   제거할지 폴백으로 남길지 미정.

## 이 작업이 풀지 못하는 것

Oracle Autonomous 의 `ORA-65114` 와 무관하다. 그것은 ADB 할당량 문제이고 이 분리는
PostgreSQL 쪽이다. 백엔드의 Oracle 탈출은 별도 과제다.
