# AI 품질 계약 (Quality Contract)

모든 답변에 대해 "무엇이 검색됐고, 어떤 근거로 생성됐으며, 변경 전후 품질이 유지되는가"를
자동 검증하기 위한 구성요소 지도. 새 AI 기능이 아니라 **검증 가능성**을 위한 계층이다.

| 영역 | 구현 | 위치 |
|---|---|---|
| 캐시 provenance | exact/semantic 캐시가 `verified`·`data_sources`·`answer_sources`·`as_of_date`·fingerprint를 저장/복원. 레거시 행(provenance 없음)은 `verified=false`. fallback 답변은 캐시 저장 안 함 | `app/core/cache_provenance.py`, migration `006` |
| Filter allowlist | retrieval filter 키는 allowlist 컬럼 / `meta.<[A-Za-z0-9_]+>` 만 SQL 식별자가 됨 | `app/core/retrieval_policy.py` |
| Fallback 제약 보존 | factual 질의에서 `player_id`/`team_id`/`season_year` 는 fallback 이 제거하지 않음 (설명형 intent·규정만 완화 허용). 완화 내역은 `constraint_relaxed`/`relaxed_fields` 로 이벤트에 기록 | `retrieval_policy.py`, `rag.py` |
| 지표 출처 구분 | 자체 계산 wRC+/OPS+/WAR 는 `(추정)` 라벨 + `metric_provenance`(version, constants source, calibrated 여부) | `app/core/kbo_metrics.py` |
| Liveness / Readiness | `/health`=프로세스, `/ready`=필수 의존성(503) | `app/main.py`, `app/core/rag_readiness.py` |
| Relevance guard | 질의의 선수/팀/시즌과 충돌하는 chunk 를 컨텍스트 전에 제거 (`RAG_RELEVANCE_GUARD_ENABLED`) | `app/core/relevance_guard.py` |
| Reranker | `score`(모델 없음) / `http`(cross-encoder rerank API, 실패 시 score 순서로 fail-open) | `app/core/reranker.py` |
| Claim↔source | 답변 문장 → `rag_chunks.id` 매핑, 미지지 문장/숫자 보고 (`claim_grounding` meta) | `app/eval/grounding.py` |
| 실제 호출 attribution | 요청(primary) vs 실제 provider/model, `fallback_depth`, `fallback_reason`(에러 클래스/`circuit_open`/`not_configured`). 스트림·일반 응답, fingerprint, 비용 집계, 캐시 provenance 가 모두 이 값을 사용 (`llm_attribution` meta) | `app/core/llm_provider.py` |
| 백엔드 통합 계약 | PostgreSQL/Oracle 이 같은 계약을 따른다: 필터는 공통 allowlist 로 해석하고 **미지원/미지 키는 무시하지 않고 거부**(Oracle 의 기존 silent-drop 제거, JSON path 는 검증된 키만), 결과 행은 entity scope(`season_year`/`team_id`/`player_id`)와 `retrieval_backend`/`index_generation` 을 항상 포함, 요청한 entity 제약을 반환 행에서 재검증(위반 시 fail closed), Oracle generation 게이트(`index_version`), 필수 capability 미충족·게이트 설정 누락·활성 generation 커버리지 부족 시 `/ready` DOWN(`RETRIEVAL_CONTRACT_NOT_MET`) | `app/core/retrieval_contract.py`, `oracle_rag.py`, `rag_readiness.py` |
| Fingerprint | prompt version/hash, planner/retrieval/reranker version, model, embedding signature 를 meta·캐시·eval 에 기록 | `app/core/fingerprint.py` |
| Golden 평가 | retrieval(Recall@5/10, MRR, nDCG, wrong-source, zero-hit) 과 generation(unsupported/numeric/entity hallucination, citation P/R) 을 분리 측정, baseline 대비 회귀 시 exit 1 | `scripts/eval_rag_golden.py`, `evals/` |
| CI 게이트 | `ai-pr-gate` 가 golden 평가를 실행 | `.github/workflows/ai-pr-gate.yml` |
| Ingest→캐시 무효화 | ingest 성공 시 영향 범위(season / 시즌 미지정 시 전체 non-stable) 캐시 삭제 | `app/core/cache_invalidation.py` |
| Semantic cache 동등성 | season/team scope, answer scope, answerability, source, freshness 불일치를 shadow 평가에서 실패 처리 | `app/core/semantic_cache_equivalence.py` |
| Embedding generation | 레지스트리 + 커버리지 감사 후 원자적 활성화/롤백. **물리 저장소**(`RAG_EMBEDDING_STORE=generations`, migration `009`): `rag_chunk_embeddings(generation_id, chunk_id, embedding, content_hash)` 에 세대별 행이 공존하고 세대별 partial HNSW 인덱스로 검색한다. 내용이 바뀌었는데 재임베딩되지 않은 행(`content_hash` 불일치)은 절대 서빙하지 않는다. inline 모드는 기존 signature 필터 | `app/core/embedding_generations.py`, `retrieval.py`, migrations `007`/`009` |
| Provider adapter/circuit | Gemini/OpenRouter 전송을 `LLMProvider` 로 분리, provider 별 CLOSED/OPEN/HALF_OPEN | `app/core/llm_provider.py`, `provider_circuit.py` |
| 실제 usage | provider 보고 토큰 우선(`usage_source=provider`), 없을 때만 estimate | `app/core/llm_usage_accounting.py` |
| Trace | `X-Request-ID` 미들웨어 + retrieval/tool/llm span (Sentry, OTel 있으면 사용) | `app/observability/tracing.py` |
| Sentry 샘플링 | `SENTRY_TRACES_SAMPLE_RATE` / `SENTRY_PROFILES_SAMPLE_RATE` (기본 1.0) | `app/config.py` |
| 이벤트 마이닝 | `rag_retrieval_events` → zero-hit/fallback/relaxed/low-sim/bad-source 후보 (미라벨) | `scripts/mine_retrieval_events.py` |
| Chunk 벤치마크 | `target/max/overlap` 그리드를 benchmark 케이스로 비교 | `scripts/benchmark_chunk_params.py` |

## 배포 전 필수 (순서)

1. **DB migration** — managed 스키마 모드는 `chat_*_cache.provenance_json` 을 계약으로 요구한다.
   `scripts/migrate_ai_runtime_schema.sh` (006, 007 포함)를 앱 배포 **전에** 적용.
2. `RAG_GENERATION_GATE_ENABLED` 는 기본 `false`. 활성 generation 등록·감사
   (`scripts/manage_embedding_generation.py`) 후에만 켠다.
3. Golden baseline(`evals/baselines/*.json`)은 **구조적 시드**다. 실 DB 로
   `scripts/eval_rag_golden.py retrieval --live --write-recorded` 를 돌려 recorded 결과와
   baseline 을 갱신하고, 케이스의 `relevant_doc_keys` 를 운영자가 검증해야 의미 있는 게이트가 된다.

## 알려진 한계 (의도적으로 남김)

- **물리 blue/green 운영 주의** (`RAG_EMBEDDING_STORE=generations`)
  - 도입은 1회: `register g1 --mirror-inline` → `backfill g1`(현재 inline 벡터 복사, 재임베딩 불필요. live ingest 와 동시에 실행해도 안전: 원본 행을 `FOR SHARE SKIP LOCKED` 로 읽어 쓰기 중인 행은 건너뛰고(트리거가 최신값을 미러링) 읽은 행은 쓰기 완료까지 갱신을 막는다. 건너뛴 행은 다음 pass 로 재시도하고 `remaining` 을 실제 테이블로 검증, 남으면 종료 코드 3. autocommit 연결에서 실행) →
    `build-index g1` → `activate g1` → 환경변수 전환 후 재시작. 도입 전에는 아무것도 바뀌지 않는다(기본 `inline`).
  - inline 로 쓰는 기존 writer(ingest/크롤러)는 DB 트리거가 signature 가 일치하는 세대(`mirror_inline`)로
    자동 미러링한다(임베딩 컬럼이 쓰일 때만 발동). **다른 모델의 새 세대**는 별도 writer 가
    `embedding_generations.write_embeddings` 또는 `rag_chunk_embeddings` 로 직접 써야 한다(크롤러 과제).
  - 세대 전환 = 포인터 flip, 롤백 = 손대지 않은 구 세대 행으로 복귀. 롤백 대상 세대는 `drop` 이 거부한다.
  - 미검증: 실 DB(로컬 pgvector 17)에서 전 흐름과 partial HNSW 사용(EXPLAIN)까지 검증했지만, **운영 규모(수십만 행)의
    인덱스 빌드 시간·`maintenance_work_mem`·질의 지연은 측정하지 않았다.** 세대 공존 중에는 벡터 저장 공간이 세대 수만큼
    늘고, inline 쓰기에 트리거 오버헤드가 붙는다. Oracle 경로는 해당 없음(위 `index_version` 설정 방식).
- Oracle 에는 generation 레지스트리가 없어 활성 generation 을 `RAG_ORACLE_ACTIVE_INDEX_VERSION`(= `rag_chunks.index_version`) 설정으로 지정한다. 전환은 설정 변경 + 재시작이며 PostgreSQL 같은 원자적 포인터/롤백은 아니다. Oracle 은 `valid_from/valid_to/expires_at` 컬럼이 없어 수명주기를 `index_status` 로만 판단한다(capability `temporal_filters=false`, 정보용).
- Chunk 벤치마크는 `EMBED_PROVIDER=local` 이면 의미 없음(리포트에 `meaningful:false`).
- Feedback: `POST /ai/chat/feedback`(내부 토큰) → `rag_answer_feedback`(migration 008). BFF 가 호출하도록 연동해야 데이터가 쌓이며, `mine_retrieval_events.py --with-feedback` 가 DOWN 평가를 골든 후보로 합친다.
- `main` branch protection: `scripts/ops/apply_branch_protection.sh OWNER/REPO --apply` (required: Python Linting, Unit Tests, Security Scan, Container Image Scan; `ci.yml` 의 PR `paths:` 필터는 제거됨).
