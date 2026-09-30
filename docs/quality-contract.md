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
| Fingerprint | prompt version/hash, planner/retrieval/reranker version, model, embedding signature 를 meta·캐시·eval 에 기록 | `app/core/fingerprint.py` |
| Golden 평가 | retrieval(Recall@5/10, MRR, nDCG, wrong-source, zero-hit) 과 generation(unsupported/numeric/entity hallucination, citation P/R) 을 분리 측정, baseline 대비 회귀 시 exit 1 | `scripts/eval_rag_golden.py`, `evals/` |
| CI 게이트 | `ai-pr-gate` 가 golden 평가를 실행 | `.github/workflows/ai-pr-gate.yml` |
| Ingest→캐시 무효화 | ingest 성공 시 영향 범위(season / 시즌 미지정 시 전체 non-stable) 캐시 삭제 | `app/core/cache_invalidation.py` |
| Semantic cache 동등성 | season/team scope, answer scope, answerability, source, freshness 불일치를 shadow 평가에서 실패 처리 | `app/core/semantic_cache_equivalence.py` |
| Embedding generation | 레지스트리 + 커버리지 감사 후 원자적 활성화/롤백 + 검색 시 활성 signature 필터 | `app/core/embedding_generations.py`, migration `007` |
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

- **물리적 blue/green 미완**: `rag_chunks` 는 `(source_table, source_row_id)` 유니크 + 단일
  `vector` 컬럼이라 같은 테이블에 구/신 임베딩이 공존할 수 없다. 여기서는 포인터 전환·감사·혼합
  방지 필터까지만 제공하고, 구 generation 행을 보존하는 per-generation 저장은 크롤러(임베딩 소유자)
  과제다. 롤백은 구 signature 행이 남아 있을 때만 허용된다.
- Oracle 읽기 경로(`oracle_rag.py`)에는 generation 게이트·relevance 컬럼 select 가 아직 없다.
- 스트리밍 accounting 의 provider 라벨은 `LLM_PROVIDER` 기준이라 fallback 시 부정확할 수 있다.
- Chunk 벤치마크는 `EMBED_PROVIDER=local` 이면 의미 없음(리포트에 `meaningful:false`).
- Feedback: `POST /ai/chat/feedback`(내부 토큰) → `rag_answer_feedback`(migration 008). BFF 가 호출하도록 연동해야 데이터가 쌓이며, `mine_retrieval_events.py --with-feedback` 가 DOWN 평가를 골든 후보로 합친다.
- `main` branch protection: `scripts/ops/apply_branch_protection.sh OWNER/REPO --apply` (required: Python Linting, Unit Tests, Security Scan, Container Image Scan; `ci.yml` 의 PR `paths:` 필터는 제거됨).
