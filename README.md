
<div id="top"> <!-- HEADER STYLE: CLASSIC --> <div align="center">

# AI SERVICE

<em>BEGA 플랫폼의 지능형 기능</em>

<!-- BADGES --> <img src="https://img.shields.io/github/last-commit/737genie/ai-service?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit"> <img src="https://img.shields.io/github/languages/top/737genie/ai-service?style=flat&color=0080ff" alt="repo-top-language"> <img src="https://img.shields.io/github/languages/count/737genie/ai-service?style=flat&color=0080ff" alt="repo-language-count">

<em>사용된 기술 스택:</em>

<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python"> <img src="https://img.shields.io/badge/FastAPI-009688.svg?style=flat&logo=FastAPI&logoColor=white" alt="FastAPI"> <img src="https://img.shields.io/badge/OpenAI-412991.svg?style=flat&logo=OpenAI&logoColor=white" alt="OpenAI"> <img src="https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=Docker&logoColor=white" alt="Docker"> <img src="https://img.shields.io/badge/Amazon%20AWS-232F3E.svg?style=flat&logo=Amazon-AWS&logoColor=white" alt="AWS"> <img src="https://img.shields.io/badge/Nginx-009639.svg?style=flat&logo=NGINX&logoColor=white" alt="Nginx"> </div> <br>

----------

## 목차

-   [개요](https://claude.ai/chat/b41b67c5-88a7-4ded-a9a2-4bed83ecd0c6#%EA%B0%9C%EC%9A%94)
-   [주요 기능](https://claude.ai/chat/b41b67c5-88a7-4ded-a9a2-4bed83ecd0c6#%EC%A3%BC%EC%9A%94-%EA%B8%B0%EB%8A%A5)
-   [프로젝트 구조](https://claude.ai/chat/b41b67c5-88a7-4ded-a9a2-4bed83ecd0c6#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B5%AC%EC%A1%B0)
-   [아키텍처](https://claude.ai/chat/b41b67c5-88a7-4ded-a9a2-4bed83ecd0c6#%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98)
-   [시작하기](https://claude.ai/chat/b41b67c5-88a7-4ded-a9a2-4bed83ecd0c6#%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0)
    -   [사전 요구사항](https://claude.ai/chat/b41b67c5-88a7-4ded-a9a2-4bed83ecd0c6#%EC%82%AC%EC%A0%84-%EC%9A%94%EA%B5%AC%EC%82%AC%ED%95%AD)
    -   [설치](https://claude.ai/chat/b41b67c5-88a7-4ded-a9a2-4bed83ecd0c6#%EC%84%A4%EC%B9%98)
    -   [환경 설정](https://claude.ai/chat/b41b67c5-88a7-4ded-a9a2-4bed83ecd0c6#%ED%99%98%EA%B2%BD-%EC%84%A4%EC%A0%95)
    -   [실행](https://claude.ai/chat/b41b67c5-88a7-4ded-a9a2-4bed83ecd0c6#%EC%8B%A4%ED%96%89)
-   [API 문서](https://claude.ai/chat/b41b67c5-88a7-4ded-a9a2-4bed83ecd0c6#api-%EB%AC%B8%EC%84%9C)
-   [RAG 흐름](https://claude.ai/chat/b41b67c5-88a7-4ded-a9a2-4bed83ecd0c6#rag-%ED%9D%90%EB%A6%84)
-   [벡터 스키마 및 데이터 수집](https://claude.ai/chat/b41b67c5-88a7-4ded-a9a2-4bed83ecd0c6#%EB%B2%A1%ED%84%B0-%EC%8A%A4%ED%82%A4%EB%A7%88-%EB%B0%8F-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%88%98%EC%A7%91)
-   [배포](https://claude.ai/chat/b41b67c5-88a7-4ded-a9a2-4bed83ecd0c6#%EB%B0%B0%ED%8F%AC)

----------

## 개요

AI Service는 BEGA(Baseball Guide) 플랫폼의 지능형 기능을 지원하는 벡터 기반 RAG(검색 증강 생성) 서비스입니다. KBO 데이터베이스 위에 최신 LLM 제공자(OpenRouter 또는 Google Gemini)를 계층화하여, pgvector 저장소, SSE 스트리밍 채팅, 경량 의도 라우터를 통해 야구 정보를 제공합니다.

### 야구 데이터 운영 원칙

- 외부 야구 웹 조회나 크롤링은 사용하지 않습니다.
- 야구 데이터는 내부 DB와 정적 문서만 사용합니다.
- 경기 메타데이터가 부족하거나 정합성이 깨지면 `MANUAL_BASEBALL_DATA_REQUIRED` payload로 운영자 데이터를 요청합니다.

**왜 AI Service인가?**

이 마이크로서비스는 다음을 통해 사용자 경험을 향상시킵니다:

-   🎤 **음성-텍스트 변환:** OpenAI Whisper 기반 한국어 음성 인식
-   🤖 **RAG 기반 챗봇:** 검색 증강 생성으로 정확한 야구 정보 제공
-   🔍 **하이브리드 검색:** pgvector + FTS(Full-Text Search) 결합
-   📊 **세이버메트릭스:** ERA-, wRC+ 등 고급 야구 지표 계산
-   ⚡ **SSE 스트리밍:** 실시간 응답 스트리밍
-   🧠 **의도 라우팅:** 질문 의도에 따른 최적화된 답변 생성

----------

## 주요 기능

### RAG 시스템

-   **검색 증강 생성 (RAG)**
    
    -   HyDE(Hypothetical Document Embeddings) 기법 적용
    -   pgvector 벡터 유사도 검색
    -   FTS(Full-Text Search) 키워드 검색
    -   하이브리드 검색으로 정확도 향상
-   **의도 라우팅**
    
    -   규칙 기반 질문 의도 분류
    -   통계 조회, 순위 질문, 용어 설명 등 분류
    -   의도별 최적화된 검색 전략
-   **컨텍스트 생성**
    
    -   원본 데이터를 LLM 친화적 텍스트로 변환
    -   세이버메트릭스 지표 동적 계산 (ERA-, wRC+ 등)
    -   선수별 순위 및 비교 분석

### AI 기능

-   **음성-텍스트 변환**
    
    -   OpenAI Whisper API 통합
    -   한국어 음성 인식
    -   다양한 오디오 포맷 지원
-   **챗봇**
    
    -   OpenRouter / Google Gemini LLM 지원
    -   SSE 스트리밍 실시간 응답
    -   대화 기록 관리
    -   컨텍스트 인식 답변

### 데이터 서비스

-   **운영자 요청형 야구 데이터 처리**
    
    -   외부 웹 조회 비활성화
    -   내부 DB 기반 분석 유지
    -   부족 데이터는 운영자 제공 데이터로 보완
-   **벡터 데이터베이스**
    
    -   pgvector 확장 기능 활용
    -   다양한 임베딩 프로바이더 지원
    -   효율적인 청크 관리

----------

## 프로젝트 구조

```
AI/
├── app/
│   ├── main.py                 # FastAPI 팩토리
│   ├── config.py               # 설정 (LLM/임베딩, DB)
│   ├── deps.py                 # 의존성 헬퍼
│   ├── core/                   # RAG 구성 요소
│   │   ├── chunking.py         # 텍스트 분할
│   │   ├── embeddings.py       # 임베딩 생성
│   │   ├── kbo_metrics.py      # 세이버메트릭스 계산
│   │   ├── prompts.py          # 프롬프트 템플릿
│   │   ├── rag.py              # RAG 파이프라인
│   │   ├── retrieval.py        # 검색 로직
│   │   └── renderers/          # 데이터 렌더러
│   ├── ml/
│   │   └── intent_router.py    # 의도 분류기
│   ├── routers/                # API 라우터
│   │   ├── chat_stream.py      # 채팅 엔드포인트
│   │   ├── search.py           # 검색 엔드포인트
│   │   └── ingest.py           # 데이터 수집 API
│   ├── agents/
│   │   ├── baseball_agent.py
│   │   └── tool_caller.py
│   ├── db/                     # DB 스키마
│   │   ├── schema.sql
│   │   └── queries.sql
│   └── tools/
│       ├── database_query.py
│       ├── game_query.py
│       └── regulation_query.py
├── scripts/
│   └── ingest_from_kbo.py      # 대량 데이터 수집
├── docs/
│   ├── kbo_metrics_explained.md
│   └── kbo_regulations/
├── requirements.txt
├── Dockerfile
└── .env.example

```

----------

## 아키텍처

### 기술 스택

-   **프레임워크:** FastAPI
-   **언어:** Python 3.14
-   **AI/ML:** OpenAI API (Whisper), OpenRouter, Google Gemini
-   **임베딩:** OpenRouter, Gemini, OpenAI
-   **야구 데이터 정책:** 외부 웹 조회 비활성화, 내부 DB/정적 문서만 사용
-   **데이터베이스:** PostgreSQL + pgvector
-   **스트리밍:** sse-starlette
-   **웹 서버:** Nginx (리버스 프록시)
-   **배포:** AWS EC2
-   **컨테이너화:** Docker

### 시스템 설계

```
클라이언트 요청
    ↓ HTTPS
Nginx (EC2:443)
    ↓
FastAPI Application (8001)
    ↓
├── 의도 라우터 (Intent Router)
├── RAG 파이프라인
│   ├── HyDE 쿼리 생성
│   ├── 하이브리드 검색 (pgvector + FTS)
│   ├── 컨텍스트 생성 (Renderers)
│   └── LLM 답변 생성
├── OpenRouter / Gemini API
└── PostgreSQL + pgvector
    └── rag_chunks (벡터 저장소)

```

----------

## 시작하기

### 사전 요구사항

이 프로젝트는 다음 종속성을 필요로 합니다:

-   **Python:** 3.14
-   **패키지 매니저:** pip
-   **데이터베이스:** PostgreSQL 14+ (pgvector 확장 필요)
-   **컨테이너 런타임:** Docker (선택사항)

### 설치

소스에서 AI service를 빌드하고 종속성을 설치합니다:

1.  **저장소 클론:**
    
    ```sh
    ❯ git clone https://github.com/737genie/ai-service
    ```
    
2.  **프로젝트 디렉토리로 이동:**
    
    ```sh
    ❯ cd ai-service
    ```
    
3.  **가상 환경 생성 및 활성화:**
    
    ```sh
    ❯ python -m venv .venv
    ❯ source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```
    
4.  **종속성 설치:**
    
    ```sh
    ❯ pip install -r requirements.txt
    ❯ pip install -r requirements-dev  # 로컬 개발: black/flake8/pytest
    ```
    

### 환경 설정

1.  **로컬 `.env` 파일 생성:**
    
    `.env.example` 파일을 복사하여 실제 값으로 채워주세요:
    
    ```env
    # --- LLM/임베딩 프로바이더 선택 ---
    LLM_PROVIDER=openrouter          # openrouter(기본), gemini
    EMBED_PROVIDER=openai            # openai(기본), openrouter, gemini, hf, local
    
    # --- OpenRouter 설정 (기본) ---
    OPENROUTER_API_KEY=sk-or-...
    OPENROUTER_MODEL=openai/gpt-4o-mini
    OPENROUTER_EMBED_MODEL=openai/text-embedding-3-small
    OPENROUTER_REFERER=https://begabaseball.xyz
    OPENROUTER_APP_TITLE=BEGA 챗봇
    
    # --- Google Gemini 설정 (선택) ---
    # GEMINI_API_KEY=...
    # GEMINI_MODEL=gemini-1.5-flash
    # GEMINI_EMBED_MODEL=text-embedding-004
    
    # --- OpenAI 설정 (선택) ---
    # OPENAI_API_KEY=sk-...
    # OPENAI_EMBED_MODEL=text-embedding-3-small
    
    # --- 데이터베이스 ---
    POSTGRES_DB_URL=postgresql://user:pass@host:5432/db
    # Optional fallback for ops scripts
    # OCI_DB_URL=postgresql://user:pass@host:5432/db

    # --- Backend와 공유하는 내부 인증 / CORS ---
    AI_INTERNAL_TOKEN=local-dev-ai-internal-token
    # JSON 배열 또는 CSV 둘 다 허용
    CORS_ORIGINS=["http://localhost:5173","http://127.0.0.1:5173"]

    # --- 공통 설정 ---
    DEFAULT_SEARCH_LIMIT=15
    MAX_OUTPUT_TOKENS=1024
    ```

2.  **Docker/prod-style `.env.prod` 파일 생성:**

    repo 루트에서 `.env.prod.example`을 `.env.prod`로 복사하고 실제 값을 채워주세요.
    `.env.prod`는 로컬 운영자 파일이며 Git에 커밋하지 않습니다.
    
3.  **pgvector 확장 설치:**
    
    PostgreSQL에서 pgvector 확장을 활성화합니다:
    
    ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    ```
    

### 실행

프로젝트를 실행합니다:

**Python 사용:**

```sh
❯ PYTHONPATH=. .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

**Docker 사용:**

```sh
❯ docker build -t 737genie/ai-service .
❯ docker run -p 8001:8001 737genie/ai-service
```

API는 `http://localhost:8001`에서 사용 가능합니다

----------

## 챗봇 검증 자동화

PR 차단 게이트와 실서비스 스모크를 분리한 하이브리드 검증 전략을 사용합니다.

### 1) PR 게이트 (오프라인/모킹)

외부 LLM/OpenRouter 호출 없이 API 구조와 핵심 필드를 검증합니다.

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_chat_api_smoke.py \
  tests/test_coach.py \
  tests/test_coach_dual.py \
  tests/test_coach_router_year_resolution.py \
  tests/test_coach_cache_key.py \
  tests/test_coach_cache_policy.py \
  tests/test_rag_caching.py \
  -q
```

### 2) 실서비스 스모크 (수동 실행)

배포 직전 또는 운영 점검 시 실호출 스모크를 실행합니다.

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_chatbot.py \
  --base-url http://127.0.0.1:8001 \
  --season-year 2025 \
  --timeout 30 \
  --output logs/live_smoke_report.json
```

### Python 3.14 신기능 적용 정책

- 기본 런타임 정책은 **Python 3.14 단일 지원**입니다.
- Free-threaded(PEP 703), `concurrent.interpreters`, t-string(PEP 750), PEP 758 무괄호 `except` 문법은 운영 경로에 즉시 적용하지 않고 단계적으로 검증합니다.
- 코드 스타일은 기존 `except (TypeError, ValueError):` 형태를 유지합니다.

#### 1) 3.14 기능 감사 리포트 생성

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_py314_features.py \
  --format json \
  --output reports/python314_audit_local.json
```

Docker 런타임도 동일 스크립트로 별도 출력 파일에 저장하여 비교합니다.

#### 2) 실서비스 품질/지연시간 요약 리포트

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_chatbot_quality.py \
  --base-url http://127.0.0.1:8001 \
  --chat-batch-size 100 \
  --output reports/smoke_chatbot_quality_full.json \
  --summary-output reports/smoke_chatbot_quality_summary.json
```

`--summary-output`에는 endpoint 별 `p50/p95/p99`, `error_rate`, `timeout_rate`가 포함됩니다.

두 번 이상 실행한 summary 결과를 baseline으로 고정하려면:

```bash
PYTHONPATH=. .venv/bin/python scripts/build_smoke_latency_baseline.py \
  --inputs reports/smoke_chatbot_quality_summary_run1.json reports/smoke_chatbot_quality_summary_run2.json \
  --output reports/smoke_chatbot_quality_baseline.json
```

baseline 대비 후보 실행 결과를 자동 비교하려면:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_smoke_candidate.py \
  --baseline reports/smoke_chatbot_quality_baseline.json \
  --candidate reports/smoke_chatbot_quality_summary_candidate.json \
  --output reports/smoke_chatbot_quality_candidate_eval.json
```

기본 판정 기준:
- p95가 baseline 대비 5% 이상 악화되면 실패
- error_rate 증가가 0.5%p 초과면 실패
- timeout_rate 증가가 0.5%p 초과면 실패
- 메모리 기준(`--baseline-memory-mb`, `--candidate-memory-mb`)을 넣었을 때 10% 초과 증가면 실패

#### 3) ingest 전처리 병렬 엔진 실험

```bash
PYTHONPATH=. .venv/bin/python scripts/ingest_from_kbo.py \
  --tables player_season_batting player_season_pitching \
  --parallel-engine subinterp \
  --workers 4
```

- `--parallel-engine`: `thread|subinterp` (기본 `thread`)
- `subinterp` 실행 실패 시 자동으로 `thread`로 fallback 됩니다.

검증 대상:
- `GET /health`
- `POST /ai/chat/completion`
- `POST /ai/chat/stream`
- `POST /coach/analyze`

### 트러블슈팅

- `coach/analyze`는 `league_context.season_year`가 필요합니다.
- 실서비스 스모크 실패 시 환경변수(`OPENROUTER_API_KEY`, `LLM_PROVIDER`)를 먼저 확인합니다.
- 장애 진단 순서는 `health -> completion -> stream -> coach`를 권장합니다.

### Coach 캐시 운영 정책

- 런타임 `POST /coach/analyze`는 자동 재생성을 수행하지 않습니다.
- 캐시 상태가 `PENDING`이면 기존 작업 완료를 짧게 대기하며, 신선한 `PENDING`에는 중복 생성이 금지됩니다.
- 캐시 상태가 `FAILED`이면 런타임 재생성을 막고, 수동 배치로만 갱신합니다.
- `auto_brief` 모드에서 `question_signature`는 질문 문구와 무관하게 `auto`로 고정되고, `manual_detail`은 질문 문구 변경 시 캐시 키가 분기되는지 확인합니다.

운영 상수:

- `PENDING_STALE_SECONDS = 300`
- `PENDING_WAIT_TIMEOUT_SECONDS = 10`
- `PENDING_WAIT_POLL_MS = 300`

`/coach/analyze` 응답 메타에는 운영 해석용 필드가 포함됩니다.

- `cache_state`: `HIT`, `PENDING_WAIT`, `PENDING_STALE_TAKEOVER`, `FAILED_LOCKED`, `MISS_GENERATE`
- `in_progress`: 동일 요청 대기 중이면 `true`, 최종 응답이면 `false`
- `cache_key_version`: 현재 캐시 스키마 버전
- `resolved_focus`, `focus_signature`, `question_signature`: 디버깅/감시용

또한 응답 메타에는 다음 필드도 포함될 수 있습니다.

- `focus_section_missing`: 누락된 focus 섹션 존재 여부
- `missing_focus_sections`: 누락된 focus 목록

운영 로그 예시:

- `[Coach] Cache HIT ...`
- `[Coach] Cache PENDING_WAIT ...`
- `[Coach] Cache PENDING_STALE_TAKEOVER ...`
- `[Coach] Cache FAILED_LOCKED ...`

예: 자동 분석이 이미 진행 중일 때는 `cache_state=PENDING_WAIT`, `in_progress=true`로 오며
`cached`는 `false`입니다. 재요청이 금지된 `FAILED` 상태는 `cache_state=FAILED_LOCKED`로 반환됩니다.

수동 매치업 캐시 생성/갱신:

```bash
# 2025 정규시즌 매치업 캐시 생성(이미 존재하면 cache hit로 skip)
PYTHONPATH=. .venv/bin/python scripts/batch_coach_matchup_cache.py \
  --years 2025 \
  --league-type REGULAR

# 실행 결과 확인 예시:
# - 1회차: generated > 0 (신규 채움)
# - 2회차: generated == 0, skipped == cases
# - 실패 항목은 failed로 누적되고 reason 기준으로 재시도 대상 분류

# 대상 키 삭제 후 강제 재생성
PYTHONPATH=. .venv/bin/python scripts/batch_coach_matchup_cache.py \
  --years 2025 \
  --league-type REGULAR \
  --force-rebuild

# 수동/상세 모드(기존 스크립트 방식)으로 갱신
PYTHONPATH=. .venv/bin/python scripts/batch_coach_matchup_cache.py \
  --years 2025 \
  --league-type REGULAR \
  --mode manual_detail \
  --focus recent_form,bullpen
```

재시작 후 검증 포인트:
- 같은 매치업 재조회 시 서버 로그에 `cache_state=HIT` 또는 `cache_state=PENDING_WAIT`가 즉시 출력되는지 확인
- 로그 키워드 검증: `Cache HIT`, `Cache PENDING_WAIT`, `Cache FAILED_LOCKED`
- 동일 조건으로 `batch_coach_matchup_cache.py`를 연속 실행했을 때 2회차 `generated`가 0인지 확인
- 강제 재생성이 필요한 경우에만 `--force-rebuild`를 사용

권장 점검 예시:

- `cache_state` 값이 `PENDING_WAIT`이면서 `in_progress=true`인 응답은 정상 대기 응답
- `cache_state`가 `PENDING_WAIT`이면서 `in_progress=false`, `cached=true`라면 대기 종료 후 결과 반영 완료
- `cache_state=FAILED_LOCKED`는 런타임 자동 복구를 금지하고 수동 배치 필요
- 재기동 직후 대표 게임 1건 기준으로 `cached`가 즉시 `true`로 돌아오면 재생성 없는 상태로 판단

----------

### Chat 응답 캐시 운영/보안

- `POST /ai/chat/stream`는 `history=null` 이고 실시간 키워드(`오늘`, `지금`, `어제`)가 없을 때만 캐시를 사용합니다.
- 캐시 HIT 응답의 `event:meta`에는 `cached=true`, MISS 응답에는 `cached=false`가 포함됩니다.
- 캐시 TTL은 intent 기반이며 대표값은 `stats_lookup=6h`, `player_profile=48h`, `recent_form=3h`입니다.
- 만료 키로 재요청이 들어오면 `chat_response_cache`는 UPSERT로 갱신되고 `hit_count`는 `0`으로 리셋됩니다.

관리 API(내부 전용):

- `GET /ai/chat/cache/stats`
- `DELETE /ai/chat/cache?intent=...`
- `DELETE /ai/chat/cache/{cache_key}`

보호 정책:

- 기본값 `CHAT_CACHE_ADMIN_ENABLED=false`일 때 모든 관리 API는 `404`
- `CHAT_CACHE_ADMIN_ENABLED=true`인데 `CHAT_CACHE_ADMIN_TOKEN` 미설정이면 `503`
- 토큰 불일치 또는 미전달 시 `401`
- 정상 토큰은 `X-Cache-Admin-Token` 헤더로 전달

운영 체크 포인트:

- 로그 키워드: `[ChatCache] HIT`, `[ChatCache] SAVED`
- 관리 API 접근 로그의 `401/404` 비율 점검
- `503` 발생 시 설정 누락(`enabled=true` + token 미설정) 여부를 즉시 확인

----------

## API 문서

### 엔드포인트 목록


### 챗봇 엔드포인트 
``` 
POST /ai/chat/completion - RAG 파이프라인 JSON 응답 
POST /ai/chat/stream     - SSE 스트림 채팅 응답 
GET /ai/search           - 상위 K개 검색된 청크 조회 
POST /ai/ingest          - 단일 문서 업서트 
GET /ai/health           - 기본 헬스 체크 
``` 
**요청:** 
```
json { 
	"message": "사용자 질문", 
	"conversation_id": "optional_id", 
	"context": "optional_context" 
	} 
``` 
**응답:** 
```
json { 
	"response": "AI 응답", 
	"sources": ["출처1", "출처2"], 
	"conversation_id": "id" 
	} 
``` 
### 데이터 요청 정책
``` 
데이터 부족 시 MANUAL_BASEBALL_DATA_REQUIRED payload 반환
```

 ### 임베딩 
 ``` 
 embeddings.py 
 ```

### cURL 예시

```bash
# 채팅 완성 요청
curl -X POST http://localhost:8001/chat/completion \
  -H "Content-Type: application/json" \
  -d '{"question":"2024년 LG에서 OPS가 가장 높은 선수 5명 알려줘"}'

# 검색
curl "http://localhost:8001/search/?q=김도영&limit=5"

# 헬스 체크
curl http://localhost:8001/health

```

----------

## RAG 흐름

BEGA AI Service의 RAG(검색 증강 생성) 파이프라인은 다음 4단계로 구성됩니다:

### 1. 의도 라우팅 (Intent Routing)

`app/ml/intent_router.py`의 규칙 기반 분류기가 사용자 질문의 의도를 파악합니다:

-   단순 통계 조회
-   순위 질문
-   용어 설명
-   비교 분석

### 2. 검색 (Retrieval)

**HyDE(Hypothetical Document Embeddings) 기법:**

-   질문과 관련된 가상의 문서를 생성하여 검색 품질 향상

**하이브리드 검색:**

-   **벡터 검색**: pgvector를 사용한 의미론적 유사도 검색
-   **FTS 검색**: Full-Text Search를 통한 키워드 매칭
-   두 방식을 결합하여 정확도 최대화

검색 대상:

-   KBO 데이터베이스의 통계 데이터
-   `docs/kbo_metrics_explained.md` 등 정적 문서

### 3. 컨텍스트 생성 (Context Generation)

`app/core/renderers/`의 렌더러를 통해 검색된 데이터를 LLM 친화적 텍스트로 변환:

**예시:**

```python
# 원본 데이터
{"player": "김도영", "avg": 0.350, "hr": 20}

# 렌더링 후
"2024년 김도영은 0.350의 타율과 20개의 홈런을 기록했습니다."

```

**세이버메트릭스 계산:**

-   `app/core/kbo_metrics.py`를 사용하여 고급 지표 계산
-   ERA-, wRC+, OPS+ 등
-   선수별 순위 자동 산출

### 4. 답변 생성 (Generation)

최종 프롬프트 구성:

-   시스템 프롬프트 (`app/core/prompts.py`)
-   이전 대화 기록
-   생성된 컨텍스트

선택된 LLM(OpenRouter/Gemini)이 답변을 생성하고 `sse-starlette`를 통해 실시간 스트리밍

----------

## 벡터 스키마 및 데이터 수집

### 스키마 설정

`app/db/schema.sql` 파일을 사용하여 `rag_chunks` 테이블을 생성합니다:

```bash
psql $POSTGRES_DB_URL -f app/db/schema.sql
```

### 초기 데이터 수집

`scripts/ingest_from_kbo.py` 스크립트는 내부 DB와 정적 문서를 벡터 데이터베이스에 적재합니다:

```bash
# AI 디렉토리에서 실행
PYTHONPATH=. .venv/bin/python scripts/ingest_from_kbo.py \
  --tables player_season_batting player_season_pitching kbo_metrics_explained
```

**스크립트 기능:**

1.  통계 데이터를 LLM 친화적 텍스트로 변환
2.  텍스트를 적절한 크기의 청크로 분할
3.  각 청크의 임베딩 생성
4.  `rag_chunks` 테이블에 업서트

### 오프라인 개발

임베딩 API 호출 없이 개발하려면:

```env
EMBED_PROVIDER=local
```

로컬 모드에서는 결정론적인 의사(pseudo) 벡터를 사용합니다.

----------

## 배포

### AWS EC2 배포

1.  **서버 설정:**
    
    ```sh
    # 시스템 업데이트
    sudo apt update && sudo apt upgrade -y
    
    # Python 설치
    sudo apt install python3.14 python3.14-venv python3-pip -y
    
    # 외부 야구 크롤링은 사용하지 않으므로 추가 웹 드라이버는 필요하지 않습니다.
    ```
    
2.  **PostgreSQL pgvector 설정:**
    
    ```sh
    # PostgreSQL 설치 (이미 설치되어 있다면 생략)
    sudo apt install postgresql postgresql-contrib
    
    # pgvector 확장 설치
    sudo apt install postgresql-14-pgvector
    
    # PostgreSQL에서 확장 활성화
    sudo -u postgres psql -d your_database -c "CREATE EXTENSION IF NOT EXISTS vector;"
    ```
    
3.  **애플리케이션 설치:**
    
    ```sh
    cd /var/www/ai-service
    python3.14 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    
4.  **Nginx 설정:**
    
    ```nginx
    server {
        listen 443 ssl;
        server_name ai.begabaseball.xyz;
        
        ssl_certificate /etc/letsencrypt/live/ai.begabaseball.xyz/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/ai.begabaseball.xyz/privkey.pem;
        
        # SSE 스트리밍을 위한 설정
        location / {
            proxy_pass http://localhost:8001;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # SSE 타임아웃 설정
            proxy_read_timeout 3600s;
            proxy_connect_timeout 3600s;
            proxy_send_timeout 3600s;
            
            # SSE 버퍼링 비활성화
            proxy_buffering off;
            proxy_cache off;
        }
    }
    ```
    
5.  **systemd 서비스 설정:**
    
    ```ini
    # /etc/systemd/system/ai-service.service
    [Unit]
    Description=BEGA AI Service
    After=network.target postgresql.service
    
    [Service]
    Type=simple
    User=www-data
    WorkingDirectory=/var/www/ai-service
    Environment="PATH=/var/www/ai-service/.venv/bin"
    Environment="PYTHONPATH=/var/www/ai-service"
    ExecStart=/var/www/ai-service/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001
    Restart=always
    
    [Install]
    WantedBy=multi-user.target
    ```
    
6.  **서비스 시작:**
    
    ```sh
    sudo systemctl daemon-reload
    sudo systemctl enable ai-service
    sudo systemctl start ai-service
    sudo systemctl status ai-service
    ```
    

### Docker Compose 배포

```yaml
version: '3.8'
services:
  ai-service:
    build: .
    ports:
      - "8001:8001"
    environment:
      - LLM_PROVIDER=${LLM_PROVIDER}
      - EMBED_PROVIDER=${EMBED_PROVIDER}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - POSTGRES_DB_URL=${POSTGRES_DB_URL}
    volumes:
      - ./data:/app/data
      - ./docs:/app/docs
    restart: unless-stopped
    depends_on:
      - postgres
    
  postgres:
    image: pgvector/pgvector:pg14
    environment:
      - POSTGRES_DB=bega
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

**실행:**

```sh
docker-compose up -d
```

**Free-threaded 실험 프로필 실행(운영 기본 경로와 분리):**

```sh
docker compose \
  -f docker-compose.yml \
  -f docker-compose.free_threaded.yml \
  --profile free-threaded-experiment \
  up -d ai-chatbot
```

- 실험 이미지는 `bega_AI/Dockerfile.free_threaded`를 사용합니다.
- 성능/안정성 기준 미달 시 실험 프로필은 즉시 중단하고 기본 이미지(`bega_AI/Dockerfile`)로 유지합니다.

### 성능 최적화

-   **캐싱:** 빈번한 쿼리 결과를 캐싱
-   **비동기 처리:** FastAPI의 비동기 기능 활용
-   **배치 임베딩:** 여러 청크를 한 번에 임베딩 생성
-   **연결 풀링:** PostgreSQL 연결 풀 최적화
-   **모니터링:**
    -   로깅: Python logging 모듈

### 프로덕션 체크리스트

-   [ ] 환경 변수 보안 설정
-   [ ] SSL 인증서 설치 (Let's Encrypt)
-   [ ] pgvector 인덱스 최적화
-   [ ] API 레이트 리미팅 설정
-   [ ] 로그 로테이션 설정
-   [ ] 백업 전략 수립
-   [ ] 모니터링 알림 설정
-   [ ] SSE 타임아웃 튜닝
-   [x] Coach `auto_brief` 캐시 키(`question_signature="auto"`) 충돌 여부 점검
-   [x] Coach `auto` 모드 `COACH_BRIEF_MAX_OUTPUT_TOKENS` 적용 여부 점검
-   [x] 예측 페이지 요청 경로 전환 확인: `/api/predictions/my-vote/*` 미사용, `/api/predictions/my-votes` 사용

### 릴리즈 노트 (운영 반영)

-   **예측 페이지 AI 최적화(코치 연동)**
    -   핵심 경기/포스트시즌 경기만 `request_mode: auto_brief`로 자동 브리핑을 요청하고, 비핵심 경기에서는 AI 분석은 시작 시 자동 호출하지 않고 `AI 분석 요청` 버튼으로 진입형 요청으로 전환.
    -   자동/수동 브리핑 모두 `cache_state`, `request_mode`, `focus_signature`, `question_signature`를 메타로 노출해 캐시 관측을 고정.
    -   legacy 단건 투표 조회 API(`/api/predictions/my-vote/*`) 사용을 중단하고, 단건 경로 재출현을 운영 알람 대상으로 모니터링.
    -   예측 페이지는 단건 조회(`/api/predictions/my-vote/*`)를 더 이상 호출하지 않고 `POST /api/predictions/my-votes`로 교체.

### 참고 사항

-   **대량 데이터 수집**: `scripts/ingest_from_kbo.py`를 템플릿으로 사용
-   **오프라인 개발**: `EMBED_PROVIDER=local`로 설정하여 API 호출 없이 개발 가능
-   **SSE 프록시**: Nginx 설정에서 버퍼링과 타임아웃 규칙 주의
-   **임베딩 프로바이더**: 용도에 맞게 OpenRouter, Gemini, OpenAI 중 선택

----------

<div align="left"><a href="#top">⬆ 돌아가기</a></div>

----------
