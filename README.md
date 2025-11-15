# KBO AI 서비스

기존 KBO 데이터베이스 위에 최신 LLM 제공자(OpenRouter 또는 Google Gemini)를 계층화한 벡터 기반 RAG(검색 증강 생성) 서비스입니다. 이 서비스는 `chatbot_system.md`에 기술된 설계를 따르며, pgvector 저장소, SSE 스트리밍 채팅, 그리고 경량의 의도 라우터를 특징으로 합니다.

## 프로젝트 구조

```
AI/
├── app/
│   ├── main.py                 # FastAPI 팩토리
│   ├── config.py               # 설정 (LLM/임베딩, DB)
│   ├── deps.py                 # 의존성 헬퍼 (생명주기, DB 풀)
│   ├── core/                   # RAG 구성 요소
│   │   ├── chunking.py         # 텍스트 분할
│   │   ├── embeddings.py       # Gemini / OpenRouter / OpenAI / HF / 로컬 임베딩
│   │   ├── kbo_metrics.py      # 세이버메트릭스 지표 계산
│   │   ├── prompts.py          # 프롬프트 템플릿
│   │   ├── rag.py              # 검색 → 증강 → 생성 파이프라인
│   │   ├── retrieval.py        # pgvector + FTS(Full-Text Search)
│   │   └── renderers/          # 원본 데이터를 LLM 친화적 텍스트로 변환
│   ├── ml/
│   │   └── intent_router.py    # 규칙 기반 의도 분류기
│   ├── routers/                # FastAPI 라우터
│   │   ├── chat_stream.py      # /chat/completion, /chat/stream, /chat/voice (SSE)
│   │   ├── search.py           # /search 디버깅용 엔드포인트
│   │   └── ingest.py           # /ingest 단일 문서 처리 API
│   ├── agents/
│   │   ├── baseball_agent.py
│   │   └── tool_caller.py
│   ├── db/                     # pgvector 스키마 헬퍼
│   │   ├── schema.sql
│   │   └── queries.sql
│   └── tools/
│       ├── database_query.py
│       ├── game_query.py
│       └── regulation_query.py
├── scripts/
│   └── ingest_from_kbo.py      # 초기 대량 데이터 수집 예제
├── docs/
│   ├── kbo_metrics_explained.md    # RAG용 야구 지표 설명 문서
│   └── kbo_regulations/            # KBO 규정집 문서 모음
├── requirements.txt
├── Dockerfile
└── .env.example
```

## 환경 변수

`.env.example` 파일을 복사하여 실제 값으로 채워주세요:

```
# --- LLM/임베딩 프로바이더 선택 ---
# LLM_PROVIDER: openrouter (기본), gemini
# EMBED_PROVIDER: openrouter (기본), gemini, openai, hf, local
LLM_PROVIDER=openrouter
EMBED_PROVIDER=openrouter

# --- OpenRouter 설정 (기본) ---
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini
# OPENROUTER_EMBED_MODEL=openai/text-embedding-3-small 또는 다른 모델
# OPENROUTER_REFERER=https://your.domain
# OPENROUTER_APP_TITLE=KBO 챗봇

# --- Google Gemini 설정 (선택) ---
# GEMINI_API_KEY=...
# GEMINI_MODEL=gemini-1.5-flash
# GEMINI_EMBED_MODEL=text-embedding-004

# --- OpenAI 설정 (선택) ---
# OPENAI_API_KEY=sk-...
# OPENAI_EMBED_MODEL=text-embedding-3-small

# --- 공통 ---
SUPABASE_DB_URL=postgresql://user:pass@host:5432/db
DEFAULT_SEARCH_LIMIT=15
MAX_OUTPUT_TOKENS=1024
```

## 시작하기

```bash
cd AI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API 실행

```bash
# AI 디렉토리에서 실행
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

엔드포인트:

| 메소드 | 경로               | 설명                              |
|--------|--------------------|-------------------------------|
| POST   | `/chat/completion` | RAG 파이프라인 JSON 응답          |
| POST   | `/chat/stream`     | SSE 스트림 (EventSource)        |
| POST   | `/chat/voice`      | 음성 파일을 텍스트로 변환 (Whisper)  |
| GET    | `/search/`         | 상위 K개 검색된 청크 조회           |
| POST   | `/ingest/`         | 단일 문서 업서트(Upsert)          |
| GET    | `/health`          | 기본 헬스 체크                    |

### 채팅 요청 예시

```bash
curl -X POST http://localhost:8001/chat/completion \
  -H "Content-Type: application/json" \
  -d '''{"question":"2024년 LG에서 OPS가 가장 높은 선수 5명 알려줘"}'''
```

### 스트리밍 사용법 (SSE)

```javascript
const source = new EventSource("/chat/stream", {
  withCredentials: false,
});
source.addEventListener("message", (event) => {
  console.log(JSON.parse(event.data));
});
source.addEventListener("done", () => source.close());
```

## RAG 흐름

1.  **의도 라우팅 (Intent Routing)**: `app/ml/intent_router.py`의 규칙 기반 분류기가 사용자의 질문 의도(단순 통계 조회, 순위 질문, 용어 설명 등)를 파악합니다.

2.  **검색 (Retrieval)**: `app/core/rag.py`는 HyDE(Hypothetical Document Embeddings) 기법을 사용해 질문과 가장 관련성 높은 가상의 문서를 생성하여 검색 쿼리를 향상시킵니다. 그 후 `app/core/retrieval.py`가 pgvector의 벡터 유사도 검색과 키워드 기반의 FTS(Full-Text Search)를 결합한 하이브리드 검색으로 `rag_chunks` DB에서 관련 정보를 찾습니다. 이 DB에는 KBO 데이터베이스의 원본 테이블과 `docs/kbo_metrics_explained.md` 같은 정적 문서에서 수집된 정보가 모두 포함됩니다.

3.  **컨텍스트 생성 (Context Generation)**: 검색된 데이터 조각들은 `app/core/renderers/` 내의 전용 렌더러를 통해 LLM이 이해하기 쉬운 자연어 텍스트로 변환됩니다. 예를 들어, `player_season_batting` 테이블의 한 행은 `render_batting_season` 함수를 통해 "2024년 김도영은 0.350의 타율과 20개의 홈런을 기록했습니다."와 같은 문장으로 재구성됩니다. 이 과정에서 `app/core/kbo_metrics.py`를 사용하여 ERA-, wRC+ 같은 고급 통계 지표를 동적으로 계산하고, 선수별 순위를 매겨 컨텍스트를 더욱 풍부하게 만듭니다.

4.  **답변 생성 (Generation)**: `app/core/prompts.py`의 시스템 프롬프트, 이전 대화 기록, 그리고 위 단계에서 생성된 풍부한 컨텍스트를 조합하여 최종 프롬프트를 만듭니다. 이 프롬프트를 기반으로 선택된 LLM(OpenRouter/Gemini)이 자연스러운 답변을 생성하고, `sse-starlette`을 통해 사용자에게 실시간으로 스트리밍합니다.

## 벡터 스키마 및 데이터 수집

`app/db/schema.sql`은 `rag_chunks` 테이블을 생성합니다. 데이터를 수집하기 전에 psql을 사용하여 스키마를 먼저 설정해야 합니다:

```bash
psql $SUPABASE_DB_URL -f app/db/schema.sql
```

초기 데이터 수집 및 업데이트는 `scripts/ingest_from_kbo.py` 스크립트를 사용합니다. 스크립트를 실행할 때는 `AI` 디렉토리 루트에서 `PYTHONPATH`를 설정해야 합니다.

```bash
# AI 디렉토리에서 실행
PYTHONPATH=. .venv/bin/python scripts/ingest_from_kbo.py --tables player_season_batting player_season_pitching kbo_metrics_explained
```

## 참고

-   대량 데이터 수집 시에는 `scripts/ingest_from_kbo.py`를 템플릿으로 사용하세요. 이 스크립트는 통계 데이터를 LLM 친화적인 텍스트로 변환하고, 청크로 나누고, 임베딩하여 `rag_chunks`에 업서트하는 방법을 보여줍니다.
-   오프라인 개발 시에는 `.env` 파일에서 `EMBED_PROVIDER=local`로 설정하세요. 임베딩은 결정론적인 의사(pseudo) 벡터로 대체됩니다.
-   스트리밍은 `sse-starlette`에 의해 구동됩니다. 프록시 뒤에 배포할 때는 하트비트/타임아웃 규칙을 처리해야 합니다.

