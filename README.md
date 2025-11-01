# ⚾ KBO AI 서비스

기존 KBO 데이터베이스 위에 최신 LLM 제공자(OpenRouter 또는 Google Gemini)를 계층화한 벡터 기반 RAG(검색 증강 생성) 서비스입니다. 이 서비스는 `chatbot_system.md`에 기술된 설계를 따르며, pgvector 저장소, SSE 스트리밍 채팅, 그리고 경량의 의도 라우터를 특징으로 합니다.

## 📦 프로젝트 구조

```
AI/
├── app/
│   ├── main.py                 # FastAPI 팩토리
│   ├── config.py               # 설정 (LLM/임베딩, DB)
│   ├── deps.py                 # 의존성 헬퍼 (생명주기, DB 풀)
│   ├── core/                   # RAG 구성 요소
│   │   ├── chunking.py         # 텍스트 분할
│   │   ├── embeddings.py       # Gemini / OpenRouter / HF / 로컬 임베딩
│   │   ├── prompts.py          # 프롬프트 템플릿
│   │   ├── rag.py              # 검색 → 증강 → 생성 파이프라인
│   │   ├── retrieval.py        # pgvector + FTS(Full-Text Search)
│   │   └── tools.py            # 직접 SQL을 사용하는 단축 경로
│   ├── ml/intent_router.py     # 규칙 기반 + (선택적) SVM 의도 분류기
│   ├── routers/                # FastAPI 라우터
│   │   ├── chat.py             # /chat/completion & /chat/stream (SSE)
│   │   ├── search.py           # /search 디버깅용 엔드포인트
│   │   └── ingest.py           # /ingest 단일 문서 처리 API
│   └── db/                     # pgvector 스키마 헬퍼
│       ├── schema.sql
│       └── queries.sql
├── scripts/ingest_from_kbo.py  # 초기 대량 데이터 수집 예제
├── chatbot.py                  # 레거시 호환용 래퍼 (RAG 파이프라인 호출)
├── requirements.txt
├── Dockerfile
└── .env.example
```

## 🔧 환경 변수

`.env.example` 파일을 복사하여 실제 값으로 채워주세요:

```
# OpenRouter (기본 예시)
LLM_PROVIDER=openrouter
EMBED_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o-mini
EMBED_MODEL=openai/text-embedding-3-small
# OPENROUTER_EMBED_MODEL=openai/text-embedding-3-small
# OPENROUTER_REFERER=https://your.domain
# OPENROUTER_APP_TITLE=KBO 챗봇

# Gemini (선택 사항)
# LLM_PROVIDER=gemini
# EMBED_PROVIDER=gemini
# GEMINI_API_KEY=...
# GEMINI_MODEL=gemini-1.5-flash
# GEMINI_EMBED_MODEL=text-embedding-004

# 공통
SUPABASE_DB_URL=postgresql://user:pass@host:5432/db
DEFAULT_SEARCH_LIMIT=6
MAX_OUTPUT_TOKENS=1024
```

## 🚀 시작하기

```bash
cd AI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

엔드포인트:

| 메소드 | 경로               | 설명                         |
|--------|--------------------|------------------------------|
| POST   | `/chat/completion` | RAG 파이프라인 JSON 응답     |
| POST   | `/chat/stream`     | SSE 스트림 (EventSource)     |
| GET    | `/search/`         | 상위 K개 검색된 청크 조회    |
| POST   | `/ingest/`         | 단일 문서 업서트(Upsert)     |
| GET    | `/health`          | 기본 헬스 체크               |

### 채팅 요청 예시

```bash
curl -X POST http://localhost:8001/chat/completion \
  -H "Content-Type: application/json" \
  -d '''{"question":"2024 LG OPS 상위 5명 알려줘"}'''
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

## 📚 RAG 흐름

1.  **의도 라우팅** – 규칙 기반(통계/설명/자유 형식)이며, 선택적으로 SVM 모델(`app/ml/intent_router.joblib`)을 사용합니다.
2.  **검색** – pgvector 유사도 검색에 선택적으로 FTS(Full-Text Search) 부스팅을 사용합니다 (`app/core/retrieval.py`). `season_year`, `team_id` 등의 필터를 포함할 수 있습니다.
3.  **직접 SQL 도구** – 일부 통계 쿼리는 LLM을 거치지 않고 SQL을 통해 직접 처리됩니다 (`app/core/tools.py`).
4.  **생성** – 선택한 LLM(OpenRouter/Gemini)으로 답변을 생성하고, 검색된 청크의 출처(citation)를 함께 제공합니다.

## 🗄️ 벡터 스키마

`app/db/schema.sql`은 `rag_chunks` 테이블을 생성하며, 이 테이블은 다음을 포함합니다:

-   메타데이터 컬럼 (`season_year`, `team_id`, `player_id`, …)
-   하이브리드 텍스트 랭킹을 위한 `content_tsv`
-   `embedding` (`vector(1536)`) + IVFFLAT + GIN 인덱스

데이터를 수집하기 전에 psql을 사용하여 파일을 한 번 실행해주세요:

```bash
psql $SUPABASE_DB_URL -f app/db/schema.sql
```

## 📝 참고

-   `chatbot.py`는 기존 인터페이스와의 호환성을 위해 RAG 파이프라인을 감싸는 래퍼입니다.
-   대량 데이터 수집 시에는 `scripts/ingest_from_kbo.py`를 템플릿으로 사용하세요. 이 스크립트는 통계 데이터를 직렬화하고, 청크로 나누고, 임베딩하여 `rag_chunks`에 업서트하는 방법을 보여줍니다.
-   오프라인 개발 시에는 `EMBED_PROVIDER=local`로 설정하세요. 임베딩은 결정론적인 의사(pseudo) 벡터로 대체됩니다.
-   스트리밍은 `sse-starlette`에 의해 구동됩니다. 프록시 뒤에 배포할 때는 하트비트/타임아웃 규칙을 처리해야 합니다.