# ⚾ KBO AI Service

Vector-enabled RAG service that layers Google Gemini on top of the existing KBO database. The design follows `chatbot_system.md` with pgvector storage, SSE streaming chat, and a lightweight intent router.

## 📦 Project layout

```
AI/
├── app/
│   ├── main.py                 # FastAPI factory
│   ├── config.py               # Settings (Gemini, DB, embeddings)
│   ├── deps.py                 # Dependency helpers (lifespan, DB pool)
│   ├── core/                   # RAG building blocks
│   │   ├── chunking.py
│   │   ├── embeddings.py       # Gemini or local embeddings
│   │   ├── prompts.py
│   │   ├── rag.py              # retrieve → augment → generate
│   │   ├── retrieval.py        # pgvector + FTS
│   │   └── tools.py            # Direct SQL shortcuts
│   ├── ml/intent_router.py     # 규칙 + (선택) SVM
│   ├── routers/                # FastAPI routers
│   │   ├── chat.py             # /chat/completion & /chat/stream (SSE)
│   │   ├── search.py           # /search debugging endpoint
│   │   └── ingest.py           # /ingest single-document API
│   └── db/                     # pgvector schema helpers
│       ├── schema.sql
│       └── queries.sql
├── scripts/ingest_from_kbo.py  # Initial bulk ingestion example
├── chatbot.py                  # Legacy shim (now delegates to RAG pipeline)
├── requirements.txt
├── Dockerfile
└── .env.example
```

## 🔧 Environment variables

Copy `.env.example` and fill in real values:

```
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash
GEMINI_EMBED_MODEL=text-embedding-004
EMBED_PROVIDER=gemini           # or local
LLM_PROVIDER=gemini
SUPABASE_DB_URL=postgresql://user:pass@host:5432/db
DEFAULT_SEARCH_LIMIT=6
MAX_OUTPUT_TOKENS=1024
```

## 🚀 Getting started

```bash
cd AI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

Endpoints:

| Method | Path               | Description                  |
|--------|--------------------|------------------------------|
| POST   | `/chat/completion` | RAG pipeline JSON response   |
| POST   | `/chat/stream`     | SSE stream (EventSource)     |
| GET    | `/search/`         | Fetch top-k retrieved chunks |
| POST   | `/ingest/`         | Upsert a single document     |
| GET    | `/health`          | Basic health check           |

### Example chat request

```bash
curl -X POST http://localhost:8001/chat/completion \
  -H "Content-Type: application/json" \
  -d '{"question":"2024 LG OPS 상위 5명 알려줘"}'
```

### Streaming usage (SSE)

```javascript
const source = new EventSource("/chat/stream", {
  withCredentials: false,
});
source.addEventListener("message", (event) => {
  console.log(JSON.parse(event.data));
});
source.addEventListener("done", () => source.close());
```

## 📚 RAG flow

1. **Intent routing** – rule-based (stats/explanatory/freeform) with optional SVM model (`app/ml/intent_router.joblib`).
2. **Retrieval** – pgvector similarity + optional FTS boost (`app/core/retrieval.py`). Filters can include `season_year`, `team_id`, etc.
3. **Direct SQL tools** – some stat queries bypass LLM via SQL (`app/core/tools.py`).
4. **Generation** – OpenRouter chat completion with citations assembled from retrieved chunks.

## 🗄️ Vector schema

`app/db/schema.sql` creates the `rag_chunks` table with:

- metadata columns (`season_year`, `team_id`, `player_id`, …)
- `content_tsv` for hybrid text ranking
- `embedding` (`vector(1536)`) + IVFFLAT + GIN indices

Run the file once (psql) before ingesting:

```bash
psql $SUPABASE_DB_URL -f app/db/schema.sql
```

## 📝 Notes

- The legacy Gemini-based implementation is commented out; `chatbot.py` now wraps the new pipeline for backward compatibility.
- For large ingests use `scripts/ingest_from_kbo.py` as a template—it demonstrates how to serialize stats rows, chunk them, embed, and upsert into `rag_chunks`.
- Set `EMBED_PROVIDER=local` for offline development; embeddings fall back to deterministic pseudo vectors.
- Streaming is powered by `sse-starlette`; remember to handle heartbeat/timeout rules when deploying behind proxies.
