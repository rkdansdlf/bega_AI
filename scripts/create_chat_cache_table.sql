-- 챗봇 응답 캐시 테이블
-- 실행: psql $SUPABASE_DB_URL -f scripts/create_chat_cache_table.sql

CREATE TABLE IF NOT EXISTS chat_response_cache (
    cache_key      VARCHAR(64)  PRIMARY KEY,
    question_text  TEXT         NOT NULL,
    filters_json   JSONB,
    intent         VARCHAR(50),
    response_text  TEXT         NOT NULL,
    model_name     VARCHAR(100),
    hit_count      INTEGER      NOT NULL DEFAULT 0,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at     TIMESTAMPTZ  NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_cache_expires_at ON chat_response_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_chat_cache_created_at ON chat_response_cache(created_at);

-- 참고: 만료 캐시 정리
-- DELETE FROM chat_response_cache WHERE expires_at <= now();

-- 캐시 통계 조회
-- SELECT intent, COUNT(*) AS cnt, AVG(hit_count)::numeric(10,2) AS avg_hits,
--        MIN(expires_at) AS earliest_expiry
-- FROM chat_response_cache WHERE expires_at > now()
-- GROUP BY intent ORDER BY cnt DESC;
