#!/usr/bin/env bash
set -euo pipefail

DB_URL="${AI_SCHEMA_DB_URL:-${OCI_DB_URL:-${POSTGRES_DB_URL:-${SUPABASE_DB_URL:-}}}}"
: "${DB_URL:?AI_SCHEMA_DB_URL, OCI_DB_URL, or POSTGRES_DB_URL is required}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AI_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

psql --set ON_ERROR_STOP=1 \
  --file "${AI_ROOT}/app/db/migrations/001_ai_runtime_cache.sql" \
  "${DB_URL}"

psql --set ON_ERROR_STOP=1 \
  --file "${AI_ROOT}/app/db/migrations/003_ai_ingest_orchestration.sql" \
  "${DB_URL}"

psql --set ON_ERROR_STOP=1 \
  --file "${AI_ROOT}/app/db/migrations/004_ai_ingest_checkpoints.sql" \
  "${DB_URL}"

case "${CHAT_SEMANTIC_CACHE_VECTOR_INDEX_ENABLED:-false}" in
  1|true|TRUE|yes|YES)
    psql --set ON_ERROR_STOP=1 \
      --file "${AI_ROOT}/app/db/migrations/002_chat_semantic_cache_vector_index.sql" \
      "${DB_URL}"
    ;;
esac

printf '%s\n' "AI runtime schema migration completed."
