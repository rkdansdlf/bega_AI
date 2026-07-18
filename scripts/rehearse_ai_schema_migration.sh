#!/usr/bin/env bash
set -euo pipefail

: "${POSTGRES_DB_URL:?POSTGRES_DB_URL is required}"
: "${AI_SCHEMA_REHEARSAL_DB_NAME:?AI_SCHEMA_REHEARSAL_DB_NAME is required}"
: "${AI_SCHEMA_REHEARSAL_CONFIRM:?AI_SCHEMA_REHEARSAL_CONFIRM is required}"

if [[ "${AI_SCHEMA_REHEARSAL_CONFIRM}" != "I_UNDERSTAND_TEMP_DB" ]]; then
  echo "Refusing to run: set AI_SCHEMA_REHEARSAL_CONFIRM=I_UNDERSTAND_TEMP_DB for a disposable database." >&2
  exit 2
fi

current_db="$(psql --set ON_ERROR_STOP=1 --tuples-only --no-align \
  "${POSTGRES_DB_URL}" \
  --command='SELECT current_database()')"
current_db="${current_db//[[:space:]]/}"
if [[ "${current_db}" != "${AI_SCHEMA_REHEARSAL_DB_NAME}" ]]; then
  echo "Refusing to run: connected database '${current_db}' does not match '${AI_SCHEMA_REHEARSAL_DB_NAME}'." >&2
  exit 2
fi

case "${current_db}" in
  *rehearsal*|*sandbox*|*tmp*|*test*) ;;
  *)
    echo "Refusing to run: database name must identify a disposable rehearsal database." >&2
    exit 2
    ;;
esac

# Rehearsal always targets the explicitly checked disposable DB, even when the
# operator shell also contains production OCI_DB_URL settings.
export AI_SCHEMA_DB_URL="${POSTGRES_DB_URL}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/migrate_ai_runtime_schema.sh"

AI_PYTHON="${AI_PYTHON:-}"
if [[ -z "${AI_PYTHON}" && -x "${SCRIPT_DIR}/../.venv/bin/python" ]]; then
  AI_PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
fi
AI_PYTHON="${AI_PYTHON:-python3}"
AI_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
(
  cd -- "${AI_ROOT}"
  "${AI_PYTHON}" -m scripts.validate_ai_runtime_schema
)

printf '%s\n' "AI schema migration rehearsal OK: database=${current_db}"
