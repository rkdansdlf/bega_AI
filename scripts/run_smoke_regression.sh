#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPORTS_DIR="${REPORTS_DIR:-${ROOT_DIR}/reports}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
INTERNAL_API_KEY="${INTERNAL_API_KEY:-${AI_INTERNAL_TOKEN:-}}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -z "${INTERNAL_API_KEY}" ]]; then
  echo "INTERNAL_API_KEY or AI_INTERNAL_TOKEN is required." >&2
  exit 2
fi

mkdir -p "${REPORTS_DIR}"

SMOKE_COMMON_ARGS=(
  --base-url "${BASE_URL}"
  --internal-api-key "${INTERNAL_API_KEY}"
)

if [[ "${SMOKE_DISABLE_CACHE:-1}" == "1" ]]; then
  SMOKE_COMMON_ARGS+=(--disable-cache)
fi

ensure_baseline() {
  local preset="$1"
  local baseline_path="${REPORTS_DIR}/smoke_latency_baseline_${preset}.json"
  if [[ -f "${baseline_path}" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_smoke_latency_baseline.py" --preset "${preset}"
}

run_regulation_canary() {
  local report_path="${REPORTS_DIR}/smoke_chatbot_quality_regulations_20_${TIMESTAMP}.json"
  local summary_path="${REPORTS_DIR}/smoke_chatbot_quality_regulations_20_${TIMESTAMP}_summary.json"
  local eval_path="${REPORTS_DIR}/smoke_chatbot_quality_regulations_20_${TIMESTAMP}_eval.json"

  ensure_baseline "regulations_20"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_chatbot_quality.py" \
    "${SMOKE_COMMON_ARGS[@]}" \
    --chat-batch-size 20 \
    --chat-question-list "${SCRIPT_DIR}/smoke_chatbot_quality_regulations_20.txt" \
    --output "${report_path}" \
    --summary-output "${summary_path}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_smoke_candidate.py" \
    --baseline-preset regulations_20 \
    --candidate "${summary_path}" \
    --output "${eval_path}"

  echo "regulations_report=${report_path}"
  echo "regulations_summary=${summary_path}"
  echo "regulations_eval=${eval_path}"
}

run_regmix_full() {
  local report_path="${REPORTS_DIR}/smoke_chatbot_quality_regmix_100_${TIMESTAMP}.json"
  local summary_path="${REPORTS_DIR}/smoke_chatbot_quality_regmix_100_${TIMESTAMP}_summary.json"
  local eval_path="${REPORTS_DIR}/smoke_chatbot_quality_regmix_100_${TIMESTAMP}_eval.json"

  ensure_baseline "regmix_100"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_chatbot_quality.py" \
    "${SMOKE_COMMON_ARGS[@]}" \
    --chat-batch-size 100 \
    --output "${report_path}" \
    --summary-output "${summary_path}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_smoke_candidate.py" \
    --baseline-preset regmix_100 \
    --candidate "${summary_path}" \
    --output "${eval_path}"

  echo "regmix_report=${report_path}"
  echo "regmix_summary=${summary_path}"
  echo "regmix_eval=${eval_path}"
}

run_regulation_canary
run_regmix_full
