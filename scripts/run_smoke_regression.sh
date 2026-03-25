#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPORTS_DIR="${REPORTS_DIR:-${ROOT_DIR}/reports}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
INTERNAL_API_KEY="${INTERNAL_API_KEY:-${AI_INTERNAL_TOKEN:-}}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
WARMUP_DIR="${SMOKE_WARMUP_DIR:-/tmp/smoke_chatbot_quality_warmup}"

if [[ -z "${INTERNAL_API_KEY}" ]]; then
  echo "INTERNAL_API_KEY or AI_INTERNAL_TOKEN is required." >&2
  exit 2
fi

mkdir -p "${REPORTS_DIR}"
mkdir -p "${WARMUP_DIR}"

SMOKE_COMMON_ARGS=(
  --base-url "${BASE_URL}"
  --internal-api-key "${INTERNAL_API_KEY}"
)

if [[ "${SMOKE_DISABLE_CACHE:-0}" == "1" ]]; then
  SMOKE_COMMON_ARGS+=(--disable-cache)
fi

warm_cache_measurement() {
  local preset="$1"
  local batch_size="$2"
  local question_list="${3:-}"

  if [[ "${SMOKE_DISABLE_CACHE:-0}" == "1" || "${SMOKE_WARM_CACHE:-1}" != "1" ]]; then
    return 0
  fi

  local warm_report="${WARMUP_DIR}/${preset}_${TIMESTAMP}_warm.json"
  local warm_summary="${WARMUP_DIR}/${preset}_${TIMESTAMP}_warm_summary.json"

  echo "warming cache for ${preset}"
  if [[ -n "${question_list}" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_chatbot_quality.py" \
      "${SMOKE_COMMON_ARGS[@]}" \
      --chat-batch-size "${batch_size}" \
      --chat-question-list "${question_list}" \
      --output "${warm_report}" \
      --summary-output "${warm_summary}" \
      >/dev/null
  else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_chatbot_quality.py" \
      "${SMOKE_COMMON_ARGS[@]}" \
      --chat-batch-size "${batch_size}" \
      --output "${warm_report}" \
      --summary-output "${warm_summary}" \
      >/dev/null
  fi
}

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
  warm_cache_measurement "regulations_20" 20 "${SCRIPT_DIR}/smoke_chatbot_quality_regulations_20.txt"

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

run_llm_canary() {
  local report_path="${REPORTS_DIR}/smoke_chatbot_quality_llm_canary_20_${TIMESTAMP}.json"
  local summary_path="${REPORTS_DIR}/smoke_chatbot_quality_llm_canary_20_${TIMESTAMP}_summary.json"
  local eval_path="${REPORTS_DIR}/smoke_chatbot_quality_llm_canary_20_${TIMESTAMP}_eval.json"

  ensure_baseline "llm_canary_20"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_chatbot_quality.py" \
    "${SMOKE_COMMON_ARGS[@]}" \
    --disable-cache \
    --chat-batch-size 20 \
    --chat-question-list "${SCRIPT_DIR}/smoke_chatbot_quality_llm_canary_20.txt" \
    --output "${report_path}" \
    --summary-output "${summary_path}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_smoke_candidate.py" \
    --baseline-preset llm_canary_20 \
    --candidate "${summary_path}" \
    --output "${eval_path}"

  echo "llm_canary_report=${report_path}"
  echo "llm_canary_summary=${summary_path}"
  echo "llm_canary_eval=${eval_path}"
}

run_regmix_full() {
  local report_path="${REPORTS_DIR}/smoke_chatbot_quality_regmix_100_${TIMESTAMP}.json"
  local summary_path="${REPORTS_DIR}/smoke_chatbot_quality_regmix_100_${TIMESTAMP}_summary.json"
  local eval_path="${REPORTS_DIR}/smoke_chatbot_quality_regmix_100_${TIMESTAMP}_eval.json"

  ensure_baseline "regmix_100"
  warm_cache_measurement "regmix_100" 100

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
run_llm_canary
run_regmix_full
