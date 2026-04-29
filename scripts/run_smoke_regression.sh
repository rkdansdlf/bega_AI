#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPORTS_DIR="${REPORTS_DIR:-${ROOT_DIR}/reports}"
export REPORTS_DIR
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
INTERNAL_API_KEY="${INTERNAL_API_KEY:-${AI_INTERNAL_TOKEN:-}}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
WARMUP_DIR="${SMOKE_WARMUP_DIR:-/tmp/smoke_chatbot_quality_warmup}"
COMPOSE_ENV_FILE="${COMPOSE_ENV_FILE:-${ROOT_DIR}/.env.prod}"
MEMORY_SAMPLE_INTERVAL_MS="${MEMORY_SAMPLE_INTERVAL_MS:-1000}"
QA_SUMMARY_PATH="${REPORTS_DIR}/smoke_chatbot_quality_${TIMESTAMP}_qa.md"
SMOKE_ALLOW_EXTRA_LOCAL_DEV_PORTS="${SMOKE_ALLOW_EXTRA_LOCAL_DEV_PORTS:-0}"
SMOKE_ALLOW_DB_CLIENT_PRESSURE="${SMOKE_ALLOW_DB_CLIENT_PRESSURE:-0}"

COMPOSE_CMD=(docker compose)
if [[ -f "${COMPOSE_ENV_FILE}" ]]; then
  COMPOSE_CMD+=(--env-file "${COMPOSE_ENV_FILE}")
fi

if [[ -z "${INTERNAL_API_KEY}" ]]; then
  echo "INTERNAL_API_KEY or AI_INTERNAL_TOKEN is required." >&2
  exit 2
fi

ensure_ai_compose_service() {
  local raw_ids
  raw_ids="$("${COMPOSE_CMD[@]}" ps -q ai-chatbot 2>/dev/null || true)"
  local -a container_ids=()
  while IFS= read -r line; do
    if [[ -n "${line}" ]]; then
      container_ids+=("${line}")
    fi
  done <<< "${raw_ids}"

  if [[ "${#container_ids[@]}" -eq 0 ]]; then
    echo "ai-chatbot docker compose service is not running." >&2
    echo "Start it first: docker compose --env-file .env.prod up -d ai-chatbot" >&2
    exit 2
  fi
  if [[ "${#container_ids[@]}" -ne 1 ]]; then
    echo "ai-chatbot docker compose service resolved to multiple containers: ${container_ids[*]}" >&2
    exit 2
  fi
}

ensure_no_conflicting_local_dev_ports() {
  if [[ "${SMOKE_ALLOW_EXTRA_LOCAL_DEV_PORTS}" == "1" ]]; then
    return 0
  fi

  local -a blocked_ports=(8080 18080)
  local port
  for port in "${blocked_ports[@]}"; do
    if lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "conflicting local dev listener detected on port ${port}." >&2
      echo "Stop the extra local server or rerun with SMOKE_ALLOW_EXTRA_LOCAL_DEV_PORTS=1 to bypass this preflight." >&2
      exit 2
    fi
  done
}

ensure_local_db_client_pressure_is_safe() {
  if [[ "${SMOKE_ALLOW_DB_CLIENT_PRESSURE}" == "1" ]]; then
    return 0
  fi

  local db_snapshot
  db_snapshot="$(lsof -nP -iTCP:5432 2>/dev/null || true)"
  if [[ -z "${db_snapshot}" ]]; then
    return 0
  fi

  local process_count
  local connection_count
  process_count="$(printf '%s\n' "${db_snapshot}" | awk 'NR>1 {print $2}' | sort -u | wc -l | tr -d ' ')"
  connection_count="$(printf '%s\n' "${db_snapshot}" | awk 'NR>1 {count++} END {print count+0}')"

  if [[ "${process_count}" -gt 2 || "${connection_count}" -gt 8 ]]; then
    echo "local DB client pressure is too high before smoke regression (processes=${process_count}, connections=${connection_count})." >&2
    echo "Stop extra local backend/AI dev processes or rerun with SMOKE_ALLOW_DB_CLIENT_PRESSURE=1 to bypass this preflight." >&2
    printf '%s\n' "${db_snapshot}" >&2
    exit 2
  fi
}

ensure_ai_compose_service
ensure_no_conflicting_local_dev_ports
ensure_local_db_client_pressure_is_safe

healthcheck_url="${BASE_URL%/}/health"
if ! curl -sS --max-time 5 "${healthcheck_url}" >/dev/null; then
  echo "AI smoke regression could not reach ${healthcheck_url}." >&2
  echo "Set BASE_URL to a reachable AI service endpoint before running smoke regression." >&2
  exit 2
fi

mkdir -p "${REPORTS_DIR}"
mkdir -p "${WARMUP_DIR}"

SMOKE_COMMON_ARGS=(
  --base-url "${BASE_URL}"
  --internal-api-key "${INTERNAL_API_KEY}"
)
SMOKE_MEMORY_ARGS=(
  --docker-compose-service ai-chatbot
  --memory-sample-interval-ms "${MEMORY_SAMPLE_INTERVAL_MS}"
)

REGULATIONS_REPORT=""
REGULATIONS_SUMMARY=""
REGULATIONS_EVAL=""
LLM_CANARY_REPORT=""
LLM_CANARY_SUMMARY=""
LLM_CANARY_EVAL=""
LLM_CANARY_RETRY_REPORT=""
LLM_CANARY_RETRY_SUMMARY=""
LLM_CANARY_RETRY_EVAL=""
LLM_CANARY_RETRY_USED=0
REGMIX_REPORT=""
REGMIX_SUMMARY=""
REGMIX_EVAL=""
COACH_REPORT=""
COACH_SUMMARY=""
OVERALL_EXIT_CODE=0
FAILED_PRESETS=()

if [[ "${SMOKE_DISABLE_CACHE:-0}" == "1" ]]; then
  SMOKE_COMMON_ARGS+=(--disable-cache)
fi

print_eval_summary() {
  local label="$1"
  local eval_path="$2"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/format_smoke_eval_summary.py" \
    --eval "${eval_path}" \
    --label "${label}" \
    --prefix "[ai-smoke]"
}

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

evaluate_preset() {
  local label="$1"
  local baseline_preset="$2"
  local summary_path="$3"
  local eval_path="$4"
  local status=0

  set +e
  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_smoke_candidate.py" \
    --baseline-preset "${baseline_preset}" \
    --candidate "${summary_path}" \
    --output "${eval_path}"
  status=$?
  set -e

  print_eval_summary "${label}" "${eval_path}"

  if [[ "${status}" -ne 0 ]]; then
    return "${status}"
  fi

  return 0
}

should_retry_llm_canary_latency_failure() {
  local eval_path="$1"

  "${PYTHON_BIN}" - "${ROOT_DIR}/bega_AI" "${eval_path}" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from scripts import evaluate_smoke_candidate as esc

report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
raise SystemExit(0 if esc.is_latency_only_failure_report(report) else 1)
PY
}

run_regulation_canary() {
  local report_path="${REPORTS_DIR}/smoke_chatbot_quality_regulations_20_${TIMESTAMP}.json"
  local summary_path="${REPORTS_DIR}/smoke_chatbot_quality_regulations_20_${TIMESTAMP}_summary.json"
  local eval_path="${REPORTS_DIR}/smoke_chatbot_quality_regulations_20_${TIMESTAMP}_eval.json"

  ensure_baseline "regulations_20"
  warm_cache_measurement "regulations_20" 20 "${SCRIPT_DIR}/smoke_chatbot_quality_regulations_20.txt"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_chatbot_quality.py" \
    "${SMOKE_COMMON_ARGS[@]}" \
    "${SMOKE_MEMORY_ARGS[@]}" \
    --chat-batch-size 20 \
    --chat-question-list "${SCRIPT_DIR}/smoke_chatbot_quality_regulations_20.txt" \
    --output "${report_path}" \
    --summary-output "${summary_path}"

  if ! evaluate_preset "regulations_20" "regulations_20" "${summary_path}" "${eval_path}"; then
    OVERALL_EXIT_CODE=1
    FAILED_PRESETS+=("regulations_20")
  fi

  REGULATIONS_REPORT="${report_path}"
  REGULATIONS_SUMMARY="${summary_path}"
  REGULATIONS_EVAL="${eval_path}"
  echo "regulations_report=${report_path}"
  echo "regulations_summary=${summary_path}"
  echo "regulations_eval=${eval_path}"
}

run_llm_canary() {
  local report_path="${REPORTS_DIR}/smoke_chatbot_quality_llm_canary_20_${TIMESTAMP}.json"
  local summary_path="${REPORTS_DIR}/smoke_chatbot_quality_llm_canary_20_${TIMESTAMP}_summary.json"
  local eval_path="${REPORTS_DIR}/smoke_chatbot_quality_llm_canary_20_${TIMESTAMP}_eval.json"
  local retry_report_path="${REPORTS_DIR}/smoke_chatbot_quality_llm_canary_20_${TIMESTAMP}_retry1.json"
  local retry_summary_path="${REPORTS_DIR}/smoke_chatbot_quality_llm_canary_20_${TIMESTAMP}_retry1_summary.json"
  local retry_eval_path="${REPORTS_DIR}/smoke_chatbot_quality_llm_canary_20_${TIMESTAMP}_retry1_eval.json"
  local llm_failed=0

  ensure_baseline "llm_canary_20"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_chatbot_quality.py" \
    "${SMOKE_COMMON_ARGS[@]}" \
    "${SMOKE_MEMORY_ARGS[@]}" \
    --disable-cache \
    --chat-batch-size 20 \
    --chat-question-list "${SCRIPT_DIR}/smoke_chatbot_quality_llm_canary_20.txt" \
    --output "${report_path}" \
    --summary-output "${summary_path}"

  if ! evaluate_preset "llm_canary_20" "llm_canary_20" "${summary_path}" "${eval_path}"; then
    llm_failed=1
  fi

  if [[ "${llm_failed}" -eq 1 ]] && should_retry_llm_canary_latency_failure "${eval_path}"; then
    echo "[ai-smoke] llm_canary_20: retrying once after latency-only failure"
    LLM_CANARY_RETRY_USED=1
    "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_chatbot_quality.py" \
      "${SMOKE_COMMON_ARGS[@]}" \
      "${SMOKE_MEMORY_ARGS[@]}" \
      --disable-cache \
      --chat-batch-size 20 \
      --chat-question-list "${SCRIPT_DIR}/smoke_chatbot_quality_llm_canary_20.txt" \
      --output "${retry_report_path}" \
      --summary-output "${retry_summary_path}"

    llm_failed=0
    if ! evaluate_preset "llm_canary_20_retry1" "llm_canary_20" "${retry_summary_path}" "${retry_eval_path}"; then
      llm_failed=1
    else
      echo "[ai-smoke] llm_canary_20: latency-only failure recovered on retry"
    fi

    LLM_CANARY_RETRY_REPORT="${retry_report_path}"
    LLM_CANARY_RETRY_SUMMARY="${retry_summary_path}"
    LLM_CANARY_RETRY_EVAL="${retry_eval_path}"
    report_path="${retry_report_path}"
    summary_path="${retry_summary_path}"
    eval_path="${retry_eval_path}"
  fi

  if [[ "${llm_failed}" -eq 1 ]]; then
    OVERALL_EXIT_CODE=1
    FAILED_PRESETS+=("llm_canary_20")
  fi

  LLM_CANARY_REPORT="${report_path}"
  LLM_CANARY_SUMMARY="${summary_path}"
  LLM_CANARY_EVAL="${eval_path}"
  echo "llm_canary_report=${report_path}"
  echo "llm_canary_summary=${summary_path}"
  echo "llm_canary_eval=${eval_path}"
  if [[ "${LLM_CANARY_RETRY_USED}" -eq 1 ]]; then
    echo "llm_canary_retry_report=${LLM_CANARY_RETRY_REPORT}"
    echo "llm_canary_retry_summary=${LLM_CANARY_RETRY_SUMMARY}"
    echo "llm_canary_retry_eval=${LLM_CANARY_RETRY_EVAL}"
  fi
}

run_regmix_full() {
  local report_path="${REPORTS_DIR}/smoke_chatbot_quality_regmix_100_${TIMESTAMP}.json"
  local summary_path="${REPORTS_DIR}/smoke_chatbot_quality_regmix_100_${TIMESTAMP}_summary.json"
  local eval_path="${REPORTS_DIR}/smoke_chatbot_quality_regmix_100_${TIMESTAMP}_eval.json"

  ensure_baseline "regmix_100"
  warm_cache_measurement "regmix_100" 100

  "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_chatbot_quality.py" \
    "${SMOKE_COMMON_ARGS[@]}" \
    "${SMOKE_MEMORY_ARGS[@]}" \
    --chat-batch-size 100 \
    --output "${report_path}" \
    --summary-output "${summary_path}"

  if ! evaluate_preset "regmix_100" "regmix_100" "${summary_path}" "${eval_path}"; then
    OVERALL_EXIT_CODE=1
    FAILED_PRESETS+=("regmix_100")
  fi

  REGMIX_REPORT="${report_path}"
  REGMIX_SUMMARY="${summary_path}"
  REGMIX_EVAL="${eval_path}"
  echo "regmix_report=${report_path}"
  echo "regmix_summary=${summary_path}"
  echo "regmix_eval=${eval_path}"
}

run_coach_informational_smoke() {
  local report_path="${REPORTS_DIR}/smoke_chatbot_coach_${TIMESTAMP}.json"
  local summary_path="${REPORTS_DIR}/smoke_chatbot_coach_${TIMESTAMP}_summary.json"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_chatbot.py" \
    --base-url "${BASE_URL}" \
    --internal-api-key "${INTERNAL_API_KEY}" \
    --chat-batch-size 0 \
    --no-strict \
    --output "${report_path}" \
    --summary-output "${summary_path}"

  "${PYTHON_BIN}" - "${report_path}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
payload = json.loads(report_path.read_text(encoding="utf-8"))
coach_result = next(
    (item for item in payload.get("results", []) if item.get("endpoint") == "/coach/analyze"),
    {},
)
status = "PASS" if coach_result.get("ok") else "FAIL"
latency = coach_result.get("latency_ms")
line = f"[ai-smoke] coach: {status}"
if isinstance(latency, (int, float)):
    line += f" latency={latency:.2f}ms"
if coach_result.get("error"):
    line += f" error={coach_result['error']}"
print(line)
PY

  COACH_REPORT="${report_path}"
  COACH_SUMMARY="${summary_path}"
  echo "coach_report=${report_path}"
  echo "coach_summary=${summary_path}"
}

write_qa_summary() {
  "${PYTHON_BIN}" - "${QA_SUMMARY_PATH}" "${REGULATIONS_EVAL}" "${LLM_CANARY_EVAL}" "${REGMIX_EVAL}" "${COACH_REPORT}" "${LLM_CANARY_SUMMARY}" "${LLM_CANARY_RETRY_EVAL}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
reg_eval = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
llm_eval = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
regmix_eval = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
coach_report = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
llm_summary = json.loads(Path(sys.argv[6]).read_text(encoding="utf-8"))
retry_eval_path = sys.argv[7]

def preset_row(label: str, payload: dict) -> str:
    memory = payload.get("memory") or {}
    return (
        f"| {label} | {payload.get('status', 'UNKNOWN')} | "
        f"{((payload.get('endpoints') or {}).get('completion') or {}).get('candidate', {}).get('p95')} | "
        f"{((payload.get('endpoints') or {}).get('stream') or {}).get('candidate', {}).get('p95')} | "
        f"{memory.get('candidate_memory_mb')} | "
        f"{', '.join(payload.get('failure_codes') or []) or '-'} |"
    )

def render_latency_cases(title: str, cases: list[dict], metric_label: str) -> list[str]:
    lines = [f"### {title}", ""]
    if not cases:
        lines.append("- none")
        lines.append("")
        return lines

    for case in cases:
        question = case.get("question") or "(missing question)"
        metric_ms = case.get("metric_ms")
        planner_mode = case.get("planner_mode") or "unknown"
        tool_mode = case.get("tool_execution_mode") or "unknown"
        status_code = case.get("status_code")
        line = (
            f"- {question}: {metric_label}={metric_ms}ms, "
            f"planner_mode={planner_mode}, tool_execution_mode={tool_mode}, "
            f"status_code={status_code}"
        )
        if case.get("error"):
            line += f", error={case['error']}"
        lines.append(line)
    lines.append("")
    return lines

latency_diagnostics = (llm_summary.get("summary") or {}).get("latency_diagnostics") or {}
llm_completion_cases = (
    ((latency_diagnostics.get("completion") or {}).get("top_latency_cases") or [])[:3]
)
llm_stream_ttfe_cases = (
    ((latency_diagnostics.get("stream") or {}).get("top_first_token_cases") or [])[:3]
)
llm_stream_first_message_cases = (
    ((latency_diagnostics.get("stream") or {}).get("top_stream_first_message_cases") or [])[:3]
)
llm_warnings = llm_eval.get("warnings") or []

coach_result = next(
    (item for item in coach_report.get("results", []) if item.get("endpoint") == "/coach/analyze"),
    {},
)
coach_status = "PASS" if coach_result.get("ok") else "FAIL"
coach_latency = coach_result.get("latency_ms")
coach_error = coach_result.get("error") or "-"

content = "\n".join(
    [
        f"# AI Smoke QA Summary ({output_path.stem})",
        "",
        "| Preset | Status | Completion p95(ms) | Stream p95(ms) | Peak Memory(MB) | Failures |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        preset_row("regulations_20", reg_eval),
        preset_row("llm_canary_20", llm_eval),
        preset_row("regmix_100", regmix_eval),
        "",
        "## Regression Policy",
        "",
        "- blocking: only `failure_codes` fail the smoke gate.",
        "- advisory: `warnings` are surfaced for operator review but do not change the exit code.",
        "",
        "## LLM Canary Slow Questions",
        "",
        *render_latency_cases(
            "Completion Outliers",
            llm_completion_cases,
            "latency_ms",
        ),
        *render_latency_cases(
            "Stream First Token Outliers",
            llm_stream_ttfe_cases,
            "first_token_ms",
        ),
        *render_latency_cases(
            "Stream First Message Outliers",
            llm_stream_first_message_cases,
            "stream_first_message_ms",
        ),
        "## LLM Canary Warnings",
        "",
        *(["- none"] if not llm_warnings else [f"- {warning}" for warning in llm_warnings]),
        "",
        "## Coach Smoke",
        "",
        f"- status: {coach_status}",
        f"- latency_ms: {coach_latency}",
        f"- error: {coach_error}",
        "",
        "## Artifacts",
        "",
        f"- regulations_eval: {sys.argv[2]}",
        f"- llm_canary_eval: {sys.argv[3]}",
        f"- llm_canary_summary: {sys.argv[6]}",
        f"- llm_canary_retry_eval: {retry_eval_path or '-'}",
        f"- regmix_eval: {sys.argv[4]}",
        f"- coach_report: {sys.argv[5]}",
        "",
    ]
)
output_path.write_text(content + "\n", encoding="utf-8")
print(f"qa_summary={output_path}")
PY
}

run_regulation_canary
run_llm_canary
run_regmix_full
run_coach_informational_smoke
write_qa_summary

if [[ "${#FAILED_PRESETS[@]}" -gt 0 ]]; then
  echo "[ai-smoke] blocking preset failures: ${FAILED_PRESETS[*]}" >&2
fi

exit "${OVERALL_EXIT_CODE}"
