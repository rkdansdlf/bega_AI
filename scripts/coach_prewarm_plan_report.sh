#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
YEARS="${COACH_PREWARM_YEARS:-2025}"
LEAGUE_TYPE="${COACH_PREWARM_LEAGUE_TYPE:-REGULAR}"
STATUS_BUCKET="${COACH_PREWARM_STATUS_BUCKET:-ANY}"
CACHE_STATE_FILTER="${COACH_PREWARM_CACHE_STATE_FILTER:-ANY}"
ORDER="${COACH_PREWARM_ORDER:-asc}"
LIMIT="${COACH_PREWARM_LIMIT:-}"
OFFSET="${COACH_PREWARM_OFFSET:-0}"
REPORT_DIR="${COACH_PREWARM_REPORT_DIR:-${PROJECT_ROOT}/outputs/coach-prewarm}"

timestamp="$(date +%Y%m%d-%H%M%S)"
report_path="${REPORT_DIR}/coach-prewarm-plan-${timestamp}.json"

mkdir -p "${REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/scripts/batch_coach_matchup_cache.py"
  --dry-run
  --years "${YEARS}"
  --league-type "${LEAGUE_TYPE}"
  --status-bucket "${STATUS_BUCKET}"
  --cache-state-filter "${CACHE_STATE_FILTER}"
  --order "${ORDER}"
  --offset "${OFFSET}"
  --quality-report "${report_path}"
)

if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

echo "Writing coach prewarm dry-run report: ${report_path}"
exec "${cmd[@]}"
