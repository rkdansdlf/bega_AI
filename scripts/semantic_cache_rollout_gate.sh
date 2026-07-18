#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
REPORT_DIR="${SEMANTIC_CACHE_SHADOW_REPORT_DIR:-$ROOT_DIR/outputs/semantic-cache}"
WINDOW_DAYS="${SEMANTIC_CACHE_SHADOW_WINDOW_DAYS:-30}"
EXPORT_LIMIT="${SEMANTIC_CACHE_SHADOW_EXPORT_LIMIT:-10000}"
MIN_SAMPLES="${SEMANTIC_CACHE_RELEASE_MIN_SAMPLES:-100}"
MIN_OBSERVATION_DAYS="${SEMANTIC_CACHE_RELEASE_MIN_OBSERVATION_DAYS:-7}"
MAX_FALSE_POSITIVE_RATE="${SEMANTIC_CACHE_RELEASE_MAX_FALSE_POSITIVE_RATE:-0.005}"
MIN_TOKEN_JACCARD="${SEMANTIC_CACHE_SHADOW_MIN_TOKEN_JACCARD:-0.72}"
CURRENT_ROLLOUT_PERCENT="${SEMANTIC_CACHE_CURRENT_ROLLOUT_PERCENT:-${CHAT_SEMANTIC_CACHE_ROLLOUT_PERCENT:-0}}"

mkdir -p "$REPORT_DIR"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
samples_path="$REPORT_DIR/semantic-cache-shadow-samples-$timestamp.json"
report_path="$REPORT_DIR/semantic-cache-rollout-gate-$timestamp.json"

"$PYTHON_BIN" "$ROOT_DIR/scripts/export_semantic_cache_shadow_samples.py" \
  --output "$samples_path" \
  --days "$WINDOW_DAYS" \
  --limit "$EXPORT_LIMIT"

set +e
"$PYTHON_BIN" "$ROOT_DIR/scripts/semantic_cache_shadow_eval.py" \
  --samples "$samples_path" \
  --output "$report_path" \
  "--release-gate" \
  --release-min-samples "$MIN_SAMPLES" \
  --release-min-observation-days "$MIN_OBSERVATION_DAYS" \
  --release-max-false-positive-rate "$MAX_FALSE_POSITIVE_RATE" \
  --min-token-jaccard "$MIN_TOKEN_JACCARD" \
  "--current-rollout-percent" "$CURRENT_ROLLOUT_PERCENT"
status=$?
set -e

echo "semantic cache shadow samples: $samples_path"
echo "semantic cache rollout report: $report_path"
exit "$status"
