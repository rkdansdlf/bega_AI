#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SAMPLES_PATH="${SEMANTIC_CACHE_SHADOW_SAMPLES:-}"
REPORT_DIR="${SEMANTIC_CACHE_SHADOW_REPORT_DIR:-$ROOT_DIR/outputs/semantic-cache}"
BASE_URL="${SEMANTIC_CACHE_SHADOW_BASE_URL:-}"
MIN_TOKEN_JACCARD="${SEMANTIC_CACHE_SHADOW_MIN_TOKEN_JACCARD:-0.72}"
INTERNAL_API_KEY_ENV="${SEMANTIC_CACHE_SHADOW_INTERNAL_KEY_ENV:-AI_INTERNAL_TOKEN}"

if [ -z "$SAMPLES_PATH" ]; then
  echo "SEMANTIC_CACHE_SHADOW_SAMPLES is required" >&2
  exit 2
fi

if [ ! -f "$SAMPLES_PATH" ]; then
  echo "Sample file not found: $SAMPLES_PATH" >&2
  exit 2
fi

mkdir -p "$REPORT_DIR"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_path="$REPORT_DIR/semantic-cache-shadow-$timestamp.json"

args=(
  "$ROOT_DIR/scripts/semantic_cache_shadow_eval.py"
  "--samples" "$SAMPLES_PATH"
  "--output" "$output_path"
  "--min-token-jaccard" "$MIN_TOKEN_JACCARD"
  "--internal-api-key-env" "$INTERNAL_API_KEY_ENV"
)

if [ -n "$BASE_URL" ]; then
  args+=("--base-url" "$BASE_URL")
fi

"$PYTHON_BIN" "${args[@]}"
echo "semantic cache shadow report: $output_path"
