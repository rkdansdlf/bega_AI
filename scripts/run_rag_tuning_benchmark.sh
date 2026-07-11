#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
REPORT_DIR="${RAG_TUNING_REPORT_DIR:-$ROOT_DIR/outputs/rag-tuning}"
RETRIEVAL_LIMIT="${RAG_TUNING_RETRIEVAL_LIMIT:-5}"
PIPELINE_RUNS="${RAG_TUNING_PIPELINE_RUNS:-10}"
PIPELINE_WITH_DB="${RAG_TUNING_WITH_DB:-false}"

mkdir -p "$REPORT_DIR"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
retrieval_report="$REPORT_DIR/retrieval-$timestamp.json"
pipeline_report="$REPORT_DIR/pipeline-$timestamp.json"

"$PYTHON_BIN" "$ROOT_DIR/scripts/benchmark_retrieval.py" \
  --limit "$RETRIEVAL_LIMIT" \
  --variant both \
  --output "$retrieval_report"

pipeline_args=(
  "$ROOT_DIR/scripts/benchmark_rag_pipeline.py"
  "--runs" "$PIPELINE_RUNS"
  "--output" "$pipeline_report"
)

if [ "$PIPELINE_WITH_DB" = "true" ]; then
  pipeline_args+=("--db")
fi

"$PYTHON_BIN" "${pipeline_args[@]}"

echo "rag retrieval report: $retrieval_report"
echo "rag pipeline report: $pipeline_report"
