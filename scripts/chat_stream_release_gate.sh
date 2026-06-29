#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${AI_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${AI_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

cd "${AI_DIR}"

echo "[chat-stream-gate] python=${PYTHON_BIN}"

"${PYTHON_BIN}" -m pytest tests/test_chat_queue.py
"${PYTHON_BIN}" -m pytest tests/test_chat_stream_live.py
"${PYTHON_BIN}" -m pytest tests/test_llm_model_candidates.py
"${PYTHON_BIN}" -m pytest tests/test_chat_api_smoke.py

echo "[chat-stream-gate] passed"
