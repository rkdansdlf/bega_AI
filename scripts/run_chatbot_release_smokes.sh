#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/bega_AI/.venv/bin/python}"
STREAM_STYLE="${STREAM_STYLE:-compact}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/reports/release_smokes_$TIMESTAMP}"

mkdir -p "$REPORT_DIR"

EXPLAINER_FILE="$(mktemp /tmp/chatbot_explainer_canary.XXXXXX.txt)"
LATEST_FILE="$(mktemp /tmp/chatbot_latest_canary.XXXXXX.txt)"

cleanup() {
  rm -f "$EXPLAINER_FILE" "$LATEST_FILE"
}
trap cleanup EXIT

cat >"$EXPLAINER_FILE" <<'EOF'
히트앤런이 뭐야
필승조가 뭐야
체인지업은 어떤 공이야
WHIP 뜻 설명해줘
QS가 뭐야
포수 프레이밍이 뭐야
수비 시프트가 뭔지 알려줘
병살타가 왜 치명적이야
스트라이크존은 어떻게 정해져
야구에서 아웃카운트는 왜 3개야
비디오 판독 대상 플레이 알려줘
승리투수 요건은 어떻게 정해져
EOF

cat >"$LATEST_FILE" <<'EOF'
2025년 KIA와 삼성 최근 맞대결 알려줘
김도영 최근 경기 활약 알려줘
지금 5위 싸움 어떻게 돼
오늘 우천취소 가능성 있는 경기 있어
KIA 최근 분위기 어때
삼성 최근 경기 결과 알려줘
EOF

echo "[1/4] explainer canary"
"$PYTHON_BIN" "$ROOT_DIR/bega_AI/scripts/smoke_chatbot_quality.py" \
  --chat-batch-size 12 \
  --chat-question-list "$EXPLAINER_FILE" \
  --disable-cache \
  --stream-style "$STREAM_STYLE" \
  --no-strict | tee "$REPORT_DIR/explainer_canary.log"

echo "[2/4] latest canary"
"$PYTHON_BIN" "$ROOT_DIR/bega_AI/scripts/smoke_chatbot_quality.py" \
  --chat-batch-size 6 \
  --chat-question-list "$LATEST_FILE" \
  --disable-cache \
  --stream-style "$STREAM_STYLE" \
  --no-strict | tee "$REPORT_DIR/latest_canary.log"

echo "[3/4] factual canary v2"
"$PYTHON_BIN" "$ROOT_DIR/bega_AI/scripts/smoke_chatbot_factual_canary.py" \
  --disable-cache \
  --stream-style "$STREAM_STYLE" \
  --output "$REPORT_DIR/factual_canary.json" \
  --summary-output "$REPORT_DIR/factual_canary_summary.json" \
  --no-strict | tee "$REPORT_DIR/factual_canary.log"

echo "[4/4] baseball 180"
"$PYTHON_BIN" "$ROOT_DIR/bega_AI/scripts/smoke_chatbot_quality.py" \
  --chat-batch-size 180 \
  --chat-question-list "$ROOT_DIR/bega_AI/scripts/smoke_chatbot_quality_baseball_180.txt" \
  --disable-cache \
  --stream-style "$STREAM_STYLE" \
  --no-strict | tee "$REPORT_DIR/baseball_180.log"

echo "release smoke reports: $REPORT_DIR"
