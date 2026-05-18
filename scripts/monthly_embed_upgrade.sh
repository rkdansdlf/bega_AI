#!/usr/bin/env bash
# =============================================================================
# monthly_embed_upgrade.sh
# 전월 local 임베딩 청크 → openrouter(openai/text-embedding-3-small) 업그레이드
#
# 용도: 매월 1일 새벽, 전월까지 쌓인 local 임베딩 청크를 production 품질
#       openrouter 벡터로 교체. game_date가 없는 청크(시즌통계 등)도 처리.
#
# crontab 예시 (매월 1일 오전 3시 KST = 전일 18시 UTC):
#   0 18 28-31 * * [ "$(date -d tomorrow +\%d)" = "01" ] && \
#       /Users/mac/project/KBO_platform/bega_AI/scripts/monthly_embed_upgrade.sh >> /tmp/monthly_upgrade.log 2>&1
#
# 또는 간단하게 (매월 1일 오전 3시 KST):
#   0 18 1 * * /Users/mac/project/KBO_platform/bega_AI/scripts/monthly_embed_upgrade.sh >> /tmp/monthly_upgrade.log 2>&1
# =============================================================================
set -euo pipefail

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEGA_AI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

find_python() {
    local candidates=(
        "$BEGA_AI_DIR/.venv/bin/python"
        "/Users/mac/project/backup/KBO_platform/AI/.venv/bin/python3"
        "$(command -v python3 2>/dev/null || true)"
    )
    for p in "${candidates[@]}"; do
        [[ -x "$p" ]] && echo "$p" && return
    done
    echo "ERROR: Python 인터프리터를 찾을 수 없습니다." >&2
    exit 1
}
PYTHON="$(find_python)"

# ── 환경변수 ──────────────────────────────────────────────────────────────────
[[ -f "$BEGA_AI_DIR/.env" ]] && set -a && source "$BEGA_AI_DIR/.env" && set +a

export BEGA_SKIP_APP_INIT=1
export EMBED_PROVIDER="${EMBED_PROVIDER:-openrouter}"   # 기본값: openrouter
export PYTHONPATH="$BEGA_AI_DIR"

# ── 전월 마지막 날 계산 ───────────────────────────────────────────────────────
# 오늘이 1일이라는 가정 하에 실행 (cron: 매월 1일)
# macOS: date -v1d -v-1d +%Y-%m-%d
# Linux: date -d "$(date +%Y-%m-01) -1 day" +%Y-%m-%d
if date -v1d -v-1d +%Y-%m-%d >/dev/null 2>&1; then
    PREV_MONTH_END="$(date -v1d -v-1d +%Y-%m-%d)"    # macOS
else
    PREV_MONTH_END="$(date -d "$(date +%Y-%m-01) -1 day" +%Y-%m-%d)"  # Linux
fi

PREV_YEAR="${PREV_MONTH_END%%-*}"

LOG_STAMP="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$LOG_STAMP] monthly_embed_upgrade.sh 시작 -- prev_month_end=$PREV_MONTH_END season=$PREV_YEAR"

# ── local → openrouter 업그레이드 ─────────────────────────────────────────────
# --model-filter local : embedding_model LIKE 'local:%' 인 청크만 대상
# --max-game-date      : game_date가 전월 말 이하인 게임 청크만 처리
#                        game_date 없는 청크(시즌통계 등)는 항상 포함됨
"$PYTHON" "$BEGA_AI_DIR/scripts/reembed_null_model_chunks.py" \
    --model-filter local \
    --season-year "$PREV_YEAR" \
    --max-game-date "$PREV_MONTH_END" \
    --batch-size 256 \
    --commit-interval 2000 \
    --max-concurrency 8

LOG_STAMP="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$LOG_STAMP] monthly_embed_upgrade.sh 완료"
