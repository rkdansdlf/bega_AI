#!/usr/bin/env bash
# =============================================================================
# daily_ingest_kbo.sh
# KBO 신규 경기 데이터 일별 자동 인제스트 (local 임베딩)
#
# 용도: 매일 새로 완료된 KBO 경기 데이터를 bega_backend에서 rag_chunks로 반영.
#       local(pseudo-vector) 임베딩으로 즉시 저장하고,
#       월말 monthly_embed_upgrade.sh 에서 openrouter로 업그레이드.
#
# crontab 예시 (매일 오전 9시 KST = 0시 UTC):
#   0 0 * * * /Users/mac/project/KBO_platform/bega_AI/scripts/daily_ingest_kbo.sh >> /tmp/daily_ingest.log 2>&1
# =============================================================================
set -euo pipefail

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEGA_AI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Python 탐색 순서: bega_AI .venv → backup .venv → system python3
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
# .env 파일이 있으면 로드 (EMBED_PROVIDER는 아래에서 덮어씀)
[[ -f "$BEGA_AI_DIR/.env" ]] && set -a && source "$BEGA_AI_DIR/.env" && set +a

export BEGA_SKIP_APP_INIT=1
export EMBED_PROVIDER=local          # 일별 ingest는 local 임베딩 (비용 없음)
export PYTHONPATH="$BEGA_AI_DIR"

# ── since 계산 (KST 기준 전날 00:00 UTC) ─────────────────────────────────────
# macOS: date -u -v-1d +%Y-%m-%dT00:00:00
# Linux: date -u -d "yesterday" +%Y-%m-%dT00:00:00
if date -u -v-1d +%Y-%m-%dT00:00:00 >/dev/null 2>&1; then
    SINCE="$(date -u -v-1d +%Y-%m-%dT00:00:00)"   # macOS
else
    SINCE="$(date -u -d 'yesterday' +%Y-%m-%dT00:00:00)"  # Linux
fi

# 현재 시즌 연도 (5월 이후는 당해연도)
SEASON_YEAR="$(date +%Y)"

LOG_STAMP="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$LOG_STAMP] daily_ingest_kbo.sh 시작 -- season=$SEASON_YEAR since=$SINCE"

# ── 인제스트 실행 ──────────────────────────────────────────────────────────────
"$PYTHON" "$BEGA_AI_DIR/scripts/ingest_from_kbo.py" \
    --source-db-url "${OCI_DB_URL:-postgresql://postgres:rkdansdlf@134.185.107.178:5432/bega_backend}" \
    --season-year "$SEASON_YEAR" \
    --since "$SINCE" \
    --tables \
        game \
        game_metadata \
        game_batting_stats \
        game_pitching_stats \
        game_inning_scores \
        game_lineups \
        game_summary \
        game_flow_summary \
        player_season_batting \
        player_season_pitching \
        stat_rankings

LOG_STAMP="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$LOG_STAMP] daily_ingest_kbo.sh 완료"
