#!/bin/bash
# manual_detail 전체 배치: chunk 2-8 순차 실행
# chunk 1은 이미 별도로 실행됨
# 사용법: bash scripts/run_manual_detail_remaining.sh

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export POSTGRES_DB_URL="postgresql://postgres:rkdansdlf@134.185.107.178:5432/bega_backend"

API_KEY="local-dev-ai-internal-token"
DELAY=3.0
TIMEOUT=180
MAX_ATTEMPTS=3
RECOVERY=2

for i in 2 3 4 5 6 7 8; do
    offset=$(( (i-1)*100 ))
    limit=100
    if [ $i -eq 8 ]; then limit=21; fi

    echo ""
    echo "========================================"
    echo "=== manual_detail Chunk $i: offset=$offset limit=$limit ==="
    echo "=== Started at $(date) ==="
    echo "========================================"

    python scripts/batch_coach_matchup_cache.py \
        --years 2025 --status-bucket COMPLETED --order desc \
        --offset $offset --limit $limit \
        --force-rebuild --delay-seconds $DELAY --timeout $TIMEOUT \
        --max-attempts $MAX_ATTEMPTS --recovery-pass-count $RECOVERY \
        --internal-api-key "$API_KEY" \
        --quality-report "reports/manual_detail_chunk${i}.json" \
        2>&1

    # 품질 체크
    python -c "
import json, sys
r = json.load(open('reports/manual_detail_chunk${i}.json'))
s = r['summary']
print(f'Chunk $i: coverage={s.get(\"coverage_rate\",0):.1%} llm_manual={s.get(\"llm_manual_rate\",0):.1%} fallback={s.get(\"fallback_rate\",0):.1%} failed={s.get(\"failed\",0)}')
if s.get('failed', 0) > 5:
    print('WARNING: >5 failures, check logs')
"
    echo "=== Chunk $i finished at $(date) ==="
done

echo ""
echo "=== All chunks complete ==="
echo "Run final verification:"
echo "python scripts/batch_coach_matchup_cache.py --years 2025 --status-bucket COMPLETED --verify-cache-only --quality-report reports/manual_detail_verify_final.json"
